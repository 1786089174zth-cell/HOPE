"""Entry audit of a ``train.py`` checkpoint: what the robot does between taking over and the first ball.

    python tools/audit_entry.py logs/supermimic/<run>/model_<N>.pt device=cuda:0 [num_envs=256 seconds=20]
        [stand_x=0.03,0.10 stand_yaw=0.3 stance_blend=0 clean=0 friction=0.3,1.2 out=<json>]
        [contact=elliptic|pyramidal handoff=1|0 handoff_s=3.0]

Every environment starts in the default stand (the 0.265 m deploy entry pose) and waits in HOLD
for ``seconds`` with no ball, as Gate3 does for about 20 s after the Runner enters MOTION. By
default the Runner's 3 s entry handoff runs and the foot contact is the non-creeping one
(``handoff=0 contact=pyramidal`` gives the simulator every checkpoint up to 39000 was trained in).
``stand_x=0,0 stand_yaw=0`` is Gate3's own entry: on the station, facing the table. Reported over
time: feet width (HOLD reference 0.47 m), torso and pelvis roll, waist roll, the distance of the
pelvis from the station, and how many environments are still narrower than 0.40 m.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402,F401 -- sets up sys.path for the pinned dependencies
from evaluate import apply_physics, from_dict  # noqa: E402


def main(argv=None):
    import torch
    from supermimic_hope import a3
    from supermimic_hope.core.rotations import quat_to_matrix
    from supermimic_hope.core.trainer import TrainCfg, Trainer

    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0])
    opts = dict(a.split("=", 1) for a in argv[1:])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(TrainCfg, payload["config"])
    cfg.device = opts.get("device", cfg.device)
    cfg.num_envs = int(opts.get("num_envs", 256))
    cfg.seed = int(opts.get("seed", 12345))
    cfg.buffer_steps, cfg.capture = 4, False
    seconds = float(opts.get("seconds", 20.0))
    steps = int(round(seconds / 0.02))
    t = cfg.task
    t.stand_probability, t.long_hold_probability = 1.0, 1.0
    t.long_hold_steps = (steps + 50, steps + 50)
    t.episode_steps = steps + 100
    t.stance_blend_probability = float(opts.get("stance_blend", 0.0))
    t.rush_probability = t.high_probability = t.edge_probability = 0.0
    if "stand_x" in opts:
        t.stand_x = tuple(float(v) for v in opts["stand_x"].split(","))
    if "stand_yaw" in opts:
        t.stand_yaw = float(opts["stand_yaw"])
    # This tool is about the deploy entry: non-creeping contact and the Runner's handoff unless told otherwise.
    contact, handoff = apply_physics(cfg, opts, contact="elliptic", handoff=1.0)
    if "handoff_s" in opts:
        t.entry_handoff_s = float(opts["handoff_s"])
    if "friction" in opts:
        cfg.sim.friction = tuple(float(v) for v in opts["friction"].split(","))
    if opts.get("clean", "0") != "0":
        t.noise_gyro = t.noise_joint_pos = t.noise_joint_vel = t.target_noise = t.target_bias = 0.0
        t.mocap_bias = t.mocap_noise = t.mocap_rpy_bias = t.mocap_rpy_noise = (0.0, 0.0, 0.0)
        t.joint_offset = 0.0
    trainer = Trainer(cfg)
    trainer.load(path)
    if opts.get("ema", "1") != "0" and "actor_ema" in payload:
        trainer.actor.load_state_dict(payload["actor_ema"])
    trainer.actor.eval()
    task, sim, dev, N = trainer.task, trainer.sim, trainer.task.device, cfg.num_envs
    # Fresh stand episodes everywhere, as Trainer.collect starts one.
    ids = torch.arange(N, device=dev)
    task.reset(ids)
    trainer.a_prev[:] = 0.0
    trainer.a_prev2[:] = 0.0
    trainer.q_prev[:] = task.q0
    trainer.history[:] = task.frame(sim.state(), trainer.a_prev, task.tape(), noise=True)[:, None]

    feet = torch.tensor(task.kin.index(list(a3.FEET)), device=dev)
    torso = task.kin.index(["torso_Link"])[0]
    waist_roll = a3.JOINTS.index("waist_roll_joint")
    legs = [a3.JOINTS.index(f"{side}_{j}_joint") for side in ("left", "right") for j in ("hip_roll", "ankle_roll")]
    roll = lambda R: torch.rad2deg(torch.atan2(R[:, 2, 1], R[:, 2, 2]))  # noqa: E731
    reference = task.reference_bodies(task.hold_ref[None])[0][0, feet]
    reference_width = float((reference[0] - reference[1]).norm())
    marks = sorted({int(round(s / 0.02)) for s in (0.0, 0.5, 1, 2, 3, 5, 10, 15, seconds) if s <= seconds})
    rows, lifted = [], torch.zeros(N, device=dev)
    for k in range(steps + 1):
        s = sim.state()
        bodies = task.kin(s)
        pos = bodies[0][:, feet]
        # a foot counts as lifted when its ankle link is 1.5 cm above where it stands in the reference
        lifted += ((pos[..., 2] - reference[None, :, 2]).amax(-1) > 0.015).float()
        if k in marks:
            R = quat_to_matrix(s[:, 3:7])
            yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
            # width across the robot's own lateral axis, so a yawed robot is measured the same way
            d = pos[:, 0, :2] - pos[:, 1, :2]
            width = -d[:, 0] * torch.sin(yaw) + d[:, 1] * torch.cos(yaw)
            torso_roll = roll(bodies[1][:, torso])
            q = s[:, 13:13 + a3.NJ]
            rows.append({
                "t_s": k * 0.02,
                "feet_width_m": {"mean": float(width.mean()), "p10": float(width.quantile(0.1)), "p90": float(width.quantile(0.9))},
                "narrower_than_0.40_share": float((width < 0.40).float().mean()),
                "torso_roll_deg": {"mean": float(torso_roll.mean()), "abs_mean": float(torso_roll.abs().mean()),
                                   "abs_p90": float(torso_roll.abs().quantile(0.9))},
                "pelvis_roll_deg_abs_mean": float(roll(R).abs().mean()),
                "waist_roll_deg": {"mean": float(torch.rad2deg(q[:, waist_roll]).mean()),
                                   "abs_mean": float(torch.rad2deg(q[:, waist_roll]).abs().mean())},
                "hip_ankle_roll_deg_mean": [float(v) for v in torch.rad2deg(q[:, legs]).mean(0)],
                "pelvis_xy_m_mean": [float(v) for v in s[:, :2].mean(0)],
                "pelvis_xy_from_station_m": float(s[:, :2].norm(dim=-1).mean()),
                "yaw_deg_abs_mean": float(torch.rad2deg(yaw).abs().mean()),
                "foot_lifted_s_mean": float(lifted.mean() * 0.02),
            })
            # split the last mark by floor friction (training randomizes 0.3-1.2)
            if k == marks[-1]:
                mu = sim.friction
                edges = torch.quantile(mu, torch.tensor([0.0, 1 / 3, 2 / 3, 1.0], device=dev))
                rows[-1]["by_friction"] = [
                    {"mu": [float(edges[i]), float(edges[i + 1])],
                     "feet_width_m": float(width[(mu >= edges[i]) & (mu <= edges[i + 1])].mean()),
                     "torso_roll_abs_deg": float(torso_roll[(mu >= edges[i]) & (mu <= edges[i + 1])].abs().mean())}
                    for i in range(3)]
        if k < steps:
            trainer.collect(1, noise=0.0, learn=False)
    result = {"checkpoint": str(path), "num_envs": N, "seconds": seconds, "stand_x": list(t.stand_x),
              "stand_yaw": t.stand_yaw, "stance_blend": t.stance_blend_probability,
              "friction": list(cfg.sim.friction), "clean": opts.get("clean", "0") != "0",
              "contact": contact, "handoff_probability": handoff, "handoff_s": t.entry_handoff_s,
              "reference_feet_width_m": reference_width, "timeline": rows}
    print(f"[entry] {path.name}: default stand, stand_x {t.stand_x} yaw +-{t.stand_yaw}, contact {contact}, "
          f"handoff {t.entry_handoff_s if handoff > 0 else 0:g} s, {N} envs; "
          f"HOLD reference feet width {reference_width:.3f} m")
    print("[entry]    t   feet width mean (p10-p90)  <0.40   torso roll mean/|mean|/|p90|  waist roll  pelvis |roll|  off-station  lifted")
    for r in rows:
        w, tr = r["feet_width_m"], r["torso_roll_deg"]
        print(f"[entry] {r['t_s']:5.1f}   {w['mean']:.3f} ({w['p10']:.3f}-{w['p90']:.3f})       {r['narrower_than_0.40_share']:.2f}   "
              f"{tr['mean']:+5.1f} / {tr['abs_mean']:4.1f} / {tr['abs_p90']:4.1f} deg      {r['waist_roll_deg']['mean']:+5.1f}       "
              f"{r['pelvis_roll_deg_abs_mean']:4.1f}        {r['pelvis_xy_from_station_m']:.3f} m     {r['foot_lifted_s_mean']:.2f} s")
    for b in rows[-1]["by_friction"]:
        print(f"[entry] end, friction {b['mu'][0]:.2f}-{b['mu'][1]:.2f}: feet width {b['feet_width_m']:.3f} m, "
              f"torso |roll| {b['torso_roll_abs_deg']:.1f} deg")
    out = Path(opts.get("out", path.with_name(f"entry_audit_{path.stem.split('_')[-1]}.json")))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[entry] wrote {out}")


if __name__ == "__main__":
    main()
