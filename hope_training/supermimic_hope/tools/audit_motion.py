"""Motion-quality audit of a ``train.py`` checkpoint: jitter, posture and footwork against the teacher.

    python tools/audit_motion.py logs/supermimic/<run>/model_<N>.pt device=cuda:0 [num_envs=256 steps=3000 clean=0]

Deterministic averaged actor, sensor noise as deployed (``clean=1`` removes it, to tell policy
chatter from noise amplification). Everything is measured in the training simulator against the
reference the policy is rewarded to track; it is not a Gate3 or hardware measurement.

* hold: environments that have been in HOLD for at least one second. q_des step RMS and
  direction reversals per joint (Gate3's idle limits are 5 mrad and 8 Hz), joint speed, and
  the posture error to the HOLD reference (pelvis height, knees, feet separation).
* strokes: from the announcement to 0.3 s after the strike, per family: how far the stepping
  foot travels and lifts (robot vs reference), the lowest pelvis height (robot vs reference),
  and the peak joint speed (robot vs reference).
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
    from supermimic_hope.core.task import FAMILIES
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
    cfg.task.midstroke_probability = 0.0
    short = float(opts.get("short_notice", 0.0))
    cfg.task.short_notice_probability = (0.0, short, short, short)
    low = float(opts.get("low_ball", 0.0))  # share of questions from the low-ball rows (Gate3 is all low)
    cfg.task.low_probability = low
    cfg.task.low_on_plane = 0.0
    cfg.task.plane_shift = 0.0
    jitter = float(opts.get("jitter", 0.0))  # 1 = the training target jitter (wider ball paths)
    cfg.task.target_jitter_pos, cfg.task.target_jitter_vel = cfg.task.target_jitter_pos * jitter, cfg.task.target_jitter_vel * jitter
    cfg.task.stance_blend_probability = float(opts.get("stance_blend", 0.0))
    rush = float(opts.get("rush", 0.0))  # share of questions announced 0.24-0.46 s ahead (fast balls)
    cfg.task.rush_probability = rush
    high = float(opts.get("high_ball", 0.0))  # share of questions from the rows above 1.12 m
    cfg.task.high_probability = high
    edge = float(opts.get("edge", 0.0))  # share of questions pushed up to 6 cm across a side boundary
    cfg.task.edge_probability = edge
    contact, handoff = apply_physics(cfg, opts)  # contact=pyramidal|elliptic, handoff=0|1 (see evaluate.py)
    clean = opts.get("clean", "0") != "0"
    if clean:
        t = cfg.task
        t.noise_gyro = t.noise_joint_pos = t.noise_joint_vel = t.target_noise = t.target_bias = 0.0
        t.mocap_bias = t.mocap_noise = t.mocap_rpy_bias = t.mocap_rpy_noise = (0.0, 0.0, 0.0)
    steps = int(opts.get("steps", 3000))
    trainer = Trainer(cfg)
    trainer.load(path)
    if opts.get("ema", "1") != "0" and "actor_ema" in payload:
        trainer.actor.load_state_dict(payload["actor_ema"])
    trainer.actor.eval()
    task, sim, dev, N, dt = trainer.task, trainer.sim, trainer.task.device, cfg.num_envs, trainer.task.dt
    feet = torch.tensor(task.kin.index(["left_ankle_roll_Link", "right_ankle_roll_Link"]), device=dev)
    knees = [a3.JOINTS.index("left_knee_joint"), a3.JOINTS.index("right_knee_joint")]
    z = lambda *shape: torch.zeros(*shape, device=dev)  # noqa: E731

    hold_run = torch.zeros(N, dtype=torch.long, device=dev)
    prev_q, prev_d = trainer.q_prev.clone(), z(N, 31)
    H = dict(n=z(1), step2=z(31), rev=z(31), qd2=z(31), dz=z(1), dz2=z(1), knee=z(1), feet=z(1), err2=z(31), act2=z(31))
    S = dict(n=z(4), travel=z(4), ref_travel=z(4), lift=z(4), ref_lift=z(4), low=z(4), ref_low=z(4),
             speed=z(4), ref_speed=z(4), root_xy=z(4), ref_root_xy=z(4))
    active = torch.zeros(N, dtype=torch.bool, device=dev)
    start, ref_start = z(N, 2, 3), z(N, 2, 3)
    root0, ref_root0 = z(N, 2), z(N, 2)
    run = {k: z(N, *shape) for k, shape in dict(travel=(2,), ref_travel=(2,), lift=(2,), ref_lift=(2,), speed=(),
                                                ref_speed=(), root_xy=(), ref_root_xy=()).items()}
    run["low"], run["ref_low"] = torch.full((N,), 9.0, device=dev), torch.full((N,), 9.0, device=dev)
    # Entry: the first three seconds of an episode that starts in the default stand.
    waist = [a3.JOINTS.index(n) for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]
    E = dict(n=z(1), tilt=z(1), tilt_peak=z(1), eps=z(1), waist=z(3), waist_abs=z(3), roll=z(1), pitch=z(1))
    ep_tilt = z(N)
    HT = dict(roll=z(1), pitch=z(1), waist=z(3))
    # Table keep-out (task.table_depth): blade rim, handle end and hand below the table top beyond its edge.
    table_x, table_z = cfg.task.table_edge_x, cfg.task.table_top_z
    T = dict(steps=z(1), deep=z(1), strokes=z(4), hit=z(4), depth=z(4))
    stroke_depth = z(N)
    prev_tts = task.tape()["tts"].clone()
    trainer.collect(200, noise=0.0, learn=False)
    for _ in range(steps):
        trainer.collect(1, noise=0.0, learn=False)
        s, tape, q_cmd = sim.state(), task.tape(), trainer.q_prev
        ref, tts, in_hold = tape["ref"], tape["tts"], tape["in_hold"] > 0.5
        fresh = task.ep_step <= 1
        kin = task.kin(s)
        pos = kin[0][:, feet]
        # pelvis tilt against the reference (deg) and waist joints
        from supermimic_hope.core.rotations import quat_to_matrix
        Rr, Rs = quat_to_matrix(ref[:, 3:7]), quat_to_matrix(s[:, 3:7])
        tilt = torch.rad2deg(torch.acos((Rr[:, 2, :] * Rs[:, 2, :]).sum(-1).clamp(-1, 1)))
        roll = torch.rad2deg(torch.atan2(Rs[:, 2, 1], Rs[:, 2, 2]))
        pitch = torch.rad2deg(-torch.asin(Rs[:, 2, 0].clamp(-1, 1)))
        wq = s[:, 13:13 + 31][:, waist]
        entry = (task.hold > 0) & (task.ep_step > 1) & (task.ep_step <= 150)
        ef = entry.float()
        E["n"] += ef.sum(); E["tilt"] += (ef * tilt).sum(); E["roll"] += (ef * roll.abs()).sum(); E["pitch"] += (ef * pitch).sum()
        E["waist"] += (ef[:, None] * wq).sum(0); E["waist_abs"] += (ef[:, None] * wq.abs()).sum(0)
        ep_tilt = torch.where(entry, torch.maximum(ep_tilt, tilt), ep_tilt)
        closing = (task.ep_step == 151) & (ep_tilt > 0)
        E["tilt_peak"] += (closing.float() * ep_tilt).sum(); E["eps"] += closing.float().sum()
        ep_tilt = torch.where(closing | (task.ep_step <= 1), torch.zeros_like(ep_tilt), ep_tilt)
        # table keep-out
        depth = task.table_depth(kin).amax(-1)
        T["steps"] += N; T["deep"] += (depth > 0.0).float().sum()
        stroke_depth = torch.maximum(stroke_depth, depth)
        ref_pos = task.reference_bodies(ref)[0][:, feet]
        # ---------------------------------------------------------------- hold
        hold_run = torch.where(in_hold & ~fresh, hold_run + 1, torch.zeros_like(hold_run))
        d = q_cmd - prev_q
        settled = (hold_run > 50)[:, None].float()
        HT["roll"] += (settled[:, 0] * roll).sum(); HT["pitch"] += (settled[:, 0] * pitch).sum()
        HT["waist"] += (settled * wq).sum(0)
        moved = (d.abs() > 1e-5) & (prev_d.abs() > 1e-5)
        H["n"] += settled.sum()
        H["step2"] += (settled * d.square()).sum(0)
        H["rev"] += (settled * (moved & (d * prev_d < 0)).float()).sum(0)
        H["qd2"] += (settled * s[:, 13 + 31:].square()).sum(0)
        H["err2"] += (settled * (q_cmd - s[:, 13:13 + 31]).square()).sum(0)
        H["act2"] += (settled * trainer.a_prev.square()).sum(0)
        dz = s[:, 2] - ref[:, 2]
        H["dz"] += (settled[:, 0] * dz).sum()
        H["dz2"] += (settled[:, 0] * dz.square()).sum()
        H["knee"] += (settled[:, 0] * (s[:, 13:13 + 31] - ref[:, 13:13 + 31])[:, knees].mean(-1)).sum()
        H["feet"] += (settled[:, 0] * ((pos[:, 0] - pos[:, 1]).norm(dim=-1) - (ref_pos[:, 0] - ref_pos[:, 1]).norm(dim=-1))).sum()
        prev_q, prev_d = q_cmd.clone(), d
        # ---------------------------------------------------------------- strokes
        announced = (tts > prev_tts + 0.1) & ~in_hold & ~fresh & (tts < 1.19)
        active = (active | announced) & ~fresh
        start = torch.where(announced[:, None, None], pos, start)
        ref_start = torch.where(announced[:, None, None], ref_pos, ref_start)
        root0 = torch.where(announced[:, None], s[:, 0:2], root0)
        ref_root0 = torch.where(announced[:, None], ref[:, 0:2], ref_root0)
        for k in run:
            fill = 9.0 if k in ("low", "ref_low") else 0.0
            run[k] = torch.where(announced.view(-1, *([1] * (run[k].ndim - 1))), torch.full_like(run[k], fill), run[k])
        run["travel"] = torch.maximum(run["travel"], (pos - start)[..., :2].norm(dim=-1))
        run["ref_travel"] = torch.maximum(run["ref_travel"], (ref_pos - ref_start)[..., :2].norm(dim=-1))
        run["lift"] = torch.maximum(run["lift"], (pos - start)[..., 2])
        run["ref_lift"] = torch.maximum(run["ref_lift"], (ref_pos - ref_start)[..., 2])
        run["low"], run["ref_low"] = torch.minimum(run["low"], s[:, 2]), torch.minimum(run["ref_low"], ref[:, 2])
        run["speed"] = torch.maximum(run["speed"], s[:, 13 + 31:].abs().amax(-1))
        run["ref_speed"] = torch.maximum(run["ref_speed"], ref[:, 13 + 31:].abs().amax(-1))
        run["root_xy"] = torch.maximum(run["root_xy"], (s[:, 0:2] - root0).norm(dim=-1))
        run["ref_root_xy"] = torch.maximum(run["ref_root_xy"], (ref[:, 0:2] - ref_root0).norm(dim=-1))
        done = active & (tts < -0.3)
        stroke_depth = torch.where(announced, depth, stroke_depth)
        if done.any():
            fam = torch.nn.functional.one_hot(task.family, 4).float() * done[:, None].float()
            T["strokes"] += fam.sum(0); T["hit"] += (fam * (stroke_depth > 0.0)[:, None]).sum(0)
            T["depth"] += (fam * stroke_depth[:, None]).sum(0)
            foot = run["ref_travel"].argmax(-1, keepdim=True)  # the foot the teacher steps with
            pick = lambda x: x.gather(1, foot)[:, 0]  # noqa: E731
            S["n"] += fam.sum(0)
            for k in ("travel", "ref_travel", "lift", "ref_lift"):
                S[k] += (fam * pick(run[k])[:, None]).sum(0)
            for k in ("low", "ref_low", "speed", "ref_speed", "root_xy", "ref_root_xy"):
                S[k] += (fam * run[k][:, None]).sum(0)
            active = active & ~done
        prev_tts = tts.clone()

    n = float(H["n"].clamp_min(1))
    seconds = n * dt
    step_rms, rev_hz = (H["step2"] / n).sqrt(), H["rev"] / seconds
    track_rms = (H["err2"] / n).sqrt()
    worst = lambda x: (a3.JOINTS[int(x.argmax())], float(x.max()))  # noqa: E731
    result = {
        "checkpoint": str(path), "iteration": int(payload["iteration"]), "num_envs": N, "steps": steps, "clean": clean,
        "hold": {
            "env_seconds": seconds,
            "qdes_step_rms_mrad_worst": {worst(step_rms)[0]: 1e3 * worst(step_rms)[1]},
            "qdes_step_rms_mrad_mean": float(1e3 * step_rms.mean()),
            "qdes_reversals_hz_worst": {worst(rev_hz)[0]: worst(rev_hz)[1]},
            "qdes_reversals_hz_mean": float(rev_hz.mean()),
            "joint_speed_rms_worst": {worst((H["qd2"] / n).sqrt())[0]: worst((H["qd2"] / n).sqrt())[1]},
            "qdes_minus_q_rms_rad_worst": {worst(track_rms)[0]: worst(track_rms)[1]},
            "raw_action_rms_worst": {worst((H["act2"] / n).sqrt())[0]: worst((H["act2"] / n).sqrt())[1]},
            "pelvis_height_minus_reference_m": float(H["dz"] / n),
            "pelvis_height_error_rms_m": float((H["dz2"] / n).sqrt()),
            "knee_bend_minus_reference_rad": float(H["knee"] / n),
            "feet_separation_minus_reference_m": float(H["feet"] / n),
        },
        "strokes": {},
    }
    for f, name in enumerate(FAMILIES):
        m = float(S["n"][f].clamp_min(1))
        result["strokes"][name] = {"count": int(S["n"][f])} | {
            k: round(float(S[k][f]) / m, 4) for k in ("travel", "ref_travel", "lift", "ref_lift", "low", "ref_low",
                                                       "speed", "ref_speed", "root_xy", "ref_root_xy")}
    en, eps = float(E["n"].clamp_min(1)), float(E["eps"].clamp_min(1))
    result["entry"] = {"env_seconds": en * dt, "episodes": int(E["eps"]), "pelvis_tilt_vs_reference_deg_mean": float(E["tilt"] / en),
                       "pelvis_tilt_vs_reference_deg_peak_mean": float(E["tilt_peak"] / eps),
                       "pelvis_abs_roll_deg_mean": float(E["roll"] / en), "pelvis_pitch_deg_mean": float(E["pitch"] / en),
                       "waist_yaw_roll_pitch_rad_mean": [float(v) for v in E["waist"] / en],
                       "waist_yaw_roll_pitch_abs_rad_mean": [float(v) for v in E["waist_abs"] / en]}
    result["hold"]["pelvis_roll_pitch_deg_mean"] = [float(HT["roll"] / n), float(HT["pitch"] / n)]
    result["hold"]["waist_yaw_roll_pitch_rad_mean"] = [float(v) for v in HT["waist"] / n]
    result["table"] = {"edge_x": table_x, "top_z": table_z,
                       "share_of_steps_below_top": float(T["deep"] / T["steps"].clamp_min(1)),
                       "share_of_strokes_below_top": {FAMILIES[f]: float(T["hit"][f] / T["strokes"][f].clamp_min(1)) for f in range(4)},
                       "mean_deepest_point_m": {FAMILIES[f]: float(T["depth"][f] / T["strokes"][f].clamp_min(1)) for f in range(4)}}
    out = Path(opts["out"]) if "out" in opts else path.with_name(
        f"motion_audit_{payload['iteration']}{'_clean' if clean else ''}.json")
    out.write_text(json.dumps(result, indent=2))
    h = result["hold"]
    print(f"[audit] hold ({seconds:.0f} env-s): q_des step RMS worst {h['qdes_step_rms_mrad_worst']} mean "
          f"{h['qdes_step_rms_mrad_mean']:.2f} mrad; reversals worst {h['qdes_reversals_hz_worst']} mean "
          f"{h['qdes_reversals_hz_mean']:.1f} Hz; joint speed RMS worst {h['joint_speed_rms_worst']}")
    print(f"[audit] hold posture: pelvis {100 * h['pelvis_height_minus_reference_m']:+.1f} cm vs reference "
          f"(rms {100 * h['pelvis_height_error_rms_m']:.1f}), knees {h['knee_bend_minus_reference_rad']:+.3f} rad, "
          f"feet separation {100 * h['feet_separation_minus_reference_m']:+.1f} cm, q_des - q worst "
          f"{h['qdes_minus_q_rms_rad_worst']}")
    e, tb = result["entry"], result["table"]
    print(f"[audit] hold tilt: pelvis roll {h['pelvis_roll_pitch_deg_mean'][0]:+.1f} pitch {h['pelvis_roll_pitch_deg_mean'][1]:+.1f} deg, "
          f"waist yaw/roll/pitch {[round(v, 3) for v in h['waist_yaw_roll_pitch_rad_mean']]} rad (reference: all zero)")
    print(f"[audit] entry (first 3 s from the default stand, {e['episodes']} episodes): pelvis tilt vs reference mean "
          f"{e['pelvis_tilt_vs_reference_deg_mean']:.1f} deg, peak {e['pelvis_tilt_vs_reference_deg_peak_mean']:.1f} deg; "
          f"waist yaw/roll/pitch mean abs {[round(v, 3) for v in e['waist_yaw_roll_pitch_abs_rad_mean']]} rad")
    print(f"[audit] table (edge x={table_x}, top z={table_z}): strokes with blade rim, handle or hand below the top: "
          + ", ".join(f"{k} {100 * v:.1f}%" for k, v in tb["share_of_strokes_below_top"].items())
          + "; mean deepest point " + ", ".join(f"{100 * v:.1f} cm" for v in tb["mean_deepest_point_m"].values()))
    for name, v in result["strokes"].items():
        print(f"[audit] {name:15s} n={v['count']:5d} step foot travel {100 * v['travel']:5.1f} cm (teacher "
              f"{100 * v['ref_travel']:5.1f})  lift {100 * v['lift']:4.1f} cm (teacher {100 * v['ref_lift']:4.1f})  "
              f"lowest pelvis {v['low']:.3f} m (teacher {v['ref_low']:.3f})  peak joint speed {v['speed']:.1f} rad/s "
              f"(teacher {v['ref_speed']:.1f})  pelvis travel {100 * v['root_xy']:.1f} cm (teacher {100 * v['ref_root_xy']:.1f})")
    print(f"[audit] wrote {out}")


if __name__ == "__main__":
    main()
