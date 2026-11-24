"""Gate3's policy entry replayed on the CPU: the stand, the Runner's 3 s handoff, then the wait for the first ball.

    python tools/entry_replica.py logs/supermimic/<run>/model_<N>.pt [device=cuda:0 num_envs=4 seconds=20]
        [contact=gate3|training handoff=3.0 friction=1.5 clean=0 trace=0 out=<json>]
        [dt=0.001 noslip=3 cone=pyramidal|elliptic impratio=1]   single solver options on top of ``contact``

The training simulator differs from Gate3's plant in two ways that decide what happens at entry,
and this tool switches each on its own (same MJCF, CPU MuJoCo, the checkpoint's averaged actor):

* ``contact=gate3``: the plant's own options -- 1 ms steps, three no-slip iterations, floor and
  sole friction 1.5. ``contact=training``: 5 ms implicit-fast steps without no-slip, as trained.
* ``handoff=S``: the Runner's MOTION entry (``PpCommandTransition::BeginImpedance``), which
  cross-fades the official stand PD into the policy PD over ``S`` seconds; ``handoff=0`` gives the
  policy full authority from the first tick, as a training episode does.

Reported: feet width (HOLD reference 0.47 m), torso and waist roll, hip-roll command error and
foot lift over time. Gate3 itself measured 0.334 m and +3.5 deg torso roll for model_39000.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402,F401 -- sets up sys.path for the pinned dependencies
from evaluate import from_dict  # noqa: E402

def handoff_weight(elapsed, duration):
    """Weight of the stand controller ``elapsed`` seconds into the entry (1 -> 0, smootherstep)."""
    if duration <= 0 or elapsed >= duration:
        return 0.0
    u = max(0.0, elapsed / duration)
    return 1 - 10 * u ** 3 + 15 * u ** 4 - 6 * u ** 5


class CpuSim:
    """``WarpSim``'s interface on CPU MuJoCo worlds, with the plant options and the entry handoff."""

    def __init__(self, n, device, contact="gate3", friction=1.5, handoff=3.0, overrides=None):
        import mujoco
        import numpy as np
        import torch
        from supermimic_hope import a3

        self.mj, self.np, self.torch, self.a3 = mujoco, np, torch, a3
        self.N, self.device = n, torch.device(device)
        self.model, self.receipt = a3.build_model()
        m = self.model
        floor = m.geom("floor").id
        soles = np.flatnonzero(np.isin(m.geom_bodyid, [m.body(name).id for name in a3.FEET]))
        if contact == "gate3":
            m.opt.timestep = 0.001
            m.opt.noslip_iterations = 3
            m.opt.noslip_tolerance = 1e-6
            m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
            m.geom_friction[floor, 0] = friction
            m.geom_friction[soles, 0] = friction
        elif contact == "training":  # a3.build_model's options; WarpSim puts the coefficient on the soles
            m.geom_friction[floor, 0] = 0.0
            m.geom_friction[soles, 0] = friction
        else:
            raise ValueError("contact must be gate3 or training")
        for key, value in (overrides or {}).items():  # single solver options on top of the base
            if key == "dt":
                m.opt.timestep = float(value)
            elif key == "noslip":
                m.opt.noslip_iterations = int(value)
            elif key == "cone":
                m.opt.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[value]
            elif key == "impratio":
                m.opt.impratio = float(value)
        self.substeps = int(round(a3.CONTROL_DT / m.opt.timestep))
        self.ix = a3.indices(m)
        self.data = [mujoco.MjData(m) for _ in range(n)]
        self.kp, self.kd = np.asarray(a3.KP, float), np.asarray(a3.KD, float)
        from supermimic_hope.core.task import stand_gains

        kp_s, kd_s = stand_gains(a3.JOINTS)
        self.kp_s, self.kd_s, self.q_s = np.asarray(kp_s), np.asarray(kd_s), np.asarray(a3.DEFAULT_Q, float)
        self.handoff, self.elapsed = float(handoff), np.zeros(n)
        self.friction = torch.full((n,), float(friction), device=self.device)
        self.feet_bodies = [m.body(name).id for name in a3.FEET]
        self.normal_force = np.zeros((n, 2))

    def state(self):
        np, ix = self.np, self.ix
        rows = [np.concatenate((d.qpos[:7], d.qvel[:6], d.qpos[ix["qpos"]], d.qvel[ix["dof"]])) for d in self.data]
        return self.torch.as_tensor(np.stack(rows), dtype=self.torch.float32, device=self.device)

    def set_state(self, ids, s):
        s, ix = s.detach().cpu().double().numpy(), self.ix
        for row, i in zip(s, ids.tolist()):
            d = self.data[i]
            self.mj.mj_resetData(self.model, d)
            d.qpos[:7], d.qvel[:6] = row[:7], row[7:13]
            d.qpos[ix["qpos"]], d.qvel[ix["dof"]] = row[13:44], row[44:75]
            self.mj.mj_forward(self.model, d)
            self.elapsed[i] = 0.0

    def step(self, qdes, kp=None, kd=None):  # the task's own handoff is off here; the gains are this class's
        mj, m, np = self.mj, self.model, self.np
        qdes = qdes.detach().cpu().double().numpy()
        for i, d in enumerate(self.data):
            h = handoff_weight(self.elapsed[i], self.handoff)
            kp = h * self.kp_s + (1 - h) * self.kp
            kd = h * self.kd_s + (1 - h) * self.kd
            target = (h * self.kp_s * self.q_s + (1 - h) * self.kp * qdes[i]) / kp
            m.actuator_gainprm[:, 0] = kp
            m.actuator_biasprm[:, 1] = -kp
            m.actuator_biasprm[:, 2] = -kd
            d.ctrl[:] = target
            for _ in range(self.substeps):
                mj.mj_step(m, d)
            self.elapsed[i] += self.a3.CONTROL_DT
            force, wrench = np.zeros(2), np.zeros(6)
            for c in range(d.ncon):
                body = m.geom_bodyid[d.contact[c].geom1], m.geom_bodyid[d.contact[c].geom2]
                for side, foot in enumerate(self.feet_bodies):
                    if foot in body:
                        mj.mj_contactForce(m, d, c, wrench)
                        force[side] += wrench[0]
            self.normal_force[i] = force


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
    cfg.num_envs = int(opts.get("num_envs", 4))
    cfg.seed = int(opts.get("seed", 12345))
    cfg.buffer_steps, cfg.capture = 4, False
    seconds = float(opts.get("seconds", 20.0))
    steps = int(round(seconds / 0.02))
    contact, handoff = opts.get("contact", "gate3"), float(opts.get("handoff", 3.0))
    friction = float(opts.get("friction", 1.5))
    t = cfg.task
    t.stand_probability, t.long_hold_probability = 1.0, 1.0
    t.long_hold_steps, t.episode_steps = (steps + 50, steps + 50), steps + 100
    t.stance_blend_probability, t.stand_x, t.stand_yaw = 0.0, (0.0, 0.0), 0.0  # Gate3: on the station, facing the table
    t.joint_offset = 0.0
    t.entry_handoff_probability = 0.0  # the handoff is CpuSim's here, written independently of the task's
    if opts.get("clean", "0") != "0":
        t.noise_gyro = t.noise_joint_pos = t.noise_joint_vel = t.target_noise = t.target_bias = 0.0
        t.mocap_bias = t.mocap_noise = t.mocap_rpy_bias = t.mocap_rpy_noise = (0.0, 0.0, 0.0)
    trainer = Trainer(cfg)
    trainer.load(path)
    if opts.get("ema", "1") != "0" and "actor_ema" in payload:
        trainer.actor.load_state_dict(payload["actor_ema"])
    trainer.actor.eval()
    task, dev, N = trainer.task, trainer.task.device, cfg.num_envs
    overrides = {k: opts[k] for k in ("dt", "noslip", "cone", "impratio") if k in opts}
    sim = CpuSim(N, dev, contact=contact, friction=friction, handoff=handoff, overrides=overrides)
    trainer.sim = task.sim = sim
    ids = torch.arange(N, device=dev)
    task.reset(ids)
    # Let the stand settle under the stand controller alone before the policy takes over.
    hold, sim.handoff = sim.handoff, 1e9
    for _ in range(50):
        sim.step(task.q0)
    sim.handoff, sim.elapsed[:] = hold, 0.0
    trainer.a_prev[:] = 0.0
    trainer.a_prev2[:] = 0.0
    trainer.q_prev[:] = task.q0
    trainer.history[:] = task.frame(sim.state(), trainer.a_prev, task.tape(), noise=True)[:, None]

    feet = torch.tensor(task.kin.index(list(a3.FEET)), device=dev)
    torso = task.kin.index(["torso_Link"])[0]
    waist_roll = a3.JOINTS.index("waist_roll_joint")
    hips = [a3.JOINTS.index("left_hip_roll_joint"), a3.JOINTS.index("right_hip_roll_joint")]
    roll = lambda R: torch.rad2deg(torch.atan2(R[:, 2, 1], R[:, 2, 2]))  # noqa: E731
    reference = task.reference_bodies(task.hold_ref[None])[0][0, feet]
    stand_z = task.kin(sim.state())[0][:, feet, 2].clone()
    every = int(opts.get("trace", 0))
    marks = {int(round(s / 0.02)) for s in (0.0, 0.5, 1, 2, 3, 4, 5, 7.5, 10, 15, seconds) if s <= seconds}
    if every:
        marks |= set(range(0, steps + 1, every))
    rows, lifted, unloaded = [], torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    steady, steady_from = [], int(round(6.0 / 0.02))  # after the handoff and the stance change have settled
    for k in range(steps + 1):
        s = sim.state()
        bodies = task.kin(s)
        pos = bodies[0][:, feet]
        lifted += ((pos[..., 2] - stand_z).amax(-1) > 0.01).float()
        force = torch.as_tensor(sim.normal_force, dtype=torch.float32, device=dev)
        unloaded += (force.amin(-1) < 30.0).float() if k else 0.0
        if k >= steady_from:
            steady.append(torch.stack((roll(bodies[1][:, torso]), torch.rad2deg(s[:, 13 + a3.NJ + waist_roll]),
                                       trainer.q_prev[:, hips[0]], trainer.q_prev[:, hips[1]]), -1))
        if k in marks:
            q = s[:, 13:13 + a3.NJ]
            width = pos[:, 0, 1] - pos[:, 1, 1]
            torso_roll = roll(bodies[1][:, torso])
            rows.append({
                "t_s": round(k * 0.02, 2), "stand_weight": handoff_weight(k * 0.02, handoff),
                "feet_width_m": [float(width.mean()), float(width.min()), float(width.max())],
                "torso_roll_deg": [float(torso_roll.mean()), float(torso_roll.abs().max())],
                "pelvis_roll_deg": float(roll(quat_to_matrix(s[:, 3:7])).mean()),
                "waist_roll_deg": float(torch.rad2deg(q[:, waist_roll]).mean()),
                "hip_roll_command_minus_joint_deg": [float(v) for v in torch.rad2deg((trainer.q_prev - q)[:, hips]).mean(0)],
                "pelvis_xyz_m": [float(v) for v in s[:, :3].mean(0)],
                "foot_normal_force_N": [float(v) for v in force.mean(0)],
                "foot_lifted_1cm_s": float(lifted.mean() * 0.02), "foot_unloaded_s": float(unloaded.mean() * 0.02),
            })
        if k < steps:
            trainer.collect(1, noise=0.0, learn=False)
    result = {"checkpoint": str(path), "contact": contact, "handoff_s": handoff, "friction": friction,
              "timestep_s": float(sim.model.opt.timestep), "noslip_iterations": int(sim.model.opt.noslip_iterations),
              "num_envs": N, "seconds": seconds, "reference_feet_width_m": float((reference[0] - reference[1]).norm()),
              "timeline": rows}
    if len(steady) > 2:
        x = torch.stack(steady)  # (T, N, 4)
        step = (x[1:, :, 2:] - x[:-1, :, 2:])
        result["steady_from_6s"] = {
            "torso_roll_deg_mean": float(x[..., 0].mean()), "torso_roll_deg_std_over_time": float(x[..., 0].std(0).mean()),
            "torso_roll_deg_abs_max": float(x[..., 0].abs().max()),
            "waist_roll_speed_rms_deg_s": float(x[..., 1].square().mean().sqrt()),
            "hip_roll_command_step_rms_mrad": float(1000 * step.square().mean().sqrt())}
    print(f"[replica] {path.name}: contact={contact} (dt {sim.model.opt.timestep * 1e3:.0f} ms, noslip "
          f"{sim.model.opt.noslip_iterations}, cone {'elliptic' if sim.model.opt.cone else 'pyramidal'}, impratio "
          f"{sim.model.opt.impratio:g}, friction {friction}), handoff {handoff:.1f} s, {N} worlds; "
          f"HOLD reference feet width {result['reference_feet_width_m']:.3f} m")
    print("[replica]    t  stand   feet width (min-max)   torso roll (max|.|)  waist roll  hip-roll cmd-q L/R   pelvis x / z     N L/R     lifted  unloaded")
    for r in rows:
        w, tr, e, p, f = r["feet_width_m"], r["torso_roll_deg"], r["hip_roll_command_minus_joint_deg"], r["pelvis_xyz_m"], r["foot_normal_force_N"]
        print(f"[replica] {r['t_s']:5.1f}  {r['stand_weight']:.2f}   {w[0]:.3f} ({w[1]:.3f}-{w[2]:.3f})    {tr[0]:+5.1f} ({tr[1]:4.1f}) deg     "
              f"{r['waist_roll_deg']:+5.1f}      {e[0]:+6.1f} / {e[1]:+6.1f}     {p[0]:+.3f} / {p[2]:.3f}   {f[0]:4.0f}/{f[1]:4.0f}   "
              f"{r['foot_lifted_1cm_s']:.2f} s  {r['foot_unloaded_s']:.2f} s")
    if "steady_from_6s" in result:
        z = result["steady_from_6s"]
        print(f"[replica] steady (from 6 s): torso roll mean {z['torso_roll_deg_mean']:+.2f} deg, std over time "
              f"{z['torso_roll_deg_std_over_time']:.2f}, max |roll| {z['torso_roll_deg_abs_max']:.2f}; waist roll speed RMS "
              f"{z['waist_roll_speed_rms_deg_s']:.1f} deg/s; hip-roll command step RMS {z['hip_roll_command_step_rms_mrad']:.1f} mrad")
    if "out" in opts:
        Path(opts["out"]).write_text(json.dumps(result, indent=2) + "\n")
        print(f"[replica] wrote {opts['out']}")
    return result


if __name__ == "__main__":
    main()
