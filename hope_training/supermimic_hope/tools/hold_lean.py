"""How a checkpoint holds when the masses are not the model's: trunk lean against hand mass (grasping container).

    /opt/drone_venv/bin/python tools/hold_lean.py <model_N.pt> [device=cuda:0] [left=0.28,0.55,0.85,1.15] [right=0,0.3] [links=1]

For each left-hand mass (kg) and right-hand extra (kg), 256 robots enter MOTION from the official stand through
the 3 s handoff and hold for 30 s; everything else is nominal unless ``links=1`` (every link's mass also varies by
the training range, +-10%). Prints the trunk and pelvis roll, the waist roll joint, the left elbow and how far the
trunk sits to the left of the midpoint of the feet, averaged over the last 5 s, and the spread across robots.
Negative roll is a lean to the robot's left (seen along its forward direction).

The robot we deploy on carries a 0.847 kg gripper on the left hand; the Gate3 MJCF's hand is 0.281 kg. A policy
trained with that hand alone leaned 4.7 deg left on the robot (docs/real_hold_lean_20261120.md). The check is that
the lean stays small over the whole range, not only at the mass it was trained with.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import apply_physics, from_dict  # noqa: E402


def main(argv=None):
    import torch

    from supermimic_hope import a3
    from supermimic_hope.core.rotations import quat_to_matrix
    from supermimic_hope.core.trainer import TrainCfg, Trainer

    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0])
    opts = dict(a.split("=", 1) for a in argv[1:])
    lefts = [float(x) for x in opts.get("left", "0.28,0.55,0.85,1.15").split(",")]
    rights = [float(x) for x in opts.get("right", "0,0.3").split(",")]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    joints = list(a3.JOINTS)
    print(f"{path}  (negative roll: lean to the robot's left)")
    print("left hand  right hand | trunk roll (sd, worst)   pelvis roll  waist roll joint  left elbow | trunk left of the feet")
    for left in lefts:
        for extra in rights:
            cfg = from_dict(TrainCfg, payload["config"])
            cfg.device, cfg.num_envs, cfg.seed = opts.get("device", "cuda:0"), 256, 7
            cfg.buffer_steps, cfg.capture = 4, False
            t, s = cfg.task, cfg.sim
            t.stand_probability, t.long_hold_probability, t.long_hold_steps = 1.0, 1.0, (1900, 1950)
            t.extra_probability, t.extra_bank, t.mocap_dropout_rate, t.push_rate = 0.0, "", 0.0, 0.0
            t.mocap_rpy_bias, t.mocap_bias, t.joint_offset = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0
            apply_physics(cfg, dict(contact="elliptic", handoff="1"))
            s.com_x = s.com_y = s.com_z = (0.0, 0.0)
            s.mass_scale, s.pd_scale, s.nominal_fraction = (1.0, 1.0), (1.0, 1.0), 0.0
            s.left_hand_mass, s.left_hand_mass_range, s.right_hand_mass_delta = 0.847, (left, left), (extra, extra)
            if opts.get("links", "0") == "0":
                s.link_mass_scale = (1.0, 1.0)
            trainer = Trainer(cfg)
            trainer.load(path)
            if "actor_ema" in payload:
                trainer.actor.load_state_dict(payload["actor_ema"])
            trainer.actor.eval()
            task, sim = trainer.task, trainer.sim
            feet = [task.kin.names.index(n) for n in a3.FEET]
            torso = task.kin.names.index("torso_Link")
            trainer.collect(1250, noise=0.0, learn=False)  # 25 s: through the handoff and settled
            rows = []
            for _ in range(10):
                trainer.collect(25, noise=0.0, learn=False)
                state = sim.state()
                pos, rot, _, _ = task.kin(state)
                R, Rt = quat_to_matrix(state[:, 3:7]), rot[:, torso]
                roll = lambda M: torch.rad2deg(torch.atan2(M[:, 2, 1], M[:, 2, 2]))  # noqa: E731
                mid, yaw = pos[:, feet].mean(1), torch.atan2(R[:, 1, 0], R[:, 0, 0])
                left_of = -(pos[:, torso, 0] - mid[:, 0]) * torch.sin(yaw) + (pos[:, torso, 1] - mid[:, 1]) * torch.cos(yaw)
                rows.append(torch.stack((roll(Rt), roll(R), torch.rad2deg(state[:, 13 + joints.index("waist_roll_joint")]),
                                         torch.rad2deg(state[:, 13 + joints.index("left_elbow_joint")]), 100 * left_of), -1))
            per = torch.stack(rows).mean(0)
            m, sd = per.mean(0).tolist(), per.std(0).tolist()
            print(f"{left:6.2f} kg  {extra:+5.2f} kg  | {m[0]:+6.1f} deg ({sd[0]:.1f}, {per[:, 0].abs().max().item():.1f})      {m[1]:+5.1f} deg   {m[2]:+6.1f} deg      "
                  f"{m[3]:+6.1f} deg | {m[4]:+5.1f} cm")
            del trainer


if __name__ == "__main__":
    main()
