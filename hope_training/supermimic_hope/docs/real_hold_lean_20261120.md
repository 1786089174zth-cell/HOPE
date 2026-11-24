# The trunk leans left in HOLD on the robot: the gripper (2026-11-20)

model_54000 on the robot (receive-only session of 2026-11-20, 29 HOLD windows, 44 282 stable frames; the log
package is in `logs/real_54000_hold/`, not in the repository):

| in HOLD | robot | Gate3 (same policy) |
| --- | --- | --- |
| pelvis roll (mocap and pelvis IMU) | -0.5 .. -0.8 deg | +0.3 deg |
| trunk roll (torso IMU) | -4.2 .. -4.9 deg | -0.2 deg |
| waist roll joint: target / actual / torque | +2.5 / -4.7 deg / +6.3 N.m | -0.9 / -0.3 deg / -0.6 N.m |
| left elbow: target / actual / torque | +10.5 / +18.6 deg / -4.2 N.m | +1.2 / +6.1 deg / -2.6 N.m |
| left shoulder pitch torque | -5.8 N.m | -4.1 N.m |

Negative roll is a lean to the robot's left. The pelvis is level; the trunk leans on the waist roll joint, which the
policy commands the other way by only 2.5 deg. The left arm carries more than in the simulator.

## Cause

The robot carries a gripper on the left hand that the training model does not have. Main's
`a3_deploy/a3_deploy_example/assets/a3_runtime/robots/A3PingPong-with-gripper/urdf/model.urdf` has 0.847 kg on
`left_wrist_yaw_Link` (gripper links folded in), centre 0.113 m along the link; the Gate3 MJCF the training model
and both simulated tests are built from has 0.281 kg at 0.084 m (and a torso 1.2 kg heavier; nothing else differs
by more than 0.05 kg).

With that hand in the simulator the same policy holds like the robot: trunk roll -4.9 deg, waist roll joint
-4.8 deg (`SimCfg.left_hand_mass`). Added mass on the left hand, HOLD of model_54000 in the trainer's simulator:
+0.3 kg -2.4 deg, +0.6 kg -4.4 deg, +1.0 kg -7.0 deg, left elbow 1-4 deg lower.

Why half a kilogram does this: the body above the waist roll axis is 29.4 kg with its centre 21 cm above it,
61 N.m/rad of gravity pulling a leaning trunk further over, against a commanded stiffness of 50 N.m/rad. The
joint alone does not hold the trunk up; the policy does, and it had never met a weight in the hand. (It does
correct what training randomized: the torso's own centre of mass moved 5 cm sideways, 8.6 N.m, changes the lean by
0.6 deg.)

What it is not, each injected in the simulator: mocap attitude offsets of 2-4 deg (0.2 deg of lean), mocap
position offsets (the pelvis moves, no lean), hip, ankle or waist joint zero offsets of 1.5 deg (at most 1.2 deg),
floor friction 0.5-4.0 (none). The policy's observations on the robot are otherwise normal: gyro mean zero,
pelvis level, at HOME.

## What changed

* The training simulator has the gripper's left hand by default (`SimCfg.left_hand_mass = 0.847`,
  `left_hand_com_x = 0.1128`) and varies both hands' mass by +-0.12 kg across worlds (`hand_mass_jitter`).
  `evaluate.py ... left_hand=bare` scores with the MJCF's hand.
* Supergate runs carry the gripper by default (`tools/supergate/hand.py`, `SUPERGATE_LEFT_HAND`; `score.sh` with
  `SUPERGATE_HAND=bare` for the old model). The Gate3 plant still has the bare hand.
* Tools: `tools/gate3/lean.py <run dir>` (lean during idle and between balls from a Gate3 plant log).

Checkpoints trained before this (up to ft22) lean on the robot.
