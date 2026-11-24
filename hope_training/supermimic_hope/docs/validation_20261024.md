# MuJoCo-Warp integration and supplied-policy noise audit — 2026-10-24

## Scope

Workspace baseline: `main_four@a60fb85a75fb2fdac869b81c9c1e6f038d11c974`, plus the
new SuperMimic integration. The original MotionQuality YAML, PPO algorithm,
reward implementation and task assets were not edited. No user-owned training
or hardware process was started, stopped or replaced. Only bounded probes and
short optimizer smoke tests were run.

The integration retains the current v24 PPQ004 bank. The physical bank SHA256
is `0b5bef38cada63b9ae373af22d32d9e121eb041ff3fe083e4e5db429245e17f1`;
runtime loaded its `ball_physics.yaml`. The robot MJCF SHA256 is
`4887b5301915830a298bee781c897da9448d45cf41ac91da6ce7440a7c3bbd22`.

## What passed

- 24 unit/regression tests: multi-step replay/reset masks, held-out model data,
  finite BPTT actor/value updates, frozen model weights during actor updates,
  passive output handling, PD torque limits, existing sensor-noise loss and
  four-stroke recovery behavior.
- Final Warp probe: 32 environments × 256 control steps, zero optimizer
  updates, all four stroke families, 107 automatic resets and two explicit
  partial resets. These resets include falls under **zero actions** and are
  not evidence of a stable trained policy.
- All probe observations, rewards, body positions and contact forces remained
  finite; applied joint torques stayed within resolved effort limits.
- Independent MuJoCo CPU versus Warp snapshot checks: maximum body-position
  error below `3e-7 m`, body COM-velocity error below `8e-7 m/s`; root COM
  velocity write/read error below `1.5e-8`. These compare the same model/state,
  not long-horizon PhysX/MuJoCo physical behavior.
- Isaac/MJCF initial link-frame alignment: maximum position error `3.0e-6 m`.
  Full resolved inertia transforms to its principal frame with off-diagonal
  residual below `3.5e-9 kg m²`.
- 256-environment default-size world models completed eight short updates,
  including seven actor/value updates, actual task metrics and a saved
  `model_8.pt`. The smoke test intentionally used one model-only iteration;
  normal training retains the 200-iteration model warmup.

Local evidence is under
`analysis/supermimic_motion_quality_20261024/`: `warp_probe_09/probe.json`,
`default_update_final/`, `default_update_smoke_02/`, and `isaac_reference_01/`.
The compact tracked receipt records the final evidence and source hashes.

## Speed: measured scope and limits

RTX 5070 Ti Laptop GPU, 256 environments, same task, same world-model sizes,
same eight-update command, seed 0. Means below exclude iteration 0; CUDA is
synchronized at collection/learning boundaries. Bootstrap, initial replay
collection, logging and checkpoint I/O are excluded.

| Backend / implementation | Collection | Learning | Iteration |
|---|---:|---:|---:|
| Warp before removing redundant reset contact solves | 0.434 s | 0.078 s | 0.513 s |
| Warp after reset optimization | 0.296 s | 0.077 s | 0.373 s |
| Isaac reference, same world-model learner | 0.280 s | 0.077 s | 0.357 s |

This short comparison does **not** establish a Warp throughput advantage:
the retained Python task managers and reset work still cost substantial time.
The solvers produce different trajectories/reset schedules, and there are
only seven timed actor-update iterations in each mean. The promising change
to test is reduced real-simulator sampling through the world model, not a
claimed speedup of the physics engine.

The new iteration collects an average 2.5 real vector steps instead of PPO's
64. Environment counts also differ from the earlier 6144-env training. Thus
`0.37 s` versus the user's earlier `10 s` is not an equivalent-work comparison
and does not prove faster convergence. Formal learning curves, fixed-bank
evaluation and Gate3 results remain **NOT_PROVEN**.

### Larger-environment follow-up

The initial 256 environments were an integration-test size. Additional
eight-update tests (32 warmup vector steps, one model-only iteration, seven
actor/value updates) completed successfully with larger real collection and
learning batches. A subsequent 6144-environment test also passed with batch
2048. The default is now **6144 environments / batch 2048**.

| Environments | World-model/actor batch | Mean iteration | Observed device memory peak |
|---:|---:|---:|---:|
| 256 | 256 | 0.373 s | not sampled |
| 2048 | 1024 | 0.425 s | 5686 MiB / 5.55 GiB |
| 4096 | 2048 | 0.547 s | 8216 MiB / 8.02 GiB |
| 6144 | 2048 | 0.611 s | 10262 MiB / 10.02 GiB |

The same timing exclusions above apply. Memory is whole-device usage sampled
with `nvidia-smi` every 0.5 s; it is an observed peak, not an allocator's exact
maximum. Sampling for the 2048 case began after bootstrap; the 4096 and 6144
cases include bootstrap. All three cases saved `model_8.pt` and completion
receipts. 6144 leaves approximately 1.9 GiB of device memory available at the
observed peak, compared with 3.9 GiB at 4096. With batch held at 2048, measured
real transition throughput rose from 19253/s to 25861/s (about 34%). There is
no hard 4096 environment limit. Increased real data throughput does not prove
faster policy convergence; long-run memory usage has not been measured.

Evidence: `analysis/supermimic_motion_quality_20261024/scale_2048_b1024/`,
`scale_4096_b2048/`, `scale_6144_b2048/`, their `.gpu.jsonl` samples and
[compact scaling receipt](scaling_20261024.json). No formal long training was
started by these tests.

## Supplied real-robot policy

The public Drive package contains `model_11999`, with ONNX metadata identifying
run `2026-10-20_01-55-32_rally_v23_wide_right_scratch_16k`.

- ONNX SHA256: `dbcb102a867c209cda962ba6a6ffda2ee1c026c3bdf39163df7e227508bb13bf`.
- Embedded checkpoint SHA256: `80cc3ffb528ca456dfb54ddc0869a96e11ec35c6800aebcf2c7712efd3e476e8`.
- Its embedded action config has the older 0–1 control-tick command-delay
  randomization; it does not contain the newer sub-tick state/command latency
  settings. The embedded HOLD command-rate multiplier is 1, versus current 10.
- The inspected package provides model/deployment artifacts, not synchronized
  real joint command/feedback, current, torque or controller timing traces.
  The exact old optimizer loss configuration is not proven by task metadata.

`tools/audit_policy_noise.py` reconstructs the frozen folded-normalizer ONNX
actor, first checks it against ONNX Runtime (eight random probes, maximum
absolute discrepancy `3.43e-5`), then applies 2048 paired perturbations at one
synthetic supplied-HOLD state. Previous action is held fixed. Outputs include
the deployment affine map, safe clamp and passive-head exclusion.

| Synthetic input perturbation | Worst joint | q_des RMS difference | Kp × difference |
|---|---|---:|---:|
| Base mocap Gaussian σ = 1 mm | Right ankle pitch | 16.68 mrad | 0.834 Nm |
| Joint position uniform ±0.01 rad | Waist roll | 38.49 mrad | 1.924 Nm |
| Joint velocity uniform ±0.5 rad/s | Waist roll | 69.16 mrad | 3.458 Nm |
| Gyro uniform ±0.2 rad/s | Right ankle pitch | 33.45 mrad | 1.673 Nm |

These are **synthetic sensitivity measurements at one state**, not measured
hardware command jitter or motor torque. `Kp × difference` is a proportional
torque equivalent, not actual motor current/torque or acoustic loudness.
They establish a plausible noise-amplification mechanism in the supplied
policy; they cannot identify every cause of the reported hardware noise.

Current code addresses that mechanism with the deterministic actor-mean
sensor-noise loss (`coef=20`, including coupled mocap perturbations), sub-tick
state delay 0–4 × 5 ms and command delay 0–3 × 5 ms, stronger HOLD command-rate
cost, and earlier quietness training. Those changes are carried into the new
learner. They do not modify the old ONNX weights. Therefore **the supplied
model's hardware jitter/noise is not proven fixed**, and neither is hardware
behavior of a newly trained model.

Raw offline audit: `analysis/supermimic_motion_quality_20261024/model_11999/noise_audit.json`.
Replay it with the frozen ONNX and
`assets/four_stroke_v15/hold_first_frame.json` using `tools/audit_policy_noise.py`.

## What Unitree RL Lab and SONIC actually contribute

The reviewed G1 implementations choose actuator gains from reflected inertia,
natural frequency and damping ratio, and scale actions relative to torque
capacity and stiffness. Unitree RL Lab also penalizes joint acceleration,
joint torque and action changes; these terms coexist with full-body motion
tracking. See its pinned
[robot configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py)
and [dance task](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/tracking_env_cfg.py).

SONIC's G1 implementation similarly ties gains/action scales to the actuator
model and has action-rate and ankle-acceleration costs. Its new-embodiment
guide emphasizes matching model, joint order and control parameters. Sources:
[robot](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/45c03facbf848e0c743091b7d9c17952096ff908/gear_sonic/envs/manager_env/robots/g1.py),
[action-rate cost](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/45c03facbf848e0c743091b7d9c17952096ff908/gear_sonic/config/manager_env/rewards/terms/action_rate_l2.yaml),
[new embodiments](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/45c03facbf848e0c743091b7d9c17952096ff908/docs/source/user_guide/new_embodiments.md).

The applicable lesson is to preserve consistent actuator dynamics and train
smooth deterministic outputs across the complete movement and latency/noise
distribution. G1's numerical gains should not be copied onto A3. The Warp
integration corrects torque-limit transfer, damping counting, full inertia and
velocity-frame semantics; those are simulator migration fixes, not proof of
the cause or resolution of the already-reported hardware fault.
