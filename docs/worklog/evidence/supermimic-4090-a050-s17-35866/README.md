# SuperMimic RTX 4090 run, α = 0.5, seed 17

Run `supermimic-long-a050-s17-20261124-01` trained the dual-body MLP from random initialization in 6,144 MuJoCo-Warp environments. It used ten observation frames, a 32-step imagination horizon, a full-body dynamics model and a shared team critic. Training stopped at a complete iteration boundary after **35,866 iterations** and exited with status 0.

- [Final checkpoint](checkpoint/final_35866.pt), [independent verification](checkpoint/final_35866.proof.json) and [stop receipt](checkpoint/stop_receipt.json)
- [Training configuration](training/config.json), [original invocation](training/command.sh), [metrics CSV](training/metrics.csv), [raw metrics JSONL](training/metrics.jsonl.gz) and [TensorBoard events](training/tensorboard.tar.gz)
- [Training curves](figures/training-curves.png), [100-ball landing figure](figures/fixed-home-100-landings.png) and [plot script](code/plot_results.py)
- [Continuous 100-ball MP4 at 60 fps](gate3/fixed-home-100/gate3_100_balls_60fps.mp4) and [video receipt](gate3/fixed-home-100/video_receipt.json)
- [Gate3 deployment ONNX](gate3/policy.onnx), [deploy configuration](gate3/deploy.yaml) and [actor parity check](code/bundle-check.log)

The checkpoint is stored in Git LFS. Run `git lfs pull` after checking out the branch to retrieve its contents. SHA-256 is `a34af2d6d80264b6c0267e4f108a2747415e26f317dfdfc2d2dbb8d9ef104364`. Resume restores the model, EMA, optimizers, and iteration count. The simulator starts a new rollout and refills the replay buffer.

## Physical-ball results

Each test starts the motion controller once and then launches balls without inter-ball resets. Results describe simulator telemetry. All three runs failed formal certification.

| Test | Engaged | Physical racket contacts | Legal returns | Falls | Formal verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| [Fixed-HOME 100](gate3/fixed-home-100/physical_summary.json) | 100/100 | 100/100 | 99/100 | 0 | FAIL |
| [Default 26-ball Gate3, simulation-local flight IDs](gate3/gate3-baseline/physical_summary.json) | 26/26 | 26/26 | 25/26 | 0 | FAIL |
| [26-ball Gate3, earlier commits and natural-tail preemption](gate3/gate3-preempt/physical_summary.json) | 26/26 | 26/26 | 26/26 | 0 | FAIL |

The 100-ball rollout recorded 100 complete physical trajectories. The report could not match the Runner's persistent Planner IDs to the simulator's shot IDs. The stable 26-ball baseline used simulation-local IDs and restored the completion join. The earlier-commit candidate completed 26/26 swings and met the rapid transition timing window. Its Planner audit required fixed post-net commit timing, and the run failed ready posture, wrist motion, and pre-strike base-speed checks. The harness and numerical thresholds were the same for the recorded 26-ball comparison.

Each result directory contains the scenario, official report, per-ball report, Planner report, and a compressed archive of plant, ball, observation, Runner and Planner CSVs. The MP4 renders those measured poses and flights at real time.

## Source and reproduction
[Reproduce the training environment and verify a checkpoint resume (English)](REPRODUCE_ZH.md).

Training started from `f3b4e809ab084d05b7019b811785c4aa9354086d`. The Trainer changes in this commit add a RunId-scoped stop request, atomic final save, graph telemetry, and bounded periodic retention. The native Gate3 workspace used fork `main` at `756918a8cfa7d496f6ea88018362be187985187e`. The [baseline compatibility patch](code/native-compatibility.patch) and [earlier-commit candidate patch](code/native-preempt-compatibility.patch) are alternatives against that base. The candidate also used the [earlier flight packetizer configuration](code/candidate-flight-packetizer.yaml). The [standalone bundle exporter](code/export_core_bundle.py) uses the tracked reference clips, native 110-D observation contract, and checkpoint EMA. Shell scripts and receipts retain the local run paths for provenance. Substitute your workspace and native install paths when rebuilding.
