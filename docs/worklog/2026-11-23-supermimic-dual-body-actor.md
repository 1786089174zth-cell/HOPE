# 2026-11-23 · Upper/lower SuperMimic actor on HOPE

- Branch `feature/supermimic-dual-body-actor` starts at `de5fd59feaf3b6d607e40ed60bc1850f1a784693` on the fork's `feature/supermimic-training-port`. Implementation commit `1e6aac4cb17ad82bff20d0b4919f5ebed3007360`. This follow-up commit records the runs.
- Allen's November 21 dual-MLP plan assigns the waist and legs to Lower, and arms and passive head slots to Upper. The SuperMimic port uses a different interleaved joint order from the plan, so the actor routes by A3 joint name. At 50 Hz, Lower reads Upper's current masked command, before exploration noise. One full-body dynamics model and one critic train the two MLPs through the existing reward.
- Changes cover the core actor and trainer, checkpoint architecture selection, ONNX/Gate3 export and bundle verification, CPU regressions, and the package README. A scoped Git attribute keeps the pinned simulator XML byte-exact on Windows checkouts. Its Git blob is unchanged.
- Both runs used 8 MuJoCo-Warp environments on one NVIDIA RTX 5080 Laptop GPU, driver 617.42, seed 17, Python 3.11.15, torch 2.7.0+cu128, MuJoCo 3.5.0, mujoco-warp 3.5.0.2 and Warp 1.12.0. The actors retained 10 frames and full hidden widths (2048, 1024, 512, 256), with 4,979,984 Upper and 4,930,575 Lower parameters. Both runs started from random initialization. Imagination horizon and minibatches were reduced for these bounded checks.
- Run `supermimic_dual_smoke_20261123_01` tested the implementation working tree before commit. The subsequent load fallback for older checkpoints was unused by this fresh run. Run `supermimic_dual_graph_20261123_01` tested the committed tree with CUDA graph capture and replay enabled.

## Reproduction

From `hope_training/supermimic_hope`, using PowerShell:

```powershell
$py = 'D:/APP/miniconda/envs/isaaclab/python.exe'
$env:HOPE_SUPERMIMIC_DEPS = 'D:/CapstoneTraining/supermimic_fork_deps_py311_20261123'
$env:PYTHONPATH = $env:HOPE_SUPERMIMIC_DEPS
& $py train.py actor_architecture=dual_body_mlp_v1 run_name=supermimic_dual_smoke_20261123_01 out=D:/CapstoneTraining/supermimic_dual_smoke_20261123_01/train num_envs=8 seed=17 device=cuda:0 max_iterations=2 wm_only_iterations=0 warmup_steps=24 buffer_steps=48 learn.wm_batches=1 learn.wm_batch=8 learn.wm_window=4 learn.batch=8 learn.horizon=4 learn.critic_minibatches=2 learn.critic_epochs=1 capture=false log_interval=1 save_interval=1
& $py train.py actor_architecture=dual_body_mlp_v1 run_name=supermimic_dual_graph_20261123_01 out=D:/CapstoneTraining/supermimic_dual_graph_20261123_01/train num_envs=8 seed=17 device=cuda:0 max_iterations=5 wm_only_iterations=0 warmup_steps=24 buffer_steps=48 learn.wm_batches=1 learn.wm_batch=8 learn.wm_window=4 learn.batch=8 learn.horizon=4 learn.critic_minibatches=2 learn.critic_epochs=1 capture=true log_interval=1 save_interval=5
& $py export.py D:/CapstoneTraining/supermimic_dual_smoke_20261123_01/train/model_2.pt
& $py evaluate.py D:/CapstoneTraining/supermimic_dual_smoke_20261123_01/train/model_2.pt num_envs=8 seed=17 steps=32 device=cuda:0 contact=elliptic
& $py -m pytest -q tests/test_dual_body_actor.py tests/test_core.py tests/test_four_stroke_port.py tests/test_a3.py
& $py -m pytest -q tests
```

## Evidence and limits

| Check | Pass criterion and observed result |
|---|---|
| Routing and differentiation | 27 targeted tests passed, including exclusive named joint routing, masked heads, gradient from a waist output into Upper, observation selection, flattened inference and checkpoint round trips. |
| Eager training | Two logged updates and two checkpoints. World-model losses 0.341672 and 0.658037, policy gradient norms 2.3546 and 15.8704. Checkpoint comparison confirmed finite nonzero changes to both branches, both EMA branches, the critic and dynamics network. |
| CUDA graphs | All five updates had finite world-model and value losses and positive policy gradient norms. The fourth update captured and the fifth replayed with the built-in three-update warmup. The final world-model loss was 0.808419. |
| Reload and export | EMA checkpoint reloaded in the GPU evaluator. Its 32 scored steps per environment after the standard 200 warmup steps logged 2 strike opportunities and 0 hits. Exported 1100-to-31 ONNX vs PyTorch max absolute error was 5.59e-9 (threshold 1e-4). |
| Full package | 61 passed, 3 failed, 1 skipped. Running the same logging tests against files archived from the base commit reproduced all 3 failures (missing `stroke_v22_metric` in the dashboard mapper). The optional W&B import skipped on a missing `click` dependency. |

Local evidence directories are `D:/CapstoneTraining/supermimic_dual_smoke_20261123_01` and `D:/CapstoneTraining/supermimic_dual_graph_20261123_01`. Each contains the training log, resolved configuration, physics receipt, JSONL metrics and checkpoint(s). The eager run also contains `verification.json` (input/config/checkpoint/export SHA-256s and parameter deltas), `verify_run.py`, evaluation JSON, export and pytest logs, and the base logging reproduction. Generated artifacts are local.

| Artifact | SHA-256 |
|---|---|
| Physical tuple bank | `980cc50b9e12e2b37340d80f7f5e51b56ea5754f696cf39bef748ffc0333440a` |
| A3 model XML | `8b06071a1a8ea60f927fa3b146007091e5348b20d979fa9db18568b7f40b4e8f` |
| Eager checkpoint 2 | `8d1397092d6c34fac98167d78045ca603f3e32bb844fbe5518c4f63a80cdc58d` |
| Graph checkpoint 5 | `182043a16318185dbd5bb91ba1a9f9aaf5c7031ff2d2a4410e1e1cb4d4c50c0c` |
| Exported EMA ONNX | `624b9f4a9aee2cc929e60acffbff996c34e4ded09611277c69010c419b3773da` |

The short runs establish execution, shared-model updates and inference parity. Playing strength, differences from a matched single-actor baseline, Gate3 integration and physical robot behavior remain unmeasured.
