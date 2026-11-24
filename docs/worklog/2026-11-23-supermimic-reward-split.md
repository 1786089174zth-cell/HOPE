# 2026-11-23 · SuperMimic branch rewards with a single team critic

Branch `experiment/supermimic-reward-split` branched from the fork's `feature/supermimic-dual-body-actor` at `2f116e718d75b40d8e7ddca39670c2d3e4b97faa`. All training used implementation commit `f7e9d4a498dd578d651dd903ba6f2f20d081b9cb`. This follow-up documentation commit records the comparison.

## Decision and implementation

The coupled upper/lower MLPs act in one full-body dynamics model. The scalar critic fits the original full-team rewards with TD(lambda)/GAE, shared soft survival and one terminal value. The actor objectives discount the other body's local terms inside the imagined horizon by `learn.body_reward_alpha`. At 1 the original single backward pass is used. Below 1 each loss differentiates only its owner's parameters with the upper-to-lower intent connection intact. Both parameter groups are clipped together and stepped once.

Upper-local terms include face alignment, resting left arm and direct face alignment. Lower-local terms include feet, station, heading, hip stance and waist recovery/direct roll. Action first and second differences, the extra hold-phase rate penalty and soft joint limits are partitioned by the actor's A3 motor indices. The original per-joint masks, weights, strike/hold gates and 20 ms time scaling are retained. Racket path/speed, returns, landing, base height, trunk and full-body imitation, table and optional torque/range guards, and mixed waist/arm targets stay shared. See `core/task.py`, `core/learners.py`, `core/nets.py`, the package README and `tests/test_body_rewards.py` under `hope_training/supermimic_hope`.

## Verification before the comparison

- CPU analytic gradients were checked at alpha 0, 0.5 and 1, including a lower action that depends on Upper's current intent.
- The GPU audit replayed the same fixed 8-environment minibatch with identical actor weights, simulator inputs and observation-noise RNG across alphas. It used small networks, zero optimizer learning rate, 4 imagined steps, fractional termination and a mixture of strike/hold gates. The independent reference recomputed per-joint penalties by name, summed the detailed team terms, built branch returns backwards from the shared bootstrap and checked every parameter gradient. GAE targets were independently checked using the full reward and backward TD(lambda) recurrence.
- Acceptance thresholds were gradient absolute/relative tolerances of 2e-5/2e-4 and critic target tolerances of 1e-6/1e-5. Between alphas the targets had to match exactly on the fixed batch, while both parameter groups had to exhibit a gradient difference. All passed.

| Alpha | Max absolute Upper gradient error | Max absolute Lower gradient error | Max absolute critic target error |
|---:|---:|---:|---:|
| 1 | 1.86e-9 | 3.73e-9 | 5.96e-8 |
| 0.5 | 9.31e-10 | 3.73e-9 | 5.96e-8 |

31 focused tests passed including the GPU audit. The ordinary full-suite invocation (GPU audit opt-in unset) produced 64 passes, 3 pre-existing dashboard mapper failures from missing `stroke_v22_metric`, and 2 skips (the GPU audit, run separately, and W&B missing `click`). The parent worklog records the same three failures.

## Matched training and evaluation

Two fresh runs, `split_a100_s17_01` and `split_a050_s17_01`, ran sequentially on one RTX 5080 Laptop GPU, driver 617.42. Python 3.11.15, torch 2.7.0+cu128, MuJoCo 3.5.0, MuJoCo-Warp 3.5.0.2, Warp 1.12.0. All resolved settings matched apart from alpha, run name and output directory. Both used seed 17, 16 environments, full-width dual MLPs, 10-frame histories, 48 warmup ticks, 12 iterations (2 world-model-only and 10 with policy updates), horizon 8 and CUDA graph capture/replay. Hashes of initial actor, critic, dynamics weights, observation histories and simulated state matched exactly. Closed-loop GPU trajectories can diverge.

| Observed metric | Alpha 1 | Alpha 0.5 |
|---|---:|---:|
| Final Upper / Lower gradient norm before joint clipping | 6.7682 / 9.4863 | 6.2154 / 11.1140 |
| Final imagined team / Upper / Lower objective | 0.78488 / 0.78488 / 0.78488 | 0.64575 / 0.65345 / 0.69493 |
| EMA eval stand hits / opportunities and mean position error | 0/22, 0.605 m | 0/21, 0.621 m |
| EMA eval step hits / opportunities and mean position error | 0/23, 0.758 m | 0/17, 0.737 m |
| Falls (termination criterion) / 6,400 scored env-ticks | 62 | 64 |

Both runs exited at the requested cap, logged finite world-model and critic losses, and produced finite nonzero gradients on both branches for every actor update. Evaluation used seed 12345 on both, 16 environments, 200 unscored warmup ticks and 400 scored ticks per environment. Opportunity counts differ because each policy drives its own episode resets. Only a short functional comparison was conducted. These results cannot rank alpha for hit rate, legal returns or robot performance.

## Commands and evidence

From `hope_training/supermimic_hope` in PowerShell:

```powershell
$py = 'D:/APP/miniconda/envs/isaaclab/python.exe'
$env:HOPE_SUPERMIMIC_DEPS = 'D:/CapstoneTraining/supermimic_fork_deps_py311_20261123'
$env:PYTHONPATH = $env:HOPE_SUPERMIMIC_DEPS
$evidence = 'D:/CapstoneTraining/supermimic_reward_split_20261123_01'
$env:SUPERMIMIC_GPU_AUDIT = '1'
& $py -m pytest -q -s tests/test_body_rewards.py tests/test_dual_body_actor.py tests/test_core.py tests/test_four_stroke_port.py tests/test_a3.py
Remove-Item Env:SUPERMIMIC_GPU_AUDIT
& $py -m pytest -q -rs --tb=line tests
$common = 'actor_architecture=dual_body_mlp_v1 num_envs=16 seed=17 device=cuda:0 max_iterations=12 wm_only_iterations=2 warmup_steps=48 buffer_steps=96 learn.wm_batches=1 learn.wm_batch=16 learn.wm_window=8 learn.batch=16 learn.horizon=8 learn.critic_minibatches=2 learn.critic_epochs=1 capture=true log_interval=1 save_interval=12'.Split(' ')
foreach ($case in @(@('a100','1.0'), @('a050','0.5'))) {
    $tag, $alpha = $case
    & $py -u "$evidence/train_with_receipt.py" @common "learn.body_reward_alpha=$alpha" "run_name=split_${tag}_s17_01" "out=$evidence/$tag"
}
foreach ($tag in @('a100','a050')) {
    & $py -u "$evidence/score_with_counts.py" "$evidence/$tag/model_12.pt" num_envs=16 seed=12345 steps=400 device=cuda:0 contact=elliptic "out=$evidence/$tag/evaluation.json"
}
& $py "$evidence/compare.py" 'C:/Users/ROG/Desktop/UCB/Capstone/HOPE-supermimic'
```

The local wrappers call the normal `train.py` and `evaluate.py` entry points. The first records initialization digests in a fresh output directory and the second preserves per-family event counts before the evaluator clears them. Use fresh output directories for a repeat. `D:/CapstoneTraining/supermimic_reward_split_20261123_01/comparison.json` contains per-family results and full asset, config, metrics, checkpoint, and evaluation hashes. Each arm directory holds `config.json`, `initial.json`, `metrics.jsonl`, `physics_receipt.json`, `model_12.pt` and `evaluation.json`. The root holds the scripts, audit/test logs and per-run output.

| SHA-256 | Digest |
|---|---|
| Initial actor weights (both) | `e171e2ac07287b980f8511332a60821b6900f18c1efd59f56a879fd5c54ad51e` |
| Initial critic weights (both) | `bf4782deeb18bb8f1d0f71b9a06949b5e0db3ed37895961f94e9942a650a148a` |
| Initial dynamics weights (both) | `12898dbf5f433bfef5f2f6cb7b07fa804e51eef050a0a3239d304d197faeeae4` |
| A3 MJCF | `8b06071a1a8ea60f927fa3b146007091e5348b20d979fa9db18568b7f40b4e8f` |
| Physical tuple bank | `980cc50b9e12e2b37340d80f7f5e51b56ea5754f696cf39bef748ffc0333440a` |
| Alpha 1 checkpoint | `0cb9e564ac56a485b9a41dc75ccfa33b6116b62bff762fb7b7e901fd506aeea9` |
| Alpha 0.5 checkpoint | `a95a30c41ca561299c7f9123be7e47dca687518dd95653ca3ef4de7e5fc29105` |
