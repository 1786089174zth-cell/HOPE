# 2026-11-23 · Paired SuperMimic hitting experiment

Run ID `supermimic_reward_paired_20261123_01` uses branch `experiment/supermimic-reward-split`. The starting tree was clean at `f3b4e809ab084d05b7019b811785c4aa9354086d`. Each run starts from fresh weights. The training code, two MLPs, full-body dynamics and scalar team critic are the ones at that commit.

## Prespecified comparison

Three training seeds (17, 29, 43) each receive both reward weights (1.0 and 0.5). Run order is 1/.5, .5/1, 1/.5. Each arm gets 256 environments, 96 warm-up ticks, 3,000 iterations including 150 dynamics-only iterations, 2,850 actor updates, and 1,944,576 training robot ticks including warm-up. The world model uses 3 batches of 256 with window 8. The policy uses batch 256 and imagined horizon 16. The critic uses 2 epochs and 4 minibatches. Full resolved configs and initial parameter/state digests are retained per arm.

Every 500 iterations, evaluate the EMA policy with held-out seed 12345, 128 environments, 200 warm-up and 500 scored ticks. Use elliptic contact, zero action noise, ordinary sensor noise, no startup handoff and no mid-stroke starts. Each checkpoint gets a fresh evaluator. All six runs continue to the fixed budget regardless of their scores.

- Pool forehand/backhand counts separately within stand and step. A hit requires simultaneous racket position error below 7.5 cm, speed-vector error below 0.5 m/s and face-normal error below 15 degrees at frame 60. The denominator is nonterminated strike ticks. Average Euclidean racket-position error over the same opportunities.
- Charge a fall to the active motion that generated the action immediately before termination. Both startup waiting and interstroke hold have their own exposure bucket. Report falls per 1,000 robot ticks in each bucket. Timeout resets are excluded. Falls use the simulator's reference-relative root-height/tilt criterion.
- Each motion independently qualifies with at least 20% hits on at least 100 strike opportunities at two consecutive checkpoints. Record the first qualifying checkpoint in the pair and its confirmation. Runs with no confirmed pair are censored at the budget. Equal-weight seed means, paired seed differences and the six-point hit-rate curves summarize the comparison. Elapsed wall time is machine-load dependent.
- Source files, the full protocol, all checkpoints, logs and scorer outputs live under `D:/CapstoneTraining/supermimic_reward_paired_20261123_01`. The first six-row protocol was written before the first run. The local wrapper also checks that summed stand/step hit counts and position errors match the standard evaluator and that stand/step/hold recover all falls and ticks.

## Commands

From `hope_training/supermimic_hope` on the Windows GPU host:

```powershell
$py = 'D:/APP/miniconda/envs/isaaclab/python.exe'
$env:HOPE_SUPERMIMIC_DEPS = 'D:/CapstoneTraining/supermimic_fork_deps_py311_20261123'
$env:PYTHONPATH = $env:HOPE_SUPERMIMIC_DEPS
$exp = 'D:/CapstoneTraining/supermimic_reward_paired_20261123_01'
& $py -u "$exp/run_paired.py"
& $py -u "$exp/score_paired.py"
& $py "$exp/analyze.py"
& $py "$exp/plot_curves.py" 'D:/CapstoneTraining/training_plot_deps'
```

`protocol.json` contains every training command. `score_paired.py` scores atomically saved checkpoints and may run concurrently. A repeat needs a fresh evidence directory and the helper scripts present there.

## Run status

Training and scoring are in progress. Results and hashes will be recorded after all six runs and 36 evaluations.
