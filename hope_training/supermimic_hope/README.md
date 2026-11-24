# HOPE four-stroke SuperMimic

This HOPE-fork copy comes from `dongc1/nohope` commit `9f76d2d1fe688b09e786f4c92815ce0b686cce1d`
(`supersuper_1`). Its active training entry is `train.py` with `supermimic_hope/core/`.
The default four-stroke question bank is tracked under `../whole_body_tracking/assets/`;
its physical tuple bank uses Git LFS. Training logs and checkpoints are not included.
The `grasping` container and `/home/dongc1/...` paths below describe the source workstation.

SuperMimic ([arXiv 2610.28480](https://arxiv.org/abs/2610.28480), code
[dongc1/supermimic](https://github.com/dongc1/supermimic), Apache-2.0) for the A3 ping-pong
policy: collect transitions, fit one differentiable dynamics model on them, and improve the
policy by backpropagation through 32 imagined steps with a critic bootstrap. Licenses of the
code this follows are in `licenses/`.

There are two implementations in this directory:

| | `supermimic_hope/core/` (use this) | `supermimic_hope/analytic/`, `trainer_analytic.py` |
|---|---|---|
| entry | `train.py` | `whole_body_tracking/scripts/train_supermimic.py` |
| simulator | MuJoCo-Warp only, no Isaac/Kit | Isaac Lab managers on MuJoCo-Warp physics |
| task | standalone four-stroke rally from the task's data assets; paper reward | the PPO task's 40 reward terms restated analytically |
| speed (RTX 4090, 4096 envs) | 1.9 iterations/s | 0.46 iterations/s |
| policy input | 10 consecutive 110-D `hitter_pure` frames | one 110-D frame |

## Standalone trainer (`core/`)

### What it does

- **Physics** (`core/sim.py`): the Gate3 A3 MJCF (`a3.py`: balls, table and vendor sole spheres
  removed, PD position actuators, 4 x 5 ms implicit-fast steps per control tick) on MuJoCo-Warp,
  4096 worlds, one captured Warp graph per tick (14 ms). Startup randomization: foot friction
  0.3-1.2, torso COM, total mass +-15%, per-joint kp and message-kd x0.85-1.15 (log-uniform);
  25% of the worlds keep nominal mass, COM and gains. These 67 numbers are the world model's
  sim context.
- **Task** (`core/task.py`): questions are the qualified rows of the physical tuple bank
  (`assets/wide_stroke_v26_urdf2_plane058_20261110`), 25% per family (stand/step x
  forehand/backhand), 35% from the deploy-plane slab x 0.50-0.58 m. The reference is the family
  clip under the task's preparation time-warp with the row's right-arm adapter blended in around
  contact; at the strike frame it places the racket on the target within 1 cm, 0.012 m/s and
  4 deg. Rally clock: startup HOLD, announcement, strike at clip frame 60, recovery, next ball
  from +0.5 s. Observation: the deployed 110-D `hitter_pure` frame with gyro / joint / mocap /
  target noise; action: `q_des = q0 + scale * a`, clamped to the soft limits, head passive,
  optional one-tick command FIFO.
- **Reward and termination** (paper Table A.2 and Appendix A): BeyondMimic tracking of the
  reference (root position and orientation, 14 tracked bodies' position, orientation, linear and
  angular velocity), action rate -0.1, joint limit -2, plus the interface's own task reward:
  racket position on the moving contact path, racket velocity and face normal inside
  +-0.12 s of the strike. Termination: root height more than 0.25 m or tilt (gravity z) more
  than 0.8 from the reference; soft in imagination.
- **World model** (`core/dynamics.py`): MLP 2048x3 (LayerNorm, GELU) on the heading-free state,
  the executed PD targets and the sim context; predicts acceleration-level deltas that are
  integrated analytically; fit on 8-step autoregressive windows with Bryson-rule weights.
- **Policy** (`core/learners.py`): actor MLP 2048-1024-512-256 on 10 frames, critic 256-128 on
  privileged state; BPTT over 32 imagined steps from replay states along the logged task tape,
  soft survival weight, critic bootstrap, GAE(0.95) critic targets from the imagined rollout,
  Adam 2e-4 (actor betas 0.9/0.95), gradient clip 1.0 then 0.1 element-wise, UTD 0.4, action
  noise 0.1 in collection. Both updates are recorded once as CUDA graphs and replayed.

Differences from the Isaac PPO task are listed in
[docs/core_20261116.md](docs/core_20261116.md).

### Run

Dependencies: PyTorch with CUDA plus the pinned `requirements.txt` (MuJoCo 3.5.0,
MuJoCo-Warp 3.5.0.2, Warp 1.12.0). On this workstation they are in the `grasping` container
(`/opt/drone_venv`, deps in `~/.local/share/hope-supermimic/deps`); no Isaac launch is needed.

```bash
distrobox enter grasping
cd /home/dongc1/workspace/nohope/hope_training/supermimic_hope
CUDA_VISIBLE_DEVICES=1 /opt/drone_venv/bin/python train.py \
  device=cuda:0 num_envs=4096 learn.batch=2048 \
  max_iterations=30000 run_name=four_stroke_core
```

Outputs: `logs/supermimic/<time>_<run_name>/` with `metrics.jsonl`, TensorBoard, `config.json`,
`physics_receipt.json` and `model_<N>.pt` every 1000 iterations. Any setting is `key=value`
(`learn.horizon=32`, `task.plane_probability=0.35`, `sim.friction=[0.3,1.2]`, ...); resume with
`resume=<run>/model_<N>.pt` (weights and optimizers; the replay refills). 30000 iterations take
about 4.4 h and 17 GiB of GPU memory. In a 6000-iteration test (53 min of training, one seed)
the averaged actor reached a deterministic hit rate of 0.78 (0.74-0.83 per family) with no
falls; nothing longer has been run yet.
`learn.batch=4096` is the paper's value and needs more than 24 GiB next to the 1000-step replay.

```bash
/opt/drone_venv/bin/python evaluate.py <run>/model_<N>.pt device=cuda:0 num_envs=1024 steps=3000
/opt/drone_venv/bin/python export.py <run>/model_<N>.pt        # exported/policy.onnx
/opt/drone_venv/bin/python play.py <run>/model_<N>.pt device=cuda:0            # play_<N>.mp4
/opt/drone_venv/bin/python play.py <run>/model_<N>.pt device=cuda:0 viewer=1   # MuJoCo window
CUDA_VISIBLE_DEVICES= /opt/drone_venv/bin/python -m pytest -q tests/test_core.py
```

Gate3 bundle (`exported/policy.onnx` + `params/deploy.yaml`) and the state of Gate3 for this policy:
`tools/build_gate3_bundle.py`, `tools/check_gate3_bundle.py`, `docs/gate3_20261118.md`.

Current best checkpoint, what each HOPE-specific reward and question setting is for, and the
measurements behind them (`tools/audit_motion.py`, `tools/gate3/summarize.py`, `tools/gate3/posture.py`):
`docs/finetune_20261118.md`.

Policy entry (the Runner's 3 s stand-to-policy handoff and the non-creeping foot contact, both in
the training simulator since 2026-11-18) and the tools that measure it (`tools/audit_entry.py`,
`tools/entry_replica.py`): `docs/entry_20261118.md`. Evaluation tools default to the earlier physics
(`contact=pyramidal handoff=0`) so older numbers stay comparable; pass `contact=elliptic handoff=1`
for the simulator checkpoints are now trained in.

Ball paths (a Gate3 scene with low/mid/high and slow/medium/fast serves, `tools/gate3/make_scenario.py`,
`tools/gate3/by_serve.py`) and the fast-ball experiment: `docs/ball_paths_20261118.md`. Foot tilt, station drift,
table-edge clearance and the forehand/backhand boundary ball in long rallies, with the tools that measure them
(`tools/gate3/feet.py`, `station.py`, `strike_points.py`): `docs/rally_20261118.md`. The reward and question options
added there were off by default until the next step.

Returns (why the ball missed the Planner's landing target, the landing reward, the forehand/backhand boundary, the
0.58 m strike plane; `tools/gate3/returns.py`, `tools/gate3/shift_strike_plane.py`): `docs/returns_20261119.md`.

Supergate (offline batch evaluation with main's kind of Planner: flat strokes announced 0.22-0.40 s ahead), where
the policy loses on it, the training questions made from its Planner (`tools/supergate/`, `task.extra_bank`, direct-stroke
reference) and the mocap dropouts in training: `docs/supergate_20261119.md`. Score a checkpoint with
`tools/supergate/score.sh <model_N.pt>` (robot 0.58 m behind the table edge, the only geometry we test in).
Current best checkpoint: `logs/supermimic/2026-11-19_18-58-17_ft17_station058/model_59000.pt`, for the robot standing
0.58 m behind the table edge (strike on the edge; Gate3 scene `tools/gate3/scenarios/fixed-home-station058.yaml`,
bundle built with `tools/build_gate3_bundle.py --strike-plane 0.58`); `TaskCfg`'s defaults are its recipe.
Gate3 counts are re-judged for a ball-report tolerance artifact (`tools/gate3/rejudge.py`).

`export.py` writes an ONNX with `obs` [1, 1100] (10 frames, oldest first), metadata
`obs_frame_history=10` and the action contract. It does not write the full `hitter_pure`
metadata or `deploy.yaml` of a Gate3 bundle.

**Use the averaged actor.** Every checkpoint holds the raw iterate (`actor`) and a Polyak
average of the actor weights (`actor_ema`, decay 0.99 per update). Consecutive raw iterates
swing by tens of points of hit rate (checkpoints 100 iterations apart: 0.05, 0.15, 0.53,
0.03), which the 50-iteration training averages hide; the averaged actor scored 0.48-0.52 on
the same checkpoints. `evaluate.py` and `export.py` use the average by default (`ema=0` /
`raw` selects the iterate). `task/hit_rate` in the training log is the raw policy under
collection noise, so judge a run by `evaluate.py` (1024 envs x 3000 steps, normalizers frozen,
no mid-stroke starts; a second evaluation seed moves the hit rate by about 0.01).

**Hitting regression check.** Before adopting any change to the task, reward or learner, train
both versions to the same iteration, score both with the same `evaluate.py` settings, and run

```bash
/opt/drone_venv/bin/python tools/check_hitting.py <candidate evaluation.json> <baseline evaluation.json>
```

It fails (exit 1) when any hit metric, overall or per family, is more than 0.05 below the
baseline, a strike error is more than 15% above it, or the fall rate rises. The 6000-iteration
reference is `docs/baselines/hitting_6000_seed0.json`.

Key metrics: `task/hit_rate` (racket within 7.5 cm, 0.5 m/s and 15 deg of the target at the
strike tick), `<family>/hit_rate`, `task/falls_per_step`, `collect/reward_per_s` (maximum 10),
`wm/k_step_*` (held-out 8-step model error).

## Isaac-manager variant (v4)

### How it works (v4, `supermimic.algorithm=analytic`)

- **Physics**: Isaac Lab/Kit builds the task, assets and startup randomization;
  `WarpBackend` then supplies all dynamics, state writes and contacts. Physics
  parity with PhysX and hardware behavior are `NOT_PROVEN`.
- **State**: MuJoCo's `qpos`/`qvel` (75-D, Isaac joint order). Within each
  `env.step` three instants are recorded: the post-reset state the command caches
  (racket pose and velocity), the post-push state observations see, and the
  post-physics pre-reset state that rewards and terminations read.
- **World model** (`analytic/world_model.py`): heading-invariant state features
  plus the executed PD targets of this and the previous tick, the episode's
  command latency and the environment's startup DR, predicting frame-local
  increments; fit on 8-step autoregressive windows (upstream design).
- **Analytic heads** (`analytic/task.py`): a differentiable torch FK of the
  Gate3 MJCF, 40 of the task's 42 weighted reward terms, its five failure
  margins and the 110-D actor observation. Everything exogenous (clocks,
  references, gates, curriculum scales, contact load, latency, mocap error) is
  recorded from the unmodified task at the exact instant it is used. HOLD dwell,
  the q_des delay queue, the previous raw action and the previous executed
  q_des increment are advanced inside imagination. A recorded
  per-step correction (real reward minus analytic reward at the real state)
  covers `undesired_contacts`, `joint_acc` and the PD-torque approximation.
- **Learning** (`analytic/learner.py`): BPTT over 32 imagined steps from replay
  states with soft (sigmoid) terminations and critic bootstraps; the critic is
  fit on imagined lambda-returns (5 epochs x 4 minibatches); Adam 2e-4 with
  cosine decay, gradient clip 1.0 / 0.1; collection noise 0.1; UTD 0.4. A
  sensor-noise gain penalty (`supermimic.sensor_noise`, `analytic/noise.py`) is
  added to the actor loss.
- **Schedule clock** (`supermimic.schedule_clock_scale: 5`): the task's reward
  ramps, DR ramps and question-coverage clock count environment steps and were
  calibrated for PPO (64 steps per update). One iteration here is 2.5 steps, so
  those lengths are divided by 5 at startup (`schedule_clock.json` in the run
  directory lists every value). Evidence windows and physical durations are
  unchanged.
- **Resets**: the SuperMimic task YAML bounds the reference-state root velocity
  noise at the push-DR scale; with the tracking default, about half of those
  starts fell within 2 s at every checkpoint (v4 notes, section 15).
- **Row-bounded actor gradient** (HOPE-specific, `action_grad_row_clip: 4.0`):
  the gradient reaching each imagined row's action, per step, is capped at 4x
  that step's batch median. Without it a few rows with steep critic gradients
  at the horizon carried most of the update. On the full objective it beat
  no cap and the critic-input cap (`critic_grad_row_clip`, off by default)
  with equal falls (v4 notes, sections 7-10).
- **Speed**: heads, observation features and the world model are
  `torch.compile`d. Compiled backward graphs can produce `0 * inf` where eager
  PyTorch returns zero, so gradients are made finite at the compiled boundary;
  the compiled and eager actor gradients agree to cosine 0.99998.

### Checks you can run

- `supermimic.analytic_check_steps=N` (optionally `analytic_check_noise`,
  `analytic_check_policy=<checkpoint>`): no training; compares every analytic
  reward term, the five failure margins, FK and observations with the real task
  for N steps and writes `analytic_check.json`.
- During training, `Imagination/check_*` replays recorded actions through the
  imagination code with real states; reward, executed targets and the state
  chain must match exactly (they do, to 2e-6).
- `+supermimic.debug_compare_gradients=true`: logs the cosine between compiled
  and eager actor gradients at the first policy updates.
- `+supermimic.diagnose_checkpoint=<run>/model_<N>.pt` (no training; use
  ~1024 envs): rolls the checkpoint in the real task and writes
  `diagnosis.json` with per-joint HOLD/stroke errors per phase and
  time-to-strike, action saturation, the actor gradient split by reward term /
  termination / critic bootstrap, per-row gradient concentration, critic
  value and gradient checks, and batch-to-batch gradient agreement per row-clip
  factor.

### Run on this workstation

Enter the `grasping` container (`distrobox enter grasping`). Tested with Isaac
Sim 5.1, Python 3.11, Torch 2.7 CUDA 12.8, MuJoCo 3.5.0, MuJoCo-Warp 3.5.0.2
and Warp 1.12.0 on an RTX 4090 (24 GiB).

One-time setup after checking out the branch (from the repository root):

```bash
git lfs install && git lfs pull
python3 hope_training/whole_body_tracking/scripts/prepare_a3_isaac_asset.py --force
cd hope_training/whole_body_tracking
source setup_train_env.sh
export HOPE_SUPERMIMIC_DEPS="$HOME/.local/share/hope-supermimic/deps"
hope_isaac_py -m pip install --no-deps --target "$HOPE_SUPERMIMIC_DEPS" \
  -r ../supermimic_hope/requirements.txt
```

The pinned dependencies stay outside the Isaac/PPO environment. The prepared A3
URDF and meshes (`assets/agibot_a3/`), generated USD, logs and checkpoints are
git-ignored. The Gate3 MJCF, the four motion clips, the reference adapter, the
HOLD reference and the question bank are repository inputs; the MJCF hash is
checked at startup.

```bash
cd hope_training/whole_body_tracking
source setup_train_env.sh
hope_isaac_py scripts/train_supermimic.py \
  task=HOPEPingPongMotionQualitySuperMimic algo=ppo_motion_quality \
  headless=true device=cuda:1 num_envs=6144 seed=0 \
  max_iterations=60000 \
  supermimic.policy_batch=4096 supermimic.replay_device=cuda:1 \
  checkpoint_path=null warm_start_actor_checkpoint=null \
  run_name=v26_supermimic_clock5
```

At 6144 environments one iteration takes ~2.1 s on the RTX 4090 (collection of
2.5 vector steps 0.6 s, world model 0.3 s, policy 1.3 s) and 14 GiB of device
memory with the upstream policy batch of 4096 and the replay on the GPU. The
YAML defaults (`policy_batch: 2048`, `replay_device: cpu`) fit a 12 GiB GPU at
the same speed (7 GiB here; the replay is 6.3 GiB of pinned host memory). The
first world-model and policy updates compile for ~15 s and ~3 min. Iterations
are SuperMimic iterations (2.5 real steps each): 60000 iterations collect 150k
vector steps in ~36 h, and the scaled task schedules finish at iteration ~23k.
For a small smoke test use `num_envs=256 max_iterations=12
supermimic.batch_size=256 supermimic.policy_batch=128 supermimic.horizon=8
supermimic.warmup_steps=16 supermimic.buffer_steps=32
supermimic.wm_only_iterations=3`.

Outputs go to `logs/supermimic/<timestamp>_<run_name>/` (resolved configs,
physics receipt, JSONL/TensorBoard, checkpoints, status). W&B is on by default
(`WANDB_PROJECT`/`WANDB_ENTITY` from the sourced environment; run `wandb login`
once on a new machine); use `supermimic.wandb.mode=offline` or
`supermimic.wandb.enabled=false` to keep runs local. Task dashboards (`StandFH`, `StandBH`, `StepFH`, `StepBH`, `Return`,
`Strike`, `Home`, `HOLD`, `RewardStep`, `Safety`) are unchanged; v4 adds
`WorldModel/validation_k_step_*` (physical units), `Imagination/*`,
`Performance/imagine_*` and `Replay/correction_abs_mean`.

To continue a run, pass `supermimic.resume_checkpoint=<run>/model_<N>.pt`:
weights, normalizers, optimizers and the cumulative environment step counter
resume at iteration N (the task's reward ramps read that counter; checkpoints
saved before it was stored are estimated from N). Replay and per-command
curriculum state restart; the replay refills in the first 64 steps.

`supermimic.buffer_steps` is the replay length in env steps (upstream 1000).
The default 400 steps are 6.3 GiB at 6144 envs (about 16 MiB per step);
`supermimic.replay_device=cpu` keeps the tape in pinned host memory at the
price of a host gather per sampled batch.

`+supermimic.evaluate_checkpoints=[a.pt,b.pt,...]` (no training) runs each
checkpoint from the same seeded reset with identical reward ramps and writes
`evaluation.json` (reward, falls, HOLD error, key terms, by episode age).

Checkpoints hold `model_state_dict` in the `train.py` actor-critic layout (the
actor is the trained policy; its PPO critic is unused), plus the world model,
the imagination critic and optimizers. This entry does not build or certify a
deploy bundle.
