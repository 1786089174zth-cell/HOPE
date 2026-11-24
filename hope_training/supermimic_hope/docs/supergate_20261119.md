# Supergate: where the four-stroke policy loses, and training on its kind of ball (2026-11-19)

Supergate (`origin/codex/supergate-tools-20261113`, `a3_deploy/a3_deploy_example/scripts/supergate`) is an offline
batch evaluation: frozen physical launches, the Runner's native policy code, the Gate3 ball-contact model and a
pinned offline Planner, all in one process on MuJoCo's batch pool. Its main suite has 1200 launches (81 cells of
lateral position x height x incoming speed x horizontal angle, 12 each, plus 228 "origin-boundary" launches);
every ball has spin (about 100 rad/s of topspin at contact). It is deterministic: the same policy scores the same
count twice.

## 1. Running it here

| step | command |
| --- | --- |
| build a tree that accepts four-stroke, 10-frame bundles | `tools/supergate/assemble.sh logs/supergate/tree` (hope container, repository root) |
| ready state (official stand, settled 1 s) | `.venv/bin/python hope_training/supermimic_hope/tools/supergate/make_ready_state.py --tree <tree> --out logs/supergate/ready_stand.npz` |
| score a checkpoint | `tools/supergate/score.sh <model_N.pt> [suite] [threads] [home_x] [label=bundle ...]` (host shell) |
| break a run down | `python3 tools/supergate/report.py <output dir> [--dataset <suite dir>]` |

`assemble.sh` copies the branch's Supergate tools, Runner source and MuJoCo module into the tree, applies the two
Runner patches the merged Gate3 runner uses (frame history, four strokes; `admission.patch` is the admission hunk
ported to that branch), and builds `native_batch.so` with the repository venv's MuJoCo. Nothing in the working tree
changes. `SUPERGATE_HOME_X=-0.58` (the default of `score.sh`) runs the suite with the robot where our deploy
geometry has it, 0.58 m behind the table edge with the strike plane on the edge. **Since 2026-11-19 that is the only
geometry we test in**; the protocol's own HOME is 0.50 m behind the edge, and the first numbers below were taken there.

Protocol facts that matter for the policy: HOME (-0.5, -0.7625), strike plane = HOME + the bundle's strike-box centre
(0.58 m, world x 0.08: 8 cm over the table), the Planner commits 0.35 s before the strike or as soon as it can
(notice 0.15-0.40 s), side split at -0.24 m with 0.04 m hysteresis, 20 s of standing before the first ball, a ball
about every 2.6 s, no entry handoff.

## 2. Baseline (main suite, 1200 launches)

| policy | legal returns | racket contact | valid plan |
| --- | --- | --- | --- |
| SuperMimic model_59000 (ft17) | 583 (48.6%) | 800 | 1200 |
| SuperMimic model_54000 (ft15) | 553 (46.1%) | 728 | 1200 |
| SuperMimic model_42000 (ft5) | 488 (40.7%) | 898 | 1164 |
| four-stroke PPO model_11999 | 569 (47.4%) | 750 | 1172 |
| model_59000, robot at -0.58 (our geometry) | 681 (56.8%) | 858 | 1198 |

Mixed-long suite (1200 launches, a third of them outside the Planner's admission): model_59000 390 (32.5%),
PPO 384 (32.0%), valid plans 798 and 790.

Where model_59000 loses (protocol HOME), legal returns:

| | low 0.8-1.0 m | mid 1.0-1.2 m | high 1.2-1.45 m |
| --- | --- | --- | --- |
| forehand, notice < 0.30 s | 9 / 77 | 7 / 76 | 0 / 66 |
| forehand, 0.30-0.36 s | 14 / 21 | 15 / 34 | 2 / 45 |
| forehand, 0.36-0.42 s | 63 / 92 | 59 / 88 | 6 / 91 |
| backhand, notice < 0.30 s | 36 / 74 | 39 / 77 | 1 / 70 |
| backhand, 0.30-0.36 s | 28 / 28 | 32 / 32 | 16 / 42 |
| backhand, 0.36-0.42 s | 89 / 93 | 94 / 97 | 73 / 97 |

* The backhand with at least 0.30 s of notice and a target below 1.2 m is solved (243 of 250).
* The forehand is the large loss (175 of 590). With less than 0.30 s the racket is still in the clip's backswing at
  the strike: 0.2-0.5 m/s forward for 1.0-1.5 m/s commanded, face 45-58 deg off.
* High balls: the racket is 9-13 cm below the target on either side (contact on 11-21% of them at short notice).
* By ball: fast (> 4 m/s) 16%, medium 58%, slow 71%; rising at contact 35%, falling 76%.

What main publishes for its own policies on the same main suite (`research/model400/20261118-followup`,
`research/model23700/20261118-supergate` on `origin/main`, protocol HOME): model400 v4 998 / 1200 (v3 978, v2 973),
model23700 790 / 1200. Their racket at the planned instant is 6-7 cm, 0.3 m/s and 7-9 deg from the target; ours is
15 cm, 1.0 m/s and 29 deg.

### Why this is not only about Supergate (main's Planner, 81e606a0c)

Our Gate3 binaries are main 7990f18c2 (2026-11-04) with a 0.65 s packetizer lead. Since then main's Planner
(53c74a909, 2ec307bed) solves the racket target with the full return physics (spin, tangential paddle gain, net
clearance; `racket_target_planner.cpp`, `config/return_physics.yaml`) and its default packetizer
(`model21800_flight_packetizer.yaml`) commits 0.35 s ahead; `four_stroke_lead065_flight_packetizer.yaml` does not
exist there. That is the Planner Supergate pins: the flat, slow, late-announced strokes of section 3 are what main
asks of any fixed-HOME policy, in Gate3 and on the robot. Porting our Gate3 stack to main's Runner and Planner is
in progress (`tools/gate3/build_main_stack.sh`).

## 3. Why: Supergate's questions are outside the training bank

`tools/supergate/targets.py <run>` puts the Planner's committed targets next to the bank's qualified rows
(quantiles 1 / 10 / 50 / 90 / 99%, station axes):

| | Supergate | bank (forehand rows) |
| --- | --- | --- |
| notice | 0.22 / 0.24 / 0.35 / 0.40 / 0.40 s | 0.50 / 0.55 / 0.69 / 0.96 / 1.14 s |
| racket speed | 0.84 / 1.13 / 1.72 / 2.27 / 2.44 m/s | 1.78 / 2.29 / 2.53 / 2.82 / 3.25 m/s |
| racket elevation | -16 / -11 / +4 / +19 / +25 deg | +18 / +25 / +32 / +37 / +46 deg |
| target height | 0.83 / 0.89 / 1.10 / 1.34 / 1.42 m | 0.83 / 0.87 / 0.95 / 1.05 / 1.14 m |
| ball speed in | 1.8 / 2.1 / 3.3 / 4.8 / 5.1 m/s | 1.1 / 1.6 / 2.3 / 2.9 / 3.5 m/s |

Every Supergate question is announced later than any bank question, asks for a flat, slower stroke (a block or a
drive against topspin; the bank lifts the ball), and a third are higher than the bank's 99th percentile.

## 4. What was added to training

1. **Fresh launches over the suite's ranges** (`tools/supergate/make_launches.py`). The frozen suites stay the
   test. Each launch of `main` and `outer` is displaced (origin +-0.15/0.12/0.06 m, velocity +-7% and 0.2 m/s, spin
   +-5 rad/s), flown with Supergate's own flight and bounce model (the vectorised copy reproduces the ledger's
   predicted contacts to 1e-11) and kept when it is a legal one-bounce ball inside the suite's envelope: 9600
   launches. model_59000 returns 49.2% of them, against 48.6% on the frozen suite: the same kind of ball.
2. **The Planner's own targets for them**: the 9600 launches run through Supergate (four frozen suites of 2400)
   give the committed racket target, the ball velocity at the strike and the notice of each.
3. **A strike pose per question** (`tools/supergate/build_questions.py` -> `assets/supergate_questions_s1.npz`,
   9303 rows kept of 9595). The bank's reference is a clip plus an arm correction at the clip's strike frame; with
   the torso where the stand clips have it the Supergate backhand targets are out of the arm's reach (median
   residual 18 cm, none within 3 cm), so the pose is solved from the HOLD stance with the waist and the right arm
   (residual median 0.02 cm, 90th percentile 1.6 cm and 4.4 deg; waist offsets up to 0.4 rad of yaw).
4. **A direct-stroke reference** (`core/task.py`, `_direct`, `task.extra_bank`, `task.extra_probability`): for these
   questions the reference goes from HOLD to the strike pose on a cubic over the notice, arriving with the strike
   velocity, and back to HOLD in 0.6 s; legs and root keep the HOLD reference. No backswing. The questions are scored
   in the `rush` metrics (`task/rush_hit_rate`); `evaluate.py ... extra=1` scores only them.
5. **Table edge per question**: these questions take the keep-out edge 0.50 m ahead of the station (Supergate's
   table), the bank questions keep `table_edge_x`.

In the simulator model_59000 misses these questions the way it does in Supergate: racket 15.5 cm from the target
and 1.0 m/s off at the strike tick (Supergate, at the planned instant: 15.0 cm, 0.98 m/s).

6. **Second question file at our geometry**: `assets/supergate_questions_s2_home058.npz` (9339 rows), from 9600 other
   launches (seed 2) run with `SUPERGATE_HOME_X=-0.58`; model_59000 returns 55.3% of them (56.8% on the main suite
   at that HOME). `task.extra_bank` takes several files, comma-separated.

## 5. Mocap dropouts (`task.mocap_dropout_rate`, `task.mocap_dropout_s`)

On the robot the mocap link occasionally drops, for up to about a second. What the Runner does (main 81e606a0c,
`pp_planner_input.hpp`: `kBasePoseHoldAfterS` 0.02 s, `kBasePoseRescueAfterS` 0.5 s, `PpBasePoseHold`;
`pp_runner_control.hpp`, `ObserveMocap`): once the newest base sample is 20 ms old the last applied base position and
attitude are held, without gyro propagation; the swing in flight continues, new flights are refused; at 0.5 s the
Runner leaves MOTION for the PD stand. The Runner our binaries were built from held for 1.0 s. In the 110-D frame
gravity, heading, base position and the target relative to the base come from that pose, so they freeze; gyro,
joints and the strike clock stay live.

Training now does the same: at 0.05 per second an environment's base pose freezes in the observation for
U(0.06, 1.0) s (the pose is logged on the tape, so imagined rollouts see the same frozen frame). `evaluate.py ...
mocap_dropout=<rate>` scores with it; it is 0 unless asked for, like the pushes.

## 6. From fitting Supergate's envelope to general balls

The questions of section 4 came from displaced Supergate launches: the same envelope as the test. The training
set is now independent of every test suite (`tools/supergate/make_launches.py --general`): origins anywhere over or
just beyond the far half, aimed at a random point of the strike plane with random speed, loft and spin, kept when
they are legal one-bounce balls crossing the plane within reach (lateral -0.70..+0.46 m, 0.80-1.47 m high, 1.5-5.6
m/s, within 24 deg), and drawn evenly over 350 cells of lateral position x height x speed x rising/falling, so a fast
high ball is as frequent as a slow middle one. `assets/general_questions_g1_home058.npz`: 9730 questions from 12000
such launches (seed 11) with the Planner's targets at HOME -0.58. Tests, none of them trained on: the field ball
paths (`docs/field_ball_paths_20261119.md`), Supergate's main suite, and 1200 general launches of another seed
(`logs/supergate/suite_general_test`).

The Supergate harness commits nothing for a ball outside its own lateral bands (forehand [-0.68, -0.24], backhand
[-0.24, 0.28] m); main's Planner takes the bands from the bundle (ours reach +0.46 m). `tools/supergate/bands.py`
makes them overridable and `SUPERGATE_BANDS=bundle tools/supergate/score.sh ...` runs a suite with the bundle's.

## 7. What the reference alone did not do (ft19) and the three direct terms

ft19 (model_59000 + 40% section-4 questions, reference only) after 1000 and 2000 iterations, on questions of the
other file, per strike (`diag`: forehand / backhand, notice 0.36-0.42 s):

| | racket to target | speed along the commanded direction | face off (median) |
| --- | --- | --- | --- |
| model_59000 | 14.8 / 9.4 cm | 1.87 of 2.21 / 1.96 of 2.13 m/s | 140 / 12 deg (61% of forehands with the other side of the blade) |
| ft19 60000 | 6.6 / 4.5 cm | 1.09 / 1.79 | 46 / 26 deg |
| ft19 61000 | 5.5 / 4.2 cm | 1.16 / 1.86 | 44 / 23 deg |
| probe with the three terms, 60000 | 8.3 / 5.1 cm | 1.28 / 1.77 | 36 / 21 deg |

The racket got into place, but it neither moved nor faced the ball: the paper's tracking kernels (0.3 m, 0.4 rad,
averaged over 14 bodies) hardly see an arm, and the strike kernels for velocity and face are flat that far from the
target. Added for these questions only (`task.direct_arm`, `direct_face`, `direct_drive`): the waist and right-arm
joints against the reference at 0.5 and 0.15 rad, (1 + cos) / 2 of the face against the commanded normal, and the
share of the commanded speed along the commanded direction.
