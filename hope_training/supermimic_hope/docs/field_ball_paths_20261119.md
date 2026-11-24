# Field ball paths: other robots' rallies as a test, and the bar (2026-11-19)

The goal is a hitter that generalizes, not one fitted to Supergate. The bar is a set of real rallies: 30 mocap
recordings of two SJTU RL2 A3 humanoids playing each other (leaderboard records, 54-153 consecutive strokes each,
2026-11-05 to 11-09). Every ball in them was returned by a robot, so parity on these ball paths means returning
about all of them.

## 1. The data and the tools

Source: the shared Drive folder (30 `.mcap`, their `.json` records, videos), downloaded to
`logs/field_ball_paths/drive` (not in the repository). Topics are VRPN mocap poses at 300 Hz in table axes (x along
the table 0..2.74 m, y -1.525..0, z above the table top): the ball, the two robots, a racket.

| step | command |
| --- | --- |
| mocap streams to `.npz` | `python -I tools/field/read_mcap.py <*.mcap> --out logs/field_ball_paths/npz` (needs `mcap`, `mcap-ros2-support`) |
| flights, fitted with the simulator's ball model | `python -I tools/field/extract_flights.py logs/field_ball_paths/npz/*.npz --out logs/field_ball_paths/flights.json` |
| a Supergate-format suite | `python3 tools/field/make_suite.py logs/field_ball_paths/flights.json --out logs/field_ball_paths/suite_field [--cadence relaxed]` |
| score a checkpoint on it | `tools/supergate/score.sh <model_N.pt> logs/field_ball_paths/suite_field` |

A flight is the ball between two racket contacts (its x velocity flips at each); flights towards the far robot are
mirrored, so each is an incoming ball for a robot behind x = 0. Each is fitted with Supergate's flight and
table-bounce model (launch position, velocity and spin, 9 parameters): 2507 flights, 2287 with a fit RMS below 3 cm
and one bounce (median RMS 1.2 cm). The suite replays the fitted launches in recorded order, six shards of 381, with
the recorded pace (`--cadence real`) or Supergate's (`relaxed`).

## 2. What these balls are (at our strike plane, the table edge; quantiles 1 / 10 / 50 / 90 / 99%)

| | field rallies | Supergate main suite (its own plane) |
| --- | --- | --- |
| lateral position from the centre line | -0.42 / -0.25 / -0.04 / +0.20 / +0.36 m | -0.64 .. +0.27 m |
| height above the floor | 0.88 / 1.02 / 1.13 / 1.19 / 1.24 m | 0.82 .. 1.43 m |
| speed | 1.79 / 2.06 / 2.41 / 2.89 / 3.39 m/s | 1.8 .. 5.1 m/s |
| horizontal angle | -14 / -8 / -1 / +4 / +10 deg | -20 .. +20 deg |
| rising at the plane | 62% | 66% |
| topspin after the bounce | 19 / 60 / 108 / 124 / 134 rad/s | about 100 rad/s |
| an incoming ball every | 1.40 / 1.52 / 1.68 / 1.85 / 2.07 s | 2.6 s |

Slower and narrower in height than Supergate, further to the left, and one and a half times its pace. The SJTU robots
stand with the pelvis 0.36 and 0.50 m behind the table edges and strike 5-11 cm behind the edge, 0.84-1.29 m high.

## 3. Baseline: model_59000, robot 0.58 m behind the edge, recorded pace

Planner and judging as in Supergate (offline Planner snapshot, 0.35 s lead).

| | legal returns |
| --- | --- |
| all 2286 | 2006 (87.8%) |
| backhand (Planner's side) | 1908 / 1935 (98.6%) |
| forehand | 98 / 258 (38.0%) |
| no plan | 93, of which 89 `swing_side_target_unsupported`: further left than +0.28 m, the lateral band hard-coded in the Supergate harness (main's Planner takes the band from the bundle, up to +0.46 m) |

By lateral position (m from the centre line): -0.20..0: 960 / 967; -0.24..-0.20: 91 / 134; **-0.28..-0.24: 13 / 84**;
-0.32..-0.28: 41 / 70; -0.40..-0.32: 28 / 58; beyond -0.40: 16 / 25; +0.20..+0.26: 103 / 103.

The loss is one thing: flat strokes on the forehand side and in the crossover zone 0.2-0.34 m right of the centre
line, the same weakness Supergate shows (forehand 175 / 590 there). On the questions of this kind in the simulator
model_59000 meets 57-61% of the forehand targets with the other side of the blade (`docs/supergate_20261119.md`).
