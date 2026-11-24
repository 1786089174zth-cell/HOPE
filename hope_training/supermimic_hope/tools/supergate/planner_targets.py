"""The Planner's target for each launch of a launch file, without simulating anything but the ball.

    # inside the hope container (the binary needs its yaml-cpp), from the repository root:
    .venv/bin/python hope_training/supermimic_hope/tools/supergate/planner_targets.py \
        --launches <general_launches_*.json> ... --out <rows.npz> [--check <run dir> <dataset dir>]

What the Planner commands for a ball is a function of the ball alone: its state where it crosses the strike
plane (position, velocity, spin), a fixed landing point (2.055 m beyond the robot's edge on the centre line)
and a fixed return flight time (0.5 s) go into ``RacketTargetPlanner::plan``, which solves for the racket
velocity (the face normal is along it). ``planner_targets.cpp`` calls that function, the copy of main's
Planner bundled with Supergate, on rows of strike states; this script builds it on first use, flies each
launch with Supergate's flight model (``make_launches.flight``) and writes rows like those
``build_questions.collect`` reads from a simulated run:

    target position (world), racket velocity, ball velocity at the strike, ball spin there, announcement time,
    side (0: left to the split), legal, contact (1: not simulated)

The announcement time is the time from the net to the plane, at most 0.405 s: the Planner commits once after
the net, 0.35 s of policy lead plus transport and solve budgets ahead.

Against 12 000-14 500 targets recorded from simulated runs (Runner + Planner + robot in Supergate's batch
simulator, 50 launches per thread and minute): racket velocity within 0.005 m/s (median; 0.017 at the 99th
percentile) from the true contact state, 0.0002 m/s from the Planner's own estimate of it; target position
within 2 mm in height (6 mm at the 90th); announcement within 2 ms (4 ms at the 90th). ``--check`` repeats the
comparison for a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_launches import HOME_Y, TABLE_Z, flight  # noqa: E402

LOGS = HERE.parents[1] / "logs" / "supergate"
SUPERGATE = LOGS / "tree" / "a3_deploy" / "a3_deploy_example" / "scripts" / "supergate"
BINARY = LOGS / "planner_targets" / "planner_targets"
LAND, RETURN_FLIGHT_S, LEAD_CAP_S = (2.055, -0.7625), 0.5, 0.405


def build(binary: Path = BINARY):
    """Compile ``planner_targets.cpp`` against the Planner sources bundled with Supergate."""
    binary.parent.mkdir(parents=True, exist_ok=True)
    planner = SUPERGATE / "native" / "planner"
    subprocess.run(["g++", "-std=c++20", "-O2", "-fopenmp", f"-I{planner / 'include'}", "-I/usr/include/eigen3",
                    str(HERE / "planner_targets.cpp"), str(planner / "src" / "racket_target_planner.cpp"), "-lyaml-cpp",
                    "-o", str(binary)], check=True)


def plan(strike, binary: Path = BINARY):
    """Racket commands for (N, 9) strike states (position in the table frame, velocity, spin): (N, 12), see the
    .cpp. Row 0 is 1 where the Planner has a command."""
    source = HERE / "planner_targets.cpp"
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        build(binary)
    out = subprocess.run([str(binary), str(SUPERGATE / "config" / "return_physics.yaml"),
                          str(SUPERGATE / "config" / "ball_physics_ppq004_20261020.yaml"), str(LAND[0]), str(LAND[1]),
                          str(RETURN_FLIGHT_S)], input=np.ascontiguousarray(strike, np.float64).tobytes(),
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(out, np.float64).reshape(-1, 12)


def fly(launch, spin, plane=0.0, dt=0.001, steps=2600, binary: Path = BINARY):
    """``make_launches.flight(..., times=True)`` compiled and on every core: the same eight arrays for (N, 6)
    launches and (N, 3) spins (state at the plane, spin there, flight time, bounces, bounce point, (y, z) at the
    net, time at the net, time of the bounce)."""
    source = HERE / "planner_targets.cpp"
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        build(binary)
    rows = np.ascontiguousarray(np.concatenate((launch, spin), 1), np.float64)
    out = subprocess.run([str(binary), "flight", str(plane), str(dt), str(steps)], input=rows.tobytes(), stdout=subprocess.PIPE, check=True).stdout
    o = np.frombuffer(out, np.float64).reshape(-1, 17)
    return o[:, 0:6], o[:, 6:9], o[:, 9], o[:, 10].astype(int), o[:, 11:13], o[:, 13:15], o[:, 15], o[:, 16]


def load_launches(paths):
    """Launch position and velocity (N, 6), spin (N, 3) and the strike plane of launch files: the JSON lists of
    make_launches.py or the npz of vary_launches.py."""
    launch, spin, plane = [], [], 0.0
    for path in paths:
        if Path(path).suffix == ".npz":
            z = np.load(path)
            launch.append(z["launch"]), spin.append(z["initial_spin"])
            plane = float(z["plane"])
        else:
            records = json.loads(Path(path).read_text())
            launch.append(np.array([x["launch"] for x in records], dtype=float))
            spin.append(np.array([x["initial_spin"] for x in records], dtype=float))
            plane = float(records[0].get("plane", 0.0))
    return np.concatenate(launch), np.concatenate(spin), plane


def rows_of(launch, spin0, plane=0.0):
    """Rows (N, 16) of ``build_questions.collect`` for launches, and how many were dropped (no one-bounce flight
    to the plane, or no command)."""
    s, w, t, bounces, _, _, t_net, _ = fly(launch, spin0, plane=plane)
    ok = np.isfinite(s).all(1) & (bounces == 1) & np.isfinite(t_net)
    command = plan(np.concatenate((s, w), 1)[ok])
    good = command[:, 0] > 0.5
    s, w, lead = s[ok][good], w[ok][good], np.minimum(LEAD_CAP_S, (t - t_net)[ok][good])
    n = len(s)
    rows = np.concatenate((s[:, :3] + np.array([0.0, 0.0, TABLE_Z]), command[good, 1:4], s[:, 3:], w, lead[:, None],
                           np.zeros((n, 1)), np.ones((n, 2))), 1)
    return rows, len(launch) - len(rows)


def check(run: Path, dataset: Path):
    """Targets of a simulated run against the ones computed here for the same launches."""
    recorded, records = [], []
    for shard in range(6):
        ledger = json.loads((dataset / f"shard-{shard + 1:02d}-ledger.json").read_text())
        res = {int(r["shot"]): r for r in csv.DictReader((run / f"env-{shard}-results.csv").open())}
        for k in csv.DictReader((run / f"env-{shard}-kinematics.csv").open()):
            shot = int(k["shot"])
            r = res.get(shot)
            if k["event"] != "planned" or r is None or int(r["valid_plans"]) < 1 or shot > len(ledger):
                continue
            recorded.append([float(k[f"target_p_{a}"]) for a in "xyz"] + [float(k[f"target_v_{a}"]) for a in "xyz"] + [float(r["lead"])])
            records.append(ledger[shot - 1])
    recorded = np.array(recorded)
    launch = np.array([x["launch"] for x in records], dtype=float)
    spin0 = np.array([x["initial_spin"] for x in records], dtype=float)
    s, w, t, bounces, _, _, t_net, _ = flight(launch, spin0, plane=float(records[0].get("plane", 0.0)), steps=2600, times=True)
    ok = np.isfinite(s).all(1) & np.isfinite(recorded).all(1) & np.isfinite(t_net)
    command = plan(np.concatenate((s, w), 1)[ok])
    good = command[:, 0] > 0.5
    rec = recorded[ok][good]
    q = lambda x: np.quantile(x, [0.5, 0.9, 0.99]).round(4).tolist()  # noqa: E731
    print(f"{run.name}: {len(recorded)} recorded targets, {int(good.sum())} with a command here")
    print(f"  racket velocity, |here - recorded| m/s (median, 90%, 99%): {q(np.linalg.norm(command[good, 1:4] - rec[:, 3:6], axis=1))}")
    print(f"  target position m: lateral {q(np.abs(s[ok][good][:, 1] - rec[:, 1]))}, height {q(np.abs(s[ok][good][:, 2] + TABLE_Z - rec[:, 2]))}")
    print(f"  announcement s: {q(np.abs(np.minimum(LEAD_CAP_S, (t - t_net)[ok][good]) - rec[:, 6]))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launches", type=Path, nargs="+", default=[], help="launch files of make_launches.py (json) or vary_launches.py (npz)")
    ap.add_argument("--out", type=Path, help="npz with 'rows' for build_questions.py --rows")
    ap.add_argument("--check", type=Path, nargs=2, metavar=("RUN", "DATASET"), help="compare with a simulated run")
    args = ap.parse_args()
    if args.check:
        check(*args.check)
    if args.launches:
        launch, spin0, plane = load_launches(args.launches)
        launches = launch
        rows, dropped = rows_of(launch, spin0, plane)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, rows=rows.astype(np.float64))
        y, z, speed = rows[:, 1] - HOME_Y, rows[:, 2], np.linalg.norm(rows[:, 6:9], axis=1)
        q = lambda x: np.quantile(x, [0.01, 0.5, 0.99]).round(2).tolist()  # noqa: E731
        print(f"{args.out}: {len(rows)} targets from {len(launches)} launches ({dropped} without a command); lateral {q(y)} m from HOME, "
              f"height {q(z)} m, ball speed {q(speed)} m/s, racket speed {q(np.linalg.norm(rows[:, 3:6], axis=1))} m/s, "
              f"announcement {q(rows[:, 12])} s")


if __name__ == "__main__":
    main()
