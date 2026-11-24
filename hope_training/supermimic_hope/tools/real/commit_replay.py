"""What an earlier commit costs in prediction accuracy, on the balls of real sessions (numpy only).

    python -I tools/real/commit_replay.py <dump dir> [<dump dir> ...] [--config gate,lead ...]     (hope container)

``<dump dir>`` holds ``mocap.npz`` (``dump_ball.py``) and ``shots.json`` (``shots.py``). The session's ball track
is fed, sample by sample, to main's packetizer and predictor (``commit_replay.cpp``: the copies bundled with
Supergate) once per configuration ``gate,lead``: the packetizer's ``commit_gate_x`` (``nan`` = at the net, as
deployed) and ``adaptive_policy_lead_s`` (0.35 as deployed; the stroke is sent 0.055 s earlier than that). For
every commit the predicted arrival at the strike plane (time, y, z) is compared with where the ball really was:
the recorded track's last 50 ms before the racket (or before the plane, if it was not struck), carried to the
plane. ``miss`` is the distance between the predicted ball and the real one at the predicted instant.

The sessions of 2026-11-21 have no ball faster than 4 m/s, so the commit gate alone changes nothing for them
(each already gets its full notice). A longer lead makes the same balls be committed earlier, from fewer samples
and before their bounce: that is the regime a fast ball is in when it is committed ahead of the net.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOGS = HERE.parents[1] / "logs" / "supergate"
SUPERGATE = LOGS / "tree" / "a3_deploy" / "a3_deploy_example" / "scripts" / "supergate"
BINARY = LOGS / "planner_targets" / "commit_replay"
FLOOR, RADIUS, PLANE_AHEAD, G = 0.031, 0.02, 0.58, 9.81


def build():
    source = HERE / "commit_replay.cpp"
    if BINARY.exists() and BINARY.stat().st_mtime >= source.stat().st_mtime:
        return
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    planner = SUPERGATE / "native" / "planner"
    names = ["batch_physics_estimator", "spin_estimator", "trajectory_predictor", "racket_target_planner", "incoming_trajectory",
             "adaptive_flight_commit", "post_net_one_shot"]
    subprocess.run(["g++", "-std=c++20", "-O2", f"-I{planner / 'include'}", "-I/usr/include/eigen3", str(source),
                    *[str(planner / "src" / f"{n}.cpp") for n in names], "-lyaml-cpp", "-o", str(BINARY)], check=True)


def commits(t, p, plane, gate, lead):
    config = SUPERGATE / "config"
    rows = np.ascontiguousarray(np.concatenate((t[:, None], p), 1), np.float64)
    out = subprocess.run([str(BINARY), str(config / "return_physics.yaml"), str(config / "ball_physics_ppq004_20261020.yaml"),
                          str(config / "model21800_flight_packetizer.yaml"), str(config / "model21800_hardware.yaml"),
                          str(plane), str(gate), str(lead)], input=rows.tobytes(), stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(out, np.float64).reshape(-1, 13)


def arrival(t, p, end, plane):
    """(time, position, velocity) of the real ball at the strike plane: the track's last 50 ms before ``end``
    (free flight: x and y straight, z a parabola), carried to the plane. None if the track is not there."""
    hi = int(np.searchsorted(t, end))
    lo = int(np.searchsorted(t, end - 0.05))
    if hi - lo < 6 or t[hi - 1] - t[lo] < 0.025:
        return None
    tau = t[lo:hi] - t[hi - 1]
    A = np.stack([np.ones_like(tau), tau], 1)
    target = p[lo:hi].copy()
    target[:, 2] += 0.5 * G * tau ** 2
    (p0, v0), *_ = np.linalg.lstsq(A, target, rcond=None)
    if v0[0] > -0.5:
        return None
    dt = (plane - p0[0]) / v0[0]
    if abs(dt) > 0.08:
        return None
    return t[hi - 1] + dt, p0 + v0 * dt - np.array([0.0, 0.0, 0.5 * G * dt * dt]), v0 - np.array([0.0, 0.0, G * dt])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    configs = [("nan", 0.35), ("9", 0.35), ("9", 0.45), ("9", 0.55), ("9", 0.65)]
    if "--config" in argv:
        i = argv.index("--config")
        configs = [(a.split(",")[0], float(a.split(",")[1])) for a in argv[i + 1:]]
        argv = argv[:i]
    build()
    results = {c: [] for c in configs}
    logged, deployed = [], []
    total = 0
    for arg in argv:
        ball = np.load(Path(arg) / "mocap.npz")["ball"]
        order = np.argsort(ball[:, 0])
        t0 = ball[order[0], 0]
        t, p = ball[order, 0] - t0, ball[order, 2:5].copy()
        keep = np.concatenate(([True], np.diff(t) > 1e-6))
        t, p = t[keep], p[keep]
        p[:, 2] += RADIUS - FLOOR  # the ball's centre above the table top
        shots = json.loads((Path(arg) / "shots.json").read_text())
        plane = float(np.median([s["home"][0] for s in shots])) + PLANE_AHEAD
        real = []
        for s in shots:
            end = s["t_strike"] + s["t_contact"] - 0.004 if "t_contact" in s else s["t_strike"] + 0.03
            a = arrival(t, p, end - t0, plane)
            if a is not None:
                real.append((a, s))
        total += len(shots)
        for c in configs:
            rows = commits(t, p, plane, *c)
            rows = rows[rows[:, 3] > 0.5]
            for (ta, pa, va), s in real:
                if not len(rows):
                    continue
                k = int(np.argmin(np.abs(rows[:, 4] - ta)))
                r = rows[k]
                if abs(r[4] - ta) > 0.15:
                    continue
                dt = r[4] - ta
                there = pa + va * dt - np.array([0.0, 0.0, 0.5 * G * dt * dt])  # the real ball at the predicted instant
                miss = r[5:8] - there
                results[c].append((r[4] - r[0], dt, r[6] - pa[1], r[7] - pa[2], float(np.linalg.norm(miss)), r[1], r[11], float(np.linalg.norm(va))))
                if c == configs[0]:
                    target = np.array(s["target"]) - np.array([0.0, 0.0, 0.76])
                    logged.append((r[4] + t0 - s["t_strike"], r[6] - target[1], r[7] - target[2]))
                    deployed.append((s["t_strike"] - t0 - ta, target[1] - pa[1], target[2] - pa[2]))
    if logged:
        e = np.abs(np.array(logged))
        print(f"replay as deployed vs the Planner's logged command ({len(logged)} strokes): strike time {1000 * np.median(e[:, 0]):.0f} ms, "
              f"y {100 * np.median(e[:, 1]):.1f} cm, z {100 * np.median(e[:, 2]):.1f} cm (medians of |difference|)")
    if deployed:
        d = np.array(deployed)
        print(f"the Planner's logged command vs the real ball at the plane: time {1000 * np.median(d[:, 0]):+.0f} ms (|.| median {1000 * np.median(np.abs(d[:, 0])):.0f}, "
              f"90% {1000 * np.quantile(np.abs(d[:, 0]), 0.9):.0f}), y {100 * np.median(d[:, 1]):+.1f} cm (|.| {100 * np.median(np.abs(d[:, 1])):.1f}, "
              f"{100 * np.quantile(np.abs(d[:, 1]), 0.9):.1f}), z {100 * np.median(d[:, 2]):+.1f} cm (|.| {100 * np.median(np.abs(d[:, 2])):.1f}, {100 * np.quantile(np.abs(d[:, 2]), 0.9):.1f})")
    print(f"{total} strokes in the sessions; prediction against the real ball at the strike plane")
    print(f"   {'gate':>4s} {'lead':>5s} | {'balls':>5s} | {'notice s':>8s} | {'fit samples':>11s} | {'before bounce':>13s} | "
          f"{'time ms (median |.|, 90%)':>26s} | {'y cm':>11s} | {'z cm':>11s} | {'miss cm (median, 90%)':>21s} | miss > 5 cm | > 8 cm")
    for c in configs:
        v = np.array(results[c])
        if not len(v):
            print(f"   {c[0]:>4s} {c[1]:5.2f} | no commits")
            continue
        q = lambda x: (np.median(np.abs(x)), np.quantile(np.abs(x), 0.9))
        print(f"   {c[0]:>4s} {c[1]:5.2f} | {len(v):5d} | {np.median(v[:, 0]):8.2f} | {np.median(v[:, 5]):11.0f} | {100 * np.mean(v[:, 6] >= 1):12.0f}% | "
              f"{1000 * q(v[:, 1])[0]:13.0f} {1000 * q(v[:, 1])[1]:12.0f} | {100 * q(v[:, 2])[0]:5.1f} {100 * q(v[:, 2])[1]:5.1f} | "
              f"{100 * q(v[:, 3])[0]:5.1f} {100 * q(v[:, 3])[1]:5.1f} | {100 * np.median(v[:, 4]):10.1f} {100 * np.quantile(v[:, 4], 0.9):10.1f} | "
              f"{100 * np.mean(v[:, 4] > 0.05):10.0f}% | {100 * np.mean(v[:, 4] > 0.08):5.0f}%")
        slow = v[:, 7] < np.median(v[:, 7])
        print(f"        signed medians (prediction - real): time {1000 * np.median(v[:, 1]):+.0f} ms, y {100 * np.median(v[:, 2]):+.1f} cm, z {100 * np.median(v[:, 3]):+.1f} cm | "
              f"spread about them (half of the 16-84% range): time {500 * np.subtract(*np.quantile(v[:, 1], [0.84, 0.16])):.0f} ms, "
              f"y {50 * np.subtract(*np.quantile(v[:, 2], [0.84, 0.16])):.1f} cm, z {50 * np.subtract(*np.quantile(v[:, 3], [0.84, 0.16])):.1f} cm | "
              f"time bias, slower half of the balls {1000 * np.median(v[slow, 1]):+.0f} ms, faster half {1000 * np.median(v[~slow, 1]):+.0f} ms")


if __name__ == "__main__":
    main()
