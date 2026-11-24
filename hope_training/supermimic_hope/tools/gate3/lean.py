"""Sideways lean while the policy holds, from a Gate3 run's plant log (stdlib only).

    python3 tools/gate3/lean.py <run dir> [<run dir> ...]

Looking along the robot's forward direction (x), left is +y. Roll here is about the forward axis, positive when the
right side goes down (ISO: y to z), so a lean to the robot's left is a NEGATIVE roll. Reported over (a) the idle
seconds 3-15 of MOTION, before the first ball, and (b) the holds between balls (Runner level 0 for at least 1 s, up to the Runner's last tick):
torso and pelvis roll, the waist roll joint, how far the torso and the pelvis sit to the left of the midpoint of
the feet, and the share of the weight on the left foot.
"""

from __future__ import annotations

import bisect
import csv
import math
import sys
from pathlib import Path


def roll(w, x, y, z):
    return math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


def yaw(w, x, y, z):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def analyse(folder: Path):
    with (folder / "pp_mujoco_plant.csv").open() as stream:
        reader = csv.reader(stream)
        ix = {name: i for i, name in enumerate(next(reader))}
        rows = [r for r in reader if len(r) >= len(ix)]
    g = lambda r, k: float(r[ix[k]])  # noqa: E731
    times, level, motion = [], [], None
    with (folder / "pp_runner_trace.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["mode"] == "3":
                t = int(row["wall_time_ns"])
                motion = t if motion is None else motion
                times.append(t)
                level.append(int(float(row["level"] or 0)))
    held = [0.0] * len(times)  # seconds the Runner has been at level 0 (no swing) up to each tick
    for i in range(1, len(times)):
        held[i] = held[i - 1] + (times[i] - times[i - 1]) * 1e-9 if level[i] == 0 else 0.0

    def stats(sel):
        out = {k: [] for k in ("torso_roll", "pelvis_roll", "waist_roll", "torso_left", "pelvis_left", "left_share")}
        for r in rows:
            t = int(r[ix["wall_time_ns"]])
            if not sel(t):
                continue
            heading = yaw(*(g(r, f"base_q{a}") for a in "wxyz"))
            mid = (0.5 * (g(r, "left_foot_x") + g(r, "right_foot_x")), 0.5 * (g(r, "left_foot_y") + g(r, "right_foot_y")))
            left = lambda x, y: -(x - mid[0]) * math.sin(heading) + (y - mid[1]) * math.cos(heading)  # noqa: E731
            out["torso_roll"].append(roll(*(g(r, f"torso_q{a}") for a in "wxyz")))
            out["pelvis_roll"].append(roll(*(g(r, f"base_q{a}") for a in "wxyz")))
            out["waist_roll"].append(math.degrees(g(r, "q_waist_roll_joint")))
            out["torso_left"].append(100 * left(g(r, "torso_x"), g(r, "torso_y")))
            out["pelvis_left"].append(100 * left(g(r, "base_x"), g(r, "base_y")))
            fl, fr = g(r, "left_foot_normal_force"), g(r, "right_foot_normal_force")
            if fl + fr > 1:
                out["left_share"].append(fl / (fl + fr))
        return {k: (sum(v) / len(v), min(v), max(v), len(v)) for k, v in out.items() if v}

    def between(t):
        i = bisect.bisect_right(times, t) - 1
        return i >= 0 and t - motion > 16e9 and t <= times[-1] and held[i] >= 1.0  # level 0 for a second: the stroke's tail is over

    return dict(idle=stats(lambda t: motion is not None and 3e9 <= t - motion <= 15e9), holds=stats(between))


def main(argv=None):
    for arg in (sys.argv[1:] if argv is None else argv):
        res = analyse(Path(arg))
        print(arg)
        for name, s in res.items():
            if not s:
                continue
            f = lambda k, unit: f"{s[k][0]:+.1f}{unit} ({s[k][1]:+.1f}..{s[k][2]:+.1f})"  # noqa: E731
            print(f"  {name:5s} torso roll {f('torso_roll', ' deg')}, pelvis roll {f('pelvis_roll', ' deg')}, waist roll joint {f('waist_roll', ' deg')}; "
                  f"left of the feet's midpoint: torso {f('torso_left', ' cm')}, pelvis {f('pelvis_left', ' cm')}; weight on the left foot {100 * s['left_share'][0]:.0f}%")


if __name__ == "__main__":
    main()
