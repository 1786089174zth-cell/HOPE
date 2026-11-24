"""Supergate results by how fast the ball arrives and how much notice the stroke had (stdlib only).

    python tools/supergate/by_speed.py <supergate output dir> [...]

For every output directory: strokes grouped by the incoming ball's speed on the strike plane (``strike_v*`` of
``env-K-results.csv``) and, separately, by the notice (``lead``: seconds between the plan and the strike), with
the share that touched the ball, the share that returned it legally, where the misses went and the racket's error
at the planned instant (``env-K-kinematics.csv``). ``--side`` splits each group into forehand and backhand.
"""

from __future__ import annotations

import csv
import glob
import math
import sys
from collections import defaultdict

SPEED = (3.0, 4.0, 5.0, 6.0, 7.0)
NOTICE = (0.30, 0.35, 0.40, 0.50)


def band(value, edges, unit):
    for lo, hi in zip((None, *edges), (*edges, None)):
        if hi is None or value < hi:
            return f"{'' if lo is None else f'{lo:g}'}..{'' if hi is None else f'{hi:g}'} {unit}"


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def load(directory):
    rows = []
    for path in sorted(glob.glob(directory + "/env-*-results.csv")):
        env = path.split("env-")[-1].split("-")[0]
        planned = {}
        with open(path.replace("-results.csv", "-kinematics.csv")) as handle:
            for r in csv.DictReader(handle):
                if r["event"] == "planned":
                    planned[r["shot"]] = r
        with open(path) as handle:
            for r in csv.DictReader(handle):
                if r["started"] != "1":
                    continue
                v = [number(r["strike_v" + a]) for a in "xyz"]
                k = planned.get(r["shot"], {})
                rows.append(dict(
                    env=env, shot=r["shot"], speed=math.sqrt(sum(x * x for x in v)), vx=v[0], vz=v[2], lead=number(r["lead"]),
                    side="FH" if r["side"] == "1" else "BH", y=number(r["strike_y"]) + 0.7625, z=number(r["strike_z"]),
                    contact=number(r["contacts"]) > 0, legal=r["legal"] == "1", net=number(r["nets"]) > 0,
                    landing=number(r["landing_x"]), torque=number(r["max_torque_ratio"]), reason=r["reason"],
                    p_error=number(k.get("p_error_m")), v_error=number(k.get("v_error_mps")), dv_x=number(k.get("dv_x")),
                    dp=[number(k.get("dp_" + a)) for a in "xyz"]))
    return rows


def mean(values):
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def line(name, rows):
    n = len(rows)
    touched = [r for r in rows if r["contact"]]
    missed = n - len(touched)
    net = sum(r["contact"] and not r["legal"] and r["net"] for r in rows)
    long = sum(r["contact"] and not r["legal"] and not r["net"] and math.isfinite(r["landing"]) and r["landing"] > 2.74 for r in rows)
    other = n - sum(r["legal"] for r in rows) - missed - net - long
    return (f"   {name:22s} n {n:4d} | contact {100 * len(touched) / n:5.1f}% legal {100 * sum(r['legal'] for r in rows) / n:5.1f}% | "
            f"no touch {missed:3d} net {net:3d} long {long:3d} other {other:3d} | notice {mean(r['lead'] for r in rows):.2f} s | "
            f"racket err {100 * mean(r['p_error'] for r in rows):4.1f} cm (x {100 * mean(r['dp'][0] for r in rows):+5.1f} "
            f"y {100 * mean(r['dp'][1] for r in rows):+5.1f} z {100 * mean(r['dp'][2] for r in rows):+5.1f}) "
            f"v err {mean(r['v_error'] for r in rows):.2f} (x {mean(r['dv_x'] for r in rows):+.2f}) m/s | torque {mean(r['torque'] for r in rows):.2f}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    by_side = "--side" in argv
    argv = [a for a in argv if a != "--side"]
    for directory in argv:
        rows = [r for r in load(directory.rstrip("/")) if math.isfinite(r["speed"])]
        print(directory.rstrip("/").split("/")[-1])
        if not rows:
            print("   no strokes")
            continue
        print(line("all", rows))
        for title, key in (("ball speed", lambda r: band(r["speed"], SPEED, "m/s")), ("notice", lambda r: band(r["lead"], NOTICE, "s"))):
            groups = defaultdict(list)
            for r in rows:
                if math.isfinite(r["lead"]) or title == "ball speed":
                    groups[(key(r), r["side"]) if by_side else (key(r),)].append(r)
            order = sorted(groups, key=lambda k: (mean(r["speed"] if title == "ball speed" else r["lead"] for r in groups[k]) // 0.01 if not by_side else k[1],
                                                  mean(r["speed"] if title == "ball speed" else r["lead"] for r in groups[k])))
            print(f"  by {title}:")
            for k in order:
                print(line(" ".join(k), groups[k]))


if __name__ == "__main__":
    main()
