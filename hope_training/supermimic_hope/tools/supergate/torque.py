"""Motor torque asked for per stroke in Supergate runs, by announcement time (stdlib only).

    python3 tools/supergate/torque.py <supergate output dir> [...]

Supergate's simulator clamps each joint's PD torque to the motor's limit and records, per shot, the largest
torque the command *asked for* as a share of the limit (``max_torque_ratio`` in env-*-results.csv) and the
control steps on which some joint was clamped (``clipped_steps`` in env-*-audit.csv). A stroke that asks for
more than a motor has is saturated: the score is honest, the command is not something to send to the robot.
One row per band of announcement time: strokes, share at or over the limit, median and 90th percentile of the
peak share, share of control steps clamped; and one line for the whole run.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

BANDS = ((0.0, 0.25), (0.25, 0.30), (0.30, 0.36), (0.36, 9.0))


def load(run: Path):
    rows = []
    for env in range(6):
        results = run / f"env-{env}-results.csv"
        if not results.is_file():
            continue
        audit = {}
        path = run / f"env-{env}-audit.csv"
        if path.is_file():
            audit = {r["shot"]: r for r in csv.DictReader(path.open())}
        for r in csv.DictReader(results.open()):
            if r["lead"] in ("", "nan") or int(r["valid_plans"]) < 1:
                continue
            a = audit.get(r["shot"], {})
            rows.append((float(r["lead"]), float(r["max_torque_ratio"]), int(a.get("clipped_steps", 0)), int(a.get("control_steps", 0)), int(r["legal"])))
    return rows


def line(name, g):
    if not g:
        return
    peak = sorted(x[1] for x in g)
    steps = sum(x[3] for x in g)
    print(f"  {name:18s} strokes {len(g):5d} | asked for the limit or more: {100 * sum(p >= 0.999 for p in peak) / len(g):5.1f}% | peak share of the limit: "
          f"median {peak[len(peak) // 2]:.2f}, 90% {peak[int(0.9 * (len(peak) - 1))]:.2f}, max {peak[-1]:.2f} | control steps clamped {100 * sum(x[2] for x in g) / max(steps, 1):.2f}% | "
          f"legal {100 * sum(x[4] for x in g) / len(g):.1f}%")


def main(argv=None):
    for arg in (sys.argv[1:] if argv is None else argv):
        rows = load(Path(arg))
        print(arg)
        for a, b in BANDS:
            line(f"notice {a:.2f}-{b:.2f} s" if b < 9 else f"notice >= {a:.2f} s", [x for x in rows if a <= x[0] < b])
        line("all", rows)


if __name__ == "__main__":
    main()
