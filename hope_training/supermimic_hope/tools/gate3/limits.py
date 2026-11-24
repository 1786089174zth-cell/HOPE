"""Limits and command smoothness over a whole Gate3 run, strokes included (stdlib only).

    python3 tools/gate3/limits.py <run dir> [<run dir> ...]

Gate3's own plant report (``pp_mujoco_plant_report.json``) gates smoothness in the no-ball idle seconds only.
This reads the same Runner trace and plant log for the rally: MOTION rows are split into *stroke* (the Runner
has an engaged stroke, ``level`` > 0) and *hold* (between strokes, from 4 s after the entry so that the stance
taken on entry is not counted), and for each part reports

* q_des reversals: sign changes of the commanded step per second, per joint (steps under 1 mrad ignored, as
  in the plant report; its idle limit is 8 Hz) and the largest commanded step in one 20 ms tick;
* joints beyond their hard limits (the Runner's ``actual_q_hard_*`` audit), commands the Runner had to clamp
  (``clamp_viol_*``) and ticks where its slew limiter cut the command (``qdes_slew_*``);
* actuator effort as a share of the torque limit (plant ``ctrl_ratio_*``): peak and rows at the limit;
* the plant report's collision ledger for the whole run: rows with the robot touching the table, the net or itself.

One line per run and part, then the worst joints. A recipe change must not raise these (the user, 2026-11-21:
"不要把 reversal 等超限带回来").
"""

from __future__ import annotations

import bisect
import csv
import json
import sys
from pathlib import Path

EPS = 0.001  # rad: commanded steps below this are not counted as a direction


def reversals(values, seconds):
    signs = [1 if b - a > EPS else -1 for a, b in zip(values, values[1:]) if abs(b - a) > EPS]
    return sum(a != b for a, b in zip(signs, signs[1:])) / seconds if seconds > 0 else 0.0


def analyse(folder: Path) -> dict:
    with (folder / "pp_runner_trace.csv").open() as stream:
        reader = csv.reader(stream)
        ix = {name: i for i, name in enumerate(next(reader))}
        rows = [r for r in reader if len(r) >= len(ix) and r[ix["mode"]] == "3"]
    joints = [n[4:] for n in ix if n.startswith("des_")]
    t = [int(r[ix["wall_time_ns"]]) for r in rows]
    stroke = [r[ix["level"]] not in ("", "0") for r in rows]
    first = stroke.index(True) if True in stroke else len(rows)
    last = len(stroke) - 1 - stroke[::-1].index(True) if True in stroke else 0
    out = {}
    for part in ("stroke", "hold"):
        # consecutive rows of the part form segments; steps are taken inside a segment only
        keep = [(s if part == "stroke" else (not s and t[i] - t[0] > 4e9 and i <= last + 150)) for i, s in enumerate(stroke)]
        segments, start = [], None
        for i, k in enumerate(keep + [False]):
            if k and start is None:
                start = i
            if not k and start is not None:
                if i - start >= 3:
                    segments.append((start, i))
                start = None
        seconds = sum(t[b - 1] - t[a] for a, b in segments) / 1e9
        per = {}
        for j in joints:
            c = ix["des_" + j]
            rev = sum(reversals([float(r[c]) for r in rows[a:b]], 1.0) for a, b in segments) / max(seconds, 1e-9)
            step = max((abs(float(rows[i + 1][c]) - float(rows[i][c])) for a, b in segments for i in range(a, b - 1)), default=0.0)
            clamp = sum(float(rows[i][ix["clamp_viol_" + j]] or 0) != 0 for a, b in segments for i in range(a, b))
            excess = max((float(rows[i][ix["actual_q_hard_excess_" + j]] or 0) for a, b in segments for i in range(a, b)), default=0.0)
            per[j] = dict(reversals_hz=rev, step_peak=step, clamped=clamp, hard_excess=excess)
        n = sum(b - a for a, b in segments)
        col = lambda name: [float(rows[i][ix[name]] or 0) for a, b in segments for i in range(a, b)]  # noqa: E731
        out[part] = dict(seconds=seconds, ticks=n, joints=per,
                         hard_ticks=sum(v > 0 for v in col("actual_q_hard_violation_count")),
                         hard_excess=max(col("actual_q_hard_max_excess_rad"), default=0.0),
                         slew_ticks=sum(v > 0 for v in col("qdes_slew_saturated_count")),
                         slew_clip=max(col("qdes_slew_max_clip_rad"), default=0.0),
                         windows=[(t[a], t[b - 1]) for a, b in segments])
    # actuator effort from the plant log, in the same windows
    with (folder / "pp_mujoco_plant.csv").open() as stream:
        reader = csv.reader(stream)
        px = {name: i for i, name in enumerate(next(reader))}
        ratio = [n for n in px if n.startswith("ctrl_ratio_")]
        prow = [r for r in reader if len(r) >= len(px)]
    pt = [int(r[px["wall_time_ns"]]) for r in prow]
    for part in out:
        peak, at_limit, total = {}, 0, 0
        for a, b in out[part].pop("windows"):
            for r in prow[bisect.bisect_left(pt, a):bisect.bisect_right(pt, b)]:
                total += 1
                worst = 0.0
                for n in ratio:
                    v = abs(float(r[px[n]] or 0))
                    worst = max(worst, v)
                    if v > peak.get(n, 0.0):
                        peak[n] = v
                at_limit += worst >= 0.999
        out[part].update(effort_peak=max(peak.values(), default=0.0), effort_joint=max(peak, key=peak.get, default="")[11:],
                         effort_at_limit=at_limit / max(total, 1))
    return out


def main(argv=None):
    for arg in (sys.argv[1:] if argv is None else argv):
        res = analyse(Path(arg))
        print(arg)
        for part, r in res.items():
            j = r["joints"]
            rev = max(j, key=lambda k: j[k]["reversals_hz"])
            step = max(j, key=lambda k: j[k]["step_peak"])
            print(f"  {part:6s} {r['seconds']:6.1f} s | q_des reversals max {j[rev]['reversals_hz']:5.2f} Hz ({rev}) | largest step {j[step]['step_peak']:.3f} rad ({step}) | "
                  f"joint beyond hard limit: {r['hard_ticks']} ticks, max {r['hard_excess']:.3f} rad | command clamped: {sum(v['clamped'] for v in j.values())} joint-ticks | "
                  f"slew-limited: {r['slew_ticks']} ticks (max cut {r['slew_clip']:.3f} rad) | effort peak {r['effort_peak']:.2f} of limit ({r['effort_joint']}), at limit {100 * r['effort_at_limit']:.2f}% of rows")
            top = sorted(j, key=lambda k: -j[k]["reversals_hz"])[:5]
            print("         reversals Hz: " + ", ".join(f"{k.removesuffix('_joint')} {j[k]['reversals_hz']:.1f}" for k in top))
            bad = [k for k in j if j[k]["clamped"] or j[k]["hard_excess"] > 0]
            if bad:
                print("         clamped / beyond limit: " + ", ".join(f"{k.removesuffix('_joint')} {j[k]['clamped']} ticks / {j[k]['hard_excess']:.3f} rad" for k in bad))
        report = Path(arg) / "pp_mujoco_plant_report.json"
        if report.is_file():
            ledger = json.loads(report.read_text()).get("collision_ledger", {})
            res["collisions"] = {k: ledger.get(k) for k in ("robot_table_contact_rows", "robot_net_contact_rows", "robot_self_contact_rows")}
            print("  contact rows: " + ", ".join(f"{k.split('_')[1]} {v}" for k, v in res["collisions"].items()))
        (Path(arg) / "limits.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
