"""Entry posture and table contacts of Gate3 result directories, from the simulator's own log (stdlib only).

    python3 tools/gate3/posture.py <run dir> [<run dir> ...]

Reads ``pp_mujoco_plant.csv`` (200 Hz plant truth) and the Runner's mode from
``pp_runner_trace.csv`` (mode 3 = MOTION, the policy in control), joined on wall time:

* idle: torso and pelvis roll/pitch and the feet width over seconds 3-15 of MOTION (no ball
  yet): a policy that enters leaning shows here; the official stand is 0.265 m wide, the
  four-stroke HOLD reference 0.47 m;
* table: bursts of robot-table contact during MOTION (up to the Runner's last tick), with the racket
  position at the first contact (the table's near edge is world x = 0, its top z = 0.76).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


def roll_pitch(w, x, y, z):
    return (math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
            math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))))


def analyse(folder: Path) -> dict:
    with (folder / "pp_mujoco_plant.csv").open() as stream:
        reader = csv.reader(stream)
        ix = {name: i for i, name in enumerate(next(reader))}
        rows = list(reader)
    g = lambda r, k: float(r[ix[k]])  # noqa: E731
    motion = end = None
    with (folder / "pp_runner_trace.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["mode"] == "3":
                motion = int(row["wall_time_ns"]) if motion is None else motion
                end = int(row["wall_time_ns"])
    if motion is None:
        raise SystemExit(f"{folder}: the Runner never entered MOTION")
    clock = lambda r: (int(r[ix["wall_time_ns"]]) - motion) * 1e-9  # noqa: E731 -- seconds since MOTION entry
    # The plant keeps logging for a few seconds after the Runner is shut down and the robot, with no
    # command, slumps onto the table: only rows up to the Runner's last MOTION tick count.
    rows = [r for r in rows if 0.0 <= clock(r) <= (end - motion) * 1e-9]
    idle = [r for r in rows if 3.0 <= clock(r) <= 15.0]
    mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
    torso = [roll_pitch(*(g(r, f"torso_q{a}") for a in "wxyz")) for r in idle]
    base = [roll_pitch(*(g(r, f"base_q{a}") for a in "wxyz")) for r in idle]
    out = {"run": str(folder), "idle_seconds_of_motion": [3.0, 15.0],
           "torso_roll_deg": mean([a for a, _ in torso]), "torso_pitch_deg": mean([b for _, b in torso]),
           "torso_abs_roll_deg_max": max((abs(a) for a, _ in torso), default=0.0),
           "pelvis_roll_deg": mean([a for a, _ in base]), "pelvis_pitch_deg": mean([b for _, b in base]),
           "feet_width_m": mean([g(r, "left_foot_y") - g(r, "right_foot_y") for r in idle]),
           "pelvis_height_m": mean([g(r, "base_z") for r in idle]), "table_bursts": []}
    burst = None
    for r in rows:
        t = clock(r)
        if g(r, "robot_table_contact_count") <= 0:
            continue
        if burst is None or t - burst["end_s"] > 0.3:
            burst = {"start_s": round(t, 2), "end_s": t, "rows": 0,
                     "racket_xyz": [round(g(r, f"racket_{a}"), 3) for a in "xyz"],
                     "racket_velocity": [round(g(r, f"racket_v{a}"), 2) for a in "xyz"]}
            out["table_bursts"].append(burst)
        burst["end_s"], burst["rows"] = t, burst["rows"] + 1
    for b in out["table_bursts"]:
        b["end_s"] = round(b["end_s"], 2)
    return out


def main(argv=None):
    for arg in (sys.argv[1:] if argv is None else argv):
        s = analyse(Path(arg))
        print(f"{s['run']}\n  idle, 3-15 s into MOTION: torso roll {s['torso_roll_deg']:+.1f} deg (max |roll| "
              f"{s['torso_abs_roll_deg_max']:.1f}), torso pitch {s['torso_pitch_deg']:+.1f} deg, pelvis roll "
              f"{s['pelvis_roll_deg']:+.1f} pitch {s['pelvis_pitch_deg']:+.1f} deg, feet width {s['feet_width_m']:.3f} m, "
              f"pelvis height {s['pelvis_height_m']:.3f} m\n  table contact bursts in MOTION: {len(s['table_bursts'])}"
              + "".join(f"\n    t={b['start_s']}-{b['end_s']} s ({b['rows']} rows) racket {b['racket_xyz']} v {b['racket_velocity']}"
                        for b in s["table_bursts"]))
        (Path(arg) / "posture_summary.json").write_text(json.dumps(s, indent=2) + "\n")


if __name__ == "__main__":
    main()
