"""Joints that come near their hard range in a Gate3 run (stdlib only).

    python3 tools/gate3/near_limit.py <run dir> [margin, rad; default 0.04]

Per joint, over the MOTION rows of the Runner trace: ticks within the margin of the low and of the high end of the
hard range, the closest approach (negative: beyond the range) and how close the command came. ``limits.py`` counts
only what is beyond the range; this shows which joints a range keep-out of a given margin would push on (ft53: the
ankle rolls of a step's stance foot ride 0.02 rad inside the range, at their command limit).
"""
import csv, sys
MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04
with open(sys.argv[1] + "/pp_runner_trace.csv") as f:
    rd = csv.reader(f); ix = {n: i for i, n in enumerate(next(rd))}
    rows = [r for r in rd if len(r) >= len(ix) and r[ix["mode"]] == "3"]
joints = [n[4:] for n in ix if n.startswith("des_")]
for j in joints:
    lo, hi = ix["actual_q_hard_lo_" + j], ix["actual_q_hard_hi_" + j]
    near_lo = near_hi = 0; best_lo = best_hi = 9.0; dmax_lo = dmax_hi = 9.0
    for r in rows:
        try:
            q, d, a, b = float(r[ix["q_" + j]]), float(r[ix["des_" + j]]), float(r[lo]), float(r[hi])
        except ValueError:
            continue
        near_lo += q - a < MARGIN; near_hi += b - q < MARGIN
        best_lo = min(best_lo, q - a); best_hi = min(best_hi, b - q)
        dmax_lo = min(dmax_lo, d - a); dmax_hi = min(dmax_hi, b - d)
    if near_lo or near_hi:
        print(f"  {j:28s} range [{a:+.3f}, {b:+.3f}] | within {MARGIN} of low: {near_lo:4d} ticks (closest {best_lo:+.3f}, command {dmax_lo:+.3f}) | of high: {near_hi:4d} ticks (closest {best_hi:+.3f}, command {dmax_hi:+.3f})")
