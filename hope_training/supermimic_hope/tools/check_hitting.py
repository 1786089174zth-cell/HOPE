"""Fail when a candidate checkpoint hits worse than a baseline (both scored by ``evaluate.py``).

    python tools/check_hitting.py <candidate evaluation.json> <baseline evaluation.json> [tol=0.05]

Compare like with like: same iteration count, same ``evaluate.py`` settings. A candidate passes
when no hit metric is more than ``tol`` (absolute rate) below the baseline, the strike errors are
not more than ``err_tol`` (relative) above it, and it does not fall more. Exit code 1 on a
regression, 2 when the two evaluations used different settings, so it can gate a change.
"""

from __future__ import annotations

import json
import sys

RATES = ("task/hit_rate", "task/position_pass_rate", "task/virtual_hit_rate",
         "stand_forehand/hit_rate", "stand_backhand/hit_rate", "step_forehand/hit_rate", "step_backhand/hit_rate")
ERRORS = ("task/strike_pos_err_m", "task/strike_vel_err_mps", "task/strike_normal_err_deg")
SETTINGS = ("steps", "num_envs", "seed", "action_noise", "actor", "midstroke_probability")


def compare(candidate, baseline, tol=0.05, err_tol=0.15, fall_tol=1e-5):
    """Rows ``(metric, baseline, candidate, ok)``; lower is better for errors and falls."""
    rows = []
    for key in RATES:
        rows.append((key, baseline[key], candidate[key], candidate[key] >= baseline[key] - tol))
    for key in ERRORS:
        rows.append((key, baseline[key], candidate[key], candidate[key] <= baseline[key] * (1 + err_tol)))
    key = "task/falls_per_step"
    rows.append((key, baseline[key], candidate[key], candidate[key] <= baseline[key] + fall_tol))
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    paths = [a for a in argv if "=" not in a and a != "force"]
    opts = {k: float(v) for k, v in (a.split("=", 1) for a in argv if "=" in a)}
    if len(paths) != 2:
        raise SystemExit(__doc__)
    candidate, baseline = (json.load(open(p)) for p in paths)
    # The hit rate depends on how it was measured: refuse to compare unlike evaluations.
    different = [k for k in SETTINGS if candidate.get(k) != baseline.get(k)]
    # Evaluations written before late announcements existed had none.
    for key in ("short_notice", "low_ball", "jitter", "rush", "high_ball", "edge", "plane", "push"):
        if float(candidate.get(key, 0.0)) != float(baseline.get(key, 0.0)):
            different.append(key)
    # ... and before the contact and handoff options: pyramidal contact, no entry handoff.
    if candidate.get("contact", "pyramidal") != baseline.get("contact", "pyramidal"):
        different.append("contact")
    if float(candidate.get("handoff", 0.0)) != float(baseline.get("handoff", 0.0)):
        different.append("handoff")
    if different and "force" not in argv:
        for key in different:
            print(f"[check_hitting] {key} differs: candidate {candidate.get(key)}, baseline {baseline.get(key)}")
        print("[check_hitting] NOT COMPARABLE: re-run evaluate.py with the baseline's settings (or pass `force`)")
        return 2
    if candidate.get("iteration") != baseline.get("iteration"):
        print(f"[check_hitting] note: iterations differ (candidate {candidate.get('iteration')}, "
              f"baseline {baseline.get('iteration')})")
    rows = compare(candidate, baseline, **{k: opts[k] for k in ("tol", "err_tol", "fall_tol") if k in opts})
    for key, base, cand, ok in rows:
        print(f"{'ok  ' if ok else 'FAIL'} {key:32s} baseline {base:.4g}  candidate {cand:.4g}  ({cand - base:+.4g})")
    failed = [r[0] for r in rows if not r[3]]
    print("[check_hitting] " + ("PASS" if not failed else "REGRESSION: " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
