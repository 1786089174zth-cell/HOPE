"""Re-judge Gate3 shots whose table bounce the ball report rejected on height by a millimetre (stdlib only).

The report (``pp_gate3_core.py``) accepts a table bounce when the first post-impulse sample is within
``5 mm + |vz| * lag + g lag^2 / 2`` of table height, with ``lag`` the measured observation lag. The
simulator's bounce samples come at four discrete heights above the table (20.1, 23.4, 26.6, 29.9 mm
for the standard serve): the sample can be up to one 3-4 ms physics step after the impact while the
reported lag is about 1 ms, so the highest of them is rejected (by about 1 mm) on roughly one shot in
a hundred. A rejected serve bounce makes the shot a contact failure; a rejected return bounce makes a
legal return illegal. This module applies the same rule with the lag taken as at least ``STEP_S``.

    python3 tools/gate3/rejudge.py <run dir> [...]      prints the shots whose verdict changes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BALL_RADIUS_M, NET_X_M, TABLE_LENGTH_M, TABLE_WIDTH_M, TABLE_HEIGHT_M = 0.020, 1.370, 2.74, 1.525, 0.760
STEP_S = 0.004  # one physics step of the plant's ball log


def legal_table_event(event, x_min, x_max, vx_sign):
    if not event:
        return False
    x, y, z = (float(v) for v in event["position_world"])
    vx, _, vz = (float(v) for v in event["velocity_world"])
    lag = max(float(event.get("edge_observation_lag_s", 0.0)), STEP_S)
    tolerance = 0.005 + abs(vz) * lag + 0.5 * 9.81 * lag * lag
    return bool(x_min <= x <= x_max and -TABLE_WIDTH_M + BALL_RADIUS_M <= y <= -BALL_RADIUS_M
                and abs(z - (TABLE_HEIGHT_M + BALL_RADIUS_M + 1e-4)) <= tolerance and vz > 0.0 and vx_sign * vx > 0.0)


def verdict(row):
    """(contact, legal, changed) for one row of ``pp_physical_ball_report.json``."""
    contact, legal = bool(row["contact_pass"]), bool(row["landing_pass"])
    serve_ok = len(row["incoming_table_events"]) == 1 and legal_table_event(
        row["incoming_table_events"][0], BALL_RADIUS_M, NET_X_M - BALL_RADIUS_M, -1)
    new_contact = contact or bool(
        row["telemetry_complete"] and serve_ok and not row["incoming_net_events"] and len(row["racket_events"]) == 1
        and float(row["racket_events"][0]["velocity_world"][0]) > 0.0)
    landing = row["post_racket_table_events"][0] if row["post_racket_table_events"] else None
    new_legal = legal or bool(new_contact and row.get("terminal_event") is not None and not row["post_racket_net_events"]
                              and legal_table_event(landing, NET_X_M + BALL_RADIUS_M, TABLE_LENGTH_M - BALL_RADIUS_M, 1))
    return new_contact, new_legal, (new_contact, new_legal) != (contact, legal)


def main(argv=None):
    for arg in (sys.argv[1:] if argv is None else argv):
        rows = json.loads((Path(arg) / "pp_physical_ball_report.json").read_text())["rows"]
        changed = [(r["shot_id"], verdict(r)) for r in rows if verdict(r)[2]]
        legal = sum(verdict(r)[1] for r in rows)
        print(f"{arg}: {legal}/{len(rows)} legal after re-judging; changed: "
              + (", ".join(f"#{s} -> contact {int(c)} legal {int(l)}" for s, (c, l, _) in changed) or "none"))


if __name__ == "__main__":
    main()
