"""Physical outcome of one or more Gate3 result directories (stdlib only).

    python3 tools/gate3/summarize.py <run dir> [<run dir> ...]

Reads ``pp_physical_ball_report.json`` (racket contact and landing per shot) and
``pp_rally_report.json`` (engage, falls, the Planner's side per shot). These are simulator
measurements; ``certification_pass`` in run.json is a separate, stricter verdict. Shots whose table
bounce the report rejected by a millimetre are re-judged (``rejudge.py``); the report's own counts are
kept as ``report_contacts`` / ``report_legal_landings``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rejudge import verdict  # noqa: E402 -- table bounces rejected by a millimetre (see rejudge.py)


def summarize(folder: Path) -> dict:
    ball = json.loads((folder / "pp_physical_ball_report.json").read_text())
    rally = json.loads((folder / "pp_rally_report.json").read_text())
    run = json.loads((folder / "run.json").read_text())
    rows = {int(r["shot_id"]): r for r in ball["rows"]}
    sides = {}
    for side, entry in (rally.get("physical_evidence_by_planner_side") or {}).get("by_side", {}).items():
        for shot in entry.get("shot_ids", []):
            sides[int(shot)] = side
    out = {"run": str(folder), "serves": len(rows), "engaged": rally.get("engaged_serves"),
           "falls": rally.get("falls"), "certification_pass": run.get("certification_pass"),
           "runner_sha256": run.get("runner_sha256"), "policy_sha256": run.get("policy_sha256"),
           "packetizer": (run.get("packetizer_config") or {}).get("path"),
           "report_contacts": sum(bool(r["contact_pass"]) for r in rows.values()),
           "report_legal_landings": sum(bool(r["landing_pass"]) for r in rows.values()),
           "contacts": sum(verdict(r)[0] for r in rows.values()),
           "legal_landings": sum(verdict(r)[1] for r in rows.values()),
           "rejudged_shots": sorted(k for k, r in rows.items() if verdict(r)[2]),
           "no_contact_shots": sorted(k for k, r in rows.items() if not verdict(r)[0]),
           "contact_not_legal_shots": sorted(k for k, r in rows.items() if verdict(r)[0] and not verdict(r)[1]),
           "by_side": {}}
    for side in sorted(set(sides.values())):
        shots = [k for k in rows if sides.get(k) == side]
        out["by_side"][side] = {"shots": len(shots), "contacts": sum(verdict(rows[k])[0] for k in shots),
                                "legal_landings": sum(verdict(rows[k])[1] for k in shots)}
    return out


def main(argv=None):
    for arg in (sys.argv[1:] if argv is None else argv):
        s = summarize(Path(arg))
        sides = "  ".join(f"{k} {v['contacts']}/{v['shots']} contact, {v['legal_landings']}/{v['shots']} legal"
                          for k, v in s["by_side"].items())
        print(f"{s['run']}\n  engaged {s['engaged']}/{s['serves']}  contacts {s['contacts']}/{s['serves']}  "
              f"legal landings {s['legal_landings']}/{s['serves']}  falls {s['falls']}  "
              f"certification {s['certification_pass']}\n  {sides}\n  no contact: {s['no_contact_shots']}  "
              f"contact but not legal: {s['contact_not_legal_shots']}"
              + (f"\n  re-judged (table bounce rejected by the report on height; report said {s['report_legal_landings']} legal): "
                 f"{s['rejudged_shots']}" if s["rejudged_shots"] else ""))
        (Path(arg) / "physical_summary.json").write_text(json.dumps(s, indent=2) + "\n")


if __name__ == "__main__":
    main()
