"""Outcome of a Gate3 run per kind of serve (stdlib only).

    python3 tools/gate3/by_serve.py <run dir> [<run dir> ...] [--kinds tools/gate3/scenarios/ball_paths.json]

For runs of a ``tools/gate3/make_scenario.py`` scene: joins the run's own ``scene.yaml`` (which
serve each shot was) with the simulator's ball report (racket contact, net, landing) and the
Runner trace (the target the policy was given at engage: height, racket speed, notice).
Prints one row per serve kind and the low/mid/high and slow/medium/fast margins, and writes
``by_serve.json`` into the run directory.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rejudge import verdict  # noqa: E402


def shot_kinds(folder: Path, kinds: dict) -> dict:
    """shot id -> (kind name, lateral y), from the scene the run was launched with."""
    rows, section = {"clean_serves": [], "rapid_cycle": []}, None
    balls = None
    for line in (folder / "scene.yaml").read_text().splitlines():
        if line.startswith("balls:"):
            balls = int(line.split(":")[1])
        elif line.rstrip(":") in rows:
            section = line.rstrip(":")
        elif line.startswith("- [") and section:
            rows[section].append(json.loads(line[2:]))
        elif not line.startswith("- "):
            section = None
    run = json.loads((folder / "run.json").read_text())
    balls = int(run.get("balls") or balls)
    serves = rows["clean_serves"] + [rows["rapid_cycle"][i % len(rows["rapid_cycle"])] for i in range(balls - 6)]
    lookup = {tuple(round(v, 4) for v in k["launch"]): name for name, k in kinds["kinds"].items()}
    lookup[tuple(round(v, 4) for v in kinds["standard"]["launch"])] = "standard"
    return {i + 1: (lookup.get((round(s[2], 4), round(s[3], 4), round(s[5], 4)), "other"), s[1]) for i, s in enumerate(serves)}


def analyse(folder: Path, kinds: dict) -> dict:
    which = shot_kinds(folder, kinds)
    ball = {int(r["shot_id"]): r for r in json.loads((folder / "pp_physical_ball_report.json").read_text())["rows"]}
    rally = json.loads((folder / "pp_rally_report.json").read_text())
    side = {int(s): name for name, e in (rally.get("physical_evidence_by_planner_side") or {}).get("by_side", {}).items()
            for s in e.get("shot_ids", [])}
    engage = {}
    with (folder / "pp_runner_trace.csv").open() as stream:
        for r in csv.DictReader(stream):
            shot = int(float(r["shot_seq"] or 0))
            if r["mode"] == "3" and r["level"] != "0" and shot and shot not in engage:
                v = [float(r[f"target_v{a}"]) for a in "xyz"]
                engage[shot] = dict(target_z=float(r["target_z"]), racket_speed=sum(x * x for x in v) ** 0.5,
                                    racket_vz=v[2], notice_s=float(r["engage_raw_tts"]))
    shots = []
    for shot, (kind, y) in which.items():
        b = ball.get(shot)
        # The report's contact_pass / landing_pass are false for every shot whose telemetry it calls
        # incomplete, and one ball left lying on the table (a low serve nobody returned) does that to all
        # the shots after it. The events themselves are still recorded: a racket contact is a contact, and
        # a first bounce on the far half without a net touch is a legal return.
        contact = bool(b and (b["contact_pass"] or b["racket_contact_count"] > 0))
        legal = bool(b and verdict(b)[1])  # verdict: bounces rejected by a millimetre re-judged (rejudge.py)
        if b and contact and not legal and not b["telemetry_complete"] and not b["post_racket_net_events"] and b["post_racket_table_events"]:
            x, yy = b["post_racket_table_events"][0]["position_table"][:2]
            legal = 1.37 < x <= 2.74 and -1.525 <= yy <= 0.0
        shots.append(dict(shot=shot, kind=kind, y=y, side=side.get(shot), engaged=shot in engage,
                          contact=contact, legal=legal, telemetry_complete=bool(b and b["telemetry_complete"]),
                          net=bool(b and b["post_racket_net_events"]),
                          landing_x=(b["landing_event"]["position_table"][0] if b and b.get("landing_event") else None),
                          **engage.get(shot, {})))
    return dict(run=str(folder), shots=shots)


def table(shots, key, title):
    groups = {}
    for s in shots:
        groups.setdefault(key(s), []).append(s)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    print(f"  {title:15s} balls  engaged  contact  legal   net  missed-table | target z   racket speed  notice")
    out = {}
    for name, g in groups.items():
        hit = [s for s in g if "target_z" in s]
        row = dict(balls=len(g), engaged=sum(s["engaged"] for s in g), contact=sum(s["contact"] for s in g),
                   legal=sum(s["legal"] for s in g), net=sum(s["net"] for s in g),
                   missed_table=sum(s["contact"] and not s["legal"] and not s["net"] for s in g),
                   target_z=mean([s["target_z"] for s in hit]), racket_speed=mean([s["racket_speed"] for s in hit]),
                   notice_s=mean([s["notice_s"] for s in hit]))
        out[name] = row
        print(f"  {name:15s} {row['balls']:5d}  {row['engaged']:7d}  {row['contact']:7d}  {row['legal']:5d}  {row['net']:4d}  "
              f"{row['missed_table']:12d} | {row['target_z']:.2f} m    {row['racket_speed']:.2f} m/s      {row['notice_s']:.2f} s")
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    kinds_path = Path(__file__).resolve().parent / "scenarios" / "ball_paths.json"
    if "--kinds" in argv:
        i = argv.index("--kinds")
        kinds_path = Path(argv[i + 1])
        del argv[i:i + 2]
    kinds = json.loads(kinds_path.read_text())
    order = ["standard"] + list(kinds["kinds"])
    for arg in argv:
        result = analyse(Path(arg), kinds)
        shots = sorted(result["shots"], key=lambda s: order.index(s["kind"]) if s["kind"] in order else 99)
        n = len(shots)
        print(f"{result['run']}\n  {n} balls: engaged {sum(s['engaged'] for s in shots)}, contact {sum(s['contact'] for s in shots)}, "
              f"legal landings {sum(s['legal'] for s in shots)}"
              + (f"  ({sum(not s['telemetry_complete'] for s in shots)} shots judged from events: the report's telemetry was incomplete)"
                 if any(not s["telemetry_complete"] for s in shots) else ""))
        result["by_kind"] = table(shots, lambda s: s["kind"], "serve")
        varied = [s for s in shots if "_" in s["kind"]]
        if varied:
            result["by_height"] = table(varied, lambda s: s["kind"].split("_")[0], "height")
            result["by_speed"] = table(varied, lambda s: s["kind"].split("_")[1], "ball speed")
            result["by_side"] = table(varied, lambda s: str(s["side"]), "side")
        failed = [s for s in shots if not s["legal"]]
        print("  not legal: " + (", ".join(
            f"#{s['shot']} {s['kind']} y{s['y']:+.2f} " + ("no engage" if not s["engaged"] else "no contact" if not s["contact"]
                                                          else "net" if s["net"] else f"landed x {s['landing_x']:.2f}"
                                                          if s["landing_x"] is not None else "no landing") for s in failed) or "none"))
        (Path(arg) / "by_serve.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
