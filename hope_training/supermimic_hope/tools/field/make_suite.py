"""A Supergate suite of field ball paths (stdlib only).

    python3 tools/field/make_suite.py <flights.json> --out <suite dir> [--cadence real|relaxed]

``flights.json`` is the output of ``extract_flights.py``. Flights whose fit is clean (RMS below 3 cm, one bounce,
reaching the strike plane) are kept in their recorded order per recording and receiving robot, so that a shard
is a sequence of real rallies, and dealt to six shards of equal length (Supergate's format: six continuous
environments). Each launch is the fitted ball state just after the sender's racket (position, velocity, spin).

``--cadence real`` (default) launches the next ball when it was launched in the recording (an incoming ball every
1.4-2.1 s, 1.7 s typically), but never sooner than 0.8 s after the ball reaches the strike plane, so the return
has landed before it is cleared. ``relaxed`` is Supergate's own timing: cleared 1.0 s after the plane, then a
0.8 s pause (about 2.6 s per ball): the same balls without the rally's pace.
The files are written like Supergate's ``freeze_dataset.py`` writes them, without its shuffle.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opt = {"--cadence": "real"}
    for name in ("--out", "--cadence"):
        if name in argv:
            i = argv.index(name)
            opt[name] = argv[i + 1]
            del argv[i:i + 2]
    out = Path(opt["--out"])
    flights = json.loads(Path(argv[0]).read_text())
    clean = [f for f in flights if f["fit_rms_m"] < 0.03 and f["bounces"] == 1 and math.isfinite(f["plane0_t"])
             and all(math.isfinite(v) for v in f["launch"] + f["initial_spin"] + f["plane007"][:6])]
    rallies = {}
    for f in clean:
        rallies.setdefault((f["recording"], f["receiver"]), []).append(f)
    shards = [[] for _ in range(6)]
    for key in sorted(rallies, key=lambda k: -len(rallies[k])):  # longest first, to the shortest shard
        rally = sorted(rallies[key], key=lambda f: f["t_launch"])
        rows = []
        for f, nxt in zip(rally, rally[1:] + [None]):
            floor = f["plane0_t"] + 0.8
            if opt["--cadence"] == "relaxed":
                duration, pause = f["plane0_t"] + 1.0, 0.8
            else:
                gap = nxt["t_launch"] - f["t_launch"] if nxt else floor + 0.2
                duration, pause = max(min(gap, 3.0), floor), 0.0
            rows.append(dict(launch=f["launch"], initial_spin=f["initial_spin"], flight_duration_s=round(duration, 3), pause_s=pause,
                             predicted_contact=f["plane007"][:6], predicted_contact_spin=f["plane007"][6:9], flight_s=f["plane007_t"],
                             plane0=f["plane0"], bounce=f["bounce"], recording=f["recording"], receiver=f["receiver"],
                             receiver_contact=f["receiver_contact"], fit_rms_m=f["fit_rms_m"]))
        min(shards, key=len).extend(rows)
    n = min(len(s) for s in shards)
    out.mkdir(parents=True, exist_ok=False)
    for k, part in enumerate(shards):
        part = part[:n]
        for index, row in enumerate(part):
            row.update(shard_id=k + 1, shot_id=index + 1)
        scene = {"balls": n, "clean_serves": [r["launch"] for r in part[:6]], "rapid_cycle": [r["launch"] for r in part[6:]],
                 "spin_rad_s_list": [r["initial_spin"] for r in part],
                 "timing": {"flight_s_list": [r["flight_duration_s"] for r in part], "pause_s_list": [r["pause_s"] for r in part]}}
        (out / f"shard-{k + 1:02d}.yaml").write_text(json.dumps(scene) + "\n")  # JSON is YAML
        (out / f"shard-{k + 1:02d}-ledger.json").write_text(json.dumps(part, indent=1, allow_nan=False) + "\n")
    design = {"version": "field-ball-paths-v1", "source": str(argv[0]), "cadence": opt["--cadence"], "shards": 6, "total_unique": 6 * n,
              "flights_in_source": len(flights), "clean_flights": len(clean), "policy_outcome_used_to_select": False,
              "scope": "Ball flights of recorded robot-vs-robot rallies, fitted with the simulator's flight and bounce model."}
    (out / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    period = [r["flight_duration_s"] + r["pause_s"] for s in shards for r in s[:n]]
    print(f"{out}: 6 x {n} launches ({len(clean)} clean of {len(flights)} flights); a ball every {sorted(period)[len(period) // 2]:.2f} s (median)")


if __name__ == "__main__":
    main()
