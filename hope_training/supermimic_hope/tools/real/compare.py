"""Real strokes against their replay in Supergate, ball by ball (stdlib only).

    python3 tools/real/compare.py <suite dir of make_suite.py> <supergate output dir> [...]

Joins each ledger row of the real-session suite (``real``: the stroke on the robot) with the same shot of the
simulated run and prints: the outcome table real x simulated, how well the replay reproduces the real command,
where the returns land, and the racket error at the commanded instant on the robot and in the simulation.
"""

from __future__ import annotations

import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    suite = Path(argv[0])
    rows = [r for k in range(1, 7) for r in json.loads((suite / f"shard-{k:02d}-ledger.json").read_text()) if not r["padding"]]
    for out in map(Path, argv[1:]):
        sim, kin = {}, {}
        for k in range(6):
            for r in csv.DictReader((out / f"env-{k}-results.csv").open()):
                sim[(k + 1, int(r["shot"]))] = r
            for r in csv.DictReader((out / f"env-{k}-kinematics.csv").open()):
                kin[(k + 1, int(r["shot"]), r["event"])] = r
        print(f"== {out.name}: {len(rows)} real strokes")
        table = collections.defaultdict(collections.Counter)
        for r in rows:
            s = sim[(r["shard_id"], r["shot_id"])]
            verdict = "legal" if s["legal"] == "1" else "no plan" if s["valid_plans"] == "0" else "no contact" if s["contacts"] == "0" else "net" if s["nets"] != "0" else "off table"
            table[r["real"]["outcome"]][verdict] += 1
        cols = ["legal", "net", "off table", "no contact", "no plan"]
        print(f"  {'robot':8s} {'n':>4s} | simulation: " + " ".join(f"{c:>10s}" for c in cols))
        for name in ("landed", "short", "net", "long", "wide", "miss"):
            c = table[name]
            if sum(c.values()):
                print(f"  {name:8s} {sum(c.values()):4d} |             " + " ".join(f"{c[k]:10d}" for k in cols))
        total = collections.Counter()
        for c in table.values():
            total.update(c)
        real_landed = sum(table["landed"].values())
        print(f"  legal returns: robot {real_landed}/{len(rows)} = {100 * real_landed / len(rows):.1f}%, simulation {total['legal']}/{len(rows)} = {100 * total['legal'] / len(rows):.1f}%")
        # the command: simulated Planner against the real one, same ball
        dp, dv, lead = [], [], []
        land, err_real, err_sim, speed = [], [], [], []
        for r in rows:
            key = (r["shard_id"], r["shot_id"])
            k = kin.get(key + ("planned",))
            if not k:
                continue
            real = r["real"]
            dp.append([float(k[f"target_p_{a}"]) - real["target"][i] for i, a in enumerate("xyz")])
            dv.append([float(k[f"target_v_{a}"]) - real["target_v"][i] for i, a in enumerate("xyz")])
            lead.append(float(sim[key]["lead"]) - real["tts"])
            err_sim.append(float(k["p_error_m"]))
            err_real.append(sum(v * v for v in real["racket_err"]) ** 0.5)
            speed.append((sum(float(k[f"target_v_{a}"]) ** 2 for a in "xyz") ** 0.5, sum(v * v for v in real["target_v"]) ** 0.5))
            if real.get("land") and sim[key]["contacts"] != "0" and sim[key]["landing_x"] not in ("", "nan"):
                land.append((real["land"][0], float(sim[key]["landing_x"]), real["net_clear"]))
        med = lambda v: st.median(v)  # noqa: E731
        for name, d in (("target position", dp), ("racket velocity", dv)):
            print(f"  simulated command - real command, {name}: median " + " ".join(f"{med([x[i] for x in d]):+.3f}" for i in range(3)) +
                  " | spread (IQR/2) " + " ".join(f"{(st.quantiles([x[i] for x in d], n=4)[2] - st.quantiles([x[i] for x in d], n=4)[0]) / 2:.3f}" for i in range(3)))
        print(f"  commanded racket speed: simulation {med([a for a, _ in speed]):.2f} m/s, robot {med([b for _, b in speed]):.2f} m/s; notice: simulation - robot {med(lead):+.3f} s")
        print(f"  racket position error at the commanded instant: robot (FK) median {100 * med(err_real):.1f} cm, simulation {100 * med(err_sim):.1f} cm")
        if land:
            print(f"  landing x of the return (net 1.37, far edge 2.74, aim 2.055), {len(land)} balls struck in both: robot median {med([a for a, _, _ in land]):.2f} m, "
                  f"simulation {med([b for _, b, _ in land]):.2f} m; robot - simulation median {med([a - b for a, b, _ in land]):+.2f} m, "
                  f"robot shorter on {sum(a < b for a, b, _ in land)}/{len(land)}")


if __name__ == "__main__":
    main()
