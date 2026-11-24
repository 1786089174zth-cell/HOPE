"""Racket tracking at the commanded instant, robot against simulated replays, by side (stdlib only).

    python3 tools/real/tracking.py <suite dir> <real dump dir> [...] -- <supergate output dir> [...]

Strokes with more than 0.3 s of notice. Per side: position error (mean and spread per axis), racket speed minus
commanded speed, and the elevation of the racket's velocity above the commanded one, on the robot (FK from the
Runner trace, ``shots.json``) and in each simulated run of the same balls.
"""
import collections, csv, json, math, statistics as st, sys
from pathlib import Path

argv = sys.argv[1:]; cut = argv.index("--")
suite, dumps, outs = Path(argv[0]), argv[1:cut], argv[cut + 1:]
rows = [r for k in range(1, 7) for r in json.loads((suite / f"shard-{k:02d}-ledger.json").read_text()) if not r["padding"] and r["real"]["tts"] > 0.3]
shots = {(Path(d).name, i): s for d in dumps for i, s in enumerate(json.loads((Path(d) / "shots.json").read_text()))}
speed = lambda v: math.sqrt(sum(x * x for x in v))  # noqa: E731
elev = lambda v: math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))  # noqa: E731


def show(tag, G):
    for side in ("BH", "FH"):
        g = G[side]
        if not g["dp"]: continue
        m = lambda i: 1000 * st.mean(x[i] for x in g["dp"]); s = lambda i: 1000 * st.pstdev([x[i] for x in g["dp"]])  # noqa: E731
        print(f"  {tag:34s} {side} n {len(g['dp']):3d} | position error mm mean {m(0):+4.0f} {m(1):+4.0f} {m(2):+4.0f} spread {s(0):3.0f} {s(1):3.0f} {s(2):3.0f} | "
              f"speed - commanded {st.mean(g['speed']):+.2f} +- {st.pstdev(g['speed']):.2f} m/s | velocity elevation - commanded {st.mean(g['elev']):+5.1f} +- {st.pstdev(g['elev']):4.1f} deg")


G = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    s = shots[(r["real"]["session"], r["real"]["shot"])]; side = "FH" if r["real"]["swing"] > 0 else "BH"
    G[side]["dp"].append(s["racket_err"]); G[side]["speed"].append(speed(s["racket_v"]) - speed(s["target_v"])); G[side]["elev"].append(elev(s["racket_v"]) - elev(s["target_v"]))
show("robot", G)
for out in outs:
    kin = {(k + 1, int(r["shot"])): r for k in range(6) for r in csv.DictReader(open(f"{out}/env-{k}-kinematics.csv")) if r["event"] == "planned"}
    G = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        k = kin.get((r["shard_id"], r["shot_id"]))
        if not k: continue
        side = "FH" if r["real"]["swing"] > 0 else "BH"
        v = [float(k[f"racket_v_{a}"]) for a in "xyz"]; tv = [float(k[f"target_v_{a}"]) for a in "xyz"]
        G[side]["dp"].append([float(k[f"dp_{a}"]) for a in "xyz"]); G[side]["speed"].append(speed(v) - speed(tv)); G[side]["elev"].append(elev(v) - elev(tv))
    show(Path(out).name.split("gripper")[-1] or "MJCF actuators, scale 1", G)
