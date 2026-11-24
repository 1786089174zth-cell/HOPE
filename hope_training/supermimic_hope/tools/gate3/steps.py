"""Stand or step, ball by ball, in a Gate3 run of the step scenario (stdlib only).

    python3 tools/gate3/steps.py <run dir> [<run dir> ...] [--kinds tools/gate3/scenarios/steps_station058.json]

For runs of ``tools/gate3/make_step_scenario.py``: joins the scene's list of shots (which stroke each ball is
for, where it crosses the strike plane) with

* the Runner trace: the target the policy was given (lateral position from the station, side, notice);
* the plant log: how far the pelvis was from the station sideways at the strike, and whether a foot left the
  ground between the announcement and 0.2 s after the strike (more than 2 cm above its height in the idle
  seconds before the first ball) -- a step;
* the simulator's ball report: racket contact and legal return (``rejudge.py``).

Prints one row per kind of ball and one per stroke, and writes ``steps.json`` into the run directory. A policy
that tells its strokes apart steps for the step balls and for none of the stand balls.
"""

from __future__ import annotations

import bisect
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rejudge import verdict  # noqa: E402


def analyse(folder: Path, kinds: dict) -> list:
    ball = {int(r["shot_id"]): r for r in json.loads((folder / "pp_physical_ball_report.json").read_text())["rows"]}
    engage, motion = {}, None
    with (folder / "pp_runner_trace.csv").open() as stream:
        for r in csv.DictReader(stream):
            if r["mode"] != "3":
                continue
            motion = motion or int(r["wall_time_ns"])
            flight = int(float(r["frozen_flight_id"] or 0))
            if r["level"] not in ("", "0") and flight and flight not in engage:
                t = int(r["wall_time_ns"])
                notice = float(r["engage_raw_tts"])
                engage[flight] = dict(t=t, strike=t + int(notice * 1e9), notice=notice, sign=float(r["swing_sign"] or 0),
                                      target=float(r["target_y"]) - float(r["pending_station_y"]),
                                      station_y=float(r["pending_station_y"]))
    # Which ball each engagement was for: in order, the next shot whose lateral position at the plane is the
    # target's (a ball the Planner did not command leaves no engagement, so counting does not do).
    centre = -0.7625
    by_shot, pending = {}, sorted(engage.values(), key=lambda e: e["t"])
    for s in kinds["shots"]:
        if pending and abs(pending[0]["target"] + pending[0]["station_y"] - centre - s["lateral"]) < 0.06:
            by_shot[s["shot_id"]] = pending.pop(0)
    with (folder / "pp_mujoco_plant.csv").open() as stream:
        reader = csv.reader(stream)
        ix = {name: i for i, name in enumerate(next(reader))}
        rows = [r for r in reader if len(r) >= len(ix)]
    times = [int(r[ix["wall_time_ns"]]) for r in rows]
    col = lambda name: [float(r[ix[name]]) for r in rows]  # noqa: E731
    base_y, left_z, right_z = col("base_y"), col("left_foot_z"), col("right_foot_z")
    idle = [i for i, t in enumerate(times) if motion is not None and 3e9 <= t - motion <= 12e9] or list(range(len(times)))
    ground = (sorted(left_z[i] for i in idle)[len(idle) // 2], sorted(right_z[i] for i in idle)[len(idle) // 2])
    shots = []
    for s in kinds["shots"]:
        shot, b = s["shot_id"], ball.get(s["shot_id"])
        e = by_shot.get(shot)
        row = dict(shot=shot, kind=s["kind"], stroke=s["stroke"], lateral=s["lateral"], engaged=e is not None,
                   contact=bool(b and verdict(b)[0]), legal=bool(b and verdict(b)[1]))
        if e:
            a, k, z = bisect.bisect_left(times, e["t"]), bisect.bisect_left(times, e["strike"]), bisect.bisect_left(times, e["strike"] + int(0.2e9))
            k = min(k, len(times) - 1)
            lift = max(max(left_z[a:z + 1], default=ground[0]) - ground[0], max(right_z[a:z + 1], default=ground[1]) - ground[1])
            row.update(target=e["target"], side="forehand" if e["sign"] > 0 else "backhand", notice=e["notice"],
                       pelvis=base_y[k] - e["station_y"], lift=lift, stepped=lift > 0.02)
        shots.append(row)
    return shots


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    kinds_path = Path(__file__).resolve().parent / "scenarios" / "steps_station058.json"
    if "--kinds" in argv:
        i = argv.index("--kinds")
        kinds_path = Path(argv[i + 1])
        del argv[i:i + 2]
    kinds = json.loads(kinds_path.read_text())
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    for arg in argv:
        shots = analyse(Path(arg), kinds)
        print(f"{arg}: {len(shots)} balls, legal {sum(s['legal'] for s in shots)}, contact {sum(s['contact'] for s in shots)}, "
              f"engaged {sum(s['engaged'] for s in shots)}")

        def table(key, title):
            groups = {}
            for s in shots:
                groups.setdefault(key(s), []).append(s)
            print(f"  {title:22s} balls  engaged  stepped  contact  legal | pelvis at the strike   foot lift   notice | side asked")
            for name, g in groups.items():
                hit = [s for s in g if s["engaged"]]
                sides = ", ".join(f"{k} {sum(1 for s in hit if s['side'] == k)}" for k in ("forehand", "backhand") if any(s["side"] == k for s in hit))
                print(f"  {name:22s} {len(g):5d}  {len(hit):7d}  {sum(s['stepped'] for s in hit):7d}  {sum(s['contact'] for s in g):7d}  "
                      f"{sum(s['legal'] for s in g):5d} | {100 * mean([s['pelvis'] for s in hit]):+6.1f} cm            "
                      f"{100 * mean([s['lift'] for s in hit]):4.1f} cm    {mean([s['notice'] for s in hit]):.2f} s | {sides}")

        table(lambda s: s["stroke"], "stroke the ball is for")
        table(lambda s: f"{s['kind']}" if s["kind"] != "standard" else "standard (stand)", "kind (m from centre)")
        (Path(arg) / "steps.json").write_text(json.dumps(shots, indent=1) + "\n")


if __name__ == "__main__":
    main()
