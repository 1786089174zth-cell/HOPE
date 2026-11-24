"""Write a Gate3 scenario with fast serves: balls the Planner can announce only 0.24-0.33 s ahead (stdlib only).

    python3 tools/gate3/make_fast_scenario.py [--out tools/gate3/scenarios/fast_station058.yaml] [--gap 1.7] [--balls 100]

Gate3's standard serve (3.0 m/s towards the robot, 2.2 m/s up) reaches the strike plane 0.6 s after the net: every
stroke gets the Planner's full 0.39 s of notice and none of its motors comes near a limit (effort peak 0.5 of the
limit). The Planner commits after the net, so the notice of a fast ball is its time from the net to the plane.
This scene serves flat fast balls among standard ones:

    serve (height above the table, vx, vz)     net -> plane    at the plane
    0.30 m, -8.0 m/s, +0.3 m/s                 about 0.23 s    about 5 m/s
    0.30 m, -7.0 m/s, +0.5 m/s                 0.26 s          4.4 m/s, 0.95 m high
    0.49 m, -6.5 m/s,  0.0 m/s                 0.30 s          3.8 m/s, 1.03 m
    0.40 m, -6.0 m/s, +0.5 m/s                 0.32 s          3.5 m/s, 1.02 m

each aimed 0.55 and 0.42 m right of the centre line (forehands; the second is next to the side split at 0.34 m),
0.25 m right (crossover backhand), on the centre line and 0.25 m left. After the six standard clean serves, fast
balls and standard stand-zone balls alternate in a fixed shuffled order, strikes ``--gap`` seconds apart.

``fast_station058.json`` lists each shot's kind; ``tools/gate3/steps.py <run dir> --kinds <that json>`` gives
returns and notice per kind, ``tools/gate3/limits.py <run dir>`` what the strokes ask of the motors.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from make_step_scenario import CENTRE_Y, LAUNCH_X, STANDARD, aim, fly

SERVES = ((0.30, -8.0, 0.3), (0.30, -7.0, 0.5), (0.49, -6.5, 0.0), (0.40, -6.0, 0.5))
LATERALS = (-0.55, -0.42, -0.25, 0.0, 0.25)
STAND = (-0.50, -0.15, 0.20)
SPLIT = -0.34


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(here / "scenarios" / "fast_station058.yaml"))
    ap.add_argument("--gap", type=float, default=1.7, help="seconds between strikes")
    ap.add_argument("--balls", type=int, default=100)
    ap.add_argument("--fast-share", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--station-x", type=float, default=-0.58)
    ap.add_argument("--strike-x", type=float, default=0.0)
    args = ap.parse_args()

    kinds = {}
    for serve in SERVES:
        for lateral in LATERALS:
            found = aim(lateral, args.strike_x, serve=serve)
            if found is None:
                print(f"serve {serve} at {lateral:+.2f} m: no clean flight")
                continue
            (y0, vy), r = found
            side = "forehand" if lateral < SPLIT else "backhand"
            name = f"fast{r['after_net_s']:.2f}s_{lateral:+.2f}"
            kinds[name] = dict(stroke=f"{side} {r['after_net_s']:.2f} s", lateral=lateral, launch=[LAUNCH_X, y0, serve[0], serve[1], vy, serve[2]],
                               **{k: round(v, 4) for k, v in r.items()})
            print(f"{name:22s}: {side}, net -> plane {r['after_net_s']:.3f} s, {r['strike_z']:.3f} m high, {r['speed']:.2f} m/s (vz {r['vz']:+.2f}), "
                  f"bounce at x {r['bounce_x']:.2f}")
    for lateral in STAND:
        (y0, vy), r = aim(lateral, args.strike_x)
        kinds[f"standard_{lateral:+.2f}"] = dict(stroke="standard", lateral=lateral, launch=[LAUNCH_X, y0, STANDARD[0], STANDARD[1], vy, STANDARD[2]],
                                                 **{k: round(v, 4) for k, v in r.items()})
    standard = fly(-1.0225, 0.0, args.strike_x)
    clean = [[LAUNCH_X, y, STANDARD[0], STANDARD[1], 0.0, STANDARD[2]] for y in (-1.0225, -0.7125, -1.0575, -0.6575, -1.0925, -0.6125)]
    rng = random.Random(args.seed)
    fast = [k for k, v in kinds.items() if v["stroke"] != "standard"]
    slow = [k for k, v in kinds.items() if v["stroke"] == "standard"]
    balls = args.balls + args.balls % 2
    names = ["standard"] * 6
    pool = []
    while len(names) < balls:
        if not pool:
            pool = fast[:]
            rng.shuffle(pool)
        names.append(pool.pop() if rng.random() < args.fast_share else rng.choice(slow))
    serves = clean + [kinds[n]["launch"] for n in names[6:]]
    arrive = [standard["t_strike"]] * 6 + [kinds[n]["t_strike"] for n in names[6:]]
    flight = [round(args.gap + arrive[i] - arrive[i + 1], 3) for i in range(balls - 1)] + [2.5]
    lines = ["schema_version: 1", "family: fixed_home", f"station_x: {args.station_x}",
             "description: 'Fast serves: six standard serves, then flat serves at 6-8 m/s (0.23-0.32 s from the net to the strike plane) among "
             f"standard stand-zone balls, strikes {args.gap} s apart (tools/gate3/make_fast_scenario.py). Simulator truth.'",
             f"balls: {balls}", "clean_serves:"]
    lines += [f"- {json.dumps(s)}" for s in clean]
    lines += ["rapid_cycle:"] + [f"- {json.dumps(s)}" for s in serves[6:]]
    lines += ["timing:", "  clean_flight_s: 2.5", "  rapid_flight_s: 1.15", "  final_flight_s: 2.5", "  clean_pause_s: 1.0",
              "  rapid_pause_s: 0.0", f"  flight_s_list: {json.dumps(flight)}", f"  pause_s_list: {json.dumps([0.0] * balls)}"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    lateral = lambda i: kinds[names[i]]["lateral"] if names[i] in kinds else round(serves[i][1] - CENTRE_Y, 4)  # noqa: E731
    out.with_suffix(".json").write_text(json.dumps(dict(
        balls=balls, gap_s=args.gap, strike_plane_x=args.strike_x, kinds=kinds,
        shots=[dict(shot_id=i + 1, kind=names[i], stroke=kinds[names[i]]["stroke"] if names[i] in kinds else "standard", lateral=lateral(i))
               for i in range(balls)]), indent=1) + "\n")
    count = sum(1 for n in names[6:] if n in fast)
    print(f"wrote {out} ({balls} balls: 6 standard clean, {count} fast, {balls - 6 - count} standard; min flight gap {min(flight):.2f} s) and "
          f"{out.with_suffix('.json').name}; run with --scenario {out} --balls {balls}")


if __name__ == "__main__":
    main()
