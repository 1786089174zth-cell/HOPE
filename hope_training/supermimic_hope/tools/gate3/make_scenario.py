"""Write a Gate3 scenario whose serves arrive low / mid / high and slow / mid / fast (stdlib only).

    python3 tools/gate3/make_scenario.py [--out tools/gate3/scenarios/ball_paths.yaml] [--gap 1.6] [--repeats 2]

Gate3's own ``fixed-home`` scene launches one serve at six lateral positions: every ball is struck
0.81-0.86 m high after the same flight. Here the serve is solved for a height and a horizontal
speed of the ball at the strike plane (the policy's deploy plane, 0.465 m ahead of the station),
with the simulator's own flight and table-bounce model (``gate3_ball_contact_model.h``: quadratic
drag, Magnus, capped-slip bounce). Nine serve kinds x six lateral positions, in a fixed shuffled
order after the six standard clean serves; launches are timed so strikes are ``--gap`` seconds apart.

The file is a normal ``run_gate3.sh --scenario <path> --balls <n>`` scene; ``ball_paths.json`` next
to it lists what each serve was solved for (``tools/gate3/by_serve.py`` joins it with a result).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

TABLE_Z, RADIUS, NET_X, NET_TOP = 0.76, 0.02, 1.37, 0.9125
DRAG, MAGNUS = 0.12453915600992894, 0.00399279130564425
REST_V, GAIN, MU_CAP, INERTIA = 0.9373319241133231, 0.41642144985036184, 0.218520906499864, 2.0 / 3.0
STRIKE_X = -0.035  # world x of the deploy plane: station -0.50 + 0.465
LAUNCH_X = 2.4
LATERAL = (-1.0925, -1.0575, -1.0225, -0.7125, -0.6575, -0.6125)  # the fixed-home scene's six
STANDARD = (0.49, -3.0, 2.2)  # launch height above the table, vx, vz of the fixed-home serve
HEIGHTS = {"low": 0.84, "mid": 1.00, "high": 1.18}
SPEEDS = {"slow": 1.00, "medium": 1.30, "fast": 1.80}  # horizontal ball speed at the strike plane


def fly(z0, vx, vz, dt=0.0005):
    """One serve from (LAUNCH_X, table + z0) to the strike plane. Returns None if it is not a clean
    serve (net, no bounce on the robot's half, two bounces), else what the robot faces."""
    x, z, wy = LAUNCH_X, TABLE_Z + z0, 0.0
    t, bounce, net_clearance = 0.0, None, None
    while t < 4.0:
        speed = math.hypot(vx, vz)
        # drag -k |v| v and Magnus k (w x v) with w = (0, wy, 0): (wy vz, 0, -wy vx)
        ax = -DRAG * speed * vx + MAGNUS * wy * vz
        az = -9.81 - DRAG * speed * vz - MAGNUS * wy * vx
        x_prev, z_prev = x, z
        vx, vz = vx + ax * dt, vz + az * dt
        x, z, t = x + vx * dt, z + vz * dt, t + dt
        if x_prev > NET_X >= x:
            net_clearance = z - RADIUS - NET_TOP
            if net_clearance < 0.03:
                return None
        if z <= TABLE_Z + RADIUS and vz < 0:
            if bounce is not None or not 0.10 < x < 1.25:
                return None
            slip = vx - RADIUS * wy
            impulse = min(GAIN * abs(slip), MU_CAP * (1 + REST_V) * abs(vz))
            dv = -math.copysign(impulse, slip)
            vx, wy = vx + dv, wy - dv / (INERTIA * RADIUS)
            vz, z = -REST_V * vz, TABLE_Z + RADIUS
            bounce = dict(t=t, x=x, vx=vx, vz=vz)
        if x <= STRIKE_X:
            if bounce is None:
                return None
            return dict(strike_z=z, speed_x=-vx, vz=vz, t_strike=t, t_bounce=bounce["t"], bounce_x=bounce["x"],
                        after_bounce_s=t - bounce["t"], net_clearance=net_clearance)
    return None


def solve(height, speed):
    """The launch (z0, vx, vz) whose ball crosses the strike plane at ``height`` with horizontal
    speed ``speed``: coarse grid, then coordinate refinement. Prefers a long flight after the bounce."""
    best = None
    for z0 in [0.15 + 0.05 * i for i in range(14)]:
        for vx in [-1.6 - 0.2 * i for i in range(30)]:
            for vz in [-1.5 + 0.25 * i for i in range(24)]:
                r = fly(z0, vx, vz, dt=0.002)
                if r is None or r["after_bounce_s"] < 0.30:
                    continue
                cost = ((r["strike_z"] - height) / 0.02) ** 2 + ((r["speed_x"] - speed) / 0.05) ** 2
                if best is None or cost < best[0]:
                    best = (cost, z0, vx, vz)
    if best is None:
        return None
    _, z0, vx, vz = best
    step = [0.02, 0.1, 0.1]
    cost = lambda r: 1e9 if r is None or r["after_bounce_s"] < 0.30 else (  # noqa: E731
        ((r["strike_z"] - height) / 0.02) ** 2 + ((r["speed_x"] - speed) / 0.05) ** 2)
    p = [z0, vx, vz]
    c = cost(fly(*p))
    for _ in range(60):
        improved = False
        for i in range(3):
            for sign in (1, -1):
                q = list(p)
                q[i] += sign * step[i]
                if not 0.10 <= q[0] <= 0.90:
                    continue
                cq = cost(fly(*q))
                if cq < c:
                    p, c, improved = q, cq, True
        if not improved:
            step = [s * 0.5 for s in step]
    r = fly(*p)
    if r is None or abs(r["strike_z"] - height) > 0.02 or abs(r["speed_x"] - speed) > 0.06:
        return None
    return [round(v, 4) for v in p], r


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(here / "scenarios" / "ball_paths.yaml"))
    ap.add_argument("--gap", type=float, default=1.6, help="seconds between strikes (the fixed-home rapid cycle is 1.15)")
    ap.add_argument("--repeats", type=int, default=2, help="times each serve kind visits each lateral position")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--station-x", type=float, default=None,
                    help="write station_x into the scene (-0.58: robot 0.58 m behind the table edge; needs the patched run_gate3.py)")
    ap.add_argument("--strike-x", type=float, default=STRIKE_X,
                    help="world x of the strike plane the serves are solved for (station -0.50 + plane; 0.08 for the 0.58 plane)")
    args = ap.parse_args()
    globals()["STRIKE_X"] = args.strike_x

    standard = fly(*STANDARD)
    print(f"standard fixed-home serve: strike height {standard['strike_z']:.3f} m, ball speed {standard['speed_x']:.2f} m/s, "
          f"bounce at x {standard['bounce_x']:.3f}, {standard['t_strike']:.2f} s after launch")
    kinds = {}
    for hname, height in HEIGHTS.items():
        for sname, speed in SPEEDS.items():
            found = solve(height, speed)
            if found is None:
                print(f"{hname:5s} {sname:7s}: no clean serve reaches {height} m at {speed} m/s")
                continue
            launch, r = found
            kinds[f"{hname}_{sname}"] = dict(launch=launch, **{k: round(v, 4) for k, v in r.items()})
            print(f"{hname:5s} {sname:7s}: launch z {launch[0]:.3f} vx {launch[1]:+.2f} vz {launch[2]:+.2f} -> strike {r['strike_z']:.3f} m, "
                  f"ball vx {r['speed_x']:.2f} vz {r['vz']:+.2f} m/s, bounce x {r['bounce_x']:.2f}, {r['after_bounce_s']:.2f} s after the "
                  f"bounce, {r['t_strike']:.2f} s after launch, net clearance {r['net_clearance']:.2f} m")
    row = lambda launch, y: [LAUNCH_X, y, launch[0], launch[1], 0.0, launch[2]]  # noqa: E731
    clean = [row(STANDARD, y) for y in (-1.0225, -0.7125, -1.0575, -0.6575, -1.0925, -0.6125)]
    rng = random.Random(args.seed)
    cycle = []
    for _ in range(args.repeats):
        block = [(name, y) for name in kinds for y in LATERAL]
        rng.shuffle(block)
        cycle += block
    balls = 6 + len(cycle)
    balls += balls % 2
    names = ["standard"] * 6 + [cycle[i % len(cycle)][0] for i in range(balls - 6)]
    serves = clean + [row(kinds[cycle[i % len(cycle)][0]]["launch"], cycle[i % len(cycle)][1]) for i in range(balls - 6)]
    arrive = [standard["t_strike"]] * 6 + [kinds[n]["t_strike"] for n in names[6:]]
    # Launch i + 1 follows launch i by flight_i + pause_i; strikes are then ``gap`` apart.
    flight = [round(args.gap + arrive[i] - arrive[i + 1], 3) for i in range(balls - 1)] + [2.5]
    if min(flight) <= 0.05:
        raise SystemExit("--gap is too short for these serves")
    lines = ["schema_version: 1", "family: fixed_home"] + ([f"station_x: {args.station_x}"] if args.station_x is not None else []) + [
             "description: 'Ball paths: six standard serves, then low/mid/high x slow/medium/fast serves at six lateral "
             f"positions, strikes {args.gap} s apart (tools/gate3/make_scenario.py). Simulator truth.'",
             f"balls: {balls}", "clean_serves:"]
    lines += [f"- {json.dumps(s)}" for s in clean]
    lines += ["rapid_cycle:"] + [f"- {json.dumps(s)}" for s in serves[6:]]
    lines += ["timing:", "  clean_flight_s: 2.5", "  rapid_flight_s: 1.15", "  final_flight_s: 2.5", "  clean_pause_s: 1.0",
              "  rapid_pause_s: 0.0", f"  flight_s_list: {json.dumps(flight)}", f"  pause_s_list: {json.dumps([0.0] * balls)}"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(dict(
        balls=balls, gap_s=args.gap, strike_plane_x=STRIKE_X, standard=dict(launch=list(STANDARD), **standard),
        kinds=kinds, shots=[dict(shot_id=i + 1, kind=names[i], y=serves[i][1]) for i in range(balls)]), indent=1) + "\n")
    print(f"wrote {out} ({balls} balls, about {20 + sum(flight):.0f} s) and {out.with_suffix('.json').name}; "
          f"run with --scenario {out} --balls {balls}")


if __name__ == "__main__":
    main()
