"""Write a Gate3 scenario with balls for the step strokes: wide right, wide left, and stand-zone balls between
them (stdlib only).

    python3 tools/gate3/make_step_scenario.py [--out tools/gate3/scenarios/steps_station058.yaml] [--gap 1.7] [--balls 100]

Gate3's ``fixed-home`` scene and ``make_scenario.py`` serve six lateral positions 0.33 m right to 0.15 m left of
the centre line: every ball is a stand stroke. Here the standard serve (launch 0.49 m above the table, 3.0 m/s
towards the robot, 2.2 m/s up) is aimed so that it crosses the strike plane at a chosen lateral position:

* step forehand: 0.70, 0.76, 0.82 m right of the centre line (the table's edge is at 0.7625 m; the last two leave
  the table over its side after the bounce, the serve is launched at an angle for that);
* step backhand: 0.47, 0.53, 0.59 m left;
* stand strokes: 0.50 and 0.15 m right, 0.20 and 0.38 m left.

After the six standard clean serves the kinds come in a fixed shuffled order, six in ten of them step balls,
strikes ``--gap`` seconds apart (1.7 s: the pace of the field rallies). The flight is the simulator's own model
(``gate3_ball_contact_model.h``: quadratic drag, Magnus, capped-slip bounce), in three dimensions.

``steps_station058.json`` next to the scene lists each shot's kind and lateral position at the plane;
``tools/gate3/steps.py <run dir>`` joins it with a result (did the robot step, where was the pelvis, was the
return legal).
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
LAUNCH_X, WIDTH, CENTRE_Y = 2.4, 1.525, -0.7625
STANDARD = (0.49, -3.0, 2.2)  # launch height above the table, vx, vz of the fixed-home serve
KINDS = {"step_forehand": (-0.70, -0.76, -0.82), "step_backhand": (0.47, 0.53, 0.59), "stand": (-0.50, -0.15, 0.20, 0.38)}


def fly(y0, vy, plane_x, dt=0.0005, serve=STANDARD):
    """The serve (the standard one unless ``serve`` gives height above the table, vx, vz) launched at lateral
    ``y0`` with lateral speed ``vy``: what crosses ``x = plane_x``. None unless it clears the net and bounces
    once, on the robot's half and at least 3 cm inside the sides."""
    z0, vx, vz = serve
    p, v, w = [LAUNCH_X, y0, TABLE_Z + z0], [vx, vy, vz], [0.0, 0.0, 0.0]
    t, bounce, t_net = 0.0, None, None
    while t < 4.0:
        speed = math.sqrt(sum(c * c for c in v))
        cross = (w[1] * v[2] - w[2] * v[1], w[2] * v[0] - w[0] * v[2], w[0] * v[1] - w[1] * v[0])
        a = [-DRAG * speed * v[i] + MAGNUS * cross[i] for i in range(3)]
        a[2] -= 9.81
        prev = list(p)
        v = [v[i] + a[i] * dt for i in range(3)]
        p = [p[i] + v[i] * dt for i in range(3)]
        t += dt
        if prev[0] > NET_X >= p[0]:
            if p[2] - RADIUS - NET_TOP < 0.03:
                return None
            t_net = t
        if p[2] <= TABLE_Z + RADIUS and v[2] < 0:
            if bounce is not None or not 0.10 < p[0] < 1.25 or not -WIDTH + 0.03 < p[1] < -0.03:
                return None
            # contact point velocity: v + w x (-R z) = (vx - R wy, vy + R wx)
            slip = (v[0] - RADIUS * w[1], v[1] + RADIUS * w[0])
            size = math.hypot(*slip)
            impulse = min(GAIN * size, MU_CAP * (1 + REST_V) * abs(v[2]))
            dv = (-impulse * slip[0] / (size + 1e-12), -impulse * slip[1] / (size + 1e-12))
            v[0], v[1] = v[0] + dv[0], v[1] + dv[1]
            w[1] -= dv[0] / (INERTIA * RADIUS)
            w[0] += dv[1] / (INERTIA * RADIUS)
            v[2], p[2] = -REST_V * v[2], TABLE_Z + RADIUS
            bounce = (p[0], p[1], t)
        if prev[0] > plane_x >= p[0]:
            if bounce is None:
                return None
            return dict(y=p[1], strike_z=p[2], speed=math.sqrt(sum(c * c for c in v)), vz=v[2], t_strike=t,
                        bounce_x=bounce[0], bounce_y=bounce[1], after_bounce_s=t - bounce[2], after_net_s=t - t_net)
    return None


def aim(lateral, plane_x, serve=STANDARD):
    """Launch (y0, vy) of a serve that crosses the plane ``lateral`` m from the centre line: straight if its bounce
    is on the table, else from as far outside as keeps the bounce on it."""
    target = CENTRE_Y + lateral
    for inset in (0.0, 0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42):
        y0 = min(max(target, -WIDTH + 0.06 + inset), -0.06 - inset) if inset else target
        lo, hi = -1.5, 1.5
        r = None
        for _ in range(40):  # the crossing moves monotonically with vy
            vy = 0.5 * (lo + hi)
            r = fly(y0, vy, plane_x, serve=serve)
            # without a clean flight, move towards a straight serve from this side
            y = r["y"] if r else (y0 + (1.0 if vy > 0 else -1.0) * 9.0)
            if y > target:
                hi = vy
            else:
                lo = vy
        if r is not None and abs(r["y"] - target) < 0.004:
            return [round(y0, 4), round(vy, 4)], r
    return None


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(here / "scenarios" / "steps_station058.yaml"))
    ap.add_argument("--gap", type=float, default=1.7, help="seconds between strikes")
    ap.add_argument("--balls", type=int, default=100)
    ap.add_argument("--step-share", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--station-x", type=float, default=-0.58)
    ap.add_argument("--strike-x", type=float, default=0.0, help="world x of the strike plane (the table edge with the station 0.58 m behind it)")
    args = ap.parse_args()

    kinds = {}
    for name, laterals in KINDS.items():
        for lateral in laterals:
            found = aim(lateral, args.strike_x)
            if found is None:
                print(f"{name} {lateral:+.2f} m: no clean serve")
                continue
            (y0, vy), r = found
            kinds[f"{name}_{lateral:+.2f}"] = dict(stroke=name, lateral=lateral, launch=[LAUNCH_X, y0, STANDARD[0], STANDARD[1], vy, STANDARD[2]],
                                                   **{k: round(v, 4) for k, v in r.items()})
            print(f"{name:13s} {lateral:+.2f} m: launch y {y0:+.3f} vy {vy:+.3f} -> crosses at y {r['y'] - CENTRE_Y:+.3f} from the centre, "
                  f"{r['strike_z']:.3f} m high, {r['speed']:.2f} m/s, bounce at ({r['bounce_x']:.2f}, {r['bounce_y'] - CENTRE_Y:+.2f}), "
                  f"{r['t_strike']:.2f} s after launch")
    standard = fly(-1.0225, 0.0, args.strike_x)
    clean = [[LAUNCH_X, y, STANDARD[0], STANDARD[1], 0.0, STANDARD[2]] for y in (-1.0225, -0.7125, -1.0575, -0.6575, -1.0925, -0.6125)]
    rng = random.Random(args.seed)
    steps = [k for k, v in kinds.items() if v["stroke"] != "stand"]
    stands = [k for k, v in kinds.items() if v["stroke"] == "stand"]
    balls = args.balls + args.balls % 2
    names = ["standard"] * 6
    while len(names) < balls:
        block = steps * 2 + stands * 2
        rng.shuffle(block)
        want = [k for k in block if (k in steps) == (rng.random() < args.step_share)] or block
        names += want
    names = names[:balls]
    serves = clean + [kinds[n]["launch"] for n in names[6:]]
    arrive = [standard["t_strike"]] * 6 + [kinds[n]["t_strike"] for n in names[6:]]
    flight = [round(args.gap + arrive[i] - arrive[i + 1], 3) for i in range(balls - 1)] + [2.5]
    lines = ["schema_version: 1", "family: fixed_home", f"station_x: {args.station_x}",
             "description: 'Step strokes: six standard serves, then the standard serve aimed 0.70-0.82 m right and 0.47-0.59 m left of the "
             f"centre line among stand-zone balls, strikes {args.gap} s apart (tools/gate3/make_step_scenario.py). Simulator truth.'",
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
        shots=[dict(shot_id=i + 1, kind=names[i], stroke=kinds[names[i]]["stroke"] if names[i] in kinds else "stand", lateral=lateral(i))
               for i in range(balls)]), indent=1) + "\n")
    count = {s: sum(1 for n in names[6:] if n in kinds and kinds[n]["stroke"] == s) for s in KINDS}
    print(f"wrote {out} ({balls} balls: 6 standard, {count}; about {20 + sum(flight):.0f} s) and {out.with_suffix('.json').name}; "
          f"run with --scenario {out} --balls {balls}")


if __name__ == "__main__":
    main()
