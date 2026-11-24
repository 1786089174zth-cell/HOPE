"""Fresh physical launches: over the ranges of a Supergate suite, or (``--general``) over the whole reachable
window without reference to any suite (numpy only).

    .venv/bin/python hope_training/supermimic_hope/tools/supergate/make_launches.py --tree <supergate tree> \
        --count 9600 --seed 1 --out <launches.json>

Supergate's frozen suites are the test. To train on its kind of ball without training on its balls, each
launch of the seed suites (``data/main`` and ``data/outer`` by default) is displaced (origin, velocity, spin),
flown with Supergate's own flight and table-bounce model (``original/flight_kernel.py``, vectorised here) to
the nominal contact plane, and kept when it is a legal one-bounce ball that crosses the plane inside the
suite's envelope (lateral position, height, speed, horizontal angle, net clearance). The result is a JSON
list for Supergate's ``freeze_dataset.py`` (``launch``, ``initial_spin``, ``flight_duration_s``, ``pause_s``)
with the predicted contact state and spin of each ball.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DRAG, MAGNUS = 0.12453915600992894, 0.00399279130564425
FRICTION, NORMAL_SHARE, RESTITUTION = 0.41642144985036184, 0.218520906499864, 0.9373319241133231
PLANE, HOME_Y, TABLE_Z = 0.07, -0.7625, 0.76


def accel(v, w):
    a = -DRAG * np.linalg.norm(v, axis=-1, keepdims=True) * v + MAGNUS * np.cross(w, v)
    a[:, 2] -= 9.81
    return a


def flight(initial, spin, dt=0.001, steps=4000, plane=PLANE, times=False):
    """Vectorised ``flight_kernel.flight``: state at the contact plane, spin there, flight time, bounces,
    the bounce point and the (y, z) at the net, for (N, 6) launches (table-relative z). With ``times`` also the
    time of the net crossing and of the (last) bounce."""
    s, w = initial.astype(float).copy(), spin.astype(float).copy()
    n = len(s)
    done, t = np.zeros(n, bool), np.zeros(n)
    out, out_w = np.full((n, 6), np.nan), np.full((n, 3), np.nan)
    bounces, bounce, net = np.zeros(n, int), np.full((n, 2), -10.0), np.full((n, 2), -10.0)
    t_out, t_net, t_bounce = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    for _ in range(steps):
        live = ~done
        if not live.any():
            break
        prev = s.copy()
        v = s[:, 3:]
        a = accel(v, w)
        vb = v + 0.5 * dt * a
        b = accel(vb, w)
        vc = v + 0.5 * dt * b
        c = accel(vc, w)
        vd = v + dt * c
        d = accel(vd, w)
        s = np.concatenate((s[:, :3] + dt * (v + 2 * vb + 2 * vc + vd) / 6, v + dt * (a + 2 * b + 2 * c + d) / 6), 1)
        t = t + dt
        over = live & (prev[:, 0] >= 1.37) & (s[:, 0] < 1.37)
        f = ((prev[:, 0] - 1.37) / (prev[:, 0] - s[:, 0] + 1e-12))[:, None]
        net = np.where(over[:, None], prev[:, 1:3] + f * (s[:, 1:3] - prev[:, 1:3]), net)
        t_net = np.where(over, t - dt * (1 - f[:, 0]), t_net)
        hit = live & (s[:, 2] <= 0.0201) & (prev[:, 2] > 0.0201) & (s[:, 5] < 0) & (s[:, 0] >= 0) & (s[:, 0] <= 2.74) & (
            s[:, 1] >= -1.525) & (s[:, 1] <= 0)
        if hit.any():
            vv = s[hit, 3:]
            u = vv + np.cross(w[hit], np.array([0.0, 0.0, -0.02]))
            u[:, 2] = 0
            size = np.linalg.norm(u, axis=-1)
            dv = -np.minimum(FRICTION * size, NORMAL_SHARE * (1 + RESTITUTION) * np.abs(vv[:, 2]))[:, None] * u / (size[:, None] + 1e-12)
            vv = vv + dv
            w[hit] -= np.cross(np.array([0.0, 0.0, 1.0]), dv) / ((2 / 3) * 0.02)
            vv[:, 2] = -vv[:, 2] * RESTITUTION
            s[hit, 3:] = vv
            s[hit, 2] = 0.0201
            bounce[hit] = s[hit, :2]
            t_bounce[hit] = t[hit]
            bounces[hit] += 1
        at = live & (s[:, 0] <= plane)
        f = ((prev[:, 0] - plane) / (prev[:, 0] - s[:, 0] + 1e-12))[:, None]
        out = np.where(at[:, None], prev + f * (s - prev), out)
        out_w = np.where(at[:, None], w, out_w)
        done = done | at | (s[:, 2] < -0.8)
        t_out = np.where(at, t, t_out)
    return (out, out_w, t_out, bounces, bounce, net) + ((t_net, t_bounce) if times else ())


# Coverage cells of the general generator, at the strike plane: lateral position from HOME, height above the
# floor, speed, and whether the ball is still rising.
LATERAL = (-0.70, -0.50, -0.34, -0.18, -0.02, 0.14, 0.30, 0.46)
HEIGHT = (0.80, 0.92, 1.04, 1.16, 1.28, 1.47)
SPEED = (1.5, 2.2, 3.0, 3.8, 4.6, 5.6)


def general(count, seed, plane, lateral=LATERAL):
    """Launches that are not derived from any suite: any legal one-bounce ball that crosses the strike plane
    within the robot's reach. Origins anywhere over or just beyond the far half, aimed at a random point of the
    plane, with random speed, loft and spin; kept by the same physical conditions as a suite ball, then drawn
    evenly over the coverage cells so that rare kinds of ball (fast and high, wide and low) are as frequent
    as common ones. ``lateral`` replaces the lateral cell edges (the default window is what the stand strokes
    reach; the step strokes' zones lie outside it). Returns the arrays of ``main`` and the number of balls tried."""
    LATERAL = tuple(lateral)  # noqa: N806 (shadows the module's default window)
    rng = np.random.default_rng(seed)
    cells, tried = {}, 0
    per_cell = int(np.ceil(1.6 * count / ((len(LATERAL) - 1) * (len(HEIGHT) - 1) * (len(SPEED) - 1) * 2)))
    for _ in range(40):
        n = 60000
        tried += n
        launch = np.empty((n, 6))
        launch[:, 0], launch[:, 1], launch[:, 2] = rng.uniform(1.85, 2.9, n), rng.uniform(-1.47, -0.06, n), rng.uniform(0.04, 1.3, n)
        aim_y = HOME_Y + rng.uniform(LATERAL[0] - 0.05, LATERAL[-1] + 0.05, n)
        heading = np.arctan2(aim_y - launch[:, 1], plane - launch[:, 0]) + rng.uniform(-0.06, 0.06, n)
        vh = rng.uniform(2.3, 9.5, n)
        launch[:, 3], launch[:, 4], launch[:, 5] = vh * np.cos(heading), vh * np.sin(heading), rng.uniform(-4.5, 4.8, n)
        top = rng.uniform(-70.0, 90.0, n)  # about the horizontal axis across the flight: positive is topspin
        spin = np.stack((np.sin(heading) * top, -np.cos(heading) * top, rng.uniform(-30.0, 30.0, n)), 1) + rng.uniform(-12.0, 12.0, (n, 3))
        # coarse pass (3 ms steps) to find the candidates, then the exact 1 ms flight for those only
        s, w, t, bounces, bounce, net = flight(launch, spin, dt=0.003, steps=470, plane=plane)
        speed = np.linalg.norm(s[:, 3:], axis=-1)
        dy, height = s[:, 1] - HOME_Y, s[:, 2] + TABLE_Z
        near = np.flatnonzero(np.isfinite(s).all(1) & (bounces == 1) & (dy > LATERAL[0] - 0.05) & (dy < LATERAL[-1] + 0.05)
                              & (height > HEIGHT[0] - 0.05) & (height < HEIGHT[-1] + 0.05) & (speed > SPEED[0] - 0.2) & (speed < SPEED[-1] + 0.2))
        launch, spin = launch[near], spin[near]
        s, w, t, bounces, bounce, net = flight(launch, spin, plane=plane, steps=1500)
        speed = np.linalg.norm(s[:, 3:], axis=-1)
        angle = np.degrees(np.arctan2(s[:, 4], -s[:, 3]))
        dy, height = s[:, 1] - HOME_Y, s[:, 2] + TABLE_Z
        ok = (np.isfinite(s).all(1) & (bounces == 1) & (bounce[:, 0] > plane + 0.03) & (bounce[:, 0] < 1.33) & (net[:, 1] > 0.18)
              & (net[:, 0] > -1.49) & (net[:, 0] < -0.035) & (dy > LATERAL[0]) & (dy < LATERAL[-1]) & (height > HEIGHT[0]) & (height < HEIGHT[-1])
              & (speed > SPEED[0]) & (speed < SPEED[-1]) & (np.abs(angle) < 24.0) & (s[:, 3] < -0.6))
        key = (np.digitize(dy, LATERAL), np.digitize(height, HEIGHT), np.digitize(speed, SPEED), (s[:, 5] > 0).astype(int))
        for i in np.flatnonzero(ok):
            bucket = cells.setdefault(tuple(int(k[i]) for k in key), [])
            if len(bucket) < per_cell:
                bucket.append((launch[i], spin[i], s[i], w[i], t[i], bounce[i]))
        if sum(len(b) for b in cells.values()) >= count:
            break
    rows = [r for b in cells.values() for r in b]
    order = rng.permutation(len(rows))[:count]
    return tuple(np.array([rows[i][j] for i in order]) for j in range(6)) + (tried, len(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, required=True, help="assembled Supergate tree (tools/supergate/assemble.sh)")
    ap.add_argument("--suites", nargs="+", default=["main", "outer"])
    ap.add_argument("--count", type=int, default=9600, help="launches to write (a multiple of six)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--general", action="store_true", help="suite-independent launches over the whole reachable window (see ``general``)")
    ap.add_argument("--plane", type=float, default=PLANE, help="strike plane the window is defined at (0.0 with the robot 0.58 m behind the edge)")
    ap.add_argument("--after-plane-s", type=float, default=1.0, help="the ball is cleared this long after it reaches the plane")
    ap.add_argument("--pause-s", type=float, default=0.8)
    ap.add_argument("--lateral", type=float, nargs="+", help="lateral cell edges of --general, m from HOME (default "
                    f"{LATERAL}); e.g. -0.92 -0.84 -0.76 -0.66 for the balls a step forehand is for")
    args = ap.parse_args()
    if args.general:
        launch, spin, s, w, t, bounce, tried, cells = general(args.count, args.seed, args.plane, args.lateral or LATERAL)
        rows = [dict(launch=launch[i].tolist(), initial_spin=spin[i].tolist(), flight_duration_s=round(float(t[i]) + args.after_plane_s, 3),
                     pause_s=args.pause_s, predicted_contact=s[i].tolist(), predicted_contact_spin=w[i].tolist(), flight_s=float(t[i]),
                     bounce=bounce[i].tolist(), plane=args.plane) for i in range(len(launch))]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows) + "\n")
        q = lambda x: " ".join(f"{v:+.2f}" for v in np.quantile(x, (0.01, 0.25, 0.5, 0.75, 0.99)))  # noqa: E731
        print(f"{args.out}: {len(rows)} general launches over {cells} cells from {tried} tried")
        print(f"  at x = {args.plane}: y - home {q(s[:, 1] - HOME_Y)} | height {q(s[:, 2] + TABLE_Z)} | speed {q(np.linalg.norm(s[:, 3:], axis=-1))} | "
              f"angle {q(np.degrees(np.arctan2(s[:, 4], -s[:, 3])))} | vz {q(s[:, 5])} | topspin {q(-w[:, 1])} | launch x {q(launch[:, 0])}")
        return
    data = args.tree / "a3_deploy/a3_deploy_example/scripts/supergate/data"
    seeds = []
    for suite in args.suites:
        for shard in range(1, 7):
            seeds += [r["launch"] + r["initial_spin"] for r in json.loads((data / suite / f"shard-{shard:02d}-ledger.json").read_text())]
    seeds = np.array(seeds)
    rng = np.random.default_rng(args.seed)
    kept, tried = [], 0
    while sum(len(k[0]) for k in kept) < args.count:
        n = 4 * args.count
        base = seeds[rng.integers(0, len(seeds), n)]
        launch, spin = base[:, :6].copy(), base[:, 6:].copy()
        launch[:, 0] = np.clip(launch[:, 0] + rng.uniform(-0.15, 0.15, n), 2.0, 2.8)
        launch[:, 1] += rng.uniform(-0.12, 0.12, n)
        launch[:, 2] = np.clip(launch[:, 2] + rng.uniform(-0.06, 0.06, n), 0.03, 1.35)
        launch[:, 3:] = launch[:, 3:] * rng.uniform(0.93, 1.07, (n, 1)) + rng.uniform(-0.20, 0.20, (n, 3))
        spin += rng.uniform(-5.0, 5.0, (n, 3))
        s, w, t, bounces, bounce, net = flight(launch, spin)
        speed = np.linalg.norm(s[:, 3:], axis=-1)
        angle = np.degrees(np.arctan2(s[:, 4], -s[:, 3]))
        dy, height = s[:, 1] - HOME_Y, s[:, 2] + TABLE_Z
        ok = (np.isfinite(s).all(1) & (bounces == 1) & (bounce[:, 0] > 0.02) & (bounce[:, 0] < 1.33) & (net[:, 1] > 0.18)
              & (net[:, 0] > -1.49) & (net[:, 0] < -0.035) & (dy > -0.66) & (dy < 0.28) & (height > 0.80) & (height < 1.45)
              & (speed > 1.7) & (speed < 5.2) & (np.abs(angle) < 21.0) & (s[:, 3] < -0.7))
        kept.append((launch[ok], spin[ok], s[ok], w[ok], t[ok], bounce[ok]))
        tried += n
    launch, spin, s, w, t, bounce = (np.concatenate(x)[:args.count] for x in zip(*kept))
    rows = [dict(launch=launch[i].tolist(), initial_spin=spin[i].tolist(), flight_duration_s=round(float(t[i]) + 1.0, 3), pause_s=0.8,
                 predicted_contact=s[i].tolist(), predicted_contact_spin=w[i].tolist(), flight_s=float(t[i]), bounce=bounce[i].tolist())
            for i in range(len(launch))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows) + "\n")
    speed = np.linalg.norm(s[:, 3:], axis=-1)
    q = lambda x: " ".join(f"{v:+.2f}" for v in np.quantile(x, (0.01, 0.25, 0.5, 0.75, 0.99)))  # noqa: E731
    print(f"{args.out}: {len(rows)} launches from {tried} displaced seeds ({100 * len(rows) / tried:.0f}% kept so far)")
    print(f"  at the plane: y - home {q(s[:, 1] - HOME_Y)} | height {q(s[:, 2] + TABLE_Z)} | speed {q(speed)} | "
          f"angle {q(np.degrees(np.arctan2(s[:, 4], -s[:, 3])))} | vz {q(s[:, 5])} | topspin {q(-w[:, 1])}")


if __name__ == "__main__":
    main()
