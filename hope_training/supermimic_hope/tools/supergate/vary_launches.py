"""Many launches from the ones we have: each new launch is an existing one displaced a little, flown with the
compiled flight model, and kept when it is a legal one-bounce ball inside the window (numpy + the compiled
``planner_targets flight``; inside the hope container).

    .venv/bin/python hope_training/supermimic_hope/tools/supergate/vary_launches.py \
        --parents <launch files> ... --count 1000000 --seed 1 --out <launches.npz> [--lateral edges ...]

``make_launches.py --general`` draws launches blindly (origin, speed, loft, spin anywhere) and keeps the 5-10% that
are legal balls in the window: 108 000 launches from 1.1 million flights in a quarter of an hour on one core.
Here every proposal starts from a launch that is already such a ball (main_urdf_2's
``expand_stationary_coverage_bank.py`` grows its bank the same way), so most proposals are kept, and the flight is
the compiled one on every core.

* Parents: the launches of the given files. A parent is drawn by its cell (lateral position x height x speed x
  rising/falling at the strike plane, ``make_launches``' cells), every populated cell as often as the others, so
  rare kinds of ball (slow and low, fast and high) get as many proposals as common ones.
* Displacement: origin +-5 cm, +-5 cm, +-4 cm; velocity x 0.95-1.05 and +-0.06 m/s (sd) per axis; spin x 0.9-1.1
  and +-3 rad/s (sd) per axis. A new launch is a new flight with its own bounce, contact state and Planner target
  (``planner_targets.py``), not a copy with a moved label; it is still a neighbour of its parent, not an
  independent draw.
* Kept by the conditions of ``make_launches.general`` (one bounce on the robot's half beyond the plane, over the
  net by 18 cm inside the posts, inside the window of lateral position, height, speed and horizontal angle), and
  evenly over the cells: at most ``1.3 * count / cells`` per cell.

Writes an npz: ``launch`` (N, 6), ``initial_spin`` (N, 3), ``contact`` (N, 6, table frame), ``contact_spin``,
``flight_s``, ``net_to_plane_s``, ``plane``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_launches as ml  # noqa: E402
from planner_targets import fly, load_launches  # noqa: E402


def legal(s, t, bounces, bounce, net, plane, lateral):
    """The conditions of ``make_launches.general`` on flights, and the cell of each (lateral, height, speed, rising)."""
    speed = np.linalg.norm(s[:, 3:], axis=-1)
    angle = np.degrees(np.arctan2(s[:, 4], -s[:, 3]))
    dy, height = s[:, 1] - ml.HOME_Y, s[:, 2] + ml.TABLE_Z
    with np.errstate(invalid="ignore"):
        ok = (np.isfinite(s).all(1) & (bounces == 1) & (bounce[:, 0] > plane + 0.03) & (bounce[:, 0] < 1.33) & (net[:, 1] > 0.18)
              & (net[:, 0] > -1.49) & (net[:, 0] < -0.035) & (dy > lateral[0]) & (dy < lateral[-1]) & (height > ml.HEIGHT[0])
              & (height < ml.HEIGHT[-1]) & (speed > ml.SPEED[0]) & (speed < ml.SPEED[-1]) & (np.abs(angle) < 24.0) & (s[:, 3] < -0.6))
    key = ((np.digitize(dy, lateral) * 8 + np.digitize(height, ml.HEIGHT)) * 8 + np.digitize(speed, ml.SPEED)) * 2 + (s[:, 5] > 0)
    return ok, np.where(ok, key, -1).astype(np.int64)


def vary(parents, spins, count, seed, plane=0.0, lateral=ml.LATERAL, rounds=60):
    rng = np.random.default_rng(seed)
    s, w, t, bounces, bounce, net, t_net, _ = fly(parents, spins, plane=plane)
    ok, key = legal(s, t, bounces, bounce, net, plane, lateral)
    parents, spins, key = parents[ok], spins[ok], key[ok]
    cells, inverse = np.unique(key, return_inverse=True)
    members = [np.flatnonzero(inverse == c) for c in range(len(cells))]
    per_cell = int(np.ceil(1.3 * count / len(cells)))
    have = dict.fromkeys(cells.tolist(), 0)
    kept, proposed, flown = [], 0, 0
    for _ in range(rounds):
        total = sum(have.values())
        if total >= count:
            break
        # proposals from the cells that are not full, each as often as the others
        open_cells = np.array([c for c in range(len(cells)) if have[int(cells[c])] < per_cell])
        if not len(open_cells):
            break
        n = int(min(2_000_000, max(50_000, 2.0 * (count - total))))
        cell = open_cells[rng.integers(0, len(open_cells), n)]
        size = np.array([len(m) for m in members])[cell]
        offset = (rng.random(n) * size).astype(np.int64)
        start = np.concatenate(([0], np.cumsum([len(m) for m in members])))[cell]
        row = np.concatenate(members)[start + offset]
        launch, spin = parents[row].copy(), spins[row].copy()
        launch[:, :3] += rng.uniform([-0.05, -0.05, -0.04], [0.05, 0.05, 0.04], (n, 3))
        launch[:, 3:] = launch[:, 3:] * rng.uniform(0.95, 1.05, (n, 1)) + rng.normal(0.0, 0.06, (n, 3))
        spin = spin * rng.uniform(0.90, 1.10, (n, 1)) + rng.normal(0.0, 3.0, (n, 3))
        s, w, t, bounces, bounce, net, t_net, _ = fly(launch, spin, plane=plane)
        ok, k = legal(s, t, bounces, bounce, net, plane, lateral)
        proposed, flown = proposed + n, flown + int(ok.sum())
        take = np.zeros(n, bool)
        for c in np.unique(k[ok]):
            room = per_cell - have.get(int(c), 0)
            if room <= 0:
                continue
            ids = np.flatnonzero(k == c)[:room]
            take[ids] = True
            have[int(c)] = have.get(int(c), 0) + len(ids)
        kept.append((launch[take], spin[take], s[take], w[take], t[take], (t - t_net)[take]))
    out = [np.concatenate([part[j] for part in kept]) for j in range(6)]
    order = rng.permutation(len(out[0]))[:count]
    return [x[order] for x in out], dict(parents=len(parents), cells=len(cells), per_cell=per_cell, proposed=proposed, legal=flown)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parents", type=Path, nargs="+", required=True, help="launch files (json of make_launches.py, npz of this tool)")
    ap.add_argument("--count", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lateral", type=float, nargs="+", help=f"lateral cell edges, m from HOME (default {ml.LATERAL}: the stand strokes' window)")
    args = ap.parse_args()
    parents, spins, plane = load_launches(args.parents)
    lateral = tuple(args.lateral or ml.LATERAL)
    (launch, spin, s, w, t, after_net), info = vary(parents, spins, args.count, args.seed, plane, lateral)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, launch=launch, initial_spin=spin, contact=s, contact_spin=w, flight_s=t, net_to_plane_s=after_net, plane=plane)
    q = lambda x: " ".join(f"{v:+.2f}" for v in np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99]))  # noqa: E731
    speed, height = np.linalg.norm(s[:, 3:], axis=1), s[:, 2] + ml.TABLE_Z
    print(f"{args.out}: {len(launch)} launches from {info['parents']} parents in the window ({info['cells']} cells, at most {info['per_cell']} a cell); "
          f"{info['proposed']} proposals, {info['legal']} legal ({100 * info['legal'] / max(info['proposed'], 1):.0f}%)")
    print(f"  at x = {plane}: y - home {q(s[:, 1] - ml.HOME_Y)} | height {q(height)} | speed {q(speed)}")
    grid = [[int(((np.digitize(speed, (2.5, 4.0)) == i) & (np.digitize(height, (1.0, 1.2)) == j)).sum()) for j in range(3)] for i in range(3)]
    print(f"  speed (< 2.5, < 4.0, faster) x height (< 1.0, < 1.2, higher): {grid}")


if __name__ == "__main__":
    main()
