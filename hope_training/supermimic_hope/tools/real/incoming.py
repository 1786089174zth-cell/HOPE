"""Every ball that came to the robot in a real session, by speed, and what the robot did with it (numpy only).

    python -I tools/real/incoming.py <dump dir> [<dump dir> ...]

``<dump dir>`` holds ``mocap.npz`` (``dump_ball.py``) and ``shots.json`` (``shots.py``). An incoming ball is a
stretch of the mocap track that comes over the net towards the robot and reaches ``NEAR`` (0.35 m in front of the
robot's table end); its speed is measured there (the strike plane is ~0.3 m further). It is ``engaged`` if the
Runner started a stroke whose strike time is within 0.35 s of the ball's arrival at the strike plane; the outcome
of that stroke is the one ``shots.py`` gave it. A ball without a stroke was never answered.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NET, NEAR, FLOOR = 1.37, 0.35, 0.031
SPEED = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)


def band(v):
    for lo, hi in zip((None, *SPEED), (*SPEED, None)):
        if hi is None or v < hi:
            return f"{'' if lo is None else f'{lo:g}'}..{'' if hi is None else f'{hi:g}'}"


def incoming(bt, bp):
    """(time at NEAR, velocity there, time over the net, bounced on the robot's half) of each incoming pass."""
    out = []
    i, n = 1, len(bt)
    while i < n:
        if bp[i - 1, 0] >= NEAR > bp[i, 0] and bt[i] - bt[i - 1] < 0.05:
            lo = int(np.searchsorted(bt, bt[i] - 0.04))
            hi = int(np.searchsorted(bt, bt[i] + 0.02))
            if hi - lo >= 8:
                v = np.polyfit(bt[lo:hi] - bt[i], bp[lo:hi], 1)[0]
                # back to the net: the track must have been beyond it within the last 0.9 s without a gap
                a = int(np.searchsorted(bt, bt[i] - 0.9))
                seg_t, seg_p = bt[a:i + 1], bp[a:i + 1]
                gaps = np.flatnonzero(np.diff(seg_t) > 0.12)
                if len(gaps):
                    seg_t, seg_p = seg_t[gaps[-1] + 1:], seg_p[gaps[-1] + 1:]
                over = np.flatnonzero(seg_p[:, 0] > NET)
                if v[0] < -0.8 and len(over):
                    after = seg_p[over[-1]:]
                    z = after[:, 2]
                    low = (z[1:-1] < FLOOR + 0.03) & (z[1:-1] <= z[:-2]) & (z[1:-1] <= z[2:])
                    out.append((bt[i], v, seg_t[over[-1]], bool(low.any()), float(bp[i, 1]), float(bp[i, 2])))
            i = int(np.searchsorted(bt, bt[i] + 0.3))
        else:
            i += 1
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    groups = defaultdict(list)
    for arg in argv:
        ball = np.load(Path(arg) / "mocap.npz")["ball"]
        order = np.argsort(ball[:, 0])
        bt, bp = ball[order, 0], ball[order, 2:5]
        shots = json.loads((Path(arg) / "shots.json").read_text())
        strikes = np.array([s["t_strike"] for s in shots])
        home = np.median([s["home"] for s in shots], 0)
        used = set()
        for t, v, t_net, bounced, y, z in incoming(bt, bp):
            if not bounced:
                continue  # a volley or a ball that never touched the robot's half: not a ball to return
            plane = home[0] + 0.58
            arrive = t + (plane - NEAR) / v[0]
            k = int(np.argmin(np.abs(strikes - arrive))) if len(strikes) else -1
            shot = shots[k] if k >= 0 and abs(strikes[k] - arrive) < 0.35 and k not in used else None
            if shot is not None:
                used.add(k)
            groups[band(float(np.linalg.norm(v)))].append(dict(
                speed=float(np.linalg.norm(v)), net_to_plane=arrive - t_net, lateral=y - home[1], height=z - FLOOR + 0.76,
                outcome=shot["outcome"] if shot else "no stroke", tts=shot["tts"] if shot else np.nan, session=Path(arg).name, t=t))
    total = sum(len(v) for v in groups.values())
    print(f"{total} incoming balls that bounced on the robot's half")
    names = ["landed", "long", "net", "wide", "miss", "no stroke"]
    print(f"   {'speed m/s':10s} {'n':>4s} | " + " ".join(f"{k:>9s}" for k in names) + " | net->plane s | notice s | lateral m (10-90%)")
    for key in sorted(groups, key=lambda k: np.mean([r["speed"] for r in groups[k]])):
        rows = groups[key]
        count = {k: sum(r["outcome"] == k for r in rows) for k in names}
        other = len(rows) - sum(count.values())
        lat = np.quantile([r["lateral"] for r in rows], [0.1, 0.9])
        tts = [r["tts"] for r in rows if np.isfinite(r["tts"])]
        print(f"   {key:10s} {len(rows):4d} | " + " ".join(f"{count[k]:9d}" for k in names) + (f" (+{other} other)" if other else "") +
              f" | {np.median([r['net_to_plane'] for r in rows]):11.2f} | {np.median(tts) if tts else float('nan'):8.2f} | {lat[0]:+.2f}..{lat[1]:+.2f}")
    if "--list" in argv:
        pass


if __name__ == "__main__":
    main([a for a in sys.argv[1:]])
