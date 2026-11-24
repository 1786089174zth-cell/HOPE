"""A Supergate suite of the balls a real session received (numpy only).

    python -I tools/real/make_suite.py <dump dir> [<dump dir> ...] --out <suite dir>

``<dump dir>`` holds ``mocap.npz`` (``dump_ball.py``) and ``shots.json`` (``shots.py``) of one real session. For
every stroke the Runner engaged, the incoming flight is the ball's mocap samples from the sender (the opponent's
racket or hand, or where the track starts) to 30 ms before the robot's racket, fitted with Supergate's flight and
table-bounce model exactly as ``tools/field/extract_flights.py`` fits the field recordings.

The real robot stood where its session HOME was and struck on the plane HOME.x + 0.58; Supergate's robot stands at
(-0.58, -0.7625) and strikes on x = 0. Each flight is shifted by that difference, so that the ball is where it was
*relative to the robot* (the shifts are a few centimetres; the bounce moves on the table by as much).

The flights are dealt in recorded order to six shards of equal length (the last ones padded with repeats marked
``padding`` in the ledger); the next ball is launched when it was launched in the session, never sooner than
0.8 s after the ball reaches the strike plane and never later than 3 s. Each ledger row carries the real stroke:
``real`` = session, shot index, outcome, the Planner's command and what the ball did off the racket.
``hit_speed_scale.txt`` lists the operator's hit-speed scale each ball was played with (score.sh:
``SUPERGATE_HIT_SPEED_SCALE=<suite>/hit_speed_scale.txt``).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "field"))
from extract_flights import crossing, fit, simulate  # noqa: E402

SIM_HOME = (-0.58, -0.7625)
PLANE_AHEAD = 0.58
NET = 1.37
FLOOR = 0.031  # the ball's mocap height at a bounce in these sessions (centre; the fit re-measures it)


def segment(bt, bp, end, trim=0.0, reach=1.6, gap=0.12):
    """Indices of the incoming flight that ends at time ``end``: back to where the ball stops coming towards the
    robot (the sender's racket or hand), a tracking gap, or ``reach`` seconds; 30 ms clear of that, and from after
    the bounce on the sender's side if there is one (a serve). ``trim`` cuts more seconds off the start."""
    hi = int(np.searchsorted(bt, end))
    lo = int(np.searchsorted(bt, end - reach))
    if hi - lo < 40:
        return None
    breaks = np.flatnonzero(np.diff(bt[lo:hi]) > gap)
    if len(breaks):
        lo = lo + int(breaks[-1]) + 1
    k = 7
    if hi - lo < 40 + k:
        return None
    vx = (bp[lo + k:hi, 0] - bp[lo:hi - k, 0]) / (bt[lo + k:hi] - bt[lo:hi - k])
    away = np.flatnonzero(vx > -0.3)
    start = lo if not len(away) else lo + int(away[-1]) + k
    t0 = bt[start] + (0.03 if len(away) else 0.0)
    z, x = bp[start:hi, 2], bp[start:hi, 0]
    low = np.flatnonzero((x[1:-1] > NET + 0.03) & (z[1:-1] < FLOOR + 0.03) & (z[1:-1] <= z[:-2]) & (z[1:-1] <= z[2:]))
    if len(low):  # the sender's own bounce: start after it
        t0 = max(t0, bt[start + 1 + int(low[-1])] + 0.02)
    a = int(np.searchsorted(bt, t0 + trim))
    return (a, hi) if hi - a >= 40 else None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    i = argv.index("--out")
    out = Path(argv[i + 1])
    del argv[i:i + 2]
    sessions = {}
    for arg in argv:
        ball = np.load(Path(arg) / "mocap.npz")["ball"]
        order = np.argsort(ball[:, 0])
        sessions[Path(arg).name] = (ball[order, 0], ball[order, 2:5], json.loads((Path(arg) / "shots.json").read_text()))
    dt = float(np.median(np.concatenate([np.diff(bt) for bt, _, _ in sessions.values()])))
    todo = [(name, n, 0.0) for name, (_, _, shots) in sessions.items() for n in range(len(shots))]
    done, lost = {}, []
    for attempt in range(6):  # a flight that does not fit loses 60 ms more of its start each round
        flights = []
        for name, n, trim in todo:
            bt, bp, shots = sessions[name]
            s = shots[n]
            ts = s["t_strike"]
            end = ts + s["t_contact"] - 0.03 if "t_contact" in s else ts + s["pred_err"][0] if "pred_err" in s else ts
            seg = segment(bt, bp, end, trim)
            if seg is None:
                lost.append((name, n, s["outcome"]))
                continue
            a, b = seg
            shift = np.array([s["home"][0] + PLANE_AHEAD, s["home"][1] - SIM_HOME[1], 0.0])
            flights.append(dict(session=name, shot=n, t=bt[a:b] - bt[a], p=bp[a:b] - shift, t_launch=float(bt[a]), real=s, shift=shift, trim=trim))
        if not flights:
            break
        steps = max(int(round(f["t"][-1] / dt)) + 1 for f in flights) + int(0.3 / dt)
        obs, mask = np.zeros((len(flights), steps, 3)), np.zeros((len(flights), steps), bool)
        grid = np.arange(steps) * dt
        for n, f in enumerate(flights):
            # mocap samples on the uniform grid of the fit; grid points in a tracking gap are masked
            k = np.clip(np.searchsorted(f["t"], grid), 1, len(f["t"]) - 1)
            ok = (grid <= f["t"][-1]) & (f["t"][k] - f["t"][k - 1] < 0.012)
            ok[:7] = True  # the fit's first guess reads the first seven grid points
            for axis in range(3):
                obs[n, :, axis] = np.interp(grid, f["t"], f["p"][:, axis])
            obs[n, :, 2] += 0.0201 - FLOOR
            mask[n] = ok
        params, rms = fit(obs, mask, dt)
        history, bounce, count = simulate(params, steps, dt)
        at0, t0 = crossing(history, dt, 0.0)
        at7, t7 = crossing(history, dt, 0.07)
        todo = []
        for n, f in enumerate(flights):
            values = [*params[n], *at7[n, :9], *at0[n, :9], t0[n], t7[n]]
            good = rms[n] < 0.025 and count[n] == 1 and all(math.isfinite(float(v)) for v in values) and bounce[n, 0] < NET
            f.update(params=params[n], rms=float(rms[n]), count=int(count[n]), at0=at0[n], at7=at7[n], t0=float(t0[n]), t7=float(t7[n]),
                     bounce=bounce[n], samples=int(mask[n].sum()))
            key = (f["session"], f["shot"])
            if good:
                done[key] = f
            else:
                done.setdefault(key + ("failed",), f)
                todo.append((f["session"], f["shot"], f["trim"] + 0.06))
        print(f"round {attempt}: {len(flights)} flights fitted, {len(flights) - len(todo)} accepted", flush=True)
        if not todo:
            break
    flights = [done[k] for k in sorted(k for k in done if len(k) == 2)]
    rows = []
    dropped = [(k[0], k[1], round(f["rms"], 3), f["count"], f["real"]["outcome"]) for k, f in sorted(done.items()) if len(k) == 3 and k[:2] not in done]
    dropped += [(name, n, None, None, outcome) for name, n, outcome in lost if (name, n) not in done]
    for f in flights:
        s = f["real"]
        rows.append(dict(launch=f["params"][:6].tolist(), initial_spin=f["params"][6:].tolist(), predicted_contact=f["at7"][:6].tolist(),
                         predicted_contact_spin=f["at7"][6:9].tolist(), flight_s=f["t7"], plane0=f["at0"].tolist(), plane0_t=f["t0"],
                         bounce=f["bounce"].tolist(), fit_rms_m=f["rms"], samples=f["samples"], t_launch=f["t_launch"],
                         real=dict(session=f["session"], shot=f["shot"], outcome=s["outcome"], swing=s["swing"], tts=s["tts"],
                                   target=(np.array(s["target"]) - f["shift"]).tolist(), target_v=s["target_v"], ball_v=s.get("ball_v"),
                                   plane_arrival_s=s["t_strike"] - f["t_launch"], v_out=s.get("v_out"), land=s.get("land"),
                                   net_clear=s.get("net_clear"), hit_scale=s.get("hit_scale"), flight_time=s.get("flight_s"),
                                   racket_err=s["racket_err"], shift=f["shift"].tolist(), start_trim_s=f["trim"])))
    for r, nxt in zip(rows, rows[1:] + [None]):
        floor = r["plane0_t"] + 0.8
        gap = nxt["t_launch"] - r["t_launch"] if nxt and nxt["real"]["session"] == r["real"]["session"] else floor + 0.2
        r["flight_duration_s"], r["pause_s"] = round(max(min(gap, 3.0), floor), 3), 0.0
    size = math.ceil(len(rows) / 6)
    out.mkdir(parents=True, exist_ok=False)
    scales = []
    for k in range(6):
        part = [dict(r, padding=False) for r in rows[k * size:(k + 1) * size]]
        part += [dict(part[j % len(part)], padding=True) for j in range(size - len(part))]
        for index, row in enumerate(part):
            row.update(shard_id=k + 1, shot_id=index + 1)
        scene = {"balls": size, "clean_serves": [r["launch"] for r in part[:6]], "rapid_cycle": [r["launch"] for r in part[6:]],
                 "spin_rad_s_list": [r["initial_spin"] for r in part],
                 "timing": {"flight_s_list": [r["flight_duration_s"] for r in part], "pause_s_list": [r["pause_s"] for r in part]}}
        (out / f"shard-{k + 1:02d}.yaml").write_text(json.dumps(scene) + "\n")  # JSON is YAML
        (out / f"shard-{k + 1:02d}-ledger.json").write_text(json.dumps(part, indent=1, allow_nan=False) + "\n")
        scales.append(",".join(f"{r['real']['hit_scale'] or 1.0:g}" for r in part))
    (out / "hit_speed_scale.txt").write_text("\n".join(scales) + "\n")  # for SUPERGATE_HIT_SPEED_SCALE (tools/supergate/hit_scale.py)
    design = {"version": "real-session-ball-paths-v1", "source": [str(a) for a in argv], "shards": 6, "rows": 6 * size, "unique": len(rows),
              "strokes_in_source": sum(len(v[2]) for v in sessions.values()), "dropped": dropped, "ball_height_at_bounce_m": FLOOR, "policy_outcome_used_to_select": False,
              "scope": "Incoming balls of real-robot sessions, shifted to the robot's place in Supergate, fitted with the simulator's flight and bounce model."}
    (out / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    r = np.array([x["fit_rms_m"] for x in rows])
    print(f"{out}: 6 x {size} launches, {len(rows)} real strokes of {sum(len(v[2]) for v in sessions.values())} (dropped {len(dropped)}: {dropped}); "
          f"fit RMS median {100 * np.median(r):.2f} cm, 90% {100 * np.quantile(r, 0.9):.2f} cm; launched before the net: {sum(x['launch'][0] > NET for x in rows)}")


if __name__ == "__main__":
    main()
