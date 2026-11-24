"""Incoming ball flights of field rally recordings, fitted with the simulator's ball model (numpy only).

    python -I tools/field/extract_flights.py <recording.npz> [...] --out <flights.json>

Input: the ``.npz`` of ``read_mcap.py`` (mocap world = table axes: x along the table from the near end 0 to the far
end 2.74 m, y from -1.525 to 0, z above the table top). In a robot-vs-robot rally the ball's x velocity flips at
every racket contact; the samples between two contacts are one flight, received by the robot at the end it flies
to. Flights towards the far end are mirrored (x -> 2.74 - x, y -> -1.525 - y), so every flight is an incoming ball
for a robot standing behind x = 0, as in Gate3 and Supergate.

Each flight is fitted with Supergate's flight and table-bounce model (``tools/supergate/make_launches.py``): position,
velocity and spin just after the sender's contact (9 parameters, Levenberg-Marquardt over all flights at once, spin
with a weak prior to zero because 0.3-0.9 s of 300 Hz positions determine it only roughly). The output lists, per
flight: the fitted launch state and spin, the fit's RMS error, the bounce, the ball state where it crosses the
planes x = 0 (our strike plane with the robot 0.58 m behind the table edge) and x = 0.07 (Supergate's), where and
when the receiving robot actually struck it, and the time since the previous incoming ball of the same robot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "supergate"))
from make_launches import FRICTION, NORMAL_SHARE, RESTITUTION, accel  # noqa: E402

LENGTH, WIDTH, NET = 2.74, 1.525, 1.37
SUB = 3  # integration substeps per mocap sample (about 1.1 ms)


def simulate(params, steps, dt):
    """Ball centre positions (N, steps, 3) at ``dt`` intervals from state ``params`` (N, 9: p, v, w), and the
    bounce positions (N, 2), count (N,) and the full state history (N, steps, 9)."""
    s, w = params[:, :6].copy(), params[:, 6:].copy()
    n = len(s)
    out = np.empty((n, steps, 9))
    bounce, count = np.full((n, 2), np.nan), np.zeros(n, int)
    h = dt / SUB
    for k in range(steps):
        out[:, k, :6], out[:, k, 6:] = s, w
        for _ in range(SUB):
            prev_z = s[:, 2].copy()
            v = s[:, 3:]
            a = accel(v, w)
            vb = v + 0.5 * h * a
            b = accel(vb, w)
            vc = v + 0.5 * h * b
            c = accel(vc, w)
            vd = v + h * c
            d = accel(vd, w)
            s = np.concatenate((s[:, :3] + h * (v + 2 * vb + 2 * vc + vd) / 6, v + h * (a + 2 * b + 2 * c + d) / 6), 1)
            hit = (s[:, 2] <= 0.0201) & (prev_z > 0.0201) & (s[:, 5] < 0) & (s[:, 0] >= 0) & (s[:, 0] <= LENGTH) & (
                s[:, 1] >= -WIDTH) & (s[:, 1] <= 0)
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
                bounce[hit & (count == 0)] = s[hit & (count == 0), :2]
                count[hit] += 1
    return out, bounce, count


def fit(obs, mask, dt, iterations=25, spin_prior=0.002):
    """Launch parameters (N, 9) for observed positions ``obs`` (N, K, 3) with validity ``mask`` (N, K)."""
    n, steps, _ = obs.shape
    lead = 6
    p0 = obs[:, 0]
    v0 = (obs[:, lead] - obs[:, 0]) / (lead * dt)
    x = np.concatenate((p0, v0, np.zeros((n, 3))), 1)
    scale = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 5.0, 5.0, 5.0])
    weight = mask[:, :, None].astype(float)

    def residual(params):
        r = (simulate(params, steps, dt)[0][:, :, :3] - obs) * weight
        return np.concatenate((r.reshape(n, -1), spin_prior * params[:, 6:]), 1)

    lam = np.full(n, 1e-2)
    r = residual(x)
    cost = (r * r).sum(1)
    for _ in range(iterations):
        J = np.empty((n, r.shape[1], 9))
        for j in range(9):
            d = np.zeros(9)
            d[j] = scale[j] * 0.1
            J[:, :, j] = (residual(x + d) - r) / d[j]
        Js = J * scale  # scaled coordinates
        A = Js.transpose(0, 2, 1) @ Js
        g = (Js.transpose(0, 2, 1) @ r[:, :, None])[:, :, 0]
        step = np.linalg.solve(A + lam[:, None, None] * np.eye(9) * np.trace(A, axis1=1, axis2=2)[:, None, None] / 9, -g[:, :, None])[:, :, 0] * scale
        trial = x + step
        trial[:, 6:] = np.clip(trial[:, 6:], -400.0, 400.0)
        r_new = residual(trial)
        cost_new = (r_new * r_new).sum(1)
        better = cost_new < cost
        x = np.where(better[:, None], trial, x)
        r = np.where(better[:, None], r_new, r)
        cost = np.where(better, cost_new, cost)
        lam = np.where(better, lam * 0.3, lam * 5.0).clip(1e-6, 1e6)
    rms = np.sqrt(((simulate(x, steps, dt)[0][:, :, :3] - obs) ** 2 * weight).sum((1, 2)) / np.maximum(mask.sum(1), 1))
    return x, rms


def crossing(history, dt, plane):
    """State (N, 9) and time (N,) where the simulated ball first crosses ``x = plane`` going towards the robot."""
    x = history[:, :, 0]
    after = (x[:, 1:] <= plane) & (x[:, :-1] > plane)
    k = np.where(after.any(1), after.argmax(1), -1)
    rows = np.arange(len(x))
    a, b = history[rows, k], history[rows, np.minimum(k + 1, x.shape[1] - 1)]
    f = ((a[:, 0] - plane) / (a[:, 0] - b[:, 0] + 1e-12))[:, None]
    state = a + f * (b - a)
    state[k < 0] = np.nan
    return state, np.where(k >= 0, (k + f[:, 0]) * dt, np.nan)


def flights_of(path: Path):
    d = np.load(path)
    t, p = d["Ball_t"], d["Ball_p"].copy()
    dt = float(np.median(np.diff(t)))
    k = 7
    v = (p[k:] - p[:-k]) / (t[k:] - t[:-k])[:, None]
    mid = k // 2
    sign = np.sign(v[:, 0])
    flips = np.flatnonzero(sign[1:] * sign[:-1] < 0) + 1 + mid  # sample index of each racket contact
    # The table top in the recording: at a bounce the ball's centre is one radius above it. Bounces are the
    # lowest sample of each flight between two racket contacts (the ball's lowest samples overall will not do:
    # after the last stroke it drops to the floor).
    lows = [float(p[a + 9:b - 9, 2].min()) for a, b in zip(flips[:-1], flips[1:]) if 60 < b - a - 18 and t[b] - t[a] < 1.6]
    if not lows:
        return [], dt
    p[:, 2] += 0.0201 - float(np.median(lows))
    rows = []
    for a, b in zip(flips[:-1], flips[1:]):
        i0, i1 = a + 9, b - 9  # 30 ms clear of either contact
        if i1 - i0 < 60 or t[b] - t[a] > 1.6:
            continue
        seg = p[i0:i1].copy()
        to_far = seg[-1, 0] > seg[0, 0]
        hit = p[b].copy()  # where the receiver struck it
        if to_far:
            seg[:, 0], seg[:, 1] = LENGTH - seg[:, 0], -WIDTH - seg[:, 1]
            hit[0], hit[1] = LENGTH - hit[0], -WIDTH - hit[1]
        rows.append(dict(recording=path.stem, t_launch=float(t[i0]), t_sender_contact=float(t[a]), t_receiver_contact=float(t[b]),
                         receiver="far" if to_far else "near", samples=seg, receiver_contact=hit.tolist()))
    return rows, dt


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    i = argv.index("--out")
    out = Path(argv[i + 1])
    del argv[i:i + 2]
    flights, dts = [], []
    for arg in argv:
        rows, dt = flights_of(Path(arg))
        flights += rows
        dts.append(dt)
    dt = float(np.median(dts))
    steps = max(len(f["samples"]) for f in flights) + int(0.25 / dt)  # run on past the last sample to reach the planes
    obs, mask = np.zeros((len(flights), steps, 3)), np.zeros((len(flights), steps), bool)
    for n, f in enumerate(flights):
        k = len(f["samples"])
        obs[n, :k], mask[n, :k] = f["samples"], True
    params, rms = fit(obs, mask, dt)
    history, bounce, count = simulate(params, steps, dt)
    at0, t0 = crossing(history, dt, 0.0)
    at7, t7 = crossing(history, dt, 0.07)
    last = {}
    result = []
    for n, f in enumerate(flights):
        key = (f["recording"], f["receiver"])
        since = f["t_launch"] - last[key] if key in last else None
        last[key] = f["t_launch"]
        result.append(dict(recording=f["recording"], receiver=f["receiver"], t_launch=f["t_launch"],
                           launch=params[n, :6].tolist(), initial_spin=params[n, 6:].tolist(), fit_rms_m=float(rms[n]),
                           samples=int(mask[n].sum()), bounces=int(count[n]), bounce=bounce[n].tolist(),
                           plane0=at0[n].tolist(), plane0_t=float(t0[n]), plane007=at7[n].tolist(), plane007_t=float(t7[n]),
                           receiver_contact=f["receiver_contact"], receiver_contact_t=f["t_receiver_contact"] - f["t_launch"],
                           since_previous_launch_s=since))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, allow_nan=True) + "\n")
    print(f"{out}: {len(result)} flights from {len(argv)} recordings; fit RMS median {100 * np.median(rms):.2f} cm, "
          f"90% {100 * np.quantile(rms, 0.9):.2f} cm; one bounce on {int((count == 1).sum())}")


if __name__ == "__main__":
    main()
