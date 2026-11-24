"""Per-shot outcome of a real session: runner trace (engage, target, racket FK) x mocap ball track.

    python -I tools/real/shots.py <session dir> <dump dir with mocap.npz>  -> <dump dir>/shots.json and a count

Frames: mocap world = table axes (x from the robot's table end towards the net at 1.37, y in [-1.525, 0]); the
ball's centre is ``FLOOR`` above the mocap origin when it bounces, so heights are measured from there. The Runner's
target / racket z are above the floor (table top 0.76).

Outcome of a stroke with contact, from what the mocap saw of the return: the height where the track crosses the
net and its first bounce; where the track ends first (lost, or the opponent takes the ball), the rest is the flight
(gravity and drag) fitted to its last 120 ms. ``landed`` = over the net and first bounce on the far half;
``net`` = first bounce on the robot's side or never over the net; ``long`` / ``wide`` = past the far edge / the
side lines. ``outcome_source`` says whether net and bounce were ``observed`` or ``extrapolated``.
"""
import collections, csv, glob, json, sys
import numpy as np

D, O = sys.argv[1], sys.argv[2]
TABLE_Z, NET_X, NET_H, LEN, WID, RAD = 0.76, 1.37, 0.1525, 2.74, 1.525, 0.02
FLOOR = 0.031  # mocap height of the ball's centre at a bounce in these sessions (tools/real/make_suite.py measures it)
G = np.array([0.0, 0.0, -9.81])
ball = np.load(O + "/mocap.npz")["ball"]
bt, bp = ball[:, 0], ball[:, 2:5]
order = np.argsort(bt); bt, bp = bt[order], bp[order]

f = glob.glob(D + "/mdu/real/attempt_001/sim2real/recordings/*/runner_trace.csv")[0]
r = csv.reader(open(f)); h = next(r); ix = {n: i for i, n in enumerate(h)}
keep = "wall_time_ns mode level current_strike_time engage_raw_tts frozen_command_seq swing_sign base_x base_y base_z target_x target_y target_z target_vx target_vy target_vz racket_x racket_y racket_z racket_vx racket_vy racket_vz racket_nx racket_ny racket_nz".split()
rows, status = [], []
for row in r:
    if len(row) < len(h): continue
    rows.append([float(row[ix[k]] or "nan") for k in keep]); status.append(row[ix["planner_status"]])
T = np.array(rows); c = {k: i for i, k in enumerate(keep)}
tw = T[:, c["wall_time_ns"]] * 1e-9
audit = {}
for row in csv.DictReader(open(glob.glob(D + "/hdu/planner_attempt_001/recordings/*/planner_audit.csv")[0])):
    audit[row["command_seq"]] = row

def interp(t, cols):
    return np.array([np.interp(t, tw, T[:, c[k]]) for k in cols])

def ballistic(t, p):
    """Position and velocity at ``t[-1]`` of the flight through samples ``(t, p)`` (least squares, gravity known)."""
    tau = t - t[-1]
    A = np.stack([np.ones(len(t)), tau], 1)
    co, *_ = np.linalg.lstsq(A, p - 0.5 * np.outer(tau ** 2, G), rcond=None)
    return co[0], co[1]


def fly(p, v, tmax=1.5, dt=0.002, k=0.1245):
    """Flight with drag from (p, v): height at the net (None if it is not reached or already passed) and the point
    where the ball's centre comes down to ``FLOOR``."""
    p, v = p.copy(), v.copy(); net = None
    for _ in range(int(tmax / dt)):
        a = -k * np.linalg.norm(v) * v + G
        pn = p + v * dt + 0.5 * a * dt * dt; v = v + a * dt
        if net is None and p[0] < NET_X <= pn[0]: net = p[2] + (pn[2] - p[2]) * (NET_X - p[0]) / (pn[0] - p[0])
        if pn[2] <= FLOOR and v[2] < 0: return net, pn
        p = pn
    return net, p


def after_contact(tc):
    """What the return did after the contact at ``tc``: dict(v_out, net_clear, land, outcome, outcome_source)."""
    m = (bt > tc + 0.012) & (bt < tc + 1.4)
    t, p = bt[m], bp[m]
    if len(t) < 10: return dict(outcome="contact_lost")
    # the usable part of the track: up to a tracking gap, the opponent's contact (x velocity turns back) or the first bounce
    end, bounce = len(t), None
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > 0.08: end = i; break
        if i >= 8 and p[i, 0] - p[i - 8, 0] < -0.02 and p[i, 0] > 0.5: end = i - 8; break            # coming back
        if i >= 2 and p[i - 1, 2] < FLOOR + 0.025 and p[i - 1, 2] <= p[i - 2, 2] and p[i - 1, 2] <= p[i, 2] and t[i - 1] - tc > 0.05:
            bounce = p[i - 1]; end = i; break
    if end < 10: return dict(outcome="contact_lost")
    t, p = t[:end], p[:end]
    out = {}
    first = t <= t[0] + 0.10
    if first.sum() >= 8:
        _, v = ballistic(t[first][::-1], p[first][::-1])   # velocity at the first sample after the contact
        out["v_out"] = v.tolist()
    j = np.flatnonzero((p[:-1, 0] < NET_X) & (p[1:, 0] >= NET_X))
    net, source = None, "observed"
    if len(j):
        j = j[0]; net = p[j, 2] + (p[j + 1, 2] - p[j, 2]) * (NET_X - p[j, 0]) / (p[j + 1, 0] - p[j, 0])
    land = bounce
    if land is None or (net is None and land[0] > NET_X):
        tail = t >= t[-1] - 0.12
        if tail.sum() < 8: return dict(out, outcome="contact_lost")
        p0, v0 = ballistic(t[tail], p[tail])
        net2, land2 = fly(p0, v0)
        source = "extrapolated"
        if land is None: land = land2
        if net is None: net = net2
    clear = None if net is None else float(net - (FLOOR - RAD) - NET_H - RAD)
    if land[0] <= NET_X or clear is None or (clear < 0 and bounce is None): name = "net"
    elif land[0] > LEN: name = "long"
    elif not (-WID < land[1] < 0): name = "wide"
    else: name = "landed"
    return dict(out, net_clear=clear, land=[float(v) for v in land], outcome=name, outcome_source=source)


shots = []
for i in np.flatnonzero(np.array(status) == "engage"):
    ts = T[i, c["current_strike_time"]]; tgt = T[i, [c["target_x"], c["target_y"], c["target_z"]]]; tv = T[i, [c["target_vx"], c["target_vy"], c["target_vz"]]]
    s = dict(t_engage=tw[i], t_strike=ts, tts=T[i, c["engage_raw_tts"]], swing=int(T[i, c["swing_sign"]]), target=tgt.tolist(), target_v=tv.tolist(),
             base=T[i, [c["base_x"], c["base_y"], c["base_z"]]].tolist(), seq=str(int(T[i, c["frozen_command_seq"]])))
    a = audit.get(s["seq"])
    if a:
        s.update(home=[float(a["session_home_x"]), float(a["session_home_y"])], ball_v=[float(a["strike_vx"]), float(a["strike_vy"]), float(a["strike_vz"])],
                 hit_scale=float(a["hit_speed_scale"]), flight_s=float(a["flight_time_requested_s"]), fit_rms=float(a["fit_rms_m"]),
                 spin=[float(a["spin_wx_rad_s"]), float(a["spin_wy_rad_s"]), float(a["spin_wz_rad_s"])], spin_valid=a["spin_valid"])
    # racket at the commanded strike time, and where it crosses the plane
    rk = interp(ts, ["racket_x", "racket_y", "racket_z", "racket_vx", "racket_vy", "racket_vz", "racket_nx", "racket_ny", "racket_nz"])
    s["racket_err"] = (rk[:3] - tgt).tolist(); s["racket_v"] = rk[3:6].tolist(); s["racket_n"] = rk[6:9].tolist()
    # ball around the strike
    m = (bt > ts - 0.25) & (bt < ts + 0.45)
    t, p = bt[m], bp[m]
    s["ball_samples"] = int(m.sum())
    out = "no_track"
    if m.sum() > 20:
        # actual plane crossing of the incoming ball
        j = np.flatnonzero((p[:-1, 0] > tgt[0]) & (p[1:, 0] <= tgt[0]) & (t[1:] - t[:-1] < 0.03))
        if len(j):
            j = j[0]; w = (p[j, 0] - tgt[0]) / (p[j, 0] - p[j + 1, 0])
            pc = p[j] + w * (p[j + 1] - p[j]); tc = t[j] + w * (t[j + 1] - t[j])
            s["pred_err"] = [float(tc - ts), float(tgt[1] - pc[1]), float(tgt[2] - TABLE_Z - pc[2])]   # planner minus actual: time, y, z
        # closest approach ball - racket centre
        rr = np.stack([np.interp(t, tw, T[:, c[k]]) for k in ("racket_x", "racket_y", "racket_z")], 1) - [0, 0, TABLE_Z]
        d = np.linalg.norm(p - rr, axis=1); k = int(np.argmin(d))
        s["closest"] = dict(d=float(d[k]), dt=float(t[k] - ts), off=(p[k] - rr[k]).tolist())
        # contact: x velocity turns positive near the robot
        tc0 = None
        for k2 in range(5, len(t) - 8):
            if t[k2 + 6] - t[k2] > 0.05: continue
            v = (p[k2 + 6, 0] - p[k2, 0]) / (t[k2 + 6] - t[k2])
            if v > 0.4 and p[k2, 0] < 0.8 and t[k2] > ts - 0.12:
                tc0 = t[k2]; break
        xmin = float(p[:, 0].min()); s["ball_xmin"] = xmin
        if tc0 is None:
            out = "miss" if xmin < tgt[0] - 0.15 else "lost_before_plane"
        else:
            near = (t > tc0 - 0.03) & (t < tc0 + 0.05)
            tc = float(t[near][np.argmin(p[near, 0])])     # the turn of the ball's x
            s["t_contact"] = tc - ts
            res = after_contact(tc)
            out = res.pop("outcome")
            s.update(res)
            vi = np.flatnonzero((t > tc - 0.08) & (t < tc - 0.012))
            if len(vi) >= 8: s["v_in"] = ballistic(t[vi], p[vi])[1].tolist()
    s["outcome"] = out
    shots.append(s)
json.dump(shots, open(O + "/shots.json", "w"), indent=1)
print(D.split("/")[-1], len(shots), "shots", collections.Counter(s["outcome"] for s in shots).most_common(),
      "| net and bounce", collections.Counter(s["outcome_source"] for s in shots if "outcome_source" in s).most_common())
