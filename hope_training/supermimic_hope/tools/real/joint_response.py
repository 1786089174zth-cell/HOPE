"""How the joints follow their commands during strokes, from a Runner trace (real session or Gate3 run; stdlib + numpy).

    python -I tools/real/joint_response.py <runner_trace.csv> [...]

For every engaged stroke (``level`` > 0) and joint: the lag of the measured angle behind the command (the shift that
best overlays ``q`` on ``des``, 0-160 ms), the tracking error at that lag and without it, the peak measured speed
against the peak commanded speed, and the peak torque estimate. Medians over the strokes of a side (forehand:
``swing_sign`` > 0). A simulator whose actuators match the robot gives the same numbers for the same checkpoint.
"""
import csv, sys
import numpy as np

JOINTS = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
          "waist_yaw", "waist_roll", "waist_pitch", "left_hip_pitch", "left_hip_roll", "left_knee", "right_hip_pitch", "right_hip_roll", "right_knee"]


def load(path):
    r = csv.reader(open(path)); h = next(r); ix = {n: i for i, n in enumerate(h)}
    cols = ["wall_time_ns", "level", "swing_sign"] + [f"{k}_{j}_joint" for j in JOINTS for k in ("des", "q", "qd")] + [f"tau_est_{j}_joint" for j in JOINTS if f"tau_est_{j}_joint" in ix]
    rows = [[float(row[ix[c]] or "nan") for c in cols] for row in r if len(row) >= len(h) and row[ix["mode"]] == "3"]
    return np.array(rows), {c: i for i, c in enumerate(cols)}


def strokes(T, c):
    on = T[:, c["level"]] > 0
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts = [e + 1 for e in edges if on[e + 1]]
    for a in starts:
        b = a
        while b < len(on) and on[b]: b += 1
        if b - a >= 15 and a >= 5 and b + 10 < len(on): yield a - 5, min(b + 10, a + 60)


def main(argv=None):
    for path in (sys.argv[1:] if argv is None else argv):
        T, c = load(path)
        t = T[:, c["wall_time_ns"]] * 1e-9
        res = {}
        for a, b in strokes(T, c):
            side = "FH" if np.nanmax(T[a:b, c["swing_sign"]]) > 0 else "BH"
            tt = t[a:b] - t[a]
            if np.any(np.diff(tt) <= 0): continue
            for j in JOINTS:
                des, q, qd = T[a:b, c[f"des_{j}_joint"]], T[a:b, c[f"q_{j}_joint"]], T[a:b, c[f"qd_{j}_joint"]]
                amp = des.max() - des.min()
                if amp < 0.15: continue  # the joint hardly moves in this stroke
                lags = np.arange(0.0, 0.161, 0.004)
                err = [np.sqrt(np.mean((q[8:] - np.interp(tt[8:] - lag, tt, des)) ** 2)) for lag in lags]
                k = int(np.argmin(err))
                vdes = np.abs(np.diff(des) / np.diff(tt)).max()
                tau = np.abs(T[a:b, c[f"tau_est_{j}_joint"]]).max() if f"tau_est_{j}_joint" in c else np.nan
                res.setdefault((side, j), []).append((lags[k], err[k], err[0], np.abs(qd).max() / max(vdes, 1e-6), np.abs(qd).max(), amp, tau))
        print(path)
        print(f"  {'side joint':28s} {'n':>4s} {'lag ms':>7s} {'err@lag':>8s} {'err@0':>7s} {'peak qd/cmd':>11s} {'peak qd':>8s} {'cmd range':>9s} {'peak tau':>8s}")
        for (side, j), v in sorted(res.items()):
            if len(v) < 5: continue
            m = np.median(np.array(v), 0)
            print(f"  {side} {j:25s} {len(v):4d} {1000 * m[0]:7.0f} {m[1]:8.3f} {m[2]:7.3f} {m[3]:11.2f} {m[4]:8.2f} {m[5]:9.2f} {m[6]:8.1f}")


if __name__ == "__main__":
    main()
