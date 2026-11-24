"""Why Gate3 returns land where they do: commanded stroke vs executed stroke vs the Planner's landing target.

    /opt/drone_venv/bin/python tools/gate3/returns.py <run dir> [<run dir> ...]     (grasping container, CPU)

The Planner asks for a racket velocity (and a face normal along it) that sends the ball to
``target_land`` (2.055, -0.7625 m on the table: the middle of the far half). For every racket
contact this tool takes

* the incoming ball (velocity, spin) from the simulator's trajectory log just before contact,
* the commanded racket velocity from the Runner trace at engage,
* the executed racket state at contact: velocity of the blade at the ball and face normal, from
  robot kinematics on the simulator's 200 Hz log,

and pushes them through the simulator's own paddle model (``gate3_ball_contact_model.h``) and
flight model to a landing point. Three landings per ball: ``ideal`` (commanded stroke),
``model`` (executed stroke through the model; its distance to ``actual`` says how far the
reconstruction can be trusted) and ``actual`` (the simulator's landing event).
``ideal - target`` is the Planner's own residual, ``actual - ideal`` the execution error.
Writes ``returns.json`` into the run directory.
"""

from __future__ import annotations

import bisect
import collections
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train  # noqa: E402,F401 -- sets up sys.path for the pinned dependencies

import numpy as np  # noqa: E402

TARGET = np.array([2.055, -0.7625])
TABLE_Z, RADIUS = 0.76, 0.02
DRAG, MAGNUS = 0.12453915600992894, 0.00379260504480258
A, B, CAP, G1, G2, INERTIA = 0.52, 0.0, 0.5, 0.759, -0.0441, 2.0 / 3.0


def paddle(v_in, v_racket, normal, spin):
    """``PredictPaddleContact``: outgoing ball velocity and spin."""
    n = normal / (np.linalg.norm(normal) + 1e-9)
    if np.dot(v_in - v_racket, n) > 0:
        n = -n
    rel = v_in + np.cross(spin, -RADIUS * n) - v_racket
    ns = float(np.dot(rel, n))
    tan = rel - ns * n
    ts, nsp = float(np.linalg.norm(tan)), abs(ns)
    e = min(0.95, max(0.05, G1 * math.exp(G2 * nsp)))
    cosine = nsp / (math.hypot(ts, ns) + 1e-9)
    impulse = min(max((A + B * cosine) * ts, 0.0), CAP * (1 + e) * nsp)
    d_tan = -tan * impulse / (ts + 1e-9) if ts > 1e-9 else np.zeros(3)
    return v_in - (1 + e) * ns * n + d_tan, spin - np.cross(n, d_tan) / (INERTIA * RADIUS)


def land(p, v, w, dt=0.001):
    """Fly from ``p`` with velocity ``v`` and spin ``w`` to table height: (x, y) there, and the
    height of the ball's lowest point over the net plane minus the net top."""
    p, v = np.array(p, float), np.array(v, float)
    clearance = None
    for _ in range(4000):
        a = np.array([0, 0, -9.81]) - DRAG * np.linalg.norm(v) * v + MAGNUS * np.cross(w, v)
        q = p + v * dt
        if p[0] < 1.37 <= q[0]:
            clearance = q[2] - RADIUS - 0.9125
        v = v + a * dt
        if q[2] <= TABLE_Z + RADIUS and v[2] < 0:
            return q[:2], clearance
        p = q
    return np.array([np.nan, np.nan]), clearance


def analyse(folder: Path, kin, a3, torch):
    wrist = kin.names.index("right_wrist_yaw_Link")
    site = np.array(kin_model.site(a3.RACKET_SITE).pos)
    ball = {int(r["shot_id"]): r for r in json.loads((folder / "pp_physical_ball_report.json").read_text())["rows"]}
    command = {}
    with (folder / "pp_runner_trace.csv").open() as f:
        for r in csv.DictReader(f):
            shot = int(float(r["shot_seq"] or 0))
            if r["mode"] == "3" and r["level"] != "0" and shot and shot not in command:
                command[shot] = dict(v=np.array([float(r[f"target_v{a}"]) for a in "xyz"]),
                                     p=np.array([float(r[f"target_{a}"]) for a in "xyz"]), swing=r["swing_sign"])
    plant = []
    with (folder / "pp_mujoco_plant.csv").open() as f:
        rd = csv.reader(f)
        h = {n: i for i, n in enumerate(next(rd))}
        state = [h[k] for k in ("base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz")] + [h["q_" + j] for j in a3.JOINTS]
        racket = [h[k] for k in ("racket_x", "racket_y", "racket_z", "racket_vx", "racket_vy", "racket_vz", "racket_wx", "racket_wy", "racket_wz")]
        for r in rd:
            plant.append((int(r[h["wall_time_ns"]]), [float(r[c]) for c in state], [float(r[c]) for c in racket]))
    stamps = [p[0] for p in plant]
    before = {}
    with (folder / "pp_ball_trajectories.csv").open() as f:
        for r in csv.DictReader(f):
            if r["active"] == "1" and r["racket_contact_count"] == "0":
                before[int(r["shot_id"])] = r  # the last pre-contact sample of each shot survives
    out = []
    for shot, b in sorted(ball.items()):
        if not b["racket_events"] or shot not in command or shot not in before:
            continue
        e, pre = b["racket_events"][0], before[shot]
        t = e["stamp_ns"] - int(e.get("edge_observation_lag_s", 0) * 1e9)
        i = min(len(plant) - 1, max(1, bisect.bisect_left(stamps, t)))
        lo, hi = plant[i - 1], plant[i]
        w = min(1.5, max(-0.5, (t - lo[0]) / max(1, hi[0] - lo[0])))
        st = np.array(lo[1]) * (1 - w) + np.array(hi[1]) * w
        rk = np.array(lo[2]) * (1 - w) + np.array(hi[2]) * w
        s = torch.zeros(1, 75)
        s[0, :7] = torch.tensor(st[:7], dtype=torch.float32)
        s[0, 3:7] /= s[0, 3:7].norm()
        s[0, 13:44] = torch.tensor(st[7:], dtype=torch.float32)
        pos, rot = kin(s)[:2]
        Rw = rot[0, wrist].numpy()
        normal = Rw[:, 1]
        centre = pos[0, wrist].numpy() + Rw @ site
        v_in = np.array([float(pre[k]) for k in ("vx", "vy", "vz")])
        spin = np.array([float(pre[k]) for k in ("wx", "wy", "wz")])
        p_ball = np.array([float(pre[k]) for k in ("x", "y", "z")])
        v_blade = rk[3:6] + np.cross(rk[6:9], p_ball - centre)  # the blade's velocity where the ball is
        cmd = command[shot]
        cmd_n = cmd["v"] / np.linalg.norm(cmd["v"])
        v_model, w_model = paddle(v_in, v_blade, normal, spin)
        v_ideal, w_ideal = paddle(v_in, cmd["v"], cmd_n, spin)
        v_actual = np.array(e["velocity_world"])
        p0 = np.array(e["position_world"])
        l_ideal, net_ideal = land(cmd["p"], v_ideal, w_ideal)
        l_model, _ = land(p0, v_model, w_model)
        l_from_actual, net_actual = land(p0, v_actual, w_model)
        n_exec = normal if np.dot(normal, cmd_n) > 0 else -normal
        az = lambda u: math.degrees(math.atan2(u[1], u[0]))  # noqa: E731
        el = lambda u: math.degrees(math.atan2(u[2], math.hypot(u[0], u[1])))  # noqa: E731
        out.append(dict(
            shot=shot, y=round(b["incoming_table_events"][0]["position_world"][1], 3) if b["incoming_table_events"] else None,
            swing=cmd["swing"], legal=bool(b["landing_pass"]), net=bool(b["post_racket_net_events"]),
            landing=(b["landing_event"]["position_table"][:2] if b.get("landing_event") else None),
            landing_extrapolated=l_from_actual.tolist(), ideal_landing=l_ideal.tolist(), model_landing=l_model.tolist(),
            net_clearance_actual=net_actual, net_clearance_ideal=net_ideal,
            speed_cmd=float(np.linalg.norm(cmd["v"])), speed_along_cmd=float(np.dot(v_blade, cmd_n)),
            velocity_error=float(np.linalg.norm(v_blade - cmd["v"])),
            normal_error_deg=math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(n_exec, cmd_n)))))),
            normal_azimuth_error_deg=((az(n_exec) - az(cmd_n) + 180) % 360) - 180, normal_elevation_error_deg=el(n_exec) - el(cmd_n),
            position_error=(centre - cmd["p"]).tolist(), contact_minus_target=(p0 - cmd["p"]).tolist(),
            v_out_actual=v_actual.tolist(), v_out_model=v_model.tolist(), v_out_ideal=v_ideal.tolist()))
    return out


def report(folder, rows):
    A_ = lambda key: np.array([r[key] for r in rows], float)  # noqa: E731
    act = np.array([r["landing"] if (r["landing"] and r["legal"]) else r["landing_extrapolated"] for r in rows], float)
    ideal, model = A_("ideal_landing"), A_("model_landing")
    ok = ~np.isnan(act).any(1) & ~np.isnan(ideal).any(1)
    print(f"{folder}\n  {len(rows)} contacts, {sum(r['legal'] for r in rows)} legal. Planner target {TARGET.tolist()} (net x 1.37, far edge 2.74, table y -1.525..0).")
    stat = lambda d: f"x {d[:, 0].mean():+.2f} +- {d[:, 0].std():.2f}, y {d[:, 1].mean():+.2f} +- {d[:, 1].std():.2f} m; distance mean {np.linalg.norm(d, axis=1).mean():.2f}, p90 {np.percentile(np.linalg.norm(d, axis=1), 90):.2f}"  # noqa: E731
    print(f"  landing - target (what you see):          {stat(act[ok] - TARGET)}")
    print(f"  ideal landing - target (Planner's own):   {stat(ideal[ok] - TARGET)}")
    print(f"  landing - ideal landing (execution):      {stat(act[ok] - ideal[ok])}")
    resid = np.linalg.norm(A_("v_out_actual") - A_("v_out_model"), axis=1)
    print(f"  reconstruction check: |outgoing velocity, simulator - model on the executed stroke| mean {resid.mean():.2f}, p90 {np.percentile(resid, 90):.2f} m/s")
    print(f"  executed stroke vs command: speed along the commanded direction {A_('speed_along_cmd').mean():.2f} of {A_('speed_cmd').mean():.2f} m/s "
          f"({(A_('speed_along_cmd') - A_('speed_cmd')).mean():+.2f} +- {(A_('speed_along_cmd') - A_('speed_cmd')).std():.2f}); |velocity error| {A_('velocity_error').mean():.2f} m/s; "
          f"face normal off {A_('normal_error_deg').mean():.1f} deg (azimuth {A_('normal_azimuth_error_deg').mean():+.1f} +- {A_('normal_azimuth_error_deg').std():.1f}, "
          f"elevation {A_('normal_elevation_error_deg').mean():+.1f} +- {A_('normal_elevation_error_deg').std():.1f})")
    by = collections.defaultdict(list)
    for r, a, i in zip(rows, act, ideal):
        by[(r["y"], r["swing"])].append((r, a, i))
    print("  per placement and side:  n legal | landing - target x, y (sd) | ideal - target x, y | speed along cmd - cmd | normal azimuth err | elevation err")
    for (y, swing), g in sorted(by.items(), key=lambda kv: (kv[0][0] is None, kv[0])):
        d = np.array([a - TARGET for _, a, _ in g])
        di = np.array([i - TARGET for _, _, i in g])
        sp = np.array([r["speed_along_cmd"] - r["speed_cmd"] for r, _, _ in g])
        na = np.array([r["normal_azimuth_error_deg"] for r, _, _ in g])
        ne = np.array([r["normal_elevation_error_deg"] for r, _, _ in g])
        print(f"    y {y:+.3f} {'backhand' if swing == '-1' else 'forehand'}: {len(g):3d} {sum(r['legal'] for r, _, _ in g):3d} | "
              f"{np.nanmean(d[:, 0]):+.2f} ({np.nanstd(d[:, 0]):.2f}), {np.nanmean(d[:, 1]):+.2f} ({np.nanstd(d[:, 1]):.2f}) | {np.nanmean(di[:, 0]):+.2f}, {np.nanmean(di[:, 1]):+.2f} | "
              f"{sp.mean():+.2f} +- {sp.std():.2f} m/s | {na.mean():+5.1f} +- {na.std():.1f} deg | {ne.mean():+5.1f} +- {ne.std():.1f} deg")


def main(argv=None):
    import torch
    from supermimic_hope import a3
    from supermimic_hope.core.fk import Kinematics

    global kin_model
    kin_model, _ = a3.build_model()
    kin = Kinematics(kin_model, a3.JOINTS)
    for arg in (sys.argv[1:] if argv is None else argv):
        rows = analyse(Path(arg), kin, a3, torch)
        report(arg, rows)
        (Path(arg) / "returns.json").write_text(json.dumps(rows) + "\n")


if __name__ == "__main__":
    main()
