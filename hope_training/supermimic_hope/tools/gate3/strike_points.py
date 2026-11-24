"""Where on the blade each Gate3 ball was struck, and how the returns scatter.

    /opt/drone_venv/bin/python tools/gate3/strike_points.py <run dir> [<run dir> ...]     (grasping container, CPU)

Robot kinematics on the simulator's 200 Hz log give the blade pose at each racket contact; the
ball position from ``pp_physical_ball_report.json`` is expressed in the blade frame (towards the
handle / across). Also prints the ball's speed, elevation and azimuth after contact and the landing
points, overall and per serve placement, and writes ``strike_points.json`` into the run directory.
"""
import sys, csv, json, bisect, math, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train  # noqa: E402,F401 -- sets up sys.path for the pinned dependencies
import numpy as np, torch
from supermimic_hope import a3
from supermimic_hope.core.fk import Kinematics
model, _ = a3.build_model(); kin = Kinematics(model, a3.JOINTS)
wrist = kin.names.index("right_wrist_yaw_Link"); site = np.array(model.site(a3.RACKET_SITE).pos)
handle = np.array([0.060, 0.019, -0.095]) - site; handle[1] = 0; handle /= np.linalg.norm(handle)   # blade centre -> handle, in the blade plane (wrist x-z)
across = np.cross([0, 1, 0], handle)
for R in sys.argv[1:]:
    ball = {int(r["shot_id"]): r for r in json.load(open(R + "/pp_physical_ball_report.json"))["rows"]}
    head = None; rows = []
    with open(R + "/pp_mujoco_plant.csv") as f:
        rd = csv.reader(f); head = {n: i for i, n in enumerate(next(rd))}
        cols = [head[k] for k in ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"]] + [head["q_" + j] for j in a3.JOINTS]
        vel = [head[k] for k in ["racket_vx", "racket_vy", "racket_vz", "racket_x", "racket_y", "racket_z"]]
        for r in rd:
            rows.append((int(r[head["wall_time_ns"]]), [float(r[c]) for c in cols], [float(r[c]) for c in vel]))
    ts = [r[0] for r in rows]
    out = []
    for shot, b in sorted(ball.items()):
        if not b["racket_events"]: continue
        e = b["racket_events"][0]; t = e["stamp_ns"] - int(e.get("edge_observation_lag_s", 0) * 1e9)
        i = min(len(rows) - 1, max(1, bisect.bisect_left(ts, t)))
        a, c = rows[i - 1], rows[i]; w = (t - a[0]) / max(1, c[0] - a[0]); w = min(1.5, max(-0.5, w))
        st = np.array(a[1]) * (1 - w) + np.array(c[1]) * w
        s = torch.zeros(1, 75); s[0, :7] = torch.tensor(st[:7], dtype=torch.float32); s[0, 3:7] /= s[0, 3:7].norm(); s[0, 13:44] = torch.tensor(st[7:], dtype=torch.float32)
        pos, rot = kin(s)[:2]; Rw = rot[0, wrist].numpy(); centre = pos[0, wrist].numpy() + Rw @ site
        rv = np.array(a[2][:3]) * (1 - w) + np.array(c[2][:3]) * w
        # the ball position logged at the first sample showing contact is after the bounce: step it back along its outgoing velocity to the blade plane
        p = np.array(e["position_world"]); vout = np.array(e["velocity_world"])
        local = Rw.T @ (p - centre); lv = Rw.T @ (vout - rv)
        if abs(lv[1]) > 1e-3:
            back = (abs(local[1]) - 0.02) / abs(lv[1]); local = local - lv * max(0.0, min(0.02, back))
        land = b["landing_event"]["position_table"] if b.get("landing_event") else None
        out.append(dict(shot=shot, along_handle=float(local @ handle), across=float(local @ across), normal=float(local[1]), speed=float(np.linalg.norm(rv)),
                        vout=vout.tolist(), legal=bool(b["landing_pass"]), net=bool(b["post_racket_net_events"]), land=land,
                        y=round(b["incoming_table_events"][0]["position_world"][1], 3) if b["incoming_table_events"] else None, force=b["peak_racket_force_n"]))
    A = np.array([[o["along_handle"], o["across"]] for o in out]) * 100; r = np.linalg.norm(A, axis=1)
    print(R.split("supermimic/")[-1])
    print(f"  {len(out)} contacts. Offset from the blade centre (cm): toward handle {A[:,0].mean():+.1f} +- {A[:,0].std():.1f}, across {A[:,1].mean():+.1f} +- {A[:,1].std():.1f}; "
          f"distance mean {r.mean():.1f}, median {np.median(r):.1f}, p90 {np.percentile(r,90):.1f}, max {r.max():.1f}; within 2 cm {np.mean(r<2):.0%}, within 4 cm {np.mean(r<4):.0%}, beyond 6 cm {np.mean(r>6):.0%}")
    V = np.array([o["vout"] for o in out]); sp = np.linalg.norm(V, axis=1); el = np.degrees(np.arctan2(V[:, 2], np.hypot(V[:, 0], V[:, 1]))); az = np.degrees(np.arctan2(V[:, 1], V[:, 0]))
    print(f"  ball after contact: speed {sp.mean():.2f} +- {sp.std():.2f} m/s (min {sp.min():.2f}, max {sp.max():.2f}), elevation {el.mean():.1f} +- {el.std():.1f} deg, azimuth {az.mean():+.1f} +- {az.std():.1f} deg")
    lands = np.array([o["land"][:2] for o in out if o["land"] and o["legal"]])
    print(f"  legal landings: x {lands[:,0].mean():.2f} +- {lands[:,0].std():.2f} m (range {lands[:,0].min():.2f}..{lands[:,0].max():.2f}; net 1.37, far edge 2.74), y {lands[:,1].mean():+.2f} +- {lands[:,1].std():.2f} m (range {lands[:,1].min():+.2f}..{lands[:,1].max():+.2f}; table -1.525..0)")
    by = collections.defaultdict(list)
    for o in out:
        if o["y"] is not None: by[o["y"]].append(o)
    print("  per serve placement (same serve every time): n | offset dist cm | speed after contact | elevation | landing x +- | landing y +-")
    for y in sorted(by):
        g = by[y]; rr = np.array([math.hypot(o["along_handle"], o["across"]) * 100 for o in g]); v = np.array([o["vout"] for o in g])
        ln = np.array([o["land"][:2] for o in g if o["land"] and o["legal"]])
        print(f"    y {y:+.3f}: {len(g):3d} | {rr.mean():.1f} +- {rr.std():.1f} | {np.linalg.norm(v,axis=1).mean():.2f} +- {np.linalg.norm(v,axis=1).std():.2f} | {np.degrees(np.arctan2(v[:,2],np.hypot(v[:,0],v[:,1]))).mean():.1f} +- {np.degrees(np.arctan2(v[:,2],np.hypot(v[:,0],v[:,1]))).std():.1f} | {ln[:,0].mean():.2f} +- {ln[:,0].std():.2f} | {ln[:,1].mean():+.2f} +- {ln[:,1].std():.2f}")
    # does the offset explain the scatter?
    if len(out) > 20:
        c1 = np.corrcoef(r, sp)[0, 1]; c2 = np.corrcoef(A[:, 0], el)[0, 1]; c3 = np.corrcoef(A[:, 1], az)[0, 1]
        print(f"  correlations: offset distance vs outgoing speed {c1:+.2f}; toward-handle offset vs elevation {c2:+.2f}; across offset vs azimuth {c3:+.2f}")
    json.dump(out, open(R + "/strike_points.json", "w"))
