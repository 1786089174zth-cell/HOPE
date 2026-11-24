"""Actuator identification: the robot's recorded joint commands replayed in the training model (mujoco, numpy).

    /opt/drone_venv/bin/python -I tools/real/replay_joints.py <runner_trace.csv> [...] [--out <json>] [--max-strokes N]
        [--set name=value ...] [--scan]                                             (grasping container)

For every stroke of a real session (Runner trace: ``des_*`` commands, ``q_*`` / ``qd_*`` measurements, PD gains,
mocap base pose at 50 Hz) the model starts from the measured joint state 0.1 s before the stroke engages and
receives the recorded commands tick by tick for 0.7 s. The pelvis is moved along its recorded path (the legs and
the floor play no part), so what is compared is the response of the waist and the right arm to the same commands
with the same base motion. Reported per joint: RMS of simulated minus measured angle, and the simulated lag behind
the measured angle (positive: the model is slower than the robot).

Model variants (``--set``; each multiplies the MJCF value of the waist and right-arm joints unless noted):
  delay=<s>          transport delay of the command (0 = applied at the tick it was computed; training: 0 or 0.02)
  damping=<x>        passive joint damping        friction=<x>   dry friction (frictionloss)
  armature=<x>       rotor inertia                kp=<x> kd=<x>  PD gains
  dt=<s>             physics step (default 0.001; the training simulator steps 0.005)
``--scan`` runs the baseline and one-at-a-time variants and prints the total error of each.
``--fit`` fits per-joint factors (damping, dry friction, kp, rotor inertia) to the given traces by coordinate descent and
reports them on the traces given with ``--validate`` (held out); ``--params <json of --fit>`` replays with fitted factors.
Columns per joint: RMS mrad / lag of the model behind the robot / peak speed model over robot.
"""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DEPS = Path.home() / ".local/share/hope-supermimic/deps"  # the trainer's pinned mujoco (train.py)
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
import mujoco  # noqa: E402

from supermimic_hope import a3  # noqa: E402

FIT = ("waist_yaw", "waist_roll", "waist_pitch", "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
       "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw")
PRE, LENGTH = 5, 35  # ticks before the engage, ticks replayed


def load(path, limit=None):
    """Stroke windows of a Runner trace: dict of arrays (ticks, 31) in ``a3.JOINTS`` order, base pose, time."""
    r = csv.reader(open(path))
    h = next(r)
    ix = {n: i for i, n in enumerate(h)}
    names = [j for j in a3.JOINTS]
    cols = {k: [ix[f"{k}_{j}"] for j in names] for k in ("des", "q", "qd", "kp", "kd")}
    base = [ix[k] for k in ("base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz")]
    rows = [row for row in r if len(row) >= len(h) and row[ix["mode"]] == "3"]
    level = np.array([row[ix["level"]] not in ("", "0") for row in rows])
    t = np.array([int(row[ix["wall_time_ns"]]) for row in rows]) * 1e-9
    starts = [i + 1 for i in np.flatnonzero(~level[:-1] & level[1:])]
    out = []
    for a in starts:
        lo, hi = a - PRE, a - PRE + LENGTH + 1
        if lo < 1 or hi > len(rows) or np.any(np.abs(np.diff(t[lo:hi]) - 0.02) > 0.008):
            continue
        get = lambda idx: np.array([[float(rows[i][c] or "nan") for c in idx] for i in range(lo, hi)])  # noqa: E731
        w = {k: get(v) for k, v in cols.items()}
        w["base"], w["t"] = get(base), t[lo:hi] - t[lo]
        w["side"] = "FH" if float(rows[a][ix["swing_sign"]] or 0) > 0 else "BH"
        if all(np.isfinite(v).all() for k, v in w.items() if k not in ("side",)):
            out.append(w)
        if limit and len(out) >= limit:
            break
    return out


def build(params):
    model, _ = a3.build_model()
    model.opt.timestep = float(params.get("dt", 0.001))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # the pelvis is driven; the floor plays no part
    # The pelvis follows its recorded path. Writing its position alone would leave the robot in free fall between
    # writes (no gravity load on any joint); with an enormous inertia on the root it does not accelerate at all.
    model.dof_armature[:6] = 1e8
    ix = a3.indices(model)
    fit = [a3.JOINTS.index(j + "_joint") for j in FIT]
    dof = ix["dof"][fit]
    model.dof_damping[dof] *= float(params.get("damping", 1.0))
    model.dof_frictionloss[dof] *= float(params.get("friction", 1.0))
    model.dof_armature[dof] *= float(params.get("armature", 1.0))
    for name, scales in params.get("joints", {}).items():  # per-joint factors on top (``--fit``)
        d = ix["dof"][a3.JOINTS.index(name + "_joint")]
        model.dof_damping[d] *= scales.get("damping", 1.0)
        model.dof_frictionloss[d] *= scales.get("friction", 1.0)
        model.dof_armature[d] *= scales.get("armature", 1.0)
    return model, ix, fit


def replay(model, ix, fit, windows, params):
    """Simulated joint angles (strokes, ticks + 1, 31) for the recorded commands."""
    delay = float(params.get("delay", 0.0))
    kp_scale, kd_scale = np.ones(a3.NJ), np.ones(a3.NJ)
    kp_scale[fit], kd_scale[fit] = float(params.get("kp", 1.0)), float(params.get("kd", 1.0))
    for name, scales in params.get("joints", {}).items():
        j = a3.JOINTS.index(name + "_joint")
        kp_scale[j] *= scales.get("kp", 1.0)
        kd_scale[j] *= scales.get("kd", 1.0)
    sub = int(round(0.02 / model.opt.timestep))
    data = mujoco.MjData(model)
    result = []
    for w in windows:
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = w["base"][0]
        data.qpos[ix["qpos"]], data.qvel[ix["dof"]] = w["q"][0], w["qd"][0]
        sim = [w["q"][0].copy()]
        for k in range(LENGTH):
            kp, kd = w["kp"][k] * kp_scale, w["kd"][k] * kd_scale
            b0, b1 = w["base"][k], w["base"][k + 1]
            for s in range(sub):
                # the command of tick k takes effect ``delay`` after the tick; until then the previous one holds
                late = (s * model.opt.timestep < delay) and k > 0
                steps_back = int(np.ceil((delay - s * model.opt.timestep) / 0.02)) if late else 0
                des = w["des"][max(k - steps_back, 0)]
                model.actuator_gainprm[:, 0] = kp
                model.actuator_biasprm[:, 1] = -kp
                model.actuator_biasprm[:, 2] = -kd
                data.ctrl[:] = des
                f = (s + 1) / sub
                data.qpos[:3] = b0[:3] + f * (b1[:3] - b0[:3])
                q = b0[3:] + f * ((b1[3:] if np.dot(b0[3:], b1[3:]) >= 0 else -b1[3:]) - b0[3:])
                data.qpos[3:7] = q / np.linalg.norm(q)
                data.qvel[:3] = (b1[:3] - b0[:3]) / 0.02
                data.qvel[3:6] = 0.0
                mujoco.mj_step(model, data)
            sim.append(data.qpos[ix["qpos"]].copy())
        result.append(np.array(sim))
    return np.array(result)


def lag_of(sim, real, t):
    """Shift (s) that best overlays the simulated angle on the measured one; positive: the model is behind."""
    best, at = None, 0.0
    for lag in np.arange(-0.06, 0.0601, 0.004):
        err = np.mean((np.interp(t[4:-4] + lag, t, sim) - real[4:-4]) ** 2)
        if best is None or err < best:
            best, at = err, lag
    return at


def score(sim, windows, fit):
    real = np.array([w["q"] for w in windows])
    t = windows[0]["t"]
    rows = {}
    for j in fit:
        moving = [n for n, w in enumerate(windows) if np.ptp(w["des"][:, j]) > 0.15]
        if len(moving) < 5:
            continue
        rms = float(np.sqrt(np.mean((sim[moving][:, :, j] - real[moving][:, :, j]) ** 2)))
        lag = float(np.median([lag_of(sim[n, :, j], real[n, :, j], np.arange(LENGTH + 1) * 0.02) for n in moving]))
        peak = float(np.median([np.abs(np.diff(sim[n, :, j])).max() / max(np.abs(np.diff(real[n, :, j])).max(), 1e-6) for n in moving]))
        rows[a3.JOINTS[j].removesuffix("_joint")] = dict(strokes=len(moving), rms=rms, lag=lag, peak_speed_ratio=peak)
    total = float(np.sqrt(np.mean([r["rms"] ** 2 for r in rows.values()])))
    return rows, total


GRID = {"damping": (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0), "friction": (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0), "armature": (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0),
        "kp": (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)}


def fit_joints(windows, params, passes=2):
    """Per-joint factors on damping, dry friction, rotor inertia and kp by coordinate descent: each factor is the
    grid value with the least RMS of that joint's own angle; two passes over all joints (the arm's joints load
    each other)."""
    params = dict(params, joints={j: {} for j in FIT})

    def evaluate(p):
        model, ix, fit = build(p)
        rows, total = score(replay(model, ix, fit, windows, p), windows, fit)
        return rows, total

    rows, total = evaluate(params)
    print(f"start: total RMS {1000 * total:.1f} mrad", flush=True)
    for n in range(passes):
        for key in ("damping", "friction", "kp", "armature"):
            # one sweep of this factor for all joints at once: each joint keeps the value best for itself
            trials = {}
            for v in GRID[key]:
                p = dict(params, joints={j: dict(params["joints"][j], **{key: v}) for j in FIT})
                trials[v] = evaluate(p)[0]
            for j in FIT:
                if j in rows:
                    best = min(GRID[key], key=lambda v: trials[v][j]["rms"])
                    if trials[best][j]["rms"] < 0.98 * rows[j]["rms"] or params["joints"][j].get(key) is not None:
                        params["joints"][j][key] = best
            rows, total = evaluate(params)
            print(f"pass {n + 1} {key}: total RMS {1000 * total:.1f} mrad | " + " ".join(f"{j.replace('right_', 'r_').replace('shoulder', 'sh')}={params['joints'][j].get(key, 1.0):g}" for j in FIT), flush=True)
    return params, rows, total


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opt = types.SimpleNamespace(out=None, limit=None, scan=False, fit=False, validate=[], params={})
    files = []
    while argv:
        a = argv.pop(0)
        if a == "--out":
            opt.out = argv.pop(0)
        elif a == "--max-strokes":
            opt.limit = int(argv.pop(0))
        elif a == "--scan":
            opt.scan = True
        elif a == "--fit":
            opt.fit = True
        elif a == "--validate":
            opt.validate.append(argv.pop(0))
        elif a == "--params":
            opt.params.update(json.loads(Path(argv.pop(0)).read_text())["params"])
        elif a == "--set":
            while argv and "=" in argv[0]:
                k, v = argv.pop(0).split("=")
                opt.params[k] = float(v)
        else:
            files.append(a)
    windows = [w for f in files for w in load(f, opt.limit)]
    print(f"{len(windows)} strokes ({sum(w['side'] == 'FH' for w in windows)} forehand)")
    def report(tag, p, ws):
        model, ix, fit = build(p)
        sim = replay(model, ix, fit, ws, p)
        out = {}
        for side in ("all", "BH", "FH"):
            sel = [n for n, w in enumerate(ws) if side == "all" or w["side"] == side]
            if len(sel) >= 5:
                rows, total = score(sim[sel], [ws[n] for n in sel], fit)
                out[side] = dict(total_rms=total, joints=rows)
                print(f"{tag:24s} {side:3s} n {len(sel):3d} total RMS {1000 * total:6.1f} mrad | " +
                      " ".join(f"{k.replace('right_', 'r_').replace('shoulder', 'sh')} {1000 * v['rms']:.0f}/{1000 * v['lag']:+.0f}ms/{v['peak_speed_ratio']:.2f}" for k, v in rows.items()), flush=True)
        return out

    if opt.fit:
        fitted, rows, total = fit_joints(windows, opt.params)
        held = [w for f in opt.validate for w in load(f, opt.limit)]
        for name in fitted["joints"]:
            fitted["joints"][name]["policy_kd"] = float(windows[0]["kd"][0, a3.JOINTS.index(name + "_joint")])
        result = dict(params=fitted, fit_files=files, validate_files=opt.validate,
                      fit=dict(before=report("fit data, before", opt.params, windows), after=report("fit data, fitted", fitted, windows)))
        if held:
            result["validation"] = dict(before=report("held out, before", opt.params, held), after=report("held out, fitted", fitted, held))
        print(json.dumps(fitted))
        if opt.out:
            Path(opt.out).write_text(json.dumps(result, indent=1) + "\n")
        return
    variants = [dict(opt.params)]
    if opt.scan:
        base = dict(opt.params)
        for key, values in (("delay", (0.01, 0.02, 0.03)), ("damping", (0.0, 0.25, 0.5, 2.0)), ("friction", (0.0, 0.5, 2.0)),
                            ("armature", (0.5, 2.0, 4.0)), ("kp", (0.85, 1.15, 1.3)), ("kd", (0.5, 0.75, 1.5)), ("dt", (0.005,))):
            variants += [dict(base, **{key: v}) for v in values if base.get(key) != v]
    results = []
    for params in variants:
        model, ix, fit = build(params)
        sim = replay(model, ix, fit, windows, params)
        rows, total = score(sim, windows, fit)
        results.append(dict(params=params, total_rms=total, joints=rows))
        print(f"{json.dumps(params):40s} total RMS {1000 * total:6.1f} mrad | " +
              " ".join(f"{k.replace('right_', 'r_').replace('shoulder', 'sh')} {1000 * v['rms']:.0f}/{1000 * v['lag']:+.0f}ms" for k, v in rows.items()), flush=True)
    if opt.out:
        Path(opt.out).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
