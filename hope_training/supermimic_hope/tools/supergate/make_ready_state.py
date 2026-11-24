"""Ready state for Supergate: the robot settled in the official stand of the Gate3 MJCF.

    .venv/bin/python hope_training/supermimic_hope/tools/supergate/make_ready_state.py --tree <supergate tree> --out <file.npz>

Supergate's ``run.py`` wants an NPZ with ``qpos``, ``qvel`` (model-sized, last row used), ``command``
((5, 31): q_des, dq_des, tau_ff, kp, kd in SDK joint order) and ``qa`` (the SDK joints' qpos addresses).
Its documented sources are serve-trace tools this checkout does not have, so the state is produced
directly: default joint angles, pelvis at the stand height, one second under the official stand PD
(``a3_pd_stand_kps/kds``) with the normal HOPE contact profile, then saved with that stand command.
``run.py`` overwrites the pelvis x, y with its HOME and the harness holds a 20 s warm-up before the first ball.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
XML = REPO / "a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
STAND_KP = dict(waist_yaw=400, waist_roll=500, waist_pitch=500, shoulder_pitch=200, shoulder_roll=200, shoulder_yaw=100,
                elbow=200, wrist_roll=100, wrist_pitch=50, wrist_yaw=50, hip_pitch=1500, hip_roll=400, hip_yaw=300,
                knee=2000, ankle_pitch=500, ankle_roll=500, head_yaw=40, head_pitch=40)
STAND_KD = dict(waist_yaw=4, waist_roll=4, waist_pitch=4, shoulder_pitch=2, shoulder_roll=2, shoulder_yaw=1,
                elbow=1, wrist_roll=1, wrist_pitch=1, wrist_yaw=1, hip_pitch=8, hip_roll=7, hip_yaw=7,
                knee=8, ankle_pitch=5, ankle_roll=5, head_yaw=2, head_pitch=2)


def default_pose():
    """Joint name -> default angle, read from the training package's table (no torch import)."""
    text = (HERE.parents[1] / "supermimic_hope/a3.py").read_text()
    names = "".join(re.findall(r'"([^"]*)"', re.search(r"JOINTS = \((.*?)\)\.split", text, re.S).group(1))).split(",")
    values = "".join(re.findall(r'"([^"]*)"', re.search(r"DEFAULT_Q = _f\((.*?)\)\n", text, re.S).group(1))).split(",")
    return dict(zip(names, map(float, values)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, required=True, help="assembled Supergate tree (tools/supergate/assemble.sh)")
    ap.add_argument("--xml", type=Path, default=XML)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.tree / "a3_deploy/a3_deploy_example/scripts"))
    from runner_sim_profile import apply_hope_contact_profile
    from validate_spin001_serve_mujoco import SDK_ORDER, _compile_model

    model, bindings = _compile_model(args.xml.resolve())
    apply_hope_contact_profile(model)
    bindings = np.asarray(bindings)
    data = mujoco.MjData(model)
    pose = default_pose()
    key = lambda n: n.removesuffix("_joint").removeprefix("left_").removeprefix("right_")  # noqa: E731
    q_des = np.array([pose[n] for n in SDK_ORDER])
    kp = np.array([float(STAND_KP[key(n)]) for n in SDK_ORDER])
    kd = np.array([float(STAND_KD[key(n)]) for n in SDK_ORDER])
    data.qpos[:7] = [-0.5, -0.7625, 1.0684, 1, 0, 0, 0]
    data.qpos[bindings[:, 0]] = q_des
    for slot in range(9):  # park the nine balls far below, as the harness does
        name = "gate3_ball_free_joint" + ("" if slot == 0 else f"_return{slot}")
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        data.qpos[adr:adr + 7] = [100 + slot, 0, -10, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    lo, hi = model.actuator_ctrlrange[bindings[:, 2]].T
    for _ in range(1000):
        tau = kp * (q_des - data.qpos[bindings[:, 0]]) - kd * data.qvel[bindings[:, 1]]
        data.ctrl[bindings[:, 2]] = np.clip(tau, lo, hi)
        for slot in range(9):  # keep the parked balls where they are
            name = "gate3_ball_free_joint" + ("" if slot == 0 else f"_return{slot}")
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 3] = [100 + slot, 0, -10]
            data.qvel[model.jnt_dofadr[j]:model.jnt_dofadr[j] + 6] = 0
        mujoco.mj_step(model, data)
    command = np.stack((q_des, np.zeros(31), np.zeros(31), kp, kd))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, qpos=data.qpos[None].copy(), qvel=data.qvel[None].copy(), command=command[None], qa=bindings[:, 0])
    print(f"{args.out}: pelvis ({data.qpos[0]:+.3f}, {data.qpos[1]:+.3f}, {data.qpos[2]:.3f}), tilt "
          f"{np.degrees(np.arccos(np.clip(1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2), -1, 1))):.2f} deg, "
          f"max |joint - default| {np.abs(data.qpos[bindings[:, 0]] - q_des).max():.3f} rad, nq {model.nq} nv {model.nv}")


if __name__ == "__main__":
    main()
