"""A3 robot for MuJoCo-Warp training, built from the Gate3 validation MJCF.

The deploy ABI (31 joints in Isaac articulation order, affine q_des = default + scale * a, A3 message
PD gains) is copied from the exported policy metadata so a trained actor runs on the existing runner.
Model edits (MjSpec), all recorded in ``model_receipt()``:
  * balls, table, net and the vendor-profile sole spheres removed; sensors removed (the runner reads q/qd/IMU, not MuJoCo sensors);
  * 5 ms implicitfast steps (SuperMimic: 4 x 5 ms per 20 ms control tick); noslip off;
  * the 31 torque motors become PD position actuators (kp, kv = A3 message gains) so the damping term
    is integrated implicitly; source motor torque limits become ``forcerange`` limits and
    the XML passive joint damping/friction loss/armature are initially kept;
  * robot-robot collision disabled (SuperMimic and our Isaac training both do this), robot-floor kept.

WarpBackend then transfers resolved per-environment Isaac properties. It puts
total Isaac drive damping into kv and zeros the separate passive term, so
damping is applied exactly once.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
GATE3_XML = REPO / "a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
GATE3_XML_SHA256 = "8b06071a1a8ea60f927fa3b146007091e5348b20d979fa9db18568b7f40b4e8f"

JOINTS = ("left_hip_pitch_joint,right_hip_pitch_joint,waist_yaw_joint,left_hip_roll_joint,right_hip_roll_joint,"
          "waist_roll_joint,left_hip_yaw_joint,right_hip_yaw_joint,waist_pitch_joint,left_knee_joint,right_knee_joint,"
          "head_yaw_joint,left_shoulder_pitch_joint,right_shoulder_pitch_joint,left_ankle_pitch_joint,right_ankle_pitch_joint,"
          "head_pitch_joint,left_shoulder_roll_joint,right_shoulder_roll_joint,left_ankle_roll_joint,right_ankle_roll_joint,"
          "left_shoulder_yaw_joint,right_shoulder_yaw_joint,left_elbow_joint,right_elbow_joint,left_wrist_roll_joint,"
          "right_wrist_roll_joint,left_wrist_pitch_joint,right_wrist_pitch_joint,left_wrist_yaw_joint,right_wrist_yaw_joint").split(",")
NJ = len(JOINTS)  # 31, deploy/Isaac articulation order
PASSIVE = tuple(i for i, n in enumerate(JOINTS) if n.startswith("head_"))  # held at default, action forced to 0
_f = lambda s: np.array([float(x) for x in s.split(",")])  # noqa: E731
DEFAULT_Q = _f("-0.1311,-0.1311,0,0.0056,-0.0056,0,-0.0348,0.0348,0,0.2468,0.2468,0,0.3,0.3,-0.1204,-0.1204,0,0.12,-0.12,"
               "-0.0078,0.0078,0,0,0.8,0.8,0,0,0,0,0,0")
ACTION_SCALE = _f("0.6875,0.6875,0.647058845,0.458333343,0.458333343,0.23,0.6875,0.6875,0.59,0.32,0.32,0.0375,0.375,0.375,"
                  "0.591,0.591,0.0375,0.375,0.375,0.27375,0.27375,0.2,0.2,0.2,0.2,0.2,0.2,0.075,0.075,0.075,0.075")
KP = _f("80,80,85,120,120,50,80,80,50,250,250,40,40,40,50,50,40,40,40,50,50,30,30,30,30,30,30,20,20,20,20")
KD = _f("3,3,3,4,4,2,3,3,2,8,8,2,3,3,2,2,2,3,3,2,2,2,2,2,2,2,2,2,2,2,2")
QDES_LO = _f("-2.2410028,-2.2410028,-2.35619426,-0.417133689,-1.49923766,-0.314159214,-2.45044231,-2.45044231,-0.443313599,"
             "0.00872659683,0.00872659683,-0.942477822,-2.59181356,-2.59181356,-0.836012602,-0.836012602,-0.401425689,"
             "0.0479966402,-2.48273039,-0.314159214,-0.314159214,-2.51327372,-2.51327372,-0.82466799,-0.82466799,"
             "-2.51327372,-2.51327372,-1.46084046,-1.46084046,-1.46084046,-1.46084046")
QDES_HI = _f("2.65988183,2.65988183,2.35619426,1.49923766,0.417133689,0.314159214,2.45044231,2.45044231,0.373500466,"
             "2.36492085,2.36492085,0.942477822,2.59181356,2.59181356,0.452040285,0.452040285,0.226892799,2.48273039,"
             "-0.0479966402,0.314159214,0.314159214,2.51327372,2.51327372,1.61006618,1.61006618,2.51327372,2.51327372,"
             "1.46084046,1.46084046,1.46084046,1.46084046")
ROOT_BODY = "pelvis_link"
RACKET_SITE = "right_racket"
FEET = ("left_ankle_roll_Link", "right_ankle_roll_Link")
CONTROL_DT, SIM_DT = 0.02, 0.005
REMOVE_GEOMS = ("gate3_table_visual", "gate3_table_collision", "gate3_net_collision")
SOLE_SPHERE = re.compile(r"(left|right)_foot\d+_collision")


def build_model(xml: Path = GATE3_XML, check_sha: bool = True) -> tuple[mujoco.MjModel, dict]:
    raw = Path(xml).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if check_sha and sha != GATE3_XML_SHA256:
        raise ValueError(f"A3 MJCF changed ({sha}); re-validate the training model before use")
    spec = mujoco.MjSpec.from_file(str(xml))
    # The serve ball and the eight return balls (gate3_ball, gate3_ball_return1..8).
    for b in list(spec.bodies):
        if b.name.startswith("gate3_ball"):
            spec.delete(b)
    # The HumanLike sole spheres (<side>_foot<N>_collision) belong to the vendor
    # plant profile only; the normal HOPE profile masks them after load and keeps
    # the ankle-roll hull, which is also what Isaac collides.
    for g in list(spec.geoms):
        if g.name in REMOVE_GEOMS or SOLE_SPHERE.fullmatch(g.name or ""):
            spec.delete(g)
    for p in list(spec.pairs):
        spec.delete(p)
    for s in list(spec.sensors):
        spec.delete(s)
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.noslip_iterations = 0
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    # robot-robot contact off, robot-floor on (floor contype=4 conaffinity=1 in the Gate3 XML)
    for g in spec.geoms:
        if g.name == "floor":
            g.contype, g.conaffinity = 1, 2
        elif g.contype or g.conaffinity:
            g.contype, g.conaffinity = 2, 1
    # PD position actuators in deploy joint order (ctrl = q_des)
    effort = {a.target: np.array(a.ctrlrange).copy() for a in spec.actuators}
    for a in list(spec.actuators):
        spec.delete(a)
    for j, name in enumerate(JOINTS):
        act = spec.add_actuator(name=name + "_pd", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = KP[j]
        act.biasprm[:3] = [0.0, -KP[j], -KD[j]]
        act.ctrllimited = False
        # The source torque motors use ctrlrange. Position control must clamp
        # physical torque, not mistake those Nm limits for position limits.
        act.forcelimited = True
        act.forcerange = effort[name]
    model = spec.compile()
    receipt = model_receipt(model, sha)
    receipt["source_xml"] = str(Path(xml).resolve())
    return model, receipt


def model_receipt(model: mujoco.MjModel, sha: str) -> dict:
    jid = [model.joint(n).id for n in JOINTS]
    return {"source_xml": str(GATE3_XML), "source_sha256": sha, "timestep": model.opt.timestep,
            "integrator": "implicitfast", "nq": model.nq, "nv": model.nv, "nu": model.nu, "ngeom": model.ngeom,
            "nbody": model.nbody, "mass_kg": float(model.body_mass.sum()),
            "armature": model.dof_armature[[model.jnt_dofadr[j] for j in jid]].tolist(),
            "passive_damping": model.dof_damping[[model.jnt_dofadr[j] for j in jid]].tolist(),
            "frictionloss": model.dof_frictionloss[[model.jnt_dofadr[j] for j in jid]].tolist(),
            "kp": KP.tolist(), "kd": KD.tolist()}


def indices(model: mujoco.MjModel) -> dict:
    jid = [model.joint(n).id for n in JOINTS]
    return {"qpos": np.array([model.jnt_qposadr[j] for j in jid]), "dof": np.array([model.jnt_dofadr[j] for j in jid]),
            "root": model.body(ROOT_BODY).id, "racket_site": model.site(RACKET_SITE).id,
            "feet": np.array([model.body(n).id for n in FEET]), "torso": model.body("torso_Link").id}
