"""Let Supergate's plant run with actuators identified on the robot (stdlib only; called by assemble.sh).

    python3 tools/supergate/actuators.py <tree>/a3_deploy/a3_deploy_example/scripts/supergate/run.py

``tools/real/replay_joints.py --fit`` replays a real session's joint commands in the model and fits, per joint, factors
on the MJCF's passive damping, dry friction and rotor inertia and on the PD stiffness the motor realizes. With
``SUPERGATE_ACTUATORS=<that json>`` run.py applies them to the compiled model before the batch starts; without it
nothing changes. A stiffness factor ``s`` is realized as the motor's gear (the Runner still commands its own kp), and
the extra damping ``(s - 1) * kd`` that scales with it is taken off the joint's passive damping.
"""

import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
anchor = "    q[:2] = [float(os.environ.get('SUPERGATE_HOME_X', '-.5')), -.7625]"
if "SUPERGATE_ACTUATORS" not in text:
    if text.count(anchor) != 1:
        raise SystemExit("run.py: HOME line not found (assemble.sh edits it first)")
    text = text.replace(anchor, (
        "    if os.environ.get('SUPERGATE_ACTUATORS'):  # actuators identified on the robot (tools/supergate/actuators.py)\n"
        "        fitted = json.loads(Path(os.environ['SUPERGATE_ACTUATORS']).read_text())['params']['joints']\n"
        "        for name, scale in fitted.items():\n"
        "            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name + '_joint')\n"
        "            motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + '_joint_motor')\n"
        "            dof = model.jnt_dofadr[joint]\n"
        "            row = [i for i in range(len(bindings)) if bindings[i, 1] == dof][0]\n"
        "            s = scale.get('kp', 1.0)\n"
        "            model.dof_damping[dof] = max(model.dof_damping[dof] * scale.get('damping', 1.0) - (s - 1.0) * scale.get('policy_kd', command[4, row]), 0.0)\n"
        "            model.dof_frictionloss[dof] *= scale.get('friction', 1.0)\n"
        "            model.dof_armature[dof] *= scale.get('armature', 1.0)\n"
        "            model.actuator_gear[motor, 0] *= s\n"
        "        mujoco.mj_setConst(model, mujoco.MjData(model))\n") + anchor)
    path.write_text(text)
