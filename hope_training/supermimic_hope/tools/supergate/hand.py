"""Let Supergate's runner carry the gripper on the left hand (stdlib only; called by assemble.sh).

    python3 tools/supergate/hand.py <tree>/a3_deploy/a3_deploy_example/scripts/supergate/run.py

The Gate3 MJCF has a bare left hand (left_wrist_yaw_Link 0.281 kg, centre 0.084 m along the link); the robot we
deploy on carries a gripper there (0.847 kg at 0.113 m, main: A3PingPong-with-gripper/urdf/model.urdf), and a policy
trained without it holds with the trunk 4-5 deg to the left on the robot. With ``SUPERGATE_LEFT_HAND=<mass kg>,<x m>``
run.py gives the compiled model that hand before the batch starts; without it nothing changes.
"""

import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
anchor = "    apply_hope_contact_profile(model)\n"
if "SUPERGATE_LEFT_HAND" not in text:
    if text.count(anchor) != 1:
        raise SystemExit("run.py: model set-up line not found")
    text = text.replace(anchor, anchor + (
        "    if os.environ.get('SUPERGATE_LEFT_HAND'):  # the gripper on the left hand (tools/supergate/hand.py)\n"
        "        hand_mass, hand_x = map(float, os.environ['SUPERGATE_LEFT_HAND'].split(','))\n"
        "        hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_wrist_yaw_Link')\n"
        "        ratio = hand_mass / model.body_mass[hand]\n"
        "        model.body_inertia[hand] *= ratio * (hand_x / model.body_ipos[hand, 0]) ** 2\n"
        "        model.body_mass[hand], model.body_ipos[hand, 0] = hand_mass, hand_x\n"
        "        mujoco.mj_setConst(model, mujoco.MjData(model))\n"))
    if "import os\n" not in text:
        text = text.replace("import json\n", "import json\nimport os\n", 1)
    path.write_text(text)
