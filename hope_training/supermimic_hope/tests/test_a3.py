"""Catch Nm limits accidentally becoming position limits during PD conversion."""

from pathlib import Path
import sys
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_hope import a3  # noqa: E402 -- optional physics dependency above


def test_pd_conversion_preserves_physical_torque_limits():
    original = mujoco.MjModel.from_xml_path(str(a3.GATE3_XML))
    expected = {}
    for i in range(original.nu):
        joint = original.joint(int(original.actuator_trnid[i, 0])).name
        expected[joint] = original.actuator_ctrlrange[i]
    model, _ = a3.build_model()
    data = mujoco.MjData(model)
    data.qpos[2] = 2.0
    data.qpos[a3.indices(model)["qpos"]] = a3.DEFAULT_Q
    for sign in (-1, 1):
        data.ctrl[:] = sign * 1.0e6
        mujoco.mj_forward(model, data)
        for i, name in enumerate(a3.JOINTS):
            np.testing.assert_allclose(model.actuator_forcerange[i], expected[name])
            assert expected[name][0] <= data.actuator_force[i] <= expected[name][1]
            assert abs(data.actuator_force[i]) > 1.0
    assert model.nq == 38 and model.nv == 37 and model.nu == 31
    assert not model.actuator_ctrllimited.any()
    assert not any(model.body(i).name.startswith("gate3_ball") for i in range(model.nbody))
    # Vendor-profile sole spheres must not add a second foot contact model.
    assert not any("_foot" in model.geom(i).name for i in range(model.ngeom))
