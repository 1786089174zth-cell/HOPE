"""Export the actor of a ``train.py`` checkpoint to ONNX.

    python export.py logs/supermimic/<run>/model_<N>.pt        # averaged actor (default)
    python export.py logs/supermimic/<run>/model_<N>.pt raw    # the raw iterate

Writes ``exported/policy.onnx`` next to the checkpoint: inputs ``obs`` [1, 110 * frames]
(consecutive hitter_pure frames, oldest first; metadata ``obs_frame_history`` = frames,
which the Runner's PpOnnxPolicy requires for a multi-frame input) and ``time_step`` [1, 1]
(unused); output ``actions`` [1, 31] raw actions. The observation normalization is inside the
graph. The file carries the action contract (joint order, default pose, scale, q_des limits)
but NOT the full hitter_pure metadata and deploy.yaml of a Gate3 bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import train  # noqa: F401 -- sets up sys.path
from evaluate import from_dict


def main(argv=None):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from torch import nn

    from supermimic_hope import a3
    from supermimic_hope.core.nets import OBS, Actor
    from supermimic_hope.core.trainer import TrainCfg

    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(TrainCfg, payload["config"])
    active = torch.ones(31)
    active[[11, 16]] = 0.0
    actor = Actor(cfg.frames, cfg.actor_hidden, active, architecture=cfg.actor_architecture)
    # Ship the averaged actor: single iterates swing by tens of points of hit rate.
    source = "actor_ema" if "actor_ema" in payload and "raw" not in argv[1:] else "actor"
    actor.load_state_dict(payload[source])
    actor.eval()

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = actor

        def forward(self, obs, time_step):
            return self.actor.flat_forward(obs) + 0.0 * time_step

    out = path.parent / "exported"
    out.mkdir(exist_ok=True)
    target = out / "policy.onnx"
    obs = torch.zeros(1, OBS * cfg.frames)
    torch.onnx.export(Policy(), (obs, torch.zeros(1, 1)), str(target), input_names=["obs", "time_step"],
                      output_names=["actions"], opset_version=17, dynamo=False)
    model = onnx.load(str(target))
    csv = lambda x: ",".join(f"{float(v):.9g}" for v in x)  # noqa: E731
    meta = dict(
        obs_frame_history=str(cfg.frames), actor_obs_total_dim=str(OBS * cfg.frames), actor_obs_frame_dim=str(OBS),
        contract="hitter_pure", joint_names=",".join(a3.JOINTS), default_joint_pos=csv(a3.DEFAULT_Q),
        action_scale=csv(a3.ACTION_SCALE), qdes_safe_lower_rad=csv(a3.QDES_LO), qdes_safe_upper_rad=csv(a3.QDES_HI),
        joint_stiffness=csv(a3.KP), joint_damping=csv(a3.KD), passive_joint_names="head_yaw_joint,head_pitch_joint",
        training_backend=str(payload["format"]), training_iteration=str(payload["iteration"]),
        supermimic_actor_architecture=cfg.actor_architecture,
        actor_weights="polyak_average" if source == "actor_ema" else "raw_iterate",
        deploy_bundle="NOT_BUILT: hitter_pure metadata and deploy.yaml are not written by this export",
    )
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(model, str(target))
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(1, OBS * cfg.frames)).astype(np.float32)
    got = session.run(["actions"], {"obs": x, "time_step": np.zeros((1, 1), np.float32)})[0]
    want = actor.flat_forward(torch.from_numpy(x)).detach().numpy()
    error = float(np.abs(got - want).max())
    if error > 1e-4:
        raise RuntimeError(f"ONNX output differs from the actor by {error}")
    print(f"[export] {target} (frames={cfg.frames}, weights={source}, max |onnx - torch| = {error:.2e})")


if __name__ == "__main__":
    main()
