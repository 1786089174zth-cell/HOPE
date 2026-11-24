"""Paired sensor-noise audit of a frozen HOPE ONNX (not a hardware diagnosis)."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


def audit(path, hold_path, samples=2048):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    graph = onnx.load(str(path))
    meta = {p.key: p.value for p in graph.metadata_props}
    task = json.loads(meta["hitter_pure_resolved_task_json"])
    arrays = {
        p.name: onnx.numpy_helper.to_array(p).copy() for p in graph.graph.initializer
    }
    # Only support the verified folded-normalizer affine/ELU export. Check all
    # outputs against ORT before using Torch to vectorize the paired probe.
    layers = []
    for i in (0, 2, 4, 6):
        w, b = arrays[f"actor.{i}.weight"], arrays[f"actor.{i}.bias"]
        layer = nn.Linear(w.shape[1], w.shape[0])
        layer.weight.data.copy_(torch.from_numpy(w))
        layer.bias.data.copy_(torch.from_numpy(b))
        layers.append(layer)
        if i != 6:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(0)
    parity = 0.0
    for _ in range(8):
        x = rng.normal(size=(1, 110)).astype(np.float32)
        feed = {
            i.name: (x if i.name == "obs" else np.zeros([1, 1], np.float32))
            for i in session.get_inputs()
        }
        expected = session.run(["actions"], feed)[0]
        parity = max(
            parity,
            float(np.abs(expected - actor(torch.from_numpy(x)).detach().numpy()).max()),
        )
    assert parity < 1.0e-3, parity

    def parse(k):
        return torch.tensor([float(x) for x in meta[k].split(",")])

    scale, default = parse("action_scale"), parse("default_joint_pos")
    lo, hi = parse("qdes_safe_lower_rad"), parse("qdes_safe_upper_rad")
    hold = json.loads(Path(hold_path).read_text())
    by_name = dict(zip(hold["joint_order"], hold["actual_q_reference_rad"]))
    q = torch.tensor([by_name[n] for n in meta["joint_names"].split(",")])
    obs = torch.zeros(samples, 110)
    obs[:, 3:34] = q - default
    obs[:, 65:96] = (q - default) / scale
    obs[:, 65 + 11] = 0
    obs[:, 65 + 16] = 0
    obs[:, 98] = -1
    obs[:, 99] = 1
    position = task["racket"]["pos_range_per_clip"]["forehand"]
    velocity = task["racket"]["vel_range_per_clip"]["forehand"]
    obs[:, 103:106] = torch.tensor([sum(position[k]) / 2 for k in ("x", "y", "z")])
    obs[:, 105] -= hold["base_reference_qpos"][2]
    obs[:, 106:109] = torch.tensor([sum(velocity[k]) / 2 for k in ("x", "y", "z")])
    obs[:, 109] = 1.2
    active = torch.ones(31, dtype=torch.bool)
    active[[11, 16]] = False

    def decode(x):
        out = (default + scale * actor(x)).clamp(lo, hi)
        out[:, ~active] = default[~active]
        return out

    result = {}
    with torch.no_grad():
        clean = decode(obs)
        for name in (
            "mocap_1mm",
            "joint_pos_001_uniform",
            "joint_vel_05_uniform",
            "gyro_02_uniform",
        ):
            noisy = obs.clone()
            if name == "mocap_1mm":
                e = torch.randn(samples, 3) * 0.001
                noisy[:, 101:103] -= e[:, :2]
                noisy[:, 103:106] -= e
            else:
                a, b, s = {
                    "joint_pos_001_uniform": (3, 34, 0.01),
                    "joint_vel_05_uniform": (34, 65, 0.5),
                    "gyro_02_uniform": (0, 3, 0.2),
                }[name]
                noisy[:, a:b] += (torch.rand(samples, b - a) * 2 - 1) * s
            delta = decode(noisy) - clean
            rms = delta.square().mean(0).sqrt()
            j = int(rms.argmax())
            result[name] = dict(
                active_joint_rms_mrad=float(
                    delta[:, active].square().mean().sqrt() * 1000
                ),
                worst_joint=meta["joint_names"].split(",")[j],
                worst_joint_rms_mrad=float(rms[j] * 1000),
                equivalent_kp_delta_rms_nm=float(
                    (delta * parse("joint_stiffness"))[:, j].square().mean().sqrt()
                ),
            )
    return dict(
        onnx_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        run=meta["run_path"],
        checkpoint_sha256=meta["hitter_pure_checkpoint_sha256"],
        onnx_torch_parity_max_abs=parity,
        scope="synthetic fixed supplied HOLD, fixed previous action, paired sensor perturbations; no closed-loop or motor-current proof",
        embedded_action_config=task["action"],
        results=result,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("onnx")
    p.add_argument("--hold", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = audit(args.onnx, args.hold)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
