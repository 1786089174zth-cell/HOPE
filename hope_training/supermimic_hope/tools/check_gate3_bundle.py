"""Check a bundle of ``build_gate3_bundle.py`` against its checkpoint (needs onnxruntime, no Isaac).

    python tools/check_gate3_bundle.py <bundle dir> [<checkpoint.pt>]

Verifies the deploy.yaml/ONNX hash binding, the input/output layout, and that the ONNX
``actions`` equal the checkpoint's actor (averaged unless the bundle says raw) on random input.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402,F401 -- adds the pinned dependency directory (onnxruntime) to sys.path


def main(argv=None):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import yaml

    from supermimic_hope.core.nets import OBS, Actor

    argv = sys.argv[1:] if argv is None else argv
    bundle = Path(argv[0])
    onnx_path, deploy = bundle / "exported/policy.onnx", yaml.safe_load((bundle / "params/deploy.yaml").read_text())
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    if deploy["provenance"]["policy_sha256"] != sha:
        raise SystemExit("deploy.yaml policy_sha256 does not match exported/policy.onnx")
    model = onnx.load(str(onnx_path))
    meta = {p.key: p.value for p in model.metadata_props}
    frames = int(meta["obs_frame_history"])
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    if shape != [1, OBS * frames] or deploy["contracts"]["actor_observation"]["total_dim"] != OBS * frames:
        raise SystemExit(f"observation layout mismatch: onnx {shape}, frames {frames}")
    checkpoint = Path(argv[1]) if len(argv) > 1 else Path(deploy["provenance"]["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != deploy["provenance"]["checkpoint_sha256"]:
        raise SystemExit("checkpoint does not match deploy.yaml checkpoint_sha256")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    active = torch.ones(31)
    active[[11, 16]] = 0.0
    architecture = payload["config"].get("actor_architecture", "single")
    if meta.get("supermimic_actor_architecture", "single") != architecture:
        raise SystemExit("Bundle actor architecture differs from checkpoint")
    actor = Actor(frames, tuple(payload["config"]["actor_hidden"]), active, architecture=architecture)
    actor.load_state_dict(payload["actor" if meta["supermimic_actor_weights"] == "raw_iterate" else "actor_ema"])
    actor.eval()
    # The graph's own bounds (build_gate3_bundle.py --command-clamp, --waist-roll-limit), from its metadata
    lo, hi = torch.full((OBS,), -1e9), torch.full((OBS,), 1e9)
    act_lo, act_hi = torch.full((31,), -1e9), torch.full((31,), 1e9)
    clamp = meta.get("supermimic_command_clamp", "none")
    if clamp != "none":
        parts = dict(part.split("=") for part in clamp.split(";"))
        z, v = [float(c) for c in parts["target_rel_z"].split(",")], [float(c) for c in parts["target_vel"].split(",")]
        lo[105], hi[105] = z
        lo[106:109], hi[106:109] = torch.tensor(v[0::2]), torch.tensor(v[1::2])
    if meta.get("supermimic_waist_roll_limit_rad", "none") != "none":
        from supermimic_hope import a3

        j, limit = list(a3.JOINTS).index("waist_roll_joint"), float(meta["supermimic_waist_roll_limit_rad"])
        act_lo[j], act_hi[j] = (-limit - a3.DEFAULT_Q[j]) / a3.ACTION_SCALE[j], (limit - a3.DEFAULT_Q[j]) / a3.ACTION_SCALE[j]
    lo, hi = lo.repeat(frames)[None], hi.repeat(frames)[None]
    x = np.random.default_rng(1).normal(size=(8, 1, OBS * frames)).astype(np.float32)
    x[4:, :, 105::OBS] *= 4.0  # commands beyond the clamp
    error, changed = 0.0, 0.0
    for row in x:
        got = session.run(["actions"], {"obs": row, "time_step": np.zeros((1, 1), np.float32)})[0]
        raw = actor.flat_forward(torch.from_numpy(row))
        want = torch.maximum(torch.minimum(actor.flat_forward(torch.maximum(torch.minimum(torch.from_numpy(row), hi), lo)), act_hi), act_lo)
        error = max(error, float(np.abs(got - want.detach().numpy()).max()))
        changed = max(changed, float((want - raw).abs().max()))
    if error > 1e-4:
        raise SystemExit(f"ONNX actions differ from the actor by {error}")
    outputs = {o.name: o.shape for o in session.get_outputs()}
    print(f"[bundle] OK frames={frames} weights={meta['supermimic_actor_weights']} iteration="
          f"{meta['supermimic_training_iteration']} clips={meta.get('clip_seg_lengths')} "
          f"max |onnx - torch| = {error:.2e} command clamp={clamp} waist roll limit="
          f"{meta.get('supermimic_waist_roll_limit_rad', 'none')} (largest change by the bounds on the probe {changed:.3f})")
    print(f"[bundle] outputs {outputs}")
    for key in ("hitter_pure_training_recipe", "hitter_pure_runtime_contract", "hitter_pure_target_sample_contract",
                "hitter_pure_station_contract", "clip_strike_phases", "hitter_pure_pos_range_per_clip",
                "hitter_pure_vel_range_per_clip", "hitter_pure_base_target_range", "hitter_pure_hold_pose_contract"):
        print(f"[bundle] {key} = {meta.get(key)}")


if __name__ == "__main__":
    main()
