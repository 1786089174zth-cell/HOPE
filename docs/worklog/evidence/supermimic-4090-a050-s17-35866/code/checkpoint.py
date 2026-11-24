"""Verify a local checkpoint produced by this SuperMimic training run."""
import hashlib
import json
import sys
from pathlib import Path

import torch


def verify(path, expected_run_id, expected_iteration):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    assert checkpoint["format"] == "supermimic_hope_core_v1"
    assert checkpoint["iteration"] == expected_iteration
    assert config["run_name"] == expected_run_id
    assert config["seed"] == 17 and config["learn"]["body_reward_alpha"] == 0.5
    assert config["actor_architecture"] == "dual_body_mlp_v1"
    assert config["num_envs"] == 6144 and config["frames"] == 10
    assert config["capture"] and config["learn"]["horizon"] == 32
    assert checkpoint["actor"].keys() == checkpoint["actor_ema"].keys()
    counts = {}
    for name in ("actor", "actor_ema", "critic", "dynamics"):
        state = checkpoint[name]
        assert state and all(torch.isfinite(tensor).all() for tensor in state.values()), name
        counts[name] = len(state)
    for name in ("opt_pi", "opt_v", "opt_wm"):
        optimizer = checkpoint[name]
        parameters = {p for group in optimizer["param_groups"] for p in group["params"]}
        first_update = 0 if name == "opt_wm" else config["wm_only_iterations"]
        assert parameters, name
        if expected_iteration > first_update:
            assert optimizer["state"].keys() == parameters, name
        else:
            assert not optimizer["state"], name
        for state in optimizer["state"].values():
            assert {"step", "exp_avg", "exp_avg_sq"} <= state.keys(), name
            assert all(torch.isfinite(tensor).all() for tensor in state.values()), name
        counts[name] = len(optimizer["state"])
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return dict(checkpoint=str(path.resolve()), iteration=expected_iteration,
                env_steps=checkpoint["env_steps"], run_id=expected_run_id,
                alpha=config["learn"]["body_reward_alpha"],
                actor_architecture=config["actor_architecture"], entries=counts,
                sha256=digest.hexdigest())


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(verify(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])), indent=2))
