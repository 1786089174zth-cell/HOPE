"""Train the HOPE four-stroke policy with SuperMimic (standalone: MuJoCo-Warp + PyTorch, no Isaac).

    python train.py device=cuda:1 run_name=four_stroke max_iterations=40000

Any ``TrainCfg`` field can be set as ``key=value``; nested ones as ``learn.batch=2048``,
``task.plane_probability=0.35``, ``sim.friction=[0.3,1.2]``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
deps = Path(os.environ.get("HOPE_SUPERMIMIC_DEPS", Path.home() / ".local/share/hope-supermimic/deps"))
if deps.is_dir():
    sys.path.insert(0, str(deps))


def apply_overrides(cfg, args):
    for arg in args:
        key, _, raw = arg.partition("=")
        if not _:
            raise SystemExit(f"expected key=value, got {arg!r}")
        node, parts = cfg, key.split(".")
        for name in parts[:-1]:
            node = getattr(node, name)
        if not hasattr(node, parts[-1]):
            raise SystemExit(f"unknown setting {key!r}")
        current = getattr(node, parts[-1])
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if current is not None and value is not None and not isinstance(value, type(current)):
            raise SystemExit(f"{key}: expected {type(current).__name__}, got {raw!r}")
        setattr(node, parts[-1], value)
    return cfg


def main(argv=None):
    from supermimic_hope.core.trainer import TrainCfg, Trainer

    cfg = apply_overrides(TrainCfg(), sys.argv[1:] if argv is None else argv)
    trainer = Trainer(cfg)
    if cfg.resume:
        trainer.load(cfg.resume)
    out = trainer.train()
    print(f"[supermimic] finished: {out}", flush=True)


if __name__ == "__main__":
    main()
