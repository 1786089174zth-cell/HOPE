"""Generate deterministic safety probes for the published model19000 diagnostic.

These scenarios deliberately include balls outside the normal return envelope.
They measure whether the robot stays upright; serve validity and returns remain
separate diagnostic fields in the evaluator.
"""
import argparse
from pathlib import Path

import numpy as np
import yaml


def make_scene(mode, seed):
    rng = np.random.default_rng(seed)
    nominal = [
        [2.4, y, 0.49, -3.0, 0.0, 2.2]
        for y in (-1.0225, -0.7125, -1.0575, -0.6575, -1.0925, -0.6125)
    ]
    cycle = []
    for i in range(94):
        if mode == "wide":
            y = float(rng.uniform(-1.5, -0.2))
            vx, vy = -3.0, 0.0
        elif mode == "oblique":
            y = float(rng.choice((-1.05, -0.92, -0.78, -0.65)))
            vx = float(rng.choice((-2.5, -3.0, -3.5, -4.0)))
            vy = float(rng.choice((-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)))
        else:
            raise ValueError(mode)
        cycle.append([2.4, round(y, 6), 0.49, vx, vy, 2.2])
    return {
        "schema_version": 1,
        "family": "fixed_home",
        "description": (
            f"model19000 {mode} safety probe, seed {seed}; first six nominal, "
            "94 perturbed serves; misses allowed, falls are failures"
        ),
        "balls": 100,
        "clean_serves": nominal,
        "rapid_cycle": cycle,
        "timing": {
            "clean_flight_s": 2.5,
            "rapid_flight_s": 1.15,
            "final_flight_s": 2.5,
            "clean_pause_s": 1.0,
            "rapid_pause_s": 0.0,
        },
        "metrics": {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("wide", "oblique"):
        path = args.output_dir / f"model19000-{mode}-100.yaml"
        path.write_text(yaml.safe_dump(make_scene(mode, args.seed), sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
