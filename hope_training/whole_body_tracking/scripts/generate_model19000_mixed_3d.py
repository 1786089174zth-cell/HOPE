"""Generate a deterministic mixed-height, mixed-angle, center-inclusive safety scene."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


NOMINAL = [
    [2.4, y, 0.49, -3.0, 0.0, 2.2]
    for y in (-1.0225, -0.7125, -1.0575, -0.6575, -1.0925, -0.6125)
]

# x, y, z, vx, vy, vz in the planner table frame. The center templates were
# checked with PPQ004 ball physics at the x=0.08 m table-frame strike plane.
# All 19 generated center shots crossed within 0.027 m of world y=0.
TEMPLATES = {
    "center": [
        [2.4, -0.7625, 0.49, -3.0, 0.0, 2.2],
        [2.4, -0.995, 0.45, -3.0, 0.3, 2.4],
        [2.4, -0.53, 0.60, -3.0, -0.3, 2.0],
        [2.4, -1.16, 0.75, -3.5, 0.6, 2.0],
        [2.4, -0.43, 0.49, -3.5, -0.5, 2.2],
    ],
    "left": [
        [2.4, -1.28, 0.42, -3.0, -0.25, 2.4],
        [2.4, -1.45, 0.60, -3.5, 0.0, 2.0],
        [2.4, -1.10, 0.75, -4.0, -0.6, 2.0],
        [2.4, -1.35, 0.49, -3.2, 0.2, 2.8],
        [2.4, -1.50, 0.49, -3.8, 0.5, 2.8],
    ],
    "right": [
        [2.4, -0.25, 0.42, -3.0, 0.25, 2.4],
        [2.4, -0.15, 0.60, -3.5, 0.0, 2.0],
        [2.4, -0.45, 0.75, -4.0, 0.6, 2.0],
        [2.4, -0.20, 0.49, -3.2, -0.2, 2.8],
        [2.4, -0.10, 0.34, -3.8, -0.5, 2.8],
    ],
    "low": [
        [2.4, -0.7625, 0.34, -3.2, 0.0, 2.4],
        [2.4, -0.95, 0.40, -3.5, 0.2, 1.8],
        [2.4, -0.60, 0.45, -3.8, -0.3, 2.0],
        [2.4, -1.10, 0.34, -4.0, 0.4, 2.4],
        [2.4, -0.40, 0.38, -3.0, -0.2, 2.8],
    ],
    "high": [
        [2.4, -0.7625, 0.75, -3.0, 0.0, 2.4],
        [2.4, -1.00, 0.80, -3.5, 0.25, 2.2],
        [2.4, -0.50, 0.70, -3.8, -0.25, 2.6],
        [2.4, -1.20, 0.65, -4.0, 0.4, 2.8],
        [2.4, -0.30, 0.80, -3.2, -0.4, 2.0],
    ],
    "oblique": [
        [2.4, -1.30, 0.49, -3.5, 0.9, 2.2],
        [2.4, -0.25, 0.60, -3.5, -0.9, 2.2],
        [2.4, -1.45, 0.70, -4.2, 1.2, 2.0],
        [2.4, -0.10, 0.45, -4.2, -1.2, 2.6],
        [2.4, -1.25, 0.34, -3.8, 1.5, 2.8],
    ],
}


def generate(balls, seed):
    if balls < 26 or balls % 2:
        raise ValueError("balls must be even and at least 26")
    rng = np.random.default_rng(seed)
    categories = tuple(TEMPLATES)
    serves = []
    labels = []
    orders = [categories if round_number == 0 else tuple(rng.permutation(categories))
              for round_number in range((balls - 6 + len(categories) - 1) // len(categories))]
    for index in range(balls - 6):
        round_number, within_round = divmod(index, len(categories))
        category = orders[round_number][within_round]
        template = TEMPLATES[category][round_number % len(TEMPLATES[category])]
        row = list(template)
        if round_number:
            # Small deterministic variations, preserving the verified center aim.
            row[1] += float(rng.uniform(-0.012 if category == "center" else -0.03,
                                        0.012 if category == "center" else 0.03))
            row[2] += float(rng.uniform(-0.015, 0.015))
            row[3] += float(rng.uniform(-0.08, 0.08))
            row[4] += float(rng.uniform(-0.015 if category == "center" else -0.08,
                                        0.015 if category == "center" else 0.08))
            row[5] += float(rng.uniform(-0.05, 0.05))
        serves.append([round(v, 6) for v in row])
        labels.append(category)
    scene = {
        "schema_version": 1,
        "family": "fixed_home",
        "description": f"model19000 mixed 3D safety probe, seed {seed}; center, left, right, low, high and oblique interleaved; misses allowed, falls fail",
        "balls": balls,
        "clean_serves": NOMINAL,
        "rapid_cycle": serves,
        "timing": {
            "clean_flight_s": 2.5, "rapid_flight_s": 1.15,
            "final_flight_s": 2.5, "clean_pause_s": 1.0,
            "rapid_pause_s": 0.0,
        },
        "metrics": {},
    }
    metadata = [
        {"shot_id":i + 1, "category": "nominal" if i < 6 else labels[i - 6],
         "x":row[0], "y":row[1], "z":row[2],
         "vx":row[3], "vy":row[4], "vz":row[5]}
        for i,row in enumerate(NOMINAL + serves)
    ]
    return scene, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--balls", type=int, default=120)
    parser.add_argument("--seed", type=int, default=19000)
    parser.add_argument("--prefix-balls", type=int, help="Keep this prefix of the generated scene")
    args = parser.parse_args()
    scene, metadata = generate(args.balls, args.seed)
    if args.prefix_balls is not None:
        if not 6 < args.prefix_balls <= args.balls:
            raise ValueError("prefix-balls must be between 7 and balls")
        scene["rapid_cycle"] = scene["rapid_cycle"][:args.prefix_balls - 6]
        scene["balls"] = args.prefix_balls
        scene["description"] += f"; first {args.prefix_balls} shots of {args.balls}-shot source"
        metadata = metadata[:args.prefix_balls]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"model19000-mixed-3d-{scene['balls']}"
    (args.output_dir / f"{stem}.yaml").write_text(yaml.safe_dump(scene, sort_keys=False))
    with (args.output_dir / f"{stem}-shots.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    print(json.dumps({"scene":str(args.output_dir / f"{stem}.yaml"),
                      "shots":str(args.output_dir / f"{stem}-shots.csv"),
                      "balls":scene["balls"], "seed":args.seed}))


if __name__ == "__main__":
    main()
