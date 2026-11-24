"""What Supergate asks of the policy, against what the training bank asks (numpy only).

    python3 tools/supergate/targets.py <supergate output dir> [--policy <label>] [--assets <clip assets dir>]

Reads the ``planned`` rows of ``env-*-kinematics.csv`` (the racket target the offline Planner committed:
position, velocity, face normal, in world axes) and the notice from ``env-*-results.csv`` for one policy
group, moves the target into station axes (HOME (-0.5, -0.7625)), and prints its quantiles next to the
qualified rows of ``physical_tuple_bank.npz`` (the questions training draws from), per side.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HOME = np.array([-0.5, -0.7625, 0.0])
Q = (0.01, 0.10, 0.50, 0.90, 0.99)


def supergate(out: Path, label: str | None):
    run = json.loads((out / "run.json").read_text())
    labels = [g["label"] for g in run["policy_groups"]]
    g = labels.index(label) if label else 0
    rows = []
    for shard in range(6):
        env = 6 * g + shard
        res = {int(r["shot"]): r for r in csv.DictReader((out / f"env-{env}-results.csv").open())}
        for k in csv.DictReader((out / f"env-{env}-kinematics.csv").open()):
            if k["event"] != "planned":
                continue
            r = res[int(k["shot"])]
            p = np.array([float(k[f"target_p_{a}"]) for a in "xyz"]) - HOME
            v = np.array([float(k[f"target_v_{a}"]) for a in "xyz"])
            n = np.array([float(k[f"target_n_{a}"]) for a in "xyz"])
            vin = np.array([float(r[f"strike_v{a}"]) for a in "xyz"])
            rows.append(np.concatenate((p, v, n, vin, [float(r["lead"]), float(r["side"]), float(r["legal"]), float(r["contacts"]) > 0])))
    return labels[g], np.array(rows)


def bank(assets: Path):
    b, a = np.load(assets / "physical_tuple_bank.npz"), np.load(assets / "reference_adapter.npz")
    keep = np.flatnonzero(a["qualified"])
    fam = a["family"][keep]
    return dict(p=b["target_position_offset"][keep], v=b["target_velocity"][keep], vin=b["incoming_velocity"][keep],
                spin=b["incoming_spin"][keep], lead=b["time_to_strike"][keep], family=fam, keys=list(b.keys()))


def q(x):
    return " ".join(f"{v:+6.2f}" for v in np.quantile(x, Q)) if len(x) else "   (none)"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opt = {}
    for name in ("--policy", "--assets"):
        if name in argv:
            i = argv.index(name)
            opt[name] = argv[i + 1]
            del argv[i:i + 2]
    here = Path(__file__).resolve().parents[2]
    assets = Path(opt.get("--assets", here.parent / "whole_body_tracking/assets/wide_stroke_v26_urdf2_plane058_20261110"))
    if not assets.is_dir():
        import glob
        assets = Path(glob.glob(str(here.parent / "**/wide_stroke_v26_urdf2_plane058_20261110"), recursive=True)[0])
    label, s = supergate(Path(argv[0]), opt.get("--policy"))
    b = bank(assets)
    print(f"Supergate targets as committed for {label} ({len(s)} planned shots); bank {assets.name} ({len(b['lead'])} qualified rows)")
    print(f"quantiles {Q}; station axes (x ahead of the station, y left, z above the floor)")
    for side, fams, name in ((1, (0, 2), "forehand"), (-1, (1, 3), "backhand")):
        m, bm = s[:, 13] == side, np.isin(b["family"], fams)
        elev = lambda v: np.degrees(np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1])))  # noqa: E731
        azim = lambda v: np.degrees(np.arctan2(v[:, 1], v[:, 0]))  # noqa: E731
        print(f"{name}: Supergate {m.sum()} shots, legal {int(s[m, 14].sum())}; bank {bm.sum()} rows")
        for title, x, y in (("target x", s[m, 0], b["p"][bm, 0]), ("target y", s[m, 1], b["p"][bm, 1]), ("target z", s[m, 2], b["p"][bm, 2]),
                            ("racket speed", np.linalg.norm(s[m, 3:6], axis=1), np.linalg.norm(b["v"][bm], axis=1)),
                            ("racket elevation deg", elev(s[m, 3:6]), elev(b["v"][bm])), ("racket azimuth deg", azim(s[m, 3:6]), azim(b["v"][bm])),
                            ("face-vs-velocity deg", np.degrees(np.arccos(np.clip((s[m, 3:6] * s[m, 6:9]).sum(1) / np.linalg.norm(s[m, 3:6], axis=1), -1, 1))), np.zeros(1)),
                            ("ball speed in", np.linalg.norm(s[m, 9:12], axis=1), np.linalg.norm(b["vin"][bm], axis=1)),
                            ("ball vz in", s[m, 11], b["vin"][bm, 2]), ("ball spin (bank only)", np.zeros(1), np.linalg.norm(b["spin"][bm], axis=1)),
                            ("notice s", s[m, 12], b["lead"][bm])):
            print(f"  {title:22s} Supergate {q(x)}   bank {q(y)}")
    np.save(Path(argv[0]) / f"targets_{label}.npy", s)


if __name__ == "__main__":
    main()
