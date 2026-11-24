"""Audit the planned runs and reduce their held-out evaluations to paired measurements."""
import csv
import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from run_paired import PKG, ROOT

MODES = ("stand", "step")
FIELDS = ("hit_pct", "error_cm", "falls_per_1000")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def aggregate_run(run, plan, hashes):
    import torch

    ev = plan["evaluation"]
    folder = ROOT / run["name"]
    config = read(folder / "config.json")
    init = read(folder / "initial.json")
    assert config["seed"] == run["seed"] and config["learn"]["body_reward_alpha"] == run["alpha"]
    lines = [json.loads(s) for s in (folder / "metrics.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in lines] == list(range(0, config["max_iterations"], config["log_interval"]))
    for r in lines:
        assert math.isfinite(r["wm/loss"])
        if r["iteration"] >= config["wm_only_iterations"]:
            for k in ("upper_grad_norm", "lower_grad_norm", "imagined_return", "value_loss"):
                assert math.isfinite(r["policy/" + k]), (folder, r["iteration"], k)
    rows = []
    for iteration in ev["iterations"]:
        model = folder / f"model_{iteration}.pt"
        scores = folder / f"evaluation_{iteration}.json"
        hashes[str(model.relative_to(ROOT))] = sha(model)
        hashes[str(scores.relative_to(ROOT))] = sha(scores)
        ck = torch.load(str(model), map_location="cpu", mmap=True, weights_only=False)
        # The checkpoint keeps dataclass tuples, while config.json serializes them as arrays.
        assert json.loads(json.dumps(ck["config"])) == config
        assert ck["iteration"] == iteration and ck["actor_ema"]
        expected_steps = config["warmup_steps"] + int(iteration / config["utd"])
        assert ck["env_steps"] == expected_steps
        del ck
        d = read(scores)
        assert (Path(d["checkpoint"]).resolve(), d["iteration"], d["seed"], d["num_envs"], d["steps"],
                d["actor"], d["contact"], d["handoff"], d["action_noise"], d["midstroke_probability"]) == (
                model.resolve(), iteration, ev["seed"], ev["envs"], ev["steps"], ev["actor"],
                ev["contact"], ev["handoff"], ev["action_noise"], 0)
        def count(mode, key):
            value = d[f"motion/{mode}/{key}"]
            assert value == int(value) and value >= 0
            return int(value)
        ticks = sum(count(m, "ticks") for m in (*MODES, "hold"))
        falls = sum(count(m, "falls") for m in (*MODES, "hold"))
        assert ticks == ev["envs"] * ev["steps"] and falls == round(ticks * d["task/falls_per_step"])
        assert sum(count(m, "strikes") for m in MODES) == d["task/strikes"]
        assert sum(count(m, "hits") for m in MODES) == round(d["task/strikes"] * d["task/hit_rate"])
        for mode in (*MODES, "hold"):
            n, f = count(mode, "ticks"), count(mode, "falls")
            assert f <= n
            row = dict(seed=run["seed"], alpha=run["alpha"], iteration=iteration, mode=mode,
                       train_robot_ticks=expected_steps * config["num_envs"],
                       ticks=n, falls=f, falls_per_1000=1000 * f / n if n else None)
            if mode != "hold":
                strikes, hits = count(mode, "strikes"), count(mode, "hits")
                es = d[f"motion/{mode}/position_error_sum_m"]
                assert hits <= strikes and math.isfinite(es) and es >= 0
                row.update(strikes=strikes, hits=hits, hit_pct=100 * hits / strikes if strikes else None,
                           error_sum_m=es, error_cm=100 * es / strikes if strikes else None)
            rows.append(row)
    for f in ("initial.json", "config.json", "physics_receipt.json", "metrics.jsonl"):
        hashes[str((folder / f).relative_to(ROOT))] = sha(folder / f)
    return rows, config, init


def main():
    plan = read(ROOT / "protocol.json")
    ev = plan["evaluation"]
    hashes = {}
    rows, inits, configs = [], {}, {}
    for run in plan["runs"]:
        series, cfg, init = aggregate_run(run, plan, hashes)
        seed, alpha = run["seed"], run["alpha"]
        inits[(seed, alpha)] = init
        cfg["learn"].pop("body_reward_alpha")
        for k in ("run_name", "out"):
            cfg.pop(k)
        configs[(seed, alpha)] = cfg
        rows.extend(series)
    seeds = sorted({k[0] for k in configs})
    for seed in seeds:
        assert configs[(seed, 1)] == configs[(seed, .5)]
        assert inits[(seed, 1)] == inits[(seed, .5)], (seed, "initial pairing")
    assert len({inits[(s, 1)]["actor"] for s in seeds}) == len(seeds)
    common = [{k: v for k, v in cfg.items() if k != "seed"} for cfg in configs.values()]
    assert all(c == common[0] for c in common)
    fields = ["seed", "alpha", "iteration", "mode", "train_robot_ticks", "strikes", "hits", "hit_pct",
              "error_sum_m", "error_cm", "ticks", "falls", "falls_per_1000"]
    with (ROOT / "learning_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    trajectory = defaultdict(list)
    for row in rows:
        trajectory[(row["mode"], row["alpha"], row["seed"])].append(row)
    stats = []
    iters = ev["iterations"]
    for mode in MODES:
        for alpha in (1.0, .5):
            for seed in seeds:
                pts = trajectory[mode, alpha, seed]
                assert [r["iteration"] for r in pts] == iters
                qual = [p["strikes"] >= ev["minimum_strikes"] and
                        p["hits"] >= p["strikes"] * ev["hit_rate_threshold"] for p in pts]
                first = next((i for i in range(len(pts) - 1) if qual[i] and qual[i+1]), None)
                auc = sum((b["iteration"] - a["iteration"]) * (a["hit_pct"] + b["hit_pct"]) / 2
                          for a, b in zip(pts, pts[1:])) / (iters[-1] - iters[0])
                stats.append(dict(mode=mode, alpha=alpha, seed=seed, first=iters[first] if first is not None else None,
                                  confirmed=iters[first+1] if first is not None else None, mean_curve_hit_pct=auc,
                                  final={k:pts[-1][k] for k in fields if k in pts[-1]}))
    grouped = []
    for mode in MODES:
        for alpha in (1.0, .5):
            final = [r for r in stats if r["mode"] == mode and r["alpha"] == alpha]
            grouped.append(dict(mode=mode, alpha=alpha,
                                reached=sum(r["confirmed"] is not None for r in final),
                                final_mean={k:st.mean(r["final"][k] for r in final) for k in FIELDS},
                                final_range={k:[min(r["final"][k] for r in final), max(r["final"][k] for r in final)] for k in FIELDS}))
    paired = []
    for mode in MODES:
        for seed in seeds:
            ref, cand = [next(r for r in stats if (r["mode"], r["seed"], r["alpha"]) == (mode,seed,a))
                         for a in (1.0,.5)]
            diff = {k:cand["final"][k] - ref["final"][k] for k in FIELDS}
            paired.append(dict(mode=mode, seed=seed, final_difference_a050_minus_a100=diff,
                               curve_hit_pct_difference=cand["mean_curve_hit_pct"] - ref["mean_curve_hit_pct"],
                               confirmed_iteration_difference=(cand["confirmed"]-ref["confirmed"]
                               if cand["confirmed"] is not None and ref["confirmed"] is not None else None)))
    assets = PKG.parent / "whole_body_tracking" / common[0]["task"]["assets"]
    repo = PKG.parent.parent
    xml = repo / "a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
    inputs = {str(p.relative_to(repo)):sha(p) for p in [xml, assets/"manifest.json", *sorted(assets.glob("*.npz"))]}
    for f in ("protocol.json", "run_paired.py", "train_with_receipt.py", "evaluate_motion.py", "score_paired.py",
              "analyze.py", "learning_curves.csv"):
        hashes[f] = sha(ROOT / f)
    report = dict(run_id=ROOT.name, source=plan["source"], grouped=grouped, seeds=stats, paired=paired,
                  initial_sha256={s:inits[(s, 1)] for s in seeds}, inputs=inputs, artifacts_sha256=hashes)
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(grouped, indent=2))

if __name__ == "__main__":
    main()
