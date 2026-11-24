"""Preregister the six runs, then train them sequentially on a single GPU."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PKG = Path(r"C:/Users/ROG/Desktop/UCB/Capstone/HOPE-supermimic/hope_training/supermimic_hope")
SOURCE = "f3b4e809ab084d05b7019b811785c4aa9354086d"
SEEDS = (17, 29, 43)
COMMON = """actor_architecture=dual_body_mlp_v1 num_envs=256 device=cuda:0 max_iterations=3000
wm_only_iterations=150 warmup_steps=96 buffer_steps=512 learn.wm_batches=3 learn.wm_batch=256
learn.wm_window=8 learn.batch=256 learn.horizon=16 learn.critic_minibatches=4 learn.critic_epochs=2
capture=true log_interval=25 save_interval=500""".split()

def main():
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PKG, text=True).strip() != SOURCE:
        raise ValueError("Training source differs from preregistration")
    runs = []
    for i, seed in enumerate(SEEDS):
        for alpha in (("1.0", "0.5") if i % 2 == 0 else ("0.5", "1.0")):
            tag = "a100" if alpha == "1.0" else "a050"
            name = f"s{seed}_{tag}"
            cmd = [sys.executable, "-u", str(ROOT / "train_with_receipt.py"), *COMMON,
                   f"seed={seed}", f"learn.body_reward_alpha={alpha}", f"run_name={name}", f"out={ROOT/name}"]
            runs.append(dict(name=name, seed=seed, alpha=float(alpha), argv=cmd))
    plan = dict(source=SOURCE, branch="experiment/supermimic-reward-split", run_id=ROOT.name, runs=runs,
                evaluation=dict(iterations=list(range(500,3001,500)), seed=12345, envs=128, warmup=200,
                                steps=500, contact="elliptic", handoff=0, actor="ema", action_noise=0,
                                hit_rate_threshold=0.2, minimum_strikes=100, consecutive_evaluations=2,
                                time_to_learn="first of two consecutive qualifying checkpoints; also report confirmation",
                                strike_metrics="pooled FH and BH within seed and mode; strikes scored at frame 60",
                                falls="stand/step when pre-action stroke is active, hold/idle separate, per 1000 robot ticks",
                                aggregation="equal-weight train-seed means and paired differences; per-seed values retained"))
    with (ROOT / "protocol.json").open("x") as f:
        json.dump(plan, f, indent=2)
    for run in runs:
        now = datetime.datetime.now().astimezone().isoformat()
        print(f"{now} train {run['name']}", flush=True)
        with (ROOT / (run["name"] + ".train.log")).open("x") as f:
            subprocess.run(run["argv"], cwd=PKG, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("Training complete", flush=True)

if __name__ == "__main__":
    main()
