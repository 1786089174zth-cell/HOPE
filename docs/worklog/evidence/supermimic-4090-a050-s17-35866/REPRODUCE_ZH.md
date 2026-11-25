# Reproduce and resume SuperMimic

## Get the code and checkpoint

`f3b4e80` was the original training base. Commit `7d414f4` archived the model and added Trainer stop handling. The commands below check out `fa90d5a`, which also includes this guide and the one-update resume receipt. Then continue at **Prepare the environment** in the same Bash session. The first section of the pinned copy still shows the earlier checkout command.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch experiment/supermimic-reward-split \
  https://github.com/1786089174zth-cell/HOPE.git
cd HOPE
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach fa90d5aff6f586837975ef58ae48e94b7bba9203
git lfs install --local
ARCHIVE=docs/worklog/evidence/supermimic-4090-a050-s17-35866
BANK=hope_training/whole_body_tracking/assets/wide_stroke_v26_urdf2_plane058_20261110/physical_tuple_bank.npz
git lfs pull --include="$ARCHIVE/checkpoint/final_35866.pt,$BANK" --exclude=""
printf '%s  %s\n' a34af2d6d80264b6c0267e4f108a2747415e26f317dfdfc2d2dbb8d9ef104364 \
  "$ARCHIVE/checkpoint/final_35866.pt" | sha256sum -c -
```

## Prepare the environment

Verified on Linux with an RTX 4090, NVIDIA driver 580.178.04, Python 3.10, and CUDA PyTorch 2.11.0+cu128. Install Git LFS and use an NVIDIA driver compatible with CUDA 12.8. Set `VENV` before running the block, or set `PY` to an existing compatible interpreter and skip installation.

```bash
VENV=${VENV:-"$PWD/.venv-supermimic"}
python3.10 -m venv "$VENV"
export PY="$VENV/bin/python"
"$PY" -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r hope_training/supermimic_hope/requirements.txt tensorboard==2.21.0
"$PY" -m pip check
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
"$PY" -c 'import numpy as n,sys; a=n.load(sys.argv[1],allow_pickle=False); print(a["target_position_offset"].shape)' "$BANK"
```

The requirements file pins MuJoCo, MuJoCo-Warp, Warp and the package dependencies. TensorBoard is optional for training. Install `onnx` separately for export and `matplotlib` for the archived plot script. Training uses the standalone entry point and needs no Isaac installation.

## Resume and verify one update

This launcher reads the [saved configuration](training/config.json), including seed 17, 6,144 environments, the dual-body MLP, shared critic, α = 0.5, ten frames and 32 imagined steps. Set `PY` and `RUNS` before defining it to use a different interpreter or output parent. Each RunId must have a fresh output directory.

```bash
ROOT=$(pwd -P)
ENTRY="$ROOT/hope_training/supermimic_hope"
ARCHIVE="$ROOT/docs/worklog/evidence/supermimic-4090-a050-s17-35866"
CKPT="$ARCHIVE/checkpoint/final_35866.pt"
PY=${PY:-"$ROOT/.venv-supermimic/bin/python"}
RUNS=${RUNS:-"$ROOT/.supermimic-runs"}

run_supermimic() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  HOPE_SUPERMIMIC_DEPS="${PY%/*}/../isolated-extra-deps" \
  "$PY" - "$ENTRY" "$ARCHIVE/training/config.json" "$RUNS/$1" "$@" <<'PYTHON'
import json, os, sys
from pathlib import Path

entry, config, output, run_id, limit, log_interval, checkpoint = sys.argv[1:]
if not run_id or not all(c.isalnum() or c in "-_" for c in run_id):
    raise ValueError("Use a RunId containing only letters, digits, dashes or underscores")
output = Path(output).resolve()
checkpoint = Path(checkpoint).resolve(strict=True)
cfg = json.loads(Path(config).read_text())
cfg.update(run_name=run_id, out=str(output), resume=str(checkpoint),
           max_iterations=int(limit), log_interval=int(log_interval))

def overrides(settings, prefix=""):
    for key, value in settings.items():
        if isinstance(value, dict):
            yield from overrides(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}={json.dumps(value, separators=(',', ':'))}"

output.mkdir(parents=True, exist_ok=False)
os.chdir(entry)
os.execv(sys.executable, [sys.executable, "-u", "train.py", *overrides(cfg)])
PYTHON
}

RUN_ID=supermimic-doccheck-35866-to-35867-20261124-01
run_supermimic "$RUN_ID" 35867 1 "$CKPT"
"$PY" "$ARCHIVE/code/checkpoint.py" "$RUNS/$RUN_ID/model_35867.pt" "$RUN_ID" 35867
```

`max_iterations` is the absolute end iteration. Starting at 35,866 with a limit of 35,867 executes one dynamics, actor and critic update and saves `model_35867.pt`. Logging every iteration exposes the update in `metrics.jsonl`; the logged row is numbered 35,866 because logging uses the pre-update index.

On the RTX 4090, the command above loaded 35,866, wrote `model_35867.pt` and exited **0** at the cap. An independent CPU process loaded the new file and verified finite actor, EMA, critic, dynamics and all three optimizer states. All four networks had changed parameters. The actor, critic and dynamics optimizer step counts advanced by 1, 20 and 10 respectively. The upper and lower gradient norms were 2.784 and 5.477, with critic loss 0.08159. The single update ran before CUDA graph capture warmed up. [Acceptance receipt](checkpoint/resume_35866_to_35867.json)

The new file remains in the local validation output. Its SHA-256 is `1c4140c0573d5c58be178c617f09324a5ea99aea6e5520c02b9ac0a2c9fdbc61`. The source checkpoint retained its original hash.

## Continue training or stop by RunId

From the same shell, choose a fresh RunId and an absolute cap above the iteration of your chosen checkpoint. The example uses the verified one-step result and restores the archived logging interval of 25.

```bash
NEXT_ID="supermimic-resume-$(date -u +%Y%m%dT%H%M%SZ)"
run_supermimic "$NEXT_ID" 2147483647 25 "$RUNS/$RUN_ID/model_35867.pt"
```

From another Bash session at the repository root, use the same output parent and the exact RunId printed or chosen at launch.

```bash
RUNS=${RUNS:-"$PWD/.supermimic-runs"}
ACTIVE_RUN_ID='replace-with-the-running-id'
touch -- "$RUNS/$ACTIVE_RUN_ID/stop.$ACTIVE_RUN_ID.request"
```

A stop request is handled at a complete iteration boundary. It writes `final_<iteration>.pt` and `stop_receipt.json` before exiting. Reaching the absolute cap writes `model_<iteration>.pt` and exits normally. Resume restores actor, EMA, critic and dynamics states including normalizers, all three optimizer states, the iteration counter, and same-size task encoder offsets. The simulator starts a fresh rollout, the RNG is seeded again, and replay refills during warmup. The new collection counter starts at zero.

[100-ball video](gate3/fixed-home-100/gate3_100_balls_60fps.mp4) · [Training curves](figures/training-curves.png) · [Gate3 report](gate3/gate3-preempt/pp_rally_report.json)
