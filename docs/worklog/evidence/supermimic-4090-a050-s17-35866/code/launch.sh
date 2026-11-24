#!/usr/bin/env bash
set -euo pipefail
run_root=$(cd -- "$(dirname -- "$0")" && pwd)
workspace=$(dirname -- "$run_root")
run_id=$1
shift
[[ "$run_id" =~ ^[[:alnum:]_-]+$ ]]
run_dir="$run_root/$run_id"
mkdir -- "$run_dir"
source "$run_root/recipe.sh"
entry="$workspace/HOPE/hope_training/supermimic_hope"
python="$workspace/.supermimic-venv/bin/python"
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'cd %q\n' "$entry"
    printf 'export CUDA_VISIBLE_DEVICES=0\n'
    printf 'exec %q ' "$python"
    printf '%q ' -u train.py "${supermimic_args[@]}" "$@" "run_name=$run_id" "out=$run_dir"
    printf '\n'
} > "$run_dir/command.sh"
systemd-run --user --unit="$run_id" --service-type=exec \
    --property=RemainAfterExit=yes --property=StandardInput=null \
    --property="StandardOutput=append:$run_dir/train.log" \
    --property="StandardError=append:$run_dir/train.log" \
    /bin/bash "$run_dir/command.sh"
systemctl --user show "$run_id" -p MainPID -p ActiveState -p SubState
