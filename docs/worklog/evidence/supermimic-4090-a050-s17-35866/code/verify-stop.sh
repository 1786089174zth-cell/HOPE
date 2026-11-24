#!/usr/bin/env bash
set -euo pipefail
run_root=$(cd -- "$(dirname -- "$0")" && pwd)
run_id=$1
[[ "$run_id" =~ ^[[:alnum:]_-]+$ ]]
run_dir="$run_root/$run_id"
python="${run_root%/*}/.supermimic-venv/bin/python"
state=$(systemctl --user show "$run_id" -p SubState --value)
if [[ "$state" != exited ]]; then
    systemctl --user show "$run_id" -p MainPID -p SubState -p Result -p ExecMainStatus
    exit 1
fi
systemctl --user show "$run_id" -p SubState -p Result -p ExecMainCode -p ExecMainStatus |
    tee "$run_dir/exit-status.txt"
[[ $(systemctl --user show "$run_id" -p Result --value) == success ]]
[[ $(systemctl --user show "$run_id" -p ExecMainStatus --value) == 0 ]]
iteration=$("$python" - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
receipt = json.loads((run / "stop_receipt.json").read_text())
iteration = receipt["iteration"]
assert receipt["run_id"] == run.name
assert Path(receipt["checkpoint"]) == run / f"final_{iteration}.pt"
print(iteration)
PY
)
"$python" "$run_root/checkpoint.py" "$run_dir/final_$iteration.pt" "$run_id" "$iteration" |
    tee "$run_dir/final_$iteration.proof.json"
