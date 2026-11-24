#!/usr/bin/env bash
# More Planner targets for the short-notice question bank: fresh general launches, drawn evenly over lateral
# position x height x speed x rising/falling at the strike plane (make_launches.py --general), flown through
# main's Runner + Planner in Supergate's batch simulator, which records the target the Planner commands for
# each ball (the "planned" rows of env-*-kinematics.csv). build_questions.py turns the runs into bank rows.
#
#   tools/supergate/grow_bank.sh launches <tag> <count> <seed> [lateral edges ...]   # host, repo .venv
#   tools/supergate/grow_bank.sh run <tag> [parts at a time=3] [threads per part=5]  # inside distrobox hope
#
# ``launches`` writes logs/supergate/general_launches_<tag>.json, splits it into parts of 6000 launches
# (six rallies of 1000) and freezes each into logs/supergate/general_suite_<tag>_part<k>.
# ``run`` plays every part that has no finished run yet into logs/supergate/general_run_<tag>_part<k>
# (about 50 launches per thread and minute) and touches logs/supergate/general_run_<tag>.done.
# The lateral bands are widened to the step strokes' zones and the left hand carries the gripper, as for g3.
# Default lateral edges: the stand strokes' window (-0.70 .. +0.46 m from HOME in seven cells); the step
# zones are "-0.92 -0.84 -0.76 -0.66" (right) and "0.42 0.50 0.58 0.66" (left).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$REPO"
O=hope_training/supermimic_hope/logs/supergate
T=$O/tree
SG=$T/a3_deploy/a3_deploy_example/scripts/supergate
BUNDLE=${GROW_BANK_BUNDLE:-hope_training/supermimic_hope/logs/supermimic/2026-11-20_07-15-15_ft25_general_g12/gate3/bundle_74000_plane058}
PART=6000
mode=$1; tag=$2; shift 2
case $mode in
launches)
  count=$1; seed=$2; shift 2
  lateral=()
  [ $# -gt 0 ] && lateral=(--lateral "$@")
  .venv/bin/python hope_training/supermimic_hope/tools/supergate/make_launches.py --tree $T --general --plane 0.0 --count "$count" \
      --seed "$seed" --after-plane-s 0.75 --pause-s 0.0 "${lateral[@]}" --out $O/general_launches_$tag.json | cut -c1-260
  parts=$(.venv/bin/python - "$O/general_launches_$tag" $PART <<'EOF'
import json, sys
stem, part = sys.argv[1], int(sys.argv[2])
rows = json.load(open(stem + ".json"))
n = len(rows) // part  # whole parts only: a rally is 1000 launches
for k in range(n):
    json.dump(rows[k * part:(k + 1) * part], open(f"{stem}_part{k}.json", "w"))
print(n)
EOF
)
  for k in $(seq 0 $((parts - 1))); do
    .venv/bin/python $SG/freeze_dataset.py --samples $O/general_launches_${tag}_part$k.json --seed $((2026112100 + seed * 100 + k)) \
        --output $O/general_suite_${tag}_part$k | tail -1 | cut -c1-150
  done
  echo "$tag: $parts parts of $PART launches"
  ;;
run)
  at_a_time=${1:-3}; threads=${2:-5}
  running=0
  for suite in $(ls -d $O/general_suite_${tag}_part* | sort -V); do
    k=${suite##*_part}
    out=$O/general_run_${tag}_part$k
    [ -f $out.done ] && continue
    ( SUPERGATE_FH_BAND=-0.95,-0.34 SUPERGATE_BH_BAND=-0.34,0.68 SUPERGATE_HOME_X=-0.58 SUPERGATE_LEFT_HAND=0.847,0.1128 \
        .venv/bin/python $SG/run.py --library $T/build/native_batch.so \
        --model-xml a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
        --ready-state $O/ready_stand.npz --policy grow=$BUNDLE --dataset $suite --output $out --threads "$threads" > $out.log 2>&1 \
        && touch $out.done ) &
    running=$((running + 1))
    if [ $running -ge "$at_a_time" ]; then wait -n || true; running=$((running - 1)); fi
  done
  wait
  touch $O/general_run_$tag.done
  ;;
*) echo "launches or run" >&2; exit 2 ;;
esac
