#!/usr/bin/env bash
# Assemble a Supergate checkout that can run four-stroke, 10-frame policies, and build it.
#
#   hope_training/supermimic_hope/tools/supergate/assemble.sh <new tree dir>      (hope container, repository root)
#
# Supergate (origin/codex/supergate-tools-20261113, a3_deploy_example/scripts/supergate) compiles the Runner's
# PpPolicy from "the current checkout". That branch's Runner is main 7990f18c2 plus a few fixes and rejects
# four-stroke bundles (HOLD contract) and has no frame history. This script copies, from that branch, the
# Supergate tools, the Runner source and the MuJoCo module into <tree> (same relative layout, so Supergate's
# REPOSITORY = <tree>), applies the two Runner patches the merged Gate3 runner uses
# (tools/gate3/runner_main_frame_history.patch, runner_main_four_stroke.patch), and builds native_batch.so with
# the repository venv's MuJoCo (batch API) and the bundled ONNX Runtime 1.19.2. Nothing in the working tree changes.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../../../.." && pwd)"
BRANCH="${SUPERGATE_BRANCH:-origin/codex/supergate-tools-20261113}"
TREE="${1:?usage: assemble.sh <new tree dir>}"
[ ! -e "$TREE" ] || { echo "refusing to overwrite $TREE" >&2; exit 2; }
mkdir -p "$TREE"
git -C "$REPO" archive "$BRANCH" a3_deploy/a3_deploy_example/scripts/supergate a3_deploy/a3_deploy_example/src/a3/a3_deploy_onnx_ref \
  a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/module a3_deploy/a3_deploy_example/scripts/validate_spin001_serve_mujoco.py \
  a3_deploy/a3_deploy_example/scripts/runner_sim_profile.py | tar -x -C "$TREE"
# The protocol's HOME is 0.50 m behind the table edge. SUPERGATE_HOME_X=-0.58 runs the same suites with the robot
# where our deploy geometry has it (strike plane on the table edge); the default is unchanged.
RUN="$TREE/a3_deploy/a3_deploy_example/scripts/supergate/run.py"
grep -q "q\[:2\] = \[-.5, -.7625\]" "$RUN" || { echo "run.py: HOME line not found" >&2; exit 3; }
sed -i -e "s/^import json$/import json\nimport os/" \
  -e "s/q\[:2\] = \[-.5, -.7625\]/q[:2] = [float(os.environ.get('SUPERGATE_HOME_X', '-.5')), -.7625]/" "$RUN"
# The four-stroke patch targets main: two of its hunks (target admission) do not apply here, because this branch
# rewrote that block for Supergate's lateral admission profile. admission.patch is the same change on this branch.
for p in runner_main_frame_history.patch runner_main_four_stroke.patch; do
  patch -d "$TREE" -p1 --no-backup-if-mismatch -r - < "$HERE/../gate3/$p" || true
done
patch -d "$TREE" -p1 --no-backup-if-mismatch < "$HERE/admission.patch"
python3 "$HERE/bands.py" "$TREE/a3_deploy/a3_deploy_example/scripts/supergate/native/native_batch.cpp"  # SUPERGATE_FH_BAND / _BH_BAND
python3 "$HERE/hand.py" "$RUN"  # SUPERGATE_LEFT_HAND: the gripper the deployed robot carries
python3 "$HERE/hit_scale.py" "$TREE/a3_deploy/a3_deploy_example/scripts/supergate/native/native_batch.cpp"  # SUPERGATE_HIT_SPEED_SCALE: the field Planner's operator scale
python3 "$HERE/actuators.py" "$RUN"  # SUPERGATE_ACTUATORS: actuators identified on the robot (tools/real/replay_joints.py --fit)
python3 "$HERE/commit_gate.py" "$TREE/a3_deploy/a3_deploy_example/scripts/supergate/native/native_batch.cpp"  # SUPERGATE_COMMIT_X / _POLICY_LEAD / _BALL_NOISE: commit before the net
PY="$REPO/.venv/bin/python"
ORT="$REPO/a3_deploy/a3_deploy_example/thirdparty/onnxruntime/onnxruntime-linux-x64-1.19.2"
"$PY" "$TREE/a3_deploy/a3_deploy_example/scripts/supergate/build.py" --ort-root "$ORT" --output "$TREE/build"
echo "Supergate: $TREE/a3_deploy/a3_deploy_example/scripts/supergate  library: $TREE/build/native_batch.so"
