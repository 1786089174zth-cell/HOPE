#!/usr/bin/env bash
# Build a Gate3 stack whose Runner and Planner are `origin/main` (81e606a0c), able to run our
# four-stroke / 10-frame bundles, without touching the working tree or the installed stack.
#
#   1. export main's sources with `git archive` (Runner package, ROS workspace, mocap packages)
#   2. apply runner_main81e_frame_history.patch + runner_main81e_four_stroke.patch (Runner) and
#      planner_main81e_commit_gate.patch (packetizer: optional commit gate ahead of the net, off by default)
#   3. build the whole Runner package out of tree with main's own scripts/build_a3_deploy_pkg.sh
#      (main changed several Runner translation units, so the one-file relink of build_runner.sh
#      against the 7990f18c2 objects is not valid here)
#   4. build main's ROS workspace (hope_msgs, hope_bringup, hope_planner_cpp, motion_capture_tracking)
#   5. write the workspace view the harness is pointed at (--hope-ws): main's install, main's
#      configs, and a model21800_hardware.yaml = main's file + main's return_physics.yaml +
#      main81e_planner_four_stroke.yaml (run_gate3.py has no option for a second planner file)
#
# Everything lands under OUT (default hope_training/supermimic_hope/logs/main_stack, gitignored):
#   OUT/src                                       exported (and patched) sources
#   OUT/build/a3_pkg_x86_64                       Runner build tree
#   OUT/dist/a3_deploy_x86_64_main81e_four_stroke Runner dist            -> --runner-dist
#   OUT/ros/{build,install,log}                   colcon workspace
#   OUT/hope_ws                                   workspace view         -> --hope-ws
#
# Run inside the `hope` container:  tools/gate3/build_main_stack.sh [OUT]
#   MAIN_STACK_JOBS=8            parallel compile jobs (Runner build; colcon uses 2 workers x JOBS/2)
#   MAIN_STACK_POLICY_DIR=<dir>  bundle to load with the new Runner and to compare with the planner bands
# A rerun re-exports the sources (git archive keeps commit mtimes, so only the patched translation
# unit recompiles). The first Runner configure downloads AimRT 1.6.0 and its dependencies from GitHub.
# The vendor SDKs (onnxruntime, unitree_sdk2) are linked from the working tree's untracked
# a3_deploy_example/thirdparty, which `source setup_a3_env.sh` downloaded when the machine was set up.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../../../.." && pwd)"
BASE="${MAIN_STACK_COMMIT:-81e606a0cb0d2f13b8649e221488e31c1d8bfc0c}"
OUT="${1:-$REPO/hope_training/supermimic_hope/logs/main_stack}"
JOBS="${MAIN_STACK_JOBS:-8}"
[ -r /opt/ros/jazzy/setup.bash ] || { echo "run inside the hope container (ROS Jazzy missing)" >&2; exit 2; }
mkdir -p "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
SRC="$OUT/src"
GEAR="$SRC/a3_deploy/a3_deploy_example"
DIST="$OUT/dist/a3_deploy_x86_64_main81e_four_stroke"
VENDOR="$REPO/a3_deploy/a3_deploy_example/thirdparty"
for dep in onnxruntime/onnxruntime-linux-x64-1.19.2/lib/libonnxruntime.so unitree_sdk2/CMakeLists.txt; do
  [ -e "$VENDOR/$dep" ] || { echo "vendor dependency missing: $VENDOR/$dep (source a3_deploy/a3_deploy_example/setup_a3_env.sh once)" >&2; exit 2; }
done

echo "[main-stack] 1/5 export $BASE -> $SRC"
mkdir -p "$SRC"
rm -rf "$SRC/a3_deploy" "$SRC/hope_ws" "$SRC/NatNet2ROS2" "$SRC/agibot"
git -C "$REPO" archive "$BASE" \
  a3_deploy/a3_deploy_example hope_ws/src NatNet2ROS2/src \
  agibot/mocap_sticker_shell_v3/documents/marker_transforms_pelvis_link_stickers.csv \
  agibot/mocap_sticker_shell_v3/motive_asset/P1.motive \
  agibot/mocap_sticker_shell_v3/motive_asset/P2.motive | tar -x -C "$SRC"
ln -s "$VENDOR/onnxruntime" "$GEAR/thirdparty/onnxruntime"
ln -s "$VENDOR/unitree_sdk2" "$GEAR/thirdparty/unitree_sdk2"

echo "[main-stack] 2/5 Runner patches"
for name in runner_main81e_frame_history.patch runner_main81e_four_stroke.patch; do
  patch -d "$SRC" -p1 -s --fuzz=0 --no-backup-if-mismatch < "$HERE/$name"
done
# Planner: optional commit gate ahead of the net (packetizer parameter commit_gate_x; default = unchanged)
patch -d "$SRC" -p1 -s --fuzz=0 --no-backup-if-mismatch < "$HERE/planner_main81e_commit_gate.patch"

echo "[main-stack] 3/5 Runner package -> $DIST"
set +u; source /opt/ros/jazzy/setup.bash; set -u
(cd "$GEAR" && A3_PACKAGE_BUILD_DIR="$OUT/build/a3_pkg_x86_64" A3_PACKAGE_OUTPUT_DIR="$DIST" \
  bash scripts/build_a3_deploy_pkg.sh --arch x86_64 \
    --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.hitter_pingpong.yaml \
    --policy-dir "$GEAR/models/model_21800/policy" --jobs "$JOBS")
LD_LIBRARY_PATH="$DIST:/opt/ros/jazzy/lib" "$DIST/a3_deploy_onnx_ref_pingpong" \
  --inspect-policy --policy-dir "$DIST/policy" >/dev/null

echo "[main-stack] 4/5 ROS workspace -> $OUT/ros/install"
mkdir -p "$OUT/ros"
workers=2
(cd "$OUT/ros" && MAKEFLAGS="-j$(( JOBS / workers > 0 ? JOBS / workers : 1 ))" \
  colcon --log-base "$OUT/ros/log" build \
    --base-paths "$SRC/hope_ws/src" "$SRC/NatNet2ROS2/src" \
    --build-base "$OUT/ros/build" --install-base "$OUT/ros/install" \
    --packages-up-to hope_msgs hope_bringup hope_planner_cpp motion_capture_tracking \
    --parallel-workers "$workers" --event-handlers console_cohesion+ \
    --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3)
for exe in hope_planner_cpp_node hope_ball_flight_packetizer; do
  [ -x "$OUT/ros/install/hope_planner_cpp/lib/hope_planner_cpp/$exe" ] || { echo "missing $exe" >&2; exit 1; }
done

echo "[main-stack] 5/5 harness workspace view -> $OUT/hope_ws"
VIEW="$OUT/hope_ws"
CONFIG="$SRC/hope_ws/src/hope_planner_cpp/config"
rm -rf "$VIEW"
mkdir -p "$VIEW/src/hope_planner_cpp/config"
ln -s ../ros/install "$VIEW/install"
ln -s "$SRC/hope_ws/src/hope_bringup" "$VIEW/src/hope_bringup"
for path in "$CONFIG"/*; do
  [ "$(basename "$path")" = model21800_hardware.yaml ] || ln -s "$path" "$VIEW/src/hope_planner_cpp/config/"
done
/usr/bin/python3 - "$CONFIG/model21800_hardware.yaml" "$CONFIG/return_physics.yaml" \
    "$HERE/main81e_planner_four_stroke.yaml" "$VIEW/src/hope_planner_cpp/config/model21800_hardware.yaml" "$BASE" <<'PY'
import hashlib, sys
from pathlib import Path
import yaml

*sources, target, base = sys.argv[1:]
merged, lines = {}, []
for index, source in enumerate(sources):
    data = Path(source).read_bytes()
    parameters = yaml.safe_load(data)["hope_planner"]["ros__parameters"]
    changed = sorted(key for key in parameters if key in merged and merged[key] != parameters[key])
    merged.update(parameters)
    lines.append(f"#   {index + 1}. {source}\n#      sha256 {hashlib.sha256(data).hexdigest()}"
                 + (f"\n#      overrides: {', '.join(changed)}" if changed else ""))
header = ("# GENERATED by tools/gate3/build_main_stack.sh; do not edit. Planner parameters for Gate3 on\n"
          f"# main {base[:9]}: later files override earlier ones.\n" + "\n".join(lines) + "\n")
Path(target).write_text(header + yaml.safe_dump({"hope_planner": {"ros__parameters": merged}},
                                                default_flow_style=None, sort_keys=False))
PY

if [ -n "${MAIN_STACK_POLICY_DIR:-}" ]; then
  echo "[main-stack] bundle check: $MAIN_STACK_POLICY_DIR"
  LD_LIBRARY_PATH="$DIST:/opt/ros/jazzy/lib" "$DIST/a3_deploy_onnx_ref_pingpong" \
    --inspect-policy --policy-dir "$MAIN_STACK_POLICY_DIR" > "$OUT/policy_inspection.yaml"
  /usr/bin/python3 - "$OUT/policy_inspection.yaml" "$VIEW/src/hope_planner_cpp/config/model21800_hardware.yaml" <<'PY'
import sys
import yaml

inspection = yaml.safe_load(open(sys.argv[1]))
planner = yaml.safe_load(open(sys.argv[2]))["hope_planner"]["ros__parameters"]
forehand, backhand = inspection["position_support"]
for key, box in (("swing_side_fh_y_range", forehand), ("swing_side_bh_y_range", backhand)):
    if any(abs(a - b) > 1e-9 for a, b in zip(planner[key], box[2:4])):
        raise SystemExit(f"{key} {planner[key]} != bundle support {box[2:4]}: edit main81e_planner_four_stroke.yaml")
print(f"bundle loads ({inspection['policy_abi']}, {inspection['station_contract']}); planner bands match its support")
PY
fi
echo "Runner dist : $DIST"
echo "hope-ws view: $VIEW"
