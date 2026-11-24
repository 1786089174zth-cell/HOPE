#!/usr/bin/env bash
# Build the ping-pong Runner that Gate3 needs for four-stroke and multi-frame policies:
# `main`'s Runner (Gate3 entry interface: --inspect-policy, --shared-session-home) plus
#
#   runner_main_frame_history.patch   obs_frame_history: [1, 110*N] input, frame history in Run()
#                                     (port of supersuper 82d5a6fee)
#   runner_main_four_stroke.patch     four-stroke runtime (port of supersuper ebd54238c + aebcae84a):
#                                     natural recovery (old_ball_tail_until_arrival_v1), qualified
#                                     admission envelope (hitter_pure_pos_support_per_clip), and the
#                                     stand_backhand_first_frame_v15 HOLD contract
#
# The Runner installed on this workstation was built from `main` 7990f18c2. This script takes that
# exact source from git, applies the two patches, recompiles the one translation unit that includes
# the patched headers and relinks it against the objects of the existing package build
# (a3_deploy_example/build/a3_pkg_x86_64). The installed dist/a3_deploy_x86_64 is copied, never modified.
#
# Run inside the `hope` container:  tools/gate3/build_runner.sh [OUT_DIST]
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../../../.." && pwd)"
GEAR="$REPO/a3_deploy/a3_deploy_example"
OUT_DIST="${1:-$GEAR/dist/a3_deploy_x86_64_four_stroke}"
BASE_COMMIT="${RUNNER_BASE_COMMIT:-7990f18c2}"
TARGET_DIR="$GEAR/build/a3_pkg_x86_64/src/a3/a3_deploy_onnx_ref"
CM="$TARGET_DIR/CMakeFiles/a3_deploy_onnx_ref_pingpong.dir"
[ -f "$CM/flags.make" ] && [ -f "$CM/link.txt" ] || { echo "missing package build: $CM" >&2; exit 2; }
[ ! -e "$OUT_DIST" ] || { echo "refusing to overwrite $OUT_DIST" >&2; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/pp-runner-four-stroke.XXXXXX")"
git -C "$REPO" archive "$BASE_COMMIT" a3_deploy/a3_deploy_example/src | tar -x -C "$WORK"
patch -d "$WORK" -p1 -s < "$HERE/runner_main_frame_history.patch"
patch -d "$WORK" -p1 -s < "$HERE/runner_main_four_stroke.patch"

OBJ="$WORK/a3_pingpong_main.cpp.o"
defines="$(sed -n 's/^CXX_DEFINES = //p' "$CM/flags.make")"
includes="$(sed -n 's/^CXX_INCLUDES = //p' "$CM/flags.make" | sed "s#-I$GEAR/src#-I$WORK/a3_deploy/a3_deploy_example/src#g")"
flags="$(sed -n 's/^CXX_FLAGS = //p' "$CM/flags.make")"
eval "/usr/bin/c++ $defines $includes $flags -o '$OBJ' -c '$WORK/a3_deploy/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/a3_deploy/a3_pingpong_main.cpp'"
cp -a "$GEAR/dist/a3_deploy_x86_64" "$OUT_DIST"
link="$(sed "s#CMakeFiles/a3_deploy_onnx_ref_pingpong.dir/src/a3_deploy/a3_pingpong_main.cpp.o#$OBJ#; s#-o $GEAR/dist/a3_deploy_x86_64/a3_deploy_onnx_ref_pingpong#-o $OUT_DIST/a3_deploy_onnx_ref_pingpong#" "$CM/link.txt")"
(cd "$TARGET_DIR" && eval "$link")
echo "Runner: $OUT_DIST/a3_deploy_onnx_ref_pingpong (base $BASE_COMMIT + frame history + four-stroke runtime)"
echo "patched source left in $WORK"
