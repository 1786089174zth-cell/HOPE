#!/usr/bin/env bash
# Score a SuperMimic checkpoint on a Supergate suite (host shell; enters the two containers itself).
#
#   tools/supergate/score.sh <model_N.pt> [suite=main] [threads=12] [home_x=-0.58] [label=dir ...]
#
# suite: main, outer, mixed-long, slow-tail, or a suite directory (e.g. the field ball paths of tools/field/make_suite.py).
#
# 1. grasping container: exports the Gate3 bundle <run>/gate3/bundle_N_plane058 (strike box centred 0.58 m ahead
#    of the station) unless it is there already;
# 2. hope container: runs Supergate (the tree made by assemble.sh, logs/supergate/tree) on the suite for the
#    checkpoint and every extra label=bundle given, into <run>/supergate/<suite>_N_home058 (<suite>_N at -0.5);
# 3. prints tools/supergate/report.py for that directory.
# home_x=-0.58 (default) puts the robot where our deploy geometry has it, strike plane on the table edge: the
# only geometry we test in since 2026-11-19. -0.5 is Supergate's own protocol.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd -- "$HERE/../.." && pwd)"
REPO="$(cd -- "$PKG/../.." && pwd)"
CKPT="$(realpath "${1:?usage: score.sh <model_N.pt> [suite] [threads] [home_x] [label=dir ...]}")"
SUITE="${2:-main}"; THREADS="${3:-12}"; HOME_X="${4:--0.58}"; shift $(( $# < 4 ? $# : 4 ))
RUN="$(dirname "$CKPT")"; N="$(basename "$CKPT" .pt)"; N="${N#model_}"
# SUPERGATE_SPLIT: the forehand | backhand boundary the checkpoint was trained with (task.side_bounds[1]; -0.275
# up to ft26, -0.34 from ft27). Other than -0.275 it goes into the bundle (bundle_N_plane058_splitXXX), into the
# bundle bands below and into the output directory's name.
SPLIT="${SUPERGATE_SPLIT:--0.275}"; SPLIT_TAG=""; SPLIT_ARG=""
if [ "$SPLIT" != "-0.275" ]; then SPLIT_TAG="_split${SPLIT#-0.}"; SPLIT_ARG="--side-split $SPLIT"; fi
# SUPERGATE_REACH=right,left: lateral limits of the contract other than the default -0.9002,0.4598 (e.g. -0.92,0.64
# for checkpoints trained with the short-notice step strokes); bundle, bands and output name follow it too.
REACH="${SUPERGATE_REACH:--0.9002,0.4598}"; FAR_R="${REACH%,*}"; FAR_L="${REACH#*,}"
if [ "$REACH" != "-0.9002,0.4598" ]; then SPLIT_TAG="${SPLIT_TAG}_reach${FAR_R#-0.}_${FAR_L#0.}"; SPLIT_ARG="$SPLIT_ARG --lateral-reach $FAR_R $FAR_L"; fi
# SUPERGATE_CLAMP=1: the bundle clamps the stroke command to the trained range (build_gate3_bundle.py --command-clamp);
# bundle and output directory get the suffix _clamp.
if [ "${SUPERGATE_CLAMP:-}" = "1" ]; then SPLIT_TAG="${SPLIT_TAG}_clamp"; SPLIT_ARG="$SPLIT_ARG --command-clamp"; fi
# SUPERGATE_WAIST_ROLL=0.29: the bundle clamps the waist-roll command to +- that (--waist-roll-limit); suffix _wr029.
if [ -n "${SUPERGATE_WAIST_ROLL:-}" ]; then SPLIT_TAG="${SPLIT_TAG}_wr${SUPERGATE_WAIST_ROLL/./}"; SPLIT_ARG="$SPLIT_ARG --waist-roll-limit $SUPERGATE_WAIST_ROLL"; fi
BUNDLE="$RUN/gate3/bundle_${N}_plane058$SPLIT_TAG"
TREE="$PKG/logs/supergate/tree"; SG="$TREE/a3_deploy/a3_deploy_example/scripts/supergate"
# a suite is one of Supergate's own (main, outer, mixed-long, slow-tail) or a directory in its format
if [ -d "$SUITE" ]; then DATA="$(realpath "$SUITE")"; SUITE="$(basename "$DATA")"; else DATA="$SG/data/$SUITE"; fi
OUT="$RUN/supergate/${SUITE}_${N}$([ "$HOME_X" = "-0.5" ] || echo "_home${HOME_X#-0.}")$SPLIT_TAG"
# SUPERGATE_BANDS=bundle: the lateral bands of our bundle's contract, as main's Planner would use them, instead of
# the harness's own (forehand [-0.68, -0.24], backhand [-0.24, 0.28]); the output directory gets the suffix _bands.
BANDS=""
if [ "${SUPERGATE_BANDS:-}" = "bundle" ]; then BANDS="SUPERGATE_FH_BAND=$FAR_R,$SPLIT SUPERGATE_BH_BAND=$SPLIT,$FAR_L"; OUT="${OUT}_bands"; fi
# The left hand: the gripper the deployed robot carries (0.847 kg, centre 0.1128 m along the link) unless
# SUPERGATE_HAND=bare (the Gate3 MJCF's 0.281 kg hand, what every run before 2026-11-20 used; no suffix).
if [ "${SUPERGATE_HAND:-gripper}" = "bare" ]; then HAND=""; else HAND="SUPERGATE_LEFT_HAND=0.847,0.1128"; OUT="${OUT}_gripper"; fi
# SUPERGATE_HIT_SPEED_SCALE: the field Planner's operator scale on the racket velocity (tools/supergate/hit_scale.py):
# a number for every shot, or a file with a scale per shot (a real-session suite's hit_speed_scale.txt); suffix _scaleXXX.
SCALE=""
if [ -n "${SUPERGATE_HIT_SPEED_SCALE:-}" ]; then
  if [ -f "$SUPERGATE_HIT_SPEED_SCALE" ]; then SCALE="SUPERGATE_HIT_SPEED_SCALE=$(realpath "$SUPERGATE_HIT_SPEED_SCALE")"; OUT="${OUT}_scalereal"
  else SCALE="SUPERGATE_HIT_SPEED_SCALE=$SUPERGATE_HIT_SPEED_SCALE"; OUT="${OUT}_scale${SUPERGATE_HIT_SPEED_SCALE/./}"; fi
fi
# SUPERGATE_ACTUATORS=<json of tools/real/replay_joints.py --fit>: the plant runs with the actuators identified on the
# robot (tools/supergate/actuators.py); suffix _act<name of the json>.
ACT=""
if [ -n "${SUPERGATE_ACTUATORS:-}" ]; then ACT="SUPERGATE_ACTUATORS=$(realpath "$SUPERGATE_ACTUATORS")"; OUT="${OUT}_act$(basename "$SUPERGATE_ACTUATORS" .json)"; fi
# SUPERGATE_COMMIT_X=<x>: the Planner may commit once the ball is inside x instead of past the net at 1.37 (tools/supergate/
# commit_gate.py); SUPERGATE_POLICY_LEAD=<s>: adaptive_policy_lead_s other than 0.35; SUPERGATE_BALL_NOISE=<m>: white
# noise on the ball positions the Planner sees. Suffixes _gateXXX, _leadXXX, _noiseXXX.
GATE=""
if [ -n "${SUPERGATE_COMMIT_X:-}" ]; then GATE="SUPERGATE_COMMIT_X=$SUPERGATE_COMMIT_X"; OUT="${OUT}_gate${SUPERGATE_COMMIT_X/./}"; fi
if [ -n "${SUPERGATE_POLICY_LEAD:-}" ]; then GATE="$GATE SUPERGATE_POLICY_LEAD=$SUPERGATE_POLICY_LEAD"; OUT="${OUT}_lead${SUPERGATE_POLICY_LEAD/./}"; fi
if [ -n "${SUPERGATE_BALL_NOISE:-}" ]; then GATE="$GATE SUPERGATE_BALL_NOISE=$SUPERGATE_BALL_NOISE"; OUT="${OUT}_noise${SUPERGATE_BALL_NOISE/./}"; fi
# SUPERGATE_PREDICTION_ERROR=st,sy,sz: the Planner's strike prediction is off by that much (s, m, m at 0.40 s of notice;
# 0.030,0.033,0.035 on the real sessions of 2026-11-21, tools/real/commit_replay.py). Suffix _perr<digits>.
if [ -n "${SUPERGATE_PREDICTION_ERROR:-}" ]; then GATE="$GATE SUPERGATE_PREDICTION_ERROR=$SUPERGATE_PREDICTION_ERROR"; OUT="${OUT}_perr$(echo "$SUPERGATE_PREDICTION_ERROR" | tr -d '.' | tr ',' '_')"; fi
[ ! -e "$OUT" ] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
if [ ! -f "$BUNDLE/exported/policy.onnx" ]; then
  mkdir -p "$RUN/gate3"
  # On the CPU by default (SUPERGATE_EXPORT_DEVICE): with a training run on each GPU the export ran out of
  # memory, left an empty bundle directory behind and every later attempt stopped at "File exists".
  [ ! -d "$BUNDLE" ] || rmdir "$BUNDLE" 2>/dev/null || true
  distrobox enter grasping -- bash -lc "cd '$REPO/hope_training/whole_body_tracking' && source setup_train_env.sh >/dev/null 2>&1 && \
    hope_isaac_py '$HERE/../build_gate3_bundle.py' --checkpoint '$CKPT' --output '$BUNDLE' --strike-plane 0.58 $SPLIT_ARG --device ${SUPERGATE_EXPORT_DEVICE:-cpu}" 2>&1 | grep GATE3_BUNDLE | cut -c1-60 \
    || { echo "bundle export failed: $BUNDLE" >&2; exit 3; }
fi
mkdir -p "$RUN/supergate"
POLICIES="--policy sm$N=$BUNDLE"
for extra in "$@"; do POLICIES="$POLICIES --policy $extra"; done
distrobox enter hope -- bash -lc "cd '$REPO' && $BANDS $HAND $SCALE $ACT $GATE SUPERGATE_HOME_X=$HOME_X .venv/bin/python '$SG/run.py' --library '$TREE/build/native_batch.so' \
  --model-xml a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
  --ready-state '$PKG/logs/supergate/ready_stand.npz' $POLICIES --dataset '$DATA' --output '$OUT' --threads $THREADS" > "$OUT.log" 2>&1
python3 "$HERE/report.py" "$OUT" --dataset "$DATA"
