#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
root="$(cd -- "$(dirname -- "$0")" && pwd)"
repo="$root/native"
cd "$repo"
export ROS_LOCALHOST_ONLY=1
args=(
  --policy-dir "$root/policy-35866-native"
  --runner-dist "$repo/a3_deploy/a3_deploy_example/dist/a3_deploy_x86_64"
  --sim-install "$root/sim-build/install"
  --hope-ws "$repo/hope_ws"
  --scenario fixed-home-100 --balls 100 --headless
)
exec bash run_gate3.sh "${args[@]}" "$@"
