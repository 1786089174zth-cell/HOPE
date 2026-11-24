#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
root=$PWD
case "$1" in
ros)
    cd hope_ws
    MAKEFLAGS=-j6 colcon build --symlink-install --base-paths src ../NatNet2ROS2/src --packages-up-to hope_msgs hope_bringup hope_planner_cpp motion_capture_tracking --parallel-workers 3 --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
    ;;
sim)
    sim="$root/a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim"
    b="$root/../sim-build"
    cmake -S "$sim" -B "$b" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$b/install" -Daimrt_LOCAL_SOURCE="$root/../AimRT-1.6.0" -DAIMRT_MUJOCO_SIM_INSTALL=ON -DAIMRT_MUJOCO_SIM_BUILD_WITH_ROS2=ON -DAIMRT_MUJOCO_SIM_BUILD_TESTS=OFF -DAIMRT_MUJOCO_SIM_BUILD_EXAMPLES=OFF -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
    cmake --build "$b" --target install --parallel 12
    ;;
runner)
    cd a3_deploy/a3_deploy_example
    source setup_a3_env.sh
    bash scripts/build_a3_deploy_pkg.sh --arch x86_64 --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.hitter_pingpong.yaml --without-teleop --jobs 12
    ;;
esac
