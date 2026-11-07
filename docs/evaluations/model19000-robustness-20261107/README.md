# model_19000 极端来球与 pelvis 丢帧仿真诊断（2026-11-07）

模型：`gru-formal-2b8c09938a / model_19000`，原始 checkpoint SHA-256 `744526d2e7a9fdabe990070054ec3bd8126b286bc79041f94a930b0078c257c0`，ONNX SHA-256 `f3f56fe7e0c31f94bea1645fae8d2f776dca01033f2c895d850bf5c708801dda`。评测从 `feature/stationary-gru-mujoco-eval` 的 `2e95473b1ec8b8a0973e37783c5f81273af693ec` 派生。MuJoCo 使用 PPQ004 实际球碰撞、真实 C++ ROS 规划器、连续 GRU 隐藏状态和无重置规则；跌倒即停止继续发球。不是正式 Gate3，也不是实机安全证明。

| 场景 | 发出/计划 100 球 | 及时规划 | 球拍接触 | 漏接 | 跌倒 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 同环境无扰动对照 | 100/100 | 100 | 100 | 0 | 1，153.02 s | 末尾恢复段失稳 |
| 更宽侧向分布 | 100/100 | 100 | 87 | 13 | 1，153.00 s | **未通过不跌倒要求** |
| 斜向速度 | 25/100 | 16 | 13 | 12 | 1，62.00 s | **第 25 球附近早期失稳**；其余 75 球未发出 |
| 原始球 + pelvis 丢帧 | 100/100 | 100 | 100 | 0 | 1，153.00 s | 丢帧区间未观察到额外跌倒，末尾仍失稳 |

宽分布：前 6 球沿用原基准；其余 94 球的发球 `y` 均匀取 `[-1.5,-0.2] m`，`vx=-3 m/s, vy=0`。斜向：前 6 球沿用原基准；其余球取 `y∈{-1.05,-0.92,-0.78,-0.65} m`、`vx∈{-2.5,-3,-3.5,-4} m/s`、`vy∈{±0.5,±1,±1.5} m/s`，水平入射角约 7°–31°。固定种子 `19000`，各场景 YAML 保留在本目录。斜向测试中 25 个已发球只有 17 个满足原判定的有效来球条件，因此接球统计不能直接解释为策略单独失误；跌倒仍是真实仿真植物状态越界（约 61.98 s 倾角 60.09°，阈值 60°）。

pelvis 丢帧在仿真时间 `21:0.04,43:0.2,61:1.0,90:0.2,120:1.0` 秒注入。Actor 的 pelvis 位置和朝向 XY 保持上次有效值，基座目标位移置零；关节、陀螺仪和重力方向保持实时；规划器在区间内不发布新的 base pose；GRU 隐藏状态不断开。实际注入 122 个 50 Hz actor tick，ROS 端省略 488 次 200 Hz base publish。五球丢帧预检也通过：5/5 及时规划、5/5 真实接触、0 跌倒。100 球丢帧组和无扰动组都在末尾跌倒，不能宣称丢帧安全。

丢帧窗口及恢复后 0.2 s 内，各关节目标的最大单步变化（rad，丢帧/对照）依次为 `0.175/0.175`、`0.150/0.150`、`0.515/0.511`、`0.109/0.108`、`0.537/0.562`。在这些窗口里没有发现比无丢帧对照明显更大的目标突变；这不覆盖随机多次丢帧、错误跳变、长时断流、真实动捕延迟或驱动器行为。

评测默认的 50 ms 实时门槛在这台 WSL 主机上偶发中断，初次宽分布运行在第 8、60 球分别因 ROS 时间偏差/51 ms 调度延迟中止，未纳入结果。正式四组将评测机时钟容差明确设为 250 ms，原始“及时规划”最小提前量 40 ms 和超时指令拒收规则保留。实测最大调度延迟：宽分布 63 ms、斜向 16 ms、丢帧 13 ms、无扰动 117 ms。因此这组结果仅用于仿真稳定性诊断，不可当作 50 ms 实时验收。每组严格合格击球和合法落台均为 0；球拍原始接触不能替代合法回球。

## 复现与证据

在本分支仓库根目录，使用 Python 3.12、NumPy 2.5.4、ONNX Runtime 1.31.1、MuJoCo 3.17.0、PyYAML 6.0.3、ROS Jazzy Cyclone DDS；`hope` Distrobox 中的 C++ 规划器是已有构建。先运行 `scripts/generate_model19000_extreme_scenes.py --output-dir <out>`，再执行：

```bash
python hope_training/whole_body_tracking/scripts/run_model19000_mujoco.py \
  --scene <out>/model19000-wide-100.yaml --ros-rmw rmw_cyclonedds_cpp \
  --max-clock-lag-s 0.25 --max-clock-skew-ms 250 \
  --ros-setup /home/rog/workspace/HOPE/hope_ws/install/setup.bash \
  --planner /home/rog/workspace/HOPE/hope_ws/install/hope_planner_cpp/lib/hope_planner_cpp/hope_planner_cpp_node
```

斜向组改用 `model19000-oblique-100.yaml`；pelvis 组不设 `--scene`，添加 `--pelvis-dropout-windows 21:0.04,43:0.2,61:1.0,90:0.2,120:1.0`；对照组不设这两个参数。运行器给每次尝试建立独立目录。完整原始轨迹、ROS 日志和逐 tick 策略 trace 位于 WSL `/home/rog/hope-evaluations/model19000-robustness/`，本目录保存小型结算快照和逐球 CSV：

- 对照 `model19000-ie2z6eb2`：[summary](control-summary.json)、[逐球](control-per-ball.csv)
- 宽分布 `model19000-k98evzig`：[summary](wide-summary.json)、[逐球](wide-per-ball.csv)、[场景](model19000-wide-100.yaml)
- 斜向 `model19000-ajieqtut`：[summary](oblique-summary.json)、[逐球](oblique-per-ball.csv)、[场景](model19000-oblique-100.yaml)
- pelvis 丢帧 `model19000-ouj7j24x`：[summary](dropout-summary.json)、[逐球](dropout-per-ball.csv)
- 五球集成预检：[无扰动](nominal_preflight-summary.json)、[丢帧](dropout_preflight-summary.json)

本次只评估 `model_19000`。原发布报告中的“100 球”表示发出并结算；该模型原报告同样在末尾恢复段发生 1 次跌倒，不等于 100 次成功回球或全程不倒。
