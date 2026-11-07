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

## 第 100 球后的跌倒：定位与诊断修正

无扰动对照在最后一次规划目标的击球期限约 `149.107 s` 后，不再有新球目标，但评测器仍逐 tick 把上次击球位置、速度及不断递减的剩余时间送入 GRU。`target_source` 一直记录为 `actual_live_planner`；`150.0 s` 的剩余时间约 `-0.89 s`、倾角 `1.47°`，到 `153.0 s` 变为 `-3.89 s`、`64.64°` 并触发跌倒。该对照没有注入 pelvis 丢帧；第 100 球后不再有新规划命令是正常的发球停止，并非证实动捕信号断流。过期目标与失稳同时出现，单靠相关性不能定因。

为隔离这个因素，评测器新增两个默认关闭的开关：`--neutral-final-tail` 在最后一球后、击球期限加 `0.5 s` 时，把目标置为当前球拍位置、速度零、剩余时间 `-1 s`；`--neutral-expired-targets` 在每次击球期限过后执行相同切换。GRU 隐藏状态、物理引擎、球序列和 ROS 规划器均保持原样。前者的对照重跑在 `150.46 s` 首次切换，`100/100` 球发出且接触、`0` 跌倒、`0` 重置，仿真走完整个 `153.48 s` 尾段；`153.0 s` 倾角 `3.87°`，而未切换对照为 `64.64°`。后者与五个 pelvis 丢帧窗口合并测试，也得到 `100/100` 发出与接触、`0` 跌倒、`0` 重置、`122` 个丢帧 actor tick、`877` 个中性目标 tick。该 A/B 结果强烈支持过期击球目标是末尾跌倒的直接诱因。

复现：在上面的命令中不设 `--scene`，分别添加 `--neutral-final-tail`，或添加 `--neutral-expired-targets --pelvis-dropout-windows 21:0.04,43:0.2,61:1.0,90:0.2,120:1.0`。小型证据：[末尾切换汇总](neutral-final-tail-summary.json)、[逐球](neutral-final-tail-per-ball.csv)；[通用过期切换与丢帧汇总](neutral-expired-dropout-summary.json)、[逐球](neutral-expired-dropout-per-ball.csv)。完整逐 tick trace 和 ROS 日志在 WSL `/home/rog/hope-evaluations/model19000-fix/`。

这仍是 MuJoCo 诊断开关，尚未接入实机。现场应分别处理规划目标过期与 pelvis 动捕过期：以单调时钟监测时间戳、拒绝无效/跳变位姿，短时保持最后有效位姿并限制目标变化，超出许可窗口后停止发球并平滑转入已验证的站立控制器；恢复时验证连续新帧再平滑接回。现有 C++ runner 有 mocap 失效转 PD_STAND 的路径，但发布的 GRU bundle 与现有实机 runner ABI 不兼容，不能用这里的仿真结果宣称硬件已安全。严格合格击球和合法落台仍为 `0`。

## 更宽来球叠加五段 pelvis 丢帧

同一 `model19000-wide-100.yaml` 与 `21:0.04,43:0.2,61:1.0,90:0.2,120:1.0` 秒的 pelvis 丢帧同时施加，并在最后一球后启用 `--neutral-final-tail`。本机使用实时 ROS 规划器的组合运行多次因主机时钟落后在第 9–21 球中止，均未走完 100 球；50 Hz 实时传感器发布虽然走完 100 球，但规划器没有给出任何有效命令，故 **不能算作组合测试通过**。

为单独评估固定宽分布和丢帧对 GRU/物理机器人稳定性的叠加影响，用原宽分布 100 球运行 `model19000-k98evzig` 的 100 条已接受规划命令按原仿真时刻重放；前 17 条命令与中止前实时组合运行的目标位置和速度逐条相同。重放桥接每 20 ms 送入已记录命令，actor 仍以 50 Hz 控制、物理仍以 1 ms 步长积分，连续 GRU 隐藏状态不重置。结果：**100/100 发出、100/100 重放目标接受、87 次原始球拍接触、13 次漏接、0 跌倒、0 重置**；五段注入覆盖 122 个 actor tick，最终恢复尾段 151 个中性目标 tick，仿真到 `153.48 s`，最大主机滞后 `0.104 s`。13 个漏接球号与原宽分布测试完全相同。

证据：[重放规划命令](wide-planner-command-replay.json)、[汇总](wide-dropout-replay-summary.json)、[逐球](wide-dropout-replay-per-ball.csv)、[桥接记录](wide-dropout-replay-bridge.txt)、[来源说明和哈希](wide-dropout-replay-provenance.json)；完整逐 tick trace、球轨迹和录制状态在 WSL `/home/rog/hope-evaluations/model19000-wide-replay-3pHEQX/scored/`。重放脚本为 `hope_training/whole_body_tracking/scripts/planner_command_replay_bridge.py`，诊断器使用原宽分布场景、`--neutral-final-tail --pelvis-dropout-windows 21:0.04,43:0.2,61:1.0,90:0.2,120:1.0 --sensor-stride 20 --video`。

**限制：**命令来自无丢帧的原宽分布运行，重放不验证实时规划器在 pelvis 断流后的新决策或 ROS 端到端时限。50 Hz 是命令重放传输率，不是原实时规划器的 200 Hz 传感器率；该结果是策略/物理稳定性诊断，不能替代完整的实时联测或实机安全验收。原始球拍接触也不等于合法回球。

## 三维混合来球与五段 pelvis 丢帧：90 球

这组按 `model19000-mixed-3d-90.yaml` 连续发出 90 球：前 6 球为名义基准，之后把中线、左侧、右侧、低球、高球、斜向球各 14 个交错排列。混合球发球台面坐标 `y=-1.505～-0.096 m`、`z=0.327～0.808 m`，`vx=-4.255～-2.940 m/s`、`vy=-1.256～1.526 m/s`、`vz=1.781～2.844 m/s`，水平偏转角最大 `21.53°`。实测 14 个中线球在机器人击球平面全部通过，世界坐标 `|y|≤0.0266 m`，故中线球确实瞄准了机器人中间。场景使用确定性种子 `19000`，各球参数见逐球场景表。

先以 100 Hz 仿真传感器驱动真实 C++ ROS 规划器收集目标。120 球原场景的前 90 球全部发出并完成物理轨迹，其中 89 球得到及时规划，第 50 球没有接受的规划命令；第 91 球在台面连续小幅弹跳，触发 PPQ004 “碰撞后 50 ms 未分离”保护，因此不将 120 球尝试报告为通过。90 球场景是该 120 球场景的完全相同前缀；重放文件只提取这 90 球的真实规划命令，并逐球检查场景参数一致。

把 89 条规划命令按原仿真时刻重放到 `model_19000` 的 50 Hz 连续 GRU / 1 ms MuJoCo 物理运行，同时注入 `21:0.04,43:0.2,61:1.0,90:0.2,120:1.0` 秒的五段 pelvis 丢帧，并启用 `--neutral-expired-targets --neutral-final-tail`。结果：**90/90 球发出并结算，71 次原始球拍接触，19 次漏接，0 次摔倒，0 次重置，0 次数值报警**；五段丢帧覆盖 122 个 actor tick，完整仿真 `141.98 s`，最大倾角 `4.02°`，最低 base 高度 `1.057 m`。原始接触按类别为：中线 `14/14`、左 `12/14`、右 `2/14`、低 `14/14`、高 `11/14`、斜向 `12/14`，另有名义球 `6/6`。右侧明显是这次分布的弱点；该结果满足本轮“不摔倒、允许漏接”的仿真判据，但接球泛化并不均匀。严格合格击球和合法回球仍不能由这些原始接触数推出。

证据：[90 球场景](model19000-mixed-3d-90.yaml)、[发球参数和类别](model19000-mixed-3d-90-shots.csv)、[规划命令重放](mixed-3d-90-planner-command-replay.json)、[汇总](mixed-3d-90-dropout-summary.json)、[逐球结果](mixed-3d-90-dropout-per-ball.csv)、[桥接计数](mixed-3d-90-dropout-bridge.txt)、[来源及哈希](mixed-3d-90-dropout-provenance.json)。生成器为 `hope_training/whole_body_tracking/scripts/generate_model19000_mixed_3d.py`，命令整理器为 `hope_training/whole_body_tracking/scripts/build_model19000_planner_replay.py`。完整 trace、实际球轨迹、逐帧录像状态保存在 WSL `/home/rog/hope-evaluations/model19000-mixed-replay-DTCqIz/scored/`。

**范围：**这是记录规划目标的重放测试；丢帧发生在 GRU 观测中，未验证实时规划器面对 pelvis 信号中断时的行为，也未完成 ROS 端到端实时验收。不能据此推断实机已保证不倒。
