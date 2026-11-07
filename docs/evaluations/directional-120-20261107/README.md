# 120 球台外斜入、扇形来球与 pelvis 丢帧：14600 / 19000 / 29999

三个模型使用相同的 120 球场景、同一份真实 C++ 规划命令记录、相同的 10 段 pelvis 丢帧和目标过期处理。控制为 50 Hz，MuJoCo 以 1 ms 积分。19000/29999 保持 GRU 隐藏状态连续，14600 是原始非 GRU actor。只在开始时重置一次；摔倒后停止继续发球，保留物理现场并结算仍在飞行的球。

| 模型 | 发出 / 计划 | 原始球拍接触 | 漏接 | 摔倒 | 已经历丢帧段数 |
|---|---:|---:|---:|---:|---:|
| model_14600（非 GRU） | 120 / 120 | 57 | 63 | 0 | 10 |
| model_19000 | 120 / 120 | 73 | 47 | 0 | 10 |
| model_29999 | 96 / 120 | 53 | 43 | 1，166.66 s | 9 |

14600 追加测试完整运行到 204.72 s，无数值报警或额外重置；最大倾角 18.146°、最大 pelvis 水平位移 0.5317 m（包括发球前 20 s 等待段，峰值在 3.70 s）。19000 对应为 4.214°、0.08477 m。14600 同样未摔倒，但位移与倾角更大。各模型保留各自原始默认姿态和动作偏置，且训练配方不同；这些结果不能单独确定 GRU 的因果作用。模型来源、非 GRU 接口核验和复现详见 [14600 说明](model14600.md)。

19000 完整运行到 204.72 s，最大控制采样倾角 4.214°，没有数值报警或额外重置。29999 在第 96 球附近达到跌倒阈值，后 24 球未发出；没有时钟或数值错误，未把它重置后继续凑满 120 球。第 95 球为右扇形球、第 96 球为左侧混合球：两个模型收到的目标位置与速度一致，目标世界 y 从约 +0.950 m 切换到 −0.6855 m，29999 随后失稳。摔倒当时不在丢帧窗口内，上一段丢帧在 151.5 s 已结束；本次组合试验不能单独确定丢帧与摔倒之间的因果关系。

新增类别的原始球拍接触 / 已发出如下。29999 的分母较少是因为提前停止，不能把两列当作相同样本量的接球率比较。两组右侧扇形球都没有球拍接触，接球泛化仍存在明显缺口。

| 新增类别 | 19000 | 29999 |
|---|---:|---:|
| 左台外斜入 | 10/19 | 6/15 |
| 右台外斜入 | 15/19 | 10/15 |
| 左扇形 | 8/19 | 6/15 |
| 中扇形 | 19/19 | 13/15 |
| 右扇形 | 0/19 | 0/15 |

## 来球覆盖

- 6 球原名义基准。
- 左右台外斜入各 19 球：发球世界 y 为 ±1.5～±2.8 m，均在球台侧边线 ±0.7625 m 之外；横向速度指向台内，包含同侧和跨向另一侧的落向，水平偏角 26.86°～51.82°。
- 19 组同起点扇形球，共 57 球：每组三球的 x/y/z 起点、vx/vz 相同，改变 vy，分别朝左、中、右发球；扇形目标横向宽度取 ±0.55、±0.75、±0.95 m，中线球目标世界 y=0。
- 19 球混合原有高、低、左右宽球及斜向球。各类别交错出现，全部参数见 [逐球场景表](directional-120-shots.csv)。

完整场景为 [directional-120.yaml](directional-120.yaml)。使用原 PPQ004 law 对 120 球逐一进行无机器人接触的物理预检，120 条都正常结束，118 条穿过 x=0.58 m 的机器人击球平面；第 23、95 球在此之前已偏出/落地。部分极端球到达时低于可击球高度，保留为允许漏接的安全压力输入。本轮没有修改碰撞系数，没有关闭 PPQ004 未分离保护，也没有把异常物理轨迹裁掉后计作通过。预检记录见 [ball-flight-validation.json](ball-flight-validation.json)。

## 丢帧与目标处理

10 段丢帧的起点:时长（秒）：`21:0.04,31:0.1,43:0.2,52:0.3,61:1.0,78:0.5,90:0.2,120:1.0,150:1.5,180:2.0`。19000 注入 342 个 actor tick；29999 在最后一段前摔倒，已注入 242 个 actor tick。丢帧时保持上次有效 pelvis 位置和朝向 XY，station delta 置零，关节、陀螺仪和重力方向继续更新，GRU 隐藏状态持续保留。使用此前验证的 `--neutral-expired-targets --neutral-final-tail`，击球 deadline+0.5 s 后输入中性目标。

规划目标来自 `model19000-90thosof`：保持机器人在初始姿态，使用 100 Hz 模拟传感器和已有 ROS Jazzy / CycloneDDS C++ 规划器，完整收集 120 球。112 球得到接受的规划目标，8 球没有目标（9、10、57、59、75、77、81、82）。两个模型的 planner-parameters.json SHA-256 相同，故正式 actor 对照使用相同目标记录；19000 重放 112 条，29999 在停止前重放 88 条。目标按相同仿真时刻送入，并保留期限、速度和方向，未按模型单独调整。

**范围：这是记录规划目标的 GRU / 物理稳定性对照。**它没有验证实时规划器遇到 pelvis 断流后的新决策，也不是实机不倒保证。原始球拍接触数不能替代严格合格击球或合法回球，本次两组合法落台计数均为 0。台外/低飞球不一定满足标准乒乓球来球有效性，因此漏接也不能全部归因于策略。

## 证据与复现

[完整对照结果及哈希](comparison-results.json)、[规划命令](planner-command-replay.json)、[规划目标收集汇总](planner-source-summary.json)、[19000 汇总](model19000-summary.json)、[19000 逐球](model19000-per-ball.csv)、[29999 汇总](model29999-summary.json)、[29999 逐球](model29999-per-ball.csv)。原始 trace、球轨迹、完整状态、日志与视频位于 WSL `/home/rog/hope-evaluations/directional-120-comparison/`；19000 为 `model19000-7uc_63xo`，29999 为 `model29999-rn8cql4b`。

场景生成器 `hope_training/whole_body_tracking/scripts/generate_directional_120.py`；逐球预检 `validate_directional_ball_flights.py`；目标收集使用 `run_model19000_mujoco.py --planner-only --sensor-stride 10`；从完整收集结果生成 replay 后，执行：

```bash
python hope_training/whole_body_tracking/scripts/run_directional_replay_comparison.py \
  --scene docs/evaluations/directional-120-20261107/directional-120.yaml \
  --replay docs/evaluations/directional-120-20261107/planner-command-replay.json \
  --output-root /home/rog/hope-evaluations/directional-120-comparison
```

实际状态录像以 `render_reference_diagnostic.py --wide-view` 离线渲染，再由 `label_directional_video.py` 标出 checkpoint、球型、仿真时间、丢帧状态和跌倒状态。渲染不执行新的策略或碰撞计算，不改变评分。两个模型使用相同相机和 20 fps / 1 倍速。画面中的 `LIVE PLANNER TARGET` 是通用渲染器沿用的目标标记，本次表示记录的目标，不是渲染时重新规划。

视频均已通过 ffmpeg 完整解码，校验信息见 [video-receipts.json](video-receipts.json)：19000 共 4095 帧、204.75 s，29999 共 3343 帧、167.15 s。Windows 本地可播放副本为 `C:/Users/ROG/.codex/visualizations/2026/11/07/01a1b748-2975-7491-a916-d27d36f7c468/model19000-directional-120-dropout.mp4` 与同目录 `model29999-directional-120-dropout.mp4`。29999 的失稳可从约 2:45 开始观看。
