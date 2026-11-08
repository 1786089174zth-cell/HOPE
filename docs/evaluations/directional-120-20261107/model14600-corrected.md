# 14600 原生待机修正：教授仓库核查、A/B 与 120 球重跑

## 权威模型来源

本次直接读取教授 `hitchopen/HOPE` 和 Jiayi `dongc1/nohope` 的远端 `main`，两者当前 HEAD 都是 `7990f18c2c1ae560c9e0708dca10228b9f2cc610`。不依据学生 fork 是否同步来判断模型。两个远端版本的 `hope_training/whole_body_tracking/checkpoints/policy.onnx` 均为 ONNX SHA-256 `a987240610bb9a49ea42da8eafcdc84c3cfbc8d5470bf7ff1a2d7f1f624cdd28`，训练来源文件明确标记 `baseline_model=model_14600`，checkpoint SHA-256 为 `20053e683cdbefb24575299b8b2b5c3d0890e145d96fa2cc543136983455f0e1`。因此此前拿到的确实是 14600 权重，问题不在模型身份。

这里的“当前”指两个仓库 **main 的已发布 checkpoints 基准**，并不等于所有试验分支/未发布训练中迭代号最大的文件。主线另有 `models/model_21800/` 导出包（不同 ONNX SHA-256 `6bf1a2418f8538e23577a0153f2fe6a1e78dee91f41650a232259432a84a4dc8`），轻量示例 runtime.yaml 仍默认指向该包，不能仅凭这个默认路径认定已加载 14600。测试始终显式绑定并校验 a987… 权重。远端核查和附加分支导出检查见 [source audit](model14600-professor-source-audit.json)。

## 已确认的接入错误

第一次测试在没有规划目标或目标过期时，将 `target=current racket position, velocity=0, time_to_strike=-1` 送入 14600。这个处理来自另一条 GRU 诊断路径，不符合 14600 原生等待语义。此前只核验了网络、关节顺序、动作解码，未核验完整等待契约；因此不能把旧视频的抽搐及 0.53 m 漂移当成 14600 原生站立表现，第一次 120 球行为对照已作废。

教授源码 `pp_policy.hpp` 的 110-D hitter_pure 等待分支采用固定世界 HOME 加训练目标盒中心，配合训练速度盒中心和固定的正向准备时钟。14600 的元数据解析结果为：

| 等待侧 | HOME 相对目标 x/y，世界 z（m） | 目标速度（m/s） | 准备时钟 |
|---|---|---|---:|
| 正手 | `(0.58, -0.44, 1.075)` | `(1.92, 0.19, 1.03)` | +0.82 s |
| 反手 | `(0.58, -0.09, 1.075)` | `(2.05, -0.13, 0.675)` | +0.96 s |

准备时钟由原始 clip 长度和击球相位计算，对应原 Runner 的 max_tts 约束。该仓库还有独立的官方 PD_STAND，以及非 policy-native 模式中的静态接管分支；不能混同这些控制模式。修正后的 120 球测试采用 **actor 持续控制的 policy-native 等待目标**，没有插入 PD_STAND，没有冻结关节、平滑动作或重置来隐藏抖动。

## 同权重 20 秒待机 A/B

全部使用同一份 a987… 原始 actor、相同初始姿态、动作偏置、PD/物理参数。前三行始终由 actor 控制，最后一行单独核验官方 PD 模式，不参与正式 120 球评分。

| 输入/控制 | 最大水平位移 | 最大倾角 | 1 s 后最大关节目标单步变化 |
|---|---:|---:|---:|
| 旧适配器：当前球拍目标，时钟 −1 | 0.53172 m | 16.631° | 1.36243 rad |
| 仅将时钟改为 +0.82，其他公式相同 | 0.03349 m | 2.259° | 0.00935 rad |
| 完整原生固定世界等待目标 | 0.01550 m | 2.962° | 0.00489 rad |
| 官方 PD_STAND，actor 不控制 | 0.00106 m | 0.287° | 0 rad |

这组隔离试验支持：负向待机时钟及不匹配等待目标是旧视频抽搐的直接接入原因。具体数据见 [A/B summary](model14600-idle-ab-summary.json)。20 秒左右分屏视频由两组记录的 qpos 渲染，左侧为错误输入、右侧为完整原生等待；两侧权重和物理配置相同，右侧 actor 仍在控制。

## 修正后的完整 120 球

正式运行：`model14600-lgdvol6x`，原始文件位于 `/home/rog/hope-evaluations/model14600-directional-corrected/model14600-lgdvol6x/scored/`。使用与 19000/29999 相同的 directional-120 YAML、112 条规划位置/速度/绝对截止时间记录、10 段丢帧窗口、固定 HOME、50 Hz 控制和 1 ms PPQ004 物理。无球及 strike deadline+0.5 s 后改用上述原生等待目标；原生准备时间上限实际触发 **0 个 active tick**，所以有效击球阶段的位置、速度及时间输入没有因本次修正额外改变。没有 PD_OVERRIDE、模型/机器人/动作历史重置或数值补救。

结果：**120/120 发出并结算、60 次原始球拍接触、60 次漏接、0 摔倒、0 数值错误、0 额外重置**；10 段丢帧均执行，共 342 个 actor tick，仿真到 204.72 s。开头 20 秒等待的最大位移 1.550 cm、最大倾角 2.962°。全程最大位移 0.3033 m，发生于 120.999 s（120～121 s 丢帧内）；全程最大倾角 13.812°、最低 base 高度 0.9952 m。初始抽搐已修正，但强扰动期间的位置与姿态变化仍需按实际 trace 评价，不能把“未摔倒”当作完全不动。

| 来球类别 | 接触 / 发出 |
|---|---:|
| 名义球 | 6/6 |
| 左台外斜入 | 8/19 |
| 右台外斜入 | 12/19 |
| 左扇形 | 5/19 |
| 中扇形 | 15/19 |
| 右扇形 | 0/19 |

另有 19 个混合球，接触 14 个。严格合格击球和合法回球计数仍为 0；接触不能替代合法回球。有效的 [summary](model14600-summary.json)、[逐球 CSV](model14600-per-ball.csv)、[完整对照指标](model14600-comparison.json)。`model14600-invalid-idle-*` 保留先前错误运行供审计，不作为模型比较。

## 对比边界与复现

14600、19000、29999 共用物理来球、记录规划目标和丢帧设置。14600 现按原生正时钟等待，GRU 结果保留其原记录的 −1 中性处理。各模型默认姿态、训练配方和运行时等待语义不同，因此结果不用于单独推断 GRU 的因果作用；记录目标重放也没有验证实时规划器面对断流的决策或实机安全。

`feedforward_eval_policy.py` 从原模型元数据生成等待目标；`reference_gru_diagnostic.py --feedforward-idle-contract native` 是默认的正确非 GRU 路径。`legacy-gru-neutral` 只保留作旧错误复现。生成模型 bundle 与完整运行命令见 [旧文档的复现路径](model14600.md)，运行器现在显式选择 native。独立待机 A/B 可运行 `audit_model14600_idle.py --bundle <bundle> --professor-parameters <从教授提交提取的 a3_policy_parameters.hpp> --output <新目录>`。

## 已验证的视频

两段均为记录状态的离线渲染并通过 ffmpeg 全片解码，校验见 [video-receipts.json](video-receipts.json)：

- 完整修正视频：`model14600-directional-120-dropout-corrected-idle.mp4`，4095 帧、20 fps、204.75 s，SHA-256 `4f23329c30ba5fe2bcf7e2c710d392f4829c8fb463b6d00202e78e32fc6b39f0`。
- 同权重站立 A/B：`model14600-idle-adapter-ab.mp4`，200 帧、10 fps、20 s，左右分别为错误输入和修正后的原生等待输入，两侧 actor 均持续控制。

Windows 文件位于 `C:/Users/ROG/.codex/visualizations/2026/11/07/01a1b748-2975-7491-a916-d27d36f7c468/`。旧 `model14600-directional-120-dropout.mp4` 已弃用，不应再用于评价模型本身。
