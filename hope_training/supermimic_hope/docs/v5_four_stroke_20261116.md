# SuperMimic 训练当前四动作任务（v26 DeployPlane）：设计问题与修复

日期：2026-11-16（America/Los_Angeles）。对象：分支 `supersuper`，任务
`HOPEPingPongMotionQualityDeployPlane`，算法 `supermimic.algorithm=analytic`（v4）。

本文只有代码核对、一致性检查和短跑计时。**没有长训练，没有新策略的击球、抖动、Gate3 或实机结果。**
下面两个新取值（时钟倍率 5、复位速度噪声上限）是按已有证据推出的候选值，要靠完整训练验证。

## 0. 起点

`supersuper` 上只有 SuperMimic 的物理模型骨架（`a3.py`）；完整的 v4 代码在
`origin/main_four_super`（最后提交 2026-10-31）。那份代码的可微任务头对应 10 月底的任务，
当前任务（v23 恢复 → RateDelay → BroadBall → DeployPlane）之后改过动作项、奖励和 MJCF，
原样搬过来第一步就报错。本次把代码搬到当前分支并逐项对齐，PPO 的配方和代码没有改动。

## 1. 设计上不合理的地方

### 1.1 课程和奖励渐入按"环境步数"计时，时钟是给 PPO 标定的（影响最大）

任务里所有 `*_ramp_steps` / `*_start_steps` 读的是 `env.common_step_counter`。PPO 每次 update
走 64 个环境步，SuperMimic 每次迭代只走 `1/utd = 2.5` 步，同一个时钟慢 25.6 倍；而 SuperMimic
达到同样击球水平需要的环境步数只有 PPO 的约 1/5（见下）。不缩放时，时钟比学习进度慢约 5 倍：

| 配置项 | 环境步 | PPO update（共 12000） | SuperMimic 迭代，原样 | SuperMimic 迭代，÷5 |
|---|---:|---:|---:|---:|
| HOLD / 站位脚 / 折返 / 力矩阶跃代价渐入 | 32,000 | 500 | 12,800 | 2,560 |
| 回位代价、动捕鲁棒性渐入 | 64,000 | 1,000 | 25,600 | 5,120 |
| 推力、起始偏航、球拍速度范围 | 96,000 | 1,500 | 38,400 | 7,680 |
| `full_body_quiet` 起点 → 满强度 | 128,000 → 256,000 | 2,000 → 4,000 | 51,200 → 102,400 | 10,240 → 20,480 |
| 题库难度下限 `coverage_floor` 起点 → 全开 | 48,000 → 288,000 | 750 → 4,500 | 19,200 → 115,200 | 3,840 → 23,040 |

原样跑 90k 迭代（225k 环境步）到结束时：题库难度下限只到 0.79（最难的 21% 题从未按时钟放开），
`full_body_quiet` 只到 0.76，前 38k 迭代都还在渐入推力、起始偏航和球拍速度。PPO 在整个训练的
前 37.5% 就完成全部日程。

倍率 5 的来源：PPO 命中率 0.95 出现在 update 10000–12000（640k–770k 环境步，BroadBall 无 CAPS run）；
上一个四动作任务上 SuperMimic 在 51.3k 迭代（128k 环境步）达到无噪声综合击球成功率 0.95
（v4 笔记第 12 节）。两者任务版本、物理后端和指标都不同，所以 5 是估计值，不是测量值。
按 5 缩放并训练 60k 迭代（150k 步）时，全部日程在第 23k 迭代（38%）完成，与 PPO 的 37.5% 对齐。

修复：`supermimic.schedule_clock_scale: 5.0`（`cfg/supermimic_train.yaml`）。启动时把任务配置里的
学习日程长度除以该值，改动清单打印在日志里并写入 run 目录的 `schedule_clock.json`。
不改 `common_step_counter` 本身，因为 HOLD 驻留连续性、课程证据窗口都按真实步数判断。
不缩放的项：阶段检查间隔（200 步）、证据最大年龄、驻留次数、HOLD 时长、延迟步数等真实时间量。
设为 1 即任务 YAML 原样。

没有处理的一处：`stroke_curriculum.close()` 里"近期成绩"的衰减常数写死为 5000 环境步
（PPO 下 78 次 update，SuperMimic 下 2000 次迭代）。阶段晋级的证据因此滞后约 2000 次迭代，
偏保守但不会卡死（难度下限时钟保证题库全开，并在全开时直接进入阶段 3）。

### 1.2 参考帧复位的根速度噪声，SuperMimic 学不过去

阶段 1 时复位组成是 25% 默认站姿 + 75% 参考帧复位（挥拍后回放概率在阶段 1 被课程压成 0）。
参考帧复位把机器人瞬移到挥拍中段，并叠加通用跟踪任务的根速度噪声
（±0.5 m/s 水平、±0.2 m/s 竖直、±0.52 rad/s 横滚/俯仰、±0.78 rad/s 偏航；任务 YAML 没写，
走 `tracking_env_cfg.py` 的默认值），是本任务自己推力随机化（0.1 m/s）的 5 倍。

v4 笔记第 15 节在上一个任务上的测量：这类起点 2 秒内跌倒的比例在 20k–52k 迭代的每个检查点都是
0.53–0.70，同期击球成功率从 0.04 升到 0.95；单独的速度噪声造成 0.48，位姿和关节噪声只有 0.03；
任何起点 5 秒之后跌倒率为 0。原因在学习器这一侧：瞬移后世界模型误差是稳态的 2 倍，
记录到的跌倒处软终止梯度与 critic 自举梯度相互抵消。PPO 在同一设置下能学会
（当前配方后期每回合跌倒 0.07–0.15），所以这个设置对 PPO 合理、对 SuperMimic 不合理。
这段复位代码在当前分支没有变化；**这个比例没有在当前任务上重新测量**（这里没有训练好的策略）。

修复：新任务 `cfg/task/HOPEPingPongMotionQualitySuperMimic.yaml`（继承 DeployPlane，只多一项覆盖）
把 `commands.motion.velocity_range` 收到推力随机化的量级（±0.1 m/s、竖直 ±0.05、
横滚/俯仰 ±0.1 rad/s、偏航 ±0.2 rad/s）。位姿和关节噪声不变，周期性推力照旧。
PPO 仍用 DeployPlane，不受影响。想保留原噪声做对照：`task=HOPEPingPongMotionQualityDeployPlane`。

### 1.3 可微任务头停在 10 月的任务版本

想象中的奖励是任务奖励的可微重写。当前任务有 4 个带权重的新项和 3 处改动没有进入任务头：

| 项 | 状态 |
|---|---|
| `racket_strike_precision`（-2.0，击球窗内移动目标位置误差） | 新增解析头 |
| `strike_preserving_action_rate`（-0.1，HOLD ×3，击球臂在击球窗豁免） | 新增；想象中携带上一拍原始动作 |
| `qdes_reversal`、`qdes_torque_step`（各 -0.05） | 新增；想象中携带上一拍执行指令增量 |
| `settle_foot_slip` 的 `all_phases` | 补上 |
| `fixed_station_feet` 的 `height_fraction`（0.25 → 0.75） | 补上（原来写死 0.25） |
| `full_body_quiet` 的非 `state_independent` 分支（站位距离、速度、倾角、角速度门控） | 补上；原来直接抛异常 |

未移植的项只能以"记录值减解析值"的常数修正进入想象，对动作没有梯度。现在带权重的项里只有
`undesired_contacts` 和 `joint_acc` 仍是录制修正（与 v4 相同）。

### 1.4 指令延迟模型换了

当前任务用 `TrainingLatency`：指令在每个控制拍内的到达相位**每拍**重新抽取（0–3 个 5 ms 子步，
25% 环境恒为 0），旧任务是每回合固定。世界模型的 `cmd_delay` 输入改为记录每拍的相位；
本配方的传感器延迟为 0，若以后打开 `sensor_max_substeps` 任务头会明确报未移植。

### 1.5 训练第一步的伪奖励

`env.reset()` 之后的第一步，指令缓存还没算过：球拍位置读成世界原点、时钟为 0，
`racket_strike_precision` 给每个环境记一笔几米量级的债（一致性检查里该项单步误差 5e2）。
PPO 里这只是一个瞬态；SuperMimic 会把这一行写进回放。现在训练和检查都先空走一步，不进回放。
修复前回放修正均值在前几次迭代是 0.27–0.39，修复后全程 1e-4。

### 1.6 物理模型没跟上 MJCF

当前 Gate3 MJCF 多了 8 个回球体和每只脚 13 个厂商 profile 的脚底球碰撞体。训练模型现在删除全部
`gate3_ball*`，并删除脚底球（与正常 HOPE profile 以及 Isaac 一致：只保留踝关节凸包接触）。
SHA 校验更新为当前文件。

### 1.7 其他

- 传感器噪声惩罚原来读 `algo.algorithm.sensor_noise_*`，当前分支的 PPO 配方没有这些键。
  已改为包内模块 `analytic/noise.py` + `supermimic.sensor_noise` 配置（数值与之前相同：系数 20）。
- 24 GB 显卡上策略批量可以用上游的 4096（之前受 12 GB 限制用 2048），回放也可以放在显卡上：
  实测 2.14 s/迭代、显存 14.0 GiB；2048 + 主机内存回放是 2.1–2.3 s/迭代、7.1 GiB。
  YAML 默认值仍是 12 GB 笔记本能跑的 2048 / cpu，本机用命令行覆盖。
- `diagnose_checkpoint` 默认分段跑完后写 `diagnosis.json` 会因报告里混入一个张量而失败
  （源分支上就有，与任务无关），已修。它的 critic 报告需要 `diagnose_steps` 大于 64。

## 2. 没有改、但要知道的限制

- **接触门控来自录制**：`settle_foot_slip`、`dynamic_support`、`full_body_quiet` 的"脚是否承重"
  取自真实那一步。想象里策略抬脚或落脚不会改变这些标志，所以"抬脚就不算搓地"这条路径对梯度不可见；
  防搓地的梯度主要来自可微的 `fixed_station_feet` 抬脚高度项。这是 v4 的既有设计。
- 学习器的 `action_smooth_weight`（0.02，HOLD ×5）与任务新增的原始动作变化率项作用相同。
  按每步计，任务项约为学习器项的 1/10（HOLD 里 1/17），两者叠加；没有重新调。
- v3（`algorithm=ppo_anchored`）没有在当前分支上运行或验证。
- 物理一致性（MuJoCo-Warp 对 PhysX）、任务收敛和硬件行为：`NOT_PROVEN`，与 v4 相同。

## 3. 验证记录

- `analytic_check`（64 环境，300 步，动作噪声 0.3）：40 个解析奖励项里 39 个与真实奖励逐项一致
  （最大误差 1.3e-4，多数 1e-6 以下）；`joint_torques` 是已知的近似（最大 0.19，由录制修正补齐）；
  五个终止余量 1.0e-3；FK 和球拍状态 1e-5。
- 训练中的一致性回放：奖励 ≤ 1.9e-6，执行指令 0，状态链 0。
- 全尺寸 6144 环境：45 次迭代（默认批量）和 180 次迭代（批量 4096 + 显卡回放），无 OOM、
  无跳过的 actor 步，退出码 0。
- 小规模（256 环境）：训练 → 保存 → 恢复 → `evaluate_checkpoints` → `diagnose_checkpoint`（默认全部分段）
  各跑通一次。
- 单元测试 `supermimic_hope/tests`：43 项通过（含新增的时钟缩放和噪声惩罚测试）。

## 4. 训练命令

进入容器（提示符已是 `[grasping]` 则跳过）：

```bash
distrobox enter grasping
```

```bash
cd /home/dongc1/workspace/nohope/hope_training/whole_body_tracking
source setup_train_env.sh
hope_isaac_py scripts/train_supermimic.py \
  task=HOPEPingPongMotionQualitySuperMimic algo=ppo_motion_quality \
  headless=true device=cuda:1 num_envs=6144 seed=0 \
  max_iterations=60000 \
  supermimic.policy_batch=4096 supermimic.replay_device=cuda:1 \
  checkpoint_path=null warm_start_actor_checkpoint=null \
  run_name=v26_supermimic_clock5
```

约 36 小时（2.14 s/迭代；前 500 次迭代只训世界模型，第一次策略更新编译约 3 分钟）。
输出在 `logs/supermimic/<时间>_v26_supermimic_clock5/`，每 300 次迭代存一个检查点。
本机还没有 W&B 登录：先 `wandb login`，或在命令后加 `supermimic.wandb.mode=offline`。
续训：加 `supermimic.resume_checkpoint=<run>/model_<N>.pt`（权重、优化器和环境步数计数器会恢复，
回放和课程状态重新开始）。

训练中看：

- `Curriculum/coverage_floor`：第 3.8k 迭代开始上升，第 23k 迭代到 1.0，随后 `Curriculum/stage` 变 3；
- `Imagination/check_*`：应保持 ≤ 2e-6，变大说明任务头与任务不一致；
- `Safety/termination_fraction`、`Safety/live_base_fell_tilt`：复位速度噪声收小后的跌倒率；
- `StandFH|StandBH|StepFH|StepBH/*` 命中与动作完成率，`WorldModel/validation_k_step_*`。

选检查点仍用无噪声固定条件评估（训练曲线带 0.1 的采集噪声）：

```bash
hope_isaac_py scripts/train_supermimic.py \
  task=HOPEPingPongMotionQualitySuperMimic algo=ppo_motion_quality \
  headless=true device=cuda:0 num_envs=1024 run_name=eval \
  '+supermimic.evaluate_checkpoints=[<run>/model_N.pt]' \
  +supermimic.evaluate_noise=0.0 supermimic.wandb.enabled=false
```

## 5. 本次改动的文件

- 从 `origin/main_four_super` 搬入：`hope_training/supermimic_hope/`（全部）、
  `whole_body_tracking/scripts/train_supermimic.py`、`whole_body_tracking/cfg/supermimic_train.yaml`。
- 在此基础上修改：`supermimic_hope/a3.py`、`analytic/task.py`、`analytic/learner.py`、
  `analytic/diagnose.py`、`analytic/check.py`、`environment.py`、`runner.py`、
  `trainer_analytic.py`、`trainer.py`、`tests/test_a3.py`、`cfg/supermimic_train.yaml`、`README.md`。
- 新增：`supermimic_hope/analytic/noise.py`、`tests/test_four_stroke_port.py`、
  `cfg/task/HOPEPingPongMotionQualitySuperMimic.yaml`、本文。
- 本机生成（git 忽略）：`assets/agibot_a3/`（`prepare_a3_isaac_asset.py --force`）、
  `~/.local/share/hope-supermimic/deps`（`requirements.txt` 的固定版本）。
