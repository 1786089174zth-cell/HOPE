"""Training loop: collect -> fit the dynamics model -> improve the policy (Algorithm A.1)."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .dynamics import Dynamics
from .learners import LearnCfg, PolicyLearner, WorldModelLearner, cosine_lr
from .nets import Actor, Critic
from .replay import Tape
from .sim import SimCfg, WarpSim
from .task import FourStrokeTask, TaskCfg


@dataclass
class TrainCfg:
    num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 0
    max_iterations: int = 40000
    utd: float = 0.4  # model/policy updates per collected vector step (1/utd steps per iteration)
    buffer_steps: int = 1000
    warmup_steps: int = 64
    wm_only_iterations: int = 200  # iterations that only fit the dynamics model
    # After a resume: iterations that only fit the dynamics model again. For a fine-tune whose simulator differs
    # from the checkpoint's (new randomization, a wider context): the policy gradient runs through the model, so
    # the model learns the new dynamics before the policy follows it.
    wm_refit_iterations: int = 0
    action_noise: float = 0.1  # collection noise per action dimension (paper Table A.5)
    frames: int = 10  # consecutive 110-D observations the actor receives
    actor_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    dynamics_hidden: tuple[int, ...] = (2048, 2048, 2048)
    lr_decay_iters: int = 250_000
    log_interval: int = 20
    save_interval: int = 1000
    out: str | None = None
    run_name: str = "four_stroke"
    resume: str | None = None
    profile: bool = False  # synchronize around each phase and log its seconds per iteration
    capture: bool = True  # replay the model and policy updates as CUDA graphs
    amp: bool = False  # run the dynamics and actor MLPs in bfloat16
    # Polyak average of the actor weights per policy update, saved as ``actor_ema``. Consecutive
    # iterates swing by tens of points of hit rate; the average is what evaluate/export use.
    ema_decay: float = 0.99
    learn: LearnCfg = field(default_factory=LearnCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    task: TaskCfg = field(default_factory=TaskCfg)


class Trainer:
    def __init__(self, cfg: TrainCfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dev = cfg.device
        self.sim = WarpSim(cfg.num_envs, dev, cfg.sim, seed=cfg.seed)
        self.task = FourStrokeTask(self.sim, cfg.task, seed=cfg.seed)
        self.ctx = self.task.context()  # (N, C): sim context + per-env task constants
        self.f = Dynamics(self.ctx.shape[1], cfg.dynamics_hidden, self.task.dt, amp=cfg.amp).to(dev)
        self.actor = Actor(cfg.frames, cfg.actor_hidden, self.task.active, amp=cfg.amp).to(dev)
        self.a_prev = torch.zeros(cfg.num_envs, 31, device=dev)
        self.a_prev2 = torch.zeros(cfg.num_envs, 31, device=dev)
        self.task.reset(torch.arange(cfg.num_envs, device=dev), initial=True)
        critic_dim = self.task.critic_obs(self.sim.state(), self.a_prev, self.task.tape(), self.ctx).shape[1]
        self.critic = Critic(critic_dim, cfg.critic_hidden).to(dev)
        self.actor_ema = copy.deepcopy(self.actor).requires_grad_(False)
        self.wm = WorldModelLearner(self.f, cfg.learn, cfg.capture)
        self.pi = PolicyLearner(self.actor, self.critic, self.f, self.task, cfg.learn, cfg.capture)
        self.tape = Tape(cfg.buffer_steps, cfg.num_envs, dev)
        self.iteration = 0
        self.resumed_at = 0
        self.steps = 0
        self.q_prev = self.task.q0.clone()
        self.push_eval, self.pushes = False, 0  # evaluation tools switch pushes on explicitly
        frame = self.task.frame(self.sim.state(), self.a_prev, self.task.tape(), noise=True)
        self.history = frame[:, None].repeat(1, cfg.frames, 1)

    # ------------------------------------------------------------------ collection
    @torch.no_grad()
    def collect(self, n, noise=None, learn=True):
        """Step the simulator ``n`` ticks with the actor. ``learn=False`` (evaluation) leaves the
        observation and dynamics normalizers untouched: scoring a policy must not change it."""
        cfg, task, sim = self.cfg, self.task, self.sim
        noise = cfg.action_noise if noise is None else noise
        acc = {}
        for _ in range(n):
            s = sim.state()
            tape = task.tape()
            frame = task.frame(s, self.a_prev, tape, noise=True)
            self.history = torch.cat((self.history[:, 1:], frame[:, None]), 1)
            if learn:
                self.actor.norm.update(frame)
            a = self.actor(self.history)
            if noise:
                a = (a + noise * torch.randn_like(a)) * task.active
            q_cmd = task.qdes(a)
            q_exec = torch.where(task.delay[:, None], self.q_prev, q_cmd)
            if cfg.task.entry_handoff_s > 0 and cfg.task.entry_handoff_probability > 0:
                # Entry handoff: the plant runs the blended PD; the tape logs the equivalent target.
                q_exec, q_pd, kp, kd = task.handoff(s, q_exec, tape, sim.kp, sim.kd)
            else:
                q_pd, kp, kd = q_exec, None, None
            if learn:
                self.f.update_normalization(s, q_exec)
            sim.step(q_pd, kp, kd)
            s_next = sim.state()
            task.advance()
            nxt = task.tape()
            reward, terms = task.reward(s_next, a, self.a_prev, nxt, detail=True, a_prev2=self.a_prev2, s_prev=s, q_exec=q_exec)
            terminated = task.terminated(s_next, nxt)
            truncated = task.timed_out() & ~terminated
            done = terminated | truncated
            record = dict(s=s, qdes=q_exec, a=a, a_prev=self.a_prev, qcmd_prev=self.q_prev, frame=frame,
                          delay=task.delay.float(), **{"tape/" + k: v for k, v in tape.items() if k != "q0"})
            # Random pushes (training only unless ``self.push_eval``): a velocity kick on the root after this
            # tick. The window boundary keeps the kick out of the dynamics model's training pairs.
            pushed = torch.zeros_like(done)
            tc = cfg.task
            if tc.push_rate > 0 and (learn or self.push_eval):
                pushed = (torch.rand(len(done), device=done.device) < tc.push_rate * task.dt) & ~done
                ids = pushed.nonzero().flatten()
                if len(ids):
                    kick = (torch.rand(len(ids), 6, device=done.device) * 2 - 1)
                    kick[:, 0:2] *= tc.push_xy
                    kick[:, 2] *= tc.push_z
                    kick[:, 3:5] = 0.0
                    kick[:, 5] *= tc.push_yaw
                    sim.qvel[ids, :6] += kick
                    self.pushes += len(ids)
            self.tape.add(record, done | pushed)
            task.observe(s_next, nxt, reward, terms, terminated, truncated)
            self.a_prev2, self.a_prev, self.q_prev = self.a_prev, a, q_cmd
            ids = done.nonzero().flatten()
            if len(ids):
                task.reset(ids)
                self.a_prev[ids] = 0.0
                self.a_prev2[ids] = 0.0
                self.q_prev[ids] = task.q0[ids]
                s0 = sim.state()
                first = task.frame(s0, self.a_prev, task.tape(), noise=True)
                # The deployed runner seeds every history slot with the first frame of an episode.
                self.history[ids] = first[ids, None]
            acc.setdefault("reward", []).append(reward.mean())
            acc.setdefault("terminated", []).append(terminated.float().mean())
            self.steps += 1
        return {"collect/reward_per_s": torch.stack(acc["reward"]).mean() / task.dt,
                "collect/termination_per_step": torch.stack(acc["terminated"]).mean()}

    # ------------------------------------------------------------------ batches
    def wm_batch(self, batches=None):
        """``batches`` stacked window batches for one model update: (batches, B, k, ...)."""
        lc = self.cfg.learn
        n = lc.wm_batches if batches is None else batches
        b = self.tape.sample(n * lc.wm_batch, lc.wm_window, keys=("s", "qdes"))
        return dict(s=b["s"].unflatten(0, (n, lc.wm_batch)), qdes=b["qdes"].unflatten(0, (n, lc.wm_batch)),
                    ctx=self.ctx[b["env"]].unflatten(0, (n, lc.wm_batch)))

    def policy_batch(self):
        lc = self.cfg.learn
        b = self.tape.sample(lc.batch, lc.horizon + 1, history=self.cfg.frames - 1)
        env = b.pop("env")
        tape = {k[5:]: b.pop(k) for k in [k for k in b if k.startswith("tape/")]}
        # Per-environment constants are not logged every step.
        tape["q0"] = self.task.q0[env][:, None].expand(-1, lc.horizon + 1, -1).contiguous()
        return dict(b, ctx=self.ctx[env], tape=tape)

    # ------------------------------------------------------------------ loop
    def train(self):
        cfg = self.cfg
        out = Path(cfg.out or f"logs/supermimic/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{cfg.run_name}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        (out / "physics_receipt.json").write_text(json.dumps(self.sim.receipt, indent=2))
        writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(out / "tensorboard"))
        except Exception:  # noqa: BLE001 -- logging to metrics.jsonl still works
            pass
        if self.steps == 0:
            self.collect(cfg.warmup_steps)
        debt = 0.0
        t_last, it_last = time.perf_counter(), self.iteration
        with (out / "metrics.jsonl").open("a") as log:
            while self.iteration < cfg.max_iterations:
                it = self.iteration
                lr = cosine_lr(cfg.learn.lr, it, cfg.lr_decay_iters)
                self.pi.set_lr(lr, self.wm)
                debt += 1.0 / cfg.utd
                n = int(debt)
                debt -= n
                t0 = self._clock()
                metrics = self.collect(n)
                t1 = self._clock()
                metrics.update(self.wm.update(self.wm_batch()))
                t2 = self._clock()
                if it >= cfg.wm_only_iterations and it >= self.resumed_at + cfg.wm_refit_iterations:
                    metrics.update(self.pi.update(self.policy_batch()))
                    self._update_ema()
                t3 = self._clock()
                if cfg.profile:
                    metrics.update({"time/collect_s": t1 - t0, "time/world_model_s": t2 - t1, "time/policy_s": t3 - t2})
                self.iteration += 1
                if it % cfg.log_interval == 0:
                    metrics.update(self.wm.validate({k: v[0] for k, v in self.wm_batch(1).items()}))
                    metrics.update(self.task.metrics())
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                    row = {k: (float(v) if torch.is_tensor(v) else v) for k, v in metrics.items()}
                    row.update(iteration=it, env_steps=self.steps, lr=lr,
                               it_per_s=(self.iteration - it_last) / max(now - t_last, 1e-9))
                    t_last, it_last = now, self.iteration
                    row = {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in row.items()}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if writer is not None:
                        for k, v in row.items():
                            if isinstance(v, (int, float)):
                                writer.add_scalar(k, v, it)
                    keys = ("iteration", "it_per_s", "collect/reward_per_s", "collect/termination_per_step",
                            "wm/loss", "policy/imagined_reward_per_s", "task/hit_rate",
                            "task/torque_at_limit_stroke_tick_rate", "task/joint_beyond_range_tick_rate",
                            "time/collect_s", "time/world_model_s", "time/policy_s")
                    print("[supermimic] " + " ".join(f"{k}={row[k]:.4g}" for k in keys if row.get(k) is not None), flush=True)
                if self.iteration % cfg.save_interval == 0 or self.iteration == cfg.max_iterations:
                    self.save(out / f"model_{self.iteration}.pt")
        if writer is not None:
            writer.close()
        return out

    @torch.no_grad()
    def _update_ema(self):
        d = self.cfg.ema_decay
        for avg, p in zip(self.actor_ema.net.parameters(), self.actor.net.parameters()):
            avg.lerp_(p, 1.0 - d)
        self.actor_ema.norm.load_state_dict(self.actor.norm.state_dict())

    def _clock(self):
        if self.cfg.profile:
            torch.cuda.synchronize()
        return time.perf_counter()

    # ------------------------------------------------------------------ checkpoints
    def save(self, path):
        payload = dict(
            format="supermimic_hope_core_v1", iteration=self.iteration, env_steps=self.steps,
            actor=self.actor.state_dict(), actor_ema=self.actor_ema.state_dict(),
            critic=self.critic.state_dict(), dynamics=self.f.state_dict(),
            opt_pi=self.pi.opt_pi.state_dict(), opt_v=self.pi.opt_v.state_dict(), opt_wm=self.wm.opt.state_dict(),
            config=asdict(self.cfg), task_state=self.task.state_dict(),
        )
        tmp = Path(str(path) + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def load(self, path):
        payload = torch.load(path, map_location=self.cfg.device, weights_only=False)
        if payload.get("format") != "supermimic_hope_core_v1":
            raise ValueError(f"{path} is not a supermimic_hope core checkpoint")
        self.actor.load_state_dict(payload["actor"])
        self.actor_ema.load_state_dict(payload.get("actor_ema", payload["actor"]))
        # A simulator with more context than the checkpoint's (SimCfg.actuator_fit appends one entry): the context
        # is the tail of the critic's and the dynamics model's input, so their first layers, the critic's input
        # normalizer and the optimizers' moments get zero columns there and start as the checkpoint's functions.
        widened = []
        for module, key in ((self.critic, "critic"), (self.f, "dynamics")):
            state, own = dict(payload[key]), module.state_dict()
            for name, value in state.items():
                target = own[name]
                if value.shape != target.shape and value.ndim == target.ndim and value.shape[:-1] == target.shape[:-1] and value.shape[-1] < target.shape[-1]:
                    pad = target.shape[-1] - value.shape[-1]
                    fill = 1.0 if name.endswith((".var", ".std")) else 0.0
                    state[name] = torch.cat((value, torch.full((*value.shape[:-1], pad), fill, dtype=value.dtype, device=value.device)), -1)
                    widened.append((tuple(value.shape), tuple(target.shape)))
            module.load_state_dict(state)
        for opt, key in ((self.pi.opt_pi, "opt_pi"), (self.pi.opt_v, "opt_v"), (self.wm.opt, "opt_wm")):
            state = payload[key]
            if widened:
                shapes = dict(widened)
                for entry in state["state"].values():
                    for name, value in entry.items():
                        if torch.is_tensor(value) and tuple(value.shape) in shapes:
                            new = shapes[tuple(value.shape)]
                            entry[name] = torch.cat((value, torch.zeros((*value.shape[:-1], new[-1] - value.shape[-1]), dtype=value.dtype, device=value.device)), -1)
            opt.load_state_dict(state)
        self.task.load_state_dict(payload.get("task_state", {}))
        self.iteration = int(payload["iteration"])
        self.resumed_at = self.iteration
        return payload
