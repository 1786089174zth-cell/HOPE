"""SuperMimic with analytic HOPE heads: collect -> fit world model -> BPTT actor + critic.

The policy update is the upstream algorithm (no PPO). Rewards, terminations and
observations in imagination are differentiable re-statements of the task,
checked against the real task by ``supermimic.analytic_check_steps``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import torch
from torch import nn
from tensordict import TensorDict
from omegaconf import OmegaConf

from .analytic import state as S
from .analytic.learner import Learner, TapeReplay, cosine_lr
from .analytic.recorder import Recorder
from .analytic.task import TaskModel
from .contracts import from_environment
from .logging import WandbLogger
from .telemetry import EpisodeStats, scalar_snapshot, task_snapshot


def build_policy(env, obs, cfg):
    """The ``train.py`` actor-critic (same ABI/checkpoint layout), without PPO storage."""
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs
    from whole_body_tracking.utils.bounded_actor_critic import BoundedStdActorCritic

    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    train_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name))).to_dict()
    policy_cfg = dict(train_cfg["policy"])
    policy_cfg.pop("class_name")
    term = env.action_manager.get_term("joint_pos")
    policy_cfg["entropy_excluded_action_indices"] = term._passive_action_cols.detach().cpu().tolist()
    for key in ("actor_obs_normalization", "critic_obs_normalization"):
        if policy_cfg.get(key) is None:
            policy_cfg[key] = bool(train_cfg["empirical_normalization"])
    policy = BoundedStdActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 31, **policy_cfg)
    return policy.to(env.device), train_cfg


def environment_xi(model, backend):
    """Startup-randomized dynamics context per environment (constant)."""
    import warp as wp
    import numpy as np

    c = model.const
    friction = wp.to_torch(backend.m.geom_friction)[..., 0]
    feet = [int(backend.body_ids[i]) for i in model.feet]
    geoms = [np.flatnonzero(backend.model.geom_bodyid == b) for b in feet]
    foot_mu = torch.stack([friction[:, torch.as_tensor(g, device=friction.device)].mean(-1) for g in geoms], -1)
    return torch.cat((
        c["kp"], c["kd"], c["masses"].sum(-1, keepdim=True), c["ipos"][:, 0], c["ipos"][:, model.anchor_body], foot_mu,
    ), -1)


def env_steps(payload, sc):
    """Cumulative environment steps of a checkpoint.

    The task's reward ramps (``full_body_hold``, ``fixed_station_feet``,
    ``full_body_quiet``, ``station_recovery``) and step-keyed curricula read
    ``env.common_step_counter``; a resume must continue it, or those terms
    restart near zero. Checkpoints before ``env_step_counter`` was saved are
    estimated from the iteration (1/utd vector steps each, plus warmup).
    """
    if "env_step_counter" in payload:
        return int(payload["env_step_counter"])
    return int(sc.warmup_steps) + int(payload["iter"] / float(sc.utd))


def resume(path, policy, learner, device, env=None, sc=None):
    """Continue from a v4 checkpoint: weights, normalizers, optimizers and,
    with ``env``, the environment step counter.

    Replay and per-command curriculum state are not saved; replay refills
    during the warmup collection before the next update.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("training_backend") != "supermimic_hope_analytic_v4":
        raise ValueError(f"{path} is not a supermimic_hope_analytic_v4 checkpoint")
    policy.load_state_dict(payload["model_state_dict"])
    learner.wm.load_state_dict(payload["world_model"])
    learner.opt_wm.load_state_dict(payload["opt_wm"])
    learner.opt_pi.load_state_dict(payload["opt_pi"])
    if payload.get("critic") is not None:
        dim = payload["critic_norm"]["mean"].numel()
        learner._ensure_critic(torch.zeros(1, dim, device=device))
        learner.critic.load_state_dict(payload["critic"])
        learner.critic_target.load_state_dict(payload.get("critic_target") or payload["critic"])
        learner.critic_norm.load_state_dict(payload["critic_norm"])
        if payload.get("opt_v") is not None:
            learner.opt_v.load_state_dict(payload["opt_v"])
    print(f"[supermimic] resumed weights from {path} at iteration {payload['iter']}", flush=True)
    if env is not None:
        env.common_step_counter = env_steps(payload, sc)
        print(f"[supermimic] environment step counter restored to {env.common_step_counter}", flush=True)
    return int(payload["iter"])


def build(env, obs, cfg, backend, sc=None):
    """Policy, analytic task heads and learner, as used for training."""
    from types import SimpleNamespace
    from .analytic.noise import from_config as sensor_noise

    sc = sc or cfg.supermimic
    policy, train_cfg = build_policy(env, TensorDict(obs, batch_size=[env.num_envs]), cfg)
    policy.train()
    with torch.no_grad():
        # Upstream: start from near-zero actions (the default stand pose).
        policy.actor[-1].weight.mul_(0.01)
        policy.actor[-1].bias.zero_()
    actor = nn.Sequential(policy.actor_obs_normalizer, policy.actor)
    model = TaskModel(env, backend)
    _, reasons, costs, _ = from_environment(env)
    model.bind_terminations(reasons)
    active = torch.ones(31, device=env.device)
    active[model.passive] = 0
    noise_penalty = sensor_noise(actor, sc.sensor_noise)
    xi = environment_xi(model, backend)
    learner = Learner(
        model, actor, list(policy.actor.parameters()), sc, xi, active, noise_penalty,
        reasons, costs[0], env.device,
    )
    term_index = model.termination_index
    keep = torch.ones(len(model.terms), device=env.device)
    keep[term_index] = 0
    return SimpleNamespace(
        policy=policy, train_cfg=train_cfg, actor=actor, model=model, learner=learner,
        active=active, term_index=term_index, keep=keep,
    )


def tape_record(model, recorder, keep, term_index, dt, reward, terminated, truncated):
    """Replay row of one real step: states, controls, context and the analytic correction."""
    rec, real = recorder.record, recorder.real
    ctx = {k[4:]: v for k, v in rec.items() if k.startswith("ctx/")}
    fr = model.frame(rec["s_cmd"], rec["s_end"], rec["action"], rec["qexec"], rec["qexec_prev"], ctx)
    analytic = model.step_reward(model.rewards(fr))
    real_rest = (real * keep).sum(-1) * dt
    record = {k: v for k, v in rec.items() if k not in ("s_start",)}
    record["push"] = (rec["s_start"] - rec["s_cmd"])[:, 7:13]
    record["correction"] = real_rest - analytic
    record["done"] = terminated | truncated
    record["terminal"] = terminated
    record["reward"] = reward
    record["terminal_reward"] = real[:, term_index] * dt
    return record, ctx


def train(env, obs, cfg, out, backend):
    from torch.utils.tensorboard import SummaryWriter

    sc = cfg.supermimic
    torch.manual_seed(int(cfg.seed))
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    obs = env.observation_manager.compute()
    parts = build(env, obs, cfg, backend)
    policy, train_cfg, actor, model, learner = parts.policy, parts.train_cfg, parts.actor, parts.model, parts.learner
    active, term_index, keep = parts.active, parts.term_index, parts.keep
    start_iteration = 0
    if sc.get("resume_checkpoint"):
        start_iteration = resume(sc.resume_checkpoint, policy, learner, env.device, env, sc)
    replay = TapeReplay(
        int(sc.buffer_steps), env.num_envs, env.device, float(sc.validation_env_fraction),
        storage=sc.get("replay_device") or env.device,
    )
    recorder = Recorder(env, backend, model)
    recorder.observe(obs)
    # The command caches (racket state, clocks) exist only after the first command
    # update. On the first step after ``env.reset`` the task reads a racket at the
    # world origin with a zero clock and charges every environment a several-metre
    # strike-precision debt. That row is not a task transition: step once with the
    # recorder attached (so its state capture stays aligned) and do not replay it.
    with torch.no_grad():
        obs = env.step(torch.zeros(env.num_envs, 31, device=env.device))[0]
    recorder.observe(obs)
    episodes = EpisodeStats(env.num_envs, env.device, env.step_dt)
    writer = SummaryWriter(str(out / "tensorboard"))
    identity = hashlib.sha256(OmegaConf.to_yaml(cfg.task, resolve=True).encode()).hexdigest()
    manifest = dict(
        algorithm="supermimic_hope_analytic_v4",
        upstream="a1508d42c9d1699a52e8233ab16d08c488441ab9",
        physics_backend=str(sc.backend),
        task_sha256=identity,
        actor_dim=110,
        action_dim=31,
        state="mujoco_qpos_qvel_isaac_order_75",
        world_model="heading_invariant_increments_executed_pd_targets",
        heads="analytic_rewards_terminations_observations_with_recorded_context",
        analytic_terms=sorted(model.analytic),
        recorded_terms=sorted(set(model.terms) - set(model.analytic) - {"termination_penalty"}),
        termination_reasons=model.reasons,
        policy_updates="bptt_through_world_model_upstream_a1",
        value_targets="imagined_lambda_returns",
        physics_parity="NOT_PROVEN",
        hardware_qualification="NOT_PROVEN",
        checkpoint_resume="weights_and_optimizers; replay, environment and curricula restart",
        logging_schema="supermimic_full_v1",
        # Task schedule lengths were divided by this factor (see schedule_clock.json).
        schedule_clock_scale=float(sc.get("schedule_clock_scale", 1.0)),
    )
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    state = dict(obs=obs, total=0)
    task_metrics = {}
    noise_std = float(sc.action_noise)

    @torch.no_grad()
    def collect(n):
        rewards, falls, timeouts = [], 0, 0
        for _ in range(n):
            o = state["obs"]
            policy.actor_obs_normalizer.train()
            policy.update_normalization(TensorDict(o, batch_size=[env.num_envs]))
            policy.actor_obs_normalizer.eval()
            a = (actor(o["policy"]) + noise_std * torch.randn(env.num_envs, 31, device=env.device)) * active
            new, r, terminated, truncated, extras = env.step(a)
            task_metrics.update({k: v.detach().clone() if torch.is_tensor(v) else v for k, v in extras.get("log", {}).items()})
            record, ctx = tape_record(model, recorder, keep, term_index, env.step_dt, r, terminated, truncated)
            replay.add(record)
            if learner.ctx_keys is None:
                learner.ctx_keys = list(ctx)
            episodes.update(r, record["done"])
            recorder.observe(new)
            state["obs"] = new
            state["total"] += 1
            rewards.append(r.mean())
            falls += int(terminated.sum())
            timeouts += int(truncated.sum())
        return float(torch.stack(rewards).mean()), falls, timeouts

    def save(iteration):
        payload = dict(
            model_state_dict=policy.state_dict(),
            iter=iteration,
            training_backend=manifest["algorithm"],
            training_manifest=manifest,
            task_sha256=identity,
            world_model=learner.wm.state_dict(),
            critic=learner.critic.state_dict() if learner.critic is not None else None,
            critic_target=learner.critic_target.state_dict() if learner.critic_target is not None else None,
            critic_norm=learner.critic_norm.state_dict() if learner.critic_norm is not None else None,
            opt_wm=learner.opt_wm.state_dict(),
            opt_pi=learner.opt_pi.state_dict(),
            opt_v=learner.opt_v.state_dict() if learner.opt_v is not None else None,
            sim_steps=state["total"],
            env_step_counter=int(env.common_step_counter),
            resume_fidelity="weights_only_no_environment_or_replay_state",
            task_config=OmegaConf.to_container(cfg.task, resolve=True),
            actor_config=train_cfg["policy"],
        )
        path = out / f"model_{iteration}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        if wandb_writer is not None:
            wandb_writer.checkpoint_saved(iteration, path)

    wm_keys = {"s_cmd", "s_end", "push", "qexec", "qexec_prev", "ctx/cmd_delay", "done"}
    wandb_writer = None
    status, error = "failed", None
    started = time.perf_counter()
    (out / "training_status.json").write_text(json.dumps(dict(status="running", pid=os.getpid()), indent=2))
    try:
        if bool(sc.get("wandb", {}).get("enabled", False)):
            import numpy as np

            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[torch.device(env.device).index]):
                    wandb_writer = WandbLogger(out, OmegaConf.to_container(cfg, resolve=True), manifest)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        collect(int(sc.warmup_steps))
        decay = int(sc.lr_decay_iters or cfg.max_iterations)
        accumulator = 0.0
        H = int(sc.horizon)
        with (out / "metrics.jsonl").open("w") as log:
            for it in range(start_iteration, int(cfg.max_iterations)):
                lr = cosine_lr(float(sc.learning_rate), it, decay)
                learner.set_learning_rate(lr)
                learner.iteration = it
                torch.cuda.synchronize()
                start = time.perf_counter()
                accumulator += 1.0 / float(sc.utd)
                n = int(accumulator)
                accumulator -= n
                reward, falls, timeouts = collect(n)
                torch.cuda.synchronize()
                collected = time.perf_counter()
                wm_rows = [
                    learner.model_update(replay.sample(int(sc.batch_size), int(sc.wm_window), keys=wm_keys))
                    for _ in range(int(sc.wm_batches))
                ]
                metrics = {k: sum(r[k] for r in wm_rows) / len(wm_rows) for k in wm_rows[0]}
                torch.cuda.synchronize()
                wm_done = time.perf_counter()
                if it % int(sc.log_interval) == 0:
                    metrics.update(learner.validate(replay.sample(int(sc.batch_size), int(sc.wm_window), True, keys=wm_keys)))
                    metrics.update(learner.consistency(replay.sample(256, int(sc.wm_window))))
                imagined = it >= int(sc.wm_only_iterations)
                if imagined:
                    sample = lambda: replay.sample(  # noqa: E731
                        int(sc.policy_batch), H + 1, strike_fraction=float(sc.get("strike_window_fraction", 0.0)),
                        dt=env.step_dt,
                    )
                    batch = [sample() for _ in range(max(1, int(sc.get("policy_accumulation", 1))))]
                    if bool(sc.get("debug_compare_gradients", False)) and it in (int(sc.wm_only_iterations), int(sc.wm_only_iterations) + 3):
                        learner.debug_compare_gradients(batch[0])
                    if bool(sc.get("debug_anomaly", False)):
                        with torch.autograd.detect_anomaly(check_nan=True):
                            metrics.update(learner.policy_update(batch))
                    else:
                        metrics.update(learner.policy_update(batch))
                torch.cuda.synchronize()
                end = time.perf_counter()
                if imagined:
                    # Return cached imagination activations to the driver before
                    # Warp collects again (it allocates outside PyTorch's pool).
                    torch.cuda.empty_cache()
                metrics.update(
                    iteration=it,
                    sim_vector_steps=state["total"],
                    sim_transitions=state["total"] * env.num_envs,
                    real_mean_reward=reward,
                    real_terminations=falls,
                    collection_seconds=collected - start,
                    learning_seconds=end - collected,
                    iteration_seconds=end - start,
                )
                metrics.update(episodes.scalars())
                metrics.update({
                    "Progress/completed_iterations": it + 1,
                    "Progress/actor_updates": max(0, it - int(sc.wm_only_iterations) + 1),
                    "Progress/actor_warmup": int(not imagined),
                    "Replay/fill_fraction": replay.size / replay.T,
                    "Replay/gib": replay.bytes() / 2**30,
                    "Safety/real_timeouts": timeouts,
                    "Safety/termination_fraction": falls / max(1, n * env.num_envs),
                    "Policy/learning_rate": lr,
                    "Policy/exploration_noise_std": noise_std,
                    "Performance/world_model_seconds": wm_done - collected,
                    "Performance/policy_seconds": end - wm_done,
                    "Performance/elapsed_seconds": end - started,
                    "Performance/real_samples_per_second": n * env.num_envs / (end - start),
                })
                if it % int(sc.log_interval) == 0:
                    metrics.update(scalar_snapshot({"task/" + k: v for k, v in task_metrics.items()}))
                    metrics.update(task_snapshot(env))
                    free, total = torch.cuda.mem_get_info(env.device)
                    metrics.update({
                        "Performance/device_memory_used_gib": (total - free) / 2**30,
                        "Performance/torch_peak_allocated_gib": torch.cuda.max_memory_allocated(env.device) / 2**30,
                        "Performance/torch_reserved_gib": torch.cuda.memory_reserved(env.device) / 2**30,
                    })
                    corr = replay.data["correction"][:replay.size]
                    metrics["Replay/correction_abs_mean"] = float(corr.abs().mean())
                    bad = [k for k, v in metrics.items() if isinstance(v, float) and not math.isfinite(v)]
                    if bad:
                        print(f"[supermimic] non-finite metrics at {it}: {bad}", flush=True)
                        metrics.update({k: None for k in bad})
                    metrics["Logging/nonfinite_metric_count"] = len(bad)
                    log.write(json.dumps(metrics, allow_nan=False) + "\n")
                    log.flush()
                    for k, v in metrics.items():
                        if v is not None:
                            writer.add_scalar(k, v, it)
                    if wandb_writer is not None:
                        wandb_writer.log(metrics)
                    print("[supermimic] " + json.dumps({k: v for k, v in metrics.items() if "/" not in k}), flush=True)
                if (it + 1) % int(sc.save_interval) == 0:
                    save(it + 1)
            save(int(cfg.max_iterations))
            (out / "completed.json").write_text(json.dumps(dict(
                status="TRAINING_LOOP_COMPLETED", iterations=int(cfg.max_iterations),
                sim_vector_steps=state["total"], formal_policy_evaluation="NOT_RUN",
            ), indent=2))
            status = "completed"
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = dict(type=type(exc).__name__, message=str(exc)[:2000])
        raise
    finally:
        recorder.close()
        writer.close()
        (out / "training_status.json").write_text(json.dumps(dict(
            status=status, pid=os.getpid(), sim_vector_steps=state["total"], error=error,
        ), indent=2))
        if wandb_writer is not None:
            if error is not None:
                wandb_writer.run.summary["Training/error_type"] = error["type"]
            wandb_writer.run.save(str(out / "training_status.json"), base_path=str(out), policy="now")
            wandb_writer.finish(status)
