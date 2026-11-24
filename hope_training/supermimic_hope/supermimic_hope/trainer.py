"""PPO on real Warp rollouts, plus world-model fitting and guarded imagined steps.

The real-data actor/critic update is the task's own ProjectionPPO (same policy
class, hyperparameters, sensor-noise and projection auxiliaries as
``scripts/train.py``). SuperMimic adds a learned world model and a few
first-order actor steps through it per update, bootstrapped by the PPO critic
and stopped by a KL budget, so model error cannot dominate behavior.
"""

from __future__ import annotations
import hashlib
import json
import os
import random
import time
from types import SimpleNamespace
import torch
from torch import nn
from tensordict import TensorDict
from omegaconf import OmegaConf
from .learning import Learner, Replay
from .logging import WandbLogger
from .telemetry import EpisodeStats, scalar_snapshot, task_snapshot
from .contracts import from_environment, failure_flags, TerminationMargins


def build_ppo(env, obs, cfg):
    """Construct the policy and ProjectionPPO exactly as the PPO runner does."""
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs
    from whole_body_tracking.utils.bounded_actor_critic import BoundedStdActorCritic
    from whole_body_tracking.utils.projection_ppo import ProjectionPPO

    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    train_cfg = RslRlOnPolicyRunnerCfg(
        **runner_kwargs(algo, str(cfg.task.experiment_name))
    ).to_dict()
    policy_cfg, alg_cfg = dict(train_cfg["policy"]), dict(train_cfg["algorithm"])
    if policy_cfg.pop("class_name") != "BoundedStdActorCritic":
        raise ValueError("SuperMimic expects the bounded-std HOPE actor-critic")
    if alg_cfg.pop("class_name") != "ProjectionPPO":
        raise ValueError("SuperMimic expects the ProjectionPPO algorithm")
    if alg_cfg.get("rnd_cfg") is not None or alg_cfg.get("symmetry_cfg") is not None:
        raise ValueError("RND/symmetry are outside the HOPE PPO contract")
    term = env.action_manager.get_term("joint_pos")
    policy_cfg["entropy_action_transform"] = (
        "tanh"
        if getattr(term, "_action_contract", None) == "feasible_qdes_v3"
        else "identity"
    )
    policy_cfg["entropy_excluded_action_indices"] = (
        term._passive_action_cols.detach().cpu().tolist()
    )
    for key in ("actor_obs_normalization", "critic_obs_normalization"):
        if policy_cfg.get(key) is None:
            policy_cfg[key] = bool(train_cfg["empirical_normalization"])
    policy = BoundedStdActorCritic(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 31, **policy_cfg
    ).to(env.device)
    alg = ProjectionPPO(policy, device=env.device, **alg_cfg)
    steps = int(cfg.supermimic.rollout_steps or train_cfg["num_steps_per_env"])
    alg.init_storage("rl", env.num_envs, steps, obs, [31])
    ids = torch.arange(env.num_envs, device=env.device)
    offset = term._decoder_parameter_rows(term._offset, ids, "offset")
    scale = term._decoder_parameter_rows(term._scale, ids, "scale")
    bound_a = (term._safe_lo - offset) / scale
    bound_b = (term._safe_hi - offset) / scale
    lower, upper = torch.minimum(bound_a, bound_b), torch.maximum(bound_a, bound_b)
    active = torch.ones(31, dtype=torch.bool, device=env.device)
    active[term._passive_action_cols] = False
    alg.bind_executable_action_projection(lower, upper, active)
    alg.bind_settled_refinement(env)
    return policy, alg, steps, lower, upper, active.float(), train_cfg


def train(env, obs, cfg, out):
    from torch.utils.tensorboard import SummaryWriter
    from .analytic.noise import from_config as sensor_noise

    sc = cfg.supermimic
    if (
        min(sc.horizon, sc.wm_window) > int(sc.bptt_replay_steps)
        or int(sc.bptt_replay_steps) > int(sc.buffer_steps)
    ):
        raise ValueError(
            "Replay must accommodate the model window and imagination horizon"
        )
    if (
        min(
            sc.wm_batches,
            sc.batch_size,
            sc.history,
            sc.horizon,
            sc.wm_window,
            sc.save_interval,
            sc.log_interval,
        )
        < 1
        or int(sc.bptt_steps) < 0
        or float(sc.bptt_max_kl) <= 0
    ):
        raise ValueError(
            "Require positive window, batch and interval sizes and a KL budget"
        )
    torch.manual_seed(int(cfg.seed))
    # Same initialization as the PPO runner: desynchronized episode clocks.
    env.episode_length_buf = torch.randint_like(
        env.episode_length_buf, high=int(env.max_episode_length)
    )
    obs = TensorDict(env.observation_manager.compute(), batch_size=[env.num_envs])
    policy, alg, rollout_steps, lower, upper, active, train_cfg = build_ppo(
        env, obs, cfg
    )
    policy.train()
    actor = nn.Sequential(policy.actor_obs_normalizer, policy.actor)
    noise_penalty = sensor_noise(actor, cfg.supermimic.sensor_noise)

    contract, failure_reasons, termination_costs, observation_fields = from_environment(
        env
    )
    margin_observer = TerminationMargins(env, failure_reasons)

    def pack(o):
        return (
            torch.cat((o["policy"], o["critic"], margin_observer.read()), -1)
            .detach()
            .clone()
        )

    state = pack(obs)
    dim = state.shape[-1]
    critic_start = obs["policy"].shape[-1]
    critic_stop = critic_start + obs["critic"].shape[-1]

    def value(s):
        x = s[..., critic_start:critic_stop]
        return policy.critic(policy.critic_obs_normalizer(x)).squeeze(-1)

    cost_tensor = torch.tensor(termination_costs, device=env.device)
    termination_reward_index = env.reward_manager.active_terms.index(
        "termination_penalty"
    )
    learner = Learner(
        actor,
        dim,
        sc,
        lower,
        upper,
        active,
        noise_penalty,
        contract,
        termination_costs,
        value=value,
        value_modules=(policy.critic,),
        action_std=lambda: policy.std,
    )
    replay = Replay(
        int(sc.buffer_steps),
        env.num_envs,
        dim,
        31,
        env.device,
        float(sc.validation_env_fraction),
        failure_dim=len(failure_reasons),
    )
    total_steps = 0
    last_checkpoint = -1
    task_metrics = {}
    episodes = EpisodeStats(env.num_envs, env.device, env.step_dt)
    writer = SummaryWriter(str(out / "tensorboard"))
    identity = hashlib.sha256(
        OmegaConf.to_yaml(cfg.task, resolve=True).encode()
    ).hexdigest()
    manifest = dict(
        algorithm="supermimic_hope_ppo_anchored_v3",
        upstream="a1508d42c9d1699a52e8233ab16d08c488441ab9",
        physics_backend=str(sc.backend),
        task_sha256=identity,
        actor_dim=110,
        action_dim=31,
        wm_state_dim=dim,
        wm_history=int(sc.history),
        rollout_steps_per_iteration=rollout_steps,
        heads="exact_action_and_packet_context; learned_residual_reward_and_failure_probabilities",
        observation_contract=contract.receipt(),
        observation_fields=observation_fields,
        failure_reasons=failure_reasons,
        termination_costs_per_step=termination_costs,
        failure_model="pre_reset_physical_margins_with_task_thresholds",
        failure_margin_temperature=float(sc.failure_margin_temperature),
        value_targets="ppo_critic_gae_on_real_rollouts",
        policy_updates="projection_ppo_on_real_rollouts_then_kl_bounded_model_bptt",
        real_task_unchanged=True,
        physics_parity="NOT_PROVEN",
        hardware_qualification="NOT_PROVEN",
        checkpoint_resume="not_supported; model_state_dict is the PPO actor-critic",
        logging_schema="supermimic_full_v1",
    )
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    command = env.command_manager.get_term("racket_target")
    motion = env.command_manager.get_term("motion")

    @torch.no_grad()
    def collect():
        nonlocal obs, state, total_steps
        rewards = []
        falls = 0
        timeouts = 0
        feedback_error = torch.zeros((), device=env.device)
        for _ in range(rollout_steps):
            # PPO samples raw actions with passive heads zeroed; the task's
            # action term owns the safe q_des clamp, exactly as in train.py.
            a = alg.act(obs)
            row = command._venue_joint_row.clone()
            hold = motion.in_hold.clone()
            new, r, terminated, truncated, extras = env.step(a)
            task_metrics.update(
                {
                    k: v.detach().clone() if torch.is_tensor(v) else v
                    for k, v in extras.get("log", {}).items()
                }
            )
            done = terminated | truncated
            extras["time_outs"] = truncated
            new = TensorDict(new, batch_size=[env.num_envs])
            alg.process_env_step(new, r, done.long(), extras)
            s1 = pack(new)
            episodes.update(r, done)
            cut = (
                done
                | motion.just_resampled
                | (command._venue_joint_row != row)
                | (motion.in_hold != hold)
            )
            if not (torch.isfinite(s1).all() and torch.isfinite(r).all()):
                raise FloatingPointError("Simulator supplied nonfinite training data")
            # The model copies the sent action into its predicted feedback;
            # verify that the real environment reports exactly that action.
            feedback_error = torch.maximum(
                feedback_error,
                (
                    (s1[:, contract.action_indices] - a[:, contract.action_sources])
                    * (~done)[:, None]
                )
                .abs()
                .amax(),
            )
            failures = failure_flags(env, failure_reasons)
            pre_reset_margin = margin_observer.last
            margin_flags = (pre_reset_margin > 0) & ~truncated[:, None]
            if not torch.equal(margin_flags, failures):
                raise ValueError(
                    "Physical failure margins differ from task terminations"
                )
            reconstructed_cost = failures.float() @ cost_tensor
            actual_cost = (
                env.reward_manager._step_reward[:, termination_reward_index]
                * env.step_dt
            )
            if not torch.allclose(
                reconstructed_cost, actual_cost, atol=1e-5, rtol=1e-5
            ):
                raise ValueError(
                    "Termination cost contract differs from the real task reward"
                )
            replay.add(
                s=state,
                s1=s1,
                a=a,
                r=r,
                cut=cut,
                terminal=terminated,
                done=done,
                failure=failures,
                failure_margin=pre_reset_margin,
            )
            state, obs = s1, new
            total_steps += 1
            rewards.append(r.mean())
            falls += int(terminated.sum())
            timeouts += int(truncated.sum())
        alg.compute_returns(obs)
        return (
            float(torch.stack(rewards).mean()),
            falls,
            timeouts,
            float(feedback_error),
        )

    def save(iteration):
        nonlocal last_checkpoint
        payload = dict(
            model_state_dict=policy.state_dict(),
            optimizer_state_dict=alg.optimizer.state_dict(),
            iter=iteration,
            training_backend=manifest["algorithm"],
            training_manifest=manifest,
            task_sha256=identity,
            world_models=learner.models.state_dict(),
            opt_wm=learner.opt_wm.state_dict(),
            opt_bptt=learner.opt_pi.state_dict(),
            sim_steps=total_steps,
            resume_fidelity="weights_only_no_environment_or_replay_state",
            task_config=OmegaConf.to_container(cfg.task, resolve=True),
            actor_config=train_cfg["policy"],
        )
        path = out / f"model_{iteration}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        last_checkpoint = iteration
        if wandb_writer is not None:
            wandb_writer.checkpoint_saved(iteration, path)

    def sample(k, validation=False, recent=None):
        return replay.sample(
            int(sc.batch_size),
            int(k),
            int(sc.history),
            validation,
            recent_steps=recent,
        )

    wandb_writer = None
    status = "failed"
    error = None
    started = time.perf_counter()
    previous_wall_iteration = None
    (out / "training_status.json").write_text(
        json.dumps(dict(status="running", pid=os.getpid()), indent=2)
    )
    try:
        if bool(sc.get("wandb", {}).get("enabled", False)):
            # SDK initialization may touch Python/NumPy/Torch RNGs. Logging
            # must not change the sampled task or the optimizer trajectory.
            import numpy as np

            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[torch.device(env.device).index]):
                    wandb_writer = WandbLogger(
                        out, OmegaConf.to_container(cfg, resolve=True), manifest
                    )
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        with (out / "metrics.jsonl").open("w") as log:
            for it in range(int(cfg.max_iterations)):
                torch.cuda.synchronize()
                start = time.perf_counter()
                reward, falls, timeouts, feedback_error = collect()
                torch.cuda.synchronize()
                collected = time.perf_counter()
                ppo = alg.update()
                torch.cuda.synchronize()
                ppo_done = time.perf_counter()
                losses, model_diagnostics = [], []
                for _ in range(int(sc.wm_batches)):
                    losses.append(learner.model_update(sample(sc.wm_window)))
                    model_diagnostics.append(learner.last_model_metrics)
                metrics = learner.validate(sample(sc.wm_window, True))
                for key in model_diagnostics[0]:
                    metrics[key] = sum(m[key] for m in model_diagnostics) / len(
                        model_diagnostics
                    )
                torch.cuda.synchronize()
                wm_done = time.perf_counter()
                imagined = (
                    it >= int(sc.wm_only_iterations)
                    and int(sc.bptt_steps) > 0
                    and float(sc.model_policy_weight) > 0
                )
                if imagined:
                    metrics.update(
                        learner.policy_update(
                            lambda: sample(sc.horizon, recent=int(sc.bptt_replay_steps))
                        )
                    )
                torch.cuda.synchronize()
                end = time.perf_counter()
                metrics.update({f"PPO/{k}": v for k, v in ppo.items()})
                metrics.update(
                    iteration=it,
                    sim_vector_steps=total_steps,
                    sim_transitions=total_steps * env.num_envs,
                    real_mean_reward=reward,
                    real_terminations=falls,
                    wm_loss=sum(losses) / len(losses),
                    collection_seconds=collected - start,
                    learning_seconds=end - collected,
                    iteration_seconds=end - start,
                )
                metrics.update(episodes.scalars())
                std = policy.std.detach()[active.bool()]
                metrics.update(
                    {
                        "Progress/completed_iterations": it + 1,
                        "Progress/rollout_steps_per_iteration": rollout_steps,
                        "Progress/world_model_updates": (it + 1) * int(sc.wm_batches),
                        "Progress/imagined_actor_updates_active": int(imagined),
                        "Checkpoint/last_saved_iteration": last_checkpoint,
                        "Checkpoint/next_periodic_save": (
                            (it // int(sc.save_interval)) + 1
                        )
                        * int(sc.save_interval),
                        "Replay/fill_fraction": replay.size / replay.T,
                        "Replay/real_action_feedback_max_abs_error": feedback_error,
                        "Replay/transitions": replay.size * env.num_envs,
                        "Replay/training_environments": env.num_envs - replay.val_n,
                        "Replay/validation_environments": replay.val_n,
                        "Safety/real_timeouts": timeouts,
                        "Safety/termination_fraction": falls
                        / (rollout_steps * env.num_envs),
                        "Safety/timeout_fraction": timeouts
                        / (rollout_steps * env.num_envs),
                        "Policy/noise_std_mean": float(std.mean()),
                        "Policy/noise_std_min": float(std.min()),
                        "Policy/sensor_noise_coefficient": float(cfg.supermimic.sensor_noise.coef),
                        "Policy/learning_rate": float(alg.learning_rate),
                        "Policy/bptt_learning_rate": learner.opt_pi.param_groups[0][
                            "lr"
                        ],
                        "WorldModel/learning_rate": learner.opt_wm.param_groups[0][
                            "lr"
                        ],
                        "Performance/ppo_seconds": ppo_done - collected,
                        "Performance/world_model_seconds": wm_done - ppo_done,
                        "Performance/imagination_seconds": end - wm_done,
                        "Performance/real_samples_per_second": rollout_steps
                        * env.num_envs
                        / (end - start),
                        "Performance/collection_samples_per_second": rollout_steps
                        * env.num_envs
                        / (collected - start),
                        "Performance/elapsed_seconds": end - started,
                    }
                )
                if previous_wall_iteration is not None:
                    metrics["Performance/previous_wall_iteration_seconds"] = (
                        previous_wall_iteration
                    )
                if it % int(sc.log_interval) == 0:
                    metrics.update(
                        scalar_snapshot(
                            {"task/" + k: v for k, v in task_metrics.items()}
                        )
                    )
                    metrics.update(task_snapshot(env))
                    free, total = torch.cuda.mem_get_info(env.device)
                    metrics.update(
                        {
                            "Performance/device_memory_used_gib": (total - free)
                            / 2**30,
                            "Performance/torch_allocated_gib": torch.cuda.memory_allocated(
                                env.device
                            )
                            / 2**30,
                            "Performance/torch_reserved_gib": torch.cuda.memory_reserved(
                                env.device
                            )
                            / 2**30,
                            "Performance/torch_peak_allocated_gib": torch.cuda.max_memory_allocated(
                                env.device
                            )
                            / 2**30,
                        }
                    )
                log.write(json.dumps(metrics, allow_nan=False) + "\n")
                log.flush()
                for k, v in metrics.items():
                    if v is not None:
                        writer.add_scalar(k, v, it)
                if wandb_writer is not None:
                    wandb_writer.log(metrics)
                if it % int(sc.log_interval) == 0:
                    print(
                        "[supermimic] "
                        + json.dumps(
                            {k: v for k, v in metrics.items() if "/" not in k}
                        ),
                        flush=True,
                    )
                if (it + 1) % int(sc.save_interval) == 0:
                    save(it + 1)
                previous_wall_iteration = time.perf_counter() - start
            save(int(cfg.max_iterations))
            (out / "completed.json").write_text(
                json.dumps(
                    dict(
                        status="TRAINING_LOOP_COMPLETED",
                        iterations=int(cfg.max_iterations),
                        sim_vector_steps=total_steps,
                        formal_policy_evaluation="NOT_RUN",
                    ),
                    indent=2,
                )
            )
            status = "completed"
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = dict(type=type(exc).__name__, message=str(exc)[:2000])
        raise
    finally:
        margin_observer.close()
        writer.close()
        (out / "training_status.json").write_text(
            json.dumps(
                dict(
                    status=status,
                    pid=os.getpid(),
                    sim_vector_steps=total_steps,
                    error=error,
                ),
                indent=2,
            )
        )
        if wandb_writer is not None:
            if error is not None:
                wandb_writer.run.summary["Training/error_type"] = error["type"]
            wandb_writer.run.save(
                str(out / "training_status.json"), base_path=str(out), policy="now"
            )
            wandb_writer.finish(status)
