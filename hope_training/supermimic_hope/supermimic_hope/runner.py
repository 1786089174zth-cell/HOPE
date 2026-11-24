"""Training entry and bounded simulator probes (no optimizer in probe mode)."""

from pathlib import Path
import json
import time
import torch
from omegaconf import OmegaConf


def load_actor(env, obs, cfg, path):
    """Deterministic actor of a ``train.py``/SuperMimic checkpoint (same 110-D ABI)."""
    from tensordict import TensorDict
    from .trainer_analytic import build_policy

    policy = build_policy(env, TensorDict(obs, batch_size=[env.num_envs]), cfg)[0]
    state = torch.load(path, map_location=env.device, weights_only=False)["model_state_dict"]
    policy.load_state_dict(state)
    policy.eval()
    return lambda o: policy.act_inference(TensorDict(o, batch_size=[env.num_envs]))


def run(env, env_cfg, backend, cfg):
    from isaaclab.utils.io import dump_yaml
    from datetime import datetime

    out = Path(
        cfg.supermimic.output_dir
        or (
            "logs/supermimic/"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + str(cfg.run_name)
        )
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "params").mkdir()
    OmegaConf.save(cfg, out / "params/task_resolved.yaml")
    dump_yaml(str(out / "params/env.yaml"), env_cfg)
    if backend:
        (out / "physics_receipt.json").write_text(json.dumps(backend.receipt, indent=2))
    (out / "schedule_clock.json").write_text(json.dumps(dict(
        scale=float(cfg.supermimic.get("schedule_clock_scale", 1.0)),
        steps_per_iteration=1.0 / float(cfg.supermimic.utd),
        scaled=getattr(env, "supermimic_schedule_clock", {}),
    ), indent=2))
    obs, _ = env.reset()
    if cfg.supermimic.get("evaluate_checkpoints"):
        from .analytic.evaluate import run as evaluate

        evaluate(env, obs, cfg, out, backend, list(cfg.supermimic.evaluate_checkpoints),
                 steps=int(cfg.supermimic.get("evaluate_steps", 1500)),
                 noise=float(cfg.supermimic.get("evaluate_noise", 0.1)))
        return
    if cfg.supermimic.get("diagnose_checkpoint"):
        from .analytic.diagnose import run as diagnose

        sections = cfg.supermimic.get("diagnose_sections")
        diagnose(env, obs, cfg, out, backend, cfg.supermimic.diagnose_checkpoint,
                 steps=int(cfg.supermimic.get("diagnose_steps", 160)),
                 **({"sections": tuple(sections)} if sections else {}))
        return
    if int(cfg.supermimic.get("analytic_check_steps", 0)):
        from .analytic.check import run as check

        policy = None
        path = cfg.supermimic.get("analytic_check_policy")
        if path:
            policy = load_actor(env, obs, cfg, path)
        check(env, backend, obs, int(cfg.supermimic.analytic_check_steps), out, policy,
              noise=float(cfg.supermimic.get("analytic_check_noise", 0.1)))
        return
    if int(cfg.supermimic.probe_steps):
        times, resets = [], 0
        families = set()
        snapshot_checks = []
        action = torch.zeros(env.num_envs, 31, device=env.device)
        for i in range(int(cfg.supermimic.probe_steps)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            obs, rew, term, trunc, _ = env.step(action)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            assert all(torch.isfinite(v).all() for v in obs.values())
            assert torch.isfinite(rew).all()
            robot = env.scene["robot"]
            assert torch.isfinite(robot.data.body_pos_w).all()
            assert torch.isfinite(
                env.scene.sensors["contact_forces"].data.net_forces_w
            ).all()
            if backend:
                assert (
                    robot.data.applied_torque.abs()
                    <= robot.data.joint_effort_limits + 1.0e-3
                ).all()
                if i in (0, int(cfg.supermimic.probe_steps) - 1):
                    snapshot_checks.append(backend.validate_snapshot())
            resets += int((term | trunc).sum())
            cmd = env.command_manager.get_term("racket_target")
            motion = env.command_manager.get_term("motion")
            families.update((motion.clip_id + 2 * cmd._stroke_step).tolist())
            if i == int(cfg.supermimic.probe_steps) // 2:
                env._reset_idx(torch.arange(min(env.num_envs, 2), device=env.device))
                env.sim.forward()
        result = dict(
            status="PROBE_PASS",
            optimizer_updates=0,
            steps=len(times),
            environments=env.num_envs,
            resets=resets,
            backend=cfg.supermimic.backend,
            mean_step_seconds=sum(times[1:] or times) / len(times[1:] or times),
            actor_dim=obs["policy"].shape[-1],
            critic_dim=obs["critic"].shape[-1],
            explicit_resets=min(env.num_envs, 2),
            observed_families=sorted(families),
            snapshot_checks=snapshot_checks,
        )
        (out / "probe.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    algorithm = str(cfg.supermimic.get("algorithm", "analytic"))
    if algorithm == "analytic":
        from .trainer_analytic import train

        train(env, obs, cfg, out, backend)
    elif algorithm == "ppo_anchored":
        from .trainer import train

        merged = OmegaConf.merge(cfg.supermimic, cfg.supermimic.ppo_anchored)
        OmegaConf.set_struct(cfg, False)
        cfg.supermimic = merged
        train(env, obs, cfg, out)
    else:
        raise ValueError(f"Unknown supermimic.algorithm {algorithm!r}")
