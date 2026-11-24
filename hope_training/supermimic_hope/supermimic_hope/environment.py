"""Construct the unmodified resolved HOPE task, then attach Warp physics."""

from pathlib import Path
import sys

WBT = Path(__file__).resolve().parents[2] / "whole_body_tracking"

# Task settings counted in environment control steps that pace LEARNING
# (reward ramps, DR ramps, admitted question difficulty). They were calibrated
# for PPO, which takes 64 environment steps per update; SuperMimic takes
# 1/utd = 2.5 per iteration, so unscaled they finish after the run does.
# Evidence windows (stage check interval, statistics age, dwell counts) and
# physical durations (HOLD lengths, delays) are real time and are not scaled.
_SCHEDULE_SUFFIXES = ("_ramp_steps", "_start_steps")
_SCHEDULE_NAMES = ("ramp_steps", "venue_tuple_bank_curriculum_steps")


def scale_schedule_clock(task, scale):
    """Divide every learning-schedule length in the task config by ``scale``.

    ``task`` is the mutable OmegaConf task node; returns {dotted key: [old, new]}.
    The environment step counter itself is untouched, so per-step bookkeeping
    (HOLD dwell continuity, evidence windows) keeps its meaning.
    """
    from omegaconf import DictConfig

    scale = float(scale)
    if not scale >= 1.0:
        raise ValueError("supermimic.schedule_clock_scale must be >= 1")
    changed = {}

    def walk(node, prefix):
        for key in list(node.keys()):
            value = node[key]
            name = f"{prefix}{key}"
            if isinstance(value, DictConfig):
                walk(value, name + ".")
            elif str(key) in _SCHEDULE_NAMES or str(key).endswith(_SCHEDULE_SUFFIXES):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"schedule setting {name} is not a step count: {value!r}")
                if value > 0 and scale != 1.0:
                    node[key] = max(1, int(round(value / scale)))
                    changed[name] = [int(value), int(node[key])]

    walk(task, "")
    return changed


def create_env(cfg):
    import gymnasium as gym
    import whole_body_tracking.tasks  # noqa: F401 -- register tasks after Kit startup
    from isaaclab_tasks.utils import parse_env_cfg
    from whole_body_tracking.tasks.tracking.actor_observation_contract import (
        validate_actor_observation_contract,
    )

    sys.path.insert(0, str(WBT / "scripts"))
    from train import _apply_task_overrides, _resolve_motion_sources

    env_cfg = parse_env_cfg(
        str(cfg.task.gym_task), device=str(cfg.device), num_envs=int(cfg.num_envs)
    )
    applied = []
    clock = scale_schedule_clock(cfg.task, cfg.supermimic.get("schedule_clock_scale", 1.0))
    if clock:
        print(f"[supermimic] schedule clock / {float(cfg.supermimic.schedule_clock_scale):g}: "
              + ", ".join(f"{k} {a}->{b}" for k, (a, b) in sorted(clock.items())), flush=True)
    _apply_task_overrides(env_cfg, cfg, applied)
    if not cfg.task.racket.virtual_ball:
        raise ValueError("This migration uses the original PPQ004 virtual-ball task")
    if abs(float(env_cfg.sim.dt) - 0.005) > 1.0e-9 or int(env_cfg.decimation) != 4:
        raise ValueError(
            "The current A3 Warp contract requires 5 ms physics and 50 Hz control"
        )
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    env_cfg.commands.motion.motion_file = _resolve_motion_sources(cfg)
    env = gym.make(str(cfg.task.gym_task), cfg=env_cfg).unwrapped
    validate_actor_observation_contract(env, str(cfg.task.actor_obs_contract))
    if cfg.supermimic.backend == "mujoco_warp":
        from .warp_backend import WarpBackend

        backend = WarpBackend(env, capture=bool(cfg.supermimic.capture))
    elif cfg.supermimic.backend == "isaac":
        backend = None  # Explicit reference backend for parity/performance checks.
    else:
        raise ValueError("Unknown physics backend")
    env.supermimic_schedule_clock = clock
    return env, env_cfg, backend
