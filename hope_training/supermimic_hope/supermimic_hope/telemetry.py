"""Read-only task snapshots and episode bookkeeping; no reward/state changes."""

from collections import deque
import math
import torch
from .logging import task_metric_mapper


def scalar_snapshot(values):
    """One device-to-host transfer for scalar tensors; missing data stays null."""
    result, keys, tensors = {}, [], []
    for key, value in values.items():
        if torch.is_tensor(value):
            if value.numel():
                keys.append(key)
                tensors.append(value.detach().float().mean())
        elif isinstance(value, (int, float)):
            result[key] = float(value) if math.isfinite(value) else None
    if tensors:
        for key, value in zip(keys, torch.stack(tensors).cpu().tolist()):
            result[key] = value if math.isfinite(value) else None
    return result


@torch.no_grad()
def task_snapshot(env):
    from whole_body_tracking.utils.stroke_curriculum import scalars as stroke_scalars
    from whole_body_tracking.utils.full_body_hold import phase_summaries

    values = {}
    mapper = task_metric_mapper()
    for term_name in env.command_manager.active_terms:
        term = env.command_manager.get_term(term_name)
        if term_name == "racket_target":
            values.update(stroke_scalars(term))
        for key, value in term.metrics.items():
            tag = f"Live/{term_name}/{key}"
            if mapper(tag) is not None:
                values[tag] = value
        if "full_body_phase" in term.metrics:
            for key, value in phase_summaries(term.metrics).items():
                tag = f"Live/{term_name}/{key}"
                # Keep every phase summary even if the legacy dashboard has
                # no alias; those are important for post-strike recovery.
                if mapper(tag) is not None:
                    values[tag] = value
                else:
                    values["HOLD/detail_" + key] = value
    action_term = env.action_manager.get_term("joint_pos")
    for name, field in (("state", "_lat_state_delay"), ("command", "_lat_cmd_delay")):
        delay = getattr(action_term, field, None)
        if delay is not None:
            values[f"Robustness/{name}_latency_mean_ms"] = (
                delay.float() * env.physics_dt * 1000
            )
            values[f"Robustness/{name}_latency_zero_fraction"] = (delay == 0).float()
    safety = getattr(action_term, "safety_instrumentation_scalars", None)
    if callable(safety):
        values.update(
            {f"Instrumentation/qdes_safety/{k}": v for k, v in safety().items()}
        )
    for i, name in enumerate(env.reward_manager.active_terms):
        values[f"RewardStep/{name}"] = env.reward_manager._step_reward[:, i]
    tm = env.termination_manager
    values["Safety/live_terminated_fraction"] = tm.terminated.float()
    values["Safety/live_timeout_fraction"] = tm.time_outs.float()
    for name in tm.active_terms:
        values[f"Safety/live_{name}"] = tm.get_term(name).float()
    action, previous = env.action_manager.action, env.action_manager.prev_action
    values["Action/raw_action_abs_mean"] = action.abs()
    values["Action/raw_action_abs_max"] = action.abs().amax()
    values["Action/raw_action_delta_abs_mean"] = (action - previous).abs()
    values["Action/raw_action_delta_abs_max"] = (action - previous).abs().amax()
    values["Overview/live_episode_age_seconds"] = (
        env.episode_length_buf.float() * env.step_dt
    )
    return scalar_snapshot(values)


class EpisodeStats:
    def __init__(self, n, device, dt, capacity=1000):
        self.returns = torch.zeros(n, device=device)
        self.lengths = torch.zeros(n, device=device)
        self.recent = deque(maxlen=capacity)
        self.total = 0
        self.dt = dt

    @torch.no_grad()
    def update(self, reward, done):
        self.returns += reward
        self.lengths += 1
        # Empty selection is valid and does not fabricate completed episodes.
        completed = (
            torch.stack((self.returns[done], self.lengths[done]), -1).cpu().tolist()
        )
        self.recent.extend(completed)
        self.total += len(completed)
        self.returns[done] = 0
        self.lengths[done] = 0

    def scalars(self):
        values = {
            "Overview/completed_episodes": self.total,
            "Overview/recent_episode_count": len(self.recent),
        }
        if self.recent:
            values["Overview/episode_return_mean"] = sum(
                x[0] for x in self.recent
            ) / len(self.recent)
            values["Overview/episode_seconds_mean"] = (
                self.dt * sum(x[1] for x in self.recent) / len(self.recent)
            )
        return values
