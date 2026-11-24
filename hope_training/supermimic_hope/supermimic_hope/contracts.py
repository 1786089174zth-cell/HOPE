"""Exact observation fields and task-owned termination costs for imagination."""

from __future__ import annotations

import math
import torch
from torch import nn


class ObservationContract(nn.Module):
    """Copy action feedback; condition on the same packet's recorded command clock.

    Robot-relative targets, proprioception and FK observations remain dynamics
    outputs. No future physical state is copied into imagined observations.
    """

    def __init__(self, action_indices=(), action_sources=(), context_indices=()):
        super().__init__()
        for name, values in (
            ("action_indices", action_indices),
            ("action_sources", action_sources),
            ("context_indices", context_indices),
        ):
            self.register_buffer(name, torch.tensor(values, dtype=torch.long))

    def forward(self, predicted, action, context):
        result = predicted.clone()
        result[..., self.action_indices] = action[..., self.action_sources]
        result[..., self.context_indices] = context[..., self.context_indices].detach()
        return result

    def context(self, batch, step, current):
        # Auto-reset states and new packets are never teacher-forced into a
        # rollout for the preceding packet. Only announced, exogenous fields
        # are read by forward(), and only before a recorded boundary.
        valid = batch["mask"][:, step].bool() & ~batch["cut"][:, step]
        return torch.where(valid[:, None], batch["s1"][:, step], current).detach()

    def receipt(self):
        return {
            name: getattr(self, name).cpu().tolist()
            for name in ("action_indices", "action_sources", "context_indices")
        }


def from_environment(env):
    """Resolve slices from the actual manager layout, not assumed critic offsets."""
    manager = env.observation_manager
    action_indices, action_sources, context_indices, fields = [], [], [], {}
    offset = 0
    for group in ("policy", "critic"):
        names = manager.active_terms[group]
        dimensions = manager.group_obs_term_dim[group]
        for term_index, (name, shape) in enumerate(zip(names, dimensions)):
            width = math.prod(shape)
            indices = list(range(offset, offset + width))
            fields[f"{group}/{name}"] = [offset, offset + width]
            if name == "actions":
                term = manager._group_obs_term_cfgs[group][term_index]
                if width != 31 or term.func.__name__ != "applied_last_action":
                    raise ValueError(
                        "SuperMimic requires the applied_last_action contract"
                    )
                action_indices.extend(indices)
                action_sources.extend(range(31))
            elif name in (
                "command",
                "time_to_strike",
                "episode_time_left",
                "racket_target_vel_w",
                "racket_target_normal_w",
                "swing_type",
            ):
                context_indices.extend(indices)
            offset += width
    if len(action_indices) != 62:
        raise ValueError(
            "Expected action feedback in both actor and critic observations"
        )
    contract = ObservationContract(action_indices, action_sources, context_indices).to(
        env.device
    )

    reward = env.reward_manager.get_term_cfg("termination_penalty")
    function = reward.func if isinstance(reward.func, type) else type(reward.func)
    if function.__name__ != "is_terminated_term":
        raise ValueError("Unsupported termination reward: expected is_terminated_term")
    # Isaac's reward counts every matching reason, not merely any termination.
    reasons = env.termination_manager.find_terms(reward.params["term_keys"])
    if not reasons:
        raise ValueError("Termination reward selected no termination reasons")
    costs = [float(reward.weight) * float(env.step_dt)] * len(reasons)
    if any(x >= 0 for x in costs):
        raise ValueError("Expected negative task-owned terminal costs")
    return contract, reasons, costs, fields


def failure_flags(env, reasons):
    return torch.stack(
        [
            env.termination_manager.get_term(name) & ~env.termination_manager.time_outs
            for name in reasons
        ],
        -1,
    )


class TerminationMargins:
    """Capture signed distances to task failure thresholds before auto-reset.

    Positive means failure. This observer changes no reward or reset behavior.
    """

    def __init__(self, env, reasons):
        from isaaclab.utils.math import quat_apply_inverse
        from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes

        self.env, self.reasons = env, reasons
        self.quat_inverse = quat_apply_inverse
        self.specs = []
        for name in reasons:
            cfg = env.termination_manager.get_term_cfg(name)
            function = cfg.func.__name__
            if function not in (
                "bad_anchor_pos_z_only",
                "bad_anchor_ori",
                "bad_motion_body_pos_z_only",
                "bad_orientation",
                "root_height_below_minimum",
            ):
                raise ValueError(
                    f"Unsupported physical failure margin: {name}/{function}"
                )
            command = (
                env.command_manager.get_term(cfg.params["command_name"])
                if "command_name" in cfg.params
                else None
            )
            bodies = (
                _get_body_indexes(command, cfg.params.get("body_names"))
                if function == "bad_motion_body_pos_z_only"
                else None
            )
            self.specs.append((function, cfg.params, command, bodies))
        self.original_compute = env.termination_manager.compute
        self.last = None

        def observed_compute():
            self.last = self.read().detach().clone()
            return self.original_compute()

        env.termination_manager.compute = observed_compute

    def read(self):
        robot = self.env.scene["robot"].data
        margins = []
        for function, p, command, bodies in self.specs:
            if function == "bad_anchor_pos_z_only":
                score = (
                    command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]
                ).abs() / p["threshold"] - 1
            elif function == "bad_anchor_ori":
                ref = self.quat_inverse(command.anchor_quat_w, robot.GRAVITY_VEC_W)
                actual = self.quat_inverse(
                    command.robot_anchor_quat_w, robot.GRAVITY_VEC_W
                )
                score = (ref[:, 2] - actual[:, 2]).abs() / p["threshold"] - 1
            elif function == "bad_motion_body_pos_z_only":
                score = (
                    command.body_pos_relative_w[:, bodies, 2]
                    - command.robot_body_pos_w[:, bodies, 2]
                ).abs().amax(-1) / p["threshold"] - 1
            elif function == "bad_orientation":
                score = (
                    torch.acos((-robot.projected_gravity_b[:, 2]).clamp(-1, 1))
                    / p["limit_angle"]
                    - 1
                )
            else:
                score = 1 - robot.root_pos_w[:, 2] / p["minimum_height"]
            margins.append(score)
        return torch.stack(margins, -1)

    def close(self):
        self.env.termination_manager.compute = self.original_compute
