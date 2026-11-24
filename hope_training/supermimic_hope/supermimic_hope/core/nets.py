"""Actor and critic networks.

The actor is the deployed policy: ``frames`` consecutive 110-D hitter_pure
observations, oldest first, flattened to one ``[B, 110 * frames]`` input (the
runner's ``obs_frame_history`` contract), mapped to 31 raw actions.
"""

from __future__ import annotations

import torch
from torch import nn

from .. import a3
from .dynamics import RunningNorm, mlp

OBS = 110
ACT = 31


def _action_mlp(inputs, hidden, outputs):
    net = mlp(inputs, hidden, outputs)
    with torch.no_grad():
        # Near-zero initial commands hold the default joint pose.
        net[-1].weight.mul_(0.01)
        net[-1].bias.zero_()
    return net


class DualBodyMLP(nn.Module):
    """Compute the arm intent first, then condition the waist/legs on that same intent."""

    def __init__(self, frames, hidden):
        super().__init__()
        # Head then arms form the 16-D message. Actuator order on the A3 is interleaved.
        arms = [i for i, name in enumerate(a3.JOINTS) if any(k in name for k in ("shoulder", "elbow", "wrist"))]
        upper = [*a3.PASSIVE, *arms]
        lower = [i for i, name in enumerate(a3.JOINTS) if any(k in name for k in ("waist", "hip", "knee", "ankle"))]
        if len(upper) != 16 or len(lower) != 15 or sorted(lower + upper) != list(range(ACT)):
            raise ValueError("Dual actor requires the A3 waist, legs, arms and two passive head joints")
        self.register_buffer("upper_joints", torch.tensor(upper), persistent=False)
        self.register_buffer("motor_order", torch.argsort(torch.tensor(lower + upper)), persistent=False)
        # Per 110-D frame: shared body feedback [0:101], station [101:103],
        # racket position/velocity [103:109], and time to strike [109].
        self.upper = _action_mlp(108 * frames, hidden, len(upper))
        self.lower = _action_mlp(104 * frames + len(upper), hidden, len(lower))

    def forward(self, history, active):
        upper_obs = torch.cat((history[..., :101], history[..., 103:]), dim=-1)
        upper = self.upper(upper_obs.flatten(1)).float() * active.index_select(0, self.upper_joints)
        lower_obs = torch.cat((history[..., :103], history[..., 109:]), dim=-1)
        lower = self.lower(torch.cat((lower_obs.flatten(1), upper), dim=-1))
        return torch.cat((lower.float(), upper), dim=-1).index_select(1, self.motor_order)


class Actor(nn.Module):
    def __init__(self, frames=10, hidden=(2048, 1024, 512, 256), active=None, amp=False, architecture="single"):
        super().__init__()
        self.frames, self.amp = frames, amp
        self.norm = RunningNorm(OBS)  # one normalizer shared by every frame of the history
        self.architecture = architecture
        mask = torch.ones(ACT) if active is None else active.clone()
        if architecture == "dual_body_mlp_v1":
            self.net = DualBodyMLP(frames, hidden)
            mask[list(a3.PASSIVE)] = 0
        elif architecture == "single":
            self.net = _action_mlp(OBS * frames, hidden, ACT)
        else:
            raise ValueError(f"Unknown SuperMimic actor architecture: {architecture!r}")
        self.register_buffer("active", mask)

    def forward(self, history):
        """``history`` (B, frames, 110), oldest first -> raw actions (B, 31); passive joints 0."""
        x = self.norm(history)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            if self.architecture == "dual_body_mlp_v1":
                out = self.net(x, self.active)
            else:
                out = self.net(x.flatten(1))
        return out.float() * self.active

    def flat_forward(self, flat):
        """Deploy entry: ``flat`` (B, 110 * frames)."""
        return self.forward(flat.unflatten(1, (self.frames, OBS)))


class Critic(nn.Module):
    def __init__(self, dim, hidden=(256, 128)):
        super().__init__()
        self.norm = RunningNorm(dim)
        self.net = mlp(dim, hidden, 1)

    def forward(self, x):
        return self.net(self.norm(x)).squeeze(-1)
