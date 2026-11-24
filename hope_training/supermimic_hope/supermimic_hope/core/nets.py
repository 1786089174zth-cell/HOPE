"""Actor and critic networks.

The actor is the deployed policy: ``frames`` consecutive 110-D hitter_pure
observations, oldest first, flattened to one ``[B, 110 * frames]`` input (the
runner's ``obs_frame_history`` contract), mapped to 31 raw actions.
"""

from __future__ import annotations

import torch
from torch import nn

from .dynamics import RunningNorm, mlp

OBS = 110
ACT = 31


class Actor(nn.Module):
    def __init__(self, frames=10, hidden=(2048, 1024, 512, 256), active=None, amp=False):
        super().__init__()
        self.frames, self.amp = frames, amp
        self.norm = RunningNorm(OBS)  # one normalizer shared by every frame of the history
        self.net = mlp(OBS * frames, hidden, ACT)
        with torch.no_grad():
            # Start from near-zero actions: q_des at the default pose.
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        self.register_buffer("active", torch.ones(ACT) if active is None else active.clone())

    def forward(self, history):
        """``history`` (B, frames, 110), oldest first -> raw actions (B, 31); passive joints 0."""
        x = self.norm(history).flatten(1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            out = self.net(x)
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
