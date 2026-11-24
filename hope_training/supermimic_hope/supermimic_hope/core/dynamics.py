"""Differentiable dynamics model (paper section 3.2, upstream ``GenCoordRepresentation``).

Full state ``s`` (75): ``[p(3), quat(4, wxyz), v_w(3), w_b(3), q(31), qd(31)]`` -
MuJoCo's free-joint ``qpos``/``qvel`` (world linear velocity of the root origin,
body-frame angular velocity) with joints in deploy order.

The network sees a heading-free representation ``[z, g_b, v_b, w_b, q, qd]``, the
executed PD targets and the per-episode sim context, and predicts acceleration-
level deltas ``[dv_b, dw_b, dqd]`` in the current root frame. Poses are
integrated analytically, so every prediction is a kinematically consistent state.
"""

from __future__ import annotations

import torch
from torch import nn

from .rotations import integrate_body_rate, quat_to_matrix

NJ = 31
STATE = 13 + 2 * NJ
REP = 10 + 2 * NJ


def split(s):
    return s[..., 0:3], s[..., 3:7], s[..., 7:10], s[..., 10:13], s[..., 13:13 + NJ], s[..., 13 + NJ:]


def pack(p, quat, v, w, q, qd):
    return torch.cat((p, quat, v, w, q, qd), -1)


def represent(s):
    p, quat, v, w, q, qd = split(s)
    R = quat_to_matrix(quat)
    g_b = -R[..., 2, :]  # world down expressed in the root frame
    v_b = (R.transpose(-1, -2) @ v[..., None])[..., 0]
    return torch.cat((p[..., 2:3], g_b, v_b, w, q, qd), -1)


class RunningNorm(nn.Module):
    """Running mean/std updated in place (safe to read from a captured CUDA graph)."""

    def __init__(self, dim, eps=1e-2, until=1.0e8):
        super().__init__()
        self.eps, self.until = eps, until
        # Samples seen, tracked on the host so the stop test never reads the device
        # (a captured update keeps whichever branch was taken when it was recorded).
        self._seen = None
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        if self._seen is None:
            self._seen = float(self.count)
        if self._seen >= self.until:  # statistics are frozen after ``until`` samples (rsl_rl default)
            return
        self._seen += n
        total = self.count + n
        delta = x.mean(0) - self.mean
        self.mean.add_(delta * (n / total))
        self.var.add_((x.var(0, unbiased=False) - self.var + delta.square() * (self.count / total)) * (n / total))
        self.std.copy_(self.var.sqrt())
        self.count.copy_(total)

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


def mlp(inp, hidden, out):
    """``Linear -> LayerNorm -> GELU`` blocks and a linear head (upstream ``build_mlp``)."""
    layers, n = [], inp
    for h in hidden:
        layers += [nn.Linear(n, h), nn.LayerNorm(h), nn.GELU()]
        n = h
    layers.append(nn.Linear(n, out))
    return nn.Sequential(*layers)


class Dynamics(nn.Module):
    def __init__(self, ctx_dim, hidden=(2048, 2048, 2048), dt=0.02, decimation=4, amp=False):
        super().__init__()
        self.dt, self.amp = dt, amp
        # Semi-implicit composition of ``decimation`` substeps under constant acceleration.
        self.c = (decimation + 1) / (2 * decimation)
        self.state_norm = RunningNorm(REP)
        self.target_norm = RunningNorm(NJ)
        self.net = mlp(REP + NJ + ctx_dim, hidden, 6 + NJ)

    @torch.no_grad()
    def update_normalization(self, s, qdes):
        self.state_norm.update(represent(s))
        self.target_norm.update(qdes)

    def forward(self, s, qdes, ctx):
        """Next state after one 20 ms control tick with PD targets ``qdes``."""
        p, quat, v, w, q, qd = split(s)
        R = quat_to_matrix(quat)
        Rt = R.transpose(-1, -2)
        v_b = (Rt @ v[..., None])[..., 0]
        rep = torch.cat((p[..., 2:3], -R[..., 2, :], v_b, w, q, qd), -1)
        x = torch.cat((self.state_norm(rep), self.target_norm(qdes), ctx), -1)
        # Only the MLP runs in bf16; normalization and the integration stay in fp32.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            d = self.net(x)
        d = d.float()
        dv, dw, dqd = d[..., 0:3], d[..., 3:6], d[..., 6:]
        c, dt = self.c, self.dt
        q_next = q + (qd + c * dqd) * dt
        qd_next = qd + dqd
        p_next = p + (R @ (v_b + c * dv)[..., None])[..., 0] * dt
        quat_next = integrate_body_rate(quat, w + c * dw, dt)
        v_next = (R @ (v_b + dv)[..., None])[..., 0]
        # The new twist was expressed in the old root frame; re-express the rate.
        w_world = (R @ (w + dw)[..., None])[..., 0]
        w_next = (quat_to_matrix(quat_next).transpose(-1, -2) @ w_world[..., None])[..., 0]
        return pack(p_next, quat_next, v_next, w_next, q_next, qd_next)


# Bryson's rule: each channel group is weighted by 1 / nominal_error^2 (upstream defaults).
NOMINAL = dict(root_pos=0.01, root_ori=0.02, joint_pos=0.1, joint_vel=3.14)


def loss_blocks(s):
    p, quat, _, _, q, qd = split(s)
    return dict(root_pos=p, root_ori=quat_to_matrix(quat).flatten(-2) * 0.5 ** 0.5, joint_pos=q, joint_vel=qd)


def state_loss(pred, target):
    """Width-weighted Bryson loss over the scored groups and the per-group MSE."""
    a, b = loss_blocks(pred), loss_blocks(target)
    total, width, groups = 0.0, 0, {}
    for name, nominal in NOMINAL.items():
        mse = (a[name] - b[name]).square().mean()
        groups[name] = mse.detach()
        n = a[name].shape[-1]
        total = total + mse * n / nominal ** 2
        width += n
    return total / width, groups
