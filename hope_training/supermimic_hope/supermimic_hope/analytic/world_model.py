"""Physical-state world model (upstream SuperMimic design, HOPE controls).

Predicts the post-physics state from the step's start state, the executed PD
targets of this and the previous tick (command latency switches between them
inside the tick), the episode's latency and the environment's startup DR.
Inputs are heading-invariant; outputs are frame-local increments.
"""

from __future__ import annotations

import torch
from torch import nn

from . import state as S
from .kinematics import quat_to_matrix

INCREMENT = 3 + 3 + 3 + 3 + 2 * S.NJ
# Per-component floors on the error normalization: position, rotation, linear
# velocity, angular velocity, q and qd increments (upstream values).
FLOORS = (3e-3,) * 3 + (5e-3,) * 3 + (2e-2,) * 3 + (5e-2,) * 3 + (3e-3,) * S.NJ + (5e-2,) * S.NJ


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), -1)


def quat_exp(v):
    angle = v.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(x/2)/x with a series near zero keeps the gradient finite.
    k = torch.where(angle > 1e-4, torch.sin(half) / angle.clamp_min(1e-12), 0.5 - angle.square() / 48)
    return torch.cat((torch.cos(half), v * k), -1)


def quat_log(q):
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(s, q[..., :1])
    k = torch.where(s > 1e-6, angle / s.clamp_min(1e-12), 2.0 / q[..., :1].clamp_min(1e-6))
    return v * k


def quat_conj(q):
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack(((0.5 * yaw).cos(), zero, zero, (0.5 * yaw).sin()), -1)


def rotate(q, v):
    return (quat_to_matrix(q) @ v[..., None])[..., 0]


def rotate_inverse(q, v):
    return (quat_to_matrix(q).transpose(-1, -2) @ v[..., None])[..., 0]


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-2, floor=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("floor", torch.tensor(floor) if floor is not None else torch.zeros(dim))
        self.eps = eps

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel()).float()
        if not x.shape[0]:
            return
        n, m, v = x.shape[0], x.mean(0), x.var(0, unbiased=False)
        total = self.count + n
        delta = m - self.mean
        self.var.copy_((self.var * self.count + v * n + delta.square() * self.count * n / total) / total)
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def std(self):
        return torch.maximum((self.var + self.eps**2).sqrt(), self.floor)

    def forward(self, x):
        return (x - self.mean) / self.std()


def mlp(n, hidden, out, layer_norm=True):
    layers = []
    for h in hidden:
        layers += [nn.Linear(n, h)]
        if layer_norm:
            layers += [nn.LayerNorm(h)]
        layers += [nn.SiLU()]
        n = h
    return nn.Sequential(*layers, nn.Linear(n, out))


def invariant(s):
    """Heading-free physical features: height, tilt frame, body velocities, joints."""
    quat = S.part(s, "quat")
    local = quat_mul(quat_conj(yaw_quat(quat)), quat)
    R = quat_to_matrix(local)
    return torch.cat((
        S.part(s, "p")[..., 2:3],
        R[..., :, 0],
        R[..., :, 1],
        rotate_inverse(quat, S.part(s, "v")),
        S.part(s, "wb"),
        S.part(s, "q"),
        S.part(s, "qd"),
    ), -1)


INVARIANT = 1 + 6 + 3 + 3 + 2 * S.NJ


class PhysicalWorldModel(nn.Module):
    def __init__(self, xi_dim, hidden=(1024, 1024, 1024)):
        super().__init__()
        self.in_dim = INVARIANT + 2 * S.NJ + 1 + xi_dim
        self.net = mlp(self.in_dim, list(hidden), INCREMENT)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(INCREMENT, eps=1e-4, floor=FLOORS)

    def inputs(self, s, qexec, qexec_prev, cmd_delay, xi):
        q = S.part(s, "q")
        return torch.cat((invariant(s), qexec - q, qexec_prev - q, cmd_delay[..., None], xi), -1)

    @staticmethod
    def increments(s0, s1):
        q0, q1 = S.part(s0, "quat"), S.part(s1, "quat")
        return torch.cat((
            rotate_inverse(yaw_quat(q0), S.part(s1, "p") - S.part(s0, "p")),
            quat_log(quat_mul(quat_conj(q0), q1)),
            rotate_inverse(q0, S.part(s1, "v") - S.part(s0, "v")),
            S.part(s1, "wb") - S.part(s0, "wb"),
            S.part(s1, "q") - S.part(s0, "q"),
            S.part(s1, "qd") - S.part(s0, "qd"),
        ), -1)

    @staticmethod
    def apply(s0, d):
        q0 = S.part(s0, "quat")
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, S.NJ, S.NJ), -1)
        q1 = quat_mul(q0, quat_exp(dth))
        q1 = q1 / q1.norm(dim=-1, keepdim=True)
        return S.pack(
            S.part(s0, "p") + rotate(yaw_quat(q0), dp),
            q1,
            S.part(s0, "v") + rotate(q0, dv),
            S.part(s0, "wb") + dw,
            S.part(s0, "q") + dq,
            S.part(s0, "qd") + dqd,
        )

    def step(self, s, qexec, qexec_prev, cmd_delay, xi):
        x = self.in_norm(self.inputs(s, qexec, qexec_prev, cmd_delay, xi))
        d = self.net(x) * self.out_norm.std() + self.out_norm.mean
        return self.apply(s, d)

    def error(self, s_hat, s):
        """Normalized frame-local error in increment units (B, INCREMENT)."""
        return self.increments(s, s_hat) / self.out_norm.std()


def push(s, delta):
    """Apply the recorded interval push (root linear and body angular velocity)."""
    return torch.cat((s[..., :7], s[..., 7:13] + delta, s[..., 13:]), -1)
