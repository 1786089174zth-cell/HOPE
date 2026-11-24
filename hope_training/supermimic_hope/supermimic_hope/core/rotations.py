"""Quaternion (wxyz) and rotation helpers shared by the sim, model and task."""

from __future__ import annotations

import torch


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), -1)


def quat_conj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), -1).reshape(*q.shape[:-1], 3, 3)


def rotate(q, v):
    """World vector of body vector ``v`` (q: body -> world)."""
    return (quat_to_matrix(q) @ v[..., None])[..., 0]


def rotate_inv(q, v):
    return (quat_to_matrix(q).transpose(-1, -2) @ v[..., None])[..., 0]


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(q):
    half = 0.5 * yaw_of(q)
    zero = torch.zeros_like(half)
    return torch.stack((half.cos(), zero, zero, half.sin()), -1)


def quat_from_yaw(yaw):
    zero = torch.zeros_like(yaw)
    return torch.stack(((0.5 * yaw).cos(), zero, zero, (0.5 * yaw).sin()), -1)


def quat_exp(v):
    """Unit quaternion of the rotation vector ``v`` (finite gradient at zero)."""
    angle = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-12)
    return torch.cat(((0.5 * angle).cos(), (0.5 * angle).sin() * v / angle), -1)


def integrate_body_rate(q, w_body, dt):
    """``q (x) exp(w_body dt)``: MuJoCo's free-joint orientation update."""
    out = quat_mul(q, quat_exp(w_body * dt))
    return out / out.norm(dim=-1, keepdim=True)


def rotation_angle(Ra, Rb):
    """Geodesic angle between rotation matrices, with a finite gradient at 0 and pi."""
    trace = (Ra * Rb).sum((-1, -2))
    cos = ((trace - 1) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos)


def rotation_error_sq(Ra, Rb):
    """Smooth squared rotation error ``2 (1 - cos theta) ~ theta^2`` (no acos)."""
    trace = (Ra * Rb).sum((-1, -2))
    return (3.0 - trace).clamp_min(0.0)
