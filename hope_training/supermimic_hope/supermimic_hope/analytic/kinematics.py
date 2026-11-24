"""Differentiable forward kinematics of the Gate3 A3 MJCF, in Isaac body order.

Matches ``WarpBackend.update``: link-origin poses (``xpos``/``xquat``), world
angular velocity, link-origin and COM linear velocity. Every A3 body has one
hinge whose anchor is the body origin and ``qpos0`` is zero, so a child's
origin does not move with its own joint.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


def quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(R):
    """Branch-free conversion with a positive scalar part (wxyz)."""
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    w = 0.5 * (1 + m00 + m11 + m22).clamp_min(1e-12).sqrt()
    x = 0.5 * (1 + m00 - m11 - m22).clamp_min(1e-12).sqrt()
    y = 0.5 * (1 - m00 + m11 - m22).clamp_min(1e-12).sqrt()
    z = 0.5 * (1 - m00 - m11 + m22).clamp_min(1e-12).sqrt()
    x = torch.copysign(x, R[..., 2, 1] - R[..., 1, 2])
    y = torch.copysign(y, R[..., 0, 2] - R[..., 2, 0])
    z = torch.copysign(z, R[..., 1, 0] - R[..., 0, 1])
    q = torch.stack((w, x, y, z), -1)
    return q / q.norm(dim=-1, keepdim=True)


def axis_rotation(axis, angle):
    """Rodrigues rotation about a fixed unit axis; ``angle`` (B,) -> (B,3,3)."""
    a = axis.to(angle.dtype)
    K = torch.zeros(3, 3, dtype=angle.dtype, device=angle.device)
    K[0, 1], K[0, 2], K[1, 0] = -a[2], a[1], a[2]
    K[1, 2], K[2, 0], K[2, 1] = -a[0], -a[1], a[0]
    s, c = angle.sin()[..., None, None], angle.cos()[..., None, None]
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + s * K + (1 - c) * (K @ K)


@dataclass
class Bodies:
    pos: torch.Tensor  # (B, nb, 3) link origins, world
    quat: torch.Tensor  # (B, nb, 4) wxyz
    rot: torch.Tensor  # (B, nb, 3, 3)
    ang_vel: torch.Tensor  # (B, nb, 3) world
    link_lin_vel: torch.Tensor  # (B, nb, 3) link-origin velocity, world
    com_pos: torch.Tensor | None = None  # (B, nb, 3)
    com_lin_vel: torch.Tensor | None = None  # (B, nb, 3)


class A3Kinematics(torch.nn.Module):
    def __init__(self, model, body_names, joint_names):
        super().__init__()
        ids = [model.body(n).id for n in body_names]
        if any(i <= 0 for i in ids):
            raise ValueError("Isaac bodies must be real MJCF bodies")
        root = ids[0]
        if model.body_parentid[root] != 0:
            raise ValueError("The first Isaac body must be the floating base")
        position = {name: i for i, name in enumerate(joint_names)}
        self.order = []  # (isaac index, parent isaac index, joint column)
        index = {b: i for i, b in enumerate(ids)}
        for b in sorted(ids):
            parent = int(model.body_parentid[b])
            if b == root:
                self.order.append((index[b], -1, -1))
                continue
            if parent not in index or model.body_jntnum[b] != 1:
                raise ValueError(f"Unsupported A3 tree at {model.body(b).name}")
            j = model.body_jntadr[b]
            if model.jnt_type[j] != 3 or abs(model.jnt_pos[j]).max() > 0:
                raise ValueError("A3 joints must be hinges anchored at the body origin")
            if abs(model.qpos0[model.jnt_qposadr[j]]) > 0:
                raise ValueError("A3 hinge reference positions must be zero")
            self.order.append((index[b], index[parent], position[model.joint(j).name]))
        n = len(ids)
        f = torch.float32
        self.register_buffer("offset", torch.tensor(model.body_pos[ids], dtype=f))
        self.register_buffer("local_rot", quat_to_matrix(torch.tensor(model.body_quat[ids], dtype=f)))
        axes = torch.zeros(n, 3)
        for i, parent, column in self.order:
            if parent >= 0:
                b = ids[i]
                axes[i] = torch.tensor(model.jnt_axis[model.body_jntadr[b]], dtype=f)
        self.register_buffer("axis", axes)
        self.num_bodies = n

    def forward(self, root_pos, root_quat, q, root_lin_vel_w, root_ang_vel_b, qd, com_offset=None):
        B = q.shape[0]
        pos = [None] * self.num_bodies
        rot = [None] * self.num_bodies
        omega = [None] * self.num_bodies
        vel = [None] * self.num_bodies
        for i, parent, column in self.order:
            if parent < 0:
                R = quat_to_matrix(root_quat)
                pos[i], rot[i] = root_pos, R
                omega[i] = (R @ root_ang_vel_b[..., None])[..., 0]
                vel[i] = root_lin_vel_w
                continue
            Rp = rot[parent]
            pos[i] = pos[parent] + (Rp @ self.offset[i])
            rot[i] = Rp @ self.local_rot[i] @ axis_rotation(self.axis[i], q[:, column])
            axis_w = rot[i] @ self.axis[i]
            omega[i] = omega[parent] + axis_w * qd[:, column, None]
            vel[i] = vel[parent] + torch.cross(omega[parent], pos[i] - pos[parent], dim=-1)
        pos, rot = torch.stack(pos, 1), torch.stack(rot, 1)
        omega, vel = torch.stack(omega, 1), torch.stack(vel, 1)
        result = Bodies(pos, matrix_to_quat(rot), rot, omega, vel)
        if com_offset is not None:
            arm = (rot @ com_offset[..., None])[..., 0]
            result.com_pos = pos + arm
            result.com_lin_vel = vel + torch.cross(omega, arm, dim=-1)
        return result
