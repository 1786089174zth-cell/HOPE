"""Physical state layout shared by capture, the world model and analytic heads.

``[root_pos_w(3), root_quat_w(4, wxyz), root_link_lin_vel_w(3),
root_ang_vel_b(3), q(31), qd(31)]`` in Isaac joint order. These are exactly
MuJoCo's free-joint ``qpos``/``qvel`` plus the environment origin.
"""

from __future__ import annotations

import torch

NJ = 31
DIM = 13 + 2 * NJ
SLICES = dict(
    p=slice(0, 3),
    quat=slice(3, 7),
    v=slice(7, 10),
    wb=slice(10, 13),
    q=slice(13, 13 + NJ),
    qd=slice(13 + NJ, DIM),
)


def part(s, name):
    return s[..., SLICES[name]]


def pack(p, quat, v, wb, q, qd):
    return torch.cat((p, quat, v, wb, q, qd), -1)


@torch.no_grad()
def from_backend(backend):
    """Read the current Warp state (link-origin root velocity, body-frame rate)."""
    qpos, qvel = backend.qpos, backend.qvel
    return pack(
        qpos[:, :3] + backend.origin,
        qpos[:, 3:7],
        qvel[:, :3],
        qvel[:, 3:6],
        qpos[:, backend.qids],
        qvel[:, backend.vids],
    ).clone()
