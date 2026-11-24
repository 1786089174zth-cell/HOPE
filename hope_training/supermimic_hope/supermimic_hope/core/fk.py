"""Differentiable forward kinematics of the A3 MJCF, batched by tree depth.

Every A3 body below the floating base has one hinge anchored at the body origin
with a zero reference angle, so a body's origin does not move with its own
joint. Bodies of one depth are evaluated together (11 batched levels instead of
one Python step per body).
"""

from __future__ import annotations

import torch
from torch import nn

from .rotations import quat_to_matrix


class Kinematics(nn.Module):
    def __init__(self, model, joint_names, root="pelvis_link"):
        super().__init__()
        root_id = model.body(root).id
        if model.body_parentid[root_id] != 0:
            raise ValueError("the root must be the floating base")
        ids = [root_id]
        for b in range(model.nbody):
            if b != root_id and b != 0 and model.body_rootid[b] == root_id:
                ids.append(b)
        column = {name: i for i, name in enumerate(joint_names)}
        index = {b: i for i, b in enumerate(ids)}
        self.names = [model.body(b).name for b in ids]
        depth = {root_id: 0}
        levels: dict[int, list[tuple[int, int, int]]] = {}
        for b in ids[1:]:
            parent = int(model.body_parentid[b])
            if model.body_jntnum[b] != 1:
                raise ValueError(f"unsupported A3 tree at {model.body(b).name}")
            j = model.body_jntadr[b]
            if model.jnt_type[j] != 3 or abs(model.jnt_pos[j]).max() > 0 or abs(model.qpos0[model.jnt_qposadr[j]]) > 0:
                raise ValueError("A3 joints must be zero-reference hinges anchored at the body origin")
            depth[b] = depth[parent] + 1
            levels.setdefault(depth[b], []).append((index[b], index[parent], column[model.joint(j).name]))
        f = torch.float32
        self.levels = []
        order = [0]
        for d in sorted(levels):
            body = torch.tensor([x[0] for x in levels[d]])
            parent = torch.tensor([x[1] for x in levels[d]])
            col = torch.tensor([x[2] for x in levels[d]])
            mj = [ids[i] for i in body.tolist()]
            axis = torch.tensor(model.jnt_axis[[model.body_jntadr[b] for b in mj]], dtype=f)
            K = torch.zeros(len(mj), 3, 3)
            K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -axis[:, 2], axis[:, 1], axis[:, 2]
            K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -axis[:, 0], -axis[:, 1], axis[:, 0]
            name = f"l{d}"
            self.register_buffer(name + "_parent", torch.tensor([order.index(int(p)) for p in parent]), persistent=False)
            self.register_buffer(name + "_col", col, persistent=False)
            self.register_buffer(name + "_offset", torch.tensor(model.body_pos[mj], dtype=f), persistent=False)
            self.register_buffer(name + "_rot", quat_to_matrix(torch.tensor(model.body_quat[mj], dtype=f)), persistent=False)
            self.register_buffer(name + "_axis", axis, persistent=False)
            self.register_buffer(name + "_K", K, persistent=False)
            self.register_buffer(name + "_K2", K @ K, persistent=False)
            self.levels.append(name)
            order += body.tolist()
        # ``order`` is the body index of each row produced level by level.
        self.register_buffer("unsort", torch.argsort(torch.tensor(order)), persistent=False)
        self.num_bodies = len(ids)

    def index(self, names):
        return [self.names.index(n) for n in names]

    def forward(self, s):
        """``s`` (B, 75) -> link-origin ``pos`` (B, nb, 3), ``rot`` (B, nb, 3, 3),
        world ``ang_vel`` and link-origin ``lin_vel`` (B, nb, 3)."""
        p, quat, v, w_b = s[:, 0:3], s[:, 3:7], s[:, 7:10], s[:, 10:13]
        q, qd = s[:, 13:44], s[:, 44:75]
        R0 = quat_to_matrix(quat)
        pos, rot = [p[:, None]], [R0[:, None]]
        omega, vel = [(R0 @ w_b[..., None])[..., 0][:, None]], [v[:, None]]
        eye = torch.eye(3, dtype=s.dtype, device=s.device)
        for name in self.levels:
            parent = getattr(self, name + "_parent")
            col = getattr(self, name + "_col")
            P = torch.cat(pos, 1)[:, parent]
            Rp = torch.cat(rot, 1)[:, parent]
            Wp = torch.cat(omega, 1)[:, parent]
            Vp = torch.cat(vel, 1)[:, parent]
            angle = q[:, col]
            hinge = (eye + angle.sin()[..., None, None] * getattr(self, name + "_K")
                     + (1 - angle.cos())[..., None, None] * getattr(self, name + "_K2"))
            arm = (Rp @ getattr(self, name + "_offset")[..., None])[..., 0]
            R = Rp @ getattr(self, name + "_rot") @ hinge
            axis_w = (R @ getattr(self, name + "_axis")[..., None])[..., 0]
            pos.append(P + arm)
            rot.append(R)
            omega.append(Wp + axis_w * qd[:, col, None])
            vel.append(Vp + torch.cross(Wp, arm, dim=-1))
        unsort = self.unsort
        return (torch.cat(pos, 1)[:, unsort], torch.cat(rot, 1)[:, unsort],
                torch.cat(omega, 1)[:, unsort], torch.cat(vel, 1)[:, unsort])
