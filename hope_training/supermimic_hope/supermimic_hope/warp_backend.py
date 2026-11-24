"""MuJoCo-Warp physics under the existing HOPE managers.

Kit/PhysX are used only to construct the existing task and resolve its assets,
actuators and startup randomization. After attachment, every dynamics step and
state/contact read used by the task comes from MuJoCo-Warp. This intentionally
keeps the existing command/reward/reset code rather than reimplementing it.
It is a migration backend, not a claim that MuJoCo and PhysX dynamics match.
"""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch
import mujoco
import mujoco_warp as mjw
import warp as wp

from . import a3


class WarpBackend:
    def __init__(self, env, *, capture=True):
        from isaaclab.utils.math import (
            quat_apply,
            quat_rotate_inverse,
            matrix_from_quat,
        )

        self.quat_apply, self.quat_inv_apply = quat_apply, quat_rotate_inverse
        self.env, self.robot = env, env.scene["robot"]
        self.original_sim = env.sim
        self.N, self.device = env.num_envs, env.device
        self.origin = env.scene.env_origins.clone()
        self.model, self.receipt = a3.build_model()
        self.ix = a3.indices(self.model)
        assert list(self.robot.joint_names) == list(a3.JOINTS)
        self.body_ids = np.array([self.model.body(n).id for n in self.robot.body_names])
        self.qids = torch.tensor(self.ix["qpos"], device=self.device)
        self.vids = torch.tensor(self.ix["dof"], device=self.device)
        self.bids = torch.tensor(self.body_ids, device=self.device)
        old = self.robot.data
        # Frame identity matters for the existing motion clips and racket FK.
        initial = mujoco.MjData(self.model)
        initial.qpos[:3] = (old.root_pos_w[0] - self.origin[0]).cpu().numpy()
        initial.qpos[3:7] = old.root_quat_w[0].cpu().numpy()
        initial.qpos[self.ix["qpos"]] = old.joint_pos[0].cpu().numpy()
        mujoco.mj_forward(self.model, initial)
        position_error = float(
            np.abs(
                initial.xpos[self.body_ids]
                - (old.body_pos_w[0] - self.origin[0]).cpu().numpy()
            ).max()
        )
        rotation_dot = np.abs(
            (initial.xquat[self.body_ids] * old.body_quat_w[0].cpu().numpy()).sum(-1)
        )
        if position_error > 1.0e-4 or float(rotation_dot.min()) < 1.0 - 1.0e-5:
            raise ValueError(
                f"Isaac/MJCF link frame mismatch: {position_error=}, {rotation_dot.min()=}"
            )
        self.receipt["isaac_mjcf_initial_fk_max_error_m"] = position_error
        self.data = SimpleNamespace()
        for name in (
            "GRAVITY_VEC_W",
            "FORWARD_VEC_B",
            "default_joint_pos_nominal",
            "default_joint_pos",
            "default_joint_vel",
            "default_root_state",
            "default_joint_stiffness",
            "default_joint_damping",
            "joint_stiffness",
            "joint_damping",
            "joint_pos_limits",
            "soft_joint_pos_limits",
            "joint_vel_limits",
            "soft_joint_vel_limits",
            "joint_effort_limits",
            "default_mass",
            "default_inertia",
            "default_joint_armature",
            "joint_armature",
            "joint_friction_coeff",
        ):
            if hasattr(old, name):
                value = getattr(old, name)
                setattr(
                    self.data, name, value.clone() if torch.is_tensor(value) else value
                )
        self.data.joint_names = list(self.robot.joint_names)
        self.data.body_names = list(self.robot.body_names)
        self.data.joint_pos_target = old.default_joint_pos.clone()
        self.data.joint_vel_target = torch.zeros_like(old.default_joint_pos)
        self.data.joint_effort_target = torch.zeros_like(old.default_joint_pos)
        self.data.joint_acc = torch.zeros_like(old.default_joint_pos)
        self.last_vel = torch.zeros_like(old.default_joint_pos)
        # Body inertia and masses are copied after the original startup events.
        view = self.robot.root_physx_view
        masses = view.get_masses().to(self.device)
        inertias = view.get_inertias().to(self.device).reshape(self.N, -1, 3, 3)
        coms = view.get_coms().to(self.device)
        self.receipt["startup_mass_range_kg"] = [
            float(masses.sum(-1).min()),
            float(masses.sum(-1).max()),
        ]
        # Per-world model arrays are required: DR may never silently become shared.
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(self.model)
            # MuJoCo-Warp 3.5 uses broadcastable model arrays. Expand the
            # derived arrays too before set_const recomputes randomized inertia.
            for key, value in vars(self.m).items():
                if isinstance(value, wp.array) and getattr(value, "_is_batched", False):
                    tensor = wp.to_torch(value)
                    tensor = tensor.repeat((self.N,) + (1,) * (tensor.ndim - 1))
                    array = wp.from_torch(tensor.contiguous(), dtype=value.dtype)
                    array._is_batched = True
                    setattr(self.m, key, array)
            d0 = mujoco.MjData(self.model)
            d0.qpos[2] = 1.06
            d0.qpos[self.ix["qpos"]] = a3.DEFAULT_Q
            d0.ctrl[:] = a3.DEFAULT_Q
            mujoco.mj_forward(self.model, d0)
            self.d = mjw.put_data(self.model, d0, nworld=self.N, nconmax=64, njmax=256)
        wp.to_torch(self.m.body_mass)[:, self.bids] = masses
        # PhysX returns inertia in the link frame and COM pose in xyzw order.
        # Copy its resolved principal frame as well as its moments; retaining
        # the MJCF frame would silently drop off-diagonal inertia after DR.
        inertial_quat = coms[..., [6, 3, 4, 5]]
        R = matrix_from_quat(inertial_quat)
        principal = R.transpose(-1, -2) @ inertias @ R
        off = principal - torch.diag_embed(principal.diagonal(dim1=-2, dim2=-1))
        self.receipt["inertia_off_diagonal_max"] = float(off.abs().max())
        if float(off.abs().max()) > 1.0e-6:
            raise ValueError(
                "Resolved PhysX principal frame does not diagonalize inertia"
            )
        wp.to_torch(self.m.body_inertia)[:, self.bids] = principal.diagonal(
            dim1=-2, dim2=-1
        )
        wp.to_torch(self.m.body_iquat)[:, self.bids] = inertial_quat
        wp.to_torch(self.m.body_ipos)[:, self.bids] = coms[..., :3]
        wp.to_torch(self.m.dof_armature)[:, self.vids] = self.data.joint_armature
        effort = self.data.joint_effort_limits
        wp.to_torch(self.m.actuator_forcerange)[:] = torch.stack((-effort, effort), -1)
        # The resolved implicit actuator damping already includes message Kd
        # plus passive damping. Put that total into kv exactly once.
        wp.to_torch(self.m.dof_damping)[:, self.vids] = 0
        self._write_pd(self.data.joint_stiffness, "joint_stiffness")
        self._write_pd(self.data.joint_damping, "joint_damping")
        materials = view.get_material_properties().to(self.device)
        self.receipt["friction_source"] = (
            "resolved per-body dynamic friction; MuJoCo max combine with floor zero"
        )
        mu = materials[..., 1]
        self.receipt["dynamic_friction_range"] = [float(mu.min()), float(mu.max())]
        self.receipt["contact_difference"] = (
            "MuJoCo solver/hulls; one Coulomb coefficient instead of separate static/dynamic friction; not PhysX parity"
        )
        geom_mu = wp.to_torch(self.m.geom_friction)
        counts = env.event_manager.get_term_cfg(
            "physics_material"
        ).func.num_shapes_per_body
        if counts is None:
            counts = [
                self.robot._physics_sim_view.create_rigid_body_view(p).max_shapes
                for p in view.link_paths[0]
            ]
        assert sum(counts) == materials.shape[1]
        start = 0
        for i, count in enumerate(counts):
            if count:
                values = mu[:, start : start + count]
                if not torch.allclose(values, values[:, :1].expand_as(values)):
                    raise ValueError(
                        f"Cannot map multiple friction values on {self.robot.body_names[i]} to different MJCF hulls"
                    )
                geoms = np.flatnonzero(self.model.geom_bodyid == self.body_ids[i])
                geom_mu[:, geoms, 0] = values[:, :1]
            start += count
        geom_mu[:, self.model.geom("floor").id, 0] = 0
        self.qpos, self.qvel = wp.to_torch(self.d.qpos), wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.ctrl.copy_(self.data.default_joint_pos)
        self.contact = env.scene.sensors["contact_forces"]
        self.sensor_ids = torch.tensor(
            [self.model.body(n).id for n in self.contact.body_names], device=self.device
        )
        B = len(self.contact.body_names)
        H = max(int(self.contact.cfg.history_length), 1)

        def z(*s):
            return torch.zeros(*s, device=self.device)

        self.contact_data = SimpleNamespace(
            net_forces_w=z(self.N, B, 3),
            net_forces_w_history=z(self.N, H, B, 3),
            current_air_time=z(self.N, B),
            last_air_time=z(self.N, B),
            current_contact_time=z(self.N, B),
            last_contact_time=z(self.N, B),
        )
        self.force = wp.zeros(
            self.d.contact.geom.shape[0], dtype=wp.spatial_vector, device=self.device
        )
        self.contact_ids = wp.array(
            np.arange(self.force.shape[0], dtype=np.int32),
            dtype=int,
            device=self.device,
        )
        self.geom_body = torch.tensor(self.model.geom_bodyid, device=self.device)
        self.all_force = z(self.N * self.model.nbody, 3)
        self.graph = None
        with wp.ScopedDevice(self.device):
            mjw.set_const(self.m, self.d)
            mjw.forward(self.m, self.d)
            self._physics_step()
            if capture:
                with wp.ScopedCapture() as cap:
                    self._physics_step()
                self.graph = cap.graph
        self.robot._data = self.data
        self.robot.set_joint_position_target = self.set_target
        self.robot.write_joint_state_to_sim = self.write_joints
        self.robot.write_root_pose_to_sim = self.write_pose
        self.robot.write_root_velocity_to_sim = self.write_velocity
        self.robot.write_root_state_to_sim = self.write_root
        self.robot.write_joint_stiffness_to_sim = lambda x, **kw: self._write_pd(
            x, "joint_stiffness", **kw
        )
        self.robot.write_joint_damping_to_sim = lambda x, **kw: self._write_pd(
            x, "joint_damping", **kw
        )
        self.robot.reset = self.reset
        self.contact._data = self.contact_data
        self.contact._update_outdated_buffers = lambda: None
        env.scene.write_data_to_sim = lambda: None
        env.scene.update = self.update
        env.scene.reset = self.reset
        env.sim = self
        self.update(0.0)
        self.receipt.update(
            backend="mujoco_warp",
            task_managers="original_HOPE",
            isaac_bootstrap_only=True,
            captured=capture,
            drive_damping="resolved Isaac total, passive MuJoCo damping zero",
            drive_effort="resolved Isaac joint_effort_limits",
            inertia_source="resolved PhysX full inertia and principal frame",
        )

    def _write_pd(self, x, name, env_ids=None, joint_ids=None):
        if joint_ids is not None:
            raise NotImplementedError(
                "partial PD writes require explicit index mapping"
            )
        ids = slice(None) if env_ids is None else env_ids
        getattr(self.data, name)[ids] = x
        gain, bias = (
            wp.to_torch(self.m.actuator_gainprm),
            wp.to_torch(self.m.actuator_biasprm),
        )
        if name == "joint_stiffness":
            gain[ids, :, 0] = x
            bias[ids, :, 1] = -x
        else:
            bias[ids, :, 2] = -x

    def has_gui(self):
        return False

    def has_rtx_sensors(self):
        return False

    def render(self, *args, **kwargs):
        pass

    def step(self, render=False):
        self.ctrl.copy_(self.data.joint_pos_target)
        with wp.ScopedDevice(self.device):
            wp.capture_launch(self.graph) if self.graph else self._physics_step()

    def _physics_step(self):
        mjw.step(self.m, self.d)
        # mj_step leaves derived poses at the pre-integration state. Managers
        # require joint and body sensors from the same post-step instant.
        mjw.kinematics(self.m, self.d)
        mjw.com_pos(self.m, self.d)
        mjw.com_vel(self.m, self.d)

    def forward(self):
        with wp.ScopedDevice(self.device):
            mjw.forward(self.m, self.d)
        self.update(0.0)

    def _sync_written_state(self):
        # Reset managers need current poses/velocities between state writes.
        # ManagerBasedRLEnv calls sim.forward once after the complete reset;
        # avoid running the contact solver for every intermediate joint/root write.
        with wp.ScopedDevice(self.device):
            mjw.kinematics(self.m, self.d)
            mjw.com_pos(self.m, self.d)
            mjw.com_vel(self.m, self.d)
        self.update(0.0, read_contacts=False)

    def __getattr__(self, name):
        return getattr(self.original_sim, name)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.contact_data.net_forces_w_history[ids] = 0
        self.contact_data.current_air_time[ids] = 0
        self.contact_data.current_contact_time[ids] = 0
        self.last_vel[ids] = self.qvel[ids][:, self.vids]

    def set_target(self, target, joint_ids=None, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        joints = slice(None) if joint_ids is None else joint_ids
        if isinstance(ids, slice):
            self.data.joint_pos_target[ids, joints] = target
        elif isinstance(joints, slice):
            self.data.joint_pos_target[ids] = target
        else:
            self.data.joint_pos_target[ids[:, None], joints] = target

    def write_joints(self, position, velocity, joint_ids=None, env_ids=None):
        ids = torch.arange(self.N, device=self.device) if env_ids is None else env_ids
        j = slice(None) if joint_ids is None else joint_ids
        self.qpos[ids[:, None], self.qids[j]] = position
        self.qvel[ids[:, None], self.vids[j]] = velocity
        if joint_ids is None:
            self.data.joint_pos_target[ids] = position
        else:
            self.data.joint_pos_target[ids[:, None], joint_ids] = position
        self._sync_written_state()

    def write_pose(self, pose, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.qpos[ids, :3] = pose[:, :3] - self.origin[ids]
        self.qpos[ids, 3:7] = pose[:, 3:7]
        self._sync_written_state()

    def write_velocity(self, velocity, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        # Isaac's root velocity writer accepts COM linear velocity. A MuJoCo
        # free joint stores link-origin translation and body-frame rotation.
        offset = self.quat_apply(
            self.qpos[ids, 3:7], wp.to_torch(self.m.body_ipos)[ids, self.ix["root"]]
        )
        self.qvel[ids, :3] = velocity[:, :3] - torch.cross(
            velocity[:, 3:], offset, dim=-1
        )
        self.qvel[ids, 3:6] = self.quat_inv_apply(self.qpos[ids, 3:7], velocity[:, 3:6])
        self._sync_written_state()

    def write_root(self, state, env_ids=None):
        self.write_pose(state[:, :7], env_ids)
        self.write_velocity(state[:, 7:], env_ids)

    def validate_snapshot(self):
        """Probe-only CPU reference for post-step frames and COM velocity writes."""
        import copy

        model = copy.copy(self.model)
        for name in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
            getattr(model, name)[:] = (
                wp.to_torch(getattr(self.m, name))[0].cpu().numpy()
            )
        ref = mujoco.MjData(model)
        mujoco.mj_setConst(model, ref)
        ref.qpos[:] = self.qpos[0].cpu().numpy()
        ref.qvel[:] = self.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, ref)
        pos = self.data.body_pos_w[0] - self.origin[0]
        position_error = float(
            np.abs(pos.cpu().numpy() - ref.xpos[self.body_ids]).max()
        )
        velocity = []
        for body in self.body_ids:
            v = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, ref, mujoco.mjtObj.mjOBJ_BODY, int(body), v, 0
            )
            velocity.append(v[3:])
        velocity_error = float(
            np.abs(self.data.body_lin_vel_w[0].cpu().numpy() - np.array(velocity)).max()
        )
        assert position_error < 2.0e-4, position_error
        assert velocity_error < 2.0e-3, velocity_error
        ids = torch.zeros(1, dtype=torch.long, device=self.device)
        original = self.data.root_vel_w[:1].clone()
        desired = original.new_tensor([[0.13, -0.21, 0.08, 0.4, -0.3, 0.2]])
        self.write_velocity(desired, env_ids=ids)
        write_error = float((self.data.root_vel_w[:1] - desired).abs().max())
        self.write_velocity(original, env_ids=ids)
        assert write_error < 1.0e-5, write_error
        assert torch.all(wp.to_torch(self.m.dof_damping)[:, self.vids] == 0)
        return dict(
            cpu_body_position_max_error_m=position_error,
            cpu_body_com_velocity_max_error_mps=velocity_error,
            root_com_velocity_write_max_error=write_error,
        )

    def update(self, dt, read_contacts=True):
        d = self.data
        d.joint_pos = self.qpos[:, self.qids]
        d.joint_vel = self.qvel[:, self.vids]
        d.joint_acc = (
            (d.joint_vel - self.last_vel) / dt if dt else torch.zeros_like(d.joint_vel)
        )
        self.last_vel.copy_(d.joint_vel)
        d.root_pos_w = self.qpos[:, :3] + self.origin
        d.root_quat_w = self.qpos[:, 3:7].clone()
        d.root_link_lin_vel_w = self.qvel[:, :3].clone()
        d.root_ang_vel_b = self.qvel[:, 3:6].clone()
        d.root_ang_vel_w = self.quat_apply(d.root_quat_w, d.root_ang_vel_b)
        d.root_link_pose_w = torch.cat((d.root_pos_w, d.root_quat_w), -1)
        d.root_link_vel_w = torch.cat((d.root_link_lin_vel_w, d.root_ang_vel_w), -1)
        d.root_link_state_w = torch.cat((d.root_link_pose_w, d.root_link_vel_w), -1)
        d.projected_gravity_b = self.quat_inv_apply(
            d.root_quat_w, d.root_pos_w.new_tensor([0.0, 0.0, -1.0]).expand(self.N, -1)
        )
        d.body_pos_w = wp.to_torch(self.d.xpos)[:, self.bids] + self.origin[:, None]
        d.body_quat_w = wp.to_torch(self.d.xquat)[:, self.bids].clone()
        d.body_link_pos_w, d.body_link_quat_w = d.body_pos_w, d.body_quat_w
        cv = wp.to_torch(self.d.cvel)[:, self.bids]
        root_com = wp.to_torch(self.d.subtree_com)[:, self.ix["root"]]
        rel = wp.to_torch(self.d.xpos)[:, self.bids] - root_com[:, None]
        d.body_ang_vel_w = cv[..., :3].clone()
        d.body_link_lin_vel_w = cv[..., 3:] + torch.cross(cv[..., :3], rel, dim=-1)
        d.body_com_lin_vel_w = cv[..., 3:] + torch.cross(
            cv[..., :3],
            wp.to_torch(self.d.xipos)[:, self.bids] - root_com[:, None],
            dim=-1,
        )
        # Isaac's historical body/root *_lin_vel aliases refer to COM, while
        # *_pos aliases refer to the link origin. Preserve that mixed contract.
        d.body_lin_vel_w = d.body_com_lin_vel_w
        d.root_lin_vel_w = d.body_com_lin_vel_w[:, 0]
        d.root_lin_vel_b = self.quat_inv_apply(d.root_quat_w, d.root_lin_vel_w)
        d.root_com_vel_w = torch.cat((d.body_com_lin_vel_w[:, 0], d.root_ang_vel_w), -1)
        d.root_vel_w = d.root_com_vel_w
        d.root_state_w = torch.cat((d.root_link_pose_w, d.root_vel_w), -1)
        d.body_state_w = torch.cat(
            (d.body_pos_w, d.body_quat_w, d.body_lin_vel_w, d.body_ang_vel_w), -1
        )
        d.applied_torque = wp.to_torch(self.d.qfrc_actuator)[:, self.vids].clone()
        d.computed_torque = (
            d.joint_stiffness * (d.joint_pos_target - d.joint_pos)
            - d.joint_damping * d.joint_vel
        )
        if not read_contacts:
            return
        with wp.ScopedDevice(self.device):
            mjw.contact_force(self.m, self.d, self.contact_ids, True, self.force)
        f = wp.to_torch(self.force)[:, :3]
        geom = wp.to_torch(self.d.contact.geom).long()
        world = wp.to_torch(self.d.contact.worldid).long()
        valid = torch.arange(len(f), device=self.device) < wp.to_torch(self.d.nacon)[0]
        f = f * valid[:, None]
        self.all_force.zero_()
        for side, sign in ((0, -1), (1, 1)):
            index = (
                world.clamp(0, self.N - 1) * self.model.nbody
                + self.geom_body[geom[:, side].clamp(0, self.model.ngeom - 1)]
            )
            self.all_force.index_add_(0, index, sign * f)
        force = self.all_force.reshape(self.N, self.model.nbody, 3)[:, self.sensor_ids]
        c = self.contact_data
        c.net_forces_w.copy_(force)
        if dt:
            c.net_forces_w_history[:, 1:] = c.net_forces_w_history[:, :-1].clone()
            c.net_forces_w_history[:, 0] = force
            touch = force.norm(dim=-1) > self.contact.cfg.force_threshold
            c.last_air_time = torch.where(
                touch & (c.current_air_time > 0), c.current_air_time, c.last_air_time
            )
            c.last_contact_time = torch.where(
                ~touch & (c.current_contact_time > 0),
                c.current_contact_time,
                c.last_contact_time,
            )
            c.current_air_time = torch.where(touch, 0.0, c.current_air_time + dt)
            c.current_contact_time = torch.where(
                touch, c.current_contact_time + dt, 0.0
            )
