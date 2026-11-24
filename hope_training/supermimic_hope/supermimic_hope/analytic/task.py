"""Differentiable HOPEPingPongMotionQuality reward, termination and observation heads.

Each reward term below re-states the corresponding task function in terms of
physical states, so imagined gradients flow through the task's real formula:

* ``s_cmd``: the state the command caches at its update (racket pose/velocity);
* ``s_end``: the post-physics, pre-reset state read through ``robot.data``.

Everything a term reads from commands/curricula (clocks, references, gates,
scales) is recorded from the unmodified task at reward time by calling the
task's own helpers. Contact load, episode bookkeeping and HOLD dwell are
recorded as well; dwell is then advanced inside imagination. For every term the
recorded correction ``real - analytic(real state)`` keeps imagined values exact
at the recorded state; ``TaskModel.check`` reports it.
"""

from __future__ import annotations

from functools import cached_property
import math
from types import SimpleNamespace

import torch

from . import state as S
from .kinematics import A3Kinematics, quat_to_matrix

GRAVITY = (0.0, 0.0, -1.0)


def huber(x):
    return torch.where(x <= 1.0, 0.5 * x.square(), x - 0.5)


def deadband_huber(value, margin, std):
    return huber(torch.clamp(value - float(margin), min=0.0) / max(float(std), 1e-6))


def safe_acos(c):
    return torch.acos(c.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))


def rpy(quat):
    """``full_body_hold.rpy`` with a finite derivative at pitch = +-90 deg."""
    w, x, y, z = quat.unbind(-1)
    return torch.stack((
        torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        torch.asin((2 * (w * y - z * x)).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)),
        torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ), -1)


def readiness_ratios(error, torso_error, foot_error, width_error, height_error,
                     hand_error, distance, speed, base_rpy, joint_speed, loaded, waist_pitch_column):
    """``complete_hold.readiness_ratios`` with a finite tilt derivative when upright."""
    tilt = safe_acos(base_rpy[:, 0].cos() * base_rpy[:, 1].cos())
    return torch.stack((
        error.abs().amax(-1) / math.radians(5),
        error[:, waist_pitch_column].abs() / math.radians(3),
        torso_error[:, :2].abs().amax(-1) / math.radians(3),
        foot_error.abs().flatten(1).amax(-1) / math.radians(5),
        width_error.abs() / .02, height_error.abs() / .02,
        hand_error.abs().amax(-1) / .03,
        distance / .05, speed / .12, base_rpy[:, 2].abs() / math.radians(5),
        tilt / math.radians(12), joint_speed.abs().amax(-1) / .5,
        torch.where(loaded, torch.zeros_like(speed), torch.full_like(speed, 2.)),
    ), -1)


def yaw_of(quat):
    q = quat
    fx = 1.0 - 2.0 * (q[..., 2] ** 2 + q[..., 3] ** 2)
    fy = 2.0 * (q[..., 1] * q[..., 2] + q[..., 0] * q[..., 3])
    return torch.atan2(fy, fx)


def rotate_inverse(R, v):
    return (R.transpose(-1, -2) @ v[..., None])[..., 0]


def yaw_frame_inverse(quat, v):
    """``quat_rotate_inverse(yaw_quat(q), v)`` without forming the quaternion."""
    yaw = yaw_of(quat)
    c, s = yaw.cos(), yaw.sin()
    x, y = v[..., 0], v[..., 1]
    return torch.stack((c * x + s * y, -s * x + c * y, v[..., 2]), -1)


class Frame:
    """One evaluation: states, controls, per-env constants and context."""

    def __init__(self, model, s_cmd, s_end, action, qcmd_raw, qexec, qexec_prev, ctx, ids):
        self.m, self.ctx, self.ids = model, ctx, ids
        self.s_cmd, self.s_end = s_cmd, s_end
        self.action, self.qcmd_raw = action, qcmd_raw
        self.qexec, self.qexec_prev = qexec, qexec_prev

    def const(self, name):
        value = self.m.const[name]
        return value[self.ids] if self.ids is not None else value

    def _bodies(self, s):
        return self.m.kin(
            S.part(s, "p"),
            S.part(s, "quat"),
            S.part(s, "q"),
            S.part(s, "v"),
            S.part(s, "wb"),
            S.part(s, "qd"),
            com_offset=self.const("ipos"),
        )

    @cached_property
    def end(self):
        return self._bodies(self.s_end)

    @cached_property
    def cmd(self):
        return self._bodies(self.s_cmd)

    # --- robot.data views at s_end -------------------------------------------------------
    @cached_property
    def q(self):
        return S.part(self.s_end, "q")

    @cached_property
    def qd(self):
        return S.part(self.s_end, "qd")

    @cached_property
    def root_pos(self):
        return S.part(self.s_end, "p")

    @cached_property
    def root_quat(self):
        return S.part(self.s_end, "quat")

    @cached_property
    def root_rot(self):
        return self.end.rot[:, 0]

    @cached_property
    def root_lin_vel(self):
        """Isaac ``root_lin_vel_w``: COM velocity of the root body."""
        return self.end.com_lin_vel[:, 0]

    @cached_property
    def gravity_b(self):
        return rotate_inverse(self.root_rot, self.root_rot.new_tensor(GRAVITY).expand_as(self.root_pos))

    @cached_property
    def racket(self):
        """``RacketTargetCommand._compute_racket_state`` at ``s_cmd``."""
        m = self.m
        b = self.cmd
        if m.racket_mode == "body":
            i = m.racket_body
            pos, R, lin = b.pos[:, i], b.rot[:, i], b.com_lin_vel[:, i]
        else:
            i = m.wrist_body
            offset = (b.rot[:, i] @ self.const("mount_offset")[..., None])[..., 0]
            pos = b.pos[:, i] + offset
            lin = b.com_lin_vel[:, i] + torch.cross(b.ang_vel[:, i], offset, dim=-1)
            R = b.rot[:, i] @ quat_to_matrix(self.const("mount_quat"))
        normal = R[..., :, m.mount_axis] * self.ctx["normal_sign"][:, None]
        return SimpleNamespace(pos=pos, lin_vel=lin, normal=normal)

    @cached_property
    def footwork(self):
        """``stroke_learning.footwork`` debt at ``s_end``."""
        from whole_body_tracking.utils.stroke_learning import footwork_debt

        actual = self.end.pos[:, self.m.feet]
        reference = self.ctx["footwork_ref"]
        error = actual - reference
        width = (actual[:, 0, 1] - actual[:, 1, 1]) - (reference[:, 0, 1] - reference[:, 1, 1])
        return footwork_debt(error, width)

    @cached_property
    def contact_scale(self):
        scale = self.ctx["contact_reward_scale"]
        if not self.m.footwork_scaled_contact:
            return scale
        return scale * (0.25 + 0.75 / (1 + self.footwork))

    def clock(self):
        """Attribute view used by the task's own pure gate helpers."""
        c = self.ctx
        return SimpleNamespace(
            time_to_strike=c["tts"],
            pre_strike=c["pre_strike"],
            strike_window=c["strike_window"],
            swing_sign=c["swing_sign"],
            cfg=self.m.command_cfg,
        )


class TaskModel(torch.nn.Module):
    """Builds constants from the live task, records context and evaluates heads."""

    def __init__(self, env, backend):
        super().__init__()
        from supermimic_hope import a3
        from whole_body_tracking.utils.full_body_hold import ACTIVE_NAMES, BODY_NAMES, GROUPS

        self.env, self.backend = env, backend
        robot = env.scene["robot"]
        cmd = env.command_manager.get_term("racket_target")
        motion = env.command_manager.get_term("motion")
        action = env.action_manager.get_term("joint_pos")
        self.command_cfg = cmd.cfg
        if list(robot.joint_names) != list(a3.JOINTS):
            raise ValueError("Analytic heads assume Isaac joint order == deploy order")
        ids = action._action_joint_ids.detach().cpu().tolist()
        if ids != list(range(31)):
            raise ValueError("Analytic heads assume the action term covers joints in order")
        self.kin = A3Kinematics(backend.model, list(robot.body_names), list(robot.joint_names)).to(env.device)
        self.dt = float(env.step_dt)
        body = {n: i for i, n in enumerate(robot.body_names)}
        self.feet = [body["left_ankle_roll_Link"], body["right_ankle_roll_Link"]]
        self.anchor_body = int(motion.robot_anchor_body_index)
        self.hold_bodies = [cmd._resolve_body_index(n) for n in BODY_NAMES]
        joint = {n: i for i, n in enumerate(robot.joint_names)}
        self.groups = {g: [joint[n] for n in names] for g, names in GROUPS.items()}
        self.active_ids = [joint[n] for n in ACTIVE_NAMES]
        self.waist_pitch_column = ACTIVE_NAMES.index("waist_pitch_joint")
        self.passive = action._passive_action_cols.detach().cpu().tolist()
        self.racket_mode = cmd._racket_mode
        self.racket_body = int(cmd._racket_body_index)
        self.wrist_body = int(cmd._wrist_body_index)
        self.mount_axis = int(cmd.cfg.mount_normal_axis)
        from whole_body_tracking.utils.stroke_learning import enabled as imitation_first

        self.footwork_scaled_contact = bool(hasattr(cmd, "_sc_level") and imitation_first(motion))
        n = env.num_envs
        rows = torch.arange(n, device=env.device)
        data = backend.data
        import warp as wp

        self.const = dict(
            origin=env.scene.env_origins.clone(),
            ipos=wp.to_torch(backend.m.body_ipos)[:, backend.bids].clone(),
            masses=wp.to_torch(backend.m.body_mass)[:, backend.bids].clone(),
            default_q=data.default_joint_pos.clone(),
            soft=data.soft_joint_pos_limits.clone(),
            hard=data.joint_pos_limits.clone(),
            effort=data.joint_effort_limits.clone(),
            offset=action._decoder_parameter_rows(action._offset, rows, "offset").clone(),
            scale=action._decoder_parameter_rows(action._scale, rows, "scale").clone(),
            mount_offset=cmd._mount_offset.clone(),
            mount_quat=cmd._mount_quat.clone(),
            kp=data.joint_stiffness.clone(),
            kd=data.joint_damping.clone(),
        )
        a_lo = (self.const["soft"][..., 0] - self.const["offset"]) / self.const["scale"]
        a_hi = (self.const["soft"][..., 1] - self.const["offset"]) / self.const["scale"]
        self.const["raw_lo"], self.const["raw_hi"] = torch.minimum(a_lo, a_hi), torch.maximum(a_lo, a_hi)
        self.const["mass_weights"] = self.const["masses"] / self.const["masses"].sum(-1, keepdim=True)
        from whole_body_tracking.utils.task_pose_reference import env_reference, reference_joints

        pose = env_reference(env)
        self.hold_ref = reference_joints(list(ACTIVE_NAMES), data.joint_pos, pose)
        self.hold_metrics = pose["metrics"]
        self.terms = list(env.reward_manager.active_terms)
        self.cfgs = {t: env.reward_manager.get_term_cfg(t) for t in self.terms}
        self.weights = torch.tensor([float(self.cfgs[t].weight) for t in self.terms], device=env.device)
        self.analytic = {t: getattr(self, "r_" + t) for t in self.terms if hasattr(self, "r_" + t)}
        self.termination_index = self.terms.index("termination_penalty")
        self._limit_cache()
        self._qdes_rate_constants(cmd, action)
        manager = env.observation_manager
        offset, layout = 0, {}
        for name, shape in zip(manager.active_terms["policy"], manager.group_obs_term_dim["policy"]):
            layout[name] = slice(offset, offset + math.prod(shape))
            offset += math.prod(shape)
        expected = ["base_ang_vel", "joint_pos", "joint_vel", "actions", "projected_gravity",
                    "base_forward_xy", "base_target_delta_xy", "racket_target_rel_base",
                    "racket_target_vel_w", "time_to_strike"]
        if list(layout) != expected or offset != 110:
            raise ValueError(f"Analytic observation head expects hitter_pure, got {list(layout)}")
        self.obs_layout, self.action_slice = layout, layout["actions"]

    # ------------------------------------------------------------------ constants
    def _limit_cache(self):
        env = self.env
        cfg = self.cfgs["actual_joint_limit"].params
        action = env.action_manager.get_term("joint_pos")
        names = list(action._joint_names)
        columns = [i for i in range(31) if i not in set(self.passive)]
        active = [names[i] for i in columns]
        waist = [active.index(f"waist_{a}_joint") for a in ("yaw", "roll", "pitch")]
        hard = self.const["hard"][:, columns]
        span = hard[..., 1] - hard[..., 0]
        requested = hard.new_full((len(columns),), float(cfg["margin_rad"]))
        requested[waist] = float(cfg["waist_margin_rad"])
        self.limit_columns, self.limit_waist = columns, waist
        self.const["limit_hard"] = hard
        self.const["limit_margin"] = torch.minimum(requested[None, :], 0.2 * span)

    def ramp(self, start, ramp):
        return min(1.0, max(0.0, (self.env.common_step_counter - start) / max(1, ramp)))

    # ------------------------------------------------------------------ capture
    @torch.no_grad()
    def context(self):
        """Exogenous task context at reward time (after physics, before reset)."""
        from whole_body_tracking.utils.reference_recovery import base_reference, static_gate
        from whole_body_tracking.utils.ready_recovery import recovery_gate
        from whole_body_tracking.utils.task_pose_reference import phase_gate
        from whole_body_tracking.utils.four_stroke import complete_reference
        from whole_body_tracking.utils.stroke_learning import reward_scale, teacher_envelope, enabled
        from whole_body_tracking.utils.stroke_tracking import joint_reference, elbow_extension_limit
        from whole_body_tracking.tasks.tracking.mdp.hope_rewards import _position_guidance_gate

        env = self.env
        cmd = env.command_manager.get_term("racket_target")
        motion = env.command_manager.get_term("motion")
        action = env.action_manager.get_term("joint_pos")
        n = env.num_envs
        ones = torch.ones(n, device=env.device)
        tts = cmd.time_to_strike.clone()
        in_hold = motion.in_hold.clone()
        complete = complete_reference(motion, cmd).clone()
        c = dict(
            tts=tts,
            pre_strike=cmd.pre_strike.clone(),
            strike_window=cmd.strike_window.clone(),
            swing_sign=cmd.swing_sign.clone(),
            in_hold=in_hold,
            complete=complete,
            base_target=cmd.base_target_pos_w.clone(),
            rt_pos=cmd.racket_target_pos_w.clone(),
            rt_vel=cmd.racket_target_vel_w.clone(),
            rt_normal=cmd.racket_target_normal_w.clone(),
        )
        held_metric = motion.metrics.get("in_hold") if hasattr(motion, "metrics") else None
        c["in_hold_ready"] = (held_metric > 0.5) if held_metric is not None else in_hold
        ref_pos, ref_vel = base_reference(env, cmd)
        c["base_ref_pos"], c["base_ref_vel"] = ref_pos.clone(), ref_vel.clone()
        c["rt_moving"] = cmd.racket_target_pos_w - cmd.racket_target_vel_w * tts[:, None]
        c["pos_gate"] = _position_guidance_gate(cmd).clone()
        c["contact_reward_scale"] = ones * float(reward_scale(cmd, "contact"))
        c["motion_reward_scale"] = ones * float(reward_scale(cmd, "motion"))
        if cmd._mount_sign_per_clip_t is not None and motion._multiseg:
            clip = motion.clip_id.clamp(max=cmd._mount_sign_per_clip_t.shape[0] - 1)
            c["normal_sign"] = cmd._mount_sign_per_clip_t[clip].float().clone()
        else:
            c["normal_sign"] = ones * float(cmd.cfg.mount_normal_sign)
        c["recovery_gate"] = recovery_gate(in_hold, tts, motion).clone()
        c["phase_gate"] = phase_gate(in_hold, tts)[0].clone()
        # References in the environment frame (feet and torso).
        names = ["left_ankle_roll_Link", "right_ankle_roll_Link"]
        ref_feet = [motion.cfg.body_names.index(x) for x in names]
        home = motion.motion.body_pos_w[0, ref_feet] + env.scene.env_origins[:, None, :]
        feet = motion.body_pos_w[:, ref_feet]
        valid = complete & ~in_hold
        c["footwork_ref"] = torch.where(valid[:, None, None], feet, home).clone()
        c["track_foot_ref"] = torch.where(complete[:, None, None], feet, home).clone()
        station = torch.where(in_hold[:, None, None], home, feet)
        station = torch.where((~complete)[:, None, None], home, station)
        c["station_foot_ref"] = station.clone()
        from whole_body_tracking.utils.full_body_hold import rpy

        ref_yaw = rpy(motion.body_quat_w[:, ref_feet])[..., 2]
        c["station_foot_yaw"] = torch.where((in_hold | ~complete)[:, None], torch.zeros_like(ref_yaw), ref_yaw).clone()
        arm = self.groups["right_arm"]
        target, target_vel, complete_ref, valid_ref = joint_reference(motion, cmd, arm)
        c["q_target"], c["qd_target"] = target.clone(), target_vel.clone()
        c["track_gate"] = (1 - static_gate(motion, tts)).clone()
        recovery = ((-tts - 0.15) / 0.70).clamp(0, 1)
        c["arm_factor"] = torch.where(valid_ref, torch.ones_like(recovery), recovery).clone()
        c["track_complete"] = complete_ref.float().clone()
        desired = rpy(motion.anchor_quat_w)[:, :2]
        c["torso_desired"] = torch.where(complete_ref[:, None], desired, torch.zeros_like(desired)).clone()
        # q_des envelopes from the published teacher only.
        soft = self.const["soft"]
        hard = self.const["hard"]
        p = self.cfgs["rally_ankle_qdes_saturation"].params
        ankle = list(p["asset_cfg"].joint_ids)
        safe = max(float(p["safe_abs"]), 0.0)
        lo = torch.maximum(hard[:, ankle, 0], torch.full_like(hard[:, ankle, 0], -safe))
        hi = torch.minimum(hard[:, ankle, 1], torch.full_like(hard[:, ankle, 1], safe))
        lo, hi = teacher_envelope(cmd, ankle, lo, hi, soft[:, ankle, 0], soft[:, ankle, 1])
        c["ankle_lo"], c["ankle_hi"] = lo.clone(), hi.clone()
        p = self.cfgs["rally_joint_qdes_saturation"].params
        span = (hard[..., 1] - hard[..., 0]).clamp_min(1.0e-6)
        lo = hard[..., 0] + float(p["safe_margin_fraction"]) * span
        hi = hard[..., 1] - float(p["safe_margin_fraction"]) * span
        if enabled(motion):
            ids = torch.arange(31, device=env.device)
            lo, hi = teacher_envelope(cmd, ids, lo, hi, action._safe_lo, action._safe_hi)
        c["barrier_lo"], c["barrier_hi"] = lo.clone(), hi.clone()
        p = self.cfgs["right_elbow_extension_debt"].params
        elbow = list(p["asset_cfg"].joint_ids)[0]
        limit = ones * float(p["extension_start"])
        if hasattr(cmd, "_sc_totals"):
            limit = elbow_extension_limit(motion, cmd, elbow, float(p["extension_start"]))
        c["elbow_limit"] = limit.clone()
        # Contacts are taken from the real step (not differentiable).
        sensor = env.scene.sensors["contact_forces"]
        sp = self.cfgs["settle_foot_slip"].params
        force = sensor.data.net_forces_w
        slip_ids = list(sp["sensor_cfg"].body_ids)
        c["slip_contact"] = (force[:, slip_ids].norm(dim=-1) > float(sp["force_threshold"])).float()
        dp = self.cfgs["dynamic_support"].params
        ds_ids = list(dp["sensor_cfg"].body_ids)
        c["support_loaded"] = force[:, ds_ids, 2] > float(dp.get("force_threshold", 10.0))
        quiet_feet, _ = sensor.find_bodies(list(("left_ankle_roll_Link", "right_ankle_roll_Link")), preserve_order=True)
        c["loaded20"] = (force[:, quiet_feet, 2] > 20).all(-1)
        scale = float(dp.get("demonstrated_step_scale", 1.0))
        airborne = torch.zeros_like(in_hold)
        if scale != 1.0:
            ref_z = motion.motion.body_pos_w[0, ref_feet, 2] + env.scene.env_origins[:, None, 2]
            airborne = (motion.body_pos_w[:, ref_feet, 2] - ref_z).amax(-1) > 0.015
            airborne &= complete & ~in_hold
        c["support_scale"] = torch.where(airborne, scale * ones, ones)
        # executed q_des: only the strike-protected upper-arm window is exogenous.
        rate = self.cfgs["executed_qdes_rate"].params
        c["qdes_protected"] = (
            (tts <= float(rate["strike_free_pre_s"])) & (tts >= -float(rate["follow_through_free_s"])) & ~in_hold
        )
        # HOLD dwell bookkeeping before this step's update.
        tracker = getattr(env, "_complete_hold_dwell", None)
        if tracker is None or tracker.step is None:
            c["dwell_prev"] = torch.zeros(n, device=env.device)
            c["ready_prev"] = torch.zeros(n, dtype=torch.bool, device=env.device)
            c["dwell_reset"] = torch.ones(n, dtype=torch.bool, device=env.device)
        else:
            reset = env.episode_length_buf <= tracker.episode_step
            if tracker.station is not None:
                reset |= (cmd.base_target_pos_w - tracker.station).abs().amax(-1) > 1e-6
            consecutive = env.common_step_counter == tracker.step + 1
            c["dwell_prev"] = tracker.elapsed.clone()
            c["ready_prev"] = tracker.was_ready.clone()
            c["dwell_reset"] = reset | (not consecutive)
        # Termination references.
        c["anchor_ref_z"] = motion.anchor_pos_w[:, 2].clone()
        ref_quat = motion.anchor_quat_w
        c["anchor_ref_gz"] = (-quat_to_matrix(ref_quat)[..., 2, 2]).clone()
        from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes

        tm = env.termination_manager
        ee = tm.get_term_cfg("ee_body_pos")
        bodies = _get_body_indexes(motion, ee.params.get("body_names"))
        c["ee_ref_z"] = motion.body_pos_relative_w[:, bodies, 2].clone()
        # Scalars that ramp with real environment steps.
        for name in ("fixed_station_feet", "full_body_hold", "full_body_quiet", "station_recovery"):
            p = self.cfgs[name].params
            c["ramp_" + name] = ones * self.ramp(int(p["start_steps"]), int(p["ramp_steps"]))
        # Latency for the world model: the command arrives ``cmd_delay`` physics
        # substeps into this tick (drawn every tick by TrainingLatency, 0 for the
        # nominal cohort); ``qdes_delay`` is the episode's 0/1-tick FIFO.
        latency = getattr(action, "_training_latency", None)
        if latency is not None and int(latency.sensor_max):
            raise NotImplementedError("delayed proprioception (sensor_max_substeps > 0) is not ported")
        c["cmd_delay"] = (latency.command_phase.float().clone() if latency is not None
                          else torch.zeros(n, device=env.device))
        c["qdes_delay"] = action._qdes_delay_steps.float().clone()
        # Command histories owned by the action term (carried inside imagination).
        c["action_prev"] = env.action_manager.prev_action.detach().clone()
        c["dq_prev"] = action._previous_qdes_delta.detach().clone()
        for name in ("qdes_reversal", "qdes_torque_step"):
            p = self.cfgs[name].params
            steps = int(p.get("ramp_steps", 32000))
            c["ramp_" + name] = ones * (min(1.0, float(env.common_step_counter) / max(1, steps)) if steps else 1.0)
            c["protected_" + name] = (
                ~in_hold & (tts <= float(p.get("strike_free_pre_s", 0.12)))
                & (tts >= -float(p.get("follow_through_free_s", 0.30)))
            )
        p = self.cfgs["strike_preserving_action_rate"].params
        c["protected_action_rate"] = (
            (tts <= float(p["strike_free_pre_s"])) & (tts >= -float(p["follow_through_free_s"])) & ~in_hold
        )
        return c

    def _qdes_rate_constants(self, cmd, action):
        from whole_body_tracking.tasks.tracking.mdp.hope_actions import resolve_action_columns_by_joint_name

        p = self.cfgs["executed_qdes_rate"].params
        effective = getattr(action, "_upper_intervention_effective", None)
        if torch.is_tensor(effective) and bool(effective.any()):
            raise NotImplementedError("upper-arm intervention masks are not ported")
        live = torch.ones(31, device=cmd.device)
        if self.passive:
            live[self.passive] = 0
        upper = torch.zeros(31, device=cmd.device)
        cols = resolve_action_columns_by_joint_name(action, tuple(p["upper_joint_names"]))
        if cols:
            upper[cols] = 1
        self.qdes_live, self.qdes_upper = live, upper
        # Per-term strike-arm columns (each term names its own joints).
        self.term_upper = {}
        for name, key in (("qdes_reversal", "upper_joint_names"), ("qdes_torque_step", "upper_joint_names"),
                          ("strike_preserving_action_rate", "strike_joint_names")):
            mask = torch.zeros(31, dtype=torch.bool, device=cmd.device)
            cols = resolve_action_columns_by_joint_name(action, tuple(self.cfgs[name].params.get(key, ())))
            if cols:
                mask[cols] = True
            self.term_upper[name] = mask
        self.qdes_hold_multiplier = float(p["hold_multiplier"])
        self.const["inv_scale_sq"] = 1 / self.const["scale"].abs().clamp_min(1.0e-6).square()

    # ------------------------------------------------------------------ controls
    def qdes_unclamped(self, action, ids=None):
        offset = self.const["offset"] if ids is None else self.const["offset"][ids]
        scale = self.const["scale"] if ids is None else self.const["scale"][ids]
        return offset + scale * action

    def qdes_project(self, action, ids=None):
        """Clamp raw actions into the interval whose q_des is not clipped."""
        lo = self.const["raw_lo"] if ids is None else self.const["raw_lo"][ids]
        hi = self.const["raw_hi"] if ids is None else self.const["raw_hi"][ids]
        return torch.minimum(torch.maximum(action, lo), hi)

    def qdes_command(self, action, ids=None):
        raw = self.qdes_unclamped(action, ids)
        soft = self.const["soft"] if ids is None else self.const["soft"][ids]
        default = self.const["default_q"] if ids is None else self.const["default_q"][ids]
        q = torch.minimum(torch.maximum(raw, soft[..., 0]), soft[..., 1])
        if self.passive:
            mask = torch.zeros(31, dtype=torch.bool, device=q.device)
            mask[self.passive] = True
            q = torch.where(mask, default, q)
        return q

    # ------------------------------------------------------------------ evaluation
    def frame(self, s_cmd, s_end, action, qexec, qexec_prev, ctx, ids=None):
        return Frame(self, s_cmd, s_end, action, self.qdes_unclamped(action, ids), qexec, qexec_prev, ctx, ids)

    def rewards(self, fr, only=None):
        """Unweighted analytic values for every implemented term (B, terms)."""
        values = []
        for t in self.terms:
            fn = self.analytic.get(t) if only is None or t in only else None
            values.append(fn(fr, self.cfgs[t].params) if fn is not None else torch.zeros_like(fr.ctx["tts"]))
        return torch.stack(values, -1)

    def heads(self, s_cmd, s_end, action, qexec, qexec_prev, ctx, ids, only=None):
        """Evaluation of all heads; returns tensors only."""
        fr = self.frame(s_cmd, s_end, action, qexec, qexec_prev, ctx, ids)
        values = self.rewards(fr, only)
        margins = self.margins(fr, self.reasons, self.reason_params)
        elapsed, ready = getattr(fr, "hold_state", (ctx["dwell_prev"], ctx["ready_prev"]))
        return values, margins, elapsed, ready

    def step_reward(self, values):
        """Weighted per-step reward of analytic values (excluding terminations)."""
        w = self.weights.clone()
        w[self.termination_index] = 0
        return (values * w).sum(-1) * self.dt

    # --- task terms ------------------------------------------------------------------------
    def r_joint_limit(self, fr, p):
        soft = fr.const("soft")
        return ((soft[..., 0] - fr.q).clamp_min(0) + (fr.q - soft[..., 1]).clamp_min(0)).sum(-1)

    def r_racket_position(self, fr, p):
        from whole_body_tracking.utils.task_tracking import tracking_kernel

        error = (fr.racket.pos - fr.ctx["rt_moving"]).square().sum(-1)
        raw = tracking_kernel(error, float(p["std"]), float(p.get("tail_blend", 0.0)))
        temporal = float(getattr(self.command_cfg, "position_guidance_temporal_scale", 1.0))
        return temporal * raw * fr.ctx["pos_gate"].float() * fr.contact_scale

    def _velocity_kernel(self, fr, std, tail):
        from whole_body_tracking.utils.task_tracking import tracking_kernel

        error = (fr.racket.lin_vel - fr.ctx["rt_vel"]).square().sum(-1)
        return tracking_kernel(error, float(std), float(tail)) * fr.ctx["strike_window"].float() * fr.contact_scale

    def _normal_kernel(self, fr, std):
        angle = safe_acos((fr.racket.normal * fr.ctx["rt_normal"]).sum(-1))
        return torch.exp(-angle.square() / float(std) ** 2) * fr.ctx["strike_window"].float() * fr.contact_scale

    def r_racket_velocity(self, fr, p):
        return self._velocity_kernel(fr, p["std"], p.get("tail_blend", 0.0))

    def r_racket_normal(self, fr, p):
        return self._normal_kernel(fr, p["std"])

    def r_base_position(self, fr, p):
        error = (fr.root_pos[:, :2] - fr.ctx["base_ref_pos"]).square().sum(-1)
        return torch.exp(-error / float(p["std"]) ** 2) * fr.ctx["pre_strike"].float()

    def r_racket_strike_success(self, fr, p):
        from whole_body_tracking.utils.task_tracking import tracking_kernel

        error = (fr.racket.pos - fr.ctx["rt_moving"]).square().sum(-1)
        rp = tracking_kernel(error, float(p["std_pos"])) * fr.ctx["pos_gate"].float()
        rv = self._velocity_kernel(fr, p["std_vel"], 0.0)
        rn = self._normal_kernel(fr, p["std_normal"])
        return rp * rv * rn / fr.contact_scale

    def r_racket_strike_precision(self, fr, p):
        """``racket_strike_precision_debt``: moving-path error inside the contact window."""
        error = (fr.racket.pos - fr.ctx["rt_moving"]).norm(dim=-1)
        gate = ~fr.ctx["in_hold"] & (fr.ctx["tts"].abs() <= float(p["window_s"]) + 1e-6)
        return deadband_huber(error, p["margin"], p["std"]) * gate.float()

    def r_strike_preserving_action_rate(self, fr, p):
        """``phase_gated_action_rate_l2``: raw-action difference, strike arm free, HOLD weighted."""
        free = fr.ctx["protected_action_rate"][:, None] & self.term_upper["strike_preserving_action_rate"]
        live = self.qdes_live * (1 - free.float())
        cost = ((fr.action - fr.ctx["action_prev"]).square() * live).sum(-1)
        return cost * torch.where(fr.ctx["in_hold"], float(p.get("hold_multiplier", 1.0)), 1.0)

    def _chatter(self, fr, p, name, mode):
        """``executed_qdes_chatter_debt`` on the executed (clamped, FIFO) command history."""
        from whole_body_tracking.utils.qdes_chatter_cost import command_chatter_cost

        live = self.qdes_live.bool() & ~(fr.ctx["protected_" + name][:, None] & self.term_upper[name])
        cost, _ = command_chatter_cost(
            fr.qexec - fr.qexec_prev, fr.ctx["dq_prev"], live, mode=mode,
            stiffness=fr.const("kp"), effort_limit=fr.const("effort"),
            reversal_deadband_rad=float(p.get("reversal_deadband_rad", 0.005)),
            reversal_scale_rad=float(p.get("reversal_scale_rad", 0.02)),
            torque_step_margin_fraction=float(p.get("torque_step_margin_fraction", 0.10)),
            torque_step_scale_fraction=float(p.get("torque_step_scale_fraction", 0.10)),
            worst_joint_blend=float(p.get("worst_joint_blend", 0.5)),
        )
        return fr.ctx["ramp_" + name] * cost

    def r_qdes_reversal(self, fr, p):
        return self._chatter(fr, p, "qdes_reversal", "reversal")

    def r_qdes_torque_step(self, fr, p):
        return self._chatter(fr, p, "qdes_torque_step", "torque_step")

    def r_joint_torques(self, fr, p):
        # PD estimate from the executed target; the recorded correction makes
        # the value exact at real states (MuJoCo reports the last substep).
        tau = fr.const("kp") * (fr.qexec - fr.q) - fr.const("kd") * fr.qd
        effort = fr.const("effort")
        tau = torch.maximum(torch.minimum(tau, effort), -effort)
        return tau.square().sum(-1)

    def r_joint_vel(self, fr, p):
        return fr.qd.square().sum(-1)

    def r_upright(self, fr, p):
        return fr.gravity_b[:, :2].square().sum(-1)

    def r_base_ang_vel_xy(self, fr, p):
        return S.part(fr.s_end, "wb")[:, :2].square().sum(-1)

    def r_base_lin_vel_z(self, fr, p):
        return rotate_inverse(fr.root_rot, fr.root_lin_vel)[:, 2].square()

    def r_hold_heading(self, fr, p):
        yaw = yaw_of(fr.root_quat)
        return torch.exp(-yaw.square() / float(p["std"]) ** 2) * fr.ctx["in_hold"].float()

    def r_post_strike_x_settle(self, fr, p):
        from whole_body_tracking.tasks.tracking.mdp.hope_rewards import _post_strike_window

        x = fr.root_pos[:, 0] - fr.ctx["base_ref_pos"][:, 0]
        gate = _post_strike_window(fr.clock(), float(p["t_hi"]), float(p.get("t_lo", 0.0)))
        return torch.exp(-x.square() / float(p["std"]) ** 2) * gate.float()

    def r_post_strike_vx_quiet(self, fr, p):
        from whole_body_tracking.tasks.tracking.mdp.hope_rewards import _post_strike_window

        vx = fr.root_lin_vel[:, 0] - fr.ctx["base_ref_vel"][:, 0]
        gate = _post_strike_window(fr.clock(), float(p["t_hi"]), float(p.get("t_lo", 0.0)))
        return torch.exp(-vx.square() / float(p["std"]) ** 2) * gate.float()

    def r_rally_ankle_qdes_saturation(self, fr, p):
        ankle = list(p["asset_cfg"].joint_ids)
        qdes = fr.qcmd_raw[:, ankle]
        excess = (fr.ctx["ankle_lo"] - qdes).clamp_min(0) + (qdes - fr.ctx["ankle_hi"]).clamp_min(0)
        raw = huber(excess / max(float(p["std"]), 1e-6)).mean(-1)
        tts = fr.ctx["tts"]
        return raw * ((tts < float(p["t_pre"])) & (tts > -float(p["t_post"]))).float()

    def r_racket_normal_alignment_debt(self, fr, p):
        cos = (fr.racket.normal * fr.ctx["rt_normal"]).sum(-1).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        excess = torch.clamp(torch.acos(cos) - float(p["margin"]), min=0.0)
        debt = huber(excess / float(p["std"]))
        return debt * fr.ctx["strike_window"].float() * fr.ctx["contact_reward_scale"]

    def r_windup_x_recovery(self, fr, p):
        signed = fr.root_pos[:, 0] - fr.ctx["base_ref_pos"][:, 0]
        x_error = signed.abs()
        position = deadband_huber(x_error, p["x_margin"], p["x_std"])
        vx = fr.root_lin_vel[:, 0] - fr.ctx["base_ref_vel"][:, 0]
        charged = torch.where(x_error > float(p["x_margin"]), torch.sign(signed) * vx, vx.abs())
        velocity = deadband_huber(charged, p["vx_margin"], p["vx_std"])
        blend = float(p["position_blend"])
        tts = fr.ctx["tts"]
        gate = (tts > float(self.command_cfg.strike_window_s)) & (tts < float(p["t_hi"]))
        return (blend * position + (1 - blend) * velocity) * gate.float()

    def r_pre_strike_station_settle(self, fr, p):
        delta = fr.ctx["base_ref_pos"] - fr.root_pos[:, :2]
        dist = delta.norm(dim=-1)
        speed = (float(p["v_gain"]) * dist).clamp(0.0, float(p["v_max"]))
        desired = fr.ctx["base_ref_vel"] + delta / dist.clamp_min(1e-6)[:, None] * speed[:, None]
        err = (fr.root_lin_vel[:, :2] - desired).norm(dim=-1)
        if p.get("debt_huber", False):
            raw = deadband_huber(err, p.get("velocity_margin", 0.0), p["std"])
        else:
            raw = torch.exp(-err.square() / float(p["std"]) ** 2)
        tts = fr.ctx["tts"]
        gate = (tts > float(self.command_cfg.strike_window_s)) & (tts < float(p["t_max"]))
        if p.get("t_min") is not None:
            gate &= tts > float(p["t_min"])
        return raw * gate.float()

    def r_passive_head_raw_action(self, fr, p):
        scaled = fr.action[:, self.passive] / float(p["std"])
        if p.get("loss_type", "huber") == "l2":
            return scaled.square().mean(-1)
        m, d = scaled.abs(), float(p["huber_delta"])
        return torch.where(m <= d, 0.5 * m.square(), d * (m - 0.5 * d)).mean(-1)

    def r_rally_joint_qdes_saturation(self, fr, p):
        hard = fr.const("hard")
        span = (hard[..., 1] - hard[..., 0]).clamp_min(1.0e-6)
        qdes = fr.qcmd_raw
        excess = (fr.ctx["barrier_lo"] - qdes).clamp_min(0) + (qdes - fr.ctx["barrier_hi"]).clamp_min(0)
        per_joint = huber(excess / (float(p["std_fraction"]) * span))
        k = min(int(p["topk"]), per_joint.shape[-1])
        blend = float(p["topk_blend"])
        return (1 - blend) * per_joint.mean(-1) + blend * torch.topk(per_joint, k=k, dim=-1).values.mean(-1)

    def r_strike_x_drift(self, fr, p):
        x_err = (fr.root_pos[:, 0] - fr.ctx["base_target"][:, 0]).abs()
        if p.get("huber_tail", False):
            raw = deadband_huber(x_err, p["margin"], p["std"])
        else:
            raw = 1.0 - torch.exp(-((x_err - float(p["margin"])).clamp_min(0) / float(p["std"])).square())
        tts = fr.ctx["tts"]
        return raw * ((tts < float(p["t_pre"])) & (tts > -float(p["t_post"]))).float()

    def r_right_elbow_extension_debt(self, fr, p):
        joint = list(p["asset_cfg"].joint_ids)[0]
        debt = deadband_huber(fr.q[:, joint] - fr.ctx["elbow_limit"], 0.0, p["std"])
        tts = fr.ctx["tts"]
        gate = (tts < float(p["t_pre"])) & (tts > -float(p["t_post"]))
        if p.get("forehand_only", False):
            gate &= fr.ctx["swing_sign"] > 0.0
        return debt * gate.float()

    def r_ready_leg_settle(self, fr, p):
        joints = list(p["asset_cfg"].joint_ids)
        speed = fr.qd[:, joints].square().mean(-1).sqrt()
        debt = deadband_huber(speed, p["margin"], p["std"])
        station = (fr.root_pos[:, :2] - fr.ctx["base_target"]).norm(dim=-1)
        gate = fr.ctx["in_hold_ready"] & (station < float(p["station_reach"]))
        gate &= yaw_of(fr.root_quat).abs() < float(p["heading_gate"])
        return debt * gate.float()

    def _side(self, fr, p, debt):
        return torch.where(
            fr.ctx["swing_sign"] > 0.0,
            torch.full_like(debt, float(p["forehand_scale"])),
            torch.full_like(debt, float(p["backhand_scale"])),
        )

    def r_strike_x_gate_margin(self, fr, p):
        x = (fr.root_pos[:, 0] - fr.ctx["base_ref_pos"][:, 0]).abs()
        debt = deadband_huber(x, p["margin"], p["std"])
        gate = fr.ctx["tts"].abs() <= float(p["half_window_s"])
        return debt * self._side(fr, p, debt) * gate.float()

    def r_post_swing_xlock(self, fr, p):
        x = (fr.root_pos[:, 0] - fr.ctx["base_ref_pos"][:, 0]).abs()
        debt = deadband_huber(x, p["margin"], p["std"])
        tts = fr.ctx["tts"]
        gate = (tts < -float(p["t_lo"])) & (tts > -float(p["t_hi"]))
        return debt * self._side(fr, p, debt) * gate.float()

    def r_post_swing_base_quiet(self, fr, p):
        speed = (fr.root_lin_vel[:, :2] - fr.ctx["base_ref_vel"]).norm(dim=-1)
        if p.get("huber_tail", False):
            raw = deadband_huber(speed, p.get("margin", 0.0), p["std"])
        else:
            raw = 1.0 - torch.exp(-speed.square() / float(p["std"]) ** 2)
        tts = fr.ctx["tts"]
        return raw * ((tts < -float(p["t_lo"])) & (tts > -float(p["t_hi"]))).float()

    def r_settle_foot_slip(self, fr, p):
        ids = list(p["asset_cfg"].body_ids)
        contact = fr.ctx["slip_contact"]
        speed = fr.end.com_lin_vel[:, ids, :2].norm(dim=-1)
        slip = (speed * contact).sum(-1) / contact.sum(-1).clamp(min=1.0)
        if p.get("huber_tail", False):
            raw = deadband_huber(slip, p.get("margin", 0.0), p["std"])
        else:
            raw = 1.0 - torch.exp(-(slip / float(p["std"])).square())
        if p.get("all_phases", False):
            return raw
        tts = fr.ctx["tts"]
        station = (fr.root_pos[:, :2] - fr.ctx["base_target"]).norm(dim=-1)
        arrived = (station < float(p["station_reach"])) & (tts > -float(p["strike_t_post"])) & (tts < float(p["pre_t_max"]))
        post = (tts < -float(p["post_t_lo"])) & (tts > -float(p["post_t_hi"]))
        return raw * (arrived | post).float()

    def r_post_swing_tilt_debt(self, fr, p):
        tilt = fr.gravity_b[:, :2].norm(dim=-1)
        debt = huber(torch.clamp(tilt - float(p["margin"]), min=0.0) / float(p["std"]))
        tts = fr.ctx["tts"]
        gate = (tts < -float(p["t_lo"])) & (tts > -float(p["t_hi"]))
        if p.get("include_replay", False) or p.get("curriculum_scaled", False):
            raise NotImplementedError("replay/curriculum tilt debt is not ported")
        return debt * gate.float()

    def r_full_stroke_tracking(self, fr, p):
        from whole_body_tracking.utils.stroke_tracking import tracking_score
        from whole_body_tracking.utils.fixed_station import foot_reference_debt
        from whole_body_tracking.utils.full_body_hold import wrap_angle

        c = fr.ctx
        pose_scale, velocity_scale = float(p["pose_scale"]), float(p["velocity_scale"])
        velocity_blend, worst = float(p["velocity_blend"]), float(p["worst_group_blend"])
        gate, arm_factor, complete = c["track_gate"], c["arm_factor"], c["track_complete"]
        pose, velocity = [], []
        for group, ids in self.groups.items():
            error = fr.q[:, ids] - c["q_target"][:, ids]
            rate = fr.qd[:, ids] - c["qd_target"][:, ids]
            factor = arm_factor if group == "right_arm" else torch.ones_like(gate)
            scale = pose_scale * (0.18 / 0.28) if group == "waist" else pose_scale
            pose.append(tracking_score(error, scale, 0.025) * factor)
            velocity.append(tracking_score(rate, velocity_scale, 0.15, tail=0.03) * factor * complete)
        actual = rpy(fr.end.quat[:, self.anchor_body])[:, :2]
        torso_error = wrap_angle(actual - c["torso_desired"])
        pose_scores = torch.stack(pose, -1)
        pose_score = pose_scores.mean(-1)
        if worst:
            eligible = torch.cat((pose_scores[:, :-1], pose_scores[:, -1:] + (1 - arm_factor)[:, None]), -1)
            pose_score = (1 - worst) * pose_score + worst * eligible.amin(-1)
        velocity_scores = torch.stack(velocity, -1)
        velocity_score = velocity_scores.mean(-1)
        if worst:
            extra = ((1 - arm_factor) * complete)[:, None]
            eligible = torch.cat((velocity_scores[:, :-1], velocity_scores[:, -1:] + extra), -1)
            velocity_score = (1 - worst) * velocity_score + worst * eligible.amin(-1)
        score = (
            (0.90 - velocity_blend) * pose_score
            + velocity_blend * velocity_score
            + 0.10 * tracking_score(torso_error, 0.12, 0.015)
        ) * gate
        foot_blend = float(p["foot_blend"])
        if foot_blend:
            debt = foot_reference_debt(fr.end.pos[:, self.feet] - c["track_foot_ref"])
            foot_score = 1 / (1 + debt) - 0.08 * debt
            score = (1 - foot_blend) * score + foot_blend * foot_score * gate
        if self.footwork_scaled_contact:
            foot = fr.footwork
            score = score * (0.25 + 0.75 / (1 + foot)) - 0.15 * foot * gate
        return score * c["motion_reward_scale"]

    def r_fixed_station_feet(self, fr, p):
        from whole_body_tracking.utils.fixed_station import foot_reference_debt
        from whole_body_tracking.utils.full_body_hold import wrap_angle

        if not p.get("persistent_home", False):
            raise NotImplementedError("only persistent-home station feet are ported")
        error = fr.end.pos[:, self.feet] - fr.ctx["station_foot_ref"]
        debt = foot_reference_debt(error)
        cs, ys = float(p.get("clearance_std", 0.0)), float(p.get("yaw_std", 0.0))
        if cs > 0.0 and ys > 0.0:
            height = huber(((error[..., 2].abs() - 0.005) / cs).clamp_min(0)).mean(-1)
            actual_yaw = rpy(fr.end.quat[:, self.feet])[..., 2]
            yaw = (wrap_angle(actual_yaw - fr.ctx["station_foot_yaw"]).abs() - 0.035) / ys
            yaw = huber(yaw.clamp_min(0)).mean(-1)
            debt = float(p["position_fraction"]) * debt + float(p.get("height_fraction", 0.25)) * height + 0.15 * yaw
        return debt * fr.ctx["ramp_fixed_station_feet"]

    def hold_terms(self, fr, p):
        """Shared ``full_body_hold`` geometry; returns reward and next dwell state."""
        from whole_body_tracking.utils.full_body_hold import (
            joint_costs, mean_and_worst, approach_factor, wrap_angle,
        )
        from whole_body_tracking.utils.ready_recovery import bounded_debt
        from whole_body_tracking.utils.complete_hold import weakest_score

        c, metrics = fr.ctx, self.hold_metrics
        gate, scale = c["recovery_gate"], c["ramp_full_body_hold"]
        q = fr.q[:, self.active_ids]
        error = q - self.hold_ref
        costs, _, _ = joint_costs(error)
        distance = (c["base_target"] - fr.root_pos[:, :2]).norm(dim=-1)
        leg = approach_factor(distance)
        one = torch.ones_like(gate)
        factors = torch.stack((leg, leg, one, one, one), -1)
        joints = mean_and_worst(costs * factors + (1 - factors))
        bodies = self.hold_bodies
        pos = fr.end.pos[:, bodies] - fr.const("origin")[:, None, :]
        angles = rpy(fr.end.quat[:, bodies])
        torso_ref = q.new_tensor(metrics["torso_rpy_deg"]) * math.pi / 180
        torso_error = wrap_angle(angles[:, 0] - torso_ref)
        torso = mean_and_worst(bounded_debt(torso_error, float(p["torso_margin"]), float(p["torso_std"])))
        foot_ref = q.new_tensor(metrics["foot_rpy_deg"]) * math.pi / 180
        foot_error = wrap_angle(angles[:, 1:3] - foot_ref)
        feet_angle = mean_and_worst(bounded_debt(foot_error.flatten(1), math.radians(3), math.radians(10)))
        width_error = pos[:, 1, 1] - pos[:, 2, 1] - metrics["foot_center_width_cm"] / 100
        feet = mean_and_worst(torch.stack((
            feet_angle,
            bounded_debt(width_error, float(p["width_margin"]), float(p["width_std"])),
            bounded_debt(pos[:, 1, 0] - pos[:, 2, 0], 0.025, 0.05),
        ), -1)) * leg + (1 - leg)
        height_error = fr.root_pos[:, 2] - fr.const("origin")[:, 2] - metrics["pelvis_height_cm"] / 100
        height = bounded_debt(height_error, 0.015, 0.04) * leg + (1 - leg)
        hand_ref = q.new_tensor([metrics[s + "_wrist_height_cm"] / 100 for s in ("left", "right")])
        hand_error = pos[:, 3:5, 2] - hand_ref
        hands = mean_and_worst(bounded_debt(hand_error, 0.02, 0.05))
        geometry = mean_and_worst(torch.stack((torso, feet, height, hands), -1))
        cost = 0.6 * joints + 0.4 * geometry
        upper = error[:, 12:]
        legacy = torch.exp(-upper.square().mean(-1) / 0.5**2) * c["in_hold"].float()
        cost_scale = float(p["cost_scale"])
        pose_score = torch.exp(-cost_scale * cost) if p.get("positive_kernel", False) else 1 - cost_scale * cost
        elapsed = c["dwell_prev"]
        ready = torch.zeros_like(c["in_hold"])
        blend = float(p.get("completion_blend", 0.0))
        if blend:
            qd = fr.qd[:, self.active_ids]
            speed = fr.root_lin_vel[:, :2].norm(dim=-1)
            ratios = readiness_ratios(
                error, torso_error, foot_error, width_error, height_error, hand_error,
                distance, speed, rpy(fr.root_quat), qd, c["loaded20"], self.waist_pitch_column,
            )
            ready = (gate > 0) & (ratios.amax(-1) <= 1)
            continuing = ready & c["ready_prev"] & ~c["dwell_reset"]
            elapsed = torch.where(continuing, c["dwell_prev"] + self.dt, torch.zeros_like(c["dwell_prev"]))
            dwell = (elapsed / float(p["completion_dwell_s"])).clamp(0, 1) * ready
            complete_score = 0.8 * weakest_score(ratios) + 0.2 * dwell
            pose_score = (1 - blend) * pose_score + blend * complete_score
        reward = (1 - scale) * legacy + scale * gate * pose_score
        tail = float(p.get("pose_tail_weight", 0.0))
        if tail:
            x = (error.abs() - math.radians(3)).clamp_min(0) / math.radians(15)
            reward = reward - scale * gate * tail * mean_and_worst(torch.sqrt(1 + x.square()) - 1)
        return reward, elapsed, ready

    def r_full_body_hold(self, fr, p):
        reward, elapsed, ready = self.hold_terms(fr, p)
        fr.hold_state = (elapsed.detach(), ready)
        return reward

    def r_full_body_quiet(self, fr, p):
        from whole_body_tracking.utils.stroke_learning import enabled, settling_velocity_debt

        if not enabled(self.env.command_manager.get_term("motion")):
            raise NotImplementedError("only the v19 imitation-first full-body quiet is ported")
        debt = settling_velocity_debt(fr.q[:, self.active_ids] - self.hold_ref, fr.qd[:, self.active_ids])
        gate = fr.ctx["recovery_gate"]
        if not p.get("state_independent", False):
            # Settled-station eligibility: both soles loaded (recorded contact),
            # near the station, slow, upright and not rotating.
            distance = (fr.ctx["base_target"] - fr.root_pos[:, :2]).norm(dim=-1)
            speed = fr.root_lin_vel[:, :2].norm(dim=-1)
            near = ((0.10 - distance) / 0.05).clamp(0, 1)
            slow = ((0.20 - speed) / 0.10).clamp(0, 1)
            # ``stroke_learning.balance_gate`` with a finite derivative when upright.
            x, y = fr.root_quat[:, 1], fr.root_quat[:, 2]
            tilt = safe_acos(1 - 2 * (x * x + y * y))
            rate = (fr.root_rot @ S.part(fr.s_end, "wb")[..., None])[..., 0][:, :2].norm(dim=-1)
            balance = ((math.radians(12) - tilt) / math.radians(6)).clamp(0, 1) * ((0.6 - rate) / 0.4).clamp(0, 1)
            gate = gate * fr.ctx["loaded20"].float() * near * slow * balance
        return fr.ctx["ramp_full_body_quiet"] * gate * debt

    def r_actual_joint_limit(self, fr, p):
        from whole_body_tracking.utils.actual_joint_limit import limit_clearance_cost

        actual = fr.q[:, self.limit_columns]
        per_joint, _ = limit_clearance_cost(actual, fr.const("limit_hard"), fr.const("limit_margin"))
        multiplier = 1 + (float(p["hold_recover_multiplier"]) - 1) * fr.ctx["phase_gate"]
        waist = self.limit_waist
        return (0.5 * per_joint.amax(-1) + 0.5 * per_joint[:, waist].amax(-1)) * multiplier

    def r_station_recovery(self, fr, p):
        from whole_body_tracking.utils.ready_recovery import station_debt

        delta = fr.ctx["base_target"] - fr.root_pos[:, :2]
        debt = station_debt(
            delta, fr.root_lin_vel[:, :2], float(p["position_margin"]), float(p["position_std"]),
            float(p["velocity_margin"]), float(p["velocity_std"]),
        )
        return debt * fr.ctx["recovery_gate"] * fr.ctx["ramp_station_recovery"]

    def r_executed_qdes_rate(self, fr, p):
        if int(p.get("order", 1)) != 1:
            raise NotImplementedError("only first-order executed q_des rate is ported")
        live = self.qdes_live * (1 - fr.ctx["qdes_protected"].float()[:, None] * self.qdes_upper)
        multiplier = torch.where(fr.ctx["in_hold"], self.qdes_hold_multiplier, 1.0)
        cost = ((fr.qexec - fr.qexec_prev).square() * live * fr.const("inv_scale_sq")).sum(-1)
        return cost * multiplier

    def r_dynamic_support(self, fr, p):
        from whole_body_tracking.tasks.tracking.mdp.recovery_safe_set import capture_point_support_excess

        w = fr.const("mass_weights")[..., None]
        com = (fr.end.com_pos * w).sum(-2)
        com_vel = (fr.end.com_lin_vel * w).sum(-2)
        feet = list(p["asset_cfg"].body_ids)
        excess, flight = capture_point_support_excess(
            com, com_vel, fr.end.pos[:, feet], fr.ctx["support_loaded"],
            foot_radius=float(p["foot_radius"]), margin=float(p["margin"]),
        )
        debt = huber(excess / float(p["std"])) + flight.float()
        return debt * fr.ctx["support_scale"]

    # ------------------------------------------------------------------ terminations
    def margins(self, fr, reasons, params):
        """Signed distance to each failure threshold at ``s_end`` (positive fails)."""
        out = []
        for name in reasons:
            p = params[name]
            if name == "anchor_pos":
                z = fr.end.pos[:, self.anchor_body, 2]
                out.append((fr.ctx["anchor_ref_z"] - z).abs() / float(p["threshold"]) - 1)
            elif name == "anchor_ori":
                gz = -fr.end.rot[:, self.anchor_body, 2, 2]
                out.append((fr.ctx["anchor_ref_gz"] - gz).abs() / float(p["threshold"]) - 1)
            elif name == "ee_body_pos":
                z = fr.end.pos[:, self.ee_bodies, 2]
                out.append((fr.ctx["ee_ref_z"] - z).abs().amax(-1) / float(p["threshold"]) - 1)
            elif name == "base_fell_tilt":
                angle = safe_acos(fr.root_rot[:, 2, 2])
                out.append(angle / float(p["limit_angle"]) - 1)
            elif name == "base_too_low":
                out.append(1 - fr.root_pos[:, 2] / float(p["minimum_height"]))
            else:
                raise ValueError(f"No analytic margin for {name}")
        return torch.stack(out, -1)

    def bind_terminations(self, reasons):
        from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes

        tm = self.env.termination_manager
        motion = self.env.command_manager.get_term("motion")
        self.reasons = list(reasons)
        self.reason_params = {r: dict(tm.get_term_cfg(r).params) for r in reasons}
        ee = tm.get_term_cfg("ee_body_pos")
        reference = _get_body_indexes(motion, ee.params.get("body_names"))
        names = [motion.cfg.body_names[i] for i in reference]
        robot = self.env.scene["robot"]
        self.ee_bodies = [list(robot.body_names).index(n) for n in names]

    # ------------------------------------------------------------------ observations
    def clean_obs(self, s, action_prev, ctx, ids=None):
        """Noise-free actor observation at ``s`` (110-D, hitter_pure order)."""
        default = self.const["default_q"] if ids is None else self.const["default_q"][ids]
        R = quat_to_matrix(S.part(s, "quat"))
        g = rotate_inverse(R, R.new_tensor(GRAVITY).expand(R.shape[0], 3))
        forward = R[..., :, 0][:, :2]
        forward = forward / (forward.norm(dim=-1, keepdim=True) + 1e-6)
        p = S.part(s, "p")
        return torch.cat((
            S.part(s, "wb"),
            S.part(s, "q") - default,
            S.part(s, "qd"),
            action_prev,
            g,
            forward,
            ctx["base_target"] - p[:, :2],
            ctx["actor_rt_pos"] - p,
            ctx["actor_rt_vel"],
            ctx["tts"][:, None],
        ), -1)

    def critic_features(self, s, action_prev, ctx, ids=None):
        """Privileged value input: clean actor view, physical state and task context."""
        from .world_model import invariant

        p = S.part(s, "p")
        c = ctx
        flags = torch.stack([c[k].float() for k in (
            "in_hold", "complete", "pre_strike", "strike_window", "track_gate", "recovery_gate",
            "phase_gate", "arm_factor", "contact_reward_scale", "motion_reward_scale", "dwell_prev",
            "ready_prev", "normal_sign", "swing_sign", "cmd_delay", "qdes_delay",
            "ramp_fixed_station_feet", "ramp_full_body_hold", "ramp_full_body_quiet", "ramp_station_recovery",
        )], -1)
        return torch.cat((
            self.clean_obs(s, action_prev, c, ids),
            invariant(s),
            c["base_ref_pos"] - p[:, :2],
            c["base_ref_vel"],
            c["rt_moving"] - p,
            c["rt_vel"],
            c["rt_normal"],
            c["q_target"] - S.part(s, "q"),
            c["qd_target"] - S.part(s, "qd"),
            c["torso_desired"],
            (c["footwork_ref"] - p[:, None]).flatten(1),
            (c["station_foot_ref"] - p[:, None]).flatten(1),
            (c["anchor_ref_z"] - p[:, 2])[:, None],
            c["ee_ref_z"] - p[:, 2:3],
            flags,
        ), -1)

    @torch.no_grad()
    def observation_context(self):
        cmd = self.env.command_manager.get_term("racket_target")
        return dict(
            actor_rt_pos=cmd.actor_racket_target_pos_w().clone(),
            actor_rt_vel=cmd.actor_racket_target_vel_w().clone(),
        )
