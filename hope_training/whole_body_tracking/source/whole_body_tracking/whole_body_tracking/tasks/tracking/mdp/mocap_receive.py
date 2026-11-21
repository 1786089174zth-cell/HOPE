"""Opt-in Isaac Lab task wiring for receiving an incoming ball from the PPQ004 bank."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from whole_body_tracking.tasks.table_tennis import geometry as table_geometry

from .mocap_ball_replay import ReplayBatch, extend_bank_flights, reconstruct_bank_flights, step_with_table
from .mocap_paddle_contact import sweep_ball_against_disc
from .mocap_preparation import advance_preparation_progress
from .physical_ball import (
    RACKET_CONTACT_RADIUS, load_venue_table_params, net_plane_crossing, predict_table_contact, racket_impulse,
)
from .virtual_ball import load_venue_params, rk4_step


class MocapReceiveCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ball_params = load_venue_params()
        self._table_params = load_venue_table_params()
        flights, row_ids = reconstruct_bank_flights(
            Path(cfg.bank), cfg.bank_sha256, cfg.rows_per_side, cfg.bank_seed,
            cfg.table_offset, rk4_step, self._ball_params,
        )
        self._strike_times = torch.tensor(
            [time[-1] for time, _ in flights], dtype=torch.float32, device=self.device,
        )
        extended, spin = extend_bank_flights(
            flights, Path(cfg.bank), row_ids, cfg.post_strike_extension_s, cfg.table_offset,
            rk4_step, self._ball_params, predict_table_contact, self._table_params,
        )
        self._replay_end_times = torch.tensor(
            [time[-1] for time, _ in extended], dtype=torch.float32, device=self.device,
        )
        self._replay_spin = torch.as_tensor(spin, dtype=torch.float32, device=self.device)
        self._replay = ReplayBatch.from_flights(extended, self.device)
        self._flight_count = self._replay.times.shape[0]
        print(f"[mocap receive] reconstructed {self._flight_count} incoming flights from {cfg.bank}", flush=True)
        print(f"[mocap receive] bank SHA-256={cfg.bank_sha256}; selected rows={row_ids}", flush=True)
        self._flight = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._delay = torch.zeros_like(self._elapsed)
        self._prev_valid = torch.zeros_like(self._flight, dtype=torch.bool)
        self._prev_blade = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_blade_normal = torch.zeros_like(self._prev_blade)
        self._contacted = torch.zeros_like(self._prev_valid)
        self._rim_contact = torch.zeros_like(self._prev_valid)
        self._return_active = torch.zeros_like(self._prev_valid)
        self._paddle_rebound = torch.zeros_like(self._prev_valid)
        self._table_bounces = torch.zeros_like(self._flight)
        self._opponent_bounces = torch.zeros_like(self._flight)
        self._net_blocked = torch.zeros_like(self._prev_valid)
        self._hit_offset_s = torch.zeros_like(self._elapsed)
        self._outbound_vx = torch.zeros_like(self._elapsed)
        self._return_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._return_vel = torch.zeros_like(self._return_pos)
        self._return_spin = torch.zeros_like(self._return_pos)
        self.hit_now = torch.zeros_like(self._prev_valid)
        self._started = torch.zeros_like(self._prev_valid)
        self.approach = torch.zeros_like(self._elapsed)
        self.metrics["contact"] = torch.zeros_like(self._elapsed)
        self._table_offset = torch.tensor(cfg.table_offset, device=self.device)
        self.preparation_progress = torch.zeros_like(self._elapsed)
        self._preparation_best = torch.full_like(self._elapsed, -1.0)
        self._preparation_targets = None
        self._trial_log = Path(cfg.trial_log) if cfg.trial_log else None
        if self._trial_log is not None:
            self._trial_log.touch(exist_ok=False)
            self._row_ids = row_ids
            self._first_visible_s = torch.full_like(self._elapsed, torch.inf)
        if cfg.preparation_enabled:
            self._strike_positions = torch.stack(
                [torch.as_tensor(positions[-1], dtype=torch.float32, device=self.device)
                 for _, positions in flights],
            ) + self._table_offset
            self._upright_min = math.cos(math.radians(cfg.preparation_max_tilt_deg))
            self.metrics["preparation_progress"] = torch.zeros_like(self._elapsed)
        env.sim.add_physics_callback("hope_mocap_receive", self._physics_step)

    @property
    def command(self):
        history = self._replay.observe(
            self._flight, self._elapsed - self._delay, period_s=self._env.step_dt,
            delay_s=self.cfg.sensor_delay_s, max_age_s=self.cfg.max_age_s,
        )
        return torch.where((self._started & ~self._contacted)[:, None], history, 0.0)

    def _resample_command(self, env_ids):
        flight = self._flight[env_ids]
        self._flight[env_ids] = torch.randint(self._flight_count, flight.shape, device=self.device)
        lo, hi = self.cfg.launch_delay_s
        self._delay[env_ids] = lo + (hi - lo) * torch.rand(flight.shape, device=self.device)
        self._elapsed[env_ids] = 0.0
        self._prev_valid[env_ids] = False
        self._contacted[env_ids] = False
        self._rim_contact[env_ids] = False
        self._return_active[env_ids] = False
        self._paddle_rebound[env_ids] = False
        self._table_bounces[env_ids] = 0
        self._opponent_bounces[env_ids] = 0
        self._net_blocked[env_ids] = False
        self._hit_offset_s[env_ids] = 0.0
        self._outbound_vx[env_ids] = 0.0
        self.hit_now[env_ids] = False
        self.approach[env_ids] = 0.0
        self.preparation_progress[env_ids] = 0.0
        self._preparation_best[env_ids] = -1.0
        if self._trial_log is not None:
            self._first_visible_s[env_ids] = torch.inf
        self._started[env_ids] = True

    def reset(self, env_ids=None):
        if self._trial_log is not None:
            self._write_trials(env_ids, "reset")
        return super().reset(env_ids)

    def write_final_trials(self):
        if self._trial_log is not None:
            self._write_trials(None, "run_end")

    def _write_trials(self, env_ids, ended_by):
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        ids = ids[self._started[ids] & (self._elapsed[ids] > 0.0)]
        flight = self._flight[ids]
        nominal = self._delay[ids] + self._strike_times[flight]
        end = self._delay[ids] + self._replay_end_times[flight]
        prep = self.metrics.get("preparation_progress", self.preparation_progress)[ids]
        columns = torch.stack((
            ids, flight, self._first_visible_s[ids], nominal, end, self._elapsed[ids],
            self._contacted[ids], prep, self._hit_offset_s[ids], self._outbound_vx[ids],
            self._paddle_rebound[ids], self._table_bounces[ids], self._opponent_bounces[ids],
            self._net_blocked[ids], self._rim_contact[ids],
        ), dim=-1).cpu().tolist()
        with self._trial_log.open("a", encoding="utf-8") as stream:
            for (env, shot, seen, nominal_s, end_s, elapsed, hit, progress, hit_offset,
                 outbound_vx, rebound, bounces, opponent_bounces, net_blocked, rim) in columns:
                visible = math.isfinite(seen)
                if hit:
                    outcome = "hit"
                elif elapsed >= end_s:
                    outcome = "miss" if visible else "no_observation"
                else:
                    outcome = "interrupted" if ended_by == "reset" else "incomplete"
                record = {
                    "env": int(env), "bank_row": self._row_ids[int(shot)],
                    "side": "forehand" if shot < self.cfg.rows_per_side else "backhand",
                    "outcome": outcome, "ended_by": ended_by,
                    "visible_lead_ms": 1000.0 * (nominal_s - seen) if visible else None,
                    "preparation_progress": progress, "elapsed_s": elapsed,
                    "hit_offset_ms": 1000.0 * hit_offset if hit else None,
                    "contact_location": ("rim" if rim else "face") if hit else None,
                    "outbound_vx_m_s": outbound_vx if hit else None,
                    "paddle_rebound": bool(rebound), "table_bounces": int(bounces),
                    "opponent_table_bounces": int(opponent_bounces), "net_blocked": bool(net_blocked),
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _update_metrics(self):
        pass  # The physics callback records contact before the episode can reset.

    def _update_command(self):
        # Isaac Lab computes rewards before commands and observations.
        self.hit_now.zero_()
        self.preparation_progress.zero_()
        if self._trial_log is not None:
            latest_is_visible = self.command[:, 4].bool()
            first = latest_is_visible & torch.isinf(self._first_visible_s)
            self._first_visible_s.copy_(torch.where(first, self._elapsed, self._first_visible_s))

    def _get_preparation_targets(self, racket):
        if self._preparation_targets is None:
            command = racket._motion()
            motion = command.motion
            if int(motion.num_segments) != 2:
                raise ValueError("mocap preparation needs the original forehand and backhand clips")
            uses_mount = racket._racket_mode != "body"
            body = racket._wrist_body_index if uses_mount else racket._racket_body_index
            frames_early = round(self.cfg.preparation_reference_lead_s / self._env.step_dt)
            offsets = []
            for side in (0, 1):
                strike, _, start, length = racket._strike_frame_for_clip(motion, side)
                if strike - frames_early < start or frames_early < 1:
                    raise ValueError("preparation reference lead does not fit the motion clip")
                frames = [strike - start - frames_early, strike - start]
                # MotionLoader keeps only tracked bodies. Use the source clip's
                # full articulation arrays to read the wrist or racket body.
                with np.load(command.cfg.motion_file[side], allow_pickle=False) as clip:
                    positions = clip["body_pos_w"]
                    if positions.shape != (length, len(racket.robot.body_names), 3):
                        raise ValueError("mocap preparation requires full articulation reference clips")
                    centers = torch.as_tensor(positions[frames, body], dtype=torch.float32, device=self.device)
                    if uses_mount:
                        quat = torch.as_tensor(
                            clip["body_quat_w"][frames, body], dtype=torch.float32, device=self.device,
                        )
                        centers = centers + quat_apply(quat, racket._mount_offset[:1].expand(2, -1))
                offsets.append(centers[0] - centers[1])
            # reconstruct_bank_flights returns the chosen forehand rows before
            # the chosen backhand rows, with exactly rows_per_side of each.
            row_offsets = torch.stack(offsets).repeat_interleave(self.cfg.rows_per_side, dim=0)
            self._preparation_targets = self._strike_positions + row_offsets
        return self._preparation_targets

    def _physics_step(self, dt):
        self._elapsed += dt
        self._advance_returns(dt)
        now = self._elapsed - self._delay
        end = self._replay_end_times[self._flight]
        replay_now = torch.minimum(now.clamp_min(0.0), end)
        position, velocity, valid = self._replay.physical_sample(
            self._flight, replay_now, max_gap_s=self.cfg.max_gap_s,
        )
        racket = self._env.command_manager.get_term("racket_target")
        blade_w, _, _, normal, _ = racket._racket_fk()
        blade = blade_w - self._env.scene.env_origins
        ball = position + self._table_offset
        incoming = self._started & valid & ~self._contacted & (velocity[:, 0] < 0.0) & (now >= 0.0)
        active = incoming & (now <= end)
        candidates = incoming & self._prev_valid & (now - dt <= end)
        hit = self._sweep_contacts(
            candidates, dt, now, replay_now, ball, racket, blade, normal,
        )
        self._contacted |= hit
        self.hit_now |= hit
        self.metrics["contact"].add_(hit)
        self._prev_valid.copy_(self._started)
        self._prev_blade.copy_(blade)
        self._prev_blade_normal.copy_(normal)
        near_end = (position[:, 0] <= 0.40) & (position[:, 0] >= -0.30)
        separation_sq = (ball - blade).square().sum(dim=-1)
        self.approach.copy_(torch.where(
            active & near_end, torch.exp(-separation_sq / self.cfg.approach_distance_m**2), 0.0,
        ))
        if self.cfg.preparation_enabled:
            ready = self._get_preparation_targets(racket)[self._flight] + self._env.scene.env_origins
            upright = -racket.robot.data.projected_gravity_b[:, 2] >= self._upright_min
            progress, best = advance_preparation_progress(
                self._preparation_best, (blade_w - ready).square().sum(dim=-1),
                self._strike_times[self._flight] - now,
                active & ~hit & (now >= self.cfg.sensor_delay_s), upright,
                self.cfg.preparation_window_s, self.cfg.preparation_distance_scale_m,
            )
            self._preparation_best.copy_(best)
            self.preparation_progress.add_(progress)
            self.metrics["preparation_progress"].add_(progress)

    def _sweep_contacts(self, candidates, dt, now, replay_now, ball, racket, blade, normal):
        hit = torch.zeros_like(candidates)
        ids = candidates.nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return hit

        # Clip the interval to actual replay data, including fractional launch and final ticks.
        previous_t = now[ids] - dt
        start_t = previous_t.clamp_min(0.0)
        stop_t = replay_now[ids]
        start_pos, _, valid = self._replay.physical_sample(
            self._flight[ids], start_t, max_gap_s=self.cfg.max_gap_s,
        )
        start_f = ((start_t - previous_t) / dt).clamp(0.0, 1.0)
        stop_f = ((stop_t - previous_t) / dt).clamp(0.0, 1.0)
        blade_start = torch.lerp(self._prev_blade[ids], blade[ids], start_f[:, None])
        blade_stop = torch.lerp(self._prev_blade[ids], blade[ids], stop_f[:, None])
        delta_start = start_pos + self._table_offset - blade_start
        delta_stop = ball[ids] - blade_stop
        motion = delta_stop - delta_start
        closest_f = -(delta_start * motion).sum(dim=-1) / motion.square().sum(dim=-1).clamp_min(1e-12)
        closest_center = delta_start + closest_f.clamp(0.0, 1.0)[:, None] * motion
        reach = RACKET_CONTACT_RADIUS + self._ball_params.ball_radius + 1e-6
        nearby = torch.linalg.vector_norm(closest_center, dim=-1) <= reach
        near = (valid & nearby & (stop_t > start_t)).nonzero(as_tuple=True)[0]
        if near.numel() == 0:
            return hit

        ids = ids[near]
        normal_start = torch.lerp(self._prev_blade_normal[ids], normal[ids], start_f[near, None])
        normal_stop = torch.lerp(self._prev_blade_normal[ids], normal[ids], stop_f[near, None])
        touched, fraction, point, outward, face, rotation = sweep_ball_against_disc(
            delta_start[near], delta_stop[near], normal_start, normal_stop,
            RACKET_CONTACT_RADIUS, self._ball_params.ball_radius,
        )
        impact = touched.nonzero(as_tuple=True)[0]
        if impact.numel() == 0:
            return hit
        ids, near = ids[impact], near[impact]
        fraction = fraction[impact]
        point, outward = point[impact], outward[impact]
        time = torch.lerp(start_t[near], stop_t[near], fraction)
        dt_left = (now[ids] - time).clamp_min(0.0)
        blade_at_hit = torch.lerp(blade_start[near], blade_stop[near], fraction[:, None])
        span = (stop_t[near] - start_t[near])[:, None]
        blade_velocity = (blade_stop[near] - blade_start[near]) / span
        ball_velocity = blade_velocity + motion[near] / span
        link = racket._racket_body_index if racket._racket_mode == "body" else racket._wrist_body_index
        # Use the same motion as the sweep. Twist about the face axis is invisible
        # to disc geometry, so retain the actual link's twist for tangential speed.
        face = face[impact]
        angular_velocity = racket.robot.data.body_link_ang_vel_w[ids, link]
        twist = (angular_velocity * face).sum(dim=-1, keepdim=True) * face
        omega = rotation[impact] / span + twist
        point_velocity = blade_velocity + torch.linalg.cross(omega, point)
        self._rim_contact[ids] = torch.linalg.vector_norm(point, dim=-1) >= RACKET_CONTACT_RADIUS - 1e-6
        self._start_returns(ids, time, blade_at_hit + point, ball_velocity, point_velocity, outward)
        self._advance_returns(dt_left[:, None], ids)
        hit[ids] = True
        return hit

    def _start_returns(self, ids, now, contact_point, velocity, blade_vel, normal):
        flight = self._flight[ids]
        offset = now - self._strike_times[flight]
        spin_sample = (offset * 1000.0).floor().long().clamp(0, self._replay_spin.shape[1] - 1)
        outbound, spin = racket_impulse(
            velocity, blade_vel, normal, self._replay_spin[flight, spin_sample], self._ball_params,
        )
        self._return_pos[ids] = contact_point + self._ball_params.ball_radius * normal
        self._return_vel[ids] = outbound
        self._return_spin[ids] = spin
        incoming_normal_speed = ((velocity - blade_vel) * normal).sum(dim=-1)
        outgoing_normal_speed = ((outbound - blade_vel) * normal).sum(dim=-1)
        self._paddle_rebound[ids] = (incoming_normal_speed < 0.0) & (outgoing_normal_speed > 0.0)
        self._hit_offset_s[ids] = offset
        self._outbound_vx[ids] = outbound[:, 0]
        self._return_active[ids] = True

    def _advance_returns(self, dt, ids=None):
        if ids is None:
            ids = self._return_active.nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return
        near_x, _, surface_z = self.cfg.table_offset
        net_x = near_x + table_geometry.NET_X
        previous = self._return_pos[ids]
        p, v, w, bounce, _, impact = step_with_table(
            previous, self._return_vel[ids], self._return_spin[ids], dt,
            rk4_step, self._ball_params, predict_table_contact, self._table_params, near_x, surface_z,
        )
        # If an impact precedes the net in this step, use the incoming leg up to that impact.
        net_check_end = torch.where(bounce[:, None], impact, p)
        crossed, net_z = net_plane_crossing(previous, net_check_end, net_x)
        fraction = ((net_x - previous[:, 0]) / (net_check_end[:, 0] - previous[:, 0]).clamp_min(1e-9))
        net_y = torch.lerp(previous[:, 1], net_check_end[:, 1], fraction.clamp(0.0, 1.0))
        radius = self._ball_params.ball_radius
        blocked = crossed & (net_y.abs() <= table_geometry.TABLE_WIDTH / 2 + table_geometry.NET_OVERHANG + radius)
        blocked &= (net_z < surface_z + table_geometry.NET_HEIGHT + radius) & (net_z >= surface_z - radius)
        bounce &= ~blocked
        self._return_pos[ids], self._return_vel[ids], self._return_spin[ids] = p, v, w
        self._net_blocked[ids] |= blocked
        self._table_bounces[ids] += bounce.long()
        self._opponent_bounces[ids] += (bounce & (impact[:, 0] > net_x)).long()
        self._return_active[ids] = ~blocked & (p[:, 2] > radius)


@configclass
class MocapReceiveCommandCfg(CommandTermCfg):
    class_type: type = MocapReceiveCommand
    bank: str = ""
    bank_sha256: str = ""
    rows_per_side: int = 16
    bank_seed: int = 0
    table_offset: tuple[float, float, float] = (0.5, 0.7625, 0.76)
    launch_delay_s: tuple[float, float] = (0.15, 0.45)
    sensor_delay_s: float = 0.010
    max_age_s: float = 0.050
    max_gap_s: float = 0.020
    post_strike_extension_s: float = 0.120
    approach_distance_m: float = 0.15
    preparation_enabled: bool = False
    preparation_window_s: tuple[float, float] = (0.10, 0.30)
    preparation_reference_lead_s: float = 0.20
    preparation_distance_scale_m: float = 0.15
    preparation_max_tilt_deg: float = 20.0
    trial_log: str | None = None


def ball_history(env):
    return env.command_manager.get_term("mocap_receive").command


def replay_contact(env):
    return env.command_manager.get_term("mocap_receive").hit_now.float() / env.step_dt


def replay_approach(env):
    return env.command_manager.get_term("mocap_receive").approach


def replay_preparation(env):
    return env.command_manager.get_term("mocap_receive").preparation_progress / env.step_dt


def configure_mocap_receive(env_cfg, replay, repo_root):
    delay = float(replay["sensor_delay_s"])
    age = float(replay["max_age_s"])
    if not (math.isfinite(delay) and math.isfinite(age) and 0.0 <= delay < age):
        raise ValueError("mocap replay requires 0 <= sensor_delay_s < max_age_s")
    extension = float(replay.get("post_strike_extension_s", 0.120))
    if not (math.isfinite(extension) and extension >= 0.0
            and math.isclose(extension * 1000, round(extension * 1000), abs_tol=1e-8)):
        raise ValueError("mocap replay extension must be a nonnegative whole number of milliseconds")
    racket = env_cfg.commands.racket_target
    if racket.venue_tuple_enabled or racket.virtual_ball or racket.physical_ball:
        raise ValueError("mocap replay must be the only source of incoming balls")
    bank = repo_root / "hope_training/whole_body_tracking" / racket.venue_tuple_bank_path
    if not racket.venue_tuple_bank_sha256:
        raise ValueError("mocap replay requires the original PPQ004 bank SHA-256")
    prep = replay.get("preparation") or {}
    prep_weight = float(prep.get("weight", 0.0))
    late, early = map(float, prep.get("window_s", (0.10, 0.30)))
    reference_lead = float(prep.get("reference_lead_s", 0.20))
    distance_scale = float(prep.get("distance_scale_m", 0.15))
    tilt = float(prep.get("max_tilt_deg", 20.0))
    if not (
        all(math.isfinite(x) for x in (prep_weight, late, early, reference_lead, distance_scale, tilt))
        and prep_weight >= 0.0 and 0.0 < late < early and late <= reference_lead <= early
        and distance_scale > 0.0 and 0.0 < tilt < 90.0
    ):
        raise ValueError("invalid mocap preparation weight, lead window, distance scale or tilt limit")
    env_cfg.commands.mocap_receive = MocapReceiveCommandCfg(
        resampling_time_range=(1e6, 1e6), sensor_delay_s=delay, max_age_s=age,
        post_strike_extension_s=extension,
        bank=str(bank), bank_sha256=str(racket.venue_tuple_bank_sha256),
        rows_per_side=int(replay["rows_per_side"]), bank_seed=int(replay["bank_seed"]),
        table_offset=(float(racket.vb_table_near_x), 0.7625, float(racket.vb_table_surface_z)),
        preparation_enabled=prep_weight > 0.0, preparation_window_s=(late, early),
        preparation_reference_lead_s=reference_lead, preparation_distance_scale_m=distance_scale,
        preparation_max_tilt_deg=tilt,
        trial_log=str(Path(replay["trial_log"]).resolve()) if replay.get("trial_log") else None,
    )

    policy = env_cfg.observations.policy
    policy.racket_target_rel_base = None
    policy.racket_target_vel_w = None
    policy.time_to_strike = None
    policy.ball_history = ObservationTermCfg(func=ball_history)
    env_cfg.observations.critic = copy.deepcopy(policy)
    env_cfg.observations.critic.enable_corruption = False

    # Keep robot regularization and replace the old entire-swing tracking rewards.
    retained = {
        "joint_torques", "joint_vel", "joint_acc", "action_rate",
        "upright", "base_ang_vel_xy", "base_lin_vel_z", "dynamic_support",
    }
    for name, term in vars(env_cfg.rewards).items():
        if isinstance(term, RewardTermCfg) and name not in retained:
            setattr(env_cfg.rewards, name, None)
    env_cfg.rewards.replay_contact = RewardTermCfg(func=replay_contact, weight=2.0)
    env_cfg.rewards.replay_approach = RewardTermCfg(func=replay_approach, weight=2.0)
    if prep_weight > 0.0:
        env_cfg.rewards.replay_preparation = RewardTermCfg(func=replay_preparation, weight=prep_weight)
