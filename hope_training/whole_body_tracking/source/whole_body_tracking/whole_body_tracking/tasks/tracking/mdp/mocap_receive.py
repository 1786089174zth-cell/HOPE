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

from .mocap_ball_replay import ReplayBatch, reconstruct_bank_flights
from .mocap_preparation import advance_preparation_progress
from .physical_ball import blade_disc_contact
from .virtual_ball import load_venue_params, rk4_step


class MocapReceiveCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        flights, row_ids = reconstruct_bank_flights(
            Path(cfg.bank), cfg.bank_sha256, cfg.rows_per_side, cfg.bank_seed,
            cfg.table_offset, rk4_step, load_venue_params(),
        )
        self._replay = ReplayBatch.from_flights(flights, self.device)
        self._flight_count = self._replay.times.shape[0]
        print(f"[mocap receive] reconstructed {self._flight_count} incoming flights from {cfg.bank}", flush=True)
        print(f"[mocap receive] bank SHA-256={cfg.bank_sha256}; selected rows={row_ids}", flush=True)
        self._flight = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._delay = torch.zeros_like(self._elapsed)
        self._prev_dn = torch.zeros_like(self._elapsed)
        self._prev_valid = torch.zeros_like(self._flight, dtype=torch.bool)
        self._contacted = torch.zeros_like(self._prev_valid)
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
        if cfg.preparation_enabled or self._trial_log is not None:
            self._strike_times = torch.tensor(
                [float(time[-1]) for time, _ in flights], dtype=torch.float32, device=self.device,
            )
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
        end = self._delay[ids] + self._strike_times[flight]
        prep = self.metrics.get("preparation_progress", self.preparation_progress)[ids]
        columns = torch.stack((
            ids, flight, self._first_visible_s[ids], end, self._elapsed[ids],
            self._contacted[ids], prep,
        ), dim=-1).cpu().tolist()
        with self._trial_log.open("a", encoding="utf-8") as stream:
            for env, shot, seen, end_s, elapsed, hit, progress in columns:
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
                    "visible_lead_ms": 1000.0 * (end_s - seen) if visible else None,
                    "preparation_progress": progress, "elapsed_s": elapsed,
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
        now = self._elapsed - self._delay
        position, velocity, valid = self._replay.physical_sample(
            self._flight, now, max_gap_s=self.cfg.max_gap_s,
        )
        racket = self._env.command_manager.get_term("racket_target")
        blade, _, blade_vel, _, normal = racket._racket_fk()
        ball = position + self._table_offset + self._env.scene.env_origins
        active = self._started & valid & ~self._contacted & (velocity[:, 0] < 0.0)
        hit, signed_distance = blade_disc_contact(
            ball, velocity, blade, blade_vel, normal, self._prev_dn, self._prev_valid & active,
        )
        hit &= active
        self._contacted |= hit
        self.hit_now |= hit
        self.metrics["contact"].add_(hit)
        self._prev_dn.copy_(signed_distance)
        self._prev_valid.copy_(active & ~hit)
        near_end = (position[:, 0] <= 0.40) & (position[:, 0] >= -0.30)
        separation_sq = (ball - blade).square().sum(dim=-1)
        self.approach.copy_(torch.where(
            active & near_end, torch.exp(-separation_sq / self.cfg.approach_distance_m**2), 0.0,
        ))
        if self.cfg.preparation_enabled:
            ready = self._get_preparation_targets(racket)[self._flight] + self._env.scene.env_origins
            upright = -racket.robot.data.projected_gravity_b[:, 2] >= self._upright_min
            progress, best = advance_preparation_progress(
                self._preparation_best, (blade - ready).square().sum(dim=-1),
                self._strike_times[self._flight] - now,
                active & ~hit & (now >= self.cfg.sensor_delay_s), upright,
                self.cfg.preparation_window_s, self.cfg.preparation_distance_scale_m,
            )
            self._preparation_best.copy_(best)
            self.preparation_progress.add_(progress)
            self.metrics["preparation_progress"].add_(progress)


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
