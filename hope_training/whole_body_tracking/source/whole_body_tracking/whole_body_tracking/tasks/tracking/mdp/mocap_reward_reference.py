"""Training-only reference for reusing the stationary PPQ004 reward functions."""

import torch


def replay_reward_clock(now, nominal, end, contacted, hit_offset, hit_now):
    """Wait for a real contact or the replay deadline before starting recovery."""
    before_result = (nominal - now).clamp_min(0.0)
    after_hit = torch.where(hit_now, 0.0, nominal + hit_offset - now)
    return torch.where(contacted, after_hit, torch.where(now < end, before_result, end - now))


class MocapRewardReference:
    """Present the original racket reward inputs without changing either live command."""

    def __init__(self, receive, env):
        self.receive = receive
        self.env = env
        self.motion_reference = _MotionRewardReference(self)

    @property
    def racket(self):
        return self.env.command_manager.get_term("racket_target")

    def __getattr__(self, name):
        return getattr(self.racket, name)

    @property
    def time_to_strike(self):
        receive = self.receive
        return replay_reward_clock(
            receive._elapsed - receive._delay,
            receive._strike_times[receive._flight], receive._replay_end_times[receive._flight],
            receive._contacted, receive._hit_offset_s, receive.hit_now,
        )

    @property
    def guidance_time_to_strike(self):
        receive = self.receive
        now = receive._elapsed - receive._delay
        nominal = receive._strike_times[receive._flight]
        visible = (
            receive._started & ~receive._contacted & (now >= receive.cfg.sensor_delay_s)
            & (now <= receive._replay_end_times[receive._flight])
        )
        return torch.where(receive.hit_now, 0.0, torch.where(visible, nominal - now, 1e3))

    @property
    def pre_strike(self):
        receive = self.receive
        return ~receive._contacted & (receive._elapsed - receive._delay < receive._replay_end_times[receive._flight])

    @property
    def strike_window(self):
        return self.guidance_time_to_strike.abs() <= self.cfg.strike_window_s

    @property
    def swing_sign(self):
        return torch.where(self.motion_reference.clip_id == 0, 1.0, -1.0)

    @property
    def reward_in_hold(self):
        return self.motion_reference.in_hold

    @property
    def racket_target_pos_w(self):
        receive = self.receive
        nominal = receive._strike_positions[receive._flight] + self.env.scene.env_origins
        return torch.where(receive.hit_now[:, None], receive._impact_ball_w, nominal)

    @property
    def racket_target_vel_w(self):
        return self.receive._target_velocities[self.receive._flight]

    @property
    def racket_target_normal_w(self):
        return self.racket._normal_from_target_velocity(self.racket_target_vel_w)

    @property
    def racket_pos_w(self):
        receive = self.receive
        return torch.where(receive.hit_now[:, None], receive._impact_blade_w, self.racket.racket_pos_w)

    @property
    def racket_lin_vel_w(self):
        receive = self.receive
        return torch.where(receive.hit_now[:, None], receive._impact_blade_vel, self.racket.racket_lin_vel_w)

    @property
    def racket_normal_w(self):
        receive = self.receive
        normal = torch.where(receive.hit_now[:, None], receive._impact_face, receive._prev_blade_normal)
        signs = self.racket._mount_sign_per_clip_t
        sign = signs[self.motion_reference.clip_id, None] if signs is not None else float(self.cfg.mount_normal_sign)
        return normal * sign

    def _motion(self):
        return self.motion_reference


class _MotionRewardReference:
    def __init__(self, parent):
        self.parent = parent
        self._strike_frames = None

    @property
    def live(self):
        return self.parent.env.command_manager.get_term("motion")

    def __getattr__(self, name):
        return getattr(self.live, name)

    @property
    def clip_id(self):
        receive = self.parent.receive
        return (receive._flight >= receive.cfg.rows_per_side).long()

    @property
    def in_hold(self):
        receive = self.parent.receive
        clock = self.parent.time_to_strike
        waiting = receive._elapsed - receive._delay < receive.cfg.sensor_delay_s
        return (self.parent.pre_strike & (waiting | (clock > receive.cfg.preparation_window_s[1]))) | (clock < -0.8)

    @property
    def joint_pos(self):
        motion = self.live.motion
        if self._strike_frames is None:
            self._strike_frames = torch.tensor(
                [self.parent.racket._strike_frame_for_clip(motion, i)[0] for i in (0, 1)], device=self.device,
            )
        clip = self.clip_id
        frame = self._strike_frames[clip] - (self.parent.time_to_strike / self.parent.env.step_dt).round().long()
        start = motion.seg_start[clip]
        frame = torch.minimum(torch.maximum(frame, start), start + motion.seg_len[clip] - 1)
        return motion.joint_pos[frame]
