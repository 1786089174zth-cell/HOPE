"""Four-stroke ping-pong rally task, standalone (no Isaac).

The exogenous side of the HOPE task is reproduced from its data assets and
clocks (``docs/spec`` references are to the Isaac task this mirrors):

* questions: rows of the physical tuple bank whose arm teacher is qualified,
  25% per family (stand/step x forehand/backhand), 35% from the deploy-plane slab;
* reference: the family clip under the preparation time-warp, with the row's
  right-arm adapter blended in around contact; HOLD pose between balls;
* rally clock: startup HOLD, announcement, strike at clip frame 60, recovery,
  next ball from +0.5 s on;
* observation: the deployed 110-D ``hitter_pure`` frame (mocap base pose and the
  announced racket target with their noise models).

Reward and termination follow the paper: BeyondMimic tracking of the reference
(Table A.2) plus the interface's own task reward, here the racket strike terms.
Everything the policy can influence is a differentiable function of the state;
everything else (references, clocks, ball commands) is logged on the tape.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .. import a3
from .dynamics import NJ, represent
from .fk import Kinematics
from .rotations import quat_exp, quat_from_yaw, quat_mul, quat_to_matrix
from .tracking import TRACK_WEIGHTS, soft_termination, termination_error, tracking_terms

WBT = Path(__file__).resolve().parents[3] / "whole_body_tracking"
FAMILIES = ("stand_forehand", "stand_backhand", "step_forehand", "step_backhand")
RIGHT_ARM = (13, 18, 22, 24, 26, 28, 30)  # shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
LEFT_ARM = tuple(a3.JOINTS.index(a3.JOINTS[i].replace("right_", "left_")) for i in RIGHT_ARM)
HEAD = (11, 16)
UPPER_REWARD_TERMS = ("racket_normal", "left_arm", "direct_face")
LOWER_REWARD_TERMS = ("feet_pos", "feet_height", "feet_ori", "root_xy", "feet_station",
                      "root_heading", "feet_heading", "feet_slip", "stance_command",
                      "waist_rest", "direct_waist_roll")
TRACKED = ("pelvis_link", "left_hip_roll_Link", "left_knee_Link", "left_ankle_roll_Link", "right_hip_roll_Link",
           "right_knee_Link", "right_ankle_roll_Link", "torso_Link", "left_shoulder_roll_Link", "left_elbow_Link",
           "left_wrist_yaw_Link", "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link")
STRIKE_FRAME, LAST_FRAME, NEXT_BALL_FRAME = 60, 139, 85
IDLE_POS, IDLE_VEL, IDLE_TTS, HOLD_TTS = (0.465, -0.5975, 1.11), (1.925, 0.0, 0.65), 1.2, -1.58


@dataclass
class TaskCfg:
    assets: str = "assets/wide_stroke_v26_urdf2_plane058_20261110"
    episode_steps: int = 2000  # 40 s
    # preparation leads per family (minimum T, complete L), seconds
    minimum_lead: tuple[float, ...] = (0.50, 0.60, 0.58, 0.64)
    complete_lead: tuple[float, ...] = (0.58, 0.72, 0.86, 0.96)
    # The deploy plane the bank is named after (strike 0.58 m ahead of the station, 8 cm over the table
    # edge) gets most of the questions since 2026-11-19: at the 0.465 m plane every standard Gate3 ball
    # is struck millimetres from the table edge (docs/returns_20261119.md). Until then 0.35 of (0.50, 0.58).
    plane_probability: float = 0.6
    plane_x: tuple[float, float] = (0.555, 0.58)
    # Late announcements. The bank has no backhand or step question announced less than 0.56 s
    # before the strike, while Gate3 commits about 0.5 s ahead (docs/gate3_20261118.md). With this
    # probability per family a drawn question is announced U(short_notice_min_s, T) before its
    # strike instead; the reference then starts inside the preparation warp.
    short_notice_probability: tuple[float, ...] = (0.0, 0.3, 0.3, 0.3)
    short_notice_min_s: float = 0.46
    # Fast balls. Gate3's Planner commits about 0.1 s after the bounce, so a ball that bounces close
    # to the robot or comes in fast is announced 0.23-0.39 s ahead (tools/gate3/make_scenario.py:
    # fast serves 21/36 legal against 72/72 for slow and medium ones). With this probability, in
    # every family, a question is announced U(rush_notice_s) before its strike.
    # Off by default: 9000 iterations with 0.2 (model_51000) lost returns on slow balls in Gate3 and did not
    # gain any on fast ones (docs/ball_paths_20261118.md); 3000 with 0.3 on U(0.30, 0.46) at the 0.58 plane
    # (model_52000) again returned fewer fast balls than the checkpoint it started from (docs/returns_20261119.md).
    rush_probability: float = 0.0
    rush_notice_s: tuple[float, float] = (0.24, 0.46)
    # Supergate questions. Supergate's offline Planner asks for strokes the bank does not have: announced
    # 0.22-0.40 s ahead (bank: 0.50 s at least), flat and slower (racket 0.8-2.4 m/s within -16..+25 deg of
    # level; bank 1.7-3.2 m/s climbing 18-46 deg), up to 1.42 m high, for balls of 1.8-5.1 m/s that are mostly
    # still rising. ``extra_bank`` is a file of such questions in the bank's format, made by
    # tools/supergate/build_questions.py from the Planner's own targets for launches that are not in the
    # frozen suites (path relative to this package's root). With ``extra_probability`` a question is drawn
    # from it, with its own notice and side (docs/supergate_20261119.md).
    # Their reference is a direct stroke, not a clip: there is no time for a clip's backswing (announced
    # 0.25 s ahead, a forehand's racket was still going back at the strike: 0.2 m/s forward for 1.5 commanded),
    # and with the torso where the stand clips have it the backhand targets are out of the arm's reach. The
    # file carries a strike pose per row, solved from the HOLD pose with the waist and the right arm; the
    # reference goes from HOLD to it on a cubic over the notice, arriving with the strike velocity, and back
    # to HOLD over ``extra_recovery_s``. Legs and root stay in the HOLD reference.
    extra_bank: str = ""
    extra_probability: float = 0.0
    extra_recovery_s: float = 0.6
    # Half of these questions are drawn cell by cell: the stroke as the rows have it (a ball's stroke is where it
    # is), then one of its nine cells of incoming speed (slow < 2.5, medium, fast >= 4.0 m/s at the strike) x
    # strike height (low < 1.0, middle, high >= 1.2 m), each as likely as the others, then a row of the cell.
    # Rows alone follow what the launches give: a slow low ball is 1 row in 130 of the stand forehands and a fast
    # one 1 in 4, whatever the size of the bank (main_urdf_2's stationary_reach_v5_coverage does the same).
    # This share of the direct questions is asked without a pose: the world's question alone (target position,
    # racket velocity along the face normal, announcement time). The reference keeps waist and right arm at HOLD,
    # and from the announcement until ``extra_free_after_s`` after the strike nothing scores the upper body
    # against it (body and trunk kernels over pelvis and legs only; ``direct_arm``, ``direct_action``,
    # ``direct_waist_roll`` off): the stroke is the policy's, judged by the strike terms. Afterwards the HOLD
    # reference counts again. Legs, root and left arm of a step row still follow the step clip. A pose is one
    # answer among many, and ours was wrong where it mattered: on the high crossover backhands it asked for a
    # shoulder the joint cannot reach in the time and for none of the trunk roll that does the job (ft30-ft39).
    extra_free: float = 0.0
    extra_free_after_s: float = 0.25
    extra_balance: float = 0.5
    extra_speed_edges: tuple[float, float] = (2.5, 4.0)
    extra_height_edges: tuple[float, float] = (1.0, 1.2)
    # Step strokes among them (rows of family 2 or 3: ``build_questions.py --step-beyond``, the balls beyond
    # the stand strokes' reach). The Planner cannot announce a rally ball earlier than about 0.42 s (it is
    # 0.40-0.53 s from the net to the strike plane; the step clips' legs start 0.56 s before the strike), so the
    # step has to fit the notice: root, legs and left arm follow the step clip with its last
    # ``extra_step_lead_s`` compressed into the notice (the clips' own preparation warp), waist and right arm
    # go from HOLD to the row's strike pose, solved with the pelvis where the clip has it at the strike.
    extra_step_lead_s: float = 0.56
    # With this probability a direct question is announced up to this much earlier than the Planner did.
    # The stroke of a ball is decided by where it is, not by how early it is announced (the actor is told
    # neither): before this, direct questions ended at 0.40 s and clip questions began at 0.46 s, and
    # model_74000 stepped for 0% of the balls left of +0.10 m at 0.44 s, 53% at 0.51 s and 100% at 0.60 s.
    extra_notice_stretch: tuple[float, float] = (0.3, 0.30)
    # (probability, lo, hi): with this probability a direct question announced later than U(lo, hi) seconds before
    # its strike is announced then instead. The bank's notice is main's Planner gated at the net: min(0.405 s,
    # net -> strike plane), 0.22-0.30 s for balls faster than 4 m/s. With the packetizer's commit gate ahead of the
    # net (tools/gate3/planner_main81e_commit_gate.patch) the same balls are announced 0.30-0.40 s ahead; the rest
    # of the probability keeps the bank's notice (a ball hit from close to the net cannot be announced earlier).
    extra_notice_floor: tuple[float, float, float] = (0.0, 0.30, 0.40)
    # Three terms for these questions only. After 1000 iterations with the reference alone (ft19) the racket was
    # in place (15 -> 6 cm) but neither moving (0.4-1.1 m/s along the commanded 1.4-2.2) nor facing the ball
    # (46-66 deg off on forehands): the paper's tracking kernels are 0.3 m / 0.4 rad wide and averaged over 14
    # bodies, so they hardly see the arm, and the strike kernels for velocity and face are flat that far out.
    # ``direct_arm``: the waist and right-arm joints against the reference, two scales (0.5 and 0.15 rad).
    # ``direct_face``: (1 + cos) / 2 of the face against the commanded normal, a slope at any angle (model_59000
    # met 60% of these forehand targets with the other side of the blade). ``direct_drive``: the share of the
    # commanded speed the racket has along the commanded direction, capped at 1 (as ``racket_drive``).
    direct_arm: float = 2.0
    direct_arm_std: tuple[float, float] = (0.5, 0.15)
    direct_face: float = 1.0
    direct_drive: float = 1.0
    # ``direct_action``: the commanded waist and arm targets against the reference a little ahead of it. The only
    # term with a gradient straight into the action, not through the learned dynamics. With the three terms above
    # alone (ft20) the racket still reached the strike 21-55 deg off after 1000 iterations and every test suite was
    # below model_59000.
    # How far ahead: a PD joint following a moving target lags it by kd / kp (0.035 s at the waist yaw, 0.067-0.075 s
    # at shoulder and elbow, 0.10 s at the wrist pitch and yaw), and the command acts a control step after the frame
    # it answers. ``direct_action_joint_lag`` adds kd / kp per joint to ``direct_action_lead_s``. With a flat 0.04 s
    # (up to ft27) the arm stayed behind its reference in proportion to how fast it had to travel: at the strike
    # each shoulder joint was short by 0.08-0.11 s of its mean speed over the notice (model_78000, 9400 strokes),
    # and on the high backhands 0.15-0.34 m right of the centre line, whose pose is 1.8 rad of shoulder roll from
    # HOLD, the racket arrived 5 cm under a reference that was itself on the target: 65-127 of 170 field balls of
    # that kind returned, getting worse by the checkpoint, against 310-318 of 319 of the lower ones beside them.
    direct_action: float = 1.0
    direct_action_std: float = 0.35
    direct_action_lead_s: float = 0.02
    direct_action_joint_lag: bool = True
    # ... and past it by what gravity takes: a PD joint under a steady load sits load / kp short of its command.
    # The bank carries that offset per strike pose (``strike_load``, build_questions.py: 0.25 rad at the shoulder
    # roll with the arm raised) and for HOLD; the command target moves between the two with the pose. Without it
    # the term asked for the command at the pose itself: the policy split the difference, the trunk pitch stayed
    # 0.12-0.13 rad and the raised shoulder 0.06-0.09 rad short in every checkpoint, and the high backhands
    # 0.15-0.34 m right of the centre line were struck with the blade's upper rim, 5.6-6.0 cm low (162 of 170
    # field balls of that kind returned by model_85000, 81 by ft34's model_90000, 0.4 cm apart).
    direct_action_gravity: bool = True
    # A joint whose strike pose is close to its commanded-position limit goes there ahead of the cubic. Close to
    # the limit a PD joint has little left to pull with (kp times what is left of the command range, less gravity
    # and damping), so the last of the way is slow whatever is commanded: on the high crossover backhands (shoulder
    # roll to -2.20 of -2.48, 1.6 rad from HOLD in 0.4 s) every checkpoint followed the cubic, ran into the limit
    # halfway (command at -2.48 for 80% of these strokes 0.12 s before the strike) and struck at -2.04, the racket
    # 6 cm low. With the command at the limit from the announcement on, the same policy (ft35 model_95000) has the
    # joint at -1.94 halfway and -2.25 at the strike, the racket 3.7 cm low. The time of the right shoulder roll
    # is warped, s -> s + c s (1 - s)^2 (same pose and velocity at the strike, 1 + c times the speed at the
    # start), with c growing from 0 at ``extra_front_margin[0]`` rad from the limit to ``extra_front_load`` at
    # ``extra_front_margin[1]``, when it travels more than 0.4-0.8 rad: 16% of the stand backhands of the bank,
    # 4% of the forehands. Only that joint: elbow and wrist yaw end as close to a limit on most forehands and
    # arrive within 0.02 rad of the cubic. 0 switches it off.
    extra_front_load: float = 1.5
    extra_front_margin: tuple[float, float] = (0.50, 0.25)
    # ... and its command is allowed to jump there: before the strike the smoothness terms (``action_rate``,
    # ``action_accel``) count that joint's action at this share, in proportion to the warp. With the warp alone
    # (ft36, 2000 iterations) the command still took half of the notice to reach the limit, its step growing
    # by 0.07 rad a tick (0.07, 0.15, 0.22, 0.27, 0.29 rad): the second difference of a 1.8 rad jump costs
    # two hundred times what that ramp does, and the joint struck at -2.08 for -2.04 before.
    extra_front_smooth: float = 0.1
    # A stroke announced late is urgent for every joint of it. The Planner commits after the net, so a fast ball
    # (4-5.5 m/s) is announced 0.22-0.30 s ahead, and those are most of what Supergate's suites lose: of the
    # forehands announced under 0.30 s 17 of 119 came back, of the backhands 158 of 260, against 93% and 85% from
    # 0.35 s on (ft42 model_119000, main). Not the ball's speed: the same fast balls announced 0.42 s ahead are
    # struck as well as slow ones (85-94% inside 4 cm, 0.8 m/s, 10 deg; 6-24% at 0.26-0.31 s). At the strike the
    # wrist yaw is 0.21-0.26 rad behind its reference (12-14 deg of face), shoulder pitch 0.12, elbow 0.10, and
    # no arm command was at a limit at any tick: the commands ramp, as the shoulder roll's did (the cubic starts
    # from rest, the smoothness terms price a jump). ``extra_urgent_load`` is the time warp of
    # ``extra_front_load`` for waist and right arm, and the same relief from the smoothness terms, growing from
    # nothing at ``extra_urgent_notice[1]`` seconds of notice to full at ``extra_urgent_notice[0]``. 0 is off.
    extra_urgent_load: float = 1.5
    extra_urgent_notice: tuple[float, float] = (0.30, 0.40)
    # ... and the share of the smoothness terms an urgent stroke's joints keep. The terms count raw actions, and
    # an action unit is 0.075 rad at the wrists, 0.2 at the elbow, 0.375 at the shoulder: a 0.8 rad jump of the
    # wrist yaw is 10.7 units, 25 times the price of the same jump at the shoulder. With the relief at 0.4 (ft46,
    # notice 0.28 s) the policy's command for wrist yaw, elbow and shoulder pitch still followed the reference
    # itself tick for tick, not the reference ahead by the joint's lag that ``direct_action`` asks for (wanted
    # -0.63, -0.93, -1.08 rad in the first three ticks; commanded -0.14, -0.22, -0.34), and the joints arrived
    # 0.1 s late.
    extra_urgent_smooth: float = 0.02
    # With the motors' torque capped (``torque_limit``) the cubic HOLD -> strike reference of an urgent stroke is
    # not something the arm can follow: it asks for its largest acceleration on the first tick (1.5 times what
    # a constant push over the same time needs, more with the warp above), and the pose terms then pay the
    # policy for being where the cubic is at every tick of the approach. This share of them (``direct_arm``,
    # ``direct_action``, ``direct_waist_roll``, and the upper bodies in the tracking means) is dropped, in
    # proportion to the urgency, until ``extra_urgent_free_window_s`` before the strike: how the arm gets to
    # the strike pose in the time it has is left to the strike terms and the torque limit. 0 keeps them.
    extra_urgent_free: float = 0.0
    extra_urgent_free_window_s: float = 0.06
    # The waist roll joint against the reference during a direct stroke, on its own and at a tight scale. A
    # racket 0.55 m from that axis moves 5.5 cm for 0.1 rad of it, and the swinging arm's reaction turns the trunk
    # the other way: the reference asked for -0.06 rad on the high crossover backhands, the joint stayed at +0.01
    # (3.8 of the 5.4 cm the racket was low, ft37 model_99000), the command at its limit for a few ticks only.
    # ``direct_arm`` averages ten joints at 0.15 rad and hardly sees it; the strike terms see it for 0.12 s around
    # the strike, after the command that would have mattered. Bank ``..._roll.npz`` asks for -0.09..-0.23 rad there.
    direct_waist_roll: float = 1.0
    direct_waist_roll_std: float = 0.05
    # The left arm at its reference and at the reference's speed, outside the stroke itself (from
    # ``left_arm_after_s`` after the strike to ``left_arm_before_s`` before the next one, and in every hold).
    # Nothing held its joints: the body terms see where the hand is, which the wrist hardly moves, and a slow
    # drift costs no action rate. With the hand's mass randomized (0.25-1.25 kg) the policy took to trimming
    # with it: over the second after a strike the left wrist bent 30-37 deg and the elbow 10 deg and came back
    # (model_68000, model_74000; 3-6 deg before the randomization), seen on the robot as corrections after the
    # hit. During the stroke the arm stays free to counter the swing (it moves 10-20 deg there, as before).
    # Kernels over the seven joints' rms error (rad) and rms speed error (rad/s).
    # The feet at the reference's own speed (zero in HOLD and in the stand strokes, the clip's in a step), from
    # ``feet_slip_after_stand_s`` after an entry from the official stand (0.265 m between the feet; the policy's
    # stance is 0.45 m, and it slides them out in the second after the 3 s handoff). The position terms let a
    # planted foot creep: on main's stack model_85000 kept sliding the right foot by a centimetre about once a
    # second while it stood (43 cm of path in 17 s for 2.6 cm net; model_59000 17 cm, all of it the stance
    # change), and did the last 2 cm forward of its stance change during the first stroke instead of at the entry.
    # 4 s: the 3 s handoff and the second the stance change takes. With 6 s the first ball of most training
    # episodes (announced 4-6 s after the stand) fell outside the term, and model_91000 still moved its feet
    # 2.6-4.4 cm forward during the first stroke in Gate3.
    feet_slip: float = 1.0
    feet_slip_std: float = 0.08
    feet_slip_after_stand_s: float = 4.0
    # In a hold the two hip-roll commands sit on the joints. How hard the legs press sideways against the floor
    # is a direction no other term sees: the stance stands either way while friction holds. model_59000 held
    # its hip rolls 2-6 mrad from their commands and its feet did not move; model_74000 to model_87000 commanded
    # them 24-30 mrad inwards, and in Gate3's entry replayed (tools/entry_replica.py contact=gate3) the stance
    # crept from 0.45 to 0.41 m in 17 s; on main's stack the right foot slid about a centimetre every second
    # and the weight swung between the feet at 1.5 Hz (the feet "trembling" and the sway in the holds).
    stance_command: float = 1.0
    stance_command_std: float = 0.012
    # The trunk at rest between strokes: the three waist joints at the reference's pose and speed outside the
    # stroke (from ``waist_rest_after_s`` after a strike, when a direct stroke's recovery is over, to
    # ``left_arm_before_s`` before the next). A backhand turns the trunk 35-45 deg and leans it 6-10 deg; in
    # Gate3's holds the waist roll of model_91000 was still moving at 8 deg/s over a 6.7 deg range (model_59000:
    # 3 deg/s, 2.3 deg), seen as the upper body swaying all the time.
    waist_rest: float = 1.0
    waist_rest_std: float = 0.05
    waist_rest_speed_std: float = 0.15
    waist_rest_after_s: float = 0.7
    left_arm: float = 1.0
    left_arm_std: float = 0.2
    left_arm_speed_std: float = 0.4
    left_arm_after_s: float = 0.3
    left_arm_before_s: float = 0.5
    # The keep-out edge of these questions. Supergate's protocol HOME is 0.50 m behind the table edge (strike
    # plane 8 cm over the table); we run it, like everything else since 2026-11-19, with the robot 0.58 m
    # behind the edge (``SUPERGATE_HOME_X=-0.58``), so it is the same edge as ``table_edge_x``.
    extra_table_edge_x: float = 0.58
    # Low balls. Gate3's fixed-HOME serves all arrive 0.82-0.84 m high, below the first percentile
    # of the bank's backhand questions (median 1.03 m). With this probability a question is drawn
    # from the family's rows no higher than ``low_z_max`` (the deploy-plane slab otherwise).
    low_probability: float = 0.25
    low_z_max: float = 0.88
    # ... and this share of them is moved onto the deploy plane (``plane_x``): the bank has no question
    # below 0.98 m there, and Gate3's short low balls at the 0.58 m plane were returned 27 times of 72.
    low_on_plane: float = 0.5
    # The same for the other questions: this share of those not drawn from the plane pool is moved onto
    # the deploy plane. With the station 0.58 m behind the table edge the strike is on the edge (x = 0)
    # and Gate3's standard ball arrives there 0.90-0.92 m high; the plane pool starts at 0.98 m.
    plane_shift: float = 0.5
    # High balls: 8% of the bank's questions are above 1.12 m and they are the weakest height in
    # the simulator (0.95-0.97 against 0.98-0.99). Drawn like the low balls.
    # (0.12 raised that to 0.99 in the simulator, but only in the run that also had the fast balls; off by default.)
    high_probability: float = 0.0
    high_z_min: float = 1.12
    # Wider ball paths: each question's target is moved by U(+-jitter) per axis, so the policy
    # also meets targets between and slightly outside the bank rows (the family boundaries are
    # thin in the bank and Gate3 serves land on them). The teacher's arm adapter is not re-solved.
    target_jitter_pos: float = 0.02
    target_jitter_vel: float = 0.10
    # Side boundaries. Gate3 serves a ball 0.26 m to the right of the station, which the Planner gives
    # to the backhand: the very edge of the backhand questions (first percentile -0.27 m), and with the
    # pelvis still off to the left from the previous step it is 0.30-0.35 m from the robot. Those returns
    # left the table sideways (11 of the 14 lost returns in two 100-ball runs, azimuth off by 12-24 deg).
    # With this probability a question comes from the outer tenth of its family's lateral range and is
    # moved up to ``edge_overreach`` further out, across the boundary to the neighbouring stroke.
    # (0.15 with questions pushed across the boundary did not work; with ``side_consistent`` below it does.)
    edge_probability: float = 0.15
    edge_overreach: float = 0.06
    # Only the forehand/backhand split: stand-forehand questions pushed left across it and stand-backhand
    # ones pushed right (the Planner switches sides at -0.25 m with a 0.04 m hysteresis, so either stroke
    # can be asked for a ball up to 6.5 cm on the other side of the policy's own -0.275 m boundary).
    edge_split_only: bool = True
    # Keep jittered and edge questions inside their own stroke's lateral range (2 mm short of each
    # boundary between neighbouring strokes): step forehand | -0.58 | stand forehand | -0.275 | stand
    # backhand | 0.10 | step backhand. Without it a +-2 cm jitter labelled the same position with both
    # strokes near every boundary, which is where the returns left the table sideways.
    side_consistent: bool = True
    side_bounds: tuple[float, ...] = (-0.58, -0.275, 0.10)
    # rally clock
    immediate_probability: float = 0.35
    next_ball_steps: tuple[int, int] = (0, 80)
    dwell_probability: float = 0.5
    dwell_steps: tuple[int, int] = (125, 275)
    # resets
    stand_probability: float = 0.25
    stand_hold_steps: tuple[int, int] = (200, 300)
    long_hold_probability: float = 0.15
    long_hold_steps: tuple[int, int] = (500, 1000)
    stand_x: tuple[float, float] = (0.03, 0.10)
    stand_yaw: float = 0.3
    # Entry stance. Gate3 enters from the 0.265 m official stand and the feet were found stuck at
    # 0.35 m, a width between the stand and the 0.47 m HOLD reference that a default-stand start
    # never shows. With this probability a stand start begins at a random blend of the two poses.
    stance_blend_probability: float = 0.5
    # Entry handoff. On entering MOTION the Runner cross-fades the official stand controller into
    # the policy over 3 s (PpCommandTransition::BeginImpedance, smootherstep on the two PD torque
    # laws). Without it a stand start hands the policy full authority at once and it learned to
    # throw the feet apart in half a second; under the handoff that became a slow push the floor
    # resisted, and Gate3 was left 0.33 m wide with the trunk rolled 3.5 deg until the first ball.
    # With this probability a stand start runs the handoff (docs/entry_20261118.md).
    entry_handoff_s: float = 3.0
    entry_handoff_probability: float = 0.8
    midstroke_probability: float = 0.5  # reference-state starts placed inside the stroke
    rsi_xy: float = 0.03
    rsi_yaw: float = 0.1
    rsi_joint: float = 0.03
    rsi_lift: float = 0.005
    # Random pushes: at this rate per second an environment's root velocity is kicked by up to these
    # amounts (m/s in x/y and z, rad/s about the vertical). The kick is not something the learned
    # dynamics can predict, so the replay treats it as a window boundary (core/trainer.py).
    push_rate: float = 0.12
    push_xy: float = 0.4
    push_z: float = 0.1
    push_yaw: float = 0.3
    # sensing (uniform half-widths / gaussian std), HOPE recipe
    joint_offset: float = 0.01  # per-env encoder offset: shifts observation and action zero
    noise_gyro: float = 0.2
    noise_joint_pos: float = 0.01
    noise_joint_vel: float = 0.5
    mocap_bias: tuple[float, ...] = (0.005, 0.005, 0.003)
    mocap_noise: tuple[float, ...] = (0.003, 0.003, 0.005)
    mocap_rpy_bias: tuple[float, ...] = (0.005236, 0.005236, 0.008727)
    mocap_rpy_noise: tuple[float, ...] = (0.003491, 0.003491, 0.005236)
    # Mocap dropouts. On the robot the motion-capture link occasionally drops, for up to about a second.
    # The Runner then holds the last base pose it applied (position and attitude: in the 110-D contract both
    # come from mocap), so gravity, heading, base position and the target relative to the base all freeze in
    # the observation while gyro, joints and the strike clock stay live. On main (81e606a0c,
    # pp_planner_input.hpp: kBasePoseHoldAfterS 0.02, kBasePoseRescueAfterS 0.5; PpBasePoseHold) the hold
    # lasts until the sample is 0.5 s old, then the Runner leaves MOTION for the PD stand
    # (pp_runner_control.hpp, ObserveMocap) and MOTION is entered again through the entry handoff; the Runner
    # our Gate3 binaries were built from (7990f18c2) held for 1.0 s. The range covers both.
    # At this rate per second an environment's mocap pose freezes for U(mocap_dropout_s).
    mocap_dropout_rate: float = 0.05
    mocap_dropout_s: tuple[float, float] = (0.06, 1.0)
    target_bias: float = 0.003
    target_noise: float = 0.003
    # The Planner's prediction of the strike is off. On the real sessions of 2026-11-21 (tools/real/commit_replay.py)
    # the command sent 0.40 s ahead missed the real ball at the strike plane by 30 ms / 3.3 cm (y) / 3.5 cm (z)
    # (spread; the miss is 6.8 cm in the median), 19 ms / 2.8 / 2.5 cm at 0.25 s, 11-13 ms / 2 / 1.6 cm at
    # 0.10-0.15 s: it predicts through the bounce. Strokes whose contact was 30 ms or more from the commanded
    # time landed 48% (86% within 15 ms). ``target_error`` = (s, m, m) at 0.40 s of notice, scaled by
    # 0.3 + 0.7 * notice / 0.40: what the actor is told (strike time, target y and z) is off by a draw of it for
    # the whole stroke, the rewards keep the true strike. Alone it can only teach the actor to hedge.
    target_error: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # ... unless the command is refined: at U(lo, hi) seconds before the true strike the error is replaced by a
    # draw of ``target_refine_error`` (a Planner that sends the stroke again once it has seen the bounce; the
    # Runner would have to accept it mid-swing: today the target is frozen at engage). (0, 0) = never.
    target_refine_tts: tuple[float, float] = (0.0, 0.0)
    target_refine_error: tuple[float, float, float] = (0.012, 0.020, 0.018)
    delay_probability: float = 0.5  # odd environments: one-tick q_des FIFO for the episode
    # reward (per second; Table A.2 plus the strike interface)
    action_rate: float = -0.1
    joint_limit: float = -2.0
    # Command smoothness beyond the paper's rate term (which at -0.1 leaves a 20 Hz, 7 mrad idle
    # chatter free): the second difference of the raw action separates chatter from a fast but
    # smooth swing, and in HOLD the command has no reason to move at all.
    action_accel: float = -5.0
    hold_action_rate: float = -10.0
    # Tighter posture terms (core/tracking.py): feet against the teacher, pelvis height.
    feet_position: float = 1.0
    feet_position_std: float = 0.06
    feet_height: float = 0.5  # the lift of a step: 2 cm short is invisible at the 6 cm position scale
    feet_height_std: float = 0.025
    root_height: float = 0.5
    root_height_std: float = 0.04
    # Flat feet. Nothing scored a foot's orientation at a useful scale (the paper's 0.4 rad kernel over
    # 14 bodies), and in Gate3 the left foot came back from every backhand step toes-up by 12-16 deg,
    # 2 cm off the floor with the heel dragging; the teacher lifts it 4 cm with the sole level.
    # 0.5 removes it (largest toe-up 16 -> 5 deg). At the 0.465 m plane it cost table-edge contacts in Gate3's
    # rapid rally (docs/rally_20261118.md); at the 0.58 m plane there are none.
    feet_flat: float = 0.5
    feet_flat_std: float = 0.15
    # Station keeping. Over a 100-ball Gate3 rally (115 s without a reset; a training episode is 40 s)
    # the feet crept 5-8 cm towards the table and the robot turned 4-6 deg; the racket then clipped
    # the table edge before the ball on backhands (one run: 17 of 100 returns lost). The pelvis and the
    # feet are scored against the reference's world position, and the pelvis against its heading.
    # With them the pelvis stays within 1 cm over 100 balls and the turn drops to 2.6 deg on average (max 6).
    station: float = 0.5
    station_std: float = 0.04
    feet_station: float = 0.5
    feet_station_std: float = 0.05
    heading: float = 0.5
    heading_std: float = 0.08
    # The turn is carried by the feet: after 100 balls both soles pointed 4-5 deg left with the pelvis
    # (model_54000: pelvis 5.4 deg on average, 8 deg at most). The feet's world orientation is scored too.
    feet_heading: float = 0.5
    feet_heading_std: float = 0.10
    strike_window_s: float = 0.12
    racket_position: float = 3.0
    racket_position_std: float = 0.10
    racket_velocity: float = 2.0
    racket_velocity_std: float = 0.6
    racket_normal: float = 1.0
    racket_normal_std: float = 0.2
    # The racket point on the contact path at the scale of the blade (radius 8.5 cm). Between the two position
    # kernels (10 cm and 2 cm) a racket 3-7 cm off earns almost the same wherever it is: the narrow one is gone
    # (0.1 at 3 cm), the wide one changes by 0.17. On the high crossover backhands, where the pose is at the end
    # of the reach, every checkpoint from ft30 to ft38 struck 5.4-6.2 cm low whatever was changed underneath
    # (limit margin, gravity offset, shoulder ahead of its cubic, trunk roll in the pose): with one joint brought
    # up, elbow and pelvis gave the height back for velocity and face. At that distance the ball meets the rim:
    # 81-162 of 170 field balls of that kind returned, on differences of 2 cm.
    racket_blade: float = 2.0
    racket_blade_std: float = 0.045
    # Two-scale strike kernels: this share of each strike weight sits on a narrow Gaussian. The wide
    # one alone is flat near the target (0.95 of its maximum at 2.3 cm), so nothing pulled the
    # racket from "inside the hit tolerance" to the centre of the blade.
    strike_fine_share: float = 0.5
    racket_position_fine_std: float = 0.02
    racket_velocity_fine_std: float = 0.2
    racket_normal_fine_std: float = 0.06
    # Drive: the share of the commanded speed the racket has along the commanded direction, capped
    # at 1. The velocity kernels are flat beyond about 1 m/s of error, which is where a ball announced
    # 0.3 s ahead leaves the racket: it arrived in place (position within 7.5 cm for 67%) but nearly
    # at rest (velocity error 1.9 m/s), a block instead of a stroke. This term has a slope everywhere.
    # Only matters with the fast balls above; off with them.
    racket_drive: float = 0.0
    # Return: where the ball goes. The executed racket velocity and face normal are pushed through the
    # simulator's paddle model with the question's own incoming ball, and the outgoing ball velocity is
    # scored against the one the commanded stroke would give (two scales, like the strike kernels). The
    # separate velocity and normal kernels leave the face 4-13 deg off (0.3-0.5 m at the landing point)
    # nearly free; this term weighs every error by what it does to the ball (docs/returns_20261119.md).
    return_velocity: float = 1.0
    return_velocity_std: float = 0.5
    return_velocity_fine_std: float = 0.15
    # Landing: the same executed stroke flown to table height (``flight_landing``) against the landing
    # of the commanded stroke, which is the Planner's landing target up to its own 0.1 m residual. Two
    # scales again: the wide one pulls a ball that would leave the table back, the narrow one centres it.
    landing: float = 2.0
    landing_std: float = 0.40
    landing_fine_std: float = 0.12
    # Table keep-out. The table is not simulated; in Gate3 the blade rim clipped its near edge on
    # the upswing to low balls. Station-relative edge and top (Gate3: station x -0.50, edge 0, top 0.76).
    # ``table_edge_x`` is how far the station is behind the table edge: 0.50 in Gate3's own profile, 0.58 when
    # the robot is placed so that the 0.58 m deploy plane is the table edge (the user's set-up, 2026-11-19).
    table_edge_x: float = 0.58
    table_top_z: float = 0.76
    table_clearance: float = 0.005
    table_keepout: float = -6.0
    table_keepout_std: float = 0.015
    blade_radius: float = 0.085
    # Actuator and joint-range keep-outs: what a stroke may not buy its return with (the user, 2026-11-21:
    # "不要把 reversal 等超限带回来").
    # ``torque_limit``: the PD torque the command asks for at the start of a tick, kp (q_des - q) - kd qd, as a
    # share of the motor's limit; the square of what is above ``torque_limit_margin``, summed over joints. The
    # simulators clamp the torque, so a command far beyond the limit costs nothing there, and the short-notice
    # urgency of ft46/ft47 (smoothness relief on waist and arm) bought its returns that way: in Supergate main
    # 69-75% of the strokes announced under 0.30 s then asked for more than a motor has (median peak 1.17 of
    # the limit, wrist yaw 6 N m and elbow 24 N m first), against 0-7% before. A command riding at the limit
    # moves the joint as fast as one beyond it.
    torque_limit: float = 0.0
    torque_limit_margin: float = 0.8
    # ``range_keepout``: joints within ``range_keepout_margin`` of their hard range or beyond it (MJCF joint
    # range; the q_des limits are 0.9 of it and ``joint_limit`` is linear and weak past them). Since the bank's
    # poses use the trunk's roll (ft39) the waist roll of every step forehand swung to 0.365 rad against the
    # 0.349 rad stop, the command already at the opposite limit.
    # ``range_keepout_joints``: joint-name fragments the keep-out is confined to (empty: every joint). A margin wider
    # than the q_des limits' own 0.035 rad on every joint (ft53: -12, 0.04) also pushed the ankle rolls off their
    # command limit, where the stance foot of a step rides (344 ticks per 100-ball step scene, inside the range):
    # the left step stopped lifting the foot (30 -> 12 of 31) and far balls fell 73.6 -> 70.0%.
    range_keepout: float = 0.0
    range_keepout_margin: float = 0.02
    range_keepout_std: float = 0.02
    range_keepout_joints: tuple[str, ...] = ()
    # Trunk (pelvis and torso) orientation against the teacher, at a tight scale.
    trunk_upright: float = 0.5
    trunk_upright_std: float = 0.12
    termination_softness: float = 0.05


# Official ground-stand PD gains by joint (a3_policy_parameters.hpp, a3_pd_stand_kps / a3_pd_stand_kds; neck 40 / 2).
STAND_KP = dict(waist_yaw=400, waist_roll=500, waist_pitch=500, shoulder_pitch=200, shoulder_roll=200, shoulder_yaw=100,
                elbow=200, wrist_roll=100, wrist_pitch=50, wrist_yaw=50, hip_pitch=1500, hip_roll=400, hip_yaw=300,
                knee=2000, ankle_pitch=500, ankle_roll=500, head_yaw=40, head_pitch=40)
STAND_KD = dict(waist_yaw=4, waist_roll=4, waist_pitch=4, shoulder_pitch=2, shoulder_roll=2, shoulder_yaw=1,
                elbow=1, wrist_roll=1, wrist_pitch=1, wrist_yaw=1, hip_pitch=8, hip_roll=7, hip_yaw=7,
                knee=8, ankle_pitch=5, ankle_roll=5, head_yaw=2, head_pitch=2)


def stand_gains(joints=a3.JOINTS):
    """Official stand ``kp``, ``kd`` in the order of ``joints``."""
    key = lambda n: n.removesuffix("_joint").removeprefix("left_").removeprefix("right_")  # noqa: E731
    return [float(STAND_KP[key(n)]) for n in joints], [float(STAND_KD[key(n)]) for n in joints]


def handoff_weight(elapsed, duration):
    """Weight of the stand controller ``elapsed`` seconds into the entry: 1 -> 0 on a smootherstep."""
    u = (elapsed / max(duration, 1e-9)).clamp(0.0, 1.0)
    return 1 - 10 * u ** 3 + 15 * u ** 4 - 6 * u ** 5


def paddle_contact(v_in, v_racket, normal, spin, radius=0.02, with_spin=False):
    """Outgoing ball velocity of the simulator's paddle model (``PredictPaddleContact`` in
    gate3_ball_contact_model.h, the same model the bank's ``outgoing_velocity`` comes from), batched."""
    n = normal / (normal.norm(dim=-1, keepdim=True) + 1e-9)
    n = torch.where(((v_in - v_racket) * n).sum(-1, keepdim=True) > 0, -n, n)
    rel = v_in + torch.cross(spin, -radius * n, dim=-1) - v_racket
    ns = (rel * n).sum(-1, keepdim=True)
    tan = rel - ns * n
    ts, nsp = tan.norm(dim=-1, keepdim=True), ns.abs()
    e = (0.759 * torch.exp(-0.0441 * nsp)).clamp(0.05, 0.95)
    impulse = torch.minimum(0.52 * ts, 0.5 * (1 + e) * nsp)
    d_tan = -tan * impulse / (ts + 1e-9)
    v_out = v_in - (1 + e) * ns * n + d_tan
    if not with_spin:
        return v_out
    return v_out, spin - torch.cross(n, d_tan, dim=-1) / (2.0 / 3.0 * radius)


def flight_landing(p0, v, w, table_z=0.78, steps=36, dt=0.03, drag=0.12453915600992894, magnus=0.00379260504480258):
    """Where a ball launched from ``p0`` with velocity ``v`` and spin ``w`` comes down to table height
    (ball centre ``table_z``): (B, 2), station-relative like ``p0``. The simulator's flight model
    (quadratic drag, Magnus) on a fixed 30 ms grid with the crossing solved inside its step: accurate to
    about a centimetre, differentiable, and the same error for the two landings a reward compares. The net is
    not modelled: a ball that would hit it just lands short. A ball still in the air after ``steps``
    is taken where it is."""
    g = torch.zeros_like(p0)  # built on the device: nothing may be copied from the host inside a captured update
    g[:, 2] = -9.81
    p, landed, where = p0, torch.zeros_like(p0[:, 0], dtype=torch.bool), p0[:, :2]
    for _ in range(steps):
        half = v + 0.5 * dt * (g - drag * v.norm(dim=-1, keepdim=True) * v + magnus * torch.cross(w, v, dim=-1))
        a = g - drag * half.norm(dim=-1, keepdim=True) * half + magnus * torch.cross(w, half, dim=-1)  # midpoint rule
        v_next = v + a * dt
        p_next = p + 0.5 * (v + v_next) * dt
        cross = (~landed) & (p_next[:, 2] <= table_z) & (v_next[:, 2] < 0)
        # inside the step the ball follows p + v tau + a tau^2 / 2: solve its height for the table
        h = (p[:, 2] - table_z).clamp_min(0.0)
        tau = ((-v[:, 2] - (v[:, 2].square() - 2 * a[:, 2] * h).clamp_min(0.0).sqrt()) / a[:, 2]).clamp(0.0, dt)
        where = torch.where(cross[:, None], p[:, :2] + v[:, :2] * tau[:, None] + 0.5 * a[:, :2] * tau[:, None].square(), where)
        landed = landed | cross
        p, v = p_next, v_next
    return torch.where(landed[:, None], where, p[:, :2])


def _envelope(t, T):
    """Cubic Hermite bump of the arm adapter: value/slope weights of (delta_q, delta_dq)."""
    pre = t < 0
    u = torch.where(pre, (t + T) / T, t).clamp(0.0, 1.0)
    a = torch.where(pre, 3 * u * u - 2 * u ** 3, 1 - 3 * u * u + 2 * u ** 3)
    b = torch.where(pre, (u ** 3 - u * u) * T, u ** 3 - 2 * u * u + u)
    da = torch.where(pre, (6 * u - 6 * u * u) / T, 6 * u * u - 6 * u)
    db = torch.where(pre, 3 * u * u - 2 * u, 3 * u * u - 4 * u + 1)
    inside = (t > -T) & (t < 1.0)
    return [torch.where(inside, x, torch.zeros_like(x)) for x in (a, b, da, db)]


class FourStrokeTask:
    dt = 0.02

    def __init__(self, sim, cfg: TaskCfg | None = None, seed: int = 0):
        self.sim, self.cfg = sim, cfg or TaskCfg()
        cfg, dev, N = self.cfg, sim.device, sim.N
        self.N, self.device = N, dev
        f32 = dict(dtype=torch.float32, device=dev)
        self.gen = torch.Generator(device=dev).manual_seed(seed + 1)
        self.kin = Kinematics(sim.model, a3.JOINTS).to(dev)
        self.track = torch.tensor(self.kin.index(TRACKED), device=dev)  # device index: capture-safe
        self.feet = torch.tensor([TRACKED.index(n) for n in a3.FEET], device=dev)  # positions inside ``track``
        self.trunk = torch.tensor([TRACKED.index("pelvis_link"), TRACKED.index("torso_Link")], device=dev)
        # tracked bodies above the waist (1) and below (0): what a stroke without a pose leaves unscored
        self.upper = torch.tensor([float(n == "torso_Link" or "shoulder" in n or "elbow" in n or "wrist" in n) for n in TRACKED], device=dev)
        self.wrist = self.kin.names.index("right_wrist_yaw_Link")
        site = sim.model.site(a3.RACKET_SITE)
        if sim.model.body(int(site.bodyid[0])).name != "right_wrist_yaw_Link":
            raise ValueError("racket site must be mounted on right_wrist_yaw_Link")
        self.racket_offset = torch.tensor(site.pos, **f32)
        # Points that must stay clear of the table, in the wrist frame: eight on the blade rim (the
        # blade lies in the wrist x-z plane, its normal is +-y), the handle end and the hand.
        angle = torch.arange(8, **f32) * (math.pi / 4)
        rim = torch.stack((torch.cos(angle), torch.zeros(8, **f32), torch.sin(angle)), -1) * cfg.blade_radius
        self.keepout_local = torch.cat((self.racket_offset[None] + rim, torch.tensor(
            [[0.060, 0.019, -0.095], [0.0, 0.0, 0.0]], **f32)))
        # --- action contract -------------------------------------------------------------
        self.default_q = torch.tensor(a3.DEFAULT_Q, **f32)
        self.scale = torch.tensor(a3.ACTION_SCALE, **f32)
        self.soft_lo, self.soft_hi = torch.tensor(a3.QDES_LO, **f32), torch.tensor(a3.QDES_HI, **f32)
        # motor torque limits and hard joint ranges, deploy joint order (TaskCfg.torque_limit, range_keepout)
        self.effort = torch.tensor([float(sim.model.actuator(n + "_pd").forcerange[1]) for n in a3.JOINTS], **f32)
        hard = [sim.model.joint(n).range for n in a3.JOINTS]
        self.hard_lo, self.hard_hi = torch.tensor([float(r[0]) for r in hard], **f32), torch.tensor([float(r[1]) for r in hard], **f32)
        self.active = torch.ones(NJ, **f32)
        self.active[list(HEAD)] = 0.0
        self.keepout = self.active * torch.tensor(
            [float(not cfg.range_keepout_joints or any(k in n for k in cfg.range_keepout_joints)) for n in a3.JOINTS], **f32)
        self.arm = torch.tensor(RIGHT_ARM, device=dev)
        self.left_arm = torch.tensor(LEFT_ARM, device=dev)
        self.hip_roll = torch.tensor([a3.JOINTS.index(f"{side}_hip_roll_joint") for side in ("left", "right")], device=dev)
        self._stroke_cols = torch.zeros(13 + 2 * NJ, dtype=torch.bool, device=dev)  # state columns of the stroke joints
        for j in (2, 5, 8) + RIGHT_ARM:
            self._stroke_cols[13 + j] = self._stroke_cols[13 + NJ + j] = True
        self.stroke_joints = torch.tensor((2, 5, 8) + RIGHT_ARM, device=dev)  # waist yaw/roll/pitch and the right arm
        offset = (torch.rand(N, NJ, generator=self.gen, **f32) * 2 - 1) * cfg.joint_offset
        self.q0 = self.default_q[None] + offset  # per-env action zero and joint_pos observation origin
        # --- clips -> reference states ----------------------------------------------------
        assets = WBT / cfg.assets
        manifest = json.loads((assets / "manifest.json").read_text())
        if list(manifest["joint_order"]) != list(a3.JOINTS) or int(manifest["strike_frame"]) != STRIKE_FRAME:
            raise ValueError("clip manifest does not match the A3 joint order / strike frame")
        clips = []
        for name in FAMILIES:
            z = np.load(assets / f"{name}.npz")
            if z["joint_pos"].shape != (LAST_FRAME + 1, NJ):
                raise ValueError(f"unexpected clip shape in {name}")
            quat = torch.tensor(z["body_quat_w"][:, 0])
            w_b = (quat_to_matrix(quat).transpose(-1, -2) @ torch.tensor(z["body_ang_vel_w"][:, 0])[..., None])[..., 0]
            clips.append(torch.cat((torch.tensor(z["body_pos_w"][:, 0]), quat, torch.tensor(z["body_lin_vel_w"][:, 0]),
                                    w_b, torch.tensor(z["joint_pos"]), torch.tensor(z["joint_vel"])), -1))
        self.clip = torch.stack(clips).to(**f32)  # (4, 140, 75)
        self.hold_ref = self.clip[1, 0].clone()
        self.hold_ref[7:13] = 0.0
        self.hold_ref[13 + NJ:] = 0.0
        self.T = torch.tensor(cfg.minimum_lead, **f32)
        self.L = torch.tensor(cfg.complete_lead, **f32)
        self.short_p = torch.tensor(cfg.short_notice_probability, **f32)
        self.short_any = bool(self.short_p.max() > 0)
        # --- question bank ----------------------------------------------------------------
        bank = np.load(assets / "physical_tuple_bank.npz")
        adapter = np.load(assets / "reference_adapter.npz")
        keep = np.flatnonzero(adapter["qualified"])
        family = adapter["family"][keep].astype(np.int64)
        pos = bank["target_position_offset"][keep]
        lead = np.ceil(np.minimum(bank["time_to_strike"][keep], 1.2 - 1e-5) / self.dt) * self.dt
        ok = lead + 1e-6 >= np.asarray(cfg.minimum_lead)[family]
        # Rows on the other side of the forehand | backhand split than their stroke are not asked: with
        # ``side_consistent`` they would be moved to the split, away from the target their reference was solved
        # for (none at the bank's own split, -0.275; a fifth of the stand forehands when it is moved to -0.34).
        right, split, left = cfg.side_bounds
        ok &= ~((family == 0) & (pos[:, 1] > split)) & ~((family == 1) & (pos[:, 1] < split))
        # ... and the same at the stand | step boundaries when those are moved (step forehand | -0.58 | stand
        # ... | +0.10 | step backhand in the bank; -0.68 and +0.44 with the step strokes of the extra bank).
        ok &= ~((family == 0) & (pos[:, 1] < right)) & ~((family == 2) & (pos[:, 1] > right))
        ok &= ~((family == 1) & (pos[:, 1] > left)) & ~((family == 3) & (pos[:, 1] < left))
        keep, family, pos, lead = keep[ok], family[ok], pos[ok], lead[ok]
        self.q_pos = torch.tensor(pos, **f32)
        self.q_vel = torch.tensor(bank["target_velocity"][keep], **f32)
        self.q_vin = torch.tensor(bank["incoming_velocity"][keep], **f32)  # the ball just before contact
        self.q_spin = torch.tensor(bank["incoming_spin"][keep], **f32)
        self.q_lead = torch.tensor(lead, **f32)
        self.q_dq = torch.tensor(adapter["delta_q"][keep], **f32)
        self.q_ddq = torch.tensor(adapter["delta_dq"][keep], **f32)
        self.q_family = torch.tensor(family, device=dev)
        plane = (pos[:, 0] >= cfg.plane_x[0]) & (pos[:, 0] <= cfg.plane_x[1])
        low = pos[:, 2] <= cfg.low_z_max
        high = pos[:, 2] >= cfg.high_z_min
        self.pools, self.planes, self.lows, self.highs, self.edges = [], [], [], [], []
        for f in range(4):
            rows, slab = np.flatnonzero(family == f), np.flatnonzero((family == f) & plane)
            floor, top = np.flatnonzero((family == f) & low), np.flatnonzero((family == f) & high)
            if len(rows) < 64:
                raise ValueError(f"question pool of {FAMILIES[f]} is too small")
            # a sub-pool that the moved boundaries (side_bounds) leave nearly empty falls back to the stroke's rows
            self.pools.append(torch.tensor(rows, device=dev))
            self.planes.append(torch.tensor(slab if len(slab) >= 64 else rows, device=dev))
            self.lows.append(torch.tensor(floor if len(floor) >= 64 else rows, device=dev))
            self.highs.append(torch.tensor(top if len(top) >= 64 else rows, device=dev))
            y = pos[rows, 1]
            lo, hi = np.quantile(y, [0.1, 0.9])
            self.edges.append((torch.tensor(rows[y <= lo], device=dev), torch.tensor(rows[y >= hi], device=dev)))
        self.bank_rows = int(len(keep))
        self.extra, self.extra_steps = None, False
        if cfg.extra_bank and cfg.extra_probability > 0:
            files = [np.load(p if p.is_absolute() else Path(__file__).resolve().parents[2] / p)
                     for p in map(Path, cfg.extra_bank.split(","))]  # several files: comma-separated
            x = {k: np.concatenate([f[k] for f in files]) for k in (
                "qualified", "target_position_offset", "target_velocity", "incoming_velocity", "incoming_spin",
                "strike_q", "strike_dq", "time_to_strike", "family")}
            rows = np.flatnonzero(x["qualified"])
            if len(rows) < 64:
                raise ValueError("extra question bank is too small")
            cat = lambda t, v: torch.cat((t, torch.tensor(v[rows], **f32)))  # noqa: E731
            self.extra = torch.arange(len(self.q_pos), len(self.q_pos) + len(rows), device=dev)
            self.q_pos, self.q_vel = cat(self.q_pos, x["target_position_offset"]), cat(self.q_vel, x["target_velocity"])
            self.q_vin, self.q_spin = cat(self.q_vin, x["incoming_velocity"]), cat(self.q_spin, x["incoming_spin"])
            pad = torch.zeros(len(rows), self.q_dq.shape[1], **f32)  # no clip adapter: these rows have a strike pose
            self.q_dq, self.q_ddq = torch.cat((self.q_dq, pad)), torch.cat((self.q_ddq, pad))
            has_load = all("strike_load" in f.files for f in files) and cfg.direct_action_gravity
            self.hold_load = torch.tensor(files[0]["hold_load"] if has_load else np.zeros(NJ), **f32)
            # A bank without poses (build_questions.py --no-poses, for ``extra_free``): one row of zeros stands for
            # all of them (two million rows of three (31,) tables are 0.7 GB on the device).
            self.extra_posed = bool(np.any(x["strike_q"][rows]))
            if self.extra_posed:
                self.extra_q, self.extra_dq = torch.tensor(x["strike_q"][rows], **f32), torch.tensor(x["strike_dq"][rows], **f32)
                load = np.concatenate([f["strike_load"] for f in files])[rows] if has_load else np.zeros_like(x["strike_q"][rows])
                self.extra_load = torch.tensor(load, **f32)
            else:
                self.extra_q, self.extra_dq = torch.zeros(1, NJ, **f32), torch.zeros(1, NJ, **f32)
                self.extra_load = self.hold_load[None].clone()
            self.extra_front = self._front_load(self.extra_q)
            # The Planner's lead is taken when it commits; the policy sees the target on its next 50 Hz tick.
            self.q_lead = torch.cat((self.q_lead, torch.tensor(
                np.maximum(np.floor(x["time_to_strike"][rows] / self.dt), 5) * self.dt, **f32)))
            self.q_family = torch.cat((self.q_family, torch.tensor(x["family"][rows].astype(np.int64), device=dev)))
            self.extra_steps = bool((x["family"][rows] >= 2).any())
            if cfg.extra_balance > 0:  # rows of the extra bank sorted by (stroke, speed bin, height bin)
                fam = torch.tensor(x["family"][rows].astype(np.int64), device=dev)
                speed = torch.bucketize(torch.tensor(np.linalg.norm(x["incoming_velocity"][rows], axis=1), **f32),
                                        torch.tensor(cfg.extra_speed_edges, **f32), right=True)
                height = torch.bucketize(torch.tensor(x["target_position_offset"][rows][:, 2], **f32),
                                         torch.tensor(cfg.extra_height_edges, **f32), right=True)
                key = fam * 9 + speed * 3 + height
                count = torch.bincount(key, minlength=36)
                self.extra_sorted = self.extra[torch.argsort(key, stable=True)]
                self.extra_cell_len, self.extra_cell_start = count, torch.cumsum(count, 0) - count
                full = (count.view(4, 9) >= 16)  # a cell needs a few rows to be a kind of ball
                self.extra_cell_count = full.sum(1)
                # per stroke its populated cells first: (4, 9) cell numbers
                self.extra_cells = torch.argsort((~full).long(), dim=1, stable=True)
        # --- per-env timeline -------------------------------------------------------------
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=dev)  # noqa: E731
        self.family, self.row = z(N, dtype=torch.long), z(N, dtype=torch.long)
        self.k, self.hold = z(N, dtype=torch.long), z(N, dtype=torch.long)
        self.wait = torch.full((N,), -1, dtype=torch.long, device=dev)  # ticks until the next ball; -1 = none scheduled
        self.ep_step = z(N, dtype=torch.long)
        self.since_stand = torch.full((N,), 10 ** 6, dtype=torch.long, device=dev)  # ticks since an entry from the stand
        self.delay = z(N, dtype=torch.bool)
        self.short = z(N, dtype=torch.bool)  # the current question was announced late
        self.rush = z(N, dtype=torch.bool)  # ... less than the shortest preparation of any clip ahead
        self.jit_pos, self.jit_vel = z(N, 3), z(N, 3)  # per-question target displacement
        self.mocap_out = z(N, dtype=torch.long)  # ticks left of the current mocap dropout
        self.mocap_pose = z(N, 7)  # the base pose held during it
        self.mocap_pose[:, 3] = 1.0
        self.direct = z(N, dtype=torch.bool)  # the current question's reference is a direct stroke (extra bank)
        self.free = z(N, dtype=torch.bool)  # ... asked without a pose (``extra_free``)
        self.notice = torch.ones(N, **f32)  # ... announced this long before its strike
        self._hold_joints = self.hold_ref[13:13 + NJ].clone()
        self.entry = torch.full((N,), -1, dtype=torch.long, device=dev)  # ticks into the entry handoff; -1 = none
        a, b, c = cfg.side_bounds  # lateral range per family: stand fh, stand bh, step fh, step bh
        self._side_lo = torch.tensor([a + 0.002, b + 0.002, -9.0, c + 0.002], **f32)
        self._side_hi = torch.tensor([b - 0.002, c - 0.002, a - 0.002, 9.0], **f32)
        kp_s, kd_s = stand_gains()
        self.stand_kp, self.stand_kd = torch.tensor(kp_s, **f32), torch.tensor(kd_s, **f32)
        self.policy_kp, self.policy_kd = torch.tensor(a3.KP, **f32), torch.tensor(a3.KD, **f32)
        self.action_lead = cfg.direct_action_lead_s + (
            (self.policy_kd / self.policy_kp)[self.stroke_joints] if cfg.direct_action_joint_lag else 0.0)
        half = 0.5 * (math.log(sim.cfg.pd_scale[1]) - math.log(sim.cfg.pd_scale[0]))
        self._pd_log = (0.5 * (math.log(sim.cfg.pd_scale[1]) + math.log(sim.cfg.pd_scale[0])), half)
        self.mocap_bias, self.mocap_rpy, self.target_bias = z(N, 3), z(N, 3), z(N, 3)
        self.target_err, self.target_err_late, self.refine_at = z(N, 3), z(N, 3), z(N)  # (time, y, z) the actor is told wrong
        self._target_error = torch.tensor(cfg.target_error, **f32)
        self._refine_error = torch.tensor(cfg.target_refine_error, **f32)
        self._yz = torch.tensor((0.0, 1.0, 1.0), **f32)
        self._tape = None
        self._idle_pos, self._idle_vel = torch.tensor(IDLE_POS, **f32), torch.tensor(IDLE_VEL, **f32)
        # Device constants: nothing may be copied from the host inside a captured update.
        self._mocap_noise = torch.tensor(cfg.mocap_noise, **f32)
        self._mocap_rpy_noise = torch.tensor(cfg.mocap_rpy_noise, **f32)
        # --- metrics ----------------------------------------------------------------------
        self._m = dict(strikes=z(4), pos=z(4), comp=z(4), vb=z(4), pos_err=z(4), vel_err=z(4), normal_err=z(4),
                       short_strikes=z(4), short_comp=z(4), rush_strikes=z(4), rush_comp=z(4), table=z(4),
                       return_err=z(4), return_ok=z(4), landing_err=z(4), landing_ok=z(4),
                       falls=z(1), resets=z(1), torque=z(1), torque_stroke=z(1), stroke_ticks=z(1), beyond=z(1), terms={})
        self._stroke_table = z(N)  # deepest table penetration since the announcement
        self.max_step_reward = self.dt * (sum(TRACK_WEIGHTS.values()) + cfg.racket_position
                                          + cfg.racket_velocity + cfg.racket_normal + cfg.racket_drive
                                          + cfg.return_velocity + cfg.landing
                                          + cfg.feet_position + cfg.feet_height + cfg.root_height
                                          + cfg.trunk_upright + cfg.feet_flat + cfg.station + cfg.feet_station
                                          + cfg.heading + cfg.feet_heading
                                          + cfg.direct_arm + cfg.direct_face + cfg.direct_drive + cfg.direct_action
                                          + cfg.left_arm + cfg.feet_slip + cfg.stance_command + cfg.waist_rest
                                          + cfg.direct_waist_roll + cfg.racket_blade)

    # ------------------------------------------------------------------ random helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def _randint(self, lo, hi, n):
        return torch.randint(int(lo), int(hi) + 1, (n,), generator=self.gen, device=self.device)

    # ------------------------------------------------------------------ questions and clock
    def _new_question(self, ids):
        """Draw a bank row for ``ids`` and start its clock at the announcement frame."""
        cfg, n = self.cfg, len(ids)
        family = self._randint(0, 3, n)
        row = torch.zeros(n, dtype=torch.long, device=self.device)
        use_plane = self._rand(n) < cfg.plane_probability
        u = self._rand(n)
        # Drawn only when enabled, so the question stream of earlier evaluations is unchanged.
        use_low = self._rand(n) < cfg.low_probability if cfg.low_probability > 0 else torch.zeros_like(use_plane)
        use_high = (self._rand(n) < cfg.high_probability) & ~use_low if cfg.high_probability > 0 else torch.zeros_like(use_plane)
        special = use_low | use_high
        reach = torch.zeros(n, device=self.device)  # lateral displacement of an edge question
        if cfg.edge_probability > 0:  # drawn only when enabled
            use_edge = (self._rand(n) < cfg.edge_probability) & ~special
            outer = self._rand(n) < 0.5  # the family's upper (left) edge, else its lower (right) one
            if cfg.edge_split_only:
                use_edge = use_edge & (family <= 1)
                outer = family == 0  # stand forehand moves left over the split, stand backhand right
            reach = torch.where(use_edge, self._rand(n) * cfg.edge_overreach * torch.where(outer, 1.0, -1.0), reach)
            special = special | use_edge
        for f in range(4):
            mask = family == f
            for pool, pick in ((self.planes[f], mask & use_plane & ~special), (self.pools[f], mask & ~use_plane & ~special),
                               (self.lows[f], mask & use_low), (self.highs[f], mask & use_high)):
                row = torch.where(pick, pool[(u * len(pool)).long().clamp(max=len(pool) - 1)], row)
            if cfg.edge_probability > 0:
                for pool, pick in ((self.edges[f][0], mask & use_edge & ~outer), (self.edges[f][1], mask & use_edge & outer)):
                    row = torch.where(pick, pool[(u * len(pool)).long().clamp(max=len(pool) - 1)], row)
        extra = torch.zeros_like(use_plane)
        if self.extra is not None:  # drawn only when enabled
            extra = self._rand(n) < cfg.extra_probability
            pick = self.extra[(self._rand(n) * len(self.extra)).long().clamp(max=len(self.extra) - 1)]
            if cfg.extra_balance > 0:  # drawn only when enabled
                f = self.q_family[pick]
                cells = self.extra_cell_count[f]
                j = (self._rand(n) * cells).long().clamp(max=8)
                cell = f * 9 + self.extra_cells[f, j]
                size = self.extra_cell_len[cell]
                inside = (self._rand(n) * size).long().clamp(max=len(self.extra_sorted) - 1)
                even = self.extra_sorted[(self.extra_cell_start[cell] + inside).clamp(max=len(self.extra_sorted) - 1)]
                pick = torch.where((self._rand(n) < cfg.extra_balance) & (cells > 0) & (size > 0), even, pick)
            row = torch.where(extra, pick, row)
            if cfg.extra_free > 0:  # drawn only when enabled
                self.free[ids] = extra & (self._rand(n) < cfg.extra_free)
            family = torch.where(extra, self.q_family[row], family)
            reach = torch.where(extra, torch.zeros_like(reach), reach)
            use_low, use_plane = use_low & ~extra, use_plane | extra  # no plane move: they are on the deploy plane
        self.family[ids], self.row[ids] = family, row
        lead = self.q_lead[row]
        if self.extra is not None and cfg.extra_notice_stretch[0] > 0:  # drawn only when enabled
            early = extra & (self._rand(n) < cfg.extra_notice_stretch[0])
            more = torch.round(self._rand(n) * cfg.extra_notice_stretch[1] / self.dt) * self.dt
            lead = torch.where(early, lead + more, lead)
        if self.extra is not None and cfg.extra_notice_floor[0] > 0:  # drawn only when enabled
            share, lo, hi = cfg.extra_notice_floor
            floor = torch.round((lo + self._rand(n) * (hi - lo)) / self.dt) * self.dt
            lead = torch.where(extra & (self._rand(n) < share) & (lead < floor), floor, lead)
        short = torch.zeros(n, dtype=torch.bool, device=self.device)
        if self.short_any:  # no draws otherwise: the question stream of earlier evaluations is unchanged
            low = cfg.short_notice_min_s
            late = torch.ceil((low + self._rand(n) * (self.T[family] - low).clamp_min(0.0)) / self.dt - 1e-4) * self.dt
            short = (self._rand(n) < self.short_p[family]) & (late < lead - 1e-6) & ~extra
            lead = torch.where(short, late, lead)
        rush = torch.zeros(n, dtype=torch.bool, device=self.device)
        if cfg.rush_probability > 0:  # drawn only when enabled
            a, b = cfg.rush_notice_s
            late = torch.ceil((a + self._rand(n) * (b - a)) / self.dt - 1e-4) * self.dt
            rush = (self._rand(n) < cfg.rush_probability) & (late < lead - 1e-6) & ~extra
            lead = torch.where(rush, late, lead)
        self.direct[ids], self.notice[ids] = extra, lead
        self.short[ids] = short & ~rush & ~extra
        self.rush[ids] = rush | extra  # scored together: questions announced less than any clip's preparation ahead
        if cfg.target_jitter_pos > 0 or cfg.target_jitter_vel > 0:  # drawn only when enabled
            self.jit_pos[ids] = (self._rand(n, 3) * 2 - 1) * cfg.target_jitter_pos
            self.jit_vel[ids] = (self._rand(n, 3) * 2 - 1) * cfg.target_jitter_vel
        elif cfg.edge_probability > 0:
            self.jit_pos[ids] = 0.0
        if cfg.edge_probability > 0:
            self.jit_pos[ids, 1] += reach
        if cfg.low_on_plane > 0 and cfg.low_probability > 0:  # drawn only when enabled
            move = use_low & (self._rand(n) < cfg.low_on_plane)
            x = cfg.plane_x[0] + self._rand(n) * (cfg.plane_x[1] - cfg.plane_x[0])
            self.jit_pos[ids, 0] = torch.where(move, x - self.q_pos[row, 0], self.jit_pos[ids, 0])
        if cfg.plane_shift > 0:  # drawn only when enabled
            move = ~use_plane & ~use_low & (self._rand(n) < cfg.plane_shift)
            x = cfg.plane_x[0] + self._rand(n) * (cfg.plane_x[1] - cfg.plane_x[0])
            self.jit_pos[ids, 0] = torch.where(move, x - self.q_pos[row, 0], self.jit_pos[ids, 0])
        if cfg.side_consistent:
            # The policy is not told which stroke is meant: it can only read it from where the ball is.
            # A displaced target must therefore stay on its own stroke's side of the lateral boundaries,
            # or the same position is asked as a forehand and as a backhand and the policy learns a blend.
            y = self.q_pos[row, 1] + self.jit_pos[ids, 1]
            lo, hi = self._side_lo[family], self._side_hi[family]
            if self.extra is not None and not self.extra_steps:  # all stand strokes over the whole side: only the split binds
                split = cfg.side_bounds[1]
                lo = torch.where(extra, torch.where(family % 2 == 0, -9.0, split + 0.002), lo)
                hi = torch.where(extra, torch.where(family % 2 == 0, split - 0.002, 9.0), hi)
            self.jit_pos[ids, 1] += (y.clamp(lo, hi) - y)
        self._stroke_table[ids] = 0.0
        self.k[ids] = STRIKE_FRAME - torch.round(lead / self.dt).long()
        self.wait[ids] = -1
        self.target_bias[ids] = self._randn(n, 3) * cfg.target_bias
        if max(cfg.target_error) > 0:  # drawn only when enabled
            horizon = 0.3 + 0.7 * lead.clamp(max=0.6) / 0.40
            self.target_err[ids] = self._randn(n, 3) * self._target_error * horizon[:, None]
            lo, hi = cfg.target_refine_tts
            self.refine_at[ids] = lo + self._rand(n) * (hi - lo)
            self.target_err_late[ids] = self._randn(n, 3) * self._refine_error

    def advance(self):
        """One control tick of the exogenous rally clock."""
        cfg = self.cfg
        held = self.hold > 0
        self.hold = (self.hold - 1).clamp(min=0)
        self.k = self.k + (~held).long()
        self.ep_step += 1
        self.since_stand += 1
        self.entry = torch.where(self.entry >= 0, self.entry + 1, self.entry)
        self.entry = torch.where(self.entry.float() * self.dt >= cfg.entry_handoff_s, torch.full_like(self.entry, -1), self.entry)
        # From +0.5 s after the strike the next ball is scheduled once.
        schedule = (self.k >= NEXT_BALL_FRAME) & (self.wait < 0) & ~held
        n = int(self.N)
        arrival = self._randint(*cfg.next_ball_steps, n)
        arrival = torch.where(self._rand(n) < cfg.immediate_probability, torch.zeros_like(arrival), arrival)
        arrival = torch.where(self._rand(n) < cfg.dwell_probability, self._randint(*cfg.dwell_steps, n), arrival)
        self.wait = torch.where(schedule, arrival, self.wait)
        arrived = (self.wait == 0) & ~held
        self.wait = torch.where(self.wait > 0, self.wait - 1, self.wait)
        self.k = self.k.clamp(max=LAST_FRAME)
        ids = arrived.nonzero().flatten()
        if len(ids):
            self._new_question(ids)
        if cfg.mocap_dropout_rate > 0:  # drawn only when enabled
            a, b = cfg.mocap_dropout_s
            start = (self.mocap_out == 0) & (self._rand(n) < cfg.mocap_dropout_rate * self.dt)
            ticks = torch.round((a + self._rand(n) * (b - a)) / self.dt).long().clamp(min=1)
            self.mocap_pose = torch.where(start[:, None], self.sim.state()[:, :7], self.mocap_pose)
            self.mocap_out = torch.where(start, ticks, (self.mocap_out - 1).clamp(min=0))
        self._tape = None

    def _reference(self):
        """Reference state (N, 75) and public clock for the current timeline."""
        f, k = self.family, self.k
        T, L = self.T[f], self.L[f]
        step = None
        if self.extra is not None and self.extra_steps:  # a direct step: the clip's preparation fits the notice
            step = self.direct & (f >= 2)
            T = torch.where(step, self.notice.clamp_min(self.dt), T)
            L = torch.where(step, torch.maximum(T, torch.full_like(T, self.cfg.extra_step_lead_s)), L)
        r = (STRIKE_FRAME - k).float() * self.dt
        u = (r / T).clamp(0.0, 1.0)
        smooth = u * u * (3 - 2 * u)
        src = r + (L - T) * smooth
        pre_frame = torch.round(STRIKE_FRAME - src / self.dt).long().clamp(min=0)
        rate = torch.where(r > 0, 1 + (L - T) / T * 6 * u * (1 - u), torch.ones_like(r))
        frame = torch.where(r > 0, pre_frame, k)
        startup = self.hold > 0
        in_hold = startup | (k >= LAST_FRAME)
        ref = self.clip[f, frame].clone()
        ref[:, 7:13] *= rate[:, None]
        ref[:, 13 + NJ:] *= rate[:, None]
        tts = torch.where(startup, torch.full_like(r, IDLE_TTS), r)
        a, b, da, db = _envelope(-tts, T)
        row = self.row
        arm_q = 13 + self.arm
        arm_qd = 13 + NJ + self.arm
        ref[:, arm_q] += a[:, None] * self.q_dq[row] + b[:, None] * self.q_ddq[row]
        ref[:, arm_qd] += da[:, None] * self.q_dq[row] + db[:, None] * self.q_ddq[row]
        if self.extra is not None:
            direct = self._direct(row, r)
            if step is not None:  # root, legs and left arm of the step clip under the direct waist and right arm
                direct = torch.where((step[:, None] & ~self._stroke_cols[None]), ref, direct)
            ref = torch.where(self.direct[:, None], direct, ref)
        ref = torch.where(in_hold[:, None], self.hold_ref[None], ref)
        return ref, tts, in_hold, startup

    def _load(self):
        """Gravity's command offset of the stroke joints for the current reference of a direct stroke (N, 10)."""
        joints = self.stroke_joints
        if self.extra is None or not hasattr(self, "_blend"):
            return torch.zeros(self.N, len(joints), device=self.device)
        i = self._extra_row(self.row)
        hold = self.hold_load[joints]
        return (hold + self._blend[:, joints] * (self.extra_load[i][:, joints] - hold)) * self.direct[:, None].float()

    def _extra_row(self, row):
        """Index of bank row ``row`` in the extra bank's pose tables (row 0 for a bank without poses)."""
        i = (row - self.extra[0]).clamp(min=0)
        return i if getattr(self, "extra_posed", True) else torch.zeros_like(i)

    def _smooth(self, tts, in_hold):
        """Weight of each joint's action in the smoothness terms this tick (N, NJ): 1, less for a joint going
        ahead of the cubic before the strike of a direct stroke (``TaskCfg.extra_front_smooth``)."""
        cfg = self.cfg
        w = torch.ones(self.N, NJ, device=self.device)
        if self.extra is None or cfg.extra_front_load <= 0:
            return w
        i = self._extra_row(self.row)
        on = (self.direct & (tts > 0) & ~in_hold.bool())[:, None].float()
        relief = (1 - cfg.extra_front_smooth) * (self.extra_front[i] / cfg.extra_front_load).clamp(0.0, 1.0)
        urgent = self._urgency()
        if urgent is not None:
            relief = torch.maximum(relief, (1 - cfg.extra_urgent_smooth) * urgent)
        return w - on * relief

    def _approach(self, tts, in_hold):
        """Share of the pose terms dropped this tick (N,): an urgent direct stroke on its way to the strike
        (``TaskCfg.extra_urgent_free``)."""
        cfg = self.cfg
        urgent = self._urgency() if cfg.extra_urgent_free > 0 and self.extra is not None else None
        if urgent is None:
            return torch.zeros_like(tts)
        on = (self.direct & ~in_hold.bool() & (tts > cfg.extra_urgent_free_window_s)).float()
        return cfg.extra_urgent_free * urgent.amax(-1) * on

    def _urgency(self):
        """How urgent the current stroke is for each joint (N, NJ) in [0, 1]: by its announcement time, for waist
        and right arm (``TaskCfg.extra_urgent_load``). None when off."""
        cfg = self.cfg
        if cfg.extra_urgent_load <= 0:
            return None
        late, early = cfg.extra_urgent_notice
        u = ((early - self.notice) / (early - late)).clamp(0.0, 1.0)
        joints = torch.zeros(NJ, device=u.device)
        joints[list((2, 5, 8) + RIGHT_ARM)] = 1.0
        return u[:, None] * joints

    def _front_load(self, strike_q):
        """Time-warp strength per row and joint (N, NJ) of direct strokes: see ``TaskCfg.extra_front_load``."""
        cfg = self.cfg
        c = torch.zeros_like(strike_q)
        if cfg.extra_front_load <= 0:
            return c
        arm = torch.tensor(RIGHT_ARM[1:2], device=strike_q.device)  # the shoulder roll
        travel = strike_q[:, arm]
        pose = self.hold_ref[13:13 + NJ][arm] + travel
        left = torch.where(travel > 0, self.soft_hi[arm] - pose, pose - self.soft_lo[arm])  # to the limit ahead
        far, near = cfg.extra_front_margin
        c[:, arm] = cfg.extra_front_load * ((far - left) / (far - near)).clamp(0.0, 1.0) * ((travel.abs() - 0.4) / 0.4).clamp(0.0, 1.0)
        return c

    def _direct(self, row, r):
        """Direct-stroke reference ``r`` seconds before (after, if negative) the strike of extra row ``row``:
        HOLD -> strike pose over the question's notice, from rest and arriving with the strike velocity, then
        back to rest at HOLD over ``extra_recovery_s`` (N, 75). Root and legs are the HOLD reference."""
        i = FourStrokeTask._extra_row(self, row)
        dq, v = self.extra_q[i], self.extra_dq[i]
        free = getattr(self, "free", None)  # questions asked without a pose (``extra_free``): waist and arm stay at HOLD
        if free is not None and len(free) == len(row):
            dq, v = dq * (~free)[:, None], v * (~free)[:, None]
        pre = (r > 0)[:, None]
        D = torch.where(pre, self.notice.clamp_min(self.dt)[:, None], torch.full_like(dq[:, :1], self.cfg.extra_recovery_s))
        s = (1 - r[:, None] / D).clamp(0.0, 1.0)
        c = self.extra_front[i]  # joints that go ahead of the cubic before the strike (``extra_front_load``)
        urgent = FourStrokeTask._urgency(self)
        if urgent is not None:
            c = torch.maximum(c, self.cfg.extra_urgent_load * urgent[:len(row)])
        u = torch.where(pre, s + c * s * (1 - s).square(), (-r[:, None] / D).expand_as(dq)).clamp(0.0, 1.0)
        du = torch.where(pre, 1 + c * (1 - s) * (1 - 3 * s), torch.ones_like(u))
        rise = u * u * (3 - 2 * u)
        a = torch.where(pre, rise, 1 - rise)
        self._blend = a  # how much of the strike pose the reference has now (the gravity load follows it)
        b = torch.where(pre, u ** 3 - u * u, u ** 3 - 2 * u * u + u) * D
        da = torch.where(pre, 6 * u * (1 - u), -6 * u * (1 - u)) / D * du
        db = torch.where(pre, 3 * u * u - 2 * u, 3 * u * u - 4 * u + 1) * du
        moving = (u < 1) | pre
        out = self.hold_ref[None].repeat(len(row), 1)
        q = out[:, 13:13 + NJ] + a * dq + b * v
        out[:, 13:13 + NJ] = torch.minimum(torch.maximum(q, self.soft_lo), self.soft_hi)
        out[:, 13 + NJ:] = torch.where(moving, da * dq + db * v, torch.zeros_like(v))
        return out

    def tape(self):
        """Exogenous record of the current tick (cached until the clock advances)."""
        if self._tape is None:
            ref, tts, in_hold, startup = self._reference()
            row = self.row
            tpos, tvel = self.q_pos[row] + self.jit_pos, self.q_vel[row] + self.jit_vel
            idle = startup[:, None]
            # what the Planner got wrong about this stroke (TaskCfg.target_error): the early draw until the refinement
            live = (~startup & ~in_hold.bool()).float()[:, None]
            err = torch.where(((self.refine_at <= 0) | (tts > self.refine_at))[:, None], self.target_err, self.target_err_late) * live
            self._tape = dict(
                ref=ref, tts=tts, in_hold=in_hold.float(), idle=startup.float(),
                tpos=tpos, tvel=tvel, vin=self.q_vin[row], spin=self.q_spin[row],
                table_edge=torch.where(self.direct, self.cfg.extra_table_edge_x, self.cfg.table_edge_x),
                direct=self.direct.float(),
                sign=torch.where(self.family % 2 == 0, 1.0, -1.0),
                family=self.family.float(),
                # what the actor is told: the idle packet before the first ball, else the biased target
                obs_tpos=torch.where(idle, self._idle_pos[None], tpos + self.target_bias + err * self._yz),
                obs_tts=tts + err[:, 0],
                obs_tvel=torch.where(idle, self._idle_vel[None], tvel),
                mocap_bias=self.mocap_bias, mocap_rpy=self.mocap_rpy, q0=self.q0,
                mocap_hold=(self.mocap_out > 0).float(), mocap_pose=self.mocap_pose,
                load=self._load(),
                smooth=self._smooth(tts, in_hold),
                # a stroke without a pose, from its announcement to shortly after the strike
                swing=torch.maximum((self.free & self.direct & ~in_hold.bool() & (tts > -self.cfg.extra_free_after_s)).float(),
                                    self._approach(tts, in_hold)),
                settled=(self.since_stand.float() * self.dt >= self.cfg.feet_slip_after_stand_s).float(),
                # weight of the stand controller in the command executed this tick (entry handoff)
                handoff=torch.where(self.entry >= 0, handoff_weight(
                    self.entry.float() * self.dt, self.cfg.entry_handoff_s), torch.zeros_like(tts)),
            )
        return self._tape

    def timed_out(self):
        return self.ep_step >= self.cfg.episode_steps

    # ------------------------------------------------------------------ resets
    def reset(self, ids, initial=False):
        cfg, n, dev = self.cfg, len(ids), self.device
        self._new_question(ids)
        stand = self._rand(n) < cfg.stand_probability
        long = self._rand(n) < cfg.long_hold_probability
        hold = torch.where(stand, self._randint(*cfg.stand_hold_steps, n), torch.zeros(n, dtype=torch.long, device=dev))
        hold = torch.where(long, self._randint(*cfg.long_hold_steps, n), hold)
        # Reference-state starts: half begin at the announcement, half inside the stroke.
        k0 = self.k[ids]
        mid = (~stand) & (hold == 0) & (self._rand(n) < cfg.midstroke_probability)
        span = (120 - k0).clamp(min=1)
        self.k[ids] = torch.where(mid, k0 + (self._rand(n) * span).long(), k0)
        self.hold[ids] = hold
        self.ep_step[ids] = self._randint(0, cfg.episode_steps - 1, n) if initial else 0
        odd = ids % 2 == 1
        self.delay[ids] = odd & (self._rand(n) < cfg.delay_probability)
        self.mocap_bias[ids] = self._randn(n, 3) * torch.tensor(cfg.mocap_bias, device=dev)
        self.mocap_rpy[ids] = self._randn(n, 3) * torch.tensor(cfg.mocap_rpy_bias, device=dev)
        self._tape = None
        ref = self._reference()[0][ids]
        s = ref.clone()
        s[:, 0:2] += (self._rand(n, 2) * 2 - 1) * cfg.rsi_xy
        s[:, 2] += cfg.rsi_lift
        s[:, 3:7] = quat_mul(quat_from_yaw((self._rand(n) * 2 - 1) * cfg.rsi_yaw), s[:, 3:7])
        joints = s[:, 13:13 + NJ] + (self._rand(n, NJ) * 2 - 1) * cfg.rsi_joint
        s[:, 13:13 + NJ] = torch.minimum(torch.maximum(joints, self.soft_lo), self.soft_hi)
        # Default stand: the deploy entry pose, displaced from the station and yawed.
        d = torch.zeros(n, 75, device=dev)
        d[:, 0] = cfg.stand_x[0] + self._rand(n) * (cfg.stand_x[1] - cfg.stand_x[0])
        d[:, 2] = 1.0684
        d[:, 3:7] = quat_from_yaw((self._rand(n) * 2 - 1) * cfg.stand_yaw)
        d[:, 13:13 + NJ] = self.q0[ids]
        if cfg.stance_blend_probability > 0:  # drawn only when enabled
            blend = self._rand(n) * (self._rand(n) < cfg.stance_blend_probability).float()
            d[:, 13:13 + NJ] += blend[:, None] * (self._hold_joints - self.default_q)[None]
            d[:, 2] += blend * (self.hold_ref[2] - 1.0684)
        self.entry[ids] = -1
        self.since_stand[ids] = torch.where(stand, torch.zeros_like(self.k[ids]), torch.full_like(self.k[ids], 10 ** 6))
        self.mocap_out[ids] = 0
        if cfg.entry_handoff_s > 0 and cfg.entry_handoff_probability > 0:  # drawn only when enabled
            handed = stand & (self._rand(n) < cfg.entry_handoff_probability)
            self.entry[ids] = torch.where(handed, torch.zeros_like(self.k[ids]), self.entry[ids])
        self.sim.set_state(ids, torch.where(stand[:, None], d, s))

    # ------------------------------------------------------------------ action contract
    def qdes(self, a, q0=None):
        """Raw action -> commanded PD target: affine, clamped to the soft limits, head at default."""
        q0 = self.q0 if q0 is None else q0
        q = torch.minimum(torch.maximum(q0 + self.scale * a, self.soft_lo), self.soft_hi)
        return torch.where(self.active.bool(), q, q0)

    def gains(self, ctx):
        """The worlds' policy PD gains from the simulation context (WarpSim._randomize)."""
        mid, half = self._pd_log
        kp = self.policy_kp * torch.exp(mid + half * ctx[:, 5:5 + NJ])
        if self.sim.fit_kp is not None:  # the stiffness the identified actuators realize (SimCfg.actuator_fit)
            lo, hi = self.sim.cfg.actuator_blend
            lam = 0.5 * (lo + hi) + 0.5 * (hi - lo) * ctx[:, 5 + 2 * NJ]
            kp = kp * (1.0 + lam[:, None] * (self.sim.fit_kp[None] - 1.0))
        return kp, self.policy_kd * torch.exp(mid + half * ctx[:, 5 + NJ:5 + 2 * NJ])

    def handoff(self, s, q_exec, tape, kp, kd):
        """The command of a tick inside the entry handoff, for policy command ``q_exec`` and the
        world's policy gains ``kp``, ``kd``.

        The Runner cross-fades the two PD torque laws with stand weight ``h``:
        ``tau = h [kps (q0 - q) - kds qd] + (1 - h) [kp (q_exec - q) - kd qd]``, sent as one PD with
        blended gains and a stiffness-weighted target. Returns ``(q_model, q_pd, kp_pd, kd_pd)``:
        what the simulator executes, and ``q_model``, the target that gives the same torque under
        the policy gains at state ``s``. The learned dynamics take ``q_model``, so they need no new
        input, and the policy's authority ``d q_model / d q_exec = 1 - h`` is in the gradient.
        """
        h = tape["handoff"][:, None]
        q, qd, q0 = s[:, 13:13 + NJ], s[:, 13 + NJ:], tape["q0"]
        kp_pd = h * self.stand_kp + (1 - h) * kp
        kd_pd = h * self.stand_kd + (1 - h) * kd
        q_pd = (h * self.stand_kp * q0 + (1 - h) * kp * q_exec) / kp_pd
        q_model = q + (kp_pd * (q_pd - q) - (kd_pd - kd) * qd) / kp
        return torch.where(h > 0, q_model, q_exec), q_pd, kp_pd, kd_pd

    # ------------------------------------------------------------------ observation
    def frame(self, s, a_prev, tape, noise=True):
        """The deployed 110-D ``hitter_pure`` frame for state ``s``."""
        cfg = self.cfg
        p, quat, w, q, qd = s[:, 0:3], s[:, 3:7], s[:, 10:13], s[:, 13:13 + NJ], s[:, 13 + NJ:]
        held = tape["mocap_hold"][:, None] > 0  # mocap dropout: the last received base pose, unchanged
        p, quat = torch.where(held, tape["mocap_pose"][:, :3], p), torch.where(held, tape["mocap_pose"][:, 3:7], quat)
        p_m = p + tape["mocap_bias"]
        rpy = tape["mocap_rpy"]
        tpos = tape["obs_tpos"]
        if noise:
            uniform = lambda x, half: x + (torch.rand_like(x) * 2 - 1) * half  # noqa: E731
            w, q, qd = uniform(w, cfg.noise_gyro), uniform(q, cfg.noise_joint_pos), uniform(qd, cfg.noise_joint_vel)
            fresh = (~held).float()  # a held sample is one sample: no new measurement noise
            p_m = p_m + torch.randn_like(p_m) * self._mocap_noise * fresh
            rpy = rpy + torch.randn_like(rpy) * self._mocap_rpy_noise * fresh
            tpos = tpos + torch.randn_like(tpos) * cfg.target_noise * (1 - tape["idle"])[:, None]
        R = quat_to_matrix(quat_mul(quat, quat_exp(rpy)))
        forward = R[:, :2, 0]
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.cat((
            w, q - tape["q0"], qd, a_prev * self.active, -R[:, 2, :], forward,
            -p_m[:, :2], tpos - p_m, tape["obs_tvel"], tape.get("obs_tts", tape["tts"])[:, None],  # tapes made by hand have one clock
        ), -1)

    # ------------------------------------------------------------------ reward and termination
    def table_depth(self, bodies, clearance=0.0, soft=False, edge=None):
        """How far the blade rim, handle or hand reaches below the table top beyond its near edge:
        per point (B, 10). ``soft`` blends the edge in over 2 cm ahead of it (for the reward).
        ``edge`` (B,) is the station-relative edge per row (the tape's), ``table_edge_x`` otherwise."""
        cfg = self.cfg
        edge = cfg.table_edge_x if edge is None else edge[:, None]
        pos, rot = bodies[0][:, self.wrist], bodies[1][:, self.wrist]
        pts = pos[:, None] + (rot[:, None] @ self.keepout_local[None, :, :, None])[..., 0]
        over = torch.sigmoid((pts[..., 0] - edge + 0.02) / 0.01) if soft else (pts[..., 0] > edge).float()
        return (cfg.table_top_z + clearance - pts[..., 2]).clamp_min(0.0) * over

    def racket(self, bodies, sign):
        pos, rot, omega, vel = bodies
        R = rot[:, self.wrist]
        arm = (R @ self.racket_offset[..., None])[..., 0]
        return (pos[:, self.wrist] + arm, vel[:, self.wrist] + torch.cross(omega[:, self.wrist], arm, dim=-1),
                R[:, :, 1] * sign[:, None])

    def reference_bodies(self, ref):
        """Kinematics of reference states (any leading shape); references carry no gradient."""
        with torch.no_grad():
            flat = self.kin(ref.reshape(-1, ref.shape[-1]))
        return tuple(x.reshape(*ref.shape[:-1], *x.shape[1:]) for x in flat)

    def reward(self, s, a, a_prev, tape, detail=False, ref_bodies=None, a_prev2=None, s_prev=None, q_exec=None,
               body_joints=None):
        """Reward of reaching ``s`` with action ``a`` under the tape of the reached tick (dt-scaled).
        ``a_prev2`` is the action before ``a_prev``; without it the second-difference term is zero.
        ``s_prev`` and ``q_exec`` are the state the tick started from and the PD target it ran with: the torque
        the command asked for (``torque_limit``); without them that term is zero.
        For actor training, body_joints=(upper, lower) also returns the two local
        contributions in timestep units. The team total always includes every term once."""
        cfg, ref = self.cfg, tape["ref"]
        bodies = self.kin(s)
        if ref_bodies is None:
            ref_bodies = self.reference_bodies(ref)
        weights = dict(TRACK_WEIGHTS, feet_pos=cfg.feet_position, feet_height=cfg.feet_height,
                       root_height=cfg.root_height, trunk_ori=cfg.trunk_upright, feet_ori=cfg.feet_flat,
                       root_xy=cfg.station, feet_station=cfg.feet_station, root_heading=cfg.heading,
                       feet_heading=cfg.feet_heading)
        tracked = tracking_terms(s, bodies, ref, ref_bodies, self.track, self.feet,
                                 cfg.feet_position_std, cfg.root_height_std, cfg.feet_height_std,
                                 self.trunk, cfg.trunk_upright_std, cfg.feet_flat_std if cfg.feet_flat > 0 else None,
                                 ((cfg.station_std, cfg.feet_station_std, cfg.heading_std)
                                  + ((cfg.feet_heading_std,) if cfg.feet_heading > 0 else ()))
                                 if cfg.station + cfg.feet_station + cfg.heading > 0 else None,
                                 weight=(1 - tape["swing"][:, None] * self.upper)
                                 if "swing" in tape and cfg.extra_free + cfg.extra_urgent_free > 0 else None)
        terms = {k: weights[k] * v for k, v in tracked.items()}
        smooth = tape["smooth"] if "smooth" in tape else 1.0  # per joint: TaskCfg.extra_front_smooth
        joint_rate = smooth * (a - a_prev).square()
        rate = joint_rate.sum(-1)
        terms["action_rate"] = cfg.action_rate * rate
        terms["hold_action_rate"] = cfg.hold_action_rate * tape["in_hold"] * rate
        joint_accel = smooth * (a - 2 * a_prev + a_prev2).square() if a_prev2 is not None else torch.zeros_like(a)
        terms["action_accel"] = cfg.action_accel * joint_accel.sum(-1)
        q = s[:, 13:13 + NJ]
        limit_excess = (self.soft_lo - q).clamp_min(0) + (q - self.soft_hi).clamp_min(0)
        terms["joint_limit"] = cfg.joint_limit * limit_excess.sum(-1)
        # Strike interface: racket on the moving contact path, at the commanded velocity and face.
        tts = tape["tts"]
        gate = ((tts.abs() <= cfg.strike_window_s + 1e-6) & (tape["in_hold"] < 0.5)).float()
        rpos, rvel, rnormal = self.racket(bodies, tape["sign"])
        tvel = tape["tvel"]
        path = tape["tpos"] - tvel * tts[:, None]
        tnormal = tvel / (tvel.norm(dim=-1, keepdim=True) + 1e-6)
        # The straight contact path of a low ball struck upwards runs below the table top and through
        # its near edge (Gate3: the blade rim clipped the edge on 8 of 26 strokes). The path is
        # lifted so that a blade on it clears the top wherever it reaches beyond the edge, and climbs
        # to that height at 45 degrees over the last centimetres before the edge.
        reach = cfg.blade_radius * (1 - tnormal.square()).clamp_min(0.0).sqrt()  # blade half-extent per world axis
        edge = tape["table_edge"]
        before = (edge - reach[:, 0] - path[:, 0]).clamp_min(0.0)  # still to go until the rim is over the edge
        floor = cfg.table_top_z + cfg.table_clearance + reach[:, 2] - before
        path = torch.cat((path[:, :2], torch.maximum(path[:, 2], floor)[:, None]), -1)
        cos = (rnormal * tnormal).sum(-1)
        fine = cfg.strike_fine_share
        k = lambda err, std, tight: (1 - fine) * torch.exp(-err / std ** 2) + fine * torch.exp(-err / tight ** 2)  # noqa: E731
        terms["racket_position"] = cfg.racket_position * gate * k(
            (rpos - path).square().sum(-1), cfg.racket_position_std, cfg.racket_position_fine_std)
        terms["racket_blade"] = cfg.racket_blade * gate * torch.exp(-(rpos - path).square().sum(-1) / cfg.racket_blade_std ** 2)
        terms["racket_velocity"] = cfg.racket_velocity * gate * k(
            (rvel - tvel).square().sum(-1), cfg.racket_velocity_std, cfg.racket_velocity_fine_std)
        terms["racket_normal"] = cfg.racket_normal * gate * k(2 * (1 - cos), cfg.racket_normal_std, cfg.racket_normal_fine_std)
        terms["racket_drive"] = cfg.racket_drive * gate * (
            (rvel * tnormal).sum(-1) / (tvel.norm(dim=-1) + 1e-6)).clamp(0.0, 1.0)
        direct = tape["direct"]  # extra-bank questions: three terms of their own (see TaskCfg.direct_arm)
        posed = direct * (1 - tape["swing"]) if "swing" in tape else direct  # the pose terms: not while a free stroke swings
        arm_err = (q[:, self.stroke_joints] - ref[:, 13 + self.stroke_joints]).square().mean(-1)
        wide, tight = cfg.direct_arm_std
        terms["direct_arm"] = cfg.direct_arm * posed * (1 - tape["in_hold"]) * 0.5 * (
            torch.exp(-arm_err / wide ** 2) + torch.exp(-arm_err / tight ** 2))
        command = self.default_q[self.stroke_joints] + self.scale[self.stroke_joints] * a[:, self.stroke_joints]
        lead = ref[:, 13 + self.stroke_joints] + self.action_lead * ref[:, 13 + NJ + self.stroke_joints] + tape["load"]
        terms["direct_action"] = cfg.direct_action * posed * (1 - tape["in_hold"]) * torch.exp(
            -(command - lead).square().mean(-1) / cfg.direct_action_std ** 2)
        terms["direct_face"] = cfg.direct_face * direct * gate * 0.5 * (1 + cos)
        terms["direct_drive"] = cfg.direct_drive * direct * gate * (
            (rvel * tnormal).sum(-1) / (tvel.norm(dim=-1) + 1e-6)).clamp(0.0, 1.0)
        hips = self.hip_roll
        press = (self.default_q[hips] + self.scale[hips] * a[:, hips] - q[:, hips]).square().mean(-1)
        terms["stance_command"] = cfg.stance_command * tape["settled"] * tape["in_hold"].float() * torch.exp(
            -press / cfg.stance_command_std ** 2)
        roll = self.stroke_joints[1:2]  # a tensor index of one (a 0-dim one cannot be captured in a CUDA graph)
        terms["direct_waist_roll"] = cfg.direct_waist_roll * posed * (1 - tape["in_hold"]) * torch.exp(
            -(q[:, roll] - ref[:, 13 + roll]).square().sum(-1) / cfg.direct_waist_roll_std ** 2)
        waist = self.stroke_joints[:3]
        rest = ((tts > cfg.left_arm_before_s) | (tts < -cfg.waist_rest_after_s) | (tape["in_hold"] > 0.5)).float()
        waist_q = (q[:, waist] - ref[:, 13 + waist]).square().mean(-1)
        waist_qd = (s[:, 13 + NJ + waist] - ref[:, 13 + NJ + waist]).square().mean(-1)
        terms["waist_rest"] = cfg.waist_rest * rest * 0.5 * (
            torch.exp(-waist_q / cfg.waist_rest_std ** 2) + torch.exp(-waist_qd / cfg.waist_rest_speed_std ** 2))
        feet = self.track[self.feet]
        slip = (bodies[3][:, feet, :2] - ref_bodies[3][:, feet, :2]).square().sum(-1).mean(-1)
        terms["feet_slip"] = cfg.feet_slip * tape["settled"] * torch.exp(-slip / cfg.feet_slip_std ** 2)
        stroke = ((tts < cfg.left_arm_before_s) & (tts > -cfg.left_arm_after_s) & (tape["in_hold"] < 0.5)).float()
        left_q = (q[:, self.left_arm] - ref[:, 13 + self.left_arm]).square().mean(-1)
        left_qd = (s[:, 13 + NJ + self.left_arm] - ref[:, 13 + NJ + self.left_arm]).square().mean(-1)
        terms["left_arm"] = cfg.left_arm * (1 - stroke) * 0.5 * (
            torch.exp(-left_q / cfg.left_arm_std ** 2) + torch.exp(-left_qd / cfg.left_arm_speed_std ** 2))
        v_exe, w_exe = paddle_contact(tape["vin"], rvel, rnormal, tape["spin"], with_spin=True)
        v_cmd, w_cmd = paddle_contact(tape["vin"], tvel, tnormal, tape["spin"], with_spin=True)
        out_err = (v_exe - v_cmd).square().sum(-1)
        terms["return_velocity"] = cfg.return_velocity * gate * k(out_err, cfg.return_velocity_std, cfg.return_velocity_fine_std)
        if cfg.landing > 0 or detail:
            # Both balls leave from the commanded strike point: where the racket is is the position terms' business.
            both = flight_landing(torch.cat((tape["tpos"], tape["tpos"])), torch.cat((v_exe, v_cmd)), torch.cat((w_exe, w_cmd)))
            land_err = (both[:len(v_exe)] - both[len(v_exe):]).square().sum(-1)  # one pass for the two flights
            terms["landing"] = cfg.landing * gate * k(land_err, cfg.landing_std, cfg.landing_fine_std)
        depth = self.table_depth(bodies, cfg.table_clearance, soft=True, edge=edge)
        terms["table_keepout"] = cfg.table_keepout * (1 - torch.exp(-depth.square().sum(-1) / cfg.table_keepout_std ** 2))
        if s_prev is not None and q_exec is not None:
            demand = self.torque_demand(s_prev, q_exec)
            terms["torque_limit"] = cfg.torque_limit * (demand - cfg.torque_limit_margin).clamp_min(0.0).square().sum(-1)
        else:
            demand = torch.zeros_like(q)
            terms["torque_limit"] = torch.zeros_like(rate)
        beyond = ((self.hard_lo + cfg.range_keepout_margin - q).clamp_min(0) + (q - self.hard_hi + cfg.range_keepout_margin).clamp_min(0)) * self.keepout
        terms["range_keepout"] = cfg.range_keepout * (1 - torch.exp(-beyond.square().sum(-1) / cfg.range_keepout_std ** 2))
        total = self.dt * sum(terms.values())
        if body_joints is not None:
            if detail:
                raise ValueError("Request body rewards or detailed diagnostics separately")
            # Reuse the very same per-joint errors (including the phase and smooth masks).
            # Racket path/speed, returns, global posture, table/torque/range guards and
            # mixed waist+arm targets stay in the team reward shared by both branches.
            rate_weight = cfg.action_rate + cfg.hold_action_rate * tape["in_hold"][:, None]
            joint_terms = rate_weight * joint_rate + cfg.action_accel * joint_accel + cfg.joint_limit * limit_excess
            upper_ids, lower_ids = body_joints
            upper = joint_terms.index_select(-1, upper_ids).sum(-1) + sum(terms[k] for k in UPPER_REWARD_TERMS)
            lower = joint_terms.index_select(-1, lower_ids).sum(-1) + sum(terms[k] for k in LOWER_REWARD_TERMS if k in terms)
            return total, self.dt * upper, self.dt * lower
        if not detail:
            return total
        info = dict(terms=terms, racket_pos_err=(rpos - tape["tpos"]).norm(dim=-1),
                    table_depth=self.table_depth(bodies, edge=edge).amax(-1),
                    racket_vel_err=(rvel - tvel).norm(dim=-1), racket_cos=cos, racket_approach=(rvel * rnormal).sum(-1),
                    return_err=out_err.sqrt(), landing_err=land_err.sqrt(),
                    torque_ratio=demand.amax(-1),
                    range_excess=(((self.hard_lo - q).clamp_min(0) + (q - self.hard_hi).clamp_min(0)) * self.active).amax(-1))
        return total, info

    def torque_demand(self, s, q_exec):
        """PD torque a command asks for in state ``s`` as a share of each motor's limit (N, NJ), at the nominal
        policy gains. The plant clamps it: above 1 the motor is saturated."""
        tau = self.policy_kp * (q_exec - s[:, 13:13 + NJ]) - self.policy_kd * s[:, 13 + NJ:13 + 2 * NJ]
        return tau.abs() / self.effort.clamp_min(1e-6) * self.active

    def term_prob(self, s, tape):
        return soft_termination(s, tape["ref"], self.cfg.termination_softness)

    def terminated(self, s, tape):
        return termination_error(s, tape["ref"]) > 1.0

    # ------------------------------------------------------------------ critic input
    def critic_obs(self, s, a_prev, tape, ctx):
        """Privileged value input: state, reference error, ball command and sim context."""
        ref = tape["ref"]
        p, quat = s[:, 0:3], s[:, 3:7]
        R = quat_to_matrix(quat)
        to_body = lambda v: (R.transpose(-1, -2) @ v[..., None])[..., 0]  # noqa: E731
        family = torch.nn.functional.one_hot(tape["family"].long(), 4).float()
        return torch.cat((
            represent(s), ref[:, 13:13 + NJ] - s[:, 13:13 + NJ], ref[:, 13 + NJ:], a_prev,
            to_body(ref[:, 0:3] - p), -quat_to_matrix(ref[:, 3:7])[:, 2, :],
            to_body(tape["tpos"] - p), to_body(tape["tvel"]), -p,
            tape["tts"][:, None], torch.stack((tape["in_hold"], tape["idle"], tape["sign"]), -1), family, ctx,
        ), -1)

    def context(self):
        return self.sim.ctx

    # ------------------------------------------------------------------ metrics
    @torch.no_grad()
    def observe(self, s, tape, reward, info, terminated, truncated):
        """Accumulate strike statistics at the exact strike tick (clock at frame 60)."""
        m = self._m
        strike = (self.k == STRIKE_FRAME) & (self.hold == 0) & ~terminated
        fam = torch.nn.functional.one_hot(self.family, 4).float() * strike[:, None].float()
        pos, vel = info["racket_pos_err"], info["racket_vel_err"]
        normal = torch.rad2deg(torch.acos(info["racket_cos"].clamp(-1, 1)))
        m["strikes"] += fam.sum(0)
        m["pos"] += (fam * (pos < 0.075)[:, None]).sum(0)
        comp = (pos < 0.075) & (vel < 0.5) & (normal < 15.0)
        m["comp"] += (fam * comp[:, None]).sum(0)
        m["short_strikes"] += (fam * self.short[:, None]).sum(0)
        # Table: deepest point from the announcement to the strike tick, counted per strike.
        self._stroke_table = torch.maximum(self._stroke_table, info["table_depth"])
        m["table"] += (fam * (self._stroke_table > 0.0)[:, None]).sum(0)
        m["short_comp"] += (fam * (comp & self.short)[:, None]).sum(0)
        m["rush_strikes"] += (fam * self.rush[:, None]).sum(0)
        m["rush_comp"] += (fam * (comp & self.rush)[:, None]).sum(0)
        m["vb"] += (fam * ((pos < 0.095) & (info["racket_approach"] > 0.3))[:, None]).sum(0)
        m["pos_err"] += (fam * pos[:, None]).sum(0)
        m["vel_err"] += (fam * vel[:, None]).sum(0)
        m["normal_err"] += (fam * normal[:, None]).sum(0)
        m["return_err"] += (fam * info["return_err"][:, None]).sum(0)
        m["return_ok"] += (fam * (info["return_err"] < 0.3)[:, None]).sum(0)
        m["landing_err"] += (fam * info["landing_err"][:, None]).sum(0)
        m["landing_ok"] += (fam * (info["landing_err"] < 0.2)[:, None]).sum(0)
        m["falls"] += terminated.float().sum()
        m["resets"] += (terminated | truncated).float().sum()
        swinging = (tape["in_hold"] < 0.5) & (tape["idle"] < 0.5)
        m["torque"] += (info["torque_ratio"] >= 1.0).float().sum()
        m["torque_stroke"] += ((info["torque_ratio"] >= 1.0) & swinging).float().sum()
        m["stroke_ticks"] += swinging.float().sum()
        m["beyond"] += (info["range_excess"] > 0.002).float().sum()
        for k, v in info["terms"].items():
            m["terms"][k] = m["terms"].get(k, 0.0) + v.mean()
        m["steps"] = m.get("steps", 0) + 1

    def metrics(self):
        """Strike rates since the last call, overall and per family."""
        m = self._m
        n = m["strikes"].clamp_min(1)
        total = m["strikes"].sum().clamp_min(1)
        out = {
            "task/strikes": m["strikes"].sum(),
            "task/hit_rate": m["comp"].sum() / total,  # pos < 7.5 cm, vel < 0.5 m/s, normal < 15 deg at the strike tick
            "task/position_pass_rate": m["pos"].sum() / total,
            "task/virtual_hit_rate": m["vb"].sum() / total,
            "task/strike_pos_err_m": m["pos_err"].sum() / total,
            "task/strike_vel_err_mps": m["vel_err"].sum() / total,
            "task/strike_normal_err_deg": m["normal_err"].sum() / total,
            "task/return_err_mps": m["return_err"].sum() / total,  # outgoing ball velocity vs the commanded stroke's
            "task/return_within_0p3_mps": m["return_ok"].sum() / total,
            "task/landing_err_m": m["landing_err"].sum() / total,  # predicted landing vs the commanded stroke's landing
            "task/landing_within_0p2_m": m["landing_ok"].sum() / total,
            "task/falls_per_step": m["falls"].sum() / max(1, m.get("steps", 0)) / self.N,
        }
        out["task/table_touch_rate"] = m["table"].sum() / total  # strokes whose blade, handle or hand dipped below the table top
        ticks = max(1, m.get("steps", 0)) * self.N
        out["task/torque_at_limit_tick_rate"] = m["torque"].sum() / ticks  # ticks whose command asked a motor for its limit or more
        out["task/torque_at_limit_stroke_tick_rate"] = m["torque_stroke"].sum() / m["stroke_ticks"].sum().clamp_min(1)
        out["task/joint_beyond_range_tick_rate"] = m["beyond"].sum() / ticks  # ticks with a joint over 2 mrad beyond its hard range
        out["task/short_notice_strikes"] = m["short_strikes"].sum()
        out["task/short_notice_hit_rate"] = m["short_comp"].sum() / m["short_strikes"].sum().clamp_min(1)
        out["task/rush_strikes"] = m["rush_strikes"].sum()
        out["task/rush_hit_rate"] = m["rush_comp"].sum() / m["rush_strikes"].sum().clamp_min(1)
        for f, name in enumerate(FAMILIES):
            out[f"{name}/hit_rate"] = m["comp"][f] / n[f]
            out[f"{name}/strike_pos_err_m"] = m["pos_err"][f] / n[f]
            out[f"{name}/return_err_mps"] = m["return_err"][f] / n[f]
            out[f"{name}/landing_err_m"] = m["landing_err"][f] / n[f]
            out[f"{name}/short_notice_strikes"] = m["short_strikes"][f].clone()
            out[f"{name}/short_notice_hit_rate"] = m["short_comp"][f] / m["short_strikes"][f].clamp_min(1)
            out[f"{name}/rush_hit_rate"] = m["rush_comp"][f] / m["rush_strikes"][f].clamp_min(1)
        steps = max(1, m.get("steps", 0))
        out.update({f"reward/{k}": v / steps for k, v in m["terms"].items()})
        for key in ("strikes", "pos", "comp", "vb", "pos_err", "vel_err", "normal_err", "short_strikes",
                    "short_comp", "rush_strikes", "rush_comp", "return_err", "return_ok", "landing_err", "landing_ok", "table", "falls", "resets",
                    "torque", "torque_stroke", "stroke_ticks", "beyond"):
            m[key].zero_()
        m["terms"], m["steps"] = {}, 0
        return out

    def state_dict(self):
        return dict(q0=self.q0)

    def load_state_dict(self, state):
        # Encoder offsets are per-environment randomization: keep them on a same-size resume,
        # redraw them (already done in __init__) when the environment count differs.
        if "q0" in state and state["q0"].shape == self.q0.shape:
            self.q0.copy_(state["q0"])
