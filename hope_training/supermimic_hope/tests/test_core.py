"""CPU checks of the standalone SuperMimic core (no simulator, no GPU)."""

from pathlib import Path
import math
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_hope.core import dynamics as D  # noqa: E402
from supermimic_hope.core.learners import CapturedUpdate, cosine_lr, gae_returns  # noqa: E402
from supermimic_hope.core.replay import Tape  # noqa: E402
from supermimic_hope.core.rotations import integrate_body_rate, quat_exp, quat_to_matrix  # noqa: E402
from supermimic_hope.core.task import _envelope  # noqa: E402
from supermimic_hope.core.tracking import soft_termination, termination_error  # noqa: E402


def random_state(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    quat = torch.randn(n, 4, generator=g)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    return D.pack(torch.randn(n, 3, generator=g), quat, torch.randn(n, 3, generator=g), torch.randn(n, 3, generator=g),
                  torch.randn(n, D.NJ, generator=g), torch.randn(n, D.NJ, generator=g))


def test_zero_prediction_is_constant_velocity_motion():
    f = D.Dynamics(ctx_dim=2, hidden=(16,))
    with torch.no_grad():
        f.net[-1].weight.zero_()
        f.net[-1].bias.zero_()
    s = random_state(8)
    nxt = f(s, torch.zeros(8, D.NJ), torch.zeros(8, 2))
    p, quat, v, w, q, qd = D.split(s)
    p2, quat2, v2, w2, q2, qd2 = D.split(nxt)
    torch.testing.assert_close(p2, p + v * 0.02, atol=1e-6, rtol=0)
    torch.testing.assert_close(q2, q + qd * 0.02, atol=1e-6, rtol=0)
    torch.testing.assert_close(v2, v, atol=1e-5, rtol=0)
    torch.testing.assert_close(qd2, qd)
    torch.testing.assert_close(quat2, integrate_body_rate(quat, w, 0.02), atol=1e-6, rtol=0)
    torch.testing.assert_close(quat2.norm(dim=-1), torch.ones(8), atol=1e-6, rtol=0)
    # The world angular velocity is preserved across the frame change.
    torch.testing.assert_close(quat_to_matrix(quat2) @ w2[..., None], quat_to_matrix(quat) @ w[..., None], atol=1e-5, rtol=0)


def test_representation_ignores_heading_and_xy():
    s = random_state(4)
    yaw = quat_exp(torch.tensor([[0.0, 0.0, 0.7]]).expand(4, -1))
    from supermimic_hope.core.rotations import quat_mul, rotate

    t = s.clone()
    t[:, 0:2] += 3.0
    t[:, 3:7] = quat_mul(yaw, s[:, 3:7])
    t[:, 7:10] = rotate(yaw, s[:, 7:10])
    torch.testing.assert_close(D.represent(t), D.represent(s), atol=1e-5, rtol=0)


def test_state_loss_is_zero_at_target_and_bryson_weighted():
    s = random_state(6)
    loss, _ = D.state_loss(s, s)
    assert float(loss) == 0.0
    t = s.clone()
    t[:, 0] += 0.01  # one root-position nominal on one of three axes
    loss, groups = D.state_loss(t, s)
    assert float(groups["root_pos"]) == pytest.approx(1e-4 / 3, rel=1e-3)
    assert float(loss) == pytest.approx((1e-4 / 3) * 3 / 0.01 ** 2 / (3 + 9 + 31 + 31), rel=1e-3)


def test_running_norm_matches_batch_statistics():
    norm = D.RunningNorm(5)
    x = torch.randn(4000, 5) * 3 + 2
    for chunk in x.split(500):
        norm.update(chunk)
    torch.testing.assert_close(norm.mean, x.mean(0), atol=1e-4, rtol=0)
    torch.testing.assert_close(norm.std, x.std(0, unbiased=False), atol=1e-3, rtol=0)


def test_adapter_envelope_hits_the_contact_deltas_and_vanishes_at_both_ends():
    T = torch.tensor([0.5])
    a, b, da, db = _envelope(torch.tensor([0.0]), T)
    assert (float(a), float(b), float(da), float(db)) == pytest.approx((1.0, 0.0, 0.0, 1.0))
    for t in (-0.5, -0.7, 1.0, 1.3):
        assert all(float(x) == 0.0 for x in _envelope(torch.tensor([t]), T))
    # value and slope are consistent: da is the derivative of a
    t = torch.tensor([-0.2], requires_grad=True)
    a, b, da, db = _envelope(t, T)
    assert float(torch.autograd.grad(a, t, retain_graph=True)[0]) == pytest.approx(float(da), rel=1e-4)
    assert float(torch.autograd.grad(b, t)[0]) == pytest.approx(float(db), rel=1e-4)


def test_termination_thresholds_match_the_paper():
    ref = random_state(3)
    ref[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    s = ref.clone()
    assert float(termination_error(s, ref).max()) == 0.0
    s[0, 2] += 0.25  # height deviation at the threshold
    s[1, 3:7] = quat_exp(torch.tensor([[math.radians(78.5), 0.0, 0.0]]))  # tilt: |cos - 1| = 0.8
    err = termination_error(s, ref)
    assert float(err[0]) == pytest.approx(1.0, abs=1e-5) and float(err[1]) == pytest.approx(1.0, abs=5e-3)
    assert float(soft_termination(s, ref)[2]) < 1e-6 and float(soft_termination(s, ref)[0]) == pytest.approx(0.5, abs=1e-3)


def test_tape_windows_stay_inside_one_episode_and_history_is_clamped():
    N, T = 6, 40
    tape = Tape(T, N, "cpu")
    for t in range(55):  # wraps the ring
        done = torch.tensor([(t + 1) % (7 + e) == 0 for e in range(N)])
        tape.add(dict(frame=torch.full((N, 2), float(t)), stamp=torch.full((N,), float(t))), done)
    torch.manual_seed(0)
    out = tape.sample(64, 5, history=3)
    assert (out["episode"][:, 0] == out["episode"][:, -1]).all()
    assert (out["stamp"][:, 1:] - out["stamp"][:, :-1] == 1).all()
    start = out["stamp"][:, 0]
    since = out["since"][:, 0].float()
    hist = out["history"][:, :, 0]  # oldest first
    for k in range(3):
        back = torch.minimum(torch.full_like(since, 3.0 - k), since)
        back = torch.minimum(back, start - (55 - T))  # never before the oldest stored step
        torch.testing.assert_close(hist[:, k], start - back)


def test_gae_with_soft_done_and_cosine_schedule():
    values = torch.zeros(3, 2)
    rewards = torch.ones(3, 2)
    dones = torch.zeros(3, 2)
    dones[1, 0] = 1.0
    ret = gae_returns(values, torch.full((2,), 10.0), rewards, dones, gamma=0.5, lam=1.0)
    assert ret[:, 1].tolist() == pytest.approx([1 + 0.5 * (1 + 0.5 * (1 + 5.0)), 1 + 0.5 * 6.0, 6.0])
    assert ret[:, 0].tolist() == pytest.approx([1.5, 1.0, 6.0])  # nothing after the done step leaks back
    assert cosine_lr(1.0, 0) == pytest.approx(1.0) and cosine_lr(1.0, 250_000) == pytest.approx(0.03)
    assert cosine_lr(1.0, 125_000) == pytest.approx(0.515, abs=1e-3)


def test_captured_update_runs_eagerly_on_cpu():
    calls = []
    update = CapturedUpdate(lambda b: calls.append(b["x"].sum()) or {"y": b["tape"]["z"] * 2})
    out = update(dict(x=torch.ones(3), tape=dict(z=torch.ones(2))))
    assert len(calls) == 1 and out["y"].tolist() == [2.0, 2.0]


def test_hitting_check_flags_only_real_regressions():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from check_hitting import ERRORS, RATES, compare

    base = {k: 0.5 for k in RATES} | {k: 1.0 for k in ERRORS} | {"task/falls_per_step": 0.0}
    assert all(ok for *_, ok in compare(dict(base), base))
    better = dict(base) | {"task/hit_rate": 0.9, "task/strike_pos_err_m": 0.5}
    assert all(ok for *_, ok in compare(better, base))
    within = dict(base) | {"step_forehand/hit_rate": 0.46}
    assert all(ok for *_, ok in compare(within, base, tol=0.05))
    for change in ({"step_forehand/hit_rate": 0.44}, {"task/strike_vel_err_mps": 1.2}, {"task/falls_per_step": 1e-4}):
        failed = [k for k, _, _, ok in compare(dict(base) | change, base, tol=0.05) if not ok]
        assert failed == list(change)


def test_posture_terms_see_a_short_step_and_a_crouch_that_the_paper_kernels_ignore():
    from supermimic_hope.core.tracking import tracking_terms

    n, bodies = 3, 5
    pos = torch.zeros(n, bodies, 3)
    pos[:, :, 0] = torch.arange(bodies).float() * 0.1
    pos[:, 0, 2] = 1.0
    rot = torch.eye(3).expand(n, bodies, 3, 3).clone()
    zero = torch.zeros(n, bodies, 3)
    state = torch.zeros(n, 75)
    state[:, 2], state[:, 3] = 1.0, 1.0
    ref = state.clone()
    track, feet = torch.arange(bodies), torch.tensor([3, 4])
    same = tracking_terms(state, (pos, rot, zero, zero), ref, (pos, rot, zero, zero), track, feet)
    assert all(torch.allclose(v, torch.ones(n)) for v in same.values())
    short = pos.clone()
    short[:, 3, 1] += 0.07  # one foot 7 cm from where the teacher puts it
    out = tracking_terms(state, (short, rot, zero, zero), ref, (pos, rot, zero, zero), track, feet)
    assert out["body_pos"][0] > 0.98 and out["feet_pos"][0] < 0.6
    low_state, low = state.clone(), pos.clone()
    low_state[:, 2] -= 0.03
    low[:, 0, 2] -= 0.03  # pelvis 3 cm below the teacher
    out = tracking_terms(low_state, (low, rot, zero, zero), ref, (pos, rot, zero, zero), track, feet)
    assert out["root_pos"][0] > 0.98 and out["root_height"][0] < 0.6
    scuff = pos.clone()
    scuff[:, 3, 2] -= 0.02  # one foot 2 cm below the teacher's lift
    out = tracking_terms(state, (scuff, rot, zero, zero), ref, (pos, rot, zero, zero), track, feet)
    assert out["feet_pos"][0] > 0.9 and out["feet_height"][0] < 0.75
    assert set(tracking_terms(state, (pos, rot, zero, zero), ref, (pos, rot, zero, zero), track)) == {
        "root_pos", "root_ori", "body_pos", "body_ori", "body_lin_vel", "body_ang_vel"}


def test_entry_handoff_is_the_runner_crossfade_and_scales_the_policy_authority():
    from types import SimpleNamespace

    from supermimic_hope import a3
    from supermimic_hope.core.task import FourStrokeTask, handoff_weight, stand_gains

    # smootherstep weight of the stand controller: 1 at entry, 0 from the end on, 0.5 halfway
    w = handoff_weight(torch.tensor([0.0, 1.5, 3.0, 9.0]), 3.0)
    assert torch.allclose(w, torch.tensor([1.0, 0.5, 0.0, 0.0]), atol=1e-6)
    kp_s, kd_s = (torch.tensor(v) for v in stand_gains())
    assert kp_s[a3.JOINTS.index("left_knee_joint")] == 2000 and kd_s[a3.JOINTS.index("right_hip_roll_joint")] == 7
    task = SimpleNamespace(stand_kp=kp_s, stand_kd=kd_s)
    torch.manual_seed(0)
    B, NJ = 5, a3.NJ
    s = torch.randn(B, 13 + 2 * NJ) * 0.2
    q, qd = s[:, 13:13 + NJ], s[:, 13 + NJ:]
    q0, q_exec = torch.randn(B, NJ) * 0.1, torch.randn(B, NJ) * 0.3
    kp, kd = torch.tensor(a3.KP, dtype=torch.float32).expand(B, NJ) * 1.1, torch.tensor(a3.KD, dtype=torch.float32).expand(B, NJ) * 0.9
    h = torch.tensor([0.0, 0.2, 0.5, 0.9, 1.0])
    q_exec = q_exec.requires_grad_(True)
    q_model, q_pd, kp_pd, kd_pd = FourStrokeTask.handoff(task, s, q_exec, dict(handoff=h, q0=q0), kp, kd)
    # no handoff: the policy command and gains untouched; full stand weight: the stand command
    assert torch.equal(q_model[0], q_exec[0]) and torch.allclose(kp_pd[0], kp[0]) and torch.allclose(q_pd[0], q_exec[0])
    assert torch.allclose(q_pd[4], q0[4], atol=1e-6) and torch.allclose(kp_pd[4], kp_s) and torch.allclose(kd_pd[4], kd_s)
    # the executed PD equals the cross-fade of the two torque laws ...
    crossfade = h[:, None] * (kp_s * (q0 - q) - kd_s * qd) + (1 - h[:, None]) * (kp * (q_exec - q) - kd * qd)
    assert torch.allclose(kp_pd * (q_pd - q) - kd_pd * qd, crossfade, atol=1e-3)
    # ... and so does the model target under the policy gains
    assert torch.allclose(kp * (q_model - q) - kd * qd, crossfade, atol=1e-3)
    # the policy's authority over the model target is 1 - h
    grad = torch.autograd.grad(q_model.sum(), q_exec)[0]
    assert torch.allclose(grad, (1 - h)[:, None].expand(B, NJ), atol=1e-6)


def test_landing_reward_model_matches_a_fine_integration_and_the_paddle_model():
    import numpy as np

    from supermimic_hope.core.task import flight_landing, paddle_contact

    # the paddle model on a flat hit: the ball leaves faster than the racket, along its normal, with topspin kept
    v_in, v_r = torch.tensor([[-1.3, 0.0, -2.8]]), torch.tensor([[1.6, 0.1, 1.45]])
    n, spin = v_r / v_r.norm(), torch.tensor([[0.0, -60.0, 0.0]])
    v_out, w_out = paddle_contact(v_in, v_r, n, spin, with_spin=True)
    assert 4.0 < float(v_out.norm()) < 6.5 and float(v_out[0, 0]) > 3.0 and float(v_out[0, 2]) > 2.0
    assert torch.allclose(paddle_contact(v_in, v_r, -n, spin), v_out, atol=1e-6)  # either face of the blade
    # the landing against a 0.2 ms integration of the same flight model
    p0 = torch.tensor([[0.58, -0.3, 1.06]])
    quick = flight_landing(p0, v_out, w_out)[0].numpy()
    p, v, w = p0[0].double().numpy(), v_out[0].double().numpy(), w_out[0].double().numpy()
    while not (p[2] <= 0.78 and v[2] < 0):
        a = np.array([0, 0, -9.81]) - 0.12453915600992894 * np.linalg.norm(v) * v + 0.00379260504480258 * np.cross(w, v)
        p, v = p + v * 2e-4, v + a * 2e-4
    assert np.linalg.norm(quick - p[:2]) < 0.015
    assert 1.9 < quick[0] < 3.3  # beyond the net (1.87 m from the station) for a stroke of this kind
    # a face turned 5 degrees sideways moves the landing by a few tens of centimetres, and the gradient sees it
    yaw = torch.tensor(0.0, requires_grad=True)
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    rot = torch.stack((torch.stack((c, -s_, torch.zeros(()))), torch.stack((s_, c, torch.zeros(()))), torch.tensor([0.0, 0.0, 1.0])))
    v2, w2 = paddle_contact(v_in, v_r, n @ rot.T, spin, with_spin=True)
    y = flight_landing(p0, v2, w2)[0, 1]
    (grad,) = torch.autograd.grad(y, yaw)
    assert 0.03 < abs(float(grad)) * math.radians(1.0) < 0.12  # 3-12 cm per degree


def test_direct_stroke_reference_is_a_smooth_hold_strike_hold_path():
    """The reference of an extra (Supergate) question: rest at HOLD when it is announced, the row's strike
    pose with its strike velocity at the strike, rest at HOLD again after the recovery, and a velocity that
    is the derivative of the position in between. A joint whose pose is near its command limit goes ahead of
    the cubic (``extra_front_load``): same start, same pose and velocity at the strike, more of the way at half
    time."""
    from types import SimpleNamespace

    from supermimic_hope.core.task import RIGHT_ARM, FourStrokeTask, TaskCfg

    torch.manual_seed(0)
    hold = torch.zeros(75)
    hold[2], hold[3], hold[13:44] = 1.0, 1.0, torch.randn(31) * 0.1
    offset, speed = torch.randn(2, 31) * 0.3, torch.randn(2, 31) * 3.0
    roll = RIGHT_ARM[1]  # row 8 raises the shoulder roll 1.6 rad, to 0.28 rad from a limit at -2.48
    offset[1, roll], speed[1, roll] = -2.20 - hold[13 + roll], 0.0
    lo = torch.full((31,), -9.0)
    lo[roll] = -2.48
    task = SimpleNamespace(hold_ref=hold, extra=torch.tensor([7, 8]), extra_q=offset, extra_dq=speed, dt=0.02,
                           cfg=TaskCfg(), soft_lo=lo, soft_hi=torch.full((31,), 9.0))
    task.extra_front = FourStrokeTask._front_load(task, offset)
    assert task.extra_front[1, roll] > 1.0 and task.extra_front.count_nonzero() == 1  # that joint of that row only
    row = torch.tensor([7, 8, 8])
    task.notice = torch.tensor([0.30, 0.24, 0.40])
    at = lambda r: FourStrokeTask._direct(task, row, torch.as_tensor(r, dtype=torch.float32).expand(3).clone())  # noqa: E731
    start = FourStrokeTask._direct(task, row, task.notice.clone())
    assert torch.allclose(start, hold.expand(3, 75), atol=1e-6)
    strike = at(0.0)
    assert torch.allclose(strike[:, 13:44], hold[13:44] + offset[[0, 1, 1]], atol=1e-6)
    assert torch.allclose(strike[:, 44:], speed[[0, 1, 1]], atol=1e-5)
    assert torch.allclose(strike[:, :13], hold[:13].expand(3, 13))  # root stays at HOLD
    assert torch.allclose(at(-task.cfg.extra_recovery_s), hold.expand(3, 75), atol=1e-6)
    assert torch.allclose(at(-1.5), hold.expand(3, 75), atol=1e-6)
    for r in (0.15, 0.05, -0.1, -0.4):
        numeric = (at(r - 1e-3)[:, 13:44] - at(r + 1e-3)[:, 13:44]) / 2e-3  # r counts down
        assert torch.allclose(at(r)[:, 44:], numeric, atol=2e-2)
    half = at(0.20)[2, 13:44] - hold[13:44]  # the row with 0.40 s of notice, half way
    assert 0.70 < half[roll] / offset[1, roll] < 0.85  # the cubic has half of the way here
    task.extra_front = torch.zeros_like(offset)
    plain = at(0.20)[2, 13:44] - hold[13:44]
    assert torch.allclose(plain[roll] / offset[1, roll], torch.tensor(0.5), atol=1e-5)
    others = torch.arange(31) != roll
    assert torch.allclose(half[others], plain[others], atol=1e-6)  # the other joints keep the cubic


def test_mocap_dropout_freezes_the_base_pose_in_the_frame_and_nothing_else():
    """During a dropout the Runner holds the last mocap base pose: gravity, heading, base position and the
    target relative to the base come from the held pose; gyro, joints and the strike clock stay live."""
    from types import SimpleNamespace

    from supermimic_hope.core.task import FourStrokeTask, TaskCfg

    live, held = random_state(2, seed=1), random_state(2, seed=2)
    task = SimpleNamespace(cfg=TaskCfg(), active=torch.ones(31), _mocap_noise=torch.zeros(3), _mocap_rpy_noise=torch.zeros(3))
    tape = dict(mocap_bias=torch.zeros(2, 3), mocap_rpy=torch.zeros(2, 3), obs_tpos=torch.tensor([[0.5, -0.3, 1.0]] * 2),
                obs_tvel=torch.ones(2, 3), tts=torch.tensor([0.3, 0.3]), idle=torch.zeros(2), q0=torch.zeros(2, 31),
                mocap_hold=torch.tensor([1.0, 0.0]), mocap_pose=held[:, :7])
    a_prev = torch.zeros(2, 31)
    out = FourStrokeTask.frame(task, live, a_prev, tape, noise=False)
    from_live = FourStrokeTask.frame(task, live, a_prev, dict(tape, mocap_hold=torch.zeros(2)), noise=False)
    swapped = live.clone()
    swapped[:, :7] = held[:, :7]
    from_held = FourStrokeTask.frame(task, swapped, a_prev, dict(tape, mocap_hold=torch.zeros(2)), noise=False)
    assert torch.equal(out[1], from_live[1])  # no dropout: the live pose
    assert torch.equal(out[0], from_held[0])  # dropout: exactly the frame of the held pose with the live rest
    assert torch.equal(out[0, :96], from_live[0, :96])  # gyro, joints, joint velocities, last action are live
    assert not torch.allclose(out[0, 96:106], from_live[0, 96:106])  # gravity, heading, base xy, relative target
    assert torch.equal(out[0, 106:], from_live[0, 106:])  # commanded velocity and clock


def test_direct_step_reference_is_the_step_clip_under_a_direct_arm():
    """A step row of the extra bank: root, legs and left arm follow the family's clip with its last
    ``extra_step_lead_s`` compressed into the notice, waist and right arm go HOLD -> strike pose; a stand row
    beside it keeps root and legs at HOLD."""
    from types import SimpleNamespace

    from supermimic_hope.core.task import LAST_FRAME, RIGHT_ARM, STRIKE_FRAME, FourStrokeTask, TaskCfg

    torch.manual_seed(0)
    cfg = TaskCfg()
    hold = torch.zeros(75)
    hold[2], hold[3], hold[13:44] = 1.0, 1.0, torch.randn(31) * 0.1
    clip = hold.repeat(4, LAST_FRAME + 1, 1)
    ramp = torch.linspace(0, 1, LAST_FRAME + 1)[None, :, None]
    clip = clip + ramp * torch.randn(4, 1, 75) * 0.2  # every clip drifts away from HOLD, frame by frame
    stroke = torch.tensor((2, 5, 8) + RIGHT_ARM)
    cols = torch.zeros(75, dtype=torch.bool)
    cols[13 + stroke] = cols[44 + stroke] = True
    offset, speed = torch.zeros(2, 31), torch.zeros(2, 31)  # a bank row: waist and right arm only
    offset[:, stroke], speed[:, stroke] = torch.randn(2, 10) * 0.3, torch.randn(2, 10) * 3.0
    notice = torch.tensor([0.40, 0.40])
    task = SimpleNamespace(cfg=cfg, dt=0.02, clip=clip, hold_ref=hold, T=torch.tensor(cfg.minimum_lead), L=torch.tensor(cfg.complete_lead),
                           family=torch.tensor([2, 0]), row=torch.tensor([7, 8]), extra=torch.tensor([7, 8]), extra_steps=True,
                           extra_q=offset, extra_dq=speed, direct=torch.tensor([True, True]), notice=notice, hold=torch.zeros(2, dtype=torch.long),
                           arm=torch.tensor(RIGHT_ARM), q_dq=torch.zeros(9, 7), q_ddq=torch.zeros(9, 7), _stroke_cols=cols,
                           extra_front=torch.zeros(2, 31),
                           soft_lo=torch.full((31,), -9.0), soft_hi=torch.full((31,), 9.0))
    task._direct = lambda row, r: FourStrokeTask._direct(task, row, r)

    def at(r):
        task.k = torch.full((2,), STRIKE_FRAME - round(r / 0.02), dtype=torch.long)
        return FourStrokeTask._reference(task)[0]

    start, strike = at(0.40), at(0.0)
    first = STRIKE_FRAME - round(cfg.extra_step_lead_s / 0.02)
    assert torch.allclose(start[0, ~cols], clip[2, first, ~cols])  # the clip, extra_step_lead_s before its strike
    assert torch.allclose(start[0, 13 + stroke], hold[13 + stroke], atol=1e-6)  # the arm starts at HOLD
    assert torch.allclose(strike[0, ~cols], clip[2, STRIKE_FRAME, ~cols])  # the clip's own strike frame
    assert torch.allclose(strike[0, 13 + stroke], hold[13 + stroke] + offset[0, stroke], atol=1e-6)
    assert torch.allclose(strike[0, 44 + stroke], speed[0, stroke], atol=1e-5)
    middle = at(0.20)
    frame = STRIKE_FRAME - round((0.20 + (cfg.extra_step_lead_s - 0.40) * 0.5) / 0.02)  # the warp's midpoint
    assert torch.allclose(middle[0, :7], clip[2, frame, :7])
    after = at(-0.30)
    assert torch.allclose(after[0, ~cols], clip[2, STRIKE_FRAME + 15, ~cols])  # the clip's own pace after the strike
    for ref in (start, strike, middle, after):  # the stand row: root and legs never leave HOLD
        assert torch.allclose(ref[1, ~cols], hold[~cols], atol=1e-6)


def test_body_weights_leave_the_upper_body_out_of_the_tracking_means():
    """A stroke asked without a pose (TaskCfg.extra_free): bodies with weight 0 do not count in the body and
    trunk kernels, so an arm far from the reference costs nothing while the legs still do."""
    from supermimic_hope.core.tracking import tracking_terms

    torch.manual_seed(0)
    B, n = 3, 6
    eye = torch.eye(3).expand(B, n, 3, 3).clone()
    ref_bodies = (torch.randn(B, n, 3), eye.clone(), torch.zeros(B, n, 3), torch.zeros(B, n, 3))
    s = torch.zeros(B, 75)
    s[:, 3] = 1.0
    s[:, 0:3] = ref_bodies[0][:, 0]
    ref = s.clone()
    track, trunk = torch.arange(n), torch.tensor([0, 3])
    moved = tuple(x.clone() for x in ref_bodies)
    moved[0][:, 3:] += 0.5  # the upper bodies (3, 4, 5) half a metre away, the torso turned
    moved[1][:, 3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    weight = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).expand(B, n)
    plain = tracking_terms(s, moved, ref, ref_bodies, track, trunk=trunk)
    free = tracking_terms(s, moved, ref, ref_bodies, track, trunk=trunk, weight=weight)
    for key in ("body_pos", "body_ori", "trunk_ori"):
        assert torch.allclose(free[key], torch.ones(B), atol=1e-6) and (plain[key] < 0.9).all()
    legs = tuple(x.clone() for x in ref_bodies)
    legs[0][:, 1] += 0.5  # a leg away: counts either way
    assert (tracking_terms(s, legs, ref, ref_bodies, track, trunk=trunk, weight=weight)["body_pos"] < 0.9).all()


def test_torque_demand_is_the_pd_torque_over_the_motor_limit():
    """TaskCfg.torque_limit: what a command asks of each motor, kp (q_des - q) - kd qd over the limit, with the
    passive joints left out; a command 0.3 rad ahead of a wrist at rest is at its 6 N m limit."""
    from types import SimpleNamespace

    from supermimic_hope.core.task import NJ, FourStrokeTask

    ns = SimpleNamespace(policy_kp=torch.full((NJ,), 20.0), policy_kd=torch.full((NJ,), 2.0),
                         effort=torch.full((NJ,), 6.0), active=torch.ones(NJ))
    ns.active[3] = 0.0
    s = torch.zeros(2, 13 + 2 * NJ)
    q_exec = torch.zeros(2, NJ)
    q_exec[0, 5], q_exec[0, 3] = 0.3, 1.0  # at the limit; a passive joint
    s[1, 13 + NJ + 7] = 4.5  # moving at 4.5 rad/s under a command at rest: kd qd = 9 N m
    ratio = FourStrokeTask.torque_demand(ns, s, q_exec)
    assert torch.isclose(ratio[0, 5], torch.tensor(1.0)) and ratio[0, 3] == 0 and torch.isclose(ratio[1, 7], torch.tensor(1.5))
    assert ratio.sum() == ratio[0, 5] + ratio[1, 7]
