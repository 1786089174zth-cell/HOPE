from pathlib import Path
import math
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_hope.analytic import state as S
from supermimic_hope.analytic.learner import TapeReplay, clip_rows, cosine_lr
from supermimic_hope.analytic.world_model import (
    PhysicalWorldModel, invariant, quat_exp, quat_log, quat_mul, push,
)


def random_state(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    quat = torch.randn(n, 4, generator=g)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    return S.pack(
        torch.randn(n, 3, generator=g), quat, torch.randn(n, 3, generator=g),
        torch.randn(n, 3, generator=g), torch.randn(n, S.NJ, generator=g), torch.randn(n, S.NJ, generator=g),
    )


def same_rotation(a, b):
    return (a * b).sum(-1).abs()


def test_quaternion_exp_log_round_trip_including_small_angles():
    v = torch.cat((torch.randn(64, 3) * 0.7, torch.randn(8, 3) * 1e-6))
    torch.testing.assert_close(quat_log(quat_exp(v)), v, atol=1e-5, rtol=1e-4)
    zero = torch.zeros(1, 3, requires_grad=True)
    (grad,) = torch.autograd.grad(quat_exp(zero)[:, 1:].sum(), zero)
    torch.testing.assert_close(grad, torch.full((1, 3), 0.5))


def test_increments_invert_apply():
    s0, s1 = random_state(32, 0), random_state(32, 1)
    d = PhysicalWorldModel.increments(s0, s1)
    back = PhysicalWorldModel.apply(s0, d)
    for name in ("p", "v", "wb", "q", "qd"):
        torch.testing.assert_close(S.part(back, name), S.part(s1, name), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(same_rotation(S.part(back, "quat"), S.part(s1, "quat")), torch.ones(32), atol=1e-5, rtol=0)


def test_invariant_features_ignore_heading_and_position():
    s = random_state(16, 2)
    yaw = torch.rand(16) * 2 * math.pi
    turn = torch.stack(((yaw / 2).cos(), torch.zeros(16), torch.zeros(16), (yaw / 2).sin()), -1)
    from supermimic_hope.analytic.world_model import rotate

    moved = S.pack(
        S.part(s, "p") + torch.tensor([3.0, -2.0, 0.0]),
        quat_mul(turn, S.part(s, "quat")),
        rotate(turn, S.part(s, "v")),
        S.part(s, "wb"), S.part(s, "q"), S.part(s, "qd"),
    )
    torch.testing.assert_close(invariant(moved), invariant(s), atol=1e-4, rtol=1e-4)


def test_push_changes_only_root_velocities():
    s = random_state(4, 3)
    delta = torch.randn(4, 6)
    pushed = push(s, delta)
    torch.testing.assert_close(pushed[:, 7:13], s[:, 7:13] + delta)
    torch.testing.assert_close(torch.cat((pushed[:, :7], pushed[:, 13:]), 1), torch.cat((s[:, :7], s[:, 13:]), 1))


def test_replay_masks_stop_at_recorded_episode_ends():
    replay = TapeReplay(12, 4, "cpu", 0.25)
    for t in range(12):
        replay.add(dict(x=torch.full((4,), float(t)), done=torch.full((4,), t == 5)))
    b = replay.sample(200, 6, keys={"x"})
    assert b["env_ids"].max() < 3
    for j in range(1, 6):
        ended = b["done"][:, :j].any(-1)
        assert (b["mask"][ended, j] == 0).all()
        assert (b["mask"][~ended, j] == 1).all()
    torch.testing.assert_close(b["x"][:, 1:] - b["x"][:, :-1], torch.ones(200, 5))


def test_strike_prioritized_sampling_puts_the_strike_inside_the_window():
    T, N, W = 60, 8, 8
    replay = TapeReplay(T, N, "cpu", validation_fraction=0.125)
    for t in range(T):
        # One strike per 40 steps; time to strike counts down 0.02 s per step.
        tts = torch.full((N,), ((20 - t) % 40 - 20) * 0.02)
        replay.add({"ctx/tts": tts, "done": torch.zeros(N, dtype=torch.bool)})
    torch.manual_seed(0)
    b = replay.sample(64, W, strike_fraction=0.5, dt=0.02)
    inside = (b["ctx/tts"].abs() <= 0.12).any(1)
    assert inside[:32].all()
    assert b["env_ids"][:32].max() < N - 1  # validation env excluded
    assert 0 < inside[32:].float().mean() < 1


def test_cosine_learning_rate_matches_upstream_endpoints():
    assert cosine_lr(2e-4, 0, 100) == pytest.approx(2e-4)
    assert cosine_lr(2e-4, 100, 100) == pytest.approx(6e-6)
    assert cosine_lr(2e-4, 50, 100) == pytest.approx(2e-4 * (0.03 + 0.97 * 0.5))


def test_row_clip_bounds_outlier_rows_only():
    g = torch.ones(8, 3)
    g[0] *= 100.0  # one outlier row
    g[1] = 0.0  # an ended row stays out of the median
    out = clip_rows(g, 4.0)
    median = math.sqrt(3.0)
    assert torch.allclose(out[0].norm(), torch.tensor(4.0 * median))
    assert torch.allclose(out[0] / out[0].norm(), g[0] / g[0].norm())
    assert torch.equal(out[2:], g[2:])
    assert torch.equal(out[1], g[1])
    assert torch.equal(clip_rows(torch.zeros(4, 3), 4.0), torch.zeros(4, 3))


def test_resume_restores_cumulative_env_steps():
    from types import SimpleNamespace
    from supermimic_hope.trainer_analytic import env_steps

    sc = SimpleNamespace(warmup_steps=64, utd=0.4)
    assert env_steps({"iter": 20400, "env_step_counter": 51123}, sc) == 51123
    # Older checkpoints: 2.5 vector steps per iteration plus the warmup.
    assert env_steps({"iter": 20400}, sc) == 64 + 51000


def test_kinematics_matches_mujoco_forward():
    mujoco = pytest.importorskip("mujoco")
    from supermimic_hope import a3
    from supermimic_hope.analytic.kinematics import A3Kinematics

    model, _ = a3.build_model()
    names = [model.body(i).name for i in range(1, model.nbody)]
    ix = a3.indices(model)
    kin = A3Kinematics(model, names, list(a3.JOINTS))
    data = mujoco.MjData(model)
    g = torch.Generator().manual_seed(4)
    data.qpos[:3] = [0.1, -0.2, 1.0]
    quat = torch.randn(4, generator=g)
    data.qpos[3:7] = (quat / quat.norm()).numpy()
    data.qpos[ix["qpos"]] = (torch.rand(31, generator=g) - 0.5).numpy()
    data.qvel[:] = (torch.randn(model.nv, generator=g)).numpy()
    mujoco.mj_forward(model, data)
    to = lambda x: torch.tensor(x, dtype=torch.float32)[None]  # noqa: E731
    out = kin(
        to(data.qpos[:3]), to(data.qpos[3:7]), to(data.qpos[ix["qpos"]]), to(data.qvel[:3]),
        to(data.qvel[3:6]), to(data.qvel[ix["dof"]]), com_offset=to(model.body_ipos[1:]),
    )
    torch.testing.assert_close(out.pos[0], to(data.xpos[1:])[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(out.com_pos[0], to(data.xipos[1:])[0], atol=1e-5, rtol=0)
    for i, body in enumerate(range(1, model.nbody)):
        v = torch.zeros(6, dtype=torch.float64).numpy()
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, v, 0)
        torch.testing.assert_close(out.ang_vel[0, i], torch.tensor(v[:3], dtype=torch.float32), atol=1e-4, rtol=0)
        torch.testing.assert_close(out.link_lin_vel[0, i], torch.tensor(v[3:], dtype=torch.float32), atol=1e-4, rtol=0)


def test_replay_host_storage_returns_batches_on_the_target_device():
    T, N, W = 10, 8, 4
    cpu = TapeReplay(T, N, "cpu", 0.25)
    host = TapeReplay(T, N, "cpu", 0.25, storage="cpu")
    for t in range(T):
        rec = dict(x=torch.arange(N, dtype=torch.float32) + 100 * t, done=torch.zeros(N, dtype=torch.bool))
        cpu.add(rec)
        host.add(rec)
    torch.manual_seed(0)
    a = cpu.sample(16, W)
    torch.manual_seed(0)
    b = host.sample(16, W)
    assert torch.equal(a["x"], b["x"]) and torch.equal(a["env_ids"], b["env_ids"])
    # Rows are consecutive steps of one env: x increases by exactly 100 along the window.
    assert torch.equal(a["x"][:, 1:] - a["x"][:, :-1], torch.full((16, W - 1), 100.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_replay_host_storage_feeds_a_cuda_learner():
    T, N, W = 12, 8, 4
    host = TapeReplay(T, N, "cuda", 0.25, storage="cpu")
    for t in range(T):
        host.add(dict(x=torch.arange(N, dtype=torch.float32, device="cuda") + 100 * t,
                      done=torch.zeros(N, dtype=torch.bool, device="cuda")))
    assert host.data["x"].device.type == "cpu" and host.data["x"].is_pinned()
    b = host.sample(16, W)
    torch.cuda.synchronize()
    assert b["x"].device.type == "cuda" and b["env_ids"].device.type == "cuda"
    assert torch.equal(b["x"][:, 1:] - b["x"][:, :-1], torch.full((16, W - 1), 100.0, device="cuda"))


def test_target_critic_polyak_tracks_the_live_critic():
    from types import SimpleNamespace
    from supermimic_hope.analytic.learner import Learner
    cfg = SimpleNamespace(critic_hidden=[8], learning_rate=1e-3, termination_temperature=0.05,
                          critic_target_decay=0.5, get=lambda k, d=None: {"critic_target_decay": 0.5}.get(k, d))
    L = Learner.__new__(Learner)
    L.cfg, L.device, L.critic, L.critic_norm, L.critic_target = cfg, "cpu", None, None, None
    L.target_decay = 0.5
    L._ensure_critic(torch.zeros(1, 3))
    before = [p.clone() for p in L.critic_target.parameters()]
    with torch.no_grad():
        for p in L.critic.parameters():
            p.add_(1.0)
    L.update_target()
    for b, pt, p in zip(before, L.critic_target.parameters(), L.critic.parameters()):
        assert torch.allclose(pt, 0.5 * b + 0.5 * p)
    assert not any(p.requires_grad for p in L.critic_target.parameters())


def test_recorded_terminations_do_not_bootstrap_in_the_objective():
    from supermimic_hope.analytic.learner import bootstrap_cut
    mask_t = torch.tensor([1.0, 1.0, 1.0, 0.0])
    mask_next = torch.tensor([0.0, 0.0, 1.0, 0.0])
    terminal = torch.tensor([True, False, False, False])
    # Row 0 ended by a termination (no future), row 1 by a timeout (bootstrap),
    # row 2 continues, row 3 is past its episode end.
    assert torch.equal(bootstrap_cut(mask_t, mask_next, terminal), torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(bootstrap_cut(mask_t, mask_next), torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_recorded_terminations_do_not_bootstrap_in_the_critic_targets():
    from types import SimpleNamespace
    from supermimic_hope.analytic.learner import Learner
    torch.manual_seed(0)
    cfg = SimpleNamespace(critic_hidden=[8], learning_rate=0.0, termination_temperature=0.05, gamma=0.9, lam=0.5,
                          critic_minibatches=1, critic_epochs=1, get=lambda k, d=None: d)
    L = Learner.__new__(Learner)
    L.cfg, L.device, L.critic, L.critic_norm, L.critic_target = cfg, "cpu", None, None, None
    L.target_decay, L.boot_scale, L.bootstrap_at_terminations = None, 1.0, False
    L._ensure_critic(torch.zeros(1, 3))
    L.critic_norm.update = lambda x: None  # frozen input normalisation
    B = 2
    features = [torch.randn(B, 3), torch.randn(B, 3), torch.randn(B, 3)]
    boot = [torch.randn(B, 3), torch.randn(B, 3)]
    Vb0 = L.value_boot(boot[0]).detach()
    V_end = L.value_boot(features[2]).detach()
    # H = 2. Rows end their recorded episode after step 0 (row 0: termination, row 1: timeout)
    # and after step 1 (row 2 … same rows, second window step is past the end for both).
    data = dict(features=features, boot=boot, rewards=[torch.zeros(B), torch.zeros(B)],
                continues=[torch.ones(B), torch.ones(B)], masks=[torch.ones(B), torch.zeros(B)],
                terminal=[torch.tensor([1.0, 0.0]), torch.zeros(B)])
    out = L.critic_update(data)
    expected = (0.0 + 0.9 * float(Vb0[1])) / 2  # only step 0 carries weight
    assert math.isclose(out["value_target_mean"], expected, rel_tol=1e-5, abs_tol=1e-6)
    # A termination on the last window step blanks the horizon bootstrap too.
    data = dict(features=features[:2], boot=boot[:1], rewards=[torch.zeros(B)], continues=[torch.ones(B)],
                masks=[torch.ones(B)], terminal=[torch.tensor([1.0, 0.0])])
    V1 = L.value_boot(features[1]).detach()
    out = L.critic_update(data)
    assert math.isclose(out["value_target_mean"], 0.9 * float(V1[1]) / 2, rel_tol=1e-5, abs_tol=1e-6)


def test_lambda_weights_mix_n_step_returns_to_one():
    from supermimic_hope.analytic.learner import lambda_weights
    H = 32
    r, b, end = lambda_weights(H, None)
    assert r == [1.0] * H and b == [0.0] * H and end == 1.0
    for lam in (0.8, 0.95):
        r, b, end = lambda_weights(H, lam)
        assert math.isclose(sum(b) + end, 1.0)
        # Reward at step t is in every n-step return with n > t: weight lambda^t.
        for t in range(H):
            assert math.isclose(r[t], sum(b[n] for n in range(t + 1, H)) + end)


def test_critic_survival_weight_ignores_post_failure_imagined_steps():
    from types import SimpleNamespace
    from supermimic_hope.analytic.learner import Learner
    torch.manual_seed(0)
    cfg = SimpleNamespace(critic_hidden=[8], learning_rate=0.0, termination_temperature=0.05, gamma=0.9, lam=0.5,
                          critic_minibatches=1, critic_epochs=1, get=lambda k, d=None: d)
    L = Learner.__new__(Learner)
    L.cfg, L.device, L.critic, L.critic_norm, L.critic_target = cfg, "cpu", None, None, None
    L.target_decay, L.boot_scale, L.bootstrap_at_terminations = None, 1.0, True
    L._ensure_critic(torch.zeros(1, 3))
    L.critic_norm.update = lambda x: None
    B = 1
    features = [torch.randn(B, 3) for _ in range(3)]
    boot = [torch.randn(B, 3), torch.randn(B, 3)]
    # Row fails at step 0 (continue 0): step 1 is a post-failure imagined state.
    data = dict(features=features, boot=boot, rewards=[torch.full((B,), -15.0), torch.full((B,), -15.0)],
                continues=[torch.zeros(B), torch.zeros(B)], masks=[torch.ones(B), torch.ones(B)],
                terminal=[torch.zeros(B), torch.zeros(B)])
    L.critic_survival_weight = False
    both = L.critic_update(dict(data))["value_target_mean"]
    L.critic_survival_weight = True
    first_only = L.critic_update(dict(data))["value_target_mean"]
    assert math.isclose(both, -15.0, abs_tol=1e-5) and math.isclose(first_only, -15.0, abs_tol=1e-5)
    # Same means here, so check the weights directly: the second step must carry no weight.
    cont = torch.stack(data["continues"], 0)
    survived = torch.cumprod(torch.cat((torch.ones_like(cont[:1]), cont[:-1]), 0), 0)
    assert survived.tolist() == [[1.0], [0.0]]
