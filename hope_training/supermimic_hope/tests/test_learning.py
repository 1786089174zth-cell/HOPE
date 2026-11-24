from pathlib import Path
import sys
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_hope.learning import Replay, Learner


def fixture():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    replay = Replay(24, 8, 114, 31, "cpu", 0.25)
    state = torch.randn(8, 114)
    for t in range(24):
        a = torch.randn(8, 31) * 0.1
        nxt = state.clone()
        nxt[:, :31] += 0.02 * a
        replay.add(
            s=state,
            s1=nxt,
            a=a,
            r=-a.square().mean(-1),
            terminal=torch.zeros(8, dtype=torch.bool),
            cut=torch.full((8,), t == 9),
        )
        state = nxt
    cfg = SimpleNamespace(
        history=4,
        hidden=[32, 32],
        learning_rate=0.001,
        horizon=4,
        gamma=0.99,
        bptt_steps=1,
        bptt_max_kl=1.0e9,
        model_policy_weight=1.0,
    )
    active = torch.ones(31)
    active[[11, 16]] = 0
    actor = torch.nn.Linear(110, 31)
    critic = torch.nn.Linear(4, 1)
    learner = Learner(
        actor,
        114,
        cfg,
        -torch.ones(8, 31),
        torch.ones(8, 31),
        active,
        lambda x: actor(x).square().mean() * 0.001,
        value=lambda s: critic(s[..., 110:]).squeeze(-1),
        value_modules=(critic,),
    )
    learner.critic = critic
    return replay, learner


def test_validation_is_disjoint_and_windows_stop_at_boundaries():
    replay, _ = fixture()
    a = replay.sample(300, 8, 4)
    b = replay.sample(300, 8, 4, True)
    assert a["env_ids"].max() < 6 and b["env_ids"].min() >= 6
    for j in range(1, 8):
        assert (a["mask"][:, j][a["cut"][:, :j].any(-1)] == 0).all()


def test_model_and_bptt_update_do_not_leak_gradients_or_move_passive_outputs():
    replay, learner = fixture()
    b = replay.sample(16, 4, 4)
    before = [p.detach().clone() for p in learner.actor.parameters()]
    assert torch.isfinite(torch.tensor(learner.model_update(b)))
    model_before = [p.detach().clone() for p in learner.models.parameters()]
    critic_before = [p.detach().clone() for p in learner.critic.parameters()]
    stats = learner.policy_update(lambda: b)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    assert stats["bptt_steps_taken"] == 1
    assert any(
        not torch.equal(a, b) for a, b in zip(before, learner.actor.parameters())
    )
    assert learner.project(torch.randn(8, 31))[:, [11, 16]].eq(0).all()
    assert all(p.requires_grad for p in learner.models.parameters())
    assert all(
        torch.equal(a, b) for a, b in zip(model_before, learner.models.parameters())
    )
    # The PPO critic is only a bootstrap here: no update and no stale gradient.
    assert all(
        torch.equal(a, b) for a, b in zip(critic_before, learner.critic.parameters())
    )
    assert all(p.grad is None for p in learner.critic.parameters())


def test_terminal_reset_states_are_not_dynamics_targets():
    replay, learner = fixture()
    b = replay.sample(16, 4, 4)
    b["cut"][:] = True
    b["mask"][:, 1:] = 0
    model = learner.models[0]
    model.observe(b)
    one = model.loss(b)[0]
    b["s1"] += 100000
    two = model.loss(b)[0]
    torch.testing.assert_close(one, two)


def test_world_model_diagnostics_preserve_loss_and_decompose_observation_error():
    replay, learner = fixture()
    b = replay.sample(16, 4, 4)
    model = learner.models[0]
    model.observe(b)
    plain = model.loss(b)
    detailed = model.loss(b, details=True)
    torch.testing.assert_close(plain[0], detailed[0])
    torch.testing.assert_close(plain[1], detailed[1])
    # Actor and privileged errors partition the same normalized state error.
    torch.testing.assert_close(
        detailed[1], (110 * detailed[2][2] + 4 * detailed[2][3]) / 114
    )
    metrics = learner.validate(b)
    assert metrics["wm_validation_reward_mse"] >= 0
    assert 0 <= metrics["wm_validation_valid_fraction"] <= 1


def test_ring_history_never_crosses_new_episode():
    replay = Replay(8, 2, 114, 31, "cpu", 0.5)
    for t in range(12):
        state = torch.full((2, 114), float(t))
        replay.add(
            s=state,
            s1=state + 1,
            a=torch.zeros(2, 31),
            r=torch.zeros(2),
            cut=torch.full((2,), t == 7),
            terminal=torch.zeros(2, dtype=torch.bool),
        )
    b = replay.sample(100, 1, 4)
    starts = b["s"][:, 0, 0]
    assert b["history"][starts >= 8, :, 0].min() >= 8


def test_successor_states_are_reconstructed_exactly_across_wraparound():
    replay = Replay(5, 3, 114, 31, "cpu", 0.34)
    for t in range(13):
        state = torch.full((3, 114), float(t))
        replay.add(
            s=state,
            s1=state + 1,
            a=torch.zeros(3, 31),
            r=torch.zeros(3),
            cut=torch.zeros(3, dtype=torch.bool),
            terminal=torch.zeros(3, dtype=torch.bool),
        )
        if t >= 1:
            b = replay.sample(64, min(replay.size, 5), 1)
            torch.testing.assert_close(b["s1"], b["s"] + 1)
