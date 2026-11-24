from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_hope.contracts import ObservationContract, TerminationMargins
from supermimic_hope.learning import Model, gaussian_kl
from test_learning import fixture


def test_exact_actions_keep_identity_gradient_and_commands_have_no_action_gradient():
    contract = ObservationContract(
        list(range(65, 96)), list(range(31)), [106, 107, 108, 109]
    )
    model = Model(114, 31, 4, [16], contract)
    hist = torch.randn(3, 4, 114)
    action = torch.randn(3, 31, requires_grad=True)
    context = torch.randn(3, 114, requires_grad=True)
    predicted = model(hist, action, context)[0]
    torch.testing.assert_close(predicted[:, 65:96], action, rtol=0, atol=0)
    torch.testing.assert_close(
        predicted[:, 106:110], context[:, 106:110], rtol=0, atol=0
    )
    grad = torch.autograd.grad(predicted[:, 65:96].sum(), action, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.ones_like(action))
    grad = torch.autograd.grad(predicted[:, 109].sum(), action)[0]
    assert grad.eq(0).all()


def test_command_tape_never_copies_auto_reset_or_new_packet():
    contract = ObservationContract(context_indices=[0])
    current = torch.tensor([[1.0], [2.0], [3.0]])
    batch = dict(
        s1=torch.full((3, 1, 1), 999.0),
        mask=torch.tensor([[1.0], [1.0], [0.0]]),
        cut=torch.tensor([[False], [True], [False]]),
    )
    result = contract.context(batch, 0, current)
    torch.testing.assert_close(result, torch.tensor([[999.0], [2.0], [3.0]]))


def test_terminal_cost_decomposition_counts_simultaneous_reasons():
    model = Model(114, 31, 4, [16], termination_costs=[-15.0, -15.0])
    b = dict(
        r=torch.tensor([[-2.0, -17.0, -32.0]]),
        failure=torch.tensor([[[False, False], [True, False], [True, True]]]),
    )
    torch.testing.assert_close(model.residual_target(b), torch.full((1, 3), -2.0))
    # The same probabilities produce the same exact YAML-owned expected cost,
    # rather than learning a second, independently softened terminal penalty.
    hist, action = torch.zeros(1, 4, 114), torch.zeros(1, 31)
    _, reward, _, residual, _ = model.predict(hist, action)
    torch.testing.assert_close(reward - residual, torch.tensor([-15.0]))
    grad = torch.autograd.grad(reward.sum(), model.net[-1].bias)[0][-2:]
    torch.testing.assert_close(grad, torch.tensor([-3.75, -3.75]))


class ConstantModel(nn.Module):
    contract = None
    state_norm = SimpleNamespace(std=lambda: torch.ones(114))

    def forward(self, hist, action, context=None):
        return (
            hist[:, -1],
            torch.ones(hist.shape[0]),
            torch.full((hist.shape[0],), 100.0),
        )


def test_imagined_packet_boundary_keeps_value_bootstrap():
    replay, learner = fixture()
    b = replay.sample(16, 4, 4)
    b["mask"][:] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    learner.cfg.gamma = 0.9
    learner.models = nn.ModuleList([ConstantModel(), ConstantModel()])
    learner.value = lambda s: torch.full(s.shape[:-1], 5.0)
    learner.value_modules = ()
    learner.lower[:] = -100
    learner.upper[:] = 100
    result = learner.policy_update(lambda: b)
    assert abs(result["imagined_return"] - (1 + 0.9 + 0.9**2 * 5)) < 1e-5


def test_physical_failure_margins_use_task_thresholds():
    observer = TerminationMargins.__new__(TerminationMargins)
    robot = SimpleNamespace(
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -0.6]]),
        root_pos_w=torch.tensor([[0.0, 0.0, 0.9], [0.0, 0.0, 0.4]]),
    )
    observer.env = SimpleNamespace(scene={"robot": SimpleNamespace(data=robot)})
    observer.specs = [
        ("bad_orientation", {"limit_angle": 0.7}, None, None),
        ("root_height_below_minimum", {"minimum_height": 0.5}, None, None),
    ]
    assert torch.equal(
        observer.read() > 0, torch.tensor([[False, False], [True, True]])
    )
    observer.specs[0][1]["limit_angle"] = 1.0
    observer.specs[1][1]["minimum_height"] = 0.3
    assert not (observer.read() > 0).any()


def test_failure_geometry_is_supervised_even_when_terminal_state_auto_resets():
    model = Model(
        116, 31, 4, [16], termination_costs=[-15.0, -15.0], failure_temperature=0.05
    )
    b = dict(
        s=torch.zeros(2, 1, 116),
        s1=torch.full((2, 1, 116), 999.0),
        history=torch.zeros(2, 4, 116),
        a=torch.zeros(2, 1, 31),
        r=torch.full((2, 1), -30.0),
        mask=torch.ones(2, 1),
        cut=torch.ones(2, 1, dtype=torch.bool),
        terminal=torch.ones(2, 1, dtype=torch.bool),
        failure=torch.ones(2, 1, 2, dtype=torch.bool),
        failure_margin=torch.tensor([[[0.1, 0.3]], [[0.2, 0.4]]]),
    )
    model.observe(b)
    original = model.loss(b)[0]
    b["s1"].fill_(-999.0)
    torch.testing.assert_close(model.loss(b)[0], original)
    b["failure_margin"] += 2.0
    assert model.loss(b)[0] > original + 1.0


def test_recent_policy_replay_excludes_stale_starts():
    replay, _ = fixture()
    batch = replay.sample(300, 4, 4, recent_steps=4)
    torch.testing.assert_close(
        batch["s"][:, 0],
        replay.data["s"][(replay.ptr - 4) % replay.T, batch["env_ids"]],
    )
    # With exactly one complete recent window, every start is the same time.
    for env_id in batch["env_ids"].unique():
        s = batch["s"][batch["env_ids"] == env_id, 0]
        torch.testing.assert_close(s, s[:1].expand_as(s))


class PushModel(ConstantModel):
    """Rewards a large first action: the actor moves as fast as Adam allows."""

    def forward(self, hist, action, context=None):
        return hist[:, -1], 1000 * action[:, 0], torch.full((hist.shape[0],), 100.0)


def test_imagined_steps_never_leave_the_kl_budget():
    replay, learner = fixture()
    b = replay.sample(64, 4, 4)
    learner.models = nn.ModuleList([PushModel(), PushModel()])
    learner.value = lambda s: torch.zeros(s.shape[:-1])
    learner.value_modules = ()
    learner.lower[:] = -100
    learner.upper[:] = 100
    learner.cfg.bptt_steps = 5
    learner.cfg.bptt_max_kl = 1.0e-6
    for group in learner.opt_pi.param_groups:
        group["lr"] = 1.0e-3
    before = [p.detach().clone() for p in learner.actor.parameters()]
    result = learner.policy_update(lambda: b)
    # The over-budget step is undone and the next step is smaller.
    assert result["bptt_steps_taken"] == 0 and result["bptt_step_rejected"] == 1
    assert all(torch.equal(a, p) for a, p in zip(before, learner.actor.parameters()))
    assert learner.opt_pi.param_groups[0]["lr"] < 1.0e-3
    learner.cfg.bptt_max_kl = 1.0e9
    result = learner.policy_update(lambda: b)
    assert result["bptt_steps_taken"] == 5 and result["bptt_step_rejected"] == 0
    assert result["bptt_kl"] > 0


def test_guard_kl_uses_policy_std_and_excludes_passive_heads():
    mean = torch.tensor([[0.2, 100.0]])
    kl = gaussian_kl(mean, torch.zeros_like(mean), torch.tensor([0.1, 0.1]), torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(kl, torch.tensor([2.0]))
    kl = gaussian_kl(mean, torch.zeros_like(mean), torch.tensor([0.2, 0.1]), torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(kl, torch.tensor([0.5]))
