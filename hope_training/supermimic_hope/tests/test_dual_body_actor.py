"""CPU regressions for the two branches of the deployed SuperMimic actor."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_hope import a3
from supermimic_hope.core.nets import Actor


def test_each_motor_reads_its_named_branch_and_lower_sees_the_masked_upper_output():
    frames = 3
    actor = Actor(frames, (24,), architecture="dual_body_mlp_v1")
    net = actor.net
    # Mark every output with a distinct value so a joint-order permutation cannot pass.
    with torch.no_grad():
        for branch, start in [(net.upper, 10), (net.lower, 40)]:
            head = branch[-1]
            head.weight.zero_()
            head.bias.copy_(torch.arange(start, start + head.out_features))
    seen = {}
    def save_input(name):
        def capture(_module, args):
            seen[name] = args[0]
        return capture
    net.lower.register_forward_pre_hook(save_input("lower"))
    net.upper.register_forward_pre_hook(save_input("upper"))
    history = torch.arange(2 * frames * 110).reshape(2, frames, 110).float()
    out = actor(history)
    normalized = actor.norm(history)
    torch.testing.assert_close(seen["upper"], torch.cat((normalized[..., :101], normalized[..., 103:]), -1).flatten(1))
    torch.testing.assert_close(seen["lower"][:, :-16], torch.cat((normalized[..., :103], normalized[..., -1:]), -1).flatten(1))
    # Current A3 articulation order, checked independently against the named contract.
    upper = [11, 16, 12, 13, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    lower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 19, 20]
    assert tuple(upper[:2]) == a3.PASSIVE
    intent = torch.arange(10, 26).float()
    intent[:2] = 0
    torch.testing.assert_close(out[:, upper], intent.expand(2, -1))
    torch.testing.assert_close(out[:, lower], torch.arange(40, 55).float().expand(2, -1))
    torch.testing.assert_close(seen["lower"][:, -16:], out[:, upper])
    torch.testing.assert_close(actor.flat_forward(history.flatten(1)), out)


def test_lower_loss_reaches_both_branches_but_never_the_passive_outputs():
    torch.manual_seed(17)
    actor = Actor(10, (32,), architecture="dual_body_mlp_v1")
    history = torch.randn(4, 10, 110, requires_grad=True)
    out = actor(history)
    upper_gradient = torch.autograd.grad(out[:, a3.JOINTS.index("right_elbow_joint")].sum(),
                                         history, retain_graph=True)[0]
    assert upper_gradient[..., 101:103].count_nonzero() == 0  # station belongs to Lower
    out[:, a3.JOINTS.index("waist_yaw_joint")].sum().backward()
    upper_bias_grad = actor.net.upper[-1].bias.grad
    assert upper_bias_grad[:2].count_nonzero() == 0
    assert upper_bias_grad[2:].abs().sum() > 0
    assert actor.net.lower[-1].weight.grad.abs().sum() > 0
    # Lower can see the racket command only through the differentiable Upper message.
    assert history.grad[..., 103:109].abs().sum() > 0


@pytest.mark.parametrize("architecture", ["single", "dual_body_mlp_v1"])
def test_actor_checkpoint_round_trip_and_legacy_keys(architecture):
    active = torch.ones(31)
    active[list(a3.PASSIVE)] = 0
    actor = Actor(2, (16,), active, architecture=architecture)
    restored = Actor(2, (16,), active) if architecture == "single" else Actor(2, (16,), active, architecture=architecture)
    weights = actor.state_dict()
    if architecture == "single":
        assert "net.0.weight" in weights  # original single-actor checkpoint layout
    assert not any(k.endswith(("upper_joints", "motor_order")) for k in weights)
    restored.load_state_dict(weights, strict=True)
    x = torch.randn(5, 220)
    torch.testing.assert_close(restored.flat_forward(x), actor.flat_forward(x))
