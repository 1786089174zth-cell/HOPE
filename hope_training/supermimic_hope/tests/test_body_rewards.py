"""Check branch gradient ownership, then optionally audit a fixed MuJoCo-Warp minibatch."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_hope import a3
from supermimic_hope.core import learners
from supermimic_hope.core.nets import Actor, Critic


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_coupled_branch_gradients_before_clipping(alpha):
    cfg = learners.LearnCfg(lr=0.0, body_reward_alpha=alpha, grad_norm=1e8, grad_value=0.0)
    actor = Actor(2, (8,), architecture="dual_body_mlp_v1")
    pi = learners.PolicyLearner(actor, Critic(2, (8,)), None, SimpleNamespace(max_step_reward=2), cfg)
    # Every parameter participates. The lower action depends on Upper with derivative 3.
    u = sum(p.sum() for p in pi.upper_params)
    lower_action = 3 * u + sum(p.sum() for p in pi.lower_params)
    shared = 2 * u + lower_action
    upper_local = 5 * u + 2 * lower_action
    lower_local = u + 4 * lower_action
    team = shared + upper_local + lower_local
    pi._update_actor(team, upper_local, lower_local)
    for p in pi.upper_params:
        torch.testing.assert_close(p.grad, torch.full_like(p, -(16 + 13 * alpha)))
    for p in pi.lower_params:
        torch.testing.assert_close(p.grad, torch.full_like(p, -(5 + 2 * alpha)))


def reference_reward_groups(task, state, action, previous, tape, previous2, terms):
    """Independent assignment from the detailed legacy terms, using actuator names."""
    shared_keys = ("root_pos", "root_ori", "body_pos", "body_ori", "body_lin_vel", "body_ang_vel",
                   "root_height", "trunk_ori", "racket_position", "racket_blade", "racket_velocity",
                   "racket_drive", "direct_arm", "direct_action", "direct_drive", "return_velocity",
                   "landing", "table_keepout", "torque_limit", "range_keepout")
    upper_keys = ("racket_normal", "left_arm", "direct_face")
    lower_keys = ("feet_pos", "feet_height", "feet_ori", "root_xy", "feet_station", "root_heading",
                  "feet_heading", "stance_command", "direct_waist_roll", "waist_rest", "feet_slip")
    joint_keys = ("action_rate", "hold_action_rate", "action_accel", "joint_limit")
    assert set(terms) == set(shared_keys + upper_keys + lower_keys + joint_keys)
    lower_ids = [i for i, name in enumerate(a3.JOINTS) if any(k in name for k in ("hip", "knee", "ankle", "waist"))]
    upper_ids = [i for i in range(31) if i not in lower_ids]
    q = state[:, 13:44]
    smooth = tape.get("smooth", 1.0)
    def joint_score(ids):
        rate = (smooth * (action - previous).square())[:, ids].sum(-1)
        accel = (smooth * (action - 2 * previous + previous2).square())[:, ids].sum(-1)
        limit = ((task.soft_lo[ids] - q[:, ids]).relu() + (q[:, ids] - task.soft_hi[ids]).relu()).sum(-1)
        return (task.cfg.action_rate * rate + task.cfg.hold_action_rate * tape["in_hold"] * rate
                + task.cfg.action_accel * accel + task.cfg.joint_limit * limit)
    shared = sum(terms[k] for k in shared_keys)
    upper = sum(terms[k] for k in upper_keys) + joint_score(upper_ids)
    lower = sum(terms[k] for k in lower_keys) + joint_score(lower_ids)
    return torch.stack([shared, upper, lower]) * task.dt


@pytest.mark.skipif(os.environ.get("SUPERMIMIC_GPU_AUDIT") != "1", reason="opt-in MuJoCo-Warp replay audit")
def test_fixed_gpu_batch_actor_gradients_and_team_critic(monkeypatch):
    from supermimic_hope.core.trainer import TrainCfg, Trainer

    cfg = TrainCfg(num_envs=8, seed=17, buffer_steps=48, capture=False, actor_architecture="dual_body_mlp_v1")
    cfg.actor_hidden, cfg.dynamics_hidden, cfg.critic_hidden = (64, 32), (64, 64), (32,)
    cfg.learn = learners.LearnCfg(lr=0.0, batch=8, horizon=4, grad_norm=1e8, grad_value=0.0)
    cfg.task.termination_softness = 0.6  # exercise fractional survival throughout the reference calculation
    tr = Trainer(cfg)
    pi, task = tr.pi, tr.task
    tr.collect(24)
    batch = tr.policy_batch()
    batch["tape"]["in_hold"][::2, 1:] = 1
    batch["tape"]["in_hold"][1::2, 1:] = 0
    batch["tape"]["tts"][:, 1:] = 0.0
    batch["tape"]["direct"][:, 1:] = 1.0  # include the optional mixed waist/arm terms
    rng = torch.cuda.get_rng_state()
    groups, survival_probs, bootstrap = [], [], []
    reward, term_prob = task.reward, task.term_prob
    def record_reward(s, a, prev, tape, **kw):
        total, info = reward(s, a, prev, tape, detail=True, **kw)
        parts = reference_reward_groups(task, s, a, prev, tape, kw["a_prev2"], info["terms"])
        torch.testing.assert_close(parts.sum(0), total, atol=1e-6, rtol=1e-5)
        groups.append(parts)
        return total
    def record_termination(s, tape):
        d = term_prob(s, tape)
        survival_probs.append(1 - d)
        return d
    with monkeypatch.context() as patch:
        patch.setattr(task, "reward", record_reward)
        patch.setattr(task, "term_prob", record_termination)
        hook = tr.critic.register_forward_hook(lambda module, args, out: bootstrap.append(out))
        try:
            legacy, _, team_steps, _, _, _ = pi.imagine(batch)
        finally:
            hook.remove()

    # Independent backward recursion from the same whole-body terminal value.
    def reference_return(weights):
        result = bootstrap[0].clamp(max=pi.v_max)
        for parts, alive in reversed(list(zip(groups, survival_probs))):
            result = sum(w * part for w, part in zip(weights, parts)) + cfg.learn.gamma * alive * result
        return result
    torch.testing.assert_close(reference_return((1, 1, 1)), legacy)
    references = {}
    for alpha in (1.0, 0.5):
        for side, weights, params in (("upper", (1, 1, alpha), pi.upper_params),
                                      ("lower", (1, alpha, 1), pi.lower_params)):
            obj = reference_return(weights)
            references[alpha, side] = (obj.detach().mean(), torch.autograd.grad(-obj.mean(), params, retain_graph=True))
    for side in ("upper", "lower"):
        g1 = references[1.0, side][1]
        gh = references[0.5, side][1]
        assert max((a - b).abs().max().item() for a, b in zip(g1, gh)) > 1e-7

    gae = learners.gae_returns
    targets = []
    def check_gae(values, tail, rewards, dones, gamma, lam):
        torch.testing.assert_close(rewards, team_steps)
        # Direct TD(lambda) recursion on full-team rewards, including fractional termination.
        target, expected = tail, []
        next_values = torch.cat((values[1:], tail[None]))
        for r, d, v in reversed(list(zip(team_steps, dones, next_values))):
            target = r + gamma * (1 - d) * ((1 - lam) * v + lam * target)
            expected.append(target)
        got = gae(values, tail, rewards, dones, gamma, lam)
        err = (got - torch.stack(expected[::-1])).abs().max().item()
        torch.testing.assert_close(got, torch.stack(expected[::-1]), atol=1e-6, rtol=1e-5)
        targets.append(got.clone())
        print(f"alpha={cfg.learn.body_reward_alpha} critic_target_max_abs_error={err:.3g}")
        return got
    monkeypatch.setattr(learners, "gae_returns", check_gae)
    norm = {k: v.clone() for k, v in tr.critic.norm.state_dict().items()}
    for alpha in (1.0, 0.5):
        tr.critic.norm.load_state_dict(norm)
        tr.critic.norm._seen = None
        cfg.learn.body_reward_alpha = alpha
        torch.cuda.set_rng_state(rng)
        metrics = pi.update(batch)
        for side, params in (("upper", pi.upper_params), ("lower", pi.lower_params)):
            objective, grads = references[alpha, side]
            torch.testing.assert_close(metrics[f"policy/{side}_return"], objective, atol=1e-6, rtol=1e-5)
            for p, want in zip(params, grads):
                torch.testing.assert_close(p.grad, want, atol=2e-5, rtol=2e-4)
            error = max((p.grad - want).abs().max().item() for p, want in zip(params, grads))
            print(f"alpha={alpha} {side}_gradient_max_abs_error={error:.3g}")
    assert len(targets) == 2
    torch.testing.assert_close(targets[0], targets[1], atol=0, rtol=0)
