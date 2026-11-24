"""World-model and policy learners (paper sections 3.2 and 3.3, Algorithm A.1).

* ``WorldModelLearner``: k-step autoregressive fit of the dynamics on logged windows.
* ``PolicyLearner``: backpropagation through ``H`` imagined steps from replay
  states with a soft survival weight and a critic bootstrap; the critic is fit
  to GAE targets of the imagined rollout only (no target network).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .dynamics import state_loss


@dataclass
class LearnCfg:
    lr: float = 2e-4
    # world model
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    # imagination and policy
    horizon: int = 32
    batch: int = 4096
    gamma: float = 0.99
    lam: float = 0.95
    grad_norm: float = 1.0
    grad_value: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    body_reward_alpha: float = 1.0  # weight of the other branch's local reward in a dual actor update


def cosine_lr(base, it, decay_iters=250_000, floor=0.03):
    """Cosine decay to ``floor`` x base over ``decay_iters`` iterations, then constant."""
    if it >= decay_iters:
        return base * floor
    return base * (floor + (1 - floor) * 0.5 * (1 + torch.cos(torch.tensor(torch.pi * it / decay_iters)).item()))


def _adam(params, lr, betas=(0.9, 0.999)):
    """Adam whose learning rate is a device tensor, so a captured update follows the schedule."""
    params = list(params)
    if params[0].is_cuda:
        return torch.optim.Adam(params, lr=torch.tensor(lr, device=params[0].device), betas=betas,
                                fused=True, capturable=True)
    return torch.optim.Adam(params, lr=lr, betas=betas)


def set_lr(opt, lr):
    for group in opt.param_groups:
        if torch.is_tensor(group["lr"]):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}/"))
        else:
            out[prefix + k] = v
    return out


def _nest(flat):
    out = {}
    for k, v in flat.items():
        node, parts = out, k.split("/")
        for name in parts[:-1]:
            node = node.setdefault(name, {})
        node[parts[-1]] = v
    return out


class CapturedUpdate:
    """Run ``fn(batch)`` eagerly ``warmup`` times, then record it once as a CUDA graph
    and replay it (after github.com/dongc1/supermimic ``learner_utils.CapturedUpdate``,
    Apache-2.0). The update is kernel-launch bound when run eagerly; a replay has no
    Python or launch overhead.

    ``fn`` must be capture-safe: fixed shapes, no host reads, and every tensor it
    reads besides its inputs updated in place (parameters, optimizer state,
    normalizers, learning-rate tensors). Outputs are overwritten by each replay.
    """

    def __init__(self, fn, enabled=True, warmup=3):
        self.fn, self.enabled, self.warmup = fn, enabled, warmup
        self.inputs = None
        self.graph = None
        self.outputs = None

    def __call__(self, batch):
        flat = _flatten(batch)
        first = next(iter(flat.values()))
        if not self.enabled or not first.is_cuda:
            return self.fn(batch)
        if self.inputs is None:
            self.inputs = {k: v.clone() for k, v in flat.items()}
        else:
            for k, v in flat.items():
                self.inputs[k].copy_(v)
        static = _nest(self.inputs)
        if self.warmup > 0:
            self.warmup -= 1
            return self.fn(static)
        if self.graph is None:
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs = self.fn(static)
        self.graph.replay()
        return self.outputs


def _step(opt, loss, params, max_norm, clip_value=0.0):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    return _apply_gradients(opt, params, max_norm, clip_value)


def _apply_gradients(opt, params, max_norm, clip_value=0.0):
    norm = nn.utils.clip_grad_norm_(params, max_norm)
    if clip_value > 0:
        nn.utils.clip_grad_value_(params, clip_value)
    opt.step()
    return norm


class WorldModelLearner:
    def __init__(self, dynamics, cfg: LearnCfg, capture=True):
        self.f, self.cfg = dynamics, cfg
        self.params = list(dynamics.net.parameters())
        self.opt = _adam(self.params, cfg.lr)
        self._update = CapturedUpdate(self._update_from, capture)

    def rollout_loss(self, s, qdes, ctx):
        """``s`` (B, k, 75) logged states, ``qdes`` (B, k, 31) executed targets."""
        x = s[:, 0]
        preds = []
        for j in range(s.shape[1] - 1):
            x = self.f(x, qdes[:, j], ctx)
            preds.append(x)
        return state_loss(torch.stack(preds, 1), s[:, 1:])

    def update(self, batch):
        """``batch``: ``s`` (wm_batches, B, k, 75), ``qdes`` (wm_batches, B, k, 31), ``ctx`` (wm_batches, B, C)."""
        return self._update(batch)

    def _update_from(self, batch):
        self.f.train()
        for p in self.params:
            p.requires_grad_(True)
        losses, groups = [], {}
        for i in range(batch["s"].shape[0]):
            loss, g = self.rollout_loss(batch["s"][i], batch["qdes"][i], batch["ctx"][i])
            _step(self.opt, loss, self.params, self.cfg.grad_norm)
            losses.append(loss.detach())
            for k, v in g.items():
                groups.setdefault(k, []).append(v)
        out = {"wm/loss": torch.stack(losses).mean()}
        out.update({f"wm/mse_{k}": torch.stack(v).mean() for k, v in groups.items()})
        return out

    @torch.no_grad()
    def validate(self, b):
        """Open-loop k-step error at the last step of held-out windows, in physical units."""
        s, qdes, ctx = b["s"], b["qdes"], b["ctx"]
        x = s[:, 0]
        for j in range(s.shape[1] - 1):
            x = self.f(x, qdes[:, j], ctx)
        e = x - s[:, -1]
        return {
            "wm/k_step_root_pos_m": e[:, 0:3].norm(dim=-1).mean(),
            "wm/k_step_joint_rad": e[:, 13:44].abs().mean(),
            "wm/k_step_joint_vel": e[:, 44:75].abs().mean(),
        }


def gae_returns(values, last_value, rewards, dones, gamma, lam):
    """GAE returns ``A_t + V_t`` of a time-major rollout; a soft done ``p_t``
    discounts everything after step ``t`` by ``1 - p_t`` (paper Eq. 3)."""
    returns = torch.empty_like(values)
    advantage = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(values.shape[0])):
        alive = 1.0 - dones[t]
        delta = rewards[t] + alive * gamma * next_value - values[t]
        advantage = delta + alive * gamma * lam * advantage
        returns[t] = advantage + values[t]
        next_value = values[t]
    return returns


class PolicyLearner:
    """``task`` supplies the analytic heads: frame(), qdes(), reward(), term_prob(), critic_obs()."""

    def __init__(self, actor, critic, dynamics, task, cfg: LearnCfg, capture=True):
        self.actor, self.critic, self.f, self.task, self.cfg = actor, critic, dynamics, task, cfg
        self.dual = getattr(actor, "architecture", "single") == "dual_body_mlp_v1"
        if not 0 <= cfg.body_reward_alpha <= 1 or (not self.dual and cfg.body_reward_alpha != 1):
            raise ValueError("body_reward_alpha must be in [0, 1] and requires a dual actor for values below 1")
        if self.dual:
            self.upper_params = list(actor.net.upper.parameters())
            self.lower_params = list(actor.net.lower.parameters())
        self.actor_params = list(actor.net.parameters())
        self.critic_params = list(critic.net.parameters())
        self.opt_pi = _adam(self.actor_params, cfg.lr, betas=(0.9, 0.95))
        self.opt_v = _adam(self.critic_params, cfg.lr)
        self.v_max = task.max_step_reward / (1 - cfg.gamma)
        self._update = CapturedUpdate(self._update_from, capture)

    def set_lr(self, lr, wm=None):
        for opt in (self.opt_pi, self.opt_v) + ((wm.opt,) if wm is not None else ()):
            set_lr(opt, lr)

    def imagine(self, b):
        """Full-team rollout, including the original full-team rewards for the critic."""
        rollout, _ = self._imagine(b, by_body=False)
        return rollout

    def _imagine(self, b, by_body):
        """Roll the actor through the model along the window's task tape.

        ``b`` holds ``horizon + 1`` logged steps: step 0 seeds the state, the
        tape (reference, ball command) of step ``t + 1`` scores the state that
        the action of step ``t`` produces.
        """
        cfg, task = self.cfg, self.task
        H = cfg.horizon
        ctx, delay = b["ctx"], b["delay"][:, 0] > 0.5
        s = b["s"][:, 0]
        a_prev, q_prev = b["a_prev"][:, 0], b["qcmd_prev"][:, 0]
        history = b["history"]
        # The frame before the window carries the action before ``a_prev`` in its last-action slot.
        a_prev2 = history[:, -1, 65:96] if history.shape[1] else a_prev
        ret = torch.zeros(s.shape[0], device=s.device)
        upper_local = lower_local = ret  # all three accumulators update out of place
        reward_options = {}
        if by_body:
            reward_options["body_joints"] = (self.actor.net.upper_joints, self.actor.net.lower_joints)
        survival = torch.ones_like(ret)
        feats, rewards, dones = [], [], []
        # Reference kinematics of the whole window in one batched pass.
        ref_bodies = task.reference_bodies(b["tape"]["ref"])
        # Entry handoff: inside it the model is driven by the blended command (task.handoff).
        handed = task.cfg.entry_handoff_s > 0 and task.cfg.entry_handoff_probability > 0
        gains = task.gains(ctx) if handed else None
        for t in range(H):
            tape = {k: v[:, t] for k, v in b["tape"].items()}
            nxt = {k: v[:, t + 1] for k, v in b["tape"].items()}
            feats.append(task.critic_obs(s, a_prev, tape, ctx).detach())
            frame = task.frame(s, a_prev, tape, noise=True)
            obs = torch.cat((history, frame[:, None]), 1)
            a = self.actor(obs)
            q_cmd = task.qdes(a, tape["q0"])
            q_exec = torch.where(delay[:, None], q_prev, q_cmd)
            if handed:
                q_exec = task.handoff(s, q_exec, tape, *gains)[0]
            s_prev, s = s, self.f(s, q_exec, ctx)
            r = task.reward(s, a, a_prev, nxt, ref_bodies=tuple(x[:, t + 1] for x in ref_bodies), a_prev2=a_prev2,
                            s_prev=s_prev, q_exec=q_exec, **reward_options)
            d = task.term_prob(s, nxt)
            discount = cfg.gamma ** t * survival
            if by_body:
                r, upper, lower = r
                upper_local = upper_local + discount * upper
                lower_local = lower_local + discount * lower
            ret = ret + discount * r
            survival = survival * (1 - d)
            rewards.append(r.detach())
            dones.append(d.detach())
            history = torch.cat((history[:, 1:], frame[:, None].detach()), 1) if history.shape[1] else history
            a_prev2, a_prev, q_prev = a_prev, a, q_cmd
        last = {k: v[:, H] for k, v in b["tape"].items()}
        last_feat = task.critic_obs(s, a_prev, last, ctx)
        value = self.critic(last_feat).clamp(max=self.v_max)
        ret = ret + cfg.gamma ** H * survival * value
        rollout = ret, torch.stack(feats), torch.stack(rewards), torch.stack(dones), last_feat.detach(), survival.detach()
        return rollout, (upper_local, lower_local)

    def update(self, b):
        return self._update(b)

    def _update_from(self, b):
        cfg = self.cfg
        self.actor.train()
        for module in (self.f, self.critic):
            for p in module.parameters():
                p.requires_grad_(False)
        rollout, local_returns = self._imagine(b, by_body=self.dual)
        ret, feats, rewards, dones, last_feat, survival = rollout
        actor_metrics = self._update_actor(ret, *local_returns)
        for p in self.critic.parameters():
            p.requires_grad_(True)
        value_loss = self.fit_critic(feats, rewards, dones, last_feat)
        return {
            "policy/imagined_return": ret.detach().mean(),
            "policy/imagined_reward_per_s": rewards.mean() / self.task.dt,
            "policy/imagined_survival": survival.mean(),
            **actor_metrics,
            "policy/value_loss": value_loss,
        }

    def _update_actor(self, team, upper_local, lower_local):
        cfg = self.cfg
        self.opt_pi.zero_grad(set_to_none=True)
        alpha = cfg.body_reward_alpha
        upper_return = lower_return = team
        if alpha == 1:
            (-team.mean()).backward()
        else:
            # team includes the common bootstrap and survival weighting. Discount
            # only the OTHER body's local contributions inside the imagined horizon.
            upper_return = team - (1 - alpha) * lower_local
            lower_return = team - (1 - alpha) * upper_local
            (-upper_return.mean()).backward(inputs=self.upper_params, retain_graph=True)
            (-lower_return.mean()).backward(inputs=self.lower_params)
        metrics = {}
        if self.dual:
            for name, params, objective in (("upper", self.upper_params, upper_return),
                                            ("lower", self.lower_params, lower_return)):
                norms = [p.grad.detach().norm() for p in params]
                metrics[f"policy/{name}_grad_norm"] = torch.stack(norms).norm()
                metrics[f"policy/{name}_return"] = objective.detach().mean()
        # Both branches participate in one global clipping and one optimizer step.
        metrics["policy/grad_norm"] = _apply_gradients(self.opt_pi, self.actor_params, cfg.grad_norm, cfg.grad_value)
        return metrics

    def fit_critic(self, feats, rewards, dones, last_feat):
        cfg = self.cfg
        self.critic.norm.update(feats.flatten(0, 1))
        with torch.no_grad():
            values = self.critic(feats.flatten(0, 1)).view(feats.shape[:2])
            last_value = self.critic(last_feat)
            targets = gae_returns(values, last_value, rewards, dones, cfg.gamma, cfg.lam).clamp(max=self.v_max)
        x, y = feats.flatten(0, 1), targets.flatten()
        n = x.shape[0] // cfg.critic_minibatches * cfg.critic_minibatches
        losses = []
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(n, device=x.device).view(cfg.critic_minibatches, -1)
            for idx in perm:
                loss = (self.critic(x[idx]) - y[idx]).square().mean()
                _step(self.opt_v, loss, self.critic_params, cfg.grad_norm)
                losses.append(loss.detach())
        return torch.stack(losses).mean()
