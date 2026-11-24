"""SuperMimic-style replay, multi-step model fitting and imagined policy gradients.

Adaptation of dongc1/supermimic_vibed a1508d42 (MIT). Known action feedback and
packet context are exact; residual rewards and failure probabilities are learned.
Termination costs come from the original task. The bootstrap value is the PPO
critic trained on real returns; the real-data actor update is the task's PPO.
Learned predictions remain approximations, never evaluation scores.
Only the 110-D actor slice reaches the deployed policy. History is WM-only.
"""

from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


def mlp(n, hidden, out):
    layers = []
    for h in hidden:
        layers += [nn.Linear(n, h), nn.SiLU(), nn.LayerNorm(h)]
        n = h
    return nn.Sequential(*layers, nn.Linear(n, out))


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=0.01):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.eps = eps

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel())
        if not x.numel():
            return
        n = x.shape[0]
        m = x.mean(0)
        v = x.var(0, unbiased=False)
        total = self.count + n
        delta = m - self.mean
        self.var.copy_(
            (self.var * self.count + v * n + delta.square() * self.count * n / total)
            / total
        )
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def std(self):
        return (self.var + self.eps**2).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std()


class Replay:
    """Time-major GPU replay; held-out envs never enter model fitting.

    ``s1`` is not stored twice. The next row's ``s`` is the observation the
    environment returned, including Isaac's auto-reset observation, so it is
    the exact successor; only the newest row reads ``latest``.
    """

    def __init__(
        self, T, N, dim, actions, device, validation_fraction=0.125, failure_dim=0
    ):
        self.T, self.N, self.ptr, self.size = T, N, 0, 0
        self.device = device
        self.val_n = max(1, int(N * validation_fraction))
        if self.val_n >= N:
            raise ValueError("Need at least two environments")
        shapes = dict(
            s=(dim,),
            a=(actions,),
            r=(),
            cut=(),
            terminal=(),
            done=(),
            failure=(failure_dim,),
            failure_margin=(failure_dim,),
        )
        self.data = {
            k: torch.zeros(
                (T, N) + s,
                device=device,
                dtype=torch.bool
                if k in ("cut", "terminal", "done", "failure")
                else torch.float32,
            )
            for k, s in shapes.items()
        }
        self.latest = torch.zeros(N, dim, device=device)

    @torch.no_grad()
    def add(self, **transition):
        transition.setdefault("done", transition["terminal"])
        if self.data["failure"].shape[-1] == 0:
            transition.setdefault("failure", self.data["failure"][self.ptr])
            transition.setdefault(
                "failure_margin", self.data["failure_margin"][self.ptr]
            )
        for k, v in self.data.items():
            v[self.ptr].copy_(transition[k])
        self.latest.copy_(transition["s1"])
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def sample(self, B, window, history, validation=False, recent_steps=None):
        if self.size < window:
            raise ValueError("Replay has insufficient history")
        available = (
            self.size if recent_steps is None else min(self.size, int(recent_steps))
        )
        if available < window:
            raise ValueError(
                "Recent replay must accommodate the complete sample window"
            )
        age = torch.randint(
            self.size - available, self.size - window + 1, (B,), device=self.device
        )
        lo, hi = (
            (self.N - self.val_n, self.N) if validation else (0, self.N - self.val_n)
        )
        e = torch.randint(lo, hi, (B,), device=self.device)
        rows = (
            self.ptr
            - self.size
            + age[:, None]
            + torch.arange(window, device=self.device)
        ) % self.T
        out = {k: v[rows, e[:, None]] for k, v in self.data.items()}
        newest = (self.ptr - 1) % self.T
        out["s1"] = torch.where(
            (rows == newest)[..., None],
            self.latest[e][:, None],
            self.data["s"][(rows + 1) % self.T, e[:, None]],
        )
        back = torch.arange(history - 1, -1, -1, device=self.device)
        ages = (age[:, None] - back).clamp_min(0)
        hr = (self.ptr - self.size + ages) % self.T
        # Repeat the first state after the most recent reset/packet boundary.
        cuts = self.data["cut"][hr, e[:, None]]
        index = torch.arange(history, device=self.device).expand(B, -1)
        boundary = (
            torch.where(cuts[:, :-1], index[:, :-1] + 1, 0).max(-1).values
            if history > 1
            else index[:, 0]
        )
        index = torch.maximum(index, boundary[:, None])
        hr = hr.gather(1, index)
        out["history"] = self.data["s"][hr, e[:, None]]
        out["mask"] = torch.cumprod(
            torch.cat(
                (torch.ones(B, 1, device=self.device), (~out["cut"][:, :-1]).float()), 1
            ),
            1,
        )
        out["env_ids"] = e
        return out


class Model(nn.Module):
    def __init__(
        self,
        dim,
        action_dim,
        history,
        hidden,
        contract=None,
        termination_costs=(),
        failure_temperature=None,
    ):
        super().__init__()
        self.dim = dim
        self.contract = deepcopy(contract)
        self.failure_temperature = failure_temperature
        self.register_buffer(
            "termination_costs", torch.tensor(termination_costs, dtype=torch.float32)
        )
        self.net = mlp(
            dim * history + action_dim, hidden, dim + 2 + len(termination_costs)
        )
        self.state_norm = RunningNorm(dim)
        self.delta_norm = RunningNorm(dim)
        self.reward_norm = RunningNorm(1, eps=0.001)
        if failure_temperature is not None:
            if not termination_costs or failure_temperature <= 0:
                raise ValueError(
                    "Physical margin model needs costs and a positive temperature"
                )
            self.failure_norm = RunningNorm(len(termination_costs))
            self.failure_delta_norm = RunningNorm(len(termination_costs))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def observe(self, b):
        mask = b["mask"].bool()
        self.state_norm.update(b["s"][mask])
        valid = mask & ~b["cut"]
        self.delta_norm.update((b["s1"] - b["s"])[valid])
        self.reward_norm.update(self.residual_target(b)[mask].unsqueeze(-1))
        if self.failure_temperature is not None:
            k = self.termination_costs.numel()
            self.failure_norm.update(b["failure_margin"][mask])
            self.failure_delta_norm.update(
                (b["failure_margin"] - b["s"][..., -k:])[mask]
            )

    def residual_target(self, b):
        if self.termination_costs.numel():
            return b["r"] - b["failure"].float() @ self.termination_costs
        return b["r"]

    def predict(self, hist, a, context=None):
        y = self.net(torch.cat((self.state_norm(hist).flatten(-2), a), -1))
        ns = (
            hist[:, -1]
            + y[:, : self.dim] * self.delta_norm.std()
            + self.delta_norm.mean
        )
        if self.contract is not None:
            ns = self.contract(ns, a, hist[:, -1] if context is None else context)
        residual = y[:, self.dim] * self.reward_norm.std()[0] + self.reward_norm.mean[0]
        failure_logits = y[:, self.dim + 2 :]
        continuation_logit = y[:, self.dim + 1]
        if self.failure_temperature is not None:
            k = self.termination_costs.numel()
            margin = (
                hist[:, -1, -k:]
                + failure_logits * self.failure_delta_norm.std()
                + self.failure_delta_norm.mean
            )
            failure_logits = margin / self.failure_temperature
            alive = continuation_logit.sigmoid() * (-failure_logits).sigmoid().prod(-1)
            continuation_logit = torch.logit(alive.clamp(1e-6, 1 - 1e-6))
        reward = residual + failure_logits.sigmoid() @ self.termination_costs
        return ns, reward, continuation_logit, residual, failure_logits

    def forward(self, hist, a, context=None):
        return self.predict(hist, a, context)[:3]

    def loss(self, b, details=False):
        hist = b["history"]
        losses = []
        errors = []
        diagnostics = []
        residual_targets = self.residual_target(b)
        for j in range(b["a"].shape[1]):
            context = (
                self.contract.context(b, j, hist[:, -1])
                if self.contract is not None
                else None
            )
            s, r, logit, residual, failure_logits = self.predict(
                hist, b["a"][:, j], context
            )
            mask = b["mask"][:, j]
            valid = mask * (~b["cut"][:, j])
            normalized_error = ((s - b["s1"][:, j]) / self.state_norm.std()).square()
            err = normalized_error.mean(-1)
            re = (
                (residual - residual_targets[:, j]) / self.reward_norm.std()[0]
            ).square()
            ce = F.binary_cross_entropy_with_logits(
                logit, (~b["terminal"][:, j]).float(), reduction="none"
            )
            if self.failure_temperature is not None:
                # Dense geometric targets include failing transitions before
                # auto-reset. Binary rare-event fitting alone hid these errors.
                ce = ce + (
                    (
                        failure_logits * self.failure_temperature
                        - b["failure_margin"][:, j]
                    )
                    / self.failure_norm.std()
                ).square().mean(-1)
            elif self.termination_costs.numel():
                ce = ce + F.binary_cross_entropy_with_logits(
                    failure_logits, b["failure"][:, j].float(), reduction="none"
                ).sum(-1)
            losses.append(
                (err * valid).sum() / valid.sum().clamp_min(1)
                + (mask * (re + ce)).sum() / mask.sum().clamp_min(1)
            )
            errors.append((err * valid).sum() / valid.sum().clamp_min(1))
            if details:
                diagnostics.append(
                    torch.stack(
                        (
                            (re * mask).sum() / mask.sum().clamp_min(1),
                            (ce * mask).sum() / mask.sum().clamp_min(1),
                            (normalized_error[:, :110].mean(-1) * valid).sum()
                            / valid.sum().clamp_min(1),
                            (normalized_error[:, 110:].mean(-1) * valid).sum()
                            / valid.sum().clamp_min(1),
                        )
                    )
                )
            hist = torch.cat((hist[:, 1:], s[:, None]), 1)
        result = torch.stack(losses).mean(), torch.stack(errors).mean()
        return (*result, torch.stack(diagnostics).mean(0)) if details else result


@contextmanager
def frozen(*modules):
    params = [p for m in modules for p in m.parameters()]
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, f in zip(params, flags):
            p.requires_grad_(f)




class Learner:
    """World-model fitting plus guarded first-order actor steps.

    ``value`` maps a 443-D state to the PPO critic's estimate; its modules are
    frozen during imagination. ``action_std`` returns the policy's physical std
    for the KL guard. PPO remains the real-data actor update.
    """

    def __init__(
        self,
        actor,
        dim,
        cfg,
        lower,
        upper,
        active,
        noise_penalty,
        contract=None,
        termination_costs=(),
        value=None,
        value_modules=(),
        action_std=None,
    ):
        self.actor, self.cfg = actor, cfg
        self.lower, self.upper, self.active = lower, upper, active
        self.noise_penalty = noise_penalty
        self.value, self.value_modules = value, tuple(value_modules)
        self.action_std = action_std or (lambda: torch.ones_like(active))
        self.models = nn.ModuleList(
            [
                Model(
                    dim,
                    active.numel(),
                    cfg.history,
                    cfg.hidden,
                    contract,
                    termination_costs,
                    getattr(cfg, "failure_margin_temperature", None),
                )
                for _ in range(2)
            ]
        ).to(lower.device)
        self.opt_wm = torch.optim.Adam(self.models.parameters(), lr=cfg.learning_rate)
        self.opt_pi = torch.optim.Adam(
            actor.parameters(),
            lr=float(getattr(cfg, "bptt_learning_rate", cfg.learning_rate)),
            betas=(0.9, 0.95),
        )

    def project(self, a, ids=None):
        lower = self.lower if ids is None else self.lower[ids]
        upper = self.upper if ids is None else self.upper[ids]
        return a.clamp(lower, upper) * self.active

    def model_update(self, b):
        for m in self.models:
            m.observe(b)
        loss = torch.stack([m.loss(b)[0] for m in self.models]).mean()
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.models.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("world-model gradient is nonfinite")
        self.opt_wm.step()
        self.last_model_metrics = dict(
            wm_grad_norm=float(norm.detach()),
            wm_valid_fraction=float((b["mask"] * (~b["cut"])).mean()),
        )
        return float(loss.detach())

    @torch.no_grad()
    def validate(self, b):
        values = [m.loss(b, details=True) for m in self.models]
        details = torch.stack([v[2] for v in values]).mean(0).cpu().tolist()
        metrics = dict(
            wm_validation_valid_fraction=float((b["mask"] * (~b["cut"])).mean()),
            wm_validation_loss=float(torch.stack([v[0] for v in values]).mean()),
            wm_validation_state_mse=float(torch.stack([v[1] for v in values]).mean()),
            **dict(
                zip(
                    (
                        "wm_validation_reward_mse",
                        "wm_validation_continuation_bce",
                        "wm_validation_actor_obs_mse",
                        "wm_validation_privileged_obs_mse",
                    ),
                    details,
                )
            ),
        )
        # Report failure tails separately: aggregate MSE can hide underestimated
        # failure costs among many routine transitions.
        predictions = []
        contract_errors = []
        margin_errors = []
        for m in self.models:
            context = (
                m.contract.context(b, 0, b["history"][:, -1])
                if m.contract is not None
                else None
            )
            state, reward, _, _, failure_logits = m.predict(
                b["history"], b["a"][:, 0], context
            )
            predictions.append(reward)
            if m.failure_temperature is not None:
                margin_errors.append(
                    (failure_logits * m.failure_temperature - b["failure_margin"][:, 0])
                    .square()
                    .mean()
                )
            if m.contract is not None:
                c = m.contract
                contract_errors.append(
                    torch.stack(
                        (
                            (
                                state[:, c.action_indices]
                                - b["a"][:, 0, c.action_sources]
                            )
                            .abs()
                            .max(),
                            (
                                state[:, c.context_indices]
                                - context[:, c.context_indices]
                            )
                            .abs()
                            .max(),
                        )
                    )
                )
        if contract_errors:
            action_error, context_error = (
                torch.stack(contract_errors).amax(0).cpu().tolist()
            )
            metrics["WorldModel/action_feedback_max_abs_error"] = action_error
            metrics["WorldModel/command_context_max_abs_error"] = context_error
        if margin_errors:
            metrics["WorldModel/failure_margin_rmse"] = float(
                torch.stack(margin_errors).mean().sqrt()
            )
        predicted = torch.stack(predictions).mean(0)
        for name, selected in (
            ("terminal", b["terminal"][:, 0]),
            ("nonterminal", ~b["terminal"][:, 0]),
        ):
            count = selected.sum()
            metrics[f"WorldModel/{name}_validation_samples"] = int(count)
            if count:
                metrics[f"WorldModel/{name}_reward_bias"] = float(
                    (predicted - b["r"][:, 0])[selected].mean()
                )
                metrics[f"WorldModel/{name}_reward_rmse"] = float(
                    (predicted - b["r"][:, 0])[selected].square().mean().sqrt()
                )
        return metrics

    def policy_update(self, sample):
        """Up to ``bptt_steps`` imagined actor steps inside a KL budget.

        ``sample()`` returns a replay window. The KL is measured against the
        policy that PPO just produced, on the first window's start states, so
        model error cannot move the next real rollout's behavior far. A step
        that leaves the budget is undone; the step size adapts like PPO's.
        """
        rows = []
        obs = reference = None
        std = self.action_std().detach()
        budget = float(self.cfg.bptt_max_kl)
        parameters = list(self.actor.parameters())
        accepted, kl = 0, 0.0
        for _ in range(int(self.cfg.bptt_steps)):
            b = sample()
            if reference is None:
                obs = b["history"][:, -1, :110].detach()
                with torch.no_grad():
                    reference = self.actor(obs)
            snapshot = [p.detach().clone() for p in parameters]
            row = self._imagined_step(b)
            with torch.no_grad():
                step_kl = float(
                    gaussian_kl(self.actor(obs), reference, std, self.active).mean()
                )
            rows.append(row)
            if step_kl > budget:
                with torch.no_grad():
                    for p, saved in zip(parameters, snapshot):
                        p.copy_(saved)
                self._scale_step(1 / 1.5)
                break
            accepted, kl = accepted + 1, step_kl
        else:
            if kl < budget / 2:
                self._scale_step(1.5)
        result = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        result["bptt_kl"] = kl
        result["bptt_steps_taken"] = accepted
        result["bptt_step_rejected"] = int(accepted < len(rows))
        return result

    def _scale_step(self, factor):
        for group in self.opt_pi.param_groups:
            group["lr"] = min(max(group["lr"] * factor, 1.0e-6), 1.0e-3)

    def _imagined_step(self, b):
        # Recorded boundaries limit imagination to the same announced packet.
        # No new ball or reset trajectory is supplied as an actor observation.
        hist = b["history"].detach()
        disagreements, projected = [], []
        survival = torch.ones(hist.shape[0], device=hist.device)
        objective = torch.zeros_like(survival)
        with frozen(self.models, *self.value_modules):
            for t in range(self.cfg.horizon):
                raw = self.actor(hist[:, -1, :110])
                # The environment receives unclamped actions with passive heads
                # zeroed, exactly as PPO samples them; replay stores the same.
                action = raw * self.active
                predictions = [
                    m(
                        hist,
                        action,
                        m.contract.context(b, t, hist[:, -1])
                        if m.contract is not None
                        else None,
                    )
                    for m in self.models
                ]
                # Alternate differentiable dynamics, penalize their disagreement.
                s, r, logit = predictions[t % len(predictions)]
                disagreement = (
                    (
                        (predictions[0][0] - predictions[1][0])
                        / self.models[0].state_norm.std()
                    )
                    .square()
                    .mean(-1)
                )
                alive = logit.sigmoid()
                outside = (raw - self.project(raw, b["env_ids"])) * self.active
                disagreements.append(disagreement.detach().mean())
                projected.append(
                    ((outside.abs() > 1.0e-6).sum()).detach()
                    / (raw.shape[0] * self.active.sum())
                )
                mask = b["mask"][:, t]
                reward = r - 0.1 * disagreement - 0.1 * outside.square().mean(-1)
                objective = objective + self.cfg.gamma**t * survival * mask * reward
                # Stop gradients through rollout weighting. Failure costs have
                # their own differentiable, task-owned reward term; the actor
                # cannot improve its objective by suppressing future weights.
                survival = survival * alive.detach() * mask
                hist = torch.cat((hist[:, 1:], s[:, None]), 1)
                # Bootstrap at each sample's actual rollout end, including
                # packet boundaries, not only at the fixed maximum horizon.
                last = (
                    torch.ones_like(mask)
                    if t + 1 == self.cfg.horizon
                    else 1 - b["mask"][:, t + 1]
                )
                objective = objective + self.cfg.gamma ** (
                    t + 1
                ) * survival * last * self.value(s)
            noise = self.noise_penalty(b["s"][:, 0, :110])
            weight = float(getattr(self.cfg, "model_policy_weight", 1.0))
            loss = -weight * objective.mean() + noise
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
            if not torch.isfinite(norm):
                raise FloatingPointError("actor gradient is nonfinite")
            self.opt_pi.step()
        return dict(
            actor_loss=float(loss.detach()),
            sensor_noise_weighted=float(noise.detach()),
            imagined_return=float(objective.detach().mean()),
            actor_grad_norm=float(norm.detach()),
            imagined_disagreement=float(torch.stack(disagreements).mean()),
            imagined_projection_fraction=float(torch.stack(projected).mean()),
            imagined_survival=float(survival.detach().mean()),
        )


def gaussian_kl(mean, reference, std, active):
    """Equal-variance Gaussian KL over executable channels."""
    return (0.5 * ((mean - reference) / std).square() * active).sum(-1)
