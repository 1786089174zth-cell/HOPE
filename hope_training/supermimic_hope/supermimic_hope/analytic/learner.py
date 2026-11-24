"""SuperMimic learner with analytic HOPE heads (upstream Algorithm A.1).

Replay holds real physical states, controls and the recorded task context.
The world model is fit on k-step autoregressive windows. The actor is trained
by backpropagation through imagined rollouts whose rewards, terminations and
observations are the analytic task heads; the critic is fit on imagined
lambda-returns and bootstraps the horizon.
"""

from __future__ import annotations

from functools import partial
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from . import state as S
from .world_model import PhysicalWorldModel, RunningNorm, mlp, push


class FiniteGrad(torch.autograd.Function):
    """Identity whose backward replaces non-finite gradients with zero.

    Compiled backward graphs decompose a few ops (e.g. vector norms) without
    eager PyTorch's zero-input masking, so 0 * inf can appear where the true
    gradient is zero. Applied only at the compiled-head boundary.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


def finite_grad(*xs):
    return tuple(FiniteGrad.apply(x) if x.requires_grad else x for x in xs)


def clip_rows(g, factor):
    """Bound each row's gradient at ``factor`` x the median nonzero row norm.

    Every actor-parameter gradient passes through some row's action at some
    step, so clipping there bounds each imagined sample's influence. Without
    it, a few rows (steep critic regions at the horizon) carried most of the
    batch gradient and the global norm clip erased everyone else's signal.
    """
    n = g.norm(dim=-1, keepdim=True)
    cap = factor * torch.where(n > 0, n, torch.full_like(n, float("nan"))).nanmedian()
    scale = (cap / n.clamp_min(1e-12)).clamp(max=1.0)
    return torch.where(torch.isfinite(cap), g * scale, g)


class TapeReplay:
    """Time-major GPU replay of per-step records; held-out envs never fit the model."""

    def __init__(self, T, N, device, validation_fraction=0.125, storage=None):
        """``storage`` is where the tape lives (default ``device``); ``"cpu"`` keeps it in
        pinned host memory so the buffer can hold far more steps than the GPU has room for.
        Sampled batches always come back on ``device``."""
        self.T, self.N, self.device = T, N, device
        self.storage = torch.device(storage if storage is not None else device)
        self.ptr = self.size = 0
        self.val_n = max(1, int(N * validation_fraction))
        self.data = None

    @torch.no_grad()
    def add(self, record):
        if self.data is None:
            pin = self.storage.type == "cpu" and torch.cuda.is_available()
            self.data = {
                k: torch.zeros((self.T,) + tuple(v.shape), dtype=v.dtype, device=self.storage, pin_memory=pin)
                for k, v in record.items()
            }
        for k, v in self.data.items():
            v[self.ptr].copy_(record[k])
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def bytes(self):
        return sum(v.numel() * v.element_size() for v in self.data.values()) if self.data else 0

    def strike_starts(self, window, dt, margin=0.12):
        """Start cells (age, env) whose window reaches the strike (|tts| <= margin inside it)."""
        tts = self.data["ctx/tts"]
        rows = (self.ptr - self.size + torch.arange(self.size - window + 1, device=self.storage)) % self.T
        t0 = tts[rows]  # (starts, N): time to strike at the window start
        return (t0 >= -margin) & (t0 <= (window - 1) * dt + margin)

    def sample(self, B, window, validation=False, recent=None, keys=None, strike_fraction=0.0, dt=0.02):
        """``strike_fraction`` of the rows start where the strike lies inside the window
        (the remainder uniform), so sparse strike rewards get a steady share of the batch."""
        available = self.size if recent is None else min(self.size, int(recent))
        if available < window:
            raise ValueError("Replay has insufficient history for the window")
        dev = self.storage
        age = torch.randint(self.size - available, self.size - window + 1, (B,), device=dev)
        lo, hi = (self.N - self.val_n, self.N) if validation else (0, self.N - self.val_n)
        e = torch.randint(lo, hi, (B,), device=dev)
        n = int(round(B * float(strike_fraction)))
        if n > 0:
            eligible = self.strike_starts(window, dt)
            eligible[: self.size - available] = False
            eligible[:, :lo] = False
            eligible[:, hi:] = False
            cells = eligible.nonzero()
            if len(cells):
                pick = cells[torch.randint(len(cells), (n,), device=dev)]
                age[:n], e[:n] = pick[:, 0], pick[:, 1]
        rows = (self.ptr - self.size + age[:, None] + torch.arange(window, device=dev)) % self.T
        out = {k: v[rows, e[:, None]] for k, v in self.data.items() if keys is None or k in keys or k == "done"}
        if dev != torch.device(self.device):
            out = {k: v.to(self.device) for k, v in out.items()}
            e = e.to(self.device)
        done = out["done"].float()
        # Transition j is usable while no episode ended at 0..j-1.
        out["mask"] = torch.cumprod(torch.cat((torch.ones_like(done[:, :1]), 1 - done[:, :-1]), 1), 1)
        out["env_ids"] = e
        return out


def context_at(batch, t, keys):
    return {k: batch["ctx/" + k][:, t] for k in keys}


def lambda_weights(H, lam):
    """TD(lambda) mixture of the n-step imagined returns, n = 1..H.

    Returns (reward weight per step t, bootstrap weight at step t for t = 1..H-1,
    bootstrap weight at the horizon). ``lam`` None reproduces the paper's single
    horizon bootstrap. The mixture puts most bootstrap weight on short paths
    through the world model, whose gradient agrees across batches, instead of
    the 32-step path whose batch-to-batch cosine was 0.18 at 51k iterations.
    """
    if lam is None:
        return [1.0] * H, [0.0] * H, 1.0
    lam = float(lam)
    reward = [lam**t for t in range(H)]
    boot = [0.0] + [(1.0 - lam) * lam ** (t - 1) for t in range(1, H)]
    return reward, boot, lam ** (H - 1)


def bootstrap_cut(mask_t, mask_next, terminal_t=None):
    """Rows whose recorded episode ends after step ``t`` AND may bootstrap.

    A timeout (truncation) keeps the value of the next state. A recorded
    termination is terminal: its future is empty, so no critic bootstrap.
    Bootstrapping V(imagined next state) there made the objective
    ``-cost*f + (1-f)*V`` flat or inverted whenever that V was negative
    (the critic had learned the fall cost): the actor was rewarded for
    failing faster, cancelling the soft termination gradient.
    """
    cut = mask_t * (1 - mask_next)
    if terminal_t is not None:
        cut = cut * (1 - terminal_t.float())
    return cut


class Learner:
    def __init__(self, model, actor, actor_params, cfg, xi, active, noise_penalty, reasons, termination_cost, device):
        self.model, self.actor, self.cfg = model, actor, cfg
        self.xi, self.active = xi, active
        self.noise_penalty = noise_penalty
        self.reasons, self.termination_cost = reasons, float(termination_cost)
        self.wm = PhysicalWorldModel(xi.shape[-1], tuple(cfg.wm_hidden)).to(device)
        self.critic_norm = None
        self.critic = None
        self.critic_target = None
        # Polyak decay of a target critic used for every bootstrap (null = live critic, as upstream).
        self.target_decay = cfg.get("critic_target_decay", None)
        # Gradient scale of the critic bootstraps inside the actor objective (1 = upstream).
        # At 51k iterations the horizon bootstrap carried 95% of the actor gradient with a
        # batch-to-batch cosine of 0.05 (noise); the reward terms agreed at 0.5-0.98.
        self.boot_scale = float(cfg.get("bootstrap_grad_scale", 1.0))
        # Paper semantics: imagination never "ends" with the recorded episode; a
        # recorded termination only matters through the soft survival weight. The
        # continuation value is therefore bootstrapped at every recorded episode
        # end (true). False treats recorded terminations as terminal (no bootstrap).
        self.bootstrap_at_terminations = bool(cfg.get("bootstrap_at_terminations", True))
        # TD(lambda) mixture of n-step imagined returns in the actor objective (null = paper).
        self.bootstrap_lambda = cfg.get("bootstrap_lambda", None)
        # Weight the critic regression by the imagined survival up to each step
        # (Dreamer's cumulative continuation weight). Without it the critic is fit
        # on post-failure imagined states, whose persisting margin violation pays
        # the failure cost every step, so V(fallen) -> -cost and the survival term
        # alive * V cancels the explicit cost at the boundary (diagnosis 51300:
        # termination and episode-end bootstrap gradients of equal norm, opposite).
        self.critic_survival_weight = bool(cfg.get("critic_survival_weight", False))
        # Direct temporal smoothness on the imagined actions (paper: -0.1 *
        # ||a_t - a_{t-1}||^2 per step, tracking; -0.3 locomotion). The task's
        # executed_qdes_rate reward head carried |g| 0.7 against a critic
        # bootstrap of 46-110, so the actor ignored it (docs section 16).
        self.action_smooth_weight = float(cfg.get("action_smooth_weight", 0.0) or 0.0)
        # Phase weighting mirroring the task's executed_qdes_rate head: x multiplier
        # in HOLD, and the upper-arm joints free inside the strike window.
        self.action_smooth_hold_multiplier = float(cfg.get("action_smooth_hold_multiplier", 1.0) or 1.0)
        self.action_smooth_strike_free = bool(cfg.get("action_smooth_strike_free", False))
        # Sensor-noise (CAPS-spatial) penalty on the imagined observations along
        # the horizon instead of the recorded obs at step 0 only, with up to
        # sensor_noise_rows rows and a linear ramp of the coefficient from 0
        # over sensor_noise_ramp_iters actor updates (LCP-style warm start).
        self.sensor_noise_horizon = bool(cfg.get("sensor_noise_horizon", False))
        self.sensor_noise_rows = int(cfg.get("sensor_noise_rows", 4096) or 4096)
        self.sensor_noise_ramp = int(cfg.get("sensor_noise_ramp_iters", 0) or 0)
        self.iteration = None  # set by the trainer every iteration
        self._ramp_start = None
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.learning_rate)
        self.opt_pi = torch.optim.Adam(actor_params, lr=cfg.learning_rate, betas=(0.9, 0.95))
        self.actor_params = list(actor_params)
        self.opt_v = None
        self.device = device
        self.ctx_keys = None
        self.temperature = float(cfg.termination_temperature)
        self.projection_penalty = float(cfg.projection_penalty)
        row_clip = getattr(cfg, "action_grad_row_clip", None)
        self.row_clip = float(row_clip) if row_clip else None
        critic_clip = getattr(cfg, "critic_grad_row_clip", None)
        self.critic_clip = float(critic_clip) if critic_clip else None
        self._clip_stats, self._critic_clip_stats = [], []
        self.step_fn = self._step
        self.wm_step = self.wm.step
        self.heads = model.heads
        # CUDA-graph replays overwrite a compiled function's outputs; copy
        # the ones imagination keeps past the next call.
        self.keep = lambda x: x
        if bool(getattr(cfg, "compile", False)) and getattr(cfg, "compile_mode", None) == "reduce-overhead":
            self.keep = lambda x: x.clone()
        if bool(getattr(cfg, "compile", False)):
            if not bool(getattr(cfg, "donated_buffers", True)):
                import torch._functorch.config as functorch_config

                functorch_config.donated_buffer = False
            # Compiling the whole imagined step is not used: its backward
            # produced non-finite gradients that the boundary guard cannot see.
            parts = list(getattr(cfg, "compile_parts", None) or ["heads", "wm_in_step", "obs", "wm"])
            # "reduce-overhead" replays CUDA graphs: the imagined step is
            # kernel-launch bound, so this removes most of its cost.
            mode = getattr(cfg, "compile_mode", None) or None
            compile_ = partial(torch.compile, dynamic=False, mode=mode)
            if "wm" in parts:
                self.wm_step = compile_(self.wm.step)
            if "heads" in parts:
                self.heads = compile_(model.heads)
            if "obs" in parts:
                model.clean_obs = compile_(model.clean_obs)
                model.critic_features = compile_(model.critic_features)
            if "wm_in_step" in parts:
                self.wm.step = compile_(self.wm.step)

    # ------------------------------------------------------------------ helpers
    def _xi(self, ids):
        return self.xi[ids]

    def _start(self, batch, t):
        return push(batch["s_cmd"][:, t], batch["push"][:, t])

    def _ensure_critic(self, features):
        if self.critic is None:
            dim = features.shape[-1]
            self.critic_norm = RunningNorm(dim).to(self.device)
            self.critic = mlp(dim, list(self.cfg.critic_hidden), 1, layer_norm=False).to(self.device)
            self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
            self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.learning_rate)

    def value(self, features):
        return self.critic(self.critic_norm(features)).squeeze(-1)

    def scaled_boot(self, v):
        """Bootstrap value with its gradient scaled by ``bootstrap_grad_scale`` (value unchanged)."""
        if self.boot_scale >= 1.0:
            return v
        return self.boot_scale * v + (1.0 - self.boot_scale) * v.detach()

    def noise_scale(self):
        """Ramp factor of the sensor-noise penalty in [0, 1] (1 without a ramp)."""
        if self.sensor_noise_ramp <= 0 or self.iteration is None:
            return 1.0
        if self._ramp_start is None:
            self._ramp_start = int(self.iteration)
        return min(1.0, max(0.0, (int(self.iteration) - self._ramp_start) / self.sensor_noise_ramp))

    def value_boot(self, features):
        """Bootstrap value: the target critic when ``critic_target_decay`` is set, else the live one."""
        net = self.critic_target if self.target_decay is not None else self.critic
        return net(self.critic_norm(features)).squeeze(-1)

    @torch.no_grad()
    def update_target(self):
        if self.target_decay is None or self.critic_target is None:
            return
        d = float(self.target_decay)
        for pt, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            pt.mul_(d).add_(p, alpha=1.0 - d)

    def _clip_hook(self, g):
        clipped = clip_rows(g, self.row_clip)
        self._clip_stats.append(((clipped - g).abs().sum(-1) > 0).float().mean())
        return clipped

    def _critic_clip_hook(self, g):
        clipped = clip_rows(g, self.critic_clip)
        self._critic_clip_stats.append(((clipped - g).abs().sum(-1) > 0).float().mean())
        return clipped

    def set_learning_rate(self, lr):
        for opt in (self.opt_wm, self.opt_pi, self.opt_v):
            if opt is not None:
                for group in opt.param_groups:
                    group["lr"] = lr

    # ------------------------------------------------------------------ world model
    def model_update(self, batch):
        wm, k = self.wm, batch["s_end"].shape[1]
        ids = batch["env_ids"]
        xi = self._xi(ids)
        s = self._start(batch, 0)
        with torch.no_grad():
            valid = batch["mask"].bool()
            s0 = torch.stack([self._start(batch, j) for j in range(k)], 1)
            wm.in_norm.update(wm.inputs(
                s0[valid], batch["qexec"][valid], batch["qexec_prev"][valid],
                batch["ctx/cmd_delay"][valid], xi[:, None].expand(-1, k, -1)[valid],
            ))
            wm.out_norm.update(wm.increments(s0[valid], batch["s_end"][valid]))
        errors = []
        for j in range(k):
            s_end = self.wm_step(s, batch["qexec"][:, j], batch["qexec_prev"][:, j], batch["ctx/cmd_delay"][:, j], xi)
            errors.append(wm.error(s_end, batch["s_end"][:, j]).square().mean(-1))
            if j + 1 < k:
                s = push(s_end, batch["push"][:, j + 1])
        error = torch.stack(errors, 1)
        loss = (error * batch["mask"]).sum() / batch["mask"].sum().clamp_min(1)
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("world-model gradient is nonfinite")
        self.opt_wm.step()
        last = batch["mask"][:, -1]
        return dict(
            wm_loss=float(loss.detach()),
            wm_last_step_error=float((error[:, -1].detach() * last).sum() / last.sum().clamp_min(1)),
            wm_grad_norm=float(norm.detach()),
        )

    @torch.no_grad()
    def validate(self, batch):
        """Held-out k-step error per state component, in physical units."""
        wm, k = self.wm, batch["s_end"].shape[1]
        ids = batch["env_ids"]
        xi = self._xi(ids)
        s = self._start(batch, 0)
        out = {}
        for j in range(k):
            s_end = wm.step(s, batch["qexec"][:, j], batch["qexec_prev"][:, j], batch["ctx/cmd_delay"][:, j], xi)
            if j + 1 < k:
                s = push(s_end, batch["push"][:, j + 1])
        mask = batch["mask"][:, -1].bool()
        d = (wm.increments(batch["s_end"][:, -1], s_end))[mask]
        if d.numel():
            out["WorldModel/validation_k_step_position_m"] = float(d[:, 0:3].norm(dim=-1).mean())
            out["WorldModel/validation_k_step_rotation_rad"] = float(d[:, 3:6].norm(dim=-1).mean())
            out["WorldModel/validation_k_step_lin_vel_mps"] = float(d[:, 6:9].norm(dim=-1).mean())
            out["WorldModel/validation_k_step_joint_rad"] = float(d[:, 12:12 + S.NJ].abs().mean())
            out["WorldModel/validation_k_step_joint_vel"] = float(d[:, 12 + S.NJ:].abs().mean())
            out["WorldModel/validation_k_step_normalized_mse"] = float(wm.error(s_end, batch["s_end"][:, -1])[mask].square().mean())
        return out

    @torch.no_grad()
    def consistency(self, batch):
        """Imagination plumbing with recorded actions and real states (no model).

        Context alignment, delayed q_des, pushes, HOLD dwell and the correction
        must reproduce the recorded rewards and executed targets exactly.
        """
        m = self.model
        H = batch["s_end"].shape[1]
        ids = batch["env_ids"]
        keys = self.ctx_keys
        s_cmd = batch["s_cmd"][:, 0]
        qexec_prev = batch["qexec_prev"][:, 0]
        qcmd_prev = batch["qcmd_prev"][:, 0]
        dwell = batch["ctx/dwell_prev"][:, 0]
        ready = batch["ctx/ready_prev"][:, 0]
        a_prev = batch["ctx/action_prev"][:, 0]
        dq_prev = batch["ctx/dq_prev"][:, 0]
        reward_error = qexec_error = state_error = 0.0
        for t in range(H):
            mask = batch["mask"][:, t].bool()
            ctx = context_at(batch, t, keys)
            ctx["dwell_prev"], ctx["ready_prev"] = dwell, ready
            ctx["action_prev"], ctx["dq_prev"] = a_prev, dq_prev
            if t:
                state_error = max(state_error, float(((s_cmd - batch["s_cmd"][:, t]) * mask[:, None]).abs().amax()))
            a = batch["action"][:, t]
            qcmd = m.qdes_command(a, ids)
            qexec = torch.where((ctx["qdes_delay"] > 0.5)[:, None], qcmd_prev, qcmd)
            qexec_error = max(qexec_error, float(((qexec - batch["qexec"][:, t]) * mask[:, None]).abs().amax()))
            s_end = batch["s_end"][:, t]
            fr = m.frame(s_cmd, s_end, a, qexec, qexec_prev, ctx, ids)
            r = m.step_reward(m.rewards(fr)) + batch["correction"][:, t]
            real = batch["reward"][:, t] - batch["terminal_reward"][:, t]
            reward_error = max(reward_error, float(((r - real) * mask).abs().amax()))
            dwell, ready = getattr(fr, "hold_state", (dwell, ready))
            a_prev, dq_prev = a, qexec - qexec_prev
            qexec_prev, qcmd_prev, s_cmd = qexec, qcmd, s_end
        return {
            "Imagination/check_reward_max_abs_error": reward_error,
            "Imagination/check_qexec_max_abs_error": qexec_error,
            "Imagination/check_state_chain_max_abs_error": state_error,
        }

    # ------------------------------------------------------------------ imagination
    def imagine(self, batch):
        """Roll the actor through the model for ``H`` steps from real replay states.

        Returns the differentiable objective (with critic bootstraps at the
        horizon and at recorded episode ends) and detached critic training data.
        Each step is gradient-checkpointed: only carried states are stored and
        the step is recomputed during the backward pass.
        """
        from functools import partial
        from torch.utils.checkpoint import checkpoint

        m, cfg = self.model, self.cfg
        H = batch["s_end"].shape[1] - 1
        ids = batch["env_ids"]
        gamma = float(cfg.gamma)
        keys = self.ctx_keys
        s_cmd = batch["s_cmd"][:, 0]
        s = self._start(batch, 0)
        a_prev = batch["action_prev"][:, 0]
        qexec_prev = batch["qexec_prev"][:, 0]
        qcmd_prev = batch["qcmd_prev"][:, 0]
        dq_prev = batch["ctx/dq_prev"][:, 0]
        dwell = batch["ctx/dwell_prev"][:, 0]
        ready = batch["ctx/ready_prev"][:, 0]
        alive = torch.ones(ids.shape[0], device=self.device)
        objective = torch.zeros_like(alive)
        data = dict(features=[], rewards=[], continues=[], masks=[], boot=[], terminal=[], action_rate=[], obs=[])
        terminal = None if self.bootstrap_at_terminations else batch.get("terminal")
        self._ensure_critic(m.critic_features(s, a_prev, context_at(batch, 0, keys), ids))
        use_checkpoint = bool(getattr(cfg, "imagination_checkpoint", True))
        w_reward, w_boot, w_end = lambda_weights(H, self.bootstrap_lambda)
        for t in range(H):
            ctx = context_at(batch, t, keys)
            fn = partial(self.step_fn, ctx, batch["env_ids"])
            if w_boot[t] > 0:
                # n-step return ending here: value of the imagined state s_t under its own context.
                ctx_v = dict(ctx)
                ctx_v["dwell_prev"], ctx_v["ready_prev"] = dwell, ready
                v_t = self.value(self.keep(m.critic_features(s, a_prev, ctx_v, ids)))
                objective = objective + gamma**t * w_boot[t] * alive * batch["mask"][:, t] * self.scaled_boot(v_t)
            # After a recorded episode end the context belongs to the next
            # episode: evaluate those rows at the recorded state, without gradient.
            valid = batch["mask"][:, t].bool()
            real_cmd, real_start = batch["s_cmd"][:, t], self._start(batch, t)
            s = torch.where(valid[:, None], s, real_start)
            s_cmd = torch.where(valid[:, None], s_cmd, real_cmd)
            a_prev = torch.where(valid[:, None], a_prev, batch["action_prev"][:, t])
            qexec_prev = torch.where(valid[:, None], qexec_prev, batch["qexec_prev"][:, t])
            qcmd_prev = torch.where(valid[:, None], qcmd_prev, batch["qcmd_prev"][:, t])
            dq_prev = torch.where(valid[:, None], dq_prev, batch["ctx/dq_prev"][:, t])
            args = (
                batch["obs_delta"][:, t], batch["correction"][:, t], batch["push"][:, t + 1],
                s, s_cmd, a_prev, qexec_prev, qcmd_prev, dq_prev, dwell, ready,
            )
            out = checkpoint(fn, *args, use_reentrant=False) if use_checkpoint else fn(*args)
            r, cont, s_end, s_next, a, qexec, qcmd, dq_prev, dwell, ready, feature, boot, value_boot, obs = out
            mask = batch["mask"][:, t]
            objective = objective + gamma**t * w_reward[t] * alive * mask * r
            # Temporal smoothness of the imagined action sequence (a_prev at
            # t = 0 is the recorded action; its noise is zero-mean).
            da = (a - a_prev).square().sum(-1)
            if self.action_smooth_weight > 0:
                w_row = torch.full_like(mask, self.action_smooth_weight)
                if self.action_smooth_hold_multiplier != 1.0:
                    w_row = torch.where(ctx["in_hold"].bool(), w_row * self.action_smooth_hold_multiplier, w_row)
                if self.action_smooth_strike_free:
                    free = ctx["qdes_protected"].float()[:, None] * m.qdes_upper
                    da_w = ((a - a_prev).square() * (1 - free)).sum(-1)
                else:
                    da_w = da
                objective = objective - gamma**t * alive * mask * w_row * da_w
            data["action_rate"].append(da.detach())
            if self.sensor_noise_horizon:
                data["obs"].append(obs)
            alive = alive * cont * mask
            term_t = terminal[:, t] if terminal is not None else torch.zeros_like(mask)
            cut = bootstrap_cut(mask, batch["mask"][:, t + 1], term_t)
            objective = objective + gamma ** (t + 1) * w_reward[t] * alive * cut * self.scaled_boot(value_boot)
            a_prev, qexec_prev, qcmd_prev, s_cmd, s = a, qexec, qcmd, s_end, s_next
            data["features"].append(feature)
            data["rewards"].append(r.detach())
            data["continues"].append(cont.detach())
            data["masks"].append(mask)
            data["boot"].append(boot)
            data["terminal"].append(term_t.float())
        ctx = context_at(batch, H, keys)
        ctx["dwell_prev"], ctx["ready_prev"] = dwell, ready
        end = self.keep(m.critic_features(s, a_prev, ctx, ids))
        if self.critic_clip and end.requires_grad:
            # Bound each row's critic gradient at the horizon (the outlier
            # source); rewards and soft terminations keep exact gradients.
            end.register_hook(self._critic_clip_hook)
        objective = objective + gamma**H * w_end * alive * batch["mask"][:, H] * self.scaled_boot(self.value(end))
        data["features"].append(end.detach())
        data["alive"] = alive.detach()
        return objective, data

    def _step(self, ctx, ids, obs_delta, correction, push_next, s, s_cmd, a_prev, qexec_prev, qcmd_prev, dq_prev,
              dwell, ready):
        m = self.model
        ctx = dict(ctx)
        ctx["dwell_prev"], ctx["ready_prev"] = dwell, ready
        # Histories the action term owns in the real task: the previous raw action
        # (raw action-rate cost) and the previous executed-command increment
        # (reversal cost). Both follow the imagined actions, not the tape.
        ctx["action_prev"], ctx["dq_prev"] = a_prev, dq_prev
        feature = self.keep(m.critic_features(s, a_prev, ctx, ids).detach())
        obs = m.clean_obs(s, a_prev, ctx, ids) + obs_delta
        raw = self.actor(obs)
        if self.row_clip and raw.requires_grad:
            raw.register_hook(self._clip_hook)
        a = raw * self.active
        qcmd = m.qdes_command(a, ids)
        qexec = torch.where((ctx["qdes_delay"] > 0.5)[:, None], qcmd_prev, qcmd)
        s_end = self.keep(self.wm.step(s, qexec, qexec_prev, ctx["cmd_delay"], self._xi(ids)))
        inputs = (s_cmd, s_end, a, qexec, qexec_prev)
        if self.heads is not m.heads:
            inputs = finite_grad(*inputs)
            ctx["action_prev"], ctx["dq_prev"] = finite_grad(ctx["action_prev"], ctx["dq_prev"])
        values, margins, elapsed, ready_next = (self.keep(x) for x in self.heads(*inputs, ctx, ids))
        r = m.step_reward(values) + correction
        logits = margins / self.temperature
        failure = torch.sigmoid(logits)
        r = r + failure.sum(-1) * self.termination_cost
        outside = (raw - m.qdes_project(raw, ids)) * self.active
        r = r - self.projection_penalty * outside.square().mean(-1)
        # prod(1 - sigmoid) in log space: finite gradients when a failure saturates.
        cont = F.logsigmoid(-logits).sum(-1).exp()
        dwell, ready = elapsed, ready_next
        s_next = push(s_end, push_next)
        # Value of the next start state under this step's context: used where
        # the recorded episode ends inside the window.
        boot = self.keep(m.critic_features(s_next, a, ctx, ids))
        return (r, cont, s_end, s_next, a, qexec, qcmd, qexec - qexec_prev, dwell, ready, feature, boot.detach(),
                self.value_boot(boot), obs.detach())

    def debug_compare_gradients(self, batch, rows=256):
        """Cosine between actor gradients with compiled and eager heads (same rows)."""
        batch = {k: v[:rows] for k, v in batch.items()}
        grads = []
        for heads in (self.heads, self.model.heads):
            saved = self.heads
            self.heads = heads
            for p in self.wm.parameters():
                p.requires_grad_(False)
            try:
                objective, _ = self.imagine(batch)
                g = torch.autograd.grad(objective.mean(), self.actor_params, allow_unused=True)
                grads.append(torch.cat([x.flatten() if x is not None else p.new_zeros(p.numel())
                                        for x, p in zip(g, self.actor_params)]))
            finally:
                self.heads = saved
                for p in self.wm.parameters():
                    p.requires_grad_(True)
        a, b = grads
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        print(f"[supermimic] compiled vs eager actor gradient: cos={cos:.6f} "
              f"|compiled|={float(a.norm()):.4e} |eager|={float(b.norm()):.4e} "
              f"finite={bool(torch.isfinite(a).all())}/{bool(torch.isfinite(b).all())}", flush=True)

    def policy_update(self, batches):
        """One actor step from one or more imagined batches (gradients accumulated,
        so several batches behave like one larger batch within the memory limit)."""
        if isinstance(batches, dict):
            batches = [batches]
        for p in self.wm.parameters():
            p.requires_grad_(False)
        if self.critic is not None:
            for p in self.critic.parameters():
                p.requires_grad_(False)
        import time

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self._clip_stats, self._critic_clip_stats = [], []
        forward_seconds = 0.0
        datas, objectives, noises = [], [], []
        try:
            self.opt_pi.zero_grad(set_to_none=True)
            for batch in batches:
                ta = time.perf_counter()
                objective, data = self.imagine(batch)
                if self.sensor_noise_horizon:
                    # Imagined observations of every valid row-step, shuffled;
                    # the penalty uses its first sensor_noise_samples rows.
                    rows = torch.cat(data["obs"])
                    rows = rows[torch.cat(data["masks"]) > 0]
                    pick = torch.randperm(rows.shape[0], device=rows.device)[: self.sensor_noise_rows]
                    obs0 = rows[pick]
                else:
                    obs0 = self.model.clean_obs(
                        self._start(batch, 0), batch["action_prev"][:, 0],
                        context_at(batch, 0, self.ctx_keys), batch["env_ids"],
                    ) + batch["obs_delta"][:, 0]
                noise = self.noise_scale() * self.noise_penalty(obs0.detach())
                loss = (-objective.mean() + noise) / len(batches)
                torch.cuda.synchronize()
                forward_seconds += time.perf_counter() - ta
                loss.backward()
                datas.append(data)
                objectives.append(objective.detach())
                noises.append(noise.detach())
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            norm = nn.utils.clip_grad_norm_(self.actor_params, 1.0)
            nn.utils.clip_grad_value_(self.actor_params, 0.1)
            stepped = bool(torch.isfinite(norm))
            if stepped:
                self.opt_pi.step()
        finally:
            for p in self.wm.parameters():
                p.requires_grad_(True)
            if self.critic is not None:
                for p in self.critic.parameters():
                    p.requires_grad_(True)
        for data in datas:
            value_metrics = self.critic_update(data)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t1 = t0 + forward_seconds
        data = datas[-1]
        objective = torch.cat(objectives)
        noise = torch.stack(noises).mean()
        loss = -objective.mean() + noise
        r = torch.cat([torch.stack(d["rewards"], 0) for d in datas], 1)
        mk = torch.cat([torch.stack(d["masks"], 0) for d in datas], 1)
        da = torch.cat([torch.stack(d["action_rate"], 0) for d in datas], 1)
        clipped = float(torch.stack(self._clip_stats).mean()) if self._clip_stats else 0.0
        critic_clipped = float(torch.stack(self._critic_clip_stats).mean()) if self._critic_clip_stats else 0.0
        return dict(
            actor_row_clip_fraction=clipped,
            critic_row_clip_fraction=critic_clipped,
            actor_loss=float(loss.detach()),
            imagined_return=float(objective.detach().mean()),
            imagined_step_reward=float((r * mk).sum() / mk.sum().clamp_min(1)),
            imagined_survival=float(torch.cat([d["alive"] for d in datas]).mean()),
            imagined_valid_steps=float(mk.sum(0).mean()),
            actor_grad_norm=float(norm.detach()),
            actor_step_skipped=int(not stepped),
            sensor_noise_weighted=float(noise.detach()),
            sensor_noise_scale=float(self.noise_scale()),
            imagined_action_rate=float((da * mk).sum() / mk.sum().clamp_min(1)),
            imagined_action_smooth_weighted=float(self.action_smooth_weight * (da * mk).sum() / mk.sum().clamp_min(1)),
            **value_metrics,
            **{
                "Performance/imagine_forward_seconds": t1 - t0,
                "Performance/imagine_backward_seconds": t2 - t1,
                "Performance/critic_seconds": t3 - t2,
            },
        )

    def critic_update(self, data):
        cfg = self.cfg
        gamma, lam = float(cfg.gamma), float(cfg.lam)
        X = torch.stack(data["features"], 0)  # (H+1, B, F)
        boot = torch.stack(data["boot"], 0)  # (H, B, F)
        H = boot.shape[0]
        self.critic_norm.update(X.flatten(0, 1))
        with torch.no_grad():
            V = self.value_boot(X.flatten(0, 1)).view(X.shape[:2])
            Vb = self.value_boot(boot.flatten(0, 1)).view(boot.shape[:2])
            if data.get("terminal"):
                # Recorded terminations are terminal states: no value after them
                # (the last window step bootstraps from the horizon value V[H]).
                term = torch.stack(data["terminal"], 0)
                Vb = Vb * (1 - term)
                V_end = V[-1] * (1 - term[-1])
            else:
                V_end = V[-1]
            R = V_end
            targets = []
            for t in range(H - 1, -1, -1):
                d = 1 - data["continues"][t]
                nxt = data["masks"][t + 1] if t + 1 < H else torch.ones_like(data["masks"][t])
                following = torch.where(nxt.bool(), V[t + 1] if t + 1 < H else V_end, Vb[t])
                R = torch.where(nxt.bool(), R, Vb[t])
                R = data["rewards"][t] + gamma * (1 - d) * ((1 - lam) * following + lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1], 0)
            weight = torch.stack(data["masks"], 0)
            if getattr(self, "critic_survival_weight", False):
                cont = torch.stack(data["continues"], 0)  # (H, B), survival of each step
                survived = torch.cumprod(torch.cat((torch.ones_like(cont[:1]), cont[:-1]), 0), 0)
                weight = weight * survived
        Xf, Yf, Wf = X[:-1].flatten(0, 1), targets.flatten(), weight.flatten()
        n = Xf.shape[0]
        mb = max(1, n // int(cfg.critic_minibatches))
        loss = torch.zeros((), device=self.device)
        for _ in range(int(cfg.critic_epochs)):
            perm = torch.randperm(n, device=self.device)
            for i in range(int(cfg.critic_minibatches)):
                idx = perm[i * mb:(i + 1) * mb]
                err = (self.value(Xf[idx]) - Yf[idx]).square()
                loss = (err * Wf[idx]).sum() / Wf[idx].sum().clamp_min(1)
                self.opt_v.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
        self.update_target()
        return dict(
            value_loss=float(loss.detach()),
            value_target_mean=float((targets * weight).sum() / weight.sum().clamp_min(1)),
        )


def cosine_lr(base, it, decay_iters, floor=0.03):
    if it >= decay_iters:
        return base * floor
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * it / decay_iters)))
