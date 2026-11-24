"""Checkpoint diagnosis: what the policy does per phase, and where its gradient comes from.

1. Real rollouts with the checkpoint actor (collection noise): per-joint error
   to the HOLD reference and to the stroke reference, per phase and per
   time-to-strike bin; raw-action saturation; real reward terms per phase.
2. Actor-gradient decomposition on imagined windows from that tape (eager
   heads): every weighted reward term, soft terminations, projection penalty,
   critic bootstraps and the sensor-noise penalty; norm, cosine with the total
   and share of the total along its direction (shares sum to one).
3. Critic check: V(s_t) against n-step real returns on the same tape.
"""

from __future__ import annotations

from functools import partial
import json
import math
import torch
from torch.nn import functional as F

from . import state as S
from .learner import TapeReplay, clip_rows, context_at, bootstrap_cut
from .recorder import Recorder
from .world_model import push

PHASES = ("hold", "strike", "recover")
TTS_EDGES = (-1.5, -1.0, -0.7, -0.4, -0.15, 0.15, 0.4, 0.8, 1.5)


class Accumulator:
    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, key, value, mask):
        """Masked sum over rows of ``value`` (rows, ...)."""
        m = mask.float()
        s = (value * m.view(-1, *([1] * (value.ndim - 1)))).sum(0)
        self.sums[key] = self.sums.get(key, 0) + s
        self.counts[key] = self.counts.get(key, 0) + float(m.sum())

    def mean(self, key):
        return self.sums[key] / max(self.counts[key], 1.0)


def _phases(ctx):
    hold = ctx["in_hold"].bool()
    recover = ~hold & (ctx["recovery_gate"] > 0)
    return dict(hold=hold, strike=~hold & ~recover, recover=recover)


@torch.no_grad()
def rollout(env, parts, recorder, obs, steps, noise, replay):
    from ..trainer_analytic import tape_record

    model, actor, active = parts.model, parts.actor, parts.active
    names = list(env.scene["robot"].joint_names)
    hold_names = [names[i] for i in model.active_ids]
    acc = Accumulator()
    falls = 0
    for _ in range(steps):
        a = (actor(obs["policy"]) + noise * torch.randn(env.num_envs, 31, device=env.device)) * active
        obs, r, terminated, truncated, _ = env.step(a)
        record, ctx = tape_record(model, recorder, parts.keep, parts.term_index, env.step_dt, r, terminated, truncated)
        replay.add(record)
        if parts.learner.ctx_keys is None:
            parts.learner.ctx_keys = list(ctx)
        recorder.observe(obs)
        falls += int(terminated.sum())
        rec = recorder.record
        q = S.part(rec["s_end"], "q")
        hold_error = q[:, model.active_ids] - model.hold_ref
        ref_error = q - ctx["q_target"]
        raw = rec["action"]
        outside = ((raw < model.const["raw_lo"]) | (raw > model.const["raw_hi"])).float() * active
        real = recorder.real
        for phase, mask in _phases(ctx).items():
            acc.add(phase + "/hold_error", hold_error, mask)
            acc.add(phase + "/hold_error_sq", hold_error.square(), mask)
            acc.add(phase + "/ref_error_sq", ref_error.square(), mask)
            acc.add(phase + "/raw_abs", raw.abs(), mask)
            acc.add(phase + "/outside", outside, mask)
            acc.add(phase + "/reward", real, mask)
        bins = torch.bucketize(ctx["tts"], ctx["tts"].new_tensor(TTS_EDGES))
        for b in range(len(TTS_EDGES) + 1):
            mask = (bins == b) & ~ctx["in_hold"].bool()
            acc.add(f"tts{b}/hold_error_sq", hold_error.square(), mask)
            acc.add(f"tts{b}/ref_error_sq", ref_error.square(), mask)
            acc.add(f"tts{b}/recovery_gate", ctx["recovery_gate"].float(), mask)
            acc.add(f"tts{b}/track_gate", ctx["track_gate"].float(), mask)
    report = dict(steps=steps, environments=env.num_envs, terminations=falls, phases={}, time_to_strike={})
    groups = {g: [model.active_ids.index(j) for j in ids] for g, ids in model.groups.items()}
    for phase in PHASES:
        if (phase + "/hold_error") not in acc.counts or not acc.counts[phase + "/hold_error"]:
            continue
        mean = acc.mean(phase + "/hold_error")
        rms = acc.mean(phase + "/hold_error_sq").sqrt()
        ref = acc.mean(phase + "/ref_error_sq").sqrt()
        report["phases"][phase] = dict(
            fraction=acc.counts[phase + "/hold_error"] / (steps * env.num_envs),
            hold_error_group_rms_rad={g: float(rms[c].square().mean().sqrt()) for g, c in groups.items()},
            stroke_error_group_rms_rad={g: float(ref[ids].square().mean().sqrt()) for g, ids in model.groups.items()},
            hold_error_per_joint={n: dict(mean_rad=round(float(mean[i]), 4), rms_rad=round(float(rms[i]), 4))
                                  for i, n in enumerate(hold_names)},
            raw_action_abs_mean={n: round(float(v), 3) for n, v in zip(names, acc.mean(phase + "/raw_abs"))},
            raw_action_saturated_fraction={n: round(float(v), 4) for n, v in zip(names, acc.mean(phase + "/outside"))
                                           if float(v) > 0},
            reward_per_second={t: round(float(v), 5) for t, v in zip(model.terms, acc.mean(phase + "/reward"))
                               if float(v) != 0},
        )
    edges = (-math.inf,) + TTS_EDGES + (math.inf,)
    for b in range(len(TTS_EDGES) + 1):
        key = f"tts{b}/hold_error_sq"
        if not acc.counts.get(key):
            continue
        rms = acc.mean(key).sqrt()
        ref = acc.mean(f"tts{b}/ref_error_sq").sqrt()
        report["time_to_strike"][f"[{edges[b]:.2f},{edges[b + 1]:.2f})"] = dict(
            samples=acc.counts[key],
            recovery_gate=float(acc.mean(f"tts{b}/recovery_gate")),
            track_gate=float(acc.mean(f"tts{b}/track_gate")),
            hold_error_group_rms_rad={g: round(float(rms[c].square().mean().sqrt()), 4) for g, c in groups.items()},
            stroke_error_group_rms_rad={g: round(float(ref[ids].square().mean().sqrt()), 4)
                                        for g, ids in model.groups.items()},
        )
    return report, obs


def _roll(learner, batch, steps, frozen_from=None, values_at=(), clip=None, critic_clip=None):
    """``Learner.imagine`` split into additive per-step components.

    From step ``frozen_from`` on, the actor runs with detached parameters, so
    later returns depend on the trained parameters only through the state
    reached at that step (what the critic's state gradient must reproduce).
    ``values_at`` returns discounted, survival-weighted V(s_t) at those steps.
    """
    m, cfg = learner.model, learner.cfg
    ids = batch["env_ids"]
    gamma = float(cfg.gamma)
    keys = learner.ctx_keys
    w = m.weights.clone()
    w[m.termination_index] = 0
    frozen = {k: v.detach() for k, v in learner.actor.named_parameters()}
    s_cmd = batch["s_cmd"][:, 0]
    s = learner._start(batch, 0)
    a_prev = batch["action_prev"][:, 0]
    qexec_prev = batch["qexec_prev"][:, 0]
    qcmd_prev = batch["qcmd_prev"][:, 0]
    dq_prev = batch["ctx/dq_prev"][:, 0]
    dwell = batch["ctx/dwell_prev"][:, 0]
    ready = batch["ctx/ready_prev"][:, 0]
    alive = torch.ones(ids.shape[0], device=learner.device)
    survived = torch.ones_like(alive)
    learner._ensure_critic(m.critic_features(s, a_prev, context_at(batch, 0, keys), ids))
    rows, values, saturated, raws, states = [], {}, [], [], []
    values_aprev = {}
    for t in range(steps + 1):
        ctx = context_at(batch, t, keys)
        ctx["dwell_prev"], ctx["ready_prev"] = dwell, ready
        if t in values_at:
            features = m.critic_features(s, a_prev, ctx, ids)
            if critic_clip and t == steps:
                features.register_hook(partial(clip_rows, factor=critic_clip))
            v = learner.value(features)
            values[t] = gamma**t * alive * batch["mask"][:, t] * v
            # Same value with the state path cut: only the last action reaches the critic.
            va = learner.value(m.critic_features(s.detach(), a_prev, ctx, ids))
            values_aprev[t] = gamma**t * alive.detach() * batch["mask"][:, t] * va
        if t == steps:
            break
        valid = batch["mask"][:, t].bool()
        s = torch.where(valid[:, None], s, learner._start(batch, t))
        s_cmd = torch.where(valid[:, None], s_cmd, batch["s_cmd"][:, t])
        a_prev = torch.where(valid[:, None], a_prev, batch["action_prev"][:, t])
        qexec_prev = torch.where(valid[:, None], qexec_prev, batch["qexec_prev"][:, t])
        qcmd_prev = torch.where(valid[:, None], qcmd_prev, batch["qcmd_prev"][:, t])
        dq_prev = torch.where(valid[:, None], dq_prev, batch["ctx/dq_prev"][:, t])
        ctx["action_prev"], ctx["dq_prev"] = a_prev, dq_prev
        states.append(s.detach())
        obs = m.clean_obs(s, a_prev, ctx, ids) + batch["obs_delta"][:, t]
        if frozen_from is not None and t >= frozen_from:
            raw = torch.func.functional_call(learner.actor, frozen, (obs,))
        else:
            raw = learner.actor(obs)
            if clip:
                raw.register_hook(partial(clip_rows, factor=clip))
        raws.append(raw)
        a = raw * learner.active
        qcmd = m.qdes_command(a, ids)
        qexec = torch.where((ctx["qdes_delay"] > 0.5)[:, None], qcmd_prev, qcmd)
        s_end = learner.wm.step(s, qexec, qexec_prev, ctx["cmd_delay"], learner._xi(ids))
        heads, margins, elapsed, ready_next = m.heads(s_cmd, s_end, a, qexec, qexec_prev, ctx, ids)
        logits = margins / learner.temperature
        outside = (raw - m.qdes_project(raw, ids)) * learner.active
        saturated.append(float((outside.abs() > 1e-6).float().sum() / (learner.active.sum() * raw.shape[0])))
        mask = batch["mask"][:, t]
        k = gamma**t * alive * mask
        row = dict(
            terms=k[:, None] * heads * w * m.dt,
            termination=k * torch.sigmoid(logits).sum(-1) * learner.termination_cost,
            projection=-k * learner.projection_penalty * outside.square().mean(-1),
            correction=k * batch["correction"][:, t],
            risk=torch.sigmoid(logits).amax(-1).detach(),
        )
        cont = F.logsigmoid(-logits).sum(-1).exp()
        # Imagined survival over the recorded episode only (not cut by its end).
        survived = survived * torch.where(mask > 0, cont, torch.ones_like(cont)).detach()
        alive = alive * cont * mask
        s_next = push(s_end, batch["push"][:, t + 1])
        term_t = (batch["terminal"][:, t] if "terminal" in batch and not getattr(learner, "bootstrap_at_terminations", True)
                  else None)
        cut = bootstrap_cut(mask, batch["mask"][:, t + 1], term_t)
        boot = learner.value(m.critic_features(s_next, a, ctx, ids))
        row["bootstrap_episode_end"] = gamma ** (t + 1) * alive * cut * boot
        rows.append(row)
        dwell, ready = elapsed, ready_next
        dq_prev = qexec - qexec_prev
        a_prev, qexec_prev, qcmd_prev, s_cmd, s = a, qexec, qcmd, s_end, s_next
    extra = dict(imagined_raw_saturated_fraction=sum(saturated) / len(saturated), imagined_survival=float(alive.mean()),
                 states=states, alive=alive.detach(), survived=survived, values_aprev=values_aprev)
    return rows, values, extra, raws


def _total(rows):
    return sum(r["terms"].sum(-1) + r["termination"] + r["projection"] + r["correction"]
               + r["bootstrap_episode_end"] for r in rows)


def components(learner, batch):
    """``Learner.imagine`` with the objective split into additive components (B,)."""
    H = batch["s_end"].shape[1] - 1
    rows, values, extra, _ = _roll(learner, batch, H, values_at=(H,))
    terms = sum(r["terms"] for r in rows)
    out = {k: sum(r[k] for r in rows) for k in ("termination", "projection", "correction", "bootstrap_episode_end")}
    out["bootstrap_horizon"] = values[H]
    m, keys = learner.model, learner.ctx_keys
    obs0 = m.clean_obs(learner._start(batch, 0), batch["action_prev"][:, 0], context_at(batch, 0, keys),
                       batch["env_ids"]) + batch["obs_delta"][:, 0]
    noise = learner.noise_penalty(obs0.detach())
    return terms, out, noise, {k: v for k, v in extra.items()
                               if k not in ("states", "alive", "survived", "values_aprev")}


def row_gradient_report(learner, batch):
    """How concentrated the actor signal is: per-row action-gradient norms per component.

    Rows are independent in imagination, so the gradient of the batch sum with
    respect to a row's actions is that row's own gradient.
    """
    H = batch["s_end"].shape[1] - 1
    rows, values, _, raws = _roll(learner, batch, H, values_at=(H,))
    objectives = dict(
        rewards=sum(r["terms"].sum(-1) + r["projection"] for r in rows),
        termination=sum(r["termination"] for r in rows),
        bootstrap_episode_end=sum(r["bootstrap_episode_end"] for r in rows),
        bootstrap_horizon=values[H],
    )
    objectives["total"] = sum(objectives.values())
    risk = torch.stack([r["risk"] for r in rows]).amax(0)
    mask = batch["mask"]
    ended = (mask[:, 1:] < mask[:, :-1]).any(1)
    risky = (risk > 0.05) | ended
    report = dict(rows=int(risk.numel()), risky_fraction=float(risky.float().mean()),
                  episode_end_fraction=float(ended.float().mean()))
    for name, objective in objectives.items():
        g = torch.autograd.grad(objective.sum(), raws, retain_graph=True)
        n2 = sum(x.square().sum(-1) for x in g)
        ordered = n2.sort(descending=True).values
        total = float(n2.sum())
        k1, k5 = max(1, n2.numel() // 100), max(1, n2.numel() // 20)
        n = n2.sqrt()
        report[name] = dict(
            median_row_norm=float(n.median()), p99_row_norm=float(n.quantile(0.99)), max_row_norm=float(n.max()),
            top1pct_share=float(ordered[:k1].sum()) / max(total, 1e-30),
            top5pct_share=float(ordered[:k5].sum()) / max(total, 1e-30),
            risky_rows_share=float(n2[risky].sum()) / max(total, 1e-30),
        )
    return report


CLIP_VARIANTS = (("none", None, None), ("action_x4", 4.0, None), ("action_x16", 16.0, None),
                 ("critic_x1+action_x4", 4.0, 1.0), ("critic_x2+action_x4", 4.0, 2.0),
                 ("critic_x4+action_x4", 4.0, 4.0), ("critic_x8+action_x4", 4.0, 8.0),
                 ("critic_x4+action_x16", 16.0, 4.0))


def window_at(replay, t0, envs, window):
    """``TapeReplay.sample`` layout for fixed tape start ``t0`` (a tape filled from step 0)."""
    rows = t0 + torch.arange(window, device=envs.device)
    out = {k: v[rows[None, :], envs[:, None]] for k, v in replay.data.items()}
    done = out["done"].float()
    out["mask"] = torch.cumprod(torch.cat((torch.ones_like(done[:, :1]), 1 - done[:, :-1]), 1), 1)
    out["env_ids"] = envs
    return out


@torch.no_grad()
def reset_report(learner, replay, H, starts=(0, 5, 10, 20, 100, 115, 127)):
    """Do imagination and the world model see the falls right after a reset?

    The diagnostic tape starts with every environment freshly reset, so
    windows at small ``t0`` begin right after a reset. Per start: the real
    fall fraction within the window, the imagined fall probability of the
    current actor, and the world model's 8-step error on recorded controls.
    """
    out = {}
    N = replay.N
    envs = torch.arange(N, device=learner.device)
    for t0 in starts:
        if t0 + H + 1 > replay.size:
            continue
        batch = window_at(replay, t0, envs, H + 1)
        real_fall = (batch["terminal"][:, :H].float() * batch["mask"][:, :H]).amax(1)
        _, _, extra, _ = _roll(learner, batch, H)
        wm = learner.validate(window_at(replay, t0, envs, 8))
        out[str(t0)] = dict(
            real_fall_within_window=float(real_fall.mean()),
            imagined_fall_probability=float(1 - extra["survived"].mean()),
            imagined_fall_probability_on_real_falls=float((1 - extra["survived"])[real_fall > 0].mean())
            if bool((real_fall > 0).any()) else None,
            imagined_fall_probability_on_real_survivors=float((1 - extra["survived"])[real_fall == 0].mean()),
            **{k.split("/")[-1]: v for k, v in wm.items()},
        )
        print(f"[diagnose] reset window t0={t0}: " + json.dumps(out[str(t0)]), flush=True)
    return out


def value_path_report(learner, batches, lengths=(1, 2, 4, 8, 16, 32), lambdas=(0.8, 0.9, 0.95)):
    """Critic bootstrap gradient versus path length through the world model.

    For each t the component is gamma^t alive_t mask_t V(s_t): its gradient reaches
    the actor through t imagined steps. ``aprev`` cuts the state path so only
    the last action feeds the critic. Batch-to-batch cosine near 1 at short t and
    decaying with t means the chain of Jacobians is chaotic; low cosine already at
    t=1 (or in ``aprev``) means the critic itself is jagged in its inputs.
    """
    from .learner import lambda_weights

    params = learner.actor_params
    H = batches[0]["s_end"].shape[1] - 1
    lengths = tuple(t for t in lengths if t <= H)
    per_batch = []
    for batch in batches:
        _, values, extra, _ = _roll(learner, batch, H, values_at=tuple(range(1, H + 1)))
        named = {}
        for t in lengths:
            named[f"V_t{t:02d}"] = values[t]
            named[f"V_t{t:02d}_aprev"] = extra["values_aprev"][t]
        for lam in lambdas:
            _, w_boot, w_end = lambda_weights(H, lam)
            named[f"V_lambda{lam:.2f}"] = sum(w_boot[t] * values[t] for t in range(1, H)) + w_end * values[H]
        grads = {}
        for name, comp in named.items():
            g = torch.autograd.grad(comp.mean(), params, retain_graph=True, allow_unused=True)
            grads[name] = torch.cat([(v if v is not None else torch.zeros_like(p)).flatten()
                                     for v, p in zip(g, params)])
        per_batch.append(grads)
        torch.cuda.empty_cache()
    out = {}
    for name in per_batch[0]:
        gs = [b[name] for b in per_batch]
        pairs = [(i, j) for i in range(len(gs)) for j in range(i + 1, len(gs))]
        out[name] = dict(
            grad_norm=sum(float(g.norm()) for g in gs) / len(gs),
            batch_to_batch_cosine=sum(float(F.cosine_similarity(gs[i], gs[j], dim=0)) for i, j in pairs) / len(pairs),
        )
    for name, row in out.items():
        print(f"[diagnose] value_path {name:16s} |g|={row['grad_norm']:.3e} batch_cos={row['batch_to_batch_cosine']:+.3f}",
              flush=True)
    return out


def signal_report(learner, batches):
    """Per objective component: gradient norm share and batch-to-batch cosine.

    A component whose gradient agrees across independent batches carries
    signal; one that does not is noise at the current batch size.
    """
    params = learner.actor_params
    m = learner.model
    per_batch = []
    for batch in batches:
        terms, out, noise, _ = components(learner, batch)
        named = {"reward/" + t: terms[:, i] for i, t in enumerate(m.terms)
                 if float(m.weights[i]) != 0 and i != m.termination_index}
        named.update(out)
        grads = {}
        for name, value in named.items():
            g = torch.autograd.grad(value.mean(), params, retain_graph=True, allow_unused=True)
            grads[name] = torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for v, p in zip(g, params)])
        grads["total"] = sum(grads.values())
        per_batch.append(grads)
        torch.cuda.empty_cache()
    out = {}
    for name in per_batch[0]:
        gs = [b[name] for b in per_batch]
        pairs = [(i, j) for i in range(len(gs)) for j in range(i + 1, len(gs))]
        out[name] = dict(
            grad_norm=sum(float(g.norm()) for g in gs) / len(gs),
            batch_to_batch_cosine=sum(float(F.cosine_similarity(gs[i], gs[j], dim=0)) for i, j in pairs) / len(pairs),
            cosine_with_total=sum(float(F.cosine_similarity(b[name], b["total"], dim=0)) for b in per_batch) / len(per_batch),
        )
    out = dict(sorted(out.items(), key=lambda kv: -kv[1]["grad_norm"]))
    for name, row in out.items():
        print(f"[diagnose] signal {name:36s} |g|={row['grad_norm']:.3e} batch_cos={row['batch_to_batch_cosine']:+.3f} "
              f"cos_total={row['cosine_with_total']:+.3f}", flush=True)
    return out


@torch.no_grad()
def racket_wm_report(learner, replay, ks=(1, 4, 8, 16, 32), rows=1024):
    """World-model error of the racket state (FK of predicted vs real state) after k steps
    on recorded controls, overall and at strike-window steps."""
    m = learner.model
    keys = learner.ctx_keys
    out = {}
    for k in ks:
        batch = replay.sample(rows, k + 1)
        ids = batch["env_ids"]
        xi = learner._xi(ids)
        s = learner._start(batch, 0)
        for j in range(k):
            s_end = learner.wm.step(s, batch["qexec"][:, j], batch["qexec_prev"][:, j], batch["ctx/cmd_delay"][:, j], xi)
            if j + 1 < k:
                s = push(s_end, batch["push"][:, j + 1])
        t = k - 1
        ctx = context_at(batch, t, keys)
        a, qe, qp = batch["action"][:, t], batch["qexec"][:, t], batch["qexec_prev"][:, t]
        pred = m.frame(s_end, s_end, a, qe, qp, ctx, ids).racket
        real = m.frame(batch["s_end"][:, t], batch["s_end"][:, t], a, qe, qp, ctx, ids).racket
        valid = batch["mask"][:, t].bool()
        pos = (pred.pos - real.pos).norm(dim=-1)
        vel = (pred.lin_vel - real.lin_vel).norm(dim=-1)
        ang = torch.rad2deg(torch.acos((pred.normal * real.normal).sum(-1).clamp(-1, 1)))
        win = ctx["strike_window"].bool() & valid
        out[str(k)] = dict(
            racket_pos_m=float(pos[valid].mean()), racket_vel_mps=float(vel[valid].mean()),
            racket_normal_deg=float(ang[valid].mean()), strike_rows=int(win.sum()),
            strike_racket_pos_m=float(pos[win].mean()) if win.any() else None,
            strike_racket_vel_mps=float(vel[win].mean()) if win.any() else None,
            strike_real_vel_mps=float(real.lin_vel.norm(dim=-1)[win].mean()) if win.any() else None,
        )
        print(f"[diagnose] racket_wm k={k}: " + json.dumps(out[str(k)]), flush=True)
    return out


@torch.no_grad()
def coverage_report(replay, H, rows=2048):
    """How much of a sampled imagination window carries strike-related context."""
    batch = replay.sample(rows, H + 1)
    mask = batch["mask"][:, :H]
    win = batch["ctx/strike_window"][:, :H].float() * mask
    gate = batch["ctx/pos_gate"][:, :H].float() * mask
    hold = batch["ctx/in_hold"][:, :H].float() * mask
    tts = batch["ctx/tts"][:, 0]
    out = dict(
        windows_with_strike=float((win.sum(1) > 0).float().mean()),
        windows_with_pos_gate=float((gate.sum(1) > 0).float().mean()),
        rowsteps_strike=float(win.sum() / mask.sum()),
        rowsteps_pos_gate=float(gate.sum() / mask.sum()),
        rowsteps_hold=float(hold.sum() / mask.sum()),
        start_tts_quantiles=[float(q) for q in tts.quantile(torch.tensor([.1, .25, .5, .75, .9], device=tts.device))],
    )
    print("[diagnose] coverage " + json.dumps(out), flush=True)
    return out


def hold_pull_report(learner, batch, groups=("left_arm", "right_arm")):
    """Per objective component: does its gradient move each arm's actions toward HOLD?

    The gradient on a group's actions is projected on the direction that moves
    q_des toward the HOLD reference (sign of the current error and of the
    action scale), averaged over row-steps in HOLD and outside it. Positive
    pulls toward HOLD, negative pushes away; components sum to the total.
    """
    m = learner.model
    H = batch["s_end"].shape[1] - 1
    rows, values, extra, raws = _roll(learner, batch, H, values_at=(H,))
    objectives = {"reward/" + t: sum(r["terms"][:, i] for r in rows) for i, t in enumerate(m.terms)
                  if float(m.weights[i]) != 0 and i != m.termination_index}
    for key in ("termination", "projection", "bootstrap_episode_end"):
        objectives[key] = sum(r[key] for r in rows)
    objectives["bootstrap_horizon"] = values[H]
    ids = batch["env_ids"]
    sign = torch.sign(m.const["scale"][ids])
    in_hold = batch["ctx/in_hold"][:, :H].bool()
    directions = {}
    for g in groups:
        joints = m.groups[g]
        ref = m.hold_ref[[m.active_ids.index(j) for j in joints]]
        directions[g] = torch.stack([-torch.sign(S.part(x, "q")[:, joints] - ref) * sign[:, joints]
                                     for x in extra["states"]], 1)
    out = {}
    for name, objective in objectives.items():
        grads = torch.autograd.grad(objective.sum(), raws, retain_graph=True, allow_unused=True)
        G = torch.stack([g if g is not None else torch.zeros_like(raws[0]) for g in grads], 1)
        row = {}
        for g in groups:
            pull = (G[..., m.groups[g]] * directions[g]).sum(-1)
            row[g + "_in_hold"] = float((pull * in_hold).sum() / in_hold.sum().clamp_min(1))
            row[g + "_outside_hold"] = float((pull * ~in_hold).sum() / (~in_hold).sum().clamp_min(1))
        out[name] = row
    out["total"] = {k: sum(r[k] for r in out.values()) for k in next(iter(out.values()))}
    return dict(sorted(out.items(), key=lambda kv: kv[1][groups[0] + "_in_hold"]))


def clipping_report(learner, batches, variants=CLIP_VARIANTS):
    """Per clipping variant: batch-to-batch cosine of the actor gradient (signal
    to noise) and its cosine with the unclipped termination and reward gradients
    of the same batch (does the update still listen to falls and rewards?)."""
    params = learner.actor_params

    def flat(g):
        return torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for v, p in zip(g, params)])

    reference = []
    for batch in batches:
        H = batch["s_end"].shape[1] - 1
        rows, _, _, _ = _roll(learner, batch, H)
        termination = sum(r["termination"] for r in rows).mean()
        rewards = sum(r["terms"].sum(-1) + r["projection"] for r in rows).mean()
        reference.append(dict(
            termination=flat(torch.autograd.grad(termination, params, retain_graph=True, allow_unused=True)),
            rewards=flat(torch.autograd.grad(rewards, params, allow_unused=True)),
        ))
        torch.cuda.empty_cache()
    out = {}
    cos = lambda a, b: float(F.cosine_similarity(a, b, dim=0))
    for name, action_clip, critic_clip in variants:
        grads = []
        for batch in batches:
            H = batch["s_end"].shape[1] - 1
            rows, values, _, _ = _roll(learner, batch, H, values_at=(H,), clip=action_clip, critic_clip=critic_clip)
            grads.append(flat(torch.autograd.grad((_total(rows) + values[H]).mean(), params, allow_unused=True)))
            torch.cuda.empty_cache()
        pairs = [(i, j) for i in range(len(grads)) for j in range(i + 1, len(grads))]
        out[name] = dict(
            batch_to_batch_cosine=sum(cos(grads[i], grads[j]) for i, j in pairs) / len(pairs),
            cosine_with_termination=sum(cos(g, r["termination"]) for g, r in zip(grads, reference)) / len(grads),
            cosine_with_rewards=sum(cos(g, r["rewards"]) for g, r in zip(grads, reference)) / len(grads),
            grad_norm=sum(float(g.norm()) for g in grads) / len(grads),
        )
        print(f"[diagnose] clipping {name}: " + json.dumps(out[name]), flush=True)
    return out


def critic_gradient_report(learner, batch, H):
    """Does dV(s_H)/dtheta point where the model's continuation return does?

    ``batch`` holds 2H+1 steps. The continuation (rewards over H..2H-1 plus
    V(s_2H)) uses a frozen actor, so both gradients reach the parameters only
    through s_H and the survival up to H.
    """
    params = learner.actor_params
    rows, values, _, _ = _roll(learner, batch, 2 * H, frozen_from=H, values_at=(H, 2 * H))
    first = _total(rows[:H]).mean()
    boot = values[H].mean()
    continuation = (_total(rows[H:]) + values[2 * H]).mean()

    def grad(x):
        g = torch.autograd.grad(x, params, retain_graph=True, allow_unused=True)
        return torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for v, p in zip(g, params)])

    g_first, g_boot, g_cont = grad(first), grad(boot), grad(continuation)
    g_cont_rewards = grad(_total(rows[H:]).mean())
    cos = lambda a, b: float(F.cosine_similarity(a, b, dim=0))
    target = g_first + g_cont
    return dict(
        critic_grad_scale_sweep={str(b): cos(g_first + b * g_boot, target) for b in (0.0, 0.1, 0.25, 0.5, 1.0)},
        cosine_bootstrap_vs_continuation_rewards=cos(g_boot, g_cont_rewards),
        continuation_rewards_grad_norm=float(g_cont_rewards.norm()),
        horizon=H, rows=int(batch["env_ids"].shape[0]),
        bootstrap_value=float(boot.detach()), continuation_value=float(continuation.detach()),
        bootstrap_grad_norm=float(g_boot.norm()), continuation_grad_norm=float(g_cont.norm()),
        rewards_grad_norm=float(g_first.norm()),
        cosine_bootstrap_vs_continuation=cos(g_boot, g_cont),
        cosine_rewards_vs_continuation=cos(g_first, g_cont),
        cosine_actor_grad_with_critic_vs_with_continuation=cos(g_first + g_boot, g_first + g_cont),
    )


def gradient_report(learner, batch):
    m = learner.model
    params = learner.actor_params
    for p in learner.wm.parameters():
        p.requires_grad_(False)
    for p in learner.critic.parameters():
        p.requires_grad_(False)
    terms, out, noise, extra = components(learner, batch)

    def grad(x):
        g = torch.autograd.grad(x, params, retain_graph=True, allow_unused=True)
        return torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for v, p in zip(g, params)])

    named = {"reward/" + t: terms[:, i] for i, t in enumerate(m.terms) if float(m.weights[i]) != 0
             and i != m.termination_index}
    named.update({k: v for k, v in out.items()})
    total = sum(v.mean() for v in named.values()) - noise
    g_total = grad(total)
    norm2 = float(g_total.square().sum())
    rows = {}
    for name, value in list(named.items()) + [("sensor_noise_penalty", None)]:
        x = -noise if value is None else value.mean()
        g = grad(x)
        rows[name] = dict(
            objective=float(x.detach()),
            grad_norm=float(g.norm()),
            cosine_with_total=float(F.cosine_similarity(g, g_total, dim=0)),
            share_of_total=float((g * g_total).sum()) / max(norm2, 1e-30),
        )
    rows = dict(sorted(rows.items(), key=lambda kv: -abs(kv[1]["share_of_total"])))
    return dict(total_grad_norm=math.sqrt(norm2), total_objective=float(total.detach()), **extra, components=rows)


@torch.no_grad()
def critic_report(learner, replay, n=64):
    """Correlation and bias of V(s_t) against n-step real returns on the recorded tape."""
    m, d = learner.model, replay.data
    T = replay.size
    gamma = float(learner.cfg.gamma)
    keys = learner.ctx_keys
    values = []
    for t in range(T):
        s = push(d["s_cmd"][t], d["push"][t])
        ctx = {k: d["ctx/" + k][t] for k in keys}
        values.append(learner.value(m.critic_features(s, d["action_prev"][t], ctx)))
    V = torch.stack(values)
    r, done = d["reward"][:T], d["done"][:T].float()
    targets, preds = [], []
    for t in range(0, T - n):
        G = torch.zeros_like(V[0])
        live = torch.ones_like(G)
        for k in range(n):
            G = G + gamma**k * live * r[t + k]
            live = live * (1 - done[t + k])
        targets.append(G + gamma**n * live * V[t + n])
        preds.append(V[t])
    G, P = torch.stack(targets).flatten(), torch.stack(preds).flatten()
    return dict(
        n_step=n, samples=G.numel(), return_mean=float(G.mean()), return_std=float(G.std()),
        value_mean=float(P.mean()), bias=float((P - G).mean()), rmse=float((P - G).square().mean().sqrt()),
        correlation=float(torch.corrcoef(torch.stack((P, G)))[0, 1]),
    )


SECTIONS = ("gradient", "rows", "clipping", "critic", "hold_pull", "reset", "signal", "value_path", "racket_wm", "coverage")


def run(env, obs, cfg, out, backend, path, steps=160, rows=512, noise=0.1, sections=SECTIONS):
    from omegaconf import OmegaConf
    from ..trainer_analytic import build, resume

    sc = OmegaConf.merge(cfg.supermimic, {"compile": False})
    parts = build(env, obs, cfg, backend, sc)
    # Reward ramps follow the run's cumulative environment step counter.
    resume(path, parts.policy, parts.learner, env.device, env, sc)
    parts.policy.eval()
    replay = TapeReplay(steps, env.num_envs, env.device, 1.0 / env.num_envs)
    recorder = Recorder(env, backend, parts.model)
    recorder.observe(obs)
    try:
        report, obs = rollout(env, parts, recorder, obs, steps, noise, replay)
    finally:
        recorder.close()
    report["checkpoint"] = str(path)
    report["env_step_counter"] = env.common_step_counter
    report["critic"] = critic_report(parts.learner, replay)
    print("[diagnose] critic " + json.dumps(report["critic"]), flush=True)
    for phase, row in report["phases"].items():
        print(f"[diagnose] {phase}: fraction={row['fraction']:.3f} hold_rms={row['hold_error_group_rms_rad']} "
              f"stroke_rms={row['stroke_error_group_rms_rad']}", flush=True)
    for b, row in report["time_to_strike"].items():
        print(f"[diagnose] tts {b}: n={row['samples']:.0f} gate={row['recovery_gate']:.2f} "
              f"hold_rms={row['hold_error_group_rms_rad']}", flush=True)
    learner = parts.learner
    H = int(sc.horizon)
    torch.manual_seed(0)
    if "gradient" in sections:
        report["gradient"] = g = gradient_report(learner, replay.sample(rows, H + 1))
        torch.cuda.empty_cache()
        print(f"[diagnose] total |grad|={g['total_grad_norm']:.3e} objective={g['total_objective']:.3f} "
              f"survival={g['imagined_survival']:.3f} saturated={g['imagined_raw_saturated_fraction']:.4f}", flush=True)
        for name, row in list(g["components"].items())[:25]:
            print(f"[diagnose] {name:40s} share={row['share_of_total']:+.3f} cos={row['cosine_with_total']:+.3f} "
                  f"|g|={row['grad_norm']:.3e} J={row['objective']:+.4f}", flush=True)
    for p in learner.wm.parameters():
        p.requires_grad_(False)
    for p in learner.critic.parameters():
        p.requires_grad_(False)
    if "rows" in sections:
        report["row_gradient"] = [row_gradient_report(learner, replay.sample(rows, H + 1)) for _ in range(3)]
        for row in report["row_gradient"]:
            print("[diagnose] row gradient " + json.dumps(row), flush=True)
        torch.cuda.empty_cache()
    if "clipping" in sections:
        report["clipping"] = clipping_report(learner, [replay.sample(rows, H + 1) for _ in range(3)])
    if "coverage" in sections:
        report["coverage"] = coverage_report(replay, H)
    if "racket_wm" in sections:
        report["racket_wm"] = racket_wm_report(learner, replay)
    if "signal" in sections:
        report["signal"] = signal_report(learner, [replay.sample(rows, H + 1) for _ in range(3)])
        torch.cuda.empty_cache()
    if "value_path" in sections:
        report["value_path"] = value_path_report(learner, [replay.sample(rows, H + 1) for _ in range(3)])
        torch.cuda.empty_cache()
    if "reset" in sections:
        report["reset"] = reset_report(learner, replay, H)
        torch.cuda.empty_cache()
    if "hold_pull" in sections:
        report["hold_pull"] = [hold_pull_report(learner, replay.sample(rows, H + 1)) for _ in range(2)]
        for name in report["hold_pull"][0]:
            a, b = report["hold_pull"][0][name], report["hold_pull"][1][name]
            print(f"[diagnose] hold_pull {name:36s} " + " ".join(
                f"{k}={a[k]:+.2e}/{b[k]:+.2e}" for k in a), flush=True)
        torch.cuda.empty_cache()
    if "critic" in sections:
        report["critic_gradient"] = []
        for _ in range(2):
            report["critic_gradient"].append(critic_gradient_report(learner, replay.sample(rows // 2, 2 * H + 1), H))
            torch.cuda.empty_cache()
            print("[diagnose] critic gradient " + json.dumps(report["critic_gradient"][-1]), flush=True)
    (out / "diagnosis.json").write_text(json.dumps(_plain(report), indent=2))
    print("[diagnose] done", flush=True)


def _plain(value, path="report"):
    """JSON-ready copy of a report; tensors that reached it are converted and named."""
    if isinstance(value, dict):
        return {str(k): _plain(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if torch.is_tensor(value):
        print(f"[diagnose] tensor in report at {path} (shape {tuple(value.shape)})", flush=True)
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value
    return report
