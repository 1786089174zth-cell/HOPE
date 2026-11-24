"""Compare every analytic head with the unmodified task on real Warp steps."""

from __future__ import annotations

import json
import torch

from . import state as S
from .recorder import Recorder
from .task import TaskModel


class Stats:
    def __init__(self):
        self.rows = {}

    def add(self, name, analytic, real, active=None):
        error = (analytic - real).abs()
        if error.ndim > 1:
            error = error.flatten(1).amax(-1)
        scale = real.abs()
        if scale.ndim > 1:
            scale = scale.flatten(1).amax(-1)
        active = (scale > 0) if active is None else active
        row = self.rows.setdefault(name, dict(max_abs=0.0, sum_abs=0.0, n=0, active=0, real_abs=0.0))
        row["max_abs"] = max(row["max_abs"], float(error.max()))
        row["sum_abs"] += float(error.sum())
        row["n"] += error.numel()
        row["active"] += int(active.sum())
        row["real_abs"] += float(scale.sum())

    def report(self):
        return {
            k: dict(
                max_abs_error=v["max_abs"],
                mean_abs_error=v["sum_abs"] / max(v["n"], 1),
                mean_abs_real=v["real_abs"] / max(v["n"], 1),
                active_fraction=v["active"] / max(v["n"], 1),
            )
            for k, v in self.rows.items()
        }


def run(env, backend, obs, steps, out, policy=None, noise=0.1, seed=0):
    from ..contracts import TerminationMargins, from_environment

    torch.manual_seed(seed)
    _, reasons, _, _ = from_environment(env)
    margins = TerminationMargins(env, reasons)
    model = TaskModel(env, backend)
    model.bind_terminations(reasons)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("racket_target")
    bodies = {}
    original = env.reward_manager.compute

    def capture_bodies(*args, **kwargs):
        d = robot.data
        bodies.update(
            pos=d.body_pos_w.clone(), quat=d.body_quat_w.clone(), ang=d.body_ang_vel_w.clone(),
            com_vel=d.body_lin_vel_w.clone(), link_vel=d.body_link_lin_vel_w.clone(),
            root_lin_vel=d.root_lin_vel_w.clone(), gravity=d.projected_gravity_b.clone(),
            racket_pos=cmd.racket_pos_w.clone(), racket_vel=cmd.racket_lin_vel_w.clone(),
            racket_normal=cmd.racket_normal_w.clone(),
        )
        return original(*args, **kwargs)

    env.reward_manager.compute = capture_bodies
    recorder = Recorder(env, backend, model)
    recorder.observe(obs)
    stats = Stats()
    action_dim = env.action_manager.total_action_dim
    # The command caches (racket state, clocks) exist only after the first command
    # update; the very first step after ``env.reset`` is not a task step.
    obs = env.step(torch.zeros(env.num_envs, action_dim, device=env.device))[0]
    recorder.observe(obs)
    try:
        for _ in range(int(steps)):
            with torch.no_grad():
                if policy is None:
                    a = torch.zeros(env.num_envs, action_dim, device=env.device)
                else:
                    a = policy(obs)
                a = a + noise * torch.randn_like(a)
                a[:, model.passive] = 0
            obs, _, terminated, truncated, _ = env.step(a)
            rec, real = recorder.record, recorder.real
            ctx = {k[4:]: v for k, v in rec.items() if k.startswith("ctx/")}
            with torch.no_grad():
                fr = model.frame(rec["s_cmd"], rec["s_end"], rec["action"], rec["qexec"], rec["qexec_prev"], ctx)
                values = model.rewards(fr) * model.weights
                for i, name in enumerate(model.terms):
                    if name in model.analytic:
                        stats.add("reward/" + name, values[:, i], real[:, i])
                    else:
                        stats.add("taped/" + name, torch.zeros_like(real[:, i]), real[:, i])
                stats.add("termination/margins", model.margins(fr, reasons, model.reason_params), margins.last)
                stats.add("fk/body_pos", fr.end.pos, bodies["pos"])
                stats.add("fk/body_rot", fr.end.rot, _rot(bodies["quat"]))
                stats.add("fk/body_ang_vel", fr.end.ang_vel, bodies["ang"])
                stats.add("fk/body_com_lin_vel", fr.end.com_lin_vel, bodies["com_vel"])
                stats.add("fk/body_link_lin_vel", fr.end.link_lin_vel, bodies["link_vel"])
                stats.add("fk/root_lin_vel", fr.root_lin_vel, bodies["root_lin_vel"])
                stats.add("fk/projected_gravity", fr.gravity_b, bodies["gravity"])
                for name, sl in model.obs_layout.items():
                    stats.add("obs_delta/" + name, rec["obs_delta"][:, sl], torch.zeros_like(rec["obs_delta"][:, sl]))
                # The analytic racket at s_cmd must equal the command cache used by rewards.
                stats.add("racket/pos_cmd", fr.racket.pos, bodies["racket_pos"])
                stats.add("racket/lin_vel_cmd", fr.racket.lin_vel, bodies["racket_vel"])
                stats.add("racket/normal_cmd", fr.racket.normal, bodies["racket_normal"])
            recorder.observe(obs)
    finally:
        recorder.close()
        env.reward_manager.compute = original
        margins.close()
    report = stats.report()
    (out / "analytic_check.json").write_text(json.dumps(report, indent=2))
    worst = sorted(report.items(), key=lambda kv: -kv[1]["max_abs_error"])
    for name, row in worst:
        print(f"[analytic-check] {name:45s} max={row['max_abs_error']:.3e} mean={row['mean_abs_error']:.3e} "
              f"|real|={row['mean_abs_real']:.3e} active={row['active_fraction']:.3f}", flush=True)
    return report


def _rot(quat):
    from .kinematics import quat_to_matrix

    return quat_to_matrix(quat)
