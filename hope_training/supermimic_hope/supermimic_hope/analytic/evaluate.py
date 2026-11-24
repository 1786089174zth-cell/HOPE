"""Evaluate v4 checkpoints under identical fresh-reset conditions (no learning).

Every checkpoint starts from the same seeded reset of all environments and runs
with the collection noise, so differences come from the weights alone, not from
replay, critic or episode-age state of a training run. Statistics are binned by
episode age to show whether quality decays within an episode.
"""

from __future__ import annotations

import json
import torch

AGE_EDGES = (100, 250, 500, 1000, 1500)  # environment steps since reset
YAW_EDGES = (0.1, 0.2, 0.3)  # |initial yaw| bins (rad) for stand starts
TERMS = ("full_body_hold", "fixed_station_feet", "full_stroke_tracking", "racket_position",
         "racket_velocity", "racket_normal", "termination_penalty")


def run(env, obs, cfg, out, backend, paths, steps=1500, noise=0.1, seed=0, env_step_counter=57000):
    from omegaconf import OmegaConf
    from ..trainer_analytic import build, resume

    sc = OmegaConf.merge(cfg.supermimic, {"compile": False})
    parts = build(env, obs, cfg, backend, sc)
    model, robot = parts.model, env.scene["robot"]
    motion = env.command_manager.get_term("motion")
    left = [model.active_ids.index(j) for j in model.groups["left_arm"]]
    columns = [model.terms.index(t) for t in TERMS]
    nb = len(AGE_EDGES) + 1
    edges = (0,) + AGE_EDGES + (None,)
    tm = env.termination_manager
    reasons = [n for n in tm.active_terms if n not in ("time_out", "actual_q_hard_limit_telemetry")]
    report = {}
    for path in paths:
        payload = torch.load(path, map_location=env.device, weights_only=False)
        # Same reward ramps for every checkpoint (cumulative steps at iteration ~22.8k).
        resume(path, parts.policy, parts.learner, env.device)
        parts.policy.eval()
        env.common_step_counter = int(env_step_counter)
        torch.manual_seed(seed)
        obs, _ = env.reset()
        z = lambda *shape: torch.zeros(*shape, device=env.device)  # noqa: E731
        acc = dict(n=z(nb), reward=z(nb), terms=z(nb, len(TERMS)), falls=z(nb), hold_n=z(nb), hold_left_sq=z(nb),
                   reasons=z(nb, len(reasons)), stand_n=z(nb), stand_falls=z(nb),
                   yaw_starts=z(len(YAW_EDGES) + 1), yaw_early_falls=z(len(YAW_EDGES) + 1))

        def yaw():
            w, x, y, zq = robot.data.root_quat_w.unbind(-1)
            return torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))

        def standing():
            # Stand starts are written at the default pose with zero velocity.
            d = robot.data
            return (((d.joint_pos - d.default_joint_pos).abs().amax(-1) < 1e-4)
                    & (d.joint_vel.abs().amax(-1) < 1e-4))

        stand = standing()
        start_yaw = yaw().abs()
        acc["yaw_starts"] += torch.bincount(torch.bucketize(start_yaw[stand], start_yaw.new_tensor(YAW_EDGES)),
                                            minlength=len(YAW_EDGES) + 1).float()
        with torch.no_grad():
            for _ in range(steps):
                age = env.episode_length_buf.clone()
                a = (parts.actor(obs["policy"]) + noise * torch.randn(env.num_envs, 31, device=env.device)) * parts.active
                obs, r, terminated, truncated, _ = env.step(a)
                why = torch.stack([tm.get_term(n).bool() for n in reasons], -1)
                bins = torch.bucketize(age, age.new_tensor(AGE_EDGES), right=True)
                terms = env.reward_manager._step_reward[:, columns]
                # Post-step pose of environments that were not reset in this step.
                err = (robot.data.joint_pos[:, model.active_ids] - model.hold_ref)[:, left]
                hold = motion.in_hold & ~(terminated | truncated)
                for k in range(nb):
                    mk = bins == k
                    acc["n"][k] += mk.sum()
                    acc["reward"][k] += r[mk].sum()
                    acc["terms"][k] += terms[mk].sum(0)
                    acc["falls"][k] += terminated[mk].sum()
                    acc["reasons"][k] += why[mk].sum(0)
                    acc["stand_n"][k] += (mk & stand).sum()
                    acc["stand_falls"][k] += (terminated & mk & stand).sum()
                    hk = mk & hold
                    acc["hold_n"][k] += hk.sum()
                    acc["hold_left_sq"][k] += err[hk].square().mean(-1).sum()
                early = terminated & stand & (age < 250)
                acc["yaw_early_falls"] += torch.bincount(
                    torch.bucketize(start_yaw[early], start_yaw.new_tensor(YAW_EDGES)), minlength=len(YAW_EDGES) + 1).float()
                reset = terminated | truncated
                stand = torch.where(reset, standing(), stand)
                start_yaw = torch.where(reset, yaw().abs(), start_yaw)
                new = reset & stand
                acc["yaw_starts"] += torch.bincount(torch.bucketize(start_yaw[new], start_yaw.new_tensor(YAW_EDGES)),
                                                    minlength=len(YAW_EDGES) + 1).float()

        def summary(sel):
            n = float(acc["n"][sel].sum())
            hold_n = float(acc["hold_n"][sel].sum())
            row = dict(
                samples=n,
                reward_per_step=float(acc["reward"][sel].sum()) / max(n, 1),
                falls_per_step=float(acc["falls"][sel].sum()) / max(n, 1),
                hold_left_arm_rms_rad=(float(acc["hold_left_sq"][sel].sum()) / max(hold_n, 1)) ** 0.5,
            )
            row.update({t + "_per_second": float(acc["terms"][sel, i].sum()) / max(n, 1) for i, t in enumerate(TERMS)})
            row.update({"termination_" + t + "_per_step": float(acc["reasons"][sel, i].sum()) / max(n, 1)
                        for i, t in enumerate(reasons)})
            stand_n = float(acc["stand_n"][sel].sum())
            row["stand_start_fraction"] = stand_n / max(n, 1)
            row["stand_start_falls_per_step"] = float(acc["stand_falls"][sel].sum()) / max(stand_n, 1)
            row["other_start_falls_per_step"] = (float(acc["falls"][sel].sum() - acc["stand_falls"][sel].sum())
                                                 / max(n - stand_n, 1))
            return row

        cmd = env.command_manager.get_term("racket_target")
        strike = {k: float(v.float().mean()) for k, v in cmd.metrics.items()
                  if k.startswith("strike_") and k.endswith("_exact") and torch.is_tensor(v)}
        motion_keys = ("qdes_step_rms_rad_hold", "qdes_step_rms_rad_strike", "qdes_reversal_hz_hold",
                       "qdes_reversal_hz_strike", "executed_qdes_step_rms_rad", "complete_hold_joints_ratio",
                       "complete_hold_feet_ratio", "complete_hold_worst_ratio", "full_body_hold_pose_pass_fraction",
                       "complete_hold_settled_fraction")
        motion_quality = {k: float(cmd.metrics[k].float().mean()) for k in motion_keys
                          if k in cmd.metrics and torch.is_tensor(cmd.metrics[k])}
        yaw_edges = (0.0,) + YAW_EDGES + (None,)
        report[str(path)] = dict(
            strike=strike,
            motion_quality=motion_quality,
            stand_start_fall_within_5s_by_initial_yaw={
                f"[{yaw_edges[k]},{yaw_edges[k + 1] or 'max'})": dict(
                    starts=float(acc["yaw_starts"][k]),
                    fell=float(acc["yaw_early_falls"][k]) / max(float(acc["yaw_starts"][k]), 1.0))
                for k in range(len(YAW_EDGES) + 1)},
            iteration=int(payload["iter"]),
            overall=summary(slice(None)),
            by_episode_age={f"[{edges[k]},{edges[k + 1] or 'end'})": summary(slice(k, k + 1)) for k in range(nb)},
        )
        print(f"[evaluate]   motion_quality: {json.dumps(motion_quality)}", flush=True)
        o = report[str(path)]["overall"]
        print(f"[evaluate] {payload['iter']}: reward/step={o['reward_per_step']:.4f} falls/step={o['falls_per_step']:.5f} "
              f"hold_left={o['hold_left_arm_rms_rad']:.3f} hold={o['full_body_hold_per_second']:.3f} "
              f"feet={o['fixed_station_feet_per_second']:.3f} track={o['full_stroke_tracking_per_second']:.3f} "
              f"racket_v={o['racket_velocity_per_second']:.4f}", flush=True)
        print("[evaluate]   strike: " + json.dumps({k: round(v, 4) for k, v in strike.items()
                                                if k in ("strike_composite_success_exact", "strike_pos_pass_exact",
                                                         "strike_vel_pass_exact", "strike_normal_pass_exact")}), flush=True)
        print("[evaluate]   stand starts falling within 5 s by |initial yaw|: " + json.dumps(
            report[str(path)]["stand_start_fall_within_5s_by_initial_yaw"]), flush=True)
        for name, row in report[str(path)]["by_episode_age"].items():
            print(f"[evaluate]   age {name:12s} n={row['samples']:.0f} reward/step={row['reward_per_step']:.4f} "
                  f"falls/step={row['falls_per_step']:.5f} hold_left={row['hold_left_arm_rms_rad']:.3f} "
                  f"stand={row['stand_start_fraction']:.2f} falls stand/other={row['stand_start_falls_per_step']:.5f}/"
                  f"{row['other_start_falls_per_step']:.5f} "
                  + " ".join(f"{t}={row['termination_' + t + '_per_step']:.5f}" for t in reasons), flush=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report
