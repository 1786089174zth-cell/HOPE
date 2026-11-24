"""Score a checkpoint of ``train.py`` on the four-stroke task with the deterministic actor.

    python evaluate.py logs/supermimic/<run>/model_<N>.pt device=cuda:1 num_envs=1024 steps=3000

Sensor noise stays on (as deployed); the collection action noise is off. Episodes start at
the default stand or at a ball announcement, never inside a stroke: a start placed on the
reference just before contact is an easy hit and would inflate the rate (by 0.05 at 1000 steps).
Prints the strike statistics overall and per family and writes ``evaluation_<N>.json`` next to
the checkpoint. Compare evaluations only when ``steps``, ``num_envs`` and ``seed`` agree.
``short_notice=1`` scores the late-announcement questions and ``low_ball=1`` the low-ball rows;
both together resemble Gate3's fixed-HOME serves (``evaluation_<N>_short_notice_low_ball.json``).
``contact=elliptic`` uses the non-creeping foot contact and ``handoff=1`` the Runner's 3 s entry
handoff on stand starts (both as trained from 2026-11-18 on, see docs/entry_20261118.md); the
defaults are the earlier pyramidal contact and no handoff, so older numbers stay comparable.
``rush=1`` announces every question 0.24-0.46 s ahead (Gate3's fast balls) and ``high_ball=1``
draws only questions above 1.12 m; both are off by default. ``plane=1`` scores only questions on the
0.58 m deploy plane, ``edge=1`` only those next to the forehand/backhand boundary, ``push=1`` adds the
training's random pushes (about one per 8 s per robot, up to 0.4 m/s).
``target_error=0.030,0.033,0.035`` tells the actor a strike that is off by that much (s, m y, m z at 0.40 s of
notice; the Planner's prediction error on the real sessions of 2026-11-21) while the strike is scored where the
ball is; ``refine=0.12,0.18`` replaces the error by a small one that long before the strike (TaskCfg.target_error).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import train  # noqa: F401 -- sets up sys.path for the package and the pinned dependencies


def from_dict(cls, data):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        default = getattr(cls(), f.name)
        if dataclasses.is_dataclass(default):
            value = from_dict(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def apply_physics(cfg, opts, contact="pyramidal", handoff=0.0):
    """The ``contact=`` and ``handoff=`` options shared by the evaluation tools; returns both."""
    contact = opts.get("contact", contact)
    if contact not in ("pyramidal", "elliptic"):
        raise SystemExit("contact must be pyramidal or elliptic")
    cfg.sim.cone, cfg.sim.impratio = ("elliptic", 10.0) if contact == "elliptic" else ("pyramidal", 1.0)
    # The friction range goes with the contact model, not with the checkpoint, so that checkpoints
    # trained before and after the change are scored on the same floors.
    cfg.sim.friction = (0.3, 1.6) if contact == "elliptic" else (0.3, 1.2)
    cfg.task.entry_handoff_probability = float(opts.get("handoff", handoff))
    # ``left_hand=bare`` scores with the MJCF's 0.28 kg left hand; the default is the gripper the robot carries
    # (0.847 kg), whatever the checkpoint was trained with.
    if opts.get("left_hand", "gripper") == "bare":
        cfg.sim.left_hand_mass = 0.0
    return contact, cfg.task.entry_handoff_probability


def main(argv=None):
    import torch
    from supermimic_hope.core.trainer import TrainCfg, Trainer

    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0])
    opts = dict(a.split("=", 1) for a in argv[1:])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(TrainCfg, payload["config"])
    cfg.device = opts.get("device", cfg.device)
    cfg.num_envs = int(opts.get("num_envs", 1024))
    cfg.seed = int(opts.get("seed", 12345))
    cfg.buffer_steps, cfg.capture = 4, False
    cfg.task.midstroke_probability = float(opts.get("midstroke", 0.0))
    # Late announcements are off unless asked for, so numbers stay comparable with earlier
    # evaluations; ``short_notice=1`` announces every backhand and step question 0.46 s..T ahead.
    short = float(opts.get("short_notice", 0.0))
    cfg.task.short_notice_probability = (0.0, short, short, short)
    low = float(opts.get("low_ball", 0.0))  # share of questions from the low-ball rows (Gate3 is all low)
    cfg.task.low_probability = low
    jitter = float(opts.get("jitter", 0.0))  # 1 = the training target jitter (wider ball paths)
    cfg.task.target_jitter_pos, cfg.task.target_jitter_vel = cfg.task.target_jitter_pos * jitter, cfg.task.target_jitter_vel * jitter
    cfg.task.stance_blend_probability = float(opts.get("stance_blend", 0.0))
    rush = float(opts.get("rush", 0.0))  # share of questions announced 0.24-0.46 s ahead (fast balls)
    cfg.task.rush_probability = rush
    high = float(opts.get("high_ball", 0.0))  # share of questions from the rows above 1.12 m
    cfg.task.high_probability = high
    edge = float(opts.get("edge", 0.0))  # share of questions pushed up to 6 cm across a side boundary
    cfg.task.edge_probability = edge
    # Mocap dropouts (base pose frozen for 0.06-1 s) at this rate per second; 0 unless asked for, like the pushes.
    cfg.task.mocap_dropout_rate = float(opts.get("mocap_dropout", 0.0))
    extra = float(opts.get("extra", 0.0))  # share of questions from the Supergate question file (scored as rush)
    cfg.task.extra_probability = extra
    cfg.task.extra_bank = opts.get("extra_bank", cfg.task.extra_bank or "assets/supergate_questions_s1.npz") if extra > 0 else ""
    # The Planner's prediction error (``target_error=0.030,0.033,0.035``: what the real sessions showed) and its
    # refinement before the strike (``refine=0.12,0.18``: seconds before the strike); both off unless asked for.
    cfg.task.target_error = tuple(float(x) for x in opts["target_error"].split(",")) if "target_error" in opts else (0.0, 0.0, 0.0)
    cfg.task.target_refine_tts = tuple(float(x) for x in opts["refine"].split(",")) if "refine" in opts else (0.0, 0.0)
    contact, handoff = apply_physics(cfg, opts)
    # The plane pool is pinned like the other question settings; ``plane=1`` scores only the deploy
    # plane the bank is named after (strike 0.555-0.58 m ahead of the station).
    plane = float(opts.get("plane", 0.0))
    cfg.task.plane_probability, cfg.task.plane_x = (plane, (0.555, 0.58)) if plane > 0 else (0.35, (0.50, 0.58))
    cfg.task.low_on_plane = 1.0 if plane > 0 else 0.0  # low balls stay where the bank has them unless the plane is asked for
    cfg.task.plane_shift = 0.0
    if "table_edge" in opts:  # how far the station is behind the table edge (0.50 Gate3 profile, 0.58 strike on the edge)
        cfg.task.table_edge_x = float(opts["table_edge"])
    steps = int(opts.get("steps", 3000))
    trainer = Trainer(cfg)
    trainer.load(path)
    # Score the averaged actor (what export.py ships) unless ``ema=0`` asks for the raw iterate.
    ema = opts.get("ema", "1") != "0" and "actor_ema" in payload
    if ema:
        trainer.actor.load_state_dict(payload["actor_ema"])
    trainer.actor.eval()
    noise = float(opts.get("noise", 0.0))  # action noise; 0 = the deployed deterministic actor
    trainer.push_eval = opts.get("push", "0") != "0"  # random pushes as in training (root velocity kicks)
    trainer.collect(200, noise=noise, learn=False)  # let the initial reset mix settle before scoring
    trainer.task.metrics()
    trainer.collect(steps, noise=noise, learn=False)
    result = {k: float(v) for k, v in trainer.task.metrics().items() if not k.startswith("reward/")}
    result.update(checkpoint=str(path), iteration=int(payload["iteration"]), num_envs=cfg.num_envs, steps=steps,
                  seed=cfg.seed, action_noise=noise, actor="ema" if ema else "raw",
                  midstroke_probability=cfg.task.midstroke_probability, short_notice=short, low_ball=low, jitter=jitter,
                  contact=contact, handoff=handoff, rush=rush, high_ball=high, edge=edge, plane=plane, push=float(trainer.push_eval), extra=extra, mocap_dropout=cfg.task.mocap_dropout_rate)
    name = (f"evaluation_{payload['iteration']}{'' if short == 0 else '_short_notice'}{'' if low == 0 else '_low_ball'}"
            f"{'' if jitter == 0 else '_jitter'}{'' if contact == 'pyramidal' else '_' + contact}"
            f"{'' if handoff == 0 else '_handoff'}{'' if rush == 0 else '_rush'}{'' if high == 0 else '_high_ball'}{'' if edge == 0 else '_edge'}{'' if extra == 0 else '_extra'}{'' if cfg.task.mocap_dropout_rate == 0 else '_mocap'}{'' if plane == 0 else '_plane'}{'_push' if trainer.push_eval else ''}.json")
    if "target_error" in opts:
        name = name.replace(".json", "_perr" + ("_refine" if "refine" in opts else "") + ".json")
        result.update(target_error=list(cfg.task.target_error), target_refine_tts=list(cfg.task.target_refine_tts))
    out = Path(opts["out"]) if "out" in opts else path.with_name(name)
    out.write_text(json.dumps(result, indent=2))
    for k, v in result.items():
        print(f"{k:40s} {v}")


if __name__ == "__main__":
    main()
