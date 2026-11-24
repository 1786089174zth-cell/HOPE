"""Build the Gate3 two-file bundle (exported/policy.onnx + params/deploy.yaml) of a ``train.py`` checkpoint.

Runs under Isaac (``hope_isaac_py``), because the deployment metadata is owned by the HOPE task
and its exporter: the task is instantiated once, the ONNX is written with the SuperMimic actor
and the task's baked reference clips, and ``attach_onnx_metadata`` stamps the same metadata a PPO
export of that task carries. Nothing is trained or simulated.

    cd hope_training/whole_body_tracking && source setup_train_env.sh
    hope_isaac_py ../supermimic_hope/tools/build_gate3_bundle.py \
        --checkpoint ../supermimic_hope/logs/supermimic/<run>/model_<N>.pt --output <new dir>

The actor takes ``frames`` consecutive 110-D frames as one [1, 110 * frames] input and the bundle
declares ``obs_frame_history``; the Runner must support that key (see docs/gate3.md).
"""
from pathlib import Path
import argparse, hashlib, json, os, sys

ROOT = Path(__file__).resolve().parents[1]
W = ROOT.parent / "whole_body_tracking"
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--task", default="HOPEPingPongMotionQualityDeployPlane")
p.add_argument("--device", default="cuda:1")
p.add_argument("--raw", action="store_true", help="export the raw iterate instead of the averaged actor")
p.add_argument("--strike-plane", type=float, default=None,
               help="strike plane ahead of the station in metres (0.58 = the deploy plane the bank is named after). The Runner "
                    "strikes at the centre of the exported strike box; this narrows the box to plane +-0.02 m. Default: the "
                    "Isaac task's own box (0.35-0.58, centre 0.465)")
p.add_argument("--lateral-reach", type=float, nargs=2, default=None, metavar=("RIGHT", "LEFT"),
               help="lateral limits of the contract in station axes, m: forehand from RIGHT, backhand up to LEFT (default the "
                    "Isaac task's, -0.92 and 0.46). With the step strokes trained for short notice: -0.92 0.64")
p.add_argument("--side-split", type=float, default=None,
               help="forehand | backhand boundary in station axes, m (default: the Isaac task's, -0.275). Must be the "
                    "split the checkpoint was trained with (task.side_bounds[1])")
p.add_argument("--command-clamp", action="store_true",
               help="clamp the stroke command inside the graph to what the policy was trained on: target height relative to "
                    "the pelvis and target (racket) velocity, in every frame of the history. The Runner's admission gate "
                    "checks the target's x and y only, and main's Planner passes a mis-fitted ball as it is: in the general "
                    "suite a command 3.64 m high with a racket velocity of 1.6 m/s downwards (0.21 s of notice) made every "
                    "checkpoint lunge 0.47 m with 13 deg of roll, and ft54 147000 fell on the next ball. Commands inside "
                    "the trained range pass unchanged")
p.add_argument("--waist-roll-limit", type=float, default=None,
               help="clamp the waist-roll command inside the graph to +- this many rad (trained q_des limit 0.314, hard "
                    "stop 0.349): the trunk overshoots the command by up to 0.047 rad in a far step forehand")
args = p.parse_args()
args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
sys.argv = sys.argv[:1]
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, device=args.device).app
success = False
try:
    import gymnasium as gym
    import numpy as np
    import onnx
    import torch
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from whole_body_tracking.utils.exporter import attach_onnx_metadata

    sys.path.insert(0, str(W / "scripts"))
    from train import _apply_task_overrides
    from play import _resolve_motion_sources

    sys.path.insert(0, str(ROOT))
    from supermimic_hope.core.nets import OBS, Actor

    with initialize_config_dir(version_base=None, config_dir=str(W / "cfg")):
        cfg = compose(config_name="train", overrides=[f"task={args.task}", "algo=ppo_motion_quality",
                                                      f"device={args.device}", "num_envs=4", "headless=true"])
    OmegaConf.resolve(cfg)
    ec = parse_env_cfg(str(cfg.task.gym_task), device=args.device, num_envs=4)
    _apply_task_overrides(ec, cfg, [])
    ec.commands.motion.motion_file = _resolve_motion_sources(cfg)
    raw = gym.make(str(cfg.task.gym_task), cfg=ec).unwrapped
    motion = raw.command_manager.get_term("motion").motion

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    frames = int(config["frames"])
    active = torch.ones(31)
    active[[11, 16]] = 0.0  # passive head joints
    actor = Actor(frames, tuple(config["actor_hidden"]), active)
    source = "actor" if args.raw or "actor_ema" not in payload else "actor_ema"
    actor.load_state_dict(payload[source])
    actor.eval()

    # Trained command range (qualified rows of the question bank v1: target 0.79-1.47 m high, velocity x 0.22-2.90,
    # y -1.00..1.21, z -0.59..2.00 m/s; pelvis 0.97-1.07 m high in Gate3) plus 3-5 cm / cm/s. Frame columns: 103-105
    # target relative to the pelvis, 106-108 target velocity (FourStrokeTask.frame).
    clamp_lo, clamp_hi = torch.full((OBS,), -1e9), torch.full((OBS,), 1e9)
    if args.command_clamp:
        clamp_lo[105], clamp_hi[105] = -0.31, 0.53
        clamp_lo[106:109] = torch.tensor([0.20, -1.05, -0.65])
        clamp_hi[106:109] = torch.tensor([3.00, 1.25, 2.05])
    clamp_lo, clamp_hi = clamp_lo.repeat(frames)[None], clamp_hi.repeat(frames)[None]

    # --waist-roll-limit: the action is (q_des - default) / scale
    act_lo, act_hi = torch.full((31,), -1e9), torch.full((31,), 1e9)
    if args.waist_roll_limit is not None:
        import types

        sys.modules.setdefault("mujoco", types.ModuleType("mujoco"))  # a3's tables only; Isaac's Python has no mujoco
        from supermimic_hope import a3

        j = list(a3.JOINTS).index("waist_roll_joint")
        act_lo[j] = (-args.waist_roll_limit - a3.DEFAULT_Q[j]) / a3.ACTION_SCALE[j]
        act_hi[j] = (args.waist_roll_limit - a3.DEFAULT_Q[j]) / a3.ACTION_SCALE[j]
    act_lo, act_hi = act_lo[None], act_hi[None]

    def bounded(obs):
        return torch.maximum(torch.minimum(actor.flat_forward(torch.maximum(torch.minimum(obs, clamp_hi), clamp_lo)), act_hi), act_lo)

    class Policy(torch.nn.Module):
        """Same outputs as the HOPE exporter: actions plus the task's reference clips by time_step."""

        def __init__(self):
            super().__init__()
            self.actor = actor
            for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                self.register_buffer(name, getattr(motion, name).detach().to("cpu").clone())
            self.last = self.joint_pos.shape[0] - 1
            self.register_buffer("clamp_lo", clamp_lo)
            self.register_buffer("clamp_hi", clamp_hi)
            self.register_buffer("act_lo", act_lo)
            self.register_buffer("act_hi", act_hi)

        def forward(self, obs, time_step):
            t = torch.clamp(time_step.long().squeeze(-1), max=self.last)
            if args.command_clamp:
                obs = torch.maximum(torch.minimum(obs, self.clamp_hi), self.clamp_lo)
            actions = self.actor.flat_forward(obs)
            if args.waist_roll_limit is not None:
                actions = torch.maximum(torch.minimum(actions, self.act_hi), self.act_lo)
            return (actions, self.joint_pos[t], self.joint_vel[t], self.body_pos_w[t],
                    self.body_quat_w[t], self.body_lin_vel_w[t], self.body_ang_vel_w[t])

    exported = args.output / "exported"
    exported.mkdir()
    target = exported / "policy.onnx"
    outputs = ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
    torch.onnx.export(Policy(), (torch.zeros(1, OBS * frames), torch.zeros(1, 1)), str(target),
                      input_names=["obs", "time_step"], output_names=outputs, opset_version=17, dynamo=False)
    task_identity = OmegaConf.to_container(cfg.task, resolve=True)
    attach_onnx_metadata(raw, args.checkpoint.parent.name, path=str(exported) + "/", filename="policy.onnx",
                         checkpoint_path=str(args.checkpoint), task_identity=task_identity)
    model = onnx.load(str(target))
    meta = {entry.key: entry.value for entry in model.metadata_props}
    meta.update(
        obs_frame_history=str(frames),  # N consecutive hitter_pure frames, oldest first, as one input
        supermimic_training_backend=str(payload["format"]), supermimic_training_iteration=str(payload["iteration"]),
        supermimic_actor_weights="polyak_average" if source == "actor_ema" else "raw_iterate",
        supermimic_task=args.task,
        supermimic_waist_roll_limit_rad="none" if args.waist_roll_limit is None else f"{args.waist_roll_limit:g}",
        supermimic_command_clamp="target_rel_z=-0.31,0.53;target_vel=0.20,3.00,-1.05,1.25,-0.65,2.05" if args.command_clamp else "none",
    )
    if args.strike_plane is not None:
        # The task's box puts the strike 3.5 cm short of the table edge, where every low ball is struck within
        # millimetres of the table (docs/rally_20261118.md, docs/returns_20261119.md).
        lo, hi = f"{args.strike_plane - 0.02:.4f}", f"{args.strike_plane + 0.02:.4f}"
        for key in ("hitter_pure_pos_range_per_clip", "hitter_pure_fallback_pos_range_per_clip", "hitter_pure_pos_support_per_clip"):
            if key in meta:
                meta[key] = ";".join(",".join([lo, hi] + clip.split(",")[2:]) for clip in meta[key].split(";"))
        meta["supermimic_strike_plane_m"] = f"{args.strike_plane:.4f}"
    if args.side_split is not None:
        # Forehand | backhand boundary of the contract (clip 0 forehand up to it, clip 1 backhand from it): Gate3's
        # fixed-home profile takes the Planner's swing_side_split_y from these ranges.
        y = f"{args.side_split:.4f}"
        for key in ("hitter_pure_pos_range_per_clip", "hitter_pure_fallback_pos_range_per_clip", "hitter_pure_pos_support_per_clip"):
            if key in meta:
                fh, bh = (clip.split(",") for clip in meta[key].split(";"))
                meta[key] = ";".join((",".join(fh[:3] + [y] + fh[4:]), ",".join(bh[:2] + [y] + bh[3:])))
        meta["hitter_pure_swing_side_split_y"] = f"{args.side_split:g}"
    if args.lateral_reach is not None:
        # How far right the forehand and how far left the backhand is offered (the Isaac task's: -0.92 and +0.46).
        right, left = (f"{v:.4f}" for v in args.lateral_reach)
        for key in ("hitter_pure_pos_range_per_clip", "hitter_pure_fallback_pos_range_per_clip", "hitter_pure_pos_support_per_clip"):
            if key in meta:
                fh, bh = (clip.split(",") for clip in meta[key].split(";"))
                meta[key] = ";".join((",".join(fh[:2] + [right] + fh[3:]), ",".join(bh[:3] + [left] + bh[4:])))
    onnx.helper.set_model_props(model, meta)
    onnx.save(model, str(target))

    # deploy.yaml: the robot-level tables come from the metadata just written, so the pair agrees.
    vec = lambda key: [float(v) for v in meta[key].split(",")]  # noqa: E731
    names = meta["joint_names"].split(",")
    sdk = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "head_yaw_joint", "head_pitch_joint",
           *[f"{side}_{j}_joint" for side in ("left", "right") for j in (
               "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")],
           *[f"{side}_{j}_joint" for side in ("left", "right") for j in (
               "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]]
    ids = [sdk.index(n) for n in names]
    kp, kd, q0 = vec("joint_stiffness"), vec("joint_damping"), vec("default_joint_pos")
    terms = meta["observation_names"].split(",")
    dims = [int(v) for v in meta["actor_obs_term_dims"].split(",")]
    sources = json.loads(meta["actor_obs_term_sources_json"])
    deploy = {
        "format": "unitree_rl_lab.deploy", "format_version": 1, "task_plugin": "a3_pingpong_hitter_pure",
        "joint_ids_map": ids, "joint_sdk_names": sdk, "step_dt": 0.02,
        "stiffness": [kp[names.index(n)] for n in sdk], "damping": [kd[names.index(n)] for n in sdk],
        "default_joint_pos": q0,
        "actions": {"JointPositionAction": {
            "joint_ids": None, "joint_names": names, "scale": vec("action_scale"), "offset": q0,
            "clip": [[lo, hi] for lo, hi in zip(vec("qdes_safe_lower_rad"), vec("qdes_safe_upper_rad"))]}},
        "observations": {"policy": {name: {"params": {}, "scale": [1.0] * dim, "clip": None, "history_length": frames,
                                           "dim": dim, "source": sources[name]} for name, dim in zip(terms, dims)}},
        "contracts": {
            "actor_observation": {"name": meta["actor_obs_contract"], "total_dim": sum(dims) * frames,
                                  "frame_dim": sum(dims), "frame_history": frames, "ordered_terms": terms},
            "action": {"name": meta["qdes_action_contract"], "total_dim": 31,
                       "hard_lower_rad": vec("qdes_hard_lower_rad"), "hard_upper_rad": vec("qdes_hard_upper_rad"),
                       "actual_q_hard_tolerance_rad": float(meta["qdes_actual_q_hard_tolerance_rad"]),
                       "policy_feedback": meta["qdes_policy_feedback_contract"]},
            "runtime": meta["hitter_pure_runtime_contract"],
            "training_recipe": {"name": meta["hitter_pure_training_recipe"],
                                "version": meta["hitter_pure_training_recipe_version"]},
        },
        "provenance": {
            "policy_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "checkpoint": str(args.checkpoint), "export_source": "supermimic_hope/tools/build_gate3_bundle.py",
            "training_backend": str(payload["format"]), "actor_weights": meta["supermimic_actor_weights"],
        },
    }
    (args.output / "params").mkdir()
    (args.output / "params/deploy.yaml").write_text(yaml.safe_dump(deploy, sort_keys=False))

    # A probe input and the actor's answer, so the graph can be checked wherever onnxruntime exists
    # (Isaac's Python has none): tools/check_gate3_bundle.py.
    x = np.random.default_rng(0).normal(size=(1, OBS * frames)).astype(np.float32)
    x[0, 105::OBS] *= 3.0  # some frames beyond the command clamp, when it is on
    np.savez(args.output / "actor_probe.npz", obs=x,
             actions=bounded(torch.from_numpy(x)).detach().numpy())
    print(f"GATE3_BUNDLE_WRITTEN {args.output} frames={frames} weights={source} "
          f"clips={meta.get('clip_seg_lengths')}", flush=True)
    success = True
except BaseException:
    import traceback

    traceback.print_exc()
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if success else 1)
