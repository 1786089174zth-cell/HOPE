"""Watch a checkpoint of ``train.py`` play the four-stroke rally (deterministic averaged actor).

    python play.py logs/supermimic/<run>/model_<N>.pt                 # writes play_<N>.mp4 next to it
    python play.py logs/supermimic/<run>/model_<N>.pt viewer=1        # interactive MuJoCo window

One environment is shown. It starts from the deploy stand, waits one to two seconds for the
first ball and then rallies. The sphere is the commanded contact point of the announced ball:
orange while the ball is on its way, green or red for half a second after the strike tick
(hit = position < 7.5 cm, velocity < 0.5 m/s, face normal < 15 deg, as in ``evaluate.py``).
The ball itself is not simulated. Every strike is printed.

Options: ``seconds`` (40), ``env`` (0; 0 has nominal physics, the others randomized ones),
``num_envs`` (4), ``seed`` (12345), ``device``, ``ema`` (1), ``out``, ``width``/``height``,
``viewer`` (0), ``camera`` (name of an MJCF camera; default a free camera following the robot).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import train  # noqa: F401 -- sets up sys.path for the package and the pinned dependencies
from evaluate import apply_physics, from_dict

ORANGE, GREEN, RED = (1.0, 0.55, 0.0, 1.0), (0.1, 0.85, 0.2, 1.0), (0.95, 0.1, 0.1, 1.0)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0])
    opts = dict(a.split("=", 1) for a in argv[1:])
    viewer = opts.get("viewer", "0") != "0"
    if not viewer and not os.environ.get("DISPLAY"):
        # No X display: render through EGL (needs PyOpenGL). With a display MuJoCo's default
        # GLFW backend renders into a hidden window. Must be decided before mujoco is imported.
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco
    import numpy as np
    import torch
    from supermimic_hope.core.task import FAMILIES, STRIKE_FRAME
    from supermimic_hope.core.trainer import TrainCfg, Trainer

    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(TrainCfg, payload["config"])
    cfg.device = opts.get("device", cfg.device)
    cfg.num_envs = int(opts.get("num_envs", 4))
    cfg.seed = int(opts.get("seed", 12345))
    cfg.buffer_steps, cfg.capture = 4, False
    # Deploy entry: every episode starts in the default stand and the first ball comes after 1-2 s.
    cfg.task.midstroke_probability = 0.0
    short = float(opts.get("short_notice", 0.0))  # late announcements, as Gate3 gives backhands
    cfg.task.short_notice_probability = (0.0, short, short, short)
    low = float(opts.get("low_ball", 0.0))  # share of questions from the low-ball rows (Gate3 is all low)
    cfg.task.low_probability = low
    cfg.task.low_on_plane = 0.0
    cfg.task.plane_shift = 0.0
    jitter = float(opts.get("jitter", 0.0))  # 1 = the training target jitter (wider ball paths)
    cfg.task.target_jitter_pos, cfg.task.target_jitter_vel = cfg.task.target_jitter_pos * jitter, cfg.task.target_jitter_vel * jitter
    cfg.task.stance_blend_probability = float(opts.get("stance_blend", 0.0))
    rush = float(opts.get("rush", 0.0))  # share of questions announced 0.24-0.46 s ahead (fast balls)
    cfg.task.rush_probability = rush
    high = float(opts.get("high_ball", 0.0))  # share of questions from the rows above 1.12 m
    cfg.task.high_probability = high
    edge = float(opts.get("edge", 0.0))  # share of questions pushed up to 6 cm across a side boundary
    cfg.task.edge_probability = edge
    apply_physics(cfg, opts)  # contact=pyramidal|elliptic, handoff=0|1 (see evaluate.py)
    cfg.task.stand_probability, cfg.task.long_hold_probability = 1.0, 0.0
    cfg.task.stand_hold_steps = (50, 100)
    env = int(opts.get("env", 0))
    seconds = float(opts.get("seconds", 40))
    trainer = Trainer(cfg)
    trainer.load(path)
    if opts.get("ema", "1") != "0" and "actor_ema" in payload:
        trainer.actor.load_state_dict(payload["actor_ema"])
    trainer.actor.eval()
    task, sim = trainer.task, trainer.sim
    task.ep_step.zero_()  # the initial reset staggers episode clocks; show one whole episode instead

    model = sim.model
    data = mujoco.MjData(model)
    qids = np.asarray(sim.ix["qpos"])

    def pose():
        s = sim.state()[env].cpu().numpy()
        data.qpos[:7] = s[:7]
        data.qpos[qids] = s[13:13 + len(qids)]
        mujoco.mj_forward(model, data)
        return s

    def marker(scene, pos, rgba, radius=0.02):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0.0]),
                            np.asarray(pos, dtype=np.float64), np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
        scene.ngeom += 1

    tally = dict(strikes=0, hits=0, result=None, shown=0)

    def step():
        """One control tick; returns (state, target position or None, marker colour, caption)."""
        trainer.collect(1, noise=0.0, learn=False)
        s = pose()
        tape = task.tape()
        _, info = task.reward(sim.state(), trainer.a_prev, trainer.a_prev, tape, detail=True)
        family = FAMILIES[int(task.family[env])]
        if int(task.k[env]) == STRIKE_FRAME and int(task.hold[env]) == 0:
            pos, vel = float(info["racket_pos_err"][env]), float(info["racket_vel_err"][env])
            normal = float(torch.rad2deg(torch.acos(info["racket_cos"][env].clamp(-1, 1))))
            hit = pos < 0.075 and vel < 0.5 and normal < 15.0
            tally["strikes"] += 1
            tally["hits"] += hit
            tally["result"], tally["shown"] = (GREEN if hit else RED), 25
            tally["last"] = f"{family}: {'HIT' if hit else 'MISS'}  pos {pos * 100:.1f} cm  vel {vel:.2f} m/s  normal {normal:.1f} deg"
            print(f"[play] t={trainer.steps * task.dt:6.2f}s  strike {tally['strikes']:2d}  {tally['last']}", flush=True)
        tts, in_hold = float(tape["tts"][env]), bool(tape["in_hold"][env] > 0.5)
        target, colour = None, ORANGE
        if tally["shown"] > 0:
            tally["shown"] -= 1
            target, colour = tape["tpos"][env].cpu().numpy(), tally["result"]
        elif not in_hold and tts > 0:
            target = tape["tpos"][env].cpu().numpy()
        clock = "waiting for a ball" if in_hold else (f"{family}  strike in {tts:4.2f} s" if tts > 0 else f"{family}  recovering")
        caption = [f"iteration {payload['iteration']}   t = {trainer.steps * task.dt:5.1f} s   hits {tally['hits']}/{tally['strikes']}",
                   clock, tally.get("last", "")]
        return s, target, colour, caption

    steps = int(round(seconds / task.dt))
    camera = mujoco.MjvCamera()
    if "camera" in opts:
        camera.type, camera.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, model.camera(opts["camera"]).id
    else:
        camera.distance, camera.azimuth, camera.elevation = 3.2, -130.0, -12.0

    def follow(s):
        if "camera" not in opts:
            camera.lookat[:] = (s[0] + 0.3, s[1], 0.95)

    if viewer:
        import mujoco.viewer

        s = pose()
        with mujoco.viewer.launch_passive(model, data) as window:
            window.cam.distance, window.cam.azimuth, window.cam.elevation = camera.distance, camera.azimuth, camera.elevation
            window.cam.lookat[:] = (s[0] + 0.3, s[1], 0.95)
            for _ in range(steps):
                if not window.is_running():
                    break
                tick = time.perf_counter()
                s, target, colour, _ = step()
                window.user_scn.ngeom = 0
                if target is not None:
                    marker(window.user_scn, target, colour)
                window.sync()
                time.sleep(max(0.0, task.dt - (time.perf_counter() - tick)))
    else:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont

        width, height = int(opts.get("width", 1280)), int(opts.get("height", 720))
        model.vis.global_.offwidth, model.vis.global_.offheight = width, height
        out = Path(opts["out"]) if "out" in opts else path.with_name(f"play_{payload['iteration']}.mp4")
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
        renderer = mujoco.Renderer(model, height, width)
        with imageio.get_writer(out, fps=round(1 / task.dt), codec="libx264", quality=8, macro_block_size=None) as video:
            for _ in range(steps):
                s, target, colour, caption = step()
                follow(s)
                renderer.update_scene(data, camera)
                if target is not None:
                    marker(renderer.scene, target, colour)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                for i, line in enumerate(caption):
                    draw.text((16, 12 + 28 * i), line, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
                video.append_data(np.asarray(image))
        renderer.close()
        print(f"[play] wrote {out}")
    print(f"[play] {tally['hits']}/{tally['strikes']} hits in {trainer.steps * task.dt:.0f} s (environment {env})")


if __name__ == "__main__":
    main()
