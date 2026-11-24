"""Capture the trainer's pre-reset states and render them side by side in MuJoCo."""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ENTRY = Path(__file__).resolve().parents[1] / "HOPE/hope_training/supermimic_hope"
sys.path.insert(0, str(ENTRY))


def capture(checkpoint, seconds, output=None, continuous=False):
    output = output or checkpoint.parent
    import torch
    from evaluate import from_dict
    from supermimic_hope.core.task import FAMILIES, STRIKE_FRAME
    from supermimic_hope.core.trainer import Trainer, TrainCfg

    torch.set_num_threads(2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = from_dict(TrainCfg, payload["config"])
    cfg.num_envs, cfg.seed, cfg.buffer_steps, cfg.capture = 2, 12345, 4, False
    cfg.task.midstroke_probability = 0.0
    cfg.task.stand_probability, cfg.task.long_hold_probability = 1.0, 0.0
    cfg.task.stand_hold_steps = (50, 100)
    if continuous:
        cfg.task.episode_steps = int(seconds / .02) + 100
    trainer = Trainer(cfg)
    trainer.actor.load_state_dict(payload["actor_ema"])
    trainer.actor.eval()
    task = trainer.task
    task.ep_step.zero_()
    states, frames = [], []
    observe = task.observe

    def record(state, tape, reward, info, terminated, truncated):
        strike = (task.k == STRIKE_FRAME) & (task.hold == 0) & ~terminated
        pos, vel = info["racket_pos_err"], info["racket_vel_err"]
        angle = torch.rad2deg(torch.acos(info["racket_cos"].clamp(-1, 1)))
        hit = strike & (pos < 0.075) & (vel < 0.5) & (angle < 15)
        events = torch.stack((terminated, truncated, strike, hit, task.family,
                              tape["in_hold"], tape["tts"], pos, vel, angle), dim=1)
        states.append(state.cpu().numpy())
        frames.append(torch.cat((events, tape["tpos"], task.q_pos[task.row], tape["vin"], tape["spin"]), dim=1).cpu().numpy())
        observe(state, tape, reward, info, terminated, truncated)

    task.observe = record
    for step in range(round(seconds / task.dt)):
        trainer.collect(1, noise=0.0, learn=False)
        if step % 500 == 499:
            print(f"Captured {(step+1)*task.dt:.0f} simulated seconds", flush=True)

    frames = np.stack(frames)
    summary = dict(iteration=payload["iteration"], duration_s=seconds, seed=cfg.seed,
                   actor="ema", contact=cfg.sim.cone, action_noise=0, pushes=False, environments=[])
    for env, layout in enumerate(("Nominal body and gains", "Randomized body and gains")):
        events = frames[:, env]
        report = dict(environment=env, layout=layout, hits=int(events[:, 3].sum()),
                      strikes=int(events[:, 2].sum()),
                      falls_s=[round((i+1)*task.dt, 2) for i in np.flatnonzero(events[:, 0])],
                      timeouts_s=[round((i+1)*task.dt, 2) for i in np.flatnonzero(events[:, 1])],
                      strokes={str(name): dict(strikes=int(((events[:, 4] == i) & (events[:, 2] > 0)).sum()),
                                                hits=int(((events[:, 4] == i) & (events[:, 3] > 0)).sum()))
                               for i, name in enumerate(FAMILIES)})
        summary["environments"].append(report)
    np.savez_compressed(output / "trajectory.npz", states=states, frames=frames,
                        dt=task.dt, qids=trainer.sim.ix["qpos"], families=FAMILIES)
    (output / "behavior.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def render(directory):
    from supermimic_hope.a3 import build_model

    model, _ = build_model()
    data = mujoco.MjData(model)
    width, height, fps = 640, 384, 25
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.25, 0, 0.88)
    camera.distance, camera.azimuth, camera.elevation = 3.70, -135, -11
    title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    detail_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    with np.load(directory / "trajectory.npz", allow_pickle=False) as log:
        states, frames, qids, families, dt = (log[name] for name in ("states", "frames", "qids", "families", "dt"))
    summary = json.loads((directory / "behavior.json").read_text())
    counts = np.cumsum(frames[:, :, :4], axis=0)
    path = directory / f"supermimic_{summary['iteration']}_behavior.mp4"
    partial = path.with_name("rendering.mp4")
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{2*width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "3", "-preset", "fast",
            "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder, mujoco.Renderer(model, height, width) as renderer:
        for begin in range(0, len(states), 2):
            end = min(begin+1, len(states)-1)
            # Keep terminal poses even when they fall between the 25 fps display samples.
            indices = [next((i for i in range(begin, end+1) if frames[i, env, :2].any()), end)
                       for env in range(states.shape[1])]
            repeats = 13 if any(frames[i, env, 0] for env, i in enumerate(indices)) else 1
            screen = Image.new("RGB", (2*width, height))
            for env, i in enumerate(indices):
                s, event = states[i, env], frames[i, env]
                data.qpos[:7], data.qpos[qids] = s[:7], s[13:13 + len(qids)]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera)
                last = frames[max(0,i-22):i+1, env]
                strikes = np.flatnonzero(last[:, 2])
                if not event[5] and (event[6] > 0 or strikes.size):
                    strike = last[strikes[-1]] if strikes.size else None
                    colour = ((0.18, 0.9, 0.32, 1) if strike[3] else (1, 0.25, 0.18, 1)) if strike is not None else (1, 0.70, 0.13, 1)
                    scene = renderer.scene
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                        np.array([0.024, 0, 0]), np.asarray(event[10:13], dtype=np.float64),
                                        np.eye(3).flatten(), np.asarray(colour, dtype=np.float32))
                    scene.ngeom += 1
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, width, 52), fill=(19, 23, 28))
                draw.rectangle((0, height-36, width, height), fill=(19, 23, 28))
                label = summary["environments"][env]["layout"]
                draw.text((14, 7), label, font=title_font, fill="white")
                draw.text((14, 30), f"Checkpoint {summary['iteration']}   |   Sim time {(i+1)*dt:05.2f} s",
                          font=detail_font, fill=(205, 218, 227))
                falls, _, opportunities, hits = map(int, counts[i, env])
                family = str(families[int(event[4])]).replace("_", " ")
                status = "Ready stance" if event[5] else family
                colour = "white"
                if event[0]:
                    status, colour = "FALL THRESHOLD - reset follows", (255, 118, 106)
                elif event[1]:
                    status, colour = "Episode timeout - reset follows", (255, 215, 135)
                elif strikes.size:
                    status += " | " + ("HIT" if last[strikes[-1], 3] else "MISS")
                draw.text((14, height-26), status, font=detail_font, fill=colour)
                draw.text((width-15, height-26), f"Hits {hits}/{opportunities}   Falls {falls}",
                          font=detail_font, fill="white", anchor="ra")
                screen.paste(panel, (env*width, 0))
            packet = screen.tobytes()
            for _ in range(repeats):
                encoder.stdin.write(packet)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg encoding failed")
    partial.replace(path)
    print(path, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(Path(sys.argv[2]).resolve(), float(sys.argv[3]),
                Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None,
                "--continuous" in sys.argv[5:])
    else:
        render(Path(sys.argv[2]).resolve())
