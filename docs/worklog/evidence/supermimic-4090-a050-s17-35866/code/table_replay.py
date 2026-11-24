"""Simulate a Gate3 ball against recorded policy poses and render contact evidence."""
import json
import subprocess
import sys
from pathlib import Path

from replay_behavior import ENTRY, mujoco, np, Image, ImageDraw, ImageFont
from supermimic_hope import a3
from supermimic_hope.core.task import paddle_contact
import torch

DT = 0.002
EDGE = 0.58
TABLE_Y = -0.34
TABLE_Z = 0.76
RADIUS = 0.020


def scene():
    spec = mujoco.MjSpec.from_file(str(a3.GATE3_XML))
    pairs = {"gate3_table_collision", "gate3_net_collision", "right_racket_collision"}
    for pair in list(spec.pairs):
        if {pair.geomname1, pair.geomname2}.isdisjoint(pairs) or "gate3_ball_collision" not in (pair.geomname1, pair.geomname2):
            spec.delete(pair)
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for body in list(spec.bodies):
        if body.name.startswith("gate3_ball_return"):
            spec.delete(body)
    for geom in spec.geoms:
        if geom.name in ("gate3_table_visual", "gate3_table_collision", "gate3_net_collision"):
            geom.pos[0] += EDGE
            geom.pos[1] = TABLE_Y
            if geom.name == "gate3_table_visual":
                geom.rgba[3] = 1
        if geom.name == "floor":
            geom.conaffinity = 2
        elif geom.name != "gate3_ball_collision":
            geom.contype = geom.conaffinity = 0
    for xpos, ypos, sx, sy in (
        (EDGE+1.37, TABLE_Y-0.7525, 1.37, .008), (EDGE+1.37, TABLE_Y+0.7525, 1.37, .008),
        (EDGE+.008, TABLE_Y, .008, .7625), (EDGE+2.732, TABLE_Y, .008, .7625),
        (EDGE+1.37, TABLE_Y, 1.37, .004)
    ):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[xpos, ypos, TABLE_Z+.001],
                                size=[sx, sy, .001], rgba=[.9,.94,.97,1])
    for x in (EDGE+.26, EDGE+2.48):
        for y in (TABLE_Y-.66, TABLE_Y+.66):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x,y,.365],
                                    size=[.025,.025,.365], rgba=[.18,.23,.30,1])
    spec.option.timestep = DT
    return spec.compile()


def fitted_bounce(v, spin):
    # Gate3's fitted slip/friction-cap impulse on a tabletop rising edge.
    slip = v[:2] + RADIUS * np.array([-spin[1], spin[0]])
    speed = np.linalg.norm(slip)
    restitution = .9313775047019153
    dv = -slip * min(.38744443855065164, .16706204822993975*(1+restitution)*abs(v[2])/max(speed, 1e-12))
    return np.r_[v[:2]+dv, -restitution*v[2]], spin + np.r_[dv[1], -dv[0], 0] / (2/3*RADIUS)


def simulate(model, states, frames, bank, strike, env, row):
    target = frames[strike, env, 10:13]
    dt_robot = .02
    launch_time = (strike + 1)*dt_robot - float(bank["time_to_strike"][row])
    first = max(0, launch_time - .22)
    end = (strike+1)*dt_robot + 1.45
    d = mujoco.MjData(model)
    qids = a3.indices(model)["qpos"]
    robot = states[:, env]
    adr = model.joint("gate3_ball_free_joint").qposadr[0]
    dof = model.joint("gate3_ball_free_joint").dofadr[0]
    ball_body = model.body("gate3_ball").id
    ball_geom = model.geom("gate3_ball_collision").id
    rz = model.site("right_racket").id
    objects = {model.geom(name).id: name for name in
               ("gate3_table_collision","gate3_net_collision","right_racket_collision","floor")}
    positions, poses, events = [], [], []
    prev = set()
    active, stopped = False, False
    q0, q1 = d.qpos.copy(), d.qpos.copy()
    velocity = np.zeros(model.nv)
    for t in np.arange(first, min(end, len(states)*dt_robot), DT):
        f = max(0., t/dt_robot-1)
        lo, hi = int(f), min(len(robot)-1, int(f)+1)
        for q, src in ((q0, robot[lo]), (q1, robot[hi])):
            q[:7], q[qids] = src[:7], src[13:44]
        # Interpolate with MuJoCo's tangent-space quaternion handling.
        mujoco.mj_differentiatePos(model, velocity, dt_robot, q0, q1)
        mujoco.mj_integratePos(model, q0, velocity, (f-lo)*dt_robot)
        d.qpos[:adr], d.qvel[:dof] = q0[:adr], velocity[:dof]
        poses.append(q0[:adr].copy())
        if not active and not stopped and t >= launch_time:
            initial = bank["physics_initial_position"][row].astype(float).copy()
            initial += target - bank["target_position_offset"][row]
            d.qpos[adr:adr+7] = np.r_[initial, [1,0,0,0]]
            d.qvel[dof:dof+3] = bank["physics_initial_velocity"][row]
            d.qvel[dof+3:dof+6] = bank["physics_initial_spin"][row]
            active = True
        if not active:
            positions.append(np.full(3, np.nan))
            continue
        v = d.qvel[dof:dof+3].copy()
        spin = np.zeros(6)
        mujoco.mj_fwdPosition(model, d)
        mujoco.mj_fwdVelocity(model, d)
        mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, ball_body, spin, 0)
        spin = spin[:3]
        d.xfrc_applied[ball_body, :3] = model.body_mass[ball_body]*(
            -.12861635416914574*np.linalg.norm(v)*v + .00379260504480258*np.cross(spin,v))
        mujoco.mj_step(model, d)
        contacts = {}
        for contact in d.contact:
            g1,g2 = contact.geom
            other = g2 if g1 == ball_geom else g1 if g2 == ball_geom else -1
            if other in objects:
                contacts[objects[other]] = contact.pos.copy()
        for name in contacts.keys()-prev:
            if name == "gate3_table_collision" and v[2] < 0:
                new_v, new_spin = fitted_bounce(v, spin)
                d.qvel[dof:dof+3] = new_v
                d.qvel[dof+3:dof+6] = d.xmat[ball_body].reshape(3,3).T @ new_spin
                d.qpos[adr+2] = max(d.qpos[adr+2], TABLE_Z+RADIUS+.0001)
            if name == "right_racket_collision":
                spatial = np.zeros(6)
                rid = model.geom(name).id
                mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_GEOM, rid, spatial, 0)
                paddle_v = spatial[3:] + np.cross(spatial[:3], contacts[name]-d.geom_xpos[rid])
                normal = d.site_xmat[rz].reshape(3,3)[:,1]
                oriented = normal if np.dot(v-paddle_v, normal) < 0 else -normal
                tensors = [torch.tensor(x, dtype=torch.float64) for x in (v,paddle_v,normal,spin)]
                new_v, new_spin = paddle_contact(*tensors, with_spin=True)
                d.qvel[dof:dof+3] = new_v.numpy()
                d.qvel[dof+3:dof+6] = d.xmat[ball_body].reshape(3,3).T @ new_spin.numpy()
                d.qpos[adr:adr+3] = contacts[name] + (RADIUS+.0001)*oriented
            if name == "gate3_table_collision":
                label = "Opponent-side table bounce" if contacts[name][0] > EDGE+1.37 else "Robot-side table bounce"
            else:
                label = {"right_racket_collision":"Racket contact","gate3_net_collision":"Net contact","floor":"Floor contact"}[name]
            events.append(dict(frame=len(positions), event=label, time=round(t,4), xyz=contacts[name].tolist()))
        prev = set(contacts)
        positions.append(d.qpos[adr:adr+3].copy())
        if d.qpos[adr+2] < -.08 or np.linalg.norm(d.qpos[adr:adr+2]) > 7:
            active, stopped = False, True
    return np.array(poses), np.array(positions), events, first


def collect(directory, keep_edge_bounces=False):
    with np.load(directory / "trajectory.npz", allow_pickle=False) as log:
        states, frames, families = [log[k] for k in ("states","frames","families")]
    bank_path = next(ENTRY.parent.joinpath("whole_body_tracking/assets").rglob("wide_stroke_v26_urdf2_plane058_20261110/physical_tuple_bank.npz"))
    keys = ("target_position_offset", "incoming_velocity", "incoming_spin", "physics_initial_position",
            "physics_initial_velocity", "physics_initial_spin", "time_to_strike", "physics_bounce_position",
            "physics_generated")
    with np.load(bank_path, allow_pickle=False) as source:
        bank = {k: source[k] for k in keys}
    model = scene()
    records, sequences = [], []
    for strike, env in np.argwhere(frames[:,:,2] > .5):
        row_data = frames[strike,env]
        # Identify the sampled question using its original position, approach velocity and spin.
        pos, vin, spin = row_data[13:16], row_data[16:19], row_data[19:22]
        same = (bank["target_position_offset"] == pos).all(axis=1)
        same &= (bank["incoming_velocity"] == vin).all(axis=1)
        candidates = np.flatnonzero(same & bank["physics_generated"])
        if not len(candidates):
            continue
        row = int(candidates[np.argmin(np.linalg.norm(bank["incoming_spin"][candidates]-spin,axis=1))])
        if not keep_edge_bounces and bank["physics_bounce_position"][row,0] < EDGE+.025:
            continue
        poses, ball, events, start_s = simulate(model, states, frames, bank, int(strike), int(env), row)
        labels = [e["event"] for e in events]
        info = dict(env=int(env), family=str(families[int(row_data[4])]), strike_s=round(float((strike+1)*.02),2),
                    bank_row=row, start_s=start_s, training_target_hit=bool(row_data[3]), events=events)
        records.append(info)
        sequences.append((poses, ball))
        print(f"{info['env']} {info['strike_s']} {info['family']} row {row} {labels}", flush=True)
    (directory / "contact_results.json").write_text(json.dumps(records, indent=2) + "\n")
    np.savez_compressed(directory / "physical_flights.npz",
                        **{f"{name}_{i}": seq[k] for i,seq in enumerate(sequences)
                           for k,name in enumerate(("poses","ball"))})


def render(directory):
    records = json.loads((directory / "contact_results.json").read_text())
    def kind(record):
        labels = {e["event"] for e in record["events"]}
        if "Opponent-side table bounce" in labels and "Racket contact" in labels:
            return record["family"]
        if "Net contact" in labels:
            return "net"
        if "Robot-side table bounce" in labels and "Racket contact" not in labels:
            return "miss"
        return None
    shown = {}
    for i, record in enumerate(records):
        result = kind(record)
        if result is not None and result not in shown:
            shown[result] = i
    selected = sorted(shown.values())
    model = scene()
    d = mujoco.MjData(model)
    address = model.joint("gate3_ball_free_joint").qposadr[0]
    width, height, fps, speed = 1280, 720, 60, .5
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (1.38, -.34, .83)
    camera.distance, camera.elevation, camera.azimuth = 5.75, -24, 108
    destination = directory / "supermimic_34500_table_ball_60fps.mp4"
    tmp = destination.with_name("rendering_60fps.mp4")
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "4", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    colors = {"Robot-side table bounce":(.20,.82,1,1), "Racket contact":(1,.78,.20,1),
              "Opponent-side table bounce":(.30,1,.45,1), "Net contact":(1,.28,.25,1)}
    with np.load(directory/"physical_flights.npz") as log, subprocess.Popen(args, stdin=subprocess.PIPE) as writer, mujoco.Renderer(model,height,width) as renderer:
        for n, i in enumerate(selected, 1):
            record, poses, ball = records[i], log[f"poses_{i}"], log[f"ball_{i}"]
            relevant = [e for e in record["events"] if e["event"] in colors]
            racket = next((e["frame"] for e in relevant if e["event"] == "Racket contact"), len(ball))
            for index in np.arange(0, len(ball), speed/fps/DT).astype(int):
                d.qpos[:address] = poses[index]
                d.qpos[address:address+3] = ball[index] if np.isfinite(ball[index]).all() else [100,0,-10]
                mujoco.mj_forward(model,d)
                renderer.update_scene(d,camera)
                scene_data = renderer.scene
                line = list(range(0,index,10)) + [index]
                for a,b in zip(line,line[1:]):
                    if not np.isfinite(ball[[a,b]]).all() or np.linalg.norm(ball[a]-ball[b]) < .002:
                        continue
                    g = scene_data.geoms[scene_data.ngeom]
                    rgba = (1,.57,.13,.90) if b>=racket else (.22,.78,1,.9)
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),
                                        np.eye(3).ravel(),np.array(rgba,dtype=np.float32))
                    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.005,ball[a],ball[b])
                    scene_data.ngeom += 1
                displayed = []
                for e in relevant:
                    if e["frame"] > index:
                        break
                    xyz = np.array(e["xyz"]) + [0,0,.021]
                    rgba = colors[e["event"]]
                    g = scene_data.geoms[scene_data.ngeom]
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.028,0,0]),xyz,
                                        np.eye(3).ravel(),np.array(rgba,dtype=np.float32))
                    scene_data.ngeom += 1
                    displayed.append(e["event"])
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0,0,width,68),fill=(12,20,29))
                draw.rectangle((0,height-48,width,height),fill=(12,20,29))
                family=record["family"].replace("_"," ").title()
                draw.text((20,14),f"Shot {n}/{len(selected)}  ·  {family}",font=bold,fill="white")
                draw.text((width-20,20),"Checkpoint 34,500  ·  60 fps  ·  ½ speed",font=font,fill=(206,225,244),anchor="ra")
                status = "  →  ".join(displayed) if displayed else "Incoming serve"
                if index*DT > (racket*DT + .15) and len(displayed) == 1:
                    status += "  →  No racket contact recorded"
                if index >= len(ball)-110 and racket == len(ball):
                    status += "  →  Miss"
                draw.text((20,height-35),status,font=font,fill="white")
                writer.stdin.write(image.tobytes())
            print(f"Rendered shot {n}/{len(selected)}: {family}",flush=True)
        writer.stdin.close()
        if writer.wait():
            raise RuntimeError("ffmpeg could not encode the replay")
    tmp.replace(destination)
    (directory/"video_shots.json").write_text(json.dumps([records[i] for i in selected],indent=2)+"\n")
    print(destination,flush=True)


if __name__ == "__main__":
    directory = Path(sys.argv[1]).resolve()
    collect(directory)
    render(directory)
