"""Show one uninterrupted environment in MuJoCo."""
import json
from pathlib import Path
import sys
import time

from table_replay import DT, scene
from replay_behavior import mujoco, np
import mujoco.viewer


def main(directory):
    model = scene()
    data = mujoco.MjData(model)
    address = model.joint("gate3_ball_free_joint").qposadr[0]
    data.qpos[2] = .98
    playing, restart = True, False

    def keypress(key):
        nonlocal playing, restart
        if key == ord(" "):
            playing = not playing
        if key in (ord("R"), ord("r")):
            playing, restart = True, True

    def overlay(window, title, status):
        window.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                          mujoco.mjtGridPos.mjGRID_TOPLEFT, title, status))

    with mujoco.viewer.launch_passive(model, data, key_callback=keypress,
                                      show_left_ui=False, show_right_ui=False) as window:
        with window.lock():
            window.cam.lookat[:] = (1.38, -.34, .83)
            window.cam.distance, window.cam.elevation, window.cam.azimuth = 5.75, -24, 108
        (directory/"window-open").touch()
        while window.is_running() and not (directory/"ready").exists():
            overlay(window,"HOPE SuperMimic | Continuous episode","Preparing a single 80-second rollout")
            window.sync()
            time.sleep(.12)
        if not window.is_running():
            return
        with np.load(directory/"trajectory.npz") as log:
            states, frames, qids, families, dt = [log[k] for k in ("states","frames","qids","families","dt")]
        records = json.loads((directory/"contact_results.json").read_text())
        counts = [sum(row["env"] == env for row in records) for env in range(states.shape[1])]
        env = int(np.argmax(counts))
        rows = [(i,r) for i,r in enumerate(records) if r["env"] == env]
        with np.load(directory/"physical_flights.npz") as log:
            flights = [(r, log[f"ball_{i}"]) for i,r in rows]
        strikes = np.flatnonzero(frames[:,env,2])
        falls = np.flatnonzero(frames[:,env,0])
        captured = {r["strike_s"] for _,r in rows}
        (directory/"view-summary.json").write_text(json.dumps(dict(
            env=env, requested_strokes=len(strikes), physical_flights=len(flights),
            falls=[round(float((n+1)*dt),2) for n in falls],
            timeouts=int(frames[:,env,1].sum())),indent=2)+"\n")
        for remaining in range(5,0,-1):
            if not window.is_running():
                return
            overlay(window,"Continuous episode | Checkpoint 35,000",f"Starts in {remaining}s  |  Space: pause  |  R: replay")
            window.sync()
            time.sleep(1)

        clock = 0.
        duration = len(states)*dt
        while window.is_running():
            tick = time.monotonic()
            if restart:
                clock, restart = 0., False
            f = min(max(0., clock/dt-1),len(states)-1)
            i, j = int(f), min(int(f)+1,len(states)-1)
            a, b = states[i,env], states[j,env]
            if frames[i,env,:2].any():
                b = a
            pose = a+(b-a)*(f-i)
            pose[3:7] /= np.linalg.norm(pose[3:7])
            current = frames[i,env]
            balls = []
            event_names = []
            for r, ball in flights:
                index = int((clock-r["start_s"])/DT)
                if not 0 <= index < len(ball) or not np.isfinite(ball[index]).all():
                    continue
                if ball[index,2] > .04:
                    balls.append(ball[index])
                recent = [e["event"] for e in r["events"]
                          if 0 <= clock-e["time"] < .7 and e["event"] != "Floor contact"]
                event_names.extend(recent[-1:])
            before = np.searchsorted(strikes,i,side="right")
            recent_missing = (before and i-strikes[before-1]<30
                              and round(float((strikes[before-1]+1)*dt),2) not in captured)
            if any(0 <= i-fall < 35 for fall in falls):
                status = "Fall threshold reached | Simulator reset"
            elif recent_missing:
                status = "Commanded stroke | No physical launch for this source question"
            elif event_names:
                status = "  |  ".join(event_names)
            elif current[5]:
                status = "Ready stance"
            else:
                phase = "Preparing" if current[6] > 0 else "Following through / recovering"
                status = phase + "  |  " + ("Ball in scene" if balls else "No ball in scene")
            with window.lock():
                data.qpos[:7], data.qpos[qids] = pose[:7], pose[13:44]
                data.qpos[address:address+3] = balls[-1] if balls else [100,0,-10]
                mujoco.mj_forward(model,data)
                window.user_scn.ngeom = 0
                for ball in balls[:-1]:
                    geom = window.user_scn.geoms[window.user_scn.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.020,0,0]),
                                        np.asarray(ball,dtype=np.float64),np.eye(3).ravel(),
                                        np.array([1,.45,.05,1],dtype=np.float32))
                    window.user_scn.ngeom += 1
            if clock >= duration:
                overlay(window,"Continuous episode complete  |  80 s","Press R to start again")
                if not (directory/"played").exists():
                    (directory/"played").touch()
            else:
                family = str(families[int(current[4])]).replace("_"," ").title()
                overlay(window,f"Continuous episode  |  {clock:05.2f}s  |  Stroke {before}/{len(strikes)}  |  {family}",status)
            window.sync()
            time.sleep(max(0., 1/60-(time.monotonic()-tick)))
            if playing and clock < duration:
                clock = min(duration, clock+min(.08,time.monotonic()-tick))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
