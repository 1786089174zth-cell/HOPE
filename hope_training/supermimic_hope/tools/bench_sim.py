"""Throughput of the A3 training model on MuJoCo-Warp (control steps/s, 4 x 5 ms per tick)."""
import sys, time
sys.path.insert(0, "/home/dongc1/workspace/nohope/hope_training/supermimic_hope")
import mujoco, mujoco_warp as mjw, numpy as np, torch, warp as wp
from supermimic_hope import a3
m, rec = a3.build_model()
m.actuator_gainprm[:,0]*=4; m.actuator_biasprm[:,1]*=4; m.actuator_biasprm[:,2]*=2  # benchmark only: stiff stand
print({k: v for k, v in rec.items() if not isinstance(v, list)})
ix = a3.indices(m)
d0 = mujoco.MjData(m)
d0.qpos[2] = 1.06; d0.qpos[ix["qpos"]] = a3.DEFAULT_Q; d0.ctrl[:] = a3.DEFAULT_Q
mujoco.mj_forward(m, d0)
for _ in range(400): mujoco.mj_step(m, d0)
print("CPU 2 s stand: base z %.3f tilt-free %s" % (d0.qpos[2], np.abs(d0.qpos[4:7]).max() < 0.05))
for N in [int(x) for x in sys.argv[1:]] or [1024, 4096, 8192]:
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, d0, nworld=N, nconmax=N * 40, njmax=200)
    def tick():
        for _ in range(4): mjw.step(mw, dw)
    tick(); wp.synchronize()
    with wp.ScopedCapture() as cap: tick()
    for _ in range(5): wp.capture_launch(cap.graph)
    wp.synchronize(); t = time.time(); n = 100
    for _ in range(n): wp.capture_launch(cap.graph)
    wp.synchronize(); dt = time.time() - t
    q = wp.to_torch(dw.qpos)
    print(f"N={N}: {N*n/dt:,.0f} control steps/s  ({dt/n*1e3:.1f} ms/tick)  base z mean {q[:,2].mean():.3f} min {q[:,2].min():.3f}  nacon max {int(wp.to_torch(dw.nacon).max())}", flush=True)
