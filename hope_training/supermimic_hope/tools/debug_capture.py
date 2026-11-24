import sys,time; sys.path.insert(0,'/home/dongc1/workspace/nohope/hope_training/supermimic_hope')
import mujoco, mujoco_warp as mjw, numpy as np, warp as wp
from supermimic_hope import a3
m,_=a3.build_model(); ix=a3.indices(m)
m.actuator_gainprm[:,0]*=4; m.actuator_biasprm[:,1]*=4; m.actuator_biasprm[:,2]*=2
d=mujoco.MjData(m); d.qpos[2]=1.06; d.qpos[ix['qpos']]=a3.DEFAULT_Q; d.ctrl[:]=a3.DEFAULT_Q; mujoco.mj_forward(m,d)
N=int(sys.argv[1]); mode=sys.argv[2]
mw=mjw.put_model(m); dw=mjw.put_data(m,d,nworld=N,nconmax=N*40,njmax=200)
def tick():
    for _ in range(4): mjw.step(mw,dw)
if mode=='capture':
    tick(); wp.synchronize()
    with wp.ScopedCapture() as cap: tick()
    run=lambda: wp.capture_launch(cap.graph)
else: run=tick
wp.synchronize();t=time.time()
for k in range(100): run()
wp.synchronize();dt=time.time()-t
z=dw.qpos.numpy()[:,2]
print(f'{mode} N={N}: z mean {z.mean():.3f} min {z.min():.3f} | {N*100/dt:,.0f} ticks/s | nacon {int(dw.nacon.numpy()[0])}',flush=True)
