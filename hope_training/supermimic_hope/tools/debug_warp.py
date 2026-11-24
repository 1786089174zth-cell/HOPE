import sys; sys.path.insert(0,'/home/dongc1/workspace/nohope/hope_training/supermimic_hope')
import mujoco, mujoco_warp as mjw, numpy as np, warp as wp
from supermimic_hope import a3
m,_=a3.build_model(); ix=a3.indices(m)
m.actuator_gainprm[:,0]*=4; m.actuator_biasprm[:,1]*=4; m.actuator_biasprm[:,2]*=2
d=mujoco.MjData(m); d.qpos[2]=1.06; d.qpos[ix['qpos']]=a3.DEFAULT_Q; d.ctrl[:]=a3.DEFAULT_Q; mujoco.mj_forward(m,d)
mw=mjw.put_model(m); dw=mjw.put_data(m,d,nworld=2,nconmax=64,njmax=256)
dc=mujoco.MjData(m); dc.qpos[:]=d.qpos; dc.ctrl[:]=d.ctrl; mujoco.mj_forward(m,dc)
print('opt', m.opt.integrator, m.opt.cone, m.opt.solver, m.opt.iterations, m.opt.ls_iterations, 'disableflags', m.opt.disableflags)
for k in range(400):
    mujoco.mj_step(m,dc); mjw.step(mw,dw)
    if k in (0,1,2,5,10,20,50,100,200,399):
        q=dw.qpos.numpy()[0]; 
        print(f'k={k} cpu z={dc.qpos[2]:.4f} warp z={q[2]:.4f} | qerr cpu {np.abs(dc.qpos[ix["qpos"]]-a3.DEFAULT_Q).max():.3f} warp {np.abs(q[ix["qpos"]]-a3.DEFAULT_Q).max():.3f} | ncon cpu {dc.ncon} warp {int(dw.nacon.numpy()[0])} | ctrl warp {dw.ctrl.numpy()[0][:3]} actfrc warp {np.abs(dw.actuator_force.numpy()[0]).max():.1f} cpu {np.abs(dc.actuator_force).max():.1f}')
