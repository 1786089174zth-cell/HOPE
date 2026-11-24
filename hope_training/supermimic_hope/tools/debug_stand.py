import sys; sys.path.insert(0,'/home/dongc1/workspace/nohope/hope_training/supermimic_hope')
import mujoco, numpy as np
from supermimic_hope import a3
m,_=a3.build_model(); ix=a3.indices(m); d=mujoco.MjData(m)
print('floor', [(m.geom(i).name, m.geom_contype[i], m.geom_conaffinity[i]) for i in range(m.ngeom) if m.geom(i).name=='floor'])
col=[i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
print('collision geoms',len(col),'types',np.unique(m.geom_type[col],return_counts=True))
print('actuator check', m.actuator_trnid[:3], m.actuator_gainprm[:2,:3], m.actuator_biasprm[:2,:3])
d.qpos[2]=1.06; d.qpos[ix['qpos']]=a3.DEFAULT_Q; d.ctrl[:]=a3.DEFAULT_Q
mujoco.mj_forward(m,d)
lf,rf=ix['feet']; print('pelvis z',d.qpos[2],'feet z', d.xpos[lf][2], d.xpos[rf][2], 'ncon',d.ncon)
# drop to contact
for k in range(800):
    mujoco.mj_step(m,d)
    if k%50==0: print(f't={k*0.005:.2f} z={d.qpos[2]:.3f} quat={np.round(d.qpos[3:7],3)} ncon={d.ncon} feetz={d.xpos[lf][2]:.3f},{d.xpos[rf][2]:.3f} maxqerr={np.abs(d.qpos[ix["qpos"]]-a3.DEFAULT_Q).max():.3f} act_frc_max={np.abs(d.actuator_force).max():.1f}')
