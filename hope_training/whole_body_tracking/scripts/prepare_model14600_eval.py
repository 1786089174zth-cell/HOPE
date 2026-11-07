"""Preserve model14600 weights/defaults; adapt only the benchmark bundle interface."""
import argparse,copy,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import mujoco as mj
import yaml
from mujoco_eval_recurrent import ROOT,Plant
from feedforward_eval_policy import FeedForwardEvalPolicy


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',required=True,type=Path);p.add_argument('--deploy',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    expected='a987240610bb9a49ea42da8eafcdc84c3cfbc8d5470bf7ff1a2d7f1f624cdd28'
    if sha(a.policy)!=expected:raise ValueError('Not the recorded Jiayi model14600')
    s=ort.InferenceSession(str(a.policy),providers=['CPUExecutionProvider']);meta=s.get_modelmeta().custom_metadata_map
    native=json.loads(meta['hitter_pure_resolved_task_json']);deploy=yaml.safe_load(a.deploy.read_text());act=deploy['actions']['JointPositionAction']
    names=meta['joint_names'].split(',');assert names==act['joint_names']
    floats=lambda k:list(map(float,meta[k].split(',')))
    sdk_index={n:i for i,n in enumerate(deploy['joint_sdk_names'])}
    np.testing.assert_allclose(floats('joint_stiffness'),[deploy['stiffness'][sdk_index[n]] for n in names],atol=1e-7,rtol=0)
    np.testing.assert_allclose(floats('a3_training_joint_damping'),np.array([deploy['damping'][sdk_index[n]] for n in names])+floats('a3_passive_joint_damping'),atol=1e-7,rtol=0)
    for key,field in [('default_joint_pos','offset'),('action_scale','scale')]:np.testing.assert_allclose(floats(key),act[field],atol=1e-7,rtol=0)
    np.testing.assert_allclose(floats('qdes_safe_lower_rad'),np.array(act['clip'])[:,0],atol=1e-7,rtol=0)
    np.testing.assert_allclose(floats('qdes_safe_upper_rad'),np.array(act['clip'])[:,1],atol=1e-7,rtol=0)
    graph=onnx.load(str(a.policy)).graph;producer={o:n for n in graph.node for o in n.output};pending=['actions'];seen=set();types=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        if name in producer:node=producer[name];types.add(node.op_type);pending.extend(node.input)
    assert 'time_step' not in seen and not types.intersection({'GRU','LSTM','RNN','Loop','Scan'}),'Actor must be stateless and independent of reference clock'
    baseline=json.loads((ROOT/'a3_deploy/a3_deploy_example/models/model_19000/policy/exported/manifest.json').read_text())
    model_path=ROOT/'a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml'
    model=mj.MjModel.from_xml_path(str(model_path));jids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n) for n in names];vadr=model.jnt_dofadr[jids]
    manifest={k:copy.deepcopy(baseline[k]) for k in ('visual_table','racket_site','observation_layout')}
    manifest.update(schema='hope_feedforward_mujoco_v1',architecture='feedforward',checkpoint_iteration=14600,checkpoint_sha256=meta['hitter_pure_checkpoint_sha256'],onnx_sha256=expected,
        joint_names=names,normalization='original_onnx_unchanged',root_default=[0.,0.,1.0684,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        physics_dt=native['sim']['dt'],decimation=native['sim']['decimation'],control_dt=deploy['step_dt'],
        action={'offset':act['offset'],'scale':act['scale'],'lower':np.array(act['clip'])[:,0].tolist(),'upper':np.array(act['clip'])[:,1].tolist(),
                'default_q':floats('default_joint_pos'),'default_qd':[0.]*31,'passive_columns':[names.index(n) for n in ('head_yaw_joint','head_pitch_joint')],
                'delay_min':int(meta['qdes_delay_min_steps']),'delay_max':int(meta['qdes_delay_max_steps']),'delay_nominal_fraction':float(meta['qdes_delay_nominal_fraction'])},
        actuator={'kp':floats('joint_stiffness'),'kd_total':floats('a3_training_joint_damping'),'effort_limit':floats('a3_joint_effort_limit'),
                  'armature':model.dof_armature[vadr].tolist(),'friction':model.dof_frictionloss[vadr].tolist()},
        task={'ball_physics':baseline['task']['ball_physics']},
        source={'repository':'dongc1/nohope','commit':'7990f18c2c1ae560c9e0708dca10228b9f2cc610','policy_path':'hope_training/whole_body_tracking/checkpoints/policy.onnx',
                'deploy_sha256':sha(a.deploy),'deploy_lf_sha256':hashlib.sha256(a.deploy.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),'native_initial_root':'agibot_a3.py init_state position (0,0,1.0684), identity rotation',
                'shared_plant':'identical MJCF Git blob ef0259258a5060a55f934910665025f2c8ad2015',
                'comparison_limit':'Native model14600 default pose/offset retained; other policy checkpoints have their own native pose/offset. Common arena, serve trajectories, command replay and outage contract.'})
    a.output.mkdir(parents=True,exist_ok=False);shutil.copy2(a.policy,a.output/'policy.onnx');shutil.copy2(a.deploy,a.output/'deploy.yaml');(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    policy=FeedForwardEvalPolicy(a.output);policy.reset_episode();plant=Plant(manifest,physical=True,ball_slots=9,table_impact_model='ppq004')
    # Independent shipped deploy wrapper reorders SDK observation/actions; compare
    # against direct actor-order inference to verify every joint column.
    sys.path.insert(0,str(ROOT/'a3_deploy/a3_deploy_example/reference'))
    from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy
    from a3_deploy_onnx_ref_pingpong.joint_order import JOINT_NAMES
    oracle=OnnxPolicy(a.policy);to_sdk=np.array([names.index(n) for n in JOINT_NAMES]);to_policy=np.array([JOINT_NAMES.index(n) for n in names])
    rng=np.random.default_rng(14600);max_error=0.;max_clock_error=0.
    for _ in range(64):
        obs=rng.normal(0,.15,110).astype(np.float32);sdk=obs.copy()
        for start in (3,34,65):sdk[start:start+31]=obs[start:start+31][to_sdk]
        action=policy.infer(obs);ref=oracle.infer(sdk)[to_policy];max_error=max(max_error,float(np.max(abs(action-ref))))
        clock=s.run(['actions'],{'obs':obs[None],'time_step':np.array([[100.]],np.float32)})[0][0];max_clock_error=max(max_clock_error,float(np.max(abs(action-clock))))
    assert max_error<2e-5 and max_clock_error==0.
    receipt={'result':'PASS','architecture':'feedforward','onnx_sha256':expected,'checkpoint_sha256':manifest['checkpoint_sha256'],
             'actor_observation_dim':110,'action_dim':31,'shipped_deploy_wrapper_max_abs_error':max_error,'reference_clock_action_max_abs_error':max_clock_error,
             'actor_depends_on_time_step':False,'actor_recurrent_nodes':False,'joint_map_bijective':len(set(plant.jids.tolist()))==31,
             'default_pose_source':'original deploy.yaml','gains_source':'original ONNX metadata: kp and total training damping; shared MJCF armature/friction',
             'source':manifest['source']}
    (a.output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
