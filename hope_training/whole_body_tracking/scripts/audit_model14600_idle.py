"""Isolate actor idle-observation contract from the official PD standing mode."""
import argparse,json,re
from pathlib import Path
import numpy as np
from mujoco_eval_recurrent import Plant,ObsTarget
from feedforward_eval_policy import FeedForwardEvalPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True,type=Path);p.add_argument('--professor-parameters',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    policy=FeedForwardEvalPolicy(a.bundle);meta=policy.session.get_modelmeta().custom_metadata_map
    boxes=lambda k:[list(map(float,b.split(','))) for b in meta[k].split(';')]
    pos=boxes('hitter_pure_pos_range_per_clip')[0];vel=boxes('hitter_pure_vel_range_per_clip')[0]
    hold_pos=np.array([(pos[0]+pos[1])/2,(pos[2]+pos[3])/2,(pos[4]+pos[5])/2]);hold_vel=np.array([(vel[0]+vel[1])/2,(vel[2]+vel[3])/2,(vel[4]+vel[5])/2])
    length=int(meta['clip_seg_lengths'].split(',')[0]);phase=float(meta['clip_strike_phases'].split(',')[0]);tts=np.floor(phase*(length-1)+.5)*policy.manifest['control_dt']
    source=a.professor_parameters.read_text()
    def array(name):
        body=re.search(r'\b'+name+r'\s*=\s*\{(.*?)\};',source,re.S).group(1);body=re.sub(r'//[^\n]*','',body)
        return list(map(float,re.findall(r'[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',body)))
    sdk=['waist_yaw_joint','waist_roll_joint','waist_pitch_joint','head_yaw_joint','head_pitch_joint','left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint','left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint','right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint']
    order=[n for n in sdk if not n.startswith('head_')];kp29=array('a3_pd_stand_kps');kd29=array('a3_pd_stand_kds');q29=array('a3_default_angles');assert len(kp29)==len(kd29)==len(q29)==29
    summary={}
    for case in ('legacy_negative_current_racket','positive_clock_current_racket','native_world_hold','official_pd_stand'):
        plant=Plant(policy.manifest,physical=True,ball_slots=1,table_impact_model='ppq004');policy.reset_episode();metrics=[];qposes=[];previous=plant.previous_qdes.copy()
        if case=='official_pd_stand':
            kp={**dict(zip(order,kp29)),'head_yaw_joint':40.,'head_pitch_joint':40.};kd={**dict(zip(order,kd29)),'head_yaw_joint':2.,'head_pitch_joint':2.}
            passive=list(map(float,meta['a3_passive_joint_damping'].split(',')));q={**dict(zip(order,q29)),'head_yaw_joint':0.,'head_pitch_joint':0.}
            np.testing.assert_allclose([q[n] for n in policy.manifest['joint_names']],plant.default,atol=1e-7,rtol=0)
            for i,(name,aid) in enumerate(zip(policy.manifest['joint_names'],plant.aids)):
                plant.model.actuator_gainprm[aid,0]=kp[name];plant.model.actuator_biasprm[aid,1]=-kp[name];plant.model.actuator_biasprm[aid,2]=-(kd[name]+passive[i])
        for step in range(1000):
            if case=='native_world_hold':target=ObsTarget(hold_pos,hold_vel,float(tts))
            else:target=ObsTarget(plant.data.site_xpos[plant.racket].copy(),np.zeros(3),-1. if case=='legacy_negative_current_racket' else float(tts))
            action=np.zeros(31,np.float32) if case=='official_pd_stand' else policy.infer(plant.observe(target))
            plant.step(action);m=plant.metrics();metrics.append({k:v for k,v in m.items() if k!='racket_pos'}|{'sim_t':plant.episode_time,'qdes_step_max_rad':float(np.max(abs(plant.previous_qdes-previous)))})
            previous=plant.previous_qdes.copy()
            if step%5==0:qposes.append(plant.data.qpos.copy())
            if m['base_z']<.8 or m['tilt_deg']>60:break
        result={'seconds':plant.episode_time,'fall':m['base_z']<.8 or m['tilt_deg']>60,'max_tilt_deg':max(x['tilt_deg'] for x in metrics),'max_drift_m':max(x['drift_m'] for x in metrics),'final_drift_m':metrics[-1]['drift_m'],'final_tilt_deg':metrics[-1]['tilt_deg'],'min_base_z':min(x['base_z'] for x in metrics),'late_qdes_step_max_rad':max(x['qdes_step_max_rad'] for x in metrics if x['sim_t']>=1.)}
        summary[case]=result;(a.output/(case+'.json')).write_text(json.dumps(metrics,indent=2)+'\n');np.savez(a.output/(case+'.npz'),qpos=np.array(qposes));print(case,json.dumps(result),flush=True)
    summary['native_hold_tuple']={'position_w':hold_pos.tolist(),'velocity_w':hold_vel.tolist(),'time_to_strike':float(tts),'source':'professor pp_policy.hpp box-center initialization and max_tts clamp'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
