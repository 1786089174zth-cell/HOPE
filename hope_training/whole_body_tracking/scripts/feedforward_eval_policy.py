"""Original stateless 110-D ONNX actor, in its embedded training joint order."""
import hashlib,json
from pathlib import Path
import numpy as np
import onnxruntime as ort


class FeedForwardEvalPolicy:
    def __init__(self,bundle):
        self.bundle=Path(bundle);self.manifest=json.loads((self.bundle/'manifest.json').read_text())
        if self.manifest['schema']!='hope_feedforward_mujoco_v1':raise ValueError('Not a stateless evaluation bundle')
        path=self.bundle/'policy.onnx'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.manifest['onnx_sha256']:raise ValueError('ONNX SHA mismatch')
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])
        if {x.name for x in self.session.get_inputs()}!={'obs','time_step'}:raise ValueError('Expected original stateless actor inputs')
        if self.session.get_modelmeta().custom_metadata_map['joint_names'].split(',')!=self.manifest['joint_names']:raise ValueError('Actor joint order mismatch')
        meta=self.session.get_modelmeta().custom_metadata_map
        if (meta.get('hitter_pure_training_recipe')!='continuous_rally_sim2real_v3'
                or meta.get('hitter_pure_runtime_contract')!='rally_final_v2'):
            raise ValueError('Unsupported stateless native idle contract; do not reuse model14600 wait semantics for another recipe')
        boxes=lambda k:[np.array(list(map(float,b.split(',')))) for b in meta[k].split(';')]
        positions=boxes('hitter_pure_pos_range_per_clip');velocities=boxes('hitter_pure_vel_range_per_clip')
        lengths=list(map(int,meta['clip_seg_lengths'].split(',')));phases=list(map(float,meta['clip_strike_phases'].split(',')))
        self.idle_contract={}
        for i,side in enumerate(('forehand','backhand')):
            self.idle_contract[side]={'position_offset':((positions[i][::2]+positions[i][1::2])/2).tolist(),
                'velocity':((velocities[i][::2]+velocities[i][1::2])/2).tolist(),
                'time_to_strike':float(np.floor(phases[i]*(lengths[i]-1)+.5)*self.manifest['control_dt'])}
        self.reset_count=0;self.steps=0

    def idle_tuple(self,home_xy,side='forehand'):
        """Professor pp_policy.hpp: world HOME + trained box center, pinned windup clock.

        This is a policy-native hold, not a moving live-racket target and not the
        negative post-strike sentinel used by the separate GRU diagnostic.
        """
        c=self.idle_contract[side];position=np.array(c['position_offset']);position[:2]+=np.asarray(home_xy)[:2]
        return position,np.array(c['velocity']),c['time_to_strike']

    def reset_episode(self):
        # Accounting only. The original actor has no persistent hidden state.
        self.reset_count+=1

    def infer(self,obs):
        obs=np.asarray(obs,dtype=np.float32).reshape(1,110)
        if not np.isfinite(obs).all():raise FloatingPointError('Nonfinite observation')
        action=self.session.run(['actions'],{'obs':obs,'time_step':np.zeros((1,1),np.float32)})[0]
        if action.shape!=(1,31) or not np.isfinite(action).all():raise FloatingPointError('Invalid stateless action')
        self.steps+=1;return action[0]
