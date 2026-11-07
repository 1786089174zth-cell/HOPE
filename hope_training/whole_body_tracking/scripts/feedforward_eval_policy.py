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
        self.reset_count=0;self.steps=0

    def reset_episode(self):
        # Accounting only. The original actor has no persistent hidden state.
        self.reset_count+=1

    def infer(self,obs):
        obs=np.asarray(obs,dtype=np.float32).reshape(1,110)
        if not np.isfinite(obs).all():raise FloatingPointError('Nonfinite observation')
        action=self.session.run(['actions'],{'obs':obs,'time_step':np.zeros((1,1),np.float32)})[0]
        if action.shape!=(1,31) or not np.isfinite(action).all():raise FloatingPointError('Invalid stateless action')
        self.steps+=1;return action[0]
