"""Export the standalone actor with the shared 110-D Gate3 ABI and tracked clips."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import torch
import yaml

work = Path(__file__).resolve().parent
entry = work.parent / "HOPE/hope_training/supermimic_hope"
sys.path.insert(0, str(entry))
from supermimic_hope import a3
from supermimic_hope.core.fk import Kinematics
from supermimic_hope.core.nets import Actor, OBS
from supermimic_hope.core.rotations import quat_to_matrix

checkpoint, out = (Path(x).resolve() for x in sys.argv[1:3])
torch.set_num_threads(2)
payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
cfg = payload["config"]
assets = entry.parent / "whole_body_tracking" / cfg["task"]["assets"]
manifest = json.loads((assets/"manifest.json").read_text())
metadata = json.loads((work/"baseline.metadata.json").read_text())
assert metadata["joint_names"].split(",") == list(a3.JOINTS) == manifest["joint_order"]
names = metadata["body_names"].split(",")
frames = cfg["frames"]
active = torch.ones(31)
active[[11,16]] = 0
actor = Actor(frames, tuple(cfg["actor_hidden"]), active, architecture=cfg["actor_architecture"])
actor.load_state_dict(payload["actor_ema"])
actor.eval()
model, _ = a3.build_model()
kin = Kinematics(model, a3.JOINTS)
body_ids = kin.index(names)


def load_clip(name):
    with np.load(assets / f"{name}.npz", allow_pickle=False) as archive:
        clip = {k: torch.from_numpy(archive[k]) for k in archive.files if k != "fps"}
    root_q = clip["body_quat_w"][:,0]
    root_w = clip["body_ang_vel_w"][:,0]
    local_w = (quat_to_matrix(root_q).transpose(-1,-2) @ root_w.unsqueeze(-1)).squeeze(-1)
    data = torch.cat((clip["body_pos_w"][:,0],root_q,clip["body_lin_vel_w"][:,0],
                      local_w,clip["joint_pos"],clip["joint_vel"]),dim=1)
    xyz, *_ = kin(data)
    distance = (xyz[:, body_ids, None] - clip["body_pos_w"][:,None]).square().sum(-1).mean(0).sqrt()
    indices = distance.argmin(1)
    errors = distance[torch.arange(len(names)),indices]
    if len(set(indices.tolist())) != len(names) or errors.max() > .05:
        raise ValueError(f"{name} body ordering disagrees with checkpoint plant: {indices}, {errors}")
    return {k: v[:,indices] if k.startswith("body_") else v for k,v in clip.items()}


clips = [load_clip(name) for name in ("stand_forehand","stand_backhand")]

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = actor
        for key in clips[0]:
            self.register_buffer(key,torch.cat([clip[key] for clip in clips]))
        self.last = len(self.joint_pos)-1

    def forward(self, obs, time_step):
        t = time_step.long().squeeze(-1).clamp(0,self.last)
        return (self.actor.flat_forward(obs),self.joint_pos[t],self.joint_vel[t],self.body_pos_w[t],
                self.body_quat_w[t],self.body_lin_vel_w[t],self.body_ang_vel_w[t])

(out/"exported").mkdir(parents=True,exist_ok=False)
(out/"params").mkdir()
path = out/"exported/policy.onnx"
outputs = ["actions","joint_pos","joint_vel","body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w"]
torch.onnx.export(Policy(),(torch.zeros(1,frames*OBS),torch.zeros(1,1)),str(path),
                  input_names=["obs","time_step"],output_names=outputs,opset_version=17,dynamo=False)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
csv = lambda values: ",".join(f"{float(v):.9g}" for v in values)
left, split, right = cfg["task"]["side_bounds"]
# Admission envelopes cover the stand and step families on each side.
with np.load(assets/"physical_tuple_bank.npz",allow_pickle=False) as bank, np.load(assets/"reference_adapter.npz",allow_pickle=False) as adapter:
    good = adapter["qualified"].astype(bool)
    family = adapter["family"][good]
    pos, vel = bank["target_position_offset"][good],bank["target_velocity"][good]
    boxes, velocities = [], []
    for index in range(2):
        mask = family % 2 == index
        p,v = pos[mask],vel[mask]
        bounds = (p[:,1].min(),split) if index == 0 else (split,p[:,1].max())
        boxes.append(csv((.56,.60,*bounds,p[:,2].min(),p[:,2].max())))
        velocities.append(csv(np.stack((v.min(0),v.max(0)),axis=1).ravel()))
for key in ("hitter_pure_station_y_step_range","hitter_pure_station_y_mixture",
            "hitter_pure_resolved_task_json","hitter_pure_resolved_task_sha256"):
    metadata.pop(key,None)
payload_sha = sha(checkpoint)
metadata.update(
    run_path=str(checkpoint.parent),
    hitter_pure_checkpoint_sha256=payload_sha,
    obs_frame_history=str(frames),
    observation_history_lengths=csv([frames]*len(metadata["observation_names"].split(","))),
    clip_seg_lengths=csv([len(clip["joint_pos"]) for clip in clips]),
    clip_strike_phases=csv([manifest["strike_frame"]/(manifest["frames"]-1)]*2),
    hitter_pure_station_contract="fixed_session_anchor_v1",
    hitter_pure_recovery_target_contract="fixed_session_anchor_v1",
    hitter_pure_hold_pose_contract="stand_backhand_first_frame_v15",
    hitter_pure_hold_reference_sha256=manifest["hold_sha256"],
    hitter_pure_motion_manifest_sha256=sha(assets/"manifest.json"),
    hitter_pure_base_target_range="0,0,0,0",
    hitter_pure_target_sample_contract="four_clip_ball_clock_v1",
    hitter_pingpong_recovery_reference_contract="old_ball_tail_until_arrival_v1",
    hitter_pure_swing_side_split_y=f"{split:.5f}",
    hitter_pure_pos_range_per_clip=";".join(boxes),
    hitter_pure_pos_support_per_clip=";".join(boxes),
    hitter_pure_position_support_contract="qualified_stroke_pool_envelope_v1",
    hitter_pure_vel_range_per_clip=";".join(velocities),
    hitter_pure_vel_core_range_per_clip=";".join(velocities),
    hitter_pure_vel_planner_range_per_clip=";".join(velocities),
    joint_stiffness=csv(a3.KP),
    joint_damping=csv(a3.KD),
    default_joint_pos=csv(a3.DEFAULT_Q),
    action_scale=csv(a3.ACTION_SCALE),
    qdes_safe_lower_rad=csv(a3.QDES_LO),
    qdes_safe_upper_rad=csv(a3.QDES_HI),
    supermimic_training_backend=payload["format"],
    supermimic_training_iteration=str(payload["iteration"]),
    supermimic_actor_weights="polyak_average",
    supermimic_actor_architecture=cfg["actor_architecture"],
    supermimic_task="standalone_four_stroke",
)
graph = onnx.load(path)
onnx.helper.set_model_props(graph, metadata)
onnx.checker.check_model(graph)
onnx.save(graph,path)
deploy = yaml.safe_load((work/"baseline-deploy.yaml").read_text())
sdk = deploy["joint_sdk_names"]
deploy["stiffness"] = [float(a3.KP[a3.JOINTS.index(n)]) for n in sdk]
deploy["damping"] = [float(a3.KD[a3.JOINTS.index(n)]) for n in sdk]
deploy["default_joint_pos"] = a3.DEFAULT_Q.tolist()
deploy["actions"]["JointPositionAction"].update(scale=a3.ACTION_SCALE.tolist(),
    offset=a3.DEFAULT_Q.tolist(),clip=np.stack((a3.QDES_LO,a3.QDES_HI),axis=1).tolist())
for observation in deploy["observations"]["policy"].values():
    observation["history_length"] = frames
deploy["contracts"]["actor_observation"].update(total_dim=OBS*frames,frame_dim=OBS,frame_history=frames)
deploy["provenance"] = dict(policy_sha256=sha(path), checkpoint_sha256=payload_sha, checkpoint=str(checkpoint),
    export_source="standalone reference clips and fork's native 110-D contract", actor_weights="polyak_average")
(out/"params/deploy.yaml").write_text(yaml.safe_dump(deploy,sort_keys=False))
print(out)
