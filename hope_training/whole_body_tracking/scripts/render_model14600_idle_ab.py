"""Recorded 20-second idle A/B: identical actor/plant, invalid vs native wait input."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from mujoco_eval_recurrent import Plant


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True,type=Path);p.add_argument('--audit',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    m=json.loads((a.bundle/'manifest.json').read_text());plant=Plant(m,physical=True,ball_slots=1,table_impact_model='ppq004')
    left=np.load(a.audit/'legacy_negative_current_racket.npz')['qpos'];right=np.load(a.audit/'native_world_hold.npz')['qpos'];assert len(left)==len(right)==200
    mj=plant.mj;plant.model.vis.global_.offwidth=640;plant.model.vis.global_.offheight=480
    renderer=mj.Renderer(plant.model,width=640,height=480);renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION]=False;renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=False
    camera=mj.MjvCamera();camera.lookat[:]=[.1,0.,1.];camera.distance=3.8;camera.azimuth=130;camera.elevation=-18
    vf="drawbox=x=0:y=0:w=iw:h=56:color=black:t=fill,drawtext=text='14600 SAME ACTOR - OLD INVALID IDLE':x=10:y=8:fontsize=18:fontcolor=white,drawtext=text='tts=-1 - current racket target':x=10:y=31:fontsize=16:fontcolor=white,drawtext=text='14600 SAME ACTOR - CORRECTED NATIVE HOLD':x=650:y=8:fontsize=18:fontcolor=white,drawtext=text='tts=+0.82 - trained fixed-world target - ACTOR ACTIVE':x=650:y=31:fontsize=15:fontcolor=white,drawtext=text='Max drift 53.2 cm':x=10:y=450:fontsize=18:fontcolor=white:box=1:boxcolor=black,drawtext=text='Max drift 1.55 cm - no PD override':x=650:y=450:fontsize=18:fontcolor=white:box=1:boxcolor=black"
    ff=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','10','-i','-','-vf',vf,'-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(a.output)],stdin=subprocess.PIPE)
    try:
        for old,new in zip(left,right):
            images=[]
            for qpos in (old,new):
                plant.data.qpos[:]=qpos;mj.mj_forward(plant.model,plant.data);renderer.update_scene(plant.data,camera=camera);images.append(renderer.render())
            ff.stdin.write(np.concatenate(images,axis=1).tobytes())
    finally:
        ff.stdin.close();rc=ff.wait(timeout=30);renderer.close()
    if rc:raise RuntimeError('A/B encoding failed')
    subprocess.run(['ffmpeg','-v','error','-i',str(a.output),'-f','null','-'],check=True);print(str(a.output),flush=True)


if __name__=='__main__':main()
