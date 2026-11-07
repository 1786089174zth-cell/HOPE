"""Check each incoming serve with the unchanged PPQ004 law, without actor/contact."""
import argparse,json
from pathlib import Path
import numpy as np
from mujoco_eval_recurrent import Plant,ROOT
from recurrent_physical_ball import PhysicalBall
from reference_gru_diagnostic import reference_ledger


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    manifest=json.loads((ROOT/'a3_deploy/a3_deploy_example/models/model_19000/policy/exported/manifest.json').read_text())
    plant=Plant(manifest,physical=True,ball_slots=1,table_impact_model='ppq004');ball=plant.ball_physics
    ledger,_=reference_ledger(a.scene);results=[]
    for serve in ledger:
        ball.park();plant.reset(0);plant.data.qpos[plant.root_q]=-10.;plant.mj.mj_forward(plant.model,plant.data)
        held=plant.data.qpos.copy();ball.current={'episode':0,'bank_row':None,'racket_contacts':0,'table_contacts':0,'net_contacts':0,'floor_contacts':0,'first_landing':None,'net_clear':False,'legal_return':False}
        ball.events=[];ball.contacts=set();ball.pending=(0.,np.array(serve['position_w']),np.array(serve['velocity_w']),np.zeros(3));ball.arrival_done=True
        crossing=None;error=None;terminal=False
        try:
            for step in range(8000):
                t=step*.001
                if step%20==0:
                    plant.data.qpos[plant.root_q:plant.root_q+7]=held[plant.root_q:plant.root_q+7]
                    plant.data.qpos[plant.qadr]=held[plant.qadr];plant.data.qvel[:6]=0.;plant.data.qvel[plant.vadr]=0.
                    plant.mj.mj_forward(plant.model,plant.data)
                ball.before(t);plant.mj.mj_step(plant.model,plant.data);ball.after(t+.001)
                pos,vel=ball.state()
                if crossing is None and ball.before_p[0]>.58>=pos[0]:
                    f=(.58-ball.before_p[0])/(pos[0]-ball.before_p[0]);crossing=(ball.before_p+f*(pos-ball.before_p)).tolist()
                if ball.current['floor_contacts'] or pos[2]<-.1:
                    terminal=True;break
            if not terminal:error='no ground terminal within 8s'
        except Exception as e:error=repr(e)
        result={'shot_id':serve['shot_id'],'crossing_w':crossing,'terminal_s':round(t+.001,3),'table_contacts':ball.current['table_contacts'],'net_contacts':ball.current['net_contacts'],'error':error}
        results.append(result)
        if error:print(json.dumps(result),flush=True)
    a.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'checked':len(results),'errors':sum(r['error'] is not None for r in results),'hit_plane_crossings':sum(r['crossing_w'] is not None for r in results)}),flush=True)
    raise SystemExit(any(r['error'] for r in results))


if __name__=='__main__':main()
