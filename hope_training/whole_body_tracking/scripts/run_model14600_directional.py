"""Evaluate verified non-GRU model14600 on the frozen directional comparison."""
import argparse,copy,hashlib,json,subprocess,sys,tempfile,time
from pathlib import Path
from mujoco_eval_recurrent import ROOT
from reference_gru_diagnostic import reference_ledger
from run_directional_replay_comparison import WINDOWS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for f in ('bundle','output-root'):p.add_argument('--'+f,required=True,type=Path)
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    scene=ROOT/'docs/evaluations/directional-120-20261107/directional-120.yaml';replay=ROOT/'docs/evaluations/directional-120-20261107/planner-command-replay.json'
    plan=json.loads(replay.read_text());m=json.loads((a.bundle/'manifest.json').read_text());verification=json.loads((a.bundle/'verification.json').read_text())
    if verification['result']!='PASS' or m['onnx_sha256']!=verification['onnx_sha256']:raise ValueError('Bundle verification missing')
    if hashlib.sha256(scene.read_bytes()).hexdigest()!=plan['scene_sha256']:raise ValueError('Scene/replay mismatch')
    a.output_root.mkdir(parents=True,exist_ok=True);out=Path(tempfile.mkdtemp(prefix='preflight-' if a.preflight else 'model14600-',dir=a.output_root));scripts=ROOT/'hope_training/whole_body_tracking/scripts'
    if a.preflight:
        ledger,_=reference_ledger(scene);selected=[ledger[i] for i in (0,1,6,7,8)];now=20.;deltas={}
        for row in selected:deltas[row['shot_id']]=now-row['nominal_launch_at_s'];now+=row['flight_s']+row['pause_s']
        plan=copy.deepcopy(plan);plan['commands']=[r for r in plan['commands'] if r['shot_id'] in deltas]
        for r in plan['commands']:r['sim_t']+=deltas[r['shot_id']];r['deadline_sim_s']+=deltas[r['shot_id']]
        replay=out/'preflight-replay.json';replay.write_text(json.dumps(plan,indent=2)+'\n')
    transport=Path(tempfile.mkdtemp(prefix='hope-14600-',dir='/tmp'));sock=transport/'bridge.sock'
    with (out/'bridge.log').open('w') as log:
        bridge=subprocess.Popen([sys.executable,str(scripts/'planner_command_replay_bridge.py'),'--replay',str(replay),'--socket',str(sock)],stdout=log,stderr=subprocess.STDOUT)
        try:
            until=time.monotonic()+10
            while not sock.exists():
                if bridge.poll() is not None or time.monotonic()>until:raise RuntimeError('Replay bridge startup failed')
                time.sleep(.05)
            cmd=[sys.executable,str(scripts/'reference_gru_diagnostic.py'),'--bundle',str(a.bundle),'--scene',str(scene),'--socket',str(sock),'--output',str(out/'scored'),'--table-impact-model','ppq004','--neutral-expired-targets','--neutral-final-tail','--pelvis-dropout-windows',WINDOWS,'--sensor-stride','20','--max-clock-lag-s','0.25','--video']
            if a.preflight:cmd.append('--preflight')
            print('OUTPUT '+str(out),flush=True)
            with (out/'actor.log').open('w') as actorlog:rc=subprocess.run(cmd,stdout=actorlog,stderr=subprocess.STDOUT).returncode
            bridge.wait(timeout=10)
        finally:
            if bridge.poll() is None:bridge.terminate();bridge.wait(timeout=10)
            sock.unlink(missing_ok=True);transport.rmdir()
    summary=json.loads((out/'scored/summary.json').read_text());print(json.dumps({'output':str(out),'exit_code':rc,'falls':summary['falls'],'fall_time_s':summary['fall_time_s'],'all':summary['groups']['all'],'error':summary['error']}),flush=True)
    raise SystemExit(rc)


if __name__=='__main__':main()
