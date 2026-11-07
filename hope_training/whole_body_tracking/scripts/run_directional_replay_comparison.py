"""Run each frozen GRU separately on identical directional serves and mocap outages."""
import argparse,hashlib,json,os,socket,subprocess,sys,tempfile,time
from pathlib import Path
from mujoco_eval_recurrent import ROOT

WINDOWS='21:0.04,31:0.1,43:0.2,52:0.3,61:1.0,78:0.5,90:0.2,120:1.0,150:1.5,180:2.0'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for f in ('scene','replay','output-root'):p.add_argument('--'+f,required=True,type=Path)
    p.add_argument('--models',nargs='+',type=int,default=[19000,29999]);a=p.parse_args()
    plan=json.loads(a.replay.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(a.scene)!=plan['scene_sha256']:raise ValueError('Replay scene hash mismatch')
    bundles={i:ROOT/f'a3_deploy/a3_deploy_example/models/model_{i}/policy/exported' for i in a.models}
    params=[sha(b.parent/'params/planner-parameters.json') for b in bundles.values()]
    if len(set(params))!=1:raise ValueError('Models use different planner parameters')
    a.output_root.mkdir(parents=True,exist_ok=True);scripts=ROOT/'hope_training/whole_body_tracking/scripts'
    batch={'scene_sha256':sha(a.scene),'replay_sha256':sha(a.replay),'pelvis_dropout_windows':WINDOWS,'planner_mode':'recorded live C++ commands, held-home source; same commands for both policies','runs':{}}
    for iteration,bundle in bundles.items():
        out=Path(tempfile.mkdtemp(prefix=f'model{iteration}-',dir=a.output_root));transport=Path(tempfile.mkdtemp(prefix='hope-directional-',dir='/tmp'));sock=transport/'bridge.sock'
        print(f'ACTOR START {iteration} {out}',flush=True)
        for script,args in [('mujoco_eval_recurrent.py',['--bundle',str(bundle),'--physical-ball','--verify-only','--output-dir',str(out/'verification')]),('verify_bundle_reset.py',['--bundle',str(bundle),'--output',str(out/'reset.json')])]:
            subprocess.run([sys.executable,str(scripts/script),*args],check=True)
        with (out/'bridge.log').open('w') as log:
            bridge=subprocess.Popen([sys.executable,str(scripts/'planner_command_replay_bridge.py'),'--replay',str(a.replay),'--socket',str(sock)],stdout=log,stderr=subprocess.STDOUT)
            try:
                until=time.monotonic()+10
                while not sock.exists():
                    if bridge.poll() is not None or time.monotonic()>until:raise RuntimeError('Replay bridge startup failed')
                    time.sleep(.05)
                cmd=[sys.executable,str(scripts/'reference_gru_diagnostic.py'),'--bundle',str(bundle),'--scene',str(a.scene),'--socket',str(sock),'--output',str(out/'scored'),'--table-impact-model','ppq004','--neutral-expired-targets','--neutral-final-tail','--pelvis-dropout-windows',WINDOWS,'--sensor-stride','20','--max-clock-lag-s','0.25','--video']
                with (out/'actor.log').open('w') as actorlog:
                    rc=subprocess.run(cmd,stdout=actorlog,stderr=subprocess.STDOUT).returncode
                bridge.wait(timeout=10)
            finally:
                if bridge.poll() is None:bridge.terminate();bridge.wait(timeout=10)
                if sock.exists():sock.unlink()
                transport.rmdir()
        summary=json.loads((out/'scored/summary.json').read_text())
        batch['runs'][str(iteration)]={'directory':str(out),'exit_code':rc,'falls':summary['falls'],'launched':summary['groups']['all']['launched'],'error':summary['error']}
        (a.output_root/'comparison.json').write_text(json.dumps(batch,indent=2)+'\n')
        print('ACTOR DONE '+json.dumps(batch['runs'][str(iteration)]),flush=True)
        if summary['error']:raise RuntimeError('Infrastructure error: '+summary['error'])


if __name__=='__main__':main()
