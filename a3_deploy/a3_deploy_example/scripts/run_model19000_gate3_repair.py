#!/usr/bin/env python3
"""Resume infrastructure repair in existing hope; preflight gates formal100.

Uses mutable prior build caches but separate installs/packages and evidence.
Original model packages, installed binaries, and failed-run evidence are hashed
before/after. No package install, security change, state deletion, or hardware.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

SCRIPT=Path(__file__).resolve()
ROOT=SCRIPT.parents[3]
GEAR=ROOT/'a3_deploy/a3_deploy_example'
SCRIPTS=GEAR/'scripts'
OLD=ROOT.parent/'hope-stationary-gru-evaluations/20261108-model19000-native-v2'
POLICY=GEAR/'models/model_19000_native_v2/policy'
DEFAULT=OLD.with_name('20261108-model19000-gate3-infrastructure-repair')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def read(path):return json.loads(path.read_text())
def quote(path):return shlex.quote(str(path))


def preflight_gate(folder):
    """Structural proof, not a policy score or a certification substitute."""
    run=read(folder/'run.json');rally=read(folder/'pp_rally_report.json')
    planner=read(folder/'pp_planner_cpp_report.json');plant=read(folder/'pp_mujoco_plant_report.json')
    physical=read(folder/'pp_physical_ball_report.json');drain=read(folder/'pp_gate3_drain_report.json')
    checks={
        'unscored_prefix':run.get('evaluation_kind')=='integration_preflight_not_certification',
        'all_six_launched':drain['launched_shot_ids']==list(range(1,7)),
        'initial_idle_preserved':len(rally['rows'])==6 and rally['rows'][0]['t']>=19.9,
        'planner_monitored_alive':not (folder/'infrastructure_failure.json').exists() and
            (folder/'planner.pid.monitor-complete.json').exists() and read(folder/'planner.pid.monitor-complete.json').get('alive') is True,
        'valid_commands_recorded':planner.get('valid_commands',0)>0 and planner.get('rows_observed',0)==6,
        'complete_physical_telemetry':len(physical['rows'])==6 and all(
            x['telemetry_complete'] and x['trajectory_complete'] for x in physical['rows']),
        'drain_complete':drain['status']=='complete' and drain['evidence_complete'] and not drain['pending_shot_ids'],
        'report_tools_compatible':not re.search(r'unrecognized arguments:|Traceback \(most recent call last\)|CONFIG/RECIPE FAIL',
                                             (folder/'console.log').read_text(errors='replace')),
    }
    for name in ('home_support_frame','torso_hold','collision_ledger'):
        checks[name+'_evidence']=plant.get(name,{}).get('evidence_complete') is True
    checks['idle_evidence']=bool(plant.get('idle_smoothness',{}).get('steady'))
    lifecycle=rally['runner_trace_engagement']
    executed=set(lifecycle['engaged_flight_ids']) & set(lifecycle['completed_flight_ids'])
    accepted={}
    for line in (folder/'pp_runner.log').read_text(errors='replace').splitlines():
        if '[pp shared-home]' not in line:continue
        fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
        if fields.get('match')!='1':
            checks['no_valid_command_HOME_mismatch']=False
            continue
        valid=(fields['epoch']==fields['expected_epoch'] and fields['epoch']!='0' and
               fields['calibration']==fields['expected_calibration'] and fields['world']==fields['expected_world'] and
               fields['calibration']!='0' and fields['world']!='0' and fields['base_fresh']=='1' and
               fields['schema']=='4' and fields['frame']=='0' and int(fields['receiver_epoch'])>0 and
               abs(float(fields['target_x'])-float(fields['expected_x']))<1e-6 and
               abs(float(fields['plane'])-.57)<1e-6)
        if valid:accepted[int(fields['flight'])]=fields
    checks.setdefault('no_valid_command_HOME_mismatch',True)
    checks['actual_swing_engages_and_completes']=bool(executed & accepted.keys())
    audit=list(csv.DictReader((folder/'pp_planner_debug.csv').open()))
    matches=[]
    for row in audit:
        if row.get('wire_valid')!='1':continue
        flight=int(row['flight_id'])
        if flight not in accepted:continue
        expected=accepted[flight]
        if (int(row['session_home_epoch'])==int(expected['expected_epoch']) and
            int(row['command_seq'])==int(expected['wire_command']) and int(row['revision_id'])==int(expected['revision']) and
            abs(float(row['strike_x'])-float(expected['expected_x']))<1e-6):matches.append(flight)
    checks['Planner_Runner_HOME_identity_and_plane_join']=bool(set(matches)&executed)
    behavior={name:plant[name]['failures'] for name in ('home_support_frame','torso_hold','collision_ledger')}
    behavior['idle_smoothness']=plant['idle_smoothness']['failures']
    categories=[]
    for name,passed in checks.items():
        if passed:continue
        if name=='actual_swing_engages_and_completes':
            category=('behavioral_fall_before_completion' if rally.get('falls',0)>0 and accepted
                      else 'execution_or_command_acceptance_not_proven')
        else:category='infrastructure_or_missing_evidence'
        categories.append({'check':name,'category':category})
    return {'infrastructure_pass':all(checks.values()),'checks':checks,
            'executed_and_HOME_verified_flights':sorted(executed & accepted.keys()),
            'behavioral_failures_retained':behavior,'failure_categories':categories,
            'observed_falls':rally.get('falls'),'certification':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    batch_lock=(out/'batch.lock').open('a')
    try:fcntl.flock(batch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('another repair batch owns this output; existing process left running')
    # These assertions bind to the frozen original actor and already verified v2 graph.
    assert sha(POLICY/'exported/policy.onnx')=='2b2a029b450fb8601ecf34eefdbd3df6ab9f8597a33593dc9b7d6419fcd34d27'
    assert sha(POLICY/'params/deploy.yaml')=='160f282fdb95d4ca5d0df5222897a203150152248fcc364f8ba723086357f3fc'
    checkpoint=OLD.parent/'20261106-eight-checkpoint-comparison/model_19000/model_19000.pt'
    assert sha(checkpoint)=='744526d2e7a9fdabe990070054ec3bd8126b286bc79041f94a930b0078c257c0'
    failed=OLD/'formal-20261108-123341'
    assert sha(GEAR/'config/gate3/scenarios/fixed-home-100.yaml')==sha(failed/'scene.yaml')
    preservation=out/'preservation.json'
    if not preservation.exists():
        original={str(p):sha(p) for base in [failed,*[GEAR/'models'/n for n in
                  ('model_19000','model_19000_native','model_19000_native_v2')]] for p in base.rglob('*') if p.is_file()}
        original.update(read(OLD/'runtime-files-sha256.json'))
        save(preservation,original)
    def preserved():
        for p,h in read(preservation).items():
            assert Path(p).is_file() and sha(p)==h,f'Preserved artifact changed: {p}'
    preserved()
    # Reuse only the hash-verified, unchanged instrumented simulator source/install.
    old_sources=read(OLD/'source-binding.json')['sources']
    for name,h in old_sources.items():
        if name.startswith('a3_deploy/A3_MuJoCo_Sim/'):
            assert sha(ROOT/name)==h,'Simulator source changed; separate validated build required'
    for p,h in read(OLD/'sim-build.receipt.json')['outputs'].items():assert sha(p)==h
    files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).split(b'\0')
    sources={p.decode():sha(ROOT/p.decode()) for p in files if p and p.decode().startswith(('a3_deploy/','hope_ws/','NatNet2ROS2/','run_gate3.sh')) and (ROOT/p.decode()).is_file()}
    for p in SCRIPTS.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:sources[str(p.relative_to(ROOT))]=sha(p)
    binding=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    source_file=out/'repair-source-binding.json'
    if source_file.exists():assert read(source_file)['binding']==binding,'Source changed: choose a fresh --output'
    else:save(source_file,{'binding':binding,'sources':sources})
    print(f'Frozen model19000; run gru-formal-2b8c09938a\nCheckpoint: {checkpoint}\n'
          f'Checkpoint SHA256: {sha(checkpoint)}\nPolicy: {POLICY}\nONNX SHA256: {sha(POLICY/"exported/policy.onnx")}\n'
          f'Six-clean-serve infrastructure preflight BEFORE unchanged formal100. ROS_DOMAIN_ID=77.\nEvidence: {out}',flush=True)
    def stage(name,command,outputs=(),inside=True,behavior_exit=False,inventory=()):
        receipt=out/(name+'.receipt.json')
        if receipt.exists():
            old=read(receipt)
            if old['binding']==binding and old['complete'] and all(Path(p).exists() and sha(p)==h for p,h in old['outputs'].items()):
                print('SKIP verified '+name,flush=True);return old
        prefix='set -eo pipefail\nsource /opt/ros/jazzy/setup.bash\nexport ROS_DOMAIN_ID=77\n'
        argv=['distrobox','enter','hope','--','bash','-lc',prefix+command] if inside else command
        save(out/(name+'.command.json'),{'argv':argv})
        start=time.monotonic();print('START '+name,flush=True)
        with (out/(name+'.log')).open('a') as log:
            proc=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while True:
                try:rc=proc.wait(timeout=30);break
                except subprocess.TimeoutExpired:print(f'RUNNING {name}: {time.monotonic()-start:.0f}s',flush=True)
        artifacts=set(map(Path,outputs))
        for directory in inventory:artifacts.update(p for p in directory.rglob('*') if p.is_file())
        complete=all(Path(p).is_file() for p in outputs) and (rc==0 or behavior_exit and rc==1)
        saved={'binding':binding,'complete':complete,'exit_code':rc,'seconds':time.monotonic()-start,
               'outputs':{str(p):sha(p) for p in sorted(artifacts) if p.is_file()}}
        save(receipt,saved);print(f'END {name}: exit={rc}, {saved["seconds"]:.1f}s; {out/(name+".log")}',flush=True)
        if not complete:raise RuntimeError(name+' failed; no later stage launched')
        return saved
    stage('container-access',['podman','inspect','hope','--format','{{.State.Status}}'],inside=False)
    stage('original-hardware-identity-diagnosis',f'/usr/bin/python3 {quote(SCRIPTS/"pp_wire_identity_probe.py")} /var/lib/hope-lifecycle/planner_wire_identity.state --json {quote(out/"hardware-lock-diagnosis.json")}',[out/'hardware-lock-diagnosis.json'])
    stage('native-identity-contract',f'''mkdir -p {quote(out/'native-contract-fixtures')}
g++ -std=c++17 -O2 -DHOPE_VERIFY_PLANNER_PACKER=1 -I/usr/include/eigen3 \
 -I {quote(GEAR/'src/a3/a3_deploy_onnx_ref/include')} -I {quote(ROOT/'hope_ws/src/hope_planner_cpp/include')} \
 {quote(SCRIPTS/'tests/gate3_identity_contract_test.cpp')} {quote(ROOT/'hope_ws/src/hope_planner_cpp/src/schema2_packer.cpp')} \
 -o {quote(out/'native-identity-contract')}
{quote(out/'native-identity-contract')} {quote(out/'native-contract-fixtures')}
''',[out/'native-identity-contract'])
    ws=out/'ros-ws';ws.mkdir(exist_ok=True)
    if not (ws/'src').exists():(ws/'src').symlink_to(ROOT/'hope_ws/src',target_is_directory=True)
    install=ws/'install';dist=out/'runner-dist'
    stage('planner-build',f'''colcon --log-base {quote(out/'ros-log')} build --base-paths {quote(ROOT/'hope_ws/src')} {quote(ROOT/'NatNet2ROS2/src')} \
 --packages-up-to hope_msgs hope_bringup hope_planner_cpp motion_capture_tracking \
 --build-base {quote(OLD/'ros-build')} --install-base {quote(install)} --cmake-force-configure \
 --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
''',[install/'local_setup.bash',install/'hope_planner_cpp/lib/hope_planner_cpp/hope_planner_cpp_node'],inventory=[install])
    deps=ROOT.parent/'nohope/a3_deploy/a3_deploy_example/thirdparty'
    env=f'''source {quote(install/'local_setup.bash')}
export A3_PACKAGE_BUILD_DIR={quote(OLD/'runner-build')} A3_PACKAGE_OUTPUT_DIR={quote(dist)}
export A3_THIRDPARTY_ROOT={quote(deps)} onnxruntime_ROOT={quote(deps/'onnxruntime/onnxruntime-linux-x64-1.19.2')}
export A3_CMAKE_PYTHON=/usr/bin/python3 HAS_ROS2=1
export LD_LIBRARY_PATH={quote(dist)}:"${{LD_LIBRARY_PATH:-}}"
'''
    stage('Runner-build',env+f'''cd {quote(GEAR)}
bash scripts/build_a3_deploy_pkg.sh --arch x86_64 --without-teleop --jobs 4 --policy-dir {quote(POLICY)} \
 --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.hitter_pingpong.yaml
cmake --build {quote(OLD/'runner-build')} --target model19000_native_parity --parallel 4
''',[dist/'a3_deploy_onnx_ref_pingpong',dist/'model19000_native_parity'],inventory=[dist])
    preserved()
    stage('native-parity',env+' '.join(map(quote,[dist/'model19000_native_parity',
        GEAR/'models/model_19000/policy/exported/actor.onnx',POLICY/'exported/policy.onnx',OLD,
        out/'native-parity.json',GEAR/'models/model_21800/policy/exported/policy.onnx'])),[out/'native-parity.json'])
    parity=read(out/'native-parity.json')
    assert parity['status']=='PASS' and parity['full_policy_loader_test'] and parity['feedforward_compatibility_pass']
    runtime=out/'runtime.yaml';save(runtime,{'runner_dist':str(dist),'sim_install':str(OLD/'sim-build/install'),'hope_ws':str(ws)})
    base=' '.join(map(quote,['bash',ROOT/'run_gate3.sh','--runtime',runtime,'--policy-dir',POLICY,
                            '--scenario',GEAR/'config/gate3/scenarios/fixed-home-100.yaml','--headless']))
    stage('native-inspection',env+f'''{quote(dist/"a3_deploy_onnx_ref_pingpong")} --inspect-policy --policy-dir {quote(POLICY)} > {quote(out/"native-inspection.yaml")}
/usr/bin/python3 - {quote(out/'native-inspection.yaml')} <<'PY'
import sys,yaml
r=yaml.safe_load(open(sys.argv[1]))
assert r['recurrent_explicit_state'] and r['recurrent_advances_during_inspection']==0
assert r['observation_dim']==110 and r['action_dim']==31 and r['step_dt']==.02
PY
''',[out/'native-inspection.yaml'])
    def run_stage(name,extra,cap,behavior=False):
        destination=out/(name+'-'+time.strftime('%Y%m%d-%H%M%S'))
        result=stage(name,env+f'timeout --signal=INT --kill-after=45s {cap}s '+base+' '+extra+' --output '+quote(destination),
                     [destination/'run.json'],behavior_exit=behavior,inventory=[destination])
        return next(Path(p).parent for p in result['outputs'] if p.endswith('/run.json'))
    check=run_stage('gate3-check','--check',90)
    # Only the simulation identity override may differ from the old effective
    # model-derived profile; all geometry and physics must compare exactly.
    stage('profile-contract-verification',env+f"""/usr/bin/python3 - {quote(check/'fixed_home_ppq004.yaml')} {quote(failed/'fixed_home_ppq004.yaml')} <<'PY'
import sys,yaml
new=yaml.safe_load(open(sys.argv[1]))['hope_planner']['ros__parameters']
old=yaml.safe_load(open(sys.argv[2]))['hope_planner']['ros__parameters']
assert new['wire_identity_state_file']==''
assert all(new[k]==v for k,v in old.items())
assert set(new)-set(old)=={{'wire_identity_state_file'}}
print('PASS unchanged model-derived geometry and physics; fresh isolated simulation identity')
PY
""")
    preflight=run_stage('integration-preflight','--integration-preflight',360,True)
    proof=preflight_gate(preflight);save(out/'preflight-infrastructure-proof.json',proof)
    print(json.dumps(proof,indent=2),flush=True)
    preserved()
    if not proof['infrastructure_pass']:raise RuntimeError('Integration preflight requirements not met; formal100 HELD; see proof failure categories')
    formal=run_stage('formal-fixed-home-100','',900,True)
    result=read(formal/'run.json');assert result['evaluation_kind']=='formal_gate3'
    preserved()
    print(json.dumps({'formal_result':str(formal),'exit_code':result.get('exit_code'),
                      'certification_pass':result.get('certification_pass')},indent=2))
    return 0 if result.get('certification_pass') is True else 1


if __name__=='__main__':
    try:sys.exit(main())
    except (Exception,KeyboardInterrupt) as e:
        print('GATE3_REPAIR_STOPPED: '+str(e),file=sys.stderr);sys.exit(2)
