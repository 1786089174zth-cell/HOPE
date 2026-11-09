"""Run one short frozen-GRU physical-ball scenario in MuJoCo 3.17 and 4.0.

This is an exploratory engine comparison, not the formal Gate3 protocol.
Provide two pre-existing isolated Python environments; no packages are installed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / 'mujoco_eval_recurrent.py'
CASES = (
    ('mujoco317', '3.17.0', False),
    ('mujoco400', '4.0.0', False),
    ('mujoco400_speculative', '4.0.0', True),
)


def run_case(python: Path, expected_version: str, speculative: bool,
             bundle: Path, output: Path, seconds: float, seed: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONUTF8='1')
    version_cmd = [str(python), '-c', 'import mujoco; print(mujoco.__version__)']
    version = subprocess.check_output(version_cmd, text=True, timeout=20, env=env).strip()
    if version != expected_version:
        raise ValueError(f'{python} has MuJoCo {version}; expected {expected_version}')
    common = [str(python), str(EVALUATOR), '--bundle', str(bundle), '--physical-ball']
    if speculative:
        common.append('--speculative-contacts')
    steps = (
        ('verification', [*common, '--output-dir', str(output / 'verification'), '--verify-only']),
        ('simulation', [*common, '--output-dir', str(output / 'simulation'), '--seconds', str(seconds),
                        '--seed', str(seed)]),
    )
    for name, command in steps:
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            try:
                finished = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=300, check=False, env=env)
                code = finished.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout_300s'
        if code != 0:
            return {'status': 'failed', 'phase': name, 'exit_code': code,
                    'log': str(output / f'{name}.log'), 'mujoco_version': version,
                    'speculative_contacts': speculative}
    result = json.loads((output / 'simulation' / 'results.json').read_text(encoding='utf-8'))
    return {'status': 'completed', 'mujoco_version': version,
            'speculative_contacts': speculative, 'checkpoint_sha256': result['checkpoint_sha256'],
            'simulated_seconds': result['simulated_seconds'], 'stop_reason': result['stop_reason'],
            'falls': result['falls'], 'episode_resets': result['episode_resets'],
            'physical_summary': result['physical_summary'],
            'results': str(output / 'simulation' / 'results.json')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-317', type=Path, required=True)
    parser.add_argument('--python-400', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=8.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.seconds <= 120:
        parser.error('--seconds must be in (0,120]')
    if args.output_dir.exists():
        parser.error('output directory already exists; preserve prior evidence')
    args.output_dir.mkdir(parents=True)
    report = {'kind': 'exploratory_mujoco_engine_comparison_not_gate3',
              'bundle': str(args.bundle.resolve()), 'seconds': args.seconds,
              'seed': args.seed, 'python_utf8': True, 'cases': {}}
    failed = False
    for name, expected, speculative in CASES:
        python = args.python_317 if expected == '3.17.0' else args.python_400
        try:
            row = run_case(python.resolve(), expected, speculative, args.bundle.resolve(),
                           args.output_dir / name, args.seconds, args.seed)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            row = {'status': 'failed', 'error': str(exc)}
        report['cases'][name] = row
        failed |= row['status'] != 'completed'
        (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n',
                                                         encoding='utf-8')
        print(name, json.dumps(row), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
