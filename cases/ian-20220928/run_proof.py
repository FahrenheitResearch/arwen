#!/usr/bin/env python3
"""Run one four-member shard of the two-host engineering proof."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parents', type=Path, required=True)
    parser.add_argument('--parent-ids', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mutex', type=Path, required=True)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    engine = Path(__file__).resolve().parents[2]
    os.environ['PYTHONPATH'] = str(engine)+os.pathsep+os.environ.get('PYTHONPATH', '')
    os.chdir(engine)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [args.parents/parent/'parent.json' for parent in args.parent_ids.split(',')]
    if len(paths) != 4:
        parser.error('this proof shard requires four distinct ancestors')
    started = time.monotonic()
    while not all(path.is_file() for path in paths):
        if time.monotonic()-started > 14400:
            raise TimeoutError('parent queue did not finish within four hours')
        time.sleep(5)
    parents = {'schema': 'gpuwm-checkpoint-parents.v1', 'kind': 'NO-DA proof subset',
        'full_roster_parents': 36, 'represented_parents': 4,
        'parents': [json.loads(path.read_text()) for path in paths]}
    manifest = args.out/'PARENTS.json'
    manifest.write_text(json.dumps(parents, indent=2)+'\n')
    case = engine/'cases/ian-20220928'
    request = json.loads((case/'products.json').read_text())
    request.update(best_track=str(case/'bal092022.dat.gz'), defer_render=True,
        experiment_label='NO-DA proof shard; combined render pending')
    product_request = args.out/'products.json'
    product_request.write_text(json.dumps(request, indent=2)+'\n')
    command = [sys.executable, str(case/'measure.py'), '--out', str(args.out/'measurement'),
        '--mutex', str(args.mutex), '--timeout-seconds', '7200']
    if args.ledger:
        command += ['--ledger', str(args.ledger)]
    command += ['--', sys.executable, '-B', '-m', 'gpuwm', 'ensemble',
        '--branch-from', str(manifest), '--members', '4', '--branch-hours', '3',
        '--cards', 'all', '--ensemble-seed', str(args.seed),
        '--cyclone-products', str(product_request), '--outdir', str(args.out/'ensemble')]
    (args.out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    result = subprocess.run(command)
    (args.out/'RESULT.json').write_text(json.dumps({'status': 'completed' if result.returncode == 0 else 'failed',
        'exit_code': result.returncode, 'command': command})+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    (args.out/'DONE.json').write_text(json.dumps({'status': 'completed', 'command': command})+'\n')


if __name__ == '__main__':
    main()
