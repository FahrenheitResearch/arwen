#!/usr/bin/env python3
"""Reuse a completed proof through the public restart door, under its mutex."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mutex', required=True)
    parser.add_argument('--ledger')
    args = parser.parse_args()
    engine = Path(__file__).resolve().parents[2]
    os.environ['PYTHONPATH'] = str(engine)
    os.chdir(engine)
    before = time.monotonic()
    while not (args.proof/'DONE.json').is_file():
        if time.monotonic()-before > 14400:
            raise TimeoutError('proof has not completed within four hours')
        time.sleep(5)
    runs = list((args.proof/'ensemble').glob('run-*/ensemble-set.json'))
    if len(runs) != 1:
        raise ValueError('proof must identify exactly one ensemble set')
    original = json.loads((args.proof/'command.json').read_text())
    forecast = original[original.index('--')+1:]
    forecast += ['--resume-ensemble', str(runs[0].parent)]
    args.out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).with_name('measure.py')),
        '--out', str(args.out/'measurement'), '--mutex', args.mutex]
    if args.ledger:
        command += ['--ledger', args.ledger]
    command += ['--', *forecast]
    (args.out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    result = subprocess.run(command)
    receipt = json.loads(runs[0].read_text())
    payload = {'exit_code': result.returncode, 'command': command,
               'reused_completed_members': receipt.get('reused_completed_members'),
               'newly_completed_members': receipt.get('newly_completed_members')}
    (args.out/'RESULT.json').write_text(json.dumps(payload, indent=2)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    if (payload['reused_completed_members'], payload['newly_completed_members']) != (4, 0):
        raise ValueError('completed proof was not fully reused')


if __name__ == '__main__':
    main()
