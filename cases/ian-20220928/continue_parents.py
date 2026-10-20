#!/usr/bin/env python3
"""Continue a measured parent queue after a proof shard releases the card."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--mutex', required=True)
    parser.add_argument('--ledger')
    parser.add_argument('--geog-root', required=True)
    parser.add_argument('--parent-ids', required=True)
    parser.add_argument('--refresh-ids', default='')
    parser.add_argument('--archive-host')
    parser.add_argument('--archive-root')
    args = parser.parse_args()
    engine = Path(__file__).resolve().parents[2]
    os.environ['PYTHONPATH'] = str(engine)
    os.chdir(engine)
    before = time.monotonic()
    while not args.wait_for.is_file():
        if time.monotonic()-before > 14400:
            raise TimeoutError('proof shard has no result after four hours')
        time.sleep(5)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    # Preserve incompatible development checkpoints and receipts; rerun from
    # their already prepared independent boundaries with the installed engine.
    for parent in filter(None, args.refresh_ids.split(',')):
        receipt = args.base/'parents'/parent/'parent.json'
        if receipt.exists():
            receipt.rename(receipt.with_name(f'parent-development-{stamp}.json'))
    case = engine/'cases/ian-20220928'
    command = [sys.executable, str(case/'measure.py'), '--out',
               str(args.base/f'measure-remaining-{stamp}'), '--mutex', args.mutex,
               '--timeout-seconds', '14400']
    if args.ledger:
        command += ['--ledger', args.ledger]
    command += ['--', sys.executable, str(case/'prepare_parents.py'), '--case',
                str(args.base/'case'), '--out', str(args.base/'parents'),
                '--geog-root', args.geog_root, '--parent-ids', args.parent_ids,
                '--compact-owned', str(args.base/'compact-owned')]
    if args.archive_host:
        command += ['--archive-host', args.archive_host, '--archive-root', args.archive_root]
    raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
