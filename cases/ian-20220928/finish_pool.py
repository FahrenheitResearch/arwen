#!/usr/bin/env python3
"""Consolidate completed owned parents, then archive them into one pool.

This is storage orchestration, not a forecast or a change to sealed weather
artifacts. It waits for compatible final checkpoints before retiring spent
downloads and spinup history. Every removed byte is recorded in parent.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

ENGINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ENGINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--parent-ids', required=True)
    parser.add_argument('--archive-host')
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--wait-for', type=Path, help='optional completed proof/resume receipt before moving its parents')
    args = parser.parse_args()
    if bool(args.archive_host) != bool(args.archive_root):
        parser.error('--archive-host and --archive-root are required together')
    from prepare_parents import atomic, sha, consolidate_owned, archive_parent
    from gpuwm.io.restart import read_restart_header
    from gpuwm.prepared_domain_tree_forecast import _runtime_source_identity
    identity = dict(_runtime_source_identity())
    directories = [args.base/'parents'/p for p in args.parent_ids.split(',')]
    started = time.monotonic()
    while True:
        ready = args.wait_for is None or args.wait_for.is_file()
        if not ready:
            if time.monotonic()-started > 21600:
                raise TimeoutError('required proof receipt did not arrive within six hours')
            time.sleep(10)
            continue
        for directory in directories:
            path = directory/'parent.json'
            if not path.is_file():
                ready = False
                break
            parent = json.loads(path.read_text())
            header = read_restart_header(parent['checkpoint'])
            if header['experiment_fingerprint_components']['runtime_source_identity'] != identity:
                ready = False
                break
        if ready:
            break
        if time.monotonic()-started > 21600:
            raise TimeoutError('final compatible parent queue did not finish within six hours')
        time.sleep(10)
    for directory in directories:
        before = time.perf_counter()
        path = directory/'parent.json'
        saved = json.loads(path.read_text())
        checkpoints = sorted(Path(saved['checkpoint']).parent.glob('gpuwmrst_*.npz'))
        with ThreadPoolExecutor(max_workers=8) as hashes:
            saved['restart_set'] = list(hashes.map(lambda f: {
                'path': str(f), 'sha256': sha(f), 'bytes': f.stat().st_size}, checkpoints))
        if len(saved['restart_set']) != 2:
            raise ValueError(f'expected one complete two-domain restart set: {directory}')
        saved.setdefault('owned_storage_followups', []).append(
            consolidate_owned(directory, saved, args.base/'compact-owned'))
        atomic(path, saved)
        if args.archive_host:
            saved = archive_parent(directory, saved, args.archive_host,
                                   args.archive_root, Path(saved['experiment_config']))
        print(json.dumps({'parent_id': saved['parent_id'], 'status': 'consolidated',
                          'wall_seconds': time.perf_counter()-before}), flush=True)
    atomic(args.base/'finish-pool.json', {'status': 'completed',
        'parent_ids': args.parent_ids.split(','), 'wall_seconds_including_wait': time.monotonic()-started})


if __name__ == '__main__':
    main()
