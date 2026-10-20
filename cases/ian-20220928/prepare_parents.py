#!/usr/bin/env python3
"""Prepare and spin up each independent GEFS ancestor through public doors.

Run under the node's GPU mutex. One subprocess per visible card; each result
is durable and reruns skip completed parents. No checkpoint averaging occurs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


# A launched -m child otherwise searches the shell's working directory before
# PYTHONPATH and can silently import a different checkout's checkpoint engine.
ENGINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINE))
os.environ['PYTHONPATH'] = str(ENGINE)+os.pathsep+os.environ.get('PYTHONPATH', '')
os.chdir(ENGINE)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, payload):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(payload, indent=2)+'\n')
    pending.replace(path)


def consolidate_owned(directory, saved, binary):
    """Retire only this driver's spent payloads after a sealed checkpoint."""
    started = time.perf_counter()
    compact = json.loads(subprocess.check_output([str(binary), saved['prepared_root']], text=True))
    candidates = list((directory/'inputs').rglob('*.grib2')) + list((directory/'inputs').rglob('*.grb2'))
    # Fetch uses .grib as well as paired .grib2; retain every JSON SHA receipt.
    candidates += list((directory/'inputs').rglob('*.grib'))
    # Upstream archive object names often end in fNNN, without an extension.
    # Only retire paths explicitly owned by this run's fetch manifests.
    owned_inputs = (directory/'inputs').resolve()
    for manifest in owned_inputs.rglob('fetch-manifest.json'):
        fetched = json.loads(manifest.read_text())
        for row in fetched.get('files', []):
            if row.get('relpath'):
                path = (manifest.parent/row['relpath']).resolve()
                if path.is_relative_to(owned_inputs) and path.is_file():
                    candidates.append(path)
    candidates = list(dict.fromkeys(p.resolve() for p in candidates))
    history = Path(saved['checkpoint']).parent/'wrfout'
    removed = [{'path': str(f), 'bytes': f.stat().st_size} for f in candidates]
    if history.is_dir():
        removed += [{'path': str(f), 'bytes': f.stat().st_size} for f in history.rglob('*') if f.is_file()]
    for f in candidates:
        f.unlink()
    if history.is_dir():
        shutil.rmtree(history)
    return {'consolidation': compact, 'retired_spent_payloads': removed,
            'retired_bytes': sum(r['bytes'] for r in removed), 'wall_seconds': time.perf_counter()-started}


def archive_parent(directory, saved, host, remote_root, case):
    """Move an owned finished parent off a capacity-limited compute node."""
    started = time.perf_counter()
    destination = remote_root/saved['parent_id']
    subprocess.run(['ssh', host, 'mkdir', '-p', str(destination)], check=True)
    subprocess.run(['rsync', '-aH', '--exclude=parent.json', str(directory)+'/', f'{host}:{destination}/'], check=True)
    checks = [(Path(saved['checkpoint']), saved['checkpoint_sha256'])]
    checks += [(Path(r['path']), r['sha256']) for r in saved['restart_set'] if Path(r['path']) != checks[0][0]]
    for path, digest in checks:
        target = destination/path.relative_to(directory)
        actual = subprocess.check_output(['ssh', host, 'sha256sum', str(target)], text=True).split()[0]
        if actual != digest:
            raise RuntimeError(f'archive restart checksum mismatch: {target}')
    def relocate(path):
        return str(destination/Path(path).relative_to(directory))
    archived = {**saved, 'checkpoint': relocate(saved['checkpoint']), 'prepared_root': relocate(saved['prepared_root']),
                'experiment_config': str(remote_root.parent/'case'/case.name),
                'archive': {'host': host, 'original_directory': str(directory), 'wall_seconds': time.perf_counter()-started,
                            'verification': 'rsync transfer checksums and SHA-256 of every restart member'}}
    archived['restart_set'] = [{**r, 'path': relocate(r['path'])} for r in saved['restart_set']]
    archived['archive_manifests'] = [{**r, 'path': relocate(r['path'])} for r in saved['archive_manifests']]
    # Publish the remapped metadata only after transfer and verification.
    payload = json.dumps(archived, indent=2)+'\n'
    publish = "from pathlib import Path; import sys; p=Path(sys.argv[1]); t=p.with_suffix('.pending'); t.write_text(sys.stdin.read()); t.replace(p)"
    # A byte pipe: on Windows a text pipe turns every newline into CRLF (the
    # release lint, T4), and the remote reads the JSON as bytes anyway.
    subprocess.run(['ssh', host, shlex.join(['python3', '-c', publish, str(destination/'parent.json')])],
                   input=payload.encode('utf-8'), check=True)
    marker = directory.parent/(directory.name+'.archived.json')
    atomic(marker, archived)
    shutil.rmtree(directory)
    return archived


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--geog-root', type=Path, required=True)
    p.add_argument('--cards', default='all')
    p.add_argument('--parent-ids', help='optional comma-separated subset; receipt names the reduced ancestry')
    p.add_argument('--compact-owned', type=Path, help='built compact_owned.rs; retire this run\'s raw downloads and spinup histories after checkpointing')
    p.add_argument('--archive-host', help='optional SSH destination for finished owned parents')
    p.add_argument('--archive-root', type=Path, help='parent pool directory on --archive-host; case configs live in its sibling case/')
    a = p.parse_args()
    a.case, a.out = a.case.resolve(), a.out.resolve()
    if bool(a.archive_host) != bool(a.archive_root):
        p.error('--archive-host and --archive-root are required together')
    a.out.mkdir(parents=True, exist_ok=True)
    roster = json.loads((a.case/'GEFS-ROSTER.json').read_text())
    rows = roster['parents']
    if a.parent_ids:
        selected = set(a.parent_ids.split(','))
        if selected - {r['parent_id'] for r in rows}:
            p.error('unknown parent IDs')
        rows = [r for r in rows if r['parent_id'] in selected]
    identities = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).split()
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    if inherited:
        identities = [identities[int(i)] if i.isdigit() else i for i in inherited.split(',')]
    cards = identities if a.cards == 'all' else [identities[int(i)] for i in a.cards.split(',')]
    if not cards or len(cards) != len(set(cards)):
        p.error('select distinct visible cards')
    # Each card owns a serial queue. The CLI children never share a card.
    queues = [rows[i::len(cards)] for i in range(len(cards))]
    def worker(item):
        card, queue = item
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': card, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
        done = []
        for row in queue:
            directory = a.out/row['parent_id']
            marker = a.out/(row['parent_id']+'.archived.json')
            if marker.is_file():
                saved = json.loads(marker.read_text())
                if saved['experiment_config_sha256'] != sha(a.case/row['config']):
                    raise ValueError('archived parent belongs to another config')
                checkpoint = Path(saved['checkpoint'])
                # A verified archive may have moved back onto this host.
                # Resume must not depend on an SSH key for the local machine.
                actual = (sha(checkpoint) if checkpoint.is_file() else
                          subprocess.check_output(['ssh', saved['archive']['host'], 'sha256sum',
                                                   saved['checkpoint']], text=True).split()[0])
                if actual != saved['checkpoint_sha256']:
                    raise ValueError('archived parent checkpoint changed')
                done.append(saved)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory/'parent.json'
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if saved['experiment_config_sha256'] != sha(a.case/row['config']):
                    raise ValueError('completed parent belongs to another experiment config')
                if sha(saved['checkpoint']) != saved['checkpoint_sha256']:
                    raise ValueError('completed parent checkpoint changed')
                done.append(saved)
                continue
            config = a.case/row['config']
            before = time.perf_counter()
            prepared_path = directory/'prepared.json'
            if prepared_path.exists():
                prepared = json.loads(prepared_path.read_text())
                prep_seconds = prepared['wall_seconds']
            else:
                command = [sys.executable, '-B', '-m', 'gpuwm', 'go', str(config), '--prepare-only',
                    '--geog-root', str(a.geog_root), '--outdir', str(directory/'inputs'), '--products', 'none', '--no-verify-visuals']
                with (directory/'prepare.log').open('w') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'parent {row["parent_id"]}: preparation failed; see {directory}/prepare.log')
                prepared = next(json.loads(line) for line in reversed((directory/'prepare.log').read_text().splitlines()) if line.startswith('{"experiment_config"'))
                prep_seconds = time.perf_counter()-before
                prepared['wall_seconds'] = prep_seconds
                atomic(prepared_path, prepared)
            command = [sys.executable, '-B', '-m', 'gpuwm', 'sim', prepared['prepared_root'],
                '--experiment-config', str(config), '--outdir', str(directory/'spinup'), '--render-products', 'none']
            before = time.perf_counter()
            with (directory/'spinup.log').open('w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'parent {row["parent_id"]}: spinup failed; see {directory}/spinup.log')
            run = Path((directory/'spinup'/'latest-run.txt').read_text().strip())
            if not run.is_absolute():
                run = directory/'spinup'/run
            checkpoint = next(run.rglob('gpuwmrst_d01_2022-09-28_07-05-00.npz'), None)
            if checkpoint is None:
                candidates = list(run.rglob('gpuwmrst_d01*07*05*.npz'))
                if len(candidates) != 1:
                    raise RuntimeError(f'analysis checkpoint missing in {run}')
                checkpoint = candidates[0]
            root = Path(prepared['prepared_root'])
            saved = {k: row[k] for k in ('parent_id', 'weight', 'provenance')}
            saved.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), prepared_root=str(root),
                experiment_config=str(config), experiment_config_sha256=sha(config),
                prepared_head_sha256=json.loads((root/'boundary-stream'/'head.json').read_text())['head_sha256'],
                stage_wall_seconds={'fetch_and_prepare': prep_seconds, 'spinup': time.perf_counter()-before},
                card=card, analysis_time=roster['analysis_time'], coverage_end=roster['forecast_end'])
            # Retain the actual archive hashes/coverage receipts with the case.
            manifests = list((directory/'inputs').rglob('fetch-manifest.json'))
            saved['archive_manifests'] = [{'path': str(f), 'sha256': sha(f)} for f in manifests]
            with ThreadPoolExecutor(max_workers=8) as hashes:
                saved['restart_set'] = list(hashes.map(lambda f: {'path': str(f), 'sha256': sha(f), 'bytes': f.stat().st_size}, run.glob('gpuwmrst_*.npz')))
            if a.compact_owned:
                saved['owned_storage'] = consolidate_owned(directory, saved, a.compact_owned)
            atomic(result_path, saved)
            if a.archive_host:
                saved = archive_parent(directory, saved, a.archive_host, a.archive_root, config)
            print(json.dumps({'parent_id': row['parent_id'], 'status': 'completed', 'timings': saved['stage_wall_seconds']}), flush=True)
            done.append(saved)
        return done
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(cards)) as pool:
        complete = [row for group in pool.map(worker, zip(cards, queues)) for row in group]
    by_id = {r['parent_id']: r for r in complete}
    payload = {**{k: roster[k] for k in ('analysis_time', 'observation_cutoff', 'forecast_end')},
        'schema': 'gpuwm-checkpoint-parents.v1', 'kind': 'NO-DA GEFS spun independently; no observations assimilated',
        'parents': [by_id[row['parent_id']] for row in rows], 'roster_ancestors': len(roster['parents']),
        'represented_ancestors': len(rows), 'wall_seconds_this_invocation': time.perf_counter()-started,
        'parallelism': 'one preparation/spinup subprocess per visible card'}
    atomic(a.out/'PARENTS.json', payload)
    print(a.out/'PARENTS.json')

if __name__ == '__main__':
    main()
