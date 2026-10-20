#!/usr/bin/env python3
"""Stream owned completed parents between nodes without a local data copy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import time


def host_command(host, command):
    return command if host == 'local' else ['ssh', host, shlex.join(command)]


def remote(host, command, **kwargs):
    return subprocess.check_output(host_command(host, command), text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-host', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--destination-host', required=True, help='SSH host, or local for this node')
    parser.add_argument('--destination-identity', help='durable SSH identity when destination-host is local')
    parser.add_argument('--destination-root', type=Path, required=True)
    parser.add_argument('--parent-ids', required=True)
    parser.add_argument('--wait-for', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.destination_host == 'local' and not args.destination_identity:
        parser.error('--destination-identity is required for a local transfer: remote resume receipts need a reachable host')
    destination_identity = args.destination_identity or args.destination_host
    ids = args.parent_ids.split(',')
    if any(Path(p).name != p or p in ('.', '..') for p in ids):
        parser.error('parent IDs must be single directory names')
    args.out.mkdir(parents=True, exist_ok=False)
    query = '''import json,sys
from pathlib import Path
root=Path(sys.argv[1]); ids=json.loads(sys.argv[2]); marker=sys.argv[3]
rows=[]
ready=not marker or Path(marker).is_file()
for parent_id in ids:
 directory=root/parent_id; receipt=directory/'parent.json'; archived=root/(parent_id+'.archived.json')
 if receipt.is_file():
  try: parent=json.loads(receipt.read_text())
  except (OSError,json.JSONDecodeError):
   ready=False
   continue
  # An upstream archive transfer may still be copying the native payloads.
  # Its receipt is ready only after every outer path has been published here.
  paths=[parent['checkpoint'],parent['prepared_root']]
  paths += [r['path'] for r in parent.get('restart_set',[])]
  paths += [r['path'] for r in parent.get('archive_manifests',[])]
  ready &= all(Path(p).is_relative_to(directory) for p in paths)
  rows.append(parent)
 elif archived.is_file(): rows.append(json.loads(archived.read_text()))
 else: ready=False
print(json.dumps(rows if ready else None))
'''
    before_wait = time.monotonic()
    while True:
        parents = json.loads(remote(args.source_host, ['python3', '-c', query,
            str(args.source_root), json.dumps(ids), str(args.wait_for or '')]))
        if parents is not None:
            break
        if time.monotonic()-before_wait > 21600:
            raise TimeoutError('completed source parent queue did not finish within six hours')
        time.sleep(10)
    remote(args.destination_host, ['mkdir', '-p', str(args.destination_root)])
    completed = []
    for parent in parents:
        started = time.perf_counter()
        parent_id = parent['parent_id']
        source = args.source_root/parent_id
        destination = args.destination_root/parent_id
        if (parent.get('archive', {}).get('host') == destination_identity
                and Path(parent['checkpoint']).is_relative_to(destination)):
            completed.append(parent)
            continue
        # The source receipt is excluded until both restart files have been
        # verified at the destination and its outer paths have been remapped.
        reader = subprocess.Popen(host_command(args.source_host, [
            'tar', '--exclude='+parent_id+'/parent.json', '-C', str(args.source_root),
            '-cf', '-', parent_id]), stdout=subprocess.PIPE)
        try:
            writer = subprocess.run(host_command(args.destination_host, [
                'tar', '-C', str(args.destination_root), '-xpf', '-']), stdin=reader.stdout)
        finally:
            reader.stdout.close()
        source_code = reader.wait()
        if source_code or writer.returncode:
            raise RuntimeError(f'parent transfer failed: source {source_code}, destination {writer.returncode}')
        def relocate(path):
            return str(destination/Path(path).relative_to(source))
        def verify(checkpoint):
            actual = remote(args.destination_host,
                ['sha256sum', relocate(checkpoint['path'])]).split()[0]
            if actual != checkpoint['sha256']:
                raise ValueError(f'transferred restart SHA changed: {parent_id}')
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(verify, parent['restart_set']))
        archived = {**parent, 'checkpoint': relocate(parent['checkpoint']),
            'prepared_root': relocate(parent['prepared_root']),
            'experiment_config': str(args.destination_root.parent/'case'/Path(parent['experiment_config']).name),
            'restart_set': [{**r, 'path': relocate(r['path'])} for r in parent['restart_set']],
            'archive_manifests': [{**r, 'path': relocate(r['path'])} for r in parent['archive_manifests']],
            'archive': {'host': destination_identity, 'original_host': args.source_host,
                'original_directory': str(source), 'wall_seconds': time.perf_counter()-started,
                'transport': 'uncompressed tar stream through orchestrator; no local weather-data copy',
                'verification': 'SHA-256 of both root and child restart files at destination'}}
        publish = '''import json,sys
from pathlib import Path
path=Path(sys.argv[1]); payload=json.load(sys.stdin)
if path.exists() and json.loads(path.read_text())['checkpoint_sha256'] != payload['checkpoint_sha256']:
 raise ValueError('destination parent belongs to a different checkpoint')
pending=path.with_suffix('.pending'); pending.write_text(json.dumps(payload,indent=2)+'\\n'); pending.replace(path)
'''
        remote(args.destination_host, ['python3', '-c', publish, str(destination/'parent.json')],
               input=json.dumps(archived))
        retire = '''import json,sys,shutil
from pathlib import Path
root=Path(sys.argv[1]); parent=sys.argv[2]; source=root/parent
if source.parent != root or source.name != parent: raise ValueError('unowned source path')
payload=json.load(sys.stdin); marker=root/(parent+'.archived.json')
marker.write_text(json.dumps(payload,indent=2)+'\\n')
shutil.rmtree(source)
'''
        remote(args.source_host, ['python3', '-c', retire, str(args.source_root), parent_id],
               input=json.dumps(archived))
        completed.append(archived)
        (args.out/(parent_id+'.json')).write_text(json.dumps(archived, indent=2)+'\n')
        print(json.dumps({'parent_id': parent_id, 'transferred_seconds': time.perf_counter()-started}), flush=True)
    (args.out/'DONE.json').write_text(json.dumps({'status': 'completed',
        'parent_ids': [p['parent_id'] for p in completed],
        'transport': 'streamed host-to-host through the relay host; no local data spool'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
