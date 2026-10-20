#!/usr/bin/env python3
"""Metadata orchestration only: original archive -> Rust subset -> native replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('output', type=Path)
    acquisition = parser.add_mutually_exclusive_group(required=True)
    acquisition.add_argument('--archive', type=Path)
    acquisition.add_argument('--download', action='store_true')
    args = parser.parse_args()
    fixture_path = args.fixture.resolve()
    fixture = json.loads(fixture_path.read_text())
    base = fixture_path.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'tmp').mkdir()
    env = os.environ.copy()
    env.update(TMPDIR=str(output / 'tmp'), RAYON_NUM_THREADS=str(fixture['workers']))
    receipt = {'fixture': str(fixture_path), 'commands': [], 'pass': False,
               'scope': 'original-source geometry selection/decoding, not parent weights or physics'}

    def run(command, label):
        start = time.perf_counter()
        with (output / (label + '.json')).open('w') as stdout, (output / (label + '.stderr')).open('w') as stderr:
            result = subprocess.run(list(map(str, command)), env=env, stdout=stdout, stderr=stderr)
        receipt['commands'].append({'argv': list(map(str, command)), 'exit_code': result.returncode,
                                    'wall_ms': (time.perf_counter() - start) * 1000})
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(result.returncode)

    archive = args.archive.resolve() if args.archive else output / 'source.7z'
    if args.download:
        run(['curl', '--fail', '--silent', '--show-error', '--location', '--max-time', '120',
             '--output', archive, '--dump-header', output / 'download.headers', fixture['archive']['url']], 'download')
    extract_config = [{
        'archive': str(archive), 'sha256': fixture['archive']['sha256'],
        'members': [source['member'] for source in fixture['sources'].values()],
        'output': str(output / 'expanded'), 'max_extracted_bytes': fixture['max_extracted_bytes'],
    }]
    (output / 'extract-config.json').write_text(json.dumps(extract_config, indent=2) + '\n')
    run([base / fixture['archive_binary'], output / 'extract-config.json'], 'extract')
    config = dict(fixture['prepare'])
    for role, source in fixture['sources'].items():
        config[role] = {'path': str(output / 'expanded' / source['member']), 'sha256': source['sha256']}
    config['network'] = {'path': str(base / fixture['network']['path']), 'sha256': fixture['network']['sha256']}
    config['output'] = str(output / 'geometry')
    (output / 'prepare-config.json').write_text(json.dumps(config, indent=2) + '\n')
    run([base / fixture['binary'], 'subset-source-geometry', output / 'prepare-config.json'], 'prepare')
    run([base / fixture['binary'], 'replay-source-geometry', '--strict',
         output / 'geometry/manifest.json', base / fixture['native']], 'compare')
    verdict = json.loads((output / 'compare.json').read_text())
    receipt['pass'] = verdict['pass']
    receipt['canonical_bytes'] = verdict['compared_canonical_bytes']
    # Only this request's spent extraction and optional download are deleted.
    # No numerical work is done in Python; stat/unlink are output management.
    removed = 0
    for path in sorted((output / 'expanded').rglob('*'), reverse=True):
        if path.is_file():
            removed += path.stat().st_size
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    (output / 'expanded').rmdir()
    if args.download:
        removed += archive.stat().st_size
        archive.unlink()
    receipt['removed_spent_bytes'] = removed
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
