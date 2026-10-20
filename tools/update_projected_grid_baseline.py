#!/usr/bin/env python3
"""Refresh the source registry baseline while removing its pairing opt-ins."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision', help='reviewed registry revision to retain')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    revision = subprocess.check_output(['git', 'rev-parse', args.revision], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(['git', 'show', f'{revision}:gpuwm/source_adapters.py'], cwd=ROOT)
    # Retain all current acquisition/decoder metadata while turning off only
    # the exact-grid opt-ins. The marked rows have their own geometry tests.
    source = re.sub(rb'^        same_grid_pairing="identity",\n', b'', source, flags=re.MULTILINE)
    directory = ROOT / 'tests/data'
    baseline_path = directory / 'source_adapter_pairing_baseline.py.gz'
    baseline_bytes = gzip.compress(source, mtime=0)
    path = directory / 'projected_grid_pairing_baseline.json'
    manifest = json.loads(path.read_text())
    manifest['source_adapters_sha256'] = hashlib.sha256(source).hexdigest()
    manifest['source_adapters_rebased_on'] = [row for row in manifest['source_adapters_rebased_on']
                                             if row['commit'] != revision]
    manifest['source_adapters_rebased_on'].append({
        'commit': revision,
        'why': 'Refresh acquired, generated and exported source metadata at the reviewed release candidate. Remove only the source row exact-grid opt-ins; retain the original frame-header oracle.'})
    manifest_bytes = (json.dumps(manifest, indent=2) + '\n').encode()
    if args.check:
        if baseline_path.read_bytes() != baseline_bytes or path.read_bytes() != manifest_bytes:
            parser.exit(1, 'Projected source baseline is stale.\n')
    else:
        baseline_path.write_bytes(baseline_bytes)
        path.write_bytes(manifest_bytes)


if __name__ == '__main__':
    main()
