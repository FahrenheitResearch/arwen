#!/usr/bin/env python3
"""Verify all 36 independent parents and retain archive coverage evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ENGINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINE))
os.chdir(ENGINE)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--parents', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    args.case, args.parents, args.evidence = (p.resolve() for p in (args.case, args.parents, args.evidence))
    roster = json.loads((args.case/'GEFS-ROSTER.json').read_text())
    from gpuwm.io.restart import read_restart_header
    from gpuwm.prepared_domain_tree_forecast import _runtime_source_identity
    from gpuwm.ingest.boundary_stream import bind_head, prepared_tree_complete
    from gpuwm.ensemble.mass_launcher import atomic_json
    identity = dict(_runtime_source_identity())
    expected_end = datetime.fromisoformat(roster['boundary_bracket_end'].replace('Z', '+00:00'))
    expected_start = datetime.fromisoformat(roster['spinup_start'].replace('Z', '+00:00'))
    args.evidence.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    def verify(expected):
        parent = json.loads((args.parents/expected['parent_id']/'parent.json').read_text())
        if parent['provenance'] != expected['provenance'] or parent['weight'] != expected['weight']:
            raise ValueError(f'ancestry changed: {expected["parent_id"]}')
        if sha(parent['experiment_config']) != expected['config_sha256']:
            raise ValueError(f'configuration changed: {expected["parent_id"]}')
        head = bind_head(parent['prepared_root'], parent['prepared_head_sha256'])
        if not prepared_tree_complete(Path(parent['prepared_root'])):
            raise ValueError('independent forcing preparation has not sealed')
        forcing_times = [datetime.fromisoformat(t.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
                         for t in head['basis']['proof_head']['forcing_times']]
        wanted_times = [expected_start+timedelta(hours=h) for h in range(0, 28, 3)]
        if forcing_times != wanted_times:
            raise ValueError('sealed parent forcing does not cover the complete forecast window')
        checkpoints = parent['restart_set']
        if len(checkpoints) != 2:
            raise ValueError('the following tree requires exactly root and child checkpoints')
        if not any(r['path'] == parent['checkpoint'] and r['sha256'] == parent['checkpoint_sha256'] for r in checkpoints):
            raise ValueError('branch checkpoint is not in the verified complete restart set')
        headers = []
        for checkpoint in checkpoints:
            if sha(checkpoint['path']) != checkpoint['sha256']:
                raise ValueError(f'checkpoint SHA changed: {checkpoint["path"]}')
            header = read_restart_header(checkpoint['path'])
            if header['elapsed_seconds'] != 25500.:
                raise ValueError('parent checkpoint is not at the 07:05Z analysis')
            actual = header['experiment_fingerprint_components']['runtime_source_identity']
            if actual != identity:
                raise ValueError(f'parent uses a different installed engine: {expected["parent_id"]}: {actual}')
            headers.append(header)
        if {h['grid_id'] for h in headers} != {1, 2} or len({h['checkpoint_set_id'] for h in headers}) != 1:
            raise ValueError('root and child do not form one saved tree')
        sources, public_sources = [], []
        for index, manifest in enumerate(parent['archive_manifests']):
            path = Path(manifest['path'])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
                raise ValueError(f'archive receipt changed: {path}')
            fetched = json.loads(raw)
            request = fetched['request']
            if not fetched['complete'] or request['member'] != expected['provenance']['source_member']:
                raise ValueError('incomplete or wrong source member')
            cycle = datetime.fromisoformat(request['cycle'].replace('Z', '+00:00'))
            wanted_cycle = datetime.fromisoformat(expected['provenance']['cycle']).replace(tzinfo=timezone.utc)
            leads = sorted(request['leads'])
            if cycle != wanted_cycle or leads != expected['forcing_leads_h']:
                raise ValueError('archive forcing cycle or forecast leads changed')
            if cycle+timedelta(hours=min(leads)) != expected_start or cycle+timedelta(hours=max(leads)) != expected_end:
                raise ValueError('forcing does not cover the full independent trajectory')
            destination = args.evidence/'sources'/expected['parent_id']/f'fetch-{index:02}.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            sources.append({'receipt': str(destination.relative_to(args.evidence)), 'sha256': manifest['sha256'],
                            'objects': len(fetched['files']), 'composed_files': len(fetched.get('composed', [])),
                            'forcing_leads_h': leads, 'coverage_start': expected_start.isoformat(),
                            'coverage_end': expected_end.isoformat()})
            public_sources.append({'manifest_sha256': manifest['sha256'],
                'route_table_sha256': fetched['route_table_sha256'], 'request': request,
                'objects': [{k: row[k] for k in ('name', 'bytes', 'sha256', 'url', 'role', 'lead')}
                            for row in fetched['files']],
                'composed': [{k: row[k] for k in ('name', 'bytes', 'sha256', 'lead', 'parts')}
                             for row in fetched.get('composed', [])]})
        if not sources:
            raise ValueError('parent has no actual GEFS archive receipt')
        return parent, {'parent_id': parent['parent_id'], 'checkpoint_set_id': headers[0]['checkpoint_set_id'],
            'checkpoints': checkpoints, 'prepared_head_sha256': parent['prepared_head_sha256'],
            'sealed_forcing_valid_times': [t.isoformat() for t in forcing_times],
            'configuration_sha256': expected['config_sha256'], 'sources': sources,
            'stage_wall_seconds': parent['stage_wall_seconds'], 'card': parent['card']}, {
            'parent_id': parent['parent_id'], 'provenance': parent['provenance'],
            'configuration_sha256': expected['config_sha256'],
            'coverage_start': expected_start.isoformat(), 'boundary_bracket_end': expected_end.isoformat(),
            'sources': public_sources}
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        verified = list(pool.map(verify, roster['parents']))
    parents, coverage, public_sources = zip(*verified)
    if (len({p['checkpoint_sha256'] for p in parents}) != len(parents)
            or len({p['prepared_head_sha256'] for p in parents}) != len(parents)):
        raise ValueError('independent ancestors must not alias one checkpoint or boundary stream')
    payload = {'schema': 'gpuwm-checkpoint-parents.v1', 'kind': 'NO-DA GEFS spun independently; no observations assimilated',
        **{k: roster[k] for k in ('analysis_time', 'observation_cutoff', 'forecast_end')},
        'parents': parents, 'roster_ancestors': 36, 'represented_ancestors': len(parents)}
    if len(parents) != 36:
        raise ValueError('the Ian production roster requires 36 parents')
    atomic_json(args.parents/'PARENTS.json', payload)
    atomic_json(args.case/'GEFS-ARCHIVE.json', {'schema': 'ian-gefs-archive.v1',
        'analysis_time': roster['analysis_time'], 'forecast_end': roster['forecast_end'],
        'parents': public_sources, 'note': 'Actual acquired source-object and composed-pair SHA-256; no local paths.'})
    atomic_json(args.evidence/'GEFS-COVERAGE.json', {'schema': 'ian-gefs-coverage.v1', 'parents': coverage,
        'runtime_source_identity': identity, 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'verification_wall_seconds': time.perf_counter()-started,
        'parallelism': '8 independent checkpoint SHA and header validation workers',
        'forecast_end': roster['forecast_end'], 'boundary_bracket_end': roster['boundary_bracket_end']})
    print(json.dumps({'parents': len(parents), 'wall_seconds': time.perf_counter()-started,
                      'manifest': str(args.parents/'PARENTS.json')}))


if __name__ == '__main__':
    main()
