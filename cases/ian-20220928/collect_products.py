#!/usr/bin/env python3
"""Combine completed member shards without changing weather fields or ancestry.

Copy each shard's ensemble-set.json and members/ diagnostics to a local folder
first. --relocate OLD=NEW remaps diagnostic paths after that copy. Checkpoints
and boundary provenance keep their original identities. All reductions,
regridding and weather pictures are made by rw_wrfbatch, through gpuwm.rustwx.
"""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import time

from gpuwm.ensemble.cyclone_products import render
from gpuwm.ensemble.mass_launcher import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ensemble', type=Path, action='append', required=True)
    parser.add_argument('--relocate', action='append', default=[], metavar='OLD=NEW')
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    relocations = [tuple(value.split('=', 1)) for value in args.relocate]
    def relocate(path):
        for old, new in relocations:
            try:
                return str(Path(new)/Path(path).relative_to(old))
            except ValueError:
                pass
        return path
    completed, sources, parents = [], [], {}
    hours = None
    signatures = set()
    for filename in args.ensemble:
        shard = json.loads(filename.read_text())
        if shard['status'] != 'completed' or len(shard['members']) != len(shard['completed']):
            raise ValueError(f'incomplete shard: {filename}')
        if hours is not None and hours != shard['branch_hours']:
            raise ValueError('member shards must cover the same forecast interval')
        hours = shard['branch_hours']
        for parent in shard['parents']:
            key = str(parent['parent_id'])
            identity = (parent['checkpoint_sha256'], parent['weight'], parent['provenance'])
            if key in parents and parents[key] != identity:
                raise ValueError(f'conflicting ancestry for {key}')
            parents[key] = identity
        for original in sorted(shard['completed'], key=lambda row: row['member_id']):
            if original['status'] != 'completed':
                raise ValueError('failed members cannot enter a probability denominator')
            signature = (original['checkpoint_sha256'], original['seed'])
            if signature in signatures:
                raise ValueError('duplicate parent/perturbation trajectory across shards')
            signatures.add(signature)
            row = copy.deepcopy(original)
            row['collection_source'] = {'ensemble': str(filename), 'member_id': row['member_id'],
                'receipt': row['receipt'], 'original_provenance': copy.deepcopy(row['provenance'])}
            row['member_id'] = len(completed)
            row['receipt'] = relocate(row['receipt'])
            for frame in row['frames']:
                frame['path'] = relocate(frame['path'])
                if not Path(frame['path']).is_file():
                    raise FileNotFoundError(frame['path'])
            row['common_wind']['directory'] = relocate(row['common_wind']['directory'])
            completed.append(row)
        sources.append({'ensemble': str(filename), 'command': shard['command'],
            'latest_invocation_wall_seconds': shard.get('wall_seconds'),
            'previous_invocations': shard.get('previous_invocations', []),
            'forecast_wall_seconds': shard.get('forecast_wall_seconds'),
            'members_per_card_hour': shard.get('members_per_card_hour'),
            'reused_completed_members': shard.get('reused_completed_members'),
            'newly_completed_members': shard.get('newly_completed_members'),
            'cards': shard['cards'], 'branch_hours': shard['branch_hours']})
    counts = Counter(str(row['parent_id']) for row in completed)
    # Parent manifests retain their prior mass (1/36 for this case). A proof
    # that represents eight parents is explicitly conditional on those eight.
    represented = math.fsum(float(parents[key][1]) for key in counts)
    prior_mass = math.fsum(float(value[1]) for value in parents.values())
    for row in completed:
        key = str(row['parent_id'])
        mass = float(parents[key][1])
        row['weight'] = mass/counts[key]/represented
        row['provenance']['ancestry'] = {'parent_weight': mass,
            'siblings': counts[key], 'represented_parent_mass': represented,
            'child_weight': row['weight']}
        from gpuwm.ensemble.wofs_recipe import provenance_label
        row['provenance']['label'] = provenance_label(row['provenance'])
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {'schema': 'gpuwm-checkpoint-ensemble-collection.v1', 'status': 'rendering',
        'completed': completed, 'sources': sources, 'branch_hours': hours,
        'represented_ancestors': len(counts), 'represented_prior_mass': represented,
        'manifest_prior_mass': prior_mass, 'probability_scope': 'conditional on represented ancestors'}
    request = json.loads(args.request.read_text())
    for key in ('best_track', 'observed_radar'):
        if request.get(key):
            request[key] = str((args.request.parent/Path(request[key])).resolve())
    request['defer_render'] = False
    atomic_json(args.out/'ensemble-set.json', receipt)
    receipt['products'] = render(args.out, receipt, request)
    receipt.update(status='completed', collection_and_render_wall_seconds=time.perf_counter()-started)
    atomic_json(args.out/'ensemble-set.json', receipt)
    print(json.dumps(receipt['products']))


if __name__ == '__main__':
    main()
