#!/usr/bin/env python3
"""Author the fixed Ian T-12 h experiment using cyclone-setup's real tree."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os

# Prefer the checkout that owns this case over an unrelated editable install.
_CHECKOUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CHECKOUT))
os.environ["PYTHONPATH"] = str(_CHECKOUT) + os.pathsep + os.environ.get("PYTHONPATH", "")
import time
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--geog-root', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = args.out / 'cyclone-starter.toml'
    command = [sys.executable, '-B', '-m', 'gpuwm', 'cyclone-setup', '--source', 'gefs',
        '--member', 'c00', '--cycle', '2022-09-28T00', '--point=25.9,-82.9',
        '--hours', '27', '--card', '96gb', '--tiles', 'off', '--name', 'ian-t12', '--out', str(base)]
    if not base.exists():
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        (args.out / 'cyclone-author.stdout.json').write_text(result.stdout)
        (args.out / 'cyclone-author.stderr.log').write_text(result.stderr)
    from gpuwm.toml_document import emit_experiment_toml
    from gpuwm.experiment import load_experiment
    template = tomllib.loads(base.read_text())
    roster = [('2022-09-28T00', member, 0) for member in ['c00'] + [f'p{i:02d}' for i in range(1, 31)]]
    roster += [('2022-09-27T18', f'p{i:02d}', 6) for i in range(1, 6)]
    def author(item):
        index, (cycle, member, lag) = item
        parent_id = f'{cycle.replace("-", "").replace(":", "")}-{member}'
        raw = copy.deepcopy(template)
        raw['experiment'].update(name=f'ian-{parent_id}', run_seconds=25500., restart_interval_s=25500.)
        raw['fetch'].update(cycle=cycle, member=member, forecast_start_hour=lag, hours=27,
                            out=f'data/{parent_id}')
        raw['domain'][0]['history_interval_s'] = 25500.
        child = raw['domain'][1]
        child.update(nx=480, ny=420, i_parent_start=41, j_parent_start=28, history_interval_s=25500.)
        child['follow'].update(field='pressure', threshold=1100., level_hpa=0.,
            pressure_center='minimum', fallback_level_hpa=850., fallback_deficit_m=25.,
            max_shift_cells=31, max_move_parent_cells=31)
        path = args.out / f'{parent_id}.toml'
        text = emit_experiment_toml(raw)
        path.write_text(text)
        # Parser is the authority for dimensions, placement, clocks and physics.
        exp = load_experiment(path)
        assert [(d.run.nx, d.run.ny, d.run.nz) for d in exp.domains] == [(200, 160, 49), (480, 420, 49)]
        assert exp.domains[1].run.dt == 15.
        return {'parent_id': parent_id, 'config': path.name, 'config_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'weight': 1/36, 'provenance': {'source': 'gefs', 'cycle': cycle, 'source_member': member,
                'lag_hours': lag, 'physics_arm': 'base'}, 'initial_lead_h': lag,
            'forcing_leads_h': list(range(lag, lag+28, 3))}
    with ThreadPoolExecutor(max_workers=8) as pool:
        parents = list(pool.map(author, enumerate(roster)))
    payload = {'schema': 'ian-gefs-roster.v1', 'analysis_time': '2022-09-28T07:05:00Z',
        'observation_cutoff': '2022-09-28T07:05:00Z', 'spinup_start': '2022-09-28T00:00:00Z',
        'forecast_end': '2022-09-29T01:05:00Z', 'boundary_bracket_end': '2022-09-29T03:00:00Z',
        'landfall_reference': '2022-09-28T19:05:00Z', 'parents': parents,
        'author_seconds': time.perf_counter()-started, 'parallelism': '8 host workers after cyclone-setup',
        'seed_position': {'latitude': 25.9, 'longitude': -82.9, 'kind': 'explicit seed; not measured centre'}}
    (args.out / 'GEFS-ROSTER.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({'parents': len(parents), 'wall_seconds': payload['author_seconds'], 'out': str(args.out)}))

if __name__ == '__main__':
    main()
