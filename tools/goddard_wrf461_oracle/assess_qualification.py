"""Mechanical qualification gates; each box assesses its own small receipts."""
from pathlib import Path
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument('--receipts', type=Path, required=True)
p.add_argument('--stage', choices=('driver', 'forecast', 'all'), default='all')
a = p.parse_args()
def read(name):
    return json.loads((a.receipts / name).read_text())
checks = {}
if a.stage in ('driver', 'all'):
    strict, default = read('driver-strict.json'), read('driver-default.json')
    sw, lw = read('driver-mutant-sw.json'), read('driver-mutant-lw.json')
    inert = read('inert-assessment.json')
    checks.update(
        strict_driver=strict['full_driver_pass'] and strict['columns'] >= 64
            and all(f['max_ulp'] == 0 for f in strict['fields'].values()),
        default_driver=default['full_driver_pass'],
        sw_mutation=sw['mutation_band'] == 1 and not sw['full_driver_pass']
            and any(f['different_words'] for f in sw['fields'].values()),
        lw_mutation=lw['mutation_band'] == 2 and not lw['full_driver_pass']
            and any(f['different_words'] for f in lw['fields'].values()),
        existing_fields_unchanged=inert['pass'] and inert['field_arrays'] >= 100,
    )
if a.stage in ('forecast', 'all'):
    forecast, render = read('forecast-fields.json'), read('render-receipt.json')
    checks.update(
        forecast=len(forecast['frames']) >= 2
            and all(v.get('finite', True) for f in forecast['frames'] for v in f['fields'].values()),
        flux_publication=all(forecast['frames'][-1]['fields'][name]['max'] > 0
            for name in ('GLW', 'SWDOWN', 'SWDDNI', 'OLR')),
        rust_sheet=render['engine'] == 'rust' and render['pass'] and bool(render['images']),
    )
result = {'stage': a.stage, 'pass': all(checks.values()), 'checks': checks}
if a.stage == 'all':
    result['full_port_complete'] = result['pass']
(a.receipts / ('QUALIFICATION-' + a.stage + '.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
raise SystemExit(0 if result['pass'] else 1)
