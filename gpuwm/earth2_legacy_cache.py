"""Validate the original single-member acquisition contract for cache reuse."""
import json
import os
from pathlib import Path
import subprocess


def default_identity(row, version):
    """Inspect installed defaults without loading weights or initializing CUDA."""
    from gpuwm.earth2_run import earth2_python
    script = '''import inspect,json,hashlib
from types import SimpleNamespace
from pathlib import Path
from importlib.metadata import version
import earth2studio.models.px as px
row=json.loads(__import__('sys').argv[1])
cls=getattr(px,row['earth2studio_class'])
method=inspect.unwrap(cls.load_default_package)
class Package:
 def __init__(self,root,**kwargs):self.root=root
 @staticmethod
 def default_cache(*args):return 'metadata-only-no-io'
method.__globals__['Package']=Package
package=cls.load_default_package()
seed=inspect.signature(cls).parameters.get('seed')
source=Path(inspect.getfile(cls))
import torch
print(json.dumps(dict(package=package.root,seed=seed.default if seed else None,cuda_initialized=torch.cuda.is_initialized(),
 earth2studio=version('earth2studio'),class_source=str(source),
 class_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                       HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    try:
        probe = subprocess.run([earth2_python(), '-c', script, json.dumps(row)], env=environment,
                               text=True, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        raise ValueError('legacy fetch default identity check exceeded 120 seconds') from error
    if probe.returncode:
        raise ValueError('legacy fetch default identity could not be checked: ' + probe.stderr[-2000:])
    identity = json.loads(probe.stdout.strip().splitlines()[-1])
    if identity['cuda_initialized'] or identity['earth2studio'] != version or identity['package'] != row['weights']['uri']:
        raise ValueError('legacy fetch runtime version or public weight revision differs')
    policy = row.get('members')
    if policy and (policy['kind'] != 'native_noise' or
                   identity['seed'] != policy['first_seed'] or
                   policy.get('perturbation', {}).get('class') != 'Zero'):
        raise ValueError('legacy fetch does not prove the requested member policy')
    return identity


def verify(spec, receipt, request, *, allow_regional=False):
    """Keep the original receipt; add checked compatibility evidence only."""
    from gpuwm.earth2_run import resolve_model, model_table
    from gpuwm.tc_tracks import inventory, cf_seconds
    _, row = resolve_model(spec['model'])
    table = model_table()
    ic = table['runtime']['initial_condition_sources'][spec['ic_source']]
    old = receipt['model_row']
    for key in ('earth2studio_class', 'profile', 'step_hours'):
        if old.get(key) != row.get(key):
            raise ValueError(f'legacy fetch {key} differs')
    if request['model_class'] != row['earth2studio_class'] or request['ic_class'] != ic['earth2studio_class']:
        raise ValueError('legacy fetch model or IC class differs')
    if request['steps'] * row['step_hours'] != receipt['hours']:
        raise ValueError('legacy fetch request horizon differs from receipt')
    if receipt['runner']['earth2studio'] != table['runtime']['pinned_version']:
        raise ValueError('legacy fetch runtime version differs')
    for key in ('loader_kwargs', 'load_model_kwargs', 'ic_kwargs', 'model_calls', 'loader_preparations', 'runtime_env'):
        if row.get(key):
            raise ValueError(f'legacy fetch cannot prove nondefault {key}')
    if ic.get('kwargs'):
        raise ValueError('legacy fetch cannot prove nondefault IC arguments')
    identity = default_identity(row, table['runtime']['pinned_version'])
    metadata = inventory(receipt['forecast'])
    variables = metadata['variables']
    times = variables['time']
    initial = cf_seconds(times['values'], times['attrs']['units'], True)
    from datetime import datetime
    wanted = datetime.fromisoformat(spec['cycle'].replace('Z', '+00:00')).timestamp()
    if initial != [wanted]:
        raise ValueError('legacy fetch NetCDF initialization differs')
    lead = variables['lead_time']
    seconds = cf_seconds(lead['values'], lead['attrs']['units'], False)
    expected = list(range(0, spec['hours'] + 1, int(row['step_hours'])))
    if [value / 3600 for value in seconds if 0 <= value <= spec['hours'] * 3600] != expected:
        raise ValueError('legacy fetch NetCDF lead ladder differs')
    if 'ensemble' in variables and len(variables['ensemble']['values']) != 1:
        raise ValueError('legacy fetch has more than one member')
    if request['area'] is None or (spec['area'] is None and not allow_regional):
        lat, lon = variables['lat']['values'], variables['lon']['values']
        if min(lat) > -89.75 or max(lat) < 89.75 or min(lon) > 0 or max(lon) < 359.75:
            raise ValueError('legacy fetch native coordinates do not cover its declared global area')
    if request['area'] is not None:
        from gpuwm.earth2_fetch_cache import covers_area
        lat = variables['lat']['values']
        lon = sorted(set(float(v) % 360 for v in variables['lon']['values']))
        gaps = [(lon[(i + 1) % len(lon)] - value) % 360 for i, value in enumerate(lon)]
        gap = max(range(len(lon)), key=gaps.__getitem__)
        west = lon[(gap + 1) % len(lon)]
        east = west + 360 - gaps[gap]
        # Crop endpoints may fall between cells on the regular 0.25 degree grid.
        actual = [min(lat) - .25, west - .25, max(lat) + .25, east + .25]
        if not covers_area(actual, request['area']):
            raise ValueError('legacy fetch native coordinates do not cover its declared area')
    return dict(runtime_defaults=identity, native_inventory_schema=metadata.get('schema'),
                checked_leads_hours=expected, member=0,
                model_row_comparison='class, profile, cadence, pinned package and default native seed',
                original_model_row=old)
