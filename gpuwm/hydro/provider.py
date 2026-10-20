"""Prepare a physical coupled provider using the ordinary Rust stage tools.

Python passes requests and immutable identities. Rust derives parameters, maps
reanalysis state, spins up water stores, collapses reservoirs and prepares CSR.
No example-packet parameters or relabelled catchment axes are synthesized here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
import time


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return path


def _tool(spec):
    from .door import _executable
    path = _executable(spec)
    if path is None:
        raise FileNotFoundError(f"provider preparation needs {spec['filename']} ({spec['env']})")
    return str(path)


def recipe_path(row):
    """Resolve the acquired source recipe without inventing source locations."""
    from .door import _builder_path
    spec = row['preparation']['parameter_recipe']
    configured = os.environ.get(spec['env'])
    if configured:
        return Path(configured).expanduser().resolve()
    root = next(_builder_path(need) for need in row['builder']['paths']
                if need.get('argument') == 'data_root')
    if root is None:
        raise FileNotFoundError('hydro parameter recipe has no source root')
    return root / spec['relative']


def preparation_identity(row):
    path = recipe_path(row)
    identity = {'parameter_recipe': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    gauges = json.loads(path.read_text()).get('gauges')
    if gauges:
        identity['gauges'] = {name: {'path': str(Path(gauges[name]).resolve()),
            'sha256': hashlib.sha256(Path(gauges[name]).read_bytes()).hexdigest()}
            for name in ('table', 'stations', 'observations') if gauges.get(name)}
    return identity


def prepare(*, row, footprint, out, config, runner):
    """Return the final warm-routing manifest, not an intermediate fabric path."""
    from gpuwm.hydrofabric import prepare as prepare_fabric
    from .initial_state import prepare as prepare_water
    policy = row['preparation']
    out = Path(out)
    stages = []

    def run(name, argv, *, json_output=True):
        start = time.perf_counter()
        result = runner(list(map(str, argv)), capture_output=True, text=True,
                        timeout=int(policy.get('timeout_seconds', 3600)))
        (out / (name + '.stderr')).write_text(result.stderr)
        (out / (name + '.stdout')).write_text(result.stdout)
        stages.append({'stage': name, 'argv': list(map(str, argv)),
                       'seconds': time.perf_counter() - start, 'returncode': result.returncode})
        _write(out / 'provider-stages.json', stages)
        if result.returncode:
            raise RuntimeError(f"hydro provider {name}: {(result.stderr or result.stdout)[-2000:]}")
        return json.loads(result.stdout) if json_output else None

    source_recipe = recipe_path(row)
    if not source_recipe.is_file():
        raise FileNotFoundError(f"hydro parameter source recipe missing: {source_recipe}; "
                                'same-network soil/land-cover source bindings are required')
    recipe = json.loads(source_recipe.read_text())
    gauges = recipe.pop("gauges", None)
    # All source rows, classification and calibration recipe choices come from
    # the acquired-source request. Only this forecast's domain/output is bound.
    recipe.update(domain=footprint['grid_key'],
                  bbox=[footprint[k] for k in ('west', 'south', 'east', 'north')],
                  catchments={'kind': 'gpkg', 'path': config['geopackage'],
                              'fabric_out': str(out / 'parameterized.gpkg')},
                  out_dir=str(out / 'parameters'))
    request = _write(out / 'parameter-request.json', recipe)
    run('parameters', [_tool(policy['parameters']), 'derive', request])
    config = dict(config, geopackage=str(out / 'parameterized.gpkg'))
    fabric_request = _write(out / 'fabric-request.json', config)
    start = time.perf_counter()
    fabric = prepare_fabric(fabric_request)
    stages.append({'stage': 'projected-fabric', 'seconds': time.perf_counter()-start})
    _write(out / 'provider-stages.json', stages)
    manifest = (Path(fabric['manifest']) if fabric.get('manifest') else
                Path(fabric['output']) / 'manifest.json')
    origin = footprint['time_origin']
    start = time.perf_counter()
    warm = prepare_water(fabric=config['geopackage'], date=origin, out=out / 'initial',
                         source=policy['initial_source'], spinup_days=policy['spinup_days'])
    stages.append({'stage': 'initial-water', 'seconds': time.perf_counter()-start,
                   'source': policy['initial_source'], 'receipt': warm})
    _write(out / 'provider-stages.json', stages)
    initial_dir = Path(warm['output_directory']) / 'spinup'
    if not (initial_dir / 'manifest.json').is_file():
        raise RuntimeError('provider initialization produced no complete spin-up state')
    tool = _tool(policy['hydrofabric'])
    routing = {'fabric_manifest': str(manifest), 'output': str(out / 'routing'),
               'initial_state_dir': str(initial_dir), 'time_origin': origin,
               'dt_seconds': policy['routing_interval_seconds'], 'data_license': config['data_license']}
    metadata = json.loads(manifest.read_text())
    has_lakes = any(layer['role'] == 'lakes' and layer['row_count']
                    for layer in metadata.get('layers', ()))
    if has_lakes:
        request = _write(out / 'reservoir-initial-request.json', {
            'fabric_manifest': str(manifest), 'initial_state_dir': str(initial_dir),
            'time_origin': origin, 'vertical_datum': policy['vertical_datum'],
            'output': str(out / 'reservoir-initial.json')})
        run('reservoir-initial', [tool, 'reservoir-initial-from-spinup', request])
        run('reservoir-prepare', [_tool(policy['reservoirs']), '--fabric', manifest,
                       out / 'reservoir-initial.json', out / 'reservoirs'])
        routing['reservoir_manifest'] = str(out / 'reservoirs' / 'manifest.json')
        if not Path(routing['reservoir_manifest']).is_file():
            raise RuntimeError('reservoir preparation wrote no manifest')
    request = _write(out / 'routing-request.json', routing)
    prepared = run('routing', [tool, 'prepare-routing-world', request])
    final = Path(prepared['manifest'])
    if not final.is_file():
        raise RuntimeError('warm routing preparation wrote no manifest')
    if gauges:
        match_out = out / 'gauge-match'
        argv = [_tool(policy['gauges']), 'match', gauges['table'], config['geopackage'],
                gauges['stations'], match_out]
        if gauges.get('observations'):
            argv.append(gauges['observations'])
        run('gauge-match', argv, json_output=False)
        shutil.copyfile(match_out / 'gauges.json', final.parent / 'gauges.json')
    _write(out / 'provider-binding.json', {
        'parameter_recipe_sha256': hashlib.sha256(source_recipe.read_bytes()).hexdigest(),
        'parent_grid_contract': footprint['parent_grid_contract'],
        'time_origin': origin, 'routing_manifest': str(final),
        'initial_state': str(initial_dir / 'manifest.json'), 'stages': stages,
        'state_method': 'reanalysis plus soil replay; complete historical runoff routed through MC and reservoirs at admitted GPU startup',
        'runoff_partition': 'parent Noah-MP RUNSRF+RUNSUB; SOLDRAIN excluded'})
    return prepared


def composition(*, manifest, footprint):
    """Create a connected stage list from the actual prepared network metadata."""
    manifest = Path(manifest)
    metadata = json.loads(manifest.read_text())
    contract = metadata.get('parent_grid_contract')
    if contract != footprint['parent_grid_contract']:
        raise ValueError('warm routing lost the exact forecast parent-grid identity')
    if 'initial_state_array' not in metadata.get('routing', {}):
        raise ValueError('coupled provider requires a complete warm channel state')
    interval = float(footprint['interval_seconds'])
    routing_interval = float(metadata['routing']['dt_seconds'])
    duration = float(footprint['duration_seconds'])
    if interval <= 0 or duration < interval or duration % interval:
        raise ValueError('forecast duration must contain whole hydro intervals')
    if routing_interval < interval or routing_interval % interval or duration % routing_interval:
        raise ValueError('routing period and forecast duration must contain whole LSM/routing intervals')
    routing_cadence = int(routing_interval / interval)
    payloads = metadata['files']
    weights_sha = payloads['exchange_weights']['sha256']
    stages = [
        {'name': 'network', 'factory': 'gpuwm.hydrofabric:prepare_resource', 'phase': 'prepare',
         'options': {'manifest': str(manifest)}},
        {'name': 'exchange', 'factory': 'gpuwm.hydro.noahowp.factory:prepare_exchange',
         'options': {'network_component': 'network', 'weights_sha256': weights_sha,
                     'fabric_sha256': metadata['input']['sha256']}},
        {'name': 'areas', 'factory': 'gpuwm.hydro.runoff:prepare_areas', 'phase': 'prepare',
         'options': {'network_component': 'network', 'array': 'catchment_areasqkm'}},
        {'name': 'runoff', 'factory': 'gpuwm.hydro.runoff:prepare',
         'options': {'source_component': 'exchange', 'network_component': 'network',
                     'area_component': 'areas', 'depth_field': 'runoff_depth_m',
                     'area_array': 'areasqkm', 'area_units': 'km2',
                     'routing_period_seconds': routing_interval,
                     'source_partition': 'RUNSRF+RUNSUB; exclude SOLDRAIN'}}]
    routing = {'network_component': 'network', 'lateral_component': 'runoff',
               'time_origin': footprint['time_origin'], 'publish_channels': True,
               'dt_seconds': routing_interval, 'continuous_initialization': True,
               'replay_manifest': str(manifest)}
    if 'reservoirs' in metadata:
        stages.append({'name': 'reservoirs', 'factory': 'gpuwm.hydro.reservoirs:prepare',
                       'phase': 'prepare', 'options': {'network_component': 'network'}})
        routing['reservoir_component'] = 'reservoirs'
    stages.append({'name': 'routing', 'factory': 'gpuwm.hydro.troute_mc:prepare',
                   'launch_method': 'launch', 'options': routing,
                   'cadence_steps': routing_cadence, 'interval_seconds': routing_interval})
    output_key = metadata['routing'].get('output_reach_ids', 'node_ids')
    count = metadata['files'][output_key]['shape'][0]
    # Reach type is the output schema's series category, not a numerical field.
    stages.append({'name': 'history', 'factory': 'gpuwm.hydro.output:prepare_history',
                   'cadence_steps': routing_cadence, 'interval_seconds': routing_interval,
                   'options': {'source': 'routing', 'origin': footprint['time_origin'],
                               'interval_seconds': routing_interval,
                               'capacity_steps': int(duration / routing_interval),
                               'cadence_steps': max(1, int(3600 / routing_interval)),
                               'types': ['flowpath'] * count}})
    return {'driver_factory': 'gpuwm.hydro.assembly:prepare', 'stages': stages,
            'time_origin': footprint['time_origin'], 'interval_seconds': interval}
