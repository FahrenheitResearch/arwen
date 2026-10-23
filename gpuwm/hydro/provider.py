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
import threading
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
    if root is not None and (root / spec['relative']).is_file():
        return root / spec['relative']
    if spec.get('template'):
        # The shipped recipe, bound to the acquired source root at preparation.
        return Path(__file__).resolve().parents[1] / 'data' / 'hydro' / spec['template']
    if root is None:
        raise FileNotFoundError('hydro parameter recipe has no source root')
    return root / spec['relative']


def preparation_identity(row):
    """What the preparation depends on: the recipe and gauge files' bytes.

    Their paths are not part of it. Breakage it prevents: the recipe's path
    names the source tree (src-f08fe0b623, src-fe99a1680d), so every new
    checkout or tip rebuilt every domain's fabric and parameters (2.2 GB and
    about 3 minutes for the Oklahoma WoFS box on a 5070 Ti host, 2026-10-21).
    """
    path = recipe_path(row)
    identity = {'parameter_recipe': path.name,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    gauges = json.loads(path.read_text()).get('gauges')
    if gauges:
        identity['gauges'] = {name: {'name': Path(gauges[name]).name,
            'sha256': hashlib.sha256(Path(gauges[name]).read_bytes()).hexdigest()}
            for name in ('table', 'stations', 'observations') if gauges.get(name)}
    return identity


class WaterPrefetch:
    """The initial-water source fetch running beside the fabric's own
    source downloads and build (:func:`start_water_prefetch`)."""

    def __init__(self, *, row, footprint, out, area):
        from .initial_state import prefetch
        policy = row['preparation']
        self.area = area
        self.result = self.error = None
        self.started = time.perf_counter()
        self.finished = None

        def fetch():
            try:
                self.result = prefetch(date=footprint['time_origin'], out=Path(out) / 'initial',
                                       area=area, source=policy['initial_source'],
                                       spinup_days=policy['spinup_days'], progress=lambda _line: None)
            except Exception as error:  # noqa: BLE001 - the preparation fetches again and names it
                self.error = f"{type(error).__name__}: {error}"
            finally:
                self.finished = time.perf_counter()

        self.thread = threading.Thread(target=fetch, name='hydro-water-prefetch', daemon=True)
        self.thread.start()

    def join(self):
        """Wait for the fetch; the stage record of what it did."""
        waited = time.perf_counter()
        self.thread.join()
        record = {'stage': 'initial-water-prefetch', 'seconds': self.finished - self.started,
                  'joined_after_seconds': waited - self.started,
                  'waited_seconds': time.perf_counter() - waited,
                  'area_south_west_north_east': list(self.area)}
        if self.error is not None:
            record['error'] = self.error
        elif self.result is not None:
            record['forcing_days'] = self.result.get('forcing_days')
        return record


def start_water_prefetch(row, footprint, out):
    """Start fetching the initial-water source for ``footprint`` now, before
    the fabric exists; None when the row prepares no spun-up water or the
    footprint has no box.

    Breakage it prevents (2.9.4 R11, 2026-10-22): on a first run in a new
    region `gpuwm go` waited about 5 minutes before step 1, because the 89 ERA5
    forcing days (107-165 s of network extraction) were fetched only after the
    fabric's own sources (45-160 s) were downloaded and the fabric built. The
    fetch reads the footprint's area (initial_state.footprint_area), not the
    fabric, so both now download at once.
    """
    from .initial_state import footprint_area
    policy = row.get('preparation') or {}
    if not policy.get('initial_source') or not policy.get('spinup_days') or 'time_origin' not in footprint:
        return None
    area = footprint_area(footprint)
    if area is None:
        return None
    return WaterPrefetch(row=row, footprint=footprint, out=out, area=area)


def prepare(*, row, footprint, out, config, runner, sources=None, static_ready=False,
            water_prefetch=None):
    """Return the final warm-routing manifest, not an intermediate fabric path.

    ``sources`` binds the recipe template's tokens (gpuwm.hydro.acquire).
    ``static_ready``: the fabric and its parameters were linked in from this
    domain's time-independent cache (gpuwm.hydro.door), so only the
    time-dependent stages run.
    ``water_prefetch``: the initial-water source fetch already running beside
    this preparation (:func:`start_water_prefetch`), joined before the water
    stage reads it.
    """
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
    if static_ready:
        stamp = json.loads((out / 'static-ready.json').read_text())
        gauges = stamp.get('gauges')
        stages.append({'stage': 'parameters', 'seconds': 0.0, 'reused_from': stamp['static_dir']})
        return _prepare_after_parameters(row=row, footprint=footprint, out=out, config=config,
                                         run=run, stages=stages, gauges=gauges, source_recipe=source_recipe,
                                         water_prefetch=water_prefetch)
    if not source_recipe.is_file():
        raise FileNotFoundError(f"hydro parameter source recipe missing: {source_recipe}; "
                                'same-network soil/land-cover source bindings are required')
    recipe = json.loads(source_recipe.read_text())
    if sources:
        from .acquire import bind_recipe
        recipe = bind_recipe(recipe, sources)
    from .acquire import bind_recipe as _bind
    # No geography fallback unless acquisition named one (SoilGrids out of reach).
    recipe = _bind(recipe, {"geog_soilcomp": ""})
    unbound = [token for token in ('{data_root}', '{params_table}', '{soil_prefix}')
               if token in json.dumps(recipe)]
    if unbound:
        raise ValueError(f'hydro parameter recipe {source_recipe} still has {unbound}: '
                         'its sources were not acquired for this domain')
    gauges = recipe.pop("gauges", None)
    # The parameter grid is never finer than the fabric's working grid: the
    # builder coarsens continental and global boxes (rw_hydro_world auto rule
    # `coarsen`), and a 1200-per-degree parameter grid over such a box is
    # 5.7 G cells (about 90 GB of land-cover counts and DEM).
    builder_receipt = out / 'builder-receipt.json'
    if builder_receipt.is_file() and 'cells_per_degree' in recipe:
        grid = json.loads(builder_receipt.read_text()).get('counts', {}).get('grid') or {}
        if grid.get('res_as'):
            recipe['cells_per_degree'] = min(int(recipe['cells_per_degree']),
                                             max(1, round(3600.0 / float(grid['res_as']))))
    # All source rows, classification and calibration recipe choices come from
    # the acquired-source request. Only this forecast's domain/output is bound.
    recipe.update(domain=footprint['grid_key'],
                  bbox=[footprint[k] for k in ('west', 'south', 'east', 'north')],
                  catchments={'kind': 'gpkg', 'path': config['geopackage'],
                              'fabric_out': str(out / 'parameterized.gpkg')},
                  out_dir=str(out / 'parameters'))
    request = _write(out / 'parameter-request.json', recipe)
    run('parameters', [_tool(policy['parameters']), 'derive', request])
    _write(out / 'static-gauges.json', {'gauges': gauges})
    return _prepare_after_parameters(row=row, footprint=footprint, out=out, config=config,
                                     run=run, stages=stages, gauges=gauges, source_recipe=source_recipe,
                                     water_prefetch=water_prefetch)


def _prepare_after_parameters(*, row, footprint, out, config, run, stages, gauges, source_recipe,
                              water_prefetch=None):
    """Projected fabric, initial water, reservoirs and routing: the stages a
    forecast's start time changes."""
    from gpuwm.hydrofabric import prepare as prepare_fabric
    from .initial_state import footprint_area, prepare as prepare_water
    policy = row['preparation']
    config = dict(config, geopackage=str(out / 'parameterized.gpkg'))
    fabric_request = _write(out / 'fabric-request.json', config)
    start = time.perf_counter()
    fabric = prepare_fabric(fabric_request)
    stages.append({'stage': 'projected-fabric', 'seconds': time.perf_counter()-start})
    _write(out / 'provider-stages.json', stages)
    manifest = (Path(fabric['manifest']) if fabric.get('manifest') else
                Path(fabric['output']) / 'manifest.json')
    origin = footprint['time_origin']
    if water_prefetch is not None:
        # Two writers of one forcing-day cache would download the same days twice.
        stages.append(water_prefetch.join())
        _write(out / 'provider-stages.json', stages)
    start = time.perf_counter()
    # Every preparation of a domain reads its source over the footprint's area
    # (what the prefetch fetched) when that holds the fabric, so a later start
    # on the same domain reuses the same cached forcing days.
    warm = prepare_water(fabric=config['geopackage'], date=origin, out=out / 'initial',
                         source=policy['initial_source'], spinup_days=policy['spinup_days'],
                         area_hint=footprint_area(footprint))
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
        'runoff_partition': 'parent LSM surface + subsurface runoff; the realization row names the exchange'})
    return prepared


def _parent_exchange(realization):
    """The realization row's parent exchange (factory, runoff partition)."""
    default = {'factory': 'gpuwm.hydro.noahowp.factory:prepare_exchange',
               'source_partition': 'RUNSRF+RUNSUB; exclude SOLDRAIN'}
    if not realization:
        return default
    from .config import resolve_configuration
    row = resolve_configuration(realization).as_dict()
    return dict(default, **row.get('parent_exchange', {}))


def composition(*, manifest, footprint, realization=None):
    """Create a connected stage list from the actual prepared network metadata."""
    exchange = _parent_exchange(realization)
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
        {'name': 'exchange', 'factory': exchange['factory'],
         'options': {'network_component': 'network', 'weights_sha256': weights_sha,
                     'fabric_sha256': metadata['input']['sha256']}},
        {'name': 'areas', 'factory': 'gpuwm.hydro.runoff:prepare_areas', 'phase': 'prepare',
         'options': {'network_component': 'network', 'array': 'catchment_areasqkm'}},
        {'name': 'runoff', 'factory': 'gpuwm.hydro.runoff:prepare',
         'options': {'source_component': 'exchange', 'network_component': 'network',
                     'area_component': 'areas', 'depth_field': 'runoff_depth_m',
                     'area_array': 'areasqkm', 'area_units': 'km2',
                     'routing_period_seconds': routing_interval,
                     'source_partition': exchange['source_partition']}}]
    if exchange.get('cadence') == 'routing':
        # An accumulated parent (Noah, RUC) differences its totals over any span,
        # so exchange and transfer run once per routing interval: the same water
        # in one launch instead of one per LSM call, and a clock the adaptive
        # parent can drive (gpuwm.hydro.driver.allow_variable_interval).
        for row in stages:
            if row['name'] in ('exchange', 'runoff'):
                row.update(cadence_steps=routing_cadence, interval_seconds=routing_interval)
        stages[1]['options']['interval_seconds'] = routing_interval
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
    # Hourly flow frames; a run shorter than an hour writes one frame at its end instead of
    # refusing at start (breakage 2026-10-22 14:59Z: a 600 s forecast with default hydro
    # stopped on 'history capacity must contain at least one chosen output interval').
    history_cadence = max(1, int(min(3600.0, duration) / routing_interval))
    output_key = metadata['routing'].get('output_reach_ids', 'node_ids')
    count = metadata['files'][output_key]['shape'][0]
    # Reach type is the output schema's series category, not a numerical field.
    stages.append({'name': 'history', 'factory': 'gpuwm.hydro.output:prepare_history',
                   'cadence_steps': routing_cadence, 'interval_seconds': routing_interval,
                   'options': {'source': 'routing', 'origin': footprint['time_origin'],
                               'interval_seconds': routing_interval,
                               'capacity_steps': int(duration / routing_interval),
                               'cadence_steps': history_cadence,
                               'types': ['flowpath'] * count}})
    return {'driver_factory': 'gpuwm.hydro.assembly:prepare', 'stages': stages,
            'time_origin': footprint['time_origin'], 'interval_seconds': interval}
