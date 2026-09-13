"""Launch a reviewed local configuration through the regional cycle engine.

No forecast integrator, observation decoder or restart publisher lives here.
The backend composes the existing preparation, member and analysis paths.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import time
import numpy as np

from gpuwm.local_da import SCHEMA, PlanError, canonical, digest, utc


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_plan(path):
    """Validate saved review, generated configuration and explicit inputs."""
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if document.get('schema') != SCHEMA:
        raise PlanError('The saved plan has a different schema; regenerate it with the local DA form.', code='PLAN_SCHEMA')
    unsigned = {k: v for k, v in document.items() if k not in ('review_sha256', 'files')}
    if digest(unsigned) != document.get('review_sha256'):
        raise PlanError('The saved review digest changed; restore the reviewed document or generate a new plan.', code='REVIEW_CHANGED')
    from gpuwm.local_da import PUBLISHED_FILES
    expected_files = {name: key for name, key in PUBLISHED_FILES}
    if set(document.get('files', {})) != set(expected_files):
        raise PlanError('The saved configuration roster is incomplete; republish into a new directory.', code='CONFIGURATION_CHANGED')
    for name, key in expected_files.items():
        expected = hashlib.sha256(document['configuration'][key].encode()).hexdigest()
        if document['files'][name] != expected or _sha(path.parent / name) != expected:
            raise PlanError(f'{name} differs from the reviewed configuration; restore it or generate a new plan.', code='CONFIGURATION_CHANGED')
    for name, expected in document.get('inputs', {}).items():
        if _sha(name) != expected:
            raise PlanError(f'Observation input {name} changed after review; regenerate the plan for these bytes.', code='OBSERVATION_CHANGED')
    from gpuwm.ensemble.config import load_ensemble_config
    from gpuwm.experiment import load_experiment
    ensemble = load_ensemble_config(path.parent / 'ensemble.toml')
    experiment = load_experiment(path.parent / 'experiment.toml')
    if ensemble.n_members != document['selected']['members'] or len(experiment.domains) != 1:
        raise PlanError('The generated roster or domain count contradicts the review; regenerate the plan.', code='ROSTER_CHANGED')
    if 'background' in document:
        from gpuwm.regional_preparation import validate_saved_background
        validate_saved_background(document, path.parent / 'experiment.toml')
    return document


def _atomic(path, value):
    from gpuwm.ensemble.manifest import write_json_atomically
    write_json_atomically(Path(path), value)


class ProductFailure(RuntimeError):
    """A completed integration whose ordinary product stage did not finish."""
    def __init__(self, products):
        self.products = products
        reasons = [str(row.get('reason', 'No product was published.'))
                   for row in products['members'] if row['status'] != 'complete']
        if not reasons:
            reasons = ['No member products were published.']
        super().__init__('Local forecast products are incomplete: ' + '; '.join(reasons)
                         + ' Resolve the reported output problem and start this case again.')


def _forecast_output_root(root: Path) -> Path:
    """Preserve completed outputless attempts while recovering their forecast."""
    from gpuwm.ensemble.manifest import read_manifest, ENSEMBLE_MANIFEST_SCHEMA
    attempt = 0
    while True:
        name = 'forecast' if attempt == 0 else f'forecast-output-{attempt}'
        directory = root / name
        manifest = directory / 'ensemble-manifest.json'
        if not manifest.is_file():
            return directory
        document = read_manifest(manifest, schema=ENSEMBLE_MANIFEST_SCHEMA)
        members = document.get('members', [])
        if (document.get('status') != 'COMPLETE' or not members
                or any(row.get('status') != 'DONE' for row in members)
                or not any(row.get('wrfout_inventory') == [] for row in members)):
            return directory
        attempt += 1


def observation_usage(method):
    """Summarize masks consumed by the analysis, separately from acquisition.

    The analysis owner's innovation batches count post-QC/post-thinning
    observations. Missing statistics stay unknown, never inferred from a
    configured stream or downloaded file. Zero is explicit for forecast-only.
    """
    from gpuwm.da.obs_goes import CWP_NAME
    innovations = method.get('innovations')
    batches = ([] if innovations is None else
        [dict(name=row['name'], accepted=int(row['observations'])) for row in innovations])
    count = sum(row['accepted'] for row in batches) if innovations is not None else (
        0 if method.get('method') == 'forecast-only' else None)
    return dict(accepted_for_analysis=count, batches=batches,
        cwp_accepted=None if count is None else sum(row['accepted'] for row in batches if row['name'] == CWP_NAME),
        basis='analysis innovation masks after QC and thinning; no forecast-skill claim',
        reason=method.get('reason'), routes=[{key: row[key] for key in
            ('id', 'route', 'status', 'reason', 'observed_columns', 'acquisition', 'latency_class') if key in row}
            for row in method.get('routes', ()) if 'id' in row or 'route' in row])


def _completed_observation_usage(manifest_path):
    from gpuwm.ensemble.manifest import read_manifest, CYCLE_MANIFEST_SCHEMA
    if not Path(manifest_path).is_file():
        return []
    manifest = read_manifest(manifest_path, schema=CYCLE_MANIFEST_SCHEMA)
    records = []
    for cycle in manifest['cycles']:
        if cycle['status'] != 'DONE' or not cycle.get('assimilation'):
            continue
        method = cycle['assimilation']['method'].get('provenance') or {}
        records.append(dict(cycle=cycle['cycle'], **observation_usage(method)))
    return records



SCRATCH_ENV = 'GPUWM_LOCAL_DA_SCRATCH_MIB'


def execution_scratch_override(environ=None):
    """Validate an execution-only override before preparation or writes."""
    value = (os.environ if environ is None else environ).get(SCRATCH_ENV)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float('nan')
    if not math.isfinite(result) or result <= 0:
        raise PlanError(f'{SCRATCH_ENV} must be a finite positive number of MiB; correct or unset this execution override.')
    return result


def analysis_execution_budget(planned_mib, *, override_mib=None, capacity=None):
    """Price bounded execution scratch from current resources, not the plan floor.

    One quarter of available device capacity leaves room for operators and
    other allocations. One eighth of available host memory covers geometry
    plus packing, transfers and output work without promising all free RAM.
    These are execution scheduling allowances, not admission or science.
    LETKF still enforces its explicit scratch and actual device ceilings.
    """
    if capacity is None:
        from gpuwm.core.preflight import host_available_bytes
        from gpuwm.da.letkf import _device_capacity
        try:
            import cupy as cp
            driver, reusable = _device_capacity(cp)
        except (ImportError, RuntimeError):
            driver, reusable = None, None
        capacity = dict(driver_free_bytes=driver, pool_reusable_bytes=reusable,
                        host_available_bytes=host_available_bytes())
    driver = capacity.get('driver_free_bytes')
    reusable = capacity.get('pool_reusable_bytes')
    host = capacity.get('host_available_bytes')
    available = None if driver is None or reusable is None else max(0, driver) + max(0, reusable)
    allowances = []
    if available is not None:
        allowances.append(available // 4)
    if host is not None:
        allowances.append(max(0, host) // 8)
    if override_mib is not None:
        budget = execution_scratch_override({SCRATCH_ENV: override_mib})
        basis = 'explicit execution override; downstream actual-capacity ceiling remains active'
    elif allowances:
        budget = max(1, min(allowances) // (1 << 20))
        basis = 'minimum of one quarter available device bytes and one eighth available host bytes, where measured; one MiB minimum request'
    else:
        budget = float(planned_mib)
        basis = 'resource readings unavailable; retained planned scratch basis'
    receipt = dict(schema='gpuwm-da.execution-scheduling.v1', memory_budget_mib=budget,
        planned_scratch_mib=planned_mib, override_environment=SCRATCH_ENV if override_mib is not None else None,
        driver_free_bytes=driver, pool_reusable_bytes=reusable, host_available_bytes=host,
        device_fraction=.25, host_fraction=.125, basis=basis,
        scientific_settings_changed=False)
    return budget, receipt


def launch(path, *, backend=None, cycle_runner=None, forecast_runner=None, roster_reader=None):
    """Run/resume with the same reviewed inputs and complete analysis rosters.

    Dependency injection is for CPU orchestration tests. The public command
    uses PreparedBackend and the existing ensemble engine unconditionally.
    """
    path = Path(path).resolve()
    plan = read_plan(path)
    scratch_override = execution_scratch_override()
    root = path.parent
    from gpuwm.ensemble.config import load_ensemble_config
    from gpuwm.ensemble.cycle import run_cycles, read_analysis_roster, cycle_root
    from gpuwm.ensemble.engine import run_ensemble
    from gpuwm.supervisor import GPUFileLock
    cfg = load_ensemble_config(root / 'ensemble.toml')
    backend = backend or PreparedBackend(plan, root)
    cycle_runner = cycle_runner or run_cycles
    forecast_runner = forecast_runner or run_ensemble
    roster_reader = roster_reader or read_analysis_roster
    # Missing dependencies and contradictory launch paths fail before a
    # directory is claimed, a source is fetched or a forecast is started.
    backend.preflight()
    started = time.monotonic()
    report = dict(schema='arwen.local-da-execution.v1', review_sha256=plan['review_sha256'],
                  status='PREPARING', forecast_started=False, cycles=[], products=None,
                  warnings=list(getattr(backend, 'warnings', ())))
    lock = GPUFileLock('local-da-' + plan['review_sha256'], path=root / '.local-da.lock', run_id=plan['review_sha256'])
    with lock:
        try:
            backend.prepare()
            _atomic(root / 'execution.json', report)
            cycle_started = None
            cycle_durations = []
            def event(value):
                nonlocal cycle_started
                now = time.monotonic()
                if value.get('event') == 'cycle-started':
                    cycle_started = now
                if value.get('event') == 'cycle-finished' and cycle_started is not None:
                    from gpuwm.local_da import cadence_projection
                    cadence = plan['selected']['cadence_seconds']
                    cycle_durations.append(now - cycle_started)
                    cycle_started = None
                    average = sum(cycle_durations) / len(cycle_durations)
                    remaining = plan['selected']['cycles'] - int(value['cycle']) - 1
                    projection = cadence_projection(epoch=plan['request']['epoch'],
                        cadence_seconds=cadence, cycles=remaining,
                        dt_seconds=plan['clock']['parent_step_ticks'] / plan['clock']['tick_hz'],
                        cycle_cost_seconds=average, measured=True) if remaining and average > 0 else None
                    report['cadence_progress'] = dict(cost_basis='measured',
                        measured_cycles_this_launch=len(cycle_durations),
                        cycle_wall_seconds=cycle_durations[-1],
                        mean_cycle_wall_seconds=average,
                        lag_seconds=max(0., sum(cycle_durations) - len(cycle_durations) * cadence),
                        scope='cycles completed during this launch; preparation excluded',
                        remaining_cycles=remaining, projection=projection)
                if value.get('event') == 'cycle-started':
                    report['status'] = 'CYCLING'
                if value.get('event') == 'cycle-finished':
                    from gpuwm.ensemble.manifest import CYCLE_MANIFEST_NAME
                    report['observation_usage'] = _completed_observation_usage(root / 'cycles' / CYCLE_MANIFEST_NAME)
                if value.get('event') == 'member-started':
                    report['forecast_started'] = True
                report['event'] = value
                report['elapsed_seconds'] = time.monotonic() - started
                _atomic(root / 'execution.json', report)
            def runner(**kwargs):
                report['forecast_started'] = True
                _atomic(root / 'execution.json', report)
                return backend.member_runner(**kwargs)
            def scratch_budget(planned_mib):
                budget, receipt = analysis_execution_budget(
                    planned_mib, override_mib=scratch_override)
                report['analysis_execution'] = receipt
                _atomic(root / 'execution.json', report)
                return budget, receipt
            def analysis_progress(value):
                report['analysis_progress'] = dict(value)
                report['elapsed_seconds'] = time.monotonic() - started
                _atomic(root / 'execution.json', report)
            from gpuwm.da.radar_assimilation import analysis_execution_options
            with analysis_execution_options(scratch_budget=scratch_budget,
                                            progress=analysis_progress):
                cycles = cycle_runner(cfg, root / 'cycles', n_cycles=plan['selected']['cycles'],
                    cycle_seconds=plan['selected']['cadence_seconds'], assimilate=backend.assimilate,
                    runner=runner, positivity='clip', restart_from_analysis=True,
                    moment_policy='full-moment', moment_repair=True, mp_physics=backend.mp_physics,
                    on_event=event,
                    **({'analysis_context': backend.analysis_context}
                       if hasattr(backend, 'analysis_context') else {}))
            if cycles.status != 'COMPLETE':
                raise RuntimeError(f'Cycling stopped with status {cycles.status}; inspect the cycle receipt before resuming.')
            # Completed cycles may have been recovered without invoking the
            # analysis again. Read their published receipts through the owner.
            report['observation_usage'] = _completed_observation_usage(cycles.manifest_path)
            restarts = roster_reader(cycle_root(root / 'cycles', plan['selected']['cycles'] - 1), n_members=cfg.n_members)
            if set(restarts) != set(range(cfg.n_members)):
                raise PlanError('The final analysis roster is incomplete; recover the cycle publication before launching the short forecast.', code='ANALYSIS_ROSTER')
            horizon = plan['selected']['cycles'] * plan['selected']['cadence_seconds'] + plan['selected']['forecast_seconds']
            report['status'] = 'FORECASTING'
            _atomic(root / 'execution.json', report)
            forecast = forecast_runner(cfg, _forecast_output_root(root), run_seconds=horizon,
                                       restarts=restarts, runner=runner, resume=True, on_event=event)
            if forecast.status != 'COMPLETE':
                raise RuntimeError(f'The short forecast stopped with status {forecast.status}; inspect its manifest before resuming.')
            report.update(status='COMPLETE', cycle_manifest=str(cycles.manifest_path),
                          forecast_manifest=str(forecast.manifest_path),
                          cycles=list(getattr(cycles, 'cycles_run', ())),
                          products=backend.products(forecast), elapsed_seconds=time.monotonic() - started)
            _atomic(root / 'execution.json', report)
            return report
        except BaseException as exc:
            if isinstance(exc, ProductFailure):
                report['products'] = exc.products
            report.update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'FAILED',
                          error=str(exc), elapsed_seconds=time.monotonic() - started,
                          recovery='Prior completed outputs and analysis checkpoints are retained. Resolve the reported operation failure and launch this saved plan again to resume.')
            _atomic(root / 'execution.json', report)
            if isinstance(exc, Exception):
                with contextlib.suppress(AttributeError, TypeError):
                    exc.forecast_started = report['forecast_started']
                    exc.recovery = report['recovery']
            raise
        finally:
            backend.close()


class PreparedBackend:
    """Source-neutral cache preparation connected to the shipped member runner."""
    def __init__(self, plan, root):
        self.plan, self.root = plan, Path(root)
        self.inputs = None
        self.setup = None
        self.state = None
        self.grid = None
        self._surface = {}
        self._frames = {}
        self.analysis_times = [utc(s) for s in plan['analysis_times']]
        self.route_receipts = []
        self.warnings = []

    def preflight(self):
        from gpuwm import capabilities, go_cli
        from gpuwm.geog_assets import default_geog_root
        from gpuwm.experiment import load_experiment, refuse_unrouted_spectral_numerics
        capabilities.require_for_command('go')
        missing_render = go_cli.render_extra_missing()
        if missing_render:
            raise RuntimeError(missing_render)
        self.exp = load_experiment(self.root / 'experiment.toml')
        refuse_unrouted_spectral_numerics(self.exp, 'local cycling member integration')
        self.mp_physics = self.exp.domains[0].run.mp_physics
        if 'background' in self.plan:
            from gpuwm.regional_preparation import validate_saved_background
            validate_saved_background(self.plan, self.root / 'experiment.toml')
            self.go = dict(run=self.root / 'background', prepared=self.root / 'background',
                           render=self.root / 'products')
            self.bridge = None
        else:
            # Saved reviews without a background selection retain the exact
            # original preparation path and already authored fetch cycle.
            self.go = go_cli.plan_from_config(self.root / 'experiment.toml', outdir=self.root / 'background', run_stamp=False)
            self.bridge = go_cli.resolve_bridge()
        self.geog = default_geog_root()
        supplied = self.plan.get('background', {}).get('inputs', {}).get('kind') == 'prepared'
        missing = None if supplied else go_cli.geography_refusal(self.geog)
        if missing:
            raise PlanError(missing + ' Install the named geography before launching local DA.', code='MISSING_GEOGRAPHY')
        # The actual launch, unlike review, checks the current card.
        go_cli._require_forecast_device()
        import cupy as cp
        free, _ = cp.cuda.runtime.memGetInfo()
        if self.plan['memory']['peak_bytes'] > free:
            self.warnings.append('The estimated peak exceeds currently free VRAM. The requested settings are retained; allocation will report any actual memory failure. Free other memory or choose different settings if needed.')
        for route in self.plan['observations']:
            binary = route.get('binary')
            if route['status'] in ('candidate', 'ready') and binary is not None and not Path(binary).is_file():
                route.update(status='unavailable', reason='The reviewed observation binary is no longer present; this stream is skipped.')
        # Scheme operator availability is checked against a lightweight
        # analytic namespace by the operator's own scheme dispatch only
        # when real columns have been prepared, before the first forecast.

    def prepare(self):
        from gpuwm import go_cli
        from gpuwm.regional_preparation import MissingPreparationManifest
        try:
            if 'background' in self.plan:
                from gpuwm.regional_preparation import prepare_background
                self.inputs = prepare_background(self.plan, self.root, self.exp, geog=self.geog)
                self.go['prepared'] = self.inputs.prepared_root
            else:
                from gpuwm.regional_preparation import prepare_legacy_background
                self.inputs, self.go = prepare_legacy_background(self.go, self.root, self.exp,
                    geog=self.geog, cadence=self.plan['selected']['cadence_seconds'],
                    review_sha256=self.plan['review_sha256'])
        except MissingPreparationManifest as error:
            raise PlanError(str(error), code='MISSING_MANIFEST') from error
        a, b = self.exp.domains[0].run, self.inputs.experiment.domains[0].run
        if asdict(a) != asdict(b) or self.exp.start_time != self.inputs.experiment.start_time:
            raise PlanError('Preparation changed the reviewed run configuration or initial clock; restore matching authorities before forecasting.', code='PREPARATION_CHANGED')
        self.exp = self.inputs.experiment
        # Prepare once to prove operator/setup contracts before any member
        # integrates. The factory restores again for each independent leg.
        self._prepared_factory(self.root / 'experiment.toml')
        self._validate_observation_inputs()
        self._release_state()

    def _prepared_factory(self, base_config):
        from gpuwm import runtime
        from gpuwm.ingest.prepared_cache import restore_prepared_cache
        # This existing source-neutral initializer retains its original
        # module path; no source-specific implementation is copied here.
        from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
        from gpuwm.case_data import trace_gas_overrides_from_config
        inputs, exp = self.inputs, self.inputs.experiment
        restored = restore_prepared_cache(inputs.prepared_cache_path,
            expected_identity=inputs.cache_identity, cfg=exp.root.run, static=inputs.static)
        if restored.surface is None:
            raise PlanError('The prepared cache has no canonical surface state; rebuild preparation before forecasting.', code='MISSING_SURFACE')
        initialize_prepared_physics(restored.initial_result, exp.root.run, restored.met, restored.surface,
            inputs.static, inputs.landuse_identity, inputs.grid, exp.start_time,
            constant_glw_wm2=runtime.declared_constant_glw(exp), p_top=exp.vertical.p_top,
            column_chunk=exp.column_chunk, trace_gas_overrides=trace_gas_overrides_from_config(
                inputs.experiment_config, expected_sha256=inputs.file_sha256['experiment_config']))
        self.state = restored.initial_result.state
        if self.setup is None:
            names = ('thb', 'phb', 'dphb_resid', 'alb', 'rdnw', 'c1h', 'c2h', 'c3h', 'c4h',
                     'c3f', 'c4f', 'dc3f', 'dc4f', 'mub2d', 'p_top', 'dnw')
            self.setup = {name: self._host(getattr(self.state, name)) for name in names}
            self.setup['static_covariance'] = self.plan['selected']['members'] == 1
        prepared = runtime.PreparedRealCase(cfg=exp.root.run, grid=inputs.grid,
            static_fields=inputs.static, initial_result=restored.initial_result,
            final_analysis=None, initial_snow_water_kgm2=np.zeros_like(self._host(self.state.mup)),
            forcing_times=tuple(exp.start_time + timedelta(hours=h) for h in inputs.forcing_hours))
        data = SimpleNamespace(output_title='Local rapid cycling', output_domain=1)
        return exp, data, prepared

    @staticmethod
    def _host(value):
        if value is None:
            return None
        return np.array(value.get() if hasattr(value, 'get') else value, copy=True)

    def _release_state(self):
        self.state = None
        import gc
        gc.collect()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except ImportError:
            pass

    def _reference_grid(self):
        """Freeze initial geometry through the same writer/readers as radar grids.

        The reference is for observation gridding, not a forecast product.
        Its initial interface heights are held fixed across the short cycle.
        """
        from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs
        from gpuwm.obs.target_grid import TargetGrid
        projection, cfg = self.inputs.grid, self.exp.root.run
        path = self.root / 'observation-reference.nc'
        lat, lon = projection.latlon_mass()
        php, phb = self._host(self.state.php), self._host(self.state.phb)
        if phb.ndim == 1:
            phb = np.broadcast_to(phb[:, None, None], php.shape)
        # HGT is an output spelling. The reference terrain is the ground
        # interface already carried by this prepared atmosphere.
        from gpuwm.core.constants import G
        fields = dict(PH=php, PHB=phb, XLAT=lat, XLONG=lon,
                      HGT=(php[0] + phb[0]) / G)
        attrs = wrf_global_attrs(projection, self.exp.start_time)
        if not path.exists():
            with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                              dx=cfg.dx, dy=cfg.dy, title='Observation reference geometry',
                              global_attrs=attrs) as writer:
                writer.write_frame(self.exp.start_time.strftime('%Y-%m-%d_%H:%M:%S'), fields)
        grid = TargetGrid.from_wrfout(path)
        # Compare against the same on-tape precision and attribute layout,
        # not a float64 projection or height invented by this reader.
        stamp = self.root / 'observation-reference.json'
        expected = dict(schema='arwen.local-da-reference.v1', review_sha256=self.plan['review_sha256'],
                        sha256=_sha(path), identity=grid.identity_sha256(),
                        height_policy='fixed initial interface heights for local gridding and localization')
        if stamp.exists():
            if json.loads(stamp.read_text()) != expected:
                raise PlanError('The saved observation reference changed; restore its reviewed bytes before resuming.', code='REFERENCE_CHANGED')
        else:
            _atomic(stamp, expected)
        self.grid, self.reference_path = grid, path

    def _validate_observation_inputs(self):
        self._reference_grid()
        from gpuwm.da.radar_assimilation import scheme_reflectivity_provider
        from gpuwm.ensemble.state_sha import serialized_state_attrs
        state = {name: self._host(getattr(self.state, name)) for name in serialized_state_attrs()
                 if getattr(self.state, name, None) is not None}
        for name in ('p', 'al', 'alt'):
            state[name] = self._host(getattr(self.state, name))
        # The forecast scheme itself owns the reflectivity operator. A
        # missing scheme implementation is recorded, not replaced by a
        # generic power law.
        try:
            provider = scheme_reflectivity_provider(self.exp.root.run, base_theta=self.setup['thb'])
            provider(0, state)
            self.reflectivity_available = True
        except (ValueError, NotImplementedError) as exc:
            self.reflectivity_available = False
            self.route_receipts.append(dict(route='reflectivity', status='unavailable', reason=str(exc)))
        from gpuwm.obs.goes_window import cwp_operator_refusal
        self.cwp_unavailable_reason = None
        if any(row['route'] == 'cloud-water-path' and row['status'] in ('ready', 'candidate')
               for row in self.plan['observations']):
            self.cwp_unavailable_reason = cwp_operator_refusal(state, self.setup, self.exp.root.run)
            if self.cwp_unavailable_reason:
                if self.plan['request']['satellite_grids']:
                    raise PlanError(self.cwp_unavailable_reason, code='CWP_OPERATOR_UNAVAILABLE')
                self.route_receipts.append(dict(route='cloud-water-path', status='unavailable',
                                               reason=self.cwp_unavailable_reason))
        from gpuwm.obs.radar_grid import read_radar_grid
        from gpuwm.obs.goes_grid import read_goes_grid
        for p in self.plan['request']['radar_grids']:
            read_radar_grid(p, expected_grid=self.grid)
        for p in self.plan['request']['satellite_grids']:
            read_goes_grid(p, expected_grid=self.grid)
        from gpuwm.da.obs_point import read_tables
        if self.plan['request']['obs_tables']:
            read_tables(self.plan['request']['obs_tables'])

    def member_runner(self, **kwargs):
        from gpuwm.ensemble.member import run_member
        try:
            outcome = run_member(**kwargs, prepare=self._prepared_factory)
            physics = getattr(self.state, 'physics', None)
            surface = {name: self._host(getattr(physics, 'fields', {}).get(name)) for name in ('t2', 'u10', 'v10')}
            surface = {k: v for k, v in surface.items() if v is not None}
            self._surface[str(Path(kwargs['member_dir']).resolve())] = surface
            receipt = dict(schema='arwen.local-da-surface-diagnostics.v1',
                elapsed_seconds=float(outcome.sim_seconds), checkpoint_sha256=outcome.final_state_sha256,
                fields=sorted(surface), label='forecast-leg-end diagnostics')
            dest = Path(kwargs['member_dir']) / 'surface-end.npz'
            with dest.open('wb') as handle:
                np.savez(handle, **surface, receipt=np.frombuffer(canonical(receipt), dtype=np.uint8))
            return outcome
        finally:
            self._release_state()

    def _surface_for_members(self, member_states):
        values = []
        for index in sorted(member_states):
            record = member_states[index]
            root = Path(record['member_dir'])
            with np.load(root / 'surface-end.npz', allow_pickle=False) as file:
                receipt = json.loads(file['receipt'].tobytes())
                if receipt['checkpoint_sha256'] != record['state_sha256']:
                    raise PlanError('Surface diagnostics do not describe the forecast state; restore its end-of-leg diagnostic record.', code='SURFACE_CHANGED')
                values.append({k: np.array(file[k], copy=True) for k in receipt['fields']})
        keys = set.intersection(*(set(v) for v in values))
        return {key: np.stack([v[key] for v in values]) for key in keys}

    def analysis_context(self, cycle_index, member_states, *, recovering):
        """Freeze through the observation owner, then bind its actual inputs."""
        from gpuwm.local_da_fetch import WINDOW_SCHEMA
        from gpuwm.output_identity import file_record
        path = self.root / 'observations' / f'cycle_{cycle_index:03d}' / 'window.json'
        if recovering and not path.is_file():
            raise PlanError('The original observation window is missing; restore it before recovering this analysis.', code='OBSERVATION_WINDOW_CHANGED')
        when = self.analysis_times[cycle_index]
        self._observation_window(cycle_index, when, member_states)
        window = json.loads(path.read_text())
        if window.get('schema') != WINDOW_SCHEMA:
            raise PlanError('The frozen observation window has another schema; restore its original receipt.', code='OBSERVATION_WINDOW_CHANGED')
        paths = [path, self.root / 'experiment.toml', self.root / 'ensemble.toml',
                 self.inputs.proof_path]
        paths.extend(Path(asset['path']) for asset in window['assets'])
        paths.extend(Path(info['member_dir']) / 'surface-end.npz' for info in member_states.values())
        return dict(review_sha256=self.plan['review_sha256'],
                    analysis_time=self.plan['analysis_times'][cycle_index],
                    observation_window_sha256=digest(window),
                    method_settings=dict(self.plan['cadence_settings']['applied']),
                    covariance_members=self.plan['selected']['covariance_members'],
                    base_seed=self.plan['request']['base_seed'],
                    grid_identity=self.grid.identity_sha256(),
                    assets=[file_record(path) for path in paths])

    def assimilate(self, cycle_index, member_states):
        from gpuwm.da.radar_assimilation import (member_background_checkpoint, read_checkpoint_state,
            RadarAssimilationConfig, assimilate_radar_grid, scheme_reflectivity_provider, _mass_field)
        from gpuwm.da.static_covariance import covariance_states, static_analysis, perturbation_options
        from gpuwm.da.letkf import Localization
        from gpuwm.da.moments import analysis_fields, pairs_present
        from gpuwm.da.obs_point import point_batches, state_columns
        from gpuwm.da.obs_surface import SurfaceObsConfig, surface_to_gridded_obs
        from gpuwm.da.obsop_cwp import checkpoint_cwp_provider
        index_list = sorted(member_states)
        if index_list != list(range(self.plan['selected']['members'])):
            raise PlanError('The forecast roster differs from the reviewed member count; recover every reviewed member before analysis.', code='ANALYSIS_ROSTER')
        checkpoints = {index: member_background_checkpoint(member_states[index]['member_dir']) for index in index_list}
        background = [read_checkpoint_state(checkpoints[i]) for i in index_list]
        surface = self._surface_for_members(member_states)
        when = self.analysis_times[cycle_index]
        obs = self._observation_window(cycle_index, when, member_states)
        states, static_receipt = background, None
        if len(background) == 1:
            states, static_receipt = covariance_states(background[0], self.setup, self.exp.root.run,
                samples=self.plan['selected']['covariance_members'],
                seed=self.plan['request']['base_seed'] + cycle_index * self.plan['selected']['covariance_members'],
                options=perturbation_options(mp_physics=self.mp_physics))
            # Frozen surface-transfer linearization for analysis-only
            # covariance samples. The actual H(background) is unchanged.
            cols = state_columns(states, self.setup, self.grid)
            if 't2' in surface:
                surface['t2'] = np.stack([surface['t2'][0] + col['t'][0] - cols[0]['t'][0] for col in cols])
            for name, key in (('u10', 'u'), ('v10', 'v')):
                if name in surface:
                    base = _mass_field(key, states[0], where='surface tangent')[0]
                    surface[name] = np.stack([surface[name][0] + _mass_field(key, state, where='surface tangent')[0] - base for state in states])
            static_receipt['surface_operator'] = 'frozen-transfer tangent approximation; actual end-of-leg background diagnostic plus lowest-level perturbation'
        # Convert diagnostic vectors to earth axes; speed itself is
        # unchanged by this rotation.
        from gpuwm.da.radar_assimilation import grid_rotation
        sina, cosa = grid_rotation(self.grid)
        if 'u10' in surface and 'v10' in surface:
            from gpuwm.da.obsop import earth_relative_winds
            surface['u10_earth'], surface['v10_earth'] = earth_relative_winds(surface['u10'], surface['v10'], sina, cosa)
        settings = self.plan['cadence_settings']['applied']
        loc = Localization(horizontal_m=settings['horizontal_loc_m'], vertical_m=settings['vertical_loc_m'])
        extra, receipts = [], []
        if obs['rows']:
            extra, report = point_batches(obs['rows'], states=states, setup=self.setup, grid=self.grid,
                analysis_time=when, analysis_times=self.analysis_times, surface=surface,
                max_age_seconds=self.plan['selected']['cadence_seconds'], localization=loc,
                error_inflation=settings['error_inflation'])
            receipts.append(report)
        for part, record in enumerate(obs['surface']):
            available_t = 't2' in surface
            available_w = 'u10' in surface and 'v10' in surface
            if available_t or available_w:
                batches, report = surface_to_gridded_obs(record, target_grid=self.grid, analysis_time=when,
                    config=SurfaceObsConfig(temperature_error_k=2. if available_t else None,
                        wind_speed_error_ms=2. if available_w else None,
                        error_inflation=settings['error_inflation'],
                        max_age_seconds=self.plan['selected']['cadence_seconds']),
                    simulated_t2=surface.get('t2'), simulated_u10=surface.get('u10'), simulated_v10=surface.get('v10'),
                    analysis_times=self.analysis_times)
                extra.extend(replace(batch, name=f'{batch.name}:part{part}') for batch in batches)
                receipts.append(report)
        fields = analysis_fields(self.mp_physics, base=('u', 'v', 'thp', 'qv'), hydrometeors=True)
        samples = states[1:] if len(background) == 1 else states
        fields = [name for name in fields if all(name in state for state in states)
                  and np.any(np.ptp(np.stack([_mass_field(name, state, where='local analysis')
                                             for state in samples]), axis=0) > 0)]
        # Never truncate a prognostic species pair merely because one of
        # its fields happens to have zero sampled spread.
        for pair in pairs_present(tuple(states[0]), mp_physics=self.mp_physics):
            if not set(pair.fields) <= set(fields):
                fields = [f for f in fields if f not in pair.fields]
        has_obs = bool(obs['radar'] is not None or obs['cwp'] is not None or any(np.any(b.mask) for b in extra))
        if not fields or not has_obs:
            return ({i: {'thp': np.zeros_like(background[slot]['thp'], dtype=float)} for slot, i in enumerate(index_list)},
                    dict(method='forecast-only', observation_count=0, reason='No accepted observations or no sampled covariance.',
                         routes=obs['receipts'], point_observations=receipts, static_covariance=static_receipt))
        radar = obs['radar'] is not None
        reflect = radar and self.reflectivity_available
        config = RadarAssimilationConfig(localization=loc, rtps_alpha=settings['rtps_alpha'],
            analysis_fields=tuple(fields), velocity=radar, reflectivity=reflect, clear_air=reflect,
            velocity_error_inflation=settings['error_inflation'],
            reflectivity_error_inflation=settings['error_inflation'],
            clear_air_error_inflation=settings['error_inflation'],
            cwp_error_inflation=settings['error_inflation'],
            fall_speed='reflectivity' if reflect else 'none', cwp=obs['cwp'] is not None,
            velocity_thinning_cells=max(1, int(np.ceil(6000. / self.grid.dx_m))),
            reflectivity_thinning_cells=max(1, int(np.ceil(6000. / self.grid.dx_m))),
            mp_physics=self.mp_physics, solve_device='auto', positivity_policy='clip',
            memory_budget_mib=self.plan['memory']['solve_memory_mib'])
        reflectivity_provider = scheme_reflectivity_provider(self.exp.root.run, base_theta=self.setup['thb']) if reflect else None
        cwp_provider = checkpoint_cwp_provider(self.exp.root.run, **{k: self.setup[k] for k in ('c1h', 'c2h', 'dnw', 'mub2d')}) if obs['cwp'] is not None else None
        with tempfile.TemporaryDirectory(prefix='analysis-', dir=self.root) as tmp:
            if len(background) == 1:
                checkpoints = {}
                for n, state in enumerate(states):
                    p = Path(tmp) / f'{n}.npz'
                    np.savez(p, **{'state/' + k: v for k, v in state.items()})
                    checkpoints[n] = p
            increments, report = assimilate_radar_grid(checkpoints, obs['radar'], self.grid, config,
                reflectivity_provider=reflectivity_provider, extra_obs=extra,
                extra_obs_provenance=receipts if extra else None, cwp_observations=obs['cwp'], cwp_provider=cwp_provider,
                analysis_runner=static_analysis if len(background) == 1 else None)
        if len(background) == 1:
            increments = {index_list[0]: increments[0]}
            report.update(method='static-covariance-oi', members=1, static_covariance=static_receipt)
        report.update(routes=obs['receipts'], point_observations=receipts)
        report['cwp_assimilated'] = observation_usage(report)['cwp_accepted'] > 0
        return increments, report

    def _observation_window(self, cycle_index, when, member_states):
        from gpuwm.local_da_fetch import observation_window
        return observation_window(self, cycle_index, when, member_states)

    def products(self, forecast):
        from gpuwm import go_cli
        from gpuwm.ensemble.manifest import read_manifest, ENSEMBLE_MANIFEST_SCHEMA
        from gpuwm.ensemble.wrfout_inventory import WRFOUT_INVENTORY_KEY
        document = read_manifest(forecast.manifest_path, schema=ENSEMBLE_MANIFEST_SCHEMA)
        reports = []
        for member in document['members']:
            from gpuwm.ensemble.wrfout_inventory import verify_entry
            root = Path(forecast.ens_root) / member['member_dir']
            inventory = member.get(WRFOUT_INVENTORY_KEY) or []
            problems = [problem for entry in inventory for problem in verify_entry(entry, member_dir=root)]
            if problems:
                raise PlanError('Output inventory verification failed: ' + '; '.join(problems) + '; restore the published frame bytes before rendering.', code='OUTPUT_CHANGED')
            frames = [root / entry['path'] for entry in inventory]
            if not frames:
                reports.append(dict(member=member['index'], status='unavailable', reason='The member manifest carries no output frames.'))
                continue
            render = dict(self.go, render=self.root / 'products' / f"member_{member['index']:03d}")
            missing = go_cli.render_extra_missing()
            if missing:
                reports.append(dict(member=member['index'], status='unavailable', reason=missing))
                continue
            try:
                go_cli._run_stage('render', go_cli.render_command(render, frames), explain=False)
                reports.append(dict(member=member['index'], status='complete', path=str(render['render'])))
            except (RuntimeError, go_cli.GoStageFailed) as exc:
                reports.append(dict(member=member['index'], status='failed', reason=str(exc)))
        products = dict(route='gpuwm render', members=reports)
        if not reports or any(row['status'] != 'complete' for row in reports):
            raise ProductFailure(products)
        return products

    def close(self):
        self._release_state()
