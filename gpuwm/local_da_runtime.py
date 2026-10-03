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


def _analysis_time(backend, index):
    origin = getattr(backend, '_continuous_epoch', None)
    if origin is not None:
        return origin + timedelta(seconds=(index + 1) * backend.plan['selected']['cadence_seconds'])
    return backend.analysis_times[index]


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
    from gpuwm.local_da import PUBLISHED_FILES, _route_companions
    bound = {name: hashlib.sha256(document['configuration'][key].encode()).hexdigest()
             for name, key in PUBLISHED_FILES}
    # The files the route reads beside the configuration, derived again from
    # the reviewed configuration as publish derived them: checked against
    # the three alone, every published case on a route with companions (the
    # HRRR chain's namelists) was refused here before anything ran.  A plan
    # published before the publisher wrote them records the three alone and
    # still launches: preparation writes the route's own set into its
    # authority folder (gpuwm.regional_preparation._prepare_hourly).
    companions = {companion.name: hashlib.sha256(text.encode()).hexdigest()
                  for companion, text in _route_companions(document, path.parent)}
    recorded = set(document.get('files', {}))
    if recorded not in (set(bound), set(bound) | set(companions)):
        raise PlanError('The saved configuration roster is incomplete; republish into a new directory.', code='CONFIGURATION_CHANGED')
    bound.update(companions)
    for name in sorted(recorded):
        sha = bound[name]
        if document['files'][name] != sha or _sha(path.parent / name) != sha:
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
        basis=('analysis innovation masks after QC and thinning; these are '
               'accepted observation counts and not a skill number, and the '
               'skill of the forecast that follows is the nowcast score '
               'written beside this receipt'),
        reason=method.get('reason'), routes=[{key: row[key] for key in
            ('id', 'route', 'status', 'reason', 'observed_columns', 'acquisition', 'latency_class') if key in row}
            for row in method.get('routes', ()) if 'id' in row or 'route' in row])


def nowcast_score(plan, root, *, analysis_time, forecast_manifest,
                  receipt_path=None, now=None):
    """The forecast's skill against the radar, scored by default at run end.

    Every lead whose scan time has passed and whose MRMS composite is
    reachable is scored now; a lead still in the future, or one the archive
    has not published, is recorded pending with its reason. A scoring
    problem never fails the run: the forecast ran, and this is a number
    about it rather than a gate on it.
    """
    from gpuwm.local_da_score import RECEIPT_NAME, cache_root_for, score_window
    root = Path(root)
    return score_window(plan=plan,
                        receipt_path=receipt_path or root / RECEIPT_NAME,
                        analysis_time=analysis_time,
                        forecast_manifest=forecast_manifest,
                        cache_root=cache_root_for(root), case_root=root,
                        now=now)


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
    if plan.get('continuous', {}).get('enabled'):
        return launch_continuous(path, plan=plan, backend=backend)
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
            report['nowcast_score'] = nowcast_score(plan, root,
                analysis_time=plan['analysis_times'][-1],
                forecast_manifest=forecast.manifest_path)
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
        self.renewal = None
        self.preserve_forcing = False

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
        from gpuwm.experiment import refuse_unrouted_simulated_radar
        refuse_unrouted_simulated_radar(self.exp, 'local cycling member integration')
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
        active = self.renewal or inputs
        restored = restore_prepared_cache(active.prepared_cache_path,
            expected_identity=active.cache_identity, cfg=exp.root.run, static=inputs.static)
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
            forcing_times=tuple(exp.start_time + timedelta(hours=h) for h in active.forcing_hours),
            preserved_forcing_prefix=self.preserve_forcing)
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
        from gpuwm.obs.target_grid import TargetGrid, identity_names_grid
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
            saved = json.loads(stamp.read_text())
            # A stamp may carry either spelling of this grid's identity
            # (TargetGrid.matches_identity); any other value is another grid.
            if isinstance(saved, dict) and identity_names_grid(grid, saved.get('identity')):
                saved = dict(saved, identity=expected['identity'])
            if saved != expected:
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
        when = _analysis_time(self, cycle_index)
        self._observation_window(cycle_index, when, member_states)
        window = json.loads(path.read_text())
        if window.get('schema') != WINDOW_SCHEMA:
            raise PlanError('The frozen observation window has another schema; restore its original receipt.', code='OBSERVATION_WINDOW_CHANGED')
        paths = [path, self.root / 'experiment.toml', self.root / 'ensemble.toml',
                 self.inputs.proof_path]
        paths.extend(Path(asset['path']) for asset in window['assets'])
        paths.extend(Path(info['member_dir']) / 'surface-end.npz' for info in member_states.values())
        return dict(review_sha256=self.plan['review_sha256'],
                    analysis_time=(when.isoformat() if getattr(self, '_continuous_epoch', None) is not None
                                   else self.plan['analysis_times'][cycle_index]),
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
        when = _analysis_time(self, cycle_index)
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
        from gpuwm.da.velocity_dispersion import (DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO,
                                                  DEFAULT_VELOCITY_DISPERSION_RATIO)
        config = RadarAssimilationConfig(localization=loc, rtps_alpha=settings['rtps_alpha'],
            velocity_dispersion_ratio=settings.get('velocity_dispersion_ratio',
                                                   DEFAULT_VELOCITY_DISPERSION_RATIO),
            velocity_dispersion_batch_ratio=settings.get('velocity_dispersion_batch_ratio',
                                                         DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO),
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

    def products(self, forecast, *, product_root=None):
        from gpuwm import go_cli
        from gpuwm.ensemble.manifest import read_manifest, ENSEMBLE_MANIFEST_SCHEMA
        from gpuwm.ensemble.wrfout_inventory import WRFOUT_INVENTORY_KEY
        document = read_manifest(forecast.manifest_path, schema=ENSEMBLE_MANIFEST_SCHEMA)
        reports = []
        warnings = []
        maps_checked = False
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
            render = dict(self.go, render=(Path(product_root) if product_root else self.root / 'products') / f"member_{member['index']:03d}")
            missing = go_cli.render_extra_missing()
            if missing:
                reports.append(dict(member=member['index'], status='unavailable', reason=missing))
                continue
            if not maps_checked:
                maps_checked = True
                # The render below prints its map-asset warning into output
                # that is captured and dropped when it succeeds, so pictures
                # with no coastlines or borders would pass unremarked. The
                # products record and the terminal carry it instead.
                from gpuwm import rustwx
                from gpuwm.render import BASEMAP_MISSING_CODE, renderer_basemap_gap
                gap = renderer_basemap_gap()
                if gap is not None:
                    warnings.append(dict(code=BASEMAP_MISSING_CODE, message=gap, remedy=rustwx.basemap_remedy()))
                    print(f'render: warning: {gap}', file=sys.stderr, flush=True)
            try:
                go_cli.run_render_pass(render, frames, explain=False)
                reports.append(dict(member=member['index'], status='complete', path=str(render['render'])))
            except (RuntimeError, go_cli.GoStageFailed) as exc:
                reports.append(dict(member=member['index'], status='failed', reason=str(exc)))
        products = dict(route='gpuwm render', members=reports)
        if warnings:
            products['warnings'] = warnings
        if not reports or any(row['status'] != 'complete' for row in reports):
            raise ProductFailure(products)
        return products

    def close(self):
        self._release_state()


def launch_continuous(path, *, plan=None, backend=None):
    """A saved continuous review runs its windows through the durable controller."""
    from gpuwm.local_da_controller import Controller
    path = Path(path).resolve()
    plan = read_plan(path) if plan is None else plan
    continuous = plan['continuous']
    selected = plan['selected']
    forcing_interval = plan.get('background', {}).get('selection', {}).get('forcing_interval_seconds', 3600)
    controller = Controller(path.parent / 'continuous', binding={'review_sha256': plan['review_sha256']},
        epoch=plan['request']['epoch'], cadence_seconds=selected['cadence_seconds'],
        forecast_seconds=selected['forecast_seconds'], members=selected['members'],
        windows=continuous['windows'], forcing_wait_seconds=forcing_interval, plan_path=path)
    backend = backend or ContinuousBackend(plan, path.parent, controller=controller)
    from gpuwm.local_da_controller import WindowFailure
    try:
        return controller.run(backend)
    except WindowFailure as error:
        failure = PlanError(str(error) + ' ' + error.recovery, code='CONTINUOUS_WINDOW_FAILED', details=error.details)
        failure.forecast_started = error.forecast_started
        raise failure from error


class ContinuousBackend(PreparedBackend):
    """One original prepared atmosphere, renewable forcing, indexed windows."""
    def __init__(self, plan, root, *, controller):
        super().__init__(plan, root)
        from gpuwm.ensemble.config import load_ensemble_config
        self.controller = controller
        self.cfg = load_ensemble_config(self.root / 'ensemble.toml')
        self._continuous_epoch = utc(plan['request']['epoch'])
        self._active_index = 0
        self.scratch_override = execution_scratch_override()
        # Every checkpoint of a continuous case carries the preserved
        # forcing-prefix contract, so a leg restarted after a renewal is
        # admitted on the forcing it ran under plus what was appended.
        self.preserve_forcing = True

    def _stage(self, name, **details):
        self.controller.stage(name, active_window=self._active_index, **details)

    def member_runner(self, **kwargs):
        self.controller.forecast_started = True
        self._stage('forecast')
        return super().member_runner(**kwargs)

    def _schedule(self, index):
        self._active_index = index
        # Only neighbouring boundaries can own a report admitted within one
        # cadence. Include the following boundary for the surface nearest rule.
        self.analysis_times = [_analysis_time(self, i) for i in range(max(0, index-1), index+2)]

    def _ensure_initial(self):
        if self.inputs is None:
            self._stage('preparation')
            super().prepare()

    def _active_times(self):
        return (self.renewal.forcing_times if self.renewal else
                tuple(self.inputs.experiment.start_time + timedelta(hours=h) for h in self.inputs.forcing_hours))

    def restore_window(self, prepared, directory):
        self._schedule(int(directory.name.split('_')[-1]))
        self._ensure_initial()
        from gpuwm.local_da_controller import _verify_files
        _verify_files(prepared['assets'])
        if prepared.get('renewal_receipt'):
            from gpuwm.regional_preparation import read_background_renewal
            self.renewal = read_background_renewal(prepared['renewal_receipt'], plan=self.plan,
                                                   root=self.root, original=self.inputs)
        else:
            self.renewal = None

    def _restore_previous_generation(self, prior):
        if prior is None:
            return
        from gpuwm.ensemble.analysis_commit import read_record
        from gpuwm.local_da_controller import _verify_files
        decision = read_record(prior['analysis_decision']['path'])
        _verify_files([decision['inputs']])
        inputs = read_record(decision['inputs']['path'])
        self.restore_window(inputs['prepared'], Path(decision['inputs']['path']).parent)

    def prepare_window(self, index, when, forecast, directory, *, prior=None):
        from gpuwm.output_identity import file_record
        self._ensure_initial()
        self._restore_previous_generation(prior)
        self._schedule(index)
        target = when + timedelta(seconds=forecast)
        current_end = self._active_times()[-1]
        current_end = current_end.replace(tzinfo=timezone.utc) if current_end.tzinfo is None else current_end.astimezone(timezone.utc)
        if current_end < target:
            if prior is None:
                raise ValueError("The reviewed forcing ends before the first window's products; the review prices at least one window, so restore the reviewed background before launching.")
            self._stage('preparation', renewal=dict(current_end=current_end.isoformat(), needed_end=target.isoformat()))
            from gpuwm.background_contract import BackgroundWindowError
            from gpuwm.local_da_controller import ForcingUnavailable
            from gpuwm.regional_preparation import renew_background
            try:
                self.renewal = renew_background(self.plan, self.root, self.inputs, previous=self.renewal,
                    end_time=target, directory=directory / 'forcing', geog=self.geog)
            except BackgroundWindowError as error:
                raise ForcingUnavailable(str(error)) from error
        active = self.renewal or self.inputs
        paths = [self.inputs.proof_path, self.inputs.source_manifest_path,
                 Path(active.prepared_cache_path) / 'header.json']
        if self.renewal:
            paths.append(self.renewal.receipt_path)
        return dict(original_epoch=self.plan['request']['epoch'],
            renewal_receipt=None if self.renewal is None else str(self.renewal.receipt_path),
            forcing_generation=0 if self.renewal is None else self.renewal.generation,
            forcing_times=[value.isoformat() for value in self._active_times()],
            assets=[file_record(path) for path in paths])

    def _observation_window(self, cycle_index, when, member_states):
        self._stage('observations')
        result = super()._observation_window(cycle_index, when, member_states)
        self._stage('analysis')
        return result

    def analyze_window(self, index, directory, prior, inputs):
        from gpuwm.ensemble.cycle import run_cycles, cycle_root, read_analysis_roster
        from gpuwm.output_identity import file_record
        self._schedule(index)
        restarts = None if prior is None else {i:row['path'] for i,row in enumerate(prior['analysis'])}
        self._stage('forecast')
        def scratch_budget(planned_mib):
            budget, receipt = analysis_execution_budget(planned_mib, override_mib=self.scratch_override)
            self._stage('analysis', analysis_execution=receipt)
            return budget, receipt
        def progress(value):
            self._stage('analysis', analysis_progress=dict(value))
        from gpuwm.da.radar_assimilation import analysis_execution_options
        binding = {k: v for k, v in inputs.items() if k != 'assets'}
        with analysis_execution_options(scratch_budget=scratch_budget, progress=progress):
            result = run_cycles(self.cfg, directory / 'cycles', n_cycles=1, first_cycle=index,
                initial_restarts=restarts, input_binding=binding, cycle_seconds=self.plan['selected']['cadence_seconds'],
                assimilate=self.assimilate, runner=self.member_runner, positivity='clip', restart_from_analysis=True,
                moment_policy='full-moment', moment_repair=True, mp_physics=self.mp_physics,
                analysis_context=self.analysis_context)
        if result.status != 'COMPLETE':
            raise RuntimeError('The analysis window did not complete; recover its existing member receipts')
        roster = read_analysis_roster(cycle_root(directory / 'cycles', index), n_members=self.cfg.n_members)
        usage = _completed_observation_usage(result.manifest_path)
        return dict(analysis=[file_record(path) for _,path in sorted(roster.items())],
                    cycle_manifest=str(result.manifest_path), observation_usage=usage)

    def produce_window(self, index, directory, decision):
        from gpuwm.ensemble.engine import run_ensemble
        from gpuwm.ensemble.analysis_commit import read_record, write_record
        from gpuwm.output_identity import file_record
        from gpuwm.local_da_controller import _verify_products
        receipt = directory / 'products.json'
        if receipt.exists():
            value = read_record(receipt)
            if value.get('window') != index or value.get('review_sha256') != self.plan['review_sha256']:
                raise ValueError('The product receipt belongs to another review or window')
            _verify_products(value)
            return {k:v for k,v in value.items() if k != 'self_sha256'}
        intent_path = directory / 'products-intent.json'
        if intent_path.exists():
            return self._commit_products(index, directory, read_record(intent_path))
        self._schedule(index)
        self._stage('forecast')
        restarts = {i:row['path'] for i,row in enumerate(decision['analysis'])}
        horizon = (index+1)*self.plan['selected']['cadence_seconds']+self.plan['selected']['forecast_seconds']
        forecast = run_ensemble(self.cfg, _forecast_output_root(directory), run_seconds=horizon,
            restarts=restarts, runner=self.member_runner, resume=True)
        if forecast.status != 'COMPLETE':
            raise RuntimeError('The window forecast did not complete; resume its existing members')
        self._stage('render')
        products = self.products(forecast, product_root=directory / 'products')
        from gpuwm.ensemble.wrfout_inventory import WRFOUT_INVENTORY_KEY, verify_entry
        document = json.loads(forecast.manifest_path.read_text(encoding='utf-8'))
        paths = [forecast.manifest_path, Path(decision['outcome']['cycle_manifest']), self.cfg.base_config]
        for member in document['members']:
            member_dir = forecast.ens_root / member['member_dir']
            inventory = member.get(WRFOUT_INVENTORY_KEY) or []
            if not inventory or any(verify_entry(entry, member_dir=member_dir) for entry in inventory):
                raise RuntimeError('The completed forecast has missing or changed frames; recover its ordinary output inventory')
            paths += [member_dir / entry['path'] for entry in inventory]
        images = sorted((directory / 'products').rglob('*.png'))
        if not images:
            raise ProductFailure({'members':[{'status':'failed','reason':'The renderer produced no images.'}]})
        paths += images
        from gpuwm.local_da_score import RECEIPT_NAME as NOWCAST_RECEIPT
        score = nowcast_score(self.plan, self.root,
            analysis_time=_analysis_time(self, index),
            forecast_manifest=forecast.manifest_path,
            receipt_path=directory / NOWCAST_RECEIPT)
        execution = dict(schema='arwen.local-da-execution.v1', review_sha256=self.plan['review_sha256'],
            status='COMPLETE', forecast_started=True, base_config=str(self.cfg.base_config),
            base_config_sha256=self.cfg.base_config_sha256, cycle_manifest=decision['outcome']['cycle_manifest'],
            forecast_manifest=str(forecast.manifest_path), products=products, cycles=[index],
            images=[str(path) for path in images],
            observation_usage=decision['outcome']['observation_usage'],
            nowcast_score=score,
            elapsed_seconds=self.controller.monotonic()-self.controller.started)
        intent = write_record(intent_path, dict(execution=execution,
                                               artifacts=[file_record(path) for path in paths]))
        return self._commit_products(index, directory, intent)

    def rescore_pending(self, *, now=None):
        """Fill in earlier windows' pending leads as later ones complete.

        At real time the 60 minute lead of window 0 cannot be scored until an
        hour after window 0's analysis, which is after window 3 has run. So
        every completed window rescores the ones before it: the receipts are
        rewritten, the status document reads them, and a lead that was
        pending stops being pending without anyone asking.
        """
        from gpuwm.local_da_score import score_case_root
        return score_case_root(self.plan, self.root, now=now,
                               plan_path=self.controller.plan_path)

    def _commit_products(self, index, directory, intent):
        from gpuwm.ensemble.analysis_commit import write_record
        from gpuwm.output_identity import file_record
        from gpuwm.local_da_controller import _verify_files
        execution = intent['execution']
        if (execution.get('review_sha256') != self.plan['review_sha256'] or execution.get('cycles') != [index]
                or execution.get('base_config_sha256') != self.cfg.base_config_sha256
                or execution.get('base_config') != str(self.cfg.base_config)
                or execution.get('status') != 'COMPLETE'):
            raise ValueError('The product publication intent belongs to another review or window')
        _verify_files(intent['artifacts'])
        execution_path = directory / 'execution.json'
        write_record(execution_path, execution)
        record = file_record(execution_path)
        value = dict(status='complete', window=index, review_sha256=self.plan['review_sha256'],
            execution_path=str(execution_path), execution_sha256=record['sha256'], execution_size_bytes=record['bytes'],
            base_config=str(self.cfg.base_config), base_config_sha256=self.cfg.base_config_sha256,
            forecast_manifest=execution['forecast_manifest'], cycle_manifest=execution['cycle_manifest'],
            images=execution.get('images', []),
            # The score as it stood when this window committed. The receipt
            # it names is rewritten as pending leads become scoreable, so
            # the receipt is the current answer and this is the record at
            # completion; the receipt is deliberately not a hashed artifact.
            nowcast_score=execution.get('nowcast_score'),
            verified_at_publication=True, artifacts=[record]+intent['artifacts'])
        write_record(directory / 'products.json', value)
        return value
