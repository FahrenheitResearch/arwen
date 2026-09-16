"""Prepare WPS met_em through native initialization and the shared forecast runtime."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from gpuwm.forecast_initialization import DomainInitialization
from gpuwm.progress_log import ProgressOptions, add_progress_arguments
from gpuwm.wrfinput_forecast import (RENDER_ROOT_NAME, WrfTreeInputs, WrfLanduseIdentity,
    _sha, announce_render_readiness, arm_door_first_products, door_render_plan,
    draw_door_products)


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')


@dataclass(frozen=True)
class MetemDomainBundle:
    grid_id: int
    cache: Path
    cache_identity: dict
    cache_reader: object
    static_fields: dict
    authority_sha256: dict
    geog_selection: WrfLanduseIdentity
    fractional_seaice: bool
    isoilwater: int


class MetemInitialization:
    """Only the restoration/physics seam varies; all stepping stays shared."""
    def __init__(self, inputs):
        self.inputs = inputs
        self.lateral_boundaries = inputs.boundaries
        from gpuwm.case_data import trace_gas_overrides_from_config
        self.trace_gas_overrides = trace_gas_overrides_from_config(
            inputs.experiment_config,
            expected_sha256=inputs.authority_sha256["experiment_config"])

    def restore_domain(self, domain, grid, bundle, *, start_time,
                       scratch_arena, dycore_state_workspace):
        from gpuwm.ingest.prepared_cache import restore_prepared_cache
        from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
        from gpuwm.runtime import declared_constant_glw
        restored = restore_prepared_cache(bundle.cache, expected_identity=bundle.cache_identity,
            cfg=domain.run, static=bundle.static_fields, allow_nested_without_lbc=domain.parent_id != 0)
        def physics():
            from gpuwm.core.cam_ozone import cam_ozone_setup, ozone_parent_for
            cam = cam_ozone_setup(exp=self.inputs.experiment, dc=domain, grid=grid)
            return initialize_prepared_physics(restored.initial_result, domain.run,
                restored.met, restored.surface, bundle.static_fields,
                bundle.geog_selection.landuse_global_attrs(), grid, start_time,
                constant_glw_wm2=declared_constant_glw(self.inputs.experiment),
                fractional_seaice=bundle.fractional_seaice, isoilwater=bundle.isoilwater,
                p_top=self.inputs.experiment.vertical.p_top,
                column_chunk=self.inputs.experiment.column_chunk,
                trace_gas_overrides=self.trace_gas_overrides,
                **({"cam_ozone": cam, "ozone_parent": ozone_parent_for(cam)}
                   if cam is not None else {}))
        return DomainInitialization(restored.initial_result, physics)

    def verify_inputs(self, inputs):
        for name, path in inputs.artifact_paths.items():
            if _sha(path) != inputs.authority_sha256[name]:
                raise RuntimeError(f'met_em input authority changed during execution: {path}')
        for bundle in inputs.domains:
            if bundle.cache_reader.verify_all()['content_sha256'] != bundle.authority_sha256['cache_content']:
                raise RuntimeError(f'met_em prepared cache changed: d{bundle.grid_id:02d}')

    def domain_content_sha256(self, bundle):
        return bundle.cache_reader.content_sha256

    def preparation_receipt_sha256(self):
        """Bind the verified preparation science, excluding resource observations."""
        import hashlib

        content = self.inputs.artifact_paths['preparation_receipt'].read_bytes()
        if hashlib.sha256(content).hexdigest() != self.inputs.authority_sha256['preparation_receipt']:
            raise RuntimeError('Metgrid preparation receipt changed before checkpoint identity was bound')
        payload = json.loads(content)
        if not isinstance(payload, dict) or payload.get('schema') != 'gpuwm-metgrid-import-v1':
            raise ValueError('Metgrid checkpoint identity requires the verified metgrid import schema')
        payload.pop('memory_admission', None)
        authority = {'schema': 'gpuwm-metgrid-scientific-authority-v1',
                     'preparation': payload}
        return hashlib.sha256(json.dumps(authority, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

    def domain_metadata(self, bundle):
        return bundle.cache_reader.metadata


def metgrid_soil(case, cfg, *, fractional_seaice):
    """Bind the actual WPS soil coordinates to the existing soil remapper."""
    from gpuwm.config import soil_layer_count
    from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
    from gpuwm.ingest.soil_contract import MAPPED_SOIL_TEMPERATURE, MAPPED_SOIL_MOISTURE
    from gpuwm.native_wrf_contract import canonical_noah_surface
    from gpuwm.static.build import deep_soil_temperature_at_terrain

    static = dict(case.statics)
    required = ('LANDSEA', 'SKINTEMP', 'LANDMASK', 'LU_INDEX', 'SCT_DOM',
                'SOILTEMP', 'GREENFRAC', 'LAI12M', 'SNOALB')
    missing = sorted(set(required)-set(static))
    if missing:
        raise ValueError(f'{case.path.name}: land-surface initialization needs {missing}; retain these metgrid/geogrid fields')
    for name in ('LU_INDEX', 'SCT_DOM'):
        if not np.equal(static[name], np.rint(static[name])).all():
            raise ValueError(f'{name} must contain integer land/soil category identities')
        static[name] = static[name].astype(np.int32)
    for name in ('GREENFRAC', 'LAI12M'):
        if static[name].shape != (12, *case.shape):
            raise ValueError(f'{name} must contain twelve monthly fields on the domain')
    static['TMN'] = deep_soil_temperature_at_terrain(static['SOILTEMP'], static['HGT_M'], static['LANDMASK'])
    contract, temperature, moisture = metgrid_soil_columns(case)
    fields = dict(case.snapshot.fields)
    fields.update({name:static[name] for name in ('LANDSEA','SKINTEMP','SST','SEAICE','XICE','SNOW','SNOWH') if name in static})
    fields[MAPPED_SOIL_TEMPERATURE] = temperature
    fields[MAPPED_SOIL_MOISTURE] = moisture
    prepared = preprocess_land_surface_soil(fields, sf_surface_physics=cfg.sf_surface_physics,
        num_soil_layers=soil_layer_count(cfg), soil_type=static['SCT_DOM'],
        deep_soil_temperature=static['TMN'], soil_layer_contract=contract,
        landmask=static['LANDMASK'], terrain=static['HGT_M'], source_orography=case.source_orography,
        water_temperature_policy='wrf_compat', fractional_seaice=fractional_seaice)
    return static, replace(case.snapshot, fields=fields), canonical_noah_surface(prepared), contract


def metgrid_soil_columns(case):
    """WPS node depths or encoded layer bounds, without a model-name table."""
    import re
    from gpuwm.ingest.soil_contract import NOAH_LAYER_BOUNDS_M, validate_soil_layer_contract
    soil = case.soil
    nodes = all(name in soil for name in ('SOILT','SOILM','SOIL_LEVELS'))
    layers = all(name in soil for name in ('ST','SM','SOIL_LAYERS'))
    if nodes == layers:
        raise ValueError(f'{case.path.name}: provide exactly one declared WPS soil geometry: SOILT/SOILM/SOIL_LEVELS nodes or ST/SM/SOIL_LAYERS layers')
    coordinate = 'SOIL_LEVELS' if nodes else 'SOIL_LAYERS'
    tname, mname = ('SOILT','SOILM') if nodes else ('ST','SM')
    depths = np.asarray(soil[coordinate])
    if depths.ndim == 3:
        if not np.equal(depths, depths[:, :1, :1]).all():
            raise ValueError(f'{coordinate} must identify the same depths throughout the domain')
        depths = depths[:, 0, 0]
    if depths.ndim != 1 or depths.size < 2 or not np.isfinite(depths).all():
        raise ValueError(f'{coordinate} must be a finite depth vector or depth/y/x coordinate')
    units = case.variable_units[coordinate].strip().lower()
    # WPS module_optional_input writes centimetre coordinates even when
    # the generated variable's units attribute is empty.
    if units not in ('', 'cm', 'centimeters', 'centimetres'):
        raise ValueError(f'{coordinate} units {units!r} do not establish WPS centimetre depths')
    if np.any(depths < 0) or np.unique(depths).size != depths.size:
        raise ValueError(f'{coordinate} depths must be distinct and nonnegative')
    for name, allowed in ((tname, ('','k','kelvin')), (mname, ('','fraction','1','m3 m-3','m^3/m^3'))):
        if case.variable_units[name].strip().lower() not in allowed:
            raise ValueError(f'{name} units do not establish the WPS soil quantity')
    if any(np.shape(soil[name]) != (depths.size,*case.shape) for name in (tname,mname)):
        raise ValueError('WPS soil temperature, moisture and coordinate counts disagree')
    order = np.argsort(depths)
    depths = depths[order]
    temperature, moisture = (np.ascontiguousarray(soil[name][order]) for name in (tname,mname))
    contract = {'temperature_field':'soil_temperature','moisture_field':'volumetric_soil_moisture',
        'depth_units':'m','target_layers':[{'top':a,'bottom':b} for a,b in NOAH_LAYER_BOUNDS_M],
        'missing':{'land':'reject','ocean':{'stage':'after_horizontal_interpolation','temperature':'skin_temperature','moisture':1.0}}}
    if nodes:
        contract['source_nodes'] = [{'depth':float(depth)/100,
            'selectors':{key:{'format':'netcdf','name':name,'layer_dimension':dim,
                             'layer_value':float(depth),'layer_units':'cm'}
                for key,name,dim in (('soil_temperature','SOILT','num_soilt_levels'),
                    ('volumetric_soil_moisture','SOILM','num_soilm_levels'))}}
            for depth in depths]
        contract['remap'] = {'kind':'linear_node_samples','source_value_location':'level_node','target_value_location':'layer_midpoint'}
    else:
        # SOIL_LAYERS stores BOTTOM depths. Only the actual WPS STtttbbb /
        # SMtttbbb names establish both bounds; never infer missing tops.
        bounds = {}
        for name in soil:
            match = re.fullmatch(r'ST(\d{3})(\d{3})',name)
            if match and 'SM'+name[2:] in soil:
                top,bottom=map(int,match.groups())
                if bottom in bounds:
                    raise ValueError('WPS soil layers repeat a bottom depth with different top bounds')
                bounds[bottom]=(top,name,'SM'+name[2:])
        if set(depths) != set(bounds):
            raise ValueError('SOIL_LAYERS needs matching STtttbbb/SMtttbbb fields to establish every layer top and bottom')
        contract['source_layers'] = []
        for index,depth in enumerate(depths):
            top,tn,mn = bounds[depth]
            for name,stack in ((tn,temperature),(mn,moisture)):
                if case.attributes.get('FLAG_'+name) != 1:
                    raise ValueError(f'{name}: WPS soil input requires its analyzed-field flag')
                if not np.array_equal(soil[name], stack[index]):
                    raise ValueError(f'{name} differs from the soil stack at its declared depth')
            contract['source_layers'].append({'top':top/100,'bottom':float(depth)/100,
                'selectors':{'soil_temperature':{'format':'netcdf','name':tn},
                             'volumetric_soil_moisture':{'format':'netcdf','name':mn}}})
        contract['remap'] = {'kind':'linear_point_samples','source_value_location':'wrf_integer_cm_layer_midpoint',
            'target_value_location':'layer_midpoint',
            'top_anchor':{'depth':0.0,'temperature':'skin_temperature','moisture':'repeat_shallowest'},
            'bottom_anchor':{'depth':3.0,'temperature':'deep_soil_temperature','moisture':'repeat_deepest'}}
    return validate_soil_layer_contract(contract), temperature, moisture


def _ladder_wrf_auto(run, argument, *, requested_e_vert=None, cpu_bridge=None):
    """WRF's own automatic generator; what a namelist without eta gets."""
    if argument:
        raise ValueError("vertical grid selector 'wrf-auto' takes no argument")
    from gpuwm.ingest.eta import generate_wrf_eta
    e_vert = requested_e_vert or run.experiment.root.run.nz + 1
    eta, receipt = generate_wrf_eta(e_vert,
        domains=run.controls.get('domains', {}),
        p_top=run.experiment.vertical.p_top,
        base_temp=run.experiment.root.run.base_temp, cpu_bridge=cpu_bridge)
    return eta, 'WRF automatic eta levels generated by shared Rust compute_eta', receipt


def _ladder_native(run, argument, *, requested_e_vert=None, cpu_bridge=None):
    """The certified ArWen profile, resampled to the requested e_vert."""
    if argument:
        raise ValueError("vertical grid selector 'native' takes no argument")
    from gpuwm.core.grid import resample_eta_levels
    from gpuwm.native_wrf_contract import CERTIFIED_ETA_LEVELS
    e_vert = requested_e_vert or run.experiment.root.run.nz + 1
    eta = resample_eta_levels(CERTIFIED_ETA_LEVELS, e_vert - 1)
    return (eta, 'explicitly selected ArWen native eta profile resampled to e_vert',
            {'algorithm': 'ArWen-native-profile', 'e_vert': e_vert})


def _ladder_explicit(run, argument, *, requested_e_vert=None, cpu_bridge=None):
    """One ladder read from a TOML file carrying an ``eta_levels`` array.

    That spelling is exactly what ``tools/build_stretched_eta_ladder.py
    --emit-toml`` prints and exactly what the config route already
    accepts, so a ladder is authored once and reaches both routes.  The
    tool's ``--json`` output is a SCORE and carries no eta array, so it is
    not accepted here.
    """
    import tomllib
    if not argument:
        raise ValueError("vertical grid selector 'explicit' needs a file: spell it explicit:PATH, a TOML file carrying an eta_levels array")
    path = Path(argument)
    if not path.is_file():
        raise ValueError(f'{path}: no such vertical ladder file; --vertical-grid explicit:PATH wants a TOML file carrying an eta_levels array')
    document = tomllib.loads(path.read_text(encoding='utf-8'))
    levels = document.get('eta_levels')
    if levels is None:
        levels = (document.get('shared') or {}).get('eta_levels')
    if levels is None:
        raise ValueError(f'{path}: carries no eta_levels array; tools/build_stretched_eta_ladder.py --emit-toml prints one in exactly the spelling this reads')
    eta = np.asarray(levels, dtype=np.float64)
    # THE FILE'S OWN LENGTH IS THE LEVEL COUNT.  An authored ladder that
    # had to carry exactly the producing namelist's e_vert could only ever
    # restretch a fixed number of levels, which is the pin --vertical-levels
    # exists to lift; a ladder file and an explicit count that disagree is
    # two answers to one question, so it is refused by name rather than
    # one of them silently winning.
    if requested_e_vert is not None and requested_e_vert != int(eta.size):
        raise ValueError(f'{path}: carries {eta.size} eta levels but '
                         f'--vertical-levels {requested_e_vert} was requested; '
                         'drop --vertical-levels and the file sets the count, '
                         'or author the ladder at the requested count')
    return (eta, f'explicit eta ladder read from {path}',
            {'algorithm': 'explicit-toml-ladder', 'path': str(path), 'e_vert': int(eta.size)})


#: The vertical ladders ``--vertical-grid`` can select, keyed by selector
#: name.  Adding a ladder is a ROW, not a branch and not an adapter file:
#: the refusal below names these keys and the module parser prints them,
#: so argparse and the table cannot disagree about what is selectable.
#: ``explicit`` is spelled ``explicit:PATH`` and takes the rest of the
#: selector as its argument.
METEM_VERTICAL_LADDERS = {
    'wrf-auto': _ladder_wrf_auto,
    'native': _ladder_native,
    'explicit': _ladder_explicit,
}


#: The smallest full-level count ``--vertical-levels`` can ask for.
#: ``gpuwm/experiment.py`` requires ``[shared] nz >= 4`` when it loads the
#: resolved TOML, so a ladder shorter than five full levels cannot be
#: built into an experiment at all.  Stated here rather than imported
#: because that floor is a literal argument there and not an exported
#: name; the refusal below names both numbers, so an operator who meets it
#: is told the bound rather than left to find it.
METEM_MINIMUM_E_VERT = 5


def resolve_metem_vertical(run, text, *, vertical_grid=None,
                           vertical_levels=None, cpu_bridge=None):
    """Materialize eta once, through the selector table, and validate it.

    An explicitly passed selector WINS over the namelist's own
    ``eta_levels``.  It used to refuse instead, which made a namelist that
    happened to carry a ladder unrunnable on any other ladder even though
    met_em carries source-level data that ``initialize_real`` interpolates
    onto whatever eta the resolved config declares.  A substitution is not
    a breakage: it is announced on its own plan-review line and recorded
    in the receipt's existing ``vertical_coordinate`` and
    ``vertical_generation`` fields.  It is deliberately NOT pushed through
    ``announce_wrf_substitutions``: that record is the namelist
    TRANSLATION, policed by ``require_preserved_wrf_selectors``, and an
    operator-requested ladder is not a translator decision.

    The one refusal left is the real one, and it fires here, at plan
    review: a ladder whose model top reaches above the source atmosphere
    is refused by name through the shared
    :func:`gpuwm.vertical_contract.validate_explicit_eta_grid`, the same
    contract the mapped-direct adapter already uses.
    """
    import re
    from gpuwm.vertical_contract import validate_explicit_eta_grid
    namelist_nz = run.experiment.root.run.nz
    p_top = run.experiment.vertical.p_top
    source_top = getattr(run, 'source_top_pressure_pa', None)
    namelist_ladder = run.experiment.vertical.eta_levels
    requested_e_vert = None
    if vertical_levels is not None:
        try:
            whole = not isinstance(vertical_levels, bool) and int(vertical_levels) == vertical_levels
        except (TypeError, ValueError):
            whole = False
        if not whole:
            raise ValueError(f'--vertical-levels {vertical_levels!r} is not a whole '
                'number of full eta levels (WRF e_vert); pass an integer count')
        requested_e_vert = int(vertical_levels)
        if requested_e_vert < METEM_MINIMUM_E_VERT:
            raise ValueError(f'--vertical-levels {requested_e_vert} is below the '
                f'{METEM_MINIMUM_E_VERT} full eta levels a resolved experiment '
                f'accepts (nz >= {METEM_MINIMUM_E_VERT - 1}); ask for at least '
                f'{METEM_MINIMUM_E_VERT}')
    if vertical_grid is None and vertical_levels is None and namelist_ladder:
        if source_top is not None:
            validate_explicit_eta_grid(namelist_ladder, nz=namelist_nz, p_top=p_top,
                source_top_pressure_pa=source_top,
                context='met_em namelist eta_levels')
        return text, 'explicit namelist eta_levels', None
    name, _, argument = str(vertical_grid or 'wrf-auto').partition(':')
    provider = METEM_VERTICAL_LADDERS.get(name)
    if provider is None:
        raise ValueError('--vertical-grid must name one of ' +
            ', '.join(METEM_VERTICAL_LADDERS) +
            f"; {name!r} names no ladder this build can build (the explicit "
            'ladder is spelled explicit:PATH)')
    selector = ' '.join(part for part in (
        '' if vertical_grid is None else f'--vertical-grid {vertical_grid}',
        '' if vertical_levels is None else f'--vertical-levels {requested_e_vert}') if part)
    selector = selector or '--vertical-grid wrf-auto (the default)'
    eta, policy, receipt = provider(run, argument,
        requested_e_vert=requested_e_vert, cpu_bridge=cpu_bridge)
    # THE RESOLVED LADDER OWNS THE LEVEL COUNT, not the producing
    # namelist.  e_vert used to be pinned to the namelist's nz, so an
    # authored ladder had to carry exactly that many levels and
    # --vertical-levels could not exist; the count now comes from the eta
    # the provider actually built, and the resolved [shared] nz is
    # rewritten to match so every consumer of the emitted TOML -- the
    # memory admission, the vertical coordinate, the prepared cache --
    # reads one number.
    nz = int(np.asarray(eta).size) - 1
    validate_explicit_eta_grid(eta, nz=nz, p_top=p_top,
        source_top_pressure_pa=source_top,
        context=f'met_em {selector}')
    if text.count('[shared]\n') != 1:
        raise ValueError('resolved experiment must carry one shared vertical coordinate')
    if namelist_ladder:
        policy += '; explicitly selected in place of the namelist eta_levels'
        print(f'met_em: {selector} replaces the '
              f"namelist's own {len(namelist_ladder)}-level eta ladder; the "
              'receipt records both the policy and the generator.', flush=True)
        text, removed = re.subn(r'(?ms)^eta_levels = \[.*?^\]\n', '', text)
        if not removed:
            text, removed = re.subn(r'(?m)^eta_levels = .*$\n', '', text)
        if removed != 1:
            raise ValueError('resolved experiment must carry one replaceable eta_levels ladder')
    if nz != namelist_nz:
        policy += f'; {nz + 1} full levels in place of the namelist e_vert {namelist_nz + 1}'
        print(f'met_em: {selector} resolves {nz + 1} full eta levels '
              f'(nz = {nz}) in place of the namelist e_vert '
              f'{namelist_nz + 1}.', flush=True)
        text, replaced = re.subn(r'(?m)^nz = \d+$', f'nz = {nz}', text)
        if replaced != 1:
            raise ValueError('resolved experiment must carry one [shared] nz to replace')
    text = text.replace('[shared]\n', '[shared]\neta_levels = '+repr(eta.tolist())+'\n', 1)
    return text, policy, receipt


def metem_window_seconds(run, run_seconds):
    """Bind a requested met_em run window to the shared forcing check."""
    from gpuwm.ingest.preflight import check_forcing_window
    coverage = float(run.coverage_seconds)
    return check_forcing_window(run_seconds, coverage_seconds=coverage,
        source='gpuwm run --met-em',
        last_valid_time=run.experiment.start_time + timedelta(seconds=coverage))


def prepare_metem_run(run, directory, *, run_seconds=None, preprocess_backend=None,
                      cpu_bridge=None, vertical_grid=None, vertical_levels=None):
    """One forcing state at a time, then immutable native prepared caches."""
    import gc
    import re
    import tomllib
    from gpuwm.experiment import build_experiment
    from gpuwm.runtime import vertical_coord_for
    from gpuwm.static.corridor import config_declares_follow_source
    from gpuwm.static.projection import grids_from_projection_config
    from gpuwm.ingest.metem import read_met_em, check_met_em_series, parse_met_em_name, met_em_series_identity
    from gpuwm.metem_door import metgrid_initialization_controls, metgrid_memory_admission
    from gpuwm.ingest.real import initialize_real
    from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend
    from gpuwm.preprocess_policy import resolve_preprocess_backend as preparation_road
    from gpuwm.ingest.lateral_bc import StateBoundaryFrames, start_last_forcing_order, attach_lateral_boundaries
    from gpuwm.ingest.prepared_cache import prepared_cache_identity, write_prepared_cache, PreparedCacheReader
    from gpuwm.prepared_domain_tree_forecast import resolve_execution_plan

    text = run.toml_text
    if run_seconds is not None:
        # The window is bounded by the FORCING, not by the duration the
        # producing namelist happened to carry.  One function says so, and
        # gpuwm/wrfinput_forecast.py calls the same one, so the two doors
        # cannot drift apart about one configuration.
        run_seconds = metem_window_seconds(run, run_seconds)
        text, count = re.subn(r'(?m)^run_seconds\s*=.*$', f'run_seconds = {float(run_seconds)}', text)
        if count != 1:
            raise ValueError('resolved experiment must carry one duration')
    text, vertical_policy, vertical_generation = resolve_metem_vertical(
        run, text, vertical_grid=vertical_grid, vertical_levels=vertical_levels,
        cpu_bridge=cpu_bridge)
    if vertical_generation is not None:
        # The label comes from the RECEIPT the ladder provider returned,
        # so a new row in METEM_VERTICAL_LADDERS prints itself.  The level
        # count comes from the receipt too: the resolved ladder owns it
        # now, so the namelist's nz is no longer what gets announced.
        controls = vertical_generation.get('controls')
        label = ('WRF automatic eta option ' + str(controls['auto_levels_opt'])
                 if controls else str(vertical_generation.get('algorithm', 'eta ladder')))
        mass_levels = int(vertical_generation.get(
            'e_vert', run.experiment.root.run.nz + 1)) - 1
        print(f'met_em: using {label} with {mass_levels} mass levels.', flush=True)
    exp = build_experiment(tomllib.loads(text), source=str(run.namelist_input))
    if config_declares_follow_source(exp):
        raise ValueError('moving nests need terrain and land-surface coverage at future positions; these met_em files contain only the initial footprints. Supply a native prepared statics corridor')
    # The shared policy governs both the estimate and the executing backend.
    # Explicit selectors and already-resolved backend objects are retained.
    road = preprocess_backend
    if road is None or isinstance(road, str):
        road = preparation_road(source='met_em', experiment=exp, requested=road)
    memory_receipt = metgrid_memory_admission(run, exp, **(
        {} if preprocess_backend is None else {
            'preprocess_backend': road if isinstance(road, str) else road.name}))
    for warning in memory_receipt.get('warnings', ()):
        print(f'met_em: {warning}', flush=True)
    directory = Path(directory)
    from gpuwm.prepared_documents import preparation_directory, json_bytes, write_document

    sources = {'namelist_input':run.namelist_input, 'adapter_code':Path(__file__).resolve(),
               'field_adapter_code':Path(__file__).with_name('ingest')/'metem.py',
               **{f'met_em_d{gid:02d}_{index}':path for gid,files in run.paths.items() for index,path in enumerate(files)}}
    source_digests = {name:_sha(path) for name,path in sources.items()}
    backend = resolve_preprocess_backend(road, cpu_bridge=cpu_bridge)
    documents = {'experiment.toml': text.encode('utf-8'),
                 'preprocess.json': json_bytes(backend.receipt())}
    import hashlib

    source_hashes = dict(source_digests,
        experiment_config=hashlib.sha256(documents['experiment.toml']).hexdigest(),
        preprocess_implementation=hashlib.sha256(documents['preprocess.json']).hexdigest())
    documents['source.json'] = json_bytes(source_hashes)
    required = ['metgrid-import.json']
    for domain in exp.domains:
        required.extend((f'static-d{domain.grid_id:02d}.npz',
                         f'd{domain.grid_id:02d}/header.json'))
    directory, reused = preparation_directory(
        directory, receipt='source.json', source_files=source_digests,
        documents=documents, required_files=required)
    directory.mkdir(parents=True, exist_ok=reused)
    config_path = directory/'experiment.toml'
    implementation = directory/'preprocess.json'
    artifacts = {'experiment_config':config_path,
                 'preprocess_implementation':implementation, **sources}
    source_manifest = directory/'source.json'
    for name, content in documents.items():
        write_document(directory/name, content, reused=reused)
    artifacts['source_manifest'] = source_manifest
    grids = tuple(grids_from_projection_config(exp))
    bundles, domain_receipts = [], {}
    root_boundaries = None
    fractional = run.controls.get('physics', {}).get('fractional_seaice', [0])[0] == 1
    for domain, grid in zip(exp.domains, grids):
        cfg = domain.run
        frames = StateBoundaryFrames(spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
        files = run.paths[domain.grid_id]
        series = {}
        coord = vertical_coord_for(exp.vertical, cfg.nz)
        # Children use parent runtime forcing; only their own initial state
        # is needed. Root boundary frames retain only each time's perimeter.
        order = start_last_forcing_order(len(files)) if domain.parent_id == 0 else (0,)
        for index in order:
            case = read_met_em(files[index])
            controls = metgrid_initialization_controls(case, run, cfg=cfg)
            lat, lon = grid.latlon_mass()
            if max(np.max(np.abs(lat-case.statics['XLAT_M'])), np.max(np.abs((lon-case.statics['XLONG_M']+180)%360-180))) > .005:
                raise ValueError(f'{case.path.name}: met_em coordinates disagree with the resolved domain-tree layout')
            series[index] = met_em_series_identity(case)
            result = initialize_real(case.snapshot, cfg, coord, case.terrain,
                source_orography=case.source_orography, p_top=exp.vertical.p_top,
                preprocess_backend=backend, state_backend='preprocess', **controls)
            result.state.set_map_coriolis(case.statics['MAPFAC_M'], case.statics['MAPFAC_U'], case.statics['MAPFAC_V'],
                case.statics['F'], case.statics['E'], sina=case.statics['SINALPHA'], cosa=case.statics['COSALPHA'])
            if domain.parent_id == 0:
                frames.add_state(result.state, index=index)
            if index != 0:
                del result, case
                gc.collect()
                continue
            static, met, surface, soil_contract = metgrid_soil(case, cfg, fractional_seaice=fractional)
            declared_count = run.controls.get('domains', {}).get('num_metgrid_soil_levels')
            if declared_count is not None and declared_count != [len(soil_contract.get('source_nodes', soil_contract.get('source_layers')))]:
                raise ValueError('num_metgrid_soil_levels differs from the actual soil coordinate count')
        times = tuple(parse_met_em_name(path)[1] for path in files)
        if domain.parent_id == 0:
            check_met_em_series([series[i] for i in range(len(files))], interval_seconds=run.interval_seconds,
                start_time=exp.start_time, end_time=exp.start_time+timedelta(seconds=exp.run_seconds))
        boundaries = frames.build(times) if domain.parent_id == 0 else None
        if domain.parent_id == 0:
            root_boundaries = boundaries
            attach_lateral_boundaries(result.state, boundaries)
        static_path = directory/f'static-d{domain.grid_id:02d}.npz'
        # Kept byte for byte on a reuse: its digest is a member of the
        # prepared cache's identity, and a rewritten archive carries a
        # new zip timestamp, which would make the cache beside it
        # unreadable for the run that wrote it.
        if not (reused and static_path.exists()):
            np.savez(static_path, **static)
        artifacts[f'static_d{domain.grid_id:02d}'] = static_path
        identity = prepared_cache_identity(bridge_manifest_sha256=_sha(implementation),
            source_manifest_sha256=_sha(source_manifest), static_cache_sha256=_sha(static_path),
            namelist_sha256=_sha(config_path), domain_config=domain,
            forcing_offsets_seconds=[(time-exp.start_time).total_seconds() for time in times],
            source_identity={'handoff':'WPS metgrid','domain':domain.grid_id})
        cache = directory/f'd{domain.grid_id:02d}'
        receipt = {'soil_contract':soil_contract,'initialization_controls':controls,
            'field_map':dict(case.field_map),'field_units':dict(case.variable_units),
            'notes':list(case.notes),'landuse':dict(case.landuse),'fractional_seaice':fractional}
        if not (reused and cache.exists()):
            write_prepared_cache(cache, identity=identity, initial_result=result, met=met,
                boundaries=boundaries, surface=surface, metadata=receipt)
        # The reader is what proves a kept cache still belongs to this
        # run: it refuses, by name, a bundle whose identity differs.
        reader = PreparedCacheReader(cache, expected_identity=identity)
        artifacts[f'cache_header_d{domain.grid_id:02d}'] = cache/'header.json'
        bundles.append(MetemDomainBundle(domain.grid_id,cache,identity,reader,static,
            {'cache_content':reader.verify_all()['content_sha256'],'static':_sha(static_path)},
            WrfLanduseIdentity({key:case.landuse[key] for key in ('MMINLU','ISWATER','ISLAKE','ISICE')}),
            fractional, int(case.landuse['ISOILWATER'])))
        domain_receipts[f'd{domain.grid_id:02d}'] = receipt
        del result, case, met, surface, series, frames
        gc.collect()
    receipt_path = directory/'metgrid-import.json'
    if reused:
        # This is the original preparation's observation, not today's
        # free-memory reading. Reusing its science does not rewrite it.
        memory_receipt = json.loads(receipt_path.read_text(encoding='utf-8'))['memory_admission']
    write_document(receipt_path, json_bytes({'schema':'gpuwm-metgrid-import-v1','source_files':source_hashes,
        'vertical_coordinate':vertical_policy,'vertical_generation':vertical_generation,
        'domains':domain_receipts,'memory_admission':memory_receipt,
        'namelist_translation':asdict(run.substitution_report),
        'namelist_translation_text':run.substitution_report.format()}), reused=reused)
    artifacts['preparation_receipt'] = receipt_path
    for name,path in artifacts.items():
        if name in source_hashes and _sha(path) != source_hashes[name]:
            raise ValueError(f'{path}: changed while met_em preparation was reading it')
    hashes = {name:_sha(path) for name,path in artifacts.items()}
    return WrfTreeInputs(directory,config_path,exp,grids,tuple(bundles),
        tuple((parse_met_em_name(path)[1]-exp.start_time).total_seconds()/3600 for path in run.paths[1]),
        run.interval_seconds,MappingProxyType({'handoff':'WPS metgrid','forcing_origin':'not declared by input files'}),
        resolve_execution_plan(exp),MappingProxyType(hashes),MappingProxyType(artifacts),root_boundaries,source='met_em')


DOOR = 'gpuwm run --met-em'


def run_metem_forecast(directory, outdir, *, run_seconds=None, restart=None,
                       io_mode='history', health_debug=False, gpu_uuid=None,
                       exclusive_gpu=True, rrtmg_variant=None, vertical_grid=None,
                       render_products=None, render_dir=None, progress_options=None,
                       relaunched=False, vertical_levels=None, allow_shared_gpu=False,
                       output_owner=None):
    import os
    import subprocess
    import sys
    import time
    from gpuwm.metem_door import resolve_metem_run, check_analyzed_scalar_capabilities
    from gpuwm.stage_reuse import claim_run_output
    from gpuwm.wrfinput_forecast import announce_wrf_substitutions, worker_exit_status
    started = time.perf_counter()
    run = resolve_metem_run(directory, rrtmg_variant=rrtmg_variant)
    # PLAN REVIEW, and it happens HERE.  Both of these used to fire inside
    # the spawned worker, after select_gpu, after the GPU file lock and
    # after outdir.mkdir: a refusal raised once the run had started.  The
    # resolved ``run`` is already in hand, so both are answerable before
    # line one of the launch, and the worker resolves the same selector
    # from the same bytes and gets the same answer.
    if run_seconds is not None:
        metem_window_seconds(run, run_seconds)
    resolve_metem_vertical(run, run.toml_text, vertical_grid=vertical_grid,
                           vertical_levels=vertical_levels)
    # And the analyzed-field capability question, for the same reason and
    # through the same function the preparation's memory admission asks.
    # It used to be reachable only from inside prepare_metem_run, which
    # the supervised parent reaches after select_gpu, the GPU file lock
    # and outdir.mkdir, so a met_em carrying a field this build cannot
    # transport was refused once the run had started.
    check_analyzed_scalar_capabilities(run)
    # Plan review, and before the card: a refused --outdir used to be
    # an errno raised inside the worker child, minutes after this
    # process had reserved a GPU for it.
    try:
        output_claim = claim_run_output(outdir, flag='--outdir',
                                  protected_roots=(Path(directory),), resume=restart,
                                  owner_token=output_owner)
    except (ValueError, FileExistsError) as error:
        print(f'{DOOR}: --outdir refused: {error}', file=sys.stderr)
        return 2
    outdir = output_claim.path
    try:
        # The supervised child re-enters this door after the parent below
        # has already printed the pair at plan review.
        missing = announce_render_readiness(DOOR, announce=not relaunched)
        if exclusive_gpu:
            from gpuwm.supervisor import (select_gpu, preflight_exclusive_gpu,
                                          priced_reservation_bytes, GPUFileLock)
            gpu = select_gpu(gpu_uuid)
            command = [sys.executable,'-m','gpuwm.metem_forecast','--met-em',str(Path(directory).resolve()),
                       '--outdir',str(outdir),'--io-mode',io_mode,'--_worker',
                       '--_output-owner',output_claim.token]
            if rrtmg_variant is not None: command += ['--rrtmg-variant',rrtmg_variant]
            if vertical_grid is not None: command += ['--vertical-grid',vertical_grid]
            if vertical_levels is not None: command += ['--vertical-levels',str(int(vertical_levels))]
            if run_seconds is not None: command += ['--run-seconds',str(run_seconds)]
            if restart is not None: command += ['--restart',str(Path(restart).resolve())]
            if health_debug: command += ['--health-debug']
            # Carried to the child, which is where the run actually happens:
            # a product or progress flag dropped here is a flag that did
            # nothing on the supervised path people use by default.
            if render_products is not None: command += ['--products',str(render_products)]
            if render_dir is not None: command += ['--render-dir',str(Path(render_dir).resolve())]
            command += ProgressOptions.worker_flags(progress_options)
            with GPUFileLock(gpu.uuid, run_id=f'metgrid-{os.getpid()}'):
                # The card is priced against THIS run's reservation, from the
                # experiment the door has already resolved and through the
                # same function `gpuwm run` prices from, so a co-tenant that
                # fits beside this run is admitted at both doors and one that
                # does not is refused at both.
                preflight_exclusive_gpu(gpu.uuid, approved_pids={os.getpid()},
                                        allow_shared_gpu=allow_shared_gpu,
                                        reservation_bytes=priced_reservation_bytes(run.experiment))
                return worker_exit_status(subprocess.run(command,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu.uuid),check=False).returncode)
        if gpu_uuid is not None:
            raise ValueError('--gpu-uuid requires the default fresh worker; remove --no-supervise')
        from gpuwm.go_cli import GoStageFailed
        from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
        from gpuwm.filesystem_paths import io_path
        worker_output = io_path(outdir)
        print('met_em: preparing native initial states and lateral boundaries.', flush=True)
        inputs = prepare_metem_run(run,worker_output/'input',run_seconds=run_seconds,
                                   vertical_grid=vertical_grid,vertical_levels=vertical_levels)
        announce_wrf_substitutions(run, inputs.prepared_root/'metgrid-import.json')
        plan = door_render_plan(worker_output, render_products=render_products,
                                render_dir=None if render_dir is None else io_path(render_dir),
                                init=inputs.experiment.start_time, can_draw=missing is None)
        first_products = arm_door_first_products(plan, outdir=worker_output, started=started)
        run_prepared_tree(inputs,output_directory=worker_output,io_mode=io_mode,
            restart=None if restart is None else io_path(restart),
            health_debug=health_debug,progress_options=progress_options,
            initialization=MetemInitialization(inputs),
            **({} if first_products is None else {'first_products': first_products}))
        try:
            draw_door_products(plan, first_products=first_products, door=DOOR)
        except GoStageFailed as failure:
            print(f'{DOOR}: the forecast finished and its output is in {outdir}; '
                  f'the render stage exited {failure.code}.', file=sys.stderr)
            return failure.code
        return 0
    finally:
        output_claim.close()


def build_parser():
    """This worker's own command line.

    A function rather than a block inside :func:`main` so the flags it
    registers can be read without running the door, which is how the
    two WRF-input doors are held to the same set.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--met-em',type=Path,required=True)
    parser.add_argument('--rrtmg-variant',choices=('rrtmg_legacy','rte-rrtmgp'),default=None)
    parser.add_argument('--vertical-grid',metavar='SELECTOR',
        help='vertical ladder selector: '+', '.join(METEM_VERTICAL_LADDERS)+
             '. The explicit ladder is spelled explicit:PATH, a TOML file carrying an '
             'eta_levels array. An explicit selector replaces the namelist eta_levels; '
             'omitting it keeps them, or generates WRF automatic levels when there are none')
    parser.add_argument('--vertical-levels',metavar='N',type=int,default=None,
        help='number of full eta levels (WRF e_vert) the resolved ladder carries, '
             'in place of the producing namelist e_vert. wrf-auto and native '
             'generate N levels; explicit:PATH takes its count from the file and '
             'refuses a different N')
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--run-seconds',type=float)
    parser.add_argument('--restart',type=Path)
    parser.add_argument('--io-mode',choices=('history','none'),default='history')
    parser.add_argument('--health-debug',action='store_true')
    parser.add_argument('--products',dest='render_products',default=None,metavar='LIST',
        help="which product plots this run draws, in gpuwm render's own spelling: a "
             'comma-separated list, `all`, or `none` for no pictures.  Absent draws the '
             'default catalog at the end of the run')
    parser.add_argument('--render-dir',type=Path,default=None,metavar='DIR',
        help="where this run's pictures go; defaults to OUTDIR/" + RENDER_ROOT_NAME)
    add_progress_arguments(parser)
    parser.add_argument('--allow-shared-gpu',action='store_true',
        help='proceed when another CUDA compute process holds the selected GPU. The requested configuration is retained')
    parser.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--_output-owner',help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    import sys
    args = build_parser().parse_args(argv)
    from gpuwm.provenance_gate import announce
    announce(DOOR)
    try:
        return run_metem_forecast(args.met_em,args.outdir,run_seconds=args.run_seconds,
            restart=args.restart,io_mode=args.io_mode,health_debug=args.health_debug,
            exclusive_gpu=not args._worker,rrtmg_variant=args.rrtmg_variant,vertical_grid=args.vertical_grid,
            render_products=args.render_products,render_dir=args.render_dir,
            progress_options=ProgressOptions.from_args(args),relaunched=args._worker,
            vertical_levels=args.vertical_levels,allow_shared_gpu=args.allow_shared_gpu,
            output_owner=args._output_owner)
    except (ValueError,OSError) as error:
        print(f'{DOOR}: {error}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
