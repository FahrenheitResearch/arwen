"""Prepare WPS met_em through native initialization and the shared forecast runtime."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from gpuwm.forecast_initialization import DomainInitialization
from gpuwm.wrfinput_forecast import WrfTreeInputs, WrfLanduseIdentity, _sha


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


def resolve_metem_vertical(run, text, *, vertical_grid=None, cpu_bridge=None):
    """Materialize eta once; explicit arrays retain their exact TOML bytes."""
    if vertical_grid not in (None, 'native'):
        raise ValueError('vertical_grid must be native or omitted')
    if run.experiment.vertical.eta_levels:
        if vertical_grid is not None:
            raise ValueError('namelist eta_levels is explicit; remove --vertical-grid to preserve it')
        return text, 'explicit namelist eta_levels', None
    e_vert = run.experiment.root.run.nz + 1
    if vertical_grid == 'native':
        from gpuwm.core.grid import resample_eta_levels
        from gpuwm.native_wrf_contract import CERTIFIED_ETA_LEVELS
        eta = resample_eta_levels(CERTIFIED_ETA_LEVELS, e_vert - 1)
        policy = 'explicitly selected ArWen native eta profile resampled to e_vert'
        receipt = {'algorithm': 'ArWen-native-profile', 'e_vert': e_vert}
    else:
        from gpuwm.ingest.eta import generate_wrf_eta
        eta, receipt = generate_wrf_eta(e_vert,
            domains=run.controls.get('domains', {}),
            p_top=run.experiment.vertical.p_top,
            base_temp=run.experiment.root.run.base_temp, cpu_bridge=cpu_bridge)
        policy = 'WRF automatic eta levels generated by shared Rust compute_eta'
    if text.count('[shared]\n') != 1:
        raise ValueError('resolved experiment must carry one shared vertical coordinate')
    text = text.replace('[shared]\n', '[shared]\neta_levels = '+repr(eta.tolist())+'\n', 1)
    return text, policy, receipt


def prepare_metem_run(run, directory, *, run_seconds=None, preprocess_backend='cuda', cpu_bridge=None, vertical_grid=None):
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
    from gpuwm.ingest.lateral_bc import StateBoundaryFrames, start_last_forcing_order, attach_lateral_boundaries
    from gpuwm.ingest.prepared_cache import prepared_cache_identity, write_prepared_cache, PreparedCacheReader
    from gpuwm.prepared_domain_tree_forecast import resolve_execution_plan

    text = run.toml_text
    if run_seconds is not None:
        if not np.isfinite(run_seconds) or not 0 < run_seconds <= run.experiment.run_seconds:
            raise ValueError('--run-seconds must shorten the producing namelist duration')
        text, count = re.subn(r'(?m)^run_seconds\s*=.*$', f'run_seconds = {float(run_seconds)}', text)
        if count != 1:
            raise ValueError('resolved experiment must carry one duration')
    text, vertical_policy, vertical_generation = resolve_metem_vertical(
        run, text, vertical_grid=vertical_grid, cpu_bridge=cpu_bridge)
    if vertical_generation is not None:
        label = ('ArWen native profile' if vertical_grid == 'native' else
                 'WRF automatic eta option ' + str(vertical_generation['controls']['auto_levels_opt']))
        print(f'met_em: using {label} with {run.experiment.root.run.nz} mass levels.', flush=True)
    exp = build_experiment(tomllib.loads(text), source=str(run.namelist_input))
    if config_declares_follow_source(exp):
        raise ValueError('moving nests need terrain and land-surface coverage at future positions; these met_em files contain only the initial footprints. Supply a native prepared statics corridor')
    memory_receipt = metgrid_memory_admission(run, exp)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory/'experiment.toml'
    config_path.write_text(text, encoding='utf-8')
    backend = resolve_preprocess_backend(preprocess_backend, cpu_bridge=cpu_bridge)
    implementation = directory/'preprocess.json'
    _json(implementation, backend.receipt())
    artifacts = {'experiment_config':config_path, 'namelist_input':run.namelist_input,
                 'preprocess_implementation':implementation, 'adapter_code':Path(__file__).resolve(),
                 'field_adapter_code':Path(__file__).with_name('ingest')/'metem.py',
                 **{f'met_em_d{gid:02d}_{index}':path for gid,files in run.paths.items() for index,path in enumerate(files)}}
    source_hashes = {name:_sha(path) for name,path in artifacts.items()}
    source_manifest = directory/'source.json'
    _json(source_manifest, source_hashes)
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
        write_prepared_cache(cache, identity=identity, initial_result=result, met=met,
            boundaries=boundaries, surface=surface, metadata=receipt)
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
    _json(receipt_path, {'schema':'gpuwm-metgrid-import-v1','source_files':source_hashes,
        'vertical_coordinate':vertical_policy,'vertical_generation':vertical_generation,
        'domains':domain_receipts,'memory_admission':memory_receipt,
        'namelist_translation':asdict(run.substitution_report),
        'namelist_translation_text':run.substitution_report.format()})
    artifacts['preparation_receipt'] = receipt_path
    for name,path in artifacts.items():
        if name in source_hashes and _sha(path) != source_hashes[name]:
            raise ValueError(f'{path}: changed while met_em preparation was reading it')
    hashes = {name:_sha(path) for name,path in artifacts.items()}
    return WrfTreeInputs(directory,config_path,exp,grids,tuple(bundles),
        tuple((parse_met_em_name(path)[1]-exp.start_time).total_seconds()/3600 for path in run.paths[1]),
        run.interval_seconds,MappingProxyType({'handoff':'WPS metgrid','forcing_origin':'not declared by input files'}),
        resolve_execution_plan(exp),MappingProxyType(hashes),MappingProxyType(artifacts),root_boundaries,source='met_em')


def run_metem_forecast(directory, outdir, *, run_seconds=None, restart=None,
                       io_mode='history', health_debug=False, gpu_uuid=None,
                       exclusive_gpu=True, rrtmg_variant=None, vertical_grid=None):
    import os
    import subprocess
    import sys
    from gpuwm.metem_door import resolve_metem_run
    from gpuwm.wrfinput_forecast import announce_wrf_substitutions, worker_exit_status
    run = resolve_metem_run(directory, rrtmg_variant=rrtmg_variant)
    if exclusive_gpu:
        from gpuwm.supervisor import select_gpu, preflight_exclusive_gpu, GPUFileLock
        gpu = select_gpu(gpu_uuid)
        command = [sys.executable,'-m','gpuwm.metem_forecast','--met-em',str(Path(directory).resolve()),
                   '--outdir',str(Path(outdir).resolve()),'--io-mode',io_mode,'--_worker']
        if rrtmg_variant is not None: command += ['--rrtmg-variant',rrtmg_variant]
        if vertical_grid is not None: command += ['--vertical-grid',vertical_grid]
        if run_seconds is not None: command += ['--run-seconds',str(run_seconds)]
        if restart is not None: command += ['--restart',str(Path(restart).resolve())]
        if health_debug: command += ['--health-debug']
        with GPUFileLock(gpu.uuid, run_id=f'metgrid-{os.getpid()}'):
            preflight_exclusive_gpu(gpu.uuid, approved_pids={os.getpid()})
            return worker_exit_status(subprocess.run(command,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu.uuid),check=False).returncode)
    if gpu_uuid is not None:
        raise ValueError('--gpu-uuid requires the default fresh worker; remove --no-supervise')
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    announce_wrf_substitutions(run, outdir/'input'/'metgrid-import.json')
    print('met_em: preparing native initial states and lateral boundaries.', flush=True)
    inputs = prepare_metem_run(run,outdir/'input',run_seconds=run_seconds,vertical_grid=vertical_grid)
    run_prepared_tree(inputs,output_directory=outdir,io_mode=io_mode,restart=restart,
        health_debug=health_debug,initialization=MetemInitialization(inputs))
    return 0


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--met-em',type=Path,required=True)
    parser.add_argument('--rrtmg-variant',choices=('rrtmg_legacy','rte-rrtmgp'),default=None)
    parser.add_argument('--vertical-grid',choices=('native',),help='explicitly choose native eta initialization when the namelist has no eta_levels')
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--run-seconds',type=float)
    parser.add_argument('--restart',type=Path)
    parser.add_argument('--io-mode',choices=('history','none'),default='history')
    parser.add_argument('--health-debug',action='store_true')
    parser.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from gpuwm.provenance_gate import announce
    announce('gpuwm run --met-em')
    try:
        return run_metem_forecast(args.met_em,args.outdir,run_seconds=args.run_seconds,
            restart=args.restart,io_mode=args.io_mode,health_debug=args.health_debug,
            exclusive_gpu=not args._worker,rrtmg_variant=args.rrtmg_variant,vertical_grid=args.vertical_grid)
    except (ValueError,OSError) as error:
        print(f'gpuwm run --met-em: {error}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
