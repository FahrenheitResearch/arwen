"""Forecast WRF real.exe inputs through ArWen's shared domain-tree runtime."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gpuwm.forecast_initialization import DomainInitialization


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class WrfLanduseIdentity:
    attributes: Mapping[str, object]

    def landuse_global_attrs(self):
        return dict(self.attributes)


@dataclass(frozen=True)
class WrfDomainBundle:
    grid_id: int
    restored: object
    static_fields: Mapping[str, np.ndarray]
    authority_sha256: Mapping[str, str]
    landuse: object
    geog_selection: WrfLanduseIdentity


@dataclass(frozen=True)
class WrfTreeInputs:
    prepared_root: Path
    experiment_config: Path
    experiment: object
    grids: tuple[object, ...]
    domains: tuple[WrfDomainBundle, ...]
    forcing_hours: tuple[float, ...]
    boundary_interval_seconds: int
    source_identity: Mapping[str, object]
    execution_plan: Mapping[str, object]
    authority_sha256: Mapping[str, str]
    artifact_paths: Mapping[str, Path]
    boundaries: object
    source: str = 'wrfinput'
    statics_corridor: object | None = None


@dataclass(frozen=True)
class WrfInitialState:
    state: object
    coord: object
    base: object


def wrf_initial_result(restored, state):
    """Keep the actual WRF coordinate and base state for shared runtime output."""
    from gpuwm.ingest.wrfinput import wrf_coordinate_and_base
    coord, base = wrf_coordinate_and_base(restored)
    return WrfInitialState(state, coord, base)


class WrfInitialization:
    def __init__(self, inputs: WrfTreeInputs):
        self.inputs = inputs
        self.lateral_boundaries = inputs.boundaries
        from gpuwm.case_data import trace_gas_overrides_from_config
        self.trace_gas_overrides = trace_gas_overrides_from_config(
            inputs.experiment_config,
            expected_sha256=inputs.authority_sha256["experiment_config"])

    def restore_domain(self, domain, grid, bundle, *, start_time,
                       scratch_arena, dycore_state_workspace):
        from gpuwm.ingest.wrfinput import restore_domain_state, initialize_wrfinput_physics
        from gpuwm.ingest.lateral_bc import attach_lateral_boundaries

        state = restore_domain_state(bundle.restored, domain.run,
                                     scratch_arena=scratch_arena,
                                     dycore_state_workspace=dycore_state_workspace)
        if domain.parent_id == 0:
            attach_lateral_boundaries(state, self.inputs.boundaries)
        from gpuwm.runtime import declared_constant_glw
        def initialize():
            from gpuwm.core.cam_ozone import cam_ozone_setup, ozone_parent_for
            cam = cam_ozone_setup(exp=self.inputs.experiment, dc=domain, grid=grid)
            from gpuwm.core.radiation_composition import make_radiation
            radiation = make_radiation(
                domain.run, start_time, bundle.restored.raw["XLAT"],
                bundle.restored.raw["XLONG"], p_top=self.inputs.experiment.vertical.p_top,
                column_chunk=self.inputs.experiment.column_chunk,
                trace_gas_overrides=self.trace_gas_overrides,
                ozone_parent=ozone_parent_for(cam))
            return initialize_wrfinput_physics(
                state, bundle.restored, domain.run, radiation=radiation,
                radiation_start_time=start_time,
                radiation_latitude=bundle.restored.raw['XLAT'],
                radiation_longitude=bundle.restored.raw['XLONG'],
                landuse=bundle.landuse,
                constant_glw_wm2=declared_constant_glw(self.inputs.experiment),
                **({"cam_ozone": cam} if cam is not None else {}))
        return DomainInitialization(wrf_initial_result(bundle.restored, state), initialize)

    def verify_inputs(self, inputs):
        current = {name: _sha(path) for name, path in inputs.artifact_paths.items()}
        if current != dict(inputs.authority_sha256):
            changed = sorted(name for name in current if current[name] != inputs.authority_sha256.get(name))
            raise RuntimeError(f'WRF input artifacts changed during execution: {changed}')

    def domain_content_sha256(self, bundle):
        return bundle.authority_sha256['wrfinput']

    def domain_metadata(self, bundle):
        return {}


def prepare_wrf_run(run, directory: Path, *, run_seconds: float | None = None) -> WrfTreeInputs:
    """Validate the complete CPU handoff and bind the exact input bytes."""
    import re
    import tomllib
    from gpuwm.config import soil_layer_count
    from gpuwm.core.landuse import initialize_landuse
    from gpuwm.namelist_import import parse_namelist
    from gpuwm.experiment import build_experiment
    from gpuwm.ingest.wrfinput import read_wrfinput, read_wrfbdy
    from gpuwm.prepared_domain_tree_forecast import resolve_execution_plan
    from gpuwm.static.corridor import config_declares_follow_source
    from gpuwm.static.projection import grids_from_projection_config

    config = run.toml_text
    if run_seconds is not None:
        if not np.isfinite(run_seconds) or not 0 < run_seconds <= run.coverage.coverage_seconds:
            raise ValueError('run duration must be positive and inside wrfbdy coverage')
        config, count = re.subn(r'(?m)^run_seconds\s*=.*$', f'run_seconds = {float(run_seconds)}', config)
        if count != 1:
            raise ValueError('resolved experiment does not carry exactly one run_seconds')
    exp = build_experiment(tomllib.loads(config), source=str(run.namelist_input))
    if config_declares_follow_source(exp):
        raise ValueError('moving nests require terrain and land-surface coverage for their future positions; this WRF directory contains only the initial footprints. Supply a prepared statics corridor through native preparation')
    grids = tuple(grids_from_projection_config(exp))
    paths = {'adapter_code': Path(__file__).resolve(),
             'state_adapter_code': Path(__file__).with_name('ingest') / 'wrfinput.py',
             'namelist_input': run.namelist_input, 'wrfbdy': run.wrfbdy_path,
             **{f'wrfinput_d{gid:02d}': path for gid, path in run.wrfinput_paths.items()}}
    original_hashes = {name: _sha(path) for name, path in paths.items()}
    fractional_values = parse_namelist(run.namelist_input).get('physics', {}).get('fractional_seaice', [0])
    fractional_seaice = bool(fractional_values[0])
    bundles = []
    for domain in exp.domains:
        cfg = domain.run
        dimensions = {'west_east':cfg.nx,'west_east_stag':cfg.nx+1,
                      'south_north':cfg.ny,'south_north_stag':cfg.ny+1,
                      'bottom_top':cfg.nz,'bottom_top_stag':cfg.nz+1,
                      'soil_layers_stag':soil_layer_count(cfg)}
        restored = read_wrfinput(run.wrfinput_paths[domain.grid_id], expected_dimensions=dimensions, cfg=cfg)
        attrs = restored.global_attributes
        names = ('MMINLU', 'NUM_LAND_CAT', 'ISWATER', 'ISLAKE', 'ISICE', 'ISURBAN', 'ISOILWATER')
        missing = sorted(set(names) - set(attrs))
        if missing:
            raise ValueError(f'{restored.path}: missing land-use identity {missing}')
        land_attrs = {name: attrs[name] for name in names}
        landuse = initialize_landuse(
            restored.raw['LU_INDEX'], soil_type=restored.raw['ISLTYP'],
            landmask=restored.raw['LANDMASK'], snow=restored.raw['SNOW'],
            xice=restored.raw.get('XICE', restored.raw.get('SEAICE')),
            valid_time=domain.start_time or exp.start_time, cen_lat=float(attrs['CEN_LAT']),
            mminlu=str(attrs['MMINLU']), iswater=int(attrs['ISWATER']),
            islake=int(attrs['ISLAKE']), isice=int(attrs['ISICE']),
            isoilwater=int(attrs['ISOILWATER']), fractional_seaice=fractional_seaice,
            soil_temperature=restored.raw['TSLB'], sst=restored.raw.get('SST'))
        static = {name:restored.raw[name] for name in ('LANDMASK','LU_INDEX','ISLTYP','MAPFAC_M','MAPFAC_U','MAPFAC_V','F','E')}
        static['HGT_M'] = restored.raw['HGT']
        bundles.append(WrfDomainBundle(domain.grid_id, restored, MappingProxyType(static),
                       MappingProxyType({'wrfinput':original_hashes[f'wrfinput_d{domain.grid_id:02d}']}),
                       landuse, WrfLanduseIdentity(MappingProxyType(land_attrs))))
    boundaries = read_wrfbdy(run.wrfbdy_path, run_seconds=exp.run_seconds,
                            restored=bundles[0].restored,
                            forcing_interval_seconds=run.coverage.forcing_interval_seconds,
                            spec_bdy_width=exp.root.run.spec_bdy_width,
                            cfg=exp.root.run)
    for name, path in paths.items():
        if _sha(path) != original_hashes[name]:
            raise ValueError(f'{path}: changed while its input fields were read')
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory/'experiment.toml'
    config_path.write_text(config, encoding='utf-8')
    receipt_path = directory/'wrf-import.json'
    receipt = {'schema':'gpuwm-wrf-input-import-v1', 'source_files':original_hashes,
               'initial_temperature':'T is dry perturbation potential temperature',
               'boundary_temperature':(
                   'moist coupled THM converted to dry theta at each forcing time'
                   if int(bundles[0].restored.global_attributes['USE_THETA_M']) == 1
                   else 'dry perturbation potential temperature'),
               'vertical_coordinate':'file ZNW, compared with explicit namelist eta when present',
               'initial_boundary_pair':'verified for every consumed field and side',
               'namelist_translation':asdict(run.substitution_report),
               'namelist_translation_text':run.substitution_report.format()}
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    paths.update(experiment_config=config_path, preparation_receipt=receipt_path)
    hashes = {name:_sha(path) for name,path in paths.items()}
    return WrfTreeInputs(directory, config_path, exp, grids, tuple(bundles),
                         tuple((t-exp.start_time).total_seconds()/3600 for t in (*run.coverage.times, run.coverage.end)),
                         int(run.coverage.forcing_interval_seconds),
                         MappingProxyType({'handoff':'WRF real.exe', 'forcing_origin':'not declared by input files'}),
                         resolve_execution_plan(exp), MappingProxyType(hashes),
                         MappingProxyType(paths), boundaries)


def announce_wrf_substitutions(run, receipt_path):
    """Show the actual scheme changes; retain the full translator record.

    A declared divergence (``item.reason``) prints its reason: the user is
    handing over a WRF run and must read, at the terminal, that ArWen will
    integrate a different variable and why.
    """
    for item in run.substitution_report.substitutions:
        print(f'WRF import: {item.wrf_name} ({item.key}={item.wrf_value}) '
              f'→ {item.gpuwm_name} ({item.gpuwm_key}={item.gpuwm_value}).'
              + (f'  Declared divergence: {item.reason}' if item.reason else ''),
              flush=True)
    if run.substitution_report.substitutions:
        print(f'WRF import details: {receipt_path}', flush=True)


def worker_exit_status(code):
    """Preserve a worker failure and name a terminating signal to the caller."""
    if code == 0:
        return 0
    import sys
    from gpuwm.runplan import StageExitError
    print(str(StageExitError('WRF forecast worker', code)), file=sys.stderr)
    return min(255, 128 - code) if code < 0 else code


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrfinput', type=Path, required=True, help='real.exe directory with wrfinput, wrfbdy and namelist.input')
    parser.add_argument('--rrtmg-variant', choices=('rrtmg_legacy','rte-rrtmgp'), default=None, help='preserve WRF RRTMG by default; choose rte-rrtmgp explicitly to change radiation')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--run-seconds', type=float, help='shorten the run inside the supplied boundary coverage')
    parser.add_argument('--io-mode', choices=('history','none'), default='history')
    parser.add_argument('--restart', type=Path)
    parser.add_argument('--health-debug', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return parser


def run_wrf_forecast(directory, outdir, *, run_seconds=None, restart=None,
                     io_mode="history", health_debug=False, gpu_uuid=None,
                     exclusive_gpu=True, rrtmg_variant=None):
    import os
    import subprocess
    import sys
    from gpuwm.wrfinput_door import resolve_wrfinput_run
    if exclusive_gpu:
        from gpuwm.supervisor import select_gpu, preflight_exclusive_gpu, GPUFileLock
        gpu = select_gpu(gpu_uuid)
        command = [sys.executable, '-m', 'gpuwm.wrfinput_forecast',
                   '--wrfinput', str(Path(directory).resolve()),
                   '--outdir', str(Path(outdir).resolve()), '--io-mode', io_mode, '--_worker']
        if rrtmg_variant is not None:
            command += ['--rrtmg-variant', rrtmg_variant]
        if run_seconds is not None:
            command += ['--run-seconds', str(run_seconds)]
        if restart is not None:
            command += ['--restart', str(Path(restart).resolve())]
        if health_debug:
            command += ['--health-debug']
        with GPUFileLock(gpu.uuid, run_id=f'wrf-input-{os.getpid()}'):
            preflight_exclusive_gpu(gpu.uuid, approved_pids={os.getpid()})
            return worker_exit_status(subprocess.run(
                command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu.uuid),
                check=False).returncode)
    if gpu_uuid is not None:
        raise ValueError('--gpu-uuid requires the default fresh worker; remove --no-supervise')
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    run = resolve_wrfinput_run(directory, rrtmg_variant=rrtmg_variant)
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    inputs = prepare_wrf_run(run, outdir/'input', run_seconds=run_seconds)
    announce_wrf_substitutions(run, outdir/'input'/'wrf-import.json')
    run_prepared_tree(inputs, output_directory=outdir, io_mode=io_mode,
                      restart=restart, health_debug=health_debug,
                      initialization=WrfInitialization(inputs))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    from gpuwm.provenance_gate import announce
    announce('gpuwm run --wrfinput')
    try:
        return run_wrf_forecast(args.wrfinput, args.outdir, run_seconds=args.run_seconds,
                               restart=args.restart, io_mode=args.io_mode,
                               health_debug=args.health_debug, exclusive_gpu=not args._worker,
                               rrtmg_variant=args.rrtmg_variant)
    except (ValueError, OSError) as error:
        import sys
        print(f'gpuwm run --wrfinput: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
