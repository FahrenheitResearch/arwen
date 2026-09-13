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
from gpuwm.progress_log import ProgressOptions, add_progress_arguments


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


def wrfinput_window_seconds(run, run_seconds):
    """Bind a requested wrfinput run window to the shared forcing check."""
    from gpuwm.ingest.preflight import check_forcing_window
    return check_forcing_window(run_seconds,
        coverage_seconds=run.coverage.coverage_seconds,
        source='gpuwm run --wrfinput', last_valid_time=run.coverage.end)


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
        # Not an inline copy of "the window is capped by the forcing" any
        # more: gpuwm/metem_forecast.py reaches the same function, so the
        # two doors say one sentence about one configuration.
        run_seconds = wrfinput_window_seconds(run, run_seconds)
        config, count = re.subn(r'(?m)^run_seconds\s*=.*$', f'run_seconds = {float(run_seconds)}', config)
        if count != 1:
            raise ValueError('resolved experiment does not carry exactly one run_seconds')
    exp = build_experiment(tomllib.loads(config), source=str(run.namelist_input))
    if config_declares_follow_source(exp):
        raise ValueError('moving nests require terrain and land-surface coverage for their future positions; this WRF directory contains only the initial footprints. Supply a prepared statics corridor through native preparation')
    grids = tuple(grids_from_projection_config(exp))
    paths = {'adapter_code': Path(__file__).resolve(),
             'state_adapter_code': Path(__file__).with_name('ingest') / 'wrfinput.py',
             'surface_adapter_code': Path(__file__).with_name('ingest') / 'wrfinput_noahmp.py',
             'namelist_input': run.namelist_input, 'wrfbdy': run.wrfbdy_path,
             **{f'wrfinput_d{gid:02d}': path for gid, path in run.wrfinput_paths.items()}}
    original_hashes = {name: _sha(path) for name, path in paths.items()}
    from gpuwm.prepared_documents import preparation_directory, json_bytes, write_document

    directory, reused = preparation_directory(
        directory, receipt='wrf-import.json', source_files=original_hashes,
        documents={'experiment.toml': config.encode('utf-8')})
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
    directory.mkdir(parents=True, exist_ok=reused)
    config_path = directory/'experiment.toml'
    write_document(config_path, config.encode('utf-8'), reused=reused)
    receipt_path = directory/'wrf-import.json'
    receipt = {'schema':'gpuwm-wrf-input-import-v1', 'source_files':original_hashes,
               'initial_temperature':'T is dry perturbation potential temperature',
               'boundary_temperature':(
                   'moist coupled THM converted to dry theta at each forcing time'
                   if int(bundles[0].restored.global_attributes['USE_THETA_M']) == 1
                   else 'dry perturbation potential temperature'),
               'vertical_coordinate':'file ZNW, compared with explicit namelist eta when present',
               'initial_boundary_pair':'verified for every consumed field and side',
               'surface_input_dispositions': {
                   f'd{bundle.grid_id:02d}': dict(bundle.restored.surface_input_dispositions)
                   for bundle in bundles},
               'namelist_translation':asdict(run.substitution_report),
               'namelist_translation_text':run.substitution_report.format()}
    write_document(receipt_path, json_bytes(receipt), reused=reused)
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


#: Where a WRF-input door's pictures go when no ``--render-dir`` was
#: given.  Named once, and claimed once: the early render of the first
#: committed frame and the finalize render at the end have to file into
#: ONE folder, or the receipt that licenses the finalize skip describes
#: a directory nothing looks in.
RENDER_ROOT_NAME = 'png'


def announce_render_readiness(door: str, *, announce: bool = True) -> str | None:
    """Say at plan review whether this install can draw, and how to fix it.

    Asked of :func:`gpuwm.go_cli.render_extra_missing`, which is the one
    answer ``gpuwm go``'s render stage already uses, so a WRF-input run
    and a chained run cannot disagree about whether this install draws.

    An absent renderer is ANNOUNCED, never refused: the forecast is
    worth running without pictures.  What it must not be is discovered
    at the end, which is what happens when the question is asked after
    the forecast: this is called before a card is selected or reserved,
    so the reader who needs `gpuwm setup` reads it while nothing is
    running.

    ``announce=False`` asks the same question and keeps the answer
    quiet.  The supervised re-launch re-enters this door in a child
    process, after the parent has already printed the pair at plan
    review, and one run that cannot draw is one message and not two.
    """

    from gpuwm.go_cli import render_extra_missing

    missing = render_extra_missing()
    if missing is not None and announce:
        print(f'{door}: this run will draw no pictures: {missing}', flush=True)
        print('  remedy: gpuwm setup', flush=True)
        print('  # stages the rust render engine, which draws the full '
              'catalog', flush=True)
    return missing


def door_render_plan(outdir, *, render_products=None, render_dir=None,
                     init=None, can_draw=True) -> dict:
    """The plan dict this run's pictures are drawn from, claimed once.

    The very dict :func:`gpuwm.go_cli.render_command` and
    :func:`gpuwm.go_cli._render_stage` take, so a WRF-input run composes
    the same render command every other door composes and its products
    land where ``gpuwm render`` looks for them.

    The run folder is claimed HERE, before the forecast opens, through
    :func:`gpuwm.run_stamp.resolve` -- the single answer every front
    door asks.  ``render_command`` hardcodes ``--run-stamp off``
    (:func:`gpuwm.run_stamp.stage_flags`), which is a stage saying the
    folder is claimed above it; something has to do the claiming, and a
    door that left it to the render stage would have got no folder and
    no ``latest-run.txt`` at all.  Claiming it before the run is what
    makes the early render and the finalize render share one folder.

    ``render_products = 'none'`` claims nothing: the stage skips itself
    with its own sentence, and an unclaimed folder is not created.
    ``can_draw=False`` claims nothing for the same reason and from the
    same answer the announcement above was made from: an install
    without the render engine draws nothing, and a stamped run folder
    per run that never receives a picture is a directory a reader opens
    for nothing.
    """

    from gpuwm import run_stamp

    outdir = Path(outdir)
    root = (outdir / RENDER_ROOT_NAME if render_dir is None
            else Path(render_dir))
    plan = {'run': outdir, 'wrfout_dir': outdir / 'wrfout',
            'render': root, 'render_products': render_products}
    if not can_draw or str(render_products or '').strip().lower() == 'none':
        return plan
    plan['render'] = run_stamp.resolve(root, init=init, create=True,
                                       publish=False)
    return plan


def arm_door_first_products(plan: dict, *, outdir, started):
    """This door's early render of the first committed frame, or ``None``.

    Built by the shared
    :func:`gpuwm.prepared_single_domain_forecast._route_owned_first_products`
    rather than by a second decision here, so "did this run ask for
    pictures early?" is answered for these two doors exactly as it is
    answered for every other one: off by ABSENCE of a product spec,
    off for ``none``, on otherwise.
    """

    from types import SimpleNamespace

    from gpuwm import prepared_single_domain_forecast as prepared_single

    return prepared_single._route_owned_first_products(
        SimpleNamespace(render_products=plan['render_products'],
                        render_dir=plan['render']),
        outdir=Path(outdir), observer=None, started=started)


def draw_door_products(plan: dict, *, first_products=None, door: str) -> bool:
    """Draw this run's pictures, then publish the pointer to them.

    The stage is :func:`gpuwm.go_cli._render_stage` verbatim, trigger
    included: the early render is joined and its digest-verified frame
    is dropped from this stage's list, so a frame drawn at second 46 is
    not redrawn at the end.

    ``latest-run.txt`` is published HERE, and only once a PNG is on
    disk.  The stage cannot publish it -- it runs with ``--run-stamp
    off``, so :mod:`gpuwm.render` allocates no folder of its own and its
    own pointer write is a no-op -- and a pointer written before
    anything was drawn names an empty folder, which is the one thing a
    script following the documented pointer must never read.
    """

    from types import SimpleNamespace

    from gpuwm import go_cli, run_stamp

    drawn = go_cli._render_stage(
        plan, explain=False, door=door,
        observer=SimpleNamespace(first_products=first_products))
    folder = Path(plan['render'])
    if drawn and run_stamp.is_run_folder(folder) and next(folder.rglob('*.png'), None):
        run_stamp.record_latest(folder.parent, folder)
    return drawn


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrfinput', type=Path, required=True, help='real.exe directory with wrfinput, wrfbdy and namelist.input')
    parser.add_argument('--rrtmg-variant', choices=('rrtmg_legacy','rte-rrtmgp'), default=None, help='preserve WRF RRTMG by default; choose rte-rrtmgp explicitly to change radiation')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--run-seconds', type=float, help='shorten the run inside the supplied boundary coverage')
    parser.add_argument('--io-mode', choices=('history','none'), default='history')
    parser.add_argument('--restart', type=Path)
    parser.add_argument('--health-debug', action='store_true')
    parser.add_argument('--products', dest='render_products', default=None, metavar='LIST',
                        help='which product plots this run draws, in gpuwm render\'s own spelling: '
                             'a comma-separated list, `all`, or `none` for no pictures.  '
                             'Absent draws the default catalog at the end of the run')
    parser.add_argument('--render-dir', type=Path, default=None, metavar='DIR',
                        help='where this run\'s pictures go; defaults to OUTDIR/' + RENDER_ROOT_NAME)
    add_progress_arguments(parser)
    parser.add_argument('--allow-shared-gpu', action='store_true',
                        help='proceed when another CUDA compute process holds the selected GPU. The requested configuration is retained')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return parser


DOOR = 'gpuwm run --wrfinput'


def run_wrf_forecast(directory, outdir, *, run_seconds=None, restart=None,
                     io_mode="history", health_debug=False, gpu_uuid=None,
                     exclusive_gpu=True, rrtmg_variant=None,
                     render_products=None, render_dir=None,
                     progress_options=None, relaunched=False, allow_shared_gpu=False):
    import os
    import subprocess
    import sys
    import time
    from gpuwm.stage_reuse import claim_run_output
    from gpuwm.wrfinput_door import resolve_wrfinput_run
    started = time.perf_counter()
    run = resolve_wrfinput_run(directory, rrtmg_variant=rrtmg_variant)
    if run_seconds is not None:
        wrfinput_window_seconds(run, run_seconds)
    # PLAN REVIEW, and it happens here rather than in the worker child
    # for one reason: everything below this in the supervised branch
    # takes a card.  A bad --outdir used to reach the reader as
    # [Errno 17] from inside the child, after select_gpu, GPUFileLock
    # and preflight_exclusive_gpu had reserved a GPU for a run that was
    # never going to start.
    try:
        outdir = claim_run_output(outdir, flag='--outdir',
                                  protected_roots=(Path(directory),),
                                  resume=restart)
    except (ValueError, FileExistsError) as error:
        print(f'{DOOR}: --outdir refused: {error}', file=sys.stderr)
        return 2
    # Identity review includes the cold versus stepped surface-state
    # contract, before a supervised worker selects or reserves a card.
    # `relaunched` is the supervised child of the branch below, which
    # has already said this at plan review in the terminal both
    # processes print to.
    missing = announce_render_readiness(DOOR, announce=not relaunched)
    if exclusive_gpu:
        from gpuwm.supervisor import select_gpu, preflight_exclusive_gpu, GPUFileLock
        gpu = select_gpu(gpu_uuid)
        command = [sys.executable, '-m', 'gpuwm.wrfinput_forecast',
                   '--wrfinput', str(Path(directory).resolve()),
                   '--outdir', str(outdir), '--io-mode', io_mode, '--_worker']
        if rrtmg_variant is not None:
            command += ['--rrtmg-variant', rrtmg_variant]
        if run_seconds is not None:
            command += ['--run-seconds', str(run_seconds)]
        if restart is not None:
            command += ['--restart', str(Path(restart).resolve())]
        if health_debug:
            command += ['--health-debug']
        # The worker is this door's real body, so a flag this door was
        # given and did not pass on is a flag that did nothing.
        if render_products is not None:
            command += ['--products', str(render_products)]
        if render_dir is not None:
            command += ['--render-dir', str(Path(render_dir).resolve())]
        command += ProgressOptions.worker_flags(progress_options)
        with GPUFileLock(gpu.uuid, run_id=f'wrf-input-{os.getpid()}'):
            preflight_exclusive_gpu(gpu.uuid, approved_pids={os.getpid()},
                                    allow_shared_gpu=allow_shared_gpu)
            return worker_exit_status(subprocess.run(
                command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu.uuid),
                check=False).returncode)
    if gpu_uuid is not None:
        raise ValueError('--gpu-uuid requires the default fresh worker; remove --no-supervise')
    from gpuwm.go_cli import GoStageFailed
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    inputs = prepare_wrf_run(run, outdir/'input', run_seconds=run_seconds)
    announce_wrf_substitutions(run, inputs.prepared_root/'wrf-import.json')
    plan = door_render_plan(outdir, render_products=render_products,
                            render_dir=render_dir,
                            init=inputs.experiment.start_time,
                            can_draw=missing is None)
    first_products = arm_door_first_products(plan, outdir=outdir, started=started)
    run_prepared_tree(inputs, output_directory=outdir, io_mode=io_mode,
                      restart=restart, health_debug=health_debug,
                      progress_options=progress_options,
                      initialization=WrfInitialization(inputs),
                      **({} if first_products is None
                         else {'first_products': first_products}))
    try:
        draw_door_products(plan, first_products=first_products, door=DOOR)
    except GoStageFailed as failure:
        # The forecast is on disk and finished; the pictures are not.
        # Saying so beats both alternatives: a silent 0 hides that the
        # products this door now promises are missing, and a traceback
        # hides the forecast.
        print(f'{DOOR}: the forecast finished and its output is in {outdir}; '
              f'the render stage exited {failure.code}.', file=sys.stderr)
        return failure.code
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    from gpuwm.provenance_gate import announce
    announce(DOOR)
    try:
        return run_wrf_forecast(args.wrfinput, args.outdir, run_seconds=args.run_seconds,
                               restart=args.restart, io_mode=args.io_mode,
                               health_debug=args.health_debug, exclusive_gpu=not args._worker,
                               rrtmg_variant=args.rrtmg_variant,
                               render_products=args.render_products,
                               render_dir=args.render_dir,
                               progress_options=ProgressOptions.from_args(args),
                               relaunched=args._worker,
                               allow_shared_gpu=args.allow_shared_gpu)
    except (ValueError, OSError) as error:
        import sys
        print(f'{DOOR}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
