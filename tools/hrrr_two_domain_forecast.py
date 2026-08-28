#!/usr/bin/env python3
"""Run the native-HRRR 3-km -> 1-km one-way easy-physics experiment.

The same executable is used for the short health gate and the 12-hour run.
The root receives hourly HRRR lateral forcing; d02 is initialized from the
same f00 analysis and subsequently forced only by the production one-way nest
coupler.  Output is compact operational evidence rather than a WRF-parity
claim or a replacement for a full wrfout archive.
"""

from __future__ import annotations

from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import MappingProxyType

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.hrrr_state_proof import (  # noqa: E402
    _assert_frozen_namelist,
    _physics_receipt,
    _proc_io,
    _read_static,
    _sha256,
    _source_identity,
    _strict_json,
)
from tools.hrrr_single_domain_benchmark import _device_name  # noqa: E402
from gpuwm.ingest.hrrr_physics import initialize_hrrr_physics  # noqa: E402


def _experiment(eta, *, run_seconds: float, sample_interval_s: float):
    from gpuwm.experiment import build_experiment

    raw = {
        "experiment": {
            "name": "hrrr_native_easy_d01_d02",
            "start_time": datetime(2026, 7, 18),
            "run_seconds": float(run_seconds),
            "feedback": 0,
            "smooth_option": 0,
            "blend_width": 5,
            "spec_bdy_width": 5,
            "restart_interval_s": 0.0,
        },
        "projection": {
            "map_proj": "lambert",
            "ref_lat": 35.5028506728143,
            "ref_lon": -98.0021669285660,
            "truelat1": 38.5,
            "truelat2": 38.5,
            "stand_lon": -97.5,
        },
        "shared": {
            "nz": 49,
            "ztop": 20000.0,
            "p_top": 10000.0,
            "eta_levels": list(map(float, eta)),
            "hybrid_opt": 2,
            "etac": 0.2,
            "base_temp": 290.0,
            "time_step_sound": 4,
            "epssm": 0.5,
            "emdiv": 0.01,
            "hypsometric_opt": 2,
            "h_sca_adv_order": 5,
            "smdiv": 0.1,
            "moist": True,
            "mp_physics": 6,
            "wsm6_hail_opt": 0,
            "moist_adv_opt": 1,
            "km_opt": 4,
            "c_s": 0.25,
            "diff_6th_opt": 2,
            "diff_6th_slopeopt": 1,
            "diff_6th_thresh": 0.10,
            "w_damping": 1,
            "damp_opt": 3,
            "zdamp": 5000.0,
            "dampcoef": 0.2,
            "khdif": 0.0,
            "kvdif": 0.0,
            "spec_zone": 1,
            "relax_zone": 4,
            "terrain_opt": 1,
            "map_proj": 1,
            "sf_sfclay_physics": 91,
            "sf_surface_physics": 2,
            "bl_pbl_physics": 1,
            "bldt": 0.0,
            "ra_physics": 0,
            "ra_lw_physics": 0,
            "ra_sw_physics": 1,
            "cu_physics": 0,
            "cudt_minutes": 0.0,
        },
        "domain": [
            {
                "grid_id": 1,
                "parent_id": 0,
                "i_parent_start": 1,
                "j_parent_start": 1,
                "parent_grid_ratio": 1,
                "parent_time_step_ratio": 1,
                "nx": 199,
                "ny": 199,
                "time_step": 15,
                "dx": 2999.4213047435587,
                "dy": 2999.4213047435587,
                "history_interval_s": float(sample_interval_s),
                "specified": True,
                "nested": False,
                "radt": 3.0,
                "radt_minutes": 3.0,
                "diff_6th_factor": 0.10,
                "spec_exp": 0.0,
            },
            {
                "grid_id": 2,
                "parent_id": 1,
                "i_parent_start": 50,
                "j_parent_start": 50,
                "parent_grid_ratio": 3,
                "parent_time_step_ratio": 3,
                "nx": 300,
                "ny": 300,
                "history_interval_s": float(sample_interval_s),
                "specified": False,
                "nested": True,
                "radt": 1.0,
                "radt_minutes": 1.0,
                "diff_6th_factor": 0.08,
                "spec_exp": 0.0,
            },
        ],
    }
    return build_experiment(raw, "programmatic:native-HRRR-two-domain")


def _initialize_state(snapshot, dc, grid, static, eta, mapping_report):
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.ingest.hrrr import interpolate_hrrr_to_lambert
    from gpuwm.ingest.real import initialize_real

    started = time.perf_counter()
    met = interpolate_hrrr_to_lambert(
        snapshot, grid, target_landmask=static["LANDMASK"],
        soil_mapping_report=mapping_report)
    horizontal_seconds = time.perf_counter() - started
    started = time.perf_counter()
    coord = make_vertical_coord(
        dc.run.nz, hybrid_opt=dc.run.hybrid_opt, etac=dc.run.etac,
        eta_levels=eta)
    result = initialize_real(
        met, dc.run, coord, static["HGT_M"], grid=grid, p_top=10000.0,
        sfcp_to_sfcp=True)
    result.state.set_map_coriolis(
        static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
        static["F"], static["E"], sina=static["SINALPHA"],
        cosa=static["COSALPHA"])
    state_seconds = time.perf_counter() - started
    return result, met, horizontal_seconds, state_seconds


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(args):
    import resource
    import cupy as cp

    import gpuwm.core.dycore as dycore_module
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.dycore import stability_report
    from gpuwm.core.health import StateHealthValidator
    from gpuwm.core.model import (
        DomainNode,
        ExperimentState,
        ModelRuntimeStatus,
        execute_experiment,
    )
    from gpuwm.core.nest import NestCoupler
    from gpuwm.ingest.hrrr import (
        load_hrrr_native_series,
        load_hrrr_pipeline_ready_window,
    )
    from gpuwm.ingest.lateral_bc import (
        BoundaryInterval,
        LateralBoundaries,
        attach_lateral_boundaries,
        build_state_lateral_boundaries,
    )
    from gpuwm.static.lambert import grids_from_projection_config

    args.outdir.mkdir(parents=True, exist_ok=False)
    progress_path = args.outdir / "progress.json"
    eta = _assert_frozen_namelist(args.namelist_input)
    exp = _experiment(
        eta, run_seconds=args.run_seconds,
        sample_interval_s=args.sample_interval_s)
    grids = grids_from_projection_config(exp)
    timing = {}
    static_started = time.perf_counter()
    static = {}
    attrs = {}
    for grid_id, (path, shape) in enumerate((
            (args.geo_d01, (199, 199)),
            (args.geo_d02, (300, 300))), 1):
        static[grid_id], attrs[grid_id] = _read_static(
            path, expected_shape=shape)
    for index, grid in enumerate(grids):
        expected = (199, 199) if index == 0 else (300, 300)
        if grid.latlon_mass()[0].shape != expected:
            raise ValueError(f"d{index + 1:02d} grid geometry shape drift")
    timing["load_cached_static"] = time.perf_counter() - static_started

    io_before = _proc_io()
    total_started = time.perf_counter()
    final_forcing_hour = max(1, int(np.ceil(args.run_seconds / 3600.0)))
    pipeline_producer = None
    pipeline_report = None
    source_hash_receipt = None
    if args.pipeline_series is not None:
        if args.run_seconds > 43200.0:
            raise ValueError("the f00..f12 pipeline covers at most 12 hours")
        from tools.hrrr_pipeline import (
            HrrrPipelineProducer,
            verify_source_tree,
            write_pipeline_receipt,
        )
        source_hash_receipt = verify_source_tree(
            source_root=args.source_root,
            manifest=args.source_manifest,
            expected_manifest_sha256=args.source_manifest_sha256,
            series=args.pipeline_series,
            workers=13)
        timing["source_hash_preflight"] = source_hash_receipt["wall_seconds"]
        pipeline_producer = HrrrPipelineProducer(
            decoder=args.pipeline_decoder,
            series=args.pipeline_series,
            output=args.bridge,
            signals=args.pipeline_signals,
            cycle="2026-07-18 00:00:00",
            window=(781, 987, 315, 521),
            workers=args.pipeline_workers,
            log=args.pipeline_signals.with_suffix(".decoder.log"))
        started = time.perf_counter()
        pipeline_producer.start()
        pipeline_producer.wait_preflight()
        timing["decoder_inventory_preflight_wait"] = (
            time.perf_counter() - started)
        available_hours = tuple(range(13))

        def acquire_snapshot(hour):
            root = pipeline_producer.wait_hour(hour)
            return load_hrrr_pipeline_ready_window(root, hour)
    else:
        if not args.manifest_sha256:
            raise ValueError("--manifest-sha256 is required without pipeline mode")
        available_hours = tuple(range(final_forcing_hour + 1))
        started = time.perf_counter()
        snapshots = load_hrrr_native_series(
            args.bridge, available_hours,
            expected_manifest_sha256=args.manifest_sha256)
        timing["verify_and_map_native_series"] = time.perf_counter() - started

        def acquire_snapshot(hour):
            return snapshots[hour]

    setup_records = []
    mapping_reports = {"d01": {}, "d02": {}}
    root_result = None
    root_met = None
    previous_result = None
    previous_valid_time = None
    initial_snapshot = None
    last_valid_time = None
    root_intervals = []
    root_dc = exp.domains[0]
    try:
        for hour in available_hours:
            ready_wait_started = time.perf_counter()
            snapshot = acquire_snapshot(hour)
            ready_wait_seconds = time.perf_counter() - ready_wait_started
            report = {}
            result, met, horizontal_seconds, state_seconds = _initialize_state(
                snapshot, root_dc, grids[0], static[1], eta, report)
            mapping_reports["d01"][f"f{hour:02d}"] = report
            setup_records.append({
                "grid_id": 1,
                "forecast_hour": hour,
                "ready_wait_seconds": ready_wait_seconds,
                "horizontal_seconds": horizontal_seconds,
                "state_seconds": state_seconds,
                "completed_seconds_from_startup": (
                    time.perf_counter() - total_started),
            })
            last_valid_time = snapshot.valid_time
            if hour == 0:
                root_result = result
                root_met = met
                initial_snapshot = snapshot
            else:
                pair = build_state_lateral_boundaries(
                    [previous_result.state, result.state],
                    [previous_valid_time, snapshot.valid_time],
                    spec_bdy_width=5, spec_zone=1, relax_zone=4)
                interval = pair.intervals[0]
                root_intervals.append(BoundaryInterval(
                    float((hour - 1) * 3600), float(hour * 3600),
                    interval.fields))
                if previous_result is not root_result:
                    del previous_result
                del met
                del snapshot
                cp.get_default_memory_pool().free_all_blocks()
            previous_result = result
            previous_valid_time = last_valid_time
            _atomic_json(progress_path, {
                "status": "PREPARING",
                "last_root_forcing_hour": hour,
                "requested_run_seconds": args.run_seconds,
                "elapsed_setup_wall_seconds": time.perf_counter() - total_started,
            })
    except BaseException:
        if pipeline_producer is not None:
            pipeline_producer.cancel()
        raise
    if root_result is None or root_met is None:
        raise AssertionError("f00 root state was not retained")
    if ((last_valid_time - initial_snapshot.valid_time).total_seconds()
            < args.run_seconds):
        raise ValueError("native bridge does not cover the requested forecast")
    if previous_result is not root_result:
        del previous_result
    boundaries = LateralBoundaries(tuple(root_intervals), 5, 1, 4)
    started = time.perf_counter()
    attach_lateral_boundaries(root_result.state, boundaries)
    timing["attach_all_root_lbc"] = time.perf_counter() - started
    timing["all_root_lbc_bound_seconds_from_startup"] = (
        time.perf_counter() - total_started)

    seal_process = None
    seal_receipt = None
    seal_started = None
    if pipeline_producer is not None:
        pipeline_report = pipeline_producer.finish()
        controller_receipt = args.pipeline_signals / "controller.json"
        write_pipeline_receipt(controller_receipt, {
            "source_hash_preflight": source_hash_receipt,
            "producer": pipeline_report,
        })
        seal_receipt = args.pipeline_signals / "seal.json"
        seal_started = time.perf_counter()
        seal_process = subprocess.Popen([
            sys.executable, str(REPO / "tools" / "seal_hrrr_native_bridge.py"),
            "--root", str(args.bridge),
            "--receipt", str(seal_receipt),
            "--source-manifest-sha256", args.source_manifest_sha256,
            "--decoder", str(args.pipeline_decoder),
            "--series", str(args.pipeline_series),
            "--time-evidence", str(controller_receipt),
        ], cwd=REPO, text=True, stdout=subprocess.PIPE,
           stderr=subprocess.PIPE)

    child_report = {}
    child_result, child_met, horizontal_seconds, state_seconds = (
        _initialize_state(
            initial_snapshot, exp.domains[1], grids[1], static[2], eta,
            child_report))
    mapping_reports["d02"]["f00"] = child_report
    setup_records.append({
        "grid_id": 2,
        "forecast_hour": 0,
        "horizontal_seconds": horizontal_seconds,
        "state_seconds": state_seconds,
    })

    started = time.perf_counter()
    from gpuwm.runtime import declared_constant_glw
    constant_glw = declared_constant_glw(exp)
    root_driver = initialize_hrrr_physics(
        root_result, exp.domains[0].run, root_met,
        static[1], attrs[1], grids[0],
        initial_snapshot.valid_time, constant_glw_wm2=constant_glw)
    child_driver = initialize_hrrr_physics(
        child_result, exp.domains[1].run, child_met,
        static[2], attrs[2], grids[1],
        initial_snapshot.valid_time, constant_glw_wm2=constant_glw)
    timing["initialize_two_domain_physics"] = time.perf_counter() - started

    # The exact bridge inventory seal is independent of d02 initialization
    # and physics setup.  Run it underneath those GPU/CPU stages, but require
    # it to pass before the first forecast step can be published.
    if seal_process is not None:
        seal_stdout, seal_stderr = seal_process.communicate()
        timing["pipeline_bridge_seal_overlapped_wall"] = (
            time.perf_counter() - float(seal_started))
        if seal_process.returncode != 0:
            raise RuntimeError(
                "pipeline bridge seal failed: " + seal_stderr[-4000:])
        if seal_receipt is None or not seal_receipt.is_file():
            raise RuntimeError("pipeline bridge seal omitted its receipt")
        seal = json.loads(seal_receipt.read_text())
        args.manifest_sha256 = seal["manifest_sha256"]
        pipeline_report["seal"] = seal
        pipeline_report["seal_stdout"] = seal_stdout.strip()

    clock = resolve_clock(exp, lbc_interval_s=3600.0)
    schedule = build_schedule(exp, clock)
    clocks = clock.clocks()
    root_node = DomainNode(
        exp.domains[0], grids[0], root_result.state, clocks[1],
        None, [], None)
    child_node = DomainNode(
        exp.domains[1], grids[1], child_result.state, clocks[2],
        root_node, [], None)
    child_node.coupler = NestCoupler(child_node)
    root_node.children.append(child_node)
    child_node.state._nest_restart_classification = "REBUILT"
    model = ExperimentState(
        root_node, MappingProxyType({1: root_node, 2: child_node}),
        schedule, None, "native-hrrr-two-domain")
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None

    initial_health = {
        f"d{gid:02d}": _strict_json(vars(StateHealthValidator(
            model.node(gid).state).validate(phase=f"initialized.d{gid:02d}")))
        for gid in (1, 2)
    }
    if not all(value["ok"] for value in initial_health.values()):
        raise FloatingPointError(f"initial health failed: {initial_health}")

    # Measure only the first three solves per domain with explicit stream
    # synchronization.  The remaining forecast is not timing-perturbed.
    original_step = dycore_module.step
    state_to_grid = {id(root_node.state): 1, id(child_node.state): 2}
    step_samples = {1: [], 2: []}
    first_gpu_step_seconds_from_startup = None

    def timed_step(state, cfg, *positional, **keywords):
        nonlocal first_gpu_step_seconds_from_startup
        gid = state_to_grid[id(state)]
        if len(step_samples[gid]) >= 3:
            return original_step(state, cfg, *positional, **keywords)
        cp.cuda.Stream.null.synchronize()
        before = time.perf_counter()
        result = original_step(state, cfg, *positional, **keywords)
        cp.cuda.Stream.null.synchronize()
        step_samples[gid].append(time.perf_counter() - before)
        if first_gpu_step_seconds_from_startup is None:
            first_gpu_step_seconds_from_startup = (
                time.perf_counter() - total_started)
        return result

    dycore_module.step = timed_step
    history = {1: [], 2: []}
    gpu_peak_used = 0
    pool_peak_total = 0

    def record_memory():
        nonlocal gpu_peak_used, pool_peak_total
        free, total = cp.cuda.runtime.memGetInfo()
        gpu_peak_used = max(gpu_peak_used, int(total - free))
        pool_peak_total = max(
            pool_peak_total, int(cp.get_default_memory_pool().total_bytes()))

    def history_handler(_model, node, ticks):
        from gpuwm.core.refl import consume_refl_10cm

        gid = node.cfg.grid_id
        health = stability_report(
            node.state, node.cfg.run,
            boundary_width=node.cfg.run.spec_bdy_width)
        history[gid].append({
            "ticks": int(ticks),
            "elapsed_seconds": float(node.clock.elapsed_seconds),
            **health,
        })
        # Production history semantics: tick zero is emitted before the first
        # microphysics call, so there is no REFL_10CM stash to consume yet.
        if (ticks != 0 and node.state.qv is not None
                and node.state.physics.mp_physics in (1, 6, 10)):
            consume_refl_10cm(node.state)
        record_memory()

    def progress_callback(**event):
        if int(event["outer_step"]) == 1 or int(event["outer_step"]) % 20 == 0:
            record_memory()
            _atomic_json(progress_path, {
                "status": "RUNNING",
                "model_elapsed_seconds": event["model_elapsed_seconds"],
                "outer_step": event["outer_step"],
                "requested_run_seconds": args.run_seconds,
                "forecast_wall_seconds": time.perf_counter() - forecast_started,
                "gpu_peak_used_bytes_observed": gpu_peak_used,
            })

    forecast_started = time.perf_counter()
    try:
        execution = execute_experiment(
            model, history_handler=history_handler,
            progress_callback=progress_callback, validate_state=True,
            skip_feedback_path=True, pool_trim_per_period=True)
        cp.cuda.Stream.null.synchronize()
    finally:
        dycore_module.step = original_step
    timing["forecast_execution"] = time.perf_counter() - forecast_started
    record_memory()

    final_health = {}
    final_stability = {}
    for gid in (1, 2):
        node = model.node(gid)
        final_health[f"d{gid:02d}"] = _strict_json(vars(
            StateHealthValidator(node.state).validate(
                phase=f"final.d{gid:02d}")))
        final_stability[f"d{gid:02d}"] = stability_report(
            node.state, node.cfg.run,
            boundary_width=node.cfg.run.spec_bdy_width)
    if not all(value["ok"] for value in final_health.values()):
        raise FloatingPointError(f"final health failed: {final_health}")

    # Bind trajectory equality to the repository's complete canonical restart
    # inventory, including held tendencies, physics state, and model time.
    from gpuwm.verify.cases.real74_d02 import canonical_state_digest
    started = time.perf_counter()
    final_state_digests = {
        f"d{gid:02d}": canonical_state_digest(
            model.node(gid).state, model.node(gid).clock,
            scope="trajectory")
        for gid in (1, 2)
    }
    timing["canonical_final_state_digest"] = time.perf_counter() - started

    total_wall = time.perf_counter() - total_started
    io_after = _proc_io()
    source = _source_identity()
    source["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    report = {
        "schema": "gpuwm-native-hrrr-two-domain-forecast-v1",
        "status": "PASS",
        "scope": (
            "native HRRR d01 3-km external-LBC plus one-way nested d02 "
            "1-km WSM6/Dudhia/YSU/Noah forecast"),
        "run_seconds": float(args.run_seconds),
        # Decoded, not str()'d: the property is bytes, so str() writes the
        # repr -- b'NVIDIA GeForce RTX 5070 Ti' -- into the one field that
        # says which card produced these numbers.  Same trap, same fix, as
        # the single-domain runner's _device_name.
        "device": _device_name(cp),
        "timing_seconds": timing,
        "setup_records": setup_records,
        "total_wall_seconds": total_wall,
        "forecast_simulated_seconds_per_wall_second": (
            float(args.run_seconds) / timing["forecast_execution"]),
        "forecast_wall_seconds_per_simulated_hour": (
            timing["forecast_execution"] / (float(args.run_seconds) / 3600.0)),
        "measured_initial_step_seconds": {
            f"d{gid:02d}": values for gid, values in step_samples.items()},
        "downloaded_hrrr_to_first_gpu_step_seconds": (
            first_gpu_step_seconds_from_startup),
        "executor": {
            "steps": int(execution.steps),
            "forces": int(execution.forces),
            "feedback_calls": int(execution.feedback_calls),
        },
        "health": {
            "initial": initial_health,
            "final": final_health,
            "final_stability": final_stability,
            "history": {f"d{gid:02d}": values
                        for gid, values in history.items()},
        },
        "final_state_digests": final_state_digests,
        "physics": {
            "d01": _physics_receipt(root_driver, cp),
            "d02": _physics_receipt(child_driver, cp),
        },
        "memory": {
            "gpu_peak_used_bytes_observed": gpu_peak_used,
            "cupy_pool_peak_total_bytes_observed": pool_peak_total,
            "cpu_peak_rss_bytes": int(resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
        "input": {
            "bridge": str(args.bridge.resolve()),
            "bridge_manifest_sha256": args.manifest_sha256,
            "bridge_bytes": sum(path.stat().st_size
                                for path in args.bridge.rglob("*")
                                if path.is_file()),
            "forcing_hours": list(available_hours),
            "namelist_input_sha256": _sha256(args.namelist_input),
            "geo_d01_sha256": _sha256(args.geo_d01),
            "geo_d02_sha256": _sha256(args.geo_d02),
        },
        "source_identity": source,
        "pipeline": pipeline_report,
        "source_hash_preflight": source_hash_receipt,
        "mapping_reports": mapping_reports,
        "process_io_delta": {
            key: int(io_after.get(key, 0) - io_before.get(key, 0))
            for key in sorted(set(io_before) | set(io_after))},
    }
    _atomic_json(progress_path, {
        "status": "PASS",
        "model_elapsed_seconds": float(args.run_seconds),
        "forecast_wall_seconds": timing["forecast_execution"],
        "report": str((args.outdir / "report.json").resolve()),
    })
    return report


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--pipeline-series", type=Path)
    parser.add_argument("--pipeline-decoder", type=Path)
    parser.add_argument("--pipeline-signals", type=Path)
    parser.add_argument("--pipeline-workers", default="8")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--namelist-input", type=Path, required=True)
    parser.add_argument("--geo-d01", type=Path, required=True)
    parser.add_argument("--geo-d02", type=Path, required=True)
    parser.add_argument("--run-seconds", type=float, required=True)
    parser.add_argument("--sample-interval-s", type=float, default=300.0)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.pipeline_series is not None:
        required = {
            "pipeline_decoder": args.pipeline_decoder,
            "pipeline_signals": args.pipeline_signals,
            "source_root": args.source_root,
            "source_manifest": args.source_manifest,
            "source_manifest_sha256": args.source_manifest_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"pipeline mode is missing: {missing}")
        if args.manifest_sha256 is not None:
            parser.error(
                "--manifest-sha256 is produced by pipeline mode; do not pass it")
    elif args.manifest_sha256 is None:
        parser.error("--manifest-sha256 is required without --pipeline-series")
    return args


def main():
    args = _parse_args()
    try:
        report = run(args)
    except BaseException as error:
        args.outdir.mkdir(parents=True, exist_ok=True)
        failed = {
            "schema": "gpuwm-native-hrrr-two-domain-forecast-v1",
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(args.outdir / "report.json", failed)
        raise
    _atomic_json(args.outdir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "run_seconds": report["run_seconds"],
        "forecast_execution_seconds": report["timing_seconds"][
            "forecast_execution"],
        "total_wall_seconds": report["total_wall_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
