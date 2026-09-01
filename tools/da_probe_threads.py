"""Spike: do N member legs in N threads actually overlap?

The phase probe showed a leg is launch-overhead-bound, not kernel-bound:
the interpreter is busy for ~100% of the wall clock issuing many small
CuPy operations while the device idles.  Whether threads help therefore
turns entirely on how much of that dispatch releases the GIL, which is
not knowable from source reading.  So measure it: wire W independent
models, run them in W threads on W non-blocking streams, and compare
against the same W legs run one after another in this thread.

Nothing here writes a receipt or claims a result -- it is the experiment
that decides whether the concurrent driver is worth building.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import threading
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace


def main() -> int:
    import cupy as cp

    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus, execute_experiment)
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.ingest.prepared_cache import restore_prepared_cache
    from gpuwm.prepared_single_domain_forecast import (
        preflight_prepared_forecast)

    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--proof-sha256", required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--prepared-content-sha256", required=True)
    p.add_argument("--physics-profile", required=True)
    p.add_argument("--run-seconds", type=float, required=True)
    p.add_argument("--history-interval-seconds", type=float, required=True)
    p.add_argument("--leg-seconds", type=float, default=900.0)
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--pool-trim", action="store_true",
                   help="leave the per-step pool trim ON (the serial "
                        "driver's setting); default is OFF")
    a = p.parse_args()

    authority = a.prepared_root.parent / "authority"
    inputs = preflight_prepared_forecast(
        source="gfs", prepared_root=a.prepared_root,
        proof_sha256=a.proof_sha256,
        source_manifest_sha256=a.source_manifest_sha256,
        prepared_content_sha256=a.prepared_content_sha256,
        experiment_config=authority / "experiment.toml",
        wps_namelist=authority / "namelist.wps",
        physics_profile=a.physics_profile,
        run_seconds=a.run_seconds,
        history_interval_seconds=a.history_interval_seconds)
    exp = inputs.experiment
    cfg = exp.root.run
    print(f"grid {cfg.nx}x{cfg.ny}x{cfg.nz} dt={cfg.dt} width={a.width} "
          f"pool_trim={a.pool_trim}", flush=True)

    def wire(run_seconds_total):
        exp_leg = dataclasses.replace(exp,
                                      run_seconds=float(run_seconds_total))
        restored = restore_prepared_cache(
            inputs.prepared_cache_path,
            expected_identity=inputs.cache_identity,
            cfg=cfg, static=inputs.static)
        driver = initialize_prepared_physics(
            restored.initial_result, cfg, restored.met, restored.surface,
            inputs.static, inputs.landuse_identity, inputs.grid,
            exp.start_time)
        tick = resolve_clock(
            exp_leg, lbc_interval_s=float(inputs.boundary_interval_seconds))
        schedule = build_schedule(exp_leg, tick)
        clocks = tick.clocks()
        node = DomainNode(exp.root, inputs.grid,
                          restored.initial_result.state, clocks[1],
                          None, [], None)
        model = ExperimentState(node, MappingProxyType({1: node}), schedule,
                                None, "probe")
        model._runtime_status = ModelRuntimeStatus()
        model._resumed = False
        model._resume_committed_history_grid_ids = frozenset()
        model._scratch_arena = None
        model._dycore_state_workspace = None
        model._io_manager = None
        model._last_checkpoint = None
        model._prepared_by_grid_id = MappingProxyType({
            1: SimpleNamespace(static_fields=inputs.static,
                               geog_selection=None,
                               initial_result=restored.initial_result)})
        return model, node, restored, driver

    def run_leg(model):
        execute_experiment(model, history_handler=None,
                           progress_callback=None, validate_state=True,
                           skip_feedback_path=True,
                           pool_trim_per_period=a.pool_trim)

    # Warm every kernel module and lazily-built device object on ONE
    # thread first, so neither arm pays a compile and the comparison is
    # of steady-state stepping.
    warm = wire(a.leg_seconds)
    run_leg(warm[0])
    cp.cuda.Stream.null.synchronize()
    del warm
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    # ---- serial arm --------------------------------------------------
    built = [wire(a.leg_seconds) for _ in range(a.width)]
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for model, *_ in built:
        run_leg(model)
    cp.cuda.Stream.null.synchronize()
    serial = time.perf_counter() - t0
    del built
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    # ---- concurrent arm ----------------------------------------------
    built = [wire(a.leg_seconds) for _ in range(a.width)]
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(a.width)]
    cp.cuda.Stream.null.synchronize()
    errors: list = []

    def worker(index):
        try:
            with streams[index]:
                run_leg(built[index][0])
                streams[index].synchronize()
        except BaseException as error:  # noqa: BLE001
            errors.append((index, error))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(a.width)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    concurrent = time.perf_counter() - t0
    for index, error in errors:
        print(f"  worker {index} FAILED: {type(error).__name__}: {error}",
              flush=True)

    print(f"\nserial     {a.width} legs: {serial:.2f} s "
          f"({serial / a.width:.2f} s/leg)")
    print(f"concurrent {a.width} legs: {concurrent:.2f} s "
          f"({concurrent / a.width:.2f} s/leg)")
    print(f"speedup: {serial / concurrent:.2f}x", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
