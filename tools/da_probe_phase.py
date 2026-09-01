"""Probe: is a DA member leg GPU-bound or host-bound?

Threads and streams can only hide DEVICE time.  If ``execute_experiment``
is really the Python interpreter issuing ~100 kernels a step, concurrent
members contend on the GIL and win nothing.  This measures the split
three ways on one wired model:

* wall time for the leg;
* device time, from CUDA events bracketing the whole leg on the null
  stream (event-to-event elapsed covers device-timeline occupancy plus
  any gaps where the device sat idle waiting for the host);
* host time inside the interpreter, from ``cProfile``'s total.

A leg whose device time is close to its wall time is GPU-bound and worth
overlapping.  A leg whose profiler total is close to its wall time, with
the device idle for most of it, is host-bound and streams are the wrong
tool.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path


def main() -> int:
    import cupy as cp

    sys.argv = [sys.argv[0]] + sys.argv[1:]
    # Reuse the driver's own preflight + wiring by importing it and
    # calling the pieces, rather than re-deriving a second wiring that
    # could drift from the one being measured.
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus, execute_experiment)
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.ingest.prepared_cache import restore_prepared_cache
    from gpuwm.prepared_single_domain_forecast import (
        preflight_prepared_forecast)
    import dataclasses
    from types import MappingProxyType, SimpleNamespace

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--proof-sha256", required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--prepared-content-sha256", required=True)
    p.add_argument("--physics-profile", required=True)
    p.add_argument("--run-seconds", type=float, required=True)
    p.add_argument("--history-interval-seconds", type=float, required=True)
    p.add_argument("--leg-seconds", type=float, default=900.0)
    p.add_argument("--repeats", type=int, default=3)
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
    print(f"grid {cfg.nx}x{cfg.ny}x{cfg.nz} dt={cfg.dt} "
          f"steps/leg={a.leg_seconds / float(cfg.dt):.0f}", flush=True)

    def wire(run_seconds_total):
        exp_leg = dataclasses.replace(exp, run_seconds=float(run_seconds_total))
        restored = restore_prepared_cache(
            inputs.prepared_cache_path, expected_identity=inputs.cache_identity,
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

    for trial in range(a.repeats):
        model, node, restored, driver = wire(a.leg_seconds)
        cp.cuda.Stream.null.synchronize()
        start_ev = cp.cuda.Event()
        end_ev = cp.cuda.Event()

        profiler = cProfile.Profile()
        t0 = time.perf_counter()
        start_ev.record()
        profiler.enable()
        execute_experiment(model, history_handler=None,
                           progress_callback=None, validate_state=True,
                           skip_feedback_path=True,
                           pool_trim_per_period=True)
        profiler.disable()
        end_ev.record()
        end_ev.synchronize()
        wall = time.perf_counter() - t0
        device_ms = cp.cuda.get_elapsed_time(start_ev, end_ev)

        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("tottime")
        stats.print_stats(12)
        total_host = stats.total_tt

        print(f"\n--- trial {trial}: wall {wall:.3f} s | "
              f"event-span {device_ms / 1000.0:.3f} s | "
              f"profiler-total {total_host:.3f} s", flush=True)
        if trial == a.repeats - 1:
            print(buf.getvalue()[:3000], flush=True)
        del model, node, restored, driver
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
