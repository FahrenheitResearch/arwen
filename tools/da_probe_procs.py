"""Spike: do N member legs in N PROCESSES overlap, and what do they cost?

Threads measured 0.97x at width 2 and 0.83x at width 4 -- the leg is
host-dispatch-bound and the GIL serializes it completely.  Processes have
no GIL, so they are the only way this workload overlaps.  The question
they raise instead is device memory: each process pays its own CUDA
context and its own local-memory backing store, and that cost is what
bounds the width.

This measures both, for real, rather than trusting the estimator:

* wall time for W legs run as W concurrent processes, against W legs run
  one after another inside a single process;
* device memory actually used at each width, sampled from nvidia-smi
  while the workers are mid-leg.

``--worker`` is the child entry point; the parent runs without it.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace


def build_parser():
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
    p.add_argument("--legs", type=int, default=1,
                   help="legs each worker runs, to amortize startup")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--serial-in-process", action="store_true")
    return p


def make_wire(a):
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus)
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.ingest.prepared_cache import restore_prepared_cache
    from gpuwm.prepared_single_domain_forecast import (
        preflight_prepared_forecast)

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

    return wire, cfg


def worker_main(a) -> int:
    import cupy as cp
    from gpuwm.core.model import execute_experiment

    wire, _cfg = make_wire(a)
    t_ready = time.perf_counter()
    for _ in range(a.legs):
        model, node, restored, driver = wire(a.leg_seconds)
        execute_experiment(model, history_handler=None,
                           progress_callback=None, validate_state=True,
                           skip_feedback_path=True,
                           pool_trim_per_period=False)
        cp.cuda.Stream.null.synchronize()
        del model, node, restored, driver
        gc.collect()
    used = cp.get_default_memory_pool().used_bytes()
    total = cp.get_default_memory_pool().total_bytes()
    print(json.dumps({"ready_to_done_s": time.perf_counter() - t_ready,
                      "pool_used_mib": used / 1048576.0,
                      "pool_total_mib": total / 1048576.0}), flush=True)
    return 0


def gpu_used_mib() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip().splitlines()[0])


def main() -> int:
    a = build_parser().parse_args()
    if a.worker:
        return worker_main(a)

    child_args = [
        sys.executable, __file__, "--worker",
        "--prepared-root", str(a.prepared_root),
        "--proof-sha256", a.proof_sha256,
        "--source-manifest-sha256", a.source_manifest_sha256,
        "--prepared-content-sha256", a.prepared_content_sha256,
        "--physics-profile", a.physics_profile,
        "--run-seconds", str(a.run_seconds),
        "--history-interval-seconds", str(a.history_interval_seconds),
        "--leg-seconds", str(a.leg_seconds),
        "--legs", str(a.legs),
    ]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    baseline = gpu_used_mib()
    peak = [baseline]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                peak.append(gpu_used_mib())
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)

    sample_thread = threading.Thread(target=sampler, daemon=True)
    sample_thread.start()

    t0 = time.perf_counter()
    procs = [subprocess.Popen(child_args, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
             for _ in range(a.width)]
    outs = [proc.communicate() for proc in procs]
    wall = time.perf_counter() - t0
    stop.set()
    sample_thread.join(timeout=2.0)

    failures = [(i, proc.returncode, outs[i][1][-800:])
                for i, proc in enumerate(procs) if proc.returncode != 0]
    for index, code, err in failures:
        print(f"  worker {index} exit {code}:\n{err}", flush=True)

    inner = []
    for stdout, _ in outs:
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    inner.append(json.loads(line))
                except ValueError:
                    pass

    total_legs = a.width * a.legs
    print(f"\nwidth {a.width} x {a.legs} leg(s) = {total_legs} legs")
    print(f"  wall (incl. process start + preflight): {wall:.2f} s")
    if inner:
        span = max(d["ready_to_done_s"] for d in inner)
        print(f"  slowest worker leg-phase only: {span:.2f} s "
              f"({span / a.legs:.2f} s/leg)")
        print(f"  worker pool_total_mib: "
              f"{[round(d['pool_total_mib']) for d in inner]}")
    print(f"  GPU baseline {baseline:.0f} MiB, peak {max(peak):.0f} MiB, "
          f"delta {max(peak) - baseline:.0f} MiB")
    print(f"  per-process device cost: "
          f"{(max(peak) - baseline) / a.width:.0f} MiB", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
