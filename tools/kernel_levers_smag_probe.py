"""Compare exact staging scalar kernels with guarded uint32 addressing.

Run only on an authorized GPU under its OWNER mutex.  The baseline argument
is smag2d.cu exported from the exact staging SHA with git show.  All inputs
are synthetic, and this probe cannot certify a complete forecast.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np


def compile_pair(baseline_path):
    import cupy as cp
    from gpuwm.core import kernels as kernel_module
    from gpuwm.core.kernels import module_source, module_source_int_defines

    ordinary = module_source("smag2d")
    marker = "// gpuwm/core/kernels/smag2d.cu"
    prefix = ordinary[:ordinary.index(marker)]
    baseline_unit = Path(baseline_path).read_text(encoding="utf-8")
    baseline = prefix + baseline_unit
    candidate = module_source_int_defines(
        "smag2d", (("GPUWM_SMAG_INDEX32", 1),))
    modules = {label: cp.RawModule(code=source, options=("-std=c++17",))
               for label, source in (("baseline", baseline),
                                     ("candidate", candidate))}
    for module in modules.values():
        module.compile()
    kernels = {
        label: {name: module.get_function("wrf_smag_" + name + "_s")
                for name in ("flux", "hd")}
        for label, module in modules.items()}
    source_receipt = {
        "baseline_path": str(Path(baseline_path).resolve()),
        "candidate_kernel_module_path": str(Path(kernel_module.__file__).resolve()),
        "baseline_unit_sha256": hashlib.sha256(baseline_unit.encode()).hexdigest(),
        "baseline_compiled_source_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
        "candidate_compiled_source_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "options": ["-std=c++17"],
        "candidate_defines": {"GPUWM_SMAG_INDEX32": 1},
        "kernel_attributes": {
            label: {name: kernel.attributes for name, kernel in pair.items()}
            for label, pair in kernels.items()},
    }
    return kernels, source_receipt


def make_case(shape, *, boundary_x=True, boundary_y=True, moist=True,
              terrain=True, full_theta=False, time_t=False, dx=3000.0):
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.dycore import _TPB, _save_time_t, _wrf_smag_grid_args
    from gpuwm.verify.npref import random_acoustic_state

    nz, ny, nx = shape
    state, config = random_acoustic_state(
        seed=734, nx=nx, ny=ny, nz=nz, stretch=1.4,
        hybrid_opt=2 if terrain else 0,
        hill_height=300.0 if terrain else 0.0, msf_amp=0.09, moist=moist)
    config = replace(config, km_opt=4, bl_pbl_physics=1, dx=dx, dy=dx,
                     open_x=boundary_x, open_y=boundary_y)
    random = np.random.default_rng(908)
    if moist:
        state.qv[...] = cp.asarray(random.uniform(0.001, 0.025, shape), cp.float32)
        update_diagnostics(state, config.hypsometric_opt)
    _save_time_t(state)
    # Saved and live inputs differ so a wrong time carrier changes words.
    state.u[...] *= cp.float32(1.5)
    state.v[...] *= cp.float32(0.75)
    state.w[...] *= cp.float32(1.25)
    state.php[...] += cp.float32(0.03125)
    if moist:
        state.qv[...] *= cp.float32(0.8)
    field = cp.asarray(random.normal(0.0, 0.5, shape), cp.float32)
    coefficient = cp.asarray(random.uniform(0.0, 700.0, shape), cp.float32)
    coefficient[:, :, ::5] = cp.float32(0.0)
    common = _wrf_smag_grid_args(state, config, time_t=time_t)
    tail = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(boundary_x), np.int32(boundary_y)]
    initial = cp.asarray(random.normal(0.0, 0.2, shape), cp.float32)
    arrays = {
        label: {"fx": cp.full((nz, ny, nx + 1), cp.nan, cp.float32),
                "fy": cp.full((nz, ny + 1, nx), cp.nan, cp.float32),
                "tend": initial.copy()}
        for label in ("baseline", "candidate")}
    arguments = {}
    for label, outputs in arrays.items():
        flux_args = tuple(common + [field, coefficient, state.thb,
                                   np.int32(full_theta), np.int32(state.thb.ndim == 3),
                                   outputs["fx"], outputs["fy"]] + tail)
        hd_args = tuple(common + [outputs["fx"], outputs["fy"], outputs["tend"]] + tail)
        arguments[label] = {"flux": (((nx + 1 + _TPB - 1) // _TPB, ny + 1, nz),
                                     (_TPB, 1, 1), flux_args),
                            "hd": (((nx + _TPB - 1) // _TPB, ny, nz),
                                   (_TPB, 1, 1), hd_args)}
    return state, arrays, arguments


def run_pair(kernels, arguments, label):
    for name in ("flux", "hd"):
        kernels[label][name](*arguments[label][name])


def check_words(kernels, arrays, arguments):
    import cupy as cp
    for label in ("baseline", "candidate"):
        run_pair(kernels, arguments, label)
    cp.cuda.get_current_stream().synchronize()
    fields = {}
    for name in ("fx", "fy", "tend"):
        before = cp.asnumpy(arrays["baseline"][name])
        after = cp.asnumpy(arrays["candidate"][name])
        mismatch = int(np.count_nonzero(before.view(np.uint32) != after.view(np.uint32)))
        fields[name] = {"shape": list(before.shape), "mismatch_words": mismatch,
                        "baseline_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
                        "candidate_sha256": hashlib.sha256(after.tobytes()).hexdigest(),
                        "nonfinite_count": int(np.count_nonzero(~np.isfinite(before)))}
    return fields


def timed_call(call, repeats):
    import cupy as cp
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end)) / repeats


def benchmark(kernels, arguments, rounds, repeats):
    timings = {name: {label: [] for label in ("baseline", "candidate")}
               for name in ("flux", "hd", "pair")}
    for label in ("baseline", "candidate"):
        for _ in range(5):
            run_pair(kernels, arguments, label)
    for trial in range(rounds):
        labels = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
        for name in ("flux", "hd", "pair"):
            for label in labels:
                if name == "pair":
                    call = lambda: run_pair(kernels, arguments, label)
                else:
                    call = lambda: kernels[label][name](*arguments[label][name])
                timings[name][label].append(timed_call(call, repeats))
    result = {}
    for name, samples in timings.items():
        base = statistics.median(samples["baseline"])
        candidate = statistics.median(samples["candidate"])
        paired_ratios = [old / new for old, new in
                         zip(samples["baseline"], samples["candidate"])]
        result[name] = {"baseline_ms": base, "candidate_ms": candidate,
                        "speedup": base / candidate,
                        "paired_speedup_min": min(paired_ratios),
                        "paired_speedup_max": max(paired_ratios),
                        "samples_ms": samples}
    return result


def identity_matrix(kernels):
    import cupy as cp
    receipts = []
    for boundary_x, boundary_y, moist, terrain, full_theta, time_t in itertools.product(
            (False, True), repeat=6):
        config = {"boundary_x": boundary_x, "boundary_y": boundary_y,
                  "moist": moist, "terrain": terrain,
                  "full_theta": full_theta, "time_t": time_t}
        state, arrays, arguments = make_case((6, 7, 129), **config)
        fields = check_words(kernels, arrays, arguments)
        receipts.append({"config": config, "fields": fields})
        del state, arrays, arguments
        cp.get_default_memory_pool().free_all_blocks()
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--identity-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"):
        raise SystemExit("GPUWM_NO_LOCAL_GPU forbids this probe on this host")
    import cupy as cp
    from gpuwm.core.smag2d import scalar_index32_fits

    started = time.time()
    kernels, source = compile_pair(args.baseline_source)
    prop = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = prop["name"].decode() if isinstance(prop["name"], bytes) else str(prop["name"])
    report = {"baseline_sha": args.baseline_sha, "source": source,
              "gpu": {"name": name, "compute_capability": [prop["major"], prop["minor"]],
                      "cupy_version": cp.__version__,
                      "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
                      "driver_version": cp.cuda.runtime.driverGetVersion()},
              "identity_matrix": identity_matrix(kernels), "benchmarks": []}
    if not args.identity_only:
        for shape, dx, label in (((50, 600, 600), 3000.0, "c3-geometry"),
                                 ((50, 400, 400), 1000.0, "c1-geometry")):
            if not scalar_index32_fits(*shape):
                raise ValueError(f"unsafe index32 benchmark shape {shape}")
            for full_theta in (False, True):
                state, arrays, arguments = make_case(shape, dx=dx, full_theta=full_theta)
                fields = check_words(kernels, arrays, arguments)
                result = {"case": label, "shape_nz_ny_nx": list(shape),
                          "config": {"dx": dx, "boundary_x": True, "boundary_y": True,
                                     "moist": True, "terrain": True,
                                     "full_theta": full_theta, "time_t": False},
                          "fields": fields, "rounds": args.rounds, "repeats": args.repeats,
                          "timings": benchmark(kernels, arguments, args.rounds, args.repeats)}
                report["benchmarks"].append(result)
                print(json.dumps({"case": label, "full_theta": full_theta,
                                  "timings": result["timings"]}), flush=True)
                del state, arrays, arguments
                cp.get_default_memory_pool().free_all_blocks()
    rows = report["identity_matrix"] + report["benchmarks"]
    report["mismatch_words"] = sum(field["mismatch_words"] for row in rows
                                    for field in row["fields"].values())
    report["wall_seconds"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "mismatch_words": report["mismatch_words"],
                      "wall_seconds": report["wall_seconds"]}), flush=True)
    return 0 if report["mismatch_words"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
