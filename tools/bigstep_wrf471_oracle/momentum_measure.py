"""Record every native output word and exact per-field parity measurements."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.verify.bigstep_momentum_oracle import (MOMENTUM_CASES, MOMENTUM_ROUTINES,
    load_momentum_fixture, momentum_port_outputs, measure_momentum_parity,
    pressure_launch_arithmetic_outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import cupy as cp
    from gpuwm.core.kernels import module_source
    from gpuwm.certify.compile_platform import compile_platform_fingerprint
    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode("ascii")
    device_receipt = {"cupy": cp.__version__, "device": device_name,
                      "runtime": cp.cuda.runtime.runtimeGetVersion(),
                      "driver": cp.cuda.runtime.driverGetVersion(),
                      "compile_platform": compile_platform_fingerprint(),
                      "kernel_source_sha256": {name: hashlib.sha256(module_source(name).encode("utf-8")).hexdigest() for name in ("dycore", "coriolis_map")},
                      "launch_paths": ["gpuwm.core.dycore._launch_slow_pgf", "gpuwm.core.dycore.launch_coriolis_curvature"]}
    table = {}
    diagnostic = {}
    for case in MOMENTUM_CASES:
        fixture = load_momentum_fixture(case, args.fixtures)
        table[case] = {}
        diagnostic[case] = {}
        for routine in MOMENTUM_ROUTINES:
            got = momentum_port_outputs(fixture, routine)
            table[case][routine] = measure_momentum_parity(fixture.reference[routine], got)
            np.savez_compressed(args.output.with_name(f"momentum-gpu-{case}-{routine}.npz"), **got)
            if routine == "horizontal_pressure_gradient":
                diagnostic[case]["explicit_fp32_operation_tree"] = measure_momentum_parity(pressure_launch_arithmetic_outputs(fixture), got)
                diagnostic[case]["original_wrf_pressure"] = measure_momentum_parity(fixture.reference["original_pressure"], got)
    # Attribution control only: preserve the source and launch argument path
    # while turning off contraction in an isolated diagnostic module.
    import gpuwm.core.dycore as dycore
    original_get_kernel = dycore.get_kernel
    no_fma = cp.RawModule(code=module_source("coriolis_map"),
                          options=("-std=c++17", "--fmad=false"))
    try:
        dycore.get_kernel = lambda module, name: (no_fma.get_function(name) if module == "coriolis_map" else original_get_kernel(module, name))
        for case in MOMENTUM_CASES:
            fixture = load_momentum_fixture(case, args.fixtures)
            for routine in ("coriolis", "curvature", "combined"):
                got = momentum_port_outputs(fixture, routine)
                diagnostic[case][routine + "_without_fma"] = measure_momentum_parity(fixture.reference[routine], got)
    finally:
        dycore.get_kernel = original_get_kernel
    ordered = module_source("coriolis_map")
    start = ordered.index("real rv4 =")
    end = ordered.index(";", start) + 1
    ordered = ordered[:start] + """real rv4 = 0.25f * (rv[I3S(k, j + 1, imc, nyf, nx)]
                          + rv[I3S(k, j + 1, ic,  nyf, nx)]
                          + rv[I3S(k, j,     imc, nyf, nx)]
                          + rv[I3S(k, j,     ic,  nyf, nx)]);""" + ordered[end:]
    old_rw = """real rw4 = 0.25f * (rw_at(w, mut, msft, c1f, c2f, k,     cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k,     cA, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cA, st));"""
    new_rw = """real rw4 = 0.25f * (rw_at(w, mut, msft, c1f, c2f, k + 1, cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k,     cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cA, st)
                          + rw_at(w, mut, msft, c1f, c2f, k,     cA, st));"""
    assert ordered.count(old_rw) == 2
    ordered = ordered.replace(old_rw, new_rw)
    ordered_module = cp.RawModule(code=ordered, options=("-std=c++17", "--fmad=false"))
    try:
        dycore.get_kernel = lambda module, name: (ordered_module.get_function(name) if module == "coriolis_map" else original_get_kernel(module, name))
        for case in MOMENTUM_CASES:
            fixture = load_momentum_fixture(case, args.fixtures)
            for routine in ("coriolis", "curvature"):
                got = momentum_port_outputs(fixture, routine)
                diagnostic[case][routine + "_wrf_sum_order_without_fma"] = measure_momentum_parity(fixture.reference[routine], got)
    finally:
        dycore.get_kernel = original_get_kernel
    args.output.write_text(json.dumps(table, indent=2) + "\n", encoding="ascii")
    args.output.with_name("momentum-diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="ascii")
    args.output.with_name("momentum-device-receipt.json").write_text(json.dumps(device_receipt, indent=2) + "\n", encoding="ascii")
    print(json.dumps(table, indent=2))
