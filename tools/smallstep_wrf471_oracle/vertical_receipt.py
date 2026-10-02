"""Write exact-word measurements for the compiled-WRF vertical routines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpuwm.verify.smallstep_vertical_oracle import measure_vertical_case, vertical_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-fma", action="store_true")
    parser.add_argument("--theta-offset", type=float, default=300.0)
    parser.add_argument("--arrays", type=Path)
    parser.add_argument("--wrf-phi-order", action="store_true")
    parser.add_argument("--preserve-subnormals", action="store_true")
    parser.add_argument("--wrf-top-order", action="store_true")
    parser.add_argument("--wrf-update-order", action="store_true")
    parser.add_argument("--correct-pi", action="store_true")
    parser.add_argument("--wrf-eos-order", action="store_true")
    args = parser.parse_args()
    import cupy as cp
    from gpuwm.core.kernels import module_source
    cases = {}
    arrays = {}
    for name, raw, metadata in vertical_cases():
        metadata = dict(metadata, theta_offset=args.theta_offset)
        case_arrays = {} if args.arrays is not None else None
        cases[name] = measure_vertical_case(raw, metadata, args.library, no_fma=args.no_fma,
                                           output_arrays=case_arrays, wrf_phi_order=args.wrf_phi_order,
                                           preserve_subnormals=args.preserve_subnormals,
                                           wrf_top_order=args.wrf_top_order,
                                           wrf_update_order=args.wrf_update_order,
                                           correct_pi=args.correct_pi, wrf_eos_order=args.wrf_eos_order)
        if case_arrays is not None:
            arrays.update({name + "__" + key: value for key, value in case_arrays.items()})
        print(name, json.dumps(cases[name], sort_keys=True), flush=True)
    result = dict(format_version=1, library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  kernel_source_sha256=hashlib.sha256(module_source("acoustic").encode()).hexdigest(),
                  no_fma=args.no_fma, theta_offset=args.theta_offset,
                  wrf_phi_order=args.wrf_phi_order,
                  preserve_subnormals=args.preserve_subnormals,
                  wrf_top_order=args.wrf_top_order,
                  wrf_update_order=args.wrf_update_order,
                  correct_pi=args.correct_pi,
                  wrf_eos_order=args.wrf_eos_order,
                  cases=cases)
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    device_name = device["name"]
    result["environment"] = dict(gpu=device_name.decode() if isinstance(device_name, bytes) else str(device_name),
                                  compute_capability=[int(device["major"]), int(device["minor"])],
                                  cupy=cp.__version__, runtime_version=cp.cuda.runtime.runtimeGetVersion())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.arrays is not None:
        np.savez_compressed(args.arrays, **arrays)
        result["arrays_sha256"] = hashlib.sha256(args.arrays.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
