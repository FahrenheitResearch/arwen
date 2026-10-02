#!/usr/bin/env python3
"""Generate native references or record every CUDA output word.

Receipts are measurements.  Repeated runs and tests require exact equality
of the recorded words and statistics, including an improvement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.verify.advect_oracle import (
    ADVECT_ORACLE_DIR, ROUTINES, SUPPORTED_ROUTINES, load_advect_cases,
    generate_reference, advect_port_outputs, arithmetic_control,
    measure_advect_parity, measure_words, defined_output_mask, sha256)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ADVECT_ORACLE_DIR)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--mutation", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--words-directory", type=Path)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    cases = load_advect_cases(args.directory)
    if args.case:
        cases = tuple(case for case in cases if case.name in args.case)
    if not cases:
        parser.error("no fixture cases selected")
    if args.executable:
        if args.scratch is None:
            parser.error("--executable requires --scratch")
        for case in cases:
            reference = generate_reference(case, args.executable.resolve(), args.scratch)
            path = args.directory / f"{case.name}-wrf.npz"
            np.savez_compressed(path, **reference)
            print(f"WRF {case.name}: {len(reference)} output arrays, {sha256(path)}", flush=True)
        cases = tuple(case for case in load_advect_cases(args.directory)
                      if not args.case or case.name in args.case)
    if not args.gpu:
        return
    if args.receipt is None or args.words_directory is None:
        parser.error("--gpu requires --receipt and --words-directory")
    try:
        words_directory = args.words_directory.resolve().relative_to(args.directory.resolve())
    except ValueError:
        parser.error("--words-directory must be inside --directory so receipt words can be replayed")
    import cupy as cp
    from gpuwm.core.kernels import module_source
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    device_name = device["name"].decode() if isinstance(device["name"], bytes) else device["name"]
    receipt = {
        "schema_version": 1,
        "gpu": device_name, "compute_capability": [device["major"], device["minor"]],
        "cupy": cp.__version__, "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "kernels": {name: hashlib.sha256(module_source(name).encode("utf-8")).hexdigest()
                    for name in ("advection", "pd_advection", "openbc")},
        "fixture_manifest_sha256": sha256(args.directory / "cases.json"),
        "words_directory": words_directory.as_posix(),
        "cases": {}, "controls": {},
    }
    args.words_directory.mkdir(parents=True, exist_ok=True)
    variants = ["production"]
    if args.controls:
        variants += ["no_fma", "wrf_flux", "wrf_flux_no_fma"]
    if args.mutation:
        variants += ["mutation"]
    for variant in variants:
        rows = {}
        with arithmetic_control(variant):
            for case in cases:
                output = advect_port_outputs(case, variant=variant)
                measured = measure_advect_parity(case, output)
                # Monotonic is unimplemented.  This is a distinction
                # measurement against the actual supported PD path.
                measured["mono_against_pd"] = measure_words(
                    output["advect_scalar_pd"], case.reference["advect_scalar_mono"])
                words_path = args.words_directory / f"{case.name}-{variant}.npz"
                np.savez_compressed(words_path, **output)
                rows[case.name] = {"measurements": measured, "words_file": words_path.name,
                                   "words_sha256": sha256(words_path)}
                rows[case.name]["levels"] = {
                    key: [measure_words(values[k], case.reference[key][k],
                                        defined=defined_output_mask(case, key)[k])
                          for k in range(values.shape[0])]
                    for key, values in output.items()}
                maxima = {name: row["max_ulp"] for name, row in measured.items()}
                print(f"CUDA {variant} {case.name}: {maxima}", flush=True)
        if variant == "production":
            receipt["cases"] = rows
        else:
            receipt["controls"][variant] = rows
        _write_json(args.receipt, receipt)
    if args.mutation:
        receipt["mutation_rejected"] = any(
            receipt["cases"][case.name]["measurements"][routine]["got_sha256"]
            != receipt["controls"]["mutation"][case.name]["measurements"][routine]["got_sha256"]
            for case in cases for routine in SUPPORTED_ROUTINES)
        if not receipt["mutation_rejected"]:
            raise AssertionError("known flux5 mutation did not change a compared output")
        _write_json(args.receipt, receipt)


if __name__ == "__main__":
    main()
