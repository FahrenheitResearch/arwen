"""A/B the observation-sparse device LETKF against another analysis route.

Builds a real-shape synthetic analysis with the profiling generator from
``tools.da_analysis_bench`` (at the real case’s shape:
241 x 241 x 49, 32 members, 14 fields, windowed radial-velocity batches,
whole-grid reflectivity and clear air, surface batches) and solves it with
``cuda-obs-sparse`` twice (run-to-run bytes) and with one reference route,
then reports wall seconds per route and, per field, the max and RMS of the
increment difference against the reference beside the increment RMS
itself.  Measurement tooling only; nothing in a cycle calls this.

    python -m tools.da_letkf_device_ab --reference cuda-resident --out ab.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


try:
    from tools.da_analysis_bench import FIELDS, build_case
except ImportError:                     # python tools/da_letkf_device_ab.py
    from da_analysis_bench import FIELDS, build_case


def _solve(route, prior, batches, grid, config):
    import cupy as cp

    from gpuwm.da import radar_assimilation as ra
    from gpuwm.da.letkf import LetkfDiagnostics, analyze

    diagnostics = LetkfDiagnostics()
    cp.get_default_memory_pool().free_all_blocks()
    started = time.perf_counter()
    namespace = np if route == "host" else cp
    increments, diagnostics, stage, unstage = ra._analysis_attempt(
        analyze, prior, batches, grid, config, namespace=namespace,
        storage=route, supports_staging=True, progress=None,
        diagnostics=diagnostics)
    wall = time.perf_counter() - started
    keep = {k: v for k, v in vars(diagnostics).items()
            if isinstance(v, (int, float, str, bool, type(None)))}
    return increments, dict(route=route, wall_seconds=wall,
                            stage_seconds=stage, unstage_seconds=unstage,
                            diagnostics=keep)


def compare(new, ref) -> dict:
    """Per field: max |d|, RMS d, RMS of the reference increment, max |ref|."""
    rows = {}
    for name in ref:
        a = np.asarray(new[name], dtype=np.float64)
        b = np.asarray(ref[name], dtype=np.float64)
        d = a - b
        rows[name] = {
            "max_abs_diff": float(np.abs(d).max()),
            "rms_diff": float(np.sqrt(np.mean(d * d))),
            "rms_increment": float(np.sqrt(np.mean(b * b))),
            "max_abs_increment": float(np.abs(b).max()),
            "identical_bytes": bool(a.tobytes() == b.tobytes()),
            "zero_support_equal": bool(np.array_equal(
                np.all(a == 0.0, axis=0), np.all(b == 0.0, axis=0))),
        }
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--members", type=int, default=32)
    parser.add_argument("--shape", type=int, nargs=3, default=(49, 241, 241))
    parser.add_argument("--radars", type=int, default=8)
    parser.add_argument("--fields", type=int, default=14)
    parser.add_argument("--reference", default="cuda-resident",
                        help="cuda-resident, host-staged-cuda, host, "
                             "cuda-obs-sparse or none")
    parser.add_argument("--memory-budget-mib", type=float, default=14336)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--route", default="cuda-obs-sparse",
                        help="the route under test (default cuda-obs-sparse)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from gpuwm.da.letkf import LetkfConfig

    fields = FIELDS[:args.fields]
    t0 = time.perf_counter()
    prior, batches, grid, loc = build_case(
        members=args.members, shape=tuple(args.shape), radars=args.radars,
        fields=fields)
    build = time.perf_counter() - t0
    config = LetkfConfig(localization=loc, analysis_fields=fields,
                         rtps_alpha=0.9,
                         memory_budget_mib=args.memory_budget_mib)
    report = {"schema": "gpuwm-da.letkf-device-ab.v1",
              "members": args.members, "shape": list(args.shape),
              "radars": args.radars, "fields": list(fields),
              "build_seconds": build, "runs": []}
    first = None
    for n in range(args.repeat):
        inc, row = _solve(args.route, prior, batches, grid, config)
        report["runs"].append(row)
        print(json.dumps(row, default=str), flush=True)
        if first is None:
            first = inc
            import hashlib
            report["sha256"] = {name: hashlib.sha256(
                np.ascontiguousarray(inc[name]).tobytes()).hexdigest()
                for name in fields}
        else:
            report[f"run_to_run_{n}"] = {
                name: bool(first[name].tobytes() == inc[name].tobytes())
                for name in fields}
            del inc
    if args.reference != "none":
        ref, row = _solve(args.reference, prior, batches, grid, config)
        report["runs"].append(row)
        print(json.dumps(row, default=str), flush=True)
        report["against_reference"] = {"reference": args.reference,
                                       "fields": compare(first, ref)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report.get("against_reference"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
