#!/usr/bin/env python3
"""Benchmark CPU/CUDA transforms on a hash-bound accepted GFS fixture."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from gpuwm.gfs_direct import _load_bridge_snapshots, _read_series
from gpuwm.ingest.backend_contract import ArrayParityRule, compare_backend_outputs
from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
from gpuwm.ingest.horiz import interpolate_regular_gpu
from gpuwm.ingest.hrrr_target import load_hrrr_target_domain
from gpuwm.ingest.prepared_cache import _array_sha256
from gpuwm.ingest.vert import wrf_vert_interp_gpu


SCHEMA = "gpuwm-preprocess-backend-benchmark-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepared_array(root: Path, header: dict[str, object], name: str):
    descriptor = header["arrays"][name]
    path = root / descriptor["file"]
    if path.stat().st_size != descriptor["nbytes"] + 128:
        raise ValueError(f"prepared array byte count changed: {name}")
    value = np.load(path, allow_pickle=False)
    if list(value.shape) != descriptor["shape"] \
            or value.dtype.name != descriptor["dtype"]:
        raise ValueError(f"prepared array descriptor changed: {name}")
    if _array_sha256(value) != descriptor["sha256"]:
        raise ValueError(f"prepared array payload digest changed: {name}")
    return value


def _measure(operation, *, repeats: int, synchronize=None):
    samples = []
    output = None
    for _ in range(repeats):
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        output = operation()
        if synchronize is not None:
            synchronize()
        samples.append(time.perf_counter() - started)
    return output, {
        "samples_seconds": samples,
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
    }


def run(args) -> dict[str, object]:
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    records = _read_series(args.series)
    cycle = datetime.strptime(args.cycle, "%Y-%m-%d_%H:%M:%S")
    snapshots = _load_bridge_snapshots(args.decoded, cycle, records)
    source = snapshots[0]
    target = load_hrrr_target_domain(args.domain_spec)
    grid = target.grid()
    target_latitude, target_longitude = grid.latlon_mass()

    prepared_header_path = args.prepared_cache / "header.json"
    prepared_header = json.loads(prepared_header_path.read_text(encoding="utf-8"))
    target_pressure = _prepared_array(
        args.prepared_cache, prepared_header, "base/pb")
    surface_pressure = _prepared_array(
        args.prepared_cache, prepared_header, "result/surface_pressure")
    surface_temperature = _prepared_array(
        args.prepared_cache, prepared_header, "met/T2")
    expected_shape = (target.ny, target.nx)
    if target_pressure.shape[1:] != expected_shape \
            or surface_pressure.shape != expected_shape \
            or surface_temperature.shape != expected_shape:
        raise ValueError("domain specification differs from prepared fixture")

    backend = CpuPreprocessBackend(args.cpu_bridge)
    import cupy as cp

    cuda_sync = cp.cuda.Stream.null.synchronize
    source_temperature = source.fields["T"]

    # Warm both implementations before measured repetitions.
    backend.interpolate_regular(
        source_temperature, source.latitude, source.longitude,
        target_latitude, target_longitude, method="parabolic",
        workers=args.workers)
    interpolate_regular_gpu(
        source_temperature, source.latitude, source.longitude,
        target_latitude, target_longitude, method="parabolic")
    cuda_sync()

    horizontal_serial, horizontal_serial_timing = _measure(
        lambda: backend.interpolate_regular(
            source_temperature, source.latitude, source.longitude,
            target_latitude, target_longitude, method="parabolic", workers=1),
        repeats=args.repeats,
    )
    horizontal_parallel, horizontal_parallel_timing = _measure(
        lambda: backend.interpolate_regular(
            source_temperature, source.latitude, source.longitude,
            target_latitude, target_longitude, method="parabolic",
            workers=args.workers),
        repeats=args.repeats,
    )
    horizontal_cuda, horizontal_cuda_timing = _measure(
        lambda: interpolate_regular_gpu(
            source_temperature, source.latitude, source.longitude,
            target_latitude, target_longitude, method="parabolic"),
        repeats=args.repeats, synchronize=cuda_sync,
    )
    horizontal_cuda = cp.asnumpy(horizontal_cuda)

    source_pressure = np.broadcast_to(
        np.asarray(source.levels_hpa, dtype=np.float32)[:, None, None]
        * np.float32(100.0), horizontal_parallel.shape)
    # Warm both vertical paths on the exact accepted target columns.
    backend.wrf_vertical_interpolate(
        horizontal_parallel, surface_temperature, source_pressure,
        surface_pressure, target_pressure, extrap="temperature",
        workers=args.workers)
    wrf_vert_interp_gpu(
        cp.asarray(horizontal_cuda), cp.asarray(surface_temperature),
        cp.asarray(source_pressure), cp.asarray(surface_pressure),
        cp.asarray(target_pressure), extrap="temperature")
    cuda_sync()

    vertical_serial, vertical_serial_timing = _measure(
        lambda: backend.wrf_vertical_interpolate(
            horizontal_parallel, surface_temperature, source_pressure,
            surface_pressure, target_pressure, extrap="temperature",
            workers=1),
        repeats=args.repeats,
    )
    vertical_parallel, vertical_parallel_timing = _measure(
        lambda: backend.wrf_vertical_interpolate(
            horizontal_parallel, surface_temperature, source_pressure,
            surface_pressure, target_pressure, extrap="temperature",
            workers=args.workers),
        repeats=args.repeats,
    )
    vertical_cuda, vertical_cuda_timing = _measure(
        lambda: wrf_vert_interp_gpu(
            cp.asarray(horizontal_cuda), cp.asarray(surface_temperature),
            cp.asarray(source_pressure), cp.asarray(surface_pressure),
            cp.asarray(target_pressure), extrap="temperature"),
        repeats=args.repeats, synchronize=cuda_sync,
    )
    vertical_cuda = cp.asnumpy(vertical_cuda)

    worker_identity = {
        "horizontal": bool(np.array_equal(
            horizontal_serial, horizontal_parallel)),
        "vertical": bool(np.array_equal(vertical_serial, vertical_parallel)),
    }
    parity = compare_backend_outputs(
        {"horizontal": horizontal_cuda, "vertical": vertical_cuda},
        {"horizontal": horizontal_parallel, "vertical": vertical_parallel},
        rules={
            "horizontal": ArrayParityRule(rtol=3.0e-6, atol=3.0e-6),
            "vertical": ArrayParityRule(rtol=3.0e-5, atol=5.0e-3),
        },
    )
    horizontal_speedup = (
        horizontal_serial_timing["median_seconds"]
        / horizontal_parallel_timing["median_seconds"])
    vertical_speedup = (
        vertical_serial_timing["median_seconds"]
        / vertical_parallel_timing["median_seconds"])
    status = "PASS" if all(worker_identity.values()) \
        and parity["status"] == "PASS" else "FAIL"
    return {
        "schema": SCHEMA,
        "status": status,
        "fixture": {
            "decoded_gate_sha256": _sha256(args.decoded / "gate.tsv"),
            "decoded_manifest_sha256": _sha256(
                args.decoded / "decoded-sha256.tsv"),
            "series_sha256": _sha256(args.series),
            "domain_spec_sha256": _sha256(args.domain_spec),
            "prepared_header_sha256": _sha256(prepared_header_path),
            "source_shape": list(source_temperature.shape),
            "target_shape": list(expected_shape),
            "source_levels": int(source_temperature.shape[0]),
            "target_levels": int(target_pressure.shape[0]),
        },
        "backend": {
            "cpu_bridge": str(backend.path),
            "cpu_bridge_sha256": _sha256(backend.path),
            "workers": args.workers,
            "cuda_device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        },
        "timing": {
            "horizontal_serial_cpu": horizontal_serial_timing,
            "horizontal_parallel_cpu": horizontal_parallel_timing,
            "horizontal_cuda": horizontal_cuda_timing,
            "horizontal_cpu_speedup": horizontal_speedup,
            "vertical_serial_cpu": vertical_serial_timing,
            "vertical_parallel_cpu": vertical_parallel_timing,
            "vertical_cuda": vertical_cuda_timing,
            "vertical_cpu_speedup": vertical_speedup,
        },
        "worker_count_byte_identity": worker_identity,
        "cuda_cpu_parity": parity,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--domain-spec", type=Path, required=True)
    parser.add_argument("--prepared-cache", type=Path, required=True)
    parser.add_argument("--cpu-bridge", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = run(args)
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
