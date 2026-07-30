#!/usr/bin/env python3
"""Bounded RTX smoke/benchmark for executable WRF Dudhia shortwave.

This script deliberately exercises ``ra_lw_physics=0,ra_sw_physics=1``.
It does not enable or approximate the unfinished legacy RRTM longwave port.
The report is printed as JSON so a remote controller can retain it verbatim.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from types import SimpleNamespace

import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core.dudhia import (DudhiaShortwaveRadiation,
                               dudhia_shortwave_columns)


def _gib(nbytes: int | float) -> float:
    return float(nbytes) / 1024.0 ** 3


def _host_parity_case(ncol: int = 19, nlay: int = 23):
    sigma = np.linspace(0.08, 0.98, nlay, dtype=np.float32)[None, :]
    scale = np.linspace(0.97, 1.03, ncol, dtype=np.float32)[:, None]
    temperature = ((214.0 + 79.0 * sigma) * scale).astype(np.float32)
    pressure = ((9000.0 + 87000.0 * sigma ** np.float32(1.35))
                * scale).astype(np.float32)
    qv = ((1.0e-4 + 0.012 * sigma ** np.float32(2.2))
          * scale).astype(np.float32)
    cloud = np.exp(-((sigma - 0.64) / 0.15) ** 2).astype(np.float32)
    ice = np.exp(-((sigma - 0.31) / 0.12) ** 2).astype(np.float32)
    column = np.linspace(0.9, 1.1, ncol, dtype=np.float32)[:, None]
    qc = (3.5e-4 * cloud * column).astype(np.float32)
    qr = (8.0e-5 * cloud * column[::-1]).astype(np.float32)
    qi = (1.4e-4 * ice * column).astype(np.float32)
    qs = (5.0e-5 * ice * column[::-1]).astype(np.float32)
    qg = (2.5e-5 * cloud * ice * column).astype(np.float32)
    dz = np.broadcast_to(
        np.linspace(900.0, 350.0, nlay, dtype=np.float32),
        (ncol, nlay)).copy()
    mu = np.linspace(-0.08, 0.94, ncol, dtype=np.float32)
    albedo = np.linspace(0.06, 0.72, ncol, dtype=np.float32)
    exner = (pressure / np.float32(100000.0)) ** np.float32(287.0 / 1004.5)
    return (temperature, pressure, qv, qc, qr, qi, qs, qg, dz,
            mu, albedo, exner.astype(np.float32))


def _parity_gate(cp) -> dict[str, float]:
    host = _host_parity_case()
    expected = dudhia_shortwave_columns(
        *host[:9], host[9], host[10], solcon=1369.7, exner=host[11],
        icloud=1, swrad_scat=0.83)
    device = tuple(cp.asarray(value) for value in host)
    actual_device = dudhia_shortwave_columns(
        *device[:9], device[9], device[10], solcon=1369.7,
        exner=device[11], icloud=1, swrad_scat=0.83)
    cp.cuda.Stream.null.synchronize()
    actual = tuple(cp.asnumpy(value) for value in actual_device)
    np.testing.assert_allclose(
        actual[0], expected[0], rtol=3.0e-4, atol=2.0e-8)
    np.testing.assert_allclose(
        actual[1], expected[1], rtol=5.0e-5, atol=2.0e-3)
    np.testing.assert_allclose(
        actual[2], expected[2], rtol=5.0e-5, atol=2.0e-3)
    return {
        "heating_max_abs": float(np.max(np.abs(actual[0] - expected[0]))),
        "swdown_max_abs_wm2": float(
            np.max(np.abs(actual[1] - expected[1]))),
        "gsw_max_abs_wm2": float(np.max(np.abs(actual[2] - expected[2]))),
    }


def _profile(cp, nz: int, ny: int, nx: int):
    frac = cp.arange(nz, dtype=cp.float32)[:, None, None]
    frac = frac / cp.float32(max(nz - 1, 1))
    shape = (nz, ny, nx)

    def materialize(values):
        return cp.ascontiguousarray(cp.broadcast_to(values, shape))

    temperature = materialize(cp.float32(292.0) - cp.float32(73.0) * frac)
    pressure = materialize(
        cp.float32(95000.0) * cp.exp(cp.float32(-2.0) * frac))
    qv = materialize(cp.float32(0.011) * cp.exp(cp.float32(-4.2) * frac))
    qc = materialize(
        cp.float32(3.0e-4)
        * cp.exp(-((frac - cp.float32(0.42)) / cp.float32(0.16)) ** 2))
    qi = materialize(
        cp.float32(1.2e-4)
        * cp.exp(-((frac - cp.float32(0.72)) / cp.float32(0.13)) ** 2))
    dz = cp.full(shape, cp.float32(650.0), dtype=cp.float32)
    exner = (pressure / cp.float32(100000.0)) \
        ** cp.float32(287.0 / 1004.5)
    return temperature, pressure, qv, qc, qi, dz, exner


def _benchmark(cp, args) -> dict[str, object]:
    nz, ny, nx = args.nz, args.ny, args.nx
    temperature, pressure, qv, qc, qi, dz, exner = _profile(
        cp, nz, ny, nx)
    zero = cp.zeros((nz, ny, nx), dtype=cp.float32)
    state = SimpleNamespace(
        elapsed_seconds=0.0, qc=qc, qr=zero, qi=qi, qs=zero, qg=zero)
    atmosphere = {
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "qc": qc, "qi": qi, "dz": dz, "exner": exner,
    }
    fields = {
        "albedo": cp.full((ny, nx), cp.float32(0.19), dtype=cp.float32),
        "glw": cp.full((ny, nx), cp.float32(311.0), dtype=cp.float32),
    }
    latitude = cp.full((ny, nx), cp.float32(39.0), dtype=cp.float32)
    longitude = cp.full((ny, nx), cp.float32(-87.0), dtype=cp.float32)
    adapter = DudhiaShortwaveRadiation(
        datetime(1974, 4, 3, 18, 0), latitude, longitude)
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=4000.0, dy=4000.0,
        ztop=float(nz) * 650.0, dt=20.0, run_seconds=60.0,
        ra_physics=0, ra_lw_physics=0, ra_sw_physics=1,
        radt_minutes=12.0)
    pool = cp.get_default_memory_pool()
    cp.cuda.Stream.null.synchronize()
    # Profile construction uses temporary arrays.  Drop only free cached
    # blocks so the retained-pool delta begins at the live persistent state,
    # not at unrelated setup temporaries.
    pool.free_all_blocks()
    persistent_pool_bytes = int(pool.total_bytes())

    result = None
    for _ in range(args.warmup):
        result = adapter(
            atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    cp.cuda.Stream.null.synchronize()

    elapsed_ms = []
    for _ in range(args.iterations):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        result = adapter(
            atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
        stop.record()
        stop.synchronize()
        elapsed_ms.append(float(cp.cuda.get_elapsed_time(start, stop)))
    assert result is not None
    if not bool(cp.all(cp.isfinite(result.rthratensw))):
        raise RuntimeError("Dudhia GPU heating contains NaN or infinity")
    if not bool(cp.all(cp.isfinite(result.swdown))):
        raise RuntimeError("Dudhia GPU SWDOWN contains NaN or infinity")
    if not bool(cp.all(result.rthratenlw == cp.float32(0.0))):
        raise RuntimeError("LW=0 smoke produced a nonzero LW tendency")
    if result.glw is not fields["glw"]:
        raise RuntimeError("SW-only call did not preserve the held GLW field")
    if float(cp.max(result.rthratensw).get()) <= 0.0:
        raise RuntimeError("daylight smoke produced no positive SW heating")
    if float(cp.min(result.swdown).get()) <= 0.0:
        raise RuntimeError("daylight smoke produced non-positive SWDOWN")

    pool_bytes = int(pool.total_bytes())
    transient_pool_bytes = max(pool_bytes - persistent_pool_bytes, 0)
    if _gib(transient_pool_bytes) > args.max_pool_gib:
        raise RuntimeError(
            "Dudhia transient CuPy pool delta exceeded gate: "
            f"{_gib(transient_pool_bytes):.3f} GiB > "
            f"{args.max_pool_gib:.3f} GiB")
    analytical_bytes = (
        24 * ny * nx * nz + 40 * ny * nx + 2 * 4 * 5) * 4
    return {
        "grid": {"nz": nz, "ny": ny, "nx": nx,
                 "columns": ny * nx},
        "iterations": args.iterations,
        "warmup_calls": args.warmup,
        "elapsed_ms": {
            "min": min(elapsed_ms),
            "median": float(np.median(elapsed_ms)),
            "max": max(elapsed_ms),
        },
        "analytical_transient_envelope_gib": _gib(analytical_bytes),
        "cupy_pool_transient_delta_gib": _gib(transient_pool_bytes),
        "cupy_pool_total_gib": _gib(pool_bytes),
        "max_pool_gate_gib": args.max_pool_gib,
        "max_heating_kps": float(cp.max(result.rthratensw).get()),
        "swdown_range_wm2": [float(cp.min(result.swdown).get()),
                              float(cp.max(result.swdown).get())],
        "adapter_update_count": adapter.update_count,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nz", type=int, default=60)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-pool-gib", type=float, default=2.0)
    args = parser.parse_args()
    if min(args.nz, args.ny, args.nx, args.iterations) <= 0:
        parser.error("nz, ny, nx, and iterations must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if not math.isfinite(args.max_pool_gib) or args.max_pool_gib <= 0.0:
        parser.error("max-pool-gib must be finite and positive")
    return args


def main() -> int:
    args = _parse_args()
    try:
        import cupy as cp
        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            raise RuntimeError("CuPy reports no CUDA device")
        device_id = int(cp.cuda.runtime.getDevice())
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = props["name"]
        name = (raw_name.decode("utf-8") if isinstance(raw_name, bytes)
                else str(raw_name))
        parity = _parity_gate(cp)
        cp.get_default_memory_pool().free_all_blocks()
        benchmark = _benchmark(cp, args)
        report = {
            "verdict": "PASS",
            "scope": {"ra_lw_physics": 0, "ra_sw_physics": 1,
                      "rrtm_lw_exercised": False},
            "device": {"id": device_id, "count": count, "name": name,
                       "compute_capability": [int(props["major"]),
                                              int(props["minor"])]},
            "versions": {"cupy": cp.__version__,
                         "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
                         "driver": int(cp.cuda.runtime.driverGetVersion())},
            "numpy_cupy_parity": parity,
            "benchmark": benchmark,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)},
                         indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
