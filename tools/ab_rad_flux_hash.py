"""Deterministic solo-GPU RRTMGP flux hash and repeated-call timer.

Run this script against each checkout through PYTHONPATH so the identical
synthetic atmosphere and script exercise trunk and the candidate branch.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from types import SimpleNamespace
import time

import numpy as np


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--nz", type=int, default=49)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nx", type=int, default=250)
    parser.add_argument("--column-chunk", type=int, default=12500)
    args = parser.parse_args()
    if min(args.calls, args.nz, args.ny, args.nx, args.column_chunk) < 1:
        parser.error("calls, dimensions, and column chunk must be positive")
    if args.nz < 4 or args.nz > 128:
        parser.error("nz must be in [4, 128]")
    return args


def _synthetic_inputs(cp, nz, ny, nx):
    plev_col = np.geomspace(100000.0, 1.1, nz + 1, dtype=np.float64)
    play_col = np.sqrt(plev_col[:-1] * plev_col[1:])
    sigma = np.linspace(0.0, 1.0, nz, dtype=np.float64)
    tlay_col = 291.0 - 78.0 * sigma + 5.0 * sigma * sigma
    exner_col = (play_col / 100000.0) ** (287.0 / 1004.0)
    qv_col = np.geomspace(1.1e-2, 8.0e-7, nz, dtype=np.float64)
    y, x = np.indices((ny, nx), dtype=np.int32)
    horizontal = ((7 * x + 11 * y) % 29).astype(np.float64) / 28.0

    def layers(profile, modulation=0.0):
        values = profile[:, None, None] * (
            1.0 + modulation * (horizontal[None, :, :] - 0.5))
        return cp.asarray(np.ascontiguousarray(values), dtype=cp.float32)

    shape = (nz, ny, nx)
    cloud_vertical = np.exp(-((sigma - 0.38) / 0.11) ** 2) * 7.0e-4
    cloud_horizontal = np.where(horizontal > 0.52,
                                0.4 + 0.6 * horizontal, 0.0)
    qc = cp.asarray(np.ascontiguousarray(
        cloud_vertical[:, None, None] * cloud_horizontal[None, :, :]),
        dtype=cp.float32)
    zero = cp.zeros(shape, dtype=cp.float32)
    atmosphere = {
        "pressure": layers(play_col, 0.008),
        "p_interface": cp.asarray(np.ascontiguousarray(np.broadcast_to(
            plev_col[:, None, None], (nz + 1, ny, nx))), dtype=cp.float32),
        "temperature": layers(tlay_col, 0.004),
        "exner": layers(exner_col, 0.0),
        "qv": layers(qv_col, 0.06),
        "qc": qc,
        "qi": zero,
    }
    fields = {
        "tsk": cp.asarray(286.0 + 4.0 * horizontal, dtype=cp.float32),
        "albedo": cp.asarray(0.12 + 0.16 * horizontal, dtype=cp.float32),
        "emiss": cp.asarray(0.94 + 0.04 * horizontal, dtype=cp.float32),
    }
    latitude = cp.asarray(32.0 + 12.0 * horizontal, dtype=cp.float32)
    longitude = cp.asarray(-112.0 + 18.0 * horizontal, dtype=cp.float32)
    state = SimpleNamespace(elapsed_seconds=0.0, qc=qc, qr=zero)
    cfg = SimpleNamespace(
        mp_physics=1, dt=60.0, radt=12.0, radt_minutes=12.0)
    return atmosphere, fields, state, cfg, latitude, longitude


def _hash_result(cp, result):
    combined = hashlib.sha256()
    hashes = {}
    for name in ("rthratenlw", "rthratensw", "swdown", "glw"):
        host = np.ascontiguousarray(cp.asnumpy(getattr(result, name)))
        payload = (name.encode("ascii") + host.dtype.str.encode("ascii")
                   + np.asarray(host.shape, dtype=np.int64).tobytes()
                   + host.tobytes(order="C"))
        hashes[name] = hashlib.sha256(payload).hexdigest()
        combined.update(payload)
    return combined.hexdigest(), hashes


def main():
    args = _arguments()
    import cupy as cp
    from gpuwm.core.rrtmgp import RRTMGPRadiation

    atmosphere, fields, state, cfg, latitude, longitude = _synthetic_inputs(
        cp, args.nz, args.ny, args.nx)
    radiation = RRTMGPRadiation(
        datetime(1974, 4, 3, 18), latitude, longitude,
        column_chunk=args.column_chunk)
    result = radiation(
        atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    cp.cuda.Stream.null.synchronize()
    combined, hashes = _hash_result(cp, result)
    print(f"shape={args.nz}x{args.ny}x{args.nx}")
    print(f"column_chunk={args.column_chunk}")
    print(f"combined_sha256={combined}")
    for name, digest in hashes.items():
        print(f"{name}_sha256={digest}")

    start = time.perf_counter()
    for _ in range(args.calls):
        result = radiation(
            atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    print(f"timed_calls={args.calls}")
    print(f"elapsed_seconds={elapsed:.9f}")
    print(f"seconds_per_call={elapsed / args.calls:.9f}")


if __name__ == "__main__":
    main()
