"""Component byte proof and timing. This does not benchmark a forecast.

Run on a claimed CUDA card. Each row compares one mass-face operator and RK
copy/zero on the same member inputs. No forecast/member-hour speedup is inferred
from this deliberately limited workload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def _elapsed(cp, fn, repeats):
    for _ in range(5):
        fn()
    cp.cuda.Device().synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - started) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    import cupy as cp
    from gpuwm.core.kernels import get_kernel, module_source
    from gpuwm.ensemble.batch_bookkeeping import prepare_bookkeeping
    from gpuwm.ensemble.batch_kernel import (
        KernelSpec, PointerSpec, generate_batch_source, prepare_batch_kernel_launch)

    spec = KernelSpec("face_mass", "average_mass_faces", (
        PointerSpec("mu", "member"), PointerSpec("out", "member")))
    source = module_source("face_mass")
    assert generate_batch_source(source, spec, 1).encode() == source.encode()
    rows = []
    for ny, nx in ((400, 400), (200, 200)):
        for members in (1, 4, 10, 20, 40):
            # GPU setup arithmetic is shared by reference and batch, and is not
            # a forecast perturbation generator or a source-data path.
            mu = cp.arange(members * ny * nx, dtype=cp.float32).reshape(members, ny, nx)
            out = cp.empty((members, ny, nx + 1), cp.float32)
            reference = cp.empty_like(out)
            scalar = get_kernel("face_mass", "average_mass_faces")
            grid = ((ny * (nx + 1) + 127) // 128,)
            tail = (np.int32(ny), np.int32(nx), np.int32(0))
            member_views = tuple((mu[m], reference[m]) for m in range(members))

            def sequential():
                for src, dst in member_views:
                    scalar(grid, (128,), (src, dst) + tail)

            strides = {"mu": mu.strides[0], "out": out.strides[0]}

            batched = prepare_batch_kernel_launch(
                spec, members, grid, (128,), (mu, out) + tail, pointer_strides=strides)

            sequential()
            batched()
            cp.cuda.Device().synchronize()
            ref_bytes = cp.asnumpy(reference).tobytes()
            got_bytes = cp.asnumpy(out).tobytes()
            assert ref_bytes == got_bytes, (ny, nx, members)
            copied = cp.empty_like(out)
            copy = prepare_bookkeeping(((out, copied),), members=members)
            zero = prepare_bookkeeping(((copied, copied),), members=members, zero=True)
            copy()
            cp.cuda.Device().synchronize()
            assert cp.asnumpy(copied).tobytes() == got_bytes
            zero()
            cp.cuda.Device().synchronize()
            assert not cp.asnumpy(copied).view(np.uint32).any()
            sequential_s = _elapsed(cp, sequential, args.repeats)
            batch_s = _elapsed(cp, batched, args.repeats)
            rows.append({"operator": "average_mass_faces", "layout": "member_outermost",
                         "spatial_shape": [ny, nx], "members": members,
                         "sequential_wall_seconds_per_call": sequential_s,
                         "batch_wall_seconds_per_call": batch_s,
                         "sequential_over_batch": sequential_s / batch_s,
                         "submission": "prepared args and raw handle; owning-device check per call",
                         "identity": "byte_identical", "rk_copy_zero_identity": "byte_identical",
                         "output_sha256": hashlib.sha256(got_bytes).hexdigest(),
                         "array_payload_bytes": mu.nbytes + out.nbytes + reference.nbytes + copied.nbytes})
            print(json.dumps(rows[-1]), flush=True)
            del mu, out, reference, copied, member_views, copy, zero, ref_bytes, got_bytes
            cp.get_default_memory_pool().free_all_blocks()
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    receipt = {"schema": "gpuwm/ensemble-component-probe/v1", "device": name,
               "driver_version": cp.cuda.runtime.driverGetVersion(), "cupy": cp.__version__,
               "repeats": args.repeats, "scope": "component operators only; no forecast throughput",
               "n1_source_sha256": hashlib.sha256(source.encode()).hexdigest(), "rows": rows}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
