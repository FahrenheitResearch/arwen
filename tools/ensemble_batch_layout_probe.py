"""Component layout/fusion word gates and conversion-inclusive timings.

Run only on a claimed CUDA card. This probe reports component measurements,
not forecast throughput. Local arrays may spill, so removed source accesses
are not a measured DRAM-byte reduction. Hardware counters are a separate arm.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--dx", type=float, default=1000)
    parser.add_argument("--members", type=int, nargs="+", default=[1, 4, 10, 20, 40])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--component", choices=("omega", "advection", "both"), default="both")
    parser.add_argument("--stagger", choices=("scalar", "u", "v", "w"), default="scalar")
    parser.add_argument("--mapped", action="store_true")
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.nx, args.ny, args.nz, args.repeats, *args.members) < 1 or min(args.warmup, args.reserve_mib) < 0:
        parser.error("grid, members and repeats must be positive; warmup and reserve nonnegative")
    if not np.isfinite(args.dx) or args.dx <= 0:
        parser.error("dx must be finite and positive")
    return args


def _timing(cp, operation, repeats, warmup):
    # Establish lazy compiler/copy state outside the measured allocation arm.
    operation()
    for _ in range(warmup):
        operation()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.get_current_stream().synchronize()
    pool = cp.get_default_memory_pool()
    before_live = pool.used_bytes()
    before_total = pool.total_bytes()
    requests = []
    original_allocator = cp.cuda.get_allocator()
    def observed_allocator(nbytes):
        requests.append(int(nbytes))
        return original_allocator(nbytes)
    wall = time.perf_counter()
    start.record()
    with cp.cuda.using_allocator(observed_allocator):
        for _ in range(repeats):
            operation()
    end.record()
    end.synchronize()
    wall_seconds = (time.perf_counter() - wall) / repeats
    if requests:
        raise RuntimeError(f"warmed component interval made {len(requests)} CUDA allocation requests")
    if pool.used_bytes() != before_live:
        raise RuntimeError("warmed component interval changed live CUDA pool bytes")
    return {"wall_seconds_per_call": wall_seconds,
            "gpu_seconds_per_call": float(cp.cuda.get_elapsed_time(start, end)) / (1000 * repeats),
            "timed_cuda_allocation_requests": 0, "setup_pool_live_bytes": before_live,
            "setup_calls": 1,
            "setup_pool_total_bytes": before_total, "after_pool_live_bytes": pool.used_bytes(),
            "after_pool_total_bytes": pool.total_bytes()}


def _hash(cp, array):
    digest = hashlib.sha256()
    # Limit proof transfer memory to one member, even at the large N=40 grid.
    for member in range(array.shape[0]):
        digest.update(cp.asnumpy(array[member]).tobytes())
    return digest.hexdigest()


def _gate(cp, actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError("layout candidate changed output shape or dtype")
    for member in range(actual.shape[0]):
        a, b = actual[member].view(cp.uint32), expected[member].view(cp.uint32)
        if not bool(cp.all(a == b)):
            changed = int(cp.count_nonzero(a != b))
            raise RuntimeError(f"candidate changed {changed} output words in member {member}")
    return _hash(cp, actual)


def _allocation_bytes(arrays):
    # N=1 conversion can be a contiguous view of the same allocation.
    return sum({(int(value.data.ptr), int(value.nbytes)): int(value.nbytes)
                for value in arrays}.values())


def _allocation_charge(arrays):
    return sum(((value + 511) // 512) * 512 for value in {
        (int(array.data.ptr), int(array.nbytes)): int(array.nbytes) for array in arrays}.values())


def _planned_payloads(args, members, component):
    nz, ny, nx = args.nz, args.ny, args.nx
    ru, rv, rw = nz * ny * (nx + 1), nz * (ny + 1) * nx, (nz + 1) * ny * nx
    if component == "omega":
        words = [members * value for value in (ru, ru, rv, rv, rw, rw, rw)]
        words += [nz, nz, ny * nx]
    else:
        field = (nz + (args.stagger == "w")) * (ny + (args.stagger == "v")) * (nx + (args.stagger == "u"))
        words = [members * value for value in (field, field, field, field, field, ru, ru, rv, rv, rw, rw)]
        words += [nz, nz, nz, (ny + (args.stagger == "v")) * (nx + (args.stagger == "u"))]
    return tuple(4 * value for value in words)


def _planned_bytes(args, members, component):
    return sum(((value + 511) // 512) * 512 for value in _planned_payloads(args, members, component))


def _admit(cp, args, members, component):
    free, _ = cp.cuda.runtime.memGetInfo()
    payload = _planned_bytes(args, members, component)
    reserve = args.reserve_mib * 1024**2
    if payload + reserve > free:
        return {"component": component, "members": members, "status": "memory_not_admitted",
                "planned_array_allocated_bytes": payload,
                "planned_array_payload_bytes": sum(_planned_payloads(args, members, component)),
                "reserve_bytes": reserve, "free_bytes": free}
    return None


def _random(cp, rng, shape, amplitude):
    array = rng.random(shape, dtype=cp.float32)
    array *= np.float32(2 * amplitude)
    array -= np.float32(amplitude)
    return array


def omega(cp, args, members):
    from gpuwm.ensemble.batch_fluxes import prepare_omega_columns
    from gpuwm.ensemble.batch_layout_trials import prepare_omega_trial, pack_member_innermost, unpack_member_innermost
    nz, ny, nx = args.nz, args.ny, args.nx
    rng = cp.random.default_rng(4123)
    ru = _random(cp, rng, (members, nz, ny, nx + 1), 80000)
    rv = _random(cp, rng, (members, nz, ny + 1, nx), 90000)
    dnw = cp.asarray(np.linspace(-0.3, -0.02, nz, dtype=np.float32))
    c1h = cp.asarray(np.linspace(1, 0.125, nz, dtype=np.float32))
    msft = cp.full((ny, nx), np.float32(1.25), cp.float32)
    reference = cp.empty((members, nz + 1, ny, nx), cp.float32)
    output = cp.empty_like(reference)
    inner_ru, inner_rv, inner_ww = tuple(pack_member_innermost(value) for value in (ru, rv, output))
    options = dict(dx=args.dx, dy=args.dx, has_msf=args.mapped, msft=msft)
    baseline = prepare_omega_columns(ru, rv, reference, dnw, c1h, **options)
    scalar = tuple(prepare_omega_columns(ru[m:m + 1], rv[m:m + 1], reference[m:m + 1], dnw, c1h,
                                         **options) for m in range(members))
    for original in scalar:
        original()
    expected_hash = _hash(cp, reference)
    input_hashes = tuple(_hash(cp, value) for value in (ru, rv))
    def pack():
        cp.copyto(inner_ru, cp.moveaxis(ru, 0, -1))
        cp.copyto(inner_rv, cp.moveaxis(rv, 0, -1))
    def unpack():
        unpack_member_innermost(inner_ww, out=output)
    arms = {"baseline_outermost": baseline}
    metadata = {}
    for layout, cached, key in (("outermost", False, "outermost_local"),
                                ("innermost", False, "innermost_local"),
                                ("innermost", True, "innermost_local_shared_coefficients")):
        fields = (ru, rv, output) if layout == "outermost" else (inner_ru, inner_rv, inner_ww)
        launch = prepare_omega_trial(*fields, dnw, c1h, layout=layout, cache_shared=cached, **options)
        launch()
        if layout == "innermost":
            unpack()
        if _gate(cp, output, reference) != expected_hash:
            raise RuntimeError("Omega hash changed after its word gate")
        arms[key] = launch
        metadata[key] = dict(launch.metadata)
        if layout == "innermost":
            def converted(operation=launch):
                pack()
                operation()
                unpack()
            arms[key + "_conversion_included"] = converted
    timings = {key: _timing(cp, operation, args.repeats, args.warmup) for key, operation in arms.items()}
    timings["pack_inputs"] = _timing(cp, pack, args.repeats, args.warmup)
    timings["unpack_output"] = _timing(cp, unpack, args.repeats, args.warmup)
    _gate(cp, output, reference)
    if tuple(_hash(cp, value) for value in (ru, rv)) != input_hashes:
        raise RuntimeError("Omega trial changed immutable member inputs")
    arrays = (ru, rv, dnw, c1h, msft, reference, output, inner_ru, inner_rv, inner_ww)
    return {"component": "omega", "members": members, "status": "byte_identical",
            "output_sha256": expected_hash, "timings": timings, "candidates": metadata,
            "array_payload_bytes": _allocation_bytes(arrays),
            "array_allocated_bytes": _allocation_charge(arrays), "allocation_quantum": 512,
            "padding": "none; exact N", "conversion": "RU/RV pack and WW unpack included in named arms"}


def advection(cp, args, members):
    from gpuwm.ensemble.batch_operators import prepare_flux_div
    from gpuwm.ensemble.batch_layout_trials import prepare_flux_div_trial, pack_member_innermost, unpack_member_innermost
    nz, ny, nx = args.nz, args.ny, args.nx
    stagger = {"scalar": "", "u": "x", "v": "y", "w": "z"}[args.stagger]
    nlev, nys, nxs = nz + (stagger == "z"), ny + (stagger == "y"), nx + (stagger == "x")
    rng = cp.random.default_rng(6911)
    field = _random(cp, rng, (members, nlev, nys, nxs), 35)
    ru = _random(cp, rng, (members, nz, ny, nx + 1), 80000)
    rv = _random(cp, rng, (members, nz, ny + 1, nx), 90000)
    rw = _random(cp, rng, (members, nz + 1, ny, nx), 4000)
    spacing = cp.asarray(np.linspace(-5, -1, nz, dtype=np.float32))
    fnm = cp.asarray(np.linspace(0.2, 0.8, nz, dtype=np.float32))
    fnp = cp.asarray(np.float32(1) - np.linspace(0.2, 0.8, nz, dtype=np.float32))
    msf = cp.full((nys, nxs), np.float32(1.25), cp.float32)
    reference = cp.empty_like(field)
    output = cp.empty_like(field)
    packed = tuple(pack_member_innermost(value) for value in (field, ru, rv, rw, output))
    options = dict(dx=args.dx, dy=args.dx, stagger=stagger, has_msf=args.mapped)
    original = prepare_flux_div(field, ru, rv, rw, reference, spacing, fnm, fnp, msf, **options)
    scalar = tuple(prepare_flux_div(field[m:m + 1], ru[m:m + 1], rv[m:m + 1], rw[m:m + 1], reference[m:m + 1],
                                   spacing, fnm, fnp, msf, **options) for m in range(members))
    reference.fill(0)
    for operation in scalar:
        operation()
    expected_hash = _hash(cp, reference)
    def baseline():
        reference.fill(0)
        original()
    def pack():
        for outer, inner in zip((field, ru, rv, rw), packed[:4]):
            cp.copyto(inner, cp.moveaxis(outer, 0, -1))
    def unpack():
        unpack_member_innermost(packed[-1], out=output)
    arms, metadata = {"baseline_zero_then_outermost": baseline}, {}
    for layout in ("outermost", "innermost"):
        fields = (field, ru, rv, rw, output) if layout == "outermost" else packed
        launch = prepare_flux_div_trial(*fields, spacing, fnm, fnp, msf,
                                        layout=layout, zero_tendency=True, **options)
        output.fill(-19.5)
        launch()
        if layout == "innermost":
            unpack()
        _gate(cp, output, reference)
        key = layout + "_fused_zero"
        arms[key], metadata[key] = launch, dict(launch.metadata)
        if layout == "innermost":
            def converted(operation=launch):
                pack()
                operation()
                unpack()
            arms[key + "_conversion_included"] = converted
    timings = {key: _timing(cp, operation, args.repeats, args.warmup) for key, operation in arms.items()}
    timings["pack_inputs"] = _timing(cp, pack, args.repeats, args.warmup)
    timings["unpack_output"] = _timing(cp, unpack, args.repeats, args.warmup)
    _gate(cp, output, reference)
    arrays = (field, ru, rv, rw, spacing, fnm, fnp, msf, reference, output) + packed
    return {"component": "advection", "stagger": stagger, "members": members, "status": "byte_identical",
            "output_sha256": expected_hash, "timings": timings, "candidates": metadata,
            "array_payload_bytes": _allocation_bytes(arrays),
            "array_allocated_bytes": _allocation_charge(arrays), "allocation_quantum": 512,
            "padding": "none; exact N", "conversion": "field/RU/RV/RW pack and tendency unpack included in named arm"}


def _write(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    args = arguments(argv)
    components = ("omega", "advection") if args.component == "both" else (args.component,)
    receipt = {"schema": "gpuwm/ensemble-layout-component-probe/v1",
               "scope": "component word gates and timings; no forecast speedup or measured DRAM claim",
               "spatial_shape": [args.nz, args.ny, args.nx], "repeats": args.repeats,
               "warmup": args.warmup, "rows": []}
    if args.estimate_only:
        receipt["rows"] = [{"component": component, "members": members,
                             "planned_array_allocated_bytes": _planned_bytes(args, members, component),
                             "planned_array_payload_bytes": sum(_planned_payloads(args, members, component)),
                             "allocation_quantum": 512,
                             "reserve_bytes": args.reserve_mib * 1024**2}
                            for component in components for members in args.members]
        _write(args.receipt, receipt)
        return
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    receipt.update(device=name.decode() if isinstance(name, bytes) else str(name),
                   driver_version=cp.cuda.runtime.driverGetVersion(), cupy=cp.__version__)
    for component in components:
        for members in args.members:
            refused = _admit(cp, args, members, component)
            if refused:
                row = refused
            else:
                try:
                    row = globals()[component](cp, args, members)
                except BaseException as error:
                    receipt["rows"].append({"component": component, "members": members,
                                            "status": "failed", "failure_type": type(error).__name__,
                                            "reason": str(error)})
                    _write(args.receipt, receipt)
                    raise
            receipt["rows"].append(row)
            _write(args.receipt, receipt)
            print(json.dumps(row), flush=True)
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
