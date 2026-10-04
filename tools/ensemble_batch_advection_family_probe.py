"""Four-stagger family word gates and one-launch component timings.

Run on a claimed CUDA card. Full conversion arms include seven input packs and
four output unpacks. These component timings do not establish forecast gains.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

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
    parser.add_argument("--layout", choices=("outermost", "innermost", "both"), default="both")
    parser.add_argument("--mapped", action="store_true")
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.nx, args.ny, args.nz, args.repeats, *args.members) < 1 or min(args.warmup, args.reserve_mib) < 0:
        parser.error("grid, members and repeats must be positive; warmup/reserve nonnegative")
    if not np.isfinite(args.dx) or args.dx <= 0:
        parser.error("dx must be finite and positive")
    return args


def _plan(args, members):
    nz, ny, nx = args.nz, args.ny, args.nx
    fields = (nz * ny * nx, nz * ny * (nx + 1), nz * (ny + 1) * nx, (nz + 1) * ny * nx)
    words = list(fields) * 3 + list(fields[1:])
    if args.layout != "outermost" and members > 1:
        words += list(fields) * 2 + list(fields[1:])
    payloads = [4 * members * value for value in words]
    payloads += [4 * nz] * 4 + [4 * ny * nx, 4 * ny * (nx + 1), 4 * (ny + 1) * nx]
    return {"planned_array_payload_bytes": sum(payloads),
            "planned_array_allocated_bytes": sum(((value + 511) // 512) * 512 for value in payloads),
            "allocation_quantum": 512, "reserve_bytes": args.reserve_mib * 1024**2}


def _write(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def measure(cp, args, members):
    from gpuwm.ensemble.batch_bookkeeping import prepare_bookkeeping
    from gpuwm.ensemble.batch_layout_trials import (
        prepare_advection_family_trial, pack_member_innermost, unpack_member_innermost)
    from gpuwm.ensemble.batch_operators import prepare_flux_div
    from tools.ensemble_batch_layout_probe import (
        _random, _gate, _hash, _timing, _allocation_bytes, _allocation_charge)
    nz, ny, nx = args.nz, args.ny, args.nx
    shapes = ((members, nz, ny, nx), (members, nz, ny, nx + 1),
              (members, nz, ny + 1, nx), (members, nz + 1, ny, nx))
    rng = cp.random.default_rng(71911)
    fields = tuple(_random(cp, rng, shape, 35) for shape in shapes)
    flows = tuple(_random(cp, rng, shape, 80000) for shape in shapes[1:])
    rdnw, rdn = tuple(cp.asarray(np.linspace(-5, -1, nz, dtype=np.float32)) for _ in range(2))
    fnm = cp.asarray(np.linspace(0.2, 0.8, nz, dtype=np.float32))
    fnp = cp.asarray(np.float32(1) - np.linspace(0.2, 0.8, nz, dtype=np.float32))
    maps = tuple(cp.full(shape[2:], np.float32(1.25), cp.float32) for shape in shapes[:3])
    spacings = (rdnw, rdnw, rdnw, rdn)
    target_maps = (maps[0], maps[1], maps[2], maps[0])
    reference = tuple(cp.empty_like(value) for value in fields)
    output = tuple(cp.empty_like(value) for value in fields)
    inner_fields = inner_flows = inner_output = ()
    if args.layout != "outermost":
        inner_fields = tuple(pack_member_innermost(value) for value in fields)
        inner_flows = tuple(pack_member_innermost(value) for value in flows)
        inner_output = tuple(pack_member_innermost(value) for value in output)
    del rng
    options = dict(dx=args.dx, dy=args.dx, has_msf=args.mapped)
    zero = prepare_bookkeeping(tuple((value, value) for value in reference), members=members, zero=True)
    separate = tuple(prepare_flux_div(field, *flows, tendency, spacing, fnm, fnp, msf,
                                      stagger=stagger, **options)
                     for field, tendency, spacing, msf, stagger in
                     zip(fields, reference, spacings, target_maps, ("", "x", "y", "z")))
    originals = tuple(prepare_flux_div(
        fields[at][member:member + 1], *(value[member:member + 1] for value in flows),
        reference[at][member:member + 1], spacings[at], fnm, fnp, target_maps[at],
        stagger=stagger, **options)
        for at, stagger in enumerate(("", "x", "y", "z")) for member in range(members))
    zero()
    for original in originals:
        original()
    expected = tuple(_hash(cp, value) for value in reference)
    immutable = fields + flows + (rdnw, rdn, fnm, fnp) + maps
    input_hashes = tuple(_hash(cp, value) for value in immutable)
    def baseline():
        zero()
        for operation in separate:
            operation()
    baseline()
    if tuple(_hash(cp, value) for value in reference) != expected:
        raise RuntimeError("separate batch entries changed the independent scalar family result")
    def pack():
        for value, inner in zip(fields + flows, inner_fields + inner_flows):
            cp.copyto(inner, cp.moveaxis(value, 0, -1))
    def unpack():
        for value, outer in zip(inner_output, output):
            unpack_member_innermost(value, out=outer)
    arms = {"baseline_zero_then_four_entries": baseline}
    metadata = {}
    layouts = ("outermost", "innermost") if args.layout == "both" else (args.layout,)
    for layout in layouts:
        working_fields = fields if layout == "outermost" else inner_fields
        working_flows = flows if layout == "outermost" else inner_flows
        working_output = output if layout == "outermost" else inner_output
        rows = tuple((field, tendency, spacing, msf, stagger) for field, tendency, spacing, msf, stagger in
                     zip(working_fields, working_output, spacings, target_maps, ("", "x", "y", "z")))
        launch = prepare_advection_family_trial(rows, *working_flows, fnm, fnp,
                                                layout=layout, zero_tendency=True, **options)
        for value in working_output:
            value.fill(-19.5)
        launch()
        if layout == "innermost":
            unpack()
        for value, target in zip(output, reference):
            _gate(cp, value, target)
        key = layout + "_family_fused_zero"
        arms[key] = launch
        metadata[key] = dict(launch.metadata)
        if layout == "innermost":
            def converted(operation=launch):
                pack()
                operation()
                unpack()
            arms[key + "_conversion_included"] = converted
    timings = {key: _timing(cp, operation, args.repeats, args.warmup) for key, operation in arms.items()}
    if inner_output:
        timings["pack_seven_inputs"] = _timing(cp, pack, args.repeats, args.warmup)
        timings["unpack_four_outputs"] = _timing(cp, unpack, args.repeats, args.warmup)
    for value, target in zip(output, reference):
        _gate(cp, value, target)
    if tuple(_hash(cp, value) for value in immutable) != input_hashes:
        raise RuntimeError("family trial changed immutable field/flux/coefficient inputs")
    arrays = immutable + reference + output + inner_fields + inner_flows + inner_output
    return {"members": members, "status": "byte_identical_four_outputs", "output_sha256": expected,
            "timings": timings, "candidates": metadata, "array_payload_bytes": _allocation_bytes(arrays),
            "array_allocated_bytes": _allocation_charge(arrays), "allocation_quantum": 512,
            "dispatch": "one family kernel per N>1 stage; N1 retains original entries",
            "conversion": "seven input packs plus four output unpacks in the named arm",
            "padding": "none; exact N"}


def main(argv=None):
    args = arguments(argv)
    receipt = {"schema": "gpuwm/advection-family-component-probe/v1",
               "scope": "all four component outputs; no complete-driver forecast gain or measured DRAM claim",
               "spatial_shape": [args.nz, args.ny, args.nx], "repeats": args.repeats,
               "warmup": args.warmup, "layout": args.layout, "rows": []}
    if args.estimate_only:
        receipt["rows"] = [{"members": members, **_plan(args, members)} for members in args.members]
        _write(args.receipt, receipt)
        return
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    receipt.update(device=name.decode() if isinstance(name, bytes) else str(name),
                   driver_version=cp.cuda.runtime.driverGetVersion(), cupy=cp.__version__)
    for members in args.members:
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        quote = _plan(args, members)
        available, _ = cp.cuda.runtime.memGetInfo()
        if quote["planned_array_allocated_bytes"] + quote["reserve_bytes"] > available:
            row = {"members": members, "status": "memory_not_admitted", "free_bytes": available, **quote}
        else:
            try:
                row = measure(cp, args, members)
            except BaseException as error:
                receipt["rows"].append({"members": members, "status": "failed",
                                        "failure_type": type(error).__name__, "reason": str(error)})
                _write(args.receipt, receipt)
                raise
        receipt["rows"].append(row)
        _write(args.receipt, receipt)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
