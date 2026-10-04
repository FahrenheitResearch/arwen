"""Measure resident GPU products and Rust output from real retained frames.

This is an output-path benchmark. It does not advance a forecast and cannot
be reported as an end-to-end ensemble forecast measurement. Each member's
input is an explicit existing NetCDF frame, decoded by the Rust reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _field(spec):
    name, separator, rest = spec.partition("=")
    variable, colon, units = rest.partition(":")
    if not separator or not colon or not name or not variable or not units:
        raise argparse.ArgumentTypeError("field syntax is diagnostic=NETCDF_VARIABLE:units")
    return name, variable, units


def _threshold(spec):
    name, separator, values = spec.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("threshold syntax is diagnostic=value,value")
    return name, tuple(float(x) for x in values.split(","))


def _read_surface(reader, name, frame):
    variable = reader.variables[name]
    if variable.dtype != np.dtype("float32"):
        raise ValueError(f"{name} is not stored as a float32 diagnostic")
    variable.set_auto_maskandscale(False)
    if len(variable.shape) == 2:
        values = variable[:]
    elif len(variable.shape) == 3 and variable.dimensions[0] == "Time":
        values = variable[frame]
    else:
        raise ValueError(f"{name} is not a two-dimensional diagnosed output surface")
    decoded = np.asarray(values)
    values = np.ascontiguousarray(decoded, dtype=np.float32)
    # rw_netcdf transports numeric variables as float64. Recover the declared
    # file type only after checking that every finite value was an exact
    # widening of a stored float32. No CF/unit transform is applied here.
    finite = np.isfinite(decoded)
    if not np.array_equal(values.astype(np.float64)[finite], decoded[finite]):
        raise ValueError(f"{name} Rust numeric transport is not exact float32 widening")
    if values.ndim != 2 or not values.flags.c_contiguous:
        raise ValueError(f"{name} is not an already prepared contiguous float32 surface")
    return values


def _rewrite_full_member(reader, target):
    """Read and write every retained variable through Rust, no field arithmetic."""
    from gpuwm.io.classic_product import ClassicProduct
    with ClassicProduct(target) as writer:
        for name, dimension in reader.dimensions.items():
            writer.createDimension(name, len(dimension))
        writer.setncatts({name: reader.getncattr(name) for name in reader.ncattrs()})
        for name, variable in reader.variables.items():
            output = writer.createVariable(name, variable.dtype, variable.dimensions)
            for attr in variable.ncattrs():
                output.setncattr(attr, variable.getncattr(attr))
            output[:] = variable[:]
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-frame", action="append", type=Path, required=True)
    parser.add_argument("--field", action="append", type=_field, required=True)
    parser.add_argument("--threshold", action="append", type=_threshold, default=[])
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--valid-time", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--requested-member", type=int, action="append", default=[])
    parser.add_argument("--spaghetti", action="store_true")
    parser.add_argument("--rewrite-full-member-baseline", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.frame_index < 0:
        parser.error("iterations must be positive and frame index nonnegative")
    members = len(args.member_frame)
    if any(not 0 <= member < members for member in args.requested_member):
        parser.error("requested member index lies outside the input roster")
    if args.out.exists():
        parser.error("output directory already exists; a new directory prevents benchmark overwrite")
    args.out.mkdir(parents=True)

    import cupy as cp
    from gpuwm.netcdf_bridge import Dataset
    from gpuwm.ensemble.batch_products import FieldProducts, prepare_product_frame, write_product_frame
    started = time.perf_counter()
    thresholds = dict(args.threshold)
    requests = tuple(FieldProducts(name, units, thresholds.get(name, ()),
                                    paintball=bool(thresholds.get(name)),
                                    spaghetti=args.spaghetti and bool(thresholds.get(name)))
                     for name, _, units in args.field)
    readers = [Dataset(path) for path in args.member_frame]
    coordinates = {name: _read_surface(readers[0], name, args.frame_index) for name in ("XLAT", "XLONG")}
    fields = {}
    initial_hashes = {}
    for name, variable, _ in args.field:
        first = _read_surface(readers[0], variable, args.frame_index)
        resident = cp.empty((members,) + first.shape, cp.float32)
        initial_hashes[name] = []
        for member, reader in enumerate(readers):
            host = first if member == 0 else _read_surface(reader, variable, args.frame_index)
            if host.shape != first.shape:
                raise ValueError(f"{name} member grid differs from the ensemble grid")
            resident[member].set(host)
            initial_hashes[name].append(hashlib.sha256(host.tobytes()).hexdigest())
        fields[name] = resident
    cp.cuda.get_current_stream().synchronize()
    input_seconds = time.perf_counter() - started
    available = cp.cuda.runtime.memGetInfo()[0]
    bound = prepare_product_frame(fields, requests, available_bytes=available)
    bound()
    cp.cuda.get_current_stream().synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(args.iterations):
        bound()
    end.record()
    end.synchronize()
    product_seconds = cp.cuda.get_elapsed_time(start, end) * 0.001 / args.iterations
    final_hashes = {name: [hashlib.sha256(array[member].get().tobytes()).hexdigest()
                          for member in range(members)] for name, array in fields.items()}
    if final_hashes != initial_hashes:
        raise AssertionError("GPU products changed resident member diagnostics")

    output = args.out / "ensemble-products.nc"
    started = time.perf_counter()
    write_product_frame(output, bound, valid_time=args.valid_time,
                        latitude=coordinates["XLAT"], longitude=coordinates["XLONG"])
    products_write_seconds = time.perf_counter() - started
    requested = []
    for member in sorted(set(args.requested_member)):
        target = args.out / f"member-requested-{member:03d}.nc"
        started = time.perf_counter()
        _rewrite_full_member(readers[member], target)
        requested.append({"member": member, "seconds": time.perf_counter() - started,
                          "bytes": target.stat().st_size, "path": str(target), "sha256": _sha(target)})
    baseline = []
    if args.rewrite_full_member_baseline:
        for member, reader in enumerate(readers):
            target = args.out / f"baseline-member-{member:03d}.nc"
            started = time.perf_counter()
            _rewrite_full_member(reader, target)
            baseline.append({"member": member, "seconds": time.perf_counter() - started,
                             "bytes": target.stat().st_size, "path": str(target), "sha256": _sha(target)})
    receipt = {"benchmark_scope": "resident products and output from retained real frames",
               "forecast_advanced": False, "members": members,
               "sources": [{"path": str(path), "sha256": _sha(path)} for path in args.member_frame],
               "input_decode_upload_seconds": input_seconds,
               "gpu_products_seconds_per_frame": product_seconds,
               "aggregate_download_rust_write_seconds": products_write_seconds,
               "aggregate_bytes": output.stat().st_size,
               "aggregate_sha256": _sha(output), "requested_members": requested,
               "full_member_output_baseline": baseline,
               "input_diagnostic_identity": "PASS", "product_receipt": bound.receipt()}
    (args.out / "product-probe.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({name: receipt[name] for name in ("benchmark_scope", "forecast_advanced", "members",
                       "gpu_products_seconds_per_frame", "aggregate_download_rust_write_seconds", "aggregate_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
