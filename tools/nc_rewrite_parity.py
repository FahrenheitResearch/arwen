#!/usr/bin/env python3
"""Compare a wrfout against its ``nc_rewrite`` rewrite, field by field.

ACCEPTANCE (2) for the Rust NetCDF writer: a real wrfout goes through
``nc_rewrite`` (netcrust/netcdf-reader in, ``netcdf-writer`` out) and the
result must be SEMANTICALLY IDENTICAL -- every dimension, every global
attribute, every variable, every attribute, every value.

The referee is netCDF4-python, i.e. the Unidata C library.  That is
deliberately a THIRD implementation: the Rust writer produced the bytes,
an independent Rust reader checks them in ``tests/classic_roundtrip.rs``,
and the C library checks them here.  A shared misreading of the format
spec cannot survive all three.

Values are compared BITWISE (``tobytes()``), not with a tolerance, and
auto-masking and auto-scaling are switched off so what is compared is
what is stored.  A tolerance would hide exactly the class of defect this
test exists to catch: a byte-swap that lands on a nearby value, a NaN
whose payload changed, a signed zero that lost its sign.

Container differences that are EXPECTED and are not failures:
  * ``file_format``  NETCDF4/NETCDF4_CLASSIC in, NETCDF3_64BIT_OFFSET or
    NETCDF3_64BIT_DATA out.  The classic containers are the ones a Rust
    writer can emit without an HDF5 encoder.
  * file SIZE.  HDF5 wrfouts are usually deflate-compressed; the classic
    format has no compression, so the rewrite is larger.
  * per-variable HDF5 storage properties (chunking, deflate level,
    shuffle) have no classic equivalent and are not carried.

DEFINITION ORDER is reported separately from content, because the two
have different causes.  Content differences are writer defects.  Order
differences come from the READ side: NetCDF-4 assigns varids in creation
order, HDF5 stores that as link creation order, and neither
``netcdf-reader`` 0.3 nor the vendored ``hdf5-reader`` 0.3 exposes it --
they enumerate a group in link-name index order.  So a NETCDF4 source is
re-emitted with its variables and global attributes permuted, addressed
by the same names with the same values.  The writer's own order fidelity
is proved by the classic-to-classic control rewrite, which must come back
byte-identical.

exit codes: 0 identical, 3 content identical but order differs,
            1 content differs

usage: nc_rewrite_parity.py ORIGINAL REWRITE [--max-report N]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import netCDF4
except ImportError:  # pragma: no cover - the referee is the whole point
    sys.exit("nc_rewrite_parity: netCDF4 is required as the independent referee")


def attr_repr(value: object) -> str:
    if isinstance(value, np.ndarray):
        return f"{value.dtype.str}{list(value.ravel()[:8])}"
    return f"{type(value).__name__}({value!r})"


def attrs_equal(left: object, right: object) -> bool:
    """Same type, same numbers -- no promotion, no rounding."""
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.dtype.kind != right_arr.dtype.kind:
        return False
    if left_arr.dtype.itemsize != right_arr.dtype.itemsize:
        return False
    if left_arr.shape != right_arr.shape:
        return False
    return left_arr.tobytes() == right_arr.tobytes()


def compare(original: str, rewrite: str, max_report: int) -> int:
    src = netCDF4.Dataset(original)
    dst = netCDF4.Dataset(rewrite)
    problems: list[str] = []
    order_notes: list[str] = []
    checked = {"dims": 0, "gattrs": 0, "vars": 0, "vattrs": 0, "values": 0}

    def note(message: str) -> None:
        if len(problems) < max_report:
            problems.append(message)

    def order_note(message: str) -> None:
        order_notes.append(message)

    # ---- dimensions -----------------------------------------------------
    src_dims = list(src.dimensions)
    dst_dims = list(dst.dimensions)
    if src_dims != dst_dims:
        if sorted(src_dims) == sorted(dst_dims):
            order_note(f"dimension order differs: {src_dims} vs {dst_dims}")
        else:
            note(f"dimension SET differs: {src_dims} vs {dst_dims}")
    for name in src_dims:
        checked["dims"] += 1
        a, b = src.dimensions[name], dst.dimensions.get(name)
        if b is None:
            note(f"dimension {name!r} missing from the rewrite")
            continue
        if len(a) != len(b):
            note(f"dimension {name!r} length {len(a)} vs {len(b)}")
        if a.isunlimited() != b.isunlimited():
            note(
                f"dimension {name!r} unlimited {a.isunlimited()} vs {b.isunlimited()}"
            )

    # ---- global attributes ---------------------------------------------
    src_gattrs = list(src.ncattrs())
    dst_gattrs = list(dst.ncattrs())
    if src_gattrs != dst_gattrs:
        missing = [n for n in src_gattrs if n not in dst_gattrs]
        extra = [n for n in dst_gattrs if n not in src_gattrs]
        if missing or extra:
            note(f"global attributes missing={missing} extra={extra}")
        else:
            moved = sum(1 for a, b in zip(src_gattrs, dst_gattrs) if a != b)
            order_note(
                f"global attribute order differs ({moved} of {len(src_gattrs)} "
                "positions moved; same names, same values)"
            )
    for name in src_gattrs:
        checked["gattrs"] += 1
        if name not in dst_gattrs:
            continue
        left, right = src.getncattr(name), dst.getncattr(name)
        if not attrs_equal(left, right):
            note(
                f"global attribute {name!r}: {attr_repr(left)} vs {attr_repr(right)}"
            )

    # ---- variables ------------------------------------------------------
    src_vars = list(src.variables)
    dst_vars = list(dst.variables)
    if src_vars != dst_vars:
        missing = [n for n in src_vars if n not in dst_vars]
        extra = [n for n in dst_vars if n not in src_vars]
        if missing or extra:
            note(f"variables missing={missing} extra={extra}")
        else:
            moved = sum(1 for a, b in zip(src_vars, dst_vars) if a != b)
            order_note(
                f"variable order differs ({moved} of {len(src_vars)} positions "
                "moved; same names, same dtypes, same values)"
            )

    for name in src_vars:
        if name not in dst_vars:
            continue
        checked["vars"] += 1
        a, b = src.variables[name], dst.variables[name]
        if a.dimensions != b.dimensions:
            note(f"{name}: dimensions {a.dimensions} vs {b.dimensions}")
            continue
        if a.dtype != b.dtype:
            note(f"{name}: dtype {a.dtype} vs {b.dtype}")
            continue
        for attr in a.ncattrs():
            checked["vattrs"] += 1
            if attr not in b.ncattrs():
                note(f"{name}: attribute {attr!r} missing from the rewrite")
                continue
            if not attrs_equal(a.getncattr(attr), b.getncattr(attr)):
                note(
                    f"{name}.{attr}: {attr_repr(a.getncattr(attr))} vs "
                    f"{attr_repr(b.getncattr(attr))}"
                )
        for attr in b.ncattrs():
            if attr not in a.ncattrs():
                note(f"{name}: attribute {attr!r} appeared in the rewrite")

        a.set_auto_maskandscale(False)
        b.set_auto_maskandscale(False)
        left = np.asarray(a[...])
        right = np.asarray(b[...])
        checked["values"] += int(left.size)
        if left.shape != right.shape:
            note(f"{name}: shape {left.shape} vs {right.shape}")
            continue
        if left.tobytes() != right.tobytes():
            if left.dtype.kind in "fc":
                delta = np.nanmax(np.abs(left.astype("f8") - right.astype("f8")))
                where = int(np.argmax((left != right).ravel()))
                note(
                    f"{name}: values differ (max |delta| {delta}, first at flat "
                    f"index {where}: {left.ravel()[where]!r} vs "
                    f"{right.ravel()[where]!r})"
                )
            else:
                note(f"{name}: values differ bitwise")

    src.close()
    dst.close()

    print(
        "CHECKED dims={dims} gattrs={gattrs} vars={vars} var_attrs={vattrs} "
        "values={values}".format(**checked)
    )
    for entry in order_notes:
        print(f"ORDER {entry}")
    if problems:
        print(f"PARITY FAIL content differences={len(problems)}")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    if order_notes:
        print(
            "PARITY CONTENT PASS every dimension, attribute and value is "
            "bit-identical; definition order differs (read-side, see the "
            "module docstring)"
        )
        return 3
    print("PARITY PASS every dimension, attribute, value AND order is identical")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original")
    parser.add_argument("rewrite")
    parser.add_argument("--max-report", type=int, default=40)
    args = parser.parse_args()
    return compare(args.original, args.rewrite, args.max_report)


if __name__ == "__main__":
    raise SystemExit(main())
