#!/usr/bin/env python3
"""Bit-identity of two directories of gpuwm wrfout frames.

The question this answers is not "are these two forecasts close" -- there is
already a matched-comparison tool for that (``matched_wrfout_stream_compare``,
correlations and MAE against a CPU reference).  This asks the only question a
REFACTOR may be judged by: did the bytes move at all.  Two integration roads
that are supposed to be the same road must produce frames that differ in zero
variables, and "zero" is the entire acceptance criterion.

Comparison is over RAW BITS, not values, and the difference matters twice:

* ``nan != nan`` under IEEE comparison, so a value test scores two identical
  NaN fields as differing and a run that went non-finite the same way on both
  arms would be reported as a divergence it is not;
* ``-0.0 == 0.0`` under IEEE comparison, so a value test scores a sign flip
  in a zero as identical -- and the sign of a zero is a real difference in
  accumulated tendency that a later step can amplify.

So each variable is viewed as unsigned integers of its own width and compared
exactly, which is the same rule ``gpuwm.core.nest_relocation`` uses when it
counts differing 32-bit patterns.

CPU-only by construction: it imports netCDF4 and NumPy and nothing else, so
it can run beside a live GPU integration without taking a byte of the card.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _bit_view(array: np.ndarray) -> np.ndarray:
    """``array`` reinterpreted as unsigned integers of the same width.

    Anything that is not a float or an int -- character arrays, which is
    what WRF's ``Times`` is -- is compared as raw bytes instead, which is
    the same test one level down.
    """
    data = np.asarray(array)
    if data.dtype.kind in ("f", "i", "u"):
        return data.view(f"u{data.dtype.itemsize}")
    return data.view("u1") if data.dtype.itemsize == 1 else \
        np.frombuffer(data.tobytes(), dtype="u1")


def compare_frames(path_a: Path, path_b: Path) -> dict:
    """One frame pair, variable by variable."""
    import netCDF4

    result: dict = {
        "a": str(path_a), "b": str(path_b),
        "variables": 0, "differing_variables": [],
        "only_in_a": [], "only_in_b": [], "shape_or_dtype_differs": [],
    }
    with netCDF4.Dataset(path_a, "r") as da, netCDF4.Dataset(path_b, "r") as db:
        names_a, names_b = set(da.variables), set(db.variables)
        result["only_in_a"] = sorted(names_a - names_b)
        result["only_in_b"] = sorted(names_b - names_a)
        common = sorted(names_a & names_b)
        result["variables"] = len(common)
        for name in common:
            va = da.variables[name][:]
            vb = db.variables[name][:]
            arr_a = np.ma.getdata(va)
            arr_b = np.ma.getdata(vb)
            if arr_a.shape != arr_b.shape or arr_a.dtype != arr_b.dtype:
                result["shape_or_dtype_differs"].append({
                    "name": name,
                    "a": [list(arr_a.shape), str(arr_a.dtype)],
                    "b": [list(arr_b.shape), str(arr_b.dtype)],
                })
                continue
            bits_a = _bit_view(arr_a).ravel()
            bits_b = _bit_view(arr_b).ravel()
            differing = int(np.count_nonzero(bits_a != bits_b))
            if differing:
                # The first differing flat index is reported because "which
                # variable" is rarely enough to localize a seam bug and
                # "where in it" usually is.
                first = int(np.flatnonzero(bits_a != bits_b)[0])
                result["differing_variables"].append({
                    "name": name, "elements": differing,
                    "of": int(bits_a.size), "first_index": first,
                })
    return result


def _frames(directory: Path, pattern: str) -> dict[str, Path]:
    return {path.name: path for path in sorted(directory.glob(pattern))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True,
                        help="one arm's directory of wrfout frames")
    parser.add_argument("--b", type=Path, required=True,
                        help="the other arm's directory")
    parser.add_argument("--pattern", default="wrfout_*",
                        help="frame glob within each directory")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the full per-frame report here")
    args = parser.parse_args(argv)

    frames_a = _frames(args.a, args.pattern)
    frames_b = _frames(args.b, args.pattern)
    paired = sorted(set(frames_a) & set(frames_b))
    report = {
        "a": str(args.a), "b": str(args.b),
        "frames_in_a": len(frames_a), "frames_in_b": len(frames_b),
        "frames_compared": len(paired),
        "unpaired_a": sorted(set(frames_a) - set(frames_b)),
        "unpaired_b": sorted(set(frames_b) - set(frames_a)),
        "frames": [],
    }
    for name in paired:
        report["frames"].append(
            compare_frames(frames_a[name], frames_b[name]))

    differing = sum(len(f["differing_variables"]) for f in report["frames"])
    mismatched = sum(len(f["shape_or_dtype_differs"]) for f in report["frames"])
    orphans = sum(len(f["only_in_a"]) + len(f["only_in_b"])
                  for f in report["frames"])
    report["total_differing_variables"] = differing
    report["total_shape_or_dtype_differs"] = mismatched
    report["total_variable_name_orphans"] = orphans
    # An empty pairing is a FAILURE, not a vacuous pass: a gate that scores
    # zero frames and prints IDENTICAL is the most expensive kind of green.
    report["identical"] = bool(
        paired and not differing and not mismatched and not orphans
        and not report["unpaired_a"] and not report["unpaired_b"])

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"frames compared: {report['frames_compared']} "
          f"(a={report['frames_in_a']}, b={report['frames_in_b']})")
    print(f"variables per frame: "
          f"{report['frames'][0]['variables'] if report['frames'] else 0}")
    print(f"differing variables: {differing}")
    print(f"shape/dtype mismatches: {mismatched}")
    print(f"variable-name orphans: {orphans}")
    print("IDENTICAL" if report["identical"] else "NOT IDENTICAL")
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
