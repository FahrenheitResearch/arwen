#!/usr/bin/env python3
"""Copy a wrfinput file and apply exactly one documented floating-point ULP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Sequence

import netCDF4
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _logical_indices(variable, k: int, j: int, i: int) -> tuple[int, ...]:
    indices = []
    for dimension in variable.dimensions:
        lower = dimension.lower()
        if lower == "time":
            indices.append(0)
        elif "bottom_top" in lower:
            indices.append(k)
        elif "south_north" in lower:
            indices.append(j)
        elif "west_east" in lower:
            indices.append(i)
        else:
            raise ValueError(
                f"cannot map WRF dimension {dimension!r} to k/j/i")
    return tuple(indices)


def _default_point(variable, ordinal: int) -> tuple[int, int, int]:
    if not 0 <= ordinal < 4:
        raise ValueError("default member ordinal must be 0..3")
    sizes = {name.lower(): size for name, size in zip(
        variable.dimensions, variable.shape) if name.lower() != "time"}
    nz = next(size for name, size in sizes.items() if "bottom_top" in name)
    ny = next(size for name, size in sizes.items() if "south_north" in name)
    nx = next(size for name, size in sizes.items() if "west_east" in name)
    fractions = ((1, 4, 1, 4, 1, 4), (1, 3, 1, 3, 2, 3),
                 (1, 2, 2, 3, 1, 3), (3, 4, 3, 4, 3, 4))
    kn, kd, jn, jd, inn, ind = fractions[ordinal]
    clamp = lambda value, size, margin: min(max(value, margin), size - margin - 1)
    return (clamp(nz * kn // kd, nz, 1),
            clamp(ny * jn // jd, ny, min(5, max(1, (ny - 1) // 3))),
            clamp(nx * inn // ind, nx, min(5, max(1, (nx - 1) // 3))))


def perturb(source: str | Path, destination: str | Path, *, field: str = "T",
            index: tuple[int, int, int] | None = None,
            member_ordinal: int | None = None, direction: str = "positive",
            record_path: str | Path | None = None) -> dict[str, object]:
    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(source)
    shutil.copy2(source, destination)
    with netCDF4.Dataset(destination, "r+") as dataset:
        if field not in dataset.variables:
            raise ValueError(f"wrfinput has no field {field!r}")
        variable = dataset.variables[field]
        if variable.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError(f"{field} must be float32/float64, got {variable.dtype}")
        variable_dtype = np.dtype(variable.dtype)
        variable_dimensions = tuple(variable.dimensions)
        if index is None:
            if member_ordinal is None:
                raise ValueError("provide --index or --member-ordinal")
            index = _default_point(variable, member_ordinal)
        k, j, i = index
        netcdf_index = _logical_indices(variable, k, j, i)
        before = np.asarray(variable[netcdf_index], dtype=variable.dtype).reshape(())[()]
        target = (np.array(np.inf, dtype=variable.dtype)[()]
                  if direction == "positive"
                  else np.array(-np.inf, dtype=variable.dtype)[()])
        after = np.nextafter(before, target, dtype=variable.dtype)
        if not np.isfinite(before) or not np.isfinite(after) or before == after:
            raise ValueError("selected perturbation cannot produce one finite ULP")
        variable[netcdf_index] = after
        dataset.sync()
    bits_dtype = np.uint32 if variable_dtype.itemsize == 4 else np.uint64
    width = 8 if bits_dtype is np.uint32 else 16
    before_bits = int(np.asarray(before).view(bits_dtype))
    after_bits = int(np.asarray(after).view(bits_dtype))
    record: dict[str, object] = {
        "schema": 1, "operation": "numpy.nextafter", "field": field,
        "k": k, "j": j, "i": i,
        "netcdf_dimensions": list(variable_dimensions),
        "netcdf_index": list(netcdf_index), "direction": direction,
        "dtype": str(variable_dtype), "before": float(before),
        "after": float(after),
        "before_hex_bits": f"0x{before_bits:0{width}x}",
        "after_hex_bits": f"0x{after_bits:0{width}x}",
        "source_sha256": source_hash,
        "perturbed_sha256": _sha256(destination),
    }
    target_path = (destination.parent / "perturbation.json"
                   if record_path is None else Path(record_path))
    target_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _parse_index(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("index must be k,j,i") from exc
    if len(result) != 3 or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("index must be three non-negative k,j,i values")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--field", default="T")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=_parse_index)
    group.add_argument("--member-ordinal", type=int, choices=range(4))
    parser.add_argument("--direction", choices=("positive", "negative"),
                        default="positive")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    perturb(args.source, args.destination, field=args.field, index=args.index,
            member_ordinal=args.member_ordinal, direction=args.direction,
            record_path=args.record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
