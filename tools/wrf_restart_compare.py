#!/usr/bin/env python3
"""Matched-wrfout differences for the WRF warm-start proof, in NCAR's diffwrf form.

    python tools/wrf_restart_compare.py OUT.json NAME=PATH [NAME=PATH ...] \
        --pair A:B [--pair C:D ...] [--interior 5]

For every ``--pair A:B`` the difference A - B of each field in NCAR's own
diffwrf.py set (the v4.4 benchmark README's table: U V W T PH QVAPOR TSLB MU
TSK RAINC RAINNC) plus the surface fields a forecast is read by (T2 Q2 U10
V10 PSFC PBLH HFX LH GLW SWDOWN) is summarised as min, max, mean, standard
deviation and rmse over the whole grid -- the numbers NCAR publishes as
"acceptable differences" between two builds of WRF -- and again over the
interior, with the lateral boundary's ``--interior`` rows (spec_bdy_width)
trimmed, where both models are free.  Files are read through gpuwm's Rust
NetCDF bridge, one variable at a time.  Evidence tool; CPU only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ("U", "V", "W", "T", "PH", "QVAPOR", "TSLB", "MU", "TSK", "RAINC",
          "RAINNC", "T2", "Q2", "U10", "V10", "PSFC", "PBLH", "HFX", "LH",
          "GLW", "SWDOWN")


def _summary(x: np.ndarray) -> dict:
    x = x.astype(np.float64).ravel()
    return {"min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()), "std": float(x.std()),
            "rmse": float(np.sqrt(np.mean(x * x)))}


def _interior(field: np.ndarray, rows: int) -> np.ndarray:
    if rows <= 0:
        return field
    return field[..., rows:-rows, rows:-rows]


def _read(path: Path, names) -> tuple[dict[str, np.ndarray], str]:
    from gpuwm import netcdf_bridge

    out = {}
    with netcdf_bridge.open_dataset(path) as dataset:
        stamp = ""
        if "Times" in dataset.variables:
            chars = np.asarray(dataset.variables["Times"][...]).reshape(-1)
            stamp = b"".join(bytes(c) for c in chars).decode("ascii", "replace").strip("\x00 ")
        for name in names:
            if name in dataset.variables:
                value = np.asarray(dataset.variables[name][...])
                if value.ndim and dataset.variables[name].dimensions[:1] == ("Time",):
                    value = value[0]
                out[name] = value
    return out, stamp


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("files", nargs="+", metavar="NAME=PATH")
    parser.add_argument("--pair", action="append", required=True, metavar="A:B")
    parser.add_argument("--interior", type=int, default=5)
    args = parser.parse_args(argv)
    named = dict(item.split("=", 1) for item in args.files)
    data, stamps = {}, {}
    for name, path in named.items():
        data[name], stamps[name] = _read(Path(path), FIELDS)
    result = {"schema": "gpuwm-wrf-restart-compare-v1", "files": named,
              "times": stamps, "interior_rows": args.interior, "pairs": {}}
    for pair in args.pair:
        a, b = pair.split(":", 1)
        rows = {}
        for field in FIELDS:
            if field not in data[a] or field not in data[b]:
                continue
            x, y = data[a][field], data[b][field]
            if x.shape != y.shape:
                rows[field] = {"shape_mismatch": [list(x.shape), list(y.shape)]}
                continue
            d = x.astype(np.float64) - y.astype(np.float64)
            rows[field] = {"grid": _summary(d),
                           "interior": _summary(_interior(d, args.interior)),
                           "b_std": float(np.std(y.astype(np.float64)))}
        result["pairs"][f"{a} - {b}"] = rows
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for key, rows in result["pairs"].items():
        print(f"== {key}  ({stamps.get(key.split(' - ')[0])})")
        for field, row in rows.items():
            if "grid" not in row:
                print(f"  {field:7s} {row}")
                continue
            g, i = row["grid"], row["interior"]
            print(f"  {field:7s} grid mean {g['mean']: .4g} std {g['std']:.4g} "
                  f"min {g['min']: .4g} max {g['max']: .4g} | interior std "
                  f"{i['std']:.4g} rmse {i['rmse']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
