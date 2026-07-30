#!/usr/bin/env python3
"""Freeze WRF ``wrfinput``/``wrfbdy`` declaration contracts.

The resulting JSON contains metadata only: dimension declarations, global
attributes, variable declarations, and small horizontal-independent constant
values.  It deliberately does not copy a WRF data file or any gridded oracle
state, so the runtime exporter cannot accidentally depend on ``real.exe``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("latin1")
    return value


def _attribute_type(value):
    if isinstance(value, str):
        return {"kind": "str"}
    if isinstance(value, bytes):
        return {"kind": "bytes"}
    array = np.asarray(value)
    return {
        "kind": "numeric",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _spec(path: Path) -> dict[str, object]:
    with netCDF4.Dataset(path) as dataset:
        dimensions = [
            {
                "name": name,
                "length": len(dimension),
                "unlimited": bool(dimension.isunlimited()),
            }
            for name, dimension in dataset.dimensions.items()
        ]
        variables = []
        horizontal = {
            "west_east", "west_east_stag", "south_north",
            "south_north_stag", "bdy_width",
        }
        for name, variable in dataset.variables.items():
            filters = variable.filters()
            fill_value = (
                _json_value(variable.getncattr("_FillValue"))
                if "_FillValue" in variable.ncattrs() else None
            )
            spec = {
                "name": name,
                "dtype": str(variable.dtype),
                "dimensions": list(variable.dimensions),
                "attributes": {
                    attr: _json_value(variable.getncattr(attr))
                    for attr in variable.ncattrs() if attr != "_FillValue"
                },
                "attribute_types": {
                    attr: _attribute_type(variable.getncattr(attr))
                    for attr in variable.ncattrs() if attr != "_FillValue"
                },
                "compression": {
                    "zlib": bool(filters.get("zlib", False)),
                    "shuffle": bool(filters.get("shuffle", False)),
                    "complevel": int(filters.get("complevel", 0)),
                    "fletcher32": bool(filters.get("fletcher32", False)),
                },
                "endian": variable.endian(),
                "chunking": (
                    list(variable.chunking())
                    if isinstance(variable.chunking(), (tuple, list))
                    else variable.chunking()
                ),
                "has_fill_value": "_FillValue" in variable.ncattrs(),
                "fill_value": fill_value,
            }
            if not horizontal.intersection(variable.dimensions):
                spec["prototype_value"] = _json_value(variable[:])
            variables.append(spec)

        return {
            "schema": "gpuwm-wrf-io-contract-v1",
            "source_name": path.name,
            "format": dataset.data_model,
            "disk_format": dataset.disk_format,
            "dimensions": dimensions,
            "global_attributes": {
                name: _json_value(dataset.getncattr(name))
                for name in dataset.ncattrs()
            },
            "global_attribute_types": {
                name: _attribute_type(dataset.getncattr(name))
                for name in dataset.ncattrs()
            },
            "variables": variables,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wrfinput", type=Path)
    parser.add_argument("wrfbdy", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema": "gpuwm-wrf-direct-contract-bundle-v1",
        "wrfinput": _spec(args.wrfinput),
        "wrfbdy": _spec(args.wrfbdy),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
