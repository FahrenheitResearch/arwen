#!/usr/bin/env python3
"""Emit exact source/mapped d02 HRRR coverage and range diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpuwm.ingest.hrrr import (
    interpolate_hrrr_to_lambert,
    load_hrrr_native_window,
)
from gpuwm.static.lambert import grids_from_wps_namelist
from tools.hrrr_state_proof import _read_static, _strict_json


def _stats(value):
    if hasattr(value, "get"):
        value = value.get()
    array = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(array)
    return {
        "shape": list(array.shape),
        "count": int(array.size),
        "finite_count": int(np.count_nonzero(finite)),
        "nonfinite_count": int(np.count_nonzero(~finite)),
        "negative_count": int(np.count_nonzero(finite & (array < 0.0))),
        "zero_count": int(np.count_nonzero(finite & (array == 0.0))),
        "minimum": float(np.min(array[finite])) if np.any(finite) else None,
        "maximum": float(np.max(array[finite])) if np.any(finite) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--namelist-wps", type=Path, required=True)
    parser.add_argument("--geo-d02", type=Path, required=True)
    args = parser.parse_args()
    snapshot = load_hrrr_native_window(
        args.bridge, 0, expected_manifest_sha256=args.manifest_sha256)
    static, _ = _read_static(args.geo_d02, expected_shape=(300, 300))
    report = {}
    mapped = interpolate_hrrr_to_lambert(
        snapshot, grids_from_wps_namelist(args.namelist_wps)[1],
        target_landmask=static["LANDMASK"],
        soil_mapping_report=report)
    output = {
        "source": {name: _stats(value)
                   for name, value in snapshot.fields.items()},
        "mapped": {name: _stats(value)
                   for name, value in mapped.fields.items()},
        "soil_mapping": report,
    }
    print(json.dumps(_strict_json(output), sort_keys=True))


if __name__ == "__main__":
    main()
