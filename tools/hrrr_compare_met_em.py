"""Compare native HRRR horizontal interpolation with a WPS met_em oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gpuwm.ingest.hrrr import (
    interpolate_hrrr_to_lambert,
    load_hrrr_native_window,
)
from gpuwm.static.lambert import grids_from_wps_namelist


def _stats(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {candidate.shape} != reference {reference.shape}")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("comparison inputs must be finite")
    delta = candidate - reference
    scale = max(float(np.sqrt(np.mean(reference * reference))), 1.0e-30)
    return {
        "shape": list(candidate.shape),
        "candidate_minimum": float(np.min(candidate)),
        "candidate_maximum": float(np.max(candidate)),
        "candidate_negative_count": int(np.count_nonzero(candidate < 0.0)),
        "reference_minimum": float(np.min(reference)),
        "reference_maximum": float(np.max(reference)),
        "reference_negative_count": int(np.count_nonzero(reference < 0.0)),
        "maximum_absolute": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "bias": float(np.mean(delta)),
        "reference_rms": scale,
        "normalized_rmse": float(np.sqrt(np.mean(delta * delta)) / scale),
        "exact_fraction": float(np.mean(candidate == reference)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--namelist-wps", required=True)
    parser.add_argument("--met-em", required=True)
    parser.add_argument("--forecast-hour", type=int, required=True)
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import cupy as cp
    from netCDF4 import Dataset

    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    snapshot = load_hrrr_native_window(
        args.bridge, args.forecast_hour,
        expected_manifest_sha256=args.manifest_sha256)
    t1 = time.perf_counter()
    grids = grids_from_wps_namelist(args.namelist_wps)
    if args.domain < 1 or args.domain > len(grids):
        raise ValueError(f"domain must be in 1..{len(grids)}")
    grid = grids[args.domain - 1]
    with Dataset(args.met_em) as dataset:
        target_landmask = np.asarray(dataset.variables["LANDMASK"][0])
    soil_mapping = {}
    horizontal = interpolate_hrrr_to_lambert(
        snapshot, grid, target_landmask=target_landmask,
        soil_mapping_report=soil_mapping)
    cp.cuda.Device().synchronize()
    t2 = time.perf_counter()

    comparisons = {
        "PRES": ("PRES", slice(1, None)),
        "HGT": ("GHT", slice(1, None)),
        "TT": ("TT", slice(1, None)),
        "SPFH": ("SPECHUMD", slice(1, None)),
        "QC": ("QC", slice(1, None)),
        "QR": ("QR", slice(1, None)),
        "QI": ("QI", slice(1, None)),
        "QS": ("QS", slice(1, None)),
        "QG": ("QG", slice(1, None)),
        "UU": ("UU", slice(1, None)),
        "VV": ("VV", slice(1, None)),
        "PSFC": ("PSFC", None),
        "SOILHGT": ("SOILHGT", None),
        "SKINTEMP": ("SKINTEMP", None),
        "SNOW": ("SNOW", None),
        "SNOWH": ("SNOWH", None),
        "LANDSEA": ("LANDMASK", None),
        "SOURCE_LANDSEA": ("LANDSEA", None),
        "XICE": ("SEAICE", None),
    }
    report = {
        "bridge": str(Path(args.bridge).resolve()),
        "manifest_sha256": args.manifest_sha256,
        "met_em": str(Path(args.met_em).resolve()),
        "forecast_hour": args.forecast_hour,
        "domain": args.domain,
        "timing_seconds": {
            "hash_verify_and_map": t1 - t0,
            "gpu_horizontal_and_sync": t2 - t1,
        },
        "fields": {},
        "soil_mapping": soil_mapping,
        "wps_soil_oracle": {
            "status": "comparison oracle, not native policy",
            "known_operator": "WPS sixteen_pt overlapping parabolic",
            "native_repair": (
                "masked non-negative convex bilinear plus bounded nearest-"
                "valid fallback; WPS negative/above-one values are rejected"),
            "depths_m": [0.0, 0.01, 0.04, 0.10, 0.30, 0.60,
                         1.0, 1.6, 3.0],
        },
    }
    with Dataset(args.met_em) as dataset:
        for candidate_name, (reference_name, levels) in comparisons.items():
            candidate = cp.asnumpy(horizontal.fields[candidate_name])
            reference = np.asarray(dataset.variables[reference_name][0])
            if levels is not None:
                # WPS met_em stores hybrid levels top-to-bottom after its
                # surface pseudo-level; the native bridge retains HRRR's
                # GRIB level-number order (bottom-to-top).
                reference = reference[levels][::-1]
            report["fields"][candidate_name] = _stats(candidate, reference)
        for candidate_name, reference_name in (
                ("T2", "TT"), ("Q2", "SPECHUMD"),
                ("U10", "UU"), ("V10", "VV")):
            candidate = cp.asnumpy(horizontal.fields[candidate_name])
            reference = np.asarray(dataset.variables[reference_name][0, 0])
            report["fields"][candidate_name] = _stats(candidate, reference)
        soil_names = ["000", "001", "004", "010", "030", "060",
                      "100", "160", "300"]
        for candidate_name, prefix in (("SOILT", "SOILT"),
                                       ("SOILW", "SOILM")):
            candidate = cp.asnumpy(horizontal.fields[candidate_name])
            reference = np.stack([
                np.asarray(dataset.variables[f"{prefix}{depth}"][0])
                for depth in soil_names
            ])
            report["fields"][candidate_name] = _stats(candidate, reference)
            report["wps_soil_oracle"][candidate_name] = {
                "minimum_by_depth": np.min(reference, axis=(1, 2)).tolist(),
                "maximum_by_depth": np.max(reference, axis=(1, 2)).tolist(),
                "negative_count_by_depth": np.sum(
                    reference < 0.0, axis=(1, 2)).astype(int).tolist(),
                "above_one_count_by_depth": np.sum(
                    reference > 1.0, axis=(1, 2)).astype(int).tolist(),
                "finite": bool(np.isfinite(reference).all()),
            }
    report["gpu_memory_bytes"] = {
        "pool_used": int(cp.get_default_memory_pool().used_bytes()),
        "pool_total": int(cp.get_default_memory_pool().total_bytes()),
        "device_free": int(cp.cuda.runtime.memGetInfo()[0]),
        "device_total": int(cp.cuda.runtime.memGetInfo()[1]),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["timing_seconds"], sort_keys=True))
    for name, stats in report["fields"].items():
        print(f"{name}\trmse={stats['rmse']:.9g}\tmax={stats['maximum_absolute']:.9g}")


if __name__ == "__main__":
    main()
