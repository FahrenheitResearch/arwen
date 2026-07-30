#!/usr/bin/env python3
"""Build and seal native static fields for a validated HRRR target domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from gpuwm.native_wrf_contract import native_geometry_contract
from gpuwm.static.build import GeogSelection, build_static
from gpuwm.static.lambert import LambertGrid
from gpuwm.ingest.hrrr_target import (
    HrrrTargetDomain,
    load_hrrr_target_domain,
    required_hrrr_source_window,
)


MASS_SHAPE = (500, 500)
DX_M = 999.8071015811862
REF_LAT = 35.5028506728143
REF_LON = -98.0021669285660
TRUELAT = 38.5
STAND_LON = -97.5


def benchmark_grid(target: HrrrTargetDomain | None = None) -> LambertGrid:
    """Return a target grid, retaining the original benchmark by default."""

    target = target or HrrrTargetDomain.legacy_500x500()
    return target.grid()


def native_static_geometry(
        target: HrrrTargetDomain,
        grid: LambertGrid | None = None,
) -> dict[str, object]:
    """The geometry document sealed into the HRRR static receipt.

    This goes through the shared contract rather than restating it, because
    ``tools/write_hrrr_native_geometry_receipt.py`` compares the sealed
    document to ``native_geometry_contract`` for exact equality.  v1.0.0 had
    the two written out independently and they drifted by one key
    (``map_proj``), which failed every new HRRR area.
    """

    grid = target.grid() if grid is None else grid
    return native_geometry_contract(grid, target.contract_cfg())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    host = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    descriptor = json.dumps(
        [str(host.dtype), list(host.shape)], separators=(",", ":"))
    digest.update(descriptor.encode("ascii"))
    digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def validate_static(
    fields: dict[str, np.ndarray],
    target: HrrrTargetDomain | None = None,
) -> dict[str, object]:
    target = target or HrrrTargetDomain.legacy_500x500()
    mass_shape = (target.ny, target.nx)
    required = {
        "HGT_M", "LANDMASK", "LU_INDEX", "SCT_DOM", "SOILTEMP",
        "SNOALB", "GREENFRAC", "LAI12M", "MAPFAC_M", "MAPFAC_U",
        "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA",
    }
    missing = sorted(required - fields.keys())
    if missing:
        raise KeyError(f"native static build omitted fields: {missing}")
    for name, value in fields.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"native static field {name} is non-finite")
    for name in ("HGT_M", "LANDMASK", "LU_INDEX", "SCT_DOM", "SOILTEMP",
                 "SNOALB", "MAPFAC_M", "F", "E", "SINALPHA",
                 "COSALPHA"):
        if fields[name].shape != mass_shape:
            raise ValueError(
                f"native static field {name} shape {fields[name].shape} "
                f"does not equal {mass_shape}")
    if fields["MAPFAC_U"].shape != (target.ny, target.nx + 1):
        raise ValueError("MAPFAC_U stagger shape mismatch")
    if fields["MAPFAC_V"].shape != (target.ny + 1, target.nx):
        raise ValueError("MAPFAC_V stagger shape mismatch")
    if not np.isin(fields["LANDMASK"], (0.0, 1.0)).all():
        raise ValueError("LANDMASK is not binary")
    land = fields["LANDMASK"] > 0.5
    if np.any(land) and not np.any(fields["HGT_M"][land] != 0.0):
        raise ValueError(
            "native static HGT_M is identically zero over every land cell; "
            "the selected WPS GEOG terrain is likely missing or outside its "
            "staged footprint")
    if fields["LU_INDEX"].min() < 1 or fields["LU_INDEX"].max() > 21:
        raise ValueError("LU_INDEX is outside the MODIS-Noah categories")
    if fields["SCT_DOM"].min() < 1 or fields["SCT_DOM"].max() > 16:
        raise ValueError("SCT_DOM is outside the Noah soil categories")
    rotation_norm_error = np.max(np.abs(
        fields["SINALPHA"] ** 2 + fields["COSALPHA"] ** 2 - 1.0))
    if rotation_norm_error > 1.0e-12:
        raise ValueError(
            f"wind-rotation unit norm error {rotation_norm_error} is too large")
    return {
        "field_count": len(fields),
        "land_fraction": float(fields["LANDMASK"].mean()),
        "terrain_min_m": float(fields["HGT_M"].min()),
        "terrain_max_m": float(fields["HGT_M"].max()),
        "rotation_norm_max_error": float(rotation_norm_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geog-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--domain-spec", type=Path,
        help=("strict gpuwm-hrrr-target-domain-v1 JSON; omission retains "
              "the sealed 500x500 benchmark geometry"),
    )
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("native static output/receipt already exists")

    total_started = time.perf_counter()
    target = load_hrrr_target_domain(args.domain_spec)
    grid = benchmark_grid(target)
    source_window = required_hrrr_source_window(target)
    selection = GeogSelection.fallback(args.geog_root)
    geog_source_coverage: dict[str, object] = {}
    build_started = time.perf_counter()
    fields = build_static(
        grid, args.geog_root, selection=selection,
        source_coverage_report=geog_source_coverage)
    build_seconds = time.perf_counter() - build_started
    fields.update({
        "MAPFAC_M": grid.mapfac_m(),
        "MAPFAC_U": grid.mapfac_u(),
        "MAPFAC_V": grid.mapfac_v(),
    })
    fields["F"], fields["E"] = grid.coriolis_m()
    fields["SINALPHA"], fields["COSALPHA"] = grid.rotation_m()
    validation = validate_static(fields, target)

    geog_tile_hashes: dict[str, str] = {}
    for evidence in geog_source_coverage.values():
        if not isinstance(evidence, dict):
            raise TypeError("GEOG source-coverage evidence must be a mapping")
        dataset = Path(evidence["dataset"])
        for tile in evidence["required_tiles"]:
            relative = Path(tile["relative_path"])
            if relative.is_absolute() or len(relative.parts) != 1:
                raise ValueError(
                    f"unsafe GEOG source tile path {str(relative)!r}")
            path = dataset / relative
            if not path.is_file() or path.stat().st_size != tile["bytes"]:
                raise FileNotFoundError(
                    f"required GEOG source tile changed during build: {path}")
            digest = sha256_file(path)
            tile["sha256"] = digest
            resolved = str(path.resolve())
            previous = geog_tile_hashes.setdefault(resolved, digest)
            if previous != digest:
                raise AssertionError(
                    f"conflicting GEOG source tile hashes for {resolved}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez(temporary, **{name: np.asarray(value, dtype=np.float64)
                          for name, value in sorted(fields.items())})
    os.replace(temporary, args.output)
    array_hashes = {
        name: array_sha256(value) for name, value in sorted(fields.items())}
    index_hashes = {}
    for name in (
            "terrain", "landuse", "soil_top", "soil_bottom", "greenfrac",
            "lai", "albedo", "snow_albedo", "soil_temperature"):
        index = selection.path(name) / "index"
        index_hashes[str(index.resolve())] = sha256_file(index)
    legacy_mode = args.domain_spec is None
    receipt = {
        "schema": (
            "gpuwm-native-hrrr-static-500x500-v1" if legacy_mode
            else "gpuwm-native-hrrr-static-v2"),
        "status": "PASS",
        "method": "gpuwm.static.build.build_static; no WPS/geogrid executable",
        "geometry": native_static_geometry(target, grid),
        "target_domain": target.to_payload(),
        "target_domain_sha256": target.identity_sha256(),
        "hrrr_source_coverage": source_window.to_dict(),
        "geog_root": str(args.geog_root.resolve()),
        "geog_selection": {
            name: str(selection.path(name).resolve()) for name in (
                "terrain", "landuse", "soil_top", "soil_bottom",
                "greenfrac", "lai", "albedo", "snow_albedo",
                "soil_temperature")},
        "geog_index_sha256": index_hashes,
        "geog_source_coverage": geog_source_coverage,
        "geog_tile_sha256": dict(sorted(geog_tile_hashes.items())),
        "validation": validation,
        "array_sha256": array_hashes,
        "cache": {
            "path": (
                str(args.output.resolve()) if legacy_mode else args.output.name),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "timing_seconds": {
            "cold_static_build": build_seconds,
            "cold_build_validate_and_cache": time.perf_counter() - total_started,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_receipt, args.receipt)
    print(json.dumps({
        "status": "PASS", "build_seconds": build_seconds,
        "cache_bytes": args.output.stat().st_size,
        "cache_sha256": receipt["cache"]["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
