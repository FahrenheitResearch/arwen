#!/usr/bin/env python3
"""Build and seal native static fields for a validated HRRR target domain."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from gpuwm.native_wrf_contract import (
    native_geometry_contract,
    require_land_terrain,
)
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
    document to ``native_geometry_contract`` key for key.  v1.0.0 had
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
    require_land_terrain(fields["HGT_M"], fields["LANDMASK"])
    # 21 categories, or the urban legend's 61 (categories 51-61 kept for
    # the urban table) when the static carries it, as the forecast's own
    # static contract reads them: with 21 here, every urban run whose land
    # cover kept a Local Climate Zone or NLCD intensity class was refused
    # at the static build.
    from gpuwm.native_wrf_contract import _landuse_category_count
    categories = (_landuse_category_count(fields["LANDUSEF"])
                  if "LANDUSEF" in fields else 21)
    if (fields["LU_INDEX"].min() < 1
            or fields["LU_INDEX"].max() > categories):
        raise ValueError(
            f"LU_INDEX is outside the MODIS-Noah categories 1..{categories}")
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
    parser.add_argument("--geog-root", type=Path)
    parser.add_argument("--static-cache", type=Path)
    parser.add_argument("--static-receipt", type=Path)
    parser.add_argument("--lake-depth", action="store_true",
                        help="build WPS lake-depth geography for the CLM lake model")
    configuration = parser.add_mutually_exclusive_group()
    configuration.add_argument("--experiment-config", type=Path)
    configuration.add_argument(
        "--static-highres", type=json.loads, metavar="JSON",
        help=("the high-resolution carrier a namelist-only preparation "
              "resolved, as the JSON identity a seal records "
              "(static_highres_identity); it has no configuration file to "
              "name"))
    parser.add_argument("--case-date", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--domain-spec", type=Path,
        help=("strict gpuwm-hrrr-target-domain-v1 JSON; omission retains "
              "the sealed 500x500 benchmark geometry"),
    )
    args = parser.parse_args()
    need_lake_depth = args.lake_depth
    experiment_tables = None
    if args.experiment_config is not None:
        import tomllib
        from gpuwm.config_authority import read_config_authority
        experiment_tables = tomllib.loads(read_config_authority(
            args.experiment_config).payload.decode("utf-8"))
        if "experiment" in experiment_tables:
            from gpuwm.experiment import load_experiment
            run = load_experiment(args.experiment_config).root.run
            need_lake_depth |= bool(run.sf_lake_physics == 1 and run.use_lakedepth == 1)
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("native static output/receipt already exists")

    total_started = time.perf_counter()
    target = load_hrrr_target_domain(args.domain_spec)
    grid = benchmark_grid(target)
    from gpuwm.static.highres_production import (
        load_static_highres, apply_prepared_highres, overlay_active,
        parse_sealed_static_highres)
    # The hrrr source metadata selects a static-source row by default.  A
    # configuration that names no source takes it only on the row's own
    # cone; a declared source is on the carrier already and still refuses
    # a mismatch by name (gpuwm.static.source_defaults).
    from gpuwm.static.source_defaults import (
        defaulted_source_fallback, source_static_defaults)
    defaults = source_static_defaults("hrrr")
    if args.static_highres is not None:
        highres = parse_sealed_static_highres(
            args.static_highres, source="--static-highres",
            base_dir=Path.cwd())
    else:
        highres = load_static_highres(args.experiment_config)
        from dataclasses import replace
        from gpuwm.static.highres_production import parse_static_table
        if ((experiment_tables is None or "experiment" in experiment_tables)
                and (highres is None or highres.static_source is None)
                and defaulted_source_fallback(defaults["source"], grid) is None):
            source_carrier = parse_static_table(
                defaults, source="source metadata", base_dir=Path.cwd())
            if highres is None:
                highres = source_carrier
            else:
                highres = replace(highres,
                                  static_source=source_carrier.static_source)
    source_fallback = (None if getattr(highres, "static_source", None) is not None
                       else defaulted_source_fallback(defaults["source"], grid))
    if source_fallback is not None:
        print(f"static source {source_fallback['id']}: "
              f"{source_fallback['reason']} (projection "
              f"{source_fallback['projection_mismatch']})", flush=True)
    from gpuwm.static.external_source import static_source_for, static_source_receipt
    source_setting = static_source_for(highres)
    from gpuwm.static.external_source import sampling_window
    exact_window = (None if source_setting is None
                    else sampling_window(source_setting.row, grid))
    if exact_window is not None:
        source_coverage = {"scope": "static-source-grid", "id": source_setting.id,
                           "window": {"i0": exact_window[0], "j0": exact_window[1],
                                      "ni": target.nx, "nj": target.ny}}
    else:
        source_coverage = required_hrrr_source_window(target).to_dict()
    # d01's terrain smoothing ([[domain]] static on the root), built here
    # as the WPS_GEOG roots of the other routes build it, and attested in
    # the receipt the root seam (require_root_smoothing) reads.  A default
    # setting builds through the call this tool always made.
    from dataclasses import replace
    from gpuwm.static.terrain_smoothing import smoothing_for
    smoothing = smoothing_for(highres, 1)
    smoothing_attestation = (
        None if smoothing.is_default
        else {"terrain_smoothing": {"d01": smoothing.echo()}})
    if overlay_active(highres, grid) and args.case_date is None:
        raise ValueError("high-resolution static preparation needs --case-date YYYY-MM-DD")
    if (args.static_cache is None) != (args.static_receipt is None):
        raise ValueError("static-cache and static-receipt must be supplied together")
    prior = None
    build_started = time.perf_counter()
    if args.static_cache is not None:
        from gpuwm.hrrr_native_static import verify_hrrr_native_static
        fields, prior = verify_hrrr_native_static(
            args.static_cache, args.static_receipt, target)
        if need_lake_depth and "LAKE_DEPTH" not in fields:
            raise ValueError(
                "the static cache has no LAKE_DEPTH required by the selected "
                "lake model; rebuild from --geog-root to include bathymetry")
        # A sealed static keeps the terrain it was built with.  One built
        # under another d01 smoothing than this preparation asks for would
        # integrate terrain the configuration did not ask for (a default
        # request is not checked by the root seam, which passes it).
        attested = (prior.get("terrain_smoothing") or {}).get("d01")
        requested = None if smoothing.is_default else smoothing.echo()
        if attested is not None and attested != requested:
            raise ValueError(
                f"the static cache {args.static_cache} was built with d01 "
                f"terrain smoothing {attested}, and this preparation asks "
                f"for {smoothing.label()}; reusing it would integrate "
                "terrain the configuration did not ask for. Build the "
                "static from --geog-root instead")
        selection = GeogSelection(
            root=Path(prior["geog_root"]), resolution_tokens=(),
            **prior["geog_selection"])
        if "LAKE_DEPTH" in fields:
            selection = replace(selection, lake_depth=True)
        geog_source_coverage = prior["geog_source_coverage"]
    else:
        if args.geog_root is None:
            raise ValueError("provide geog-root or a verified static-cache/static-receipt pair")
        selection = GeogSelection.fallback(args.geog_root)
        if need_lake_depth:
            selection = replace(selection, lake_depth=True)
        if smoothing_attestation is not None:
            selection = replace(selection, terrain_smoothing=smoothing)
        selection = replace(selection, static_source=static_source_for(highres))
        geog_source_coverage: dict[str, object] = {}
        fields = build_static(
            grid, args.geog_root, selection=selection,
            source_coverage_report=geog_source_coverage)
    fields, overlay_binding = apply_prepared_highres(
        fields, grid, config=highres, domain_id=1, case_date=args.case_date,
        landuse_attrs=(selection.landuse_global_attrs()
                       if overlay_active(highres, grid) else None),
        baseline_receipt=(prior if args.static_cache is not None else {
            **(smoothing_attestation or {}),
            **({"static_source": static_source_receipt(highres)}
               if geog_source_coverage.get("static_source", {}).get("status") == "APPLIED"
               else {}),
        }))
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
    for name, evidence in geog_source_coverage.items():
        if name == "static_source":
            continue
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
        if name in geog_source_coverage:
            index = selection.path(name) / "index"
            index_hashes[str(index.resolve())] = sha256_file(index)
    if "lake_depth" in geog_source_coverage:
        index = Path(geog_source_coverage["lake_depth"]["dataset"]) / "index"
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
        "hrrr_source_coverage": source_coverage,
        "geog_root": str(selection.root.resolve()),
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
    if overlay_active(highres, grid):
        receipt["highres"] = overlay_binding["highres"]
    if geog_source_coverage.get("static_source", {}).get("status") == "APPLIED":
        receipt["static_source"] = static_source_receipt(highres)
    if source_fallback is not None:
        # The record of a defaulted source set aside for its projection
        # (gpuwm.static.source_defaults.defaulted_source_fallback).
        receipt["static_source_fallback"] = {"d01": source_fallback}
    if args.static_cache is None and smoothing_attestation is not None:
        receipt.update(smoothing_attestation)
    elif isinstance(prior, dict) and "terrain_smoothing" in prior:
        # A rebuild from a sealed static keeps the terrain it was handed,
        # and with it the attestation that terrain was built under.
        receipt["terrain_smoothing"] = prior["terrain_smoothing"]
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
