"""Verified native HRRR static inputs for hierarchy initialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpuwm.ingest.hrrr_target import (
    HrrrTargetDomain,
    required_hrrr_source_window,
)
from gpuwm.static.build import GeogSelection
from gpuwm.static.geog import GeogDataset


_GEOG_FIELDS = (
    "terrain", "landuse", "soil_top", "soil_bottom", "greenfrac",
    "lai", "albedo", "snow_albedo", "soil_temperature",
)

_MASS_2D_FIELDS = {
    "COSALPHA", "E", "F", "HGT_M", "LANDMASK", "LU_INDEX",
    "MAPFAC_M", "SCB_DOM", "SCT_DOM", "SINALPHA", "SNOALB",
    "SOILTEMP", "TMN",
}
_MONTHLY_FIELDS = {"ALBEDO12M", "GREENFRAC", "LAI12M"}
_CATEGORY_FIELDS = {
    "LANDUSEF": 21,
    "SOILCBOT": 16,
    "SOILCTOP": 16,
}
_STATIC_ARRAY_FIELDS = (
    _MASS_2D_FIELDS | _MONTHLY_FIELDS | set(_CATEGORY_FIELDS)
    | {"MAPFAC_U", "MAPFAC_V"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    host = np.ascontiguousarray(value)
    descriptor = json.dumps(
        [str(host.dtype), list(host.shape)], separators=(",", ":"))
    digest = hashlib.sha256()
    digest.update(descriptor.encode("ascii"))
    digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def verify_geog_source_evidence(receipt: dict[str, object]) -> None:
    """Verify the optional dense-window GEOG tile binding on new receipts.

    Production v2 receipts must bind coverage and hashes together; a legacy,
    partially edited, or partially staged inventory fails closed.
    """

    coverage = receipt.get("geog_source_coverage")
    tile_hashes = receipt.get("geog_tile_sha256")
    if coverage is None and tile_hashes is None:
        raise ValueError(
            "root static receipt lacks mandatory dense GEOG source "
            "coverage evidence")
    if not isinstance(coverage, dict) or not isinstance(tile_hashes, dict):
        raise ValueError(
            "root static receipt must bind GEOG source coverage and tile "
            "hashes together")
    if set(coverage) != set(_GEOG_FIELDS):
        raise ValueError(
            "root static GEOG source-coverage fields mismatch: expected "
            f"{sorted(_GEOG_FIELDS)}, got {sorted(coverage)}")

    observed_paths: set[str] = set()
    for field in _GEOG_FIELDS:
        evidence = coverage[field]
        if not isinstance(evidence, dict):
            raise ValueError(
                f"root static GEOG coverage for {field!r} is not a mapping")
        if (evidence.get("schema") != "gpuwm-geog-source-coverage-v1"
                or evidence.get("status") != "PASS"
                or evidence.get("field") != field):
            raise ValueError(
                f"root static GEOG coverage for {field!r} is not PASS")
        required = evidence.get("required_cells")
        covered = evidence.get("covered_cells")
        if (not isinstance(required, int) or isinstance(required, bool)
                or required <= 0 or covered != required
                or evidence.get("missing_tile_cells") != 0
                or evidence.get("outside_extent_cells") != 0
                or evidence.get("coverage_fraction") != 1.0):
            raise ValueError(
                f"root static GEOG coverage for {field!r} is incomplete")
        dataset = Path(str(evidence.get("dataset", "")))
        source_window = evidence.get("source_window")
        expected_window_keys = {"x_start", "x_end", "y_start", "y_end"}
        if (not isinstance(source_window, dict)
                or set(source_window) != expected_window_keys
                or not all(isinstance(source_window[key], int)
                           and not isinstance(source_window[key], bool)
                           for key in expected_window_keys)):
            raise ValueError(
                f"root static GEOG source window for {field!r} is invalid")
        x0, x1 = source_window["x_start"], source_window["x_end"]
        y0, y1 = source_window["y_start"], source_window["y_end"]
        source = GeogDataset(dataset)
        geometry = {
            "nx_global": int(source.nx_global),
            "ny_global": int(source.ny_global),
            "wraps_x": bool(source.wraps_x),
            "extent_basis": str(source.extent_basis),
            "tile_inventory_bounds": [
                int(value) for value in source.tile_inventory_bounds],
        }
        if (evidence.get("declared_sparse") != source.declared_sparse
                or evidence.get("source_geometry") != geometry):
            raise ValueError(
                f"root static GEOG source geometry for {field!r} changed")
        actual_coverage = source.tile_coverage_mask(x0, x1, y0, y1)
        actual_required = int(actual_coverage.size)
        actual_covered = int(np.count_nonzero(actual_coverage))
        if (not bool(np.all(actual_coverage))
                or required != actual_required
                or covered != actual_covered):
            raise ValueError(
                f"root static GEOG source coverage for {field!r} changed")
        expected_origins = source.required_tile_origins(x0, x1, y0, y1)
        tiles = evidence.get("required_tiles")
        if (not isinstance(tiles, list) or not tiles
                or evidence.get("required_tile_count") != len(tiles)
                or len(tiles) != len(expected_origins)):
            raise ValueError(
                f"root static GEOG tile inventory for {field!r} is invalid")
        for tile, expected_origin in zip(tiles, expected_origins):
            if not isinstance(tile, dict):
                raise ValueError(
                    f"root static GEOG tile for {field!r} is not a mapping")
            if tile.get("origin") != list(expected_origin):
                raise ValueError(
                    f"root static GEOG tile origin for {field!r} changed")
            relative = Path(str(tile.get("relative_path", "")))
            if relative.is_absolute() or len(relative.parts) != 1:
                raise ValueError(
                    f"root static GEOG tile path is unsafe: {relative}")
            path = dataset / relative
            resolved = str(path.resolve())
            expected_hash = tile_hashes.get(resolved)
            if (not path.is_file()
                    or path.stat().st_size != tile.get("bytes")
                    or tile.get("sha256") != expected_hash
                    or sha256_file(path) != expected_hash):
                raise ValueError(
                    f"root static GEOG source tile changed: {path}")
            observed_paths.add(resolved)
    if observed_paths != set(tile_hashes):
        raise ValueError(
            "root static GEOG tile hashes differ from required coverage")


def _target_from_payload(payload) -> HrrrTargetDomain:
    if not isinstance(payload, dict):
        raise ValueError("HRRR static receipt lacks a target-domain document")
    values = dict(payload)
    if values.pop("schema", None) != "gpuwm-hrrr-target-domain-v1":
        raise ValueError("HRRR static receipt target-domain schema mismatch")
    expected = set(HrrrTargetDomain.__dataclass_fields__)
    # The SAME optional set as gpuwm.ingest.hrrr_target.
    # load_hrrr_target_domain, and it must stay the same: to_payload()
    # deliberately omits the rational-clock remainder for a whole-second
    # clock (so every stored integer-clock identity_sha256 survived the
    # 2026-08-05 widening), which means every whole-second static
    # receipt reaches this verifier WITHOUT the two fract keys.  When
    # this copy defaulted only surface_fallback_radius_cells, the
    # writer and the verifier disagreed within one tree and the nested
    # HRRR hierarchy refused every integer-dt preparation (found
    # 2026-08-06 by the init-perturbation demo's 15 s root clock).
    optional_v1_defaults = {
        "surface_fallback_radius_cells": 8,
        "time_step_fract_num": 0,
        "time_step_fract_den": 1,
    }
    missing = expected - set(values)
    if set(values) - expected or not missing <= set(optional_v1_defaults):
        raise ValueError("HRRR static receipt target-domain fields mismatch")
    for name in missing:
        values[name] = optional_v1_defaults[name]
    return HrrrTargetDomain(**values)


def verify_hrrr_native_static(
        cache_path: Path, receipt_path: Path,
        expected_target: HrrrTargetDomain,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Verify and load one rich HRRR static cache and its GEOG evidence."""

    cache_path = Path(cache_path)
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "gpuwm-native-hrrr-static-v2"
            or receipt.get("status") != "PASS"):
        raise ValueError("root static receipt is not a PASS HRRR v2 receipt")
    verify_geog_source_evidence(receipt)
    target = _target_from_payload(receipt.get("target_domain"))
    if target != expected_target:
        raise ValueError("root static receipt target differs from domain spec")
    if receipt.get("target_domain_sha256") != target.identity_sha256():
        raise ValueError("root static receipt target identity mismatch")
    if receipt.get("hrrr_source_coverage") != \
            required_hrrr_source_window(target).to_dict():
        raise ValueError("root static receipt source-coverage mismatch")

    cache = receipt.get("cache")
    if not isinstance(cache, dict):
        raise ValueError("root static receipt lacks a cache binding")
    if (cache.get("bytes") != cache_path.stat().st_size
            or cache.get("sha256") != sha256_file(cache_path)):
        raise ValueError("root static cache differs from its receipt")
    named_path = Path(str(cache.get("path", "")))
    if named_path.is_absolute():
        if named_path.resolve() != cache_path.resolve():
            raise ValueError("root static receipt names another cache")
    elif named_path.name != cache_path.name:
        raise ValueError("root static receipt cache name mismatch")

    with np.load(cache_path, allow_pickle=False) as stored:
        if set(stored.files) != _STATIC_ARRAY_FIELDS:
            raise ValueError(
                "root static cache inventory mismatch: expected "
                f"{sorted(_STATIC_ARRAY_FIELDS)}, got {sorted(stored.files)}")
        fields = {}
        for name in stored.files:
            value = np.asarray(stored[name])
            if value.dtype != np.dtype(np.float64):
                raise ValueError(
                    f"root static field {name} must be exact float64")
            fields[name] = np.ascontiguousarray(value)
    if any(not np.isfinite(value).all() for value in fields.values()):
        raise ValueError("root static cache contains non-finite values")
    mass_shape = (target.ny, target.nx)
    for name in sorted(_MASS_2D_FIELDS):
        if fields[name].shape != mass_shape:
            raise ValueError(f"root static field {name} shape mismatch")
    for name in sorted(_MONTHLY_FIELDS):
        if fields[name].shape != (12, *mass_shape):
            raise ValueError(f"root static field {name} shape mismatch")
    for name, categories in _CATEGORY_FIELDS.items():
        if fields[name].shape != (categories, *mass_shape):
            raise ValueError(f"root static field {name} shape mismatch")
    if fields["MAPFAC_U"].shape != (target.ny, target.nx + 1):
        raise ValueError("root static MAPFAC_U shape mismatch")
    if fields["MAPFAC_V"].shape != (target.ny + 1, target.nx):
        raise ValueError("root static MAPFAC_V shape mismatch")
    if not np.isin(fields["LANDMASK"], (0.0, 1.0)).all():
        raise ValueError("root static LANDMASK is not binary")
    land = fields["LANDMASK"] > 0.5
    if np.any(land) and not np.any(fields["HGT_M"][land] != 0.0):
        raise ValueError(
            "root static HGT_M is identically zero over every land cell")
    if not (fields["LU_INDEX"].min() >= 1.0
            and fields["LU_INDEX"].max() <= 21.0):
        raise ValueError("root static LU_INDEX is outside MODIS categories")
    for name in ("SCT_DOM", "SCB_DOM"):
        if not (fields[name].min() >= 1.0 and fields[name].max() <= 16.0):
            raise ValueError(
                f"root static {name} is outside Noah soil categories")
    observed_arrays = {
        name: _array_sha256(value) for name, value in sorted(fields.items())}
    if observed_arrays != receipt.get("array_sha256"):
        raise ValueError("root static array inventory differs from receipt")

    geog_hashes = receipt.get("geog_index_sha256")
    if not isinstance(geog_hashes, dict) or not geog_hashes:
        raise ValueError("root static receipt lacks GEOG index evidence")
    for raw_path, expected in geog_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"root static GEOG index changed: {path}")
    return fields, receipt


def verified_static_catalog(
        wps_namelist: Path, geog_root: Path, domain_ids,
) -> tuple[object, dict[str, object]]:
    """Build the child GEOG catalog and bind every selected index file."""

    wps_namelist = Path(wps_namelist).resolve()
    geog_root = Path(geog_root).resolve()
    case = SimpleNamespace(wps_namelist=wps_namelist, geog_root=geog_root)
    files = [SimpleNamespace(role="wps_namelist", path=wps_namelist)]
    index_hashes: dict[str, str] = {}
    selections: dict[str, dict[str, str]] = {}
    observed: set[Path] = set()
    for grid_id in domain_ids:
        selection = GeogSelection.from_case_data(case, domain_id=grid_id)
        selected: dict[str, str] = {}
        for name in _GEOG_FIELDS:
            directory = selection.path(name).resolve()
            index = directory / "index"
            if not index.is_file():
                raise FileNotFoundError(index)
            selected[name] = str(directory)
            index_hashes[str(index)] = sha256_file(index)
            if index not in observed:
                observed.add(index)
                files.append(SimpleNamespace(role="geog_index", path=index))
        selections[f"d{int(grid_id):02d}"] = selected
    catalog = SimpleNamespace(files=tuple(files))
    receipt = {
        "schema": "gpuwm-native-hrrr-static-catalog-v1",
        "status": "PASS",
        "wps_namelist": {
            "path": str(wps_namelist),
            "sha256": sha256_file(wps_namelist),
        },
        "geog_root": str(geog_root),
        "selections": selections,
        "geog_index_sha256": dict(sorted(index_hashes.items())),
    }
    return catalog, receipt


__all__ = [
    "sha256_file", "verified_static_catalog", "verify_geog_source_evidence",
    "verify_hrrr_native_static",
]
