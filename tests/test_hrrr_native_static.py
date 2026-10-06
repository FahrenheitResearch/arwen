"""Strict evidence checks for reusable native HRRR static inputs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gpuwm.hrrr_native_static import (
    _array_sha256,
    sha256_file,
    verify_hrrr_native_static,
)
from gpuwm.ingest.hrrr_target import (
    HrrrTargetDomain,
    required_hrrr_source_window,
)


def _target() -> HrrrTargetDomain:
    return HrrrTargetDomain(
        name="strict-static-test", map_proj="lambert",
        nx=14, ny=13, nz=9, dx_m=3000.0, dy_m=3000.0,
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0,
        truelat2=60.0, stand_lon=-97.0, time_step_seconds=15)


def _arrays(target, *, bad_monthly_shape=False):
    mass = (target.ny, target.nx)
    fields = {
        name: np.zeros(mass, dtype=np.float64)
        for name in (
            "COSALPHA", "E", "F", "HGT_M", "LANDMASK", "MAPFAC_M",
            "SINALPHA", "SNOALB", "SOILTEMP", "TMN")
    }
    fields["COSALPHA"].fill(1.0)
    fields["LU_INDEX"] = np.ones(mass, dtype=np.float64)
    fields["SCT_DOM"] = np.ones(mass, dtype=np.float64)
    fields["SCB_DOM"] = np.ones(mass, dtype=np.float64)
    monthly = ((1, 12, *mass) if bad_monthly_shape else (12, *mass))
    for name in ("ALBEDO12M", "GREENFRAC", "LAI12M"):
        fields[name] = np.zeros(monthly, dtype=np.float64)
    fields["LANDUSEF"] = np.zeros((21, *mass), dtype=np.float64)
    fields["SOILCBOT"] = np.zeros((16, *mass), dtype=np.float64)
    fields["SOILCTOP"] = np.zeros((16, *mass), dtype=np.float64)
    fields["MAPFAC_U"] = np.ones(
        (target.ny, target.nx + 1), dtype=np.float64)
    fields["MAPFAC_V"] = np.ones(
        (target.ny + 1, target.nx), dtype=np.float64)
    return fields


def _fixture(tmp_path, *, bad_monthly_shape=False):
    target = _target()
    fields = _arrays(target, bad_monthly_shape=bad_monthly_shape)
    cache = tmp_path / "native-static.npz"
    np.savez(cache, **fields)
    dataset = tmp_path / "geog-source"
    dataset.mkdir()
    index = dataset / "index"
    index.write_text(
        "type=continuous\nprojection=regular_ll\ndx=360\ndy=180\n"
        "known_x=1\nknown_y=1\nknown_lat=0\nknown_lon=0\n"
        "wordsize=1\ntile_x=1\ntile_y=1\ntile_z=1\n",
        encoding="ascii",
    )
    tile = dataset / "00001-00001.00001-00001"
    tile.write_bytes(b"exact-source-tile")
    tile_hash = sha256_file(tile)
    coverage_fields = (
        "terrain", "landuse", "soil_top", "soil_bottom", "greenfrac",
        "lai", "albedo", "snow_albedo", "soil_temperature",
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema": "gpuwm-native-hrrr-static-v2",
        "status": "PASS",
        "target_domain": target.to_payload(),
        "target_domain_sha256": target.identity_sha256(),
        "hrrr_source_coverage": required_hrrr_source_window(target).to_dict(),
        "cache": {
            "path": cache.name,
            "bytes": cache.stat().st_size,
            "sha256": sha256_file(cache),
        },
        "array_sha256": {
            name: _array_sha256(value)
            for name, value in sorted(fields.items())
        },
        "geog_index_sha256": {str(index): sha256_file(index)},
        "geog_source_coverage": {
            field: {
                "schema": "gpuwm-geog-source-coverage-v1",
                "status": "PASS",
                "field": field,
                "dataset": str(dataset.resolve()),
                "declared_sparse": False,
                "source_geometry": {
                    "nx_global": 1,
                    "ny_global": 1,
                    "wraps_x": True,
                    "extent_basis": "regular_ll_complete",
                    "tile_inventory_bounds": [1, 1, 1, 1],
                },
                "source_window": {
                    "x_start": 1, "x_end": 1,
                    "y_start": 1, "y_end": 1,
                },
                "required_cells": 1,
                "covered_cells": 1,
                "missing_tile_cells": 0,
                "outside_extent_cells": 0,
                "coverage_fraction": 1.0,
                "required_tile_count": 1,
                "required_tiles": [{
                    "origin": [1, 1],
                    "relative_path": tile.name,
                    "bytes": tile.stat().st_size,
                    "sha256": tile_hash,
                }],
            }
            for field in coverage_fields
        },
        "geog_tile_sha256": {str(tile.resolve()): tile_hash},
    }), encoding="utf-8")
    return target, cache, receipt


def test_strict_static_verifier_accepts_exact_complete_inventory(tmp_path):
    target, cache, receipt = _fixture(tmp_path)
    fields, evidence = verify_hrrr_native_static(cache, receipt, target)
    assert set(fields) == set(evidence["array_sha256"])
    assert all(value.dtype == np.float64 for value in fields.values())


def test_strict_static_verifier_rejects_extra_leading_month_axis(tmp_path):
    target, cache, receipt = _fixture(tmp_path, bad_monthly_shape=True)
    with pytest.raises(ValueError, match="ALBEDO12M shape mismatch"):
        verify_hrrr_native_static(cache, receipt, target)


def test_strict_static_verifier_rejects_legacy_v2_without_dense_coverage(
        tmp_path):
    target, cache, receipt = _fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload.pop("geog_source_coverage")
    payload.pop("geog_tile_sha256")
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lacks mandatory dense GEOG"):
        verify_hrrr_native_static(cache, receipt, target)


def test_strict_static_verifier_binds_required_geog_tiles(tmp_path):
    target, cache, receipt = _fixture(tmp_path)
    dataset = tmp_path / "geog-source"
    tile = dataset / "00001-00001.00001-00001"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    verify_hrrr_native_static(cache, receipt, target)

    edited = json.loads(json.dumps(payload))
    edited["geog_source_coverage"]["terrain"]["source_window"]["x_end"] = 2
    receipt.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="source coverage.*changed"):
        verify_hrrr_native_static(cache, receipt, target)

    receipt.write_text(json.dumps(payload), encoding="utf-8")
    tile.write_bytes(b"edited-source-tile")
    with pytest.raises(ValueError, match="GEOG source tile changed"):
        verify_hrrr_native_static(cache, receipt, target)


# ---- recorded coverage: one check for every reader ---------------------------
#
# Since 6a69b356f the HRRR route's static builder applies a pinned static
# source and records that source's exact grid window (scope
# "static-source-grid") instead of the HRRR crop.  verify_hrrr_native_static
# learned that; the single-domain prepare's own reader
# (tools/hrrr_single_domain_benchmark._load_static) did not, so every
# `gpuwm go` HRRR prepare inside the pinned source's grid stopped with
# "native static HRRR source-coverage receipt mismatch".  Both now read
# require_static_coverage.

def _static_source_receipt(target, *, window=None):
    window = window or {"i0": 804, "j0": 335, "ni": target.nx, "nj": target.ny}
    return {
        "hrrr_source_coverage": {"scope": "static-source-grid",
                                 "id": "hrrr-conus-v4", "window": dict(window)},
        "geog_source_coverage": {"static_source": {
            "status": "APPLIED", "id": "hrrr-conus-v4", "window": dict(window)}},
    }


def test_an_applied_static_source_carries_its_own_window():
    from gpuwm.hrrr_native_static import require_static_coverage
    target = _target()
    require_static_coverage(_static_source_receipt(target), target, prefix="t")


def test_an_applied_static_source_with_another_window_is_refused():
    from gpuwm.hrrr_native_static import require_static_coverage
    target = _target()
    receipt = _static_source_receipt(target)
    receipt["hrrr_source_coverage"]["window"]["i0"] += 1
    with pytest.raises(ValueError, match="t static-source coverage mismatch"):
        require_static_coverage(receipt, target, prefix="t")


def test_without_a_static_source_the_hrrr_crop_is_required():
    from gpuwm.hrrr_native_static import require_static_coverage
    target = _target()
    require_static_coverage(
        {"hrrr_source_coverage": required_hrrr_source_window(target).to_dict()},
        target, prefix="t")
    with pytest.raises(ValueError, match="t source-coverage mismatch"):
        require_static_coverage(
            {"hrrr_source_coverage": _static_source_receipt(
                target)["hrrr_source_coverage"]}, target, prefix="t")


def test_the_single_domain_prepare_reads_the_shared_coverage_check(
        tmp_path, monkeypatch):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    import hrrr_single_domain_benchmark as bench
    target, cache, receipt = _fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    applied = _static_source_receipt(target)
    payload["hrrr_source_coverage"] = applied["hrrr_source_coverage"]
    payload["geog_source_coverage"]["static_source"] = (
        applied["geog_source_coverage"]["static_source"])
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    calls = []
    import gpuwm.hrrr_native_static as hns
    real = hns.require_static_coverage

    def spy(receipt_doc, tgt, *, prefix):
        calls.append(prefix)
        return real(receipt_doc, tgt, prefix=prefix)

    monkeypatch.setattr(hns, "require_static_coverage", spy)
    monkeypatch.setattr(hns, "verify_geog_source_evidence", lambda doc: None)
    try:
        # The fixture carries no geometry block, so the load stops after
        # the coverage check; it must not stop AT it.
        bench._load_static(cache, receipt, target)
    except (ValueError, KeyError) as exc:
        assert "coverage" not in str(exc), exc
    assert calls == ["native static HRRR receipt"]
    payload["hrrr_source_coverage"]["window"]["j0"] += 1
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="static-source coverage mismatch"):
        bench._load_static(cache, receipt, target)
