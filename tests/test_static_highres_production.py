"""Focused gates for the production high-resolution static geography lane.

Covers the config surface (unknown keys refuse; absence is the identity),
the footprint-parametric tile enumeration, the cache-hit path, refusal on
synthetic coverage gaps, and the US-interior/coast refusal gates -- all
without touching the network or any real raster.
"""
from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gpuwm.static.highres_fetch import (
    CoverageError,
    FootprintBBox,
    SourceAbsent,
    domain_footprint,
    fetch_file,
    fetch_three_dep_tiles,
    nlcd_year_for,
    three_dep_tile_ids,
)
from gpuwm.static.highres_production import (
    refuse_inert_highres,
    HighresRefusal,
    HighresStaticConfig,
    apply_highres_statics,
    parse_static_table,
)
from gpuwm.static.lambert import LambertGrid

MODIS21_ATTRS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                 "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}


def _us_interior_grid(dx: float = 3000.0, n: int = 41) -> LambertGrid:
    return LambertGrid(
        ref_lat=38.68, ref_lon=-98.15, truelat1=30.0, truelat2=60.0,
        stand_lon=-98.15, dx=dx, dy=dx, e_we=n, e_sn=n)


def _baseline(ny: int, nx: int) -> dict[str, np.ndarray]:
    return {
        "HGT_M": np.full((ny, nx), 500.0),
        "LU_INDEX": np.full((ny, nx), 10.0),
        "LANDMASK": np.ones((ny, nx)),
    }


# ---------------------------------------------------------------------------
# Footprint and tile enumeration
# ---------------------------------------------------------------------------

def test_domain_footprint_covers_halo_extended_corners():
    grid = _us_interior_grid()
    inner = domain_footprint(grid, halo=0, margin_deg=0.0)
    outer = domain_footprint(grid, halo=3)
    assert outer.lat_min < inner.lat_min
    assert outer.lat_max > inner.lat_max
    assert outer.lon_min < inner.lon_min
    assert outer.lon_max > inner.lon_max
    lat, lon = grid.ij_to_latlon(
        np.array([0.5, grid.e_we - 0.5]), np.array([0.5, grid.e_sn - 0.5]))
    assert outer.lat_min < float(np.min(lat)) <= float(np.max(lat)) \
        < outer.lat_max
    assert outer.lon_min < float(np.min(lon)) <= float(np.max(lon)) \
        < outer.lon_max


def test_three_dep_tile_enumeration_from_bbox():
    bbox = FootprintBBox(lat_min=38.05, lat_max=39.31,
                         lon_min=-98.975, lon_max=-97.325)
    assert set(three_dep_tile_ids(bbox)) == {
        "n39w099", "n39w098", "n40w099", "n40w098"}


def test_three_dep_tile_enumeration_single_tile_interior():
    bbox = FootprintBBox(lat_min=37.2, lat_max=37.8,
                         lon_min=-98.9, lon_max=-98.2)
    assert three_dep_tile_ids(bbox) == ("n38w099",)


def test_three_dep_tile_enumeration_refuses_other_quadrants():
    with pytest.raises(CoverageError, match="quadrant"):
        three_dep_tile_ids(FootprintBBox(lat_min=-2.0, lat_max=-1.0,
                                         lon_min=-70.0, lon_max=-69.0))


def test_nlcd_year_nearest_and_anachronism():
    assert nlcd_year_for(date(1974, 4, 3)) == (1985, 11)
    assert nlcd_year_for(date(2021, 5, 15)) == (2021, 0)
    assert nlcd_year_for(date(2030, 1, 1)) == (2024, 6)


# ---------------------------------------------------------------------------
# Fetch: cache hit and synthetic coverage gap
# ---------------------------------------------------------------------------

class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_fetch_file_records_sha_and_hits_cache(tmp_path):
    calls = []

    def urlopen(url, offset):
        calls.append(url)
        return _FakeResponse(b"tile-payload")

    target = tmp_path / "cache" / "artifact.bin"
    first = fetch_file("https://example.invalid/a", target, urlopen=urlopen)
    assert first.cache_hit is False
    assert first.bytes == len(b"tile-payload")
    sidecar = json.loads(
        (target.parent / (target.name + ".sha256.json")).read_text())
    assert sidecar["sha256"] == first.sha256
    assert calls == ["https://example.invalid/a"]

    second = fetch_file("https://example.invalid/a", target, urlopen=urlopen)
    assert second.cache_hit is True
    assert second.sha256 == first.sha256
    assert calls == ["https://example.invalid/a"]  # no second network touch


def test_fetch_three_dep_refuses_naming_missing_tiles(tmp_path):
    def urlopen(url, offset):
        if "n40w099" in url:
            raise SourceAbsent(f"{url} -> HTTP 404")
        return _FakeResponse(b"elevation")

    bbox = FootprintBBox(lat_min=38.05, lat_max=39.31,
                         lon_min=-98.975, lon_max=-97.325)
    with pytest.raises(CoverageError) as failure:
        fetch_three_dep_tiles(bbox, tmp_path, urlopen=urlopen)
    assert "n40w099" in str(failure.value)
    assert "incomplete" in str(failure.value)


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_parse_static_table_absent_is_none(tmp_path):
    assert parse_static_table(None, source="case.toml",
                              base_dir=tmp_path) is None


def test_parse_static_table_accepts_and_resolves(tmp_path):
    config = parse_static_table(
        {"highres": {"enabled": True, "cache_root": "hr-cache",
                     "on_refuse": "fallback-30s"}},
        source="case.toml", base_dir=tmp_path)
    assert config == HighresStaticConfig(
        enabled=True, cache_root=tmp_path / "hr-cache",
        on_refuse="fallback-30s")
    assert config.echo() == {
        "enabled": "true", "cache_root": str(tmp_path / "hr-cache"),
        "on_refuse": "fallback-30s", "terrain_source": "auto",
        "fields": "auto"}


def test_parse_static_table_refuses_unknown_key(tmp_path):
    with pytest.raises(ValueError, match="cache_roots"):
        parse_static_table(
            {"highres": {"enabled": True, "cache_roots": "x"}},
            source="case.toml", base_dir=tmp_path)


def test_parse_static_table_refuses_unknown_subtable(tmp_path):
    with pytest.raises(ValueError, match="hires"):
        parse_static_table({"hires": {}}, source="case.toml",
                           base_dir=tmp_path)


def test_parse_static_table_refuses_empty_static(tmp_path):
    with pytest.raises(ValueError, match="declares nothing"):
        parse_static_table({}, source="case.toml", base_dir=tmp_path)


def test_parse_static_table_refuses_bad_on_refuse(tmp_path):
    with pytest.raises(ValueError, match="on_refuse"):
        parse_static_table(
            {"highres": {"enabled": True, "cache_root": "x",
                         "on_refuse": "warn"}},
            source="case.toml", base_dir=tmp_path)


def test_parse_static_table_refuses_missing_cache_root(tmp_path):
    with pytest.raises(ValueError, match="cache_root"):
        parse_static_table({"highres": {"enabled": True}},
                           source="case.toml", base_dir=tmp_path)


def test_case_data_loader_carries_static_block(tmp_path):
    from gpuwm.case_data import build_case_data

    for name in ("forcing.grib", "Vtable", "namelist.wps"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "WPS_GEOG").mkdir()
    data = build_case_data(
        {"forcing": "forcing.grib", "vtable": "Vtable",
         "wps_namelist": "namelist.wps", "geog_root": "WPS_GEOG",
         "sfcp_to_sfcp": True, "output_title": "t"},
        source="case.toml", base_dir=tmp_path)
    # Absence of the block is the identity: the field defaults to None and
    # every seam guards on it, so current behavior is bit-identical.
    assert data.static_highres is None


# ---------------------------------------------------------------------------
# Application identity and refusal gates (no network, no rasters)
# ---------------------------------------------------------------------------

def test_apply_absent_config_is_identity_object():
    baseline = _baseline(4, 4)
    fields, receipt = apply_highres_statics(
        baseline, _us_interior_grid(), config=None, domain_id=1,
        case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS)
    assert fields is baseline
    assert receipt is None


def test_apply_disabled_config_is_identity_object(tmp_path):
    baseline = _baseline(4, 4)
    config = HighresStaticConfig(enabled=False, cache_root=tmp_path)
    fields, receipt = apply_highres_statics(
        baseline, _us_interior_grid(), config=config, domain_id=1,
        case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS)
    assert fields is baseline
    assert receipt is None


def test_apply_refuses_outside_us_coverage(tmp_path):
    grid = LambertGrid(
        ref_lat=52.0, ref_lon=-98.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-98.0, dx=3000.0, dy=3000.0, e_we=41, e_sn=41)
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 terrain_source="usgs-3dep-13as")
    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            _baseline(40, 40), grid, config=config, domain_id=1,
            case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS)
    assert failure.value.reason == "outside-source-coverage"
    assert "usgs-3dep-13as" in failure.value.detail


def test_apply_refuses_coastal_footprint_naming_method(tmp_path):
    grid = _us_interior_grid()
    baseline = _baseline(40, 40)
    baseline["LU_INDEX"][3, 7] = 17.0  # WRF ocean category in the baseline
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path)
    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            baseline, grid, config=config, domain_id=1,
            case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS)
    assert failure.value.reason == "coastal-footprint"
    assert "30-arc-second baseline LU_INDEX" in failure.value.detail


def test_apply_refuses_non_modis21_landuse(tmp_path):
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path)
    attrs = dict(MODIS21_ATTRS, ISWATER=16, ISLAKE="")
    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            _baseline(40, 40), _us_interior_grid(), config=config,
            domain_id=1, case_date=date(2021, 5, 15), landuse_attrs=attrs)
    assert failure.value.reason == "landuse-inventory-mismatch"


def test_fallback_30s_returns_identical_baseline_with_receipt(tmp_path):
    grid = LambertGrid(
        ref_lat=52.0, ref_lon=-98.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-98.0, dx=3000.0, dy=3000.0, e_we=41, e_sn=41)
    baseline = _baseline(40, 40)
    frozen = {name: value.copy() for name, value in baseline.items()}
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 on_refuse="fallback-30s",
                                 terrain_source="usgs-3dep-13as")
    fields, receipt = apply_highres_statics(
        baseline, grid, config=config, domain_id=2,
        case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS)
    assert fields is baseline
    for name, value in frozen.items():
        np.testing.assert_array_equal(fields[name], value)
    assert receipt["status"] == "REFUSED"
    assert receipt["refusal"]["reason"] == "outside-source-coverage"
    written = Path(receipt["receipt_path"])
    assert written.is_file()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["status"] == "REFUSED"
    assert payload["grid"]["domain_id"] == 2


# ---------------------------------------------------------------------------
# An enabled block on a lane that cannot honor it refuses, naming the lane
# ---------------------------------------------------------------------------

def _config_with_static_block(tmp_path: Path, *, enabled: bool) -> Path:
    """A TOML carrying nothing but a [static.highres] block.

    ``refuse_inert_highres`` is deliberately readable in isolation: it
    parses the file itself rather than depending on a lane having already
    built an ExperimentConfig, so the gate fires before any of the
    expensive preparation a lane would otherwise do first.
    """

    path = tmp_path / "case.toml"
    path.write_text(
        "[static.highres]\n"
        f"enabled = {'true' if enabled else 'false'}\n"
        f'cache_root = "{tmp_path.as_posix()}"\n',
        encoding="utf-8")
    return path


@pytest.mark.parametrize("lane", [
    "ERA5-direct adapter",
    "mapped adapter",
    "native-HRRR static path",
])
def test_an_enabled_block_refuses_on_each_lane_that_cannot_honor_it(
        tmp_path, lane):
    """One gate per alternate static lane, each naming itself.

    These three routes build their GEOG fields through their own seams
    and load config with ``load_experiment``, which never reads the
    ``[static]`` table.  Before this refusal the block was silently
    INERT there: the run produced the 30 arc second baseline while the
    user's file said otherwise, which reads afterwards as a setting that
    took effect.
    """

    config = _config_with_static_block(tmp_path, enabled=True)
    with pytest.raises(ValueError) as failure:
        refuse_inert_highres(config, lane=lane)
    message = str(failure.value)
    assert lane in message, message
    # The remedy is named, not merely the complaint.
    assert "gpuwm run" in message
    assert "enabled = false" in message


@pytest.mark.parametrize("raw", ["disabled", "absent"])
def test_a_block_that_asks_for_nothing_passes_every_lane(tmp_path, raw):
    """Absence and ``enabled = false`` are not refusals.

    The gate exists to stop a silent no-op, and a user who wrote the
    baseline down deliberately -- or wrote no block at all -- asked for
    exactly what the lane does.
    """

    if raw == "absent":
        config = tmp_path / "case.toml"
        config.write_text("[case_data]\n", encoding="utf-8")
    else:
        config = _config_with_static_block(tmp_path, enabled=False)
    for lane in ("ERA5-direct adapter", "mapped adapter",
                 "native-HRRR static path"):
        refuse_inert_highres(config, lane=lane)
