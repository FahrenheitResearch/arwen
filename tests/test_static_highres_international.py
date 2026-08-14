"""Gates for the international (terrain-only) high-resolution lane.

Covers the per-source coverage model, the near-global tile enumerators,
the absent-tile-means-water cross-check, the terrain-only science and the
config surface -- all without touching the network or any real raster.
The end-to-end agreement against 3DEP on a real US footprint is a
separate, network-bound validation recorded in the evidence gallery.
"""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gpuwm.static.highres import (build_terrain_override,
                                  merge_terrain_override)
from gpuwm.static.highres_fetch import (
    COPERNICUS_DEM_TILE_URL,
    CoverageError,
    FootprintBBox,
    SRTM_GL1_TILE_URL,
    SourceAbsent,
    TERRAIN_SOURCES,
    copernicus_dem_tile_ids,
    fetch_copernicus_dem_tiles,
    fetch_srtm_gl1_tiles,
    one_degree_tile_bbox,
    srtm_tile_ids,
    terrain_source_coverage,
)
from gpuwm.static.highres_production import (
    HighresRefusal,
    HighresStaticConfig,
    apply_highres_statics,
    parse_static_table,
)
from gpuwm.static.lambert import LambertGrid

MODIS21_ATTRS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                 "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}


def _european_grid(dx: float = 3000.0, n: int = 41) -> LambertGrid:
    """A domain over central Europe -- outside every US collection."""
    return LambertGrid(
        ref_lat=48.2, ref_lon=16.4, truelat1=30.0, truelat2=60.0,
        stand_lon=16.4, dx=dx, dy=dx, e_we=n, e_sn=n)


def _us_grid(dx: float = 3000.0, n: int = 41) -> LambertGrid:
    return LambertGrid(
        ref_lat=38.68, ref_lon=-98.15, truelat1=30.0, truelat2=60.0,
        stand_lon=-98.15, dx=dx, dy=dx, e_we=n, e_sn=n)


def _baseline(ny: int, nx: int, *, land: bool = True) -> dict:
    return {
        "HGT_M": np.full((ny, nx), 500.0),
        "LU_INDEX": np.full((ny, nx), 10.0),
        "LANDMASK": np.full((ny, nx), 1.0 if land else 0.0),
        "SOILTEMP": np.full((ny, nx), 285.0),
    }


# ---------------------------------------------------------------------------
# Per-source coverage model
# ---------------------------------------------------------------------------

def test_every_terrain_source_declares_licence_and_attribution():
    for source_id, coverage in TERRAIN_SOURCES.items():
        assert coverage.source_id == source_id
        assert coverage.role == "terrain"
        assert coverage.license_id and coverage.license_url
        assert coverage.attribution, f"{source_id} carries no attribution"
        assert coverage.source_url.startswith("https://")


def test_coverage_check_names_source_footprint_and_overshoot():
    coverage = terrain_source_coverage("usgs-3dep-13as")
    bbox = FootprintBBox(lat_min=47.0, lat_max=52.0,
                         lon_min=5.0, lon_max=10.0)
    with pytest.raises(CoverageError) as failure:
        coverage.check(bbox)
    message = str(failure.value)
    assert "usgs-3dep-13as" in message
    assert "lat_max" in message           # the footprint is quoted
    assert "east_by_deg" in message       # the observed overshoot is quoted
    assert coverage.outside(bbox)["north_by_deg"] == pytest.approx(2.5)


def test_copernicus_covers_europe_but_srtm_stops_at_sixty_north():
    scandinavia = FootprintBBox(lat_min=61.0, lat_max=62.0,
                                lon_min=9.0, lon_max=10.0)
    terrain_source_coverage("copernicus-dem-glo30").check(scandinavia)
    with pytest.raises(CoverageError, match="srtm-gl1"):
        terrain_source_coverage("srtm-gl1").check(scandinavia)


def test_unknown_terrain_source_names_the_known_ones():
    with pytest.raises(CoverageError) as failure:
        terrain_source_coverage("aster-gdem")
    assert "copernicus-dem-glo30" in str(failure.value)


# ---------------------------------------------------------------------------
# Near-global tile enumeration
# ---------------------------------------------------------------------------

def test_copernicus_tiles_in_all_four_quadrants():
    assert copernicus_dem_tile_ids(FootprintBBox(
        39.2, 39.8, -104.8, -104.2)) == ("N39_00_W105_00",)
    assert copernicus_dem_tile_ids(FootprintBBox(
        -33.9, -33.2, 18.2, 18.8)) == ("S34_00_E018_00",)
    assert set(copernicus_dem_tile_ids(FootprintBBox(
        47.8, 48.3, 16.2, 16.7))) == {"N47_00_E016_00", "N48_00_E016_00"}


def test_copernicus_tile_bbox_round_trips_both_hemispheres():
    box = one_degree_tile_bbox("S34_00_E018_00")
    assert (box.lat_min, box.lat_max) == (-34.0, -33.0)
    assert (box.lon_min, box.lon_max) == (18.0, 19.0)
    box = one_degree_tile_bbox("N39W105")
    assert (box.lat_min, box.lat_max) == (39.0, 40.0)
    assert (box.lon_min, box.lon_max) == (-105.0, -104.0)


def test_srtm_tile_ids_use_the_compact_naming():
    assert srtm_tile_ids(FootprintBBox(39.2, 39.8, -104.8, -104.2)) \
        == ("N39W105",)


def test_antimeridian_span_refuses_at_enumeration():
    with pytest.raises(CoverageError, match="antimeridian"):
        copernicus_dem_tile_ids(FootprintBBox(30.0, 31.0, -179.9, 179.9))


# ---------------------------------------------------------------------------
# Absent tiles: water, unless the baseline says land
# ---------------------------------------------------------------------------

class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_absent_copernicus_tile_is_returned_not_swallowed(tmp_path):
    def urlopen(url, offset):
        if "N48_00_E016_00" in url:
            raise SourceAbsent(f"{url} -> HTTP 404")
        return _FakeResponse(b"elevation")

    tiles, absent = fetch_copernicus_dem_tiles(
        FootprintBBox(47.8, 48.3, 16.2, 16.7), tmp_path, urlopen=urlopen)
    assert len(tiles) == 1
    assert absent == ("N48_00_E016_00",)


def test_all_tiles_absent_refuses_as_open_water(tmp_path):
    def urlopen(url, offset):
        raise SourceAbsent(f"{url} -> HTTP 404")

    with pytest.raises(CoverageError) as failure:
        fetch_copernicus_dem_tiles(
            FootprintBBox(29.2, 29.8, -40.8, -40.2), tmp_path,
            urlopen=urlopen)
    assert "open water" in str(failure.value)
    assert "N29_00_W041_00" in str(failure.value)


def test_srtm_fetch_uses_the_anonymous_mirror(tmp_path):
    seen = []

    def urlopen(url, offset):
        seen.append(url)
        return _FakeResponse(b"elevation")

    fetch_srtm_gl1_tiles(FootprintBBox(39.2, 39.8, -104.8, -104.2),
                         tmp_path, urlopen=urlopen)
    assert seen == [SRTM_GL1_TILE_URL.format(tile="N39W105")]
    assert "opentopography" in seen[0]
    # No credential, token or signature is ever appended.
    assert "?" not in seen[0] and "X-Amz" not in seen[0]


def test_absent_tile_over_baseline_land_refuses_by_name(tmp_path):
    """An absent tile means water -- unless our own mask says otherwise."""
    grid = _european_grid()
    baseline = _baseline(grid.e_sn - 1, grid.e_we - 1, land=True)

    def urlopen(url, offset):
        raise SourceAbsent(f"{url} -> HTTP 404")

    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 fields="terrain")
    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(baseline, grid, config=config, domain_id=1,
                              case_date=__import__("datetime").date(
                                  2021, 5, 4),
                              landuse_attrs=MODIS21_ATTRS, urlopen=urlopen)
    # Every tile absent trips the open-water refusal first, which is the
    # correct precedence: there is nothing to cross-check against.
    assert "open water" in failure.value.detail


def test_absent_tile_cross_check_counts_land_cells(tmp_path):
    from gpuwm.static.highres_production import _absent_tiles_over_land
    grid = _european_grid()
    baseline = _baseline(grid.e_sn - 1, grid.e_we - 1, land=True)
    hits = _absent_tiles_over_land(("N48_00_E016_00",), grid, baseline)
    assert hits["N48_00_E016_00"] > 0
    baseline_water = _baseline(grid.e_sn - 1, grid.e_we - 1, land=False)
    assert _absent_tiles_over_land(("N48_00_E016_00",), grid,
                                   baseline_water) == {}


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_parse_accepts_terrain_source_and_fields(tmp_path):
    config = parse_static_table(
        {"highres": {"enabled": True, "cache_root": str(tmp_path),
                     "terrain_source": "copernicus-dem-glo30",
                     "fields": "terrain"}},
        source="case.toml", base_dir=tmp_path)
    assert config.terrain_source == "copernicus-dem-glo30"
    assert config.fields == "terrain"
    assert config.echo()["fields"] == "terrain"


def test_parse_defaults_both_new_keys_to_auto(tmp_path):
    config = parse_static_table(
        {"highres": {"enabled": True, "cache_root": str(tmp_path)}},
        source="case.toml", base_dir=tmp_path)
    assert (config.terrain_source, config.fields) == ("auto", "auto")


def test_parse_refuses_unknown_terrain_source_naming_choices(tmp_path):
    with pytest.raises(ValueError, match="copernicus-dem-glo30"):
        parse_static_table(
            {"highres": {"enabled": True, "cache_root": str(tmp_path),
                         "terrain_source": "aster"}},
            source="case.toml", base_dir=tmp_path)


def test_parse_refuses_unknown_fields_choice(tmp_path):
    with pytest.raises(ValueError, match="'terrain'"):
        parse_static_table(
            {"highres": {"enabled": True, "cache_root": str(tmp_path),
                         "fields": "landuse"}},
            source="case.toml", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Plan resolution and refusals
# ---------------------------------------------------------------------------

def _plan(config, grid):
    from gpuwm.static.highres_fetch import domain_footprint
    from gpuwm.static.build import HALO
    from gpuwm.static.highres_production import _resolve_plan
    return _resolve_plan(config, domain_footprint(grid, HALO))


def test_auto_picks_us_stack_inside_the_us(tmp_path):
    mode, coverage = _plan(
        HighresStaticConfig(enabled=True, cache_root=tmp_path), _us_grid())
    assert (mode, coverage.source_id) == ("all", "usgs-3dep-13as")


def test_auto_picks_terrain_only_copernicus_abroad(tmp_path):
    mode, coverage = _plan(
        HighresStaticConfig(enabled=True, cache_root=tmp_path),
        _european_grid())
    assert (mode, coverage.source_id) == ("terrain", "copernicus-dem-glo30")


def test_fields_all_abroad_refuses_naming_the_landcover_source(tmp_path):
    with pytest.raises(HighresRefusal) as failure:
        _plan(HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                  fields="all"), _european_grid())
    assert failure.value.reason == "landcover-source-missing"
    assert "annual-nlcd" in failure.value.detail
    assert "fields = \"terrain\"" in failure.value.detail


def test_pinned_us_source_abroad_refuses_naming_the_source(tmp_path):
    with pytest.raises(HighresRefusal) as failure:
        _plan(HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                  fields="terrain",
                                  terrain_source="usgs-3dep-13as"),
              _european_grid())
    assert failure.value.reason == "outside-source-coverage"
    assert "usgs-3dep-13as" in failure.value.detail


def test_terrain_only_is_allowed_inside_the_us_for_cross_validation(tmp_path):
    mode, coverage = _plan(
        HighresStaticConfig(enabled=True, cache_root=tmp_path,
                            fields="terrain",
                            terrain_source="copernicus-dem-glo30"),
        _us_grid())
    assert (mode, coverage.source_id) == ("terrain", "copernicus-dem-glo30")


# ---------------------------------------------------------------------------
# Terrain-only science
# ---------------------------------------------------------------------------

def test_merge_terrain_override_recomputes_tmn_and_holds_the_mask():
    baseline = _baseline(6, 5)
    baseline["LANDMASK"][0, 0] = 0.0
    baseline["TMN"] = baseline["SOILTEMP"] - 0.0065 * baseline["HGT_M"]
    new_hgt = np.full((6, 5), 1500.0)
    merged, audit = merge_terrain_override(baseline, {"HGT_M": new_hgt})
    assert np.array_equal(merged["LANDMASK"], baseline["LANDMASK"])
    assert np.array_equal(merged["LU_INDEX"], baseline["LU_INDEX"])
    # Land cells lapse with the new height; the water cell keeps SOILTEMP.
    assert merged["TMN"][1, 1] == pytest.approx(285.0 - 0.0065 * 1500.0)
    assert merged["TMN"][0, 0] == pytest.approx(285.0)
    assert audit["terrain_cells_changed"] == 30
    assert audit["newly_land_nearest_climatology_fallback_cells"] == 0


def test_merge_terrain_override_refuses_non_terrain_overrides():
    baseline = _baseline(4, 4)
    with pytest.raises(KeyError, match="LANDMASK"):
        merge_terrain_override(
            baseline, {"HGT_M": np.zeros((4, 4)),
                       "LANDMASK": np.ones((4, 4))})


def test_merge_terrain_override_refuses_shape_mismatch():
    baseline = _baseline(4, 4)
    with pytest.raises(ValueError, match="shape"):
        merge_terrain_override(baseline, {"HGT_M": np.zeros((3, 3))})


def test_build_terrain_override_returns_terrain_alone(monkeypatch):
    grid = _us_grid(n=11)
    calls = {}

    def fake_resample(source, target_grid, *, method):
        calls["method"] = method
        ny, nx = target_grid.e_sn - 1, target_grid.e_we - 1
        return np.linspace(0.0, 1000.0, ny * nx).reshape(ny, nx)

    class _Terrain:
        def receipt(self):
            return {"source_id": "copernicus-dem-glo30"}

    monkeypatch.setattr("gpuwm.static.highres.resample_continuous",
                        fake_resample)
    fields, audit = build_terrain_override(grid, terrain=_Terrain(), halo=3)
    assert set(fields) == {"HGT_M"}
    assert fields["HGT_M"].shape == (grid.e_sn - 1, grid.e_we - 1)
    assert calls["method"] == "average"
    assert "terrain only" in audit["method"]


# ---------------------------------------------------------------------------
# Antimeridian
# ---------------------------------------------------------------------------

def test_antimeridian_domain_refuses_before_any_fetch(tmp_path):
    from gpuwm.static.highres_production import _require_single_lobe
    with pytest.raises(HighresRefusal) as failure:
        _require_single_lobe(FootprintBBox(30.0, 31.0, -179.9, 179.9))
    assert failure.value.reason == "antimeridian-footprint"


# ---------------------------------------------------------------------------
# One domain, two sources: the receipts must not overwrite each other
# ---------------------------------------------------------------------------

def test_two_terrain_sources_on_one_domain_write_two_receipts(tmp_path):
    """Cross-validating a domain through two DEMs must keep both receipts.

    The documented way to find out what changing ``terrain_source`` does to
    your terrain is to build the same domain twice and compare.  The
    receipt is the artifact that records which source ran, so if both runs
    write to the same path the first run's provenance is destroyed by the
    second -- and the comparison it exists to support becomes unauditable.
    Two runs that differ only in the source they asked for are two runs.
    """
    import json

    # 70 N: outside 3DEP (a US collection) and outside SRTM's 60 N ceiling,
    # so both refuse on coverage without touching the network.
    grid = LambertGrid(ref_lat=70.0, ref_lon=25.0, truelat1=30.0,
                       truelat2=60.0, stand_lon=25.0, dx=3000.0, dy=3000.0,
                       e_we=41, e_sn=41)
    baseline = _baseline(40, 40)

    written = {}
    for source in ("usgs-3dep-13as", "srtm-gl1"):
        config = HighresStaticConfig(
            enabled=True, cache_root=tmp_path, on_refuse="fallback-30s",
            terrain_source=source, fields="terrain")
        _, receipt = apply_highres_statics(
            baseline, grid, config=config, domain_id=1,
            case_date=date(2024, 6, 1), landuse_attrs=MODIS21_ATTRS)
        assert receipt["status"] == "REFUSED"
        written[source] = Path(receipt["receipt_path"])

    assert written["usgs-3dep-13as"] != written["srtm-gl1"], (
        "both terrain sources wrote the same receipt file "
        f"{written['usgs-3dep-13as']}; the first run's provenance was "
        "overwritten by the second")
    for source, path in written.items():
        assert path.is_file(), path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["config"]["terrain_source"] == source
