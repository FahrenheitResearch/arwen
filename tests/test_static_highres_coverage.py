"""A high-resolution source covers only where it is published.

The breakage these gates prevent, named: a 1 km, 225x225-cell parent that
reached about 110 km out to sea failed "Prepare root static fields" with
"high-resolution land cover lacks 225162 target values".  The land-cover
collection is published over land and a strip of near-shore water; past
that strip no source pixel reaches a model cell, the resampled fractions
there are NaN, and the overlay refused the whole preparation instead of
taking the 30-arc-second land use the engine runs without it.  The same
refusal waited for a domain that crosses a national border (the far side
is not in a national collection), for an unpublished terrain tile, and
for a footprint that runs past the land-cover raster's extent.

What is required instead, for every field the overlay replaces: cells
outside the source's coverage take the baseline (water stays water, and
the land/water mask stays the baseline's own there), the hand-over runs
over the nest terrain ramp so the edge leaves no step, the count and
bounds of the cells are in the receipt, one plain warning names them,
and the only refusal left is a cell neither source covers.

Each geometry below is half inside and half outside its source, and each
test fails on the tree before this change.
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gpuwm.static import highres as highres_module
from gpuwm.static.build import HALO, landmask_from_landusef
from gpuwm.static.highres import (BoundRaster, baseline_ocean_mask,
                                  build_highres_overrides,
                                  build_terrain_override, sha256_file)
from gpuwm.static.highres_fetch import (FetchedFile, FootprintBBox,
                                        SourceAbsent, domain_footprint,
                                        fetch_three_dep_tiles)
from gpuwm.static.highres_production import (HighresStaticConfig,
                                             _resolve_plan,
                                             apply_highres_statics)
from gpuwm.static.lambert import LambertGrid

MODIS21_ATTRS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                 "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}

HIGHRES_FIXTURES = (Path(__file__).resolve().parent.parent
                    / "tools" / "rustwx" / "crates" / "static-fields"
                    / "tests" / "fixtures" / "highres")

N = 25                       # mass cells per side of the synthetic domain
WIDTH = 5                    # WRF's default blend_width, the nest ramp


def _expected_weight(covered: np.ndarray, width: int = WIDTH) -> np.ndarray:
    """The ramp written out the slow way, independent of the module: the
    square-ring (Chebyshev) distance to the nearest uncovered cell, over
    width+1, capped at one."""
    covered = np.asarray(covered, dtype=bool)
    ys, xs = np.nonzero(~covered)
    weight = np.ones(covered.shape)
    if ys.size == 0:
        return weight
    for j in range(covered.shape[0]):
        for i in range(covered.shape[1]):
            ring = int(np.min(np.maximum(np.abs(ys - j), np.abs(xs - i))))
            weight[j, i] = min(ring, width + 1) / (width + 1.0)
    return weight


def _grid(n: int = N + 1, dx: float = 3000.0) -> LambertGrid:
    return LambertGrid(
        ref_lat=38.7, ref_lon=-98.2, truelat1=30.0, truelat2=60.0,
        stand_lon=-98.2, dx=dx, dy=dx, e_we=n, e_sn=n)


def _extended_shape(grid, halo: int = HALO) -> tuple[int, int]:
    return grid.e_sn - 1 + 2 * halo, grid.e_we - 1 + 2 * halo


def _east_uncovered(grid, halo: int = HALO, columns: int = 12) -> np.ndarray:
    """Extended-grid coverage with the east ``columns`` of the DOMAIN (and
    the halo beyond them) outside the source."""
    ny, nx = _extended_shape(grid, halo)
    covered = np.ones((ny, nx), dtype=bool)
    covered[:, nx - halo - columns:] = False
    return covered


class _Stub:
    def __init__(self, source_id: str):
        self._source_id = source_id

    def receipt(self):
        return {"source_id": self._source_id}


def _noah_baseline(ny: int, nx: int, *, sea_columns: int = 0
                   ) -> dict[str, np.ndarray]:
    """A complete, self-consistent 30-arc-second state: cropland land,
    and optionally WRF ocean in the east ``sea_columns`` columns."""
    luf = np.zeros((21, ny, nx))
    luf[11] = 1.0                                   # MODIS 12, cropland
    soil = np.zeros((16, ny, nx))
    soil[2] = 1.0                                   # soil category 3
    landmask = np.ones((ny, nx))
    lu_index = np.full((ny, nx), 12.0)
    hgt = np.full((ny, nx), 200.0)
    if sea_columns:
        sea = (slice(None), slice(nx - sea_columns, None))
        luf[(slice(None),) + sea] = 0.0
        luf[(16,) + sea] = 1.0                      # WRF ocean 17
        soil[(slice(None),) + sea] = 0.0
        soil[(13,) + sea] = 1.0                     # soil water 14
        landmask[sea] = 0.0
        lu_index[sea] = 17.0
        hgt[sea] = 0.0
    return {
        "HGT_M": hgt, "LANDUSEF": luf, "LANDMASK": landmask,
        "LU_INDEX": lu_index, "SOILCTOP": soil.copy(),
        "SOILCBOT": soil.copy(),
        "SCT_DOM": np.argmax(soil, axis=0) + 1.0,
        "SCB_DOM": np.argmax(soil, axis=0) + 1.0,
        "GREENFRAC": np.full((12, ny, nx), 0.5),
        "LAI12M": np.full((12, ny, nx), 1.0),
        "ALBEDO12M": np.full((12, ny, nx), 18.0),
        "SNOALB": np.full((ny, nx), 60.0),
        "SOILTEMP": np.full((ny, nx), 285.0),
    }


def _stub_sources(monkeypatch, *, terrain_covered=None,
                  landcover_covered=None, soil_covered=None,
                  hr_height: float = 1000.0):
    """Stub the three resamples on the halo-extended grid.

    ``*_covered`` is an extended-grid boolean plane (None = everywhere);
    outside it the stub returns NaN, which is exactly what the Rust warp
    returns for a cell no source pixel reaches.  Land cover is deciduous
    forest (NLCD 41 -> MODIS 4) wherever it is published.
    """
    def plane(covered, ny, nx):
        return (np.ones((ny, nx), dtype=bool) if covered is None
                else np.asarray(covered, dtype=bool))

    def fake_continuous(source, grid, *, method):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        out = np.full((ny, nx), float(hr_height))
        out[~plane(terrain_covered, ny, nx)] = np.nan
        return out

    def fake_categories(source, grid, mapping, *, category_count):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        fractions = np.zeros((category_count, ny, nx))
        fractions[3] = 1.0                           # MODIS 4
        fractions[:, ~plane(landcover_covered, ny, nx)] = np.nan
        return fractions

    def fake_soil(sources, weights, grid, *, category_count):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        fractions = np.zeros((category_count, ny, nx))
        fractions[5] = 1.0                           # soil category 6
        fractions[:, ~plane(soil_covered, ny, nx)] = np.nan
        return fractions, {"raw_component_total_percent_min": 100.0,
                           "raw_component_total_percent_max": 100.0,
                           "valid_source_pixels": ny * nx}

    monkeypatch.setattr(highres_module, "resample_continuous",
                        fake_continuous)
    monkeypatch.setattr(highres_module, "resample_mapped_categories",
                        fake_categories)
    monkeypatch.setattr(highres_module, "soilgrids_category_fractions",
                        fake_soil)


def _crop(plane: np.ndarray, halo: int = HALO) -> np.ndarray:
    return plane[..., halo:plane.shape[-2] - halo, halo:plane.shape[-1] - halo]


# ---------------------------------------------------------------------------
# The ramp
# ---------------------------------------------------------------------------

def test_the_hand_over_ramp_is_the_nest_terrain_ramp():
    """k-th ring in from the edge carries k/(width+1) of the source."""
    assert highres_module.COVERAGE_BLEND_CELLS == WIDTH
    _coverage_weight = highres_module._coverage_weight
    covered = np.ones((3, 20), dtype=bool)
    covered[:, :4] = False
    weight = _coverage_weight(covered, width=5)
    assert np.all(weight[:, :4] == 0.0)
    for ring in range(1, 6):
        assert np.all(weight[:, 3 + ring] == pytest.approx(ring / 6.0))
    assert np.all(weight[:, 9:] == 1.0)

    # Rings are squares: a diagonal neighbour is on the first ring.
    covered = np.ones((9, 9), dtype=bool)
    covered[4, 4] = False
    weight = _coverage_weight(covered, width=5)
    assert weight[4, 4] == 0.0
    assert weight[3, 3] == weight[5, 5] == weight[3, 4] == pytest.approx(
        1.0 / 6.0)

    # Fully covered and fully uncovered planes are untouched.
    assert np.all(_coverage_weight(np.ones((4, 4), bool)) == 1.0)
    assert np.all(_coverage_weight(np.zeros((4, 4), bool)) == 0.0)

    # And the slow, independent spelling agrees on an irregular edge.
    rng = np.random.default_rng(7)
    covered = rng.random((17, 23)) > 0.08
    np.testing.assert_allclose(_coverage_weight(covered),
                               _expected_weight(covered), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

def test_terrain_outside_its_source_takes_the_baseline_across_the_ramp(
        monkeypatch):
    grid = _grid()
    covered = _east_uncovered(grid)
    _stub_sources(monkeypatch, terrain_covered=covered)
    baseline = _noah_baseline(N, N)

    fields, audit = build_terrain_override(
        grid, terrain=_Stub("copernicus-dem-glo30"), halo=HALO,
        baseline=baseline)

    hgt = fields["HGT_M"]
    weight = _crop(_expected_weight(covered))
    outside = weight == 0.0
    assert outside.sum() == 12 * N
    # Past the edge: the baseline, exactly.
    np.testing.assert_array_equal(hgt[outside], 200.0)
    # Rings 3..5 lie beyond the smoother's two-cell reach of the fill, so
    # they are exactly the ramp between the two heights.
    for ring in (3, 4, 5):
        on_ring = np.isclose(weight, ring / 6.0)
        assert on_ring.sum() == N
        np.testing.assert_allclose(hgt[on_ring],
                                   200.0 + 800.0 * ring / 6.0, rtol=1e-12)
    # Beyond the ramp and the smoother: the source, untouched.
    deep = np.zeros_like(outside)
    deep[:, :N - 12 - WIDTH - 2] = True
    np.testing.assert_allclose(hgt[deep], 1000.0, rtol=1e-12)
    # The 800 m difference is spread over the ramp instead of taken in
    # one step (a sixth per ring, plus the smoother's ringing beside the
    # fill on the first two rings).
    step = np.abs(np.diff(hgt, axis=1)).max()
    assert step <= 800.0 / 4.0

    record = audit["coverage"]["fields"]["terrain"]
    assert record["source"] == "copernicus-dem-glo30"
    assert record["cells_outside_coverage"] == 12 * N
    assert record["cells_blended"] == WIDTH * N
    lat, lon = grid.latlon_mass()
    bounds = record["outside_bounds"]
    assert bounds["lon_min"] == pytest.approx(float(lon[outside].min()),
                                              abs=1e-4)
    assert bounds["lat_max"] == pytest.approx(float(lat[outside].max()),
                                              abs=1e-4)
    assert bounds["lon_min"] > float(lon[~outside].min())


def test_a_fully_covered_domain_is_unchanged_by_the_coverage_rule(
        monkeypatch):
    """The hand-over only exists where a source stops; full coverage is
    the old path byte for byte, with nothing counted."""
    grid = _grid()
    _stub_sources(monkeypatch)
    baseline = _noah_baseline(N, N)
    with_baseline, audit = build_terrain_override(
        grid, terrain=_Stub("s"), halo=HALO, baseline=baseline)
    without, _ = build_terrain_override(grid, terrain=_Stub("s"), halo=HALO)
    assert with_baseline["HGT_M"].tobytes() == without["HGT_M"].tobytes()
    record = audit["coverage"]["fields"]["terrain"]
    assert record["cells_outside_coverage"] == 0
    assert record["cells_blended"] == 0
    assert record["outside_bounds"] is None


def test_the_only_refusal_left_is_a_cell_neither_source_covers(monkeypatch):
    grid = _grid()
    _stub_sources(monkeypatch, terrain_covered=_east_uncovered(grid))

    with pytest.raises(ValueError) as failure:
        build_terrain_override(grid, terrain=_Stub("srtm-gl1"), halo=HALO)
    message = str(failure.value)
    assert "neither source covers" in message
    assert "srtm-gl1" in message
    assert "HGT_M" in message
    assert "lat_min" in message                  # where the cells are

    holed = _noah_baseline(N, N)
    holed["HGT_M"][:, -3:] = np.nan              # the baseline has a hole too
    with pytest.raises(ValueError) as failure:
        build_terrain_override(grid, terrain=_Stub("srtm-gl1"), halo=HALO,
                               baseline=holed)
    assert "not finite there either" in str(failure.value)
    assert f"{3 * N} cell(s)" in str(failure.value)


# ---------------------------------------------------------------------------
# Land use
# ---------------------------------------------------------------------------

def _overrides(grid, baseline):
    return build_highres_overrides(
        grid, terrain=_Stub("copernicus-dem-glo30"),
        landcover=_Stub("annual-nlcd-2024"),
        soil_sources={("sand", "0-5cm"): _Stub("soilgrids-v2")},
        baseline_ocean=baseline_ocean_mask(baseline), halo=HALO,
        baseline=baseline)


def test_land_use_past_the_collections_offshore_edge_keeps_the_sea(
        monkeypatch):
    """The reported geometry in small: the east of the domain is sea the
    land-cover collection does not reach."""
    grid = _grid()
    covered = _east_uncovered(grid, columns=12)
    _stub_sources(monkeypatch, landcover_covered=covered)
    baseline = _noah_baseline(N, N, sea_columns=15)

    fields, audit = _overrides(grid, baseline)

    weight = _crop(_expected_weight(covered))
    outside = weight == 0.0
    luf, landmask = fields["LANDUSEF"], fields["LANDMASK"]
    # Water stays water, as the baseline's own mask and index.
    np.testing.assert_array_equal(fields["LU_INDEX"][outside], 17.0)
    np.testing.assert_array_equal(landmask[outside], 0.0)
    np.testing.assert_array_equal(luf[:, outside],
                                  baseline["LANDUSEF"][:, outside])
    # Everywhere: fractions sum to one and the mask is the land-use rule's.
    np.testing.assert_allclose(luf.sum(axis=0), 1.0, rtol=1e-12)
    np.testing.assert_array_equal(
        landmask, landmask_from_landusef(luf, iswater=17, islake=21))
    # The ramp: source and baseline mixed by the ring weight.
    ramp = (weight > 0.0) & (weight < 1.0)
    np.testing.assert_allclose(luf[3][ramp], weight[ramp], rtol=1e-12)
    np.testing.assert_allclose(
        luf[16][ramp], (1.0 - weight[ramp]) * baseline["LANDUSEF"][16][ramp],
        rtol=1e-12)
    # Soil over the kept sea is water.
    np.testing.assert_array_equal(fields["SOILCTOP"][13][outside], 1.0)

    record = audit["coverage"]["fields"]["land_use"]
    assert record["source"] == "annual-nlcd-2024"
    assert record["cells_outside_coverage"] == int(outside.sum()) == 12 * N
    assert record["cells_blended"] == WIDTH * N
    assert record["outside_bounds"]["lat_min"] < record["outside_bounds"][
        "lat_max"]


def test_land_use_across_a_border_blends_two_land_classifications(
        monkeypatch):
    """Land on both sides: the far side keeps the baseline's land use and
    the ramp mixes the two classifications instead of stepping."""
    grid = _grid()
    covered = _east_uncovered(grid, columns=10)
    _stub_sources(monkeypatch, landcover_covered=covered)
    baseline = _noah_baseline(N, N)

    fields, _ = _overrides(grid, baseline)

    weight = _crop(_expected_weight(covered))
    luf = fields["LANDUSEF"]
    np.testing.assert_array_equal(fields["LU_INDEX"][weight == 0.0], 12.0)
    np.testing.assert_array_equal(fields["LU_INDEX"][weight > 0.5], 4.0)
    np.testing.assert_allclose(luf[3], weight, rtol=1e-12)
    np.testing.assert_allclose(luf[11], 1.0 - weight, rtol=1e-12)
    # No column-to-column jump in any class fraction beyond one ring.
    assert np.abs(np.diff(luf, axis=2)).max() <= 1.0 / 6.0 + 1e-12
    assert np.all(fields["LANDMASK"] == 1.0)


def test_a_land_cover_source_that_reaches_no_cell_hands_over_everything(
        monkeypatch):
    grid = _grid()
    _stub_sources(monkeypatch)
    baseline = _noah_baseline(N, N, sea_columns=6)
    fields, audit = build_highres_overrides(
        grid, terrain=_Stub("t"), landcover=None,
        soil_sources={("sand", "0-5cm"): _Stub("soilgrids-v2")},
        baseline_ocean=baseline_ocean_mask(baseline), halo=HALO,
        baseline=baseline)
    for name in ("LANDUSEF", "LANDMASK", "LU_INDEX"):
        np.testing.assert_array_equal(fields[name], baseline[name])
    record = audit["coverage"]["fields"]["land_use"]
    assert record["source"] is None
    assert record["cells_outside_coverage"] == N * N


# ---------------------------------------------------------------------------
# Soil
# ---------------------------------------------------------------------------

def test_soil_outside_its_source_takes_the_baseline_and_is_counted(
        monkeypatch):
    grid = _grid()
    covered = _east_uncovered(grid, columns=8)
    _stub_sources(monkeypatch, soil_covered=covered)
    baseline = _noah_baseline(N, N)

    fields, audit = _overrides(grid, baseline)

    outside = ~_crop(covered)
    top = fields["SOILCTOP"]
    np.testing.assert_array_equal(top[:, outside],
                                  baseline["SOILCTOP"][:, outside])
    np.testing.assert_array_equal(top[5][~outside], 1.0)
    np.testing.assert_allclose(top.sum(axis=0), 1.0, rtol=1e-12)
    for key, label in (("soil_top_0_30cm", "soil 0-30 cm"),
                       ("soil_bottom_30_100cm", "soil 30-100 cm")):
        record = audit["coverage"]["fields"][key]
        assert record["field"] == label
        assert record["source"] == "soilgrids-v2"
        assert record["cells_outside_coverage"] == 8 * N
        assert record["outside_bounds"] is not None


# ---------------------------------------------------------------------------
# The production door
# ---------------------------------------------------------------------------

def test_a_coastal_parent_past_the_land_cover_edge_prepares_and_warns_once(
        tmp_path, monkeypatch, capsys):
    """The reported failure, through apply_highres_statics."""
    grid = _grid()
    covered = _east_uncovered(grid, columns=12)
    _stub_sources(monkeypatch, landcover_covered=covered)

    def fake_fetch_and_bind(bbox, cache_root, case_date, *, coverage, grid,
                            baseline, urlopen=None, landcover_source=None):
        return (_Stub("usgs-3dep-13as"), _Stub("annual-nlcd-2024"),
                {("sand", "0-5cm"): _Stub("soilgrids-v2")},
                {"landcover_year": 2024, "landcover_anachronism_years": 0,
                 "bytes_fetched": 0})

    monkeypatch.setattr(
        "gpuwm.static.highres_production._fetch_and_bind",
        fake_fetch_and_bind)
    baseline = _noah_baseline(N, N, sea_columns=15)
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 landcover_source="annual-nlcd")

    fields, receipt = apply_highres_statics(
        baseline, grid, config=config, domain_id=1,
        case_date=date(2024, 6, 1), landuse_attrs=MODIS21_ATTRS)

    assert receipt["status"] == "APPLIED"
    record = receipt["coverage"]["fields"]["land_use"]
    assert record["cells_outside_coverage"] == 12 * N
    assert record["outside_bounds"] is not None
    assert receipt["coverage"]["blend_cells"] == WIDTH
    outside = _crop(_expected_weight(covered)) == 0.0
    np.testing.assert_array_equal(fields["LU_INDEX"][outside], 17.0)
    written = json.loads(Path(receipt["receipt_path"]).read_text(
        encoding="utf-8"))
    assert written["coverage"]["fields"]["land_use"][
        "cells_outside_coverage"] == 12 * N

    printed = capsys.readouterr().out
    warnings = [line for line in printed.splitlines() if "WARNING" in line]
    assert len(warnings) == 1, printed
    assert "land use (annual-nlcd-2024)" in warnings[0]
    assert f"{12 * N} of {N * N} cells" in warnings[0]
    assert "30-arc-second baseline" in warnings[0]


def test_a_domain_no_terrain_tile_reaches_prepares_on_the_baseline(
        tmp_path, capsys):
    """RETIRES test_absent_tile_over_baseline_land_refuses_by_name and
    test_all_tiles_absent_refuses_as_open_water.

    Every tile over this footprint is unpublished.  That is a coverage
    fact about the source, and the baseline covers every cell, so the
    preparation proceeds on the baseline terrain and says so; it is not
    the zero-cells refusal, which is for an overlay that reached cells
    and changed none of them.
    """
    grid = LambertGrid(
        ref_lat=48.2, ref_lon=16.4, truelat1=30.0, truelat2=60.0,
        stand_lon=16.4, dx=3000.0, dy=3000.0, e_we=41, e_sn=41)
    baseline = _noah_baseline(40, 40)

    def urlopen(url, offset):
        raise SourceAbsent(f"{url} -> HTTP 404")

    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 fields="terrain")
    fields, receipt = apply_highres_statics(
        baseline, grid, config=config, domain_id=1,
        case_date=date(2021, 5, 4), landuse_attrs=MODIS21_ATTRS,
        urlopen=urlopen)

    assert receipt["status"] == "APPLIED"
    np.testing.assert_array_equal(fields["HGT_M"], baseline["HGT_M"])
    record = receipt["coverage"]["fields"]["terrain"]
    assert record["cells_outside_coverage"] == 40 * 40
    assert record["source"] is None
    assert receipt["fetch"]["terrain_tiles_absent"]
    assert receipt["cells_replaced"]["total"] == 0
    printed = capsys.readouterr().out
    assert sum("WARNING" in line for line in printed.splitlines()) == 1


# ---------------------------------------------------------------------------
# Plan resolution: partial reach builds, only disjoint refuses
# ---------------------------------------------------------------------------

def _straddling_grid() -> LambertGrid:
    """A domain whose footprint straddles the land-cover collection's
    northern envelope edge."""
    return LambertGrid(
        ref_lat=49.5, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=3000.0, dy=3000.0, e_we=41, e_sn=41)


def test_auto_takes_the_full_overlay_where_land_cover_reaches_part_of_it(
        tmp_path):
    bbox = domain_footprint(_straddling_grid(), HALO)
    assert bbox.lat_min < 49.5 < bbox.lat_max
    mode, coverage = _resolve_plan(
        HighresStaticConfig(enabled=True, cache_root=tmp_path,
                            landcover_source="annual-nlcd"), bbox)
    assert mode == "all"
    # 3DEP does not contain the footprint, so auto takes the global DEM.
    assert coverage.source_id == "copernicus-dem-glo30"


def test_a_named_source_partly_outside_its_envelope_is_built(tmp_path):
    bbox = domain_footprint(_straddling_grid(), HALO)
    mode, coverage = _resolve_plan(
        HighresStaticConfig(enabled=True, cache_root=tmp_path, fields="all",
                            terrain_source="usgs-3dep-13as",
                            landcover_source="annual-nlcd"), bbox)
    assert (mode, coverage.source_id) == ("all", "usgs-3dep-13as")


# ---------------------------------------------------------------------------
# The fetch and window layer
# ---------------------------------------------------------------------------

class _FakeResponse:
    status = 200

    def __init__(self, payload: bytes):
        self._payload = payload
        self._done = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        if self._done:
            return b""
        self._done = True
        return self._payload


def test_an_unstaged_3dep_tile_is_absent_not_a_refusal(tmp_path):
    """RETIRES test_fetch_three_dep_refuses_naming_missing_tiles."""
    def urlopen(url, offset):
        if "n40w099" in url:
            raise SourceAbsent(f"{url} -> HTTP 404")
        return _FakeResponse(b"elevation")

    bbox = FootprintBBox(lat_min=38.05, lat_max=39.31,
                         lon_min=-98.975, lon_max=-97.325)
    tiles, absent = fetch_three_dep_tiles(bbox, tmp_path, urlopen=urlopen)
    assert absent == ("n40w099",)
    assert len(tiles) == 3


def _fixture(name: str) -> Path:
    path = HIGHRES_FIXTURES / name
    if not path.is_file():
        pytest.skip(f"{name} not present in this checkout")
    return path


def _bridge_or_fail():
    from gpuwm.static import rust_bridge

    reason = rust_bridge.unavailable_reason()
    if reason is not None:
        pytest.fail(f"the Rust static-fields bridge is not loadable: {reason}")


def test_a_footprint_past_the_land_cover_raster_is_clipped_not_refused(
        tmp_path):
    """The land-cover clip used to refuse any footprint the raster's extent
    did not wholly contain, before a single cell was resampled."""
    _bridge_or_fail()
    from gpuwm.static.highres_fetch import derive_landcover_window

    source = _fixture("landcover.tif")
    staged = tmp_path / "landcover.tif"
    shutil.copyfile(source, staged)
    raster = FetchedFile(path=staged, url="file:landcover.tif",
                         sha256=sha256_file(staged),
                         bytes=staged.stat().st_size, fetched_utc="",
                         cache_hit=False)
    # The fixture window is a few kilometres across; this footprint runs
    # well past its east edge.
    bbox = FootprintBBox(lat_min=39.45, lat_max=39.55,
                         lon_min=-84.05, lon_max=-83.6)
    window = derive_landcover_window(raster, bbox, tmp_path / "cache")
    assert window.path.is_file()

    # Resampled onto a domain half past the raster, the cells beyond it
    # have no source pixel -- exactly what the overlay hands to the
    # baseline.
    meta = json.loads(_fixture("meta.json").read_text(encoding="utf-8"))
    case = meta["landcover"]
    spec = dict(case["grid_spec"])
    grid = LambertGrid(
        spec["ref_lat"], spec["ref_lon"], spec["truelat1"],
        spec["truelat2"], spec["stand_lon"], spec["dx"], spec["dy"],
        spec["e_we"] + 16, spec["e_sn"], known_x=spec["known_x"],
        known_y=spec["known_y"])
    bound = BoundRaster(
        path=window.path, sha256=window.sha256, source_id="fixture",
        role="landcover", source_url="https://example.invalid/source",
        license_id="test-only", license_url="https://example.invalid/l",
        nominal_resolution="30 m", nodata_override=float(case["nodata"]))
    fractions = highres_module.resample_mapped_categories(
        bound, grid,
        {int(raw): int(target) for raw, target in case["mapping"]},
        category_count=21)
    covered = np.all(np.isfinite(fractions), axis=0)
    assert covered.any() and not covered.all()


def test_the_global_terrain_window_keeps_unpublished_squares_as_no_data(
        tmp_path):
    """The window used to fill every hole with sea level, which put 0 m
    under any land an unpublished tile hid."""
    _bridge_or_fail()
    from gpuwm.static.highres_fetch import (derive_global_terrain_window,
                                            record_local_artifact)

    meta = json.loads(_fixture("meta.json").read_text(encoding="utf-8"))
    # Only the west square is published; the footprint runs on into the
    # east square, which is not.
    name = meta["mosaic"]["tiles"][0]
    staged = tmp_path / "tiles"
    staged.mkdir()
    shutil.copyfile(_fixture(name), staged / name)
    tiles = [record_local_artifact(staged / name, url=f"file:{name}")]
    bounds = meta["mosaic"]["bounds_wsen"]
    bbox = FootprintBBox(lat_min=bounds[1] + 0.01, lat_max=bounds[3] - 0.01,
                         lon_min=bounds[0] + 0.01, lon_max=bounds[2] - 0.01)

    _, kept = derive_global_terrain_window(
        tiles, bbox, tmp_path / "kept", sea_level_fill=None,
        resolution_deg=meta["mosaic"]["resolution_deg"])
    _, filled = derive_global_terrain_window(
        tiles, bbox, tmp_path / "filled", sea_level_fill=0.0,
        resolution_deg=meta["mosaic"]["resolution_deg"])

    holes = filled["sea_level_filled_pixels"]
    # The unpublished half of the window: about half its pixels.
    assert 0.3 * filled["total_pixels"] < holes < 0.7 * filled["total_pixels"]
    assert kept["sea_level_filled_pixels"] == 0
    assert kept["sea_level_fill_m"] is None
    assert kept["no_data_pixels_outside_coverage"] == holes


def test_an_unpublished_tile_over_land_takes_the_baseline_not_a_refusal(
        tmp_path):
    """RETIRES test_absent_tile_cross_check_counts_land_cells.

    One published terrain tile and one unpublished one over a domain the
    baseline calls land from edge to edge: this refused as
    ``terrain-tile-absent-over-land``.  Now the published square gives
    its relief, the unpublished square keeps the baseline's, and the two
    meet across the ramp.  Real tiles, the real Rust window and warp.
    """
    _bridge_or_fail()
    meta = json.loads(_fixture("meta.json").read_text(encoding="utf-8"))
    west_tile = meta["mosaic_clip_west"]["source_tile"]
    published = west_tile.split("_COG_10_")[1].removesuffix("_DEM.tif")
    payload = _fixture("mosaic_west.tif").read_bytes()

    def urlopen(url, offset):
        if published in url:
            return _FakeResponse(payload)
        raise SourceAbsent(f"{url} -> HTTP 404")

    grid = LambertGrid(
        ref_lat=46.46, ref_lon=8.0, truelat1=46.0, truelat2=47.0,
        stand_lon=8.0, dx=250.0, dy=250.0, e_we=17, e_sn=13)
    baseline = _noah_baseline(12, 16)
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 fields="terrain",
                                 terrain_source="copernicus-dem-glo30")
    fields, receipt = apply_highres_statics(
        baseline, grid, config=config, domain_id=1,
        case_date=date(2021, 5, 4), landuse_attrs=MODIS21_ATTRS,
        urlopen=urlopen)

    assert receipt["status"] == "APPLIED"
    assert receipt["fetch"]["terrain_tiles_absent"]
    record = receipt["coverage"]["fields"]["terrain"]
    assert 0 < record["cells_outside_coverage"] < 12 * 16
    hgt = fields["HGT_M"]
    lat, lon = grid.latlon_mass()
    east = lon > 8.01
    assert east.any()
    np.testing.assert_array_equal(hgt[east], 200.0)
    # The published square carries real alpine relief.
    assert hgt.max() > 2000.0


# ---------------------------------------------------------------------------
# The front door: a config that turns the overlay on reaches preparation
# ---------------------------------------------------------------------------

def test_a_config_carrying_the_overlay_block_loads_on_the_prepared_route(
        tmp_path, capsys):
    """``gpuwm go`` loads a prepared-route config through
    ``experiment_from_text``, which split off [fetch] and [case_data] but
    not [static]; any config with ``[static.highres]`` stopped there as an
    "unsplit" table before a byte was fetched.  The block is validated
    and split, and a bad key in it is still refused by name."""
    from gpuwm.cli import main as cli_main
    from gpuwm.domain_wizard import experiment_from_text

    out = tmp_path / "area.toml"
    assert cli_main(["domain", "--point=35.3,-97.5", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--hours", "6",
                     "--card", "12gb", "--out", str(out)]) == 0
    capsys.readouterr()
    text = out.read_text(encoding="utf-8")
    block = ('\n[static.highres]\nenabled = true\n'
             'cache_root = "highres-cache"\nfields = "auto"\n')

    exp = experiment_from_text(text + block, source=str(out))
    assert exp.domains

    with pytest.raises(ValueError, match="cache_roots"):
        experiment_from_text(
            text + block.replace("cache_root", "cache_roots"),
            source=str(out))
