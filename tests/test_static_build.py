"""Phase 3 Task 5: WPS_GEOG tile readers + static-field builder vs geo_em.

Authority: the WPS geog binary format (each dataset directory carries a
self-describing ASCII ``index`` file + flat big-endian binary tiles) and,
for every interpolation-convention ambiguity, the bundle geo_em files
(geogrid v4.6.0 output) arbitrate.

Plan gates (Task 5), all full-field vs both d01 and d04 ``geo_em`` files:
  HGT_M      RMSE < 5 m and max|delta| < 50 m
  LU_INDEX   >= 98 % cell agreement
  SCT_DOM / SCB_DOM   >= 95 % dominant-category agreement
  SOILCTOP / SOILCBOT fraction-vector mean|delta| <= 1e-2
  GREENFRAC / ALBEDO12M   monthly <= 1e-2 abs (geo_em native units)
  LANDMASK   >= 99.5 % agreement
"""
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.static.geog import GeogDataset, parse_index
from gpuwm.static.build import (
    _DomainSampler,
    GeogSelection,
    average_16pt,
    average_4pt,
    build_static,
    dominant_category,
    four_pt,
    geog_selection_from_catalog,
    landmask_from_landusef,
    lu_index_from_landusef,
    search_nearest,
    sixteen_pt,
    smth_desmth,
    smth_desmth_special,
)

BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
GEOG_ROOT = BUNDLE / "static" / "WPS_GEOG"
GEO_EM_DIR = BUNDLE / "geo_em"
NAMELIST_WPS = BUNDLE / "namelists" / "namelist.wps"

requires_bundle = pytest.mark.skipif(
    not GEOG_ROOT.is_dir() or not GEO_EM_DIR.is_dir(),
    reason="WRF_1974_MP55 reference bundle not present",
)


# ---------------------------------------------------------------------------
# Synthetic-dataset helpers (no bundle required)
# ---------------------------------------------------------------------------

def test_geog_selection_resolves_wps_tokens_under_case_data_root(tmp_path):
    wps = tmp_path / "namelist.wps"
    wps.write_text(
        "&share\n max_dom = 2,\n/\n&geogrid\n"
        " geog_data_res = 'default', 'modis_lai+default',\n/\n",
        encoding="utf-8")
    data = SimpleNamespace(geog_root=tmp_path / "GEOG",
                           wps_namelist=wps)

    root = GeogSelection.from_case_data(data, domain_id=1)
    child = GeogSelection.from_case_data(data, domain_id=2)
    assert root.resolution_tokens == ("default",)
    assert root.path("terrain") == (
        data.geog_root / "topo_gmted2010_30s")
    assert root.path("lai") == data.geog_root / "lai_modis_10m"
    assert child.resolution_tokens == ("modis_lai", "default")
    assert child.path("lai") == data.geog_root / "lai_modis_30s"
    # The no-selection legacy call is the exact historical inventory.
    assert GeogSelection.fallback(data.geog_root) == root


def test_catalog_recovers_child_geog_selection_without_bundle_inference(
        tmp_path):
    geog_root = tmp_path / "declared-geog"
    wps = tmp_path / "declared-namelist.wps"
    wps.write_text(
        "&share\n max_dom = 2,\n/\n&geogrid\n"
        " geog_data_res = 'default', 'modis_lai+default',\n/\n",
        encoding="utf-8")
    catalog = SimpleNamespace(files=(
        SimpleNamespace(role="wps_namelist", path=wps),
        SimpleNamespace(role="geog_index",
                        path=geog_root / "landuse" / "index"),
    ))

    child = geog_selection_from_catalog(catalog, 2)
    assert child.root == geog_root.resolve()
    assert child.resolution_tokens == ("modis_lai", "default")


def test_geog_selection_rejects_every_unrecognized_or_non_wps_alias(tmp_path):
    with pytest.raises(ValueError, match="unrecognized token"):
        GeogSelection.from_tokens(tmp_path, "unknown_resolution")
    with pytest.raises(ValueError, match="unrecognized token"):
        GeogSelection.from_tokens(tmp_path, "unknown_resolution+default")
    for alias in ("30s", "modis_30s"):
        with pytest.raises(ValueError, match="recognized:.*5m.*default.*modis_lai"):
            GeogSelection.from_tokens(tmp_path, alias)


def test_geog_selection_resolves_complete_global_five_minute_inventory(
        tmp_path):
    selection = GeogSelection.from_tokens(tmp_path, "5m")
    assert selection.resolution_tokens == ("5m",)
    assert selection.terrain == "topo_gmted2010_5m"
    assert selection.landuse == "modis_landuse_20class_5m_with_lakes"
    assert selection.soil_top == "soiltype_top_5m"
    assert selection.soil_bottom == "soiltype_bot_5m"
    assert selection.greenfrac == "greenfrac_fpar_modis_5m"
    assert selection.lai == "lai_modis_10m"
    assert selection.albedo == "albedo_modis"
    assert selection.snow_albedo == "maxsnowalb_modis"
    assert selection.soil_temperature == "soiltemp_1deg"

    # Resolution tokens retain WPS per-field priority: the LAI-specific
    # product wins for LAI while all other fields fall through to 5m.
    mixed = GeogSelection.from_tokens(tmp_path, "modis_lai+5m")
    assert mixed.lai == "lai_modis_30s"
    assert mixed.terrain == "topo_gmted2010_5m"


def test_geog_selection_uses_fortran_per_element_defaults_not_broadcast(
        tmp_path):
    wps = tmp_path / "namelist.wps"
    wps.write_text(
        "&share\n max_dom = 3,\n/\n&geogrid\n"
        " geog_data_res = 'modis_lai',\n/\n",
        encoding="utf-8")
    data = SimpleNamespace(geog_root=tmp_path / "GEOG", wps_namelist=wps)

    assert GeogSelection.from_case_data(
        data, domain_id=1).resolution_tokens == ("modis_lai",)
    # WPS v4.6 initializes every omitted Fortran-array element to default;
    # the single explicit d01 value is not broadcast to d02/d03.
    for domain_id in (2, 3):
        selection = GeogSelection.from_case_data(data, domain_id=domain_id)
        assert selection.resolution_tokens == ("default",)
        assert selection.lai == "lai_modis_10m"
    with pytest.raises(ValueError, match="exceeds.*max_dom=3"):
        GeogSelection.from_case_data(data, domain_id=4)

def _write_index(dirpath, **over):
    """Write a WPS-style index file; keyword args override the defaults."""
    kv = dict(type="continuous", signed="yes", projection="regular_ll",
              dx=1.0, dy=1.0, known_x=1.0, known_y=1.0,
              known_lat=-1.5, known_lon=0.5, wordsize=2,
              tile_x=4, tile_y=2, tile_z=1)
    kv.update(over)
    lines = [f"{k} = {v}" for k, v in kv.items() if v is not None]
    (dirpath / "index").write_text("\n".join(lines) + "\n")
    return kv


def _write_tiles(dirpath, data, kv, bdr=0):
    """Write ``data`` (nz, ny_glob, nx_glob) as WPS binary tiles.

    Border cells (if bdr > 0) are filled with a sentinel that must never be
    read back (asserts the reader crops tile borders).
    """
    nz, nyg, nxg = data.shape
    tx, ty = int(kv["tile_x"]), int(kv["tile_y"])
    signed = str(kv.get("signed", "no")).lower() in ("yes", ".true.", "true")
    ws = int(kv["wordsize"])
    base = {1: "i1", 2: "i2", 4: "i4"}[ws]
    if not signed:
        base = "u" + base[1:]
    dt = np.dtype(">" + base) if ws > 1 else np.dtype(base)
    for ys in range(1, nyg + 1, ty):
        for xs in range(1, nxg + 1, tx):
            tile = np.full((nz, ty + 2 * bdr, tx + 2 * bdr), 99, dtype=dt)
            core = data[:, ys - 1:ys - 1 + ty, xs - 1:xs - 1 + tx]
            if bdr:
                tile[:, bdr:-bdr, bdr:-bdr] = core
            else:
                tile[:] = core
            name = (f"{xs:05d}-{xs + tx - 1:05d}."
                    f"{ys:05d}-{ys + ty - 1:05d}")
            tile.astype(dt).tofile(dirpath / name)


def _synthetic(tmp_path, nz=1, nxg=8, nyg=4, bdr=0, **over):
    """Global 8x4 (45 deg) synthetic dataset with value = x + 100*y + 1000*z."""
    over.setdefault("dx", 45.0)
    over.setdefault("dy", 45.0)
    over.setdefault("known_lat", -67.5)
    over.setdefault("known_lon", -157.5)
    if bdr:
        over["tile_bdr"] = bdr
    if nz > 1:
        over["tile_z"] = nz
    kv = _write_index(tmp_path, **over)
    z, y, x = np.meshgrid(np.arange(nz), np.arange(1, nyg + 1),
                          np.arange(1, nxg + 1), indexing="ij")
    data = x + 100 * y + 1000 * z
    _write_tiles(tmp_path, data, kv, bdr=bdr)
    return GeogDataset(tmp_path)


# ---------------------------------------------------------------------------
# Index parser
# ---------------------------------------------------------------------------

def test_parse_index_keys_and_defaults(tmp_path):
    (tmp_path / "index").write_text(
        "type = continuous\n"
        "signed = yes\n"
        "projection = regular_ll\n"
        "dx = 0.00833333\n"
        "dy = 0.00833333\n"
        "known_x = 1.0\n"
        "known_y = 1.0\n"
        "known_lat = -89.99583\n"
        "known_lon = 0.004166667\n"
        "wordsize = 2\n"
        "tile_x = 1200\n"
        "tile_y = 1200\n"
        "tile_z = 1\n"
        "tile_bdr=3\n"
        'units="meters MSL"\n'
        'description="GMTED2010 30-arc-second topography height"\n')
    idx = parse_index(tmp_path / "index")
    assert idx.type == "continuous"
    assert idx.signed is True
    assert idx.wordsize == 2
    assert idx.tile_x == 1200 and idx.tile_y == 1200
    assert idx.tile_bdr == 3
    assert idx.nz == 1
    assert idx.scale_factor == 1.0          # default
    assert idx.missing_value is None        # default
    assert idx.endian == "big"              # default
    assert idx.units == "meters MSL"        # quotes stripped
    assert abs(idx.known_lon - 0.004166667) < 1e-12


def test_parse_index_z_range_and_categories(tmp_path):
    (tmp_path / "index").write_text(
        "type=categorical\ncategory_min=1\ncategory_max=21\n"
        "projection=regular_ll\ndx=0.05\ndy=-0.05\n"
        "known_x=1.0\nknown_y=1.0\nknown_lat=89.975\nknown_lon=-179.975\n"
        "wordsize=2\ntile_x=1200\ntile_y=1200\n"
        "tile_z_start=1\ntile_z_end=12\n"
        "scale_factor=0.01\nmissing_value=-999\nsigned = yes\n")
    idx = parse_index(tmp_path / "index")
    assert idx.type == "categorical"
    assert idx.category_min == 1 and idx.category_max == 21
    assert idx.nz == 12
    assert idx.scale_factor == 0.01
    assert idx.missing_value == -999.0
    assert idx.dy == -0.05 and idx.known_lat == 89.975


def test_parse_index_preserves_optional_interp_default(tmp_path):
    _write_index(tmp_path, interp_option="four_pt+average_4pt+search")
    idx = parse_index(tmp_path / "index")
    assert idx.interp_option == "four_pt+average_4pt+search"


# ---------------------------------------------------------------------------
# Mosaic reader (synthetic datasets)
# ---------------------------------------------------------------------------

def test_dataset_scan_dims_and_wrap_flag(tmp_path):
    ds = _synthetic(tmp_path)
    assert ds.nx_global == 8 and ds.ny_global == 4
    assert ds.wraps_x            # 8 * 45 deg == 360
    assert ds.index.nz == 1


def test_read_window_across_tile_seams(tmp_path):
    ds = _synthetic(tmp_path)
    w = ds.read_window(2, 7, 1, 4)      # spans both tile columns and rows
    v = w.values(0)
    assert v.shape == (4, 6)
    x, y = np.meshgrid(np.arange(2, 8), np.arange(1, 5))
    np.testing.assert_array_equal(v, (x + 100 * y).astype(np.float64))


def test_read_window_scale_and_missing(tmp_path):
    ds = _synthetic(tmp_path, scale_factor=0.5, missing_value=105.0)
    w = ds.read_window(1, 8, 1, 4)
    v = w.values(0)
    # raw value 105 (x=5, y=1) masked BEFORE scaling
    assert np.isnan(v[0, 4])
    assert v[0, 0] == pytest.approx(0.5 * 101)
    assert np.isnan(v).sum() == 1


def test_domain_static_coverage_binds_complete_required_tiles(tmp_path):
    ds = _synthetic(tmp_path, missing_value=105.0)
    window = ds.read_window(1, 8, 1, 4)

    evidence = _DomainSampler.require_source_coverage(
        ds, window, field="terrain")

    assert evidence["status"] == "PASS"
    assert evidence["field"] == "terrain"
    assert evidence["required_cells"] == evidence["covered_cells"] == 32
    assert evidence["coverage_fraction"] == 1.0
    assert evidence["required_tile_count"] == 4
    assert len(evidence["required_tiles"]) == 4
    # A source missing-value sentinel inside a present tile is intentionally
    # distinct from an absent staged tile and retains WPS fallback semantics.
    assert np.isnan(window.values(0)).sum() == 1


def test_domain_static_coverage_rejects_absent_sparse_required_tile(tmp_path):
    _synthetic(tmp_path)
    missing = tmp_path / "00001-00004.00001-00002"
    missing.unlink()
    ds = GeogDataset(tmp_path, sparse=True)
    window = ds.read_window(1, 8, 1, 4)

    with pytest.raises(
            FileNotFoundError,
            match=(r"mandatory source coverage failed.*terrain.*covered "
                   r"24/32.*missing_tile_origins=\[\[1, 1\]\].*"
                   r"declared_sparse=True")):
        _DomainSampler.require_source_coverage(
            ds, window, field="terrain")


def test_staged_regular_ll_inventory_retains_global_geometry(tmp_path):
    _synthetic(tmp_path)
    for tile in tuple(tmp_path.iterdir()):
        if tile.name.startswith(("00001-00004",)):
            tile.unlink()

    ds = GeogDataset(tmp_path)

    assert ds.tile_inventory_bounds == (5, 8, 1, 4)
    assert (ds.nx_global, ds.ny_global) == (8, 4)
    assert ds.wraps_x
    assert ds.extent_basis == "regular_ll_staged_inventory"
    x, y = ds.latlon_to_xy(-67.5, -157.5)
    assert x == pytest.approx(1.0) and y == pytest.approx(1.0)
    assert not ds.tile_coverage_mask(1, 4, 1, 4).any()
    assert ds.tile_coverage_mask(5, 8, 1, 4).all()


def test_read_window_wraps_in_x(tmp_path):
    ds = _synthetic(tmp_path)
    w = ds.read_window(7, 10, 2, 3)     # x = 7, 8, 1, 2 after wrap
    v = w.values(0)
    exp_x = np.array([7, 8, 1, 2])
    x, y = np.meshgrid(exp_x, np.arange(2, 4))
    np.testing.assert_array_equal(v, (x + 100 * y).astype(np.float64))


def test_read_window_crops_tile_border(tmp_path):
    ds = _synthetic(tmp_path, bdr=1)
    v = ds.read_window(1, 8, 1, 4).values(0)
    x, y = np.meshgrid(np.arange(1, 9), np.arange(1, 5))
    np.testing.assert_array_equal(v, (x + 100 * y).astype(np.float64))


def test_read_tile_window_retains_interpolation_border(tmp_path):
    ds = _synthetic(tmp_path, bdr=1)
    w = ds.read_tile_window(1, 1)
    assert w is not None
    assert (w.x0, w.y0, w.raw.shape) == (0, 0, (1, 4, 6))
    np.testing.assert_array_equal(w.raw[0, 0], 99)
    np.testing.assert_array_equal(w.raw[0, -1], 99)


def test_read_window_multilevel(tmp_path):
    ds = _synthetic(tmp_path, nz=3)
    w = ds.read_window(3, 6, 2, 3)
    for z in range(3):
        v = w.values(z)
        x, y = np.meshgrid(np.arange(3, 7), np.arange(2, 4))
        np.testing.assert_array_equal(v, (x + 100 * y + 1000 * z).astype(float))


def test_read_window_discards_complete_zero_padded_z_planes(tmp_path):
    kv = _write_index(
        tmp_path, dx=45.0, dy=45.0, known_lat=-67.5,
        known_lon=-157.5, wordsize=1, signed="no", tile_z=2)
    declared = np.arange(2 * 4 * 8, dtype=np.uint8).reshape(2, 4, 8)
    padded = np.concatenate((declared, np.zeros((2, 4, 8), dtype=np.uint8)))
    _write_tiles(tmp_path, padded, kv)

    window = GeogDataset(tmp_path).read_window(1, 8, 1, 4)

    assert window.raw.shape == declared.shape
    np.testing.assert_array_equal(window.raw, declared)


def test_read_window_rejects_nonzero_undeclared_z_plane(tmp_path):
    kv = _write_index(
        tmp_path, dx=45.0, dy=45.0, known_lat=-67.5,
        known_lon=-157.5, wordsize=1, signed="no", tile_z=2)
    data = np.zeros((3, 4, 8), dtype=np.uint8)
    data[2, 1, 2] = 7
    _write_tiles(tmp_path, data, kv)

    with pytest.raises(
            ValueError,
            match=r"index declares 2; undeclared trailing planes contain nonzero"):
        GeogDataset(tmp_path).read_window(1, 8, 1, 4)


def test_read_window_rejects_partial_undeclared_z_plane(tmp_path):
    kv = _write_index(
        tmp_path, dx=45.0, dy=45.0, known_lat=-67.5,
        known_lon=-157.5, wordsize=1, signed="no", tile_z=2)
    declared = np.zeros((2, 4, 8), dtype=np.uint8)
    _write_tiles(tmp_path, declared, kv)
    tile = next(path for path in tmp_path.iterdir() if path.name != "index")
    with tile.open("ab") as handle:
        handle.write(b"\x00")

    with pytest.raises(ValueError, match=r"has 17 words, expected 16"):
        GeogDataset(tmp_path).read_window(1, 8, 1, 4)


def test_continuous_category_planes_are_interpolated_and_normalized():
    class FractionalSampler:
        nxe = 2
        nye = 1

        def __init__(self):
            self.calls = []
            self.fields = (
                np.array([[25.0, 0.0]]),
                np.array([[75.0, 0.0]]),
            )

        def continuous(self, ds, win, z=0, seq=(), fill=None,
                       gcell=None, active=None):
            self.calls.append((z, seq, fill, gcell))
            return self.fields[z]

    sampler = FractionalSampler()
    ds = SimpleNamespace(index=SimpleNamespace(
        type="continuous", category_min=3, category_max=4, nz=2))

    frac = _DomainSampler.categorical(
        sampler, ds, SimpleNamespace(), fractional_gcell=True)

    np.testing.assert_array_equal(
        frac, np.array([[[0.25, 0.0]], [[0.75, 0.0]]]))
    assert sampler.calls == [
        (0, ("four_pt",), 0.0, True),
        (1, ("four_pt",), 0.0, True),
    ]


def test_continuous_category_plane_count_must_match_category_range():
    ds = SimpleNamespace(index=SimpleNamespace(
        type="continuous", category_min=1, category_max=16, nz=24))

    with pytest.raises(
            ValueError,
            match=r"declares 24 z planes for 16 categories \(1\.\.16\)"):
        _DomainSampler.categorical(object(), ds, SimpleNamespace())


def test_scalar_categorical_source_keeps_count_accumulation_path():
    class ScalarSampler:
        nxe = 2
        nye = 1

        @staticmethod
        def pixel_cells(ds, win):
            return np.array([0, 0, 1, 1], dtype=np.int64)

        @staticmethod
        def cell_coords(ds, win):
            raise AssertionError("nonempty cells must not need interpolation")

    ds = SimpleNamespace(index=SimpleNamespace(
        type="categorical", category_min=1, category_max=2, nz=1))
    win = SimpleNamespace(raw=np.array([[[1, 2, 2, 2]]], dtype=np.uint8))

    frac = _DomainSampler.categorical(ScalarSampler(), ds, win)

    np.testing.assert_array_equal(
        frac, np.array([[[0.5, 0.0]], [[0.5, 1.0]]]))


def test_signed_negative_values_bigendian(tmp_path):
    kv = _write_index(tmp_path, dx=45.0, dy=45.0, known_lat=-67.5,
                      known_lon=-157.5)
    data = -(np.arange(32).reshape(1, 4, 8) + 1)
    _write_tiles(tmp_path, data, kv)
    ds = GeogDataset(tmp_path)
    v = ds.read_window(1, 8, 1, 4).values(0)
    np.testing.assert_array_equal(v, data[0].astype(np.float64))


def test_latlon_xy_roundtrip(tmp_path):
    ds = _synthetic(tmp_path)
    lat = np.array([-67.5, 0.0, 60.0])
    lon = np.array([-157.5, 12.0, 170.0])
    x, y = ds.latlon_to_xy(lat, lon)
    lat2, lon2 = ds.xy_to_latlon(x, y)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)
    np.testing.assert_allclose(lon2, lon, atol=1e-9)
    # known point (1,1) sits at (known_lat, known_lon)
    x0, y0 = ds.latlon_to_xy(-67.5, -157.5)
    assert x0 == pytest.approx(1.0) and y0 == pytest.approx(1.0)


def test_latlon_xy_negative_dy(tmp_path):
    """Top-down datasets (albedo style): dy < 0, known_lat at the north."""
    ds = _synthetic(tmp_path, dy=-45.0, known_lat=67.5)
    x, y = ds.latlon_to_xy(67.5, -157.5)
    assert x == pytest.approx(1.0) and y == pytest.approx(1.0)
    x, y = ds.latlon_to_xy(67.5 - 45.0, -157.5)
    assert y == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Interpolators (unit)
# ---------------------------------------------------------------------------

def test_four_pt_is_bilinear():
    vals = np.arange(20, dtype=np.float64).reshape(4, 5)  # f = x + 5*y (0-based)
    xi = np.array([2.25, 1.0]); yi = np.array([1.5, 2.0])
    out = four_pt(vals, xi, yi, x0=1, y0=1)   # window origin at (1,1)
    np.testing.assert_allclose(out, [(2.25 - 1) + 5 * (1.5 - 1),
                                     0.0 + 5.0], atol=1e-12)


def test_four_pt_nan_propagates_and_average_4pt_recovers():
    vals = np.full((3, 3), 2.0)
    vals[1, 1] = np.nan
    out4 = four_pt(vals, np.array([1.5]), np.array([1.5]), x0=1, y0=1)
    assert np.isnan(out4[0])
    outa = average_4pt(vals, np.array([1.5]), np.array([1.5]), x0=1, y0=1)
    np.testing.assert_allclose(outa, [2.0])


def test_four_pt_exact_coordinate_ignores_unused_missing_neighbors():
    vals = np.array([[7.0, np.nan], [np.nan, np.nan]])
    out = four_pt(vals, np.array([1.0]), np.array([1.0]), x0=1, y0=1)
    np.testing.assert_array_equal(out, [7.0])


def test_sixteen_pt_exact_on_quadratic():
    x, y = np.meshgrid(np.arange(8, dtype=float), np.arange(8, dtype=float))
    vals = 2.0 + 0.5 * x + 0.25 * y + 0.125 * x * x + 0.0625 * y * y
    xi = np.array([3.3]); yi = np.array([4.7])   # 1-based source coords
    out = sixteen_pt(vals, xi, yi, x0=1, y0=1)
    xt, yt = 3.3 - 1, 4.7 - 1                     # 0-based truth
    truth = 2.0 + 0.5 * xt + 0.25 * yt + 0.125 * xt * xt + 0.0625 * yt * yt
    np.testing.assert_allclose(out, [truth], atol=1e-12)


def test_sixteen_pt_missing_falls_through():
    vals = np.ones((8, 8)); vals[2, 2] = np.nan
    out = sixteen_pt(vals, np.array([3.5]), np.array([3.5]), x0=1, y0=1)
    assert np.isnan(out[0])


def test_search_nearest_finds_closest_valid():
    vals = np.full((5, 7), np.nan)
    vals[0, 0] = 5.0
    vals[4, 6] = 9.0
    out = search_nearest(vals, np.array([2.0, 6.0]), np.array([1.0, 4.0]),
                         x0=1, y0=1)
    np.testing.assert_allclose(out, [5.0, 9.0])


def test_search_nearest_follows_wps_breadth_first_frontier():
    vals = np.full((5, 5), np.nan)
    vals[2, 1] = 5.0   # first valid point on the Manhattan-radius-1 frontier
    vals[3, 3] = 9.0   # Euclidean-closer, but on the next BFS frontier
    out = search_nearest(vals, np.array([3.49]), np.array([3.49]), x0=1, y0=1)
    np.testing.assert_array_equal(out, [5.0])


def test_average_16pt_uses_all_valid_points():
    vals = np.arange(36, dtype=np.float64).reshape(6, 6)
    vals[1, 1] = np.nan
    out = average_16pt(vals, np.array([3.25]), np.array([3.75]), x0=1, y0=1)
    np.testing.assert_allclose(out, [np.nanmean(vals[1:5, 1:5])])


# ---------------------------------------------------------------------------
# Smoothers (unit)
# ---------------------------------------------------------------------------

def _reference_one_pass(a, coef):
    """Loop-based reference: x-sweep then y-sweep, boundaries untouched."""
    out = a.astype(np.float64).copy()
    ny, nx = a.shape
    for j in range(ny):
        for i in range(1, nx - 1):
            out[j, i] = a[j, i] + coef * (0.5 * (a[j, i - 1] + a[j, i + 1])
                                          - a[j, i])
    mid = out.copy()
    for j in range(1, ny - 1):
        for i in range(nx):
            out[j, i] = mid[j, i] + coef * (0.5 * (mid[j - 1, i]
                                                   + mid[j + 1, i])
                                            - mid[j, i])
    return out


def test_smth_desmth_matches_loop_reference():
    rng = np.random.default_rng(7)
    a = rng.uniform(0.0, 100.0, size=(9, 11))
    ref = _reference_one_pass(_reference_one_pass(a, 0.5), -0.52)
    np.testing.assert_allclose(smth_desmth(a, passes=1), ref, atol=1e-12)


def test_smth_desmth_preserves_constant_and_plane():
    c = np.full((6, 7), 42.0)
    np.testing.assert_allclose(smth_desmth(c), c, atol=1e-12)
    x, y = np.meshgrid(np.arange(7.0), np.arange(6.0))
    p = 3.0 * x - 2.0 * y + 1.0
    np.testing.assert_allclose(smth_desmth(p), p, atol=1e-12)


def test_smth_desmth_damps_checkerboard():
    x, y = np.meshgrid(np.arange(12), np.arange(10))
    cb = ((-1.0) ** (x + y)) * 5.0
    out = smth_desmth(cb, passes=1)
    assert np.abs(out[2:-2, 2:-2]).max() < np.abs(cb[2:-2, 2:-2]).max() * 0.1


def test_smth_desmth_special_restores_new_negative_values():
    a = np.zeros((7, 7), dtype=np.float64)
    a[3, 3] = 100.0
    a[1, 1] = -25.0
    ordinary = smth_desmth(a)
    special = smth_desmth_special(a)
    new_negative = (ordinary < 0.0) & (a >= 0.0)
    preexisting_negative = a < 0.0
    assert np.any(new_negative)
    assert np.any(preexisting_negative)
    np.testing.assert_array_equal(special[new_negative], a[new_negative])
    np.testing.assert_array_equal(
        special[preexisting_negative], ordinary[preexisting_negative])


# ---------------------------------------------------------------------------
# Dominant-category / landmask rules (unit; conventions pinned by geo_em)
# ---------------------------------------------------------------------------

def test_landmask_water_at_half_fraction():
    luf = np.zeros((21, 1, 3))
    luf[16, 0, 0] = 0.5             # exactly half water -> water (geo_em pin)
    luf[0, 0, 0] = 0.5
    luf[16, 0, 1] = 0.49; luf[1, 0, 1] = 0.51
    luf[20, 0, 2] = 0.6; luf[2, 0, 2] = 0.4
    lm = landmask_from_landusef(luf, iswater=17, islake=21)
    np.testing.assert_array_equal(lm[0], [0.0, 1.0, 0.0])


def test_lu_index_rules():
    luf = np.zeros((21, 1, 4))
    # water cell, lake > ocean -> 21
    luf[16, 0, 0] = 0.2; luf[20, 0, 0] = 0.4; luf[3, 0, 0] = 0.4
    # water cell, tie lake == ocean -> 17
    luf[16, 0, 1] = 0.3; luf[20, 0, 1] = 0.3; luf[3, 0, 1] = 0.4
    # land cell: dominant LAND category even if water single-largest
    luf[16, 0, 2] = 0.4; luf[4, 0, 2] = 0.35; luf[5, 0, 2] = 0.25
    # land tie between categories 5 and 8 -> lowest index wins
    luf[4, 0, 3] = 0.5; luf[7, 0, 3] = 0.5
    lm = landmask_from_landusef(luf, iswater=17, islake=21)
    lu = lu_index_from_landusef(luf, lm, iswater=17, islake=21)
    np.testing.assert_array_equal(lm[0], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(lu[0], [21, 17, 5, 5])


def test_dominant_category_plain_argmax_lowest_tie():
    f = np.zeros((16, 1, 2))
    f[13, 0, 0] = 0.9; f[0, 0, 0] = 0.1     # water soil dominates -> 14
    f[2, 0, 1] = 0.5; f[10, 0, 1] = 0.5     # tie -> lowest (3)
    np.testing.assert_array_equal(dominant_category(f)[0], [14, 3])


# ---------------------------------------------------------------------------
# Bundle-gated: reader plausibility on the real WPS_GEOG tree
# ---------------------------------------------------------------------------

@requires_bundle
def test_topo_reader_pikes_peak():
    ds = GeogDataset(GEOG_ROOT / "topo_gmted2010_30s")
    assert ds.nx_global == 43200 and ds.ny_global == 21600
    x, y = ds.latlon_to_xy(38.8409, -105.0423)      # Pikes Peak
    x, y = int(round(float(x))), int(round(float(y)))
    v = ds.read_window(x - 3, x + 3, y - 3, y + 3).values(0)
    assert np.isfinite(v).all()
    assert 3500.0 < v.max() < 4600.0


@requires_bundle
def test_landuse_and_soil_water_in_atlantic():
    lu = GeogDataset(GEOG_ROOT / "modis_landuse_20class_30s_with_lakes")
    assert lu.index.iswater == 17 and lu.index.islake == 21
    x, y = lu.latlon_to_xy(30.0, -70.0)
    v = lu.read_window(int(x), int(x) + 4, int(y), int(y) + 4).values(0)
    np.testing.assert_array_equal(v, 17.0)
    st = GeogDataset(GEOG_ROOT / "soiltype_top_30s")
    x, y = st.latlon_to_xy(30.0, -70.0)
    v = st.read_window(int(x), int(x) + 4, int(y), int(y) + 4).values(0)
    np.testing.assert_array_equal(v, 14.0)


# ---------------------------------------------------------------------------
# Bundle-gated: full d01 build vs geo_em (the plan's acceptance gates)
# ---------------------------------------------------------------------------

_D01_VARS = ("HGT_M", "LANDMASK", "LU_INDEX", "LANDUSEF", "SOILCTOP",
             "SCT_DOM", "SOILCBOT", "SCB_DOM", "GREENFRAC", "ALBEDO12M",
             "LAI12M", "SNOALB", "SOILTEMP")


@lru_cache(maxsize=2)
def _geo(dom):
    import netCDF4
    with netCDF4.Dataset(GEO_EM_DIR / f"geo_em.d{dom:02d}.nc") as ds:
        return {n: np.asarray(ds.variables[n][0], dtype=np.float64)
                for n in _D01_VARS}


@lru_cache(maxsize=2)
def _grid(dom):
    from gpuwm.static.lambert import grids_from_wps_namelist
    return grids_from_wps_namelist(NAMELIST_WPS)[dom - 1]


@pytest.fixture(scope="module")
def d01():
    if not GEOG_ROOT.is_dir() or not GEO_EM_DIR.is_dir():
        pytest.skip("bundle not present")
    return build_static(_grid(1), GEOG_ROOT)


@pytest.fixture(scope="module")
def d04():
    if not GEOG_ROOT.is_dir() or not GEO_EM_DIR.is_dir():
        pytest.skip("bundle not present")
    return build_static(_grid(4), GEOG_ROOT)


@requires_bundle
def test_d01_field_shapes_and_dtypes(d01):
    ny, nx = 200, 250
    assert d01["HGT_M"].shape == (ny, nx)
    assert d01["LANDUSEF"].shape == (21, ny, nx)
    assert d01["SOILCTOP"].shape == (16, ny, nx)
    assert d01["GREENFRAC"].shape == (12, ny, nx)
    assert d01["ALBEDO12M"].shape == (12, ny, nx)
    assert d01["LAI12M"].shape == (12, ny, nx)
    for name in ("HGT_M", "LANDMASK", "TMN", "SOILTEMP", "SNOALB"):
        assert d01[name].dtype == np.float64, name
        assert np.isfinite(d01[name]).all(), name


@requires_bundle
def test_d01_hgt_gate(d01):
    delta = d01["HGT_M"] - _geo(1)["HGT_M"]
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    mx = float(np.abs(delta).max())
    assert rmse < 5.0, f"HGT_M RMSE {rmse:.3f} m"
    assert mx < 50.0, f"HGT_M max|delta| {mx:.2f} m"


@requires_bundle
def test_d01_landmask_gate(d01):
    agree = float((d01["LANDMASK"] == _geo(1)["LANDMASK"]).mean())
    assert agree >= 0.995, f"LANDMASK agreement {agree:.4f}"


@requires_bundle
def test_d01_lu_index_gate(d01):
    agree = float((d01["LU_INDEX"] == _geo(1)["LU_INDEX"]).mean())
    assert agree >= 0.98, f"LU_INDEX agreement {agree:.4f}"


@requires_bundle
def test_d01_landusef_fractions_close(d01):
    err = np.abs(d01["LANDUSEF"] - _geo(1)["LANDUSEF"])
    assert float(err.mean()) < 5e-3
    assert float(err.max()) < 0.15      # secondary: fractions, not a plan gate


@requires_bundle
def test_d01_soil_gates(d01):
    ref = _geo(1)
    top = float((d01["SCT_DOM"] == ref["SCT_DOM"]).mean())
    bot = float((d01["SCB_DOM"] == ref["SCB_DOM"]).mean())
    assert top >= 0.95, f"SCT_DOM agreement {top:.4f}"
    assert bot >= 0.95, f"SCB_DOM agreement {bot:.4f}"
    top_frac = float(np.abs(d01["SOILCTOP"] - ref["SOILCTOP"]).mean())
    bot_frac = float(np.abs(d01["SOILCBOT"] - ref["SOILCBOT"]).mean())
    assert top_frac <= 1e-2, f"SOILCTOP fraction mean|delta| {top_frac:.6f}"
    assert bot_frac <= 1e-2, f"SOILCBOT fraction mean|delta| {bot_frac:.6f}"


@requires_bundle
def test_d01_greenfrac_gate(d01):
    err = np.abs(d01["GREENFRAC"] - _geo(1)["GREENFRAC"])
    worst = float(err.max())
    assert worst <= 1e-2, f"GREENFRAC worst monthly max abs {worst:.4f}"


@requires_bundle
def test_d01_albedo_gate(d01):
    err = np.abs(d01["ALBEDO12M"] - _geo(1)["ALBEDO12M"])
    worst = float(err.max())
    assert worst <= 1e-2, f"ALBEDO12M worst monthly max abs {worst:.4f}"


@requires_bundle
def test_d01_lai_close(d01):
    err = np.abs(d01["LAI12M"] - _geo(1)["LAI12M"])
    assert float(err.max()) <= 5e-2  # secondary (not plan gate)


@requires_bundle
def test_d01_snoalb_close(d01):
    err = np.abs(d01["SNOALB"] - _geo(1)["SNOALB"])
    assert float(np.sqrt(np.mean(err ** 2))) <= 5e-3
    assert float(err.max()) <= 5e-2   # secondary gate (not in plan)


@requires_bundle
def test_d01_soiltemp_close(d01):
    ref = _geo(1)["SOILTEMP"]
    err = np.abs(d01["SOILTEMP"] - ref)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    assert rmse <= 0.5, f"SOILTEMP full-field RMSE {rmse:.3f} K"


@requires_bundle
def test_d01_water_fills_pinned_by_oracle(d01):
    """Masked-field fills: ALBEDO12M=8, GREENFRAC/LAI12M/SNOALB/SOILTEMP=0
    over oracle water, exactly as geogrid wrote them."""
    water = _geo(1)["LANDMASK"] == 0
    assert np.all(d01["ALBEDO12M"][:, water] == 8.0)
    assert np.all(d01["GREENFRAC"][:, water] == 0.0)
    assert np.all(d01["LAI12M"][:, water] == 0.0)
    assert np.all(d01["SNOALB"][water] == 0.0)
    assert np.all(d01["SOILTEMP"][water] == 0.0)


@requires_bundle
def test_d01_tmn_elevation_correction(d01):
    """TMN = SOILTEMP - 0.0065*HGT_M on land (module_soil_pre.F:973),
    untouched over water."""
    land = d01["LANDMASK"] > 0.5
    exp = d01["SOILTEMP"] - 0.0065 * d01["HGT_M"]
    np.testing.assert_allclose(d01["TMN"][land], exp[land], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(d01["TMN"][~land], d01["SOILTEMP"][~land])


# ---------------------------------------------------------------------------
# Bundle-gated: d04 spot-check (source coarser than grid -> interp path)
# ---------------------------------------------------------------------------

@requires_bundle
def test_d04_hgt_spot_gate(d04):
    delta = d04["HGT_M"] - _geo(4)["HGT_M"]
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    mx = float(np.abs(delta).max())
    assert rmse < 5.0, f"d04 HGT_M RMSE {rmse:.3f} m"
    assert mx < 50.0, f"d04 HGT_M max|delta| {mx:.2f} m"


@requires_bundle
def test_d04_lu_index_spot_gate(d04):
    agree = float((d04["LU_INDEX"] == _geo(4)["LU_INDEX"]).mean())
    assert agree >= 0.98, f"d04 LU_INDEX agreement {agree:.4f}"


@requires_bundle
def test_d04_landmask_spot_gate(d04):
    agree = float((d04["LANDMASK"] == _geo(4)["LANDMASK"]).mean())
    assert agree >= 0.995, f"d04 LANDMASK agreement {agree:.4f}"


@requires_bundle
def test_d04_soil_spot_gates(d04):
    ref = _geo(4)
    top = float((d04["SCT_DOM"] == ref["SCT_DOM"]).mean())
    bot = float((d04["SCB_DOM"] == ref["SCB_DOM"]).mean())
    assert top >= 0.95, f"d04 SCT_DOM agreement {top:.4f}"
    assert bot >= 0.95, f"d04 SCB_DOM agreement {bot:.4f}"
    top_frac = float(np.abs(d04["SOILCTOP"] - ref["SOILCTOP"]).mean())
    bot_frac = float(np.abs(d04["SOILCBOT"] - ref["SOILCBOT"]).mean())
    assert top_frac <= 1e-2, f"d04 SOILCTOP fraction mean|delta| {top_frac:.6f}"
    assert bot_frac <= 1e-2, f"d04 SOILCBOT fraction mean|delta| {bot_frac:.6f}"


@requires_bundle
def test_d04_greenfrac_spot_gate(d04):
    worst = float(np.abs(d04["GREENFRAC"] - _geo(4)["GREENFRAC"]).max())
    assert worst <= 1e-2, f"d04 GREENFRAC full-field max|delta| {worst:.4f}"


@requires_bundle
def test_d04_albedo_spot_gate(d04):
    worst = float(np.abs(d04["ALBEDO12M"] - _geo(4)["ALBEDO12M"]).max())
    assert worst <= 1e-2, f"d04 ALBEDO12M full-field max|delta| {worst:.4f}"
