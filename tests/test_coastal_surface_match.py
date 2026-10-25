"""Coastal matching runs through the built Rust preprocessing artifact."""

from datetime import datetime

import numpy as np
import pytest

from gpuwm.ingest.cpu_backend import CpuPreprocessBackend, SURFACE_MATCH_COUNT_KEYS
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.horiz import (
    _coastal_regular_geometry, _WPS_FULL_CHAIN, _WPS_SNOW_CHAIN,
    interpolate_era5_to_lambert,
)
from gpuwm.static.lambert import LambertGrid


def _backend():
    try:
        backend = CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f"native CPU bridge is not built: {error}")
    if not backend.surface_match_entry:
        pytest.fail("the selected Rust bridge lacks gpuwm_surface_match_f32")
    return backend


def _geometry(y, x, source_shape, method="parabolic"):
    y = np.asarray(y, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    iy = np.floor(y).astype(np.int32)
    ix = np.floor(x).astype(np.int32)
    if method == "bilinear":
        iy = np.minimum(iy, source_shape[0] - 2)
        ix = np.minimum(ix, source_shape[1] - 2)
    return iy, ix, y - iy.astype(np.float32), x - ix.astype(np.float32)


@pytest.mark.parametrize("surface", [False, True])
@pytest.mark.parametrize("method", ["bilinear", "parabolic"])
def test_uniform_support_preserves_original_bytes(surface, method):
    backend = _backend()
    rng = np.random.default_rng(781)
    source = rng.uniform(280, 310, (2, 7, 8)).astype(np.float32)
    y = np.array([[1.0, 2.125], [4.75, 6.0]], dtype=np.float32)
    x = np.array([[2.0, 3.375], [5.875, 7.0]], dtype=np.float32)
    original = backend.regular_plan(
        np.arange(7), np.arange(8), y, x).apply(source, method=method)
    answer, counts = backend.surface_match(
        source, original, np.full((7, 8), surface),
        *_geometry(y, x, (7, 8), method), np.full(y.shape, surface),
        _WPS_FULL_CHAIN, method=method, workers=4)
    assert answer.tobytes() == original.tobytes()
    assert not np.any(counts)


def test_measured_coastal_stencil_uses_the_single_land_corner():
    backend = _backend()
    source = np.array([
        [290.01953125, 290.20703125, 290.51953125, 296.51953125],
        [289.76953125, 290.01953125, 290.70703125, 298.20703125],
        [289.64453125, 289.89453125, 296.26953125, 298.95703125],
        [289.58203125, 290.33203125, 298.33203125, 298.70703125],
    ], dtype=np.float32)
    land = np.array([
        [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1],
    ], dtype=bool)
    # Retain the original full-grid FP32 coordinate before translating the
    # stencil origin. Re-rounding an FP64 local coordinate changes fractions.
    y = np.array([[np.float32(596.93147925019) - np.float32(595)]])
    x = np.array([[np.float32(182.485568546024) - np.float32(181)]])
    original = backend.regular_plan(
        np.arange(4), np.arange(4), y, x).apply(source, method="parabolic")
    # This regular plan rounds the full coordinates to FP32 before choosing
    # its crop. The exact projected-fraction replay uses a distinct split.
    assert original[0, 0] == np.float32(292.6592712402344)
    answer, counts = backend.surface_match(
        source, original, land, *_geometry(y, x, source.shape),
        np.ones(y.shape, dtype=bool), _WPS_FULL_CHAIN, workers=1)
    assert answer[0, 0] == np.float32(296.26953125)
    assert counts[0, :3].tolist() == [1, 1, 0]


def test_both_target_surfaces_exclude_the_opposite_source():
    backend = _backend()
    land = np.indices((6, 6))[1] >= 3
    source = np.where(land, 298.0, 287.0).astype(np.float32)
    y = np.full((1, 2), 2.5)
    x = np.full((1, 2), 2.5)
    original = np.full((1, 2), 292.5, dtype=np.float32)
    answer, counts = backend.surface_match(
        source, original, land, *_geometry(y, x, source.shape),
        np.array([[True, False]]), _WPS_FULL_CHAIN)
    np.testing.assert_array_equal(answer, [[298.0, 287.0]])
    assert counts[0, :3].tolist() == [2, 2, 0]


def test_missing_matching_surface_preserves_original_with_receipt():
    backend = _backend()
    source = np.full((4, 4), 287.0, dtype=np.float32)
    original = np.array([[291.12347]], dtype=np.float32)
    y, x = np.array([[1.25]]), np.array([[1.5]])
    answer, counts = backend.surface_match(
        source, original, np.zeros(source.shape, dtype=bool),
        *_geometry(y, x, source.shape), np.ones(y.shape, dtype=bool),
        _WPS_FULL_CHAIN)
    assert answer.tobytes() == original.tobytes()
    assert counts[0, SURFACE_MATCH_COUNT_KEYS.index("no_matching_surface")] == 1


def test_open_crop_cannot_establish_no_matching_source():
    backend = _backend()
    source = np.full((5, 5), 287.0, dtype=np.float32)
    y, x = np.array([[2.25]]), np.array([[2.5]])
    with pytest.raises(ValueError, match="expand the decoded source window"):
        backend.surface_match(
            source, np.array([[287.0]], dtype=np.float32),
            np.zeros(source.shape, dtype=bool),
            *_geometry(y, x, source.shape), np.ones(y.shape, dtype=bool),
            _WPS_FULL_CHAIN, closed_edges=0)


def test_open_crop_cannot_supply_a_clamped_coastal_halo():
    backend = _backend()
    land = np.indices((5, 5))[1] >= 2
    source = np.where(land, 298.0, 287.0).astype(np.float32)
    y, x = np.array([[0.25]]), np.array([[1.5]])
    with pytest.raises(ValueError, match="open source-window edge"):
        backend.surface_match(
            source, np.array([[292.0]], dtype=np.float32), land,
            *_geometry(y, x, source.shape), np.ones(y.shape, dtype=bool),
            _WPS_FULL_CHAIN, closed_edges=0)


def test_snow_preserves_uniform_water_bits_and_fills_mixed_water():
    backend = _backend()
    land = np.zeros((5, 5), dtype=bool)
    land[2, 2] = True
    source = np.full((5, 5), 1.0, dtype=np.float32)
    y, x = np.array([[1.5, 4.0]]), np.array([[1.5, 4.0]])
    original = np.array([[1.0, -0.0]], dtype=np.float32)
    answer, counts = backend.surface_match(
        source, original, land, *_geometry(y, x, source.shape),
        np.zeros(y.shape, dtype=bool), _WPS_SNOW_CHAIN, land_only=True)
    assert answer[0, 0].view(np.uint32) == np.float32(0).view(np.uint32)
    assert answer[0, 1].view(np.uint32) == original[0, 1].view(np.uint32)
    assert counts[0, SURFACE_MATCH_COUNT_KEYS.index("inactive_surface_fill")] == 1


def test_nearest_sea_ice_uses_fractional_water_corners_on_the_coast():
    backend = _backend()
    water = np.indices((5, 5))[1] >= 3
    source = np.where(water, 0.75, 0.0).astype(np.float32)
    source[4, 4] = -0.0
    y, x = np.array([[2.25, 4.0, 2.25]]), np.array([[2.25, 4.0, 2.75]])
    original = np.array([[0.0, -0.0, 0.75]], dtype=np.float32)
    answer, counts = backend.surface_match(
        source, original, water, *_geometry(y, x, source.shape),
        np.array([[True, True, False]]), _WPS_SNOW_CHAIN,
        method="nearest", land_only=True,
        nearest_y=np.array([[2, 4, 2]], dtype=np.int32),
        nearest_x=np.array([[2, 4, 3]], dtype=np.int32))
    assert answer[0, 0] == np.float32(0.75)
    assert answer[0, 1].view(np.uint32) == original[0, 1].view(np.uint32)
    assert answer[0, 2] == np.float32(0.0)
    assert counts[0, :3].tolist() == [2, 1, 0]
    assert counts[0, SURFACE_MATCH_COUNT_KEYS.index("inactive_surface_fill")] == 1


@pytest.mark.parametrize("target_is_land", [False, True])
def test_native_route_sea_ice_uses_the_shared_coastal_rule(target_is_land):
    backend = _backend()
    from dataclasses import replace
    from gpuwm.ingest.hrrr import (
        hrrr_source_grid, interpolate_hrrr_to_lambert,
    )
    from test_hrrr_island_donor import _window_snapshot

    base = _window_snapshot()
    land = np.indices((base.ny, base.nx))[1] >= 51
    fields = dict(base.fields)
    fields["LANDSEA"] = land.astype(np.float32)
    fields["XICE"] = np.where(land, 0.0, 0.75).astype(np.float32)
    snapshot = replace(base, fields=fields)
    source_grid = hrrr_source_grid()
    ref_lat, ref_lon = source_grid.ij_to_latlon(
        base.i_start + 50.75 + 1, base.j_start + 50.25 + 1)
    grid = LambertGrid(
        ref_lat=float(ref_lat), ref_lon=float(ref_lon),
        truelat1=source_grid.truelat1, truelat2=source_grid.truelat2,
        stand_lon=source_grid.stand_lon, dx=1000.0, dy=1000.0,
        e_we=2, e_sn=2, known_x=1.0, known_y=1.0)
    mapped = interpolate_hrrr_to_lambert(
        snapshot, grid, target_landmask=np.array([[float(target_is_land)]]),
        backend="cpu", cpu_bridge=backend.path, workers=1)
    assert mapped.fields["XICE"][0, 0] == np.float32(0.0 if target_is_land else 0.75)
    assert mapped.masked_field_repairs["XICE"]["mixed_surface"] == int(not target_is_land)


def test_worker_count_and_exact_projected_fraction_are_stable():
    backend = _backend()
    land = np.indices((7, 7))[1] >= 3
    source = np.where(land, 298.0, 287.0).astype(np.float32)
    # The fraction remains just below one even when integer + fraction would
    # round to the next integer in FP32. Exact donor ownership must survive.
    shape = (1, 9000)
    iy = np.full(shape, 2, dtype=np.int32)
    ix = np.full(shape, 2, dtype=np.int32)
    fy = np.full(shape, 0.5, dtype=np.float32)
    fx = np.full(shape, np.nextafter(np.float32(1), np.float32(0)))
    original = np.full(shape, 292.0, dtype=np.float32)
    targets = np.ones(shape, dtype=bool)
    serial = backend.surface_match(
        source, original, land, iy, ix, fy, fx, targets,
        _WPS_FULL_CHAIN, workers=1)
    parallel = backend.surface_match(
        source, original, land, iy, ix, fy, fx, targets,
        _WPS_FULL_CHAIN, workers=7)
    assert serial[0].tobytes() == parallel[0].tobytes()
    np.testing.assert_array_equal(serial[1], parallel[1])
    np.testing.assert_array_equal(serial[0], np.full(shape, 298.0))


def test_regular_geometry_uses_exact_crop_roundtrip_at_a_large_origin():
    backend = _backend()
    from gpuwm.ingest.interpolation_support import regular_source_support

    shape = (620, 300)
    y = np.array([[596.999999999, 596.93147925019]])
    x = np.array([[182.999999999, 182.485568546024]])
    support = regular_source_support(shape, y, x)
    assert support is not None and support.rows.start > 0
    geometry = _coastal_regular_geometry(shape, y, x, "parabolic")
    iy, ix, fy, fx = geometry
    np.testing.assert_array_equal(iy, np.floor(support.y).astype(np.int32) + support.rows.start)
    np.testing.assert_array_equal(ix, np.floor(support.x).astype(np.int32) + support.columns.start)
    np.testing.assert_array_equal(fy, support.y - np.floor(support.y))
    np.testing.assert_array_equal(fx, support.x - np.floor(support.x))
    assert fx[0, 0] == 0.0 and ix[0, 0] == 183
    land = np.indices(shape)[1] >= 183
    source = np.where(land, 298.0, 287.0).astype(np.float32)
    plan = backend.regular_plan(np.arange(shape[0]), np.arange(shape[1]), y, x)
    original = plan.apply(source, source_support=True)
    answer, counts = backend.surface_match(
        source, original, land, *geometry,
        np.ones(y.shape, dtype=bool), _WPS_FULL_CHAIN)
    assert answer[0, 0].view(np.uint32) == original[0, 0].view(np.uint32)
    np.testing.assert_array_equal(answer, np.full(y.shape, 298.0))
    assert counts[0, 0] == 1


@pytest.mark.parametrize("carry_land", [False, True])
def test_regular_default_preserves_uniform_fields_and_atmospheric_levels(carry_land):
    backend = _backend()
    grid = LambertGrid(e_we=5, e_sn=5, dx=2000.0, dy=2000.0,
                       ref_lat=37.8, ref_lon=-122.2,
                       truelat1=30.0, truelat2=60.0, stand_lon=-98.0)
    latitude = np.linspace(37.0, 38.5, 15)
    longitude = np.linspace(-123.0, -121.0, 18)
    rng = np.random.default_rng(54)
    fields = {
        "T": rng.uniform(280, 305, (2, 15, 18)),
        "T2": rng.uniform(280, 305, (15, 18)),
        "D2": rng.uniform(275, 280, (15, 18)),
        "RH2": rng.uniform(55, 85, (15, 18)),
        "Q2": rng.uniform(0.002, 0.012, (15, 18)),
    }
    if carry_land:
        fields["LANDSEA"] = np.ones((15, 18))
    snapshot = Era5Snapshot(valid_time=datetime(2026, 1, 1),
                            latitude=latitude, longitude=longitude,
                            levels_hpa=np.array([850.0, 1000.0]), fields=fields)
    mapped = interpolate_era5_to_lambert(
        snapshot, grid, backend="cpu", cpu_bridge=backend.path,
        relative_humidity_convention="water")
    y, x = grid.latlon_mass()
    plan = backend.regular_plan(latitude, longitude, y, x)
    for name, output, method in (("T", "TT", "parabolic"),
                                 ("T2", "T2", "parabolic"),
                                 ("D2", "D2", "parabolic"),
                                 ("RH2", "RH2", "parabolic"),
                                 ("Q2", "Q2", "bilinear")):
        baseline = plan.apply(fields[name], method=method, source_support=True)
        assert mapped.fields[output].tobytes() == baseline.tobytes()
