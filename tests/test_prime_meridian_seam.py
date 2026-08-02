"""A global source crop is a ring, and its cut must not land on the target.

`Area.as_nomads` (gpuwm/fetch.py:348-362) cannot express a box that crosses
the source's own 0/360 origin as one ``0 <= left < right <= 360`` subregion,
so it widens such a box to the whole longitude band.  The decode then hands
back 1440 columns running 0.00 .. 359.75 -- a ring stored as a finite array
with an artificial cut at longitude 0.  Every consumer treats that cut as
the edge of the world, so a domain straddling longitude 0 (the UK, Iberia,
West Africa, Svalbard) had target points in the one cell that the array does
not contain.  The antimeridian never had the problem: that crop is a genuine
subregion which simply continues past 180.
"""
import numpy as np
import pytest

from gpuwm.fetch import Area
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.horiz import (
    _regular_coordinates,
    global_longitude_period_columns,
    orient_global_source_longitudes,
)
from gpuwm.static.lambert import LambertGrid
from gpuwm.static.projection import MercatorGrid, PolarStereoGrid

from datetime import datetime

DX_DEG = 0.25
CYCLE = datetime(2026, 7, 28, 0, 0, 0)


def _nomads_axis(area: Area, dx: float = DX_DEG):
    """The longitude axis gpuwm/gfs_direct.py:697-713 builds for a box."""
    box = area.as_nomads()
    left = box["left_lon"]
    right = box["right_lon"]
    columns = int(round((right - left) / dx))
    # A whole-band request selects the 1440 distinct GFS columns; a
    # subregion request is inclusive of both its edges.
    nx = columns if (left, right) == (0.0, 360.0) else columns + 1
    start = (left + 180.0) % 360.0 - 180.0
    return start + np.arange(nx, dtype=np.float64) * dx


def _domain(centre_lat: float, centre_lon: float, nx: int = 86, ny: int = 68,
            dx_m: float = 12000.0):
    """The wizard's own projection choice for this centre latitude."""
    if abs(centre_lat) < 25.0:
        return MercatorGrid(
            ref_lat=centre_lat, ref_lon=centre_lon, truelat1=centre_lat,
            truelat2=0.0, stand_lon=centre_lon, dx=dx_m, dy=dx_m,
            e_we=nx + 1, e_sn=ny + 1)
    if abs(centre_lat) > 60.0:
        return PolarStereoGrid(
            ref_lat=centre_lat, ref_lon=centre_lon, truelat1=centre_lat,
            truelat2=0.0, stand_lon=centre_lon, dx=dx_m, dy=dx_m,
            e_we=nx + 1, e_sn=ny + 1)
    return LambertGrid(
        ref_lat=centre_lat, ref_lon=centre_lon,
        truelat1=centre_lat - 10.0, truelat2=centre_lat + 10.0,
        stand_lon=centre_lon, dx=dx_m, dy=dx_m, e_we=nx + 1, e_sn=ny + 1)


def _fetch_area(grid: LambertGrid, margin_deg: float = 15.45) -> Area:
    lat, lon = grid.latlon_mass()
    centre = grid.ref_lon
    unwrapped = centre + ((np.asarray(lon) - centre + 180.0) % 360.0 - 180.0)
    west = (float(unwrapped.min()) - margin_deg + 180.0) % 360.0 - 180.0
    east = (float(unwrapped.max()) + margin_deg + 180.0) % 360.0 - 180.0
    return Area(float(lat.min()) - margin_deg, west,
                float(lat.max()) + margin_deg, east)


def _snapshot(longitude: np.ndarray, latitude: np.ndarray) -> Era5Snapshot:
    """A snapshot whose one field records its own source longitude."""
    field = np.broadcast_to(longitude[None, :],
                            (latitude.size, longitude.size)).copy()
    return Era5Snapshot(
        valid_time=CYCLE, levels_hpa=np.asarray([500.0]),
        latitude=latitude, longitude=longitude, fields={"LONMARK": field})


def _map(grid: LambertGrid, snapshot: Era5Snapshot):
    lat, lon = grid.latlon_mass()
    return _regular_coordinates(snapshot.latitude, snapshot.longitude,
                                lat, lon)


# The sweep the independent tester ran: 13 identical domains at latitude
# 40, walking the centre longitude across 0.  The 86x68 12 km domain is
# +/-6.05 degrees wide, so its EDGE reaches longitude 0 for |centre| < 6.05.
_SWEEP_CROSSING = (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
_SWEEP_CLEAR = (-20.0, -12.0, -8.0, 8.0, 12.0, 20.0)


@pytest.mark.parametrize("centre_lon", _SWEEP_CROSSING + _SWEEP_CLEAR)
def test_every_sweep_domain_maps_onto_its_source(centre_lon):
    grid = _domain(40.0, centre_lon)
    area = _fetch_area(grid)
    longitude = _nomads_axis(area)
    latitude = np.arange(21.0, 59.0 + 0.5 * DX_DEG, DX_DEG)
    snapshot = orient_global_source_longitudes(
        _snapshot(longitude, latitude),
        grid.latlon_mass()[1], grid.latlon_u()[1], grid.latlon_v()[1])
    y, x = _map(grid, snapshot)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    # The parabolic and masked-surface stencils both reach floor-1..floor+2,
    # which gfs_direct.py:352-359 refuses a crop for lacking.
    assert int(np.floor(x).min()) - 1 >= 0
    assert int(np.floor(x).max()) + 2 < snapshot.longitude.size


@pytest.mark.parametrize("centre_lon", _SWEEP_CROSSING)
def test_negative_control_unoriented_source_still_refuses(centre_lon):
    """Without the re-cut the same domains fail, exactly as reported."""
    grid = _domain(40.0, centre_lon)
    longitude = _nomads_axis(_fetch_area(grid))
    latitude = np.arange(21.0, 59.0 + 0.5 * DX_DEG, DX_DEG)
    with pytest.raises(ValueError, match="fall outside the source grid"):
        _map(grid, _snapshot(longitude, latitude))


def test_refusal_names_the_point_and_the_source_span():
    grid = _domain(40.0, 0.0)
    longitude = _nomads_axis(_fetch_area(grid))
    latitude = np.arange(21.0, 59.0 + 0.5 * DX_DEG, DX_DEG)
    with pytest.raises(ValueError) as caught:
        _map(grid, _snapshot(longitude, latitude))
    message = str(caught.value)
    assert "lat/lon" in message and "source index" in message
    assert "the source covers" in message


@pytest.mark.parametrize("centre", [(51.5, -0.12), (0.0, 0.0), (78.2, 15.6),
                                    (35.0, 3.0), (6.5, 3.4)])
def test_reoriented_source_preserves_the_value_at_every_target(centre):
    """The re-cut moves indices, never values: the marker field agrees."""
    grid = _domain(centre[0], centre[1])
    longitude = _nomads_axis(_fetch_area(grid))
    latitude = np.arange(max(-89.0, centre[0] - 25.0),
                         min(89.0, centre[0] + 25.0), DX_DEG)
    snapshot = orient_global_source_longitudes(
        _snapshot(longitude, latitude),
        grid.latlon_mass()[1], grid.latlon_u()[1], grid.latlon_v()[1])
    y, x = _map(grid, snapshot)
    marker = np.asarray(snapshot.fields["LONMARK"])
    nearest = marker[np.rint(y).astype(int), np.rint(x).astype(int)]
    target_lon = grid.latlon_mass()[1]
    separation = (nearest - target_lon + 180.0) % 360.0 - 180.0
    assert np.abs(separation).max() < DX_DEG


@pytest.mark.parametrize("centre_lon", _SWEEP_CLEAR)
def test_a_ring_whose_cut_is_clear_of_the_target_is_left_alone(centre_lon):
    """No-drift: re-cutting moves the FP32 fractional index, so a ring that
    did not need rotating must not be rotated.  These six sweep domains have
    whole-band crops but do not straddle the cut."""
    grid = _domain(40.0, centre_lon)
    longitude = _nomads_axis(_fetch_area(grid))
    latitude = np.arange(21.0, 59.0 + 0.5 * DX_DEG, DX_DEG)
    snapshot = _snapshot(longitude, latitude)
    assert global_longitude_period_columns(longitude) == 1440
    oriented = orient_global_source_longitudes(
        snapshot, grid.latlon_mass()[1], grid.latlon_u()[1],
        grid.latlon_v()[1])
    assert oriented is snapshot


def test_regional_crop_is_returned_unchanged():
    """A CONUS-shaped subregion is not a ring and must not be touched."""
    longitude = np.arange(230.0, 300.0 + 0.5 * DX_DEG, DX_DEG)
    latitude = np.arange(20.0, 55.0 + 0.5 * DX_DEG, DX_DEG)
    snapshot = _snapshot(longitude, latitude)
    grid = _domain(35.5, -97.5)
    oriented = orient_global_source_longitudes(
        snapshot, grid.latlon_mass()[1])
    assert oriented is snapshot
    assert global_longitude_period_columns(longitude) is None


def test_antimeridian_crop_is_returned_unchanged():
    """The dateline route already worked and must stay byte-identical."""
    grid = _domain(52.9, 172.9)
    longitude = _nomads_axis(_fetch_area(grid))
    latitude = np.arange(33.0, 72.0 + 0.5 * DX_DEG, DX_DEG)
    snapshot = _snapshot(longitude, latitude)
    assert global_longitude_period_columns(longitude) is None
    oriented = orient_global_source_longitudes(
        snapshot, grid.latlon_mass()[1])
    assert oriented is snapshot
    y, x = _map(grid, snapshot)
    assert int(np.floor(x).min()) - 1 >= 0
    assert int(np.floor(x).max()) + 2 < longitude.size


def test_global_axis_is_recognised_and_regional_axis_is_not():
    assert global_longitude_period_columns(
        np.arange(1440, dtype=np.float64) * DX_DEG) == 1440
    assert global_longitude_period_columns(
        np.arange(0.0, 360.0 + 0.5 * DX_DEG, DX_DEG)) == 1440
    assert global_longitude_period_columns(
        np.arange(-27.5, 15.5 + 0.5 * DX_DEG, DX_DEG)) is None


def test_reorientation_keeps_the_axis_uniform_and_increasing():
    grid = _domain(51.5, -0.12)
    longitude = _nomads_axis(_fetch_area(grid))
    latitude = np.arange(30.0, 70.0 + 0.5 * DX_DEG, DX_DEG)
    oriented = orient_global_source_longitudes(
        _snapshot(longitude, latitude), grid.latlon_mass()[1])
    axis = np.asarray(oriented.longitude)
    assert axis.size == longitude.size
    assert np.all(np.diff(axis) > 0.0)
    np.testing.assert_allclose(np.diff(axis), DX_DEG, rtol=1e-11, atol=1e-12)
    # The same set of source meridians, re-cut, not resampled.
    np.testing.assert_allclose(np.sort(axis % 360.0),
                               np.sort(longitude % 360.0), atol=1e-9)
