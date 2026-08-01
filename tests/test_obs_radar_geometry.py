"""Beam geometry, analytically.

Every assertion here is against a closed-form answer or a published table
value, never against another run of the same code.  A synthetic radial at a
known azimuth, elevation and range must land in the grid cell and at the
height that trigonometry says it does, or the superob is decorating noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.obs.geometry import (REFRACTION_FACTOR, beam_geometry,
                                beam_unit_vector, effective_earth_radius_m,
                                forward_azimuth, gate_locations,
                                great_circle_point)
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid
from gpuwm.static.projection import EARTH_RADIUS_M

_DEG = np.pi / 180.0


def _grid(nx: int = 61, ny: int = 61, dx: float = 2000.0,
          nz: int = 20, top_m: float = 20000.0) -> TargetGrid:
    """A flat-terrain Lambert domain centred on the KTLX site coordinates.

    Terrain is zero and the levels are uniform in height so a level index
    is a hand-checkable division; the projection is a real WRF Lambert, so
    the horizontal placement is not simplified at all.
    """

    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    z_w = np.linspace(0.0, top_m, nz + 1)
    return TargetGrid.from_projection(projection, z_w=z_w, name="analytic")


def test_beam_height_matches_the_four_thirds_earth_closed_form():
    """The height formula, evaluated independently, to the metre."""

    ranges = np.array([1.0e3, 25.0e3, 50.0e3, 100.0e3, 200.0e3, 250.0e3])
    for elevation in (0.5, 1.5, 4.0, 19.5):
        for alt in (0.0, 370.0):
            height, _, _ = beam_geometry(ranges, elevation, alt)
            radius = effective_earth_radius_m() + alt
            expected = (np.sqrt(ranges ** 2 + radius ** 2
                                + 2.0 * ranges * radius
                                * np.sin(elevation * _DEG))
                        - radius + alt)
            assert np.allclose(height, expected, atol=1e-6)


def test_beam_height_reproduces_the_textbook_table():
    """0.5 deg at 100 km is ~1.5 km ARL; a straight ray would be 0.87 km.

    The point of the check is that the curvature term is present and has
    the right magnitude, not merely that the arithmetic is self-consistent.
    """

    height, _, _ = beam_geometry(100.0e3, 0.5, 0.0)
    assert 1400.0 < float(height) < 1550.0
    straight_ray = 100.0e3 * np.sin(0.5 * _DEG)
    assert float(height) > 1.5 * straight_ray

    # At 250 km the 0.5 deg beam is well into the mid troposphere.
    far, _, _ = beam_geometry(250.0e3, 0.5, 0.0)
    assert 5500.0 < float(far) < 6300.0


def test_local_elevation_exceeds_the_antenna_elevation_and_grows_with_range():
    """Over the curve the beam tilts up relative to the local horizontal."""

    ranges = np.array([1.0e3, 50.0e3, 150.0e3, 250.0e3])
    _, _, local = beam_geometry(ranges, 0.5, 0.0)
    assert np.all(local >= 0.5 - 1e-9)
    assert np.all(np.diff(local) > 0.0)
    # The excess is the effective-earth arc angle: 250 km of ground
    # distance subtends 250/8493 rad = 1.69 deg on the k*a sphere, so the
    # local elevation is 0.5 + 1.69 ~ 2.19 deg, not 0.5.
    assert float(local[-1]) == pytest.approx(
        0.5 + 250.0e3 / effective_earth_radius_m() / _DEG, abs=0.01)
    # At zero range the local and antenna elevations coincide.
    _, _, at_radar = beam_geometry(0.0, 0.5, 0.0)
    assert float(at_radar) == pytest.approx(0.5, abs=1e-9)


def test_surface_arc_is_shorter_than_slant_range_and_matches_flat_earth_near_in():
    ranges = np.array([500.0, 5.0e3, 100.0e3])
    _, arc, _ = beam_geometry(ranges, 0.0, 0.0)
    assert np.all(arc < ranges)
    # At short range with a level beam the arc is the range to a part in 1e6.
    assert float(arc[0]) == pytest.approx(500.0, rel=1e-6)
    assert float(arc[1]) == pytest.approx(5.0e3, rel=1e-5)


def test_synthetic_radial_lands_in_the_expected_grid_cell():
    """Due east, due north, and the diagonal, at a range that is an exact
    number of grid cells."""

    grid = _grid()
    centre_j, centre_i = grid.ny // 2, grid.nx // 2
    radar_lat = float(grid.lat[centre_j, centre_i])
    radar_lon = float(grid.lon[centre_j, centre_i])
    cells = 10
    # A near-level beam so the surface arc is the slant range to <1 m.
    slant = cells * grid.dx_m
    for azimuth, di, dj in ((90.0, cells, 0), (270.0, -cells, 0),
                            (0.0, 0, cells), (180.0, 0, -cells)):
        lat, lon, _, _, _, _ = gate_locations(
            radar_lat, radar_lon, 0.0, azimuth, slant, 0.0)
        i_frac, j_frac = grid.mass_index(lat, lon)
        assert int(round(float(i_frac))) == centre_i + di, azimuth
        assert int(round(float(j_frac))) == centre_j + dj, azimuth
        # And within a tenth of a cell of the exact index: the Lambert
        # projection is conformal, so 20 km from the reference longitude
        # the grid is very nearly north-up.
        assert abs(float(i_frac) - (centre_i + di)) < 0.1, azimuth
        assert abs(float(j_frac) - (centre_j + dj)) < 0.1, azimuth


def test_mass_index_inverts_the_grids_own_latlon_arrays():
    """The index convention is verified against the artifact, not asserted.

    ``latlon_mass`` is what wrfout's XLAT/XLONG carry; if ``mass_index``
    used a different offset every observation would be placed a cell off
    and nothing downstream would notice.
    """

    grid = _grid(nx=25, ny=19, dx=4000.0)
    j_expected, i_expected = np.meshgrid(np.arange(grid.ny),
                                         np.arange(grid.nx), indexing="ij")
    i_frac, j_frac = grid.mass_index(grid.lat, grid.lon)
    assert np.allclose(i_frac, i_expected, atol=1e-9)
    assert np.allclose(j_frac, j_expected, atol=1e-9)


def test_synthetic_radial_lands_at_the_expected_model_level():
    """Uniform 1 km layers make the level index a division by hand."""

    grid = _grid(nz=20, top_m=20000.0)          # 1 km layers, 0..20 km
    centre_j, centre_i = grid.ny // 2, grid.nx // 2
    radar_lat = float(grid.lat[centre_j, centre_i])
    radar_lon = float(grid.lon[centre_j, centre_i])

    for elevation, slant in ((0.5, 100.0e3), (4.0, 50.0e3), (10.0, 30.0e3)):
        lat, lon, height, _, _, _ = gate_locations(
            radar_lat, radar_lon, 0.0, 45.0, slant, elevation)
        i_index = np.rint(grid.mass_index(lat, lon)[0]).astype(np.intp)
        j_index = np.rint(grid.mass_index(lat, lon)[1]).astype(np.intp)
        if not bool(grid.inside(i_index, j_index)):
            continue
        level = grid.level_index(np.atleast_1d(i_index),
                                 np.atleast_1d(j_index),
                                 np.atleast_1d(height))
        assert int(level[0]) == int(float(height) // 1000.0), (
            elevation, slant, float(height))


def test_level_index_fails_closed_below_terrain_and_above_the_model_top():
    grid = _grid(nz=10, top_m=10000.0)
    i_index = np.array([5, 5, 5], dtype=np.intp)
    j_index = np.array([5, 5, 5], dtype=np.intp)
    heights = np.array([-10.0, 5500.0, 10000.1])
    level = grid.level_index(i_index, j_index, heights)
    assert level[0] == -1
    assert level[1] == 5
    assert level[2] == -1


def test_beam_unit_vector_is_the_radial_velocity_operator():
    """Due east at zero elevation projects onto u alone, and so on round."""

    east, north, up = beam_unit_vector(90.0, 0.0)
    assert (float(east), float(north)) == pytest.approx((1.0, 0.0), abs=1e-12)
    assert float(up) == pytest.approx(0.0, abs=1e-12)

    east, north, up = beam_unit_vector(0.0, 0.0)
    assert (float(east), float(north)) == pytest.approx((0.0, 1.0), abs=1e-12)

    east, north, up = beam_unit_vector(180.0, 30.0)
    assert float(north) == pytest.approx(-np.cos(30.0 * _DEG), abs=1e-12)
    assert float(up) == pytest.approx(0.5, abs=1e-12)

    # Always a unit vector, for any look direction.
    azimuth = np.linspace(0.0, 359.0, 37)
    elevation = np.linspace(-2.0, 25.0, 37)
    east, north, up = beam_unit_vector(azimuth, elevation)
    assert np.allclose(east ** 2 + north ** 2 + up ** 2, 1.0, atol=1e-12)


def test_great_circle_point_and_back_azimuth_are_consistent():
    lat, lon = great_circle_point(35.0, -97.0, 90.0, 200.0e3)
    # Due east from 35N: the parallel curves, so the endpoint is south of
    # the origin latitude on a great circle.
    assert lat < 35.0
    assert lon > -97.0
    # The forward bearing at the far point has turned clockwise.
    bearing = float(forward_azimuth(35.0, -97.0, lat, lon))
    assert 90.0 < bearing < 92.0

    # A northward shot keeps the longitude and advances the latitude by
    # exactly the arc angle.
    lat, lon = great_circle_point(35.0, -97.0, 0.0, 111.0e3)
    assert float(lon) == pytest.approx(-97.0, abs=1e-9)
    assert float(lat) == pytest.approx(
        35.0 + 111.0e3 / EARTH_RADIUS_M / _DEG, abs=1e-9)


def test_geometry_refuses_impossible_inputs():
    with pytest.raises(ValueError):
        beam_geometry(-1.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        effective_earth_radius_m(refraction_factor=0.0)
    with pytest.raises(ValueError):
        effective_earth_radius_m(earth_radius_m=-1.0)


def test_refraction_factor_is_the_standard_four_thirds():
    assert REFRACTION_FACTOR == pytest.approx(4.0 / 3.0)
    assert effective_earth_radius_m() == pytest.approx(
        EARTH_RADIUS_M * 4.0 / 3.0)
    # A straight-ray (k=1) beam is lower than the refracted one everywhere
    # beyond the antenna.
    refracted, _, _ = beam_geometry(150.0e3, 0.5, 0.0)
    straight, _, _ = beam_geometry(150.0e3, 0.5, 0.0, refraction_factor=1.0)
    assert float(refracted) < float(straight)
