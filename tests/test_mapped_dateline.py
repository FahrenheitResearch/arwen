"""A domain on the antimeridian prepares from every mapped global source.

Named breakage, measured by the 2026-09-27 world initialization sweep on
the Metro-50 tile shape: every tile touching 180 degrees was refused by
every mapped global source (GEFS, GDAS, AI-GFS, AI-GEFS, IFS, AIFS, GEM,
ICON) while the GFS route built the same tiles.  Two breaks, both made by
the tree itself rather than by the data:

* a whole-globe source decodes to one ring stored -180..180, and the
  mapped route never re-cut that ring away from the target the way the
  GFS route does, so a target in the last cell was "outside the source
  grid";
* a regional crop across the seam (the GDT-101 remap writes one whenever
  the domain crosses it) was wrapped and sorted into an axis with a jump
  of nearly a full turn inside it, and refused as "not a regular axis".
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import test_mapped_frameset_streaming as fixture
from gpuwm.ingest.horiz import (
    _regular_coordinates, interpolate_era5_to_lambert,
    orient_global_source_longitudes,
)
from gpuwm.ingest.source_metadata import snapshot_metadata
from gpuwm.mapped_source import _regular_latlon_frame
from gpuwm.source_hierarchy import _spatial_coverage_receipt
from gpuwm.static.lambert import LambertGrid


# ----------------------------------------------------------------------
# The regional crop keeps its own column order.
# ----------------------------------------------------------------------

def _row(lon1: float, nx: int, dx: float = 0.5) -> dict[str, str]:
    return {"gdt": "0", "scan_mode": "0x40", "nx": str(nx), "ny": "3",
            "lat1": "10", "lon1": repr(lon1), "dx": repr(dx), "dy": "0.5"}


def test_a_regional_crop_across_the_antimeridian_keeps_its_column_order():
    values = np.arange(3 * 41, dtype=np.float64).reshape(3, 41)
    _, longitude, oriented = _regular_latlon_frame(
        _row(175.0, 41), values.ravel().copy())
    assert longitude[0] == 175.0 and longitude[-1] == 195.0
    np.testing.assert_array_equal(np.diff(longitude), np.full(40, 0.5))
    np.testing.assert_array_equal(oriented, values)


def test_a_crop_clear_of_the_seam_and_a_whole_ring_read_as_before():
    values = np.arange(3 * 11, dtype=np.float64).reshape(3, 11)
    _, longitude, oriented = _regular_latlon_frame(
        _row(336.5, 11), values.ravel().copy())
    assert longitude.tolist() == [-23.5 + 0.5 * k for k in range(11)]
    np.testing.assert_array_equal(oriented, values)
    ring = np.arange(3 * 360, dtype=np.float64).reshape(3, 360)
    _, longitude, oriented = _regular_latlon_frame(
        _row(0.0, 360, dx=1.0), ring.ravel().copy())
    assert longitude[0] == -180.0 and longitude[-1] == 179.0
    np.testing.assert_array_equal(oriented[:, 0], ring[:, 180])
    # A 1/12-degree ring whose octets round the spacing to micro-degrees
    # spans a hair under 360 degrees; it is still a ring, not a crop.
    ring = np.arange(3 * 4320, dtype=np.float64).reshape(3, 4320)
    _, longitude, oriented = _regular_latlon_frame(
        _row(0.0, 4320, dx=0.083333), ring.ravel().copy())
    assert -180.0 <= longitude[0] < -179.9 and 179.9 < longitude[-1] < 180.0
    np.testing.assert_array_equal(oriented[:, 0], ring[:, 2161])


# ----------------------------------------------------------------------
# The whole-globe ring is re-cut opposite the targets.
# ----------------------------------------------------------------------

@pytest.fixture
def global_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(fixture, "_NY", 91)
    monkeypatch.setattr(fixture, "_NX", 180)
    frame = replace(fixture._one_frame(),
                    latitude=np.linspace(-90.0, 90.0, 91),
                    longitude=-180.0 + 2.0 * np.arange(180, dtype=np.float64))
    authority = tmp_path / "authority"
    authority.write_text("dateline witness")
    return fixture._bundle_from_frames((frame,), authority)


def _grid(lon):
    return LambertGrid(52.0, lon, 45.0, 60.0, lon, 3000.0, 3000.0, 12, 14)


def _targets(grid):
    return tuple(pair[1] for pair in (grid.latlon_mass(), grid.latlon_u(),
                                      grid.latlon_v()))


def test_a_mapped_global_ring_is_cut_opposite_a_target_on_the_antimeridian(
        global_bundle):
    grid = _grid(179.95)
    unoriented = global_bundle.regular_snapshots()[0]
    with pytest.raises(ValueError, match="outside the source grid"):
        _regular_coordinates(unoriented.latitude, unoriented.longitude,
                             *grid.latlon_mass())
    sequence = global_bundle.regular_snapshots().for_grids((grid,))
    snapshot = sequence[0]
    # The GFS route's rule, applied to the same ring: same axis, same bytes.
    expected = orient_global_source_longitudes(unoriented, *_targets(grid))
    assert snapshot.longitude.tobytes() == expected.longitude.tobytes()
    assert snapshot.fields.keys() == expected.fields.keys()
    for name in snapshot.fields:
        assert snapshot.fields[name].tobytes() == expected.fields[name].tobytes(), name
    # The geometry stated before any field is read is the geometry packed.
    metadata = snapshot_metadata(sequence, 0)
    assert metadata.longitude.tobytes() == snapshot.longitude.tobytes()
    for lat, lon in (grid.latlon_mass(), grid.latlon_u(), grid.latlon_v()):
        _regular_coordinates(metadata.latitude, metadata.longitude, lat, lon)
    met = interpolate_era5_to_lambert(snapshot, grid, backend="cpu")
    assert np.isfinite(np.asarray(met.fields["TT"])).all()


def test_a_hierarchy_coverage_receipt_reads_the_re_cut_geometry(global_bundle):
    grid = _grid(-179.97)
    sequence = global_bundle.regular_snapshots().for_grids((grid,))
    receipt = _spatial_coverage_receipt(
        sequence, (grid,), SimpleNamespace(domains=(SimpleNamespace(grid_id=1),)),
        "mapped source")
    assert receipt["status"] == "PASS"
    x_low, x_high = receipt["domains"]["d01"]["masked_surface_donor_x"]
    assert 0 <= x_low and x_high < 180


def test_a_ring_whose_cut_is_clear_of_the_target_is_left_untouched(global_bundle):
    from gpuwm.ingest.horiz import global_ring_cut

    grid = _grid(10.0)
    assert global_ring_cut(
        global_bundle.regular_snapshots()[0].longitude, *_targets(grid)) is None
    from gpuwm.ingest.atmospheric_window import ATMOSPHERIC_FIELDS, WindowedAtmosphericSnapshot

    unoriented = global_bundle.regular_snapshots()[0]
    snapshot = global_bundle.regular_snapshots().for_grids((grid,))[0]
    assert snapshot.longitude.tobytes() == unoriented.longitude.tobytes()
    # Since A135 a ring whose cut is clear of every stencil takes the
    # atmospheric window (this pin compared whole fields while a ring was
    # never windowed): its atmospheric fields are the STORED ring's crop,
    # and every other field is the stored field itself.  Nothing is re-cut.
    assert isinstance(snapshot, WindowedAtmosphericSnapshot)
    for name in snapshot.fields:
        expected = unoriented.fields[name]
        if name in ATMOSPHERIC_FIELDS:
            expected = snapshot.window.crop(expected)
        assert snapshot.fields[name].tobytes() == expected.tobytes(), name


def test_a_re_cut_taken_on_another_axis_is_refused(global_bundle):
    from gpuwm.ingest.horiz import global_ring_cut, recut_global_ring

    snapshot = global_bundle.regular_snapshots()[0]
    other = snapshot.longitude + 1.0
    cut = global_ring_cut(other, *_targets(_grid(179.95)))
    assert cut is not None
    with pytest.raises(ValueError, match="another source axis"):
        recut_global_ring(snapshot, cut)


@pytest.mark.parametrize("pole", [90.0, -90.0])
def test_polar_cap_keeps_periodic_donors_on_every_stagger(global_bundle, pole):
    from gpuwm.static.projection import PolarStereoGrid
    from gpuwm.ingest.horiz import global_ring_cut

    grid = PolarStereoGrid(pole, 0.0, pole, pole, 0.0,
                           12000.0, 12000.0, 250, 200)
    original = global_bundle.regular_snapshots()[0]
    cut = global_ring_cut(original.longitude, *_targets(grid))
    assert cut is not None and cut.halo == 3
    oriented = orient_global_source_longitudes(original, *_targets(grid))
    assert oriented.longitude.size == original.longitude.size + 6
    # The source bytes, including opposite-side seam donors, are retained.
    for name, value in original.fields.items():
        np.testing.assert_array_equal(oriented.fields[name][..., 3:-3, :],
                                      np.asarray(value)[..., cut.columns()])
    for lat, lon in (grid.latlon_mass(), grid.latlon_u(), grid.latlon_v()):
        y, x = _regular_coordinates(oriented.latitude, oriented.longitude, lat, lon)
        assert x.min() >= 1.0
        assert x.max() <= oriented.longitude.size - 3.0
        assert np.isfinite(y).all()
    # The lazy forcing geometry and loaded field agree on the expanded ring.
    sequence = global_bundle.regular_snapshots().for_grids((grid,))
    assert snapshot_metadata(sequence, 0).longitude.tobytes() == oriented.longitude.tobytes()
    assert snapshot_metadata(sequence, 0).latitude.tobytes() == oriented.latitude.tobytes()
    assert sequence[0].longitude.tobytes() == oriented.longitude.tobytes()

    from gpuwm.gfs_direct import _source_coverage_receipt
    receipt = _source_coverage_receipt(
        oriented, grid, np.zeros(grid.latlon_mass()[0].shape, dtype=bool))
    for stagger in receipt["staggerings"].values():
        assert 0 <= stagger["parabolic_donor_y"][0]
        assert stagger["parabolic_donor_y"][1] < oriented.latitude.size


@pytest.mark.parametrize("pole", [90.0, -90.0])
def test_polar_donors_preserve_a_smooth_cartesian_tangent_wind(global_bundle, pole):
    from gpuwm.static.projection import PolarStereoGrid
    from gpuwm.ingest.cpu_backend import shared_cpu_backend

    grid = PolarStereoGrid(pole, 0.0, pole, pole, 0.0,
                           12000.0, 12000.0, 250, 200)
    original = global_bundle.regular_snapshots()[0]
    lon, lat = np.meshgrid(np.deg2rad(original.longitude),
                          np.deg2rad(original.latitude))
    # Tangential projection of the fixed Cartesian vector (1,0,0).
    # Its earth components turn with longitude but the vector is smooth
    # through the pole. Scalars also reverse their longitude there.
    fields = {"U10": -np.sin(lon), "V10": -np.sin(lat) * np.cos(lon),
              "SKINTEMP": np.cos(lat) * np.cos(lon)}
    snapshot = replace(original, fields=fields)
    oriented = orient_global_source_longitudes(snapshot, *_targets(grid))
    lon, lat = np.meshgrid(np.deg2rad(oriented.longitude),
                          np.deg2rad(oriented.latitude))
    expected = {"U10": -np.sin(lon), "V10": -np.sin(lat) * np.cos(lon),
                "SKINTEMP": np.cos(lat) * np.cos(lon)}
    for name in fields:
        np.testing.assert_allclose(oriented.fields[name], expected[name],
                                   rtol=0, atol=3e-15)
    target_lat, target_lon = grid.latlon_mass()
    plan = shared_cpu_backend().regular_plan(
        oriented.latitude, oriented.longitude, target_lat, target_lon)
    actual_u = plan.apply(oriented.fields["U10"], "parabolic", workers=8)
    actual_v = plan.apply(oriented.fields["V10"], "parabolic", workers=8)
    np.testing.assert_allclose(actual_u, -np.sin(np.deg2rad(target_lon)),
                               rtol=0, atol=2e-6)
    np.testing.assert_allclose(
        actual_v, -np.sin(np.deg2rad(target_lat)) * np.cos(np.deg2rad(target_lon)),
        rtol=0, atol=3e-6)


@pytest.mark.parametrize("pole", [90.0, -90.0])
def test_pole_latitude_donors_do_not_require_a_longitude_recut(global_bundle, pole):
    from gpuwm.ingest.horiz import global_ring_cut
    original = global_bundle.regular_snapshots()[0]
    target_lat = np.array([[pole * (89.99 / 90.0)]])
    target_lon = np.array([[0.1]])
    assert global_ring_cut(original.longitude, target_lon) is None
    oriented = orient_global_source_longitudes(
        original, target_lon, target_latitudes=(target_lat,))
    assert oriented.longitude.tobytes() == original.longitude.tobytes()
    assert oriented.latitude.size == original.latitude.size + 3
    y, x = _regular_coordinates(oriented.latitude, oriented.longitude,
                                target_lat, target_lon)
    assert np.floor(y).min() - 1 >= 0
    assert np.floor(y).max() + 2 < oriented.latitude.size
    retained = slice(0, -3) if pole > 0 else slice(3, None)
    for name in original.fields:
        np.testing.assert_array_equal(oriented.fields[name][..., retained, :],
                                      original.fields[name])
