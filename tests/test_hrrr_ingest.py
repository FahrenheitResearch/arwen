"""CPU contracts for the native HRRR bridge loader and geometry."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest.hrrr import (
    HRRR_WPS_EQUIVALENT_DX_M,
    _build_masked_bilinear_stencil,
    _gate_forecast_hours,
    hrrr_source_grid,
    load_hrrr_native_series,
    load_hrrr_native_window,
)
from gpuwm.static.lambert import LambertGrid


_ATMOSPHERE_3D = (
    "PRES", "QC", "QI", "QR", "QS", "QG", "HGT", "TT", "SPFH",
    "U_MASS", "V_MASS",
)
_ATMOSPHERE_2D = (
    "PSFC", "SOILHGT", "SKINTEMP", "SNOW", "SNOWH", "T2", "Q2",
    "U10_MASS", "V10_MASS", "LANDSEA", "XICE",
)


def test_gate_accepts_absolute_contiguous_windows_through_cycle_horizon():
    for hours, cycle in (
            (tuple(range(49)), "2026-07-18 00:00:00"),
            (tuple(range(12, 19)), "2026-07-18 05:00:00"),
            (tuple(range(40, 47)), "2026-07-18 18:00:00")):
        gate = {
            "forecast_hours": ",".join(map(str, hours)),
            "series_count": str(len(hours)),
            "cycle": cycle,
        }
        assert _gate_forecast_hours(gate) == hours

    with pytest.raises(ValueError, match="contiguous, ordered"):
        _gate_forecast_hours({
            "forecast_hours": "12,13,15", "series_count": "3",
            "cycle": "2026-07-18 18:00:00"})
    with pytest.raises(ValueError, match="at least two"):
        _gate_forecast_hours({
            "forecast_hours": "12", "series_count": "1",
            "cycle": "2026-07-18 18:00:00"})
    with pytest.raises(ValueError, match="horizon f18"):
        _gate_forecast_hours({
            "forecast_hours": "18,19", "series_count": "2",
            "cycle": "2026-07-18 05:00:00",
        })


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_bridge(root: Path) -> str:
    root.mkdir()
    (root / "gate.txt").write_text(
        "status\tPASS\n"
        "cycle\t2026-07-18 00:00:00\n"
        "atmosphere_selected_per_time\t561\n"
        "hybrid_levels\t50\n"
        "soil_selected_per_time\t18\n"
        "window_zero_based_inclusive\ti=10..10 j=20..20\n"
        "window_shape\t1x1\n"
        "qice_mapping\tPASS discipline=0 category=1 parameter=82 "
        "level_type=105; finite/nonnegative/nonzero\n"
        "cross_time_inventory\tPASS exact selected keys/levels/grid\n")
    for hour in (0, 1):
        atmosphere = root / f"atmosphere-f{hour:02d}"
        soil = root / f"soil-f{hour:02d}"
        atmosphere.mkdir()
        soil.mkdir()
        for index, name in enumerate(_ATMOSPHERE_3D):
            np.full((50, 1, 1), index + hour, dtype="<f4").tofile(
                atmosphere / f"{name}.f32le")
        for index, name in enumerate(_ATMOSPHERE_2D):
            np.full((1, 1), index + hour, dtype="<f4").tofile(
                atmosphere / f"{name}.f32le")
        for index, name in enumerate(("SOILT", "SOILW")):
            np.full((9, 1, 1), index + hour, dtype="<f4").tofile(
                soil / f"{name}.f32le")
    payloads = sorted(path for path in root.rglob("*") if path.is_file())
    lines = [f"{_hash(path)}  ./{path.relative_to(root).as_posix()}"
             for path in payloads]
    manifest = root / "SHA256SUMS"
    manifest.write_text("\n".join(lines) + "\n")
    return _hash(manifest)


def test_loader_requires_external_manifest_binding_and_exact_shapes(tmp_path):
    root = tmp_path / "bridge"
    manifest_hash = _fake_bridge(root)
    snapshot = load_hrrr_native_window(
        root, 1, expected_manifest_sha256=manifest_hash)
    assert snapshot.valid_time.isoformat() == "2026-07-18T01:00:00"
    assert (snapshot.i_start, snapshot.j_start, snapshot.ny, snapshot.nx) == (
        10, 20, 1, 1)
    assert snapshot.fields["PRES"].shape == (50, 1, 1)
    assert snapshot.fields["SOILT"].shape == (9, 1, 1)
    series = load_hrrr_native_series(
        root, (0, 1), expected_manifest_sha256=manifest_hash)
    assert [item.forecast_hour for item in series] == [0, 1]

    with pytest.raises(ValueError, match="SHA256SUMS hash mismatch"):
        load_hrrr_native_window(
            root, 0, expected_manifest_sha256="0" * 64)
    with (root / "gate.txt").open("a") as stream:
        stream.write("edited\tverdict\n")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_hrrr_native_window(
            root, 0, expected_manifest_sha256=manifest_hash)


def test_radius_corrected_aligned_d01_maps_to_integer_hrrr_indices():
    source = hrrr_source_grid()
    target = LambertGrid(
        ref_lat=35.5028506728143,
        ref_lon=-98.002166928566,
        truelat1=38.5,
        truelat2=38.5,
        stand_lon=-97.5,
        dx=HRRR_WPS_EQUIVALENT_DX_M,
        dy=HRRR_WPS_EQUIVALENT_DX_M,
        e_we=200,
        e_sn=200,
    )
    lat, lon = target.latlon_mass()
    x, y = source.latlon_to_ij(lat, lon)
    expected_x, expected_y = np.meshgrid(
        np.arange(786.0, 985.0), np.arange(320.0, 519.0))
    np.testing.assert_allclose(x, expected_x, rtol=0.0, atol=3.0e-10)
    np.testing.assert_allclose(y, expected_y, rtol=0.0, atol=3.0e-10)


def test_masked_bilinear_is_convex_and_renormalizes_valid_land_only():
    source_land = np.array([[1, 0], [1, 0]], dtype=bool)
    target_land = np.ones((1, 1), dtype=bool)
    iy, ix, weights, report = _build_masked_bilinear_stencil(
        np.array([[0.25]]), np.array([[0.75]]), source_land, target_land)
    source = np.array([[0.2, -10.0], [0.6, -20.0]])
    mapped = sum(
        source[iy[corner], ix[corner]] * weights[corner]
        for corner in range(4))
    np.testing.assert_allclose(mapped, [[0.5]], rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.sum(weights, axis=0), 1.0)
    assert np.all(weights >= 0.0)
    assert report["renormalized_target_count"] == 1
    assert report["fallback_target_count"] == 0


def test_masked_bilinear_uses_bounded_nearest_valid_fallback():
    source_land = np.zeros((5, 5), dtype=bool)
    source_land[0, 0] = True
    target_land = np.ones((1, 1), dtype=bool)
    iy, ix, weights, report = _build_masked_bilinear_stencil(
        np.array([[2.25]]), np.array([[2.25]]), source_land, target_land,
        fallback_radius=4)
    assert (iy[0, 0, 0], ix[0, 0, 0]) == (0, 0)
    np.testing.assert_array_equal(weights[:, 0, 0], [1.0, 0.0, 0.0, 0.0])
    assert report["fallback_target_count"] == 1
    assert report["fallback_max_distance_cells"] == pytest.approx(
        np.sqrt(2.0 * 2.25 ** 2))
    assert report["fallback_distance_ceiling_histogram_cells"] == {"4": 1}
    assert report["unresolved_target_count"] == 0
    assert report["cross_surface_donor_count"] == 0

    with pytest.raises(ValueError, match="no valid surface-matched"):
        _build_masked_bilinear_stencil(
            np.array([[2.25]]), np.array([[2.25]]), source_land,
            target_land, fallback_radius=1)


def test_unresolved_donor_search_reports_the_radius_that_works():
    """The refusal carries the facts remediation must be computed from.

    Field 2026-08: "no valid surface-matched HRRR donor within 8 cells
    for 2 target point(s)" was followed by advice to raise the radius,
    and the recommended raise was impossible.  Validating advice needs
    the failure to say WHAT radius would reach a donor and WHICH target
    cells failed -- both known here and nowhere else.
    """
    source_land = np.zeros((12, 12), dtype=bool)
    source_land[0, 0] = True
    target_land = np.ones((2, 2), dtype=bool)
    x = np.array([[0.0, 7.0], [0.0, 7.0]])
    y = np.array([[0.0, 0.0], [7.0, 7.0]])

    with pytest.raises(ValueError, match="no valid surface-matched") \
            as failure:
        _build_masked_bilinear_stencil(
            x, y, source_land, target_land, fallback_radius=8)
    error = failure.value
    # (0,0) resolves directly; (0,7) and (7,0) are exactly 7 away and
    # resolve by fallback; (7,7) is sqrt(98) = 9.90 away, so radius 10
    # is the smallest integer radius whose disk holds a valid donor.
    assert error.fallback_radius_cells == 8
    assert error.required_radius_cells == 10
    assert error.unresolved_targets == ((1, 1),)

    # Negative control: a window with no valid donor at ANY radius says
    # so, rather than inventing a radius that cannot work.
    with pytest.raises(ValueError, match="no valid surface-matched") \
            as hopeless:
        _build_masked_bilinear_stencil(
            np.array([[5.0]]), np.array([[5.0]]),
            np.zeros((12, 12), dtype=bool), np.ones((1, 1), dtype=bool),
            fallback_radius=8)
    assert hopeless.value.required_radius_cells is None
    assert hopeless.value.unresolved_targets == ((0, 0),)


def test_ohio_like_lake_edge_requires_explicit_radius_ten():
    source_land = np.zeros((12, 12), dtype=bool)
    source_land[0, 0] = True
    target_land = np.ones((1, 1), dtype=bool)
    x = np.array([[7.0]])
    y = np.array([[7.0]])

    with pytest.raises(ValueError, match="within 8 cells"):
        _build_masked_bilinear_stencil(
            x, y, source_land, target_land, fallback_radius=8)

    iy, ix, weights, report = _build_masked_bilinear_stencil(
        x, y, source_land, target_land, fallback_radius=10)
    assert (iy[0, 0, 0], ix[0, 0, 0]) == (0, 0)
    np.testing.assert_array_equal(
        weights[:, 0, 0], [1.0, 0.0, 0.0, 0.0])
    assert report["fallback_radius_cells"] == 10
    assert report["fallback_max_distance_cells"] == pytest.approx(np.sqrt(98.0))
    assert report["fallback_distance_ceiling_histogram_cells"] == {"10": 1}
    assert report["unresolved_target_count"] == 0
    assert report["cross_surface_donor_count"] == 0
