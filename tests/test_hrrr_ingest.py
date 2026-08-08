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
    _wps_oned_cpu,
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


#: The two HRRR 2 m specific-humidity stencils that refused a nested run.
#:
#: gpuwm 1.8.4, HRRR cycle 2026-08-08T09 f00, nested 12-3 km tree centred
#: 39.0,-103.0.  Both preparation stages passed and the tree forecast then
#: refused its own d02 input: "prepared near-surface surface_qv is outside
#: the physical range 0.0..0.2".  These are the source values behind it,
#: read out of that run's native bridge window -- HRRR's GRIB2 packing
#: quantises Q2 to 1e-5, so over the San Juan Mountains (source SOILHGT
#: 2557..3535 m) it decodes to EXACTLY zero beside neighbours three orders
#: of magnitude larger.  WPS routes SPECHUMD through the overshooting
#: sixteen_pt operator (METGRID.TBL
#: interp_option=sixteen_pt+four_pt+average_4pt), which undershoots such a
#: stencil below zero.  Real numbers, not a constructed contrast: nothing
#: synthetic reproduces how flat the dry side of these stencils is.
_SAN_JUAN_Q2_STENCILS = (
    # d02 (j=64, i=32); 37.75206 N, 106.58660 W; target terrain 3061.6 m
    {
        "stencil": np.array([
            [3.9e-04, 4.1e-04, 5.4e-04, 7.6e-04],
            [1.8e-04, 3.0e-05, 0.0e+00, 3.9e-04],
            [2.5e-04, 8.0e-05, 1.5e-04, 8.5e-04],
            [2.3e-04, 4.1e-04, 5.9e-04, 9.2e-04],
        ], dtype=np.float32),
        "fx": np.float32(0.2811026275),
        "fy": np.float32(0.513708055),
        "mapped": -1.785677523e-05,
    },
    # d02 (j=65, i=28); 37.77501 N, 106.72648 W; target terrain 3226.5 m
    {
        "stencil": np.array([
            [1.48e-03, 5.60e-04, 7.00e-05, 1.20e-04],
            [1.40e-04, 0.00e+00, 1.00e-05, 6.00e-05],
            [0.00e+00, 0.00e+00, 1.00e-05, 1.40e-04],
            [1.00e-04, 7.00e-05, 5.00e-05, 2.40e-04],
        ], dtype=np.float32),
        "fx": np.float32(0.2865612507),
        "fy": np.float32(0.767305851),
        "mapped": -1.478769718e-05,
    },
)


def _sixteen_point(stencil, fx, fy):
    """Run the shipped operator over one 4x4 stencil, as ``apply`` does.

    ``_ProjectedCpuPlan.apply`` substitutes 1e-20 for exact zeros before
    ``oned`` and maps an exact 1e-20 result back to zero; that is WPS's
    own REAL*4 quirk (interp_module.F:1255-1257,1299) and it is what
    makes a stencil containing zeros run the full overshooting parabolic
    instead of collapsing to WPS's ``b*c == 0`` zero.  Reproduced here so
    the pin exercises the same arithmetic the mapper does.
    """
    tiny = np.float32(1.0e-20)
    zero = np.float32(0.0)
    values = np.where(stencil == zero, tiny, stencil)
    rows = [_wps_oned_cpu(fx, *values[row]) for row in range(4)]
    result = _wps_oned_cpu(fy, *rows)
    return np.where(result == tiny, zero, result)


@pytest.mark.parametrize("case", _SAN_JUAN_Q2_STENCILS)
def test_wps_sixteen_point_undershoots_zero_valued_hrrr_surface_moisture(case):
    """The WPS operator really does go negative here -- pin it, do not fix it.

    METGRID.TBL puts SPECHUMD on ``sixteen_pt``, so this undershoot is
    what WPS produces and what ``real.exe`` then carries into ``grid%q2``
    unfloored (module_initialize_real.F:1157, :1257).  The engine's answer
    is a floor on the published surface value
    (:func:`gpuwm.ingest.real._floor_flag_sh_surface_mixing_ratio`), NOT a
    quietly de-overshot operator: swapping this for a non-negative
    interpolator would change every HRRR field's numbers and diverge from
    WPS for no stated reason.  If this test starts failing because the
    result is no longer negative, the operator was changed.
    """
    mapped = _sixteen_point(case["stencil"], case["fx"], case["fy"])

    assert case["stencil"].min() == 0.0
    assert mapped < 0.0
    assert float(mapped) == pytest.approx(case["mapped"], rel=1e-6)


def test_zero_free_hrrr_surface_moisture_stencil_stays_positive():
    """The same operator on the same shape of stencil, minus the zeros.

    The control for the pin above: what makes those two cells negative is
    the exact zeros, not the terrain and not the operator on its own.  A
    stencil with the same span whose floor is a real HRRR value maps well
    clear of zero, which is why the flat-terrain Oklahoma tree at the same
    release completes.
    """
    case = _SAN_JUAN_Q2_STENCILS[0]
    lifted = np.maximum(case["stencil"], np.float32(8.0e-4))

    mapped = _sixteen_point(lifted, case["fx"], case["fy"])

    assert float(mapped) > 0.0
