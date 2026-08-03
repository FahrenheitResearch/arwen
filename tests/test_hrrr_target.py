from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pytest

import gpuwm.ingest.horiz as horiz
import gpuwm.ingest.hrrr as hrrr
from gpuwm.ingest.hrrr import HrrrNativeSnapshot, interpolate_hrrr_to_lambert
from gpuwm.ingest.hrrr_target import (
    HRRR_SOURCE_NX,
    HRRR_SOURCE_NY,
    HrrrTargetDomain,
    TARGET_DOMAIN_SCHEMA,
    load_hrrr_target_domain,
    required_hrrr_source_window,
)
from tools.write_hrrr_stock_wrf_namelist import render_namelist
from tools.hrrr_build_native_static import validate_static
from tools.seal_hrrr_native_bridge import _series_hours, _window_shape


def _target(**updates) -> HrrrTargetDomain:
    values = {
        "name": "relocated_test",
        "map_proj": "lambert",
        "nx": 24,
        "ny": 18,
        "nz": 49,
        "dx_m": 3000.0,
        "dy_m": 3000.0,
        "ref_lat": 39.7,
        "ref_lon": -84.0,
        "truelat1": 30.0,
        "truelat2": 60.0,
        "stand_lon": -85.0,
        "time_step_seconds": 15,
        "spec_bdy_width": 5,
        "spec_zone": 1,
        "relax_zone": 4,
    }
    values.update(updates)
    return HrrrTargetDomain(**values)


def test_target_domain_load_is_strict_and_shape_is_parameterized(tmp_path):
    target = _target(nx=37, ny=29, dx_m=2500.0, dy_m=2500.0)
    path = tmp_path / "target.json"
    path.write_text(json.dumps(target.to_payload()))
    restored = load_hrrr_target_domain(path)
    assert restored == target
    assert restored.grid().latlon_mass()[0].shape == (29, 37)
    assert len(restored.identity_sha256()) == 64

    payload = target.to_payload()
    payload["unexpected"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="extra=.*unexpected"):
        load_hrrr_target_domain(path)


def test_target_v1_omission_preserves_radius_eight_compatibility(tmp_path):
    payload = _target().to_payload()
    payload.pop("surface_fallback_radius_cells")
    path = tmp_path / "legacy-v1-target.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = load_hrrr_target_domain(path)

    assert restored.surface_fallback_radius_cells == 8
    assert restored.to_payload()["surface_fallback_radius_cells"] == 8


def test_packaged_hrrr_targets_load_with_bound_effective_radius():
    configs = Path(__file__).parents[1] / "configs"
    observed = {
        name: load_hrrr_target_domain(configs / name)
        for name in (
            "hrrr_target_oklahoma_192x160_3km.json",
            "hrrr_target_oklahoma_1000x1000_1km.json",
            "hrrr_target_ohio_192x160_3km.json",
        )
    }

    assert observed[
        "hrrr_target_oklahoma_192x160_3km.json"
    ].surface_fallback_radius_cells == 8
    assert observed[
        "hrrr_target_oklahoma_1000x1000_1km.json"
    ].surface_fallback_radius_cells == 8
    assert observed[
        "hrrr_target_ohio_192x160_3km.json"
    ].surface_fallback_radius_cells == 10


def test_target_domain_rejects_unsupported_projection_and_bad_geometry():
    with pytest.raises(ValueError, match="other projection families"):
        _target(map_proj="mercator")
    with pytest.raises(ValueError, match="dx_m == dy_m"):
        _target(dx_m=3000.0, dy_m=2999.0)
    with pytest.raises(ValueError, match="at least 4"):
        _target(nz=3)
    with pytest.raises(ValueError, match=r"must be in \[0, 64\]"):
        _target(surface_fallback_radius_cells=65)
    with pytest.raises(TypeError, match="must be an integer"):
        _target(surface_fallback_radius_cells=True)


@pytest.mark.parametrize("nz", [4, 17, 49, 80, 113])
def test_target_domain_vertical_count_is_parameterized(nz):
    target = _target(nz=nz)
    assert target.nz == nz


def test_required_window_covers_parabolic_and_surface_halos():
    legacy = required_hrrr_source_window(
        HrrrTargetDomain.legacy_500x500())
    assert legacy.bridge_tuple() == (793, 975, 327, 509)
    assert (legacy.ny, legacy.nx) == (183, 183)
    assert legacy.surface_fallback_radius_cells == 8

    relocated = required_hrrr_source_window(_target())
    assert 0 <= relocated.i_start <= relocated.i_end < 1799
    assert 0 <= relocated.j_start <= relocated.j_end < 1059
    assert relocated.bridge_tuple() != legacy.bridge_tuple()

    with pytest.raises(ValueError, match="leaves HRRR coverage"):
        required_hrrr_source_window(
            _target(ref_lat=5.0, ref_lon=20.0))


def test_target_radius_is_single_source_window_policy():
    target = _target(surface_fallback_radius_cells=10)
    window = required_hrrr_source_window(target)

    assert window.surface_fallback_radius_cells == 10
    assert window.to_dict()["surface_fallback_radius_cells"] == 10
    with pytest.raises(ValueError, match="differs from the target-domain"):
        required_hrrr_source_window(target, surface_fallback_radius=8)


def test_near_edge_fallback_bounds_do_not_request_ineligible_cell():
    source = hrrr.hrrr_source_grid()
    # With nx=13, target mass x=1 maps to zero-based q=7.49.  The exact
    # radius-eight donor bound is ceil(7.49-8)=0; round(q)-8 would falsely
    # request -1 and reject this valid near-edge target.
    ref_lat, ref_lon = source.ij_to_latlon(14.49, 500.0)
    target = _target(
        nx=13,
        ny=13,
        dx_m=source.dx,
        dy_m=source.dy,
        ref_lat=float(ref_lat),
        ref_lon=float(ref_lon),
        truelat1=38.5,
        truelat2=38.5,
        stand_lon=-97.5,
    )
    window = required_hrrr_source_window(target)
    assert window.i_start == 0


def test_source_grid_shape_and_rotation_have_independent_literal_anchor():
    source = hrrr.hrrr_source_grid()
    assert (source.e_we - 1, source.e_sn - 1) == (
        HRRR_SOURCE_NX, HRRR_SOURCE_NY) == (1799, 1059)
    snapshot = HrrrNativeSnapshot(
        valid_time=datetime(2026, 7, 18),
        forecast_hour=0,
        i_start=0,
        j_start=0,
        ny=1,
        nx=1,
        fields={},
    )
    sina, cosa = hrrr._source_window_rotation(snapshot)
    assert sina[0, 0] == pytest.approx(0.2705924701431765, abs=2e-15)
    assert cosa[0, 0] == pytest.approx(0.9626939882962883, abs=2e-15)


def test_projected_plan_support_and_nearest_are_crop_invariant(monkeypatch):
    monkeypatch.setattr(hrrr, "_cupy", lambda: np)
    # qx is exactly half-integer: floor(q+0.5) must pick global donor 101
    # independently of even/odd crop-origin parity.  qy just below integer
    # also proves the donor floor is selected in FP64 before FP32 weights.
    qx = 100.5
    qy = 200.0 - 1.0e-7

    class ExactSourceCoordinates:
        @staticmethod
        def latlon_to_ij(_latitude, _longitude):
            return np.asarray([[qx + 1.0]]), np.asarray([[qy + 1.0]])

    monkeypatch.setattr(
        hrrr, "hrrr_source_grid", lambda: ExactSourceCoordinates())
    target_lat = np.zeros((1, 1))
    target_lon = np.zeros((1, 1))
    selected = []
    for origin in (90, 91):
        snapshot = HrrrNativeSnapshot(
            valid_time=datetime(2026, 7, 18),
            forecast_hour=0,
            i_start=origin,
            j_start=190,
            ny=24,
            nx=24,
            fields={},
        )
        plan = hrrr._ProjectedGpuPlan(snapshot, target_lat, target_lon)
        field = np.broadcast_to(
            np.arange(origin, origin + snapshot.nx, dtype=np.float32),
            (snapshot.ny, snapshot.nx),
        )
        selected.append(float(plan.apply(field, method="nearest")[0, 0]))
        assert int(plan.iy[0, 0]) + snapshot.j_start == 199
    assert selected == [101.0, 101.0]


def test_projected_cpu_plan_preserves_fp64_donor_selection(monkeypatch):
    qx = 100.5
    qy = 200.0 - 1.0e-7

    class ExactSourceCoordinates:
        @staticmethod
        def latlon_to_ij(_latitude, _longitude):
            return np.asarray([[qx + 1.0]]), np.asarray([[qy + 1.0]])

    monkeypatch.setattr(
        hrrr, "hrrr_source_grid", lambda: ExactSourceCoordinates())
    target_lat = np.zeros((1, 1))
    target_lon = np.zeros((1, 1))
    selected = []
    for origin in (90, 91):
        snapshot = HrrrNativeSnapshot(
            valid_time=datetime(2026, 7, 18), forecast_hour=0,
            i_start=origin, j_start=190, ny=24, nx=24, fields={})
        plan = hrrr._ProjectedCpuPlan(
            snapshot, target_lat, target_lon, object())
        field = np.broadcast_to(
            np.arange(origin, origin + snapshot.nx, dtype=np.float32),
            (snapshot.ny, snapshot.nx))
        selected.append(float(plan.apply(field, method="nearest")[0, 0]))
        assert int(plan.iy[0, 0]) + snapshot.j_start == 199
    assert selected == [101.0, 101.0]


def _constant_earth_wind_snapshot(target: HrrrTargetDomain):
    window = required_hrrr_source_window(target)
    ny, nx = window.ny, window.nx
    zeros_3d = np.zeros((50, ny, nx), dtype=np.float32)
    zeros_2d = np.zeros((ny, nx), dtype=np.float32)
    fields = {
        "PRES": np.full_like(zeros_3d, 80_000.0),
        "QC": zeros_3d.copy(),
        "QI": zeros_3d.copy(),
        "QR": zeros_3d.copy(),
        "QS": zeros_3d.copy(),
        "QG": zeros_3d.copy(),
        "HGT": np.full_like(zeros_3d, 1000.0),
        "TT": np.full_like(zeros_3d, 280.0),
        "SPFH": np.full_like(zeros_3d, 0.005),
        "PSFC": np.full_like(zeros_2d, 95_000.0),
        "SOILHGT": np.full_like(zeros_2d, 200.0),
        "SKINTEMP": np.full_like(zeros_2d, 285.0),
        "SNOW": zeros_2d.copy(),
        "SNOWH": zeros_2d.copy(),
        "T2": np.full_like(zeros_2d, 284.0),
        "Q2": np.full_like(zeros_2d, 0.006),
        "LANDSEA": np.ones_like(zeros_2d),
        "XICE": zeros_2d.copy(),
        "SOILT": np.full((9, ny, nx), 284.0, dtype=np.float32),
        "SOILW": np.full((9, ny, nx), 0.25, dtype=np.float32),
    }
    snapshot = HrrrNativeSnapshot(
        valid_time=datetime(2026, 7, 18),
        forecast_hour=0,
        i_start=window.i_start,
        j_start=window.j_start,
        ny=ny,
        nx=nx,
        fields=fields,
    )
    source_sina, source_cosa = hrrr._source_window_rotation(snapshot)
    u_earth = 12.0
    v_earth = -3.0
    u_grid = u_earth * source_cosa + v_earth * source_sina
    v_grid = v_earth * source_cosa - u_earth * source_sina
    fields["U_MASS"] = np.broadcast_to(
        u_grid, (50, ny, nx)).astype(np.float32).copy()
    fields["V_MASS"] = np.broadcast_to(
        v_grid, (50, ny, nx)).astype(np.float32).copy()
    fields["U10_MASS"] = u_grid.astype(np.float32)
    fields["V10_MASS"] = v_grid.astype(np.float32)
    return HrrrNativeSnapshot(
        valid_time=snapshot.valid_time,
        forecast_hour=0,
        i_start=window.i_start,
        j_start=window.j_start,
        ny=ny,
        nx=nx,
        fields=fields,
    ), u_earth, v_earth


def test_grid_relative_winds_rotate_through_earth_basis(monkeypatch):
    # The interpolation kernels use only the NumPy-compatible CuPy surface in
    # this compact constant-vector test.
    monkeypatch.setattr(hrrr, "_cupy", lambda: np)
    monkeypatch.setattr(horiz, "_cupy", lambda: np)
    monkeypatch.setattr(np, "asnumpy", np.asarray, raising=False)
    target = _target(
        nx=16,
        ny=14,
        ref_lat=35.5,
        ref_lon=-98.0,
        truelat1=30.0,
        truelat2=60.0,
        stand_lon=-88.0,
    )
    snapshot, u_earth, v_earth = _constant_earth_wind_snapshot(target)
    mapped = interpolate_hrrr_to_lambert(
        snapshot,
        target.grid(),
        target_landmask=np.ones((target.ny, target.nx)),
    )
    sina_u, cosa_u = horiz.lambert_rotation(target.grid(), "u")
    sina_v, cosa_v = horiz.lambert_rotation(target.grid(), "v")
    expected_u = u_earth * cosa_u + v_earth * sina_u
    expected_v = v_earth * cosa_v - u_earth * sina_v
    np.testing.assert_allclose(mapped.fields["UU"][0], expected_u, atol=2e-5)
    np.testing.assert_allclose(mapped.fields["VV"][0], expected_v, atol=2e-5)
    np.testing.assert_allclose(mapped.fields["U10"], expected_u, atol=2e-5)
    np.testing.assert_allclose(mapped.fields["V10"], expected_v, atol=2e-5)
    assert mapped.fields["UU"].shape == (50, target.ny, target.nx + 1)
    assert mapped.fields["VV"].shape == (50, target.ny + 1, target.nx)


class HostBackend:
    """A device-free preprocess backend, shared by the tests below."""

    name = "cpu-test"
    array_module = np

    @staticmethod
    def float32(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def bool_array(value):
        return np.asarray(value, dtype=bool)

    @staticmethod
    def rotate_earth_to_grid(u, v, sina, cosa):
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        sina = np.asarray(sina, dtype=np.float32)
        cosa = np.asarray(cosa, dtype=np.float32)
        return u * cosa + v * sina, v * cosa - u * sina

    @staticmethod
    def receipt():
        return {"backend": "cpu-test", "implementation": "numpy"}

    # Required by the common backend-object contract; HRRR does not use
    # regular-grid mapping, masked nearest, or vertical setup here.
    regular_plan = masked_nearest = era5_rh_to_water = staticmethod(
        lambda *_args, **_kwargs: None)
    prepare_wrf_vertical = staticmethod(lambda *_args, **_kwargs: None)


def _mapped_with_landmask(landmask, *, source_land=True,
                          target_name="domain 1"):
    """Map one constant snapshot onto LANDMASK, recording the soil report."""
    target = _target(
        nx=16, ny=14, ref_lat=35.5, ref_lon=-98.0,
        truelat1=30.0, truelat2=60.0, stand_lon=-88.0)
    snapshot, _, _ = _constant_earth_wind_snapshot(target)
    if not source_land:
        fields = dict(snapshot.fields)
        fields["LANDSEA"] = np.zeros_like(fields["LANDSEA"])
        snapshot = HrrrNativeSnapshot(
            valid_time=snapshot.valid_time,
            forecast_hour=snapshot.forecast_hour,
            i_start=snapshot.i_start, j_start=snapshot.j_start,
            ny=snapshot.ny, nx=snapshot.nx, fields=fields)
    report: dict = {}
    mapped = interpolate_hrrr_to_lambert(
        snapshot, target.grid(), target_landmask=landmask,
        soil_mapping_report=report, backend=HostBackend(),
        target_name=target_name)
    return mapped, report


def test_an_all_water_target_maps_its_soil_instead_of_refusing():
    """A coastal domain's ocean boundary strip is a legal configuration.

    The four specified-boundary strips are mapped by independent calls,
    so one Pacific-facing west strip with no land in it used to abort a
    whole preparation -- with a sentence that named neither the strip nor
    the domain.  Nothing about the mapping needs land here: every soil
    cell of a water target is the target-water fill the mapper already
    writes (SOILT = SKINTEMP, SOILW = 1), and no land donor is consulted.
    """
    mapped, report = _mapped_with_landmask(
        np.zeros((14, 16)), target_name="the west boundary strip of domain 1")

    # The water fill, everywhere, exactly as the policy says.
    np.testing.assert_allclose(
        np.asarray(mapped.fields["SOILT"]),
        np.broadcast_to(np.asarray(mapped.fields["SKINTEMP"])[None, :, :],
                        np.asarray(mapped.fields["SOILT"]).shape))
    np.testing.assert_allclose(np.asarray(mapped.fields["SOILW"]), 1.0)

    # And the receipt says so, naming the strip, rather than going silent.
    assert report["target"] == "the west boundary strip of domain 1"
    assert report["target_land_count"] == 0
    for field in ("SOILT", "SOILW"):
        entry = report["fields"][field]
        assert entry["land_window_statistics"] is None
        assert entry["target_land_cell_count"] == 0
        assert "no land cells" in entry["land_window_statistics_absent_because"]
        # The admission checks that need no land still ran: before this
        # fix the empty-land refusal came first and skipped them.
        assert "all_target_minimum" in entry


def test_a_land_target_with_no_reachable_source_land_refuses_by_name():
    """The case that genuinely cannot be mapped, named and remedied.

    Watched firing: the same geometry with source land present is the
    test above's sibling, and it maps.  Here the target has land cells
    and the HRRR window has none, so no donor exists at any radius --
    the refusal must say WHICH grid and what to do about it, and must
    NOT tell the reader to raise a radius no radius can satisfy.
    """
    with pytest.raises(ValueError) as refusal:
        _mapped_with_landmask(np.ones((14, 16)), source_land=False,
                              target_name="domain 2")
    message = str(refusal.value)
    assert "domain 2" in message
    assert "surface_fallback_radius_cells" in message
    assert "target point(s)" in message
    assert "cannot help" in message
    assert "Raising surface_fallback_radius_cells to" not in message


def _snapshot_with_water_rows(target, water_rows):
    """The constant snapshot with whole window rows turned to water."""
    snapshot, _, _ = _constant_earth_wind_snapshot(target)
    fields = dict(snapshot.fields)
    landsea = np.array(fields["LANDSEA"], copy=True)
    landsea[water_rows, :] = 0.0
    fields["LANDSEA"] = landsea
    return HrrrNativeSnapshot(
        valid_time=snapshot.valid_time,
        forecast_hour=snapshot.forecast_hour,
        i_start=snapshot.i_start, j_start=snapshot.j_start,
        ny=snapshot.ny, nx=snapshot.nx, fields=fields)


def _donor_truth(target, landsea, radius):
    """Independent recount: unresolved land targets and the radius that
    reaches a donor for all of them (None when no donor exists at all).

    Mirrors the stencil builder's DEFINITION -- integer source cells,
    surface-matched, distance**2 <= radius**2 -- from the projection
    geometry alone, so the error's own numbers have a second witness.
    """
    source = hrrr.hrrr_source_grid()
    window = required_hrrr_source_window(target)
    lat, lon = target.grid().latlon_mass()
    sx, sy = source.latlon_to_ij(lat, lon)
    x = np.asarray(sx, dtype=np.float64) - 1.0 - window.i_start
    y = np.asarray(sy, dtype=np.float64) - 1.0 - window.j_start
    valid_y, valid_x = np.nonzero(np.asarray(landsea) >= 0.5)
    unresolved, worst = [], 0.0
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            best = float(np.min(
                (valid_x - x[row, col]) ** 2 + (valid_y - y[row, col]) ** 2))
            if best > float(radius) ** 2:
                unresolved.append((row, col))
                worst = max(worst, best)
    required = int(np.ceil(np.sqrt(worst))) if unresolved else None
    return tuple(unresolved), required


def test_soil_donor_refusal_recommends_a_radius_the_guard_accepts():
    """Remediation advice is validated before it is printed (half one).

    An interior domain with a lake-like all-water band: some land
    target cells have no donor within the configured 8 cells, but a
    donor exists a couple of cells further and the enlarged source
    window still fits HRRR's native grid.  The refusal must name that
    exact radius, and the radius it names must pass the coverage guard
    it would be checked by (required_hrrr_source_window).
    """
    from dataclasses import replace

    target = _target(
        nx=16, ny=14, ref_lat=35.5, ref_lon=-98.0,
        truelat1=30.0, truelat2=60.0, stand_lon=-88.0)
    snapshot, _, _ = _constant_earth_wind_snapshot(target)
    # A water band across the window, centered on the target's middle
    # rows, wide enough to defeat radius 8 but not radius 10.
    window = required_hrrr_source_window(target)
    lat, _lon = target.grid().latlon_mass()
    source = hrrr.hrrr_source_grid()
    _sx, sy = source.latlon_to_ij(*target.grid().latlon_mass())
    y_mid = float(np.asarray(sy)[7, 8]) - 1.0 - window.j_start
    band = [row for row in range(snapshot.ny)
            if abs(row - y_mid) <= 9.4]
    carved = _snapshot_with_water_rows(target, band)

    unresolved, required = _donor_truth(
        target, carved.fields["LANDSEA"], 8)
    assert unresolved, "the carve must defeat the configured radius"
    assert required is not None and required > 8
    # Precondition: the raise is feasible -- the window grown by the
    # difference stays inside the native grid.
    growth = required - 8
    assert carved.j_start - growth >= 0
    assert carved.j_start + carved.ny - 1 + growth <= HRRR_SOURCE_NY - 1

    with pytest.raises(ValueError) as refusal:
        interpolate_hrrr_to_lambert(
            carved, target.grid(),
            target_landmask=np.ones((target.ny, target.nx)),
            backend=HostBackend(), target_name="domain 1")
    message = str(refusal.value)
    assert "domain 1" in message
    assert f"Raising surface_fallback_radius_cells to {required}" in message
    # The advice passes the guard it names.
    required_hrrr_source_window(
        replace(target, surface_fallback_radius_cells=required))


def test_soil_donor_refusal_computes_the_trim_when_no_radius_can_pass():
    """Remediation advice is validated before it is printed (half two).

    The field shape: a fitter-maximum domain whose radius-8 window sits
    exactly on HRRR's native top edge.  Unfillable land cells near the
    top cannot be remedied by ANY radius the coverage guard accepts, so
    the refusal must not recommend the knob -- it must compute the trim
    that removes every unfillable cell, and say which side.
    """
    from dataclasses import replace

    source = hrrr.hrrr_source_grid()
    target = None
    for j_center in np.arange(1041.0, 1053.0, 0.25):
        ref_lat, ref_lon = source.ij_to_latlon(900.0, float(j_center))
        candidate = _target(
            nx=16, ny=14, dx_m=3000.0, dy_m=3000.0,
            ref_lat=float(ref_lat), ref_lon=float(ref_lon),
            truelat1=38.5, truelat2=38.5, stand_lon=-97.5)
        try:
            window = required_hrrr_source_window(candidate)
        except ValueError:
            continue
        if window.j_end == HRRR_SOURCE_NY - 1:
            target = candidate
            break
    assert target is not None, "no aligned target found on the top edge"

    snapshot, _, _ = _constant_earth_wind_snapshot(target)
    assert snapshot.j_start + snapshot.ny - 1 == HRRR_SOURCE_NY - 1
    # Water from just below the topmost target rows to the window top:
    # their nearest donor sits ~10 cells south, and radius 10 needs
    # source rows past the native edge -- the impossible raise.
    _sx, sy = source.latlon_to_ij(*target.grid().latlon_mass())
    y_top = float(np.asarray(sy)[-1, 8]) - 1.0 - snapshot.j_start
    band = [row for row in range(snapshot.ny) if row > y_top - 9.4]
    carved = _snapshot_with_water_rows(target, band)

    unresolved, required = _donor_truth(
        target, carved.fields["LANDSEA"], 8)
    assert unresolved and required is not None
    growth = required - 8
    assert (snapshot.j_start + snapshot.ny - 1 + growth
            > HRRR_SOURCE_NY - 1), "the raise must be infeasible here"
    # The knob setting that reaches the donors is refused by the guard
    # -- this is the constraint the printed advice must respect.
    with pytest.raises(ValueError, match="leaves HRRR coverage"):
        required_hrrr_source_window(
            replace(target, surface_fallback_radius_cells=required))

    with pytest.raises(ValueError) as refusal:
        interpolate_hrrr_to_lambert(
            carved, target.grid(),
            target_landmask=np.ones((target.ny, target.nx)),
            backend=HostBackend(), target_name="domain 1")
    message = str(refusal.value)
    assert "Raising surface_fallback_radius_cells to" not in message
    assert "cannot work here" in message
    # The trim is computed, not guessed: exactly the rows that carry
    # unfillable cells, counted from the north (j-max) side.
    trim = target.ny - min(row for row, _col in unresolved)
    assert f"trim {trim} cell(s) from its north (j-max) side" in message
    assert all(row >= target.ny - trim for row, _col in unresolved)


def test_a_mixed_land_water_target_still_reports_land_statistics():
    """Negative control: land present, so the diagnostics are not skipped."""
    landmask = np.ones((14, 16))
    landmask[:, :5] = 0.0                      # an ocean west edge
    _mapped, report = _mapped_with_landmask(landmask)
    assert report["target_land_count"] == 14 * 11
    entry = report["fields"]["SOILT"]
    assert entry["convex_bounds_conserved"] is True
    assert "land_window_statistics" not in entry


def test_full_hrrr_host_backend_is_deterministic_and_device_free():
    class HostBackend:
        name = "cpu-test"
        array_module = np

        @staticmethod
        def float32(value):
            return np.asarray(value, dtype=np.float32)

        @staticmethod
        def bool_array(value):
            return np.asarray(value, dtype=bool)

        @staticmethod
        def rotate_earth_to_grid(u, v, sina, cosa):
            u = np.asarray(u, dtype=np.float32)
            v = np.asarray(v, dtype=np.float32)
            sina = np.asarray(sina, dtype=np.float32)
            cosa = np.asarray(cosa, dtype=np.float32)
            return u * cosa + v * sina, v * cosa - u * sina

        @staticmethod
        def receipt():
            return {"backend": "cpu-test", "implementation": "numpy"}

        # Required by the common backend-object contract; HRRR does not use
        # regular-grid mapping, masked nearest, or vertical setup here.
        regular_plan = masked_nearest = era5_rh_to_water = staticmethod(
            lambda *_args, **_kwargs: None)
        prepare_wrf_vertical = staticmethod(lambda *_args, **_kwargs: None)

    target = _target(
        nx=16, ny=14, ref_lat=35.5, ref_lon=-98.0,
        truelat1=30.0, truelat2=60.0, stand_lon=-88.0)
    snapshot, _, _ = _constant_earth_wind_snapshot(target)
    kwargs = dict(
        target_landmask=np.ones((target.ny, target.nx)),
        backend=HostBackend())
    first = interpolate_hrrr_to_lambert(snapshot, target.grid(), **kwargs)
    second = interpolate_hrrr_to_lambert(snapshot, target.grid(), **kwargs)
    assert set(first.fields) == set(second.fields)
    for name in first.fields:
        assert isinstance(first.fields[name], np.ndarray)
        np.testing.assert_array_equal(first.fields[name], second.fields[name])


def test_target_schema_constant_is_stable():
    assert TARGET_DOMAIN_SCHEMA == "gpuwm-hrrr-target-domain-v1"


def test_hrrr_source_specific_humidity_has_physical_envelope():
    fields = {
        "LANDSEA": np.ones((2, 2)),
        "SOILW": np.ones((1, 2, 2)),
        "SOILT": np.full((1, 2, 2), 280.0),
        "SPFH": np.full((1, 2, 2), 0.02),
        "Q2": np.full((2, 2), 0.01),
    }
    hrrr._require_source_physical_ranges(fields)
    fields["SPFH"][0, 0, 0] = 0.100001
    with pytest.raises(ValueError, match="SPFH.*outside 0..0.1"):
        hrrr._require_source_physical_ranges(fields)
    fields["SPFH"][0, 0, 0] = 0.02
    fields["Q2"][0, 0] = -1.0e-6
    with pytest.raises(ValueError, match="Q2.*outside 0..0.1"):
        hrrr._require_source_physical_ranges(fields)


def test_bridge_seal_window_shape_is_dynamic_and_strict():
    assert _window_shape("180x212") == "180x212"
    for invalid in ("0x212", "180X212", "180x", " 180x212"):
        with pytest.raises(Exception, match="positive NYxNX"):
            _window_shape(invalid)


def test_bridge_seal_series_horizon_is_dynamic_and_strict(tmp_path):
    series = tmp_path / "series.tsv"
    series.write_text("0\ta0\ts0\n1\ta1\ts1\n")
    assert _series_hours(series) == (0, 1)

    series.write_text("0\ta0\ts0\n2\ta2\ts2\n")
    with pytest.raises(ValueError, match="expected contiguous"):
        _series_hours(series)

    series.write_text("0\ta0\ts0\n")
    with pytest.raises(ValueError, match="at least two hourly frames"):
        _series_hours(series)


@pytest.mark.parametrize("nz", [4, 17, 49, 80, 113])
def test_stock_wrf_acceptance_namelist_uses_target_geometry_and_vertical(nz):
    target = _target(nx=37, ny=29, nz=nz, time_step_seconds=11)
    text = render_namelist(
        target=target,
        eta=np.linspace(1.0, 0.0, nz + 1),
        p_top=12_345.0,
        etac=0.37,
        valid_time=datetime(2026, 7, 18, 23, 59, 55),
        run_seconds=10,
    )
    assert "time_step                           = 11," in text
    assert "e_we                                = 38," in text
    assert "e_sn                                = 30," in text
    assert f"e_vert                              = {nz + 1}," in text
    assert "p_top_requested                     = 12345," in text
    assert "etac                                = 0.37," in text
    assert "end_day                           = 19," in text
    assert "end_second                        = 05," in text


def test_static_validation_uses_declared_target_staggering():
    target = _target(nx=17, ny=15)
    mass = np.ones((target.ny, target.nx), dtype=np.float64)
    fields = {
        "HGT_M": mass.copy(),
        "LANDMASK": mass.copy(),
        "LU_INDEX": mass.copy(),
        "SCT_DOM": mass.copy(),
        "SOILTEMP": 280.0 * mass,
        "SNOALB": 50.0 * mass,
        "GREENFRAC": np.ones((12, target.ny, target.nx)),
        "LAI12M": np.ones((12, target.ny, target.nx)),
        "MAPFAC_M": mass.copy(),
        "MAPFAC_U": np.ones((target.ny, target.nx + 1)),
        "MAPFAC_V": np.ones((target.ny + 1, target.nx)),
        "F": mass.copy(),
        "E": mass.copy(),
        "SINALPHA": np.zeros_like(mass),
        "COSALPHA": mass.copy(),
    }
    assert validate_static(fields, target)["field_count"] == len(fields)
    zero_terrain = dict(fields)
    zero_terrain["HGT_M"] = np.zeros_like(mass)
    with pytest.raises(ValueError, match="identically zero over every land"):
        validate_static(zero_terrain, target)
    fields["MAPFAC_U"] = np.ones((target.ny, target.nx))
    with pytest.raises(ValueError, match="MAPFAC_U stagger"):
        validate_static(fields, target)
