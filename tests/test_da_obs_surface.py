"""The ASOS surface adapter, against the real seam and the real filter.

Three authorities are exercised and none is imitated:

* the RECORD is the real ``rw_asos`` writer's own output over a subset of
  a real archived IEM fetch (``tests/fixtures/asos_surface_real``, see its
  README for the provenance chain), and when ``GPUWM_RW_ASOS`` names a
  binary the decode is re-run against the committed CSV and compared —
  the reader is tested against the other lane's writer, never against a
  fixture its own lane typed;
* the GRID is a real WRF Lambert projection over the stations, so
  placement goes through the same ``mass_index``/``inside`` authority the
  radar path binds to;
* the FILTER is the real :func:`gpuwm.da.letkf.analyze`, because the open
  question the scout left — does a k = 0-only batch behave under the
  vertical stencil — is only answerable by running it.

Everything here is CPU/numpy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gpuwm.da.letkf import (GridGeometry, LetkfConfig, Localization,
                            analyze)
from gpuwm.da.obs_surface import (OBS_SCHEMA, SurfaceObsConfig,
                                  SurfaceObsError, surface_to_gridded_obs)
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

FIXTURES = Path(__file__).parent / "fixtures" / "asos_surface_real"
RECORD = FIXTURES / "surface_subset.v1.json"

#: The fixture's own valid times (UTC); the record is the authority, these
#: are just spellings for the tests.
T12 = datetime(2024, 5, 21, 12, 0, tzinfo=timezone.utc)
T13 = datetime(2024, 5, 21, 13, 0, tzinfo=timezone.utc)


def _record() -> dict:
    with open(RECORD, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _grid(nx: int = 31, ny: int = 31, dx: float = 3000.0, nz: int = 10,
          surface_m: float = 300.0, depth_m: float = 15000.0,
          terrain_m: np.ndarray | None = None) -> TargetGrid:
    """A real Lambert domain centred between the fixture's stations."""

    projection = LambertGrid(
        ref_lat=41.6, ref_lon=-93.6, truelat1=40.0, truelat2=43.0,
        stand_lon=-93.6, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    z_w = np.linspace(surface_m, surface_m + depth_m, nz + 1)
    return TargetGrid.from_projection(projection, z_w=z_w,
                                      terrain_m=terrain_m, name="sfc-test")


def _members(grid: TargetGrid, r: int = 4, t2_k: float = 290.0,
             u10_ms: float = 3.0, v10_ms: float = 4.0):
    """(R, ny, nx) diagnostics with a deterministic member spread."""

    shape = (r, grid.ny, grid.nx)
    spread = np.linspace(-1.0, 1.0, r)[:, None, None]
    t2 = np.full(shape, t2_k) + spread
    u10 = np.full(shape, u10_ms) + 0.5 * spread
    v10 = np.full(shape, v10_ms) - 0.25 * spread
    return t2, u10, v10


def _config(**overrides) -> SurfaceObsConfig:
    base = dict(temperature_error_k=2.0, wind_speed_error_ms=2.0)
    base.update(overrides)
    return SurfaceObsConfig(**base)


# ---------------------------------------------------------------------------
# the real seam, read whole
# ---------------------------------------------------------------------------


def test_fixture_is_the_real_seam():
    record = _record()
    assert record["schema"] == OBS_SCHEMA
    assert record["status"] == "READY"
    assert record["provenance"]["product"] == "iem-asos-metar"
    assert not record["provenance"]["is_stub"]
    # The writer's own completeness refusals are part of the artifact.
    assert record["screen"]["stations_dropped_by_completeness"]
    # SI units the ABI marker promises: K temperatures, m/s winds.
    # A quantity can be legitimately absent from a report; it is never
    # present in the wrong units.
    for report in record["reports"]:
        values = report["values"]
        if "temperature_2m" in values:
            assert 233.15 < values["temperature_2m"] < 328.15
        if "wind_speed_10m" in values:
            assert 0.0 <= values["wind_speed_10m"] < 75.0


def test_adapter_places_real_reports_on_a_real_grid():
    grid = _grid()
    record = _record()
    t2, u10, v10 = _members(grid)
    batches, provenance = surface_to_gridded_obs(
        record, target_grid=grid, analysis_time=T12, config=_config(),
        simulated_t2=t2, simulated_u10=u10, simulated_v10=v10)

    assert [b.name for b in batches] == ["temperature_2m:asos",
                                        "wind_speed_10m:asos"]
    t_batch, w_batch = batches
    shape = (grid.nz, grid.ny, grid.nx)
    assert np.shape(t_batch.mask) == shape
    # Surface observations live at k = 0 and nowhere else.
    assert not t_batch.mask[1:].any()
    assert not w_batch.mask[1:].any()
    observed = int(t_batch.mask.sum())
    assert observed > 0

    counts = provenance["counts"]
    # The 31x31 (+/-46 km) domain genuinely excludes some fixture
    # stations (the record spans ~130 km of Iowa); the refusals must be
    # counted, not vanished.  Compute the expectation from the record and
    # the grid — the artifacts are the authority, not a hardcoded number.
    expected_outside = 0
    expected_inside = set()
    stations = {s["station_id"]: s for s in record["stations"]}
    at_t12 = {r["station_id"] for r in record["reports"]
              if r["valid_time"] == "2024-05-21T12:00:00"}
    for sid in at_t12:
        s = stations[sid]
        i_f, j_f = grid.mass_index(s["latitude"], s["longitude"])
        i, j = int(np.rint(float(i_f))), int(np.rint(float(j_f)))
        if bool(grid.inside(i, j)):
            expected_inside.add((j, i))
        else:
            expected_outside += 1
    assert counts["stations_outside_domain"] == expected_outside
    assert expected_outside > 0
    assert observed == len(expected_inside)

    # Values under the mask are the record's own numbers, in SI.
    values = np.asarray(t_batch.values)
    assert np.isfinite(values[np.asarray(t_batch.mask)]).all()
    # Errors are the stated sigma, unsquared.
    errors = np.asarray(t_batch.errors)
    assert float(errors[np.asarray(t_batch.mask)][0]) == pytest.approx(2.0)
    # The wind H(x) is the modulus of the member 10 m wind.
    sim = np.asarray(w_batch.simulated)
    expected_speed = np.hypot(u10, v10)
    mask0 = np.asarray(w_batch.mask)[0]
    for member in range(sim.shape[0]):
        assert np.allclose(sim[member, 0][mask0], expected_speed[member][mask0])

    # Per-station QC outcomes name every decision.
    qc = provenance["station_qc"]
    assert all(entry["outcome"] in ("accepted", "refused")
               for entry in qc.values())
    assert provenance["station_table_sha256"] == \
        record["station_table_sha256"]


def test_decode_through_the_real_writer_roundtrips():
    """Re-run the REAL rw_asos decode over the committed real CSV.

    Skips unless GPUWM_RW_ASOS names the binary (the vendored rustwx
    workspace lives on the obs-battery lane; this lane consumes its seam).
    """

    exe = os.environ.get("GPUWM_RW_ASOS")
    if not exe or not Path(exe).is_file():
        pytest.skip("GPUWM_RW_ASOS does not name an rw_asos binary")
    out = FIXTURES / "roundtrip.tmp.json"
    try:
        subprocess.run(
            [exe, "decode", "--stations", str(FIXTURES / "stations.json"),
             "--obs", str(FIXTURES / "observations_subset.csv"),
             "--start", "2024-05-21T12:00:00Z",
             "--end", "2024-05-21T14:00:00Z", "--step-hours", "1",
             "--out", str(out)],
            check=True, capture_output=True, timeout=120)
        with open(out, "r", encoding="utf-8") as handle:
            fresh = json.load(handle)
        committed = _record()
        # The seam fields must agree exactly; provenance differs only in
        # fetched_at/uri (this run's clock and path).
        for key in ("schema", "status", "station_table_sha256",
                    "valid_times", "match_seconds", "stations", "reports",
                    "screen"):
            assert fresh[key] == committed[key], key
    finally:
        if out.exists():
            out.unlink()


# ---------------------------------------------------------------------------
# the elevation gate
# ---------------------------------------------------------------------------


def test_elevation_gate_refuses_and_counts():
    record = _record()
    grid_flat = _grid()
    # Expectation from the artifacts: gate 20 m against 300 m flat
    # terrain refuses exactly the placeable stations whose table
    # elevation is outside [280, 320].
    stations = {s["station_id"]: s for s in record["stations"]}
    at_t12 = {r["station_id"] for r in record["reports"]
              if r["valid_time"] == "2024-05-21T12:00:00"}
    placeable, should_refuse = set(), set()
    for sid in at_t12:
        s = stations[sid]
        i_f, j_f = grid_flat.mass_index(s["latitude"], s["longitude"])
        i, j = int(np.rint(float(i_f))), int(np.rint(float(j_f)))
        if not bool(grid_flat.inside(i, j)):
            continue
        placeable.add(sid)
        if abs(float(s["elevation_m"]) - 300.0) > 20.0:
            should_refuse.add(sid)
    assert should_refuse, "fixture no longer exercises the gate"

    t2, u10, v10 = _members(grid_flat)
    _, provenance = surface_to_gridded_obs(
        record, target_grid=grid_flat, analysis_time=T12,
        config=_config(elevation_max_diff_m=20.0),
        simulated_t2=t2, simulated_u10=u10, simulated_v10=v10)
    counts = provenance["counts"]
    assert counts["stations_refused_elevation"] == len(should_refuse)
    for sid in should_refuse:
        entry = provenance["station_qc"][sid]
        assert entry["outcome"] == "refused"
        assert entry["reason"] == "elevation gate"
        assert abs(entry["difference_m"]) > 20.0
        assert entry["gate_m"] == 20.0
    # The default 200 m gate accepts all of them (Iowa is flat).
    _, wide = surface_to_gridded_obs(
        record, target_grid=grid_flat, analysis_time=T12,
        config=_config(),
        simulated_t2=t2, simulated_u10=u10, simulated_v10=v10)
    assert wide["counts"]["stations_refused_elevation"] == 0


# ---------------------------------------------------------------------------
# the ob-age window and once-per-report routing
# ---------------------------------------------------------------------------


def test_reports_outside_the_age_window_are_refused_and_counted():
    grid = _grid()
    t2, u10, v10 = _members(grid)
    # 12:30 sits 1800 s from every hourly valid time; a 900 s window
    # leaves nothing, and the adapter must say so rather than guess.
    batches, provenance = surface_to_gridded_obs(
        _record(), target_grid=grid,
        analysis_time=T12 + timedelta(minutes=30), config=_config(),
        simulated_t2=t2, simulated_u10=u10, simulated_v10=v10)
    assert provenance["counts"]["reports_outside_window"] == \
        provenance["counts"]["reports_in_record"]
    for batch in batches:
        assert not np.asarray(batch.mask).any()


def test_each_report_enters_exactly_one_analysis():
    grid = _grid()
    t2, u10, v10 = _members(grid)
    # A 15 min cycle across the top of the hour: 12:45, 13:00, 13:15.
    # With a 1200 s window the 13:00 reports are eligible at all three
    # (900 s away from the outer two); the schedule must route each to
    # 13:00 alone, and the outer analyses must count the routing rather
    # than silently double-assimilating the same numbers.
    schedule = [T13 - timedelta(minutes=15), T13,
                T13 + timedelta(minutes=15)]
    config = _config(max_age_seconds=1200.0)
    seen = {}
    for analysis_time in schedule:
        batches, provenance = surface_to_gridded_obs(
            _record(), target_grid=grid, analysis_time=analysis_time,
            analysis_times=schedule, config=config,
            simulated_t2=t2, simulated_u10=u10, simulated_v10=v10)
        seen[analysis_time] = int(np.asarray(batches[0].mask).sum())
        if analysis_time != T13:
            assert provenance["counts"][
                "reports_routed_to_other_analysis"] > 0
    assert seen[T13] > 0
    assert seen[schedule[0]] == 0
    assert seen[schedule[2]] == 0


# ---------------------------------------------------------------------------
# refusals: fail closed, every one named
# ---------------------------------------------------------------------------


def test_config_refuses_nothing_enabled():
    with pytest.raises(SurfaceObsError, match="assimilate nothing"):
        SurfaceObsConfig()


def test_config_refuses_bad_sigma_and_deflation():
    with pytest.raises(SurfaceObsError, match="standard deviation"):
        SurfaceObsConfig(temperature_error_k=0.0)
    with pytest.raises(SurfaceObsError, match="claim of skill"):
        _config(error_inflation=0.5)


def test_wrong_schema_is_refused():
    record = dict(_record())
    record["schema"] = "gpuwm-obs.radar-grid.v1"
    grid = _grid()
    t2, u10, v10 = _members(grid)
    with pytest.raises(SurfaceObsError, match="adapter reads"):
        surface_to_gridded_obs(record, target_grid=grid,
                               analysis_time=T12, config=_config(),
                               simulated_t2=t2, simulated_u10=u10,
                               simulated_v10=v10)


def test_stub_provenance_is_refused():
    record = json.loads(json.dumps(_record()))
    record["provenance"]["is_stub"] = True
    record["provenance"]["stub_reason"] = "wiring test"
    grid = _grid()
    t2, u10, v10 = _members(grid)
    with pytest.raises(SurfaceObsError, match="is_stub"):
        surface_to_gridded_obs(record, target_grid=grid,
                               analysis_time=T12, config=_config(),
                               simulated_t2=t2, simulated_u10=u10,
                               simulated_v10=v10)


def test_flagged_reports_are_refused_and_counted():
    record = json.loads(json.dumps(_record()))
    flagged = 0
    for report in record["reports"]:
        if report["valid_time"] == "2024-05-21T12:00:00":
            report["flags"] = ["suspect"]
            flagged += 1
    grid = _grid()
    t2, u10, v10 = _members(grid)
    batches, provenance = surface_to_gridded_obs(
        record, target_grid=grid, analysis_time=T12, config=_config(),
        simulated_t2=t2, simulated_u10=u10, simulated_v10=v10)
    assert provenance["counts"]["reports_refused_flagged"] == flagged
    for batch in batches:
        assert not np.asarray(batch.mask).any()


def test_missing_simulated_is_refused():
    grid = _grid()
    t2, u10, v10 = _members(grid)
    with pytest.raises(SurfaceObsError, match="simulated_t2"):
        surface_to_gridded_obs(_record(), target_grid=grid,
                               analysis_time=T12, config=_config(),
                               simulated_u10=u10, simulated_v10=v10)
    with pytest.raises(SurfaceObsError, match="hypot"):
        surface_to_gridded_obs(_record(), target_grid=grid,
                               analysis_time=T12, config=_config(),
                               simulated_t2=t2, simulated_u10=u10)


def test_missing_values_are_counted_never_defaulted():
    record = json.loads(json.dumps(_record()))
    removed = 0
    for report in record["reports"]:
        if report["valid_time"] == "2024-05-21T12:00:00":
            if "wind_speed_10m" in report["values"]:
                del report["values"]["wind_speed_10m"]
                removed += 1
    assert removed
    grid = _grid()
    t2, u10, v10 = _members(grid)
    batches, provenance = surface_to_gridded_obs(
        record, target_grid=grid, analysis_time=T12, config=_config(),
        simulated_t2=t2, simulated_u10=u10, simulated_v10=v10)
    t_batch, w_batch = batches
    assert not np.asarray(w_batch.mask).any()
    assert np.asarray(t_batch.mask).any()
    missing = provenance["counts"]["values_missing_by_quantity"]
    assert missing["wind_speed_10m"] == provenance["stations_placed"]


# ---------------------------------------------------------------------------
# through the real filter: the k = 0 batch the scout flagged
# ---------------------------------------------------------------------------


def _letkf_geometry(grid: TargetGrid) -> GridGeometry:
    z_w = np.asarray(grid.z_w)
    midpoints = 0.5 * (z_w[:-1] + z_w[1:])
    return GridGeometry(dx_m=grid.dx_m, dy_m=grid.dy_m,
                        heights_m=np.ascontiguousarray(midpoints),
                        lat_deg=np.asarray(grid.lat),
                        lon_deg=np.asarray(grid.lon))


def test_k0_wind_speed_batch_through_analyze():
    """A surface wind-speed observation moves the lowest-level wind toward
    the observed speed, stays inside its localisation lens, and leaves
    levels above the vertical cutoff bitwise untouched."""

    grid = _grid(nz=10, depth_m=15000.0)   # levels every 1500 m
    r = 6
    rng = np.random.default_rng(20260805)
    # Member 10 m winds: eastward, spread ~1 m/s, mean 4 m/s; the state's
    # lowest level IS the diagnostic (linear H by construction), upper
    # levels correlated so the vertical localisation has something to cut.
    u10 = 4.0 + rng.normal(0.0, 1.0, (r, grid.ny, grid.nx))
    v10 = np.zeros_like(u10)
    prior_u = np.repeat(u10[:, None, :, :], grid.nz, axis=1)

    record = json.loads(json.dumps(_record()))
    # Observed speed well above the ensemble mean, at every station.
    for report in record["reports"]:
        report["values"]["wind_speed_10m"] = 8.0

    batches, _ = surface_to_gridded_obs(
        record, target_grid=grid, analysis_time=T12,
        config=SurfaceObsConfig(wind_speed_error_ms=1.0),
        simulated_u10=u10, simulated_v10=v10)
    (w_batch,) = batches
    mask0 = np.asarray(w_batch.mask)[0]
    assert mask0.any()

    loc = Localization(horizontal_m=12000.0, vertical_m=3000.0)
    config = LetkfConfig(localization=loc, analysis_fields=("u",),
                         rtps_alpha=0.0, chunk_points=512)
    increments = analyze({"u": prior_u}, [w_batch],
                         _letkf_geometry(grid), config)
    du = np.asarray(increments["u"])
    assert np.isfinite(du).all()

    # At an observed column the posterior mean speed moves toward 8 m/s.
    j, i = [tuple(x) for x in np.argwhere(mask0)][0]
    before = float(np.mean(u10[:, j, i]))
    after = float(np.mean(u10[:, j, i] + du[:, 0, j, i]))
    assert after > before
    assert abs(8.0 - after) < abs(8.0 - before)

    # Vertical localisation from k = 0: with 1500 m level spacing and a
    # 3000 m full-support cutoff, level 3 (4500 m above the observed
    # midpoint) is strictly outside the lens -- BITWISE zero, the
    # filter's own inactive-point guarantee.  Level 2 sits exactly at
    # the cutoff, a zero-weight boundary this test deliberately does not
    # claim either way.
    assert np.any(du[:, 0] != 0.0)
    assert np.all(du[:, 3:] == 0.0)

    # Horizontal localisation: columns strictly outside 12 km of every
    # observation (one full cell of margin past the cutoff) are bitwise
    # zero too.
    obs_j, obs_i = np.nonzero(mask0)
    jj, ii = np.mgrid[0:grid.ny, 0:grid.nx]
    dist2 = np.min(
        (jj[..., None] - obs_j) ** 2 + (ii[..., None] - obs_i) ** 2,
        axis=-1)
    far = dist2 > (loc.horizontal_m / grid.dx_m + 1.0) ** 2
    assert far.any()
    assert np.all(du[:, :, far] == 0.0)


def test_k0_temperature_batch_through_analyze():
    grid = _grid(nz=6, depth_m=12000.0)
    r = 4
    rng = np.random.default_rng(7)
    t2 = 290.0 + rng.normal(0.0, 0.8, (r, grid.ny, grid.nx))
    prior_t = np.repeat(t2[:, None, :, :], grid.nz, axis=1)

    record = json.loads(json.dumps(_record()))
    for report in record["reports"]:
        report["values"]["temperature_2m"] = 293.0

    batches, _ = surface_to_gridded_obs(
        record, target_grid=grid, analysis_time=T12,
        config=SurfaceObsConfig(temperature_error_k=1.0),
        simulated_t2=t2)
    (t_batch,) = batches
    mask0 = np.asarray(t_batch.mask)[0]
    config = LetkfConfig(
        localization=Localization(horizontal_m=12000.0, vertical_m=3000.0),
        analysis_fields=("t",), rtps_alpha=0.0, chunk_points=512)
    increments = analyze({"t": prior_t}, [t_batch],
                         _letkf_geometry(grid), config)
    dt = np.asarray(increments["t"])
    assert np.isfinite(dt).all()
    j, i = [tuple(x) for x in np.argwhere(mask0)][0]
    before = float(np.mean(t2[:, j, i]))
    after = before + float(np.mean(dt[:, 0, j, i]))
    assert abs(293.0 - after) < abs(293.0 - before)
