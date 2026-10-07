"""Native Q2 inversion and actual surface adapter, CPU only."""
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gpuwm.da.obs_surface import SurfaceObsConfig, SurfaceObsError, surface_to_gridded_obs
from gpuwm.da.letkf import GridGeometry, LetkfConfig, Localization, analyze
from gpuwm.da.surface_dewpoint import dewpoint_to_q2
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

RECORD = Path(__file__).parent / "fixtures/asos_surface_real/surface_subset.v2.json"
T12 = datetime(2024, 5, 21, 12, 0, tzinfo=timezone.utc)


def _grid():
    projection = LambertGrid(ref_lat=41.6, ref_lon=-93.6, truelat1=40.0,
                             truelat2=43.0, stand_lon=-93.6, dx=3000.0,
                             dy=3000.0, e_we=32, e_sn=32)
    return TargetGrid.from_projection(projection, z_w=np.linspace(300, 15300, 11),
                                      name="surface-native-test")


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _fields(grid):
    shape = (4, grid.ny, grid.nx)
    spread = np.arange(4)[:, None, None]
    return (np.full(shape, 290.0) + spread,
            np.full(shape, 3.0) + spread * .1,
            np.full(shape, 4.0) - spread * .1,
            np.full(shape, .006) + spread * .001,
            np.full(shape, 90_000.0) + spread * 2000.0)


def _adapt(record=None, config=None, analysis_time=T12, analysis_times=None):
    grid = _grid()
    t2, u, v, q2, pressure = _fields(grid)
    return surface_to_gridded_obs(
        record or _record(), target_grid=grid, analysis_time=analysis_time,
        analysis_times=analysis_times,
        config=config or SurfaceObsConfig(dewpoint_error_k=2.0),
        simulated_t2=t2, simulated_u10=u, simulated_v10=v,
        simulated_q2=q2, simulated_psfc=pressure)


def test_native_pin_is_mixing_ratio_and_sigma_matches_centered_difference():
    q, sigma, p = dewpoint_to_q2([273.15, 293.15], [[90_000, 95_000], [100_000, 95_000]], 2.0)
    assert p.tolist() == [95_000.0, 95_000.0]
    assert q[0] == .622 * 611.2 / (95_000 - 611.2)
    assert q[0] != pytest.approx(.622 * 611.2 / (95_000 - .378 * 611.2), rel=1e-5)
    for index, td in enumerate([273.15, 293.15]):
        step = .001
        plus = dewpoint_to_q2([td + step], [[95_000.0]], 1.0)[0][0]
        minus = dewpoint_to_q2([td - step], [[95_000.0]], 1.0)[0][0]
        assert sigma[index] == pytest.approx((plus - minus) / (2 * step) * 2.0, rel=1e-8)
    _, doubled, _ = dewpoint_to_q2([273.15, 293.15], [[95_000, 95_000]], 2.0, 2.0)
    np.testing.assert_array_equal(doubled, sigma * 2)


@pytest.mark.parametrize("td,pressure,sigma,match", [
    (20.0, 100_000, 2.0, "Td in K"),
    (293.15, 1000, 2.0, "PSFC in Pa"),
    (350.0, 20_000, 2.0, "vapor pressure"),
    (293.15, 100_000, 0.0, "sigma"),
])
def test_native_units_and_domain_have_no_implicit_conversion(td, pressure, sigma, match):
    with pytest.raises(ValueError, match=match):
        dewpoint_to_q2([td], [[pressure]], sigma)


def test_optional_operator_preserves_temperature_and_wind_arrays():
    baseline, _ = _adapt(config=SurfaceObsConfig(temperature_error_k=2.0, wind_speed_error_ms=2.0))
    combined, prov = _adapt(config=SurfaceObsConfig(temperature_error_k=2.0,
                                                   wind_speed_error_ms=2.0,
                                                   dewpoint_error_k=2.0))
    assert len(combined) == 3
    for left, right in zip(baseline, combined[:2]):
        for attr in ("values", "errors", "mask", "simulated"):
            a, b = np.asarray(getattr(left, attr)), np.asarray(getattr(right, attr))
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.tobytes() == b.tobytes()
    dewpoint = combined[2]
    assert dewpoint.name == "dewpoint_2m:asos"
    assert dewpoint.mask.sum() > 0 and not dewpoint.mask[1:].any()
    _, _, _, q2, _ = _fields(_grid())
    np.testing.assert_array_equal(dewpoint.simulated[:, 0], q2)
    meta = prov["batches"][2]
    assert meta["units"] == "kg kg-1 mixing ratio"
    assert meta["input_units"] == "K" and meta["error_stddev"] is None
    assert meta["input_error_stddev_k"] == 2.0
    assert "not propagated" in meta["pressure_error"]
    for station in prov["station_qc"].values():
        if "dewpoint_pressure_proxy_pa" in station:
            assert station["dewpoint_pressure_proxy_pa"] == 93_000.0
    assert np.isfinite(dewpoint.errors[dewpoint.mask]).all()


def test_dewpoint_observation_uses_real_seam_value_and_declared_pressure():
    batches, prov = _adapt()
    record = _record()
    seen = 0
    for station_id, qc in prov["station_qc"].items():
        if "dewpoint_pressure_proxy_pa" not in qc:
            continue
        reports = [report for report in record["reports"]
                   if report["station_id"] == station_id
                   and report["valid_time"] == "2024-05-21T12:00:00"]
        td = reports[0]["values"]["dewpoint_2m"]
        e = 611.2 * math.exp(17.67 * (td - 273.15) / (td - 273.15 + 243.5))
        expected = .622 * e / (93_000 - e)
        assert batches[0].values[0, qc["j"], qc["i"]] == pytest.approx(expected, rel=1e-14)
        seen += 1
    assert seen == int(batches[0].mask.sum()) > 0


def test_existing_elevation_colocation_and_once_per_report_gates_still_apply():
    record = _record()
    _, baseline = _adapt()
    original = next(sid for sid, qc in baseline["station_qc"].items()
                    if "dewpoint_pressure_proxy_pa" in qc)
    station = dict(next(s for s in record["stations"] if s["station_id"] == original))
    station["station_id"] = original + "_duplicate"
    record["stations"].append(station)
    for report in list(record["reports"]):
        if report["station_id"] == original:
            duplicate = dict(report, station_id=station["station_id"])
            record["reports"].append(duplicate)
    _, prov = _adapt(record=record)
    assert prov["counts"]["stations_superseded_colocated"] >= 1
    schedule = [T12 - timedelta(minutes=10), T12, T12 + timedelta(minutes=10)]
    entered = []
    for analysis in schedule:
        _, receipt = _adapt(analysis_time=analysis, analysis_times=schedule)
        for sid, qc in receipt["station_qc"].items():
            if "dewpoint_pressure_proxy_pa" in qc:
                entered.append((sid, (analysis - timedelta(seconds=qc["age_s"])).isoformat()))
    assert entered and len(entered) == len(set(entered))
    elevated = _record()
    for row in elevated["stations"]:
        row["elevation_m"] += 10_000
    batches, receipt = _adapt(record=elevated)
    assert not batches[0].mask.any()
    assert receipt["counts"]["stations_refused_elevation"] > 0


def test_missing_pressure_and_member_mismatch_are_refused():
    grid = _grid()
    _, _, _, q2, pressure = _fields(grid)
    config = SurfaceObsConfig(dewpoint_error_k=2)
    with pytest.raises(SurfaceObsError, match="simulated_q2/simulated_psfc"):
        surface_to_gridded_obs(_record(), target_grid=grid, analysis_time=T12,
                              config=config, simulated_q2=q2)
    with pytest.raises(SurfaceObsError, match="members"):
        surface_to_gridded_obs(_record(), target_grid=grid, analysis_time=T12,
                              config=config, simulated_q2=q2, simulated_psfc=pressure[:3])


def test_q2_batch_through_actual_letkf_moves_vapor_toward_observation():
    grid = _grid()
    record = _record()
    for report in record["reports"]:
        report["values"]["dewpoint_2m"] = 293.15
    batches, _ = _adapt(record=record)
    _, _, _, q2, _ = _fields(grid)
    prior = np.repeat(q2[:, None], grid.nz, axis=1)
    z_w = np.asarray(grid.z_w)
    geometry = GridGeometry(dx_m=grid.dx_m, dy_m=grid.dy_m,
                            heights_m=np.ascontiguousarray(.5 * (z_w[:-1] + z_w[1:])),
                            lat_deg=np.asarray(grid.lat), lon_deg=np.asarray(grid.lon))
    config = LetkfConfig(localization=Localization(horizontal_m=12000, vertical_m=3000),
                         analysis_fields=("qv",), rtps_alpha=0, chunk_points=512)
    increments = analyze({"qv": prior}, batches, geometry, config)
    dq = np.asarray(increments["qv"])
    assert np.isfinite(dq).all()
    j, i = np.argwhere(batches[0].mask[0])[0]
    observed = batches[0].values[0, j, i]
    before = float(q2[:, j, i].mean())
    after = before + float(dq[:, 0, j, i].mean())
    assert abs(observed - after) < abs(observed - before)
    assert np.all(dq[:, 3:] == 0)
