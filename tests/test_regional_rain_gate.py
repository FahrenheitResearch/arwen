"""Independent physical checks for the regional-rain/v1 scoring contract.

The fixtures are analytic fields, not reference calls to the implementation.
No GPU, network, or weather simulation is used.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify import rain_gate as gate


def _grid(x, y):
    return gate.RectGrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float), "analytic-equal-area")


def _metric(rows, name):
    return next(row for row in rows if row["metric"] == name)


def test_conservative_shifted_grid_preserves_analytic_volume():
    source = _grid([0, 1000, 2000], [0, 1000, 2000])
    target = _grid([-500, 500, 1500, 2500], [-500, 500, 1500, 2500])
    values = np.array([[1.0, 3.0], [5.0, 7.0]])
    remapped, valid, receipt = gate.conservative_remap(values, np.ones((2, 2), bool), source, target)
    # The central target square contains one quarter of each source cell.
    assert remapped[1, 1] == pytest.approx(4.0)
    assert valid[1, 1]
    # Outlying target squares lack source support. They cannot become dry.
    assert not valid[0, 0]
    assert receipt["volume_error_fraction"] <= 0.001


def test_conservative_different_cell_sizes_match_exact_integral():
    source = _grid([0, 1000, 2000, 3000], [0, 1500, 3000])
    target = _grid([0, 1500, 3000], [0, 1000, 2000, 3000])
    values = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    remapped, valid, receipt = gate.conservative_remap(values, np.ones(values.shape, bool), source, target)
    assert np.all(valid)
    # 2*1.0 + 4*0.5 in the lower-left 1.5 km wide target cell.
    assert remapped[0, 0] == pytest.approx(8.0 / 3.0)
    source_volume = 42.0 * 1_500_000.0
    target_volume = float(np.sum(remapped)) * 1_500_000.0
    assert target_volume == pytest.approx(source_volume, rel=0.001)
    assert receipt["volume_error_fraction"] <= 0.001


def test_conservative_invalid_source_is_missing_not_zero():
    source = _grid([0, 1000, 2000], [0, 1000])
    target = _grid([0, 2000], [0, 1000])
    _, valid, _ = gate.conservative_remap(np.array([[2.0, 200.0]]), np.array([[True, False]]), source, target)
    assert not valid[0, 0]


@pytest.mark.parametrize("lead,start_value,end_value", [(1, 0, 2), (3, 9, 20), (6, 41, 60)])
def test_hourly_window_is_exact_last_hour(lead, start_value, end_value):
    times = np.arange(0, 21601, 120, dtype=float)
    hourly = np.array([0, 2, 9, 20, 29, 41, 60], dtype=float)
    totals = np.interp(times, np.arange(7) * 3600, hourly)[:, None, None]
    rain, valid, _ = gate.hourly_accumulation(times, totals, np.zeros(len(times), dtype=int), np.zeros_like(totals), lead)
    assert np.all(valid)
    assert rain[0, 0] == end_value - start_value


def test_explicit_reset_adds_water_before_and_after_reset():
    times = np.arange(0, 3601, 120, dtype=float)
    totals = np.where(times < 1800, 10.0 + 4.0 * times / 1800, 2.0 + 3.0 * (times - 1800) / 1800)[:, None, None]
    # 14 mm existed just before the reset at the middle frame. A further
    # 2 mm fell after it, followed by another 3 mm before the endpoint.
    carry = np.zeros_like(totals)
    carry[times == 1800] = 14.0
    rain, valid, _ = gate.hourly_accumulation(times, totals, (times >= 1800).astype(int), carry, 1)
    assert np.all(valid)
    assert rain[0, 0] == pytest.approx(9.0)


def test_unexplained_negative_accumulation_never_clips_to_zero():
    totals = np.array([3.0, 1.0])[:, None, None]
    rain, valid, receipt = gate.hourly_accumulation([0, 3600], totals, [0, 0], np.zeros_like(totals), 1)
    assert not valid[0, 0]
    assert receipt.get("reason")


def test_missing_hourly_endpoint_is_pending():
    totals = np.array([0.0, 2.0])[:, None, None]
    _, valid, receipt = gate.hourly_accumulation([0, 3500], totals, [0, 0], np.zeros_like(totals), 1)
    assert not np.any(valid)
    assert receipt.get("reason")


def _single_event_fss(width_km, event_column):
    # A 31-cell-wide complete grid leaves room for the full 50 km square.
    observed = np.zeros((41, 41), dtype=float)
    forecast = np.zeros_like(observed)
    observed[20, 20] = 5.0
    forecast[20, event_column] = 5.0
    return gate.fss_exact(forecast, observed, np.ones_like(observed, bool), 1.0, width_km, dx_m=3000.0)


@pytest.mark.parametrize("width_km,edge_weight,half_cells", [(10.0, 0.5, 2), (25.0, 2.0, 4), (50.0, 2.5, 8)])
def test_exact_physical_fss_width_has_fractional_edge_cells(width_km, edge_weight, half_cells):
    # Across an event displaced one 3 km cell, FSS equals the normalized
    # autocorrelation of a 1-D overlap kernel. All y factors cancel.
    # The physical square covers full 3 km cells and two fractional ends.
    weights = np.full(2 * half_cells + 1, 3.0)
    weights[0] = weights[-1] = edge_weight
    expected = float(np.dot(weights[:-1], weights[1:]) / np.dot(weights, weights))
    result = _single_event_fss(width_km, 21)
    assert result["value"] == pytest.approx(expected, abs=1e-12)


def test_fss_drops_every_neighborhood_touching_invalid_cell():
    field = np.ones((11, 11), dtype=float) * 2.0
    valid = np.ones_like(field, bool)
    valid[5, 5] = False
    result = gate.fss_exact(field, field, valid, 1.0, 10.0, dx_m=3000.0)
    # Width 10 km spans five cells, including fractional outer cells.
    # 7*7 complete interior neighborhoods minus 5*5 touching the hole.
    assert result["scored_cells"] == 24
    assert result["value"] == pytest.approx(1.0)


def test_dry_fss_has_no_wet_skill_evidence():
    field = np.zeros((21, 21), dtype=float)
    result = gate.fss_exact(field, field, np.ones_like(field, bool), 1.0, 10.0, dx_m=3000.0)
    assert result["value"] is None
    assert result.get("reason")


def test_threshold_dry_observation_remains_pending_with_false_rain():
    observed = np.zeros((21, 21), dtype=float)
    forecast = np.ones_like(observed) * 2.0
    result = gate.fss_exact(forecast, observed, np.ones_like(observed, bool), 1.0, 10.0, dx_m=3000.0)
    assert result["value"] is None
    assert result.get("reason")


def test_observed_footprint_not_trimmed_to_model_coverage():
    grid = _grid(np.arange(22) * 3000, np.arange(22) * 3000)
    observed = np.zeros((21, 21), dtype=float)
    observed[10, 9:12] = 2.0
    model = observed.copy()
    common = np.ones_like(observed, bool)
    model_valid = common.copy()
    model_valid[10, 11] = False
    echo = np.zeros_like(observed)
    rows = gate.score_hour(model, observed, model_valid, common, echo, echo, common, footprint=observed > 0, grid=grid)
    row = _metric(rows, "footprint_rain_ratio")
    assert row["value"] is None
    assert row["status"] == "pending"


def test_observed_object_crossing_grid_edge_is_incomplete():
    grid = _grid(np.arange(22) * 3000, np.arange(22) * 3000)
    observed = np.zeros((21, 21), dtype=float)
    observed[0:3, 10] = 2.0
    valid = np.ones_like(observed, bool)
    echo = np.zeros_like(observed)
    rows = gate.score_hour(observed, observed, valid, valid, echo, echo, valid, footprint=observed > 0, grid=grid)
    row = _metric(rows, "footprint_rain_ratio")
    assert row["value"] is None
    assert row["status"] == "pending"


def test_echo_ratio_uses_hourly_time_fraction_not_endpoint_area():
    grid = _grid(np.arange(22) * 3000, np.arange(22) * 3000)
    rain = np.zeros((21, 21), dtype=float)
    rain[10, 10] = 2.0
    valid = np.ones_like(rain, bool)
    observed_echo = np.zeros_like(rain)
    model_echo = np.zeros_like(rain)
    observed_echo[10, 10] = 1.0
    model_echo[10, 10] = 0.5
    rows = gate.score_hour(rain, rain, valid, valid, model_echo, observed_echo, valid, grid=grid)
    row = _metric(rows, "echo_area_multiple")
    assert row["value"] == pytest.approx(0.5)


def test_echo_integration_requires_every_two_minute_interval():
    times = np.arange(0, 3601, 120, dtype=float)
    dbz = np.full((len(times), 2, 2), 40.0)
    valid = np.ones_like(dbz, dtype=bool)
    fraction, support, receipt, footprint = gate.integrate_hour(times, dbz, valid, 1, mode="echo")
    assert np.all(support)
    assert np.allclose(fraction, 1.0)
    missing = np.arange(len(times)) != 13
    _, support, receipt, _ = gate.integrate_hour(times[missing], dbz[missing], valid[missing], 1, mode="echo")
    assert not np.any(support)
    assert receipt.get("reason")


def test_rate_integration_has_mm_hour_units_and_exact_window():
    times = np.arange(0, 21601, 120, dtype=float)
    # First hour is deliberately different. A six-hour accumulation or
    # mistakenly scored [0, L] window would return the wrong answer.
    rate = np.full((len(times), 1, 1), 4.0)
    rate[times < 3600] = 20.0
    rain, valid, _, _ = gate.integrate_hour(times, rate, np.ones_like(rate, bool), 6, mode="rate")
    assert valid[0, 0]
    assert rain[0, 0] == pytest.approx(4.0)


def test_echo_integral_distinguishes_duration_from_snapshot():
    times = np.arange(0, 3601, 120, dtype=float)
    dbz = np.full((len(times), 1, 1), 0.0)
    dbz[times < 1800] = 40.0
    fraction, valid, _, _ = gate.integrate_hour(times, dbz, np.ones_like(dbz, bool), 1, mode="echo")
    assert valid[0, 0]
    # Trapezoid integration on the exact timestamps. There are fourteen
    # fully echoed two-minute intervals and one half echoed interval.
    assert fraction[0, 0] == pytest.approx(29.0 / 60.0)


def test_archive_rate_clocks_bracket_exact_hour_without_time_shift():
    times = np.arange(-42, 3679, 120, dtype=float)
    rate = np.full((len(times), 1, 1), 4.0)
    rain, valid, receipt, _ = gate.integrate_hour(times, rate, np.ones_like(rate, bool), 1, mode="rate")
    assert valid[0, 0]
    assert rain[0, 0] == pytest.approx(4.0)
    assert receipt["window_seconds"] == [0.0, 3600.0]


def test_partial_footprint_cell_keeps_fractional_area_and_native_truth_volume():
    grid = _grid(np.arange(22) * 3000, np.arange(22) * 3000)
    observed = np.zeros((21, 21), dtype=float)
    model = np.zeros_like(observed)
    observed[10, 10], model[10, 10] = 4.0, 8.0
    fraction = np.zeros_like(observed)
    fraction[10, 10] = 0.25
    masked_native_truth = np.zeros_like(observed)
    masked_native_truth[10, 10] = 1.0
    valid = np.ones_like(observed, bool)
    rows = gate.score_hour(model, observed, valid, valid, fraction, fraction, valid, footprint=fraction,
                           footprint_observed_rain=masked_native_truth, grid=grid)
    row = _metric(rows, "footprint_rain_ratio")
    assert row["support_area_m2"] == pytest.approx(2_250_000.0)
    assert row["numerator"] == pytest.approx(18_000_000.0)
    assert row["denominator"] == pytest.approx(9_000_000.0)
    assert row["value"] == pytest.approx(2.0)


def test_series_masks_native_footprint_rain_before_coarse_remap():
    times = np.arange(0, 3601, 120, dtype=float)
    truth_grid = _grid(np.arange(10) * 1000, np.arange(10) * 1000)
    model_grid = _grid(np.arange(4) * 3000, np.arange(4) * 3000)
    observed = np.zeros((len(times), 9, 9))
    observed[:, 4, 4] = 4.0
    echo = np.zeros_like(observed)
    echo[:, 4, 4] = 40.0
    truth = gate.Series(grid=truth_grid, times_seconds=times, rate_mm_h=observed,
                        rain_valid=np.ones_like(observed, bool), echo_dbz=echo, echo_valid=np.ones_like(observed, bool))
    forecast_rain = np.full((len(times), 3, 3), 8.0)
    forecast = gate.Series(grid=model_grid, times_seconds=times, rate_mm_h=forecast_rain,
                           rain_valid=np.ones_like(forecast_rain, bool), echo_dbz=np.zeros_like(forecast_rain),
                           echo_valid=np.ones_like(forecast_rain, bool))
    rows, _ = gate.score_series(forecast, truth, leads_hours=(1,))
    row = _metric(rows, "footprint_rain_ratio")
    assert row["support_area_m2"] == pytest.approx(1_000_000.0)
    assert row["numerator"] == pytest.approx(8_000_000.0)
    assert row["denominator"] == pytest.approx(4_000_000.0)
    assert row["value"] == pytest.approx(2.0)


def _saved_manifest(tmp_path, **changes):
    import json

    definition = {
        "schema": "regional-rain/input.v1",
        "grid": {"x_edges_m": [0, 3000, 6000], "y_edges_m": [0, 3000, 6000],
                 "projection_id": "analytic-equal-area"},
        "times_seconds": [0, 3600],
        "rain_accum_mm": [[[0, 0], [0, 0]], [[1, 2], [3, 4]]],
        "rain_valid": [[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
        "reset_ids": [0, 0],
        "is_stub": True,
    }
    definition.update(changes)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    return path


def test_manifest_loader_pins_quality_clock_and_stub_status(tmp_path):
    from gpuwm.verify.rain_gate_readers import load_series

    series, provenance = load_series(_saved_manifest(tmp_path))
    assert np.array_equal(series.times_seconds, [0, 3600])
    assert series.rain_valid.dtype == np.bool_
    assert series.rain_accum_mm[1, 1, 1] == 4.0
    assert provenance["is_stub"] is True
    assert len(provenance["manifest_sha256"]) == 64


def test_manifest_loader_refuses_missing_quality(tmp_path):
    import json
    from gpuwm.verify.rain_gate_readers import load_series

    path = _saved_manifest(tmp_path)
    definition = json.loads(path.read_text())
    definition.pop("rain_valid")
    path.write_text(json.dumps(definition))
    with pytest.raises(ValueError, match="explicit rain_valid"):
        load_series(path)


def test_manifest_loader_refuses_unpinned_reset_water(tmp_path):
    from gpuwm.verify.rain_gate_readers import load_series

    with pytest.raises(ValueError, match="reset_carry_mm"):
        load_series(_saved_manifest(tmp_path, reset_ids=[0, 1]))


def test_manifest_loader_saved_array_hash_is_rechecked(tmp_path):
    from gpuwm.verify.rain_gate_readers import load_series

    saved = tmp_path / "rain.npz"
    np.savez(saved, totals=np.zeros((2, 2, 2)))
    path = _saved_manifest(tmp_path, rain_accum_mm={"path": "rain.npz", "variable": "totals", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_series(path)


def test_headerless_saved_rain_cannot_override_canonical_units(tmp_path):
    from gpuwm.verify.rain_gate_readers import load_series

    np.savez(tmp_path / "rain.npz", totals=np.ones((2, 2, 2)))
    manifest = _saved_manifest(tmp_path, rain_accum_mm={"path": "rain.npz", "variable": "totals", "units": "in"})
    with pytest.raises(ValueError, match="disagree with canonical units 'mm'"):
        load_series(manifest)


def test_manifest_loader_absolute_utc_clock_uses_final_analysis(tmp_path):
    import json
    from gpuwm.verify.rain_gate_readers import load_series

    path = _saved_manifest(tmp_path, times_utc=["2026-06-14T00:00:00Z", "2026-06-14T01:00:00Z"])
    definition = json.loads(path.read_text())
    definition.pop("times_seconds")
    path.write_text(json.dumps(definition))
    series, _ = load_series(path, analysis_end="2026-06-14T00:00:00Z")
    assert np.array_equal(series.times_seconds, [0, 3600])


def test_manifest_loader_rust_netcdf_payload_door_is_used(tmp_path, monkeypatch):
    from gpuwm import netcdf_bridge
    from gpuwm.verify.rain_gate_readers import load_series

    path = tmp_path / "forecast.nc"
    path.write_bytes(b"synthetic-reader-route")
    calls = []

    class Variable:
        units = "mm"

        def __getitem__(self, item):
            return np.zeros((2, 2, 2))

    class Dataset:
        variables = {"RAINC": Variable(), "RAINNC": Variable()}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def native_open(source):
        calls.append(source)
        return Dataset()

    monkeypatch.setattr(netcdf_bridge, "open_dataset", native_open)
    manifest = _saved_manifest(tmp_path, rain_accum_mm={"sum": [
        {"path": "forecast.nc", "variable": "RAINC", "units": "mm"},
        {"path": "forecast.nc", "variable": "RAINNC", "units": "mm"},
    ]})
    _, provenance = load_series(manifest)
    assert calls == [path.resolve(), path.resolve()]
    assert len(provenance["sources"]) == 1


def test_saved_fields_cli_scores_every_primary_lead_and_paired_baseline(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    times = np.arange(0, 21601, 120, dtype=float)
    rate = np.zeros((len(times), 41, 41), dtype=float)
    rate[:, 19:22, 19:22] = 12.0
    accumulated = rate * times[:, None, None] / 3600.0
    echo = np.zeros_like(rate)
    echo[:, 19:22, 19:22] = 40.0
    mask = np.ones_like(rate, dtype=np.uint8)
    saved = tmp_path / "analytic.npz"
    np.savez_compressed(saved, times=times, rate=rate, accum=accumulated, echo=echo, mask=mask, resets=np.zeros(len(times), dtype=np.int64))
    field = lambda name: {"path": saved.name, "variable": name}
    common = {
        "schema": "regional-rain/input.v1",
        "grid": {"x_edges_m": (np.arange(42) * 3000).tolist(), "y_edges_m": (np.arange(42) * 3000).tolist(),
                 "projection_id": "analytic-equal-area"},
        "times_seconds": field("times"), "rain_valid": field("mask"),
        "echo_dbz": field("echo"), "echo_valid": field("mask"), "is_stub": True,
    }
    model = dict(common, rain_accum_mm=field("accum"), reset_ids=field("resets"))
    truth = dict(common, rate_mm_h=field("rate"))
    forecast_path, truth_path = tmp_path / "forecast.json", tmp_path / "truth.json"
    forecast_path.write_text(json.dumps(model))
    truth_path.write_text(json.dumps(truth))
    output = tmp_path / "scores.jsonl"
    script = Path(__file__).resolve().parents[1] / "tools" / "regional_rain_score.py"
    process = subprocess.run([
        sys.executable, str(script), "--forecast", str(forecast_path), "--truth", str(truth_path),
        "--baseline", str(forecast_path), "--analysis-end", "2026-06-14T00:00:00Z",
        "--event", "analytic-only", "--seed", "1", "--product", "synthetic-member-0", "--out", str(output),
    ], text=True, capture_output=True, env=dict(os.environ, GPUWM_NO_LOCAL_GPU="1"))
    assert process.returncode == 0, process.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 36
    assert {row["lead_hours"] for row in rows} == {1.0, 3.0, 6.0}
    assert all(row["measurement_status"] == "complete" for row in rows)
    assert all(row["status"] == "pending" for row in rows)
    assert all(row["value"] == pytest.approx(1.0) for row in rows)
    fss_rows = [row for row in rows if row["metric"] == "fss"]
    assert len(fss_rows) == 27
    assert all(row["paired_gain"] == 0.0 for row in fss_rows)
    assert {(row["threshold_mm_h"], row["scale_km"]) for row in fss_rows} == {
        (threshold, scale) for threshold in (1, 5, 10) for scale in (10, 25, 50)
    }
    receipt = json.loads(output.with_suffix(".receipt.json").read_text())
    assert receipt["exit_code"] == 0
    assert receipt["truth_sources"]["is_stub"] is True
    assert len(receipt["scores_sha256"]) == 64


@pytest.mark.parametrize("coordinate_mode", ["none", "valid", "wrong"])
def test_native_rust_netcdf_reader_and_wps_corner_geometry(tmp_path, coordinate_mode):
    import json
    from gpuwm.io.nc_writer_bridge import ClassicSchema
    from gpuwm.static import rust_bridge
    from gpuwm.verify.rain_gate_readers import load_series

    source = tmp_path / "forecast.nc"
    schema = ClassicSchema()
    dimensions = [schema.def_dim("time", 2), schema.def_dim("south_north", 3), schema.def_dim("west_east", 3)]
    rainc = schema.def_var("RAINC", "f8", dimensions)
    rainnc = schema.def_var("RAINNC", "f8", dimensions)
    schema.put_var_attr(rainc, "units", "mm")
    schema.put_var_attr(rainnc, "units", "kg/m^2")
    coordinate_vars = None
    if coordinate_mode != "none":
        coordinate_vars = (schema.def_var("XLAT", "f4", dimensions[1:]), schema.def_var("XLONG", "f4", dimensions[1:]))
    for name, value in {"CEN_LAT": 35.0, "CEN_LON": -97.0, "TRUELAT1": 30.0, "TRUELAT2": 60.0,
                        "STAND_LON": -98.0, "DX": 3000.0, "DY": 3000.0, "MAP_PROJ": 1}.items():
        schema.put_global_attr(name, value, dtype="f8")
    rain = np.zeros((2, 3, 3), dtype=float)
    rain[1] = 2.0
    coordinates = None
    if coordinate_mode == "valid":
        handle = rust_bridge.grid_new({"kind": "lambert", "ref_lat": 35.0, "ref_lon": -97.0, "truelat1": 30.0,
            "truelat2": 60.0, "stand_lon": -98.0, "dx": 3000.0, "dy": 3000.0, "e_we": 4, "e_sn": 4,
            "known_x": 2.0, "known_y": 2.0, "moad_cen_lat": 35.0, "moad_cen_lon": -97.0})
        try:
            coordinates = tuple(rust_bridge.grid_array(handle, 0, kind, 3, 3).astype("f4") for kind in (0, 1))
        finally:
            rust_bridge.grid_free(handle)
    elif coordinate_mode == "wrong":
        coordinates = np.full((3, 3), 35, dtype="f4"), np.full((3, 3), -97, dtype="f4")
    with schema.create(source) as writer:
        writer.write_var(rainc, rain)
        writer.write_var(rainnc, np.full_like(rain, 0.5))
        if coordinates is not None:
            for variable, field in zip(coordinate_vars, coordinates):
                writer.write_var(variable, field)
    manifest = {
        "schema": "regional-rain/input.v1", "times_seconds": [0, 3600],
        "grid": {"wrf_projection": {"path": source.name}, "equal_area_center": [-97.0, 35.0]},
        "rain_accum_mm": {"sum": [{"path": source.name, "variable": name} for name in ("RAINC", "RAINNC")]},
        "rain_valid": {"constant": 1, "shape": [2, 3, 3]}, "reset_ids": [0, 0], "is_stub": True,
    }
    path = tmp_path / "native.json"
    path.write_text(json.dumps(manifest))
    if coordinate_mode == "wrong":
        with pytest.raises(ValueError, match="projection disagrees with native mass centers"):
            load_series(path)
        return
    series, provenance = load_series(path)
    assert series.grid.shape == (3, 3)
    assert np.all(series.rain_accum_mm[0] == 0.5)
    assert np.all(series.rain_accum_mm[1] == 2.5)
    assert provenance["geometry_source"] == "native_wps_projection_corners_r6370000"
    assert provenance["projection_id"] == "spherical-laea-r6370000:-97,35"
    assert len(provenance["sources"]) == 1
    assert provenance["geometry_check"]["status"] == ("passed_native_mass_coordinates" if coordinate_mode == "valid" else "not_measured_no_XLAT_XLONG")
    xc, yc = series.grid.corners()
    assert xc.shape == yc.shape == (4, 4)
    assert np.all(np.isfinite(xc)) and np.all(np.isfinite(yc))


def test_model_manifest_uses_native_times_and_prefers_issuance_at_analysis_fork(tmp_path):
    import json
    import os
    from datetime import datetime, timezone
    from pathlib import Path
    import subprocess
    import sys
    from gpuwm.io.surface_wrfout import write_surface_wrfout
    from gpuwm.static import rust_bridge
    from gpuwm.verify.rain_gate_readers import load_series

    spec = {"kind": "lambert", "ref_lat": 35.0, "ref_lon": -97.0, "truelat1": 30.0,
            "truelat2": 60.0, "stand_lon": -98.0, "dx": 3000.0, "dy": 3000.0, "e_we": 4, "e_sn": 4,
            "known_x": 2.0, "known_y": 2.0, "moad_cen_lat": 35.0, "moad_cen_lon": -97.0}
    handle = rust_bridge.grid_new(spec)
    try:
        latitude = rust_bridge.grid_array(handle, 0, 0, 3, 3)
        longitude = rust_bridge.grid_array(handle, 0, 1, 3, 3)
    finally:
        rust_bridge.grid_free(handle)
    attributes = {"MAP_PROJ": 1, "CEN_LAT": 35.0, "CEN_LON": -97.0, "TRUELAT1": 30.0,
                  "TRUELAT2": 60.0, "STAND_LON": -98.0, "GPUWM_RAIN_RESET_ID": 0}
    for name, total in [("wrfout_leg.nc", 1.0), ("wrfout_start.nc", 2.0)]:
        write_surface_wrfout(tmp_path / name, {
            "XLAT": latitude, "XLONG": longitude, "RAINC": np.full((3, 3), total),
            "RAINNC": np.zeros((3, 3)), "REFL_COMPOSITE": np.full((3, 3), 40.0),
        }, time_str="2026-06-14_00:00:00", dx=3000.0, global_attrs=attributes,
            start_time=datetime(2026, 6, 14, tzinfo=timezone.utc))
    script = Path(__file__).resolve().parents[1] / "tools" / "regional_rain_manifest.py"
    output = tmp_path / "model.json"
    process = subprocess.run([sys.executable, str(script), "model", "--frames", str(tmp_path),
        "--analysis-end", "2026-06-14T00:00:00Z", "--center-lon", "-97", "--center-lat", "35", "--out", str(output)],
        text=True, capture_output=True, env=dict(os.environ, GPUWM_NO_LOCAL_GPU="1"))
    assert process.returncode == 0, process.stderr
    manifest = json.loads(output.read_text())
    assert manifest["times_utc"] == ["2026-06-14T00:00:00Z"]
    assert len(manifest["frame_selection"]) == 1
    assert manifest["frame_selection"][0]["selected"].endswith("wrfout_start.nc")
    series, provenance = load_series(output, analysis_end="2026-06-14T00:00:00Z")
    assert series.rain_accum_mm.shape == series.echo_dbz.shape == (1, 3, 3)
    assert np.all(series.rain_accum_mm == 2.0)
    assert np.all(series.echo_dbz == 40.0)
    assert provenance["geometry_check"]["status"] == "passed_native_mass_coordinates"
