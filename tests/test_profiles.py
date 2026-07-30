"""CPU pins for reusable metrics and data-driven verification profiles."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
from types import SimpleNamespace
import struct

import numpy as np
import pytest

from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.verify import metrics
from gpuwm.verify.cases import real74_d01
from gpuwm.verify.profiles import (
    AnalysisProfile, AnalysisRecipe, HealthProfile, OracleProfile,
    OutputSchema, Threshold, VerificationProfile)
from tools.derive_persistence_baseline import derive_persistence


FROZEN_GATES_BYTES = (
    b"{'temperature_rmse_max_k': (None, 1.0), "
    b"'wind_rmse_max_ms': (None, 2.0), "
    b"'mslp_pattern_correlation': (0.98, None), "
    b"'boundary_zone_blowup': (None, 0.5), "
    b"'t2_snow_free_abs_bias_k': (None, 10.0), "
    b"'t2_min_k': (200.0, None), "
    b"'ysu_nan_guard_fires': (None, 0.5), "
    b"'wrf_tooling_diagnostic_count': (4.5, None), "
    b"'era5_t500_rmse_k': (None, 2.5), "
    b"'era5_t500_pattern_correlation': (0.95, None)}"
)


def test_real74_frozen_profile_gate_and_persistence_bytes_are_unchanged():
    """Any value, type, or insertion-order drift changes these bytes."""
    assert repr(real74_d01.GATES).encode("ascii") == FROZEN_GATES_BYTES
    persistence = struct.pack(
        ">dd", real74_d01.ERA5_PERSISTENCE_T500_RMSE_K,
        real74_d01.ERA5_PERSISTENCE_T500_CORRELATION)
    assert persistence.hex() == "4007ef34d6a161e53feb9652bd3c3611"
    assert real74_d01.GATES == real74_d01.REAL74_PROFILE.gates()


def test_real74_profile_separates_health_analysis_and_optional_oracle():
    profile = real74_d01.REAL74_PROFILE
    assert isinstance(profile, VerificationProfile)
    assert isinstance(profile.health, HealthProfile)
    assert isinstance(profile.analysis, AnalysisProfile)
    assert isinstance(profile.oracle, OracleProfile)
    assert profile.oracle.reference_paths == ((
        datetime(1974, 4, 3, 13), real74_d01.REFERENCE_13Z),)
    assert profile.oracle.masks == ("interior",)
    assert [recipe.valid_time for recipe in profile.analysis.recipes] == [
        datetime(1974, 4, 4, 0)] * 4


def test_extracted_metric_functions_are_the_frozen_case_functions():
    assert real74_d01._rmse is metrics._rmse
    assert real74_d01._pattern_correlation is metrics._pattern_correlation
    assert real74_d01._interpolate_to_pressure is metrics._interpolate_to_pressure
    assert real74_d01._wrf_diagnostics is metrics._wrf_diagnostics
    assert real74_d01.interior_region is metrics.interior_region
    assert real74_d01._dcomputeseaprs is metrics._dcomputeseaprs


def test_extracted_metric_arithmetic_has_exact_regression_pins():
    left = np.array(
        [[1.0, 2.0, np.nan], [4.0, 8.0, 16.0]], dtype=np.float64)
    right = np.array(
        [[1.5, 1.0, np.nan], [5.0, 7.0, np.nan]], dtype=np.float64)
    assert metrics._rmse(left, right).hex() == "0x1.cd82b446159f3p-1"
    assert metrics._pattern_correlation(left, right).hex() == (
        "0x1.e2db65b5aa9bep-1")

    pressure = np.array([
        [[1000.0, 950.0], [925.0, 900.0]],
        [[800.0, 750.0], [700.0, 650.0]],
        [[500.0, 450.0], [400.0, 350.0]],
    ])
    field = np.array([
        [[10.0, 20.0], [30.0, 40.0]],
        [[50.0, 60.0], [70.0, 80.0]],
        [[90.0, 100.0], [110.0, 120.0]],
    ])
    interpolated = metrics._interpolate_to_pressure(field, pressure, 700.0)
    assert hashlib.sha256(interpolated.tobytes()).hexdigest() == (
        "63facd050c0d23628a86442f163ac451963d7b4fc8df4dcda0297d379ba03895")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_candidate_nonfinite_on_reference_support_fails_pair_metrics(bad):
    """Reference NaNs define missing support; candidate failures do not."""
    candidate = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    reference = candidate.copy()
    candidate[0, 0] = bad

    assert np.isnan(metrics.rmse(candidate, reference))
    assert np.isnan(metrics.pattern_correlation(candidate, reference))


def test_reference_missing_data_remains_excluded_from_pair_metrics():
    candidate = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float64)
    reference = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float64)

    assert metrics.rmse(candidate, reference) == 0.0
    assert metrics.pattern_correlation(candidate, reference) == 1.0

    # NaN is the explicit derived-field missing marker.  Infinity is never
    # valid candidate data, even at a reference-missing cell.
    candidate[-1, -1] = np.inf
    assert np.isnan(metrics.rmse(candidate, reference))
    assert np.isnan(metrics.pattern_correlation(candidate, reference))


def test_profile_records_are_frozen_and_recipe_schema_is_exact():
    assert [field.name for field in fields(AnalysisRecipe)] == [
        "valid_time", "field", "level", "mask", "metric", "threshold"]
    threshold = Threshold("t500_rmse", upper=2.5)
    recipe = AnalysisRecipe(
        datetime(1999, 5, 4), "TT", 500.0, "interior", "rmse", threshold)
    with pytest.raises(FrozenInstanceError):
        recipe.mask = "full"
    with pytest.raises(ValueError, match="no bound"):
        Threshold("unbounded")
    with pytest.raises(ValueError, match="unique"):
        AnalysisProfile("duplicate", (recipe, recipe))


def test_generic_health_profile_passes_without_any_oracle():
    start = datetime(1999, 5, 3, 12)
    summary = SimpleNamespace(
        nan_free=True, completed_seconds=21660.0, w_max_ms=12.0,
        boundary_w_max_ms=8.0, interior_w_max_ms=12.0,
        boundary_zone_blowup=False, ysu_nan_guard_fires=0,
        wrfout_paths=())
    # Deliberately NO OracleProfile: this exercises the oracle-free path.
    profile = HealthProfile.generic_real_case()
    report = profile.evaluate(
        summary, start_time=start, expected_completed_seconds=21660.0,
        forcing_times=(start, start + timedelta(hours=6),
                       start + timedelta(hours=12)))
    assert report.ok, report.failures
    assert report.metrics["forcing_coverage"] is True

    bad = SimpleNamespace(**{**vars(summary), "completed_seconds": 1.0,
                             "ysu_nan_guard_fires": 1})
    failures = profile.evaluate(
        bad, start_time=start, expected_completed_seconds=21660.0,
        forcing_times=(start, start + timedelta(hours=12))).failures
    assert any("completed clock" in failure for failure in failures)
    assert any("ysu_nan_guard_fires" in failure for failure in failures)


def test_health_profile_expresses_cfl_and_vertical_velocity_bounds():
    profile = HealthProfile(
        name="bounded",
        finite_state=False, completed_clock=False, forcing_coverage=False,
        boundary_diagnostics=False, interior_diagnostics=False,
        cfl_bound=Threshold("cfl_max", upper=1.0),
        vertical_velocity_bound=Threshold("w_max_ms", upper=50.0))
    good = SimpleNamespace(cfl_max=0.75, w_max_ms=20.0)
    assert profile.evaluate(good).ok
    bad = SimpleNamespace(cfl_max=1.25, w_max_ms=51.0)
    failures = profile.evaluate(bad).failures
    assert any("cfl_max" in failure for failure in failures)
    assert any("w_max_ms" in failure for failure in failures)


def test_output_schema_validates_calendar_inventory_dimensions_and_finiteness(
        tmp_path):
    from gpuwm.io.wrfout import WrfoutWriter, wrfout_filename

    start = datetime(1999, 5, 3, 12)
    nx, ny, nz = 4, 3, 2
    path = tmp_path / wrfout_filename(start, 1)
    zeros3 = np.zeros((nz, ny, nx), dtype=np.float32)
    fields_by_name = {
        "U": np.zeros((nz, ny, nx + 1), np.float32),
        "V": np.zeros((nz, ny + 1, nx), np.float32),
        "W": np.zeros((nz + 1, ny, nx), np.float32),
        "T": zeros3, "P": zeros3, "PB": zeros3,
        "PH": np.zeros((nz + 1, ny, nx), np.float32),
        "PHB": np.zeros((nz + 1, ny, nx), np.float32),
        "QVAPOR": zeros3,
        "XLAT": np.zeros((ny, nx), np.float32),
        "XLONG": np.zeros((ny, nx), np.float32),
        "T2": np.full((ny, nx), 290.0, np.float32),
    }
    with WrfoutWriter(
            path, nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0) as writer:
        writer.write_frame(start.strftime("%Y-%m-%d_%H:%M:%S"),
                           fields_by_name)
    assert OutputSchema().failures(
        (path,), start_time=start, run_seconds=0.0,
        history_interval_s=3600.0, domain_id=1,
        nx=nx, ny=ny, nz=nz) == ()

    # Failure paths: an implementation that always returns () must not pass.
    calendar = OutputSchema().failures(
        (path,), start_time=start, run_seconds=3600.0,
        history_interval_s=3600.0, domain_id=1, nx=nx, ny=ny, nz=nz)
    assert any("output calendar mismatch" in f for f in calendar)

    ghost = tmp_path / wrfout_filename(start + timedelta(hours=1), 1)
    missing_file = OutputSchema().failures(
        (path, ghost), start_time=start, run_seconds=3600.0,
        history_interval_s=3600.0, domain_id=1, nx=nx, ny=ny, nz=nz)
    assert any("output is missing" in f for f in missing_file)

    wrong_dims = OutputSchema().failures(
        (path,), start_time=start, run_seconds=0.0,
        history_interval_s=3600.0, domain_id=1, nx=nx, ny=ny, nz=nz + 1)
    assert any("dimension" in f and "bottom_top" in f for f in wrong_dims)

    wants_refl = OutputSchema(
        required_variables=OutputSchema().required_variables + ("REFL_10CM",))
    inventory = wants_refl.failures(
        (path,), start_time=start, run_seconds=0.0,
        history_interval_s=3600.0, domain_id=1, nx=nx, ny=ny, nz=nz)
    assert any("missing variables" in f and "REFL_10CM" in f
               for f in inventory)

    nan_path = tmp_path / "nan" / wrfout_filename(start, 1)
    nan_path.parent.mkdir()
    nan_fields = dict(fields_by_name)
    nan_fields["T2"] = np.full((ny, nx), np.nan, np.float32)
    with WrfoutWriter(
            nan_path, nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0) as writer:
        writer.write_frame(start.strftime("%Y-%m-%d_%H:%M:%S"), nan_fields)
    non_finite = OutputSchema().failures(
        (nan_path,), start_time=start, run_seconds=0.0,
        history_interval_s=3600.0, domain_id=1, nx=nx, ny=ny, nz=nz)
    assert any("T2 is non-finite" in f for f in non_finite)


def _snapshot(valid_time, values) -> Era5Snapshot:
    values = np.asarray(values, dtype=np.float64)
    return Era5Snapshot(
        valid_time=valid_time,
        levels_hpa=np.array([500.0], dtype=np.float64),
        latitude=np.array([0.0, 1.0], dtype=np.float64),
        longitude=np.array([10.0, 11.0, 12.0], dtype=np.float64),
        fields={"TT": values[None]},
    )


def test_persistence_uses_the_supplied_case_initial_and_final_analyses():
    initial = _snapshot(
        datetime(1999, 5, 3, 12), [[250.0, 251.0, 252.0],
                                    [253.0, 254.0, 255.0]])
    final = _snapshot(
        datetime(1999, 5, 4, 0), [[251.0, 252.0, 253.0],
                                  [254.0, 255.0, 256.0]])
    rmse, correlation = derive_persistence(
        initial, final, field="TT", level_hpa=500.0, mask="full")
    assert rmse == 1.0
    assert correlation == 1.0
    with pytest.raises(ValueError, match="later"):
        derive_persistence(final, initial)


def test_runtime_modules_contain_no_verification_profile_gate_constants():
    root = Path(__file__).resolve().parents[1]
    runtime_source = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("gpuwm/runtime.py", "gpuwm/case_data.py"))
    # Raw health facts (boundary_zone_blowup and guard-fire count) are part
    # of RealCaseRunSummary; case thresholds and all analysis/oracle gate
    # policy must remain absent from runtime modules.
    runtime_health_facts = {"boundary_zone_blowup", "ysu_nan_guard_fires"}
    for forbidden in (
        *(set(real74_d01.GATES) - runtime_health_facts),
        "ERA5_PERSISTENCE_T500_RMSE_K",
        "ERA5_PERSISTENCE_T500_CORRELATION",
        "OracleProfile", "AnalysisProfile", "HealthProfile", "Threshold(",
        "GATES =",
    ):
        assert forbidden not in runtime_source
