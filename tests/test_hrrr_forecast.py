"""HRRR absolute source-lead and model-relative forcing contracts."""

from datetime import datetime

import pytest

from gpuwm.hrrr_forecast import (
    hrrr_cycle_horizon,
    hrrr_source_window,
    validate_hrrr_source_forecast_hours,
)


def test_extended_and_standard_cycles_have_explicit_public_horizons():
    for hour in (0, 6, 12, 18):
        assert hrrr_cycle_horizon(datetime(2026, 7, 18, hour)) == 48
    for hour in (1, 5, 17, 23):
        assert hrrr_cycle_horizon(datetime(2026, 7, 18, hour)) == 18


@pytest.mark.parametrize(
    ("cycle", "start", "duration", "expected"),
    (
        (datetime(2026, 7, 18, 5), 12, 6 * 3600, tuple(range(12, 19))),
        (datetime(2026, 7, 18, 18), 40, 6 * 3600, tuple(range(40, 47))),
        (datetime(2026, 7, 18, 0), 0, 48 * 3600, tuple(range(49))),
    ),
)
def test_source_windows_retain_absolute_leads_without_duration_cap(
        cycle, start, duration, expected):
    assert hrrr_source_window(
        cycle=cycle, start_hour=start, run_seconds=duration,
        end_hour=expected[-1]) == expected
    assert tuple(range(len(expected))) == tuple(
        hour - expected[0] for hour in expected)


def test_source_window_rejects_duration_end_drift_and_cycle_overrun():
    with pytest.raises(ValueError, match="must be f18"):
        hrrr_source_window(
            cycle=datetime(2026, 7, 18, 5), start_hour=12,
            run_seconds=6 * 3600, end_hour=17)
    with pytest.raises(ValueError, match="horizon f18"):
        hrrr_source_window(
            cycle=datetime(2026, 7, 18, 5), start_hour=18,
            run_seconds=3600)


def test_source_lead_validator_rejects_gaps_duplicates_and_global_overrun():
    for hours in ((12, 14), (12, 12), (48, 49)):
        with pytest.raises(ValueError):
            validate_hrrr_source_forecast_hours(hours)
