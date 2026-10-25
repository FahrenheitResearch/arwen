"""Smoke acquisition must reach the final atmosphere preparation endpoint."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from gpuwm.go_cli import chem_fetch_command, chem_window


@pytest.mark.parametrize("start_hour,run_hours,forcing_hours,end_hour", [
    (12, 1, 3, 15),  # Short run still prepares the +3 h boundary.
    (18, 1, 3, 21),  # A nonzero forecast lead is already in start_time.
    (12, 4, 3, 16),  # Never shorten the chemistry needed by the run.
])
def test_chem_fetch_covers_prepared_boundary(start_hour, run_hours, forcing_hours,
                                           end_hour):
    experiment = SimpleNamespace(start_time=datetime(2026, 10, 21, start_hour),
                                 run_seconds=run_hours * 3600)
    window = chem_window(experiment, forcing_hours=forcing_hours)
    assert window == (experiment.start_time.isoformat(),
                      datetime(2026, 10, 21, end_hour).isoformat())
    command = chem_fetch_command({"chem_window": window, "area": "24,-125,50,-66",
                                  "chem_data_store_sources": ["rap-native-smoke"]})
    assert command[command.index("--start") + 1] == window[0]
    assert command[command.index("--end") + 1] == window[1]
