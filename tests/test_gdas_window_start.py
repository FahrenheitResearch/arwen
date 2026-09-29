"""A GDAS window may begin on any published lead, and a single lead takes a cadence.

Breakage this prevents: GDAS publishes every hour f000..f009, yet
``--forecast-start-hour 1 --hours 3 --cadence 3`` (leads f001 and f004)
was refused with "--forecast-start-hour 1 is not on the 3 h cadence", a
rule GFS had already dropped, and ``--hours 0 --cadence 3`` (one lead)
was refused with "--cadence does not apply to a single time", which GFS
accepts.  The cadence spaces the leads of a window; it neither fixes where
the window starts nor has anything to break in a window of one lead.

Both doors ask the same resolver: ``gpuwm fetch`` and a ``[fetch]`` table.
"""
from __future__ import annotations

import pytest

from gpuwm.fetch import (GDAS_PUBLISHED_HOURS, container_forecast_hours,
                         gdas_forecast_hours, validate_fetch_hints)


@pytest.mark.parametrize("hours,cadence,start,expected", [
    (3, 3, 1, (1, 4)),
    (4, 2, 1, (1, 3, 5)),
    (6, 3, 2, (2, 5, 8)),
    (0, None, 1, (1,)),
    (0, 3, 1, (1,)),
    (0, 6, 0, (0,)),
    (6, None, 0, (0, 1, 2, 3, 4, 5, 6)),  # the row's hourly spacing
])
def test_every_published_window_is_accepted_at_both_doors(hours, cadence, start, expected):
    assert set(expected) <= set(GDAS_PUBLISHED_HOURS)
    assert container_forecast_hours("gdas", hours, cadence, start) == expected
    hints = {"source": "gdas", "hours": hours, "forecast_start_hour": start}
    if cadence is not None:
        hints["cadence"] = cadence
    validate_fetch_hints(hints, source="gdas.toml")


@pytest.mark.parametrize("hours,cadence,start,needle", [
    (9, 3, 1, "publishes"),      # ends at f010, past the published f009
    (4, 3, 0, "does not divide"),  # the last frame would be dropped
    (3, 0, 0, "positive whole number"),
    (3, -1, 0, "positive whole number"),
    (3, 3, -1, "nonnegative forecast lead"),
    (3, 3, True, "nonnegative forecast lead"),
])
def test_unpublished_or_malformed_windows_still_refuse(hours, cadence, start, needle):
    with pytest.raises(ValueError, match=needle):
        container_forecast_hours("gdas", hours, cadence, start)


def test_the_start_is_checked_the_same_way_on_gdas_and_gfs():
    """Both ladders accept f001 with a 3 h cadence; neither accepts a negative lead."""
    from gpuwm.fetch import gfs_forecast_hours

    assert gdas_forecast_hours(3, 3, 1) == gfs_forecast_hours(3, 3, 1) == (1, 4)
    for ladder in (gdas_forecast_hours, gfs_forecast_hours):
        with pytest.raises(ValueError, match="nonnegative forecast lead"):
            ladder(3, 3, -3)
