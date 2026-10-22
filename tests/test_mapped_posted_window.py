"""The as-posted head binds the run window's leads, not every scheduled lead.

Breakage: full CONUS HRRR from HRRR f00 with RAP f03-f06 boundaries (fetch 3 h,
run 2 h) was refused at "an as-posted head names 4 forcing leads for 2
boundary intervals" (2026-10-21).
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from gpuwm.mapped_direct import _posted_window_leads


def _posted(first_lead, count, cycle=datetime(2026, 10, 21, 3)):
    leads = tuple(range(first_lead, first_lead + count))
    return SimpleNamespace(
        leads=leads,
        valid_times=tuple(cycle + timedelta(hours=lead) for lead in leads))


def test_a_fetch_longer_than_the_run_binds_only_the_window():
    posted = _posted(3, 4)  # RAP 03Z f03..f06, valid 06Z..09Z
    start = datetime(2026, 10, 21, 6)
    times = tuple(start + timedelta(hours=h) for h in range(3))  # 2 h run
    assert _posted_window_leads(posted, times) == (3, 4, 5)


def test_a_window_matching_the_schedule_is_unchanged():
    posted = _posted(0, 4)
    times = posted.valid_times
    assert _posted_window_leads(posted, times) == posted.leads


def test_a_time_the_fetch_never_scheduled_is_refused():
    posted = _posted(3, 3)
    times = (datetime(2026, 10, 21, 6), datetime(2026, 10, 21, 10))
    with pytest.raises(ValueError, match="not all scheduled leads"):
        _posted_window_leads(posted, times)
