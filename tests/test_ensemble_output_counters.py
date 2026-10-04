"""Exact endpoint calendars and selected resident/streamed counter reads."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.clock import DomainClock, DomainTicks
from gpuwm.ensemble.output_counters import CounterDeadlineCalendar, capture_due_counter


def test_legacy_valid_time_calendar_retains_original_fractional_dates_without_clock_or_added_frames():
    start = datetime(2024, 1, 1)
    actual = (start + timedelta(seconds=3 * 0.1 + 3 * 3600),)
    calendar = CounterDeadlineCalendar.from_valid_times(grid_id=1, start_time=start,
        run_end_time=start + timedelta(hours=4), history_valid_times=actual)
    assert calendar.history_ticks == (10_800_300_000,)
    assert 300_000 in calendar.deadlines and calendar.start_ticks == 0
    assert calendar.tick_den == 1_000_000
    assert calendar.observe(299_999, member_id=19)[0] is False
    assert calendar.observe(300_001, member_id=19)[0] is False
    assert any(row["ticks"] == 300_000 for row in calendar.receipt(member_id=19)["unavailable_ticks"])


def _node(*, start=0, begin=4 * 3600, end=8 * 3600, interval=2 * 3600, stop=10 * 3600,
          mp=8, cu=1, grid_id=1):
    spec = DomainTicks(grid_id, 0 if grid_id == 1 else 1, 1, 60, np.float32(60), interval,
        None, None, None, None, None, None, None, start_ticks=start,
        history_begin_ticks=begin, history_end_ticks=end)
    clock = DomainClock(spec, 1, stop)
    clock.ticks = start
    cfg = SimpleNamespace(ny=3, nx=4, mp_physics=mp, cu_physics=cu)
    rainnc, rainc = np.full((3, 4), 50.0, np.float32), np.full((3, 4), 5.0, np.float32)
    physics = SimpleNamespace(microphysics=SimpleNamespace(rainnc=rainnc), rainc=rainc)
    return SimpleNamespace(clock=clock, cfg=SimpleNamespace(run=cfg, grid_id=grid_id),
                           state=SimpleNamespace(physics=physics), _started=True)


def _capture(node, calendar, collector, *, member=0, streamed=None, fields=None):
    metadata = {"XLAT": np.zeros((3, 4), np.float32), "XLONG": np.ones((3, 4), np.float32)}
    return capture_due_counter(node, collector, member, calendar=calendar,
        start_time=datetime(2024, 1, 1), metadata=metadata, streamed=streamed, counter_fields=fields)


def _collector(calls):
    def capture(fields, **kwargs):
        calls.append((fields, kwargs))
    return SimpleNamespace(capture_rain_counters=capture)


def test_deadlines_include_baseline_and_window_endpoints_without_extra_output_alarms():
    node = _node()
    original = vars(node.clock.spec).copy()
    calendar = CounterDeadlineCalendar.from_node(node)
    assert calendar.history_ticks == (14400, 21600, 28800)
    assert calendar.deadlines == tuple(hour * 3600 for hour in range(9))
    assert vars(node.clock.spec) == original
    assert not node.clock.history_due()
    calls = []
    assert _capture(node, calendar, _collector(calls)).status == "captured"
    assert len(calls) == 1
    assert calls[0][1]["valid_time"] == datetime(2024, 1, 1)
    assert calls[0][1]["absent_zero_fields"] == ("RAINSH",)
    assert not node.clock.history_due()


def test_counter_deadlines_track_every_member_without_changing_counter_words():
    node = _node()
    values = np.resize(np.array([0, 0x80000000, 0x7FC00017], np.uint32), 12).reshape(3, 4).view(np.float32)
    node.state.physics.microphysics.rainnc = values
    before = values.tobytes()
    calendar, calls = CounterDeadlineCalendar.from_node(node), []
    collector = _collector(calls)
    for member in (0, 1):
        assert _capture(node, calendar, collector, member=member).status == "captured"
        assert _capture(node, calendar, collector, member=member).status == "not_due"
    assert [kwargs["member_id"] for fields, kwargs in calls] == [0, 1]
    assert calls[0][0]["RAINNC"] is values
    assert values.tobytes() == before


def test_adaptive_jump_marks_the_exact_missed_endpoint_and_never_interpolates():
    node = _node(begin=7200, end=7200, interval=7200, stop=7200)
    calendar, calls = CounterDeadlineCalendar.from_node(node), []
    collector = _collector(calls)
    assert _capture(node, calendar, collector).status == "captured"
    node.clock.ticks = 3599
    assert _capture(node, calendar, collector).status == "not_due"
    node.clock.ticks = 3601
    result = _capture(node, calendar, collector)
    assert result.status == "not_due" and result.missed[0]["ticks"] == 3600
    node.clock.ticks = 7200
    assert _capture(node, calendar, collector).status == "captured"
    unavailable = calendar.unavailable_endpoints(member_id=0)
    assert any(row["field"] == "qpf_1h" and row["endpoint_ticks"] == 3600 for row in unavailable)
    assert len(calls) == 2
    assert node.clock.step_ticks == 60 and node.clock.dt_fp32 == np.float32(60)


def test_delayed_child_captures_at_its_activation_and_marks_earlier_windows_unavailable():
    node = _node(start=7200, begin=0, end=3600, interval=3600, stop=10800, grid_id=2)
    calendar, calls = CounterDeadlineCalendar.from_node(node), []
    node.clock.ticks, node._started = 0, False
    assert _capture(node, calendar, _collector(calls)).status == "dormant"
    assert calls == []
    node.clock.ticks, node._started = 7200, True
    assert _capture(node, calendar, _collector(calls)).status == "captured"
    assert calls[0][1]["valid_time"] == datetime(2024, 1, 1, 2)
    assert calendar.receipt(member_id=0)["captured_ticks"] == [7200]
    assert any(row["field"] == "qpf_3h" and "activation" in row["reason"]
               for row in calendar.unavailable_endpoints(member_id=0))


def test_nonranked_stream_reads_only_live_surface_counters_after_scatter_drain():
    node = _node()
    live_nc, live_c = np.full((3, 4), 70.0, np.float32), np.full((3, 4), 9.0, np.float32)
    calls, drains = [], []
    class Store(dict):
        def __getitem__(self, key):
            assert key in ("scratch/mp_rainnc", "scratch/cu_rainc")
            return super().__getitem__(key)
    store = Store({"scratch/mp_rainnc": live_nc, "scratch/cu_rainc": live_c, "state/thp": object()})
    streamed = SimpleNamespace(store=store, _run=SimpleNamespace(drain=lambda: drains.append(True)))
    result = _capture(node, CounterDeadlineCalendar.from_node(node), _collector(calls), streamed=streamed)
    assert result.status == "captured" and drains == [True]
    assert calls[0][0]["RAINNC"] is live_nc and calls[0][0]["RAINC"] is live_c
    assert calls[0][0]["RAINNC"] is not node.state.physics.microphysics.rainnc


def test_ranked_stream_downloads_only_selected_two_dimensional_members():
    node = _node()
    live = {"scratch/mp_rainnc": np.full((3, 4), 77.0, np.float32),
            "scratch/cu_rainc": np.full((3, 4), 8.0, np.float32), "state/thp": object()}
    calls, operations = [], []
    class Run:
        ranked = True
        raw_store = live
        @property
        def store(self):
            raise AssertionError("a ranked full-store drain was requested")
        def download(self, names):
            operations.append(("download", names))
        def pending_downloads(self):
            return ({"names": ("scratch/mp_rainnc", "scratch/cu_rainc"), "events": ()},)
        def wait_downloads(self, pending):
            operations.append(("wait", pending))
    streamed = SimpleNamespace(_run=Run())
    result = _capture(node, CounterDeadlineCalendar.from_node(node), _collector(calls), streamed=streamed)
    assert result.status == "captured"
    assert operations[0] == ("download", ("scratch/mp_rainnc", "scratch/cu_rainc"))
    assert operations[1][0] == "wait"
    assert set(calls[0][0]) == {"RAINNC", "RAINC"}


def test_unknown_streaming_api_reports_unavailable_without_reading_stale_state_or_history():
    node = _node()
    def forbidden():
        raise AssertionError("full-volume projection or stale state was read")
    streamed = SimpleNamespace(_run=SimpleNamespace(ranked=True), history_fields=forbidden)
    calls = []
    calendar = CounterDeadlineCalendar.from_node(node)
    result = _capture(node, calendar, _collector(calls), streamed=streamed)
    assert result.status == "unavailable" and "selected-counter" in result.reason
    assert calls == []
    assert calendar.receipt(member_id=0)["unavailable_ticks"][0]["ticks"] == 0


def test_original_off_scheme_zeros_require_no_driver_store_or_field_allocation():
    node = _node(mp=0, cu=0)
    node.state.physics = None
    calls = []
    result = _capture(node, CounterDeadlineCalendar.from_node(node), _collector(calls), streamed=object())
    assert result.status == "captured" and calls[0][0] == {}
    assert calls[0][1]["absent_zero_fields"] == ("RAINC", "RAINNC", "RAINSH")


def test_explicit_native_member_views_are_borrowed_without_output_field_construction():
    node = _node(cu=0)
    node.state.physics = None
    rain = np.full((3, 4), 66.0, np.float32)
    calls = []
    result = _capture(node, CounterDeadlineCalendar.from_node(node), _collector(calls), fields={"RAINNC": rain})
    assert result.status == "captured" and calls[0][0]["RAINNC"] is rain
    assert calls[0][1]["absent_zero_fields"] == ("RAINC", "RAINSH")


def test_counter_capture_rejects_volumes_and_does_not_call_the_collector():
    node = _node()
    calls = []
    with pytest.raises(ValueError, match="two-dimensional"):
        _capture(node, CounterDeadlineCalendar.from_node(node), _collector(calls), fields={"RAINNC": np.zeros((8, 3, 4), np.float32)})
    assert calls == []


@pytest.mark.parametrize("invalid", [(True,), (0,), (1, 1), (1.5,)])
def test_deadline_calendar_does_not_coerce_invalid_windows(invalid):
    with pytest.raises((TypeError, ValueError)):
        CounterDeadlineCalendar.from_node(_node(), windows_hours=invalid)
