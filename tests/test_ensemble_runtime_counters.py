"""Original runtime counter positions, late history and streamed providers."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np

from gpuwm.ensemble.runtime_context import (MemberOutputCapture, bind_current_member_state,
    member_output_scope)
from gpuwm.ensemble.runtime_counters import fixed_counter_observer_for_current


class _Counters:
    def __init__(self):
        self.snapshots = {}
        self.frames = []

    def submit(self, **frame):
        self.frames.append(frame)

    def capture_rain_counters(self, fields, **metadata):
        self.snapshots[metadata["valid_time"]] = {"fields": fields, "metadata": metadata}
        return True

    def has_rain_counter(self, valid_time, **metadata):
        return valid_time in self.snapshots


def _fixture(monkeypatch, *, dt=12., history_begin=1, output_steps=1, outer_steps=2):
    from gpuwm import runtime
    from gpuwm.core import streaming
    rain = np.full((2, 3), 10., np.float32)
    cfg = SimpleNamespace(dt=dt, ny=2, nx=3, mp_physics=8, cu_physics=0)
    state = SimpleNamespace(elapsed_seconds=0., physics=SimpleNamespace(
        microphysics=SimpleNamespace(rainnc=rain), rainc=None))
    prepared = SimpleNamespace(cfg=cfg, grid=object(), static_fields={},
                               initial_result=SimpleNamespace(state=state))
    coords = np.zeros((2, 3), np.float32)
    monkeypatch.setattr(runtime, "_metadata_frame", lambda *a: {"XLAT": coords, "XLONG": coords})
    monkeypatch.setattr(streaming, "is_streaming", lambda stepper: False)
    options = {"start_time": datetime(2026, 10, 2), "domain_id": 1,
        "run_seconds": dt * outer_steps, "outer_steps": outer_steps,
        "output_outer_steps": output_steps, "history_begin_step": history_begin,
        "history_end_step": None, "write_final_output": False}
    return prepared, rain, options


def test_plain_runtime_builds_no_counter_calendar(monkeypatch):
    prepared, _, options = _fixture(monkeypatch)
    assert fixed_counter_observer_for_current(prepared, object(), **options) is None


def test_late_history_seeds_actual_start_without_inventing_a_frame(monkeypatch):
    prepared, rain, options = _fixture(monkeypatch)
    collector = _Counters()
    capture = MemberOutputCapture(collector.submit, 19)
    with member_output_scope(capture):
        observer = fixed_counter_observer_for_current(prepared, object(), **options)
        observer.observe()
        prepared.initial_result.state.elapsed_seconds = 12.
        rain[:] = 13.
        before = rain.tobytes()
        observer.observe()
        observer.history_consumed(options["start_time"] + timedelta(seconds=12))
    assert collector.frames == []
    assert tuple(collector.snapshots) == (options["start_time"], options["start_time"] + timedelta(seconds=12))
    assert rain.tobytes() == before
    assert collector.snapshots[options["start_time"]]["fields"]["RAINNC"] is rain
    assert collector.snapshots[options["start_time"]]["metadata"]["absent_zero_fields"] == ("RAINC", "RAINSH")
    assert observer.calendar.receipt(member_id=19)["captured_ticks"] == [0, 12_000_000]


def test_fractional_actual_clock_misses_endpoint_without_clipping(monkeypatch):
    prepared, _, options = _fixture(monkeypatch, dt=0.1, history_begin=3, outer_steps=3)
    collector = _Counters()
    capture = MemberOutputCapture(collector.submit, 0)
    with member_output_scope(capture):
        observer = fixed_counter_observer_for_current(prepared, object(), **options)
        observer.observe()
        prepared.initial_result.state.elapsed_seconds = 0.299999
        observer.observe()
        prepared.initial_result.state.elapsed_seconds = 0.300001
        observer.observe()
    assert prepared.initial_result.state.elapsed_seconds == 0.300001
    assert tuple(collector.snapshots) == (options["start_time"],)
    assert observer.calendar.receipt(member_id=0)["unavailable_ticks"][0]["ticks"] == 300000


def test_resume_does_not_fabricate_forecast_start_counter(monkeypatch):
    prepared, _, options = _fixture(monkeypatch)
    prepared.initial_result.state.elapsed_seconds = 12.
    collector = _Counters()
    with member_output_scope(MemberOutputCapture(collector.submit, 0)):
        observer = fixed_counter_observer_for_current(prepared, object(), **options)
        observer.observe()
    assert options["start_time"] not in collector.snapshots
    receipt = observer.calendar.receipt(member_id=0)
    assert receipt["captured_ticks"] == [12_000_000]
    assert receipt["unavailable_ticks"][0]["ticks"] == 0


def test_fixed_streaming_reads_only_actual_two_dimensional_store(monkeypatch):
    from gpuwm.core import streaming
    prepared, _, options = _fixture(monkeypatch)
    class StaleState:
        @property
        def physics(self):
            raise AssertionError("counter observation read stale resident physics")
    prepared.initial_result.state = StaleState()
    actual = np.full((2, 3), 4., np.float32)
    drains = []
    stepper = SimpleNamespace(scalars={"elapsed_seconds": 12.},
        store={"scratch/mp_rainnc": actual}, _run=SimpleNamespace(drain=lambda: drains.append(True)))
    monkeypatch.setattr(streaming, "is_streaming", lambda value: value is stepper)
    collector = _Counters()
    with member_output_scope(MemberOutputCapture(collector.submit, 3)):
        observer = fixed_counter_observer_for_current(prepared, stepper, **options)
        observer.observe()
    assert drains == [True]
    row = collector.snapshots[options["start_time"] + timedelta(seconds=12)]
    assert row["fields"]["RAINNC"] is actual


def test_single_state_binding_passes_original_owners_without_model_or_clock():
    calls = []
    prepared, state, cfg, grid = object(), object(), object(), object()
    capture = MemberOutputCapture(lambda **frame: None, 4,
                                  initialize_callback=lambda **owner: calls.append(owner))
    with member_output_scope(capture):
        bind_current_member_state(prepared_case=prepared, state=state, cfg=cfg, grid=grid)
    assert calls == [{"prepared_case": prepared, "state": state, "cfg": cfg, "grid": grid, "clock": None}]


def test_tree_counter_context_forwards_live_streamed_marker(monkeypatch):
    from gpuwm import runtime
    prepared, _, options = _fixture(monkeypatch)
    actual = np.full((2, 3), 9., np.float32)
    stream = SimpleNamespace(store={"scratch/mp_rainnc": actual}, _run=SimpleNamespace(drain=lambda: None))
    class StaleState:
        _streamed_domain = stream
        @property
        def physics(self):
            raise AssertionError("tree counter observation read stale resident physics")
    clock = SimpleNamespace(ticks=0, tick_den=1, run_ticks=24, elapsed_seconds=0.,
        spec=SimpleNamespace(grid_id=1, start_ticks=0, history_ticks=12,
                             history_begin_ticks=12, history_end_ticks=None))
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=1, run=prepared.cfg), clock=clock,
                           state=StaleState(), grid=prepared.grid, _started=True)
    model = SimpleNamespace(walk_parent_first=lambda: (node,), _prepared_by_grid_id={1: prepared})
    collector = _Counters()
    capture = MemberOutputCapture(collector.submit, 2)
    capture.observe_model_counters(model, start_time=options["start_time"])
    assert collector.snapshots[options["start_time"]]["fields"]["RAINNC"] is actual
