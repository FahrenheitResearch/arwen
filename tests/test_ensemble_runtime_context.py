from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.runtime_context import (
    MemberOutputCapture, current_capture, current_session, ensemble_scope,
    member_output_scope, initialized_bootstrap_handoff,
)


def test_member_scope_suppresses_recursive_dispatch_and_restores_on_failure():
    session = object()
    capture = MemberOutputCapture(lambda **kw: None, 7)
    with ensemble_scope(session):
        assert current_session() is session
        with pytest.raises(RuntimeError):
            with member_output_scope(capture):
                assert current_capture() is capture
                assert current_session() is None
                raise RuntimeError("integration failed")
        assert current_session() is session
        assert current_capture() is None
    assert current_session() is None


def test_tree_capture_uses_live_streamed_owner_and_keeps_actual_paths_empty(tmp_path):
    from gpuwm.io.wrfout import PerDomainWrfoutWriters
    rows = []
    writer = object.__new__(PerDomainWrfoutWriters)
    writer.start_time = datetime(2024, 1, 1)
    writer.output_dir = tmp_path
    writer._episode_by_grid_id = {2: 3}
    writer._published_paths = set()
    writer._captured_paths = []
    metadata = {"XLAT": object(), "XLONG": object()}
    writer._metadata_by_grid_id = {2: metadata}
    # No ordinary writer exists in this shell: aggregate capture must return
    # before a full-frame staging allocation or ordinary file submission.
    writer._writers = {}
    writer._archived_paths = []
    live_store = object()
    state = SimpleNamespace(_streamed_domain=live_store)
    node = SimpleNamespace(state=state, cfg=SimpleNamespace(grid_id=2),
                           clock=SimpleNamespace(tick_den=10))
    refl = object()
    with member_output_scope(MemberOutputCapture(lambda **kw: rows.append(kw), 4)):
        writer.submit(node, 300, refl_field=refl)
        with pytest.raises(RuntimeError, match="duplicate valid time"):
            writer.submit(node, 300, refl_field=refl)
    assert len(rows) == 1
    assert rows[0] == dict(state=state, streamed=live_store, metadata=metadata,
                           refl_field=refl, valid_time=datetime(2024, 1, 1, 0, 0, 30),
                           grid_id=2, episode=3, member_id=4)
    assert writer.paths == ()
    assert len(writer.captured_paths) == 1
    assert writer.captured_paths[0].parent == tmp_path / "d02" / "episode-003"
    assert not list(tmp_path.rglob("wrfout*"))


def test_single_capture_returns_without_full_state_transfer(tmp_path, monkeypatch):
    from gpuwm import runtime
    from gpuwm.io import wrfout
    rows = []
    metadata = {"XLAT": object(), "XLONG": object()}
    state = SimpleNamespace(_streamed_domain=object())
    prepared = SimpleNamespace(initial_result=SimpleNamespace(state=state),
                               grid=object(), static_fields=object(),
                               cfg=SimpleNamespace(mp_physics=0))
    monkeypatch.setattr(runtime, "_metadata_frame", lambda *args: metadata)
    monkeypatch.setattr(wrfout, "state_frame", lambda *args, **kw: pytest.fail("full D2H"))
    valid = datetime(2024, 1, 1)
    with member_output_scope(MemberOutputCapture(lambda **kw: rows.append(kw), 9)):
        path = runtime.write_case_output(prepared, tmp_path, valid,
                                         start_time=valid, title="test", domain_id=3)
    assert path is None
    assert rows[0]["streamed"] is state._streamed_domain
    assert rows[0]["member_id"] == 9
    assert rows[0]["metadata"] is metadata
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("captured", [False, True])
def test_single_streamed_history_consumes_each_due_reflectivity_handoff_once(tmp_path, monkeypatch, captured):
    from contextlib import nullcontext
    from datetime import timedelta
    import numpy as np
    from gpuwm import runtime
    from gpuwm.core.refl import stash_refl_10cm
    from gpuwm.io import wrfout
    words = np.array([0x80000000, 0x3f800001], np.uint32).view(np.float32).reshape(1, 1, 2)
    driver = SimpleNamespace(refl_10cm=None)
    fields = {"REFL_10CM": words}
    state = SimpleNamespace(qv=np.zeros_like(words), physics=driver,
        _streamed_domain=SimpleNamespace(history_fields=lambda: dict(fields)))
    cfg = SimpleNamespace(mp_physics=8, nx=2, ny=1, nz=1, dx=3000., dy=3000.,
                          sf_surface_physics=2, num_soil_layers=4)
    prepared = SimpleNamespace(cfg=cfg, grid=object(), static_fields={},
        initial_result=SimpleNamespace(state=state, coord=object()))
    monkeypatch.setattr(runtime, "_metadata_frame", lambda *args: {})
    monkeypatch.setattr(runtime, "_global_wrf_attrs", lambda *args, **kwargs: {})
    written, frames = [], []
    class Writer:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def write_frame(self, valid, frame):
            written.append(frame["REFL_10CM"].tobytes())
    monkeypatch.setattr(wrfout, "WrfoutWriter", Writer)
    capture = MemberOutputCapture(lambda **row: frames.append(row), 17)
    start = datetime(2024, 1, 1)
    with member_output_scope(capture) if captured else nullcontext():
        for step in (1, 2):
            stash_refl_10cm(state, words)
            runtime.write_case_output(prepared, tmp_path, start+timedelta(seconds=step*12),
                start_time=start, title="field", expect_refl_10cm=True)
            assert driver.refl_10cm is None
    assert (written == [] and [row["refl_field"].tobytes() for row in frames] == [words.tobytes()] * 2
            if captured else written == [words.tobytes()] * 2)


def test_nested_member_scope_restores_parent_capture():
    first = MemberOutputCapture(lambda **kw: None, 0)
    second = MemberOutputCapture(lambda **kw: None, 1)
    with member_output_scope(first):
        with member_output_scope(second):
            assert current_capture() is second
        assert current_capture() is first
    assert current_capture() is None


def test_bootstrap_decline_preserves_original_runner_and_log():
    closes = []
    log = SimpleNamespace(enabled=False, close=lambda **kw: closes.append(kw))
    args = dict(inputs=object(), model=object(), node=object(),
                output_directory=Path("out"), observer=object(), step_log=log)
    assert initialized_bootstrap_handoff(None, **args) is None
    assert initialized_bootstrap_handoff(lambda **kw: None, **args) is None
    assert closes == []


def test_bootstrap_success_preserves_exact_initialized_owners_and_closes_log():
    closes, seen = [], []
    step = object()
    log = SimpleNamespace(enabled=True, step_observer=step,
                          close=lambda **kw: closes.append(kw))
    args = dict(inputs=object(), model=object(), node=object(),
                output_directory=Path("out"), observer=object(), step_log=log)
    def callback(**kw):
        seen.append(kw)
        return {"status": "PASS", "completed_seconds": 120}
    result = initialized_bootstrap_handoff(callback, **args)
    assert result["status"] == "PASS"
    assert seen[0]["node"] is args["node"]
    assert seen[0]["model"] is args["model"]
    assert seen[0]["step_observer"] is step
    assert closes == [{"status": "SUCCESS"}]


def test_bootstrap_failure_is_terminal_without_ordinary_retry():
    closes = []
    log = SimpleNamespace(enabled=False, close=lambda **kw: closes.append(kw))
    def fail(**kw):
        raise RuntimeError("bad member state")
    with pytest.raises(RuntimeError, match="bad member state"):
        initialized_bootstrap_handoff(fail, inputs=object(), model=object(), node=object(),
            output_directory=Path("out"), observer=None, step_log=log)
    assert closes == [{"status": "FAIL", "error": "RuntimeError: bad member state"}]


def test_counter_context_observes_only_original_deadlines_without_extra_frames(monkeypatch):
    import numpy as np
    from gpuwm.core.clock import DomainClock, DomainTicks
    from gpuwm import runtime
    calls, metadata_reads = [], []
    class Collector:
        def submit(self, **kw):
            pytest.fail("counter capture must not invent a history frame")
        def capture_rain_counters(self, fields, **kw):
            calls.append((fields, kw))
            return True
    spec = DomainTicks(1, 0, 1, 60, np.float32(60), 7200,
        None, None, None, None, None, None, None, start_ticks=0,
        history_begin_ticks=7200, history_end_ticks=7200)
    clock = DomainClock(spec, 1, 7200)
    cfg = SimpleNamespace(nx=4, ny=3, mp_physics=8, cu_physics=0)
    rain = np.ones((3, 4), np.float32)
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=1, run=cfg), clock=clock,
        state=SimpleNamespace(physics=SimpleNamespace(microphysics=SimpleNamespace(rainnc=rain), rainc=None)),
        grid=object(), _started=True)
    model = SimpleNamespace(walk_parent_first=lambda: (node,),
        _prepared_by_grid_id={1: SimpleNamespace(static_fields=object())},
        _io_manager=SimpleNamespace(_episode_by_grid_id={1: 2}))
    def metadata(*args):
        metadata_reads.append(args)
        return {"XLAT": np.zeros((3, 4), np.float32), "XLONG": np.zeros((3, 4), np.float32)}
    monkeypatch.setattr(runtime, "_metadata_frame", metadata)
    collector = Collector()
    capture = MemberOutputCapture(collector.submit, 7)
    before = vars(clock.spec).copy()
    capture.observe_model_counters(model, start_time=datetime(2024, 1, 1))
    clock.ticks = 60
    capture.observe_model_counters(model, start_time=datetime(2024, 1, 1))
    clock.ticks = 3601
    capture.observe_model_counters(model, start_time=datetime(2024, 1, 1))
    clock.ticks = 7200
    capture.observe_model_counters(model, start_time=datetime(2024, 1, 1))
    assert len(calls) == len(metadata_reads) == 2
    assert calls[0][0]["RAINNC"] is rain
    assert all(kw["episode"] == 2 and kw["member_id"] == 7 for fields, kw in calls)
    calendar = next(iter(capture.counter_calendars.values()))
    assert calendar.receipt(member_id=7, episode=2)["unavailable_ticks"][0]["ticks"] == 3600
    assert vars(clock.spec) == before
