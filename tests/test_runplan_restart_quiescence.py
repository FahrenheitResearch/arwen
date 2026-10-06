"""A hosted retry must prove readers stopped before it can rewind history."""
from types import SimpleNamespace

import pytest

from gpuwm.runplan import EventStream, RunObserver, _GoObserver
from gpuwm.supervisor import restart_attempt


class _RenderReaders:
    def __init__(self, *, ended):
        self.halts = []
        self.waits = []
        self.ended = ended

    def halt(self, timeout):
        self.halts.append(timeout)

    def render_threads(self):
        return [(SimpleNamespace(ident=None), self)]

    def wait(self, timeout):
        self.waits.append(timeout)
        return self.ended


@pytest.mark.parametrize("through_go", [False, True])
@pytest.mark.parametrize("stranded", ["first", "live"])
def test_restart_refuses_when_actual_halt_helper_reports_a_reader_left(
        tmp_path, through_go, stranded):
    """Exercise the real halt helper's False result, rather than mock it."""
    restarted, rearmed, warnings = [], [], []
    observer = RunObserver(EventStream(tmp_path / "events.jsonl", mirror=None),
        heartbeat=SimpleNamespace(restarting=restarted.append))
    first = _RenderReaders(ended=stranded != "first")
    live = _RenderReaders(ended=stranded != "live")
    observer._first_products, observer._live_products = first, live
    observer._render_plan = {"run": tmp_path}
    observer.arm_first_products = rearmed.append
    observer.warn = lambda *args, **kwargs: warnings.append((args, kwargs))
    observer._last_model_seconds = 7200.0
    observer._committed = 3

    with pytest.raises(RuntimeError, match="still.*reading|quiescen"):
        restart_attempt(_GoObserver(observer) if through_go else observer,
                        "retry from the last hourly checkpoint")

    assert first.halts and live.halts
    assert observer.first_products is first
    assert observer.live_products is live
    assert rearmed == [] and restarted == [] and warnings == []
    assert observer.last_model_seconds == 7200.0
    assert observer.outputs_committed == 3


def test_restart_rearms_only_after_both_readers_quiesce(tmp_path):
    restarted, rearmed = [], []
    observer = RunObserver(EventStream(tmp_path / "events.jsonl", mirror=None),
        heartbeat=SimpleNamespace(restarting=restarted.append))
    first, live = _RenderReaders(ended=True), _RenderReaders(ended=True)
    observer._first_products, observer._live_products = first, live
    observer._render_plan = {"run": tmp_path}

    def rearm(plan):
        assert first.waits and live.waits
        assert observer.first_products is None and observer.live_products is None
        rearmed.append(plan)

    observer.arm_first_products = rearm
    restart_attempt(observer, "a proven checkpoint retry")
    assert restarted == ["a proven checkpoint retry"]
    assert rearmed == [observer._render_plan]


def test_health_recovery_keeps_history_when_hosted_reader_survives(
        monkeypatch, tmp_path):
    """A valid tree restore still cannot authorize deletion under a reader."""
    from gpuwm.core.health import HealthCheckError
    from gpuwm.stability_recovery import NestedHealthRecovery
    from test_stability_recovery import _checkpoint, _health_error

    model, experiment, _ = _checkpoint(monkeypatch, tmp_path)
    observer = RunObserver(EventStream(tmp_path / "events.jsonl", mirror=None))
    observer._first_products = _RenderReaders(ended=False)
    rewinds = []
    recovery = NestedHealthRecovery(model=model, experiment=experiment,
        output_directory=tmp_path, observer=_GoObserver(observer),
        writers=SimpleNamespace(drain=lambda: None,
            rewind_to_checkpoint=lambda *args, **kwargs:
                                rewinds.append((args, kwargs))))

    def failed_leg(_active):
        model.root.clock.ticks = 3700
        model.root.clock.elapsed_seconds = 3700
        raise _health_error()

    with pytest.raises(HealthCheckError):
        recovery.run(failed_leg)
    assert rewinds == []
    assert recovery.receipt["status"] == "REFUSED"
    assert "still reading" in recovery.receipt["refusal"]
