"""A run-plan or hosted-go forecast's writes and final beats reach its heartbeat.

The runtime announces a write between two steps and each beat after the
last step only to a progress object that has the ``writing``, ``written``
and ``finalizing`` hooks.  ``RunObserver`` and ``_GoObserver`` forwarded
``preparing`` and not those three, so ``run-progress.json`` said
"integrating" through a long frame write and the checkpoint read-back, and
the write was timed as a model step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpuwm import runtime, supervisor
from gpuwm.runplan import EVENTS_FILENAME, EventStream, RunObserver, _GoObserver


def _step(observer, step: int) -> None:
    observer(model_elapsed_seconds=60. * step, outer_step=step,
             last_durable_wrfout=None, last_checkpoint=None)


@pytest.mark.parametrize("hosted", [False, True],
                         ids=["run-plan", "hosted-go"])
def test_writes_and_final_beats_reach_the_heartbeat(tmp_path, hosted):
    path = tmp_path / supervisor.HEARTBEAT_NAME
    beat = supervisor.RuntimeHeartbeat(
        path, run_id="run", config_sha256="0" * 64,
        started_at_utc="2026-09-28T00:00:00Z")
    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events, heartbeat=beat)
        progress = _GoObserver(observer) if hosted else observer
        _step(progress, 1)
        with supervisor.writing_progress(progress, "history-d01",
                                         work_bytes=5 * 1024 ** 3):
            record = supervisor.read_heartbeat(path)
            assert (record.status, record.work_bytes) == (
                "writing:history-d01", 5 * 1024 ** 3)
        # The write hands back the status it interrupted.
        record = supervisor.read_heartbeat(path)
        assert (record.status, record.outer_step) == ("integrating", 1)

        runtime._finalizing_progress(progress, "verify-inputs",
                                     work_bytes=2048)
        record = supervisor.read_heartbeat(path)
        assert (record.status, record.work_bytes) == (
            "finalizing:verify-inputs", 2048)
        runtime._finalizing_progress(progress, "write-receipts")
        record = supervisor.read_heartbeat(path)
        assert (record.status, record.work_bytes) == (
            "finalizing:write-receipts", None)


def test_an_observer_without_the_hooks_is_left_alone(tmp_path):
    """`gpuwm downscale` wraps its child's progress object, which has none."""

    stage_only = SimpleNamespace(preparing=lambda phase: None)
    chain = _GoObserver(stage_only)
    with supervisor.writing_progress(chain, "checkpoint", work_bytes=1):
        pass
    runtime._finalizing_progress(chain, "synchronize-device")

    # And a run-plan observer with no heartbeat publishes nothing.
    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        with supervisor.writing_progress(observer, "checkpoint", work_bytes=1):
            pass
        runtime._finalizing_progress(observer, "synchronize-device")
    assert not (tmp_path / supervisor.HEARTBEAT_NAME).exists()
