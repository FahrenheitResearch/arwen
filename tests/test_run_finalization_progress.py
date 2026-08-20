"""The stretch between the last model step and rc 0 must be visible.

A four-domain 24 h run integrated every one of its 345,600 ticks, wrote
all 484 history frames and its final checkpoints, and then spent minutes
draining writers and SHA-256ing several hundred GiB of those frames into
its success capsule -- publishing nothing the whole time.  The
supervisor's integration watchdog cannot tell that from a hang, so it
killed the worker, restarted from the final checkpoint, and the restore
refused because the checkpoint WAS the stop tick.  A scientifically
complete run reported failure.

These gates hold the finalization stretch to the same rule the
integration loop already lives by: say what you are doing, per unit of
work, on the record the supervisor reads.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import gpuwm.runtime as runtime
import gpuwm.supervisor as supervisor


class _Recorder:
    """The optional-hook surface a progress callback exposes."""

    def __init__(self):
        self.phases: list[str] = []

    def finalizing(self, phase: str) -> None:
        self.phases.append(phase)


def _frame(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_frame_hashing_publishes_one_beat_per_frame(tmp_path):
    """The capsule's frame hash is the longest silence in the run.

    Every frame read is one beat, because the frames are what the wall
    clock is actually being spent on: one 250 m domain's hourly set is
    hundreds of GiB, and a run whose hashing pass outlasts the stale
    threshold was killed mid-receipt.
    """
    paths = [_frame(tmp_path / f"wrfout_d01_{index:02d}", b"x" * (index + 1))
             for index in range(4)]
    recorder = _Recorder()

    records = runtime._frame_records(paths, progress_callback=recorder)

    assert [record["path"] for record in records] == [
        str(path.resolve()) for path in paths]
    assert len(recorder.phases) == len(paths)
    assert all(phase.startswith("hash-output-frames")
               for phase in recorder.phases)
    # The beat carries progress, not just a repeated word: a user
    # watching the heartbeat can tell 3-of-484 from 400-of-484.
    assert recorder.phases[0].endswith("1-of-4")
    assert recorder.phases[-1].endswith("4-of-4")


def test_frame_hashing_without_a_progress_callback_is_unchanged(tmp_path):
    paths = [_frame(tmp_path / "wrfout_d01_00", b"bytes")]
    assert (runtime._frame_records(paths)
            == runtime._frame_records(paths, progress_callback=None))


def test_finalizing_progress_is_an_optional_hook():
    """Back-compat: a callback without the hook is not an error."""
    recorder = _Recorder()
    runtime._finalizing_progress(recorder, "drain-history-writers")
    assert recorder.phases == ["drain-history-writers"]
    runtime._finalizing_progress(None, "drain-history-writers")
    runtime._finalizing_progress(object(), "drain-history-writers")


def test_the_run_summary_carries_the_frame_records_it_hashed():
    summary = runtime.ExperimentRunSummary(
        wrfout_paths=(), completed_seconds=1.0, nan_free=True)
    assert summary.frame_records == ()


def test_the_success_capsule_reuses_the_records_the_run_already_hashed():
    """One hashing pass, not two.

    ``run_experiment`` hashes every frame for the front-door capsule and
    the worker then hashed every frame AGAIN for the success capsule, so
    the finalization cost -- and the silent window -- was paid twice over
    the same bytes.
    """
    records = ({"path": "/gone/wrfout_d01_00", "bytes": 5,
                "sha256": "0" * 64},)
    summary = SimpleNamespace(
        wrfout_paths=(Path("/gone/wrfout_d01_00"),),
        frame_records=records, trajectory_digest={"d01": "abc"})

    output = supervisor._success_output(summary)

    # The paths do not exist: a second hashing pass would raise here.
    assert output["frames"] == list(records)
    assert output["trajectory_digest"] == {"d01": "abc"}


def test_the_success_capsule_still_hashes_when_no_records_were_handed_over(
        tmp_path):
    """CONTROL: the capsule never silently ships an empty frame list."""
    path = _frame(tmp_path / "wrfout_d01_00", b"bytes")
    summary = SimpleNamespace(
        wrfout_paths=(path,), frame_records=(), trajectory_digest=None)

    output = supervisor._success_output(summary)

    assert len(output["frames"]) == 1
    assert output["frames"][0]["path"] == str(path.resolve())


def test_a_single_domain_restore_at_the_stop_is_complete_not_an_error():
    """The one-domain route owns the same ruling as the tree route.

    Found by the artifact, not by reading: the real CLI restarted from a
    one-domain run's own final checkpoint and died with a BARE
    ``ValueError: restart file is already at 100.0 s`` -- a different
    line from the tree route's refusal, the same defect, and the same
    supervisor restart loop on top of it.
    """
    assert runtime._resumed_start_step(
        elapsed_seconds=100.0, dt=50.0, outer_steps=2,
        run_seconds=100.0) == 2
    assert runtime._resumed_start_step(
        elapsed_seconds=50.0, dt=50.0, outer_steps=2,
        run_seconds=100.0) == 1

    with pytest.raises(ValueError) as caught:
        runtime._resumed_start_step(
            elapsed_seconds=150.0, dt=50.0, outer_steps=2,
            run_seconds=100.0)
    message = str(caught.value)
    assert "150" in message and "100" in message
    assert "run_seconds" in message


def test_a_run_restored_at_its_stop_tick_does_not_integrate(monkeypatch):
    """Leg 1 at the route: completion finalizes, it does not step.

    ``execute_schedule`` refuses a start period at or past the end of the
    schedule, so a restore at the stop tick that fell through to the
    integration loop would trade a named refusal for a bare ValueError.
    """
    assert runtime._restart_is_complete(
        SimpleNamespace(already_complete=True)) is True
    assert runtime._restart_is_complete(
        SimpleNamespace(already_complete=False)) is False
    assert runtime._restart_is_complete(None) is False
