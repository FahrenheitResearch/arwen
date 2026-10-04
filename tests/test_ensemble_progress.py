from pathlib import Path

import pytest

from gpuwm.ensemble.progress import EnsembleProgressAdapter
from gpuwm.supervisor import RuntimeHeartbeat, read_heartbeat, _heartbeat_regression


def heartbeat(tmp_path):
    return RuntimeHeartbeat(tmp_path / "heartbeat.json", run_id="ensemble",
        config_sha256="a" * 64, started_at_utc="2026-10-02T00:00:00Z")


def test_independent_member_resets_are_monotonic_supervised_work(tmp_path):
    parent = heartbeat(tmp_path)
    adapter = EnsembleProgressAdapter(parent, member_ids=(7, 20), run_seconds=24.)
    previous = None
    for member in (7, 20):
        callback = adapter.callback_for_member(member)
        callback.preparing("prepare-case")
        for step, elapsed in enumerate((0., 12., 24.)):
            callback(model_elapsed_seconds=elapsed, outer_step=step)
            current = read_heartbeat(parent.path)
            if previous is not None:
                assert _heartbeat_regression(previous, current) is None
            previous = current
    assert previous.model_elapsed_seconds == 24. and previous.outer_step == 4
    assert [row["member_id"] for row in adapter.receipt()["members"]] == [7, 20]


def test_native_packs_update_only_their_original_member_ids():
    events = []
    adapter = EnsembleProgressAdapter(lambda **event: events.append(event),
        member_ids=(10, 4, 2, 9), run_seconds=60.)
    adapter.callback_for_member(10)(member_ids=(10, 4), model_elapsed_seconds=60., outer_step=5)
    adapter.callback_for_member(2)(member_ids=(2, 9), model_elapsed_seconds=30., outer_step=3)
    assert [event["model_elapsed_seconds"] for event in events] == [30., 45.]
    assert events[-1]["outer_step"] == 16
    assert events[-1]["member_outer_step"] == 3
    assert events[-1]["member_model_elapsed_seconds"] == 30.
    with pytest.raises(ValueError, match="outside"):
        adapter.callback_for_member(10)(member_ids=(10, 99), model_elapsed_seconds=60., outer_step=6)
    assert adapter.receipt()["members"][0]["outer_step"] == 5


def test_partial_member_checkpoint_cannot_be_an_ensemble_restart():
    events = []
    adapter = EnsembleProgressAdapter(lambda **event: events.append(event), member_ids=(0, 1), run_seconds=24.)
    adapter.callback_for_member(0)(model_elapsed_seconds=12., outer_step=1,
        last_checkpoint=Path("member-0/checkpoint.npz"))
    assert events[0]["last_checkpoint"] is None
    assert adapter.receipt()["members"][0]["last_checkpoint"] == str(Path("member-0/checkpoint.npz"))


def test_single_member_retains_the_original_restart_event():
    events = []
    adapter = EnsembleProgressAdapter(lambda **event: events.append(event), member_ids=(73,), run_seconds=24.)
    adapter.callback_for_member(73)(model_elapsed_seconds=12., outer_step=1, last_checkpoint="checkpoint.npz")
    assert events[0]["last_checkpoint"] == "checkpoint.npz"


def test_member_completion_does_not_publish_global_terminal_success(tmp_path):
    parent = heartbeat(tmp_path)
    adapter = EnsembleProgressAdapter(parent, member_ids=(0, 1), run_seconds=24.)
    adapter.callback_for_member(0).complete(24.)
    assert read_heartbeat(parent.path).status == "integrating"
    adapter.callback_for_member(1).complete(24.)
    assert read_heartbeat(parent.path).status == "integrating"


def test_actual_source_restart_keeps_the_original_restart_protocol(tmp_path):
    parent = heartbeat(tmp_path)
    adapter = EnsembleProgressAdapter(parent, member_ids=(0, 1), run_seconds=24.)
    callback = adapter.callback_for_member(0)
    callback(model_elapsed_seconds=24., outer_step=2)
    previous = read_heartbeat(parent.path)
    with pytest.raises(ValueError, match="declared restart"):
        callback(model_elapsed_seconds=0., outer_step=0)
    callback.restarting("source interval moved")
    callback(model_elapsed_seconds=0., outer_step=0)
    current = read_heartbeat(parent.path)
    assert _heartbeat_regression(previous, current) is None
    assert current.restart["attempt"] == 2
