"""Early launch orchestration must expose packing and cold-start failures."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import time

import pytest

from gpuwm.ensemble import canary
from test_clock import _chain_experiment


@dataclass(frozen=True)
class Inputs:
    experiment: object
    authority: object


def test_canary_checks_every_source_without_inventing_a_scientific_roster():
    rows = [{"source": "source-a", "prepared_root": "a0"},
            {"source": "source-a", "prepared_root": "a1"},
            {"source": "source-b", "prepared_root": "b0"}]
    assert canary.representatives(rows) == (rows[0], rows[2])
    with pytest.raises(canary.CanaryRefusal, match="own cold-start check"):
        canary.representatives([{"prepared_root": "unknown"}])


def test_canary_preserves_physics_authorities_and_checks_50_root_steps():
    exp = _chain_experiment((1, 3), run_seconds=7200)
    original = Inputs(exp, object())
    limited, seconds = canary._limit(original, 50)
    assert seconds == float(exp.dt_exact(exp.root.grid_id) * 50)
    assert limited.authority is original.authority
    assert exp.run_seconds == 7200
    for before, after in zip(exp.domains, limited.experiment.domains):
        assert replace(after.run, run_seconds=before.run.run_seconds) == before.run
        assert after.history_interval_s == seconds


def test_adaptive_or_short_window_cannot_falsely_certify_50_steps():
    exp = _chain_experiment((1,), run_seconds=7200)
    short = replace(exp, run_seconds=1)
    with pytest.raises(canary.CanaryRefusal, match="forcing cannot support"):
        canary._limit(Inputs(short, None), 50)
    domain = replace(exp.root, run=replace(exp.root.run, use_adaptive_time_step=True))
    adaptive = replace(exp, domains=(domain,))
    with pytest.raises(canary.CanaryRefusal, match="adaptive step count"):
        canary._limit(Inputs(adaptive, None), 50)


def test_canary_failure_stops_remaining_preparation_before_full_roster(tmp_path, monkeypatch):
    def refusal(*args, **kwargs):
        raise canary.CanaryRefusal("member packing mapping must name every sampled card exactly")
    monkeypatch.setattr(canary, "run_canary", refusal)
    manager = canary.HeadCanaries(tmp_path, None)
    manager.submit({"source": "source-a", "prepared_root": "head"})
    manager._futures[0].exception(timeout=5)
    with pytest.raises(canary.CanaryRefusal, match="sampled card exactly"):
        manager.check()
    manager.close(raise_error=False)


def test_source_checks_are_serialized_and_extend_the_packed_probe(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(canary, "run_canary", lambda rows, **kwargs: calls.append(rows))
    manager = canary.HeadCanaries(tmp_path, None)
    manager.submit({"source": "source-a", "prepared_root": "a"})
    manager.submit({"source": "source-a", "prepared_root": "another-a"})
    manager.submit({"source": "source-b", "prepared_root": "b"})
    manager.close()
    assert [tuple(row["source"] for row in rows) for rows in calls] == [("source-a",), ("source-a", "source-b")]


def test_canary_cannot_check_a_different_spatial_device_graph(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(canary, "run_canary", lambda rows, **kwargs: calls.append(rows))
    manager = canary.HeadCanaries(tmp_path, None, devices=2)
    manager.submit({"source": "source-a", "prepared_root": "head"})
    manager.close()
    assert calls[0][0]["devices"] == 2


def test_retry_cannot_launch_canary_from_previous_attempt_head(tmp_path, monkeypatch):
    from gpuwm import stage_cli
    calls = []
    monkeypatch.setattr(canary, "run_canary", lambda rows, **kwargs: calls.append(rows))
    monkeypatch.setattr(stage_cli, "_schema_index", lambda: {"prepared-fixture": {}})
    head_path = tmp_path / "prepare/tree/boundary-stream/head.json"
    head_path.parent.mkdir(parents=True)
    head = {"created_utc": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "decision": {"chained": True}, "basis": {"proof_head": {"schema": "prepared-fixture"}},
            "head_sha256": "a" * 64}
    head_path.write_text(json.dumps(head))
    manager = canary.HeadCanaries(tmp_path / "canaries", None)
    check, stop = manager.watch(tmp_path, source="source-a", config="config", namelist="namelist")
    try:
        time.sleep(.3)
        assert not calls
        head["created_utc"] = datetime.now(timezone.utc).isoformat()
        partial = head_path.with_suffix(".pending")
        partial.write_text(json.dumps(head))
        partial.replace(head_path)
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            check()
            time.sleep(.02)
        assert calls[0][0]["head_sha256"] == "a" * 64
        assert calls[0][0]["prepared_root"] == str(head_path.parent.parent)
    finally:
        stop()
        manager.close()
