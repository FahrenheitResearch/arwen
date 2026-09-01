"""The ledger is the cycle's control plane; crash recovery is a replay.

``settle()`` is the ONLY supported reader.  These tests hold it to the
three-phase commit: a staged write left behind by a crash is rolled
forward, not silently dropped and not silently double-counted.
"""

import json

import pytest

from gpuwm.cycle.contracts import CycleRefusal, LEDGER_SCHEMA
from gpuwm.cycle.ledger import CycleLedger


def test_append_stamps_schema_seq_and_time(tmp_path):
    ledger = CycleLedger(tmp_path)
    first = ledger.append({"event": "cycle-started", "cycle": 0})
    second = ledger.append({"event": "cycle-completed", "cycle": 0})
    assert first["schema"] == LEDGER_SCHEMA
    assert (first["seq"], second["seq"]) == (0, 1)
    assert first["written_at"].endswith("+00:00")
    assert [r["event"] for r in ledger.settle()] == ["cycle-started",
                                                     "cycle-completed"]


def test_append_refuses_a_record_without_event_and_cycle(tmp_path):
    ledger = CycleLedger(tmp_path)
    with pytest.raises(CycleRefusal) as excinfo:
        ledger.append({"cycle": 0})
    assert "event" in excinfo.value.observed["missing_keys"]
    assert ledger.path.exists() is False


def test_settle_recovers_a_staged_write(tmp_path):
    ledger = CycleLedger(tmp_path)
    ledger.append({"event": "cycle-started", "cycle": 0})
    # Hand-write the staged file a crash would have left behind: phase
    # two (the append) never happened.
    staged = {"schema": LEDGER_SCHEMA, "seq": 1, "event": "cycle-completed",
              "cycle": 0, "written_at": "2026-08-14T18:00:00+00:00"}
    ledger.staged_path.write_text(json.dumps(staged) + "\n", encoding="utf-8")
    records = ledger.settle()
    assert [r["event"] for r in records] == ["cycle-started",
                                             "cycle-completed"]
    assert ledger.staged_path.exists() is False
    # Settling twice must not double-count the recovered record.
    assert len(ledger.settle()) == 2


def test_settle_drops_a_staged_write_already_in_the_log(tmp_path):
    ledger = CycleLedger(tmp_path)
    record = ledger.append({"event": "cycle-completed", "cycle": 0})
    ledger.staged_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert len(ledger.settle()) == 1


def test_state_folds_forecast_only_streak(tmp_path):
    ledger = CycleLedger(tmp_path)
    for cycle in range(3):
        ledger.append({"event": "analysis", "cycle": cycle,
                       "state": "SKIPPED_NO_OBS"})
        ledger.append({"event": "cycle-completed", "cycle": cycle})
    assert ledger.state()["forecast_only_streak"] == 3
    ledger.append({"event": "analysis", "cycle": 3, "state": "APPLIED"})
    ledger.append({"event": "cycle-completed", "cycle": 3})
    state = ledger.state()
    assert state["forecast_only_streak"] == 0
    assert state["last_completed_cycle"] == 3
    assert state["halted"] is False


def test_state_folds_children_and_halt(tmp_path):
    ledger = CycleLedger(tmp_path)
    ledger.append({"event": "child", "cycle": 0, "grid_id": 2,
                   "state": "LIVE"})
    ledger.append({"event": "child", "cycle": 0, "grid_id": 3,
                   "state": "LIVE"})
    ledger.append({"event": "child", "cycle": 1, "grid_id": 3,
                   "state": "DIVERGED"})
    state = ledger.state()
    assert state["live_children"] == {"2": "LIVE", "3": "DIVERGED"}
    with pytest.raises(CycleRefusal):
        ledger.halt(cycle=1, reason="PARENT_DIVERGED", observed_field="w",
                    observed_value=1e30)
    state = ledger.state()
    assert state["halted"] is True
    assert state["halt_reason"] == "PARENT_DIVERGED"


def test_halt_requires_a_known_reason(tmp_path):
    ledger = CycleLedger(tmp_path)
    with pytest.raises(CycleRefusal) as excinfo:
        ledger.halt(cycle=0, reason="BECAUSE_I_SAID_SO", detail="none")
    assert excinfo.value.observed["reason"] == "BECAUSE_I_SAID_SO"
    # Nothing was appended: an unknown reason is refused before the log
    # is touched, so a typo cannot poison a lineage.
    assert ledger.path.exists() is False
    assert ledger.state()["halted"] is False
