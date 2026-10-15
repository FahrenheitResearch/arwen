"""Suite controller failures, tested without a model or GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from gpuwm.nwpsuite.ledger import SuiteLedger
from gpuwm.nwpsuite.recovery import next_action, replan, lost_box, disk_victim
from gpuwm.nwpsuite.retention import register, prune, receipt_bytes
from gpuwm.nwpsuite.run import SuiteRunner, apply_plan, load_plan
from gpuwm.nwpsuite.status import status_document

CYCLE = "2026-10-03T12:00:00Z"


def plan(tmp_path, *, seconds=0, code=0, extra=None):
    door = tmp_path / "door.py"
    door.write_text("import pathlib,sys,time,json\n"
        "p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)\n"
        "with p.open('a') as f: f.write('launch\\n')\n"
        f"time.sleep({seconds!r})\n"
        f"sys.exit({code!r})\n")
    row = dict(id="forecast", box="local", cards=0, priority=1, every="6h",
               argv=[sys.executable, str(door), str(tmp_path / "launches")])
    row.update(extra or {})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(schema="gpuwm.suite.runtime-fixture.v1", admitted=True,
        rows=[row], boxes=[dict(id="local", cards=3)], retention={})))
    return load_plan(path)


def test_settle_staged_and_duplicate(tmp_path, monkeypatch):
    ledger = SuiteLedger(tmp_path)
    def crash(line):
        raise RuntimeError("kill between stage and append")
    original = ledger._commit
    monkeypatch.setattr(ledger, "_commit", crash)
    with pytest.raises(RuntimeError):
        ledger.append("phase", row="a", cycle=CYCLE, suite_hash="hash", phase="ready")
    monkeypatch.setattr(ledger, "_commit", original)
    assert len(ledger.settle()) == 1
    ledger.staged.write_text(json.dumps(ledger.settle()[0]))
    assert len(ledger.settle()) == 1


def test_settle_torn_append(tmp_path, monkeypatch):
    ledger = SuiteLedger(tmp_path)
    original = ledger._commit
    def crash(line):
        with ledger.path.open("ab") as stream:
            stream.write(line[:30])
        raise RuntimeError("torn append")
    monkeypatch.setattr(ledger, "_commit", crash)
    with pytest.raises(RuntimeError):
        ledger.append("phase", row="a", cycle=CYCLE, suite_hash="hash", phase="ready")
    monkeypatch.setattr(ledger, "_commit", original)
    assert ledger.settle()[0]["phase"] == "ready"


def test_compaction_survives_old_journal_and_preserves_pending(tmp_path):
    ledger = SuiteLedger(tmp_path)
    ledger.append("configuration", suite_hash="hash", plan="pinned.json")
    for hour in range(8):
        ledger.append("phase", row="a", cycle=f"2026-10-03T{hour:02d}:00:00Z", suite_hash="hash", phase="completed")
    ledger.append("phase", row="a", cycle=CYCLE, suite_hash="hash", phase="running", work="owned")
    old_journal = ledger.path.read_bytes()
    ledger.compact(max_bytes=1, keep_events=2)
    expected = ledger.state()
    assert expected["jobs"][("a", CYCLE)]["phase"] == "running"
    assert expected["watermarks"]["a"] == "2026-10-03T07:00:00Z"
    ledger.path.write_bytes(old_journal)  # crash after snapshot, before replacement
    assert ledger.state() == expected
    assert ledger.append("heartbeat")["seq"] == 10


def test_compacted_cycles_do_not_launch_again(tmp_path):
    fixture = plan(tmp_path)
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.run(cycles=[CYCLE], poll=.02)
    for i in range(5):
        runner.ledger.append("heartbeat", tick=i)
    runner.ledger.compact(max_bytes=1, keep_events=1)
    runner.run(cycles=[CYCLE], poll=.02)
    assert (tmp_path / "launches").read_text().splitlines() == ["launch"]


def test_second_scheduler_refuses_immediately_without_another_launch(tmp_path):
    fixture = plan(tmp_path, seconds=3)
    root = tmp_path / "state"
    command = [sys.executable, "-m", "gpuwm.cli", "suite", "run", fixture["suite_file"],
               "--state-dir", str(root), "--poll-seconds", ".02"]
    first = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        until = time.monotonic() + 10
        while time.monotonic() < until:
            if any(x.get("phase") == "running" for x in SuiteLedger(root).state()["jobs"].values()):
                break
            time.sleep(.02)
        else:
            pytest.fail("owned first scheduler did not reach running")
        second = subprocess.run([*command, "--once"], capture_output=True, text=True, timeout=5)
        assert second.returncode == 2 and "suite_scheduler_active" in second.stderr
        assert sum(e.get("phase") == "launch_intent" for e in SuiteLedger(root).settle()) == 1
    finally:
        first.terminate()
        first.wait(timeout=5)
        from gpuwm.nwpsuite.ledger import halt_row
        halt_row(root, "forecast", "owned test cleanup", fixture["suite_hash"])


@pytest.mark.parametrize("phase", ["planned", "ready", "launch_intent", "running", "receipted", "completed"])
def test_scheduler_killed_at_every_phase(tmp_path, phase):
    fixture = plan(tmp_path, seconds=0.2)
    root = tmp_path / "state"
    code = ("import os,json,sys; from gpuwm.nwpsuite.run import SuiteRunner; "
            "plan=json.load(open(sys.argv[1])); phase=sys.argv[3]; "
            "runner=SuiteRunner(plan,sys.argv[2],phase_hook=lambda p: os._exit(91) if p==phase else None); "
            f"runner.run(cycles=[{CYCLE!r}],poll=.02)")
    pinned = tmp_path / "pinned.json"
    pinned.write_text(json.dumps(fixture))
    process = subprocess.Popen([sys.executable, "-c", code, str(pinned), str(root), phase])
    assert process.wait(timeout=15) == 91
    runner = SuiteRunner(fixture, root)
    runner.run(cycles=[CYCLE], poll=.02)
    assert runner.ledger.state()["jobs"][("forecast", CYCLE)]["phase"] == "completed"
    assert (tmp_path / "launches").read_text().splitlines() == ["launch"]
    assert all(e["suite_hash"] == fixture["suite_hash"] for e in runner.ledger.settle() if e.get("cycle"))


@pytest.mark.parametrize("history,expected", [({}, "retry"), ({"retries": 3}, "checkpoint"),
    ({"retries": 3, "checkpoint_tried": True}, "degrade"),
    ({"retries": 3, "checkpoint_tried": True, "degrade_index": 0}, "halt")])
def test_crash_ladder(history, expected):
    assert next_action(dict(resume_argv=["door"], degrade=[{"argv": ["door"]}]),
                       {"exit_code": 1}, history, now=1, cutoff=10)["action"] == expected


def test_full_ladder_executes_each_door(tmp_path):
    fixture = plan(tmp_path, code=1)
    row = fixture["rows"][0]
    row["resume_argv"] = row["argv"]
    row["degrade"] = [{"argv": [sys.executable, "-c", "pass"]}]
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.run(cycles=[CYCLE], poll=.02)
    assert (tmp_path / "launches").read_text().splitlines() == ["launch"] * 5
    assert runner.ledger.state()["jobs"][("forecast", CYCLE)]["phase"] == "completed"
    actions = [x["decision"]["action"] for x in runner.ledger.settle() if x["event"] == "recovery"]
    assert actions == ["retry", "retry", "retry", "checkpoint", "degrade", "complete"]


def test_probe_card_loss_never_drops_priority_one(tmp_path):
    fixture = plan(tmp_path)
    fixture["rows"][0].update(cards=1)
    fixture["rows"].append(dict(fixture["rows"][0], id="optional", priority=3))
    fixture["boxes"][0]["health_argv"] = [sys.executable, "-c", 'print(\'{"alive":true,"cards":1}\')']
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner._health()
    assert "optional" in runner.ledger.state()["halted"]
    assert "forecast" not in runner.ledger.state()["halted"]


def test_probe_box_loss_requests_profile(tmp_path):
    fixture = plan(tmp_path)
    fixture["boxes"][0]["health_argv"] = [sys.executable, "-c", 'print(\'{"alive":false}\')']
    fixture["boxes"].append(dict(id="survivor", cards=3))
    fixture["single_box_profile"] = dict(fixture, suite_hash="survivor-hash", boxes=[fixture["boxes"][1]])
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner._health()
    assert (runner.root / "pending-plan.json").exists()
    assert any(x["event"] == "health_recovery" for x in runner.ledger.settle())


def test_refused_plan_cannot_run(tmp_path):
    path = tmp_path / "refused.json"
    path.write_text(json.dumps({"schema": "gpuwm.suite.plan.v1", "admitted": False, "rows": []}))
    with pytest.raises(ValueError, match="suite_not_admitted"):
        load_plan(path)


def test_actual_fresh_process_retries_and_child_isolation(tmp_path):
    fixture = plan(tmp_path, code=1)
    parent = dict(fixture["rows"][0], id="parent", argv=[sys.executable, "-c", "pass"])
    fixture["rows"].append(parent)
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.run(cycles=[CYCLE], poll=.02)
    state = runner.ledger.state()
    assert state["jobs"][("forecast", CYCLE)]["phase"] == "halted"
    assert state["jobs"][("parent", CYCLE)]["phase"] == "completed"
    assert len((tmp_path / "launches").read_text().splitlines()) == 4


def test_source_deadline_and_fallback():
    row = {"sources": ["first", "second"]}
    assert next_action(row, {"exit_code": 75, "readiness": True}, {}, now=9, cutoff=10)["action"] == "wait"
    assert next_action(row, {"exit_code": 75, "readiness": True}, {}, now=10, cutoff=10)["source_index"] == 1
    assert next_action(row, {"exit_code": 75}, {"source_index": 1}, now=10, cutoff=10)["reason"] == "source_behind"


def test_da_failure_public_analysis():
    result = next_action({"kind": "da"}, {"exit_code": 1}, {}, now=0, cutoff=2)
    assert result == {"action": "skip", "reason": "da_failed", "initialization": "public_analysis"}


def test_da_door_failure_releases_public_forecast(tmp_path):
    fixture = plan(tmp_path, code=1)
    da = dict(fixture["rows"][0], id="analysis", kind="da")
    forecast = dict(fixture["rows"][0], argv=[sys.executable, "-c", "pass"], dependencies=["analysis"])
    fixture["rows"] = [da, forecast]
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.run(cycles=[CYCLE], poll=.02)
    state = runner.ledger.state()["jobs"]
    assert state[("analysis", CYCLE)]["initialization"] == "public_analysis"
    assert state[("forecast", CYCLE)]["phase"] == "completed"
    assert (tmp_path / "launches").read_text().splitlines() == ["launch"]


def test_late_source_door_is_bounded_and_dependents_skip(tmp_path):
    fixture = plan(tmp_path, code=75, extra={"sources": ["one", "two"]})
    fixture["rows"].append(dict(fixture["rows"][0], id="products", dependencies=["forecast"],
                               argv=[sys.executable, "-c", "raise RuntimeError('must not launch')"]))
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.run(cycles=[CYCLE], poll=.02)
    state = runner.ledger.state()["jobs"]
    assert state[("forecast", CYCLE)]["reason"] == "source_behind"
    assert state[("products", CYCLE)]["reason"] == "dependency_skipped"
    assert len((tmp_path / "launches").read_text().splitlines()) == 2


def test_disk_pressure_prunes_then_halts_low_priority(tmp_path, monkeypatch):
    fixture = plan(tmp_path)
    fixture["rows"].append(dict(fixture["rows"][0], id="optional", priority=3))
    fixture["retention"] = {"history": "0s", "disk_floor_gb": 1}
    runner = SuiteRunner(fixture, tmp_path / "state")
    artifact = runner.root / "old-history"
    artifact.write_bytes(b"data")
    register(runner.root, artifact, "history", row="optional", cycle=CYCLE)
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100, 100, 0))
    runner.tick([CYCLE])
    assert not artifact.exists()
    assert "optional" in runner.ledger.state()["halted"]
    assert "forecast" not in runner.ledger.state()["halted"]


def test_status_score_delta_and_alert_once(tmp_path):
    from gpuwm.nwpsuite.status import score_delta, publish_status
    assert score_delta([{"rmse": 5}], [{"score": {"rmse": 2}}, {"score": {"rmse": 4}}]) == {"/rmse": 2}
    fixture = plan(tmp_path)
    log = tmp_path / "hook-count"
    fixture["on_state_change"] = [sys.executable, "-c", f"open({str(log)!r},'a').write('hook\\n')"]
    runner = SuiteRunner(fixture, tmp_path / "state")
    publish_status(fixture, runner.root)
    publish_status(fixture, runner.root)
    assert log.read_text().splitlines() == ["hook"]


def test_card_lost_priority_one():
    rows = [dict(id="keep", cards=2, priority=1), dict(id="drop", cards=2, priority=3)]
    assert replan(rows, 2)["drop"] == ["drop"]
    assert replan(rows, 1)["halt"] == "priority_one_capacity"
    assert replan([dict(id="a", cards=2, priority=1), dict(id="b", cards=2, priority=1)], 2)["keep"] == ["a", "b"]


def test_active_receipt_cap_stops_owned_door(tmp_path):
    fixture = plan(tmp_path)
    fixture["retention"] = {"receipt_cap": "10KB"}
    fixture["rows"][0]["argv"] = [sys.executable, "-c",
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'large.json').write_text('x'*20000); time.sleep(20)", "{out}"]
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.run(cycles=[CYCLE], poll=.02)
    entry = runner.ledger.state()["jobs"][("forecast", CYCLE)]
    assert entry["phase"] == "halted"
    assert entry["result"]["error_class"] == "receipt_cap_exceeded"


def test_box_lost_profile():
    assert lost_box({"single_box_profile": {"rows": []}}, "survivor")["action"] == "profile"
    assert lost_box({}, "survivor")["halt"] == "single_box_profile_missing"


def test_retention_cap_latest_and_protection(tmp_path):
    for name, kind, cycle in [("r1", "receipts", CYCLE), ("r2", "receipts", CYCLE),
                             ("c1", "checkpoints", "2026-10-03T06:00:00Z"), ("c2", "checkpoints", CYCLE)]:
        path = tmp_path / name
        path.write_bytes(b"x" * 100)
        register(tmp_path, path, kind, row="row", cycle=cycle)
    cap = receipt_bytes(tmp_path, json.loads((tmp_path / "artifacts.json").read_text())) - 100
    result = prune(tmp_path, {"receipt_cap": f"{cap}B", "checkpoints": "latest"})
    assert result["receipt_bytes"] <= cap
    assert {Path(x["path"]).name for x in result["deleted"]} == {"r1", "c1"}
    assert disk_victim([dict(id="one", priority=1), dict(id="low", priority=3)]) == "low"


def test_retention_counts_control_and_preserves_active_anchors(tmp_path):
    old = tmp_path / "old.json"
    old.write_bytes(b"x" * 20000)
    register(tmp_path, old, "receipts", row="old", cycle=CYCLE)
    active = tmp_path / "active"
    active.mkdir()
    anchor = active / "intent.json"
    anchor.write_bytes(b"x" * 10000)
    protected = tmp_path / "protected.log"
    protected.write_bytes(b"x" * 10000)
    register(tmp_path, protected, "receipts", row="old", cycle=CYCLE, protected=True)
    control = tmp_path / "suite-ledger.jsonl"
    control.write_bytes(b"x" * 10000)
    result = prune(tmp_path, {"receipt_cap": "25KB"})
    assert [x["path"] for x in result["deleted"]] == [str(old)]
    assert result["cap_unmet"]
    assert anchor.exists() and protected.exists() and control.exists()


def test_active_worker_prunes_completed_receipts_from_combined_cap(tmp_path):
    fixture = plan(tmp_path)
    fixture["retention"] = {"receipt_cap": "50KB"}
    fixture["rows"][0]["argv"] = [sys.executable, "-c",
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'active.json').write_bytes(b'x'*10000); time.sleep(.6)", "{out}"]
    runner = SuiteRunner(fixture, tmp_path / "state")
    old = runner.root / "old.json"
    old.write_bytes(b"x" * 40000)
    register(runner.root, old, "receipts", row="old", cycle=CYCLE)
    runner.run(cycles=[CYCLE], poll=.02)
    entry = runner.ledger.state()["jobs"][("forecast", CYCLE)]
    assert entry["phase"] == "completed"
    assert not old.exists()
    audited = entry["result"].get("retention_deleted", [])
    for record in runner.ledger.settle():
        if record["event"] == "retained":
            audited.extend(record["deleted"])
    assert any(x["path"] == str(old) for x in audited)


def test_active_worker_stops_for_combined_control_and_active_cap(tmp_path):
    fixture = plan(tmp_path)
    fixture["retention"] = {"receipt_cap": "50KB"}
    fixture["rows"][0]["argv"] = [sys.executable, "-c",
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'active.json').write_bytes(b'x'*10000); time.sleep(20)", "{out}"]
    runner = SuiteRunner(fixture, tmp_path / "state")
    control = runner.root / "control.log"
    control.write_bytes(b"x" * 45000)
    runner.run(cycles=[CYCLE], poll=.02)
    entry = runner.ledger.state()["jobs"][("forecast", CYCLE)]
    assert entry["phase"] == "halted"
    assert entry["result"]["error_class"] == "receipt_cap_exceeded"
    assert control.exists()
    assert entry["attempt"] == 0


def test_retention_does_not_follow_replaced_link(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    path = root / "file"
    path.write_text("owned")
    register(root, path, "history", row="x", cycle=CYCLE)
    path.unlink()
    outside = tmp_path / "outside"
    outside.write_text("preserve")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="retention_path_changed"):
        prune(root, {"history": "0s"}, now=time.time() + 1)
    assert outside.read_text() == "preserve"


def test_rehearsal_retains_score_and_marks_allowed_da_fallback(tmp_path):
    from gpuwm.nwpsuite.rehearse import rehearse
    fixture = plan(tmp_path)
    fixture["rows"].insert(0, dict(id="analysis", kind="da", cards=0, box="local",
        every="6h", argv=[sys.executable, "-c", "raise SystemExit(1)"]))
    fixture["rows"][1]["dependencies"] = ["analysis"]
    fixture["rows"].append(dict(id="score", kind="verify", cards=0, box="local", every="6h",
        dependencies=["forecast"], score_files=["{out}/score.json"],
        argv=[sys.executable, "-c", "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'score.json').write_text('{\"rmse\":2}')", "{out}"]))
    root = tmp_path / "state"
    result = rehearse(fixture, root, start=CYCLE, days=1, cycles_limit=1)
    assert result["accepted"] and result["degraded"]
    assert result["fallbacks"] == [dict(row="analysis", cycle=CYCLE, reason="da_failed", initialization="public_analysis")]
    score = SuiteLedger(root).state()["jobs"][("score", CYCLE)]
    Path(score["work"], "output/score.json").unlink()
    document = status_document(fixture, root)
    forecast = next(row for row in document["rows"] if row["id"] == "forecast")
    assert forecast["last_score"] == [{"rmse": 2}]
    assert document["exit_code"] == 1


def test_rehearsal_never_accepts_missing_required_row(tmp_path):
    from gpuwm.nwpsuite.rehearse import rehearse
    fixture = plan(tmp_path)
    fixture["rows"].append(dict(id="score", kind="verify", cards=0, box="local", every="6h",
        score_files=["{out}/score.json"],
        argv=[sys.executable, "-c", "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'score.json').write_text('{\"rmse\":2}')", "{out}"]))
    root = tmp_path / "state"
    runner = SuiteRunner(fixture, root)
    runner.ledger.append("halt", row="forecast", reason="test_halt")
    result = rehearse(fixture, root, start=CYCLE, days=1, cycles_limit=1)
    assert not result["accepted"]
    assert not (root / "baseline.json").exists()


def test_apply_waits_for_boundary(tmp_path):
    fixture = plan(tmp_path, seconds=.1)
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.tick([CYCLE])
    changed = dict(fixture, suite_hash="different")
    changed["rows"] = [dict(fixture["rows"][0], priority=2)]
    assert apply_plan(runner.root, changed)["changed"] == ["forecast"]
    runner.tick([CYCLE])
    assert runner.plan["suite_hash"] == fixture["suite_hash"]
    runner.run(cycles=[CYCLE], poll=.02)
    runner.tick([CYCLE])
    assert runner.plan["suite_hash"] == "different"


def test_apply_shows_changed_box_settings(tmp_path):
    fixture = plan(tmp_path)
    runner = SuiteRunner(fixture, tmp_path / "state")
    changed = dict(fixture, suite_hash="changed", boxes=[dict(id="local", cards=2)])
    diff = apply_plan(runner.root, changed)
    assert diff["settings_changed"] == ["boxes"]
    assert diff["changed"] == []


@pytest.mark.parametrize("phase,reason,expected", [("completed", None, 0), ("waiting", None, 75),
    ("skipped", "source_behind", 1), ("halted", "recovery_exhausted", 2)])
def test_status_exit_codes(tmp_path, phase, reason, expected):
    fixture = plan(tmp_path)
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.record(fixture["rows"][0], CYCLE, phase, reason=reason, deadline_unix=time.time() + 100)
    from gpuwm.cycle.ledger import write_receipt_atomically
    write_receipt_atomically(runner.root / "heartbeat.json", {"updated_unix": time.time()})
    assert status_document(fixture, runner.root)["exit_code"] == expected


def test_queue_wait_is_not_source_wait(tmp_path):
    fixture = plan(tmp_path, extra={"sources": ["public"]})
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.record(fixture["rows"][0], CYCLE, "waiting", reason="card_admission")
    document = status_document(fixture, runner.root)
    assert document["rows"][0]["state"] == "queued"
    assert document["exit_code"] == 1
    assert document["boxes"][0]["sources_behind"] == []


def test_status_only_names_waited_or_exhausted_sources(tmp_path):
    fixture = plan(tmp_path, extra={"sources": ["first", "fallback"]})
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.record(fixture["rows"][0], CYCLE, "waiting", source_index=0)
    assert status_document(fixture, runner.root)["boxes"][0]["sources_behind"] == ["first"]
    runner.record(fixture["rows"][0], CYCLE, "skipped", source_index=1, reason="source_behind")
    assert status_document(fixture, runner.root)["boxes"][0]["sources_behind"] == ["fallback", "first"]


def test_forecast_health_uses_its_downstream_scorecard(tmp_path):
    fixture = plan(tmp_path)
    fixture["rows"].append(dict(id="score", kind="verify", cards=0, box="local", dependencies=["forecast"], argv=["unused"]))
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.record(fixture["rows"][0], CYCLE, "completed")
    runner.record(fixture["rows"][1], CYCLE, "completed", result={"score": {"rmse": 4}})
    (runner.root / "baseline.json").write_text(json.dumps({"score": [{"cycle": CYCLE, "score": {"rmse": 3}}]}))
    row = status_document(fixture, runner.root)["rows"][0]
    assert row["last_score"] == [{"rmse": 4}]
    assert row["score_delta"] == {"/rmse": 1}
    assert row["score_rows"] == ["score"]


def test_readiness_timeout_uses_bounded_recovery(tmp_path, monkeypatch):
    fixture = plan(tmp_path, extra={"readiness_argv": ["missing-ready-door"]})
    runner = SuiteRunner(fixture, tmp_path / "state")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 120)
    monkeypatch.setattr(subprocess, "run", timeout)
    for _ in range(4):
        runner.tick([CYCLE])
    entry = runner.ledger.state()["jobs"][("forecast", CYCLE)]
    assert entry["phase"] == "halted"
    assert entry["reason"] == "recovery_exhausted"
    assert not (tmp_path / "launches").exists()


def test_malformed_success_receipt_is_a_published_failure(tmp_path):
    fixture = plan(tmp_path, extra={"door_kind": "hex", "receipts": ["{out}/receipt.json"]})
    fixture["rows"][0]["argv"] = [sys.executable, "-c",
        "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'receipt.json').write_text('{broken')", "{out}"]
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.run(cycles=[CYCLE], poll=.02)
    entry = runner.ledger.state()["jobs"][("forecast", CYCLE)]
    assert entry["phase"] == "halted"
    assert entry["result"]["error_class"] == "receipt_invalid"


def test_receipt_declared_refusal_does_not_retry_exit_one(tmp_path):
    fixture = plan(tmp_path, extra={"door_kind": "hex", "receipts": ["{out}/receipt.json"]})
    fixture["rows"][0]["argv"] = [sys.executable, "-c",
        "import pathlib,sys,json; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'receipt.json').write_text(json.dumps({'schema':'gpuwm-hex.forecast-run/v1','status':'preflight_refused'})); sys.exit(1)", "{out}"]
    runner = SuiteRunner(fixture, tmp_path / "state")
    runner.run(cycles=[CYCLE], poll=.02)
    entry = runner.ledger.state()["jobs"][("forecast", CYCLE)]
    assert entry["reason"] == "door_refused"
    assert entry["attempt"] == 0
    assert entry["result"]["status_class"] == "refused"
