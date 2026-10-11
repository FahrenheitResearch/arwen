"""DA telemetry accounting and failure controls. No model fields or CUDA."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.da import member_transport as transport
from gpuwm.timeline.taxonomy import validate


def events(path):
    return [validate(json.loads(line)) for line in Path(path).read_text().splitlines()]


def fake_clock(monkeypatch, tmp_path, *, card=2):
    now = [10.0]
    monkeypatch.setattr(transport, "time", SimpleNamespace(
        monotonic=lambda: now[0], time=lambda: 1000.0+now[0]))
    monkeypatch.delenv("GPUWM_TIME_LEDGER", raising=False)
    clock = transport.DAClock(tmp_path, cycle=1, member=0,
                              card=card, resolve_card=False)
    return now, clock


def test_phase_partition_and_step_boundaries_exclude_output_drain(tmp_path, monkeypatch):
    now, clock = fake_clock(monkeypatch, tmp_path)
    clock.transition("prep.member.restore")
    now[0] = 12
    clock.transition("post.member.integrate")
    now[0] = 14
    clock.step(step_wall_seconds=1.5, grid_id=1)
    now[0] = 18
    clock.step(step_wall_seconds=2, grid_id=1)
    clock.finish_steps(10, complete=True)
    now[0] = 19
    clock.transition("forecast.output.member")
    now[0] = 25
    clock.transition()
    rows = events(tmp_path/"timeline.jsonl")
    integrate = [e for e in rows if e["stage"] == "forecast.integrate"]
    assert [(e["phase"], e["t_mono"]) for e in integrate] == [("begin", 12.5), ("end", 18)]
    assert all(e["attrs"]["card_busy"] and e["card"] == 2 for e in integrate)
    assert sum(clock.seconds.values()) == 15
    assert clock.seconds["forecast.output.member"] == 6
    assert all(e["product"] == "da" for e in rows)
    assert all(e["attrs"]["card_busy"] is False for e in rows if e not in integrate)


def test_steps_emit_only_at_leg_boundary_and_zero_steps_are_unknown(tmp_path, monkeypatch):
    now, clock = fake_clock(monkeypatch, tmp_path)
    for i in range(100):
        now[0] += 1
        clock.step(step_wall_seconds=0.2)
    assert not (tmp_path/"timeline.jsonl").exists()
    clock.finish_steps(10, complete=True)
    assert len(events(tmp_path/"timeline.jsonl")) == 6
    empty = transport.DAClock(tmp_path/"empty", resolve_card=False)
    empty.finish_steps(10, complete=True)
    assert events(tmp_path/"empty"/"timeline.jsonl")[0]["attrs"]["complete"] is False


def test_barriers_use_each_member_readiness_and_keep_cards_and_span_ids(tmp_path, monkeypatch):
    now, first = fake_clock(monkeypatch, tmp_path, card=1)
    ready_first = first.ready()
    now[0] = 17
    second = transport.DAClock(tmp_path, cycle=1, member=1, card=3)
    ready_second = second.ready()
    results = {0: SimpleNamespace(record={"da_timing": {"ready": ready_first}}),
               1: SimpleNamespace(record={"da_timing": {"ready": ready_second}}),
               2: SimpleNamespace(record={})}
    waits = transport.member_barriers(results, tmp_path, 1, 20)
    assert waits == {"0": 10, "1": 3}
    rows = events(tmp_path/"timeline.jsonl")
    barriers = [e for e in rows if e["stage"] == "post.member.barrier"]
    assert [(e["card"], e["phase"], e["t_mono"]) for e in barriers] == [
        (1, "begin", 10), (1, "end", 20), (3, "begin", 17), (3, "end", 20)]
    assert {e["attrs"]["span_id"] for e in barriers} == {"1:0", "1:1"}
    assert not any(e["attrs"]["card_busy"] for e in barriers)
    assert rows[-1]["stage"] == "post.member.barrier_coverage"


def test_ledger_failure_preserves_work_and_warns_once(tmp_path, monkeypatch, capsys):
    now, clock = fake_clock(monkeypatch, tmp_path)
    blocked = tmp_path/"blocker"
    blocked.write_bytes(b"keep")
    clock.ledger.path = blocked/"timeline.jsonl"
    clock.transition("prep.member.restore")
    now[0] += 4
    clock.transition()
    assert sum(clock.seconds.values()) == 4
    assert blocked.read_bytes() == b"keep"
    assert capsys.readouterr().err.count("cannot append") == 1


def test_async_spans_close_on_failure_without_swallowing_it(tmp_path, monkeypatch):
    now, clock = fake_clock(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="write failed"):
        with clock.span("post.analysis.generation_write", span_id="1:save"):
            now[0] += 3
            raise RuntimeError("write failed")
    rows = events(tmp_path/"timeline.jsonl")
    assert [e["phase"] for e in rows] == ["begin", "end"]
    assert rows[1]["t_mono"]-rows[0]["t_mono"] == 3


def test_uuid_card_lookup_is_cached_and_unknown_is_not_guessed(monkeypatch):
    import subprocess
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    calls = []
    def query(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout="3, GPU-test\n7, GPU-other\n")
    monkeypatch.setattr(subprocess, "run", query)
    transport._timing_card.cache_clear()
    assert transport._timing_card("GPU-test") == 3
    assert transport._timing_card("GPU-test") == 3
    assert len(calls) == 1
    assert "--id=GPU-test" in calls[0]
    assert transport._timing_card("GPU-missing") is None
    assert transport._timing_card("3,7") is None
    assert transport._timing_card("4") == 4


def test_real_timeline_consumer_uses_union_and_all_allocated_cards(tmp_path, monkeypatch):
    from gpuwm.timeline.report import Timeline
    now, clock = fake_clock(monkeypatch, tmp_path)
    clock.emit("worker.claim", "mark", 10, proxy=True, card_count=4)
    clock.interval("forecast.integrate", 12, 20, card_busy=True)
    other = transport.DAClock(tmp_path, cycle=1, member=1, card=3)
    other.interval("forecast.integrate", 15, 25, card_busy=True)
    clock.interval("forecast.output.member", 20, 30)
    clock.emit("finish", "mark", 40, proxy=True)
    report = Timeline().read(tmp_path/"timeline.jsonl").report(product="da")
    assert report["ff"] is None
    assert report["proxy_wall_seconds"] == 30
    assert report["integrate_union_seconds"] == 13
    assert report["proxy_ff"] == pytest.approx(13/30)
    assert report["card_busy_fraction"] == pytest.approx(18/(4*30))


def test_result_publication_keeps_restart_and_array_bytes(tmp_path, monkeypatch):
    import numpy as np
    from test_da_member_transport import make_context
    from test_da_member_wave import native_restart
    from tools.da_member_leg import MemberLegResult
    monkeypatch.delenv("GPUWM_TIME_LEDGER", raising=False)
    context = make_context(tmp_path)
    root = tmp_path/"job"
    root.mkdir()
    restart = root/"gpuwmrst_d01_end.npz"
    native_restart(restart, 0)
    before = transport.sha256(restart)
    values = np.array([0x7ff8000000000001, 0x8000000000000000], np.uint64).view(np.float64)
    result = MemberLegResult(restart=restart, snapshot=None, H_Z=values,
        H_surface=None, hot_pending=None, setup_arrays={}, thb_host=None,
        nest_birth=None, record={"elapsed_seconds": 60.0},
        consume_restart=None, pending_consumed=False)
    document = transport.write_result(root/"result.json", context, 0, result, "a"*64)
    loaded = transport.load_result(document)
    assert loaded.H_Z.tobytes() == values.tobytes()
    assert transport.sha256(restart) == before
    assert loaded.record == result.record
    assert document["record"]["da_timing"]["ready"]["attrs"]["member"] == 0


def test_main_restores_ledger_destination_when_cycle_refuses(tmp_path, monkeypatch):
    from tools import da_cycle_prepared as cycle
    monkeypatch.setenv("GPUWM_TIME_LEDGER", "original")
    monkeypatch.setattr(cycle, "_unwind_on_termination", lambda: {})
    def refuse(stages):
        import os
        os.environ["GPUWM_TIME_LEDGER"] = str(tmp_path/"replacement")
        raise RuntimeError("admission refused")
    monkeypatch.setattr(cycle, "cycle", refuse)
    with pytest.raises(RuntimeError, match="admission refused"):
        cycle.main()
    import os
    assert os.environ["GPUWM_TIME_LEDGER"] == "original"


def test_card_server_prefetch_emits_its_actual_overlapping_work(tmp_path, monkeypatch):
    import threading
    monkeypatch.delenv("GPUWM_TIME_LEDGER", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    queue = tmp_path/"queue"
    queue.mkdir()
    request = tmp_path/"request.json"
    request.write_text("{}")
    job = {"schema": transport.SERVER_JOB_SCHEMA, "request": str(request),
           "request_sha256": "a"*64}
    (queue/"job-00001.json").write_text(json.dumps(job))
    context = SimpleNamespace(out=tmp_path/"out", absolute_leg_number=2)
    monkeypatch.setattr(transport, "read_context", lambda *args, **kwargs:
                        (context, {"member": 4, "restart_files": []}))
    ahead = transport._Prefetch(queue, 1, threading.Event())
    assert ahead.done.wait(5)
    assert ahead.result_for(request, "a"*64)[0] is context
    rows = events(tmp_path/"out"/"timeline.jsonl")
    assert [row["phase"] for row in rows] == ["begin", "end"]
    assert all(row["stage"] == "prep.member.prefetch" and row["card"] == 3
               and row["attrs"]["member"] == 4 and row["attrs"]["cycle"] == 2
               and row["attrs"]["asynchronous"] is True for row in rows)


def test_cycle_input_and_bookkeeping_are_part_of_the_measured_wall(tmp_path, monkeypatch):
    from gpuwm.timeline.report import Timeline
    now, clock = fake_clock(monkeypatch, tmp_path)
    clock.emit("worker.claim", "mark", proxy=True)
    for stage, start in [("prep.cycle.inputs", 10), ("post.cycle.forecast", 13),
                         ("post.cycle.analysis", 20), ("post.cycle.recovery", 23),
                         ("post.cycle.bookkeeping", 25)]:
        now[0] = start
        clock.transition(stage)
    now[0] = 27
    clock.transition()
    clock.emit("finish", "mark", proxy=True)
    report = Timeline().read(tmp_path/"timeline.jsonl").report(product="da")
    assert report["proxy_wall_seconds"] == 17
    assert sum(clock.seconds.values()) == 17
    assert clock.seconds["prep.cycle.inputs"] == 3
    assert clock.seconds["post.cycle.bookkeeping"] == 2
