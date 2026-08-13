"""The instrument has to be right about its own arithmetic.

A perf receipt that mis-attributes nested time is worse than no receipt: it
ranks the wrong thing and the ranking looks measured.  These pin the three
properties every reading depends on -- off means off, nested time is
subtracted from its parent, and a raised exception does not leave the stack
mis-nested for the rest of the process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from gpuwm import perf_timing


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv(perf_timing.ENV_ENABLE, raising=False)
    perf_timing.reset()
    yield
    perf_timing.reset()


def test_disabled_records_nothing_and_still_yields_a_usable_object():
    with perf_timing.stage("obs.example") as timed:
        timed.count(gates=10)
    timing = perf_timing.phases("obs.example")
    timing.mark("one")
    timing.close(gates=1)
    perf_timing.record("obs.other", 1.0)
    snapshot = perf_timing.snapshot()
    assert snapshot["enabled"] is False
    assert snapshot["stages"] == []


def test_enabled_records_calls_and_counters(monkeypatch):
    monkeypatch.setenv(perf_timing.ENV_ENABLE, "1")
    perf_timing.reset()
    for _ in range(3):
        with perf_timing.stage("obs.example", gates=5) as timed:
            timed.count(gates=5)
    rows = {row["stage"]: row for row in perf_timing.snapshot()["stages"]}
    assert rows["obs.example"]["calls"] == 3
    assert rows["obs.example"]["counters"]["gates"] == 30


def test_nested_time_is_subtracted_from_the_parent(monkeypatch):
    monkeypatch.setenv(perf_timing.ENV_ENABLE, "1")
    perf_timing.reset()
    with perf_timing.stage("outer"):
        with perf_timing.stage("inner"):
            time.sleep(0.05)
    rows = {row["stage"]: row for row in perf_timing.snapshot()["stages"]}
    assert rows["inner"]["self_seconds"] >= 0.04
    assert rows["outer"]["seconds"] >= rows["inner"]["seconds"]
    # The parent did nothing of its own, so its self time is near zero even
    # though its inclusive time carries the child.
    assert rows["outer"]["self_seconds"] < rows["inner"]["self_seconds"]


def test_phases_are_siblings_and_close_at_the_end(monkeypatch):
    monkeypatch.setenv(perf_timing.ENV_ENABLE, "1")
    perf_timing.reset()
    timing = perf_timing.phases("obs.pipeline")
    timing.mark("first")
    time.sleep(0.01)
    timing.mark("second", regions=4)
    timing.close()
    rows = {row["stage"]: row for row in perf_timing.snapshot()["stages"]}
    assert set(rows) == {"obs.pipeline.first", "obs.pipeline.second"}
    assert rows["obs.pipeline.second"]["counters"]["regions"] == 4


def test_a_nested_exception_does_not_mis_nest_later_stages(monkeypatch):
    monkeypatch.setenv(perf_timing.ENV_ENABLE, "1")
    perf_timing.reset()
    with pytest.raises(RuntimeError):
        with perf_timing.stage("outer"):
            inner = perf_timing.stage("inner")
            inner.__enter__()
            raise RuntimeError("boom")
    with perf_timing.stage("after"):
        pass
    rows = {row["stage"]: row for row in perf_timing.snapshot()["stages"]}
    assert "after" in rows
    assert rows["after"]["calls"] == 1


def test_out_env_writes_a_receipt_at_exit(tmp_path):
    destination = tmp_path / "perf.json"
    script = (
        "from gpuwm import perf_timing\n"
        "with perf_timing.stage('io.example', bytes_written=4):\n"
        "    pass\n"
    )
    environment = {perf_timing.ENV_ENABLE: "1",
                   perf_timing.ENV_OUT: str(destination),
                   "GPUWM_NO_LOCAL_GPU": "1"}
    import os
    env = dict(os.environ)
    env.update(environment)
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == perf_timing.SCHEMA
    assert payload["stages"][0]["stage"] == "io.example"
    assert payload["stages"][0]["counters"]["bytes_written"] == 4
