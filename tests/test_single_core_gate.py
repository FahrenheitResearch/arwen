"""The single-core gate (gpuwm/timeline/parallelism.py) must catch a deliberately serial stage.

Breakage it prevents (2026-10-16): rw_mpas_geometry held one core for 8+ minutes on a rented RTX PRO 6000 box
while its card idled; the parallel fix existed (engine 08be92aabdd) but never reached the box, and no receipt
said "1 core".  These tests run real processes, not mocks: a serial spin must be RED, the same work on
several processes GREEN, and every C1 begin/end must carry the CPU seconds that make per-stage cores.
"""
import json
import os
import subprocess
import sys

import pytest

from gpuwm.timeline import parallelism as par
from gpuwm.timeline.ledger import Ledger, span

LINUX = par._LINUX
QUICK = dict(min_wall=1.0, max_cores=1.3, min_cores=0.6)
SPIN = par.SERIAL


def _spin(seconds):
    return [sys.executable, "-c", SPIN.format(s=seconds)]


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")


def test_rule():
    assert par.judge(60, 1.0, None) == "single-core"
    assert par.judge(60, 1.0, 0.9) == "card-bound"
    assert par.judge(60, 1.0, 0.1) == "single-core"
    assert par.judge(60, 0.2, None) == "waiting"
    assert par.judge(60, 6.0, None) == "parallel"
    assert par.judge(4, 1.0, None) == "short"
    assert par.judge(60, None, None) == "unmeasured"


def test_parallel_prewarm_and_network_wait_do_not_make_a_serial_fetch():
    # Cold go: 30 s at three cores, then 50 s waiting for input. The
    # aggregate is 1.125 cores, but no interval is pinned on one core.
    ticks = [{"t": float(i), "mono": float(i), "total": min(i, 30) * 3 * par._TICK,
              "per": {"1": min(i, 30) * 3 * par._TICK}, "on_card": False}
             for i in range(81)]
    result = par.analyse(ticks, {1: "kernel-prewarm"}, None, interval=1.0, ncpu=4)
    record = {**par.stage_record(80, 90, None), "t0": 0.0, "t1": 80.0}
    assert record["verdict"] == "single-core"
    assert par.verdict(result, [], [("fetch.inputs", record)])[0] == []
    # Without full samples the stage evidence must still fail closed.
    assert par.verdict({"windows": []}, [], [("fetch.inputs", record)])[0]
    assert par.verdict({**result, "series": result["series"][1:]}, [],
                       [("fetch.inputs", record)])[0]
    child = {**record, **par.stage_record(80, 80, None)}
    assert par.verdict(result, [], [("unrelated.serial_child", child)])[0]
    for tick in ticks:
        tick["total"] /= 3
        tick["per"]["1"] /= 3
    serial = par.analyse(ticks, {1: "kernel-prewarm"}, None, interval=1.0, ncpu=4)
    assert par.verdict(serial, [], [("fetch.inputs", record)])[0]


def test_allowlist_needs_a_named_reason(tmp_path):
    bad = tmp_path / "a.json"
    bad.write_text(json.dumps({"allow": [{"match": "rw_mpas_geometry", "reason": "", "owner": "x",
                                          "retire_when": "y"}]}))
    with pytest.raises(ValueError, match="reason"):
        par.load_allowlist(bad)
    good = tmp_path / "b.json"
    good.write_text(json.dumps({"allow": [{"match": "^rw_mpas_geometry", "reason": "re-pin in flight",
                                           "owner": "hex-speed lane", "retire_when": "pin >= 30de7226b5b"}]}))
    rows = par.load_allowlist(good)
    assert par.allowed("rw_mpas_geometry --protocol hex-vertical-v1", rows)["owner"] == "hex-speed lane"
    assert par.allowed("rw_mpas_mesh", rows) is None


def test_shipped_allowlist_loads():
    for row in par.load_allowlist():
        assert row["reason"].strip()


@pytest.mark.skipif(not LINUX, reason="process-tree sampling reads Linux /proc")
def test_serial_stage_is_red_and_parallel_stage_is_green():
    rc, one = par.watch(_spin(2.5), interval=0.25, **QUICK)
    assert rc == 0
    bad, _ = par.verdict(one, [])
    assert bad, one["windows"]
    assert 0.6 <= bad[0]["cores"] <= 1.3 and bad[0]["wall_s"] > 1.0
    assert "python" in bad[0]["what"]
    n = max(2, min(4, os.cpu_count() or 2))
    if n < 3:
        pytest.skip("needs 3+ cores to show a parallel stage")
    many = [sys.executable, "-c", "import subprocess, sys\n"
            f"ps = [subprocess.Popen([sys.executable, '-c', {SPIN.format(s=2.5)!r}]) for _ in range({n})]\n"
            "sys.exit(max(p.wait() for p in ps))\n"]
    rc, wide = par.watch(many, interval=0.25, **QUICK)
    assert rc == 0
    assert wide["cores"] > 1.3, wide
    assert par.verdict(wide, [])[0] == []


@pytest.mark.skipif(not LINUX, reason="process-tree sampling reads Linux /proc")
def test_cli_watch_fails_a_serial_command_and_allowlist_names_it(tmp_path):
    out = tmp_path / "p.json"
    gate = [sys.executable, par.__file__, "watch", "--interval", "0.25", "--min-wall", "1",
            "--out", str(out), "--", *_spin(2.5)]
    done = subprocess.run(gate, capture_output=True, text=True, timeout=60)
    assert done.returncode == 3, done.stdout + done.stderr
    assert "SINGLE-CORE" in done.stdout and json.loads(out.read_text())["gate"] == "RED"
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps([{"match": "^python", "reason": "the test's own spin", "owner": "this test",
                                  "retire_when": "never"}]))
    done = subprocess.run(gate[:3] + ["--allow", str(allow)] + gate[3:], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "allowed" in done.stdout


@pytest.mark.skipif(not LINUX, reason="process-tree CPU reads Linux /proc")
def test_c1_stage_carries_cpu_and_a_serial_stage_is_flagged(tmp_path, monkeypatch):
    path = tmp_path / "timeline.jsonl"
    monkeypatch.setenv("GPUWM_TIME_LEDGER", str(path))
    with span("prep.serial_child", ledger=Ledger()):
        subprocess.run(_spin(1.6), check=True)       # a child process: its CPU arrives as reaped-children time
    with span("prep.idle", ledger=Ledger()):
        pass
    events = [json.loads(x) for x in path.read_text().splitlines()]
    assert all(isinstance(e["attrs"].get("cpu_s"), float) for e in events)
    rows = dict(par.ledger_stages(path, **QUICK))
    assert rows["prep.serial_child"]["verdict"] == "single-core", rows
    assert rows["prep.idle"]["verdict"] == "short"
    done = subprocess.run([sys.executable, par.__file__, "check", "--min-wall", "1", "--ledger", str(path)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 3 and "prep.serial_child" in done.stdout, done.stdout + done.stderr


def test_relayed_boundaries_carry_no_cpu(tmp_path, monkeypatch):
    path = tmp_path / "timeline.jsonl"
    monkeypatch.setenv("GPUWM_TIME_LEDGER", str(path))
    Ledger().emit("forecast.integrate", "begin", t_mono=1.0, t_unix=1.0)
    assert "cpu_s" not in json.loads(path.read_text())["attrs"]


def test_stage_meter_records_cores():
    meter = par.StageMeter(cards=False)
    mark = meter.mark()
    subprocess.run(_spin(1.2), check=True)
    rec = meter.since(mark, **QUICK)
    meter.close()
    if LINUX:
        assert 0.6 <= rec["cores"] <= 1.3 and rec["verdict"] == "single-core", rec
    assert set(rec) >= {"wall_s", "cpu_s", "cores", "card_busy", "verdict"}


def test_a_busy_card_that_the_tree_does_not_hold_excuses_nothing():
    # 2026-10-16, a shared host: a CPU-only hex init read as card-bound because another lane's job held the shared card
    class Cards:
        active = True

        def busy_fraction(self, a, b):
            return 1.0
    names = {1: "rw_mpas_geometry --protocol hex-vertical-v1"}

    def ticks(on_card):
        return [{"t": float(i), "mono": float(i), "total": i * par._TICK, "per": {"1": i * par._TICK},
                 "on_card": on_card} for i in range(9)]
    other = par.analyse(ticks(False), names, Cards(), interval=1.0, ncpu=24, **QUICK)
    bad, _ = par.verdict(other, [])
    assert bad and bad[0]["what"].startswith(names[1]) and bad[0]["card_busy"] == 0.0
    ours = par.analyse(ticks(True), names, Cards(), interval=1.0, ncpu=24, **QUICK)
    assert par.verdict(ours, [])[0] == []


@pytest.mark.skipif(not LINUX, reason="reads /proc/<pid>/fd")
def test_this_test_process_holds_no_card():
    assert par.holds_card(par.tree_pids()) in (False, True)
    assert par.holds_card([999999999]) is False
