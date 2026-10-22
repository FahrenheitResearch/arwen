"""A forecast leg's fixed cost: what was moved, and what must not change with it.

Measured on a 300 x 300 x 49 member, RTX 4090, history and restart every
15 min: each checkpoint held the stepping thread 1.2 to 1.5 s, a leg spent
about 8 s before its first step, and short legs paid all of it every time.
The GPU proofs (every wrfout and checkpoint array bitwise) are card runs;
these are the CPU halves:

* file digests are computed once per process per unchanged file, and a
  changed file is hashed again (``gpuwm.file_digest``);
* ``--checkpoint-at`` selects scheduled checkpoints and refuses an instant
  the run never writes (``gpuwm.checkpoint_schedule``);
* checkpoint staging leaves host arrays to the synchronous route
  (``gpuwm.io.checkpoint_staging``);
* a seed cache fills only missing kernel entries (``gpuwm.kernel_cache_seed``);
* the warm leg executor runs the door's own ``main`` in process, refuses
  supervisor flags, and stops at a failed leg (``gpuwm.leg_executor``);
* ``--benchmark`` says where the start-up went (``gpuwm.benchmark_timing``).
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest


# -- file digests ---------------------------------------------------------------

def test_a_digest_is_reused_for_an_unchanged_file_and_redone_for_a_changed_one(tmp_path):
    import hashlib

    from gpuwm import file_digest

    file_digest.forget()
    path = tmp_path / "input.bin"
    path.write_bytes(b"a" * 4096)
    first = file_digest.sha256_file(path)
    assert first == hashlib.sha256(b"a" * 4096).hexdigest()
    calls = []
    real = hashlib.sha256

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    try:
        hashlib.sha256 = counting
        assert file_digest.sha256_file(path) == first
        assert not calls, "an unchanged file was hashed again"
        # Same size, different bytes, written in place: hashed again.
        time.sleep(0.01)
        with open(path, "r+b") as handle:
            handle.write(b"b")
        changed = file_digest.sha256_file(path)
        assert calls and changed == real(b"b" + b"a" * 4095).hexdigest()
    finally:
        hashlib.sha256 = real
    # Replaced by rename: a new inode, hashed again.
    other = tmp_path / "other.bin"
    other.write_bytes(b"c" * 4096)
    os.replace(other, path)
    assert file_digest.sha256_file(path) == hashlib.sha256(b"c" * 4096).hexdigest()


def test_the_door_and_table_hashers_use_the_digest_memo(tmp_path):
    from gpuwm import file_digest, wrfinput_forecast
    from gpuwm.core import thompson_contract
    from gpuwm.io import restart

    path = tmp_path / "table.bin"
    path.write_bytes(os.urandom(1 << 16))
    import hashlib
    want = hashlib.sha256(path.read_bytes()).hexdigest()
    file_digest.forget()
    assert wrfinput_forecast._sha(path) == want
    assert thompson_contract._sha256_file(path) == want
    assert restart._asset_sha256(path) == want


# -- checkpoint selection -------------------------------------------------------

def test_checkpoint_at_parses_valid_times_and_end():
    from gpuwm import checkpoint_schedule as cs

    assert cs.parse(None) is None
    assert cs.parse("2022-06-13_22:00:00, end") == ("2022-06-13_22:00:00", "end")
    with pytest.raises(ValueError, match="not a WRF valid time"):
        cs.parse("22:00")
    with pytest.raises(ValueError, match="names no instant"):
        cs.parse(" , ")


def test_checkpoint_at_resolves_to_restart_instants_and_refuses_the_rest():
    from gpuwm import checkpoint_schedule as cs

    start = datetime(2022, 6, 13, 21)
    tick_den = 4
    run_ticks = 3600 * tick_den
    restart_ticks = 900 * tick_den
    chosen = cs.resolve(("2022-06-13_21:30:00", "end"), start_time=start,
                        run_ticks=run_ticks, restart_ticks=restart_ticks,
                        tick_den=tick_den)
    assert chosen.keeps(1800 * tick_den) and chosen.keeps(run_ticks)
    assert not chosen.keeps(900 * tick_den) and not chosen.keeps(2700 * tick_den)
    with pytest.raises(ValueError, match="not a restart_interval"):
        cs.resolve(("2022-06-13_21:20:00",), start_time=start, run_ticks=run_ticks,
                   restart_ticks=restart_ticks, tick_den=tick_den)
    with pytest.raises(ValueError, match="outside this run"):
        cs.resolve(("2022-06-13_23:00:00",), start_time=start, run_ticks=run_ticks,
                   restart_ticks=restart_ticks, tick_den=tick_den)
    with pytest.raises(ValueError, match="schedules none"):
        cs.resolve(("end",), start_time=start, run_ticks=run_ticks,
                   restart_ticks=None, tick_den=tick_den)


def test_checkpoint_selection_is_scoped_and_a_restart_request_always_writes(
        tmp_path, monkeypatch):
    from gpuwm import checkpoint_schedule as cs
    from gpuwm.core.restart_request import RESTART_REQUEST_ENV

    assert cs.current() is None
    with cs.selected(("end",)):
        assert cs.current() == ("end",)
    assert cs.current() is None
    request = tmp_path / "request"
    monkeypatch.setenv(RESTART_REQUEST_ENV, str(request))
    assert not cs.restart_requested()
    request.write_text("")
    assert cs.restart_requested()


def test_the_door_forwards_checkpoint_at_to_its_worker():
    from gpuwm import wrfinput_forecast

    args = wrfinput_forecast.build_parser().parse_args(
        ["--wrfinput", "w", "--outdir", "o", "--checkpoint-at", "end"])
    assert args.checkpoint_at == "end"


# -- checkpoint staging ---------------------------------------------------------

def test_staging_leaves_host_arrays_to_the_synchronous_route():
    from gpuwm.io.checkpoint_staging import StagedSnapshot

    manifest = {"b": np.arange(6, dtype=np.float32).reshape(2, 3),
                "a": np.asfortranarray(np.ones((3, 2), dtype=np.float64)),
                "z": np.float32(3.0)}
    seen = []

    def owned(value):
        seen.append(value)
        return np.array(value, copy=True)

    snapshot = StagedSnapshot()
    out = snapshot.stage(manifest, owned)
    assert list(out) == ["a", "b", "z"]
    assert snapshot.staged_arrays == 0 and snapshot.fallback_arrays == 3
    for key in manifest:
        assert np.asarray(out[key]).tobytes(order="A") == np.asarray(manifest[key]).tobytes(order="A")
        assert np.asarray(out[key]).flags.f_contiguous == np.asarray(manifest[key]).flags.f_contiguous
    snapshot.wait()
    snapshot.release()


# -- kernel cache seed ----------------------------------------------------------

def test_a_seed_fills_missing_entries_and_never_replaces_one(tmp_path):
    from gpuwm import kernel_cache_seed

    seed = tmp_path / "baked"
    cache = tmp_path / "cache"
    (seed / "gpuwm-direct-nvrtc").mkdir(parents=True)
    cache.mkdir()
    (seed / "aaaa.cubin").write_bytes(b"seeded-a")
    (seed / "bbbb.cubin").write_bytes(b"seeded-b")
    (seed / "gpuwm-direct-nvrtc" / "cccc.image").write_bytes(b"seeded-c")
    (cache / "bbbb.cubin").write_bytes(b"compiled-here")
    receipt = kernel_cache_seed.seed(cache, [seed])
    assert receipt["seeded"] == 2 and receipt["already_present"] == 1
    assert (cache / "aaaa.cubin").read_bytes() == b"seeded-a"
    assert (cache / "bbbb.cubin").read_bytes() == b"compiled-here"
    assert (cache / "gpuwm-direct-nvrtc" / "cccc.image").read_bytes() == b"seeded-c"
    assert not [p for p in cache.rglob(".seed-*")]
    assert kernel_cache_seed.seed(cache, [seed])["seeded"] == 0


def test_no_seed_means_no_work(monkeypatch, tmp_path):
    from gpuwm import kernel_cache_seed

    monkeypatch.delenv(kernel_cache_seed.ENV, raising=False)
    assert kernel_cache_seed.seed(tmp_path) == {"seeded": 0}


# -- the warm leg executor --------------------------------------------------------

def test_the_executor_refuses_supervisor_flags():
    from gpuwm import leg_executor

    with pytest.raises(leg_executor.LegRefused, match="--_worker"):
        leg_executor._check_leg({"argv": ["--wrfinput", "w", "--_worker"]})
    with pytest.raises(leg_executor.LegRefused, match="list of strings"):
        leg_executor._check_leg({"argv": "--wrfinput w"})


def test_the_executor_runs_each_leg_through_the_door_in_process_and_stops_at_a_failure(
        monkeypatch, tmp_path):
    from gpuwm import leg_executor, wrfinput_forecast

    calls = []

    def door(argv, *, in_process=False):
        calls.append((list(argv), in_process))
        print(f"leg output {argv[-1]}")
        return 0 if argv[-1] != "bad" else 1

    monkeypatch.setattr(wrfinput_forecast, "main", door)
    legs = [{"name": f"l{k}", "argv": ["--outdir", name], "log": str(tmp_path / f"{name}.log")}
            for k, name in enumerate(("one", "two", "bad", "never"))]
    receipt = tmp_path / "receipt.json"
    assert leg_executor.run_legs(legs, receipt=receipt) == 3
    assert calls == [(["--outdir", "one"], True), (["--outdir", "two"], True),
                     (["--outdir", "bad"], True)]
    assert (tmp_path / "one.log").read_text().strip() == "leg output one"
    record = json.loads(receipt.read_text())
    assert [leg["rc"] for leg in record["legs"]] == [0, 0, 1]
    assert record["stopped_at_failure"]


def test_the_executor_serves_legs_from_a_spool(monkeypatch, tmp_path):
    from gpuwm import leg_executor, wrfinput_forecast

    monkeypatch.setattr(wrfinput_forecast, "main", lambda argv, *, in_process=False: 0)
    spool = tmp_path / "spool"
    done = {}

    def serve():
        done["rc"] = leg_executor.serve(spool, idle_exit=5.0, poll=0.01)

    thread = threading.Thread(target=serve)
    thread.start()
    deadline = time.time() + 10
    while not (spool / "ready").exists() and time.time() < deadline:
        time.sleep(0.01)
    request = spool / "inbox" / ".leg001.tmp"
    request.write_text(json.dumps({"name": "leg1", "argv": ["--outdir", "x"]}))
    os.replace(request, spool / "inbox" / "leg001.json")
    answer = spool / "done" / "leg001.json"
    while not answer.exists() and time.time() < deadline:
        time.sleep(0.01)
    (spool / "stop").write_text("")
    thread.join(10)
    assert json.loads(answer.read_text())["rc"] == 0
    assert done["rc"] == 0


# -- the benchmark's start-up split ---------------------------------------------

def test_the_benchmark_says_where_the_start_up_went(tmp_path, monkeypatch):
    from gpuwm import benchmark_timing
    from tests.test_benchmark_timing import _write_log

    t0 = 1_000_000.0
    marks = {"door_start": t0, "door_read_start": t0 + 0.5, "door_spawn": t0 + 1.0,
             "worker_start": t0 + 1.2, "read_start": t0 + 2.0, "prepare_start": t0 + 2.5,
             "read_end": t0 + 12.0}
    last = _write_log(tmp_path, t0)
    summary = benchmark_timing.summarize(tmp_path, process_start=t0,
                                         process_end=last + 2.0, marks=marks)
    split = summary["startup_split_seconds"]
    assert split["door"] == pytest.approx(1.0)
    assert split["worker_spawn"] == pytest.approx(0.2)
    assert split["worker_imports"] == pytest.approx(0.8)
    assert split["worker_plan_review"] == pytest.approx(0.5)
    assert split["input_read"] == pytest.approx(9.5)
    assert "before the first step: door 1.00 s" in benchmark_timing.format_summary(summary)
    monkeypatch.setenv(benchmark_timing.DOOR_MARKS_ENV, json.dumps(
        {"door_start": t0, "door_spawn": t0 + 1.0, "door_read_start": None}))
    assert benchmark_timing.Marks().values["door_spawn"] == t0 + 1.0


# -- kernel prebuild ----------------------------------------------------------------

def test_the_recorder_keeps_each_distinct_compile_once_and_passes_it_through(
        monkeypatch, tmp_path):
    import cupy.cuda.compiler as compiler

    from gpuwm import kernel_prebuild, nvrtc_ptx_cache

    seen = []

    def fake_cached(source, options, arch, extra_source=None, backend="nvrtc",
                    enable_cooperative_groups=False, name_expressions=None,
                    log_stream=None, cache_in_memory=False, jitify=False,
                    to_ltoir=False):
        seen.append(("cupy", source, tuple(options), arch))
        return "module"

    def fake_direct(source, options=(), arch=None, filename="kern.cu",
                    name_expressions=None, log_stream=None,
                    cache_in_memory=False, jitify=None, method=None):
        seen.append(("direct", source, tuple(options), arch))
        return ("image", None)

    monkeypatch.setattr(compiler, "_compile_with_cache_cuda", fake_cached)
    monkeypatch.setattr(nvrtc_ptx_cache, "compile_using_nvrtc", fake_direct)
    monkeypatch.setattr(kernel_prebuild, "_INSTALLED", False)
    monkeypatch.setattr(kernel_prebuild, "_SEEN", set())
    record = tmp_path / "kernels.jsonl"
    assert kernel_prebuild.install_recorder(record)
    assert compiler._compile_with_cache_cuda("k1", ("-std=c++17",), None) == "module"
    assert compiler._compile_with_cache_cuda("k1", ("-std=c++17",), None) == "module"
    assert nvrtc_ptx_cache.compile_using_nvrtc("k2", ("--ftz=false",)) == ("image", None)
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    assert [(r["route"], r["source"], r["options"]) for r in rows] == [
        ("cupy", "k1", ["-std=c++17"]), ("direct", "k2", ["--ftz=false"])]
    assert [s[0] for s in seen] == ["cupy", "cupy", "direct"]


def test_prebuild_refuses_a_malformed_arch(capsys, tmp_path):
    from gpuwm import kernel_prebuild

    record = tmp_path / "r.jsonl"
    record.write_text("")
    assert kernel_prebuild.main(["build", str(record), "--arch", "sm_89",
                                 "--cache-dir", str(tmp_path / "c")]) == 2


def test_a_served_leg_that_fails_ends_the_executor_but_a_malformed_request_does_not(
        monkeypatch, tmp_path):
    from gpuwm import leg_executor, wrfinput_forecast

    monkeypatch.setattr(wrfinput_forecast, "main",
                        lambda argv, *, in_process=False: 0 if argv[-1] == "ok" else 2)
    spool = tmp_path / "spool"
    (spool / "inbox").mkdir(parents=True)
    (spool / "inbox" / "a.json").write_text(json.dumps({"argv": "not a list"}))
    (spool / "inbox" / "b.json").write_text(json.dumps({"argv": ["--outdir", "ok"]}))
    (spool / "inbox" / "c.json").write_text(json.dumps({"argv": ["--outdir", "bad"]}))
    (spool / "inbox" / "d.json").write_text(json.dumps({"argv": ["--outdir", "ok"]}))
    assert leg_executor.serve(spool, idle_exit=2.0, poll=0.01) == 3
    done = {p.name: json.loads(p.read_text())["rc"] for p in (spool / "done").glob("*.json")}
    assert done == {"a.json": 2, "b.json": 0, "c.json": 2}
    assert (spool / "inbox" / "d.json").exists()
