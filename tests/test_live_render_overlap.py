"""A busy import must leave room for the next landed frame to draw."""
import subprocess
import threading
from pathlib import Path

import pytest

from gpuwm import live_products
from test_live_products import _Renderer, _Recorder, _frame, _plan


def test_default_live_workers_overlap_imports_and_preserve_receipts(tmp_path, monkeypatch):
    # Free memory for three frames and no container quota, whichever box
    # runs the test: the processors alone set this budget.
    monkeypatch.setattr(live_products, "_available_memory_bytes", lambda: 64 << 30)
    monkeypatch.setattr(live_products, "_cgroup_cpu_limit", lambda: None)
    monkeypatch.setattr(live_products.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(live_products.os, "process_cpu_count", lambda: 8, raising=False)
    monkeypatch.setattr(live_products.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    entered = threading.Barrier(3, timeout=10)
    renderer = _Renderer()
    recorder = _Recorder()
    scratch = []

    def overlap(command):
        scratch.append(command[command.index("--out") + 1])
        entered.wait()
        return renderer(command)

    live = live_products.LiveProducts(_plan(tmp_path), report=recorder.report,
                                     warn=recorder.warn, runner=overlap)
    for hour in (0, 1, 2):
        frame, valid = _frame(tmp_path, 2, hour)
        live.frame_committed(domain=2, valid_time=valid, path=frame)
    result = live.stop(timeout=30)
    assert not recorder.warnings
    assert result["published"] == 3
    assert len(set(scratch)) == 3
    assert all(not Path(path).exists() for path in scratch)
    remaining, published, _ = live_products.published_frames(
        [Path(entry["frame"]) for entry in live.published], _plan(tmp_path))
    assert remaining == []
    assert len(published) == 3


def test_live_workers_respect_cpu_affinity_on_python312(tmp_path, monkeypatch):
    # Free memory for three frames and no container quota, whichever box
    # runs the test: the processors alone set this budget.
    monkeypatch.setattr(live_products, "_available_memory_bytes", lambda: 64 << 30)
    monkeypatch.setattr(live_products, "_cgroup_cpu_limit", lambda: None)
    from gpuwm import first_products

    monkeypatch.setattr(live_products.os, "cpu_count", lambda: 64)
    monkeypatch.delattr(live_products.os, "process_cpu_count", raising=False)
    monkeypatch.setattr(live_products.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(first_products, "_run_render", run)
    live = live_products.LiveProducts(_plan(tmp_path), report=lambda _: None,
                                     warn=lambda *args, **kwargs: None)
    live._run_render(["render"])
    assert calls == [{"env_overrides": {
        "RUSTWX_LIVE_RENDER_SLOTS": "3", "RAYON_NUM_THREADS": "2"}}]


@pytest.mark.parametrize("width", [1, 2, 3])
def test_halt_ends_all_render_workers_before_any_can_publish(tmp_path, monkeypatch, width):
    from gpuwm import first_products

    monkeypatch.setattr(live_products, "_render_concurrency", lambda: width)
    entered = threading.Barrier(width + 1, timeout=10)
    stops = threading.Barrier(width, timeout=10)
    release = threading.Event()
    ended_threads = []

    def render(command):
        entered.wait()
        release.wait(10)
        return subprocess.CompletedProcess(command, 0, "", "")

    def end(thread, **kwargs):
        ended_threads.append(thread.ident)
        stops.wait()
        release.set()

    monkeypatch.setattr(first_products, "end_render", end)
    live = live_products.LiveProducts(_plan(tmp_path), report=lambda _: None,
                                     warn=lambda *args, **kwargs: None, runner=render)
    for hour in range(width):
        frame, valid = _frame(tmp_path, 2, hour)
        live.frame_committed(domain=2, valid_time=valid, path=frame)
    entered.wait()
    result = live.halt(timeout=10)
    assert len(set(ended_threads)) == width
    assert not live.running
    assert result["published"] == 0


def test_early_publication_holds_all_shared_render_slots():
    slots = threading.Semaphore(2)
    guard = live_products._ExclusiveSlots(slots, 2)
    for _ in range(2):
        with guard:
            assert not slots.acquire(blocking=False)
    assert slots.acquire(blocking=False)
    assert slots.acquire(blocking=False)
    assert not slots.acquire(blocking=False)
