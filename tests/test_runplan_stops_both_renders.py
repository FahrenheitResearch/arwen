"""A run-plan run that ends early stops, or collects, both render workers it armed.

``RunObserver`` arms two: the early render of the analysis frame and the
every-frame render behind it.  Its stop used to reach the every-frame one
only, so a stopped run's analysis render kept drawing and could publish
after the stop, and a failed run walked out on it still running.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime

import pytest

from gpuwm import first_products, go_cli, render
from gpuwm.runplan import EventStream, RunObserver


def _render_process_runs(monkeypatch, seconds: float) -> None:
    monkeypatch.setattr(render, "announce_missing_basemap", lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "render_command", lambda *a, **k: [
        sys.executable, "-S", "-c", f"import time; time.sleep({seconds})"])
    if os.name == "nt":
        monkeypatch.setattr(first_products, "_own_group_options", lambda: {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW})


def _armed(tmp_path):
    observer = RunObserver(EventStream(tmp_path / "events.jsonl", mirror=None))
    observer.arm_first_products({"run": tmp_path, "render": tmp_path / "png",
                                 "render_products": "t2"})
    frame = tmp_path / "wrfout_d01_2026-09-27_00_00_00"
    frame.write_bytes(b"CPU history identity fixture")
    observer.output_committed(domain=1, valid_time=datetime(2026, 9, 27), path=frame)
    return observer


def _process_of(worker) -> subprocess.Popen:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with first_products._RUNNING_LOCK:
            process = first_products._RUNNING.get(worker._thread.ident)
        if process is not None:
            return process
        time.sleep(0.01)
    raise AssertionError("the analysis render never started")


def _tidy(observer, process) -> None:
    if process is not None:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
    observer.first_products._thread.join(10)
    live = observer.live_products
    if live is not None and live._thread is not None:
        live._thread.join(10)


@pytest.mark.parametrize("later_frame_waiting", [False, True])
def test_a_stopped_run_ends_its_analysis_render(tmp_path, monkeypatch,
                                                later_frame_waiting):
    _render_process_runs(monkeypatch, 30)
    observer = _armed(tmp_path)
    early = observer.first_products
    process = None
    try:
        process = _process_of(early)
        assert process.poll() is None
        if later_frame_waiting:
            # The every-frame worker is then waiting on the analysis render.
            later = tmp_path / "wrfout_d01_2026-09-27_01_00_00"
            later.write_bytes(b"CPU later history fixture")
            observer.output_committed(domain=1, valid_time=datetime(2026, 9, 27, 1),
                                      path=later)
        observer.stop_live_products(halt=True)
        assert process.poll() is not None, "the stop left the analysis render running"
        assert early._ended.ended and early._abandoned
        assert not observer.live_products.running
    finally:
        _tidy(observer, process)


def test_a_failed_run_collects_its_analysis_render(tmp_path, monkeypatch):
    _render_process_runs(monkeypatch, 1.5)
    observer = _armed(tmp_path)
    early = observer.first_products
    process = None
    try:
        process = _process_of(early)
        observer.stop_live_products(halt=False)
        assert early._ended.ended, "the failed run returned with its analysis render still drawing"
        assert process.poll() is not None
    finally:
        _tidy(observer, process)


@pytest.mark.parametrize("halt", [True, False])
def test_stopping_with_no_render_armed_is_a_no_op(tmp_path, halt):
    observer = RunObserver(EventStream(tmp_path / "events.jsonl", mirror=None))
    assert observer.stop_live_products(halt=halt) is None
