"""A render worker the run stopped waiting for publishes nothing and is ended.

The finalize stage waits a bounded time for the early render of the first
frame and for the every-frame render.  When that wait runs out, finalize
draws the frames itself into the same folder, so the render it gave up on
must not publish afterwards and must not keep running past the run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from gpuwm import first_products, go_cli, live_products, render

_PICTURE = Path("d01-12km") / "2m_temperature" / "2026-09-27" / "fixture.png"
_EARLY = b"EARLY opaque fixture, not an image"
_FINAL = b"FINAL opaque fixture, not an image"


def _frame(tmp_path: Path, hour: int = 0) -> Path:
    frame = tmp_path / f"wrfout_d01_2026-09-27_{hour:02d}_00_00"
    frame.write_bytes(b"CPU history identity fixture")
    return frame


def _blocked_runner(entered: threading.Event, release: threading.Event):
    """A render that draws one picture into its --out once it is released."""

    def run(command, **_options):
        entered.set()
        assert release.wait(10)
        scratch = Path(command[command.index("--out") + 1])
        picture = scratch / _PICTURE
        picture.parent.mkdir(parents=True, exist_ok=True)
        picture.write_bytes(_EARLY)
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(render, "announce_missing_basemap", lambda *a, **k: None)
    monkeypatch.setattr(first_products, "HALT_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(first_products, "_SPAWN_LOOK_SECONDS", 0.01)
    monkeypatch.setattr(live_products, "DEFAULT_WAIT_SECONDS", 0.01)


@pytest.fixture
def sleeping_render(monkeypatch):
    """Every render is a real process that would run for 30 s."""

    monkeypatch.setattr(render, "announce_missing_basemap", lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "render_command", lambda *a, **k: [
        sys.executable, "-S", "-c", "import time; time.sleep(30)"])
    if os.name == "nt":
        monkeypatch.setattr(first_products, "_own_group_options", lambda: {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW})
    monkeypatch.setattr(live_products, "DEFAULT_WAIT_SECONDS", 0.01)


def _running_process(worker) -> subprocess.Popen:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        idents = ({thread.ident for thread, _ in worker._workers}
                  if hasattr(worker, "_workers") else {worker._thread.ident})
        with first_products._RUNNING_LOCK:
            process = next((first_products._RUNNING[ident] for ident in idents
                            if ident in first_products._RUNNING), None)
        if process is not None:
            return process
        time.sleep(0.01)
    raise AssertionError("the render process never started")


def test_a_timed_out_early_render_cannot_replace_what_finalize_drew(tmp_path, quick):
    entered, release = threading.Event(), threading.Event()
    reports, warnings = [], []
    target = tmp_path / "png"
    early = first_products.FirstProducts(
        {"run": tmp_path, "render": target, "render_products": "t2"},
        runner=_blocked_runner(entered, release), report=reports.append,
        warn=lambda code, *a, **k: warnings.append(code))
    early.frame_committed(domain=1, valid_time=datetime(2026, 9, 27),
                          path=_frame(tmp_path))
    try:
        assert entered.wait(5)
        assert early.wait(timeout=0.01) is None
        # Finalize owns the folder from here and draws the frame itself.
        finalized = target / _PICTURE
        finalized.parent.mkdir(parents=True, exist_ok=True)
        finalized.write_bytes(_FINAL)
        release.set()
        early._thread.join(10)
        assert not early._thread.is_alive()
        assert finalized.read_bytes() == _FINAL
        assert reports == []
        assert not (target / first_products.FIRST_PRODUCTS_RECEIPT).exists()
        assert "first_products_timeout" in warnings
    finally:
        release.set()
        early._thread.join(10)


def test_a_timed_out_every_frame_render_cannot_replace_what_finalize_drew(
        tmp_path, quick):
    entered, release = threading.Event(), threading.Event()
    reports, warnings = [], []
    target = tmp_path / "png"
    live = live_products.LiveProducts(
        {"run": tmp_path, "render": target, "render_products": "t2"},
        runner=_blocked_runner(entered, release), report=reports.append,
        warn=lambda code, *a, **k: warnings.append(code))
    live.frame_committed(domain=1, valid_time=datetime(2026, 9, 27),
                         path=_frame(tmp_path))
    try:
        assert entered.wait(5)
        live.stop(timeout=0.01)
        finalized = target / _PICTURE
        finalized.parent.mkdir(parents=True, exist_ok=True)
        finalized.write_bytes(_FINAL)
        release.set()
        live._thread.join(10)
        assert finalized.read_bytes() == _FINAL
        assert reports == []
        assert not (target / live_products.LIVE_PRODUCTS_RECEIPT).exists()
        assert "live_products_timeout" in warnings
    finally:
        release.set()
        live._thread.join(10)


@pytest.mark.parametrize("kind", ["early", "every-frame"])
def test_a_timed_out_render_process_is_ended(tmp_path, sleeping_render, kind):
    plan = {"run": tmp_path, "render": tmp_path / "png", "render_products": "t2"}
    worker = (first_products.FirstProducts if kind == "early"
              else live_products.LiveProducts)(
        plan, report=lambda *a: None, warn=lambda *a, **k: None, own_group=True)
    worker.frame_committed(domain=1, valid_time=datetime(2026, 9, 27),
                           path=_frame(tmp_path))
    process = None
    try:
        process = _running_process(worker)
        assert process.poll() is None
        if kind == "early":
            assert worker.wait(timeout=0.01) is None
        else:
            worker.stop(timeout=0.01)
        assert process.poll() is not None, (
            f"the {kind} render finalize stopped waiting for is still running")
        assert worker._ended.ended
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        worker._thread.join(10)


def test_a_runners_timed_out_early_render_leaves_the_every_frame_render_drawing(
        tmp_path, quick):
    """The every-frame worker waits on the early render; its timeout ends that render only.

    ``LandingRenders.halt`` stops the every-frame queue too, and the
    caller here is that queue's own worker, so a timeout that halted the
    whole pair would stop the frames still to draw.
    """

    entered, release = threading.Event(), threading.Event()
    renders = live_products.LandingRenders(
        {"run": tmp_path, "render": tmp_path / "png", "render_products": "t2"},
        report=lambda *a: None, report_live=lambda *a: None,
        warn=lambda *a, **k: None, runner=_blocked_runner(entered, release))
    assert renders.frame_committed(domain=1, valid_time=datetime(2026, 9, 27),
                                   path=_frame(tmp_path))
    try:
        assert entered.wait(5)
        # Exactly the wait the every-frame worker makes before its next frame.
        assert live_products._EarlyRenderOnly(renders).wait(timeout=0.01) is None
        assert renders._abandoned, "the timed-out early render may still publish"
        assert not renders.live._closed and not renders.live._abandoned
    finally:
        release.set()
        renders._thread.join(10)
        renders.live.halt(timeout=10)
