"""A downscaled child draws every frame as it lands, and a stop stops it.

The breakage these pin: the draw-as-it-lands render was armed on the
run-plan route only.  A downscaled child forwarded its committed frames
to the first-frame render alone, so the analysis frame was drawn early
and every other frame waited for the forecast to end: in a 250 m child
f001 landed at 10:40 and was drawn at 13:16, and both children measured
emitted 0 ``live_products_ready`` events while their parent emitted 12.

A stand-in renderer writes what the real one writes (a picture in the
render layout and a ``render-georef.json``), so the queue, the finalize
skip and the stop are pinned without spending a real render; the real
render is proved on a real child separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gpuwm import first_products, live_products, offline_child_run, render_georef
from gpuwm.live_products import LiveProducts
from gpuwm.offline_child_run import _ChildProgress
from gpuwm.runplan import WARNING_CODES, RunObserver, EventStream, read_events

_START = datetime(1974, 4, 3, 12)
_DRAWN = ("first_products_ready", "live_products_ready")


def _frame(outdir: Path, hour: int) -> Path:
    valid = _START + timedelta(hours=hour)
    path = outdir / f"wrfout_d02_{valid:%Y-%m-%d_%H_%M_%S}"
    path.write_bytes(f"CDF child frame {hour}".encode())
    return path


def _picture(frame: Path) -> str:
    return frame.name.replace("wrfout_", "picture_") + ".png"


def _valid(hour: int) -> str:
    return (_START + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M:%SZ")


#: How long a test waits for an event it is owed.  Generous, because a
#: loaded machine only makes the event later: a wait returns the moment
#: its event happens, and only a defect that never produces it spends the
#: whole bound.
_WAIT_S = 120.0


class _Renderer:
    """Draws one picture per frame into the layout, as ``gpuwm render`` does.

    ``hold`` names frames whose render waits on ``gate`` (for at most
    ``hold_seconds``) before it returns, which is a render still in
    flight when the run is stopped.
    """

    def __init__(self, *, hold=(), hold_seconds=_WAIT_S):
        self.frames = []
        self.gate = threading.Event()
        self.hold = set(hold)
        self.hold_seconds = hold_seconds
        self._lock = threading.Lock()

    def __call__(self, command, **_options):
        # ``own_group`` is how the child asks the real runner to start
        # the render in a process group of its own; a stand-in has none.
        frame = Path(command[command.index("--series") - 1])
        with self._lock:
            self.frames.append(frame.name)
        out = Path(command[command.index("--out") + 1])
        # Not named after the frame: the frame walk is recursive and
        # takes any ``wrfout_d*`` name, a picture's included.
        key = (f"d02-250m/composite_reflectivity/19740403/"
               f"{_picture(frame)}")
        (out / key).parent.mkdir(parents=True, exist_ok=True)
        (out / key).write_bytes(b"\x89PNG " + frame.name.encode())
        (out / render_georef.GEOREF_FILENAME).write_text(json.dumps({
            "schema": render_georef.GEOREF_SCHEMA,
            "generated_utc": "1974-04-03T12:00:00Z",
            "panels": {key: {"projection": "lambert", "frame": frame.name}},
            "without_georeference": []}), encoding="utf-8")
        if frame.name in self.hold:
            self.gate.wait(self.hold_seconds)
        return subprocess.CompletedProcess(list(command), 0, "", "")


def _events(outdir: Path) -> list[dict]:
    path = outdir / "events.jsonl"
    if not path.exists():
        return []
    return read_events(path, allow_partial_tail=True)


def _drawn(outdir: Path, frame: Path) -> bool:
    return any(event["event"] in _DRAWN and event.get("frame") == str(frame)
               for event in _events(outdir))


def _wait_for(predicate, timeout: float = _WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _workers(before=frozenset()) -> list[threading.Thread]:
    """The live-picture workers alive now that ``before`` did not hold.

    A test counts the workers its own run started.  Another test in the
    same process can have left one alive, and counting that one failed
    these tests for the order the suite ran them in.
    """

    return [thread for thread in threading.enumerate()
            if thread.name == "gpuwm-live-products" and thread.is_alive()
            and thread not in before]


def _start(progress: _ChildProgress, outdir: Path, config: Path,
           products: str = "all") -> None:
    progress.start(outdir=outdir, child_config=config, ratio=12,
                   start_time=_START,
                   parent={"run_dir": str(outdir.parent), "frames": 3},
                   name="Downscale of parent")
    progress.arm_render(outdir=outdir, render_products=products)


@pytest.fixture
def child(tmp_path):
    outdir = tmp_path / "child-run"
    outdir.mkdir()
    config = tmp_path / "child.toml"
    config.write_text("# child\n", encoding="utf-8")
    return outdir, config


def test_each_child_frame_is_drawn_as_it_is_committed(child, monkeypatch):
    """Three committed frames, three pictures, each before the next frame.

    Fails on 2.7.7 and on 2.8 before this fix at the second frame: the
    analysis is drawn early and f001 is never drawn while the child runs.
    """

    outdir, config = child
    renderer = _Renderer()
    monkeypatch.setattr(first_products, "_run_render", renderer)
    progress = _ChildProgress()
    _start(progress, outdir, config)
    frames = []
    for hour in range(3):
        frame = _frame(outdir, hour)
        frames.append(frame)
        progress.output_committed(domain=2, valid_time=_valid(hour),
                                  path=str(frame),
                                  bytes=frame.stat().st_size)
        assert _wait_for(lambda: _drawn(outdir, frame)), (
            f"{frame.name} was committed and not drawn while the child ran")
    progress.close()

    events = _events(outdir)
    ready = [event for event in events
             if event["event"] == "live_products_ready"]
    assert [Path(event["frame"]).name for event in ready] == [
        frames[1].name, frames[2].name]
    assert all(event["domain"] == 2 and event["pictures"] == 1
               and event["valid_time"] == _valid(hour + 1)
               for hour, event in enumerate(ready))
    # Each frame's picture is in the stream before the next frame lands.
    order = [(event["event"], Path(event.get("frame")
                                   or event.get("path") or "").name)
             for event in events
             if event["event"] in ("output_committed", *_DRAWN)]
    assert order == [
        ("output_committed", frames[0].name),
        ("first_products_ready", frames[0].name),
        ("output_committed", frames[1].name),
        ("live_products_ready", frames[1].name),
        ("output_committed", frames[2].name),
        ("live_products_ready", frames[2].name),
    ]
    assert renderer.frames == [frame.name for frame in frames]
    # Published into the layout the finalize render uses, and recorded.
    for frame in frames:
        assert (outdir / "png" / "d02-250m" / "composite_reflectivity"
                / "19740403" / _picture(frame)).is_file()
    receipt = live_products.read_receipt(outdir / "png")
    assert [Path(entry["frame"]).name for entry in receipt["frames"]] == [
        frames[1].name, frames[2].name]


def _door(tmp_path, monkeypatch, forecast):
    """``offline_child_run.run`` with a stub forecast and the real finalize.

    The finalize render stage (``go_cli._render_stage``) runs for real;
    only the subprocess it would spawn is recorded instead of run.
    """

    import gpuwm.go_cli as go_cli

    stages = []

    def run_stage(label, command, **_):
        stages.append(list(command))

    monkeypatch.setattr(go_cli, "_run_stage", run_stage)
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    config = tmp_path / "child.toml"
    config.write_text("# child\n", encoding="utf-8")
    outdir = tmp_path / "child-run"

    def fake_run(args, progress):
        outdir.mkdir(parents=True, exist_ok=True)
        _start(progress, outdir, config, products=args.render_products)
        progress.emit("stage_started", stage="forecast", phase="integrate")
        return forecast(progress, outdir)

    monkeypatch.setattr(offline_child_run, "_run", fake_run)
    args = argparse.Namespace(outdir=outdir, render_products="all")
    return args, outdir, stages


def test_the_child_route_draws_as_it_lands_and_finalize_draws_the_rest(
        tmp_path, monkeypatch):
    """Through the door: live pictures before finalize, none drawn twice.

    Before this fix the finalize render drew f001 and f002 with every
    product.  Each frame is now drawn as it lands beside the whole hour
    before it, which closes that hour's windows, so finalize draws only
    the windowed pictures that need a longer series than the live render
    held: f001's render held f000 and f001, every frame up to it, and is
    not drawn again; f002's held f001 alone, so its longer windows are
    drawn over the whole series.
    """

    renderer = _Renderer()
    monkeypatch.setattr(first_products, "_run_render", renderer)
    committed = []
    drawn_before_next = []

    def forecast(progress, outdir):
        for hour in range(3):
            frame = _frame(outdir, hour)
            committed.append(frame)
            progress.output_committed(domain=2, valid_time=_valid(hour),
                                      path=str(frame),
                                      bytes=frame.stat().st_size)
            # The integration between two frames takes minutes, so each
            # frame's picture is out before the next frame lands.  A fixed
            # wait here let a loaded machine commit the next frame first,
            # and the queue was then finished by finalize instead.
            drawn_before_next.append(_wait_for(lambda: _drawn(outdir, frame)))
        report = {"result": "PASS", "outputs": [str(f) for f in committed]}
        offline_child_run._publish_report(report, outdir)
        return report

    args, outdir, stages = _door(tmp_path, monkeypatch, forecast)
    before = set(threading.enumerate())
    report = offline_child_run.run(args)
    assert drawn_before_next == [True, True, True], (
        "a committed frame was not drawn while the child ran")

    events = _events(outdir)
    tags = [event["event"] for event in events]
    finalize = next(index for index, event in enumerate(events)
                    if event["event"] == "stage_started"
                    and event.get("stage") == "finalize")
    live = [index for index, tag in enumerate(tags)
            if tag == "live_products_ready"]
    assert len(live) == 2 and all(index < finalize for index in live)
    assert tags[-1] == "completed"
    # Finalize drew no frame again: one windowed pass over the one frame
    # whose live render did not hold every frame before it, with the
    # frames before it as its context.
    assert len(stages) == 1
    command = stages[0]
    targets = command[command.index("render") + 1:command.index("--series")]
    assert [Path(target).name for target in targets] == [committed[2].name]
    assert command[command.index("--products") + 1] == "windowed"
    context = [Path(command[i + 1]).name for i, part in enumerate(command)
               if part == "--context-wrfout"]
    assert context == [committed[0].name, committed[1].name]
    # f001 was drawn beside f000 as it landed, which is what licenses
    # leaving it out.
    held = live_products.live_held(outdir / "png")
    assert held[committed[1].resolve()] == frozenset(
        {committed[0].resolve(), committed[1].resolve()})
    assert report["products"]["status"] == "DRAWN"
    assert _workers(before) == []


@pytest.mark.parametrize("width", [1, 3])
def test_a_stopped_child_draws_nothing_more(tmp_path, monkeypatch, width):
    """A stop: the frame in flight publishes nothing and the queue is dropped.

    The desktop kills a run 5 s after asking it to stop, so a stop that
    finished the queue first would be killed before its banner and report
    were written; and a picture landing after the banner would contradict
    its count.
    """

    # Each active render stays in flight until the stop ends it: the halt's own
    # ending of the render it no longer wants is what lets it return, as
    # it ends a real render's process.  A fixed hold ran out first on a
    # loaded machine, and the render then finished and published.
    monkeypatch.setattr(live_products, "_render_concurrency", lambda: width)
    held = {(_START + timedelta(hours=hour)).strftime("wrfout_d02_%Y-%m-%d_%H_%M_%S")
            for hour in range(1, width + 1)}
    renderer = _Renderer(hold=held)
    monkeypatch.setattr(first_products, "_run_render", renderer)
    end_render = first_products.end_render

    def ending(thread, **options):
        if getattr(thread, "name", "").startswith("gpuwm-live-render-"):
            renderer.gate.set()
        return end_render(thread, **options)

    monkeypatch.setattr(first_products, "end_render", ending)
    reached = []
    stopped_at = []

    def forecast(progress, outdir):
        analysis = _frame(outdir, 0)
        progress.output_committed(domain=2, valid_time=_valid(0),
                                  path=str(analysis), bytes=1)
        reached.append(_wait_for(lambda: _drawn(outdir, analysis)))
        for hour in range(1, width + 2):
            frame = _frame(outdir, hour)
            progress.output_committed(domain=2, valid_time=_valid(hour),
                                      path=str(frame), bytes=1)
        # All live slots are occupied, with one further frame queued.
        reached.append(_wait_for(lambda: held <= set(renderer.frames), timeout=10))
        stopped_at.append(time.monotonic())
        raise KeyboardInterrupt

    args, outdir, stages = _door(tmp_path, monkeypatch, forecast)
    before = set(threading.enumerate())
    with pytest.raises(KeyboardInterrupt):
        offline_child_run.run(args)
    # The desktop kills a run 5 s after asking it to stop.
    assert time.monotonic() - stopped_at[0] < 15.0
    assert reached == [True, True], (
        "the analysis or an active live render never began before the stop")
    renderer.gate.set()
    assert _wait_for(lambda: _workers(before) == [])

    assert set(renderer.frames) == held | {"wrfout_d02_1974-04-03_12_00_00"}
    assert len(renderer.frames) == width + 1
    events = _events(outdir)
    assert [event for event in events
            if event["event"] == "live_products_ready"] == []
    pictures = sorted(path.name for path in (outdir / "png").rglob("*.png"))
    assert pictures == ["picture_d02_1974-04-03_12_00_00.png"]
    kept = [event for event in events
            if event.get("code") == "early_render_kept"]
    assert len(kept) == 1 and kept[0]["pictures"] == 1
    assert "the 1 picture drawn while it ran is kept" in kept[0]["message"]
    assert stages == []
    # Nothing was published after the stop either.
    assert sorted(path.name for path in (outdir / "png").rglob("*.png")) == [
        "picture_d02_1974-04-03_12_00_00.png"]


def test_a_child_that_failed_on_its_own_finishes_the_frames_it_wrote(
        tmp_path, monkeypatch):
    """Not a stop: a refusal raised mid-run still draws what was written."""

    renderer = _Renderer(hold={"wrfout_d02_1974-04-03_13_00_00"},
                         hold_seconds=0.5)
    monkeypatch.setattr(first_products, "_run_render", renderer)

    def forecast(progress, outdir):
        for hour in range(3):
            frame = _frame(outdir, hour)
            progress.output_committed(domain=2, valid_time=_valid(hour),
                                      path=str(frame), bytes=1)
        raise offline_child_run.OfflineChildContractError(
            "offline child became non-finite at step 900")

    args, outdir, _stages = _door(tmp_path, monkeypatch, forecast)
    with pytest.raises(offline_child_run.OfflineChildContractError):
        offline_child_run.run(args)

    events = _events(outdir)
    ready = [Path(event["frame"]).name for event in events
             if event["event"] == "live_products_ready"]
    assert sorted(ready) == ["wrfout_d02_1974-04-03_13_00_00",
                            "wrfout_d02_1974-04-03_14_00_00"]
    kept = [event for event in events
            if event.get("code") == "early_render_kept"]
    assert len(kept) == 1 and kept[0]["pictures"] == 3
    # The banner is written after the last picture, so its count holds.
    assert events.index(kept[0]) > max(
        index for index, event in enumerate(events)
        if event["event"] in _DRAWN)
    banner = (outdir / "png" / first_products.DID_NOT_FINISH_BANNER).read_text(
        encoding="utf-8")
    assert ("3 pictures are in this folder.  Every one of them is of a "
            "frame written before the forecast stopped (the frames below)"
            ) in banner


def _plan(tmp_path):
    return {"run": tmp_path, "wrfout_dir": tmp_path,
            "render": tmp_path / "png", "render_products": "all"}


class _Recorder:
    def __init__(self):
        self.reports = []
        self.warnings = []

    def report(self, entry):
        self.reports.append(entry)

    def warn(self, code, message, **fields):
        self.warnings.append((code, message, fields))


@pytest.mark.parametrize("returncode", [130, -2, -9])
def test_a_render_that_was_stopped_publishes_nothing(tmp_path, returncode):
    """Exit 130 or a signal is a stopped render, not a drawn frame.

    The renderer's scratch then holds whatever it had staged at the
    instant it was stopped, which the sweep found published as flat
    staging names and counted as pictures.
    """

    recorder = _Recorder()
    drawn = _Renderer()

    def stopped(command):
        drawn(command)
        return subprocess.CompletedProcess(list(command), returncode, "", "")

    live = LiveProducts(_plan(tmp_path), report=recorder.report,
                        warn=recorder.warn, runner=stopped)
    frame = _frame(tmp_path, 1)
    live.frame_committed(domain=2, valid_time=_valid(1), path=frame)
    live.stop()
    assert recorder.reports == []
    assert [code for code, _, _ in recorder.warnings] == [
        "live_products_stopped"]
    assert "live_products_stopped" in WARNING_CODES
    assert list((tmp_path / "png").rglob("*.png")) == []
    assert live_products.read_receipt(tmp_path / "png") is None


def test_a_halt_ends_the_render_it_no_longer_wants(tmp_path, monkeypatch):
    """The render in flight is a real subprocess, and the halt ends it."""

    import gpuwm.go_cli as go_cli

    monkeypatch.setattr(
        go_cli, "render_command",
        lambda plan, frames=None, **_: [sys.executable, "-c",
                                        "import time; time.sleep(120)"])
    recorder = _Recorder()
    live = LiveProducts(_plan(tmp_path), report=recorder.report,
                        warn=recorder.warn)
    frame = _frame(tmp_path, 1)
    live.frame_committed(domain=2, valid_time=_valid(1), path=frame)
    assert _wait_for(lambda: bool(first_products._RUNNING), timeout=30.0)
    process = next(iter(first_products._RUNNING.values()))
    started = time.monotonic()
    live.halt()
    assert time.monotonic() - started < 5.0
    assert process.poll() is not None
    assert _wait_for(lambda: first_products._RUNNING == {}, timeout=5.0)
    assert recorder.reports == []
    # Taking no frame after it, and safe to call again.
    live.frame_committed(domain=2, valid_time=_valid(2),
                         path=_frame(tmp_path, 2))
    assert live.pending == 0
    live.halt()


def test_the_run_plan_observer_halts_on_a_stop(tmp_path):
    """The run-plan route's stop is the same halt, not a drained queue.

    For both renders it armed, in the two-worker order: the every-frame
    render closed first and joined last, the early render ended between.
    A run that failed on its own drains the queue and then collects the
    early render.
    """

    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events, root_domain=1)
    observer.arm_first_products(_plan(tmp_path))
    live = observer.live_products
    first = observer.first_products
    assert first is not None
    calls = []
    live.halt = lambda *a, **k: calls.append("halt") or {}
    live.stop = lambda *a, **k: calls.append("stop") or {}
    first.halt = lambda *a, **k: calls.append("first halt") or True
    first.wait = lambda *a, **k: calls.append("first wait")
    observer.stop_live_products(halt=True)
    observer.stop_live_products()
    events.close()
    assert calls == ["halt", "first halt", "halt", "stop", "first wait"]
