"""Every committed frame of every grid, drawn as it lands.

The breakage these pin: a nested run's page showed only its root grid
until the forecast ended, because only the first root frame was drawn
early.  A stand-in renderer writes what a real one would (pictures in the
render layout and a ``render-georef.json``) so the queue, the ordering,
the concurrency bound, the finalize skip and the map record are pinned
without spending a real render.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gpuwm import live_products, render_georef
from gpuwm.gui import frames as gui_frames
from gpuwm.gui import runs as gui_runs
from gpuwm.live_products import (LIVE_PRODUCTS_RECEIPT, LiveProducts,
                                 live_render_requested, published_frames)
from gpuwm.runplan import (EVENT_TAGS, WARNING_CODES, EventStream,
                           RunObserver, read_events)

_START = datetime(2026, 9, 25, 12)
_GRID_KM = {1: 12, 2: 3, 3: 1}


def _frame(tmp_path, domain, hour):
    valid = _START + timedelta(hours=hour)
    path = (tmp_path / "run" / "wrfout" /
            f"wrfout_d{domain:02d}_{valid:%Y-%m-%d_%H_%M_%S}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"frame d{domain} h{hour}".encode())
    return path, valid


class _Renderer:
    """Writes one picture per frame into the layout, plus the map record.

    Records every launch and the most launches ever running at once, so
    the concurrency bound is measured alongside the published frames.
    """

    def __init__(self, delay=0.0):
        self.calls = []
        self.running = 0
        self.most = 0
        self.delay = delay
        self._lock = threading.Lock()

    def __call__(self, command):
        with self._lock:
            self.running += 1
            self.most = max(self.most, self.running)
            self.calls.append(list(command))
        try:
            time.sleep(self.delay)
            out = Path(command[command.index("--out") + 1])
            frame = Path(command[command.index("--series") - 1])
            domain = int(frame.name.split("_")[1][1:])
            folder = f"d{domain:02d}-{_GRID_KM[domain]}km"
            key = f"{folder}/2m_temperature/20260925/{frame.name}.png"
            (out / key).parent.mkdir(parents=True, exist_ok=True)
            (out / key).write_bytes(b"\x89PNG " + frame.name.encode())
            (out / render_georef.GEOREF_FILENAME).write_text(json.dumps({
                "schema": render_georef.GEOREF_SCHEMA,
                "generated_utc": "2026-09-25T12:00:00Z",
                "panels": {key: {"projection": "lambert", "frame": frame.name}},
                "without_georeference": []}), encoding="utf-8")
            return subprocess.CompletedProcess(list(command), 0, "", "")
        finally:
            with self._lock:
                self.running -= 1


class _Recorder:
    def __init__(self):
        self.reports = []
        self.warnings = []

    def report(self, entry):
        self.reports.append(entry)

    def warn(self, code, message, **fields):
        self.warnings.append((code, message, fields))


def _plan(tmp_path, products=None):
    return {"run": tmp_path / "run", "render": tmp_path / "png",
            "render_products": products}


def _context(command):
    return [Path(command[i + 1]) for i, part in enumerate(command)
            if part == "--context-wrfout"]


def _drawn_frame(command):
    return Path(command[command.index("--series") - 1])


def _products(command):
    return command[command.index("--products") + 1]


def _quarter(tmp_path, domain, minutes):
    """A frame ``minutes`` after the start, on a 15-minute history grid."""

    valid = _START + timedelta(minutes=minutes)
    path = (tmp_path / "run" / "wrfout" /
            f"wrfout_d{domain:02d}_{valid:%Y-%m-%d_%H_%M_%S}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"frame d{domain} m{minutes}".encode())
    return path, valid


@pytest.mark.parametrize("products,armed", [
    (None, True), ("", True), ("all", True), ("t2", True),
    ("none", False), (" NONE ", False),
])
def test_live_drawing_is_on_whenever_the_end_of_run_draws(products, armed):
    """An unset spec is the default set, which finalize draws, so it is ON."""

    assert live_render_requested(products) is armed


def test_every_grid_is_drawn_within_the_concurrency_bound(tmp_path):
    renderer = _Renderer(delay=0.05)
    recorder = _Recorder()
    live = LiveProducts(_plan(tmp_path), report=recorder.report,
                        warn=recorder.warn, runner=renderer)
    committed = []
    for hour in range(3):
        for domain in (1, 2, 3):
            frame, valid = _frame(tmp_path, domain, hour)
            live.frame_committed(domain=domain, valid_time=valid, path=frame)
            committed.append(frame)
    deadline = time.monotonic() + 30
    while len(recorder.reports) < len(committed) and time.monotonic() < deadline:
        time.sleep(0.02)
    summary = live.stop(timeout=30)

    assert recorder.warnings == []
    # The width the pass was built with, not a second reading of the
    # host's free memory, which moves while the test runs.
    assert renderer.most <= live._concurrency
    assert {Path(r["frame"]) for r in recorder.reports} == set(committed)
    assert summary["published"] == 9 and summary["dropped"] == 0
    # A whole hour is drawn beside the previous whole hour of its OWN
    # grid and nothing more: one baseline is one import however long the
    # run, where the grid's whole history made every frame dearer than
    # the last (about 6 s a frame at 1 km).  A grid's first is alone.
    closing = {_names([_drawn_frame(command)])[0]: _names(_context(command))
               for command in renderer.calls}
    for position, frame in enumerate(committed):
        earlier = committed[position - 3] if position >= 3 else None
        assert closing[frame.name] == ([] if earlier is None
                                       else [earlier.name])
    # Pictures of all three grids are published under the layout, never
    # the scratch they were drawn in.
    grids = {p.relative_to(tmp_path / "png").parts[0]
             for p in (tmp_path / "png").rglob("*.png")}
    assert grids == {"d01-12km", "d02-3km", "d03-1km"}
    assert not (tmp_path / "png" / ".live-products-scratch").exists()


def test_the_map_record_holds_every_picture_of_every_grid(tmp_path):
    renderer = _Renderer()
    recorder = _Recorder()
    render_dir = tmp_path / "png"
    render_dir.mkdir()
    # An entry already in the folder (an earlier render) is kept.
    kept = render_dir / "d01-12km" / "old" / "20260925" / "old.png"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"old")
    (render_dir / render_georef.GEOREF_FILENAME).write_text(json.dumps({
        "schema": render_georef.GEOREF_SCHEMA, "generated_utc": "x",
        "panels": {"d01-12km/old/20260925/old.png": {"projection": "old"}},
        "without_georeference": []}), encoding="utf-8")
    live = LiveProducts(_plan(tmp_path), report=recorder.report,
                        warn=recorder.warn, runner=renderer)
    for domain in (1, 2, 3):
        frame, valid = _frame(tmp_path, domain, 0)
        live.frame_committed(domain=domain, valid_time=valid, path=frame)
    deadline = time.monotonic() + 30
    while len(recorder.reports) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    live.stop(timeout=30)

    manifest = render_georef.read(render_dir / render_georef.GEOREF_FILENAME)
    pictures = {p.relative_to(render_dir).as_posix()
                for p in render_dir.rglob("*.png")}
    assert set(manifest["panels"]) == pictures
    assert len(pictures) == 4
    assert not (render_dir / "render-georef.json.lock").exists()


def test_the_first_root_frame_claimed_early_is_not_drawn_again(tmp_path):
    renderer = _Renderer()
    recorder = _Recorder()
    live = LiveProducts(_plan(tmp_path), report=recorder.report,
                        warn=recorder.warn, runner=renderer)
    first, valid = _frame(tmp_path, 1, 0)
    live.frame_committed(domain=1, valid_time=valid, path=first, draw=False)
    second, valid = _frame(tmp_path, 1, 1)
    live.frame_committed(domain=1, valid_time=valid, path=second)
    deadline = time.monotonic() + 30
    while not recorder.reports and time.monotonic() < deadline:
        time.sleep(0.02)
    live.stop(timeout=30)

    assert len(renderer.calls) == 1
    # Only the next frame is drawn; the claimed one rides along as the
    # baseline that closes its hour, and its pictures are not delivered.
    assert _drawn_frame(renderer.calls[0]) == second
    assert _context(renderer.calls[0]) == [first]


def test_a_nest_frame_before_the_root_first_never_overlaps_the_early_render(
        tmp_path):
    """One render at a time, whichever grid the writer commits first."""

    from gpuwm.first_products import FirstProducts

    renderer = _Renderer(delay=0.2)
    recorder = _Recorder()
    slot = threading.Lock()
    first = FirstProducts(_plan(tmp_path, "t2"), report=recorder.report,
                          warn=recorder.warn, runner=renderer, slot=slot)
    live = LiveProducts(_plan(tmp_path, "t2"), report=recorder.report,
                        warn=recorder.warn, first=first, runner=renderer,
                        slot=slot)
    nest, valid = _frame(tmp_path, 3, 0)
    live.frame_committed(domain=3, valid_time=valid, path=nest)
    time.sleep(0.05)  # the nest render is under way
    root, valid = _frame(tmp_path, 1, 0)
    assert first.frame_committed(domain=1, valid_time=valid, path=root)
    live.frame_committed(domain=1, valid_time=valid, path=root, draw=False)
    first.wait(30)
    live.stop(timeout=30)

    assert len(renderer.calls) == 2
    assert renderer.most == 1


def test_finalize_finishes_the_queue_and_takes_no_frame_after(tmp_path):
    renderer = _Renderer(delay=0.1)
    recorder = _Recorder()
    live = LiveProducts(_plan(tmp_path), report=recorder.report,
                        warn=recorder.warn, runner=renderer)
    for hour in range(4):
        frame, valid = _frame(tmp_path, 1, hour)
        live.frame_committed(domain=1, valid_time=valid, path=frame)
    summary = live.stop(timeout=30)
    late, valid = _frame(tmp_path, 1, 5)
    live.frame_committed(domain=1, valid_time=valid, path=late)
    time.sleep(0.2)

    assert summary["published"] == 4 and summary["dropped"] == 0
    assert renderer.running == 0 and len(renderer.calls) == 4


def test_a_queue_the_wait_cannot_finish_is_left_to_finalize(tmp_path, monkeypatch):
    monkeypatch.setattr(live_products, "_render_concurrency", lambda: 1)
    renderer = _Renderer(delay=0.3)
    recorder = _Recorder()
    live = LiveProducts(_plan(tmp_path), report=recorder.report,
                        warn=recorder.warn, runner=renderer)
    for hour in range(4):
        frame, valid = _frame(tmp_path, 1, hour)
        live.frame_committed(domain=1, valid_time=valid, path=frame)
    summary = live.stop(timeout=0.1)

    assert summary["published"] == 1 and summary["dropped"] == 3
    assert renderer.running == 0


def test_a_render_the_wait_gives_up_on_never_publishes_under_finalize(
        tmp_path, monkeypatch):
    """A wedged render must not move pictures in while finalize draws."""

    from gpuwm import live_products

    monkeypatch.setattr(live_products, "DEFAULT_WAIT_SECONDS", 0.1)
    release = threading.Event()
    inner = _Renderer()

    def wedged(command):
        release.wait(30)
        return inner(command)

    recorder = _Recorder()
    live = LiveProducts(_plan(tmp_path), report=recorder.report,
                        warn=recorder.warn, runner=wedged)
    frame, valid = _frame(tmp_path, 2, 0)
    live.frame_committed(domain=2, valid_time=valid, path=frame)
    summary = live.stop(timeout=0.1)
    assert summary["published"] == 0
    assert [code for code, _, _ in recorder.warnings] == ["live_products_timeout"]

    # Finalize now owns the folder; the late render finishes afterwards.
    release.set()
    live._thread.join(10)
    assert not live._thread.is_alive()
    png = tmp_path / "png"
    assert not list(png.glob("d02-3km/**/*.png"))
    assert not (png / live_products.LIVE_PRODUCTS_RECEIPT).exists()
    assert not (png / render_georef.GEOREF_FILENAME).exists()
    assert recorder.reports == [] and live.published == []


def _names(frames):
    return [Path(f).name for f in frames]


def test_windowed_pictures_are_drawn_once_per_grid_over_its_whole_hours(
        tmp_path):
    hourly = [_frame(tmp_path, 1, h)[0] for h in range(3)]
    frames = list(hourly)
    drawn = list(hourly)

    passes = live_products.windowed_passes(frames, drawn, None)
    assert len(passes) == 1
    wanted, context, products = passes[0]
    # The first frame ends no window; it stays a baseline.
    assert _names(wanted) == _names(hourly[1:])
    assert _names(context) == _names(hourly[:1])
    assert products == "windowed"
    # Frames the end-of-run batch draws itself, beside their baselines,
    # need no pass.
    assert live_products.windowed_passes(frames, hourly[:1], None) == []


def test_a_fifteen_minute_nest_has_its_windows_drawn_on_its_whole_hours(
        tmp_path):
    """The 3 km nest of a 12/3 km run had no qpf_1h at all (0 of 49).

    The domain wizard gives a nest 15-minute history, and the finalize
    pass left any such grid out, so its windows were never drawn.  They
    close on its whole hours, and those are the frames asked.
    """

    root = [_frame(tmp_path, 1, h)[0] for h in range(3)]
    nest = [_quarter(tmp_path, 2, minutes)[0] for minutes in range(0, 121, 15)]
    frames = root + nest

    passes = live_products.windowed_passes(frames, frames, "t2,qpf_1h",
                                           windowed_slugs=lambda f: {"qpf_1h"})
    by_grid = {_names(wanted)[0][:10]: (wanted, context, products)
               for wanted, context, products in passes}
    wanted, context, products = by_grid["wrfout_d02"]
    assert _names(wanted) == ["wrfout_d02_2026-09-25_13_00_00",
                              "wrfout_d02_2026-09-25_14_00_00"]
    assert sorted(_names([*wanted, *context])) == sorted(_names(nest))
    assert products == "qpf_1h"
    # A nest hour the live pass did not draw is drawn by the end-of-run
    # batch beside the nest's whole series (baseline_frames), so no pass
    # asks it again.
    missed = live_products.windowed_passes(frames, root, "t2,qpf_1h",
                                           windowed_slugs=lambda f: {"qpf_1h"})
    assert [_names(w) for w, _, _ in missed
            if _names(w)[0].startswith("wrfout_d02")] == []


def test_a_fifteen_minute_nest_finalize_batch_holds_every_frame_of_each_window(
        tmp_path):
    """A 15-minute nest's 1 h UH maximum read the last quarter hour only.

    The history writer resets ``UP_HELI_MAX`` and the other 1 h maxima at
    each write, so the frame on the hour holds only the 15 minutes before
    it.  The finalize pass drew the nest's windows over its whole-hour
    frames alone, and each hour's maximum was the last quarter hour's,
    under a note that called it the exact trailing hour.  Every frame of
    each window now reaches the renderer, which folds all of them.
    """

    nest = [_quarter(tmp_path, 3, minutes)[0] for minutes in range(0, 181, 15)]
    by_minutes = dict(zip(range(0, 181, 15), nest))

    for request in ("all", "uh_2to5km_1h_max,10m_wind_1h_max"):
        passes = live_products.windowed_passes(
            nest, nest, request,
            windowed_slugs=lambda f: {"uh_2to5km_1h_max", "10m_wind_1h_max"})
        assert len(passes) == 1
        wanted, context, _products = passes[0]
        batch = {path.name for path in [*wanted, *context]}
        assert _names(wanted) == ["wrfout_d03_2026-09-25_13_00_00",
                                  "wrfout_d03_2026-09-25_14_00_00",
                                  "wrfout_d03_2026-09-25_15_00_00"]
        for end in (60, 120, 180):
            window = [by_minutes[minutes].name
                      for minutes in range(end - 60, end + 1, 15)]
            missing = [name for name in window if name not in batch]
            assert missing == [], f"window ending +{end} min lacks {missing}"


def test_a_fifteen_minute_nest_draws_every_window_at_finalize(tmp_path):
    """The nest drew its rainfall windows only.

    Its 1 h and run maxima were left off because each whole-hour frame
    holds only its last quarter hour of ``UP_HELI_MAX``.  The engine now
    folds every frame inside a window, and the pass carries every frame
    of the grid, so the nest is asked for the same windows as its root.
    """

    root = [_frame(tmp_path, 1, h)[0] for h in range(3)]
    nest = [_quarter(tmp_path, 2, minutes)[0] for minutes in range(0, 121, 15)]
    frames = root + nest
    engine = lambda frame: frozenset({  # noqa: E731
        "qpf_1h", "qpf_total", "uh_2to5km_1h_max", "uh_2to5km_run_max"})

    def by_grid(products):
        return {_names(w)[0][:10]: p for w, _, p in live_products.windowed_passes(
            frames, frames, products, windowed_slugs=engine)}

    assert by_grid("all") == {"wrfout_d01": "windowed",
                              "wrfout_d02": "windowed"}
    assert by_grid("t2,uh_2to5km_1h_max") == {
        "wrfout_d01": "uh_2to5km_1h_max", "wrfout_d02": "uh_2to5km_1h_max"}


def test_a_nest_hour_whose_live_render_held_every_earlier_frame_is_skipped(
        tmp_path):
    nest = [_quarter(tmp_path, 2, minutes)[0] for minutes in range(0, 181, 15)]
    whole = [f for f in nest if f.name.endswith("_00_00")]
    # Each whole hour was drawn live beside every frame of the hour it
    # closes.
    held = {}
    for end in whole[1:]:
        at = nest.index(end)
        held[end.resolve()] = frozenset(f.resolve() for f in nest[at - 4:at + 1])

    passes = live_products.windowed_passes(nest, nest, "all", live_held=held)
    # F001's render held every frame up to it; F002 and F003 need the
    # series for their run maxima.
    assert [(_names(w), p) for w, _, p in passes] == [
        (_names(whole[2:]), "windowed")]
    assert sorted(_names(passes[0][1])) == sorted(
        _names([f for f in nest if f not in whole[2:]]))
    # Nothing drawn live: the end-of-run batch draws every frame beside
    # the whole series, and no pass is left.
    assert live_products.windowed_passes(nest, [], "all", live_held=held) == []


def test_a_group_request_skips_the_frames_whose_live_render_held_every_earlier(
        tmp_path):
    """The default request redrew every hour's windows at finalize."""

    hourly = [_frame(tmp_path, 1, h)[0] for h in range(4)]
    held = {hourly[h].resolve(): frozenset({hourly[h - 1].resolve(),
                                            hourly[h].resolve()})
            for h in range(1, 4)}
    passes = live_products.windowed_passes(hourly, hourly, "all",
                                           live_held=held)
    # F001's live render held F000 and F001, which is every frame the
    # series has up to it; F002 on need the series for longer windows.
    assert [(_names(w), _names(c), p) for w, c, p in passes] == [
        (_names(hourly[2:]), _names(hourly[:2]), "windowed")]
    # A frame not proven drawn is never skipped on its record.
    passes = live_products.windowed_passes(hourly, hourly[2:], "all",
                                           live_held=held)
    assert [_names(w) for w, _, _ in passes] == [_names(hourly[2:])]


def test_the_live_record_names_what_each_render_imported(tmp_path):
    render_dir = tmp_path / "png"
    render_dir.mkdir()
    a, b = tmp_path / "a", tmp_path / "b"
    (render_dir / LIVE_PRODUCTS_RECEIPT).write_text(json.dumps({
        "schema": live_products.LIVE_PRODUCTS_SCHEMA,
        "frames": [{"frame": str(b), "context": [str(a)], "written": []},
                   {"frame": str(a), "written": []}, "junk"]}),
        encoding="utf-8")
    assert live_products.live_held(render_dir) == {
        b.resolve(): frozenset({a.resolve(), b.resolve()}),
        a.resolve(): frozenset({a.resolve()})}
    assert live_products.live_held(tmp_path / "missing") == {}


def test_a_whole_hour_frame_is_drawn_beside_the_frame_that_closes_its_hour(
        tmp_path):
    """qpf_1h was drawn only after the forecast ended (98 min in run b).

    Each frame was drawn alone, and the engine draws no window on a
    one-frame store ("windowed accumulations need more than one stored
    whole-hour frame").
    """

    renderer = _Renderer()
    recorder = _Recorder()
    live = LiveProducts(_plan(tmp_path, "2m_temperature,qpf_1h"),
                        report=recorder.report, warn=recorder.warn,
                        runner=renderer)
    f000, valid = _frame(tmp_path, 1, 0)
    # The analysis frame is the early render's; it still closes F001.
    live.frame_committed(domain=1, valid_time=valid, path=f000, draw=False)
    f001, valid = _frame(tmp_path, 1, 1)
    live.frame_committed(domain=1, valid_time=valid, path=f001)
    f002, valid = _frame(tmp_path, 1, 2)
    live.frame_committed(domain=1, valid_time=valid, path=f002)
    nest = {}
    for minutes in (0, 15, 45, 60, 75):
        nest[minutes], valid = _quarter(tmp_path, 2, minutes)
        live.frame_committed(domain=2, valid_time=valid, path=nest[minutes])
    deadline = time.monotonic() + 30
    while len(recorder.reports) < 7 and time.monotonic() < deadline:
        time.sleep(0.02)
    live.stop(timeout=30)

    assert recorder.warnings == []
    launches = {}
    for command in renderer.calls:
        launches.setdefault(_drawn_frame(command).name, []).append(command)
    calls = {name: commands[-1] for name, commands in launches.items()}
    assert _context(calls[f002.name]) == [f001]
    assert "qpf_1h" in _products(calls[f002.name]).split(",")
    assert _context(calls[f001.name]) == [f000]
    # A 15-minute grid: its hour is closed by its previous whole hour
    # AND every frame between, whose interval maxima make up the hour;
    # a frame between hours is drawn alone.
    assert len(launches[f002.name]) == 1
    assert len(launches[nest[60].name]) == 1
    assert _context(calls[nest[60].name]) == [nest[0], nest[15], nest[45]]
    for minutes in (0, 15, 45, 75):
        assert _context(calls[nest[minutes].name]) == []
    # The record says which baselines each frame was drawn beside.
    receipt = json.loads((tmp_path / "png" / LIVE_PRODUCTS_RECEIPT)
                         .read_text(encoding="utf-8"))
    context = {Path(entry["frame"]).name: entry["context"]
               for entry in receipt["frames"]}
    assert context[f002.name] == [str(f001)]
    assert context[nest[60].name] == [str(nest[m]) for m in (0, 15, 45)]
    assert context[nest[15].name] == []


def test_a_baseline_gone_from_disk_leaves_the_frame_drawn_alone(tmp_path):
    renderer = _Renderer()
    recorder = _Recorder()
    live = LiveProducts(_plan(tmp_path), report=recorder.report,
                        warn=recorder.warn, runner=renderer)
    f000, valid = _frame(tmp_path, 1, 0)
    live.frame_committed(domain=1, valid_time=valid, path=f000, draw=False)
    f000.unlink()
    f001, valid = _frame(tmp_path, 1, 1)
    live.frame_committed(domain=1, valid_time=valid, path=f001)
    deadline = time.monotonic() + 30
    while not recorder.reports and time.monotonic() < deadline:
        time.sleep(0.02)
    live.stop(timeout=30)

    assert [_context(command) for command in renderer.calls] == [[]]
    assert len(recorder.reports) == 1


def _nest_hour(tmp_path, products, runner):
    """Draw a 15-minute grid's first hour live; ``(launches, recorder, nest)``."""

    recorder = _Recorder()
    live = LiveProducts(_plan(tmp_path, products), report=recorder.report,
                        warn=recorder.warn, runner=runner)
    nest = {}
    for minutes in (0, 15, 30, 45, 60):
        nest[minutes], valid = _quarter(tmp_path, 2, minutes)
        live.frame_committed(domain=2, valid_time=valid, path=nest[minutes])
    deadline = time.monotonic() + 30
    while (len(recorder.reports) + len(recorder.warnings) < 5
           and time.monotonic() < deadline):
        time.sleep(0.02)
    live.stop(timeout=30)
    launches = {}
    for command in getattr(runner, "calls", ()):
        launches.setdefault(_drawn_frame(command).name, []).append(command)
    return launches, recorder, nest


@pytest.mark.parametrize("request_", ["all", "2m_temperature,uh_2to5km_1h_max"])
def test_a_fifteen_minute_grid_closes_its_hour_in_one_command(tmp_path,
                                                               request_):
    """The nest's hour was drawn alone and again beside its previous whole
    hour for the rainfall windows only, so its 1 h UH maximum was never
    drawn as it landed.  The engine folds every frame of the hour, so the
    frame is drawn once, beside all of them, for the whole request."""

    launches, recorder, nest = _nest_hour(tmp_path, request_, _Renderer())
    assert recorder.warnings == []
    [closing] = launches[nest[60].name]
    assert _context(closing) == [nest[m] for m in (0, 15, 30, 45)]
    assert _products(closing) == request_
    receipt = json.loads((tmp_path / "png" / LIVE_PRODUCTS_RECEIPT)
                         .read_text(encoding="utf-8"))
    assert {Path(e["frame"]).name: e["context"]
            for e in receipt["frames"]}[nest[60].name] == [
        str(nest[m]) for m in (0, 15, 30, 45)]


class _Blank(_Renderer):
    """Draws nothing at all."""

    def __call__(self, command):
        with self._lock:
            self.calls.append(list(command))
        return subprocess.CompletedProcess(list(command), 1, "", "")


def test_a_nest_hour_that_drew_nothing_is_left_to_finalize(tmp_path):
    """A frame that drew no picture is not recorded, so finalize draws it."""

    launches, recorder, nest = _nest_hour(tmp_path, "all", _Blank())
    assert len(launches[nest[60].name]) == 1
    assert "live_products_empty" in [c for c, _, _ in recorder.warnings]
    assert recorder.reports == []


def test_a_request_is_windows_only_when_every_product_it_names_is_a_window(
        tmp_path):
    slugs = lambda: frozenset({"qpf_1h", "10m_wind_run_max"})  # noqa: E731

    def unasked():
        raise AssertionError("a keyword needs no listing")

    assert live_products.windows_only("windowed", unasked)
    assert live_products.windows_only("10m_wind_run_max,qpf_1h", slugs)
    assert live_products.windows_only("windowed,qpf_1h", slugs)
    for mixed in ("t2,qpf_1h", "all", "windowed,direct", None, "",
                  "qpf_1h,var:wrf_x"):
        assert not live_products.windows_only(mixed, slugs), mixed
    # With no listing to ask, every frame is drawn as it was.
    assert not live_products.windows_only("qpf_1h", lambda: frozenset())
    assert not live_products.windows_only("qpf_1h", None)

    # Windows end on whole hours after each grid's first frame.
    root = [_frame(tmp_path, 1, h)[0] for h in range(3)]
    nest = [_quarter(tmp_path, 2, minutes)[0] for minutes in range(0, 121, 15)]
    series = root + nest
    assert _names(f for f in series
                  if live_products.closes_a_window(f, series)) == [
        "wrfout_d01_2026-09-25_13_00_00", "wrfout_d01_2026-09-25_14_00_00",
        "wrfout_d02_2026-09-25_13_00_00", "wrfout_d02_2026-09-25_14_00_00"]


def test_a_request_made_only_of_windows_is_drawn_live_where_windows_end(
        tmp_path):
    """It drew nothing on the analysis frame or between the hours.

    Each of those renders exited 1 with no picture and the frame was said
    as left to the end of the run, which had nothing to draw there
    either.  They are not drawn, and are still the baselines of the
    hours they belong to.
    """

    renderer = _Renderer()
    recorder = _Recorder()
    live = LiveProducts(
        _plan(tmp_path, "qpf_1h,10m_wind_run_max"), report=recorder.report,
        warn=recorder.warn, runner=renderer,
        windowed_slugs=lambda: frozenset({"qpf_1h", "10m_wind_run_max"}))
    quarters = [_quarter(tmp_path, 2, minutes) for minutes in range(0, 121, 15)]
    nest = [frame for frame, _ in quarters]
    for frame, valid in quarters:
        live.frame_committed(domain=2, valid_time=valid, path=frame)
    deadline = time.monotonic() + 30
    while len(recorder.reports) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    live.stop(timeout=30)
    assert recorder.warnings == []
    calls = sorted(renderer.calls, key=_drawn_frame)
    assert [_drawn_frame(c) for c in calls] == [nest[4], nest[8]]
    assert [_context(c) for c in calls] == [nest[:4], nest[4:8]]

    # A request with an instant in it draws every frame.
    renderer = _Renderer()
    live = LiveProducts(
        _plan(tmp_path / "mixed", "t2,qpf_1h"), report=recorder.report,
        warn=recorder.warn, runner=renderer,
        windowed_slugs=lambda: frozenset({"qpf_1h"}))
    for frame, valid in quarters:
        live.frame_committed(domain=2, valid_time=valid, path=frame)
    live.stop(timeout=30)
    assert sorted(_drawn_frame(c) for c in renderer.calls) == nest


def test_finalize_hands_a_windows_only_batch_only_the_frames_windows_end_on(
        tmp_path, monkeypatch):
    """The batch over frames no window ends on drew nothing and exited 1,
    which stopped the render stage before the windowed pass."""

    from gpuwm import go_cli, render

    monkeypatch.setattr(render, "drawable_engine",
                        lambda: ("rust", "declared by the test"))
    nest = [_quarter(tmp_path, 2, minutes)[0] for minutes in range(0, 121, 15)]
    # Nothing drawn live but F001, and F002 left in the queue.
    monkeypatch.setattr(
        live_products, "published_frames",
        lambda frames, plan: ([f for f in frames if f != nest[4]], [nest[4]],
                              "drawn while it ran"))
    monkeypatch.setattr(live_products, "catalog_windowed_slugs",
                        lambda: frozenset({"qpf_1h", "10m_wind_run_max"}))
    monkeypatch.setattr(live_products, "engine_windowed_slugs",
                        lambda frame, **where: frozenset({"qpf_1h",
                                                          "10m_wind_run_max"}))
    commands = []
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: commands.append(list(command)))
    plan = {**_plan(tmp_path, "qpf_1h,10m_wind_run_max"),
            "wrfout_dir": tmp_path / "run" / "wrfout"}

    assert go_cli._render_stage(plan, explain=False)

    from gpuwm.cli import build_parser
    batch, windowed = [build_parser().parse_args(command[3:])
                       for command in commands]
    # F002, beside every other frame of the grid; nothing else is drawn.
    assert _names(batch.wrfout) == ["wrfout_d02_2026-09-25_14_00_00"]
    assert sorted(_names(batch.context_wrfout)) == sorted(_names(nest[:8]))
    # F001's own windows, over the whole series, as before.
    assert _names(windowed.wrfout) == ["wrfout_d02_2026-09-25_13_00_00"]


class _WindowRenderer(_Renderer):
    """Also files a ``qpf_1h`` picture whenever a baseline is imported."""

    def __call__(self, command):
        completed = super().__call__(command)
        if _context(command):
            out = Path(command[command.index("--out") + 1])
            frame = _drawn_frame(command)
            domain = int(frame.name.split("_")[1][1:])
            key = (f"d{domain:02d}-{_GRID_KM[domain]}km/qpf_1h/20260925/"
                   f"{frame.name}.png")
            (out / key).parent.mkdir(parents=True, exist_ok=True)
            (out / key).write_bytes(b"\x89PNG qpf " + frame.name.encode())
        return completed


def test_windows_the_live_pass_closed_are_not_drawn_again_at_finalize(
        tmp_path):
    renderer = _WindowRenderer()
    recorder = _Recorder()
    plan = _plan(tmp_path, "t2,qpf_1h")
    live = LiveProducts(plan, report=recorder.report, warn=recorder.warn,
                        runner=renderer)
    frames = []
    for hour in range(3):
        frame, valid = _frame(tmp_path, 1, hour)
        live.frame_committed(domain=1, valid_time=valid, path=frame)
        frames.append(frame)
    deadline = time.monotonic() + 30
    while len(recorder.reports) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    live.stop(timeout=30)
    remaining, drawn, _note = published_frames(frames, plan)
    assert remaining == []

    recorded = live_products.recorded_products(tmp_path / "png")
    assert recorded[frames[1].resolve()] == {"2m_temperature", "qpf_1h"}
    assert recorded[frames[0].resolve()] == {"2m_temperature"}
    engine = lambda frame: frozenset({"qpf_1h", "qpf_6h"})  # noqa: E731
    # Every window the request names was closed live: nothing to import.
    assert live_products.windowed_passes(frames, drawn, "t2,qpf_1h",
                                         windowed_slugs=engine,
                                         recorded=recorded) == []
    # A window the pair could not close is still drawn over the series.
    passes = live_products.windowed_passes(frames, drawn, "t2,qpf_1h,qpf_6h",
                                           windowed_slugs=engine,
                                           recorded=recorded)
    assert [(_names(w), _names(c), p) for w, c, p in passes] == [
        (_names(frames[1:]), _names(frames[:1]), "qpf_1h,qpf_6h")]
    # A frame missing one: only that frame is asked again.
    thinned = {**recorded, frames[2].resolve(): frozenset({"2m_temperature"})}
    passes = live_products.windowed_passes(frames, drawn, "t2,qpf_1h",
                                           windowed_slugs=engine,
                                           recorded=thinned)
    assert [(_names(w), _names(c)) for w, c, _ in passes] == [
        (_names(frames[2:]), _names(frames[:2]))]
    # A group keyword promises no one product: its pass is kept.
    assert live_products.windowed_passes(frames, drawn, "all",
                                         windowed_slugs=engine,
                                         recorded=recorded) != []


def _picture(tmp_path, grid, product, name="a.png"):
    path = tmp_path / "png" / grid / product / "2026-09-25" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")


def test_the_closing_note_names_a_product_a_grid_has_no_picture_of(tmp_path):
    """The nest's missing qpf_1h hid behind a note read as a first-frame skip."""

    for grid in ("d01-12km", "d02-3km"):
        _picture(tmp_path, grid, "2m_temperature")
        _picture(tmp_path, grid, "total_qpf")
    _picture(tmp_path, "d01-12km", "qpf_1h")

    _picture(tmp_path, "d01-12km", "uh_2to5km_1h_max")

    note = live_products.unpictured_note(
        tmp_path / "png", "t2,qpf_1h,cloud_cover,precip,var:wrf_x,xsec:t")
    assert note is not None and note.startswith("note: ")
    assert "no picture of qpf_1h on d02-3km." in note
    # A product no grid drew is each render's own skip line, with the
    # engine's reason; the run's closing note from the render summary
    # names it too, and a third line would say it again.
    named = note.split(".  The engine's reason")[0]
    for absent in ("cloud_cover", "t2", "precip", "total_qpf", "var:",
                   "xsec:", "d01-12km", "uh_2to5km"):
        assert absent not in named
    # A group keyword names no one product, so every product one grid
    # drew is asked of the others: the default request now says it.
    grouped = live_products.unpictured_note(tmp_path / "png", "all")
    assert "no picture of qpf_1h, uh_2to5km_1h_max on d02-3km." in grouped
    assert live_products.unpictured_note(tmp_path / "png", "t2,precip") is None
    assert live_products.unpictured_note(tmp_path / "missing", "t2") is None


def test_finalize_draws_the_nest_windows_and_names_what_is_missing(
        tmp_path, monkeypatch, capsys):
    """End of run on a 12/3 km run whose every frame was drawn live."""

    from gpuwm import go_cli, render

    monkeypatch.setattr(render, "drawable_engine",
                        lambda: ("rust", "declared by the test"))
    root = [_frame(tmp_path, 1, h)[0] for h in range(3)]
    nest = [_quarter(tmp_path, 2, minutes)[0] for minutes in range(0, 121, 15)]
    frames = root + nest
    monkeypatch.setattr(
        live_products, "published_frames",
        lambda frames, plan: ([], list(frames), "drawn while it ran"))
    # The render stage asks with the store's place and token as keywords.
    monkeypatch.setattr(live_products, "engine_windowed_slugs",
                        lambda frame, **where: frozenset({"qpf_1h"}))
    for grid in ("d01-12km", "d02-3km"):
        _picture(tmp_path, grid, "2m_temperature")
    _picture(tmp_path, "d01-12km", "qpf_1h")
    commands = []
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: commands.append(list(command)))
    plan = {**_plan(tmp_path, "t2,qpf_1h"), "wrfout_dir": tmp_path / "run" / "wrfout"}

    assert go_cli._render_stage(plan, explain=False)

    from gpuwm.cli import build_parser
    requests = [build_parser().parse_args(command[3:]) for command in commands]
    nest_passes = [r for r in requests
                   if r.wrfout and r.wrfout[0].name.startswith("wrfout_d02")]
    assert len(nest_passes) == 1
    assert _names(nest_passes[0].wrfout) == ["wrfout_d02_2026-09-25_13_00_00",
                                              "wrfout_d02_2026-09-25_14_00_00"]
    assert _names(nest_passes[0].context_wrfout) == [
        "wrfout_d02_2026-09-25_12_00_00", "wrfout_d02_2026-09-25_12_15_00",
        "wrfout_d02_2026-09-25_12_30_00", "wrfout_d02_2026-09-25_12_45_00",
        "wrfout_d02_2026-09-25_13_15_00", "wrfout_d02_2026-09-25_13_30_00",
        "wrfout_d02_2026-09-25_13_45_00"]
    assert nest_passes[0].products == "qpf_1h"
    printed = capsys.readouterr().out
    assert "no picture of qpf_1h on d02-3km" in printed


def test_finalize_on_the_default_request_draws_every_nest_window(
        tmp_path, monkeypatch, capsys):
    from gpuwm import go_cli, render

    monkeypatch.setattr(render, "drawable_engine",
                        lambda: ("rust", "declared by the test"))
    root = [_frame(tmp_path, 1, h)[0] for h in range(3)]
    nest = [_quarter(tmp_path, 2, minutes)[0] for minutes in range(0, 121, 15)]
    monkeypatch.setattr(
        live_products, "published_frames",
        lambda frames, plan: ([], list(frames), "drawn while it ran"))
    asked = []

    def listing(frame, **_):
        asked.append(frame)
        return frozenset({"qpf_1h", "uh_2to5km_1h_max"})

    monkeypatch.setattr(live_products, "engine_windowed_slugs", listing)
    for grid in ("d01-12km", "d02-3km"):
        _picture(tmp_path, grid, "2m_temperature")
    _picture(tmp_path, "d01-12km", "uh_2to5km_1h_max")
    commands = []
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: commands.append(list(command)))
    plan = {**_plan(tmp_path, "all"), "wrfout_dir": tmp_path / "run" / "wrfout"}

    assert go_cli._render_stage(plan, explain=False)

    from gpuwm.cli import build_parser
    requests = {build_parser().parse_args(c[3:]).wrfout[0].name[:10]:
                build_parser().parse_args(c[3:]) for c in commands}
    # The nest is asked for every window, as its root is; it drew its
    # rainfall windows only before the engine folded every frame.
    assert requests["wrfout_d02"].products == "windowed"
    assert requests["wrfout_d01"].products == "windowed"
    assert len(requests["wrfout_d02"].context_wrfout) == 7
    # A group request needs no listing to decide anything.
    assert asked == []
    printed = capsys.readouterr().out
    assert "no picture of uh_2to5km_1h_max on d02-3km" in printed


def test_a_named_list_draws_only_its_windowed_members_at_finalize(tmp_path):
    """No picture drawn live is requested again by the windowed pass."""

    hourly = [_frame(tmp_path, 1, h)[0] for h in range(3)]
    asked = []

    def engine(frame):
        asked.append(frame)
        return frozenset({"qpf_1h", "qpf_total", "uh_2to5km_run_max"})

    passes = live_products.windowed_passes(hourly, hourly, "t2,qpf_1h",
                                           windowed_slugs=engine)
    assert [products for *_, products in passes] == ["qpf_1h"]
    assert len(asked) == 1
    # No windowed member: no pass, and nothing is imported to find out
    # beyond the one catalog listing.
    assert live_products.windowed_passes(hourly, hourly, "t2,refc",
                                         windowed_slugs=engine) == []
    # A group keyword needs no listing at all.
    asked.clear()
    passes = live_products.windowed_passes(hourly, hourly, "direct,windowed",
                                           windowed_slugs=engine)
    assert [products for *_, products in passes] == ["windowed"]
    assert asked == []
    assert live_products.windowed_passes(hourly, hourly, "direct",
                                         windowed_slugs=engine) == []
    # A catalog that cannot be had keeps the request (time, not pictures).
    assert live_products.windowed_passes(
        hourly, hourly, "t2,qpf_1h",
        windowed_slugs=lambda frame: frozenset())[0][2] == "t2,qpf_1h"


def test_the_end_of_run_batch_takes_baselines_only_from_its_own_grid(
        tmp_path):
    hourly = [_frame(tmp_path, 1, h)[0] for h in range(3)]
    nest = [_frame(tmp_path, 2, h)[0] for h in range(3)]
    quarter = []
    for minutes in (0, 15, 30):
        path = (tmp_path / "run" / "wrfout" /
                f"wrfout_d03_2026-09-25_12_{minutes:02d}_00")
        path.write_bytes(b"x")
        quarter.append(path)
    drawn = hourly[:2] + nest + quarter[:2]

    # d01 misses one frame: its own drawn frames are its baselines, and
    # no other grid's.
    assert (_names(live_products.baseline_frames([hourly[2]], drawn))
            == _names(hourly[:2]))
    # A frame between hours closes no window: nothing goes beside it.
    assert live_products.baseline_frames([quarter[2]], drawn) == []
    assert live_products.baseline_frames([], drawn) == []
    # A d03 whole hour left to draw takes every drawn d03 frame, the
    # frames between its hours included, and no other grid's.
    hour, _ = _quarter(tmp_path, 3, 60)
    assert (_names(live_products.baseline_frames([quarter[2], hour], drawn))
            == _names(quarter[:2]))


def test_a_nest_hour_left_to_the_batch_is_drawn_beside_the_whole_series(
        tmp_path):
    """A 15-minute nest's F001 and F002 left to the end-of-run batch alone
    reached the engine as an hourly store, which drew each 1 h maximum
    from its last quarter hour, and beside frames that did not cover their
    hours they were refused ("missing stored frame(s)").  The batch now
    holds the nest's whole series, so each window folds every frame."""

    nest = [_quarter(tmp_path, 2, minutes)[0] for minutes in range(0, 121, 15)]
    whole = [f for f in nest if f.name.endswith("_00_00")]
    between = [f for f in nest if f not in whole]
    drawn = [f for f in nest if f not in whole[1:]]
    assert _names(live_products.baseline_frames(whole[1:], drawn)) == _names(drawn)
    assert _names(live_products.baseline_frames(whole[2:], drawn)) == _names(drawn)
    left = [between[-1], whole[2]]
    rest = [f for f in nest if f not in left]
    assert _names(live_products.baseline_frames(left, rest)) == _names(rest)
    # Frames between hours close no window: nothing is imported for them.
    assert live_products.baseline_frames(
        between[-2:], [f for f in nest if f not in between[-2:]]) == []


def test_finalize_skips_only_frames_whose_record_still_holds(tmp_path):
    renderer = _Renderer()
    recorder = _Recorder()
    plan = _plan(tmp_path, "t2")
    live = LiveProducts(plan, report=recorder.report, warn=recorder.warn,
                        runner=renderer)
    drawn = []
    for domain in (1, 2, 3):
        frame, valid = _frame(tmp_path, domain, 0)
        live.frame_committed(domain=domain, valid_time=valid, path=frame)
        drawn.append(frame)
    deadline = time.monotonic() + 30
    while len(recorder.reports) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    live.stop(timeout=30)
    missed, _ = _frame(tmp_path, 3, 1)

    remaining, already, note = published_frames([*drawn, missed], plan)
    assert remaining == [missed] and already == drawn
    assert "3 frame(s) were drawn while the forecast ran" in note

    # A picture edited after it was recorded voids that frame only.
    receipt = json.loads((tmp_path / "png" / LIVE_PRODUCTS_RECEIPT)
                         .read_text(encoding="utf-8"))
    record = next(row for row in receipt["frames"] if row["domain"] == 2)
    edited = tmp_path / "png" / record["written"][0]["name"]
    edited.write_bytes(b"changed")
    remaining, already, note = published_frames([*drawn, missed], plan)
    assert remaining == [drawn[1], missed]
    assert "no longer match" in note

    # A different product spec voids the whole record.
    remaining, already, _ = published_frames(drawn, _plan(tmp_path, "all"))
    assert remaining == drawn and already == []


def test_the_observer_draws_every_grid_and_says_so_on_the_stream(tmp_path):
    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events, root_domain=1)
    observer.arm_first_products(_plan(tmp_path))
    # No products named: the first-frame render stays off, live is on.
    assert observer.first_products is None
    live = observer.live_products
    assert live is not None
    live._runner = _Renderer()
    for domain in (1, 2):
        frame, valid = _frame(tmp_path, domain, 0)
        observer.output_committed(domain=domain, valid_time=valid, path=frame)
    deadline = time.monotonic() + 30
    while len(live.published) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    observer.stop_live_products()
    events.close()

    records = read_events(tmp_path / "events.jsonl")
    ready = [r for r in records if r["event"] == "live_products_ready"]
    assert sorted(r["domain"] for r in ready) == [1, 2]
    assert all(r["pictures"] == 1 for r in ready)
    assert "live_products_ready" in EVENT_TAGS
    for code in ("live_products_empty", "live_products_failed",
                 "live_products_timeout"):
        assert code in WARNING_CODES


def test_a_run_that_asked_for_no_pictures_draws_none_as_it_runs(tmp_path):
    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events, root_domain=1)
    observer.arm_first_products(_plan(tmp_path, "none"))
    assert observer.live_products is None
    events.close()


def test_progress_names_each_grid_with_its_own_step_wall(tmp_path):
    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events, root_domain=1)
    observer(model_elapsed_seconds=60.0, outer_step=1, step_wall_seconds=2.0,
             domain_clocks={1: 60.0, 2: 60.0, 3: 60.0},
             domain_step_wall={1: 0.2, 2: 0.6, 3: 1.1})
    events.close()
    record = [r for r in read_events(tmp_path / "events.jsonl")
              if r["event"] == "model_progress"][0]
    assert record["domains"] == [
        {"domain": 1, "model_seconds": 60.0, "step_wall_seconds": 0.2},
        {"domain": 2, "model_seconds": 60.0, "step_wall_seconds": 0.6},
        {"domain": 3, "model_seconds": 60.0, "step_wall_seconds": 1.1}]


def _resolved(domains):
    return {"event": "resolved_plan", "configuration": {"experiment": {
        "run_seconds": 7200.0, "start_time": "2026-09-25T12:00:00",
        "domains": [{"grid_id": gid, "run": {"dx": dx}}
                    for gid, dx in domains]}}}


def test_the_page_names_every_grid_and_the_one_that_sets_the_pace(tmp_path):
    log = tmp_path / "events.jsonl"
    rows = [_resolved([(1, 12000.0), (2, 3000.0), (3, 1000.0)]),
            {"event": "model_progress", "model_seconds": 600.0,
             "speed_x": 21.2, "domains": [
                 {"domain": 1, "model_seconds": 600.0, "step_wall_seconds": 3.0},
                 {"domain": 2, "model_seconds": 600.0, "step_wall_seconds": 9.0},
                 {"domain": 3, "model_seconds": 600.0, "step_wall_seconds": 17.0}]}]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    grids = gui_runs.pace(gui_runs.event_facts(log))
    assert grids == {"grids_km": [12.0, 3.0, 1.0], "grids_label": "12 / 3 / 1 km",
                     "pace_km": 1.0, "pace_label": "1 km"}


def test_a_one_grid_run_names_its_grid_and_no_pace(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text(json.dumps(_resolved([(1, 750.0)])) + "\n", encoding="utf-8")
    grids = gui_runs.pace(gui_runs.event_facts(log))
    assert grids["grids_label"] == "0.75 km" and grids["pace_km"] is None


def test_watching_can_pick_each_grid_while_it_is_drawn(tmp_path):
    for folder, name in (("d01-12km", "a.png"), ("d02-3km", "b.png"),
                         ("d03-1km", "c.png"), ("d03-1km", "d.png")):
        path = tmp_path / "png" / folder / "2m_temperature" / "20260925" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    index = gui_frames.index(tmp_path)
    assert index["domains"] == ["d01-12km", "d02-3km", "d03-1km"]
    assert index["by_domain"]["d03-1km"]["count"] == 2
    assert index["by_domain"]["d02-3km"]["watch"] == \
        "png/d02-3km/2m_temperature/20260925/b.png"


def test_a_batch_is_rekeyed_to_where_its_pictures_were_filed_and_merged(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    drawn = out / "rustwx_t2_d02.png"
    drawn.write_bytes(b"png")
    filed = out / "d02-3km" / "t2" / "20260925" / "t2.png"
    earlier = out / "d01-12km" / "t2" / "20260925" / "t2.png"
    earlier.parent.mkdir(parents=True)
    earlier.write_bytes(b"png")
    prior = {"schema": render_georef.GEOREF_SCHEMA, "generated_utc": "a",
             "panels": {"d01-12km/t2/20260925/t2.png": {"p": 1},
                        "gone/x.png": {"p": 0}},
             "without_georeference": []}
    # The engine REPLACED the folder's file with its own batch, flat-keyed.
    (out / render_georef.GEOREF_FILENAME).write_text(json.dumps({
        "schema": render_georef.GEOREF_SCHEMA, "generated_utc": "b",
        "panels": {"rustwx_t2_d02.png": {"p": 2}},
        "without_georeference": []}), encoding="utf-8")

    def place(png):
        filed.parent.mkdir(parents=True, exist_ok=True)
        os.replace(png, filed)
        return filed

    assert render_georef.file_pictures(out, [drawn], place,
                                       prior=prior) == [filed]
    merged = render_georef.read(out / render_georef.GEOREF_FILENAME)
    assert merged["panels"] == {"d01-12km/t2/20260925/t2.png": {"p": 1},
                                "d02-3km/t2/20260925/t2.png": {"p": 2}}


def test_a_picture_drawn_again_moves_between_the_halves(tmp_path):
    (tmp_path / "a.png").write_bytes(b"png")
    base = {"panels": {"a.png": {"p": 1}}, "without_georeference": []}
    batch = {"panels": {}, "without_georeference": [
        {"path": "a.png", "reason": "withheld"}]}
    merged = render_georef.merge(base, batch, root=tmp_path)
    assert merged["panels"] == {}
    assert merged["without_georeference"] == [{"path": "a.png", "reason": "withheld"}]


def test_a_context_series_draws_only_the_one_frame_it_delivers(monkeypatch):
    """Drawing and discarding every baseline made one frame cost them all."""

    from gpuwm import render

    stamps = {Path(f"f{i}"): (None, (_START + timedelta(hours=i),))
              for i in range(6)}
    monkeypatch.setattr(render, "_history_series_record",
                        lambda path: stamps[Path(path)])
    series = list(stamps)
    assert render._wanted_slots(series, {_START + timedelta(hours=5)}) == ["5"]
    # Two or more wanted: each launch imports the whole series again,
    # which costs more than drawing the baselines once.
    two = {_START + timedelta(hours=i) for i in (4, 5)}
    assert render._wanted_slots(series, two) is None


def test_a_renderer_with_no_map_assets_is_told_to_the_run_once(tmp_path,
                                                               monkeypatch):
    """THE DEFECT: the every-frame render drew 22,734 pictures of one run
    with no coastlines, borders or state lines and no event said so.  The
    render subprocess printed its warning into output this module drops.
    The run is told now, once, however many frames are drawn."""

    from gpuwm import render, rustwx
    from test_render_basemap_delivery import wheel_install

    wheel_install(tmp_path, monkeypatch)
    # The companion removed or edited.  ``raising=False`` so this also runs
    # against a tree from before the companion carried map assets, where
    # it fails on the missing event rather than on the fixture.
    monkeypatch.setattr(rustwx, "companion_basemap_dir", lambda: None,
                        raising=False)
    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events, root_domain=1)
    observer.arm_first_products(_plan(tmp_path))
    live = observer.live_products
    live._runner = _Renderer()
    for hour in range(2):
        for domain in (1, 2):
            frame, valid = _frame(tmp_path, domain, hour)
            observer.output_committed(domain=domain, valid_time=valid,
                                      path=frame)
    deadline = time.monotonic() + 30
    while len(live.published) < 4 and time.monotonic() < deadline:
        time.sleep(0.02)
    observer.stop_live_products()
    events.close()

    records = read_events(tmp_path / "events.jsonl")
    told = [r for r in records if r["event"] == "warning"
            and r.get("code") == render.BASEMAP_MISSING_CODE]
    assert len(told) == 1, told
    assert "no coastlines, borders or state lines" in told[0]["message"]
    assert told[0]["remedy"].startswith("pip install --force-reinstall "
                                        "gpuwm-data")
    assert told[0]["render_stage"] == "as-drawn"
    # The pictures are still drawn: a map with no coastline is reported,
    # never withheld.
    assert len(live.published) == 4
    assert render.BASEMAP_MISSING_CODE in WARNING_CODES
    # And the page reading this run's log shows it.
    assert gui_runs.event_facts(tmp_path / "events.jsonl")["basemap_missing"]


# ---------------------------------------------------------------------------
# The real renderer
# ---------------------------------------------------------------------------


def _this_tree_s_renderer():
    """This tree's rw_wrfbatch, or a skip naming why there is none."""

    from gpuwm import rustwx
    from gpuwm.render import renderer_refusal

    renderer = rustwx.find_renderer()
    if renderer is None:
        pytest.skip("rust renderer not built (cd tools/rustwx && cargo "
                    "build --release --locked --offline)")
    reason = renderer_refusal(renderer)
    if reason is not None:
        pytest.skip(f"the renderer at {renderer} is not this tree's: {reason}")
    return renderer


def _wind_history(folder: Path, step_minutes: int = 15, hours: int = 2,
                  *, analysis_reflectivity: bool = True
                  ) -> list[tuple[Path, datetime]]:
    """A 3 km nest's history every ``step_minutes``, one real wrfout per frame.

    Written by the project's own writer with the production global
    attributes, as a forecast writes them.  The 10 m wind blows hardest
    in the first hour, so a run maximum folded from the second hour
    alone reads low on every cell.  ``analysis_reflectivity=False``
    writes the first frame without ``REFL_10CM``, as a forecast's
    analysis frame is written.
    """

    import numpy as np
    from types import SimpleNamespace

    from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs, wrfout_filename

    nz, ny, nx = 4, 12, 16
    start = datetime(2026, 9, 25, 18)
    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(grid, start, grid_id=2, parent_id=1,
                             i_parent_start=5, j_parent_start=5,
                             parent_grid_ratio=3, dt=6.0)
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(60 // step_minutes * hours + 1):
        minutes = step_minutes * index
        valid = start + timedelta(minutes=minutes)
        rng = np.random.default_rng(100 + index)
        strong = 20.0 if 0 < minutes <= 60 else 4.0
        fields = {
            "T": np.zeros((nz, ny, nx), np.float32),
            "MU": np.zeros((ny, nx), np.float32),
            "REFL_10CM": rng.uniform(-20.0, 65.0,
                                     (nz, ny, nx)).astype(np.float32),
            "T2": rng.uniform(280.0, 300.0, (ny, nx)).astype(np.float32),
            "Q2": rng.uniform(0.004, 0.012, (ny, nx)).astype(np.float32),
            "PSFC": rng.uniform(96000.0, 98000.0,
                                (ny, nx)).astype(np.float32),
            "U10": rng.uniform(0.5 * strong, strong,
                               (ny, nx)).astype(np.float32),
            "V10": rng.uniform(-2.0, 2.0, (ny, nx)).astype(np.float32),
            "RAINC": np.full((ny, nx), 0.1 * index, np.float32),
            "RAINNC": np.full((ny, nx), 0.4 * index, np.float32),
            "XLAT": np.tile(np.linspace(38.0, 40.0, ny)[:, None],
                            (1, nx)).astype(np.float32),
            "XLONG": np.tile(np.linspace(-98.0, -95.0, nx)[None, :],
                             (ny, 1)).astype(np.float32),
            "HGT": np.zeros((ny, nx), np.float32),
            "SINALPHA": np.zeros((ny, nx), np.float32),
            "COSALPHA": np.ones((ny, nx), np.float32),
        }
        if index == 0 and not analysis_reflectivity:
            del fields["REFL_10CM"]
        path = folder / wrfout_filename(valid, domain_id=2)
        with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0,
                          global_attrs=attrs) as writer:
            writer.write_frame(valid.strftime("%Y-%m-%d_%H:%M:%S"), fields)
        frames.append((path, valid))
    return frames


def _pictures(root: Path, product: str) -> dict[str, str]:
    from gpuwm.fetch import sha256_file

    return {p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted(root.rglob("*.png"))
            if p.relative_to(root).parts[1] == product}


_WIND_MAXIMA = "10m_wind_run_max,10m_wind_1h_max"


@pytest.mark.parametrize("products,step_minutes", [
    (_WIND_MAXIMA, 15),
    # The Wind preset's shape: an instant beside the two maxima.
    ("10m_wind_speed_and_direction," + _WIND_MAXIMA, 15),
    (_WIND_MAXIMA, 60),
])
def test_a_named_wind_request_ends_with_the_series_pictures(
        tmp_path, products, step_minutes):
    """The live pass and finalize deliver what one render of the series draws.

    THE DEFECT: the live pass draws each whole hour of a sub-hourly grid
    beside that hour's frames only.  The engine folded the 10 m wind run
    maximum at F002 from +1:00 to +2:00 alone (a lower bound of the wrong
    frames), the live record then held that picture, and finalize, seeing
    every product of the named request drawn at F002, never drew it
    again: the delivered picture read low on every cell this history
    blows hardest in the first hour.  The engine now refuses a maximum
    whose frames are missing, and finalize draws it over the series.

    AND: a request made only of windows drew nothing on the analysis
    frame or between the hours, each such live render exited 1 and was
    said as left to the end of the run, and the end-of-run batch over
    those frames exited 1 too, which stopped the render stage
    (GoStageFailed) before the windowed pass drew F002 at all.  That was
    so on an hourly history as well.
    """

    import subprocess

    from gpuwm import go_cli, render
    from gpuwm.go_cli import _stage_cwd, _stage_env, render_command

    _this_tree_s_renderer()
    wrfout = tmp_path / "run" / "wrfout"
    frames = _wind_history(wrfout, step_minutes)
    plan = {"run": tmp_path / "run", "render": tmp_path / "png",
            "render_products": products, "wrfout_dir": wrfout}

    recorder = _Recorder()
    live = LiveProducts(plan, report=recorder.report, warn=recorder.warn)
    for path, valid in frames:
        live.frame_committed(domain=2, valid_time=valid, path=path)
    live.stop(timeout=600.0)
    # A build without the map assets says so once and still draws.
    assert [warning[:2] for warning in recorder.warnings
            if warning[0] != render.BASEMAP_MISSING_CODE] == []
    held = live_products.recorded_products(tmp_path / "png")
    last = frames[-1][0].resolve()
    assert "10m_wind_1h_max" in held[last]
    if step_minutes < 60:
        assert "10m_wind_run_max" not in held[last], (
            "the live render held one hour and still drew the run maximum")

    assert go_cli._render_stage(plan, explain=False)

    series = tmp_path / "series"
    completed = subprocess.run(
        render_command({**plan, "render": series},
                       [path for path, _ in frames]),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        errors="replace", cwd=str(_stage_cwd()), env=_stage_env())
    assert completed.returncode == 0, completed.stderr
    for product in _WIND_MAXIMA.split(","):
        expected = _pictures(series, product)
        # At F001 and F002.
        assert len(expected) == 2, (product, expected)
        assert any("f002" in name for name in expected), expected
        assert _pictures(tmp_path / "png", product) == expected, product
    if "10m_wind_speed" in products:
        assert len(_pictures(tmp_path / "png", "10m_wind_speed_and_direction")
                   ) == len(frames)


# -- a whole hour drawn beside an analysis frame that stores no REFL_10CM --
#
# Measured on a 3 km CONUS run with a 10-minute history, asked for
# composite_reflectivity,2m_temperature,10m_wind_speed_and_direction: the
# lead-1h frame was drawn beside 21:00 to 21:50 as baselines although no
# window was asked for (76.8 s against 13.8 s for every other frame), and
# the renderer's catalog of that series was the catalog of its first
# frame, the analysis, which stores no REFL_10CM.  The frame was published
# with wind and temperature only; leads 2 to 6 h were whole.

_NO_WINDOW = "composite_reflectivity,2m_temperature,10m_wind_speed_and_direction"


def _drawn_live(tmp_path, products):
    """One real hour every 10 minutes, drawn live; ``(frames, plan, receipt)``."""

    from gpuwm import render

    _this_tree_s_renderer()
    wrfout = tmp_path / "run" / "wrfout"
    frames = _wind_history(wrfout, 10, hours=1, analysis_reflectivity=False)
    plan = {"run": tmp_path / "run", "render": tmp_path / "png",
            "render_products": products, "wrfout_dir": wrfout}
    recorder = _Recorder()
    live = LiveProducts(plan, report=recorder.report, warn=recorder.warn)
    for path, valid in frames:
        live.frame_committed(domain=2, valid_time=valid, path=path)
    live.stop(timeout=600.0)
    assert [warning[:2] for warning in recorder.warnings
            if warning[0] != render.BASEMAP_MISSING_CODE] == []
    receipt = live_products.read_receipt(tmp_path / "png")
    assert receipt is not None
    return [path for path, _ in frames], plan, receipt


def _held(receipt) -> dict[str, frozenset[str]]:
    held = {}
    for entry in receipt["frames"]:
        held[Path(entry["frame"]).name] = frozenset(
            Path(picture["name"]).parts[-3] for picture in entry["written"])
    return held


def test_a_request_with_no_window_draws_every_frame_alone_and_whole(tmp_path):
    frames, _, receipt = _drawn_live(tmp_path, _NO_WINDOW)
    held = _held(receipt)
    lead_1h = frames[-1].name
    assert lead_1h.endswith("_19_00_00")
    assert "composite_reflectivity" in held[lead_1h], held[lead_1h]
    for frame in frames[1:]:
        assert held[frame.name] == {
            "composite_reflectivity", "2m_temperature",
            "10m_wind_speed_and_direction"}, frame.name
    # The analysis frame stores none.
    assert "composite_reflectivity" not in held[frames[0].name]
    # No window was asked for, so no frame imported an earlier one.
    assert [entry["context"] for entry in receipt["frames"]] == [
        [] for _ in frames]


@pytest.mark.parametrize("products", [
    "composite_reflectivity,2m_temperature,qpf_1h", "all"])
def test_a_whole_hour_drawn_beside_its_hour_keeps_its_own_reflectivity(
        tmp_path, products):
    """The baselines a window needs are imported, and an instant product
    the whole hour stores is drawn from it whatever they store."""

    frames, _, receipt = _drawn_live(tmp_path, products)
    held = _held(receipt)
    lead_1h = frames[-1].name
    closing = [entry for entry in receipt["frames"]
               if Path(entry["frame"]).name == lead_1h]
    assert [Path(path).name for path in closing[0]["context"]] == [
        frame.name for frame in frames[:-1]]
    assert "qpf_1h" in held[lead_1h], held[lead_1h]
    assert "composite_reflectivity" in held[lead_1h], held[lead_1h]


def test_a_run_rendered_from_its_analysis_frame_draws_every_reflectivity(
        tmp_path):
    """The end-of-run batch over the whole series, nothing drawn live."""

    from gpuwm import go_cli

    _this_tree_s_renderer()
    wrfout = tmp_path / "run" / "wrfout"
    frames = [path for path, _ in
              _wind_history(wrfout, 10, hours=1, analysis_reflectivity=False)]
    plan = {"run": tmp_path / "run", "render": tmp_path / "png",
            "render_products": "composite_reflectivity,2m_temperature",
            "wrfout_dir": wrfout}

    assert go_cli._render_stage(plan, explain=False)

    drawn = _pictures(tmp_path / "png", "composite_reflectivity")
    assert len(drawn) == len(frames) - 1, sorted(drawn)
    assert len(_pictures(tmp_path / "png", "2m_temperature")) == len(frames)
