"""A run that SUCCEEDED leaves no render working store behind.

MEASURED on the 2026-09-26 user sweep (finding F7), two Windows runs of
the 3 km GFS forecast: run b ended PASS with 741 MB of hour files in
``png.render-scratch/rwstore-<token>-*`` and 30 MB in
``run/wrfout.render-scratch/rwstore-*``, and the nested run c kept 50 MB
plus 3.9 MB.  About 0.8 GB per 24 h forecast, growing with the domain,
with no line in the run's events saying so.

Two faults made it, and both are pinned here:

* The removal could not reach the engine's hour files.  The engine files
  one hour at ``<store>/wrf/local_<init>_<64 hex>_<profile>_science_v1/
  f000.rws``, 147 characters below the store, so on that run (a
  104-character run folder) the full path was 294 to 296 characters.
  Windows refuses a path past 260 characters through the ordinary API
  unless long paths are switched on for the whole machine (they were
  not), so ``shutil.rmtree`` failed on every one of its retries with
  WinError 145, whatever it waited for.  The engine itself writes them
  through the extended-length spelling, which is why they exist at all.
* Nothing swept after a success.  The door that owns a render stage
  swept that stage's stores only when the stage FAILED, so a store the
  render could not remove stayed for good, and the render subprocess's
  own warning died with its captured stderr.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gpuwm import chain_events, go_cli, live_products, render, runplan, rustwx
from gpuwm.render_layout import fs_path

#: The engine's hour-file layout inside a working store, with the same
#: component lengths ``rw-wrfbatch`` writes: ``local_<YYYYMMDDHHMMSS>_
#: <sha256 hex>_<profile>_science_v1``, the profile as a full render
#: names it.  MEASURED on the engine's own store: its hour file sits 147
#: characters below the store.
_ENGINE_RUN = (f"local_20260926060000_{'0' * 64}"
               f"_full_wrf_science_v4_{'b' * 16}_science_v1")

#: How many times a render's own cleanup tries ``rmtree`` before it gives
#: up: the eight tries of :func:`gpuwm.render._remove_scratch_store`'s
#: loop and its last ``ignore_errors`` pass.
_RENDER_OWN_TRIES = 9

_START = datetime(2026, 9, 26, 6)


def _hour_file(store: Path) -> Path:
    return Path(store) / "wrf" / _ENGINE_RUN / "f000.rws"


def _write(path: Path, data: bytes = b"x" * 4096) -> None:
    """Create ``path`` at any length, as the engine does."""
    os.makedirs(fs_path(path.parent, descend=True), exist_ok=True)
    with open(fs_path(path, descend=True), "wb") as handle:
        handle.write(data)


def _exists(path) -> bool:
    return os.path.lexists(fs_path(path, descend=True))


def _key(path) -> str:
    """One spelling for a path whichever way a caller spelled it."""
    text = os.fspath(path)
    if text.startswith("\\\\?\\"):
        text = text[4:]
    return os.path.normcase(os.path.abspath(text))


def _run_folder(case: Path, stamp: str) -> Path:
    run = case / f"run-{stamp}_i202609260600Z"
    (run / "png").mkdir(parents=True)
    return run


def _case(tmp_path: Path) -> Path:
    """A case folder whose run folders are as deep as the sweep's.

    The run folder on the sweep machine was 104 characters, which put
    its hour files at 294 to 296 characters, past the Windows ceiling.
    The name is padded only as far as that needs: a temp root that is
    long already would otherwise push the run folder itself past the
    248 characters Windows allows a directory in the ordinary spelling,
    and the test would fail setting up rather than on what it tests.
    """
    stamp = len(os.sep + "run-20260926-105133Z_i202609260600Z")
    pad = 104 - len(os.path.abspath(tmp_path)) - len(os.sep + "case-") - stamp
    return tmp_path / ("case-" + "c" * max(pad, 1))


def _scratch_left(case: Path) -> list[str]:
    """Every render scratch root under ``case``, and every store in one.

    Roots sit at most three levels down (``<run>/run/wrfout.render-
    scratch`` is the deepest a render has put one).  Listed level by
    level rather than with ``**``, which would walk into the stores' hour
    files and trip over the very path ceiling this file is about.
    """
    found = []
    for pattern in ("*.render-scratch", "*/*.render-scratch",
                    "*/*/*.render-scratch"):
        for root in case.glob(pattern):
            found.append(str(root))
            found.extend(str(entry) for entry in root.iterdir())
    return sorted(found)


class _Rmtree:
    """``shutil.rmtree`` that refuses a working store while it is held.

    ``held(store, attempt)`` answers whether this attempt on that store
    fails; everything else goes to the real ``rmtree``.  A store is a
    directory whose name starts ``rwstore-``, whichever spelling reached
    the call.
    """

    def __init__(self, held):
        self._real = shutil.rmtree
        self._held = held
        self.attempts: dict[str, int] = {}

    def __call__(self, path, ignore_errors=False, *args, **kwargs):
        key = _key(path)
        if Path(key).name.startswith("rwstore-"):
            attempt = self.attempts.get(key, 0) + 1
            self.attempts[key] = attempt
            if self._held(Path(key), attempt):
                if ignore_errors:
                    return None
                raise OSError(145, "The directory is not empty", str(path))
        return self._real(path, ignore_errors, *args, **kwargs)


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _warnings(path: Path, code: str) -> list[dict]:
    return [event for event in _events(path)
            if event.get("event") == "warning" and event.get("code") == code]


@pytest.fixture()
def quick(monkeypatch):
    """No real waiting in a render's retry loop."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


def _run_b(monkeypatch, run: Path, *, stores: list):
    """The sweep's run b, as its finalize stage saw it.

    Every frame was drawn while the forecast ran, so finalize spawns ONE
    render subprocess: the windowed pass over the whole series.  Before
    it, this process imports the smallest frame into a store of its own
    to read the engine's windowed catalog.  Both write engine-shaped hour
    files, and both then run their own cleanup.
    """
    frames = []
    for hour in range(3):
        valid = _START + timedelta(hours=hour)
        frame = run / "run" / "wrfout" / f"wrfout_d01_{valid:%Y-%m-%d_%H_%M_%S}"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"frame")
        frames.append(frame)
    plan = {"run": run / "run", "wrfout_dir": run / "run" / "wrfout",
            "render": run / "png", "render_products": "t2,qpf_1h"}

    import gpuwm.first_products as first_products

    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "wrfout_frames", lambda _plan: list(frames))
    monkeypatch.setattr(first_products, "published_frames",
                        lambda found, _plan: (list(found), [], None))
    monkeypatch.setattr(live_products, "published_frames",
                        lambda found, _plan: ([], list(found), None))
    monkeypatch.setattr(go_cli, "render_command", lambda *a, **k: ["renderer"])
    monkeypatch.setattr(rustwx, "find_renderer", lambda: Path("rw_wrfbatch"))

    def catalog(renderer, wrfouts, *, store_root, heavy=False):
        stores.append(Path(store_root))
        _write(_hour_file(store_root))
        return [("qpf_1h", "windowed", "renderable", "", "")], "total=1"

    monkeypatch.setattr(rustwx, "catalog_rows", catalog)

    def windowed_pass(label, command, **kwargs):
        # The render subprocess: its stores take the token its door
        # handed down, it writes its hour files, runs its own cleanup
        # and exits 0.
        token = (kwargs.get("env") or {}).get(render.SCRATCH_PREFIX_ENV)
        with render.scratch_store(plan["render"], prefix=token) as store:
            stores.append(store)
            _write(_hour_file(store))

    monkeypatch.setattr(go_cli, "_run_stage", windowed_pass)
    return plan


def _observer(path: Path) -> runplan.RunObserver:
    return runplan.RunObserver(runplan.EventStream(path, mirror=None))


def test_a_finalize_whose_render_could_not_remove_its_store_leaves_none(
        tmp_path, monkeypatch, quick, capsys):
    """THE F7 CASE: a store the render could not remove goes once it exits.

    Each working store refuses ``rmtree`` for the render's own nine
    tries, as a store behind the path ceiling or a held handle did, and
    accepts the next one.  Red on the base: the stage passed, nothing
    swept after a pass, and both stores stayed -- the windowed pass's
    under ``png.render-scratch`` and the catalog import's under
    ``run/wrfout.render-scratch``, exactly the two run b kept.
    """
    case = _case(tmp_path)
    run = _run_folder(case, "20260926-105133Z")
    stores: list = []
    plan = _run_b(monkeypatch, run, stores=stores)
    monkeypatch.setattr(shutil, "rmtree", _Rmtree(
        lambda _store, attempt: attempt <= _RENDER_OWN_TRIES))
    events = tmp_path / "events.jsonl"

    runplan._finish_render(plan, observer=_observer(events))

    assert len(stores) == 2, "the premise: the import and the windowed pass"
    assert all(len(os.path.abspath(_hour_file(store))) > 260
               for store in stores), "the premise: past the path ceiling"
    assert _scratch_left(case) == [], (
        "a finalize that passed left render scratch behind")
    # The render's own "left behind" line is captured and dropped on a
    # passing stage, so the run's events carry it instead.
    [left] = _warnings(events, "render_scratch_left")
    assert left["stage"] == "render"
    assert left["stores"] == 2 and left["removed"] == 2 and left["kept"] == 0
    assert left["bytes"] >= 2 * 4096
    assert left["stage_failed"] is False
    assert left["scratch_root"] == str(render.scratch_root_for(plan["render"]))
    assert "code" in left and left["code"] in runplan.WARNING_CODES
    assert "render: warning:" in capsys.readouterr().err


def test_a_store_still_held_after_the_run_goes_with_the_next_run_in_that_folder(
        tmp_path, monkeypatch, quick, capsys):
    """What even the closing sweep cannot remove is marked, then removed.

    Another program keeps an hour file open past the end of run 1, so
    its closing sweep cannot remove the store either.  It says so, in the
    events, and marks the store; run 2 in the same case folder removes it
    before it draws anything.
    """
    case = _case(tmp_path)
    first = _run_folder(case, "20260926-105133Z")
    stores: list = []
    plan = _run_b(monkeypatch, first, stores=stores)
    held = {"on": True}
    monkeypatch.setattr(shutil, "rmtree", _Rmtree(
        lambda store, _attempt: held["on"]
        and store.parent == render.scratch_root_for(plan["render"])
        and store.name.startswith(render.DEFAULT_SCRATCH_PREFIX)))
    events = tmp_path / "events-1.jsonl"
    runplan._finish_render(plan, observer=_observer(events))

    [left] = _warnings(events, "render_scratch_left")
    assert left["kept"] == 2 and left["removed"] == 0
    assert str(case) in left["message"], (
        "the warning names the folder whose next run removes them")
    for store in stores:
        assert _exists(store)
        assert store.with_name(store.name + render.ABANDONED_SUFFIX).is_file()
    capsys.readouterr()

    # The other program lets go.  A second run in the same case folder
    # starts, and a concurrent run's live store (its own token, no mark)
    # sits in a third run folder beside them.
    held["on"] = False
    third = _run_folder(case, "20260926-120000Z")
    live = render.scratch_root_for(third / "png") / "rwstore-cafe0123-live1234"
    _write(_hour_file(live))
    second = _run_folder(case, "20260926-113000Z")
    later: list = []
    plan2 = _run_b(monkeypatch, second, stores=later)
    events2 = tmp_path / "events-2.jsonl"
    runplan._finish_render(plan2, observer=_observer(events2))

    assert not any(_exists(store) for store in stores), (
        "the next run in the folder did not remove what run 1 marked")
    assert not any(case.glob(f"**/*{render.ABANDONED_SUFFIX}"))
    assert _exists(live), "a live run's unmarked store was removed"
    [swept] = _warnings(events2, "render_scratch_swept")
    assert swept["stores"] == 2 and swept["folder"] == str(case)
    assert "render_scratch_left" not in {
        event.get("code") for event in _events(events2)}
    assert _scratch_left(case) == sorted(
        [str(render.scratch_root_for(third / "png")), str(live)])


def test_the_closing_sweep_leaves_every_other_render_alone(tmp_path,
                                                           monkeypatch,
                                                           quick):
    """Only the stage's own token is swept after a pass, as after a failure.

    A concurrent render into the same delivery (another door's token,
    and the plain prefix of a `gpuwm render` typed by hand) keeps its
    live store and the scratch root it sits in.
    """
    case = _case(tmp_path)
    run = _run_folder(case, "20260926-105133Z")
    stores: list = []
    plan = _run_b(monkeypatch, run, stores=stores)
    root = render.scratch_root_for(plan["render"])
    other_door = root / "rwstore-beef4567-abcd1234"
    by_hand = root / "rwstore-q1w2e3r4"
    for store in (other_door, by_hand):
        _write(_hour_file(store))
    monkeypatch.setattr(shutil, "rmtree", _Rmtree(
        lambda store, attempt: (store.name.startswith(("rwstore-beef4567-",
                                                        "rwstore-q1w2e3r4"))
                                or attempt <= _RENDER_OWN_TRIES)))

    runplan._finish_render(plan, observer=_observer(tmp_path / "e.jsonl"))

    assert _exists(other_door) and _exists(by_hand)
    assert not any(_exists(store) for store in stores)
    assert not list(root.glob(f"*{render.ABANDONED_SUFFIX}")), (
        "a store this run never owned was marked for removal")


def test_a_store_past_the_windows_path_ceiling_is_removed(tmp_path, quick,
                                                          capsys):
    """The render's own cleanup reaches an hour file at any path length.

    Red on Windows before the repair: every ``rmtree`` of the store
    failed on the 294-character hour file, the retries waited out their
    ten seconds for a handle nobody held, and the store was left with a
    warning.  On POSIX there is no ceiling and this passes either way;
    the finalize test above is the one that is red everywhere.
    """
    run = _run_folder(_case(tmp_path), "20260926-105133Z")
    delivery = run / "png"
    with render.scratch_store(delivery) as store:
        hour = _hour_file(store)
        _write(hour)
        assert len(os.path.abspath(hour)) > 260, (
            "the premise: the hour file sits past the Windows ceiling")
    assert not _exists(store), f"the working store was left: {store}"
    assert not _exists(render.scratch_root_for(delivery))
    assert "left behind" not in capsys.readouterr().err


def test_a_door_sweep_reaches_a_store_past_the_path_ceiling(tmp_path):
    """The failure-path sweep had the same blind spot, so it is pinned too."""
    run = _run_folder(_case(tmp_path), "20260926-105133Z")
    delivery = run / "png"
    token = render.stage_scratch_prefix()
    store = render.scratch_root_for(delivery) / f"{token}abcd1234"
    _write(_hour_file(store))

    assert render.sweep_abandoned_scratch(delivery, prefix=token) == [store]
    assert not _exists(store)


def test_only_a_minted_store_can_be_marked_for_a_later_run(tmp_path):
    """The mark is the ownership proof, so a store nobody minted gets none."""
    root = tmp_path / "png.render-scratch"
    plain = root / "rwstore-q1w2e3r4"
    minted = root / "rwstore-0a1b2c3d-q1w2e3r4"
    for store in (plain, minted):
        store.mkdir(parents=True)

    written = render.mark_abandoned_scratch([plain, minted])

    assert written == [minted.with_name(minted.name + render.ABANDONED_SUFFIX)]
    assert render.sweep_marked_scratch(tmp_path) == [minted]
    assert plain.is_dir()


def test_the_bare_go_stream_carries_the_warning_too(tmp_path, monkeypatch,
                                                    quick):
    """`gpuwm go` writes run-plan's grammar, so the relay lands there as well."""
    delivery = _run_folder(_case(tmp_path), "20260926-105133Z") / "png"
    token = render.stage_scratch_prefix()
    store = render.scratch_root_for(delivery) / f"{token}abcd1234"
    _write(_hour_file(store))
    chain = chain_events.GoChainEvents()
    chain.open(tmp_path / "events.jsonl")

    go_cli._close_stage_scratch({"render": delivery}, prefix=token,
                                observer=chain)
    chain.close()

    [left] = _warnings(tmp_path / "events.jsonl", "render_scratch_left")
    assert left["stage"] == "render" and left["removed"] == 1
    assert not _exists(store)
