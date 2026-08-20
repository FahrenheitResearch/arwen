"""Render-on-first-committed-frame: the TTFP finisher, and its guards.

Three groups.  The first drives :class:`gpuwm.first_products.FirstProducts`
and the observer that arms it with a stand-in renderer, so the dispatch
rules, the event and every way this declines to publish are pinned
without spending a real render on them.  The second is the receipt: what
licenses the finalize stage to skip a frame, and every mutation that
must void that licence.  The third runs the REAL renderer on a real
wrfout and checks the two things that cannot be checked any other way --
that a frame rendered early is byte-identical to the same frame rendered
at finalize, and that the render process never creates a CUDA context
while the forecast owns the card.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from gpuwm.first_products import (FIRST_PRODUCTS_RECEIPT,
                                  FIRST_PRODUCTS_SCHEMA, FirstProducts,
                                  early_render_requested, published_frames,
                                  read_receipt)
from gpuwm.runplan import EVENT_TAGS, EventStream, RunObserver, read_events

_VALID = datetime(1974, 4, 3, 18, 0, 0)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _stand_in_renderer(*, names=("refl_d01-1km_x.png",),
                       payload=b"\x89PNG\r\n\x1a\nstand-in", returncode=0,
                       raises=None):
    """A renderer that writes the files a real one would, and no more.

    It reads its output directory out of the composed command rather
    than being told it, so a change to how ``render_command`` spells
    ``--out`` breaks these tests instead of silently bypassing them.
    """

    def run(command):
        if raises is not None:
            raise raises
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        for name in names:
            (out / name).write_bytes(payload)
        return subprocess.CompletedProcess(list(command), returncode, "", "")

    return run


def _plan(tmp_path, *, products="refl,t2"):
    return {"run": tmp_path / "run", "render": tmp_path / "png",
            "render_products": products}


def _frame(tmp_path, name="wrfout_d01_1974-04-03_18_00_00"):
    path = tmp_path / "run" / "wrfout" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"one durable history frame")
    return path


class _Recorder:
    def __init__(self):
        self.reports = []
        self.warnings = []

    def report(self, receipt):
        self.reports.append(receipt)

    def warn(self, code, message, **fields):
        self.warnings.append((code, message, fields))


@pytest.fixture(scope="module")
def wrf_package():
    """The mandated science core, as a FIXTURE rather than a module skip.

    This was a bare ``pytest.importorskip("wrf")`` at module scope, sitting
    two thirds of the way down the file under a "the real renderer"
    banner.  A module-level skip is not local to the section it is
    written under: it fires at IMPORT, so it took the 21 tests defined
    ABOVE it with it.  Those 21 are the whole early-render /
    time-to-first-plot front door, added to
    ``tools/battery/stage1_files.txt`` in 1.8.7 precisely because a
    regression in them is silent -- and from that day until this one the
    file collected ZERO tests on every cut.  A stage-1 entry that reports
    green while running nothing is worse than an absent one.

    A fixture skips exactly its requesters, so the section boundary is
    now real: every test that does not ask for this runs without the
    science core, and every test that genuinely drives a render asks.
    """

    return pytest.importorskip(
        "wrf", reason="this proof drives the real renderer, whose derived "
                      "fields all come from the wrf package")


# ---------------------------------------------------------------------------
# Arming: off by default, on when the run named products
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("products,armed", [
    (None, False), ("", False), ("   ", False),
    ("none", False), ("NONE", False), (" None ", False),
    ("refl", True), ("all", True), ("refl,t2", True),
])
def test_the_early_render_is_off_unless_the_run_named_products(
        products, armed):
    """One field answers "which products", including "not early either"."""

    assert early_render_requested(products) is armed


def test_arming_a_run_that_named_no_products_leaves_nothing_armed(tmp_path):
    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events)
    observer.arm_first_products(_plan(tmp_path, products=None))
    assert observer.first_products is None

    # And the hook stays inert: a committed frame is announced exactly as
    # it always was, with no second event behind it.
    observer.output_committed(domain=1, valid_time=_VALID, path="frame")
    events.close()
    tags = [record["event"] for record in
            read_events(tmp_path / "events.jsonl")]
    assert tags == ["output_committed"]


# ---------------------------------------------------------------------------
# Dispatch: the first root frame, once
# ---------------------------------------------------------------------------


def test_only_the_first_committed_frame_dispatches_a_render(tmp_path):
    recorder = _Recorder()
    calls = []
    inner = _stand_in_renderer()

    def counting(command):
        calls.append(list(command))
        return inner(command)

    trigger = FirstProducts(_plan(tmp_path), report=recorder.report,
                            warn=recorder.warn, runner=counting)
    first = _frame(tmp_path)
    second = _frame(tmp_path, "wrfout_d01_1974-04-03_19_00_00")

    assert trigger.frame_committed(domain=1, valid_time=_VALID, path=first)
    assert not trigger.frame_committed(
        domain=1, valid_time=_VALID, path=second)
    trigger.wait(timeout=30.0)

    assert len(calls) == 1
    assert str(first) in calls[0]
    assert str(second) not in calls[0]


def test_a_nest_frame_is_not_the_analysis_and_does_not_dispatch(tmp_path):
    """The trigger is the ROOT domain's first frame, not any domain's."""

    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events, root_domain=1)
    observer.arm_first_products(_plan(tmp_path))
    observer.first_products._runner = _stand_in_renderer()

    observer.output_committed(domain=2, valid_time=_VALID,
                              path=_frame(tmp_path))
    assert not observer.first_products.dispatched


def test_the_first_frame_reaches_the_render_through_the_real_observer(
        tmp_path, monkeypatch):
    from gpuwm import first_products as module

    monkeypatch.setattr(module, "_run_render", _stand_in_renderer())
    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events, root_domain=1)
    observer.arm_first_products(_plan(tmp_path))
    frame = _frame(tmp_path)

    observer.output_committed(domain=1, valid_time=_VALID, path=frame)
    observer.first_products.wait(timeout=30.0)
    events.close()

    records = read_events(tmp_path / "events.jsonl")
    tags = [record["event"] for record in records]
    assert tags == ["output_committed", "first_products_ready"]
    assert set(tags) <= set(EVENT_TAGS)

    ready = records[-1]
    assert ready["domain"] == 1
    assert ready["frame"] == str(frame)
    assert ready["render_products"] == "refl,t2"
    assert ready["valid_time"] == _VALID.isoformat()
    assert [Path(p).name for p in ready["paths"]] == ["refl_d01-1km_x.png"]
    assert all(Path(p).is_file() for p in ready["paths"])
    # THE number: wall from plan_accepted to pictures on disk.
    assert ready["seconds_from_plan_accepted"] >= 0.0
    assert ready["render_seconds"] >= 0.0


def test_the_ttfp_number_is_measured_from_plan_accepted_not_from_the_render(
        tmp_path, monkeypatch):
    """A clock that starts when the observer is built measures nothing."""

    import time

    from gpuwm import first_products as module

    monkeypatch.setattr(module, "_run_render", _stand_in_renderer())
    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    # Fetch and preparation already cost this run 120 s before an
    # observer existed; the TTFP number has to include them.
    observer = RunObserver(events, root_domain=1,
                           accepted_wall=time.perf_counter() - 120.0)
    observer.arm_first_products(_plan(tmp_path))
    observer.output_committed(domain=1, valid_time=_VALID,
                              path=_frame(tmp_path))
    observer.first_products.wait(timeout=30.0)
    events.close()

    ready = read_events(tmp_path / "events.jsonl")[-1]
    assert ready["seconds_from_plan_accepted"] >= 120.0
    # And the same number is on the observer, for the `completed` event
    # to repeat, so comparing runs is not a stream scan.
    assert observer.first_products_seconds == ready[
        "seconds_from_plan_accepted"]


def test_a_run_that_published_nothing_early_reports_a_null_ttfp(tmp_path):
    """`first_products_seconds` is null, not zero, when there is none."""

    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events, root_domain=1)

    assert observer.first_products_seconds is None


# ---------------------------------------------------------------------------
# Declining to publish, in every way it can
# ---------------------------------------------------------------------------


def test_a_render_that_draws_nothing_publishes_nothing_and_says_so(tmp_path):
    """The cold-start frame carries no REFL_10CM.  That is not a failure."""

    recorder = _Recorder()
    trigger = FirstProducts(
        _plan(tmp_path, products="refl"), report=recorder.report,
        warn=recorder.warn, runner=_stand_in_renderer(names=(),
                                                      returncode=1))
    trigger.frame_committed(domain=1, valid_time=_VALID,
                            path=_frame(tmp_path))
    trigger.wait(timeout=30.0)

    assert recorder.reports == []
    assert trigger.receipt is None
    assert [code for code, _, _ in recorder.warnings] == [
        "first_products_empty"]
    # Nothing claimed means the finalize stage is untouched.
    assert not (tmp_path / "png" / FIRST_PRODUCTS_RECEIPT).exists()


def test_a_render_that_raises_warns_and_never_reaches_the_caller(tmp_path):
    recorder = _Recorder()
    trigger = FirstProducts(
        _plan(tmp_path), report=recorder.report, warn=recorder.warn,
        runner=_stand_in_renderer(raises=OSError("no renderer here")))
    trigger.frame_committed(domain=1, valid_time=_VALID,
                            path=_frame(tmp_path))
    assert trigger.wait(timeout=30.0) is None

    assert recorder.reports == []
    assert [code for code, _, _ in recorder.warnings] == [
        "first_products_failed"]
    assert "no renderer here" in recorder.warnings[0][1]


def test_a_dispatch_failure_never_escapes_the_observer(tmp_path):
    """``runtime._output_committed`` does not wrap this call.  So we do."""

    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events, root_domain=1)
    observer.arm_first_products(_plan(tmp_path))

    class _Exploding:
        dispatched = False

        def frame_committed(self, **_fields):
            raise RuntimeError("the thread would not start")

    observer._first_products = _Exploding()
    observer.output_committed(domain=1, valid_time=_VALID, path="frame")
    events.close()

    tags = [record["event"] for record in
            read_events(tmp_path / "events.jsonl")]
    assert tags == ["output_committed", "warning"]


def test_the_scratch_directory_never_survives_a_render(tmp_path):
    recorder = _Recorder()
    trigger = FirstProducts(_plan(tmp_path), report=recorder.report,
                            warn=recorder.warn,
                            runner=_stand_in_renderer())
    trigger.frame_committed(domain=1, valid_time=_VALID,
                            path=_frame(tmp_path))
    trigger.wait(timeout=30.0)

    assert sorted(p.name for p in (tmp_path / "png").iterdir()) == [
        FIRST_PRODUCTS_RECEIPT, "refl_d01-1km_x.png"]


def test_a_render_still_running_at_finalize_publishes_nothing(tmp_path):
    """A wedged render must not hold a finished forecast hostage."""

    import threading

    recorder = _Recorder()
    release = threading.Event()

    def slow(command):
        release.wait(30.0)
        return _stand_in_renderer()(command)

    trigger = FirstProducts(_plan(tmp_path), report=recorder.report,
                            warn=recorder.warn, runner=slow)
    trigger.frame_committed(domain=1, valid_time=_VALID,
                            path=_frame(tmp_path))
    try:
        assert trigger.wait(timeout=0.2) is None
        assert [code for code, _, _ in recorder.warnings] == [
            "first_products_timeout"]
    finally:
        release.set()
        trigger.wait(timeout=30.0)


# ---------------------------------------------------------------------------
# The receipt, and what voids it
# ---------------------------------------------------------------------------


def _published(tmp_path, *, products="refl,t2"):
    """One completed early render; returns (plan, frame, receipt)."""

    recorder = _Recorder()
    plan = _plan(tmp_path, products=products)
    frame = _frame(tmp_path)
    trigger = FirstProducts(plan, report=recorder.report,
                            warn=recorder.warn,
                            runner=_stand_in_renderer())
    trigger.frame_committed(domain=1, valid_time=_VALID, path=frame)
    trigger.wait(timeout=30.0)
    assert recorder.warnings == []
    return plan, frame, trigger.receipt


def test_the_receipt_names_the_frame_and_every_picture_by_digest(tmp_path):
    plan, frame, receipt = _published(tmp_path)
    on_disk = read_receipt(plan["render"])

    assert on_disk == receipt
    assert on_disk["schema"] == FIRST_PRODUCTS_SCHEMA
    assert on_disk["frame"] == str(frame)
    assert len(on_disk["frame_sha256"]) == 64
    assert [entry["name"] for entry in on_disk["written"]] == [
        "refl_d01-1km_x.png"]
    assert len(on_disk["written"][0]["sha256"]) == 64
    assert on_disk["render_products"] == "refl,t2"
    # The command is recorded so a reader can see the early render and
    # the finalize render were the same invocation over one frame.
    assert "--products" in on_disk["command"]


def test_the_receipt_states_what_its_instant_measures(tmp_path):
    """One definition, written where the number is.

    THE DRIFT THIS CLOSES, measured on both 3080 walks: `go` printed
    "time to first plot 0m 46s (first-products receipt)" while the
    earliest PNG in the published run tree carried 2m 45s.  Two
    quantities were both being called time to first plot and the receipt
    named neither, so a reader could not tell which one they held.
    """

    from gpuwm.first_products import FIRST_PLOT_DEFINITION

    plan, _frame, _receipt = _published(tmp_path)
    on_disk = read_receipt(plan["render"])

    assert on_disk["measures"] == FIRST_PLOT_DEFINITION
    assert "published_unix_ms" in FIRST_PLOT_DEFINITION
    assert "readable at its final path" in FIRST_PLOT_DEFINITION
    assert "LATER mtime" in FIRST_PLOT_DEFINITION


def test_a_rewritten_picture_is_no_longer_the_one_the_receipt_stamped(
        tmp_path):
    """The check the digests structurally cannot make.

    The early picture and the finalize one are byte-identical by
    construction, so a re-render leaves every recorded sha256 matching
    and moves only the mtime -- which is exactly what a reader comparing
    the printed number against the tree is looking at.
    """

    from gpuwm.first_products import published_pictures_are_original

    plan, _frame, receipt = _published(tmp_path)
    render_dir = plan["render"]
    assert published_pictures_are_original(receipt, render_dir) is True

    picture = render_dir / receipt["written"][0]["name"]
    later = (receipt["published_unix_ms"] + 120_000) / 1000.0
    os.utime(picture, (later, later))
    assert published_pictures_are_original(receipt, render_dir) is False

    # ...and the digests still match, which is the point.
    remaining, already, _note = published_frames([_frame], plan)
    assert already == [_frame] and remaining == []


def test_a_missing_picture_is_not_an_original_one_either(tmp_path):
    from gpuwm.first_products import published_pictures_are_original

    plan, _frame, receipt = _published(tmp_path)
    (plan["render"] / receipt["written"][0]["name"]).unlink()
    assert published_pictures_are_original(receipt, plan["render"]) is False


def test_an_unset_product_spec_is_the_render_default_not_a_different_one(
        tmp_path):
    """`go` asks the runner for the explicit ``all``; the finalize stage
    leaves ``--products`` off, and ``gpuwm render`` defaults to ``all``.
    Comparing the two spellings literally made every `go` receipt look
    like it had been drawn for a different product set, so the skip this
    module exists for never happened on the front door people use."""

    from gpuwm.first_products import (DEFAULT_RENDER_PRODUCTS,
                                      effective_products)

    assert DEFAULT_RENDER_PRODUCTS == "all"
    assert effective_products(None) == "all"
    assert effective_products("") == "all"
    assert effective_products("  ") == "all"
    assert effective_products("all") == "all"
    assert effective_products(" refl,t2 ") == "refl,t2"

    plan, frame, _receipt = _published(tmp_path, products="all")
    plan["render_products"] = None
    remaining, already, note = published_frames([frame], plan)
    assert already == [frame], note
    assert remaining == []


def test_finalize_drops_the_frame_the_receipt_proves(tmp_path):
    plan, frame, _receipt = _published(tmp_path)
    later = _frame(tmp_path, "wrfout_d01_1974-04-03_19_00_00")

    remaining, already, note = published_frames([frame, later], plan)

    assert remaining == [later]
    assert already == [frame]
    assert "1 frame already published" in note
    assert frame.name in note


@pytest.mark.parametrize("mutation", [
    "frame_edited", "frame_removed", "picture_removed", "picture_edited",
    "products_changed", "frame_not_in_list",
])
def test_an_unproven_claim_is_never_used_to_skip_work(tmp_path, mutation):
    """A receipt is a claim about the past; only digests keep it true."""

    plan, frame, receipt = _published(tmp_path)
    picture = plan["render"] / receipt["written"][0]["name"]
    frames = [frame]

    if mutation == "frame_edited":
        frame.write_bytes(b"a different history frame entirely")
    elif mutation == "frame_removed":
        frame.unlink()
    elif mutation == "picture_removed":
        picture.unlink()
    elif mutation == "picture_edited":
        picture.write_bytes(b"not the picture that was published")
    elif mutation == "products_changed":
        plan["render_products"] = "olr"
    elif mutation == "frame_not_in_list":
        frames = [_frame(tmp_path, "wrfout_d01_1974-04-03_19_00_00")]

    remaining, already, note = published_frames(frames, plan)

    assert remaining == frames, mutation
    assert already == [], mutation
    assert note is not None and "not used" in note, mutation


def test_a_receipt_of_another_schema_is_ignored_rather_than_trusted(
        tmp_path):
    render_dir = tmp_path / "png"
    render_dir.mkdir(parents=True)
    (render_dir / FIRST_PRODUCTS_RECEIPT).write_text(
        json.dumps({"schema": "something.else.v9"}), encoding="utf-8")
    frame = _frame(tmp_path)

    remaining, already, note = published_frames([frame], _plan(tmp_path))

    assert (remaining, already, note) == ([frame], [], None)


# ---------------------------------------------------------------------------
# The finalize stage itself
# ---------------------------------------------------------------------------


class _ObserverWith:
    def __init__(self, trigger):
        self.first_products = trigger


class _CollectedTrigger:
    def __init__(self):
        self.waited = False

    def wait(self, timeout=None):
        self.waited = True
        return None


def _a_box_that_can_draw(monkeypatch) -> None:
    """Declare a usable render engine for the three bookkeeping tests.

    ``_render_stage`` asks ``gpuwm.render.drawable_engine`` whether this
    install can draw at all and skips with a remedy when it cannot.
    Since the render law's one-fallback clause was enforced (audit F7)
    the only chainable engine is the rust one, so on a box whose only
    resolvable ``rw_wrfbatch`` belongs to another checkout that answer is
    "no engine" -- and these three tests are about the EARLY-RENDER
    BOOKKEEPING, not about which engine draws (the subprocess is stubbed
    out entirely).  Declaring the engine keeps them measuring the thing
    they are named for on every box.
    """

    from gpuwm import render

    monkeypatch.setattr(render, "drawable_engine",
                        lambda: ("rust", "declared by the test"))


def test_the_render_stage_collects_the_early_render_before_it_draws(
        wrf_package, tmp_path, monkeypatch):
    # ``_render_stage`` returns False and prints a remedy when no engine
    # can draw, so the three tests that assert it returns True declare
    # one even though they stub the subprocess out.
    from gpuwm import go_cli

    _a_box_that_can_draw(monkeypatch)
    plan, frame, _receipt = _published(tmp_path)
    later = _frame(tmp_path, "wrfout_d01_1974-04-03_19_00_00")
    commands = []
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: commands.append(list(command)))

    trigger = _CollectedTrigger()
    assert go_cli._render_stage(plan, explain=False,
                                observer=_ObserverWith(trigger))

    assert trigger.waited
    assert len(commands) == 1
    assert str(later) in commands[0]
    assert str(frame) not in commands[0]


def test_a_run_whose_only_frame_was_published_early_draws_nothing_again(
        wrf_package, tmp_path, monkeypatch, capsys):
    from gpuwm import go_cli

    _a_box_that_can_draw(monkeypatch)
    plan, _frame_path, _receipt = _published(tmp_path)
    commands = []
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: commands.append(list(command)))

    assert go_cli._render_stage(plan, explain=False,
                                observer=_ObserverWith(_CollectedTrigger()))

    assert commands == []
    printed = capsys.readouterr().out
    # "done" and "produced nothing" are opposite outcomes; they used to
    # print the same sentence.
    assert "render complete" in printed
    assert "published no wrfout frame" not in printed


def test_the_hrrr_chain_arms_with_the_dict_its_finalize_stage_uses(
        tmp_path, monkeypatch):
    """Arm and finalize must not be able to drift apart.

    They are two call sites in one function, minutes of forecast apart.
    Two copies of the literal is exactly how a run publishes its first
    frame into one directory and the rest into another, so both read
    ``_chain_render_plan`` -- and this is what says so.
    """

    from types import SimpleNamespace

    from gpuwm import go_cli, runplan

    plan = SimpleNamespace(
        run_options={"render_products": "refl,t2"})
    forecast_dir = tmp_path / "chain" / "run"
    run_dir = tmp_path

    armed = runplan._chain_render_plan(
        plan, forecast_dir=forecast_dir, run_dir=run_dir)
    assert armed == {"run": forecast_dir,
                     "render": run_dir / "chain" / "png",
                     "render_products": "refl,t2"}

    seen = []
    monkeypatch.setattr(go_cli, "_render_stage",
                        lambda p, **kw: seen.append(dict(p)))
    monkeypatch.setattr(runplan, "_chain_summary", lambda *a, **kw: {})
    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    runplan._chain_render(
        plan, forecast_dir=forecast_dir, run_dir=run_dir,
        observer=RunObserver(events, root_domain=1))
    events.close()

    assert seen == [armed]


def test_arming_the_hrrr_chain_dict_produces_a_live_trigger(tmp_path):
    """The dict the chain arms with is one this module accepts."""

    from types import SimpleNamespace

    from gpuwm import runplan

    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = RunObserver(events, root_domain=1)
    observer.arm_first_products(runplan._chain_render_plan(
        SimpleNamespace(run_options={"render_products": "refl"}),
        forecast_dir=tmp_path / "run", run_dir=tmp_path))

    assert observer.first_products is not None
    assert observer.first_products.render_dir == tmp_path / "chain" / "png"
    assert observer.first_products.render_products == "refl"

    # And the same chain with no products asks for nothing.
    quiet = RunObserver(events, root_domain=1)
    quiet.arm_first_products(runplan._chain_render_plan(
        SimpleNamespace(run_options={}),
        forecast_dir=tmp_path / "run", run_dir=tmp_path))
    events.close()
    assert quiet.first_products is None


def test_a_proven_receipt_is_honoured_without_an_armed_trigger(
        wrf_package, tmp_path, monkeypatch, capsys):
    """The receipt is a fact on disk, not a property of this process.

    THIS TEST USED TO ASSERT THE OPPOSITE -- "no trigger, so no receipt
    is consulted and every frame is drawn" -- and that is precisely the
    defect it was pinning.  `gpuwm go` never arms the trigger: it asks
    the runner SUBPROCESS for the early render on its command line
    (``go_cli.forecast_command``) and keeps its process isolation.  So on
    the front door people use, the skip never happened, this stage redrew
    the frame the early render had already published, and the published
    tree's earliest picture carried the FINALIZE timestamp (2m 45s on
    both 3080 walks) while the receipt said 46s -- a headline
    contradicted by the only artifact a reader can check.

    Nothing about the safety changed: ``published_frames`` re-checks the
    frame and every picture the receipt names against their recorded
    digests before a single frame is dropped, and the test below feeds it
    a receipt that no longer holds.
    """

    from gpuwm import go_cli

    _a_box_that_can_draw(monkeypatch)
    plan, frame, _receipt = _published(tmp_path)
    commands = []
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: commands.append(list(command)))

    assert go_cli._render_stage(plan, explain=False, observer=None)

    # The one frame was published early and proven by digest, so this
    # stage has nothing left to draw and says so.
    assert commands == []
    printed = capsys.readouterr().out
    assert "already published by the early render" in printed
    assert "digests verified" in printed
    assert str(frame.name) in printed


def test_an_unproven_receipt_is_not_honoured_without_a_trigger_either(
        wrf_package, tmp_path, monkeypatch, capsys):
    """The control on the test above: reading the receipt off disk is not
    trusting it.  A picture that no longer matches its digest sends every
    frame back through the render stage, with the reason said out loud."""

    from gpuwm import go_cli
    from gpuwm.first_products import FIRST_PRODUCTS_RECEIPT, read_receipt

    _a_box_that_can_draw(monkeypatch)
    plan, frame, _receipt = _published(tmp_path)
    receipt = read_receipt(plan["render"])
    picture = plan["render"] / receipt["written"][0]["name"]
    picture.write_bytes(picture.read_bytes() + b"edited")

    commands = []
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: commands.append(list(command)))

    assert go_cli._render_stage(plan, explain=False, observer=None)

    assert len(commands) == 1
    assert str(frame) in commands[0]
    printed = capsys.readouterr().out
    assert "early-render receipt not used" in printed
    assert (plan["render"] / FIRST_PRODUCTS_RECEIPT).is_file()


# ---------------------------------------------------------------------------
# The real renderer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_wrfout(wrf_package, tmp_path_factory):
    """A frame written by the project's own writer, as a run produces.

    With the production global-attribute profile, not a bare one: the
    rust renderer's fail-closed preflight requires ``START_DATE``, and
    the rust engine is what ``auto`` selects on a box where the binary
    is built -- so this fixture must be what a forecast really writes or
    the proof below would be about the fallback engine instead.
    """

    import datetime
    from types import SimpleNamespace

    import numpy as np

    from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs

    nz, ny, nx = 4, 12, 16
    rng = np.random.default_rng(11)
    path = (tmp_path_factory.mktemp("first-products")
            / "wrfout_d01_1974-04-03_18-00-00.nc")
    fields = {
        "T": np.zeros((nz, ny, nx), np.float32),
        "MU": np.zeros((ny, nx), np.float32),
        "REFL_10CM": rng.uniform(-20.0, 65.0,
                                 (nz, ny, nx)).astype(np.float32),
        "T2": rng.uniform(280.0, 300.0, (ny, nx)).astype(np.float32),
        "U10": rng.uniform(-10.0, 10.0, (ny, nx)).astype(np.float32),
        "V10": rng.uniform(-10.0, 10.0, (ny, nx)).astype(np.float32),
        "RAINC": rng.uniform(0.0, 5.0, (ny, nx)).astype(np.float32),
        "RAINNC": rng.uniform(0.0, 30.0, (ny, nx)).astype(np.float32),
        "OLR": rng.uniform(90.0, 320.0, (ny, nx)).astype(np.float32),
        "Q2": rng.uniform(0.004, 0.012, (ny, nx)).astype(np.float32),
        "PSFC": rng.uniform(96000.0, 98000.0, (ny, nx)).astype(np.float32),
        "XLAT": np.tile(np.linspace(38.0, 40.0, ny)[:, None],
                        (1, nx)).astype(np.float32),
        "XLONG": np.tile(np.linspace(-98.0, -95.0, nx)[None, :],
                         (ny, 1)).astype(np.float32),
        "HGT": np.zeros((ny, nx), np.float32),
        "SINALPHA": np.zeros((ny, nx), np.float32),
        "COSALPHA": np.ones((ny, nx), np.float32),
    }
    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(1974, 4, 3, 18), grid_id=1, parent_id=1,
        i_parent_start=1, j_parent_start=1, parent_grid_ratio=1, dt=6.0)
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0,
                      global_attrs=attrs) as writer:
        writer.write_frame("1974-04-03_18:00:00", fields)
    return path


def test_a_frame_rendered_early_is_byte_identical_to_one_rendered_late(
        real_wrfout, tmp_path):
    """The whole licence for skipping finalize work, proven end to end."""

    from gpuwm.fetch import sha256_file
    from gpuwm.go_cli import _stage_cwd, _stage_env, render_command

    recorder = _Recorder()
    early_plan = {"run": tmp_path / "run", "render": tmp_path / "early",
                  "render_products": "refl,t2"}
    trigger = FirstProducts(early_plan, report=recorder.report,
                            warn=recorder.warn)
    trigger.frame_committed(domain=1, valid_time=_VALID, path=real_wrfout)
    trigger.wait(timeout=600.0)
    assert recorder.warnings == [], recorder.warnings
    assert recorder.reports, "the early render published nothing"

    late = tmp_path / "late"
    completed = subprocess.run(
        render_command({"run": tmp_path / "run", "render": late,
                        "render_products": "refl,t2"}, [real_wrfout]),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        errors="replace", cwd=str(_stage_cwd()), env=_stage_env())
    assert completed.returncode == 0, completed.stderr

    # Keyed by the path RELATIVE to each render directory, not by the
    # bare filename: since 2.5.0 the render writes a tree
    # (gpuwm.render_layout), and identity has to cover where the file
    # landed as well as what is in it -- an early render that published
    # the right bytes into the wrong directory is not the same run.
    def _tree(root):
        return {p.relative_to(root).as_posix(): sha256_file(p)
                for p in sorted(root.rglob("*.png"))}

    early_pngs = _tree(tmp_path / "early")
    late_pngs = _tree(late)
    assert early_pngs and early_pngs == late_pngs

    # And the receipt's recorded digests are those same bytes, which is
    # what the finalize skip actually checks.
    receipt = read_receipt(tmp_path / "early")
    assert {entry["name"]: entry["sha256"]
            for entry in receipt["written"]} == early_pngs


def test_the_render_front_door_creates_no_cuda_context(tmp_path):
    """The forecast owns the GPU; the early render runs beside it.

    cupy is imported transitively by the package, so "does it import
    cupy" is the wrong question and answers yes.  The question is
    whether a context exists on the device, and the driver is the only
    honest witness: ``cuCtxGetCurrent`` reports 3
    (``CUDA_ERROR_NOT_INITIALIZED``) until something initialises CUDA.
    """

    from gpuwm.go_cli import _stage_cwd, _stage_env

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import ctypes, sys\n"
        "sys.argv = ['gpuwm', 'render', '--list-products']\n"
        "import gpuwm.cli, gpuwm.render, gpuwm.rustwx\n"
        "name = 'nvcuda.dll' if sys.platform == 'win32' "
        "else 'libcuda.so.1'\n"
        "try:\n"
        "    lib = ctypes.CDLL(name)\n"
        "except OSError:\n"
        "    print('NO-DRIVER')\n"
        "    raise SystemExit(0)\n"
        "ctx = ctypes.c_void_p()\n"
        "print('RC', lib.cuCtxGetCurrent(ctypes.byref(ctx)), ctx.value)\n",
        encoding="utf-8", newline="\n")
    completed = subprocess.run(
        [sys.executable, str(probe)], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, cwd=str(_stage_cwd()),
        env=_stage_env())

    assert completed.returncode == 0, completed.stderr
    verdict = completed.stdout.strip().splitlines()[-1]
    if verdict == "NO-DRIVER":
        pytest.skip("no CUDA driver on this box to interrogate")
    # rc 3 is CUDA_ERROR_NOT_INITIALIZED and the context pointer is
    # null: importing the whole render front door touched no device.
    assert verdict.startswith("RC 3 "), verdict
    assert verdict.endswith(" None"), verdict
