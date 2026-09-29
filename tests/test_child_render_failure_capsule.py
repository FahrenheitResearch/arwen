"""A failed child render says what the RENDERER said, not only what to type.

WHAT BREAKAGE THESE PIN (gate law).  The capsule a desktop run view
shows carried the exit code and a 24-product render command and nothing
else, because ``GoStageFailed`` carried only the code and the stage's own
output went to a terminal the run view does not have.  A reader was
handed a command and left to re-run the whole render -- after six and a
half hours of integration -- to learn which product had failed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuwm import offline_child_run
from gpuwm.go_cli import GoStageFailed


_SAID = (
    "FAILED qpf_1h F000: invalid store metadata: run wrf/local_x uses an "
    "exact-time ordinal axis\n"
    "batch render incomplete: rendered=13 skipped=0 failed=13")


class _Progress:
    """The two things :func:`_finish_child_render` reads off the observer."""

    def __init__(self, outdir: Path, early: int = 20):
        self.render_plan = {"render": str(outdir / "png"),
                            "render_products": "composite_reflectivity,qpf_1h",
                            "wrfout": str(outdir)}
        self.outdir = outdir
        self._render_summary = None
        self._early = early

    def early_pictures(self) -> int:
        return self._early

    #: The real counter, run against a real tree: the number in the
    #: sentence is the one a reader can check with ``find``.
    pictures_drawn = offline_child_run._ChildProgress.pictures_drawn

    def finish_stage(self) -> None:  # pragma: no cover - the pass arm
        raise AssertionError("a failed render does not finish its stage")


def _fail(monkeypatch, diagnostic: str) -> None:
    def raise_it(*_args, **_kwargs):
        raise GoStageFailed(1, diagnostic)

    monkeypatch.setattr("gpuwm.runplan._finish_render", raise_it)
    monkeypatch.setattr(offline_child_run, "_render_command_text",
                        lambda plan: "gpuwm render ... --series")


def _capsule(tmp_path, monkeypatch, diagnostic=_SAID):
    _fail(monkeypatch, diagnostic)
    report: dict = {"result": "PASS"}
    with pytest.raises(offline_child_run.OfflineChildContractError) as failure:
        offline_child_run._finish_child_render(
            _Progress(tmp_path), report=report)
    return str(failure.value), report


def _pictures(tmp_path: Path, count: int) -> None:
    """A picture tree in this run's own layout: one folder per frame."""

    for index in range(count):
        frame = tmp_path / "png" / f"f{index // 11:03d}"
        frame.mkdir(parents=True, exist_ok=True)
        (frame / f"arwen_wrf_{index:03d}.png").write_bytes(b"")


def test_the_capsule_carries_the_renderers_own_lines(tmp_path, monkeypatch):
    text, _report = _capsule(tmp_path, monkeypatch)
    assert "The render stage said:" in text
    assert "exact-time ordinal axis" in text
    assert "batch render incomplete: rendered=13 skipped=0 failed=13" in text


def test_each_paragraph_of_the_refusal_begins_its_own_line(tmp_path, monkeypatch):
    """A run view has line breaks and nothing else to format with.

    The engine's lines used to be joined to the sentence before them
    with a leading space, and the ``Next:`` clause after them carried
    one too, so the refusal read ``... said:`` then an indented block
    then `` Next:`` with a stray space opening the line.
    """

    text, _report = _capsule(tmp_path, monkeypatch)
    printed = text.splitlines()
    assert "The render stage said:" in printed
    assert any(line.startswith("Next: draw the saved frames") for line in printed)
    assert not any(line.startswith(" ") and line.strip()
                   .startswith(("Next:", "The render stage"))
                   for line in printed)


def test_the_capsule_still_carries_the_command_and_the_count(tmp_path, monkeypatch):
    """The new text is added to the sentence, never in place of it."""

    _pictures(tmp_path, 22)
    text, _report = _capsule(tmp_path, monkeypatch)
    assert "20 of them drawn early from the first frame" in text
    assert "gpuwm render ... --series" in text
    assert "draw the saved frames by hand" in text


def test_the_capsule_counts_the_pictures_that_are_on_disk(tmp_path, monkeypatch):
    """Measured on the shipped 2.7.5 wheel: a 13-frame child whose two
    requested snow variables could not be drawn left 143 pictures on
    disk, every frame's other eleven, while the capsule called them not
    drawn and sent the reader back to re-draw six hours of frames."""

    _pictures(tmp_path, 143)
    text, report = _capsule(tmp_path, monkeypatch)
    assert "143 pictures are on disk" in text
    assert "20 of them drawn early from the first frame" in text
    assert "at least one product was not drawn" in text
    assert "the rest not drawn" not in text
    assert report["products"]["pictures_on_disk"] == 143
    assert report["products"]["drawn_early"] == 20


def test_one_picture_is_counted_in_the_singular(tmp_path, monkeypatch):
    _pictures(tmp_path, 1)
    text, report = _capsule(tmp_path, monkeypatch)
    assert "1 picture is on disk" in text
    assert report["products"]["pictures_on_disk"] == 1


def test_an_empty_tree_still_says_this_run_has_no_pictures(tmp_path, monkeypatch):
    """The arm a finished child whose render stage drew nothing leaves:
    the forecast reached its end, the render ran, and there is no
    picture on disk to send a reader to."""

    text, report = _capsule(tmp_path, monkeypatch)
    assert "this run has no pictures" in text
    assert "pictures are on disk" not in text
    assert report["products"]["pictures_on_disk"] == 0


def test_a_tree_that_cannot_be_read_is_not_reported_as_an_empty_one(
        tmp_path, monkeypatch):
    """A count that could not be TAKEN is not a count of zero.

    "this run has no pictures" sends a reader to re-draw a whole child.
    It has to be a reading of the tree, and a tree whose listing failed
    -- a permission wall, a dead mount, a path that is a file -- is not
    a reading of anything.  The two were one answer, so the sentence a
    reader acted on was a guess whenever the count failed.
    """

    # The render plan points at a regular file, so listing it fails
    # rather than coming back empty.
    (tmp_path / "png").write_bytes(b"")
    text, report = _capsule(tmp_path, monkeypatch)
    assert "this run has no pictures" not in text
    assert "could not be read" in text
    assert report["products"]["pictures_on_disk"] is None
    assert report["products"]["pictures_on_disk_error"]


def test_a_readable_tree_carries_no_error_beside_its_count(
        tmp_path, monkeypatch):
    """The empty case is unchanged, and so is the counted one."""

    _pictures(tmp_path, 3)
    _text, report = _capsule(tmp_path, monkeypatch)
    assert report["products"]["pictures_on_disk"] == 3
    assert report["products"]["pictures_on_disk_error"] is None


def test_the_report_carries_the_same_lines(tmp_path, monkeypatch):
    """``report.json`` and the refusal are one run's two documents and
    must not disagree about why the pictures are missing."""

    _text, report = _capsule(tmp_path, monkeypatch)
    assert report["products"]["status"] == "FAILED"
    assert "exact-time ordinal axis" in report["products"]["renderer_output"]
    assert report["result"] == "PASS", "the forecast's verdict is its own"
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["products"] == report["products"]


def test_a_stage_that_said_nothing_adds_no_heading(tmp_path, monkeypatch):
    """An empty heading over nothing is worse than no heading."""

    text, report = _capsule(tmp_path, monkeypatch, diagnostic="")
    assert "The render stage said:" not in text
    assert report["products"]["renderer_output"] == ""


def test_the_stage_failure_carries_its_tail_on_the_exception():
    """The contract the capsule reads."""

    failure = GoStageFailed(1, "the last line")
    assert (failure.code, failure.diagnostic) == (1, "the last line")
    assert GoStageFailed(1).diagnostic == ""


# ---------------------------------------------------------------------------
# The OTHER capsule: a child that did not finish at all
# ---------------------------------------------------------------------------
#
# WHAT BREAKAGE THIS PINS (gate law).  A child that stopped mid-run was
# told its pictures had been removed and that the frames could be drawn
# by hand.  Both halves sent the reader to redraw work that was already
# on disk, and the second half is the wrong instruction for the reader
# who has just lost a forecast: what they want is the pictures of the
# part that ran.


def test_the_stopped_capsule_points_at_the_pictures_and_the_banner():
    kept = {"pictures": 20, "render": r"C:\runs\child\png",
            "banner": r"C:\runs\child\png\DID-NOT-FINISH.txt"}
    text = offline_child_run._did_not_finish_capsule(kept)
    assert "20 pictures" in text
    assert r"Next: open C:\runs\child\png" in text
    assert "DID-NOT-FINISH.txt" in text
    assert "drawn by hand" not in text
    assert "removed" not in text


def test_the_stopped_capsule_counts_one_picture_in_the_singular():
    """The WHOLE clause agrees, verb included.  A count made singular
    over a plural verb is the same carelessness the parenthesised
    plural was, in a sentence a reader who has just lost a forecast
    reads first."""

    kept = {"pictures": 1, "render": "/runs/child/png",
            "banner": "/runs/child/png/DID-NOT-FINISH.txt"}
    text = offline_child_run._did_not_finish_capsule(kept)
    assert "the 1 picture drawn while it ran is kept" in text
    assert "are kept" not in text


def test_the_stopped_capsule_keeps_the_plural_for_several_pictures():
    kept = {"pictures": 20, "render": "/runs/child/png",
            "banner": "/runs/child/png/DID-NOT-FINISH.txt"}
    text = offline_child_run._did_not_finish_capsule(kept)
    assert "the 20 pictures drawn while it ran are kept" in text


def test_a_stopped_child_that_drew_nothing_says_there_is_none_to_keep():
    """A count that is zero is not the sentence about kept pictures: a
    reader sent to an empty folder learns nothing."""

    text = offline_child_run._did_not_finish_capsule(
        {"pictures": 0, "render": "/runs/child/png", "banner": None})
    assert "No picture had been drawn yet" in text
    assert "Next: open" not in text


def test_a_banner_that_could_not_be_written_still_names_the_folder():
    text = offline_child_run._did_not_finish_capsule(
        {"pictures": 4, "render": "/runs/child/png", "banner": None})
    assert "Next: open /runs/child/png" in text
    assert "DID-NOT-FINISH" not in text


def test_the_finished_childs_render_failure_still_points_at_redrawing(
        tmp_path, monkeypatch):
    """The two capsules are for two different runs and must not converge.

    This one is a forecast that REACHED ITS END and whose render stage
    then exited nonzero: there is no stop to report, the frames are all
    there, and drawing them by hand is exactly the right next step.
    """

    text, _report = _capsule(tmp_path, monkeypatch)
    assert "Next: draw the saved frames by hand" in text
    assert "DID-NOT-FINISH" not in text


def test_the_stop_reason_is_the_refusals_first_sentence():
    """The banner quotes one sentence; the whole capsule is already on
    the stream's own failure event."""

    first = offline_child_run._first_sentence
    assert first(RuntimeError(
        "offline child became non-finite at step 900")) == (
        "offline child became non-finite at step 900")
    assert first(RuntimeError(
        "The child stopped.  Next: read the log.")) == "The child stopped"
    assert first(RuntimeError("  a\n  b  ")) == "a b"
    assert first(KeyboardInterrupt()) == "KeyboardInterrupt"
    assert len(first(RuntimeError("x" * 900))) == 400
