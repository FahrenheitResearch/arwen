"""A run that did not finish keeps its pictures, under a banner.

WHAT BREAKAGE THESE PIN (gate law).  A downscale child drew its analysis
frame early, stopped part way through its forecast, and the door then
removed every picture it had published -- so the pictures that would
have shown a reader what the forecast was doing before it stopped
existed only until the moment they became useful.  The frames survived,
but a reader opening the render directory found nothing at all, which
reads as "this run drew nothing" rather than "this run stopped".

Four groups: the pictures stay, the banner says where the forecast
stopped, the render summary carries the same status the banner does,
and the names this module publishes all resolve on it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gpuwm import first_products, render_receipts


_WHY = "offline child became non-finite at step 900"

_STOPPED = {"model_seconds": 2700.0, "run_seconds": 3600.0,
            "step": 900, "total_steps": 1200}

_FRAMES = ["wrfout_d02_1974-04-03_12_00_00",
           "wrfout_d02_1974-04-03_12_15_00",
           "wrfout_d02_1974-04-03_12_30_00"]


def _drawn(render_dir: Path, count: int) -> None:
    """A picture tree in this run's own nested layout."""

    folder = render_dir / "d02" / "composite_reflectivity" / "1974-04-03"
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (folder / f"picture-{index}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n")


def _keep(render_dir: Path, **over):
    fields = {"why": _WHY, "stopped": dict(_STOPPED), "frames": list(_FRAMES)}
    fields.update(over)
    return first_products.keep(render_dir, **fields)


def _banner(render_dir: Path) -> str:
    return (render_dir / first_products.DID_NOT_FINISH_BANNER).read_text(
        encoding="utf-8")


# ---------------------------------------------------------------------------
# 1.  The pictures stay
# ---------------------------------------------------------------------------


def test_the_pictures_the_early_render_drew_are_still_there(tmp_path):
    render_dir = tmp_path / "png"
    _drawn(render_dir, 20)
    kept = _keep(render_dir)
    assert kept["pictures"] == 20
    assert len(list(render_dir.rglob("*.png"))) == 20


def test_nothing_under_the_render_directory_is_removed(tmp_path):
    """Not only the pictures.  The early render's own receipt is the
    document that licenses the finalize stage to skip a frame, and a
    reader of a stopped run needs it as much as a reader of a finished
    one does."""

    render_dir = tmp_path / "png"
    _drawn(render_dir, 3)
    receipt = render_dir / first_products.FIRST_PRODUCTS_RECEIPT
    receipt.write_text('{"schema": "gpuwm.first-products.v1"}\n',
                       encoding="utf-8", newline="\n")
    _keep(render_dir)
    assert receipt.is_file()


def test_a_run_that_drew_nothing_is_counted_as_nothing(tmp_path):
    kept = _keep(tmp_path / "png")
    assert kept["pictures"] == 0


def test_a_render_directory_that_cannot_hold_a_banner_is_not_a_failure(
        tmp_path):
    """Best effort, exactly like the removal it replaces: a banner that
    cannot be written must not fail a run that has already failed.

    The count on this tree is the unreadable reading rather than a
    zero -- a path that is a regular file is a tree nobody can list --
    and the point pinned here is that neither answer raises.
    """

    blocked = tmp_path / "png"
    blocked.write_bytes(b"")
    kept = _keep(blocked)
    assert kept["banner"] is None
    assert kept["pictures"] is None
    assert kept["pictures_error"]


# ---------------------------------------------------------------------------
# 2.  The banner
# ---------------------------------------------------------------------------


def test_the_banner_is_a_plain_file_at_the_top_of_the_render_directory(
        tmp_path):
    render_dir = tmp_path / "png"
    _drawn(render_dir, 20)
    kept = _keep(render_dir)
    banner = render_dir / first_products.DID_NOT_FINISH_BANNER
    assert banner.is_file()
    assert kept["banner"] == str(banner)
    assert banner.read_bytes().count(b"\r\n") == 0


def test_the_banner_says_where_the_forecast_stopped(tmp_path):
    render_dir = tmp_path / "png"
    _drawn(render_dir, 20)
    _keep(render_dir)
    text = _banner(render_dir)
    assert "model second 2,700 of 3,600" in text
    assert "step 900 of 1,200" in text
    assert _WHY in text
    assert "20 pictures are in this folder" in text
    assert "3 frames were written before the stop" in text
    for frame in _FRAMES:
        assert frame in text


def test_one_picture_and_one_frame_are_counted_in_the_singular(tmp_path):
    """A banner is read by a person, so it says `1 picture is` rather
    than a parenthesised plural -- and the WHOLE clause agrees, pronoun
    included.  A count made singular over a sentence that kept its
    plural reads as carelessly as the parenthesis did, on a file whose
    reader has just lost a forecast.
    """

    render_dir = tmp_path / "png"
    _drawn(render_dir, 1)
    _keep(render_dir, frames=[_FRAMES[0]])
    text = _banner(render_dir)
    assert ("1 picture is in this folder.  It was drawn before the "
            "forecast stopped, from the frame below.") in text
    assert "1 frame was written before the stop" in text
    assert "(s)" not in text
    assert "Every one of them" not in text
    assert "frames below" not in text


def test_a_banner_over_several_pictures_keeps_the_plural_clause(tmp_path):
    render_dir = tmp_path / "png"
    _drawn(render_dir, 4)
    _keep(render_dir, frames=[_FRAMES[0]])
    text = _banner(render_dir)
    assert ("4 pictures are in this folder.  Every one of them was drawn "
            "before the forecast stopped, from the frame below.") in text


def test_a_banner_over_no_pictures_says_there_are_none(tmp_path):
    """A count of zero is not "every one of them": the early render had
    published nothing when the forecast stopped, and a banner that says
    so sends its reader to the frames instead of to an empty folder."""

    render_dir = tmp_path / "png"
    _keep(render_dir)
    text = _banner(render_dir)
    assert "No pictures are in this folder" in text
    assert "Every one of them" not in text
    assert "0 pictures" not in text


def test_a_banner_over_no_pictures_and_no_frames_does_not_contradict_itself(
        tmp_path):
    """The one case the two blocks share, read whole.

    A child that stopped before its first committed frame has neither
    pictures nor frames.  The zero-picture clause used to end "What it
    did write is named below." and the frame block below it then said
    "No frames were written before the stop.", so the file a reader
    opens after losing a forecast promised a list and denied it in the
    next line.  The frame block alone says what was named.
    """

    render_dir = tmp_path / "png"
    _keep(render_dir, frames=[])
    text = _banner(render_dir)
    assert ("No pictures are in this folder: the early render had not "
            "published one before the forecast stopped.") in text
    assert "No frames were written before the stop." in text
    assert "named below" not in text
    assert "below:" not in text
    # And nothing further down promises frames either.
    assert "The frames are kept too" not in text
    assert "drawn again" not in text


def test_a_banner_over_frames_and_no_picture_does_not_say_they_are_here(
        tmp_path):
    """The paragraph under the frame list is about PICTURES.

    "The frames are kept too" and "this folder is what the run had
    already drawn for itself" both read, over a folder holding frames
    and no picture, as though the pictures were in it -- which is the
    one thing this banner exists to keep a reader from concluding.  The
    clause is gated on there being a picture, and the folder with none
    gets the sentence that is true of it.
    """

    render_dir = tmp_path / "png"
    _keep(render_dir)
    text = _banner(render_dir)
    assert "No pictures are in this folder" in text
    assert "The frames are kept too" not in text
    assert "what the run had already drawn" not in text
    assert ("The frames are kept, and pictures can be drawn from them at "
            "any time") in text


def _wall(path: Path) -> bool:
    """Make ``path`` unlistable, and say whether the wall held.

    A mode is not a wall everywhere: root ignores it, and so do some
    filesystems this tree is run on.  The tests below need a directory
    a listing really is refused on, so they ask for one and skip when
    the platform does not give them one, rather than asserting a wall
    that was never built.
    """

    try:
        os.chmod(path, 0o000)
    except OSError:
        return False
    try:
        os.listdir(path)
    except OSError:
        return True
    return False


def test_a_picture_tree_that_cannot_be_listed_is_not_a_tree_with_none(
        tmp_path):
    """WHAT BREAKAGE THIS PINS (gate law).  A listing that FAILED was
    swallowed and counted as zero, so a reader whose pictures sat behind
    a permission wall or a dropped mount was told by the banner, by the
    render summary and by the report that the run had drawn none -- the
    one sentence that sends them off to re-draw a whole child.  The
    failed-render capsule already separates the two readings; so does
    this one now.

    ON A REAL TREE, because the counter this pins is the one that reads
    the disk.  Driven through a stub the previous form of this test
    passed over a counter that swallowed every refusal a real directory
    raises: the reader it used walks with ``rglob``, which drops a
    refused ``scandir`` on the floor, so all three walled trees below
    came back as zero pictures with no error at all.
    """

    render_dir = tmp_path / "png"
    render_dir.mkdir(parents=True)
    _drawn(render_dir, 2)
    walled = render_dir / "d02" / "composite_reflectivity"
    if not _wall(walled):
        pytest.skip("this platform lists a directory at mode 000")
    try:
        kept = _keep(render_dir)

        assert kept["pictures"] is None
        assert "d02" in kept["pictures_error"]
        text = _banner(render_dir)
        assert "could not be listed" in text
        assert "No pictures are in this folder" not in text
        assert "0 pictures" not in text
        summary = render_receipts.read_summary(render_dir)
        assert summary["pictures_on_disk"] is None
        assert summary["pictures_on_disk_error"] == kept["pictures_error"]
        assert "could not be listed" in summary["count_basis"]
    finally:
        os.chmod(walled, 0o755)
    # And the pictures are still there: nothing was removed.
    assert len(list(render_dir.rglob("*.png"))) == 2


def test_a_render_root_that_cannot_be_listed_is_not_a_tree_with_none(
        tmp_path):
    """The same refusal at the top of the tree rather than inside it.

    The banner cannot be written into a folder nothing may write to, and
    that is already best effort; what may not happen is the count coming
    back as zero, which is the sentence the whole reading exists to keep
    a reader from being handed.
    """

    render_dir = tmp_path / "png"
    render_dir.mkdir(parents=True)
    _drawn(render_dir, 1)
    if not _wall(render_dir):
        pytest.skip("this platform lists a directory at mode 000")
    try:
        kept = _keep(render_dir)

        assert kept["pictures"] is None
        assert kept["pictures_error"]
        assert str(render_dir) in kept["pictures_error"]
    finally:
        os.chmod(render_dir, 0o755)
    assert len(list(render_dir.rglob("*.png"))) == 1


def test_a_render_path_that_is_a_file_is_not_an_empty_tree(tmp_path):
    """A path that is a regular file is a tree nobody can list.

    It is the third way the count cannot be taken, it needs no mode to
    reproduce, and the reader this replaced answered it with ``0``
    because "not a directory" was its own early return.
    """

    render_dir = tmp_path / "png"
    render_dir.write_bytes(b"")

    kept = _keep(render_dir)

    assert kept["pictures"] is None
    assert str(render_dir) in kept["pictures_error"]
    assert kept["banner"] is None


def test_the_early_renders_scratch_is_not_counted_as_a_picture(tmp_path):
    """A temporary under the dotted scratch is not published.

    The early render draws into it and moves each picture onto its final
    name afterwards, so a file still in there is one nobody may look at;
    counting it would put a number in the banner that does not match
    what the folder shows.
    """

    render_dir = tmp_path / "png"
    _drawn(render_dir, 1)
    scratch = render_dir / ".first-products-scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "half-written.png").write_bytes(b"")

    kept = _keep(render_dir)

    assert kept["pictures"] == 1
    assert "1 picture is in this folder" in _banner(render_dir)


def test_a_directory_that_was_never_created_is_still_the_empty_reading(
        tmp_path):
    """A render that never made its output directory drew nothing.  That
    is a reading, not a failure to read, so it keeps the zero."""

    kept = _keep(tmp_path / "png")
    assert kept["pictures"] == 0
    assert kept["pictures_error"] is None


def test_the_stop_point_never_prints_as_a_mantissa(tmp_path):
    """A long or fine child still reads as numbers a person can use.

    ``%g`` goes exponential at a million, which a step count passes at
    about 28 hours of forecast on a step of 0.1 s and a model second
    passes at 11.6 days, so the one sentence this file exists for
    printed "model second 1.08e+06 of 1.2e+06, after step 1.2e+06 of
    1.5e+06" to a reader who had just lost the run.
    """

    text = first_products.banner_text(
        why=_WHY, frames=list(_FRAMES), pictures=20,
        stopped={"model_seconds": 1080000.0, "run_seconds": 1200000.0,
                 "step": 1200000, "total_steps": 1500000})
    assert ("The forecast stopped at model second 1,080,000 of 1,200,000, "
            "after step 1,200,000 of 1,500,000.") in text
    assert "e+" not in text


def test_a_step_count_carried_as_a_float_still_prints_as_an_integer(tmp_path):
    """Whatever reaches the banner, a count is written as a count.

    The progress sample keeps the two counts as integers, and the
    formatter holds the same rule for anything else that gets here, so
    a caller that carried them as floats cannot put a decimal point or
    a mantissa into the sentence.
    """

    text = first_products.banner_text(
        why=_WHY, frames=[], pictures=0,
        stopped={"model_seconds": 1080000.0, "run_seconds": 1200000.0,
                 "step": 1200000.0, "total_steps": 1500000.0})
    assert "after step 1,200,000 of 1,500,000." in text
    assert "1200000.0" not in text
    assert "e+" not in text


def test_a_sub_second_stop_keeps_its_fraction(tmp_path):
    """A fixed format is not a rounded one: a child that stopped inside
    its first second still says where."""

    text = first_products.banner_text(
        why=_WHY, frames=[], pictures=0,
        stopped={"model_seconds": 0.5, "run_seconds": 900.0,
                 "step": 5, "total_steps": 300})
    assert "model second 0.5 of 900, after step 5 of 300." in text


def test_the_banner_says_every_picture_here_predates_the_stop(tmp_path):
    render_dir = tmp_path / "png"
    _drawn(render_dir, 4)
    _keep(render_dir)
    text = _banner(render_dir)
    assert "before the forecast stopped" in text


def test_the_banner_says_the_stop_point_was_not_recorded_when_it_was_not(
        tmp_path):
    """A child that refused itself before a single step has no model
    second to quote, and a banner that invented one would be a number
    nobody measured."""

    render_dir = tmp_path / "png"
    _drawn(render_dir, 1)
    _keep(render_dir, stopped=None, frames=[])
    text = _banner(render_dir)
    assert "not recorded" in text
    assert "of 3600" not in text
    assert "of 3,600" not in text


def test_a_long_frame_list_is_bounded_and_says_how_many_it_dropped(tmp_path):
    render_dir = tmp_path / "png"
    frames = [f"wrfout_d02_frame_{index:04d}" for index in range(200)]
    _keep(render_dir, frames=frames)
    text = _banner(render_dir)
    assert "200 frames were written before the stop" in text
    assert "wrfout_d02_frame_0000" in text
    assert "wrfout_d02_frame_0199" not in text
    assert "more" in text


# ---------------------------------------------------------------------------
# 3.  The render summary carries the same status
# ---------------------------------------------------------------------------


def test_the_render_summary_carries_the_status_and_the_count(tmp_path):
    render_dir = tmp_path / "png"
    _drawn(render_dir, 20)
    _keep(render_dir)
    summary = json.loads(
        (render_dir / render_receipts.SUMMARY_FILENAME).read_text(
            encoding="utf-8"))
    assert summary["schema"] == render_receipts.SUMMARY_SCHEMA
    assert summary["status"] == first_products.DID_NOT_FINISH_STATUS
    assert summary["pictures_on_disk"] == 20
    assert summary["banner_path"] == str(
        render_dir / first_products.DID_NOT_FINISH_BANNER)


def test_an_existing_summary_keeps_every_row_it_already_carried(tmp_path):
    """The early render's own summary is that render's record and is
    amended, not replaced: a reader that keyed on its families must not
    find them gone because the run stopped."""

    render_dir = tmp_path / "png"
    _drawn(render_dir, 2)
    existing = {"schema": render_receipts.SUMMARY_SCHEMA,
                "summary_path": str(render_dir
                                    / render_receipts.SUMMARY_FILENAME),
                "rendered_png_count": 2, "rendered_family_count": 1,
                "rendered_families": [{"name": "composite_reflectivity",
                                       "count": 2}],
                "skipped_families": [], "failures": [], "receipt_paths": []}
    (render_dir / render_receipts.SUMMARY_FILENAME).write_text(
        json.dumps(existing) + "\n", encoding="utf-8", newline="\n")
    _keep(render_dir)
    summary = render_receipts.read_summary(render_dir)
    assert summary["rendered_families"] == existing["rendered_families"]
    assert summary["status"] == first_products.DID_NOT_FINISH_STATUS
    assert summary["pictures_on_disk"] == 2


def test_the_summary_is_stamped_even_when_the_banner_cannot_be_written(
        tmp_path):
    """Two independent best-effort writes, not one gated on the other.
    A banner that cannot be written (something already occupies its
    name) says nothing about whether the summary can be, and the
    summary is the document the desktop's native-plots door and every
    run browser key on -- losing the status with the banner would hide
    the stop from the reader most likely to see it.
    """

    render_dir = tmp_path / "png"
    _drawn(render_dir, 3)
    (render_dir / first_products.DID_NOT_FINISH_BANNER).mkdir()
    kept = _keep(render_dir)
    assert kept["banner"] is None
    assert kept["pictures"] == 3
    summary = render_receipts.read_summary(render_dir)
    assert summary["status"] == first_products.DID_NOT_FINISH_STATUS
    assert summary["pictures_on_disk"] == 3
    assert summary["banner_path"] is None


def test_a_summary_written_where_there_was_none_names_how_it_was_counted(
        tmp_path):
    """The desktop's native-plots door opens on the presence of this
    document (int-d106 src/native_plot_link.rs, ``products_at``), so a
    stopped run that drew pictures and published no summary of its own
    would leave that door shut over pictures that are on disk."""

    render_dir = tmp_path / "png"
    _drawn(render_dir, 7)
    _keep(render_dir)
    summary = render_receipts.read_summary(render_dir)
    assert summary["rendered_png_count"] == 7
    assert "counted" in summary["count_basis"]


# ---------------------------------------------------------------------------
# 4.  Every published name resolves
# ---------------------------------------------------------------------------


def test_every_name_this_module_publishes_resolves_on_it():
    """WHAT BREAKAGE THIS PINS (gate law): a rename that swept the
    callers and missed the export list.  ``withdraw`` was replaced by
    ``keep`` in this change, every caller moved, the whole battery
    stayed green, and ``from gpuwm.first_products import *`` raised
    ``AttributeError`` because the name was still listed.  A star import
    is the one caller no grep for the identifier finds.
    """

    missing = [name for name in first_products.__all__
               if not hasattr(first_products, name)]
    assert missing == []


def test_this_change_s_own_names_are_published():
    """The other half: a name the rest of the tree imports and the
    module does not publish is the same defect facing the other way."""

    for name in ("keep", "banner_text", "DID_NOT_FINISH_BANNER",
                 "DID_NOT_FINISH_STATUS"):
        assert name in first_products.__all__
    assert "withdraw" not in first_products.__all__


def test_no_module_in_this_package_publishes_a_name_it_does_not_have():
    """The gate, package wide, because the miss above was not special to
    one module and walking the package is what makes the next one
    impossible to leave behind.  A module that cannot be imported at all
    here (an optional dependency) is skipped: it cannot be star-imported
    in this environment either.
    """

    import importlib
    import pkgutil

    import gpuwm

    broken = []
    for info in pkgutil.walk_packages(gpuwm.__path__, prefix="gpuwm."):
        try:
            module = importlib.import_module(info.name)
        except Exception:      # optional deps (cupy, wrf-rust) may be absent
            continue
        for name in getattr(module, "__all__", ()) or ():
            if not hasattr(module, name):
                broken.append(f"{info.name}.{name}")
    assert broken == []
