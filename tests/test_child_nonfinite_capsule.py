"""A child that blew up says WHAT blew up, WHERE, and how it got there.

WHAT BREAKAGE THESE PIN (gate law).  A downscaled child that stopped
being finite said "offline child became non-finite at step N" and nothing
else: no field, no cell, no model time, and none of the trend its own
health record was already holding.  On the run these tests are built from
-- a 798 x 798 child at 83.33 m on its parent's 49 levels -- the record
held six checks of w_max climbing 10.73, 13.22, 15.77, 18.12, 21.05,
22.97 m/s over the five minutes before the end, with the CFL flat in the
0.19-0.21 band, and every one of those numbers was dropped at the moment
it mattered.  The run also wrote no ``report.json`` at all, so the one
outcome a reader most needs to read afterwards was the only one that left
no document behind.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm import offline_child_run
from gpuwm.config import RunConfig
from gpuwm.offline_child_run import (
    OfflineChildNonFinite,
    describe_nonfinite_child,
    nonfinite_trend_checks,
)


def _row(step, model_seconds, w_max, cfl):
    """One health row in the carrying shape the run loop records.

    A number travels as itself beside ``"measured"``; the two kinds of
    non-number travel as ``null`` beside the word that says which kind,
    so nothing in this document can reach a serializer as a non-finite
    float.  ``test_the_terminal_row_is_the_shape_the_product_writes``
    holds this spelling to ``child_health_log_fields``' own output.
    """

    return {"step": step, "model_seconds": model_seconds,
            "w_max": w_max, "cfl": cfl,
            "w_max_state": "measured" if w_max is not None else "non-finite",
            "cfl_state": "measured" if cfl is not None else "not computed"}


#: The health record the user's own run wrote, read straight off its log:
#: ``child_step`` lines at the 60-second cadence over the five minutes
#: before the end, then the check that found the fields gone.  Every
#: number here is that log's.
_TREND = [
    _row(5760, 2400.0, 10.733054161071777, 0.1903411102294922),
    _row(5904, 2460.0, 13.220698356628418, 0.1923489570617676),
    _row(6048, 2520.0, 15.768431663513184, 0.19554689407348635),
    _row(6192, 2580.0, 18.120616912841797, 0.19874500274658208),
    _row(6336, 2640.0, 21.04691505432129, 0.20257333755493168),
    _row(6480, 2700.0, 22.971025466918945, 0.208536434173584),
    _row(6624, 2760.0, None, None),
]


def _survey(count=1, field="W"):
    return {
        "surveyed": ["W", "U", "V", "T"],
        "fields": [{
            "field": field, "carrier": field.lower(),
            "shape": [50, 798, 798], "size": 31840200, "count": count,
            "first_cell": {"k": 12, "j": 401, "i": 388},
            "bounding_box": {"k": [10, 14], "j": [398, 404],
                             "i": [385, 391]},
        }],
    }


def _capsule(**overrides):
    fields = dict(
        step=6624, total_steps=69120, model_seconds=2760.0,
        run_seconds=28800.0, cadence_seconds=60.0, trend=_TREND,
        survey=_survey())
    fields.update(overrides)
    return describe_nonfinite_child(**fields)


# --- the sentence ------------------------------------------------------


def test_the_sentence_carries_the_climb_the_record_already_held():
    """Six checks of w_max, in the order they were taken."""

    summary = _capsule()["summary"]
    assert summary.startswith("The child blew up:")
    for reading in ("10.73", "13.22", "15.77", "18.12", "21.05", "22.97"):
        assert reading in summary, summary
    assert "m/s over the 300 model seconds before" in summary


def test_the_sentence_names_the_field_and_the_cell():
    summary = _capsule()["summary"]
    assert "W went non-finite at cell (k=12, j=401, i=388)" in summary


def test_the_sentence_still_carries_the_step_and_adds_the_model_second():
    """The one fact the old refusal had is not lost to the new ones."""

    summary = _capsule()["summary"]
    assert "step 6624 of 69120" in summary
    assert "model second 2760 of 28800" in summary


def test_several_non_finite_carriers_are_all_named():
    survey = _survey()
    survey["fields"].append({
        "field": "T", "carrier": "thp", "shape": [49, 798, 798],
        "size": 31201596, "count": 3,
        "first_cell": {"k": 12, "j": 401, "i": 388},
        "bounding_box": {"k": [11, 13], "j": [400, 402], "i": [387, 389]}})
    summary = _capsule(survey=survey)["summary"]
    assert "W went non-finite at cell (k=12, j=401, i=388)" in summary
    # After the full stop, so the model second and the step stay on the
    # one line a run view shows of a first sentence.
    assert summary.endswith("step 6624 of 69120.  T went with it.")


# --- the block under it ------------------------------------------------


def test_one_bad_cell_is_named_and_not_boxed():
    """A bounding box around a single cell is that cell written twice."""

    message = _capsule()["message"]
    assert "W: 1 cell at (k=12, j=401, i=388)" in message
    assert "all inside" not in message


def test_many_bad_cells_are_counted_and_boxed():
    message = _capsule(survey=_survey(count=4812))["message"]
    assert "W: 4,812 cells of 31,840,200" in message
    assert "first at (k=12, j=401, i=388)" in message
    assert "all inside k 10-14, j 398-404, i 385-391" in message


def test_the_trend_table_carries_w_max_and_the_cfl_of_every_check():
    """The CFL is in the capsule because its FLATNESS is the reading: it
    never left the 0.19-0.21 band while w_max doubled, which says the
    time step was not what ran out."""

    message = _capsule()["message"]
    assert "The last 7 health checks, 60 model seconds apart:" in message
    assert "step 6480  model second 2700  w_max 22.97 m/s  CFL 0.2085" in message
    # And no unit on a row that carries no quantity, in EITHER spelling.
    assert "w_max non-finite m/s" not in message
    assert "w_max not computed m/s" not in message
    # The check that FOUND it is in the table, with the two words the
    # record carries for a non-finite state rather than a hole.
    assert ("step 6624  model second 2760  w_max non-finite  "
            "CFL not computed") in message


def test_the_terminal_row_is_the_shape_the_product_writes():
    """THE FIXTURE IS NOT ITS OWN AUTHORITY.

    Every row above is a hand-written spelling of what the run loop
    records, so the one row whose spelling is the subject of these tests
    is held to the function that writes it: the decoder's own output for
    a record whose fields have gone.
    """

    from gpuwm.core.dycore import decode_stability_record
    from gpuwm.offline_child_run import child_health_log_fields

    gone = child_health_log_fields(decode_stability_record(
        np.array([np.inf, np.nan, np.nan, 0.0, 0.0, np.nan, 0.0, 0.0],
                 dtype=np.float64), cfg=None))
    terminal = _TREND[-1]
    assert gone["w_max"] == terminal["w_max"] is None
    assert gone["w_max_state"] == terminal["w_max_state"] == "non-finite"
    assert gone["cfl"] == terminal["cfl"] is None
    assert gone["cfl_state"] == terminal["cfl_state"] == "not computed"


def test_a_reading_nothing_computed_reads_as_that_and_carries_no_unit():
    """THE OTHER NON-NUMBER.  "non-finite" is a number that went;
    "not computed" is a number nothing ever produced, and the unit guard
    used to suppress the unit for the first spelling only, so a row with
    no quantity at all was printed as ``w_max not computed m/s``."""

    trend = list(_TREND[:-1]) + [
        {"step": 6624, "model_seconds": 2760.0,
         "w_max": None, "w_max_state": "not computed",
         "cfl": None, "cfl_state": "not computed"}]
    message = _capsule(trend=trend)["message"]
    assert ("step 6624  model second 2760  w_max not computed  "
            "CFL not computed") in message
    assert "not computed m/s" not in message


def test_the_capsule_carries_no_non_finite_float_however_it_was_built():
    """A row handed in the old shape is normalised on the way in, so the
    document this composes is strict JSON whoever built the row."""

    import json

    trend = list(_TREND[:-1]) + [
        {"step": 6624, "model_seconds": 2760.0,
         "w_max": float("nan"), "cfl": None}]
    capsule = _capsule(trend=trend)
    terminal = capsule["trend"][-1]
    assert terminal["w_max"] is None
    assert terminal["w_max_state"] == "non-finite"
    assert "w_max non-finite" in capsule["message"]
    assert "NaN" not in json.dumps(capsule, allow_nan=False)


def test_the_report_a_blown_up_child_writes_is_strict_json(tmp_path):
    """WHAT BREAKAGE THIS PINS (gate law).  ``NaN`` is not a JSON token.
    RFC 8259 has no spelling for it, so ``JSON.parse``, ``serde_json``,
    ``encoding/json`` and ``jq`` refuse a document carrying one while
    Python's default parse accepts it -- and this is the ONLY outcome
    that writes ``report.json`` for a child, so every blown-up child
    published a document no strict reader could open, under a claim that
    the report carries the same facts a reader can open afterwards.
    """

    import json

    from gpuwm.offline_child_run import (_ChildProgress,
                                         _publish_failure_report)

    def refuse(token):
        raise ValueError(f"invalid JSON token: {token}")

    progress = _ChildProgress()
    progress.outdir = tmp_path
    trend = list(_TREND[:-1]) + [
        {"step": 6624, "model_seconds": 2760.0,
         "w_max": float("nan"), "cfl": None}]
    _publish_failure_report(
        progress, _capsule(trend=trend),
        kept={"pictures": 20, "pictures_error": None,
              "render": str(tmp_path / "png"),
              "banner": str(tmp_path / "png" / "DID-NOT-FINISH.txt")})

    text = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    report = json.loads(text, parse_constant=refuse)
    assert report["result"] == "FAIL"
    terminal = report["failure"]["trend"][-1]
    assert terminal["w_max"] is None
    assert terminal["w_max_state"] == "non-finite"
    assert terminal["cfl_state"] == "not computed"


def test_the_window_is_a_duration_and_not_a_row_count():
    """N comes from the record's own cadence, so the sentence "over the
    last 300 model seconds" stays true at any --health-interval-seconds."""

    assert nonfinite_trend_checks(60.0) == 6
    assert nonfinite_trend_checks(300.0) == 2
    assert nonfinite_trend_checks(1.0) == 12      # capped, not 301
    assert nonfinite_trend_checks(0.0) == 2       # a cadence it cannot use
    assert nonfinite_trend_checks(None) == 2


def test_a_coarse_cadence_quotes_what_it_has():
    coarse = [dict(row) for row in _TREND[::5]]
    message = _capsule(cadence_seconds=300.0, trend=coarse)["message"]
    assert "The last 2 health checks, 300 model seconds apart:" in message


# --- the way out -------------------------------------------------------


def test_the_refusal_names_the_frames_that_are_on_disk():
    """Gate law's other half: a refusal names a way forward."""

    message = _capsule(render_command="gpuwm render out/child --series")["message"]
    assert "Next: every frame the run did reach is on disk" in message
    assert "gpuwm render out/child --series" in message


def test_without_a_render_plan_the_way_out_is_still_named():
    message = _capsule()["message"]
    assert "gpuwm render" in message
    assert "docs/public/DOWNSCALE.md" in message


def test_an_les_shaped_child_is_told_what_shape_it_was():
    from gpuwm.offline_child import les_child_regime

    regime = les_child_regime(
        _les_cfg(), inherits_parent_levels=True, parent_levels=49)
    message = _capsule(regime=regime)["message"]
    assert "This child's shape:" in message
    assert "83.3333 m spacing" in message
    assert "--child-levels N,STRETCH" in message
    assert "km_opt = 3 (3-D Smagorinsky)" in message


def _les_cfg():
    return RunConfig(
        nx=96, ny=96, nz=49, dx=83.33333333333333, dy=83.33333333333333,
        ztop=20000.0, dt=0.4166666666666667, run_seconds=600.0,
        moist=True, mp_physics=10, bl_pbl_physics=1, km_opt=4)


# --- the survey, run against a real DomainState ------------------------


def _domain_state(nz=6, ny=8, nx=10):
    from gpuwm.core.state import DomainState

    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0, ztop=6400.0,
                    dt=0.5, run_seconds=10.0, moist=True)
    return DomainState(cfg, array_module=np)


def test_the_survey_names_the_carrier_and_the_cell_it_found():
    """Run against a REAL DomainState, not a stand-in for one: the survey
    has to read the carriers a child actually allocates."""

    from gpuwm.core.dycore import nonfinite_field_survey

    state = _domain_state()
    state.w[3, 4, 5] = np.float32("nan")
    survey = nonfinite_field_survey(state)
    assert [entry["field"] for entry in survey["fields"]] == ["W"]
    entry = survey["fields"][0]
    assert entry["count"] == 1
    assert entry["first_cell"] == {"k": 3, "j": 4, "i": 5}
    assert entry["bounding_box"] == {"k": [3, 3], "j": [4, 4], "i": [5, 5]}
    assert "QVAPOR" in survey["surveyed"]


def test_the_survey_boxes_a_plume_and_counts_it():
    from gpuwm.core.dycore import nonfinite_field_survey

    state = _domain_state()
    state.w[2:5, 3:6, 4:7] = np.float32("inf")
    state.thp[1, 0, 0] = np.float32("nan")
    survey = nonfinite_field_survey(state)
    by_name = {entry["field"]: entry for entry in survey["fields"]}
    assert by_name["W"]["count"] == 27
    assert by_name["W"]["bounding_box"] == {
        "k": [2, 4], "j": [3, 5], "i": [4, 6]}
    assert by_name["T"]["count"] == 1
    assert by_name["T"]["first_cell"] == {"k": 1, "j": 0, "i": 0}


def test_a_two_dimensional_carrier_is_reported_without_a_fake_level():
    from gpuwm.core.dycore import nonfinite_field_survey

    state = _domain_state()
    state.mup[2, 3] = np.float32("nan")
    entry = nonfinite_field_survey(state)["fields"][0]
    assert entry["field"] == "MU"
    assert entry["first_cell"] == {"j": 2, "i": 3}


def test_a_finite_state_surveys_clean():
    """NEGATIVE CONTROL: the survey must not invent a finding."""

    from gpuwm.core.dycore import nonfinite_field_survey

    assert nonfinite_field_survey(_domain_state())["fields"] == []


def test_a_survey_that_cannot_be_taken_never_replaces_the_refusal():
    """THE WHOLE POINT of the survey: 2.7.5 died in its own diagnostic,
    in the health line immediately before the refusal it was decorating
    (``offline_child_run.py`` lines 1460 and 1477 of the 2.7.5 tree)."""

    class _Boom:
        @staticmethod
        def refresh_streamed_state(_stepper, _state):
            raise MemoryError("out of memory allocating 124.5 MiB")

    survey = offline_child_run._survey_nonfinite_child(_Boom(), None, None)
    assert survey["fields"] == []
    assert "MemoryError" in survey["error"]
    message = _capsule(survey=survey)["message"]
    assert "The field survey could not be taken" in message
    assert "out of memory" in message
    # And the sentence is still a sentence.
    assert _capsule(survey=survey)["summary"].startswith("The child blew up:")


# --- the two documents -------------------------------------------------


class _Published:
    """An early render that has already published and joins instantly.

    The render itself is not the subject here: what is pinned is what
    the stop arm does with a picture tree that exists, so the tree is
    written and the trigger only has to be joinable.
    """

    def wait(self, timeout=None):
        return None


def _start_progress(progress, root: Path, *, pictures: int = 0):
    from datetime import datetime

    outdir = root / "child-run"
    outdir.mkdir(parents=True, exist_ok=True)
    config = root / "child.toml"
    config.write_text("nx = 4\n", encoding="utf-8")
    progress.start(outdir=outdir, child_config=config, ratio=3,
                   start_time=datetime(1970, 1, 2, 12),
                   parent={"frames": 3}, name="Downscale of parent-run")
    if pictures:
        # The plan `arm_render` builds, and a tree in this run's own
        # nested layout, so `keep` counts a real directory.
        folder = (outdir / "png" / "d02" / "composite_reflectivity"
                  / "1974-04-03")
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(pictures):
            (folder / f"picture-{index}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        progress.render_plan = {"run": outdir, "wrfout_dir": outdir,
                                "render": outdir / "png",
                                "render_products": "all"}
        progress._first_products = _Published()
    return outdir


def test_the_report_and_the_refusal_are_the_same_facts(tmp_path, monkeypatch):
    """One run, one account of why it stopped -- and one folder.

    The report, the banner over the kept pictures and the render summary
    beside them are written from the same two things: the capsule the
    refusal carries and what the keep found on disk.  Nothing is removed
    from the picture folder on this path or on any other failure path.
    """

    from gpuwm import render_receipts, runplan
    from gpuwm.first_products import (DID_NOT_FINISH_BANNER,
                                      DID_NOT_FINISH_STATUS)

    capsule = _capsule(render_command="gpuwm render out/child --series")

    def explode(_args, progress):
        _start_progress(progress, tmp_path, pictures=2)
        raise OfflineChildNonFinite(capsule)

    monkeypatch.setattr(offline_child_run, "_run", explode)
    with pytest.raises(OfflineChildNonFinite):
        offline_child_run.run(object())

    outdir = tmp_path / "child-run"
    events = runplan.read_events(outdir / "events.jsonl")
    report = json.loads((outdir / "report.json").read_text(encoding="utf-8"))
    assert report["result"] == "FAIL"
    assert report["failure"]["kind"] == "non-finite"
    assert report["failure"]["step"] == 6624
    assert report["failure"]["summary"] == capsule["summary"]
    assert report["failure"]["fields"][0]["first_cell"] == {
        "k": 12, "j": 401, "i": 388}
    # The window's six checks, plus the one that found the fields gone.
    assert len(report["failure"]["trend"]) == 7
    # WHAT THE PICTURES BECAME: kept, counted, and pointed at.  The
    # early render published two before the child stopped and they are
    # still there; the block a reader opens says so and names the banner
    # standing over them, so the folder and the document agree.
    banner = outdir / "png" / DID_NOT_FINISH_BANNER
    assert report["products"]["status"] == "KEPT"
    assert report["products"]["run_status"] == DID_NOT_FINISH_STATUS
    assert report["products"]["pictures_on_disk"] == 2
    assert report["products"]["pictures_on_disk_error"] is None
    assert report["products"]["banner"] == str(banner)
    reason = report["products"]["reason"]
    assert f"Next: open {outdir / 'png'}" in reason
    assert "were removed" not in reason
    assert "can be drawn by hand" not in reason
    # NOTHING IS REMOVED on a failure path.
    assert sorted(path.name for path in (outdir / "png").rglob("*.png")) == [
        "picture-0.png", "picture-1.png"]
    assert banner.is_file()
    assert "THIS FORECAST DID NOT FINISH" in banner.read_text(encoding="utf-8")
    summary = render_receipts.read_summary(outdir / "png")
    assert summary["status"] == DID_NOT_FINISH_STATUS
    assert summary["pictures_on_disk"] == 2
    kept = [event for event in events if event.get("code")
            == "early_render_kept"]
    assert len(kept) == 1 and kept[0]["pictures"] == 2
    assert kept[0]["banner"] == str(banner)
    # The banner's "Why it stopped" is the capsule's own sentence, not
    # the first sentence of its whole text run together.
    assert kept[0]["why"] == capsule["summary"]
    assert capsule["summary"] in banner.read_text(encoding="utf-8")

    assert events[-1]["event"] == "failed"
    # THE FIRST SENTENCE, not the whole capsule run together on one line.
    assert events[-1]["message"] == (
        "OfflineChildNonFinite: " + capsule["summary"])
    assert "health checks" not in events[-1]["message"]


def test_a_stop_that_carries_no_capsule_publishes_the_report_too(
        tmp_path, monkeypatch):
    """WHAT BREAKAGE THIS PINS (gate law).  The document was written
    only where the exception had composed a capsule, so an interrupt, an
    ``OSError`` from a mount that dropped and a contract error raised
    mid-run each left a folder of kept pictures under a banner and no
    report beside it -- while the page, the layout document and the
    changelog row all said a child that stops inside its forecast
    publishes one.  A reader who was told the report is there and opens
    a directory that has none is the defect this arm was opened to
    remove, three exception classes narrower.

    The document such a stop gets carries what there is: the sentence
    the banner and the ``failed`` event carry, the whole of what the
    exception said, and the class that was raised.
    """

    from gpuwm.first_products import DID_NOT_FINISH_STATUS

    def explode(_args, progress):
        _start_progress(progress, tmp_path, pictures=2)
        raise RuntimeError("the card fell over.  It is not coming back.")

    monkeypatch.setattr(offline_child_run, "_run", explode)
    with pytest.raises(RuntimeError):
        offline_child_run.run(object())

    outdir = tmp_path / "child-run"
    report = json.loads((outdir / "report.json").read_text(encoding="utf-8"))
    assert report["result"] == "FAIL"
    assert report["failure"]["summary"] == "the card fell over"
    assert report["failure"]["message"] == (
        "the card fell over.  It is not coming back.")
    assert report["failure"]["error_type"] == "RuntimeError"
    # The run-plan failed event carries that same one sentence, with the
    # class in front, and not the whole text run together.
    events = [json.loads(line) for line in
              (outdir / "events.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    failed = [event for event in events if event.get("event") == "failed"]
    assert failed, events
    assert failed[-1]["message"] == "RuntimeError: the card fell over"
    # The pictures are kept and the document says so, exactly as it does
    # for a stop that did compose a capsule.
    assert report["products"]["status"] == "KEPT"
    assert report["products"]["run_status"] == DID_NOT_FINISH_STATUS
    assert report["products"]["pictures_on_disk"] == 2
    banner = outdir / "png" / "DID-NOT-FINISH.txt"
    assert report["products"]["banner"] == str(banner)
    assert banner.is_file()
    # And the directory is recognised as a child by the resume door, so
    # the reader of this one is answered like the reader of any other.
    from gpuwm.resume import offline_child_run_at

    child = offline_child_run_at(outdir)
    assert child is not None and child.finished is False
    assert child.failure == "the card fell over"


def test_the_refusal_prints_as_a_sentence_and_not_a_traceback():
    """``gpuwm.cli`` prints a ValueError at exit 2 with no traceback; a
    bare RuntimeError from this door prints a stack."""

    error = OfflineChildNonFinite(_capsule())
    assert isinstance(error, ValueError)
    assert str(error).startswith("The child blew up:")
    assert error.summary.endswith("step 6624 of 69120.")
