"""``dt`` in the step log: present under an adaptive clock, absent under a fixed one.

WHY IT IS CONDITIONAL.  Under a fixed step the timestep is a constant
sitting in the experiment config, so putting it on every step record
would be noise -- and it would move EVERY run's stream to v4, making a
v3-only consumer refuse streams whose shape it understands perfectly.
Under an adaptive step it is the run's most diagnostic quantity and it is
otherwise unrecoverable: differencing ``model_seconds`` is the only other
route, and ``StepLog.domain_step`` already documents why that derivation
is wrong for a delayed-start nest.

So the gate is the run's configuration, and the two properties that
matter are opposite ones: a fixed run must emit EXACTLY what it emitted
before this feature existed, and an adaptive run must emit the field AND
declare the schema that warns a consumer about it.
"""

from __future__ import annotations

import json
from datetime import datetime

from gpuwm.progress_log import (      # noqa: E402
    FRAME_MARKER_DIRNAME, STEP_LOG_FILENAME, STEP_LOG_SCHEMA,
    STEP_LOG_SCHEMA_ADAPTIVE, STEP_LOG_SCHEMAS, StepLog, format_step_line,
)

START = datetime(2025, 10, 25, 18, 0, 0)


def _run(tmp_path, *, adaptive, dts=(30.0, 31.5, 33.075)):
    log = StepLog(start_time=START, run_seconds=600.0, text_stream=None,
                  jsonl_path=tmp_path / STEP_LOG_FILENAME,
                  frame_marker_dir=tmp_path / FRAME_MARKER_DIRNAME,
                  adaptive_dt=adaptive)
    at = 0.0
    for n, dt in enumerate(dts, start=1):
        at += dt
        log.domain_step(grid_id=1, step_count=n, model_seconds=at,
                        step_wall_seconds=0.01, dt=dt)
    log.close(status="SUCCESS")
    rows = [json.loads(line) for line
            in (tmp_path / STEP_LOG_FILENAME).read_text().splitlines() if line]
    return [r for r in rows if r.get("event") == "step"]


# ------------------------------------------------------- the fixed case

def test_a_fixed_run_carries_no_dt_and_stays_on_v3(tmp_path):
    """The compatibility property: an existing consumer sees no change."""
    steps = _run(tmp_path, adaptive=False)
    assert steps, "no step records -- the gate saw an empty corpus"
    for r in steps:
        assert "dt" not in r, r
        assert r["schema"] == STEP_LOG_SCHEMA == "gpuwm.step-log/v3"
        assert " dt " not in r["text"]


def test_the_caller_may_pass_dt_on_a_fixed_run_and_it_is_still_dropped(tmp_path):
    """model.py passes dt unconditionally; the LOG decides.

    If the decision lived at the call site instead, the schema and the
    field could disagree -- two places to change, one of them silent.
    """
    steps = _run(tmp_path, adaptive=False, dts=(30.0, 30.0))
    assert all("dt" not in r for r in steps)


# ---------------------------------------------------- the adaptive case

def test_an_adaptive_run_carries_dt_and_declares_v4(tmp_path):
    steps = _run(tmp_path, adaptive=True)
    assert steps, "no step records -- the gate saw an empty corpus"
    assert [r["dt"] for r in steps] == [30.0, 31.5, 33.075]
    for r in steps:
        assert r["schema"] == STEP_LOG_SCHEMA_ADAPTIVE == "gpuwm.step-log/v4"


def test_the_human_line_shows_dt_only_when_it_varies(tmp_path):
    """The text line and the record are one formatting call, so they agree."""
    adaptive = _run(tmp_path / "a", adaptive=True)
    fixed = _run(tmp_path / "f", adaptive=False)
    assert all("dt" in r["text"] for r in adaptive)
    assert all("dt" not in r["text"] for r in fixed)


def test_dt_is_the_step_taken_not_the_step_coming(tmp_path):
    """model_seconds advances BY the dt reported on the same record."""
    steps = _run(tmp_path, adaptive=True, dts=(30.0, 45.0, 60.0))
    previous = 0.0
    for r in steps:
        assert r["model_seconds"] - previous == r["dt"], r
        previous = r["model_seconds"]


# ------------------------------------------------------------- plumbing

def test_v4_is_replayable_and_v3_did_not_move():
    """A published stream outlives the wheel that wrote it."""
    assert STEP_LOG_SCHEMA_ADAPTIVE in STEP_LOG_SCHEMAS
    for shipped in ("gpuwm.step-log/v1", "gpuwm.step-log/v2",
                    "gpuwm.step-log/v3"):
        assert shipped in STEP_LOG_SCHEMAS
    # The DEFAULT must not have moved: that is what keeps every fixed run
    # on the schema its consumers already handle.
    assert STEP_LOG_SCHEMA == "gpuwm.step-log/v3"


def test_format_step_line_appends_dt_only_when_given():
    without = format_step_line(domain=1, step=7, valid_time=START,
                               wall_seconds=0.125)
    with_dt = format_step_line(domain=1, step=7, valid_time=START,
                               wall_seconds=0.125, dt=33.075)
    assert without.endswith("step 7")
    assert with_dt.startswith(without)
    assert with_dt[len(without):] == "  dt 33.08"
