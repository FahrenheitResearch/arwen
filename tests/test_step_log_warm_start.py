"""A warm start's run line and step totals count only that run's steps.

The defect (NV recipe dry run, D7): the 50-step restart leg of NCAR's
CONUS 12 km benchmark, warm-started from wrfrst at 23Z (step 550 of a run
from 12Z), printed::

    gpuwm: STARTING SIMULATION at 2019-11-26_12:00:00 for 43200 model seconds
    d01 2019-11-27_00:00:00 gpuwm: domain end, 600 steps
    gpuwm: SUCCESS COMPLETE SIMULATION, 600 steps, 45.2 wall seconds

so steps per second read from the log were 12 times too high.  The step
records keep WRF's own itimestep (551..600), as WRF's rsl does.

CPU only; no card.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from io import StringIO
from types import SimpleNamespace

import pytest

from gpuwm.progress_log import (STEP_LOG_FILENAME, StepLog, StepLogResume,
                                open_step_log)

START = datetime(2019, 11, 26, 12, 0, 0)
DT = 72.0
RESTART_STEP = 550


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _restart_leg(tmp_path, *, every=1):
    text = StringIO()
    log = StepLog(start_time=START, run_seconds=43200.0, text_stream=text,
                  jsonl_path=tmp_path / STEP_LOG_FILENAME, every=every,
                  resume={"model_seconds": RESTART_STEP * DT,
                          "steps": {1: RESTART_STEP}})
    for step in range(RESTART_STEP + 1, 601):
        log.domain_step(grid_id=1, step_count=step, model_seconds=step * DT,
                        step_wall_seconds=0.05)
    log.close()
    return text.getvalue().splitlines(), _records(tmp_path / STEP_LOG_FILENAME)


def test_warm_start_run_line_names_the_restart_instant_and_window(tmp_path):
    lines, records = _restart_leg(tmp_path)
    assert lines[0] == ("gpuwm: STARTING SIMULATION at 2019-11-26_23:00:00 for "
                        "3600 model seconds, a restart at step 550 of the run "
                        "from 2019-11-26_12:00:00")
    start = records[0]
    assert start["event"] == "run_start"
    assert start["restart"] == {"valid_time": "2019-11-26_23:00:00",
                                "model_seconds": 39600.0,
                                "window_seconds": 3600.0,
                                "steps": {"1": 550}}
    # Unchanged meaning: model_seconds on every record is measured from it.
    assert start["start_time"] == "2019-11-26_12:00:00"
    assert start["run_seconds"] == 43200.0


def test_warm_start_totals_count_only_this_runs_steps(tmp_path):
    lines, records = _restart_leg(tmp_path)
    assert "d01 2019-11-27_00:00:00 gpuwm: domain end, 50 steps" in lines
    assert lines[-1].startswith("gpuwm: SUCCESS COMPLETE SIMULATION, 50 steps, ")
    end = next(r for r in records if r["event"] == "domain_end")
    assert end["steps"] == 50 and end["last_step"] == 600
    assert records[-1]["event"] == "run_end" and records[-1]["steps"] == 50


def test_warm_start_steps_keep_wrf_itimestep_and_fraction_spans_the_window(tmp_path):
    _, records = _restart_leg(tmp_path)
    steps = [r for r in records if r["event"] == "step"]
    assert [r["step"] for r in steps] == list(range(551, 601))
    assert steps[0]["fraction"] == pytest.approx(1 / 50)
    assert steps[-1]["fraction"] == pytest.approx(1.0)
    assert steps[0]["valid_time"] == "2019-11-26_23:01:12"


def test_warm_start_thinned_log_still_reports_this_runs_first_step(tmp_path):
    _, records = _restart_leg(tmp_path, every=7)
    steps = [r["step"] for r in records if r["event"] == "step"]
    assert steps[0] == 551
    assert steps[-1] == 600


def test_cold_run_is_unchanged(tmp_path):
    text = StringIO()
    log = StepLog(start_time=START, run_seconds=360.0, text_stream=text,
                  jsonl_path=tmp_path / STEP_LOG_FILENAME)
    for step in range(1, 6):
        log.domain_step(grid_id=1, step_count=step, model_seconds=step * DT,
                        step_wall_seconds=0.05)
    log.close()
    lines = text.getvalue().splitlines()
    records = _records(tmp_path / STEP_LOG_FILENAME)
    assert lines[0] == ("gpuwm: STARTING SIMULATION at 2019-11-26_12:00:00 "
                        "for 360 model seconds")
    assert "restart" not in records[0]
    end = next(r for r in records if r["event"] == "domain_end")
    assert end["steps"] == 5 and "last_step" not in end
    assert records[-1]["steps"] == 5
    assert [r["fraction"] for r in records if r["event"] == "step"][-1] == 1.0


def test_open_step_log_passes_the_resume_through(tmp_path):
    log = open_step_log(outdir=tmp_path, start_time=START, run_seconds=43200.0,
                        text_stream=StringIO(),
                        resume=StepLogResume.of({"model_seconds": 39600.0,
                                                 "steps": {"1": 550}}))
    log.close()
    first = _records(tmp_path / STEP_LOG_FILENAME)[0]
    assert first["restart"]["steps"] == {"1": 550}


@pytest.mark.parametrize("bad", [{"model_seconds": -1.0},
                                 {"model_seconds": float("nan")},
                                 {"model_seconds": 10.0, "steps": {1: -2}}])
def test_resume_refuses_an_impossible_restart_instant(bad):
    with pytest.raises(ValueError):
        StepLogResume.of(bad)


def test_tree_runner_reads_the_resume_from_the_warm_start_clocks():
    from gpuwm.prepared_domain_tree_forecast import _warm_start_resume

    clock = SimpleNamespace(valid_time=START + timedelta(hours=11),
                            itimestep=RESTART_STEP)
    nest = SimpleNamespace(valid_time=START + timedelta(hours=11),
                           itimestep=3 * RESTART_STEP)
    resume = _warm_start_resume(
        SimpleNamespace(warm_start_clocks={1: clock, 2: nest}), START)
    assert resume.model_seconds == 39600.0
    assert dict(resume.steps) == {1: 550, 2: 1650}
    assert _warm_start_resume(SimpleNamespace(), START) is None
    assert _warm_start_resume(None, START) is None
