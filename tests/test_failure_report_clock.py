"""A failed forecast's report states the model time of the step that failed.

Breakage this prevents: ``progress.json`` is published on the first and every
60th step only, and the failure report read its clock from there.  A 500 m
run whose health check failed at model second 40 (step 16 of 2.5 s) wrote
``model_elapsed_seconds: 2.5`` in both report.json and the final
progress.json, while its step log said 40.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_cupy
from gpuwm import prepared_single_domain_forecast as runner
from gpuwm.core.clock import DomainClock, DomainTicks

from test_prepared_single_domain_forecast import (
    _bind_synthetic_preflight_geometry, _prepared_fixture, _sha256)


def _clock_after(steps: int) -> DomainClock:
    """The root clock of a 2.5 s run after ``steps`` completed steps."""
    spec = DomainTicks(
        grid_id=1, parent_id=0, parent_time_step_ratio=1, step_ticks=5,
        dt_fp32=np.float32(2.5), history_ticks=7200, restart_ticks=None,
        radt_ticks=None, stepra=None, cudt_ticks=None, stepcu=None,
        bldt_ticks=None, stepbl=None)
    clock = DomainClock(spec, tick_den=2, run_ticks=7200)
    for _ in range(steps):
        clock.advance()
    return clock


def test_the_failure_carries_the_clock_of_the_step_that_failed():
    error = FloatingPointError("w exceeds upper bound")
    runner._mark_failure_clock(error, _clock_after(16))
    assert runner._failure_clock(error, 2.5) == 40.0
    # A failure before integration has no clock: the heartbeat's stands.
    assert runner._failure_clock(RuntimeError("preflight"), 2.5) == 2.5


def test_reading_the_clock_can_never_replace_the_failure():
    class Broken:
        @property
        def elapsed_seconds(self):
            raise RuntimeError("clock gone")

    error = FloatingPointError("the failure")
    runner._mark_failure_clock(error, Broken())
    assert runner._failure_clock(error, 7.5) == 7.5


# The door refuses without cupy before it reaches the runner; no device opens.
@requires_cupy
def test_the_report_and_final_progress_state_the_failed_step_s_clock(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "gfs", physics_profile=None)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    out = tmp_path / "out"

    def fail(inputs, *, output_directory, **_kwargs):
        # The throttled heartbeat, as the run left it after step 1.
        Path(output_directory, "progress.json").write_text(
            json.dumps({"model_elapsed_seconds": 2.5}), encoding="utf-8")
        error = FloatingPointError("completed-step health failure")
        runner._mark_failure_clock(error, _clock_after(16))
        raise error

    monkeypatch.setattr(runner, "run_prepared_forecast", fail)
    with pytest.raises(FloatingPointError, match="completed-step health failure"):
        runner.main([
            "--source", "gfs", "--prepared-root", str(fixture.prepared),
            "--proof-sha256", _sha256(fixture.proof),
            "--source-manifest-sha256", _sha256(fixture.source_manifest),
            "--prepared-content-sha256", fixture.content_sha256,
            "--experiment-config", str(fixture.experiment),
            "--wps-namelist", str(fixture.wps), "--io-mode", "history",
            "--outdir", str(out)])
    for name in ("report.json", "progress.json"):
        assert json.loads((out / name).read_text(encoding="utf-8"))[
            "model_elapsed_seconds"] == 40.0, name
