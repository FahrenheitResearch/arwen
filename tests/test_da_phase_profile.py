"""The DA phase profiler measures without changing what it wraps."""
from __future__ import annotations

import json
import sys
import time
import types

import numpy as np

from tools import da_phase_profile as dpp


def test_wrap_counts_inclusive_and_exclusive_and_restores(monkeypatch):
    module = types.ModuleType("fake_da_phase_target")

    def inner(x):
        time.sleep(0.02)
        return x + 1

    def outer(x):
        time.sleep(0.01)
        return module.inner(x) * 2

    module.inner = inner
    module.outer = outer
    monkeypatch.setitem(sys.modules, "fake_da_phase_target", module)
    profiler = dpp.PhaseProfiler()
    profiler.install(["fake_da_phase_target:outer", "fake_da_phase_target:inner",
                      "fake_da_phase_target:absent"])
    assert module.outer(3) == 8
    rows = profiler.rows
    assert rows["fake_da_phase_target:outer"]["calls"] == 1
    assert rows["fake_da_phase_target:inner"]["calls"] == 1
    outer_row = rows["fake_da_phase_target:outer"]
    inner_row = rows["fake_da_phase_target:inner"]
    assert outer_row["inclusive_s"] >= inner_row["inclusive_s"]
    assert abs(outer_row["exclusive_s"]
               - (outer_row["inclusive_s"] - inner_row["inclusive_s"])) < 1e-6
    assert profiler.missing == ["fake_da_phase_target:absent"]
    profiler.uninstall()
    assert module.outer is outer and module.inner is inner


def test_bench_under_profiler_matches_unprofiled_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    from tools import da_analysis_bench as bench

    plain = tmp_path / "plain.json"
    assert bench.main(["--members", "4", "--shape", "4", "12", "12",
                       "--radars", "1", "--fields", "3", "--device", "host",
                       "--out", str(plain)]) == 0
    profiled = tmp_path / "profiled.json"
    phase = tmp_path / "phase.json"
    code = dpp.run(["--out", str(phase), "--", "-m", "tools.da_analysis_bench",
                    "--members", "4", "--shape", "4", "12", "12", "--radars",
                    "1", "--fields", "3", "--device", "host", "--out",
                    str(profiled)])
    assert code == 0
    a = json.loads(plain.read_text())["diagnostics"]
    b = json.loads(profiled.read_text())["diagnostics"]
    for key in ("active_points", "max_local_obs", "mean_increment_rms"):
        if key in a:
            assert a[key] == b[key]
    report = json.loads(phase.read_text())
    assert report["schema"] == "gpuwm-da.phase-profile.v1"
    assert report["phases"]["gpuwm.da.radar_assimilation:_analysis_attempt"]["calls"] == 1
    assert np.isfinite(report["wall_seconds"])


def test_pyspy_summary_charges_the_deepest_project_frame():
    from tools.da_pyspy_summary import summarise

    raw = [
        "<module> (tools/da_x.py:1);analyze (gpuwm/da/letkf.py:10);"
        "eigh (numpy/linalg/_linalg.py:5) 30",
        "<module> (tools/da_x.py:1);timed (tools/da_phase_profile.py:108);"
        "analyze (gpuwm/da/letkf.py:20) 10",
        "<frozen runpy> (x:1) 5",
    ]
    out = summarise(raw, rate=10.0)
    assert out["total_seconds"] == 4.5
    lines = {(r["file"], r["line"]): r["seconds"] for r in out["lines"]}
    assert lines[("gpuwm/da/letkf.py", 10)] == 3.0
    assert lines[("gpuwm/da/letkf.py", 20)] == 1.0
    inclusive = {r["function"]: r["seconds"] for r in out["functions_inclusive"]}
    assert inclusive["analyze"] == 4.0 and inclusive["<module>"] == 4.0
