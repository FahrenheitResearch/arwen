"""``gpuwm run --benchmark``: no render, no verification, a WRF-shaped summary.

The defect (NV recipe dry run, D6): ``gpuwm run`` always rendered and
verified after the forecast (14 to 33 s on NCAR's CONUS 12 km 1 h restart
leg, 70 s on the 12 h leg) and had no flag to leave either out, so a
wall-clock benchmark timed pictures.  CPU only; no card.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gpuwm import benchmark_timing


def _write_log(outdir: Path, t0: float, *, steps=5, first_wall=4.0, wall=0.05,
               restart=None):
    rows = []
    seq = 0

    def emit(event, at, **fields):
        nonlocal seq
        seq += 1
        rows.append({"schema": "x", "sequence": seq, "event": event,
                     "emitted_unix_ms": int(round(at * 1000)), "text": "", **fields})

    start = {"start_time": "2019-11-26_12:00:00", "run_seconds": 43200.0}
    if restart:
        start["restart"] = restart
    emit("run_start", t0 + 10.0, **start)
    at = t0 + 20.0 + first_wall
    base = 550 if restart else 0
    emit("step", at, domain=1, step=base + 1, valid_time="v1",
         step_wall_seconds=first_wall)
    for i in range(2, steps + 1):
        at += wall + 0.01
        emit("step", at, domain=1, step=base + i, valid_time=f"v{i}",
             step_wall_seconds=wall)
    emit("domain_end", at + 0.5, domain=1, steps=steps, valid_time="vend")
    emit("run_end", at + 1.0, status="SUCCESS", steps=steps)
    (outdir / "progress.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return at


def test_summary_cuts_the_wrf_phases_from_marks_and_step_log(tmp_path):
    t0 = 1_000_000.0
    marks = {"worker_start": t0 + 1.0, "read_start": t0 + 2.0, "read_end": t0 + 12.0}
    last = _write_log(tmp_path, t0, restart={"steps": {"1": 550}})
    summary = benchmark_timing.summarize(
        tmp_path, process_start=t0, process_end=last + 2.0, marks=marks)
    p = summary["phases_seconds"]
    assert p["startup"] == pytest.approx(2.0)
    assert p["read"] == pytest.approx(10.0)
    # The first step record lands at t0 + 24 after a 4 s step: init ends at t0 + 20.
    assert p["init"] == pytest.approx(8.0)
    assert p["first_step"] == pytest.approx(4.0)
    assert p["steps"] == pytest.approx(4 * 0.06, abs=2e-3)
    assert p["write"] == pytest.approx(1.0, abs=2e-3)
    assert p["total"] == pytest.approx(last + 2.0 - t0)
    assert p["finish"] == pytest.approx(1.0, abs=2e-3)
    rows = ("startup", "read", "init", "first_step", "steps", "write", "finish")
    assert sum(p[row] for row in rows) == pytest.approx(p["total"], abs=5e-3)
    assert summary["steps"] == 5 and summary["first_step"] == 551
    assert summary["median_step_seconds"] == pytest.approx(0.05)
    assert summary["excluded"] == ["render", "observation verification"]
    text = benchmark_timing.format_summary(summary)
    for word in ("read", "init", "first step", "steps", "write", "finish", "total"):
        assert f"  {word} " in text
    assert "551 to 555" in text


def test_report_prints_and_records(tmp_path, capsys):
    t0 = time.time() - 60.0
    benchmark_timing.Marks  # noqa: B018 - public
    (tmp_path / benchmark_timing.MARKS_NAME).write_text(json.dumps(
        {"worker_start": t0, "read_start": t0 + 1, "read_end": t0 + 2}))
    _write_log(tmp_path, t0)
    summary = benchmark_timing.report(tmp_path, process_start=t0)
    out = capsys.readouterr().out
    assert out.startswith("gpuwm benchmark: 5 steps")
    saved = json.loads((tmp_path / benchmark_timing.SUMMARY_NAME).read_text())
    assert saved == summary and saved["schema"] == benchmark_timing.SCHEMA


def test_summary_refuses_a_run_that_did_not_finish(tmp_path):
    t0 = 1_000_000.0
    _write_log(tmp_path, t0)
    log = tmp_path / "progress.jsonl"
    log.write_text("".join(line + "\n" for line in log.read_text().splitlines()
                           if '"run_end"' not in line))
    with pytest.raises(ValueError, match="no run_end"):
        benchmark_timing.summarize(tmp_path, process_start=t0,
                                   marks={"worker_start": t0, "read_end": t0})


def test_summary_refuses_a_missing_step_log(tmp_path):
    with pytest.raises(ValueError, match="no step log"):
        benchmark_timing.summarize(tmp_path, process_start=0.0,
                                   marks={"worker_start": 0.0, "read_end": 0.0})


def test_process_start_is_before_now():
    start = benchmark_timing.process_start_unix()
    assert start <= time.time()
    assert time.time() - start < 24 * 3600


def test_marks_record_read_window(tmp_path):
    marks = benchmark_timing.Marks()
    marks.mark("read_start")
    marks.mark("read_end")
    saved = json.loads(marks.write(tmp_path).read_text())
    assert saved["worker_start"] <= saved["read_start"] <= saved["read_end"]


def test_gpuwm_run_has_the_flag_and_refuses_it_off_the_wrf_door(tmp_path, capsys):
    from gpuwm import cli

    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is None:
        pytest.skip("cli exposes no parser builder")
    run = next(a for a in parser._subparsers._group_actions[0].choices.items()
               if a[0] == "run")[1]
    assert any("--benchmark" in action.option_strings for action in run._actions)
    config = tmp_path / "case.toml"
    config.write_text("[experiment]\n")
    with pytest.raises(SystemExit):
        cli.main(["run", str(config), "--benchmark"])
    assert "--benchmark times the WRF-input door" in capsys.readouterr().err


def test_wrf_door_worker_takes_the_flag():
    from gpuwm.wrfinput_forecast import build_parser

    args = build_parser().parse_args(["--wrfinput", "x", "--outdir", "y", "--benchmark"])
    assert args.benchmark is True
    assert build_parser().parse_args(["--wrfinput", "x", "--outdir", "y"]).benchmark is False


def test_benchmark_refuses_a_product_list(tmp_path):
    from gpuwm.wrfinput_forecast import run_wrf_forecast

    with pytest.raises(ValueError, match="--benchmark draws no products"):
        run_wrf_forecast(tmp_path, tmp_path / "out", benchmark=True,
                         render_products="t2m", exclusive_gpu=False)


def test_wrf_door_forwards_the_flag_to_its_worker_and_skips_the_render():
    source = Path(__file__).resolve().parents[1] / "gpuwm" / "wrfinput_forecast.py"
    text = source.read_text(encoding="utf-8")
    assert "command += ['--benchmark']" in text
    assert "render_products = 'none'" in text
