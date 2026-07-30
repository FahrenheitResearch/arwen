from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.nssl2_wrf461_composed_replay import compare_replay


def _write_qv(path: Path, *, engine: str, value: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("engine", "k", "qv"))
        writer.writeheader()
        writer.writerow({"engine": engine, "k": 1, "qv": value})


def test_composed_replay_comparator_returns_pass(tmp_path: Path) -> None:
    wrf = tmp_path / "wrf.csv"
    gpu = tmp_path / "gpu.csv"
    report_path = tmp_path / "report.json"
    _write_qv(wrf, engine="wrf", value=0.01)
    _write_qv(gpu, engine="gpu", value=0.01)

    report = compare_replay.compare(wrf, gpu, report_path)

    assert report["status"] == "PASS"
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "PASS"


def test_composed_replay_cli_exits_nonzero_on_failed_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrf = tmp_path / "wrf.csv"
    gpu = tmp_path / "gpu.csv"
    report_path = tmp_path / "report.json"
    _write_qv(wrf, engine="wrf", value=0.01)
    _write_qv(gpu, engine="gpu", value=0.02)
    monkeypatch.setattr(
        "sys.argv", ["compare_replay.py", str(wrf), str(gpu), str(report_path)]
    )

    with pytest.raises(SystemExit) as raised:
        compare_replay.main()

    assert raised.value.code == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "FAIL"


def test_composed_replay_rejects_hidden_ice_number_process_error(
    tmp_path: Path,
) -> None:
    wrf = tmp_path / "wrf.csv"
    gpu = tmp_path / "gpu.csv"
    report_path = tmp_path / "report.json"
    with wrf.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("engine", "k", "qni"))
        writer.writeheader()
        writer.writerow({"engine": "wrf", "k": 1, "qni": 674.65283203125})
    with gpu.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("engine", "k", "qni"))
        writer.writeheader()
        writer.writerow({"engine": "gpu", "k": 1, "qni": 674.9795532226562})

    report = compare_replay.compare(wrf, gpu, report_path)

    assert report["status"] == "FAIL"
    assert report["results"]["qni"]["violations"] == 1
