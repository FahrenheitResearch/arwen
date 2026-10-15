"""Exercise the source-leg controller with real pytest children and CPU toys."""
import json
from pathlib import Path
import subprocess

import pytest

from tools.battery import run_source_leg as runner


def prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k selected")
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -k selected\n")
    for index, name in enumerate(runner.FILES):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        function = "selected" if index == 0 else "required"
        target.write_text(f"def test_{function}():\n    pass\n")
    source = tmp_path / "wrf/phys/module_radiation_driver.F"
    source.parent.mkdir(parents=True)
    source.write_text("! controller fixture only\n")
    for name in ("wrf480", "ysu", "coupled"):
        (tmp_path / name).mkdir()
    return ["--wrf461", str(tmp_path / "wrf"),
            "--wrf480-source", str(tmp_path / "wrf480"),
            "--ysu-fixture", str(tmp_path / "ysu"),
            "--coupled-fixture", str(tmp_path / "coupled"),
            "--out", str(tmp_path / "out")]


@pytest.mark.parametrize("case", ["all", "failure", "empty_file", "skip"])
def test_every_file_runs_despite_inherited_selection(tmp_path, monkeypatch, case):
    args = prepare(tmp_path, monkeypatch)
    target = tmp_path / runner.FILES[-1]
    if case == "failure":
        target.write_text('def test_required():\n    assert False, "required toy gate"\n')
    elif case == "empty_file":
        target.write_text("# no tests\n")
    elif case == "skip":
        target.write_text('import pytest\ndef test_required():\n    pytest.skip("toy skip")\n')
    assert runner.main(args) == (0 if case == "all" else 1)
    receipt = json.loads((tmp_path / "out/source-leg.json").read_text())
    assert [row["exact"] for row in receipt["runs"]] == ["0", "1"]
    for row in receipt["runs"]:
        assert row["tests"] == (4 if case == "empty_file" else 5)
        assert row["failed"] == (1 if case == "failure" else 0)
        assert row["skipped"] == (1 if case == "skip" else 0)
        assert row["missing_files"] == ([runner.FILES[-1]] if case == "empty_file" else [])


@pytest.mark.parametrize("replacement", [None, "<broken", '<testsuites><testcase file="tests/test_cldfra2_gpu.py"><failure/></testcase></testsuites>'])
def test_missing_or_invalid_report_cannot_borrow_old_pass(tmp_path, monkeypatch, replacement):
    args = prepare(tmp_path, monkeypatch)
    output = tmp_path / "out"
    output.mkdir()
    old = "<testsuites>" + "".join(f'<testcase file="{name}"/>' for name in runner.FILES) + "</testsuites>"
    for exact in ("0", "1"):
        (output / f"source-exact-{exact}.xml").write_text(old)

    def interrupted(command, **kwargs):
        report = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        assert not report.exists()
        if replacement is not None:
            report.write_text(replacement)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", interrupted)
    assert runner.main(args) == 1
    receipt = json.loads((output / "source-leg.json").read_text())
    assert all(not row["passed"] for row in receipt["runs"])
