"""A stopped or partially reported contract run cannot qualify a candidate.

These controls also run against the previous standalone runner by setting
ARWEN_PRECUT_GATE_SOURCE. Every remote operation and archive command is replaced
with a recorded response; no connection or project suite is started by a control.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("ARWEN_PRECUT_GATE_SOURCE", ROOT / "tools/release/precut_gate.py"))
HEAD = "0123456789abcdef0123456789abcdef01234567"
REMOTE = "/tmp/precut-control"
DOCTOR = "tests/test_doctor.py"
NATIVE = "tests/test_native_wrf_distribution.py"
NAME = "test_the_provenance_check_reports_the_path_a_run_will_take"


def report(*, failure=None, error=False, missing=False, count_delta=0, skipped=False):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", name="pytest", tests=str(2 + count_delta),
                          errors=str(int(error)), failures=str(int(failure is not None)),
                          skipped=str(2 if skipped else 0))
    row = ET.SubElement(suite, "testcase", file=DOCTOR, classname="tests.test_doctor", name=NAME)
    if failure is not None:
        ET.SubElement(row, "failure", message=failure).text = failure
    if error:
        ET.SubElement(row, "error", message="collection failed")
    if skipped:
        ET.SubElement(row, "skipped", message="missing capability")
    row2 = ET.SubElement(suite, "testcase", file=NATIVE if not missing else DOCTOR,
                         classname="tests.test_native_wrf_distribution", name="test_distribution")
    if skipped:
        ET.SubElement(row2, "skipped", message="missing capability")
    return ET.tostring(root, encoding="unicode")


def drive(tmp_path, monkeypatch, *, child_exit=0, output="2 passed in 0.01s\n", xml=None,
          head_exit=0, status_exit=0, archive_exit=0, ship_exit=0, read_exit=0,
          dirty="", remote=REMOTE):
    spec = importlib.util.spec_from_file_location("precut_control", SOURCE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.__file__ = str(tmp_path / "runner.py")
    tree = tmp_path / "tree"
    (tree / ".git").mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        rc, stdout, stderr = 0, "", ""
        if isinstance(command, str):
            # The previous pipeline reports only the extraction status, even
            # when its upstream archive failed. The corrected runner splits it.
            rc = ship_exit
        elif command[0] == "git":
            if "rev-parse" in command:
                rc, stdout = head_exit, HEAD + "\n"
            elif "status" in command:
                rc, stdout = status_exit, dirty
            elif "archive" in command:
                rc = archive_exit
                if not rc:
                    target = next(v.split("=", 1)[1] for v in command if v.startswith("--output="))
                    Path(target).write_bytes(b"archive stand-in")
            else:
                raise AssertionError(command)
        elif command[0] == "ssh":
            if command[-1].startswith("cat -- "):
                rc, stdout = read_exit, report() if xml is None else xml
            elif "-m pytest" in command[-1]:
                rc, stdout = child_exit, remote + "/gpuwm/__init__.py\n" + output
            else:
                rc = ship_exit
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, rc, stdout, stderr)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    monkeypatch.setattr(gate, "contract_files", lambda _: [DOCTOR, NATIVE])
    args = ["--tree", str(tree), "--node", "build-account", "--python", "/never/run/python",
            "--remote-dir", remote]
    kwargs = {"defaults": {"evidence_dir": tmp_path / "evidence"}} if hasattr(gate, "completed_result") else {}
    rc = gate.main(args, **kwargs)
    path = tmp_path / "evidence" / ("precut-gate-" + HEAD[:9] + ".json")
    receipt = json.loads(path.read_text()) if path.exists() else {}
    return rc, receipt, calls


@pytest.mark.parametrize("child_exit,output", [
    (2, "KeyboardInterrupt\nno tests ran\n"),
    (3, "INTERNALERROR> worker crashed\n"),
    (137, "Killed\n"),
    (255, "Connection lost\n"),
    (2, "5 passed in 1.0s\nKeyboardInterrupt\n"),
])
def test_an_interrupted_process_never_passes_even_after_a_passing_summary(tmp_path, monkeypatch, child_exit, output):
    rc, _, _ = drive(tmp_path, monkeypatch, child_exit=child_exit, output=output)
    assert rc != 0, "a stopped contract process was recorded as a passing candidate"


@pytest.mark.parametrize("xml,child_exit", [
    ("", 0), ("<broken", 0), (report(count_delta=1), 0),
    (report(missing=True), 0), (report(error=True), 1),
    (report(), 1), (report(failure="ordinary assertion"), 0), (report(skipped=True), 0),
])
def test_missing_incomplete_or_conflicting_results_cannot_pass(tmp_path, monkeypatch, xml, child_exit):
    rc, _, _ = drive(tmp_path, monkeypatch, child_exit=child_exit, xml=xml)
    assert rc != 0, "an incomplete contract report was accepted"


@pytest.mark.parametrize("fault", ["head_exit", "status_exit", "archive_exit", "ship_exit", "read_exit"])
def test_every_upstream_operation_must_succeed(tmp_path, monkeypatch, fault):
    rc, _, calls = drive(tmp_path, monkeypatch, **{fault: 128})
    assert rc != 0, f"{fault} was hidden by a later successful operation"
    if fault in {"head_exit", "status_exit", "archive_exit", "ship_exit"}:
        assert not any(isinstance(c, list) and c[0] == "ssh" and "-m pytest" in c[-1] for c in calls)


def test_a_completed_run_passes_and_names_its_limited_scope(tmp_path, monkeypatch):
    rc, receipt, _ = drive(tmp_path, monkeypatch)
    assert rc == 0
    assert receipt["scope"] == "candidate release contract tests before export"
    assert receipt["full_release_qualification"] is False
    assert receipt["counts"]["passed"] == 2


def test_only_the_documented_shadow_assertion_is_allowed(tmp_path, monkeypatch):
    failure = "assert 'missing' == 'verified'\n  - verified\n  + missing"
    output = f"FAILED {DOCTOR}::{NAME} - {failure}\n1 failed, 1 passed in 0.01s\n"
    rc, receipt, _ = drive(tmp_path, monkeypatch, child_exit=1, output=output, xml=report(failure=failure))
    assert rc == 0
    assert receipt["environmental"] == [DOCTOR + "::" + NAME]


def test_a_different_failure_in_the_allowlisted_test_is_real(tmp_path, monkeypatch):
    output = f"FAILED {DOCTOR}::{NAME} - assert False\n1 failed, 1 passed in 0.01s\n"
    rc, _, _ = drive(tmp_path, monkeypatch, child_exit=1, output=output, xml=report(failure="assert False"))
    assert rc == 1, "the test name hid an unrelated contract failure"


def test_an_ordinary_contract_failure_refuses_the_candidate(tmp_path, monkeypatch):
    xml = report(failure="assert False").replace(NAME, "test_other_contract")
    output = f"FAILED {DOCTOR}::test_other_contract - assert False\n1 failed, 1 passed in 0.01s\n"
    rc, _, _ = drive(tmp_path, monkeypatch, child_exit=1, output=output, xml=xml)
    assert rc == 1


def test_a_dirty_tree_cannot_be_archived_as_the_claimed_candidate(tmp_path, monkeypatch):
    rc, _, _ = drive(tmp_path, monkeypatch, dirty=" M changed.py")
    assert rc != 0


@pytest.mark.parametrize("remote", ["/", "/tmp", "/tmp/../elsewhere", "/tmp/run\nnext"])
def test_unsafe_scratch_destinations_are_rejected_before_transport(tmp_path, monkeypatch, remote):
    rc, _, calls = drive(tmp_path, monkeypatch, remote=remote)
    assert rc != 0
    assert not any(isinstance(c, str) or c[0] == "ssh" for c in calls)


def test_each_attempt_uses_new_scratch_and_never_removes_prior_data(tmp_path, monkeypatch):
    _, first, calls = drive(tmp_path / "one", monkeypatch)
    _, second, _ = drive(tmp_path / "two", monkeypatch)
    assert first.get("remote_directory") != second.get("remote_directory")
    assert all("rm -rf" not in str(command) for command in calls)


def test_failed_evidence_storage_does_not_print_a_pass(tmp_path, monkeypatch, capsys):
    (tmp_path / "evidence").write_text("not a directory")
    rc, _, _ = drive(tmp_path, monkeypatch)
    assert rc != 0
    assert "precut gate: PASS" not in capsys.readouterr().out


@pytest.mark.parametrize("statement,expected", [
    ("assert True", 0), ("assert False", 1),
    ("raise KeyboardInterrupt", 2), ("raise RuntimeError('ordinary failure')", 1),
])
def test_real_pytest_reports_agree_with_the_process_outcome(tmp_path, statement, expected):
    spec = importlib.util.spec_from_file_location("precut_real_control", SOURCE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    (tmp_path / "test_control.py").write_text(
        "def test_first():\n    assert True\n\ndef test_second():\n    " + statement + "\n")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "test_control.py", "-q", "-o", "addopts=",
        "-o", "junit_family=xunit1", "--rootdir=.", "--confcutdir=.",
        "--junitxml=completed.xml", "-p", "no:cacheprovider", "-n", "2",
    ], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTEST_ADDOPTS": "", "GPUWM_NO_LOCAL_GPU": "1"})
    assert result.returncode == expected, result.stdout + result.stderr
    xml = (tmp_path / "completed.xml").read_text()
    if expected == 2:
        with pytest.raises(gate.GateError):
            gate.completed_result(xml, result.returncode, ["test_control.py"])
    else:
        checked = gate.completed_result(xml, result.returncode, ["test_control.py"])
        assert bool(checked["failures"]) == (expected == 1)


def test_the_real_shadow_assertion_is_the_narrow_environmental_exception(tmp_path):
    spec = importlib.util.spec_from_file_location("precut_shadow_control", SOURCE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    (tmp_path / "tests").mkdir()
    (tmp_path / DOCTOR).write_text(
        "def test_first():\n    assert True\n\ndef " + NAME +
        "():\n    status = 'missing'\n    assert status == 'verified'\n")
    result = subprocess.run([
        sys.executable, "-m", "pytest", DOCTOR, "-q", "-o", "addopts=",
        "-o", "junit_family=xunit1", "--rootdir=.", "--confcutdir=.",
        "--junitxml=completed.xml", "-p", "no:cacheprovider", "-n", "2",
    ], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTEST_ADDOPTS": "", "GPUWM_NO_LOCAL_GPU": "1"})
    assert result.returncode == 1, result.stdout + result.stderr
    xml = ET.fromstring((tmp_path / "completed.xml").read_text())
    # The gate consumes a remote POSIX report. A local Windows subprocess
    # fixture serializes native separators, so give this fixture that format.
    for row in xml.iter("testcase"):
        row.set("file", Path(row.attrib["file"]).as_posix())
    checked = gate.completed_result(ET.tostring(xml, encoding="unicode"), result.returncode, [DOCTOR])
    assert checked["failures"] == []
    assert checked["environmental"] == [DOCTOR + "::" + NAME]



def test_remote_pytest_records_subtest_identities(tmp_path, monkeypatch):
    _, _, commands = drive(tmp_path, monkeypatch)
    pytest_command = next(c[-1] for c in commands
                          if c[0] == "ssh" and "-m pytest" in c[-1])
    assert "-p tools.release.precut_gate" in pytest_command


@pytest.mark.parametrize("outcome", ["pass", "fail", "multiple-fail", "skip"])
def test_real_unittest_subtests_preserve_outcomes_and_complete_xml(tmp_path, outcome):
    spec = importlib.util.spec_from_file_location("precut_subtest_control", SOURCE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    # Load exactly the gate's reporting hook, without the repository conftest.
    (tmp_path / "precut_reporting.py").write_bytes(SOURCE.read_bytes())
    body = {
        "pass": "self.assertTrue(True)",
        "fail": "self.assertNotEqual(item, 1)",
        "multiple-fail": "self.assertEqual(item, 1)",
        "skip": "self.skipTest('one skipped subtest') if item == 1 else None",
    }[outcome]
    (tmp_path / "test_subtests.py").write_text(
        "import unittest\nfrom pathlib import Path\n"
        "class Cases(unittest.TestCase):\n"
        "    def test_subtests(self):\n"
        "        for item in range(3):\n"
        "            with self.subTest(item=item):\n"
        "                with Path('visited').open('a') as stream: stream.write(str(item))\n"
        "                " + body + "\n"
        "def test_other():\n    assert True\n")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "test_subtests.py", "-q", "-o", "addopts=",
        "-o", "junit_family=xunit1", "--rootdir=.", "--confcutdir=.",
        "--junitxml=completed.xml", "-p", "no:cacheprovider", "-p", "precut_reporting", "-n", "2",
    ], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "PYTEST_ADDOPTS": "", "GPUWM_NO_LOCAL_GPU": "1"})
    expected = int(outcome in ("fail", "multiple-fail"))
    assert result.returncode == expected, result.stdout + result.stderr
    xml = (tmp_path / "completed.xml").read_text()
    checked = gate.completed_result(xml, result.returncode, ["test_subtests.py"])
    assert bool(checked["failures"]) == bool(expected)
    assert not checked["environmental"]
    if outcome == "pass":
        assert (tmp_path / "visited").read_text() == "012"
    if pytest.version_tuple >= (9,):
        assert checked["subtest_cases"] == 3
        assert checked["counts"]["tests"] == 5
        assert checked["counts"]["failures"] == (2 if outcome == "multiple-fail" else expected)
        assert (tmp_path / "visited").read_text() == "012"
    # Removing an actual recorded case still fails, including a passing subtest.
    truncated = ET.fromstring(xml)
    suite = truncated.find("testsuite")
    suite.remove(suite.findall("testcase")[-1])
    with pytest.raises(gate.GateError, match="incomplete"):
        gate.completed_result(ET.tostring(truncated, encoding="unicode"), result.returncode, ["test_subtests.py"])


@pytest.mark.skipif(pytest.version_tuple < (9,), reason="builtin subtests fixture requires pytest9")
def test_real_expected_subtest_failure_is_recorded_as_skipped(tmp_path):
    spec = importlib.util.spec_from_file_location("precut_xfail_control", SOURCE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    (tmp_path / "precut_reporting.py").write_bytes(SOURCE.read_bytes())
    (tmp_path / "test_subtests.py").write_text(
        "import pytest\ndef test_subtests(subtests):\n"
        "    for item in range(3):\n        with subtests.test(item=item):\n"
        "            if item == 1: pytest.xfail('expected subtest')\n"
        "            assert True\n")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "test_subtests.py", "-q", "-o", "addopts=",
        "-o", "junit_family=xunit1", "--rootdir=.", "--confcutdir=.",
        "--junitxml=completed.xml", "-p", "no:cacheprovider", "-p", "precut_reporting", "-n", "2",
    ], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "PYTEST_ADDOPTS": "", "GPUWM_NO_LOCAL_GPU": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    checked = gate.completed_result((tmp_path / "completed.xml").read_text(), 0, ["test_subtests.py"])
    assert checked["subtest_cases"] == 3
    assert checked["counts"] == dict(tests=4, failures=0, errors=0, skipped=1, passed=3)
