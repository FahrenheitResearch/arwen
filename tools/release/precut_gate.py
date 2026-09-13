"""Check the candidate release contract set before export.

This check covers the contract files named by the publication workflow, including
the native distribution check. It does not qualify built distributions, desktop
packages, or an installed end-to-end forecast. A stopped test process must never
produce a passing receipt just because it printed no failing test names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

SCOPE = "candidate release contract tests before export"
ENVIRONMENTAL = {
    "tests/test_doctor.py::test_the_provenance_check_reports_the_path_a_run_will_take":
        "A shadowed interpreter uses another tree's distribution metadata. The "
        "missing-to-verified assertion must pass in the fresh installed release check.",
}


class GateError(Exception):
    """A failed operation cannot establish a complete contract result."""


def pytest_configure(config):
    """Give genuine subtest reports distinct JUnit cases on the controller.

    Pytest 9 counts successful unittest subtests in the suite total but merges
    their XML into the parent case. Disabling the subtests plugin does not stop
    that unittest integration. Keep every report and outcome, only giving each
    subtest a distinct identity so the strict completion check can verify it.
    """
    if hasattr(config, "workerinput"):
        return  # xdist forwards reports; only its JUnit-writing controller names them.
    try:
        from _pytest.subtests import SubtestReport
    except ImportError:
        try:
            from pytest_subtests.plugin import SubTestReport as SubtestReport
        except ImportError:
            return  # Older pytest without subtest reports already has one row per case.
    import pytest
    counters = {}

    class SubtestJUnitIdentity:
        @pytest.hookimpl(tryfirst=True)
        def pytest_runtest_logreport(self, report):
            if isinstance(report, SubtestReport):
                original = report.nodeid
                index = counters.get(original, 0) + 1
                counters[original] = index
                report.nodeid = original + f"[arwen-subtest-{index}]"

    config.pluginmanager.register(SubtestJUnitIdentity(), "arwen-precut-subtest-identities")


def run(command, **kwargs):
    return subprocess.run(command, text=True, capture_output=True, **kwargs)


def checked(command, operation, **kwargs):
    result = run(command, **kwargs)
    if result.returncode != 0:
        raise GateError(f"{operation} failed with exit {result.returncode}. "
                        f"Correct that failure and run the check again.\n{result.stderr[-2000:]}")
    return result


def contract_files(tree: Path) -> list[str]:
    text = (tree / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    files = sorted(set(re.findall(r"tests/test_[a-z0-9_]+\.py", text)))
    if not files:
        raise GateError("The publication workflow names no contract files. Restore its test selection and run again.")
    native = "tests/test_native_wrf_distribution.py"
    return sorted(set(files + [native]))


def remote_directory(value: str) -> str:
    # Preserve the existing workaround for automatic argument conversion.
    marker = "/Program Files/Git"
    if marker in value:
        value = value[value.index(marker) + len(marker):]
    if not value.startswith("/") or ".." in value.split("/") or "\x00" in value:
        raise GateError("The remote scratch directory must be an absolute path without '..'. Choose a dedicated directory and run again.")
    path = PurePosixPath(value)
    if len(path.parts) < 3 or any(c in value for c in "\r\n"):
        raise GateError("The remote scratch path is too broad. Choose a dedicated directory below the account directory and run again.")
    return str(path)


def completed_result(xml: str, returncode: int, files: list[str]) -> dict:
    """Require a completed process and a consistent report for every selected file."""
    if returncode not in (0, 1):
        raise GateError(f"The test process ended with exit {returncode}, so the contract run did not complete. Resolve the interruption or execution failure and run again.")
    try:
        root = ET.fromstring(xml)
        suites = list(root) if root.tag == "testsuites" else [root]
        if not suites or any(s.tag != "testsuite" for s in suites):
            raise ValueError("missing test suites")
        counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        cases = []
        for suite in suites:
            rows = suite.findall("testcase")
            values = {key: int(suite.attrib[key]) for key in counts}
            if any(v < 0 for v in values.values()) or values["tests"] != len(rows):
                raise ValueError("test count differs from recorded cases")
            for tag, key in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
                if values[key] != sum(len(row.findall(tag)) for row in rows):
                    raise ValueError(f"{key} count differs from recorded cases")
            if sum(values[k] for k in ("failures", "errors", "skipped")) > len(rows):
                raise ValueError("a case has conflicting outcomes")
            for row in rows:
                if sum(len(row.findall(tag)) for tag in ("failure", "error", "skipped")) > 1:
                    raise ValueError("a case has conflicting outcomes")
            for key in counts:
                counts[key] += values[key]
            cases.extend(rows)
        if not counts["tests"]:
            raise ValueError("no tests ran")
        covered = {row.attrib.get("file", "") for row in cases}
        if not set(files).issubset(covered):
            raise ValueError("selected files missing from the report: " + ", ".join(sorted(set(files) - covered)))
    except (ET.ParseError, ValueError, KeyError, TypeError) as error:
        raise GateError(f"The structured test report is missing or incomplete ({error}). Run the complete contract selection again.") from error
    if counts["errors"]:
        raise GateError("The contract run contains collection, setup or teardown errors. Correct those errors and run the complete selection again.")
    if (returncode == 0) != (counts["failures"] == 0):
        raise GateError("The process exit and structured report disagree about failures. Run the check again and inspect both records.")
    real, environmental = [], []
    for row in cases:
        failure = row.find("failure")
        if failure is None:
            continue
        nodeid = row.attrib.get("file", "") + "::" + row.attrib.get("name", "")
        message = failure.attrib.get("message", "").removeprefix("AssertionError: ")
        # Only this specific assertion is a shadow artifact, not any failure in
        # the same test. Setup/teardown failures are errors and were refused above.
        allowed = (nodeid in ENVIRONMENTAL
                   and row.attrib.get("classname") == "tests.test_doctor"
                   and message.splitlines()[0:1] == ["assert 'missing' == 'verified'"])
        (environmental if allowed else real).append(nodeid)
    passed = counts["tests"] - counts["failures"] - counts["skipped"]
    if not passed:
        raise GateError("No contract test passed. Restore the missing test prerequisites and run the complete selection again.")
    subtests = sum(bool(re.search(r"\[arwen-subtest-\d+\]$", row.attrib.get("name", "")))
                   for row in cases)
    return {"counts": {**counts, "passed": passed}, "subtest_cases": subtests,
            "counts_unit": "JUnit testcase rows, including individually recorded subtests",
            "failures": real,
            "environmental": environmental,
            "summary": f"{counts['failures']} failed, {passed} passed, {counts['skipped']} skipped"
                       + (f" ({subtests} individually recorded subtest cases included)" if subtests else "") }


def main(argv=None, *, defaults=None) -> int:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--node", default=defaults.get("node"))
    parser.add_argument("--python", default=defaults.get("python", "python3"))
    parser.add_argument("--remote-dir", default=defaults.get("remote_dir"))
    parser.add_argument("--cuda-path", default=defaults.get("cuda_path"))
    parser.add_argument("--evidence-dir", type=Path, default=defaults.get("evidence_dir"))
    args = parser.parse_args(argv)
    record = {"schema": "arwen.precut-gate.v2", "scope": SCOPE,
              "full_release_qualification": False, "status": "ERROR"}
    started = time.perf_counter()
    evidence = None
    try:
        if not args.node or args.node.startswith("-") or any(c in args.node for c in "\r\n\x00"):
            raise GateError("A remote execution account is missing or invalid. Supply --node and run again.")
        if not args.remote_dir or not args.evidence_dir:
            raise GateError("The scratch or evidence directory is missing. Supply --remote-dir and --evidence-dir and run again.")
        remote_base = remote_directory(args.remote_dir)
        tree = args.tree.resolve()
        head = checked(["git", "-C", str(tree), "rev-parse", "--verify", "HEAD^{commit}"], "Reading the candidate commit").stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", head):
            raise GateError("The candidate identity is not a complete commit. Select a Git worktree and run again.")
        record["candidate"] = head
        evidence = args.evidence_dir / f"precut-gate-{head[:9]}.json"
        dirty = checked(["git", "--no-optional-locks", "-C", str(tree), "status", "--porcelain"], "Reading the candidate status").stdout.strip()
        if dirty:
            raise GateError("The candidate has uncommitted changes that its archive would omit. Commit or stash them before running the check.")
        files = contract_files(tree)
        record["files"] = files
        attempt = uuid.uuid4().hex
        remote = remote_base + "/run-" + attempt
        record["remote_directory"] = remote
        quote = shlex.quote
        with tempfile.TemporaryDirectory(prefix="arwen-precut-") as scratch:
            archive = Path(scratch) / "candidate.tar"
            checked(["git", "-C", str(tree), "archive", "--format=tar", "--output=" + str(archive), head], "Archiving the candidate")
            if not archive.is_file() or not archive.stat().st_size:
                raise GateError("The candidate archive is missing or empty. Correct the archive failure and run again.")
            with archive.open("rb") as data:
                command = (f"mkdir -p -- {quote(remote_base)} && mkdir -- {quote(remote)} && "
                           f"tar --warning=no-timestamp -x -f - -C {quote(remote)}")
                checked(["ssh", "-o", "BatchMode=yes", args.node, command], "Transferring and extracting the candidate", stdin=data)
        report_path = remote + "/contract-" + attempt + ".xml"
        record["structured_report"] = report_path
        identity = ("import pathlib,gpuwm; "
                    f"expected=pathlib.Path({remote!r})/'gpuwm'/'__init__.py'; "
                    "actual=pathlib.Path(gpuwm.__file__).resolve(); "
                    "assert actual == expected.resolve(), (actual, expected); print(actual)")
        env = f"export PYTHONPATH={quote(remote + ':' + remote + '/gpuwm-data')} GPUWM_NO_LOCAL_GPU=1 PYTEST_ADDOPTS='' && "
        if args.cuda_path:
            env += f"export CUDA_PATH={quote(args.cuda_path)} LD_LIBRARY_PATH={quote(args.cuda_path + '/lib')} && "
        command = (f"cd {quote(remote)} && {env}{quote(args.python)} -c {quote(identity)} && "
                   f"{quote(args.python)} -m pytest -q -p no:cacheprovider -p tools.release.precut_gate -n 8 -rfE "
                   f"-m {quote('not gpu and not slow and not network')} "
                   f"-o addopts= -o junit_family=xunit1 --maxfail=0 --junitxml={quote(report_path)} "
                   + " ".join(quote(f) for f in files))
        result = run(["ssh", "-o", "BatchMode=yes", args.node, command])
        record["process_exit"] = result.returncode
        record["output_tail"] = (result.stdout + result.stderr)[-8000:]
        if result.returncode not in (0, 1):
            raise GateError(f"The contract process exited {result.returncode}. Resolve the execution failure and run the complete check again.")
        xml = checked(["ssh", "-o", "BatchMode=yes", args.node, "cat -- " + quote(report_path)], "Reading the completed test report").stdout
        record["report_sha256"] = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        record.update(completed_result(xml, result.returncode, files))
        record["status"] = "FAIL" if record["failures"] else "PASS"
        record["environmental_reasons"] = {key: ENVIRONMENTAL[key] for key in record["environmental"]}
        rc = 1 if record["failures"] else 0
    except (GateError, OSError, subprocess.SubprocessError) as error:
        record["error"] = str(error)
        rc = 2
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    if evidence is not None:
        try:
            evidence.parent.mkdir(parents=True, exist_ok=True)
            temporary = evidence.with_name(evidence.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(evidence)
        except OSError as error:
            print(f"precut gate: evidence could not be saved ({error}). Restore writable evidence storage and run again.", file=sys.stderr)
            return 2
    print(f"precut gate: {record['status']} ({SCOPE})")
    print(record.get("summary", record.get("error", "")))
    if evidence is not None:
        print(f"record: {evidence}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
