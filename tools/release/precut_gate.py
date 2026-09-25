"""Check the candidate release contract set before export.

This check covers the contract files named by the publication workflow, including
the native distribution check. It does not qualify built distributions, desktop
packages, or an installed end-to-end forecast. A stopped test process must never
produce a passing receipt just because it printed no failing test names.

It runs with -m "not gpu and not slow and not network" under GPUWM_NO_LOCAL_GPU=1,
so no test that needs a card runs here. The card stage is
tools/release/precut_gpu_gate.py, which reuses the archive, transfer, identity
and strict-completion functions below rather than carrying a second copy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
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
CONTRACT_MARKERS = "not gpu and not slow and not network"


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


def remote_path(value: str | None) -> str | None:
    """A POSIX path meant for the node, with Git Bash's automatic conversion
    of a leading slash into its own install prefix undone. It hit --remote-dir
    first and, measured 2026-09-17, --python as well: the card stage's probe
    ran the node's interpreter under Git's own install prefix, a directory
    that exists on the controller and not on the node."""
    marker = "/Program Files/Git"
    if value and marker in value:
        value = value[value.index(marker) + len(marker):]
    return value


def remote_directory(value: str) -> str:
    value = remote_path(value)
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


def skipped_cases(xml: str) -> list[dict]:
    """Every skipped JUnit case by name with its reason, from a report
    completed_result has already accepted."""
    root = ET.fromstring(xml)
    suites = list(root) if root.tag == "testsuites" else [root]
    rows = [row for suite in suites for row in suite.findall("testcase")]
    out = []
    for row in rows:
        skipped = row.find("skipped")
        if skipped is None:
            continue
        out.append({"test": row.attrib.get("file", "") + "::" + row.attrib.get("name", ""),
                    "reason": skipped.attrib.get("message", "") or (skipped.text or "").strip()})
    return out


# ---- the machinery both gates share ----------------------------------------

def validated_node(node: str | None) -> str:
    if not node or node.startswith("-") or any(c in node for c in "\r\n\x00"):
        raise GateError("A remote execution account is missing or invalid. Supply --node and run again.")
    return node


def candidate_commit(tree: Path) -> str:
    """The clean candidate's full commit, refusing a dirty tree."""
    head = checked(["git", "-C", str(tree), "rev-parse", "--verify", "HEAD^{commit}"], "Reading the candidate commit").stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise GateError("The candidate identity is not a complete commit. Select a Git worktree and run again.")
    dirty = checked(["git", "--no-optional-locks", "-C", str(tree), "status", "--porcelain"], "Reading the candidate status").stdout.strip()
    if dirty:
        raise GateError("The candidate has uncommitted changes that its archive would omit. Commit or stash them before running the check.")
    return head


def ship_candidate(tree: Path, head: str, node: str, remote_base: str) -> tuple[str, str]:
    """Archive exactly ``head`` and extract it into a new directory below
    ``remote_base`` on ``node``. Returns ``(remote_directory, attempt)``."""
    attempt = uuid.uuid4().hex
    remote = remote_base + "/run-" + attempt
    quote = shlex.quote
    with tempfile.TemporaryDirectory(prefix="arwen-precut-") as scratch:
        archive = Path(scratch) / "candidate.tar"
        checked(["git", "-C", str(tree), "archive", "--format=tar", "--output=" + str(archive), head], "Archiving the candidate")
        if not archive.is_file() or not archive.stat().st_size:
            raise GateError("The candidate archive is missing or empty. Correct the archive failure and run again.")
        with archive.open("rb") as data:
            command = (f"mkdir -p -- {quote(remote_base)} && mkdir -- {quote(remote)} && "
                       f"tar --warning=no-timestamp -x -f - -C {quote(remote)}")
            checked(["ssh", "-o", "BatchMode=yes", node, command], "Transferring and extracting the candidate", stdin=data)
    return remote, attempt


def identity_probe(remote: str) -> str:
    """Python that proves the interpreter imports the shipped tree, not another."""
    return ("import pathlib,gpuwm; "
            f"expected=pathlib.Path({remote!r})/'gpuwm'/'__init__.py'; "
            "actual=pathlib.Path(gpuwm.__file__).resolve(); "
            "assert actual == expected.resolve(), (actual, expected); print(actual)")


def remote_prefix(remote: str, python: str, *, cuda_path: str | None = None,
                  local_gpu: bool = False) -> str:
    """``cd``, the environment and the identity probe, ending in ``&&``.

    ``local_gpu=False`` bans the device with GPUWM_NO_LOCAL_GPU=1, the contract
    gate's setting; the card stage passes True and runs on the node's card.
    """
    quote = shlex.quote
    env = f"export PYTHONPATH={quote(remote + ':' + remote + '/gpuwm-data')} "
    env += "" if local_gpu else "GPUWM_NO_LOCAL_GPU=1 "
    env += "PYTEST_ADDOPTS='' && "
    if cuda_path:
        env += f"export CUDA_PATH={quote(cuda_path)} LD_LIBRARY_PATH={quote(cuda_path + '/lib')} && "
    return f"cd {quote(remote)} && {env}{quote(python)} -c {quote(identity_probe(remote))} && "


def remote_test_command(python: str, report_path: str, files: list[str], *,
                   workers: int = 8, markers: str | None = CONTRACT_MARKERS) -> str:
    """The remote pytest invocation both gates run, differing only in the
    worker count, the marker expression and the selection."""
    quote = shlex.quote
    command = (f"{quote(python)} -m pytest -q -p no:cacheprovider -p tools.release.precut_gate "
               f"-n {int(workers)} -rfE ")
    if markers:
        command += f"-m {quote(markers)} "
    command += (f"-o addopts= -o junit_family=xunit1 --maxfail=0 --junitxml={quote(report_path)} "
                + " ".join(quote(f) for f in files))
    return command


def run_remote(node: str, command: str):
    return run(["ssh", "-o", "BatchMode=yes", node, command])


def read_report(node: str, report_path: str) -> str:
    return checked(["ssh", "-o", "BatchMode=yes", node, "cat -- " + shlex.quote(report_path)],
                   "Reading the completed test report").stdout


def evidence_path(directory: Path, stem: str) -> tuple[Path, dict | None]:
    """The receipt this run writes, and the latest earlier receipt for the
    same commit, as ``{"path", "sha256"}`` or None.

    A repeated run for one commit never replaces an earlier receipt: the
    first is ``<stem>.json``, the next ``<stem>.2.json`` and so on, and the
    new receipt names the one it follows with its sha256.  A card-stage
    reading that was discarded because the card was shared stays on record
    beside the run that replaced it, instead of vanishing under the next
    receipt as the first card-stage receipts did.
    """
    first = directory / f"{stem}.json"
    if not first.exists():
        return first, None
    previous, index = first, 2
    while (directory / f"{stem}.{index}.json").exists():
        previous, index = directory / f"{stem}.{index}.json", index + 1
    digest = hashlib.sha256(previous.read_bytes()).hexdigest()
    return directory / f"{stem}.{index}.json", {"path": previous.name, "sha256": digest}


#: The rule once it has been loaded. Loading reads a file and runs it,
#: while the record writer runs the rule once per string in a record.
_REDACTION = None


def own_tree() -> Path:
    """The tree this gate file ships in. A gate checks the tree it sits in,
    so that tree is also where its rules come from."""
    return Path(__file__).resolve().parents[2]


def _rule_from_this_tree_installed():
    """The rule under the name ``gpuwm.report_bundle``, taken only when that
    name resolves to a file inside this gate's own tree."""
    try:
        module = importlib.import_module("gpuwm.report_bundle")
    except Exception:  # noqa: BLE001 - any resolution failure means "not here"
        return None
    where = getattr(module, "__file__", None)
    if not where:
        return None
    try:
        same = Path(where).resolve() == own_tree() / "gpuwm" / "report_bundle.py"
    except OSError:
        return None
    return getattr(module, "redact_home_directories", None) if same else None


def _rule_from_this_tree_file():
    """The rule read straight out of this gate's own tree, by path."""
    source = own_tree() / "gpuwm" / "report_bundle.py"
    if not source.is_file():
        return None
    root = str(own_tree())
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("precut_gate_home_rule", source)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return getattr(module, "redact_home_directories", None)


def _rule_from_any_distribution():
    """The rule from whatever ``gpuwm`` this interpreter can reach: the last
    resort, for a gate file that has been copied away from its own tree."""
    try:
        module = importlib.import_module("gpuwm.report_bundle")
    except Exception:  # noqa: BLE001
        return None
    return getattr(module, "redact_home_directories", None)


def home_directory_redaction():
    """The one home-directory rule, as THIS gate's own tree spells it.

    THE BREAKAGE THIS PREVENTS: both gates run as scripts as well as
    modules, and a script's first path entry is its own directory, not the
    tree root. The bare name ``gpuwm`` then binds whichever distribution
    the interpreter has installed, which on a controller is an editable
    install of a DIFFERENT tree. Measured on such a controller, importing
    this rule by name raised ImportError, because that older version has no
    such name, and the failure surfaced inside the record writer: a
    completed remote run ended in a traceback and wrote no receipt at all.
    Had that version carried the name instead, the record would have been
    scrubbed by another tree's rule and the miss would have been silent. So
    the name is accepted only when it resolves to a file in this tree, and
    the file is read directly otherwise.

    Fetched here rather than imported at module scope: this file is also
    loaded as a pytest plugin on the node (``-p tools.release.precut_gate``),
    and a plugin that imports the product at load time does that work on
    every worker for nothing. The result is kept, so a record costs one load.
    """
    global _REDACTION
    if _REDACTION is not None:
        return _REDACTION
    notes = []
    for where, load in (("installed, and this tree", _rule_from_this_tree_installed),
                        ("this tree's own file", _rule_from_this_tree_file),
                        ("any reachable distribution", _rule_from_any_distribution)):
        try:
            rule = load()
        except Exception as error:  # noqa: BLE001 - every reason is reported below
            notes.append(f"{where}: {type(error).__name__}: {error}")
            continue
        if rule is None:
            notes.append(f"{where}: carries no such rule")
            continue
        _REDACTION = rule
        return rule
    raise GateError(
        "The home-directory rule every record is written through could not be "
        "loaded, and a record written without it can carry a machine path into "
        "a shipped file (" + "; ".join(notes) + "). Run the gate from a complete "
        "tree and run the check again.")


def without_machine_paths(value):
    """``value`` with every home-directory prefix in every string gone.

    THE BREAKAGE THIS PREVENTS: these receipts are committed beside the
    tests they back, and the machine paths they carried -- the node's
    scratch directory, the interpreter a sampled process ran, the
    traceback of a failing pin -- were 54 of the 66 findings that stopped
    the 2.7.6 release snapshot from building, counted over the receipts
    folder the card stage commits to: 35 in the receipt files this writer
    produces, 16 in the phase-2 pin comparisons and 3 in pytest structured
    reports; 57 of the 66 counting everything under tests/data/receipts/,
    which adds the two notes that sit beside them.
    Scrubbing in the WRITER is
    what stops the class coming back: a record cannot acquire a machine
    path between here and the file, wherever that file is written, so the
    receipts kept outside the tree are clean too.

    Keys are left alone, the way the report bundler leaves them alone:
    every key in these records is a constant in these two files, while a
    value is whatever the node said.
    """
    redact = home_directory_redaction()

    def walk(node):
        if isinstance(node, str):
            return redact(node)
        if isinstance(node, dict):
            return {key: walk(item) for key, item in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(value)


def record_name(value: str) -> str:
    """The last segment of a remote path, which is what a record may carry.

    The directory a run sits under is an ARGUMENT to the gate, not a fact
    about the run: the same candidate at the same commit gives the same
    result whatever scratch root the operator chose. So the record keeps
    the part that identifies the run -- the run directory's own name below
    that root, and the report's own name inside the run directory -- and
    the caller keeps the absolute path it needs to talk to the node.
    """
    return PurePosixPath(value).name


def save_evidence(evidence: Path, record: dict) -> None:
    """Write the receipt atomically, with LF line endings on every platform
    and with no machine path in it: a receipt written on Windows carried CRLF
    and the repository's line-ending hook refused to commit it beside the
    tests it backs, and a receipt carrying the node's home directory refused
    the release snapshot itself."""
    evidence.parent.mkdir(parents=True, exist_ok=True)
    temporary = evidence.with_name(evidence.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(without_machine_paths(record), indent=2) + "\n")
    temporary.replace(evidence)


def add_arguments(parser: argparse.ArgumentParser, defaults: dict) -> None:
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--node", default=defaults.get("node"))
    parser.add_argument("--python", default=defaults.get("python", "python3"))
    parser.add_argument("--remote-dir", default=defaults.get("remote_dir"))
    parser.add_argument("--cuda-path", default=defaults.get("cuda_path"))
    parser.add_argument("--evidence-dir", type=Path, default=defaults.get("evidence_dir"))


def finish(record: dict, evidence: Path | None, started: float, rc: int, label: str) -> int:
    """Store the receipt and print the verdict; a receipt that cannot be
    stored is not a pass, and neither is one that cannot be scrubbed. The
    writer refuses rather than dropping a machine path into a shipped file,
    and a refusal here is reported as a verdict instead of a traceback."""
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    if evidence is not None:
        try:
            save_evidence(evidence, record)
        except (OSError, GateError) as error:
            print(f"{label}: evidence could not be saved ({error}). Restore writable evidence storage and run again.", file=sys.stderr)
            return 2
    print(f"{label}: {record['status']} ({record['scope']})")
    print(record.get("summary", record.get("error", "")))
    if evidence is not None:
        print(f"record: {evidence}")
    return rc


def main(argv=None, *, defaults=None) -> int:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_arguments(parser, defaults)
    args = parser.parse_args(argv)
    record = {"schema": "arwen.precut-gate.v2", "scope": SCOPE,
              "full_release_qualification": False, "status": "ERROR"}
    started = time.perf_counter()
    evidence = None
    try:
        node = validated_node(args.node)
        if not args.remote_dir or not args.evidence_dir:
            raise GateError("The scratch or evidence directory is missing. Supply --remote-dir and --evidence-dir and run again.")
        remote_base = remote_directory(args.remote_dir)
        tree = args.tree.resolve()
        head = candidate_commit(tree)
        record["candidate"] = head
        evidence, previous = evidence_path(args.evidence_dir, f"precut-gate-{head[:9]}")
        if previous is not None:
            record["previous_receipt"] = previous
        files = contract_files(tree)
        record["files"] = files
        remote, attempt = ship_candidate(tree, head, node, remote_base)
        record["remote_directory"] = record_name(remote)
        report_path = remote + "/contract-" + attempt + ".xml"
        record["structured_report"] = record_name(report_path)
        python, cuda_path = remote_path(args.python), remote_path(args.cuda_path)
        command = (remote_prefix(remote, python, cuda_path=cuda_path)
                   + remote_test_command(python, report_path, files))
        result = run_remote(node, command)
        record["process_exit"] = result.returncode
        record["output_tail"] = (result.stdout + result.stderr)[-8000:]
        if result.returncode not in (0, 1):
            raise GateError(f"The contract process exited {result.returncode}. Resolve the execution failure and run the complete check again.")
        xml = read_report(node, report_path)
        record["report_sha256"] = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        record.update(completed_result(xml, result.returncode, files))
        record["status"] = "FAIL" if record["failures"] else "PASS"
        record["environmental_reasons"] = {key: ENVIRONMENTAL[key] for key in record["environmental"]}
        rc = 1 if record["failures"] else 0
    except (GateError, OSError, subprocess.SubprocessError) as error:
        record["error"] = str(error)
        rc = 2
    return finish(record, evidence, started, rc, "precut gate")


if __name__ == "__main__":
    raise SystemExit(main())
