"""Static qualification must not disguise broken or incomplete CI execution."""
from __future__ import annotations

import ast
from importlib import import_module
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "tools/battery/static_qualification.json"
PYTHON_SOURCE = ROOT / "tests/test_static_rust_parity.py"


@pytest.fixture
def policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def runner():
    return import_module("tools.battery.run_static_qualification")


def _rust_source(policy):
    return (ROOT / policy["rust_source"]).read_text(encoding="utf-8")


def _rust_line(source, function, fragment):
    start = source.index("fn " + function + "(")
    end = source.find("\nfn ", start + 1)
    offset = source.index(fragment, start, None if end < 0 else end)
    return source[:offset].count("\n") + 1


def _native_log(policy, *, failures=None, omitted=(), ignored=(), duplicate=None,
                source_path="tests/lane1_goldens.rs", extra=""):
    failures = failures or {}
    names = policy["rust_functional_tests"] + policy["rust_qualification_tests"]
    rows = [name for name in names if name not in omitted]
    rows.extend(name for name in failures if name not in names)
    events = []
    for name in rows:
        result = "FAILED" if name in failures else "ignored" if name in ignored else "ok"
        events.append(f"test {name} ... {result}")
    if duplicate:
        events.append(f"test {duplicate} ... ok")
    failed = len(set(rows) & failures.keys())
    skipped = len(set(rows) & set(ignored))
    lines = [f"running {len(rows)} tests", *events]
    if failed:
        lines.append("\nfailures:\n")
        for name, (line, message) in failures.items():
            lines.extend([f"---- {name} stdout ----",
                f"thread '{name}' panicked at {source_path}:{line}:9:", message, ""])
        lines.extend(["failures:", *(f"    {name}" for name in failures)])
    lines.append(f"test result: {'FAILED' if failed else 'ok'}. "
        f"{len(rows)-failed-skipped} passed; {failed} failed; {skipped} ignored; "
        "0 measured; 0 filtered out; finished in 0.12s")
    if failed:
        lines.append("error: test failed, to rerun pass `-p static-fields --test lane1_goldens`")
    lines.append(extra)
    return "\n".join(lines)


def _junit(tmp_path, policy, *, failures=None, errors=None, skips=None,
           omit=(), duplicate=None, empty=False):
    failures, errors, skips = failures or {}, errors or {}, skips or {}
    names = [] if empty else [n for n in policy["python_tests"] if n not in omit]
    if duplicate:
        names.append(duplicate)
    testsuites = ET.Element("testsuites")
    suite = ET.SubElement(testsuites, "testsuite", name="pytest", tests=str(len(names)),
        failures=str(len(failures)), errors=str(len(errors)), skipped=str(len(skips)))
    for nodeid in names:
        filename, classname, name = nodeid.split("::")
        item = ET.SubElement(suite, "testcase", name=name,
            classname=filename[:-3].replace("/", ".") + "." + classname,
            file=filename, time="0.01")
        if nodeid in failures:
            message = failures[nodeid]
            failure = ET.SubElement(item, "failure", message=message)
            exception = message.partition(":")[0] if ":" in message else "AssertionError"
            failure.text = f"E   {message}\n\n{filename}:120: {exception}"
        if nodeid in errors:
            error = ET.SubElement(item, "error", message=errors[nodeid])
            error.text = errors[nodeid]
        if nodeid in skips:
            ET.SubElement(item, "skipped", type="pytest.skip", message=skips[nodeid])
    path = tmp_path / "static.xml"
    ET.ElementTree(testsuites).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_policy_accounts_for_every_native_test_and_exact_functional_controls(policy):
    source = _rust_source(policy)
    actual = set(re.findall(r"#\[test\]\s*fn\s+(\w+)\s*\(", source))
    functional = set(policy["rust_functional_tests"])
    qualification = set(policy["rust_qualification_tests"])
    assert functional == {"grid_refusals_name_the_breakage", "corridor_crop_bit_equal_and_refuses_off_corridor"}
    assert len(actual) == 14 and len(qualification) == 12
    assert not functional & qualification
    assert actual == functional | qualification
    assert len(policy["rust_functional_tests"] + policy["rust_qualification_tests"]) == 14


def test_only_three_exact_python_checks_receive_the_qualification_marker(policy):
    tree = ast.parse(PYTHON_SOURCE.read_text(encoding="utf-8"))
    marker = "static_platform_qualification"
    def marked(nodes):
        return any(isinstance(node, ast.Attribute) and node.attr == marker
                   for parent in nodes for node in ast.walk(parent))
    # A module/class-wide marker would quietly remove functional tests too.
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            assert not marked([node])
    observed = set()
    for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        assert not marked(cls.decorator_list)
        for method in (node for node in cls.body if isinstance(node, ast.FunctionDef)):
            if marked(method.decorator_list):
                observed.add(f"tests/test_static_rust_parity.py::{cls.name}::{method.name}")
    assert observed == set(policy["python_tests"])
    assert len(observed) == 3
    assert policy["python_allowed_skips"] == {
        "tests/test_static_rust_parity.py::TestLane2BuildParity::test_build_static_bytes_equal":
        "WPS_GEOG reference tree not present"}


def test_complete_native_success_is_passed(runner, policy):
    report = runner.classify_native(_native_log(policy), 0, policy, _rust_source(policy))
    assert report["status"] == "passed"
    assert not report["operational_errors"]


@pytest.mark.parametrize("windows_path", [False, True])
def test_only_known_native_assertion_locations_are_qualification_failures(runner, policy, windows_path):
    source = _rust_source(policy)
    line = _rust_line(source, "assert_f64_bits", "assert!(")
    failed = policy["rust_qualification_tests"][0]
    path = r"C:\checkout\tools\rustwx\crates\static-fields\tests\lane1_goldens.rs" if windows_path else "tests/lane1_goldens.rs"
    output = _native_log(policy, failures={failed:(line,"assertion failed: observed.to_bits() == expected.to_bits()")}, source_path=path)
    assert runner.classify_native(output, 101, policy, source)["status"] == "assertion_failure"


@pytest.mark.parametrize("kind", ["functional", "unwrap", "manifest", "wrong_source", "unknown_test"])
def test_native_operational_faults_cannot_be_downgraded_to_platform_mismatch(runner, policy, kind):
    source = _rust_source(policy)
    name = policy["rust_qualification_tests"][0]
    line = _rust_line(source, "assert_f64_bits", "assert!(")
    message = "assertion failed: compared bytes differ"
    path = "tests/lane1_goldens.rs"
    if kind == "functional":
        name = policy["rust_functional_tests"][0]
    elif kind == "unwrap":
        line = _rust_line(source, "lambert_chain", ".unwrap()")
        message = "called `Result::unwrap()` on an `Err` value: missing fixture"
    elif kind == "manifest":
        line = source[:source.index('assert_eq!(manifest["numpy"]')].count("\n") + 1
    elif kind == "wrong_source":
        path = "tests/unrelated_test.rs"
    else:
        name = "unknown_native_test"
    output = _native_log(policy, failures={name:(line,message)}, source_path=path)
    assert runner.classify_native(output, 101, policy, source)["status"] == "operational_failure"


@pytest.mark.parametrize("kind", ["zero", "truncated", "duplicate", "ignored", "summary", "exit", "build"])
def test_native_run_completeness_and_exit_status_remain_strict(runner, policy, kind):
    names = policy["rust_functional_tests"] + policy["rust_qualification_tests"]
    code = 0
    options = {}
    if kind == "truncated": options["omitted"] = [names[-1]]
    if kind == "duplicate": options["duplicate"] = names[-1]
    if kind == "ignored": options["ignored"] = [names[-1]]
    output = _native_log(policy, **options)
    if kind == "zero": output = "running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
    if kind == "summary": output = output.replace("14 passed;", "13 passed;")
    if kind == "exit": code = 1
    if kind == "build": output, code = "error: could not compile `static-fields` due to previous error", 101
    assert runner.classify_native(output, code, policy, _rust_source(policy))["status"] == "operational_failure"


def test_known_npz_byte_comparison_is_distinct_from_an_arbitrary_panic(runner, policy):
    source = _rust_source(policy)
    name = "npz_seal_bytes_equal_python"
    line = _rust_line(source, name, "panic!(")
    message = "NPZ bytes differ first at offset 30: 03 != 00"
    output = _native_log(policy, failures={name:(line,message)})
    assert runner.classify_native(output, 101, policy, source)["status"] == "assertion_failure"
    arbitrary = _native_log(policy, failures={name:(line,"out of disk space while writing NPZ")})
    assert runner.classify_native(arbitrary, 101, policy, source)["status"] == "operational_failure"
    changed_guard = source.replace("if observed != expected {", "if true {")
    assert runner.classify_native(output, 101, policy, changed_guard)["status"] == "operational_failure"


@pytest.mark.parametrize("message", ["AssertionError: Arrays are not equal", "assert 1 == 2"])
def test_python_success_and_an_exact_assertion_failure_are_distinct(runner, policy, tmp_path, message):
    path = _junit(tmp_path, policy)
    assert runner.classify_python(path, 0, policy)["status"] == "passed"
    path = _junit(tmp_path, policy, failures={policy["python_tests"][0]:message})
    assert runner.classify_python(path, 1, policy)["status"] == "assertion_failure"


@pytest.mark.parametrize("kind", ["error", "wrong_exception", "bridge_failure", "zero", "missing", "duplicate", "exit", "bad_xml"])
def test_python_harness_failures_are_not_qualification_differences(runner, policy, tmp_path, kind):
    name = policy["python_tests"][0]
    code = 1
    options = {}
    if kind == "error": options["errors"] = {name:"AssertionError: fixture setup failed"}
    if kind == "wrong_exception": options["failures"] = {name:"RuntimeError: AssertionError in subprocess text"}
    if kind == "bridge_failure": options["failures"] = {name:"Failed: Rust static-fields bridge is not loadable"}
    if kind == "zero": options["empty"] = True
    if kind == "missing": options["omit"] = [name]
    if kind == "duplicate": options["duplicate"] = name
    if kind in {"zero", "missing", "duplicate", "bad_xml"}: code = 0
    path = _junit(tmp_path, policy, **options)
    if kind == "bad_xml": path.write_text("not XML", encoding="utf-8")
    assert runner.classify_python(path, code, policy)["status"] == "operational_failure"


def test_only_the_named_missing_wps_tree_skip_is_allowed(runner, policy, tmp_path):
    allowed = next(iter(policy["python_allowed_skips"]))
    reason = policy["python_allowed_skips"][allowed]
    path = _junit(tmp_path, policy, skips={allowed:reason + " at /fixtures/WPS_GEOG"})
    report = runner.classify_python(path, 0, policy)
    assert report["status"] != "operational_failure"
    assert report["skipped"]
    for name, message in [(allowed, "bridge is missing"), (policy["python_tests"][0], reason)]:
        path = _junit(tmp_path, policy, skips={name:message})
        assert runner.classify_python(path, 0, policy)["status"] == "operational_failure"
