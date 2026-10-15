"""A tiny actual job proves isolation, step order, exports and failure propagation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tools import workflow_job_replay as replay


@pytest.fixture
def tree(tmp_path):
    tree = tmp_path / "tree"
    (tree / ".github/workflows").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    # Test repositories only; no engine identity override.
    subprocess.run(["git", "-C", str(tree), "config", "user.name", "rehearsal fixture"], check=True)
    subprocess.run(["git", "-C", str(tree), "config", "user.email", "fixture@example.invalid"], check=True)
    return tree


def save(tree, steps, **extra):
    version = ".".join(map(str, sys.version_info[:2]))
    document = {"jobs": {"test": {"runs-on": "ubuntu-24.04", "env": {"DECLARED": "yes"},
        "steps": [{"uses": "actions/checkout@fixture"},
                  {"uses": "actions/setup-python@fixture", "with": {"python-version": version}}, *steps], **extra}}}
    (tree / ".github/workflows/publish.yml").write_text(yaml.safe_dump(document))
    subprocess.run(["git", "-C", str(tree), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tree), "commit", "-qm", "fixture"], check=True)


def invoke(tree, work):
    return replay.main(["--tree", str(tree), "--work", str(work), "--python", sys.executable])


@pytest.mark.skipif(os.name != "posix", reason="Linux job driver")
def test_actual_steps_export_env_build_outside_tree_and_do_not_inherit_host(tree, tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_RW_FETCH", "/host/binary")
    monkeypatch.setenv("HOST_SENTINEL", "must not leak")
    save(tree, [{"run": 'test -z "${HOST_SENTINEL:-}"; test -z "${GPUWM_RW_FETCH:-}"; test "$DECLARED" = yes; echo NEXT=ok >> "$GITHUB_ENV"; mkdir -p "$RUNNER_TEMP/build"; echo built > "$RUNNER_TEMP/build/native"'},
        {"run": 'test "$NEXT" = ok; test -f "$NATIVE"', "env": {"NATIVE": "${{ runner.temp }}/build/native"}},
        {"uses": "actions/upload-artifact@fixture", "with": {"path": "${{ runner.temp }}/build/native"}}])
    assert invoke(tree, tmp_path / "work") == 0


@pytest.mark.skipif(os.name != "posix", reason="Linux job driver")
def test_failure_stops_later_steps_and_unsupported_features_refuse(tree, tmp_path):
    save(tree, [{"run": "exit 7"}, {"run": "touch should-not-exist"}])
    assert invoke(tree, tmp_path / "fail-work") == 7
    assert not (tree / "should-not-exist").exists()
    save(tree, [{"run": "true", "if": "false"}])
    with pytest.raises(replay.Unreproducible, match="does not evaluate"):
        invoke(tree, tmp_path / "unknown-work")


def test_native_contamination_and_unsupported_runner_refuse(tree, tmp_path):
    save(tree, [], strategy={"matrix": {"os": ["windows-2025"]}})
    with pytest.raises(replay.Unreproducible, match="matrix"):
        invoke(tree, tmp_path / "matrix-work")
    (tree / "tools/test/target").mkdir(parents=True)
    with pytest.raises(replay.Unreproducible, match="built natives"):
        replay.refuse_existing_natives(tree, [])


def test_unknown_expressions_and_multiline_exports_refuse(tmp_path):
    with pytest.raises(replay.Unreproducible, match="expression"):
        replay.render("${{ secrets.value }}", {})
    path = tmp_path / "env"
    path.write_text("VALUE<<EOF\n")
    with pytest.raises(replay.Unreproducible, match="GITHUB_ENV"):
        replay.read_env_file(path)
