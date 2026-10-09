"""A transplanted lane retains its registry blob, with a new commit ID."""
import json
import subprocess
from copy import deepcopy

import pytest

from tools import registry_physics_history as history
from gpuwm.physics_registry import canonical_json


def _transplanted_record(tmp_path, monkeypatch):
    saved = history.record(["HEAD"])
    for row in saved["documents"].values():
        row["commit"] = "0000000000"
        row["registry_blob_sha1"] = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{history.REGISTRY}"],
            cwd=history.MODEL, text=True).strip()
    path = tmp_path / "history.json"
    path.write_text(canonical_json(saved), encoding="utf-8")
    monkeypatch.setattr(history, "REGISTRY_PHYSICS_HISTORY_PATH", path)
    return saved, path


def test_transplanted_registry_is_checked_from_its_real_git_blob(tmp_path, monkeypatch):
    _transplanted_record(tmp_path, monkeypatch)
    assert history.check() == []


def test_absent_blob_cannot_qualify_a_transplanted_registry(tmp_path, monkeypatch):
    saved, path = _transplanted_record(tmp_path, monkeypatch)
    for row in saved["documents"].values():
        row["registry_blob_sha1"] = "0" * 40
    path.write_text(json.dumps(saved), encoding="utf-8")
    failures = history.check()
    assert len(failures) == 1 and "missing" in failures[0]


def test_wrong_physics_parts_cannot_qualify_a_present_blob(tmp_path, monkeypatch):
    saved, path = _transplanted_record(tmp_path, monkeypatch)
    for key in saved["physics"]:
        saved["physics"][key] = {}
    path.write_text(json.dumps(saved), encoding="utf-8")
    failures = history.check()
    assert len(failures) == 1 and "differ" in failures[0]


def test_narrow_intake_write_preserves_older_preparation_identities(tmp_path, monkeypatch):
    previous = json.loads(history.REGISTRY_PHYSICS_HISTORY_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "retained-history.json"
    path.write_text(json.dumps(previous), encoding="utf-8")
    monkeypatch.setattr(history, "REGISTRY_PHYSICS_HISTORY_PATH", path)
    assert history.main(["--since", "HEAD", "--until", "HEAD", "--write"]) == 0
    after = json.loads(path.read_text(encoding="utf-8"))
    assert set(previous["documents"]) <= set(after["documents"])
    assert set(previous["physics"]) <= set(after["physics"])
    assert history.check() == []


def test_retained_history_refuses_conflicting_physics_parts():
    current = history.record(["HEAD"])
    previous = deepcopy(current)
    digest = next(iter(previous["physics"]))
    previous["physics"][digest] = {}
    with pytest.raises(ValueError, match="conflicting parts"):
        history.retain_history(previous, current)
