"""Registry history admits new declarations and rejects changed identities."""
import hashlib
import json

import pytest

import test_registry_physics_identity as identity
from gpuwm import physics_registry as registry_module
from tools import registry_physics_history as history_tool


def _extended_history(tmp_path, monkeypatch):
    registry = registry_module.physics_registry()
    options = registry["components"]["microphysics"]["options"]
    options["future-option"] = dict(options["kessler-mp1"])
    history = registry_module.registry_physics_history()
    digest = registry_module.registry_sha256(registry)
    physics = registry_module.registry_physics_sha256(registry)
    history["documents"][digest] = {
        "commit": "eeeeeeeeee", "physics_sha256": physics}
    history["physics"][physics] = registry_module.registry_physics_parts(registry)
    saved = tmp_path / "history.json"
    saved.write_text(json.dumps(history))
    monkeypatch.setattr(registry_module, "_REGISTRY", registry)
    monkeypatch.setattr(registry_module, "_registry_physics_history", history)
    monkeypatch.setattr(history_tool, "REGISTRY_PHYSICS_HISTORY_PATH", saved)
    original = history_tool._document
    monkeypatch.setattr(history_tool, "_document",
        lambda commit: registry if commit == "eeeeeeeeee" else original(commit))
    return registry, history, saved, physics


def test_a_new_registered_option_needs_no_second_identity_inventory(tmp_path, monkeypatch):
    _extended_history(tmp_path, monkeypatch)
    identity.test_the_history_resolves_the_280_document_and_its_successors()


def test_a_changed_recorded_part_still_fails(tmp_path, monkeypatch):
    _, history, saved, physics = _extended_history(tmp_path, monkeypatch)
    part = "components.microphysics.options.future-option"
    history["physics"][physics][part] = hashlib.sha256(b"changed identity").hexdigest()
    saved.write_text(json.dumps(history))
    with pytest.raises(AssertionError):
        identity.test_the_history_resolves_the_280_document_and_its_successors()
    assert any("recorded parts" in failure for failure in history_tool.check())


def test_a_changed_registry_identity_still_moves_the_receipt(tmp_path, monkeypatch):
    registry, _, _, physics = _extended_history(tmp_path, monkeypatch)
    option = registry["components"]["microphysics"]["options"]["future-option"]
    option["consumers"] = dict(option["consumers"])
    option["consumers"]["restart_algorithm_identity"] = "changed-v2"
    assert registry_module.registry_physics_sha256(registry) != physics
    with pytest.raises(AssertionError):
        identity.test_the_history_resolves_the_280_document_and_its_successors()
