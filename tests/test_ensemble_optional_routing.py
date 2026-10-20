"""Generic ensemble operation when optional routing packages are absent."""
import builtins
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def excluded(monkeypatch):
    original = builtins.__import__
    attempts = []
    def guarded(name, *args, **kwargs):
        if (name == "gpuwm.hydro" or name.startswith("gpuwm.hydro.") or
                name.startswith("gpuwm.ensemble.nextgen_")):
            attempts.append(name)
            raise ModuleNotFoundError(f"excluded optional module: {name}", name=name)
        return original(name, *args, **kwargs)
    for name in tuple(sys.modules):
        if name == "gpuwm.hydro" or name.startswith("gpuwm.hydro."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.delenv("GPUWM_NEXTGEN_ENABLED", raising=False)
    monkeypatch.delenv("GPUWM_NEXTGEN_SETTINGS", raising=False)
    return attempts


def test_ordinary_session_without_routing(tmp_path, monkeypatch, excluded):
    # Exercise the production request, admission and original member runner.
    from test_ensemble_production_sizing import test_default_session_uses_every_visible_card_and_original_member_clocks
    test_default_session_uses_every_visible_card_and_original_member_clocks(tmp_path, monkeypatch)
    import json
    manifest = json.loads((tmp_path / "report.json").read_text())
    decisions = manifest["hydro_activation"]
    assert [row["member_id"] for row in decisions] == [0, 1, 2, 3]
    assert all(not row["enabled"] and row["reason"] == "no_prepared_hydro_configuration"
               for row in decisions)
    assert excluded == []


def test_disabled_table_without_runtime(tmp_path, excluded):
    from gpuwm.ensemble.optional_routing import member_settings_scope
    path = tmp_path / "disabled.toml"
    path.write_text("[hydro]\nenabled=false\n")
    with member_settings_scope(SimpleNamespace(experiment_config=path)):
        pass
    assert excluded == []


def test_native_eligibility_without_routing(excluded):
    from test_ensemble_prepared_batch import _source
    from gpuwm.ensemble.prepared_batch import native_prepared_eligibility
    inputs, node = _source()
    assert native_prepared_eligibility(inputs, node, members=10).eligible
    assert excluded == []


def test_absent_product_policy_and_generic_capture(tmp_path, excluded):
    from gpuwm.ensemble.optional_routing import read_product_options
    from gpuwm.ensemble.request import EnsembleRequest
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope, current_capture
    request = EnsembleRequest.from_mapping({"members": 2})
    assert "hydro" not in request.receipt()
    path = tmp_path / "ordinary.toml"
    path.write_text("[ensemble]\nmembers=2\n")
    assert read_product_options(path) is None
    assert read_product_options(None) is None
    rows = []
    capture = MemberOutputCapture(lambda **row: rows.append(row), member_id=1)
    with member_output_scope(capture):
        current_capture().submit(state=None, streamed=None, metadata={}, refl_field=None,
                                 valid_time="T", grid_id=1)
    assert current_capture() is None and rows[0]["member_id"] == 1
    assert excluded == []


def test_explicit_missing_products_cannot_be_silently_dropped(excluded):
    from gpuwm.ensemble.request import EnsembleRequest
    with pytest.raises(ValueError, match="products are unavailable"):
        EnsembleRequest.from_mapping({"members": 2, "hydro": {}})
    assert excluded == ["gpuwm.ensemble.nextgen_points"]


def test_explicit_enabled_runtime_cannot_be_silently_dropped(tmp_path, excluded):
    from gpuwm.ensemble.optional_routing import member_settings_scope
    path = tmp_path / "enabled.toml"
    path.write_text("[hydro]\nenabled=true\n")
    with pytest.raises(ValueError, match="runtime is unavailable"):
        with member_settings_scope(SimpleNamespace(experiment_config=path)):
            pytest.fail("an enabled forecast reached its runner without routing")
    assert excluded == ["gpuwm.hydro"]


def test_default_prepared_runtime_cannot_be_silently_dropped(tmp_path, excluded):
    from gpuwm.ensemble.optional_routing import member_settings_scope
    path = tmp_path / "prepared.toml"
    path.write_text('[hydro]\nrealization="prepared-test-row"\n')
    with pytest.raises(ValueError, match="runtime is unavailable"):
        with member_settings_scope(SimpleNamespace(experiment_config=path)):
            pytest.fail("prepared hydro reached its runner without the runtime")
    assert excluded == ["gpuwm.hydro"]
