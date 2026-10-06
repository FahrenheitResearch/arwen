"""Runtime physics choices keep the original prepared authorities intact."""
import pytest

from gpuwm import prepared_single_domain_forecast as runner
from gpuwm.experiment import load_experiment
from test_prepared_single_domain_forecast import (
    _bind_synthetic_preflight_geometry, _prepared_fixture, _sha256)


def _preflight(fixture, overrides=None):
    return runner.preflight_prepared_forecast(
        source=fixture.source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.RUC_PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds,
        history_interval_seconds=load_experiment(fixture.experiment).root.history_interval_s,
        runtime_run_overrides=overrides)


def test_runtime_ruc_choices_preserve_original_files_and_prepared_identity(tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "era5", physics_profile=runner.RUC_PHYSICS_PROFILE)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    original = _preflight(fixture)
    hashes = {"proof": _sha256(fixture.proof), "experiment": _sha256(fixture.experiment),
              "header": _sha256(fixture.prepared / "prepared-cache" / "header.json")}
    requested = {"rdlai2d": True, "usemonalb": True,
                 "ruc_qvg_cold_start": "air", "ruc_2m_diagnostic": "log_profile"}
    executed = _preflight(fixture, requested)
    assert executed.cache_identity == original.cache_identity
    assert executed.file_sha256 == original.file_sha256
    assert executed.experiment.root.run.rdlai2d is True
    assert executed.experiment.root.run.ruc_qvg_cold_start == "air"
    changes = {row["field"]: row for row in executed.execution_plan["physics_overrides"]}
    assert set(changes) == {f"domain_config.run.{key}" for key in requested}
    assert changes["domain_config.run.rdlai2d"]["prepared"] is False
    assert changes["domain_config.run.rdlai2d"]["executed"] is True
    assert all(row["prepared_state_changed"] is False for row in changes.values())
    assert all(row["model_state_or_physics_changed"] is True for row in changes.values())
    requested["rdlai2d"] = False
    assert executed.preflight_arguments["runtime_run_overrides"]["rdlai2d"] is True
    runner._verify_inputs_unchanged(executed)
    assert hashes == {"proof": _sha256(fixture.proof), "experiment": _sha256(fixture.experiment),
                      "header": _sha256(fixture.prepared / "prepared-cache" / "header.json")}


def test_no_runtime_override_preserves_the_existing_plan(tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "era5", physics_profile=runner.RUC_PHYSICS_PROFILE)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    original = _preflight(fixture)
    empty = _preflight(fixture, {})
    assert original.execution_plan == empty.execution_plan
    assert original.execution_plan["physics_overrides"] == []
    assert "runtime_run_overrides" not in original.preflight_arguments


@pytest.mark.parametrize("overrides", [{"nx": 99}, {"num_soil_layers": 6}, {"unknown": 1}])
def test_runtime_mapping_rejects_preparation_inputs_and_unknown_fields(tmp_path, monkeypatch, overrides):
    fixture = _prepared_fixture(tmp_path, "era5", physics_profile=runner.RUC_PHYSICS_PROFILE)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    with pytest.raises(ValueError, match="changes a preparation input or is unknown"):
        _preflight(fixture, overrides)


def test_runtime_mapping_rejects_invalid_physics_values(tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "era5", physics_profile=runner.RUC_PHYSICS_PROFILE)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    with pytest.raises(ValueError, match="ruc_qvg_cold_start"):
        _preflight(fixture, {"ruc_qvg_cold_start": "unknown"})


@pytest.mark.parametrize("overrides", [
    {"rdlai2d": "true"}, {"usemonalb": 1},
    {"diff_6th_factor2": float("nan")}, {"mynn_sfclay_variant": 1}])
def test_runtime_mapping_rejects_untyped_or_nonfinite_values(tmp_path, monkeypatch, overrides):
    fixture = _prepared_fixture(tmp_path, "era5", physics_profile=runner.RUC_PHYSICS_PROFILE)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    with pytest.raises(ValueError, match="declared type"):
        _preflight(fixture, overrides)


def test_runtime_mapping_keeps_the_original_file_byte_binding(tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "era5", physics_profile=runner.RUC_PHYSICS_PROFILE)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    fixture.experiment.write_text(fixture.experiment.read_text() + "\n# changed bytes\n")
    with pytest.raises(ValueError, match="SHA|receipt|authority|manifest|bytes"):
        _preflight(fixture, {"rdlai2d": True})
