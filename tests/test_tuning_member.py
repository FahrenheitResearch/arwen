"""Tuning adapter admission and actual scope/receipt behavior."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from gpuwm import tuning_member as tm


def member():
    return {"id": "m001", "seed": 47, "product": "research",
            "physics": {"sf_surface_physics": 3, "num_soil_layers": 6,
                        "bl_mynn_mixlength": 1, "sf_urban_physics": 0},
            "surface": {"soil_moisture_scale": 1.0, "sst_offset_k": 0.0},
            "urban": {"fraction_scale": 1.0, "heat_capacity_scale": 1.0, "conductivity_scale": 1.0}}


def inputs(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("# hash-bound input fixture\n", encoding="utf-8")
    root = tmp_path / "prepared"
    root.mkdir()
    run = SimpleNamespace(**member()["physics"], bl_pbl_physics=5)
    return SimpleNamespace(prepared_root=root, experiment_config=config,
                           experiment=SimpleNamespace(domains=(SimpleNamespace(grid_id=1, run=run),), physics_params=None),
                           physics_profile_assertion=None)


def test_invalid_or_inactive_controls_cannot_be_labelled_as_members():
    cases = []
    changed = member(); changed.pop("product"); cases.append(changed)
    changed = member(); changed["product"] = "hrrr-exact"; cases.append(changed)
    changed = member(); changed["surface"]["soil_moisture_scale"] = 0.0; cases.append(changed)
    changed = member(); changed["surface"]["sst_offset_k"] = float("nan"); cases.append(changed)
    changed = member(); changed["urban"]["heat_capacity_scale"] = 1.1; cases.append(changed)
    changed = member(); changed["physics"]["sf_urban_physics"] = 1; cases.append(changed)
    changed = member(); changed["seed"] = True; cases.append(changed)
    changed = member(); changed["physics"]["sf_surface_physics"] = 2; cases.append(changed)
    for case in cases:
        with pytest.raises(tm.TuningMemberError):
            tm.validate_member(case)


def test_validation_detaches_and_records_exact_fp32_values():
    original = member()
    original["surface"]["soil_moisture_scale"] = 0.65
    normalized = tm.validate_member(original)
    original["surface"]["soil_moisture_scale"] = 0.8
    assert normalized["surface"]["soil_moisture_scale"] == 0.6499999761581421
    assert tm.initialization_callback(member(), record=lambda value: None) is None


def test_control_sanitizes_inherited_parameter_set_and_restores_it(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWM_PHYSICS_PARAMS", "inherited-setting.toml")
    with tm.parameter_environment(member()):
        assert "GPUWM_PHYSICS_PARAMS" not in tm.os.environ
    assert tm.os.environ["GPUWM_PHYSICS_PARAMS"] == "inherited-setting.toml"
    path = tmp_path / "member.toml"
    path.write_text("[physics_params]\nname='setting'\n", encoding="utf-8")
    changed = member(); changed["physics_params_file"] = str(path)
    with tm.parameter_environment(changed):
        assert tm.os.environ["GPUWM_PHYSICS_PARAMS"] == str(path.resolve())
    assert tm.os.environ["GPUWM_PHYSICS_PARAMS"] == "inherited-setting.toml"


def test_selector_changes_require_an_actual_new_preparation(tmp_path):
    prepared = inputs(tmp_path)
    before = prepared.experiment_config.read_bytes()
    changed = member(); changed["physics"]["bl_mynn_mixlength"] = 2
    with pytest.raises(tm.TuningMemberError, match="prepare this selector group"):
        tm.preflight_member(changed, prepared, check_dependencies=False)
    assert prepared.experiment.domains[0].run.bl_mynn_mixlength == 1
    assert prepared.experiment_config.read_bytes() == before
    prepared.physics_profile_assertion = {"profile": "source-exact"}
    with pytest.raises(tm.TuningMemberError, match="source-exact"):
        tm.preflight_member(member(), prepared, check_dependencies=False)


def test_temporary_urban_settings_cannot_emit_a_misidentified_checkpoint(tmp_path):
    prepared = inputs(tmp_path)
    changed = member()
    changed["physics"].update(sf_surface_physics=2, num_soil_layers=4, sf_urban_physics=1)
    changed["urban"]["heat_capacity_scale"] = 1.1
    prepared.experiment.restart_interval_s = 3600.0
    with pytest.raises(tm.TuningMemberError, match="not in checkpoint identity"):
        tm.preflight_member(changed, prepared, check_dependencies=False)


def test_missing_surface_interface_refuses_without_a_fallback(monkeypatch):
    changed = member(); changed["surface"]["sst_offset_k"] = 0.5
    original = tm.importlib.import_module
    def load(name):
        if name == "gpuwm.ensemble.surface_recipe":
            raise ImportError("interface not integrated")
        return original(name)
    monkeypatch.setattr(tm.importlib, "import_module", load)
    with pytest.raises(tm.TuningMemberError, match="missing gpuwm.ensemble.surface_recipe"):
        tm.dependency_capabilities(changed)


def test_runner_failure_keeps_failure_receipt_and_restores_output_scope(monkeypatch, tmp_path):
    from gpuwm.ensemble.runtime_context import current_capture
    prepared = inputs(tmp_path)
    monkeypatch.setattr(tm, "dependency_capabilities", lambda value: {})
    monkeypatch.setattr(tm, "_CLAIMED_MEMBER", None)
    receipt = tmp_path / "receipt.json"
    def run(actual_inputs, *, output_directory, io_mode):
        assert actual_inputs is prepared
        assert io_mode == "history"
        assert current_capture().keep_member_files
        raise RuntimeError("forecast failure fixture")
    with pytest.raises(RuntimeError, match="forecast failure fixture"):
        tm.run_member(member(), prepared, output_directory=tmp_path / "out", receipt_path=receipt, runner=run)
    report = json.loads(receipt.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert "forecast failure fixture" in report["error"]
    assert current_capture() is None
    with pytest.raises(tm.TuningMemberError, match="fresh process"):
        tm.run_member(member(), prepared, output_directory=tmp_path / "second", receipt_path=receipt, runner=run)


def test_a_missing_initialization_call_cannot_produce_a_pass(monkeypatch, tmp_path):
    prepared = inputs(tmp_path)
    monkeypatch.setattr(tm, "dependency_capabilities", lambda value: {})
    monkeypatch.setattr(tm, "_CLAIMED_MEMBER", None)
    monkeypatch.setattr(tm, "initialization_callback", lambda value, record: lambda **kw: None)
    changed = member(); changed["surface"]["soil_moisture_scale"] = 0.8
    receipt = tmp_path / "receipt.json"
    with pytest.raises(tm.TuningMemberError, match="never acted"):
        tm.run_member(changed, prepared, output_directory=tmp_path / "out", receipt_path=receipt,
                      runner=lambda *args, **kwargs: {"status": "PASS"})
    assert json.loads(receipt.read_text())["status"] == "FAIL"


@pytest.mark.gpu
def test_actual_gpu_urban_adapter_owns_private_tables_and_fraction():
    """Uses the real pinned UCM loader and actual GPU transform, no forecast mock."""
    if tm.os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("GPU hidden by local lane policy")
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA card")
    from gpuwm.core import urban_ucm
    from gpuwm.core.urban_tables import load_urban_params
    params = load_urban_params(1)
    original = urban_ucm._packed(params)
    original_buffers = urban_ucm._device_params(original)
    original_sha = tm._device_sha(original_buffers[0])
    fraction = cp.asarray([0.2, 0.9, 0.7], dtype=cp.float32)
    urban = SimpleNamespace(option=1, params=params, frc_urb2d=fraction,
                            urban_mask=cp.asarray([1, 1, 0], dtype=cp.int32))
    state = SimpleNamespace(physics=SimpleNamespace(urban=urban), elapsed_seconds=0.0)
    changed = member()
    changed["physics"].update(sf_surface_physics=2, num_soil_layers=4, sf_urban_physics=1)
    changed["urban"] = {"fraction_scale": 1.25, "heat_capacity_scale": 1.25, "conductivity_scale": 0.75}
    changed = tm.validate_member(changed)
    receipt = tm._apply_urban(state, changed, 1)
    assert list(fraction.get()) == pytest.approx([0.25, 1.0, 0.7], abs=1e-7)
    assert urban.params is not params
    assert tm._device_sha(original_buffers[0]) == original_sha
    private = urban_ucm._packed(urban.params)
    private_table = urban_ucm._device_params(private)[0].get().reshape(original.table.shape)
    for column, name in enumerate(urban_ucm.UCM_TABLE_COLUMNS):
        factor = 1.25 if name in {"capr", "capb", "capg"} else 0.75 if name in {"aksr", "aksb", "aksg"} else 1.0
        assert private_table[:, column] == pytest.approx(original.table[:, column] * factor, rel=1e-7)
    assert receipt["before"]["table_sha256"] == original_sha
    assert tm._apply_urban(state, changed, 1) is receipt
    different = deepcopy(changed); different["id"] = "m002"
    with pytest.raises(tm.TuningMemberError, match="different tuning member"):
        tm._apply_urban(state, different, 1)
