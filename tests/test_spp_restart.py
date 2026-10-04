"""SPP driver bindings do not become a second owner of provider state."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from gpuwm.io import restart
from test_restart import _cfg, _fill_setup, _shim_driver_state


SPP_BINDING_ATTRS = {"_spp_flags", "_spp_shapes", "spp_patterns"}


@pytest.mark.parametrize("bound", [False, True])
def test_spp_bindings_do_not_expand_checkpoint_or_carrier_arrays(monkeypatch, bound):
    from tilestream.physics_inventory import carrier_manifest

    state, driver = _shim_driver_state(_cfg(), monkeypatch)
    # The baseline is the same driver before SPP binding attributes existed.
    bindings = {name: vars(driver).pop(name) for name in SPP_BINDING_ATTRS}
    before = (restart._driver_manifest(driver), carrier_manifest(state))
    vars(driver).update(bindings)
    if bound:
        # Binding validation is covered by the CUDA consumer tests. Here the
        # manifest must ignore even populated provider-owned numerical views.
        driver._spp_flags = {name: 1 for name in driver._spp_flags}
        driver.spp_patterns = {
            name: np.ones(shape, dtype=np.float32)
            for name, shape in driver._spp_shapes.items()}
    after = (restart._driver_manifest(driver), carrier_manifest(state))
    assert SPP_BINDING_ATTRS <= restart.DRIVER_REBUILT_ATTRS
    for old, new in zip(before, after):
        assert old.keys() == new.keys()
        assert all(old[key] is new[key] for key in old)


def test_default_spp_off_driver_restart_roundtrip(monkeypatch, tmp_path):
    cfg = _cfg()
    source, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    driver.rthratenlw[...] = np.float32(0.125)
    path = restart.write_restart(tmp_path / "spp-off.npz", source, cfg)
    target, target_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(target)
    restart.restore_restart(path, target, cfg)
    original = restart._driver_manifest(driver)
    restored = restart._driver_manifest(target_driver)
    assert original.keys() == restored.keys()
    for key in original:
        assert original[key].tobytes() == restored[key].tobytes(), key
    assert target_driver.spp_patterns == {}
    assert target_driver._spp_flags == {"conv": 0, "pbl": 0, "lsm": 0}
    with np.load(path, allow_pickle=False) as archive:
        assert not any("spp" in key for key in archive.files)


def test_optional_ruc_field_sf_is_a_normal_fields_carrier(monkeypatch, tmp_path):
    from tilestream.physics_inventory import carrier_manifest

    cfg = _cfg()
    source, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    # An optional runtime output belongs to fields, not the stochastic
    # provider. The default driver does not allocate this diagnostic.
    assert "field_sf" not in driver.fields
    diagnostic = np.arange(9 * cfg.ny * cfg.nx, dtype=np.float32).reshape(
        9, cfg.ny, cfg.nx)
    driver.fields["field_sf"] = diagnostic
    assert restart._driver_manifest(driver)["fields/field_sf"] is diagnostic
    assert carrier_manifest(source)["fields/field_sf"] is diagnostic
    path = restart.write_restart(tmp_path / "ruc-diagnostic.npz", source, cfg)
    target, target_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(target)
    target_driver.fields["field_sf"] = np.zeros_like(diagnostic)
    restart.restore_restart(path, target, cfg)
    assert target_driver.fields["field_sf"].tobytes() == diagnostic.tobytes()


@pytest.mark.parametrize("name", ["spp_conv", "spp_pbl", "spp_lsm"])
def test_enabled_spp_stays_bound_in_all_checkpoint_identities(monkeypatch, name):
    from gpuwm.core.model import restart_identity_payload
    from gpuwm import experiment
    from test_nest_spawn_init import _experiment

    cfg = _cfg()
    current = asdict(cfg)
    previous = {key: value for key, value in current.items()
                if key not in ("spp_conv", "spp_pbl")}
    assert restart._configuration_digest_values(current) == (
        restart._configuration_digest_values(previous))
    enabled = replace(cfg, **{name: 1})
    assert restart.configuration_echo(enabled)[name] == 1
    assert restart._configuration_digest_values(asdict(enabled))[name] == 1
    assert restart._configuration_fingerprint(cfg) != (
        restart._configuration_fingerprint(enabled))

    exp = _experiment()
    base = restart_identity_payload(exp)
    live_document = experiment.experiment_config_document

    def old_document(value):
        result = live_document(value)
        for domain in result["domains"]:
            for key in ("spp_conv", "spp_pbl"):
                domain["run"].pop(key, None)
        return result

    monkeypatch.setattr(experiment, "experiment_config_document", old_document)
    assert restart_identity_payload(exp) == base
    monkeypatch.setattr(experiment, "experiment_config_document", live_document)
    changed_root = replace(exp.root, run=replace(exp.root.run, **{name: 1}))
    changed = replace(exp, domains=(changed_root,) + exp.domains[1:])
    assert restart_identity_payload(changed) != base
