"""Raw PBL forcing has one lifetime across resident and streamed execution."""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.io import restart
from test_restart import (
    _cfg, _fill_setup, _rewrite_restart_archive, _shim_driver_state as _unbound_state,
)


def _shim_driver_state(cfg, monkeypatch):
    from gpuwm.core.gf import GrellFreitas
    from gpuwm.core.ntiedtke import NewTiedtke

    if cfg.sf_sfclay_physics == 5:
        from gpuwm.core import physics
        from test_restart import _NumpyCupyShim, _shim_state
        state = _shim_state(cfg, monkeypatch)
        monkeypatch.setattr(physics, "cp", _NumpyCupyShim)
        fields = {name: np.zeros(state.mup.shape, np.float32)
                  for name in physics.MYNN_SURFACE_OUTPUTS}
        driver = physics.PhysicsDriver(
            state, cfg, fields=fields, sfclay_result=None, noah_params=None)
        state.physics = driver
    else:
        state, driver = _unbound_state(cfg, monkeypatch)
    if cfg.cu_physics in (3, 16):
        driver.cumulus_callable = (GrellFreitas() if cfg.cu_physics == 3
                                   else NewTiedtke())
        driver.cumulus_callable.bind_driver(driver)
    return state, driver


@pytest.mark.parametrize("cu", [0, 1, 3, 16])
def test_held_inventory_is_eager_shared_and_priced(monkeypatch, cu):
    from gpuwm.core.preflight import physics_array_shapes
    from tilestream.physics_inventory import carrier_manifest, checkpointed_carriers

    cfg = _cfg(moist=True, cu_physics=cu, bldt=2.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    names = restart.DRIVER_HELD_FORCING_ATTRS
    keys = {f"held/{name}" for name in names} if cu in (3, 16) else set()
    manifests = [restart._driver_manifest(driver), carrier_manifest(state)]
    for manifest in manifests:
        assert {key for key in manifest if key.startswith("held/")} == keys
        assert keys <= checkpointed_carriers(manifest).keys()
        for key in keys:
            assert manifest[key] is getattr(driver, key[5:])
            assert not manifest[key].any()
    shapes = physics_array_shapes(cfg)
    assert {name for name in names if name in shapes} == {k[5:] for k in keys}
    assert sum(getattr(driver, key[5:]).nbytes for key in keys) == (
        2 * cfg.nz * cfg.ny * cfg.nx * 4 if keys else 0)


@pytest.mark.parametrize("cu", [3, 16])
def test_pbl_producer_updates_owned_carriers_in_place(monkeypatch, cu):
    from gpuwm.core import physics
    from tilestream.physics_inventory import carrier_manifest

    cfg = _cfg(moist=True, cu_physics=cu, bldt=2.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    captured = carrier_manifest(state)
    monkeypatch.setattr(physics, "couple_ysu_tendencies", lambda *_: None)
    rates = {name: np.arange(state.p.size, dtype=np.float32).reshape(
        state.p.shape) * scale for name, scale in [("dtheta", 1e-4), ("dqv", 1e-7)]}
    for step in (1, 2):
        driver._couple_pbl_slot(cfg, rates)
        for name, source in [("gf_rthblten", "dtheta"), ("gf_rqvblten", "dqv")]:
            target = captured[f"held/{name}"]
            assert target is getattr(driver, name)
            assert not np.shares_memory(target, rates[source])
            np.testing.assert_array_equal(target, rates[source])
            rates[source] *= np.float32(2)
            assert not np.array_equal(target, rates[source])


@pytest.mark.parametrize("cu", [3, 16])
def test_resident_roundtrip_restores_both_held_rates_in_place(monkeypatch, tmp_path, cu):
    cfg = _cfg(moist=True, cu_physics=cu, bldt=2.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    for i, name in enumerate(sorted(restart.DRIVER_HELD_FORCING_ATTRS), 1):
        getattr(driver, name)[...] = np.arange(state.p.size, dtype=np.float32).reshape(
            state.p.shape) * np.float32(i * 1e-7)
    state.elapsed_seconds = 20.0
    path = restart.write_restart(tmp_path / "held.npz", state, cfg)
    resumed, resumed_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(resumed)
    identities = {name: getattr(resumed_driver, name) for name in restart.DRIVER_HELD_FORCING_ATTRS}
    restart.restore_restart(path, resumed, cfg)
    for name, target in identities.items():
        assert target is getattr(resumed_driver, name)
        np.testing.assert_array_equal(target, getattr(driver, name))


@pytest.mark.parametrize("defect", ["missing", "unknown", "shape", "dtype"])
def test_malformed_held_rates_refuse_before_mutation(monkeypatch, tmp_path, defect):
    cfg = _cfg(moist=True, cu_physics=3, bldt=2.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    state.thp.fill(12)
    path = restart.write_restart(tmp_path / "held.npz", state, cfg)
    key = "held/gf_rthblten"
    def change(payload, header):
        if defect == "missing":
            payload.pop(key)
        elif defect == "unknown":
            payload["held/not_a_rate"] = payload[key].copy()
        elif defect == "shape":
            payload[key] = payload[key][:-1]
        else:
            payload[key] = payload[key].astype(np.float64)
    bad = _rewrite_restart_archive(path, tmp_path / "bad.npz", change)
    resumed, resumed_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(resumed)
    before = {k: v.copy() for k, v in restart.state_manifest(resumed).items()}
    with pytest.raises(restart.RestartMismatchError, match="held/"):
        restart.restore_restart(bad, resumed, cfg)
    for key, target in restart.state_manifest(resumed).items():
        np.testing.assert_array_equal(target, before[key])
    assert not resumed_driver.gf_rthblten.any()


@pytest.mark.parametrize("transport", ["store", "stream"])
@pytest.mark.parametrize("defect", ["missing", "unknown", "shape", "dtype"])
def test_streamed_held_validation_is_complete_before_copy(
        monkeypatch, tmp_path, transport, defect):
    from tilestream import checkpoint, restart_stream
    from tilestream.physics_inventory import carrier_manifest

    cfg = _cfg(moist=True, cu_physics=3, bldt=2.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    state.thp.fill(12)
    path = restart.write_restart(tmp_path / "held.npz", state, cfg)
    key = "held/gf_rthblten"
    def change(payload, header):
        if defect == "missing":
            payload.pop(key)
        elif defect == "unknown":
            payload["held/not_a_rate"] = payload[key].copy()
        elif defect == "shape":
            payload[key] = payload[key][:-1]
        else:
            payload[key] = payload[key].astype(np.float64)
    bad = _rewrite_restart_archive(path, tmp_path / "bad.npz", change)
    store = {k: np.zeros_like(v) for k, v in carrier_manifest(state).items()}
    before = {k: v.copy() for k, v in store.items()}
    with pytest.raises((restart.RestartMismatchError, restart_stream.RestartRefused),
                       match="held/"):
        if transport == "store":
            checkpoint.read_store_restart(
                bad, store, checkpoint.DomainSetup.capture(state, cfg), cfg)
        else:
            restart_stream.read_streamed_restart(
                bad, store, cfg, setup=restart_stream.capture_domain_setup(state),
                template_state=state, scalars={})
    for key, target in store.items():
        np.testing.assert_array_equal(target, before[key])


@pytest.mark.parametrize("pbl,sfclay", [(1, 91), (2, 2), (5, 5), (11, 91)])
@pytest.mark.parametrize("cu", [0, 3])
def test_positive_cadence_pbl_raw_inventory_is_canonical(monkeypatch, pbl, sfclay, cu):
    from gpuwm.core import physics
    from gpuwm.core.preflight import physics_array_shapes
    from tilestream.physics_inventory import carrier_manifest

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=pbl,
               sf_sfclay_physics=sfclay, bldt=2., cu_physics=cu)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    assert set(driver.pbl_raw_rates) == {"du", "dv", "dtheta", "dqv", "dqc", "dqi"}
    manifest = restart.pbl_raw_manifest(driver)
    assert len(manifest) == 6
    assert len({id(value) for value in manifest.values()}) == 6
    assert all(carrier_manifest(state)[key] is value for key, value in manifest.items())
    if cu == 3:
        assert driver.pbl_raw_rates["dtheta"] is driver.gf_rthblten
        assert driver.pbl_raw_rates["dqv"] is driver.gf_rqvblten
        assert "pbl/dtheta" not in manifest and "pbl/dqv" not in manifest
    shapes = physics_array_shapes(cfg)
    priced = [key for key in shapes if key.startswith("pbl_raw_rates/")
              or key in restart.DRIVER_HELD_FORCING_ATTRS]
    assert len(priced) == 6
    monkeypatch.setattr(physics, "couple_ysu_tendencies", lambda *_: None)
    rates = {name: np.full(state.p.shape, (index + 1) * 1e-5, np.float32)
             for index, name in enumerate(driver.pbl_raw_rates)}
    driver._couple_pbl_slot(cfg, rates)
    for name, target in driver.pbl_raw_rates.items():
        np.testing.assert_array_equal(target, rates[name])


def test_all_raw_pbl_rates_roundtrip_after_restart(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1,
               sf_sfclay_physics=91, bldt=2., cu_physics=3)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    for index, value in enumerate(driver.pbl_raw_rates.values()):
        value[...] = np.arange(value.size, dtype=np.float32).reshape(value.shape) * (index + 1)
    path = restart.write_restart(tmp_path / "raw.npz", state, cfg)
    resumed, target_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(resumed)
    restart.restore_restart(path, resumed, cfg)
    for name, value in driver.pbl_raw_rates.items():
        np.testing.assert_array_equal(value, target_driver.pbl_raw_rates[name])
    assert target_driver.pbl_raw_rates["dtheta"] is target_driver.gf_rthblten


@pytest.mark.parametrize("transport", ["resident", "store", "stream"])
@pytest.mark.parametrize("cu", [0, 3])
def test_old_positive_cadence_checkpoint_refuses_before_copy(
        monkeypatch, tmp_path, transport, cu):
    from tilestream import checkpoint, restart_stream
    from tilestream.physics_inventory import carrier_manifest

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1,
               sf_sfclay_physics=91, bldt=2., cu_physics=cu)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    state.thp.fill(12)
    path = restart.write_restart(tmp_path / "new.npz", state, cfg)
    # A pre-repair checkpoint cannot reconstruct A-grid momentum rates
    # from face-averaged tendencies, even if its GF pair was already saved.
    def remove_old_missing_rates(payload, header):
        for key in list(payload):
            if key.startswith("pbl/"):
                payload.pop(key)
    old = _rewrite_restart_archive(path, tmp_path / "old.npz", remove_old_missing_rates)
    resumed, target_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(resumed)
    store = {key: np.zeros_like(value) for key, value in carrier_manifest(resumed).items()}
    before = {key: value.copy() for key, value in carrier_manifest(resumed).items()}
    with pytest.raises((restart.RestartMismatchError, restart_stream.RestartRefused),
                       match="pbl/"):
        if transport == "resident":
            restart.restore_restart(old, resumed, cfg)
        elif transport == "store":
            checkpoint.read_store_restart(
                old, store, checkpoint.DomainSetup.capture(resumed, cfg), cfg)
        else:
            restart_stream.read_streamed_restart(
                old, store, cfg, setup=restart_stream.capture_domain_setup(resumed),
                template_state=resumed, scalars={})
    for key, value in carrier_manifest(resumed).items():
        np.testing.assert_array_equal(value, before[key])
    assert all(not value.any() for value in store.values())
