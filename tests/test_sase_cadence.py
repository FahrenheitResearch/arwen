"""Held vertical momentum follows the ordinary PBL cadence and carrier routes."""
from dataclasses import replace

import numpy as np
import pytest

from test_sase import _sase_shim_driver


@pytest.mark.parametrize('radiation', [False, True])
def test_positive_pbl_cadence_holds_w_through_composition_and_mass_change(monkeypatch, radiation):
    calls = []
    ph, state, cfg, driver = _sase_shim_driver(monkeypatch, calls,
        bldt=0.1, dt=1.0, radt_minutes=0.0,
        mix_rates=(3.e-5, 1.e-7, -1.e-8, 2.e-9)*2,
        ra_lw_physics=4 if radiation else 0,
        ra_sw_physics=4 if radiation else 0)
    state.alt[...] = np.float32(1.0)
    from gpuwm.config import validate_run_config
    from gpuwm.io import restart
    validate_run_config(cfg)
    initial_keys = set(restart._driver_manifest(driver))
    first = driver.compute(state, cfg).rw.copy()
    assert first.shape == state.w.shape and first[1:-1].any()
    assert 'driver/pbl_tendencies/rw' in initial_keys
    assert 'pbl/dw' in initial_keys
    assert set(restart._driver_manifest(driver)) == initial_keys
    # An off-cadence read must keep the historically coupled acceleration;
    # recoupling with the newer mass would produce a different answer.
    state.mup[...] += np.float32(100.0)
    for time in (1.0, 2.0, 3.0, 4.0):
        state.elapsed_seconds = time
        actual = driver.compute(state, cfg)
        np.testing.assert_array_equal(actual.rw, first)
        state.rw_t[...] = 0.0
        actual.add_to_slow(state)
        np.testing.assert_array_equal(state.rw_t, first)
    assert calls.count('sase_step') == 1
    assert driver.tendencies.rw is driver.pbl_tendencies.rw
    state.elapsed_seconds = 5.0
    driver.compute(state, cfg)
    assert calls.count('sase_step') == 2


def test_positive_cadence_moves_raw_w_and_recouples_on_new_geometry(monkeypatch):
    from gpuwm.core import physics_continuation as continuation
    ph, state, cfg, driver = _sase_shim_driver(monkeypatch, [], bldt=0.1)
    state.alt[...] = np.float32(1.0)
    driver.compute(state, cfg)
    captured = continuation.capture_continuation(state, driver)
    assert captured['pbl/dw'].any()
    class OneColumnMove:
        def window(self, shape):
            return ((slice(None), slice(None)), (slice(0,-1), slice(1,None)))
    shifted = continuation.shift_continuation(captured, OneColumnMove())
    state.mup[...] += np.float32(100.0)
    continuation.restore_continuation(state, driver, shifted)
    driver.recouple_after_relocation(state, cfg)
    expected = ph.couple_sase_w_tendency(state, cfg, shifted['pbl/dw'])
    np.testing.assert_array_equal(driver.pbl_tendencies.rw, expected)
    assert not driver.pbl_tendencies.rw[..., -1].any()


def _restart_fixture(monkeypatch, *, cadence=0.1, diagnostics=False, flux=None):
    ph, state, cfg, driver = _sase_shim_driver(monkeypatch, [],
        bldt=cadence, dt=1.0, run_seconds=20.0,
        sase_flux_diag=diagnostics if flux is None else flux, hmix_k_diag=diagnostics,
        mix_rates=(3.e-5, 1.e-7, -1.e-8, 2.e-9)*3)
    state.alt[...] = np.float32(1.0)
    return ph, state, cfg, driver


def test_real_restart_archive_restores_held_w_before_an_off_cadence_step(monkeypatch, tmp_path):
    from gpuwm.io import restart
    ph, straight, cfg, straight_driver = _restart_fixture(monkeypatch, diagnostics=True)
    straight_driver.compute(straight, cfg)
    for number, value in enumerate(restart.pbl_diagnostic_manifest(straight_driver).values(), 1):
        value[...] = np.float32(number)
    straight.elapsed_seconds = 2.0
    saved = restart.write_restart(tmp_path/'held.npz', straight, cfg)
    _, resumed, _, resumed_driver = _restart_fixture(monkeypatch, diagnostics=True)
    restart.restore_restart(saved, resumed, cfg)
    expected = restart._driver_manifest(straight_driver)
    actual = restart._driver_manifest(resumed_driver)
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    # The closure is forbidden in this step; its held tendency must still
    # reach the RK slot identically on both the uninterrupted and resumed leg.
    monkeypatch.setattr(ph.PhysicsDriver, '_run_sase',
        lambda *a: pytest.fail('off-cadence SASE recomputed after restore'))
    for state, driver in ((straight,straight_driver),(resumed,resumed_driver)):
        state.rw_t[...] = 0.0
        driver.compute(state,cfg).add_to_slow(state)
    np.testing.assert_array_equal(resumed.rw_t, straight.rw_t)
    assert resumed.rw_t[1:-1].any()


@pytest.mark.parametrize('defect', ['missing','shape','raw'])
def test_incomplete_held_w_restart_refuses_before_mutating_state(monkeypatch,tmp_path,defect):
    from gpuwm.io import restart
    from test_restart import _rewrite_restart_archive
    _, source, cfg, driver = _restart_fixture(monkeypatch)
    driver.compute(source,cfg)
    source.elapsed_seconds = 2.0
    path=restart.write_restart(tmp_path/'source.npz',source,cfg)
    def damage(payload,header):
        key='pbl/dw' if defect=='raw' else 'driver/pbl_tendencies/rw'
        if defect=='shape':
            payload[key]=payload[key][:-1]
        else:
            del payload[key]
    bad=_rewrite_restart_archive(path,tmp_path/'bad.npz',damage)
    _, target, _, _ = _restart_fixture(monkeypatch)
    target.w[...] = np.float32(17.)
    before=target.w.tobytes()
    with pytest.raises(restart.RestartMismatchError,match='(rw|dw)'):
        restart.restore_restart(bad,target,cfg)
    assert target.w.tobytes()==before


def test_legacy_every_step_restart_can_rebuild_previously_unstored_w(monkeypatch,tmp_path):
    from gpuwm.io import restart
    from test_restart import _rewrite_restart_archive
    _, source, cfg, driver = _restart_fixture(monkeypatch,cadence=0.0)
    driver.compute(source,cfg)
    source.elapsed_seconds=2.0
    path=restart.write_restart(tmp_path/'current.npz',source,cfg)
    old=_rewrite_restart_archive(path,tmp_path/'old.npz',
        lambda payload,header: payload.pop('driver/pbl_tendencies/rw'))
    _, target, _, target_driver = _restart_fixture(monkeypatch,cadence=0.0)
    restart.restore_restart(old,target,cfg)
    expected=driver.compute(source,cfg).rw
    actual=target_driver.compute(target,cfg).rw
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('enabled', [False,True])
def test_output_only_flux_toggle_preserves_restart_state(monkeypatch,tmp_path,enabled):
    from gpuwm.io import restart
    _,source,cfg,driver=_restart_fixture(monkeypatch,diagnostics=True,flux=not enabled)
    driver.compute(source,cfg)
    source.elapsed_seconds=2.
    path=restart.write_restart(tmp_path/'source.npz',source,cfg)
    _,target,target_cfg,target_driver=_restart_fixture(monkeypatch,diagnostics=True,flux=enabled)
    restart.restore_restart(path,target,target_cfg)
    np.testing.assert_array_equal(target.w,source.w)
    np.testing.assert_array_equal(target_driver.pbl_tendencies.rw,driver.pbl_tendencies.rw)
    if enabled:
        assert all(not value.any() for value in target_driver.sase_flux_diag.values())


def test_missing_or_unknown_held_diagnostic_is_not_silently_dropped(monkeypatch,tmp_path):
    from gpuwm.io import restart
    from test_restart import _rewrite_restart_archive
    _,source,cfg,driver=_restart_fixture(monkeypatch,diagnostics=True)
    driver.compute(source,cfg)
    source.elapsed_seconds=2.
    path=restart.write_restart(tmp_path/'source.npz',source,cfg)
    for unknown in (False,True):
        def edit(payload,header):
            key='pbl/diagnostics/sase_flux_diag/fqv_vent'
            value=payload.pop(key)
            if unknown:payload[key+'-unknown']=value
        bad=_rewrite_restart_archive(path,tmp_path/f'bad-{unknown}.npz',edit)
        _,target,_,_=_restart_fixture(monkeypatch,diagnostics=True)
        before=target.w.tobytes()
        with pytest.raises(restart.RestartMismatchError,match='PBL diagnostic'):
            restart.restore_restart(bad,target,cfg)
        assert target.w.tobytes()==before


def test_legacy_absent_ice_placeholder_has_no_state_consumer(monkeypatch,tmp_path):
    from gpuwm.io import restart
    from test_restart import _rewrite_restart_archive
    _,source,cfg,driver=_restart_fixture(monkeypatch,cadence=0.)
    driver.compute(source,cfg)
    source.elapsed_seconds=2.
    path=restart.write_restart(tmp_path/'new.npz',source,cfg)
    def old_inventory(payload,header):
        payload.pop('driver/pbl_tendencies/rw')
        payload['driver/pbl_tendencies/rqi']=np.full(source.p.shape,2.e-9,np.float32)
    old=_rewrite_restart_archive(path,tmp_path/'old.npz',old_inventory)
    _,target,_,target_driver=_restart_fixture(monkeypatch,cadence=0.)
    restart.restore_restart(old,target,cfg)
    assert target_driver.pbl_tendencies.rqi is None
    np.testing.assert_array_equal(driver.compute(source,cfg).rw,target_driver.compute(target,cfg).rw)


def test_held_w_and_raw_relocation_acceleration_are_priced(monkeypatch):
    from gpuwm.core.preflight import physics_array_shapes
    from gpuwm.core.physics_inventory import physics_retains_ysu_output
    _,state,cfg,driver=_restart_fixture(monkeypatch)
    shapes=physics_array_shapes(cfg)
    assert shapes['pbl_tendencies/rw']==state.w.shape
    assert shapes['pbl_raw_rates/dw']==state.p.shape
    assert not physics_retains_ysu_output(cfg)
    assert not any(name.startswith('last_ysu/') for name in shapes)
    assert 'tendencies/rw' not in shapes  # composition aliases the held field
    default=physics_array_shapes(replace(cfg,bldt=0.))
    assert default['pbl_tendencies/rw']==state.w.shape
    assert 'pbl_raw_rates/dw' not in default
