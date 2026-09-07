"""Actual absorption, selected spectra, per-tile twins and setup identity."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import cupy as cp

from test_rrtmg_legacy_wiring import profile, env, START
from gpuwm.core.radiation_composition import make_radiation, radiation_adapters
from gpuwm.core.streaming import _tile_scheme
from gpuwm.io.restart import _radiation_setup_identity

pytestmark = pytest.mark.gpu
F = np.float32
OUTPUTS = ("rthratenlw", "rthratensw", "glw", "olr", "swdown", "gsw", "coszen")


def _arguments(env, lw, sw, variant):
    cfg = SimpleNamespace(**(vars(env.cfg) | dict(ra_physics=0, ra_lw_physics=lw,
        ra_sw_physics=sw, ra_rrtmg_variant=variant, o3input=2, use_mp_re=1,
        swrad_scat=1., dt=10., wrf_rrtmg_compatibility="none")))
    state = SimpleNamespace(**(vars(env.state) | {
        name: cp.full_like(env.state.qv, radius)
        for name, radius in (("effc",10.),("effi",30.),("effs",50.))}))
    fields = dict(env.fields, glw=cp.full((env.ny,env.nx),300.,cp.float32))
    atmosphere = dict(env.atmosphere, theta=env.atmosphere["temperature"]/env.atmosphere["exner"])
    return cfg, dict(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)


def _make(env, cfg, gases):
    return make_radiation(cfg, START, env.lat.reshape(env.ny,env.nx),
        env.lon.reshape(env.ny,env.nx), p_top=env.p_top,
        column_chunk=16, trace_gas_overrides=gases)


def _host(result):
    return {name: cp.asnumpy(getattr(result,name)) for name in OUTPUTS
            if getattr(result,name) is not None}


def _equal(got, want):
    assert got.keys() == want.keys()
    for key in got:
        assert got[key].dtype == want[key].dtype
        assert np.isfinite(got[key]).all(), key
        assert got[key].tobytes() == want[key].tobytes(), key


@pytest.mark.parametrize("lw,sw,variant", [(1,1,"rte-rrtmgp"),
    (4,4,"rrtmg_legacy"),(4,1,"rrtmg_legacy"),(1,4,"rrtmg_legacy"),
    (4,4,"rte-rrtmgp"),(1,4,"rte-rrtmgp")])
def test_override_changes_absorption_survives_tile_and_binds_restart(env,lw,sw,variant):
    cfg, args = _arguments(env,lw,sw,variant)
    baseline = _make(env,cfg,None)
    before = _host(baseline(**args))
    empty = _make(env,cfg,{})
    _equal(_host(empty(**args)),before)
    adapter = _make(env,cfg,{"co2":800e-6})
    after = _host(adapter(**args))
    assert np.max(np.abs(after["glw"]-before["glw"])) > .01
    setup = lambda obj: _radiation_setup_identity(SimpleNamespace(radiation_callable=obj),cfg)
    assert setup(adapter) != setup(baseline), "a changed gas must invalidate setup identity"
    twin = _tile_scheme(adapter, adapter.latitude_deg, adapter.longitude_deg)
    assert setup(twin) == setup(adapter)
    _equal(_host(twin(**args)),after)
    for leaf in radiation_adapters(twin):
        if hasattr(leaf,"trace_gas_overrides"):
            assert leaf.trace_gas_overrides == {"co2":800e-6}


@pytest.mark.parametrize("lw,sw,gas,value", [
    (1,0,"co2",800e-6),(1,0,"n2o",800e-9),(1,0,"ch4",4e-6),
    (4,0,"co2",800e-6),(4,0,"n2o",800e-9),(4,0,"ch4",4e-6),
    (4,0,"o2",.3),(4,0,"cfc11",2e-9),(4,0,"cfc12",2e-9),
    (4,0,"cfc22",2e-9),(4,0,"ccl4",2e-9),
    (0,4,"co2",800e-6),(0,4,"ch4",4e-6),(0,4,"o2",.3)])
def test_each_claimed_gas_reaches_an_active_coefficient_consumer(env,lw,sw,gas,value):
    cfg,args = _arguments(env,lw,sw,"rrtmg_legacy")
    before = _host(_make(env,cfg,None)(**args))
    after = _host(_make(env,cfg,{gas:value})(**args))
    names = ("rthratenlw","glw","olr") if lw else ("rthratensw","swdown","gsw")
    assert any(np.any(after[k] != before[k]) for k in names), (lw,sw,gas)
    assert all(np.isfinite(v).all() for v in after.values())


def test_legacy_sw_n2o_is_a_retained_but_inactive_interface_operand(env):
    cfg,args = _arguments(env,0,4,"rrtmg_legacy")
    before = _host(_make(env,cfg,None)(**args))
    adapter = _make(env,cfg,{"n2o":800e-9})
    _equal(_host(adapter(**args)),before)
    assert radiation_adapters(adapter)[0].trace_gas_overrides == {"n2o":800e-9}


def test_cfc_is_lw_only_and_mixed_result_matches_independent_selected_engines(env):
    cfg,args = _arguments(env,4,1,"rrtmg_legacy")
    mixed = _make(env,cfg,{"co2":800e-6,"cfc11":2e-9})
    result = _host(mixed(**args))
    lwcfg = SimpleNamespace(**(vars(cfg)|dict(ra_sw_physics=0)))
    reference = _host(_make(env,lwcfg,{"co2":800e-6,"cfc11":2e-9})(**(args|dict(cfg=lwcfg))))
    for name in ("rthratenlw","glw","olr"):
        assert result[name].tobytes() == reference[name].tobytes()
    # Direct SW-only construction must identify an unused CFC operand.
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
    with pytest.raises(ValueError, match="no absorption operand.*cfc11"):
        RRTMGLegacyRadiation(START, mixed.latitude_deg,mixed.longitude_deg,
            longwave=False,trace_gas_overrides={"cfc11":2e-9})


def test_classic_default_identity_unchanged_and_declared_identity_owned():
    from dataclasses import replace
    from gpuwm.core.rrtm_lw import RRTMLongwaveRadiation
    geography = np.zeros((2,3), F)
    old = RRTMLongwaveRadiation(datetime(2021,1,1), geography, geography)
    declared = {"co2": 6e-4}
    new = replace(old, trace_gas_overrides=declared)
    assert "trace_gas_overrides" not in old.restart_identity
    assert new.restart_identity == {**old.restart_identity, "trace_gas_overrides": declared}
    declared["co2"] = 8e-4
    assert new.restart_identity["trace_gas_overrides"] == {"co2": 6e-4}
    assert replace(new).restart_identity == new.restart_identity


def test_stock_classic_pair_has_complete_canonical_setup_and_strict_gas_mismatch(monkeypatch):
    from types import SimpleNamespace
    from dataclasses import replace
    from test_radiation_composition import _cfg
    from gpuwm.core.rrtm_lw import RRTMDudhiaRadiation
    from gpuwm.io import restart
    cfg = _cfg(1,1)
    geography = np.zeros((2,3),F)
    old = RRTMDudhiaRadiation(datetime(2021,1,1),geography,geography)
    new = replace(old,trace_gas_overrides={"co2":8e-4})
    identity = lambda obj: restart._radiation_setup_identity(SimpleNamespace(radiation_callable=obj),cfg)
    before, after = identity(old), identity(new)
    assert before["callable"]["implementation"] == "stock"
    assert before["above_atmosphere_policies"]["lw"] == restart.LONGWAVE_ABOVE_ATMOSPHERE_POLICIES[1]
    assert before["classic"]["implementation"] == old.restart_identity
    assert before != after
    assert identity(replace(old,start_time=datetime(2022,1,1))) != before
    # Exercise the real stored-identity validation/compare boundary. Only the
    # outer physics inventory is supplied here; both radiation identities above
    # are actual adapters, and an unchanged self-consistent header succeeds.
    stored = dict(schema_version=restart.PHYSICS_SETUP_SCHEMA_VERSION, radiation=before)
    header = dict(physics_setup=stored,physics_setup_fingerprint=restart._json_sha256(stored))
    monkeypatch.setattr(restart,"physics_setup_identity",lambda state,cfg: stored)
    restart._require_physics_setup_match(header,None,cfg,"checkpoint")
    monkeypatch.setattr(restart,"physics_setup_identity",lambda state,cfg: dict(stored,radiation=after))
    with pytest.raises(restart.RestartMismatchError,match="physics setup"):
        restart._require_physics_setup_match(header,None,cfg,"checkpoint")
