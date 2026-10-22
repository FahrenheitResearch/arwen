"""Default selector and consumer-table admission for the real spectral state."""
from dataclasses import replace
import numpy as np
from gpuwm.config import RunConfig
from gpuwm.core import state as S,preflight as P
from gpuwm.io.restart import classify_state_attr,STATE_REBUILT_ATTRS
from gpuwm.core.fast_sbm_contract import BIN_SPECIES,EXTRA_SPECIES


def cfg():
    return RunConfig(nx=8,ny=6,nz=8,dx=1000.,dy=1000.,ztop=10000.,dt=1.,run_seconds=10.,moist=True,mp_physics=30)


def test_spectral_restart_inventory_and_real_workspace(monkeypatch):
    monkeypatch.setattr(S,"cp",np)
    from types import SimpleNamespace
    c=cfg(); ws=S.build_shared_dycore_state_workspace((SimpleNamespace(run=c),))
    state=S.DomainState(c,dycore_state_workspace=ws)
    declared=P.state_array_shapes(c)
    allocated={k:v for k,v in vars(state).items() if isinstance(v,np.ndarray)}
    infra=set()
    for name in allocated:
        if classify_state_attr(name)=="infra": infra.add(name)
    assert set(declared)==set(allocated)-infra
    for name in BIN_SPECIES:
        assert classify_state_attr(name)=="serialize"
        assert name+"0" in STATE_REBUILT_ATTRS
        assert np.shares_memory(getattr(state,name+"0"),ws.backing(name+"0"))
    slots=P.scratch_slot_registry(c)
    assert all(P.scratch_slot_lifetime(name) is not None for name in slots)


def test_native_input_and_export_keep_every_spectral_field(monkeypatch):
    from gpuwm.ingest.wrfinput import active_moisture_map,_restore_active_moisture
    from gpuwm.wrf_direct import _physics_contract_bundle,_load_contract
    from gpuwm.wrf_physics_inventory import stock_wrf_physics_inventory
    monkeypatch.setattr(S,"cp",np)
    c=cfg(); state=S.DomainState(c)
    names=active_moisture_map(c)
    assert set(BIN_SPECIES)<=set(names)
    raw={name:np.full(state.p.shape, np.float32((i+1)*1.e-8)) for i,name in enumerate(names)}
    _restore_active_moisture(state,raw,c,np)
    for name in BIN_SPECIES:
        assert np.array_equal(getattr(state,name),raw[name])
        assert np.array_equal(getattr(state,name+"0"),raw[name])
    declared={field.netcdf_name for field in stock_wrf_physics_inventory(30).wrfinput_fields}
    contract=_physics_contract_bundle(_load_contract(),30)
    exported={field["name"] for field in contract["wrfinput"]["variables"]}
    assert declared<=exported


def test_ice_mass_consumers_use_the_spectral_package():
    from gpuwm.core.rrtmg_legacy import legacy_cloud_fraction_flags
    from gpuwm.core.physics_inventory import PBL_RQI_MICROPHYSICS,pbl_raw_rate_names
    assert legacy_cloud_fraction_flags(30)==(True,True)
    from gpuwm.core.mynn_pbl_runtime import mynn_flag_qs
    assert mynn_flag_qs(30)
    assert 30 in PBL_RQI_MICROPHYSICS
    assert "dqi" in pbl_raw_rate_names(replace(cfg(),bl_pbl_physics=1,bldt=1.))


def test_native_number_units_follow_the_common_wrf_registry():
    from gpuwm.wrf_physics_inventory import stock_wrf_physics_inventory
    rows={field.netcdf_name:field for field in stock_wrf_physics_inventory(30).wrfinput_fields}
    for name in ("QNCCN","QNCLOUD","QNRAIN","QNSNOW","QNGRAUPEL"):
        assert rows[name].units=="# kg(-1)"
    assert rows["QNICE"].units=="# kg-1"
