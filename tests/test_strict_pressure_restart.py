"""The strict pressure carrier survives digest and checkpoint operations."""
import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import state as state_module
from gpuwm.io import restart
from gpuwm import wrf_exact


def config():
    return RunConfig(nx=6,ny=4,nz=5,dx=2000.0,dy=2000.0,ztop=8000.0,dt=10.0,run_seconds=0.0)


def test_strict_pressure_roundtrips_without_reconstruction(tmp_path,monkeypatch):
    monkeypatch.setattr(state_module,"DIAGNOSTICS_ENABLED",True)
    original=state_module.DomainState(config(),array_module=np)
    pressure=original.p_perturbation
    pressure.flat[:4]=np.array([-0.0,1.0e-42,-12.345,0.03125],dtype=np.float32)
    assert restart.classify_state_attr("p_perturbation")=="serialize"
    assert "state/p_perturbation" in restart.state_manifest(original)
    path=restart.write_restart(tmp_path/"state.npz",original,config())
    fresh=state_module.DomainState(config(),array_module=np)
    restart.restore_restart(path,fresh,config())
    np.testing.assert_array_equal(fresh.p_perturbation.view(np.uint32),pressure.view(np.uint32))


def test_ordinary_state_has_no_extra_restart_payload(monkeypatch):
    monkeypatch.setattr(state_module,"DIAGNOSTICS_ENABLED",False)
    state=state_module.DomainState(config(),array_module=np)
    assert state.p_perturbation is None
    assert "state/p_perturbation" not in restart.state_manifest(state)


def test_strict_inventory_and_batch_extension_do_not_duplicate_the_carrier(monkeypatch):
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.ensemble.batch_diagnostics import diagnostics_specs
    monkeypatch.setattr(wrf_exact,"DIAGNOSTICS_ENABLED",True)
    assert state_array_shapes(config())["p_perturbation"]==(5,4,6)
    assert diagnostics_specs(config(),strict=True)==()
    monkeypatch.setattr(wrf_exact,"DIAGNOSTICS_ENABLED",False)
    assert "p_perturbation" not in state_array_shapes(config())
    assert len(diagnostics_specs(config(),strict=True))==1
