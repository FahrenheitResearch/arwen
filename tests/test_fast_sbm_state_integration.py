"""Real state allocation, transport registry and serialization for mp30."""
import numpy as np
from gpuwm.config import RunConfig
from gpuwm.core.state import DomainState
from gpuwm.core.fast_sbm_contract import BIN_SPECIES,EXTRA_SPECIES
from gpuwm.state_serialization_contract import serialized_state_attrs


def test_real_state_carries_each_spectral_scalar_and_time_t_copy():
    cfg=RunConfig(nx=6,ny=5,nz=8,dx=1000.,dy=1000.,ztop=1000.,dt=10.,run_seconds=60.,moist=True,mp_physics=30)
    state=DomainState(cfg,array_module=np)
    names=serialized_state_attrs(state)
    assert len(BIN_SPECIES)==142
    for index,name in enumerate(BIN_SPECIES):
        a=getattr(state,name); old=getattr(state,name+"0")
        assert a.shape==(8,5,6) and a.dtype==np.float32
        assert old.shape==a.shape and not np.shares_memory(old,a)
        assert np.shares_memory(a,state.fast_sbm_bins[index])
        a.fill(index+1)
        assert name in names
    assert np.array_equal(state.fast_sbm_bins[:,0,0,0],np.arange(1,143))
    for name in ("th_old","qv_old"):
        assert getattr(state,name).shape==(8,5,6) and name in names


def test_other_scheme_keeps_its_original_transport_inventory():
    cfg=RunConfig(nx=6,ny=5,nz=8,dx=1000.,dy=1000.,ztop=1000.,dt=10.,run_seconds=60.,moist=True,mp_physics=8)
    state=DomainState(cfg,array_module=np)
    assert not any(getattr(state,name,None) is not None for name in BIN_SPECIES)
