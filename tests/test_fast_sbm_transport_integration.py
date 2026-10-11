"""The real transport selector exposes every spectral and number scalar."""
import numpy as np
from gpuwm.config import RunConfig
from gpuwm.core.state import DomainState
from gpuwm.core.moist import extra_moist_species,moist_species
from gpuwm.core.fast_sbm_contract import EXTRA_SPECIES


def test_spectral_transport_registry_uses_real_state():
    state=DomainState(RunConfig(nx=6,ny=5,nz=8,dx=1000.,dy=1000.,ztop=1000.,dt=10.,run_seconds=60.,moist=True,mp_physics=30),array_module=np)
    assert extra_moist_species(state)==EXTRA_SPECIES
    assert moist_species(state)==("qv","qc","qr")+EXTRA_SPECIES
    for name in EXTRA_SPECIES:
        assert getattr(state,name).shape==getattr(state,name+"0").shape


def test_existing_bulk_transport_registry_stays_unchanged():
    state=DomainState(RunConfig(nx=6,ny=5,nz=8,dx=1000.,dy=1000.,ztop=1000.,dt=10.,run_seconds=60.,moist=True,mp_physics=8),array_module=np)
    assert extra_moist_species(state)==("qi","qs","qg","nr","ni")
