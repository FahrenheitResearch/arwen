"""GF's stock set_tend carrier has coupled units between physics calls."""
from types import SimpleNamespace as NS
import numpy as np
import pytest


@pytest.mark.parametrize("strict", [False, True])
def test_gf_transport_carrier_keeps_stock_coupled_words(monkeypatch, strict):
    from gpuwm.core import dycore, moist
    monkeypatch.setattr(dycore,"WRF_EXACT",strict)
    monkeypatch.setattr(moist,"WRF_EXACT",strict)
    shape=(2,2,2)
    state=NS(cfg=NS(cu_physics=3),has_msf=True,
             rthften=np.zeros(shape,np.float32),rqvften=np.zeros(shape,np.float32),
             rth_t=np.full(shape,8,np.float32),c1h=np.ones(2,np.float32),
             c2h=np.zeros(2,np.float32),msft=np.full((2,2),2,np.float32),
             total_mu=lambda:np.full((2,2),4,np.float32))
    dycore.capture_advective_theta_forcing(state,cfg=state.cfg)
    moist._capture_advective_qv_forcing(state,np.full(shape,1.25,np.float32),state.total_mu(),cfg=state.cfg)
    np.testing.assert_array_equal(state.rthften,np.float32(16 if strict else 4))
    np.testing.assert_array_equal(state.rqvften,np.float32(2.5 if strict else .625))
