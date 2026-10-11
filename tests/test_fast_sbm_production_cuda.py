"""Packaged production entry against the independent host replay path."""
import numpy as np
import pytest
cp=pytest.importorskip("cupy")
from gpuwm.config import RunConfig
from gpuwm.core.state import DomainState
from gpuwm.core.fast_sbm_cuda import launch
from tools.fast_sbm_host import host as H,replay as RP
from tools.fast_sbm_wrf480_oracle import fixtures as fx

@pytest.mark.parametrize("family",("kc1","kc2edge"))
def test_packaged_cuda_entry_matches_host(family):
    case=RP.Case(family,"o32")
    host=RP.host_for("o32",H.flags_corrected())
    owner=DomainState(RunConfig(nx=6,ny=5,nz=8,dx=1000.,dy=1000.,ztop=1000.,dt=12.,run_seconds=24.,moist=True,mp_physics=30))
    for step,pre,_post in list(fx.iter_replay(f"{family}_replay_o32"))[:2]:
        state=case.state(pre.fields)
        want={name:a.copy(order="F") for name,a in state.items()}
        status,_=host.call(step,12.,1000.,1000.,case.tile.dims(),want)
        assert status==0
        fields={name:cp.array(a,order="F") for name,a in state.items()}
        launch(step,12.,1000.,1000.,case.tile.dims(),fields,owner.scratch)
        for name in H.STATE_FIELDS:
            actual=cp.asnumpy(fields[name]).ravel(order="F").view(np.uint32)
            expected=want[name].ravel(order="F").view(np.uint32)
            assert np.array_equal(actual,expected),(family,step,name)


def test_bulk_only_input_cannot_silently_lose_condensate():
    """A bulk-only start (HRRR, RAP) used to be refused here; the bulk-to-bin
    start (gpuwm/core/fast_sbm_bulk_start.py) retired that refusal.  The
    property it guarded still holds and is what this measures: the bulk
    condensate lands in the bins mass-exactly, no bulk field changes, and a
    cell that already holds bins is never refilled."""
    from gpuwm.core.fast_sbm_bulk_start import start_bins_from_bulk
    cfg=RunConfig(nx=6,ny=5,nz=8,dx=1000.,dy=1000.,ztop=1000.,dt=12.,run_seconds=24.,moist=True,mp_physics=30)
    state=DomainState(cfg)
    state.alt.fill(np.float32(1.0))
    state.qc.fill(np.float32(0.001))
    before=cp.asnumpy(state.qc)
    xland=cp.ones((cfg.ny,cfg.nx),dtype=cp.float32)
    receipt=start_bins_from_bulk(state,xland)
    assert receipt["cells_filled"]==before.size
    liquid=cp.asnumpy(state.fast_sbm_bins[:33].astype(cp.float64).sum(axis=0))
    np.testing.assert_allclose(liquid,before.astype(np.float64),rtol=1e-6,atol=0)
    assert np.array_equal(cp.asnumpy(state.qc),before)
    assert start_bins_from_bulk(state,xland)["cells_filled"]==0


def test_a_start_without_xland_is_refused_before_its_water_changes():
    from gpuwm.core.fast_sbm_runtime import apply
    cfg=RunConfig(nx=6,ny=5,nz=8,dx=1000.,dy=1000.,ztop=1000.,dt=12.,run_seconds=24.,moist=True,mp_physics=30)
    state=DomainState(cfg)
    state.qc.fill(np.float32(0.001))
    before=cp.asnumpy(state.qc)
    bins=cp.asnumpy(state.fast_sbm_bins)
    with pytest.raises(ValueError,match="xland"):
        apply(state,cfg,12.)
    assert np.array_equal(cp.asnumpy(state.qc),before)
    assert np.array_equal(cp.asnumpy(state.fast_sbm_bins),bins)
