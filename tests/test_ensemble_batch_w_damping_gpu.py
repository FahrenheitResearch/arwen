"""An active limiter preserves every original per-member tendency word."""
from dataclasses import replace
import numpy as np
import pytest
from conftest import requires_gpu
pytestmark=[pytest.mark.gpu,requires_gpu]

@pytest.mark.parametrize('members',(1,4,10))
@pytest.mark.parametrize('terrain',(False,True))
@pytest.mark.parametrize('shared',(False,True))
def test_active_member_limiter_matches_original(members,terrain,shared):
    import cupy as cp
    from test_ensemble_batch_bigstep_gpu import _inputs,_scalar_state
    from gpuwm.core import dycore
    from gpuwm.ensemble.batch_bigstep import prepare_w_damping
    from gpuwm.ensemble.batch_state import BatchedDomainState,SHARED_STATE_CANDIDATES
    prepared=[]
    for member,source in enumerate(_inputs(members,terrain=terrain,mapped=True)):
        cfg=replace(source.cfg,w_damping=1)
        source=replace(source,cfg=cfg)
        source.scratch['rk_ww'].fill(np.float32(100000*(member+1)))
        index=np.arange(source.arrays['w'].size).reshape(source.arrays['w'].shape)
        source.arrays['w'][...]=np.where(index%2,np.float32(-0.5),np.float32(0.5))
        source.arrays['rw_t'][...]=np.float32(0.25)
        prepared.append(source)
    batch=BatchedDomainState.from_prepared(prepared,array_module=cp,available_bytes=1<<28,
        shared_fields=tuple(SHARED_STATE_CANDIDATES & prepared[0].arrays.keys()) if shared else ())
    before={name:array.get().tobytes() for name,array in batch.storage.arrays.items() if name!='rw_t'}
    prepare_w_damping(batch)()
    for member,source in enumerate(prepared):
        scalar=_scalar_state(source)
        dycore.apply_w_damping(scalar,source.cfg,scalar.existing_scratch('rk_ww'))
        actual=batch.member_view('rw_t',member).get()
        assert actual.tobytes()==scalar.rw_t.get().tobytes()
        assert np.any(actual[1:-1]!=np.float32(0.25))
        assert np.all(actual[[0,-1]]==np.float32(0.25))
    assert {name:array.get().tobytes() for name,array in batch.storage.arrays.items() if name!='rw_t'}==before
