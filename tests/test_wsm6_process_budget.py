"""Budget attribution must not alter physics or hide missing stage transfers.
SPDX-License-Identifier: Apache-2.0
"""
import json
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.verify.wsm6_process_budget import ProcessLedger, instrument_source

FIXTURE = Path(__file__).parent / 'data/oracles/wsm6'


def test_changed_source_is_rejected():
    with pytest.raises(ValueError, match='anchor changed'):
        instrument_source('unqualified source')


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('hail,dt,steps', [(0,30.,12),(1,30.,12),
    (0,150.,3),(1,150.,3),(0,300.,4),(1,300.,4)])
def test_ledger_telescopes_and_preserves_stock_and_wrf_words(hail,dt,steps):
    import cupy as cp
    from gpuwm import wrf_exact
    from gpuwm.core.wsm6 import launch_wsm6
    from tools.wsm6_wrf461_column_oracle.run_gpu import read_columns, tag_of
    if wrf_exact.ENABLED:
        pytest.skip('research observer deliberately unavailable in strict mode')
    ncol,nz,level,surf=read_columns(FIXTURE/'columns.bin')
    d={k:cp.asarray(v.reshape(nz,1,ncol)) for k,v in level.items()}
    names=('th','qv','qc','qr','qi','qs','qg')
    def state():
        s=[d[k].copy() for k in names]+[cp.zeros((1,ncol),cp.float32) for _ in range(7)]
        for i,k in ((7,'rain'),(9,'snow'),(11,'graupel')):
            s[i][...]=cp.asarray(surf[k].reshape(1,ncol))
        return s+[cp.full_like(d['th'],x) for x in (2.49,4.99,9.99)]
    stock=state();obs=state()
    bands=cp.broadcast_to((cp.arange(nz,dtype=cp.uint8)%4)[:,None,None],d['th'].shape).copy()
    ledger=ProcessLedger(bands)
    weight=d['den'].astype(cp.float64)*d['dz']
    def content(s):
        return cp.stack([cp.sum(v.astype(cp.float64)*weight,axis=0) for v in
            (s[0]*d['pii'],s[1],s[2],s[4],s[3],s[5],s[6])])
    initial=content(obs)
    def advance(s):
        launch_wsm6(*s[:7],d['den'],d['pii'],d['p'],d['dz'],*s[7:14],dt,
            effc=s[14],effi=s[15],effs=s[16],hail_opt=hail)
    for _ in range(steps):advance(stock)
    with ledger.observe():
        for _ in range(steps):advance(obs)
    assert bool(cp.stack([cp.all(a.view(cp.uint32)==b.view(cp.uint32)) for a,b in zip(stock,obs)]).all().get())
    residual=content(obs)-initial-ledger.values[:56].reshape(8,7,4,1,ncol).sum(axis=(0,2))
    assert float(cp.abs(residual[1:]).max().get())<1e-9
    assert float(cp.abs(residual[0]).max().get())<1e-8
    # Independent WRF 4.6.1 final words, excluding its two subnormal columns
    # in the default FTZ build, exactly as the production oracle gate does.
    meta=json.loads((FIXTURE/'columns.json').read_text())
    keep=[i for i,c in enumerate(meta['columns']) if c['name'] not in ('edge-negative','edge-dry-tiny')]
    tag=tag_of(hail,dt,steps)
    expected=np.load(FIXTURE/'wrf461-final-step.npz')
    targets=names+('rain','rainncv','snow','snowncv','graupel','graupelncv','sr','re_cloud','re_ice','re_snow')
    for key,value in zip(targets,obs):
        actual=value.get().reshape((-1,ncol))[:,keep]
        want=expected[tag+'/'+key].reshape((-1,ncol))[:,keep]
        np.testing.assert_array_equal(actual.view(np.uint32),want.view(np.uint32),err_msg=key)
