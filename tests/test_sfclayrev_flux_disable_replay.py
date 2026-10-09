"""WRF preserves CHS, CHS2 and CQS2 when ISFFLX is zero.

WRF v4.6.1 sf_sfclayrev.F90:794 skips assignments at 893, 902, 903.
The wrapper copies their incoming state in and back out at 187:189, 251:253.
"""
import cupy as cp
import numpy as np
import pytest
from conftest import requires_gpu
from _sfclayrev_oracle import load_fixture
from gpuwm.core.sfclay import sfclay, launch_sfclay

@requires_gpu
def test_flux_disable_preserves_previous_exchange_coefficients():
    fixture = load_fixture()
    g = {k: cp.asarray(v[:1][None, :]) for k, v in fixture.inputs.items()}
    result = sfclay(*[g[k] for k in ('u','v','t','qv','p','dz8w','psfc','tsk','znt','pblh','mavail','xland')], option=1, dx=float(g['dx'].item()), **{k:g[k] for k in ('qsfc','zol','ust','ustm','mol','hfx','qfx','lakemask')})
    saved = {k: cp.asnumpy(getattr(result,k)).view(np.uint32).copy() for k in ('chs','chs2','cqs2')}
    launch_sfclay(*[g[k] for k in ('u','v','t','qv','p','dz8w','psfc','tsk','pblh','mavail','xland','lakemask')],result,option=1,dx=float(g['dx'].item()),isfflx=False)
    for name, before in saved.items():
        after = cp.asnumpy(getattr(result,name)).view(np.uint32)
        assert np.array_equal(after,before), (name,before,after)


@requires_gpu
@pytest.mark.parametrize('option', [1, 91])
def test_flux_off_seeds_and_carries_exchange_buffers(option):
    fixture = load_fixture()
    g = {k: cp.asarray(v[:4][None, :]) for k, v in fixture.inputs.items()}
    args = [g[k] for k in ('u', 'v', 't', 'qv', 'p', 'dz8w', 'psfc', 'tsk')]
    result = sfclay(*args, *[g[k] for k in ('znt', 'pblh', 'mavail', 'xland')],
        option=option, dx=float(g['dx'][0, 0].item()), isfflx=False,
        **{k: g[k] for k in ('qsfc', 'zol', 'ust', 'ustm', 'mol', 'hfx', 'qfx', 'lakemask')})
    for name in ('chs', 'chs2', 'cqs2'):
        assert np.array_equal(cp.asnumpy(getattr(result, name)), np.zeros((1, 4), np.float32))
    # Signed zero and subnormals must also be retained word for word.
    words = np.array([[0x80000000, 1, 0x3e800000, 0x3f400000]], np.uint32)
    for name in ('chs', 'chs2', 'cqs2'):
        getattr(result, name)[:] = cp.asarray(words.view(np.float32))
    for step in range(2):
        launch_sfclay(*args, *[g[k] for k in ('pblh', 'mavail', 'xland', 'lakemask')],
            result, option=option, dx=float(g['dx'][0, 0].item()), isfflx=False)
        for name in ('chs', 'chs2', 'cqs2'):
            assert np.array_equal(cp.asnumpy(getattr(result, name)).view(np.uint32), words), name
        for name in ('hfx', 'qfx', 'lh', 'flhc', 'flqc'):
            assert not cp.asnumpy(getattr(result, name)).any(), (step, name)
