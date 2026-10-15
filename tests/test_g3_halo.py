"""G3 forcing/spreading across the existing seam transport on one card.

Two logical partitions use the production local transport. An uneven split
separates source and destination pitches. This does not qualify a peer link.
"""
import numpy as np
import pytest
from test_g3_cu5_parity import cp, _fixture, _lev, run_fixture
from tilestream.multigpu import plan_split, seam_plan, _bands, _SeamChannel

pytestmark = pytest.mark.gpu


def _window(fx, axis, start, stop):
    out = {}
    for key, value in fx.items():
        if key.startswith('in_') and value.ndim >= 2:
            cut = -1 if axis == 'x' else 0
            sl = [slice(None)] * value.ndim
            sl[cut] = slice(start, stop)
            value = np.ascontiguousarray(value[tuple(sl)])
        out[key] = value
    return out


def _owned(out, axis, start, stop):
    result = {}
    for key, value in out.items():
        cut = -1 if axis == 'x' else (-2 if key in ('xf_ens','pr_ens') else 0)
        sl = [slice(None)] * value.ndim
        sl[cut] = slice(start, stop)
        result[key] = np.ascontiguousarray(value[tuple(sl)])
    return result


@pytest.mark.parametrize('axis', ['x','y'])
@pytest.mark.parametrize('case', ['g3-a','g3-c'])
def test_g3_reads_repaired_halos_through_existing_exchange(axis, case):
    fx = _fixture(case)
    # 17 cells split 8/9 with seven-cell halos: distinct copy pitches.
    cut = -1 if axis == 'x' else 0
    for key, value in list(fx.items()):
        if key.startswith('in_') and value.ndim >= 2:
            fx[key] = np.concatenate((value, np.take(value, [-1], axis=cut)), axis=cut)
    ny, nz, nx = fx['in_w'].shape
    specs = plan_split(nx, ny, 7, gx=2 if axis=='x' else 1,
                       gy=2 if axis=='y' else 1, periodic=False)
    want = run_fixture(fx)
    local = []
    winds = []
    for sp in specs:
        origin, extent = (sp.ci0,sp.cnx) if axis=='x' else (sp.cj0,sp.cny)
        part = _window(fx, axis, origin, origin+extent)
        w = cp.asarray(_lev(part['in_w']))
        low, high = (sp.i0,sp.i1) if axis=='x' else (sp.j0,sp.j1)
        keep = slice(low-origin, high-origin)
        dirty = cp.ones(w.shape[-1] if axis=='x' else w.shape[-2], dtype=cp.bool_)
        dirty[keep] = False
        if axis=='x': w[...,dirty] += cp.float32(5.)
        else: w[:,dirty,:] += cp.float32(5.)
        winds.append({'w':cp.ascontiguousarray(w)})
        local.append(part)
    # The omission control must move a retained output, not just a halo word.
    moved = 0
    for sp,part,arrays in zip(specs,local,winds):
        part['in_w'] = arrays['w'].get().transpose(1,0,2)
        low,high,origin = (sp.i0,sp.i1,sp.ci0) if axis=='x' else (sp.j0,sp.j1,sp.cj0)
        got = _owned(run_fixture(part), axis, low-origin,high-origin)
        ref = _owned(want,axis,low,high)
        moved += sum(np.count_nonzero(got[k].view('u4') != ref[k].view('u4')) for k in ref)
    assert moved > 0, 'the halo omission control did not move retained fields'
    stream = cp.cuda.get_current_stream()
    dev = cp.cuda.runtime.getDevice()
    channels = []
    for seam in seam_plan(specs,7,nx,ny):
        a,b = seam.src_gpu,seam.dst_gpu
        bands,nbytes = _bands(seam,winds[a],winds[b],nz,specs[a],specs[b],['w'])
        channel = _SeamChannel(seam,dev,dev,bands,nbytes,'local')
        channels.append(channel)
        channel.pack(winds[a],stream.ptr)
        channel.transfer(stream.ptr)
        channel.unpack(winds[b],stream.ptr)
    stream.synchronize()
    for sp,part,arrays in zip(specs,local,winds):
        part['in_w'] = arrays['w'].get().transpose(1,0,2)
        low,high,origin = (sp.i0,sp.i1,sp.ci0) if axis=='x' else (sp.j0,sp.j1,sp.cj0)
        got = _owned(run_fixture(part),axis,low-origin,high-origin)
        ref = _owned(want,axis,low,high)
        for key in ref:
            np.testing.assert_array_equal(got[key].view('u4'),ref[key].view('u4'),err_msg=f'{axis}/{case}/{key}')
