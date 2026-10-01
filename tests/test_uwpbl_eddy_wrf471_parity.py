"""Exact stage tests of trbintd, sfdiag and compute_eddy_diff against the SPY records.

Run on a glibc 2.43 x86-64 host with UWPBL_ORACLE_DIR set to a build.sh fixture
directory.
"""
import inspect
import sys
import types

import numpy as np
import pytest
from gpuwm.verify.uwpbl_oracle import stage
from gpuwm.verify.uwpbl_ref import eddy
from uwpbl_eddy_support import (fixture, records, lists, assert_words, arguments,
    TRB_OUT, EDDY_OUT, StageGpu, require_lanes, cuda_source, TrigProbe,
    words_equal, eddy_with_kernel_trig,
    test_saturation_install as install_sat)


@pytest.mark.parametrize('grid', ['g35','g44','g61'])
@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
def test_trbintd(grid, backend, monkeypatch):
    fx = fixture(grid)
    install_sat(monkeypatch)
    gpu = None
    calls = words = 0
    for col, step, it in records(fx,'trbintd_in'):
        before = stage(fx,col,step,it,'trbintd_in')
        expected = stage(fx,col,step,it,'trbintd_out')
        if backend == 'cpu':
            result = lists(before)
            eddy.trbintd(*arguments('trbintd',result,step))
            result = {n:a[1:] for n,a in result.items()}
        else:
            if gpu is None:
                gpu = StageGpu('trbintd',list(before),len(before['z'])+1)
            result = gpu.run(before)
        for n in TRB_OUT:
            words += assert_words(result[n],expected[n],f'{grid} c{col}/s{step}/it{it} {n}')
        calls += 1
    assert calls == 360, f'unexpected record coverage: {calls}'
    print(f'{grid} {backend} trbintd: {calls} calls, {words} words, 0 mismatches, max ULP 0')


@pytest.mark.parametrize('grid', ['g35','g44','g61'])
@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
def test_iteration_retrieval_and_relaxation(grid, backend, monkeypatch):
    fx = fixture(grid)
    install_sat(monkeypatch)
    gpu = None
    calls = words = 0
    for col, step, it in records(fx,'trbintd_in'):
        if it == 5:
            continue
        vd = lists(stage(fx,col,step,it,'vdiff_in_loop_out'))
        next_in = stage(fx,col,step,it+1,'trbintd_in')
        current = lists(stage(fx,col,step,it,'trbintd_in'))
        pver = len(current['z'])-1
        cal = lists(stage(fx,col,step,it,'caleddy_out'))
        if backend == 'cpu':
            eddy._retrieve(pver,vd['slfd'],vd['qtfd'],current['qi'],current['z'],
                           current['pmid'],current['tfd'],current['qvfd'],current['qlfd'])
            if it > 1:
                eddy._relax(pver,cal['kvm_out'],cal['kvh_out'],cal['kvm'],cal['kvh'])
        else:
            if gpu is None:
                gpu = IterationGpu(pver)
            result = gpu.run(vd,current,cal,it)
            for n in ('tfd','qvfd','qlfd'):
                current[n][1:] = result[n][:pver]
            for n in ('kvm_out','kvh_out'):
                cal[n][1:] = result[n]
        for n in ('tfd','qvfd','qlfd'):
            words += assert_words(current[n][1:],next_in[n],f'{grid} c{col}/s{step}/it{it+1} {n}')
        for n in ('kvm_out','kvh_out'):
            words += assert_words(cal[n][1:],vd[n][1:],f'{grid} c{col}/s{step}/it{it} relaxation {n}')
        calls += 1
    assert calls == 288
    print(f'{grid} {backend} iteration replay: {calls} calls, {words} words, 0 mismatches, max ULP 0')


class IterationGpu:
    def __init__(self,pver):
        self.cp = pytest.importorskip('cupy')
        self.pver = pver
        self.names = 'slfd qtfd qi z pmid tfd qvfd qlfd kvm_out kvh_out kvm kvh'.split()
        views = {n:f'V{{data+{i}*width,1}}' for i,n in enumerate(self.names)}
        code = cuda_source()
        code += '\nextern "C" __global__ void iteration(int n,int width,double* data,int it){\n'
        code += 'uw_eddy_retrieve(n,'+','.join(views[n] for n in self.names[:8])+');\n'
        code += 'if(it>1)uw_eddy_relax(n,'+','.join(views[n] for n in self.names[8:])+');\n}\n'
        self.mod = self.cp.RawModule(code=code,options=('-std=c++17',),name_expressions=('iteration',))
        self.kernel = self.mod.get_function('iteration')
        from gpuwm.core.uwpbl_constants import ESTBL
        ptr = self.mod.get_global('UW_ESTBL')
        table = np.asarray(ESTBL,np.float64)
        self.cp.cuda.runtime.memcpy(ptr.ptr,table.ctypes.data,table.nbytes,self.cp.cuda.runtime.memcpyHostToDevice)
    def run(self,vd,current,cal,it):
        values = {n:vd[n] for n in ('slfd','qtfd')}
        values.update({n:current[n] for n in self.names[2:8]})
        values.update({n:cal[n] for n in self.names[8:]})
        data = np.zeros((12,self.pver+1),np.float64)
        for i,n in enumerate(self.names):
            a = values[n][1:]
            data[i,:len(a)] = a
        dev = self.cp.asarray(data)
        self.kernel((1,),(1,),(np.int32(self.pver),np.int32(self.pver+1),dev,np.int32(it)))
        result = dev.get()
        return {n:result[i] for i,n in enumerate(self.names)}


@pytest.mark.parametrize('grid', ['g35','g44','g61'])
@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
def test_compute_eddy_diff(grid, backend):
    require_lanes()
    fx = fixture(grid)
    gpu = None
    calls = words = 0
    residual = []
    for col, step, it in records(fx,'eddy_in'):
        assert it == 0
        before = stage(fx,col,step,it,'eddy_in')
        # make_spy leaves the shared iteration context at the final iturb.
        expected = stage(fx,col,step,5,'eddy_out')
        if backend == 'cpu':
            result = lists(before)
            eddy.compute_eddy_diff(*arguments('compute_eddy_diff',result,step))
            result = {n:a[1:] for n,a in result.items()}
        else:
            if gpu is None:
                gpu = StageGpu('compute_eddy_diff',list(before),len(before['zm8'])+1)
                probe = TrigProbe()
            result = gpu.run(before,step == 1)
            if not all(words_equal(result[n],expected[n]) for n in EDDY_OUT):
                # The kernel's cos/acos are correctly rounded and glibc's are
                # not (glibc_flt64.cuh).  Such a record passes only if the
                # CPU reference, replayed with the kernel's trig words, IS
                # the kernel's result and a trig word really differs.
                replay, differences = eddy_with_kernel_trig(before, step, probe)
                assert differences, (f'{grid} c{col}/s{step}: the kernel differs '
                                     'from WRF with no cos/acos residual')
                for n in EDDY_OUT:
                    words += assert_words(result[n],replay[n],
                                          f'{grid} c{col}/s{step} {n} vs trig replay')
                residual.append((col, step, differences))
                calls += 1
                continue
        for n in EDDY_OUT:
            words += assert_words(result[n],expected[n],f'{grid} c{col}/s{step}/it0 {n}')
        calls += 1
    assert calls == 72
    print(f'{grid} {backend} eddy: {calls} calls, {words} words, '
          f'{len(residual)} records explained by cos/acos: {residual}')


@pytest.mark.parametrize('grid', ['g35','g44','g61'])
def test_compute_eddy_call_replay(grid, monkeypatch):
    """Run the real iteration driver with caleddy/vdiff spy replies.

    This checks orchestration independently of other lane implementations.
    It does not qualify either replayed routine's arithmetic.
    """
    fx = fixture(grid)
    install_sat(monkeypatch)
    calnames = ('slfd qtfd qlfd slv ufd vfd pi z zi qflx shflx slslope qtslope '
        'chu chs cmu cms sfuh sflh n2 s2 ri rrho pblh ustar kvh kvm kvh_out '
        'kvm_out tpert qpert qrl kvf tke wstarent bprod sprod minpblh wpert tkes '
        'turbtype sm_aw kbase_o ktop_o ncvfin_o kbase_mg ktop_mg ncvfin_mg '
        'kbase_f ktop_f ncvfin_f wet web jtbu jbbu evhc jt2slv n2ht n2hb '
        'lwp opt_depth radinvfrac radf wstar wstar3fact ebrk wbrk lbrk ricl '
        'ghcl shcl smcl ghi shi smi rii lengi wcap pblhp cldn ipbl kpblh wsedl').split()
    ctx = {}
    def caleddy(pver, *args):
        ctx['it'] += 1
        cin = stage(fx,ctx['col'],ctx['step'],ctx['it'],'caleddy_in')
        cout = stage(fx,ctx['col'],ctx['step'],ctx['it'],'caleddy_out')
        for name, value in zip(calnames,args):
            if name in ('slfd','qtfd','qlfd','slv','ufd','vfd','slslope','qtslope',
                        'chu','chs','cmu','cms','sfuh','sflh','n2','s2','ri','rrho',
                        'ustar','kvh','kvm','qrl','minpblh','cldn','wsedl','qflx','shflx','pi','z','zi'):
                assert_words(value[1:len(cin[name])+1],cin[name],f'caleddy_in {name}')
            if name in cout:
                value[1:len(cout[name])+1] = [float(x) for x in cout[name]]
    def vdiff(pver,ncnst,pmid,pi,rpdel,t,ztodt,taux,tauy,shflx,cflx,
              ntop,nbot,kvh,kvm,kvq,cgs,cgh,zi,ksrftms,qmincg,fieldlist,
              u,v,q,dse,tautmsx,tautmsy,dtk,topflx,tauresx,tauresy,itaures):
        vin = stage(fx,ctx['col'],ctx['step'],ctx['it'],'vdiff_in_loop_in')
        vout = stage(fx,ctx['col'],ctx['step'],ctx['it'],'vdiff_in_loop_out')
        assert ncnst == 1 and itaures == 0 and ntop == 1 and nbot == pver
        assert fieldlist == {'u':True,'v':True,'s':True,'q':[None,True]}
        assert qmincg[1] == 0.0
        for name,value in dict(kvh_out=kvh,kvm_out=kvm,cgs=cgs,cgh=cgh,ufd=u,vfd=v,qtfd=q[1],slfd=dse).items():
            assert_words(value[1:len(vin[name])+1],vin[name],f'vdiff_in_loop_in {name}')
        for name,value in dict(ufd=u,vfd=v,qtfd=q[1],slfd=dse,tauresx=tauresx,tauresy=tauresy,jnk2d=dtk).items():
            value[1:len(vout[name])+1] = [float(x) for x in vout[name]]
        return ''
    for name, func in (('caleddy',caleddy),('vdiff',vdiff)):
        module = types.ModuleType('gpuwm.verify.uwpbl_ref.'+name)
        setattr(module, 'caleddy' if name=='caleddy' else 'compute_vdiff',func)
        monkeypatch.setitem(sys.modules,module.__name__,module)
    calls = words = 0
    for col, step, it in records(fx,'eddy_in'):
        ctx.update(col=col,step=step,it=0)
        r = lists(stage(fx,col,step,it,'eddy_in'))
        eddy.compute_eddy_diff(*arguments('compute_eddy_diff',r,step))
        expected = stage(fx,col,step,5,'eddy_out')
        for n in EDDY_OUT:
            words += assert_words(r[n][1:],expected[n],f'eddy replay c{col}/s{step} {n}')
        assert ctx['it'] == 5
        calls += 1
    assert calls == 72
    print(f'{grid} CPU eddy call replay: {calls} columns, {words} words, 0 mismatches, max ULP 0')
