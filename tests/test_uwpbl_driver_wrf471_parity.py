"""Full WRF column parity and an independent mixed-precision boundary test."""
import numpy as np
import pytest
from gpuwm.verify.uwpbl_oracle import dims, step_arrays, stage, MASS_INPUTS, MASS_OUTPUTS, FULL_OUTPUTS, SURFACE_OUTPUTS
from gpuwm.verify.uwpbl_ref.driver import camuwpbl_step, _prepare
from uwpbl_eddy_support import fixture, assert_words, require_lanes, cuda_source, ROOT
import sys
import types


class DriverGpu:
    def __init__(self, nk):
        self.cp = pytest.importorskip('cupy')
        self.nk = nk
        self.inputs = list(MASS_INPUTS)+['p8w','z_at_w','hfx','qfx','ust','ht']
        self.outputs = list(MASS_OUTPUTS)+list(FULL_OUTPUTS)+[n for n in SURFACE_OUTPUTS if n != 'kpbl2d']
        def view(name,names,base):
            return f'UwF32View{{{base}+{names.index(name)}*width,1}}'
        inviews = [view(n,self.inputs,'in') for n in MASS_INPUTS+('p8w','z_at_w')]
        inval = [f'in[{self.inputs.index(n)}*width]' for n in ('hfx','qfx','ust','ht')]+['dt','step']
        outviews = [view(n,self.outputs,'out') for n in ('kvm3d','kvh3d')+MASS_OUTPUTS+('tke_pbl','turbtype3d','smaw3d')]
        outptrs = [f'out+{self.outputs.index(n)}*width' for n in ('tauresx2d','tauresy2d','tpert2d','qpert2d','wpert2d','pblh2d')]+['kpbl']
        code = cuda_source(full=True)+(ROOT/'gpuwm/core/kernels/uwpbl_driver.cuh').read_text()
        code += '\nextern "C" __global__ void driver(int nk,int width,float* in,float* out,int* kpbl,float dt,int step,double* pool,int* ipool,int* err){\n'
        code += 'Ws ws{pool,ipool,1,0,32768,0,4096,err};\n'
        code += 'UwColumnIn ci{'+','.join(inviews+inval)+'};\n'
        code += 'UwColumnOut co{'+','.join(outviews+outptrs)+'};\n'
        code += 'uw_camuwpbl_column(nk,ci,co,ws);\n}\n'
        self.mod = self.cp.RawModule(code=code,options=('-std=c++17',),name_expressions=('driver',))
        self.kernel = self.mod.get_function('driver')
        from gpuwm.core.uwpbl_constants import ESTBL
        ptr = self.mod.get_global('UW_ESTBL')
        table = np.asarray(ESTBL,np.float64)
        self.cp.cuda.runtime.memcpy(ptr.ptr,table.ctypes.data,table.nbytes,self.cp.cuda.runtime.memcpyHostToDevice)
        self.pool = self.cp.empty(32768,self.cp.float64)
        self.ipool = self.cp.empty(4096,self.cp.int32)
    def run(self, a):
        ncol = len(a['u'])
        out = {n:np.empty_like(a[n]) for n in MASS_OUTPUTS+FULL_OUTPUTS+SURFACE_OUTPUTS}
        width = self.nk+1
        for c in range(ncol):
            i = np.zeros((len(self.inputs),width),np.float32)
            o = np.zeros((len(self.outputs),width),np.float32)
            for row,n in enumerate(self.inputs):
                vals = np.asarray(a[n][c]).reshape(-1)
                i[row,:len(vals)] = vals
            for n in ('kvm3d','kvh3d','tauresx2d','tauresy2d'):
                vals = np.asarray(a[n+'_in'][c]).reshape(-1)
                o[self.outputs.index(n),:len(vals)] = vals
            di,do = self.cp.asarray(i),self.cp.asarray(o)
            kpbl,err = self.cp.zeros(1,self.cp.int32),self.cp.zeros(1,self.cp.int32)
            self.kernel((1,),(1,),(np.int32(self.nk),np.int32(width),di,do,kpbl,np.float32(a['dt']),np.int32(a['itimestep']),self.pool,self.ipool,err))
            assert int(err.get()[0]) == 0, 'workspace overflow or WRF fatal'
            result = do.get()
            for row,n in enumerate(self.outputs):
                if n in MASS_OUTPUTS:
                    out[n][c] = result[row,:self.nk]
                elif n in FULL_OUTPUTS:
                    out[n][c] = result[row]
                else:
                    out[n][c] = result[row,0]
            out['kpbl2d'][c] = kpbl.get()[0]
        return out


@pytest.mark.parametrize('grid',['g35','g44','g61'])
@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
def test_driver(grid,backend):
    require_lanes()
    fx = fixture(grid,False)
    ncol,nk,nsteps = dims(fx)
    assert (ncol,nsteps) == (18,4)
    gpu = DriverGpu(nk) if backend == 'gpu' else None
    words = 0
    for step in range(1,nsteps+1):
        a = step_arrays(fx,step)
        result = gpu.run(a) if gpu else camuwpbl_step(a)
        assert set(result) == set(MASS_OUTPUTS+FULL_OUTPUTS+SURFACE_OUTPUTS)
        for n in result:
            assert result[n].dtype == a[n].dtype, n
            words += assert_words(result[n],a[n],f'{grid} s{step} {n}')
    print(f'{grid} {backend} driver: 72 columns, {words} words, 0 mismatches, max ULP 0')


@pytest.mark.parametrize('grid',['g35','g44','g61'])
def test_driver_input_boundary(grid):
    fx, spy = fixture(grid,False),fixture(grid)
    ncol,nk,nsteps = dims(fx)
    mapping = dict(u='u8',v='v8',pmid='pmid8',rpdel='rpdel8',z='zm8',t='t8',
        s='s8',qrl='qrl8',wsedl='wsedl8',cldn='cldn8',pi='pint8',zi='zi8',
        kvh='kvh_in',kvm='kvm_in',taux='taux',tauy='tauy',shflx='shflx',
        tauresx='tauresx',tauresy='tauresy',kvq='kvq',cgh='cgh',cgs='cgs')
    words = 0
    for step in range(1,nsteps+1):
        a = step_arrays(fx,step)
        for col in range(ncol):
            inputs = {n:v if n in ('dt','itimestep') else v[col] for n,v in a.items()}
            prepared = _prepare(inputs,{})
            expected = stage(spy,col+1,step,0,'eddy_in')
            for src,dest in mapping.items():
                words += assert_words(prepared[src][1:],expected[dest],f'{grid} c{col+1}/s{step} input {dest}')
            for m in range(1,4):
                words += assert_words(prepared['cloud'][m][1:],expected[f'cloud_{m}'],f'{grid} cloud{m}')
    print(f'{grid} CPU driver input boundary: 72 columns, {words} words, 0 mismatches, max ULP 0')


@pytest.mark.parametrize('grid',['g35','g44','g61'])
def test_driver_call_replay(grid,monkeypatch):
    """Qualify driver arithmetic with eddy and outer vdiff replies from spies."""
    import inspect
    from gpuwm.verify.uwpbl_ref import driver, eddy
    from uwpbl_eddy_support import EDDY_MAP, test_saturation_install
    test_saturation_install(monkeypatch)
    fx,spy = fixture(grid,False),fixture(grid)
    ctx = {}
    def eddy_reply(*args):
        names = list(inspect.signature(eddy.compute_eddy_diff).parameters)
        reply = stage(spy,ctx['col'],ctx['step'],5,'eddy_out')
        for n,value in zip(names,args):
            dest = EDDY_MAP.get(n,n)
            if isinstance(value,list) and dest in reply:
                value[1:] = [float(x) for x in reply[dest]]
    monkeypatch.setattr(driver,'compute_eddy_diff',eddy_reply)
    def vdiff_reply(*args):
        (pver,ncnst,pmid,pi,rpdel,t,ztodt,taux,tauy,shflx,cflx,ntop,nbot,
         kvh,kvm,kvq,cgs,cgh,zi,ksrftms,qmincg,fl,u,v,q,dse,tx,ty,dtk,topflx,
         tauresx,tauresy,itaures) = args
        vin = stage(spy,ctx['col'],ctx['step'],-1,'vdiff_outer_in')
        vout = stage(spy,ctx['col'],ctx['step'],-1,'vdiff_outer_out')
        assert ncnst == 5 and itaures == 1
        assert fl == {'u':True,'v':True,'s':True,'q':[None]+[True]*5}
        for n,value in dict(pmid8=pmid,pint8=pi,rpdel8=rpdel,t8=t,taux=taux,tauy=tauy,shflx=shflx,
                            cflx_all=cflx,kvh=kvh,kvm=kvm,kvq=kvq,cgs=cgs,cgh=cgh,zi8=zi,
                            ksrftms=ksrftms,wind_tends_1=u,wind_tends_2=v,stnd=dse,
                            **{f'cloudtnd_{m}':q[m] for m in range(1,6)}).items():
            assert_words(value[1:],vin[n],f'outer vdiff input {n}')
        for n,value in dict(wind_tends_1=u,wind_tends_2=v,stnd=dse,tauresx=tauresx,tauresy=tauresy,
                            tautmsx=tx,tautmsy=ty,dtk=dtk,topflx=topflx,
                            **{f'cloudtnd_{m}':q[m] for m in range(1,6)}).items():
            value[1:] = [float(x) for x in vout[n]]
        return ''
    module = types.ModuleType('gpuwm.verify.uwpbl_ref.vdiff')
    module.compute_vdiff = vdiff_reply
    monkeypatch.setitem(sys.modules,module.__name__,module)
    # aqsat's diagnostic calls are inert for outputs; still compute their qs.
    sat = sys.modules.get('gpuwm.verify.uwpbl_ref.wvsat')
    if sat is None:
        from gpuwm.verify.uwpbl_ref import wvsat as sat
    if not hasattr(sat,'aqsat'):
        def aqsat(t,p,es,qs,kstart,kend):
            for k in range(kstart,kend+1):
                status,es[k],qs[k],gam = sat.fqsatd(t[k],p[k])
        monkeypatch.setattr(sat,'aqsat',aqsat,raising=False)
    ncol,nk,nsteps = dims(fx)
    words = 0
    for step in range(1,nsteps+1):
        a = step_arrays(fx,step)
        ctx['step'] = step
        for c in range(ncol):
            ctx['col'] = c+1
            inputs = {n:v if n in ('dt','itimestep') else v[c] for n,v in a.items()}
            result = driver.camuwpbl_column(inputs,{})
            for n in MASS_OUTPUTS+FULL_OUTPUTS+SURFACE_OUTPUTS:
                assert np.asarray(result[n]).dtype == a[n].dtype,n
                words += assert_words(result[n],a[n][c],f'{grid} c{c+1}/s{step} output {n}')
    print(f'{grid} CPU driver call replay: 72 columns, {words} words, 0 mismatches, max ULP 0')
