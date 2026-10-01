"""Stage adapters and exact word grading shared by this lane's tests."""
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from gpuwm.verify.uwpbl_oracle import load, stage
from gpuwm.verify.uwpbl_ref import eddy

ROOT = Path(__file__).resolve().parents[1]
TRB_MAP = dict(u='ufd', v='vfd', t='tfd', taux='tautotx', tauy='tautoty',
               cld='cldn', qt='qtfd', qv='qvfd', ql='qlfd', sl='slfd')
EDDY_MAP = dict(t='t8', qv='cloud_1', ql='cloud_2', qi='cloud_3', s='s8',
    rpdel='rpdel8', cldn='cldn8', qrl='qrl8', wsedl='wsedl8', z='zm8', zi='zi8',
    pmid='pmid8', pi='pint8', u='u8', v='v8', qflx='cflx_1', ustar='ustar8',
    kvm_out='kvm', kvh_out='kvh', tke='tke8', sm_aw='smaw')
TRB_OUT = 'ustar rrho s2 n2 ri qtfd sfi sfuh sflh slfd slv slslope qtslope chs chu cms cmu minpblh'.split()
EDDY_OUT = 'ustar8 pblh kvm kvh kvq cgh cgs tpert qpert wpert tke8 bprod sprod sfi tauresx tauresy ipbl kpblh wstarPBL turbtype smaw'.split()


def fixture(grid, spy=True):
    directory = os.environ.get('UWPBL_ORACLE_DIR')
    if not directory:
        pytest.skip('UWPBL_ORACLE_DIR is unset')
    stem = Path(directory) / ('cases-'+grid+('-spy' if spy else ''))
    if not Path(str(stem)+'.manifest').is_file():
        pytest.skip('missing oracle fixture '+str(stem))
    return load(stem)


def records(fx, stage_name):
    suffix = '/'+stage_name+'/z'
    # Eddy stage has zm8 rather than z.
    if stage_name == 'eddy_in':
        suffix = '/eddy_in/zm8'
    contexts = [n[:-len(suffix)] for n in fx.names() if n.endswith(suffix)]
    for context in contexts:
        c, s, it = context.split('/')
        yield int(c[1:]), int(s[1:]), int(it[2:])


def lists(record):
    return {n: [None]+[float(x) for x in v] for n, v in record.items()}


def assert_words(actual, expected, label):
    expected = np.asarray(expected)
    actual = np.asarray(actual, dtype=expected.dtype)
    dt = np.uint64 if expected.dtype.itemsize == 8 else np.uint32
    a, b = actual.view(dt).reshape(-1), expected.view(dt).reshape(-1)
    mismatches = int(np.count_nonzero(a != b))
    # Map IEEE words monotonically, including negative values, for ULP distance.
    bits = expected.dtype.itemsize*8
    sign = 1 << (bits-1)
    def ordered(x):
        x = int(x)
        return (~x & ((1 << bits)-1)) if x & sign else x | sign
    ulp = max((abs(ordered(x)-ordered(y)) for x,y in zip(a,b)), default=0)
    assert mismatches == 0, f'{label}: {mismatches}/{a.size} mismatches, max ULP {ulp}'
    return a.size


def arguments(which, r, step):
    func = getattr(eddy, which)
    mapping = TRB_MAP if which == 'trbintd' else EDDY_MAP
    pver = len(r['z' if which == 'trbintd' else 'zm8'])-1
    extra = {'pver': pver, 'nturb': 5, 'wstarent': True, 'kvinit': step == 1}
    if which == 'compute_eddy_diff':
        extra['ztodt'] = r['ztodt'][1]
    return [extra[n] if n in extra else r[mapping.get(n,n)] for n in inspect.signature(func).parameters]


def require_lanes():
    for name in ('wvsat', 'vdiff', 'caleddy'):
        if importlib.util.find_spec('gpuwm.verify.uwpbl_ref.'+name) is None:
            pytest.skip('other lane missing: '+name)


def test_saturation_install(monkeypatch):
    """Temporary local copy of vqsatd, only used when the lane is absent.

    Never installs or writes another lane's files. CPU arithmetic is literal.
    """
    name = 'gpuwm.verify.uwpbl_ref.wvsat'
    if importlib.util.find_spec(name) is not None:
        return False
    from gpuwm.core.uwpbl_constants import ESTBL, PCF, EPSILO, LATVAP, LATICE, RH2O, CPAIR, TMELT
    from gpuwm.verify.uwpbl_ref.fortran import fmin, fmax, aint
    def fqsatd(t, p):
        e = fmax(fmin(t, 375.16), 173.16)
        i = int(e-173.16)+1
        ai = aint(e-173.16)
        es = (173.16+ai-e+1.0)*ESTBL[i-1]-(173.16+ai-e)*ESTBL[i]
        omeps = 1.0-EPSILO
        qs = fmin(1.0, EPSILO*es/(p-omeps*es))
        if qs < 0.0:
            qs, es = 1.0, p
        trinv = 1.0/20.0
        tc = t-TMELT
        weight = fmin(-tc*trinv,1.0)
        hlatsb = LATVAP+weight*LATICE
        hlatvp = LATVAP-2369.0*tc
        hltalt = hlatsb if t < TMELT else hlatvp
        tterm = PCF[0]+tc*(PCF[1]+tc*(PCF[2]+tc*(PCF[3]+tc*PCF[4]))) if tc >= -20.0 and tc < 0.0 else 0.0
        desdt = hltalt*es/(RH2O*t*t)+tterm*trinv
        gam = hltalt*qs*p*desdt/(CPAIR*es*(p-omeps*es))
        if qs == 1.0:
            gam = 0.0
        return 1, es, qs, gam
    module = types.ModuleType(name)
    module.fqsatd = fqsatd
    monkeypatch.setitem(sys.modules, name, module)
    return True


def cuda_source(full=False):
    kdir = ROOT/'gpuwm/core/kernels'
    libm = kdir/'glibc_flt64.cuh'
    if not libm.exists():
        libm = kdir/'uwpbl_libm_stub.cuh'
    if not libm.exists():
        pytest.skip('neither glibc_flt64.cuh nor uwpbl_libm_stub.cuh is present')
    code = libm.read_text()+(kdir/'uwpbl_common.cuh').read_text()
    sat = kdir/'uwpbl_wvsat.cuh'
    code += sat.read_text() if sat.exists() else temporary_cuda_saturation()
    if full:
        for name in ('uwpbl_vdiff.cuh','uwpbl_zisocl.cuh','uwpbl_caleddy.cuh'):
            if not (kdir/name).exists():
                pytest.skip('other CUDA lane missing: '+name)
            code += (kdir/name).read_text()
    ed = (kdir/'uwpbl_eddy.cuh').read_text()
    code += ed if full else ed.split('__device__ void uw_compute_eddy_diff')[0]+'\n#endif\n'
    return code


def temporary_cuda_saturation():
    from gpuwm.core.uwpbl_constants import PCF
    pcf = [x.hex() for x in PCF]
    return '''
// Test-only temporary copy: module_cam_wv_saturation.F:82-98, 556-618.
__device__ int uw_fqsatd(R8 t,R8 p,R8& es,R8& qs,R8& gam) {
    R8 e=uw_max(uw_min(t,R8(375.16)),R8(173.16));
    int i=uw_int(e-R8(173.16))+1;
    R8 ai=uw_aint(e-R8(173.16));
    es=(R8(173.16)+ai-e+R8(1.0))*R8(UW_ESTBL[i-1])-(R8(173.16)+ai-e)*R8(UW_ESTBL[i]);
    R8 omeps=R8(1.0)-R8(UW_EPSILO);
    qs=uw_min(R8(1.0),R8(UW_EPSILO)*es/(p-omeps*es));
    if(qs<R8(0.0)){ qs=R8(1.0); es=p; }
    R8 trinv=R8(0x1.999999999999ap-5); // 1/20, folded
    R8 tc=t-R8(UW_TMELT),weight=uw_min(-(tc*trinv),R8(1.0));
    R8 hlatsb=R8(UW_LATVAP)+weight*R8(UW_LATICE),hlatvp=R8(UW_LATVAP)-R8(2369.0)*tc;
    R8 hltalt=t<R8(UW_TMELT)?hlatsb:hlatvp;
    R8 tterm=tc>=R8(-20.0)&&tc<R8(0.0)?R8(PCF0)+tc*(R8(PCF1)+tc*(R8(PCF2)+tc*(R8(PCF3)+tc*R8(PCF4)))):R8(0.0);
    R8 desdt=hltalt*es/(R8(UW_RH2O)*t*t)+tterm*trinv;
    gam=hltalt*qs*p*desdt/(R8(UW_CPAIR)*es*(p-omeps*es));
    if(qs==R8(1.0))gam=R8(0.0);
    return 1;
}
'''.replace('PCF0',pcf[0]).replace('PCF1',pcf[1]).replace('PCF2',pcf[2]).replace('PCF3',pcf[3]).replace('PCF4',pcf[4])


class StageGpu:
    def __init__(self, which, names, width):
        self.cp = pytest.importorskip('cupy')
        self.names, self.width = names, width
        params = list(inspect.signature(getattr(eddy,which)).parameters)
        mapping = TRB_MAP if which == 'trbintd' else EDDY_MAP
        scalar = set('taux tauy ustar rrho minpblh shflx qflx pblh tpert qpert wpert tauresx tauresy ksrftms ipbl kpblh wstarPBL'.split())
        args = []
        for n in params:
            if n in ('pver','nturb','wstarent','kvinit','ztodt'):
                args.append(dict(pver='pver', nturb='5', wstarent='true', kvinit='first', ztodt='ztodt')[n])
            else:
                view = f'V{{data+{names.index(mapping.get(n,n))}*width,1}}'
                args.append(view+'(1)' if n in scalar else view)
        body = f'uw_{which}('+','.join(args)+(',ws' if which=='compute_eddy_diff' else '')+');'
        code = cuda_source(full=which=='compute_eddy_diff')
        code += '''
extern "C" __global__ void replay(int pver,int width,double* data,bool first,double dt,double* pool,int* ipool,int* err){
    Ws ws{pool,ipool,1,0,32768,0,4096,err}; R8 ztodt(dt);
''' + body + '\n}\n'
        self.mod = self.cp.RawModule(code=code,options=('-std=c++17',),name_expressions=('replay',))
        self.kernel = self.mod.get_function('replay')
        from gpuwm.core.uwpbl_constants import ESTBL
        ptr = self.mod.get_global('UW_ESTBL')
        table = np.asarray(ESTBL,np.float64)
        self.cp.cuda.runtime.memcpy(ptr.ptr,table.ctypes.data,table.nbytes,self.cp.cuda.runtime.memcpyHostToDevice)
        self.pool = self.cp.empty(32768,self.cp.float64)
        self.ipool = self.cp.empty(4096,self.cp.int32)
    def run(self, r, first=False):
        data = np.zeros((len(self.names),self.width),np.float64)
        for i,n in enumerate(self.names):
            data[i,:len(r[n])] = r[n]
        dev = self.cp.asarray(data)
        err = self.cp.zeros(1,self.cp.int32)
        self.kernel((1,),(1,),(np.int32(self.width-1),np.int32(self.width),dev,np.bool_(first),np.float64(r.get('ztodt',[0.0])[0]),self.pool,self.ipool,err))
        assert int(err.get()[0]) == 0, 'workspace overflow'
        result = dev.get()
        return {n: result[i,:len(r[n])] for i,n in enumerate(self.names)}


class TrigProbe:
    """The kernel's own cos/acos words, one argument at a time."""

    def __init__(self):
        self.cp = pytest.importorskip('cupy')
        kdir = ROOT/'gpuwm/core/kernels'
        code = (kdir/'glibc_flt64.cuh').read_text()+(kdir/'uwpbl_common.cuh').read_text()
        code += ('\nextern "C" __global__ void uw_trig_probe(int fn,double x,double* y)'
                 '{y[0]=fn?uw_acos(R8(x)).v:uw_cos(R8(x)).v;}\n')
        self.kernel = self.cp.RawModule(code=code,options=('-std=c++17',)).get_function('uw_trig_probe')
        self.out = self.cp.empty(1,self.cp.float64)
        self.cache = {}

    def __call__(self, fn, x):
        key = (fn, x.hex())
        if key not in self.cache:
            self.kernel((1,),(1,),(np.int32(fn == 'acos'),np.float64(x),self.out))
            self.cache[key] = float(self.cp.asnumpy(self.out)[0])
        return self.cache[key]


def words_equal(actual, expected):
    expected = np.asarray(expected)
    actual = np.asarray(actual, dtype=expected.dtype)
    dt = np.uint64 if expected.dtype.itemsize == 8 else np.uint32
    return bool(np.array_equal(actual.view(dt).reshape(-1), expected.view(dt).reshape(-1)))


def eddy_with_kernel_trig(before, step, probe):
    """compute_eddy_diff on the CPU with the kernel's cos/acos words.

    Returns the outputs and the trig calls whose kernel word differs from
    glibc's.  Nothing else is substituted.
    """
    import math
    originals = {fn: getattr(math, fn) for fn in ('cos', 'acos')}
    differences = []

    def hook(fn):
        def evaluate(x):
            word = probe(fn, x)
            if word.hex() != originals[fn](x).hex():
                differences.append((fn, x.hex(), originals[fn](x).hex(), word.hex()))
            return word
        return evaluate

    result = lists(before)
    for fn in originals:
        setattr(math, fn, hook(fn))
    try:
        eddy.compute_eddy_diff(*arguments('compute_eddy_diff', result, step))
    finally:
        for fn, original in originals.items():
            setattr(math, fn, original)
    return {n: a[1:] for n, a in result.items()}, differences
