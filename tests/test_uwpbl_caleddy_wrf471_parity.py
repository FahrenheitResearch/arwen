"""Bitwise stage replay of caleddy, exacol and zisocl against the SPY records.

The CPU reference calls the host C library, so its trig words are the oracle's
only on a glibc 2.43 x86-64 host; set UWPBL_ORACLE_DIR to a build.sh fixture
directory to run it.
"""
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pytest

from gpuwm.verify.uwpbl_oracle import load
from gpuwm.verify.uwpbl_ref import caleddy as ref

ROOT = Path(__file__).resolve().parents[1]
ALIASES = dict(sl='slfd', qt='qtfd', ql='qlfd', u='ufd', v='vfd',
               kvh_in='kvh', kvm_in='kvm', kvh='kvh_out', kvm='kvm_out',
               qrlin='qrl', turbtype_f='turbtype', gh_a='ghi', sh_a='shi',
               sm_a='smi', ri_a='rii', leng='lengi', cld='cldn')
for _a in ('wet web jtbu jbbu evhc jt2slv n2ht n2hb lwp opt_depth radinvfrac radf wstar wstar3fact').split():
    ALIASES[_a+'_CL'] = _a
OUT = {
    'exacol': 'ktop kbase ncvfin'.split(),
    'zisocl': 'ncvfin kbase ktop belongcv ricl ghcl shcl smcl lbrk wbrk ebrk extend extend_up extend_dn'.split(),
    'caleddy': ('kvh kvm pblh pblhp tpert qpert wpert tke bprod sprod turbtype_f sm_aw ipbl kpblh tkes '
                'kbase_o ktop_o ncvfin_o kbase_mg ktop_mg ncvfin_mg kbase_f ktop_f ncvfin_f '
                'wet_cl web_cl jtbu_cl jbbu_cl evhc_cl jt2slv_cl n2ht_cl n2hb_cl lwp_cl opt_depth_cl '
                'radinvfrac_cl radf_cl wstar_cl wstar3fact_cl ebrk wbrk lbrk ricl ghcl shcl smcl '
                'gh_a sh_a sm_a ri_a leng wcap').split(),
}
OUT['caleddy']=[a.replace('_cl','_CL') for a in OUT['caleddy']]

def fixtures():
    directory = os.environ.get('UWPBL_ORACLE_DIR')
    if not directory: pytest.skip('UWPBL_ORACLE_DIR is unset')
    paths = sorted(Path(directory).glob('cases-*-spy.manifest'))
    if not paths: pytest.skip('No stage fixtures in UWPBL_ORACLE_DIR')
    return [load(p) for p in paths]

def records(fx, name):
    groups={}
    for k,v in fx.arrays.items():
        for side in ('in','out'):
            token='/'+name+'_'+side+'/'
            if token in k:
                key,a=k.split(token)
                groups.setdefault(key,{'in':{},'out':{}})[side][a]=v
    for key,group in sorted(groups.items()):
        yield key,group['in'],group['out']

def prepare(name, ins):
    pver=len(ins['zi'])-1
    data={}
    for a in inspect.signature(getattr(ref,name)).parameters:
        if a=='pver': data[a]=pver
        elif a=='wstarent': data[a]=True
        elif a=='kvf': data[a]=[0.0]*(pver+2)  # Dead input, use_kvf=.false., F:1716.
        else:
            label=ALIASES.get(a,a) if name=='caleddy' else a
            assert label in ins,(name,a,label)
            data[a]=[0]+ins[label].tolist()
    return data

def diff(actual, expected):
    actual=np.asarray(actual,dtype=expected.dtype)
    if expected.dtype.kind=='f':
        a=actual.view(np.uint64); b=expected.view(np.uint64)
        # Ordered integer words across signed zero and negative numbers.
        ao=np.where(a>>63,~a,a|(np.uint64(1)<<np.uint64(63)))
        bo=np.where(b>>63,~b,b|(np.uint64(1)<<np.uint64(63)))
        distance=np.maximum(ao,bo)-np.minimum(ao,bo)
        return int(np.count_nonzero(a!=b)),int(distance.max(initial=0))
    return int(np.count_nonzero(actual!=expected)),0

def replay_cpu(name, fx):
    mismatches=[]; count=words=0
    for key,ins,outs in records(fx,name):
        data=prepare(name,ins)
        getattr(ref,name)(**data)
        count+=1
        for a in OUT[name]:
            label=ALIASES.get(a,a) if name=='caleddy' else a
            expected=outs[label]
            actual=data[a][1:]
            # All these entries are initialized/written, see F:1677-1727,
            # F:1792-1795, F:1873-1876, F:2108-2111 and F:3305-3312.
            assert len(actual)==len(expected),(name,a,len(actual),len(expected))
            bad,ulp=diff(actual,expected); words+=len(expected)
            if bad: mismatches.append((key,a,bad,ulp))
    return dict(records=count,words=words,mismatches=mismatches)

@pytest.mark.parametrize('name',['exacol','zisocl','caleddy'])
def test_cpu(name):
    results={str(f.path):replay_cpu(name,f) for f in fixtures()}
    print(json.dumps(results,indent=2))
    assert all(not r['mismatches'] for r in results.values())

def gpu_module(cp):
    # Prefer the real library when the lead supplies it. Stub results remain
    # failures, never treated as equal by tolerance.
    lib=ROOT/'gpuwm/core/kernels/glibc_flt64.cuh'
    if not lib.exists(): lib=ROOT/'gpuwm/core/kernels/uwpbl_libm_stub.cuh'
    pieces=[lib]+[ROOT/'gpuwm/core/kernels'/p for p in ('uwpbl_common.cuh','uwpbl_zisocl.cuh','uwpbl_caleddy.cuh')]
    source='\n'.join(p.read_text() for p in pieces)
    signatures={}
    for name in ('exacol','zisocl','caleddy'):
        m=re.search(r'__device__ void uw_'+name+r'\((.*?)\) \{',source)
        signatures[name]=[tuple(s.strip().split()) for s in m[1].split(',')][:-1]
    wrappers=[]
    wrappers.append('extern "C" __global__ void uw_trig_probe(int fn,double x,double* y){y[0]=fn?uw_acos(R8(x)).v:uw_cos(R8(x)).v;}')
    for name,sig in signatures.items():
        setup=[]; actual=[]; save=[]; ro=io=0
        for typ,a in sig:
            if a=='pver': actual.append('pver'); continue
            if a=='wstarent': actual.append('true'); continue
            integral=typ.startswith(('int','bool','VI','VL'))
            pool='ip' if integral else 'rp'; offset=io if integral else ro
            if typ in ('V','VI','VL'): actual.append(typ+'{'+pool+' + '+str(offset)+' * stride + col, stride}')
            else:
                t=typ.rstrip('&'); setup.append(t+' '+a+' = '+('R8(' if t=='R8' else '(')+pool+'['+str(offset)+' * stride + col]);')
                actual.append(a)
                if '&' in typ: save.append(pool+'['+str(offset)+' * stride + col] = '+a+('.v' if t=='R8' else '')+';')
            if integral: io+=128
            else: ro+=128
        wrappers.append('extern "C" __global__ void replay_'+name+'(int pver, int stride, double* rp, int* ip, double* wr, int* wi, int* err) { int col=blockDim.x*blockIdx.x+threadIdx.x; if(col>=stride)return; Ws ws{wr+col,wi+col,stride,0,8192,0,2048,err+col}; '+''.join(setup)+' uw_'+name+'('+','.join(actual)+',ws); '+''.join(save)+'}')
    module=cp.RawModule(code=source+'\n'+'\n'.join(wrappers),options=('-std=c++17',))
    from gpuwm.core.uwpbl_constants import ESTBL
    table=module.get_global('UW_ESTBL'); table.copy_from_host(np.asarray(ESTBL,dtype=np.float64).ctypes.data,2000)
    return module,signatures,str(lib)

def replay_gpu(cp,module,signatures,name,fx):
    rec=list(records(fx,name)); count=len(rec)
    if not count: return dict(records=0,words=0,mismatches=[])
    pver=len(rec[0][1]['zi'])-1
    sig=signatures[name]; nr=sum(not t.startswith(('int','bool','VI','VL')) for t,a in sig if a not in ('pver','wstarent'))*128
    ni=sum(t.startswith(('int','bool','VI','VL')) for t,a in sig if a not in ('pver','wstarent'))*128
    rp=np.zeros((nr,count),dtype=np.float64); ip=np.zeros((max(ni,1),count),dtype=np.int32)
    offsets={}; ro=io=0
    for t,a in sig:
        if a in ('pver','wstarent'): continue
        integral=t.startswith(('int','bool','VI','VL')); offset=io if integral else ro
        offsets[a]=(integral,offset)
        for col,(_,ins,_) in enumerate(rec):
            data=prepare(name,ins)[a][1:]; (ip if integral else rp)[offset:offset+len(data),col]=data
        if integral: io+=128
        else: ro+=128
    dr=cp.asarray(rp); di=cp.asarray(ip); err=cp.zeros(count,dtype=cp.int32)
    module.get_function('replay_'+name)(((count+63)//64,),(64,),(np.int32(pver),np.int32(count),dr,di,cp.empty((8192,count),dtype=cp.float64),cp.empty((2048,count),dtype=cp.int32),err))
    assert not cp.asnumpy(err).any(),cp.asnumpy(err)
    rp=cp.asnumpy(dr); ip=cp.asnumpy(di); mismatch=[]; words=0
    residual=[]; trig_rows=[]; cache={}; probe=module.get_function('uw_trig_probe'); output=cp.empty(1,dtype=cp.float64)
    originals={fn:getattr(math,fn) for fn in ('cos','acos')}
    for col,(key,_,outs) in enumerate(rec):
        differences=[]
        def hook(fn):
            def evaluate(x):
                token=(fn,x.hex())
                if token not in cache:
                    probe((1,),(1,),(np.int32(fn=='acos'),np.float64(x),output))
                    cache[token]=float(cp.asnumpy(output)[0])
                gpu=cache[token]; glibc=originals[fn](x)
                row=dict(grid=fx.path.name,record=key,function=fn,argument=x.hex(),glibc=glibc.hex(),gpu=gpu.hex(),equal=glibc.hex()==gpu.hex())
                trig_rows.append(row)
                if not row['equal']: differences.append(row)
                return gpu
            return evaluate
        data=prepare(name,rec[col][1])
        for fn in originals: setattr(math,fn,hook(fn))
        try: getattr(ref,name)(**data)
        finally:
            for fn,original in originals.items(): setattr(math,fn,original)
        for a in OUT[name]:
            label=ALIASES.get(a,a) if name=='caleddy' else a; expected=outs[label]
            integral,off=offsets[a]; actual=(ip if integral else rp)[off:off+len(expected),col]
            bad,ulp=diff(actual,expected); words+=len(expected)
            if bad:
                attributable,remaining=diff(actual,np.asarray(data[a][1:],dtype=expected.dtype))
                if differences and not attributable:
                    residual.append((key,a,bad,ulp,differences))
                else: mismatch.append((key,a,bad,ulp))
    evidence=os.environ.get('UWPBL_GPU_EVIDENCE_DIR')
    if evidence:
        dest=Path(evidence); dest.mkdir(parents=True,exist_ok=True)
        (dest/(fx.path.name+'-'+name+'-trig.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in trig_rows))
        (dest/(fx.path.name+'-'+name+'-residual.json')).write_text(json.dumps(residual,indent=2))
    return dict(records=count,words=words,mismatches=mismatch,trig_residual_outputs=len(residual),trig_calls=len(trig_rows))

@pytest.mark.gpu
@pytest.mark.parametrize('name',['exacol','zisocl','caleddy'])
def test_gpu(name):
    """Every record bitwise, except outputs a cos/acos residual explains.

    The kernel's cos/acos are correctly rounded; WRF's are glibc's, which are
    not (glibc_flt64.cuh's header).  A mismatching output is accepted only
    when the CPU reference, replayed with the kernel's own trig words at
    exactly the arguments this record reaches, reproduces the kernel's words
    and at least one of those trig words differs from glibc's.  Anything
    else -- exp/log/pow, arithmetic, control flow -- fails.
    """
    import cupy  # noqa: F401  (marks this test for -m "not gpu")
    cp=pytest.importorskip('cupy')
    module,sig,lib=gpu_module(cp)
    results={str(f.path):replay_gpu(cp,module,sig,name,f) for f in fixtures()}
    print(json.dumps(dict(library=lib,results=results),indent=2))
    assert all(not r['mismatches'] for r in results.values())
