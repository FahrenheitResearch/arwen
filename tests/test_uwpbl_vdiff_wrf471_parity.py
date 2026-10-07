"""Every recorded inner and outer diffusion call, exact binary64 words."""
import os
from pathlib import Path
import numpy as np
import pytest
from gpuwm.verify.uwpbl_oracle import load
from gpuwm.verify.uwpbl_ref.vdiff import compute_vdiff

ROOT = Path(__file__).resolve().parents[1]

def words(actual, expected, label):
    a = np.asarray(actual,dtype=np.float64).view(np.uint64)
    b = np.asarray(expected,dtype=np.float64).view(np.uint64)
    bad = a != b
    if np.any(bad):
        # Monotonic IEEE word order handles negative numbers and signed zero.
        def ordered(x):
            return (~x & ((1<<64)-1)) if x>>63 else x | (1<<63)
        distances = [abs(ordered(int(x))-ordered(int(y))) for x,y in zip(a[bad],b[bad])]
        pytest.fail(f'{label}: mismatches={int(bad.sum())}/{a.size}, max_ULP={max(distances)}')

def source(kernel):
    folder = ROOT/'gpuwm/core/kernels'
    libm = 'glibc_flt64.cuh' if (folder/'glibc_flt64.cuh').exists() else 'uwpbl_libm_stub.cuh'
    return '\n'.join((folder/f).read_text(encoding='utf-8-sig') for f in [libm,'uwpbl_common.cuh','uwpbl_wvsat.cuh','uwpbl_vdiff.cuh'])+'\n'+kernel

def gpu_module(kernel):
    cp = pytest.importorskip('cupy')
    mod = cp.RawModule(code=source(kernel),options=('-std=c++17',))
    from gpuwm.core.uwpbl_constants import ESTBL
    ptr = mod.get_global('UW_ESTBL')
    cp.ndarray((250,),dtype=cp.float64,memptr=ptr).set(np.array(ESTBL))
    return cp,mod

def records():
    directory = os.environ.get('UWPBL_ORACLE_DIR')
    if not directory:
        pytest.skip('UWPBL_ORACLE_DIR is unset')
    alias_stem=ROOT/'tests/data/oracles/uwpbl/vdiff_alias'
    alias=load(alias_stem) if alias_stem.with_suffix('.bin').exists() else None
    for grid in ['g35','g44','g61']:
        stem = Path(directory)/f'cases-{grid}-spy'
        if not stem.with_suffix('.bin').exists():
            pytest.skip(f'missing spy fixture {stem}')
        fx = load(stem)
        prefixes = sorted({n.rsplit('/',1)[0] for n in fx.names() if '/vdiff_' in n and '_in/' in n})
        assert len(prefixes) == 18*4*5
        for prefix in prefixes:
            incoming = {n[len(prefix)+1:]:fx[n] for n in fx.names(prefix+'/')}
            op = prefix[:-3]+'_out'
            outgoing = {n[len(op)+1:]:fx[n] for n in fx.names(op+'/')}
            if alias is not None and 'in_loop' in prefix:
                incoming['jnk1d']=alias[grid+'/'+prefix+'/jnk1d']
                outgoing['jnk1d']=alias[grid+'/'+op+'/jnk1d']
            yield grid+'/'+prefix,incoming,outgoing

def mapped(d, inner):
    names = ['pmid','pint','rpdel','t','taux','tauy','shflx','cflx','kvh','kvm','kvq','cgs','cgh','zi','ksrftms','u','v','dse','tautmsx','tautmsy','dtk','topflx','tauresx','tauresy']
    aliases = dict(zip(names,names))
    aliases.update({'pmid':'pmid' if inner else 'pmid8','pint':'pi' if inner else 'pint8','rpdel':'rpdel' if inner else 'rpdel8','t':'t' if inner else 't8','cflx':'qflx' if inner else 'cflx_all','kvh':'kvh_out' if inner else 'kvh','kvm':'kvm_out' if inner else 'kvm','kvq':'kvh_out' if inner else 'kvq','zi':'zi' if inner else 'zi8','u':'ufd' if inner else 'wind_tends_1','v':'vfd' if inner else 'wind_tends_2','dse':'slfd' if inner else 'stnd','dtk':'jnk2d' if inner else 'dtk'})
    a = {n:[None]+d[key].tolist() for n,key in aliases.items() if key in d}
    if inner:
        a['kvq'] = a['kvh']
        a['tautmsx'] = a['tautmsy'] = a['topflx'] = [None,float(d['jnk1d'][0]) if 'jnk1d' in d else 0.0]
    a['q'] = [None]+[[None]+d['qtfd' if inner else f'cloudtnd_{m}'].tolist() for m in range(1,2 if inner else 6)]
    return a,aliases

def cpu(a, d, inner):
    pver,ncnst = len(a['t'])-1,len(a['q'])-1
    from gpuwm.core.uwpbl_constants import QMIN_Q
    fl = {'u':True,'v':True,'s':True,'q':[None]+[True]*ncnst}
    return compute_vdiff(pver,ncnst,a['pmid'],a['pint'],a['rpdel'],a['t'],float(d['ztodt'][0]),a['taux'],a['tauy'],a['shflx'],a['cflx'],1,pver,a['kvh'],a['kvm'],a['kvq'],a['cgs'],a['cgh'],a['zi'],a['ksrftms'],[None]+([0.0] if inner else [QMIN_Q,0.,0.,0.,0.]),fl,a['u'],a['v'],a['q'],a['dse'],a['tautmsx'],a['tautmsy'],a['dtk'],a['topflx'],a['tauresx'],a['tauresy'],0 if inner else 1)

def compare(a,aliases,out,label,inner):
    if inner and 'jnk1d' in out:
        words(a['tautmsy'][1:],out['jnk1d'],label+'/aliased jnk1d')
    for n in ['kvh','kvm','kvq','cgs','cgh','u','v','dse','dtk','tauresx','tauresy']+([] if inner else ['tautmsx','tautmsy','topflx']):
        words(a[n][1:],out[aliases[n]],label+'/'+n)
    for m in range(1,len(a['q'])):
        words(a['q'][m][1:],out['qtfd' if inner else f'cloudtnd_{m}'],label+f'/q{m}')

def test_cpu_all_records():
    count=0
    for label,d,out in records():
        inner = 'in_loop' in label
        a,aliases = mapped(d,inner)
        assert cpu(a,d,inner) == ''
        compare(a,aliases,out,label,inner);count+=1
    print(f'CPU diffusion records={count}, mismatches=0')

@pytest.mark.gpu
def test_gpu_all_records():
    # A packed argument buffer keeps the test kernel independent of the engine.
    entries = list(records())
    cp = pytest.importorskip('cupy')
    count=0
    for pver in [35,44,61]:
        subset = [x for x in entries if len(x[1]['t' if 'in_loop' in x[0] else 't8'])==pver]
        for inner in [True,False]:
            selected = [x for x in subset if ('in_loop' in x[0])==inner]
            a,_ = mapped(selected[0][1],inner)
            keys = list(a.keys());keys.remove('q')
            offsets={};offset=0
            for key in keys:
                offsets[key]=offset;offset+=len(a[key])-1
            ncnst=1 if inner else 5
            qo=offset;offset+=pver*ncnst
            kernel='extern "C" __global__ void grade(double* b,double* pool,int* flags,int n) {int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;double* x=b+c*'+str(offset)+'; int e=0;Ws ws{pool+c*2048,nullptr,1,0,2048,0,0,&e};'
            scalar={'taux','tauy','shflx','ksrftms','tautmsx','tautmsy','topflx','tauresx','tauresy'}
            for key in keys:
                off=offsets[key]
                if inner and key in ['tautmsy','topflx']:off=offsets['tautmsx']
                kernel+= ('R8& '+key+'=*reinterpret_cast<R8*>(x+'+str(off)+');' if key in scalar else 'V '+key+'{x+'+str(off)+',1};')
            kernel+=f'V q[{ncnst}];for(int m=0;m<{ncnst};++m)q[m]=V{{x+{qo}+m*{pver},1}};'
            kernel+= 'R8 qmin[5]={R8('+('0' if inner else 'UW_QMIN_Q')+'),R8(0),R8(0),R8(0),R8(0)};UwVdiffFields fl{true,true,true,{true,true,true,true,true}};'
            kernel+=f'uw_compute_vdiff({pver},{ncnst},pmid,pint,rpdel,t,R8(DT_PLACEHOLDER),taux,tauy,shflx,reinterpret_cast<R8*>(cflx.p),1,{pver},kvh,kvm,'+('kvh' if inner else 'kvq')+',cgs,cgh,zi,ksrftms,qmin,fl,u,v,q,dse,tautmsx,tautmsy,dtk,topflx,e,tauresx,tauresy,'+('0' if inner else '1')+',ws);flags[c]=e;}'
            kernel=kernel.replace('DT_PLACEHOLDER',float(selected[0][1]['ztodt'][0]).hex())
            cp,mod=gpu_module(kernel)
            packed=[]
            for label,d,out in selected:
                a,_=mapped(d,inner)
                packed.append(sum([a[k][1:] for k in keys],[])+sum([v[1:] for v in a['q'][1:]],[]))
            buf=cp.array(packed,dtype=cp.float64);pool=cp.empty((len(selected),2048),dtype=cp.float64);flags=cp.zeros(len(selected),dtype=cp.int32)
            mod.get_function('grade')(((len(selected)+31)//32,),(32,),(buf,pool,flags,np.int32(len(selected))))
            result=buf.get();assert not np.any(flags.get())
            for row,(label,d,out) in zip(result,selected):
                a,aliases=mapped(d,inner)
                for key in keys:a[key]=[None]+row[offsets[key]:offsets[key]+len(a[key])-1].tolist()
                if inner:a['tautmsy']=a['tautmsx']
                a['q']=[None]+[[None]+row[qo+m*pver:qo+(m+1)*pver].tolist() for m in range(ncnst)]
                compare(a,aliases,out,label,inner);count+=1
    print(f'GPU diffusion records={count}, mismatches=0')
