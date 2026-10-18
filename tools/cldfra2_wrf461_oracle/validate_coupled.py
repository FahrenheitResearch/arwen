"""Exercise the real CUDA preparation and LW/SW engines against public WRF."""
import argparse, inspect, json, os
from pathlib import Path
import numpy as np
import cupy as cp
from gpuwm.core import rrtmgp,rrtmg_legacy as leg,rrtmg_legacy_device as dev
from gpuwm.core import rrtmg_lw,rrtmg_sw,rrtmg_legacy_prep as prep
from tools.rrtmg_wrf461_oracle.lw_fixtures import read_fixture
from tools.rrtmg_wrf461_oracle.sw_dumpio import read_swd
from test_rrtmg_legacy_prep import _lw_prep_kwargs,_sw_prep_kwargs

p=argparse.ArgumentParser(); p.add_argument('root',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args()
fields={}; rows=[]
def compare(name, got, want, case):
 got=np.asarray(cp.asnumpy(got) if isinstance(got,cp.ndarray) else got,dtype='f4')
 want=np.asarray(want,dtype='f4')
 assert got.shape==want.shape,(name,got.shape,want.shape)
 gu=got.view('u4'); wu=want.view('u4')
 from gpuwm.core.fp32_ulp import monotone_fp32_key
 ulp=np.abs(monotone_fp32_key(got)-monotone_fp32_key(want))
 r=fields.setdefault(name,dict(max_ulp=0,max_abs=0.0,mismatched_words=0,words=0))
 r['max_ulp']=max(r['max_ulp'],int(np.max(ulp,initial=0)))
 r['max_abs']=max(r['max_abs'],float(np.max(np.abs(got-want),initial=0)))
 r['mismatched_words']+=int(np.count_nonzero(gu!=wu));r['words']+=int(got.size)
 if np.any(gu!=wu):rows.append(dict(case=case,field=name,max_ulp=int(np.max(ulp))))
def device_kwargs(kw):
 return {k:cp.asarray(v[None,:]) if isinstance(v,np.ndarray) and v.ndim else
         cp.asarray(np.atleast_1d(v)) if k in ('tsk','emiss','xland','xice','snow','xlat','albedo','xcoszen') else v
         for k,v in kw.items()}
def solve(fn, values, **kwargs):
 args=[]
 for name,param in inspect.signature(fn).parameters.items():
  if param.default is not inspect.Parameter.empty: continue
  args.append(values[name])
 return fn(*args,**kwargs)
C=leg._lw_coeffs(); sw=leg._cuda_sw(leg._sw_tables())
for path in sorted((a.root/'lw-results').glob('lw_case_*.bin')):
 fx=read_fixture(path);case=int(fx['meta/caseid']); group=((case-1)//8)%4
 fqc=group in (1,3); fqi=group in (2,3)
 kw=_lw_prep_kwargs(fx);kw.update(f_qc=fqc,f_qi=fqi)
 dkw=device_kwargs(kw)
 cf=rrtmgp.cal_cldfra2(dkw['qc3d'],dkw['qi3d'],f_qc=fqc,f_qi=fqi)
 compare('CLDFRA_LW',cf[0],fx['wrfin/cldfra'],case)
 dkw['cldfra3d']=cf
 pl=dev.lwrad_prep_batch_device(**dkw);pl['C']=C
 result=solve(rrtmg_lw.gpu_rrtmg_lw_batched_device,pl,mcica_layout='column')
 out=prep.lwrad_outputs_batch(**{k:result[k] for k in ('uflx','dflx','hr','uflxc','dflxc','hrc')},pi3d=cp.asarray(fx['wrap/pi3d'][None,:]))
 for n,key in [('GLW','glw'),('OLR','olr'),('RTHRATEN_LW','rthratenlw')]:compare(n,out[key][0],fx['wrap/'+key],case)
f=read_swd(a.root/'sw-results.swd')
for case in sorted({k.split('/')[0] for k in f}):
 kw=_sw_prep_kwargs(f,case)
 fqc=bool(f[case+'/in/f_qc']);fqi=bool(f[case+'/in/f_qi'])
 kw.update(f_qc=fqc,f_qi=fqi);dkw=device_kwargs(kw)
 cf=rrtmgp.cal_cldfra2(dkw['qc3d'],dkw['qi3d'],f_qc=fqc,f_qi=fqi)
 compare('CLDFRA_SW',cf[0],f[case+'/in/cldfra3d'],case)
 dkw['cldfra3d']=cf;ps=dev.swrad_prep_batch_device(**dkw)
 ps['adjes']=np.full(1,ps['adjes'],np.float32)
 result=solve(sw.rrtmg_sw_batched_device,ps,mcica_layout='column')
 host={k:cp.asnumpy(v)[0] for k,v in result.items()}
 # The real adapter's output kernels map the solved fluxes and rates.
 gsw=cp.empty(1,cp.float32); tendency=cp.empty((40,1),cp.float32)
 leg._adapter_kernel('rla_sw_out_grid')((1,),(128,),(
     np.int32(1),np.int32(40),np.int32(41),result['swdflx'],result['swuflx'],result['swhr'],
     cp.asarray(f[case+'/in/pi3d'][:,None]),np.float32(86400),np.int64(1),
     cp.asarray([0],dtype=cp.int64),tendency,gsw))
 down=cp.empty(1,cp.float32)
 leg._adapter_kernel('rla_swdown')((1,),(128,),(
     np.int64(1),gsw,cp.asarray([f[case+'/in/albedo']],dtype=cp.float32),down))
 want=np.float32(f[case+'/wrf/gsw']/np.float32(np.float32(1)-f[case+'/in/albedo']))
 compare('SWDOWN',down[0],want,case)
 compare('RTHRATEN_SW',tendency[:,0],f[case+'/wrf/rthratensw'],case)
receipt=dict(exact=os.environ.get('GPUWM_WRF_EXACT','0'),columns_lw=64,columns_sw=64,fields=fields,misses=rows,
    lw_flags='four F_QC/F_QI memberships',sw_flags='four F_QC/F_QI memberships; public SW wrapper used directly for false flags')
a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt['fields'],indent=2))
