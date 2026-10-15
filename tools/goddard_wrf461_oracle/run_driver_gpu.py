"""Full preparation, LUT, two-band solve and driver publication against WRF."""
import argparse,gzip,json,struct
from pathlib import Path
import numpy as np
import cupy as cp
from gpuwm.core.kernels import get_kernel_int_defines
from gpuwm.core.radiation_tables import radiation_table_words
from gpuwm.wrf_exact import ENABLED

p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--mutate-band',type=int,choices=(1,2));a=p.parse_args()
f=Path(__file__).parent/'fixtures'
raw=gzip.decompress((f/'driver-inputs.bin.gz').read_bytes())
ref=gzip.decompress((f/'driver-reference.bin.gz').read_bytes())
if struct.unpack('<4i',raw[:16])!=(0x47524936,64,32,4):raise ValueError('driver input header')
if struct.unpack('<4i',ref[:16])!=(0x47524436,64,32,4):raise ValueError('driver output header')
n=32;c=16;at=16;rt=16;fields={};success=True
defines=(('GDD_MAX_LEVELS',n),)+( (('GDD_MUTATE_BAND',a.mutate_band),) if a.mutate_band else ())
kernel=lambda name:get_kernel_int_defines('goddard',name,defines)
table=radiation_table_words('goddard-cloud')
names=['RTHRATENLW','RTHRATENSW','SWDOWN','GLW','GSW','OLR','SWDDNI','SWDDIR','SWDDIF','TLWDN','TLWUP','SLWDN','SLWUP','TSWDN','TSWUP','SSWDN','SSWUP']
results=[];references=[]
for batch in range(4):
    nr=(20*n+7)*c;raw_d=cp.asarray(np.frombuffer(raw,dtype='<f4',count=nr,offset=at));at+=nr*4
    band=cp.empty((5*n+15)*c,cp.float32);flags=cp.empty(3*c,cp.int32)
    dz=cp.empty(n*c,cp.float32);q=cp.empty(6*n*c,cp.float32);re=cp.empty_like(q)
    kernel('goddard_prepare_columns')((1,),(c,),(raw_d,band,flags,dz,q,re,np.int32(n),np.int32(c),np.int32(160)))
    optics=cp.empty(63*n*c,cp.float32)
    kernel('goddard_cloud_columns')(((n*c+127)//128,),(128,),(dz,q,re,table,optics,np.int32(n),np.int32(c)))
    flux=cp.empty((6*(n+1)+4)*c,cp.float32)
    kernel('goddard_bands_columns')((1,),(c,),(band,flags,optics,flux,np.int32(n),np.int32(c)))
    output=cp.empty((2*n+15)*c,cp.float32)
    kernel('goddard_publish_columns')((1,),(c,),(band,flux,raw_d,output,np.int32(n),np.int32(c),np.float32(1365)))
    results.append(output.get().reshape(2*n+15,c))
    references.append(np.frombuffer(ref,dtype='<f4',count=(2*n+15)*c,offset=rt).reshape(2*n+15,c));rt+=(2*n+15)*c*4
x=np.concatenate(results,axis=1);y=np.concatenate(references,axis=1)
if at!=len(raw) or rt!=len(ref):raise ValueError('driver byte count')
def ordered(x):
    b=x.view(np.uint32).astype(np.int64)
    return np.where(b&0x80000000,0xffffffff-b,b+0x80000000)
for name,lo,hi in [(names[0],0,n),(names[1],n,2*n)]+[(name,2*n+i,2*n+i+1) for i,name in enumerate(names[2:])]:
    a0=x[lo:hi];r=y[lo:hi]
    mismatch=a0.view(np.uint32)!=r.view(np.uint32)
    error=np.abs(a0.astype(np.float64)-r.astype(np.float64))
    tolerance=(1.e-7 if lo<2*n else .1)+1.e-4*np.abs(r.astype(np.float64))
    passed=bool(np.isfinite(a0).all() and np.isfinite(r).all() and (not mismatch.any() if ENABLED else (error<=tolerance).all()))
    success &= passed
    fields[name]={'words':int(a0.size),'different_words':int(mismatch.sum()),'max_ulp':int(np.abs(ordered(a0)-ordered(r)).max()),'max_abs':float(error.max()),'pass':passed}
receipt={'strict':ENABLED,'columns':64,'layers':n,'full_driver_pass':success,'fields':fields,'mutation_band':a.mutate_band}
a.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(0 if (not success if a.mutate_band else success) else 1)
