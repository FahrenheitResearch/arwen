"""Bitwise CUDA sweep. Only the C scalar libm calls supply exp/log/pow references.
Run under the OWNER protocol. numpy is used only for buffers and bit comparison.
"""
import os,time,json,ctypes as ct,argparse
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ.setdefault('CUPY_CACHE_DIR',str(root/'cupy-cache'))
os.environ.setdefault('TMPDIR',str(root))
import numpy as np
import cupy as cp
p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=1<<28);p.add_argument('--chunk',type=int,default=1<<20);args=p.parse_args()
lib=ct.CDLL(str(root/'reference_batch.so'));lib.g64_batch.argtypes=[ct.c_int,ct.c_uint64,ct.c_uint64]+[ct.c_void_p]*3
source=(root/'glibc_flt64.cuh').read_text()+r'''
extern "C" __global__ void sweep(int f,const double *x,const double *y,double *o,unsigned n){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 o[i]=f==0?glibc_exp(x[i]):f==1?glibc_log(x[i]):f==2?glibc_pow(x[i],y[i]):f==3?uw_cos(x[i]):uw_acos(x[i]);
}
'''
mod=cp.RawModule(code=source,options=('-std=c++17',));kernel=mod.get_function('sweep')
results=[]
for f,name in enumerate(['exp','log','pow','cos','acos']):
 started=time.monotonic();mismatches=0;examples=[]
 for start in range(0,args.count,args.chunk):
  n=min(args.chunk,args.count-start);x=np.empty(n,np.float64);y=np.empty_like(x);ref=np.empty_like(x)
  lib.g64_batch(f,start,n,x.ctypes.data,y.ctypes.data,ref.ctypes.data)
  dx=cp.asarray(x);dy=cp.asarray(y);out=cp.empty_like(dx)
  kernel(((n+255)//256,),(256,),(np.int32(f),dx,dy,out,np.uint32(n)))
  got=cp.asnumpy(out);idx=np.flatnonzero(got.view(np.uint64)!=ref.view(np.uint64));mismatches+=len(idx)
  for i in idx[:max(0,20-len(examples))]:
   examples.append({'i':start+int(i),'x':hex(int(x.view(np.uint64)[i])),'y':hex(int(y.view(np.uint64)[i])),'reference':hex(int(ref.view(np.uint64)[i])),'gpu':hex(int(got.view(np.uint64)[i]))})
  if time.monotonic()-started>540:raise RuntimeError('bounded hold near expiry; stop and release')
 r={'function':name,'samples':args.count,'mismatches':mismatches,'seconds':time.monotonic()-started,'examples':examples};results.append(r);print(json.dumps(r),flush=True)
(root/'gpu-results.json').write_text(json.dumps(results,indent=2))
raise SystemExit(any(r['mismatches'] for r in results))
