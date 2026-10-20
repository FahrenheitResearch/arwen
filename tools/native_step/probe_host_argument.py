"""Current synchronized D2H scalar bits become a later kernel argument."""
import ctypes as C,json
from pathlib import Path
import cupy as cp
import numpy as np
from gpuwm.native_step import Plan
ROOT=Path(__file__).resolve().parents[2]
class Frame(C.Structure):_fields_=[('bits',C.c_uint32)]
k=cp.RawKernel('extern "C" __global__ void copy_scalar_bits(unsigned int x, unsigned int* out){out[0]=x;}','copy_scalar_bits')
output=cp.zeros(1,cp.uint32);k((1,),(1,),(np.uint32(0),output));cp.cuda.get_current_stream().synchronize()
cases=[]
for label,stream in [('default',cp.cuda.Stream.null),('nonblocking',cp.cuda.Stream(non_blocking=True))]:
 with stream:
  device=cp.asarray(np.array([11,29],np.uint32));plan=Plan()
  try:
   with plan.record(device,output,k,stream):
    with plan.readback() as reads:observed=cp.asnumpy(device)
    k((1,),(1,),(np.uint32(observed[1]),output))
   stream.synchronize();ops=plan.describe()['operations']
   matches=[x for x in ops if x['kind']==1 and x['kernel']=='copy_scalar_bits'];assert len(matches)==1,matches
   op=matches[0]['index'];plan.bind_host_argument(op,0,reads[0],byte_offset=4)
   observed[:]=999
   for bits in [0,0x80000000,0x7fc00123]:
    device.set(np.array([11,bits],np.uint32));plan.submit();stream.synchronize()
    got=int(output.get()[0]);assert got==bits and np.all(observed==999)
    cases.append({'case':label+' live exact scalar','bits':bits,'got':got,'capture_host_array_ignored':True})
   plan.bind(op,0,Frame,'bits');plan.submit(Frame(47));stream.synchronize();assert int(output.get()[0])==47
   cases.append({'case':label+' frame binding replaces readback','got':47})
   plan.bind_host_argument(op,0,reads[0],byte_offset=0);plan.submit();stream.synchronize();assert int(output.get()[0])==11
   cases.append({'case':label+' readback binding replaces frame','got':11})
  finally:plan.close()
r={'status':'PASS','cases':cases,'case_count':len(cases),'cupy':cp.__version__,'timing_claim':False,'model_claim':False}
(ROOT/'receipts/host-argument-gpu.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
