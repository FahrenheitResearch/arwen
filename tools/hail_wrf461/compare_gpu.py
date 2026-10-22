"""Independent WRF Fortran versus forecast CUDA entry. Run on a leased GPU."""
import argparse,hashlib,json,struct,subprocess,time
from pathlib import Path
import cupy as cp
import numpy as np
from gpuwm.core.kernels import get_kernel
p=argparse.ArgumentParser();p.add_argument("evidence",type=Path);p.add_argument("--reference",type=Path);a=p.parse_args()
a.evidence.mkdir(exist_ok=True,parents=True)
n,nz=256,64
src=r"""
extern "C" __global__ void profile(float* a,float* ht,float* dur,int n,int nz) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 float surface=286.f+float(c%16)*1.4f,peak=12.f+float(c/16)*2.2f;
 for(int k=0;k<nz;k++){
  float z=100.f+float(k)*240.f,t=surface-.0065f*z,p=100000.f*expf(-z/8200.f);
  float sat=611.2f*expf(17.67f*(t-273.155f)/(t-29.655f));
  float values[11]={t,z,p,p/(287.f*t),.62197f*sat/(p-sat)*.95f,
   z>5000.f?.0003f:0.f,z>1500.f&&z<10000.f?.003f:0.f,
   z<6000.f?.001f:0.f,z>6000.f?.0005f:0.f,z>3000.f?.001f:0.f,
   peak*expf(-powf((z-6000.f)/4000.f,2.f))};
  for(int v=0;v<11;v++) a[(v*nz+k)*n+c]=values[v];
 }ht[c]=0;dur[c]=1000.f+float(c%5)*250.f;
}
"""
x=cp.empty((11,nz,n),cp.float32);ht=cp.empty(n,cp.float32);dur=cp.empty(n,cp.float32)
cp.RawKernel(src,"profile")((2,),(128,),(x,ht,dur,np.int32(n),np.int32(nz)))
host=x.get();h=ht.get();d=dur.get()
input_path=a.evidence/"oracle_input.bin"
with input_path.open("wb") as f:
 f.write(struct.pack("ii",n,nz))
 for c in range(n):
  f.write(host[:,:,c].tobytes());f.write(struct.pack("ff",h[c],d[c]))
out=a.evidence/"oracle_output.bin"
start=time.monotonic()
if a.reference:
 out=a.reference
else:
 subprocess.run([str(a.evidence/"hail_oracle"),str(input_path),str(out)],check=True,timeout=120)
oracle_wall=time.monotonic()-start
emb=cp.zeros((5,n),cp.float32);mx=cp.zeros(n,cp.float32)
mean=cp.zeros_like(mx);std=cp.zeros_like(mx);run=cp.zeros_like(mx)
kernel=get_kernel("hail","hailcast_columns")
start=time.monotonic()
kernel((8,),(32,),(*tuple(x[v] for v in range(11)),ht,dur,*(emb[j] for j in range(5)),mx,mean,std,run,np.int32(n),np.int32(1),np.int32(nz)))
cp.cuda.Stream.null.synchronize()
wall=time.monotonic()-start
result=emb.get().T;ref=np.fromfile(out,np.float32).reshape(n,5)
delta=np.abs(result-ref);tol=.1+.005*np.abs(ref)
receipt={"reference_location":"separate CPU host" if a.reference else "GPU host",
 "input_sha256":hashlib.sha256(input_path.read_bytes()).hexdigest(),
 "output_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"columns":n,"levels":nz,"oracle_wall_s":None if a.reference else oracle_wall,"gpu_wall_s":wall,
 "max_absolute_error_mm":float(delta.max()),"comparison_failures":int((delta>tol).sum()),
 "finite_gpu":bool(np.isfinite(result).all()),"positive_oracle":int((ref>0).sum()),
 "maximum_oracle_mm":float(ref.max()),"maximum_gpu_mm":float(result.max()),
 "tolerance":"0.1 mm + 0.5 percent reference",
 "peak_used_bytes":int(cp.get_default_memory_pool().used_bytes())}
(a.evidence/"oracle_compare.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2),flush=True)
assert receipt["finite_gpu"] and receipt["positive_oracle"]>0
assert receipt["comparison_failures"]==0
