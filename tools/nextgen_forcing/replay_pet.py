"""Replay native BMI captures on a pool-owned card; Rust I/O, CUDA comparisons.

Run with --help for paths. This is a verification tool, not product stepping.
"""
from __future__ import annotations
import argparse
import ctypes
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--records",type=Path,required=True)
    parser.add_argument("--rust-io",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--method",type=int,default=5)
    parser.add_argument("--wind-witness",action="store_true")
    args=parser.parse_args()
    import cupy as cp
    import numpy as np
    from gpuwm.hydro.pet import PreparedPET
    from gpuwm.hydro.forcing import FIELDS
    started=time.perf_counter()
    io=ctypes.CDLL(str(args.rust_io))
    read=io.arwen_forcing_read_f64
    read.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t]
    read.restype=ctypes.c_int
    count=args.records.stat().st_size//8
    storage=(ctypes.c_double*count)()
    if read(bytes(args.records),storage,count): raise RuntimeError("Rust capture read failed")
    records=cp.asarray(np.ctypeslib.as_array(storage)).reshape(-1,39)
    fields={key:records[:,12+k].copy() for k,key in enumerate(FIELDS)}
    parameters=records[:,:12].copy()
    component=PreparedPET(fields,parameters,method=args.method,identity={"case":"pinned ngen forcing/native PET BMI calls","records":str(args.records)})
    compiled=time.perf_counter()
    component.advance(3600.0)
    cp.cuda.get_current_stream().synchronize()
    reference=records[:,32]
    output=component.fields["potential_et_m_s"]
    exact=int(cp.count_nonzero(output.view(cp.uint64)==reference.copy().view(cp.uint64)).item())
    ulps=cp.abs(output.view(cp.int64)-reference.copy().view(cp.int64))
    mismatch=cp.flatnonzero(ulps)
    state_mismatch=int(cp.count_nonzero(parameters.view(cp.uint64)!=records[:,20:32].copy().view(cp.uint64)).item())
    samples=[]
    for index in mismatch[:8].get().tolist():
        samples.append({"record":index,"gpu":float(output[index].item()),"native":float(reference[index].item()),"ulps":int(ulps[index].item()),"gpu_diagnostics":component.arrays["diagnostics"][index].get().tolist(),"native_diagnostics":records[index,33:39].get().tolist()})
    if args.wind_witness:
        from gpuwm import numerics
        import gpuwm.hydro.pet as pet_module
        here=Path(pet_module.__file__).parent
        code=(here/'pet.cu').read_text().split('// NVRTC')[0]
        if numerics.is_strict():
            code=(here.parent/'core/kernels/glibc_flt64.cuh').read_text()+'\n'+code+'\n#define log glibc_log\n'
        code+='''\nextern "C" __global__ void wind_probe(const long long n,const double*r,double*out){
          const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
          const double*p=r+i*39; const double w=pet_wind_norm(p[14],p[15]);
          const double a=log(2.0/p[3]),b=log(p[0]/p[3]);
          out[i*4]=w;out[i*4+1]=a;out[i*4+2]=b;out[i*4+3]=w*a/b;
        }'''
        witness=cp.empty((records.shape[0],4),dtype=cp.float64)
        cp.RawKernel(code,'wind_probe',options=numerics.effective_options(('--std=c++17','--fmad=false')))(((records.shape[0]+127)//128,),(128,),(np.int64(records.shape[0]),records,witness))
        for sample in samples:
            sample['wind_witness_hex']=[float(x).hex() for x in witness[sample['record']].get()]
    # Independent captures are a replay batch, not a full network speed claim.
    begin,end=cp.cuda.Event(),cp.cuda.Event()
    for _ in range(10):component.advance(3600.0)
    begin.record()
    for _ in range(200):component.advance(3600.0)
    end.record();end.synchronize()
    receipt={"method":args.method,"numerics":component.restart_identity,"source_pin":"6c410f2a8bc843043ad25e13239928031b8c5e64","card":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),"capture_records":records.shape[0],"exact_pet":exact,"max_pet_ulp":int(ulps.max().item()),"mutable_parameter_bit_mismatches":state_mismatch,"invalid_status":int(cp.count_nonzero(component.arrays["status"]).item()),"samples":samples,"parallelism":"128 threads/block, one captured catchment state per thread; float64; shared pool; contraction follows named numerics mode","pet_kernel_ms":cp.cuda.get_elapsed_time(begin,end)/200,"setup_seconds":compiled-started,"total_seconds":time.perf_counter()-started,"acceptance":"bitwise" if exact==records.shape[0] and state_mismatch==0 else "pending math witness / exact comparison","whole_step_timing":None}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
