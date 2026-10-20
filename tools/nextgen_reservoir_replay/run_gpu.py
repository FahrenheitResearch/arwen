"""Offline CUDA replay; shared Rust raw-I/O bridge handles payload bytes."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-cuda", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    import cupy as cp
    from numpy import int32
    from gpuwm.hydro.reservoirs import Reservoirs, kernel_source, PARAMETERS, CONTROLS, SOURCE_PIN
    from gpuwm.core.ieee_module import compile_module
    spec = importlib.util.spec_from_file_location("shared_raw_cuda", args.raw_cuda)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    io = mod.RawCuda(args.bridge)
    args.output.mkdir(parents=True, exist_ok=True)
    count = (args.fixture / "input.f32").stat().st_size // 56
    stages = {}
    start = time.perf_counter()
    def load(name, shape):
        return io.load({"path":name,"dtype":"f32","order":"c","shape":shape,
                        "sha256":io.sha256(args.fixture/name)}, args.fixture)
    captured = load("input.f32", [14,count])
    result = cp.empty((2,count), dtype="float32")
    strict = compile_module(kernel_source(True), ("--std=c++17", "--ftz=false", "--fmad=false", "--prec-div=true", "--prec-sqrt=true"), "nextgen_reservoir_replay")
    stages["setup_seconds"] = time.perf_counter()-start
    start=time.perf_counter()
    strict.get_function("ngen_reservoir_replay")(((count+127)//128,),(128,), (int32(count),captured,result))
    cp.cuda.get_current_stream().synchronize()
    stages["independent_replay_ms_boundary_included"]=(time.perf_counter()-start)*1e3
    io.store(result,args.output/"actual.f32",order="c")
    def compare(name, actual):
        request={"schema":"nextgen.raw.compare/v1","unit":"reservoirs",
                 "case":"t-route-shipped-Pocono-4185105-287x300s-"+name,"numerics":"strict",
                 "candidate_revision":args.revision,"command":["run_gpu.py",str(args.fixture),str(args.output)],
                 "provenance":{"origin":"native-module","sources":[{"name":"NOAA-OWP/t-route","revision":SOURCE_PIN,"licence":"Apache-2.0"}],
                    "evidence":[{"path":str(args.native_receipt),"sha256":io.sha256(args.native_receipt)}]},
                 "fields":[{"name":"height_m/outflow_m3_per_s","dtype":"f32","shape":[2,count],"order":"c",
                     "expected":{"path":str(args.fixture/"expected.f32"),"sha256":io.sha256(args.fixture/"expected.f32")},"actual":str(actual)}]}
        path=args.output/(name+"-request.json");path.write_text(json.dumps(request,indent=2)+"\n")
        result=subprocess.run([args.comparator,str(path),str(args.output/(name+"-verdict.json"))],capture_output=True,text=True)
        print(result.stdout,result.stderr)
        return result
    independent=compare("independent",args.output/"actual.f32")

    # All data-array initialization and time-dependent feeding below execute
    # on CUDA. Python only constructs fixed views and sequences launches.
    a={}
    a["parameters"]=load("parameters.f32",[9,1]);a["height"]=load("height.f32",[1])
    for name in ("previous_inflow","inflow","lateral","initial_height","levelpool_release","levelpool_height"):
        a[name]=cp.zeros(1,dtype="float32")
    a["types"]=cp.ones(1,dtype="int32")
    for name in ("indices","dynamic_type","active_source","fault"):
        a[name]=cp.zeros(1,dtype="int32")
    a["offsets"]=cp.zeros(2,dtype="int64")
    a["controls"]=cp.zeros((8,1),dtype="float64")
    a["carriers"]=cp.zeros((3,1),dtype="float64")
    for name in ("clock","release","assimilated","series","times"):
        a[name]=cp.zeros(1,dtype="float64")
    metadata={"source_pin":SOURCE_PIN,"parameter_columns":PARAMETERS,"control_columns":CONTROLS,
              "validated_csr":True,"network_sha256":"shipped-Pocono-single-reservoir",
              "parameters_sha256":io.sha256(args.fixture/"parameters.f32"),
              "series_sha256":io.sha256(args.fixture/"input.f32"),
              "time_origin":"shipped-test-first-inflow","reservoir_order_sha256":"4185105"}
    resident=Reservoirs(a,metadata,strict=True).subset(cp.zeros(1,dtype="int32"),selection_sha256="Pocono-row-zero")
    helpers=compile_module(r'''
extern "C" __global__ void feed(int n,const float* captured,const double* clock,float* prev,float* now) {
 if(threadIdx.x==0) {int j=int(clock[0]/300.);prev[0]=now[0]=captured[n+j];}
}
extern "C" __global__ void record(int n,const double* clock,const float* h,const double* q,float* out) {
 if(threadIdx.x==0) {int j=int(clock[0]/300.)-1;out[j]=h[0];out[n+j]=float(q[0]);}
}
''', ("--std=c++17","--ftz=false"), "nextgen_reservoir_fixture_feed")
    feed,record=helpers.get_function("feed"),helpers.get_function("record")
    trajectory=cp.empty((2,count),dtype="float32")
    cp.cuda.get_current_stream().synchronize()
    start=time.perf_counter()
    for _ in range(count):
        feed((1,),(1,),(int32(count),captured,a["clock"],a["previous_inflow"],a["inflow"]))
        resident.launch(300.)
        record((1,),(1,),(int32(count),a["clock"],a["height"],a["release"],trajectory))
    cp.cuda.get_current_stream().synchronize()
    stages["strict_resident_trajectory_ms"]=(time.perf_counter()-start)*1e3
    io.store(trajectory,args.output/"trajectory.f32",order="c")
    carried=compare("resident",args.output/"trajectory.f32")
    # Compile fast separately and time actual two-kernel component launch,
    # with fixed first-case parameters/inflow and no diagnostic record/feed.
    fast=Reservoirs(a,metadata,strict=False)
    cp.cuda.get_current_stream().synchronize()
    start=time.perf_counter()
    for _ in range(count): fast.launch(300.)
    cp.cuda.get_current_stream().synchronize()
    stages["fast_component_batch_ms"]=(time.perf_counter()-start)*1e3
    # Separate event-instrumented single update, never part of throughput.
    from numpy import float32, float64
    events=[cp.cuda.Event() for _ in range(3)]
    events[0].record()
    fast._levelpool(fast._grid,fast._block,(*fast._selection_args,*fast._lp_args,float32(300.),*fast._lp_output))
    events[1].record()
    fast._assimilate(fast._grid,fast._block,(*fast._selection_args,*fast._da_input,float64(300.),*fast._da_output))
    events[2].record();events[2].synchronize()
    stages["separate_single_update_profile_ms"]={"levelpool":cp.cuda.get_elapsed_time(events[0],events[1]),"assimilation_commit":cp.cuda.get_elapsed_time(events[1],events[2])}
    receipt={"source_pin":SOURCE_PIN,"candidate_revision":args.revision,
             "scope":"t-route shipped Pocono lake4185105 network inflows; not LowerColorado/coupled acceptance",
             "card":cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"].decode(),
             "calls":count,"compared_float32_words":count*2,
             "independent_bitwise_match":independent.returncode==0,
             "resident_trajectory_bitwise_match":carried.returncode==0,
             "comparison_command":[args.comparator,"*-request.json","*-verdict.json"],
             "comparison_output":independent.stdout+carried.stdout,
             "parallelism":"independent replay: one GPU thread/call in 128-thread blocks; resident: one active reservoir thread in 128-thread block",
             "cuda_graphs":False,"warmup_steps":0,"first_launch_included":True,
             "timing":stages,"fast_component_ms_per_step":stages["fast_component_batch_ms"]/count,
             "fast_timing_scope":"two direct launches/update, fixed final captured inflow, continuing carried state; one final batch sync; no per-step events, copies, allocations, sync or host numeric processing",
             "strict_trajectory_timing_scope":"four launches/update including device fixture feed and device output recording; not production speed",
             "raw_io_dependency":"fresh shared NextGen Rust bridge/comparator 8f6a99aba001; no withdrawn code/source/fixture/ABI",
             "inputs_sha256":io.sha256(args.fixture/"input.f32"),
             "expected_sha256":io.sha256(args.fixture/"expected.f32"),
             "actual_sha256":io.sha256(args.output/"actual.f32"),
             "trajectory_sha256":io.sha256(args.output/"trajectory.f32")}
    (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
    if independent.returncode or carried.returncode: raise SystemExit(1)


if __name__=="__main__": main()
