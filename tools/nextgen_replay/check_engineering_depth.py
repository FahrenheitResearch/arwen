"""Execute the diagnostic factory on sealed real-DEM engineering depth frames.

SPDX-License-Identifier: Apache-2.0
Not an observed-event, native routing, or complete terrain advance validation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from raw_cuda import RawCuda


def run(args):
    import cupy as cp
    from gpuwm import numerics
    from gpuwm.nextgen.inundation import prepare

    fixture = json.loads(args.fixture.read_text())
    if fixture["schema"] != "nextgen.inundation.engineering-fixture/v1":
        raise ValueError("wrong engineering fixture schema")
    args.out.mkdir(parents=True, exist_ok=False)
    raw = RawCuda(args.library)
    stream = cp.cuda.Stream(non_blocking=True)
    started = time.perf_counter()
    specs = fixture["arrays"]
    fields = raw.load_many({k: specs[k] for k in ("initial_depth", "final_depth", "active", "healthy")}, stream=stream)
    source = SimpleNamespace(arrays={"surface_depth": fields["initial_depth"], "active": fields["active"],
                                    "health": fields["healthy"].view("uint32")},
                             restart_identity=fixture["source_identity"])
    with stream:
        diagnostic = prepare(settings={}, options={"terrain_component": "captured-terrain"},
                             state=None, physics=None, device_id=cp.cuda.runtime.getDevice(), mode="standalone",
                             components={"captured-terrain": source})
        cp.cuda.runtime.memcpyAsync(fields["initial_depth"].data.ptr, fields["final_depth"].data.ptr,
                                   fields["final_depth"].nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
        period = source.restart_identity["period_seconds"]
        diagnostic.advance(period, stream=stream)
    stream.synchronize()
    setup_ms = (time.perf_counter()-started)*1000
    pointers = {name:a.data.ptr for name,a in diagnostic.arrays.items()}
    started = time.perf_counter()
    for _ in range(64):
        diagnostic.advance(period, stream=stream)
    stream.synchronize()
    wall_ms = (time.perf_counter()-started)*1000
    if pointers != {name:a.data.ptr for name,a in diagnostic.arrays.items()}:
        raise AssertionError("diagnostic carriers moved")
    outputs = [("depth_m",diagnostic.arrays["depth_m"],"expected_depth"),
               ("maximum_depth_m",diagnostic.arrays["maximum_depth_m"],"expected_maximum"),
               ("health",diagnostic.arrays["health"].view("uint8"),"healthy")]
    raw.store_many([(a,args.out/f"{name}.raw") for name,a,_ in outputs],stream=stream)
    comparisons = []
    for name,_,expected in outputs:
        spec = specs[expected]
        comparisons.append(dict(name=name,dtype=spec["dtype"],shape=spec["shape"],order="c",
                                units="m" if name!="health" else "bit flags",
                                expected={"path":spec["path"],"sha256":spec["sha256"]},actual=str(args.out/f"{name}.raw")))
    request = dict(schema="nextgen.raw.compare/v1",unit="inundation-diagnostic",case="real-dem-engineering-two-frames",
                   numerics=numerics.mode(),candidate_revision=args.revision,
                   command=["check_engineering_depth.py","--fixture",str(args.fixture),"--library",str(args.library),
                            "--comparator",str(args.comparator),"--out",str(args.out),"--revision",args.revision],
                   provenance={"origin":"engineering-depth-replay","sources":[{"name":"original inundation diagnostic",
                       "revision":args.revision,"licence":"Apache-2.0"}],
                       "evidence":[{"path":str(args.fixture),"sha256":raw.sha256(args.fixture)}]+fixture["evidence"]},fields=comparisons)
    (args.out/"request.json").write_text(json.dumps(request,indent=2)+"\n")
    subprocess.run([str(args.comparator),str(args.out/"request.json"),str(args.out/"verdict.json")],check=True)
    card = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"]
    if isinstance(card,bytes): card=card.decode()
    execution = dict(schema="nextgen.inundation.engineering-check/v1",candidate_revision=args.revision,
        numerics=numerics.mode(),card=card,grid_shape=specs["active"]["shape"],stable_pointers=True,factory_executed=True,
        source_identity=source.restart_identity,restart_identity=diagnostic.restart_identity,
        setup_ms=setup_ms,batch_calls=64,batch_ms=wall_ms,diagnostic_call_ms=wall_ms/64,
        parallelism="GPU one thread/cell, 256 threads/block; Rust offline Rayon workers selected by RAYON_NUM_THREADS",
        cuda_graphs=False,per_call_syncs=0,per_call_events=0,shared_gpu_pool=True,
        scope="Only inundation diagnostic on captured imposed-water depths/public DEM. Repeated final-state observations measure diagnostic launches, not new physical advances. Source health reconstructed from sealed owner receipt, not live terrain. No native hydrologic replay, event, gauge skill or whole hydro speed claim.")
    (args.out/"execution.json").write_text(json.dumps(execution,indent=2)+"\n")
    print(json.dumps(execution,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("fixture","library","comparator","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--revision",required=True)
    run(parser.parse_args())
