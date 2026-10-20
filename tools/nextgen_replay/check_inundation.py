"""Run the real CUDA diagnostic against Rust-created exact synthetic words.

SPDX-License-Identifier: Apache-2.0
This checks depth diagnostics only, not observed/native routing physics.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from raw_cuda import RawCuda


def run(args):
    import cupy as cp
    from gpuwm import numerics
    from gpuwm.nextgen.inundation import InundationDepth

    fixture = json.loads(args.fixture.read_text())
    specs = fixture["arrays"]
    args.out.mkdir(parents=True, exist_ok=False)
    raw = RawCuda(args.library)
    stream = cp.cuda.Stream(non_blocking=True)
    arrays = raw.load_many({k: specs[k] for k in ("initial_depth", "next_depth", "active")}, stream=stream)
    outputs = []
    with stream:
        source_health = cp.zeros((1,), dtype="uint32")
        controls = dict(surface_depth=arrays["initial_depth"], active=arrays["active"],
                        source_health=source_health, source_identity={"origin":"synthetic-diagnostic-check"},
                        period_seconds=1, stream=stream)
        good = InundationDepth(**controls)  # initial observation contributes to the sampled maximum
        pointers = {k: a.data.ptr for k, a in good.arrays.items()}
        cp.cuda.runtime.memcpyAsync(arrays["initial_depth"].data.ptr, arrays["next_depth"].data.ptr,
                                   arrays["next_depth"].nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
        good.step(1, stream=stream)
        outputs += [("good_depth", good.arrays["depth_m"], "expected_depth"),
                    ("good_maximum", good.arrays["maximum_depth_m"], "expected_maximum"),
                    ("good_health", good.arrays["health"].view("uint8"), "healthy")]
        if pointers != {k:a.data.ptr for k,a in good.arrays.items()}:
            raise AssertionError("diagnostic pointers changed")
        stream.synchronize()
    raw.store_many([(a,args.out/f"{name}.raw") for name,a,_ in outputs],stream=stream)
    with stream:
        source_health.fill(1)
        failed = InundationDepth(**controls)
        source_health.fill(0)
        arrays["initial_depth"].fill(-1)
        bad_depth = InundationDepth(**controls)
        arrays["initial_depth"].fill(0)
        arrays["active"].fill(2)
        bad_mask = InundationDepth(**controls)
    failures = [("source_failed",failed,"source_failed"),("negative_depth",bad_depth,"bad_depth"),("invalid_mask",bad_mask,"bad_mask")]
    extra = []
    for name, component, expected in failures:
        extra += [(name+"_health",component.arrays["health"].view("uint8"),expected)]
    # Source failure and invalid masks yield no-data everywhere. Negative-depth
    # check also has inactive cells, so its depth is entirely no-data as well.
    for name, component, _ in failures:
        extra += [(name+"_depth",component.arrays["depth_m"],"expected_failed_depth")]
    raw.store_many([(a,args.out/f"{name}.raw") for name,a,_ in extra],stream=stream)
    fields=[]
    for name,array,expected in outputs+extra:
        spec=specs[expected]
        fields.append(dict(name=name,dtype=spec["dtype"],shape=spec["shape"],order="c",
                           expected={"path":spec["path"],"sha256":spec["sha256"]},actual=str(args.out/f"{name}.raw")))
    request=dict(schema="nextgen.raw.compare/v1",unit="inundation-diagnostic",case="synthetic-copy-max-mask-health",
                 numerics=numerics.mode(),candidate_revision=args.revision,
                 command=["check_inundation.py","--fixture",str(args.fixture)],
                 provenance={"origin":"tool-self-test","sources":[{"name":"original diagnostic fixture","revision":args.revision,"licence":"Apache-2.0"}],
                             "evidence":[{"path":str(args.fixture),"sha256":raw.sha256(args.fixture)}]},fields=fields)
    (args.out/"request.json").write_text(json.dumps(request,indent=2)+"\n")
    subprocess.run([str(args.comparator),str(args.out/"request.json"),str(args.out/"verdict.json")],check=True)
    # One separate, explicitly synthetic 1M-cell diagnostic timing. No routing
    # speed acceptance or physical-area equivalence is inferred from this grid.
    with stream:
        shape=(1024,1024)
        depth=cp.zeros(shape,dtype="float32"); active=cp.ones(shape,dtype="int32")
        health=cp.zeros((1,),dtype="uint32")
        large=InundationDepth(surface_depth=depth,active=active,source_health=health,
                              source_identity={"origin":"synthetic-diagnostic-benchmark"},period_seconds=1,stream=stream)
    stream.synchronize()
    start=time.perf_counter()
    for _ in range(64): large.step(1,stream=stream)
    stream.synchronize()
    wall_ms=(time.perf_counter()-start)*1000
    begin,end=cp.cuda.Event(),cp.cuda.Event()
    with stream:
        begin.record(stream); large.step(1,stream=stream);end.record(stream)
    end.synchronize()
    card=cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"]
    if isinstance(card,bytes):card=card.decode()
    receipt=dict(schema="nextgen.inundation.check/v1",candidate_revision=args.revision,numerics=numerics.mode(),card=card,
                 cuda_graphs=False,origin="tool-self-test",stable_pointers=True,grid_shape=list(shape),
                 parallelism="GPU one thread/cell; 256 threads/block; Rust offline I/O/comparison 8 Rayon workers",
                 batch_steps=64,batch_wall_ms=wall_ms,diagnostic_step_ms=wall_ms/64,
                 separate_device_interval_ms=float(cp.cuda.get_elapsed_time(begin,end)),
                 scope="Synthetic diagnostic only. Initial setup observation outside timing. Batch has two boundary synchronizations, zero per-step events/sync, direct prepared launches. Separate one-step CUDA-event interval includes submission gaps. Shared GPU pool; no whole hydro or terrain-step speed claim. No native routing, USGS skill, observed-event inundation or inspected physical picture established.")
    (args.out/"execution.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("fixture","library","comparator","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--revision",required=True)
    run(p.parse_args())
