"""Pinned vs pageable host-device bandwidth, and concurrent bidirectional.

Run on the target card before sizing any host-parked residency plan: pinned
staging is 3-4x pageable, and a silent fallback to pageable is a 3x runtime
regression with no error.  Needs cupy; holds no mutex of its own.

    python pcie_bandwidth_probe.py

Measured for task #260; see Downloads/GPUWM-HEX-SINGLE-CARD-STREAMING.md.
"""

import json, time, sys
import numpy as np, cupy as cp

def bench(nbytes, pinned, iters=20, warm=5):
    n = nbytes // 8
    if pinned:
        mem = cp.cuda.alloc_pinned_memory(n*8)
        h = np.frombuffer(mem, dtype=np.float64, count=n)
    else:
        h = np.empty(n, dtype=np.float64)
    h[:] = 1.0
    d = cp.empty(n, dtype=cp.float64)
    st = cp.cuda.Stream(non_blocking=True)
    def h2d():
        with st: d.set(h, stream=st)
        st.synchronize()
    def d2h():
        with st: d.get(out=h, stream=st)
        st.synchronize()
    out={}
    for name, fn in (("h2d",h2d),("d2h",d2h)):
        for _ in range(warm): fn()
        t=time.perf_counter()
        for _ in range(iters): fn()
        el=time.perf_counter()-t
        out[name]= nbytes*iters/el/1e9
    del d
    cp.get_default_memory_pool().free_all_blocks()
    return out

props = cp.cuda.runtime.getDeviceProperties(0)
name = props["name"]; name = name.decode() if isinstance(name,bytes) else str(name)
res = {"gpu": name, "cupy": cp.__version__, "results": []}
for mb in (64, 256, 1024, 4096):
    nb = mb*(1<<20)
    for pinned in (True, False):
        r = bench(nb, pinned)
        res["results"].append({"mb":mb,"pinned":pinned,"h2d_GBs":round(r["h2d"],2),"d2h_GBs":round(r["d2h"],2)})
        print(json.dumps(res["results"][-1]), flush=True)
# concurrent bidirectional (two streams) at 1 GiB
n=(1<<30)//8
mem_a=cp.cuda.alloc_pinned_memory(n*8); ha=np.frombuffer(mem_a,dtype=np.float64,count=n); ha[:]=1.0
mem_b=cp.cuda.alloc_pinned_memory(n*8); hb=np.frombuffer(mem_b,dtype=np.float64,count=n)
da=cp.empty(n,dtype=cp.float64); db=cp.empty(n,dtype=cp.float64)
s1=cp.cuda.Stream(non_blocking=True); s2=cp.cuda.Stream(non_blocking=True)
for _ in range(3):
    with s1: da.set(ha,stream=s1)
    with s2: db.get(out=hb,stream=s2)
    s1.synchronize(); s2.synchronize()
t=time.perf_counter(); IT=10
for _ in range(IT):
    with s1: da.set(ha,stream=s1)
    with s2: db.get(out=hb,stream=s2)
    s1.synchronize(); s2.synchronize()
el=time.perf_counter()-t
res["bidir_1GiB_total_GBs"]=round(2*(1<<30)*IT/el/1e9,2)
print(json.dumps({"bidir_total_GBs":res["bidir_1GiB_total_GBs"]}), flush=True)
res["pcie"] = {
 "gen": cp.cuda.runtime.deviceGetAttribute(cp.cuda.runtime.cudaDevAttrPciBusId,0) if False else None,
}
print("FINAL "+json.dumps(res))
