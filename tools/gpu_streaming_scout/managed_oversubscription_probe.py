"""Managed-memory (UVM) throughput, resident vs oversubscribed.

Bounds the "just use cudaMallocManaged and let the driver page it" shortcut.
Allocates progressively larger managed arrays and times a full read-modify-
write pass over each, so the collapse point and the post-collapse rate are
both measured rather than assumed.  Needs cupy.

    python managed_oversubscription_probe.py

Measured for task #260; see Downloads/GPUWM-HEX-SINGLE-CARD-STREAMING.md.
"""

import json, time
import numpy as np, cupy as cp

free, total = cp.cuda.runtime.memGetInfo()
res = {"total_GiB": round(total/2**30,2), "free_GiB": round(free/2**30,2)}

def stream_kernel_test(gib, managed, iters=3):
    n = int(gib * 2**30 // 4)
    if managed:
        pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        old = cp.cuda.get_allocator()
        cp.cuda.set_allocator(pool.malloc)
    try:
        a = cp.zeros(n, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        # touch-all elementwise: a = a*1.000001 + 1  (read+write whole array)
        t0=time.perf_counter()
        for _ in range(iters):
            a *= cp.float32(1.000001)
            cp.cuda.Stream.null.synchronize()
        el=(time.perf_counter()-t0)/iters
        gbs = 2*n*4/el/1e9   # read+write
        del a
    finally:
        if managed:
            cp.cuda.set_allocator(old)
            pool.free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()
    return {"gib": gib, "managed": managed, "sec_per_pass": round(el,4), "eff_GBs": round(gbs,1)}

out=[]
for gib, managed in [(8,False),(8,True),(24,True),(40,True),(56,True)]:
    try:
        r = stream_kernel_test(gib, managed)
    except Exception as e:
        r = {"gib":gib,"managed":managed,"error":type(e).__name__+": "+str(e)[:160]}
    out.append(r); print(json.dumps(r), flush=True)
res["passes"]=out
print("FINAL "+json.dumps(res))
