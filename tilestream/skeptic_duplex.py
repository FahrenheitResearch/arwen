"""Why the 1-GPU 'bidir' number came out BELOW the 1-GPU unidirectional number.

The mgstream probe reported bidirectional 10.93 GB/s on one 4090 against
13.43 GB/s H2D, and explained the deficit -- and the resulting >2x '2.29x
scaling' -- as hardware: "the 4090 reports asyncEngineCount == 1, so the two
directions serialise on one DMA queue".

Both GPUs on that box report asyncEngineCount == 2, in the probe's own five
result JSONs and live on the hardware.  So that explanation is not available.

This module tests the alternative: the serialisation is IN THE BENCHMARK.
``bw_probe._worker`` issues, for direction 'bidir'::

    memcpy(dev_ptr, host_ptr + off, n, H2D, sptr)
    memcpy(host_ptr + off, dev_ptr, n, D2H, sptr)

Both copies go to the SAME stream ``sptr``, and both touch the SAME device
buffer ``dev_ptr`` and the SAME host offset.  Work on one CUDA stream runs in
issue order, so the D2H cannot begin until the H2D has completed -- and even
with independent streams the shared buffer is a read-after-write hazard.  The
directions therefore serialise by construction, on any GPU, whatever the
engine count.

ARMS
  h2d          one stream, one direction              -- the reference
  d2h          one stream, one direction              -- the reference
  bidir_1s     one stream, shared buffer              -- reproduces the probe
  bidir_2s     two streams, two device buffers,
               two disjoint host spans                -- the real full duplex

If the deficit is the benchmark, bidir_2s >> bidir_1s and bidir_2s approaches
h2d + d2h.  If it is the hardware, the two bidir arms agree.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import threading
import time


def _span(chunk_bytes, span_bytes):
    import cupy as cp
    import numpy as np
    nchunk = max(1, span_bytes // chunk_bytes)
    total = nchunk * chunk_bytes
    host = cp.cuda.alloc_pinned_memory(total)
    hview = np.frombuffer(host, dtype=np.uint8, count=total)
    hview[::4096] = 1
    ptr = ctypes.addressof(ctypes.c_char.from_buffer(hview))
    return host, hview, ptr, nchunk


def run_arm(dev, arm, *, seconds=3.0, chunk_mib=256, span_mib=4096, queue=4):
    import cupy as cp

    chunk = chunk_mib * 1024 * 1024
    span = span_mib * 1024 * 1024
    cp.cuda.Device(dev).use()

    memcpy = cp.cuda.runtime.memcpyAsync
    H2D = cp.cuda.runtime.memcpyHostToDevice
    D2H = cp.cuda.runtime.memcpyDeviceToHost

    hostA, viewA, ptrA, nA = _span(chunk, span)
    sA = cp.cuda.Stream(non_blocking=True)
    bufA = cp.empty(chunk, dtype=cp.uint8)
    pA = int(bufA.data.ptr)

    hostB = viewB = None
    if arm == "bidir_2s":
        hostB, viewB, ptrB, nB = _span(chunk, span)
        sB = cp.cuda.Stream(non_blocking=True)
        bufB = cp.empty(chunk, dtype=cp.uint8)
        pB = int(bufB.data.ptr)

    def issue(i):
        off = (i % nA) * chunk
        if arm == "h2d":
            memcpy(pA, ptrA + off, chunk, H2D, sA.ptr)
        elif arm == "d2h":
            memcpy(ptrA + off, pA, chunk, D2H, sA.ptr)
        elif arm == "bidir_1s":
            # exactly what bw_probe does: one stream, one buffer
            memcpy(pA, ptrA + off, chunk, H2D, sA.ptr)
            memcpy(ptrA + off, pA, chunk, D2H, sA.ptr)
        elif arm == "bidir_2s":
            # independent streams, independent buffers, independent host spans
            memcpy(pA, ptrA + off, chunk, H2D, sA.ptr)
            memcpy(ptrB + (i % nB) * chunk, pB, chunk, D2H, sB.ptr)

    per_copy = chunk * (2 if arm.startswith("bidir") else 1)

    for i in range(queue):
        issue(i)
    sA.synchronize()
    if arm == "bidir_2s":
        sB.synchronize()

    reps = []
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < seconds:
        r0 = time.perf_counter()
        for _ in range(queue):
            issue(i)
            i += 1
        sA.synchronize()
        if arm == "bidir_2s":
            sB.synchronize()
        r1 = time.perf_counter()
        reps.append(per_copy * queue / (r1 - r0) / 1e9)

    body = sorted(reps)[1:] if len(reps) > 2 else sorted(reps)
    out = {
        "arm": arm, "device": dev, "chunk_mib": chunk_mib,
        "span_mib": span_mib, "queue": queue,
        "gbs_median": statistics.median(body),
        "gbs_min": body[0], "gbs_max": body[-1],
        "n": len(body),
    }
    del bufA, viewA, hostA
    if arm == "bidir_2s":
        del bufB, viewB, hostB
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--chunk-mib", type=int, default=256)
    ap.add_argument("--span-mib", type=int, default=4096)
    ap.add_argument("--queue", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import cupy as cp

    d = args.device
    props = cp.cuda.runtime.getDeviceProperties(d)
    name = props["name"]
    name = name.decode() if isinstance(name, bytes) else name
    bus = f"{props['pciDomainID']:04x}:{props['pciBusID']:02x}:{props['pciDeviceID']:02x}.0"
    link = {}
    for k in ("current_link_speed", "current_link_width", "max_link_width"):
        try:
            with open(f"/sys/bus/pci/devices/{bus}/{k}") as fh:
                link[k] = fh.read().strip()
        except OSError:
            link[k] = None
    rep = {"host": os.uname().nodename,
           "when": time.strftime("%FT%TZ", time.gmtime()),
           "device": d, "name": name, "pci": bus,
           "async_engine_count": int(props.get("asyncEngineCount", -1)),
           "link": link, "arms": {}}
    print(f"{name}  {bus}  link {link.get('current_link_speed')} "
          f"x{link.get('current_link_width')} (max x{link.get('max_link_width')})")
    print(f"asyncEngineCount = {rep['async_engine_count']}")

    for arm in ("h2d", "d2h", "bidir_1s", "bidir_2s"):
        runs = [run_arm(d, arm, seconds=args.seconds,
                        chunk_mib=args.chunk_mib, span_mib=args.span_mib,
                        queue=args.queue)
                for _ in range(args.reps)]
        med = statistics.median(r["gbs_median"] for r in runs)
        lo = min(r["gbs_median"] for r in runs)
        hi = max(r["gbs_median"] for r in runs)
        spread = (hi - lo) / med * 100
        rep["arms"][arm] = {"gbs_median": med, "spread_pct": spread,
                            "reps": runs}
        print(f"  {arm:10s} {med:7.2f} GB/s   spread {spread:5.1f}%")

    a = rep["arms"]
    rep["bidir_2s_over_1s"] = a["bidir_2s"]["gbs_median"] / a["bidir_1s"]["gbs_median"]
    rep["unidir_sum"] = a["h2d"]["gbs_median"] + a["d2h"]["gbs_median"]
    rep["bidir_2s_over_sum"] = a["bidir_2s"]["gbs_median"] / rep["unidir_sum"]
    print(f"\nbidir_2s / bidir_1s = {rep['bidir_2s_over_1s']:.2f}x")
    print(f"h2d + d2h           = {rep['unidir_sum']:.2f} GB/s")
    print(f"bidir_2s / (h2d+d2h)= {rep['bidir_2s_over_sum']:.2f}")
    if rep["bidir_2s_over_1s"] > 1.3:
        print("VERDICT: the deficit is the BENCHMARK (single stream + shared "
              "buffer), not the DMA engine count.")
    else:
        print("VERDICT: the two bidir arms agree; the deficit is not the "
              "stream structure.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
