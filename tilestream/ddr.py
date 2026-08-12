"""Host DDR copy bandwidth vs thread count -- the denominator for "flattened".

Standalone and deliberately dumb: numpy releases the GIL inside a large
contiguous copy, so N threads copying disjoint buffers is a serviceable
STREAM Copy.  Reported as read+write traffic, which is what a memcpy puts on
the bus.

Buffers are 256 MiB per thread, which is larger than this part's whole 256 MiB
L3, so a single thread already misses cache; the aggregate is what matters
anyway and at 8 threads the working set is 4 GiB.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np


def worker(nbytes, seconds, out, barrier):
    src = np.ones(nbytes, dtype=np.uint8)
    dst = np.empty(nbytes, dtype=np.uint8)
    np.copyto(dst, src)                      # first-touch, discarded
    barrier.wait()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        np.copyto(dst, src)
        n += 1
    dt = time.perf_counter() - t0
    out.append(2 * nbytes * n / dt / 1e9)


def main(counts=(1, 2, 4, 8, 16, 32, 64), mib=256, seconds=1.5):
    best = 0.0
    for nt in counts:
        nb = mib << 20
        barrier = threading.Barrier(nt)
        hold = [[] for _ in range(nt)]
        ths = [threading.Thread(target=worker,
                                args=(nb, seconds, hold[i], barrier),
                                daemon=True) for i in range(nt)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=180)
        got = [h[0] for h in hold if h]
        agg = sum(got)
        best = max(best, agg)
        print(f"{nt:3d} threads  {agg:8.1f} GB/s  ({len(got)}/{nt} reported)",
              flush=True)
    print(f"PEAK {best:.1f} GB/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
