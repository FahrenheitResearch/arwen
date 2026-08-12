"""Host-to-device bandwidth under N concurrent GPUs, and the host RAM ceiling.

THE QUESTION.  Streaming puts the domain in pinned host RAM and pulls tiles
across PCIe.  One GPU doing that on a PCIe 4.0 x16 link asks for ~26 GB/s.
Eight of them would ask for ~208 GB/s of SUSTAINED host memory bandwidth,
while the CPU wants the same bus.  If the aggregate flattens as GPUs are
added, the streaming penalty grows with GPU count and the
"cheap-cards-plus-RAM" argument weakens.  This module measures the aggregate.

WHAT IS MEASURED, and what each control is for
----------------------------------------------
``h2d`` / ``d2h`` / ``bidir``
    Pinned host <-> device ``cudaMemcpyAsync`` at a chunk size big enough that
    per-copy launch cost is noise (default 256 MiB, ~20 ms on a 4.0 x8 link).

``--host-span``
    The host side is read from a rotating window whose span must exceed the
    last-level cache or the "host memory" bandwidth is really cache bandwidth.
    EPYC 7713 carries 256 MiB of L3, so the default span is 4 GiB and the
    stride never revisits a line inside one pass.  This is not decoration: a
    256 MiB host buffer on this part reports numbers that are pure L3.

``--pin``
    ``sched_setaffinity`` per worker thread.  On a dual-socket box an unbound
    thread can allocate its pinned buffer on the far socket and then measure
    the inter-socket link rather than PCIe.  Reported either way, because the
    difference IS the NUMA artefact.

``cpu`` mode
    A multi-threaded host-only copy (numpy releases the GIL), so the DDR
    ceiling is measured on the same box in the same session rather than
    quoted from a spec sheet.  Without it "the aggregate flattened" has no
    denominator.

WHY THE PYTHON DOES NOT SHOW UP
-------------------------------
Each worker queues ``--queue`` copies on its own stream and synchronises once,
so the GIL is taken a handful of times per second while the DMA engines run
for tens of milliseconds per copy.  ``--queue 1`` is kept as the control that
demonstrates this: if the 1-GPU number is the same at queue 1 and queue 8,
submission cost is not in the measurement.

Every worker runs for the same wall window and the workers are released by a
barrier, so "two GPUs at once" means genuinely at once; each worker also
reports the fraction of its own window that overlapped every other worker's,
and a run whose overlap is below ``--min-overlap`` is REFUSED rather than
reported, because two GPUs that ran one after the other trivially show
perfect scaling.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import threading
import time


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _cuda():
    import cupy as cp
    return cp


def gpu_count() -> int:
    import cupy as cp
    return cp.cuda.runtime.getDeviceCount()


def device_info(dev: int) -> dict:
    import cupy as cp
    with cp.cuda.Device(dev):
        props = cp.cuda.runtime.getDeviceProperties(dev)
        free, total = cp.cuda.runtime.memGetInfo()
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return {
        "device": dev,
        "name": name,
        "pci_bus_id": f"{props['pciDomainID']:04x}:{props['pciBusID']:02x}:"
                      f"{props['pciDeviceID']:02x}.0",
        "async_engine_count": int(props.get("asyncEngineCount", -1)),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def pcie_link(bus_id: str) -> dict:
    """The link the driver actually trained, read from sysfs.

    ``max_link_width`` is what the slot could do and ``current_link_width``
    what it does; a x16 card in a x8 slot is the single most common reason a
    measured ceiling is half of what an agent expected, and it is invisible to
    ``nvidia-smi``.
    """
    base = f"/sys/bus/pci/devices/{bus_id}"
    out = {}
    for key in ("max_link_speed", "max_link_width",
                "current_link_speed", "current_link_width"):
        try:
            with open(f"{base}/{key}") as fh:
                out[key] = fh.read().strip()
        except OSError:
            out[key] = None
    return out


def theoretical_pcie_gbs(speed: str | None, width: str | None) -> float | None:
    """Usable one-way GB/s for a trained PCIe link, encoding included."""
    if not speed or not width:
        return None
    try:
        gts = float(str(speed).split()[0])
        lanes = int(width)
    except (ValueError, IndexError):
        return None
    # 8b/10b below 8 GT/s, 128b/130b at 8 and 16 GT/s.
    ratio = 0.8 if gts < 8 else 128.0 / 130.0
    return gts * lanes * ratio / 8.0


# --------------------------------------------------------------------------
# the copy worker
# --------------------------------------------------------------------------

class _Result:
    __slots__ = ("dev", "bytes", "t0", "t1", "reps", "error")

    def __init__(self, dev):
        self.dev = dev
        self.bytes = 0
        self.t0 = None
        self.t1 = None
        self.reps = []
        self.error = None


def _worker(dev, res, barrier, stop_after, *, chunk_bytes, host_span_bytes,
            queue, direction, cores):
    import cupy as cp
    import numpy as np

    try:
        if cores:
            os.sched_setaffinity(0, cores)
        cp.cuda.Device(dev).use()

        # Host side: one pinned span, far larger than L3, walked with a stride
        # that never revisits a cache line within a pass.
        nchunk = max(1, host_span_bytes // chunk_bytes)
        span = nchunk * chunk_bytes
        host = cp.cuda.alloc_pinned_memory(span)
        hview = np.frombuffer(host, dtype=np.uint8, count=span)
        hview[::4096] = 1                      # fault every page in
        dev_buf = cp.empty(chunk_bytes, dtype=cp.uint8)
        dev_ptr = int(dev_buf.data.ptr)
        host_ptr = ctypes.addressof(ctypes.c_char.from_buffer(hview))

        stream = cp.cuda.Stream(non_blocking=True)
        sptr = stream.ptr
        memcpy = cp.cuda.runtime.memcpyAsync
        H2D = cp.cuda.runtime.memcpyHostToDevice
        D2H = cp.cuda.runtime.memcpyDeviceToHost

        def issue(i):
            off = (i % nchunk) * chunk_bytes
            if direction == "h2d":
                memcpy(dev_ptr, host_ptr + off, chunk_bytes, H2D, sptr)
            elif direction == "d2h":
                memcpy(host_ptr + off, dev_ptr, chunk_bytes, D2H, sptr)
            else:
                memcpy(dev_ptr, host_ptr + off, chunk_bytes, H2D, sptr)
                memcpy(host_ptr + off, dev_ptr, chunk_bytes, D2H, sptr)

        # Warm-up, discarded: the first copies pay link training and pool
        # setup, and on an idle card the link sits at 2.5 GT/s until used.
        for i in range(queue):
            issue(i)
        stream.synchronize()

        barrier.wait()
        res.t0 = time.perf_counter()
        i = 0
        total = 0
        per_copy = chunk_bytes * (2 if direction == "bidir" else 1)
        while time.perf_counter() - res.t0 < stop_after:
            r0 = time.perf_counter()
            for _ in range(queue):
                issue(i)
                i += 1
            stream.synchronize()
            r1 = time.perf_counter()
            total += per_copy * queue
            res.reps.append(per_copy * queue / (r1 - r0) / 1e9)
        res.t1 = time.perf_counter()
        res.bytes = total

        del dev_buf, hview, host
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:                                # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
        try:
            barrier.wait(timeout=1)
        except Exception:                                   # noqa: BLE001
            pass


def measure(devices, *, seconds=3.0, chunk_mib=256, host_span_mib=4096,
            queue=4, direction="h2d", pin=None, min_overlap=0.9) -> dict:
    """Run every device in ``devices`` concurrently and report GB/s.

    ``pin`` is ``None`` (unbound), ``"local"`` (each worker gets a disjoint
    block of cores) or an explicit ``{dev: [cores]}``.  The overlap check is
    not optional: a "2 GPU" number whose two workers barely overlapped is the
    1-GPU number twice, and it looks like perfect scaling.
    """
    ncpu = os.cpu_count() or 1
    chunk = chunk_mib * 1024 * 1024
    span = host_span_mib * 1024 * 1024
    n = len(devices)

    if pin == "local":
        per = max(1, ncpu // max(1, n))
        cores = {d: set(range(i * per, min(ncpu, (i + 1) * per)))
                 for i, d in enumerate(devices)}
    elif isinstance(pin, dict):
        cores = {d: set(pin[d]) for d in devices}
    else:
        cores = {d: None for d in devices}

    barrier = threading.Barrier(n)
    results = {d: _Result(d) for d in devices}
    threads = [threading.Thread(
        target=_worker, args=(d, results[d], barrier, seconds),
        kwargs=dict(chunk_bytes=chunk, host_span_bytes=span, queue=queue,
                    direction=direction, cores=cores[d]), daemon=True)
        for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=seconds * 20 + 120)

    errs = {d: r.error for d, r in results.items() if r.error}
    if errs:
        return {"error": errs, "devices": list(devices)}

    # Overlap: the fraction of the shortest window that every worker shared.
    t0 = max(r.t0 for r in results.values())
    t1 = min(r.t1 for r in results.values())
    shortest = min(r.t1 - r.t0 for r in results.values())
    overlap = max(0.0, t1 - t0) / shortest if shortest > 0 else 0.0

    per_dev = {}
    for d, r in results.items():
        reps = sorted(r.reps)
        # Warm-up rep discarded; median of the rest, spread reported.
        body = reps[1:] if len(reps) > 2 else reps
        per_dev[d] = {
            "gbs_median": statistics.median(body),
            "gbs_min": body[0],
            "gbs_max": body[-1],
            "spread_pct": (body[-1] - body[0]) / statistics.median(body) * 100,
            "reps": len(body),
            "bytes": r.bytes,
            "window_s": r.t1 - r.t0,
        }
    agg = sum(v["gbs_median"] for v in per_dev.values())
    out = {
        "devices": list(devices),
        "ngpu": n,
        "direction": direction,
        "chunk_mib": chunk_mib,
        "host_span_mib": host_span_mib,
        "queue": queue,
        "pin": pin if not isinstance(pin, dict) else "explicit",
        "cores": {d: (sorted(c)[:4] + ["..."] if c and len(c) > 4
                      else sorted(c) if c else None) for d, c in cores.items()},
        "per_device": per_dev,
        "aggregate_gbs": agg,
        "overlap": overlap,
    }
    if overlap < min_overlap:
        out["REFUSED"] = (
            f"workers overlapped only {overlap:.2f} of the window "
            f"(min {min_overlap}); this is not a concurrent measurement")
    return out


# --------------------------------------------------------------------------
# the host-only ceiling
# --------------------------------------------------------------------------

def _cpu_worker(nbytes, seconds, res, barrier, cores):
    import numpy as np
    if cores:
        os.sched_setaffinity(0, cores)
    src = np.ones(nbytes, dtype=np.uint8)
    dst = np.empty(nbytes, dtype=np.uint8)
    np.copyto(dst, src)
    barrier.wait()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        np.copyto(dst, src)          # releases the GIL
        n += 1
    dt = time.perf_counter() - t0
    # A copy touches the buffer twice: one read, one write.
    res.append(2 * nbytes * n / dt / 1e9)


def cpu_bandwidth(threads=(1, 2, 4, 8, 16, 32), *, mib=512,
                  seconds=2.0) -> dict:
    """Host-only copy bandwidth vs thread count -- the DDR ceiling, measured.

    Reported as read+write traffic, which is what a memcpy actually puts on
    the bus and the convention STREAM Copy uses.
    """
    ncpu = os.cpu_count() or 1
    out = {}
    for nt in threads:
        if nt > ncpu:
            continue
        res = []
        lock = threading.Lock()
        vals = []

        def collect(v, _lock=lock, _vals=vals):
            with _lock:
                _vals.append(v)

        barrier = threading.Barrier(nt)
        holders = [[] for _ in range(nt)]
        ths = [threading.Thread(target=_cpu_worker,
                                args=(mib * 1024 * 1024, seconds, holders[i],
                                      barrier, None), daemon=True)
               for i in range(nt)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=seconds * 10 + 60)
        got = [h[0] for h in holders if h]
        out[nt] = {"aggregate_gbs": sum(got), "per_thread_gbs": got[:4],
                   "threads_reporting": len(got)}
        del res, vals
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devices", default=None,
                    help="comma list, default every visible GPU")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--chunk-mib", type=int, default=256)
    ap.add_argument("--host-span-mib", type=int, default=4096)
    ap.add_argument("--queue", type=int, default=4)
    ap.add_argument("--direction", default="h2d",
                    choices=("h2d", "d2h", "bidir"))
    ap.add_argument("--pin", default=None, choices=(None, "local"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cpu", action="store_true",
                    help="also measure the host-only DDR ceiling")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import cupy as cp

    if args.devices:
        devs = [int(x) for x in args.devices.split(",")]
    else:
        devs = list(range(cp.cuda.runtime.getDeviceCount()))

    report = {"host": os.uname().nodename, "when": time.strftime("%FT%TZ",
                                                                time.gmtime()),
              "cpu_count": os.cpu_count(), "gpus": [], "scaling": []}
    for d in devs:
        info = device_info(d)
        info["pcie"] = pcie_link(info["pci_bus_id"])
        info["pcie_theoretical_gbs"] = theoretical_pcie_gbs(
            info["pcie"].get("max_link_speed"),
            info["pcie"].get("current_link_width"))
        report["gpus"].append(info)
        print(f"GPU{d} {info['name']}  {info['pci_bus_id']}  "
              f"link {info['pcie'].get('current_link_speed')} x"
              f"{info['pcie'].get('current_link_width')} "
              f"(max x{info['pcie'].get('max_link_width')})  "
              f"theoretical {info['pcie_theoretical_gbs']:.1f} GB/s"
              if info["pcie_theoretical_gbs"] else f"GPU{d} {info['name']}")

    if args.cpu:
        print("\n-- host-only copy bandwidth (read+write) --")
        cpu = cpu_bandwidth()
        report["cpu_bandwidth"] = cpu
        for nt, v in cpu.items():
            print(f"  {nt:3d} threads  {v['aggregate_gbs']:8.1f} GB/s")

    print(f"\n-- {args.direction} pinned, chunk {args.chunk_mib} MiB, "
          f"span {args.host_span_mib} MiB, queue {args.queue}, "
          f"pin={args.pin} --")
    for n in range(1, len(devs) + 1):
        subset = devs[:n]
        runs = []
        for _ in range(args.reps):
            runs.append(measure(subset, seconds=args.seconds,
                                chunk_mib=args.chunk_mib,
                                host_span_mib=args.host_span_mib,
                                queue=args.queue, direction=args.direction,
                                pin=args.pin))
        good = [r for r in runs if "error" not in r and "REFUSED" not in r]
        if not good:
            print(f"  {n} GPU: REFUSED/ERROR {runs[0].get('REFUSED') or runs[0].get('error')}")
            report["scaling"].append({"ngpu": n, "runs": runs})
            continue
        aggs = sorted(r["aggregate_gbs"] for r in good)
        med = statistics.median(aggs)
        spread = (aggs[-1] - aggs[0]) / med * 100 if med else 0.0
        per = {d: statistics.median(
            [r["per_device"][d]["gbs_median"] for r in good]) for d in subset}
        entry = {"ngpu": n, "devices": subset, "aggregate_gbs_median": med,
                 "aggregate_spread_pct": spread,
                 "per_device_gbs_median": per,
                 "overlap_min": min(r["overlap"] for r in good),
                 "reps": len(good), "runs": good}
        report["scaling"].append(entry)
        flag = "  <-- SPREAD >10%" if spread > 10 else ""
        print(f"  {n} GPU: aggregate {med:7.2f} GB/s  "
              f"(spread {spread:4.1f}%{flag})  per-GPU " +
              " ".join(f"{d}:{v:.2f}" for d, v in per.items()) +
              f"  overlap {entry['overlap_min']:.2f}")

    base = None
    for e in report["scaling"]:
        if e.get("ngpu") == 1 and "aggregate_gbs_median" in e:
            base = e["aggregate_gbs_median"]
    if base:
        print("\n-- scaling against 1 GPU --")
        for e in report["scaling"]:
            if "aggregate_gbs_median" not in e:
                continue
            n = e["ngpu"]
            got = e["aggregate_gbs_median"]
            print(f"  {n} GPU: {got:7.2f} GB/s = {got / base:4.2f}x of one "
                  f"(ideal {n}.00x, efficiency {got / base / n * 100:5.1f}%)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
