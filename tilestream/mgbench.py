"""Does the streaming penalty get worse as GPUs are added?

THE EXPERIMENT, and why it needs three arms rather than one.

The tempting measurement is "one GPU streaming vs two GPUs streaming".  It is
not enough.  Two GPUs in one chassis can slow each other down for at least
four reasons, and only the first is the one this lane is about:

1. host memory bandwidth -- both DMA engines reading the same DRAM;
2. the PCIe root complex / IOMMU serialising two DMA streams;
3. the two Python worker threads contending for the GIL while submitting;
4. power and thermal limits shared across the board.

So the arms are:

``independent-host``
    Each GPU runs its OWN full-domain streamed forecast out of its OWN pinned
    host store.  No shared data, no barrier, no coupling of any kind: the only
    thing the two GPUs share is the machine.  Per-GPU throughput here, one GPU
    vs two, is the cleanest possible measure of (1)+(2)+(3)+(4).

``independent-vram``
    THE CONTROL.  Identical in every respect except that the store lives in
    the GPU's own VRAM, so the tile gathers never touch host memory.  Same
    kernels, same tiles, same threads, same submission pattern, same power
    envelope -- and zero host traffic.  Whatever slowdown survives here is
    (2)+(3)+(4) and is NOT the streaming penalty.  The difference between the
    two arms is the part attributable to host transport.

    Without this arm a drop in the host arm would be reported as bandwidth
    contention when it might be the GIL.  This project has already produced
    six false results; that is exactly the shape of the seventh.

``shared-host``
    The real thing: ONE pinned host domain, tiles shared out over the GPUs
    (:mod:`tilestream.mgstream`).  This is the configuration a forecast would
    actually run in and it is the only arm that also carries coupling --
    a per-sweep barrier, so the slowest GPU sets the pace.

DIGESTS.  Every timed configuration digests its final store, and every
configuration in a sweep starts from the same state, so all of them must agree
bit for bit.  A transport that quietly skips work looks exactly like a
speedup, and this is the check that catches it.  The digest is over the whole
persisted inventory; the ``independent`` arms run the same domain on each GPU,
so their per-GPU digests must also agree with each other.

COMPUTE WINDOW.  Refused below 500 cells on a side.  A tile smaller than that
leaves the GPU idle between kernel launches and measures launch latency, which
this project has already lost hours to.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
import warnings

import numpy as np

from tilestream import driver, gather, harness, mgstream
from tilestream import spec as tspec

MIN_COMPUTE = 500


def _digest(arrays) -> str:
    h = hashlib.blake2b(digest_size=16)
    for name in sorted(arrays):
        a = arrays[name]
        a = a.get() if hasattr(a, "get") else np.asarray(a)
        h.update(name.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _start_state(cfg, seed):
    state = harness.make_state(cfg, seed=seed)
    out = {n: np.array(a.get() if hasattr(a, "get") else a)
           for n, a in harness.state_arrays(state).items()}
    del state
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    return out


def _refill(store, start) -> None:
    """Put ``start`` back into ``store``.

    The warm-up sweep is not free of consequences: it ADVANCES the store, so a
    timed run that follows it starts one step late and its digest is the
    digest of a different forecast.  That is how the first version of this
    bench reported "DISAGREE" while every arm was in fact correct.  Every arm
    therefore refills before the timed region, and the refill is outside it.
    """
    import cupy as cp
    arrays = driver._arrays_of(store)
    for n, a in start.items():
        dst = arrays[n]
        if gather.is_device_array(dst):
            dst[...] = cp.asarray(a)
        else:
            dst[...] = a
    cp.cuda.runtime.deviceSynchronize()


def _make_store(cfg, start, *, on_device: bool):
    import cupy as cp
    if on_device:
        return {n: cp.asarray(a) for n, a in start.items()}
    from tilestream.hoststore import HostDomainStore
    store = HostDomainStore(cfg)
    for n, a in start.items():
        store.arrays[n][...] = a
    return store


# --------------------------------------------------------------------------
# arm 1 and 2: independent per-GPU forecasts
# --------------------------------------------------------------------------

def _independent_worker(dev, cfg, start, tile, halo, nsteps, nbuffers,
                        write_mode, on_device, barrier, out, warmup):
    import cupy as cp
    try:
        cp.cuda.Device(dev).use()
        store = _make_store(cfg, start, on_device=on_device)
        rep: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Warm-up sweep, discarded: first-touch of the pinned pages, the
            # plan cache, and the PCIe link coming up from its idle 2.5 GT/s.
            if warmup:
                driver.run_tiled(store, cfg, tile, tile, halo=halo, nsteps=2,
                                 nbuffers=nbuffers, write_mode=write_mode)
                cp.cuda.runtime.deviceSynchronize()
                _refill(store, start)
            barrier.wait()
            t0 = time.perf_counter()
            driver.run_tiled(store, cfg, tile, tile, halo=halo,
                             nsteps=nsteps, nbuffers=nbuffers,
                             write_mode=write_mode, report=rep)
            cp.cuda.runtime.deviceSynchronize()
            t1 = time.perf_counter()
        arrays = driver._arrays_of(store)
        out.update(dev=dev, t0=t0, t1=t1, seconds=t1 - t0,
                   digest=_digest(arrays),
                   gathered=rep["gathered_bytes"],
                   scattered=rep["scattered_bytes"],
                   tiles=rep["tiles"], compute=rep["compute"])
        del arrays
        if hasattr(store, "free"):
            store.free()
        del store
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except BaseException as exc:                            # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        try:
            barrier.abort()
        except Exception:                                   # noqa: BLE001
            pass


def bench_independent(devices, cfg, start, *, tile, halo, nsteps, nbuffers=2,
                      write_mode="ring", on_device=False, warmup=True) -> dict:
    """One full-domain streamed run per GPU, all at once, nothing shared."""
    barrier = threading.Barrier(len(devices))
    outs = [{} for _ in devices]
    ths = [threading.Thread(
        target=_independent_worker,
        args=(d, cfg, start, tile, halo, nsteps, nbuffers, write_mode,
              on_device, barrier, outs[i], warmup), daemon=True)
        for i, d in enumerate(devices)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    errs = [o["error"] for o in outs if "error" in o]
    if errs:
        return {"error": errs}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    cells = nz * ny * nx * nsteps
    t0 = max(o["t0"] for o in outs)
    t1 = min(o["t1"] for o in outs)
    shortest = min(o["seconds"] for o in outs)
    per = []
    for o in outs:
        per.append({
            "device": o["dev"], "seconds": o["seconds"],
            "cells_per_s": cells / o["seconds"],
            "host_gbs": ((o["gathered"] + o["scattered"]) / o["seconds"] / 1e9
                         if not on_device else 0.0),
            "digest": o["digest"],
        })
    return {
        "arm": "independent-vram" if on_device else "independent-host",
        "devices": list(devices), "ngpu": len(devices),
        "per_device": per,
        "overlap": max(0.0, t1 - t0) / shortest if shortest > 0 else 0.0,
        "cells_per_s_total": sum(p["cells_per_s"] for p in per),
        "cells_per_s_per_gpu": statistics.median(
            [p["cells_per_s"] for p in per]),
        "host_gbs_total": sum(p["host_gbs"] for p in per),
        "digests": sorted({p["digest"] for p in per}),
        "compute": outs[0]["compute"], "tiles": outs[0]["tiles"],
    }


# --------------------------------------------------------------------------
# arm 3: one shared host domain
# --------------------------------------------------------------------------

def bench_shared(devices, cfg, start, *, tile, halo, nsteps, nbuffers=2,
                 write_mode="shadow", partition="block", warmup=True) -> dict:
    """One pinned host domain, tiles shared over ``devices``."""
    import cupy as cp
    store = _make_store(cfg, start, on_device=False)
    rep: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if warmup:
            mgstream.run_mgstream(store, cfg, tile, tile, halo=halo, nsteps=2,
                                  devices=devices, nbuffers=nbuffers,
                                  write_mode=write_mode, partition=partition)
            _refill(store, start)
        t0 = time.perf_counter()
        mgstream.run_mgstream(store, cfg, tile, tile, halo=halo,
                              nsteps=nsteps, devices=devices,
                              nbuffers=nbuffers, write_mode=write_mode,
                              partition=partition, report=rep)
        t1 = time.perf_counter()
    arrays = driver._arrays_of(store)
    dig = _digest(arrays)
    cells = cfg.nz * cfg.ny * cfg.nx * nsteps
    out = {
        "arm": "shared-host", "devices": list(devices), "ngpu": len(devices),
        "write_mode": write_mode, "partition": partition,
        "seconds": t1 - t0, "cells_per_s_total": cells / (t1 - t0),
        "cells_per_s_per_gpu": cells / (t1 - t0) / len(devices),
        "host_gbs_total": rep["host_bytes"] / (t1 - t0) / 1e9,
        "digest": dig, "tiles": rep["tiles"], "compute": rep["compute"],
        "per_worker": [(w["device"], w["ntiles"], w["seconds"])
                       for w in rep["per_worker"]],
    }
    del arrays
    if hasattr(store, "free"):
        store.free()
    del store
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------
# monolithic anchor
# --------------------------------------------------------------------------

def bench_monolithic(dev, cfg, nsteps, seed=harness.DEFAULT_SEED) -> dict:
    """Un-tiled, VRAM-resident, one GPU: the denominator for "penalty".

    Built from the SAME seed the streamed arms start from, and warmed up on a
    THROWAWAY state rather than by stepping and restoring the timed one --
    restoring only the persisted arrays leaves the domain clock and the
    scratch behind, which is enough to move the digest and turn the anchor
    into a different experiment.
    """
    import cupy as cp
    cp.cuda.Device(dev).use()
    warm = harness.make_state(cfg, seed=seed)
    harness.run_steps(warm, cfg, 1)
    cp.cuda.runtime.deviceSynchronize()
    del warm
    cp.get_default_memory_pool().free_all_blocks()

    state = harness.make_state(cfg, seed=seed)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    harness.run_steps(state, cfg, nsteps)
    cp.cuda.runtime.deviceSynchronize()
    t1 = time.perf_counter()
    dig = _digest(harness.state_arrays(state))
    del state
    cp.get_default_memory_pool().free_all_blocks()
    cells = cfg.nz * cfg.ny * cfg.nx * nsteps
    return {"arm": "monolithic", "device": dev, "seconds": t1 - t0,
            "cells_per_s_total": cells / (t1 - t0), "digest": dig}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _med(xs):
    return statistics.median(xs)


def _summarise(label, runs, key="cells_per_s_total"):
    vals = sorted(r[key] for r in runs)
    med = _med(vals)
    spread = (vals[-1] - vals[0]) / med * 100 if med else 0.0
    return {"label": label, "median": med, "min": vals[0], "max": vals[-1],
            "spread_pct": spread, "reps": len(vals)}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=1536)
    ap.add_argument("--ny", type=int, default=1536)
    ap.add_argument("--nz", type=int, default=harness.DEFAULT_NZ)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--halo", type=int, default=None,
                    help="default harness.halo_radius(cfg); never tune it")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--nbuffers", type=int, default=2)
    ap.add_argument("--devices", default=None)
    ap.add_argument("--arms", default="independent-vram,independent-host,shared-host")
    ap.add_argument("--write-mode", default="shadow", choices=("shadow", "ring"),
                    help="same mode in EVERY arm, or the arms are not comparable")
    ap.add_argument("--partition", default="block", choices=("block", "stripe"))
    ap.add_argument("--monolithic", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import cupy as cp
    devs = ([int(x) for x in args.devices.split(",")] if args.devices
            else list(range(cp.cuda.runtime.getDeviceCount())))
    cfg = harness.make_config(args.nx, args.ny, args.nz)
    halo = args.halo if args.halo is not None else harness.halo_radius(cfg)
    compute = args.tile + 2 * halo
    if compute < MIN_COMPUTE:
        print(f"REFUSED: compute window {compute} < {MIN_COMPUTE} cells; a "
              f"window this small measures launch latency, not the transport")
        return 2
    if args.nx % args.tile or args.ny % args.tile:
        print(f"REFUSED: tile {args.tile} must divide {args.nx}x{args.ny}")
        return 2
    if args.write_mode == "shadow" and args.steps % 2:
        # An odd sweep count leaves the newest data in the shadow, and both
        # drivers then memcpy a whole domain back into the caller's store --
        # inside the timed region.  At 1536^2 that is 3.7 GiB of host-to-host
        # traffic charged to the transport, which is not what is being
        # measured.  Even sweep counts have no copy-back at all.
        print(f"REFUSED: write_mode=shadow needs an EVEN --steps (got "
              f"{args.steps}); an odd count charges a whole-domain copy-back "
              f"to the timing")
        return 2

    print(f"domain {args.nx}x{args.ny}x{args.nz}  tile {args.tile}  "
          f"halo {halo} (harness.halo_radius)  compute window {compute}  "
          f"steps {args.steps}  reps {args.reps}  devices {devs}")
    # The start state is built on a GPU, and it must be the FIRST REQUESTED
    # device rather than CUDA's default.  Without this a run confined to GPU1
    # still allocates its initial state on GPU0 and dies there if GPU0 is
    # busy -- which is not a memory-sizing problem and reads like one.
    cp.cuda.Device(devs[0]).use()
    for d in devs:
        with cp.cuda.Device(d):
            free, total = cp.cuda.runtime.memGetInfo()
        print(f"  GPU{d}: {free / 2**30:.2f} GiB free of {total / 2**30:.2f} "
              f"(memGetInfo; nvidia-smi is unreliable here)")
    start = _start_state(cfg, harness.DEFAULT_SEED)
    store_gib = sum(a.nbytes for a in start.values()) / 2**30
    print(f"one domain store = {store_gib:.2f} GiB\n")

    report = {"config": vars(args), "halo": halo, "compute": compute,
              "devices": devs, "store_gib": store_gib, "results": []}
    digests = {}

    if args.monolithic:
        runs = [bench_monolithic(devs[0], cfg, args.steps)
                for _ in range(args.reps)]
        s = _summarise("monolithic 1 GPU (VRAM, un-tiled)", runs)
        digests["monolithic"] = runs[0]["digest"]
        report["results"].append({"arm": "monolithic", "ngpu": 1,
                                  "summary": s, "runs": runs})
        print(f"  monolithic 1 GPU          {s['median']:12.3e} cells/s"
              f"  spread {s['spread_pct']:4.1f}%")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        print(f"\n-- {arm} --")
        for n in range(1, len(devs) + 1):
            subset = devs[:n]
            runs = []
            for _ in range(args.reps):
                if arm == "shared-host":
                    r = bench_shared(subset, cfg, start, tile=args.tile,
                                      halo=halo, nsteps=args.steps,
                                      nbuffers=args.nbuffers,
                                      write_mode=args.write_mode,
                                      partition=args.partition)
                else:
                    r = bench_independent(
                        subset, cfg, start, tile=args.tile, halo=halo,
                        nsteps=args.steps, nbuffers=args.nbuffers,
                        write_mode=args.write_mode,
                        on_device=(arm == "independent-vram"))
                if "error" in r:
                    print(f"  {n} GPU: ERROR {r['error']}")
                    runs = []
                    break
                runs.append(r)
            if not runs:
                continue
            s = _summarise(f"{arm} {n} GPU", runs)
            per = _summarise(f"{arm} {n} GPU per-gpu", runs,
                             key="cells_per_s_per_gpu")
            hb = _summarise(f"{arm} {n} GPU host", runs, key="host_gbs_total")
            digests[f"{arm}-{n}"] = (runs[0].get("digest")
                                     or runs[0]["digests"][0])
            entry = {"arm": arm, "ngpu": n, "devices": subset,
                     "summary": s, "per_gpu": per, "host_gbs": hb,
                     "runs": runs}
            report["results"].append(entry)
            flag = "  <-- SPREAD >10%" if s["spread_pct"] > 10 else ""
            print(f"  {n} GPU  total {s['median']:12.3e} cells/s"
                  f"  per-GPU {per['median']:12.3e}"
                  f"  host {hb['median']:6.2f} GB/s"
                  f"  spread {s['spread_pct']:4.1f}%{flag}")

    print("\n-- scaling (total domain throughput vs the same arm on 1 GPU) --")
    base = {}
    for e in report["results"]:
        if e["ngpu"] == 1:
            base[e["arm"]] = e["summary"]["median"]
    for e in report["results"]:
        b = base.get(e["arm"])
        if not b:
            continue
        n = e["ngpu"]
        got = e["summary"]["median"]
        print(f"  {e['arm']:18s} {n} GPU: {got / b:4.2f}x "
              f"(ideal {n}.00x, efficiency {got / b / n * 100:5.1f}%)")

    print("\n-- digests (every timed configuration; all must agree) --")
    for k, v in digests.items():
        print(f"  {k:24s} {v}")
    uniq = sorted(set(digests.values()))
    print(f"  {len(uniq)} distinct digest(s): "
          f"{'AGREE' if len(uniq) == 1 else 'DISAGREE -- a run skipped work'}")
    report["digests"] = digests
    report["digests_agree"] = len(uniq) == 1

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
