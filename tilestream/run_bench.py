"""Stage runner for the out-of-core streaming measurements.

Each stage is a SEPARATE PROCESS on purpose.  The stages differ by tens of
gigabytes of device and pinned-host residency, and a leftover allocation from
an earlier stage is exactly the kind of thing that turns into a mysterious
1.4x.  Separate processes make residency a property of the stage rather than
of the order the stages ran in.

That is affordable because :func:`tilestream.bench.seed_host_store` is
deterministic: the same ``--n``/``--seed`` reconstructs a bit-identical
initial store in any process, so digests are comparable ACROSS stages without
ever writing the state to disk.  Every stage prints the digest of the state it
ends on, and the stages that can be compared are compared.

Usage::

    python -m tilestream.run_bench selftest
    python -m tilestream.run_bench baseline  --n 1950 --steps 4
    python -m tilestream.run_bench vram      --n 1950 --tiles 512
    python -m tilestream.run_bench host      --n 1950 --tiles 512
    python -m tilestream.run_bench sweep     --n 1950 --tiles 256,384,512,768
    python -m tilestream.run_bench headline  --n 3378 --tile 512 --steps 2
    python -m tilestream.run_bench overcommit --n 2200 --steps 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from tilestream import bench, gather, harness, hoststore


def _banner(title: str) -> None:
    free, total = bench.free_vram_gib()
    mem = bench.host_mem_gib()
    print(f"\n=== {title} ===")
    print(f"    VRAM free {free:.2f} / {total:.2f} GiB | "
          f"host avail {mem['MemAvailable']:.1f} / {mem['MemTotal']:.1f} GiB")
    sys.stdout.flush()


def _emit(payload: dict) -> None:
    print("RESULT " + json.dumps(payload, default=str))
    sys.stdout.flush()


def _seeded_store(n: int, nz: int, seed: int, *, budget_gib: float = 40.0):
    cfg = harness.make_config(n, n, nz)
    t0 = time.perf_counter()
    store = hoststore.HostDomainStore(cfg, budget_bytes=int(budget_gib * 2 ** 30))
    bench.seed_host_store(store, cfg, seed=seed)
    print(f"    seeded {store.nbytes / 2**30:.2f} GiB pinned store in "
          f"{time.perf_counter() - t0:.1f}s "
          f"({store.bytes_per_cell:.1f} B/cell, largest block "
          f"{store.largest_field_bytes / 2**30:.2f} GiB)")
    rep = store.pinning_report()
    if not rep["all_pinned"]:
        raise SystemExit(f"store is NOT pinned: {rep}")
    print("    pinning verified via cudaPointerGetAttributes: all fields type=1")
    sys.stdout.flush()
    return store, cfg


def _digest_of(arrays) -> str:
    import hashlib

    h = hashlib.sha256()
    inner = getattr(arrays, "arrays", None)
    if isinstance(inner, dict):
        arrays = inner
    arrays = gather.inventory(arrays)
    for name in gather.serialized_attrs():
        if name in arrays:
            h.update(harness._digest_payload(f"state.{name}", arrays[name]))
    return h.hexdigest()


# --------------------------------------------------------------------------

def stage_selftest(args) -> None:
    _banner("SELFTEST: the timed loop is the loop the gate validated")
    d = bench.selftest_runner_matches_driver(nsteps=args.steps)
    for k, v in d.items():
        print(f"    {k:<12s} {v}")
    ok = len(set(d.values())) == 1
    print("    " + ("AGREE - TiledRunner == run_tiled == monolithic"
                    if ok else "DISAGREE - DO NOT TRUST ANY NUMBER BELOW"))
    _emit({"stage": "selftest", "agree": ok, "digests": d})
    if not ok:
        raise SystemExit(1)


def stage_baseline(args) -> None:
    """Whole domain resident in VRAM, stepped monolithically."""
    import cupy as cp

    from tilestream import driver

    _banner(f"BASELINE monolithic {args.n}x{args.n}x{args.nz}")
    store, cfg = _seeded_store(args.n, args.nz, args.seed)
    print(f"    initial digest {_digest_of(store)[:16]}")

    t0 = time.perf_counter()
    state = driver.make_tile_state(cfg)
    for name, arr in harness.state_arrays(state).items():
        arr.set(np.ascontiguousarray(store.arrays[name]))
    cp.cuda.runtime.deviceSynchronize()
    print(f"    uploaded to a resident DomainState in {time.perf_counter()-t0:.1f}s")
    store.free()
    free_pre, _ = bench.free_vram_gib()
    cells = args.n * args.n * args.nz
    print(f"    VRAM free after upload {free_pre:.2f} GiB "
          f"({(30.21 - free_pre) * 2**30 / cells:.1f} B/cell, state only)")
    sys.stdout.flush()

    res = bench.time_monolithic(state, cfg, steps=args.steps, warmup=args.warmup,
                                label=f"monolithic {args.n}^2")
    digest = res.digest
    print(f"    digest after {args.warmup + args.steps} steps {digest[:16]}")
    # Measured only NOW: dycore.step allocates its per-step scratch on the
    # first call, so a reading taken before the warm-up misses ~41 B/cell and
    # makes the resident footprint look a third smaller than it is.
    free, _ = bench.free_vram_gib()
    print(f"    resident after stepping: {(30.21 - free) * 2**30 / cells:.1f} "
          f"B/cell ({free:.2f} GiB free)")

    samples = [res.ms_per_step]
    probes = [bench.gpu_probe()]
    for _ in range(max(0, args.reps - 1)):
        r = bench.time_monolithic(state, cfg, steps=args.steps, warmup=0,
                                  hash_after=False)
        samples.append(r.ms_per_step)
        probes.append(bench.gpu_probe())
    st = bench.repeat_stats(samples)
    ns = st["min"] * 1e6 / cells
    print(f"    GPU during timing: util "
          f"{min(p.get('util_pct', 0) for p in probes):.0f}-"
          f"{max(p.get('util_pct', 0) for p in probes):.0f}%, "
          f"sm clock {min(p.get('sm_clock_mhz', 0) for p in probes):.0f}-"
          f"{max(p.get('sm_clock_mhz', 0) for p in probes):.0f} MHz, "
          f"mem {min(p.get('mem_used_mib', 0) for p in probes):.0f}-"
          f"{max(p.get('mem_used_mib', 0) for p in probes):.0f} MiB")
    print(f"    {st['min']:9.2f} ms/step (min of {st['n']}, median "
          f"{st['median']:9.2f}, spread {st['spread']:.3f})  {ns:7.3f} ns/cell")
    _emit({"stage": "baseline", "n": args.n, "nz": args.nz,
           "cells": cells, "ms_per_step": st["min"], "stats": st,
           "samples": samples, "ns_per_cell": ns, "digest": digest,
           "total_steps": args.warmup + args.steps,
           "resident_bytes_per_cell": (30.21 - free) * 2 ** 30 / cells,
           "gpu_probes": probes})


def _run_tiled_stage(args, *, on_device: bool, label: str) -> dict:
    """Time a tiled run, once per issue-order, each from the SAME state.

    The state is restored before every pipeline mode so the digests are
    directly comparable: a reordering that changed the answer would show up
    here at full scale, not just in the 192^2 selftest.
    """
    import cupy as cp

    _banner(f"{label} {args.n}x{args.n}x{args.nz} tile {args.tile} halo {args.halo}")
    store, cfg = _seeded_store(args.n, args.nz, args.seed)
    initial = _digest_of(store)
    print(f"    initial digest {initial[:16]}")

    if on_device:
        dev = bench.device_store_like(store, cfg)

        def restore():
            for name, arr in dev.items():
                arr.set(np.ascontiguousarray(store.arrays[name]))
            cp.cuda.runtime.deviceSynchronize()

        restore()
        target, shadow = dev, None             # device shadow built by runner
    else:
        def restore():
            bench.seed_host_store(store, cfg, seed=args.seed)

        target = store
        shadow = None
        if args.write_mode == "shadow":
            shadow = hoststore.HostDomainStore(cfg,
                                               budget_bytes=int(40 * 2 ** 30))
            print(f"    pinned shadow {shadow.nbytes / 2**30:.2f} GiB "
                  "allocated -- the whole point of write_mode='ring' is that "
                  "this is not needed")

    free, _ = bench.free_vram_gib()
    print(f"    VRAM free before runner {free:.2f} GiB")
    sys.stdout.flush()

    modes = (["naive", "prefetch"] if args.pipeline == "both"
             else [args.pipeline])

    # ONE runner, whose per-sweep `pipeline=` argument selects the issue
    # order.  Two runners would double the tile-buffer residency and make the
    # two modes see different amounts of free VRAM, which is precisely the
    # kind of asymmetry that manufactures a difference.
    runner = bench.TiledRunner(target, cfg, args.tile, args.tile,
                               halo=args.halo, nbuffers=args.nbuffers,
                               shadow=shadow, pipeline=modes[0],
                               write_mode=args.write_mode)
    if runner.ring is not None:
        print(f"    ring arena {runner.ring_bytes() / 2**30:.3f} GiB = "
              f"{100 * runner.ring_bytes() / sum(a.nbytes for a in runner.home.values()):.2f}"
              "% of the store (a shadow would be 100%)")
    free, _ = bench.free_vram_gib()
    print(f"    {len(runner.specs)} tiles, compute {runner.tile_cfg.nz}x"
          f"{runner.cny}x{runner.cnx}, redundancy {runner.redundancy:.3f}x, "
          f"VRAM free {free:.2f} GiB")
    g, s_ = runner.bytes_per_sweep()
    print(f"    {(g + s_) / 1e9:.3f} GB moved per sweep "
          f"({g / 1e9:.3f} H2D + {s_ / 1e9:.3f} D2H)")
    sys.stdout.flush()

    # --- correctness pass: one clean run per mode, from the same state ---
    digests = {}
    for mode in modes:
        restore()
        assert _digest_of(target) == initial, "restore did not reproduce"
        runner._src, runner._dst = runner.home, runner._dst_home()
        runner.sweep(args.warmup, pipeline=mode)
        runner.sweep(args.steps, pipeline=mode)
        runner.settle()
        digests[mode] = _digest_of(runner.home)
        print(f"    [{mode}] digest after {args.warmup + args.steps} steps "
              f"{digests[mode][:16]}")
    agree = len(set(digests.values())) == 1
    print(f"    issue orders agree bit-for-bit: {agree}")
    sys.stdout.flush()

    # --- timing pass: interleaved reps, minimum wins (see repeat_stats) ---
    samples: dict[str, list] = {m: [] for m in modes}
    probes = [bench.gpu_probe()]
    for rep in range(args.reps):
        for mode in modes:
            ph = runner.sweep(args.steps, pipeline=mode)
            samples[mode].append(ph.wall_ms)
        probes.append(bench.gpu_probe())
    ext = max(p.get("mem_used_mib", 0) for p in probes) - \
        min(p.get("mem_used_mib", 0) for p in probes)
    print(f"    GPU during timing: util "
          f"{min(p.get('util_pct', 0) for p in probes):.0f}-"
          f"{max(p.get('util_pct', 0) for p in probes):.0f}%, "
          f"sm clock {min(p.get('sm_clock_mhz', 0) for p in probes):.0f}-"
          f"{max(p.get('sm_clock_mhz', 0) for p in probes):.0f} MHz, "
          f"mem {min(p.get('mem_used_mib', 0) for p in probes):.0f}-"
          f"{max(p.get('mem_used_mib', 0) for p in probes):.0f} MiB "
          f"(swing {ext:.0f})")

    runs = []
    cells = args.n * args.n * args.nz
    for mode in modes:
        st = bench.repeat_stats(samples[mode])
        ns = st["min"] * 1e6 / cells
        ns_c = st["min"] * 1e6 / (runner.tile_cfg.nz * runner.cny * runner.cnx
                                  * len(runner.specs))
        moved = g + s_
        print(f"    [{mode:<8s}] {st['min']:9.2f} ms/step (min of {st['n']}, "
              f"median {st['median']:9.2f}, spread {st['spread']:.3f})  "
              f"{ns:7.3f} ns/cell  {ns_c:7.3f} ns/computed  "
              f"{moved / (st['min'] * 1e-3) / 1e9:5.1f} GB/s")
        runs.append({
            "pipeline": mode, "ms_per_step": st["min"], "stats": st,
            "samples": samples[mode], "ns_per_cell": ns,
            "ns_per_computed_cell": ns_c, "digest": digests[mode],
            "pcie_gbps": moved / (st["min"] * 1e-3) / 1e9,
        })
    sys.stdout.flush()

    # --- attribution (destroys state; last) ---
    best = min(modes, key=lambda m: bench.repeat_stats(samples[m])["min"])
    ph_ev = runner.sweep(args.steps, events=True, pipeline=best)
    attr = runner.attribute(args.steps, pipeline=best)
    print(f"    [{best}] events ms/sweep: gather {ph_ev.gather_ms:8.1f}  "
          f"stall {ph_ev.stall_ms:8.1f}  compute {ph_ev.compute_ms:8.1f}  "
          f"scatter {ph_ev.scatter_ms:8.1f}  wall {ph_ev.wall_ms:8.1f}  "
          f"sum/wall {ph_ev.overlap_ratio:.3f}")
    print(f"    [{best}] attrib ms/sweep: full {attr['full_ms']:8.1f}  "
          f"compute-only {attr['compute_only_ms']:8.1f}  "
          f"transfer-only {attr['transfer_only_ms']:8.1f}  "
          f"serial {attr['serial_ms']:8.1f}  ideal {attr['ideal_ms']:8.1f}  "
          f"overlap-eff {attr['overlap_efficiency']:.2f}  "
          f"exposed {attr['exposed_transfer_ms']:7.1f} ms  "
          f"(finite: {attr['compute_only_finite']})")
    for r in runs:
        if r["pipeline"] == best:
            r["events"] = {
                "gather_ms": ph_ev.gather_ms, "stall_ms": ph_ev.stall_ms,
                "compute_ms": ph_ev.compute_ms, "scatter_ms": ph_ev.scatter_ms,
                "wall_ms": ph_ev.wall_ms, "overlap_ratio": ph_ev.overlap_ratio}
            r["attribution"] = attr
    res = runner.result(ph_ev, label, digest=digests[best], steps=args.steps)
    mem = bench.host_mem_gib()
    print(f"    host avail {mem['MemAvailable']:.1f} / {mem['MemTotal']:.1f} GiB")
    runner.free()
    cp.get_default_memory_pool().free_all_blocks()
    payload = {
        "stage": label, "n": args.n, "nz": args.nz, "tile": args.tile,
        "halo": args.halo, "nbuffers": args.nbuffers, "tiles": res.tiles,
        "cells": args.n * args.n * args.nz, "redundancy": res.redundancy,
        "gathered_bytes": res.gathered_bytes,
        "scattered_bytes": res.scattered_bytes,
        "initial_digest": initial, "total_steps": args.warmup + args.steps,
        "runs": runs, "digests_agree": agree,
    }
    _emit(payload)
    return payload


def stage_vram(args) -> None:
    _run_tiled_stage(args, on_device=True, label="tiled/VRAM-store")


def stage_host(args) -> None:
    _run_tiled_stage(args, on_device=False, label="tiled/host-store")


def stage_sweep(args) -> None:
    """Tile-size sweep on a host-resident store, re-seeded per configuration."""
    import cupy as cp

    tiles = [int(t) for t in args.tiles.split(",")]
    _banner(f"TILE SWEEP {args.n}x{args.n}x{args.nz} halo {args.halo}")
    store, cfg = _seeded_store(args.n, args.nz, args.seed)
    initial = _digest_of(store)
    shadow = None
    if args.write_mode == "shadow":
        shadow = hoststore.HostDomainStore(cfg,
                                           budget_bytes=int(40 * 2 ** 30))
    print(f"    initial digest {initial[:16]}, shadow "
          f"{0.0 if shadow is None else shadow.nbytes / 2**30:.2f} GiB")
    sys.stdout.flush()

    mode = "prefetch" if args.pipeline == "both" else args.pipeline
    cells = args.n * args.n * args.nz

    # The runner is rebuilt for every (rep, tile) rather than built once and
    # kept.  Keeping all of them alive would need 37 GB of tile buffers at
    # 1950^2 for tiles 256..1024, so the sweep would silently drop its largest
    # and most interesting points.  Rebuilding costs one warm-up sweep each
    # time and buys two things: bounded residency, and an INTERLEAVED
    # schedule (rep-major, tile-minor) so drift in the machine's state cannot
    # map onto tile size -- the variable under test.
    rows: dict[int, dict] = {}

    def build(tile):
        return bench.TiledRunner(store, cfg, tile, tile, halo=args.halo,
                                 nbuffers=args.nbuffers, shadow=shadow,
                                 pipeline=mode, write_mode=args.write_mode)

    # Correctness first: every tile size must reproduce the same trajectory.
    for tile in list(tiles):
        try:
            runner = build(tile)
        except Exception as exc:               # VRAM too small for this tile
            print(f"    tile {tile:5d}: SKIPPED {type(exc).__name__}: "
                  f"{str(exc)[:110]}")
            cp.get_default_memory_pool().free_all_blocks()
            tiles.remove(tile)
            continue
        bench.seed_host_store(store, cfg, seed=args.seed)
        assert _digest_of(store) == initial, "re-seed did not reproduce the state"
        runner._src, runner._dst = runner.home, runner._dst_home()
        runner.sweep(args.warmup)
        runner.sweep(args.steps)
        runner.settle()
        d = _digest_of(store)
        g, s_ = runner.bytes_per_sweep()
        free, _ = bench.free_vram_gib()
        rows[tile] = {"tile": tile, "tiles": len(runner.specs),
                      "cnx": runner.cnx, "redundancy": runner.redundancy,
                      "digest": d, "samples": [], "moved_bytes": g + s_,
                      "computed_cells": runner.tile_cfg.nz * runner.cny
                      * runner.cnx * len(runner.specs)}
        print(f"    tile {tile:5d}: {len(runner.specs):4d} tiles, cmp "
              f"{runner.cnx}^2, redundancy {runner.redundancy:.3f}x, "
              f"{(g + s_) / 1e9:.2f} GB/sweep, VRAM free {free:.2f} GiB, "
              f"digest {d[:16]}")
        sys.stdout.flush()
        runner.free()
        cp.get_default_memory_pool().free_all_blocks()

    probes = [bench.gpu_probe()]
    attribution: dict[int, dict] = {}
    for rep in range(args.reps):
        for tile in tiles:
            runner = build(tile)
            runner.sweep(args.warmup)
            rows[tile]["samples"].append(runner.sweep(args.steps).wall_ms)
            if rep == args.reps - 1:
                attribution[tile] = runner.attribute(args.steps)
            runner.free()
            cp.get_default_memory_pool().free_all_blocks()
        probes.append(bench.gpu_probe())

    print(f"    GPU during timing: util "
          f"{min(p.get('util_pct', 0) for p in probes):.0f}-"
          f"{max(p.get('util_pct', 0) for p in probes):.0f}%, "
          f"mem {min(p.get('mem_used_mib', 0) for p in probes):.0f}-"
          f"{max(p.get('mem_used_mib', 0) for p in probes):.0f} MiB")
    print(f"    {'tile':>6s} {'tiles':>6s} {'cmp':>8s} {'red':>7s} "
          f"{'ms/step':>10s} {'ns/cell':>9s} {'ns/cmp':>9s} {'GB/s':>7s} "
          f"{'spread':>7s} {'ovl-eff':>8s}")
    for tile in tiles:
        row = rows[tile]
        st = bench.repeat_stats(row["samples"])
        row["ms_per_step"] = st["min"]
        row["stats"] = st
        row["ns_per_cell"] = st["min"] * 1e6 / cells
        row["ns_per_computed_cell"] = st["min"] * 1e6 / row["computed_cells"]
        row["pcie_gbps"] = row["moved_bytes"] / (st["min"] * 1e-3) / 1e9
        row["attribution"] = attribution.get(tile, {})
        print(f"    {tile:6d} {row['tiles']:6d} {row['cnx']:6d}^2 "
              f"{row['redundancy']:6.3f}x {row['ms_per_step']:10.2f} "
              f"{row['ns_per_cell']:9.3f} {row['ns_per_computed_cell']:9.3f} "
              f"{row['pcie_gbps']:7.1f} {st['spread']:7.3f} "
              f"{row['attribution'].get('overlap_efficiency', float('nan')):8.2f}")
        sys.stdout.flush()
    rows = [rows[t] for t in tiles]

    digests = {r["digest"] for r in rows}
    best = min(rows, key=lambda r: r["ns_per_cell"]) if rows else None
    print(f"    digests agree across tile sizes: {len(digests) == 1} "
          f"({len(digests)} distinct)")
    if best:
        print(f"    optimum: tile {best['tile']} at {best['ns_per_cell']:.3f} "
              f"ns/cell")
    _emit({"stage": "sweep", "n": args.n, "nz": args.nz, "halo": args.halo,
           "rows": rows, "digests_agree": len(digests) == 1,
           "total_steps": args.warmup + args.steps})


def stage_headline(args) -> None:
    """A domain that cannot fit in VRAM at all."""
    _banner(f"HEADLINE {args.n}x{args.n}x{args.nz} (> VRAM) tile {args.tile}")
    cells = args.n * args.n * args.nz
    print(f"    {cells/1e6:.1f} Mcell; resident would need "
          f"{cells * 163.0 / 2**30:.1f} GiB of the 31.8 GiB card")
    args.tiles = str(args.tile)
    _run_tiled_stage(args, on_device=False, label="tiled/host-store HEADLINE")


def stage_overcommit(args) -> None:
    """The do-nothing alternative: let the driver page a too-big domain.

    On WSL2 the WDDM driver will accept an allocation past physical VRAM and
    back it with host memory.  It runs.  This measures what it costs, because
    "just allocate it anyway" is the obvious thing a reader will ask about and
    it deserves a number rather than an assertion.
    """
    import cupy as cp

    from tilestream import driver

    _banner(f"OVERCOMMIT monolithic {args.n}x{args.n}x{args.nz}")
    cfg = harness.make_config(args.n, args.n, args.nz)
    cells = args.n * args.n * args.nz
    print(f"    {cells/1e6:.1f} Mcell; resident needs ~"
          f"{cells * 163.0 / 2**30:.1f} GiB of the 31.8 GiB card")
    sys.stdout.flush()
    try:
        state = driver.make_tile_state(cfg)
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:
        print(f"    REFUSED: {type(exc).__name__}: {str(exc)[:200]}")
        _emit({"stage": "overcommit", "n": args.n, "refused": True,
               "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        return
    free, _ = bench.free_vram_gib()
    print(f"    DomainState allocated; VRAM free {free:.2f} GiB")
    sys.stdout.flush()
    # The state and the per-step scratch are allocated at DIFFERENT times, and
    # they fail differently.  MEASURED at 2200^2: the 26.4 GiB state is
    # accepted, and then dycore.step's first call asks for ~9.0 GiB of scratch
    # with 4.05 GiB left and raises.  So "it allocated" is not "it runs", and
    # the two outcomes are reported separately.
    try:
        res = bench.time_monolithic(state, cfg, steps=args.steps, warmup=1,
                                    label=f"overcommit {args.n}^2",
                                    hash_after=False)
    except Exception as exc:
        print(f"    STEP FAILED after the state allocated: "
              f"{type(exc).__name__}: {str(exc)[:150]}")
        _emit({"stage": "overcommit", "n": args.n, "nz": args.nz,
               "cells": cells, "refused": True, "state_allocated": True,
               "free_after_state_gib": free,
               "error": f"{type(exc).__name__}: {str(exc)[:150]}"})
        return
    print(f"    {res.ms_per_step:.1f} ms/step  {res.ns_per_cell:.3f} ns/cell")
    _emit({"stage": "overcommit", "n": args.n, "nz": args.nz, "cells": cells,
           "ms_per_step": res.ms_per_step, "ns_per_cell": res.ns_per_cell,
           "refused": False, "state_allocated": True})


# --------------------------------------------------------------------------

_STAGES = {
    "selftest": stage_selftest,
    "baseline": stage_baseline,
    "vram": stage_vram,
    "host": stage_host,
    "sweep": stage_sweep,
    "headline": stage_headline,
    "overcommit": stage_overcommit,
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=sorted(_STAGES))
    p.add_argument("--n", type=int, default=1950)
    p.add_argument("--nz", type=int, default=49)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--tiles", type=str, default="256,384,512,768")
    p.add_argument("--halo", type=int, default=16)
    p.add_argument("--nbuffers", type=int, default=2)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=harness.DEFAULT_SEED)
    p.add_argument("--reps", type=int, default=3,
                   help="interleaved repetitions of each timed configuration; "
                        "the MINIMUM is reported (see bench.repeat_stats)")
    p.add_argument("--write-mode", choices=("ring", "shadow"),
                   default="ring", dest="write_mode",
                   help="ring keeps ONE store plus a few per cent; shadow "
                        "keeps two. Both are correct and give identical "
                        "digests; only ring reaches the big domains.")
    p.add_argument("--pipeline", choices=("prefetch", "naive", "both"),
                   default="both",
                   help="copy/compute issue order; 'both' times each and "
                        "checks they give the same digest")
    args = p.parse_args(argv)
    if args.steps % 2 or args.warmup % 2:
        raise SystemExit("--steps and --warmup must be EVEN: the shadow buffer "
                         "swaps per sweep and an odd count leaves the live "
                         "state in the shadow, forcing a full-domain copy "
                         "that would be timed as if it were tiling overhead")
    _STAGES[args.stage](args)


if __name__ == "__main__":
    main()
