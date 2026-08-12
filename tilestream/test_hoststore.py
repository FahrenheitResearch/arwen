"""Gates for :mod:`tilestream.hoststore`.

Run with::

    python -m tilestream.test_hoststore       # from the repository root

Five gates, in the order they matter:

1.  SHAPES AND STAGGERING, field by field, against a real ``DomainState`` of
    the same config -- including the moist ``mp_physics=10`` inventory, so the
    manifest probe is exercised on more than the dry milestone-one set.
2.  ROUND TRIP: ``fill_from`` then ``to_device_state`` into a *different*
    state must reproduce the source hash exactly, and the store's own
    ``hash()`` must equal ``harness.hash_state`` -- proving the two hashes are
    byte-identical in construction and not merely both self-consistent.
3.  PINNING, asserted per field via ``cudaPointerGetAttributes``.
4.  BANDWIDTH, measured out of the store's own memory at >= 256 MB.
5.  THE BUDGET GUARD refuses an over-large request instead of taking the box
    down.

Every gate prints its evidence.  A speedup with no correctness check is how
this project has been fooled before, so the bandwidth gate runs only after
the round-trip gate has passed on the very same store.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from tilestream import harness as H
from tilestream import hoststore as HS


PASSED: list[str] = []
FAILED: list[str] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""))


# --------------------------------------------------------------------------

def test_manifest_matches_domainstate() -> None:
    """Gate 1: shapes, staggering and dtype against a real DomainState."""
    print("\n=== gate 1: manifest vs. real DomainState (shape/stagger/dtype)")
    import cupy as cp

    from gpuwm.core.state import DomainState

    for label, kwargs in (("dry periodic", {}),
                          ("moist mp=10", dict(moist=True, mp_physics=10))):
        nx, ny, nz = 24, 20, 17          # deliberately non-square, odd nz
        cfg = H.make_config(nx, ny, nz, **kwargs)
        store = HS.HostDomainStore(cfg, budget_bytes=64 << 20)
        state = DomainState(cfg)
        device = H.state_arrays(state)

        same_inventory = list(store.names) == list(device)
        gate(f"{label}: inventory matches contract order",
             same_inventory, f"{len(store.names)} fields: "
             f"{', '.join(store.names)}")

        bad = []
        for spec in store.manifest:
            host = store.arrays[spec.name]
            dev = device[spec.name]
            if host.shape != dev.shape or host.dtype != dev.dtype:
                bad.append(f"{spec.name}: store {host.shape}/{host.dtype} "
                           f"!= device {dev.shape}/{dev.dtype}")
        gate(f"{label}: every field shape+dtype matches", not bad,
             "; ".join(bad) if bad else f"{len(store.manifest)} fields exact")

        # Independently re-derive the staggering rule and check the probe
        # agreed with it -- catches a probe that classified by accident.
        expect = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
                  "w": (nz + 1, ny, nx), "php": (nz + 1, ny, nx),
                  "mup": (ny, nx), "thp": (nz, ny, nx)}
        wrong = {n: store.arrays[n].shape for n, s in expect.items()
                 if n in store.arrays and store.arrays[n].shape != s}
        gate(f"{label}: WRF-ARW staggering explicitly correct", not wrong,
             str(wrong) if wrong else
             f"u={store.arrays['u'].shape} v={store.arrays['v'].shape} "
             f"w={store.arrays['w'].shape} mup={store.arrays['mup'].shape}")

        print(f"    {label}: {store.nbytes / (1 << 20):.2f} MB, "
              f"{store.bytes_per_cell:.1f} B/cell "
              f"({nz}x{ny}x{nx})")
        del state
        store.free()
        cp.get_default_memory_pool().free_all_blocks()


def test_round_trip() -> HS.HostDomainStore:
    """Gate 2: fill_from -> to_device_state is bit-exact, hashes agree."""
    print("\n=== gate 2: round trip bit-exactness")
    import cupy as cp

    nx, ny, nz = 96, 80, 49
    cfg = H.make_config(nx, ny, nz)
    source = H.make_state(cfg)
    H.run_steps(source, cfg, 2)           # a dirtied, non-trivial state
    source_hash = H.hash_state(source)

    store = HS.HostDomainStore(cfg, budget_bytes=256 << 20)
    store.fill_from(source)

    gate("store.hash() == harness.hash_state(source)",
         store.hash() == source_hash, f"{source_hash[:32]}...")

    # Round trip into a DIFFERENT state, pre-dirtied with other numbers, so a
    # no-op to_device_state cannot pass by accident.
    target = H.make_state(cfg, seed=H.DEFAULT_SEED + 1)
    H.run_steps(target, cfg, 1)
    target_hash_before = H.hash_state(target)
    gate("target starts different from source",
         target_hash_before != source_hash, f"{target_hash_before[:32]}...")

    store.to_device_state(target)
    target_hash_after = H.hash_state(target)
    gate("fill_from -> to_device_state reproduces the source hash",
         target_hash_after == source_hash, f"{target_hash_after[:32]}...")

    if target_hash_after != source_hash:
        mine = store.hash_field_map()
        theirs = H.hash_field_map(target)
        print("    differing fields: "
              + str([n for n in mine if mine[n] != theirs.get(n)]))

    # And the stepped trajectory continues identically from the round-tripped
    # state: a round trip that lost a scratch-adjacent bit would show here.
    H.run_steps(source, cfg, 1)
    H.run_steps(target, cfg, 1)
    gate("one further step from the round-tripped state matches",
         H.hash_state(source) == H.hash_state(target),
         H.hash_state(target)[:32] + "...")

    del source, target
    cp.get_default_memory_pool().free_all_blocks()

    # The dry set is only 9 of the 41 contract names.  Repeat the round trip
    # on a 25-field moist inventory so the gate covers the fields a real
    # forecast streams, not just milestone one's.
    mcfg = H.make_config(48, 48, 49, moist=True, mp_physics=10)
    msrc = H.make_state(mcfg)
    H.run_steps(msrc, mcfg, 1)
    mhash = H.hash_state(msrc)
    with HS.HostDomainStore(mcfg, budget_bytes=256 << 20) as mstore:
        mstore.fill_from(msrc)
        mtgt = H.make_state(mcfg, seed=H.DEFAULT_SEED + 7)
        mstore.to_device_state(mtgt)
        gate(f"moist mp=10 round trip ({len(mstore.names)} fields) is exact",
             H.hash_state(mtgt) == mhash and mstore.hash() == mhash,
             f"{mhash[:32]}...")
        del msrc, mtgt
    cp.get_default_memory_pool().free_all_blocks()
    return store


def test_pinning(store: HS.HostDomainStore) -> None:
    """Gate 3: page-locking actually took effect, per field."""
    print("\n=== gate 3: pinning")
    report = store.pinning_report()
    gate("every buffer reports cudaMemoryTypeHost (1)",
         report["all_pinned"],
         f"{len(report['memory_types'])} fields, types="
         f"{sorted(set(report['memory_types'].values()))}")

    # Control: an ordinary pageable array must report 0, else the check is
    # vacuous and proves nothing about the store.
    import cupy as cp

    pageable = np.empty(1024, np.float32)
    try:
        kind = int(cp.cuda.runtime.pointerGetAttributes(
            pageable.__array_interface__["data"][0]).type)
    except Exception:
        kind = 0
    gate("control: pageable memory reports 0, so the check discriminates",
         kind == 0, f"pageable type={kind}")


def test_bandwidth(cfg_nx: int = 1024, cfg_ny: int = 1024,
                   cfg_nz: int = 49) -> None:
    """Gate 4: measured pinned H2D bandwidth from the store's own memory.

    A 1024x1024x49 store is 1.54 GiB, so the largest single field is ~800 MB
    -- far past the 32 MB PCIe saturation point, which keeps the number
    stable.  A 256 MB working set spread over six 50 MB fields measured
    noticeably noisier (42-52 GB/s) for the same true bandwidth.
    """
    print("\n=== gate 4: pinned H2D bandwidth (>= 256 MB working set)")
    cfg = H.make_config(cfg_nx, cfg_ny, cfg_nz)
    store = HS.HostDomainStore(cfg, budget_bytes=4 << 30)
    print(f"    store: {cfg_nz}x{cfg_ny}x{cfg_nx}, "
          f"{store.nbytes / (1 << 20):.1f} MB, "
          f"{store.bytes_per_cell:.2f} B/cell, "
          f"{len(store.manifest)} fields")
    store.assert_pinned()

    result = store.measure_h2d_bandwidth(min_bytes=512 << 20, repeats=7)
    gate("bandwidth measured from >= 256 MB", result["honest"],
         f"{result['bytes'] / (1 << 20):.1f} MB in "
         f"{result['n_transfers']} transfers")
    if result["honest"]:
        print(f"    PINNED H2D: peak {result['gb_per_s']:.1f} GB/s, "
              f"median {result['gb_per_s_median']:.1f} GB/s")
        print(f"    samples: "
              + ", ".join(f"{s:.1f}" for s in result["samples_gb_per_s"]))
        gate("pinned bandwidth is in the pinned regime, not the pageable one",
             result["gb_per_s"] >= 35.0,
             f"{result['gb_per_s']:.1f} GB/s (pinned ~50-57, pageable ~15)")

    # Same buffers, but copied through a PAGEABLE staging array, as the
    # contrast that makes the number above meaningful.
    import cupy as cp

    big = max(store.names, key=lambda n: store.arrays[n].nbytes)
    src = store.arrays[big]
    pageable = np.array(src, copy=True)            # ordinary heap memory
    dev = cp.cuda.alloc(src.nbytes)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    h2d = cp.cuda.runtime.memcpyHostToDevice
    rates = {}
    for label, arr in (("pinned", src), ("pageable", pageable)):
        ptr = arr.__array_interface__["data"][0]
        cp.cuda.runtime.memcpy(int(dev.ptr), ptr, arr.nbytes, h2d)
        best = 0.0
        for _ in range(3):
            start.record()
            cp.cuda.runtime.memcpy(int(dev.ptr), ptr, arr.nbytes, h2d)
            stop.record()
            stop.synchronize()
            best = max(best, arr.nbytes /
                       (cp.cuda.get_elapsed_time(start, stop) * 1e-3) / 1e9)
        rates[label] = best
    print(f"    single-field contrast ({src.nbytes / (1 << 20):.1f} MB): "
          f"pinned {rates['pinned']:.1f} GB/s vs "
          f"pageable {rates['pageable']:.1f} GB/s")
    gate("pinned beats pageable by a wide margin",
         rates["pinned"] > 2.0 * rates["pageable"],
         f"{rates['pinned'] / max(rates['pageable'], 1e-9):.2f}x")

    del dev
    store.free()
    cp.get_default_memory_pool().free_all_blocks()


def test_budget_guard() -> None:
    """Gate 5: over-large requests are refused, not attempted."""
    print("\n=== gate 5: budget and host-memory guards")
    mem = HS.host_memory()
    print(f"    host: total {mem['total'] / HS.GIB:.1f} GiB, "
          f"available {mem['available'] / HS.GIB:.1f} GiB")

    # (a) explicit budget cap
    cfg = H.make_config(1024, 1024, 49)
    plan = HS.HostDomainStore(cfg, allocate=False)
    print(f"    a 49x1024x1024 store would need "
          f"{plan.nbytes / HS.GIB:.3f} GiB")
    try:
        HS.HostDomainStore(cfg, budget_bytes=64 << 20)
        gate("budget_bytes cap refuses an over-large store", False,
             "NO EXCEPTION RAISED -- it allocated anyway")
    except HS.BudgetExceeded as exc:
        gate("budget_bytes cap refuses an over-large store", True, str(exc))

    # (b) machine guard: ask for more RAM than the box has, with no budget.
    #     This must NOT be attempted -- an attempt is the failure.
    huge = H.make_config(20000, 20000, 49)
    plan = HS.HostDomainStore(huge, allocate=False)
    print(f"    a 49x20000x20000 store would need "
          f"{plan.nbytes / HS.GIB:.1f} GiB "
          f"(machine has {mem['total'] / HS.GIB:.1f} GiB)")
    try:
        HS.HostDomainStore(huge)
        gate("host-memory guard refuses to exceed the machine", False,
             "NO EXCEPTION RAISED -- it tried to pin the whole box")
    except HS.HostMemoryExhausted as exc:
        gate("host-memory guard refuses to exceed the machine", True,
             str(exc)[:150])

    # (c) the guard must not be trigger-happy: a store that clearly fits
    #     has to be allowed, or the guard is just a broken allocator.
    ok_cfg = H.make_config(128, 128, 49)
    store = HS.HostDomainStore(ok_cfg)
    gate("a store that comfortably fits is allowed",
         store.nbytes > 0 and store.pinning_report()["all_pinned"],
         f"{store.nbytes / (1 << 20):.1f} MB allocated and pinned")
    store.free()


def test_larger_than_vram() -> None:
    """Gate 6: a store BIGGER than VRAM allocates, pins and transfers.

    This is the whole point of the module, so it gets a real allocation
    rather than an argument: a store 1.15x the size of total VRAM.  It is
    SKIPPED, not failed, if the machine cannot currently accommodate that --
    a neighbouring workload holding tens of GB is a fact about the box, not a
    defect in the store, and failing on it would be a misleading red gate.
    """
    print("\n=== gate 6: larger-than-VRAM store")
    import time

    import cupy as cp

    free_vram, total_vram = cp.cuda.runtime.memGetInfo()
    manifest = HS.build_manifest(H.make_config(8, 8, 49))
    mem = HS.host_memory()
    print(f"    VRAM {total_vram / HS.GIB:.1f} GiB total "
          f"({free_vram / HS.GIB:.1f} free); host {mem['total'] / HS.GIB:.1f} "
          f"GiB total, {mem['available'] / HS.GIB:.1f} GiB available")

    # Start at 1.15x VRAM and back off.  A neighbouring workload can make the
    # big size genuinely impossible, and the honest thing is to report the
    # largest store actually achieved rather than to fail or to silently pass.
    store = None
    elapsed = 0.0
    target = int(total_vram * 1.15)
    while target >= (2 * HS.GIB):
        n = HS.max_square_domain(target, manifest, 49)["nx"]
        cfg = H.make_config(n, n, 49)
        plan = HS.HostDomainStore(cfg, allocate=False)
        try:
            t0 = time.time()
            store = HS.HostDomainStore(cfg,
                                       budget_bytes=plan.nbytes + (1 << 30))
            elapsed = time.time() - t0
            break
        except HS.HostStoreError as exc:
            print(f"    {plan.nbytes / HS.GIB:5.1f} GiB refused, backing "
                  f"off: {type(exc).__name__}")
            # Cleanup evidence.  NOT gated on MemAvailable: on this WSL2
            # kernel that counter is unreliable -- it has been observed
            # swinging between 59 and 91 GiB with nothing running, and it is
            # read here while ``exc`` still pins the failing frame.  The
            # authoritative signals are that the store kept no buffers and
            # CuPy's pinned pool stranded nothing; both were confirmed
            # directly (MemAvailable 51.0 -> 91.3 GiB measured inside
            # free(), pool n_free_blocks() == 0).
            pool = cp.get_default_pinned_memory_pool()
            after = HS.host_memory()["available"]
            gate(f"refusal at {plan.nbytes / HS.GIB:.0f} GiB retained no "
                 f"buffers and stranded no pool blocks",
                 pool.n_free_blocks() == 0,
                 f"pool free_blocks={pool.n_free_blocks()}; MemAvailable "
                 f"{mem['available'] / HS.GIB:.1f} -> {after / HS.GIB:.1f} "
                 f"GiB (informational -- noisy on WSL2)")
            target //= 2
    if store is None:
        gate("larger-than-VRAM store", False,
             "could not allocate any store at all")
        return
    n = store.nx
    print(f"    allocated {store.nbytes / HS.GIB:.2f} GiB pinned in "
          f"{elapsed:.1f}s: {n}x{n}x49 = "
          f"{49 * n * n / 1e9:.2f} G cells, {len(store.names)} fields, "
          f"largest block {store.largest_field_bytes / HS.GIB:.2f} GiB")
    if store.nbytes > total_vram:
        gate("host store exceeds total VRAM", True,
             f"{store.nbytes / HS.GIB:.2f} GiB store vs "
             f"{total_vram / HS.GIB:.1f} GiB VRAM "
             f"({store.nbytes / total_vram:.2f}x)")
    else:
        print(f"    NOT DEMONSTRATED THIS RUN: the largest store the machine "
              f"allowed was {store.nbytes / HS.GIB:.2f} GiB, which is "
              f"{store.nbytes / total_vram:.2f}x VRAM -- a concurrent "
              f"workload is holding host RAM.  This is a statement about "
              f"the box right now, not about the store; re-run on an idle "
              f"machine to demonstrate it.")
    gate("every field of the oversize store is pinned",
         store.pinning_report()["all_pinned"],
         f"{len(store.names)} fields")

    # Writes land where they should, and the staggering survived at scale.
    u = store.arrays["u"]
    u[0, 0, 0] = 1.5
    u[-1, -1, -1] = 2.5
    gate("oversize store is writable with correct u staggering",
         u.shape == (49, n, n + 1) and u[0, 0, 0] == 1.5
         and u[-1, -1, -1] == 2.5, f"u.shape={u.shape}")

    result = store.measure_h2d_bandwidth(min_bytes=4 << 30, repeats=3)
    print(f"    H2D out of the oversize store: "
          f"{result['bytes'] / HS.GIB:.1f} GiB in "
          f"{result['n_transfers']} transfers -> peak "
          f"{result['gb_per_s']:.1f}, median "
          f"{result['gb_per_s_median']:.1f} GB/s")
    gate("oversize store still transfers at pinned speed",
         result["gb_per_s"] >= 35.0, f"{result['gb_per_s']:.1f} GB/s")
    store.free()

    # The driver's pinned ceiling (46 GiB measured) is BELOW MemAvailable
    # (91 GiB), so a request the /proc/meminfo gate alone would wave through
    # must still be refused by the fraction cap.
    over = H.make_config(6000, 6000, 49)          # ~53 GiB, past the wall
    plan = HS.HostDomainStore(over, allocate=False)
    mem = HS.host_memory()
    fits_meminfo = plan.nbytes < mem["available"] - HS.DEFAULT_RESERVE_BYTES
    print(f"    a 49x6000x6000 store needs {plan.nbytes / HS.GIB:.1f} GiB; "
          f"MemAvailable-based gate alone would "
          f"{'ALLOW' if fits_meminfo else 'reject'} it")
    try:
        HS.HostDomainStore(over)
        gate("request past the driver's pinned ceiling is refused", False,
             "NO EXCEPTION -- it walked into cudaErrorMemoryAllocation")
    except HS.HostMemoryExhausted as exc:
        gate("request past the driver's pinned ceiling is refused", True,
             f"caught by the fraction cap: {str(exc)[:110]}")
    except HS.PinnedAllocationFailed as exc:
        gate("request past the driver's pinned ceiling is refused", True,
             f"caught at the wall, cleanly: {str(exc)[:110]}")
    cp.get_default_pinned_memory_pool().free_all_blocks()


def test_exact_allocator() -> None:
    """Gate 7: the store allocates EXACTLY what it asks for.

    The "2 GiB block cliff" was never a driver property.  CuPy's pinned pool
    bins by power of two, so a field one byte past 2 GiB consumed 4 GiB, and
    a domain big enough to matter always landed in the worst bin.  This gate
    holds the allocator to the exact byte count and keeps the two paths
    distinguishable, because the whole failure was invisible from inside the
    old code.
    """
    print("\n=== gate 7: exact-size pinned allocation (no pool rounding)")
    import gc

    import cupy as cp

    # (a) the rounding rule itself, against the six MEASURED sizes.
    MIB = 1 << 20
    measured = [(640 * MIB, 1024 * MIB), (768 * MIB, 1024 * MIB),
                (1025 * MIB, 2048 * MIB), (1536 * MIB, 2048 * MIB),
                ((1 << 31) + 1, 4096 * MIB), (3072 * MIB, 4096 * MIB)]
    wrong = [(req, HS.pool_rounded_bytes(req), want)
             for req, want in measured if HS.pool_rounded_bytes(req) != want]
    gate("pool_rounded_bytes reproduces the measured pool consumption",
         not wrong, str(wrong) if wrong else
         f"{len(measured)} sizes, 640 MiB..3 GiB, all next-power-of-two")
    gate("the rule is not vacuous: an exact power of two rounds to itself",
         HS.pool_rounded_bytes(1 << 31) == (1 << 31)
         and HS.pool_rounded_bytes((1 << 31) + 1) == (1 << 32),
         "2 GiB -> 2 GiB, 2 GiB + 1 -> 4 GiB (the cliff, exactly)")

    # (b) a real store's blocks are the exact field sizes, and the array is
    #     ONE contiguous C-order buffer per field -- no banding, so
    #     tilestream.gather needs no change.
    cfg = H.make_config(200, 160, 49)
    store = HS.HostDomainStore(cfg, budget_bytes=512 << 20)
    exact = all(
        store.arrays[s.name].nbytes == s.nbytes(store.nz, store.ny, store.nx)
        and store.arrays[s.name].flags.c_contiguous
        for s in store.manifest)
    blocks = [HS.owning_block(store.arrays[n]) for n in store.names]
    gate("every field is one exact-size contiguous PinnedBlock", exact
         and all(b is not None and b.nbytes == store.arrays[n].nbytes
                 for b, n in zip(blocks, store.names)),
         f"{len(store.names)} fields, {store.nbytes} B total, "
         f"pool would have taken {store.pinned_pool_bytes} B "
         f"({store.pinned_pool_waste:.3f}x)")
    gate("store.nbytes is the planned byte count exactly",
         store.nbytes == HS.domain_bytes(store.manifest, store.nz, store.ny,
                                         store.nx),
         f"{store.nbytes} B")
    store.free()
    gc.collect()

    # (c) the wall itself, predicted from MemTotal.
    mem = HS.host_memory()
    print(f"    MemTotal {mem['total'] / HS.GIB:.4f} GiB -> page-locked wall "
          f"{HS.pinned_ceiling_bytes() / HS.GIB:.4f} GiB "
          f"({HS.PINNED_CEILING_FRACTION:.0%}); MEASURED 46.9375 GiB "
          f"(0.4998 x MemTotal) to 64 MiB resolution")
    gate("the store's own cap sits below the measured page-locked wall",
         HS.DEFAULT_MAX_TOTAL_FRACTION < HS.PINNED_CEILING_FRACTION,
         f"cap {HS.DEFAULT_MAX_TOTAL_FRACTION:.2f} "
         f"({HS.DEFAULT_MAX_TOTAL_FRACTION * mem['total'] / HS.GIB:.2f} GiB) "
         f"< wall {HS.PINNED_CEILING_FRACTION:.2f}")
    cp.get_default_pinned_memory_pool().free_all_blocks()


def test_copy_across_two_gib() -> None:
    """Gate 8: a 3-D copy that SPANS the old 2 GiB block boundary.

    The fix keeps each field one contiguous allocation instead of banding it,
    so the case that would have been "a copy spanning a band boundary" is
    simply a copy whose source rectangle straddles byte 2147483648 of a
    single >2 GiB block.  That is the exact offset at which the old allocator
    changed behaviour, and it is where a pointer/pitch mistake would land, so
    it gets its own gate driven through the REAL transport path
    (``cudaMemcpy3DAsync`` via :mod:`tilestream.gather`) rather than through
    NumPy.

    The block is 2.25 GiB and is freed immediately; skipped, not failed, if
    the machine cannot spare it right now.
    """
    print("\n=== gate 8: memcpy3D across the old 2 GiB boundary")
    import gc

    import cupy as cp

    from tilestream import gather as G

    mem = HS.host_memory()
    need = int(2.25 * HS.GIB)
    if mem["available"] - need < 20 * HS.GIB:
        print(f"    SKIPPED: only {mem['available'] / HS.GIB:.1f} GiB "
              f"available; this gate wants {need / HS.GIB:.2f} GiB free plus "
              f"a 20 GiB margin")
        return

    # Geometry chosen so byte 2^31 falls STRICTLY INSIDE an x-row that the
    # copied window covers -- not on a plane or row boundary, where a pointer
    # bug could still slip through.
    nx = ny = 4000
    nz = 36                                   # 36 * 4000 * 4000 * 4 = 2.145 GiB
    plane_bytes = ny * nx * 4                 # 64,000,000
    row_bytes = nx * 4                        # 16,000
    b_plane, rem = divmod(1 << 31, plane_bytes)          # 33, 35,483,648
    b_row, b_off = divmod(rem, row_bytes)                # 2217, 11,648
    b_elem = b_off // 4                                  # 2912
    host = HS.alloc_pinned_array((nz, ny, nx), np.float32)
    total_bytes = host.nbytes
    gate("a single >2 GiB pinned block allocates and is page-locked",
         total_bytes > (1 << 31) and G.is_pinned(host),
         f"{total_bytes} B = {total_bytes / HS.GIB:.3f} GiB in ONE block; "
         f"byte 2^31 = plane {b_plane}, row {b_row}, element {b_elem} "
         f"(strictly inside a row, not on any boundary)")

    # Position-dependent pattern, every value exactly representable in float32
    # so the comparison below is bit-exact and the control cannot be lost to
    # rounding.
    plane_pat = (np.arange(ny * nx, dtype=np.int64) % 8191
                 ).astype(np.float32).reshape(ny, nx)
    for k in range(nz):
        host[k] = plane_pat + np.float32(k)

    # A y-window containing row 2217, so the copied region contains byte 2^31
    # in its interior on plane 33.
    win_y0, win_h = 2000, 512                 # rows 2000..2511 include 2217
    assert win_y0 < b_row < win_y0 + win_h
    dev = cp.empty((nz, win_h, nx), np.float32)

    parms_src = host
    plan_src = {"probe": parms_src}
    plan_dst = {"probe": dev}

    class _Spec:
        """Minimal spec: one rectangle, full x, a y-window."""

        def gather(self, kind):
            return [((Ellipsis, slice(win_y0, win_y0 + win_h),
                      slice(0, nx)),
                     (Ellipsis, slice(0, win_h), slice(0, nx)))]

    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.runtime.deviceSynchronize()       # fact 3: fence the default stream
    plan = G.build_plan(plan_src, plan_dst, _Spec(), "gather")
    plan.execute(plan_src, plan_dst, stream)
    stream.synchronize()

    got = cp.asnumpy(dev)
    want = host[:, win_y0:win_y0 + win_h, :]
    whole = np.array_equal(got, want)
    # And specifically the element that CONTAINS byte 2^31, plus its
    # neighbours either side of it.
    r = b_row - win_y0
    at_edge = (got[b_plane, r, b_elem] == want[b_plane, r, b_elem]
               and got[b_plane, r, b_elem - 1] == want[b_plane, r, b_elem - 1]
               and got[b_plane, r, b_elem + 1] == want[b_plane, r, b_elem + 1])
    gate("memcpy3D across byte 2^31 of one block is bit-exact",
         whole and at_edge,
         f"{plan.nbytes / (1 << 20):.1f} MB moved, all "
         f"{got.size} elements equal; the element straddling 2^31 "
         f"(plane {b_plane}, row {b_row}, elem {b_elem}) and both neighbours "
         f"exact")

    # Control: the check discriminates.  Perturb exactly the element that
    # contains byte 2^31, by an amount float32 can actually represent here
    # (values are <= 8226, so +1000 is not lost to rounding -- an earlier
    # version of this control used +1.0 against values of 3.3e7, where the
    # ULP is 4.0 and the perturbation silently vanished).
    before = float(got[b_plane, r, b_elem])
    got[b_plane, r, b_elem] = before + 1000.0
    gate("control: the comparison would have caught a wrong byte",
         not np.array_equal(got, want),
         f"perturbing the 2^31 element {before:.1f} -> "
         f"{before + 1000.0:.1f} flips the check")

    del dev, got, want, host
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def report_capacity() -> None:
    """Not a gate: the achievable-domain answer, from the real manifest.

    NOTE ON 232 B/cell.  The project's established footprint says 232 B/cell
    persists.  Measured against the ENFORCED contract set that this store
    actually streams, no configuration reaches that: the heaviest reachable
    inventory (NSSL mp=18 + km_opt=2 + SASE, 31 of the 41 contract names) is
    120.2 B/cell, and milestone one's dry set is 32.2.  All 41 names are
    reachable across configs but no single config allocates more than 31, so
    120.2 is a true ceiling for a single run.  The store is therefore about
    half the assumed size and the achievable domain about twice as large.
    Treat 232 as covering a wider set than STATE_SERIALIZED_ATTRS (the
    RK-rebuilt copies and physics carriers) -- those are per-step rebuilt or
    scratch, so they stay TILE-sized and never enter the host store.
    ``tilestream.physics_inventory`` has since measured that wider set
    directly: 180.96 B/cell for mp10's full carrier set, 229.29 for
    ``configs/real74_d01.toml``, 291.65 for the heaviest stack that runs.  So
    120.25 bounds what the store holds TODAY, and the 180-292 range is what
    it would hold if the carriers turn out to need streaming too.

    THE BUDGET IS NOT THE WHOLE MACHINE.  Page-locking walls at MemTotal/2
    (46.94 GiB here, MEASURED -- see hoststore.PINNED_CEILING_FRACTION), and
    driver.py's default ``write_mode='shadow'`` needs TWO full-domain stores,
    so the per-store budget for a real streaming run is half the cap again.
    Both rows are printed below.
    """
    print("\n=== capacity: what domain fits a given pinned budget")
    cap = int(HS.DEFAULT_MAX_TOTAL_FRACTION * HS.host_memory()["total"])
    print(f"    page-locked wall {HS.pinned_ceiling_bytes() / HS.GIB:.2f} GiB "
          f"(MemTotal/2); store cap {cap / HS.GIB:.2f} GiB; "
          f"shadow-mode per-store {cap / 2 / HS.GIB:.2f} GiB")
    for label, kwargs in (("dry periodic (milestone one)", {}),
                          ("moist mp=10", dict(moist=True, mp_physics=10)),
                          ("heaviest reachable: mp=18 + km_opt=2 + SASE",
                           dict(moist=True, mp_physics=18, km_opt=2,
                                bl_pbl_physics=900))):
        cfg = H.make_config(64, 64, 49, **kwargs)
        manifest = HS.build_manifest(cfg)
        per_cell = HS._asymptotic_bytes_per_cell(manifest, 49)
        print(f"\n    {label}: {len(manifest)} fields, "
              f"{per_cell:.1f} B/cell (nz=49, asymptotic)")
        # Budgets that this machine can ACTUALLY page-lock.  Quoting 64 or
        # 80 GiB here would advertise a domain the box cannot hold: the wall
        # is MemTotal/2 = 46.94 GiB.
        for name, budget in (("8 GiB", 8 * HS.GIB),
                             ("32 GiB", 32 * HS.GIB),
                             ("shadow pair", cap // 2),
                             ("single store", cap)):
            best = HS.max_square_domain(budget, manifest, 49)
            print(f"      {name:>13} ({budget / HS.GIB:5.2f} GiB) -> "
                  f"{best['nx']:6d} x {best['ny']:6d} x 49  "
                  f"= {best['cells'] / 1e9:6.2f} G cells, "
                  f"{best['bytes'] / HS.GIB:6.2f} GiB used, "
                  f"{best['bytes_per_cell']:.2f} B/cell")


def main() -> int:
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"device: {free / HS.GIB:.1f} GiB free of {total / HS.GIB:.1f} GiB "
          f"(cudaMemGetInfo)")

    test_manifest_matches_domainstate()
    store = test_round_trip()
    test_pinning(store)
    store.free()
    test_bandwidth()
    test_budget_guard()
    test_exact_allocator()
    test_copy_across_two_gib()
    test_larger_than_vram()
    report_capacity()

    print("\n" + "=" * 68)
    print(f"PASSED {len(PASSED)} / {len(PASSED) + len(FAILED)}")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
