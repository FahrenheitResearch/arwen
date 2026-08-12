"""N GPUs streaming tiles out of ONE pinned host domain.

:mod:`tilestream.driver` streams every tile of a host-resident domain through
ONE GPU.  ``multigpu`` gives each GPU a sub-domain that lives PERMANENTLY in
its VRAM and exchanges halos over the peer link.  This module is the case
nobody had run: the domain stays in pinned host RAM, and *G* GPUs each pull
their own share of the tiles out of that one store.


THE HALO EXCHANGE DISAPPEARS, AND THAT IS THE POINT
---------------------------------------------------
The brief for this lane warned that "a sub-domain's halo exchange and its tile
gathers are two different things touching the same edges".  On a host-resident
domain they are not two things.  They are one.

With sub-domains resident in VRAM, GPU 0 cannot see GPU 1's cells at all, so
the seam needs an explicit transport: pack the band, ``cudaMemcpyPeerAsync``,
unpack.  With the domain in host RAM every GPU can already address every cell.
A tile whose window crosses the boundary between GPU 0's slab and GPU 1's slab
gathers the neighbouring cells the same way it gathers its own -- one
``cudaMemcpy3DAsync`` out of the shared store, halo and interior together, with
no idea that a boundary was crossed.  There is no pack, no peer copy, no
unpack, and no seam code to get wrong.

What survives is the ORDERING obligation, which was never really about
transport: every tile must read its neighbours as they were at the start of
the sweep (:mod:`tilestream.driver`'s read-at-time-t rule).  Across GPUs that
obligation is identical to the single-GPU one, because it is a property of the
STORE, not of who is reading it.  So the entire multi-GPU correctness question
reduces to: is the time-t invariant still enforced when the readers are on
different devices?  That is what ``write_mode`` decides, and the two answers
behave very differently.


``write_mode="shadow"`` -- PARALLEL
-----------------------------------
Two stores.  Every tile reads ``src`` and writes ``dst``; nothing a tile writes
is ever read in the same sweep.  There are therefore NO cross-tile hazards at
all, on one GPU or on eight, and the workers need exactly one barrier per
sweep.  This is the design that parallelises, and it is what the scaling
numbers in this module were taken with.  It costs a second host store.

``write_mode="ring"`` -- SEQUENTIAL, and this is a real limitation
------------------------------------------------------------------
:mod:`tilestream.rings` removes the second store by keeping ONE store and
saving each tile's outer band before it is overwritten, so a later tile can be
patched back to time-t values.  "Later" is the load-bearing word: the scheme
is defined over a TOTAL ORDER on tiles, and tile *k* patches from the saved
rings of its neighbours that come before it.  Those dependencies are real
device-side dependencies, and they do not care that two tiles are on different
GPUs.

With a block partition (GPU 0 takes tiles ``0..m-1``, GPU 1 takes ``m..``),
GPU 1's FIRST tile has tile ``m-1`` -- GPU 0's LAST tile -- among its patch
dependencies.  So GPU 1 cannot start until GPU 0 has essentially finished, and
two GPUs take as long as one.  Striping (``partition="stripe"``) turns the
whole-slab stall into a per-tile one, which pipelines a little and still does
not scale.  This is not a bug in this module and no scheduling fixes it: the
ring is a Gauss-Seidel sweep and a Gauss-Seidel sweep is sequential.

The honest cure for a memory-constrained multi-GPU box is not to parallelise
the ring but to give each GPU its OWN sub-domain store with its own halo band
and its own private ring, and to refresh those bands between steps -- i.e. to
put :mod:`tilestream.multigpu`'s decomposition in host RAM instead of VRAM.
Then every ring dependency is local to one GPU.  That design is sketched in
``subdomain_plan`` and is NOT implemented here; what is implemented is the
shared-store case, because that is the one that answers the bandwidth
question, and the bandwidth question is what decides whether any of this is
worth doing.


ORDERING ACROSS DEVICES, when the ring is used anyway
-----------------------------------------------------
Two things have to be true and only one of them is CUDA's problem.

*Device side.*  ``cudaStreamWaitEvent`` accepts an event recorded on another
device, so the dependency itself crosses GPUs with no special handling.

*Host side.*  It does NOT accept an event that has not been recorded yet: a
wait on an unrecorded event is a no-op, silently.  In the single-GPU driver
that can never happen, because one thread submits tiles in order and the
dependency always names an earlier tile.  With one thread per GPU that
guarantee is gone -- GPU 1's thread can reach its wait before GPU 0's thread
has recorded anything -- and the failure is invisible: the run completes, the
fields look like weather, and the digest is wrong only sometimes.  So each
tile also carries a plain ``threading.Event`` that is set at SUBMISSION time,
and a worker blocks on that before issuing the CUDA wait.  It cannot deadlock
because every dependency names a lower-numbered tile and every worker walks
its own tiles in increasing order.

``cross_worker_sync="none"`` drops exactly that host-side handshake and keeps
everything else.  It is the negative control for the ring path: with it the
two-GPU digest must DIFFER from the monolithic one, and if it does not, the
ordering being tested was never load-bearing and a "bit-exact" result proves
nothing.

The shadow path has exactly one ordering obligation -- the per-sweep barrier,
without which a fast worker starts reading the next sweep's ``src`` while a
slow one is still writing it -- and ``sweep_barrier=False`` is its negative
control.  Both live in :func:`tilestream.test_mgstream.negative_controls`.
A negative control that does not fire is reported as a failure of the control,
not as a pass: this project has been fooled before by an ordering that looked
enforced and was not.


PHYSICS ON, AND WHERE TWO CARDS ACTUALLY BROKE
----------------------------------------------
Everything above was measured DRY -- nine arrays, no ``PhysicsDriver``, no
scalar clock.  ``tile_state_factory=`` is the seam that lets a buffer carry
physics (it was hard-wired to ``driver.make_tile_state``, which attaches no
driver, so ``dycore.step`` raised at every rung above dry).  With it,
:mod:`tilestream.test_mgphys` streams the whole ``full fast cadence`` carrier
set -- radiation firing every step, cumulus every other -- and the TRANSPORT
is unchanged by physics: one worker, and two workers on ONE card, reproduce a
resident run bit for bit in ring and shadow, block and stripe.

What broke was not here.  Four physics schemes memoize DEVICE arrays (or
device-bound kernel handles) in caches keyed on everything except the device,
so the second card in a process dereferences the first card's pointers:
``rrtmgp.GasTables/CloudTables.to_device``, ``kf._device_table``,
``mynn_pbl_runtime._VALIDITY_FLAGS``, ``noahmp_vegeflux_gpu._module`` and
``noahmp_slab_libm._KERNEL_CACHE``.  Diagnosed with an order control -- touch
device 0 first and device 1 dies, touch device 1 first and device 0 dies --
and fixed by keying each on the current device.  Two consequences worth
carrying:

* an illegal address destroys the CUDA context for the whole PROCESS, so ONE
  such step makes every later run in that process fail at some unrelated
  allocation.  A multi-GPU worker that fails must therefore DRAIN before it
  aborts its barriers, or its in-flight DMA lands on the pinned store the
  caller is about to free -- which is how the first symptom was produced.
* the per-process lookup tables are now per CARD.  ``preflight
  .k_distribution_bytes`` counts them once per process and is, from here on,
  a PER-DEVICE figure.
"""

from __future__ import annotations

import threading
import time
import warnings

import numpy as np

from tilestream import driver as _driver
from tilestream import gather as _gather
from tilestream import harness as _harness
from tilestream import spec as _spec


class MGStreamError(RuntimeError):
    """A multi-GPU streaming setup that cannot be right."""


# --------------------------------------------------------------------------
# who runs what
# --------------------------------------------------------------------------

def partition_tiles(ntiles: int, nworkers: int, mode: str = "block"):
    """``(order, owner)`` -- the global write order and who owns each slot.

    ``order`` is a permutation of the planned tiles; index *k* below means
    "the k-th tile in write order", which is the numbering
    :mod:`tilestream.rings` builds its dependency lists over.  ``owner[k]`` is
    the worker that runs it.

    ``"block"``
        Contiguous runs, so each worker owns a connected slab of the domain --
        the closest thing to "each GPU owns a sub-domain".  The write order is
        unchanged from the plan's.
    ``"stripe"``
        Round robin.  Neighbouring tiles land on different GPUs, which is the
        worst case for locality and the best case for exposing an ordering
        bug, so the gate runs it.
    """
    if mode == "block":
        order = list(range(ntiles))
        edges = [round(i * ntiles / nworkers) for i in range(nworkers + 1)]
        owner = [0] * ntiles
        for w in range(nworkers):
            for k in range(edges[w], edges[w + 1]):
                owner[k] = w
        return order, owner
    if mode == "stripe":
        order = list(range(ntiles))
        return order, [k % nworkers for k in range(ntiles)]
    raise ValueError(f"partition must be 'block' or 'stripe', got {mode!r}")


def subdomain_plan(cfg, nsub: int, halo: int | None = None) -> dict:
    """Sizing for the NOT-IMPLEMENTED per-GPU sub-domain store design.

    Kept as arithmetic rather than prose so the trade-off can be checked:
    each sub-domain carries its own ``halo`` band on both sides, so the
    redundant compute is ``2*halo/T`` and the exchange traffic per step is two
    bands per seam.  Returned for the report; nothing here runs it.
    """
    h = int(_harness.halo_radius(cfg) if halo is None else halo)
    t = cfg.nx // nsub
    return {
        "nsub": nsub, "halo": h, "slab_nx": t, "array_nx": t + 2 * h,
        "compute_redundancy": (t + 2 * h) / t,
        "seam_cells_per_step": 2 * nsub * h * cfg.ny * cfg.nz,
    }


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run_mgstream(store, cfg, tile_nx, tile_ny, halo: int = 16, nsteps: int = 1,
                 *, devices=(0,), nbuffers: int = 2, periodic: bool = True,
                 periodic_x: bool | None = None,
                 periodic_y: bool | None = None,
                 write_mode: str = "shadow", partition: str = "block",
                 pipeline: str = "prefetch", shadow=None,
                 allow_pageable: bool = False, poison: bool = True,
                 tile_state_factory=None,
                 names=None, inventory_fn=None, nz=None, scalars=None,
                 geography=None, geography_names=None,
                 check_geography: bool = True,
                 impose_geography_flags: bool = True,
                 ring_margin: str = "exact",
                 cross_worker_sync: str = "events",
                 sweep_barrier: bool = True,
                 report: dict | None = None) -> None:
    """Integrate ``store`` for ``nsteps``, streaming its tiles through ``devices``.

    Same contract as :func:`tilestream.driver.run_tiled` -- ``store`` holds the
    whole domain in pinned host RAM and is updated IN PLACE -- with the tiles
    shared out over ``devices``.  ``devices=(0,)`` must reproduce ``run_tiled``
    bit for bit, and ``devices=(0, 0)`` (two workers, one GPU) must too; both
    are gated, because they separate "the partition is wrong" from "two
    physical GPUs interfere".
    """
    import cupy as cp
    from gpuwm.core.dycore import step

    if write_mode not in ("ring", "shadow"):
        raise ValueError(f"write_mode must be 'ring' or 'shadow', "
                         f"got {write_mode!r}")
    if cross_worker_sync not in ("events", "none"):
        raise ValueError("cross_worker_sync must be 'events' or 'none'")
    devices = list(devices)
    nworkers = len(devices)
    if nworkers < 1:
        raise ValueError("need at least one device")
    nsteps = int(nsteps)
    nbuffers = max(1, int(nbuffers))

    home = _driver._arrays_of(store, names, inventory_fn)
    if not home:
        raise MGStreamError("store holds none of the persisted attributes")
    dz, dy, dx = _gather.domain_extents(home, nz=nz)
    for axis, got, want in (("nz", dz, cfg.nz), ("ny", dy, cfg.ny),
                            ("nx", dx, cfg.nx)):
        if int(got) != int(want):
            raise MGStreamError(
                f"store {axis}={got} but cfg.{axis}={want}")
    nz_arg = nz

    # PER AXIS, for the reason ``tilestream.spec``'s module docstring gives:
    # ``open_x`` without ``open_y`` is non-periodic in x and WRAPPING in y,
    # and a plan built from one boolean clamps the axis the kernels wrap.
    # ``periodic=`` still sets both, so every existing caller is unchanged.
    specs0 = _spec.plan_tiles(dx, dy, int(tile_nx), int(tile_ny), int(halo),
                              periodic, periodic_x=periodic_x,
                              periodic_y=periodic_y)
    _spec.validate_plan(specs0, dy, dx)
    order, owner = partition_tiles(len(specs0), nworkers, partition)
    specs = [specs0[i] for i in order]
    ntiles = len(specs)
    if ntiles < nworkers:
        raise MGStreamError(
            f"{ntiles} tiles cannot keep {nworkers} GPUs busy; use a smaller "
            "tile or a bigger domain")

    cnx, cny = specs[0].cnx, specs[0].cny
    tile_cfg = _harness.tile_config(cfg, cnx, cny)
    need = _harness.halo_radius(cfg)
    if int(halo) < need:
        warnings.warn(
            f"halo={halo} is below the per-step dependency radius {need}",
            RuntimeWarning, stacklevel=2)

    # Each worker's tiles, in the global write order.
    mine = [[k for k in range(ntiles) if owner[k] == w]
            for w in range(nworkers)]
    for w, ks in enumerate(mine):
        if not ks:
            raise MGStreamError(f"worker {w} was given no tiles")
    # slot[k] is the buffer index tile k lands in -- position within its OWN
    # worker's list, because that is what decides which stream serves it.
    slot = [0] * ntiles
    for ks in mine:
        for pos, k in enumerate(ks):
            slot[k] = pos % nbuffers

    take = inventory_fn or _gather.inventory
    # PHYSICS.  This used to be hard-wired to make_tile_state, which attaches
    # no PhysicsDriver, so dycore.step raised "physics is enabled but the
    # state has no PhysicsDriver" at every rung above dry and NOTHING had ever
    # run multi-GPU with physics on.  It is the same seam run_tiled has
    # (driver.py:1115) and takes the same buffer builders --
    # driver.make_physics_tile_state for a carrier-set run, and the closure
    # driver.geography_run_kwargs builds for a real projection.  Each worker
    # calls it nbuffers times on ITS OWN device, so a builder that allocates
    # (and make_physics_tile_state does: it runs a warm-up step so Kain-
    # Fritsch's lazily-allocated cumulus/w0avg exists before the inventory is
    # matched) lands its allocations on the right card.
    factory = tile_state_factory or _driver.make_tile_state

    # -- the second store, or the ring ------------------------------------
    ring = None
    other = None
    if write_mode == "shadow":
        other = shadow if shadow is not None else _driver._empty_like_store(
            home, poison=poison)
        if not isinstance(other, dict):
            other = _driver._arrays_of(other, names, inventory_fn)
        for name, arr in home.items():
            if name not in other or tuple(other[name].shape) != tuple(arr.shape):
                raise MGStreamError(
                    f"shadow buffer is missing or mis-shaped for {name!r}")

    geo_home = None
    geo_flags = {}
    if geography is not None:
        geo_home = _driver.geography_inventory(geography, geography_names)
        if not geo_home:
            raise MGStreamError("geography= holds no gatherable arrays")
        gz, gy, gx = _gather.domain_extents(geo_home, nz=nz_arg)
        if (gy, gx) != (dy, dx):
            raise MGStreamError(
                f"geography describes {gy}x{gx}, store holds {dy}x{dx}")
        geo_flags = _driver.geography_scalars(geo_home)

    # -- shared state ------------------------------------------------------
    buffers = {"src": home, "dst": home if other is None else other}
    stats = [dict(gathered=0, scattered=0, saved=0, patched=0, geo=0,
                  geo_gathers=0, tiles=0, seconds=0.0) for _ in range(nworkers)]
    worker_tiles: list[list] = [None] * nworkers          # type: ignore
    errors: list[BaseException | None] = [None] * nworkers
    setup_lock = threading.Lock()
    ring_events: list = [None] * ntiles
    recorded = [threading.Event() for _ in range(ntiles)]
    clock_box = {"clock": None if scalars is None else dict(scalars)}
    _physics = None
    if clock_box["clock"] is not None:
        from tilestream import physics_inventory as _physics   # noqa: F811

    depth = nbuffers - 1 if pipeline == "prefetch" else 0

    ready = threading.Barrier(nworkers)
    swept = threading.Barrier(nworkers)
    swapped = threading.Barrier(nworkers)

    # The ring arena needs ONE set of tile buffers to resolve its block lists
    # against; every worker's buffers have the same shapes and the arena
    # resolves the tile-side pointer by NAME at issue time, so one arena
    # serves every device.  It is built after worker 0 has its buffers.
    arena_box: dict = {}

    def _worker(w: int) -> None:
        try:
            dev = devices[w]
            cp.cuda.Device(dev).use()
            with setup_lock:
                tiles = [factory(tile_cfg) for _ in range(nbuffers)]
                tinvs = [take(t, names) for t in tiles]
                for inv in tinvs:
                    if set(inv) != set(home):
                        raise MGStreamError(
                            f"tile inventory != store inventory; missing "
                            f"{sorted(set(home) - set(inv))}, extra "
                            f"{sorted(set(inv) - set(home))}")
                if geo_home is not None:
                    for t in tiles:
                        _driver._pin_scheme_geography(t)
                        dst_geo = _driver.geography_inventory(
                            t, geography_names)
                        if set(dst_geo) != set(geo_home):
                            raise MGStreamError(
                                "tile geography inventory != domain's")
                        if impose_geography_flags:
                            for nm, val in geo_flags.items():
                                setattr(t, nm, val)
                    if check_geography and w == 0:
                        _driver.assert_geography_gathered(
                            tiles[0], keys=geography_names)
                worker_tiles[w] = tiles
                streams = [cp.cuda.Stream(non_blocking=True)
                           for _ in range(nbuffers)]
                if write_mode == "ring" and w == 0:
                    from tilestream import rings as _rings
                    kinds = sorted({k for _n, k, _d, _i
                                    in _rings.field_geometry(home, nz=nz_arg)})
                    plan = _rings.build_ring_plan(specs, kinds,
                                                  margin_mode=ring_margin)
                    arena_box["ring"] = _rings.RingArena(
                        plan, home, tiles, nz=nz_arg, names=names,
                        inventory_fn=inventory_fn,
                        allow_pageable=allow_pageable)
                if write_mode == "ring":
                    for k in mine[w]:
                        ring_events[k] = cp.cuda.Event()

            # MEASURED HAZARD (driver.py has the long form): a non-blocking
            # stream does not synchronise with the legacy default stream, and
            # every buffer above was built on it.  Once per device.
            cp.cuda.runtime.deviceSynchronize()
            ready.wait()
            _ring = arena_box.get("ring")
            geo_tile: list[int | None] = [None] * nbuffers
            st = stats[w]
            my = mine[w]

            def gather_into(pos: int) -> None:
                k = my[pos]
                tspec = specs[k]
                b = pos % nbuffers
                stream = streams[b]
                with stream:
                    if geo_home is not None and geo_tile[b] != k:
                        st["geo"] += _gather.gather_tile(
                            geo_home, tiles[b], tspec, stream,
                            allow_pageable=allow_pageable,
                            names=geography_names,
                            inventory_fn=_driver.geography_inventory,
                            nz=nz_arg).nbytes
                        geo_tile[b] = k
                        st["geo_gathers"] += 1
                    st["gathered"] += _gather.gather_tile(
                        buffers["src"], tiles[b], tspec, stream,
                        allow_pageable=allow_pageable, names=names,
                        inventory_fn=inventory_fn, nz=nz_arg).nbytes
                    if _ring is not None:
                        tinv = take(tiles[b], names)
                        st["saved"] += _ring.save(k, tinv, stream)
                        ring_events[k].record(stream)
                        if cross_worker_sync == "events":
                            recorded[k].set()
                        for j in _ring.plan.patch_deps[k]:
                            if owner[j] == w and slot[j] == slot[k]:
                                continue          # same stream, already ordered
                            if cross_worker_sync == "events":
                                recorded[j].wait()
                            stream.wait_event(ring_events[j])
                        st["patched"] += _ring.patch(k, tinv, stream)

            t_start = time.perf_counter()
            for istep in range(nsteps):
                for pos in range(min(depth, len(my))):
                    gather_into(pos)
                for pos, k in enumerate(my):
                    if depth:
                        nxt = pos + depth
                        if nxt < len(my):
                            gather_into(nxt)
                    else:
                        gather_into(pos)
                    b = pos % nbuffers
                    stream = streams[b]
                    if clock_box["clock"] is not None:
                        _physics.set_carrier_scalars(tiles[b],
                                                     clock_box["clock"])
                    with stream:
                        step(tiles[b], tile_cfg)
                        if _ring is not None:
                            for j in _ring.plan.war_deps[k]:
                                if owner[j] == w and slot[j] == slot[k]:
                                    continue
                                if cross_worker_sync == "events":
                                    recorded[j].wait()
                                stream.wait_event(ring_events[j])
                        st["scattered"] += _gather.scatter_tile(
                            tiles[b], buffers["dst"], specs[k], stream,
                            allow_pageable=allow_pageable, names=names,
                            inventory_fn=inventory_fn, nz=nz_arg).nbytes
                    st["tiles"] += 1
                for s in streams:
                    s.synchronize()
                cp.cuda.runtime.deviceSynchronize()
                if sweep_barrier:
                    swept.wait()
                if w == 0:
                    for ev in recorded:
                        ev.clear()
                    if clock_box["clock"] is not None:
                        used = []
                        for ww in range(nworkers):
                            used.extend(worker_tiles[ww][:min(
                                nbuffers, len(mine[ww]))])
                        clock_box["clock"] = _driver._advance_clock(
                            clock_box["clock"], used, len(used) - 1, _physics)
                    if other is not None:
                        buffers["src"], buffers["dst"] = (buffers["dst"],
                                                          buffers["src"])
                if sweep_barrier:
                    swapped.wait()
            st["seconds"] = time.perf_counter() - t_start
        except BaseException as exc:                        # noqa: BLE001
            errors[w] = exc
            # DRAIN BEFORE ABORTING, and this is not defensive tidiness.
            # MEASURED on the dual-4090 box: worker 1 died in setup, worker 0
            # unblocked from ``ready.wait()`` with BrokenBarrierError while
            # its gathers were still in flight against the caller's PINNED
            # store, the caller then freed that store in its except handler,
            # and the DMA landed on unmapped pages -- cudaErrorIllegalAddress,
            # which destroys the CUDA context for the whole PROCESS.  Every
            # subsequent run in that process then failed at an unrelated
            # cudaHostAlloc, so one setup failure looked like six independent
            # multi-GPU failures.  The sync itself can only wait on work
            # already submitted, so it cannot deadlock on a peer.
            try:
                cp.cuda.runtime.deviceSynchronize()
            except BaseException:                           # noqa: BLE001
                pass
            for bar in (ready, swept, swapped):
                bar.abort()

    threads = [threading.Thread(target=_worker, args=(w,), daemon=True,
                                name=f"mgstream-w{w}")
               for w in range(nworkers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    # WHICH worker to blame.  A worker that fails aborts every barrier, so
    # every OTHER worker then dies with BrokenBarrierError -- and reporting
    # the lowest-numbered failure would name the innocent one and chain the
    # barrier abort as the cause, hiding the real exception entirely.  This
    # cost an hour once: worker 1 died building a physics tile buffer on
    # device 1 and the run reported "worker 0 (device 0) failed:
    # BrokenBarrierError", which is true and useless.  A barrier abort is
    # never a root cause, so it is only reported when it is the ONLY thing
    # that happened.
    failed = [(w, e) for w, e in enumerate(errors) if e is not None]
    if failed:
        # Second drain, from the calling thread, over every device this run
        # touched: a worker killed by something the try/except above could
        # not run past (a hard abort) still has queued work, and the caller's
        # very next act is normally to free the store.  See the worker's
        # except handler for what that costs.
        for dev in dict.fromkeys(devices):
            try:
                cp.cuda.Device(dev).use()
                cp.cuda.runtime.deviceSynchronize()
            except BaseException:                           # noqa: BLE001
                pass
    real = [(w, e) for w, e in failed
            if not isinstance(e, threading.BrokenBarrierError)]
    if failed:
        w, exc = (real or failed)[0]
        also = [f"worker {ww} ({type(ee).__name__})"
                for ww, ee in failed if ww != w]
        raise MGStreamError(
            f"worker {w} (device {devices[w]}) failed"
            + (f"; also down: {', '.join(also)}" if also else "")) from exc

    if other is not None and buffers["src"] is not home:
        # Odd number of sweeps left the newest data in the shadow.
        _driver._copy_into(home, buffers["src"])
    if clock_box["clock"] is not None and scalars is not None:
        scalars.clear()
        scalars.update(clock_box["clock"])

    if report is not None:
        gathered = sum(s["gathered"] for s in stats)
        scattered = sum(s["scattered"] for s in stats)
        geo_bytes = sum(s["geo"] for s in stats)
        report.update(
            tiles=ntiles, steps=nsteps, nbuffers=nbuffers, halo=int(halo),
            write_mode=write_mode, partition=partition, pipeline=pipeline,
            devices=list(devices), nworkers=nworkers,
            cross_worker_sync=cross_worker_sync,
            tile_cfg=tile_cfg, domain=(dz, dy, dx),
            compute=(tile_cfg.nz, cny, cnx),
            gathered_bytes=gathered, scattered_bytes=scattered,
            geography_bytes=geo_bytes,
            host_bytes=gathered + scattered + geo_bytes,
            efficiency=_spec.plan_efficiency(specs),
            fields=len(home), seconds=wall, scalars=clock_box["clock"],
            per_worker=[dict(s, device=devices[w], ntiles=len(mine[w]))
                        for w, s in enumerate(stats)],
            host_gbs=(gathered + scattered + geo_bytes) / wall / 1e9,
            second_store_bytes=(0 if other is None else
                                sum(int(a.nbytes) for a in other.values())),
        )
        rng = arena_box.get("ring")
        if rng is not None:
            report.update(ring_bytes=rng.nbytes,
                          ring_saved_bytes=sum(s["saved"] for s in stats),
                          ring_patched_bytes=sum(s["patched"] for s in stats))

    for w in range(nworkers):
        worker_tiles[w] = None
    arena_box.clear()


# --------------------------------------------------------------------------
# throughput
# --------------------------------------------------------------------------

def cells_per_second(report: dict) -> float:
    """Interior cells advanced one step per wall second.

    The honest throughput measure when GPUs are added: it is per-DOMAIN, not
    per-GPU, so redundant halo compute does not inflate it.
    """
    nz, ny, nx = report["domain"]
    return nz * ny * nx * report["steps"] / report["seconds"]
