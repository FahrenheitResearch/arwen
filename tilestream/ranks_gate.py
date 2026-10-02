"""Same-card ranked integration evidence, driven through StreamedDomain."""
from __future__ import annotations
import argparse
from dataclasses import dataclass, replace
from types import SimpleNamespace
import json
import os
import numpy as np

from tilestream import driver, gather, harness, physics_inventory as physics
from tilestream import multigpu
from gpuwm.core import streaming
from gpuwm.core.devices import DeviceOptions


class _SingleCardGateOptions(DeviceOptions):
    """Select the requested card for a one-rank reference, without a table."""

    def __init__(self, card):
        super().__init__()
        object.__setattr__(self, "_card", int(card))

    def device_ids(self):
        return (self._card,)


def config(nx=128, ny=104, nz=20, rung="lean"):
    values = dict(moist=True, mp_physics=8, km_opt=4, bl_pbl_physics=1,
                  sf_sfclay_physics=91, sf_surface_physics=2, ra_physics=4,
                  radt_minutes=0.1, dt=1.0, time_step_sound=4, ztop=20000.0,
                  use_adaptive_time_step=True, max_time_step=6,
                  min_time_step=1, starting_time_step=1)
    if rung == "mynn":
        values.update(bl_pbl_physics=5, sf_sfclay_physics=5, sf_surface_physics=3,
                      num_soil_layers=9, ra_rrtmg_variant="rrtmg_legacy")
    return replace(multigpu.forced_config(nx, ny, nz), **values)


def fixture(cfg, *, wind=0.0):
    import cupy as cp
    from gpuwm.ingest.lateral_bc import build_state_lateral_boundaries, attach_lateral_boundaries
    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    state, _ = harness.make_physics_state(cfg, 4242, geography=geo)
    other, _ = harness.make_physics_state(cfg, 4243, geography=geo)
    if wind:
        from gpuwm.core.diagnostics import update_diagnostics
        for source in (state, other):
            source.u += np.float32(wind)
            update_diagnostics(source)
    bnd = build_state_lateral_boundaries([state, other], [0., 3600.],
        spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    del other
    attach_lateral_boundaries(state, bnd)
    streaming.prime_lazy_carriers(state, cfg)
    take = streaming.streamed_store_inventory()
    initial = {k: cp.asnumpy(a) for k, a in take(state).items()}
    store = {k: gather.pinned_empty(a.shape, a.dtype) for k, a in initial.items()}
    geography = driver.geography_store(state, host=True)
    # Assemble the pinned host store by scattering slab interiors, retaining
    # only the last slab as metadata. No full-device store copy is allocated.
    specs = multigpu.plan_split(cfg.nx, cfg.ny, 0, gx=1,
                               gy=max(1, (cfg.ny+15)//16), periodic=False)
    factory = streaming.prepared_tile_state_factory(state, cfg)
    flags = driver.geography_scalars(geography)
    template = None
    for window in specs:
        slab = factory(harness.tile_config(cfg, window.cnx, window.cny))
        driver._pin_scheme_geography(slab)
        gather.gather_tile(geography, slab, window,
            inventory_fn=driver.geography_inventory, nz=cfg.nz)
        for name, value in flags.items():
            setattr(slab, name, value)
        # The independent resident reference supplies the synthetic prepared
        # carriers. Production preparation reads these from its host cache.
        gather.gather_tile(initial, slab, window, allow_pageable=True,
                           inventory_fn=take, nz=cfg.nz)
        gather.scatter_tile(slab, store, window, inventory_fn=take, nz=cfg.nz)
        cp.cuda.runtime.deviceSynchronize()
        template = slab
    del initial
    cp.cuda.runtime.deviceSynchronize()
    bundle = SimpleNamespace(store=store, scalars=physics.carrier_scalars(state),
        geography=geography, boundaries=bnd, template=template)
    return state, bundle


def make_ranked(bundle, cfg, decision, options, clock, mode, *, short=False):
    if not short:
        return streaming.ranked_domain_builder(bundle, clock=clock, options=options,
            step_mode=mode)(None, cfg, decision)
    from tilestream.ranks import RankedRun
    run = RankedRun(bundle.store, cfg, options=options, scalars=bundle.scalars,
        geography=bundle.geography, template=bundle.template, clock=clock,
        boundaries=bundle.boundaries, step_mode=mode, halo=13, _unsafe_short_halo=True)
    streamed = streaming.StreamedDomain(run, decision, scalars=bundle.scalars,
        state=None, host_store=True, geography=bundle.geography,
        boundaries=bundle.boundaries, template=bundle.template,
        inventory_fn=streaming.streamed_store_inventory())
    streamed.ranked = True
    streamed.devices_report = run.devices_report
    return streamed


def integrate(cfg, grid=(1, 2), mode="sequential", nsteps=12, devices=None,
              control=None, timing=False, wind=None, change_live=True):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.state_digest import canonical_state_digest, canonical_store_digest
    visible = cp.cuda.runtime.getDeviceCount()
    cards = list(devices) if devices else list(range(min(2, visible)))
    if not cards or any(card < 0 or card >= visible for card in cards):
        raise ValueError(f"rank gate devices {cards}: visible card count is {visible}")
    # Build the resident reference and prepared template on the first
    # requested card, so a gate confined to another card never uses card 0.
    cp.cuda.Device(cards[0]).use()
    state, bundle = fixture(cfg, wind=(25.0 if control is not None else 0.0) if wind is None else wind)
    from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock
    resident_clock = _clock(cfg)
    ranked_clock = _clock(cfg)
    bind_lateral_boundary_clock(state, resident_clock)
    count = grid[0]*grid[1]
    ids = tuple(cards[rank % len(cards)] for rank in range(count))
    # The wrong-card control runs with peer access OFF between the cards: a
    # card-0 table read by a card-1 kernel through a live peer mapping is
    # correct by construction, so only the staged path can show the fault.
    transport = ("staged" if control == "wrong_card" and len(set(ids)) > 1
                 else "auto")
    options = (_SingleCardGateOptions(cards[0]) if count == 1 else
               DeviceOptions(count=count, grid=grid, ids=ids, transport=transport))
    decision = streaming.ranked_decision(cfg, options,
        max_map_factor=streaming.StreamedDomain.maximum_map_factor(
            SimpleNamespace(_state=None, _geography=bundle.geography)))
    streamed = make_ranked(bundle, cfg, decision, options, ranked_clock, mode,
                           short=control == "short")
    run = streamed.tiled_run
    original_exchange = run.exchange_events
    if control == "no_exchange":
        run.exchange_events = lambda: None
    elif control == "crossed":
        run.seams = multigpu.cross_seams(run.seams, run.specs, run.halo)
        run._build_channels()
    elif control == "mup":
        run.exchange_names.remove("state/mup")
        run._build_channels()
    restore_kernels = None
    seam_hits = None
    if control == "wrong_card":
        restore_kernels, seam_hits = _wrong_card_tables(ids[0])
    times = []
    try:
        live = cfg
        for index in range(nsteps):
            if index == nsteps//2 and control is None and change_live:
                live = replace(cfg, dt=0.5, time_step_sound=2)
            if control == "stale":
                run.exchange_events = original_exchange if index % 2 == 0 else lambda: None
            for clock in (resident_clock, ranked_clock):
                clock.step_ticks = int(round(live.dt * clock.tick_den))
                clock.dt_fp32 = np.float32(live.dt)
                if clock.lbc_reset_due():
                    clock.mark_force()
                clock.prepare_step()
            dycore.step(state, live)
            if timing:
                report = {"timing": True}
                run.sweep(1, live_config=live, report=report)
                times.append(report)
            else:
                streamed(None, live)
            resident_clock.advance()
            ranked_clock.advance()
            if control is not None:
                # Inspect without feeding the joined host frame back to ranks.
                # Otherwise a digest read itself repairs a missing exchange.
                sample = streamed.canonical_digest(ranked_clock)
                run._exposed = False
                resident_sample = canonical_state_digest(state, resident_clock)
                if (sample["sha256"] != resident_sample["sha256"] and
                        (control != "short" or index + 1 > multigpu.SHORT_HALO_VISIBLE_AT)):
                    break
        completed_steps = index + 1
        cp.cuda.runtime.deviceSynchronize()
        got = run.store
        ref = {k: cp.asnumpy(a) for k, a in streaming.streamed_store_inventory()(state).items()}
        differing = [key for key in ref if ref[key].tobytes() != got[key].tobytes()]
        scalars = physics.carrier_scalars(state)
        scalar_equal = scalars == bundle.scalars
        ref_digest = canonical_state_digest(state, resident_clock)
        got_digest = streamed.canonical_digest(ranked_clock)
        bad = sum(int(np.count_nonzero(~np.isfinite(v))) for v in ref.values()
                  if v.dtype.kind == "f")
        if bad:
            raise AssertionError(f"resident reference contains {bad} nonfinite values")
        if control is None and change_live and nsteps >= 12 and scalars["call_counts"]["radiation"] < 2:
            raise AssertionError("radiation did not fire twice on the resident reference")
        health_equal = None
        if control is None and streamed.stability is not None and not timing:
            rank_health = streamed.stability(cfg=live)
            resident_health = dycore.stability_report(state, live,
                boundary_width=int(cfg.spec_bdy_width) or None)
            health_equal = rank_health == resident_health
            if not health_equal:
                print(json.dumps(dict(resident_health=resident_health, ranked_health=rank_health)), flush=True)
        result = dict(config={name: getattr(cfg, name) for name in (
            "nx", "ny", "nz", "dx", "dy", "dt", "time_step_sound", "mp_physics",
            "bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics", "ra_physics",
            "ra_rrtmg_variant", "num_soil_layers", "radt_minutes")},
            health_equal=health_equal, grid=grid, mode=mode, control=control, completed_steps=completed_steps, differing=differing,
            scalar_equal=scalar_equal, resident_digest=ref_digest["sha256"], ranked_digest=got_digest["sha256"],
            resident_inventory=ref_digest["inventory_sha256"], ranked_inventory=got_digest["inventory_sha256"],
            resident_fires=scalars.get("call_counts"), ranked_fires=bundle.scalars.get("call_counts"),
            devices=run.devices_report(), timings=times)
        print(json.dumps(result, default=str), flush=True)
        equal = not differing and scalar_equal and ref_digest == got_digest
        if control is None and health_equal is False:
            raise AssertionError("rank-ordered stability fold differs from resident health")
        if control is None and not equal:
            raise AssertionError("ranked carriers, scalars or canonical digest differ from resident")
        if control is not None and equal:
            raise AssertionError(f"control {control} did not fire")
        return result
    finally:
        if seam_hits is not None:
            print("WRONG_CARD_SEAM " + json.dumps(seam_hits, sort_keys=True), flush=True)
        if restore_kernels is not None:
            restore_kernels()
        run.close()
        dycore.reset_wrf_cfl_recording()


def _wrong_card_tables(first_card):
    """THE WRONG-CARD CACHE CONTROL: every cached device table comes from one card.

    Reinstates the defect the per-card cache keys removed (9d13335605): an
    immutable device table the step reads -- Noah's parameter tables, the
    RTE-RRTMGP solar source and trace-gas rows, RUC's tables -- uploaded once
    on the FIRST card and handed to a slab stepping on another.  Every such
    table reaches the step through ``device_cache.cached_ready``, so that one
    name is rebound (in device_cache and in every module that imported it by
    name): the table is resolved under the first card whatever card asks.

    The first version of this control rebound ``get_kernel`` instead, and it
    could not fire on CuPy 14.2: a RawKernel resolved under card 0 loads
    itself for whichever card launches it (MEASURED on two RTX PRO 6000s,
    with peer access off and on).  A card-0 ARRAY read by a card-1 kernel
    with no access to it is a real fault, so the answer must change or the
    run must stop on a CUDA error.

    Peer access alone cannot be trusted to stay off: CuPy enables it by
    itself the first time an elementwise kernel is handed an array that
    lives on another card (``cupy._core._kernel._check_peer_access``), and
    after that a card-1 kernel reads card 0's table correctly, so the
    control would pass a broken split.  The first card's copies therefore
    live in its stream-ordered memory pool (``cudaMallocAsync``), whose
    allocations no other card can read until ``cudaMemPoolSetAccess``
    grants it; ``cudaDeviceEnablePeerAccess`` does not reach them, and
    neither CuPy nor gpuwm grants pool access.  :func:`integrate` still runs
    the control on the staged transport, which turns peer access off.

    Returns ``(undo, hits)``: ``hits["calls"]`` counts the rebound lookups
    the step made and ``hits["foreign"]`` those made from another card.  A
    control whose seam the step never reaches cannot fire, so the gate
    requires ``calls > 0`` on any card count, and ``foreign > 0`` on two.
    """
    import sys
    import cupy as cp
    from gpuwm.core import device_cache as _device_cache
    real = _device_cache.cached_ready
    poisoned = {}
    hits = {"calls": 0, "foreign": 0, "first_card": int(first_card)}
    with cp.cuda.Device(first_card):
        unshared = cp.cuda.MemoryAsyncPool("default")

    def on_first_card(cp_module, cache, key, factory):
        hits["calls"] += 1
        if int(cp.cuda.Device().id) != int(first_card):
            hits["foreign"] += 1

        def in_unshared_pool():
            with cp.cuda.using_allocator(unshared.malloc):
                return factory()

        with cp.cuda.Device(first_card):
            return real(cp_module, poisoned, (id(cache), repr(key)[:2000]),
                        in_unshared_pool)

    rebound = [_device_cache]
    _device_cache.cached_ready = on_first_card
    for module in list(sys.modules.values()):
        if module is _device_cache:
            continue
        if getattr(module, "cached_ready", None) is real:
            setattr(module, "cached_ready", on_first_card)
            rebound.append(module)

    def undo():
        for module in rebound:
            setattr(module, "cached_ready", real)
    return undo, hits


def wrong_card_child(devices):
    """Run the wrong-card control in this (disposable) process.

    A card-0 pointer dereferenced on card 1 with peer access off is
    cudaErrorIllegalAddress, which is STICKY: the process's CUDA context is
    unusable afterwards.  So the control runs in a child the gate spawns and
    reads, never in the gate's own process.  Prints one ``WRONG_CARD`` verdict
    line: FIRED-DIFFER, FIRED-REFUSED or NOT-FIRED.
    """
    control_cfg = _control_config()
    try:
        integrate(control_cfg, control="wrong_card", nsteps=12, devices=devices)
    except AssertionError as exc:
        verdict = ("NOT-FIRED" if str(exc) == "control wrong_card did not fire"
                   else "ERROR")
        print(f"WRONG_CARD {verdict} {str(exc).splitlines()[0][:200]}", flush=True)
        return 1
    except Exception as exc:
        # Only the wrong pointer's CUDA memory fault is a firing. A setup
        # failure, a missing table or a kernel compile error after the first
        # foreign lookup must still fail this gate.
        fault = False
        seen = set()
        cause = exc
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if (type(cause).__name__ == "CUDARuntimeError"
                    and type(cause).__module__.startswith("cupy")
                    and getattr(cause, "status", None) in (700, 719)):
                fault = True
                break
            cause = cause.__cause__ if cause.__cause__ is not None else cause.__context__
        print(f"WRONG_CARD {'FIRED-REFUSED' if fault else 'ERROR'} {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:200]}", flush=True)
        return 0 if fault else 1
    print("WRONG_CARD FIRED-DIFFER", flush=True)
    return 0


def _control_config():
    return replace(config(), dx=500.0, dy=500.0, dt=3.0,
                   starting_time_step=3, max_time_step=3, radt_minutes=12.0)


def run_wrong_card_control(cards):
    """Spawn :func:`wrong_card_child` and judge it.  Returns the verdict line.

    Two distinct cards: the child must FIRE (a CUDA error or a different
    answer) with ``foreign > 0`` rebound lookups, else this raises.  One
    card: the fault cannot exist, so the verdict is SKIPPED, and still only
    after the child shows the step reaches the rebound seam (``calls > 0``).
    """
    import subprocess
    import sys
    argv = [sys.executable, "-m", "tilestream.ranks_gate", "wrong-card"]
    for card in cards:
        argv += ["--devices", str(int(card))]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    lines = done.stdout.splitlines()
    verdict = next((l for l in lines if l.startswith("WRONG_CARD ")), None)
    seam = next((l for l in lines if l.startswith("WRONG_CARD_SEAM ")), None)
    hits = json.loads(seam.split(" ", 1)[1]) if seam else {}
    if not hits.get("calls"):
        raise AssertionError(
            "wrong-card control never reached the per-card table seam "
            f"(hits {hits}); a control the step does not reach cannot fire.  "
            f"child rc {done.returncode}: {done.stderr[-2000:]}")
    if len(set(cards)) < 2:
        # On one card every slab IS on the first card, so the rebinding and
        # the unshared pool must change nothing.  If they did, a firing on
        # two cards could be the instrument's own doing, not the cross-card
        # read it exists to catch.
        if done.returncode != 1 or verdict is None or not verdict.startswith(
                "WRONG_CARD NOT-FIRED control wrong_card did not fire"):
            raise AssertionError(
                "wrong-card control changed a one-card run, so its two-card "
                f"firing would prove nothing: {verdict} hits {hits}; child rc "
                f"{done.returncode}: {done.stderr[-2000:]}")
        return (f"CONTROL wrong_card: SKIPPED, one card; the step reached the "
                f"rebound table seam {hits['calls']} times and the rebound "
                "tables left the one-card answer unchanged; the fault needs "
                "two physical cards")
    if done.returncode != 0 or verdict is None or not verdict.startswith((
            "WRONG_CARD FIRED-DIFFER", "WRONG_CARD FIRED-REFUSED")):
        raise AssertionError(f"wrong-card control did not fire: {verdict} hits {hits}; "
                             f"child rc {done.returncode}")
    if not hits.get("foreign"):
        raise AssertionError(f"wrong-card control fired without a cross-card lookup: hits {hits}")
    return (f"CONTROL wrong_card: {verdict.split(' ', 2)[1]} (fired): "
            f"{verdict.split(' ', 2)[2] if verdict.count(' ') > 1 else ''} "
            f"foreign lookups {hits['foreign']} of {hits['calls']}").rstrip()


@dataclass(frozen=True)
class _NodeConfig:
    grid_id: int
    run: object
    parent_time_step_ratio: int = 1


def _clock(cfg):
    from gpuwm.core.clock import DomainClock, DomainTicks
    spec = DomainTicks(grid_id=cfg.grid_id, parent_id=0, parent_time_step_ratio=1,
        step_ticks=100, dt_fp32=np.float32(cfg.dt), history_ticks=0, restart_ticks=None,
        radt_ticks=None, stepra=None, cudt_ticks=None, stepcu=None, bldt_ticks=None,
        stepbl=None, lbc_interval_ticks=360000)
    return DomainClock(spec, 100, 1000000)


def adaptive_loop(*, poison=False, nsteps=12):
    """One-node production adaptive driver, clock recurrence and real solves."""
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.adaptive_clock import AdaptiveClockDriver, maximum_map_factor
    from gpuwm.core.physics_step_control import PhysicsStepControl
    from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock
    from gpuwm.state_digest import canonical_state_digest
    cfg = config()
    resident, bundle = fixture(cfg)
    ranked_cfg = replace(cfg, grid_id=2)
    clocks = [_clock(cfg), _clock(ranked_cfg)]
    bind_lateral_boundary_clock(resident, clocks[0])
    options = DeviceOptions(count=2, ids=(0, 0))
    streamed = make_ranked(bundle, ranked_cfg,
        streaming.ranked_decision(ranked_cfg, options), options, clocks[1], "threads")
    run = streamed.tiled_run
    nodes = [SimpleNamespace(cfg=_NodeConfig(1, cfg), clock=clocks[0], state=resident,
                             parent=None, children=[]),
             SimpleNamespace(cfg=_NodeConfig(2, ranked_cfg), clock=clocks[1],
                             state=run.tiles[0], parent=None, children=[])]
    controllers = []
    for node in nodes:
        model = SimpleNamespace(root=node, node=lambda gid, node=node: node)
        controllers.append(AdaptiveClockDriver(model, cfl_source=dycore.take_wrf_cfl,
            tick_den=100, map_factor_source=lambda gid: maximum_map_factor(geography=bundle.geography)))
    original_window = dycore.set_wrf_cfl_tile_window
    original_record = dycore.record_wrf_vertical_cfl
    fired = [0]
    if poison:
        def window(gid, spec):
            if gid == 2 and spec.tx == 0:
                spec = SimpleNamespace(i0=spec.i0, i1=spec.i1+1, ci0=spec.ci0,
                    j0=spec.j0, j1=spec.j1, cj0=spec.cj0)
            original_window(gid, spec)
        def record(state, live, ww):
            if state is run.tiles[0]:
                ww = ww.copy()
                ww[..., run.specs[0].i1-run.specs[0].ci0] = np.float32(1.e7)
                fired[0] += 1
            original_record(state, live, ww)
        dycore.set_wrf_cfl_tile_window = window
        dycore.record_wrf_vertical_cfl = record
    series = [[], []]
    cfls = [[], []]
    dycore.reset_wrf_cfl_recording()
    dycore.enable_wrf_cfl_recording()
    try:
        for index in range(nsteps):
            for arm, node in enumerate(nodes):
                controllers[arm](index, {node.cfg.grid_id: node.clock})
                controllers[arm].before_step(node.cfg.grid_id)
                live = node.cfg.run
                series[arm].append([live.dt, live.time_step_sound])
                if node.clock.lbc_reset_due():
                    node.clock.mark_force()
                node.clock.prepare_step()
                if arm == 0:
                    resident.elapsed_seconds = node.clock.elapsed_seconds
                    dycore.step(resident, live)
                else:
                    streamed.impose_clock(node.clock.elapsed_seconds)
                    streamed.stability.begin_sweep()
                    run.sweep(1, live_config=live,
                        physics_control=PhysicsStepControl.from_driver(node.state.physics))
                node.clock.advance()
                if arm == 0:
                    resident.elapsed_seconds = node.clock.elapsed_seconds
                else:
                    streamed.impose_clock(node.clock.elapsed_seconds)
                cfls[arm].append(dycore.take_wrf_cfl(node.cfg.grid_id))
        ref = canonical_state_digest(resident, clocks[0])
        got = streamed.canonical_digest(clocks[1])
        equal = ref == got and series[0] == series[1] and cfls[0] == cfls[1]
        print(json.dumps(dict(adaptive=True, poison=poison, poison_fires=fired[0],
            resident_dt=series[0], ranked_dt=series[1], resident_cfl=cfls[0], ranked_cfl=cfls[1],
            resident_digest=ref["sha256"], ranked_digest=got["sha256"],
            resident_fires=physics.carrier_scalars(resident)["call_counts"],
            ranked_fires=bundle.scalars["call_counts"])), flush=True)
        if poison:
            assert fired[0] > 0 and series[0] != series[1] and ref["sha256"] != got["sha256"], \
                "poisoned CFL window did not change adaptive dt and final digest"
        else:
            assert equal, "adaptive ranked dt, acoustic counts, CFL or canonical digest differ"
    finally:
        dycore.set_wrf_cfl_tile_window = original_window
        dycore.record_wrf_vertical_cfl = original_record
        run.close()
        dycore.reset_wrf_cfl_recording()


def benchmark(devices=None):
    """Median step of one resident card against the same domain split in two.

    ``devices`` is the two cards the split runs on (``--devices`` twice):
    default both slabs on card 0, which measures what splitting one card
    costs; two different cards measure the split's real speed.  The resident
    reference always runs on the first.
    """
    import cupy as cp
    import gc
    from time import perf_counter
    from statistics import median
    from gpuwm.core import dycore
    devices = [0, 0] if devices is None else [int(d) for d in devices]
    if len(devices) != 2:
        raise ValueError("bench splits the domain in two: pass --devices twice "
                         "(or not at all, for both slabs on card 0)")
    visible = cp.cuda.runtime.getDeviceCount()
    if any(d < 0 or d >= visible for d in devices):
        raise ValueError(f"bench --devices {devices}: visible card count is {visible}")
    cp.cuda.Device(devices[0]).use()
    cfg = config(400, 300, 50)
    state, bundle = fixture(cfg)
    resident_ms, ranked_ms, exchange_ms = [], [], []
    peak_pool_bytes = cp.get_default_memory_pool().used_bytes()
    for index in range(8):
        cp.cuda.runtime.deviceSynchronize()
        begin = perf_counter()
        dycore.step(state, cfg)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = 1000*(perf_counter()-begin)
        peak_pool_bytes = max(peak_pool_bytes, cp.get_default_memory_pool().used_bytes())
        if index >= 2:
            resident_ms.append(elapsed)
    # The initial store and scalar snapshot have not moved. Drop the whole
    # resident device state before allocating ranks, to stay below 6 GiB.
    del state
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    options = DeviceOptions(count=2, ids=tuple(devices))
    streamed = make_ranked(bundle, cfg, streaming.ranked_decision(cfg, options),
                            options, None, "threads")
    run = streamed.tiled_run
    try:
        for index in range(8):
            report = {"timing": True}
            run.sweep(1, report=report)
            for dev in dict.fromkeys(devices):
                with cp.cuda.Device(dev):
                    peak_pool_bytes = max(peak_pool_bytes,
                                          cp.get_default_memory_pool().used_bytes())
            if index >= 2:
                ranked_ms.append(1000*report["wall_seconds"])
                exchange_ms.append(1000*report["exchange_seconds"])
        if peak_pool_bytes > 6*1024**3:
            raise AssertionError(f"benchmark live CuPy allocations {peak_pool_bytes} exceed 6 GiB")
        print(json.dumps(dict(shape=[50, 300, 400], grid=[1, 2], devices=devices,
            mode="threads", samples=len(resident_ms), resident_ms=median(resident_ms),
            ranked_ms=median(ranked_ms), exchange_ms=median(exchange_ms),
            speedup=median(resident_ms) / median(ranked_ms),
            transport=[row["actual"] for row in run.transport_report()],
            peak_cupy_live_bytes=peak_pool_bytes)), flush=True)
    finally:
        run.close()


def transport_timing(devices=None):
    import cupy as cp
    from time import perf_counter
    from statistics import median
    cfg = config()
    if devices is None:
        devices = [0, 1] if cp.cuda.runtime.getDeviceCount() >= 2 else [0, 0]
    if len(devices) != 2:
        raise ValueError("transport timing needs two ranks, with --devices repeated twice")
    with cp.cuda.Device(devices[0]):
        state, bundle = fixture(cfg)
        before = multigpu.hash_host(bundle.store)
        options = DeviceOptions(count=2, ids=tuple(devices))
        streamed = make_ranked(bundle, cfg, streaming.ranked_decision(cfg, options),
                               options, None, "threads")
    run = streamed.tiled_run
    samples = []
    try:
        for sample in range(7):
            totals = [0., 0., 0.]
            for group in run.channel_phases:
                run.sync_all()
                begin = perf_counter()
                for _, ch in group:
                    rank = ch.seam.src_gpu
                    with cp.cuda.Device(ch.src_dev):
                        ch.pack(run.arrays[rank], run.copy_streams[rank].ptr)
                run.sync_all()
                packed = perf_counter()
                for _, ch in group:
                    with cp.cuda.Device(ch.src_dev):
                        ch.transfer(run.copy_streams[ch.seam.src_gpu].ptr)
                run.sync_all()
                for _, ch in group:
                    with cp.cuda.Device(ch.dst_dev):
                        ch.transfer_finish(run.unpack_streams[ch.seam.dst_gpu].ptr)
                run.sync_all()
                transferred = perf_counter()
                for _, ch in group:
                    rank = ch.seam.dst_gpu
                    with cp.cuda.Device(ch.dst_dev):
                        ch.unpack(run.arrays[rank], run.unpack_streams[rank].ptr)
                run.sync_all()
                end = perf_counter()
                totals[0] += packed-begin
                totals[1] += transferred-packed
                totals[2] += end-transferred
            if sample >= 2:
                samples.append(totals)
        run._ahead = True
        assert multigpu.hash_host(run.store) == before, "transport changed the initial carriers"
        print(json.dumps(dict(**run.devices_report(), samples=len(samples),
            pack_ms=1000*median(q[0] for q in samples),
            transfer_ms=1000*median(q[1] for q in samples),
            unpack_ms=1000*median(q[2] for q in samples))), flush=True)
        if len(set(devices)) < 2:
            print("same-device copies; no pair of cards exists", flush=True)
    finally:
        run.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="?", default=os.environ.get("RANKS_WHAT", "gate"), choices=("gate", "controls", "bench", "transport", "quick", "loop", "wrong-card"))
    parser.add_argument("--devices", type=int, action="append")
    parser.add_argument("--rung", choices=("lean", "mynn"))
    parser.add_argument("--step-mode", choices=("threads", "sequential"))
    parser.add_argument("--grid", choices=("1x1", "1x2", "2x1", "2x2"))
    args = parser.parse_args(argv)
    try:
        if args.what == "quick":
            integrate(config(rung=args.rung or "lean"), nsteps=2, devices=args.devices,
                      mode=args.step_mode or "sequential")
        elif args.what == "gate":
            grids = ((1, 1), (1, 2), (2, 1), (2, 2)) if args.grid is None else (tuple(map(int, args.grid.split("x"))),)
            failures = []
            for rung in (("lean", "mynn") if args.rung is None else (args.rung,)):
                for grid in grids:
                    for mode in (("sequential", "threads") if args.step_mode is None else (args.step_mode,)):
                        try:
                            integrate(config(rung=rung), grid=grid, mode=mode,
                                      devices=args.devices)
                        except Exception as exc:
                            failures.append(f"{rung} {grid} {mode}: {type(exc).__name__}: {exc}")
                            print("CASE FAIL " + failures[-1], flush=True)
            if failures:
                raise AssertionError("; ".join(failures))
        elif args.what == "wrong-card":
            return wrong_card_child(args.devices or [0, 0])
        elif args.what == "controls":
            control_cfg = _control_config()
            integrate(control_cfg, wind=25.0, change_live=False)
            for control in ("no_exchange", "crossed", "stale", "short", "mup"):
                integrate(control_cfg, control=control, nsteps=12)
            adaptive_loop(poison=True)
            # In a child process: a wrong-card dereference poisons its
            # process's CUDA context, so it never runs in this one.
            import cupy as cp
            cards = (list(args.devices) if args.devices else
                     list(range(min(2, cp.cuda.runtime.getDeviceCount()))))
            if len(cards) == 1:
                cards = cards * 2
            print(run_wrong_card_control(cards), flush=True)
        elif args.what == "loop":
            adaptive_loop()
        elif args.what == "transport":
            transport_timing(args.devices)
        else:
            benchmark(args.devices)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"RANKS {args.what}: FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    print(f"RANKS {args.what}: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
