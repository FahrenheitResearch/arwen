"""Attribute a streamed step's wall clock BEFORE building the overlap.

    python -m tilestream.overlap_attrib --config CFG [--tile N] [--nbuffers B]
        [--steps 12] [--warmup 2] [--json OUT.json]

The question this answers, per model step of the PRODUCTION streamed path
(:func:`gpuwm.core.streaming.prepared_domain_builder` -> ``attach`` ->
``StreamedDomain.__call__``, one sweep per step, exactly as
``execute_experiment`` drives it):

*   how much of the wall clock is COMPUTE (``dycore.step`` on tile buffers),
*   how much is TRANSFER, split H2D carrier gather / D2H scatter / geography
    gather / ring save / ring patch,
*   how much of the transfer is EXPOSED -- device transfer time during which
    no compute interval is running -- versus hidden under compute,
*   and how much is the three named host-side candidates: the ``tile_hook``
    bind (the eager ``attach_lateral_boundaries`` re-upload),
    ``_advance_clock``'s per-buffer scalar readback, and
    ``set_carrier_scalars``.

MEASURED, not inferred: every device-side category is bracketed by CUDA
events recorded on the operation's own stream and reduced against a per-step
origin event; host-side categories are ``perf_counter`` around the exact
call.  The instrument changes no arithmetic and no issue order -- wrappers
call straight through -- and its own overhead is stated by running the same
window uninstrumented first (``--control``).

The exposure arithmetic is interval arithmetic, not subtraction of sums:
``exposed = |union(transfer) - union(compute)|``, so three-deep transfer
overlap is not triple-counted and transfer under compute is not blamed.

Nothing here is case-specific; the config is whatever gpuwm TOML the caller
hands in.  ``asyncEngineCount`` and the device identity are printed with the
result because the overlap ceiling is a property of THIS card: one copy
engine still overlaps one copy with compute, it just cannot overlap H2D with
D2H as well.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


# --------------------------------------------------------------------------
# the recorder
# --------------------------------------------------------------------------

class _Recorder:
    """Per-step event and host-timer collection.  Off until armed."""

    def __init__(self):
        self.active = False
        self.dev: list = []      # (category, start_event, end_event)
        self.host: list = []     # (category, seconds)
        self.origin = None

    def begin_step(self):
        import cupy as cp
        self.dev = []
        self.host = []
        self.origin = cp.cuda.Event()
        self.origin.record()      # current (default) stream
        self.active = True

    def end_step(self):
        """Harvest after the sweep's own deviceSynchronize.  Returns spans."""
        import cupy as cp
        self.active = False
        spans: dict[str, list] = {}
        for cat, e0, e1 in self.dev:
            t0 = cp.cuda.get_elapsed_time(self.origin, e0)
            t1 = cp.cuda.get_elapsed_time(self.origin, e1)
            spans.setdefault(cat, []).append((t0, t1))
        host: dict[str, float] = {}
        for cat, dt in self.host:
            host[cat] = host.get(cat, 0.0) + dt
        return spans, host


REC = _Recorder()


def _timed_dev(cat, fn, *, stream_arg: int | None = None):
    """Wrap ``fn`` with CUDA events on its stream.  Transparent when off."""
    import cupy as cp

    def wrapped(*args, **kwargs):
        if not REC.active:
            return fn(*args, **kwargs)
        stream = (args[stream_arg] if stream_arg is not None
                  and len(args) > stream_arg else None)
        if stream is None:
            stream = cp.cuda.get_current_stream()
        e0 = cp.cuda.Event()
        e0.record(stream)
        out = fn(*args, **kwargs)
        e1 = cp.cuda.Event()
        e1.record(stream)
        REC.dev.append((cat, e0, e1))
        return out

    wrapped.__wrapped__ = fn
    return wrapped


def _timed_host(cat, fn):
    def wrapped(*args, **kwargs):
        if not REC.active:
            return fn(*args, **kwargs)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        REC.host.append((cat, time.perf_counter() - t0))
        return out

    wrapped.__wrapped__ = fn
    return wrapped


def install() -> None:
    """Patch the production seams.  Idempotent; call BEFORE attach()."""
    import gpuwm.core.dycore as dycore
    import gpuwm.core.streaming as streaming
    from tilestream import driver, gather, physics_inventory, rings

    if getattr(install, "_done", False):
        return
    install._done = True

    dycore.step = _timed_dev("compute", dycore.step)

    real_gather = gather.gather_tile

    def gather_tile(home, tile, tspec, stream, **kwargs):
        if not REC.active:
            return real_gather(home, tile, tspec, stream, **kwargs)
        geo = kwargs.get("inventory_fn") is driver.geography_inventory
        cat = "geo_h2d" if geo else "h2d"
        import cupy as cp
        e0 = cp.cuda.Event()
        e0.record(stream)
        out = real_gather(home, tile, tspec, stream, **kwargs)
        e1 = cp.cuda.Event()
        e1.record(stream)
        REC.dev.append((cat, e0, e1))
        return out

    gather.gather_tile = gather_tile
    gather.scatter_tile = _timed_dev("d2h", gather.scatter_tile,
                                     stream_arg=3)
    rings.RingArena.save = _timed_dev("ring_save", rings.RingArena.save,
                                      stream_arg=3)
    rings.RingArena.patch = _timed_dev("ring_patch", rings.RingArena.patch,
                                       stream_arg=3)
    driver._advance_clock = _timed_host("advance_clock",
                                        driver._advance_clock)
    physics_inventory.set_carrier_scalars = _timed_host(
        "set_scalars", physics_inventory.set_carrier_scalars)

    real_make_hook = streaming.make_tile_hook

    def make_tile_hook(per_tile, **kwargs):
        hook = real_make_hook(per_tile, **kwargs)
        return _timed_host("tile_hook", hook)

    streaming.make_tile_hook = make_tile_hook


# --------------------------------------------------------------------------
# interval arithmetic
# --------------------------------------------------------------------------

def _union(intervals):
    """Merged intervals, sorted.  Input: [(t0, t1), ...] in ms."""
    out = []
    for t0, t1 in sorted(intervals):
        if out and t0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], t1))
        else:
            out.append((t0, t1))
    return out


def _length(intervals) -> float:
    return sum(t1 - t0 for t0, t1 in intervals)


def _subtract(a, b):
    """Intervals of ``a`` not covered by ``b``.  Both merged and sorted."""
    out = []
    j = 0
    for t0, t1 in a:
        cur = t0
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < t1:
            if b[k][0] > cur:
                out.append((cur, b[k][0]))
            cur = max(cur, b[k][1])
            k += 1
        if cur < t1:
            out.append((cur, t1))
    return out


TRANSFER_CATS = ("h2d", "d2h", "geo_h2d", "ring_save", "ring_patch")


def attribute_step(spans: dict, host: dict, wall_s: float) -> dict:
    compute = _union(spans.get("compute", []))
    per_cat = {c: _length(_union(spans.get(c, []))) for c in TRANSFER_CATS}
    transfer = _union([iv for c in TRANSFER_CATS for iv in spans.get(c, [])])
    exposed = _subtract(transfer, compute)
    row = {
        "wall_ms": wall_s * 1e3,
        "compute_busy_ms": _length(compute),
        "transfer_busy_ms": _length(transfer),
        "transfer_exposed_ms": _length(exposed),
        "transfer_hidden_ms": _length(transfer) - _length(exposed),
        "per_category_ms": per_cat,
        # How much of each category runs while NO compute is running --
        # which transfers the overlap build actually has to move.
        "exposed_by_category_ms": {
            c: _length(_subtract(_union(spans.get(c, [])), compute))
            for c in TRANSFER_CATS},
        # Exposure before the first compute interval starts (pipeline FILL)
        # and after the last one ends (DRAIN), against mid-sweep stalls.
        "exposed_fill_ms": _length(
            [iv for iv in exposed if compute and iv[1] <= compute[0][0]]),
        "exposed_drain_ms": _length(
            [iv for iv in exposed if compute and iv[0] >= compute[-1][1]]),
        "host_ms": {k: v * 1e3 for k, v in host.items()},
    }
    accounted = (row["compute_busy_ms"] + row["transfer_exposed_ms"]
                 + sum(row["host_ms"].values()))
    row["unattributed_ms"] = row["wall_ms"] - accounted
    return row


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def _device_identity() -> dict:
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    return {
        "name": props["name"].decode(),
        "asyncEngineCount": int(props["asyncEngineCount"]),
        "totalGlobalMem_GiB": props["totalGlobalMem"] / 2 ** 30,
        "cupy": cp.__version__,
        "driver": cp.cuda.runtime.driverGetVersion(),
        "runtime": cp.cuda.runtime.runtimeGetVersion(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tile", type=int, default=None,
                    help="pin the tiling; default lets the planner choose, "
                         "which is what a production run does")
    ap.add_argument("--nbuffers", type=int, default=None)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--control", action="store_true",
                    help="run the same window UNinstrumented and report "
                         "wall only, to state the instrument's overhead")
    ap.add_argument("--overlap", choices=("on", "off", "unchained"),
                    default="on",
                    help="TiledRun's overlap mode.  The on/off pair on the "
                         "same window IS the overlap build's perf receipt; "
                         "'unchained' exists for the gate, not for numbers")
    ap.add_argument("--prepare-rows", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=1,
                    help="measure the drained window this many times in one "
                         "process, one prepare paying for all of them; a "
                         "WINDOW line is printed per repeat.  Repeats after "
                         "the first always run uninstrumented, so pair this "
                         "with --control when the windows must be identical")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import warnings

    import cupy as cp

    from gpuwm.core import streaming
    from tilestream import driver, realcase
    from tilestream.run_realcase import _run_kwargs

    ident = _device_identity()
    print(f"DEVICE {json.dumps(ident)}", flush=True)

    prep, exp, _peak = realcase.prepare_slabbed(
        args.config, rows_per_slab=args.prepare_rows)
    cfg = prep.cfg

    if not args.control:
        install()

    # The stage_forecast wiring: stores in pinned host RAM, prepared state
    # released, tile buffers the only resident footprint.  The settling
    # step is what a past-the-ceiling domain cannot take; build_stores'
    # own docstring carries the measurement, and run_tiled's inventory
    # match guards the skip.
    case = realcase.build_stores(prep, exp, settle=False)
    realcase.release_prepared(prep)
    prep = None

    # The tiling the PRODUCTION planner would choose on this card, asked
    # with the card in the state a streamed run would find it (store on
    # host, device empty).  --tile pins it instead, exactly like [tiles].
    options = streaming.StreamingOptions(
        mode="on", tile_nx=args.tile, tile_ny=args.tile,
        nbuffers=args.nbuffers)
    decision = streaming.decide(cfg, options)
    if not decision.stream:
        print(f"REFUSED: {decision.explain()}")
        sys.exit(2)
    print(f"DECISION {decision.explain()}", flush=True)

    specs, halo = realcase.plan(cfg, int(decision.tile_nx),
                                int(decision.tile_ny))
    print(f"PLAN {len(specs)} tiles of {specs[0].cny}x{specs[0].cnx}, "
          f"halo {halo}, nbuffers {decision.nbuffers}", flush=True)

    binder = realcase.tile_boundary_binder(case.boundaries, specs, cfg,
                                           verbose=print)
    kwargs = _run_kwargs(cfg, exp, case.geo_store, binder.per_tile[0])
    kwargs["scalars"] = case.scalars
    kwargs["tile_hook"] = (binder if args.control
                           else _timed_host("tile_hook", binder))

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run = driver.TiledRun(
            case.store, cfg, int(decision.tile_nx), int(decision.tile_ny),
            halo=halo, nbuffers=int(decision.nbuffers), periodic=False,
            write_mode="ring", overlap=args.overlap, **kwargs)
    print(f"SETUP TiledRun {time.perf_counter() - t0:.1f}s "
          f"(factory {run.factory_seconds:.1f}s)  overlap={run.overlap} "
          f"deferred_seam={run.deferred_seam}", flush=True)

    # THE WINDOW CLOCK.  Under the deferred seam a sweep returns with its
    # tail still in flight, so a per-step perf_counter is ISSUE time, not
    # wall time -- that is the lever, not an accounting error.  The honest
    # end-to-end number is the drained window: drain, clock, run every
    # timed step, drain, clock.  Reported beside the per-step rows in both
    # modes so on/off pairs compare like for like.
    rows = []
    window_t0 = None
    for i in range(args.warmup + args.steps):
        timed = i >= args.warmup
        if timed and window_t0 is None:
            run.drain()
            window_t0 = time.perf_counter()
        if timed and not args.control:
            REC.begin_step()
        report: dict = {}
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            run.sweep(1, report=report)
        wall = time.perf_counter() - t0
        if not timed:
            continue
        if args.control:
            rows.append({"wall_ms": wall * 1e3})
            print(f"STEP {i} wall {wall*1e3:8.1f} ms  (control)", flush=True)
            continue
        # The recorder reads its events back, which needs them complete;
        # the sweep no longer barriers for us.  The per-step drain here
        # re-serialises the SEAM, so instrumented rows measure the
        # within-step overlap only; the drained window (and the control
        # mode, which never drains per step) carries the cross-step lever.
        run.drain()
        spans, host = REC.end_step()
        row = attribute_step(spans, host, wall)
        row["report"] = {k: report.get(k) for k in
                         ("gathered_bytes", "scattered_bytes",
                          "ring_saved_bytes", "ring_patched_bytes",
                          "geography_bytes", "tiles", "nbuffers")}
        rows.append(row)
        print(f"STEP {i} wall {row['wall_ms']:8.1f} ms  "
              f"compute {row['compute_busy_ms']:8.1f}  "
              f"xfer {row['transfer_busy_ms']:7.1f} "
              f"(exposed {row['transfer_exposed_ms']:7.1f})  "
              f"host {sum(row['host_ms'].values()):6.1f}  "
              f"unattr {row['unattributed_ms']:6.1f}", flush=True)

    run.drain()
    window_s = (time.perf_counter() - window_t0) if window_t0 else 0.0
    window = {"window_s": window_s, "steps": args.steps,
              "per_step_ms": window_s * 1e3 / max(1, args.steps),
              "overlap": run.overlap,
              "deferred_seam": bool(run.deferred_seam), "rep": 1}
    print("WINDOW " + json.dumps(window), flush=True)
    windows = [window]
    for rep in range(2, max(1, args.repeat) + 1):
        run.drain()
        rep_t0 = time.perf_counter()
        for _ in range(args.steps):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                run.sweep(1)
        run.drain()
        rep_s = time.perf_counter() - rep_t0
        window = {"window_s": rep_s, "steps": args.steps,
                  "per_step_ms": rep_s * 1e3 / max(1, args.steps),
                  "overlap": run.overlap,
                  "deferred_seam": bool(run.deferred_seam), "rep": rep}
        print("WINDOW " + json.dumps(window), flush=True)
        windows.append(window)

    # The tile_hook bind, both ways, on a real tile buffer with the run's
    # own windowed tables.  The harness binder above is the LAZY swap
    # (attach_streaming_lateral_boundaries); the production attach() path
    # installs the EAGER gpuwm.ingest.lateral_bc.attach_lateral_boundaries,
    # which re-uploads every interval on every buffer-tile change.  Measured
    # here side by side so the arm's own numbers can be projected onto the
    # production hook without running a second full arm.
    bind = None
    if not args.control:
        from gpuwm.ingest.lateral_bc import attach_lateral_boundaries
        tile = run.tiles[0]
        stream = cp.cuda.get_current_stream()
        nrep = min(len(specs), 6)
        lazy = []
        for itile in range(nrep):
            t0 = time.perf_counter()
            binder(tile, specs[itile], itile, stream)
            cp.cuda.runtime.deviceSynchronize()
            lazy.append(time.perf_counter() - t0)
        eager = []
        for itile in range(nrep):
            t0 = time.perf_counter()
            attach_lateral_boundaries(tile, binder.per_tile[itile])
            cp.cuda.runtime.deviceSynchronize()
            eager.append(time.perf_counter() - t0)
        bind = {"lazy_bind_ms": [x * 1e3 for x in lazy],
                "eager_bind_ms": [x * 1e3 for x in eager],
                "binds_per_sweep": (len(specs)
                                    if len(specs) > run.nbuffers else 0)}
        print("BIND " + json.dumps(bind), flush=True)

    result = {
        "device": ident,
        "config": args.config,
        "bind": bind,
        "nx": int(cfg.nx), "ny": int(cfg.ny), "nz": int(cfg.nz),
        "dx_m": float(cfg.dx), "dt_s": float(cfg.dt),
        "decision": decision.explain(),
        "steps": args.steps, "warmup": args.warmup,
        "control": bool(args.control),
        "overlap": run.overlap,
        "deferred_seam": bool(run.deferred_seam),
        "window": windows[0],
        "windows": windows,
        "rows": rows,
    }
    if not args.control and rows:
        n = len(rows)
        mean = {k: sum(r[k] for r in rows) / n for k in
                ("wall_ms", "compute_busy_ms", "transfer_busy_ms",
                 "transfer_exposed_ms", "transfer_hidden_ms",
                 "exposed_fill_ms", "exposed_drain_ms",
                 "unattributed_ms")}
        mean["per_category_ms"] = {
            c: sum(r["per_category_ms"][c] for r in rows) / n
            for c in TRANSFER_CATS}
        mean["exposed_by_category_ms"] = {
            c: sum(r["exposed_by_category_ms"][c] for r in rows) / n
            for c in TRANSFER_CATS}
        hostcats = sorted({k for r in rows for k in r["host_ms"]})
        mean["host_ms"] = {
            c: sum(r["host_ms"].get(c, 0.0) for r in rows) / n
            for c in hostcats}
        result["mean"] = mean
        print("MEAN " + json.dumps(mean), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1)
    print("RESULT " + json.dumps({k: v for k, v in result.items()
                                  if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
