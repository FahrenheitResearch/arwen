"""What CUDA graph replay is worth to a tiled sweep, as a function of window.

The hypothesis this exists to test, stated before it was run so it could be
refuted: a tiled sweep re-issues the step's kernel launches once per TILE, so
the launch overhead is multiplied by the tile count, and the smaller the
compute window the larger a fraction of the step that overhead is.  If that is
right, replaying the step from a captured graph should be worth little at a
672^2 window and a lot at 224^2 -- and the small windows are exactly what a 12
GB card is forced into, so it would be the fix for the small-card lane rather
than a general speedup.

Run it::

    python -m tilestream.bench_graph --rung "dry (control)" \\
        --domain 1344 --windows 224,288,384,480,544,672 --steps 8

HOW TO READ A NUMBER FROM THIS, AND WHEN NOT TO
-----------------------------------------------
A compute window below roughly 500 cells a side does NOT saturate a modern
card.  Every timing at 224^2 or 288^2 is therefore a measurement of launch and
submission cost with an idle GPU underneath, NOT of the solver's throughput,
and this project has already lost hours to quoting one as if it were the
latter.  That is not a caveat that weakens the result here -- it IS the result.
The graph is worth something precisely because the GPU is idle waiting for the
host, and the number to quote is the RATIO between the two tiled runs at the
same window, never the absolute ms/step of a small window on its own.

WHAT IT MEASURED
----------------
RTX 5070 (12 GB, native Linux, CuPy 14.1.1), verified idle, one buffer, no
gather or scatter, min of 3 x 20 steps.  Two independent runs agree to 0.1%
wherever they overlap.

Dry lane, 232 captured nodes::

    window   stream ms   graph ms   saved   speedup
       128       5.195      4.831   0.364    1.075x
       160       8.487      8.180   0.307    1.037x
       224      17.069     16.765   0.304    1.018x
       288      27.015     26.755   0.260    1.010x
       352      40.321     40.127   0.194    1.005x
       448      67.773     67.615   0.159    1.002x
       544     100.638    100.424   0.214    1.002x
       672     152.595    152.386   0.209    1.001x

Ship config ``full(real74) +KF``, 876 captured nodes, radiation 0 and cumulus
0 firings inside the window (production cadence), surface/PBL every step::

    window   stream ms   graph ms   saved   speedup
       128      29.883     25.977   3.907    1.150x
       160      46.350     43.118   3.232    1.075x
       224      85.246     83.579   1.667    1.020x

The shape is the one the hypothesis predicted -- the win is a shrinking
FRACTION as the window grows -- and the magnitude is not.  At the 224 window
where the tiling tax was measured at 5.37x, the graph removes 2% of a dry
tile-step and 2% of a physics one.  Graph capture is a real few-percent
saving on small windows and it is NOT the explanation for the tiling tax.
The absolute saving tracks the number of launches, not the window: 0.2-0.36
ms over 232 nodes is 1.3 us a launch, 3.2-3.9 ms over 876 nodes is 3.7-4.5
us a launch, both in the band PERF-FINDINGS measured for the mechanism, and
both far short of what a 5x would need.

AND THEN THE PLATFORM TURNED OUT TO MATTER MORE THAN THE WINDOW
---------------------------------------------------------------
The same benchmark on an RTX 5090 under WSL2 -- which is the machine
PERF-SURVEY.md profiled and the machine this is being built for -- gives a
completely different magnitude for the same code:

    ship config, 876 nodes (two runs)
                                window   stream ms      graph ms     speedup
                                   128  23.50/38.87   12.14/15.60  1.94x/2.49x
                                   160  28.24/26.05   19.24/18.42  1.47x/1.42x
                                   224  40.25/43.49   35.12/35.56  1.15x/1.22x
                                   288      -/67.90       -/61.55       -/1.10x

    dry, 232 nodes (two runs)      128    4.97/4.07   2.84/2.84  1.75x/1.44x
                                   224   12.83/16.55  7.59/7.66  1.69x/2.16x
                                   352   26.92/29.04 18.42/23.47 1.46x/1.24x
                                   544   46.30/47.36 45.02/45.50 1.03x/1.04x

Native Linux removes 3.7-4.5 us a launch; WSL2 removes 5.9-13.0.  WSL2's
kernel launch goes through a paravirtualised driver and costs several times
what a native one does, and it is the SAME 876 launches either way, so the
same graph is worth three times more there.  Read the two tables together
before quoting either: this is a 2-5% saving on a dedicated Linux card and a
15-95% one on the desktop.

Note also which column is stable.  Across repeats on a shared card the graph
times move by under 1% while the stream times move by 25%, because host
submission cost is what a graph removes and host contention is what makes
submission expensive.  A graph does not only make the step faster on this
platform, it makes it PREDICTABLE.

The second trap is physics cadence.  Radiation and cumulus fire on a cadence
of minutes; a short timed window at production cadence fires them ZERO times
and measures a step that is not the step the forecast runs.  Every row below
therefore prints how many times radiation and cumulus actually fired inside
the timed window, on BOTH sides of the comparison, and the two must agree --
if they do not, the two sides are not the same work and the ratio is
meaningless.
"""

from __future__ import annotations

import argparse
import json
import time


def _counts(state) -> dict:
    driver = getattr(state, "physics", None)
    if driver is None:
        return {"radiation": 0, "cumulus": 0, "sfclay": 0, "noah": 0,
                "ysu": 0}
    return dict(driver.call_counts)


def _delta(before: dict, after: dict) -> dict:
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in after}


def monolithic_step_ms(cfg, state, *, steps: int, warmup: int) -> tuple:
    """Resident, untiled, on the same config: the denominator of the tax."""
    import cupy as cp

    from gpuwm.core.dycore import step

    for _ in range(warmup):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    before = _counts(state)
    t0 = time.perf_counter()
    for _ in range(steps):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0
    return 1e3 * elapsed / steps, _delta(before, _counts(state))


def tiled_step_ms(store, cfg, tile, halo, *, steps: int, warmup: int,
                  use_graph, nbuffers: int, kwargs: dict) -> tuple:
    """ms per SWEEP for one tiling, plus what physics fired inside it."""
    import cupy as cp

    from tilestream import driver

    warm: dict = {}
    driver.run_tiled(store, cfg, tile, tile, halo=halo, nsteps=warmup,
                     nbuffers=nbuffers, use_graph=use_graph,
                     report=warm, **kwargs)
    cp.cuda.runtime.deviceSynchronize()

    before = dict(kwargs.get("scalars") or {})
    report: dict = {}
    t0 = time.perf_counter()
    driver.run_tiled(store, cfg, tile, tile, halo=halo, nsteps=steps,
                     nbuffers=nbuffers, use_graph=use_graph,
                     report=report, **kwargs)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0
    after = dict(kwargs.get("scalars") or {})
    fired = _delta(before.get("call_counts", {}) or {},
                   after.get("call_counts", {}) or {})
    return 1e3 * elapsed / steps, report, fired


def run_step_mode(rung: str, windows, steps: int, warmup: int, nz: int,
                  repeats: int, reuse: str = "run") -> dict:
    """The MECHANISM, isolated: one buffer, stepped both ways, no transport.

    ``run_sweep_mode`` measures what a whole tiled sweep costs, which is what
    the user pays -- but a sweep also gathers and scatters, and on a
    VRAM-resident store those copies are large enough to hide the thing under
    test.  This mode removes them entirely: build ONE state at the compute
    window, step it ``steps`` times through the ordinary stream path, then
    step it ``steps`` times by replaying a captured graph, and compare.  The
    difference is launch and submission cost and nothing else.

    That is also why this is the honest place to read the window dependence
    from.  A tiled sweep pays this difference once per TILE; how much of the
    sweep that recovers depends on what else the sweep is doing, which is the
    other mode's business.
    """
    import cupy as cp

    from tilestream import graphcap, harness, physics_inventory
    from tilestream.test_gate import PHYSICS_RUNGS
    from gpuwm.core.dycore import step

    dry = rung == "dry (control)"
    out = {"mode": "step", "rung": rung, "nz": nz, "steps": steps,
           "repeats": repeats, "reuse": reuse,
           "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
           "rows": []}
    print(f"# {out['device']}  {rung}  window^2 x {nz}  "
          f"one buffer, no gather/scatter")
    print(f"{'window':>7s} {'stream ms':>10s} {'graph ms':>10s} "
          f"{'saved ms':>9s} {'speedup':>8s} {'nodes':>6s} "
          f"{'us/launch':>10s} {'cap ms':>7s} {'pool MB':>8s}  "
          f"physics fired")
    for window in windows:
        cfg = harness.make_config(int(window), int(window), nz,
                                  **PHYSICS_RUNGS[rung])
        if dry:
            state = harness.make_state(cfg)
        else:
            state, _drv = physics_inventory.default_builder(
                cfg, harness.DEFAULT_SEED)
        harness.run_steps(state, cfg, max(1, warmup))

        stream_samples, graph_samples, fired = [], [], {}
        # ONE stepper per window, not one per repeat: each stepper owns a
        # private capture pool that must stay reserved for its graphs, so
        # three per window is three step-sized pools held at once and a 12 GB
        # card runs out at the third window.
        s = cp.cuda.Stream(non_blocking=True)
        stepper = graphcap.GraphStepper(
            cfg, mode="require", reuse=reuse,
            scalars_fn=physics_inventory.carrier_scalars,
            set_scalars_fn=physics_inventory.set_carrier_scalars)
        # Capture (and settle) OUTSIDE every timed window: a real run pays
        # the capture once per sweep and amortises it over the tiles, so
        # charging it to each step here would measure a workload nobody runs.
        # It is reported separately, in full.
        with s:
            stepper.run(state, s)
        s.synchronize()
        held = next(iter(stepper.graphs.values()))
        nodes, capture_ms = held.nodes, 1e3 * held.capture_seconds
        # What the graph path COSTS in memory: the private capture pool holds
        # one step's worth of temporaries per buffer, permanently, so that no
        # other allocation can be handed the addresses the graph baked in.
        # On a card that is already at its ceiling this is the number that
        # decides whether graphs are affordable at all.
        pool_mb = stepper._pool.total_bytes() / 1e6
        for _ in range(repeats):
            before = _counts(state)
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                step(state, cfg)
            cp.cuda.runtime.deviceSynchronize()
            stream_samples.append(1e3 * (time.perf_counter() - t0) / steps)
            fired_stream = _delta(before, _counts(state))

            before = _counts(state)
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            with s:
                for _ in range(steps):
                    stepper.run(state, s)
            s.synchronize()
            cp.cuda.runtime.deviceSynchronize()
            graph_samples.append(1e3 * (time.perf_counter() - t0) / steps)
            fired = (fired_stream, _delta(before, _counts(state)))

        row = {"window": int(window),
               "stream": min(stream_samples), "graph": min(graph_samples),
               "stream_samples": stream_samples, "graph_samples": graph_samples,
               "nodes": nodes, "capture_ms": capture_ms,
               "pool_mb": pool_mb, "fired": fired}
        row["saved"] = row["stream"] - row["graph"]
        row["speedup"] = row["stream"] / row["graph"]
        row["us_per_launch"] = 1e3 * row["saved"] / nodes if nodes else 0.0
        print(f"{row['window']:7d} {row['stream']:10.3f} {row['graph']:10.3f} "
              f"{row['saved']:9.3f} {row['speedup']:7.3f}x {nodes:6d} "
              f"{row['us_per_launch']:10.2f} {capture_ms:7.1f} "
              f"{pool_mb:8.0f}  {fired[0]} vs {fired[1]}")
        out["rows"].append(row)
        # The capture pool is only safe to release once every graph that
        # points into it is dead -- which is here, and nowhere else.  A
        # sweep of eight windows holds eight step-sized private pools
        # otherwise, and the 12 GB card runs out at the fourth.
        pool = stepper._pool
        del state, stepper, held, s
        if pool is not None:
            pool.free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()
    return out


def run(rung: str, domain: int, windows, steps: int, warmup: int,
        nz: int, nbuffers: int, host_store: bool, repeats: int) -> dict:
    import cupy as cp

    from tilestream import driver, gather, harness, physics_inventory
    from tilestream.test_gate import PHYSICS_RUNGS

    cfg = harness.make_config(domain, domain, nz, **PHYSICS_RUNGS[rung])
    halo = harness.halo_radius(cfg)
    dry = rung == "dry (control)"

    if dry:
        state = harness.make_state(cfg)
        harness.run_steps(state, cfg, 1)
        start = {k: cp.asnumpy(v) for k, v in harness.state_arrays(state).items()}
        extra: dict = {}
    else:
        state, _drv = physics_inventory.default_builder(cfg, harness.DEFAULT_SEED)
        harness.run_steps(state, cfg, 1)
        manifest = physics_inventory.carrier_inventory(state)
        start = {k: cp.asnumpy(v) for k, v in manifest.items()}
        extra = dict(
            inventory_fn=physics_inventory.carrier_inventory,
            nz=int(cfg.nz),
            tile_state_factory=driver.make_physics_tile_state)

    free, total = cp.cuda.runtime.memGetInfo()
    out = {
        "rung": rung, "domain": domain, "nz": nz, "halo": halo,
        "steps": steps, "warmup": warmup, "nbuffers": nbuffers,
        "host_store": host_store, "repeats": repeats,
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "free_gib_at_start": free / 2**30, "total_gib": total / 2**30,
        "rows": [],
    }

    mono_ms, mono_fired = monolithic_step_ms(
        cfg, state, steps=steps, warmup=warmup)
    out["monolithic_ms"] = mono_ms
    out["monolithic_fired"] = mono_fired
    print(f"# {out['device']}  {rung}  {domain}^2 x {nz}  halo {halo}")
    print(f"# monolithic (resident, untiled): {mono_ms:.3f} ms/step   "
          f"physics fired {mono_fired}")
    del state
    cp.get_default_memory_pool().free_all_blocks()

    print(f"{'window':>7s} {'tile':>6s} {'tiles':>6s} "
          f"{'stream ms':>10s} {'graph ms':>10s} {'speedup':>8s} "
          f"{'tax off':>8s} {'tax on':>7s} {'nodes':>6s} "
          f"{'cap/rep':>9s}  physics fired")
    for window in windows:
        tile = int(window) - 2 * halo
        if tile <= 0:
            continue
        row = {"window": int(window), "tile": tile}
        for label, use_graph in (("stream", False), ("graph", "require")):  # noqa: E501
            samples, report, fired = [], {}, {}
            for _ in range(repeats):
                if host_store:
                    store = {k: gather.pinned_copy(v) for k, v in start.items()}
                else:
                    store = {k: cp.asarray(v) for k, v in start.items()}
                kwargs = dict(extra)
                if not dry:
                    kwargs["scalars"] = {
                        "elapsed_seconds": 0.0, "call_counts": {},
                        "ysu_nan_guard_fires": 0, "microphysics_updates": 0}
                    # The domain clock the tiles must agree on; taken from
                    # the state the store was snapshotted from.
                    kwargs["scalars"] = dict(_START_SCALARS)
                ms, report, fired = tiled_step_ms(
                    store, cfg, tile, halo, steps=steps, warmup=warmup,
                    use_graph=use_graph, nbuffers=nbuffers, kwargs=kwargs)
                samples.append(ms)
                del store
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            row[label] = min(samples)
            row[f"{label}_samples"] = samples
            row[f"{label}_fired"] = fired
            row["tiles"] = report.get("tiles")
            row["compute"] = report.get("compute")
            if label == "graph":
                row["graph_info"] = report.get("graph")
        row["speedup"] = row["stream"] / row["graph"]
        row["tax_stream"] = row["stream"] / mono_ms
        row["tax_graph"] = row["graph"] / mono_ms
        info = row.get("graph_info") or {}
        nodes = "-".join(str(n) for n in (info.get("nodes") or [])) or "?"
        print(f"{row['window']:7d} {tile:6d} {row['tiles']:6d} "
              f"{row['stream']:10.3f} {row['graph']:10.3f} "
              f"{row['speedup']:7.3f}x {row['tax_stream']:7.2f}x "
              f"{row['tax_graph']:6.2f}x {nodes:>6s} "
              f"{info.get('captures', 0):4d}/{info.get('replays', 0):<4d} "
              f" {row['stream_fired']} vs {row['graph_fired']}")
        out["rows"].append(row)
    return out


_START_SCALARS: dict = {}


def main(argv=None) -> int:
    global _START_SCALARS

    import cupy as cp

    from tilestream import harness, physics_inventory
    from tilestream.test_gate import PHYSICS_RUNGS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="dry (control)")
    ap.add_argument("--domain", type=int, default=1344)
    ap.add_argument("--windows", default="224,288,384,480,544,672")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--nz", type=int, default=49)
    ap.add_argument("--nbuffers", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--host-store", action="store_true")
    ap.add_argument("--mode", default="sweep", choices=("sweep", "step"))
    ap.add_argument("--reuse", default="run", choices=("sweep", "run"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.rung not in PHYSICS_RUNGS:
        raise SystemExit(f"unknown rung {args.rung!r}")

    if args.rung != "dry (control)" and args.mode == "sweep":
        cfg = harness.make_config(args.domain, args.domain, args.nz,
                                  **PHYSICS_RUNGS[args.rung])
        probe, _drv = physics_inventory.default_builder(cfg, harness.DEFAULT_SEED)
        harness.run_steps(probe, cfg, 1)
        _START_SCALARS = physics_inventory.carrier_scalars(probe)
        del probe
        cp.get_default_memory_pool().free_all_blocks()

    windows = [int(w) for w in args.windows.split(",")]
    if args.mode == "step":
        out = run_step_mode(args.rung, windows, args.steps, args.warmup,
                            args.nz, args.repeats, reuse=args.reuse)
    else:
        out = run(args.rung, args.domain, windows,
                  args.steps, args.warmup, args.nz, args.nbuffers,
                  args.host_store, args.repeats)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"# wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
