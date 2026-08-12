"""Scaling measurement for the ArWen two-GPU decomposition.

Every timed configuration also produces a SHA-256 over the assembled domain,
because a decomposition bug that skips work looks exactly like a speedup.
Where a monolithic reference exists (the domain fits one card) the digest is
compared against it; where it does not (the whole point of the weak-scaling
row) the digest is cross-checked between pipelines and a no-exchange control
must differ.

One process per configuration.  CuPy's pool does not hand memory back cleanly
between a 22 GiB monolithic run and a 22 GiB sub-domain in the same process,
and a ceiling measured after a failed allocation measures the allocator.  Each
``cmd`` below is therefore one invocation and one process; a driver script
loops over configurations by loop over ``python -m tilestream.scaling ...``,
never by calling ``main`` twice.

WHY THE INITIAL STATE IS CACHED ON DISK, when run_bench reseeds
---------------------------------------------------------------
:mod:`tilestream.run_bench` runs its stages in separate processes for the same
allocator reason, and gets an identical starting state in each of them by
DETERMINISTIC RESEED -- ``bench.seed_host_store`` reconstructs the same bytes
from ``--n``/``--seed``, so nothing has to be written down.  That is the better
mechanism where it applies and it does not apply here, twice over:

* the sizes this module exists to measure are ones ``harness.make_state``
  cannot build.  It draws every perturbation on the host and uploads it, which
  overshoots the pool by ~30% and OOMs on a domain the card can perfectly well
  STEP.  :func:`frugal_state` draws on the device instead and is used for the
  timing-only rows -- but a frugal state is not a reseedable reference, so the
  only way the 1-GPU and 2-GPU runs at that size can be shown to have started
  from the same state is that they loaded the same file;
* :func:`build_init_x2` builds a ``2nx``-wide domain by concatenating two
  independently-seeded halves.  There is no seed that produces it.

So the discipline here is: build the initial condition ONCE, persist it, and
have every configuration load THAT.  Nothing times the build.

This module is the multi-GPU sibling of :mod:`tilestream.run_bench`, and is
kept separate from it for the same reason :mod:`tilestream.multigpu` is kept
separate from :mod:`tilestream.driver`: the single-GPU out-of-core lane must
not acquire an import of the multi-GPU one.
"""
import argparse
import gc
import json
import os
import time

import numpy as np

from tilestream import driver, harness, multigpu

#: Where the persisted initial conditions live.  On the box this was measured
#: on it was a cache directory on that node's own scratch; it is an env var
#: because the cache is
#: tens of GiB per size and belongs on whichever filesystem the box has room
#: on, not next to the checkout.
CACHE = os.environ.get("ARWEN_SCALE_CACHE",
                       os.path.join(os.path.expanduser("~"), "scale", "cache"))


# ---------------------------------------------------------------------------
# initial-condition cache
# ---------------------------------------------------------------------------

def cache_dir(nx: int, ny: int, seed: int) -> str:
    return os.path.join(CACHE, f"{nx}x{ny}x{harness.DEFAULT_NZ}_s{seed}")


def build_init(nx: int, ny: int, seed: int, device: int = 0) -> dict:
    """Seed a monolithic state, save its persisted inventory to disk.

    Cached because every configuration at this size must start from the SAME
    initial condition for its digest to mean anything, and rebuilding it costs
    ~33 s of host-side normal draws at 1728^2.
    """
    import cupy as cp

    d = cache_dir(nx, ny, seed)
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "_names.json")):
        return {"cached": True, "dir": d}
    os.makedirs(d, exist_ok=True)
    cfg = harness.make_config(nx, ny)
    with cp.cuda.Device(device):
        t0 = time.perf_counter()
        state = harness.make_state(cfg, seed=seed)
        cp.cuda.runtime.deviceSynchronize()
        names = []
        for name, arr in harness.state_arrays(state).items():
            np.save(os.path.join(d, f"{name}.npy"), cp.asnumpy(arr))
            names.append(name)
        del state
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    with open(os.path.join(d, "_names.json"), "w") as fh:
        json.dump(names, fh)
    return {"cached": False, "dir": d, "build_s": time.perf_counter() - t0,
            "names": names}


def build_init_frugal(nx: int, ny: int, seed: int, device: int = 0) -> dict:
    """Save an initial condition for a domain ``make_state`` cannot build.

    Needed for the TRUE single-card ceiling: 1776^2 steps fine, but
    ``harness.make_state`` draws every perturbation on the host and uploads it,
    which overshoots the pool and OOMs before it can return.  The frugal
    builder draws on the device instead.  This is not the seeded reference
    state and its digest is not comparable with any other size's -- what
    matters is that the 1-GPU and 2-GPU runs at this size start from THE SAME
    state, which is exactly what caching it guarantees.
    """
    import cupy as cp

    d = cache_dir(nx, ny, seed)
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "_names.json")):
        return {"cached": True, "dir": d}
    os.makedirs(d, exist_ok=True)
    cfg = harness.make_config(nx, ny)
    with cp.cuda.Device(device):
        state = frugal_state(cfg, seed)
        names = []
        for name, arr in harness.state_arrays(state).items():
            np.save(os.path.join(d, f"{name}.npy"), cp.asnumpy(arr))
            names.append(name)
        del state
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    with open(os.path.join(d, "_names.json"), "w") as fh:
        json.dump(names, fh)
    return {"cached": False, "dir": d, "names": names, "builder": "frugal"}


def load_init(nx: int, ny: int, seed: int, mmap: bool = True) -> dict:
    d = cache_dir(nx, ny, seed)
    with open(os.path.join(d, "_names.json")) as fh:
        names = json.load(fh)
    return {n: np.load(os.path.join(d, f"{n}.npy"),
                       mmap_mode="r" if mmap else None) for n in names}


def build_init_x2(nx: int, ny: int, seed_a: int, seed_b: int,
                  out_seed: int) -> dict:
    """A 2nx-wide initial condition: two independently-seeded halves in x.

    Legal because the persisted diagnostics (p, al, alt) are COLUMN-LOCAL, so
    concatenating two discretely-balanced states in x leaves a state that is
    still discretely balanced everywhere, and periodic in x with period 2nx.
    Two DIFFERENT seeds, so the two halves are not copies of each other and a
    1x2 split does not accidentally give both GPUs identical work.

    ``u`` carries ``nx+1`` faces whose last is the periodic alias of face 0;
    the doubled array carries ``2nx+1`` and its alias slot is half A's face 0.
    """
    d = cache_dir(2 * nx, ny, out_seed)
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "_names.json")):
        return {"cached": True, "dir": d}
    os.makedirs(d, exist_ok=True)
    a = load_init(nx, ny, seed_a)
    b = load_init(nx, ny, seed_b)
    names = []
    for name in a:
        xa, xb = a[name], b[name]
        if xa.ndim < 2:
            np.save(os.path.join(d, f"{name}.npy"), np.asarray(xa))
            names.append(name)
            continue
        w = xa.shape[-1]
        if w == nx:
            out = np.empty(xa.shape[:-1] + (2 * nx,), dtype=xa.dtype)
            out[..., :nx] = xa
            out[..., nx:] = xb
        elif w == nx + 1:
            out = np.empty(xa.shape[:-1] + (2 * nx + 1,), dtype=xa.dtype)
            out[..., :nx] = xa[..., :nx]
            out[..., nx:2 * nx] = xb[..., :nx]
            out[..., 2 * nx] = xa[..., 0]           # alias of face 0
        else:
            raise RuntimeError(f"{name}: last dim {w} is neither {nx} "
                               f"nor {nx + 1}")
        np.save(os.path.join(d, f"{name}.npy"), out)
        names.append(name)
        del out
        gc.collect()
    with open(os.path.join(d, "_names.json"), "w") as fh:
        json.dump(names, fh)
    return {"cached": False, "dir": d, "names": names}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def frugal_state(cfg, seed: int):
    """A non-trivial state built without ``make_state``'s host staging spike.

    ``harness.make_state`` draws every perturbation on the host and uploads it,
    which pushes CuPy's pool ~30% above the steady footprint and turns a
    domain the card CAN step into an OOM.  MEASURED: at 1728^2 this builder
    and ``make_state`` give 700.96 vs 701.68 ms/step -- 0.1% -- so the step
    cost does not depend on which one produced the numbers.  Used only where
    the seeded reference state is not needed (timing-only rows).
    """
    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics

    state = driver.make_tile_state(cfg)
    rng = cp.random.default_rng(seed)
    for name, amp in (("u", 1.0), ("v", 1.0), ("w", 0.1),
                      ("thp", 1.0), ("php", 10.0), ("mup", 1.0)):
        arr = getattr(state, name, None)
        if arr is None:
            continue
        arr[...] = (amp * rng.standard_normal(arr.shape,
                                              dtype=cp.float32)).astype(
            arr.dtype)
    state.u[..., -1] = state.u[..., 0]
    state.v[:, -1, :] = state.v[:, 0, :]
    state.w[0] = 0.0
    state.w[-1] = 0.0
    update_diagnostics(state)
    cp.cuda.runtime.deviceSynchronize()
    return state


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def stats(times):
    m = median(times)
    return {"ms": m, "spread": (max(times) - min(times)) / m if m else 0.0,
            "times": [round(t, 4) for t in times]}


def freemem():
    import cupy as cp

    out = []
    for d in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(d):
            f, t = cp.cuda.runtime.memGetInfo()
            out.append(round((t - f) / 2**20, 1))
    return out


# ---------------------------------------------------------------------------
# monolithic
# ---------------------------------------------------------------------------

def cmd_mono(a) -> dict:
    import cupy as cp

    cfg = harness.make_config(a.nx, a.ny)
    host0 = load_init(a.nx, a.ny, a.seed)
    out = {"kind": "mono", "nx": a.nx, "ny": a.ny, "nz": harness.DEFAULT_NZ,
           "device": a.device, "steps": a.steps, "reps": a.reps}
    with cp.cuda.Device(a.device):
        state = driver.make_tile_state(cfg)
        cp.cuda.runtime.deviceSynchronize()
        inv = harness.state_arrays(state)
        if sorted(inv) != sorted(host0):
            raise RuntimeError(f"inventory mismatch {sorted(inv)} vs "
                               f"{sorted(host0)}")
        for name, arr in inv.items():
            arr[...] = cp.asarray(np.ascontiguousarray(host0[name]))
        cp.cuda.runtime.deviceSynchronize()
        out["vram_after_load_MiB"] = freemem()

        harness.run_steps(state, cfg, a.steps)
        cp.cuda.runtime.deviceSynchronize()
        out["hash"] = harness.hash_state(state)

        harness.run_steps(state, cfg, 2)            # warm
        cp.cuda.runtime.deviceSynchronize()
        ts = []
        for _ in range(a.reps):
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            harness.run_steps(state, cfg, 1)
            cp.cuda.runtime.deviceSynchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        out.update(stats(ts))
        lt = []
        for _ in range(5):
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            harness.run_steps(state, cfg, 1, sync=False)
            lt.append((time.perf_counter() - t0) * 1e3)
            cp.cuda.runtime.deviceSynchronize()
        out["host_launch_ms"] = median(lt)
        out["vram_peak_MiB"] = freemem()
        out["persisted_MiB"] = sum(v.nbytes for v in inv.values()) / 2**20
    out["cells"] = a.nx * a.ny * harness.DEFAULT_NZ
    out["ns_per_cell_step"] = out["ms"] * 1e6 / out["cells"]
    return out


def cmd_solo(a) -> dict:
    """Time ONE sub-domain, alone, on one card.

    Two shapes: the interior alone (``sub_nx`` wide) and the interior plus its
    two 16-cell halos (``sub_nx + 2h``), which is what a GPU in the
    decomposition actually steps.  The ratio IS the redundant-halo-compute
    cost, measured rather than predicted.
    """
    import cupy as cp

    halo = harness.halo_radius(harness.make_config(a.nx, a.ny))
    gy, gx = (1, a.ngpu) if a.grid is None else a.grid
    sub_nx = a.nx // gx
    sub_ny = a.ny // gy
    out = {"kind": "solo", "nx": a.nx, "ny": a.ny, "grid": [gy, gx],
           "halo": halo, "sub_nx": sub_nx, "sub_ny": sub_ny}
    for tag, w, h in (("interior", sub_nx, sub_ny),
                      ("with_halo", sub_nx + (2 * halo if gx > 1 else 0),
                       sub_ny + (2 * halo if gy > 1 else 0))):
        cfg = harness.make_config(w, h)
        with cp.cuda.Device(a.device):
            state = frugal_state(cfg, a.seed) if a.frugal \
                else harness.make_state(cfg, seed=a.seed)
            harness.run_steps(state, cfg, 2)
            cp.cuda.runtime.deviceSynchronize()
            ts = []
            for _ in range(a.reps):
                cp.cuda.runtime.deviceSynchronize()
                t0 = time.perf_counter()
                harness.run_steps(state, cfg, 1)
                cp.cuda.runtime.deviceSynchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            out[tag] = stats(ts)
            out[tag]["shape"] = [w, h]
            out[tag]["ns_per_cell_step"] = \
                out[tag]["ms"] * 1e6 / (w * h * harness.DEFAULT_NZ)
            del state
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
    out["halo_recompute_measured"] = out["with_halo"]["ms"] / \
        out["interior"]["ms"]
    out["halo_recompute_predicted"] = \
        ((sub_nx + 2 * halo) if gx > 1 else sub_nx) * \
        ((sub_ny + 2 * halo) if gy > 1 else sub_ny) / (sub_nx * sub_ny)
    return out


# ---------------------------------------------------------------------------
# multi-GPU
# ---------------------------------------------------------------------------

def make_domain(a, cfg):
    return multigpu.MultiGPUDomain(
        cfg, ngpu=a.ngpu,
        grid=None if a.grid is None else tuple(a.grid),
        devices=None if a.devices is None else a.devices,
        transport=a.transport,
        state_factory=driver.make_tile_state)


def cmd_mg(a) -> dict:
    import cupy as cp

    cfg = harness.make_config(a.nx, a.ny)
    out = {"kind": "mg", "nx": a.nx, "ny": a.ny, "nz": harness.DEFAULT_NZ,
           "steps": a.steps, "reps": a.reps, "step_mode": a.step_mode,
           "exchange_mode": a.exchange_mode, "exchange": not a.no_exchange,
           "transport": a.transport}
    dom = make_domain(a, cfg)
    out["grid"] = list(dom.grid)
    out["devices"] = list(dom.devices)
    out["halo"] = dom.halo
    out["sub_shape"] = [[s.cny, s.cnx] for s in dom.specs]
    out["seam_MiB"] = dom.seam_bytes / 2**20
    out["vram_MiB"] = [b / 2**20 for b in dom.vram_bytes()]
    if not a.skip_load:
        host0 = load_init(a.nx, a.ny, a.seed)
        dom.load_from_host(host0)
        del host0
        gc.collect()
    out["vram_after_load_MiB"] = freemem()

    kw = dict(step_mode=a.step_mode, exchange_mode=a.exchange_mode,
              exchange=not a.no_exchange)
    dom.run(a.steps, **kw)
    out["hash"] = dom.hash()

    dom.run(2, **kw)                                # warm
    ts = []
    for _ in range(a.reps):
        dom.sync_all()
        t0 = time.perf_counter()
        dom.run(1, **kw)
        ts.append((time.perf_counter() - t0) * 1e3)
    out.update(stats(ts))
    out["vram_peak_MiB"] = freemem()
    out["cells"] = a.nx * a.ny * harness.DEFAULT_NZ
    out["ns_per_cell_step"] = out["ms"] * 1e6 / out["cells"]
    dom.close()
    return out


def cmd_phase(a) -> dict:
    """Split the step into compute / exchange / wait with CUDA events.

    An event is per-device, so the two devices' timelines cannot be compared
    directly -- what CAN be compared is each device's own elapsed intervals
    and the host wall clock that brackets both.  The decomposition reported:

      compute[g]   device g's step, stream-elapsed (includes any gap the host
                   leaves between launches, which is the point)
      exch[g]      device g's pack+transfer, and its unpack, stream-elapsed
      wait         max(compute) - min(compute): the faster card's idle time
      exposed      wall - max(compute): what the exchange costs after overlap
    """
    import cupy as cp

    cfg = harness.make_config(a.nx, a.ny)
    dom = make_domain(a, cfg)
    if not a.skip_load:
        host0 = load_init(a.nx, a.ny, a.seed)
        dom.load_from_host(host0)
        del host0
        gc.collect()
    ng = dom.ngpu
    ev = {k: [] for k in ("c0", "c1", "x0", "x1")}
    for dev in dom.devices:
        with cp.cuda.Device(dev):
            for k in ev:
                ev[k].append(cp.cuda.Event(block=False, disable_timing=False))

    kw = dict(step_mode=a.step_mode, exchange_mode=a.exchange_mode,
              exchange=not a.no_exchange)
    dom.run(2, **kw)                                # warm

    rows = []
    for _ in range(a.reps):
        dom.sync_all()
        t0 = time.perf_counter()
        # --- compute, launched on both devices without a barrier between
        for g, dev in enumerate(dom.devices):
            with cp.cuda.Device(dev):
                ev["c0"][g].record(dom.compute_streams[g])
        from gpuwm.core.dycore import step
        launch0 = time.perf_counter()
        for g, dev in enumerate(dom.devices):
            with cp.cuda.Device(dev):
                with dom.compute_streams[g]:
                    step(dom.states[g], dom.sub_cfgs[g])
        launch1 = time.perf_counter()
        for g, dev in enumerate(dom.devices):
            with cp.cuda.Device(dev):
                ev["c1"][g].record(dom.compute_streams[g])
        for g, dev in enumerate(dom.devices):
            with cp.cuda.Device(dev):
                dom.compute_streams[g].synchronize()
        tc = time.perf_counter()
        # --- exchange
        if not a.no_exchange:
            for g, dev in enumerate(dom.devices):
                with cp.cuda.Device(dev):
                    ev["x0"][g].record(dom.copy_streams[g])
            dom.exchange_stream()
            for g, dev in enumerate(dom.devices):
                with cp.cuda.Device(dev):
                    ev["x1"][g].record(dom.unpack_streams[g])
        dom.sync_all()
        t1 = time.perf_counter()
        row = {"wall": (t1 - t0) * 1e3,
               "launch": (launch1 - launch0) * 1e3,
               "compute_wall": (tc - t0) * 1e3,
               "exch_wall": (t1 - tc) * 1e3}
        comp = []
        for g, dev in enumerate(dom.devices):
            with cp.cuda.Device(dev):
                comp.append(cp.cuda.get_elapsed_time(ev["c0"][g],
                                                     ev["c1"][g]))
        row["compute"] = comp
        row["wait"] = max(comp) - min(comp)
        if not a.no_exchange:
            xs = []
            for g, dev in enumerate(dom.devices):
                with cp.cuda.Device(dev):
                    xs.append(cp.cuda.get_elapsed_time(ev["x0"][g],
                                                       ev["x1"][g]))
            row["exch"] = xs
        rows.append(row)

    out = {"kind": "phase", "nx": a.nx, "ny": a.ny, "grid": list(dom.grid),
           "reps": a.reps, "exchange": not a.no_exchange,
           "seam_MiB": dom.seam_bytes / 2**20, "halo": dom.halo,
           "sub_shape": [[s.cny, s.cnx] for s in dom.specs]}
    for key in ("wall", "launch", "compute_wall", "exch_wall", "wait"):
        out[key] = median([r[key] for r in rows])
    out["compute"] = [median([r["compute"][g] for r in rows])
                      for g in range(ng)]
    if not a.no_exchange:
        out["exch"] = [median([r["exch"][g] for r in rows]) for g in range(ng)]
    out["exposed_exchange"] = out["wall"] - max(out["compute"])
    out["rows"] = rows[:3]
    dom.close()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("init", "initf", "initx2", "mono", "mg",
                                   "phase", "solo"))
    p.add_argument("nx", type=int)
    p.add_argument("ny", type=int)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--ngpu", type=int, default=2)
    p.add_argument("--grid", default=None)
    p.add_argument("--devices", default=None)
    p.add_argument("--step-mode", default="interleaved")
    p.add_argument("--exchange-mode", default="stream")
    p.add_argument("--transport", default="peer")
    p.add_argument("--no-exchange", action="store_true")
    p.add_argument("--skip-load", action="store_true")
    p.add_argument("--frugal", action="store_true")
    p.add_argument("--seed", type=int, default=harness.DEFAULT_SEED)
    p.add_argument("--seed-b", type=int, default=999331)
    p.add_argument("--out-seed", type=int, default=777777)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)
    if a.grid:
        a.grid = tuple(int(v) for v in a.grid.lower().split("x"))
    if a.devices:
        a.devices = [int(v) for v in a.devices.split(",")]

    if a.cmd == "init":
        res = build_init(a.nx, a.ny, a.seed, a.device)
    elif a.cmd == "initf":
        res = build_init_frugal(a.nx, a.ny, a.seed, a.device)
    elif a.cmd == "initx2":
        build_init(a.nx, a.ny, a.seed, a.device)
        build_init(a.nx, a.ny, a.seed_b, a.device)
        res = build_init_x2(a.nx, a.ny, a.seed, a.seed_b, a.out_seed)
    else:
        res = {"mono": cmd_mono, "mg": cmd_mg, "phase": cmd_phase,
               "solo": cmd_solo}[a.cmd](a)
    res["tag"] = a.tag
    print("RESULT " + json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
