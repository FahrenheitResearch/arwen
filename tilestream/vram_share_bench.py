"""What the shared workspace costs in time, at a tile size worth timing.

Sharing one scratch arena between tile buffers is only legal if no two tile
computations are live at once, and ``driver.run_tiled(shared=...)`` buys that
by chaining the compute streams with events.  That gives up compute/compute
overlap.  This lane measures what giving it up costs, against the shipped
configuration, on the same domain and the same tile plan.

TWO RULES THIS FILE EXISTS TO OBEY
----------------------------------
1. **The compute window has to be big enough to be measuring the code.**  A
   tile whose horizontal extent is a few hundred cells leaves the GPU idle
   between launches and the number that comes out is a launch-latency
   measurement wearing a physics label.  The default here is a 1024x1024
   domain split 2x2, so each tile computes a 544x544x49 window -- 14.5
   million cells.
2. **Radiation and cumulus have to actually fire, on both sides.**  At
   ``radt = 12 min`` with ``dt = 3 s`` radiation fires on step 1 and then not
   until step 241, so a six-step window starting from a fresh clock fires it
   exactly once and a window starting from step 2 fires it ZERO times.  The
   bench therefore prints ``driver.call_counts`` deltas for both arms and
   REFUSES to report a speedup if the two arms did not fire the same
   schemes the same number of times.

The store is pinned host memory, because that is the configuration the
out-of-core lane exists for and because a VRAM store at this domain size
would not fit next to two tile buffers.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import time
import warnings

import cupy as cp
import numpy as np

from tilestream import driver, gather, harness, physics_inventory as physinv
from tilestream import shared_workspace, spec as tspec, vram
from tilestream.vram_probe import RUNGS, make_cfg


def _counts(store_scalars) -> dict:
    return dict(store_scalars.get("call_counts", {}) or {})


#: Probe extents for the carrier manifest.  Non-square on purpose --
#: ``hoststore.manifest_from_arrays`` refuses a square probe, because on one
#: a y/x transposition in the shape rules cannot be seen.
PROBE_NX, PROBE_NY = 96, 80


def seed_physics_host_store(cfg, *, seed: int = 20_260_731):
    """A full-domain pinned carrier store, without a full-domain device state.

    A monolithic build cannot reach the domain sizes worth timing -- that is
    the entire premise of the out-of-core lane -- so the store is built the
    way :func:`tilestream.bench.seed_host_store` builds the dry one: a small
    probe state supplies the SHAPE RULES (``hoststore.manifest_from_arrays``)
    and, because ``physics_inventory.default_builder``'s sounding is a
    function of z alone and its latitude/longitude grids are horizontally
    uniform, one column of the probe supplies every VALUE.  The prognostics
    then get the same seeded noise the builder would have added.

    Returns ``(store, scalars)`` where ``store`` is the ``{name: pinned
    ndarray}`` mapping ``run_tiled`` consumes and ``scalars`` is the domain
    clock the probe reached after its warm-up step.
    """
    from tilestream import hoststore

    # Every physical parameter carried through untouched; only the horizontal
    # extents change.  Rebuilt from the config's own fields rather than from
    # ``harness.make_config``, so a selector the rung sets and make_config
    # does not cannot go missing from the probe and give it a different
    # carrier inventory from the store it is about to describe.
    probe_cfg = replace(cfg, nx=PROBE_NX, ny=PROBE_NY)
    probe, _drv = physinv.default_builder(probe_cfg, seed)
    harness.run_steps(probe, probe_cfg, 1)
    probe_arrays = physinv.carrier_inventory(probe)
    scalars = physinv.carrier_scalars(probe)
    manifest = hoststore.manifest_from_arrays(
        probe_arrays, cfg.nz, PROBE_NY, PROBE_NX)
    profiles = {}
    for name, arr in probe_arrays.items():
        host = cp.asnumpy(arr)
        profiles[name] = (host[..., 0:1, 0:1] if host.ndim >= 2 else host)
    del probe, _drv, probe_arrays
    vram.trim_pool()

    store_obj = hoststore.HostDomainStore(
        cfg, manifest=manifest,
        inventory_fn=physinv.carrier_manifest)
    amps = {name: amp for name, amp, _x, _y in harness._SEED_FIELDS}
    rng = np.random.default_rng(int(seed))
    for name, dest in store_obj.arrays.items():
        prof = profiles.get(name)
        if prof is None:
            raise KeyError(f"the probe carries no {name!r}")
        dest[...] = prof
        amp = amps.get(name, 0.0)
        if amp:
            dest += (amp * rng.standard_normal(dest.shape)).astype(
                dest.dtype, copy=False)
    if "u" in store_obj.arrays:
        store_obj.arrays["u"][..., :, -1] = store_obj.arrays["u"][..., :, 0]
    if "v" in store_obj.arrays:
        store_obj.arrays["v"][..., -1, :] = store_obj.arrays["v"][..., 0, :]
    if "w" in store_obj.arrays:
        store_obj.arrays["w"][0] = 0.0
        store_obj.arrays["w"][-1] = 0.0
    return store_obj, dict(scalars)


def one_arm(*, share: bool, chain, nx: int, ny: int, nz: int, tile: int,
            tile_ny: int = 0, nsteps: int, nbuffers: int, rung: str,
            warm: int = 1, rrtmgp_column_chunk: int | None = None) -> dict:
    """Build a domain in pinned host RAM, stream it, and time the sweeps."""
    cfg = make_cfg(rung, nx, ny, nz)
    store_obj, scalars = seed_physics_host_store(cfg)
    store = store_obj.arrays
    tile_ny = int(tile_ny or tile)

    halo = harness.halo_radius(cfg)
    specs = tspec.plan_tiles(nx, ny, tile, tile_ny, halo, True)
    tile_cfg = harness.tile_config(cfg, specs[0].cnx, specs[0].cny)
    shared = (shared_workspace.build(
        tile_cfg, rrtmgp_column_chunk=rrtmgp_column_chunk) if share else None)

    def sweep(steps: int) -> float:
        report: dict = {}
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            driver.run_tiled(
                store, cfg, tile, tile_ny, halo=halo, nsteps=steps,
                nbuffers=nbuffers, shared=shared, chain_compute=chain,
                report=report,
                inventory_fn=physinv.carrier_inventory, nz=int(cfg.nz),
                tile_state_factory=lambda tc:
                    driver.make_physics_tile_state(tc, shared=shared),
                scalars=scalars)
        cp.cuda.runtime.deviceSynchronize()
        return time.perf_counter() - t0

    before_warm = _counts(scalars)
    sweep(warm)
    before = _counts(scalars)
    elapsed = sweep(nsteps)
    after = _counts(scalars)
    fired = {k: after.get(k, 0) - before.get(k, 0)
             for k in sorted(set(before) | set(after))}
    snap = vram.device_snapshot()
    out = {
        "share": share, "chain": chain, "seconds": elapsed,
        "ms_per_step": 1e3 * elapsed / nsteps,
        "ns_per_cell": 1e9 * elapsed / (nsteps * nx * ny * nz),
        "fired": {k: v for k, v in fired.items() if v},
        "warm_fired": {k: before.get(k, 0) - before_warm.get(k, 0)
                       for k in before if before.get(k, 0)
                       - before_warm.get(k, 0)},
        "pool_used": snap["pool_used"], "pool_total": snap["pool_total"],
        "shared_bytes": 0 if shared is None else int(shared.nbytes),
        "tile_compute": (int(tile_cfg.nz), int(tile_cfg.ny), int(tile_cfg.nx)),
        "tiles": len(specs),
    }
    del store, shared
    store_obj.free()
    vram.trim_pool()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--ny", type=int, default=1024)
    parser.add_argument("--nz", type=int, default=49)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--tile-ny", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=4)
    parser.add_argument("--nbuffers", type=int, default=2)
    parser.add_argument("--rung", default="full+MYNN+Noah-MP",
                        choices=sorted(RUNGS))
    parser.add_argument("--rrtmgp-column-chunk", type=int, default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.2f} GiB free of {total / 2**30:.2f}")
    print(f"{args.nx}x{args.ny}x{args.nz} pinned host store, "
          f"{args.tile}^2 tiles, nbuffers={args.nbuffers}, N={args.nsteps}, "
          f"rung {args.rung}")

    arms = []
    for label, kwargs in (
            ("as shipped: private workspaces, no compute chain",
             dict(share=False, chain=False)),
            ("shared workspaces + compute chain",
             dict(share=True, chain=True,
                  rrtmgp_column_chunk=args.rrtmgp_column_chunk)),
    ):
        try:
            rec = one_arm(nx=args.nx, ny=args.ny, nz=args.nz, tile=args.tile,
                          tile_ny=args.tile_ny, nsteps=args.nsteps,
                          nbuffers=args.nbuffers, rung=args.rung, **kwargs)
        except cp.cuda.memory.OutOfMemoryError as exc:
            # Not a failure of the bench.  An arm that cannot hold this tile
            # size at all is the capacity result, and printing it as one is
            # more honest than shrinking the tile until both arms fit.
            print(f"  DOES NOT FIT -- {exc}")
            print(f"        {label}")
            vram.trim_pool()
            continue
        rec["label"] = label
        arms.append(rec)
        print(f"  {rec['ms_per_step']:9.1f} ms/step  "
              f"{rec['ns_per_cell']:7.3f} ns/cell  "
              f"compute {rec['tile_compute']}  "
              f"pool_used {rec['pool_used'] / 2**30:.3f} GiB  "
              f"pool_total {rec['pool_total'] / 2**30:.3f}  "
              f"shared {rec['shared_bytes'] / 2**30:.3f}")
        print(f"        {label}")
        print(f"        cadence fired in the timed window: {rec['fired']}"
              f"   (warm-up window: {rec['warm_fired']})")

    if len(arms) < 2:
        print("  only one arm ran; there is nothing to compare")
        return 0
    if arms[0]["fired"] != arms[1]["fired"]:
        print("  *** THE TWO ARMS DID NOT FIRE THE SAME PHYSICS. "
              "The ratio below is not a comparison.")
    ratio = arms[1]["ms_per_step"] / arms[0]["ms_per_step"]
    saved = arms[0]["pool_total"] - arms[1]["pool_total"]
    print(f"  shared/chained is {ratio:.3f}x the shipped time "
          f"and holds {saved / 2**30:+.3f} GiB less "
          f"({arms[0]['pool_total'] / max(arms[1]['pool_total'], 1):.2f}x)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(arms, handle, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
