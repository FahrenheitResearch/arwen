"""STRUCTURAL ATTACK 2+3: is the VACUOUS escape hatch a disarmed control,
and is the sharing physically real?

ATTACK 2 -- THE INSTRUMENT.
``test_share._dry_verdict`` turns the workstream's headline negative control
from a FAIL into a PASS whenever ``report['overlapping_steps'] == 0``.  That
is defensible ONLY if the counter can actually be non-zero.  If the timeline
is broken -- events on the wrong stream, elapsed-time read the wrong way --
it reports 0 everywhere, every hazard row is VACUOUS on every card, and the
control can never fail again.  That is precisely the shape of this project's
six false results.

So: run the driver UNCHAINED and UNSHARED (nothing to corrupt, no chain to
serialise) on tiles small enough that two obviously co-schedule, and require
``overlapping_steps > 0``.  If the counter is still 0, the instrument is
dead and every VACUOUS verdict in the report is worthless.
Then run the SAME geometry chained and require 0.  Both directions, one card.

ATTACK 3 -- IS THE SHARING REAL?
Every positive row in ``test_share`` passes if the sharing silently did
NOTHING: identical answers is exactly what "no sharing happened" predicts.
The memory numbers are the only evidence offered that the feature is on, and
they are pool deltas, which are indirect.  So compare DEVICE POINTERS: two
tile buffers built from one ``SharedTileWorkspaces`` must hand out the SAME
address for an arena-admitted scratch slot, and DIFFERENT addresses without
it.  If the pointers differ under sharing, the whole saving is fictitious and
every bit-exact row is a tautology.
"""
import sys

sys.argv = [sys.argv[0]]

import warnings

import cupy as cp

from tilestream import (driver, harness, physics_inventory as physinv,
                        shared_workspace, spec as tspec, vram)
from tilestream.test_gate import NZ, SEED, physics_cfg
from tilestream.test_share import _dry_tile_factory

free, total = cp.cuda.runtime.memGetInfo()
name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
print(f"{name}  {free / 2**30:.2f} GiB free of {total / 2**30:.2f}")
print("=" * 78)

failures = []

# ---------------------------------------------------------------- ATTACK 2
print("ATTACK 2: can report['overlapping_steps'] EVER be non-zero?")
print("-" * 78)


def dry_timeline(nx, ny, tile, *, chain, share=False, nsteps=3, nbuffers=2):
    cfg = harness.make_config(nx, ny, NZ)
    state = harness.make_state(cfg, SEED)
    start = {k: cp.asnumpy(v).copy()
             for k, v in harness.state_arrays(state).items()}
    del state
    vram.trim_pool()
    halo = harness.halo_radius(cfg)
    specs = tspec.plan_tiles(nx, ny, tile, tile, halo, True)
    tile_cfg = harness.tile_config(cfg, specs[0].cnx, specs[0].cny)
    shared = (shared_workspace.build(tile_cfg, rrtmgp=False) if share
              else None)
    store = {k: cp.asarray(v) for k, v in start.items()}
    report = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver.run_tiled(store, cfg, tile, tile, halo=halo, nsteps=nsteps,
                         nbuffers=nbuffers, shared=shared, chain_compute=chain,
                         report=report, timeline=True,
                         tile_state_factory=_dry_tile_factory(shared))
    cp.cuda.runtime.deviceSynchronize()
    del store, shared
    vram.trim_pool()
    return report


# Small tiles first: a 32x32x49 compute block cannot fill any modern card, so
# two of them MUST be able to run at once if nothing orders them.
for nx, ny, tile in ((192, 192, 32), (192, 192, 64), (96, 96, 32)):
    rep = dry_timeline(nx, ny, tile, chain=False)
    ntiles = rep["tiles"]
    print(f"  UNCHAINED {nx}x{ny} tiles {tile}x{tile} ({ntiles} tiles x 3 "
          f"steps): overlapping_steps = {rep['overlapping_steps']}")

rep_chained = dry_timeline(192, 192, 32, chain=True, share=True)
print(f"  CHAINED   192x192 tiles 32x32 (shared arena): "
      f"overlapping_steps = {rep_chained['overlapping_steps']}")

unchained_max = max(
    dry_timeline(nx, ny, t, chain=False)["overlapping_steps"]
    for nx, ny, t in ((192, 192, 32), (96, 96, 32)))
if unchained_max == 0:
    failures.append(
        "INSTRUMENT DEAD: overlapping_steps stayed 0 for every unchained "
        "configuration, so the VACUOUS escape hatch can never be closed and "
        "the hazard control can never fail on this card by construction")
    print("  *** the counter never moved -- see verdict")
else:
    print(f"  the counter reaches {unchained_max}, so it is alive")
if rep_chained["overlapping_steps"] != 0:
    failures.append(
        "THE CHAIN DOES NOT SERIALISE: chained run still reported "
        f"{rep_chained['overlapping_steps']} overlapping steps")

# ---------------------------------------------------------------- ATTACK 3
print()
print("ATTACK 3: is the sharing physically real (same device pointer)?")
print("-" * 78)

cfg = physics_cfg("full+MYNN+Noah-MP")
tile_cfg = harness.tile_config(cfg, 96, 80)

shared = shared_workspace.build(tile_cfg)
print(f"  shared workspaces: {shared.describe()}")

sa = driver.make_physics_tile_state(tile_cfg, shared=shared, warmup=1)
sb = driver.make_physics_tile_state(tile_cfg, shared=shared, warmup=1)
pa = driver.make_physics_tile_state(tile_cfg, shared=None, warmup=1)
pb = driver.make_physics_tile_state(tile_cfg, shared=None, warmup=1)


def slot_ptr(state, slot):
    """Device address of a scratch slot, without creating it if absent."""
    cache = getattr(state, "_scratch", {})
    buf = cache.get(slot)
    if buf is None:
        return None
    return int(buf.data.ptr)


live = sorted(set(getattr(sa, "_scratch", {})) & set(getattr(sb, "_scratch", {}))
              & set(getattr(pa, "_scratch", {})) & set(getattr(pb, "_scratch", {})))
admitted = set(shared_workspace.arena_shapes(tile_cfg))
checked = [s for s in live if s in admitted]
print(f"  scratch slots realised in all four buffers: {len(live)}, "
      f"of which arena-admitted: {len(checked)}")

same_shared = [s for s in checked if slot_ptr(sa, s) == slot_ptr(sb, s)]
same_private = [s for s in checked if slot_ptr(pa, s) == slot_ptr(pb, s)]
print(f"  SHARED  buffers agreeing on address: {len(same_shared)}/{len(checked)}")
print(f"  PRIVATE buffers agreeing on address: {len(same_private)}/{len(checked)}")
for s in checked[:6]:
    print(f"     {s:28s} shared {hex(slot_ptr(sa, s))}/{hex(slot_ptr(sb, s))}"
          f"   private {hex(slot_ptr(pa, s))}/{hex(slot_ptr(pb, s))}")

if not checked:
    failures.append("ATTACK 3 inconclusive: no arena-admitted slot was "
                    "realised in all four buffers")
elif len(same_shared) != len(checked):
    failures.append(
        f"SHARING IS NOT REAL: only {len(same_shared)}/{len(checked)} "
        "arena-admitted slots share an address between two buffers built "
        "from one SharedTileWorkspaces")
if same_private:
    failures.append(
        f"CONTROL BROKEN: {len(same_private)} slots share an address even "
        "WITHOUT sharing, so pointer identity proves nothing")

# does the arena actually back them?
if shared.arena is not None and checked:
    base = int(shared.arena._buffer.data.ptr) if hasattr(shared.arena, "_buffer") else None
    print(f"  arena nbytes {shared.arena.nbytes / 2**20:.1f} MiB"
          + (f", base {hex(base)}" if base else ""))

print()
print("=" * 78)
if failures:
    print(f"ATTACKS LANDED -- {len(failures)}:")
    for f in failures:
        print(f"  * {f}")
    raise SystemExit(1)
print("both attacks repelled on this card")
