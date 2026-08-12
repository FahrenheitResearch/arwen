"""THE BIT-EXACT GATE: a tiled run must equal a monolithic run, exactly.

Run it::

    python -m tilestream.test_gate            # from the repository root

Everything else in this prototype is performance work.  This file is the only
thing that says the performance work is worth anything, because a streaming
bug that skips or reorders work looks exactly like a speedup and only an
exact comparison can tell the two apart.

What is compared
----------------
A seeded, fully periodic domain is stepped ``N`` times two ways -- once as one
array, once as tiles gathered with a halo, stepped independently and scattered
back -- and the SHA-256 over the whole persisted inventory must be identical.
Not close: identical.  ``harness.hash_state`` folds each array's name, dtype,
shape and raw bytes into one digest, so a shape change cannot masquerade as a
value change.

Periodic is deliberate.  With ``open_x=open_y=specified=nested=False`` every
tile is an interior tile, the halo gather is a wraparound slice, and there is
no lateral-boundary special case anywhere -- which removes the entire BC
question from the milestone and leaves exactly one thing under test: whether a
16-cell halo makes a tile's interior exact.

The negative controls matter as much as the positives
-----------------------------------------------------
A gate that has only ever passed proves nothing.  Four configurations are
asserted to FAIL, and each one fails for a different reason:

* ``halo=8`` and ``halo=11`` -- too narrow.  11 is one cell below the measured
  minimum and fails by ~1e-5 on ``php``/``w`` only, which is what a
  marginally-short halo looks like as distinct from a broken one.
* ``write_mode="inplace"`` -- the read-at-time-t bug: tiles read the buffer
  their predecessors have already written, so a tile's halo sees its
  neighbours at ``t + dt``.  This is the single most likely way to write a
  wrong tiling, it produces a perfectly plausible-looking forecast, and this
  is the row that proves the gate would catch it.
* ``halo`` above the measured minimum is NOT slack to be reclaimed.  The
  sweep reports the smallest width that passes at each step count, and the
  answer moves: 12 at N=1, 13 at N=3, 14 at N=8.  See :func:`halo_vs_steps`
  -- a one-step gate certifies halo 12, and halo 12 is wrong by 0.2 after two
  steps.  The halo comes from ``harness.halo_radius(cfg)``, and the
  measurement's job is to confirm that value, not to shrink it.

Milestone two: the same gate with FULL PHYSICS
----------------------------------------------
A dry bit-exact gate proves the MECHANISM.  Only a full-physics gate proves
the product, and the physics section (``--physics-only`` runs just it,
``--dry-only`` just the original) is the one that does.  It streams the whole
restart manifest -- 138 to 229 arrays, ``state/*`` plus ``scratch/*`` plus
``driver/*`` plus ``fields/*`` plus ``cumulus/w0avg`` -- across fourteen
rungs from dry to full physics + MYNN + Noah-MP, at 2x1 / 2x2 / 3x3-ragged /
1x1 geometry, N = 1 / 3 / 8, VRAM store and pinned host store.

Its negative controls are what make it mean anything, and each names a
distinct way this milestone could have been faked:

* streaming ``state/*`` only -- which is EXACTLY milestone one's inventory,
  run at a physics rung.  127 of 229 carriers differ.  That failure is the
  reason this milestone exists.
* not carrying the scalar clock at a fast cadence -- 46 of 153 differ.  Note
  the same control at ``radt=12 min, cudt=5 min`` PASSES: 8 steps of 3 s
  never reach a cadence boundary, so a long-cadence short run cannot see a
  clock bug at all.
* ``write_mode="inplace"`` -- the read-at-time-t bug, 89 of 155 differ.
* ``halo=13`` at 256x192 with 4x4 tiles -- 111 of 229 differ.  It has to be
  run at that size: at the 96x80 gate size a 48x40 tile with halo 16 already
  gathers 83% by 90% of the domain, and halo 13 PASSES there.  The cheap
  test certifies the wrong halo; that is FACT 1 again.
* a horizontally varying radiation latitude grid -- refused by
  ``physics_inventory.assert_tileable_geography``, because in THAT section a
  tile still rebuilds its geography and the tile centred on the domain is the
  one that agrees.  The geography section below removes the restriction
  rather than living with it.

Milestone three: the same gate with a REAL MAP PROJECTION
---------------------------------------------------------
``--geography-only`` runs it.  A real Lambert conformal grid at
``configs/real74_d01.toml``'s own dx=dy=12 km, a lat/lon-anchored terrain and
per-column latitude/longitude on every scheme -- so ``has_msf`` and
``rotational`` are both TRUE, the Coriolis+curvature kernel and the
msf-weighted paths are live, ``thb/pb/alb/phb`` are 3-D and terrain-following,
and the solar-zenith path reads a latitude that varies.  The 17 geography
arrays are GATHERED per tile, exactly like a carrier, but ONCE per buffer
occupancy rather than per step, and never scattered.

Its negative controls, each naming a different way this could have been faked:

* geography REBUILT from ``tile_cfg`` -- today's behaviour.  114 of 229
  carriers differ.  The rebuilt tile is displaced by
  ``ci0 + (cnx+1)/2 - (nx+1)/2`` cells per axis, MEASURED up to 12.41 deg of
  latitude and 2069 km at 192x160 split 3x3.
* ``has_msf``/``rotational`` derived from the TILE's window instead of the
  domain's -- 32 of 229 differ, with the geography arrays themselves correct.
* the setup arrays gathered but the SCHEME lat/lon grids rebuilt, at the fast
  cadence where radiation fires every step -- 47 of 153 differ.  Those four
  arrays are neither on the state nor in the carrier manifest, so nothing
  else in this pipeline would move them.
* the scheme lat/lon gathered but the setup arrays rebuilt -- 48 of 153.
* ``periodic_faces=False``: a periodic domain whose ``msfv[ny,:]`` does not
  duplicate ``msfv[0,:]`` -- 54 of 229 differ.
* ``halo=7`` at 256x192 with 4x4 tiles.  NOTE the number: at dx=12 km the
  smallest bit-exact halo is 8, where the identical geometry at dx=500 m
  needs 14.  A coarse grid is a WEAKER halo test.

Diagnosis, not just a verdict
-----------------------------
A failing row is analysed rather than reported: per-field digests localise
which field broke first, and :func:`spatial_signature` classifies the pattern
of differing cells -- clustered within ``halo`` of a tile seam (a halo width
problem), spread uniformly (a global reduction or a changed reduction order),
or confined to one tile (ragged-edge handling).  The max relative difference
separates the two failure families that need completely different fixes: ~1e-7
relative is floating-point reassociation, while ~1e-1 is a wrong stencil.
"""

from __future__ import annotations

import hashlib
import sys
import time
import traceback
import warnings

import numpy as np

from tilestream import driver, gather, harness
from tilestream import spec as tspec
# One implementation of the box-wide context count, shared with the slice
# gate rather than spelled a second time: two counters that disagree is a
# worse outcome than one counter in a module this one already sits beside.
from tilestream.test_decomp_gate import (context_verdict,
                                         cuda_contexts)


NZ = 49
SEED = harness.DEFAULT_SEED


#: The prefix a case prints when THE CARD, not the code, decided the answer.
#: Counted separately from PASS and FAIL because it is neither: nothing was
#: proved and nothing was disproved, and a run carrying any of these is a run
#: with a hole in its coverage that the verdict has to name.
MACHINE_LIMITED = "  MACHINE-LIMITED"


class _CheckCounter:
    """A stdout wrapper that counts the gate's own ``PASS`` / ``FAIL`` lines.

    The verdict at the bottom of this gate used to assert totality --
    "every configuration behaved as specified" -- while saying nothing about
    how many configurations there were, and the 233 that gets quoted lived
    only in the operator's scrollback as a count of these lines.  An empty
    case list therefore printed the strongest sentence in the file having
    proved nothing, which is the defect ``d998bb667`` fixed across four
    release gates on the same night.

    Counting the emitted lines rather than instrumenting each print site is
    deliberate: five of the six sections are delegated to sibling modules
    that print their own ``PASS`` lines, and a counter that only saw this
    module's prints would state a size smaller than the verdict covers.
    """

    def __init__(self, inner):
        self._inner = inner
        self._partial = ""
        self.passed = 0
        self.failed = 0
        self.machine_limited = 0

    def write(self, text):
        written = self._inner.write(text)
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            if line.startswith(MACHINE_LIMITED):
                self.machine_limited += 1
            elif line.startswith("  PASS"):
                self.passed += 1
            elif line.startswith("  FAIL"):
                self.failed += 1
        return written

    def flush(self):
        self._inner.flush()

    def __getattr__(self, name):
        return getattr(self._inner, name)


# --------------------------------------------------------------------------
# comparison machinery
# --------------------------------------------------------------------------

def hash_arrays(arrays) -> str:
    """``harness.hash_state``'s digest over a plain ``{name: array}`` map."""
    digest = hashlib.sha256()
    for name, array in arrays.items():
        digest.update(harness._digest_payload(f"state.{name}", array))
    return digest.hexdigest()


def field_digests(arrays) -> dict[str, str]:
    return {name: hashlib.sha256(
        harness._digest_payload(f"state.{name}", array)).hexdigest()
        for name, array in arrays.items()}


def _as_numpy(array) -> np.ndarray:
    import cupy as cp
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) else np.asarray(array)


def spatial_signature(diff: np.ndarray, spec_list, halo: int) -> dict:
    """Classify WHERE a field differs.  This is the decisive diagnostic.

    ``diff`` is ``|tiled - monolithic|`` at full rank; it is reduced over every
    leading (vertical) axis to a horizontal mask of differing columns, and each
    differing column is measured against the nearest tile-interior boundary in
    the periodic metric.  The returned ``verdict`` reads:

    ``"seam-local"``
        every differing column is within ``halo`` of a seam -- the halo is too
        narrow, or something reaches further than it is supposed to.
    ``"uniform"``
        more than half the domain differs -- a global reduction, or a
        reduction whose order changed with the array size.
    ``"single-tile"``
        differences confined to one tile's interior -- ragged or edge-tile
        handling.
    ``"corners"``
        differing columns sit only near tile corners -- the corner halo was
        not populated.
    """
    mask = diff > 0.0
    while mask.ndim > 2:
        mask = mask.any(axis=0)
    ny, nx = mask.shape
    total = int(mask.sum())
    out: dict = {"differing_columns": total, "columns": mask.size}
    if total == 0:
        out["verdict"] = "identical"
        return out

    xs_edges = sorted({s.i0 for s in spec_list} | {s.i1 % max(nx, 1)
                                                   for s in spec_list})
    ys_edges = sorted({s.j0 for s in spec_list} | {s.j1 % max(ny, 1)
                                                   for s in spec_list})

    def _dist(idx, edges, n):
        return min(min((idx - e) % n, (e - idx) % n) for e in edges)

    jj, ii = np.nonzero(mask)
    dx = np.array([_dist(int(i), xs_edges, nx) for i in ii])
    dy = np.array([_dist(int(j), ys_edges, ny) for j in jj])
    near = np.minimum(dx, dy)
    out.update(max_distance_to_seam=int(near.max()),
               mean_distance_to_seam=float(near.mean()),
               fraction=total / mask.size)

    owning = {}
    for s in spec_list:
        sel = ((jj >= s.j0) & (jj < s.j1) & (ii >= s.i0) & (ii < s.i1))
        if sel.any():
            owning[s.index] = int(sel.sum())
    out["tiles_touched"] = len(owning)

    if total > 0.5 * mask.size:
        out["verdict"] = "uniform"
    elif len(owning) == 1:
        out["verdict"] = "single-tile"
    elif int(near.max()) <= halo:
        corner = ((dx <= halo) & (dy <= halo)).mean()
        out["verdict"] = "corners" if corner > 0.9 else "seam-local"
    else:
        out["verdict"] = "scattered"
    return out


def compare(tiled_arrays, reference_arrays, spec_list, halo: int) -> dict:
    """Full comparison record: digests, magnitudes and spatial pattern.

    ``the_comparison_is_not_empty`` is a DECLARED CONDITION and is evaluated
    before any per-carrier one, on the release line's certify / dual-run
    template (``d998bb667`` / ``9b1b99289``).  Two empty carrier maps hash to
    the SHA-256 of nothing, those two digests are equal, and the per-carrier
    loop below never runs, so ``bitexact`` would report True having compared
    nothing: a statement about all carriers that is true of none.  The floor
    is ONE rather than the full expected count, so a legitimately partial
    comparison stays a result instead of becoming a refusal.
    """
    got = field_digests(tiled_arrays)
    want = field_digests(reference_arrays)
    compared_count = len(set(got) | set(want))
    record: dict = {
        "compared_count": compared_count,
        "the_comparison_is_not_empty": compared_count >= 1,
        "bitexact": (compared_count >= 1
                     and hash_arrays(tiled_arrays)
                     == hash_arrays(reference_arrays)),
        "hash": hash_arrays(tiled_arrays),
        "ref_hash": hash_arrays(reference_arrays),
        "fields": {},
        "max_abs": 0.0, "max_rel": 0.0, "worst_field": None,
        "nonfinite": 0,
    }
    for name in want:
        a = _as_numpy(tiled_arrays[name]).astype(np.float64)
        b = _as_numpy(reference_arrays[name]).astype(np.float64)
        d = np.abs(a - b)
        # Relative error is measured against the FIELD's scale, not
        # pointwise. A pointwise |a-b|/|b| is meaningless here because w and
        # phi' are identically zero at the boundaries and small in the
        # interior, so a 1e-7 absolute wobble next to a 1e-30 reference
        # reports as 1e+23 and tells you nothing. Scaling by max|reference|
        # keeps the number interpretable, which is the whole point: ~1e-7
        # means floating-point reassociation (something summed in a different
        # order), while ~1e-1 means a wrong stencil. Those need opposite
        # fixes and must not be confused.
        field_scale = float(np.abs(b).max()) if b.size else 0.0
        rel = (float(d.max()) / field_scale) if field_scale > 0 else 0.0
        entry = {
            "bitexact": got[name] == want[name],
            "max_abs": float(d.max()) if d.size else 0.0,
            "max_rel": rel,
            "field_scale": field_scale,
            "n_differ": int((d > 0).sum()),
            "size": int(d.size),
        }
        record["nonfinite"] += int((~np.isfinite(a)).sum())
        if not entry["bitexact"] and entry["max_abs"] >= record["max_abs"]:
            record["max_abs"] = entry["max_abs"]
            record["worst_field"] = name
            record["signature"] = spatial_signature(d, spec_list, halo)
        record["max_rel"] = max(record["max_rel"], rel if not entry["bitexact"]
                                else 0.0)
        record["fields"][name] = entry
    return record


# --------------------------------------------------------------------------
# reference and tiled runs
# --------------------------------------------------------------------------

_REF_CACHE: dict = {}


def monolithic(nx: int, ny: int, nsteps: int, nz: int = NZ, seed: int = SEED):
    """``(cfg, start_arrays, reference_arrays)`` for one domain and step count.

    ``start_arrays`` is the seeded state BEFORE stepping (host copies, so the
    device memory can be released); ``reference_arrays`` is the same state
    after ``nsteps`` monolithic ``dycore.step`` calls.  Cached, because a
    192x192x49 reference is the expensive part of the gate and every row at
    that size shares it.
    """
    import cupy as cp

    key = (nx, ny, nz, nsteps, seed)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    cfg = harness.make_config(nx, ny, nz)
    state = harness.make_state(cfg, seed=seed)
    start = {n: _as_numpy(a).copy() for n, a in harness.state_arrays(state).items()}
    harness.run_steps(state, cfg, nsteps)
    ref = {n: _as_numpy(a).copy() for n, a in harness.state_arrays(state).items()}
    del state
    cp.get_default_memory_pool().free_all_blocks()
    _REF_CACHE[key] = (cfg, start, ref)
    return _REF_CACHE[key]


def run_case(nx, ny, tile_nx, tile_ny, halo, nsteps, *, nbuffers=1,
             write_mode="ring", host_store=False, nz=NZ, seed=SEED,
             ring_margin="exact", ring_ordering="events",
             graph=None) -> dict:
    """Run one tiled configuration and compare it to the monolithic answer.

    ``graph`` is the ``run_tiled`` graph-capture keyword set (or ``None``
    for stream launching).  It must not move a single bit, which is the
    whole reason it is threaded through the existing case runners rather
    than given a comparison of its own.
    """
    import cupy as cp

    cfg, start, ref = monolithic(nx, ny, nsteps, nz=nz, seed=seed)
    specs = tspec.plan_tiles(nx, ny, tile_nx, tile_ny, halo, True)
    tspec.validate_plan(specs, ny, nx)

    store_obj = None
    if host_store:
        from tilestream.hoststore import HostDomainStore
        store_obj = HostDomainStore(cfg)
        for name, arr in start.items():
            store_obj.arrays[name][...] = arr
        store = store_obj
    else:
        store = {n: cp.asarray(a) for n, a in start.items()}

    report: dict = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        extra = {} if write_mode != "ring" else dict(
            ring_margin=ring_margin, ring_ordering=ring_ordering)
        driver.run_tiled(store, cfg, tile_nx, tile_ny, halo=halo,
                         nsteps=nsteps, nbuffers=nbuffers,
                         write_mode=write_mode, report=report,
                         **extra, **(graph or {}))
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0

    final = driver._arrays_of(store)
    record = compare(final, ref, specs, halo)
    record.update(seconds=elapsed, tiles=len(specs),
                  compute=report.get("compute"),
                  redundancy=report["efficiency"]["redundancy"],
                  gathered_bytes=report["gathered_bytes"],
                  scattered_bytes=report["scattered_bytes"],
                  second_store_bytes=report.get("second_store_bytes"),
                  ring_bytes=report.get("ring_bytes", 0),
                  ring_over_store=report.get("ring_over_store", 0.0),
                  graph=report.get("graph"))

    del store, final
    if store_obj is not None:
        store_obj.free()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return record


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------

def precondition_inventory(nx=64, ny=64, nz=NZ) -> str:
    """The streamed set must be the WHOLE cross-step state (audit risk #1)."""
    cfg = harness.make_config(nx, ny, nz)
    driver.assert_streaming_inventory_complete(cfg, nsteps=8)
    names = sorted(gather.inventory(harness.make_state(cfg)))
    return (f"streamed inventory is complete at N=8: overwriting only "
            f"{len(names)} fields {names} on a dirtied state reproduces a "
            f"fresh run exactly")


def precondition_setup(nx=192, ny=192, tile=64, halo=16, nz=NZ) -> str:
    """A tile's setup arrays must equal the parent's window."""
    cfg = harness.make_config(nx, ny, nz)
    parent = harness.make_state(cfg, seed=SEED)
    specs = tspec.plan_tiles(nx, ny, tile, tile, halo, True)
    tile_cfg = harness.tile_config(cfg, specs[0].cnx, specs[0].cny)
    tile_state = driver.make_tile_state(tile_cfg)
    bad = {}
    for spec in specs:
        bad.update(driver.setup_window_mismatches(tile_state, parent, spec))
    n = len(harness.setup_arrays(parent))
    if bad:
        raise AssertionError(f"setup arrays differ from the parent window: {bad}")
    import cupy as cp
    del parent, tile_state
    cp.get_default_memory_pool().free_all_blocks()
    return (f"all {n} setup arrays equal the parent window for every one of "
            f"{len(specs)} tiles")


def precondition_hoststore(nx=96, ny=96, nz=NZ) -> str:
    """A pinned host store must round-trip a state without losing a bit."""
    import cupy as cp
    from tilestream.hoststore import HostDomainStore

    cfg = harness.make_config(nx, ny, nz)
    state = harness.make_state(cfg, seed=SEED)
    harness.run_steps(state, cfg, 1)
    want = harness.hash_state(state)
    store = HostDomainStore(cfg)
    store.fill_from(state)
    store.assert_pinned()
    got = store.hash()
    if got != want:
        raise AssertionError(f"host store round trip changed the state: "
                             f"{got[:16]} != {want[:16]}")
    mb = store.nbytes / 1e6
    store.free()
    del state
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return (f"pinned host store round trip is bit-exact ({mb:.1f} MB, "
            f"digest {got[:16]})")


def api_contracts(nx=128, ny=128, nz=NZ) -> list[str]:
    """The documented entry points, exercised so they cannot rot silently.

    The gate matrix drives ``run_tiled`` one way (dict or host store, driver
    allocates the shadow).  Everything else ``run_tiled`` promises in its
    docstring is checked here, because a documented path with no test is a
    liability: a caller-supplied shadow, a ``DomainState`` used directly as
    the store, ``plan_for``'s halo derivation, and the three refusals that
    stop a wrong run before it produces a plausible answer.
    """
    import cupy as cp

    from tilestream.hoststore import HostDomainStore

    cfg, start, ref = monolithic(nx, ny, 2, nz=nz)
    want = hash_arrays(ref)
    out: list[str] = []

    a, b = HostDomainStore(cfg), HostDomainStore(cfg)
    for name, arr in start.items():
        a.arrays[name][...] = arr
    driver.run_tiled(a, cfg, 64, 64, halo=16, nsteps=2, nbuffers=2, shadow=b,
                     write_mode="shadow")
    if a.hash() != want:
        raise AssertionError("caller-supplied pinned shadow diverged")
    a.free()
    b.free()
    out.append("caller-supplied pinned shadow store, N=2: bit-exact")

    # A shadow handed to the ring path would be silently ignored -- it keeps
    # one store -- so it is refused rather than accepted and wasted.
    c, d = HostDomainStore(cfg), HostDomainStore(cfg)
    try:
        driver.run_tiled(c, cfg, 64, 64, halo=16, nsteps=1, shadow=d)
    except driver.TiledRunError:
        out.append("a shadow supplied to write_mode='ring' is refused")
    else:                                              # pragma: no cover
        raise AssertionError("ring accepted a shadow buffer it cannot use")
    finally:
        c.free()
        d.free()

    store = {n: cp.asarray(v) for n, v in start.items()}
    specs, tile_cfg, dims = driver.plan_for(store, cfg, 64, 64)
    if specs[0].halo != harness.halo_radius(cfg):
        raise AssertionError("plan_for did not derive the halo from cfg")
    out.append(f"plan_for derives halo={specs[0].halo} from cfg, "
               f"{len(specs)} tiles, tile_cfg {tile_cfg.ny}x{tile_cfg.nx}")

    pageable = {n: np.ascontiguousarray(v) for n, v in start.items()}
    try:
        driver.run_tiled(pageable, cfg, 64, 64, halo=16, nsteps=1)
    except gather.PageableHostMemoryError:
        out.append("pageable host store refused (would run at 15 GB/s, not 57)")
    else:
        raise AssertionError("pageable host memory was accepted")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        driver.run_tiled(store, cfg, 64, 64, halo=10, nsteps=1)
    if not any("below the per-step dependency radius" in str(w.message)
               for w in caught):
        raise AssertionError("a too-narrow halo did not warn")
    out.append("halo below harness.halo_radius(cfg) warns")

    try:
        driver.run_tiled(store, harness.make_config(64, 64, nz), 32, 32,
                         nsteps=1)
    except driver.TiledRunError:
        out.append("cfg describing a different domain than the store refused")
    else:
        raise AssertionError("a mismatched cfg was accepted")

    state = harness.make_state(cfg, seed=SEED)
    driver.run_tiled(state, cfg, 64, 64, halo=16, nsteps=2, nbuffers=2)
    if harness.hash_state(state) != want:
        raise AssertionError("DomainState-as-store diverged")
    out.append("a DomainState used directly as the store, N=2: bit-exact")

    del store, state, pageable
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------

#: (label, kwargs, expected-to-pass)
CASES: list[tuple[str, dict, bool]] = [
    # --- easiest first: one seam, one step -------------------------------
    ("2x1 tiles (x seam only), N=1",
     dict(nx=192, ny=192, tile_nx=96, tile_ny=192, halo=16, nsteps=1), True),
    ("1x2 tiles (y seam only), N=1",
     dict(nx=192, ny=192, tile_nx=192, tile_ny=96, halo=16, nsteps=1), True),
    ("2x2 tiles (corners appear), N=1",
     dict(nx=192, ny=192, tile_nx=96, tile_ny=96, halo=16, nsteps=1), True),
    ("3x3 tiles, N=1",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=1), True),
    # --- then more steps --------------------------------------------------
    ("3x3 tiles, N=3",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3), True),
    ("3x3 tiles, N=3, nbuffers=2 (pipelined)",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3,
          nbuffers=2), True),
    ("3x3 tiles, N=3, nbuffers=3",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3,
          nbuffers=3), True),
    # --- then harder geometry --------------------------------------------
    ("6x6 tiles (36 tiles, tile 32 < 2*halo), N=1",
     dict(nx=192, ny=192, tile_nx=32, tile_ny=32, halo=16, nsteps=1), True),
    ("1x1 tile (window wraps onto itself), N=1",
     dict(nx=192, ny=192, tile_nx=192, tile_ny=192, halo=16, nsteps=1), True),
    ("ragged x: 200x192, tile 64 -> last tile 8 wide, N=1",
     dict(nx=200, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=1), True),
    ("ragged both, odd sizes: 173x149, tile 48x40, N=3",
     dict(nx=173, ny=149, tile_nx=48, tile_ny=40, halo=16, nsteps=3), True),
    # --- the real out-of-core path ---------------------------------------
    ("PINNED HOST STORE, 3x3 tiles, N=3, nbuffers=2",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3,
          nbuffers=2, host_store=True), True),
    # --- negative controls: these MUST fail ------------------------------
    ("NEGATIVE halo=8, 3x3, N=1",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=8, nsteps=1), False),
    ("NEGATIVE halo=11, 3x3, N=1 (one below the minimum)",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=11, nsteps=1), False),
    ("NEGATIVE write_mode=inplace, 3x3, N=1 (read-at-time-t bug)",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=1,
          write_mode="inplace"), False),
    ("NEGATIVE write_mode=inplace, 3x3, N=3",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3,
          write_mode="inplace"), False),
]

# --------------------------------------------------------------------------
# rings versus the shadow they replace
# --------------------------------------------------------------------------

def ring_vs_shadow(nx, ny, tile_nx, tile_ny, halo, nsteps, *, nbuffers=1,
                   host_store=False, nz=NZ, seed=SEED) -> dict:
    """One configuration run BOTH ways; the two digests must be identical.

    A capacity optimisation that changes the answer is a bug, not an
    optimisation, so this compares ring to shadow directly as well as each to
    the monolithic reference.  Ring and shadow are independent
    implementations of the read-at-time-t rule -- one keeps a whole second
    domain, the other keeps ~5% of one and orders the sweep -- so agreement
    between them is worth more than either agreeing with a cached answer.
    """
    kw = dict(nx=nx, ny=ny, tile_nx=tile_nx, tile_ny=tile_ny, halo=halo,
              nsteps=nsteps, nbuffers=nbuffers, host_store=host_store,
              nz=nz, seed=seed)
    ring = run_case(write_mode="ring", **kw)
    shadow = run_case(write_mode="shadow", **kw)
    return {
        "bitexact": ring["bitexact"] and shadow["bitexact"],
        "agree": ring["hash"] == shadow["hash"],
        "hash": ring["hash"],
        "ring": ring, "shadow": shadow,
        "tiles": ring["tiles"],
        "ring_bytes": ring["ring_bytes"],
        "ring_over_store": ring["ring_over_store"],
        "shadow_bytes": shadow["second_store_bytes"],
        "seconds": ring["seconds"] + shadow["seconds"],
    }


#: The full geometry x steps x store matrix, run both ways.  ``nbuffers`` is
#: not decoration here: with a prefetch depth of ``nbuffers-1``, a plan of
#: ``nbuffers`` tiles or fewer issues EVERY gather before ANY scatter, so the
#: store is never read after being written and the ring is never exercised at
#: all.  MEASURED: the ``x_only`` negative control, which changes the answer
#: on a 2-tile y-seam plan at ``nbuffers=1``, is bit-exact on the same plan at
#: ``nbuffers=2``.  Small plans therefore appear at ``nbuffers=1``.
RING_CASES: list[tuple[str, dict, bool]] = [
    ("2x1 (x seam), N=1, nb=1",
     dict(nx=192, ny=192, tile_nx=96, tile_ny=192, halo=16, nsteps=1), True),
    ("1x2 (y seam), N=1, nb=1",
     dict(nx=192, ny=192, tile_nx=192, tile_ny=96, halo=16, nsteps=1), True),
    ("2x2 (corners), N=1, nb=1",
     dict(nx=192, ny=192, tile_nx=96, tile_ny=96, halo=16, nsteps=1), True),
    ("2x2, N=3, nb=2",
     dict(nx=192, ny=192, tile_nx=96, tile_ny=96, halo=16, nsteps=3,
          nbuffers=2), True),
    ("3x3, N=1, nb=1",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=1), True),
    ("3x3, N=3, nb=2",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3,
          nbuffers=2), True),
    ("3x3, N=8, nb=2",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=8,
          nbuffers=2), True),
    ("3x3, N=3, nb=3",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3,
          nbuffers=3), True),
    ("3x3, N=3, nb=4",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3,
          nbuffers=4), True),
    ("6x6 (36 tiles, tile == 2*halo), N=1, nb=2",
     dict(nx=192, ny=192, tile_nx=32, tile_ny=32, halo=16, nsteps=1,
          nbuffers=2), True),
    ("6x6, N=3, nb=1",
     dict(nx=192, ny=192, tile_nx=32, tile_ny=32, halo=16, nsteps=3), True),
    ("1x1 (window wraps onto itself), N=3, nb=1",
     dict(nx=192, ny=192, tile_nx=192, tile_ny=192, halo=16, nsteps=3), True),
    ("ragged x (200x192, last tile 8 wide), N=3, nb=2",
     dict(nx=200, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3,
          nbuffers=2), True),
    ("ragged both, odd (173x149, 48x40), N=3, nb=2",
     dict(nx=173, ny=149, tile_nx=48, tile_ny=40, halo=16, nsteps=3,
          nbuffers=2), True),
    ("ragged both, N=8, nb=1",
     dict(nx=173, ny=149, tile_nx=48, tile_ny=40, halo=16, nsteps=8), True),
    ("PINNED HOST STORE, 3x3, N=3, nb=2",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3,
          nbuffers=2, host_store=True), True),
    ("PINNED HOST STORE, ragged, N=8, nb=2",
     dict(nx=200, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=8,
          nbuffers=2, host_store=True), True),
    ("PINNED HOST STORE, 6x6, N=1, nb=1",
     dict(nx=192, ny=192, tile_nx=32, tile_ny=32, halo=16, nsteps=1,
          host_store=True), True),
]

#: Rings that are wrong, and what each one proves.  ``expect`` is whether the
#: run should still come out bit-exact -- and the interesting entry is the one
#: where it does.
RING_NEGATIVES: list[tuple[str, dict, bool, str]] = [
    ("ring_margin='x_only', 3x3, N=1",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=1,
          ring_margin="x_only"), False,
     "forgetting the y bands changes the answer, so the gate can see a "
     "broken ring at all"),
    ("ring_margin='x_only', 1x2 y seam, N=2, nb=1",
     dict(nx=192, ny=192, tile_nx=192, tile_ny=96, halo=16, nsteps=2,
          ring_margin="x_only"), False,
     "and it still sees it on the smallest plan that has a y seam -- but "
     "only at nbuffers=1; at nbuffers=2 the prefetch hides the hazard"),
    ("ring_margin='halo', 3x3, N=1",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=1,
          ring_margin="halo"), True,
     "an UNSOUND ring that is nonetheless bit-exact: the 192 cells it drops "
     "are outside the influence cone. Only rings.assert_ring_covers_reads "
     "catches this, which is why that check exists"),
    ("ring_margin='halo', ragged x, N=2",
     dict(nx=200, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=2,
          ring_margin="halo"), True,
     "same, with 9,344 dropped cells -- the size of the violation says "
     "nothing about whether the hash notices"),
]

#: Configurations whose outcome is a MEASUREMENT of the machine, not a
#: property of the code, so the gate reports them and never fails on them.
#: There is exactly one, and it is the most interesting entry in this file.
RING_OBSERVATIONS: list[tuple[str, dict, str]] = [
    ("ring_ordering='submission' (cross-stream events REMOVED), 3x3, N=3, nb=2",
     dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16, nsteps=3,
          nbuffers=2, ring_ordering="submission"),
     "MEASURED BOTH WAYS, same code, same machine, hours apart: BIT-EXACT on "
     "an idle card, and WRONG (maxabs 3.7e+02) while another process shared "
     "the GPU. The 'one DMA queue drained in submission order' argument is "
     "therefore NOT sufficient -- it is a property of an uncontended card, "
     "and a tiled run that relied on it would corrupt intermittently, "
     "depending on what else the machine was doing. This is why run_tiled "
     "defaults to ring_ordering='events'. Either outcome here is a pass; a "
     "MISMATCH is the more informative one."),
]


def ring_plan_survey(cases=None) -> list[str]:
    """The invariant, and the size, for every plan the gate runs.

    :func:`tilestream.rings.build_ring_plan` already refuses a plan whose
    bands do not cover every later read; this walks the gate's own geometries
    so a plan shape that only the matrix uses cannot slip past.
    """
    from tilestream import rings

    out = []
    for label, kwargs, _expect in (cases or RING_CASES):
        specs = tspec.plan_tiles(kwargs["nx"], kwargs["ny"],
                                 kwargs["tile_nx"], kwargs["tile_ny"],
                                 kwargs["halo"], True)
        plan = rings.build_ring_plan(specs)      # raises unless sound
        rings.assert_ring_covers_reads(plan)
        rep = rings.ring_report(plan)
        out.append(f"{label:46s} {int(rep['tiles']):3d} tiles, "
                   f"{int(rep['bands']):4d} bands, "
                   f"{int(rep['patches']):5d} patch blocks, ring is "
                   f"{100 * rep['ring_fraction']:5.2f}% of a domain "
                   f"(shadow is 100%)")
    return out


def precondition_carrier_identity(rung="full fast cadence", nsteps=4) -> str:
    """Carrier arrays are NOT identity-stable, and the ring path must not care.

    ``PhysicsDriver`` replaces whole tendency bundles when the scheme that
    owns them runs.  A ring arena that resolved its tile-side pointers once
    would then save from, and patch into, memory the model has stopped using
    -- which is a bug this module had, and which showed at exactly one rung
    of fourteen.  So the gate measures that the hazard is still THERE: if
    gpuwm ever stops reallocating, the physics matrix stops covering this and
    somebody has to notice.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    cfg = physics_cfg(rung, 64, 48, NZ)
    state, _drv = physinv.default_builder(cfg, SEED)
    harness.run_steps(state, cfg, 1)
    seen = {k: int(v.data.ptr)
            for k, v in physinv.carrier_inventory(state).items()}
    moved: set[str] = set()
    for _ in range(nsteps):
        harness.run_steps(state, cfg, 1)
        now = {k: int(v.data.ptr)
               for k, v in physinv.carrier_inventory(state).items()}
        moved |= {k for k in now if now[k] != seen.get(k)}
        seen = now
    total = len(seen)
    del state
    cp.get_default_memory_pool().free_all_blocks()
    if not moved:
        raise AssertionError(
            f"no carrier at rung {rung!r} changed its device pointer over "
            f"{nsteps} steps. The ring path resolves tile pointers at issue "
            "time BECAUSE they move; if they no longer move, this gate no "
            "longer covers that and the reason for the design is unrecorded.")
    return (f"{len(moved)} of {total} carriers change device pointer within "
            f"{nsteps} steps at rung {rung!r} "
            f"({sorted(moved)[0]}, ...); the ring arena resolves the tile "
            "side at issue time, so it follows them")


HALO_SWEEP = (8, 10, 11, 12, 13, 14, 16, 17, 20, 24, 32)


def halo_sweep(nx=192, ny=192, tile=64, nsteps=1, widths=HALO_SWEEP) -> list:
    """``[(halo, bitexact, max_abs, worst_field)]`` -- the smallest that works."""
    out = []
    for h in widths:
        if tile + 2 * h > max(nx, ny) * 4:
            continue
        rec = run_case(nx, ny, tile, tile, h, nsteps)
        out.append((h, rec["bitexact"], rec["max_abs"], rec["worst_field"],
                    rec.get("signature", {}).get("verdict")))
    return out


def halo_vs_steps(nx=192, ny=192, tile=64, halos=(11, 12, 13, 14, 15, 16),
                  step_counts=(1, 2, 3, 5, 8)) -> dict:
    """DO NOT TUNE THE HALO ON A ONE-STEP GATE.  This is why.

    A halo narrower than the structural dependency radius does not fail
    immediately.  The cells it truncates contribute less than a float32 ULP at
    first, so a single step comes out bit-exact -- and then the seeded
    perturbation grows, because this is a nonlinear fluid model.  MEASURED on
    192x192x49, 3x3 tiles of 64x64, as ``max|tiled - monolithic|``::

        halo   N=1      N=2      N=3      N=5      N=8    N=12  N=16  N=24
        11     7.8e-3   4.4e-1   4.9e-1   7.3e-1   7.2e-1   --    --    --
        12     exact    2.1e-1   3.7e-1   7.0e-1   7.0e-1   --    --    --
        13     exact    exact    exact    3.8e-6   4.9e-1  7.1e-1 7.3e-1 7.3e-1
        14     exact    exact    exact    exact    exact   exact exact exact
        15     exact    exact    exact    exact    exact   exact exact exact
        16     exact    exact    exact    exact    exact   exact exact exact

    Read the ``halo=13`` row left to right: bit-exact for three steps,
    3.8e-06 at five, 0.49 at eight, saturated at 0.73 by twelve.  That is a
    truncated stencil seeding a perturbation below the float32 rounding
    threshold and the nonlinear dynamics amplifying it.  A gate that only ever
    ran one step would have certified halo 12 -- the value two independent
    earlier probes reported as "the true minimum, so 16 is conservative by 4
    cells" -- and the resulting forecast would be visibly wrong within a
    minute of model time while every one-step test kept passing.  This is the
    most dangerous failure mode the project has: it is silent, it is faster,
    and a short test certifies it.

    So the halo is set from the derivation, not from a measurement:
    ``harness.halo_radius(cfg) = 10 + 3*ns//2``, which is 16 at
    ``time_step_sound=4``.  The measurement's only job is to confirm the
    derived value is sufficient, which it does at every step count tried
    (24 steps, 216 tile-steps, bit-exact).  Note even 14 survives N=24 here --
    which is exactly why an empirical minimum must not be trusted: there is no
    step count at which "it has not failed yet" becomes "it cannot fail".
    """
    table: dict = {}
    for h in halos:
        row = {}
        for n in step_counts:
            rec = run_case(nx, ny, tile, tile, h, n)
            row[n] = (rec["bitexact"], rec["max_abs"])
        table[h] = row
    return table


# --------------------------------------------------------------------------
# PHYSICS: the same gate, with the whole carrier set streamed
# --------------------------------------------------------------------------
#
# Milestone one proved the MECHANISM on nine dry arrays.  This proves the
# PRODUCT: the same tiled loop, streaming the entire restart manifest -- 138
# to 229 arrays including the mp/cu precipitation accumulators, the held
# radiation and PBL tendencies, the 88-to-162 surface/soil/snow fields and
# Kain-Fritsch's lazily-allocated w0avg trigger memory -- plus the scalar
# carriers that decide every physics cadence.
#
# Three things had to change and each is a separate way this could have been
# quietly wrong:
#
#   1. the inventory.  gather.inventory reaches STATE_SERIALIZED_ATTRS by
#      getattr and cannot see state._scratch or state.physics AT ALL, so a
#      physics run streaming it drops 1.79x more carried bytes than it keeps.
#   2. the tile buffer.  harness.make_state attaches no PhysicsDriver, so
#      dycore.step raised at every rung above dry; and two carriers are
#      allocated lazily on first use, so the buffer must have stepped once
#      before its inventory matches the store's.
#   3. the clock.  dycore.step advances elapsed_seconds per CALL, and every
#      due test is a function of it, so a buffer serving k tiles would run
#      k*dt ahead of the domain inside one sweep and tiles would disagree
#      about which schemes are due.

PHYSICS_MOIST = dict(moist=True, mp_physics=10, ztop=20000.0)
PHYSICS_FULL = dict(PHYSICS_MOIST, km_opt=4, sf_sfclay_physics=91,
                    bl_pbl_physics=1, bldt=0.0, sf_surface_physics=2,
                    ra_sw_physics=4, ra_lw_physics=4, radt_minutes=12.0,
                    cu_physics=1, cudt_minutes=5.0)

#: One physics option at a time, then the stacks.  ``full(real74)`` is the
#: selector set of ``configs/real74_d01.toml``.  ``fast cadence`` exists
#: because the long-cadence rungs never cross a radiation/cumulus boundary in
#: a short run, so they cannot see a clock bug: radt 0.05 min fires radiation
#: EVERY step and bldt 1.0 min makes the PBL tendencies genuinely held.
PHYSICS_RUNGS: dict[str, dict] = {
    "dry (control)":     dict(),
    "mp10 Morrison":     dict(PHYSICS_MOIST),
    "+km_opt=2 (TKE)":   dict(PHYSICS_MOIST, km_opt=2),
    "+km_opt=4 (2D Smag)": dict(PHYSICS_MOIST, km_opt=4),
    "+sfclay MM5":       dict(PHYSICS_MOIST, km_opt=4, sf_sfclay_physics=91),
    "+YSU PBL":          dict(PHYSICS_MOIST, km_opt=4, sf_sfclay_physics=91,
                              bl_pbl_physics=1, bldt=0.0),
    "+Noah LSM":         dict(PHYSICS_MOIST, km_opt=4, sf_sfclay_physics=91,
                              bl_pbl_physics=1, bldt=0.0,
                              sf_surface_physics=2),
    "+RRTMGP radiation": dict(PHYSICS_MOIST, km_opt=4, sf_sfclay_physics=91,
                              bl_pbl_physics=1, bldt=0.0,
                              sf_surface_physics=2, ra_sw_physics=4,
                              ra_lw_physics=4, radt_minutes=12.0),
    "full(real74) +KF":  dict(PHYSICS_FULL),
    "full+MYNN":         dict(PHYSICS_FULL, sf_sfclay_physics=5,
                              bl_pbl_physics=5),
    "full+Noah-MP":      dict(PHYSICS_FULL, sf_surface_physics=4),
    "full+MYNN+Noah-MP": dict(PHYSICS_FULL, sf_sfclay_physics=5,
                              bl_pbl_physics=5, sf_surface_physics=4),
    "full+NSSL mp18":    dict(PHYSICS_FULL, mp_physics=18),
    "full fast cadence": dict(PHYSICS_FULL, radt_minutes=0.05,
                              cudt_minutes=0.1, bldt=1.0),
}

#: Domain for the physics matrix.  Non-square so a y/x transposition cannot
#: pass unnoticed, and small enough that 14 rungs x several geometries is a
#: gate rather than a benchmark.  The HALO cases use a bigger domain on
#: purpose -- see :func:`physics_halo_vs_steps`.
PHYS_NX, PHYS_NY = 96, 80
PHYS_WARMUP = 1

_PHYS_REF_CACHE: dict = {}


def physics_cfg(rung: str, nx=PHYS_NX, ny=PHYS_NY, nz=NZ):
    return harness.make_config(nx, ny, nz, **PHYSICS_RUNGS[rung])


def physics_reference(rung, nx, ny, nsteps, nz=NZ, seed=SEED, warmup=PHYS_WARMUP):
    """``(cfg, start_arrays, start_scalars, ref_digests, ref_scalars)``.

    ``warmup`` steps run BEFORE the snapshot for a reason that is not
    cosmetic: Kain-Fritsch allocates ``cumulus/w0avg`` on its first call
    (kf.py:335), so a state that has never stepped has a SHORTER carrier
    manifest than one that has, and a store sized from it would be missing a
    field that later appears.  Warming up also means the comparison starts
    from a state with real physics history rather than from initialisation
    transients.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    key = (rung, nx, ny, nz, nsteps, seed, warmup)
    if key in _PHYS_REF_CACHE:
        return _PHYS_REF_CACHE[key]
    cfg = physics_cfg(rung, nx, ny, nz)
    state, _drv = physinv.default_builder(cfg, seed)
    harness.run_steps(state, cfg, warmup)
    start = {k: _as_numpy(v).copy()
             for k, v in physinv.carrier_inventory(state).items()}
    start_scalars = physinv.carrier_scalars(state)
    harness.run_steps(state, cfg, nsteps)
    ref = {k: _as_numpy(v).copy()
           for k, v in physinv.carrier_inventory(state).items()}
    ref_scalars = physinv.carrier_scalars(state)
    del state
    cp.get_default_memory_pool().free_all_blocks()
    _PHYS_REF_CACHE[key] = (cfg, start, start_scalars, ref, ref_scalars)
    return _PHYS_REF_CACHE[key]


def physics_case(rung, tile_nx, tile_ny, nsteps, *, halo=None, nx=PHYS_NX,
                 ny=PHYS_NY, nz=NZ, nbuffers=2, host_store=False,
                 write_mode="ring", carry_scalars=True, names=None,
                 seed=SEED, graph=None) -> dict:
    """One tiled physics configuration, compared to the monolithic answer.

    ``carry_scalars=False`` and ``names=<subset>`` are the two negative
    controls that only exist at this milestone; everything else mirrors
    :func:`run_case`.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    cfg, start, start_scalars, ref_arrays, ref_scalars = physics_reference(
        rung, nx, ny, nsteps, nz=nz, seed=seed)
    ref = physinv.field_digests(ref_arrays)
    if halo is None:
        halo = harness.halo_radius(cfg)
    specs = tspec.plan_tiles(nx, ny, tile_nx, tile_ny, halo, True)
    tspec.validate_plan(specs, ny, nx)

    if host_store:
        store = {k: gather.pinned_copy(v) for k, v in start.items()}
    else:
        store = {k: cp.asarray(v) for k, v in start.items()}
    scalars = dict(start_scalars) if carry_scalars else None

    report: dict = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver.run_tiled(store, cfg, tile_nx, tile_ny, halo=halo,
                         nsteps=nsteps, nbuffers=nbuffers,
                         write_mode=write_mode, report=report,
                         names=names,
                         inventory_fn=physinv.carrier_inventory,
                         nz=int(cfg.nz),
                         tile_state_factory=driver.make_physics_tile_state,
                         scalars=scalars, **(graph or {}))
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0

    got = physinv.field_digests(physinv.carrier_inventory(store))
    differing = sorted(k for k in ref if ref.get(k) != got.get(k))
    record = {
        "bitexact": not differing,
        "carriers": len(ref),
        "differing": differing,
        "tiles": len(specs),
        "halo": int(halo),
        "seconds": elapsed,
        "scalars_ok": (scalars is None) or (scalars == ref_scalars),
        "compute": report.get("compute"),
        "gathered_bytes": report["gathered_bytes"],
        "scattered_bytes": report["scattered_bytes"],
        # Per GATHERED tile cell, per sweep -- comparable to the inventory
        # agent's 32.3 (dry) / 275.5 (full+MYNN+Noah-MP) B/cell figures.
        "bytes_per_cell": ((report["gathered_bytes"] / len(specs) / nsteps)
                           / float(nz * report["compute"][1]
                                   * report["compute"][2])),
        "graph": report.get("graph"),
    }
    if differing:
        # Localise: which carrier is worst, by how much, and WHERE -- the
        # split that decides the fix.  seam-local => the halo; uniform => a
        # global or non-column-local operation; one tile => ragged/edge
        # handling; ~1e-7 relative => FP reassociation, a different fix
        # entirely.
        worst, max_abs, max_rel, sig = differing[0], 0.0, 0.0, {}
        for key in differing:
            got_a = np.asarray(_as_numpy(store[key]), dtype=np.float64)
            want_a = np.asarray(ref_arrays[key], dtype=np.float64)
            diff = np.abs(got_a - want_a)
            if not diff.size:
                continue
            this_abs = float(diff.max())
            # Relative error only where there is something to be relative TO.
            # Dividing by a floor of finfo.tiny turns "0 vs 1e-30" into inf
            # and destroys the 1e-7-means-reassociation signal, which is the
            # whole reason this number is reported.
            scale = np.maximum(np.abs(want_a), np.abs(got_a))
            live = scale > 0.0
            this_rel = float((diff[live] / scale[live]).max()) if live.any() \
                else 0.0
            if this_abs > max_abs:
                worst, max_abs = key, this_abs
                sig = spatial_signature(diff, specs, int(halo))
            max_rel = max(max_rel, this_rel)
        record.update(worst_field=worst, max_abs=max_abs, max_rel=max_rel,
                      signature=sig)
    del store
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return record


def physics_ring_vs_shadow(rung, tile_nx, tile_ny, nsteps, **kw) -> dict:
    """One physics configuration both ways; every carrier digest must agree.

    Compared carrier by carrier rather than through one hash, so a
    disagreement names the field.
    """
    from tilestream import physics_inventory as physinv

    ring = physics_case(rung, tile_nx, tile_ny, nsteps, write_mode="ring", **kw)
    shadow = physics_case(rung, tile_nx, tile_ny, nsteps, write_mode="shadow",
                          **kw)
    return {
        "bitexact": ring["bitexact"] and shadow["bitexact"],
        # both were compared against the same reference digests, so equal
        # "differing" sets plus both empty is exact agreement
        "agree": ring["differing"] == shadow["differing"],
        "carriers": ring["carriers"],
        "differing": sorted(set(ring["differing"]) ^ set(shadow["differing"])),
        "tiles": ring["tiles"],
        "seconds": ring["seconds"] + shadow["seconds"],
        "ring": ring, "shadow": shadow,
    }


#: Rungs the ring/shadow agreement runs at.  Not all fourteen: the whole
#: matrix already runs on the ring path (``physics_case`` defaults to it), so
#: this is the independent-implementation cross-check, and it is deliberately
#: weighted towards the rungs where the two paths could differ -- the ones
#: whose schemes actually FIRE within the step count.
PHYSICS_RING_RUNGS = ("dry (control)", "mp10 Morrison", "+Noah LSM",
                      "full(real74) +KF", "full+MYNN+Noah-MP",
                      "full+NSSL mp18", "full fast cadence")

PHYSICS_RING_CASES: list[tuple[str, dict, bool]] = (
    [(f"{rung}  |  3x3 ragged, N=3, nb=1",
      dict(rung=rung, tile_nx=40, tile_ny=30, nsteps=3, nbuffers=1), True)
     for rung in PHYSICS_RING_RUNGS]
    + [(f"{rung}  |  2x2, N=3, nb=2",
        dict(rung=rung, tile_nx=48, tile_ny=40, nsteps=3, nbuffers=2), True)
       for rung in PHYSICS_RING_RUNGS]
    + [("full+MYNN+Noah-MP  |  3x3 ragged, N=8, PINNED HOST STORE",
        dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=8,
             host_store=True), True),
       ("full fast cadence  |  3x3 ragged, N=8, nb=1",
        dict(rung="full fast cadence", tile_nx=40, tile_ny=30, nsteps=8,
             nbuffers=1), True)]
)


#: (label, kwargs, expected-to-be-bit-exact).  The escalation the dry gate
#: used, because WHICH rung first fails is the diagnosis: one physics option
#: at a time, then 2x1 seam -> 2x2 corners -> 3x3 ragged, then VRAM store ->
#: pinned host store.
PHYSICS_CASES: list[tuple[str, dict, bool]] = (
    [(f"{rung}  |  2x1 seam, N=1",
      dict(rung=rung, tile_nx=48, tile_ny=80, nsteps=1), True)
     for rung in PHYSICS_RUNGS]
    + [(f"{rung}  |  2x2 corners, N=3",
        dict(rung=rung, tile_nx=48, tile_ny=40, nsteps=3), True)
       for rung in PHYSICS_RUNGS]
    + [(f"{rung}  |  3x3 ragged (40x30), N=8",
        dict(rung=rung, tile_nx=40, tile_ny=30, nsteps=8), True)
       for rung in PHYSICS_RUNGS]
    + [("full+MYNN+Noah-MP  |  3x3 ragged, N=8, PINNED HOST STORE",
        dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=8,
             host_store=True), True),
       ("full fast cadence  |  3x3 ragged, N=8, PINNED HOST STORE",
        dict(rung="full fast cadence", tile_nx=40, tile_ny=30, nsteps=8,
             host_store=True), True),
       ("full+MYNN+Noah-MP  |  1x1 (window wraps onto itself), N=3",
        dict(rung="full+MYNN+Noah-MP", tile_nx=96, tile_ny=80, nsteps=3), True),
       ("full+MYNN+Noah-MP  |  3x3 ragged, N=8, nbuffers=1",
        dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=8,
             nbuffers=1), True),
       ("full+MYNN+Noah-MP  |  3x3 ragged, N=8, nbuffers=4",
        dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=8,
             nbuffers=4), True),
       # --- negative controls: these MUST fail --------------------------
       # MEASURED and worth stating: halo=13 at the 96x80 gate size PASSES,
       # because a 48x40 tile with halo 16 already gathers 80x72 of a 96x80
       # domain -- 83% by 90% of it -- so the halo has almost nothing left to
       # get wrong.  The control has to run at 256x192 with 4x4 tiles, where
       # a gathered tile is ~40% of the domain, and there it differs in 111
       # of 229 carriers.  This is FACT 1 wearing a different hat: the CHEAP
       # test is the one that certifies a wrong halo.
       ("NEGATIVE halo=13 at 256x192 4x4 (one below the measured minimum), "
        "full+MYNN+Noah-MP N=8",
        dict(rung="full+MYNN+Noah-MP", nx=256, ny=192, tile_nx=64, tile_ny=48,
             nsteps=8, halo=13), False),
       ("NEGATIVE stream state/* only (25 of 229), full+MYNN+Noah-MP N=8",
        dict(rung="full+MYNN+Noah-MP", tile_nx=48, tile_ny=40, nsteps=8,
             names=None), False),          # names filled in below
       ("NEGATIVE clock not carried, fast cadence, N=8",
        dict(rung="full fast cadence", tile_nx=48, tile_ny=40, nsteps=8,
             carry_scalars=False), False),
       ("NEGATIVE write_mode=inplace, full(real74) N=8",
        dict(rung="full(real74) +KF", tile_nx=48, tile_ny=40, nsteps=8,
             write_mode="inplace"), False),
       ]
)


def _install_state_only_control() -> None:
    """Fill the ``names`` of the state-only negative control.

    Kept out of the literal because it needs the contract import, and because
    the point of the control deserves saying: this IS milestone one's
    streaming set, run at a physics rung.  It fails, and that failure is the
    reason this milestone exists.
    """
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    names = [f"state/{n}" for n in STATE_SERIALIZED_ATTRS]
    for _label, kwargs, _expect in PHYSICS_CASES:
        if _label.startswith("NEGATIVE stream state/* only"):
            kwargs["names"] = names


_install_state_only_control()


def precondition_physics_inventory(nx=40, ny=32, nz=NZ, nsteps=8) -> list[str]:
    """``assert_streaming_inventory_complete`` must pass at EVERY rung.

    This is the structural guarantee, and it is worth more than the hash
    gate: the hash gate can pass by luck on a short run, this cannot pass
    unless the streamed set really is the whole cross-step state.
    """
    out = []
    for rung in PHYSICS_RUNGS:
        cfg = physics_cfg(rung, nx, ny, nz)
        driver.assert_streaming_inventory_complete(cfg, nsteps=nsteps)
        out.append(f"inventory complete at N={nsteps}: {rung}")
    return out


def precondition_geography(nx=40, ny=32, nz=NZ) -> str:
    """The UNIFORM-geography lane, unchanged: a rebuilt tile must be safe.

    This is milestone two's check and it still has to hold, because the
    ordinary physics matrix below still runs ``map_proj=0``, flat terrain
    and a uniform lat/lon: under those settings a tile REBUILDS its geography
    and that is only harmless while nothing varies.  The guard is the
    statement that nothing does.

    Its negative control is the whole point.  Perturb one cell of the
    radiation latitude grid -- the array a real map projection makes vary --
    and the guard must refuse, because a test tile placed at the domain
    CENTRE certifies the bug: the projection's reference point IS the domain
    centre (projection.py:122-123), so the centred tile is exactly right
    while every other one is displaced.

    What has changed is that this is no longer the END of the story.  The
    GEOGRAPHY section runs the same gate with a real Lambert projection and
    real terrain, where the arrays are GATHERED instead of rebuilt; see
    :func:`precondition_geography_gathered`.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    cfg = physics_cfg("full+MYNN+Noah-MP", nx, ny, nz)
    state, drv = physinv.default_builder(cfg, SEED)
    harness.run_steps(state, cfg, 1)
    physinv.assert_tileable_geography(state, drv)
    report = physinv.geography_report(state, drv)
    n = len(report["setup"]) + len(report["driver"])

    # NEGATIVE CONTROL: perturb one cell of the radiation latitude grid --
    # the array a real map projection makes vary -- and demand a refusal.
    lat = drv.radiation_callable.latitude_deg
    lat[0, 0] = float(lat[0, 0]) + 1.0
    try:
        physinv.assert_tileable_geography(state, drv)
    except physinv.GeographyNotTileable:
        pass
    else:
        raise AssertionError(
            "the geography guard accepted a horizontally VARYING radiation "
            "latitude grid; it would not catch a real map projection")
    outputs = [rec[0] for rec in report["output_only"]]
    del state, drv
    cp.get_default_memory_pool().free_all_blocks()
    return (f"{n} rebuilt horizontal arrays are all uniform (so a tile's "
            f"equals the parent's window), has_msf/rotational both False; "
            f"negative control refused a varying latitude.  NOT carried and "
            f"NOT gathered, hence per-tile under tiling: {outputs} "
            f"(restart.py declares these output-only)")


def precondition_geography_gathered(nx=192, ny=160, tile=64, halo=16, nz=NZ,
                                    rung="full+MYNN+Noah-MP") -> list[str]:
    """A GATHERED tile's geography must equal the parent's window, bitwise.

    The positive half of what :func:`precondition_geography` refuses, and it
    carries three negative controls because the failure it guards against is
    the most flattering kind: a tile centred on the domain reproduces the
    parent EXACTLY, so a one-tile test certifies the bug.

    * BEFORE -- the same buffer with its geography REBUILT from ``tile_cfg``
      -- must differ on every tile, and the reported displacement is the
      headline number.
    * :func:`driver.assert_geography_gathered` must refuse a driver carrying
      legacy RRTMG's ``(ny*nx, 59, 12)`` latitude-interpolated ozone cache,
      and must ACCEPT a uniform one (a guard that refuses everything is not a
      guard).
    * geography must be READ-ONLY across a real 8-step run at this rung, with
      a carrier changing to prove the run was not a no-op.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    out: list[str] = []
    cfg = geography_cfg(rung, nx, ny, nz)
    geo = harness.make_geography(cfg)
    parent, pdrv = harness.make_physics_state(cfg, SEED, geography=geo)
    harness.run_steps(parent, cfg, 1)
    specs = tspec.plan_tiles(nx, ny, tile, tile, halo, True)
    tspec.validate_plan(specs, ny, nx)
    tile_cfg = harness.tile_config(cfg, specs[0].cnx, specs[0].cny)
    keys = sorted(driver.geography_inventory(parent))

    # --- BEFORE: the per-tile rebuild ------------------------------------
    rebuilt = driver.make_physics_tile_state(
        tile_cfg, builder=harness.geography_builder(harness.make_geography))
    bad_tiles, worst_km, worst_lat, worst_f = 0, 0.0, 0.0, 0.0
    plat = np.asarray(_as_numpy(pdrv.radiation_callable.latitude_deg),
                      dtype=np.float64)
    plon = np.asarray(_as_numpy(pdrv.radiation_callable.longitude_deg),
                      dtype=np.float64)
    fmax = float(np.abs(_as_numpy(parent.f)).max())
    for s in specs:
        bad = driver.geography_window_mismatches(rebuilt, parent, s)
        if bad:
            bad_tiles += 1
        worst_f = max(worst_f, bad.get("setup/f", 0.0) / fmax)
        tlat = np.asarray(_as_numpy(
            rebuilt.physics.radiation_callable.latitude_deg), np.float64)
        tlon = np.asarray(_as_numpy(
            rebuilt.physics.radiation_callable.longitude_deg), np.float64)
        wlat, wlon = np.empty_like(tlat), np.empty_like(tlon)
        s.apply_gather(plat, wlat, "mass")
        s.apply_gather(plon, wlon, "mass")
        worst_lat = max(worst_lat, float(np.abs(tlat - wlat).max()))
        worst_km = max(worst_km, _great_circle_km(tlat, tlon, wlat, wlon))
    if bad_tiles != len(specs):
        raise AssertionError(
            f"the rebuild control did not fire: only {bad_tiles} of "
            f"{len(specs)} tiles differ from the parent window")
    out.append(f"NEGATIVE CONTROL rebuild: {bad_tiles}/{len(specs)} tiles "
               f"differ in all {len(keys)} geography arrays -- worst tile "
               f"{worst_lat:.4f} deg of latitude, {worst_km:.1f} km great "
               f"circle, {100 * worst_f:.2f}% relative in Coriolis")

    # THE ASYMMETRY THAT LETS THIS SLIP THROUGH.  On a SQUARE domain the
    # middle tile of a 3x3 plan is exactly centred (offset_x = ci0 +
    # (cnx+1)/2 - (nx+1)/2 = 48 + 48.5 - 96.5 = 0), and a rebuilt tile
    # there reproduces the parent BIT FOR BIT.  A one-tile test placed at
    # the centre therefore CERTIFIES the bug.
    # ``periodic_faces=False`` here on purpose: the alias duplicate is a
    # property of PERIODICITY, not of the projection, and it makes even the
    # centred tile differ in msfv (its own last v-face row is overwritten
    # with its own first, which is not the parent's value at that row).
    # With the raw Lambert values the centred tile is exactly right, which
    # is the phenomenon this control is about.
    sq_n, sq_tile = 3 * (halo * 2), halo * 2
    raw = lambda c: harness.make_geography(c, periodic_faces=False)  # noqa: E731
    sq_specs = tspec.plan_tiles(sq_n, sq_n, sq_tile, sq_tile, halo, True)
    sq_cfg = geography_cfg(rung, sq_n, sq_n, nz)
    sq_parent, _ = harness.make_physics_state(sq_cfg, SEED,
                                              geography=raw(sq_cfg))
    sq_tile_cfg = harness.tile_config(sq_cfg, sq_specs[0].cnx,
                                      sq_specs[0].cny)
    sq_rebuilt = driver.make_physics_tile_state(
        sq_tile_cfg, builder=harness.geography_builder(raw))
    centred = [s for s in sq_specs
               if s.ci0 + (s.cnx + 1) / 2 == (sq_n + 1) / 2
               and s.cj0 + (s.cny + 1) / 2 == (sq_n + 1) / 2]
    if not centred:
        raise AssertionError("no exactly-centred tile in the square plan; "
                             "the asymmetry control cannot run")
    off = [len(driver.geography_window_mismatches(sq_rebuilt, sq_parent, s))
           for s in sq_specs]
    n_centre = len(driver.geography_window_mismatches(sq_rebuilt, sq_parent,
                                                     centred[0]))
    if n_centre != 0 or sorted(off)[-1] == 0:
        raise AssertionError(
            f"the centred-tile asymmetry did not reproduce: centre differs "
            f"in {n_centre} arrays, worst tile in {max(off)}")
    out.append(f"ASYMMETRY: on a square {sq_n}x{sq_n} domain the "
               f"exactly-centred "
               f"tile (ci0={centred[0].ci0}, cj0={centred[0].cj0}) is "
               f"BIT-EXACT when REBUILT -- 0 of {len(keys)} arrays differ -- "
               f"while the other {len(sq_specs) - 1} differ in up to "
               f"{max(off)}.  A one-tile test placed at the centre certifies "
               f"the bug; that is why every tile is checked")
    del sq_parent, sq_rebuilt
    cp.get_default_memory_pool().free_all_blocks()

    # --- AFTER: gathered --------------------------------------------------
    buf = driver.make_physics_tile_state(
        tile_cfg, builder=harness.geography_builder())
    driver._pin_scheme_geography(buf)
    store = driver.geography_store(parent, host=True)
    stream = cp.cuda.Stream(non_blocking=False)
    nbytes = 0
    for s in specs:
        nbytes = gather.gather_tile(
            store, buf, s, stream, inventory_fn=driver.geography_inventory,
            nz=nz).nbytes
        stream.synchronize()
        bad = driver.geography_window_mismatches(buf, parent, s)
        if bad:
            raise AssertionError(
                f"gathered tile at ci0={s.ci0} cj0={s.cj0} still differs "
                f"from the parent window: {bad}")
    scalars = driver.setup_scalar_mismatches(buf, parent)
    if scalars:
        raise AssertionError(f"setup SCALARS differ tile vs parent: {scalars}")
    cells = specs[0].cnx * specs[0].cny * nz
    out.append(f"GATHERED: all {len(keys)} geography arrays equal the parent "
               f"window BITWISE on every one of {len(specs)} tiles "
               f"(corners and ragged included), {nbytes / 1e6:.3f} MB = "
               f"{nbytes / cells:.3f} B per gathered cell; setup scalars "
               f"agree")

    # --- the guard --------------------------------------------------------
    driver.assert_geography_gathered(parent)
    from gpuwm.ingest import wrf_ozone
    climo = wrf_ozone.load_ozone_climatology()
    flat_lat = np.asarray(geo.lat, dtype=np.float32).reshape(-1)
    varying = wrf_ozone.interp_ozone_to_latitudes(flat_lat, climo)
    uniform = wrf_ozone.interp_ozone_to_latitudes(
        np.full_like(flat_lat, 35.0), climo)
    pdrv.radiation_callable._ozone_lat_interp = varying
    try:
        driver.assert_geography_gathered(parent)
    except driver.GeographyNotGatherable:
        pass
    else:
        raise AssertionError(
            "assert_geography_gathered accepted a latitude-interpolated "
            "ozone cache; it would not catch legacy RRTMG")
    pdrv.radiation_callable._ozone_lat_interp = uniform
    driver.assert_geography_gathered(parent)
    del pdrv.radiation_callable._ozone_lat_interp
    # ... and it must refuse a gather set that leaves a scheme's lat/lon out,
    # which is the mistake a reasonable implementation makes (those two grids
    # are neither on the state nor in the carrier manifest).
    setup_only = [k for k in keys if k.startswith("setup/")]
    try:
        driver.assert_geography_gathered(parent, keys=setup_only)
    except driver.GeographyNotGatherable:
        pass
    else:
        raise AssertionError(
            "assert_geography_gathered accepted a gather set with no scheme "
            "latitude/longitude in it")
    out.append(f"NEGATIVE CONTROL guard: a {tuple(varying.shape)} "
               f"latitude-interpolated ozone cache "
               f"({varying.nbytes / (nx * ny * nz):.1f} B/mass-cell, the "
               f"legacy-RRTMG layout) is REFUSED -- horizontal axes are the "
               f"LEADING axis, so no gather and no halo reaches it; the "
               f"uniform-latitude version is accepted; and a gather set of "
               f"the {len(setup_only)} setup arrays alone is REFUSED for "
               f"leaving the scheme lat/lon behind")

    # --- read-only --------------------------------------------------------
    before = physinv.field_digests(driver.geography_inventory(parent))
    thp_before = hashlib.sha256(
        _as_numpy(parent.thp).tobytes()).hexdigest()
    harness.run_steps(parent, cfg, 8)
    after = physinv.field_digests(driver.geography_inventory(parent))
    thp_after = hashlib.sha256(_as_numpy(parent.thp).tobytes()).hexdigest()
    changed = sorted(k for k in before if before[k] != after[k])
    if thp_before == thp_after:
        raise AssertionError("the read-only control is vacuous: 8 steps did "
                             "not change state.thp")
    if changed:
        raise AssertionError(f"geography CHANGED over 8 steps: {changed}")
    out.append(f"READ-ONLY: all {len(keys)} geography arrays are "
               f"bit-identical after 8 steps at {rung!r} (state.thp changed, "
               f"so the run was not a no-op) -- hence gathered ONCE per "
               f"buffer occupancy, never streamed per step")

    del parent, pdrv, rebuilt, buf, store
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return out


def _great_circle_km(lat_a, lon_a, lat_b, lon_b) -> float:
    """Worst-cell great-circle separation, on the WPS sphere.

    ``EARTH_RADIUS_M = 6370000`` (gpuwm/static/projection.py:50) -- the same
    sphere the projection itself uses, so the number is internally
    consistent rather than a WGS84 approximation of a spherical grid.
    """
    from gpuwm.static.projection import EARTH_RADIUS_M

    p1, p2 = np.deg2rad(lat_a), np.deg2rad(lat_b)
    dlat, dlon = p2 - p1, np.deg2rad(lon_b) - np.deg2rad(lon_a)
    h = (np.sin(dlat / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2)
    return float((2.0 * EARTH_RADIUS_M
                  * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))).max() / 1000.0)


def precondition_carrier_hoststore(rung="full+MYNN+Noah-MP", nx=96, ny=80,
                                   nz=NZ) -> str:
    """A pinned :class:`HostDomainStore` must hold the whole carrier manifest.

    The plain ``{name: pinned array}`` store the matrix uses proves the
    TRANSPORT.  This proves the STORE: the same capacity guards, the same raw
    ``cudaHostAlloc`` blocks and the same pinning assertion as milestone one,
    now sized from shape rules read off a live physics state -- including the
    layered soil and snow fields, whose leading axis is not vertical and
    which therefore need their own ``FieldSpec.layers`` rule rather than an
    abused ``has_z``.
    """
    import cupy as cp

    from tilestream import hoststore
    from tilestream import physics_inventory as physinv

    cfg = physics_cfg(rung, nx, ny, nz)
    state, _drv = physinv.default_builder(cfg, SEED)
    harness.run_steps(state, cfg, PHYS_WARMUP)
    inv = physinv.carrier_inventory(state)
    manifest = hoststore.manifest_from_arrays(inv, cfg.nz, cfg.ny, cfg.nx)
    store = hoststore.HostDomainStore(cfg, manifest=manifest,
                                      inventory_fn=physinv.carrier_inventory)
    store.assert_pinned()
    store.fill_from(state)
    want = physinv.field_digests(physinv.carrier_manifest(state))
    got = physinv.field_digests(store.arrays)
    if want != got:
        bad = sorted(k for k in want if want[k] != got.get(k))
        raise AssertionError(f"carrier store round trip changed {bad[:8]}")
    layered = sorted(s.name for s in manifest if s.layers is not None)
    summary = (f"pinned carrier store round trip is bit-exact: "
               f"{len(manifest)} fields, {store.nbytes / 1e6:.1f} MB, "
               f"{store.bytes_per_cell:.1f} B/cell, "
               f"{len(layered)} layered fields sized by their own rule "
               f"({layered[:3]}...)")
    store.free()
    del state
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return summary


def physics_halo_vs_steps(rung="full+MYNN+Noah-MP", nx=256, ny=192,
                          tile_nx=64, tile_ny=48,
                          halos=(12, 13, 14, 15, 16, 17),
                          step_counts=(1, 2, 3, 5, 8)) -> dict:
    """FACT 1 re-measured WITH PHYSICS ON.

    The domain and tile are chosen so the halo actually matters: 4x4 tiles of
    64x48 means a gathered tile spans 88x72 to 98x82 of a 256x192 domain --
    38% to 43% of it.  At the 96x80 gate size a tile with halo 16 already
    covers 83% x 90% of the domain, so a too-narrow halo can pass there.
    That is the same trap as FACT 1 itself: the test that certifies a wrong
    halo is the CHEAP one.
    """
    from tilestream import physics_inventory as physinv

    table: dict[int, dict[int, tuple[bool, int]]] = {}
    for halo in halos:
        row: dict[int, tuple[bool, int]] = {}
        for nsteps in step_counts:
            rec = physics_case(rung, tile_nx, tile_ny, nsteps, halo=halo,
                               nx=nx, ny=ny)
            row[nsteps] = (rec["bitexact"], len(rec["differing"]))
        table[halo] = row
    del physinv
    return table


# --------------------------------------------------------------------------
# GEOGRAPHY: the same gate on a REAL map projection
# --------------------------------------------------------------------------
#
# Everything above runs map_proj=0, flat terrain and a uniform lat/lon, where
# a tile can REBUILD its geography because none of it varies.  This section
# removes that restriction: a real Lambert conformal grid at
# configs/real74_d01.toml's own dx=dy=12 km, a lat/lon-anchored terrain, and
# per-column latitude/longitude handed to every scheme -- so has_msf and
# rotational are BOTH TRUE, the Coriolis+curvature kernel and every
# msf-weighted path are live, thb/pb/alb/phb are 3-D and terrain-following,
# and the solar-zenith path reads a latitude that actually varies.
#
# The geography is GATHERED, not rebuilt.  17 arrays at the Noah-MP rung --
# 13 STATE_SETUP_ARRAYS with a horizontal extent plus the four scheme lat/lon
# grids, which are NOT on the state and NOT in the carrier manifest, so
# nothing else in this pipeline would move them.
#
# WHY A ONE-TILE TEST WOULD CERTIFY THE BUG, which is why this section is
# built around asymmetry: projection.py:122-123 defaults the reference point
# to the domain centre, so the tile centred on the domain is BIT-EXACT while
# every other one is displaced by ci0 + (cnx+1)/2 - (nx+1)/2 cells per axis.
#
# READ THIS BEFORE USING THIS SECTION TO CERTIFY ANYTHING ELSE.  dx here is
# 12 km, not the 500 m of the sections above, and MEASURED that makes it a
# much WEAKER halo test: at 256x192 with 4x4 tiles of 64x48 and N=8 the
# smallest bit-exact halo is 8, where the same geometry at dx=500 m needs 14.
# The per-cell gradients are 24x smaller, so a truncated halo falls below a
# float32 ULP.  The halo control here is therefore halo=7, and the halo
# itself still comes from harness.halo_radius(cfg) = 16.

GEO_NX, GEO_NY = 96, 80

#: Rungs for the geography matrix.  Fewer than the physics matrix on
#: purpose -- this section varies the GEOGRAPHY, not the scheme list -- but
#: it keeps the two that decide different things: ``full+MYNN+Noah-MP`` is
#: the 229-carrier rung and the only one with a Noah-MP lat/lon grid, and
#: ``full fast cadence`` is the ONLY rung that calls radiation during the
#: compared steps.  MEASURED at 96x80x49, 8 steps of 3 s: radt=12 min fires
#: radiation 0 times, radt=0.05 min fires it 8 times.  Without the fast rung
#: the solar-zenith path is exercised only by the warmup step, i.e. before
#: the comparison starts.
GEO_RUNGS: tuple[str, ...] = (
    "dry (control)",
    "mp10 Morrison",
    "+RRTMGP radiation",
    "full(real74) +KF",
    "full+MYNN+Noah-MP",
    "full fast cadence",
)

_GEO_REF_CACHE: dict = {}


def geography_cfg(rung: str, nx=GEO_NX, ny=GEO_NY, nz=NZ):
    """``physics_cfg`` plus ``map_proj=1``, ``terrain_opt=1``, dx=dy=12 km."""
    from dataclasses import replace

    return replace(physics_cfg(rung, nx, ny, nz),
                   **harness.GEOGRAPHY_OVERRIDES)


def geography_reference(rung, nx, ny, nsteps, nz=NZ, seed=SEED,
                        warmup=PHYS_WARMUP, periodic_faces=True):
    """The monolithic answer on a real projection, plus the geography store.

    ``periodic_faces`` is threaded through to :func:`harness.make_geography`
    because it is a negative control rather than a detail: under
    ``periodic=True`` a tile's gather never reads a staggered array's alias
    slot (spec.py's ``_axis_gather`` reduces mod nx), so the domain is only
    self-consistent when that slot duplicates the opposite face.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    key = (rung, nx, ny, nz, nsteps, seed, warmup, periodic_faces)
    if key in _GEO_REF_CACHE:
        return _GEO_REF_CACHE[key]
    cfg = geography_cfg(rung, nx, ny, nz)
    geo = harness.make_geography(cfg, periodic_faces=periodic_faces)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo)
    harness.run_steps(state, cfg, warmup)
    start = {k: _as_numpy(v).copy()
             for k, v in physinv.carrier_inventory(state).items()}
    start_scalars = physinv.carrier_scalars(state)
    geo_start = {k: _as_numpy(v).copy()
                 for k, v in driver.geography_inventory(state).items()}
    harness.run_steps(state, cfg, nsteps)
    ref = physinv.field_digests(physinv.carrier_inventory(state))
    ref_arrays = {k: _as_numpy(v).copy()
                  for k, v in physinv.carrier_inventory(state).items()}
    ref_scalars = physinv.carrier_scalars(state)
    del state, _drv
    cp.get_default_memory_pool().free_all_blocks()
    _GEO_REF_CACHE[key] = (cfg, start, start_scalars, geo_start, ref,
                           ref_arrays, ref_scalars)
    return _GEO_REF_CACHE[key]


def geography_case(rung, tile_nx, tile_ny, nsteps, *, halo=None, nx=GEO_NX,
                   ny=GEO_NY, nz=NZ, nbuffers=2, host_store=False,
                   geo_host=True, write_mode="shadow", gather_geography=True,
                   rebuild=False, impose_flags=True, geography_names=None,
                   check_geography=True, periodic_faces=True,
                   seed=SEED, graph=None) -> dict:
    """One tiled configuration on a real projection, versus monolithic.

    The five keywords that only exist here are the negative controls:
    ``gather_geography=False`` (with ``rebuild=True``) reproduces today's
    behaviour exactly, ``impose_flags=False`` leaves ``has_msf``/
    ``rotational`` at the buffer's own, ``geography_names=<subset>``
    gathers only part of the geography, and ``periodic_faces=False`` breaks
    the staggered alias rule.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    cfg, start, start_scalars, geo_start, ref, ref_arrays, ref_scalars = \
        geography_reference(rung, nx, ny, nsteps, nz=nz, seed=seed,
                            periodic_faces=periodic_faces)
    if halo is None:
        halo = harness.halo_radius(cfg)
    specs = tspec.plan_tiles(nx, ny, tile_nx, tile_ny, halo, True)
    tspec.validate_plan(specs, ny, nx)

    if host_store:
        store = {k: gather.pinned_copy(v) for k, v in start.items()}
    else:
        store = {k: cp.asarray(v) for k, v in start.items()}
    geo_store = ({k: gather.pinned_copy(v) for k, v in geo_start.items()}
                 if geo_host else
                 {k: cp.asarray(v) for k, v in geo_start.items()})
    scalars = dict(start_scalars)
    build = (harness.make_geography if rebuild else harness.neutral_geography)

    kwargs = driver.geography_run_kwargs(cfg, None, geography=geo_store,
                                         geography_fn=build)
    kwargs["scalars"] = scalars
    kwargs["impose_geography_flags"] = impose_flags
    kwargs["check_geography"] = check_geography
    if geography_names is not None:
        kwargs["geography_names"] = list(geography_names)
    if not gather_geography:
        kwargs.pop("geography")

    report: dict = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver.run_tiled(store, cfg, tile_nx, tile_ny, halo=halo,
                         nsteps=nsteps, nbuffers=nbuffers,
                         write_mode=write_mode, report=report, **kwargs,
                         **(graph or {}))
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0

    got = physinv.field_digests(physinv.carrier_inventory(store))
    differing = sorted(k for k in ref if ref.get(k) != got.get(k))
    # Geography is INPUT.  Nothing may have written it back.
    geo_now = physinv.field_digests(geo_store)
    geo_was = physinv.field_digests(geo_start)
    geo_touched = sorted(k for k in geo_was if geo_was[k] != geo_now.get(k))

    record = {
        "bitexact": not differing,
        "carriers": len(ref),
        "differing": differing,
        "tiles": len(specs),
        "halo": int(halo),
        "seconds": elapsed,
        "scalars_ok": scalars == ref_scalars,
        "geo_readonly": not geo_touched,
        "geo_touched": geo_touched,
        "compute": report.get("compute"),
        "graph": report.get("graph"),
        "geo_fields": report.get("geography_fields", 0),
        "geo_gathers": report.get("geography_gathers", 0),
        "geo_bytes": report.get("geography_bytes", 0),
        "geo_fraction": report.get("geography_over_carrier", 0.0),
        "gathered_bytes": report["gathered_bytes"],
        "scattered_bytes": report["scattered_bytes"],
    }
    if differing:
        worst, max_abs, max_rel, sig = differing[0], 0.0, 0.0, {}
        for key in differing:
            got_a = np.asarray(_as_numpy(store[key]), dtype=np.float64)
            want_a = np.asarray(ref_arrays[key], dtype=np.float64)
            diff = np.abs(got_a - want_a)
            if not diff.size:
                continue
            this_abs = float(diff.max())
            scale = np.maximum(np.abs(want_a), np.abs(got_a))
            live = scale > 0.0
            max_rel = max(max_rel, float((diff[live] / scale[live]).max())
                          if live.any() else 0.0)
            if this_abs > max_abs:
                worst, max_abs = key, this_abs
                sig = spatial_signature(diff, specs, int(halo))
        record.update(worst_field=worst, max_abs=max_abs, max_rel=max_rel,
                      signature=sig)
    del store, geo_store
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return record


#: (label, kwargs, expected-to-be-bit-exact).
GEOGRAPHY_CASES: list[tuple[str, dict, bool]] = (
    [(f"{rung}  |  2x1 seam, N=1",
      dict(rung=rung, tile_nx=48, tile_ny=80, nsteps=1), True)
     for rung in GEO_RUNGS]
    + [(f"{rung}  |  2x2 corners, N=3",
        dict(rung=rung, tile_nx=48, tile_ny=40, nsteps=3), True)
       for rung in GEO_RUNGS]
    + [(f"{rung}  |  3x3 ragged (40x30), N=8",
        dict(rung=rung, tile_nx=40, tile_ny=30, nsteps=8), True)
       for rung in GEO_RUNGS]
    + [("full+MYNN+Noah-MP  |  3x3 ragged, N=8, PINNED HOST STORE",
        dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=8,
             host_store=True), True),
       ("full+MYNN+Noah-MP  |  3x3 ragged, N=8, geography store in VRAM",
        dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=8,
             geo_host=False), True),
       ("full+MYNN+Noah-MP  |  1x1 (window wraps onto itself), N=3",
        dict(rung="full+MYNN+Noah-MP", tile_nx=96, tile_ny=80, nsteps=3),
        True),
       ("full+MYNN+Noah-MP  |  3x3 ragged, N=8, nbuffers=1",
        dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=8,
             nbuffers=1), True),
       # nbuffers >= ntiles is the "gather geography ONCE" case: every buffer
       # serves one tile for the whole run, so geography_gathers == tiles.
       ("full+MYNN+Noah-MP  |  2x2, N=8, nbuffers=4 (one buffer per tile)",
        dict(rung="full+MYNN+Noah-MP", tile_nx=48, tile_ny=40, nsteps=8,
             nbuffers=4), True),
       ("full+MYNN+Noah-MP  |  3x3 ragged, N=3, write_mode=ring",
        dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=3,
             write_mode="ring"), True),
       # --- negative controls: these MUST fail --------------------------
       ("NEGATIVE geography REBUILT per tile (today's behaviour), "
        "full+MYNN+Noah-MP N=1",
        dict(rung="full+MYNN+Noah-MP", tile_nx=48, tile_ny=40, nsteps=1,
             gather_geography=False, rebuild=True), False),
       ("NEGATIVE geography REBUILT per tile, fast cadence N=8 "
        "(radiation every step)",
        dict(rung="full fast cadence", tile_nx=48, tile_ny=40, nsteps=8,
             gather_geography=False, rebuild=True), False),
       ("NEGATIVE has_msf/rotational taken from the TILE's window, "
        "full+MYNN+Noah-MP N=1",
        dict(rung="full+MYNN+Noah-MP", tile_nx=48, tile_ny=40, nsteps=1,
             impose_flags=False), False),
       ("NEGATIVE setup/* gathered but the SCHEME lat/lon rebuilt, "
        "fast cadence N=8",
        dict(rung="full fast cadence", tile_nx=48, tile_ny=40, nsteps=8,
             geography_names=None), False),   # names filled in below
       ("NEGATIVE scheme lat/lon gathered but setup/* rebuilt, "
        "fast cadence N=8",
        dict(rung="full fast cadence", tile_nx=48, tile_ny=40, nsteps=8,
             geography_names=None), False),   # names filled in below
       ("NEGATIVE periodic_faces=False (msfv[ny,:] != msfv[0,:]), "
        "full+MYNN+Noah-MP N=1",
        dict(rung="full+MYNN+Noah-MP", tile_nx=48, tile_ny=40, nsteps=1,
             periodic_faces=False), False),
       ("NEGATIVE halo=7 at 256x192 4x4 (one below the measured minimum "
        "of 8 AT dx=12 km), full+MYNN+Noah-MP N=8",
        dict(rung="full+MYNN+Noah-MP", nx=256, ny=192, tile_nx=64,
             tile_ny=48, nsteps=8, halo=7), False),
       ]
)

#: The geography inventory at the fast-cadence rung, split in two.  Noah (not
#: Noah-MP) at that rung, so there is no ``noahmp/*`` pair.
_GEO_SETUP_KEYS = tuple(f"setup/{n}" for n in (
    "thb", "pb", "alb", "phb", "mub2d", "ht", "msft", "msfu", "msfv",
    "f", "e", "sina", "cosa"))
_GEO_SCHEME_KEYS = ("radiation/latitude_deg", "radiation/longitude_deg",
                    "noahmp/latitude_deg", "noahmp/longitude_deg")


def _install_partial_geography_controls() -> None:
    """Fill the ``geography_names`` of the two partial-gather controls.

    They matter more than they look.  The scheme lat/lon grids are the only
    geography that is neither on the ``DomainState`` nor in the carrier
    manifest, so nothing else in this pipeline would ever move them -- and
    they are what every solar-zenith path in the tree reads.  Gathering the
    setup arrays and leaving them behind is the mistake a reasonable
    implementation makes, so it gets its own row.
    """
    for label, kwargs, _expect in GEOGRAPHY_CASES:
        if "SCHEME lat/lon rebuilt" in label:
            kwargs["geography_names"] = list(_GEO_SETUP_KEYS)
            # run_tiled REFUSES this subset -- assert_geography_gathered
            # requires every scheme lat/lon to be in the gathered set.  The
            # refusal is proved in precondition_geography_gathered; here the
            # guard is switched off so the row can measure the DAMAGE the
            # guard prevents.
            kwargs["check_geography"] = False
        elif "setup/* rebuilt" in label:
            kwargs["geography_names"] = list(_GEO_SCHEME_KEYS)


_install_partial_geography_controls()


def geography_halo_vs_steps(rung="full+MYNN+Noah-MP", nx=256, ny=192,
                            tile_nx=64, tile_ny=48,
                            halos=(4, 6, 7, 8, 10, 12, 14, 16),
                            step_counts=(1, 3, 8)) -> dict:
    """FACT 1 again, and this time it points the OTHER way.

    MEASURED at dx=12 km, 256x192x49, 4x4 tiles of 64x48, full+MYNN+Noah-MP:
    halo 8 is bit-exact at N=8 while the SAME geometry at dx=500 m needs 14.
    The dependency radius has not changed -- it is structural, set by the
    stencil -- but the amplitude of what a short halo truncates has, by the
    24x in dx.  So a coarse-grid gate certifies a halo that a fine-grid run
    refutes, which is the same trap as FACT 1 with the sign flipped.
    Take the halo from ``harness.halo_radius(cfg)``; do not read it off this
    table.
    """
    table: dict[int, dict[int, tuple[bool, int]]] = {}
    for halo in halos:
        row: dict[int, tuple[bool, int]] = {}
        for nsteps in step_counts:
            rec = geography_case(rung, tile_nx, tile_ny, nsteps, halo=halo,
                                 nx=nx, ny=ny)
            row[nsteps] = (rec["bitexact"], len(rec["differing"]))
        table[halo] = row
    return table


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# milestone four: the same gate with the step REPLAYED FROM A CUDA GRAPH
# --------------------------------------------------------------------------
#
# ``--graph-only`` runs it.  A tiled sweep re-issues the step's ~1,200 kernel
# launches once per TILE, so the launch overhead -- 26% of the step's wall in
# ArWen's own profile -- is multiplied by the tile count and is worst on the
# small compute windows a small card is forced into.  Capturing a tile's step
# once and replaying it for the other tiles removes that multiplication.
#
# It also introduces four new ways to be silently wrong, and this section
# exists because the first three of them produce a perfectly plausible
# forecast:
#
#   * the graph is captured EMPTY (capture without making the stream current)
#     and every replay does nothing.  Caught by graphcap's node-count floor,
#     not here, because an empty step does not merely differ -- it does not
#     move at all, which the digest would also show.
#   * the graph is replayed under a DIFFERENT cadence than it was captured
#     under, so radiation or cumulus silently runs when it should not, or
#     does not when it should.  ``graph_key="none"`` is that failure on
#     purpose and it MUST be caught.
#   * the replay skips the step's HOST bookkeeping, so the buffer's clock and
#     call counters stop advancing.  ``graph_scalars=False`` is that failure
#     on purpose.
#   * two buffers replay concurrently into the same captured scratch
#     addresses.  That one is prevented structurally -- one private memory
#     pool per buffer -- and every physics case below runs ``nbuffers=2``,
#     which is what puts two graphs on two streams at the same time.
#
# The positives are deliberately the SAME cases the earlier sections run, with
# one keyword added.  A graph that changes any answer changes a digest that
# already has a reference.
#
# THIS SECTION IS CARD-DEPENDENT AND SAYS SO.  Every capture grows a private
# memory pool inside an active capture, so whether the section runs is a fact
# about the card and its driver, not only about capacity.  Measured, both
# idle and both 16 GB: an RTX 4080 runs the whole gate 233 checks passed / 0
# failed, and an RTX 5080 (sm_120, CUDA 13.1, cupy 14.1.1) refuses capture
# allocations with 15.2 GiB free, one or two of the three negative controls
# per run.  A row the card would not serve reports as MACHINE-LIMITED: not a
# pass, not a fail, counted separately, and named in the verdict as a
# coverage hole.  See :func:`capture_would_not_fit` for why neither other
# outcome is honest.

#: (label, kwargs, expect-bit-exact).  ``kind`` selects which case runner.
GRAPH_CASES: list[tuple[str, dict, bool]] = [
    ("dry, 2x2, N=3, graph replay",
     dict(kind="dry", nx=96, ny=96, tile_nx=48, tile_ny=48, halo=16, nsteps=3,
          nbuffers=2), True),
    ("dry, 3x3 ragged (40x30), N=8, graph replay",
     dict(kind="dry", nx=120, ny=90, tile_nx=40, tile_ny=30, halo=16,
          nsteps=8, nbuffers=2), True),
    ("mp10 Morrison  |  2x2, N=3, graph replay",
     dict(kind="physics", rung="mp10 Morrison", tile_nx=48, tile_ny=40,
          nsteps=3), True),
    ("+YSU PBL  |  2x2, N=3, graph replay",
     dict(kind="physics", rung="+YSU PBL", tile_nx=48, tile_ny=40,
          nsteps=3), True),
    ("full(real74) +KF  |  2x2, N=3, graph replay  (THE SHIP CONFIG)",
     dict(kind="physics", rung="full(real74) +KF", tile_nx=48, tile_ny=40,
          nsteps=3), True),
    ("full fast cadence  |  2x2, N=3, graph replay  (radiation EVERY step, "
     "cumulus every 2nd)",
     dict(kind="physics", rung="full fast cadence", tile_nx=48, tile_ny=40,
          nsteps=3), True),
    ("full fast cadence  |  3x3 ragged (40x30), N=8, graph replay",
     dict(kind="physics", rung="full fast cadence", tile_nx=40, tile_ny=30,
          nsteps=8), True),
    ("full fast cadence  |  2x2, N=3, graph replay + verify_topology "
     "(recaptures every step and demands the cached graph still describes it)",
     dict(kind="physics", rung="full fast cadence", tile_nx=48, tile_ny=40,
          nsteps=3, graph_verify_topology=True), True),
    ("full(real74) +KF  |  REAL Lambert projection, 2x2, N=3, graph replay",
     dict(kind="geography", rung="full(real74) +KF", tile_nx=48, tile_ny=40,
          nsteps=3), True),
    ("full+MYNN+Noah-MP  |  2x2, N=3, graph AUTO -> must FALL BACK and still "
     "be exact (MYNN and Noah-MP still branch on device data)",
     dict(kind="physics", rung="full+MYNN+Noah-MP", tile_nx=48, tile_ny=40,
          nsteps=3, use_graph=True, expect_fallback=True), True),
]

#: The two ways a correct-looking graph replays the wrong work.  Both MUST be
#: detected -- as a digest mismatch, or as the driver's own clock-agreement
#: check raising, which is also a detection.
GRAPH_NEGATIVES: list[tuple[str, dict, str]] = [
    ("graph_reuse='run': one graph per TOPOLOGY, re-used across steps",
     dict(kind="physics", rung="full fast cadence", tile_nx=48, tile_ny=40,
          nsteps=3, graph_reuse="run"),
     "a graph bakes in every kernel's scalar arguments, and RRTMGP takes the "
     "solar hour angle as one (rrtmgp.py:2045, :2201); replaying a "
     "radiation-due step at a later clock re-uses the earlier step's sun.  "
     "THIS is why the default keys on the sweep, and it is a measurement "
     "rather than a worry"),
    ("graph_key='none' AND graph_reuse='run': the cadence ignored as well",
     dict(kind="physics", rung="full fast cadence", tile_nx=48, tile_ny=40,
          nsteps=3, graph_key="none", graph_reuse="run"),
     "cumulus is due on alternate steps at cudt=0.1 min, so one graph for "
     "the whole run also replays a cumulus step's kernels on a "
     "non-cumulus step.  Note the cadence key is only load-bearing under "
     "reuse='run': under the default the sweep index already separates "
     "every step, and this control is what shows the difference"),
    ("graph_scalars=False: the replay does not re-apply the step's scalar "
     "carrier increment",
     dict(kind="physics", rung="full fast cadence", tile_nx=48, tile_ny=40,
          nsteps=3, graph_scalars=False),
     "the buffer clock stops advancing, so every later sweep evaluates the "
     "cadence at the wrong itimestep"),
]


def graph_case(kind="physics", *, use_graph="require", graph_reuse="sweep",  # noqa: E501
               graph_key="cadence", graph_scalars=True,
               graph_verify_topology=False, expect_fallback=False,
               **kwargs) -> dict:
    """One case from an earlier section, with the step replayed from a graph.

    ``use_graph="require"`` by default: a case that silently fell back to
    stream launching would pass this section while testing nothing, which is
    the single most likely way for a graph gate to be worthless.  The one
    case that is EXPECTED to fall back says so with ``expect_fallback``.
    """
    import gc

    import cupy as cp

    graph = dict(use_graph=use_graph, graph_reuse=graph_reuse,
                 graph_key=graph_key, graph_scalars=graph_scalars,
                 graph_verify_topology=graph_verify_topology)
    if kind == "dry":
        rec = run_case(graph=graph, **kwargs)
    elif kind == "physics":
        rec = physics_case(graph=graph, **kwargs)
    elif kind == "geography":
        rec = geography_case(graph=graph, **kwargs)
    else:
        raise ValueError(f"unknown graph case kind {kind!r}")
    # The private capture pools are held by the graphs, the graphs by the
    # steppers, and the steppers by the run that has just returned -- so they
    # are released only once the cycle is collected.  Without this the graph
    # section accumulates one step-sized pool per case and the LAST cases die
    # of an out-of-memory that has nothing to do with what they test.  A
    # negative control that "passes" because it ran out of memory has proved
    # nothing at all, which is why this is here and not left to chance.
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    info = rec.get("graph") or {}
    rec["graph_ok"] = bool(
        info
        and ((info.get("fallbacks", 0) > 0) == bool(expect_fallback))
        and (info.get("replays", 0) > 0 or expect_fallback))
    rec["graph_info"] = info
    return rec


def capture_would_not_fit(exc: BaseException) -> str | None:
    """The reason string when THIS CARD could not hold the capture, else None.

    A graph capture grows a private memory pool inside an ACTIVE capture,
    and whether the card will serve an allocation there is a property of
    the card and its driver rather than of this code: the same section that
    runs 233/0 on an idle 16 GB RTX 4080 has been measured refusing capture
    allocations on an idle 16 GB RTX 5080 with 15.2 GiB free, at 1.1 MB in
    one run and 201.6 MB in another.  Bulk capacity is not what ran out.
    Those are two different facts and the gate has to be able to say which
    one it hit.

    An out of memory is NOT a control firing and it is NOT a control failing:
    nothing was proved and nothing was disproved, so counting it either way
    is a lie in one direction or the other.  Counting it as a PASS is the
    green-on-nothing failure this lane exists to close; counting it as a FAIL
    tells an operator on a smaller card that the transport is broken when
    what is actually true is that their card cannot run this section.  It is
    reported as its own third outcome, named in the verdict as a coverage
    hole, and never silently.

    The whole cause chain is walked because ``graphcap`` wraps a capture
    failure under ``use_graph="require"``: the cupy ``OutOfMemoryError``
    arrives as the ``__cause__`` of a ``GraphCaptureError`` whose own type
    name says nothing about memory.
    """
    seen: set[int] = set()
    cursor: BaseException | None = exc
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        name = type(cursor).__name__
        if ("OutOfMemory" in name
                or isinstance(cursor, MemoryError)
                or "out of memory" in str(cursor).lower()):
            return f"{name}: {cursor}"
        cursor = cursor.__cause__ or cursor.__context__
    return None


def graph_line(rec: dict) -> str:
    info = rec.get("graph_info") or {}
    if not info:
        return "no graph report"
    nodes = info.get("nodes") or []
    return (f"{info.get('captures', 0)} captures / "
            f"{info.get('replays', 0)} replays, "
            f"{'-'.join(str(n) for n in nodes) or '?'} nodes, "
            f"{info.get('health_records', 0)} deferred health records, "
            f"{info.get('fallbacks', 0)} fallbacks, "
            f"capture {1e3 * info.get('capture_seconds', 0.0):.1f} ms"
            + (f", {info['reason']}" if info.get("reason") else ""))


def _fmt(rec: dict, expect: bool) -> str:
    ok = rec["bitexact"] == expect
    verdict = "PASS" if ok else "FAIL"
    detail = ("bit-exact" if rec["bitexact"] else
              f"maxabs={rec['max_abs']:.3e} maxrel={rec['max_rel']:.2e} "
              f"worst={rec['worst_field']} "
              f"pattern={rec.get('signature', {}).get('verdict', '?')}")
    return f"{verdict:4s}  {detail}"


def main(argv=None) -> int:
    """Run the gate with its own output counted, so the verdict states a size."""
    checks = _CheckCounter(sys.stdout)
    saved, sys.stdout = sys.stdout, checks
    try:
        return _run_gate(argv, checks)
    finally:
        sys.stdout = saved


def _run_gate(argv, checks: _CheckCounter) -> int:
    import gc

    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    # --dry-only / --physics-only / --geography-only each select ONE section;
    # with none of them every section runs.
    only = {"--dry-only", "--physics-only", "--geography-only",
            "--restart-only", "--receipts-only",
            "--graph-only"} & set(argv)
    run_dry = not only or "--dry-only" in only
    run_physics = not only or "--physics-only" in only
    run_geography = not only or "--geography-only" in only
    # The restart round trip lives in its own module because it is a
    # different KIND of proof -- it compares two runs separated by a file
    # rather than two ways of integrating one run -- but it is part of the
    # gate, not an optional extra: a forecast that cannot be restarted
    # cannot be operated.
    run_restart = not only or "--restart-only" in only
    # The domain-receipt gate is here for the same reason the restart one is:
    # a conservation receipt that is quietly wrong is worse than no receipt,
    # and nothing else in this file would notice, because a receipt changes
    # no byte of the forecast.
    run_receipts = not only or "--receipts-only" in only
    run_graph = not only or "--graph-only" in only
    # The two names that predate the three-way split, kept so a section added
    # against the old spelling still selects correctly.
    dry_only, physics_only = "--dry-only" in only, "--physics-only" in only

    # Box-wide CUDA context count, beside the verdict, on the decomp gate's
    # own discipline: contention on this hardware changes RESULTS and not
    # only timings, so a bit-exactness verdict taken on a shared card is
    # provisional and the only way to know that afterwards is to have
    # written the count down at both ends.
    #
    # READ BEFORE THE FIRST DEVICE CALL, and that ordering is the whole
    # measurement.  ``context_verdict`` reads ctx0 as "contexts that were
    # standing there before this gate touched a card", so anything it counts
    # belongs to somebody else.  The memGetInfo below creates this process's
    # own context; taken first it made ctx0 == 1 on a provably empty box and
    # every run -- idle or not -- printed "NOT idle ... the verdict is
    # provisional".  A gate that can never report a clean box has stopped
    # measuring contention and only reports that it ran.  The other three
    # runners already read it first (test_seam_gate.py:436); this one now
    # does too.
    ctx0 = cuda_contexts()
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    print(f"CUDA CONTEXTS AT START: {ctx0}"
          + ("   <- CONTENDED; a bit-exactness verdict taken here is NOT a "
             "pass" if ctx0 > 0
             else "   (nothing on the card yet, this gate included)"))
    print()

    failures: list[str] = []
    #: Cases the CARD could not run, as distinct from cases that failed.
    #: Never empty silently: every entry is printed where it happened and
    #: named again in the verdict.
    machine_limited: list[str] = []

    print("=" * 78)
    print("PRECONDITIONS")
    print("=" * 78)
    for name, fn in (("streaming inventory complete", precondition_inventory),
                     ("tile setup == parent window", precondition_setup),
                     ("pinned host store round trip", precondition_hoststore)):
        try:
            msg = fn()
            print(f"  PASS  {name}\n        {msg}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"precondition {name}: {exc}")
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
    try:
        for line in api_contracts():
            print(f"  PASS  {line}")
    except Exception as exc:                           # noqa: BLE001
        failures.append(f"api contract: {exc}")
        print(f"  FAIL  api contract: {exc}")
        traceback.print_exc()
    print()

    print("=" * 78)
    print("RING GEOMETRY  (the invariant, for every plan the matrix runs)")
    print("=" * 78)
    try:
        for line in ring_plan_survey():
            print(f"  PASS  {line}")
    except Exception as exc:                           # noqa: BLE001
        failures.append(f"ring geometry: {exc}")
        print(f"  FAIL  ring geometry: {exc}")
        traceback.print_exc()
    print()

    print("=" * 78)
    print("RING vs SHADOW  (same configuration both ways; digests must match "
          "each other AND monolithic)")
    print("=" * 78)
    rcases = RING_CASES[:8] + RING_CASES[-3:] if quick else RING_CASES
    for label, kwargs, expect in (rcases if run_dry else []):
        try:
            rec = ring_vs_shadow(**kwargs)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"ring {label}: raised {exc!r}")
            print(f"  ERROR {label}: {exc!r}")
            traceback.print_exc()
            continue
        ok = rec["bitexact"] == expect and rec["agree"]
        print(f"  {'PASS' if ok else 'FAIL':4s}  ring "
              f"{'==' if rec['ring']['bitexact'] else '!='} monolithic, "
              f"shadow {'==' if rec['shadow']['bitexact'] else '!='} "
              f"monolithic, ring {'==' if rec['agree'] else '!='} shadow")
        print(f"        {label}")
        print(f"        {rec['tiles']} tiles, arena "
              f"{rec['ring_bytes'] / 1e6:.1f} MB = "
              f"{100 * rec['ring_over_store']:.1f}% of the store, against a "
              f"shadow of {rec['shadow_bytes'] / 1e6:.1f} MB = 100%, "
              f"{rec['seconds']:.2f} s")
        if rec["bitexact"]:
            print(f"        sha256 {rec['hash'][:32]}")
        if not ok:
            failures.append(
                f"ring {label}: bitexact={rec['bitexact']} (expected "
                f"{expect}), ring==shadow={rec['agree']}")
    print()

    print("=" * 78)
    print("RING NEGATIVE CONTROLS  (a broken ring, and what the hash can and "
          "cannot see)")
    print("=" * 78)
    for label, kwargs, expect, why in (RING_NEGATIVES if run_dry
                                       else []):
        try:
            rec = run_case(**kwargs)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"ring control {label}: raised {exc!r}")
            print(f"  ERROR {label}: {exc!r}")
            continue
        ok = rec["bitexact"] == expect
        detail = ("bit-exact" if rec["bitexact"] else
                  f"differs, maxabs={rec['max_abs']:.3e} "
                  f"worst={rec['worst_field']}")
        print(f"  {'PASS' if ok else 'FAIL':4s}  {detail}")
        print(f"        {label}")
        print(f"        {why}")
        if not ok:
            failures.append(
                f"ring control {label}: expected "
                f"{'bit-exact' if expect else 'a MISMATCH'}, got "
                f"{'bit-exact' if rec['bitexact'] else 'a mismatch'}")
    for label, kwargs, why in (RING_OBSERVATIONS if run_dry else []):
        try:
            rec = run_case(**kwargs)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"ring observation {label}: raised {exc!r}")
            print(f"  ERROR {label}: {exc!r}")
            continue
        detail = ("bit-exact -- the card behaved as its single copy engine "
                  "suggests it would" if rec["bitexact"] else
                  f"MISMATCH, maxabs={rec['max_abs']:.3e} worst="
                  f"{rec['worst_field']} -- the hazard is real on this "
                  "machine right now")
        print(f"  OBS   {detail}")
        print(f"        {label}")
        print(f"        {why}")
    print()

    if run_physics or run_geography:
        print("=" * 78)
        print("PHYSICS PRECONDITIONS")
        print("=" * 78)
        try:
            msg = precondition_carrier_identity()
            print(f"  PASS  carrier arrays move, and the ring follows them\n"
                  f"        {msg}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"precondition carrier identity: {exc}")
            print(f"  FAIL  carrier identity: {exc}")
            traceback.print_exc()
        try:
            msg = precondition_geography()
            print(f"  PASS  geography is not rebuilt wrong per tile\n"
                  f"        {msg}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"precondition geography: {exc}")
            print(f"  FAIL  geography: {exc}")
            traceback.print_exc()
        if run_geography:
            try:
                for line in precondition_geography_gathered():
                    print(f"  PASS  {line}")
            except Exception as exc:                   # noqa: BLE001
                failures.append(f"precondition geography gathered: {exc}")
                print(f"  FAIL  geography gathered: {exc}")
                traceback.print_exc()
        try:
            msg = precondition_carrier_hoststore()
            print(f"  PASS  pinned carrier host store round trip\n"
                  f"        {msg}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"precondition carrier host store: {exc}")
            print(f"  FAIL  carrier host store: {exc}")
            traceback.print_exc()
        try:
            for line in precondition_physics_inventory(
                    nsteps=3 if quick else 8):
                print(f"  PASS  {line}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"precondition physics inventory: {exc}")
            print(f"  FAIL  physics inventory: {exc}")
            traceback.print_exc()
        print()

    print("=" * 78)
    print("GATE MATRIX  (192x192x49 periodic unless stated, seed "
          f"{SEED}, halo 16 unless stated)")
    print("=" * 78)
    cases = (CASES[:6] + CASES[12:] if quick else CASES) if run_dry else []
    for label, kwargs, expect in cases:
        try:
            rec = run_case(**kwargs)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"{label}: raised {exc!r}")
            print(f"  ERROR {label}: {exc!r}")
            traceback.print_exc()
            continue
        line = _fmt(rec, expect)
        print(f"  {line}")
        print(f"        {label}")
        print(f"        {rec['tiles']} tiles, compute {rec['compute']}, "
              f"redundancy {rec['redundancy']:.2f}x, "
              f"{rec['gathered_bytes'] / 1e6:.0f} MB gathered / "
              f"{rec['scattered_bytes'] / 1e6:.0f} MB scattered, "
              f"{rec['seconds']:.2f} s")
        if rec["bitexact"]:
            print(f"        sha256 {rec['hash'][:32]}")
        else:
            bad = [n for n, e in rec["fields"].items() if not e["bitexact"]]
            print(f"        fields differing: {bad}")
            sig = rec.get("signature", {})
            if sig:
                print(f"        pattern: {sig}")
        if rec["bitexact"] != expect:
            failures.append(
                f"{label}: expected {'bit-exact' if expect else 'a MISMATCH'}, "
                f"got {'bit-exact' if rec['bitexact'] else 'a mismatch'}")
    print()

    if run_physics:
        print("=" * 78)
        print(f"PHYSICS GATE MATRIX  ({PHYS_NX}x{PHYS_NY}x{NZ} periodic, "
              f"WK82 sounding, warmup {PHYS_WARMUP}, halo "
              f"{harness.halo_radius(physics_cfg('dry (control)'))}, whole "
              f"restart manifest streamed)")
        print("=" * 78)
        pcases = PHYSICS_CASES
        if quick:
            pcases = [c for c in PHYSICS_CASES
                      if ("256x192" not in c[0]
                          and ("N=8" not in c[0]
                               or c[0].startswith("NEGATIVE")))]
        for label, kwargs, expect in pcases:
            try:
                rec = physics_case(**kwargs)
            except Exception as exc:                   # noqa: BLE001
                failures.append(f"{label}: raised {exc!r}")
                print(f"  ERROR {label}: {exc!r}")
                traceback.print_exc()
                continue
            ok = rec["bitexact"] == expect
            if rec["bitexact"]:
                detail = (f"bit-exact over all {rec['carriers']} carriers, "
                          f"clock {'ok' if rec['scalars_ok'] else 'WRONG'}")
            else:
                detail = (f"{len(rec['differing'])}/{rec['carriers']} differ, "
                          f"maxabs={rec['max_abs']:.3e} "
                          f"maxrel={rec['max_rel']:.2e} "
                          f"worst={rec['worst_field']} "
                          f"pattern={rec['signature'].get('verdict', '?')}")
            print(f"  {'PASS' if ok else 'FAIL':4s}  {detail}")
            print(f"        {label}")
            print(f"        {rec['tiles']} tiles, compute {rec['compute']}, "
                  f"halo {rec['halo']}, "
                  f"{rec['bytes_per_cell']:.1f} B/cell gathered, "
                  f"{rec['gathered_bytes'] / 1e6:.0f} MB gathered / "
                  f"{rec['scattered_bytes'] / 1e6:.0f} MB scattered, "
                  f"{rec['seconds']:.2f} s")
            if not rec["bitexact"]:
                print(f"        first differing: {rec['differing'][:8]}")
            if not ok:
                failures.append(
                    f"{label}: expected "
                    f"{'bit-exact' if expect else 'a MISMATCH'}, got "
                    f"{'bit-exact' if rec['bitexact'] else 'a mismatch'}")
            if rec["bitexact"] and not rec["scalars_ok"]:
                failures.append(
                    f"{label}: arrays match but the scalar carriers do not; "
                    "the domain clock is wrong and a longer run would diverge")
        print()

        print("=" * 78)
        print("PHYSICS RING vs SHADOW  (every carrier, both ways)")
        print("=" * 78)
        prcases = (PHYSICS_RING_CASES[:len(PHYSICS_RING_RUNGS)]
                   if quick else PHYSICS_RING_CASES)
        for label, kwargs, expect in prcases:
            try:
                rec = physics_ring_vs_shadow(**kwargs)
            except Exception as exc:                   # noqa: BLE001
                failures.append(f"physics ring {label}: raised {exc!r}")
                print(f"  ERROR {label}: {exc!r}")
                traceback.print_exc()
                continue
            ok = rec["bitexact"] == expect and rec["agree"]
            print(f"  {'PASS' if ok else 'FAIL':4s}  ring and shadow both "
                  f"bit-exact over all {rec['carriers']} carriers"
                  if ok else
                  f"  FAIL  ring bitexact={rec['ring']['bitexact']} "
                  f"shadow bitexact={rec['shadow']['bitexact']} "
                  f"disagree on {rec['differing'][:6]}")
            print(f"        {label}   ({rec['seconds']:.1f} s)")
            if not ok:
                failures.append(
                    f"physics ring {label}: bitexact={rec['bitexact']}, "
                    f"ring==shadow={rec['agree']}")
        print()

    if run_geography:
        print("=" * 78)
        print(f"GEOGRAPHY GATE MATRIX  ({GEO_NX}x{GEO_NY}x{NZ} periodic, "
              f"REAL Lambert conformal (map_proj=1) at dx=dy=12 km, "
              f"real terrain,")
        print(f"                        per-column lat/lon, halo "
              f"{harness.halo_radius(geography_cfg('dry (control)'))}, "
              f"geography GATHERED not rebuilt)")
        print("=" * 78)
        gcases = GEOGRAPHY_CASES
        if quick:
            gcases = [c for c in GEOGRAPHY_CASES
                      if ("256x192" not in c[0]
                          and ("N=8" not in c[0] or c[0].startswith("NEG")))]
        for label, kwargs, expect in gcases:
            try:
                rec = geography_case(**kwargs)
            except Exception as exc:                   # noqa: BLE001
                failures.append(f"{label}: raised {exc!r}")
                print(f"  ERROR {label}: {exc!r}")
                traceback.print_exc()
                continue
            ok = rec["bitexact"] == expect
            if rec["bitexact"]:
                detail = (f"bit-exact over all {rec['carriers']} carriers, "
                          f"clock {'ok' if rec['scalars_ok'] else 'WRONG'}")
            else:
                detail = (f"{len(rec['differing'])}/{rec['carriers']} differ, "
                          f"maxabs={rec['max_abs']:.3e} "
                          f"maxrel={rec['max_rel']:.2e} "
                          f"worst={rec['worst_field']} "
                          f"pattern={rec['signature'].get('verdict', '?')}")
            print(f"  {'PASS' if ok else 'FAIL':4s}  {detail}")
            print(f"        {label}")
            print(f"        {rec['tiles']} tiles, compute {rec['compute']}, "
                  f"halo {rec['halo']}, {rec['geo_fields']} geography fields "
                  f"gathered {rec['geo_gathers']}x "
                  f"({rec['geo_bytes'] / 1e6:.1f} MB = "
                  f"{100 * rec['geo_fraction']:.1f}% of the carrier gather), "
                  f"{rec['seconds']:.2f} s")
            if not rec["bitexact"]:
                print(f"        first differing: {rec['differing'][:8]}")
            if not ok:
                failures.append(
                    f"{label}: expected "
                    f"{'bit-exact' if expect else 'a MISMATCH'}, got "
                    f"{'bit-exact' if rec['bitexact'] else 'a mismatch'}")
            if not rec["geo_readonly"]:
                failures.append(
                    f"{label}: the run WROTE geography back "
                    f"({rec['geo_touched'][:6]}); it is INPUT and nothing "
                    "may scatter it")
            if rec["bitexact"] and not rec["scalars_ok"]:
                failures.append(
                    f"{label}: arrays match but the scalar carriers do not; "
                    "the domain clock is wrong and a longer run would "
                    "diverge")
        print()

    if run_dry and not quick:
        print("=" * 78)
        print("HALO SWEEP  (192x192x49, 3x3 tiles of 64x64)")
        print("=" * 78)
        for nsteps in (1, 3):
            print(f"  N = {nsteps} step(s):")
            smallest = None
            for h, ok, mx, worst, verdict in halo_sweep(nsteps=nsteps):
                mark = "bit-exact" if ok else (
                    f"FAIL maxabs={mx:.3e} worst={worst} pattern={verdict}")
                print(f"    halo={h:<3d} {mark}")
                if ok and smallest is None:
                    smallest = h
            print(f"    -> smallest halo that passes at N={nsteps}: {smallest}"
                  f"   (harness.halo_radius says 16)")
        print()
        print("  NOTE the minimum MOVED with the step count. It is not a "
              "structural bound;")
        print("  run --deep for the halo x step-count table that shows a "
              "'passing' halo")
        print("  going bit-exact at N=3 and wrong by 0.5 at N=8. Set the halo "
              "from")
        print("  harness.halo_radius(cfg), never from a short measurement.")
        print()

    if "--deep" in argv:
        if run_dry:
            print("=" * 78)
            print("HALO x STEP COUNT  (why a one-step gate must not set the halo)")
            print("=" * 78)
            counts = (1, 2, 3, 5, 8)
            table = halo_vs_steps(step_counts=counts)
            print("  halo " + "".join(f"{'N=' + str(n):>14s}" for n in counts))
            for h, row in table.items():
                cells = "".join(f"{'.':>14s}" if row[n][0]
                                else f"{row[n][1]:14.3e}" for n in counts)
                print(f"  {h:4d} {cells}")
            print("  ('.' = bit-exact; a number is max|tiled - monolithic|)")
            print()
        if run_physics:
            print("=" * 78)
            print("HALO x STEP COUNT, PHYSICS ON  (does physics widen the "
                  "dependency radius?)")
            print("=" * 78)
            counts = (1, 2, 3, 5, 8)
            for rung in ("full+MYNN+Noah-MP", "full fast cadence"):
                table = physics_halo_vs_steps(rung=rung, step_counts=counts)
                print(f"  rung {rung}, 256x192x49, 4x4 tiles of 64x48 "
                      f"(a gathered tile is ~40% of the domain)")
                print("  halo " + "".join(f"{'N=' + str(n):>10s}"
                                          for n in counts))
                for h, row in table.items():
                    cells = "".join(f"{'.':>10s}" if row[n][0]
                                    else f"{row[n][1]:10d}" for n in counts)
                    print(f"  {h:4d} {cells}")
                print("  ('.' = bit-exact; a number is how many carriers "
                      "differ)")
                print()
            print("  harness.halo_radius(cfg) = "
                  f"{harness.halo_radius(physics_cfg('full+MYNN+Noah-MP'))} "
                  "must be >= every minimum in these tables.")
            print()
        if run_geography:
            print("=" * 78)
            print("HALO x STEP COUNT, REAL GEOGRAPHY AT dx=12 km")
            print("=" * 78)
            counts = (1, 3, 8)
            table = geography_halo_vs_steps(step_counts=counts)
            print("  rung full+MYNN+Noah-MP, 256x192x49, 4x4 tiles of 64x48, "
                  "real Lambert + terrain")
            print("  halo " + "".join(f"{'N=' + str(n):>10s}"
                                      for n in counts))
            for h, row in table.items():
                cells = "".join(f"{'.':>10s}" if row[n][0]
                                else f"{row[n][1]:10d}" for n in counts)
                print(f"  {h:4d} {cells}")
            print("  ('.' = bit-exact; a number is how many carriers differ)")
            print("  The SAME geometry at dx=500 m needs halo 14 (see the "
                  "table above).")
            print("  A coarse grid is a WEAKER halo test, not a stronger "
                  "one: the dependency")
            print("  radius is structural but what a short halo truncates "
                  "is 24x smaller here")
            print("  and rounds to zero in float32.  Take the halo from "
                  "harness.halo_radius(cfg).")
            print()

    if run_restart:
        # Delegated whole, including its own PASS/FAIL printing and its own
        # negative controls, so there is exactly one implementation of the
        # round trip and no chance of the two drifting.  ``--no-cost`` keeps
        # the I/O sweep out of the correctness gate: it allocates a resident
        # device state at 400x320x49 and its numbers are a measurement, not
        # a pass criterion.
        from tilestream import test_restart_gate

        print("=" * 78)
        print("RESTART ROUND TRIP  (see tilestream/test_restart_gate.py)")
        print("=" * 78)
        if test_restart_gate.main(["--no-cost"] + (["--quick"] if quick
                                                   else [])):
            failures.append("restart round trip gate failed (see above)")

    if run_receipts:
        # Delegated whole, like the restart round trip, and for the same
        # reason: it is a different KIND of proof.  Every other section here
        # compares two ways of integrating a domain and demands the same
        # BYTES.  This one compares two ways of REDUCING a domain, where the
        # bytes are already known equal and the question is whether the
        # number taken off them is -- which is a question the carrier
        # matrix cannot ask, because a receipt is not a carrier and nothing
        # in the forecast reads it.
        from tilestream import test_receipts_gate

        print("=" * 78)
        print("DOMAIN RECEIPTS  (see tilestream/test_receipts_gate.py)")
        print("=" * 78)
        if test_receipts_gate.main(["--quick"] if quick else []):
            failures.append("domain receipts gate failed (see above)")

    if run_graph:
        print("=" * 78)
        print("CUDA GRAPH REPLAY  (the same cases, with the step captured "
              "once per buffer and replayed per tile)")
        print("=" * 78)
        gcases = GRAPH_CASES[:4] + GRAPH_CASES[-2:] if quick else GRAPH_CASES
        for label, kwargs, expect in gcases:
            try:
                rec = graph_case(**kwargs)
            except Exception as exc:                       # noqa: BLE001
                # A capture the CARD cannot hold is not a gate failure and
                # not a pass: see :func:`capture_would_not_fit`.  It is
                # named, counted, and carried into the verdict below as a
                # coverage hole, so an operator on a smaller card is told
                # which rows their card did not run instead of being told
                # the transport is broken.
                limit = capture_would_not_fit(exc)
                if limit is None:
                    failures.append(f"graph {label}: raised {exc!r}")
                    print(f"  ERROR {label}: {exc!r}")
                    traceback.print_exc()
                    continue
                machine_limited.append(f"graph {label}: {limit}")
                print(f"{MACHINE_LIMITED}  the graph capture does not fit "
                      f"this card, so this case was NOT evaluated: {limit}")
                print(f"        {label}")
                # The same release the negative controls do below, for the
                # same reason: the live traceback holds the frames, the
                # frames hold the run that raised, and the run holds one
                # step-sized capture pool per graph.  Without it the case
                # after this one dies of the previous case's memory.
                traceback.clear_frames(exc.__traceback__)
                del exc
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                continue
            ok = rec["bitexact"] == expect and rec["graph_ok"]
            print(f"  {'PASS' if ok else 'FAIL':4s}  "
                  f"{_fmt(rec, expect)}")
            print(f"        {label}")
            print(f"        {graph_line(rec)}")
            if not ok:
                failures.append(
                    f"graph {label}: bitexact={rec['bitexact']} (expected "
                    f"{expect}), graph_ok={rec['graph_ok']}, "
                    f"{graph_line(rec)}")
        print()

        print("=" * 78)
        print("CUDA GRAPH NEGATIVE CONTROLS  (a graph replayed under the "
              "wrong assumptions)")
        print("=" * 78)
        for label, kwargs, why in GRAPH_NEGATIVES:
            detected, limited, detail = False, None, ""
            try:
                rec = graph_case(**kwargs)
                detected = not rec["bitexact"]
                detail = ("differs in "
                          f"{len(rec.get('differing', []))} carriers, "
                          f"maxabs={rec.get('max_abs', 0.0):.3e}"
                          if detected else "bit-exact -- NOT DETECTED")
            except Exception as exc:                       # noqa: BLE001
                # The driver's own cross-buffer clock check refusing IS a
                # detection, and a louder one than a digest mismatch.
                blame = f"{type(exc).__name__}: {exc}"
                # THREE OUTCOMES, not two, and the middle one is the whole
                # point of this block.
                #
                # A TypeError / AttributeError is the HARNESS failing -- a
                # keyword the driver no longer takes, a helper that moved --
                # and it looked exactly like a clean sheet: all three of
                # these controls reported PASS against
                # "run_tiled() got an unexpected keyword argument
                # 'use_graph'" while the graph section itself was erroring
                # on every case.  A control that passes because the call did
                # not happen is the control that cannot fail, so that stays
                # a FAIL and the gate stays red.
                #
                # An OUT OF MEMORY IN THE CAPTURE is a different animal: the
                # card could not hold the private pool, so the control never
                # ran and neither a pass nor a fail is a true report of it.
                # Counting it as a pass is green-on-nothing.  Counting it as
                # a fail says the transport is broken to an operator whose
                # only problem is a different card -- measured, both on the
                # same 16 GB: an idle RTX 4080 runs the whole section
                # 233/0, an idle RTX 5080 is refused capture allocations
                # with 15.2 GiB free.  So it is MACHINE-LIMITED: named
                # here, counted
                # by the check counter, and spelled out in the verdict as a
                # coverage hole.  It never disappears.
                limited = capture_would_not_fit(exc)
                harness_bug = isinstance(exc, (TypeError, AttributeError,
                                               NameError, ImportError))
                detected = not (limited or harness_bug)
                detail = (f"refused: {blame}" if detected else
                          f"the capture does not fit this card, so this "
                          f"control was NOT evaluated: {blame}" if limited
                          else f"INCONCLUSIVE, the harness failed: {blame}")
                # ONE UNEVALUATED CONTROL MUST NOT POISON THE REST.  A live
                # exception holds its traceback, the traceback holds the
                # frames, and the frames hold the run that raised -- the
                # steppers, their graphs, and the private capture pool each
                # graph owns.  So the case that ran out of memory keeps the
                # memory, and every control after it dies for a reason that
                # has nothing to do with what it tests.  Measured on a 16 GB
                # RTX 4080: control 2 died at a 201.6 MB capture allocation
                # and control 3 then died at the same call with 75 MB
                # allocated, on a card that started the gate with 15.3 GiB
                # free.  Clearing the frames and releasing the pool costs
                # nothing when nothing failed.
                traceback.clear_frames(exc.__traceback__)
                del exc
                gc.collect()
                cupy_pool = cp.get_default_memory_pool()
                cupy_pool.free_all_blocks()
            if limited:
                machine_limited.append(f"graph control {label}: {limited}")
                print(f"{MACHINE_LIMITED}  {detail}")
            else:
                print(f"  {'PASS' if detected else 'FAIL':4s}  {detail}")
            print(f"        {label}")
            print(f"        {why}")
            if not (detected or limited):
                failures.append(
                    f"graph control {label}: expected a MISMATCH, got "
                    f"{detail}")
        print()

    print("=" * 78)
    ctx1, verdict = context_verdict(ctx0)
    print(f"CUDA CONTEXTS AT END: {ctx1} (start {ctx0}).  " + verdict)
    # THE DECLARED PRECONDITION, evaluated before the verdict below and not
    # as one more line inside it: "every configuration behaved as specified"
    # is a statement about all checks, and over zero checks it is true of
    # none.  The floor is ONE rather than the full expected count so that
    # --quick and the --*-only selectors stay legitimate partial runs.
    total = checks.passed + checks.failed
    sections = ", ".join(name for name, ran in (
        ("dry tiling", run_dry), ("physics", run_physics),
        ("geography", run_geography), ("restart", run_restart),
        ("domain receipts", run_receipts), ("graph replay", run_graph),
    ) if ran)
    if total < 1:
        print("GATE REFUSED -- the matrix is empty: 0 checks were evaluated "
              "against a floor of 1, so no verdict is available.  Sections "
              f"selected: {sections or 'none'}.  Drop the --*-only selector "
              "that emptied it, or run without --quick.")
        return 2
    size = (f"{checks.passed} checks passed and {checks.failed} failed "
            f"over sections: {sections}")
    if machine_limited:
        size += (f", with {len(machine_limited)} MACHINE-LIMITED "
                 f"({checks.machine_limited} such line(s) printed)")
    if failures:
        print(f"GATE FAILED -- {len(failures)} problem(s), {size}:")
        for f in failures:
            print(f"  * {f}")
        return 1
    # A CARD THAT COULD NOT RUN A ROW IS NOT A CARD THAT PASSED IT.  The
    # graph section grows a private capture pool per graph and is
    # card-dependent at the same capacity: an idle 16 GB RTX 4080 runs it
    # whole, an idle 16 GB RTX 5080 refuses the allocation.  The gate does
    # not fail on that -- nothing was disproved -- but the strongest sentence
    # in the file is not available either, so the verdict states the hole and
    # lists every row inside it.
    if machine_limited:
        print(f"GATE PASSED WITH A COVERAGE HOLE -- every configuration this "
              f"card could run behaved as specified, including every negative "
              f"control it could run, but {len(machine_limited)} row(s) were "
              f"NOT evaluated on this card and this verdict says nothing "
              f"about them ({size}):")
        for m in machine_limited:
            print(f"  * {m}")
        return 0
    print(f"GATE PASSED -- every configuration behaved as specified, including "
          f"every negative control ({size}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
