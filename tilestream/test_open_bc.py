"""OPEN LATERAL BOUNDARIES, streamed: does a tile know which of its four
window edges is a REAL domain edge?

``cfg.open_x`` / ``cfg.open_y`` select WRF's Klemp-Wilhelmson radiative-open
lateral boundary: the advection kernels take open loop bounds and drop the
boundary-normal flux at the two boundary faces
(:func:`gpuwm.core.advection._launch`), the small-step u/v kernels clamp
instead of wrapping the boundary-face column mass
(``dycore._boundary_x``/``_boundary_y``), an outbound-only radiative term is
added to the coupled slow tendency at the boundary-normal velocity faces
(``dycore.apply_open_radiative_bc``) and the boundary COLUMN's theta, w,
tangential velocity and moisture are OVERWRITTEN from their first interior
neighbour at the end of every RK stage wherever the flow is outbound
(``dycore.apply_open_zero_gradient``).

Nothing in ``tilestream`` had ever run with either flag set.  ``autoplan
.is_periodic`` reads them only to decide that a tile window must not wrap.

WHY THIS IS NOT THE SPECIFIED-BC PROBLEM IN A DIFFERENT COSTUME
---------------------------------------------------------------
:func:`gpuwm.core.streaming.window_interval` solved the specified case by
giving a tile's TRUE domain edges the domain's own tables and its interior
seams inert ones, and the reason inert seams are provably safe is stated in
its docstring: specified forcing perturbs the RK TENDENCY, so its influence
cone is strictly inside the dycore's own and the halo covers it with room to
spare -- measured, zeros / self / 1e6-poison seam tables all bit-identical
out to N=24.

An open boundary is not a tendency.  ``apply_open_zero_gradient`` assigns
STATE at the window's edge column, three times per step (once per RK stage),
and the open advection bounds stop computing tendencies for the three
columns nearest each face.  A state perturbation present at the start of a
step has exactly the dycore's own dependency radius as its cone, and that
radius IS the halo (``harness.halo_radius``).  Marginal by construction --
and marginal is how a too-small halo hides: silently, and faster.

WHAT ``harness.tile_config`` DOES WITH THE FLAG
----------------------------------------------
``harness.tile_config`` (harness.py:178-186) replaces ``nx``/``ny`` and
carries every other field of ``RunConfig`` through untouched, and
``driver.TiledRun`` builds ONE ``tile_cfg`` for every buffer
(driver.py:1159).  So with ``open_x=True`` every tile buffer applies the
open-boundary treatment at BOTH of its x window edges -- including the ones
that are interior seams hundreds of kilometres inside the domain.

There is no per-tile spelling of "open on the west only": ``RunConfig`` has
one ``open_x`` for both x sides, where WRF carries ``open_xs`` and
``open_xe`` separately (share/module_bc.F).  A tile whose window spans the
whole axis owns both edges and is correct; a tile whose window spans neither
edge wants the flag OFF (its seams are then wrapped, which is what the halo
already covers, and which is exactly the proven periodic case); a tile that
owns exactly ONE x edge cannot be expressed at all.

THE SECOND DEFECT, WHICH IS INDEPENDENT OF THE FIRST
----------------------------------------------------
``autoplan.is_periodic`` collapses BOTH axes into one boolean, and
``spec.plan_tiles`` takes one boolean.  ``open_x=True, open_y=False`` is a
domain that is non-periodic in x and genuinely PERIODIC in y
(``dycore._boundary_y`` is false, so every y stencil wraps).  ``is_periodic``
returns False, so ``plan_tiles`` CLAMPS the y windows into the domain
instead of wrapping them, and the south-most tile's row 0 is an owned cell
whose y neighbours the kernel takes by wrapping to row ``cny-1`` OF THE
WINDOW -- a row 64 cells away instead of the domain's row ``ny-1``.  Mixed
periodicity is not expressible in the plan.

THE ARMS
--------
``periodic``   the positive control the whole lane rests on; must pass.
``specified``  the positive control the join lane proved; must pass.
``open_x``     x open, y periodic.  Exercises BOTH defects.
``open_xy``    both axes open.  Both axes are non-periodic, so the clamping
               is right and this arm isolates the edge-conditional defect
               alone.

Every arm runs the same geometry, the same seed and the same halo, so the
only difference between them is four boolean fields of ``RunConfig``.

    python -m tilestream.test_open_bc              # the whole matrix
    python -m tilestream.test_open_bc --quick      # dry + fast cadence, N=1,8

RADIATION AND CUMULUS FIRE COUNTS ARE PRINTED FOR EVERY CASE, on both sides.
At ``dt=30 s`` with ``radt_minutes=12`` radiation is due every 24 steps and
cumulus every 10, so an N=8 window at the ordinary cadence fires NEITHER --
which is how three of this project's six false results happened.  The
``full fast cadence`` rung exists to make both fire on every step, and the
count is printed rather than assumed.
"""

from __future__ import annotations

import hashlib
import sys
import time
import warnings

import numpy as np

from gpuwm.core import streaming
from tilestream import driver, gather, harness
from tilestream import physics_inventory as physinv
from tilestream import spec as tspec

# --------------------------------------------------------------------------
# the configuration
# --------------------------------------------------------------------------

NZ = 49
SEED = 20_260_731

#: 256x192 at dx = 12 km, split 8x6 into 32x32 tiles.  The join lane's own
#: geometry, chosen there because it puts 24 tiles with NO true edge, 20 with
#: one and 4 with two into the same plan -- which is precisely the population
#: an edge-conditional feature has to get right.  The compute window is 64
#: cells, far below the ~500 a TIMING would need; nothing here is timed.
NX, NY = 256, 192
TX, TY = 32, 32
STEP_COUNTS = (1, 3, 8)
#: The controls run at N=3, not N=1.  A one-step window certifies a halo that
#: is wrong (``harness.halo_radius``'s FACT 1: halo 13 is bit-exact at N=1 and
#: differs in 13 / 34 / 107 carriers at N = 2 / 3 / 5), so a control taken at
#: N=1 is the weakest form of every one of these questions.
NSTEPS_CONTROL = 3

#: open_x/open_y REFUSE terrain (config.py:2094 and again dycore.py:2259-2266:
#: ``set_w_surface`` and the ``advance_w_phi`` kinematic surface BC difference
#: ``ht`` with unconditional periodic wraps).  So the geography here is a real
#: Lambert projection with FLAT terrain: ``msft``, ``msfu``, ``msfv``, ``f``
#: and ``e`` all vary horizontally and are gathered per buffer exactly as the
#: join lane gathers them; ``thb/pb/alb/phb`` stay 1-D.  That is the most
#: geography an open-boundary domain is allowed to have, and saying so is the
#: point -- this is a documented limit of the FEATURE, not of the transport.
GEOGRAPHY = dict(map_proj=1, terrain_opt=0, dx=12000.0, dy=12000.0)

_MOIST = dict(moist=True, mp_physics=10, ztop=20000.0)
_FULL = dict(_MOIST, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
             bldt=0.0, sf_surface_physics=2, ra_sw_physics=4,
             ra_lw_physics=4, radt_minutes=12.0, cu_physics=1,
             cudt_minutes=5.0)

#: The same selector sets as ``test_gate.PHYSICS_RUNGS`` and
#: ``test_join.RUNGS``, so a failure here can be compared rung for rung.
#: ``full fast cadence`` is not decoration: see the module docstring.
RUNGS: dict[str, dict] = {
    "dry":               dict(ztop=20000.0),
    "mp10":              dict(_MOIST),
    "full+MYNN+Noah-MP": dict(_FULL, sf_sfclay_physics=5, bl_pbl_physics=5,
                              sf_surface_physics=4),
    "full fast cadence": dict(_FULL, sf_sfclay_physics=5, bl_pbl_physics=5,
                              sf_surface_physics=4, radt_minutes=0.05,
                              cudt_minutes=0.1, bldt=1.0),
}

#: Four boolean fields of ``RunConfig``, and nothing else, separate the arms.
ARMS: dict[str, dict] = {
    "periodic":  dict(open_x=False, open_y=False, specified=False,
                      nested=False),
    "specified": dict(open_x=False, open_y=False, specified=True,
                      nested=False, spec_bdy_width=5, spec_zone=1,
                      relax_zone=4, spec_exp=0.0),
    "open_x":    dict(open_x=True, open_y=False, specified=False,
                      nested=False),
    "open_xy":   dict(open_x=True, open_y=True, specified=False,
                      nested=False),
}

#: MEASURED stability ladder of THIS harness's random initial state at
#: dx = 12 km (test_join.RUNG_DT found the same split): dry is clean to 60 s,
#: the moist rungs go non-finite at 60 s and are clean at 30 s.  A reference
#: that is not finite has nothing to compare against.
DT_DRY = 60.0
DT_MOIST = 30.0
BDY_SECONDS = 21600.0


def open_cfg(nx: int, ny: int, rung: str, arm: str, nz: int = NZ, **over):
    """One arm of one rung: real Lambert, flat terrain, one BC selection."""
    kwargs = dict(GEOGRAPHY)
    kwargs.update(RUNGS[rung])
    kwargs.update(ARMS[arm])
    kwargs.setdefault("dt", DT_DRY if rung == "dry" else DT_MOIST)
    kwargs.update(over)
    return harness.make_config(nx, ny, nz, periodic=(arm == "periodic"),
                               **kwargs)


# --------------------------------------------------------------------------
# states
# --------------------------------------------------------------------------

def build_domain(cfg, arm, *, seed: int = SEED, boundaries=None,
                 warmup: int = 1, faces=None):
    """A prepared, resident domain on the real projection.

    ``periodic_faces`` follows the ARM's PER-AXIS periodicity, for the reason
    ``test_join`` states and one more.  On a periodic axis
    ``TileSpec._axis_gather`` reduces every window mod nx and never reads the
    alias slot, so ``msfu[:, nx]`` must equal ``msfu[:, 0]`` or a tile and a
    monolithic run disagree about a face that physically IS the same face; on
    a non-periodic axis column ``nx`` is a real east boundary face and
    duplicating would install the WEST edge's map factor on the EAST edge.

    The two axes must be asked SEPARATELY.  ``open_x`` alone is non-periodic
    in x and periodic in y, so the y face must be duplicated and the x one
    must not -- and a plan that got the periodicity right while the geography
    duplicated neither face still differs from the monolithic run in all nine
    dry carriers after one step, in the two y-boundary tile rows.
    """
    from gpuwm.ingest.lateral_bc import attach_lateral_boundaries

    px, py = streaming._periodic_axes(cfg) if faces is None else faces
    geo = harness.make_geography(cfg, terrain=False,
                                 periodic_faces_x=px, periodic_faces_y=py)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo)
    if boundaries is not None:
        attach_lateral_boundaries(state, boundaries)
    if warmup:
        harness.run_steps(state, cfg, int(warmup))
    return state, geo


def domain_boundaries(cfg, arm, *, seconds: float = BDY_SECONDS):
    """The specified arm's own forcing, from two genuinely different states.

    Two seeds give a NONZERO time tendency, so ``dtbc`` -- and therefore
    ``elapsed_seconds`` -- actually reaches the answer.  ``None`` for every
    other arm: an open domain has no forcing tables at all, which removes a
    confound rather than adding one.
    """
    if not cfg.specified:
        return None
    from gpuwm.ingest.lateral_bc import build_state_lateral_boundaries

    import cupy as cp

    a, _ = build_domain(cfg, arm, seed=SEED, warmup=0)
    b, _ = build_domain(cfg, arm, seed=SEED + 1, warmup=0)
    bnd = build_state_lateral_boundaries(
        [a, b], [0.0, float(seconds)],
        spec_bdy_width=int(cfg.spec_bdy_width),
        spec_zone=int(cfg.spec_zone), relax_zone=int(cfg.relax_zone))
    del a, b
    cp.get_default_memory_pool().free_all_blocks()
    return bnd


def tile_factory(cfg, boundaries0, *, seed: int = 4242, warmup: int = 1):
    """A ``tile_state_factory`` for an open-boundary tile buffer.

    Buffers are built on :func:`harness.neutral_geography` -- the POISON
    fill, ``msf == 1``, ``f == 0``, a latitude nowhere near the domain's --
    so a geography array the gather fails to write stays obviously wrong.
    The warmup step is still required for the lazily allocated carriers
    (Kain-Fritsch's ``cumulus/w0avg`` above all).

    ``terrain=None``, not a flat terrain ARRAY.  ``neutral_geography``
    returns a full ``(ny, nx)`` field of zeros on purpose -- the join lane
    runs ``terrain_opt=1``, where ``thb/pb/alb/phb`` are 3-D and a buffer
    built flat has the wrong SHAPES (state.py:605-607).  An open-boundary
    domain is ``terrain_opt=0`` by refusal, so here the opposite is true:
    handing ``make_base_state`` any ``terrain_z`` at all builds 3-D base
    profiles that ``init_at_rest`` then rejects against a 1-D allocation.
    """
    import dataclasses

    from gpuwm.ingest.lateral_bc import attach_lateral_boundaries

    def make(tile_cfg):
        geo = dataclasses.replace(harness.neutral_geography(tile_cfg),
                                  terrain=None)
        state, _drv = harness.make_physics_state(tile_cfg, seed,
                                                 geography=geo)
        if boundaries0 is not None:
            attach_lateral_boundaries(state, boundaries0)
        if warmup:
            harness.run_steps(state, tile_cfg, int(warmup))
        return state

    return make


# --------------------------------------------------------------------------
# comparison and localisation
# --------------------------------------------------------------------------

def _as_numpy(value):
    import cupy as cp

    return cp.asnumpy(value) if isinstance(value, cp.ndarray) \
        else np.asarray(value)


def digest_arrays(arrays) -> dict[str, str]:
    out = {}
    for name in sorted(arrays):
        host = np.ascontiguousarray(_as_numpy(arrays[name]))
        h = hashlib.sha256()
        h.update(name.encode())
        h.update(host.dtype.str.encode())
        h.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        h.update(host.tobytes(order="C"))
        out[name] = h.hexdigest()
    return out


def seam_columns(specs, nx: int, ny: int) -> tuple[list[int], list[int]]:
    """The x and y coordinates of every INTERIOR SEAM in a plan.

    A window edge that coincides with the domain's own extent is a TRUE
    DOMAIN EDGE and the open treatment there is correct; every other window
    edge is a seam, and the open treatment there is the defect under test.
    :func:`gpuwm.core.streaming.owned_edges` makes exactly this distinction
    for the specified tables and this function makes it for the geometry.
    """
    xs, ys = set(), set()
    for s in specs:
        for x in (s.ci0, s.ci0 + s.cnx - 1):
            if 0 < x < nx - 1:
                xs.add(int(x))
        for y in (s.cj0, s.cj0 + s.cny - 1):
            if 0 < y < ny - 1:
                ys.add(int(y))
    return sorted(xs), sorted(ys)


def localise(ref: dict, got: dict, specs, halo: int, nx: int, ny: int
             ) -> dict:
    """WHERE the streamed answer left the monolithic one, and what that means.

    The decisive split, and it is the reason this function exists rather than
    a bare digest comparison:

    ``seam-local``
        every differing column sits within ``halo`` of an INTERIOR SEAM.
        That localises the defect to the tile ``cfg`` carrying the open flag
        into a window edge that is not a domain edge -- and the fix is the
        one ``window_interval`` already applies to the specified tables, not
        a wider halo.

    ``spread``
        differing columns further from any seam than the halo is wide.  That
        indicts the halo (or something that is not column-local), and a
        wider halo would be the honest response.

    ``edge-local``
        differences hug the TRUE domain edges only.  The tile treats its real
        edges differently from the monolithic run -- a boundary-kernel bug,
        not a transport one.
    """
    da, db = digest_arrays(ref), digest_arrays(got)
    differing = sorted(n for n in da if da.get(n) != db.get(n))
    out: dict = {"bitexact": not differing, "ndiff": len(differing),
                 "ntotal": len(da), "differing": differing[:10]}
    nonfinite = sum(int(np.count_nonzero(~np.isfinite(_as_numpy(v))))
                    for v in got.values() if _as_numpy(v).dtype.kind == "f")
    out["nonfinite"] = nonfinite
    if not differing:
        return out

    seam_x, seam_y = seam_columns(specs, nx, ny)
    worst, worst_abs, worst_mask = None, -1.0, None
    for name in differing:
        a = _as_numpy(ref[name]).astype(np.float64)
        b = _as_numpy(got[name]).astype(np.float64)
        if a.shape != b.shape or a.dtype.kind != "f":
            continue
        d = np.abs(a - b)
        if not d.size:
            continue
        m = float(np.nanmax(d))
        if m > worst_abs:
            worst, worst_abs = name, m
            mask = d > 0.0
            while mask.ndim > 2:
                mask = mask.any(axis=0)
            worst_mask = mask
    out["worst_field"] = worst
    out["max_abs"] = worst_abs
    if worst_mask is None or not worst_mask.any():
        out["verdict"] = "no-float-difference"
        return out

    jj, ii = np.nonzero(worst_mask)
    my, mx = worst_mask.shape

    def _near(idx, seams, n):
        if not seams:
            return np.full(idx.shape, n, dtype=np.int64)
        s = np.asarray(seams)
        return np.abs(idx[:, None] - s[None, :]).min(axis=1)

    dx_seam = _near(ii, seam_x, mx)
    dy_seam = _near(jj, seam_y, my)
    near_seam = np.minimum(dx_seam, dy_seam)
    edge = np.minimum(np.minimum(ii, mx - 1 - ii),
                      np.minimum(jj, my - 1 - jj))

    out.update(
        differing_columns=int(worst_mask.sum()),
        columns=int(worst_mask.size),
        fraction=float(worst_mask.mean()),
        max_distance_to_seam=int(near_seam.max()),
        mean_distance_to_seam=float(near_seam.mean()),
        max_distance_to_domain_edge=int(edge.max()),
        frac_within_halo_of_seam=float((near_seam <= halo).mean()),
        frac_within_halo_of_domain_edge=float((edge <= halo).mean()),
        x_extent=(int(ii.min()), int(ii.max())),
        y_extent=(int(jj.min()), int(jj.max())),
    )
    if worst_mask.mean() > 0.5:
        out["verdict"] = "uniform"
    elif int(near_seam.max()) <= halo:
        out["verdict"] = "seam-local"
    elif int(edge.max()) <= halo:
        out["verdict"] = "edge-local"
    else:
        out["verdict"] = "spread"
    return out


# --------------------------------------------------------------------------
# the two runs
# --------------------------------------------------------------------------

class UnstableReference(RuntimeError):
    """The reference run itself went non-finite, so there is no comparison."""


def retry_on_oom(fn, *args, tries: int = 6, **kwargs):
    """Run ``fn``, retrying a CUDA out-of-memory as a NEIGHBOUR, not a result.

    Every box this runs on is shared with a dozen other agents, and a
    transient allocation failure caused by somebody else's process is not a
    statement about this code.  It is also not something to paper over: an
    OOM that persists through six attempts is reported as the failure it is,
    and the retry never touches the numbers -- the same seed, the same
    config, the same comparison.
    """
    import cupy as cp

    for attempt in range(int(tries)):
        try:
            return fn(*args, **kwargs)
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            if attempt == tries - 1:
                raise
            print(f"          (OOM on a shared card, attempt {attempt + 1} "
                  f"of {tries}; waiting for the neighbour)")
            time.sleep(45.0)


def _fire_counts(before: dict, after: dict) -> dict:
    a = dict(before.get("call_counts", {}) or {})
    b = dict(after.get("call_counts", {}) or {})
    return {k: int(b.get(k, 0)) - int(a.get(k, 0))
            for k in sorted(set(a) | set(b))
            if int(b.get(k, 0)) - int(a.get(k, 0)) or k in
            ("radiation", "cumulus")}


def monolithic(cfg, arm, step_counts, *, boundaries=None, warmup=1,
               faces=None) -> dict:
    """The reference, snapshotted at every N in ``step_counts``.

    Deliberately the same loop shape as :func:`streamed`, so the only
    difference between the two is which callable ``make_stepper`` returned.
    """
    import cupy as cp

    from gpuwm.core.dycore import step as dycore_step

    state, _geo = build_domain(cfg, arm, boundaries=boundaries,
                               warmup=warmup, faces=faces)
    stepper = streaming.make_stepper(state, cfg, streaming.OFF)
    assert stepper is dycore_step, (
        "the resident reference must run the dycore's own step itself")
    out: dict = {}
    fired: dict = {}
    done = 0
    # Counted from the SAME origin for every target, because the streamed
    # run counts each N from its own warmed state.  Differencing against the
    # previous snapshot instead would report N=8 as seven fires against the
    # streamed run's eight and manufacture a cadence mismatch that is not
    # there -- which is the same class of bookkeeping error as the clock bug
    # the carriers exist to prevent, and it showed up on the first run.
    origin = physinv.carrier_scalars(state)
    for target in sorted(step_counts):
        for _ in range(target - done):
            stepper(state, cfg, refl_10cm_due=False)
        done = target
        cp.cuda.runtime.deviceSynchronize()
        fired[target] = _fire_counts(origin, physinv.carrier_scalars(state))
        snap = {name: _as_numpy(arr).copy()
                for name, arr in physinv.carrier_inventory(state, None).items()}
        bad = sum(int(np.count_nonzero(~np.isfinite(v))) for v in snap.values()
                  if v.dtype.kind == "f")
        if bad:
            # Loud, because the silent version of this is a gate comparing
            # one run's NaNs against another's and reporting agreement.
            raise UnstableReference(
                f"the MONOLITHIC reference has {bad} non-finite cells after "
                f"{target} steps at dt={cfg.dt} s")
        out[target] = snap
    del state
    cp.get_default_memory_pool().free_all_blocks()
    return {"snapshots": out, "fired": fired}


def streamed(cfg, arm, nsteps, tile_nx, tile_ny, *, boundaries=None,
             warmup=1, nbuffers=2, halo=None, seam="zeros",
             carry_clock=True, faces=None, plan_axes=None,
             report: dict | None = None) -> dict:
    """The streamed run, driven one sweep per model step through the seam.

    ``plan_axes=(px, py)`` forces the plan's periodicity instead of taking it
    from the config.  It is a NEGATIVE CONTROL and nothing else: passing
    ``(False, False)`` on an ``open_x`` domain reproduces exactly the
    behaviour this lane found and fixed, where one boolean decided both axes.
    It is applied by rebinding ``streaming._periodic_axes`` for the duration
    of the run, because that is the single point every plan in the streamed
    path reads -- ``tile_specs`` for the boundary tables and ``attach`` for
    the ``TiledRun`` -- so a control that patched only one of them would
    build a self-inconsistent run and prove nothing.
    """
    domain, _geo = build_domain(cfg, arm, boundaries=boundaries,
                                warmup=warmup, faces=faces)
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile_nx), tile_ny=int(tile_ny),
        nbuffers=int(nbuffers), halo=halo, store="host")
    decision = streaming.decide(cfg, options)
    build = _make_builder(domain, boundaries=boundaries, seam=seam,
                          carry_clock=carry_clock)
    saved = streaming._periodic_axes
    if plan_axes is not None:
        streaming._periodic_axes = lambda _cfg, _a=tuple(plan_axes): _a
    try:
        stepper = streaming.make_stepper(domain, cfg, options,
                                         decision=decision, build=build)
        assert streaming.is_streaming(stepper)
        before = dict(stepper.scalars or {})
        for _ in range(int(nsteps)):
            stepper(domain, cfg, refl_10cm_due=False)
    finally:
        streaming._periodic_axes = saved
    if report is not None:
        report.update(stepper.report)
        report["decision"] = stepper.decision.explain()
        report["fired"] = _fire_counts(before, dict(stepper.scalars or {}))
    return {name: np.asarray(arr) for name, arr in stepper.store.items()}


def _make_builder(domain, *, boundaries, seam, carry_clock):
    """The route-owned construction :func:`streaming.make_stepper` needs."""

    def build(state, run_cfg, decision):
        geo_inv = {k: gather.pinned_copy(v) for k, v in
                   driver.geography_inventory(domain).items()}
        scalars = physinv.carrier_scalars(domain) if carry_clock else None
        per_tile = None
        if boundaries is not None:
            per_tile = streaming.tile_boundary_tables(
                boundaries, streaming.tile_specs(run_cfg, decision), seam=seam)
        factory = tile_factory(run_cfg,
                               None if per_tile is None else per_tile[0])
        return streaming.attach(
            state, run_cfg, decision, tile_state_factory=factory,
            geography=geo_inv, boundary_tables=per_tile, scalars=scalars,
            check_geography=False)

    return build


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def _line(label: str, ok: bool, detail: str = "") -> str:
    return f"  {'PASS' if ok else 'FAIL':4s}  {label:44s} {detail}"


def edge_census(cfg, arm, tile_nx, tile_ny, halo) -> dict:
    px, py = streaming._periodic_axes(cfg)
    specs = tspec.plan_tiles(int(cfg.nx), int(cfg.ny), int(tile_nx),
                             int(tile_ny), int(halo),
                             periodic_x=px, periodic_y=py)
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for s in specs:
        counts[sum(bool(v) for v in streaming.owned_edges(s).values())] += 1
    return {"tiles": len(specs), "by_true_edges": counts, "specs": specs}


def resident_alias_invariant(cfg, arm, nsteps: int = 1) -> dict:
    """Does the RESIDENT model keep a periodic axis's alias slot an alias?

    ``tilestream.spec``'s convention is that on a periodic axis slot ``n`` IS
    slot 0 -- ``harness.make_state`` seeds it that way and the tiled scatter
    writes the alias FROM slot 0, so a domain that breaks the invariant
    cannot be tiled at all, however wide the halo.

    This is a monolithic measurement with no tiling in it, which is what
    makes it a diagnosis rather than a symptom: ``open_x`` with a periodic y
    used to leave ``max|v[ny] - v[0]| = 2.51 m/s`` after ONE step, at exactly
    the two x boundary columns, from a seeded state where it was zero.  The
    ``periodic`` arm is the positive control (it must stay at zero, which is
    what shows the probe can see the invariant), and the ``open_xy`` arm is
    the row where a nonzero answer is CORRECT -- neither slot is an alias
    there, both are real boundary faces.
    """
    import cupy as cp

    px, py = streaming._periodic_axes(cfg)
    state, _geo = build_domain(cfg, arm, warmup=0)
    harness.run_steps(state, cfg, int(nsteps))
    v = cp.asnumpy(state.v)
    u = cp.asnumpy(state.u)
    out = {
        "periodic_x": px, "periodic_y": py,
        "v_alias_max": float(np.abs(v[:, -1, :] - v[:, 0, :]).max()),
        "u_alias_max": float(np.abs(u[:, :, -1] - u[:, :, 0]).max()),
    }
    out["ok"] = ((not py or out["v_alias_max"] == 0.0)
                 and (not px or out["u_alias_max"] == 0.0))
    del state
    cp.get_default_memory_pool().free_all_blocks()
    return out


#: Every control, and the ONE capability each removes.  All are run at the
#: dry rung on the ``open_x`` arm -- the arm that needs every one of them --
#: and all MUST differ.  A control that passes is a capability that was doing
#: nothing.
CONTROLS: tuple[tuple[str, dict], ...] = (
    # The pre-fix behaviour exactly: one boolean decided both axes, so the
    # periodic y windows were clamped and the two y-boundary tile rows were
    # stepped with an owned row 0 that had no halo beneath it.
    ("plan clamps the periodic y axis", dict(plan_axes=(False, False))),
    # The x face duplicated (x is a real open boundary: this installs the
    # WEST map factor on the EAST edge) and the y face not (y wraps: the tile
    # then reads row 0 where the monolithic run reads row ny).
    ("closing faces duplicated on the wrong axis", dict(faces=(True, False))),
    # HALF the dependency radius, not one cell below it.  The smallest halo
    # that happens to pass moves with step count, domain size and GPU, so the
    # control is a value the dependency argument says CANNOT work.
    ("halo 8, half the dependency radius", dict(halo=8)),
)


def _controls(halo: int) -> list[str]:
    """Run every negative control; return the ones that failed to fire."""
    import cupy as cp

    print("-- THE RESIDENT ALIAS INVARIANT (monolithic; no tiling in it)")
    bad: list[str] = []
    for arm in ARMS:
        cfg = open_cfg(NX // 2, NY // 2, "dry", arm)
        if cfg.specified:
            continue                       # needs tables; covered by the arm
        inv = resident_alias_invariant(cfg, arm)
        print(_line(f"{arm}: periodic x={inv['periodic_x']} "
                    f"y={inv['periodic_y']}", inv["ok"],
                    f"max|v[ny]-v[0]|={inv['v_alias_max']:.6g} "
                    f"max|u[nx]-u[0]|={inv['u_alias_max']:.6g}"))
        if not inv["ok"]:
            bad.append(f"resident alias invariant broken on {arm}")
        cp.get_default_memory_pool().free_all_blocks()

    print()
    print("-- EVERY NEGATIVE CONTROL, on open_x / dry (each MUST differ)")
    cfg = open_cfg(NX, NY, "dry", "open_x")
    px, py = streaming._periodic_axes(cfg)
    specs = tspec.plan_tiles(NX, NY, TX, TY, halo, periodic_x=px,
                             periodic_y=py)
    for label, kwargs in CONTROLS:
        faces = kwargs.get("faces")
        ref = retry_on_oom(monolithic, cfg, "open_x", (NSTEPS_CONTROL,),
                           faces=faces)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            got = retry_on_oom(streamed, cfg, "open_x", NSTEPS_CONTROL, TX,
                               TY, halo=kwargs.get("halo", halo),
                               faces=faces,
                               plan_axes=kwargs.get("plan_axes"))
        res = localise(ref["snapshots"][NSTEPS_CONTROL], got, specs, halo,
                       NX, NY)
        fired = not res["bitexact"]
        print(_line(label, fired,
                    f"ndiff={res['ndiff']}/{res['ntotal']}"
                    + ("" if not fired
                       else f" max|d|={res['max_abs']:.6g} "
                            f"verdict={res.get('verdict')}")))
        if not fired:
            bad.append(f"control '{label}' did NOT fire")
        del ref, got
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    print()
    return bad


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    rungs = (["dry", "full fast cadence"] if quick else list(RUNGS))
    steps = ((1, 8) if quick else STEP_COUNTS)
    arms = list(ARMS)
    for a in list(ARMS):
        if f"--only={a}" in argv:
            arms = [a]
    if "--controls-only" in argv:
        arms, rungs = [], []

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    print("=" * 78)
    print("OPEN LATERAL BOUNDARIES, STREAMED -- four arms, one geometry")
    print(f"  {NX}x{NY}x{NZ} at dx=12 km, tile {TX}x{TY}, "
          f"N in {tuple(steps)}, real Lambert, FLAT terrain")
    print("=" * 78)

    cfg0 = open_cfg(NX, NY, "dry", "periodic")
    halo = harness.halo_radius(cfg0)
    print(f"  halo {halo} = 10 + 3*{cfg0.time_step_sound}//2, from "
          "harness.halo_radius and NEVER from a sweep")
    for arm in arms:
        c = open_cfg(NX, NY, "dry", arm)
        census = edge_census(c, arm, TX, TY, halo)
        sx, sy = seam_columns(census["specs"], NX, NY)
        cpx, cpy = streaming._periodic_axes(c)
        print(f"  {arm:10s} periodic-plan x={cpx!s:5s} y={cpy!s:5s} "
              f"{census['tiles']} tiles, true-edge count "
              f"{census['by_true_edges']}, "
              f"{len(sx)} x-seams, {len(sy)} y-seams")
    print()

    failures: list[str] = []
    results: list[tuple] = []

    for arm in arms:
        print(f"-- ARM {arm}")
        for rung in rungs:
            cfg = open_cfg(NX, NY, rung, arm)
            try:
                bnd = domain_boundaries(cfg, arm)
            except Exception as exc:                       # noqa: BLE001
                failures.append(f"{arm}/{rung}: boundaries: {exc}")
                print(_line(f"{rung}", False,
                            f"BOUNDARIES RAISED {type(exc).__name__}: {exc}"))
                continue
            try:
                ref = retry_on_oom(monolithic, cfg, arm, steps,
                                   boundaries=bnd)
            except UnstableReference as exc:
                failures.append(f"{arm}/{rung}: {exc}")
                print(_line(f"{rung}", False, f"REFERENCE NON-FINITE: {exc}"))
                cp.get_default_memory_pool().free_all_blocks()
                continue
            except (NotImplementedError, ValueError) as exc:
                failures.append(f"{arm}/{rung}: resident refused: {exc}")
                print(_line(f"{rung}", False,
                            f"RESIDENT REFUSED {type(exc).__name__}: "
                            f"{str(exc)[:90]}"))
                cp.get_default_memory_pool().free_all_blocks()
                continue

            px, py = streaming._periodic_axes(cfg)
            specs = tspec.plan_tiles(NX, NY, TX, TY, halo,
                                     periodic_x=px, periodic_y=py)
            for n in sorted(steps):
                t0 = time.perf_counter()
                report: dict = {}
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        got = retry_on_oom(streamed, cfg, arm, n, TX, TY,
                                           boundaries=bnd, halo=halo,
                                           report=report)
                except Exception as exc:                   # noqa: BLE001
                    failures.append(f"{arm}/{rung}/N={n}: "
                                    f"{type(exc).__name__}: {exc}")
                    print(_line(f"{rung} N={n}", False,
                                f"STREAMED RAISED {type(exc).__name__}: "
                                f"{str(exc)[:80]}"))
                    cp.get_default_memory_pool().free_all_blocks()
                    continue
                res = localise(ref["snapshots"][n], got, specs, halo, NX, NY)
                ok = res["bitexact"] and res["nonfinite"] == 0
                if not ok:
                    failures.append(f"{arm}/{rung}/N={n} not bit-exact")
                mono_fire = ref["fired"][n]
                strm_fire = report.get("fired", {})
                fire_ok = mono_fire == strm_fire
                print(_line(f"{rung} N={n}", ok,
                            f"{res['ntotal']} carriers ndiff={res['ndiff']} "
                            f"nonfinite={res['nonfinite']} "
                            f"{time.perf_counter() - t0:.1f}s"))
                print(f"          fired mono={mono_fire} "
                      f"streamed={strm_fire}"
                      f"{'' if fire_ok else '   <-- CADENCE MISMATCH'}")
                if not fire_ok:
                    failures.append(f"{arm}/{rung}/N={n} cadence mismatch")
                if not ok:
                    print(f"          verdict={res.get('verdict')} "
                          f"worst={res.get('worst_field')} "
                          f"max|d|={res.get('max_abs'):.6g}")
                    print(f"          differing columns "
                          f"{res.get('differing_columns')}/"
                          f"{res.get('columns')} "
                          f"(frac {res.get('fraction'):.4f}), "
                          f"max dist to seam "
                          f"{res.get('max_distance_to_seam')} "
                          f"(halo {halo}), "
                          f"within halo of a seam "
                          f"{res.get('frac_within_halo_of_seam'):.3f}")
                    print(f"          x extent {res.get('x_extent')} "
                          f"y extent {res.get('y_extent')}  "
                          f"first fields {res['differing'][:6]}")
                results.append((arm, rung, n, res))
                del got
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                if not ok:
                    # The FIRST N that differs is the interesting one; the
                    # later ones only report chaos downstream of it.
                    break
            del ref
            cp.get_default_memory_pool().free_all_blocks()
        print()

    if "--no-controls" not in argv:
        failures.extend(_controls(halo))

    print("=" * 78)
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
    else:
        print("every arm bit-exact at every rung and every N")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
