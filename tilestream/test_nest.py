"""NESTING ACROSS A STREAMED PARENT: does FORCE read the domain, or a ghost?

``gpuwm.core.nest.NestCoupler`` couples a FULL parent prognostic field, couples
the FULL child field, runs ``bdy_interp1`` over both, and hangs the resulting
rolling tables on the child.  Every read is off ``node.state`` --
``nest.py:229`` (``parent_state = self.child_node.parent.state``),
``nest.py:309`` and ``nest.py:438``.

For a RESIDENT domain ``node.state`` IS the domain, so that is correct.  For a
STREAMED domain it is not: :func:`gpuwm.core.streaming.attach` copies the
carriers into a store and every subsequent sweep updates THE STORE.  The
``DomainState`` object the model still holds is frozen at the instant of
attach.  Nothing raises, nothing warns and nothing goes non-finite -- the child
is simply forced from the parent's t=0 air for the whole forecast, which looks
exactly like a nest that is running.

This module measures that, in the three positions the feature occupies:

    python -m tilestream.test_nest              # d01 512^2, d02 256^2
    python -m tilestream.test_nest --medium     # d01 320^2, for a 24 GB card
    python -m tilestream.test_nest --quick      # d01 192^2, same legs, 6 steps
    python -m tilestream.test_nest --arena      # the VRAM number only

Exit code 3 means the CARD WAS TAKEN, not that anything failed.  These boxes
are shared and an allocation that loses a race is not a measurement; the
whole run is abandoned rather than printed with an OOM sitting in the FAIL
column where a reader would count it as evidence.

THE THREE LEGS
--------------
``leg 1``  d01 and d02 both RESIDENT.  The control.  If the coupler is wrong
           here it is wrong independently of streaming, which is worth knowing
           on its own.
``leg 2``  d01 STREAMED (host store, tile that DIVIDES the domain, halo from
           ``harness.halo_radius`` and never from a sweep), d02 resident.
           PASS is: the carrier digests of BOTH domains are bit-identical to
           leg 1 and ``coupler.force_count`` is equal.
``leg 3``  the same with ``feedback=1``, so the parent's own prognostics are
           written from OUTSIDE ``dycore.step`` and a streamed parent must
           scatter them back into its store or lose them.

THE NEGATIVE CONTROL, WHICH IS TODAY'S CODE PATH
------------------------------------------------
``store_aware=False`` runs leg 2 with ``force()`` reading ``node.state``
exactly as it does on ``main``.  It MUST differ from leg 1.  If it does not,
the test cannot see the bug and proves nothing -- which is the whole reason it
is run first and printed as a control rather than assumed.

WHY THE COMPARISON IS AGAINST THE STORE AND NOT AGAINST THE STATE
-----------------------------------------------------------------
A streamed domain's truth is ``stepper.store``.  Comparing leg 2's
``node.state`` against leg 1's would compare a frozen snapshot against a
forecast and would "fail" for a reason that has nothing to do with nesting.
Every d01 digest below is taken from the store when d01 streams and from the
state when it does not; d02 is resident in every leg and is always taken from
its state.

CADENCE IS PRINTED, NOT ASSUMED
-------------------------------
``radt_minutes = 12`` and ``cudt_minutes = 5`` at a 9 s parent step put
radiation 80 steps apart and cumulus 33 apart, so a 20-step window fires each
of them exactly ONCE, at the first step -- MEASURED, printed as
``rad=1 cu=1``, and identical on both sides.  Symmetric is enough for a
bit-exactness claim and is NOT enough for coverage, so the ``full fast
cadence`` rung sets ``radt_minutes = 0.15`` (one parent step) and
``cudt_minutes = 0.3``, which fires both many times inside the same window.
Every leg prints its ``call_counts`` -- both sides of every comparison, not
just the reference -- so the reader can see the numbers rather than trust that
they matched, and :func:`_require_fired` turns a zero at a ``_FULL`` rung into
a GATE FAILURE rather than a printed 0 nobody reads.

Printing them is not enough if the print reads the wrong key, which it did.
``_cadence`` reported ``pbl=counts.get("pbl", 0)`` and
:class:`gpuwm.core.physics.PhysicsDriver` has no ``pbl`` counter -- the PBL is
counted as ``ysu`` (physics.py:3445), beside ``sfclay`` and ``noah``.  So this
gate printed a hard ``pbl=0`` on every leg of every rung, including the legs in
which YSU ran on all 20 steps, and the single line a reader would audit to
answer "did the physics fire on both sides" answered "no" for a scheme that
fired every step.  The counters are now named from the driver's own
initialiser and an ABSENT key prints ``?``, because an unmeasured quantity and
a measured zero are different claims.

That is also how this gate caught its own first false result.  The warmup
step every domain takes to allocate the lazily allocated carriers left
``elapsed_seconds`` at 9 s; the resident leg then had it reset to 0 by
``refresh_model_time`` and the streamed leg carried on from 9 s, so radiation
fired twice on one side and once on the other and 117 of 155 d01 carriers
differed.  Nothing to do with nesting.  See :func:`rewind_clock_carriers`.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time

import numpy as np

from gpuwm.core import streaming
from tilestream import driver, gather, harness
from tilestream import physics_inventory as physinv

# --------------------------------------------------------------------------
# the two-domain configuration
# --------------------------------------------------------------------------

#: d01's specified lateral boundary selectors -- the outer domain of a real
#: nested forecast is driven by an external analysis, exactly as in
#: ``tilestream.test_join``.
SPEC_BC = dict(specified=True, open_x=False, open_y=False, nested=False,
               spec_bdy_width=5, spec_zone=1, relax_zone=4, spec_exp=0.0)

#: d02's selectors.  ``nested=True`` is what makes ``dycore.step`` consume the
#: rolling tables ``NestCoupler.force`` attaches (``dycore._boundary_forced``).
NEST_BC = dict(specified=False, open_x=False, open_y=False, nested=True,
               spec_bdy_width=5, spec_zone=1, relax_zone=4, spec_exp=0.0)

_MOIST = dict(moist=True, mp_physics=10, ztop=20000.0)
_FULL = dict(_MOIST, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
             bldt=0.0, sf_surface_physics=2, ra_sw_physics=4,
             ra_lw_physics=4, radt_minutes=12.0, cu_physics=1,
             cudt_minutes=5.0)

RUNGS: dict[str, dict] = {
    "dry": dict(ztop=20000.0),
    "full(real74)+KF": dict(_FULL),
    # radiation every 0.15 min = 9 s = ONE parent step, cumulus every 0.3 min.
    # The point is not realism, it is that both cadences fire many times
    # inside the window on both sides of the comparison.
    "full fast cadence": dict(_FULL, radt_minutes=0.15, cudt_minutes=0.3,
                              bldt=1.0),
}

NZ = 49
SEED = 20_260_731
RATIO = 3

#: The configuration the feature brief names.  d02 covers 256/3 = 85.3 parent
#: cells, so the placement below centres it: 213 + 86 = 299 < 512 - 3, which
#: is what ``nest_interp._donor_maps`` requires for its +-2 SINT stencil.
FULL_SIZE = dict(parent_nx=512, parent_ny=512, child_nx=256, child_ny=256,
                 i_parent_start=213, j_parent_start=213, tile=128)
#: The brief's shape at two thirds the linear size, for a 24 GB card that is
#: being shared: still 4x4 tiles that divide the domain exactly, still 3:1,
#: still 20 parent steps.  320/3 = 106.7 parent cells under the child, centred
#: at 160 - 27 = 133.
MEDIUM_SIZE = dict(parent_nx=320, parent_ny=320, child_nx=160, child_ny=160,
                   i_parent_start=133, j_parent_start=133, tile=80)
#: The same shape, small enough to iterate on.  192/64 = 3 tiles per axis, and
#: 96/3 = 32 parent cells for the child, centred at 96 - 16 = 80.
QUICK_SIZE = dict(parent_nx=192, parent_ny=192, child_nx=96, child_ny=96,
                  i_parent_start=80, j_parent_start=80, tile=64)

PARENT_DX = 9000.0
CHILD_DX = PARENT_DX / RATIO
PARENT_DT = 9.0
CHILD_DT = PARENT_DT / RATIO
PARENT_STEPS = 20
#: d01's own external forcing has to span the whole run and then some, because
#: ``LateralBoundaries.interval_at`` raises outside it.
BDY_SECONDS = 21600.0


def parent_cfg(size, rung="full(real74)+KF", **over):
    kwargs = dict(harness.GEOGRAPHY_OVERRIDES)
    kwargs.update(dx=PARENT_DX, dy=PARENT_DX)
    kwargs.update(SPEC_BC)
    kwargs.update(RUNGS[rung])
    kwargs.update(dt=PARENT_DT, grid_id=1)
    kwargs.update(over)
    return harness.make_config(size["parent_nx"], size["parent_ny"], NZ,
                               periodic=False, **kwargs)


def child_cfg(size, rung="full(real74)+KF", **over):
    kwargs = dict(harness.GEOGRAPHY_OVERRIDES)
    kwargs.update(dx=CHILD_DX, dy=CHILD_DX)
    kwargs.update(NEST_BC)
    kwargs.update(RUNGS[rung])
    kwargs.update(dt=CHILD_DT, grid_id=2)
    kwargs.update(over)
    return harness.make_config(size["child_nx"], size["child_ny"], NZ,
                               periodic=False, **kwargs)


def domain_config(run, *, grid_id, parent_id, ratio, i0, j0):
    """The placement surface ``NestCoupler`` reads, as a plain namespace.

    ``gpuwm.experiment.DomainConfig`` carries a whole experiment's worth of
    provenance that the coupler never touches; ``tests/test_nest_coupler.py``
    already drives the coupler off a ``SimpleNamespace`` with exactly these
    six attributes, and this keeps the gate free of the experiment loader.
    """
    from types import SimpleNamespace

    return SimpleNamespace(grid_id=int(grid_id), parent_id=int(parent_id),
                           parent_grid_ratio=int(ratio),
                           i_parent_start=int(i0), j_parent_start=int(j0),
                           run=run)


# --------------------------------------------------------------------------
# clocks -- the executor's, built by hand
# --------------------------------------------------------------------------

def make_clocks(*, parent_steps=PARENT_STEPS):
    """d01/d02 :class:`DomainClock`s on a 1 Hz tick lattice.

    ``dt = 9`` and ``dt = 3`` are whole seconds, so ``tick_den = 1`` and every
    elapsed value is exact in FP32 -- deliberately, because a run whose
    ``elapsed_seconds`` rounds differently between two legs would fail this
    gate for a reason that has nothing to do with nesting.
    """
    from gpuwm.core.clock import DomainClock, DomainTicks

    run_ticks = int(PARENT_DT) * int(parent_steps)
    never = 10 ** 9
    pspec = DomainTicks(
        grid_id=1, parent_id=0, parent_time_step_ratio=1,
        step_ticks=int(PARENT_DT), dt_fp32=np.float32(PARENT_DT),
        history_ticks=never, restart_ticks=None, radt_ticks=None,
        stepra=None, cudt_ticks=None, stepcu=None, bldt_ticks=None,
        stepbl=None)
    cspec = DomainTicks(
        grid_id=2, parent_id=1, parent_time_step_ratio=RATIO,
        step_ticks=int(CHILD_DT), dt_fp32=np.float32(CHILD_DT),
        history_ticks=never, restart_ticks=None, radt_ticks=None,
        stepra=None, cudt_ticks=None, stepcu=None, bldt_ticks=None,
        stepbl=None)
    return (DomainClock(pspec, tick_den=1, run_ticks=run_ticks),
            DomainClock(cspec, tick_den=1, run_ticks=run_ticks))


# --------------------------------------------------------------------------
# domains
# --------------------------------------------------------------------------

def domain_boundaries(cfg, state_a, state_b, *, seconds=BDY_SECONDS):
    """d01's external forcing, from two genuinely different coupled states.

    Two seeds rather than one snapshot twice, so the time tendency is nonzero
    and ``dtbc`` -- and therefore the carried clock -- actually reaches the
    answer.  Same argument as ``test_join.domain_boundaries``.
    """
    from gpuwm.ingest.lateral_bc import build_state_lateral_boundaries

    return build_state_lateral_boundaries(
        [state_a, state_b], [0.0, float(seconds)],
        spec_bdy_width=int(cfg.spec_bdy_width), spec_zone=int(cfg.spec_zone),
        relax_zone=int(cfg.relax_zone))


def rewind_clock_carriers(state) -> None:
    """Put the scalar carriers back to t = 0 after the warmup step.

    The warmup exists only to allocate the LAZILY allocated carriers
    (Kain-Fritsch's ``cumulus/w0avg`` above all), so that a store sized from
    the state is not missing a field that appears one step later.  Its clock
    is an artefact: a domain the model prepares is at ``elapsed_seconds = 0``
    with zero physics calls, and every cadence test in the tree is a function
    of exactly those two numbers.

    Rewinding matters because the two legs read the clock from DIFFERENT
    places.  A resident domain takes it from ``refresh_model_time``, which
    imposes the executor's :class:`~gpuwm.core.clock.DomainClock` -- zero at
    the start of the run.  A streamed domain takes it from the ``scalars``
    dict captured at attach, which the sweep advances on its own.  Leave the
    warmup's step on the state and the resident leg silently restarts its
    cadence at zero while the streamed leg carries on from 9 s, so radiation
    fires twice on one side and once on the other -- MEASURED, and it is what
    the first run of this gate reported before this function existed.  It has
    nothing to do with nesting, and a gate that let it through would have
    blamed the coupler for it.
    """
    scalars = physinv.carrier_scalars(state)
    zeroed = {"elapsed_seconds": 0.0, "ysu_nan_guard_fires": 0,
              "microphysics_updates": 0}
    if "call_counts" in scalars:
        zeroed["call_counts"] = {k: 0 for k in scalars["call_counts"]}
    physinv.set_carrier_scalars(state, zeroed)


def build_domain(cfg, *, seed, boundaries=None, warmup=1):
    """A prepared resident domain on the real Lambert projection and terrain.

    ``periodic_faces=False``: on a domain with real edges, duplicating the
    closing map-factor faces installs the WEST edge's map factor on the EAST
    edge (see ``test_join.periodic_face_lie``).
    """
    from gpuwm.ingest.lateral_bc import attach_lateral_boundaries

    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo)
    if boundaries is not None:
        attach_lateral_boundaries(state, boundaries)
    if warmup:
        harness.run_steps(state, cfg, int(warmup))
        rewind_clock_carriers(state)
    return state, geo


def build_child(cfg, *, seed):
    """d02, unwarmed.

    A nested child CANNOT take a warmup step before its first FORCE: with
    ``nested=True`` ``dycore.step`` calls the boundary path, which raises
    without an attachment, and the only thing that may legally attach one is
    ``NestCoupler.force``.  The lazily allocated carriers (Kain-Fritsch's
    ``cumulus/w0avg`` above all) therefore appear on the child's FIRST real
    step -- identically in every leg, because every leg does this.
    """
    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo)
    return state, geo


# --------------------------------------------------------------------------
# the streamed d01
# --------------------------------------------------------------------------

def tile_factory(cfg, boundaries0, *, seed=4242, warmup=1):
    """A ``tile_state_factory`` for one d01 tile buffer (test_join's)."""
    from gpuwm.ingest.lateral_bc import attach_lateral_boundaries

    def make(tile_cfg):
        state, _drv = harness.make_physics_state(
            tile_cfg, seed, geography=harness.neutral_geography(tile_cfg))
        if boundaries0 is not None:
            attach_lateral_boundaries(state, boundaries0)
        if warmup:
            harness.run_steps(state, tile_cfg, int(warmup))
        return state

    return make


def stream_parent(state, cfg, *, tile, boundaries, nbuffers=2, halo=None):
    """Attach d01 to the streaming transport and return its stepper."""
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile), tile_ny=int(tile),
        nbuffers=int(nbuffers), halo=halo, store="host")
    decision = streaming.decide(cfg, options)

    def build(prepared_state, run_cfg, dec):
        geo_inv = {k: gather.pinned_copy(v)
                   for k, v in driver.geography_inventory(state).items()}
        per_tile = streaming.tile_boundary_tables(
            boundaries, streaming.tile_specs(run_cfg, dec), seam="zeros")
        factory = tile_factory(run_cfg, per_tile[0])
        return streaming.attach(
            prepared_state, run_cfg, dec, tile_state_factory=factory,
            geography=geo_inv, boundary_tables=per_tile,
            scalars=physinv.carrier_scalars(state), check_geography=False)

    stepper = streaming.make_stepper(state, cfg, options, decision=decision,
                                     build=build)
    assert streaming.is_streaming(stepper)
    return stepper


# --------------------------------------------------------------------------
# the loop -- gpuwm.core.clock.execute_schedule's op order, by hand
# --------------------------------------------------------------------------

def integrate(parent, child, coupler, *, parent_stepper, child_stepper,
              nsteps=PARENT_STEPS, feedback=0):
    """One period per outer iteration: STEP d01, FORCE d02, 3x STEP d02, FEEDBACK.

    Transcribed from :func:`gpuwm.core.clock.execute_schedule`'s op loop
    (clock.py:1008-1060) for the two-domain ratio-3 tree
    :func:`gpuwm.core.clock.build_schedule` expands: ``prepare_step`` before
    the solve and ``advance`` after it, FORCE while the parent leads by
    exactly one parent interval and ``mark_force`` after, FEEDBACK only when
    the clocks are equal.  The final period's F8 stop-time guard is honoured
    the same way the schedule builder honours it: the last period issues no
    FEEDBACK.
    """
    from gpuwm.core.model import FeedbackScratch
    from gpuwm.core.state import refresh_model_time

    scratch = FeedbackScratch()
    if feedback:
        # build_experiment's own initialization transaction (model.py:611-615):
        # WRF's med_nest_initial feeds the child back into the parent BEFORE
        # the first step, not after the first period.
        coupler.feedback_prepare(child, scratch)
        coupler.feedback_commit(child)
        coupler.feedback_finalize(child)
    for outer in range(int(nsteps)):
        refresh_model_time(parent.state, parent.clock, kernel_launch=True)
        parent.clock.prepare_step()
        parent_stepper(parent.state, parent.cfg.run, refl_10cm_due=False)
        parent.clock.advance()
        refresh_model_time(parent.state, parent.clock, after_step=True)

        coupler.force(child)
        child.clock.mark_force()

        for _ in range(RATIO):
            refresh_model_time(child.state, child.clock, kernel_launch=True)
            child.clock.prepare_step()
            child_stepper(child.state, child.cfg.run, refl_10cm_due=False)
            child.clock.advance()
            refresh_model_time(child.state, child.clock, after_step=True)

        if parent.clock.ticks != child.clock.ticks:
            raise RuntimeError(
                f"period {outer}: parent at {parent.clock.ticks} ticks, "
                f"child at {child.clock.ticks}")
        if feedback and not (parent.clock.at_stop_time
                             or child.clock.at_stop_time):
            coupler.feedback_prepare(child, scratch)
            coupler.feedback_commit(child)
            coupler.feedback_finalize(child)


# --------------------------------------------------------------------------
# digests
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


def compare(ref: dict, got: dict) -> dict:
    da, db = digest_arrays(ref), digest_arrays(got)
    differing = sorted(n for n in da if da.get(n) != db.get(n))
    worst = 0.0
    for name in differing:
        a, b = _as_numpy(ref[name]), _as_numpy(got[name])
        if a.dtype.kind == "f" and a.shape == b.shape:
            worst = max(worst, float(np.nanmax(np.abs(
                a.astype(np.float64) - b.astype(np.float64)))))
    nonfinite = sum(int(np.count_nonzero(~np.isfinite(_as_numpy(v))))
                    for v in got.values()
                    if _as_numpy(v).dtype.kind == "f")
    return dict(bitexact=not differing, ndiff=len(differing),
                ntotal=len(da), differing=differing[:8], max_abs=worst,
                nonfinite=nonfinite)


class UnstableReference(RuntimeError):
    """The resident control itself went non-finite: nothing to compare."""


class CardTooBusy(RuntimeError):
    """The card ran out of memory, which says NOTHING about the code.

    These boxes are shared.  An allocation that fails because a neighbour
    took the card is not a result, and the one thing it must never do is
    print as a FAIL beside the legs that really ran -- a reader would count
    it as evidence against the feature.  So it is raised out of every leg
    and becomes exit code 3, which the retry wrapper retries.
    """


def _reraise_if_busy(exc) -> None:
    """Re-raise ``exc`` as :class:`CardTooBusy` when it is an out-of-memory."""
    import cupy as cp

    busy = [cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError]
    driver = getattr(getattr(cp.cuda, "driver", None), "CUDADriverError", None)
    if driver is not None:
        busy.append(driver)
    if isinstance(exc, tuple(busy)):
        raise CardTooBusy(f"{type(exc).__name__}: {exc}") from exc


def _finite_or_raise(arrays, label):
    bad = sum(int(np.count_nonzero(~np.isfinite(_as_numpy(v))))
              for v in arrays.values() if _as_numpy(v).dtype.kind == "f")
    if bad:
        raise UnstableReference(
            f"{label} has {bad} non-finite cells; a gate that compares one "
            "run's NaNs against another's reports agreement for free")
    return arrays


# --------------------------------------------------------------------------
# one leg
# --------------------------------------------------------------------------

class _no_feedback_writeback:
    """Disarm ONLY the feedback write-back, leaving the store-aware reads on.

    The sharp control for leg 3.  ``store_aware=False`` removes both halves
    at once, so it cannot distinguish "the parent was forced from stale air"
    from "the parent's feedback was thrown away"; this removes exactly the
    second.  With it in place ``feedback_commit`` still reads the store and
    still mutates the parent's device arrays -- and the next sweep gathers
    from the store and overwrites every one of those writes.
    """

    def __enter__(self):
        self._real = streaming.commit_to_store
        streaming.commit_to_store = lambda state, attrs: 0
        # nest.py resolves the symbol per call through a function-local
        # import, so patching the module attribute is enough.
        return self

    def __exit__(self, *exc):
        streaming.commit_to_store = self._real
        return False


def build_boundaries(size, rung):
    """d01's external forcing, built ONCE per rung and shared by every leg.

    Two full d01 states have to be alive simultaneously for
    ``build_state_lateral_boundaries`` to difference them, which at
    512x512x49 with the full carrier set is the single largest device
    allocation in this module -- larger than any leg.  Building it inside
    ``leg`` paid that peak six times per rung and, on a shared card, died at
    it (MEASURED: CUDA_ERROR_OUT_OF_MEMORY on a 4090 with 13.6 GiB free).
    The tables themselves are HOST arrays, so holding them across the legs
    costs no VRAM -- and every leg must use the SAME forcing anyway or the
    comparison is between two different experiments.
    """
    import cupy as cp

    pcfg = parent_cfg(size, rung)
    bnd_a, _ = build_domain(pcfg, seed=SEED, warmup=0)
    bnd_b, _ = build_domain(pcfg, seed=SEED + 1, warmup=0)
    bnd = domain_boundaries(pcfg, bnd_a, bnd_b)
    del bnd_a, bnd_b
    cp.get_default_memory_pool().free_all_blocks()
    return bnd


def leg(size, rung, *, stream_d01: bool, feedback: int = 0,
        store_aware: bool = True, writeback: bool = True,
        nsteps=PARENT_STEPS, halo=None, boundaries=None,
        report: dict | None = None) -> dict:
    """Run the two-domain tree once and return everything a verdict needs."""
    import contextlib

    import cupy as cp

    from gpuwm.core.nest import NestCoupler
    from gpuwm.core.dycore import step as dycore_step
    from types import SimpleNamespace

    pcfg = parent_cfg(size, rung)
    ccfg = child_cfg(size, rung)

    if boundaries is None:
        boundaries = build_boundaries(size, rung)
    bnd = boundaries

    pstate, _pgeo = build_domain(pcfg, seed=SEED, boundaries=bnd, warmup=1)
    cstate, _cgeo = build_child(ccfg, seed=SEED + 5)

    pclock, cclock = make_clocks(parent_steps=nsteps)
    parent = SimpleNamespace(
        cfg=domain_config(pcfg, grid_id=1, parent_id=0, ratio=1, i0=1, j0=1),
        state=pstate, clock=pclock, parent=None)
    child = SimpleNamespace(
        cfg=domain_config(ccfg, grid_id=2, parent_id=1, ratio=RATIO,
                          i0=size["i_parent_start"], j0=size["j_parent_start"]),
        state=cstate, clock=cclock, parent=parent)

    stepper = dycore_step
    if stream_d01:
        stepper = stream_parent(pstate, pcfg, tile=size["tile"],
                                boundaries=bnd, halo=halo)
        if not store_aware:
            # THE NEGATIVE CONTROL, and it is today's code path exactly:
            # unpublish the store so ``force()`` and ``feedback_commit``
            # fall back to reading and writing d01's frozen DomainState.
            # Nothing else changes -- same tiling, same halo, same sweep --
            # so a difference below is attributable to this and only this.
            delattr(pstate, streaming._STORE_ATTR)
    coupler = NestCoupler(child, feedback=int(feedback))

    t0 = time.perf_counter()
    with (_no_feedback_writeback() if not writeback
          else contextlib.nullcontext()):
        integrate(parent, child, coupler, parent_stepper=stepper,
                  child_stepper=dycore_step, nsteps=nsteps,
                  feedback=feedback)
    cp.cuda.runtime.deviceSynchronize()
    wall = time.perf_counter() - t0

    d01 = ({name: _as_numpy(arr) for name, arr in stepper.store.items()}
           if stream_d01 else
           {name: _as_numpy(arr)
            for name, arr in physinv.carrier_inventory(pstate, None).items()})
    d02 = {name: _as_numpy(arr)
           for name, arr in physinv.carrier_inventory(cstate, None).items()}
    scalars = (dict(stepper.scalars) if stream_d01
               else physinv.carrier_scalars(pstate))
    out = dict(d01=d01, d02=d02, force_count=int(coupler.force_count),
               feedback_count=int(coupler.feedback_count),
               d01_scalars=scalars,
               d02_scalars=physinv.carrier_scalars(cstate),
               seconds=wall)
    if report is not None:
        report.update(out)
    # ``bnd`` is the caller's and stays alive; everything else is this leg's.
    del pstate, cstate, stepper, coupler, parent, child
    cp.get_default_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------
# the arena number
# --------------------------------------------------------------------------

def arena_slot_bytes(parent_nx, parent_ny, child_nx, child_ny, *,
                     rung="full(real74)+KF", ratio=RATIO):
    """The ``nest_parent_field`` / ``nest_child_field`` device slots, in bytes.

    ``preflight.nest_slot_shapes`` sizes both from
    ``_full_field_capacity``: the LARGEST single full field of the domain,
    which for a moist run is ``w``/``ph`` at ``(nz+1, ny, nx)``.  It is one
    FIELD, not one state -- that distinction decides whether nesting off a
    streamed parent is merely expensive or impossible.
    """
    from gpuwm.core.preflight import nest_slot_shapes

    size = dict(parent_nx=parent_nx, parent_ny=parent_ny,
                child_nx=child_nx, child_ny=child_ny,
                i_parent_start=1, j_parent_start=1, tile=1)
    prun = parent_cfg(size, rung)
    crun = child_cfg(size, rung)
    pcfg = domain_config(prun, grid_id=1, parent_id=0, ratio=1, i0=1, j0=1)
    ccfg = domain_config(crun, grid_id=2, parent_id=1, ratio=ratio,
                         i0=1, j0=1)
    shapes = nest_slot_shapes(ccfg, int(prun.spec_bdy_width), pcfg)
    rolling = sum(math.prod(s) for name, s in shapes.items()
                  if name.startswith("nest_") and "_b" in name
                  and not name.startswith("nest_sint"))
    return {
        "parent_field_bytes": 4 * math.prod(shapes["nest_parent_field"]),
        "child_field_bytes": 4 * math.prod(shapes["nest_child_field"]),
        "rolling_table_bytes": 4 * rolling,
        "total_nest_slot_bytes": 4 * sum(math.prod(s)
                                         for s in shapes.values()),
    }


def measure_arena_on_device(parent_nx, parent_ny, child_nx, child_ny, *,
                            rung="full(real74)+KF"):
    """Allocate the two full-field slots for real and report the delta.

    "It allocated" is not "it runs", but for a slot whose only job is to hold
    one field it is exactly the question: the number below is measured with
    ``cupy.cuda.runtime.memGetInfo`` on both sides of the allocation, because
    under WSL2 ``nvidia-smi``'s used/free is not trustworthy and this project
    has been burned by it.
    """
    import cupy as cp

    predicted = arena_slot_bytes(parent_nx, parent_ny, child_nx, child_ny,
                                 rung=rung)
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()
    free0, total = cp.cuda.runtime.memGetInfo()
    buffers = [
        cp.zeros(predicted["parent_field_bytes"] // 4, dtype=cp.float32),
        cp.zeros(predicted["child_field_bytes"] // 4, dtype=cp.float32),
    ]
    buffers[0][0] = 1.0
    buffers[1][0] = 1.0
    cp.cuda.runtime.deviceSynchronize()
    free1, _ = cp.cuda.runtime.memGetInfo()
    del buffers
    cp.get_default_memory_pool().free_all_blocks()
    return dict(predicted, measured_delta_bytes=int(free0 - free1),
                free_before=int(free0), device_total=int(total))


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def _line(label, ok, detail=""):
    return f"  {'PASS' if ok else 'FAIL':4s}  {label:54s} {detail}"


#: The keys :class:`gpuwm.core.physics.PhysicsDriver` actually counts under
#: (physics.py:1582).  THERE IS NO ``pbl`` KEY -- the PBL is counted as
#: ``ysu``, the surface layer as ``sfclay``, the land surface as ``noah``.
_COUNTER_KEYS = (("rad", "radiation"), ("cu", "cumulus"), ("pbl", "ysu"),
                 ("sfc", "sfclay"), ("lsm", "noah"))

#: What a rung built on ``_FULL`` MUST fire at least once, on both sides of
#: every comparison, or the comparison is between two runs that never ran the
#: schemes the rung is named for.
_MUST_FIRE = ("radiation", "cumulus", "ysu")


def _cadence(scalars):
    """The physics call counts, read from the keys the driver really uses.

    THE FIRST VERSION OF THIS FUNCTION LIED, AND IT IS WORTH SAYING HOW.  It
    reported ``pbl=counts.get("pbl", 0)``, and
    :class:`gpuwm.core.physics.PhysicsDriver` has no ``pbl`` counter: it
    initialises ``radiation``, ``sfclay``, ``noah``, ``ysu``, ``cumulus`` and
    ``cumulus_history`` (physics.py:1582) and increments ``ysu`` for the PBL
    (physics.py:3445).  So the gate printed a hard ``pbl=0`` on every leg of
    every rung -- including legs in which YSU ran on all 20 steps -- and the
    one line a reader would audit to answer "did the physics fire on both
    sides of this comparison" answered "no" for a scheme that fired every
    step.  A cadence report that cannot tell "did not fire" from "I am
    reading the wrong key" is worse than no report, because it is trusted.

    So the labels below come from the driver's own initialiser, and a key
    that is ABSENT prints ``?`` rather than ``0`` -- an unmeasured quantity
    and a measured zero are different claims and must not share a glyph.
    """
    counts = scalars.get("call_counts", {})
    parts = [f"elapsed={scalars.get('elapsed_seconds')}"]
    for label, key in _COUNTER_KEYS:
        parts.append(f"{label}={counts[key] if key in counts else '?'}")
    parts.append(f"mp={scalars.get('microphysics_updates', 0)}")
    return " ".join(parts)


def _require_fired(scalars, label, rung, failures):
    """Refuse to report a full-physics comparison in which nothing fired.

    The recurring false result in this project is a number measured in a
    window where radiation and cumulus never ran: two runs agree bit-for-bit
    because neither of them did the thing under test.  Printing the counts
    is not enough -- printed numbers are not read.  At the ``dry`` rung there
    is nothing to fire and this is skipped; at every rung built on ``_FULL``
    a zero here is a GATE FAILURE, on BOTH sides of every comparison.
    """
    if rung == "dry":
        return True
    counts = scalars.get("call_counts", {})
    dead = [key for key in _MUST_FIRE if int(counts.get(key, 0)) == 0]
    if int(scalars.get("microphysics_updates", 0)) == 0:
        dead.append("microphysics")
    if dead:
        failures.append(
            f"{rung}: {label} never fired {dead} -- a bit-exactness claim "
            "taken in a window where the scheme under test did not run is "
            "not evidence about that scheme")
        return False
    return True


def main(argv=None) -> int:
    """Exit 0 = passed, 1 = a real failure, 3 = the card was taken."""
    try:
        return _main(argv)
    except CardTooBusy as exc:
        print(f"\nINTERRUPTED, not failed: {exc}")
        print("Another process took the card; nothing above is a verdict on "
              "the feature.")
        return 3
    except Exception as exc:                              # noqa: BLE001
        try:
            _reraise_if_busy(exc)
        except CardTooBusy as busy:
            print(f"\nINTERRUPTED, not failed: {busy}")
            return 3
        raise


def _main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    medium = "--medium" in argv
    size = QUICK_SIZE if quick else (MEDIUM_SIZE if medium else FULL_SIZE)
    nsteps = 6 if quick else PARENT_STEPS
    # ``--fast-only`` is not a shortcut, it is the CADENCE COVERAGE run: at
    # radt = 12 min and a 9 s parent step the ordinary rung fires radiation
    # once in the whole window, so a card that only ever has room for one
    # rung should be given this one.
    rungs = ["full(real74)+KF", "full fast cadence"]
    if "--one-rung" in argv:
        rungs = ["full(real74)+KF"]
    if "--fast-only" in argv:
        rungs = ["full fast cadence"]

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")

    if "--arena" in argv:
        for nx in (512, 1024, 2048, 4096):
            got = measure_arena_on_device(nx, nx, 256, 256)
            print(f"  d01 {nx}x{nx}x{NZ}: nest_parent_field "
                  f"{got['parent_field_bytes'] / 2**20:9.1f} MiB  "
                  f"+ nest_child_field "
                  f"{got['child_field_bytes'] / 2**20:7.1f} MiB  "
                  f"measured delta {got['measured_delta_bytes'] / 2**20:9.1f}"
                  f" MiB of {got['device_total'] / 2**30:.1f} GiB")
        return 0

    pcfg0 = parent_cfg(size, "dry")
    halo = harness.halo_radius(pcfg0)
    print("=" * 78)
    print("NEST FORCE/FEEDBACK ACROSS A STREAMED PARENT")
    print(f"  d01 {size['parent_nx']}x{size['parent_ny']}x{NZ} dx "
          f"{PARENT_DX / 1000:g} km specified, dt {PARENT_DT:g} s, "
          f"tile {size['tile']}x{size['tile']} "
          f"({size['parent_nx'] // size['tile']}x"
          f"{size['parent_ny'] // size['tile']} tiles, exact)")
    print(f"  d02 {size['child_nx']}x{size['child_ny']}x{NZ} dx "
          f"{CHILD_DX / 1000:g} km nested {RATIO}:1 at "
          f"({size['i_parent_start']},{size['j_parent_start']}), "
          f"dt {CHILD_DT:g} s")
    print(f"  halo {halo} = 10 + 3*{pcfg0.time_step_sound}//2 from "
          f"harness.halo_radius, never tuned")
    print(f"  {nsteps} parent steps = {nsteps * RATIO} child steps")
    print("=" * 78)

    slots = arena_slot_bytes(size["parent_nx"], size["parent_ny"],
                             size["child_nx"], size["child_ny"])
    print(f"  nest_parent_field {slots['parent_field_bytes'] / 2**20:.1f} MiB,"
          f" nest_child_field {slots['child_field_bytes'] / 2**20:.1f} MiB,"
          f" rolling tables {slots['rolling_table_bytes'] / 2**20:.1f} MiB")
    print()

    failures: list[str] = []
    for rung in rungs:
        print(f"-- RUNG {rung}")
        bnd = build_boundaries(size, rung)
        try:
            ref = leg(size, rung, stream_d01=False, feedback=0,
                      nsteps=nsteps, boundaries=bnd)
        except UnstableReference as exc:
            failures.append(f"{rung}: resident control unusable: {exc}")
            print(_line("leg 1 RESIDENT control", False, str(exc)))
            continue
        _finite_or_raise(ref["d01"], "leg 1 d01")
        _finite_or_raise(ref["d02"], "leg 1 d02")
        d01_bytes = sum(int(a.nbytes) for a in ref["d01"].values())
        print(_line("leg 1 RESIDENT (the control)", True,
                    f"forces={ref['force_count']}, "
                    f"{len(ref['d01'])} d01 carriers, "
                    f"{len(ref['d02'])} d02 carriers, "
                    f"{ref['seconds']:.1f} s"))
        print(f"        d01 whole-domain carrier set "
              f"{d01_bytes / 2**20:.0f} MiB; nest_parent_field is "
              f"{100.0 * slots['parent_field_bytes'] / d01_bytes:.2f}% of it")
        print(f"        d01 cadence  {_cadence(ref['d01_scalars'])}")
        print(f"        d02 cadence  {_cadence(ref['d02_scalars'])}")
        _require_fired(ref["d01_scalars"], "leg 1 d01", rung, failures)
        _require_fired(ref["d02_scalars"], "leg 1 d02", rung, failures)

        # THE NEGATIVE CONTROL FIRST: today's code path must DIFFER.
        bad = leg(size, rung, stream_d01=True, feedback=0, nsteps=nsteps,
                  store_aware=False, boundaries=bnd)
        r1 = compare(ref["d01"], bad["d01"])
        r2 = compare(ref["d02"], bad["d02"])
        fired = (not r1["bitexact"]) or (not r2["bitexact"])
        if not fired:
            failures.append(
                f"{rung}: the negative control did not fire -- force() "
                "ignoring the store changed nothing, so this test cannot "
                "see the defect and proves nothing")
        # ``nonfinite`` is printed because a control that "fires" by going NaN
        # has not shown that the child was forced from stale air, it has shown
        # that the run fell over.  The expected signature is FINITE and large:
        # d01 bit-exact (streaming itself is not the defect) and d02 moved a
        # long way (the child integrated its whole window against t=0 air).
        print(_line("negative control: force() ignores d01's store", fired,
                    f"d01 ndiff={r1['ndiff']}/{r1['ntotal']} "
                    f"max|d|={r1['max_abs']:.4g}; "
                    f"d02 ndiff={r2['ndiff']}/{r2['ntotal']} "
                    f"max|d|={r2['max_abs']:.4g} "
                    f"nonfinite={r1['nonfinite']}/{r2['nonfinite']}"))
        # BOTH SIDES of the control fired their physics, or the difference
        # above is between a forecast and a stall rather than between two
        # forecasts that disagree about the parent.
        print(f"        d01 cadence  {_cadence(bad['d01_scalars'])}")
        print(f"        d02 cadence  {_cadence(bad['d02_scalars'])}")
        _require_fired(bad["d01_scalars"], "negative control d01", rung,
                       failures)
        _require_fired(bad["d02_scalars"], "negative control d02", rung,
                       failures)
        del bad
        cp.get_default_memory_pool().free_all_blocks()

        # LEG 2: streamed d01, store-aware force.
        got = leg(size, rung, stream_d01=True, feedback=0, nsteps=nsteps,
                  boundaries=bnd)
        r1 = compare(ref["d01"], got["d01"])
        r2 = compare(ref["d02"], got["d02"])
        ok = (r1["bitexact"] and r2["bitexact"]
              and r1["nonfinite"] == 0 and r2["nonfinite"] == 0
              and got["force_count"] == ref["force_count"])
        if not ok:
            failures.append(f"{rung} leg 2 (streamed d01, feedback=0)")
        print(_line("leg 2 STREAMED d01, feedback=0", ok,
                    f"forces={got['force_count']}, "
                    f"d01 ndiff={r1['ndiff']}/{r1['ntotal']}, "
                    f"d02 ndiff={r2['ndiff']}/{r2['ntotal']}, "
                    f"max|d|={max(r1['max_abs'], r2['max_abs']):.4g}, "
                    f"{got['seconds']:.1f} s"))
        if not ok:
            print(f"        d01 first differing: {r1['differing']}")
            print(f"        d02 first differing: {r2['differing']}")
        print(f"        d01 cadence  {_cadence(got['d01_scalars'])}")
        print(f"        d02 cadence  {_cadence(got['d02_scalars'])}")
        _require_fired(got["d01_scalars"], "leg 2 d01", rung, failures)
        _require_fired(got["d02_scalars"], "leg 2 d02", rung, failures)
        del got
        cp.get_default_memory_pool().free_all_blocks()

        # LEG 3: feedback=1.
        try:
            ref3 = leg(size, rung, stream_d01=False, feedback=1,
                       nsteps=nsteps, boundaries=bnd)
        except Exception as exc:                          # noqa: BLE001
            _reraise_if_busy(exc)
            failures.append(f"{rung} leg 3 resident feedback=1 refused: {exc}")
            print(_line("leg 3 RESIDENT feedback=1 (control)", False,
                        f"{type(exc).__name__}: {exc}"))
            print()
            continue
        # feedback=1 has to CHANGE something, or leg 3 is a comparison of two
        # runs that both did nothing.
        f1 = compare(ref["d01"], ref3["d01"])
        f2 = compare(ref["d02"], ref3["d02"])
        moved = (not f1["bitexact"]) or (not f2["bitexact"])
        if not moved:
            failures.append(
                f"{rung}: feedback=1 gave the same answer as feedback=0 on "
                "the RESIDENT control, so leg 3 tests nothing")
        print(_line("leg 3 RESIDENT feedback=1 (control)", True,
                    f"forces={ref3['force_count']}, "
                    f"feedbacks={ref3['feedback_count']}, "
                    f"{ref3['seconds']:.1f} s"))
        print(_line("  feedback=1 differs from feedback=0 (it must)", moved,
                    f"d01 ndiff={f1['ndiff']}/{f1['ntotal']} "
                    f"max|d|={f1['max_abs']:.4g}; "
                    f"d02 ndiff={f2['ndiff']}/{f2['ntotal']}"))
        print(f"        d01 cadence  {_cadence(ref3['d01_scalars'])}")
        print(f"        d02 cadence  {_cadence(ref3['d02_scalars'])}")
        _require_fired(ref3["d01_scalars"], "leg 3 control d01", rung,
                       failures)
        _require_fired(ref3["d02_scalars"], "leg 3 control d02", rung,
                       failures)
        # THE SHARP FEEDBACK CONTROL: reads still consult the store, only
        # the write-back is removed, so a difference is attributable to the
        # discarded parent mutation and to nothing else.
        try:
            noback = leg(size, rung, stream_d01=True, feedback=1,
                         nsteps=nsteps, writeback=False, boundaries=bnd)
            n1 = compare(ref3["d01"], noback["d01"])
            n2 = compare(ref3["d02"], noback["d02"])
            back_fired = (not n1["bitexact"]) or (not n2["bitexact"])
            back_detail = (f"d01 ndiff={n1['ndiff']}/{n1['ntotal']} "
                           f"max|d|={n1['max_abs']:.4g}; "
                           f"d02 ndiff={n2['ndiff']}/{n2['ntotal']} "
                           f"max|d|={n2['max_abs']:.4g}")
            del noback
        except Exception as exc:                          # noqa: BLE001
            _reraise_if_busy(exc)
            back_fired = True
            back_detail = f"refused: {type(exc).__name__}: {exc}"
        if not back_fired:
            failures.append(
                f"{rung}: the feedback write-back control did not fire -- "
                "discarding the parent mutation changed nothing, so "
                "feedback=1 is doing no work this test can see")
        print(_line("negative control: feedback write-back removed",
                    back_fired, back_detail))
        cp.get_default_memory_pool().free_all_blocks()
        try:
            got3 = leg(size, rung, stream_d01=True, feedback=1,
                       nsteps=nsteps, boundaries=bnd)
            r1 = compare(ref3["d01"], got3["d01"])
            r2 = compare(ref3["d02"], got3["d02"])
            ok3 = (r1["bitexact"] and r2["bitexact"]
                   and got3["feedback_count"] == ref3["feedback_count"])
            detail = (f"feedbacks={got3['feedback_count']}, "
                      f"d01 ndiff={r1['ndiff']}/{r1['ntotal']}, "
                      f"d02 ndiff={r2['ndiff']}/{r2['ntotal']}, "
                      f"max|d|={max(r1['max_abs'], r2['max_abs']):.4g}")
            cad3 = (_cadence(got3["d01_scalars"]),
                    _cadence(got3["d02_scalars"]))
            _require_fired(got3["d01_scalars"], "leg 3 streamed d01", rung,
                           failures)
            _require_fired(got3["d02_scalars"], "leg 3 streamed d02", rung,
                           failures)
            del got3
        except Exception as exc:                          # noqa: BLE001
            _reraise_if_busy(exc)
            ok3, detail = False, f"refused: {type(exc).__name__}: {exc}"
            cad3 = None
        if not ok3:
            failures.append(f"{rung} leg 3 (streamed d01, feedback=1)")
        print(_line("leg 3 STREAMED d01, feedback=1", ok3, detail))
        if cad3 is not None:
            print(f"        d01 cadence  {cad3[0]}")
            print(f"        d02 cadence  {cad3[1]}")
        del ref, ref3, bnd
        cp.get_default_memory_pool().free_all_blocks()
        print()

    print("=" * 78)
    if failures:
        print(f"NEST GATE FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("NEST GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
