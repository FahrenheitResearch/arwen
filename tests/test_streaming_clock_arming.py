"""THE ARMING TEST: three bound roads, one carrier inventory, on the card.

Bundle-07 review, finding 1 (CRITICAL, #219-not-structurally-closed).  The
ship refusal the review placed on the store-direct prepared-cache route
stands "until arm-3 of the clock test is green on the release line", and
this file is that test.

WHAT THE OTHER GATES DO NOT COVER.  ``tests/test_streaming_clock_bind.py``
proves the state-backed streamed road consumes the bound ``dtbc``
recurrence, but it reaches that road through
``streaming.standalone_domain_builder`` -- which always has a resident
``DomainState`` to derive the binding from.  ``store_domain_builder`` has
none: it calls ``attach(None, ...)``, and the lazy ``domain_clock()`` behind
``make_tile_hook`` returned ``None`` there, latched ``converted[...]``
permanently, and ran every buffer of the forecast on the retired
``elapsed - interval.start`` recurrence.  That is the road the LARGEST
domains take, and the one least likely to have a resident control beside it.
``tests/test_streaming.py`` pins the POLICY (the refusal, the source shape,
the state-less hook binding without a card); nothing anywhere ran the three
roads against each other with real forcing on real tiles.

THE SHAPE, exactly as the review specified it:

* a small specified-boundary case with NONZERO boundary tendencies,
* ``warmup=0`` (production primes carriers directly),
* ``nbuffers=2`` with MORE TILES THAN BUFFERS, so buffers really do change
  tiles and every conversion is re-checked rather than latched once,
* six steps, which crosses one boundary-interval seam,
* THREE BOUND ARMS -- resident, state-backed streamed, store-direct --
  whose carrier inventories must be BYTE-IDENTICAL,
* every converted tile's attachment clock asserted to BE the node's clock
  object, per launch, not merely non-None,
* plus a FOURTH, legacy-elapsed arm that must DIFFER, which is what proves
  the forcing in this fixture is strong enough to expose a dropped binding.
  Without it the three greens above could all be the same zero.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from conftest import requires_gpu
from gpuwm.core import streaming
from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock
from gpuwm.ingest.prepared_store import PreparedStore
from gpuwm.offline_child_run import _child_boundary_clock
from tilestream import physics_inventory as physinv
from tilestream import test_join as join

pytestmark = requires_gpu

#: 96x72 at tile 32 gives 3x3 = 9 tiles against 2 buffers, so a buffer is
#: re-used for several tiles per sweep -- the table-swap path a binding has
#: to survive, and the path the permanent ``converted`` latch hid.
NX, NY, NZ = 96, 72, 33
TILE = 32
NBUFFERS = 2
#: Six steps crosses one lateral-boundary interval seam at this cadence,
#: so the arms are compared across a table swap and not only within one.
NSTEPS = 6


def _cfg():
    return join.join_cfg(NX, NY, nz=NZ, rung="dry")


def _boundaries(cfg):
    """Two DIFFERENT seeds, so the tabulated series has real tendencies.

    A zero-tendency forcing is the way this whole class of test passes for
    the wrong reason: with nothing to be late for, a one-step-late consumer
    and an on-time one produce the same numbers.  ``test_the_legacy_arm_
    differs`` below is the standing check that this has not gone flat.
    """
    state_a, _ = join.build_domain(cfg, seed=join.SEED, warmup=0)
    state_b, _ = join.build_domain(cfg, seed=join.SEED + 1, warmup=0)
    bnd = join.domain_boundaries(cfg, state_a, state_b)
    del state_a, state_b
    cp.get_default_memory_pool().free_all_blocks()
    return bnd


def _clock(cfg, nsteps=NSTEPS):
    """The production integer-tick clock, a fresh instance per arm."""
    return _child_boundary_clock(
        cfg, lbc_interval_seconds=float(join.BDY_SECONDS),
        steps=int(nsteps), output_steps=int(nsteps))


def _drive(stepper, state, cfg, clock, nsteps=NSTEPS):
    """The executor's exact per-step recurrence (``gpuwm/core/clock.py``)."""
    for _ in range(int(nsteps)):
        if clock is not None:
            if clock.lbc_reset_due():
                clock.mark_force()
            clock.prepare_step()
        stepper(state, cfg, refl_10cm_due=False)
        if clock is not None:
            clock.advance()
    cp.cuda.runtime.deviceSynchronize()


def _options():
    return streaming.StreamingOptions(
        mode="on", tile_nx=TILE, tile_ny=TILE, nbuffers=NBUFFERS,
        halo=None, store="host")


class _ClockWitness:
    """Records the clock every tile conversion actually bound.

    The assertion the review asked for is identity, not truthiness: a hook
    that binds *some* clock is exactly what the lazy derivation did before
    it returned None.  So each observation carries the object, and the arms
    below assert ``is`` against the node's own clock.
    """

    def __init__(self):
        self.seen: list = []

    def install(self, monkeypatch):
        import gpuwm.ingest.lateral_bc as lbc

        real = lbc.bind_lateral_boundary_clock
        seen = self.seen

        def spy(state, clock, *args, **kwargs):
            seen.append(clock)
            return real(state, clock, *args, **kwargs)

        monkeypatch.setattr(lbc, "bind_lateral_boundary_clock", spy)
        monkeypatch.setattr(streaming, "bind_lateral_boundary_clock", spy,
                            raising=False)


def _carriers(mapping):
    return {name: join._as_numpy(arr) for name, arr in mapping.items()}


# --------------------------------------------------------------- arm 1
def _arm_resident(cfg, bnd, *, bind):
    from gpuwm.core.dycore import step as dycore_step

    state, _ = join.build_domain(cfg, seed=join.SEED, boundaries=bnd,
                                 warmup=0)
    clock = None
    if bind:
        clock = _clock(cfg)
        bind_lateral_boundary_clock(state, clock)
    _drive(dycore_step, state, cfg, clock)
    out = _carriers(physinv.carrier_inventory(state, None))
    bad = sum(int(np.count_nonzero(~np.isfinite(v))) for v in out.values()
              if v.dtype.kind == "f")
    assert bad == 0, (
        f"the RESIDENT reference has {bad} non-finite cells; there is "
        "nothing for the streamed arms to be compared against")
    return out, state, clock


# --------------------------------------------------------------- arm 2
def _arm_state_backed(cfg, bnd, *, bind):
    """``standalone_domain_builder``: the road with a resident state."""
    state, _ = join.build_domain(cfg, seed=join.SEED, boundaries=bnd,
                                 warmup=0)
    clock = None
    if bind:
        clock = _clock(cfg)
        bind_lateral_boundary_clock(state, clock)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stepper = streaming.make_stepper(
            state, cfg, _options(),
            build=streaming.standalone_domain_builder(
                grid_id=int(cfg.grid_id)))
    assert streaming.is_streaming(stepper), (
        "mode='on' must stream; a resident fallback would compare the "
        "reference against itself")
    _drive(stepper, state, cfg, clock)
    return _carriers(dict(stepper.store)), stepper, clock


# --------------------------------------------------------------- arm 3
def _fresh_bundle(cfg, bnd):
    """A :class:`PreparedStore` with no resident domain behind it.

    The full-domain pinned host store, geography and scalars come from a
    domain that has been moved off the card and NOT YET STEPPED -- which is
    what ``store_from_prepared_cache`` produces slab by slab off a prepared
    cache, and the reason this file can exercise the store-direct BUILDER
    without a multi-gigabyte prepared cache in the test tree.  Harvesting a
    store that had already been swept would hand arm 3 the state at step 6
    and compare two different forecasts, which is a silent way to fail.  The template
    is height-invariant by contract (scheme adapters, vertical coordinate,
    ``ndim < 2`` setup arrays), so a domain-shaped state built by the same
    factory stands in for the slab a prepared cache would carry.
    """
    template, _ = join.build_domain(cfg, seed=join.SEED, boundaries=bnd,
                                    warmup=0)
    source, _ = join.build_domain(cfg, seed=join.SEED, boundaries=bnd,
                                  warmup=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stepper = streaming.make_stepper(
            source, cfg, _options(),
            build=streaming.standalone_domain_builder(
                grid_id=int(cfg.grid_id)))
    assert streaming.is_streaming(stepper), (
        "the harvest arm did not stream, so there is no host store to "
        "hand the store-direct builder")
    return template, PreparedStore(
        store={k: v for k, v in dict(stepper.store).items()},
        geography=dict(stepper._geography or {}),
        scalars=dict(stepper.scalars or {}),
        template=template,
        coord=getattr(template, "coord", None),
        base=getattr(template, "base", None),
        boundaries=bnd,
        missing=(),
        receipt=None,
    )


def _arm_store_direct(cfg, bundle, *, clock, state):
    """``store_domain_builder``: ``attach(None, ...)``, no state anywhere.

    ``clock`` is the POLICY, not a convenience: ``DERIVE_CLOCK`` is refused
    on this road (there is nothing to derive from), a ``DomainClock`` binds,
    and ``None`` asks for the legacy elapsed-seconds semantics on purpose.

    ``state`` is the tree node's state object, exactly as production hands
    it (``steppers_for_tree`` -> ``make_stepper(node.state, ...)``): the
    sweeper checks identity against it and ``publish_store`` binds the store
    onto it for the readers that take a domain.  The BUILDER is what makes
    this road store-direct -- it calls ``attach(None, ...)`` and reads
    geography, tables and template off the bundle -- and it never looks at
    this object.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stepper = streaming.make_stepper(
            state, cfg, _options(),
            build=streaming.store_domain_builder(bundle, clock=clock,
                                                 warmup=0))
    assert streaming.is_streaming(stepper), (
        "the store-direct road must stream; there is no resident fallback "
        "for a domain that was never resident")
    _drive(stepper, state, cfg, clock)
    return _carriers(dict(stepper.store)), stepper


# =====================================================================
def test_three_bound_roads_carry_one_inventory(monkeypatch):
    """ARM 1 == ARM 2 == ARM 3, byte for byte, with the clock bound.

    This is the receipt the bundle-07 ship refusal names.  On the unfixed
    tree arm 3 forces one timestep late for its whole length and differs
    from both of the others, while arms 1 and 2 agree -- which is exactly
    why the pre-existing gate could not see it.
    """
    cfg = _cfg()
    bnd = _boundaries(cfg)

    ref, _resident_state, _c1 = _arm_resident(cfg, bnd, bind=True)

    witness = _ClockWitness()
    witness.install(monkeypatch)
    backed, backed_stepper, clock2 = _arm_state_backed(cfg, bnd, bind=True)
    assert witness.seen, (
        "no tile conversion bound any clock on the state-backed road; the "
        "witness is not observing the seam it claims to")
    assert all(seen is clock2 for seen in witness.seen), (
        "a converted tile bound a clock that is NOT the node's: "
        f"{[type(s).__name__ for s in witness.seen if s is not clock2]}")

    res2 = join.compare(ref, dict(backed_stepper.store))
    assert res2["nonfinite"] == 0
    assert res2["bitexact"], (
        f"ARM 2 (state-backed streamed, bound) differs from ARM 1 "
        f"(resident, bound) on {res2['ndiff']}/{res2['ntotal']} carriers, "
        f"max|d|={res2['max_abs']:.6g}")

    template, bundle = _fresh_bundle(cfg, bnd)
    clock3 = _clock(cfg)
    witness.seen.clear()
    direct, direct_stepper = _arm_store_direct(cfg, bundle, clock=clock3,
                                              state=template)
    assert witness.seen, (
        "no tile conversion bound any clock on the STORE-DIRECT road -- "
        "which is the defect this file exists for: the state-less "
        "attachment derived None and latched it")
    assert all(seen is clock3 for seen in witness.seen), (
        "a store-direct tile bound a clock that is not the node's clock; "
        "the binding is being derived again instead of passed")

    res3 = join.compare(ref, dict(direct_stepper.store))
    assert res3["nonfinite"] == 0
    assert res3["bitexact"], (
        f"ARM 3 (store-direct via store_domain_builder, bound) differs "
        f"from ARM 1 on {res3['ndiff']}/{res3['ntotal']} carriers, "
        f"max|d|={res3['max_abs']:.6g}; the store-direct road is not "
        "consuming the bound dtbc recurrence")

    # And the two streamed roads against each other, so a common bias in
    # the resident reference could not carry both of them.
    assert set(backed) == set(direct), (
        "the two streamed roads produced different carrier inventories: "
        f"{sorted(set(backed) ^ set(direct))}")
    for name in sorted(backed):
        np.testing.assert_array_equal(
            backed[name], direct[name],
            err_msg=f"state-backed and store-direct disagree on {name!r}")


def test_the_legacy_arm_differs():
    """THE ARMING CHECK.  ``clock=None`` is the retired recurrence, asked
    for deliberately, and it MUST differ from the bound resident reference.

    If this ever goes bit-identical the fixture's boundary tendencies have
    gone flat and every assertion in this file is void -- three arms
    agreeing on a forcing that does nothing is not evidence that the
    binding survives.
    """
    cfg = _cfg()
    bnd = _boundaries(cfg)
    ref, _state, _clock1 = _arm_resident(cfg, bnd, bind=True)
    backed, backed_stepper, _c2 = _arm_state_backed(cfg, bnd, bind=True)
    template, bundle = _fresh_bundle(cfg, bnd)
    legacy, _legacy_stepper = _arm_store_direct(cfg, bundle, clock=None,
                                               state=template)
    same = all(np.array_equal(ref[k], legacy[k])
               for k in ref if k in legacy)
    assert not same, (
        "the LEGACY elapsed-seconds store-direct arm is bit-identical to "
        "the bound resident reference, so this fixture cannot see a "
        "dropped Davies binding at all and every green in this file is "
        "meaningless")


def test_deriving_a_clock_from_nothing_is_refused():
    """The policy at the door, on the card, with a real prepared store.

    ``tests/test_streaming.py`` asserts the refusal text exists in the
    source; this asserts the door actually closes when a real store with
    tabulated boundaries arrives with no clock policy at all.
    """
    cfg = _cfg()
    bnd = _boundaries(cfg)
    _backed, backed_stepper, _c = _arm_state_backed(cfg, bnd, bind=True)
    template, bundle = _fresh_bundle(cfg, bnd)
    with pytest.raises(streaming.StreamingRefused) as excinfo:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            streaming.make_stepper(
                template, cfg, _options(),
                build=streaming.store_domain_builder(bundle, warmup=0))
    assert "ONE TIMESTEP LATE" in str(excinfo.value)
