"""A streamed ``[tiles]`` domain consumes Davies forcing on the BOUND clock.

Task #219.  Every production driver-forced root binds a ``DomainClock`` to
its external LBC mirror (``gpuwm.core.model.build_experiment``,
``gpuwm.prepared_domain_tree_forecast``, ``gpuwm.offline_child_run``), which
switches Davies consumers to WRF's post-increment ``dtbc`` recurrence
(``dt .. T_bdy``).  ``gpuwm.core.streaming.make_tile_hook`` converts every
tile buffer to ``attach_streaming_lateral_boundaries``, and until this task
that conversion silently DROPPED the binding: the buffers fell back to the
retired ``elapsed - interval.start`` path (``0 .. T-dt``) and the streamed
run consumed its lateral boundaries one timestep late.  Measured on the
offline child at t+15 min: 41 of 76 wrfout fields differed, W by
0.0043 m/s against a 0.174 m/s field, constant one-step phase error.

WHY THE EXISTING GATE MISSED IT: ``tilestream/test_join.py`` binds no clock,
so BOTH its arms took the retired path and agreed.  These tests bind the
production clock through the PRODUCTION builder
(``streaming.standalone_domain_builder`` -- the exact object
``gpuwm.offline_child_run`` and ``gpuwm go``'s d01 step with), so the blind
configuration can never again be the only coverage:

* ``test_clock_bound_streamed_matches_resident`` is the red-on-revert
  regression: bit-identity with the clock bound on both arms.
* ``test_unbound_arms_still_agree`` pins the legacy compatibility path
  (direct era5/gfs routes without a DomainClock) to bit-identity, so the
  fix provably did not move the unbound semantics.
* ``test_dropped_binding_is_visible`` is the tripwire: the defect's exact
  shape (bound resident, unbound streamed) MUST differ, which proves this
  instrument can see the defect class at all (a zero-tendency forcing
  would quietly disarm both tests above).
* ``test_streamed_refresh_publishes_driver_diagnostics`` is the second
  finding of the same investigation: ``StreamedDomain.refresh_state``
  copied only the carrier manifest, so output-only driver diagnostics
  (``OLR`` at every shipped rung) published as zeros through the
  ``state_frame`` route while the store held the sky the tiles computed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from conftest import requires_gpu
from gpuwm.core import streaming
from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock
from gpuwm.offline_child_run import _child_boundary_clock
from tilestream import physics_inventory as physinv
from tilestream import test_join as join

pytestmark = requires_gpu

#: Small enough for seconds-per-arm, large enough that the 32x32 tiling
#: yields interior seams AND true domain edges in the same plan, with
#: nbuffers=2 < ntiles so buffers really do change tiles (the table-swap
#: path a binding must survive).
NX, NY, NZ = 96, 72, 33
TILE = 32
NBUFFERS = 2
NSTEPS = 6


def _cfg():
    return join.join_cfg(NX, NY, nz=NZ, rung="dry")


def _boundaries(cfg):
    state_a, _ = join.build_domain(cfg, seed=join.SEED, warmup=0)
    state_b, _ = join.build_domain(cfg, seed=join.SEED + 1, warmup=0)
    bnd = join.domain_boundaries(cfg, state_a, state_b)
    del state_a, state_b
    cp.get_default_memory_pool().free_all_blocks()
    return bnd


def _clock(cfg, nsteps=NSTEPS):
    """The production integer-tick clock, one fresh instance per arm."""
    return _child_boundary_clock(
        cfg, lbc_interval_seconds=float(join.BDY_SECONDS),
        steps=int(nsteps), output_steps=int(nsteps))


def _drive(stepper, state, cfg, clock, nsteps=NSTEPS):
    """The executor's exact per-step recurrence (core/clock.py schedule):
    seam reset -> prepare_step -> solve -> advance."""
    for _ in range(int(nsteps)):
        if clock is not None:
            if clock.lbc_reset_due():
                clock.mark_force()
            clock.prepare_step()
        stepper(state, cfg, refl_10cm_due=False)
        if clock is not None:
            clock.advance()
    cp.cuda.runtime.deviceSynchronize()


def _resident(cfg, bnd, *, bind, nsteps=NSTEPS):
    from gpuwm.core.dycore import step as dycore_step

    state, _ = join.build_domain(cfg, seed=join.SEED, boundaries=bnd,
                                 warmup=0)
    clock = None
    if bind:
        clock = _clock(cfg, nsteps)
        bind_lateral_boundary_clock(state, clock)
    _drive(dycore_step, state, cfg, clock, nsteps)
    out = {name: join._as_numpy(arr) for name, arr in
           physinv.carrier_inventory(state, None).items()}
    bad = sum(int(np.count_nonzero(~np.isfinite(v))) for v in out.values()
              if v.dtype.kind == "f")
    assert bad == 0, (
        f"the RESIDENT reference has {bad} non-finite cells; there is "
        "nothing to compare a streamed run against")
    return out, state


def _streamed(cfg, bnd, *, bind, nsteps=NSTEPS):
    """The PRODUCTION streamed arm: standalone_domain_builder, the builder
    ``gpuwm.offline_child_run`` wires and ``gpuwm go``'s d01 shape."""
    state, _ = join.build_domain(cfg, seed=join.SEED, boundaries=bnd,
                                 warmup=0)
    clock = None
    if bind:
        clock = _clock(cfg, nsteps)
        bind_lateral_boundary_clock(state, clock)
    options = streaming.StreamingOptions(
        mode="on", tile_nx=TILE, tile_ny=TILE, nbuffers=NBUFFERS,
        halo=None, store="host")
    with warnings.catch_warnings():
        # The host-store staleness warning is understood here: the arms
        # below read the STORE (or refresh_state), never the stale state.
        warnings.simplefilter("ignore", RuntimeWarning)
        stepper = streaming.make_stepper(
            state, cfg, options,
            build=streaming.standalone_domain_builder(
                grid_id=int(cfg.grid_id)))
    assert streaming.is_streaming(stepper), (
        "mode='on' must stream; a resident fallback would compare the "
        "reference against itself")
    _drive(stepper, state, cfg, clock, nsteps)
    return stepper, state


def test_clock_bound_streamed_matches_resident():
    """RED-ON-REVERT (task #219): with the production clock bound on both
    arms, streamed carriers must be bit-identical to resident ones.  On the
    unfixed tree the tile re-attach drops the binding and the buffers force
    one step late, so this differs on ~everything the forcing reaches."""
    cfg = _cfg()
    bnd = _boundaries(cfg)
    ref, _state = _resident(cfg, bnd, bind=True)
    stepper, _domain = _streamed(cfg, bnd, bind=True)
    res = join.compare(ref, dict(stepper.store))
    assert res["nonfinite"] == 0
    assert res["bitexact"], (
        f"clock-bound streamed run differs from its resident twin on "
        f"{res['ndiff']}/{res['ntotal']} carriers "
        f"(max|d|={res['max_abs']:.6g}, first {res['differing']}); the "
        "tile buffers are not consuming the bound dtbc recurrence")


def test_unbound_arms_still_agree():
    """The legacy compatibility semantics (no DomainClock: direct era5/gfs
    paths) must remain bit-identical streamed vs resident -- the fix may
    not move the unbound path."""
    cfg = _cfg()
    bnd = _boundaries(cfg)
    ref, _state = _resident(cfg, bnd, bind=False)
    stepper, _domain = _streamed(cfg, bnd, bind=False)
    res = join.compare(ref, dict(stepper.store))
    assert res["nonfinite"] == 0
    assert res["bitexact"], (
        f"UNBOUND streamed run stopped matching its resident twin "
        f"({res['ndiff']}/{res['ntotal']} differ); the task-#219 fix must "
        "not change the legacy elapsed-based semantics")


def test_dropped_binding_is_visible():
    """The defect's exact signature must be visible to this instrument: a
    bound resident run against an unbound streamed run MUST differ.  If it
    does not, the forcing tendency is zero and the two positives above are
    disarmed -- passing for the wrong reason."""
    cfg = _cfg()
    bnd = _boundaries(cfg)
    ref, _state = _resident(cfg, bnd, bind=True)
    stepper, _domain = _streamed(cfg, bnd, bind=False)
    res = join.compare(ref, dict(stepper.store))
    assert not res["bitexact"], (
        "bound-resident vs unbound-streamed are bit-identical, so this "
        "test file cannot see the one-step boundary lag at all; the "
        "forcing tendency has gone flat and every positive here is void")


def test_streamed_refresh_publishes_driver_diagnostics():
    """``refresh_state`` must land the store's scattered ``diag/*`` members
    back on the driver, so the ``state_frame`` publish route (the offline
    child) emits the OLR the tiles computed instead of the snapshot's
    zeros.  Bit-for-bit against the resident twin's driver, and nonzero."""
    cfg = join.join_cfg(NX, NY, nz=NZ, rung="full+MYNN+Noah-MP")
    bnd = _boundaries(cfg)
    nsteps = 2
    _ref, resident_state = _resident(cfg, bnd, bind=False, nsteps=nsteps)
    stepper, domain = _streamed(cfg, bnd, bind=False, nsteps=nsteps)
    assert "diag/olr" in stepper.store, (
        "the streamed store carries no diag/olr member; the production "
        "builder stopped wiring diagnostic_inventory")
    resident_olr = cp.asnumpy(resident_state.physics.olr)
    assert np.any(resident_olr != 0.0), (
        "the resident reference produced an all-zero OLR; radiation never "
        "ran and this test is comparing nothing")
    stepper.refresh_state(domain)
    streamed_olr = cp.asnumpy(domain.physics.olr)
    np.testing.assert_array_equal(
        streamed_olr, resident_olr,
        err_msg="refreshed driver OLR differs from the resident twin's; "
                "the streamed state_frame route is publishing wrong "
                "diagnostics")
