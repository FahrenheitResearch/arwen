"""Tests for the graph-capture machinery and for the deferred health ledger.

The bit-exact end-to-end proof lives in ``test_gate --graph-only``: fourteen
rungs, real geography, ring and shadow, N = 1 / 3 / 8, every digest compared
against a monolithic run, plus the two negative controls that show the cache
key and the scalar replay are load-bearing.  This module tests the pieces the
gate cannot isolate, and every test here is written so that it FAILS if the
mechanism is switched off or subtly weakened rather than merely absent.

The one that matters most is
:func:`test_a_clean_tile_must_not_erase_a_dirty_tile_s_flag`.  The ledger
accumulates a scheme's status word with OR, and OR is not an implementation
detail: a sweep runs many TILES through one ledger, so a store instead of an
accumulate would let tile 3's clean status overwrite tile 1's non-finite one
and the run would continue with a NaN in it and no error.  That failure is
invisible to any bit-exactness test, because a run that should have aborted
and did not still produces a digest.

Run it::

    python -m tilestream.test_graphcap
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------

def test_no_ledger_means_the_historical_blocking_read():
    import cupy as cp

    from gpuwm.core import health_ledger

    status = cp.asarray([0x5], dtype=cp.uint32)
    assert health_ledger.active() is None
    assert health_ledger.read_status(
        status, site="unit", describe=lambda f: None) == 0x5
    assert health_ledger.check_finite(
        cp.asarray([np.nan], dtype=cp.float32), site="unit",
        message="x") is False


def test_a_ledger_defers_the_read_and_the_drain_raises():
    import cupy as cp

    from gpuwm.core import health_ledger

    ledger = health_ledger.HealthLedger()
    seen = []

    def describe(flags):
        seen.append(flags)
        raise FloatingPointError(f"unit site flagged {flags:#x}")

    with health_ledger.deferring(ledger):
        # Deferred, so the caller is told "nothing wrong" and carries on ...
        assert health_ledger.read_status(
            cp.asarray([0x5], dtype=cp.uint32), site="unit",
            describe=describe) == 0
    try:
        ledger.drain()
    except FloatingPointError as exc:
        assert "0x5" in str(exc), exc
    else:                                              # pragma: no cover
        raise AssertionError("the drain did not raise; a deferred failure "
                             "that is never reported is worse than a slow one")
    assert seen == [0x5]


def test_a_clean_tile_must_not_erase_a_dirty_tile_s_flag():
    """THE control for the accumulate.  Fails if ``record`` ever stores.

    A sweep pushes every tile through one ledger.  If tile 1 flags a
    non-finite output and tile 2 is clean, a ledger that STORES the latest
    status word has silently thrown the fault away.  Nothing else in this
    project can see that: the run finishes and produces a digest.
    """
    import cupy as cp

    from gpuwm.core import health_ledger

    ledger = health_ledger.HealthLedger()

    def describe(flags):
        raise FloatingPointError(f"site flagged {flags:#x}")

    with health_ledger.deferring(ledger):
        health_ledger.read_status(cp.asarray([0x2], dtype=cp.uint32),
                                  site="tile", describe=describe)
        health_ledger.read_status(cp.asarray([0x0], dtype=cp.uint32),
                                  site="tile", describe=describe)
    try:
        ledger.drain()
    except FloatingPointError:
        pass
    else:                                              # pragma: no cover
        raise AssertionError(
            "a clean tile erased an earlier tile's fault flag: the ledger is "
            "storing the status word instead of accumulating it")


def test_the_drain_clears_so_a_fault_is_reported_once():
    import cupy as cp

    from gpuwm.core import health_ledger

    ledger = health_ledger.HealthLedger()
    with health_ledger.deferring(ledger):
        health_ledger.read_status(
            cp.asarray([0x1], dtype=cp.uint32), site="unit",
            describe=lambda f: (_ for _ in ()).throw(FloatingPointError("x")))
    try:
        ledger.drain()
    except FloatingPointError:
        pass
    ledger.drain()          # must not raise the same historical failure again


def test_check_finite_still_sees_a_nan_through_the_ledger():
    import cupy as cp

    from gpuwm.core import health_ledger

    ledger = health_ledger.HealthLedger()
    array = cp.zeros((4, 5), dtype=cp.float32)
    array[2, 3] = cp.float32("nan")
    with health_ledger.deferring(ledger):
        assert health_ledger.check_finite(
            array, site="unit", message="field went non-finite") is True
    try:
        ledger.drain()
    except FloatingPointError as exc:
        assert "non-finite" in str(exc)
    else:                                              # pragma: no cover
        raise AssertionError("the deferred finite check lost a NaN")


def test_masked_clear_writes_exactly_what_where_writes():
    """Bit-exactness of the KF expiry rewrite, both ways round.

    The empty-mask case is the one that matters in practice -- nearly every
    step -- and it is also the one an in-place predicated kernel could get
    wrong by writing zeros everywhere.
    """
    import cupy as cp

    from gpuwm.core import health_ledger

    rng = np.random.default_rng(7)
    for fraction in (0.0, 0.3, 1.0):
        base = cp.asarray(rng.standard_normal((6, 5, 4)), dtype=cp.float32)
        flat = cp.asarray(
            (rng.random((5, 4)) < fraction).astype(np.float32))
        mask = (flat[None] != np.float32(0.0))
        want = cp.where(mask, np.float32(0.0), base)
        got = base.copy()
        health_ledger.masked_clear(mask, got)
        assert bool((got == want).all()), (
            f"masked_clear disagrees with cp.where at fraction {fraction}")
        # and a 2-D carrier under the same 3-D mask
        base2 = cp.asarray(rng.standard_normal((5, 4)), dtype=cp.float32)
        want2 = cp.where(mask[0], np.float32(0.0), base2)
        got2 = base2.copy()
        health_ledger.masked_clear(mask, got2)
        assert bool((got2 == want2).all())


# ---------------------------------------------------------------------------
# the scalar carriers a replay owes
# ---------------------------------------------------------------------------

def test_scalar_delta_round_trips_through_a_dict_of_counters():
    from tilestream import graphcap

    before = {"elapsed_seconds": 9.0,
              "call_counts": {"radiation": 2, "ysu": 7},
              "microphysics_updates": 4}
    after = {"elapsed_seconds": 12.0,
             "call_counts": {"radiation": 2, "ysu": 8},
             "microphysics_updates": 5}
    delta = graphcap.scalar_delta(before, after)
    assert graphcap.apply_scalar_delta(before, delta) == after
    # Applied from a DIFFERENT starting clock -- which is the whole reason it
    # is a delta and not the post-step values, because a straight-line loop
    # replays from a clock that keeps moving.
    later = {"elapsed_seconds": 30.0,
             "call_counts": {"radiation": 5, "ysu": 20},
             "microphysics_updates": 9}
    assert graphcap.apply_scalar_delta(later, delta) == {
        "elapsed_seconds": 33.0,
        "call_counts": {"radiation": 5, "ysu": 21},
        "microphysics_updates": 10}


def test_the_host_fingerprint_separates_a_rebind_from_a_drift():
    """Identity moves and structure moves must not read the same.

    ``capture_step`` tolerates the first (a fresh tendency bundle every step,
    which ``bldt=0`` produces and which replays correctly) and refuses the
    second.  If both rendered identically, either every physics rung would be
    refused or a real drift would be accepted.
    """
    from tilestream.graphcap import _fingerprint_value

    class Bundle:
        def __init__(self, n):
            self.ru = np.zeros((n, 3))

    a, b, c = Bundle(4), Bundle(4), Bundle(5)
    assert _fingerprint_value(a) != _fingerprint_value(b)
    assert _fingerprint_value(a, ids=False) == _fingerprint_value(b, ids=False)
    assert _fingerprint_value(a, ids=False) != _fingerprint_value(c, ids=False)


# ---------------------------------------------------------------------------
# capture and replay, end to end on one buffer
# ---------------------------------------------------------------------------

def test_capture_replay_is_bit_exact_on_the_dry_lane():
    import cupy as cp

    from gpuwm.core.dycore import step
    from tilestream import graphcap, harness, physics_inventory

    cfg = harness.make_config(64, 64, 32)
    state = harness.make_state(cfg)
    harness.run_steps(state, cfg, 2)
    snapshot = physics_inventory.snapshot_carriers(state)

    harness.run_steps(state, cfg, 4)
    want = harness.hash_state(state)

    physics_inventory.load_carriers(state, snapshot)
    cp.cuda.runtime.deviceSynchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    stepper = graphcap.GraphStepper(
        cfg, mode="require", reuse="run",
        scalars_fn=physics_inventory.carrier_scalars,
        set_scalars_fn=physics_inventory.set_carrier_scalars)
    kinds = []
    for _ in range(4):
        with stream:
            kinds.append(stepper.run(state, stream))
        stream.synchronize()
    cp.cuda.runtime.deviceSynchronize()
    assert kinds == ["capture", "replay", "replay", "replay"], kinds
    assert harness.hash_state(state) == want, (
        "graph replay changed the answer on the dry lane")
    assert step is not None


def test_an_empty_capture_is_refused():
    """Failure 1 from the module docstring, made to happen.

    Capturing without making the stream current records nothing, and the
    resulting graph launches successfully and does nothing -- which once
    reported a 99.9% speedup.  The node-count floor is what stops that being
    mistaken for a result.
    """
    import cupy as cp

    from tilestream import graphcap

    from tilestream import harness

    cfg = harness.make_config(32, 32, 8)
    state = harness.make_state(cfg)
    stream = cp.cuda.Stream(non_blocking=True)
    try:
        graphcap.capture_step(state, cfg, stream, step_fn=lambda: None,
                              verify_host=False)
    except graphcap.GraphCaptureError as exc:
        assert "not a step" in str(exc), exc
    else:                                              # pragma: no cover
        raise AssertionError(
            "a capture that recorded nothing was accepted; it would launch "
            "successfully, do nothing, and read as a total speedup")


def _run_all():
    fns = [(name, obj) for name, obj in sorted(globals().items())
           if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
        except Exception as exc:                       # pragma: no cover
            import traceback

            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
