"""Physics continuation state across a discrete nest relocation.

THE DEFECT THIS PINS (user report, 2026-08-16: "when using kf it makes
really weird artifacts" on moving domains): a relocation carries the
restart layer's serialised STATE (:func:`relocatable_attrs`) and the
land-surface continuation fields, but every driver-held per-column
physics CONTINUATION array -- the KF NCA hold timers, the held cumulus
rates, PRATEC/RAINCV, the RAINC/RAINNC precipitation accumulators, the
W0AVG trigger history -- was re-initialised from cold on the whole
child at every accepted move.  Convection died domain-wide at each
move, every column became simultaneously re-eligible, and the
accumulated-precipitation products reset to zero mid-run.

The contract under test: the same registry that makes this state
restart-serialised (``gpuwm.io.restart.SERIALIZED_SCRATCH_SLOTS`` and
``CUMULUS_CALLABLE_ARRAYS``) drives the relocation carry, so a slot
added to the restart registry tomorrow moves across relocations the
day it is added.  The overlap shifts in index space exactly like the
serialised state; the freshly exposed strip takes each slot's
documented COLD value (cu_nca = -100 so strip columns are eligible,
everything else 0), because new ground has no convection memory.

Instrument rule: every shift assertion is paired with a control that
fails under a wrong (or missing) shift.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.nest_relocation import Placement, plan_relocation

F32 = np.float32
RATIO = 3
NX = NY = 12
NZ = 4


def _plan(di=2, dj=1):
    return plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=10,
                                 j_parent_start=10),
        placement_to=Placement(grid_id=2, i_parent_start=10 + di,
                               j_parent_start=10 + dj, generation=1),
        parent_grid_ratio=RATIO, child_nx=NX, child_ny=NY)


def _distinct(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(F32)


class _ScratchState:
    """A DomainState stand-in exposing the scratch-slot surface."""

    def __init__(self, arrays=None):
        self._scratch = dict(arrays or {})

    def existing_scratch(self, slot):
        return self._scratch.get(slot)

    def scratch(self, shape, slot, dtype=None):
        shape = tuple(shape)
        buf = self._scratch.get(slot)
        if buf is None:
            buf = np.zeros(shape, dtype=np.dtype(dtype or np.float32))
            self._scratch[slot] = buf
        assert buf.shape == shape
        return buf


# ---------------------------------------------------------------------------
# The inventory is the restart registry, not a hand list
# ---------------------------------------------------------------------------

def test_continuation_inventory_is_the_restart_registry():
    from gpuwm.core.physics_continuation import continuation_slots
    from gpuwm.io.restart import SERIALIZED_SCRATCH_SLOTS

    assert set(continuation_slots()) == set(SERIALIZED_SCRATCH_SLOTS)


def test_cold_values_cover_only_registered_slots():
    from gpuwm.core.physics_continuation import (
        PHYSICS_CONTINUATION_COLD_VALUES, continuation_slots)

    unknown = set(PHYSICS_CONTINUATION_COLD_VALUES) - set(
        continuation_slots())
    assert not unknown, unknown
    # The one non-zero cold value is the KF eligibility sentinel.
    assert PHYSICS_CONTINUATION_COLD_VALUES["cu_nca"] == -100.0


# ---------------------------------------------------------------------------
# Capture -> shift -> restore, both directions
# ---------------------------------------------------------------------------

def _outgoing():
    state = _ScratchState({
        "cu_nca": _distinct((NY, NX), 1),
        "cu_pratec": _distinct((NY, NX), 2),
        "cu_rainc": _distinct((NY, NX), 3),
        "cu_rthcuten": _distinct((NZ, NY, NX), 4),
        "mp_rainnc": _distinct((NY, NX), 5),
        "up_heli_max": _distinct((NY, NX), 6),
    })
    w0avg = _distinct((NZ, NY, NX), 7)
    driver = SimpleNamespace(cumulus_callable=SimpleNamespace(
        w0avg=w0avg, _history_state=state, _history_time=123.0))
    return state, driver


def test_capture_shift_restore_moves_the_overlap_bitwise():
    from gpuwm.core.physics_continuation import (
        capture_continuation, restore_continuation, shift_continuation)

    state, driver = _outgoing()
    plan = _plan(di=2, dj=1)
    shift_i, shift_j = plan.shift_i, plan.shift_j

    captured = capture_continuation(state, driver)
    shifted = shift_continuation(captured, plan)

    new_state = _ScratchState()
    new_adapter = SimpleNamespace(w0avg=None, _history_state=None,
                                  _history_time=None)
    new_driver = SimpleNamespace(cumulus_callable=new_adapter)
    receipt = restore_continuation(new_state, new_driver, shifted)

    for slot in ("cu_nca", "cu_pratec", "cu_rainc", "cu_rthcuten",
                 "mp_rainnc", "up_heli_max"):
        old = state.existing_scratch(slot)
        new = new_state.existing_scratch(slot)
        assert new is not None, slot
        window = plan.window(old.shape)
        (dst_j, src_j), (dst_i, src_i) = window
        # Overlap: bitwise the shifted outgoing values.
        assert np.array_equal(new[..., dst_j, dst_i],
                              old[..., src_j, src_i]), slot
        # Control -- an unshifted copy must NOT satisfy the check
        # (the outgoing arrays are dense random, so any wrong shift
        # differs somewhere).
        assert not np.array_equal(new, old), slot
    assert sorted(receipt["slots_moved"]) == sorted(
        ("cu_nca", "cu_pratec", "cu_rainc", "cu_rthcuten",
         "mp_rainnc", "up_heli_max"))

    # W0AVG rides along and binds to the NEW state so the adapter's
    # identity check does not re-zero it on the next due call.
    old_w = driver.cumulus_callable.w0avg
    window = plan.window(old_w.shape)
    (dst_j, src_j), (dst_i, src_i) = window
    assert np.array_equal(new_adapter.w0avg[..., dst_j, dst_i],
                          old_w[..., src_j, src_i])
    assert new_adapter._history_state is new_state
    assert receipt["w0avg_moved"] is True


def test_strip_takes_the_cold_value_not_zero_garbage():
    from gpuwm.core.physics_continuation import (
        capture_continuation, restore_continuation, shift_continuation)

    state, driver = _outgoing()
    plan = _plan(di=2, dj=1)
    shifted = shift_continuation(capture_continuation(state, driver), plan)
    new_state = _ScratchState()
    restore_continuation(
        new_state,
        SimpleNamespace(cumulus_callable=SimpleNamespace(
            w0avg=None, _history_state=None, _history_time=None)),
        shifted)

    nca = new_state.existing_scratch("cu_nca")
    window = plan.window(nca.shape)
    (dst_j, _), (dst_i, _) = window
    strip = np.ones(nca.shape, dtype=bool)
    strip[..., dst_j, dst_i] = False
    # Fresh ground has no convection memory: eligible immediately, like
    # a cold start (physics.py seeds NCA = -100), and zero accumulation.
    assert np.all(nca[strip] == F32(-100.0))
    assert np.all(new_state.existing_scratch("cu_rainc")[strip] == 0.0)
    assert np.all(new_state.existing_scratch("mp_rainnc")[strip] == 0.0)
    rth = new_state.existing_scratch("cu_rthcuten")
    strip3 = np.broadcast_to(strip, rth.shape) if strip.ndim == rth.ndim \
        else np.broadcast_to(strip[None], rth.shape)
    assert np.all(rth[strip3] == 0.0)


def test_a_null_move_is_the_identity():
    from gpuwm.core.physics_continuation import (
        capture_continuation, restore_continuation, shift_continuation)

    state, driver = _outgoing()
    plan = _plan(di=0, dj=0)
    shifted = shift_continuation(capture_continuation(state, driver), plan)
    new_state = _ScratchState()
    restore_continuation(
        new_state,
        SimpleNamespace(cumulus_callable=SimpleNamespace(
            w0avg=None, _history_state=None, _history_time=None)),
        shifted)
    for slot in ("cu_nca", "cu_rainc", "cu_rthcuten"):
        assert np.array_equal(new_state.existing_scratch(slot),
                              state.existing_scratch(slot)), slot


def test_absent_slots_and_absent_cumulus_do_not_invent_state():
    """A KF-off outgoing child (no cu_* slots, no w0avg) restores nothing."""
    from gpuwm.core.physics_continuation import (
        capture_continuation, restore_continuation, shift_continuation)

    state = _ScratchState({"mp_rainnc": _distinct((NY, NX), 8)})
    driver = SimpleNamespace(cumulus_callable=None)
    shifted = shift_continuation(capture_continuation(state, driver),
                                 _plan())
    new_state = _ScratchState()
    receipt = restore_continuation(
        new_state, SimpleNamespace(cumulus_callable=None), shifted)
    assert receipt["slots_moved"] == ["mp_rainnc"]
    assert receipt["w0avg_moved"] is False
    assert new_state.existing_scratch("cu_nca") is None


# ---------------------------------------------------------------------------
# The preparer wires the mechanism (the front-door routes get it)
# ---------------------------------------------------------------------------

def _child_dc(i0, j0):
    return SimpleNamespace(
        grid_id=2, i_parent_start=i0, j_parent_start=j0,
        parent_grid_ratio=1, run=SimpleNamespace(nx=6, ny=6))


def _preparer_fixture(monkeypatch):
    from gpuwm.runtime import RealRelocationChildPreparer

    ny = nx = 6
    child_dc = _child_dc(4, 4)
    ground = np.arange(20 * 20, dtype=np.float64).reshape(20, 20)

    def statics_for(dc):
        i0, j0 = int(dc.i_parent_start), int(dc.j_parent_start)
        return {"HGT_M": ground[j0:j0 + ny, i0:i0 + nx].copy(),
                "LANDMASK": np.ones((ny, nx))}

    old_case = SimpleNamespace(static_fields=statics_for(child_dc))
    model = SimpleNamespace(_prepared_by_grid_id={2: old_case},
                            _activation_context={})
    preparer = RealRelocationChildPreparer(
        exp=SimpleNamespace(), data=SimpleNamespace(), model=model)
    monkeypatch.setattr(preparer, "_rebuild_driver", lambda *args: 0.125)

    out_state = _ScratchState({
        "cu_nca": _distinct((ny, nx), 11),
        "cu_rainc": _distinct((ny, nx), 12),
    })
    out_adapter = SimpleNamespace(w0avg=_distinct((3, ny, nx), 13),
                                  _history_state=out_state,
                                  _history_time=60.0)
    out_state.physics = SimpleNamespace(
        fields={"tsk": np.arange(ny * nx, dtype=F32).reshape(ny, nx)},
        cumulus_callable=out_adapter)
    node = SimpleNamespace(cfg=child_dc, state=out_state)

    new_dc = SimpleNamespace(**{**vars(child_dc), "i_parent_start": 6,
                                "j_parent_start": 4})
    new_state = _ScratchState()
    new_state.physics = SimpleNamespace(
        fields={}, cumulus_callable=SimpleNamespace(
            w0avg=None, _history_state=None, _history_time=None))
    initialized = SimpleNamespace(static_fields=statics_for(new_dc),
                                  grid="new-grid", state=new_state)
    return preparer, node, new_dc, initialized, out_state, new_state


def test_preparer_moves_physics_continuation_and_reports_it(monkeypatch):
    preparer, node, new_dc, initialized, out_state, new_state = (
        _preparer_fixture(monkeypatch))
    preparer.capture_outgoing(node)
    preparer(initialized, new_dc, SimpleNamespace())
    receipt = preparer.last_receipt["physics_continuation"]
    assert sorted(receipt["slots_moved"]) == ["cu_nca", "cu_rainc"]
    assert receipt["w0avg_moved"] is True

    # di = +2 parent cells at ratio 1: the overlap of the incoming child
    # equals the outgoing child's shifted in index space.
    old = out_state.existing_scratch("cu_rainc")
    new = new_state.existing_scratch("cu_rainc")
    assert np.array_equal(new[:, :4], old[:, 2:])
    # And the accumulators are NOT reported as re-initialised any more.
    assert preparer.last_receipt["accumulators_reinitialized"] is False


def test_preparer_still_honest_when_the_rebuilt_child_has_no_driver(
        monkeypatch):
    preparer, node, new_dc, initialized, _out, new_state = (
        _preparer_fixture(monkeypatch))
    del new_state.physics
    preparer.capture_outgoing(node)
    preparer(initialized, new_dc, SimpleNamespace())
    receipt = preparer.last_receipt["physics_continuation"]
    assert receipt["restored"] is False
