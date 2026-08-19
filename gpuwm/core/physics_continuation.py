"""Move driver-held physics continuation state across a nest relocation.

WHY THIS EXISTS (user report, 2026-08-16, moving nests with Kain-
Fritsch: "really weird artifacts").  A discrete relocation carries the
restart layer's serialised STATE (:func:`gpuwm.core.nest_relocation.
relocatable_attrs`) and -- through the route preparers -- the land-
surface continuation fields.  Everything else the physics driver holds
per column was re-initialised from cold on the WHOLE child at every
accepted move: the KF NCA hold timers, the held cumulus rates,
PRATEC/RAINCV, the RAINC/RAINNC precipitation accumulators, the W0AVG
running trigger history.  Measured consequence on a moving KF nest:
convection cut off domain-wide at each move (held heating zeroed
mid-NCA), every column became simultaneously re-eligible (NCA back to
-100), the trigger memory restarted from zero (one 1/TST sample instead
of a converged mean), and the accumulated-precipitation products reset
to zero mid-run.  None of it is KF-only: RAINNC and UP_HELI_MAX reset
the same way.

THE RULE, so the inventory cannot rot: the shift set is DERIVED from
the restart registry, never hand-listed here.  What makes a slot
restart-serialised (``gpuwm.io.restart.SERIALIZED_SCRATCH_SLOTS``) is
exactly the property -- cross-step per-column memory that nothing
recomputes -- that makes it relocation-carried, and the cumulus
adapter's array inventory (``CUMULUS_CALLABLE_ARRAYS``) rides the same
contract.  A slot added to the restart registry tomorrow moves across
relocations the day it is added.  Driver state deliberately NOT
carried: the held PBL/surface/radiation tendencies and their timers,
because every one of them is recomputed from the instantaneous model
state at the scheme's own cadence (surface/PBL every step, radiation at
its next due call, which a rebuilt driver fires immediately) -- there
is no event memory in them to lose.  The held CUMULUS tendencies are
the exception and are rebuilt from the carried raw rates by
:meth:`gpuwm.core.physics.PhysicsDriver.recouple_cumulus_tendencies`
after the move.

GEOMETRY.  The overlap shifts in index space with the same
:class:`~gpuwm.core.nest_relocation.RelocationPlan` window the
serialised-state transplant uses.  The freshly exposed strip takes each
slot's documented COLD value -- new ground has no convection memory and
no accumulated rain -- which is 0 for every slot except ``cu_nca``,
whose cold value is the -100 eligibility sentinel
(gpuwm/core/physics.py driver init; module_cu_kfeta.F:3152-3156).
"""

from __future__ import annotations

import numpy as np

#: Cold-start value per registry slot for freshly exposed ground.  Every
#: slot not named here cold-starts at exactly 0.0 (the driver's own
#: zero-fill); ``cu_nca`` is the one non-zero initialisation the driver
#: performs (physics.py: ``self.cu_nca[...] = -100`` so every column is
#: eligible on the first call, WRF module_cu_kfeta.F:3152-3156).
PHYSICS_CONTINUATION_COLD_VALUES: dict[str, float] = {
    "cu_nca": -100.0,
}

#: The key the cumulus adapter's trigger history travels under.  Not a
#: scratch slot -- it lives on the callable (restart serialises it as
#: ``cumulus/w0avg``) -- so it is namespaced the same way here.
W0AVG_KEY = "cumulus/w0avg"


def continuation_slots() -> tuple[str, ...]:
    """The relocation-carried scratch inventory: the restart registry.

    One list, owned by :mod:`gpuwm.io.restart`, answers both "what does
    a checkpoint carry" and "what does a relocation carry" for
    driver-held per-column continuation state -- the same single-list
    principle :func:`gpuwm.core.nest_relocation.relocatable_attrs`
    applies to the serialised state.
    """
    from gpuwm.io.restart import SERIALIZED_SCRATCH_SLOTS

    return tuple(sorted(SERIALIZED_SCRATCH_SLOTS))


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    if callable(get) and hasattr(value, "__cuda_array_interface__"):
        return np.ascontiguousarray(get())
    return np.ascontiguousarray(np.asarray(value))


def capture_continuation(state, driver) -> dict[str, np.ndarray]:
    """Host copies of every registry slot present on the outgoing child.

    Called while the outgoing child is whole (the runner's
    ``capture_outgoing`` seam -- host staging releases the device arrays
    right after).  Absent slots are simply not captured: a scheme that
    never allocated ``cu_*`` has no cumulus memory to move, and the
    restore side invents nothing.
    """
    captured: dict[str, np.ndarray] = {}
    existing = getattr(state, "existing_scratch", None)
    if callable(existing):
        for slot in continuation_slots():
            value = existing(slot)
            if value is not None:
                captured[slot] = _host(value)
    from gpuwm.io.restart import CUMULUS_CALLABLE_ARRAYS

    adapter = getattr(driver, "cumulus_callable", None)
    for name in sorted(CUMULUS_CALLABLE_ARRAYS):
        value = getattr(adapter, name, None)
        if value is not None:
            captured[f"cumulus/{name}"] = _host(value)
    return captured


def shift_continuation(captured: dict[str, np.ndarray],
                       plan) -> dict[str, np.ndarray]:
    """Index-space shift of every captured array onto the new footprint.

    The overlap is a pure copy through ``plan.window`` -- the same
    arithmetic, and therefore the same cells, as the serialised-state
    transplant -- and the strip is the slot's cold value.  A disjoint
    plan (nothing shared) yields all-cold arrays, which is exactly what
    a brand-new domain would carry.
    """
    shifted: dict[str, np.ndarray] = {}
    for name, value in captured.items():
        cold = np.float32(PHYSICS_CONTINUATION_COLD_VALUES.get(name, 0.0))
        staged = np.full_like(value, cold)
        window = plan.window(value.shape)
        if window is not None:
            (dst_j, src_j), (dst_i, src_i) = window
            staged[..., dst_j, dst_i] = value[..., src_j, src_i]
        shifted[name] = staged
    return shifted


def restore_continuation(state, driver,
                         shifted: dict[str, np.ndarray]) -> dict[str, object]:
    """Write the shifted continuation onto the rebuilt child, in place.

    The driver's ``cu_*``/``mp_*`` members alias the state's canonical
    scratch slots (gpuwm/io/restart.py driver-alias classification), so
    writing ``state.scratch(slot)[...]`` updates the driver's own arrays
    with no second copy.  W0AVG is re-bound the way the restart restore
    binds it: the array attaches to the adapter and ``_history_state``
    points at the NEW state, so the adapter's identity check does not
    re-zero the history on its next due call.
    """
    moved: list[str] = []
    cold_only: list[str] = []
    for slot in continuation_slots():
        staged = shifted.get(slot)
        if staged is None:
            continue
        target = state.scratch(staged.shape, slot)
        if hasattr(target, "__cuda_array_interface__"):
            import cupy as cp

            target[...] = cp.asarray(staged)
        else:
            target[...] = staged
        moved.append(slot)
        if not staged.any():
            cold_only.append(slot)
    w0avg_moved = False
    staged = shifted.get(W0AVG_KEY)
    adapter = getattr(driver, "cumulus_callable", None)
    if staged is not None and adapter is not None:
        current = getattr(adapter, "w0avg", None)
        if (current is not None
                and tuple(current.shape) == tuple(staged.shape)):
            if hasattr(current, "__cuda_array_interface__"):
                import cupy as cp

                current[...] = cp.asarray(staged)
            else:
                current[...] = staged
        else:
            value = staged
            sample = getattr(state, "w", None)
            if sample is not None and hasattr(
                    sample, "__cuda_array_interface__"):
                import cupy as cp

                value = cp.asarray(staged)
            adapter.w0avg = value
        adapter._history_state = state
        # The next due call ADDS a sample rather than double-counting
        # the pre-move one: the recorded time is the outgoing child's,
        # which the rebuilt clock has already passed.
        adapter._history_time = None
        w0avg_moved = True
    return {
        "registry": "gpuwm.io.restart.SERIALIZED_SCRATCH_SLOTS"
                    " + CUMULUS_CALLABLE_ARRAYS",
        "slots_moved": moved,
        "slots_all_cold": cold_only,
        "w0avg_moved": w0avg_moved,
    }


__all__ = [
    "PHYSICS_CONTINUATION_COLD_VALUES", "W0AVG_KEY",
    "capture_continuation", "continuation_slots", "restore_continuation",
    "shift_continuation",
]
