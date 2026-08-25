"""Move driver-held physics continuation state across a nest relocation.

TWO INVENTORIES LIVE HERE, and they are separate because their fill
rules differ.  The per-column CONTINUATION slots below cold-start on
fresh ground.  The surface-radiation CARRIERS at the end of this module
cannot: a carrier is consumed on every surface step but produced only on
the radiation cadence, so fresh ground has to arrive already carrying a
flux, and it takes the same nearest-same-landmask-class donor fill the
land-surface continuation fields take.

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


# ---------------------------------------------------------------------------
# The surface-radiation carriers
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS (node-2 GPU campaign, 2026-08-24).  A relocation
# rebuilds the moved child's physics driver from cold
# (`gpuwm.runtime.rebuild_child_driver_from_land_state` ->
# `initialize_physics`), which allocates a fresh buffer for every
# radiative carrier and seeds a fresh CarrierContract in which glw and
# swdown are `unwritten`.  Neither the buffers nor the ledger were in
# any carry set, so after every accepted move the moved domain's
# radiation provenance read "nothing has ever written this buffer" and
# the first surface call before the next due radiation call refused:
#
#   CarrierContractError: GLW (downward longwave at the surface, W m-2)
#   has no producer, and Noah (sf_surface_physics=2) is about to consume
#   it at model second 360.
#
# MEASURED, one forecast hour per arm on a real GFS case: relocation off
# with radt = 12 min passed; relocation on with radt = 12 min refused at
# the first move; relocation on with radt = 6 min -- aligned so a
# radiation call fell due on the very step the 360 s move landed on --
# passed with five executed moves.  Under the wizard's shipped defaults
# ANY relocation cadence shorter than the radiation interval refused at
# the first move.  The producer guard is correct; the transplant was
# short by exactly this inventory.
#
# THE RULE, so this inventory cannot rot either: the carried set is
# DERIVED from `radiation_carriers.CONSUMER_CARRIERS`, the same matrix
# the refusal reads to decide what a land-surface scheme eats.  A
# carrier added to that matrix tomorrow moves across relocations the day
# it is added.
#
# WHAT MOVES, in two halves that must both move or neither is worth
# anything.  The BUFFERS shift in index space through the same plan
# window as the serialised state, and the freshly exposed strip takes
# the nearest same-landmask-class donor inside the overlap -- the
# identical `DonorFillPlan` the land-surface continuation fields take,
# because a carrier is exactly that kind of field: consumed every
# surface step, produced only on the radiation cadence.  A zero strip
# would be a fabricated flux and a cold-start strip would be the
# allocation fill the contract exists to refuse.  The LEDGER moves
# VERBATIM -- source and last-write model time both -- because the
# second half of the same guard is a staleness test against
# (radiation cadence + one step), and re-stamping the move time would
# blind it: the transplanted flux really is as old as the outgoing
# child's, and saying so is what keeps "a producer that stopped"
# detectable across a move.


def relocatable_carriers() -> tuple[str, ...]:
    """The surface-radiation carriers a relocation carries.

    The consumer matrix, and nothing hand-listed beside it: whatever a
    land-surface scheme is checked for before it consumes is exactly
    what a relocation has to keep, which is the same single-list
    principle :func:`continuation_slots` and
    :func:`gpuwm.core.nest_relocation.relocatable_attrs` apply.
    """
    from gpuwm.core.radiation_carriers import CONSUMER_CARRIERS

    names: set[str] = set()
    for carriers in CONSUMER_CARRIERS.values():
        names.update(carriers)
    return tuple(sorted(names))


def capture_carriers(driver) -> dict[str, object]:
    """Host copies of the outgoing child's carriers, plus its ledger.

    Called from the preparer's ``capture_outgoing`` seam while the
    outgoing child is still whole.  A carrier the configuration never
    allocated is simply not captured -- a Noah run has no GSW to move --
    and a driver assembled without a contract yields ``None``, which the
    restore side reports rather than papers over.
    """
    fields: dict[str, np.ndarray] = {}
    contract = None
    if driver is not None:
        driver_fields = getattr(driver, "fields", None) or {}
        for name in relocatable_carriers():
            value = driver_fields.get(name)
            if value is not None:
                fields[name] = _host(value)
        carriers = getattr(driver, "carriers", None)
        if carriers is not None:
            contract = carriers.state()
    return {"fields": fields, "contract": contract}


def shift_carriers(captured: dict[str, np.ndarray], plan,
                   fill) -> dict[str, np.ndarray]:
    """Index-space shift of every captured carrier, strip donor-filled.

    ``fill`` is the move's own :class:`~gpuwm.ingest.relocation_init.
    DonorFillPlan` -- the one the land-surface continuation fields
    already ride -- so a carrier and the skin temperature it drives
    arrive on the fresh strip from the SAME donor column.  It is a
    required argument, not an optional one: there is no admissible
    degraded fill for a flux, and a strip left at zero is the fabricated
    measurement the carrier contract exists to refuse.
    """
    if fill is None:
        raise ValueError(
            "shift_carriers requires the move's donor fill plan; a "
            "radiation carrier consumed on the freshly exposed strip has "
            "no cold value -- zero is a fabricated flux and the "
            "allocation fill is what the carrier contract refuses -- so "
            "the strip must come from a real donor column")
    shifted: dict[str, np.ndarray] = {}
    for name, value in captured.items():
        staged = np.zeros_like(value)
        window = plan.window(value.shape)
        if window is None:
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        staged[..., dst_j, dst_i] = value[..., src_j, src_i]
        shifted[name] = fill.apply(staged)
    return shifted


def restore_carriers(driver, shifted: dict[str, np.ndarray],
                     contract) -> dict[str, object]:
    """Write the shifted carriers and their provenance onto the rebuilt child.

    The buffers land in the driver's own ``fields`` mapping, which is
    the same storage the radiation seam writes and the consumption check
    reads, so there is no second copy to disagree with.  The ledger is
    re-established through :meth:`~gpuwm.core.radiation_carriers.
    CarrierContract.restore`, the restart layer's own entry point, with
    the outgoing child's model times intact.
    """
    moved: list[str] = []
    absent: list[str] = []
    fields = getattr(driver, "fields", None)
    fields = {} if fields is None else fields
    for name in relocatable_carriers():
        staged = shifted.get(name)
        if staged is None:
            continue
        target = fields.get(name)
        if target is None:
            absent.append(name)
            continue
        if tuple(target.shape) != tuple(staged.shape):
            from gpuwm.core.nest_relocation import RelocationRefusal

            raise RelocationRefusal(
                f"carrier {name!r} is {tuple(staged.shape)} on the "
                f"outgoing child and {tuple(target.shape)} on the "
                "incoming one; a relocation changes position, never "
                "extent")
        if hasattr(target, "__cuda_array_interface__"):
            import cupy as cp

            target[...] = cp.asarray(staged, dtype=target.dtype)
        else:
            target[...] = staged.astype(target.dtype, copy=False)
        moved.append(name)
    carriers = getattr(driver, "carriers", None)
    restored = False
    if carriers is not None and contract is not None:
        carriers.restore(contract)
        restored = True
    return {
        "matrix": "gpuwm.core.radiation_carriers.CONSUMER_CARRIERS",
        "restored": True,
        "carriers_moved": moved,
        "carriers_absent": absent,
        "ledger_restored": restored,
        "ledger": {} if contract is None else {
            name: dict(row) for name, row in sorted(contract.items())},
    }


__all__ = [
    "PHYSICS_CONTINUATION_COLD_VALUES", "W0AVG_KEY",
    "capture_carriers", "capture_continuation", "continuation_slots",
    "relocatable_carriers", "restore_carriers", "restore_continuation",
    "shift_carriers", "shift_continuation",
]
