"""A wrfout history frame built straight out of the pinned host store.

``gpuwm.io.wrfout`` builds a frame from a resident ``DomainState`` on the
device: ``_device_state_frame`` reads ~15 to 122 live cupy arrays and the
async writer stages them D2H into pinned buffers.  A streamed run has no such
state.  The domain is in pinned host RAM, one halo tile at a time visits the
GPU, and by the time a history frame is due the only device object in
existence is a tile buffer holding somebody else's window.

This module writes the frame anyway, and it does it without a single byte of
device traffic.

WHERE EVERY FIELD OF A FRAME ACTUALLY COMES FROM
------------------------------------------------
Not transcribed -- MEASURED, by :func:`frame_plan`, which builds the real
``_device_state_frame`` on a live state and matches every field's ``id()``
against the carrier manifest and against ``STATE_SETUP_ARRAYS``.  A
transcribed table would drift the first time a scheme published a new
diagnostic; this one cannot, and it classifies a field it has never heard of
as UNAVAILABLE rather than dropping it.  Four classes come out:

``carrier``
    Already in the store, under the restart member name the store keys it
    by.  Served as a ZERO-COPY view of the pinned buffer -- no copy, no
    allocation, no transfer.  MEASURED: 110 of the 122 fields at
    ``full+MYNN+Noah-MP``, 43 of 55 at ``full(real74) +KF``, 34 of 46 at
    ``mp10``, and 5 of 15 dry (a dry frame is mostly base state).
``setup``
    ``MUB``/``HGT``/``ZNU``/``ZNW``, plus the ``PB``/``PHB`` column
    broadcasts and ``P_TOP``.  Input, not state: materialised once by
    :meth:`tilestream.checkpoint.DomainSetup.output_statics` and reused by
    every frame for the whole run.
``derived``
    ``T = (thb + thp) - 300`` and ``P = p - pb``, and ``PSFC`` on a
    physics-free state.  Computed on the host, into destinations allocated
    once -- see THE TWO WAYS TO GET THE DERIVE WRONG below.
``driver-diagnostic``
    Produced by a physics scheme, published to output, and read back by
    NOTHING -- so ``gpuwm/io/restart.py`` classifies it REBUILT and it is not
    a carrier, so ``tilestream.driver.run_tiled`` neither gathers nor
    scatters it, so after a sweep the tile buffer holds whatever the LAST
    tile wrote.  MEASURED: exactly ONE such field at every shipped rung --
    ``OLR`` -- plus ``XKMH``/``XKHH`` under ``cfg.hmix_k_diag`` and the SASE
    flux rows under SASE.  They are not uncomputable: each tile's own
    radiation call gets its own window right, and
    :func:`diagnostic_inventory` scatters them like carriers for 0.082 B/cell
    (``OLR``, one horizontal plane) or 8.0 B/cell (the eddy-viscosity pair,
    two full 3-D fields).  Without the scatter the published field is the
    last tile's, wrong over 3 of 4 tiles at a 2x2 decomposition.

THE TWO WAYS TO GET THE DERIVE WRONG, BOTH MEASURED
---------------------------------------------------
1. **Reassociating the arithmetic.**  The device computes
   ``(thb + thp) - 300``.  Saving a pass by folding the constant into the
   base state -- ``thp + (thb - 300)`` -- is not the same number: float
   addition is not associative, and it changes **26.2% of T's bytes at
   96x80x49, max |delta| = 1.526e-05 K** (§12.1 reports the same 43.9% at
   256^2).  :func:`derive_t` keeps the device's order.

   The trap sits one level down and :func:`tilestream.test_io.
   negative_reassociated_t` measures it: the difference is **exactly zero on
   a state that has taken one dycore step**, because ``thb - 300`` is exact
   for any ``thb`` in [150, 600] (Sterbenz) and a stepped ``thp`` is 100.00%
   on the representable grid at magnitude 300.  A control that warmed its
   state up first -- which is the normal, careful thing to do here -- would
   certify the reassociated derive.
2. **Allocating the destinations per frame.**  ``np.subtract(a, b)`` without
   ``out=`` first-touches fresh pages every frame; MEASURED 2.7x the cost of
   writing into a preallocated buffer, all of it page faulting.
   :class:`StoreFrame` allocates ``T``/``P`` and the ``PSFC`` work plane once.

AND THE THIRD, WHICH IS NOT ARITHMETIC AT ALL
---------------------------------------------
**Field presentation order is load-bearing.**  The frame dict doubles as the
writer's schema, so it fixes the order ``WrfoutWriter`` creates netCDF
variables in, and HDF5 lays its name heap out in creation order.  The same
data in a different order gives a file that is 189 bytes larger and hashes
differently while every variable in it compares equal -- MEASURED, constant
at 189 bytes from 192^2 to 256^2.  :class:`FramePlan` therefore carries the
device frame's own iteration order and :meth:`StoreFrame.fields` yields it.

WHEN A FRAME MAY BE TAKEN
-------------------------
``run_tiled``'s ``write_mode="ring"`` keeps ONE store and every tile reads and
writes it in the same sweep, so a reader that looks mid-sweep sees mixed
generations: MEASURED 1 clean snapshot in 9 tiles, i.e. a consistent window of
zero duration.  ``write_mode="shadow"`` leaves the caller's store untouched
for a whole sweep and then makes it the next sweep's scatter target, which is
9 clean in 9 for one sweep in two -- still shorter than the 7-9 timesteps a
frame takes to write.  A frame is therefore taken at a SWEEP BOUNDARY under
either mode, and that is the only rule this module imposes.  Two ways to spend
it:

* ``overlap=False`` (default) -- serve the carriers as zero-copy views and
  write synchronously between sweeps.  Zero extra host RAM, and the netCDF
  write is on the critical path.  At forecast cadence that write is 0.077% of
  wall time at hourly output and 0.31% at every 60 steps, so this is the
  right default and the "streaming makes output cheaper" question is not
  worth optimising.
* ``overlap=True`` -- snapshot the volatile carriers into a second set of
  destinations and hand them to the writer thread, which then genuinely
  overlaps the next sweep (netCDF4/HDF5 release the GIL; Phase 1 measured 97%
  overlap efficiency).  The price is a host memcpy of every volatile field
  plus a second copy of them in the RAM the ring scheme exists to conserve.
  ``T`` and ``P`` are free either way: they are derived into their own
  destinations, which IS a snapshot.

MEASURED FRAME COST, mp10 (46 fields), AGAINST THE RESIDENT PATH
----------------------------------------------------------------
The resident path is what ``AsyncDomainWrfoutWriter.submit`` does: device
derive, then D2H into RECYCLED pinned destinations on a side stream.  Recycled
matters -- a COLD pinned allocation is 100.48 ms against 0.04 ms recycled, and
timing the cold one is the artefact that produced this project's seventh false
result.  Both paths are timed to the same endpoint, a complete host frame dict
ready for ``WrfoutWriter``, because the netCDF write that follows is
measurably identical for both (§12.3: the two write medians differ by 2.3-6.2%
while their own spreads are 11-51%).

==============  ========  ==========  ==========  ===============
domain          frame     resident    store,      store, overlap
                                      zero-copy   snapshot
==============  ========  ==========  ==========  ===============
384x368x49        542 MB       1.627       0.637            5.027
512x496x49        973 MB       1.471       0.658            4.918
640x624x49       1531 MB       1.572       0.732            5.167
==============  ========  ==========  ==========  ===============

ns per useful cell, minimum of 5 reps, flat in domain size.  **Zero-copy is
2.2-2.4x CHEAPER than the resident path; the overlap snapshot is 3.1-3.4x
DEARER.**  That is the whole trade in two numbers: streaming removes the
transfer for nothing, and can only buy the concurrency back at host memory
bandwidth, which on this box is 17-18 GB/s against PCIe 5.0's 50-57.  Against
a solver step of ~3.3 ns/cell dry and ~19 with physics, even the dear branch
is one step's worth of work per frame.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gpuwm.state_serialization_contract import STATE_SETUP_ARRAYS

from tilestream import physics_inventory as physinv


__all__ = [
    "FramePlan",
    "SOURCE_CARRIER",
    "SOURCE_DERIVED",
    "SOURCE_DRIVER_DIAGNOSTIC",
    "SOURCE_SETUP",
    "SOURCE_ZERO",
    "ShortFrameRefused",
    "StoreFrame",
    "StoreHistoryWriter",
    "derive_psfc",
    "derive_t",
    "diagnostic_inventory",
    "diagnostic_members",
    "frame_plan",
    "scatter_cost",
    "write_frame",
]

SOURCE_CARRIER = "carrier"
SOURCE_SETUP = "setup"
SOURCE_DERIVED = "derived"
SOURCE_ZERO = "structural-zero"
SOURCE_DRIVER_DIAGNOSTIC = "driver-diagnostic"


class ShortFrameRefused(ValueError):
    """A streamed frame would have published fewer rows than a resident one.

    Its own class, not a bare ``ValueError``, because the whole point is
    that a caller can tell this apart from a broken store: the run is
    healthy, the trajectory is bit-identical, and the ONLY thing wrong is
    that the transport was not asked to carry a REBUILT diagnostic.  The
    remedy is a one-line change to the run's ``inventory_fn`` and the
    refusal names it.
    """

#: Frame fields this module derives on the host, with the rule named.  Every
#: other unmatched field is reported UNAVAILABLE rather than guessed at.
_DERIVED_RULES = {
    "T": "(thb + thp) - 300",
    "P": "p - pb",
    "PB": "broadcast(pb)",
    "PHB": "broadcast(phb)",
    "PSFC": "z-extrapolated p (physics-free states only)",
    "P_TOP": "setup scalar",
}

#: WRF history rows gpuwm emits UNCONDITIONALLY and fills with exact zeros
#: when the producing scheme is absent.  ``RAINSH`` is always in this set --
#: gpuwm implements no shallow-cumulus scheme and zero is what WRF writes for
#: ``shcu_physics=0`` -- and the other five join it only when their producer
#: is off, in which case ``PhysicsDriver.output_fields`` hands out a fresh
#: ``_zero_accumulator()``.  When the producer IS running they alias
#: serialized scratch and classify as carriers on their own.  See
#: ``gpuwm/core/physics.py:3510-3530``: the six are core ``misc`` Registry
#: rows with no package gate, so omitting one reads as "this file is broken"
#: to every wrf-python precipitation recipe.
_ZERO_ROWS = frozenset({"RAINC", "RAINSH", "RAINNC", "SNOWNC",
                        "GRAUPELNC", "HAILNC"})


# --------------------------------------------------------------------------
# where each field comes from -- measured, not transcribed
# --------------------------------------------------------------------------

@dataclass
class FramePlan:
    """Every field of a wrfout frame, and where a streamed run gets it.

    ``order`` is the device frame's own iteration order and is what
    :meth:`StoreFrame.fields` yields; see the module docstring on why that is
    load-bearing rather than cosmetic.
    """

    order: tuple[str, ...]
    source: dict[str, str]
    #: ``name -> store key`` for carriers, ``name -> setup attribute`` for
    #: setup fields, ``name -> rule`` for derived ones.
    origin: dict[str, str]
    shape: dict[str, tuple[int, ...]]
    dtype: dict[str, Any]

    def names(self, kind: str) -> tuple[str, ...]:
        return tuple(n for n in self.order if self.source[n] == kind)

    @property
    def unavailable(self) -> tuple[str, ...]:
        return self.names(SOURCE_DRIVER_DIAGNOSTIC)

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for name in self.order:
            counts[self.source[name]] = counts.get(self.source[name], 0) + 1
        parts = ", ".join(f"{counts[k]} {k}" for k in sorted(counts))
        return f"{len(self.order)} frame fields: {parts}"


def frame_plan(state, *, include_diagnostic_pressure: bool = True,
               extra_available=()) -> FramePlan:
    """Classify a wrfout frame's fields against the streamed inventory.

    Built from the REAL ``_device_state_frame`` on a live ``state``, matched
    by object identity against ``physics_inventory.carrier_manifest`` and
    ``STATE_SETUP_ARRAYS``.  Identity is the right test and equality is not:
    two distinct arrays can hold equal values at t=0 and diverge on the first
    step, which would classify a carrier as setup and freeze it for the rest
    of the run.

    ``extra_available`` names store keys a run has arranged to scatter
    beyond the carrier set (see :func:`diagnostic_inventory`); fields served
    by them classify as carriers rather than as unavailable diagnostics.
    """
    from gpuwm.io import wrfout

    fields = wrfout._device_state_frame(
        state, include_diagnostic_pressure=include_diagnostic_pressure)
    carriers = {id(value): key
                for key, value in physinv.carrier_manifest(state).items()}
    extra = {str(key) for key in extra_available}
    setup = {}
    for name in STATE_SETUP_ARRAYS:
        value = getattr(state, name, None)
        if value is not None and hasattr(value, "shape"):
            setup.setdefault(id(value), name)

    source: dict[str, str] = {}
    origin: dict[str, str] = {}
    shape: dict[str, tuple[int, ...]] = {}
    dtype: dict[str, Any] = {}
    for name, value in fields.items():
        shape[name] = tuple(int(s) for s in np.shape(value))
        dtype[name] = np.dtype(getattr(value, "dtype", np.float32))
        key = carriers.get(id(value))
        if key is not None:
            source[name], origin[name] = SOURCE_CARRIER, key
            continue
        attr = setup.get(id(value))
        if attr is not None:
            source[name], origin[name] = SOURCE_SETUP, attr
            continue
        if name in _DERIVED_RULES:
            source[name], origin[name] = SOURCE_DERIVED, _DERIVED_RULES[name]
            continue
        # A field this run has arranged to scatter is a carrier like any
        # other.  The store key comes from WHERE the array lives on the
        # driver, not from the frame name: ``driver.olr`` and ``OLR`` happen
        # to correspond, ``driver.hmix_k_diag['XKMH']`` and ``XKMH`` do not.
        diagnostic_path = _diagnostic_origin(state, value)
        for diag_key in (f"diag/{name}", _diagnostic_key(diagnostic_path)):
            if diag_key is not None and diag_key in extra:
                source[name], origin[name] = SOURCE_CARRIER, diag_key
                break
        if name in source:
            continue
        if name in _ZERO_ROWS:
            # Assert rather than assume: the classification says "the driver
            # handed out a fresh _zero_accumulator() because its producer is
            # off", and a nonzero array here would mean the producer IS
            # running and the field should have matched a carrier.
            if np.any(np.asarray(_host(value))):
                raise ValueError(
                    f"frame field {name} is nonzero but matched no carrier "
                    "and no setup array; it is being produced by something "
                    "the streaming inventory does not carry")
            source[name], origin[name] = SOURCE_ZERO, "zeros"
            continue
        source[name] = SOURCE_DRIVER_DIAGNOSTIC
        origin[name] = diagnostic_path
    return FramePlan(order=tuple(fields), source=source, origin=origin,
                     shape=shape, dtype=dtype)


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _diagnostic_key(path: str) -> str | None:
    """``diag/...`` store key for a ``driver.<attr>[...]`` origin path.

    The inverse of :func:`diagnostic_inventory`'s naming, kept beside it so
    the two cannot drift apart.
    """
    if not path.startswith("driver."):
        return None
    rest = path[len("driver."):]
    if rest.endswith("]") and "[" in rest:
        attr, _, key = rest.partition("[")
        return f"diag/{attr}/{key[:-1].strip(chr(39)).strip(chr(34))}"
    return f"diag/{rest}"


def _diagnostic_origin(state, value) -> str:
    """Name the driver attribute an unavailable frame field came out of."""
    driver = getattr(state, "physics", None)
    if driver is None:
        return "unknown"
    for attr in physinv.OUTPUT_ONLY_DRIVER_ATTRS:
        held = getattr(driver, attr, None)
        if held is value:
            return f"driver.{attr}"
        if isinstance(held, Mapping):
            for key, item in held.items():
                if item is value:
                    return f"driver.{attr}[{key!r}]"
    return "unknown"


# --------------------------------------------------------------------------
# making the unavailable ones available
# --------------------------------------------------------------------------

#: Output-only frame rows whose driver buffer is NOT what a step writes.
#:
#: ``PhysicsDriver.output_fields`` fills ``hmix_k_diag`` by COPYING out of the
#: dycore's ``smag_km``/``smag_kh`` scratch at the moment a frame is built
#: (gpuwm/core/physics.py:3551-3565), so the bundle itself carries nothing
#: between steps and scattering it moves zeros -- MEASURED: with the bundle
#: alone in the inventory, ``XKMH``/``XKHH`` came back bit-exact against the
#: monolithic reference because BOTH were exactly zero, which is a control
#: passing for the wrong reason.  The producer is the scratch slot, and that
#: is what has to be scattered.  Keyed by the FRAME name so the frame plan
#: can find it without knowing this table.
_DIAGNOSTIC_PRODUCERS = {
    "XKMH": ("scratch", "smag_km"),
    "XKHH": ("scratch", "smag_kh"),
}


def diagnostic_inventory(obj, names=None) -> dict:
    """``carrier_inventory`` plus the output-only driver diagnostics.

    An ``inventory_fn`` for :func:`tilestream.driver.run_tiled`.  The extra
    members are keyed ``diag/<attr>`` (``diag/<attr>/<row>`` for the
    dict-valued bundles, ``diag/<FRAME NAME>`` for the rows whose producer is
    a dycore scratch slot) and are gathered and scattered exactly like
    carriers, which is the whole point: each tile's own physics computes its
    own window of the diagnostic correctly, and scattering is what stops the
    LAST tile's values from being published for the entire domain.

    They are NOT carriers and adding them here does not make them carriers.
    Nothing reads them back into the trajectory -- ``gpuwm/io/restart.py``
    classifies every one of them REBUILT, with the argument inline -- so a
    run that omits them integrates the identical forecast and simply cannot
    publish those rows.  That is the trade :func:`scatter_cost` prices, and
    the two prices are very different: ``OLR`` is one horizontal plane at
    0.082 B/cell, the eddy-viscosity pair is two full 3-D fields at 8 B/cell.
    """
    inner = getattr(obj, "arrays", None)
    if isinstance(inner, dict):
        obj = inner
    if isinstance(obj, Mapping):
        source: dict = dict(obj)
    else:
        source = dict(physinv.carrier_manifest(obj))
        source.update(diagnostic_members(obj))
    keys = sorted(source) if names is None else list(names)
    return {key: source[key] for key in keys
            if source.get(key) is not None}


def diagnostic_members(obj) -> dict:
    """The ``diag/...`` members ALONE, for an object that holds a driver.

    Split out of :func:`diagnostic_inventory` so a caller composing over a
    DIFFERENT base -- ``gpuwm.core.streaming.diagnostic_inventory`` wraps
    ``streaming_inventory``, which is the restart manifest rather than the
    carrier manifest -- adds exactly the same members by exactly the same
    naming, and the two cannot drift apart.  The naming's inverse is
    :func:`_diagnostic_key`, which is what lets :func:`frame_plan` find a
    scattered diagnostic without knowing this table.

    Returns ``{}`` for a mapping (a store already holds its members under
    their keys) and for an object with no driver.  Never allocates: a
    producer slot that does not exist yet is simply absent, so the caller's
    refusal can still fire rather than being papered over by a fresh array
    of zeros that no step writes.
    """
    inner = getattr(obj, "arrays", None)
    if isinstance(inner, dict):
        obj = inner
    if isinstance(obj, Mapping):
        return {}
    members: dict = {}
    driver = getattr(obj, "physics", None)
    for attr in physinv.OUTPUT_ONLY_DRIVER_ATTRS:
        held = getattr(driver, attr, None)
        if held is None:
            continue
        if isinstance(held, Mapping):
            for key, item in held.items():
                producer = _producer_array(obj, key)
                if producer is not None:
                    members[f"diag/{key}"] = producer
                elif hasattr(item, "shape"):
                    members[f"diag/{attr}/{key}"] = item
        elif hasattr(held, "shape"):
            members[f"diag/{attr}"] = held
    return {key: value for key, value in members.items() if value is not None}


def _producer_array(state, frame_name: str):
    """The array a step actually writes for an output-only frame row."""
    entry = _DIAGNOSTIC_PRODUCERS.get(frame_name)
    if entry is None:
        return None
    kind, slot = entry
    if kind != "scratch":                             # no other kind yet
        raise KeyError(f"unknown diagnostic producer kind {kind!r}")
    return getattr(state, "existing_scratch", lambda _s: None)(slot)


def scatter_cost(state, cfg) -> dict:
    """What carrying the output-only diagnostics costs, in bytes per cell.

    Returned as ``{'carriers': B/cell, 'diagnostics': B/cell, 'names': ...}``
    so the decision to publish ``OLR`` is a number rather than a preference.
    Every one of them is a horizontal ``(ny, nx)`` plane, so the cost falls
    as ``4/nz`` per cell and is negligible at any real ``nz``.
    """
    carriers = physinv.carrier_manifest(state)
    extended = diagnostic_inventory(state)
    cells = float(cfg.nx * cfg.ny * cfg.nz)
    extra = {k: v for k, v in extended.items() if k not in carriers}
    return {
        "carriers": sum(int(a.nbytes) for a in carriers.values()) / cells,
        "diagnostics": sum(int(a.nbytes) for a in extra.values()) / cells,
        "names": tuple(sorted(extra)),
    }


# --------------------------------------------------------------------------
# the host derives
# --------------------------------------------------------------------------

def derive_t(thb, thp, out):
    """``T = (thb + thp) - 300``, in the device's operation order.

    The order is the whole content of this function.  ``thb + thp`` first,
    then the constant -- which is what ``DomainState.total_theta()`` followed
    by ``- cp.float32(300.0)`` does.  Folding the 300 into the base state
    changes 44% of T's bytes; see the module docstring.
    """
    thb3 = thb if np.ndim(thb) == 3 else np.asarray(thb)[:, None, None]
    np.add(thb3, thp, out=out)
    np.subtract(out, np.float32(300.0), out=out)
    return out


def derive_psfc(phb, php, p, work):
    """WRF's z-extrapolated surface pressure, for a state with no physics.

    ``gpuwm.io.wrfout`` materialises the whole ``(nz+1, ny, nx)`` geopotential
    and then slices three levels out of it.  Only levels 0..2 can reach the
    answer, so this slices FIRST.  That changes the amount of WORK, not the
    arithmetic: the surviving operations and their order are identical, and
    the bytes are asserted equal in :mod:`tilestream.test_io`.
    """
    from gpuwm.core import constants as c

    phb3 = phb if np.ndim(phb) == 3 else np.asarray(phb)[:, None, None]
    zif = np.add(phb3[:3], php[:3], out=work[:3])
    np.divide(zif, np.float32(c.G), out=zif)
    z_mid0 = 0.5 * (zif[0] + zif[1])
    z_mid1 = 0.5 * (zif[1] + zif[2])
    w1 = (zif[0] - z_mid1) / (z_mid0 - z_mid1)
    return w1 * p[0] + (np.float32(1.0) - w1) * p[1]


# --------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------

class StoreFrame:
    """A reusable wrfout frame over a pinned host store.

    Construct once per run.  Every :meth:`fields` call refreshes the derived
    fields in place and returns the frame dict; the carriers in it are views
    of the store's own pinned pages unless ``overlap=True``, in which case
    they are copies the writer thread may hold while the solver runs on.

    REFUSES A SHORT FRAME (``require_complete``, default on).  A plan with
    unavailable driver-diagnostic rows describes a frame that is MISSING
    rows a resident run of the same configuration writes, and until 2.2.0
    this class published it anyway -- :meth:`fields` skipped those rows with
    a comment, the writer took the short dict as its schema, and the file
    validated, so the run reported forecast validity PASS and health
    status_bits 0 with ``OLR`` simply absent.  MEASURED at the 2.2.0 cut:
    a 438x350x49 streamed run through ``gpuwm go`` published 73 of the
    resident run's 74 variables at both frames, with no error anywhere.
    A missing output field is not a smaller frame, it is a wrong one, and
    nothing downstream can tell it from a configuration that never asked
    for the field.

    The remedy is in the refusal because it is one call: list the
    diagnostics in the run's ``inventory_fn`` (:func:`diagnostic_inventory`,
    or :func:`gpuwm.core.streaming.diagnostic_inventory` composed over the
    streaming manifest) and the ordinary gather/scatter carries them --
    they are REBUILT rows, so the trajectory is bit-identical either way
    and the only cost is 0.082 B/cell for ``OLR``.  ``require_complete =
    False`` is for the benches that price exactly this trade
    (``tilestream.test_io.case_diagnostic_sources``, whose whole output is
    the list of unavailable rows) and never for a route that publishes.
    """

    def __init__(self, plan: FramePlan, store, setup, cfg, *,
                 overlap: bool = False, require_complete: bool = True):
        self.plan = plan
        self.cfg = cfg
        self.overlap = bool(overlap)
        inner = getattr(store, "arrays", None)
        self._store = inner if isinstance(inner, dict) else store
        self.nz, self.ny, self.nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
        missing = [plan.origin[n] for n in plan.names(SOURCE_CARRIER)
                   if plan.origin[n] not in self._store]
        if missing:
            raise KeyError(
                f"the store does not carry {missing[:8]}, which this frame's "
                "plan says are carriers -- the plan was taken from a "
                "different configuration than the store was sized from")
        unavailable = tuple(plan.unavailable)
        if unavailable and require_complete:
            raise ShortFrameRefused(
                f"this frame plan cannot publish {list(unavailable)}, which "
                "a resident run of the same configuration writes.  They are "
                "output-only driver diagnostics "
                f"({', '.join(plan.origin[n] for n in unavailable)}): the "
                "transport neither gathers nor scatters them, so after a "
                "sweep the tile buffer holds whatever the LAST tile wrote "
                "and the store has nowhere to join the tiles' windows.  A "
                "frame missing rows a resident run writes is not the same "
                "frame, and writing it short is silent.\n"
                "  remedy: carry them -- inventory_fn="
                "gpuwm.core.streaming.diagnostic_inventory(base) -- which "
                "costs 0.082 B/cell for OLR and leaves the trajectory "
                "bit-identical, since every one of these rows is REBUILT "
                "and nothing reads it back.")
        self._statics = setup.output_statics(self.nz, self.ny, self.nx)
        self._thb = np.asarray(setup.thb)
        self._pb = np.asarray(setup.pb)
        self._phb = np.asarray(setup.phb)
        # Allocated once, first-touched here.  Per-frame allocation is 2.7x
        # dearer and all of the difference is page faulting.
        self._dst: dict[str, np.ndarray] = {}
        for name in plan.names(SOURCE_DERIVED):
            if name in ("PB", "PHB", "P_TOP"):
                continue
            self._dst[name] = np.zeros(plan.shape[name], plan.dtype[name])
        for name in plan.names(SOURCE_ZERO):
            self._dst[name] = np.zeros(plan.shape[name], plan.dtype[name])
        if "PSFC" in plan.names(SOURCE_DERIVED):
            self._psfc_work = np.zeros((3, self.ny, self.nx), np.float32)
        self._snapshot: dict[str, np.ndarray] = {}
        if self.overlap:
            for name in plan.names(SOURCE_CARRIER):
                self._snapshot[name] = np.empty(plan.shape[name],
                                                plan.dtype[name])

    @property
    def snapshot_bytes(self) -> int:
        """Host RAM the ``overlap=True`` double buffer costs."""
        return sum(int(a.nbytes) for a in self._snapshot.values())

    def _carrier(self, name):
        array = self._store[self.plan.origin[name]]
        if not self.overlap:
            return array
        dst = self._snapshot[name]
        np.copyto(dst, array)
        return dst

    def fields(self) -> dict[str, np.ndarray]:
        """The frame, in the device frame's field order.

        MUST be called at a sweep boundary.  ``run_tiled``'s ring mode has
        ``src is dst is home`` and every tile reads and writes the store in
        the same sweep, so a frame taken mid-sweep mixes generations -- 8 of
        9 tile boundaries torn, MEASURED.  Nothing here can detect that; the
        caller owns the discipline.
        """
        plan = self.plan
        store = self._store
        out: dict[str, np.ndarray] = {}
        for name in plan.order:
            kind = plan.source[name]
            if kind == SOURCE_CARRIER:
                out[name] = self._carrier(name)
            elif kind == SOURCE_SETUP:
                out[name] = self._statics[name]
            elif kind == SOURCE_ZERO:
                out[name] = self._dst[name]
            elif kind == SOURCE_DERIVED:
                if name in ("PB", "PHB", "P_TOP"):
                    out[name] = self._statics[name]
                elif name == "T":
                    out[name] = derive_t(self._thb, store["state/thp"],
                                         self._dst["T"])
                elif name == "P":
                    pb3 = (self._pb if self._pb.ndim == 3
                           else self._pb[:, None, None])
                    out[name] = np.subtract(store["state/p"], pb3,
                                            out=self._dst["P"])
                elif name == "PSFC":
                    out[name] = derive_psfc(self._phb, store["state/php"],
                                            store["state/p"],
                                            self._psfc_work)
                else:                                  # unreachable by _DERIVED_RULES
                    raise KeyError(f"no host rule for derived field {name}")
            else:
                # driver-diagnostic, and UNREACHABLE for a frame built with
                # require_complete (the default): __init__ refuses the plan
                # rather than letting this skip publish a short frame, which
                # is exactly how OLR went missing from every streamed run
                # up to 2.2.0.  Reachable only for the benches that price
                # the unavailable set on purpose.
                continue
        return out


# --------------------------------------------------------------------------
# putting it on disk
# --------------------------------------------------------------------------

def _make_writer(path, cfg, fields, *, title="gpuwm", global_attrs=None):
    """The production ``WrfoutWriter``, with the frame dict as its schema.

    ``soil_layers`` comes from the config rather than the writer's default:
    WRF's soil axis is the land-surface scheme's own geometry (Noah/Noah-MP
    4, RUC 6 or 9), and a caller that has a ``RunConfig`` and lets the
    default stand is silently declaring "no land-surface scheme".
    """
    from gpuwm.config import soil_layer_count
    from gpuwm.io.wrfout import WrfoutWriter

    return WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                        dx=cfg.dx, dy=cfg.dy, title=title,
                        global_attrs=dict(global_attrs or {}),
                        field_schema=fields,
                        soil_layers=soil_layer_count(cfg))


def write_frame(path, fields, cfg, time_str: str, *, title="gpuwm",
                global_attrs=None) -> Path:
    """One complete, validated, atomically published wrfout frame.

    The same create/write/close/fsync/validate/publish sequence
    ``WrfoutWriter.close()`` performs, spelled out so the caller can see that
    an out-of-core frame is published on exactly the terms a resident one is:
    a temporary file, a completion attribute, a reopen that proves the
    inventory/shapes/Times, then the rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(path, cfg, fields, title=title,
                          global_attrs=global_attrs)
    try:
        writer.write_frame(time_str, fields)
    except BaseException:
        writer.abort()
        raise
    writer.close()
    return path


class StoreHistoryWriter:
    """A background wrfout writer fed from the pinned store.

    netCDF4/HDF5 release the GIL around every HDF5 call, so the writer thread
    genuinely overlaps the solver -- Phase 1 measured 97% efficiency -- and
    that property is worth preserving even though at forecast cadence the
    write is 0.077% to 0.31% of wall time.

    :meth:`submit` MUST be called at a sweep boundary.  With
    ``overlap=False`` it builds the frame, blocks until the previous write
    has drained, and enqueues views of the store -- correct only because the
    caller does not resume stepping until :meth:`drain` returns.  With
    ``overlap=True`` it snapshots the carriers first and returns immediately;
    the solver may step on while the file is written, at the cost of
    :attr:`StoreFrame.snapshot_bytes` of host RAM and one host-memcpy pass.
    """

    def __init__(self, frame: StoreFrame, cfg, *, title="gpuwm",
                 global_attrs=None):
        self.frame = frame
        self.cfg = cfg
        self._title = title
        self._global_attrs = dict(global_attrs or {})
        self._queue: queue.Queue = queue.Queue()
        self._failure: BaseException | None = None
        self._written: list[Path] = []
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="tilestream-wrfout")
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, time_str, fields = item
                try:
                    self._written.append(
                        write_frame(path, fields, self.cfg, time_str,
                                    title=self._title,
                                    global_attrs=self._global_attrs))
                except BaseException as exc:            # noqa: BLE001
                    if self._failure is None:
                        self._failure = exc
            finally:
                self._queue.task_done()

    def _raise_failure(self) -> None:
        if self._failure is not None:
            failure, self._failure = self._failure, None
            raise failure

    def submit(self, path, time_str: str) -> None:
        self._raise_failure()
        if not self.frame.overlap:
            self._queue.join()
        fields = self.frame.fields()
        self._queue.put((Path(path), time_str, fields))
        if not self.frame.overlap:
            self._queue.join()
            self._raise_failure()

    def drain(self) -> None:
        self._queue.join()
        self._raise_failure()

    def close(self) -> tuple[Path, ...]:
        self.drain()
        self._queue.put(None)
        self._thread.join()
        self._raise_failure()
        return tuple(self._written)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self._queue.put(None)
            self._thread.join()
