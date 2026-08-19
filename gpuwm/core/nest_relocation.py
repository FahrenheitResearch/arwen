"""Discrete relocation of a child domain to a new parent-cell placement.

WHAT THIS IS.  A moving nest here is **a sequence of static nests joined by
a re-grid at cycle boundaries** -- not WRF's per-step continuous motion.
The distinction is the whole design.  WRF's mechanism moves the child by
one parent cell inside the integration and therefore invalidates the SINT
donor index/weight tables constantly; this module moves the child at a
cycle boundary, rebuilds the tables **once per placement generation**, and
leaves the per-step numerics exactly as they were.  That is why the
namelist keys guarding the per-step mechanism stay rejected
(``gpuwm.experiment._MOVING_NEST_KEYS``) while this path is admissible:
the premise those keys protect -- "the donor tables are precomputed once at
setup" -- is preserved, with "setup" reading "per placement generation".

THE ARITHMETIC THAT MAKES IT EXACT.  A placement is a whole number of
PARENT cells, so a move of ``di`` parent cells is a move of ``di * ratio``
child cells: an exact shift in child index space.  Write the WRF donor
pickup (``interp_fcn.F:975-985``) for child index ``n`` under placement
``p``::

    ci = p + (n-1)/ratio        ip = mod(n-1, ratio)

For the new placement ``p1`` and child index ``n'``, the cell covering the
same ground under the old placement ``p0`` is ``n = n' + (p1-p0)*ratio``,
and substituting gives ``ci`` and ``ip`` **identical** in both.  Two
consequences follow, and both are load-bearing:

1. the overlap transplant is a pure index-space copy -- no interpolation,
   no resampling, no loss; and
2. anything the child derives from its parent by SINT (its base state, its
   terrain) is *bitwise identical* on the overlap before and after the
   move, because it is the same donor cell with the same sub-cell offset.

Property (2) is checked at every relocation (:func:`donor_alignment_check`)
and is an independent instrument: it is computed by the SINT operator, not
by the shift arithmetic this module implements, so a wrong shift cannot
make it pass.

WHAT MOVES AND WHAT DOES NOT.  The parent is read and never written --
:func:`parent_only_init` fills the child out-of-place, so the relocation
inherits the coupler's registered non-mutating property unchanged.  On the
child, the newly covered strip is filled by the ordinary cold-start path
(a full SINT of the live parent) and the overlap is then stamped bitwise
from the outgoing child, so the child keeps every fine-scale structure it
had integrated over ground it still covers.  The seam between strip and
overlap is real and is what the placement policy's "lead the storm, do not
chase it" bias exists to keep out of the way; it is reported as
``spin_up_cells`` rather than hidden.

WHAT THIS MODULE DELIBERATELY DOES NOT DO.  It does not decide *when* or
*where* to move -- there is no tracker, no scoring, no hysteresis here.
Placement is an input.

WHAT A RESTART ACROSS A MOVE PROMISES: NOTHING.  Ruled by Drew on
2026-08-06 -- *"a restart across a move promises nothing imo its a pure
efficiency experiment"*.  Moving nests exist so that a low-VRAM card can
run a smaller nest that FOLLOWS the weather at a resolution a static nest
of the same cost could never reach; that is the whole value, and it does
not need a resume story.  So a relocation invalidates any restart claim
outright, and this module keeps every existing refusal exactly as it
found it: the prepared-cache key and the tree restart fingerprint still
bind the placement, and moving a nest still invalidates them.

Determinism WITHIN a placement is unaffected and still holds -- not
because anything here protects it, but because nothing here touches the
machinery that provides it.  Across a placement boundary nothing is
promised at all.  :class:`Placement` and :class:`RelocationSegment` are
therefore internal bookkeeping: they record what happened for receipts
and diagnostics, and they are NOT a reproducibility contract.  See
``docs/nest-relocation-identity-decision.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

#: Versioned label for the relocation receipt, hashed into segment ids.
RELOCATION_CONTRACT = "gpuwm-nest-relocation.v1"

#: The ruled posture, spelled once so every receipt that has to state it
#: states the same words.  Stamped into checkpoint headers written after a
#: move and surfaced by any restore that reads one.
RESTART_ACROSS_MOVE_POSTURE = (
    "TOLERATED_EXPERIMENT: a restart across a nest relocation promises "
    "nothing (Drew, 2026-08-06) -- it is a pure efficiency experiment, "
    "never certified, never comparable to an unbroken run")

#: Placement is Layer 2 of the three-layer identity.  These are the
#: :class:`gpuwm.experiment.DomainConfig` fields it owns, and therefore the
#: fields a *placement-independent* domain identity must not contain.
PLACEMENT_FIELDS = ("i_parent_start", "j_parent_start")


class RelocationRefusal(ValueError):
    """A relocation this module will not perform."""


# ---------------------------------------------------------------------------
# Layer 2 of the three-layer identity: Placement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    """Where a child sits in its parent, and which generation that is.

    ``generation`` counts relocations, not steps: it is 0 for a domain that
    has never moved and increments once per accepted move.  It exists so a
    receipt can say "same domain, third position" without a hash over the
    position having to change meaning.
    """

    grid_id: int
    i_parent_start: int
    j_parent_start: int
    generation: int = 0

    def __post_init__(self) -> None:
        for name in ("i_parent_start", "j_parent_start"):
            if int(getattr(self, name)) < 1:
                raise RelocationRefusal(
                    f"{name} is 1-based WRF namelist semantics and must be "
                    f">= 1, got {getattr(self, name)!r}")
        if int(self.generation) < 0:
            raise RelocationRefusal(
                f"placement generation must be >= 0, got {self.generation!r}")

    @property
    def position(self) -> tuple[int, int]:
        return (int(self.i_parent_start), int(self.j_parent_start))

    def same_position(self, other: "Placement") -> bool:
        """Position equality, deliberately ignoring the generation counter."""
        return self.position == other.position

    def to_json(self) -> dict[str, int]:
        return {
            "grid_id": int(self.grid_id),
            "i_parent_start": int(self.i_parent_start),
            "j_parent_start": int(self.j_parent_start),
            "generation": int(self.generation),
        }


def placement_of(domain_config, *, generation: int = 0) -> Placement:
    """Read the placement off a :class:`~gpuwm.experiment.DomainConfig`."""
    return Placement(
        grid_id=int(domain_config.grid_id),
        i_parent_start=int(domain_config.i_parent_start),
        j_parent_start=int(domain_config.j_parent_start),
        generation=int(generation))


def placement_independent_identity(domain_config) -> dict[str, object]:
    """The domain identity with :data:`PLACEMENT_FIELDS` factored out.

    This is Layer 1 ("are these two states the same kind of thing?") of the
    architecture's three-layer identity, computed from the SAME document
    the prepared-cache identity is computed from so the two cannot drift.
    Two domain configs with equal placement-independent identity describe
    the same grid, physics and timestep, and differ at most in where the
    child sits inside its parent.

    NOTHING IN THE TREE CONSUMES THIS, and under the 2026-08-06 ruling
    nothing needs to: a restart across a move promises nothing, so there
    is no resume path that has to be taught to see past the placement.
    The prepared-cache key and the tree restart fingerprint still bind the
    placement and still invalidate on a move, which is now the intended
    behaviour rather than a blocker.  This function exists for
    comparability questions and for the segment bookkeeping below -- "are
    these two configs the same domain in different places?" -- not to
    relax a refusal.  See ``docs/nest-relocation-identity-decision.md``.
    """
    from gpuwm.ingest.prepared_cache import prepared_domain_config_identity

    document = dict(prepared_domain_config_identity(domain_config))
    removed = {name: document.pop(name, None) for name in PLACEMENT_FIELDS}
    return {"identity": document, "placement_fields_removed": removed}


# ---------------------------------------------------------------------------
# Layer 3: the append-only relocation record and segment
# ---------------------------------------------------------------------------

def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class RelocationRecord:
    """One accepted move, recorded for the receipt.

    BOOKKEEPING, NOT A CONTRACT.  A record names its predecessor's digest
    so a sequence of moves reads as an ordered history rather than a set,
    which is what makes a receipt legible after the fact.  It is not a
    replay guarantee: a restart across a move promises nothing
    (2026-08-06 ruling), and no code may treat the presence of a chain as
    permission to resume across one.
    """

    placement_from: Placement
    placement_to: Placement
    parent_grid_ratio: int
    shift_i: int
    shift_j: int
    overlap_cells: int
    child_cells: int
    null_move: bool
    parent_state_sha256: str | None = None
    child_state_sha256_before: str | None = None
    child_state_sha256_after: str | None = None
    donor_alignment: dict | None = None
    predecessor_sha256: str | None = None

    @property
    def overlap_fraction(self) -> float:
        return (0.0 if self.child_cells == 0
                else self.overlap_cells / self.child_cells)

    @property
    def spin_up_cells(self) -> int:
        return int(self.child_cells) - int(self.overlap_cells)

    def to_json(self) -> dict[str, object]:
        return {
            "contract": RELOCATION_CONTRACT,
            "placement_from": self.placement_from.to_json(),
            "placement_to": self.placement_to.to_json(),
            "parent_grid_ratio": int(self.parent_grid_ratio),
            "shift_child_cells": [int(self.shift_i), int(self.shift_j)],
            "overlap_cells": int(self.overlap_cells),
            "child_cells": int(self.child_cells),
            "overlap_fraction": float(self.overlap_fraction),
            "spin_up_cells": int(self.spin_up_cells),
            "null_move": bool(self.null_move),
            "parent_state_sha256": self.parent_state_sha256,
            "child_state_sha256_before": self.child_state_sha256_before,
            "child_state_sha256_after": self.child_state_sha256_after,
            "donor_alignment": self.donor_alignment,
            "predecessor_sha256": self.predecessor_sha256,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.to_json())).hexdigest()


@dataclass(frozen=True)
class RelocationSegment:
    """A base preparation plus the append-only chain of moves off it.

    ``base_identity`` is the placement-independent identity of the domain
    as prepared, and every record after it says how the live domain got
    from that preparation to where it is now.  ``segment_id`` changes with
    every move, so a receipt can name the exact placement history a state
    came out of.

    BOOKKEEPING, NOT A CONTRACT.  This addresses a history; it does not
    promise one can be replayed.  Under the 2026-08-06 ruling a restart
    across a move promises nothing, so nothing may read a matching
    ``segment_id`` as licence to resume across a placement boundary.  What
    it is for is receipts and diagnostics: knowing which moves produced
    the state in front of you.
    """

    base_identity_sha256: str
    records: tuple[RelocationRecord, ...] = ()

    @property
    def generation(self) -> int:
        return len(self.records)

    @property
    def segment_id(self) -> str:
        digest = hashlib.sha256(RELOCATION_CONTRACT.encode("ascii") + b"\0")
        digest.update(self.base_identity_sha256.encode("ascii") + b"\0")
        for record in self.records:
            digest.update(record.sha256.encode("ascii") + b"\0")
        return digest.hexdigest()

    def append(self, record: RelocationRecord) -> "RelocationSegment":
        """Return the successor segment, chaining the predecessor digest."""
        predecessor = (self.records[-1].sha256 if self.records
                       else self.base_identity_sha256)
        linked = replace(record, predecessor_sha256=predecessor)
        return RelocationSegment(
            base_identity_sha256=self.base_identity_sha256,
            records=(*self.records, linked))

    def to_json(self) -> dict[str, object]:
        return {
            "contract": RELOCATION_CONTRACT,
            "base_identity_sha256": self.base_identity_sha256,
            "segment_id": self.segment_id,
            "generation": self.generation,
            "records": [record.to_json() for record in self.records],
        }


def base_segment(domain_config) -> RelocationSegment:
    """The zero-move segment for a domain as prepared."""
    identity = placement_independent_identity(domain_config)
    return RelocationSegment(
        base_identity_sha256=hashlib.sha256(
            _canonical(identity["identity"])).hexdigest())


# ---------------------------------------------------------------------------
# The plan: pure index-space geometry, no device, no state
# ---------------------------------------------------------------------------

def _axis_window(extent: int, shift: int):
    """``dst[d] <- src[d + shift]`` over the valid range, or ``None``.

    Source and destination extents are equal because a relocation changes
    where the child is, never how big it is.
    """
    extent = int(extent)
    shift = int(shift)
    dst_lo = max(0, -shift)
    dst_hi = min(extent, extent - shift)
    if dst_lo >= dst_hi:
        return None
    return slice(dst_lo, dst_hi), slice(dst_lo + shift, dst_hi + shift)


@dataclass(frozen=True)
class RelocationPlan:
    """Everything a relocation needs that does not touch the device."""

    placement_from: Placement
    placement_to: Placement
    parent_grid_ratio: int
    child_nx: int
    child_ny: int
    shift_i: int
    shift_j: int

    @property
    def null_move(self) -> bool:
        return self.shift_i == 0 and self.shift_j == 0

    @property
    def child_cells(self) -> int:
        return int(self.child_nx) * int(self.child_ny)

    @property
    def overlap_cells(self) -> int:
        window = self.window((self.child_ny, self.child_nx))
        if window is None:
            return 0
        (dst_j, _), (dst_i, _) = window
        return ((dst_j.stop - dst_j.start) * (dst_i.stop - dst_i.start))

    @property
    def overlap_fraction(self) -> float:
        return (0.0 if self.child_cells == 0
                else self.overlap_cells / self.child_cells)

    @property
    def disjoint(self) -> bool:
        return self.overlap_cells == 0

    def window(self, shape):
        """Destination/source slice pairs for one array's trailing (y, x).

        Returns ``((dst_j, src_j), (dst_i, src_i))`` or ``None`` when the
        old and new footprints do not overlap at all.  Shape-driven rather
        than stagger-driven on purpose: a staggered extent is ``n+1`` and
        shifts by the same whole number of child cells as the mass extent,
        so reading the extent off the array handles every stagger without
        this module having to know which one it is looking at.
        """
        shape = tuple(int(n) for n in shape)
        if len(shape) < 2:
            raise RelocationRefusal(
                f"a relocatable field needs at least (ny, nx), got {shape}")
        rows = _axis_window(shape[-2], self.shift_j)
        cols = _axis_window(shape[-1], self.shift_i)
        if rows is None or cols is None:
            return None
        return (rows, cols)

    def to_json(self) -> dict[str, object]:
        return {
            "contract": RELOCATION_CONTRACT,
            "placement_from": self.placement_from.to_json(),
            "placement_to": self.placement_to.to_json(),
            "parent_grid_ratio": int(self.parent_grid_ratio),
            "child_nx": int(self.child_nx), "child_ny": int(self.child_ny),
            "shift_child_cells": [int(self.shift_i), int(self.shift_j)],
            "shift_parent_cells": [
                int(self.shift_i) // int(self.parent_grid_ratio),
                int(self.shift_j) // int(self.parent_grid_ratio)],
            "null_move": bool(self.null_move),
            "overlap_cells": int(self.overlap_cells),
            "child_cells": int(self.child_cells),
            "overlap_fraction": float(self.overlap_fraction),
            "spin_up_cells": int(self.child_cells - self.overlap_cells),
        }


def plan_relocation(*, placement_from: Placement, placement_to: Placement,
                    parent_grid_ratio: int, child_nx: int,
                    child_ny: int) -> RelocationPlan:
    """Resolve a move into an index-space shift.

    A null move (``placement_to`` at the same position) is admissible and
    planned exactly like any other; its shift is zero, its overlap is the
    whole child, and the resulting transplant is the identity.  That is
    deliberate -- the null move is the calibration point of the whole
    mechanism, and special-casing it would remove the only case whose
    answer is known in advance.
    """
    if placement_from.grid_id != placement_to.grid_id:
        raise RelocationRefusal(
            f"a relocation moves ONE domain: from grid_id "
            f"{placement_from.grid_id} to {placement_to.grid_id}")
    ratio = int(parent_grid_ratio)
    if ratio < 1:
        raise RelocationRefusal(
            f"parent_grid_ratio must be >= 1, got {parent_grid_ratio!r}")
    return RelocationPlan(
        placement_from=placement_from, placement_to=placement_to,
        parent_grid_ratio=ratio,
        child_nx=int(child_nx), child_ny=int(child_ny),
        shift_i=(placement_to.i_parent_start
                 - placement_from.i_parent_start) * ratio,
        shift_j=(placement_to.j_parent_start
                 - placement_from.j_parent_start) * ratio)


def check_admissible(plan: RelocationPlan, bounds) -> dict[str, object]:
    """Judge a planned move against a config's relocation bounds.

    ``bounds`` is a :class:`gpuwm.experiment.RelocationConfig`.  A disabled
    one refuses every non-null move: a config that never opted in must not
    acquire a movable nest because a caller passed a placement.  The null
    move is always admissible, because it is not a move.
    """
    from gpuwm.experiment import DISCRETE_RELOCATION_MODE

    if plan.null_move:
        return {"admissible": True, "null_move": True, "checks": {}}
    if not getattr(bounds, "enabled", False):
        raise RelocationRefusal(
            "this experiment does not enable nest relocation; a moving "
            "nest is opt-in, through [relocation] with enabled = true and "
            f"mode = {DISCRETE_RELOCATION_MODE!r}")
    grid_id = getattr(bounds, "grid_id", None)
    checks: dict[str, object] = {}
    if grid_id is not None and int(grid_id) != plan.placement_to.grid_id:
        raise RelocationRefusal(
            f"[relocation] authorises grid_id {grid_id}, but the move "
            f"targets grid_id {plan.placement_to.grid_id}")
    ratio = int(plan.parent_grid_ratio)
    move_cells = max(abs(plan.shift_i), abs(plan.shift_j)) // ratio
    limit = getattr(bounds, "max_move_parent_cells", None)
    checks["move_parent_cells"] = int(move_cells)
    checks["max_move_parent_cells"] = (
        None if limit is None else int(limit))
    if limit is not None and move_cells > int(limit):
        raise RelocationRefusal(
            f"the move is {move_cells} parent cells, over the configured "
            f"maximum of {limit}; a larger jump exposes a spin-up strip "
            "the child cannot recover before it matters")
    floor = getattr(bounds, "min_overlap_fraction", None)
    checks["overlap_fraction"] = float(plan.overlap_fraction)
    checks["min_overlap_fraction"] = (None if floor is None
                                      else float(floor))
    if floor is not None and plan.overlap_fraction < float(floor):
        raise RelocationRefusal(
            f"the move keeps only {plan.overlap_fraction:.4f} of the "
            f"child's cells, under the configured floor of {float(floor):.4f}")
    return {"admissible": True, "null_move": False, "checks": checks}


# ---------------------------------------------------------------------------
# The transplant
# ---------------------------------------------------------------------------

def relocatable_attrs() -> tuple[str, ...]:
    """The state inventory a relocation carries across the move.

    The restart layer's serialised-state contract, and nothing invented
    beside it: whatever a checkpoint must carry to resume a domain is
    exactly what a relocation must carry to keep it.  Using one list means
    a field added to the model reaches both paths at once.
    """
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    return tuple(STATE_SERIALIZED_ATTRS)


def _bit_mismatches(actual, expected) -> int:
    """Count differing 32-bit patterns, so signed zeros and NaNs count."""
    actual = np.ascontiguousarray(_host(actual))
    expected = np.ascontiguousarray(_host(expected))
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return max(int(actual.size), int(expected.size), 1)
    if actual.dtype == np.float32:
        return int(np.count_nonzero(
            actual.view(np.uint32) != expected.view(np.uint32)))
    return int(np.count_nonzero(actual != expected))


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    if callable(get) and hasattr(value, "__cuda_array_interface__"):
        return np.asarray(get())
    return np.asarray(value)


def transplant_overlap(*, source_state, target_state, plan: RelocationPlan,
                       attrs=None) -> dict[str, object]:
    """Stamp the outgoing child's state onto the incoming one, bitwise.

    ``target_state`` arrives already filled everywhere by the cold-start
    path (a full SINT of the live parent at the NEW placement), so this
    function only has to overwrite the part of it that the outgoing child
    already knows better.  Writing the whole field and then stamping the
    overlap, rather than filling only the strip, is not laziness: it makes
    the null move provably the identity, because at zero shift the stamp
    covers the entire field.
    """
    attrs = relocatable_attrs() if attrs is None else tuple(attrs)
    stamped: dict[str, object] = {}
    skipped: dict[str, str] = {}
    for name in attrs:
        source = getattr(source_state, name, None)
        target = getattr(target_state, name, None)
        if source is None and target is None:
            continue
        if source is None or target is None:
            skipped[name] = ("absent on the outgoing child" if source is None
                             else "absent on the incoming child")
            continue
        if tuple(source.shape) != tuple(target.shape):
            raise RelocationRefusal(
                f"field {name!r} has shape {tuple(source.shape)} on the "
                f"outgoing child and {tuple(target.shape)} on the incoming "
                "one; a relocation changes position, never extent")
        window = plan.window(source.shape)
        if window is None:
            skipped[name] = "old and new footprints are disjoint"
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        chunk = source[..., src_j, src_i]
        if (hasattr(target, "__cuda_array_interface__")
                and isinstance(chunk, np.ndarray)):
            # Host-staged upload: CuPy's sliced __setitem__ refuses a
            # non-scalar NumPy right-hand side, so the H2D transfer is
            # explicit.  A byte transport either way.
            import cupy as cp

            chunk = cp.asarray(np.ascontiguousarray(chunk))
        target[..., dst_j, dst_i] = chunk
        stamped[name] = {
            "shape": list(int(n) for n in source.shape),
            "stamped_cells": int((dst_j.stop - dst_j.start)
                                 * (dst_i.stop - dst_i.start)),
        }
    return {
        "contract": RELOCATION_CONTRACT,
        "attrs_considered": list(attrs),
        "stamped": stamped,
        "skipped": skipped,
        "stamped_field_count": len(stamped),
    }


#: Base-state fields a child derives from its parent by SINT.  They are
#: NOT transplanted -- the cold start recomputes them at the new placement
#: -- which is precisely why comparing them across the move is a valid
#: independent check of the shift arithmetic (see module docstring).
_DONOR_ALIGNMENT_FIELDS = ("pb", "alb", "thb", "phb", "mub2d")


def _interior_window(window, shape, shift_j: int, shift_i: int,
                     frame_width: int):
    """Shrink an overlap window to cells >= frame_width inside BOTH
    footprints, or ``None`` when nothing is left."""
    frame = int(frame_width)
    (dst_j, src_j), (dst_i, src_i) = window
    out = []
    for dst, shift, extent in ((dst_j, shift_j, shape[-2]),
                               (dst_i, shift_i, shape[-1])):
        lo = max(dst.start, frame, frame - shift)
        hi = min(dst.stop, extent - frame, extent - frame - shift)
        if lo >= hi:
            return None
        out.append((slice(lo, hi), slice(lo + shift, hi + shift)))
    return (out[0], out[1])


def donor_alignment_check(*, source_state, target_state,
                          plan: RelocationPlan,
                          frame_width: int = 0) -> dict[str, object]:
    """Prove the shift lands on the same donor cells, without using the shift.

    Every field here is produced by the SINT operator from the parent's
    (non-evolving) base state.  Under a placement change of whole parent
    cells the donor index and sub-cell offset of an overlapped child cell
    are unchanged, so these fields must agree BITWISE on the overlap.  They
    are computed independently of :meth:`RelocationPlan.window`, so a wrong
    shift shows up here as a nonzero mismatch count rather than as a
    plausible-looking field.

    A null move makes this a whole-field identity check, which is the
    stronger statement and is the reason the null move is run first.

    ``frame_width`` scopes the comparison to overlap cells at least that
    many cells inside BOTH footprints.  It exists for the real-data
    initializer, whose base fields are the t=0 machinery's -- analytic
    on the child's own terrain, blended toward the parent inside each
    footprint's ``spec_bdy_width + blend_width`` frame -- so the
    placement invariant genuinely holds only outside both frames.  The
    instrument stays armed there: a wrong shift still breaks the
    interior bitwise equality.  Zero keeps the exact historical
    whole-overlap comparison.
    """
    fields: dict[str, object] = {}
    for name in _DONOR_ALIGNMENT_FIELDS:
        source = getattr(source_state, name, None)
        target = getattr(target_state, name, None)
        if source is None or target is None:
            continue
        source_host = _host(source)
        target_host = _host(target)
        if source_host.ndim < 2:
            # A horizontally uniform base (idealized, terrain_opt off) has
            # nothing to align; recording that is more useful than a skip.
            fields[name] = {
                "horizontally_uniform": True,
                "bit_mismatches": _bit_mismatches(source_host, target_host),
                "compared_cells": int(source_host.size),
            }
            continue
        window = plan.window(source_host.shape)
        if window is None:
            fields[name] = {"disjoint": True, "compared_cells": 0}
            continue
        if frame_width:
            window = _interior_window(
                window, source_host.shape, plan.shift_j, plan.shift_i,
                frame_width)
            if window is None:
                # The frame swallowed the whole overlap: the instrument
                # cannot see, which is a refusal (pass stays False on
                # zero compared cells), never a silent skip.
                fields[name] = {"interior_empty": True, "compared_cells": 0}
                continue
        (dst_j, src_j), (dst_i, src_i) = window
        actual = target_host[..., dst_j, dst_i]
        expected = source_host[..., src_j, src_i]
        fields[name] = {
            "horizontally_uniform": False,
            "bit_mismatches": _bit_mismatches(actual, expected),
            "compared_cells": int(actual.size),
        }
    compared = [item for item in fields.values()
                if item.get("compared_cells", 0) > 0]
    return {
        "fields": fields,
        "compared_field_count": len(compared),
        "pass": bool(compared and all(
            item.get("bit_mismatches") == 0 for item in compared)),
    }


# ---------------------------------------------------------------------------
# Host staging: one child resident on the device, not two
# ---------------------------------------------------------------------------

class HostStateSnapshot:
    """Host copies of a child's state, staged across a relocation.

    Attribute access mirrors the source state (absent field -> absent
    attribute), so this object can stand in as ``source_state`` for
    :func:`transplant_overlap` and :func:`donor_alignment_check` after the
    device allocation it was taken from has been released.  A D2H copy and
    the H2D upload the transplant performs are byte transports, so staging
    through here cannot change a single bit of the overlap -- a property
    the tests assert rather than assume.
    """

    def __init__(self, arrays, *, pinned: bool):
        self._arrays = dict(arrays)
        self._pinned = bool(pinned)

    def __getattr__(self, name):
        arrays = object.__getattribute__(self, "_arrays")
        if name in arrays:
            return arrays[name]
        raise AttributeError(name)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(sorted(object.__getattribute__(self, "_arrays")))

    @property
    def nbytes(self) -> int:
        arrays = object.__getattribute__(self, "_arrays")
        return int(sum(int(value.nbytes) for value in arrays.values()))

    @property
    def pinned(self) -> bool:
        return object.__getattribute__(self, "_pinned")


def _pinned_host_empty(shape, dtype) -> np.ndarray:
    """A page-locked host array (falls back to pageable without CUDA)."""
    import cupy as cp

    count = int(np.prod(shape)) if len(shape) else 1
    nbytes = int(np.dtype(dtype).itemsize) * max(count, 1)
    memory = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(memory, dtype=dtype, count=count).reshape(shape)


def snapshot_state_to_host(state, attrs, *, pin: bool = True
                           ) -> HostStateSnapshot:
    """Copy the named fields to host buffers, pinned where a device exists.

    Fields absent (or ``None``) on the state are absent from the snapshot,
    which is exactly how :func:`transplant_overlap` records them as
    skipped.  Host-resident fields are deep-copied, so releasing the
    source cannot reach back into the snapshot.
    """
    arrays: dict[str, np.ndarray] = {}
    pinned = False
    for name in dict.fromkeys(attrs):
        value = getattr(state, name, None)
        if value is None:
            continue
        if hasattr(value, "__cuda_array_interface__"):
            import cupy as cp

            device = cp.ascontiguousarray(value)
            if pin:
                host = _pinned_host_empty(device.shape, device.dtype)
                pinned = True
            else:
                host = np.empty(device.shape, dtype=device.dtype)
            device.get(out=host)
            arrays[name] = host
        else:
            arrays[name] = np.array(value, copy=True)
    return HostStateSnapshot(arrays, pinned=pinned)


def release_state_arrays(state) -> dict[str, object]:
    """Drop every array reference a state object holds, and count what fell.

    This is the step that keeps peak device residency at ~one child: after
    the snapshot, the outgoing child's arrays are dereferenced BEFORE the
    incoming child allocates, so the allocator can hand the same bytes
    back.  ``owned_device_bytes`` counts allocations the state owns
    outright; views into the shared scratch arena / dycore workspace
    release nothing of their own and are counted apart.  The physics
    driver is dropped too -- its accumulators are per-placement
    continuation state the relocation contract re-initialises anyway.
    """
    owned_device_bytes = 0
    owned_device_arrays = 0
    device_views = 0
    host_arrays = 0
    for name, value in list(vars(state).items()):
        if hasattr(value, "__cuda_array_interface__"):
            if getattr(value, "base", None) is None:
                owned_device_bytes += int(value.nbytes)
                owned_device_arrays += 1
            else:
                device_views += 1
            setattr(state, name, None)
        elif isinstance(value, np.ndarray):
            host_arrays += 1
            setattr(state, name, None)
    if getattr(state, "physics", None) is not None:
        state.physics = None
    return {
        "owned_device_bytes": int(owned_device_bytes),
        "owned_device_arrays": int(owned_device_arrays),
        "shared_view_arrays": int(device_views),
        "host_arrays": int(host_arrays),
    }


def _device_used_bytes():
    """Live CuPy default-pool bytes, or ``None`` off the card."""
    try:
        import cupy as cp

        return int(cp.get_default_memory_pool().used_bytes())
    except Exception:
        return None


def _prevalidate_placement(new_dc, parent_node) -> None:
    """Fail an off-parent move BEFORE the outgoing child is released.

    The in-device path gets this refusal for free from the initializer's
    ``register_nest`` call, which fires while the outgoing child is still
    whole.  The host-staged path releases the outgoing child first, so the
    same +-2 SINT stencil rule is evaluated here, on CPU index tables
    alone, while everything is still intact.  ``register_nest`` stays the
    owner of the rule; this merely asks it early.
    """
    from gpuwm.core.nest_interp import register_nest

    child_run = new_dc.run
    parent_run = parent_node.cfg.run
    ratio = int(new_dc.parent_grid_ratio)
    for stagger in ("", "x", "y"):
        register_nest(
            nri=ratio, nrj=ratio,
            i_parent_start=int(new_dc.i_parent_start),
            j_parent_start=int(new_dc.j_parent_start),
            child_nx=child_run.nx, child_ny=child_run.ny,
            parent_nx=parent_run.nx, parent_ny=parent_run.ny,
            stagger=stagger, wrapper="bdy")


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------

#: The explicit statics fallback this module ships tonight, stated on
#: every receipt that uses it.  The 2026-08-06 requirement (Drew, via the
#: WRF moving-nest discussion): a relocated nest's STATIC fields --
#: terrain, landuse, soil categories -- must be REBUILT for the new
#: footprint from the nest's own static source at nest resolution
#: ([static.highres] / the 30s baseline, both footprint-parametric),
#: because over-land storm-following lives on resolved terrain and
#: parent-interpolated terrain on the fresh strip defeats the nest.  The
#: ``initializer`` parameter of :func:`relocate_child` is that seam -- a
#: route passes a per-footprint builder that constructs own-grid statics
#: and runs the SAME terrain adjustment the t=0 nest cold start performs
#: (gpuwm.ingest.nest_init) -- and the default ``parent_only_init`` is
#: the explicit, receipt-recorded fallback, never a silent choice.
#: The real-data initializer landed as
#: :func:`gpuwm.ingest.relocation_init.real_relocation_initializer`:
#: own-source statics for the new footprint plus the t=0 cold-start
#: terrain adjustment on the parent-interpolated state.
PARENT_INTERPOLATED_STATICS_FALLBACK = (
    "parent-interpolated statics (SINT of the live parent): the explicit "
    "fallback for idealized trees; real-data routes rebuild own-source "
    "statics per footprint (gpuwm.ingest.relocation_init, 2026-08-06 "
    "requirement)")


def relocate_child(child_node, *, i_parent_start: int, j_parent_start: int,
                   segment: RelocationSegment | None = None,
                   bounds=None, initializer=None, on_child_built=None,
                   scratch_arena=None, dycore_state_workspace=None,
                   state_digest=None, staging: str = "device",
                   on_before_release=None,
                   static_provenance: str | None = None) -> dict[str, object]:
    """Move a live child domain to a new placement, in place on the node.

    The three things this does, in the order they must happen:

    1. build the child afresh at the new placement from the LIVE parent,
       through the ordinary cold-start path, which also rebuilds the
       child's own grid, map factors and base state for where it now is;
    2. stamp the outgoing child's serialised state onto the overlap, so
       everything the child had integrated over ground it still covers
       survives the move bitwise;
    3. rebuild the coupler's SINT donor tables for the new placement, and
       classify the child's rolling boundary tables invalid -- they were
       built for a footprint that no longer exists.

    The parent is not written at any point.  ``parent_state_sha256`` in the
    returned receipt is taken before and after and compared, so that is a
    measured claim rather than an architectural one.

    ``child_node`` must be at rest: relocation is a cycle-boundary
    operation, and calling it with the parent leading the child would leave
    the rebuilt tables describing an instant the child has not reached.

    MEMORY -- TWO STAGINGS.  ``staging="device"`` allocates the incoming
    child while the outgoing one is still resident, so peak child
    residency is transiently doubled (a cycle-boundary cost, not a
    per-step one).  That was the right trade for proving the mechanism
    against the certified cold-start path, and it is the wrong trade for
    the user this feature is FOR: the point of a moving nest is a low-VRAM
    card running a smaller nest at higher resolution, and a card that
    cannot hold two children cannot move one.  ``staging="host"`` is the
    production answer: snapshot the outgoing child's serialised state
    (plus the SINT base fields the alignment instrument needs) to pinned
    host buffers, RELEASE the outgoing device allocation, build the
    incoming child into the bytes the allocator just got back, and upload
    the overlap from host.  D2H and H2D are byte transports, so the two
    stagings are bitwise-identical on the overlap -- asserted by
    tests/test_nest_relocation_staging.py, not assumed.  The receipt's
    ``staging`` block carries live-pool samples around each phase so the
    peak claim is measured, never architectural.

    ATOMICITY UNDER HOST STAGING, STATED HONESTLY.  The in-device path
    leaves the tree untouched on any refusal.  The host path releases the
    outgoing child before the rebuild, so a failure AFTER the release
    (an allocation failure, or the donor-alignment self-check tripping on
    a code defect) leaves the child unusable and the run over.  The
    off-parent refusal -- the one plausible input error -- is evaluated
    BEFORE the release (:func:`_prevalidate_placement`), so what remains
    behind the release is the class of failure that would have killed the
    run anyway.  ``on_before_release`` is the executor's seam for dropping
    its own references (health validators) so the release actually frees;
    it fires only on the host path, immediately before the release.

    SCOPE, STATED SO IT CANNOT BE ASSUMED AWAY.  What THIS PRIMITIVE moves
    is the restart layer's serialised state (:func:`relocatable_attrs`).
    Physics CONTINUATION state -- precipitation accumulators, scheme
    timers, surface fields held by the driver -- is not in that contract:
    it is the ``on_child_built`` preparer's to carry.  The front-door
    route preparers (:class:`gpuwm.runtime.RealRelocationChildPreparer`
    and the prepared-tree subclass) DO carry it -- the land-surface
    continuation fields by donor-filled transplant, and the registry-
    derived per-column driver state (KF timers/held rates, precipitation
    accumulators, W0AVG) through :mod:`gpuwm.core.physics_continuation`
    -- because a preparer that re-initialised it cold-restarted cumulus
    on the whole child at every move (the 2026-08-16 moving-nest KF
    artifact report).  A bespoke ``on_child_built`` that carries nothing
    still gets a cold child, and its receipt says so
    (``accumulators_reinitialized``).
    """
    from gpuwm.ingest.nest_init import parent_only_init, seed_rk_time_t_copies

    if initializer is None:
        initializer = parent_only_init
        if static_provenance is None:
            static_provenance = PARENT_INTERPOLATED_STATICS_FALLBACK
    if state_digest is None:
        from gpuwm.ensemble.state_sha import live_state_sha256
        state_digest = live_state_sha256

    if staging not in ("device", "host"):
        raise RelocationRefusal(
            f"staging must be 'device' or 'host', got {staging!r}")
    parent_node = child_node.parent
    if parent_node is None:
        raise RelocationRefusal(
            "the root domain has no parent to be placed in; relocation "
            "applies to a child")
    if getattr(child_node, "children", None):
        raise RelocationRefusal(
            f"child d{child_node.cfg.grid_id:02d} has children of its own; "
            "moving it would silently change the ground under every "
            "grandchild while their placements and donor tables still "
            "describe the old footprint.  Relocation moves a LEAF domain; "
            "a moving mid-tree domain is unimplemented, not implied")
    if child_node.coupler is None:
        raise RelocationRefusal(
            f"child d{child_node.cfg.grid_id:02d} has no coupler; a domain "
            "with no parent forcing edge cannot be relocated coherently")
    lead = int(parent_node.clock.ticks) - int(child_node.clock.ticks)
    if lead != 0:
        raise RelocationRefusal(
            f"relocation is a cycle-boundary operation and requires "
            f"synchronized clocks; parent leads child by {lead} ticks. "
            "Relocate between legs, not mid-step")

    old_dc = child_node.cfg
    placement_from = placement_of(
        old_dc, generation=(0 if segment is None else segment.generation))
    placement_to = Placement(
        grid_id=int(old_dc.grid_id),
        i_parent_start=int(i_parent_start),
        j_parent_start=int(j_parent_start),
        generation=placement_from.generation + 1)
    plan = plan_relocation(
        placement_from=placement_from, placement_to=placement_to,
        parent_grid_ratio=int(old_dc.parent_grid_ratio),
        child_nx=int(old_dc.run.nx), child_ny=int(old_dc.run.ny))
    if plan.disjoint:
        raise RelocationRefusal(
            f"the new placement {placement_to.position} shares no cell with "
            f"the old one {placement_from.position}; that is a new domain, "
            "not a relocation, and nothing of the child's integrated state "
            "would survive it")
    admissibility = (None if bounds is None
                     else check_admissible(plan, bounds))
    if static_provenance is None:
        raise RelocationRefusal(
            "a custom relocation initializer must state its "
            "static_provenance: whether the rebuilt child's terrain/"
            "landuse/soil categories are footprint-rebuilt from the "
            "nest's own static source or parent-interpolated is a "
            "recorded fact of every move, never a silent property "
            "(2026-08-06 statics-on-move requirement)")

    parent_sha_before = state_digest(parent_node.state)
    child_sha_before = state_digest(child_node.state)

    new_dc = replace(
        old_dc, i_parent_start=int(i_parent_start),
        j_parent_start=int(j_parent_start))
    extra = {}
    if scratch_arena is not None:
        extra["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        extra["dycore_state_workspace"] = dycore_state_workspace

    used_steady = _device_used_bytes()
    if staging == "host":
        # The off-parent refusal, evaluated while the outgoing child is
        # still whole; everything after the release is committed.
        _prevalidate_placement(new_dc, parent_node)
        source_state = snapshot_state_to_host(
            child_node.state,
            tuple(relocatable_attrs()) + _DONOR_ALIGNMENT_FIELDS)
        if on_before_release is not None:
            on_before_release()
        released = release_state_arrays(child_node.state)
        used_after_release = _device_used_bytes()
    else:
        source_state = child_node.state
        released = None
        used_after_release = None

    # register_nest refuses a placement whose donors fall outside the
    # parent's +-2 SINT stencil, so on the in-device path an off-grid move
    # fails HERE, before the outgoing child has been touched (the host
    # path asked the same rule above, before the release).
    initialized = initializer(new_dc, parent_node, **extra)
    # The caller's per-domain seam: map policy, and the physics driver the
    # cold-start path does not attach.  It fires BEFORE the transplant, so
    # a driver is initialised from the parent-interpolated fields and the
    # transplant's stamped diagnostics are the last write to survive.
    if on_child_built is not None:
        on_child_built(initialized, new_dc, parent_node)

    # An initializer whose base fields are blended toward the parent at
    # each footprint's own edge (the real-data t=0 lineage) declares the
    # frame inside which the placement invariant is undefined; the
    # instrument is scoped to where it IS defined and stays bitwise there.
    alignment_frame = int(
        getattr(initializer, "donor_alignment_frame_width", 0) or 0)
    alignment = donor_alignment_check(
        source_state=source_state, target_state=initialized.state,
        plan=plan, frame_width=alignment_frame)
    if not alignment["pass"]:
        raise RelocationRefusal(
            "the relocated child's SINT-derived base state does not match "
            "the outgoing child's on the overlap, so the two placements do "
            f"not share donor cells: {alignment}")

    transplant = transplant_overlap(
        source_state=source_state, target_state=initialized.state,
        plan=plan)
    # The initializer's post-transplant hook: the real-data route rebases
    # blend-frame perturbations onto the rebuilt base so totals survive
    # exactly where the base bytes differ, then re-derives the EOS
    # diagnostics.  It runs BEFORE the RK time-t reseed below, so the
    # seeds copy the corrected fields.
    post_transplant = getattr(initializer, "post_transplant", None)
    post_receipt = None
    if post_transplant is not None:
        post_receipt = post_transplant(
            source_state=source_state, target_state=initialized.state,
            plan=plan)
    used_after_rebuild = _device_used_bytes()
    staging_receipt: dict[str, object] = {
        "mode": staging,
        "device_pool_used_bytes": {
            "steady_state_before": used_steady,
            "after_release": used_after_release,
            "after_rebuild_and_transplant": used_after_rebuild,
        },
    }
    if staging == "host":
        staging_receipt.update({
            "snapshot_fields": len(source_state.field_names),
            "snapshot_bytes": int(source_state.nbytes),
            "snapshot_pinned": bool(source_state.pinned),
            "released": released,
        })
    else:
        staging_receipt["note"] = (
            "in-device transplant: the incoming child is allocated while "
            "the outgoing one is resident, so peak child residency is "
            "transiently doubled; staging='host' is the production path")
    # WRF's start_domain seeds the RK time-t copies from the interpolated
    # state; after the stamp the current fields have changed, so the seeds
    # must be taken again or the first substep would read the cold-start
    # values over ground the transplant just corrected.
    seeded = seed_rk_time_t_copies(initialized.state)

    # Everything that reads the parent has now run.  Check the parent
    # BEFORE the node is touched, so a violation leaves the live tree
    # exactly as it was rather than half-moved with a traceback.
    parent_sha_after = state_digest(parent_node.state)
    if parent_sha_after != parent_sha_before:
        raise RelocationRefusal(
            "the parent state changed across a relocation; the re-grid "
            "reads the parent and must never write it "
            f"({parent_sha_before} -> {parent_sha_after})")
    child_sha_after = state_digest(initialized.state)

    child_node.cfg = new_dc
    child_node.grid = getattr(initialized, "grid", child_node.grid)
    child_node.state = initialized.state
    child_node.state._nest_restart_classification = "REBUILT"
    coupler_receipt = child_node.coupler.relocate()

    record = RelocationRecord(
        placement_from=placement_from, placement_to=placement_to,
        parent_grid_ratio=int(old_dc.parent_grid_ratio),
        shift_i=plan.shift_i, shift_j=plan.shift_j,
        overlap_cells=plan.overlap_cells, child_cells=plan.child_cells,
        null_move=plan.null_move,
        parent_state_sha256=parent_sha_before,
        child_state_sha256_before=child_sha_before,
        child_state_sha256_after=child_sha_after,
        donor_alignment=alignment)
    segment = (base_segment(old_dc) if segment is None else segment)
    segment = segment.append(record)
    return {
        "contract": RELOCATION_CONTRACT,
        "plan": plan.to_json(),
        "admissibility": admissibility,
        "staging": staging_receipt,
        "child_rebuild": {
            "initializer": getattr(initializer, "__qualname__",
                                   repr(initializer)),
            "static_fields": static_provenance,
            # The initializer's own rebuild receipt (static source,
            # rebuild timings, adjustment applied) when it produced one.
            "rebuild": getattr(initialized, "preprocess_receipt", None),
            "donor_alignment_frame_width": alignment_frame,
            "post_transplant": post_receipt,
        },
        "coupler": coupler_receipt,
        "transplant": transplant,
        "rk_seeds_refreshed": list(seeded),
        "donor_alignment": alignment,
        "parent_state_sha256_before": parent_sha_before,
        "parent_state_sha256_after": parent_sha_after,
        "parent_bitwise_unchanged": parent_sha_before == parent_sha_after,
        "child_state_sha256_before": child_sha_before,
        "child_state_sha256_after": child_sha_after,
        "child_state_unchanged": child_sha_before == child_sha_after,
        "segment": segment.to_json(),
        # The live object, so a caller can chain the next move onto this
        # one; "segment" above is its JSON for the receipt.
        "segment_state": segment,
        "record_sha256": segment.records[-1].sha256,
    }


def mark_fingerprint_across_move(fingerprint: str,
                                 record_sha256: str) -> str:
    """Chain a relocation record into a run's live restart fingerprint.

    The tolerated-experiment convention, extended from config to the live
    tree: relocation BOUNDS are byte-inert on every fingerprint
    (``RESTART_TOLERATED_EXPERIMENT_FIELDS``), while an actual MOVE binds
    -- the placement is part of the restart identity, so a tree that has
    moved must not share a fingerprint with the tree as built.  Chaining
    keeps the marking deterministic (same base, same move history, same
    mark) and one-way: no fresh build ever computes a marked value, so a
    checkpoint written after a move refuses to resume into a fresh run BY
    CONSTRUCTION, and the refusal names relocation instead of shrugging a
    hash (:data:`RESTART_ACROSS_MOVE_POSTURE`).  Nothing here builds a
    resume across a move; per the 2026-08-06 ruling, nothing may.
    """
    digest = hashlib.sha256()
    digest.update(RELOCATION_CONTRACT.encode("ascii") + b"\0moved\0")
    digest.update(str(fingerprint).encode("utf-8") + b"\0")
    digest.update(str(record_sha256).encode("ascii") + b"\0")
    return digest.hexdigest()


def relocating_child_initializer(source_state, *, placement_from: Placement,
                                 initializer=None, on_child_built=None):
    """A child initializer that transplants instead of cold-starting.

    The leg-boundary form of :func:`relocate_child`, shaped to drop into
    the assembly hooks that already take a ``child_initializer`` (the
    idealized tree assembler, and the DA route's per-leg rebuild).  The
    same three-step mechanism runs; only the seam differs, because on that
    route the whole model is rebuilt per leg and there is no live node to
    mutate.
    """
    from gpuwm.ingest.nest_init import parent_only_init, seed_rk_time_t_copies

    if initializer is None:
        initializer = parent_only_init
    receipts: list[dict] = []

    def initialize(child_dc, parent_node, **kwargs):
        initialized = initializer(child_dc, parent_node, **kwargs)
        if on_child_built is not None:
            on_child_built(initialized, child_dc, parent_node)
        placement_to = Placement(
            grid_id=int(child_dc.grid_id),
            i_parent_start=int(child_dc.i_parent_start),
            j_parent_start=int(child_dc.j_parent_start),
            generation=placement_from.generation + 1)
        plan = plan_relocation(
            placement_from=placement_from, placement_to=placement_to,
            parent_grid_ratio=int(child_dc.parent_grid_ratio),
            child_nx=int(child_dc.run.nx), child_ny=int(child_dc.run.ny))
        if plan.disjoint:
            raise RelocationRefusal(
                f"placements {placement_from.position} and "
                f"{placement_to.position} share no cell")
        alignment = donor_alignment_check(
            source_state=source_state, target_state=initialized.state,
            plan=plan)
        if not alignment["pass"]:
            raise RelocationRefusal(
                f"relocated base state disagrees on the overlap: {alignment}")
        transplant = transplant_overlap(
            source_state=source_state, target_state=initialized.state,
            plan=plan)
        seeded = seed_rk_time_t_copies(initialized.state)
        receipts.append({
            "contract": RELOCATION_CONTRACT,
            "plan": plan.to_json(),
            "transplant": transplant,
            "donor_alignment": alignment,
            "rk_seeds_refreshed": list(seeded),
        })
        return initialized

    initialize.receipts = receipts
    return initialize


__all__ = [
    "HostStateSnapshot", "PARENT_INTERPOLATED_STATICS_FALLBACK",
    "PLACEMENT_FIELDS", "RELOCATION_CONTRACT",
    "RESTART_ACROSS_MOVE_POSTURE", "Placement",
    "RelocationPlan", "RelocationRecord", "RelocationRefusal",
    "RelocationSegment", "base_segment", "check_admissible",
    "donor_alignment_check", "mark_fingerprint_across_move",
    "placement_independent_identity", "placement_of", "plan_relocation",
    "release_state_arrays",
    "relocatable_attrs", "relocate_child", "relocating_child_initializer",
    "snapshot_state_to_host", "transplant_overlap",
]
