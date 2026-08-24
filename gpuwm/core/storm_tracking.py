"""Storm tracking for the storm-following moving nest: the WHEN and WHERE.

:mod:`gpuwm.core.nest_relocation` is the mechanism half of a moving nest
and says so in its own docstring: *"it does not decide when or where to
move -- there is no tracker, no scoring, no hysteresis here.  Placement is
an input."*  This module is that missing half.  It watches the RUNNING
MODEL'S OWN FIELDS on the parent domain and turns them into discrete
whole-parent-cell shift proposals a relocation runner can hand to
:func:`gpuwm.core.nest_relocation.relocate_child`.

THE SEAM (the plan-provider contract).  The runner side of the moving-nest
program consumes a tracker through one callable, evaluated once per
relocation cadence (a cycle boundary, never mid-step)::

    desired_shift(parent_state, nest_footprint, t) -> (di, dj) | None

``parent_state`` is the PARENT domain's live state (host or device);
``nest_footprint`` is where the child currently sits, as a
:class:`NestFootprint` or anything carrying the five
:class:`gpuwm.experiment.DomainConfig` placement/extent attributes;
``t`` is model seconds since the experiment start (model time, not
wallclock, so the hysteresis is deterministic and restart-independent).
The return is the desired move in WHOLE PARENT CELLS -- ``di`` along the
x/i axis (the LAST array axis; positive means a larger ``i_parent_start``)
and ``dj`` along y/j (axis -2, positive means larger ``j_parent_start``)
-- or ``None`` for "hold".  The runner applies it as
``i_parent_start + di`` / ``j_parent_start + dj`` and the leg-1
admissibility bounds (:func:`gpuwm.core.nest_relocation.check_admissible`)
remain the runner's to enforce; a tracker proposes, it never moves
anything.  :class:`StormTracker` implements the contract and ``__call__``
aliases it, so the provider is literally "a callable returning the desired
whole-parent-cell shift each cadence".  One optional duck-typed hook
completes the seam: the runner calls
:meth:`StormTracker.notify_move_executed` with the shift it really
applied after each successful relocation, and from then on the cooldown
counts from EXECUTED moves; a runner that never calls it leaves the
proposal-burn fallback in force.

WHAT IT TRACKS.  Primary signal: a column updraft-helicity running-max
plane, folded every step by :mod:`gpuwm.core.uh_diag` from WRF's own
cal_helicity columns.  It is THIS CONSUMER'S window
(``uh_diag.UH_FOLLOW_WINDOW_SLOT``), reset by the relocation runner at
every evaluation, so it is exactly "the strongest rotation since the
tracker last had a chance to move".

It is deliberately NOT WRF's UP_HELI_MAX diagnostic, which is folded from
the same columns in the same pass but zeroed by the HISTORY WRITER.
Reading that one made a nest's placement a function of
``history_interval_s``: an output knob silently steering the model
(closed 2026-08-07).  The window is a max over the interval rather than
an instantaneous sample because UH is spiky -- sampling it at cadence
boundaries would alias transient mesocyclone pulses, and a threshold
cannot un-alias a signal.

A rotating storm is what a tornado nest exists to follow, so rotation is
the primary vote.  Fallback:
the column-max (composite) simulated reflectivity from the persistent
``refl_10cm`` slot, for the window before the storm rotates -- an echo
centroid keeps the nest on the convection until a mesocyclone exists to
take over.  The centroid arithmetic is the WaH survey's proven idiom
(``echo_stats`` / ``motion_from_centroids`` in tools/da_nowcast.py):
threshold, then a weighted centroid of the exceedance -- adapted here to
grid index space rather than radar gate space, and kept in core so the
model never imports from tools.

WHY HYSTERESIS.  A centroid jitters: convective cells pulse, split and
merge, and a nest that re-centers on every wobble spends its life paying
relocation spin-up strips for moves that bought nothing.  Three guards,
all configured, all logged when they fire:

- dead-band: a proposed shift below ``min_shift_cells`` (Chebyshev norm)
  is suppressed -- the storm has not really left the middle of the nest;
- clamp: a shift is limited to ``max_shift_cells`` per axis per event, so
  one bad centroid (or one real jump) cannot trade away most of the
  overlap transplant in a single move;
- cooldown: after a proposal, further proposals are suppressed for
  ``cooldown_seconds`` of model time, so the donor tables are not rebuilt
  every cadence while a storm crosses cells at speed.

Every call appends a receipt carrying the centroid evidence and the
decision, so a run's move/hold history is auditable after the fact.

CONFIG.  ``[relocation.follow]`` under the leg-1 ``[relocation]`` table
(:func:`gpuwm.experiment._build_relocation` wires it): ``field``
(``"uh"`` | ``"reflectivity"``), ``threshold`` (m2 s-2 for uh, dBZ for
reflectivity), ``fallback_threshold`` (dBZ; required with ``field =
"uh"``, refused otherwise -- a m2 s-2 number cannot be reused as a dBZ
one), ``search_margin_cells``, ``min_shift_cells``, ``max_shift_cells``,
``cooldown_seconds``.  Every key is required, unknown keys refuse, and
the constructed config echoes its values (:meth:`FollowConfig.to_json`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from gpuwm.core import streaming, uh_diag

#: Versioned label carried by every receipt this module emits.
FOLLOW_CONTRACT = "gpuwm-storm-follow.v1"

#: The two signals the tracker knows how to read off a parent state.
TRACKED_FIELDS = ("uh", "reflectivity")

#: Scratch slots the signals live in (gpuwm.core.uh_diag /
#: gpuwm.core.refl).  Literal names: the preflight scratch registry and
#: this module must agree on them.
#:
#: The UH signal is a CONSUMER-OWNED window, never WRF's UP_HELI_MAX.
#: The diagnostic's own accumulator is zeroed by the history writer, so
#: reading it made this tracker's decisions a function of the output
#: cadence -- change ``history_interval_s`` and the nest goes somewhere
#: else.  ``uh_follow_window`` is folded from the same columns in the
#: same pass and is reset by THIS consumer at every evaluation, so its
#: window is exactly one relocation cadence (Drew's ruling, 2026-08-07).
#: A caller that owns a different evaluation rhythm passes its own slot:
#: gpuwm.core.nest_spawn reads ``uh_spawn_window`` on leg boundaries.
UH_SLOT = uh_diag.UH_FOLLOW_WINDOW_SLOT
REFLECTIVITY_SLOT = "refl_10cm"

#: Keys of ``[relocation.follow]``.  All required; unknown keys refuse.
FOLLOW_KEYS = frozenset({
    "field", "threshold", "fallback_threshold", "search_margin_cells",
    "min_shift_cells", "max_shift_cells", "cooldown_seconds",
})

#: A proposal is clipped so the child's footprint keeps this many parent
#: cells clear of the parent's edge.  5 is WRF's default specified-boundary
#: width (Registry.EM_COMMON spec_bdy_width; spec_zone 1 + relax_zone 4):
#: a nest whose edge sits inside the parent's boundary relaxation zone is
#: being forced by boundary blending, not by the parent's interior
#: solution.  Tracker policy, not config: no case should want a nest in
#: the parent's spec zone.
PARENT_EDGE_KEEPOUT_CELLS = 5

_LOG = logging.getLogger("gpuwm.storm_tracking")


class TrackerRefusal(ValueError):
    """A tracking request this module will not serve quietly."""


# ---------------------------------------------------------------------------
# Config: [relocation.follow]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FollowConfig:
    """The validated ``[relocation.follow]`` block.

    ``threshold`` is in the primary field's own units (m2 s-2 for
    ``"uh"``, dBZ for ``"reflectivity"``); ``fallback_threshold`` is
    always dBZ and exists only under ``field = "uh"``, where it gates the
    reflectivity handoff for the pre-rotation window.
    """

    field: str
    threshold: float
    search_margin_cells: int
    min_shift_cells: int
    max_shift_cells: int
    cooldown_seconds: float
    fallback_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.field not in TRACKED_FIELDS:
            raise ValueError(
                f"follow field must be one of {TRACKED_FIELDS}, got "
                f"{self.field!r}")
        if not math.isfinite(float(self.threshold)):
            raise ValueError(
                f"follow threshold must be finite, got {self.threshold!r}")
        if self.field == "uh":
            if self.fallback_threshold is None:
                raise ValueError(
                    "follow field = 'uh' requires fallback_threshold (dBZ): "
                    "the reflectivity handoff for the window before the "
                    "storm rotates cannot reuse a m2 s-2 number as dBZ")
            if not math.isfinite(float(self.fallback_threshold)):
                raise ValueError(
                    "follow fallback_threshold must be finite, got "
                    f"{self.fallback_threshold!r}")
        elif self.fallback_threshold is not None:
            raise ValueError(
                "follow field = 'reflectivity' refuses fallback_threshold: "
                "reflectivity is already the fallback signal and there is "
                "nothing below it to hand off to")
        if int(self.search_margin_cells) < 0:
            raise ValueError(
                "search_margin_cells must be >= 0 parent cells, got "
                f"{self.search_margin_cells!r}")
        if int(self.min_shift_cells) < 1:
            raise ValueError(
                "min_shift_cells must be >= 1 parent cell (the null shift "
                f"is never proposed), got {self.min_shift_cells!r}")
        if int(self.max_shift_cells) < int(self.min_shift_cells):
            raise ValueError(
                f"max_shift_cells = {self.max_shift_cells!r} is below "
                f"min_shift_cells = {self.min_shift_cells!r}; the dead-band "
                "would suppress every admissible proposal")
        if not (math.isfinite(float(self.cooldown_seconds))
                and float(self.cooldown_seconds) >= 0.0):
            raise ValueError(
                "cooldown_seconds must be a finite non-negative model-time "
                f"duration, got {self.cooldown_seconds!r}")

    def to_json(self) -> dict[str, object]:
        """Echo every configured value (the receipts' config record)."""
        out: dict[str, object] = {
            "contract": FOLLOW_CONTRACT,
            "field": self.field,
            "threshold": float(self.threshold),
            "search_margin_cells": int(self.search_margin_cells),
            "min_shift_cells": int(self.min_shift_cells),
            "max_shift_cells": int(self.max_shift_cells),
            "cooldown_seconds": float(self.cooldown_seconds),
        }
        if self.fallback_threshold is not None:
            out["fallback_threshold"] = float(self.fallback_threshold)
        return out


def _require_number(table: dict, key: str, source: str, *,
                    integer: bool = False):
    value = table[key]
    ok = (isinstance(value, int) and not isinstance(value, bool)) if integer \
        else (isinstance(value, (int, float)) and not isinstance(value, bool))
    if not ok:
        kind = "an integer" if integer else "a number"
        raise ValueError(
            f"{key} in [relocation.follow] of {source} must be {kind}, "
            f"got {value!r}")
    return value


def build_follow_config(table: dict, source: str) -> FollowConfig:
    """Validate a parsed ``[relocation.follow]`` table.

    Honored or refused, never ignored: every key is required by name (a
    follow block is seven short lines, and a default the user never chose
    is how a nest starts moving on a threshold nobody picked), and every
    unknown key is refused by name.
    """
    from gpuwm.experiment import did_you_mean

    unknown = sorted(set(table) - FOLLOW_KEYS)
    if unknown:
        named = ", ".join(
            f"{key!r}{did_you_mean(key, FOLLOW_KEYS)}" for key in unknown)
        raise ValueError(
            f"[relocation.follow] of {source} does not have key(s) {named}; "
            "no key is ignored, because a dropped key tracks a storm with a "
            "value nobody chose.")
    required = sorted(FOLLOW_KEYS - {"fallback_threshold"})
    missing = [key for key in required if key not in table]
    if missing:
        raise ValueError(
            f"[relocation.follow] of {source} is missing required key(s) "
            f"{missing}; present: {sorted(table)}. Every follow key is "
            "chosen deliberately -- there are no defaults to inherit.")
    field = table["field"]
    if not isinstance(field, str):
        raise ValueError(
            f"field in [relocation.follow] of {source} must be a string, "
            f"got {field!r}")
    fallback = None
    if "fallback_threshold" in table:
        fallback = float(_require_number(table, "fallback_threshold", source))
    try:
        return FollowConfig(
            field=field,
            threshold=float(_require_number(table, "threshold", source)),
            fallback_threshold=fallback,
            search_margin_cells=int(_require_number(
                table, "search_margin_cells", source, integer=True)),
            min_shift_cells=int(_require_number(
                table, "min_shift_cells", source, integer=True)),
            max_shift_cells=int(_require_number(
                table, "max_shift_cells", source, integer=True)),
            cooldown_seconds=float(_require_number(
                table, "cooldown_seconds", source)))
    except ValueError as err:
        raise ValueError(f"[relocation.follow] of {source}: {err}") from None


# ---------------------------------------------------------------------------
# Geometry: where the child sits, in parent index space
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NestFootprint:
    """The child's current placement and extent, in parent-cell terms.

    ``i_parent_start``/``j_parent_start`` keep their 1-based WRF namelist
    semantics (:class:`gpuwm.core.nest_relocation.Placement` refuses 0).
    Continuous parent-array coordinates (0-based) derive from them: child
    mass cell ``n`` (1-based) sits at parent coordinate ``i_parent_start
    + (n-1)/ratio``, WRF's donor arithmetic, so the child spans
    ``(child_nx - 1)/ratio`` parent cells.
    """

    grid_id: int
    i_parent_start: int
    j_parent_start: int
    child_nx: int
    child_ny: int
    parent_grid_ratio: int

    def __post_init__(self) -> None:
        if int(self.parent_grid_ratio) < 1:
            raise ValueError(
                f"parent_grid_ratio must be >= 1, got "
                f"{self.parent_grid_ratio!r}")
        if int(self.i_parent_start) < 1 or int(self.j_parent_start) < 1:
            raise ValueError(
                "i_parent_start/j_parent_start are 1-based WRF namelist "
                f"semantics and must be >= 1, got "
                f"({self.i_parent_start!r}, {self.j_parent_start!r})")
        if int(self.child_nx) < 2 or int(self.child_ny) < 2:
            raise ValueError(
                f"child extents must be >= 2, got "
                f"({self.child_nx!r}, {self.child_ny!r})")

    @classmethod
    def coerce(cls, value) -> "NestFootprint":
        """Accept a NestFootprint, or duck-type a DomainConfig-like."""
        if isinstance(value, cls):
            return value
        run = getattr(value, "run", None)
        return cls(
            grid_id=int(value.grid_id),
            i_parent_start=int(value.i_parent_start),
            j_parent_start=int(value.j_parent_start),
            child_nx=int(run.nx if run is not None else value.child_nx),
            child_ny=int(run.ny if run is not None else value.child_ny),
            parent_grid_ratio=int(value.parent_grid_ratio))

    @property
    def span_parent_i(self) -> float:
        return (int(self.child_nx) - 1) / int(self.parent_grid_ratio)

    @property
    def span_parent_j(self) -> float:
        return (int(self.child_ny) - 1) / int(self.parent_grid_ratio)

    @property
    def center_parent_ij(self) -> tuple[float, float]:
        """(ci, cj): footprint center in 0-based parent-array coords."""
        return (int(self.i_parent_start) - 1 + self.span_parent_i / 2.0,
                int(self.j_parent_start) - 1 + self.span_parent_j / 2.0)

    def search_box(self, plane_shape: tuple[int, int],
                   margin_cells: int) -> tuple[slice, slice]:
        """(j_slice, i_slice) of the footprint + margin, clipped to the
        parent plane."""
        ny, nx = int(plane_shape[-2]), int(plane_shape[-1])
        margin = int(margin_cells)
        i_lo = int(self.i_parent_start) - 1
        j_lo = int(self.j_parent_start) - 1
        i0 = max(0, i_lo - margin)
        j0 = max(0, j_lo - margin)
        i1 = min(nx, int(math.ceil(i_lo + self.span_parent_i)) + 1 + margin)
        j1 = min(ny, int(math.ceil(j_lo + self.span_parent_j)) + 1 + margin)
        if i0 >= i1 or j0 >= j1:
            raise TrackerRefusal(
                f"the nest footprint (i {i_lo}..{i_lo + self.span_parent_i:g},"
                f" j {j_lo}..{j_lo + self.span_parent_j:g}) lies outside the "
                f"parent plane {(ny, nx)}; the footprint and the parent "
                "state disagree about the tree geometry")
        return slice(j0, j1), slice(i0, i1)


# ---------------------------------------------------------------------------
# Signal extraction and centroid arithmetic
# ---------------------------------------------------------------------------

def _host(value) -> np.ndarray:
    """Read-only host view/copy of a host or device array (the
    nest_relocation idiom)."""
    if isinstance(value, np.ndarray):
        return value
    return value.get()


def _plane_from_state(state, field: str, *,
                      uh_slot: str = UH_SLOT) -> np.ndarray:
    """The named signal as a host (ny, nx) float plane, fail-loud.

    ``"uh"`` reads the caller's own UH window (``uh_slot``, defaulting to
    the relocation consumer's); ``"reflectivity"`` reads ``refl_10cm``
    and collapses a 3-D volume to its column max (the composite).  A
    missing slot refuses with the configuration that would provide it --
    silently tracking nothing is how a nest quietly stops following its
    storm.

    ``uh_slot`` exists because the UH window belongs to whoever resets
    it.  Passing another consumer's slot would read a window emptied on
    a rhythm this caller does not control, which is the whole defect the
    consumer-owned windows fixed.
    """
    if not uh_diag.is_tracker_window_slot(uh_slot):
        raise ValueError(
            f"uh_slot={uh_slot!r} is not a consumer-owned tracking "
            "window; use a fixed consumer slot or a slot returned by "
            "uh_diag.follow_window_slot(grid_id). The "
            "WRF UP_HELI_MAX diagnostic is reset by the history writer, "
            "so reading it would make this decision depend on the output "
            "cadence.")
    slot = uh_slot if field == "uh" else REFLECTIVITY_SLOT
    # WHEREVER the domain is keeping it.  Under [tiles] the window is
    # folded in the tile buffers and lands in the store, and the copy on the
    # state stopped moving when streaming.attach took it -- so reading the
    # state would steer the nest on a frozen plane, which is the same defect
    # that swallowed the resets (gpuwm/core/streaming.py:live_scratch).
    # Resident, domain_scratch IS state.existing_scratch(slot).
    buf = streaming.domain_scratch(state, slot)
    if buf is None:
        # Test doubles and reduced states that hang the plane straight off
        # the object rather than in a scratch pool.
        buf = getattr(state, slot, None)
    if buf is None:
        remedy = ("nwp_diagnostics = 1 populates it every step"
                  if field == "uh" else
                  "the microphysics stashes it at history cadence "
                  "(gpuwm.core.refl)")
        raise TrackerRefusal(
            f"the parent state carries no {slot!r} plane, so the tracker's "
            f"{field!r} signal does not exist in this run; {remedy}. "
            "Refused rather than holding forever: a follow block that can "
            "never see its field is a configuration error, not a quiet "
            "storm.")
    plane = np.asarray(_host(buf), dtype=np.float64)
    if plane.ndim == 3:
        plane = plane.max(axis=0)
    if plane.ndim != 2:
        raise TrackerRefusal(
            f"{slot!r} must be a (ny, nx) plane or (nz, ny, nx) volume, "
            f"got shape {plane.shape}")
    return plane


#: Public name for the signal read: the spawn trigger
#: (:mod:`gpuwm.core.nest_spawn`) watches the SAME two parent planes with
#: the SAME missing-slot refusals, so it imports this rather than keeping
#: a second reader that could drift from the tracker's.
signal_plane = _plane_from_state


def weighted_centroid(plane: np.ndarray, threshold: float,
                      box: tuple[slice, slice]) -> dict | None:
    """Exceedance-weighted centroid of ``plane >= threshold`` inside
    ``box``.

    Weights are ``value - threshold`` so the centroid sits on the signal's
    core rather than on the area of its threshold-crossing plateau (the
    grid-space adaptation of the WaH echo centroid).  When every
    qualifying cell sits exactly AT the threshold the weights degenerate
    to uniform.  Returns ``{"ci", "cj", "cells", "max_value"}`` in FULL
    parent-array 0-based coordinates, or ``None`` when nothing qualifies.
    """
    j_slice, i_slice = box
    window = plane[j_slice, i_slice]
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(window) & (window >= threshold)
    cells = int(mask.sum())
    if cells == 0:
        return None
    weights = np.where(mask, window - threshold, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        weights = mask.astype(np.float64)
        total = float(cells)
    jj, ii = np.nonzero(mask)
    w = weights[jj, ii]
    ci = float((ii * w).sum() / total) + i_slice.start
    cj = float((jj * w).sum() / total) + j_slice.start
    return {"ci": ci, "cj": cj, "cells": cells,
            "max_value": float(window[mask].max())}


def _round_cells(value: float) -> int:
    """Symmetric round-half-away-from-zero, so eastward and westward
    proposals of equal magnitude round identically."""
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


# ---------------------------------------------------------------------------
# The tracker (the plan provider)
# ---------------------------------------------------------------------------

#: The two cooldown anchors, as a checkpoint's follower entry names them.
TRACKER_STATE_KEYS = ("last_proposal_t", "last_move_t")


def _timer(value, what: str) -> float | None:
    """A cooldown anchor that survives the header's ``allow_nan=False``."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise TrackerRefusal(
            f"the tracker's {what} is {value!r}; a non-finite cooldown "
            "anchor compares false against every model time, so the "
            "hysteresis would never suppress and the nest would move at "
            "every cadence boundary")
    return number


class StormTracker:
    """Feature tracking plus hysteresis; implements the plan-provider seam.

    Stateful only in the two ways the contract needs: the model time of
    the last proposal (cooldown) and the receipts ledger.  Position state
    lives with the runner -- the footprint is an argument every call -- so
    a tracker never disagrees with the runner about where the nest is.
    """

    def __init__(self, config: FollowConfig, *, uh_slot: str = UH_SLOT) -> None:
        if not isinstance(config, FollowConfig):
            raise TypeError("config must be a FollowConfig")
        self.config = config
        if not uh_diag.is_tracker_window_slot(uh_slot):
            raise ValueError(f"invalid tracker UH slot {uh_slot!r}")
        self.uh_slot = str(uh_slot)
        self._last_proposal_t: float | None = None
        self._last_move_t: float | None = None
        self.receipts: list[dict] = []
        self._receipt({"decision": "configured",
                       "config": config.to_json()})

    # -- receipts ----------------------------------------------------------

    def _receipt(self, entry: dict) -> dict:
        entry = {"contract": FOLLOW_CONTRACT, **entry}
        self.receipts.append(entry)
        _LOG.info("storm-follow %s", entry)
        return entry

    def drain_receipts(self) -> list[dict]:
        """Hand the accumulated receipts to the runner's ledger."""
        out, self.receipts = self.receipts, []
        return out

    # -- the restart state -------------------------------------------------

    def state_json(self) -> dict:
        """The tracker's half of a checkpoint's per-follower entry.

        Two numbers, and nothing else, because position is an argument on
        every call: only the cooldown anchors survive a boundary, and
        nothing but this object holds them.  Pure state -- no I/O, no
        checkpoint vocabulary.
        """
        return {"last_proposal_t": _timer(self._last_proposal_t,
                                          "last_proposal_t"),
                "last_move_t": _timer(self._last_move_t, "last_move_t")}

    def restore_state(self, block) -> None:
        """Seed the cooldown anchors from a checkpoint's follower entry.

        Restored rather than recomputed because there is nothing to
        recompute from: a tracker that just moved its nest and forgets it
        proposes again at the resumed run's FIRST cadence boundary, moving
        the nest a full cooldown early and spinning up the strip it just
        paid for.  The two anchors are restored separately because they
        mean different things -- an executed move and a proposal the runner
        declined -- and collapsing them re-anchors the whole hysteresis.
        """
        if not isinstance(block, dict):
            raise TrackerRefusal(
                "a tracker restart block must be a mapping, got "
                f"{type(block).__name__}")
        unknown = sorted(set(block) - set(TRACKER_STATE_KEYS))
        missing = sorted(set(TRACKER_STATE_KEYS) - set(block))
        if unknown or missing:
            raise TrackerRefusal(
                f"a tracker restart block has key(s) {unknown} this build "
                f"does not know and is missing {missing}; the cooldown is "
                "honored in full or refused, because an anchor restored by "
                "omission lets the nest move at the first boundary of the "
                "resumed run")
        proposal = _timer(block["last_proposal_t"], "last_proposal_t")
        move = _timer(block["last_move_t"], "last_move_t")
        self._last_proposal_t = proposal
        self._last_move_t = move

    # -- the contract ------------------------------------------------------

    def desired_shift(self, parent_state, nest_footprint,
                      t: float) -> tuple[int, int] | None:
        """The plan-provider callable: whole-parent-cell shift or hold.

        Evidence is gathered before any suppression fires, so a
        suppressed receipt still records WHERE the storm was seen -- a
        hold decision without evidence cannot be audited.
        """
        fp = NestFootprint.coerce(nest_footprint)
        cfg = self.config
        plane = _plane_from_state(parent_state, cfg.field, uh_slot=self.uh_slot)
        box = fp.search_box(plane.shape, cfg.search_margin_cells)
        field_used = cfg.field
        threshold_used = float(cfg.threshold)
        found = weighted_centroid(plane, threshold_used, box)
        if found is None and cfg.field == "uh":
            # The handoff: no rotation signal yet; follow the echo.
            field_used = "reflectivity"
            threshold_used = float(cfg.fallback_threshold)
            plane = _plane_from_state(parent_state, field_used)
            found = weighted_centroid(plane, threshold_used, box)
        evidence: dict[str, object] = {
            "t": float(t),
            "field_requested": cfg.field,
            "field_used": field_used,
            "threshold_used": threshold_used,
            "search_box": [[int(box[0].start), int(box[0].stop)],
                           [int(box[1].start), int(box[1].stop)]],
            "footprint": {
                "grid_id": int(fp.grid_id),
                "i_parent_start": int(fp.i_parent_start),
                "j_parent_start": int(fp.j_parent_start),
                "center_parent_ij": [round(c, 3)
                                     for c in fp.center_parent_ij],
            },
        }
        if found is None:
            self._receipt({"decision": "no-signal", **evidence})
            return None
        center_i, center_j = fp.center_parent_ij
        raw_di = found["ci"] - center_i
        raw_dj = found["cj"] - center_j
        evidence.update({
            "cells_above_threshold": found["cells"],
            "max_value": round(found["max_value"], 3),
            "centroid_parent_ij": [round(found["ci"], 3),
                                   round(found["cj"], 3)],
            "raw_shift_parent_cells": [round(raw_di, 3), round(raw_dj, 3)],
        })
        di = _round_cells(raw_di)
        dj = _round_cells(raw_dj)
        if max(abs(di), abs(dj)) < int(cfg.min_shift_cells):
            self._receipt({"decision": "suppressed:dead-band",
                           "shift_parent_cells": [di, dj], **evidence})
            return None
        # Cooldown anchor: the last EXECUTED move once the runner reports
        # one through notify_move_executed; until then, the last proposal
        # (the pre-hook fallback, so an old runner still gets hysteresis).
        anchor = (self._last_move_t if self._last_move_t is not None
                  else self._last_proposal_t)
        if (anchor is not None
                and float(t) - anchor < float(cfg.cooldown_seconds)):
            self._receipt({
                "decision": "suppressed:cooldown",
                "shift_parent_cells": [di, dj],
                "cooldown_anchor": ("executed-move"
                                    if self._last_move_t is not None
                                    else "proposal"),
                "cooldown_remaining_s": round(
                    float(cfg.cooldown_seconds) - (float(t) - anchor), 3),
                **evidence})
            return None
        limit = int(cfg.max_shift_cells)
        clamped = max(abs(di), abs(dj)) > limit
        di = max(-limit, min(limit, di))
        dj = max(-limit, min(limit, dj))
        di, dj, clipped = self._clip_to_parent(fp, plane.shape, di, dj)
        if di == 0 and dj == 0:
            # Clamp/clip can collapse a real proposal to the null move
            # (storm leaving through a parent edge the nest already
            # hugs).  The null move is never proposed.
            self._receipt({"decision": "suppressed:at-parent-edge",
                           "clamped": bool(clamped),
                           "clipped_to_parent": clipped, **evidence})
            return None
        self._last_proposal_t = float(t)
        self._receipt({"decision": "proposed",
                       "shift_parent_cells": [di, dj],
                       "clamped": bool(clamped),
                       "clipped_to_parent": clipped, **evidence})
        return (di, dj)

    __call__ = desired_shift

    def notify_move_executed(self, t: float,
                             executed_shift: tuple[int, int]) -> None:
        """Optional runner hook: a proposal was actually EXECUTED.

        The leg-2 runner probes for this duck-typed hook and calls it with
        the clamped shift it really applied, after the relocation
        succeeds.  From the first call on, the cooldown counts from
        executed moves instead of proposals -- a proposal the runner
        refused (leg-1 admissibility, a busy cycle) no longer burns the
        cooldown, so the tracker may re-propose at the next cadence.  A
        runner that never calls it keeps the proposal-burn fallback.
        """
        di, dj = (int(executed_shift[0]), int(executed_shift[1]))
        self._last_move_t = float(t)
        self._receipt({"decision": "move-executed", "t": float(t),
                       "executed_shift_parent_cells": [di, dj]})

    @staticmethod
    def _clip_to_parent(fp: NestFootprint, plane_shape: tuple[int, int],
                        di: int, dj: int) -> tuple[int, int, bool]:
        """Keep the proposed footprint PARENT_EDGE_KEEPOUT_CELLS clear of
        the parent's edge; the leg-1 admissibility check still has the
        final word."""
        ny, nx = int(plane_shape[-2]), int(plane_shape[-1])
        keep = PARENT_EDGE_KEEPOUT_CELLS
        i_lo = int(fp.i_parent_start) - 1
        j_lo = int(fp.j_parent_start) - 1
        di_min = keep - i_lo
        di_max = int(math.floor(nx - 1 - keep - fp.span_parent_i)) - i_lo
        dj_min = keep - j_lo
        dj_max = int(math.floor(ny - 1 - keep - fp.span_parent_j)) - j_lo
        # A footprint already inside the keepout must not be dragged; only
        # shrink the proposal toward zero, never push past it.
        out_i = _clip_toward_zero(di, di_min, di_max)
        out_j = _clip_toward_zero(dj, dj_min, dj_max)
        return out_i, out_j, (out_i != di or out_j != dj)


def _clip_toward_zero(step: int, lo: int, hi: int) -> int:
    """Clip ``step`` into [lo, hi] without ever crossing zero: a keepout
    violation shrinks a proposal, it never manufactures a move."""
    lo = min(lo, 0)
    hi = max(hi, 0)
    return max(lo, min(hi, step))


def make_plan_provider(experiment) -> StormTracker | None:
    """The runner's one-line hookup: a tracker from the experiment config.

    Returns ``None`` when the experiment carries no ``[relocation.follow]``
    block -- relocation stays available as a manual/API mechanism exactly
    as leg 1 shipped it.
    """
    relocation = getattr(experiment, "relocation", None)
    follow = getattr(relocation, "follow", None)
    if follow is None:
        return None
    if not getattr(relocation, "enabled", False):
        raise TrackerRefusal(
            "a follow block on a disabled [relocation] cannot exist; the "
            "config loader refuses it, so this experiment object was built "
            "by hand inconsistently")
    return StormTracker(follow)


__all__ = [
    "FOLLOW_CONTRACT", "FOLLOW_KEYS", "FollowConfig", "NestFootprint",
    "PARENT_EDGE_KEEPOUT_CELLS", "REFLECTIVITY_SLOT", "StormTracker",
    "TRACKED_FIELDS", "TRACKER_STATE_KEYS", "TrackerRefusal", "UH_SLOT",
    "build_follow_config",
    "make_plan_provider", "signal_plane", "weighted_centroid",
]
