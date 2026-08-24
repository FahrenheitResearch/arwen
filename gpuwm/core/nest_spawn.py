"""Nest spawn-at-trigger: a dormant nest's WHEN and WHERE.

:mod:`gpuwm.core.storm_tracking` decides when and where an EXISTING nest
moves.  This module is the same decision for a nest that does not exist
yet: a ``[[domain]]`` declared with a ``spawn`` table is DORMANT -- fully
declared (size, resolution, physics, a placeholder placement), reserved
in the memory plan at startup exactly as if it existed, and integrating
nothing -- until its trigger fires on the RUNNING PARENT'S OWN FIELDS,
at which point it materializes at the trigger-chosen placement and is a
static nest from then on ([relocation.follow] takes over if configured).

RESERVATION, STATED PLAINLY.  A declared-but-never-triggered nest costs
its full reserved VRAM for the whole run and zero compute.  That is the
design, not an accident: the reservation is what makes VRAM
deterministic, lets preflight refuse honestly at planning time, and
makes spawning an ACTIVATION rather than a mid-run allocation that can
OOM after hours of integration.  ``gpuwm check`` says this per dormant
domain (:func:`gpuwm.core.preflight.check_advisories`).

THE SEAM (the spawn-provider contract).  The runner side -- the same
cadence loop that consumes :class:`~gpuwm.core.storm_tracking
.StormTracker` -- consumes a :class:`SpawnController` through one call,
evaluated once per relocation cadence (a cycle boundary, never
mid-step)::

    controller.evaluate(grid_id, parent_state, t,
                        active_footprints=...) -> SpawnEvent | None

``parent_state`` is the dormant domain's PARENT's live state; ``t`` is
model seconds since the experiment start (model time, so the window
arithmetic is deterministic); ``active_footprints`` is where the
parent's LIVE children currently sit, as
:class:`~gpuwm.core.storm_tracking.NestFootprint`-coercible objects --
position state lives with the runner, exactly as it does for the
tracker, so this module never disagrees with the runner about the tree.
A returned :class:`SpawnEvent` names the domain and the chosen
whole-parent-cell placement; the runner materializes it through
:func:`gpuwm.ingest.nest_spawn_init.spawn_child_from_parent` and the
leg/schedule surgery of :func:`gpuwm.experiment.active_experiment`.  A
controller proposes; it never builds anything.

MULTIPLE DORMANT NESTS are first-class: each ``[[domain]]`` carries its
own trigger and its own reservation, and two nests must be able to fire
on two different storms.  Two rules provide that:

- per-nest search boxes (``search_box``, or the declared footprint plus
  the ``[relocation.follow]`` margin, or the whole parent); and
- exclusion: a trigger IGNORES signal inside another ACTIVE nest's
  footprint -- the simplest honest rule for "that storm is taken".
  :meth:`SpawnController.evaluate_all` additionally feeds each event
  fired at a boundary into the exclusion set of the nests evaluated
  after it (grid_id order), so two triggers at one boundary cannot
  claim the same centroid.

CONFIG.  ``spawn = { ... }`` inline table on the dormant ``[[domain]]``:
``trigger`` (``"uh"`` | ``"reflectivity"`` | ``"time"``).  Field
triggers require ``threshold`` (m2 s-2 for uh, dBZ for reflectivity),
``earliest_s`` and ``latest_s`` (the model-time window; after
``latest_s`` the watch closes and the nest never spawns), and admit an
optional ``search_box = [i_lo, j_lo, i_hi, j_hi]`` (1-based inclusive
parent cells).  ``trigger = "time"`` is the manual, deterministic form
for testability: it requires ``at_s`` alone and spawns at the DECLARED
placement at the first cadence at or after ``at_s``.  Every key is
honored or refused, never ignored, and the constructed config echoes
its values (:meth:`SpawnConfig.to_json`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from gpuwm.core.uh_diag import UH_SPAWN_WINDOW_SLOT
from gpuwm.core.storm_tracking import (NestFootprint, signal_plane,
                                       weighted_centroid)

#: Versioned label carried by every receipt this module emits.
SPAWN_CONTRACT = "gpuwm-nest-spawn.v1"

#: The trigger vocabulary: the tracker's two field signals, plus the
#: manual deterministic time trigger.
SPAWN_TRIGGERS = ("uh", "reflectivity", "time")

#: Keys of the ``spawn`` inline table.  Unknown keys refuse.
SPAWN_KEYS = frozenset({
    "trigger", "threshold", "search_box", "earliest_s", "latest_s", "at_s",
})

_LOG = logging.getLogger("gpuwm.nest_spawn")


class SpawnRefusal(ValueError):
    """A spawn request this module will not serve quietly."""


# ---------------------------------------------------------------------------
# Config: the [[domain]] spawn table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpawnConfig:
    """The validated ``spawn`` table of one dormant ``[[domain]]``.

    ``threshold`` is in the trigger field's own units.  ``search_box``
    is 1-based inclusive parent-cell bounds ``(i_lo, j_lo, i_hi,
    j_hi)``; ``None`` defers to the declared footprint plus the
    ``[relocation.follow]`` margin when a follow block exists, else to
    the whole parent.  ``at_s`` exists only for ``trigger = "time"``.
    """

    trigger: str
    threshold: float | None = None
    search_box: tuple[int, int, int, int] | None = None
    earliest_s: float | None = None
    latest_s: float | None = None
    at_s: float | None = None

    def __post_init__(self) -> None:
        if self.trigger not in SPAWN_TRIGGERS:
            raise ValueError(
                f"spawn trigger must be one of {SPAWN_TRIGGERS}, got "
                f"{self.trigger!r}")
        if self.trigger == "time":
            if self.at_s is None or not math.isfinite(float(self.at_s)) \
                    or float(self.at_s) < 0.0:
                raise ValueError(
                    "spawn trigger = 'time' requires at_s, a finite "
                    "non-negative model time in seconds, got "
                    f"{self.at_s!r}")
            stray = [name for name in
                     ("threshold", "search_box", "earliest_s", "latest_s")
                     if getattr(self, name) is not None]
            if stray:
                raise ValueError(
                    f"spawn trigger = 'time' refuses {stray}: the manual "
                    "trigger spawns at the DECLARED placement at at_s and "
                    "reads no field, so a threshold or window on it would "
                    "be a value nobody consumes")
            return
        if self.at_s is not None:
            raise ValueError(
                f"spawn trigger = {self.trigger!r} refuses at_s; the "
                "field trigger decides its own instant. Use trigger = "
                "'time' for a manual spawn")
        if self.threshold is None or not math.isfinite(float(self.threshold)):
            raise ValueError(
                f"spawn trigger = {self.trigger!r} requires a finite "
                f"threshold in the field's own units, got "
                f"{self.threshold!r}")
        for name in ("earliest_s", "latest_s"):
            value = getattr(self, name)
            if value is None or not math.isfinite(float(value)) \
                    or float(value) < 0.0:
                raise ValueError(
                    f"spawn {name} must be a finite non-negative model "
                    f"time in seconds, got {value!r}; the window is chosen "
                    "deliberately -- there are no defaults to inherit")
        if float(self.latest_s) <= float(self.earliest_s):
            raise ValueError(
                f"spawn latest_s = {self.latest_s!r} must exceed "
                f"earliest_s = {self.earliest_s!r}; an empty window is a "
                "disabled feature wearing an enabled name")
        if self.search_box is not None:
            box = tuple(int(v) for v in self.search_box)
            if len(box) != 4:
                raise ValueError(
                    "spawn search_box must be [i_lo, j_lo, i_hi, j_hi] "
                    f"(1-based inclusive parent cells), got "
                    f"{self.search_box!r}")
            i_lo, j_lo, i_hi, j_hi = box
            if i_lo < 1 or j_lo < 1 or i_hi < i_lo or j_hi < j_lo:
                raise ValueError(
                    f"spawn search_box {box} is not an ordered 1-based "
                    "box: require 1 <= i_lo <= i_hi and 1 <= j_lo <= j_hi")
            object.__setattr__(self, "search_box", box)

    def to_json(self) -> dict[str, object]:
        """Echo every configured value (the receipts' config record)."""
        out: dict[str, object] = {
            "contract": SPAWN_CONTRACT,
            "trigger": self.trigger,
        }
        if self.trigger == "time":
            out["at_s"] = float(self.at_s)
            return out
        out["threshold"] = float(self.threshold)
        out["earliest_s"] = float(self.earliest_s)
        out["latest_s"] = float(self.latest_s)
        if self.search_box is not None:
            out["search_box"] = [int(v) for v in self.search_box]
        return out


def _require_number(table: dict, key: str, source: str, label: str):
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{key} in {label} of {source} must be a number, got {value!r}")
    return value


def build_spawn_config(table: dict, source: str, *,
                       grid_id: int) -> SpawnConfig:
    """Validate one parsed ``spawn`` inline table.

    Honored or refused, never ignored: unknown keys are refused by name,
    the keys each trigger requires are required by name, and a key a
    trigger cannot consume is refused rather than dropped.
    """
    from gpuwm.experiment import did_you_mean

    label = f"[[domain]] grid_id={grid_id} spawn"
    unknown = sorted(set(table) - SPAWN_KEYS)
    if unknown:
        named = ", ".join(
            f"{key!r}{did_you_mean(key, SPAWN_KEYS)}" for key in unknown)
        raise ValueError(
            f"{label} of {source} does not have key(s) {named}; no key is "
            "ignored, because a dropped key spawns a nest on a value "
            "nobody chose.")
    if "trigger" not in table:
        raise ValueError(
            f"{label} of {source} is missing the required key 'trigger' "
            f"(one of {SPAWN_TRIGGERS}); present: {sorted(table)}.")
    trigger = table["trigger"]
    if not isinstance(trigger, str):
        raise ValueError(
            f"trigger in {label} of {source} must be a string, got "
            f"{trigger!r}")
    kwargs: dict[str, object] = {"trigger": trigger}
    for key in ("threshold", "earliest_s", "latest_s", "at_s"):
        if key in table:
            kwargs[key] = float(_require_number(table, key, source, label))
    if "search_box" in table:
        box = table["search_box"]
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, int)
                       for v in box)):
            raise ValueError(
                f"search_box in {label} of {source} must be a 4-integer "
                f"array [i_lo, j_lo, i_hi, j_hi], got {box!r}")
        kwargs["search_box"] = tuple(int(v) for v in box)
    try:
        return SpawnConfig(**kwargs)
    except ValueError as err:
        raise ValueError(f"{label} of {source}: {err}") from None


# ---------------------------------------------------------------------------
# The event a fired trigger produces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpawnEvent:
    """One fired trigger: which domain, and where it materializes."""

    grid_id: int
    i_parent_start: int
    j_parent_start: int
    t: float
    receipt: dict

    @property
    def position(self) -> tuple[int, int]:
        return (int(self.i_parent_start), int(self.j_parent_start))


# ---------------------------------------------------------------------------
# One dormant nest's watch
# ---------------------------------------------------------------------------

class SpawnWatch:
    """Trigger evaluation for ONE dormant domain.

    Stateful in exactly two ways: whether it has fired (a spawn is a
    one-shot) and whether its window has closed.  Everything else --
    the parent state, the model time, where the live siblings sit --
    arrives as arguments every call, so a watch can never disagree with
    the runner about the world.
    """

    def __init__(self, config: SpawnConfig, declared: NestFootprint, *,
                 keepout_cells: int, follow=None) -> None:
        if not isinstance(config, SpawnConfig):
            raise TypeError("config must be a SpawnConfig")
        self.config = config
        self.declared = NestFootprint.coerce(declared)
        if int(keepout_cells) < 1:
            raise SpawnRefusal(
                f"keepout_cells must be >= 1 parent cell, got "
                f"{keepout_cells!r}")
        #: Parent rows the chosen placement keeps clear of the parent
        #: edge on every side: the loader's clearance rule
        #: (spec_bdy_width + blend_width), so a fired placement re-passes
        #: the exact admission check a declared one did.
        self.keepout_cells = int(keepout_cells)
        self.follow = follow
        self.fired = False
        self.closed = False
        self.receipts: list[dict] = []
        self._receipt({"decision": "configured",
                       "grid_id": int(self.declared.grid_id),
                       "declared_placement": [
                           int(self.declared.i_parent_start),
                           int(self.declared.j_parent_start)],
                       "keepout_cells": self.keepout_cells,
                       "config": config.to_json()})

    # -- receipts ----------------------------------------------------------

    def _receipt(self, entry: dict) -> dict:
        entry = {"contract": SPAWN_CONTRACT, **entry}
        self.receipts.append(entry)
        _LOG.info("nest-spawn %s", entry)
        return entry

    def drain_receipts(self) -> list[dict]:
        out, self.receipts = self.receipts, []
        return out

    def rearm(self, *, t: float, episode: int) -> None:
        """Return a one-shot watch to its declared dormant state."""
        self.fired = False
        self.closed = False
        self._receipt({"decision": "rearmed", "grid_id": int(self.declared.grid_id),
                       "t": float(t), "episode": int(episode)})

    # -- geometry ----------------------------------------------------------

    def _search_box(self, plane_shape) -> tuple[tuple[slice, slice], str]:
        ny, nx = int(plane_shape[-2]), int(plane_shape[-1])
        cfg = self.config
        if cfg.search_box is not None:
            i_lo, j_lo, i_hi, j_hi = cfg.search_box
            j_slice = slice(max(0, j_lo - 1), min(ny, j_hi))
            i_slice = slice(max(0, i_lo - 1), min(nx, i_hi))
            if j_slice.start >= j_slice.stop or i_slice.start >= i_slice.stop:
                raise SpawnRefusal(
                    f"spawn search_box {cfg.search_box} lies outside the "
                    f"parent plane {(ny, nx)}; the box and the parent "
                    "state disagree about the tree geometry")
            return (j_slice, i_slice), "explicit"
        if self.follow is not None:
            margin = int(getattr(self.follow, "search_margin_cells"))
            return (self.declared.search_box(plane_shape, margin),
                    "declared-footprint+follow-margin")
        return ((slice(0, ny), slice(0, nx)), "whole-parent")

    def _placement_bounds(self, plane_shape) -> tuple[int, int, int, int]:
        """(i_min, i_max, j_min, j_max) admissible 1-based starts."""
        ny, nx = int(plane_shape[-2]), int(plane_shape[-1])
        fp = self.declared
        span_i = int(fp.child_nx) // int(fp.parent_grid_ratio)
        span_j = int(fp.child_ny) // int(fp.parent_grid_ratio)
        need = self.keepout_cells
        i_min, i_max = need + 1, nx - span_i + 1 - need
        j_min, j_max = need + 1, ny - span_j + 1 - need
        if i_min > i_max or j_min > j_max:
            raise SpawnRefusal(
                f"the declared d{fp.grid_id:02d} footprint "
                f"({fp.child_nx}x{fp.child_ny} at ratio "
                f"{fp.parent_grid_ratio}) cannot be placed anywhere in a "
                f"{ny}x{nx} parent with {need} rows of edge clearance; "
                "the nest is too large to spawn at all, which is a "
                "configuration error, not a quiet storm")
        return i_min, i_max, j_min, j_max

    def _placement_from_centroid(self, ci: float, cj: float,
                                 plane_shape) -> tuple[int, int, bool]:
        """Center the footprint on the centroid, whole-parent-cell
        aligned per leg-1's relocation convention, clamped to keepout."""
        fp = self.declared
        i_min, i_max, j_min, j_max = self._placement_bounds(plane_shape)
        # leg-1 center convention: center = (start - 1) + span/2 with the
        # donor span (nx - 1)/ratio, so start = centroid - span/2 + 1.
        raw_i = ci - fp.span_parent_i / 2.0 + 1.0
        raw_j = cj - fp.span_parent_j / 2.0 + 1.0
        i_start = int(math.floor(raw_i + 0.5))
        j_start = int(math.floor(raw_j + 0.5))
        clamped_i = min(max(i_start, i_min), i_max)
        clamped_j = min(max(j_start, j_min), j_max)
        return (clamped_i, clamped_j,
                clamped_i != i_start or clamped_j != j_start)

    @staticmethod
    def _mask_active_footprints(plane: np.ndarray,
                                footprints) -> tuple[np.ndarray, list]:
        """Zero out signal a live nest already owns (the exclusion rule)."""
        masked: list[dict] = []
        if not footprints:
            return plane, masked
        out = np.array(plane, copy=True)
        ny, nx = out.shape[-2], out.shape[-1]
        for value in footprints:
            fp = NestFootprint.coerce(value)
            i0 = max(0, int(fp.i_parent_start) - 1)
            j0 = max(0, int(fp.j_parent_start) - 1)
            i1 = min(nx, int(math.ceil(i0 + fp.span_parent_i)) + 1)
            j1 = min(ny, int(math.ceil(j0 + fp.span_parent_j)) + 1)
            if i0 < i1 and j0 < j1:
                out[..., j0:j1, i0:i1] = -np.inf
                masked.append({"grid_id": int(fp.grid_id),
                               "cells": [[j0, j1], [i0, i1]]})
        return out, masked

    # -- the contract ------------------------------------------------------

    def evaluate(self, parent_state, t: float, *,
                 exclude_footprints=()) -> SpawnEvent | None:
        """One cadence evaluation: a SpawnEvent, or None with a receipt."""
        cfg = self.config
        gid = int(self.declared.grid_id)
        if self.fired:
            return None
        if cfg.trigger == "time":
            if float(t) < float(cfg.at_s):
                self._receipt({"decision": "waiting:before-at_s",
                               "grid_id": gid, "t": float(t),
                               "at_s": float(cfg.at_s)})
                return None
            self.fired = True
            receipt = self._receipt({
                "decision": "fired",
                "grid_id": gid, "t": float(t),
                "trigger": "time", "at_s": float(cfg.at_s),
                "placement": [int(self.declared.i_parent_start),
                              int(self.declared.j_parent_start)],
                "placement_source": "declared",
            })
            return SpawnEvent(
                grid_id=gid,
                i_parent_start=int(self.declared.i_parent_start),
                j_parent_start=int(self.declared.j_parent_start),
                t=float(t), receipt=receipt)
        if self.closed:
            return None
        if float(t) > float(cfg.latest_s):
            self.closed = True
            self._receipt({"decision": "window-closed", "grid_id": gid,
                           "t": float(t), "latest_s": float(cfg.latest_s),
                           "note": "the nest will never spawn; its "
                                   "reservation remains held (activation "
                                   "is the only thing that was declined)"})
            return None
        if float(t) < float(cfg.earliest_s):
            self._receipt({"decision": "waiting:before-window",
                           "grid_id": gid, "t": float(t),
                           "earliest_s": float(cfg.earliest_s)})
            return None
        # THIS consumer's window, not the relocation runner's and not
        # the history-reset diagnostic: a spawn watch evaluates on LEG
        # boundaries, a different rhythm from the relocation cadence,
        # so it owns and resets its own max-since-I-last-looked.
        plane = signal_plane(parent_state, cfg.trigger,
                             uh_slot=UH_SPAWN_WINDOW_SLOT)
        box, box_source = self._search_box(plane.shape)
        plane_masked, masked = self._mask_active_footprints(
            plane, exclude_footprints)
        evidence: dict[str, object] = {
            "grid_id": gid, "t": float(t),
            "field": cfg.trigger, "threshold": float(cfg.threshold),
            "search_box": [[int(box[0].start), int(box[0].stop)],
                           [int(box[1].start), int(box[1].stop)]],
            "search_box_source": box_source,
            "excluded_active_footprints": masked,
        }
        # TWO STORMS, ONE BOX: a box-wide weighted centroid of two
        # exceedance regions lands BETWEEN them -- a birth position on
        # neither storm.  So the loudest cell claims the nest first
        # (argmax within the box), and the centroid is then computed
        # over a footprint-sized window around it: the position is the
        # weighted center of THAT storm alone, which is also what makes
        # the exclusion rule compose -- once a fired sibling masks storm
        # one, the next watch's argmax is storm two.
        window = plane_masked[box[0], box[1]]
        with np.errstate(invalid="ignore"):
            qualifying = np.isfinite(window) & (
                window >= float(cfg.threshold))
        if not bool(qualifying.any()):
            self._receipt({"decision": "no-signal", **evidence})
            return None
        peak_flat = int(np.argmax(np.where(qualifying, window, -np.inf)))
        peak_j, peak_i = np.unravel_index(peak_flat, window.shape)
        peak_j = int(peak_j) + int(box[0].start)
        peak_i = int(peak_i) + int(box[1].start)
        half_j = max(1, int(math.ceil(self.declared.span_parent_j / 2.0)))
        half_i = max(1, int(math.ceil(self.declared.span_parent_i / 2.0)))
        local = (slice(max(box[0].start, peak_j - half_j),
                       min(box[0].stop, peak_j + half_j + 1)),
                 slice(max(box[1].start, peak_i - half_i),
                       min(box[1].stop, peak_i + half_i + 1)))
        evidence["peak_parent_ij"] = [peak_i, peak_j]
        evidence["local_window"] = [
            [int(local[0].start), int(local[0].stop)],
            [int(local[1].start), int(local[1].stop)]]
        found = weighted_centroid(plane_masked, float(cfg.threshold), local)
        if found is None:
            # Cannot happen while the peak qualifies; kept as a loud
            # guard rather than an assumption.
            self._receipt({"decision": "no-signal", **evidence})
            return None
        i_start, j_start, clamped = self._placement_from_centroid(
            found["ci"], found["cj"], plane.shape)
        self.fired = True
        receipt = self._receipt({
            "decision": "fired",
            "trigger": cfg.trigger,
            "cells_above_threshold": int(found["cells"]),
            "max_value": round(float(found["max_value"]), 3),
            "centroid_parent_ij": [round(float(found["ci"]), 3),
                                   round(float(found["cj"]), 3)],
            "placement": [int(i_start), int(j_start)],
            "placement_source": "centroid",
            "clamped_to_keepout": bool(clamped),
            "keepout_cells": self.keepout_cells,
            **evidence,
        })
        return SpawnEvent(grid_id=gid, i_parent_start=int(i_start),
                          j_parent_start=int(j_start), t=float(t),
                          receipt=receipt)


# ---------------------------------------------------------------------------
# The controller (the spawn provider)
# ---------------------------------------------------------------------------

class SpawnController:
    """Every dormant nest of one experiment, behind the runner seam."""

    def __init__(self, watches: dict[int, SpawnWatch],
                 parent_of: dict[int, int]) -> None:
        self.watches = dict(watches)
        self.parent_of = dict(parent_of)

    @classmethod
    def from_experiment(cls, experiment) -> "SpawnController | None":
        """The runner's one-line hookup, mirroring ``make_plan_provider``.

        Returns ``None`` when the experiment declares no dormant nest.
        The clearance keepout is the experiment's own admission rule
        (``spec_bdy_width + blend_width``), so a fired placement
        re-passes exactly the check a declared one did.
        """
        watches: dict[int, SpawnWatch] = {}
        parent_of: dict[int, int] = {}
        relocation = getattr(experiment, "relocation", None)
        follow = getattr(relocation, "follow", None)
        keepout = (int(getattr(experiment, "spec_bdy_width", 5))
                   + int(getattr(experiment, "blend_width", 5)))
        for dc in experiment.domains:
            spawn = getattr(dc, "spawn", None)
            if spawn is None:
                continue
            domain_follow = getattr(dc, "follow", None)
            watches[int(dc.grid_id)] = SpawnWatch(
                spawn, NestFootprint.coerce(dc), keepout_cells=keepout,
                follow=(domain_follow.tracker if domain_follow is not None
                        else (follow if getattr(relocation, "grid_id", None)
                              in (None, dc.grid_id) else None)))
            parent_of[int(dc.grid_id)] = int(dc.parent_id)
        if not watches:
            return None
        return cls(watches, parent_of)

    @property
    def pending(self) -> tuple[int, ...]:
        """Dormant domains still able to fire, in grid_id order."""
        return tuple(sorted(
            gid for gid, watch in self.watches.items()
            if not watch.fired and not watch.closed))

    def evaluate(self, grid_id: int, parent_state, t: float, *,
                 active_footprints=()) -> SpawnEvent | None:
        """Evaluate ONE dormant nest's trigger (the seam call)."""
        watch = self.watches.get(int(grid_id))
        if watch is None:
            raise SpawnRefusal(
                f"grid_id {grid_id} is not a declared dormant nest; "
                f"declared: {sorted(self.watches)}")
        return watch.evaluate(parent_state, t,
                              exclude_footprints=active_footprints)

    def evaluate_all(self, parent_states, t: float, *,
                     active_footprints=()) -> tuple[SpawnEvent, ...]:
        """Evaluate every pending watch at one cadence boundary.

        ``parent_states`` maps parent grid_id -> live parent state.
        Watches are processed in grid_id order and every event fired at
        this boundary immediately joins the exclusion set of the watches
        after it, so two nests cannot claim one storm even when both
        triggers cross threshold on the same boundary.
        """
        events: list[SpawnEvent] = []
        exclusions = list(active_footprints)
        for gid in self.pending:
            parent_id = self.parent_of[gid]
            if parent_id not in parent_states:
                # Escalation ladders legitimately declare a dormant child of
                # another dormant/retired SLOT.  Preserve the historical
                # refusal for every other missing parent: only a declared
                # watch proves that absence is intentional lifecycle state.
                if parent_id not in self.watches:
                    raise SpawnRefusal(
                        f"no parent state supplied for d{gid:02d}'s parent "
                        f"d{parent_id:02d}; supplied: {sorted(parent_states)}")
                self.watches[gid]._receipt({
                    "decision": "held:parent-not-live", "t": float(t),
                    "grid_id": int(gid), "parent_id": int(parent_id)})
                continue
            event = self.watches[gid].evaluate(
                parent_states[parent_id], t,
                exclude_footprints=tuple(exclusions))
            if event is None:
                continue
            events.append(event)
            declared = self.watches[gid].declared
            exclusions.append(NestFootprint(
                grid_id=gid,
                i_parent_start=event.i_parent_start,
                j_parent_start=event.j_parent_start,
                child_nx=declared.child_nx, child_ny=declared.child_ny,
                parent_grid_ratio=declared.parent_grid_ratio))
        return tuple(events)

    def drain_receipts(self) -> list[dict]:
        out: list[dict] = []
        for gid in sorted(self.watches):
            out.extend(self.watches[gid].drain_receipts())
        return out


__all__ = [
    "SPAWN_CONTRACT", "SPAWN_KEYS", "SPAWN_TRIGGERS", "SpawnConfig",
    "SpawnController", "SpawnEvent", "SpawnRefusal", "SpawnWatch",
    "build_spawn_config",
]
