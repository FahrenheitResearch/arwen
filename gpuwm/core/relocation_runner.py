"""Cadence-driven execution of discrete nest relocations inside a run.

:mod:`gpuwm.core.nest_relocation` is the mechanism -- it moves a live
child once, when told where.  :mod:`gpuwm.core.storm_tracking` is the
policy -- it proposes where.  This module is the runner between them: at
every complete cycle boundary that lies on the configured cadence, ask
the follow source for a desired whole-parent-cell shift and execute it
through :func:`~gpuwm.core.nest_relocation.relocate_child` with host
staging, admissibility bounds, and a receipt appended to the run's
relocation receipts.

WHERE IT FIRES.  ``execute_experiment`` calls
:meth:`RelocationRunner.on_period_begin` at PERIOD_BEGIN, the instant
every domain clock in the tree sits on the same complete d01 boundary.
Every such boundary is a parent-step boundary for every child, which is
the synchronization :func:`relocate_child` requires; the cadence is
therefore validated to be a whole number of periods and can never ask
for a mid-step move.  ``cadence_seconds`` absent means every boundary,
with the tracker's own dead-band/cooldown as the rate limit.

THE FOLLOW SOURCE is the storm tracker's published plan-provider seam
(:mod:`gpuwm.core.storm_tracking`, adopted verbatim)::

    provider(parent_state, nest_footprint, t) -> (di, dj) | None

``parent_state`` is the PARENT domain's live state; ``nest_footprint``
is anything :class:`~gpuwm.core.storm_tracking.NestFootprint` can
coerce (the runner passes the child's live DomainConfig); ``t`` is model
seconds.  The return is the desired shift in WHOLE PARENT CELLS (``di``
increments ``i_parent_start``, ``dj`` increments ``j_parent_start``) or
``None`` for hold.  :class:`gpuwm.core.storm_tracking.StormTracker`
implements it; :class:`ManualMoveProvider` implements the SAME contract
from the config's ``[[relocation.move]]`` rows, so the manual mode is a
standing test of the exact interface a tracker must satisfy.

The provider proposes; the runner enforces.  An executed shift is the
proposal clamped to ``max_move_parent_cells`` and to the parent's
admissible placement band, and both the requested and the executed shift
land in the receipt -- a clamped decision is visible, never silent.
After an executed move the runner calls the provider's OPTIONAL
``notify_move_executed(t, executed_shift)`` hook (accepted-move
semantics for trackers whose cooldown should count acceptances rather
than proposals); a provider without the hook simply is not notified.

A restart across a move promises nothing (Drew, 2026-08-06).  After
every executed move the runner chains the record into the model's live
restart fingerprint
(:func:`~gpuwm.core.nest_relocation.mark_fingerprint_across_move`), so
checkpoints written later refuse to resume into a fresh build BY
CONSTRUCTION and the refusal names relocation.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from fractions import Fraction
from pathlib import Path

from gpuwm.core.nest_relocation import (RelocationRefusal, RelocationSegment,
                                        base_segment,
                                        mark_fingerprint_across_move,
                                        relocate_child)

#: Versioned label for the runner's own receipt rows.
from gpuwm.core.uh_diag import (UH_FOLLOW_WINDOW_SLOT,
                               reset_tracker_window)

RELOCATION_RUNNER_CONTRACT = "gpuwm-nest-relocation-runner.v1"

#: The keys of a checkpoint's per-follower entry that the RUNNER owns.
RUNNER_STATE_KEYS = ("segment", "moves_executed", "last_proposal_t",
                     "last_move_t")

#: The rest of the same entry, filled by the checkpoint writer because it
#: describes the TREE rather than this runner's history.  Tolerated on the
#: way back in so a resume hands the entry over whole instead of stripping
#: it first, and never read here.
RUNNER_WRITER_KEYS = ("kind", "current_placement", "declared_placement")

#: What fills the strip a move exposes.  Stated on every move receipt so
#: the fill provenance is a recorded fact, not an assumption: the child
#: is rebuilt at the new placement by the ordinary cold-start path (a
#: full SINT of the LIVE parent), and the overlap is then stamped bitwise
#: from the outgoing child, so only the newly exposed strip keeps the
#: parent-interpolated values.
STRIP_FILL_SOURCE = ("parent_only_init: cold start from the live parent "
                     "(SINT), overlap then stamped from the outgoing child")


class ManualMoveProvider:
    """The config-driven follow source: ``[[relocation.move]]`` rows.

    Implements the storm tracker's plan-provider contract
    (``provider(parent_state, nest_footprint, t)``) so the runner cannot
    tell it from a tracker.  Each row fires at its exact ``at_seconds``
    opportunity and exactly once; opportunities with no row are holds.
    The rows were validated at config admission (strictly increasing,
    on-boundary, inside the run), so a row left unconsumed at run end is
    a runner defect, and :meth:`unconsumed` lets the receipts say so.
    """

    def __init__(self, moves):
        self._pending = list(moves)

    def __call__(self, parent_state, nest_footprint, t):
        del parent_state, nest_footprint  # the itinerary is time-driven
        if not self._pending:
            return None
        head = self._pending[0]
        if abs(float(head.at_seconds) - float(t)) > (
                1.0e-6 * max(1.0, abs(float(head.at_seconds)))):
            return None
        self._pending.pop(0)
        return (int(head.di_parent_cells), int(head.dj_parent_cells))

    desired_shift = __call__

    def unconsumed(self):
        return tuple(move.to_json() for move in self._pending)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                   encoding="utf-8")
    # Windows refuses the atomic rename while ANY reader holds the
    # destination open -- and this file exists precisely to be read
    # while the run is alive (a user tailing the receipts killed a 6 h
    # forecast with WinError 5 here, measured).  Receipts are telemetry:
    # retry briefly, then keep the run alive and leave the .tmp beside
    # the stale receipt rather than aborting integration over it.
    for delay in (0.05, 0.25, 1.0, 2.0):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(delay)
    try:
        os.replace(tmp, path)
    except PermissionError as error:
        warnings.warn(
            f"relocation receipts stayed stale at {path} (a reader holds "
            f"it open: {error}); the current receipts are in {tmp} and "
            "the next move rewrites both", stacklevel=2)


def _cadence_periods(cadence_seconds, schedule) -> int:
    """The cadence as whole periods; ``None`` means every period."""
    if cadence_seconds is None:
        return 1
    tick_den = int(schedule.clock.tick_den)
    period_ticks = int(schedule.period_ticks)
    cadence_ticks = Fraction(float(cadence_seconds)) * tick_den
    if (cadence_ticks.denominator != 1
            or int(cadence_ticks) % period_ticks != 0):
        raise RelocationRefusal(
            f"cadence_seconds = {cadence_seconds} does not resolve to a "
            f"whole number of complete cycle boundaries (period = "
            f"{period_ticks / tick_den} s); the config-time root-step "
            "check and this schedule disagree, which is a defect, not an "
            "input error")
    periods = int(cadence_ticks) // period_ticks
    if periods < 1:
        raise RelocationRefusal(
            f"cadence_seconds = {cadence_seconds} is shorter than one "
            "complete cycle; there is no boundary inside a period to "
            "relocate at")
    return periods


class RelocationRunner:
    """Consult the follow source on cadence and execute admissible moves.

    Construction is the route's job because the route owns the two things
    neither the mechanism nor the policy can invent: the
    ``on_child_built`` preparer that gives a rebuilt child its physics,
    and the model tree the provider watches.  ``staging`` defaults to
    ``"host"`` -- the production trade -- and is a parameter, not a
    config key, because it changes memory behaviour and nothing else.
    """

    def __init__(self, *, config, schedule, on_child_built,
                 provider=None, staging: str = "host",
                 receipts_path=None, initializer=None,
                 static_provenance=None):
        if not getattr(config, "enabled", False):
            raise RelocationRefusal(
                "RelocationRunner requires [relocation] enabled = true; a "
                "disabled config schedules nothing and must not acquire a "
                "runner")
        if on_child_built is None:
            raise RelocationRefusal(
                "RelocationRunner requires the route's on_child_built "
                "preparer: the cold-start path attaches no physics driver, "
                "and a child rebuilt without one integrates nothing")
        if provider is None:
            if getattr(config, "moves", ()):
                provider = ManualMoveProvider(config.moves)
            elif getattr(config, "follow", None) is not None:
                from gpuwm.core.storm_tracking import StormTracker

                provider = StormTracker(config.follow)
            else:
                raise RelocationRefusal(
                    "[relocation] names no follow source: no "
                    "[relocation.follow] tracker block, no "
                    "[[relocation.move]] itinerary, and no provider was "
                    "handed in.  A runner with nothing to consult is a "
                    "nest that silently never moves, so this refuses")
        self.config = config
        self.provider = provider
        self.on_child_built = on_child_built
        self.staging = str(staging)
        self.receipts_path = (None if receipts_path is None
                              else Path(receipts_path))
        self.cadence_periods = _cadence_periods(
            getattr(config, "cadence_seconds", None), schedule)
        #: The per-footprint child rebuild seam (2026-08-06 statics-on-
        #: move requirement): a route that rebuilds own-source statics at
        #: the new footprint passes its builder here, with the provenance
        #: it must state; None takes relocate_child's explicit
        #: parent-interpolated fallback, which every receipt records.
        self.initializer = initializer
        self.static_provenance = static_provenance
        self.receipts: list[dict] = []
        self.moves_executed = 0
        self._segment = None

    @classmethod
    def from_experiment(cls, exp, *, schedule, on_child_built,
                        provider=None, staging: str = "host",
                        receipts_path=None, initializer=None,
                        static_provenance=None) -> "RelocationRunner":
        """Build from a validated experiment (the routes' entry point).

        ``provider`` overrides the config-derived source only for the
        no-source case (a route handing in a bespoke tracker); a config
        that already names a source refuses a second one.
        """
        relocation = exp.relocation
        if provider is not None and (
                getattr(relocation, "moves", ())
                or getattr(relocation, "follow", None) is not None):
            raise RelocationRefusal(
                "this [relocation] already names its follow source "
                "([relocation.follow] or [[relocation.move]]); a second, "
                "programmatic provider would silently shadow it")
        return cls(config=relocation, schedule=schedule,
                   on_child_built=on_child_built, provider=provider,
                   staging=staging, receipts_path=receipts_path,
                   initializer=initializer,
                   static_provenance=static_provenance)

    # -- the restart state -------------------------------------------------
    #
    # PURE STATE, both directions: no I/O, no checkpoint vocabulary, no
    # model.  ``gpuwm.io.restart`` owns where a follower entry lives in the
    # header and fills the three keys that describe the TREE
    # (``kind``, ``current_placement``, ``declared_placement``); the runner
    # owns the four that describe its own history, and tolerates the
    # writer's three so a resume can hand the entry over whole.

    def state_json(self) -> dict:
        """This follower's half of a checkpoint's per-follower entry."""
        provider_state = getattr(self.provider, "state_json", None)
        timers = {} if provider_state is None else dict(provider_state())
        return {
            "segment": (None if self._segment is None
                        else self._segment.to_json()),
            "moves_executed": int(self.moves_executed),
            "last_proposal_t": timers.get("last_proposal_t"),
            "last_move_t": timers.get("last_move_t"),
        }

    def restore_state(self, entry) -> None:
        """Seed this follower from a checkpoint's per-follower entry.

        The segment is the part that cannot be recomputed: it counts moves
        off ONE base preparation, and a follower that resumes at generation
        zero chains its next move's record onto the base instead of onto
        its real predecessor -- so the digest the checkpoint's restart
        fingerprint was folded from can never be reproduced, and every
        later checkpoint of the resumed run is addressed by a history that
        did not happen.

        Validated in full before anything is applied, so a refusal leaves
        the runner exactly as it was.
        """
        if not isinstance(entry, dict):
            raise RelocationRefusal(
                "a follower restart entry must be a mapping, got "
                f"{type(entry).__name__}")
        unknown = sorted(set(entry) - set(RUNNER_STATE_KEYS)
                         - set(RUNNER_WRITER_KEYS))
        if unknown:
            raise RelocationRefusal(
                f"a follower restart entry carries key(s) {unknown} this "
                "build does not know; a follower seeded from policy state "
                "it can only read in part moves the nest from a history "
                "that is not its own, so this refuses")
        missing = sorted(set(RUNNER_STATE_KEYS) - set(entry))
        if missing:
            raise RelocationRefusal(
                f"a follower restart entry is missing {missing}; the "
                "segment fixes which move the next record chains onto, the "
                "counter is the run's account of what it did, and the two "
                "cooldown anchors are what stop the nest moving again at "
                "the first boundary after the resume")

        raw_segment = entry["segment"]
        segment = (None if raw_segment is None
                   else RelocationSegment.from_json(raw_segment))
        moves = int(entry["moves_executed"])
        if moves < 0:
            raise RelocationRefusal(
                f"a follower reports {moves} executed moves; the run cannot "
                "have un-moved a nest")
        recorded = 0 if segment is None else int(segment.generation)
        if recorded != moves:
            raise RelocationRefusal(
                f"a follower reports {moves} executed move(s) but its "
                f"chain carries {recorded}; the counter and the placement "
                "history disagree, and restoring either one would leave the "
                "next move chaining onto the wrong predecessor")

        timers = {"last_proposal_t": entry["last_proposal_t"],
                  "last_move_t": entry["last_move_t"]}
        provider_restore = getattr(self.provider, "restore_state", None)
        if provider_restore is None:
            held = sorted(key for key, value in timers.items()
                          if value is not None)
            if held:
                raise RelocationRefusal(
                    f"this follower's source ({type(self.provider).__name__}) "
                    f"holds no cooldown, but the checkpoint carries {held}. "
                    "Dropping a tracker's hysteresis lets the resumed nest "
                    "move at its first cadence boundary, so the mismatched "
                    "source refuses instead of silently discarding it")
        else:
            provider_restore(timers)

        self._segment = segment
        self.moves_executed = moves

    # -- receipts ---------------------------------------------------------

    def _record(self, model, entry: dict, *, unique: bool = False) -> dict:
        """Append one receipt row, or REPLACE the run's single unique row.

        ``unique`` belongs to the run-end summary, which is a statement
        about the whole run rather than about an instant: it is written
        whenever the executor finishes, and the executor finishes once per
        leg on a route that walks legs (``gpuwm.runtime.walk_spawn_legs``)
        rather than once per run.  Appending gave a receipts list with two
        or more byte-identical summary rows in it, so a consumer counting
        them double-counted.

        The superseded row is DROPPED and the new one appended, not
        overwritten in place: an in-place rewrite would leave the summary
        stranded before the moves that followed it, and an idempotence
        guard that kept the first row would freeze ``moves_executed`` at
        the first leg's total.  One row, current, last.
        """

        entry = {"contract": RELOCATION_RUNNER_CONTRACT, **entry}
        drain = getattr(self.provider, "drain_receipts", None)
        if callable(drain):
            tracker_rows = drain()
            if tracker_rows:
                entry["tracker_receipts"] = tracker_rows
        if unique:
            self.receipts[:] = [row for row in self.receipts
                                if row.get("event") != entry["event"]]
        self.receipts.append(entry)
        if getattr(model, "_relocation_receipts", None) is not self.receipts:
            model._relocation_receipts = self.receipts
        if self.receipts_path is not None:
            _atomic_json(self.receipts_path, {
                "contract": RELOCATION_RUNNER_CONTRACT,
                "config": self.config.receipt(),
                "receipts": self.receipts,
            })
        return entry

    # -- the shift policy -------------------------------------------------

    def _clamped_shift(self, node, shift) -> tuple[int, int, list[str]]:
        """Bound the desired shift; every adjustment is named."""
        from dataclasses import replace as _replace

        from gpuwm.core.nest_relocation import _prevalidate_placement

        di, dj = (int(shift[0]), int(shift[1]))
        clamps: list[str] = []
        limit = self.config.max_move_parent_cells
        if limit is not None:
            limit = int(limit)
            bounded_i = max(-limit, min(limit, di))
            bounded_j = max(-limit, min(limit, dj))
            if (bounded_i, bounded_j) != (di, dj):
                clamps.append("max_move_parent_cells")
                di, dj = bounded_i, bounded_j

        # Walk back toward the current placement until register_nest's
        # +-2 SINT stencil admits the target.  The current placement is
        # live, so the walk terminates; each probe is CPU index tables.
        def admissible(candidate_di, candidate_dj) -> bool:
            candidate = _replace(
                node.cfg,
                i_parent_start=int(node.cfg.i_parent_start) + candidate_di,
                j_parent_start=int(node.cfg.j_parent_start) + candidate_dj)
            if (candidate.i_parent_start < 1
                    or candidate.j_parent_start < 1):
                return False
            try:
                _prevalidate_placement(candidate, node.parent)
            except ValueError:
                return False
            return True

        edge_clamped = False
        while (di, dj) != (0, 0) and not admissible(di, dj):
            edge_clamped = True
            if abs(di) >= abs(dj) and di != 0:
                di -= 1 if di > 0 else -1
            elif dj != 0:
                dj -= 1 if dj > 0 else -1
        if edge_clamped:
            clamps.append("parent_edge")
        return di, dj, clamps

    # -- the cadence hook -------------------------------------------------

    def is_due(self, model, clocks) -> bool:
        """Whether THIS boundary is a cadence opportunity.

        The same gate :meth:`on_period_begin` opens with, published so a
        caller can do work that must happen BEFORE the provider is
        consulted and that would be wasteful at every other boundary.  The
        streamed-domain projection is the one that needs it: a streamed
        parent's UH plane lives in its store, so the executor has to copy it
        onto the state before the tracker reduces it -- once per cadence,
        not once per parent step.
        """
        root_clock = clocks[model.root.cfg.grid_id]
        ticks = int(root_clock.ticks)
        period_ticks = int(model.schedule.period_ticks)
        return ticks != 0 and ticks % (
            period_ticks * self.cadence_periods) == 0

    def on_period_begin(self, model, clocks, *, period=None,
                        before_rebuild=None):
        """One cadence opportunity; returns the receipt row or ``None``.

        ``before_rebuild`` is the executor's seam for dropping references
        it holds to the outgoing child (health validators) so the
        host-staged release actually frees; it receives the grid_id.
        """
        root_clock = clocks[model.root.cfg.grid_id]
        ticks = int(root_clock.ticks)
        period_ticks = int(model.schedule.period_ticks)
        if ticks == 0 or ticks % (
                period_ticks * self.cadence_periods) != 0:
            return None
        elapsed = float(root_clock.elapsed_seconds)
        grid_id = int(self.config.grid_id)
        node = model.node(grid_id)
        if not bool(getattr(node, "_started", True)):
            return self._record(model, {
                "event": "held", "elapsed_seconds": elapsed,
                "reason": f"d{grid_id:02d} has not started; a domain that "
                          "does not exist yet cannot move"})
        # The tracker seam, verbatim: parent state, footprint, model time.
        desired = self.provider(node.parent.state, node.cfg, elapsed)
        # This consumer's window is "max since I last looked", so it is
        # zeroed HERE -- immediately after the look, before any branch on
        # what the look returned, and on HELD evaluations exactly as on
        # accepted ones (Drew's ruling, 2026-08-07).  Resetting only when
        # a move is accepted would let the window grow across holds and
        # cooldowns, which makes the next move more likely just because
        # more time passed: cadence dependence coming back in through the
        # reset rule.  Tolerant of a provider that reads no UH at all (a
        # scripted move list) and of a state with no window allocated.
        follow_slot = getattr(self.provider, "uh_slot", UH_FOLLOW_WINDOW_SLOT)
        reset_tracker_window(node.parent.state, follow_slot)
        if desired is None:
            return self._record(model, {
                "event": "held", "elapsed_seconds": elapsed,
                "reason": "follow source returned None (hold)"})
        di, dj = (int(desired[0]), int(desired[1]))
        if (di, dj) == (0, 0):
            return self._record(model, {
                "event": "held", "elapsed_seconds": elapsed,
                "reason": "follow source requested the null shift"})
        executed_i, executed_j, clamps = self._clamped_shift(node, (di, dj))
        if (executed_i, executed_j) == (0, 0):
            return self._record(model, {
                "event": "held", "elapsed_seconds": elapsed,
                "requested_shift_parent_cells": [di, dj],
                "clamped_by": clamps,
                "reason": "the requested shift clamps to the null move at "
                          "the parent's admissible band"})
        if self._segment is None:
            self._segment = base_segment(node.cfg)
        # The preparer's pre-move seam, duck-typed and optional: a
        # real-data preparer snapshots the outgoing child's driver-held
        # land-surface state (and the statics it will hold the rebuilt
        # footprint's overlap against) while the outgoing child is still
        # whole -- on_child_built itself fires after the host-staged
        # release, when nothing of the outgoing driver remains.
        capture = getattr(self.on_child_built, "capture_outgoing", None)
        if callable(capture):
            capture(node)
        fingerprint_before = model.experiment_fingerprint
        receipt = relocate_child(
            node,
            i_parent_start=int(node.cfg.i_parent_start) + executed_i,
            j_parent_start=int(node.cfg.j_parent_start) + executed_j,
            segment=self._segment, bounds=self.config,
            initializer=self.initializer,
            static_provenance=self.static_provenance,
            on_child_built=self.on_child_built,
            scratch_arena=getattr(model, "_scratch_arena", None),
            dycore_state_workspace=getattr(
                model, "_dycore_state_workspace", None),
            staging=self.staging,
            on_before_release=(
                None if before_rebuild is None
                else (lambda: before_rebuild(grid_id))))
        self._segment = receipt["segment_state"]
        self.moves_executed += 1
        from gpuwm.core.state import refresh_model_time

        refresh_model_time(node.state, node.clock)
        # The preparer's post-move seam: the node now carries the new
        # placement, grid and state, which is what output metadata and
        # the route's prepared-case bookkeeping must be refreshed from.
        after_move = getattr(self.on_child_built, "after_move", None)
        if callable(after_move):
            after_move(node)
        # Accepted-move semantics for providers that want it (a tracker's
        # cooldown should burn on acceptance, not on a clamped-away
        # proposal); the hook is optional and duck-typed.
        notify = getattr(self.provider, "notify_move_executed", None)
        if callable(notify):
            notify(elapsed, (executed_i, executed_j))
        # A restart across a move promises nothing: chain the record into
        # the live fingerprint so later checkpoints are keyed to the move
        # history and refuse a fresh build by name.
        record_sha = receipt["record_sha256"]
        model.experiment_fingerprint = mark_fingerprint_across_move(
            fingerprint_before, record_sha)
        components = getattr(
            model, "_experiment_fingerprint_components", None)
        if components is not None:
            marked = dict(components)
            history = list(marked.get("relocation", {}).get("records", []))
            history.append(record_sha)
            marked["relocation"] = {"records": history}
            model._experiment_fingerprint_components = marked
        transplant = receipt["transplant"]
        plan = receipt["plan"]
        stamped_cells = sum(
            int(item["stamped_cells"])
            for item in transplant["stamped"].values())
        strip_fill_source = getattr(
            self.initializer, "strip_fill_source", STRIP_FILL_SOURCE)
        land_surface = getattr(self.on_child_built, "last_receipt", None)
        return self._record(model, {
            "event": "relocated",
            "elapsed_seconds": elapsed,
            "period": (None if period is None else int(period)),
            "grid_id": grid_id,
            "requested_shift_parent_cells": [di, dj],
            "executed_shift_parent_cells": [executed_i, executed_j],
            "clamped_by": clamps,
            "placement_from": plan["placement_from"],
            "placement_to": plan["placement_to"],
            "overlap_fraction": plan["overlap_fraction"],
            "fill": {
                "overlap_cells": plan["overlap_cells"],
                "spin_up_cells": plan["spin_up_cells"],
                "strip_fill_source": strip_fill_source,
                "static_fields": receipt["child_rebuild"]["static_fields"],
                "initializer": receipt["child_rebuild"]["initializer"],
                "rebuild": receipt["child_rebuild"].get("rebuild"),
                "post_transplant": receipt["child_rebuild"].get(
                    "post_transplant"),
                "fields_transplanted": transplant["stamped_field_count"],
                "fields_skipped": sorted(transplant["skipped"]),
                "stamped_cells_total": stamped_cells,
                # The route preparer's land-surface receipt: donor-fill
                # provenance counts, overlap-statics assertion, driver
                # rebuild timings.  None on routes whose preparer keeps
                # no such inventory (the idealized tree).
                "land_surface": land_surface,
            },
            "staging": receipt["staging"],
            "donor_alignment_pass": bool(receipt["donor_alignment"]["pass"]),
            "parent_bitwise_unchanged": bool(
                receipt["parent_bitwise_unchanged"]),
            "rk_seeds_refreshed": len(receipt["rk_seeds_refreshed"]),
            "segment_id": receipt["segment"]["segment_id"],
            "generation": receipt["segment"]["generation"],
            "record_sha256": record_sha,
            "experiment_fingerprint": {
                "before": fingerprint_before,
                "after": model.experiment_fingerprint,
            },
        })

    def close_receipt(self, model) -> dict:
        """The run-end summary row (unconsumed itinerary is a finding).

        Exactly ONE such row exists in the receipts, however many times the
        executor closes; see :meth:`_record`.
        """
        summary = {
            "event": "summary",
            "moves_executed": int(self.moves_executed),
            "staging": self.staging,
        }
        unconsumed = getattr(self.provider, "unconsumed", None)
        if callable(unconsumed):
            summary["unconsumed_moves"] = list(unconsumed())
        return self._record(model, summary, unique=True)


class RelocationRunnerCollection:
    """Independent relocation runners behind the executor's scalar seam.

    Each child owns a runner/provider/segment/cooldown state.  The collection
    merely orders them by grid id; it never shares tracker state.
    """

    is_collection = True

    def __init__(self, runners=()):
        self.runners = {int(r.config.grid_id): r for r in runners}

    @property
    def target_grid_ids(self):
        return tuple(sorted(self.runners))

    @property
    def config(self):
        # Compatibility for code that only asks whether a named target exists.
        from types import SimpleNamespace
        gid = self.target_grid_ids[0] if self.runners else None
        return SimpleNamespace(grid_id=gid)

    def is_due(self, model, clocks) -> bool:
        return any(r.is_due(model, clocks) for r in self.runners.values()
                   if int(r.config.grid_id) in model.nodes_by_grid_id)

    def on_period_begin(self, model, clocks, *, period=None, before_rebuild=None):
        outcomes = []
        for gid in self.target_grid_ids:
            if gid not in model.nodes_by_grid_id:
                continue
            runner = self.runners[gid]
            row = runner.on_period_begin(
                model, clocks, period=period, before_rebuild=before_rebuild)
            if row is not None:
                outcomes.append(row)
        return None if not outcomes else {"event": "batch", "outcomes": outcomes}

    def close_receipt(self, model):
        return [self.runners[gid].close_receipt(model) for gid in self.target_grid_ids]

    def merge_from(self, other):
        """Add newly-live targets without resetting existing hysteresis."""
        if other is None:
            return self
        incoming = (other.runners if isinstance(other, RelocationRunnerCollection)
                    else {int(other.config.grid_id): other})
        for gid, runner in incoming.items():
            self.runners.setdefault(int(gid), runner)
        return self

    def drop_absent(self, model):
        # Retired runners are discarded; a later re-arm gets a fresh tracker
        # and therefore a fresh episode-local cooldown/window.
        for gid in list(self.runners):
            if gid not in model.nodes_by_grid_id:
                self.runners.pop(gid, None)

    def attach_writers(self, writers):
        for runner in self.runners.values():
            attach = getattr(runner.on_child_built, "attach_writers", None)
            if callable(attach):
                attach(writers)


__all__ = [
    "ManualMoveProvider", "RELOCATION_RUNNER_CONTRACT",
    "RUNNER_STATE_KEYS", "RUNNER_WRITER_KEYS", "RelocationRunner",
    "RelocationRunnerCollection",
    "STRIP_FILL_SOURCE",
]
