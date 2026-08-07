"""Activate spawn-triggered nests at leg boundaries.

The companion to :class:`gpuwm.core.relocation_runner.RelocationRunner`,
and deliberately a DIFFERENT cadence granularity, for a structural
reason worth stating once:

A relocation changes a live child's ``i_parent_start``/``j_parent_start``
and nothing the schedule depends on, so the runner mutates the node in
place inside one :func:`gpuwm.core.model.execute_experiment` call.  A
SPAWN adds a domain to the tree, and
:func:`gpuwm.core.clock.build_schedule` bakes each domain's activation
tick into precomputed per-period op lists
(``active = {gid for gid in step if starts[gid] <= start_ticks}``) --
a trigger-driven domain has no activation tick when that expansion runs,
so it cannot be in the schedule, and no in-place hook can add it.
Activation is therefore leg-boundary schedule surgery, exactly as
``docs/nest-spawn-at-trigger.md`` specifies: the leg before a fire
integrates ``active_experiment(exp)``, the leg after integrates
``active_experiment(exp, {gid: (i, j)})``.

This module is the mechanism for that boundary.  Construction is the
route's job because the route owns the two things neither the mechanism
nor the policy can invent: the ``on_child_built`` preparer that gives a
newborn child its physics/land driver, and the assembler that rebuilds
the next leg's tree.  The runner supplies the assembler's missing half
through :meth:`SpawnRunner.child_initializer`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from gpuwm.core.uh_diag import (UH_SPAWN_WINDOW_SLOT,
                               reset_tracker_window)

SPAWN_RUNNER_CONTRACT = "gpuwm.spawn-runner.v1"

_LOG = logging.getLogger(__name__)


class SpawnRunnerRefusal(RuntimeError):
    """A spawn runner was asked for something it must not invent."""


def _atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SpawnRunner:
    """Consult every dormant nest at each leg boundary; activate the fired.

    ``statics_provider`` is the per-footprint own-grid static builder
    (:func:`gpuwm.ingest.nest_spawn_init.prepare_spawn_statics` bound to
    the route's catalog), or ``None`` for the statics-free branch, where
    the newborn keeps the parent-SINT ground exactly as every idealized
    cold start does.  It is a parameter rather than a config key for the
    same reason ``staging`` is on the relocation runner: it changes what
    the route must have on disk and nothing a config can assert.
    """

    def __init__(self, *, experiment, on_child_built, controller=None,
                 statics_provider=None, receipts_path=None,
                 array_module=None):
        from gpuwm.core.nest_spawn import SpawnController

        if on_child_built is None:
            raise SpawnRunnerRefusal(
                "SpawnRunner requires the route's on_child_built preparer: "
                "the spawn path attaches no physics driver, and a nest born "
                "without one integrates nothing")
        if controller is None:
            controller = SpawnController.from_experiment(experiment)
        if controller is None:
            raise SpawnRunnerRefusal(
                "this experiment declares no dormant ([[domain]] spawn = "
                "{...}) nest; a spawn runner with nothing to watch is a "
                "silent no-op, so this refuses.  Use "
                "SpawnRunner.from_experiment, which returns None instead")
        self.experiment = experiment
        self.controller = controller
        self.on_child_built = on_child_built
        self.statics_provider = statics_provider
        self.array_module = array_module
        self.receipts_path = (None if receipts_path is None
                              else Path(receipts_path))
        #: grid_id -> fired (i_parent_start, j_parent_start)
        self.spawned: dict[int, tuple[int, int]] = {}
        #: grid_id -> the live ChildInitResult carried across legs
        self._child_results: dict[int, object] = {}
        self.receipts: list[dict] = []
        self.spawns_executed = 0

    @classmethod
    def from_experiment(cls, exp, *, on_child_built, statics_provider=None,
                        receipts_path=None,
                        array_module=None) -> "SpawnRunner | None":
        """The route's one-line hookup; ``None`` when nothing is dormant."""
        from gpuwm.core.nest_spawn import SpawnController

        controller = SpawnController.from_experiment(exp)
        if controller is None:
            return None
        return cls(experiment=exp, on_child_built=on_child_built,
                   controller=controller, statics_provider=statics_provider,
                   receipts_path=receipts_path, array_module=array_module)

    # -- the leg views ----------------------------------------------------

    @property
    def active(self):
        """The experiment view the NEXT leg integrates.

        Before any fire this is ``pre_spawn_experiment(exp)`` (dormant
        nests absent); after each fire the fired nests join at their
        trigger-chosen placements.
        """
        from gpuwm.experiment import active_experiment

        return active_experiment(self.experiment, dict(self.spawned))

    @property
    def pending(self) -> tuple[int, ...]:
        return self.controller.pending

    # -- receipts ---------------------------------------------------------

    def _record(self, entry: dict) -> dict:
        entry = {"contract": SPAWN_RUNNER_CONTRACT, **entry}
        rows = self.controller.drain_receipts()
        if rows:
            entry["watch_receipts"] = rows
        self.receipts.append(entry)
        if self.receipts_path is not None:
            _atomic_json(self.receipts_path, {
                "contract": SPAWN_RUNNER_CONTRACT,
                "receipts": self.receipts,
            })
        return entry

    # -- the leg-boundary hook --------------------------------------------

    def refresh_from_model(self, model) -> None:
        """Carry each live spawned child forward into the next leg.

        A relocation between two legs REPLACES the child's state object
        (``relocate_child`` rebuilds it), so the adoption source is the
        model's live node, never the object the spawn receipt captured.
        """
        from gpuwm.ingest.nest_init import ChildInitResult

        for gid in list(self._child_results):
            try:
                node = model.node(gid)
            except Exception:
                continue
            if node is None or not bool(getattr(node, "_started", True)):
                continue
            self._child_results[gid] = ChildInitResult(
                state=node.state, grid=node.grid, coord=None, real=None,
                static_fields=None, horizontal=None, soil=None,
                domain=node.cfg)

    def on_leg_boundary(self, model, *, t=None) -> dict | None:
        """One spawn opportunity; the activation record, or ``None``.

        Returns ``None`` when nothing fired (the tree for the next leg is
        unchanged); otherwise a record whose ``"experiment"`` is the view
        the next leg must assemble and whose ``"child_initializer"`` is
        the adoption seam that attaches the newborn's node/clock/coupler.
        """
        from gpuwm.core.storm_tracking import NestFootprint
        from gpuwm.ingest.nest_spawn_init import spawn_child_from_parent

        self.refresh_from_model(model)
        if not self.controller.pending:
            return None

        if t is None:
            t = float(model.root.clock.elapsed_seconds)
        t = float(t)

        started = [node for node in model.walk_parent_first()
                   if bool(getattr(node, "_started", True))]
        parent_states = {int(node.cfg.grid_id): node.state
                         for node in started}
        # A watch must not read signal that already lives under another
        # nest, so every LIVE child's footprint is excluded; evaluate_all
        # chains the ones fired at this same boundary into the set too.
        active_footprints = tuple(
            NestFootprint.coerce(node.cfg) for node in started
            if node.parent is not None)

        events = self.controller.evaluate_all(
            parent_states, t, active_footprints=active_footprints)
        # Every watch has now looked at every parent it watches, so every
        # spawn window is zeroed -- fired, held or excluded alike (Drew's
        # ruling, 2026-08-07: the window is "max since I last looked").
        # This is the spawn consumer's OWN slot; the relocation runner
        # resets a different one on its own cadence, so neither can blind
        # the other at a boundary they happen to share.
        for parent_state in parent_states.values():
            reset_tracker_window(parent_state, UH_SPAWN_WINDOW_SLOT)
        if not events:
            self._record({"event": "held", "elapsed_seconds": t,
                          "pending": list(self.controller.pending)})
            return None

        # active_experiment FIRST: it is what carries the fired placement
        # onto the DomainConfig (a field trigger's placement is nothing
        # like the declared placeholder) and it re-runs the clearance rule
        # on it.  Materialization then reads that adjudicated config.
        #
        # Computed on a CANDIDATE map and committed only once every nest
        # of this boundary has materialized, so a refusal mid-way leaves
        # self.spawned describing what actually exists.  (The watch itself
        # has already recorded the fire -- the controller owns that state
        # -- so this bounds the damage rather than undoing it.)
        candidate = dict(self.spawned)
        for event in events:
            candidate[int(event.grid_id)] = (
                int(event.i_parent_start), int(event.j_parent_start))
        from gpuwm.experiment import active_experiment

        active = active_experiment(self.experiment, candidate)

        born = []
        for event in events:
            gid = int(event.grid_id)
            child_dc = active.domain(gid)
            parent_node = model.node(int(self.controller.parent_of[gid]))
            statics = (None if self.statics_provider is None
                       else self.statics_provider(child_dc, parent_node))
            receipt = spawn_child_from_parent(
                child_dc, parent_node,
                static_fields=statics,
                blend_width=int(getattr(self.experiment, "blend_width", 5)),
                scratch_arena=getattr(model, "_scratch_arena", None),
                dycore_state_workspace=getattr(
                    model, "_dycore_state_workspace", None),
                array_module=self.array_module,
                on_child_built=self.on_child_built,
                trigger_receipt=event.receipt)
            self._child_results[gid] = receipt["child_result"]
            self.spawned[gid] = candidate[gid]
            self.spawns_executed += 1
            born.append({
                "grid_id": gid,
                "placement": [int(event.i_parent_start),
                              int(event.j_parent_start)],
                "spawn_receipt": {key: value
                                  for key, value in receipt.items()
                                  if key != "child_result"},
            })

        recorded = self._record({
            "event": "spawned",
            "elapsed_seconds": t,
            "grid_ids": [int(e.grid_id) for e in events],
            "born": born,
            "active_grid_ids": [int(dc.grid_id) for dc in active.domains],
            "pending_after": list(self.controller.pending),
        })
        # The RETURNED record carries live objects for the driver; the
        # RECORDED one must stay JSON-safe, because self.receipts is what
        # gets written to receipts_path and what callers serialise into
        # their own receipts.  Copy rather than mutate in place, or an
        # ExperimentConfig and a closure end up inside the run's
        # durable evidence.
        record = dict(recorded)
        record["experiment"] = active
        record["child_initializer"] = self.child_initializer()
        # The live newborns, for a driver that attaches them to an
        # EXISTING tree in place (the real-data leg walk) rather than
        # re-assembling the tree around them.  Keyed by grid_id, the
        # ``segment_state``/``child_result`` precedent again: a live
        # object beside the JSON, never inside it.
        record["child_results"] = {
            int(e.grid_id): self._child_results[int(e.grid_id)]
            for e in events}
        return record

    # -- the leg-boundary rebuild's missing half ---------------------------

    def child_initializer(self, default=None):
        """Adopt every already-spawned child into the next leg's assembly.

        Hand this to the route's assembler as ``child_initializer``: a
        spawned domain is returned as-is (its live state, grid and coord),
        so the assembler builds the node, binds the clock the rebuilt
        schedule minted for it, and attaches the coupler -- the same
        node/clock/coupler attachment every leg boundary performs.  Any
        other domain falls through to ``default``.
        """
        from gpuwm.ingest.nest_init import ChildInitResult, parent_only_init

        if default is None:
            default = parent_only_init
        results = self._child_results

        def initialize(child_dc, parent_node, **kwargs):
            gid = int(child_dc.grid_id)
            if gid not in results:
                return default(child_dc, parent_node, **kwargs)
            carried = results[gid]
            return ChildInitResult(
                state=carried.state, grid=carried.grid,
                coord=getattr(carried, "coord", None), real=None,
                static_fields=None, horizontal=None, soil=None,
                domain=child_dc)

        return initialize

    # -- the closing summary ----------------------------------------------

    def close_receipt(self, model=None) -> dict:
        """Summarize the run, naming every watch that never fired."""
        unfired = []
        for gid in sorted(self.controller.watches):
            watch = self.controller.watches[gid]
            if int(gid) in self.spawned:
                continue
            unfired.append({
                "grid_id": int(gid),
                "closed": bool(getattr(watch, "closed", False)),
                "note": ("the window closed without the trigger firing; the "
                         "reservation was held for the whole run and cost "
                         "zero compute -- that is the contract, not a leak"
                         if getattr(watch, "closed", False) else
                         "the run ended with this watch still open"),
            })
        entry = self._record({
            "event": "closed",
            "spawns_executed": int(self.spawns_executed),
            "spawned": {str(gid): list(pos)
                        for gid, pos in sorted(self.spawned.items())},
            "never_fired": unfired,
        })
        if model is not None and getattr(
                model, "_spawn_receipts", None) is not self.receipts:
            model._spawn_receipts = self.receipts
        return entry


__all__ = ["SPAWN_RUNNER_CONTRACT", "SpawnRunner", "SpawnRunnerRefusal"]
