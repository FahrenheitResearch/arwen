"""The cycling spine's state machine.

The supervisor owns one cycle boundary at a time and nothing else.  It
advances the parent, proves the clock is ARMED, offers the analysis a
chance to land, verifies the analysis was actually eaten, replans the
children, advances them, writes a receipt, and appends the transition.
Every one of those steps is an injected callable, so the whole machine
is testable with no GPU, no model and no NetCDF -- which is the point:
the failure policy is where a cycling run lives or dies, and a policy
you can only exercise by running a forecast is a policy nobody tests.

The deliberate asymmetry, stated once here because it governs
everything below: **a child may die without killing the cycle; the
parent may not.**  That is what makes unattended arbitrary child
spawning safe -- a bad placement or a blown-up 500 m child costs one
slot, not the run.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from gpuwm.cycle.contracts import (CycleRefusal, INGESTION_SCHEMA, TickRatio)
from gpuwm.cycle.ledger import (CycleLedger, transition_receipt,
                                write_receipt_atomically)

#: Child records the supervisor treats as still flying.  Anything else
#: is a slot returning to the pool.
_LIVE_CHILD_STATES = ("LIVE", "PLANNED")


def _increment_fields(ingestion: Mapping) -> list[str]:
    """Which fields the increment touched, from either producer's spelling.

    :func:`gpuwm.cycle.ingestion.verify_ingestion` -- the only producer any
    real run uses -- calls this key ``fields``.  The halt path read
    ``increment_fields``, which is supplied by exactly one hand-built
    supervisor test fixture and by nothing that ships.  The consequence
    was silent and nasty: at an ANALYSIS_NOT_INGESTED halt, the one place
    an operator needs to know WHICH fields reached the anchor but not the
    device, the receipt named an empty list.  Green tests, empty evidence.
    """

    for key in ("fields", "increment_fields"):
        value = ingestion.get(key)
        if value:
            return [str(name) for name in value]
    return []


class CycleSupervisor:
    """Drive the DA/forecast cycle across boundaries, fail-closed."""

    def __init__(self, *, clock, ledger: CycleLedger, root: str | Path,
                 advance_parent,
                 analyse=None,
                 plan_children=None,
                 advance_children=None,
                 max_forecast_only_cycles: int = 3,
                 allow_placement_clamp: bool = False) -> None:
        self.clock = clock
        self.ledger = ledger
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.advance_parent = advance_parent
        self.analyse = analyse
        self.plan_children = plan_children
        self.advance_children = advance_children
        self.max_forecast_only_cycles = int(max_forecast_only_cycles)
        self.allow_placement_clamp = bool(allow_placement_clamp)

    # -- the run ---------------------------------------------------------

    def run(self, *, resume: bool = True) -> dict:
        """Advance every remaining cycle, or refuse naming the reason."""
        state = self.ledger.state()
        if state["halted"]:
            raise CycleRefusal(
                "this cycle root is halted; anchor N-1 is intact, but "
                "resuming past a halt would build a lineage on an analysis "
                "this run already refused",
                root=str(self.root), halt_reason=state["halt_reason"],
                last_completed_cycle=state["last_completed_cycle"])
        start = state["last_completed_cycle"] + 1 if resume else 1
        start = max(start, 1)
        completed: list[int] = []
        receipts: list[dict] = []
        for cycle_index in range(start, self.clock.n_cycles + 1):
            receipts.append(self._run_one(cycle_index))
            completed.append(cycle_index)
        return {
            "root": str(self.root),
            "resumed_from_cycle": start,
            "cycles_completed": completed,
            "receipts": receipts,
            "halted": False,
            "clock": self.clock.to_json(),
        }

    def _run_one(self, cycle_index: int) -> dict:
        self.ledger.append({"event": "cycle-started", "cycle": cycle_index,
                            "valid_time":
                                self.clock.valid_time(cycle_index).isoformat(),
                            "boundary_ticks":
                                self.clock.boundary_ticks(cycle_index)})
        refusals: list[dict] = []

        parent_record = self._advance_parent(cycle_index)
        placements = self._plan(cycle_index, parent_record, refusals)
        child_records = self._advance_children(cycle_index, parent_record,
                                               placements)
        arming = self._assert_armed(cycle_index, parent_record, child_records)
        ingestion = self._analyse(cycle_index, parent_record)
        children = self._settle_children(cycle_index, child_records, refusals)

        receipt = transition_receipt(
            clock=self.clock, cycle_index=cycle_index, arming=arming,
            ingestion=ingestion, placements=placements, children=children,
            refusals=refusals,
            ingestion_evidence=self._ingestion_evidence(
                cycle_index, ingestion))
        write_receipt_atomically(
            self.root / f"cycle_{cycle_index:03d}" / "RECEIPT.json", receipt)
        self.ledger.append({"event": "cycle-completed", "cycle": cycle_index,
                            "armed": arming["armed"],
                            "children_live": sum(
                                1 for child in children
                                if child["state"] == "LIVE"),
                            "refusals": len(refusals)})
        return receipt

    # -- parent ----------------------------------------------------------

    def _advance_parent(self, cycle_index: int) -> dict:
        anchor_in = self.root / f"cycle_{cycle_index - 1:03d}"
        try:
            record = self.advance_parent(cycle_index, anchor_in)
        except Exception as exc:                     # noqa: BLE001 - policy
            # The parent is the lineage.  There is no continuing past it,
            # and a parent that refused is as fatal as a parent that blew
            # up: both leave the cycle with no atmosphere to hand on.
            self.ledger.halt(
                cycle=cycle_index, reason="PARENT_DIVERGED",
                error=str(exc), error_type=type(exc).__name__,
                parent_observed=dict(getattr(exc, "observed", {})))
        if not isinstance(record, Mapping):
            raise CycleRefusal(
                "advance_parent must return the parent's anchor record",
                cycle=cycle_index, got_type=type(record).__name__)
        return dict(record)

    # -- the arming test -------------------------------------------------

    def _assert_armed(self, cycle_index: int, parent_record: Mapping,
                      child_records) -> dict:
        """Prove the clock triple at the boundary, or halt CLOCK_UNARMED.

        Deleting any branch of this method must turn a test red.  That is
        what makes it an arming test rather than a comment.
        """
        expected = self.clock.boundary_ticks(cycle_index)
        parent_ticks = int(parent_record.get("parent_ticks", -1))
        anchor_ticks = int(parent_record.get("anchor_ticks", parent_ticks))
        if parent_ticks != expected:
            self.ledger.halt(cycle=cycle_index, reason="CLOCK_UNARMED",
                             observed_parent_ticks=parent_ticks,
                             expected_ticks=expected, which="parent")
        if anchor_ticks != expected:
            self.ledger.halt(cycle=cycle_index, reason="CLOCK_UNARMED",
                             observed_anchor_ticks=anchor_ticks,
                             observed_parent_ticks=parent_ticks,
                             expected_ticks=expected, which="anchor")
        children = []
        for child in child_records or ():
            if child.get("state") not in _LIVE_CHILD_STATES:
                # A child already retired or diverged is not held to the
                # triple; its slot is going back to the pool.
                continue
            grid_id = child.get("grid_id")
            child_step = int(child.get("child_step_ticks", 0))
            child_ticks = int(child.get("child_ticks", -1))
            ratio = TickRatio(self.clock.parent_step_ticks, child_step)
            scaled = child_ticks * ratio.child_ticks
            if scaled != expected:
                self.ledger.halt(
                    cycle=cycle_index, reason="CLOCK_UNARMED",
                    grid_id=grid_id, which="child",
                    observed_child_ticks=child_ticks,
                    observed_child_ticks_scaled=scaled,
                    observed_parent_ticks=parent_ticks,
                    expected_ticks=expected,
                    child_step_ticks=child_step, ratio=ratio.ratio,
                    drift_seconds=(scaled - expected) / 1000.0)
            children.append({"grid_id": grid_id, "child_ticks": child_ticks,
                             "child_step_ticks": child_step,
                             "ratio": ratio.ratio, "scaled_ticks": scaled})
        return {"armed": True, "cycle_index": cycle_index,
                "expected_ticks": expected, "parent_ticks": parent_ticks,
                "anchor_ticks": anchor_ticks, "children": children}

    # -- analysis --------------------------------------------------------

    def _analyse(self, cycle_index: int, parent_record: Mapping):
        if self.analyse is None:
            # Forecast-only BY CONSTRUCTION is not data starvation: the
            # staleness budget guards a DA cycle that stopped receiving
            # obs, and there is nothing here to starve.
            self.ledger.append({"event": "analysis", "cycle": cycle_index,
                                "state": "SKIPPED_NO_OBS",
                                "forecast_only_by_construction": True,
                                "detail": "no analysis callable is wired"})
            return None
        report = self.analyse(cycle_index, parent_record)
        if report is None:
            return self._skip_no_obs(cycle_index, {})
        report = dict(report)
        state = report.get("state")
        if state == "SKIPPED_NO_OBS":
            return self._skip_no_obs(cycle_index, report)
        if state == "REJECTED":
            # The analysis's own keys go in a nested block: an analysis
            # that reports a "reason" of its own must not collide with
            # the halt reason and lose the halt's own name.
            self.ledger.halt(cycle=cycle_index, reason="ANALYSIS_REJECTED",
                             analysis_report={
                                 key: value for key, value in report.items()
                                 if key != "state"},
                             remedy="an unusable analysis poisons the "
                                    "lineage; anchor N-1 is intact, rerun "
                                    "this cycle once the analysis is sound")
        ingestion = self._ingestion_block(cycle_index, report)
        if ingestion is not None:
            self._check_ingestion(cycle_index, ingestion)
        self.ledger.append({"event": "analysis", "cycle": cycle_index,
                            "state": state or "APPLIED",
                            "ingestion": dict(ingestion) if ingestion else None})
        return ingestion

    def _ingestion_evidence(self, cycle_index: int, ingestion) -> dict:
        """Where this cycle's gate evidence lives -- ALWAYS, even when null.

        ``"ingestion": null`` in a receipt is a trap: it reads as "this
        cycle did not assimilate" when it may equally mean "the evidence
        is somewhere else".  Both happen, and only one is a problem, so
        the receipt says which.

        The offset is real and is not a defect.  The gate compares the
        state BEFORE the increment against the state AFTER it, and the
        second hash is only available once the next parent leg has
        rehydrated and stepped.  So the evidence for the increment applied
        at cycle N surfaces at cycle N+1, and cycle 1 has genuinely
        nothing to report because no increment preceded it.
        """
        anchors = self.root / "anchors"
        if ingestion is not None:
            return {
                "state": str(ingestion.get("state")),
                "where": "this receipt's own 'ingestion' block",
                "describes_increment_applied_at_cycle": int(cycle_index) - 1,
                "also_in_anchor": str(anchors),
            }
        return {
            "state": "NONE_YET",
            "where": str(anchors),
            "why": ("the three-hash gate needs the post-increment state, "
                    "which only exists after the next parent leg has "
                    "stepped; cycle 1 has no increment before it"),
            "describes_increment_applied_at_cycle": None,
            "not_a_missing_gate": True,
        }

    def _ingestion_block(self, cycle_index: int, report: Mapping):
        """The gate's evidence, from either shape an analysis may report.

        Two shapes exist in the tree and BOTH are legitimate callers:

        * the FLAT ``verify_ingestion`` block, which is what
          :func:`gpuwm.cycle.engine.build_replay_parent_engine`,
          ``tools/cycle_mpas_leg.py`` and ``tools/cycle_demo_b.py`` all
          hand back -- every real run takes this path;
        * the block NESTED under ``"ingestion"``, which is what
          :mod:`gpuwm.cycle.ingestion`'s docstring describes and what the
          hand-built supervisor tests use.

        Reading only the nested key made every real run return ``None``
        here: the gate never ran inside the supervisor and the receipt
        carried ``"ingestion": null`` about the most important gate in the
        system, while the real evidence sat in the anchor where no reader
        of the receipt would look.  A receipt that says ``null`` about the
        gate is indistinguishable from a cycle that never assimilated, so
        the ambiguity is resolved here rather than left to the reader.

        A report that is NEITHER shape is refused.  Silently returning
        ``None`` for an unrecognised report is exactly how the defect
        shipped green.
        """
        if report.get("schema") == INGESTION_SCHEMA:
            return dict(report)
        if "ingestion" in report:
            block = report.get("ingestion")
            return dict(block) if block is not None else None
        raise CycleRefusal(
            "analysis report carries no ingestion evidence",
            cycle=int(cycle_index),
            observed_keys=sorted(str(key) for key in report),
            remedy="return the verify_ingestion block itself, or nest it "
                   "under an 'ingestion' key; a report with neither cannot "
                   "prove the analysis was ingested")

    def _skip_no_obs(self, cycle_index: int, report: Mapping):
        record = {"event": "analysis", "cycle": cycle_index,
                  "state": "SKIPPED_NO_OBS"}
        record.update({key: value for key, value in report.items()
                       if key != "state"})
        self.ledger.append(record)
        streak = self.ledger.state()["forecast_only_streak"]
        if streak > self.max_forecast_only_cycles:
            # Forecast-only is legitimate; forecast-only forever is a run
            # that has quietly stopped being a DA cycle.
            self.ledger.halt(
                cycle=cycle_index,
                reason="STALE_ANALYSIS_BUDGET_EXHAUSTED",
                forecast_only_streak=streak,
                max_forecast_only_cycles=self.max_forecast_only_cycles,
                last_analysis_detail=dict(report))
        return None

    def _check_ingestion(self, cycle_index: int, ingestion: Mapping) -> None:
        """Both arms.  An exact-zero delta means the experiment never ran."""
        nonzero = int(ingestion.get("increment_nonzero_cells", 0))
        background = ingestion.get("background_sha256")
        analysis = ingestion.get("analysis_sha256")
        moved = analysis != background
        if nonzero > 0 and not moved:
            self.ledger.halt(
                cycle=cycle_index, reason="ANALYSIS_NOT_INGESTED",
                increment_nonzero_cells=nonzero,
                increment_fields=_increment_fields(ingestion),
                increment_l2=dict(ingestion.get("increment_l2") or {}),
                shared_sha256=background,
                remedy="the increment reached the anchor but not the device "
                       "state; check the resume path applied it before the "
                       "first step")
        if nonzero == 0 and moved:
            self.ledger.halt(
                cycle=cycle_index, reason="ANALYSIS_NOT_INGESTED",
                arm="NULL_ARM", increment_nonzero_cells=nonzero,
                background_sha256=background, analysis_sha256=analysis,
                remedy="rehydration itself perturbed the state; a zero "
                       "increment must be bit-stable through the anchor")

    # -- children --------------------------------------------------------

    def _plan(self, cycle_index: int, parent_record: Mapping,
              refusals: list[dict]) -> list[dict]:
        if self.plan_children is None:
            return []
        planned = []
        for item in self.plan_children(cycle_index, parent_record) or ():
            placement = dict(item)
            if placement.get("clamped") and not self.allow_placement_clamp:
                # A refused placement never silently becomes a clamped
                # one.  Clamping is opt-in and always receipted.
                placement["state"] = "REFUSED"
                placement.setdefault(
                    "reason",
                    "placement required clamping into the parent; pass "
                    "--allow-placement-clamp to accept a clamped nest")
            if placement.get("state") == "REFUSED":
                refusals.append({"kind": "PLACEMENT", "cycle": cycle_index,
                                 "grid_id": placement.get("grid_id"),
                                 "reason": placement.get("reason"),
                                 "requested": {
                                     key: placement.get(key)
                                     for key in ("lat", "lon", "dx_m",
                                                 "nx", "ny")
                                     if key in placement}})
                self.ledger.append({"event": "child", "cycle": cycle_index,
                                    "grid_id": placement.get("grid_id"),
                                    "state": "REFUSED",
                                    "reason": placement.get("reason")})
            planned.append(placement)
        return planned

    def _advance_children(self, cycle_index: int, parent_record: Mapping,
                          placements) -> list[dict]:
        if self.advance_children is None:
            return []
        live = [item for item in placements
                if item.get("state") != "REFUSED"]
        if placements and not live:
            # Every slot was refused: there is nothing to advance, and
            # the cycle carries on with the parent alone.
            return []
        records = self.advance_children(cycle_index, parent_record, live)
        return [dict(item) for item in records or ()]

    def _settle_children(self, cycle_index: int, child_records,
                         refusals: list[dict]) -> list[dict]:
        settled = []
        for record in child_records:
            child = dict(record)
            if child.get("state") == "DIVERGED":
                # One slot, not the run.
                child["state"] = "RETIRED"
                child["retired_because"] = "DIVERGED"
                refusals.append({"kind": "INTEGRATION", "cycle": cycle_index,
                                 "grid_id": child.get("grid_id"),
                                 "field": child.get("field"),
                                 "cell": child.get("cell"),
                                 "value": child.get("value")})
            self.ledger.append({"event": "child", "cycle": cycle_index,
                                "grid_id": child.get("grid_id"),
                                "state": child["state"],
                                "retired_because":
                                    child.get("retired_because")})
            settled.append(child)
        return settled
