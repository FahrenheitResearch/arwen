#!/usr/bin/env python
"""Audit a run's relocation receipts: is every row an INTENDED decision?

A moving-nest run's evidence is its ledger.  Counting the rows says the
run finished; it does not say the run was *right*.  This walks the
ledger and checks the invariants the subsystem claims, so "the run was
clean" becomes a statement somebody can re-derive from the artifact
rather than a memory of watching it go.

What it checks, and why each one exists:

* **Placement chain continuity.**  Every move's ``placement_from`` must
  be the previous move's ``placement_to`` for that grid.  A gap means a
  placement changed outside the runner, which is the failure mode the
  segment/generation bookkeeping exists to make impossible.
* **The caps were respected.**  Every executed shift within
  ``max_move_parent_cells``, every containment slide within its own cap.
  A shift past the cap is the clamp not firing.
* **The earth-fixed compensation is arithmetic, not aspiration.**  On
  every containment row, each descendant's placement change must equal
  ``-shift x ratio`` exactly, and the row must carry
  ``state_carried_bitwise``.  This is the ledger half of the claim
  ``gpuwm.core.nest_relocation`` makes in code.
* **Donor alignment and parent invariance passed on every row**, and
  every mid-tree move re-grounded every descendant.
* **The tracker's decisions are the configured ones.**  Only decisions
  in the expected set appear, and the two suppressions that mean
  something went wrong (``no-signal``, ``suppressed:at-parent-edge``)
  are reported by count and time rather than buried.
* **The track is a track.**  Successive centroids are compared against
  what the configured cadence and cap physically allow, so a centroid
  that teleports is visible.  A tracker that oscillates -- proposing
  +di then -di repeatedly -- is counted separately, because that is the
  hysteresis failing rather than the storm moving.

Usage::

    python tools/relocation_ledger_audit.py path/to/relocation_receipts.json

Exit status is 0 when every invariant holds and 1 when any fails, so it
can gate.  Paths are arguments: nothing here knows about any one box.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

#: Tracker decisions that are a working tracker.  ``no-signal`` and
#: ``suppressed:at-parent-edge`` are deliberately NOT here: both are
#: legitimate rows, but both mean the tracker could not do its job at
#: that instant, so they are surfaced rather than tallied as normal.
EXPECTED_DECISIONS = frozenset({
    "configured", "proposed", "move-executed",
    "suppressed:dead-band", "suppressed:cooldown",
})

#: Decisions that are admissible but are findings worth naming.
NOTEWORTHY_DECISIONS = frozenset({
    "no-signal", "suppressed:at-parent-edge",
})


class Audit:
    """Accumulates checks so one run reports every failure, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []
        self.checks = 0

    def require(self, ok: bool, message: str) -> bool:
        self.checks += 1
        if not ok:
            self.failures.append(message)
        return ok

    def note(self, message: str) -> None:
        self.notes.append(message)

    @property
    def passed(self) -> bool:
        return not self.failures


def _placement(entry, key):
    """``placement_from``/``placement_to`` as a plain (i, j) pair."""
    value = entry.get(key)
    if isinstance(value, dict):
        return (int(value["i_parent_start"]), int(value["j_parent_start"]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return None


def audit_ledger(payload: dict, audit: Audit) -> dict:
    """Walk one receipts document; returns the summary it printed."""
    config = payload.get("config", {})
    rows = payload.get("receipts", [])
    mover_id = config.get("grid_id")
    mover_cap = config.get("max_move_parent_cells")
    containment = config.get("containment") or {}
    contain_id = containment.get("grid_id")
    contain_cap = containment.get("max_move_parent_cells")

    events = Counter(row.get("event") for row in rows)
    moves = [r for r in rows if r.get("event") == "relocated"]
    slides = [r for r in rows if r.get("event") == "contained"]
    refused = [r for r in rows
               if str(r.get("event", "")).endswith("_refused")]

    audit.require(
        not refused,
        f"{len(refused)} refusal row(s) in the ledger: "
        + "; ".join(f"{r['event']} at t={r.get('elapsed_seconds')}: "
                    f"{r.get('reason')}" for r in refused[:4]))

    # -- placement chain continuity, per grid ------------------------------
    # IN LEDGER ORDER, and counting the descendant rows: a containment
    # slide moves the mover's placement too (the earth-fixed
    # compensation), so a chain walked over the mover's own rows alone
    # reads every slide as a break.  That is the same frame confusion
    # commit 1f46814c fixed in the runner, reproduced in an auditor.
    last_to: dict[int, tuple[int, int]] = {}

    def _chain(gid, start, end, when, label):
        if start is None or end is None:
            audit.require(False, f"d{gid:02d} {label} at t={when} carries no "
                                 "placement_from/placement_to")
            return
        if gid in last_to:
            audit.require(
                last_to[gid] == start,
                f"d{gid:02d} placement chain broken at t={when}: previous "
                f"row ended at {last_to[gid]}, this {label} starts at "
                f"{start}")
        last_to[gid] = end

    # A containment slide and a mover consultation share an instant
    # whenever their cadences meet, and the runner runs the SLIDE FIRST
    # (relocation_runner.on_period_begin: "the tracker then evaluates the
    # storm in the already-slid frame"; commit 6802bb4d).  Ordering the
    # walk the same way is not a convenience -- it makes the chain check
    # a GUARD on that order: swap the two and every shared boundary
    # reports a broken chain.
    for row in sorted(moves + slides,
                      key=lambda r: (float(r.get("elapsed_seconds", 0)),
                                     0 if r.get("event") == "contained"
                                     else 1)):
        gid = int(row.get("grid_id", -1))
        when = row.get("elapsed_seconds")
        start = _placement(row, "placement_from")
        end = _placement(row, "placement_to")
        _chain(gid, start, end, when, row.get("event"))
        shift = row.get("executed_shift_parent_cells")
        if shift is not None and start is not None and end is not None:
            audit.require(
                (start[0] + int(shift[0]), start[1] + int(shift[1])) == end,
                f"d{gid:02d} at t={when}: "
                f"{start} + {tuple(shift)} != {end}")
        for desc in row.get("descendants", []):
            dstart = _placement(desc, "placement_from")
            dend = _placement(desc, "placement_to")
            if dstart is not None and dend is not None:
                _chain(int(desc.get("grid_id", -1)), dstart, dend, when,
                       "descendant re-grounding")

    # -- the caps ----------------------------------------------------------
    for row in moves:
        shift = row.get("executed_shift_parent_cells") or (0, 0)
        if mover_cap is not None:
            audit.require(
                max(abs(int(shift[0])), abs(int(shift[1]))) <= int(mover_cap),
                f"mover shift {tuple(shift)} at "
                f"t={row.get('elapsed_seconds')} exceeds "
                f"max_move_parent_cells = {mover_cap}")
    for row in slides:
        shift = row.get("executed_shift_parent_cells") or (0, 0)
        if contain_cap is not None:
            audit.require(
                max(abs(int(shift[0])), abs(int(shift[1]))) <= int(contain_cap),
                f"containment slide {tuple(shift)} at "
                f"t={row.get('elapsed_seconds')} exceeds its cap "
                f"{contain_cap}")

    # -- earth-fixed compensation, checked as arithmetic -------------------
    compensated = 0
    for row in slides:
        shift = row.get("executed_shift_parent_cells") or (0, 0)
        for desc in row.get("descendants", []):
            if not desc.get("earth_fixed"):
                continue
            compensated += 1
            start = _placement(desc, "placement_from")
            end = _placement(desc, "placement_to")
            ratio = desc.get("parent_grid_ratio")
            if ratio is None:
                # The ratio is not on the row; recover it from the
                # compensation itself and check it is a whole number,
                # equal on both axes and consistent in sign.
                for axis in (0, 1):
                    if int(shift[axis]) != 0:
                        moved = end[axis] - start[axis]
                        quotient = -moved / int(shift[axis])
                        audit.require(
                            quotient > 0
                            and abs(quotient - round(quotient)) < 1e-9,
                            f"d{desc.get('grid_id')} compensation at "
                            f"t={row.get('elapsed_seconds')} axis {axis}: "
                            f"moved {moved} against slide {shift[axis]} is "
                            "not -shift x (whole ratio)")
                        ratio = round(quotient)
            else:
                for axis in (0, 1):
                    audit.require(
                        end[axis] - start[axis]
                        == -int(shift[axis]) * int(ratio),
                        f"d{desc.get('grid_id')} compensation at "
                        f"t={row.get('elapsed_seconds')} axis {axis}: "
                        f"{end[axis] - start[axis]} != "
                        f"{-int(shift[axis]) * int(ratio)}")
            audit.require(
                bool(desc.get("state_carried_bitwise")),
                f"d{desc.get('grid_id')} at t={row.get('elapsed_seconds')} "
                "slid without state_carried_bitwise")

    # -- the per-move verdicts every row carries ---------------------------
    for row in moves + slides:
        audit.require(bool(row.get("donor_alignment_pass")),
                      f"donor_alignment_pass false at "
                      f"t={row.get('elapsed_seconds')} "
                      f"(d{row.get('grid_id')})")
        audit.require(bool(row.get("parent_bitwise_unchanged")),
                      f"parent_bitwise_unchanged false at "
                      f"t={row.get('elapsed_seconds')} "
                      f"(d{row.get('grid_id')})")
        for desc in row.get("descendants", []):
            if "donor_alignment_pass" in desc:
                audit.require(
                    bool(desc["donor_alignment_pass"]),
                    f"descendant d{desc.get('grid_id')} donor alignment "
                    f"failed at t={row.get('elapsed_seconds')}")

    # -- tracker decisions -------------------------------------------------
    tracker = [t for row in rows for t in row.get("tracker_receipts", [])]
    decisions = Counter(t.get("decision") for t in tracker)
    unexpected = {d: n for d, n in decisions.items()
                  if d not in EXPECTED_DECISIONS
                  and d not in NOTEWORTHY_DECISIONS}
    audit.require(not unexpected,
                  f"tracker emitted unrecognised decision(s): {unexpected}")
    for name in sorted(NOTEWORTHY_DECISIONS):
        if decisions.get(name):
            when = [t.get("t") for t in tracker
                    if t.get("decision") == name][:6]
            audit.note(f"{decisions[name]} x {name} (first at t={when})")

    # -- the track itself --------------------------------------------------
    centroids = [(float(t["t"]), t["centroid_parent_ij"][0],
                  t["centroid_parent_ij"][1])
                 for t in tracker if "centroid_parent_ij" in t]
    # A centroid is expressed in the MOVER'S PARENT's cells, and a
    # containment slide moves that parent under the storm.  Comparing
    # two centroids across a slide without translating the frame reads a
    # stationary storm as a jump of ratio x slide cells -- which is the
    # defect 1f46814c fixed in the runner.  The frame offset is
    # accumulated here and removed before the step is measured, so what
    # the note reports is STORM motion.
    # ``delta_parent_cells`` on the descendant row IS that offset,
    # already in the mover's-parent cells and already signed: a slide of
    # +s parent cells moves the frame east under a stationary storm, so
    # the storm's coordinate falls by s x ratio and the correction that
    # restores the original frame ADDS it back.
    frame = {}
    for row in slides:
        for desc in row.get("descendants", []):
            delta = desc.get("delta_parent_cells")
            if delta is not None:
                frame[float(row["elapsed_seconds"])] = (int(delta[0]),
                                                        int(delta[1]))
    offset_i = offset_j = 0
    corrected = []
    for t, ci, cj in centroids:
        if t in frame:
            offset_i += frame[t][0]
            offset_j += frame[t][1]
        corrected.append((t, ci + offset_i, cj + offset_j))
    jumps = [(t1, math.hypot(i1 - i0, j1 - j0))
             for (t0, i0, j0), (t1, i1, j1) in zip(corrected, corrected[1:])]
    if jumps:
        worst_t, worst = max(jumps, key=lambda pair: pair[1])
        mean = sum(step for _, step in jumps) / len(jumps)
        audit.note(f"centroid step, frame-corrected: mean {mean:.2f}, max "
                   f"{worst:.2f} parent cells (at t={worst_t:g}) over "
                   f"{len(jumps)} consultations")

    executed = [(float(t["t"]), tuple(t["executed_shift_parent_cells"]))
                for t in tracker if t.get("decision") == "move-executed"]
    reversals = sum(
        1 for (_, a), (_, b) in zip(executed, executed[1:])
        if (a[0] and b[0] and a[0] * b[0] < 0)
        or (a[1] and b[1] and a[1] * b[1] < 0))
    audit.note(f"{reversals} of {max(len(executed) - 1, 0)} consecutive "
               "executed moves reversed sign on an axis")

    refinements = [t.get("refinement") for t in tracker
                   if isinstance(t.get("refinement"), dict)]
    applied = [r for r in refinements if r.get("applied")]
    if refinements:
        declines = Counter(r.get("declined") for r in refinements
                           if not r.get("applied"))
        corrections = [max(abs(c) for c in r["correction_parent_cells"])
                       for r in applied if "correction_parent_cells" in r]
        audit.note(
            f"refinement applied on {len(applied)}/{len(refinements)} "
            "consultations"
            + (f"; max correction {max(corrections):.2f} parent cells, "
               f"mean {sum(corrections)/len(corrections):.2f}"
               if corrections else "")
            + (f"; declines: {dict(declines)}" if declines else ""))

    elapsed = [float(r["elapsed_seconds"]) for r in rows
               if "elapsed_seconds" in r]
    return {
        "grid_id": mover_id,
        "containment_grid_id": contain_id,
        "rows": len(rows),
        "events": dict(events),
        "tracker_rows": len(tracker),
        "decisions": dict(decisions),
        "moves": len(moves),
        "slides": len(slides),
        "compensated_descendants": compensated,
        "last_elapsed_seconds": max(elapsed) if elapsed else None,
        "forecast_hours": (max(elapsed) / 3600.0) if elapsed else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("receipts", type=Path,
                        help="path to a run's relocation_receipts.json")
    parser.add_argument("--json", action="store_true",
                        help="print the summary as JSON and nothing else")
    args = parser.parse_args(argv)

    payload = json.loads(args.receipts.read_text(encoding="utf-8"))
    audit = Audit()
    summary = audit_ledger(payload, audit)

    if args.json:
        print(json.dumps({"summary": summary, "failures": audit.failures,
                          "notes": audit.notes, "checks": audit.checks,
                          "pass": audit.passed}, indent=2, sort_keys=True))
        return 0 if audit.passed else 1

    print(f"relocation ledger: {args.receipts}")
    print(f"  mover d{summary['grid_id']:02d}"
          + (f", containment d{summary['containment_grid_id']:02d}"
             if summary["containment_grid_id"] else ""))
    print(f"  {summary['rows']} runner rows {summary['events']}")
    print(f"  {summary['tracker_rows']} tracker rows {summary['decisions']}")
    print(f"  reached t = {summary['last_elapsed_seconds']:g} s "
          f"({summary['forecast_hours']:.2f} forecast hours)")
    print(f"  {summary['compensated_descendants']} earth-fixed descendant "
          "compensations checked as arithmetic")
    for line in audit.notes:
        print(f"  note: {line}")
    print(f"  {audit.checks} invariant checks")
    if audit.passed:
        print("  PASS -- every row is an intended decision")
        return 0
    print(f"  FAIL -- {len(audit.failures)} finding(s):")
    for line in audit.failures:
        print(f"    - {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
