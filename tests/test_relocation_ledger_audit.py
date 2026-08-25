"""The ledger auditor must FAIL on the ledgers it exists to catch.

``tools/relocation_ledger_audit.py`` is the evidence instrument for a
moving-nest run: it turns "the run looked clean" into a re-derivable
statement about the artifact.  An auditor that passes everything is
worse than none, so every invariant it claims gets a ledger here that
violates exactly that one invariant and nothing else.

Two of these cases are the auditor's own bugs, kept as tests because
each was a real wrong answer first:

* the placement chain must be walked in LEDGER-TIME order with the
  containment slide before the mover consultation at a shared instant
  (relocation_runner.on_period_begin, commit 6802bb4d).  Walking the
  mover's rows alone reads every slide as a break;
* a centroid lives in the mover's-parent frame, and a slide moves that
  frame.  Comparing across a slide without translating reads a
  stationary storm as a jump of ratio x slide cells -- the same frame
  confusion commit 1f46814c fixed inside the runner.
"""

from __future__ import annotations

import copy

import pytest

from tools.relocation_ledger_audit import Audit, audit_ledger

RATIO = 3


def _slide(t, shift, d02_from, d03_from):
    """One containment row with an exact earth-fixed compensation."""
    d02_to = (d02_from[0] + shift[0], d02_from[1] + shift[1])
    d03_to = (d03_from[0] - shift[0] * RATIO, d03_from[1] - shift[1] * RATIO)
    return {
        "event": "contained", "elapsed_seconds": float(t), "grid_id": 2,
        "executed_shift_parent_cells": list(shift),
        "placement_from": {"i_parent_start": d02_from[0],
                           "j_parent_start": d02_from[1]},
        "placement_to": {"i_parent_start": d02_to[0],
                         "j_parent_start": d02_to[1]},
        "donor_alignment_pass": True, "parent_bitwise_unchanged": True,
        "descendants": [{
            "grid_id": 3, "earth_fixed": True,
            "state_carried_bitwise": True,
            "delta_parent_cells": [shift[0] * RATIO, shift[1] * RATIO],
            "placement_from": list(d03_from), "placement_to": list(d03_to),
        }],
    }


def _move(t, shift, d03_from, centroid):
    d03_to = (d03_from[0] + shift[0], d03_from[1] + shift[1])
    return {
        "event": "relocated", "elapsed_seconds": float(t), "grid_id": 3,
        "executed_shift_parent_cells": list(shift),
        "placement_from": {"i_parent_start": d03_from[0],
                           "j_parent_start": d03_from[1]},
        "placement_to": {"i_parent_start": d03_to[0],
                         "j_parent_start": d03_to[1]},
        "donor_alignment_pass": True, "parent_bitwise_unchanged": True,
        "tracker_receipts": [
            {"decision": "proposed", "t": float(t),
             "centroid_parent_ij": list(centroid),
             "raw_shift_parent_cells": [0.0, 0.0]},
            {"decision": "move-executed", "t": float(t),
             "executed_shift_parent_cells": list(shift)},
        ],
    }


def _clean_ledger():
    """A small ledger that is clean, INCLUDING a slide sharing an instant
    with a move -- the case both auditor bugs lived in."""
    rows = [
        _move(360.0, (1, 0), (94, 94), (120.0, 120.0)),
        # t = 720: the slide runs FIRST, then the mover consults in the
        # already-slid frame.  d03 92 -> 92-6 = 86 ... then +1 from the
        # move that follows at the same instant.
        _slide(720.0, (2, 0), (74, 46), (95, 94)),
        _move(720.0, (1, 0), (89, 94), (114.0, 120.0)),
        _move(1080.0, (0, 1), (90, 94), (114.2, 120.1)),
    ]
    return {"config": {"grid_id": 3, "max_move_parent_cells": 2,
                       "containment": {"grid_id": 2,
                                       "max_move_parent_cells": 2}},
            "receipts": rows}


def _run(payload):
    audit = Audit()
    summary = audit_ledger(payload, audit)
    return audit, summary


def test_clean_ledger_passes_including_a_shared_slide_instant():
    audit, summary = _run(_clean_ledger())
    assert audit.passed, audit.failures
    assert summary["moves"] == 3 and summary["slides"] == 1
    assert summary["compensated_descendants"] == 1
    assert audit.checks > 0


def test_frame_correction_makes_a_stationary_storm_read_stationary():
    """The centroid falls by ratio x slide across the slide because the
    FRAME moved, not the storm.  The note must report ~0, not ~6."""
    audit, _ = _run(_clean_ledger())
    step = next(n for n in audit.notes if n.startswith("centroid step"))
    # 120.0 -> 114.0 raw across a +2 parent-cell slide (6 mover-parent
    # cells); corrected, the storm moved a fraction of a cell.
    assert "max 0." in step, step


def test_broken_placement_chain_is_caught():
    payload = _clean_ledger()
    payload["receipts"][3]["placement_from"]["i_parent_start"] = 99
    audit, _ = _run(payload)
    assert not audit.passed
    assert any("placement chain broken" in f for f in audit.failures)


def test_shift_that_does_not_match_the_placement_delta_is_caught():
    payload = _clean_ledger()
    payload["receipts"][0]["executed_shift_parent_cells"] = [2, 0]
    audit, _ = _run(payload)
    assert not audit.passed
    assert any("!=" in f for f in audit.failures)


def test_shift_past_the_cap_is_caught():
    payload = _clean_ledger()
    payload["receipts"][0]["executed_shift_parent_cells"] = [5, 0]
    payload["receipts"][0]["placement_to"]["i_parent_start"] = 99
    payload["receipts"][2]["placement_from"]["i_parent_start"] = 93
    audit, _ = _run(payload)
    assert not audit.passed
    assert any("max_move_parent_cells" in f for f in audit.failures)


def test_wrong_earth_fixed_compensation_is_caught():
    """-shift x ratio is the whole claim; break it by one cell."""
    payload = _clean_ledger()
    slide = payload["receipts"][1]["descendants"][0]
    slide["placement_to"] = [slide["placement_to"][0] + 1,
                             slide["placement_to"][1]]
    audit, _ = _run(payload)
    assert not audit.passed
    assert any("compensation" in f or "chain broken" in f
               for f in audit.failures)


def test_slide_without_a_bitwise_state_carry_is_caught():
    payload = _clean_ledger()
    payload["receipts"][1]["descendants"][0]["state_carried_bitwise"] = False
    audit, _ = _run(payload)
    assert not audit.passed
    assert any("state_carried_bitwise" in f for f in audit.failures)


@pytest.mark.parametrize("flag", ["donor_alignment_pass",
                                  "parent_bitwise_unchanged"])
def test_a_failed_per_move_verdict_is_caught(flag):
    payload = _clean_ledger()
    payload["receipts"][0][flag] = False
    audit, _ = _run(payload)
    assert not audit.passed
    assert any(flag in f for f in audit.failures)


def test_a_refusal_row_is_caught():
    payload = _clean_ledger()
    payload["receipts"].append({
        "event": "containment_refused", "elapsed_seconds": 1440.0,
        "grid_id": 2, "reason": "no initializer wired"})
    audit, _ = _run(payload)
    assert not audit.passed
    assert any("refusal row" in f for f in audit.failures)


def test_an_unrecognised_tracker_decision_is_caught():
    payload = _clean_ledger()
    payload["receipts"][0]["tracker_receipts"][0]["decision"] = "improvised"
    audit, _ = _run(payload)
    assert not audit.passed
    assert any("unrecognised decision" in f for f in audit.failures)


def test_noteworthy_decisions_are_surfaced_not_failed():
    """no-signal is admissible and must be REPORTED, never swallowed and
    never treated as a failure -- a storm can genuinely be absent."""
    payload = _clean_ledger()
    payload["receipts"][0]["tracker_receipts"].append(
        {"decision": "no-signal", "t": 360.0})
    audit, _ = _run(payload)
    assert audit.passed, audit.failures
    assert any("no-signal" in n for n in audit.notes)


def test_slide_after_move_at_a_shared_instant_breaks_the_chain():
    """The ordering guard: the runner runs the slide FIRST.  A ledger in
    which the mover moved first cannot chain, and that is the point."""
    payload = _clean_ledger()
    rows = payload["receipts"]
    # Same two rows, but the mover consulted in the UN-slid frame: it
    # starts where the previous move left it and the slide then follows.
    rows[1], rows[2] = (
        _move(720.0, (1, 0), (95, 94), (120.1, 120.0)),
        _slide(720.0, (2, 0), (74, 46), (96, 94)),
    )
    rows[3] = _move(1080.0, (0, 1), (90, 94), (114.2, 120.1))
    audit, _ = _run(payload)
    assert not audit.passed
    assert any("placement chain broken" in f for f in audit.failures)


def test_summary_reports_the_run_extent():
    audit, summary = _run(_clean_ledger())
    assert summary["last_elapsed_seconds"] == 1080.0
    assert summary["forecast_hours"] == pytest.approx(0.3)
    assert summary["tracker_rows"] == 6


def test_auditor_reports_every_failure_not_only_the_first():
    payload = _clean_ledger()
    payload["receipts"][0]["donor_alignment_pass"] = False
    payload["receipts"][0]["parent_bitwise_unchanged"] = False
    audit, _ = _run(payload)
    assert len(audit.failures) >= 2


def test_deep_copy_of_a_clean_ledger_is_still_clean():
    """Guards against a check that mutates what it walks."""
    payload = _clean_ledger()
    once, _ = _run(copy.deepcopy(payload))
    twice, _ = _run(payload)
    assert once.passed and twice.passed
    assert once.checks == twice.checks
