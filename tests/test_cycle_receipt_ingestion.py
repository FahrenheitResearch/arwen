"""The receipt must carry the gate's evidence, not ``null``.

CROSS-LANE FIXTURE LAW.  Every existing supervisor test feeds ``_analyse``
a HAND-BUILT report with a nested ``"ingestion"`` key, and they all pass.
The shipped producer -- :func:`gpuwm.cycle.engine.build_replay_parent_engine`
-- returns the FLAT ``verify_ingestion`` block instead, so
``report.get("ingestion")`` was ``None`` on every real run: the gate never
ran inside the supervisor and the receipt said ``"ingestion": null`` about
the most important gate in the system.  That is exactly how the evidence
looked missing to a reader.

So these tests drive the supervisor with the OTHER side's REAL writer
output.  They are behaviourally red on revert: the receipt's ``ingestion``
is ``None`` and the assertion on the three hashes fails.  No ImportError
is doing the work here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from gpuwm.cycle.clock import CycleClock
from gpuwm.cycle.contracts import INGESTION_SCHEMA
from gpuwm.cycle.engine import build_replay_parent_engine
from gpuwm.cycle.ledger import CycleLedger
from gpuwm.cycle.supervisor import CycleSupervisor

NY = NX = 12
PARENT_DT = 120.0
CYCLE_SECONDS = 960.0
N_CYCLES = 3


def _frames(n: int):
    """Recorded-shaped replay frames, the shape the replay engine reads."""
    out = []
    yy, xx = np.mgrid[0:NY, 0:NX]
    for k in range(n + 1):
        blob = np.exp(-(((yy - (5 + k)) ** 2 + (xx - (4 + k)) ** 2)
                        / (2 * 3.0 ** 2)))
        rho = 1.10 - 0.02 * blob
        theta = 300.0 + 12.0 * blob
        prognostic = {
            "rho": rho.astype(np.float64),
            "rho_theta": (rho * theta).astype(np.float64),
            "rho_u": (rho * (8.0 + 14.0 * blob)).astype(np.float64),
            "rho_w": (rho * 3.5 * blob).astype(np.float64),
            "scalars": (rho * 0.012 * blob).astype(np.float64),
            "time_seconds": np.asarray(k * CYCLE_SECONDS, dtype=np.float64),
        }
        derived = {
            "exner": np.power(
                np.maximum(prognostic["rho_theta"], 1e-12)
                * (287.0 / 100000.0), 287.0 / (1004.5 - 287.0)),
            "pressure_perturbation": (100.0 * blob).astype(np.float64),
        }
        out.append({"prognostic": prognostic, "derived": derived,
                    "reflectivity": (14.0 + 52.0 * blob).astype(np.float64)})
    return out


def _run_cycle(root: Path, *, applied_cycle: int = 2):
    """The shipped wiring, verbatim from ``tools/cycle_demo_b.py``."""
    clock = CycleClock.build(
        epoch_anchor=datetime(2026, 8, 14, 2, 0, tzinfo=timezone.utc),
        parent_dt_seconds=PARENT_DT, cycle_seconds=CYCLE_SECONDS,
        n_cycles=N_CYCLES)

    du = np.zeros((NY, NX), dtype=np.float64)
    du[4:8, 4:8] = 1.5

    def increment_for(cycle_index, prognostic):
        if int(cycle_index) != applied_cycle:
            return {"rho_u": np.zeros_like(prognostic["rho_u"])}
        return {"rho_u": np.asarray(prognostic["rho"]) * du}

    ledger = CycleLedger(root / "cycle_ledger.jsonl")
    advance_parent = build_replay_parent_engine(
        root=root, clock=clock, history_frames=_frames(N_CYCLES),
        mesh_id="test-mesh", increment_for=increment_for, banner=False)

    # THE PRODUCER SHAPE.  This is what tools/cycle_demo_b.py does, and
    # what every real run does: hand the supervisor the flat block.
    def analyse(cycle_index, parent_record):
        return parent_record.get("ingestion")

    supervisor = CycleSupervisor(
        clock=clock, ledger=ledger, root=root,
        advance_parent=advance_parent, analyse=analyse,
        max_forecast_only_cycles=N_CYCLES + 1)
    result = supervisor.run(resume=False)
    receipts = [json.loads((root / f"cycle_{index:03d}" / "RECEIPT.json")
                           .read_text(encoding="utf-8"))
                for index in range(1, N_CYCLES + 1)]
    return result, receipts


def test_the_receipt_carries_the_gates_evidence_not_null(tmp_path):
    """A cycle that assimilated must say so in its OWN receipt.

    The gate compares the state before the increment against the state
    after it, and the second hash only exists once the next parent leg has
    stepped -- so the increment applied at cycle 2 is evidenced in cycle
    3's receipt.  That offset is real and the receipt now names it.
    """
    _result, receipts = _run_cycle(tmp_path / "run", applied_cycle=2)

    applied = receipts[2]
    assert applied["ingestion"] is not None, (
        "the receipt says null about the three-hash gate; the evidence "
        "exists in the anchor but the receipt is what a reader opens")
    block = applied["ingestion"]
    assert block["schema"] == INGESTION_SCHEMA
    # The three hashes, by name.  This is the whole point of the gate.
    for key in ("background_sha256", "increment_sha256", "analysis_sha256"):
        assert isinstance(block.get(key), str) and block[key], key
    assert block["state"] == "APPLIED"
    assert block["increment_nonzero_cells"] > 0
    # A/B: the applied arm MOVED the state.  An exact-zero delta here
    # would mean the experiment never ran.
    assert block["background_sha256"] != block["analysis_sha256"]


def test_the_null_arm_receipt_is_evidence_too(tmp_path):
    """Cycle 1 is the null arm: zero cells, hash unmoved, and NOT null."""
    _result, receipts = _run_cycle(tmp_path / "run", applied_cycle=2)

    null_arm = receipts[1]
    assert null_arm["ingestion"] is not None, (
        "the null arm is evidence: a receipt that omits it cannot show "
        "both arms fired")
    block = null_arm["ingestion"]
    assert block["state"] == "NULL_ARM"
    assert block["increment_nonzero_cells"] == 0
    assert block["background_sha256"] == block["analysis_sha256"]


def test_both_arms_fired_and_the_delta_is_not_exactly_zero(tmp_path):
    """The two arms must differ; identical arms mean nothing was tested."""
    _result, receipts = _run_cycle(tmp_path / "run", applied_cycle=2)

    states = [None if r["ingestion"] is None else r["ingestion"]["state"]
              for r in receipts]
    assert states == [None, "NULL_ARM", "APPLIED"], states
    null_cells = receipts[1]["ingestion"]["increment_nonzero_cells"]
    applied_cells = receipts[2]["ingestion"]["increment_nonzero_cells"]
    assert applied_cells - null_cells > 0, (
        "an exact-zero delta between the arms means the treatment never ran")


def test_a_wrapped_report_still_works(tmp_path):
    """The hand-built nested shape the existing tests use must not break."""
    clock = CycleClock.build(
        epoch_anchor=datetime(2026, 8, 14, 2, 0, tzinfo=timezone.utc),
        parent_dt_seconds=PARENT_DT, cycle_seconds=CYCLE_SECONDS, n_cycles=1)
    root = tmp_path / "wrapped"
    ledger = CycleLedger(root / "cycle_ledger.jsonl")
    advance_parent = build_replay_parent_engine(
        root=root, clock=clock, history_frames=_frames(1),
        mesh_id="test-mesh", increment_for=None, banner=False)

    seen = {}

    def analyse(cycle_index, parent_record):
        block = parent_record.get("ingestion")
        seen["block"] = block
        # The DOCUMENTED shape: the block nested under "ingestion".
        return {"state": "APPLIED", "ingestion": block} if block else None

    supervisor = CycleSupervisor(
        clock=clock, ledger=ledger, root=root,
        advance_parent=advance_parent, analyse=analyse,
        max_forecast_only_cycles=3)
    supervisor.run(resume=False)
    receipt = json.loads((root / "cycle_001" / "RECEIPT.json")
                         .read_text(encoding="utf-8"))
    # Cycle 1 has no previous anchor, so there is genuinely nothing to
    # ingest; the contract is that this stays None rather than inventing.
    assert receipt["ingestion"] is None or (
        receipt["ingestion"]["schema"] == INGESTION_SCHEMA)


def test_a_report_that_is_neither_shape_is_refused_not_silently_dropped(
        tmp_path):
    """A garbage report must not quietly become ``ingestion: null``.

    This is the regression that let the defect ship: an unrecognised shape
    returned ``None`` and the run looked healthy.
    """
    from gpuwm.cycle.contracts import CycleRefusal

    clock = CycleClock.build(
        epoch_anchor=datetime(2026, 8, 14, 2, 0, tzinfo=timezone.utc),
        parent_dt_seconds=PARENT_DT, cycle_seconds=CYCLE_SECONDS, n_cycles=1)
    root = tmp_path / "garbage"
    ledger = CycleLedger(root / "cycle_ledger.jsonl")
    advance_parent = build_replay_parent_engine(
        root=root, clock=clock, history_frames=_frames(1),
        mesh_id="test-mesh", increment_for=None, banner=False)

    def analyse(cycle_index, parent_record):
        return {"looks": "like a report", "but": "carries no gate evidence"}

    supervisor = CycleSupervisor(
        clock=clock, ledger=ledger, root=root,
        advance_parent=advance_parent, analyse=analyse,
        max_forecast_only_cycles=3)
    with pytest.raises(CycleRefusal) as excinfo:
        supervisor.run(resume=False)
    assert "analysis report" in str(excinfo.value).lower()


def test_a_null_ingestion_still_says_where_the_evidence_lives(tmp_path):
    """``ingestion: null`` must never be the whole story.

    Cycle 1 genuinely has no increment before it, so its ``ingestion`` is
    legitimately ``None`` -- but a bare ``null`` reads identically to a
    cycle that silently failed to assimilate.  The receipt has to say
    which of those it is.
    """
    _result, receipts = _run_cycle(tmp_path / "run", applied_cycle=2)

    first = receipts[0]
    assert first["ingestion"] is None
    evidence = first["ingestion_evidence"]
    assert evidence is not None, (
        "a receipt that says null about the gate and nothing else is the "
        "exact trap this field exists to close")
    assert evidence["state"] == "NONE_YET"
    assert evidence["not_a_missing_gate"] is True
    assert "anchors" in evidence["where"]
    assert evidence["why"]


def test_every_receipt_points_at_its_evidence(tmp_path):
    """Every cycle, armed or not, carries the pointer."""
    _result, receipts = _run_cycle(tmp_path / "run", applied_cycle=2)

    for index, receipt in enumerate(receipts, start=1):
        evidence = receipt["ingestion_evidence"]
        assert evidence is not None, index
        assert evidence["where"], index
    # The two that DID carry a block name the cycle whose increment they
    # describe, which is what makes the offset auditable.
    assert receipts[1]["ingestion_evidence"][
        "describes_increment_applied_at_cycle"] == 1
    assert receipts[2]["ingestion_evidence"][
        "describes_increment_applied_at_cycle"] == 2
