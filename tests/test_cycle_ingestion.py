"""The three-hash ingestion gate, exercised in both directions.

"the analysis was applied" and "the analysis was silently dropped" are
indistinguishable from a forecast that ran to completion.  These tests
are the only thing that tells them apart, so both arms are here: a
nonzero increment that left the state unchanged, and a zero increment
that changed it anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.cycle.contracts import CycleRefusal, INGESTION_SCHEMA
from gpuwm.cycle.ingestion import increment_summary, verify_ingestion


def _increment(*, nonzero: bool) -> dict[str, np.ndarray]:
    zero = np.zeros((3, 8), dtype=np.float64)
    if not nonzero:
        return {"u": zero.copy(), "v": zero.copy(), "qr": zero.copy()}
    u = zero.copy()
    u[1, 2] = 1.5
    u[2, 5] = -0.5
    v = zero.copy()
    v[0, 0] = 2.0
    qr = zero.copy()
    qr[2, 7] = 3.0e-4
    return {"u": u, "v": v, "qr": qr}


def test_increment_summary_counts_cells_and_l2():
    summary = increment_summary(_increment(nonzero=True))
    assert summary["schema"] == INGESTION_SCHEMA
    assert summary["increment_nonzero_cells"] == 4
    assert summary["fields"] == ["qr", "u", "v"]
    assert summary["increment_l2"]["v"] == pytest.approx(2.0)
    assert len(summary["increment_sha256"]) == 64


def test_dropped_analysis_is_refused():
    """RED-ON-REVERT ANCHOR: delete the drop branch and this goes green."""
    shared = "4f2c" + "0" * 60
    with pytest.raises(CycleRefusal) as excinfo:
        verify_ingestion(background_sha256=shared,
                         increment=_increment(nonzero=True),
                         analysis_sha256=shared,
                         label="cycle=3 valid=2026-08-14T02:00:00Z")
    message = str(excinfo.value)
    assert "analysis was not ingested" in message
    assert "increment_nonzero_cells=4" in message
    assert "'qr'" in message and "'u'" in message and "'v'" in message
    assert shared in message
    observed = excinfo.value.observed
    assert observed["increment_nonzero_cells"] == 4
    assert observed["shared_sha256"] == shared
    assert observed["fields"] == ["qr", "u", "v"]


def test_null_arm_must_be_bit_stable():
    background = "aa" * 32
    with pytest.raises(CycleRefusal) as excinfo:
        verify_ingestion(background_sha256=background,
                         increment=_increment(nonzero=False),
                         analysis_sha256="bb" * 32,
                         label="null-arm cycle=0")
    assert "null-increment arm is not bit-stable" in str(excinfo.value)

    block = verify_ingestion(background_sha256=background,
                             increment=_increment(nonzero=False),
                             analysis_sha256=background,
                             label="null-arm cycle=0")
    assert block["state"] == "NULL_ARM"
    assert block["increment_nonzero_cells"] == 0


def test_applied_arm_returns_full_block():
    block = verify_ingestion(background_sha256="11" * 32,
                             increment=_increment(nonzero=True),
                             analysis_sha256="22" * 32,
                             label="applied cycle=1")
    assert set(block) >= {"background_sha256", "increment_sha256",
                          "increment_nonzero_cells", "increment_l2",
                          "analysis_sha256", "state"}
    assert block["state"] == "APPLIED"
    assert block["background_sha256"] == "11" * 32
    assert block["analysis_sha256"] == "22" * 32
    assert block["increment_nonzero_cells"] == 4
