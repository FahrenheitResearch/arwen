"""No [tiles] gate may report success over an empty comparison.

The release line spent a night on this class in its own gates
(``d998bb667`` / ``9b1b99289``): ``gpuwm certify`` exited 0 on a metrics CSV
with a header and no data rows, and ``gpuwm dual-run`` reported "capsules are
identical field for field" over two empty documents.  Every row condition is
a statement about all rows and is true of none.

The gates that arrived with the streamed-run port had the same three shapes:

  * ``tilestream.test_gate.compare`` hashed two empty carrier maps to the
    SHA-256 of nothing, found the two digests equal, and recorded
    ``bitexact`` having compared no carrier;
  * ``tilestream.decomp.assert_slice_faithful`` returned ``ok=not bad``, so
    ``checked == 0`` was a pass;
  * the three ``GATE PASSED`` lines asserted totality and stated no size, so
    the count everybody quotes lived in the operator's scrollback rather
    than in the verdict.

This suite pins the fixes.  It is numpy-only and touches no card, which is
why it is on the stage-1 list rather than the GPU shard: the defect is in
the verdict arithmetic, and the verdict arithmetic runs anywhere.

The floors here are ONE, deliberately, and not the full expected count: a
``--quick`` run or a ``--*-only`` selector is a legitimate partial result and
must stay a result rather than become a refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from tilestream import decomp
from tilestream import test_gate


# --------------------------------------------------------------------------
# the comparison primitive
# --------------------------------------------------------------------------

def test_the_comparison_refuses_to_call_nothing_bit_exact() -> None:
    record = test_gate.compare({}, {}, [], 16)
    assert record["compared_count"] == 0
    assert record["the_comparison_is_not_empty"] is False
    assert record["bitexact"] is False


def test_a_real_comparison_still_passes_and_states_its_size() -> None:
    """The positive control, without which the test above proves nothing.

    A condition that refuses everything is indistinguishable from a
    condition that refuses the empty case, so the same primitive is driven
    with one real carrier and has to come back bit-exact over a size of one.
    """
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    record = test_gate.compare({"u": array}, {"u": array.copy()}, [], 16)
    assert record["compared_count"] == 1
    assert record["the_comparison_is_not_empty"] is True
    assert record["bitexact"] is True


def test_a_real_comparison_that_differs_is_still_caught(monkeypatch) -> None:
    """The other direction: the floor must not have made the gate blind.

    ``spatial_signature`` is stubbed because it needs a real tile plan to
    say WHERE a difference sits, and where is a diagnostic.  What is gated
    here is that one moved bit takes ``bitexact`` to False.
    """
    monkeypatch.setattr(test_gate, "spatial_signature",
                        lambda *a, **k: {"verdict": "stubbed"})
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    moved = array.copy()
    moved[0, 0, 0] = np.nextafter(moved[0, 0, 0], np.float32(1e30))
    record = test_gate.compare({"u": moved}, {"u": array}, [], 16)
    assert record["compared_count"] == 1
    assert record["bitexact"] is False
    assert record["fields"]["u"]["bitexact"] is False


# --------------------------------------------------------------------------
# the reassembly verdict
# --------------------------------------------------------------------------

def test_the_reassembly_verdict_refuses_an_empty_comparison(monkeypatch) -> None:
    monkeypatch.setattr(decomp, "reassemble", lambda *a, **k: {})
    verdict = decomp.assert_slice_faithful([], [], object())
    assert verdict["checked"] == 0
    assert verdict["the_comparison_is_not_empty"] is False
    assert verdict["ok"] is False


def test_the_reassembly_verdict_passes_over_one_carrier(monkeypatch) -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)

    class _Store:
        arrays = {"state/mup": array}
        geography: dict = {}

    monkeypatch.setattr(decomp, "reassemble",
                        lambda *a, **k: {"state/mup": array.copy()})
    verdict = decomp.assert_slice_faithful([], [], _Store())
    assert verdict["checked"] == 1
    assert verdict["the_comparison_is_not_empty"] is True
    assert verdict["ok"] is True


# --------------------------------------------------------------------------
# the verdict states its size
# --------------------------------------------------------------------------

class _Sink:
    def __init__(self) -> None:
        self.text = ""

    def write(self, chunk: str) -> int:
        self.text += chunk
        return len(chunk)

    def flush(self) -> None:
        pass


@pytest.mark.parametrize(
    "lines, passed, failed",
    [
        ([], 0, 0),
        (["  PASS  one\n"], 1, 0),
        (["  PASS  one\n", "  FAIL  two\n", "        detail\n"], 1, 1),
        # A line delivered in fragments must still be counted once, because
        # the gate prints through f-strings that split at arbitrary points.
        (["  PA", "SS  split over two writes\n"], 1, 0),
        # Lines the gate prints that are not verdicts must not be counted.
        (["GATE PASSED -- ...\n", "    PASS nested prose\n"], 0, 0),
    ],
)
def test_the_check_counter_counts_what_the_operator_counts(
        lines, passed, failed) -> None:
    counter = test_gate._CheckCounter(_Sink())
    for line in lines:
        counter.write(line)
    assert (counter.passed, counter.failed) == (passed, failed)


def test_the_check_counter_passes_the_text_through_unchanged() -> None:
    """A counter that ate the output would silently blind the gate's log."""
    sink = _Sink()
    counter = test_gate._CheckCounter(sink)
    counter.write("  PASS  one\n")
    counter.write("plain\n")
    assert sink.text == "  PASS  one\nplain\n"
