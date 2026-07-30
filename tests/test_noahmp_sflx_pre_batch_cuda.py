"""The SFLX-prefix batch, and the slot orders it depends on.

``tests/test_noahmp_leaf_batches_cuda.py`` already holds
:func:`gpuwm.core.noahmp_sflx_pre_gpu.evaluate_sflx_pre_calls` to the CPython
``sflx_pre`` bit for bit on the physical calls the four whole-column WRF
fixtures pause at, and already shows that comparison rejecting a misrouted
column.  This file adds the two falsifications that are specific to *this*
packer, because a misroute and a slot transposition are different defects and
only one of them is covered by swapping two answers:

* ATM is reached through a **flat 11-slot input row and a flat 17-slot output
  row**.  Nothing in either row is named on the device; a value in the wrong
  slot is silently a different physical quantity.  Both orders are held here by
  transposing two entries of the module's own tables and requiring the answer
  to move.

No arithmetic is gated here.  ``noahmp_leaf_atm``, ``k_sflx_marshal``,
``noahmp_phenology`` and ``noahmp_precip_heat`` are each already bitwise
against their own unmodified-WRF oracle CSV in
``tests/test_noahmp_vegprecip_cuda.py``, ``tests/test_noahmp_sflx_cuda.py`` and
the leaves suite.  The composition adds packing and nothing else, so packing is
what this file attacks.
"""

from __future__ import annotations

import copy

import pytest

from conftest import requires_gpu

import test_noahmp_sflx as T

from test_noahmp_leaf_batches_cuda import COLLECTED, _numbers


def _device_answers(calls):
    from gpuwm.core.noahmp_sflx_pre_gpu import evaluate_sflx_pre_calls

    return evaluate_sflx_pre_calls(copy.deepcopy(calls))


def _differs(got, want):
    """Indices where any compared word of the two answers differs."""
    return [k for k, (g, w) in enumerate(zip(got, want))
            if _numbers(g) != _numbers(w)]


@requires_gpu
def test_the_prefix_batch_is_reached_by_every_fixture_column():
    calls, want, _after = COLLECTED["sflx_pre"]
    assert len(calls) == len(T.CASES), len(calls)
    got = _device_answers(calls)
    assert not _differs(got, want)
    # The comparison has to be looking at a substantial answer, not at two
    # scalars that happen to agree: PreEnergy carries the whole ENERGY call.
    assert len(_numbers(got[0])) >= 120, len(_numbers(got[0]))


@requires_gpu
@pytest.mark.parametrize("table,first,second", [
    ("_ATM_IN", "sfcprs", "sfctmp"),
    ("_ATM_OUT", "rain", "snow"),
])
def test_a_transposed_atm_slot_is_rejected(table, first, second, monkeypatch):
    """Transposing two flat ATM slots must move the answer.

    ``sfcprs``/``sfctmp`` are adjacent input slots carrying a pressure and a
    temperature; ``rain``/``snow`` are adjacent output slots that reach
    different PRECIP_HEAT terms.  Either transposition is the defect a flat row
    invites, and until it is observed being rejected the bitwise comparison
    above is a gate that has never been seen to fail.
    """
    import gpuwm.core.noahmp_sflx_pre_gpu as batch

    calls, want, _after = COLLECTED["sflx_pre"]
    order = list(getattr(batch, table))
    i, j = order.index(first), order.index(second)
    order[i], order[j] = order[j], order[i]
    monkeypatch.setattr(batch, table, tuple(order))

    moved = _differs(_device_answers(calls), want)
    assert moved, (
        f"transposing {first} and {second} in {table} left every compared "
        "word unchanged; the prefix gate cannot see a slot-order defect")


@requires_gpu
def test_the_batch_refuses_an_inadmissible_column():
    """``sflx_pre``'s own guards must survive the conversion.

    The dead irrigation and crop regions are dead *because* CROPTYPE and the
    four irrigation amounts are zero.  A batch that dropped those checks would
    run the admitted arithmetic on a column the port has never validated and
    report nothing, so each guard is driven here.
    """
    from gpuwm.core.noahmp_sflx_pre_gpu import evaluate_sflx_pre_calls

    calls, _want, _after = COLLECTED["sflx_pre"]

    crop = copy.deepcopy(calls)
    crop[0][1]["croptype"] = 1
    with pytest.raises(ValueError, match="croptype must be 0"):
        evaluate_sflx_pre_calls(crop)

    for name in ("irrfra", "iramtsi", "iramtmi", "iramtfi"):
        wet = copy.deepcopy(calls)
        wet[0][1][name] = 0.25
        with pytest.raises(ValueError, match=f"{name} must be 0.0"):
            evaluate_sflx_pre_calls(wet)


@requires_gpu
def test_an_empty_batch_is_an_empty_answer():
    from gpuwm.core.noahmp_sflx_pre_gpu import evaluate_sflx_pre_calls

    assert evaluate_sflx_pre_calls([]) == []
