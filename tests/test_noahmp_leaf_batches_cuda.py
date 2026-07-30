"""Every device leaf batch, against the host authority, on real paused calls.

The per-leaf CUDA tests in this tree drive each kernel from its own oracle CSV
through a hand-built flat row.  That gates the *arithmetic* and says nothing
about the packing the runtime actually uses, which is a different piece of code
and the one a whole-column conversion adds.  This file closes that gap for all
seven batched leaves at once:

1. run the four unmodified-WRF whole-column fixtures through ``sflx_steps``,
   the generator the runtime uses, and record every physical leaf call it
   pauses at, together with the answer the CPython leaf gives;
2. hand those same call tuples to the device batch; and
3. require every returned number to be bit identical.

A packer that reads the wrong keyword, writes the wrong slot, or transposes two
layers fails here and nowhere else.  ``test_the_batch_comparison_can_fail``
shows the comparison rejecting exactly those defects before any of it counts as
evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

import test_noahmp_sflx as T

from gpuwm.core.noahmp_energy import LeafRequest, answer_on_host
from gpuwm.core.noahmp_sflx import sflx_steps


def _numbers(value, prefix="", out=None):
    """Every float/int leaf of a leaf answer, keyed by its path.

    Deliberately structural: it walks tuples, lists, dicts, arrays and plain
    attribute-carrying objects, so a leaf that grows an output is compared
    without this file being edited.  Non-numeric fields (VegeFluxState's
    ``branches`` trace dictionary, ``iters``) are dropped, because they are
    diagnostics rather than trajectory state and the device batch does not
    reconstruct them.
    """
    if out is None:
        out = {}
    if isinstance(value, (bool, np.bool_)):
        return out
    if isinstance(value, np.ndarray):
        for k, item in enumerate(value.ravel()):
            _numbers(item, f"{prefix}[{k}]", out)
        return out
    if isinstance(value, (int, float, np.floating, np.integer)):
        out[prefix] = np.float32(value).tobytes()
        return out
    if isinstance(value, (tuple, list)):
        for k, item in enumerate(value):
            _numbers(item, f"{prefix}[{k}]", out)
        return out
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            _numbers(value[key], f"{prefix}.{key}", out)
        return out
    fields = getattr(value, "__dict__", None)
    if fields is None:
        slots = getattr(type(value), "__slots__", None)
        if slots:
            fields = {name: getattr(value, name) for name in slots}
    if fields:
        for key in sorted(fields):
            if key in ("branches", "iters"):
                continue
            _numbers(fields[key], f"{prefix}.{key}", out)
    return out


def _collect():
    """``{leaf: (pristine_calls, host_answers, host_arguments_after)}``.

    The pristine copy is not fastidiousness.  WATER is INOUT in WRF and mutates
    the arrays it is handed -- ``SH2O``, ``SICE``, the snow stack, ``STC``,
    ``ZSNSO`` -- so a harness that answers on the host and then hands the *same*
    tuples to the device is comparing two different calls and will report a
    difference that is entirely its own.  Each call is therefore deep-copied
    before the host sees it, and the host's post-call argument state is recorded
    so the device's in-place writes can be compared too.
    """
    import copy

    collected: dict[str, tuple[list, list, list]] = {}
    for case in T.CASES:
        x = T.SF[(case, "input")]
        p = T._params(x)
        col = T._column(x)
        steps = sflx_steps(
            p, col, T._vec(x, "smc", 1, T.NSOIL),
            ficeold=T._vec(x, "ficeold", -T.NSNOW + 1, 0),
            wa=x[("wa", 0)], wslake=x[("wslake", 0)], **T._forcing(x))
        reply = None
        while True:
            try:
                request = steps.send(reply)
            except StopIteration:
                break
            pristine = copy.deepcopy((request.args, request.kwargs))
            reply = answer_on_host(request)
            calls, answers, afters = collected.setdefault(
                request.leaf, ([], [], []))
            calls.append(pristine)
            answers.append(reply)
            afters.append(_numbers((request.args, request.kwargs)))
    return collected


COLLECTED = _collect()

#: Every leaf the runtime batches.  Asserting the set rather than iterating it
#: is what makes a *newly* batched leaf fail this file instead of silently
#: skipping it.
EXPECTED_LEAVES = {"sflx_pre", "thermoprop", "radiation", "vege_flux",
                   "bare_flux", "tsnosoi", "phasechange", "water"}

#: Outputs a device batch deliberately does not reconstruct, and why.  Listing
#: them is what keeps the comparison from being loosened silently: the key sets
#: must differ by *exactly* this and nothing else, so a leaf that starts
#: dropping a live output fails here even though its numbers still match.
UNRECONSTRUCTED = {
    "water": {
        ".etrani": "WRF computes ETRANI as a local inside WATER "
                   "(module_sf_noahmplsm.F:6107, :6170); NOAHMP_SFLX never "
                   "reads it and the device row does not carry it, so "
                   "gpuwm.core.noahmp_water_gpu returns None rather than a "
                   "second implementation of something nothing consumes",
    },
}


def test_the_fixtures_reach_every_batched_leaf():
    from gpuwm.core.noahmp_runtime import DEVICE_LEAF_BATCH

    assert set(DEVICE_LEAF_BATCH) == EXPECTED_LEAVES, sorted(DEVICE_LEAF_BATCH)
    assert set(COLLECTED) == EXPECTED_LEAVES, sorted(COLLECTED)
    # VEGE_FLUX is the only leaf a column can skip, so it is the only one that
    # may be reached fewer than once per fixture column.
    for leaf, (calls, _answers, _after) in COLLECTED.items():
        if leaf == "vege_flux":
            assert 0 < len(calls) < len(T.CASES), len(calls)
        else:
            assert len(calls) == len(T.CASES), (leaf, len(calls))


@requires_gpu
@pytest.mark.parametrize("leaf", sorted(EXPECTED_LEAVES))
def test_the_device_batch_reproduces_the_host_leaf_bitwise(leaf):
    from gpuwm.core.noahmp_runtime import evaluate_leaf_batch_on_device

    import copy

    calls, want, want_after = COLLECTED[leaf]
    exempt = UNRECONSTRUCTED.get(leaf, {})
    device_calls = copy.deepcopy(calls)
    got = evaluate_leaf_batch_on_device(leaf, device_calls)
    assert len(got) == len(want), (leaf, len(got), len(want))

    # The in-place half.  WATER writes its answer into the arrays it is handed
    # as much as into what it returns, and a packer that unpacked into the
    # wrong slice would leave the returned record right and the column wrong.
    for index, (call, after) in enumerate(zip(device_calls, want_after)):
        device_after = _numbers(call)
        assert device_after.keys() == after.keys(), (leaf, index)
        differing = [key for key in after if device_after[key] != after[key]]
        assert not differing, (leaf, index, "in-place", differing[:10])

    compared = 0
    for index, (g, w) in enumerate(zip(got, want)):
        gn = _numbers(g)
        wn = _numbers(w)
        missing = {key.split("[")[0] for key in set(wn) - set(gn)}
        extra = set(gn) - set(wn)
        assert not extra, (leaf, index, sorted(extra)[:10])
        assert missing == set(exempt), (
            leaf, index, "the device batch dropped an output that is not "
            "listed in UNRECONSTRUCTED", sorted(missing ^ set(exempt))[:10])
        assert gn, (leaf, index, "nothing was compared")
        differing = [key for key in gn if gn[key] != wn[key]]
        assert not differing, (leaf, index, differing[:10])
        compared += len(gn)
    assert compared >= len(calls), (leaf, compared)


@requires_gpu
@pytest.mark.parametrize("leaf", sorted(EXPECTED_LEAVES))
def test_the_batch_comparison_can_fail(leaf):
    """Misroute one column's answer and the same comparison must reject it.

    This is the defect a batched seam actually produces -- results returned in
    an order the caller does not expect -- and until it is seen rejected, the
    comparison above is a test that has never been observed to fail.  Every
    fixture column reaches every leaf here except VEGE_FLUX, and three of the
    four reach that, so each parametrisation has at least two answers to swap.
    """
    from gpuwm.core.noahmp_runtime import evaluate_leaf_batch_on_device

    import copy

    calls, want, _after = COLLECTED[leaf]
    assert len(calls) > 1, (leaf, "nothing to misroute")
    got = list(evaluate_leaf_batch_on_device(leaf, copy.deepcopy(calls)))
    got[0], got[1] = got[1], got[0]
    moved = [k for k, (g, w) in enumerate(zip(got, want))
             if _numbers(g) != _numbers(w)]
    assert moved, (
        f"swapping two {leaf} answers left every compared word unchanged; "
        "the comparison cannot see a misrouted column")


@requires_gpu
def test_a_transposed_radiation_slot_is_rejected():
    """The RADIATION packer's slot order is load-bearing and is shown to be.

    SOLAD and SOLAI are adjacent, carry different numbers, and reach different
    terms of SURRAD.  Transposing them is the packing defect this seam is most
    likely to produce, so it is the one that has to be observed failing.
    """
    from gpuwm.core.noahmp_radiation_gpu import (N_OUTPUT,
                                                 pack_radiation_calls,
                                                 unpack_radiation_row)

    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    calls, want, _after = COLLECTED["radiation"]
    inputs, ist = pack_radiation_calls(calls)
    daylight = [k for k in range(inputs.shape[0])
                if inputs[k, 25] > 0.0 and
                (inputs[k, 52:54] != inputs[k, 54:56]).any()]
    assert daylight, "no fixture column has COSZ > 0 and SOLAD != SOLAI"

    swapped = inputs.copy()
    swapped[:, 52:54], swapped[:, 54:56] = \
        inputs[:, 54:56].copy(), inputs[:, 52:54].copy()

    count = swapped.shape[0]
    out = cp.empty(count * N_OUTPUT, dtype=cp.float32)
    kernel = get_kernel("noahmp_radiation", "noahmp_radiation_batch")
    kernel(((count + 63) // 64,), (64,),
           (cp.asarray(swapped.ravel()), cp.asarray(ist), out,
            np.int32(count)))
    rows = cp.asnumpy(out).reshape(count, N_OUTPUT)

    moved = [k for k in daylight
             if _numbers(unpack_radiation_row(rows[k])) != _numbers(want[k])]
    assert moved, (
        "transposing SOLAD and SOLAI changed nothing on any daylight column; "
        "the RADIATION gate cannot see a packing order defect")
