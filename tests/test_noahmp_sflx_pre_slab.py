"""The whole-slab SFLX prefix, against the per-column batch, bit for bit.

:func:`gpuwm.core.noahmp_sflx_pre_slab.evaluate_sflx_pre_slab` answers the same
physical calls as :func:`gpuwm.core.noahmp_sflx_pre_gpu.evaluate_sflx_pre_calls`
with the same four kernels; what it removes is the CPython between them.  So the
only thing that can be wrong with it is *packing* -- a value in the wrong
column, the wrong keyword or the wrong slot -- and packing is the one defect a
bitwise arithmetic comparison of the kernels themselves cannot see.  The gate is
therefore: run both on identical inputs and require every field of
:class:`~gpuwm.core.noahmp_sflx.PreEnergy` and of its nested
:class:`~gpuwm.core.noahmp_sflx.EnergyCall` to be bitwise identical, field by
field, column by column.

The columns are the four unmodified-WRF whole-column fixtures crossed against
each other three ways -- parameters and snow column from one case, forcing from
a second, SMC from a third -- which gives 64 heterogeneous land columns and
invents no number.  The diagonal of that cross product is the four fixture
columns themselves; the off-diagonal entries pair a fixture's snow pack with
another fixture's weather, which is exactly the heterogeneity a real tile has
and which a per-column list can hide from a slab packer.
``test_the_slab_columns_are_heterogeneous`` asserts the coverage the gate needs
(rain, snowfall, bare, vegetated, a multi-layer snow pack, ``isnow == 0``) out
of the *host* answers, so it is proved rather than claimed.

Two negative controls run first in spirit and last in file order: rolling one
input field one column onto its neighbour, and transposing two input fields.
Both are the defect a slab seam actually produces, and both are shown being
rejected -- until that is observed, the comparison above is a gate that has
never been seen to fail.
"""

from __future__ import annotations

import copy
from dataclasses import fields as dc_fields

import numpy as np
import pytest

from conftest import requires_gpu

import test_noahmp_sflx as T

from gpuwm.core import noahmp_sflx_pre_gpu as _batch
from gpuwm.core import noahmp_sflx_pre_slab as _slab
from gpuwm.core.noahmp_sflx import EnergyCall, PreEnergy


# ---------------------------------------------------------------------------
# the columns
# ---------------------------------------------------------------------------

def _base_calls():
    """One ``(args, kwargs)`` per fixture case, as ``sflx_steps`` yields it."""
    calls = []
    for case in T.CASES:
        x = T.SF[(case, "input")]
        kwargs = T._forcing(x)
        kwargs["wa"] = x[("wa", 0)]
        kwargs["acc_ssoil"] = np.float32(0.0)
        calls.append(((T._params(x), T._column(x),
                       T._vec(x, "smc", 1, T.NSOIL)), kwargs))
    return calls


def _mixed_calls():
    """64 heterogeneous columns, every number of them out of a fixture."""
    base = _base_calls()
    out = []
    for par, col, _smc in (c[0] for c in base):
        for kwargs in (c[1] for c in base):
            for smc in (c[0][2] for c in base):
                out.append(((par, col.copy(), smc.copy()),
                            copy.deepcopy(kwargs)))
    return out


CALLS = _mixed_calls()
N = len(CALLS)


# ---------------------------------------------------------------------------
# the input contract, and the packing that satisfies it
# ---------------------------------------------------------------------------

#: Every name :func:`evaluate_sflx_pre_slab` requires.  Asserted below rather
#: than only derived, so a silently widening or narrowing contract is visible
#: here instead of only in the module that defines it.
REQUIRED = (set(_slab._SCALAR_FIELDS) | set(_slab._INT_FIELDS)
            | set(_slab._LAYER_FIELDS) | set(_batch._ATM_IN))

_INT_NAMES = set(_slab._INT_FIELDS)
_LAYER_NAMES = dict(_slab._LAYER_FIELDS)


def _value(name, args, kwargs):
    """One column's value for one slab field, from wherever it lives.

    ``sflx_pre``'s arguments are three different objects -- the keyword set,
    the :class:`~gpuwm.core.noahmp_snow.SnowColumn` and the parameter handle --
    and the slab's contract is that its field names are those objects' names
    verbatim.  Resolving by name is what holds that contract: a slab field that
    is not one of the three is a ``KeyError`` here.
    """
    par, col, smc = args
    if name == "smc":
        return smc
    if name in kwargs:
        return kwargs[name]
    if hasattr(col, name):
        return getattr(col, name)
    return getattr(par, name)


def _slab_fields(calls):
    """The whole batch as one CuPy array per field, layer axis last."""
    import cupy as cp

    fields = {}
    for name in sorted(REQUIRED):
        column = [_value(name, args, kwargs) for args, kwargs in calls]
        if name in _LAYER_NAMES:
            packed = np.asarray([np.asarray(v, dtype=np.float32)
                                 for v in column], dtype=np.float32)
        elif name == "urban_flag":
            packed = np.asarray([int(bool(v)) for v in column],
                                dtype=np.int32)
        elif name in _INT_NAMES:
            packed = np.asarray([int(v) for v in column], dtype=np.int32)
        else:
            packed = np.asarray([np.float32(v) for v in column],
                                dtype=np.float32)
        fields[name] = cp.asarray(packed)
    return fields


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------

PRE_NAMES = tuple(f.name for f in dc_fields(PreEnergy) if f.name != "call")
CALL_NAMES = tuple(f.name for f in dc_fields(EnergyCall))

#: Every compared field, in the order the two records declare them.  Pinning
#: the count is what stops a comparison that quietly shrank to two scalars from
#: passing: the gate asserts this number and the per-column product of it.
COMPARED_FIELDS = PRE_NAMES + tuple(f"call.{n}" for n in CALL_NAMES)


def _bits(value) -> bytes:
    """A value's exact machine representation, tagged by kind.

    Bytes, not ``==``: this comparison has to be bitwise, and ``==`` cannot see
    ``-0.0`` versus ``+0.0`` -- which is the precise hazard a vectorised
    ``maximum`` introduces -- nor call two identical NaNs equal.
    """
    a = np.asarray(value)
    if a.dtype.kind in "biu":
        return b"i" + np.ascontiguousarray(a, dtype=np.int64).tobytes()
    return b"f" + np.ascontiguousarray(a, dtype=np.float32).tobytes()


def _host_answers(calls):
    from gpuwm.core.noahmp_sflx_pre_gpu import evaluate_sflx_pre_calls

    return evaluate_sflx_pre_calls(copy.deepcopy(calls))


def _slab_answer(fields, n):
    from gpuwm.core.noahmp_sflx_pre_slab import evaluate_sflx_pre_slab

    return evaluate_sflx_pre_slab(fields, n)


def _differences(want, got):
    """``(compared, [(field, column, host_bits, slab_bits), ...])``."""
    import cupy as cp

    assert set(got) == set(COMPARED_FIELDS), (
        "the slab bundle does not carry exactly PreEnergy + EnergyCall: "
        f"missing {sorted(set(COMPARED_FIELDS) - set(got))}, "
        f"extra {sorted(set(got) - set(COMPARED_FIELDS))}")

    compared, bad = 0, []
    for key in COMPARED_FIELDS:
        name = key.split(".", 1)[-1]
        host = [getattr(w.call if key.startswith("call.") else w, name)
                for w in want]
        device = cp.asnumpy(got[key])
        assert len(device) == len(host), (key, len(device), len(host))
        for k, value in enumerate(host):
            compared += 1
            hb, sb = _bits(value), _bits(device[k])
            if hb != sb:
                bad.append((key, k, hb, sb))
    return compared, bad


# ---------------------------------------------------------------------------
# 1. the gate
# ---------------------------------------------------------------------------

def test_the_compared_field_set_is_the_whole_record():
    """A shrinking comparison must fail here before it can pass silently."""
    assert len(PRE_NAMES) == 37, len(PRE_NAMES)
    assert len(CALL_NAMES) == 67, len(CALL_NAMES)
    assert len(COMPARED_FIELDS) == 104, len(COMPARED_FIELDS)
    assert len(set(COMPARED_FIELDS)) == len(COMPARED_FIELDS)
    assert N == 64, N


def test_the_slab_input_contract_is_the_sflx_pre_argument_set():
    assert len(REQUIRED) == 67, sorted(REQUIRED)
    # Every required name resolves out of one of ``sflx_pre``'s three
    # arguments; nothing has to be invented to drive the slab.
    args, kwargs = CALLS[0]
    for name in sorted(REQUIRED):
        assert _value(name, args, kwargs) is not None, name


@requires_gpu
def test_the_slab_columns_are_heterogeneous():
    """The gate is only as good as the paths its columns reach."""
    want = _host_answers(CALLS)
    isnow = [int(w.call.isnow) for w in want]
    assert any(w.rain > 0.0 for w in want), "no column has rain"
    assert any(w.snow > 0.0 for w in want), "no column has snowfall"
    assert any(w.fveg == 0.0 for w in want), "no bare column"
    assert any(w.fveg > 0.0 for w in want), "no vegetated column"
    assert any(k <= -2 for k in isnow), "no multi-layer snow pack"
    assert any(k == 0 for k in isnow), "no snow-free column"
    assert any(w.qsnow > 0.0 for w in want), "PRECIP_HEAT never sees snowfall"
    assert any(w.canliq > 0.0 for w in want), "no column intercepts rain"


@requires_gpu
def test_the_slab_reproduces_the_per_column_batch_bitwise():
    want = _host_answers(CALLS)
    got = _slab_answer(_slab_fields(CALLS), N)

    compared, bad = _differences(want, got)
    assert compared == 104 * N, compared
    assert not bad, [
        f"{key} column {k}: batch {hb.hex()} slab {sb.hex()}"
        for key, k, hb, sb in bad[:12]]


# ---------------------------------------------------------------------------
# 2. the same comparison, shown failing
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("name", ["sfctmp", "shdmax", "snowh", "stc"])
def test_a_rolled_input_column_is_rejected(name):
    """One field one column out of step must be seen.

    This is the defect a slab packer produces that a per-column list cannot:
    every array is the right shape and the right dtype and carries the right
    values, and one of them is indexed by the wrong column.  ``sfctmp`` reaches
    ATM, ``shdmax`` reaches the FVEG selection, ``snowh`` reaches PHENOLOGY and
    ``stc`` is a whole layer vector reaching the marshalling kernel, so the
    four between them cover every stage of the slab.
    """
    import cupy as cp

    want = _host_answers(CALLS)
    fields = _slab_fields(CALLS)
    fields[name] = cp.roll(fields[name], 1, axis=0)

    _compared, bad = _differences(want, _slab_answer(fields, N))
    assert bad, (
        f"rolling {name} one column onto its neighbour left every one of the "
        f"{len(COMPARED_FIELDS)} compared fields bitwise identical on all "
        f"{N} columns; the slab gate cannot see a misindexed column")


@requires_gpu
@pytest.mark.parametrize("first,second", [
    ("sfcprs", "sfctmp"),
    ("zsnso", "stc"),
])
def test_two_transposed_input_fields_are_rejected(first, second):
    """Two same-shaped fields swapped must move the answer.

    A slab is addressed by name only at its edges; inside, ``sfcprs`` and
    ``sfctmp`` are two ``(n,)`` float32 arrays that fill adjacent slots of one
    flat ATM row, and ``zsnso`` and ``stc`` are two ``(n, NSNOW+NSOIL)`` arrays
    that reach ``k_sflx_marshal`` as adjacent pointers.  Nothing about the
    shapes or the dtypes distinguishes either pair, so nothing but this test
    does.
    """
    fields = _slab_fields(CALLS)
    assert fields[first].shape == fields[second].shape
    assert fields[first].dtype == fields[second].dtype
    fields[first], fields[second] = fields[second], fields[first]

    _compared, bad = _differences(_host_answers(CALLS),
                                  _slab_answer(fields, N))
    assert bad, (
        f"transposing {first} and {second} left every compared field bitwise "
        "identical on every column; the slab gate cannot see a transposition")


# ---------------------------------------------------------------------------
# 3. the guards, and the empty tile
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_slab_refuses_an_inadmissible_column():
    """``sflx_pre``'s own guards must survive the conversion.

    The dead irrigation and crop regions are dead *because* CROPTYPE and the
    four irrigation amounts are zero.  A slab that dropped the checks would run
    the admitted arithmetic on a column the port has never validated and report
    nothing.  Each guard is driven on a single column of a 64-column slab, so
    what is tested is the reduction over the column axis and not a scalar.
    """
    import cupy as cp

    fields = _slab_fields(CALLS)

    crop = dict(fields)
    crop["croptype"] = fields["croptype"].copy()
    crop["croptype"][7] = 1
    with pytest.raises(ValueError, match="croptype must be 0"):
        _slab_answer(crop, N)

    for name in ("irrfra", "iramtsi", "iramtmi", "iramtfi"):
        wet = dict(fields)
        wet[name] = cp.zeros(N, dtype=cp.float32)
        wet[name][11] = np.float32(0.25)
        with pytest.raises(ValueError, match=f"{name} must be 0.0"):
            _slab_answer(wet, N)


@requires_gpu
def test_a_missing_field_is_named():
    fields = _slab_fields(CALLS)
    fields.pop("shdmax")
    with pytest.raises(KeyError, match="shdmax"):
        _slab_answer(fields, N)


@requires_gpu
def test_a_layer_axis_the_wrong_way_round_is_refused():
    """``(NFULL, n)`` is not ``(n, NFULL)``, and the slab must say so."""
    fields = _slab_fields(CALLS)
    fields["stc"] = fields["stc"].T
    with pytest.raises(ValueError, match="layer axis last"):
        _slab_answer(fields, N)


@requires_gpu
def test_an_empty_tile_is_an_empty_bundle():
    """A tile with no land columns launches nothing and still answers.

    A grid of zero blocks is a CUDA configuration error, so the empty slab is
    the one shape that can crash a launcher that never tries it.
    """
    import cupy as cp

    empty = {name: (cp.zeros((0, _LAYER_NAMES[name]), dtype=cp.float32)
                    if name in _LAYER_NAMES else
                    cp.zeros(0, dtype=cp.int32 if name in _INT_NAMES
                             else cp.float32))
             for name in REQUIRED}
    got = _slab_answer(empty, 0)
    assert set(got) == set(COMPARED_FIELDS)
    assert all(value.shape[0] == 0 for value in got.values())
