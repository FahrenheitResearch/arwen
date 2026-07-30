"""Bitwise gate for the whole-slab WATER packing.

:mod:`gpuwm.core.noahmp_water_slab` exists to delete a Python loop, not to
change an answer.  So the bar is not "close", it is *identical*: the same
fixture columns packed by the per-column
:func:`gpuwm.core.noahmp_water_gpu.pack_water_calls` and by the slab packer
must produce byte-identical device rows, and the two evaluations must produce
byte-identical outputs, field by field.

Three things this file has to establish, in this order:

1. the packing is the same packing
   (``test_the_slab_packing_is_byte_identical``);
2. the evaluation is the same evaluation, including the column state WRF
   mutates in place, not just the flux record
   (``test_the_slab_evaluation_is_byte_identical``);
3. the comparison in 1 and 2 can *fail*.  Three negative controls do that,
   each constructing a corrupted packing rather than asserting a claim about
   one: an adjacent-block transposition (SH2O against SICE, the swap a reader
   is most likely to make), a one-column roll of a live forcing field (the
   failure mode a wrong axis order produces), and the same roll carried
   through the kernel to the outputs.

The column values come from ``gpuwm/data/noahmp/oracle/noahmp-water.csv`` by
way of :mod:`tests.test_noahmp_water`, so nothing here invents physics.  The
36 fixture cases are heterogeneous in exactly the ways that matter to a
packing gate -- ISNOW at 0, -1, -2 and -3, soil moisture from 0.008 to 0.20,
rain columns, snowfall columns, lake columns and soil columns --
and ``test_the_fixture_columns_are_heterogeneous`` refuses to let that
silently stop being true, because a slab of identical columns cannot detect a
transposition at all.

The ``fields`` slabs are built here by an explicit, named transcription off
the fixture table -- the same independent reading
``test_noahmp_water._fixture_rows_packed`` uses -- rather than by restacking
what ``_call`` already ordered.  A gate that fed both packers the same
pre-ordered tuple would be checking that CuPy can copy an array.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import requires_gpu
from test_noahmp_water import CASES, NSNOW, NSOIL, TABLE, _call

from gpuwm.core.noahmp_water import WaterFluxes
from gpuwm.core.noahmp_water_gpu import (INT_STRIDE, IN_STRIDE, P_STRIDE,
                                         STATE_SLOT, STATE_WIDTH)

pytestmark = pytest.mark.gpu

NFULL = NSNOW + NSOIL

#: Every ``SnowColumn`` attribute WATER mutates, i.e. what ``_unpack`` writes
#: back.  Kept beside ``STATE_SLOT`` so a new state slot fails this file.
COLUMN_STATE = ("snowh", "sneqv", "snice", "snliq", "stc", "zsnso", "dzsnso",
                "sh2o", "sice")


# ---------------------------------------------------------------------------
# the slab inputs, read off the fixture by name
# ---------------------------------------------------------------------------

def _scalar(rows, name, dtype=np.float32):
    """One ``(n,)`` slab from a per-case scalar column of the fixture."""
    return np.asarray([r[(name, 0)] for r in rows], dtype=dtype)


def _layers(rows, name, lo, count, dtype=np.float32):
    """One ``(n, count)`` slab, layer axis last, in WRF's own subscripts."""
    return np.asarray([[r[(name, lo + k)] for k in range(count)]
                       for r in rows], dtype=dtype)


def _slab_fields(cp) -> dict:
    """The whole-slab spelling of the 36 fixture columns."""
    x = [TABLE[(c, "input")] for c in CASES]
    p = [TABLE[(c, "param")] for c in CASES]
    top = -NSNOW + 1

    host: dict[str, np.ndarray] = {}
    # the soil/veg parameter handle, flattened
    for name in ("smcmax", "smcwlt", "bexp", "dksat", "dwsat"):
        host[name] = _layers(p, name, 1, NSOIL)
    for name in ("kdt", "frzx", "slope", "ch2op", "ssi", "snow_ret_fac"):
        host[name] = _scalar(p, name)
    for name in ("urban_flag", "nroot"):
        host[name] = _scalar(p, name, dtype=np.int32)

    # the SnowColumn, flattened
    for name in ("snowh", "sneqv"):
        host[name] = _scalar(x, name)
    for name in ("snice", "snliq"):
        host[name] = _layers(x, name, top, NSNOW)
    for name in ("stc", "zsnso", "dzsnso"):
        host[name] = _layers(x, name, top, NFULL)
    for name in ("sh2o", "sice"):
        host[name] = _layers(x, name, 1, NSOIL)

    # the forcing
    for name in ("dt", "fcev", "fctr", "elai", "esai", "fveg", "bdfall",
                 "sfctmp", "qvap", "qdew", "qsnow", "qrain", "snowhin",
                 "ponding", "canliq", "canice", "tv", "wslake", "acc_qinsur",
                 "acc_qseva"):
        host[name] = _scalar(x, name)
    for name in ("zsoil", "btrani", "acc_etrani", "smc"):
        host[name] = _layers(x, name, 1, NSOIL)
    host["ficeold"] = _layers(x, "ficeold", top, NSNOW)

    # the integer row
    for name in ("isnow", "ist", "frozen_canopy", "frozen_ground"):
        host[name] = _scalar(x, name, dtype=np.int32)
    host["imelt"] = _layers(x, "imelt", top, NSNOW, dtype=np.int32)

    return {name: cp.asarray(value) for name, value in host.items()}


def _calls():
    """The per-column physical call list, one fresh SnowColumn per column."""
    return [(_call(case), {}) for case in CASES]


# ---------------------------------------------------------------------------
# comparison helpers.  Everything is compared as raw bits, never as a float:
# 0.0 against -0.0, and any NaN payload, must be a failure here.
# ---------------------------------------------------------------------------

def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.uint8)


def _same_bits(got, want, what: str):
    assert got.dtype == want.dtype, \
        f"{what}: dtype {got.dtype} against {want.dtype}"
    assert got.shape == want.shape, \
        f"{what}: shape {got.shape} against {want.shape}"
    np.testing.assert_array_equal(_bits(got), _bits(want),
                                  err_msg=f"{what}: bytes differ")
    assert _bits(got).tobytes() == _bits(want).tobytes(), what


def _host_outputs(calls, fluxes) -> dict:
    """The per-column results restacked as slabs, for comparison only."""
    cols = [args[1] for args, _kwargs in calls]
    got = {"isnow": np.asarray([c.isnow for c in cols], dtype=np.int32)}
    for name in COLUMN_STATE:
        if STATE_WIDTH[name] == 1:
            got[name] = np.asarray([getattr(c, name) for c in cols],
                                   dtype=np.float32)
        else:
            got[name] = np.stack([np.asarray(getattr(c, name),
                                             dtype=np.float32)
                                  for c in cols])
    for field in dataclasses.fields(WaterFluxes):
        name = field.name
        if name == "etrani":
            continue
        value = [getattr(f, name) for f in fluxes]
        got[name] = (np.stack([np.asarray(v, dtype=np.float32) for v in value])
                     if np.ndim(value[0]) else
                     np.asarray(value, dtype=np.float32))
    return got


# ---------------------------------------------------------------------------
# the fixture itself
# ---------------------------------------------------------------------------

def test_the_fixture_columns_are_heterogeneous():
    """A slab of identical columns cannot detect a transposition.

    Everything below rests on the 36 columns actually differing from each
    other in the fields a wrong slot or a wrong axis would move, so that is
    asserted rather than assumed.
    """
    x = [TABLE[(c, "input")] for c in CASES]
    assert len(CASES) >= 32, f"only {len(CASES)} columns"

    isnow = {int(r[("isnow", 0)]) for r in x}
    assert 0 in isnow, "no bare-ground column"
    assert any(k <= -2 for k in isnow), "no multi-layer snow pack"
    assert len(isnow) >= 3, f"ISNOW takes only {sorted(isnow)}"

    sh2o = {float(r[("sh2o", k)]) for r in x for k in range(1, NSOIL + 1)}
    assert len(sh2o) >= 4, f"soil moisture takes only {sorted(sh2o)}"

    assert any(float(r[("qrain", 0)]) > 0.0 for r in x), "no rain column"
    assert any(float(r[("qsnow", 0)]) > 0.0 for r in x), "no snowfall column"
    assert {int(r[("ist", 0)]) for r in x} == {1, 2}, "lake or soil missing"

    # No two columns are the same column, so a one-column roll is always
    # detectable -- which is what the negative controls below depend on.
    for name in ("qrain", "sneqv", "dt"):
        values = [float(r[(name, 0)]) for r in x]
        assert len(set(values)) > 1, f"{name} is constant across the slab"


def test_the_field_name_space_is_unambiguous():
    """No name may be a float slab in one place and an int slab in another."""
    from gpuwm.core import noahmp_water_slab as slab

    overlap = set(slab.FLOAT_FIELDS) & set(slab.INT_FIELDS)
    assert not overlap, f"{sorted(overlap)} is both a float and an int field"
    assert len(slab.FIELD_NAMES) == len(set(slab.FIELD_NAMES))
    assert set(COLUMN_STATE) == set(STATE_SLOT), \
        "the packed column grew a slot this file does not compare"


# ---------------------------------------------------------------------------
# 1. the packing gate
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_slab_packing_is_byte_identical():
    """The slab rows must be the per-column rows, bit for bit."""
    import cupy as cp

    from gpuwm.core.noahmp_water_gpu import pack_water_calls
    from gpuwm.core.noahmp_water_slab import pack_water_slab

    want_par, want_fin, want_iin = pack_water_calls(_calls())
    par, fin, iin = pack_water_slab(_slab_fields(cp), len(CASES))

    assert (par.shape, fin.shape, iin.shape) == (
        (len(CASES), P_STRIDE), (len(CASES), IN_STRIDE),
        (len(CASES), INT_STRIDE))
    for got, want, what in ((par, want_par, "par"), (fin, want_fin, "fin"),
                            (iin, want_iin, "iin")):
        assert got.flags.c_contiguous, f"{what} is not contiguous"
        _same_bits(cp.asnumpy(got), want, what)


@requires_gpu
def test_a_strided_input_slab_packs_identically():
    """A view is a legal input; a view is not a legal kernel argument.

    The failure this lane has hit twice is a strided CuPy view reaching a raw
    kernel, where the pointer arrives and the stride does not.  The packer
    must therefore accept a strided field -- slice assignment respects strides
    -- and must still hand the launch contiguous rows.
    """
    import cupy as cp

    from gpuwm.core.noahmp_water_slab import pack_water_slab

    fields = _slab_fields(cp)
    n = len(CASES)
    strided = {}
    for name, value in fields.items():
        if value.ndim == 1:
            pad = cp.zeros((n, 2), dtype=value.dtype)
            pad[:, 0] = value
            view = pad[:, 0]
        else:
            pad = cp.zeros((n, value.shape[1], 2), dtype=value.dtype)
            pad[:, :, 0] = value
            view = pad[:, :, 0]
        assert not view.flags.c_contiguous or n <= 1
        strided[name] = view

    want_par, want_fin, want_iin = pack_water_slab(fields, n)
    par, fin, iin = pack_water_slab(strided, n)
    for got, want, what in ((par, want_par, "par"), (fin, want_fin, "fin"),
                            (iin, want_iin, "iin")):
        assert got.flags.c_contiguous, f"{what} is not contiguous"
        _same_bits(cp.asnumpy(got), cp.asnumpy(want), f"strided {what}")


@requires_gpu
@pytest.mark.parametrize("name,width", [
    ("sh2o", NSOIL), ("stc", NFULL), ("ficeold", NSNOW), ("dt", 1),
])
def test_a_mis_shaped_field_is_rejected(name, width):
    """A short vector broadcast into a wide slot is a wrong forecast."""
    import cupy as cp

    from gpuwm.core.noahmp_water_slab import pack_water_slab

    fields = _slab_fields(cp)
    n = len(CASES)
    bad = dict(fields)
    bad[name] = (cp.zeros((n, 1), dtype=fields[name].dtype) if width == 1
                 else cp.zeros((n, width - 1), dtype=fields[name].dtype))
    with pytest.raises(ValueError, match=name):
        pack_water_slab(bad, n)

    missing = dict(fields)
    del missing[name]
    with pytest.raises(KeyError, match=name):
        pack_water_slab(missing, n)


# ---------------------------------------------------------------------------
# 2. the evaluation gate
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_slab_evaluation_is_byte_identical():
    """Every output, including the mutated column, bit for bit."""
    import cupy as cp

    from gpuwm.core.noahmp_water_gpu import evaluate_water_calls
    from gpuwm.core.noahmp_water_slab import (ABSENT_OUTPUTS, OUTPUT_NAMES,
                                              evaluate_water_slab)

    calls = _calls()
    fluxes = evaluate_water_calls(calls)
    want = _host_outputs(calls, fluxes)
    got = evaluate_water_slab(_slab_fields(cp), len(CASES))

    assert set(got) == set(OUTPUT_NAMES)
    assert set(got) == set(want), (
        f"slab-only {sorted(set(got) - set(want))}, "
        f"column-only {sorted(set(want) - set(got))}")
    for name in ABSENT_OUTPUTS:
        assert getattr(fluxes[0], name) is None, \
            f"{name} is no longer absent from the per-column path"
        assert name not in got

    for name in sorted(got):
        assert got[name].flags.c_contiguous, f"{name} is a strided view"
        _same_bits(cp.asnumpy(got[name]), want[name], name)


@requires_gpu
def test_the_slab_evaluation_covers_every_flux_and_state_member():
    """Nothing WATER produces may quietly not be compared above."""
    from gpuwm.core.noahmp_water_slab import ABSENT_OUTPUTS, OUTPUT_NAMES

    members = {f.name for f in dataclasses.fields(WaterFluxes)}
    missing = members - set(OUTPUT_NAMES) - set(ABSENT_OUTPUTS)
    assert not missing, f"the slab returns no {sorted(missing)}"
    assert set(COLUMN_STATE) | {"isnow"} <= set(OUTPUT_NAMES)


# ---------------------------------------------------------------------------
# 3. the negative controls -- the failing form, shown
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_packing_gate_rejects_an_adjacent_block_transposition():
    """SH2O and SICE are adjacent, both live, and the likeliest swap."""
    import cupy as cp

    from gpuwm.core.noahmp_water_gpu import pack_water_calls
    from gpuwm.core.noahmp_water_slab import pack_water_slab

    _wp, want_fin, _wi = pack_water_calls(_calls())
    _par, fin, _iin = pack_water_slab(_slab_fields(cp), len(CASES))
    good = cp.asnumpy(fin)
    _same_bits(good, want_fin, "fin")          # the uncorrupted form passes

    a, b = STATE_SLOT["sh2o"], STATE_SLOT["sice"]
    assert b == a + STATE_WIDTH["sh2o"], "SH2O and SICE are no longer adjacent"
    bad = good.copy()
    bad[:, a:a + NSOIL] = good[:, b:b + NSOIL]
    bad[:, b:b + NSOIL] = good[:, a:a + NSOIL]
    assert (bad != good).any(), "the transposition changed nothing to detect"

    with pytest.raises(AssertionError):
        _same_bits(bad, want_fin, "fin")

    first = int(np.flatnonzero(bad.ravel().view(np.uint32)
                               != want_fin.ravel().view(np.uint32))[0])
    assert first % IN_STRIDE in range(a, b + NSOIL), \
        "the first detected difference is not in the transposed blocks"


@requires_gpu
def test_the_packing_gate_rejects_a_one_column_roll():
    """The failure a wrong axis order produces: right values, wrong column."""
    import cupy as cp

    from gpuwm.core.noahmp_water_gpu import pack_water_calls
    from gpuwm.core.noahmp_water_slab import pack_water_slab

    _wp, want_fin, _wi = pack_water_calls(_calls())
    fields = _slab_fields(cp)
    rolled = dict(fields)
    rolled["qrain"] = cp.roll(fields["qrain"], 1)
    assert not bool((rolled["qrain"] == fields["qrain"]).all()), \
        "rolling QRAIN moved nothing"

    _par, fin, _iin = pack_water_slab(rolled, len(CASES))
    with pytest.raises(AssertionError):
        _same_bits(cp.asnumpy(fin), want_fin, "fin")


@requires_gpu
def test_the_evaluation_gate_rejects_a_one_column_roll():
    """And the same corruption is still visible after the launch."""
    import cupy as cp

    from gpuwm.core.noahmp_water_slab import evaluate_water_slab

    fields = _slab_fields(cp)
    n = len(CASES)
    good = evaluate_water_slab(fields, n)

    rolled = dict(fields)
    rolled["qrain"] = cp.roll(fields["qrain"], 1)
    bad = evaluate_water_slab(rolled, n)

    moved = sorted(name for name in good
                   if _bits(cp.asnumpy(good[name])).tobytes()
                   != _bits(cp.asnumpy(bad[name])).tobytes())
    assert moved, "rolling QRAIN by one column moved no output at all"
    assert "sh2o" in moved and "runsrf" in moved, \
        f"QRAIN moved only {moved}, which does not include the soil column"
