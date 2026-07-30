"""NOAHMP_SFLX's postfix over a slab, against the scalar authority.

:mod:`gpuwm.core.noahmp_sflx_post_slab` evaluates the same statements as
:func:`gpuwm.core.noahmp_sflx.sflx_post_steps` over the column axis.  Only the
scalar routine is pinned by the unmodified-WRF whole-column fixture
``noahmp-sflx.csv``, so every value the slab produces is compared with the
scalar one **bitwise**, on the four fixture columns at once.

How the reference is taken
--------------------------
The scalar routine is *driven*, once per column, and everything the comparison
needs is recorded **on the way past**:

* the WATER request and the answer to it.  WATER mutates the column it is
  handed -- ``ISNOW SNOWH SNEQV SNICE SNLIQ STC ZSNSO DZSNSO SH2O SICE`` are
  all INOUT -- so calling it a second time on the same objects is a different
  call, and the ENERGY exit column has to be snapshotted before the answer and
  the post-WATER column after it.  The request is unpacked through
  ``inspect.signature(water)``, so a reordering of WATER's positional argument
  list moves this comparison with it instead of silently reading QDEW where it
  meant QVAP;
* ERROR's whole :class:`~gpuwm.core.noahmp_sflx.ErrorResult`, by wrapping
  ``noahmp_sflx.error`` for the duration of the drive.  ERRSW and ERRENG are
  the two residuals the abort gates test and NOAHMP_SFLX keeps neither, so a
  transcription that computed them wrongly and never showed them would pass
  every whole-column check in this tree.  Recording beats recomputing: a test
  that rebuilt ERROR's argument list itself would be a second transcription of
  the marshalling it is supposed to be checking.

What is compared
----------------
Every field seg1 and seg2 own -- 13 and 19 of them, listed in the slab module
itself as ``SEG1_LANDED``/``SEG1_COMPUTED``/``SEG2_FIELDS`` -- field by field,
column by column, as bytes.  The word count is asserted exactly, so a
comparison that quietly shrank to a handful of scalars fails here instead of
passing.

Bytes rather than ``==``, because ``==`` cannot see ``-0.0`` against ``+0.0``
and that is the precise hazard ``cupy.minimum`` introduces at :983's
``ABS(MIN(RATIO, 0.0))``.

The negative controls
---------------------
Four, and all four were run and observed before this docstring was written:

* one column's inputs rolled onto its neighbour, in each segment;
* every column's WATER answer rolled onto its neighbour -- the defect a batched
  seam actually produces;
* two input fields transposed, in each segment, chosen so that the
  transposition is genuinely observable;
* ``SAV``/``SAG`` transposed, which is a **no-op**, kept as a measurement.  The
  postfix reads that pair only as ``SAV + SAG`` inside ERRENG (:1662) and FP32
  addition is commutative, so swapping them cannot move anything.  It is here
  so that the claim is checked rather than assumed: if the postfix ever starts
  reading the two separately, this test fails and the reason the pair was
  rejected as a control stops being true.

Rolling anything into ERROR usually trips one of WRF's three ``wrf_error_fatal``
gates rather than moving a bit, because those gates exist to catch exactly a
column whose water and energy no longer belong to it.  An abort is a stronger
rejection than a differing word, so the controls admit either and report which
one fired.
"""

from __future__ import annotations

import contextlib
import inspect

import numpy as np
import pytest

from conftest import requires_gpu

import test_noahmp_sflx as T

import gpuwm.core.noahmp_sflx as S
from gpuwm.core.noahmp_energy import answer_on_host
from gpuwm.core.noahmp_sflx import NoahmpBalanceError
from gpuwm.core.noahmp_water import water as _water

NSOIL = T.NSOIL
NSNOW = T.NSNOW


# ---------------------------------------------------------------------------
# the scalar run, recorded
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _recording_error(store):
    """Record every :class:`ErrorResult` ``sflx_post_steps`` produces.

    ``sflx_post_steps`` keeps six of ERROR's nine outputs and drops ERRSW,
    ERRENG and the ``ErrorResult`` itself.  Wrapping the module-global is how
    the three that never reach :class:`SflxResult` become comparable at all;
    the wrapper calls the unmodified routine and changes nothing.
    """
    original = S.error

    def spy(**kwargs):
        result = original(**kwargs)
        store.append(result)
        return result

    S.error = spy
    try:
        yield
    finally:
        S.error = original


def _f32(value):
    return np.float32(value)


def _vec(value):
    return np.asarray(value, dtype=np.float32).copy()


def _capture(case, *, urban=False):
    """One fixture column, driven through the scalar postfix and recorded.

    ENERGY is replayed out of ``noahmp-energy.csv`` rather than computed, for
    the reason ``test_noahmp_sflx.test_replayed_column_bitwise`` gives: a
    difference is then attributable to the postfix and cannot be absorbed into
    whatever the ENERGY port happens to do.

    ``urban`` flips ``parameters.urban_flag`` *after* the prefix has run, so it
    reaches :1062's override -- the one branch of the postfix no fixture column
    takes -- without perturbing FVEG at :874.  Both the scalar and the slab are
    driven from the same handle, so the comparison stays a comparison.
    """
    parameters, col, _smc, pre, x = T._run_pre(case)
    energy = T._replayed_energy(case)
    if urban:
        parameters.urban_flag = True

    steps = S.sflx_post_steps(
        parameters, col, pre, energy,
        dt=x[("dt", 0)], ist=int(x[("ist", 0)]),
        zsoil=T._vec(x, "zsoil", 1, NSOIL),
        ficeold=T._vec(x, "ficeold", -NSNOW + 1, 0),
        wa=x[("wa", 0)], wslake=x[("wslake", 0)],
        acc_qinsur=_f32(0.0), acc_qseva=_f32(0.0),
        acc_etrani=np.zeros(NSOIL, dtype=np.float32),
        acc_dwater=_f32(0.0), acc_prcp=_f32(0.0), acc_ecan=_f32(0.0),
        acc_etran=_f32(0.0), acc_edir=_f32(0.0))

    request = steps.send(None)
    assert request.leaf == "water", request.leaf
    bound = inspect.signature(_water).bind(*request.args, **request.kwargs)
    bound.apply_defaults()
    wat_in = bound.arguments

    # The ENERGY exit column as WATER receives it, before WATER mutates it.
    landed = {name: _vec(getattr(col, name))
              for name in ("stc", "snice", "snliq", "sh2o", "dzsnso", "sice")}
    landed["snowh"] = _f32(col.snowh)
    landed["sneqv"] = _f32(col.sneqv)
    landed["smc"] = _vec(wat_in["smc"])
    qvap, qdew = _f32(wat_in["qvap"]), _f32(wat_in["qdew"])

    errors = []
    with _recording_error(errors):
        answer = answer_on_host(request)
        # The column at the ERROR call: after WATER, before the snow clamp.
        after_water = {"sneqv": _f32(col.sneqv), "snowh": _f32(col.snowh),
                       "dzsnso": _vec(col.dzsnso)}
        try:
            steps.send(answer)
        except StopIteration as stop:
            result = stop.value
        else:                                       # pragma: no cover - guard
            raise AssertionError("sflx_post_steps yielded a second leaf")
    assert len(errors) == 1, f"{case}: ERROR ran {len(errors)} times"
    err = errors[0]

    seg1_in = {
        "stc": _vec(energy.stc), "snice": _vec(energy.snice),
        "snliq": _vec(energy.snliq), "sh2o": _vec(energy.sh2o),
        "smc": _vec(energy.smc), "dzsnso": _vec(pre.dzsnso),
        "snowh": _f32(energy.snowh), "sneqv": _f32(energy.sneqv),
        "fgev": _f32(energy.fgev), "latheag": _f32(energy.latheag),
    }
    seg1_want = dict(landed)
    seg1_want.update({"sneqvo": _f32(result.sneqvo), "qvap": qvap,
                      "qdew": qdew, "edir": _f32(result.edir)})

    seg2_in = {
        # ENERGY
        "fsa": _f32(energy.fsa), "fsr": _f32(energy.fsr),
        "fira": _f32(energy.fira), "fsh": _f32(energy.fsh),
        "fcev": _f32(energy.fcev), "fgev": _f32(energy.fgev),
        "fctr": _f32(energy.fctr), "ssoil": _f32(energy.ssoil),
        "sav": _f32(energy.sav), "sag": _f32(energy.sag),
        "pah": _f32(energy.pah), "canhs": _f32(energy.canhs),
        "qsfc": _f32(energy.qsfc), "q2b": _f32(energy.q2b),
        "ch": _f32(energy.ch),
        # the prefix
        "swdown": _f32(pre.swdown), "beg_wb": _f32(pre.beg_wb),
        "prcp": _f32(pre.prcp), "rhoair": _f32(pre.rhoair),
        "qair": _f32(pre.qair),
        # WATER's answer
        "canliq": _f32(answer.canliq), "canice": _f32(answer.canice),
        "ecan": _f32(answer.ecan), "etran": _f32(answer.etran),
        "runsrf": _f32(answer.runsrf), "runsub": _f32(answer.runsub),
        "qtldrn": _f32(answer.qtldrn), "smc": _vec(answer.smc),
        # the column at the ERROR call
        "sneqv": after_water["sneqv"], "snowh": after_water["snowh"],
        "dzsnso": after_water["dzsnso"],
        # the rest of ERROR's argument list, and the urban gate
        "wa": _f32(x[("wa", 0)]), "dt": _f32(x[("dt", 0)]),
        "ist": int(x[("ist", 0)]),
        "acc_dwater": _f32(0.0), "acc_prcp": _f32(0.0),
        "acc_ecan": _f32(0.0), "acc_etran": _f32(0.0),
        "acc_edir": _f32(0.0),
        "urban_flag": bool(parameters.urban_flag),
    }
    seg2_want = {
        "nee": _f32(result.nee), "gpp": _f32(result.gpp),
        "npp": _f32(result.npp), "prcp": _f32(result.prcp),
        "fsh": _f32(result.fsh), "errwat": _f32(result.errwat),
        "end_wb": _f32(result.end_wb), "errsw": _f32(err.errsw),
        "erreng": _f32(err.erreng),
        "acc_dwater": _f32(result.acc_dwater),
        "acc_prcp": _f32(result.acc_prcp), "acc_ecan": _f32(result.acc_ecan),
        "acc_etran": _f32(result.acc_etran),
        "acc_edir": _f32(result.acc_edir),
        "qsfc": _f32(result.qsfc), "q2b": _f32(result.q2b),
        # the column's own exit, after the clamp at :1067-1069
        "snowh": _f32(col.snowh), "sneqv": _f32(col.sneqv),
        "albedo": _f32(result.albedo),
    }
    return {"case": case, "seg1_in": seg1_in, "seg1_want": seg1_want,
            "seg2_in": seg2_in, "seg2_want": seg2_want,
            "result": result, "answer": answer}


def _columns(urban=False):
    """The four fixture cases as four land columns of one slab."""
    return [_capture(case, urban=urban) for case in T.CASES]


# ---------------------------------------------------------------------------
# the slab
# ---------------------------------------------------------------------------

#: Inputs the slab reads as something other than float32.
_INT_INPUTS = frozenset({"ist"})
_BOOL_INPUTS = frozenset({"urban_flag"})

#: The WATER-derived seg2 inputs, i.e. everything a misrouted WATER answer
#: would carry from the wrong column.  Named here so the control below rolls
#: exactly the seam and nothing else.
_FROM_WATER = ("canliq", "canice", "ecan", "etran", "runsrf", "runsub",
               "qtldrn", "smc", "sneqv", "snowh", "dzsnso")


def _slab(columns, part):
    """One CuPy array per field of ``part``, column axis first."""
    import cupy as cp

    fields = {}
    for name in columns[0][part]:
        values = [column[part][name] for column in columns]
        if name in _INT_INPUTS:
            packed = np.array([int(v) for v in values], dtype=np.int32)
        elif name in _BOOL_INPUTS:
            packed = np.array([bool(v) for v in values], dtype=np.bool_)
        else:
            packed = np.array([np.asarray(v, dtype=np.float32)
                               for v in values], dtype=np.float32)
        fields[name] = cp.asarray(packed)
    return fields


def _run_segments(columns, seg1_over=None, seg2_over=None):
    """seg1, then seg2 with seg1's own EDIR chained into it."""
    from gpuwm.core.noahmp_sflx_post_slab import (sflx_post_seg1,
                                                  sflx_post_seg2)

    state1 = _slab(columns, "seg1_in")
    state1.update(seg1_over or {})
    seg1 = sflx_post_seg1(state1)

    state2 = _slab(columns, "seg2_in")
    state2["edir"] = seg1["edir"]
    state2.update(seg2_over or {})
    seg2 = sflx_post_seg2(state2)
    return seg1, seg2


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------

def _bits(value) -> bytes:
    """A value's exact machine representation.

    Bytes, not ``==``: this comparison has to be bitwise, and ``==`` calls
    ``-0.0`` and ``+0.0`` equal -- which is the one difference a vectorised
    ``minimum`` introduces at :983 and the one this file exists to catch.
    """
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32)).tobytes()


def _differences(columns, got, part, names):
    """``(words_compared, [(field, case, got_bits, want_bits), ...])``."""
    import cupy as cp

    missing = [name for name in names if name not in got]
    assert not missing, f"the slab bundle does not carry {missing}"

    compared, bad = 0, []
    for name in names:
        device = cp.asnumpy(got[name])
        assert len(device) == len(columns), (name, len(device))
        for index, column in enumerate(columns):
            want = np.asarray(column[part][name], dtype=np.float32)
            have = np.asarray(device[index], dtype=np.float32)
            assert have.shape == want.shape, (name, have.shape, want.shape)
            compared += int(want.size)
            if _bits(have) != _bits(want):
                bad.append((name, column["case"], _bits(have), _bits(want)))
    return compared, bad


def _seg_names():
    from gpuwm.core.noahmp_sflx_post_slab import (SEG1_COMPUTED, SEG1_LANDED,
                                                  SEG2_FIELDS)

    return SEG1_LANDED + SEG1_COMPUTED, SEG2_FIELDS


#: 8 landed arrays (7 + 3 + 3 + 4 + 4 + 7 + 1 + 1 = 30 words) and 5 computed
#: (4 + 1 + 1 + 1 + 1 = 8 words) per column for seg1; 19 scalars for seg2.
#: Spelled out rather than bounded, so a field that stops being compared fails.
SEG1_WORDS_PER_COLUMN = 38
SEG2_WORDS_PER_COLUMN = 19


def _compare_all(columns, seg1, seg2):
    seg1_names, seg2_names = _seg_names()
    n1, bad1 = _differences(columns, seg1, "seg1_want", seg1_names)
    n2, bad2 = _differences(columns, seg2, "seg2_want", seg2_names)
    return n1 + n2, bad1 + bad2


# ---------------------------------------------------------------------------
# 0. the inventory, and the columns
# ---------------------------------------------------------------------------

def test_the_compared_field_set_is_every_field_the_segments_own():
    """A shrinking comparison must fail here before it can pass silently."""
    seg1_names, seg2_names = _seg_names()
    assert len(seg1_names) == 13, seg1_names
    assert len(seg2_names) == 19, seg2_names
    assert len(set(seg1_names)) == len(seg1_names)
    assert len(set(seg2_names)) == len(seg2_names)
    assert len(T.CASES) == 4, T.CASES

    # Every SflxResult field the postfix computes rather than passes through is
    # compared.  ``errsw`` and ``erreng`` are ERROR's residuals, which
    # SflxResult does not carry at all; the rest of this list is exactly the
    # intersection with it.
    from dataclasses import fields as dc_fields
    from gpuwm.core.noahmp_sflx import SflxResult

    declared = {f.name for f in dc_fields(SflxResult)}
    computed = {"sneqvo", "qvap", "qdew", "edir", "smc",
                "nee", "gpp", "npp", "prcp", "fsh", "errwat", "end_wb",
                "acc_dwater", "acc_prcp", "acc_ecan", "acc_etran", "acc_edir",
                "qsfc", "q2b", "albedo"}
    assert computed <= declared, sorted(computed - declared)
    assert computed <= set(seg1_names) | set(seg2_names), \
        sorted(computed - (set(seg1_names) | set(seg2_names)))


def test_the_slab_columns_are_heterogeneous():
    """The gate is only as good as the branches its columns reach."""
    columns = _columns()
    swdown = [float(c["seg2_in"]["swdown"]) for c in columns]
    sneqv = [float(c["seg2_in"]["sneqv"]) for c in columns]
    assert any(v == 0.0 for v in swdown), "no night column: :1075 unreached"
    assert any(v > 0.0 for v in swdown), "no daylit column: :1073 unreached"
    assert any(v > 0.0 for v in sneqv), "no snow pack"
    assert any(float(c["seg2_want"]["sneqv"]) == 0.0 for c in columns), \
        "the snow clamp at :1067-1069 never fires"
    assert any(float(c["seg1_want"]["qdew"]) > 0.0
               or float(c["seg1_want"]["qvap"]) > 0.0 for c in columns)
    assert all(int(c["seg2_in"]["ist"]) == 1 for c in columns), \
        "a lake column reached the slab; ERROR's :1733 branch is untested here"


# ---------------------------------------------------------------------------
# 1. the gate
# ---------------------------------------------------------------------------

@requires_gpu
def test_seg1_and_seg2_reproduce_every_field_they_own_bitwise():
    columns = _columns()
    seg1, seg2 = _run_segments(columns)
    compared, bad = _compare_all(columns, seg1, seg2)

    expected = len(columns) * (SEG1_WORDS_PER_COLUMN + SEG2_WORDS_PER_COLUMN)
    assert compared == expected, f"{compared} words compared, not {expected}"
    assert not bad, [f"{name} [{case}]: slab {have.hex()} scalar {want.hex()}"
                     for name, case, have, want in bad[:12]]


@requires_gpu
def test_the_urban_override_is_reached_and_reproduced():
    """:1061-1064 is the one postfix branch no fixture column takes.

    ``URBAN_FLAG`` is a land-use property, not an option, so the branch is live
    -- and with every fixture column non-urban, QSFC and Q2B would otherwise be
    pass-throughs of ENERGY's values and a slab that dropped the override
    entirely would pass the gate above.  The flag is flipped after the prefix
    on both sides at once; what is compared is still the scalar routine's own
    answer.
    """
    columns = _columns(urban=True)
    seg1, seg2 = _run_segments(columns)
    compared, bad = _compare_all(columns, seg1, seg2)
    assert compared == len(columns) * (SEG1_WORDS_PER_COLUMN
                                       + SEG2_WORDS_PER_COLUMN)
    assert not bad, [f"{name} [{case}]" for name, case, _h, _w in bad[:12]]

    # And the branch really did change the answer, or this test is vacuous.
    plain = _columns()
    moved = [c for c, u in zip(plain, columns)
             if _bits(c["seg2_want"]["qsfc"]) != _bits(u["seg2_want"]["qsfc"])]
    assert moved, "URBAN_FLAG moved no QSFC; :1063 was never exercised"


@requires_gpu
def test_a_missing_state_field_is_named():
    columns = _columns()
    state = _slab(columns, "seg1_in")
    state.pop("latheag")
    from gpuwm.core.noahmp_sflx_post_slab import sflx_post_seg1
    with pytest.raises(KeyError, match="latheag"):
        sflx_post_seg1(state)


@requires_gpu
def test_a_layer_axis_the_wrong_way_round_is_refused():
    """``(NFULL, n)`` is not ``(n, NFULL)``, and the slab must say so.

    ``stc`` and not ``sh2o``: with four fixture columns and four soil layers a
    transposed ``(n, NSOIL)`` slab is square, so it would sail through a shape
    check and be read layer-major.  That is the shape a width check cannot
    catch and the reason the layer-axis convention is stated in the contract.
    """
    columns = _columns()
    state = _slab(columns, "seg1_in")
    assert state["sh2o"].shape[0] == state["sh2o"].shape[1], (
        "the soil slab is no longer square; say so here rather than leaving "
        "this test's reason for choosing STC unexplained")
    state["stc"] = state["stc"].T
    from gpuwm.core.noahmp_sflx_post_slab import sflx_post_seg1
    with pytest.raises(ValueError, match="layer axis last"):
        sflx_post_seg1(state)


@requires_gpu
def test_an_empty_tile_is_an_empty_bundle():
    """A tile with no land columns launches nothing and still answers.

    A grid of zero blocks is a CUDA configuration error, so the empty slab is
    the one shape that can crash a launcher which never tries it -- and a tile
    with no land is ordinary, not exotic.
    """
    import cupy as cp
    from gpuwm.core.noahmp_sflx_post_slab import (SEG1_COMPUTED, SEG1_LANDED,
                                                  SEG2_FIELDS, sflx_post_seg1,
                                                  sflx_post_seg2)

    columns = _columns()
    empty = {}
    for part in ("seg1_in", "seg2_in"):
        for name, value in _slab(columns, part).items():
            empty[name] = value[:0]
    seg1 = sflx_post_seg1(empty)
    assert set(seg1) == set(SEG1_LANDED) | set(SEG1_COMPUTED)
    empty["edir"] = seg1["edir"]
    seg2 = sflx_post_seg2(empty)
    assert set(seg2) == set(SEG2_FIELDS)
    assert all(value.shape[0] == 0 for value in seg1.values())
    assert all(value.shape[0] == 0 for value in seg2.values())
    assert cp.asnumpy(seg2["albedo"]).size == 0


def test_the_error_packing_is_read_off_the_kernel():
    """The slab does not carry its own copy of ERROR's 34-slot layout.

    ``_slot_table`` parses the ``SC_``/``OU_`` blocks of ``noahmp_sflx.cu``, so
    the launch below cannot pack a stale layout; what this asserts is that the
    parse found the blocks and that they are the contiguous tables the kernel
    declares, since a silently empty parse would fill nothing and launch zeros.
    """
    from gpuwm.core.noahmp_sflx_post_slab import (error_input_slots,
                                                  error_output_slots)

    sc, ou = error_input_slots(), error_output_slots()
    assert len(sc) == 34, sorted(sc)
    assert len(ou) == 9, sorted(ou)
    assert sorted(sc.values()) == list(range(34))
    assert sorted(ou.values()) == list(range(9))
    assert set(ou) == {"errwat", "acc_dwater", "acc_prcp", "acc_ecan",
                       "acc_etran", "acc_edir", "errsw", "erreng", "end_wb"}
    # Every ERROR input the postfix supplies must be a slot the kernel reads.
    for name in ("swdown", "fsa", "fsr", "beg_wb", "edir", "qtldrn", "canhs"):
        assert name in sc, name


# ---------------------------------------------------------------------------
# 2. the same comparison, shown failing
# ---------------------------------------------------------------------------

def _rolled(columns, part, name):
    import cupy as cp

    state = _slab(columns, part)
    return {name: cp.roll(state[name], 1, axis=0)}


@requires_gpu
@pytest.mark.parametrize("name", ["fgev", "sh2o", "smc"])
def test_rolling_one_seg1_input_onto_its_neighbour_is_rejected(name):
    """One field one column out of step must be seen.

    ``fgev`` drives :981-984 through LATHEAG, ``sh2o`` and ``smc`` are the two
    layer vectors :979 subtracts and both are also landed on the column WATER
    is handed, so the three between them cover every shape seg1 touches.
    """
    columns = _columns()
    seg1, _seg2 = _run_segments(columns,
                                seg1_over=_rolled(columns, "seg1_in", name))
    seg1_names, _ = _seg_names()
    _n, bad = _differences(columns, seg1, "seg1_want", seg1_names)
    assert bad, (
        f"rolling {name} one column onto its neighbour left every field seg1 "
        "owns bitwise identical on all four columns")


@requires_gpu
@pytest.mark.parametrize("name", ["swdown", "beg_wb", "canliq"])
def test_rolling_one_seg2_input_onto_its_neighbour_is_rejected(name):
    """The same for seg2, where an abort is the likelier rejection.

    ``swdown`` reaches both ERRSW and the albedo, ``beg_wb`` reaches ERRWAT
    only and ``canliq`` reaches END_WB.  A rolled column's water and energy no
    longer belong to it, which is exactly what WRF's three ``wrf_error_fatal``
    gates exist to catch, so an exception counts as an observation -- and it is
    a stronger one than a differing word.

    Measured: all three roll onto a gate rather than onto a differing word --
    ``swdown`` on the shortwave gate, the other two on the water gate.  The
    exception is required to name the offending column, because that is the
    slab's own reporting path and nothing else in this file reaches it.
    """
    columns = _columns()
    _, seg2_names = _seg_names()
    try:
        _seg1, seg2 = _run_segments(
            columns, seg2_over=_rolled(columns, "seg2_in", name))
    except NoahmpBalanceError as raised:
        assert "land column" in str(raised), (
            "the slab aborted without naming the column; a slab cannot raise "
            f"for one column and continue for the rest: {raised}")
        return
    _n, bad = _differences(columns, seg2, "seg2_want", seg2_names)
    assert bad, (
        f"rolling {name} one column onto its neighbour neither tripped a "
        "balance gate nor moved any field seg2 owns")


@requires_gpu
def test_misrouting_every_water_answer_onto_its_neighbour_is_rejected():
    """The defect a batched seam actually produces.

    Every column gets an answer of the right shape carrying its neighbour's
    water: the eleven values WATER returns or leaves on the column, rolled
    together so the misrouting is self-consistent rather than a torn mixture.
    That is what a scheduler that paired the wrong result with the wrong column
    would hand seg2.
    """
    import cupy as cp

    columns = _columns()
    state = _slab(columns, "seg2_in")
    rolled = {name: cp.roll(state[name], 1, axis=0) for name in _FROM_WATER}
    _, seg2_names = _seg_names()
    try:
        _seg1, seg2 = _run_segments(columns, seg2_over=rolled)
    except NoahmpBalanceError as raised:
        # Measured: the water gate, at 13 kg/m2 of water appearing from
        # nowhere on column 0.  Same rejection the scalar staged path gets.
        assert "land column" in str(raised), raised
        return
    _n, bad = _differences(columns, seg2, "seg2_want", seg2_names)
    assert bad, (
        "rolling every column's WATER answer onto its neighbour neither "
        "tripped a balance gate nor moved any field seg2 owns")


def _transposed(columns, part, first, second):
    state = _slab(columns, part)
    assert state[first].shape == state[second].shape, (first, second)
    assert state[first].dtype == state[second].dtype, (first, second)
    return {first: state[second], second: state[first]}


@requires_gpu
def test_transposing_two_seg1_inputs_is_rejected():
    """SMC and SH2O, which seg1 reads separately and in a fixed order.

    :979 is ``MAX(0.0, SMC - SH2O)`` and subtraction does not commute, so the
    swap is observable in SICE; both are also landed on the column WATER is
    handed, under different names, so it is observable twice.  That second
    route is the point -- the landing is arithmetic-free and a bitwise check of
    the arithmetic alone could not see two same-shaped arrays exchanged.
    """
    columns = _columns()
    seg1, _seg2 = _run_segments(
        columns, seg1_over=_transposed(columns, "seg1_in", "smc", "sh2o"))
    seg1_names, _ = _seg_names()
    _n, bad = _differences(columns, seg1, "seg1_want", seg1_names)
    assert bad, "transposing SMC and SH2O left every field seg1 owns unchanged"


@requires_gpu
def test_transposing_two_seg2_inputs_is_rejected():
    """ECAN and ETRAN, which ERROR accumulates into two different outputs.

    :1704 and :1705 are ``ACC_ECAN += ECAN*DT`` and ``ACC_ETRAN += ETRAN*DT``,
    two separate reads into two separate outputs, so the swap is observable
    without going anywhere near a balance gate: the pair enters ERRWAT as a
    difference of both accumulators and the residual barely moves.
    """
    columns = _columns()
    _seg1, seg2 = _run_segments(
        columns, seg2_over=_transposed(columns, "seg2_in", "ecan", "etran"))
    _, seg2_names = _seg_names()
    _n, bad = _differences(columns, seg2, "seg2_want", seg2_names)
    assert bad, "transposing ECAN and ETRAN left every field seg2 owns unchanged"
    assert {name for name, *_ in bad} & {"acc_ecan", "acc_etran"}, \
        f"the swap moved {sorted({n for n, *_ in bad})}, not the accumulators"


@requires_gpu
def test_the_clamp_at_982_is_exercised_and_the_dew_leg_is_not():
    """What the fixture reaches in :982-983, measured rather than assumed.

    Every fixture column has ``FGEV > 0``, so RATIO is positive, ``QVAP`` takes
    MAX's *first* argument and ``QDEW`` is identically zero on all four.  That
    means :983's MIN and its ``ABS`` are exercised only on their zero leg and
    a fixture column cannot distinguish ``fmn`` from ``cupy.minimum`` there;
    the tie-correctness of both is gated in ``tests/test_noahmp_slab_libm.py``,
    against the scalar spellings, and not here.  Recording it makes the gap a
    measurement instead of a silence -- a future dewing column makes this test
    fail and the note stops being true.

    The MAX itself is *not* in that gap, and this proves it: dropping the clamp
    to "always the second argument" must move the answer, or seg1 would not be
    exercising :982 at all.
    """
    import cupy as cp
    import gpuwm.core.noahmp_sflx_post_slab as slab

    columns = _columns()
    for column in columns:
        assert float(column["seg2_in"]["fgev"]) > 0.0, column["case"]
        assert _bits(column["seg1_want"]["qdew"]) == _bits(np.float32(0.0)), \
            f"{column['case']} dews; the note above is stale"

    seg1, _seg2 = _run_segments(columns)
    reference = {name: cp.asnumpy(seg1[name]) for name in ("qvap", "edir")}

    original = slab.fmx
    slab.fmx = lambda a, b: cp.asarray(b, dtype=cp.float32)
    try:
        broken, _ = _run_segments(columns)
        moved = any(_bits(cp.asnumpy(broken[name])) != _bits(reference[name])
                    for name in reference)
    finally:
        slab.fmx = original
    assert moved, (
        "dropping the MAX at :982 moved neither QVAP nor EDIR on any fixture "
        "column, so this file does not exercise the clamp")


@requires_gpu
def test_the_sav_sag_transposition_really_is_unobservable():
    """A control that does not fire, kept as a measurement rather than deleted.

    The postfix reads SAV and SAG in exactly one place -- ERRENG's
    ``SAV + SAG`` at :1662 -- and FP32 addition is commutative, so transposing
    them cannot move anything and is not a transposition control at all.  It is
    kept because the claim is checkable: if the postfix ever starts reading the
    two separately, this test fails and the note in the control beside it stops
    being true.  Deleting it would leave that reasoning as a comment nothing
    holds.
    """
    columns = _columns()
    _seg1, reference = _run_segments(columns)
    _seg1b, swapped = _run_segments(
        columns, seg2_over=_transposed(columns, "seg2_in", "sav", "sag"))

    _, seg2_names = _seg_names()
    _n, bad = _differences(columns, swapped, "seg2_want", seg2_names)
    assert not bad, (
        "SAV and SAG are now read separately somewhere seg2 reaches; "
        f"the transposition moved {sorted({name for name, *_ in bad})}")

    import cupy as cp
    for name in seg2_names:
        np.testing.assert_array_equal(
            cp.asnumpy(reference[name]), cp.asnumpy(swapped[name]),
            err_msg=f"seg2 {name} moved under a commutative transposition")
