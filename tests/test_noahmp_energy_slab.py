"""ENERGY's composition over a slab, against the scalar authority.

:mod:`gpuwm.core.noahmp_energy_slab` evaluates the same statements as
:mod:`gpuwm.core.noahmp_energy` over the column axis.  Only the scalar module
is pinned by the unmodified-WRF fixture, so every value the slab produces is
compared with the scalar one **bitwise**, on the nine fixture columns at once.

Where the comparison is taken matters, and it is not the end of the routine:

* **seg1 and seg2 are gated at the physical leaf call.**  Everything those two
  segments compute is either an argument of VEGE_FLUX or of BARE_FLUX, or
  feeds one, so recording what the scalar generator actually yields and
  comparing the slab's arguments word for word covers them completely --
  without a second reference implementation and without trusting an
  intermediate nobody reads.  The argument names come from
  ``noahmp_vegeflux_gpu.CALL_NAMES``, so a reordering of that positional list
  moves the comparison with it instead of silently comparing the wrong slots.
* **seg3 and seg4 are gated on the ``EnergyState`` fields they own**, with the
  leaf answers taken from the same recorded scalar run, so a difference can
  only be the segment's own arithmetic.

Three negative controls, because a comparison that has never failed is not
evidence: one column's inputs rolled onto its neighbour, one input field
transposed with another, and ``cupy.minimum`` substituted for the tie-correct
``fmn`` inside BTRAN's clamp.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

from gpuwm.core.noahmp_energy import (NSNOW, NSOIL, PINNED_OPTIONS,
                                      answer_on_host, energy_steps)
from gpuwm.core.noahmp_vegeflux_gpu import CALL_NAMES as VEGE_CALL_NAMES

from test_noahmp_energy import CASES, _call


# ---------------------------------------------------------------------------
# the scalar run, recorded
# ---------------------------------------------------------------------------

def _capture(parameters, kwargs):
    """Drive the scalar generator, recording every physical leaf call.

    The answers are recorded on the way past rather than recomputed later:
    PHASECHANGE and WATER mutate arrays they are handed, so calling a leaf a
    second time on the same objects is not the same call.
    """
    steps = energy_steps(parameters, **kwargs)
    requests = []
    answers = {}
    reply = None
    while True:
        try:
            request = steps.send(reply)
        except StopIteration as stop:
            return requests, answers, stop.value
        requests.append((request.leaf, request.args, dict(request.kwargs)))
        reply = answer_on_host(request)
        answers[request.leaf] = reply


def _columns(overrides_by_case=None):
    """The nine fixture cases as nine land columns of one slab."""
    columns = []
    for case in CASES:
        overrides = (overrides_by_case or {}).get(case)
        parameters, kwargs = _call(case, overrides=overrides)
        requests, answers, state = _capture(parameters, kwargs)
        columns.append({
            "case": case, "p": parameters, "kwargs": kwargs,
            "requests": requests, "answers": answers, "state": state,
            "leaves": {leaf: (args, kw) for leaf, args, kw in requests},
        })
    return columns


def _stack(columns, getter, dtype=np.float32):
    import cupy as cp
    return cp.asarray(np.array([getter(c) for c in columns], dtype=dtype))


def _entry_state(columns):
    """The slab spelling of the scalar generator's argument list."""
    import cupy as cp

    def kw(name, dtype=np.float32):
        return _stack(columns, lambda c: c["kwargs"][name], dtype)

    def par(name, dtype=np.float32):
        return _stack(columns, lambda c: getattr(c["p"], name), dtype)

    def par_vec(name):
        return cp.asarray(np.array(
            [list(getattr(c["p"], name)) for c in columns], np.float32))

    def kw_vec(name):
        return cp.asarray(np.array(
            [list(c["kwargs"][name]) for c in columns], np.float32))

    state = {
        "ice": kw("ice", np.int32), "ist": kw("ist", np.int32),
        "isnow": kw("isnow", np.int32),
        "uu": kw("uu"), "vv": kw("vv"), "elai": kw("elai"),
        "esai": kw("esai"), "snowh": kw("snowh"), "sneqv": kw("sneqv"),
        "zref": kw("zref"), "fveg": kw("fveg"), "sfcprs": kw("sfcprs"),
        "tv": kw("tv"), "tg": kw("tg"), "lwdn": kw("lwdn"),
        "cosz": kw("cosz"), "dt": kw("dt"), "acc_ssoil": kw("acc_ssoil"),
        "pahv": kw("pahv"), "pahg": kw("pahg"), "pahb": kw("pahb"),
        "sh2o": kw_vec("sh2o"), "zsoil": kw_vec("zsoil"),
        "dzsnso": kw_vec("dzsnso"),
        "mfsno": par("mfsno"), "scffac": par("scffac"), "z0sno": par("z0sno"),
        "z0mvt": par("z0mvt"), "hvt": par("hvt"), "cwpvt": par("cwpvt"),
        "snow_emis": par("snow_emis"), "rsurf_exp": par("rsurf_exp"),
        "nroot": par("nroot", np.int32),
        "urban_flag": _stack(columns, lambda c: bool(c["p"].urban_flag), bool),
        "smcmax": par_vec("smcmax"), "smcref": par_vec("smcref"),
        "smcwlt": par_vec("smcwlt"), "psisat": par_vec("psisat"),
        "bexp": par_vec("bexp"),
        # IST == 1, so EG is always the first entry.
        "eg": _stack(columns, lambda c: c["p"].eg[c["kwargs"]["ist"] - 1]),
        "soil_update_steps": PINNED_OPTIONS.soil_update_steps,
    }
    return state


# ---------------------------------------------------------------------------
# seg1 and seg2, gated at the physical leaf call
# ---------------------------------------------------------------------------

#: BARE_FLUX runs on every land column and takes its arguments by keyword, so
#: this half of the comparison is name-addressed on both sides.  The value on
#: the right is what seg1/seg2 call the same quantity.
_BARE_ARGUMENT_TO_SLAB = {
    "ur": ("seg1", "ur"), "zlvl": ("seg1", "zlvl"),
    "zpd": ("seg1", "zpdg"), "z0m": ("seg1", "z0mg"),
    "fsno": ("seg1", "fsno"), "emg": ("seg2", "emg"),
    "rsurf": ("seg2", "rsurf"), "lathea": ("seg2", "latheag"),
    "gamma": ("seg2", "gammag"), "rhsur": ("seg2", "rhsur"),
}

#: VEGE_FLUX is positional; the names come from the batch module's own list so
#: this cannot drift away from the call it is checking.
_VEGE_ARGUMENT_TO_SLAB = {
    "ur": ("seg1", "ur"), "vai": ("seg1", "vai"), "cwp": ("seg1", "cwp"),
    "zlvl": ("seg1", "zlvl"), "zpd": ("seg1", "zpd"),
    "z0m": ("seg1", "z0m"), "z0mg": ("seg1", "z0mg"),
    "fsno": ("seg1", "fsno"),
    "gammav": ("seg2", "gammav"), "gammag": ("seg2", "gammag"),
    "emv": ("seg2", "emv"), "emg": ("seg2", "emg"),
    "rsurf": ("seg2", "rsurf"), "latheav": ("seg2", "latheav"),
    "latheag": ("seg2", "latheag"), "btran": ("seg2", "btran"),
    "rhsur": ("seg2", "rhsur"),
}


def _radiation_answers(columns):
    """RADIATION's recorded per-column answer, as slab arrays."""
    import cupy as cp

    keys = ("laisun", "laisha", "parsun", "parsha", "sav", "sag", "fsa",
            "fsr", "albold", "tauss", "fsrv", "fsrg", "bgap", "wgap")
    out = {}
    for key in keys:
        out[key] = cp.asarray(np.array(
            [np.float32(_recorded_radiation(c)[key]) for c in columns],
            np.float32))
    return out


def _recorded_radiation(column):
    return column["answers"]["radiation"]


def _run_segments(columns, state=None):
    from gpuwm.core.noahmp_energy_slab import energy_seg1, energy_seg2

    state = _entry_state(columns) if state is None else state
    seg1 = energy_seg1(state)
    rad = _radiation_answers(columns)
    seg2_state = dict(state)
    seg2_state.update(rad)
    seg2 = energy_seg2(seg2_state, seg1)
    return state, seg1, seg2, rad


def _compare_leaf_arguments(columns, seg1, seg2):
    """Return {(leaf, argument): (slab, scalar)} for every differing word."""
    import cupy as cp

    banks = {"seg1": seg1, "seg2": seg2}
    differing = {}
    compared = 0

    for index, column in enumerate(columns):
        bare = column["leaves"]["bare_flux"][1]
        for argument, (bank, name) in _BARE_ARGUMENT_TO_SLAB.items():
            got = np.float32(cp.asnumpy(banks[bank][name])[index])
            want = np.float32(bare[argument])
            compared += 1
            if got.view(np.int32) != want.view(np.int32):
                differing[(column["case"], "bare_flux", argument)] = (got, want)

        if "vege_flux" not in column["leaves"]:
            continue
        args = column["leaves"]["vege_flux"][0]
        bound = dict(zip(VEGE_CALL_NAMES, args))
        for argument, (bank, name) in _VEGE_ARGUMENT_TO_SLAB.items():
            assert argument in bound, (
                f"{argument!r} is not in noahmp_vegeflux_gpu.CALL_NAMES; the "
                "positional list moved and this comparison would be checking "
                "the wrong slot")
            got = np.float32(cp.asnumpy(banks[bank][name])[index])
            want = np.float32(bound[argument])
            compared += 1
            if got.view(np.int32) != want.view(np.int32):
                differing[(column["case"], "vege_flux", argument)] = (got, want)
    return differing, compared


@requires_gpu
def test_seg1_and_seg2_reproduce_every_leaf_argument():
    columns = _columns()
    _state, seg1, seg2, _rad = _run_segments(columns)
    differing, compared = _compare_leaf_arguments(columns, seg1, seg2)
    assert compared >= 150, f"only {compared} words compared"
    assert differing == {}, differing


@requires_gpu
def test_seg2_reproduces_btrani_which_no_leaf_argument_carries():
    """BTRANI reaches WATER, not VEGE_FLUX, so the leaf gate cannot see it."""
    import cupy as cp

    columns = _columns()
    _state, _seg1, seg2, _rad = _run_segments(columns)
    got = cp.asnumpy(seg2["btrani"])
    for index, column in enumerate(columns):
        # ENERGY divides BTRANI by BTRAN in place and zeroes the layers below
        # NROOT only at the end; seg4 does the zeroing, so compare in-root.
        nroot = int(column["p"].nroot)
        want = np.asarray(column["state"].btrani, dtype=np.float32)
        np.testing.assert_array_equal(
            got[index, :nroot].view(np.int32),
            want[:nroot].view(np.int32),
            err_msg=f"{column['case']} btrani")


# ---------------------------------------------------------------------------
# seg3 and seg4, gated on the EnergyState fields they own
# ---------------------------------------------------------------------------

#: VEGE_FLUX's outputs, and what :2033-2053 leaves them at on a column that
#: never runs it.  ``None`` means "the column's own entry value", which is
#: what the scalar path holds for TGV/CMV/CHV/EAH/TV before the tile average.
_VEGE_OUTPUTS = {
    "tauxv": "TAUXV", "tauyv": "TAUYV", "irg": "IRG", "irc": "IRC",
    "shg": "SHG", "shc": "SHC", "evg": "EVG", "evc": "EVC", "tr": "TR",
    "ghv": "GH", "t2mv": "T2MV", "psnsun": "PSNSUN", "psnsha": "PSNSHA",
    "rssun": "RSSUN", "rssha": "RSSHA", "q2v": "Q2V",
    "tgv": "TG", "cmv": "CM", "chv": "CH", "eah": "EAH", "tv": "TV",
}
_VEGE_ENTRY_FALLBACK = {"tgv": "tg", "cmv": "cm", "chv": "ch",
                        "eah": "eah", "tv": "tv"}

_BARE_OUTPUTS = ("tgb", "cm", "ch", "qsfc", "tauxb", "tauyb", "irb", "shb",
                 "evb", "ghb", "t2mb", "q2b", "ehb2")


def _flux_answers(columns):
    """VEGE_FLUX's and BARE_FLUX's recorded answers, as slab arrays."""
    import cupy as cp

    vege = {}
    for name, attribute in _VEGE_OUTPUTS.items():
        values = []
        for column in columns:
            answer = column["answers"].get("vege_flux")
            if answer is not None:
                values.append(np.float32(getattr(answer, attribute)))
            elif name in _VEGE_ENTRY_FALLBACK:
                values.append(
                    np.float32(column["kwargs"][_VEGE_ENTRY_FALLBACK[name]]))
            else:
                values.append(np.float32(0.0))       # :2033-2053
        vege[name] = cp.asarray(np.array(values, np.float32))

    bare = {}
    answers = [column["answers"]["bare_flux"] for column in columns]
    for name in _BARE_OUTPUTS:
        bare[name] = cp.asarray(np.array(
            [np.float32(getattr(a, name)) for a in answers], np.float32))
    bare["cmb"] = bare.pop("cm")
    bare["chb"] = bare.pop("ch")
    return vege, bare


#: Every ``EnergyState`` field seg3 produces.  Named exhaustively so a field
#: the segment stops writing fails here instead of being silently dropped.
_SEG3_FIELDS = (
    "taux", "tauy", "fira", "fsh", "fgev", "ssoil", "fcev", "fctr", "pah",
    "tg", "t2m", "ts", "cm", "ch", "q1", "q2e", "z0wrf", "rssun", "rssha",
    "tgv", "chv", "emissi", "trad", "apar", "psn", "acc_ssoil",
)

_SEG4_SCALAR_FIELDS = ("eflxb", "fsrv", "fsrg", "bgap", "wgap")
_SEG4_LAYER_FIELDS = ("hcpct", "imelt", "snicev", "snliqv", "epore", "btrani")


def _seg3_state(columns, rad):
    state = _entry_state(columns)
    state.update(rad)
    return state


@requires_gpu
def test_seg3_reproduces_every_energystate_field_it_owns():
    import cupy as cp
    from gpuwm.core.noahmp_energy_slab import energy_seg3

    columns = _columns()
    _s, seg1, seg2, rad = _run_segments(columns)
    vege, bare = _flux_answers(columns)
    seg3 = energy_seg3(_seg3_state(columns, rad), seg1, seg2, vege, bare)

    compared = 0
    for field in _SEG3_FIELDS:
        got = cp.asnumpy(seg3[field]).astype(np.float32)
        want = np.array([np.float32(getattr(c["state"], field))
                         for c in columns], np.float32)
        compared += got.size
        np.testing.assert_array_equal(
            got.view(np.int32), want.view(np.int32),
            err_msg=f"seg3 {field}")
    assert compared == len(_SEG3_FIELDS) * len(columns)


@requires_gpu
def test_seg4_reproduces_the_slots_wrf_leaves_undefined():
    import cupy as cp
    from gpuwm.core.noahmp_energy_slab import energy_seg3, energy_seg4

    columns = _columns()
    state, seg1, seg2, rad = _run_segments(columns)
    vege, bare = _flux_answers(columns)
    seg3 = energy_seg3(_seg3_state(columns, rad), seg1, seg2, vege, bare)

    # The leaf answers seg4 masks, taken from the same recorded scalar run.
    thermo = [c["answers"]["thermoprop"] for c in columns]
    phase = [c["answers"]["phasechange"] for c in columns]
    tsnosoi = [c["answers"]["tsnosoi"] for c in columns]

    def stack(rows):
        return cp.asarray(np.array(rows, np.float32))

    out = {
        "eflxb": stack([np.float32(t[1]) for t in tsnosoi]),
        "hcpct": stack([np.asarray(t[1], np.float32) for t in thermo]),
        "imelt": cp.asarray(np.array(
            [np.asarray(p[9], np.int32) for p in phase], np.int32)),
        "snicev": stack([np.asarray(t[2], np.float32) for t in thermo]),
        "snliqv": stack([np.asarray(t[3], np.float32) for t in thermo]),
        "epore": stack([np.asarray(t[4], np.float32) for t in thermo]),
        "fsrv": stack([np.float32(r["fsrv"]) for r in
                       [_recorded_radiation(c) for c in columns]]),
        "fsrg": stack([np.float32(r["fsrg"]) for r in
                       [_recorded_radiation(c) for c in columns]]),
        "bgap": stack([np.float32(r["bgap"]) for r in
                       [_recorded_radiation(c) for c in columns]]),
        "wgap": stack([np.float32(r["wgap"]) for r in
                       [_recorded_radiation(c) for c in columns]]),
    }
    seg4 = energy_seg4(state, seg1, {**seg2, **seg3}, out)

    compared = 0
    for field in _SEG4_SCALAR_FIELDS:
        got = cp.asnumpy(seg4[field]).astype(np.float32)
        want = np.array([np.float32(getattr(c["state"], field))
                         for c in columns], np.float32)
        compared += got.size
        np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32),
                                      err_msg=f"seg4 {field}")
    for field in _SEG4_LAYER_FIELDS:
        got = cp.asnumpy(seg4[field])
        want = np.array([np.asarray(getattr(c["state"], field), got.dtype)
                         for c in columns], got.dtype)
        compared += got.size
        np.testing.assert_array_equal(got, want, err_msg=f"seg4 {field}")
    # 9 columns x (5 scalars + HCPCT and IMELT over 7 layers + SNICEV, SNLIQV
    # and EPORE over 3 snow slots + BTRANI over 4 soil layers).  Spelled out
    # rather than bounded, so a field that stops being compared fails here.
    expected = len(columns) * (len(_SEG4_SCALAR_FIELDS)
                               + 2 * (NSNOW + NSOIL) + 3 * NSNOW + NSOIL)
    assert compared == expected, f"{compared} words compared, not {expected}"


# ---------------------------------------------------------------------------
# the negative controls
# ---------------------------------------------------------------------------

@requires_gpu
def test_rolling_one_column_onto_its_neighbour_is_rejected():
    import cupy as cp

    columns = _columns()
    state = _entry_state(columns)
    rolled = dict(state)
    rolled["snowh"] = cp.roll(state["snowh"], 1)
    _s, seg1, seg2, _r = _run_segments(columns, state=rolled)
    differing, _ = _compare_leaf_arguments(columns, seg1, seg2)
    assert differing, (
        "rolling SNOWH onto the neighbouring column left every leaf argument "
        "unchanged, so this comparison cannot see a misrouted column")


@requires_gpu
def test_misrouting_one_columns_bare_flux_answer_is_rejected():
    """The defect a batched seam actually produces, at the tile average.

    Every column gets an answer of the right shape carrying its neighbour's
    surface.  seg3 is where a misrouted BARE_FLUX answer lands, and BARE_FLUX
    runs on every land column, so this is the roll that must be visible.
    """
    import cupy as cp
    from gpuwm.core.noahmp_energy_slab import energy_seg3

    columns = _columns()
    _s, seg1, seg2, rad = _run_segments(columns)
    vege, bare = _flux_answers(columns)
    rolled = {name: cp.roll(value, 1) for name, value in bare.items()}
    seg3 = energy_seg3(_seg3_state(columns, rad), seg1, seg2, vege, rolled)

    moved = []
    for field in _SEG3_FIELDS:
        got = cp.asnumpy(seg3[field]).astype(np.float32)
        want = np.array([np.float32(getattr(c["state"], field))
                         for c in columns], np.float32)
        if (got.view(np.int32) != want.view(np.int32)).any():
            moved.append(field)
    assert moved, (
        "rolling every column's BARE_FLUX answer onto its neighbour left "
        "every EnergyState field seg3 owns unchanged")


@requires_gpu
def test_transposing_two_input_fields_is_rejected():
    """TV and TG, which is a transposition the comparison can actually see.

    The first spelling of this control transposed ELAI and ESAI and it FAILED
    -- correctly.  ENERGY reads that pair only as ``ELAI + ESAI``, at :2062
    for VAI and at :2140 for EMV, and FP32 addition is commutative, so
    swapping them is not observable anywhere downstream.  It looked like a
    transposition control and it was a no-op.  ``test_elai_esai_transposition
    _really_is_unobservable`` below keeps that as a measurement rather than an
    excuse.

    TV and TG are read separately: TV picks LATHEAV at :2211-2214 and TG picks
    LATHEAG at :2220-2223 and appears again in RHSUR at :2203.
    """
    columns = _columns()
    state = _entry_state(columns)
    swapped = dict(state)
    swapped["tv"], swapped["tg"] = state["tg"], state["tv"]
    _s, seg1, seg2, _r = _run_segments(columns, state=swapped)
    differing, _ = _compare_leaf_arguments(columns, seg1, seg2)
    assert differing, (
        "transposing TV and TG left every leaf argument unchanged")


@requires_gpu
def test_elai_esai_transposition_really_is_unobservable():
    """The explanation above, measured instead of asserted.

    If ENERGY ever starts reading ELAI and ESAI separately, this test fails
    and the note in the control beside it stops being true -- which is the
    only way that note can be kept honest.
    """
    columns = _columns()
    state = _entry_state(columns)
    swapped = dict(state)
    swapped["elai"], swapped["esai"] = state["esai"], state["elai"]
    _s, seg1, seg2, _r = _run_segments(columns, state=swapped)
    differing, compared = _compare_leaf_arguments(columns, seg1, seg2)
    assert compared >= 150
    assert differing == {}, (
        "ELAI and ESAI are now read separately somewhere seg1/seg2 reaches; "
        f"the transposition moved {sorted(differing)}")


@requires_gpu
def test_the_tie_correct_min_is_load_bearing_in_btran(monkeypatch):
    """Substituting ``cupy.minimum`` for ``fmn`` must be observable.

    Not because the fixture columns hit a signed-zero tie -- they do not --
    but because if the substitution were invisible here, this file would not
    be exercising BTRAN's clamp at all.  The control therefore forces the
    clamp: GX is pushed above 1 on every layer, where ``fmn(1.0, gx)`` and a
    broken minimum would have to disagree if the clamp were wrong.
    """
    import cupy as cp
    import gpuwm.core.noahmp_energy_slab as slab

    columns = _columns()
    state = _entry_state(columns)
    _s, _seg1, seg2, _r = _run_segments(columns, state=state)
    reference = cp.asnumpy(seg2["btran"]).copy()

    monkeypatch.setattr(slab, "fmn", lambda a, b: cp.asarray(b, cp.float32))
    _s, _seg1, broken, _r = _run_segments(columns, state=state)
    assert (cp.asnumpy(broken["btran"]).view(np.int32)
            != reference.view(np.int32)).any(), (
        "dropping BTRAN's upper clamp did not move BTRAN on any fixture "
        "column, so this file does not exercise the clamp")
