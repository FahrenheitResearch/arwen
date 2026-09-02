"""``cu_ntiedtke_run``:278-320 graded at ``max_ulp == 0`` -- the missing link.

``cumastrn`` leaves TENDENCIES -- ``ptte``, ``pqte``, ``pvom``, ``pvol`` --
and a detrained condensate rate ``pcte``. ``cu_ntiedtke_post_run``
differences updated STATE against reference state. This block is what turns
one into the other, so without it the chain from the last cumastrn stage to
the eight graded fields of ``nt-levels.csv`` has a hole in it, and the
assembler would have had nothing to put between them.

IT CANNOT BE CALLED. ``cu_ntiedtke_run`` is public but this block is inside
it, so unlike ``cu_ntiedtke_pre_run`` and ``cu_ntiedtke_post_run`` no
objcopy reaches it and a transcription is forced. That transcription is one
of the three ranges under ``run_nt_cumastrn.F90``'s single remaining
convergence argument (docs/ntiedtke/PORT-RECORD.md §29). Grading the mirror against a
capture taken at the block's OWN boundary is strictly stronger than the
convergence proof, and it is what this file does.

THE CAPTURE BOUNDARY MATTERS MORE HERE THAN USUAL. ``zqp1`` is updated in
place and then read back -- ``pqv = zqp1/(1 - zqp1)`` uses the NEW value --
so a capture taken after the block would record the answer as the input.
The fixture records it immediately after cumastrn returns.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_post_conversion

_IN = ("pcte", "ztp1", "ptte", "ztt", "pqte", "zqq", "zqp1",
       "qcf", "qif", "uf", "vf", "pvom", "pvol")
_OUT = ("pqc", "pqi", "pt", "pqv", "pu", "pv")


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-postconv-in-levels.csv"):
        s = cols.setdefault((int(r["case"]), float(r["dx"])), {})
        k = int(r["k"]) - 1
        for f in _IN:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-postconv-in-surface.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        s["prsfc"] = word(r["prsfc"])
        s["pssfc"] = word(r["pssfc"])
        s["delt"] = word(r["delt"])
    for r in load_csv("nt-postconv-out-levels.csv"):
        e = exp.setdefault((int(r["case"]), float(r["dx"])), {})
        k = int(r["k"]) - 1
        for f in _OUT:
            e.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-postconv-out-surface.csv"):
        exp[(int(r["case"]), float(r["dx"]))]["zprecc"] = word(r["zprecc"])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)
_GOT = {k: np_ntiedtke_post_conversion(**_COLS[k]) for k in _KEYS}


def _diff(got, want):
    g = np.atleast_1d(np.float32(got)).view(np.uint32)
    w = np.atleast_1d(np.float32(want)).view(np.uint32)
    return np.nonzero(g != w)[0]


@pytest.mark.parametrize("field", _OUT + ("zprecc",))
def test_every_output_field_is_bitwise(field):
    bad = []
    for key in _KEYS:
        d = _diff(_GOT[key][field], _EXP[key][field])
        if d.size:
            bad.append((key, d.tolist()[:4]))
    assert not bad, f"{field}: {bad[:3]}"


def test_the_condensate_arm_is_actually_taken_somewhere():
    """Green must not come from ``pcte <= 0`` everywhere.

    If no level detrained, ``pqc``/``pqi`` would be a straight copy on
    every column and the ``foealfa`` split -- the only interesting
    arithmetic in the block -- would be entirely ungraded.
    """
    levels = sum(int(np.count_nonzero(_COLS[k]["pcte"] > 0.0)) for k in _KEYS)
    assert levels >= 100, f"only {levels} levels detrain condensate"
    changed = sum(1 for k in _KEYS
                  if _diff(_GOT[k]["pqc"], _COLS[k]["qcf"]).size)
    assert changed >= 10, f"only {changed} columns change pqc"


def test_the_false_arm_CARRIES_the_incoming_value():
    """Class 2, and the reason the kernel must not zero at entry.

    On ``pcte <= 0`` the reference leaves ``pqc``/``pqi`` holding what the
    caller passed. The natural CUDA idiom of zeroing outputs at entry would
    diverge on every level that does not detrain -- which is most of them,
    measured below -- so this checks the carry directly rather than
    trusting that the bitwise test would have noticed.
    """
    carried = quiet = 0
    for key in _KEYS:
        mask = _COLS[key]["pcte"] <= 0.0
        quiet += int(np.count_nonzero(mask))
        carried += int(np.count_nonzero(
            _GOT[key]["pqc"][mask].view(np.uint32)
            == _COLS[key]["qcf"][mask].view(np.uint32)))
    assert quiet > 0 and carried == quiet, (
        f"{quiet - carried} of {quiet} non-detraining levels did not carry "
        f"the incoming pqc")
    # And the carry must be load-bearing: some of those incoming values
    # non-zero, or "carried" and "zeroed" are the same answer.
    nonzero = sum(int(np.count_nonzero(
        _COLS[k]["qcf"][_COLS[k]["pcte"] <= 0.0])) for k in _KEYS)
    assert nonzero > 0, (
        "every carried value is zero, so this fixture cannot tell a carry "
        "from a zeroing -- the same blindness the cutypen fixture had")


def test_zqp1_is_updated_in_place_and_the_new_value_feeds_pqv():
    """The ordering the capture boundary exists to protect.

    ``pqv = zqp1/(1 - zqp1)`` reads the UPDATED ``zqp1``. A mirror that
    used the incoming value would be wrong wherever ``pqte != zqq``, so
    this asserts the returned ``zqp1`` actually moved and that ``pqv``
    follows it rather than the input.
    """
    moved = [k for k in _KEYS
             if _diff(_GOT[k]["zqp1"], _COLS[k]["zqp1"]).size]
    assert len(moved) >= 10, f"only {len(moved)} columns move zqp1"
    key = moved[0]
    q = _GOT[key]["zqp1"]
    expect = (q / (np.float32(1.0) - q)).astype(np.float32)
    assert not _diff(expect, _GOT[key]["pqv"]).size


def test_the_precipitation_clamp_is_present_but_UNEXERCISED():
    """``amax1(0., ...)`` -- written in the direction of the gap.

    ``prsfc + pssfc`` is non-negative on every column of this fixture, so
    the clamp never fires and a port that dropped it would still be green.
    That is a real coverage gap and it is stated as one; this fails the day
    a column produces a negative surface flux sum, which is the signal to
    grade the clamp instead of noting it.
    """
    negative = [k for k in _KEYS
                if float(_COLS[k]["prsfc"]) + float(_COLS[k]["pssfc"]) < 0.0]
    assert not negative, (
        f"columns {negative[:4]} now have a negative surface flux sum, so "
        f"amax1's clamp is exercised -- grade it and delete this test")


# ===========================================================================
# The kernel
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_post_conversion():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.ntiedtke import NtLaunchGeometry, NtStages

    keys = _KEYS
    ncol, n1 = len(keys), NT_NZ + 2

    def pack(name):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, k in enumerate(keys):
            a[1:1 + NT_NZ, c] = _COLS[k][name]
        return cp.asarray(a)

    def col(name):
        return cp.asarray(np.array([float(_COLS[k][name]) for k in keys],
                                   dtype=np.float32))

    ro = {n: pack(n) for n in _IN if n != "zqp1"}
    d_zqp1 = pack("zqp1")
    outs = {n: cp.zeros((n1, ncol), dtype=np.float32) for n in _OUT}
    d_zprecc = cp.zeros(ncol, dtype=np.float32)

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_post_conversion", (
        ro["pcte"], ro["ztp1"], ro["ptte"], ro["ztt"], ro["pqte"], ro["zqq"],
        d_zqp1, ro["qcf"], ro["qif"], ro["uf"], ro["vf"],
        ro["pvom"], ro["pvol"], col("prsfc"), col("pssfc"),
        outs["pqc"], outs["pqi"], outs["pt"], outs["pqv"],
        outs["pu"], outs["pv"], d_zprecc,
        np.int32(ncol), np.int32(NT_NZ),
        np.float32(float(_COLS[keys[0]]["delt"]))))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    got = {n: cp.asnumpy(outs[n])[1:1 + NT_NZ, :] for n in _OUT}
    got["zprecc"] = cp.asnumpy(d_zprecc)
    return keys, got


@pytest.mark.parametrize("field", _OUT + ("zprecc",))
def test_kernel_is_bitwise(cuda_post_conversion, field):
    """Graded against WRF, never against the mirror."""
    keys, got = cuda_post_conversion
    bad = []
    for c, key in enumerate(keys):
        g = got[field][:, c] if got[field].ndim == 2 else got[field][c]
        d = _diff(g, _EXP[key][field])
        if d.size:
            bad.append((key, d.tolist()[:4]))
    assert not bad, f"{field}: {bad[:3]}"


def test_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function(
        "ntiedtke_post_conversion").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
