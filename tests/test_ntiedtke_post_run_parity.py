"""``cu_ntiedtke_post_run`` graded at ``max_ulp == 0`` -- the eight fields.

THE FIELDS PHASE 1 IS GRADED ON. ``nt-levels.csv`` carries eight tendency
columns -- ``rthcuten rqvcuten rqccuten rqicuten rucuten rvcuten raincv
pratec`` -- and until this slice existed every one of them traced to
NOTHING in the tree: ``cu_ntiedtke_run`` produces ``pt``/``pqv``/``pqc``/
``pqi``/``pu``/``pv`` and ``zprecc``, and no component turned those into
what the model consumes. The gap was visible in the oracle's own header the
whole time and was found by a gate driven off that header
(``test_ntiedtke_output_provenance.py``), not by anyone reading the list.

TWO VERTICAL CONVENTIONS IN ONE STATEMENT, and this is the only slice in
the port where that is true. ``exner/qv/qc/qi/t/u/v`` are the driver's
untouched WRF-order inputs -- k = 1 the SURFACE -- and they are the
reference state the tendency is measured against. ``tf/qvf/qcf/qif/uf/vf``
carry ``cu_ntiedtke_run``'s answer in SCHEME order, k = 1 the model TOP.
The routine pairs them by flipping.

The fixture records each array at its OWN index rather than pre-pairing
them, so the flip stays the port's job -- exactly as it is the routine's
job. Pairing them in the fixture would hide a flip error in the one capture
built to expose it, and ``test_the_flip_is_load_bearing`` below is what
turns that from an argument into a measurement.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_post_run

#: WRF order (k = 1 is the surface) -- the reference state.
_WRF = ("exner", "qv", "qc", "qi", "t", "u", "v")
#: Scheme order (k = 1 is the model top) -- cu_ntiedtke_run's answer.
_SCHEME = ("tf", "qvf", "qcf", "qif", "uf", "vf")
_OUT = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten")


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-post-in-levels.csv"):
        s = cols.setdefault((int(r["case"]), float(r["dx"])), {})
        k = int(r["k"]) - 1
        for f in _WRF + _SCHEME:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-post-in-surface.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        s["rn"] = word(r["rn"])
        s["dt"] = word(r["dt"])
        s["stepcu"] = int(r["stepcu"])
    for r in load_csv("nt-post-out-levels.csv"):
        e = exp.setdefault((int(r["case"]), float(r["dx"])), {})
        k = int(r["k"]) - 1
        for f in _OUT:
            e.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-post-out-surface.csv"):
        e = exp[(int(r["case"]), float(r["dx"]))]
        e["raincv"] = word(r["raincv"])
        e["pratec"] = word(r["pratec"])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)
_GOT = {k: np_ntiedtke_post_run(**_COLS[k]) for k in _KEYS}


def _diff(got, want):
    g = np.atleast_1d(np.float32(got)).view(np.uint32)
    w = np.atleast_1d(np.float32(want)).view(np.uint32)
    return np.nonzero(g != w)[0]


@pytest.mark.parametrize("field", _OUT + ("raincv", "pratec"))
def test_every_output_field_is_bitwise(field):
    bad = []
    for key in _KEYS:
        d = _diff(_GOT[key][field], _EXP[key][field])
        if d.size:
            bad.append((key, d.tolist()[:4]))
    assert not bad, f"{field}: {bad[:3]}"


def test_the_fixture_covers_every_column():
    """108 columns: 18 cases across the 6-member dx sweep."""
    assert len(_KEYS) == 108, len(_KEYS)


def test_the_flip_is_load_bearing():
    """Green must not be reachable without the vertical inversion.

    The port's one structural inversion has a SILENT failure mode, so a
    slice that pairs the two conventions has to prove that the pairing
    matters on this fixture rather than assume it. Feeding the scheme-order
    arrays unflipped must move the answer -- on every column, or the
    fixture's columns are too symmetric to grade the flip at all.
    """
    unmoved = []
    for key in _KEYS:
        c = dict(_COLS[key])
        for f in _SCHEME:
            c[f] = c[f][::-1].copy()
        other = np_ntiedtke_post_run(**c)
        if not _diff(other["rthcuten"], _GOT[key]["rthcuten"]).size:
            unmoved.append(key)
    assert not unmoved, (
        f"{len(unmoved)} columns give the same answer with the scheme-order "
        f"arrays reversed, so the flip is ungraded on them: {unmoved[:4]}")


def test_the_tendencies_are_ASSIGNED_not_accumulated():
    """The opposite contract from cudtdqn and the KE dissipation.

    Those two ADD into ptte and a mirror that assigned would discard the
    forcing. This one ASSIGNS, unconditionally, at every level of every
    column -- :514-524 has no ``if`` -- which is why post_run has no class-2
    rows in the aliasing audit despite six ``intent(inout)`` arrays.

    The mirror takes no output seed at all, so the contract is checked the
    only way it can be: the answer depends on nothing but the inputs named,
    and two calls on the same inputs agree bitwise.
    """
    key = _KEYS[0]
    again = np_ntiedtke_post_run(**_COLS[key])
    for f in _OUT + ("raincv", "pratec"):
        assert not _diff(again[f], _GOT[key][f]).size, f


def test_the_precipitation_pair_is_not_the_same_number():
    """``raincv`` is an accumulation and ``pratec`` a rate.

    ``rn/stepcu`` against ``rn/(stepcu*dt)``: with dt = 60 they differ by
    exactly that factor, so a port that copied one into the other would be
    green on neither. Checked on a column that actually rains, because on a
    dry column both are zero and the distinction is invisible.
    """
    wet = [k for k in _KEYS if float(_COLS[k]["rn"]) > 0.0]
    assert len(wet) >= 10, f"only {len(wet)} columns precipitate"
    for key in wet:
        r, p = float(_GOT[key]["raincv"]), float(_GOT[key]["pratec"])
        assert r != p, key
        assert abs(p * 60.0 - r) < 1e-6 * max(1.0, abs(r)), (key, r, p)


def test_stepcu_is_one_on_this_fixture_and_that_is_a_COVERAGE_GAP():
    """Written in the direction of the gap, so a gain breaks it.

    ``stepcu`` appears three times -- ``delt = dt*stepcu``,
    ``raincv = rn/stepcu``, ``pratec = rn/(stepcu*dt)`` -- and the fixture
    drives it at 1, where all three are identities. So the port's handling
    of a cumulus step longer than the model step is TRANSCRIBED AND
    UNGRADED, and the integer-to-real promotion in each is ungraded with it.

    This fails the day the fixture sweeps stepcu, which is the signal to
    delete it and grade those three expressions properly.
    """
    assert {int(_COLS[k]["stepcu"]) for k in _KEYS} == {1}, (
        "the fixture now drives stepcu at more than one value -- grade the "
        "three stepcu expressions and remove this test")


# ===========================================================================
# The kernel
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_post_run():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.ntiedtke import NtLaunchGeometry, NtStages

    keys = _KEYS
    ncol, n1 = len(keys), NT_NZ + 2

    def pack(name):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, k in enumerate(keys):
            a[1:1 + NT_NZ, c] = _COLS[k][name]
        return cp.asarray(a)

    ro = {n: pack(n) for n in _WRF + _SCHEME}
    d_rn = cp.asarray(np.array([float(_COLS[k]["rn"]) for k in keys],
                               dtype=np.float32))
    outs = {n: cp.zeros((n1, ncol), dtype=np.float32) for n in _OUT}
    d_raincv = cp.zeros(ncol, dtype=np.float32)
    d_pratec = cp.zeros(ncol, dtype=np.float32)

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_post_run", (
        ro["exner"], ro["qv"], ro["qc"], ro["qi"], ro["t"], ro["u"], ro["v"],
        ro["tf"], ro["qvf"], ro["qcf"], ro["qif"], ro["uf"], ro["vf"], d_rn,
        outs["rthcuten"], outs["rqvcuten"], outs["rqccuten"],
        outs["rqicuten"], outs["rucuten"], outs["rvcuten"],
        d_raincv, d_pratec,
        np.int32(ncol), np.int32(NT_NZ),
        np.int32(int(_COLS[keys[0]]["stepcu"])),
        np.float32(float(_COLS[keys[0]]["dt"]))))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    got = {n: cp.asnumpy(outs[n])[1:1 + NT_NZ, :] for n in _OUT}
    got["raincv"] = cp.asnumpy(d_raincv)
    got["pratec"] = cp.asnumpy(d_pratec)
    return keys, got


@pytest.mark.parametrize("field", _OUT + ("raincv", "pratec"))
def test_kernel_is_bitwise(cuda_post_run, field):
    """Graded against WRF, never against the mirror."""
    keys, got = cuda_post_run
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
    a = load_module("ntiedtke").get_function("ntiedtke_post_run").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
