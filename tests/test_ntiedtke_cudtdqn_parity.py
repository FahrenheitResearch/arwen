"""cudtdqn graded against the pinned WRF v4.6.1 oracle at ``max_ulp == 0``.

Where the mass fluxes become RTHCUTEN and RQVCUTEN. The first routine in
this port whose hazards were known before it was written: the aliasing
audit's third report named ``ptent``/``ptenq`` as self-referential
accumulators while cuflxn was still being graded.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_DXSWEEP, NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_cudtdqn

_IN_LEV = ("paph", "pten", "plglac", "plude", "pmfus", "pmfds", "pmfuq",
           "pmfdq", "pmful", "pdmfup", "pdmfdp", "pdpmel",
           "ptent_in", "ptenq_in", "pcte_in", "pmfu", "pmfd")
_OUT = ("ptent", "ptenq", "pcte")


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-cudtdqn-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "kctop": int(r["kctop"]),
            "ktopm2": int(r["ktopm2"]), "paph_sfc": word(r["paph_sfc"])}
    for r in load_csv("nt-cudtdqn-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _IN_LEV:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-cudtdqn-out-levels.csv"):
        s = exp.setdefault((int(r["case"]), float(r["dx"])), {})
        k = int(r["k"]) - 1
        for f in _OUT:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)


def _paph(c):
    out = np.empty(NT_NZ + 1, dtype=np.float32)
    out[:NT_NZ] = c["paph"]
    out[NT_NZ] = c["paph_sfc"]
    return out


def _run(key, ptent=None):
    c = _COLS[key]
    return np_ntiedtke_cudtdqn(
        ktopm2=c["ktopm2"], ldcum=bool(c["ldcum"]), paph=_paph(c),
        pten=c["pten"], plglac=c["plglac"], plude=c["plude"],
        pmfus=c["pmfus"], pmfds=c["pmfds"], pmfuq=c["pmfuq"],
        pmfdq=c["pmfdq"], pmful=c["pmful"], pdmfup=c["pdmfup"],
        pdmfdp=c["pdmfdp"], pdpmel=c["pdpmel"],
        ptent=c["ptent_in"] if ptent is None else ptent,
        ptenq=c["ptenq_in"], pcte=c["pcte_in"])


_GOT = {k: _run(k) for k in _KEYS}


@pytest.mark.parametrize("dx", NT_DXSWEEP)
@pytest.mark.parametrize("field", _OUT)
def test_outputs_are_bitwise(dx, field):
    bad = []
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = _GOT[key][field], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key[0], d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field} @ dx={dx}: {bad[:3]}"


def test_the_tendencies_are_ACCUMULATED_not_assigned():
    """The port's contract question, answered against the fixture.

    :3140-3141 is `ptent = ptent + zdtdt`, and the incoming array is NOT
    zero -- measured, 4,428 of 5,292 rows. So a mirror that assigned would
    be wrong on 84% of the fixture, and this proves the mirror adds by
    perturbing the seed and requiring the output to move by exactly the
    same amount.

    A test that merely compared to the oracle would also catch it, but not
    say WHY; this one names the property.
    """
    key = next(k for k in _KEYS if _COLS[k]["ldcum"]
               and np.any(_COLS[k]["ptent_in"] != 0))
    bump = np.float32(1.5)
    seed = (_COLS[key]["ptent_in"] + bump).astype(np.float32)
    other = _run(key, ptent=seed)
    k2 = _COLS[key]["ktopm2"] - 1
    delta = (other["ptent"][k2:] - _GOT[key]["ptent"][k2:]).astype(np.float32)
    assert np.allclose(delta, bump, rtol=0, atol=1e-5), (
        "cudtdqn must ADD to the incoming ptent, not assign it; "
        f"perturbing the seed by {bump} moved the output by {delta[:3]}")


def test_the_incoming_tendencies_really_are_non_zero():
    """The measurement §17 rests on, pinned so it cannot rot.

    If a future fixture seeds these with zeros, accumulate and assign
    become indistinguishable and the test above proves nothing.
    """
    nz_rows = sum(int(np.count_nonzero(_COLS[k]["ptent_in"])) for k in _KEYS)
    assert nz_rows > 4000, (
        f"only {nz_rows} non-zero ptent rows at entry; the accumulate-vs-"
        "assign distinction is no longer observable in this fixture")
    assert all(not np.any(_COLS[k]["pcte_in"]) for k in _KEYS), \
        "pcte is measured zero on entry; if that changed, say so in §17"


def test_the_fixture_actually_produces_tendencies():
    moved = sum(1 for k in _KEYS
                if not np.array_equal(_GOT[k]["ptent"], _COLS[k]["ptent_in"]))
    assert moved >= 30, f"only {moved} columns produced a heat tendency"
    detr = sum(1 for k in _KEYS if np.any(_GOT[k]["pcte"] != 0))
    assert detr >= 20, f"only {detr} columns detrained condensate"


def test_pdmfdp_is_zero_so_the_zdqdt_GROUPING_is_unverifiable():
    """A named coverage gap, found by a compile error rather than a test.

    :3117-3119 chains nine terms left to right with no internal
    parentheses. The mirror originally grouped the last two as
    `- (pdmfup + pdmfdp)`, which is different arithmetic -- and it passed
    at max_ulp == 0 on all 5,292 rows BOTH WAYS.

    The reason: pdmfdp is zero on every one of the 5,292 slots at this
    routine's entry, so the two forms are identical by construction. The
    correct form is now in, verified against the source rather than
    against the fixture, because the fixture cannot see the difference.

    This gap is DOWNSTREAM of section 16's: pdmfdp reaches here through
    cuflxn, whose evaporation block -- the one measured never to fire --
    is what would make it non-zero. Closing that case-table gap closes
    this one too.
    """
    zeros = sum(int(np.count_nonzero(_COLS[k]["pdmfdp"])) for k in _KEYS)
    assert zeros == 0, (
        f"{zeros} pdmfdp slots are now non-zero, so the zdqdt term "
        "grouping at :3117-3119 IS observable. That is a coverage GAIN: "
        "invert this assertion and remove the gap from section 18.")


# ===========================================================================
# The kernel, graded against the SAME oracle rows as the mirror
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_cudtdqn():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.ntiedtke import NtLaunchGeometry, NtStages

    keys = _KEYS
    ncol, n1 = len(keys), NT_NZ + 2

    def pack(name):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, k in enumerate(keys):
            a[1:1 + NT_NZ, c] = _COLS[k][name]
        return cp.asarray(a)

    def pack_paph():
        """klev+1 valid: zdp reads paph(jk+1) at jk = klev."""
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, k in enumerate(keys):
            a[1:1 + NT_NZ, c] = _COLS[k]["paph"]
            a[1 + NT_NZ, c] = _COLS[k]["paph_sfc"]
        return cp.asarray(a)

    ro = {n: pack(n) for n in ("pten", "plglac", "plude", "pmfus", "pmfds",
                               "pmfuq", "pmfdq", "pmful", "pdmfup",
                               "pdmfdp", "pdpmel")}
    io_ = {"ptent": pack("ptent_in"), "ptenq": pack("ptenq_in"),
           "pcte": pack("pcte_in")}
    d_ld = cp.asarray(np.array([int(_COLS[k]["ldcum"]) for k in keys],
                               dtype=np.int32))
    ktopm2 = _COLS[keys[0]]["ktopm2"]
    assert all(_COLS[k]["ktopm2"] == ktopm2 for k in keys), \
        "ktopm2 differs per column; it is a scalar kernel argument"

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_cudtdqn", (
        d_ld, pack_paph(), ro["pten"], ro["plglac"], ro["plude"],
        ro["pmfus"], ro["pmfds"], ro["pmfuq"], ro["pmfdq"], ro["pmful"],
        ro["pdmfup"], ro["pdmfdp"], ro["pdpmel"],
        io_["ptent"], io_["ptenq"], io_["pcte"],
        np.int32(ncol), np.int32(NT_NZ), np.int32(ktopm2),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    return keys, {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in io_.items()}


@pytest.mark.parametrize("field", _OUT)
def test_kernel_outputs_are_bitwise(cuda_cudtdqn, field):
    """Graded against WRF, never against the mirror."""
    keys, got = cuda_cudtdqn
    bad = []
    for c, key in enumerate(keys):
        g, e = got[field][:, c], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_cudtdqn").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
