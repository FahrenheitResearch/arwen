"""cududvn graded against the pinned WRF v4.6.1 oracle at ``max_ulp == 0``.

The last routine, and the one that closes the momentum story. cuascn,
cudlfsn and cuddrafn were each found to take ``puu``/``pvu``/``pud``/``pvd``
and never write them -- three separate findings, each gated on the mirror's
shape because the oracle cannot tell "left alone" from "not implemented".
This is where they are consumed.

CORRECTED: an earlier version of this note concluded that cuinin sets them
and nothing between touches them. **cumastrn:927-995 does** -- ``puu``
differs on 1,926 of 5,292 slots between cuinin's exit and cududvn's entry.
The values below are still right because they are captured at cududvn's own
call site, which is capture-first producing a correct answer despite
incorrect reasoning.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_DXSWEEP, NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_cududvn

_IN_LEV = ("paph", "puen", "pven", "pmfu", "pmfd", "puu", "pud", "pvu",
           "pvd", "ptenu_in", "ptenv_in")
_OUT = ("ptenu", "ptenv")


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-cududvn-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "ktype": int(r["ktype"]),
            "kcbot": int(r["kcbot"]), "ktopm2": int(r["ktopm2"]),
            "paph_sfc": word(r["paph_sfc"])}
    for r in load_csv("nt-cududvn-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _IN_LEV:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-cududvn-out-levels.csv"):
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


def _run(key, ptenu=None):
    c = _COLS[key]
    return np_ntiedtke_cududvn(
        ktopm2=c["ktopm2"], ktype=c["ktype"], kcbot=c["kcbot"],
        ldcum=bool(c["ldcum"]), paph=_paph(c), puen=c["puen"],
        pven=c["pven"], pmfu=c["pmfu"], pmfd=c["pmfd"], puu=c["puu"],
        pud=c["pud"], pvu=c["pvu"], pvd=c["pvd"],
        ptenu=c["ptenu_in"] if ptenu is None else ptenu,
        ptenv=c["ptenv_in"])


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


def test_the_momentum_seed_is_measured_zero_not_assumed_zero():
    """cu_ntiedtke_run:258-259 sets pvom = pvol = 0. before the call.

    That is what makes accumulate and replace coincide for RUCUTEN and
    RVCUTEN -- unlike RTHCUTEN/RQVCUTEN, which are seeded with the forcing
    (see docs/ntiedtke/PORT-RECORD.md 17). It is a property of the REFERENCE's
    pipeline, so it is measured on the fixture rather than trusted, and it
    carries a call-site debt until a stage performs :258-259.
    """
    nonzero = sum(int(np.count_nonzero(_COLS[k]["ptenu_in"]))
                  + int(np.count_nonzero(_COLS[k]["ptenv_in"]))
                  for k in _KEYS)
    assert nonzero == 0, (
        f"{nonzero} momentum-tendency slots are non-zero at cududvn's "
        "entry. accumulate and replace no longer coincide for RUCUTEN/"
        "RVCUTEN, and the Phase 2 contract in section 10 needs revisiting.")


def test_the_mirror_still_ADDS_even_though_the_seed_is_zero():
    """A zero seed makes assign and add indistinguishable against the
    oracle. So the property is tested directly instead.

    This is the cuentrn lesson: a fixture that cannot discriminate grades
    a tautology. If the driver ever hands a non-zero seed -- and §17 shows
    the sibling pair already does -- an assigning mirror would be silently
    wrong and nothing here would have caught it.
    """
    key = next(k for k in _KEYS if _COLS[k]["ldcum"])
    bump = np.float32(0.25)
    seed = (_COLS[key]["ptenu_in"] + bump).astype(np.float32)
    other = _run(key, ptenu=seed)
    k2 = _COLS[key]["ktopm2"] - 1
    delta = (other["ptenu"][k2:] - _GOT[key]["ptenu"][k2:]).astype(np.float32)
    assert np.allclose(delta, bump, rtol=0, atol=1e-6), (
        "cududvn must ADD to the incoming ptenu, not assign it; "
        f"perturbing the seed by {bump} moved the output by {delta[:3]}")


def test_the_fixture_actually_transports_momentum():
    moved = sum(1 for k in _KEYS if np.any(_GOT[k]["ptenu"] != 0))
    assert moved >= 30, f"only {moved} columns produced a u tendency"
    below = sum(1 for k in _KEYS
                if _COLS[k]["ldcum"] and _COLS[k]["kcbot"] < NT_NZ)
    assert below >= 20, (
        f"only {below} columns have levels below cloud base, so the linear "
        "taper at :3203-3215 is barely exercised")
    deep = sum(1 for k in _KEYS if _COLS[k]["ktype"] == 3)
    assert deep >= 1, "no ktype == 3 column, so the zzp squaring is untested"


# ===========================================================================
# The kernel, graded against the SAME oracle rows as the mirror
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_cududvn():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.ntiedtke import NtLaunchGeometry, NtStages

    keys = _KEYS
    ncol, n1 = len(keys), NT_NZ + 2

    def pack(name, sfc=None):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, k in enumerate(keys):
            a[1:1 + NT_NZ, c] = _COLS[k][name]
            if sfc:
                a[1 + NT_NZ, c] = _COLS[k][sfc]
        return cp.asarray(a)

    def ivec(name):
        return cp.asarray(np.array([int(_COLS[k][name]) for k in keys],
                                   dtype=np.int32))

    ro = {n: pack(n) for n in ("puen", "pven", "pmfu", "pmfd",
                               "puu", "pud", "pvu", "pvd")}
    io_ = {"ptenu": pack("ptenu_in"), "ptenv": pack("ptenv_in")}
    # The port's only scratch: the four zmf* arrays are not single-level
    # lookback, so they cannot be registers.  Caller-owned, so the FRAME
    # stays 0 B and the cost is visible in this allocation.
    scr = [cp.zeros((n1, ncol), dtype=np.float32) for _ in range(4)]

    ktopm2 = _COLS[keys[0]]["ktopm2"]
    assert all(_COLS[k]["ktopm2"] == ktopm2 for k in keys)

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_cududvn", (
        ivec("ldcum"), ivec("ktype"), ivec("kcbot"),
        pack("paph", sfc="paph_sfc"),
        ro["puen"], ro["pven"], ro["pmfu"], ro["pmfd"],
        ro["puu"], ro["pud"], ro["pvu"], ro["pvd"],
        io_["ptenu"], io_["ptenv"], scr[0], scr[1], scr[2], scr[3],
        np.int32(ncol), np.int32(NT_NZ), np.int32(ktopm2),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    return keys, {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in io_.items()}


@pytest.mark.parametrize("field", _OUT)
def test_kernel_outputs_are_bitwise(cuda_cududvn, field):
    """Graded against WRF, never against the mirror."""
    keys, got = cuda_cududvn
    bad = []
    for c, key in enumerate(keys):
        g, e = got[field][:, c], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_kernel_holds_no_local_frame():
    """0 B even though this kernel is the one that needs scratch.

    The four zmf* arrays are caller-allocated, so they cost VRAM in the
    launcher's budget rather than per-thread local memory -- which is the
    whole distinction standing rule 3 turns on.
    """
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_cududvn").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
