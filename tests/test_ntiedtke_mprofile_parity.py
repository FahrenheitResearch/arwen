"""cumastrn:927-995 graded at ``max_ulp == 0`` -- the momentum profiles.

What produces the ``puu``/``pvu``/``pud``/``pvd`` that cududvn consumes,
and the block that falsified this port's eighth wrong claim: cuascn,
cudlfsn and cuddrafn genuinely never write them, but the chained conclusion
"so cuinin sets them and nothing between touches them" was wrong, because
THIS does -- ``puu`` differs on 1,926 of 5,292 slots between cuinin's exit
and cududvn's entry.

``momtrans = 2`` is a parameter, so ``:943-955`` -- the ``momtrans == 1``
arm -- is a third dead block and the pressure-gradient ``else`` is live.
Only the live arm is transcribed; the omission is recorded rather than
silent.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_momentum_profile

_IN_LEV = ("pmfu", "pmfd", "puen", "pven", "puu", "pvu", "pud", "pvd",
           "pmfude_rate")
_OUT = ("puu", "pvu", "pud", "pvd")


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-mprofile-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "ktype": int(r["ktype"]),
            "kcbot": int(r["kcbot"]), "kctop": int(r["kctop"]),
            "kdpl": int(r["kdpl"]), "idtop": int(r["kdtop"])}
    for r in load_csv("nt-mprofile-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _IN_LEV:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    # pmfdde_rate comes from the same place cududvn's entry does; the
    # adjust block's exit capture is the last write to it before :927.
    for r in load_csv("nt-adjust-out-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        s.setdefault("pmfdde_rate", np.zeros(NT_NZ, dtype=np.float32))[
            int(r["k"]) - 1] = word(r["pmfdde_rate"])
    # cududvn's captured entry IS this block's exit.  :996-1016 runs
    # between them and assigns ONLY zmfuus/zmfdus -- grepped, 0 other
    # assignment targets in that range -- so puu/pvu/pud/pvd pass through.
    for r in load_csv("nt-cududvn-in-levels.csv"):
        s = exp.setdefault((int(r["case"]), float(r["dx"])), {})
        k = int(r["k"]) - 1
        for f in _OUT:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)

_GOT = {k: np_ntiedtke_momentum_profile(
    ldcum=bool(_COLS[k]["ldcum"]), ktype=_COLS[k]["ktype"],
    kcbot=_COLS[k]["kcbot"], kctop=_COLS[k]["kctop"],
    kdpl=_COLS[k]["kdpl"], idtop=_COLS[k]["idtop"],
    puen=_COLS[k]["puen"], pven=_COLS[k]["pven"], pmfu=_COLS[k]["pmfu"],
    pmfd=_COLS[k]["pmfd"], puu=_COLS[k]["puu"], pvu=_COLS[k]["pvu"],
    pud=_COLS[k]["pud"], pvd=_COLS[k]["pvd"],
    pmfude_rate=_COLS[k]["pmfude_rate"],
    pmfdde_rate=_COLS[k]["pmfdde_rate"]) for k in _KEYS}


@pytest.mark.parametrize("field", _OUT)
def test_outputs_are_bitwise(field):
    bad = []
    for key in _KEYS:
        g, e = _GOT[key][field], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_this_block_really_is_what_rewrites_puu():
    """Instance 8's disproof, kept as a test.

    The claim was "cuinin sets puu/pvu and nothing between touches them".
    This measures the opposite directly: the block's output differs from
    its own input on a large number of slots, so it is the writer.
    """
    changed = sum(
        int(np.count_nonzero(
            _GOT[k]["puu"].view(np.uint32) != _COLS[k]["puu"].view(np.uint32)))
        for k in _KEYS)
    assert changed > 1000, (
        f"only {changed} puu slots change across this block; instance 8's "
        "claim that nothing between cuinin and cududvn touches puu would "
        "then be nearly true, and the correction in section 21 should be "
        "re-examined")


# ===========================================================================
# The kernel
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_mprofile():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.ntiedtke import NtLaunchGeometry, NtStages

    keys = _KEYS
    ncol, n1 = len(keys), NT_NZ + 2

    def pack(name):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, k in enumerate(keys):
            a[1:1 + NT_NZ, c] = _COLS[k][name]
        return cp.asarray(a)

    def ivec(name):
        return cp.asarray(np.array([int(_COLS[k][name]) for k in keys],
                                   dtype=np.int32))

    ro = {n: pack(n) for n in ("puen", "pven", "pmfu", "pmfd",
                               "pmfude_rate", "pmfdde_rate")}
    io_ = {n: pack(n) for n in ("puu", "pvu", "pud", "pvd")}

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_momentum_profile", (
        ivec("ldcum"), ivec("ktype"), ivec("kcbot"), ivec("kctop"),
        ivec("kdpl"), ivec("idtop"),
        ro["puen"], ro["pven"], ro["pmfu"], ro["pmfd"],
        ro["pmfude_rate"], ro["pmfdde_rate"],
        io_["puu"], io_["pvu"], io_["pud"], io_["pvd"],
        np.int32(ncol), np.int32(NT_NZ)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    return keys, {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in io_.items()}


@pytest.mark.parametrize("field", _OUT)
def test_kernel_outputs_are_bitwise(cuda_mprofile, field):
    """Graded against WRF, never against the mirror."""
    keys, got = cuda_mprofile
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
    a = load_module("ntiedtke").get_function(
        "ntiedtke_momentum_profile").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
