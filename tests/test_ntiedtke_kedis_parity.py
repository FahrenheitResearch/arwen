"""cumastrn:1030-1056 graded at ``max_ulp == 0`` -- the KE dissipation.

THE LAST ARITHMETIC IN ``cumastrn``, and the only place the momentum
tendency feeds back into the heat tendency: cududvn has just changed
``pvom``/``pvol``, and this returns the kinetic energy that change removed
from the resolved flow as sensible heat.

``ztenu``/``ztenv`` are copies of ``pvom``/``pvol`` taken *before* cududvn,
so they are cududvn's own captured input and output -- which is exactly the
"a neighbour's capture will do" shape that has been wrong seven times. The
fixture records them at this block's own boundary instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_ke_dissipation

_IN = ("ztenu", "ztenv", "pvom", "pvol", "puen", "pven", "ptte")


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-cududvn-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "paph_sfc": word(r["paph_sfc"])}
    for r in load_csv("nt-mprofile-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))]["kctop"] = int(r["kctop"])
    for r in load_csv("nt-cududvn-in-levels.csv"):
        cols[(int(r["case"]), float(r["dx"]))].setdefault(
            "paph", np.zeros(NT_NZ, dtype=np.float32))[
            int(r["k"]) - 1] = word(r["paph"])
    for r in load_csv("nt-kedis-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _IN:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-kedis-out-levels.csv"):
        exp.setdefault((int(r["case"]), float(r["dx"])), {}).setdefault(
            "ptte", np.zeros(NT_NZ, dtype=np.float32))[
            int(r["k"]) - 1] = word(r["ptte"])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)


def _paph(c):
    out = np.empty(NT_NZ + 1, dtype=np.float32)
    out[:NT_NZ] = c["paph"]
    out[NT_NZ] = c["paph_sfc"]
    return out


_GOT = {k: np_ntiedtke_ke_dissipation(
    ldcum=bool(_COLS[k]["ldcum"]), kctop=_COLS[k]["kctop"],
    paph=_paph(_COLS[k]), puen=_COLS[k]["puen"], pven=_COLS[k]["pven"],
    ztenu=_COLS[k]["ztenu"], ztenv=_COLS[k]["ztenv"],
    pvom=_COLS[k]["pvom"], pvol=_COLS[k]["pvol"],
    ptte=_COLS[k]["ptte"]) for k in _KEYS}


def test_ptte_is_bitwise():
    bad = []
    for key in _KEYS:
        g, e = _GOT[key]["ptte"], _EXP[key]["ptte"]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"ptte: {bad[:3]}"


def test_the_dissipation_is_ADDED_not_assigned():
    """Same add-not-assign contract as cudtdqn, checked the same way.

    ptte arrives carrying the forcing plus cudtdqn's convective increment,
    so an assigning mirror would discard both. Perturbing the seed must
    move the output by exactly the perturbation.
    """
    key = next(k for k in _KEYS if _COLS[k]["ldcum"]
               and np.any(_GOT[k]["ptte"] != _COLS[k]["ptte"]))
    c = _COLS[key]
    bump = np.float32(3.25)
    other = np_ntiedtke_ke_dissipation(
        ldcum=True, kctop=c["kctop"], paph=_paph(c), puen=c["puen"],
        pven=c["pven"], ztenu=c["ztenu"], ztenv=c["ztenv"],
        pvom=c["pvom"], pvol=c["pvol"],
        ptte=(c["ptte"] + bump).astype(np.float32))
    k0 = max(0, c["kctop"] - 2)
    delta = (other["ptte"][k0:] - _GOT[key]["ptte"][k0:]).astype(np.float32)
    assert np.allclose(delta, bump, rtol=0, atol=1e-4), (
        f"perturbing ptte by {bump} moved the output by {delta[:3]}")


def test_the_dissipation_actually_heats():
    """Green must not come from every column dissipating nothing.

    zsum12 is minus the work done against the environmental wind. If it
    were zero everywhere, ztdis would be zero and this whole block would
    be an identity on ptte -- which is the cuentrn degeneracy shape.
    """
    heated = sum(1 for k in _KEYS
                 if not np.array_equal(_GOT[k]["ptte"], _COLS[k]["ptte"]))
    assert heated >= 20, f"only {heated} columns dissipate any KE"
    nonzero = sum(1 for k in _KEYS if float(_GOT[k]["zsum22"]) != 0.0)
    assert nonzero >= 20, f"only {nonzero} columns have a non-zero zsum22"


# ===========================================================================
# The kernel
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_kedis():
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

    ro = {n: pack(n) for n in ("puen", "pven", "ztenu", "ztenv",
                               "pvom", "pvol")}
    d_ptte = pack("ptte")
    d_zuv2 = cp.zeros((n1, ncol), dtype=np.float32)

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_ke_dissipation", (
        ivec("ldcum"), ivec("kctop"), pack("paph", sfc="paph_sfc"),
        ro["puen"], ro["pven"], ro["ztenu"], ro["ztenv"],
        ro["pvom"], ro["pvol"], d_ptte, d_zuv2,
        np.int32(ncol), np.int32(NT_NZ),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    return keys, cp.asnumpy(d_ptte)[1:1 + NT_NZ, :]


def test_kernel_ptte_is_bitwise(cuda_kedis):
    """Graded against WRF, never against the mirror."""
    keys, got = cuda_kedis
    bad = []
    for c, key in enumerate(keys):
        g, e = got[:, c], _EXP[key]["ptte"]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"ptte: {bad[:3]}"


def test_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function(
        "ntiedtke_ke_dissipation").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
