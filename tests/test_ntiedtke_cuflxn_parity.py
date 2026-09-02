"""cuflxn graded against the pinned WRF v4.6.1 oracle at ``max_ulp == 0``.

The final convective fluxes: flux-form anomalies, the cloud-base taper,
snow melt, and evaporation of falling precipitation into the sub-cloud
layer. The largest routine after cuascn.

``ktopm2`` is the port's one derived-not-assumed horizontal claim, and it
gets its own test below rather than a comment.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_DXSWEEP, NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_cuflxn

_IN_LEV = ("pten", "pqen", "pqsen", "ptenh", "pqenh", "paph", "pap",
           "pgeoh", "pmfu", "pmfd", "pmfus", "pmfds", "pmfuq", "pmfdq",
           "pmful", "pdmfup", "pdmfdp",
           # The THIRD hazardous class: intent(inout) whose first use is a
           # self-referential write.  :2887 is plglac = pmfu*plglac.
           "plglac_in", "plude_in", "pmfdde_rate_in")
_OUT_LEV = ("pmfu", "pmfd", "pmfus", "pmfds", "pmfuq", "pmfdq", "pmful",
            "plglac", "pdmfup", "pdmfdp", "pdpmel", "pqsen", "plude",
            "pmfdde_rate")
_ZTMST = np.float32(60.0)


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-cuflxn-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "lddraf": int(r["lddraf"]),
            "ktype": int(r["ktype"]), "kcbot": int(r["kcbot"]),
            "kctop": int(r["kctop"]), "kdtop": int(r["kdtop"]),
            "paph_sfc": word(r["paph_sfc"])}
    for r in load_csv("nt-cuflxn-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _IN_LEV:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-cuflxn-out-surface.csv"):
        exp[(int(r["case"]), float(r["dx"]))] = {
            "prain": word(r["prain"]),
            "pmflxr_sfc": word(r["pmflxr_sfc"]),
            "pmflxs_sfc": word(r["pmflxs_sfc"])}
    for r in load_csv("nt-cuflxn-out-levels.csv"):
        s = exp[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in (*_OUT_LEV, "pmflxr", "pmflxs"):
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)

#: lndj is not in this capture; it reaches cuflxn only through rhevap, and
#: the cudlfsn capture records it for the same columns.
_LNDJ = {(int(r["case"]), float(r["dx"])): int(r["lndj"])
         for r in load_csv("nt-cudlfsn-in-surface.csv")}


def _paph(col):
    out = np.empty(NT_NZ + 1, dtype=np.float32)
    out[:NT_NZ] = col["paph"]
    out[NT_NZ] = col["paph_sfc"]
    return out


def _run(key, plude=None):
    c = _COLS[key]
    return np_ntiedtke_cuflxn(
        ldcum=bool(c["ldcum"]), lddraf=bool(c["lddraf"]), ktype=c["ktype"],
        kcbot=c["kcbot"], kctop=c["kctop"], kdtop=c["kdtop"],
        lndj=_LNDJ[key], pten=c["pten"], pqen=c["pqen"], pqsen=c["pqsen"],
        ptenh=c["ptenh"], pqenh=c["pqenh"], paph=_paph(c), pap=c["pap"],
        pgeoh=c["pgeoh"], pmfu=c["pmfu"], pmfd=c["pmfd"],
        pmfus=c["pmfus"], pmfds=c["pmfds"], pmfuq=c["pmfuq"],
        pmfdq=c["pmfdq"], pmful=c["pmful"],
        plude=c["plude_in"] if plude is None else plude,
        plglac=c["plglac_in"],
        pdmfup=c["pdmfup"], pdmfdp=c["pdmfdp"],
        pmfdde_rate=c["pmfdde_rate_in"], ztmst=_ZTMST)


_GOT = {k: _run(k) for k in _KEYS}


@pytest.mark.parametrize("dx", NT_DXSWEEP)
@pytest.mark.parametrize("field", _OUT_LEV)
def test_level_outputs_are_bitwise(dx, field):
    bad = []
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = _GOT[key][field], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key[0], d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field} @ dx={dx}: {bad[:3]}"


@pytest.mark.parametrize("dx", NT_DXSWEEP)
@pytest.mark.parametrize("field", ("pmflxr", "pmflxs"))
def test_the_precipitation_fluxes_are_bitwise(dx, field):
    """klev+1 arrays; the surface slot is the scheme's actual rain/snow."""
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = _GOT[key][field][:NT_NZ], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        assert not d.size, f"{field} {key} levels {d[:5]}"
        gs = np.float32(_GOT[key][field][NT_NZ])
        es = _EXP[key][f"{field}_sfc"]
        assert gs.view(np.uint32) == es.view(np.uint32), \
            f"{field} surface {key}: {gs} vs {es}"


@pytest.mark.parametrize("dx", NT_DXSWEEP)
def test_prain_is_bitwise(dx):
    """An accumulator, graded on its own axis."""
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = np.float32(_GOT[key]["prain"]), _EXP[key]["prain"]
        assert g.view(np.uint32) == e.view(np.uint32), f"prain {key}"


def test_ktopm2_is_two_and_the_leak_is_dead():
    """cumastrn:565's horizontal leak into itopm2 cannot reach anything.

    :565 sets `itopm2 = kctop(jl)` INSIDE a do-jl loop, so the value that
    survives is the LAST column's cloud top -- a genuine horizontal leak,
    and it is passed here as an intent(inout) dummy. But :2877 sets
    `ktopm2 = 2` unconditionally at routine top level, with no read of
    ktopm2 anywhere before it, and cuflxn runs before cudtdqn and cududvn
    which are its only other consumers.

    So the mirror ignores the incoming value entirely, and this pins that
    that is correct rather than convenient. It is the one horizontal claim
    in this port that was re-derived from source after a sibling claim
    (cuascn's column independence) failed the same scrutiny.
    """
    for key in _KEYS:
        assert _GOT[key]["ktopm2"] == 2
    import inspect
    from gpuwm.verify.ntiedtke_ref import np_ntiedtke_cuflxn as fn
    assert "ktopm2" not in inspect.signature(fn).parameters, (
        "ktopm2 must NOT be an input: taking it would make the mirror "
        "depend on a value the reference overwrites before reading.")


def test_the_fixture_exercises_melt_and_evaporation():
    """Green must not come from every column skipping both branches."""
    melted = sum(1 for k in _KEYS if np.any(_GOT[k]["pdpmel"] != 0))
    assert melted >= 5, f"only {melted} columns melted snow"
    rained = sum(1 for k in _KEYS if _GOT[k]["pmflxr"][NT_NZ] > 0)
    assert rained >= 20, f"only {rained} columns reach the surface with rain"


def test_the_evaporation_MAGNITUDE_is_a_named_coverage_gap():
    """:3018's zdrfl1 is computed on 48 columns and is zero on all of them.

    MEASURED, not assumed:

      columns reaching zrfl > 1e-20 (the gate)      48
      columns where pdmfup moved (the magnitude)     0

    zdrfl1 carries a factor max(0, pqsen - pqen), and these tropical
    marine soundings are saturated or supersaturated below cloud base at
    every level the block evaluates. So `zdrfl = min(0, zrfln - zrfl)` is
    identically zero and the whole evaporation adjustment -- the 0.5777
    power law, zrmin, the rhevap land/sea split -- is transcribed and
    GRADED ONLY IN THE SENSE THAT ITS GUARD IS EVALUATED.

    That matters more than the cuddrafn gap: rhevap is the only place
    lndj enters cuflxn, so the land/sea distinction is untested here too.

    Written in the direction of the gap, as cuentrn's degeneracy was: this
    FAILS when a case with sub-saturated sub-cloud air is added, and that
    failure is the signal to invert it.
    """
    moved = sum(1 for k in _KEYS
                if not np.array_equal(_GOT[k]["pdmfup"], _COLS[k]["pdmfup"]))
    assert moved == 0, (
        f"{moved} columns now evaporate falling precipitation, so "
        "cuflxn:3018-3025 IS exercised. That is a coverage GAIN: invert "
        "this assertion to require it, and remove the gap from the "
        "excluded list in docs/ntiedtke/PORT-RECORD.md.")


def test_the_surface_interface_is_load_bearing():
    """paph[klev+1] drives every taper ratio; perturbing it must move."""
    key = next(k for k in _KEYS if _COLS[k]["ldcum"]
               and np.any(_GOT[k]["pmfu"] != 0))
    c = dict(_COLS[key])
    base = _GOT[key]["pmfu"]
    c["paph_sfc"] = np.float32(c["paph_sfc"] * np.float32(1.02))
    saved, _COLS[key] = _COLS[key], c
    try:
        other = _run(key)
    finally:
        _COLS[key] = saved
    assert not np.array_equal(other["pmfu"], base), \
        "perturbing paph[klev+1] changed nothing"


# ===========================================================================
# The kernel, graded against the SAME oracle rows as the mirror
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_cuflxn():
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

    def ivec(name, src=None):
        return cp.asarray(np.array(
            [int((src or _COLS)[k][name]) for k in keys], dtype=np.int32))

    ro = {n: pack(n) for n in ("pten", "ptenh", "pqenh", "pap", "pqen",
                               "pgeoh")}
    # Everything cuflxn rewrites in place, seeded from the ENTRY capture --
    # including plglac and pmfdde_rate, which have no intent attribute and
    # were invisible to three of the audit's four reports.
    io_ = {n: pack(n) for n in ("pmfu", "pmfd", "pmfus", "pmfds", "pmfuq",
                                "pmfdq", "pmful")}
    io_["plude"] = pack("plude_in")
    io_["plglac"] = pack("plglac_in")
    io_["pdmfup"] = pack("pdmfup")
    io_["pdmfdp"] = pack("pdmfdp")
    io_["pmfdde_rate"] = pack("pmfdde_rate_in")
    io_["pqsen"] = pack("pqsen")
    io_["pdpmel"] = cp.zeros((n1, ncol), dtype=np.float32)
    io_["pmflxr"] = cp.zeros((n1, ncol), dtype=np.float32)
    io_["pmflxs"] = cp.zeros((n1, ncol), dtype=np.float32)
    d_prain = cp.zeros(ncol, dtype=np.float32)
    d_ldcum = ivec("ldcum")
    d_lddraf = ivec("lddraf")
    d_ktype = ivec("ktype")
    d_lndj = cp.asarray(np.array([_LNDJ[k] for k in keys], dtype=np.int32))

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_cuflxn", (
        ivec("kcbot"), ivec("kctop"), ivec("kdtop"), d_lndj,
        ro["pten"], ro["ptenh"], ro["pqenh"], pack("paph", sfc="paph_sfc"),
        ro["pap"], ro["pqen"], ro["pgeoh"],
        d_ldcum, d_lddraf, d_ktype,
        io_["pmfu"], io_["pmfd"], io_["pmfus"], io_["pmfds"], io_["pmfuq"],
        io_["pmfdq"], io_["pmful"], io_["plude"], io_["plglac"],
        io_["pdmfup"], io_["pdmfdp"], io_["pmfdde_rate"], io_["pqsen"],
        io_["pdpmel"], io_["pmflxr"], io_["pmflxs"], d_prain,
        np.int32(ncol), np.int32(NT_NZ), _ZTMST,
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    out = {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in io_.items()}
    out["_pmflxr_sfc"] = cp.asnumpy(io_["pmflxr"])[1 + NT_NZ, :]
    out["_pmflxs_sfc"] = cp.asnumpy(io_["pmflxs"])[1 + NT_NZ, :]
    return keys, out, cp.asnumpy(d_prain)


@pytest.mark.parametrize("field", _OUT_LEV)
def test_kernel_level_outputs_are_bitwise(cuda_cuflxn, field):
    """Graded against WRF, never against the mirror."""
    keys, got, _ = cuda_cuflxn
    bad = []
    for c, key in enumerate(keys):
        g, e = got[field][:, c], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


@pytest.mark.parametrize("field", ("pmflxr", "pmflxs"))
def test_kernel_precipitation_fluxes_are_bitwise(cuda_cuflxn, field):
    keys, got, _ = cuda_cuflxn
    for c, key in enumerate(keys):
        g, e = got[field][:, c], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        assert not d.size, f"{field} {key} levels {d[:5]}"
        gs = np.float32(got[f"_{field}_sfc"][c])
        es = _EXP[key][f"{field}_sfc"]
        assert gs.view(np.uint32) == es.view(np.uint32), \
            f"{field} surface {key}"


def test_kernel_prain_is_bitwise(cuda_cuflxn):
    keys, _, prain = cuda_cuflxn
    for c, key in enumerate(keys):
        g, e = np.float32(prain[c]), _EXP[key]["prain"]
        assert g.view(np.uint32) == e.view(np.uint32), f"prain {key}"


def test_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_cuflxn").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
