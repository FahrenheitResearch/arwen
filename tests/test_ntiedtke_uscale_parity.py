"""cumastrn:743-819 graded at ``max_ulp == 0`` -- the updraft rescale.

WHERE THE CLOSURE'S ANSWER ACTUALLY LANDS. :745 forms
``zmfs = zmfub1 / max(cmfcmin, zmfub)`` and this block applies it to the
whole updraft. ``zmfub1`` is what the CAPE closure produced -- the quantity
``scale_fac`` and ``scale_fac2`` act on, and the reason this port exists.
§9 measured its retention; this is the code that spends it.

Outputs are cuflxn's captured entry state, so only the entry side needed a
new capture.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_updraft_scale

_ZTMST = np.float32(60.0)
_IN_LEV = ("pmfu", "pmfus", "pmfuq", "pmful", "pdmfup", "plude",
           "pmfude_rate", "paph",
           "pmfd", "pmfds", "pmfdq", "pdmfdp", "pmfdde_rate")
_OUT = ("pmfu", "pmfus", "pmfuq", "pmful", "pdmfup", "pmfd", "pmfds",
        "pmfdq", "pdmfdp")


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-uscale-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "ktype": int(r["ktype"]),
            "kcbot": int(r["kcbot"]), "idtop": int(r["kdtop"]),
            "zmfub1": word(r["zmfub1"]), "zmfub": word(r["zmfub"])}
    for r in load_csv("nt-cuascn-surface.csv"):
        k = (int(r["case"]), float(r["dx"]))
        if k in cols:
            cols[k]["kctop"] = int(r["kctop"])
    for r in load_csv("nt-cuddrafn-in-surface.csv"):
        k = (int(r["case"]), float(r["dx"]))
        cols[k]["loddraf"] = int(r["lddraf"])
        cols[k]["paph_sfc"] = word(r["paph_sfc"])
    for r in load_csv("nt-uscale-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _IN_LEV:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    # SEVENTH INSTANCE.  These were sourced from nt-downdraft-levels --
    # the CLOSURE's pre-state -- on the reasoning that nothing between
    # touches them.  The closure itself does: :726-740 scales the whole
    # downdraft by zmfub1/zmfub.  Captured at :743 instead.
    # cuflxn's captured ENTRY is this block's exit.
    for r in load_csv("nt-cuflxn-in-surface.csv"):
        exp[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "ktype": int(r["ktype"]),
            "idtop": int(r["kdtop"])}
    for r in load_csv("nt-cuflxn-in-levels.csv"):
        s = exp[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _OUT:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
        s.setdefault("plude", np.zeros(NT_NZ, dtype=np.float32))[k] = \
            word(r["plude_in"])
        s.setdefault("pmfdde_rate", np.zeros(NT_NZ, dtype=np.float32))[k] = \
            word(r["pmfdde_rate_in"])
    # cuflxn does not write pmfude_rate, so its EXIT value is this block's.
    for r in load_csv("nt-cuflxn-out-levels.csv"):
        s = exp[(int(r["case"]), float(r["dx"]))]
        s.setdefault("pmfude_rate", np.zeros(NT_NZ, dtype=np.float32))[
            int(r["k"]) - 1] = word(r["pmfude_rate"])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)


def _paph(c):
    out = np.zeros(NT_NZ + 1, dtype=np.float32)
    out[:NT_NZ] = c["paph"]
    out[NT_NZ] = c["paph_sfc"]
    return out


_GOT = {k: np_ntiedtke_updraft_scale(
    ldcum=bool(_COLS[k]["ldcum"]), ktype=_COLS[k]["ktype"],
    kcbot=_COLS[k]["kcbot"], kctop=_COLS[k]["kctop"],
    idtop=_COLS[k]["idtop"], loddraf=bool(_COLS[k]["loddraf"]),
    zmfub1=_COLS[k]["zmfub1"], zmfub=_COLS[k]["zmfub"],
    paph=_paph(_COLS[k]), pmfu=_COLS[k]["pmfu"], pmfus=_COLS[k]["pmfus"],
    pmfuq=_COLS[k]["pmfuq"], pmful=_COLS[k]["pmful"],
    pdmfup=_COLS[k]["pdmfup"], plude=_COLS[k]["plude"],
    pmfude_rate=_COLS[k]["pmfude_rate"], pmfd=_COLS[k]["pmfd"],
    pmfds=_COLS[k]["pmfds"], pmfdq=_COLS[k]["pmfdq"],
    pdmfdp=_COLS[k]["pdmfdp"], pmfdde_rate=_COLS[k]["pmfdde_rate"],
    ztmst=_ZTMST) for k in _KEYS}


@pytest.mark.parametrize("field", (*_OUT, "plude", "pmfude_rate",
                                   "pmfdde_rate"))
def test_level_outputs_are_bitwise(field):
    bad = []
    for key in _KEYS:
        g, e = _GOT[key][field], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_the_scalar_outputs_are_exact():
    """6.6 can switch a column off; 6.7 can push idtop below cloud top."""
    for key in _KEYS:
        for f in ("ldcum", "ktype", "idtop"):
            assert int(_GOT[key][f]) == _EXP[key][f], f"{f} {key}"


def test_the_closure_factor_actually_varies():
    """zmfs is the closure's answer applied. If it were 1 everywhere, this
    block would be an identity and §9's retention numbers would not reach
    the forecast at all.
    """
    vals = {round(float(_GOT[k]["zmfs"]), 6) for k in _KEYS
            if _COLS[k]["ldcum"]}
    assert len(vals) > 5, f"zmfs takes only {len(vals)} distinct values"
    assert any(v < 0.5 for v in vals), (
        "no column retains less than half its first-guess mass flux; the "
        "gray-zone damping this port exists to fix is not visible here")


# ===========================================================================
# The kernel
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_uscale():
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

    def fvec(name):
        return cp.asarray(np.array([float(_COLS[k][name]) for k in keys],
                                   dtype=np.float32))

    io_ = {n: pack(n) for n in ("pmfu", "pmfus", "pmfuq", "pmful",
                                "pdmfup", "plude", "pmfude_rate",
                                "pmfd", "pmfds", "pmfdq", "pdmfdp",
                                "pmfdde_rate")}
    d_ld = ivec("ldcum")
    d_kt = ivec("ktype")
    d_idt = ivec("idtop")

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_updraft_scale", (
        ivec("loddraf"), ivec("kcbot"), ivec("kctop"),
        fvec("zmfub1"), fvec("zmfub"), pack("paph", sfc="paph_sfc"),
        d_ld, d_kt, d_idt,
        io_["pmfu"], io_["pmfus"], io_["pmfuq"], io_["pmful"],
        io_["pdmfup"], io_["plude"], io_["pmfude_rate"],
        io_["pmfd"], io_["pmfds"], io_["pmfdq"], io_["pdmfdp"],
        io_["pmfdde_rate"],
        np.int32(ncol), np.int32(NT_NZ), _ZTMST,
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    return (keys, {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in io_.items()},
            cp.asnumpy(d_ld), cp.asnumpy(d_kt), cp.asnumpy(d_idt))


@pytest.mark.parametrize("field", (*_OUT, "plude", "pmfude_rate",
                                   "pmfdde_rate"))
def test_kernel_level_outputs_are_bitwise(cuda_uscale, field):
    """Graded against WRF, never against the mirror."""
    keys, got, _, _, _ = cuda_uscale
    bad = []
    for c, key in enumerate(keys):
        g, e = got[field][:, c], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_kernel_scalar_outputs_are_exact(cuda_uscale):
    keys, _, ld, kt, idt = cuda_uscale
    for c, key in enumerate(keys):
        assert int(ld[c]) == _EXP[key]["ldcum"], f"ldcum {key}"
        assert int(kt[c]) == _EXP[key]["ktype"], f"ktype {key}"
        assert int(idt[c]) == _EXP[key]["idtop"], f"idtop {key}"


def test_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function(
        "ntiedtke_updraft_scale").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
