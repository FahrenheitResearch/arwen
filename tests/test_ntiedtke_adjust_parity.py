"""cumastrn:833-919 graded at ``max_ulp == 0`` -- the adjustments block.

Between cuflxn and cudtdqn. Five things: the downdraft stability cap, its
application (with a precipitation correction that runs downward through the
column), the entrainment-rate floors, and two humidity guards.

Its inputs are cuflxn's captured outputs and its outputs are cudtdqn's
captured entry state plus one small new capture, so no new replication was
needed -- the same bracketing property §22 records.

NOT the momentum rescale. That is :996-1016, off a different ``zmfs``
computed against a different limit. The two use the same local name, and
this port attributed one range's job to the other once already.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_adjust

_ZTMST = np.float32(60.0)

#: cuflxn's exit fields, which are this block's entry.
_FROM_CUFLXN = ("pmfu", "pmfd", "pmfds", "pmfdq", "pmfuq", "pmful",
                "plude", "pdmfup", "pdmfdp", "pmfdde_rate",
                "pmflxr", "pmflxs")
_OUT = ("pmfd", "pmfds", "pmfdq", "pdmfdp", "pdmfup", "pmfdde_rate",
        "pmfude_rate")


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-cuflxn-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "loddraf": int(r["lddraf"]),
            "kctop": int(r["kctop"]), "kcbot": int(r["kcbot"]),
            "paph_sfc": word(r["paph_sfc"])}
    for r in load_csv("nt-cuflxn-out-surface.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        s["idtop"] = int(r["kdtop_out"])
        s["pmflxr_sfc"] = word(r["pmflxr_sfc"])
        s["pmflxs_sfc"] = word(r["pmflxs_sfc"])
    for r in load_csv("nt-cuflxn-out-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _FROM_CUFLXN:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-cuflxn-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in ("pqen", "paph"):
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    # pmfude_rate at THIS block's entry is cuflxn's exit, not cuascn's:
    # the 6.5 updraft rescale at :746-819 runs between them and scales it.
    # Sourcing it from cuascn came out 1.26x low on 42 columns, which is
    # the sixth time "a neighbour's capture will do" has been wrong here.
    for r in load_csv("nt-cuflxn-out-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        s.setdefault("pmfude_rate_in",
                     np.zeros(NT_NZ, dtype=np.float32))[
            int(r["k"]) - 1] = word(r["pmfude_rate"])
    for r in load_csv("nt-adjust-out-surface.csv"):
        exp[(int(r["case"]), float(r["dx"]))] = {
            "prsfc": word(r["prsfc"]), "pssfc": word(r["pssfc"])}
    for r in load_csv("nt-adjust-out-levels.csv"):
        s = exp[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _OUT:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    # plude's exit is cudtdqn's captured entry -- nothing between them
    # touches it.
    for r in load_csv("nt-cudtdqn-in-levels.csv"):
        s = exp[(int(r["case"]), float(r["dx"]))]
        s.setdefault("plude", np.zeros(NT_NZ, dtype=np.float32))[
            int(r["k"]) - 1] = word(r["plude"])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)


def _edge(a, sfc):
    out = np.empty(NT_NZ + 1, dtype=np.float32)
    out[:NT_NZ] = a
    out[NT_NZ] = sfc
    return out


def _run(key):
    c = _COLS[key]
    return np_ntiedtke_adjust(
        ldcum=bool(c["ldcum"]), loddraf=bool(c["loddraf"]),
        idtop=c["idtop"], kctop=c["kctop"], kcbot=c["kcbot"],
        ztmst=_ZTMST, paph=_edge(c["paph"], c["paph_sfc"]),
        pqen=c["pqen"], pmfu=c["pmfu"], pmfd=c["pmfd"], pmfds=c["pmfds"],
        pmfdq=c["pmfdq"], pmfuq=c["pmfuq"], pmful=c["pmful"],
        plude=c["plude"], pdmfup=c["pdmfup"], pdmfdp=c["pdmfdp"],
        pmfdde_rate=c["pmfdde_rate"],
        pmfude_rate=c["pmfude_rate_in"],
        pmflxr=_edge(c["pmflxr"], c["pmflxr_sfc"]),
        pmflxs=_edge(c["pmflxs"], c["pmflxs_sfc"]))


_GOT = {k: _run(k) for k in _KEYS}


@pytest.mark.parametrize("field", _OUT)
def test_level_outputs_are_bitwise(field):
    bad = []
    for key in _KEYS:
        g, e = _GOT[key][field], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_the_surface_fluxes_are_bitwise():
    """prsfc/pssfc are the scheme's actual surface rain and snow."""
    for key in _KEYS:
        for f in ("prsfc", "pssfc"):
            g, e = np.float32(_GOT[key][f]), _EXP[key][f]
            assert g.view(np.uint32) == e.view(np.uint32), f"{f} {key}"


def test_the_stability_cap_is_exercised():
    """:838-861 IS covered -- measured, 30 columns of 108.

    zmfs is the largest factor keeping |pmfd| under 0.98*pmfu at every
    level, and the application block is guarded by `zmfs < 1`. This was
    written toward the gap on the assumption it might not fire; it does,
    so the assertion is inverted to REQUIRE it -- which is what a
    direction-of-the-gap test is for.

    It matters because the application block carries `zmfuub`, an
    accumulator running DOWNWARD through the column that hands the
    precipitation a capped downdraft no longer transports to pmflxr at
    the level below. An untested accumulator of that shape is exactly
    what the cuentrn degeneracy was.
    """
    capped = [k for k in _KEYS if float(_GOT[k]["zmfs"]) < 1.0]
    assert len(capped) >= 20, (
        f"only {len(capped)} columns need the downdraft stability cap; "
        ":849-861 and the zmfuub correction are no longer meaningfully "
        "exercised, so this slice would be grading its guards only")


# ===========================================================================
# The kernel
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_adjust():
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

    ro = {n: pack(n) for n in ("pqen", "pmfu", "pmfuq", "pmful")}
    io_ = {n: pack(n) for n in ("pmfd", "pmfds", "pmfdq", "plude",
                                "pdmfup", "pdmfdp", "pmfdde_rate")}
    io_["pmfude_rate"] = pack("pmfude_rate_in")
    io_["pmflxr"] = pack("pmflxr", sfc="pmflxr_sfc")
    io_["pmflxs"] = pack("pmflxs", sfc="pmflxs_sfc")
    d_prsfc = cp.zeros(ncol, dtype=np.float32)
    d_pssfc = cp.zeros(ncol, dtype=np.float32)

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_adjust", (
        ivec("ldcum"), ivec("loddraf"), ivec("idtop"), ivec("kctop"),
        ivec("kcbot"), pack("paph", sfc="paph_sfc"),
        ro["pqen"], ro["pmfu"], ro["pmfuq"], ro["pmful"],
        io_["pmfd"], io_["pmfds"], io_["pmfdq"], io_["plude"],
        io_["pdmfup"], io_["pdmfdp"], io_["pmfdde_rate"],
        io_["pmfude_rate"], io_["pmflxr"], io_["pmflxs"],
        d_prsfc, d_pssfc,
        np.int32(ncol), np.int32(NT_NZ), _ZTMST,
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    return (keys, {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in io_.items()},
            cp.asnumpy(d_prsfc), cp.asnumpy(d_pssfc))


@pytest.mark.parametrize("field", (*_OUT, "plude"))
def test_kernel_level_outputs_are_bitwise(cuda_adjust, field):
    """Graded against WRF, never against the mirror."""
    keys, got, _, _ = cuda_adjust
    bad = []
    for c, key in enumerate(keys):
        g, e = got[field][:, c], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_kernel_surface_fluxes_are_bitwise(cuda_adjust):
    keys, _, prsfc, pssfc = cuda_adjust
    for c, key in enumerate(keys):
        for f, arr in (("prsfc", prsfc), ("pssfc", pssfc)):
            g, e = np.float32(arr[c]), _EXP[key][f]
            assert g.view(np.uint32) == e.view(np.uint32), f"{f} {key}"


def test_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_adjust").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
