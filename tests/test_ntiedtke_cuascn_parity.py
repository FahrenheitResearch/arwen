"""cuascn graded against the pinned WRF v4.6.1 oracle at ``max_ulp == 0``.

cuascn is the largest routine in New Tiedtke and the one that owns the
plume.  Everything it reads is captured at its own call site inside
``cumastrn`` -- see ``tools/ntiedtke_wrf461_oracle/nt_cumastrn_body.inc`` --
because eight of its dummies are read before they are written and
reconstructing those has failed five times in this port.

Three things in here are gates rather than assertions of the obvious, and
each exists because the alternative is a green suite that proves nothing:

``test_llo3_is_true_throughout``
    ``llo3`` is a TILE-WIDE monotone OR-reduction (cu_ntiedtke.F90:1994).
    Passing ``llo3=True`` to the mirror is only exact while some column in
    the tile carries a label at the first iteration.  This proves that
    precondition on the fixture instead of assuming it.

``test_the_fixture_actually_ascends``
    Green on a fixture where no column ever enters the plume would be
    worthless.  This counts the columns that do.

``test_pgeoh_and_paph_above_klev_are_never_read``
    Those two arrays are ``klev+1`` long in the reference and the capture
    carries only ``klev``.  The mirror is fed NaN in the missing slot, so a
    read poisons the answer and every comparison fails loudly.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_DXSWEEP, NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_cuascn

_IN1 = ("ptenh", "pqenh", "pqsenh", "ptu", "pqu", "plu", "pmfu", "pmfub_s",
        "pqsen", "pap", "paph", "pgeoh", "pgeo")
_IN2 = ("puen", "pven", "pten", "pqen", "pqte", "pverv", "puu", "pvu",
        "pmfus", "pmfuq", "pmful", "plude")
_OUT_LEV = ("ptu", "pqu", "plu", "pmfu", "pmfus", "pmfuq", "pmful",
            "plude", "pdmfup", "plglac", "pmfude_rate")
_OUT_INT = ("ldcum", "ktype", "kcbot", "kctop", "kctop0")

#: The dt run_nt_cumastrn.F90 drives cumastrn with.
_ZTMST = np.float32(60.0)


def _columns():
    """``{(case, dx): dict}`` -- every input cuascn reads, at its call site."""
    cols: dict[tuple[int, float], dict] = {}
    for r in load_csv("nt-cuascn-in2-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "klwmin": int(r["klwmin"]), "kctop0": int(r["kctop0"]),
            "kdpl": int(r["kdpl"]), "ldcum": int(r["ldcum"]),
            "ktype": int(r["ktype"]), "kcbot": int(r["kcbot"]),
            "kctop": int(r["kctop"]), "lndj": int(r["lndj"]),
            "wbase": word(r["wbase"]),
        }
    for name, fields in (("nt-cuascn-in-levels.csv", _IN1),
                         ("nt-cuascn-in2-levels.csv", _IN2)):
        for r in load_csv(name):
            slot = cols[(int(r["case"]), float(r["dx"]))]
            k = int(r["k"]) - 1
            for f in fields:
                slot.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = \
                    word(r[f])
    for r in load_csv("nt-cuascn-in-levels.csv"):
        slot = cols[(int(r["case"]), float(r["dx"]))]
        slot.setdefault("klab", np.zeros(NT_NZ, dtype=np.int32))[
            int(r["k"]) - 1] = int(r["klab"])
    return cols


def _expected():
    exp: dict[tuple[int, float], dict] = {}
    for r in load_csv("nt-cuascn-surface.csv"):
        exp[(int(r["case"]), float(r["dx"]))] = {
            **{f: int(r[f]) for f in _OUT_INT},
            "kdpl": int(r["kdpl"]), "wup": word(r["wup"]),
        }
    for r in load_csv("nt-cuascn-out-levels.csv"):
        slot = exp[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in (*_OUT_LEV, "ptenh_out", "pqenh_out"):
            slot.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = \
                word(r[f])
        slot.setdefault("klab", np.zeros(NT_NZ, dtype=np.int32))[k] = \
            int(r["klab"])
    return exp


def _edge(a):
    """A ``klev+1`` interface array whose surface slot is NaN.

    The capture carries k = 1..klev only.  Nothing in cuascn reads
    klev+1 -- every subscript is jk or jk+-1 with jk <= klevm1 -- and the
    NaN turns a mistake in that reading into a hard failure rather than a
    silent zero.
    """
    out = np.full(NT_NZ + 1, np.nan, dtype=np.float32)
    out[:NT_NZ] = a
    return out


def _run(col):
    return np_ntiedtke_cuascn(
        ptenh=col["ptenh"], pqenh=col["pqenh"], pten=col["pten"],
        pqen=col["pqen"], pqsen=col["pqsen"], pgeo=col["pgeo"],
        pgeoh=_edge(col["pgeoh"]), pap=col["pap"], paph=_edge(col["paph"]),
        ldcum=bool(col["ldcum"]), ktype=col["ktype"], klab=col["klab"],
        ptu=col["ptu"], pqu=col["pqu"], plu=col["plu"], pmfu=col["pmfu"],
        pmfub=col["pmfub_s"][0], pmfus=col["pmfus"], pmfuq=col["pmfuq"],
        pmful=col["pmful"], plude=col["plude"],
        pdmfup=np.zeros(NT_NZ, dtype=np.float32),
        plglac=np.zeros(NT_NZ, dtype=np.float32),
        kcbot=col["kcbot"], kctop=col["kctop"], kctop0=col["kctop0"],
        ztmst=_ZTMST, lndj=col["lndj"], wbase=col["wbase"],
        kdpl=col["kdpl"], pverv=col["pverv"])


_COLS = _columns()
_EXP = _expected()
_KEYS = sorted(_COLS)
_GOT = {k: _run(_COLS[k]) for k in _KEYS}


@pytest.mark.parametrize("dx", NT_DXSWEEP)
@pytest.mark.parametrize("field", _OUT_LEV)
def test_level_outputs_are_bitwise(dx, field):
    bad = []
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = _GOT[key][field], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key[0], d.tolist()[:5], g[d[0]], e[d[0]]))
    assert not bad, f"{field} @ dx={dx}: {bad[:3]}"


@pytest.mark.parametrize("dx", NT_DXSWEEP)
@pytest.mark.parametrize("field", ("ptenh", "pqenh"))
def test_the_negative_buoyancy_rewrite_is_bitwise(dx, field):
    """cuascn:2118-2119 rewrites ptenh/pqenh; grade what it produced."""
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = _GOT[key][field], _EXP[key][f"{field}_out"]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        assert not d.size, f"{field} case {key[0]} dx={dx} levels {d[:5]}"


@pytest.mark.parametrize("dx", NT_DXSWEEP)
def test_integer_outputs_and_klab_are_exact(dx):
    for key in (k for k in _KEYS if k[1] == dx):
        got, exp = _GOT[key], _EXP[key]
        for f in _OUT_INT:
            assert int(got[f]) == exp[f], f"{f} case {key[0]} dx={dx}"
        assert np.array_equal(got["klab"], exp["klab"]), \
            f"klab case {key[0]} dx={dx}"


@pytest.mark.parametrize("dx", NT_DXSWEEP)
def test_wup_is_bitwise(dx):
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = np.float32(_GOT[key]["wup"]), _EXP[key]["wup"]
        assert g.view(np.uint32) == e.view(np.uint32), \
            f"wup case {key[0]} dx={dx}: {g} vs {e}"


def test_llo3_is_true_throughout():
    """The precondition that makes ``llo3=True`` exact.

    ``is`` at the first iteration is the sum of ``klab(:,klev)`` over the
    whole tile; ``llo3`` never clears once set.  So one labelled column at
    jk = klevm1 makes llo3 true for every level and every column.  Every
    column entering cuascn with ldcum true carries klab(klev) > 0, which is
    a stronger statement and the one worth pinning: it means the flag is
    true whenever the routine has anything to do.

    If this ever fails, the kernel needs a block-wide OR reduction and the
    mirror needs llo3 threaded per level -- not a tolerance.
    """
    for dx in NT_DXSWEEP:
        keys = [k for k in _KEYS if k[1] == dx]
        labelled = sum(1 for k in keys if _COLS[k]["klab"][NT_NZ - 1] > 0)
        assert labelled > 0, f"dx={dx}: no column labelled at klev"
        for k in keys:
            if _COLS[k]["ldcum"]:
                assert _COLS[k]["klab"][NT_NZ - 1] > 0, \
                    f"ldcum column {k} enters with klab(klev) == 0"


def test_the_fixture_actually_ascends():
    """Green must not be reachable by every column doing nothing."""
    ascending = sum(1 for k in _KEYS if _EXP[k]["ldcum"]
                    and _EXP[k]["kctop"] < _EXP[k]["kcbot"])
    assert ascending >= 40, f"only {ascending} columns built a plume"
    precip = sum(1 for k in _KEYS if np.any(_EXP[k]["pdmfup"] != 0))
    assert precip >= 20, f"only {precip} columns produced precipitation"
    detrained = sum(1 for k in _KEYS if np.any(_EXP[k]["pmfude_rate"] != 0))
    assert detrained >= 20, f"only {detrained} columns detrained"


def test_pgeoh_and_paph_above_klev_are_never_read():
    """The NaN in the klev+1 slot must not reach any output.

    This is the same class of bug as the paph surface interface that was
    missed once already: an uncaptured edge silently defaulting to zero.
    Here it defaults to NaN, so this test is the receipt that it is unread.
    """
    for key in _KEYS:
        for f in (*_OUT_LEV, "ptenh", "pqenh"):
            assert not np.any(np.isnan(_GOT[key][f])), \
                f"{f} case {key} touched the klev+1 slot"
        assert not np.isnan(_GOT[key]["wup"])


# ===========================================================================
# The kernel, graded against the SAME oracle rows as the mirror
# ===========================================================================
# Not against the mirror.  Grading the kernel against the mirror would make
# a shared transcription error invisible; both are graded against WRF.


@pytest.fixture(scope="module")
def cuda_cuascn():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.ntiedtke import NtLaunchGeometry, NtStages

    keys = _KEYS
    ncol, n1 = len(keys), NT_NZ + 2

    def pack(name, dtype=np.float32, edge=False):
        a = np.zeros((n1, ncol), dtype=dtype)
        for c, k in enumerate(keys):
            x = _COLS[k][name]
            a[1:1 + NT_NZ, c] = x
            if edge:
                # klev+1 is not captured and must never be read; NaN so a
                # read poisons the answer instead of defaulting to zero.
                a[NT_NZ + 1, c] = np.nan
        return cp.asarray(a)

    def ivec(name):
        return cp.asarray(np.array([int(_COLS[k][name]) for k in keys],
                                   dtype=np.int32))

    def fvec(name):
        return cp.asarray(np.array([float(_COLS[k][name]) for k in keys],
                                   dtype=np.float32))

    ro = {n: pack(n) for n in ("pten", "pqen", "pqsen", "pgeo", "pap",
                               "pverv")}
    ro["pgeoh"] = pack("pgeoh", edge=True)
    ro["paph"] = pack("paph", edge=True)
    io_lev = {n: pack(n) for n in ("ptenh", "pqenh", "ptu", "pqu", "plu",
                                   "pmfu", "pmfus", "pmfuq", "pmful",
                                   "plude")}
    io_lev["pdmfup"] = cp.zeros((n1, ncol), dtype=np.float32)
    io_lev["plglac"] = cp.zeros((n1, ncol), dtype=np.float32)
    io_lev["pmfude_rate"] = cp.zeros((n1, ncol), dtype=np.float32)
    d_klab = pack("klab", dtype=np.int32)

    sc = {n: ivec(n) for n in ("ldcum", "ktype", "kcbot", "kctop",
                               "kctop0")}
    d_pmfub = cp.asarray(np.array(
        [float(_COLS[k]["pmfub_s"][0]) for k in keys], dtype=np.float32))
    d_wup = cp.zeros(ncol, dtype=np.float32)

    # The production descriptor, not an open-coded grid: a hand-rolled one
    # would keep passing after someone re-tiled the stage.
    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_cuascn", (
        ro["pten"], ro["pqen"], ro["pqsen"], ro["pgeo"], ro["pgeoh"],
        ro["pap"], ro["paph"], ro["pverv"],
        ivec("lndj"), ivec("kdpl"), fvec("wbase"),
        io_lev["ptenh"], io_lev["pqenh"], io_lev["ptu"], io_lev["pqu"],
        io_lev["plu"], io_lev["pmfu"], io_lev["pmfus"], io_lev["pmfuq"],
        io_lev["pmful"], io_lev["plude"], io_lev["pdmfup"],
        io_lev["plglac"], io_lev["pmfude_rate"], d_klab,
        sc["ldcum"], sc["ktype"], sc["kcbot"], sc["kctop"], sc["kctop0"],
        d_pmfub, d_wup,
        np.int32(ncol), np.int32(NT_NZ), _ZTMST, np.int32(1),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()

    out = {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in io_lev.items()}
    out["klab"] = cp.asnumpy(d_klab)[1:1 + NT_NZ, :]
    return keys, out, {n: cp.asnumpy(v) for n, v in sc.items()}, \
        cp.asnumpy(d_wup)


@pytest.mark.parametrize("field", (*_OUT_LEV, "ptenh", "pqenh"))
def test_kernel_level_outputs_are_bitwise(cuda_cuascn, field):
    keys, lev, _, _ = cuda_cuascn
    src = f"{field}_out" if field in ("ptenh", "pqenh") else field
    bad = []
    for c, key in enumerate(keys):
        g, e = lev[field][:, c], _EXP[key][src]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_kernel_integer_outputs_and_wup_are_exact(cuda_cuascn):
    keys, lev, sc, wup = cuda_cuascn
    for c, key in enumerate(keys):
        for f in _OUT_INT:
            assert int(sc[f][c]) == _EXP[key][f], f"{f} {key}"
        assert np.array_equal(lev["klab"][:, c], _EXP[key]["klab"]), \
            f"klab {key}"
        g, e = np.float32(wup[c]), _EXP[key]["wup"]
        assert g.view(np.uint32) == e.view(np.uint32), f"wup {key}"


def test_kernel_holds_no_local_frame():
    """cuascn is the largest routine in the scheme and still costs 0 B.

    The five klon x klev locals it declares would be a (nz+2, ncol) global
    array each -- 81 MiB at nz = 62 on a 372x284 domain.  Two are dead
    (zodetr is never assigned; pdmfen is written and never read) and the
    other three are strictly one-level lookback, so they are registers.
    Standing rule 3 is why this is a test and not a remark.
    """
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_cuascn").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]


def test_kernel_and_mirror_are_independently_graded():
    """Both are compared to WRF, never to each other.

    A shared transcription error is exactly what a mirror-to-kernel
    comparison cannot see, and this port has produced five errors that
    would have looked identical in both.
    """
    src = open("tests/test_ntiedtke_cuascn_parity.py", encoding="utf-8").read()
    assert "_EXP[key]" in src
    for fn in ("def test_kernel_level_outputs_are_bitwise",
               "def test_kernel_integer_outputs_and_wup_are_exact"):
        body = src[src.index(fn):]
        body = body[:body.index("\ndef ", 1)]
        assert "_GOT" not in body, f"{fn} grades against the mirror"
