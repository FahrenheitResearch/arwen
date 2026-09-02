"""cudlfsn graded against the pinned WRF v4.6.1 oracle at ``max_ulp == 0``.

The level of free sinking: where downdrafts start. Inputs are captured at
cudlfsn's own call site inside ``cumastrn``, and its outputs are captured
BEFORE ``cuddrafn`` runs so the two routines grade separately — the
pre-closure capture further down is the state after both and cannot
separate them.

Two gates in here are about things the oracle structurally cannot disagree
with, which is the class this port has been caught by five times:

``test_pud_and_pvd_are_never_written``
    Both are dummies of cudlfsn that its body never mentions. An oracle
    comparison cannot distinguish "correctly left alone" from "not
    implemented", so the mirror's signature is asserted not to carry them.

``test_the_is_zero_cycle_is_inert``
    cudlfsn has its own ``if (is == 0) cycle`` reduction over all columns
    (:2448). Unlike ``cuascn``'s ``llo3`` this one is inert, and the reason
    is checked here rather than asserted in a comment.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_DXSWEEP, NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_cudlfsn

_IN_LEV = ("ptenh", "pqenh", "puen", "pven", "pten", "pqsen", "pgeo",
           "pgeoh", "paph", "ptu", "pqu", "plu", "puu", "pvu",
           # the six class-2 slots, at cudlfsn's own entry
           "ptd_in", "pqd_in", "pmfd_in", "pmfds_in", "pmfdq_in",
           "pdmfdp_in")
_OUT_LEV = ("ptd", "pqd", "pud", "pvd", "pmfd", "pmfds", "pmfdq", "pdmfdp")
#: What the mirror actually produces.  pud/pvd are excluded on purpose --
#: see test_pud_and_pvd_are_never_written.
_GRADED = ("ptd", "pqd", "pmfd", "pmfds", "pmfdq", "pdmfdp")


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-cudlfsn-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "ktype": int(r["ktype"]),
            "kcbot": int(r["kcbot"]), "kctop": int(r["kctop"]),
            "lndj": int(r["lndj"]), "pmfub": word(r["pmfub"]),
            "prfl": word(r["prfl"]),
        }
    for r in load_csv("nt-cudlfsn-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _IN_LEV:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-cudlfsn-out-surface.csv"):
        exp[(int(r["case"]), float(r["dx"]))] = {
            "kdtop": int(r["kdtop"]), "lddraf": int(r["lddraf"]),
            "prfl": word(r["prfl_out"]),
        }
    for r in load_csv("nt-cudlfsn-out-levels.csv"):
        s = exp[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _OUT_LEV:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    return cols, exp


def _edge(a):
    """A klev+1 interface array whose surface slot is NaN.

    Same discipline as the cuascn fixture: the capture carries k = 1..klev,
    and a read of the uncaptured slot must poison the answer rather than
    silently default to zero.
    """
    out = np.full(NT_NZ + 1, np.nan, dtype=np.float32)
    out[:NT_NZ] = a
    return out


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)
_GOT = {k: np_ntiedtke_cudlfsn(
    kcbot=_COLS[k]["kcbot"], kctop=_COLS[k]["kctop"], lndj=_COLS[k]["lndj"],
    ldcum=bool(_COLS[k]["ldcum"]), ptenh=_COLS[k]["ptenh"],
    pqenh=_COLS[k]["pqenh"], pten=_COLS[k]["pten"], pqsen=_COLS[k]["pqsen"],
    pgeo=_COLS[k]["pgeo"], pgeoh=_edge(_COLS[k]["pgeoh"]),
    paph=_edge(_COLS[k]["paph"]), ptu=_COLS[k]["ptu"], pqu=_COLS[k]["pqu"],
    pmfub=_COLS[k]["pmfub"], prfl=_COLS[k]["prfl"],
    ptd=_COLS[k]["ptd_in"], pqd=_COLS[k]["pqd_in"],
    pmfd=_COLS[k]["pmfd_in"], pmfds=_COLS[k]["pmfds_in"],
    pmfdq=_COLS[k]["pmfdq_in"], pdmfdp=_COLS[k]["pdmfdp_in"])
    for k in _KEYS}


@pytest.mark.parametrize("dx", NT_DXSWEEP)
@pytest.mark.parametrize("field", _GRADED)
def test_level_outputs_are_bitwise(dx, field):
    bad = []
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = _GOT[key][field], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key[0], d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field} @ dx={dx}: {bad[:3]}"


@pytest.mark.parametrize("dx", NT_DXSWEEP)
def test_scalars_are_exact(dx):
    """kdtop, lddraf and the UPDATED prfl, on their own axis.

    prfl is graded separately from the fields for the reason §12 records:
    a scalar reached through a different reduction is not covered by the
    field sweep around it, and that cost 1 ULP in cuascn's wup with 84 of
    85 assertions green.
    """
    for key in (k for k in _KEYS if k[1] == dx):
        got, exp = _GOT[key], _EXP[key]
        assert int(got["kdtop"]) == exp["kdtop"], f"kdtop {key}"
        assert int(got["lddraf"]) == exp["lddraf"], f"lddraf {key}"
        g, e = np.float32(got["prfl"]), exp["prfl"]
        assert g.view(np.uint32) == e.view(np.uint32), \
            f"prfl {key}: {g} vs {e}"


def test_the_fixture_actually_sinks():
    """Green must not be reachable by no column ever finding an LFS."""
    found = sum(1 for k in _KEYS if _EXP[k]["lddraf"])
    assert found >= 10, f"only {found} columns found a level of free sinking"
    evap = sum(1 for k in _KEYS
               if not np.array_equal(_GOT[k]["prfl"], _COLS[k]["prfl"]))
    assert evap >= 10, f"only {evap} columns evaporated any precipitation"


def test_pud_and_pvd_are_never_written():
    """cudlfsn's body never mentions pud or pvd; downdraft momentum is
    cududvn's job.

    THE ORACLE CANNOT DISAGREE WITH THIS. Both slots come out of the
    reference carrying whatever the caller had, so "correctly left alone"
    and "not implemented" are the same bytes. So the gate is on the
    mirror's shape instead: it must not accept or return either name, and
    a future edit that starts writing them fails here rather than passing
    a comparison that was never able to see it.
    """
    import inspect
    src = inspect.getsource(np_ntiedtke_cudlfsn)
    sig = inspect.signature(np_ntiedtke_cudlfsn).parameters
    for name in ("pud", "pvd"):
        assert name not in sig, f"{name} is not an input to cudlfsn"
        assert f'"{name}"' not in src, f"cudlfsn must not return {name}"


def test_the_is_zero_cycle_is_inert():
    """cudlfsn:2448's `if (is == 0) cycle` is a horizontal reduction that
    does not change any answer -- unlike cuascn's llo3, which does.

    The argument has three limbs and all three are checked against the
    mirror's structure, because reasoning by analogy to llo3 is exactly
    what this port has been burned by:

    1. ztenwb/zqenwb/zph are assigned for EVERY column before the cycle;
    2. cuadjtqn is masked by the same per-column llo2 the cycle sums;
    3. everything after it is inside `if (llo2(jl))`.

    So the skipped work is a no-op for every column it skips. The mirror
    therefore drops the reduction entirely and uses the per-column flag,
    and this test pins that that is deliberate.
    """
    import inspect
    src = inspect.getsource(np_ntiedtke_cudlfsn)
    body = src[src.index("for jk in range(3, ike + 1):"):]
    pre = body[:body.index("if not llo2:")]
    for name in ("ztenwb[jk]", "zqenwb[jk]", "zph"):
        assert name in pre, f"{name} must be set before the per-column exit"
    # Nothing after the exit may run for a column with llo2 false: the
    # mirror expresses that as an early `continue`, not a mask.
    assert "continue" in body[:body.index("nt_cuadjtqn2")]


# ===========================================================================
# The kernel, graded against the SAME oracle rows as the mirror
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_cudlfsn():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.ntiedtke import NtLaunchGeometry, NtStages

    keys = _KEYS
    ncol, n1 = len(keys), NT_NZ + 2

    def pack(name, sfc=None):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, k in enumerate(keys):
            a[1:1 + NT_NZ, c] = _COLS[k][name]
            if sfc is not None:
                a[1 + NT_NZ, c] = sfc
        return cp.asarray(a)

    def ivec(name):
        return cp.asarray(np.array([int(_COLS[k][name]) for k in keys],
                                   dtype=np.int32))

    def fvec(name):
        return cp.asarray(np.array([float(_COLS[k][name]) for k in keys],
                                   dtype=np.float32))

    ro = {n: pack(n) for n in ("ptenh", "pqenh", "pten", "pqsen", "pgeo",
                               "pgeoh", "paph", "ptu", "pqu")}
    # THE CLASS-2 SLOTS ARE SEEDED FROM THE ENTRY CAPTURE, NOT ZEROED.
    # The kernel writes each at exactly one level; every other level must
    # come out carrying the caller's value, and a zero-filled buffer here
    # would make that indistinguishable from the kernel doing nothing.
    io_ = {n: pack(f"{n}_in") for n in ("ptd", "pqd", "pmfd", "pmfds",
                                        "pmfdq", "pdmfdp")}
    d_prfl = fvec("prfl")
    d_kdtop = cp.zeros(ncol, dtype=np.int32)
    d_lddraf = cp.zeros(ncol, dtype=np.int32)

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_cudlfsn", (
        ivec("ldcum"), ivec("kcbot"), ivec("kctop"),
        ro["ptenh"], ro["pqenh"], ro["pten"], ro["pqsen"], ro["pgeo"],
        ro["pgeoh"], ro["paph"], ro["ptu"], ro["pqu"], fvec("pmfub"),
        io_["ptd"], io_["pqd"], io_["pmfd"], io_["pmfds"], io_["pmfdq"],
        io_["pdmfdp"], d_prfl, d_kdtop, d_lddraf,
        np.int32(ncol), np.int32(NT_NZ),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    return (keys, {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in io_.items()},
            cp.asnumpy(d_prfl), cp.asnumpy(d_kdtop), cp.asnumpy(d_lddraf))


@pytest.mark.parametrize("field", _GRADED)
def test_kernel_level_outputs_are_bitwise(cuda_cudlfsn, field):
    """Graded against WRF, never against the mirror."""
    keys, lev, _, _, _ = cuda_cudlfsn
    bad = []
    for c, key in enumerate(keys):
        g, e = lev[field][:, c], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_kernel_scalars_are_exact(cuda_cudlfsn):
    keys, _, prfl, kdtop, lddraf = cuda_cudlfsn
    for c, key in enumerate(keys):
        assert int(kdtop[c]) == _EXP[key]["kdtop"], f"kdtop {key}"
        assert int(lddraf[c]) == _EXP[key]["lddraf"], f"lddraf {key}"
        g, e = np.float32(prfl[c]), _EXP[key]["prfl"]
        assert g.view(np.uint32) == e.view(np.uint32), f"prfl {key}"


def test_the_kernel_LEAVES_untouched_class2_levels_alone(cuda_cudlfsn):
    """The reflex this kernel must not have.

    Zeroing outputs at entry is what almost every CUDA kernel does. Here
    it is wrong on every level but one: cudlfsn writes each of its six
    level outputs at exactly ONE line inside the LFS branch. This checks
    the untouched levels came out carrying the ENTRY value rather than
    zero -- which the oracle comparison above can only see because the
    entry values are non-zero (section 14's limb 2).
    """
    keys, lev, _, _, _ = cuda_cudlfsn
    witnesses = 0
    for c, key in enumerate(keys):
        entry = _COLS[key]["ptd_in"]
        got = lev["ptd"][:, c]
        same = (got.view(np.uint32) == entry.view(np.uint32)) & (entry != 0)
        witnesses += int(np.count_nonzero(same))
    assert witnesses > 1000, (
        f"only {witnesses} untouched non-zero ptd slots survived; either "
        "the kernel is zeroing outputs it must leave alone, or the fixture "
        "stopped being able to tell the difference")


def test_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_cudlfsn").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
