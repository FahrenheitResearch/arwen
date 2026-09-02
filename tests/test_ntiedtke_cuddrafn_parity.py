"""cuddrafn graded against the pinned WRF v4.6.1 oracle at ``max_ulp == 0``.

The moist downdraft descent. Inputs are captured at cuddrafn's own call
site: its six class-1 dummies are cudlfsn's outputs and nothing runs
between the two calls, so cudlfsn's exit capture would have served -- but
stitching one routine's exit into another's entry is the reconstruction
this port keeps being burned by, so it is captured again.

``paph[klev+1]`` is load-bearing here and captured. cuascn never reads it
and that fixture poisons the slot with NaN; cuddrafn reads it three times
(:2618, :2648, :2649), which is why the two fixtures treat it oppositely.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_DXSWEEP, NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_cuddrafn

_IN_LEV = ("ptenh", "pqenh", "pgeo", "pgeoh", "paph", "pmfu",
           "ptd", "pqd", "pmfd", "pmfds", "pmfdq", "pdmfdp")
#: The post-cuddrafn capture, which is the closure's own pre-state.
_OUT_MAP = {"ptd": "ztd", "pqd": "zqd", "pmfd": "zmfd", "pmfds": "zmfds",
            "pmfdq": "zmfdq", "pdmfdp": "zdmfdp",
            "pmfdde_rate": "pmfdde_rate"}


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-cuddrafn-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "lddraf": int(r["lddraf"]), "prfl": word(r["prfl"]),
            "paph_sfc": word(r["paph_sfc"])}
    for r in load_csv("nt-cuddrafn-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in _IN_LEV:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    for r in load_csv("nt-cuddrafn-out-surface.csv"):
        exp[(int(r["case"]), float(r["dx"]))] = {"prfl": word(r["prfl_out"])}
    for r in load_csv("nt-downdraft-levels.csv"):
        s = exp[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for ours, theirs in _OUT_MAP.items():
            s.setdefault(ours, np.zeros(NT_NZ, dtype=np.float32))[k] = \
                word(r[theirs])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)


def _paph(col):
    """klev+1 entries: the surface interface is READ here, not poisoned."""
    out = np.empty(NT_NZ + 1, dtype=np.float32)
    out[:NT_NZ] = col["paph"]
    out[NT_NZ] = col["paph_sfc"]
    return out


_GOT = {k: np_ntiedtke_cuddrafn(
    lddraf=bool(_COLS[k]["lddraf"]), ptenh=_COLS[k]["ptenh"],
    pqenh=_COLS[k]["pqenh"], pgeo=_COLS[k]["pgeo"], pgeoh=_COLS[k]["pgeoh"],
    paph=_paph(_COLS[k]), pmfu=_COLS[k]["pmfu"], ptd=_COLS[k]["ptd"],
    pqd=_COLS[k]["pqd"], pmfd=_COLS[k]["pmfd"], pmfds=_COLS[k]["pmfds"],
    pmfdq=_COLS[k]["pmfdq"], pdmfdp=_COLS[k]["pdmfdp"],
    prfl=_COLS[k]["prfl"]) for k in _KEYS}


@pytest.mark.parametrize("dx", NT_DXSWEEP)
@pytest.mark.parametrize("field", tuple(_OUT_MAP))
def test_level_outputs_are_bitwise(dx, field):
    bad = []
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = _GOT[key][field], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key[0], d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field} @ dx={dx}: {bad[:3]}"


@pytest.mark.parametrize("dx", NT_DXSWEEP)
def test_the_updated_prfl_is_bitwise(dx):
    """prfl on its own axis: it is an accumulator, not a field."""
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = np.float32(_GOT[key]["prfl"]), _EXP[key]["prfl"]
        assert g.view(np.uint32) == e.view(np.uint32), \
            f"prfl {key}: {g} vs {e}"


def test_the_fixture_actually_descends():
    """Green must not be reachable by no column running a downdraft."""
    active = sum(1 for k in _KEYS if _COLS[k]["lddraf"])
    assert active >= 10, f"only {active} columns have lddraf"
    detr = sum(1 for k in _KEYS if np.any(_GOT[k]["pmfdde_rate"] != 0))
    assert detr >= 10, f"only {detr} columns detrained"


def test_the_buoyancy_shutoff_is_a_NAMED_COVERAGE_GAP():
    """:2703 is transcribed and NOT exercised.  Measured, not assumed.

        if (zbuo >= 0 .or. prfl <= pmfd*zcond) then
          pmfd(jl,jk) = 0.

    A column that trips this has pmfd zeroed at jk, so llo2 is false at
    jk+1 and the descent stops -- the downdraft terminates before the
    surface.  MEASURED: 0 of the 42 columns with an active downdraft do
    that; every one runs to the surface.  So the branch is graded only in
    the sense that its guard is evaluated, and the assignment inside it is
    NOT covered by any fixture row.

    This assertion is written in the direction of the gap, exactly as the
    cuentrn degeneracy was: it FAILS when a future case table starts
    exercising the branch.  That failure is the signal to invert it and
    say so, rather than letting a silent coverage gain look like it had
    always been there.  Deleting the assertion would lose that.
    """
    terminated = 0
    for k in _KEYS:
        pm = _GOT[k]["pmfd"]
        idx = np.nonzero(pm != 0)[0]
        if idx.size and idx[-1] < len(pm) - 1:
            terminated += 1
    assert terminated == 0, (
        f"{terminated} columns now terminate their downdraft before the "
        "surface, so cuddrafn:2703's shut-off IS exercised.  That is a "
        "coverage GAIN: invert this assertion to require it, and remove "
        "the gap from the excluded list in docs/ntiedtke/PORT-RECORD.md.")


def test_the_surface_interface_is_load_bearing():
    """paph[klev+1] must actually change the answer.

    cuascn's fixture poisons this slot with NaN because cuascn never reads
    it. cuddrafn reads it three times. If perturbing it changed nothing,
    one of those two claims would be wrong.
    """
    key = next(k for k in _KEYS if _COLS[k]["lddraf"]
               and np.any(_GOT[k]["pmfd"] != 0))
    col = _COLS[key]
    bad = _paph(col).copy()
    bad[NT_NZ] = np.float32(bad[NT_NZ] * np.float32(1.05))
    other = np_ntiedtke_cuddrafn(
        lddraf=True, ptenh=col["ptenh"], pqenh=col["pqenh"],
        pgeo=col["pgeo"], pgeoh=col["pgeoh"], paph=bad, pmfu=col["pmfu"],
        ptd=col["ptd"], pqd=col["pqd"], pmfd=col["pmfd"],
        pmfds=col["pmfds"], pmfdq=col["pmfdq"], pdmfdp=col["pdmfdp"],
        prfl=col["prfl"])
    assert not np.array_equal(other["pmfd"], _GOT[key]["pmfd"]), \
        "perturbing paph[klev+1] changed nothing; it is not load-bearing"


def test_pud_and_pvd_are_never_written():
    """As cudlfsn: intent(inout) dummies the body never mentions.

    The oracle cannot disagree -- "correctly left alone" and "not
    implemented" are the same bytes -- so the gate is on the mirror's
    shape.
    """
    import inspect
    src = inspect.getsource(np_ntiedtke_cuddrafn)
    sig = inspect.signature(np_ntiedtke_cuddrafn).parameters
    for name in ("pud", "pvd"):
        assert name not in sig, f"{name} is not an input to cuddrafn"
        assert f'"{name}"' not in src, f"cuddrafn must not return {name}"


# ===========================================================================
# The kernel, graded against the SAME oracle rows as the mirror
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_cuddrafn():
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

    ro = {n: pack(n) for n in ("ptenh", "pqenh", "pgeo", "pgeoh", "pmfu")}
    # The six class-1 slots, seeded from the entry capture.
    io_ = {n: pack(n) for n in ("ptd", "pqd", "pmfd", "pmfds", "pmfdq",
                                "pdmfdp")}
    d_rate = cp.zeros((n1, ncol), dtype=np.float32)
    d_prfl = cp.asarray(np.array([float(_COLS[k]["prfl"]) for k in keys],
                                 dtype=np.float32))
    d_ldd = cp.asarray(np.array([int(_COLS[k]["lddraf"]) for k in keys],
                                dtype=np.int32))

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_cuddrafn", (
        d_ldd, ro["ptenh"], ro["pqenh"], ro["pgeo"], ro["pgeoh"],
        pack("paph", sfc="paph_sfc"), ro["pmfu"],
        io_["ptd"], io_["pqd"], io_["pmfd"], io_["pmfds"], io_["pmfdq"],
        io_["pdmfdp"], d_rate, d_prfl,
        np.int32(ncol), np.int32(NT_NZ),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    out = {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in io_.items()}
    out["pmfdde_rate"] = cp.asnumpy(d_rate)[1:1 + NT_NZ, :]
    return keys, out, cp.asnumpy(d_prfl)


@pytest.mark.parametrize("field", tuple(_OUT_MAP))
def test_kernel_level_outputs_are_bitwise(cuda_cuddrafn, field):
    """Graded against WRF, never against the mirror."""
    keys, got, _ = cuda_cuddrafn
    bad = []
    for c, key in enumerate(keys):
        g, e = got[field][:, c], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_kernel_prfl_is_bitwise(cuda_cuddrafn):
    keys, _, prfl = cuda_cuddrafn
    for c, key in enumerate(keys):
        g, e = np.float32(prfl[c]), _EXP[key]["prfl"]
        assert g.view(np.uint32) == e.view(np.uint32), f"prfl {key}"


def test_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_cuddrafn").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
