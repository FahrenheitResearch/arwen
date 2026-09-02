"""New Tiedtke prep stage, graded bitwise against WRF v4.6.1.

Three things are on trial here and the first is the reason this stage is
graded before any physics:

1. THE VERTICAL FLIP.  cu_ntiedtke_pre_run reverses the column, so the
   scheme runs TOP-DOWN where gf.cu and kf.cu run bottom-up.  An
   upside-down column produces finite, plausible, entirely wrong numbers --
   there is no crash and no NaN to notice -- so the inversion is graded
   first and on its own.

2. scale_fac / scale_fac2, the whole reason this scheme is being ported.
   Both branches, and the discontinuity at dx == 15000 where the test is
   `<` and not `<=`.

3. THE FP-CONTRACTION HAZARD.  The CUDA arm runs all 108 columns in ONE
   launch, spanning both arms of the dx branch.  ptxas contracts by local
   register pressure, so a runtime branch can leave two clones of the same
   arithmetic rounding differently; putting both arms in one launch is what
   makes that visible here rather than somewhere non-obvious later.

max_ulp == 0 throughout.  There is no honest reason for it to be anything
else, and a tolerance would hide exactly the mixed-precision and
reassociation mistakes this stage exists to prevent.
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import (
    NT_DT, NT_DXSWEEP, NT_GRAV, NT_ITIMESTEP, NT_NZ, NT_STEPCU,
    prep_expected, prep_inputs, prep_surface, stage_a_surface,
)
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_prep, np_ntiedtke_scale_fac


def _stages(ncol, nz):
    """One descriptor, the same one the production launcher builds.

    The parity tests deliberately do NOT open-code a grid: if they did, they
    would keep passing after someone re-tiled a stage, which is the failure
    the descriptor exists to prevent.
    """
    from gpuwm.core.ntiedtke import NtLaunchGeometry, NtStages
    return NtStages(NtLaunchGeometry(ncol=ncol, nz=nz))


_FIELDS = ("prsl", "ghtl", "omg", "tf", "qvf", "qcf", "qif", "uf", "vf",
           "qvftenz", "thftenz", "prsi", "ghti")


def _bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def _differing(got, want):
    return int(np.count_nonzero(_bits(got) != _bits(want)))


@pytest.fixture(scope="module")
def fixture():
    inp, exp, sur, sa = (prep_inputs(), prep_expected(),
                         prep_surface(), stage_a_surface())
    assert set(inp) == set(exp) == set(sur), "fixture keys disagree"
    assert len(inp) == 18 * len(NT_DXSWEEP)
    return inp, exp, sur, sa


def _run_mirror(key, inp, sur, sa):
    i, s, a = inp[key], sur[key], sa[key]
    return np_ntiedtke_prep(
        t3d=i["t3d"], qv3d=i["qv3d"], qc3d=i["qc3d"], qi3d=i["qi3d"],
        u3d=i["u3d"], v3d=i["v3d"], pcps=i["pcps"], p8w=i["p8w"],
        dz8w=i["dz8w"], rho3d=i["rho3d"], w=i["w"],
        qvften=i["qvften"], thften=i["thften"],
        xland=a["xland"], hfx=a["hfx"], qfx=a["qfx"], dx=s["dx_hv"],
        dt=NT_DT, stepcu=NT_STEPCU, itimestep=NT_ITIMESTEP, grav=NT_GRAV,
    )


# ---------------------------------------------------------------------------
# the scale factor, on its own
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dx", NT_DXSWEEP)
def test_scale_factors_are_bitwise(dx, fixture):
    _, _, _, sa = fixture
    want = sa[(1, dx)]
    sf, sf2 = np_ntiedtke_scale_fac(dx)
    assert _bits(sf) == _bits(want["scale_fac"]), (
        f"scale_fac at dx={dx}: {sf!r} vs {want['scale_fac']!r}")
    assert _bits(sf2) == _bits(want["scale_fac2"])


def test_the_branch_at_dxref_is_strict_and_discontinuous(fixture):
    """dx == 15000 takes the ELSE arm, and the two arms disagree there.

    This is the single column in the sweep that catches a port which
    transcribes `dx < dxref` as `dx <= dxref`.  If the branch were `<=`,
    dx = 15000 would take the log arm and give 1.06133**3 = 1.1956; WRF
    gives 1 + 1.33e-5*15000 = 1.1995.
    """
    _, _, _, sa = fixture
    sf_at, _ = np_ntiedtke_scale_fac(15000.0)
    assert _bits(sf_at) == _bits(sa[(1, 15000.0)]["scale_fac"])
    below = np.float32(1.06133) ** 3
    assert not np.isclose(float(sf_at), float(below), rtol=0, atol=1e-6), (
        "the two arms agree at the join; the discontinuity this test "
        "exists to pin is gone")


def test_scale_fac_is_not_monotonic_above_the_reference(fixture):
    """27 km damps MORE than 15 km, because the coarse arm grows with dx."""
    _, _, _, sa = fixture
    assert (float(sa[(1, 27000.0)]["scale_fac"])
            > float(sa[(1, 15000.0)]["scale_fac"]))


# ---------------------------------------------------------------------------
# the NumPy mirror
# ---------------------------------------------------------------------------

def test_mirror_reproduces_every_prep_word(fixture):
    inp, exp, sur, sa = fixture
    bad = {}
    for key in sorted(exp):
        got = _run_mirror(key, inp, sur, sa)
        for f in _FIELDS:
            n = _differing(got[f][:NT_NZ], exp[key][f])
            if n:
                bad[(key, f)] = n
    assert not bad, f"differing words: {dict(list(bad.items())[:10])}"


def test_mirror_reproduces_slimsk_and_delt(fixture):
    inp, exp, sur, sa = fixture
    for key in sorted(sur):
        got = _run_mirror(key, inp, sur, sa)
        assert got["slimsk"] == sur[key]["slimsk"], key
        assert _bits(got["delt"]) == _bits(sur[key]["delt"]), key


def test_the_flip_actually_inverts(fixture):
    """A guard against the mirror and the oracle being wrong the same way.

    prsl is monotonically DECREASING in the scheme's index (k = 0 is the
    model top), which is the opposite of the WRF-order input.  If both the
    reference and the mirror silently dropped the flip, every bitwise test
    above would still pass; this one would not.
    """
    inp, exp, _, _ = fixture
    key = (1, 4500.0)
    assert np.all(np.diff(exp[key]["prsl"]) > 0), (
        "scheme-order pressure must INCREASE with index (index 0 = top)")
    assert np.all(np.diff(inp[key]["pcps"]) < 0), (
        "WRF-order pressure must DECREASE with index (index 0 = surface)")
    assert _bits(exp[key]["prsl"][0]) == _bits(inp[key]["pcps"][NT_NZ - 1])


# ---------------------------------------------------------------------------
# the CUDA kernel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cuda_prep(fixture):
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module

    inp, exp, sur, sa = fixture
    keys = sorted(exp)
    ncol, nz = len(keys), NT_NZ

    def pack(field, nk):
        a = np.empty((nk, ncol), dtype=np.float32)
        for c, key in enumerate(keys):
            a[:, c] = inp[key][field][:nk]
        return cp.asarray(a)

    d_in = {f: pack(f, nz) for f in
            ("t3d", "qv3d", "qc3d", "qi3d", "u3d", "v3d", "pcps",
             "dz8w", "rho3d", "qvften", "thften")}
    d_in["p8w"] = pack("p8w", nz + 1)
    d_in["w"] = pack("w", nz + 1)

    scal = {n: cp.asarray(np.array(
        [float(sa[k]["xland"] if n == "xland" else
               sa[k]["hfx"] if n == "hfx" else
               sa[k]["qfx"] if n == "qfx" else sur[k]["dx_hv"])
         for k in keys], dtype=np.float32))
        for n in ("xland", "hfx", "qfx", "dx")}

    out = {f: cp.zeros((nz, ncol), dtype=np.float32) for f in _FIELDS
           if f not in ("prsi", "ghti")}
    out["prsi"] = cp.zeros((nz + 1, ncol), dtype=np.float32)
    out["ghti"] = cp.zeros((nz + 1, ncol), dtype=np.float32)
    d_slimsk = cp.zeros(ncol, dtype=np.int32)
    d_sf = cp.zeros(ncol, dtype=np.float32)
    d_sf2 = cp.zeros(ncol, dtype=np.float32)
    d_delt = cp.zeros(ncol, dtype=np.float32)

    stages = _stages(ncol, nz)
    stages.launch("ntiedtke_prep", (
        d_in["t3d"], d_in["qv3d"], d_in["qc3d"], d_in["qi3d"],
        d_in["u3d"], d_in["v3d"], d_in["pcps"], d_in["dz8w"],
        d_in["rho3d"], d_in["p8w"], d_in["w"],
        d_in["qvften"], d_in["thften"],
        scal["xland"], scal["hfx"], scal["qfx"], scal["dx"],
        out["prsl"], out["ghtl"], out["omg"], out["tf"], out["qvf"],
        out["qcf"], out["qif"], out["uf"], out["vf"],
        out["qvftenz"], out["thftenz"], out["prsi"], out["ghti"],
        d_slimsk, d_sf, d_sf2, d_delt,
        np.int32(ncol), np.int32(nz), np.float32(NT_DT),
        np.int32(NT_STEPCU), np.int32(NT_ITIMESTEP), np.float32(NT_GRAV)))
    stages.check_geometry()
    cp.cuda.Stream.null.synchronize()

    return (keys, {f: cp.asnumpy(v) for f, v in out.items()},
            cp.asnumpy(d_slimsk), cp.asnumpy(d_sf), cp.asnumpy(d_sf2),
            cp.asnumpy(d_delt))


def test_kernel_holds_no_local_frame():
    """The prep must cost nothing in the launch-time reservation.

    MEASURED: the 79 column arrays cumastrn needs compile to a 15,496 B
    frame at nz = 49 as function-scope locals, which reserves 1,483.9 MiB
    under (frame - 1024) x 107,520.  The prep holds no column array at all
    and must stay that way, so the row can never price above the 1,024 B
    default stack.
    """
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    attrs = load_module("ntiedtke").get_function("ntiedtke_prep").attributes
    assert attrs["local_size_bytes"] == 0, attrs["local_size_bytes"]


def test_cuda_reproduces_every_prep_word(cuda_prep, fixture):
    _, exp, _, _ = fixture
    keys, out, _, _, _, _ = cuda_prep
    bad = {}
    for c, key in enumerate(keys):
        for f in _FIELDS:
            n = _differing(out[f][:NT_NZ, c], exp[key][f])
            if n:
                bad[(key, f)] = n
    assert not bad, f"differing words: {dict(list(bad.items())[:10])}"


def test_cuda_matches_the_mirror_exactly(cuda_prep, fixture):
    """Kernel and mirror must agree word for word, not merely both match.

    They are graded against the same oracle, so this is redundant while
    everything passes -- and it is the test that localises the fault when
    something stops passing, because it separates 'the transcription is
    wrong' from 'the mirror and the kernel disagree about association'.
    """
    inp, exp, sur, sa = fixture
    keys, out, slimsk, sf, sf2, delt = cuda_prep
    for c, key in enumerate(keys):
        ref = _run_mirror(key, inp, sur, sa)
        for f in _FIELDS:
            assert _differing(out[f][:NT_NZ, c], ref[f][:NT_NZ]) == 0, (key, f)
        assert slimsk[c] == ref["slimsk"], key
        assert _bits(sf[c]) == _bits(ref["scale_fac"]), key
        assert _bits(sf2[c]) == _bits(ref["scale_fac2"]), key
        assert _bits(delt[c]) == _bits(ref["delt"]), key


def test_cuda_scale_factors_survive_both_branches_in_one_launch(cuda_prep,
                                                                fixture):
    """The FP-contraction guard.

    All 108 columns run in a single launch and dx spans both arms of the
    branch in nt_scale_factors.  If ptxas cloned the surrounding arithmetic
    per arm and contracted the clones differently, the columns on one side
    would drift from the oracle while the other side stayed exact.
    """
    _, _, _, sa = fixture
    keys, _, _, sf, sf2, _ = cuda_prep
    below = [c for c, k in enumerate(keys) if k[1] < 15000.0]
    at_or_above = [c for c, k in enumerate(keys) if k[1] >= 15000.0]
    assert below and at_or_above, "the sweep must span the branch"
    for c in below + at_or_above:
        want = sa[keys[c]]
        assert _bits(sf[c]) == _bits(want["scale_fac"]), keys[c]
        assert _bits(sf2[c]) == _bits(want["scale_fac2"]), keys[c]


# ===========================================================================
# Slice 2: the conversion block and cuinin
# ===========================================================================
# cuinin is COLUMN-UNIVERSAL -- no ldcum, no ktype, no ierr -- and runs at
# cumastrn:474, before cutypen:490 decides the convection type.  So it is
# graded against this same 108-column fixture with no decomposition harness.
# The oracle reaches it through objcopy --globalize-symbol, which build.sh
# proves leaves .text byte-identical.

from gpuwm.verify.ntiedtke_oracle import (            # noqa: E402
    conv_expected, cuinin_expected, cuinin_surface,
)
from gpuwm.verify.ntiedtke_ref import (               # noqa: E402
    NtConstants, np_ntiedtke_convert, np_ntiedtke_cuinin,
)

_CONV_F = ("ztp1", "zqp1", "zqsat", "pgeo", "pverv", "ptte", "pqte")
_CUININ_F = ("ptenh", "pqenh", "ptu", "pqu", "ptd", "pqd",
             "puu", "pvu", "pud", "pvd", "plu")


@pytest.fixture(scope="module")
def slice2(fixture):
    inp, exp, sur, sa = fixture
    cv, ci, cs = conv_expected(), cuinin_expected(), cuinin_surface()
    assert set(cv) == set(ci) == set(cs) == set(exp)
    return cv, ci, cs


def _conv_mirror(key, exp, c):
    """Run the conversion mirror off the PREP fixture's own outputs.

    Chaining slice 2 onto slice 1's recorded words rather than onto the
    mirror's is deliberate: it keeps a fault in the prep from showing up
    here as a conversion fault.
    """
    p = exp[key]
    ghti_full = np.append(p["ghti"], np.float32(0.0))
    return np_ntiedtke_convert(
        tf=p["tf"], qvf=p["qvf"], uf=p["uf"], vf=p["vf"], omg=p["omg"],
        ghtl=p["ghtl"], ghti=ghti_full, prsl=p["prsl"],
        qvftenz=p["qvftenz"], thftenz=p["thftenz"], c=c)


def test_conversion_mirror_is_bitwise(fixture, slice2):
    """foeewm's exp is the thing on trial here.

    np.exp on a float32 is NumPy's own single-precision path; glibc's expf
    evaluates in double and rounds once.  The mirror models glibc the same
    way it does for logf, and the kernel calls gfk_exp, which IS glibc.
    """
    _, exp, _, _ = fixture
    cv, _, _ = slice2
    c = NtConstants()
    bad = {}
    for key in sorted(cv):
        got = _conv_mirror(key, exp, c)
        for f in _CONV_F:
            n = _differing(got[f], cv[key][f])
            if n:
                bad[(key, f)] = n
        n = _differing(got["pgeoh"][:NT_NZ], cv[key]["pgeoh"])
        if n:
            bad[(key, "pgeoh")] = n
    assert not bad, f"differing words: {dict(list(bad.items())[:8])}"


def test_cuinin_mirror_is_bitwise(slice2):
    cv, ci, cs = slice2
    c = NtConstants()
    bad = {}
    for key in sorted(ci):
        v = cv[key]
        got = np_ntiedtke_cuinin(
            pten=v["ztp1"], pqen=v["zqp1"], pqsen=v["zqsat"],
            puen=v["ztp1"] * 0, pven=v["ztp1"] * 0,
            pverv=v["pverv"], pgeo=v["pgeo"],
            paph=_paph(cv, key), pgeoh=v["pgeoh"], c=c)
        for f in ("ptenh", "pqenh", "ptu", "pqu", "ptd", "pqd", "plu"):
            n = _differing(got[f], ci[key][f])
            if n:
                bad[(key, f)] = n
        # pqsenh[0] is UNDEFINED in the reference -- never written.
        n = _differing(got["pqsenh"][1:], ci[key]["pqsenh"][1:])
        if n:
            bad[(key, "pqsenh")] = n
        assert got["klwmin"] == cs[key], key
        assert np.array_equal(got["klab"], ci[key]["klab"]), key
    assert not bad, f"differing words: {dict(list(bad.items())[:8])}"


def _paph(cv, key):
    """cuinin reads paph only at 1..klev-1, so the prep fixture's nz entries
    are enough and the uncaptured surface interface is never touched."""
    from gpuwm.verify.ntiedtke_oracle import prep_expected
    return prep_expected()[key]["prsi"]


def test_pqsenh_index_zero_is_never_written(slice2):
    """A finding, pinned so it cannot be quietly 'fixed'.

    cuinin's jk loop runs 2..klev and its tail block writes only ptenh(1)
    and pqenh(1), so pqsenh(1) is left at whatever the caller's array held.
    It is a cumastrn local in WRF and nothing downstream reads it.  A port
    that helpfully initialises it is not wrong, but a port GRADED on it
    would be grading uninitialised memory -- so the exclusion is explicit
    here rather than implicit in a slice.
    """
    import inspect
    from gpuwm.verify import ntiedtke_ref
    src = inspect.getsource(ntiedtke_ref.np_ntiedtke_cuinin)
    assert "pqsenh[0] IS NEVER WRITTEN" in src


# ---------------------------------------------------------------------------
# the CUDA arm
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cuda_slice2(fixture, slice2):
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module

    _, exp, _, _ = fixture
    cv, ci, cs = slice2
    keys = sorted(ci)
    ncol, nz = len(keys), NT_NZ

    def pack(src, field, nk):
        a = np.empty((nk, ncol), dtype=np.float32)
        for c, key in enumerate(keys):
            a[:, c] = src[key][field][:nk]
        return cp.asarray(a)

    d_pten = pack(cv, "ztp1", nz)
    d_pqen = pack(cv, "zqp1", nz)
    d_pqsen = pack(cv, "zqsat", nz)
    d_pverv = pack(cv, "pverv", nz)
    d_pgeo = pack(cv, "pgeo", nz)
    d_zero = cp.zeros((nz, ncol), dtype=np.float32)
    # paph/pgeoh are (nz+1) in the kernel; the fixture captured nz, and
    # cuinin never reads index nz, so the tail row is padding.
    paph = np.zeros((nz + 1, ncol), dtype=np.float32)
    pgeoh = np.zeros((nz + 1, ncol), dtype=np.float32)
    for c, key in enumerate(keys):
        paph[:nz, c] = exp[key]["prsi"]
        pgeoh[:nz, c] = cv[key]["pgeoh"]
    d_paph, d_pgeoh = cp.asarray(paph), cp.asarray(pgeoh)

    out = {f: cp.zeros((nz, ncol), dtype=np.float32) for f in _CUININ_F}
    out["pqsenh"] = cp.zeros((nz, ncol), dtype=np.float32)
    d_klab = cp.zeros((nz, ncol), dtype=np.int32)
    d_klwmin = cp.zeros(ncol, dtype=np.int32)

    stages = _stages(ncol, nz)
    stages.launch("ntiedtke_cuinin", (
        d_pten, d_pqen, d_pqsen, d_zero, d_zero, d_pverv, d_pgeo,
        d_paph, d_pgeoh,
        out["ptenh"], out["pqenh"], out["pqsenh"],
        out["ptu"], out["pqu"], out["ptd"], out["pqd"],
        out["puu"], out["pvu"], out["pud"], out["pvd"], out["plu"],
        d_klab, d_klwmin,
        np.int32(ncol), np.int32(nz),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    stages.check_geometry()
    cp.cuda.Stream.null.synchronize()
    return keys, {f: cp.asnumpy(v) for f, v in out.items()}, \
        cp.asnumpy(d_klab), cp.asnumpy(d_klwmin)


def test_slice2_kernels_hold_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    m = load_module("ntiedtke")
    for name in ("ntiedtke_convert", "ntiedtke_cuinin"):
        a = m.get_function(name).attributes
        assert a["local_size_bytes"] == 0, (name, a["local_size_bytes"])


def test_cuda_cuinin_is_bitwise(cuda_slice2, slice2):
    _, ci, cs = slice2
    keys, out, klab, klwmin = cuda_slice2
    bad = {}
    for c, key in enumerate(keys):
        for f in ("ptenh", "pqenh", "ptu", "pqu", "ptd", "pqd", "plu"):
            n = _differing(out[f][:, c], ci[key][f])
            if n:
                bad[(key, f)] = n
        n = _differing(out["pqsenh"][1:, c], ci[key]["pqsenh"][1:])
        if n:
            bad[(key, "pqsenh")] = n
        assert klwmin[c] == cs[key], key
        assert np.array_equal(klab[:, c], ci[key]["klab"]), key
    assert not bad, f"differing words: {dict(list(bad.items())[:8])}"


# ===========================================================================
# Slice 3: cutypen -- the trigger
# ===========================================================================
# cutypen decides ktype, and ktype selects which scale factor applies
# downstream (:676 scale_fac for deep, :716 scale_fac2 for shallow).  It is
# the branchiest routine in the scheme, and the one the whole port turns on.
#
# It assigns ktype 0, 1 and 2 ONLY.  ktype 3 (mid-level) comes later, from
# cubasmcn via cuascn:1968, and has no coverage yet -- recorded in
# docs/ntiedtke/PORT-RECORD.md as an open gap rather than left to be discovered.

from gpuwm.verify.ntiedtke_oracle import (              # noqa: E402
    cutypen_expected, cutypen_surface,
)
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_cutypen   # noqa: E402

_CT_F = ("cutu", "cuqu", "culu")


@pytest.fixture(scope="module")
def slice3(fixture, slice2):
    inp, exp, sur, sa = fixture
    cv, ci, cs = slice2
    ct, ctsur = cutypen_expected(), cutypen_surface()
    assert set(ct) == set(ctsur) == set(ci)
    return cv, ci, ct, ctsur


def _cutypen_args(key, exp, cv, ci, sa):
    v, u = cv[key], ci[key]
    a = sa[key]
    return dict(
        pqen=v["zqp1"], ptenh=u["ptenh"], pqenh=u["pqenh"],
        pqsenh=u["pqsenh"],
        pgeoh=np.append(v["pgeoh"], np.float32(0.0)),
        paph=np.append(exp[key]["prsi"], np.float32(0.0)),
        hfx=a["hfx"], qfx=a["qfx"], pgeo=v["pgeo"], pqsen=v["zqsat"],
        pap=exp[key]["prsl"], pten=v["ztp1"],
        lndj=int(abs(float(a["xland"]) - 2.0)),
        cutu=u["ptu"], cuqu=u["pqu"], culab=u["klab"], culu=u["plu"])


def test_the_fixture_spans_all_three_ktype_outcomes(slice3):
    """cutypen's whole output space, in the fixture, before anything grades.

    Deep is the arm this port exists for.  It had SIX columns until the
    trigger capture made ktype visible and the case table could be tuned
    against a measurement -- five cases believed deep turned out shallow.
    A graded cutypen on that fixture would have left the deep path untested.
    """
    _, _, _, ctsur = slice3
    seen = collections.Counter(v["ktype"] for v in ctsur.values())
    assert seen[1] >= 24, f"deep arm too thin to grade: {dict(seen)}"
    assert seen[2] >= 6, f"no shallow control: {dict(seen)}"
    assert seen[0] >= 24, f"no rejected columns: {dict(seen)}"


def test_cutypen_mirror_is_bitwise(fixture, slice3):
    _, exp, _, sa = fixture
    cv, ci, ct, ctsur = slice3
    bad, scal = {}, {}
    for key in sorted(ct):
        got = np_ntiedtke_cutypen(**_cutypen_args(key, exp, cv, ci, sa))
        for f in _CT_F:
            n = _differing(got[f], ct[key][f])
            if n:
                bad[(key, f)] = n
        if not np.array_equal(got["culab"], ct[key]["culab"]):
            bad[(key, "culab")] = "int mismatch"
        e = ctsur[key]
        for f in ("ktype", "cubot", "cutop", "kdpl"):
            if got[f] != e[f]:
                scal[(key, f)] = (got[f], e[f])
        if int(got["ldcum"]) != e["ldcum"]:
            scal[(key, "ldcum")] = (got["ldcum"], e["ldcum"])
        if _bits(got["wbase"]) != _bits(e["wbase"]):
            scal[(key, "wbase")] = (got["wbase"], e["wbase"])
    assert not scal, f"scalar mismatches: {dict(list(scal.items())[:8])}"
    assert not bad, f"differing words: {dict(list(bad.items())[:8])}"


@pytest.fixture(scope="module")
def cuda_cutypen(fixture, slice3):
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module

    _, exp, _, sa = fixture
    cv, ci, ct, ctsur = slice3
    keys = sorted(ct)
    ncol, nz = len(keys), NT_NZ
    n1 = nz + 2                       # 1-based, index 0 unused

    def pack1(get):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, key in enumerate(keys):
            v = np.asarray(get(key), dtype=np.float32)
            a[1:1 + min(len(v), n1 - 1), c] = v[:n1 - 1]
        return cp.asarray(a)

    d = {
        "pqen": pack1(lambda k: cv[k]["zqp1"]),
        "ptenh": pack1(lambda k: ci[k]["ptenh"]),
        "pqenh": pack1(lambda k: ci[k]["pqenh"]),
        "pgeoh": pack1(lambda k: cv[k]["pgeoh"]),
        "paph": pack1(lambda k: exp[k]["prsi"]),
        "pgeo": pack1(lambda k: cv[k]["pgeo"]),
        "pqsen": pack1(lambda k: cv[k]["zqsat"]),
        "pap": pack1(lambda k: exp[k]["prsl"]),
        "pten": pack1(lambda k: cv[k]["ztp1"]),
        "cutu": pack1(lambda k: ci[k]["ptu"]),
        "cuqu": pack1(lambda k: ci[k]["pqu"]),
        "culu": pack1(lambda k: ci[k]["plu"]),
    }
    culab = np.zeros((n1, ncol), dtype=np.int32)
    for c, key in enumerate(keys):
        culab[1:1 + nz, c] = ci[key]["klab"]
    d_culab = cp.asarray(culab)
    d_hfx = cp.asarray(np.array([float(sa[k]["hfx"]) for k in keys],
                                dtype=np.float32))
    d_qfx = cp.asarray(np.array([float(sa[k]["qfx"]) for k in keys],
                                dtype=np.float32))
    d_scr = cp.zeros(10 * n1 * ncol, dtype=np.float32)
    d_scri = cp.zeros(n1 * ncol, dtype=np.int32)
    outs = {n: cp.zeros(ncol, dtype=np.int32)
            for n in ("ldcum", "ktype", "cubot", "cutop", "kdpl")}
    d_wbase = cp.zeros(ncol, dtype=np.float32)

    stages = _stages(ncol, nz)
    stages.launch("ntiedtke_cutypen", (
        d["pqen"], d["ptenh"], d["pqenh"], d["pgeoh"], d["paph"],
        d["pgeo"], d["pqsen"], d["pap"], d["pten"], d_hfx, d_qfx,
        d["cutu"], d["cuqu"], d["culu"], d_culab, d_scr, d_scri,
        outs["ldcum"], outs["ktype"], outs["cubot"], outs["cutop"],
        outs["kdpl"], d_wbase,
        np.int32(ncol), np.int32(nz),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    stages.check_geometry()
    cp.cuda.Stream.null.synchronize()

    return (keys,
            {f: cp.asnumpy(d[f])[1:1 + nz, :] for f in _CT_F},
            cp.asnumpy(d_culab)[1:1 + nz, :],
            {n: cp.asnumpy(v) for n, v in outs.items()},
            cp.asnumpy(d_wbase))


def test_cutypen_kernel_holds_no_local_frame():
    """11 per-column working arrays, and none of them on the stack.

    At nz = 49 they would be 2,156 B of frame -- over the 1,024 B default
    stack, so ~118 MiB of launch-time reservation under
    (frame - 1024) x 107,520.  They live in caller-provided global scratch
    instead and the frame stays 0 B.
    """
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_cutypen").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]


def test_cuda_cutypen_is_bitwise_across_all_ktype_arms(cuda_cutypen, slice3):
    """All 108 columns in ONE launch, spanning ktype 0, 1 and 2.

    This is the FP-contraction guard for the branchiest routine in the
    scheme.  ptxas contracts by local register pressure, so a runtime
    branch can leave two clones of the same arithmetic rounding
    differently.  cutypen's shallow and deep ascents are near-identical in
    shape, which is exactly the situation that invites two clones -- so the
    kernel writes ONE ascent body taking a `deep` flag, and this test puts
    both arms in the same launch where a divergence would show.
    """
    _, _, ct, ctsur = slice3
    keys, lev, culab, outs, wbase = cuda_cutypen
    bad, scal = {}, {}
    for c, key in enumerate(keys):
        for f in _CT_F:
            n = _differing(lev[f][:, c], ct[key][f])
            if n:
                bad[(key, f)] = n
        if not np.array_equal(culab[:, c], ct[key]["culab"]):
            bad[(key, "culab")] = "int mismatch"
        e = ctsur[key]
        for f in ("ktype", "cubot", "cutop", "kdpl", "ldcum"):
            if int(outs[f][c]) != e[f]:
                scal[(key, f)] = (int(outs[f][c]), e[f])
        if _bits(wbase[c]) != _bits(e["wbase"]):
            scal[(key, "wbase")] = (wbase[c], e["wbase"])
    assert not scal, f"scalar mismatches: {dict(list(scal.items())[:8])}"
    assert not bad, f"differing words: {dict(list(bad.items())[:8])}"

    seen = collections.Counter(int(outs["ktype"][c])
                               for c in range(len(keys)))
    assert {0, 1, 2} <= set(seen), (
        f"the launch did not span all three arms: {dict(seen)}")


# ===========================================================================
# Slice 4a: cubasmcn and cuentrn -- the mid-level arm
# ===========================================================================
# cubasmcn assigns ktype = 3, so this slice closes the coverage gap the
# cutypen slice left open: the fixture now runs 36 deep, 6 shallow, 60
# mid-level and 6 rejected, and every ktype value the scheme produces is
# exercised.
#
# ALL THIRTEEN of cubasmcn's outputs are conditionally written, so a column
# that does not trigger keeps the caller's value in every one of them.  The
# mirror mutates in place and the kernel leaves untriggered slots alone;
# both sides of that guard are graded here, on 60 triggering and 48
# non-triggering columns.

from gpuwm.verify.ntiedtke_oracle import (               # noqa: E402
    midlevel_expected, midlevel_surface, cuentrn_expected,
)
from gpuwm.verify.ntiedtke_ref import (                  # noqa: E402
    np_ntiedtke_cubasmcn, np_ntiedtke_cuentrn,
)

_ML_F = ("ptu", "pqu", "plu", "pmfu", "pmfus", "pmfuq", "pmful",
         "pdmfup", "plrain")


def _prologue(key, cv, ct, cts):
    """cuascn's own prologue (:1903-1952), which builds cubasmcn's inputs.

    Consumes cumastrn:500-541's outputs -- the possibly-cleared ldcum and
    the first-guess pmfub -- rather than cutypen's ldcum and pmfub = 0.
    That block runs BETWEEN cutypen and cuascn and is not optional: with
    pmfub = 0 every mass-flux quantity downstream is structurally zero, and
    the capture grades green while meaning nothing.

    The prologue itself is NOT proven here -- it is cuascn's, and the
    cuascn slice proves it when the full routine runs.  What is graded in
    this file is cubasmcn and cuentrn GIVEN this state.
    """
    from gpuwm.verify.ntiedtke_oracle import mfub_surface
    nz = NT_NZ
    v, t, e = cv[key], ct[key], cts[key]
    m = mfub_surface()[key]
    st = {
        "ldcum": [m["ldcum"] != 0], "ktype": [e["ktype"]],
        "kcbot": [e["cubot"]], "pmfub": [np.float32(m["zmfub"])],
        "ptu": t["cutu"].copy(), "pqu": t["cuqu"].copy(),
        "plu": t["culu"].copy(),
        "klab": t["culab"].copy().astype(np.int32),
    }
    if not st["ldcum"][0]:
        st["ktype"][0] = 0
        st["kcbot"][0] = -1
        st["pmfub"][0] = np.float32(0.0)
        st["pqu"][nz - 1] = np.float32(0.0)
    for n in ("pmfu", "pmfus", "pmfuq", "pmful", "pdmfup", "plrain"):
        st[n] = np.zeros(nz, dtype=np.float32)
    for k in range(nz):
        if (k + 1) != st["kcbot"][0]:
            st["plu"][k] = np.float32(0.0)
        if (not st["ldcum"][0]) or st["ktype"][0] == 3:
            st["klab"][k] = 0
    if st["ktype"][0] == 3:
        st["ldcum"][0] = False
    if st["ldcum"][0]:
        ikb = st["kcbot"][0]
        st["pmfu"][ikb - 1] = st["pmfub"][0]
        st["pmfus"][ikb - 1] = np.float32(
            st["pmfub"][0] * np.float32(np.float32(NtConstants().cpd
                                                   * st["ptu"][ikb - 1])
                                        + np.float32(v["pgeoh"][ikb - 1])))
        st["pmfuq"][ikb - 1] = np.float32(st["pmfub"][0] * st["pqu"][ikb - 1])
        st["pmful"][ikb - 1] = np.float32(st["pmfub"][0] * st["plu"][ikb - 1])
    return st


@pytest.fixture(scope="module")
def slice4(slice3):
    cv, ci, ct, ctsur = slice3
    ml, mls, en = midlevel_expected(), midlevel_surface(), cuentrn_expected()
    assert set(ml) == set(mls) == set(ct)
    return cv, ct, ctsur, ml, mls, en


def test_the_fixture_now_spans_every_ktype_the_scheme_produces(slice4):
    """The mid-level gap, closed and pinned.

    cutypen produces only 0/1/2; ktype = 3 comes from cubasmcn.  Until this
    slice the fixture had ZERO mid-level columns, so nothing graded touched
    that arm.  A narrowing of this spread would silently un-cover it again.
    """
    *_, mls, _ = slice4
    seen = collections.Counter(v["ktype"] for v in mls.values())
    assert set(seen) == {0, 1, 2, 3}, f"not all ktype arms present: {seen}"
    assert seen[3] >= 24, f"mid-level arm too thin: {dict(seen)}"


def test_cubasmcn_mirror_is_bitwise(slice4):
    cv, ct, ctsur, ml, mls, _ = slice4
    c = NtConstants()
    bad, scal = {}, {}
    for key in sorted(ml):
        v, st = cv[key], _prologue(key, cv, ct, ctsur)
        pgeoh = np.append(v["pgeoh"], np.float32(0.0))
        for kk in range(NT_NZ - 1, 2, -1):
            np_ntiedtke_cubasmcn(
                kk, pten=v["ztp1"], pqen=v["zqp1"], pqsen=v["zqsat"],
                pverv=v["pverv"], pgeo=v["pgeo"], pgeoh=pgeoh,
                ldcum=st["ldcum"], ktype=st["ktype"], klab=st["klab"],
                plrain=st["plrain"], pmfu=st["pmfu"], pmfub=st["pmfub"],
                kcbot=st["kcbot"], ptu=st["ptu"], pqu=st["pqu"],
                plu=st["plu"], pmfus=st["pmfus"], pmfuq=st["pmfuq"],
                pmful=st["pmful"], pdmfup=st["pdmfup"], c=c)
        for f in _ML_F:
            n = _differing(st[f], ml[key][f])
            if n:
                bad[(key, f)] = n
        if not np.array_equal(st["klab"], ml[key]["klab"]):
            bad[(key, "klab")] = "int mismatch"
        e = mls[key]
        if st["ktype"][0] != e["ktype"]:
            scal[(key, "ktype")] = (st["ktype"][0], e["ktype"])
        if st["kcbot"][0] != e["kcbot"]:
            scal[(key, "kcbot")] = (st["kcbot"][0], e["kcbot"])
        if _bits(st["pmfub"][0]) != _bits(e["pmfub"]):
            scal[(key, "pmfub")] = (st["pmfub"][0], e["pmfub"])
    assert not scal, f"scalar mismatches: {dict(list(scal.items())[:8])}"
    assert not bad, f"differing words: {dict(list(bad.items())[:8])}"


def test_cuentrn_mirror_is_bitwise_and_the_arithmetic_is_now_covered(slice4):
    """cuentrn, with its arithmetic actually exercised.

    This test used to assert its own fixture was DEGENERATE.  Under the old
    pre-state `pmfu` was zero everywhere it was read -- `pmfub` comes from
    cumastrn:500-541, which was not transcribed -- so all 4,968 captured
    rows were correctly-produced zeros, grading the guards (`ldwork`,
    `ldcum`, `kk < kcbot`) and NOT the `0.75e-4` multiply.

    That block has now landed, `pmfub` is non-zero on 42 columns, and the
    degeneracy assertion fired on its own, which is what it was for: it
    told the next person that something changed rather than letting a
    silent coverage gain look like it had always been there.

    So the assertion is inverted: the fixture must now carry non-zero
    `pdmfde` rows, and a regression to all-zeros would mean the mass-flux
    prerequisite came unwired again.

    `pdmfen` stays identically zero, and that is NOT a fixture weakness:
    `zentr` is set to zero and never reassigned, so it is zero by
    construction in v4.6.1.  Asserted separately, so a later WRF reviving
    `zentr` breaks this rather than passing quietly.
    """
    cv, ct, ctsur, ml, mls, en = slice4
    c = NtConstants()
    bad = 0
    nonzero_de = 0
    nonzero_en = 0
    for key in sorted(en):
        v, st = cv[key], _prologue(key, cv, ct, ctsur)
        pgeoh = np.append(v["pgeoh"], np.float32(0.0))
        for kk in range(NT_NZ - 1, 2, -1):
            e0, d0 = [np.float32(0.0)], [np.float32(0.0)]
            np_ntiedtke_cuentrn(kk, kcbot=st["kcbot"], ldcum=st["ldcum"],
                                ldwork=True, pgeoh=pgeoh, pmfu=st["pmfu"],
                                pdmfen=e0, pdmfde=d0, c=c)
            w = en[key].get(kk)
            if w is None:
                continue
            if _bits(e0[0]) != _bits(w[0]) or _bits(d0[0]) != _bits(w[1]):
                bad += 1
            nonzero_en += int(float(w[0]) != 0.0)
            nonzero_de += int(float(w[1]) != 0.0)
    assert bad == 0, f"{bad} cuentrn rows differ"
    assert nonzero_de > 0, (
        "every pdmfde row is zero again -- the cumastrn:500-541 mass-flux "
        "prerequisite has come unwired, and this test is back to grading "
        "guards only")
    assert nonzero_en == 0, (
        "pdmfen is no longer identically zero.  zentr is set to zero and "
        "never reassigned in v4.6.1, so this means the reference changed")


@pytest.fixture(scope="module")
def cuda_midlevel(slice4):
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module

    cv, ct, ctsur, ml, mls, en = slice4
    keys = sorted(ml)
    ncol, nz = len(keys), NT_NZ
    n1 = nz + 2

    def pack1(get):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, key in enumerate(keys):
            x = np.asarray(get(key), dtype=np.float32)
            a[1:1 + min(len(x), n1 - 1), c] = x[:n1 - 1]
        return cp.asarray(a)

    st = [_prologue(k, cv, ct, ctsur) for k in keys]
    d_in = {
        "pten": pack1(lambda k: cv[k]["ztp1"]),
        "pqen": pack1(lambda k: cv[k]["zqp1"]),
        "pqsen": pack1(lambda k: cv[k]["zqsat"]),
        "pverv": pack1(lambda k: cv[k]["pverv"]),
        "pgeo": pack1(lambda k: cv[k]["pgeo"]),
        "pgeoh": pack1(lambda k: cv[k]["pgeoh"]),
    }

    def pack_state(name, dtype=np.float32):
        a = np.zeros((n1, ncol), dtype=dtype)
        for c, s in enumerate(st):
            a[1:1 + nz, c] = s[name]
        return cp.asarray(a)

    io_f = {n: pack_state(n) for n in
            ("ptu", "pqu", "plu", "pmfu", "pmfus", "pmfuq", "pmful",
             "pdmfup", "plrain")}
    d_klab = pack_state("klab", np.int32)
    d_ldcum = cp.asarray(np.array([int(s["ldcum"][0]) for s in st],
                                  dtype=np.int32))
    d_ktype = cp.asarray(np.array([s["ktype"][0] for s in st],
                                  dtype=np.int32))
    d_kcbot = cp.asarray(np.array([s["kcbot"][0] for s in st],
                                  dtype=np.int32))
    d_pmfub = cp.asarray(np.array([float(s["pmfub"][0]) for s in st],
                                  dtype=np.float32))
    d_en = cp.zeros((n1, ncol), dtype=np.float32)
    d_de = cp.zeros((n1, ncol), dtype=np.float32)

    stages = _stages(ncol, nz)
    stages.launch("ntiedtke_midlevel", (
        d_in["pten"], d_in["pqen"], d_in["pqsen"], d_in["pverv"],
        d_in["pgeo"], d_in["pgeoh"],
        d_ldcum, d_ktype, d_kcbot, d_klab,
        io_f["plrain"], io_f["pmfu"], d_pmfub, io_f["ptu"], io_f["pqu"],
        io_f["plu"], io_f["pmfus"], io_f["pmfuq"], io_f["pmful"],
        io_f["pdmfup"], d_en, d_de,
        np.int32(ncol), np.int32(nz),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    stages.check_geometry()
    cp.cuda.Stream.null.synchronize()
    return (keys, {f: cp.asnumpy(v)[1:1 + nz, :] for f, v in io_f.items()},
            cp.asnumpy(d_klab)[1:1 + nz, :],
            cp.asnumpy(d_ktype), cp.asnumpy(d_kcbot), cp.asnumpy(d_pmfub))


def test_midlevel_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_midlevel").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]


def test_cuda_cubasmcn_is_bitwise_on_both_sides_of_the_guard(cuda_midlevel,
                                                             slice4):
    """60 triggering columns and 48 that do not, in ONE launch.

    The non-triggering half is the half that matters: the kernel must LEAVE
    all thirteen outputs alone there.  Zeroing them at entry -- the CUDA
    reflex -- would pass on the 60 and fail on the 48, which is exactly the
    quiet failure this test exists to make loud.
    """
    *_, ml, mls, _ = slice4
    keys, lev, klab, ktype, kcbot, pmfub = cuda_midlevel
    bad, scal = {}, {}
    for c, key in enumerate(keys):
        for f in _ML_F:
            n = _differing(lev[f][:, c], ml[key][f])
            if n:
                bad[(key, f)] = n
        if not np.array_equal(klab[:, c], ml[key]["klab"]):
            bad[(key, "klab")] = "int mismatch"
        e = mls[key]
        if int(ktype[c]) != e["ktype"]:
            scal[(key, "ktype")] = (int(ktype[c]), e["ktype"])
        if int(kcbot[c]) != e["kcbot"]:
            scal[(key, "kcbot")] = (int(kcbot[c]), e["kcbot"])
        if _bits(pmfub[c]) != _bits(e["pmfub"]):
            scal[(key, "pmfub")] = (pmfub[c], e["pmfub"])
    assert not scal, f"scalar mismatches: {dict(list(scal.items())[:8])}"
    assert not bad, f"differing words: {dict(list(bad.items())[:8])}"

    trig = sum(1 for c in range(len(keys)) if int(ktype[c]) == 3)
    assert trig >= 24 and trig < len(keys), (
        f"the launch must span both sides of the guard; {trig} triggered")


# ===========================================================================
# Slice 4b: cumastrn's first-guess cloud-base mass flux (:500-541)
# ===========================================================================
# A PREREQUISITE, not a stage.  It runs between cutypen and cuascn and
# produces pmfub, which cuascn (:1949-1952, :1992), cudlfsn (:2469) and the
# closure (via zmfub1) all read -- and it can CLEAR ldcum (:536).
#
# Without it a cuascn fixture is fed pmfub = 0 and every mass-flux quantity
# downstream is structurally zero: green, and meaningless.  The same defect
# is what made the cuentrn capture degenerate, and its showing up twice is
# what identified this as a prerequisite rather than a coverage gap.

from gpuwm.verify.ntiedtke_oracle import mfub_surface        # noqa: E402
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_mfub       # noqa: E402


def _mfub_args(key, fixture, slice2, slice3):
    inp, exp, sur, sa = fixture
    cv, ci, _ = slice2
    _, _, ct, cts = slice3
    v, u, t, e = cv[key], ci[key], ct[key], cts[key]
    return dict(
        ktype=e["ktype"], kcbot=e["cubot"],
        ptte=v["ptte"], pqte=v["pqte"],
        # prsi(klev+1) is the SURFACE interface after the flip, and it is the
        # one entry the level fixture does not carry.  cumastrn:509 reads it
        # every step through paph(jk+1) at jk = klev, so it has to come from
        # the surface fixture.  Feeding a zero there was wrong on all 42
        # ldcum columns -- measured, before this was fixed.
        paph=np.append(exp[key]["prsi"], sa[key]["psfc"]),
        puen=exp[key]["uf"], pven=exp[key]["vf"],
        ptu=t["cutu"], pqu=t["cuqu"], plu=t["culu"],
        ztenh=u["ptenh"], zqenh=u["pqenh"],
        lndj=int(abs(float(sa[key]["xland"]) - 2.0)),
        dt=NT_DT)


def test_mfub_mirror_is_bitwise(fixture, slice2, slice3):
    mf = mfub_surface()
    c = NtConstants()
    bad = {}
    for key in sorted(mf):
        ldcum = [cutypen_surface()[key]["ldcum"] != 0]
        zdhpbl, upbl, zmfub = np_ntiedtke_mfub(
            ldcum=ldcum, c=c, **_mfub_args(key, fixture, slice2, slice3))
        w = mf[key]
        for name, got, want in (("zdhpbl", zdhpbl, w["zdhpbl"]),
                                ("upbl", upbl, w["upbl"]),
                                ("zmfub", zmfub, w["zmfub"])):
            if _bits(got) != _bits(want):
                bad[(key, name)] = (got, want)
        if int(ldcum[0]) != w["ldcum"]:
            bad[(key, "ldcum")] = (int(ldcum[0]), w["ldcum"])
    assert not bad, f"mismatches: {dict(list(bad.items())[:8])}"


def test_the_mass_flux_block_actually_produces_something(fixture):
    """A prerequisite that returned zeros would look exactly like success.

    This is the assertion that would have caught the original defect: the
    old pre-state had pmfub = 0 on every column and every downstream
    mass-flux quantity was structurally zero, which graded green.
    """
    mf = mfub_surface()
    nonzero_mfub = sum(1 for v in mf.values() if float(v["zmfub"]) != 0.0)
    nonzero_upbl = sum(1 for v in mf.values() if float(v["upbl"]) != 0.0)
    assert nonzero_mfub >= 24, (
        f"only {nonzero_mfub} columns have a non-zero first-guess mass "
        "flux; a cuascn or cudlfsn fixture built on this would be "
        "structurally zero downstream")
    assert nonzero_upbl >= 12, (
        f"only {nonzero_upbl} columns have a non-zero upbl; the closure's "
        "ztaubl would be untested")


def test_cuda_mfub_is_bitwise(fixture, slice2, slice3):
    cp = pytest.importorskip("cupy")
    mf = mfub_surface()
    cts = cutypen_surface()
    keys = sorted(mf)
    ncol, nz = len(keys), NT_NZ
    n1 = nz + 2

    def pack1(get):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, key in enumerate(keys):
            x = np.asarray(get(key), dtype=np.float32)
            a[1:1 + min(len(x), n1 - 1), c] = x[:n1 - 1]
        return cp.asarray(a)

    args = [_mfub_args(k, fixture, slice2, slice3) for k in keys]
    d = {n: pack1(lambda k, n=n: args[keys.index(k)][n])
         for n in ("ptte", "pqte", "paph", "puen", "pven",
                   "ptu", "pqu", "plu", "ztenh", "zqenh")}
    d_ktype = cp.asarray(np.array([a["ktype"] for a in args], dtype=np.int32))
    d_kcbot = cp.asarray(np.array([a["kcbot"] for a in args], dtype=np.int32))
    d_lndj = cp.asarray(np.array([a["lndj"] for a in args], dtype=np.int32))
    d_ldcum = cp.asarray(np.array([int(cts[k]["ldcum"] != 0) for k in keys],
                                  dtype=np.int32))
    d_dh = cp.zeros(ncol, dtype=np.float32)
    d_up = cp.zeros(ncol, dtype=np.float32)
    d_mf = cp.zeros(ncol, dtype=np.float32)

    stages = _stages(ncol, nz)
    stages.launch("ntiedtke_mfub", (
        d["ptte"], d["pqte"], d["paph"], d["puen"], d["pven"],
        d["ptu"], d["pqu"], d["plu"], d["ztenh"], d["zqenh"],
        d_ktype, d_kcbot, d_lndj, d_ldcum, d_dh, d_up, d_mf,
        # tiedtke_closure = 0: cu16's own first guess, which is what this
        # oracle was captured against.  The kernel takes the selector
        # immediately before ncol, and a hand-built tuple that omits it
        # shifts every argument after it -- nz is then read from dt and
        # the launch access-violates rather than raising, because CuPy
        # does not check arity against the signature.
        np.int32(0),
        np.int32(ncol), np.int32(nz), np.float32(NT_DT),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()

    dh, up, mfv = (cp.asnumpy(d_dh), cp.asnumpy(d_up), cp.asnumpy(d_mf))
    ld = cp.asnumpy(d_ldcum)
    bad = {}
    for c, key in enumerate(keys):
        w = mf[key]
        for name, got, want in (("zdhpbl", dh[c], w["zdhpbl"]),
                                ("upbl", up[c], w["upbl"]),
                                ("zmfub", mfv[c], w["zmfub"])):
            if _bits(got) != _bits(want):
                bad[(key, name)] = (got, want)
        if int(ld[c]) != w["ldcum"]:
            bad[(key, "ldcum")] = (int(ld[c]), w["ldcum"])
    assert not bad, f"mismatches: {dict(list(bad.items())[:8])}"


def test_mfub_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_mfub").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]


# ===========================================================================
# Slice 5: the CAPE closure -- the arithmetic the port turns on
# ===========================================================================
# scale_fac and scale_fac2 are applied HERE, to different ktypes, and :716
# is the only use of scale_fac2 in the whole scheme.
#
# Everything below is driven from captures taken at the closure's OWN entry
# inside the cumastrn replication -- in particular ktype, which :566-568
# flips between cuascn and here.  Feeding cuascn's ktype runs the wrong arm;
# that cost two rounds of debugging and is why the capture point matters.

from gpuwm.verify.ntiedtke_ref import np_ntiedtke_closure    # noqa: E402

_CLOSURE_DEEP = ("zheat", "zcape", "zcape1", "zcape2", "ztauc", "ztaubl",
                 "ztau", "upbl")


def _closure_load():
    """Every closure input and expectation, from the captures."""
    from gpuwm.verify.ntiedtke_oracle import load_csv, word

    def lv(name, fields):
        out = {}
        for r in load_csv(name):
            k = (int(r["case"]), float(r["dx"]))
            i = int(r["k"]) - 1
            s = out.setdefault(k, {f: np.zeros(NT_NZ, dtype=np.float32)
                                   for f in fields})
            for f in fields:
                s[f][i] = word(r[f])
        return out

    def sv(name, fl=(), it=()):
        return {(int(r["case"]), float(r["dx"])):
                dict(**{f: word(r[f]) for f in fl},
                     **{f: int(r[f]) for f in it})
                for r in load_csv(name)}

    return {
        "cin": lv("nt-cuascn-in-levels.csv",
                  ("pap", "paph", "pgeoh", "pgeo")),
        "cout": lv("nt-cuascn-out-levels.csv",
                   ("ptu", "pqu", "plu", "pmfu")),
        "dd": lv("nt-downdraft-levels.csv",
                 ("zmfd", "zmfds", "zmfdq", "zdmfdp", "ztd", "zqd",
                  "pmfdde_rate", "ztenh", "zqenh")),
        "conv": lv("nt-conv-levels.csv",
                   ("ztp1", "zqp1", "ptte", "pqte")),
        "cs": sv("nt-cuascn-surface.csv", ("wup",),
                 ("ldcum", "kcbot", "kctop", "kdpl")),
        "ds": sv("nt-downdraft-surface.csv", ("upbl_pre", "zdhpbl"),
                 ("loddraf", "ktype_closure")),
        "cl": sv("nt-closure-surface.csv",
                 ("zheat", "zcape", "zcape1", "zcape2", "ztauc", "ztaubl",
                  "ztau", "zmfub", "zmfub1", "upbl", "scale_fac",
                  "scale_fac2")),
        "sa": sv("nt-surface.csv", ("xland", "psfc")),
    }


@pytest.fixture(scope="module")
def closure():
    d = _closure_load()
    assert set(d["cl"]) == set(d["cs"]) == set(d["ds"])
    return d


def _closure_kwargs(key, d):
    a, b, dn, v = d["cin"][key], d["cout"][key], d["dd"][key], d["conv"][key]
    e, f, g, s = d["cs"][key], d["ds"][key], d["cl"][key], d["sa"][key]
    return dict(
        ldcum=e["ldcum"] != 0,
        # ktype is the CLOSURE-TIME value: :566-568 flips it between cuascn
        # and here, and the arm is selected on the flipped one.
        ktype=f["ktype_closure"],
        kcbot=e["kcbot"], kctop=e["kctop"], kdpl=e["kdpl"],
        loddraf=f["loddraf"] != 0, wup=e["wup"], upbl=f["upbl_pre"],
        zmfub=g["zmfub"], scale_fac=g["scale_fac"],
        scale_fac2=g["scale_fac2"],
        lndj=int(abs(float(s["xland"]) - 2.0)),
        pgeoh=np.append(a["pgeoh"], np.float32(0.0)),
        paph=np.append(a["paph"], s["psfc"]),
        pap=a["pap"], pgeo=a["pgeo"], pten=v["ztp1"], pqen=v["zqp1"],
        ptenh=dn["ztenh"], pqenh=dn["zqenh"],
        ptu=b["ptu"], pqu=b["pqu"], plu=b["plu"], pmfu=b["pmfu"],
        pmfd=dn["zmfd"], ptd=dn["ztd"], pqd=dn["zqd"],
        ptte=v["ptte"], pqte=v["pqte"],
        pmfds=dn["zmfds"], pmfdq=dn["zmfdq"], pdmfdp=dn["zdmfdp"],
        pmfdde_rate=dn["pmfdde_rate"], zdhpbl=f["zdhpbl"],
        dt=np.float32(60.0))


def test_the_fixture_spans_every_closure_arm(closure):
    """All three arms, at CLOSURE time.

    :716 -- zmfub1/scale_fac2 -- is the only use of scale_fac2 in the whole
    scheme.  It was reached ZERO times until case 11 was built to stay under
    zdnoprc, because every earlier shallow case rose 885 hPa and :566
    promoted it to deep.  A regression to zero here silently un-covers the
    line the port's entire scheme choice rests on.
    """
    seen = collections.Counter(
        v["ktype_closure"] for k, v in closure["ds"].items()
        if closure["cs"][k]["ldcum"])
    assert seen[1] >= 24, f"deep arm too thin: {dict(seen)}"
    assert seen[2] >= 6, (
        f"the closure's SHALLOW arm is uncovered again: {dict(seen)}. "
        ":716 is the only use of scale_fac2 in the scheme.")
    assert seen[3] >= 6, f"mid-level arm uncovered: {dict(seen)}"


def test_closure_mirror_is_bitwise(closure):
    c = NtConstants()
    bad, cnt = {}, collections.Counter()
    for key in sorted(closure["cl"]):
        got = np_ntiedtke_closure(c=c, **_closure_kwargs(key, closure))
        g = closure["cl"][key]
        ld = closure["cs"][key]["ldcum"] != 0
        kt = closure["ds"][key]["ktype_closure"]
        if ld and kt == 1:
            cnt["deep"] += 1
            for f in _CLOSURE_DEEP:
                if _bits(got[f]) != _bits(g[f]):
                    bad[(key, f)] = (got[f], g[f])
        if ld:
            cnt[f"ktype{kt}"] += 1
            if _bits(got["zmfub1"]) != _bits(g["zmfub1"]):
                bad[(key, f"zmfub1(kt{kt})")] = (got["zmfub1"], g["zmfub1"])
    assert cnt["deep"] and cnt["ktype2"] and cnt["ktype3"], dict(cnt)
    assert not bad, f"mismatches: {dict(list(bad.items())[:8])}"


@pytest.fixture(scope="module")
def cuda_closure(closure):
    cp = pytest.importorskip("cupy")
    d = closure
    keys = sorted(d["cl"])
    ncol, nz = len(keys), NT_NZ
    n1 = nz + 2

    def pack1(get):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, key in enumerate(keys):
            x = np.asarray(get(key), dtype=np.float32)
            a[1:1 + min(len(x), n1 - 1), c] = x[:n1 - 1]
        return cp.asarray(a)

    kw = [_closure_kwargs(k, d) for k in keys]
    lev = {n: pack1(lambda k, n=n: kw[keys.index(k)][n])
           for n in ("pgeoh", "paph", "pap", "pgeo", "pten", "pqen",
                     "ptenh", "pqenh", "ptu", "pqu", "plu", "pmfu",
                     "ptd", "pqd", "ptte", "pqte")}
    io_lev = {n: pack1(lambda k, n=n: kw[keys.index(k)][n])
              for n in ("pmfd", "pmfds", "pmfdq", "pdmfdp", "pmfdde_rate")}

    def ivec(name):
        return cp.asarray(np.array([int(a[name]) for a in kw], dtype=np.int32))

    def fvec(name):
        return cp.asarray(np.array([float(a[name]) for a in kw],
                                   dtype=np.float32))

    d_ldcum = cp.asarray(np.array([int(a["ldcum"]) for a in kw],
                                  dtype=np.int32))
    d_lodd = cp.asarray(np.array([int(a["loddraf"]) for a in kw],
                                 dtype=np.int32))
    outs = {n: cp.zeros(ncol, dtype=np.float32) for n in
            ("zheat", "zcape", "zcape1", "zcape2", "ztauc", "ztaubl",
             "ztau", "zmfub1")}
    d_upbl = fvec("upbl")

    stages = _stages(ncol, nz)
    stages.launch("ntiedtke_closure", (
        d_ldcum, ivec("ktype"), ivec("kcbot"), ivec("kctop"), ivec("kdpl"),
        d_lodd, ivec("lndj"), fvec("wup"), fvec("zmfub"), fvec("zdhpbl"),
        fvec("scale_fac"), fvec("scale_fac2"),
        lev["pgeoh"], lev["paph"], lev["pap"], lev["pgeo"], lev["pten"],
        lev["pqen"], lev["ptenh"], lev["pqenh"], lev["ptu"], lev["pqu"],
        lev["plu"], lev["pmfu"], lev["ptd"], lev["pqd"], lev["ptte"],
        lev["pqte"],
        io_lev["pmfd"], io_lev["pmfds"], io_lev["pmfdq"],
        io_lev["pdmfdp"], io_lev["pmfdde_rate"],
        outs["zheat"], outs["zcape"], outs["zcape1"], outs["zcape2"],
        outs["ztauc"], outs["ztaubl"], outs["ztau"], d_upbl, outs["zmfub1"],
        # tiedtke_closure = 0, same position and same reason as the mfub
        # launch above: cu16's scale_fac-scaled ztau, which is what this
        # oracle holds.  Passing 1 here would select the fixed 2400 s and
        # the comparison would be against the wrong reference.
        np.int32(0),
        np.int32(ncol), np.int32(nz), np.float32(60.0),
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    return keys, {n: cp.asnumpy(v) for n, v in outs.items()}, \
        cp.asnumpy(d_upbl)


def test_closure_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function("ntiedtke_closure").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]


def test_cuda_closure_is_bitwise_across_every_arm(cuda_closure, closure):
    """All 108 columns in ONE launch, spanning ktype 0, 1, 2 and 3.

    The FP-contraction guard for the closure: deep and shallow compute
    different expressions under a runtime branch on ktype, which is exactly
    the shape ptxas can contract inconsistently.  Both arms run in the same
    launch so a divergence between them shows here.
    """
    keys, outs, upbl = cuda_closure
    d = closure
    bad, cnt = {}, collections.Counter()
    for c, key in enumerate(keys):
        g = d["cl"][key]
        ld = d["cs"][key]["ldcum"] != 0
        kt = d["ds"][key]["ktype_closure"]
        if ld and kt == 1:
            cnt["deep"] += 1
            for f in ("zheat", "zcape", "zcape1", "zcape2", "ztauc",
                      "ztaubl", "ztau"):
                if _bits(outs[f][c]) != _bits(g[f]):
                    bad[(key, f)] = (outs[f][c], g[f])
            if _bits(upbl[c]) != _bits(g["upbl"]):
                bad[(key, "upbl")] = (upbl[c], g["upbl"])
        if ld:
            cnt[f"ktype{kt}"] += 1
            if _bits(outs["zmfub1"][c]) != _bits(g["zmfub1"]):
                bad[(key, f"zmfub1(kt{kt})")] = (outs["zmfub1"][c],
                                                 g["zmfub1"])
    assert cnt["deep"] and cnt["ktype2"] and cnt["ktype3"], (
        f"the launch must span every arm: {dict(cnt)}")
    assert not bad, f"mismatches: {dict(list(bad.items())[:8])}"


def test_the_kernel_leaves_deep_only_slots_alone_on_other_columns(
        cuda_closure, closure):
    """zheat/zcape/... are undefined outside the deep branch.

    They are dimension(klon) arrays in cumastrn assigned only inside
    `if(ldcum .and. ktype==1)`.  A kernel that initialised them would
    diverge on 66 of 108 columns and still pass every deep check -- the
    quiet failure.  Here the buffers arrive zeroed, so a non-deep column
    must come back exactly zero.
    """
    keys, outs, _ = cuda_closure
    d = closure
    for c, key in enumerate(keys):
        ld = d["cs"][key]["ldcum"] != 0
        kt = d["ds"][key]["ktype_closure"]
        if ld and kt == 1:
            continue
        for f in ("zheat", "zcape", "zcape1", "zcape2", "ztauc", "ztaubl",
                  "ztau"):
            assert float(outs[f][c]) == 0.0, (
                f"{key} ktype={kt} had {f} written outside the deep branch")
