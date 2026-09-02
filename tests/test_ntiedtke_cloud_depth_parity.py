"""cumastrn:562-590 graded at ``max_ulp == 0`` -- the ktype flip.

THE FIRST PIECE OF ORCHESTRATION TO GAIN AN OWNER, and the one the port
exists for. Thirty lines between cuascn and cudlfsn carrying:

* ``:566-568`` -- the ktype flip. A deep column whose cloud is shallower
  than 200 hPa becomes ktype 2; a shallow one that is deeper becomes
  ktype 1. ktype selects ``scale_fac`` (deep) or ``scale_fac2`` (shallow)
  in the closure. Feeding cuascn's ktype to the closure runs the wrong arm
  -- that was the fifth failure in this port, and it cost two rounds.
* ``:580-588`` -- the downdraft-array zeroing that four class-2 excuses in
  ``test_ntiedtke_aliasing_audit.py`` rest on.

NO NEW CAPTURE WAS NEEDED. This block's inputs are cuascn's outputs and its
outputs are cudlfsn's and cuddrafn's captured entry state, so it is graded
against rows that already existed -- which is the capture architecture being
reusable for something it was not built for.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_DXSWEEP, NT_NZ, load_csv, word
from gpuwm.verify.ntiedtke_ref import np_ntiedtke_cloud_depth


def _load():
    """Inputs from cuascn's exit; expectations from cudlfsn's entry."""
    cols, exp = {}, {}
    for r in load_csv("nt-cuascn-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "ktype_in": int(r["ktype"]),
            "kcbot": int(r["kcbot"]), "kctop": int(r["kctop"]),
            "kctop0": int(r["kctop0"])}
    for r in load_csv("nt-cuascn-out-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        s.setdefault("pdmfup", np.zeros(NT_NZ, dtype=np.float32))[
            int(r["k"]) - 1] = word(r["pdmfup"])
    for r in load_csv("nt-cuascn-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        s.setdefault("paph", np.zeros(NT_NZ, dtype=np.float32))[
            int(r["k"]) - 1] = word(r["paph"])
    # cudlfsn's entry IS this block's exit: nothing runs between them.
    for r in load_csv("nt-cudlfsn-in-surface.csv"):
        exp[(int(r["case"]), float(r["dx"]))] = {
            "ktype": int(r["ktype"]), "prfl": word(r["prfl"])}
    for r in load_csv("nt-cudlfsn-in-levels.csv"):
        s = exp[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in ("pmfd", "pmfds", "pmfdq", "pdmfdp"):
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = \
                word(r[f"{f}_in"])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)


def _paph(c):
    out = np.zeros(NT_NZ + 1, dtype=np.float32)
    out[:NT_NZ] = c["paph"]
    return out


_GOT = {k: np_ntiedtke_cloud_depth(
    ldcum=bool(_COLS[k]["ldcum"]), ktype=_COLS[k]["ktype_in"],
    kcbot=_COLS[k]["kcbot"], kctop=_COLS[k]["kctop"],
    kctop0=_COLS[k]["kctop0"], paph=_paph(_COLS[k]),
    pdmfup=_COLS[k]["pdmfup"], nz=NT_NZ) for k in _KEYS}


@pytest.mark.parametrize("dx", NT_DXSWEEP)
def test_the_ktype_flip_is_exact(dx):
    """The arm selection, against the closure's own captured input."""
    for key in (k for k in _KEYS if k[1] == dx):
        assert _GOT[key]["ktype"] == _EXP[key]["ktype"], (
            f"ktype {key}: cuascn gave {_COLS[key]['ktype_in']}, "
            f"we flipped to {_GOT[key]['ktype']}, reference has "
            f"{_EXP[key]['ktype']}")


@pytest.mark.parametrize("dx", NT_DXSWEEP)
def test_the_precipitation_sum_is_bitwise(dx):
    """zrfl is an accumulator over all klev levels, graded on its own axis."""
    for key in (k for k in _KEYS if k[1] == dx):
        g, e = np.float32(_GOT[key]["prfl"]), _EXP[key]["prfl"]
        assert g.view(np.uint32) == e.view(np.uint32), \
            f"prfl {key}: {g} vs {e}"


@pytest.mark.parametrize("field", ("pmfd", "pmfds", "pmfdq", "pdmfdp"))
def test_the_downdraft_arrays_are_zeroed(field):
    """:580-588, and the four class-2 excuses that depend on it.

    ``_CALLER_ALWAYS_ZEROES`` excuses cudlfsn's pmfd/pmfds/pmfdq/pdmfdp
    because cumastrn zeroes them immediately before the call. That was a
    property of the REFERENCE's pipeline, inherited by the port only if
    the port reproduced it. This is the stage that reproduces it, so the
    excuses stop being conditional.
    """
    for key in _KEYS:
        g, e = _GOT[key][field], _EXP[key][field]
        assert np.array_equal(g, e), f"{field} {key}"
        assert not np.any(g), f"{field} {key} is not zero"


def test_the_flip_ACTUALLY_FIRES_on_this_fixture():
    """Green must not be reachable by no column ever flipping.

    If the fixture never exercises :566-568, this stage is graded only in
    the sense that its guard is evaluated -- and the guard is the thing the
    port exists for. Measured rather than assumed.
    """
    flipped = [k for k in _KEYS
               if _COLS[k]["ldcum"] and _COLS[k]["ktype_in"] != _GOT[k]["ktype"]]
    assert flipped, (
        "no column changes ktype, so the flip at :566-568 is transcribed "
        "and never exercised. That is the one branch this port cannot "
        "afford to leave ungraded -- widen the case table.")
    both = {(_COLS[k]["ktype_in"], _GOT[k]["ktype"]) for k in flipped}
    assert (2, 1) in both, f"the shallow->deep flip never fires: {both}"


def test_the_DEEP_TO_SHALLOW_direction_is_a_named_coverage_gap():
    """:566 fires on no column; :567 fires on six.  Measured.

        ktype 1 -> 1   36 columns     ktype 2 -> 1    6   <- :567
        ktype 2 -> 2    6 columns     ktype 3 -> 3   12

    So the flip IS exercised and graded, but only in the shallow-to-deep
    direction. `:566` -- a DEEP column whose cloud is shallower than
    200 hPa being demoted to shallow -- is transcribed and never runs.

    That direction is the one a hurricane eyewall would take if its cloud
    were thin, and it is the direction that would move a column from
    scale_fac to scale_fac2. So the gap matters more than most.

    THIS IS A PHASE 1 COMPLETION ITEM, NOT A STANDING GAP (review,
    review), and the pricing is why:

      dx = 4500    deep    1/scale_fac    8.6%
                   shallow 1/scale_fac2  29.3%

    The untested transition MORE THAN TRIPLES the retained mass flux at
    fine-nest resolution -- 3.4x, larger than the entire GF-vs-NT gap the
    port was justified on. And it fires exactly where the campaign lives:
    thin, marginal-depth columns in an eyewall ring are the population
    most likely to satisfy `ktype == 1 .and. zpbmpt < zdnoprc`, and the
    population whose behaviour decides whether the port works. Section 3
    also records that the SHALLOW branch has already been misunderstood
    once in this port; the transition into it being ungraded is the same
    blind spot one level upstream.

    The failure mode is what makes it urgent rather than tidy: a reference tropical-cyclone
    run whose eyewall columns take an ungraded transition does not crash
    and does not look wrong. It produces a plausible number, and a
    disappointing f012 could not be attributed to the physics or to this
    arm. Unfalsifiable after the fact, cheap to prevent now.

    THE CASE TABLE ALREADY BRACKETS IT. Section 9 records case 1 as too
    deep to demote (kcbot 47 to kctop 3) and cases 8-11 as too weak to
    trigger. The demotion case is between them: forcing strong enough
    that cutypen calls it deep and cuascn sustains a plume, under a cap
    that terminates that plume at modest depth. That is the same "two
    things at once" shape that took three rounds on case 11.

    Written toward the gap: this FAILS when such a case is added, and
    that failure is the signal to require both directions instead.
    """
    demoted = [k for k in _KEYS if _COLS[k]["ldcum"]
               and _COLS[k]["ktype_in"] == 1 and _GOT[k]["ktype"] == 2]
    assert not demoted, (
        f"{len(demoted)} columns now flip deep->shallow at :566. That is a "
        "coverage GAIN: require both directions in the test above and "
        "remove this gap from docs/ntiedtke/PORT-RECORD.md.")


def test_ictop0_is_left_alone_on_non_cumulus_columns():
    """Conditional-write discipline: :569 is inside `if (ldcum)`.

    The oracle cannot see this -- ictop0 is not in the downstream capture
    -- so it is checked against the mirror's contract directly, the same
    way pud/pvd are.
    """
    for key in _KEYS:
        if not _COLS[key]["ldcum"]:
            assert _GOT[key]["ictop0"] == _COLS[key]["kctop0"], (
                f"ictop0 {key} was written on a non-cumulus column")


# ===========================================================================
# The kernel
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_depth():
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

    d_ktype = ivec("ktype_in")
    d_ictop0 = ivec("kctop0")
    d_prfl = cp.zeros(ncol, dtype=np.float32)
    out = {n: cp.full((n1, ncol), np.float32(7.0), dtype=np.float32)
           for n in ("pmfd", "pmfds", "pmfdq", "pdmfdp", "pdpmel")}

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_cloud_depth", (
        ivec("ldcum"), ivec("kcbot"), ivec("kctop"),
        pack("paph"), pack("pdmfup"),
        d_ktype, d_ictop0, d_prfl,
        out["pmfd"], out["pmfds"], out["pmfdq"], out["pdmfdp"],
        out["pdpmel"],
        np.int32(ncol), np.int32(NT_NZ)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    return (keys, cp.asnumpy(d_ktype), cp.asnumpy(d_ictop0),
            cp.asnumpy(d_prfl),
            {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in out.items()})


def test_kernel_flip_and_sum_are_exact(cuda_depth):
    """Graded against WRF, never against the mirror."""
    keys, ktype, _, prfl, _ = cuda_depth
    for c, key in enumerate(keys):
        assert int(ktype[c]) == _EXP[key]["ktype"], f"ktype {key}"
        g, e = np.float32(prfl[c]), _EXP[key]["prfl"]
        assert g.view(np.uint32) == e.view(np.uint32), f"prfl {key}"


@pytest.mark.parametrize("field", ("pmfd", "pmfds", "pmfdq", "pdmfdp"))
def test_kernel_zeroes_the_downdraft_arrays(cuda_depth, field):
    """Seeded with 7.0 so "zeroed" is distinguishable from "untouched".

    A zero-filled output buffer would make this test pass whether or not
    the kernel wrote anything -- the same non-discrimination the cuentrn
    degeneracy and the ptenu zero-seed turned on.
    """
    keys, _, _, _, lev = cuda_depth
    for c, key in enumerate(keys):
        assert not np.any(lev[field][:, c]), \
            f"{field} {key} kept its 7.0 sentinel; the kernel did not write"


def test_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    a = load_module("ntiedtke").get_function(
        "ntiedtke_cloud_depth").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
