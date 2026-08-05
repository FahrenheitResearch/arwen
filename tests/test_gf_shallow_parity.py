"""``CUP_gf_sh`` float32 reference against the WRF v4.6.1 per-stage capture.

The oracle is ``gf-shallow-levels.csv`` / ``gf-shallow-surface.csv``, written
by ``tools/gf_wrf461_oracle/run_gf_shallow.F90`` -- a statement-order
replication of ``CUP_gf_sh``'s body that reproduces the module's own
``CUP_gf_sh`` bitwise on all 108 (case, dx) rows
(``gf-shallow-consistency.csv``).  Every field below is therefore WRF's own
word at that point in the routine.

Coverage is 18 columns, not 216, and that is the scheme rather than the
fixture: ``CUP_gf_sh`` takes no ``dx`` and none of its fourteen inputs
depends on one, so the six-point dx sweep produces six identical answers.
The same consistency file carries the proof (``ndiff_words_vs_dx1``), and
``build.sh`` fails if a future WRF release makes the shallow arm
scale-aware.  5 of the 18 converge; the other 13 exercise ``ierr`` 3, 5 and
21.

Every assertion is ``max_ulp == 0`` except ``fzu``, which goes through
``tgammaf`` -- the reference's one modelled libm call.  As in the deep gate,
the chain is run twice and the pinned run attributes the residual rather than
absorbing it.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.gf_oracle import GF_NCASE, GF_NZ, GF_ORACLE_DIR, load_gf_oracle
from gpuwm.verify.gf_shallow_ref import cup_gf_sh_column

NZ = GF_NZ


def _read(name):
    with (GF_ORACLE_DIR / name).open(newline="", encoding="ascii") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def shal():
    """Run the shallow reference on every case, twice.

    ``got`` uses the reference's own ``tgammaf``; ``pin`` uses the oracle's
    captured ``fzu``.
    """
    fixture = load_gf_oracle()
    lv = fixture.stage_levels
    sf = fixture.stage_surface
    rows = _read("gf-shallow-surface.csv")
    want_s = {k: [None] * GF_NCASE for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            want_s[k][int(r["case"]) - 1] = v

    # The prepared column, from the stage fixture.  Any (idx, arm) will do --
    # GFDRV's preparation of the fields CUP_gf_sh reads has no dx in it -- so
    # take the first row the loader placed for each case and assert that the
    # choice does not matter.
    col_of_case = {}
    for ci, (case, idx, arm) in enumerate(fixture.key):
        col_of_case.setdefault(int(case), ci)

    got = []
    pin = []
    for case in range(1, GF_NCASE + 1):
        ci = col_of_case[case]
        args = dict(
            zo=lv["zo"][ci], t=lv["t2d"][ci], q=lv["q2d"][ci],
            z1=sf["ter11"][ci], tn=lv["tshall"][ci], qo=lv["qshall"][ci],
            po=lv["po"][ci], psur=sf["psur"][ci], dhdt=lv["dhdt"][ci],
            kpbl=int(sf["kpbli"][ci]), rho=lv["rhoi"][ci],
            hfx=sf["hfxi"][ci], qfx=sf["qfxi"][ci], xland=sf["xlandi"][ci],
            dtime=sf["dt"][ci],
        )
        got.append(cup_gf_sh_column(**args))
        f = np.float32(want_s["sh_fzu"][case - 1])
        pin.append(
            cup_gf_sh_column(**args, fzu_override=f if f > 0 else None)
        )
    return dict(
        fixture=fixture, got=got, pin=pin, want_s=want_s,
        want_l=fixture.shallow_levels, ncase=GF_NCASE,
    )


def _level(got, name):
    return np.stack([np.asarray(g[name], dtype=np.float32)[1:] for g in got])


def _scalar(got, name):
    return np.array([np.float32(g[name]) for g in got], dtype=np.float32)


def _int(got, name):
    return np.array([int(g[name]) for g in got], dtype=np.int64)


def _ulp(a, b, mask=None):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if mask is not None:
        a = a[mask]
        b = b[mask]
    d = fp32_ulp_distance(a, b)
    finite = d >= 0
    bad = int(np.count_nonzero(~finite))
    worst = int(d[finite].max()) if finite.any() else -1
    loc = np.unravel_index(int(np.argmax(np.where(finite, d, -1))), a.shape)
    return worst, bad, loc


def _want_l(shal, name):
    return shal["want_l"][name]


def _want_s(shal, name):
    return np.array(
        [np.float32(v) for v in shal["want_s"][name]], dtype=np.float32
    )


def _want_i(shal, name):
    return np.array([int(v) for v in shal["want_s"][name]], dtype=np.int64)


# ==========================================================================
# the fixture's own claims
# ==========================================================================
def test_the_replication_is_bitwise_and_dx_free(shal):
    """Both columns of gf-shallow-consistency.csv are zero on every row.

    The first is what makes every other test in this file a statement about
    WRF; the second is what makes an 18-column capture sufficient for a
    216-column fixture.
    """
    c = shal["fixture"].shallow_consistency
    assert int(np.count_nonzero(c["ndiff_words_vs_cup_gf_sh"])) == 0
    assert int(np.count_nonzero(c["ndiff_words_vs_dx1"])) == 0
    assert c["case"].shape[0] == GF_NCASE * 6


def test_the_capture_reaches_four_rejection_reasons_and_five_clouds(shal):
    """What the 18 columns actually exercise, so a future case-table edit
    that narrows the coverage fails loudly rather than quietly."""
    ierr = _want_i(shal, "ierr")
    counts = {int(v): int(np.count_nonzero(ierr == v)) for v in np.unique(ierr)}
    assert counts == {0: 5, 3: 2, 5: 9, 21: 2}


# ==========================================================================
# stage 1: w*, the excesses, and the cap
# ==========================================================================
@pytest.mark.parametrize(
    "field", ["buo_flux", "zws", "ztexec", "zqexec", "cap_max", "entr_rate"]
)
def test_convective_scale_velocity_bitwise(shal, field):
    w, bad, loc = _ulp(_scalar(shal["got"], field), _want_s(shal, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at case {loc[0] + 1}"


def test_the_cap_collapses_to_the_pbl_top_outside_the_ierr_guard(shal):
    """``if(kpbl(i).gt.3)cap_max(i)=po_cup(i,kpbl(i))`` (:371) sits BEFORE the
    ``ierr`` test, so it fires on rejected columns too.  The fixture has both
    sides: cases with ``kpbl <= 3`` keep the 125 mb default."""
    cap = _want_s(shal, "cap_max")
    kpbl = np.array(
        [int(v) for v in shal["fixture"].stage_surface["kpbli"][:GF_NCASE]],
    )
    assert np.any(cap == np.float32(125.0))
    assert np.any(cap != np.float32(125.0))
    assert _ulp(_scalar(shal["got"], "cap_max"), cap)[0] == 0


def test_xland1_exact(shal):
    assert np.array_equal(_int(shal["got"], "xland1"), _want_i(shal, "xland1"))


# ==========================================================================
# stage 2: the two environments
# ==========================================================================
@pytest.mark.parametrize("field", ["qes", "he", "hes", "qeso", "heo", "heso"])
def test_cup_env_bitwise(shal, field):
    w, bad, loc = _ulp(_level(shal["got"], field), _want_l(shal, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at case {loc[0] + 1} k={loc[1] + 1}"


@pytest.mark.parametrize(
    "field",
    [
        "qes_cup", "q_cup", "he_cup", "hes_cup", "z_cup", "p_cup",
        "gamma_cup0", "t_cup", "qeso_cup", "qo_cup", "heo_cup", "heso_cup",
        "zo_cup", "po_cup0", "gammao_cup", "tn_cup",
    ],
)
def test_cup_env_clev_bitwise(shal, field):
    w, bad, loc = _ulp(_level(shal["got"], field), _want_l(shal, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at case {loc[0] + 1} k={loc[1] + 1}"


# ==========================================================================
# stage 3: the trigger -- kbmax, k22, cup_kbcon at iloop = 5
# ==========================================================================
@pytest.mark.parametrize(
    "field", ["kbmax", "k22_0", "k22_1", "kbcon_1", "ierr_1"]
)
def test_trigger_indices_exact(shal, field):
    assert np.array_equal(_int(shal["got"], field), _want_i(shal, field)), (
        f"{field}: {_int(shal['got'], field)} vs {_want_i(shal, field)}"
    )


def test_k22_is_the_maxloc_offset_wrf_does_not_apply(shal):
    """WRF's ``k22`` is the position inside ``heo_cup(2:kbmax)``, used as an
    absolute index.  A port that adds the section offset -- the reading a
    careful eye produces -- is one level high wherever the argmax is above
    the second level, and the fixture has such a column (case 13, where
    ``k22`` comes out 8 rather than 9)."""
    got = _int(shal["got"], "k22_0")
    assert np.array_equal(got, _want_i(shal, "k22_0"))
    assert int(got[12]) == 8


def test_cup_kbcon_iloop_5_is_a_different_routine(shal):
    """iloop = 5 changes four things (see gf_deep_ref.cup_kbcon).  The
    witness that the branch is live: on every converged column ``kbcon``
    comes back at ``k22 + n`` for some ``n > 1``, which the iloop = 1 arm
    would have rejected at its ``KBCON-K22 == 1`` exit."""
    ok = _want_i(shal, "ierr_1") == 0
    kb = _want_i(shal, "kbcon_1")[ok]
    k22 = _want_i(shal, "k22_1")[ok]
    assert np.all(kb - k22 >= 1)
    assert np.any(kb - k22 > 1)
    assert np.array_equal(_int(shal["got"], "kbcon_1"), _want_i(shal, "kbcon_1"))


@pytest.mark.parametrize("field", ["hkb0", "hkbo0", "hkbo_1", "hkb_2"])
def test_cloud_base_values_bitwise(shal, field):
    w, bad, loc = _ulp(_scalar(shal["got"], field), _want_s(shal, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at case {loc[0] + 1}"


# ==========================================================================
# stage 4: cup_minimi, get_inversion_layers, and the first ktop
# ==========================================================================
def test_kstabi_exact(shal):
    assert np.array_equal(_int(shal["got"], "kstabi"), _want_i(shal, "kstabi"))


def test_get_inversion_layers_derivative_bitwise(shal):
    w, bad, loc = _ulp(_level(shal["got"], "dtempdz"), _want_l(shal, "dtempdz"))
    assert bad == 0
    assert w == 0, f"dtempdz: {w} ULP at case {loc[0] + 1} k={loc[1] + 1}"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_get_inversion_layers_slots_exact(shal, n):
    got = np.array([int(g["k_inv"][n]) for g in shal["got"]], dtype=np.int64)
    assert np.array_equal(got, _want_i(shal, f"kinv{n}"))


def test_the_out_of_bounds_read_is_not_exercised_on_this_fixture(shal):
    """``get_inversion_layers`` reads ``t_cup(kend+8)``.  At the shallow call
    site ``kend`` is ``kstabi``, which ``cup_minimi`` bounds by ``kbmax`` and
    ``kbmax`` is bounded by ``ktf/2 = 20`` -- so ``kend+8 <= 28 < kte`` and
    the read is in bounds here.  This is a measurement of the fixture, not a
    proof over all soundings; the deep call site has no such bound."""
    assert int(np.count_nonzero(_want_i(shal, "kstabi_oob"))) == 0
    assert not any(bool(g["kstabi_oob"]) for g in shal["got"])
    assert int(_want_i(shal, "kstabi").max()) <= 20


@pytest.mark.parametrize("field", ["entr2d_a", "cd_a"])
def test_entrainment_profile_bitwise(shal, field):
    w, bad, loc = _ulp(_level(shal["got"], field), _want_l(shal, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at case {loc[0] + 1} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["ierr_231", "ktop_0"])
def test_first_ktop_exact(shal, field):
    assert np.array_equal(_int(shal["got"], field), _want_i(shal, field))


def test_ktop_comes_from_the_inversion_slot_on_every_live_column(shal):
    """:437-447 has two arms.  On this fixture the ``k_inv_layers(1)`` arm
    takes every converged column; the 200 mb scan below it is reached only
    where the first arm's pressure test fails, and no case does that.
    Recorded so a later fixture that reaches it is a visible change."""
    ok = _want_i(shal, "ierr_231") == 0
    ktop0 = _want_i(shal, "ktop_0")[ok]
    kinv1 = _want_i(shal, "kinv1")[ok]
    assert np.array_equal(ktop0, kinv1)


# ==========================================================================
# stage 5: rates_up_pdf / get_zu_zd_pdf_fim, draft SH2
# ==========================================================================
@pytest.mark.parametrize("field", ["sh_tun", "sh_alpha", "sh_beta"])
def test_sh2_shape_parameters_bitwise(shal, field):
    w, bad, loc = _ulp(_scalar(shal["got"], field), _want_s(shal, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at case {loc[0] + 1}"


@pytest.mark.parametrize("field", ["ktop_pdf", "kbcon_2", "ierr_2", "sh_kbadj",
                                   "sh_kfinal"])
def test_sh2_indices_exact(shal, field):
    assert np.array_equal(_int(shal["got"], field), _want_i(shal, field))


def test_sh2_beta_is_two_and_a_half_not_the_deep_arms_one_point_three(shal):
    ok = _want_i(shal, "ierr_2") == 0
    assert np.all(_want_s(shal, "sh_beta")[ok] == np.float32(2.5))


def test_sh2_tunning_hits_both_clamps(shal):
    """Both ``max(0.2, ...)`` and ``min(0.8, ...)`` are live on this fixture,
    and the interior is populated too.  ``SH2``'s upper clamp is 0.8 against
    ``UP``'s 0.9, which is one of the three things that separate the two
    branches -- so a port that reuses ``UP``'s constant is wrong exactly on
    the column that saturates."""
    ok = _want_i(shal, "ierr_2") == 0
    tun = _want_s(shal, "sh_tun")[ok]
    assert np.any(tun == np.float32(0.2))
    assert np.any(tun == np.float32(0.8))
    assert np.any((tun > np.float32(0.2)) & (tun < np.float32(0.8)))
    assert _ulp(_scalar(shal["got"], "sh_tun"), _want_s(shal, "sh_tun"))[0] == 0


def test_fzu_is_the_one_measured_divergence(shal):
    """``fzu = tgammaf(a+b)/(tgammaf(a)*tgammaf(b))``, and glibc's
    ``tgammaf`` is not correctly rounded.  The reference models it with a
    float64 gamma, which the pow/gamma probe measured as 1-2 ULP off on 31 of
    51 arguments.

    The shallow arm is WORSE than the deep one here, and by a factor of two:
    ``SH2``'s ``beta = 2.5`` is not an integer, so ``tgammaf(beta)`` is a
    genuine gamma evaluation rather than the exactly-representable
    ``tgammaf(1.3)``-free case, and all three calls contribute.  Measured
    over the 16 columns that reach the pdf: 4 exact, and a worst case of 4
    ULP against the deep arm's 2.  Asserted at the measurement, not at a
    round number.
    """
    w, bad, _ = _ulp(_scalar(shal["got"], "sh_fzu"), _want_s(shal, "sh_fzu"))
    assert bad == 0
    assert w == 4, f"fzu residual moved from the measured 4 ULP to {w}"


def test_fzu_pinned_to_the_oracle_is_bitwise(shal):
    """With the oracle's own ``fzu`` the whole shallow chain is exact, which
    is what attributes every residual below to ``tgammaf`` alone."""
    w, bad, _ = _ulp(_scalar(shal["pin"], "sh_fzu"), _want_s(shal, "sh_fzu"))
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["zu_pdf", "zuo_b"])
def test_updraft_massflux_profile_bitwise_when_fzu_is_pinned(shal, field):
    wp, bad, loc = _ulp(_level(shal["pin"], field), _want_l(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1} k={loc[1] + 1}"


def test_the_sh2_kb_adj_scan_is_dead(shal):
    """Unlike ``UP`` and ``MID``, ``SH2`` computes ``kb_adj`` from its
    ``zu < 1.e-6`` scan and then neither raises it to 2 nor zeroes ``zu``
    below it -- so the scan cannot affect the profile at all.  Ported as
    spelled and asserted dead two ways: the scan's answer never moves off
    ``max(k22, 2)`` on this fixture, and ``zu`` below ``k22`` is zero because
    the pdf's own fill loop starts at ``kb_adj``, not because anything zeroed
    it.  ``CUP_gf_sh:462-468`` is what does the zeroing, at ``k22``."""
    ok = _want_i(shal, "ierr_2") == 0
    kbadj = _want_i(shal, "sh_kbadj")[ok]
    k22 = _want_i(shal, "k22_1")[ok]
    assert np.array_equal(kbadj, np.maximum(k22, 2))
    zu_pdf = _want_l(shal, "zu_pdf")[ok]
    for row in range(int(ok.sum())):
        assert np.all(zu_pdf[row, : kbadj[row] - 1] == 0.0)
        assert zu_pdf[row, kbadj[row] - 1] != 0.0


@pytest.mark.parametrize("field", ["ktop_3", "k22_3"])
def test_massflux_trim_indices_exact(shal, field):
    assert np.array_equal(_int(shal["pin"], field), _want_i(shal, field))


# ==========================================================================
# stage 6: get_lateral_massflux
# ==========================================================================
@pytest.mark.parametrize("field", ["upme", "upmd", "cd_b", "entr2d_b"])
def test_get_lateral_massflux_bitwise(shal, field):
    wp, bad, loc = _ulp(_level(shal["pin"], field), _want_l(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1} k={loc[1] + 1}"


# ==========================================================================
# stage 7: the in-cloud updraft
# ==========================================================================
@pytest.mark.parametrize(
    "field", ["hc", "hco", "dby", "dbyo", "dbyt", "qco_a", "qrco", "pwo",
              "cupclw", "qco", "cnvwt"]
)
def test_updraft_profiles_bitwise(shal, field):
    wp, bad, loc = _ulp(_level(shal["pin"], field), _want_l(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["ki_dbyt", "ktop_4", "ierr_4"])
def test_updraft_indices_exact(shal, field):
    assert np.array_equal(_int(shal["pin"], field), _want_i(shal, field))


def test_qaver_bitwise(shal):
    w, bad, _ = _ulp(_scalar(shal["pin"], "qaver"), _want_s(shal, "qaver"))
    assert bad == 0 and w == 0


def test_the_dbyt_ktop_clip_is_not_reached_on_this_fixture(shal):
    """``if(ktop(i).gt.ki+1)`` (:533) re-tops the cloud at the maximum of the
    buoyancy integral ``dbyt``.  No converged column of this fixture trips it,
    and the reason is the scheme rather than the case table.

    Two targeted soundings were built against it and neither fired, which is
    the measurement.  ``ktop`` comes from ``get_inversion_layers``' shallow
    slot, and an inversion IS a moist-static-energy barrier -- so the level
    the inversion search picks and the level ``hco - heso_cup`` changes sign
    at are the same feature, and ``ktop`` lands at ``ki`` or ``ki + 1`` by
    construction.  Decoupling them needs :441-447's OTHER arm, where ``ktop``
    comes from pressure alone, which in turn needs the only second-derivative
    feature to sit more than 200 mb above cloud base; the attempt at that
    found ``get_inversion_layers`` latching onto float32 curvature noise in
    the nominally straight part of the profile instead, so the sounding
    generator cannot currently produce a column with no low-level feature.

    Transcribed, unexercised, and recorded here rather than left to be
    discovered.  A CUDA mirror inherits the same hole.
    """
    ok = _want_i(shal, "ierr_4") == 0
    ktop3 = _want_i(shal, "ktop_3")[ok]
    ki = _want_i(shal, "ki_dbyt")[ok]
    assert np.all(ktop3 <= ki + 1)
    assert np.all(ktop3 >= ki)


# ==========================================================================
# stage 8: the cloud work functions
# ==========================================================================
@pytest.mark.parametrize("field", ["aa0", "aa1"])
def test_cloud_work_functions_bitwise(shal, field):
    wp, bad, loc = _ulp(_scalar(shal["pin"], field), _want_s(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1}"


def test_ierr_after_the_cloud_work_function_check(shal):
    assert np.array_equal(_int(shal["pin"], "ierr_5"), _want_i(shal, "ierr_5"))


# ==========================================================================
# stage 9: the dellas and the mbdt-perturbed state
# ==========================================================================
@pytest.mark.parametrize(
    "field", ["dellah", "dellaq", "dellaqc", "dellat"]
)
def test_della_bitwise(shal, field):
    wp, bad, loc = _ulp(_level(shal["pin"], field), _want_l(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1} k={loc[1] + 1}"


def test_c1_shal_is_zero_so_the_below_ktop_dellaqc_limb_vanishes(shal):
    """``c1_shal = 0.`` (module_cu_gf_sh.F:48) makes ``dellaqc`` identically
    zero below ``ktop`` and leaves only the detrainment term at ``ktop``.  It
    also drops out of the ``qrco`` denominator.  Ported as spelled, and this
    pins the consequence so a future WRF that turns it on is a visible
    change."""
    ok = _want_i(shal, "ierr_5") == 0
    dqc = _want_l(shal, "dellaqc")[ok]
    ktop = _want_i(shal, "ktop_4")[ok]
    for row in range(int(ok.sum())):
        assert np.all(dqc[row, : ktop[row] - 1] == 0.0)
        assert dqc[row, ktop[row] - 1] != 0.0


@pytest.mark.parametrize("field", ["xhe", "xq", "xt", "xqes", "xhes"])
def test_perturbed_state_bitwise(shal, field):
    wp, bad, loc = _ulp(_level(shal["pin"], field), _want_l(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1} k={loc[1] + 1}"


def test_the_perturbed_cup_env_overwrites_xhe(shal):
    """``call cup_env(xz,xqes,xhe,xhes,...)`` (:753) takes ``xhe`` as its
    ``he`` OUTPUT, and ``cup_env`` writes ``he`` whenever ``itest .le. 0`` --
    which -1 is.  So the moist static energy built from ``dellah`` at :731 is
    thrown away before ``xhc`` reads it.  A port that leaves ``xhe`` alone
    disagrees on every converged column.

    The size of that, measured: 3 to 5 of the 40 lanes per column, 1-2 ULP
    each.  Small, and not a tolerance question -- the difference walks into
    ``xhc``, ``xdby``, ``xaa0``, and out through ``xkshal`` into ``xmb``.
    """
    ok = _want_i(shal, "ierr_5") == 0
    got = _level(shal["pin"], "xhe")[ok]
    want = _want_l(shal, "xhe")[ok]
    assert _ulp(got, want)[0] == 0
    dellah = _want_l(shal, "dellah")[ok]
    heo = _want_l(shal, "heo")[ok]
    naive = (dellah * np.float32(0.5) + heo).astype(np.float32)
    d = fp32_ulp_distance(naive, want)
    assert int(np.count_nonzero(d != 0)) > 0
    assert int(d.max()) == 2


@pytest.mark.parametrize(
    "field",
    ["xqes_cup", "xq_cup", "xhe_cup", "xhes_cup", "gamma_cupx", "xt_cup",
     "po_cupx"],
)
def test_perturbed_cloud_levels_bitwise(shal, field):
    wp, bad, loc = _ulp(_level(shal["pin"], field), _want_l(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1} k={loc[1] + 1}"


def test_po_cup_is_zeroed_on_rejected_columns(shal):
    """The perturbed ``cup_env_clev``'s 13th actual is ``po_cup`` itself and
    the routine zeroes its outputs before the ierr guard, so a rejected
    column loses the pressure column it built at :345.  Nothing reads it
    afterwards, so this is invisible in WRF and load-bearing in a capture."""
    bad = _want_i(shal, "ierr_5") != 0
    assert np.all(_want_l(shal, "po_cupx")[bad] == 0.0)
    assert np.any(_want_l(shal, "po_cup0")[bad] != 0.0)


@pytest.mark.parametrize("field", ["xhc", "xdby", "xzu"])
def test_perturbed_updraft_bitwise(shal, field):
    wp, bad, loc = _ulp(_level(shal["pin"], field), _want_l(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["xhkb", "xaa0"])
def test_perturbed_work_function_bitwise(shal, field):
    wp, bad, loc = _ulp(_scalar(shal["pin"], field), _want_s(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1}"


# ==========================================================================
# stage 10: the three-member shallow closure
# ==========================================================================
@pytest.mark.parametrize(
    "field", ["xkshal", "xff1", "xff2", "xff3", "blqe", "trash_kb", "xmbmax",
              "xmb"]
)
def test_shallow_closure_bitwise(shal, field):
    wp, bad, loc = _ulp(_scalar(shal["pin"], field), _want_s(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1}"


def test_the_closure_is_a_three_way_average_not_an_ensemble(shal):
    """``ichoice_s`` is a parameter 0 (module_cu_gf_wrfdrv.F:71), so the
    single-closure arm at :847 is dead and ``xmb`` is always the mean of the
    three, clipped at ``xmbmax = 1``."""
    ok = _want_i(shal, "ierr_6") == 0
    a = _want_s(shal, "xff1")[ok]
    b = _want_s(shal, "xff2")[ok]
    c = _want_s(shal, "xff3")[ok]
    mean = ((a + b).astype(np.float32) + c).astype(np.float32) / np.float32(3.0)
    assert _ulp(np.minimum(mean, np.float32(1.0)), _want_s(shal, "xmb")[ok])[0] == 0


# ==========================================================================
# stage 11: end to end
# ==========================================================================
@pytest.mark.parametrize("field", ["outt", "outq", "outqc", "zuo"])
def test_cup_gf_sh_end_to_end_bitwise(shal, field):
    wp, bad, loc = _ulp(_level(shal["pin"], field), _want_l(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["xmb_out", "pre"])
def test_cup_gf_sh_end_to_end_scalars_bitwise(shal, field):
    wp, bad, loc = _ulp(_scalar(shal["pin"], field), _want_s(shal, field))
    assert bad == 0
    assert wp == 0, f"{field}: {wp} ULP at case {loc[0] + 1}"


@pytest.mark.parametrize("field", ["k22", "kbcon", "ktop", "ierr"])
def test_cup_gf_sh_end_to_end_indices_exact(shal, field):
    assert np.array_equal(_int(shal["pin"], field), _want_i(shal, field)), (
        f"{field}: {_int(shal['pin'], field)} vs {_want_i(shal, field)}"
    )


def test_the_tgammaf_residual_reaches_the_shallow_tendencies(shal):
    """The unpinned run is the honest statement of what a modelled
    ``tgammaf`` costs the shallow arm, and the answer is not a rounding
    footnote either.

    ``xkshal = (xaa0 - aa1)/mbdt`` is the same difference-of-two-cloud-work-
    functions cancellation the deep arm's ``xk`` is, so ``fzu``'s 2-4 ULP --
    order 3e-7 relative -- comes out of the closure at up to **8.4e-3
    relative in xmb**, and ``pre`` follows it.  Four orders of magnitude of
    amplification, against the deep arm's five to six.  The shallow arm is
    milder only because ``mbdt = .5`` against the deep ``.1`` widens the
    denominator, not because the mechanism is absent.

    So a CUDA ``tgammaf`` that disagrees with glibc by one ULP disagrees with
    WRF's shallow mass flux in the third significant figure.  No branch
    flips: ``ierr`` is identical on every column either way.
    """
    ok = _want_i(shal, "ierr") == 0
    got = _scalar(shal["got"], "xmb")[ok]
    want = _want_s(shal, "xmb")[ok]
    rel = np.abs(got - want) / np.maximum(np.abs(want), np.float32(1e-30))
    assert float(rel.max()) > 1e-3
    assert float(rel.max()) < 2e-2, f"shallow xmb moved {float(rel.max()):.3e}"
    gp = _scalar(shal["got"], "pre")[ok]
    wp = _want_s(shal, "pre")[ok]
    relp = np.abs(gp - wp) / np.maximum(np.abs(wp), np.float32(1e-30))
    assert float(relp.max()) > 1e-3
    assert np.array_equal(_int(shal["got"], "ierr"), _want_i(shal, "ierr"))
    assert np.array_equal(_int(shal["got"], "ktop"), _want_i(shal, "ktop"))
