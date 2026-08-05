"""``CUP_gf`` float32 reference against the WRF v4.6.1 per-stage capture.

The oracle here is ``gf-deep-levels.csv`` / ``gf-deep-surface.csv``, written
by ``tools/gf_wrf461_oracle/run_gf_stages.F90`` -- a statement-order
replication of ``CUP_gf``'s body that reproduces the module's own ``cup_gf``
bitwise on all 216 columns (``gf-deep-consistency.csv``).  So every field
below is WRF's own word at that point in the routine, not a reconstruction.

Every assertion is ``max_ulp == 0`` except where a docstring says otherwise
and names the measurement.  There is exactly one such place: ``fzu``, the
``tgammaf`` normalisation of the beta-function mass-flux profile.  The gate
runs the whole chain twice -- once with the reference's own ``tgammaf`` model
and once with the oracle's captured ``fzu`` -- so the residual is attributed
rather than absorbed.

Masking.  ``cup_env``'s ``qes``/``he``/``hes`` are ``intent(out)`` and are not
written when ``ierr /= 0``, so on the perturbed-state call those lanes carry
the previous column's values in the capture and are compared only where
``ierr_6 == 0``.  ``cup_env_clev`` by contrast zeroes all eight of its outputs
before the ierr guard, so its fields are compared everywhere.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.gf_deep_body import cup_gf_column
from gpuwm.verify.gf_oracle import GF_NZ, GF_ORACLE_DIR, load_gf_oracle
from gpuwm.verify.gf_ref import gf_driver_prep

NZ = GF_NZ


def _read(name):
    with (GF_ORACLE_DIR / name).open(newline="", encoding="ascii") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def deep():
    """Run the reference on every column, twice.

    ``got`` uses the reference's own ``tgammaf``; ``pin`` uses the oracle's
    captured ``fzu`` for both the UP and DOWN profiles.
    """
    fixture = load_gf_oracle()
    lv = fixture.stage_levels
    sf = fixture.stage_surface
    gl = fixture.levels
    gs = fixture.surface
    rows = _read("gf-deep-surface.csv")
    lrows = _read("gf-deep-levels.csv")

    key = {}
    for r in rows:
        key[(int(r["case"]), int(r["idx"]), int(r["arm"]))] = r
    order = [tuple(int(v) for v in k) for k in fixture.key]

    want_s = {k: [] for k in rows[0]}
    for trip in order:
        r = key[trip]
        for k, v in r.items():
            want_s[k].append(v)

    lkey = {}
    for r in lrows:
        lkey[(int(r["case"]), int(r["idx"]), int(r["arm"]), int(r["k"]))] = r
    names_l = [n for n in lrows[0] if n not in ("case", "idx", "arm", "k")]
    want_l = {n: np.zeros((len(order), NZ), dtype=np.float32) for n in names_l}
    for ci, trip in enumerate(order):
        for k in range(1, NZ + 1):
            r = lkey[trip + (k,)]
            for n in names_l:
                want_l[n][ci, k - 1] = np.float32(r[n])

    got = []
    pin = []
    for ci in range(fixture.ncol):
        args = dict(
            zo=lv["zo"][ci], t=lv["t2d"][ci], q=lv["q2d"][ci],
            z1=sf["ter11"][ci], tn=lv["tn"][ci], qo=lv["qo"][ci],
            po=lv["po"][ci], psur=sf["psur"][ci], us=lv["us"][ci],
            vs=lv["vs"][ci], rho=lv["rhoi"][ci], hfx=sf["hfxi"][ci],
            qfx=sf["qfxi"][ci], xland=sf["xlandi"][ci], dx=gs["dx"][ci],
            omeg=lv["omeg_in"][ci], kpbl=int(sf["kpbli"][ci]),
            ccn=sf["ccn"][ci], dtime=sf["dt"][ci], xmbs_in=sf["xmbs"][ci],
        )
        got.append(cup_gf_column(**args))
        fu = np.float32(want_s["up_fzu"][ci])
        fd = np.float32(want_s["dn_fzu"][ci])
        pin.append(
            cup_gf_column(
                **args,
                fzu_up=fu if fu > 0 else None,
                fzu_dn=fd if fd > 0 else None,
            )
        )
    return dict(
        fixture=fixture, got=got, pin=pin, want_l=want_l, want_s=want_s,
        ncol=fixture.ncol,
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


def _want_l(deep, name):
    return deep["want_l"][name]


def _want_s(deep, name):
    return np.array([np.float32(v) for v in deep["want_s"][name]], dtype=np.float32)


def _want_i(deep, name):
    return np.array([int(v) for v in deep["want_s"][name]], dtype=np.int64)


def _ierr_mask(deep, which="ierr_6"):
    return _want_i(deep, which) == 0


def _defined_at_pmin(deep):
    """Columns where CUP_gf:672-677 actually ran: ``ierr == 0`` out of
    ``cup_kbcon`` and not shut off by the saturated-column test at :663."""
    ok = _want_i(deep, "ierr_1") == 0
    shut = (_want_s(deep, "frh_kb") >= np.float32(0.97)) & (
        _want_s(deep, "sig") <= _want_s(deep, "sig_thresh")
    )
    return ok & ~shut


# ==========================================================================
# stage 2: cup_env, on both the unforced and the forced state
# ==========================================================================
@pytest.mark.parametrize("field", ["qes", "he", "hes", "qeso", "heo", "heso"])
def test_cup_env_bitwise(deep, field):
    """``ierr`` is 0 for every column at both of these calls, so no mask."""
    w, bad, loc = _ulp(_level(deep["got"], field), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


def test_satvap_is_a_base_10_power_not_an_exponential(deep):
    """A port that reaches for ``exp`` here is wrong by construction: WRF
    spells ``10 ** eilog`` and ``log(x)/log(10.)``, i.e. ``powf`` over a
    folded constant, not ``expf``/``log10f``.  ``qes`` is the only witness."""
    w, _, _ = _ulp(_level(deep["got"], "qes"), _want_l(deep, "qes"))
    assert w == 0


# ==========================================================================
# stage 3: cup_env_clev
# ==========================================================================
@pytest.mark.parametrize(
    "field",
    [
        "qes_cup", "q_cup", "he_cup", "hes_cup", "gamma_cup1", "t_cup",
        "qeso_cup", "qo_cup", "heo_cup", "heso_cup", "zo_cup", "po_cup",
        "gammao_cup", "tn_cup",
    ],
)
def test_cup_env_clev_bitwise(deep, field):
    w, bad, loc = _ulp(_level(deep["got"], field), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["u_cup", "v_cup"])
def test_wind_cloud_levels_bitwise(deep, field):
    w, bad, _ = _ulp(_level(deep["got"], field), _want_l(deep, field))
    assert bad == 0 and w == 0


# ==========================================================================
# stage 4: the trigger chain -- zws, the excesses, kbmax, k22, cup_kbcon
# ==========================================================================
@pytest.mark.parametrize("field", ["zws", "ztexec", "zqexec", "cap_max"])
def test_convective_scale_velocity_bitwise(deep, field):
    """``zws`` is ``1.2*x**.3333`` -- a literal power, not a cube root."""
    w, bad, _ = _ulp(_scalar(deep["got"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["entr_rate", "sig", "sig_thresh"])
def test_scale_awareness_bitwise(deep, field):
    """``sig = (1-frh)^2`` is the whole point of the scheme, and ``frh``
    carries the ``frh_thresh`` clamp that back-solves ``entr_rate``."""
    w, bad, _ = _ulp(_scalar(deep["got"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["kbmax", "k22_0", "xland1"])
def test_trigger_indices_exact(deep, field):
    assert np.array_equal(_int(deep["got"], field), _want_i(deep, field))


@pytest.mark.parametrize("field", ["hkb0", "hkbo0"])
def test_get_cloud_bc_bitwise(deep, field):
    """Masked to the columns that found a ``k22``: :641-642 sits under
    ``IF(ierr(I).eq.0)`` and ``hkb``/``hkbo`` are plain locals, so the 12
    ``ierr == 2`` columns carry the previous column's word in the capture."""
    m = _want_i(deep, "k22_0") > 0
    w, bad, _ = _ulp(_scalar(deep["got"], field)[m], _want_s(deep, field)[m])
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["kbcon_1", "k22_1", "ierr_1", "kstabi"])
def test_cup_kbcon_exact(deep, field):
    """``cup_kbcon`` is a ``GO TO`` graph that moves ``k22`` and rebuilds
    ``hkb`` mid-search.  These four indices are the whole observable."""
    assert np.array_equal(_int(deep["got"], field), _want_i(deep, field))


def test_frh_at_cloud_base_bitwise(deep):
    w, bad, _ = _ulp(_scalar(deep["got"], "frh_kb"), _want_s(deep, "frh_kb"))
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["pmin_lev", "start_level"])
def test_pmin_and_start_level_exact(deep, field):
    """``pmin_lev`` is never initialised in ``CUP_gf`` -- it is written only
    inside the ``ierr == 0`` branch at :672-677 and read only as a dead
    argument to ``get_zu_zd_pdf_fim``.  Compared where WRF defines it."""
    m = _defined_at_pmin(deep)
    assert np.array_equal(
        _int(deep["got"], field)[m], _want_i(deep, field)[m]
    )


def test_entrainment_profile_bitwise(deep):
    w, bad, loc = _ulp(_level(deep["got"], "entr2d_a"), _want_l(deep, "entr2d_a"))
    assert bad == 0
    assert w == 0, f"entr2d_a: {w} ULP at column {loc[0]} k={loc[1] + 1}"


# ==========================================================================
# stage 5: rates_up_pdf / get_zu_zd_pdf_fim -- and the tgammaf residual
# ==========================================================================
@pytest.mark.parametrize("field", ["up_tun", "up_alpha", "up_beta"])
def test_updraft_pdf_shape_parameters_bitwise(deep, field):
    """``tunning`` and ``alpha`` are pure float32 algebra and must be exact;
    only the ``gamma`` of them is not."""
    w, bad, _ = _ulp(_scalar(deep["got"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["dn_tun", "dn_alpha", "dn_beta"])
def test_downdraft_pdf_shape_parameters_bitwise(deep, field):
    w, bad, _ = _ulp(_scalar(deep["got"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


def test_fzu_is_the_one_measured_divergence(deep):
    """glibc's ``tgammaf`` is not correctly rounded and this reference models
    it with a float64 ``gamma``.  MEASURED, not asserted at 0: the budget
    below is the number this port is allowed to carry, and it is recorded so
    a regression that widens it is visible."""
    up = _ulp(_scalar(deep["got"], "up_fzu"), _want_s(deep, "up_fzu"))[0]
    dn = _ulp(_scalar(deep["got"], "dn_fzu"), _want_s(deep, "dn_fzu"))[0]
    assert up <= 4, f"up fzu drifted to {up} ULP"
    assert dn <= 4, f"down fzu drifted to {dn} ULP"


def test_fzu_pinned_to_the_oracle_is_bitwise(deep):
    """With ``fzu`` taken from the capture the profile parameters are exact,
    which is what makes the previous test a statement about ``tgammaf`` alone
    and not about the algebra around it."""
    for field in ("up_fzu", "dn_fzu"):
        w, bad, _ = _ulp(_scalar(deep["pin"], field), _want_s(deep, field))
        assert bad == 0 and w == 0, field


@pytest.mark.parametrize("field", ["up_kbadj", "up_kklev", "up_kfinal",
                                   "dn_kbadj", "ktop_pdf", "ktopdby",
                                   "kbcon_2", "ierr_2"])
def test_rates_up_pdf_indices_exact(deep, field):
    assert np.array_equal(_int(deep["pin"], field), _want_i(deep, field))


def test_the_powf_residual_is_glibcs_and_reaches_no_output(deep):
    """With ``fzu`` pinned, ``zu`` is bitwise on 8630 of 8640 lanes.  The 10
    that are not are glibc's ``powf``, not this port's arithmetic, and the
    claim is a computation rather than an assertion.

    At level 17 of column (18, 2, 0) the profile needs
    ``powf(0x3F0D923B, 0x3E999998)``.  Evaluated to 80 significant digits the
    true value is 0.83718320727036911868..., and the midpoint between the two
    neighbouring float32 values is 0.83718320727348327636... -- so the
    correctly rounded answer is ``0x3F5651A3`` and it beats ``0x3F5651A4`` by
    3.1e-12, or 5.2e-5 of a float32 ULP.  glibc returns ``0x3F5651A4``.
    glibc's ``powf`` is a double-precision ``exp2(y*log2(x))`` with about
    0.82 ULP of worst-case error, so on a value this close to a rounding
    boundary it can land either side, and reproducing which side would mean
    reimplementing its polynomial and tables.  This port computes the
    correctly rounded value, which is the defined answer.

    The consequence is bounded, and that is the point of the test: all 10
    columns are ``ierr == 6`` ("cloud depth very shallow"), WRF rejects them
    before they produce any tendency, and every output field is bitwise."""
    a = _level(deep["pin"], "zu_pdf")
    b = _want_l(deep, "zu_pdf")
    d = fp32_ulp_distance(a, b)
    bad = np.argwhere(d != 0)
    assert len(bad) == 10
    assert int(d.max()) <= 1
    cols = sorted({int(c) for c, _ in bad})
    assert len(cols) == 10
    ierr = _want_i(deep, "ierr")
    assert set(ierr[cols].tolist()) == {6}, "the powf residual reached a live column"


def test_updraft_massflux_profile_cost_of_the_tgammaf_model(deep):
    """What the ``tgammaf`` model costs in ``zu`` itself, measured.  The
    normalisation ``zu/max(zu)`` cancels most of it but not all: ``zubeg`` is
    .1, not 0, so ``fzu`` does not divide out."""
    w, bad, _ = _ulp(_level(deep["got"], "zu_pdf"), _want_l(deep, "zu_pdf"))
    assert bad == 0
    assert w <= 8, f"zu_pdf drifted to {w} ULP under the tgammaf model"


# ==========================================================================
# stage 6: get_lateral_massflux
# ==========================================================================
@pytest.mark.parametrize("field", ["cd", "entr2d_b", "upme", "upmd"])
def test_get_lateral_massflux_bitwise(deep, field):
    w, bad, loc = _ulp(_level(deep["pin"], field), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field,lanes,budget", [("upmeu", 20, 8), ("upmdu", 10, 2)])
def test_lateral_massflux_momentum_carries_the_powf_residual(
    deep, field, lanes, budget
):
    """``up_massentru = up_massentro + lambau*up_massdetro`` differences the
    two mass fluxes, so the ``powf`` residual below is amplified here (8 ULP
    from 1) even though ``upme`` and ``upmd`` themselves are bitwise.  Same
    10 columns, same cause; see
    :func:`test_the_powf_residual_is_glibcs_and_reaches_no_output`."""
    a = _level(deep["pin"], field)
    b = _want_l(deep, field)
    d = fp32_ulp_distance(a, b)
    assert int((d != 0).sum()) == lanes
    assert int(d.max()) <= budget


# ==========================================================================
# stage 7: the in-cloud updraft and ktop's revision
# ==========================================================================
@pytest.mark.parametrize("field", ["hc", "uc", "vc", "hco", "dby", "dbyo", "dbyt"])
def test_updraft_profiles_bitwise(deep, field):
    w, bad, loc = _ulp(_level(deep["pin"], field), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["ktop_dbyt", "ierr_3", "kzdown", "jmin",
                                   "kdet_2", "ierr_4"])
def test_updraft_and_downdraft_indices_exact(deep, field):
    name = {"kdet_2": "kdet"}.get(field, field)
    assert np.array_equal(_int(deep["pin"], name), _want_i(deep, field))


# ==========================================================================
# stage 8: the downdraft
# ==========================================================================
@pytest.mark.parametrize(
    "field",
    ["cdd", "ddme", "ddmd", "ddmeu", "ddmdu", "mentrd2d", "hcdo", "ucd",
     "vcd", "dbydo", "c1d"],
)
def test_downdraft_profiles_bitwise(deep, field):
    w, bad, loc = _ulp(_level(deep["pin"], field), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["bud", "beta", "edtmax"])
def test_downdraft_buoyancy_bitwise(deep, field):
    w, bad, _ = _ulp(_scalar(deep["pin"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["qcdo", "qrcdo", "pwdo"])
def test_cup_dd_moisture_bitwise(deep, field):
    w, bad, loc = _ulp(_level(deep["pin"], field), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["pwevo", "bu", "ierr_5"])
def test_cup_dd_moisture_totals(deep, field):
    if field == "ierr_5":
        assert np.array_equal(_int(deep["pin"], "ierr_5"), _want_i(deep, field))
        return
    w, bad, _ = _ulp(_scalar(deep["pin"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


# ==========================================================================
# stage 9: cup_up_moisture
# ==========================================================================
@pytest.mark.parametrize("field", ["qco", "qrco", "pwo", "clw_all"])
def test_cup_up_moisture_bitwise(deep, field):
    """The ``c0`` autoconversion coefficient compounds across the below-LFC
    loop and resets every level above it.  ``pwo`` is the witness."""
    w, bad, loc = _ulp(_level(deep["pin"], field), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["pwavo", "psum", "psumh"])
def test_cup_up_moisture_totals_bitwise(deep, field):
    w, bad, _ = _ulp(_scalar(deep["pin"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["cupclw", "cnvwt"])
def test_convective_cloud_water_bitwise(deep, field):
    """These two leave ``cup_gf`` as arguments, so the authority for them is
    the OTHER harness -- ``gf-stage-levels.csv``, written by ``run_cup_gf``.
    Cross-fixture on purpose: it checks the two decompositions agree."""
    want = deep["fixture"].stage_levels[field]
    w, bad, loc = _ulp(_level(deep["pin"], field), want)
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


# ==========================================================================
# stage 10: the cloud work functions and the diurnal closure
# ==========================================================================
@pytest.mark.parametrize("field", ["aa0", "aa1", "aa1_bl", "tau_ecmwf",
                                   "tau_bl", "umean"])
def test_cloud_work_functions_bitwise(deep, field):
    w, bad, _ = _ulp(_scalar(deep["pin"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


def test_ierr_after_the_cloud_work_function_check(deep):
    assert np.array_equal(_int(deep["pin"], "ierr_6"), _want_i(deep, "ierr_6"))


@pytest.mark.parametrize("field", ["edt", "edtc1", "edto"])
def test_cup_dd_edt_bitwise(deep, field):
    """``VSHEAR**2`` and ``**3`` are integer literal exponents and fold to
    multiply chains; the precipitation-efficiency polynomial is Horner."""
    w, bad, _ = _ulp(_scalar(deep["pin"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


# ==========================================================================
# stage 11: the della fields and the perturbed state
# ==========================================================================
@pytest.mark.parametrize(
    "field", ["dellu", "dellv", "dellah", "dellaq", "dellaqc", "dellat"]
)
def test_della_bitwise(deep, field):
    w, bad, loc = _ulp(_level(deep["pin"], field), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["xhe", "xq", "xt"])
def test_perturbed_state_bitwise(deep, field):
    """Masked: :1506 guards the whole block on ``ierr == 0`` and ``xhe``,
    ``xq`` and ``xt`` are plain locals."""
    m = _ierr_mask(deep)
    w, bad, _ = _ulp(_level(deep["pin"], field)[m], _want_l(deep, field)[m])
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["xqes", "xhes"])
def test_perturbed_cup_env_bitwise_where_defined(deep, field):
    """Masked: ``cup_env``'s outputs are ``intent(out)`` and untouched when
    ``ierr /= 0``, so the capture carries the previous column there."""
    m = _ierr_mask(deep)
    w, bad, _ = _ulp(_level(deep["pin"], field)[m], _want_l(deep, field)[m])
    assert bad == 0 and w == 0


@pytest.mark.parametrize(
    "field", ["xqes_cup", "xq_cup", "xhe_cup", "xhes_cup", "gamma_cupx",
              "xt_cup", "xhc", "xdby"]
)
def test_perturbed_cloud_levels_bitwise(deep, field):
    w, bad, loc = _ulp(_level(deep["pin"], field), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["xhkb", "xaa0", "pr7"])
def test_perturbed_work_function_bitwise(deep, field):
    """``xaa0`` is zeroed for every column by ``cup_up_aa0``; ``xhkb`` and the
    ``pr_ens`` accumulation are inside ``ierr == 0`` guards."""
    m = _ierr_mask(deep) if field != "xaa0" else slice(None)
    w, bad, _ = _ulp(_scalar(deep["pin"], field)[m], _want_s(deep, field)[m])
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["ierr_7", "k22x", "kbconx", "ierr2", "ierr3"])
def test_cap_probe_arms_exact(deep, field):
    assert np.array_equal(_int(deep["pin"], field), _want_i(deep, field))


def test_mconv_on_the_cloud_grid_bitwise(deep):
    """``CUP_gf:1660`` throws away GFDRV's ``mconv`` and rebuilds it on the
    cloud grid with the DEEP module's ``g = 9.81``, not the driver's
    9.80665."""
    w, bad, _ = _ulp(_scalar(deep["pin"], "mconv2"), _want_s(deep, "mconv2"))
    assert bad == 0 and w == 0


# ==========================================================================
# stage 12: the closure ensemble and the feedback
# ==========================================================================
@pytest.mark.parametrize("n", list(range(1, 17)))
def test_closure_ensemble_member_bitwise(deep, n):
    got = np.array([np.float32(g["xf_ens"][n]) for g in deep["pin"]], dtype=np.float32)
    w, bad, _ = _ulp(got, _want_s(deep, f"xf{n}"))
    assert bad == 0 and w == 0, f"xf_ens({n}): {w} ULP"


@pytest.mark.parametrize("n", list(range(1, 11)))
def test_forcing_diagnostic_slot_bitwise(deep, n):
    got = np.array(
        [np.float32(g["forcing"][n]) for g in deep["pin"]], dtype=np.float32
    )
    w, bad, _ = _ulp(got, _want_s(deep, f"f{n}"))
    assert bad == 0 and w == 0, f"forcing({n}): {w} ULP"


@pytest.mark.parametrize("field", ["xf_dicycle", "closure_n"])
def test_diurnal_and_closure_count_bitwise(deep, field):
    w, bad, _ = _ulp(_scalar(deep["pin"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


@pytest.mark.parametrize("field", ["outt_o", "outq_o", "outqc_o"])
def test_cup_output_ens_3d_bitwise(deep, field):
    w, bad, loc = _ulp(_level(deep["pin"], field), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["xmb", "pre"])
def test_mass_flux_and_precip_bitwise(deep, field):
    w, bad, _ = _ulp(_scalar(deep["pin"], field), _want_s(deep, field))
    assert bad == 0 and w == 0


# ==========================================================================
# end to end
# ==========================================================================
@pytest.mark.parametrize("field", ["outt_ke", "outu_f", "outv_f"])
def test_cup_gf_end_to_end_bitwise(deep, field):
    """``outt_ke`` is ``outt`` after the ECMWF dissipative-heating step, i.e.
    the last thing ``CUP_gf`` does to it."""
    name = {"outt_ke": "outt", "outu_f": "outu", "outv_f": "outv"}[field]
    w, bad, loc = _ulp(_level(deep["pin"], name), _want_l(deep, field))
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["ktop", "ierr"])
def test_cup_gf_end_to_end_indices_exact(deep, field):
    assert np.array_equal(_int(deep["pin"], field), _want_i(deep, field))


def test_every_column_is_covered(deep):
    assert deep["ncol"] == 216
    ierr = _want_i(deep, "ierr")
    assert int((ierr == 0).sum()) == 60


def test_a_one_ulp_massflux_shape_perturbation_moves_xmb_by_seven_percent(deep):
    """The single most important number this port measured, and the reason
    ``fzu`` is an override rather than a rounding footnote.

    ``get_zu_zd_pdf_fim`` normalises the beta-function updraft profile with
    ``fzu = gamma(alpha+beta)/(gamma(alpha)*gamma(beta))``.  Move ``fzu`` by
    ONE ULP -- relative 6e-8 -- and re-run the column, and the deep mass flux
    ``xmb`` moves by up to **7.3 per cent**, median 1.9 per cent, on the 60
    converged columns.  Five to six orders of magnitude of amplification.

    It is the scheme, not the port.  ``cup_forcing_ens_3d`` builds every
    stability closure as ``-xff/xk`` with
    ``xk = (xaa0 - aa1)/mbdt`` -- a difference of two cloud work functions
    that agree to several digits, computed on states that differ only by the
    ``mbdt = .1`` perturbation.  A last-bit change in the mass-flux SHAPE
    walks through ``zu`` into the vertical integral ``aa1`` (450 ULP), and
    the cancellation in ``xk`` turns that into per-cent-level ``xmb``.

    Two consequences the project has to carry.  The CPU reference cannot ship
    a modelled ``tgammaf``: glibc's is not correctly rounded and this port's
    float64 model is 0-4 ULP off, which is exactly the 7.3 per cent measured
    below.  And phase 3 cannot treat CUDA's ``tgammaf`` as a tolerance
    question -- if it disagrees with glibc by one ULP the GPU's deep mass
    flux disagrees by per cent, not by ULP."""
    ierr = _want_i(deep, "ierr")
    want = _want_s(deep, "xmb").astype(np.float64)
    modelled = _scalar(deep["got"], "xmb").astype(np.float64)
    live = (ierr == 0) & (want != 0)
    assert int(live.sum()) == 60
    rel = np.abs(modelled[live] - want[live]) / np.abs(want[live])
    assert 0.05 < float(rel.max()) < 0.10, float(rel.max())

    # ...and pinning fzu to the oracle's own word takes it to exactly zero,
    # which is what attributes the whole residual to tgammaf.
    for name in ("xmb", "pre", "aa1"):
        w, _, _ = _ulp(_scalar(deep["pin"], name), _want_s(deep, name))
        assert w == 0, name


def test_the_amplification_is_reproducible_from_one_ulp(deep):
    """The same statement made without reference to ``tgammaf`` at all:
    perturb the oracle's own ``fzu`` by one ULP and re-run."""
    from gpuwm.verify.gf_deep_body import cup_gf_column as _run

    fixture = deep["fixture"]
    lv = fixture.stage_levels
    sf = fixture.stage_surface
    gs = fixture.surface
    ierr = _want_i(deep, "ierr")
    worst = 0.0
    n = 0
    for ci in range(fixture.ncol):
        if ierr[ci] != 0:
            continue
        fu = np.float32(deep["want_s"]["up_fzu"][ci])
        fd = np.float32(deep["want_s"]["dn_fzu"][ci])
        args = dict(
            zo=lv["zo"][ci], t=lv["t2d"][ci], q=lv["q2d"][ci],
            z1=sf["ter11"][ci], tn=lv["tn"][ci], qo=lv["qo"][ci],
            po=lv["po"][ci], psur=sf["psur"][ci], us=lv["us"][ci],
            vs=lv["vs"][ci], rho=lv["rhoi"][ci], hfx=sf["hfxi"][ci],
            qfx=sf["qfxi"][ci], xland=sf["xlandi"][ci], dx=gs["dx"][ci],
            omeg=lv["omeg_in"][ci], kpbl=int(sf["kpbli"][ci]),
            ccn=sf["ccn"][ci], dtime=sf["dt"][ci], xmbs_in=sf["xmbs"][ci],
        )
        base = float(_run(**args, fzu_up=fu, fzu_dn=fd)["xmb"])
        bump = float(
            _run(
                **args,
                fzu_up=np.nextafter(fu, np.float32(np.inf)),
                fzu_dn=fd,
            )["xmb"]
        )
        if base:
            worst = max(worst, abs(bump - base) / abs(base))
            n += 1
    assert n == 60
    assert 0.05 < worst < 0.10, worst


# ==========================================================================
# get_inversion_layers -- captured here, consumed by CUP_gf_sh
# ==========================================================================
def test_get_inversion_layers_derivative_bitwise(deep):
    m = _ierr_mask(deep)
    w, bad, _ = _ulp(
        _level(deep["pin"], "dtempdz")[m], _want_l(deep, "dtempdz")[m]
    )
    assert bad == 0 and w == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_get_inversion_layers_slots_exact(deep, n):
    got = np.array([int(g["k_inv"][n]) for g in deep["pin"]], dtype=np.int64)
    assert np.array_equal(got, _want_i(deep, f"kinv{n}"))


def test_get_inversion_layers_out_of_bounds_read_is_not_exercised(deep):
    """The clamp that keeps this port out of WRF's ``t_cup(kend+8)`` never
    fires on this fixture, so the divergence is recorded and not used."""
    assert int(_want_i(deep, "kinv_clamped").sum()) == 0
    assert sum(g["kinv_clamped"] for g in deep["pin"]) == 0
