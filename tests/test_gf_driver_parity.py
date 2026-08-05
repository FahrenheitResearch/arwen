"""The whole GF reference against the pinned WRF boundary, end to end.

The oracle here is ``gf-levels.csv`` / ``gf-surface.csv`` -- ``GFDRV`` itself,
driven by ``tools/gf_wrf461_oracle/run_cu_gf.F90``.  Not a decomposition, not
a replication: WRF's own entry point, on 18 cases x 6 grid spacings x 2
``ishallow`` arms = 216 columns, compared against
``gpuwm.verify.gf_driver.gfdrv_column`` running the same column through
preparation, ``CUP_gf_sh``, ``neg_check``, ``cup_gf``, ``neg_check`` and the
driver's output algebra.

The other three gates in this suite grade pieces:
``test_gf_wrf461_parity`` the preparation, ``test_gf_deep_parity`` the deep
cloud model, ``test_gf_shallow_parity`` the shallow one.  This one grades the
composition, which is where an ordering mistake lives -- ``neg_check`` on the
shallow tendencies before the deep call, ``cutens`` decided from ``xmbs``
before that ``neg_check``, the ``cuten`` gate erasing ``kbcon``/``ktop`` on a
deep cloud that did not rain.

Two residuals, both named rather than absorbed.

``tgammaf``.  The reference models glibc's ``tgammaf`` with a float64 gamma;
the deep arm amplifies its 1-2 ULP into per-cent ``xmb`` and the shallow arm
into 8e-3 relative.  So the gate runs twice -- once with the model and once
with the oracle's own ``fzu`` for all three profiles -- and the pinned run is
the one asserted bitwise.

**The driver's own mixed precision, on 8 of the 216 columns.**  Those 8 are
exactly ``gf_oracle.stage_rows_to_distrust`` -- the rows where
``run_cup_gf.F90``'s reconstruction of this same algebra also disagrees with
GFDRV, measured in session two and traced to ``module_gfs_physcons``
declaring ``real(8)`` constants and initialising them from ``real(4)``
literals.  The port is not adding anything: on ALL 216 columns it reproduces
the stage path -- preparation, both schemes, both ``neg_check`` calls, and
the ten post-``neg_check`` tendency fields -- bitwise
(:func:`test_the_port_reproduces_the_stage_path_on_every_column`), so the
residual is entirely GFDRV's.  It is a 1-2 ULP shift in ``xmb`` that scales
the tendency profile; no branch flips, and ``GDC``/``GDC2``/``HTOP``/
``HBOT``/``ktop_deep`` and the shallow diagnostics are bitwise on all 216
because none of them carries ``xmb``.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.gf_driver import gfdrv_column
from gpuwm.verify.gf_oracle import (
    GF_NZ,
    GF_ORACLE_DIR,
    load_gf_oracle,
    stage_rows_to_distrust,
)

NZ = GF_NZ


def _read(name):
    with (GF_ORACLE_DIR / name).open(newline="", encoding="ascii") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def drv():
    fixture = load_gf_oracle()
    gl = fixture.levels
    gs = fixture.surface

    deep_rows = _read("gf-deep-surface.csv")
    dkey = {
        (int(r["case"]), int(r["idx"]), int(r["arm"])): r for r in deep_rows
    }
    shal_rows = _read("gf-shallow-surface.csv")
    skey = {int(r["case"]): r for r in shal_rows}

    got = []
    pin = []
    for ci in range(fixture.ncol):
        case, idx, arm = (int(v) for v in fixture.key[ci])
        args = dict(
            u=gl["u"][ci], v=gl["v"][ci], w=gl["w"][ci], t=gl["t"][ci],
            qv=gl["qv"][ci], p=gl["p"][ci], pi=gl["pi"][ci],
            rho=gl["rho"][ci], dz8w=gl["dz8w"][ci], p8w=gl["p8w"][ci],
            rthften=gl["rthften"][ci], rqvften=gl["rqvften"][ci],
            rthraten=gl["rthraten"][ci], rthblten=gl["rthblten"][ci],
            rqvblten=gl["rqvblten"][ci], ht=gs["ht"][ci], hfx=gs["hfx"][ci],
            qfx=gs["qfx"][ci], xland=gs["xland"][ci],
            kpbl=int(gs["kpbl"][ci]), dt=gs["dt"][ci], dx=gs["dx"][ci],
            ishallow=int(gs["ishallow"][ci]), ichoice=int(gs["ichoice"][ci]),
        )
        got.append(gfdrv_column(**args))
        d = dkey[(case, idx, arm)]
        s = skey[case]
        fu = np.float32(d["up_fzu"])
        fd = np.float32(d["dn_fzu"])
        fs = np.float32(s["sh_fzu"])
        pin.append(
            gfdrv_column(
                **args,
                fzu_up=fu if fu > 0 else None,
                fzu_dn=fd if fd > 0 else None,
                fzu_sh=fs if fs > 0 else None,
            )
        )
    return dict(fixture=fixture, got=got, pin=pin, gl=gl, gs=gs,
                ncol=fixture.ncol, sl=fixture.stage_levels,
                ss=fixture.stage_surface,
                driver_exact=~stage_rows_to_distrust(fixture))


def _level(rs, name):
    return np.stack([np.asarray(r[name], dtype=np.float32) for r in rs])


def _scalar(rs, name):
    return np.array([np.float32(r[name]) for r in rs], dtype=np.float32)


def _int(rs, name):
    return np.array([int(r[name]) for r in rs], dtype=np.int64)


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


# ==========================================================================
# the tendencies GFDRV writes
# ==========================================================================
@pytest.mark.parametrize(
    "field",
    ["rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt_phy", "dvdt_phy",
     "gdc", "gdc2"],
)
def test_gfdrv_level_output_bitwise(drv, field):
    """Bitwise on the 208 columns where GFDRV and its own decomposition
    agree; see this module's docstring for the other 8."""
    ok = drv["driver_exact"]
    w, bad, loc = _ulp(_level(drv["pin"], field)[ok], drv["gl"][field][ok])
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]} k={loc[1] + 1}"


@pytest.mark.parametrize("field", ["raincv", "pratec"])
def test_gfdrv_precip_bitwise(drv, field):
    ok = drv["driver_exact"]
    w, bad, loc = _ulp(_scalar(drv["pin"], field)[ok], drv["gs"][field][ok])
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]}"


@pytest.mark.parametrize("field", ["htop", "hbot", "xmb_shallow"])
def test_gfdrv_xmb_free_surface_output_bitwise_everywhere(drv, field):
    """These three carry no deep ``xmb``, so the driver's mixed-precision
    residual cannot reach them and all 216 columns are exact."""
    w, bad, loc = _ulp(_scalar(drv["pin"], field), drv["gs"][field])
    assert bad == 0
    assert w == 0, f"{field}: {w} ULP at column {loc[0]}"


@pytest.mark.parametrize("field", ["gdc", "gdc2"])
def test_the_cloud_water_diagnostics_are_bitwise_on_all_216(drv, field):
    """``GDC``/``GDC2`` carry ``cupclw``, which ``cup_up_moisture`` builds
    without the mass flux, so the 8 mixed-precision columns cannot move them
    -- and do not.  A useful witness that the residual really is confined to
    ``xmb``."""
    w, bad, _ = _ulp(_level(drv["pin"], field), drv["gl"][field])
    assert bad == 0 and w == 0


# ==========================================================================
# the 8 columns, and where they come from
# ==========================================================================
def test_the_port_reproduces_the_stage_path_on_every_column(drv):
    """The port against ``run_cup_gf.F90``'s own capture: preparation, both
    schemes, both ``neg_check`` calls.  Bitwise on all 216 columns and all
    ten fields -- which is what proves the 8-column disagreement with GFDRV
    below is the DRIVER's, inherited, and not something this port introduces.
    """
    for f in ["outt", "outq", "outqc", "outu", "outv", "outts", "outqs",
              "outqcs"]:
        w, bad, loc = _ulp(_level(drv["pin"], f), drv["sl"][f])
        assert bad == 0
        assert w == 0, f"{f}: {w} ULP at column {loc[0]} k={loc[1] + 1}"
    idx = {}
    ss = drv["ss"]
    for i in range(ss["case"].shape[0]):
        idx[(int(ss["case"][i]), int(ss["idx"][i]), int(ss["arm"][i]))] = i
    order = [idx[tuple(int(v) for v in k)] for k in drv["fixture"].key]
    for f in ["pret", "prets"]:
        w, bad, _ = _ulp(_scalar(drv["pin"], f), ss[f][order])
        assert bad == 0 and w == 0, f


def test_the_eight_inexact_columns_are_the_fixtures_own(drv):
    """Not a new set: exactly ``stage_rows_to_distrust``.  If a change to the
    port made a 9th column inexact this fails, and if a change made one exact
    it fails too -- either is a finding, not an improvement."""
    ok = drv["driver_exact"]
    got = _level(drv["pin"], "rthcuten")
    want = drv["gl"]["rthcuten"]
    inexact = ~np.all(got.view(np.int32) == want.view(np.int32), axis=1)
    assert np.array_equal(inexact, ~ok)
    assert int(inexact.sum()) == 8


def test_on_those_eight_the_residual_is_amplitude_not_structure(drv):
    """A 1-2 ULP shift in ``xmb`` scaling the whole profile.  Bounded here,
    and shown to flip no branch: ``ktop_deep``, ``HTOP``, ``HBOT`` and the
    shallow indices agree with GFDRV on those columns too."""
    bad = ~drv["driver_exact"]
    got = _level(drv["pin"], "rthcuten")[bad]
    want = drv["gl"]["rthcuten"][bad]
    live = np.abs(want) > np.float32(1e-12)
    rel = np.abs(got[live] - want[live]) / np.abs(want[live])
    assert float(rel.max()) < 1e-4, f"grew to {float(rel.max()):.3e}"
    w, _, _ = _ulp(got, want)
    assert w <= 34, f"the 8-column ULP spread grew to {w}"
    assert np.array_equal(
        _int(drv["pin"], "ktop_deep")[bad],
        drv["gs"]["ktop_deep"].astype(np.int64)[bad],
    )
    assert _ulp(_scalar(drv["pin"], "htop")[bad], drv["gs"]["htop"][bad])[0] == 0
    assert _ulp(_scalar(drv["pin"], "hbot")[bad], drv["gs"]["hbot"][bad])[0] == 0


@pytest.mark.parametrize(
    "field", ["ktop_deep", "k22_shallow", "kbcon_shallow", "ktop_shallow"]
)
def test_gfdrv_surface_indices_exact(drv, field):
    got = _int(drv["pin"], field)
    want = drv["gs"][field].astype(np.int64)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, f"{field}: columns {bad[:8].tolist()}"


def test_every_column_is_covered(drv):
    assert drv["ncol"] == 216
    assert len(drv["pin"]) == 216


# ==========================================================================
# the gating, which is where composition mistakes live
# ==========================================================================
def test_the_cuten_gate_erases_a_deep_cloud_that_did_not_rain(drv):
    """:727-733.  ``pret <= 0`` sets ``cuten = 0`` AND zeroes ``kbcon`` and
    ``ktop`` -- but ``ktop_deep`` was already taken from the un-zeroed
    ``ktop`` at :726, so the diagnostic and the gate disagree by design.  The
    fixture has columns on both sides."""
    pret = _scalar(drv["pin"], "pret")
    cuten = _scalar(drv["pin"], "cuten")
    assert np.array_equal(cuten, (pret > 0).astype(np.float32))
    assert np.any(cuten == 1) and np.any(cuten == 0)
    dry = cuten == 0
    assert np.all(_int(drv["pin"], "ktop")[dry] == 0)
    assert np.all(_int(drv["pin"], "kbcon")[dry] == 0)
    assert np.array_equal(
        _int(drv["pin"], "ktop_deep"), drv["gs"]["ktop_deep"].astype(np.int64)
    )


def test_cutens_is_off_on_the_ishallow_zero_arm(drv):
    """:331.  Half the fixture runs with the shallow arm switched off, and on
    that half every shallow contribution must vanish from the OUTPUT even
    though the shallow diagnostics are also zeroed at :278-287."""
    arm = np.array([int(k[2]) for k in drv["fixture"].key])
    cutens = _scalar(drv["pin"], "cutens")
    assert np.all(cutens[arm == 0] == 0.0)
    assert np.all(_scalar(drv["pin"], "xmb_shallow")[arm == 0] == 0.0)


def test_cutens_is_also_off_where_the_shallow_arm_produced_no_mass_flux(drv):
    """:525, and it is decided from ``xmbs`` -- not from ``prets``, and not
    after ``neg_check``.  On the ishallow = 1 arm the fixture has columns
    where the shallow scheme was rejected (``xmbs == 0``) and columns where
    it fired."""
    arm = np.array([int(k[2]) for k in drv["fixture"].key])
    on = arm == 1
    xmbs = _scalar(drv["pin"], "xmb_shallow")[on]
    cutens = _scalar(drv["pin"], "cutens")[on]
    assert np.any(xmbs > 0) and np.any(xmbs == 0)
    assert np.array_equal(cutens, (xmbs > 0).astype(np.float32))


def test_pratec_is_gated_by_an_or_over_all_three_arms(drv):
    """:800.  A column where only the shallow arm rained still gets
    ``RAINCV``; a port that gates on the deep ``pret`` alone loses it."""
    pret = _scalar(drv["pin"], "pret")
    prets = _scalar(drv["pin"], "prets")
    pratec = drv["gs"]["pratec"]
    fired = (pret > 0) | (prets > 0)
    assert np.all(pratec[~fired] == 0.0)
    assert np.any(fired)


def test_raincv_is_pratec_times_the_timestep(drv):
    dt = drv["gs"]["dt"]
    want = (drv["gs"]["pratec"] * dt).astype(np.float32)
    assert _ulp(want, drv["gs"]["raincv"])[0] == 0


def test_htop_and_hbot_start_crossed(drv):
    """``HBOT = REAL(KTE)`` and ``HTOP = REAL(KTS)`` (:349-350) -- the top
    diagnostic starts at the bottom index and vice versa, so an untouched
    column reports HTOP 1 and HBOT 40.  This is not a typo to fix."""
    pret = _scalar(drv["pin"], "pret")
    prets = _scalar(drv["pin"], "prets")
    quiet = ~((pret > 0) | (prets > 0))
    assert np.any(quiet)
    assert np.all(drv["gs"]["htop"][quiet] == np.float32(1.0))
    assert np.all(drv["gs"]["hbot"][quiet] == np.float32(NZ))


def test_rthcuten_is_the_only_output_divided_by_exner(drv):
    """:747 divides the temperature sum by ``pi`` and :748-752 divide nothing.
    Reconstructed here from the raw contributions to prove the division is
    where the port puts it."""
    ok = drv["driver_exact"]
    pi = drv["gl"]["pi"][ok]
    got = _level(drv["pin"], "rthcuten")[ok]
    want = drv["gl"]["rthcuten"][ok]
    assert _ulp(got, want)[0] == 0
    # the same sum without the division is not the answer anywhere it is
    # nonzero, which is what makes the assertion above load-bearing
    live = np.abs(want) > 0
    assert np.any(live)
    assert not np.allclose((got * pi).astype(np.float32)[live], want[live])


def test_the_258k_split_sends_one_number_to_one_of_two_slots(drv):
    """:820-840.  ``RQICUTEN`` and ``RQCCUTEN`` are never both nonzero at a
    level, ``GDC`` and ``GDC2`` likewise, and the split is on ``t2d`` -- the
    UNforced temperature -- not on ``tn``."""
    qi = drv["gl"]["rqicuten"]
    qc = drv["gl"]["rqccuten"]
    assert not np.any((qi != 0) & (qc != 0))
    cold = drv["gl"]["t"] < np.float32(258.0)
    assert np.all(qc[cold] == 0.0)
    assert np.all(qi[~cold] == 0.0)
    assert np.any(cold) and np.any(~cold)
    g1 = drv["gl"]["gdc"]
    g2 = drv["gl"]["gdc2"]
    assert not np.any((g1 != 0) & (g2 != 0))
    assert np.all(g1[cold] == 0.0)
    assert np.all(g2[~cold] == 0.0)


def test_the_condensate_reaching_both_slots_is_the_same_number(drv):
    """One value, routed -- so a port may compute it once."""
    qi = drv["gl"]["rqicuten"]
    qc = drv["gl"]["rqccuten"]
    assert _ulp((qi + qc).astype(np.float32), np.where(qi != 0, qi, qc))[0] == 0


# ==========================================================================
# what the modelled tgammaf costs at the WRF boundary
# ==========================================================================
def test_no_branch_flips_when_tgammaf_is_modelled(drv):
    """The structural half of the ``tgammaf`` answer, and the good news.

    Over all 216 columns, modelling ``tgammaf`` instead of reproducing it
    moves no integer and no gate: the deep ``ierr``, ``ktop_deep``,
    ``ktop_shallow``, ``cuten`` and ``cutens`` are identical between the two
    runs.  The disagreement is entirely amplitude.
    """
    for f in ["ktop_deep", "ktop_shallow", "k22_shallow", "kbcon_shallow"]:
        assert np.array_equal(_int(drv["got"], f), _int(drv["pin"], f)), f
    for f in ["cuten", "cutens"]:
        assert np.array_equal(_scalar(drv["got"], f), _scalar(drv["pin"], f)), f
    ig = np.array([int(r["deep"]["ierr"]) for r in drv["got"]])
    ip = np.array([int(r["deep"]["ierr"]) for r in drv["pin"]])
    assert np.array_equal(ig, ip)


def test_the_tgammaf_residual_at_the_driver_boundary(drv):
    """And the amplitude half, which is the number phase 3 inherits.

    Three sizes, all measured on this fixture:

    * the deep mass flux moves by up to **7.3 per cent**, median 1.9 -- the
      figure the deep gate pins by perturbing ``fzu`` directly;
    * the shallow mass flux by up to 8.4e-3;
    * and ``RTHCUTEN`` by **more than 100 per cent on a single lane**, which
      is neither of the above and is the reason this test exists.  On the
      ``ishallow = 1`` arm the shallow and deep tendencies have opposite
      signs at some levels, so :747 sums two numbers that nearly cancel, and
      a sub-per-cent shift in either mass flux is order-unity in their
      difference.  A third cancellation, on top of ``xk``'s.

    Normalised by each column's own peak ``RTHCUTEN`` the error is 7.3 per
    cent, i.e. it IS the mass-flux error and nothing else: the deep tendency
    profile is linear in ``xmb``, so ``xmb``'s residual scales the whole
    column.  The two numbers agree to four digits, which is the cleanest
    available statement that ``tgammaf`` is the only thing moving.

    A "relative-error tolerance" gate would have to be set at 200 per cent to
    pass the cancellation lane while 7 per cent covers everything else.  That
    is the argument for grading this bitwise against a pinned ``fzu``.
    """
    want = drv["gl"]["rthcuten"]
    got = _level(drv["got"], "rthcuten")
    live = np.abs(want) > np.float32(1e-12)
    rel = np.zeros_like(want)
    rel[live] = np.abs(got[live] - want[live]) / np.abs(want[live])
    assert float(rel.max()) > 1.0, "the cancellation lane vanished"

    xg = np.array([float(r["deep"]["xmb"]) for r in drv["got"]])
    xp = np.array([float(r["deep"]["xmb"]) for r in drv["pin"]])
    m = xp > 0
    xrel = np.abs(xg[m] - xp[m]) / xp[m]
    assert 0.05 < float(xrel.max()) < 0.1
    assert float(np.median(xrel)) > 0.01

    # normalised by the column's own peak, the tendency error is the mass
    # flux error
    peak = np.abs(want).max(axis=1, keepdims=True)
    scaled = np.where(peak > 0, np.abs(got - want) / np.maximum(peak, 1e-30), 0.0)
    assert abs(float(scaled.max()) - float(xrel.max())) < 1e-4


def test_the_pinned_run_is_bitwise_on_every_output_word(drv):
    """The single claim this file exists to make, counted.

    With the oracle's own ``fzu``, the float32 reference reproduces GFDRV
    word for word on the 208 columns where the driver's own decomposition is
    exact: eight level fields and five surface fields, 208 * 40 * 8 + 208 * 5
    = 67600 words, zero differing.
    """
    ok = drv["driver_exact"]
    n = 0
    for f in ["rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt_phy",
              "dvdt_phy", "gdc", "gdc2"]:
        a = _level(drv["pin"], f)[ok]
        b = drv["gl"][f][ok]
        assert np.array_equal(a.view(np.int32), b.view(np.int32)), f
        n += a.size
    for f in ["raincv", "pratec", "htop", "hbot", "xmb_shallow"]:
        a = _scalar(drv["pin"], f)[ok]
        b = drv["gs"][f][ok]
        assert np.array_equal(a.view(np.int32), b.view(np.int32)), f
        n += a.size
    assert int(ok.sum()) == 208
    assert n == 208 * NZ * 8 + 208 * 5
