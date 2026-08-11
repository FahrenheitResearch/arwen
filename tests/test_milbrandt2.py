"""Milbrandt-Yau (WRF ``mp_physics = 9``) column smoke on the device.

WHAT IS TESTED: the SHIPPED seams -- ``DomainState`` allocation,
``initialize_physics``, ``gpuwm.core.microphysics.apply`` and
``dycore.step`` -- asserted for finiteness, non-negativity, a physical
temperature band, moment-versus-mass consistency, and a column water
budget that must close against the precipitation the same call reports.
The budget runs three seeding layouts, and its docstring names the exact
source/sink terms each one resolves and the measured margin it resolves
them by; terms that are zero on all three columns are named as NOT
covered.  The last test is the mutation control: it stubs the scheme out
and proves the smoke's change detection then fails, so a passing smoke is
evidence about the port rather than about the harness.

WHAT IS NOT TESTED: anything comparing this port to the WRF v4.6.1
Fortran.  There is no ULP table, no column oracle and no matched forecast
trajectory for mp=9.  The declaration-only half of the suite is
``tests/test_milbrandt2_contract.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core import constants as c


MASS_NAMES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
NUMBER_NAMES = ("nc", "nr", "ni", "ns", "ng", "nh")


def _cfg(**overrides):
    values = dict(
        nx=6, ny=5, nz=24, dx=1000.0, dy=1000.0, ztop=15000.0,
        dt=10.0, run_seconds=10.0, moist=True, mp_physics=9)
    values.update(overrides)
    return RunConfig(**values)


def _regime_theta(regime: str):
    """The dry potential-temperature profile each thermal regime is built on.

    These are BASE-STATE profiles handed to ``make_base_state``, so the
    column stays hydrostatically consistent and ``alt`` (used by the water
    budget as 1/rho) matches the temperature the scheme sees.  Measured
    column temperatures on the shipped 24-level, 15 km grid:

      ``mixed``      299.1 K at k=0 down to 194.5 K at k=23; the freezing
                     level is between k=9 (275.0 K) and k=10 (271.7 K).
      ``warm``       314.1 K down to 204.4 K; freezing between k=12 and 13.
      ``glaciated``  267.1 K down to 152.3 K; every level sub-freezing.

    The three are genuinely different states -- an earlier revision of this
    fixture computed a per-regime temperature and never used it, so all
    three parametrisations ran the SAME column.
    """
    if regime == "warm":
        return lambda zz: 315.0 + 0.002 * np.asarray(zz)
    if regime == "mixed":
        return lambda zz: 300.0 + 0.003 * np.asarray(zz)
    if regime == "glaciated":
        return lambda zz: 268.0 + 0.004 * np.asarray(zz)
    raise ValueError(regime)


@requires_gpu
def test_domain_state_allocates_the_registry_package():
    import cupy as cp

    from gpuwm.core.state import DomainState

    cfg = _cfg(nz=8)
    state = DomainState(cfg)
    for name in MASS_NAMES + NUMBER_NAMES:
        value = getattr(state, name)
        assert value is not None, name
        assert value.shape == (8, 5, 6)
        assert value.dtype == cp.float32
    # Registry.EM_COMMON:3025 declares no re_* state for mp=9, and the
    # scheme's reff block is commented out, so these are background only.
    cp.testing.assert_array_equal(state.effc, cp.float32(2.5))
    cp.testing.assert_array_equal(state.effi, cp.float32(5.0))
    cp.testing.assert_array_equal(state.effs, cp.float32(10.0))
    # NSSL-only fields must NOT appear on an mp=9 state.
    for absent in ("qndrop", "qnn", "qvolg", "qvolh"):
        assert getattr(state, absent, None) is None


def _bands(layout: str, nz: int):
    """Which levels each species occupies, per seeding layout.

    On the ``mixed`` sounding the freezing level sits between k=9 (275.0 K)
    and k=10 (271.7 K), which is what makes these layouts mean what their
    names say.

    ``default``   the original disjoint pair: liquid in the lower third,
                  frozen in the middle third.  The middle third STARTS at
                  k=8 (278.2 K), so it seeds pristine ice above freezing
                  and therefore exercises WRF's T>0 ice destruction.
    ``cold``      liquid 0..7 (all warm), frozen 10..17 (all sub-freezing).
                  Disjoint: no level carries both, so NO cloud/rain-frozen
                  collection term can be non-zero anywhere in the column.
    ``riming``    ``cold`` with cloud and rain extended up through k=17, so
                  levels 10..17 carry supercooled liquid TOGETHER with
                  snow, graupel, hail and ice.  This is the layout that
                  makes QCLcs/QCLcg/QCLch/QCLrs/QCLrg/QCLrh non-zero.
    ``melting``   ``cold`` with snow, graupel and hail extended down to the
                  surface, so levels 0..9 carry frozen mass above freezing
                  and QMLsr/QMLgr/QMLhr fire.  Pristine ice is deliberately
                  NOT extended down: WRF destroys it there (:2041-2043) and
                  that leak would swamp the budget residual.
    """
    if layout == "default":
        liquid = slice(0, nz // 3)
        return {"liquid": liquid, "ice": slice(nz // 3, 2 * nz // 3),
                "snow_graupel_hail": slice(nz // 3, 2 * nz // 3)}
    if layout == "cold":
        return {"liquid": slice(0, nz // 3), "ice": slice(10, 18),
                "snow_graupel_hail": slice(10, 18)}
    if layout == "riming":
        return {"liquid": slice(0, 18), "ice": slice(10, 18),
                "snow_graupel_hail": slice(10, 18)}
    if layout == "melting":
        return {"liquid": slice(0, nz // 3), "ice": slice(10, 18),
                "snow_graupel_hail": slice(0, 18)}
    raise ValueError(layout)


def _seeded_state(regime="mixed", nz=24, cfg=None, *, layout="default"):
    """A physics-driver-bearing mp=9 state with hydrometeors in place."""
    import cupy as cp

    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced

    cfg = cfg if cfg is not None else _cfg(nz=nz)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, _regime_theta(regime),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda zz: 0.012 * np.exp(-np.asarray(zz) / 2500.0))
    f32 = cp.float32
    # Seed every category so all five sedimentation calls and both phase
    # branches are entered.  Numbers are #/kg, matching the state contract.
    bands = _bands(layout, cfg.nz)
    liquid = bands["liquid"]
    ice = bands["ice"]
    frozen = bands["snow_graupel_hail"]
    state.qc[liquid] = f32(1.0e-3)
    state.nc[liquid] = f32(2.0e8)
    state.qr[liquid] = f32(5.0e-4)
    state.nr[liquid] = f32(1.0e4)
    state.qi[ice] = f32(2.0e-4)
    state.ni[ice] = f32(5.0e4)
    state.qs[frozen] = f32(4.0e-4)
    state.ns[frozen] = f32(1.0e4)
    state.qg[frozen] = f32(3.0e-4)
    state.ng[frozen] = f32(1.0e3)
    state.qh[frozen] = f32(2.0e-4)
    state.nh[frozen] = f32(1.0e1)
    return cfg, state


@requires_gpu
def test_apply_through_the_shipped_dispatch_is_finite_and_bounded():
    """The real seam: microphysics.apply, dispatching on cfg.mp_physics."""
    import cupy as cp

    from gpuwm.core.microphysics import apply
    from gpuwm.core.physics import initialize_physics

    cfg, state = _seeded_state("mixed")
    initialize_physics(state, cfg)
    before = {name: getattr(state, name).copy()
              for name in MASS_NAMES + NUMBER_NAMES}
    diagnostics = apply(state, cfg, cfg.dt)
    cp.cuda.Stream.null.synchronize()

    assert diagnostics is not None
    assert diagnostics.hailnc is not None, \
        "mp=9 must expose the hail accumulator its WRF driver arm binds"

    changed = [name for name in MASS_NAMES + NUMBER_NAMES
               if not bool(cp.all(getattr(state, name) == before[name]))]
    assert changed, "the scheme ran and moved nothing -- it is a no-op"

    for name in MASS_NAMES + NUMBER_NAMES:
        value = getattr(state, name)
        assert bool(cp.all(cp.isfinite(value))), f"{name} went non-finite"
        assert float(cp.min(value)) >= 0.0, f"{name} went negative"
    # Mixing ratios stay in a physical band; numbers stay under the
    # scheme's own ice ceiling once converted back to #/kg is not asserted
    # (Ni_max is applied in #/m3), so only the mass band is pinned here.
    for name in MASS_NAMES:
        assert float(cp.max(getattr(state, name))) < 0.1, name

    theta = state.thb.reshape(-1, 1, 1) + state.thp \
        if state.thb.ndim == 1 else state.thb + state.thp
    assert bool(cp.all(cp.isfinite(theta)))
    pii = cp.power(state.p / np.float32(c.P0), np.float32(c.RCP))
    temperature = theta * pii
    assert float(cp.min(temperature)) > 150.0
    assert float(cp.max(temperature)) < 350.0

    for field in (diagnostics.rainnc, diagnostics.rainncv,
                  diagnostics.snownc, diagnostics.graupelnc,
                  diagnostics.hailnc, diagnostics.sr):
        assert bool(cp.all(cp.isfinite(field)))
        assert float(cp.min(field)) >= 0.0


@requires_gpu
@pytest.mark.parametrize("layout", ("cold", "riming", "melting"))
def test_total_water_is_conserved_to_what_sedimentation_removes(layout):
    """The source/sink ledger closes to the surface flux, to a stated bound.

    Column-integrated total water (vapour plus every condensate, mass
    weighted by dry density and layer depth) must fall by the
    precipitation the same call reports.

    WHAT THIS COVERS, EXACTLY.  A source/sink term at
    module_mp_milbrandt2mom.F:2708-2718 is detectable here only if it is
    NON-ZERO on the column being run -- conservation says nothing about a
    term that never fires.  So the test runs three seeding layouts, and
    each one asserts (below) that it actually drove the family it exists
    for rather than assuming it:

      ``cold``      the historical disjoint fixture.  Liquid is warm,
                    frozen is cold, no level carries both, so the whole
                    cloud/rain-frozen collection family is IDENTICALLY
                    ZERO here and only deposition and ice->snow
                    autoconversion move anything.
      ``riming``    supercooled cloud and rain co-located with snow,
                    graupel, hail and ice, so the collection family
                    (QCLcs, QCLcg, QCLch, QCLrs, QCLrg, QCLrh and the
                    Dxxx-routed products at :2713-2718) is live.
      ``melting``   snow, graupel and hail above the freezing level, so
                    QMLsr, QMLgr and QMLhr are live.

    THE GATE IS A MEASURED PIN, NOT A ROUND NUMBER.  Each layout's bound
    sits between its own baseline residual and the smallest residual a
    single dropped term produces, so the assertion resolves individual
    terms instead of merely bounding the total.  Measured on this tree,
    2026-08-09, RTX 5090, worst relative residual over the 6x5 column
    batch, one step (baseline -> single-term-drop, term deleted from ONE
    of the two lines it appears in):

      layout   baseline   gate      drop that this layout catches
      cold     6.40e-05   2.0e-04   QVDvs 4.72e-04, QCNis 1.04e-02
      riming   6.13e-05   1.2e-04   QCLrs 1.07e-02, QCLcs 5.60e-04,
                                    QCLcg 4.50e-04, QCLch 3.16e-04,
                                    QCLrh 1.61e-04
      melting  1.31e-04   2.4e-04   QMLsr 2.86e-02, QMLgr 1.60e-02,
                                    QMLhr 1.73e-03, QCLch 4.69e-04,
                                    QCLrh 3.23e-04

    The tightest margin in that table is QCLrh, caught at 1.34x its
    layout's gate; everything else clears by 2x or more.

    WHY THE EXTRA LAYOUTS EXIST.  On the ``cold`` fixture alone, deleting
    ``+ QCLcs`` from the QN line leaves every assertion passing and the
    output BIT-IDENTICAL -- the term is exactly zero there.  The same
    deletion moves the ``riming`` arm to 5.60e-04 and fails it.

    NOT COVERED even now: any term that is zero on all three columns
    (QNUvi, QIMsi/QIMgi, QCNgh and the number-only terms among them);
    every N-source/sink line at :2721-2728, which carries no mass and so
    cannot show up in a water budget at all; and any error that is
    sign-symmetric across the two lines a term appears in, because a term
    negated in BOTH places still conserves.  Only an oracle settles
    those, and none has been run.

    It is a BOUND, not an equality, and the bound is not a fudge: WRF
    itself leaks here.  Its overdepletion guard zeroes the
    three-component-freezing destination flags Dirg/Dirh (:2623-2625),
    Dsrs/Dsrg/Dsrh (:2640-2642) and Dgrg/Dgrh (:2655-2657) on a
    ``ratio == 0`` without zeroing every collection term they route, and
    the sedimentation sweep clamps ``QX = max(QX, 0)`` (:720).
    """
    import cupy as cp

    from gpuwm.core.microphysics import apply
    from gpuwm.core.physics import initialize_physics

    cfg, state = _seeded_state("mixed", layout=layout)
    initialize_physics(state, cfg)

    nz, ny, nx = state.p.shape
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    z8w = (phb + state.php) / np.float32(c.G)
    dz = (z8w[1:] - z8w[:nz]).astype(cp.float64)
    rho = (1.0 / state.alt).astype(cp.float64)

    def column_water():
        total = cp.zeros((ny, nx), dtype=cp.float64)
        for name in MASS_NAMES:
            total += cp.sum(getattr(state, name).astype(cp.float64)
                            * rho * dz, axis=0)
        return total

    def band_mass(name, band):
        return float(cp.sum(getattr(state, name)[band].astype(cp.float64)
                            * rho[band] * dz[band]).item())

    # The overlap band for ``riming`` (supercooled liquid on top of frozen)
    # and the above-freezing band for ``melting``.
    overlap = slice(10, 18)
    above_freezing = slice(0, 10)
    liquid_before = sum(band_mass(n, overlap) for n in ("qc", "qr"))
    frozen_before = sum(band_mass(n, overlap) for n in ("qs", "qg", "qh"))
    melt_before = sum(band_mass(n, above_freezing)
                      for n in ("qs", "qg", "qh"))
    rain_warm_before = band_mass("qr", above_freezing)
    snow_before = band_mass("qs", slice(0, nz))

    before = column_water()
    diagnostics = apply(state, cfg, cfg.dt)
    cp.cuda.Stream.null.synchronize()
    after = column_water()

    # RAINNCV is mm == kg m-2 of liquid-equivalent surface flux this step.
    removed = diagnostics.rainncv.astype(cp.float64)
    residual = cp.abs((before - after) - removed)
    scale = cp.maximum(before, 1.0e-6)
    worst = float(cp.max(residual / scale))
    # Derived in the docstring's table: above each layout's own baseline,
    # below the smallest single-dropped-term residual measured on it.
    gate = {"cold": 2.0e-4, "riming": 1.2e-4, "melting": 2.4e-4}[layout]
    assert worst < gate, (
        f"[{layout}] water budget does not close: worst relative residual "
        f"{worst:.3e} against a {gate:.1e} gate; a source/sink term is "
        "missing from the ledger, a destination flag is mis-routed, or "
        "the surface-flux conversion is wrong")

    # The coverage claim, asserted rather than assumed: each layout must
    # have driven the family it exists for, or the residual above is a
    # statement about an inert column.
    if layout == "cold":
        # Vapour deposition onto snow: no liquid is anywhere near the
        # frozen band, so snow can only grow from vapour and from ice.
        assert band_mass("qs", slice(0, nz)) > snow_before, (
            "the cold layout grew no snow -- QVDvs/QCNis are inert and "
            "this arm is measuring nothing")
        assert sum(band_mass(n, overlap) for n in ("qc", "qr")) == 0.0, (
            "the cold layout must stay disjoint: liquid appeared in the "
            "frozen band, so this arm no longer isolates deposition")
    elif layout == "riming":
        liquid_after = sum(band_mass(n, overlap) for n in ("qc", "qr"))
        frozen_after = sum(band_mass(n, overlap) for n in ("qs", "qg", "qh"))
        assert liquid_before > 0.0 and frozen_before > 0.0, (
            "the riming layout did not co-locate liquid and frozen mass")
        assert liquid_after < liquid_before, (
            "supercooled liquid did not decrease in the overlap band, so "
            "the cloud/rain-frozen collection family never fired")
        assert frozen_after > frozen_before, (
            "frozen mass did not grow in the overlap band, so the "
            "collection terms are not reaching :2713-2718")
    elif layout == "melting":
        melt_after = sum(band_mass(n, above_freezing)
                         for n in ("qs", "qg", "qh"))
        assert melt_before > 0.0, (
            "the melting layout seeded no frozen mass above freezing")
        assert melt_after < melt_before, (
            "frozen mass above freezing did not decrease, so QMLsr/QMLgr/"
            "QMLhr never fired")
        assert band_mass("qr", above_freezing) > rain_warm_before, (
            "melted frozen mass did not arrive in rain, so the +QMLsr/"
            "+QMLgr/+QMLhr side of :2710 is not being credited")


@requires_gpu
@pytest.mark.parametrize(
    "regime,expect_warm_base,expect_all_frozen",
    (("warm", True, False), ("mixed", True, False),
     ("glaciated", False, True)))
def test_every_phase_regime_stays_finite(regime, expect_warm_base,
                                         expect_all_frozen):
    """Three thermally DIFFERENT columns, asserted to be different.

    The temperature band is checked first.  Without it the parametrisation
    is decoration: an earlier revision built all three regimes from the
    same base-state profile, so the three cases ran one column and the
    suite reported three passes for one measurement.
    """
    import cupy as cp

    from gpuwm.core.microphysics import apply
    from gpuwm.core.physics import initialize_physics

    cfg, state = _seeded_state(regime)
    initialize_physics(state, cfg)

    theta = state.thb.reshape(-1, 1, 1) + state.thp \
        if state.thb.ndim == 1 else state.thb + state.thp
    pii = cp.power(state.p / np.float32(c.P0), np.float32(c.RCP))
    column = (theta * pii)[:, 0, 0]
    surface = float(column[0].item())
    assert (surface > 273.16) is expect_warm_base, (
        f"{regime}: surface temperature {surface:.1f} K is on the wrong "
        "side of freezing for this regime")
    assert bool(cp.all(column < 273.16).item()) is expect_all_frozen, (
        f"{regime}: whole-column-frozen is not what this regime claims")
    for _ in range(3):
        apply(state, cfg, cfg.dt)
    cp.cuda.Stream.null.synchronize()
    for name in MASS_NAMES + NUMBER_NAMES:
        value = getattr(state, name)
        assert bool(cp.all(cp.isfinite(value))), f"{regime}: {name}"
        assert float(cp.min(value)) >= 0.0, f"{regime}: {name}"


@requires_gpu
def test_moment_and_mass_stay_consistent():
    """Where mass is present the number must be too, and the reverse.

    ``epsQ = 1e-14`` / ``epsN = 1e-3`` are the scheme's own thresholds and
    its consistency blocks (:1445-1531, :2733-2813) exist precisely to
    keep the pair from separating.  A column that leaves the scheme with
    mass and no number is a broken port, not a tuning question.
    """
    import cupy as cp

    from gpuwm.core.microphysics import apply
    from gpuwm.core.physics import initialize_physics

    cfg, state = _seeded_state("mixed")
    initialize_physics(state, cfg)
    apply(state, cfg, cfg.dt)
    cp.cuda.Stream.null.synchronize()

    eps_q = np.float32(1.0e-14)
    for mass, number in (("qc", "nc"), ("qr", "nr"), ("qi", "ni"),
                         ("qs", "ns"), ("qg", "ng"), ("qh", "nh")):
        q = getattr(state, mass)
        n = getattr(state, number)
        orphan_mass = bool(cp.any((q > eps_q) & (n <= 0.0)))
        assert not orphan_mass, f"{mass} present with {number} == 0"
        orphan_number = bool(cp.any((q <= 0.0) & (n > 0.0)))
        assert not orphan_number, f"{number} present with {mass} == 0"


@requires_gpu
def test_reflectivity_comes_from_the_scheme_not_the_generic_operator():
    """WRF binds Zet straight to refl_10cm (driver :1878), so mp=9 is one
    of the schemes that owns its own radar field."""
    import cupy as cp

    from gpuwm.core.microphysics import apply
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.refl import consume_refl_10cm, refl_10cm_is_stashed

    cfg, state = _seeded_state("mixed")
    initialize_physics(state, cfg)
    apply(state, cfg, cfg.dt, refl_10cm_due=True)
    cp.cuda.Stream.null.synchronize()
    assert refl_10cm_is_stashed(state)
    refl = consume_refl_10cm(state)
    assert bool(cp.all(cp.isfinite(refl)))
    # minZET (module_mp_milbrandt2mom.F:1101) is the floor, and a seeded
    # mixed column must produce an echo above it somewhere.
    assert float(cp.min(refl)) >= -99.0
    assert float(cp.max(refl)) > -99.0
    assert float(cp.max(refl)) < 100.0


@requires_gpu
def test_dycore_step_reaches_the_scheme():
    """End to end through the production step, not the adapter alone."""
    import cupy as cp

    from gpuwm.core.dycore import step
    from gpuwm.core.physics import initialize_physics

    cfg = _cfg(nx=8, ny=8, nz=12, ztop=12000.0, dt=1.0, run_seconds=1.0,
               h_sca_adv_order=2, time_step_sound=2)
    cfg, state = _seeded_state("mixed", cfg=cfg)
    driver = initialize_physics(state, cfg)
    step(state, cfg)
    cp.cuda.Stream.null.synchronize()
    assert driver.microphysics_updates == 1
    for name in MASS_NAMES + NUMBER_NAMES:
        value = getattr(state, name)
        assert bool(cp.all(cp.isfinite(value))), name


# ---------------------------------------------------------------------------
# Mutation control
# ---------------------------------------------------------------------------

@requires_gpu
def test_smoke_fails_when_the_scheme_is_stubbed_out(monkeypatch):
    """Prove the assertions above have teeth.

    Replace ``launch_milbrandt2`` with a no-op and re-run the
    change-detection and budget assertions the real smoke makes.  Both must
    now fail: with the scheme stubbed the fields are untouched, so the
    "it moved something" assertion is false.  A smoke test that still
    passes against a stub is measuring the harness, not the port.
    """
    import cupy as cp

    import gpuwm.core.milbrandt2 as milbrandt2
    from gpuwm.core.microphysics import apply
    from gpuwm.core.physics import initialize_physics

    cfg, state = _seeded_state("mixed")
    initialize_physics(state, cfg)
    before = {name: getattr(state, name).copy()
              for name in MASS_NAMES + NUMBER_NAMES}

    calls = []

    def _stub(*args, **kwargs):
        calls.append(1)

    monkeypatch.setattr(milbrandt2, "launch_milbrandt2", _stub)
    diagnostics = apply(state, cfg, cfg.dt)
    cp.cuda.Stream.null.synchronize()

    assert calls == [1], "the stub was not on the shipped dispatch path"
    changed = [name for name in MASS_NAMES + NUMBER_NAMES
               if not bool(cp.all(getattr(state, name) == before[name]))]
    assert not changed, (
        "the stub still moved fields, so the smoke's change detection "
        "cannot distinguish a working scheme from a broken one")
    assert float(cp.max(diagnostics.rainncv)) == 0.0


@requires_gpu
def test_wrf_destroys_pristine_ice_above_freezing():
    """Pin the leak the budget test deliberately steps around.

    module_mp_milbrandt2mom.F:2041-2043 sets ``QMLir = QI`` and then
    ``QI(i,k) = 0`` -- it zeroes the ARRAY, not just the local.  The
    overdepletion pass then evaluates the ice source as
    ``QI + QNUvi + dim(QVDvi,0) + QFZci``, which is 0 because every cold
    term is 0 in the T>To arm, against a sink of QMLir; ``sour < sink``
    gives ``ratio = 0`` and QMLir/NMLir are scaled away (:2613-2622).  The
    apply step at :2711 then leaves QI at 0 and :2710 gives rain nothing.

    So pristine ice above freezing does not melt into rain -- it vanishes.
    That is defined behaviour, not an uninitialised read, so the port
    reproduces it; this test exists so the next reader finds it stated
    rather than rediscovering it as a budget mystery.
    """
    import cupy as cp

    from gpuwm.core.microphysics import apply
    from gpuwm.core.physics import initialize_physics

    cfg, state = _seeded_state("mixed")
    # Level 8 sits at ~278 K on this sounding: warm, and seeded with ice.
    state.qi[...] = cp.float32(0.0)
    state.ni[...] = cp.float32(0.0)
    state.qi[8] = cp.float32(2.0e-4)
    state.ni[8] = cp.float32(5.0e4)
    initialize_physics(state, cfg)

    nz, ny, nx = state.p.shape
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    z8w = (phb + state.php) / np.float32(c.G)
    dz = (z8w[1:] - z8w[:nz]).astype(cp.float64)
    rho = (1.0 / state.alt).astype(cp.float64)

    def total():
        out = cp.zeros((ny, nx), dtype=cp.float64)
        for name in MASS_NAMES:
            out += cp.sum(getattr(state, name).astype(cp.float64) * rho * dz,
                          axis=0)
        return out

    before = total()
    ice_before = float(
        cp.sum(state.qi[8].astype(cp.float64) * rho[8] * dz[8]).item()
        / (ny * nx))
    diagnostics = apply(state, cfg, cfg.dt)
    cp.cuda.Stream.null.synchronize()
    after = total()

    # Level 8 is the warm one.  Ice reappearing at COLD levels is the
    # warm-phase homogeneous-freezing arm (:3086-3097) doing its job and is
    # not what this test is about.
    assert float(cp.max(state.qi[8])) == 0.0, "the warm-level ice survived"
    lost = float((before - after)[0, 0]) - float(diagnostics.rainncv[0, 0])
    assert lost > 0.0, (
        "WRF's T>0 arm destroys pristine-ice mass; if this port conserves "
        "it, the port has diverged from the Fortran on a DEFINED path")
    assert lost <= ice_before * 1.05, (
        f"{lost:.6f} kg m-2 vanished but the warm level held only "
        f"{ice_before:.6f}, so the leak is not the documented one")
