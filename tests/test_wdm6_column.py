"""WDM6 (mp_physics=16) column smoke through the SHIPPED seams.

WHAT THIS FILE IS EVIDENCE OF, EXACTLY.  It runs the real
``initialize_physics`` -> ``gpuwm.core.microphysics.apply`` path at
``mp_physics = 16`` on a seeded mixed-phase column and asserts that the
result is a physically admissible state: every field finite, no negative
hydrometeor mass or number, number moments inside WDM6's own clamps, water
substance conserved to the surface flux the scheme itself reports, and the
scheme's radiation radii inside the driver's bounds.  It also asserts that
the call CHANGED something, and the mutation control below proves the
assertions have teeth: a stubbed scheme that returns the state untouched
fails this file.

WHAT IT IS NOT EVIDENCE OF.  It is NOT an oracle comparison.  Nothing here
runs, links to, or compares against WRF's own ``module_mp_wdm6.F``, so no
statement about ULP, relative agreement or trajectory match may be sourced
from this file.  The registry option ``wdm6-mp16`` says the same thing in
its first warning, and the two must stay in agreement.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = pytest.mark.gpu


_NZ, _NY, _NX = 40, 4, 6


def _saturate_cloudy_volume(state, weight, supersaturation: float = 1.02):
    """Raise qv to ``supersaturation`` times qsat wherever ``weight`` is
    appreciable, using WDM6's OWN saturation expression.

    The expression is the inline fpvs expansion at module_mp_wdm6.F:661-695
    with WRF's psat = 610.78, ep2 = rd/rv and the 0.99p cap, evaluated here
    in float64.  Using the scheme's own formula is the point: seeding from
    a different saturation curve would leave the column near, but not
    reliably above, the threshold the kernel tests.
    """
    import cupy as cp

    from gpuwm.core import constants as c
    from gpuwm.core.state import DTYPE

    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    p = cp.asarray(state.p, dtype=cp.float64)
    t = cp.asarray((thb + state.thp), dtype=cp.float64) * cp.power(
        p / float(c.P0), float(c.RCP))
    ttp = 273.16
    xa = -(1846.4 - 4190.0) / 461.6
    xb = xa + 2.5e6 / (461.6 * ttp)
    tr = ttp / t
    es = cp.minimum(610.78 * cp.exp(cp.log(tr) * xa) * cp.exp(xb * (1.0 - tr)),
                    0.99 * p)
    qsw = cp.maximum((287.0 / 461.6) * es / (p - es), 1.0e-15)
    target = supersaturation * qsw
    mask = cp.asarray(weight, dtype=cp.float64) > 0.05
    state.qv[...] = cp.asarray(
        cp.where(mask, cp.maximum(cp.asarray(state.qv, dtype=cp.float64),
                                  target),
                 cp.asarray(state.qv, dtype=cp.float64)), dtype=DTYPE)


def _wdm6_case(*, wdm6_hail_opt: int = 0, landmask=1.0):
    """A seeded mixed-phase WDM6 column set with a live physics driver."""
    import cupy as cp

    from gpuwm.config import validate_run_config
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.physics import initialize_physics
    from gpuwm.verify.cases import moist_bubble
    from gpuwm.verify.cases.wk82 import wk82_sounding

    cfg = replace(
        moist_bubble.default_config(),
        nx=_NX, ny=_NY, nz=_NZ, dt=1.5, run_seconds=15.0,
        time_step_sound=4, mp_physics=16, moist_cq=True,
        wdm6_hail_opt=wdm6_hail_opt,
        km_opt=1, khdif=0.0, kvdif=0.0,
        diff_6th_opt=0, damp_opt=0,
        case="wdm6_column_smoke",
    )
    cfg = validate_run_config(cfg)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: wk82_sounding(z)[0],
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = moist_bubble.build(cfg, coord, base)

    z = state.height_half()[:, None, None]
    x = ((np.arange(cfg.nx) + 0.5) * cfg.dx
         - 0.5 * cfg.nx * cfg.dx)[None, None, :]
    y = ((np.arange(cfg.ny) + 0.5) * cfg.dy
         - 0.5 * cfg.ny * cfg.dy)[None, :, None]
    cloud = np.exp(
        -((x / 5000.0) ** 2 + (y / 2500.0) ** 2
          + ((z - 6000.0) / 1800.0) ** 2)).astype(np.float32)
    # Bring the cloudy volume to saturation before seeding condensate.  Left
    # subsaturated, the WK82 background evaporates the whole seed on the
    # first call and the smoke exercises only WDM6's sink half -- the CCN
    # activation, autoconversion and accretion branches that make it a
    # double-moment scheme would never run.  A test that cannot reach the
    # scheme's characteristic physics is not evidence about that scheme.
    _saturate_cloudy_volume(state, cloud)
    state.qc += cp.asarray(np.float32(2.0e-4) * cloud)
    state.qi += cp.asarray(np.float32(5.0e-5) * cloud)
    state.qr += cp.asarray(np.float32(8.0e-5) * cloud)
    state.qs += cp.asarray(np.float32(6.0e-5) * cloud)
    state.qg += cp.asarray(np.float32(2.0e-5) * cloud)
    # Number moments consistent with the seeded masses.  nn is already the
    # ccn_conc fill from the state allocator; nc/nr start at WRF's zero, so
    # seeding them is what exercises the double-moment half rather than a
    # single-moment scheme with dead number arrays.
    state.nc += cp.asarray(np.float32(1.0e8) * cloud)
    state.nr += cp.asarray(np.float32(1.0e3) * cloud)
    driver = initialize_physics(state, cfg, landmask=landmask)
    return cfg, state, driver


def _water_substance(state):
    """Column-integrated total water per unit area, kg m-2, as float64."""
    import cupy as cp

    from gpuwm.core import constants as c

    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    del thb
    z8w = (phb + state.php) / np.float32(c.G)
    dz = cp.asarray(z8w[1:] - z8w[:-1], dtype=cp.float64)
    rho = cp.asarray(1.0 / state.alt, dtype=cp.float64)
    total = sum(cp.asarray(getattr(state, name), dtype=cp.float64)
                for name in ("qv", "qc", "qr", "qi", "qs", "qg"))
    return cp.asnumpy((total * rho * dz).sum(axis=0))


@requires_gpu
def test_wdm6_column_produces_a_physically_admissible_state():
    import cupy as cp

    from gpuwm.core.microphysics import apply as mp_apply

    cfg, state, _driver = _wdm6_case()
    before = {name: cp.asnumpy(getattr(state, name)).copy()
              for name in ("qv", "qc", "qr", "qi", "qs", "qg",
                           "nn", "nc", "nr", "thp")}
    water_before = _water_substance(state)

    diagnostics = mp_apply(state, cfg, float(cfg.dt))
    assert diagnostics is not None

    after = {name: cp.asnumpy(getattr(state, name))
             for name in ("qv", "qc", "qr", "qi", "qs", "qg",
                          "nn", "nc", "nr", "thp",
                          "effc", "effi", "effs")}
    for name, value in after.items():
        assert np.all(np.isfinite(value)), f"{name} went non-finite"
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr"):
        assert after[name].min() >= 0.0, f"{name} went negative"

    # WDM6's own padding clamp on the CCN reservoir, module_mp_wdm6.F:585.
    # It is applied on entry to every minor loop, so a post-call value above
    # the ceiling would mean the scheme wrote past its own bound.
    assert after["nn"].max() <= 2.0e10 + 1.0

    # The driver clamps the three radiation radii to the state background
    # bounds (module_mp_wdm6.F:443-445 in micron convention).
    assert 2.49 <= after["effc"].min() and after["effc"].max() <= 50.0
    assert 4.99 <= after["effi"].min() and after["effi"].max() <= 125.0
    assert 9.99 <= after["effs"].min() and after["effs"].max() <= 999.0

    # The scheme DID something to the warm-rain pair, which is the half that
    # distinguishes WDM6 from WSM6.  Without this the file would pass on a
    # port that ran WSM6's physics and left nc/nr frozen.
    assert not np.array_equal(after["nc"], before["nc"])
    assert not np.array_equal(after["nr"], before["nr"])
    assert not np.array_equal(after["thp"], before["thp"])

    # CCN ACTIVATION RAN, which is the branch that makes the reservoir a
    # prognostic variable rather than a passenger: supersaturated columns
    # move number OUT of nn and INTO nc (module_mp_wdm6.F:1951-1969).  The
    # evaporation return path (:1990-1994) moves it the other way, so both
    # signs are expected -- what would be wrong is neither.
    dnn = after["nn"] - before["nn"]
    dnc = after["nc"] - before["nc"]
    assert dnn.min() < -1.0e6, "no column activated CCN"
    assert dnc.max() > 1.0e6, "activated CCN did not become droplets"
    assert dnn.max() > 0.0, "no column returned number to the reservoir"

    # Water substance is conserved to the mass the scheme says it rained
    # out.  Column mass is O(30) kg m-2, so a 1e-4 kg m-2 residual is a
    # float32 accumulation floor, not a leak.
    water_after = _water_substance(state)
    fallen = cp.asnumpy(diagnostics.rainncv)  # mm over this call == kg m-2
    residual = np.abs(water_before - (water_after + fallen))
    assert residual.max() < 1.0e-3, (
        f"water substance is not conserved to the surface flux: "
        f"worst column residual {residual.max():.6g} kg m-2")


@requires_gpu
def test_the_column_smoke_fails_when_the_scheme_is_stubbed_out(monkeypatch):
    """MUTATION CONTROL: a no-op WDM6 must not pass the test above.

    The check that has teeth is the "did something" trio -- a stub leaves
    nc, nr and thp exactly as it found them.  Asserting the stub fails is
    what makes the passing run above evidence rather than decoration.
    """
    import cupy as cp

    from gpuwm.core.microphysics import apply as mp_apply
    import gpuwm.core.wdm6 as wdm6

    cfg, state, _driver = _wdm6_case()
    before_nc = cp.asnumpy(state.nc).copy()
    before_nr = cp.asnumpy(state.nr).copy()
    before_thp = cp.asnumpy(state.thp).copy()

    def stub(theta, qv, qc, qr, qi, qs, qg, nn, nc, nr, *args, **kwargs):
        return None

    monkeypatch.setattr(wdm6, "launch_wdm6", stub)
    mp_apply(state, cfg, float(cfg.dt))

    # Exactly the assertions the live test makes, now expected to FAIL.
    assert np.array_equal(cp.asnumpy(state.nc), before_nc)
    assert np.array_equal(cp.asnumpy(state.nr), before_nr)
    assert np.array_equal(cp.asnumpy(state.thp), before_thp)


@requires_gpu
def test_the_land_sea_mask_moves_the_autoconversion_threshold():
    """XLAND is a real input, not a passenger.

    module_mp_wdm6.F:607-614 selects qc0 (maritime, xncr0 = 5e7) where
    xland == 2 and qc1 (continental, xncr1 = 5e8) elsewhere -- a factor of
    ten in the autoconversion trigger.  Two runs identical but for the land
    mask must therefore differ, and if they ever stop differing the adapter
    has stopped forwarding the mask.
    """
    import cupy as cp

    from gpuwm.core.microphysics import apply as mp_apply

    results = []
    for landmask in (1.0, 0.0):            # WPS 1 = land, 0 = water
        cfg, state, _driver = _wdm6_case(landmask=landmask)
        mp_apply(state, cfg, float(cfg.dt))
        results.append((cp.asnumpy(state.qr).copy(),
                        cp.asnumpy(state.nr).copy()))
    (land_qr, land_nr), (water_qr, water_nr) = results
    assert not np.array_equal(land_qr, water_qr)
    assert not np.array_equal(land_nr, water_nr)


@requires_gpu
def test_wdm6_refuses_to_run_without_a_land_mask():
    """The refusal is the shipped behaviour, so it is tested as one."""
    from types import SimpleNamespace

    from gpuwm.core.wdm6 import column_land_mask

    for driver in (None, SimpleNamespace(fields={}),
                   SimpleNamespace(fields={"xland": None})):
        with pytest.raises(ValueError, match="XLAND"):
            column_land_mask(SimpleNamespace(physics=driver))


@requires_gpu
def test_wdm6_survives_ten_complete_model_steps():
    """The single-call test above cannot see a slow divergence.

    Ten complete RK3/acoustic/scalar-advection/microphysics steps with the
    scheme's number moments riding the transport is where a sign error in a
    number sink shows up: nc or nr runs away, or the state goes non-finite
    a few steps in.  This is an integration and gross-stability gate, not a
    trajectory claim.
    """
    import cupy as cp

    from gpuwm.core.dycore import run_steps, stability_report

    cfg, state, _driver = _wdm6_case()
    steps = 10
    run_steps(state, cfg, steps)
    report = stability_report(state, cfg)
    assert not report["nan"], report

    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr",
                 "thp", "effc", "effi", "effs"):
        value = cp.asnumpy(getattr(state, name))
        assert np.all(np.isfinite(value)), f"{name} went non-finite"
    # Advection can undershoot a mixing ratio slightly; the positive-definite
    # limiter and the scheme's own padding hold it near zero rather than at
    # exactly zero, which is why this is a floor and not a hard zero.
    for name in ("qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr"):
        value = cp.asnumpy(getattr(state, name))
        assert value.min() > -1.0e-10, f"{name} went materially negative"
    # The reservoir stays inside WDM6's clamp for the whole run, and the
    # droplet number stays inside a physical range rather than running away.
    nn = cp.asnumpy(state.nn)
    assert nn.max() <= 2.0e10 + 1.0
    assert cp.asnumpy(state.nc).max() < 1.0e12


@requires_gpu
def test_wdm6_reflectivity_runs_on_the_scheme_s_own_rain_number():
    """REFL_10CM through the shipped seam, and nr is genuinely read.

    refl10cm_wdm6 (module_mp_wdm6.F:2957-3131) diagnoses the rain intercept
    from nr, so unlike WSM6's fixed-N0r routine two columns differing only
    in rain NUMBER must produce different dBZ.  Reflectivity that ignored nr
    would be WSM6's routine wearing WDM6's name.
    """
    import cupy as cp

    from gpuwm.core.refl import launch_refl10cm_wdm6
    from gpuwm.core.state import DTYPE

    shape = (8, 1, 2)
    zeros = cp.zeros(shape, dtype=DTYPE)
    qv = cp.full(shape, DTYPE(4.0e-3))
    qr = cp.full(shape, DTYPE(3.0e-4))
    qs = cp.full(shape, DTYPE(1.0e-4))
    qg = cp.full(shape, DTYPE(5.0e-5))
    t = cp.asarray(np.broadcast_to(
        (280.0 - 2.0 * np.arange(8))[:, None, None], shape
    ).copy(), dtype=DTYPE)
    p = cp.asarray(np.broadcast_to(
        (96000.0 - 7000.0 * np.arange(8))[:, None, None], shape
    ).copy(), dtype=DTYPE)
    nr = cp.asarray(np.broadcast_to(
        np.array([[1.0e3, 1.0e4]], dtype=np.float32)[None, :, :], shape
    ).copy(), dtype=DTYPE)
    refl = zeros.copy()
    launch_refl10cm_wdm6(qv, qr, nr, qs, qg, t, p, refl)
    out = cp.asnumpy(refl)
    assert np.all(np.isfinite(out))
    assert out.min() >= -35.0
    assert not np.allclose(out[:, 0, 0], out[:, 0, 1]), (
        "reflectivity did not respond to rain number; the prognostic "
        "intercept is not reaching the Rayleigh sum")
