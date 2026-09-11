"""Milbrandt-Yau (mp_physics=9) radiates its OWN radii under RTE+RRTMGP.

THE BREAKAGE THESE GATES PREVENT
--------------------------------
mp=9 was refused against the 4/4 pair on every ``ra_rrtmg_variant`` but
``rrtmg_legacy`` because ``gpuwm.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME`` had
no row for it, and the variant defaults to RTE+RRTMGP -- so every bare mp=9
configuration was refused at the door.  The recorded reason said Morrison's
row "would derive the radii from a gamma distribution that is not this
scheme's", which argued for a row of the scheme's own rather than for a
refusal.  That row is ``9: "milbrandt2"`` and these tests pin it.

THE SCHEME'S OWN RADII
----------------------
WRF ships them, commented out, at the end of mp_milbrandt2mom_main
(phys/module_mp_milbrandt2mom.F:3351-3378): ``r_eff = M_D(3)/(2 M_D(2))``
over MY2005a eqn (2), hard-coded for the scheme's shape parameters ::

    !cloud:  hardcoded for alpha_c = 1. and mu_c = 3.
       iLAMc = iLAMDA_x(DE(i,k),QC(i,k),iNC,icexc9,thrd)
    !   reff_c(i,k) = 0.664639*iLAMc                            (:3362)
    !ice:    hardcoded for alpha_i = 0. and mu_i = 1.
       iLAMi = max( iLAMmin2, iLAMDA_x(DE(i,k),QI(i,k),iNY,icexi9,thrd) )
    !   reff_i(i,k) = 1.5*iLAMi                                 (:3372)

0.664639 is Gamma(3)/(2 Gamma(8/3)) and 1.5 is Gamma(4)/(2 Gamma(3)); both
are checked below against the gamma function rather than trusted.  Snow is
not in WRF's block; the adapter extends the same moment ratio to the
scheme's alpha_s = 0 exponential snow over the Brandes m(D) pair the pinned
snowSpherical=.false. selects (cms = 0.1597, dms = 2.078, :1290), through
the scheme's own ``iGS20`` and ``idms`` constants.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires_gpu

# The scheme's constant vector, the same float32 values the CUDA kernel
# integrates with (gpuwm/core/kernels/milbrandt2.cu is generated against it).
from gpuwm.core.milbrandt2_constants import CONSTANTS as MY2

RE_C_PER_ILAM = 0.664639
RE_I_PER_ILAM = 1.5
RE_S_PER_ILAM = 1.5
ILAM_MIN = 1.0e-10


# ---------------------------------------------------------------------------
# A NumPy reference of the radii, written from the Fortran, not from the
# adapter.  Per-kg numbers: DE cancels between (DE*Q*icex) and N_m3 = N_kg*DE.
# ---------------------------------------------------------------------------

def _ilam(q, n, icex, inverse_dm):
    return np.power(q * float(icex) / n, inverse_dm)


def _np_my2_radii_um(qc, nc, qi, ni, qs, ns):
    re_c = RE_C_PER_ILAM * _ilam(qc, nc, MY2["icexc9"], 1.0 / 3.0) * 1.0e6
    re_i = RE_I_PER_ILAM * np.maximum(
        ILAM_MIN, _ilam(qi, ni, MY2["icexi9"], 1.0 / 3.0)) * 1.0e6
    re_s = RE_S_PER_ILAM * np.maximum(
        ILAM_MIN, _ilam(qs, ns, MY2["iGS20"], float(MY2["idms"]))) * 1.0e6
    return re_c, re_i, re_s


def _paths(plev, qc, qi, qs, **kwargs):
    import cupy as cp

    from gpuwm.core.rrtmgp import hydrometeor_paths

    def dev(x):
        return cp.asarray(np.asarray(x, dtype=np.float32))

    return hydrometeor_paths(
        dev(plev), dev(qc), None, dev(qi), dev(qs), microphysics="milbrandt2",
        **{k: (dev(v) if isinstance(v, (list, tuple, np.ndarray, float))
               else v) for k, v in kwargs.items()})


# ---------------------------------------------------------------------------
# 1.  The constants ARE the scheme's, not round numbers.
# ---------------------------------------------------------------------------

def test_the_radius_constants_are_the_moment_ratios_of_the_schemes_psd():
    """0.664639 and 1.5 are Gamma ratios of MY2005a eqn (2), not tuning."""
    from gpuwm.core import rrtmgp

    # r_eff = M3/(2 M2) for N(D) = N0 D^((alpha+1) mu - 1) exp(-(lambda D)^mu)
    # is Gamma(alpha + 1 + 3/mu) / (2 Gamma(alpha + 1 + 2/mu)) / lambda.
    def ratio(alpha, mu):
        return math.gamma(alpha + 1.0 + 3.0 / mu) / (
            2.0 * math.gamma(alpha + 1.0 + 2.0 / mu))

    assert ratio(1.0, 3.0) == pytest.approx(RE_C_PER_ILAM, abs=1.0e-6)
    assert ratio(0.0, 1.0) == pytest.approx(RE_I_PER_ILAM, abs=0.0)
    assert rrtmgp.MILBRANDT2_CLOUD_RADIUS_PER_ILAMBDA == RE_C_PER_ILAM
    assert rrtmgp.MILBRANDT2_ICE_RADIUS_PER_ILAMBDA == RE_I_PER_ILAM
    assert rrtmgp.MILBRANDT2_SNOW_RADIUS_PER_ILAMBDA == RE_S_PER_ILAM
    assert rrtmgp.MILBRANDT2_ILAMBDA_MIN_M == ILAM_MIN

    # And the icex constants the radii divide by are the scheme's own
    # first-call chain (gpuwm/core/milbrandt2_constants.py, hoisted from
    # module_mp_milbrandt2mom.F:1257-1438):
    #   icexc9 = 1/(GC2/GC1 * cm_r),  icexi9 = 1/(cm_i Gamma(1+dmi)/GI31),
    #   iGS20  = 1/(GS40/GS31 * cms)  with GS40 = Gamma(1 + alpha_s + dms).
    f = np.float32
    assert MY2["icexc9"] == pytest.approx(
        1.0 / (float(MY2["GC2"]) / float(MY2["GC1"])
               * float(MY2["PIov6"]) * 1000.0), rel=1.0e-6)
    assert MY2["icexi9"] == pytest.approx(
        1.0 / (440.0 * float(MY2["GI40"]) * float(MY2["iGI31"])), rel=1.0e-6)
    assert MY2["iGS20"] == pytest.approx(
        1.0 / (float(MY2["GS40"]) * float(MY2["iGS31"]) * float(MY2["cms"])),
        rel=1.0e-6)
    assert MY2["cms"] == f(0.1597) and MY2["dms"] == f(2.078)
    assert MY2["idms"] == pytest.approx(1.0 / 2.078, rel=1.0e-6)


def test_the_row_is_the_schemes_own_and_ice_active_with_snow():
    from gpuwm.core.rrtmgp import (
        EFFECTIVE_RADIUS_PLAUSIBLE_UM, _MP_CLOUD_OPTICS_SCHEME,
        _NO_CLOUD_OPTICS_COUPLING, cloud_optics_scheme,
        effective_radius_bands, scheme_has_snow_species,
        scheme_is_ice_active)

    assert _MP_CLOUD_OPTICS_SCHEME[9] == "milbrandt2"
    assert cloud_optics_scheme(9) == "milbrandt2"
    assert 9 not in _NO_CLOUD_OPTICS_COUPLING
    # Registry.EM_COMMON:3025: moist:qv,qc,qr,qi,qs,qg,qh -> F_QI and F_QS.
    assert scheme_is_ice_active("milbrandt2")
    assert scheme_has_snow_species("milbrandt2")
    # The band row exists and is the generic one by value; the radii are
    # derived in-adapter and clipped there, so nothing on the production
    # path is gated by it.
    assert effective_radius_bands("milbrandt2") == EFFECTIVE_RADIUS_PLAUSIBLE_UM
    assert effective_radius_bands("milbrandt2") is not EFFECTIVE_RADIUS_PLAUSIBLE_UM


def test_a_bare_mp9_configuration_validates_on_the_default_variant():
    """Fixed means default: no flag, no variant switch, no acknowledgement."""
    from gpuwm.config import RunConfig, validate_run_config

    cfg = validate_run_config(RunConfig(
        nx=4, ny=3, nz=12, dx=2000.0, dy=2000.0, ztop=8000.0, dt=10.0,
        run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=9, ra_lw_physics=4, ra_sw_physics=4))
    assert cfg.ra_rrtmg_variant == "rte-rrtmgp"
    # The legacy arm still validates too; it computes its own radii as WRF
    # does under has_reqc=0 and is unchanged by this row.
    validate_run_config(RunConfig(
        nx=4, ny=3, nz=12, dx=2000.0, dy=2000.0, ztop=8000.0, dt=10.0,
        run_seconds=0.0, time_step_sound=4, moist=True, mp_physics=9,
        ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant="rrtmg_legacy"))


# ---------------------------------------------------------------------------
# 2.  The radii on device agree with the Fortran-written reference.
# ---------------------------------------------------------------------------

_PLEV = [[100000.0, 90000.0]]


@pytest.mark.gpu
@requires_gpu
def test_the_device_radii_match_the_reference_written_from_the_fortran():
    import cupy as cp

    qc, nc = 1.0e-3, 8.0e7          # 15 um class droplets
    qi, ni = 4.0e-4, 5.0e6          # tens-of-microns ice
    qs, ns = 1.0e-4, 1.0e6          # small snow, inside the clip window
    re_c, re_i, re_s = _np_my2_radii_um(qc, nc, qi, ni, qs, ns)
    assert 2.5 < re_c < 21.5 and 5.0 < re_i < 90.0 and 5.0 < re_s < 90.0

    got = _paths(_PLEV, [[qc]], [[qi]], [[qs]],
                 nc=[[nc]], ni=[[ni]], ns=[[ns]])
    assert float(got.reliq[0, 0]) == pytest.approx(re_c, rel=3.0e-6)
    expected_dgice = 2.0 * (ni * re_i + ns * re_s) / (ni + ns)
    assert float(got.dgice[0, 0]) == pytest.approx(expected_dgice, rel=3.0e-6)
    # Paths are WRF's: cloud water only for liquid, ice plus snow for ice.
    mass_path = 10000.0 * 1000.0 / 9.80665
    assert float(got.clwp[0, 0]) == pytest.approx(qc * mass_path, rel=3.0e-6)
    assert float(got.ciwp[0, 0]) == pytest.approx(
        (qi + qs) * mass_path, rel=3.0e-6)
    assert got.reliq.dtype == cp.float32 and got.dgice.dtype == cp.float32


@pytest.mark.gpu
@requires_gpu
def test_limits_clear_sky_fallbacks_and_the_table_clips():
    # Nothing active: the adapter's clear-sky placeholders, as Morrison's.
    empty = _paths(_PLEV, [[0.0]], [[0.0]], [[0.0]],
                   nc=[[0.0]], ni=[[0.0]], ns=[[0.0]])
    assert float(empty.reliq[0, 0]) == 10.0
    assert float(empty.dgice[0, 0]) == 50.0
    # A species below the scheme's own epsQ/epsN thresholds is inactive.
    below = _paths(_PLEV, [[1.0e-15]], [[1.0e-15]], [[0.0]],
                   nc=[[1.0e-4]], ni=[[1.0e-4]], ns=[[0.0]])
    assert float(below.reliq[0, 0]) == 10.0
    assert float(below.dgice[0, 0]) == 50.0
    # Huge numbers -> tiny radii -> the RRTMGP table floor; tiny numbers ->
    # huge radii -> the table ceiling.  Neither is a NaN or a raise.
    small = _paths(_PLEV, [[1.0e-3]], [[1.0e-4]], [[0.0]],
                   nc=[[1.0e12]], ni=[[1.0e12]], ns=[[0.0]])
    assert float(small.reliq[0, 0]) == 2.5
    assert float(small.dgice[0, 0]) == 10.0
    large = _paths(_PLEV, [[1.0e-3]], [[0.0]], [[1.0e-3]],
                   nc=[[1.0e3]], ni=[[0.0]], ns=[[1.0e0]])
    assert float(large.reliq[0, 0]) == 21.5
    assert float(large.dgice[0, 0]) == 180.0
    # Snow alone drives the ice radius; a number-weighted merge with ice
    # lands strictly between the two species radii.
    _, re_i, re_s = _np_my2_radii_um(0.0, 1.0, 4.0e-4, 5.0e6, 1.0e-4, 1.0e6)
    both = _paths(_PLEV, [[0.0]], [[4.0e-4]], [[1.0e-4]],
                  nc=[[0.0]], ni=[[5.0e6]], ns=[[1.0e6]])
    lo, hi = sorted((2.0 * re_i, 2.0 * re_s))
    assert lo < float(both.dgice[0, 0]) < hi


@pytest.mark.gpu
@requires_gpu
def test_the_radii_are_monotone_in_mass_and_in_number():
    """More mass at fixed number: bigger particles.  More number at fixed
    mass: smaller ones.  Checked inside the clip window for all three."""
    # Four one-layer columns: (ncol, nlay) = (4, 1), plev (4, 2).
    def columns(*values):
        return np.asarray(values, dtype=np.float64).reshape(4, 1)

    plev = [[100000.0, 90000.0]] * 4
    zeros = columns(0.0, 0.0, 0.0, 0.0)

    qc = columns(2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3)
    liquid = _paths(plev, qc, zeros, zeros,
                    nc=np.full_like(qc, 8.0e7), ni=zeros, ns=zeros)
    reliq = np.ravel(liquid.reliq.get())
    assert np.all(np.diff(reliq) > 0) and 2.5 < reliq[0] and reliq[-1] < 21.5

    nc_up = columns(5.0e7, 1.0e8, 2.0e8, 4.0e8)
    liquid_n = _paths(plev, np.full_like(nc_up, 1.0e-3), zeros, zeros,
                      nc=nc_up, ni=zeros, ns=zeros)
    reliq_n = np.ravel(liquid_n.reliq.get())
    assert np.all(np.diff(reliq_n) < 0)
    assert 2.5 < reliq_n[-1] and reliq_n[0] < 21.5

    qi = columns(1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4)
    ice = _paths(plev, zeros, qi, zeros,
                 nc=zeros, ni=np.full_like(qi, 5.0e6), ns=zeros)
    dgice = np.ravel(ice.dgice.get())
    assert np.all(np.diff(dgice) > 0) and 10.0 < dgice[0] and dgice[-1] < 180.0

    ns_up = columns(5.0e5, 1.0e6, 2.0e6, 4.0e6)
    snow = _paths(plev, zeros, zeros, np.full_like(ns_up, 1.0e-4),
                  nc=zeros, ni=zeros, ns=ns_up)
    dgsnow = np.ravel(snow.dgice.get())
    assert np.all(np.diff(dgsnow) < 0)
    assert 10.0 < dgsnow[-1] and dgsnow[0] < 180.0


# ---------------------------------------------------------------------------
# 3.  Agreement with Morrison's branch, as far as two PSDs can agree.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_agreement_with_the_morrison_branch_on_morrison_shaped_inputs():
    """Same inputs into both branches.

    What MUST agree: the mass paths (WRF's, independent of the scheme), the
    clear-sky placeholders and the clip domain -- the machinery the two
    branches share.  What agrees UP TO THE SCHEMES' OWN CONSTANTS: the ice
    radius, because both are ``1.5/lambda`` over an exponential PSD and
    differ only in the mass-diameter constant -- Morrison's
    ``pi rho_i = pi * 500`` against Milbrandt-Yau's ``Gamma(4) cm_i =
    6 * 440`` -- so the ratio is ``(pi 500 / 2640)^(1/3)`` exactly.  What
    does NOT agree and is not asked to: the droplet radius (Morrison's
    pgam(nc) gamma against MY2's alpha=1, mu=3 form) and the snow radius
    (Morrison's rho_s = 100 spheres against Brandes m(D)).
    """
    import cupy as cp

    from gpuwm.core.rrtmgp import hydrometeor_paths

    def dev(x):
        return cp.asarray(np.asarray(x, dtype=np.float32))

    plev = dev([[100000.0, 90000.0]] * 3)
    qc = dev([[1.0e-3], [0.0], [0.0]])
    qi = dev([[0.0], [4.0e-4], [0.0]])
    qs = dev([[0.0], [0.0], [0.0]])
    numbers = dict(nc=dev([[8.0e7], [0.0], [0.0]]),
                   nr=dev([[0.0], [0.0], [0.0]]),
                   ni=dev([[0.0], [5.0e6], [0.0]]),
                   ns=dev([[0.0], [0.0], [0.0]]))
    thermo = dict(play=dev([[95000.0]] * 3), tlay=dev([[270.0]] * 3))
    cldfra = dev([[0.6], [0.6], [0.0]])

    my2 = hydrometeor_paths(plev, qc, None, qi, qs, microphysics="milbrandt2",
                            cldfra=cldfra, **numbers, **thermo)
    morr = hydrometeor_paths(plev, qc, None, qi, qs, microphysics="morrison",
                             cldfra=cldfra, **numbers, **thermo)

    # Shared machinery: bit-identical paths, identical clear-sky column.
    np.testing.assert_array_equal(my2.clwp.get(), morr.clwp.get())
    np.testing.assert_array_equal(my2.ciwp.get(), morr.ciwp.get())
    assert float(my2.reliq[2, 0]) == float(morr.reliq[2, 0]) == 10.0
    assert float(my2.dgice[2, 0]) == float(morr.dgice[2, 0]) == 50.0
    assert float(my2.dgice[0, 0]) == float(morr.dgice[0, 0]) == 50.0
    assert float(my2.reliq[1, 0]) == float(morr.reliq[1, 0]) == 10.0

    # Ice: the same exponential radius up to the two schemes' m(D) constants.
    ratio = (math.pi * 500.0 / (6.0 * 440.0)) ** (1.0 / 3.0)
    assert 10.0 < float(morr.dgice[1, 0]) < 180.0
    assert 10.0 < float(my2.dgice[1, 0]) < 180.0
    assert float(my2.dgice[1, 0]) == pytest.approx(
        float(morr.dgice[1, 0]) * ratio, rel=1.0e-5)

    # Droplets: both inside the clip window, and DIFFERENT -- the PSDs are
    # different, which is the whole reason mp=9 has its own row.
    assert 2.5 < float(my2.reliq[0, 0]) < 21.5
    assert 2.5 < float(morr.reliq[0, 0]) < 21.5
    assert float(my2.reliq[0, 0]) != float(morr.reliq[0, 0])


# ---------------------------------------------------------------------------
# 4.  Refusals that name the breakage.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_the_branch_refuses_background_radii_and_missing_numbers():
    with pytest.raises(ValueError, match="commented out"):
        _paths(_PLEV, [[1.0e-3]], [[0.0]], [[0.0]],
               nc=[[8.0e7]], ni=[[0.0]], ns=[[0.0]], effc=[[2.5]])
    with pytest.raises(ValueError, match="require nc, ni and ns"):
        _paths(_PLEV, [[1.0e-3]], [[0.0]], [[0.0]], nc=[[8.0e7]], ni=[[0.0]])
    with pytest.raises(ValueError):
        _paths(_PLEV, [[1.0e-3]], [[0.0]], [[0.0]],
               nc=[[-1.0]], ni=[[0.0]], ns=[[0.0]])


# ---------------------------------------------------------------------------
# 5.  End to end through the shipped adapter, on the state mp=9 allocates.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_an_mp9_radiation_call_radiates_the_transported_numbers():
    """A bare mp_physics=9 call runs on the default adapter, and the
    NUMBER moments move the answer -- which is what "the scheme's own
    radii" means operationally.  ``state.effc/effi/effs`` are present at
    their allocation-time background, as gpuwm/core/state.py leaves them
    for mp=9, and are NOT what radiation reads: changing them changes
    nothing."""
    from datetime import datetime
    from types import SimpleNamespace

    import cupy as cp

    from gpuwm.core.rrtmgp import RRTMGPRadiation

    nz, ny, nx = 20, 1, 1
    plev_col = np.geomspace(100000.0, 1.1, nz + 1)
    play_col = np.sqrt(plev_col[:-1] * plev_col[1:])
    t_col = np.linspace(290.0, 215.0, nz)
    exner_col = (play_col / 100000.0) ** (287.0 / 1004.0)
    shape = (nz, ny, nx)

    def expand(x):
        return cp.asarray(
            np.broadcast_to(x[:, None, None], shape).copy(), dtype=cp.float32)

    qc = cp.zeros(shape, cp.float32)
    qi = cp.zeros(shape, cp.float32)
    qs = cp.zeros(shape, cp.float32)
    qc[4:8] = 4.0e-4
    qi[10:14] = 4.0e-4
    qs[9:12] = 1.0e-4
    atmosphere = {
        "pressure": expand(play_col),
        "p_interface": cp.asarray(np.broadcast_to(
            plev_col[:, None, None], (nz + 1, ny, nx)).copy(),
            dtype=cp.float32),
        "temperature": expand(t_col),
        "theta": expand(t_col / exner_col),
        "exner": expand(exner_col),
        "qv": expand(np.geomspace(8.0e-3, 1.0e-6, nz)),
        "qc": qc,
        "qi": qi,
    }
    fields = {
        "tsk": cp.full((ny, nx), 288.0, cp.float32),
        "albedo": cp.full((ny, nx), 0.18, cp.float32),
        "emiss": cp.full((ny, nx), 0.96, cp.float32),
    }

    def number(value, where):
        out = cp.zeros(shape, cp.float32)
        out[where] = np.float32(value)
        return out

    def call(nc_per_kg, ni_per_kg, background_um=(2.5, 5.0, 10.0)):
        effc, effi, effs = background_um
        state = SimpleNamespace(
            elapsed_seconds=0.0, qc=qc, qi=qi, qs=qs,
            qr=cp.zeros(shape, cp.float32),
            nc=number(nc_per_kg, slice(4, 8)),
            nr=cp.zeros(shape, cp.float32),
            ni=number(ni_per_kg, slice(10, 14)),
            ns=number(1.0e6, slice(9, 12)),
            effc=cp.full(shape, np.float32(effc), cp.float32),
            effi=cp.full(shape, np.float32(effi), cp.float32),
            effs=cp.full(shape, np.float32(effs), cp.float32),
            physics=SimpleNamespace(microphysics_updates=3))
        radiation = RRTMGPRadiation(
            datetime(1974, 4, 3, 18), cp.asarray([[40.0]]),
            cp.asarray([[-100.0]]), trace_gas_overrides={"co2": 330.0e-6})
        result = radiation(
            atmosphere=atmosphere, fields=fields, state=state,
            cfg=SimpleNamespace(mp_physics=9, dt=60.0, radt=12.0,
                                radt_minutes=12.0))
        return np.concatenate([
            np.ravel(cp.asnumpy(getattr(result, name)))
            for name in ("rthratenlw", "rthratensw", "swdown", "glw")])

    base = call(8.0e7, 5.0e6)
    assert np.all(np.isfinite(base))
    # Droplet number moves the answer (the cloud radius is 0.664639/lambda_c).
    assert not np.array_equal(base, call(3.0e8, 5.0e6))
    # Ice number moves the answer (1.5/lambda_i, merged by number with snow).
    assert not np.array_equal(base, call(8.0e7, 5.0e7))
    # The allocation-time background radii are not read: bit-identical.
    np.testing.assert_array_equal(
        base, call(8.0e7, 5.0e6, background_um=(19.0, 60.0, 400.0)))
