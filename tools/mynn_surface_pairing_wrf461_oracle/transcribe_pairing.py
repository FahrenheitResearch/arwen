"""Independent WRF v4.6.1 MYNN/RUC/Noah-MP ownership transcription.

This module intentionally imports no ``gpuwm`` code.  It transcribes only
the pinned surface-driver assignments in NumPy float32; neither LSM column
solver is duplicated here.
"""

from __future__ import annotations

import numpy as np


F = np.float32


def _array(value):
    return np.asarray(value, dtype=np.float32)


MYNN_FRACTIONAL_IMMEDIATE = (
    "br", "gz1oz0", "mol", "psih", "psim", "rmol", "ust", "wspd",
    "zol", "ch", "cd", "cda", "ck", "cka", "q2", "t2", "th2",
    "u10", "ustm", "v10",
)
MYNN_FRACTIONAL_WAIT_FOR_LSM = (
    "chs2", "chs", "cpm", "cqs2", "flhc", "flqc",
    "hfx", "lh", "qfx", "qgh", "qsfc", "znt",
)


def get_local_ice_tsk(
        *, xice, sst, tsk, itimestep, xice_threshold=F(0.5),
        tice2tsk_if2cold=False):
    """``module_surface_driver.F:7117-7208`` in float32."""
    xice = _array(xice)
    sst = _array(sst).copy()
    tsk = _array(tsk)
    active = (xice >= F(xice_threshold)) & (xice <= F(1.0))
    sst = np.where(active & (sst < F(271.4)), F(271.4), sst).astype(
        np.float32)
    if int(itimestep) <= 3:
        warm = active & (sst > F(273.0))
        sst = np.where(
            warm & (xice >= F(0.6)),
            F(271.4),
            np.where(
                warm & (xice >= F(0.4)),
                F(273.0),
                np.where(
                    warm & (xice >= F(0.2)) & (sst > F(275.0)),
                    F(275.0),
                    np.where(
                        warm & (sst > F(278.0)), F(278.0), sst))),
        ).astype(np.float32)
    tsk_sea = np.where(active, sst, tsk).astype(np.float32)
    if tice2tsk_if2cold:
        tsk_ice = np.minimum(tsk, F(273.15)).astype(np.float32)
    else:
        denominator = np.where(active, xice, F(1.0)).astype(np.float32)
        tsk_ice = np.maximum(
            (tsk - (F(1.0) - xice) * sst) / denominator,
            F(221.4),
        ).astype(np.float32)
    tsk_ice = np.where(active, tsk_ice, tsk).astype(np.float32)
    tsk_ice = np.where(
        active & (xice < F(0.2)) & (tsk < F(253.15)),
        F(253.15), tsk_ice).astype(np.float32)
    tsk_ice = np.where(
        active & (xice < F(0.1)) & (tsk < F(263.15)),
        F(263.15), tsk_ice).astype(np.float32)
    return {"sst": sst, "tsk_sea": tsk_sea, "tsk_ice": tsk_ice}


def mynn_fractional_seaice_staging(*, ice, sea, xice, active):
    """``MYNN_SEAICE_WRAPPER`` at surface-driver :5508-5554.

    The immediate set is blended before the LSM.  The wait set remains as
    separate full-ice and open-water components for the LSM-specific
    post-call blend.
    """
    xice = _array(xice)
    active = np.asarray(active, dtype=bool)
    result = {}
    for name in MYNN_FRACTIONAL_IMMEDIATE:
        ice_value = _array(ice[name])
        sea_value = _array(sea[name])
        blended = (
            ice_value * xice + (F(1.0) - xice) * sea_value
        ).astype(np.float32)
        result[name] = np.where(active, blended, ice_value).astype(np.float32)
    for name in MYNN_FRACTIONAL_WAIT_FOR_LSM:
        result[name] = _array(ice[name]).copy()
        result[f"{name}_sea"] = _array(sea[name]).copy()
    return result


def ruc_seam_ownership(*, mynn, lsm, rho, mavail):
    """Driver ownership at ``module_surface_driver.F:3500-3585``.

    ``LSMRUC`` receives ``CHS/FLHC/FLQC`` as ``INTENT(IN)`` and does not
    receive ``UST/CHS2/CQS2``. It returns ``TSK/HFX/QFX/LH`` as
    ``INTENT(INOUT)``. The driver then unconditionally rebuilds CHS and its
    local CQS from the retained MYNN coefficients and post-LSM MAVAIL.
    """
    rho = _array(rho)
    mavail = _array(mavail)
    result = {
        name: _array(mynn[name]).copy()
        for name in ("ust", "flhc", "flqc", "chs2", "cqs2", "cpm")
    }
    for name in ("tsk", "hfx", "qfx", "lh", "qsfc", "znt"):
        result[name] = _array(lsm[name]).copy()
    result["cqs"] = (result["flqc"] / (mavail * rho)).astype(np.float32)
    result["chs"] = (result["flhc"] / (result["cpm"] * rho)).astype(
        np.float32)
    return result


def noahmp_seam_ownership(*, mynn, lsm):
    """Driver ownership at surface-driver :3127-3181 / noahmpdrv :1206-1285."""
    result = {
        name: _array(mynn[name]).copy()
        for name in ("ust", "chs", "chs2", "cqs2", "flhc", "flqc")
    }
    for name in ("tsk", "hfx", "qfx", "lh", "qsfc", "znt"):
        result[name] = _array(lsm[name]).copy()
    return result


def noahmp_flux_writeback(*, trad, fsh, ecan, edir, etran, fcev, fgev,
                          fctr, qsfc, z0wrf):
    """``module_sf_noahmpdrv.F:1206-1207,1223-1225,1245,1280-1281``."""
    return {
        "tsk": _array(trad).copy(),
        "hfx": _array(fsh).copy(),
        "qfx": ((_array(ecan) + _array(edir)) + _array(etran)).astype(
            np.float32),
        "lh": ((_array(fcev) + _array(fgev)) + _array(fctr)).astype(
            np.float32),
        "qsfc": _array(qsfc).copy(),
        "znt": _array(z0wrf).copy(),
    }


def noahmp_post_lsm_diagnostics(
        *, ivgtyp, xice, iswater, isice, isurban, lcz, psfc, tsk, hfx,
        qfx, qsfc, chs2, cqs2, fveg, t2mv, t2mb, q2mv, q2mb):
    """``module_surface_driver.F:3333-3370`` with urban physics off."""
    ivgtyp = np.asarray(ivgtyp, dtype=np.int32)
    xice = _array(xice)
    psfc = _array(psfc)
    tsk = _array(tsk)
    hfx = _array(hfx)
    qfx = _array(qfx)
    qsfc = _array(qsfc)
    chs2 = _array(chs2)
    cqs2 = _array(cqs2)
    fveg = _array(fveg)
    t2mv = _array(t2mv)
    t2mb = _array(t2mb)
    q2mv = _array(q2mv)
    q2mb = _array(q2mb)

    water_or_full_ice = (
        (ivgtyp == int(iswater))
        | ((ivgtyp == int(isice)) & (xice >= F(0.5)))
    )
    urban_or_partial_ice = (
        np.isin(ivgtyp, np.asarray((isurban, *lcz), dtype=np.int32))
        | ((ivgtyp == int(isice)) & (xice < F(0.5)))
    )

    rho = (psfc / (F(287.0) * tsk)).astype(np.float32)
    q_guard = cqs2 < F(1.0e-5)
    t_guard = chs2 < F(1.0e-5)
    safe_cqs2 = np.where(q_guard, F(1.0), cqs2).astype(np.float32)
    safe_chs2 = np.where(t_guard, F(1.0), chs2).astype(np.float32)
    q_flux = np.where(
        q_guard, qsfc, qsfc - qfx / (rho * safe_cqs2)
    ).astype(np.float32)
    t_flux = np.where(
        t_guard, tsk,
        tsk - hfx / (rho * F(1004.5) * safe_chs2)
    ).astype(np.float32)
    t_land = (fveg * t2mv + (F(1.0) - fveg) * t2mb).astype(np.float32)
    q_land = (fveg * q2mv + (F(1.0) - fveg) * q2mb).astype(np.float32)
    t2 = np.where(
        water_or_full_ice,
        t_flux,
        np.where(urban_or_partial_ice, t2mb, t_land),
    ).astype(np.float32)
    q2 = np.where(
        water_or_full_ice,
        q_flux,
        np.where(urban_or_partial_ice, q2mb, q_land),
    ).astype(np.float32)
    th2 = (
        t2 * np.power(F(1.0e5) / psfc, F(287.0 / 1004.5))
    ).astype(np.float32)
    return {"t2": t2, "th2": th2, "q2": q2}


_RSLF_C = (
    F(.611583699e03), F(.444606896e02), F(.143177157e01),
    F(.264224321e-1), F(.299291081e-3), F(.203154182e-5),
    F(.702620698e-8), F(.379534310e-11), F(-.321582393e-13),
)
_RSIF_C = (
    F(.609868993e03), F(.499320233e02), F(.184672631e01),
    F(.402737184e-1), F(.565392987e-3), F(.521693933e-5),
    F(.307839583e-7), F(.105785160e-9), F(.161444444e-12),
)


def _horner(coefficients, x):
    result = np.zeros_like(x, dtype=np.float32) + coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = (result * x + coefficient).astype(np.float32)
    return result


def _saturation_mixing_ratio(pressure, temperature):
    pressure = _array(pressure)
    temperature = _array(temperature)
    over_ice = (temperature - F(273.15)) <= F(0.0)
    xl = np.maximum(F(-80.0), temperature - F(273.15)).astype(np.float32)
    xi = np.maximum(F(-80.0), temperature - F(273.16)).astype(np.float32)
    esl = _horner(_RSLF_C, xl)
    esi = _horner(_RSIF_C, xi)
    rslf = (F(.622) * esl / (pressure - esl)).astype(np.float32)
    rsif = (F(.622) * esi / (pressure - esi)).astype(np.float32)
    return np.where(over_ice, rsif, rslf).astype(np.float32)


def ruc_post_lsm_diagnostics(
        *, psfc, tsk, hfx, qfx, qsfc, chs2, cqs2, cqs,
        t1, qv1, rho1, p1):
    """``module_sf_sfcdiags_ruclsm.F:47-146`` (hardwired flux arm)."""
    psfc = _array(psfc)
    tsk = _array(tsk)
    hfx = _array(hfx)
    qfx = _array(qfx)
    qsfc = _array(qsfc)
    chs2 = _array(chs2)
    cqs2 = _array(cqs2)
    cqs = _array(cqs)
    t1 = _array(t1)
    qv1 = _array(qv1)
    rho1 = _array(rho1)
    p1 = _array(p1)
    rovcp = F(287.0 / 1004.5)
    scale = np.power(F(1.0e5) / psfc, rovcp).astype(np.float32)
    inverse = np.power(F(1.0e-5) * psfc, rovcp).astype(np.float32)
    th2 = np.where(
        chs2 < F(1.0e-5),
        (t1 * scale).astype(np.float32),
        (tsk * scale - hfx / (rho1 * F(1004.5) * chs2)).astype(np.float32),
    ).astype(np.float32)
    t2 = (th2 * inverse).astype(np.float32)
    t2 = np.minimum(
        np.maximum(tsk, t1), np.maximum(np.minimum(tsk, t1), t2)
    ).astype(np.float32)
    th2 = (t2 * scale).astype(np.float32)

    qlev1 = np.minimum(_saturation_mixing_ratio(p1, t1), qv1).astype(
        np.float32)
    qsfcprox = (qlev1 + qfx / (rho1 * cqs)).astype(np.float32)
    qsfcmr = (qsfc / (F(1.0) - qsfc)).astype(np.float32)
    q2 = np.where(
        cqs2 < F(1.0e-5),
        qlev1,
        (qsfcprox - qfx / (rho1 * cqs2)).astype(np.float32),
    ).astype(np.float32)
    q2 = np.minimum(
        np.maximum(qsfcmr, qlev1),
        np.maximum(np.minimum(qsfcmr, qlev1), q2),
    ).astype(np.float32)
    q2 = np.minimum(_saturation_mixing_ratio(psfc, t2), q2).astype(np.float32)
    return {"t2": t2, "th2": th2, "q2": q2}
