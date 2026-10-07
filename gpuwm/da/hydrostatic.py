"""Hydrostatic consistency of an analysis increment (DA design E2).

The analysis moves potential temperature, vapour and the hydrometeor masses
and leaves the column dry mass ``mu`` and the geopotential ``ph`` where the
background had them.  ``ph`` is prognostic in this dycore and the pressure is
diagnosed from it (``calc_p_alpha``: the layer thickness gives the inverse
density, the equation of state gives the pressure).  A warm increment at a
fixed thickness therefore reads as a pressure jump of about ``gamma *
dtheta / theta`` -- some 4 hPa for 1 K -- which the first acoustic substeps
answer with a vertical velocity burst and an acoustic/gravity ring.  That
is the insertion shock the one-shot cycle shows at f00:30.

What this module does
---------------------
:func:`geopotential_increment` returns the geopotential increment that
keeps the column hydrostatic at its own dry mass: the column is integrated
twice, once with the background's theta/vapour/water and once with the
analysis's, and only the difference is returned.

1. The hydrostatic pressure change from the water loading change, by the
   moist w-balance recurrence of WRF ``module_initialize_real``
   (:func:`gpuwm.ingest.real._rebalance_moist_pressure_serial`, the same
   rows, top down from the rigid lid).  Theta does not enter it: a
   hydrostatic pressure is the weight of what lies above.
2. The inverse dry density of both columns from the equation of state at
   those pressures (the background at the model's own diagnosed pressure,
   the analysis at that pressure plus step 1's change).
3. The layer thickness change from the inverse density change by the
   operator the runtime diagnoses with (``hypsometric_opt`` 1: ``-dnw *
   (c1h*mu + c2h) * alt``; 2: ``alt * phm * log(pfd/pfu)`` on the dry
   reference pressures), summed upward from the surface, where ``ph`` is
   the terrain and does not move.

Differencing two integrations of the same operator keeps the background's
own departure from hydrostatic balance (a storm's non-hydrostatic pressure
is the model's, not the analysis's to erase) and makes the result exactly
zero in every column the increment did not touch: both integrations see
identical inputs there and the arithmetic is column-local and fixed-order.

The dry mass stays as it is.  The analysis carries no surface-pressure
increment today; when one arrives it enters through ``mup`` and step 1.

All arithmetic is float64 on the array module of the state (cupy on a card,
numpy for a host state), column-vectorised with a loop over levels.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from gpuwm.core import constants as c

#: Registry ``moist`` masses: the species WRF's calc_cq and the real-init
#: recurrence sum as loading.  Number moments are not mass.
LOADING_MASSES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")

#: Fields whose increment changes the hydrostatic column.
COLUMN_FIELDS = ("thp",) + LOADING_MASSES

SCHEMA = "gpuwm-da.hydrostatic.v1"


class HydrostaticError(ValueError):
    """A refusal by this module."""


def _xp(array):
    if hasattr(array, "__cuda_array_interface__"):
        import cupy as cp
        return cp
    return np


def _f64(xp, value):
    return xp.asarray(value, dtype=xp.float64)


def _profile(xp, value, nz):
    """A base profile as (nz, ny, nx) or (nz, 1, 1), float64."""
    array = _f64(xp, value)
    if array.ndim == 1:
        if array.shape[0] != nz:
            raise HydrostaticError(
                f"base profile has {array.shape[0]} levels, the state {nz}")
        return array[:, None, None]
    return array


def loading_masses_for_scheme(mp_physics) -> tuple[str, ...]:
    """The moist masses a state of this microphysics carries, before it exists.

    The registry's moment rows (:func:`gpuwm.da.moments.scheme_moments`)
    are the masses ``gpuwm.core.state`` allocates per scheme; a scheme
    with no row gets every loading mass, the over-price that never runs a
    card out of memory.
    """
    from gpuwm.da.moments import scheme_moments
    try:
        masses = set(scheme_moments(int(mp_physics)).mass_fields)
    except (ValueError, KeyError):
        return LOADING_MASSES
    return tuple(name for name in LOADING_MASSES if name in masses)


def loading_masses(state) -> tuple[str, ...]:
    """The moist masses ``state`` carries (vapour first)."""
    return tuple(name for name in LOADING_MASSES
                 if getattr(state, name, None) is not None)


def hydrostatic_pressure_perturbation(xp, qtot, mup, mub, c1f, c2f, rdnw,
                                      rdn):
    """WRF's moist w-balance pressure recurrence, column-vectorised.

    The rows of ``module_initialize_real`` (F:3913-3935), as
    :func:`gpuwm.ingest.real._rebalance_moist_pressure_serial` spells them:
    the top half level from the rigid lid, then downward.  Returns the
    perturbation pressure (Pa) relative to the base state, float64,
    shaped like ``qtot``.
    """
    nz = qtot.shape[0]
    out = xp.empty_like(qtot)
    cq = 1.0 / (1.0 + qtot[-1])
    load = qtot[-1] * cq
    out[-1] = (-0.5 * (c1f[nz] * mup + load * (c1f[nz] * mub + c2f[nz]))
               / rdnw[nz - 1] / cq)
    for k in range(nz - 2, -1, -1):
        kw = k + 1
        qbar = 0.5 * (qtot[k] + qtot[k + 1])
        cq = 1.0 / (1.0 + qbar)
        load = qbar * cq
        out[k] = (out[k + 1]
                  - (c1f[kw] * mup + load * (c1f[kw] * mub + c2f[kw]))
                  / cq / rdn[kw])
    return out


def inverse_density(xp, theta, qv, pressure):
    """``alt`` from the equation of state: ``Rd theta (1 + Rv/Rd qv)
    (p/p0)^(R/cp) / p`` (WRF calc_p_rho_phi inverted), float64."""
    return (c.RD * theta * (1.0 + c.RVOVRD * qv)
            * xp.power(pressure / c.P0, c.RCP) / pressure)


def _layer_operator(xp, state, mu, hypsometric_opt: int):
    """``d(phi) = op * alt`` per layer, float64 (nz, ny, nx)."""
    if hypsometric_opt == 2:
        if state.p_top is None:
            raise HydrostaticError(
                "hypsometric_opt=2 needs state.p_top (load_base)")
        c3f = _f64(xp, state.c3f)[:, None, None]
        c4f = _f64(xp, state.c4f)[:, None, None]
        c3h = _f64(xp, state.c3h)[:, None, None]
        c4h = _f64(xp, state.c4h)[:, None, None]
        p_top = float(state.p_top)
        pfu = c3f[1:] * mu[None] + c4f[1:] + p_top
        pfd = c3f[:-1] * mu[None] + c4f[:-1] + p_top
        phm = c3h * mu[None] + c4h + p_top
        return phm * xp.log(pfd / pfu)
    if hypsometric_opt == 1:
        dnw = _f64(xp, state.dnw)[:, None, None]
        c1h = _f64(xp, state.c1h)[:, None, None]
        c2h = _f64(xp, state.c2h)[:, None, None]
        return -dnw * (c1h * mu[None] + c2h)
    raise HydrostaticError(
        f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")


def capture_column(state) -> dict:
    """Copies of the fields the column integration reads off a state:
    ``thp``, the loading masses and the diagnosed ``p`` (float64, on the
    state's array module).  Taken before a one-shot insertion so the
    rebalance differences against the background exactly, whatever the
    applier's clipping and saturation cap did to the increment."""
    xp = _xp(state.p)
    names = ("thp", "p") + loading_masses(state)
    return {name: _f64(xp, getattr(state, name)).copy() for name in names}


def _column_dphp(state, before: Mapping[str, object],
                 after: Mapping[str, object], *, hypsometric_opt: int):
    """The core: ``php`` change taking the column from ``before`` to
    ``after`` (both: ``thp`` and the loading masses, float64) at the dry
    mass of ``state``, ``before["p"]`` being the background's diagnosed
    pressure.  Returns ``(dphp, dp_hyd)``."""
    xp = _xp(state.p)
    nz = int(state.p.shape[0])
    masses = loading_masses(state)
    if "qv" not in masses:
        raise HydrostaticError(
            "a hydrostatic rebalance needs a moist state (no qv)")
    mub = _f64(xp, state.mub2d)
    mup = _f64(xp, state.mup)
    mu = mub + mup
    c1f = _f64(xp, state.c1f)
    c2f = _f64(xp, state.c2f)
    rdnw = _f64(xp, state.rdnw)
    rdn = _f64(xp, state.rdn)

    qtot_b = xp.zeros(state.p.shape, dtype=xp.float64)
    qtot_a = xp.zeros(state.p.shape, dtype=xp.float64)
    for name in masses:
        qtot_b += before[name]
        qtot_a += after[name]
    dp_hyd = (hydrostatic_pressure_perturbation(xp, qtot_a, mup, mub, c1f,
                                                c2f, rdnw, rdn)
              - hydrostatic_pressure_perturbation(xp, qtot_b, mup, mub, c1f,
                                                  c2f, rdnw, rdn))
    del qtot_a, qtot_b
    thb = _profile(xp, state.thb, nz)
    p_b = before["p"]
    p_a = p_b + dp_hyd
    if not bool(xp.isfinite(p_a).all()) or bool((p_a <= 0.0).any()):
        raise HydrostaticError(
            "the analysed hydrostatic pressure is not finite and positive")
    dalt = (inverse_density(xp, thb + after["thp"], after["qv"], p_a)
            - inverse_density(xp, thb + before["thp"], before["qv"], p_b))
    ddphi = _layer_operator(xp, state, mu, int(hypsometric_opt)) * dalt
    dphp = xp.zeros((nz + 1,) + tuple(state.p.shape[1:]), dtype=xp.float64)
    xp.cumsum(ddphi, axis=0, out=dphp[1:])
    if not bool(xp.isfinite(dphp).all()):
        raise HydrostaticError("the geopotential increment is not finite")
    return dphp, dp_hyd


def _receipt(names, state, dphp, dp_hyd, hypsometric_opt, how):
    xp = _xp(dphp)
    return {
        "schema": SCHEMA,
        "applied": True,
        "how": how,
        "column_fields": list(names),
        "loading_masses": list(loading_masses(state)),
        "hypsometric_opt": int(hypsometric_opt),
        "max_abs_dphp_m2s2": float(xp.abs(dphp).max()),
        "max_abs_dp_hyd_pa": float(xp.abs(dp_hyd).max()),
        "rule": ("php += integral(op * (alt(theta_a, qv_a, p + dp_hyd) - "
                 "alt(theta_b, qv_b, p))), dp_hyd from the moist w-balance "
                 "recurrence on the loading change; mu unchanged"),
    }


def geopotential_increment(state, increments: Mapping[str, object], *,
                           hypsometric_opt: int):
    """The ``php`` increment that keeps the analysed column hydrostatic.

    ``state`` is the BACKGROUND (diagnostics current: ``p`` is read), and
    ``increments`` the analysis increment as the cycle hands it to the
    applier (any fields; only :data:`COLUMN_FIELDS` matter here).  Negative
    water is clipped at zero as the forecast's own clamp will clip it.
    Returns ``(dphp, receipt)``: ``dphp`` float64 on the state's array
    module, shaped like ``state.php`` (zero at the surface), or ``None``
    when the increment names no column field.  The IAU arms force it with
    the rest of the increment.
    """
    names = [name for name in COLUMN_FIELDS if name in increments]
    if not names:
        return None, {"schema": SCHEMA, "applied": False,
                      "reason": "the increment names no column field"}
    xp = _xp(state.p)
    for name in names:
        if name != "thp" and getattr(state, name, None) is None:
            raise HydrostaticError(
                f"the increment names {name!r}, which this state does not "
                "carry")
    before = capture_column(state)
    after = {}
    for name, value in before.items():
        delta = increments.get(name)
        if delta is None or name == "p":
            after[name] = value
        elif name == "thp":
            after[name] = value + _f64(xp, delta)
        else:
            after[name] = xp.maximum(value + _f64(xp, delta), 0.0)
    dphp, dp_hyd = _column_dphp(state, before, after,
                                hypsometric_opt=hypsometric_opt)
    return dphp, _receipt(names, state, dphp, dp_hyd, hypsometric_opt,
                          "increment-forced")


#: What a state must carry for a column to be integrated.
COLUMN_SETUP_ATTRS = ("mub2d", "mup", "c1f", "c2f", "rdnw", "rdn", "thb",
                      "php", "p")


def rebalance_columns(state, before: Mapping[str, object], *,
                      hypsometric_opt: int, names=(), how: str) -> dict:
    """Add the hydrostatic ``php`` change to an ALREADY changed state.

    ``before`` is :func:`capture_column` of the state taken before the
    thermodynamic change; the changed side is read off ``state`` as the
    writer left it (clipped, saturation-capped, moment-repaired).  Only
    ``php`` is written, and only where the change reaches: a column the
    change did not touch keeps its words byte for byte (``-0.0 + 0.0``
    would otherwise come back ``+0.0``).  The diagnosed pressure and
    density are NOT refreshed here; the caller owns that refresh, as it
    does for the perturbation contract.
    """
    xp = _xp(state.p)
    after = capture_column(state)
    dphp, dp_hyd = _column_dphp(state, before, after,
                                hypsometric_opt=hypsometric_opt)
    moved = dphp != 0.0
    state.php[...] = xp.where(
        moved, (state.php.astype(dphp.dtype) + dphp).astype(state.php.dtype),
        state.php)
    receipt = _receipt(names or COLUMN_FIELDS, state, dphp, dp_hyd,
                       hypsometric_opt, how)
    receipt["columns_moved"] = int(xp.count_nonzero(moved.any(axis=0)))
    return receipt


def rebalance_after_insertion(state, before: Mapping[str, object], *,
                              hypsometric_opt: int, names=()) -> dict:
    """Add the hydrostatic ``php`` change to an ALREADY analysed state.

    The one-shot insertion: ``before`` is :func:`capture_column` of the
    background taken before the applier wrote; the analysed side is read
    off ``state`` as the applier left it (clipped, saturation-capped,
    moment-repaired).  Refreshes the diagnosed pressure and density.
    """
    from gpuwm.core.diagnostics import update_diagnostics

    receipt = rebalance_columns(state, before, hypsometric_opt=hypsometric_opt,
                                names=names, how="one-shot, after the applier")
    update_diagnostics(state, int(hypsometric_opt))
    return receipt
