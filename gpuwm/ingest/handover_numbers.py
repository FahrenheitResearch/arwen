"""Number concentrations for hydrometeor mass handed from one model to another.

THE BREAKAGE.  Every route that starts a child from another model's state
(``prep --source hex-global`` and the nest add-ons that feed it, ``gpuwm
downscale``, any analysis carrying condensate) can hand the child a
hydrometeor MASS whose NUMBER concentration is zero: the parent published no
number (a single-moment parent such as hex's WSM6), the child's scheme
carries a moment the parent's does not, or an interpolation clamp zeroed the
number beside a positive mass.  A two-moment scheme handed ``q > 0, N = 0``
sizes the mass at its slope limit.  Morrison (mp=10) then holds 2.8 mm rain,
2 mm snow and graupel and 350 um ice (module_mp_morr_two_moment.F
lammin bounds, morrison.cu ``morr_bound_one``); WRF's real.exe leaves exactly
those zeros for Morrison (it seeds numbers for Thompson only,
module_initialize_real.F:4829-4852).  Measured on the init-from-any-run
proof (GFS 2024-05-06 12Z, hex 15 km -> ArWen 3 km -> ArWen 1 km): the 3 km
child's first map was empty (341,746 NaN reflectivity values), every hourly
boundary update brought 469-518 such columns into its outer rows (40-55 dBZ
edge bands), and the 1 km child started with 60-70 dBZ rings, 151 dBZ peaks
and 32 m/s updrafts 15 minutes in where its parent had 4 m/s.

WHAT THIS DOES.  Where a child scheme's number is at or below zero and its
paired mass is above the scheme's activity floor, the number is diagnosed
from the mass with relations of that scheme's family; a number the state
already holds above zero (one carried from the parent) is kept.  Thompson
(mp 8 and 28) keeps its own real.exe seeding, already exact in
:mod:`gpuwm.ingest.real` for the analysed-state routes; this module applies
the same make_RainNumber / make_IceNumber / make_DropletNumber
(:mod:`gpuwm.core.thompson_entry`) on the routes that do not pass through
that initializer (``gpuwm downscale``).  Morrison has no mass-to-number
relation of its own, so its closure is written here, from the distributions
the scheme itself assumes (inverse-exponential, mu = 0, the scheme's own
densities and lambda bounds):

* rain: Thompson's make_RainNumber (an exponential with the Marshall-Palmer
  intercept 8e6 m^-4 above 0 C, 8e8 at or below -2 C, which is the
  distribution family Morrison's rain carries), then Morrison's
  [lamminr, lammaxr] bound;
* cloud ice: the Cooper (1986) curve Morrison nucleates ice with
  (morrison.cu INUC = 0: 0.005 exp(0.304 (273.15 - T)) per litre, at most
  500 per litre), then Morrison's [lammini, lammaxi] bound;
* snow: an exponential with the Gunn-Marshall intercept 3e6 m^-4, then
  [lammins, lammaxs];
* graupel: an exponential with Lin et al. (1983)'s intercept, 4e6 m^-4 for
  graupel (morr_rimed_ice = 0, 400 kg m^-3) and 4e4 m^-4 for hail
  (morr_rimed_ice = 1, 900 kg m^-3), then [lamming, lammaxg];
* cloud droplets: Morrison's droplet number is not transported; the scheme
  writes NDCNST = 250 per cm^3 over every cell it runs on, every call
  (morrison.cu ``nc3d = NDCNST*1e6/rho``).  A handed-over Morrison state
  starts with exactly that value everywhere, so the first frame and any
  cell the scheme does not visit carry the number the scheme would.

THE SWITCH.  The closure is correct-mode finding HN1
(:mod:`gpuwm.correct_mode`), RunConfig key ``hydrometeor_handover_numbers``.
``physics.mode = "exact"`` authors the key False, and so does a strict WRF
process (``GPUWM_WRF_EXACT=1``, which also refuses it True): those keep
real.exe's zeros and WRF's tendency-only scalar ring, byte-identical to the
code before this module existed.  Every gate reads the run's configuration
(:func:`closure_active`), never the process alone: a ``physics.mode =
"exact"`` run without the environment variable once still got the closure.
"""
from __future__ import annotations

from typing import Mapping, MutableMapping

import numpy as np

#: Morrison's own constants (morrison_constants.cuh, the
#: module_mp_morr_two_moment.F MORR_TWO_MOMENT_INIT values).
MORRISON_RHOW = 997.0
MORRISON_RHOI = 500.0
MORRISON_RHOSN = 100.0
MORRISON_RHOG = {0: 400.0, 1: 900.0}
MORRISON_NDCNST_PER_CM3 = 250.0
MORRISON_LAMBDA_BOUNDS = {
    # (lammin, lammax) in m^-1
    "rain": (1.0 / 2800.0e-6, 1.0 / 20.0e-6),
    "ice": (1.0 / (2.0 * 125.0e-6 + 100.0e-6), 1.0 / 1.0e-6),
    "snow": (1.0 / 2000.0e-6, 1.0 / 10.0e-6),
    "graupel": (1.0 / 2000.0e-6, 1.0 / 20.0e-6),
}
#: Exponential intercepts (m^-4) for the species Morrison sizes from a fixed
#: intercept here (rain takes make_RainNumber's temperature ramp instead).
MORRISON_SNOW_N0 = 3.0e6
MORRISON_GRAUPEL_N0 = {0: 4.0e6, 1: 4.0e4}
#: Cooper (1986) as Morrison evaluates it, per m^3, and its cap.
COOPER_COEFF_PER_M3 = 0.005 * 1000.0
COOPER_SLOPE = 0.304
COOPER_CAP_PER_M3 = 500.0e3

#: (mass, number) pairs a handed-over state closes, per child scheme.  The
#: Thompson rows are real.exe's own (gpuwm.boundary_fields
#: COLD_START_SEEDED_NUMBERS); the Morrison row is this module's closure.
MORRISON_HANDOVER_NUMBERS = (("qr", "nr"), ("qi", "ni"), ("qs", "ns"), ("qg", "ng"))
_SPECIES_KIND = {"qc": "cloud", "qr": "rain", "qi": "ice", "qs": "snow", "qg": "graupel"}

#: The number moment that sizes each hydrometeor mass, where a scheme
#: carries one (the ring forces a number back together with its mass).
PAIRED_NUMBER_OF_MASS = {"qc": "nc", "qr": "nr", "qi": "ni", "qs": "ns", "qg": "ng"}


#: The RunConfig key of correct-mode finding HN1 (gpuwm/correct_mode.py).
HANDOVER_NUMBERS_KEY = "hydrometeor_handover_numbers"


def closure_active(cfg) -> bool:
    """Whether ``cfg`` (a RunConfig, a mapping, or None) runs the closure.

    Its ``hydrometeor_handover_numbers`` key decides, as every correct-mode
    key does: ``physics.mode = "exact"`` sets it False.  A configuration that
    does not name the key (None, or an object without it) reads the RunConfig
    default, :data:`gpuwm.correct_mode.CORRECT_KEYS_DEFAULT`.  Always False in
    a strict WRF process, which keeps real.exe's zeros.
    """
    from gpuwm import correct_mode
    if correct_mode.STRICT:
        return False
    default = correct_mode.CORRECT_KEYS_DEFAULT
    if cfg is None:
        value = default
    elif isinstance(cfg, Mapping):
        value = cfg.get(HANDOVER_NUMBERS_KEY, default)
    else:
        value = getattr(cfg, HANDOVER_NUMBERS_KEY, default)
    return value is True


def _closure_pairs(mp: int) -> tuple[tuple[str, str], ...]:
    if mp == 10:
        return MORRISON_HANDOVER_NUMBERS
    from gpuwm.boundary_fields import COLD_START_SEEDED_NUMBERS
    return tuple(COLD_START_SEEDED_NUMBERS.get(mp, ()))


def handover_number_pairs(mp_physics: int, cfg) -> tuple[tuple[str, str], ...]:
    """The (mass, number) pairs a child of ``mp_physics`` under ``cfg`` seeds.

    Thompson's rows are real.exe's own seeds and hold in every mode; the
    Morrison row is this module's closure and only where it is on.
    """
    mp = int(mp_physics)
    if mp == 10 and not closure_active(cfg):
        return ()
    return _closure_pairs(mp)


def _xp(value):
    try:
        import cupy
        return cupy.get_array_module(value)
    except Exception:  # cupy absent: host only
        return np


def morrison_droplet_number(rho, xp=None):
    """Morrison's diagnostic droplet number, per kg, at density ``rho``."""
    xp = xp or _xp(rho)
    return (xp.float32(MORRISON_NDCNST_PER_CM3 * 1.0e6)
            / xp.asarray(rho, dtype=xp.float32)).astype(xp.float32)


def _bounded_number(xp, mass_per_m3, number_per_m3, density, kind):
    """Morrison's own lambda limiter on an inverse-exponential (mu = 0)."""
    lammin, lammax = MORRISON_LAMBDA_BOUNDS[kind]
    c = np.pi * density
    lam = xp.cbrt(c * number_per_m3 / mass_per_m3)
    lam = xp.clip(lam, lammin, lammax)
    return lam * lam * lam * mass_per_m3 / c


def morrison_least_number(kind: str, mass, *, rimed: int = 1):
    """The number (per kg) at which ``mass`` (kg/kg) sits on Morrison's lammin."""
    density = {"rain": MORRISON_RHOW, "ice": MORRISON_RHOI,
               "snow": MORRISON_RHOSN,
               "graupel": MORRISON_RHOG[int(rimed)]}[kind]
    lammin = MORRISON_LAMBDA_BOUNDS[kind][0]
    return lammin ** 3 * mass / (np.pi * density)


def morrison_number_per_kg(kind: str, mass, rho, temperature, *, rimed: int = 1):
    """Diagnosed Morrison number (per kg) for ``mass`` (kg/kg) of ``kind``."""
    xp = _xp(mass)
    q = xp.asarray(mass, dtype=xp.float64)
    rho64 = xp.asarray(rho, dtype=xp.float64)
    mass_m3 = q * rho64
    if kind == "rain":
        from gpuwm.core.thompson_entry import make_rain_number
        host_mass = mass_m3.get() if xp is not np else mass_m3
        host_t = (xp.asarray(temperature).get() if xp is not np
                  else np.asarray(temperature))
        number = xp.asarray(make_rain_number(host_mass, host_t), dtype=xp.float64)
        number = _bounded_number(xp, mass_m3, number, MORRISON_RHOW, "rain")
    elif kind == "ice":
        t = xp.asarray(temperature, dtype=xp.float64)
        number = xp.minimum(COOPER_COEFF_PER_M3 * xp.exp(COOPER_SLOPE * (273.15 - t)),
                            COOPER_CAP_PER_M3)
        number = _bounded_number(xp, mass_m3, number, MORRISON_RHOI, "ice")
    elif kind in ("snow", "graupel"):
        density = MORRISON_RHOSN if kind == "snow" else MORRISON_RHOG[int(rimed)]
        n0 = MORRISON_SNOW_N0 if kind == "snow" else MORRISON_GRAUPEL_N0[int(rimed)]
        lam = (np.pi * density * n0 / mass_m3) ** 0.25
        number = _bounded_number(xp, mass_m3, n0 / lam, density, kind)
    else:
        raise ValueError(f"no Morrison number relation for {kind!r}")
    # A vanishing mass keeps a representable positive number (the smallest
    # normal FP32) rather than rounding to the zero this closure removes.
    return xp.maximum(number / rho64, 1.1754944e-38).astype(xp.float32)


def _thompson_number_per_kg(kind, mass, rho, temperature, *, nwfa=None, xland=None):
    from gpuwm.core.thompson_entry import (make_droplet_number, make_ice_number,
                                           make_rain_number)
    q = np.asarray(mass, dtype=np.float32)
    r = np.asarray(rho, dtype=np.float32)
    mass_m3 = (q * r).astype(np.float32)
    if kind == "rain":
        number = make_rain_number(mass_m3, temperature)
    elif kind == "ice":
        number = make_ice_number(mass_m3, temperature)
    elif kind == "cloud":
        aerosol = (np.zeros_like(mass_m3) if nwfa is None
                   else (np.asarray(nwfa, dtype=np.float32) * r).astype(np.float32))
        land = np.ones_like(mass_m3) if xland is None else np.asarray(xland, dtype=np.float32)
        number = make_droplet_number(mass_m3, aerosol, land)
    else:
        raise ValueError(f"no Thompson number relation for {kind!r}")
    return (np.asarray(number, dtype=np.float32) / r).astype(np.float32)


def take(array, idx, shape):
    """``array`` (broadcast to ``shape``) at index tuple ``idx``, on idx's side.

    Host arrays indexed by device indices (or the reverse) are moved to the
    index's side first, element by element only for the seeded cells.
    """
    ixp = _xp(idx[0])
    axp = _xp(array)
    if axp is not ixp:
        if ixp is np:          # device array, host index
            array = axp.asnumpy(axp.broadcast_to(array, shape)[tuple(axp.asarray(i) for i in idx)])
            return array
        host_idx = tuple(i.get() for i in idx)   # host array, device index
        return ixp.asarray(np.broadcast_to(np.asarray(array), shape)[host_idx])
    return axp.broadcast_to(axp.asarray(array), shape)[idx]


def close_numbers(mp_physics: int, masses: Mapping[str, object],
                  numbers: MutableMapping[str, object], *, rho, temperature,
                  rimed: int = 1, nwfa=None, xland=None,
                  bound_thompson: bool = False) -> dict | None:
    """Fill ``numbers`` in place where the paired mass has no number.

    ``bound_thompson`` also applies Thompson's entry clamps to numbers the
    child already holds (:func:`_bound_thompson_numbers`); the offline
    child asks for it, because its numbers are carried across a scheme edge.

    ``masses`` and ``numbers`` map state names to same-shaped uncoupled
    arrays (per kg); ``rho`` (kg m^-3) and ``temperature`` (K) match them.
    Only cells with mass above the activity floor and a number at or below
    zero are written.  Returns a receipt, or None when the scheme closes
    nothing here (no pairs, or a strict process).  The caller gates on the
    run's configuration (:func:`closure_active`).
    """
    from gpuwm import correct_mode
    mp = int(mp_physics)
    pairs = _closure_pairs(mp)
    if not pairs or correct_mode.STRICT:
        return None
    # rho and temperature may be callables taking the seeded cells' index
    # tuple and returning their values there, so only the seeded cells'
    # density and temperature are ever formed.
    def at(value, idx, shape):
        if callable(value):
            return value(idx)
        return take(value, idx, shape)

    receipt = {"mp_physics": mp, "species": {}, "cells_total": 0,
               "policy": ("morrison-handover-closure-v1" if mp == 10
                          else "thompson-real-exe-seed")}
    for mass_name, number_name in pairs:
        if mass_name not in masses or number_name not in numbers:
            continue
        q = masses[mass_name]
        n = numbers[number_name]
        xp = _xp(n)
        # Any mass above zero, as real.exe's Thompson seed reads it: a
        # number left at zero beside a positive mass is the defect itself.
        seed = (xp.asarray(q) > 0) & (xp.asarray(n) <= 0)
        if mp == 10:
            # So is a number far below the least Morrison's own limiter
            # allows its mass (lambda under lammin): an interpolation that
            # carried the mass and rounded the number to ~0 (measured: rain
            # 6.4e-8 kg/kg beside 2.6e-16 per kg, a 151 dBZ first-frame
            # cell).  Half the limiter's number, so a parent cell the scheme
            # itself left at its bound is kept.
            seed |= (xp.asarray(q) > 0) & (
                xp.asarray(n) < 0.5 * morrison_least_number(_SPECIES_KIND[mass_name],
                                                            xp.asarray(q), rimed=rimed))
        count = int(seed.sum())
        receipt["species"][number_name] = count
        receipt[f"{_SPECIES_KIND[mass_name]}_number_seed"] = {"seeded_cells": count}
        receipt["cells_total"] += count
        if not count:
            continue
        idx = xp.nonzero(seed)
        qs = xp.asarray(q)[idx]
        rs = xp.asarray(at(rho, idx, seed.shape))
        ts = xp.asarray(at(temperature, idx, seed.shape))
        kind = _SPECIES_KIND[mass_name]
        if mp == 10:
            value = morrison_number_per_kg(kind, qs, rs, ts, rimed=rimed)
        else:
            host = (lambda a: a.get() if xp is not np else np.asarray(a))
            aero = None if nwfa is None else host(take(nwfa, idx, seed.shape))
            # A 2-D land mask broadcasts over the levels.
            land = None if xland is None else host(take(xland, idx, seed.shape))
            value = xp.asarray(_thompson_number_per_kg(
                kind, host(qs), host(rs), host(ts), nwfa=aero, xland=land))
        if not bool(xp.isfinite(value).all()) or bool((value <= 0).any()):
            raise FloatingPointError(
                f"the {number_name} closure produced a non-finite or non-positive "
                f"number in {count} seeded cell(s); refusing rather than handing "
                "the child a broken moment")
        n[idx] = value.astype(n.dtype)
    if mp in (8, 28) and bound_thompson:
        _bound_thompson_numbers(masses, numbers, rho=rho, at=at,
                                receipt=receipt)
    return receipt


def _bound_thompson_numbers(masses, numbers, *, rho, at, receipt) -> None:
    """Thompson's own entry clamps on numbers a child was handed.

    THE BREAKAGE THIS PREVENTS (CP-CHILD-PHYSICS-2026-10-24): a Thompson
    child of a Morrison parent carries the parent's rain number, and where
    Morrison holds a trace of rain with almost no drops (7e-8 kg/kg beside
    1e-6 per kg) the child's analysis frame drew 50-66 dBZ points the
    parent's own map does not have.  Thompson bounds the rain median volume
    diameter to [0.75 D0r, 2.5 mm] and cloud ice to [5, 300] um at its first
    call (module_mp_thompson.F:1853-1898, ported in
    :mod:`gpuwm.core.thompson_entry`); a handed-over number outside those
    bounds is set to the bound here, so the frames written before the first
    step show the state the scheme runs.  Numbers already inside are left
    bit for bit.
    """
    from gpuwm.core.thompson_entry import R1, np_thompson_entry_numbers

    for mass_name, number_name, species in (("qr", "nr", "rain"),
                                            ("qi", "ni", "ice")):
        if mass_name not in masses or number_name not in numbers:
            continue
        n = numbers[number_name]
        xp = _xp(n)
        q = xp.asarray(masses[mass_name])
        live = (q > R1) & (xp.asarray(n) > 0)
        if not bool(live.any()):
            continue
        idx = xp.nonzero(live)
        host = (lambda a: a.get() if xp is not np else np.asarray(a))
        q_live = host(q[idx])
        n_live = host(xp.asarray(n)[idx])
        rho_live = host(xp.asarray(at(rho, idx, live.shape)))
        bounded = np_thompson_entry_numbers(species, q_live, n_live, rho_live)
        moved = np.abs(bounded - n_live) > 1.0e-3 * np.maximum(n_live, 1.0e-30)
        count = int(moved.sum())
        receipt[f"{species}_number_bounded"] = {"cells": count}
        if not count:
            continue
        cells = tuple(xp.asarray(np.asarray(host(i))[moved]) for i in idx)
        n[cells] = xp.asarray(bounded[moved]).astype(n.dtype)


def temperature_from_theta(theta, pressure):
    """Absolute temperature from potential temperature and full pressure."""
    xp = _xp(theta)
    return (xp.asarray(theta, dtype=xp.float64)
            * (xp.asarray(pressure, dtype=xp.float64) / 1.0e5) ** (287.0 / 1004.5))


def density_from_state(pressure, temperature, qv):
    """Moist-air density from the equation of state (kg m^-3)."""
    xp = _xp(pressure)
    return (xp.asarray(pressure, dtype=xp.float64)
            / (287.0 * xp.asarray(temperature, dtype=xp.float64)
               * (1.0 + 0.608 * xp.asarray(qv, dtype=xp.float64))))


__all__ = [
    "HANDOVER_NUMBERS_KEY", "MORRISON_HANDOVER_NUMBERS", "close_numbers",
    "closure_active",
    "density_from_state", "handover_number_pairs", "morrison_droplet_number",
    "morrison_number_per_kg", "temperature_from_theta",
]
