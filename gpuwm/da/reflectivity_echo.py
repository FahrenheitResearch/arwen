"""Reflectivity echo conditioning: a common echo floor and outlier tempering.

The breakage this prevents, named (CONUS DA first light, box E, 9 km,
2026-10-01 19Z analysis, 32 members, mp=28):

* The superob writer keeps every return of -15 dBZ or more as a
  reflectivity observation (:class:`gpuwm.obs.superob.SuperobParams`
  ``min_reflectivity_dbz``), while the mp=28 forward operator floors at
  -35 dBZ.  64 % of the 684,835 merged observations were below 15 dBZ and
  14 % below 0 dBZ, and they carried the SMALLEST stated errors (about
  3 dB, against 8.5 dB above 40 dBZ).  Each one sat against a member H(x)
  near -35 dBZ: the batch innovation was +36 dBZ on average (rms 39)
  against an ensemble spread of 4.2 dBZ and an error of 4.0 dBZ.
* The spread that exists below about 15 dBZ is trace condensate
  (1e-7 kg/kg) in a few members.  The LETKF regresses a 30 to 50 dB
  innovation onto it and extrapolates far outside the ensemble: member 0's
  increment reached 16.8 K in theta, 57 m/s in wind and 12 g/kg in vapour,
  added three times more vapour than condensate, and 36 % of that vapour
  went into columns whose strongest observed echo was 0 to 15 dBZ.
* One hour later the 32 members held 2.2 times their own pre-analysis
  echo area at every threshold (35 dBZ ratio to MRMS 2.79, no-DA control
  1.42; 20 dBZ area 16,226 cells against MRMS 8,008).

Two corrections, both default-on and applied to the echo batch only (the
clear-air batch is untouched and keeps its own floor and error):

1. **Common echo floor.**  Observation and H(x) are both raised to
   ``floor_dbz`` before differencing (the convention of DART's radar
   operator, which floors observation and forward operator at one common
   value, and of the WoFS-family systems that treat returns below 15 dBZ
   as no precipitation echo).  15 dBZ is this engine's own echo threshold
   (:class:`gpuwm.da.hotstart` ``echo_threshold_dbz``).  A weak return then
   says "no significant echo": it still pulls a member whose H(x) exceeds
   the floor down to it, and it no longer asks the filter to grow
   precipitation out of trace condensate.
2. **Outlier tempering** (a Huber-type observation-error inflation in the
   family of Minamide and Zhang 2017, Mon. Wea. Rev. 145, 1063-1081, with
   DART's 3-sigma outlier threshold): an observation whose innovation ``d``
   lies more than ``k`` standard deviations outside ``sqrt(s^2 +
   sigma_o^2)`` (``s`` the ensemble spread of H(x)) has its error variance
   raised to ``d^2 / k^2 - s^2``, which puts it exactly at ``k`` sigma.
   Where the ensemble cannot reach an observation the filter is told so
   through sigma_o, so the increment stays inside what the members can
   represent instead of extrapolating past them.  Observations inside
   ``k`` sigma are untouched: plain AOEI (max(sigma_o^2, d^2 - s^2))
   inflated 118 of 384 statistically consistent observations in the
   perfect-model twin of tests/test_radar_assimilation.py and cost its
   rain analysis its skill (posterior/prior error 0.924 to 1.018), while
   tempering beyond 3 sigma leaves that twin's analysis unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

#: The engine's own echo threshold (gpuwm.da.hotstart echo_threshold_dbz)
#: and the WoFS-family precipitation-echo threshold.
DEFAULT_REFLECTIVITY_FLOOR_DBZ = 15.0
#: Outlier tempering on the echo batch, in standard deviations of
#: sqrt(spread^2 + sigma_o^2); on by default.  DART's outlier threshold.
DEFAULT_REFLECTIVITY_OUTLIER_SIGMAS = 3.0

SCHEMA = "gpuwm-da.reflectivity-echo-conditioning.v1"


class ReflectivityEchoError(ValueError):
    """A conditioning request this module cannot honour."""


def check_sigmas(value, label: str = "reflectivity_outlier_sigmas"):
    """``None`` or a finite positive number of standard deviations."""
    if value is None:
        return None
    k = float(value)
    if not np.isfinite(k) or k <= 0.0:
        raise ReflectivityEchoError(
            f"{label} must be finite and positive or None, got {value!r}")
    return k


def check_floor(value, label: str = "reflectivity_floor_dbz"):
    """``None`` or a finite dBZ value; anything else is refused."""
    if value is None:
        return None
    floor = float(value)
    if not np.isfinite(floor):
        raise ReflectivityEchoError(
            f"{label} must be finite or None, got {value!r}")
    return floor


def _stats(y, sim, err):
    hx = sim.mean(axis=0)
    d = y - hx
    spread = (sim.std(axis=0, ddof=1) if sim.shape[0] > 1
              else np.zeros_like(hx))
    return d, spread, {
        "innovation_mean": float(d.mean()) if d.size else 0.0,
        "innovation_rms": float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0,
        "ensemble_spread_mean": float(spread.mean()) if d.size else 0.0,
        "obs_error_mean": float(err.mean()) if d.size else 0.0,
    }


def condition_reflectivity_batch(batch, *, floor_dbz, outlier_sigmas):
    """``(batch, receipt)``: the echo batch with the common floor applied to
    observation and H(x), then the outlier tempering, in that order.

    The input batch is never written to: its ``simulated`` array is shared
    with the clear-air batch, which must keep the raw H(x).  With
    ``floor_dbz=None`` and ``outlier_sigmas=None`` the same batch object is
    returned and the receipt says nothing was applied.
    """
    floor = check_floor(floor_dbz)
    k = check_sigmas(outlier_sigmas)
    mask = np.asarray(batch.mask, dtype=bool)
    receipt = {
        "schema": SCHEMA,
        "batch": batch.name,
        "floor_dbz": floor,
        "outlier_sigmas": k,
        "observations": int(np.count_nonzero(mask)),
    }
    if floor is None and k is None:
        receipt["applied"] = False
        return batch, receipt

    values = np.asarray(batch.values, dtype=np.float64)
    simulated = np.asarray(batch.simulated)
    errors_in = np.asarray(batch.errors, dtype=np.float64)
    y = values[mask]
    sim = simulated[:, mask].astype(np.float64)
    err = (np.full(y.shape, float(errors_in)) if errors_in.ndim == 0
           else errors_in[mask])
    _, _, before = _stats(y, sim, err)
    receipt["before"] = before

    if floor is not None:
        receipt["observations_raised_to_floor"] = int(
            np.count_nonzero(y < floor))
        receipt["member_points_raised_to_floor"] = int(
            np.count_nonzero(sim < floor))
        y = np.maximum(y, floor)
        sim = np.maximum(sim, floor)
        new_values = values.copy()
        new_values[mask] = y
        # H(x) is only read at masked points (letkf.GriddedObs); a new
        # array, never the shared one, carries the floored member values.
        new_sim = np.array(simulated, dtype=np.float64, copy=True)
        new_sim[:, mask] = sim
    else:
        new_values, new_sim = values, simulated

    if k is not None:
        d, spread, _ = _stats(y, sim, err)
        variance = np.maximum(err ** 2, d ** 2 / k ** 2 - spread ** 2)
        eff = np.sqrt(variance)
        inflated = eff > err
        receipt["observations_error_inflated"] = int(
            np.count_nonzero(inflated))
        receipt["error_inflation_factor_mean"] = (
            float((eff / err).mean()) if eff.size else 1.0)
        receipt["error_inflation_factor_max"] = (
            float((eff / err).max()) if eff.size else 1.0)
        new_errors = (np.full(values.shape, float(errors_in))
                      if errors_in.ndim == 0 else errors_in.copy())
        new_errors[mask] = eff
        err = eff
    else:
        new_errors = batch.errors

    _, _, after = _stats(y, sim, err)
    receipt["after"] = after
    receipt["applied"] = True
    return replace(batch, values=new_values, errors=new_errors,
                   simulated=new_sim), receipt
