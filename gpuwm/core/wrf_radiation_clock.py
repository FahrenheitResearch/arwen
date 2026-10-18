"""WRF v4.6.1's radiation-call COSZEN, word for word.

WRF's radiation driver writes the COSZEN carrier on every radiation call as
``radconst`` + ``calc_coszen`` (phys/module_radiation_driver.F:1188-1189 and
:1206-1208), called with ``xtime + radt*0.5`` so the hour angle sits at the
middle of the radiation interval while JULIAN (declination, equation of
time) stays at the call time.  Every operand is REAL(4), and the three
calendar scalars come from WRF's frame, not from the radiation code:

``julian``
    frame/module_domain.F:2157-2165: ``REAL(dayOfYear_r8) - 1.0``, where
    WRF's time library gives ``dayOfYear_r8 = seconds_into_year/86400 + 1``
    in REAL(8) (external/esmf_time_f90/ESMF_Time.F90:439-450).  The
    rounding to REAL(4) happens at ``1 + day``, not at ``day``, so on the
    first day of the year and on days whose ``day`` and ``day + 1`` sit in
    different binades the word differs from ``REAL(day)``.
``gmt``
    frame/module_domain.F:2447, the simulation start hour as
    ``hr + real(mn)/60. + real(sec)/3600. + real(ms)/(1000*3600)``.
``xtime``
    frame/module_domain.F:2186-2191, minutes since the simulation start as
    ``REAL(days)*24.*60. + REAL(seconds)/60.`` (plus the fractional second).

``radt`` is the namelist radiation period in minutes, used raw: WRF adds
``radt*0.5`` whatever ``NINT(radt*60/dt)`` rounds the call cadence to, and
adds nothing when ``radt`` is 0.

The cosine itself is the ``swint_coszen_loc`` kernel (gpuwm/core/kernels/
swint.cu), which already evaluates ``radconst`` + ``calc_coszen`` with
single-rounded operations and WOOF's own float32 sine, cosine and arcsine;
4.6.1's ``calc_coszen`` then clamps to [-1, 1], applied here.
tools/analytic_clear_sky_oracle holds this module to WRF 4.6.1's own
routines, compiled unmodified, bit for bit.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

F = np.float32

__all__ = ["radiation_call_scalars", "wrf_calendar_scalars",
           "wrf_radiation_coszen"]

_KERNEL = None


def wrf_calendar_scalars(start_time: datetime, elapsed_seconds: float):
    """``(julian, xtime, gmt)`` as REAL(4), formed as WRF's frame forms them.

    ``start_time`` is the simulation start (gmt is its hour), and the
    current time is ``start_time + elapsed_seconds``.
    """
    if not isinstance(start_time, datetime):
        raise TypeError("start_time must be a datetime")
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("elapsed_seconds must be finite and non-negative")
    current = start_time + timedelta(seconds=elapsed)
    year_start = datetime(current.year, 1, 1, tzinfo=current.tzinfo)
    into_year = current - year_start
    rsec = (float(into_year.days * 86400 + into_year.seconds)
            + into_year.microseconds / 1.0e6)
    day_of_year_r8 = rsec / 86400.0 + 1.0
    julian = F(F(day_of_year_r8) - F(1.0))

    ms = int(round(start_time.microsecond / 1000.0))
    gmt = F(start_time.hour) + F(start_time.minute) / F(60.0)
    gmt = F(gmt + F(start_time.second) / F(3600.0))
    gmt = F(gmt + F(ms) / F(1000 * 3600))

    whole = math.floor(elapsed)
    days, seconds = divmod(int(whole), 86400)
    xtime = F(F(days) * F(24.0) * F(60.0)) + F(seconds) / F(60.0)
    fraction = elapsed - whole
    if fraction:
        xtime = F(xtime + F(F(fraction) / F(60.0)))
    return F(julian), F(xtime), F(gmt)


def radiation_call_scalars(start_time: datetime, elapsed_seconds: float,
                           radt_minutes: float):
    """``(julian, xtime + radt*0.5, gmt)``: what ``calc_coszen`` receives
    on a radiation call (module_radiation_driver.F:1206-1208)."""
    radt = float(radt_minutes)
    if not math.isfinite(radt) or radt < 0.0:
        raise ValueError("radt_minutes must be finite and non-negative")
    julian, xtime, gmt = wrf_calendar_scalars(start_time, elapsed_seconds)
    return julian, F(xtime + F(radt) * F(0.5)), gmt


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        from gpuwm.core.kernels import load_module
        _KERNEL = load_module("swint").get_function("swint_coszen_loc")
    return _KERNEL


def wrf_radiation_coszen(start_time: datetime, elapsed_seconds: float,
                         radt_minutes: float, latitude_deg, longitude_deg):
    """WRF's radiation-call COSZEN on the device, float32, shaped like the
    latitude grid."""
    import cupy as cp
    from gpuwm.core.rrtmg_legacy import _DEGRAD, _DPD

    lat = cp.ascontiguousarray(cp.asarray(latitude_deg, dtype=cp.float32))
    lon = cp.ascontiguousarray(cp.asarray(longitude_deg, dtype=cp.float32))
    if lat.shape != lon.shape:
        raise ValueError("radiation latitude/longitude shapes must match")
    julian, xtime, gmt = radiation_call_scalars(
        start_time, elapsed_seconds, radt_minutes)
    coszen = cp.empty(lat.shape, dtype=cp.float32)
    n = int(lat.size)
    if n:
        block = 256
        _kernel()(((n + block - 1) // block,), (block,),
                  (np.int64(n), lat.reshape(-1), lon.reshape(-1),
                   F(julian), F(xtime), F(gmt), F(_DEGRAD), F(_DPD),
                   np.int32(0),  # solar_480 = 0: WRF 4.6.1's radconst
                   coszen.reshape(-1)))
    # calc_coszen: coszen = min(max(coszen, -1.0), 1.0)
    return cp.minimum(cp.maximum(coszen, F(-1.0)), F(1.0))
