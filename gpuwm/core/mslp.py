"""Sea-level pressure reduction, and the smoother its consumers share.

They live here, in a stdlib-plus-numpy module, because of where they
have to be readable from.  Their historical home is
``gpuwm.verify.metrics``, and ``gpuwm/verify`` is a DEVELOPER
verification tree that the standalone CPU preprocessing distribution
deliberately omits -- so a core-layer import of it breaks that wheel
(``tests/test_native_wrf_distribution.py`` is the gate that says so, and
it caught exactly that: ``gpuwm.core.storm_tracking`` reduces sea-level
pressure to find a vortex centre, and reached across the boundary for
the transcription rather than restating it).

Same shape as ``sase_limits.py``: the values and the arithmetic move,
nothing is duplicated.  ``gpuwm.verify.metrics`` imports them back from
here, so its own API -- and the identity every case module and gate
binds to -- is unchanged.
"""

from __future__ import annotations

import numpy as np


def _dcomputeseaprs(pressure_pa, temperature, qv, height_msl) -> np.ndarray:
    """Sea-level pressure (hPa): wrf-python DCOMPUTESEAPRS (Shuell 1995).

    Transcribed from the validated wrf-rust oracle
    (wrf-core/src/diag/pressure.rs:60-147, which the metrics audit
    matched to wrf-python's Fortran to 4.6e-10 hPa on this case):
    reference level ~PCONST = 100 hPa above the surface, log-p
    interpolation of Tv/z to it, US-standard-lapse surface and
    sea-level temperatures, the TC = 290.66 K "ridiculous MM5 test"
    capping, and the two-temperature exponential reduction with
    wrf-python's rounded G = 9.81 / Rd = 287.  ``height_msl`` is full
    geopotential / 9.80665 (the oracle's height_msl).  Inputs are
    (nz, ny, nx) bottom-to-top with pressure in Pa; returns (ny, nx) hPa.
    """
    g_slp, rd_slp, ussalr = 9.81, 287.0, 0.0065
    pconst, tc = 10000.0, 273.16 + 17.5
    p_sfc = pressure_pa[0]
    above = (p_sfc[None] - pressure_pa) >= pconst
    if not above.any(axis=0).all():
        raise ValueError(
            "DCOMPUTESEAPRS requires a level 100 hPa above every surface")
    klo = np.argmax(above, axis=0)
    if int(klo.min()) < 1:
        raise ValueError("DCOMPUTESEAPRS reference level fell on k=0")
    khi = klo - 1

    def at(field, k):
        return np.take_along_axis(field, k[None], axis=0)[0]

    plo, phi_p = at(pressure_pa, klo), at(pressure_pa, khi)
    # Virtual temperature with wrf-python's 0.608 (pressure.rs:106-110).
    tlo = at(temperature, klo) * (1.0 + 0.608 * np.maximum(at(qv, klo), 0.0))
    thi = at(temperature, khi) * (1.0 + 0.608 * np.maximum(at(qv, khi), 0.0))
    zlo, zhi = at(height_msl, klo), at(height_msl, khi)
    p_at_pconst = p_sfc - pconst
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = ((np.log(p_at_pconst) - np.log(phi_p))
                / (np.log(plo) - np.log(phi_p)))
    degenerate = np.abs(plo - phi_p) < 1.0
    t_at_pconst = np.where(degenerate, tlo, thi + frac * (tlo - thi))
    z_at_pconst = np.where(degenerate, zlo, zhi + frac * (zlo - zhi))
    t_surf = t_at_pconst * (p_sfc / p_at_pconst) ** (ussalr * rd_slp / g_slp)
    t_sea_level = t_at_pconst + ussalr * z_at_pconst
    t_sea_level = np.where((t_surf <= tc) & (t_sea_level >= tc), tc,
                           tc - 0.005 * (t_surf - tc) ** 2)
    return 0.01 * p_sfc * np.exp(
        2.0 * g_slp * height_msl[0] / (rd_slp * (t_sea_level + t_surf)))


#: Nine-point smoother policy for the display treatment: three passes
#: with centre weight 4 (the Shuman 1957 smoother family; the same
#: shape and pass count wrf-python's documented SLP plotting workflow
#: applies via ``smooth2d(slp, 3)``).  A grid-scale checkerboard is
#: attenuated by (1/3)**3 = 1/27 per interior cell.
MSLP_SMOOTH_PASSES = 3
MSLP_SMOOTH_CENTER_WEIGHT = 4.0


def _nine_point_smooth(field: np.ndarray, passes: int,
                       center_weight: float) -> np.ndarray:
    """``passes`` applications of the standard nine-point smoother.

    Kernel: centre ``center_weight``, each of the eight neighbours 1,
    normalised; edges are handled by replicating the boundary row and
    column (no field is invented outside the domain).
    """
    out = np.asarray(field, dtype=np.float64)
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = (center_weight * out
               + padded[:-2, 1:-1] + padded[2:, 1:-1]
               + padded[1:-1, :-2] + padded[1:-1, 2:]
               + padded[:-2, :-2] + padded[:-2, 2:]
               + padded[2:, :-2] + padded[2:, 2:]) / (center_weight + 8.0)
    return out
