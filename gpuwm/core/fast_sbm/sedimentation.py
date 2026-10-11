"""Sedimentation and surface precipitation of the fast SBM, float64 on the host.

Transcribed from WRF v4.8.0 ``phys/module_mp_fast_sbm.F`` (LF SHA-256
``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``,
identical to v4.6.1).  WRF is NCAR/UCAR public domain; the notice is in the
repository ``NOTICE`` and in ``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.  The
scheme is FSBM-2 by Jacob Shpund and co-authors (see the package docstring);
the module comment of ``FALFLUXHUCM_Z`` says the fall scheme was adapted from
GSFC code for the Hebrew University Cloud Model.

Routines and lines transcribed
------------------------------
* ``FAST_SBM`` :4269-4276, the level heights ``zcgs`` (``level_heights_cgs``)
  and the CGS pressure and density (:4271, :4272; ``p_cgs_from_si``).
* ``FAST_SBM`` :4518-4531, the table fall speeds corrected for pressure,
  ``FACTOR_P = DSQRT(1.0D6/PCGS)`` (:4519) times ``VR1``, ``VR3``, ``VR4``
  and ``VR5`` (:4523-4530; ``corrected_velocities``).
* ``FAST_SBM`` :4891-4962, the sedimentation stage: per column, drops
  (:4896-4914), then with ``ICEPROCS = 1`` snow (:4916-4935) and hail or
  graupel (:4936-4959), each through the ``chem_new/rhocgs`` and
  ``ffx_z*rhocgs`` round trip around ``FALFLUXHUCM_Z`` (``fall_species``,
  ``sediment_column``).
* ``FALFLUXHUCM_Z`` :5324-5382 (``falflux``): the per-bin activity test at
  1e-20 (:5348-5352), ``TFALL`` as the smallest level height over fall speed
  (:5354-5359), the bare ``STOP`` when ``TFALL >= 1e10`` (:5360),
  ``NSUB = INT(2.0*DT/TFALL)+1`` and ``DTFALL = DT/NSUB`` (:5361-5362), the
  upstream flux divergence over the distance between level centres
  (:5364-5370), the top-level form (:5371-5373) and the update (:5374-5376).
* ``FAST_SBM`` :5056-5121, surface precipitation (``surface_precip_level1``,
  ``surface_precip_wrf_block``): drops into RAINNC and RAINNCV (:5058-5071),
  snow into RAINNC, RAINNCV, SNOWNC and SNOWNCV (:5072-5088), and hail
  (:5092-5105) or graupel (:5106-5119) into RAINNC, RAINNCV, GRAUPELNC and
  GRAUPELNCV.

Beside them, the corrected forms of findings D1 and D2 of the fast-sbm design
(section 3.7), each selected by the caller through ``FsbmBehavior``:

* D1 (``rainnc_velocity``).  ``VR1_Z``, ``VR3_Z``, ``VR4_Z`` and ``VR5_Z`` have
  no i, j index (:4132-4138); every point overwrites them (:4523-4526), and
  the surface loop reads level 1 after the point loop (:5064, :5075, :5093,
  :5107), so every column's surface precipitation uses the fall-speed
  pressure factor of the tile's last column in loop order (j outer, i
  inner).  ``surface_fall_speeds`` returns either that or each column's own.
* D2 (``rainnc_source``).  WRF's RAINNC is the fall speed times the level-1
  spectrum after sedimentation, times dt (:5058-5071), not what left the
  column.  ``falflux`` also returns the bottom-face outflow integrated over
  the substeps, and ``surface_precip_outflow`` turns it into millimetres.

D8, measured and not changed here.  ``FALFLUXHUCM_Z`` divides each level's
flux divergence by ``ZCGS(K+1)-ZCGS(K)``, the distance to the level centre
above (:5368-5369), and the top level by ``ZCGS(KTE)-ZCGS(KTE-1)``
(:5373), not by the layer depth.  The quantity the scheme conserves exactly
is therefore ``sum_k f_k d_k`` with those divisors ``d_k``
(``scheme_depths``); on a stretched grid the physical column content
``sum_k f_k dz_k`` plus the outflow is not conserved.  ``column_content``
takes either set of depths so the difference can be measured.  On a grid
whose layers deepen by a constant ratio r, every level below the top has
layer depth over divisor ``2/(1+r)``, so the physical column is kept by
moves between those levels and the non-closure sits at the two ends: the
ground (the column loses ``2/(1+r)`` of what leaves it, so ``1 - 2/(1+r)``
of the outflow, 1/81 for KC-1's r = 1.025, stays behind as new water) and
the top level (whose divisor is the centre distance below it, so water
falling out of it shrinks the physical column).

Milestone 1 is drops only.  The snow and dense-ice paths (the snow and
hail/graupel sedimentation calls and their surface accumulations) are
transcribed, but they refuse with ``FsbmRefusal`` naming milestone 3 unless
every one of their bins is zero; with zero bins they run exactly as WRF
does and change nothing.

Units (CGS, as the scheme works).  ``f`` is the scheme's size distribution
(``FF1R`` and the others, number per gram of particle mass per cm3 of air).
``ffx`` is what ``FALFLUXHUCM_Z`` receives, ``f/rho_cgs`` (:4904).  Fall
speeds are cm/s, pressures dyn/cm2, densities g/cm3, heights cm.  An
outflow or a column content is ``f`` times a length (per gram per cm2): its
water depth is ``10 * (3/RO1BL) * COL * outflow * x^2`` mm summed over bins,
the same conversion WRF applies to ``DT * v * f`` at :5066.

Every operation keeps WRF's order: products left to right as the Fortran
writes them, bins in increasing order, sums sequential (never numpy's
pairwise ``np.sum``).  Inputs are not modified.  Arrays are ``(nz, nbins)``
with level 1 (the surface level) first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpuwm.core.fast_sbm.constants import (
    COL,
    FsbmBehavior,
    FsbmRefusal,
    HAIL_OPT_DEFAULT,
    ICEPROCS_DEFAULT,
    NKR,
)
from gpuwm.core.fast_sbm.state import FsbmDistributions

#: ``FALFLUXHUCM_Z``'s activity threshold: a bin falls when any level holds
#: at least this much (:5351).
FALL_ACTIVE_THRESHOLD = 1.0e-20
#: Added to the fall speed in ``TFALL`` (:5358).
FALL_SPEED_EPSILON = 1.0e-20
#: Initial ``TFALL`` and the bare ``STOP`` bound (:5354, :5360).
TFALL_STOP = 1.0e10
#: The pressure in ``FACTOR_P = DSQRT(1.0D6/PCGS)``: 1000 hPa in dyn/cm2
#: (:4519).
FALL_SPEED_REFERENCE_P_CGS = 1.0e6
#: Largest ``INT`` result a default-integer ``NSUB`` holds.  Above it WRF's
#: ``INT(2.0*DT/TFALL)`` overflows (undefined behaviour); the reference
#: refuses instead.
_NSUB_INT_LIMIT = 2 ** 31 - 1

_MILESTONE_3 = ("milestone 1 is drops only; the snow and dense-ice paths are "
                "milestone 3 (fast-sbm design 6-7)")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FalfluxResult:
    """What ``FALFLUXHUCM_Z`` returns for one column and one species.

    ``ffx``
        ``(nz, nbins)``, the distribution over air density after the fall,
        in the units it was passed (``chem_new/rhocgs``, :4904).
    ``outflow``
        ``(nbins,)``, the bottom-face outflow of each bin integrated over
        its substeps, ``sum_n RHOCGS(1)*VFALL(1)*ffx(1)^n * DTFALL`` with
        ``ffx(1)^n`` the level-1 value at the start of substep n.  Units:
        ``f`` times cm.  Zero for a bin that did not fall.
    ``nsub``
        ``(nbins,)`` int, WRF's ``NSUB`` for each bin; 0 for a bin that did
        not fall (every level below the 1e-20 threshold).
    ``active``
        ``(nbins,)`` bool, the ``IFALL = 1`` bins.
    """

    ffx: np.ndarray
    outflow: np.ndarray
    nsub: np.ndarray
    active: np.ndarray


@dataclass(frozen=True)
class ColumnFall:
    """The sedimentation stage of one column (:4891-4962).

    ``dist`` holds the distributions after the fall; the aerosol ``fccn``
    and the species that did not fall are passed through unchanged.
    ``drops``, ``snow`` and ``dense`` are the ``FALFLUXHUCM_Z`` results of
    each call; ``snow`` and ``dense`` are ``None`` when ``iceprocs = 0``,
    as WRF skips those calls (:4915).
    """

    dist: FsbmDistributions
    drops: FalfluxResult
    snow: FalfluxResult | None
    dense: FalfluxResult | None


@dataclass(frozen=True)
class SurfacePrecip:
    """Surface precipitation of one column after one call, mm.

    The ``*nc`` fields are the accumulated totals (the caller's value plus
    this call's), the ``*ncv`` fields this call's increments, as WRF's
    RAINNC/RAINNCV, SNOWNC/SNOWNCV and GRAUPELNC/GRAUPELNCV (:5058-5121).
    """

    rainnc: float
    rainncv: float
    snownc: float
    snowncv: float
    graupelnc: float
    graupelncv: float


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def _finite(array: np.ndarray, name: str, where: str) -> None:
    if not np.all(np.isfinite(array)):
        raise FsbmRefusal(where, f"{name} holds a non-finite value")


def _column(values, name: str, where: str, nz: int | None = None) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    if out.ndim != 1:
        raise FsbmRefusal(where, f"{name} has shape {out.shape}, expected (nz,)")
    if nz is not None and out.shape[0] != nz:
        raise FsbmRefusal(where, f"{name} has {out.shape[0]} levels, expected {nz}")
    _finite(out, name, where)
    return out


def _levels_bins(values, name: str, where: str,
                 shape: tuple[int, int] | None = None) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    if out.ndim != 2:
        raise FsbmRefusal(where, f"{name} has shape {out.shape}, expected "
                          "(nz, nbins)")
    if shape is not None and out.shape != shape:
        raise FsbmRefusal(where, f"{name} has shape {out.shape}, expected {shape}")
    return out


def _check_geometry(rho: np.ndarray, zcgs: np.ndarray, where: str) -> None:
    if zcgs.shape[0] < 2:
        raise FsbmRefusal(where, "at least two levels are needed: the top "
                          "level's divisor is ZCGS(KTE)-ZCGS(KTE-1) (:5373)")
    if not np.all(rho > 0.0):
        raise FsbmRefusal(where, "rho_cgs must be positive at every level: "
                          "it divides the flux (:5368)")
    if not np.all(np.diff(zcgs) > 0.0):
        raise FsbmRefusal(where, "zcgs must increase with the level: "
                          "ZCGS(K+1)-ZCGS(K) divides the flux (:5368-5369)")
    if not zcgs[0] > 0.0:
        raise FsbmRefusal(where, "zcgs(1) must be positive: it is the level "
                          "height TFALL divides (:5358)")


def _check_dt(dt, where: str) -> float:
    value = float(dt)
    if not np.isfinite(value) or value <= 0.0:
        raise FsbmRefusal(where, f"dt must be positive and finite, got {dt!r}")
    return value


# ---------------------------------------------------------------------------
# Geometry and fall speeds (:4269-4276, :4518-4531)
# ---------------------------------------------------------------------------

def p_cgs_from_si(p_pa) -> np.ndarray:
    """``PCGS = P_PHY*10.``, Pa to dyn/cm2 as WRF writes it (:4271)."""
    return np.asarray(p_pa, dtype=np.float64) * 10.0


def level_heights_cgs(dz8w_m) -> np.ndarray:
    """``zcgs``, each level's centre height above the surface in cm.

    :4269-4276: ``z_full = 0.``, then per level
    ``zcgs = z_full+0.5*dz8w*100`` and ``z_full = z_full+dz8w*100.``, with
    ``dz8w`` the layer depth in m.  The distance between two level centres is
    therefore half the sum of the two layer depths, which is what
    ``FALFLUXHUCM_Z`` divides by (D8).
    """
    where = "level_heights_cgs"
    dz = _column(dz8w_m, "dz8w_m", where)
    if not np.all(dz > 0.0):
        raise FsbmRefusal(where, "dz8w_m must be positive at every level")
    out = np.empty_like(dz)
    z_full = 0.0
    for k in range(dz.shape[0]):
        out[k] = z_full + 0.5 * dz[k] * 100
        z_full = z_full + dz[k] * 100.0
    return out


def scheme_depths(zcgs) -> np.ndarray:
    """The divisors ``FALFLUXHUCM_Z`` uses per level, cm (D8).

    ``ZCGS(K+1)-ZCGS(K)`` for every level below the top (:5368-5369) and
    ``ZCGS(KTE)-ZCGS(KTE-1)`` at the top (:5373).  Weighted by these depths,
    a column's content plus its outflow is what the scheme conserves.
    """
    z = _column(zcgs, "zcgs", "scheme_depths")
    if z.shape[0] < 2:
        raise FsbmRefusal("scheme_depths", "at least two levels are needed")
    out = np.empty_like(z)
    out[:-1] = z[1:] - z[:-1]
    out[-1] = z[-1] - z[-2]
    return out


def pressure_factor(p_cgs) -> np.ndarray:
    """``FACTOR_P = DSQRT(1.0D6/PCGS)`` per level (:4519)."""
    p = _column(p_cgs, "p_cgs", "pressure_factor")
    if not np.all(p > 0.0):
        raise FsbmRefusal("pressure_factor", "p_cgs must be positive")
    return np.sqrt(FALL_SPEED_REFERENCE_P_CGS / p)


def corrected_velocities(vr, p_cgs) -> np.ndarray:
    """Table fall speeds corrected to each level's pressure, ``(nz, nbins)``.

    :4518-4531: ``FACTOR_P = DSQRT(1.0D6/PCGS(I,K,J))`` and
    ``VR1_Z3D(1:nkr,I,K,J) = VR1(1:nkr)*FACTOR_P`` (likewise ``VR3``,
    ``VR4``, ``VR5``).  ``vr`` is one table's ``(nbins,)`` speeds at
    1000 hPa in cm/s (``tables.vr1`` for drops, ``vr3`` snow, ``vr5`` hail,
    ``vr4`` graupel); ``p_cgs`` the column's pressures in dyn/cm2.
    """
    table = np.asarray(vr, dtype=np.float64)
    if table.ndim != 1:
        raise FsbmRefusal("corrected_velocities",
                          f"vr has shape {table.shape}, expected (nbins,)")
    factor = pressure_factor(p_cgs)
    return table[None, :] * factor[:, None]


def surface_fall_speeds(level1_speeds, behavior: FsbmBehavior) -> np.ndarray:
    """The level-1 fall speeds each column's surface precipitation uses (D1).

    ``level1_speeds`` is ``(ncol, nbins)``: each column's own corrected
    level-1 speeds (row ``c`` is ``corrected_velocities(vr, p_cgs_c)[0]``),
    with the columns in WRF's loop order, j outer and i inner, so the last
    row is the tile's last column.  ``behavior.rainnc_velocity`` selects
    ``"tile_last"``, WRF as written (every column gets the last row, because
    ``VR1_Z(KRR,1)`` read at :5064 is what the last point of the loop at
    :4514-4523 left there), or ``"own_column"``.  The same holds for
    ``VR3_Z`` (:5075), ``VR5_Z`` (:5093) and ``VR4_Z`` (:5107).  It only
    matters for the level-1 form; the outflow form carries each column's own
    speeds from ``falflux``.
    """
    speeds = _levels_bins(level1_speeds, "level1_speeds", "surface_fall_speeds")
    if speeds.shape[0] < 1:
        raise FsbmRefusal("surface_fall_speeds", "no columns")
    if behavior.rainnc_velocity == "tile_last":
        return np.repeat(speeds[-1:, :], speeds.shape[0], axis=0)
    return speeds


# ---------------------------------------------------------------------------
# FALFLUXHUCM_Z (:5324-5382)
# ---------------------------------------------------------------------------

def falflux_substep_count(v_bin, zcgs, dt) -> tuple[float, int]:
    """``TFALL`` and ``NSUB`` of one active bin (:5354-5362).

    ``TFALL = min_k ZCGS(K)/(VFALL(K)+1.E-20)`` from ``1.E10`` (:5354-5359),
    the height of each level centre above the surface over its fall speed,
    then ``NSUB = INT(2.0*DT/TFALL)+1`` (:5361).  WRF stops with a bare
    ``STOP`` when ``TFALL >= 1.E10`` (:5360); the reference refuses.
    """
    where = "FALFLUXHUCM_Z (:5354-5361)"
    v = _column(v_bin, "v_bin", where)
    z = _column(zcgs, "zcgs", where, v.shape[0])
    dt = _check_dt(dt, where)
    if np.any(v < 0.0):
        raise FsbmRefusal(where, "fall speeds must be non-negative (:5346)")
    tfall = TFALL_STOP
    for k in range(v.shape[0]):
        tfall = min(tfall, z[k] / (v[k] + FALL_SPEED_EPSILON))
    if tfall >= TFALL_STOP:
        raise FsbmRefusal(
            "FALFLUXHUCM_Z (:5360)",
            f"TFALL = {tfall!r} >= 1e10 for a bin holding at least 1e-20: "
            "its fall speeds are zero or negligible, and WRF stops with a "
            "bare STOP (exit status 0, no message)")
    ratio = 2.0 * dt / tfall
    if not ratio < _NSUB_INT_LIMIT:
        raise FsbmRefusal(
            "FALFLUXHUCM_Z (:5361)",
            f"INT(2.0*DT/TFALL) = INT({ratio!r}) overflows WRF's default "
            "integer NSUB")
    return tfall, int(ratio) + 1


def falflux_bin(ffx_bin, v_bin, rho_cgs, zcgs, dt, nsub: int
                ) -> tuple[np.ndarray, float]:
    """The ``NSUB`` substeps of one bin (:5362-5377), and its outflow.

    ``DTFALL = DT/NSUB`` (:5362).  Each substep forms every level's
    ``DWFLUX`` from the values at its start, the upstream flux divergence
    ``-(RHOCGS(K)*VFALL(K)*F(K) - RHOCGS(K+1)*VFALL(K+1)*F(K+1)) /
    (RHOCGS(K)*(ZCGS(K+1)-ZCGS(K)))`` below the top (:5365-5370) and
    ``-(RHOCGS(KTE)*VFALL(KTE)*F(KTE))/(RHOCGS(KTE)*(ZCGS(KTE)-ZCGS(KTE-1)))``
    at the top (:5372-5373), then adds ``DWFLUX*DTFALL`` (:5374-5376).
    The outflow adds ``RHOCGS(1)*VFALL(1)*F(1)`` times ``DTFALL`` at the start
    of every substep: the flux WRF's own level-1 divergence removes through
    the ground (the comment at :5340-5345).

    ``nsub`` is taken as given, so a test can show that WRF's count, and no
    other, reproduces the oracle.  Returns ``(ffx_bin, outflow)``.
    """
    where = "FALFLUXHUCM_Z (:5362-5377)"
    f = _column(ffx_bin, "ffx_bin", where)
    nz = f.shape[0]
    v = _column(v_bin, "v_bin", where, nz)
    rho = _column(rho_cgs, "rho_cgs", where, nz)
    z = _column(zcgs, "zcgs", where, nz)
    _check_geometry(rho, z, where)
    dt = _check_dt(dt, where)
    if isinstance(nsub, bool) or not isinstance(nsub, (int, np.integer)) or nsub < 1:
        raise FsbmRefusal(where, f"nsub must be a positive int, got {nsub!r}")
    dtfall = dt / int(nsub)
    rho_v = rho * v
    den = np.empty(nz)
    den[:-1] = rho[:-1] * (z[1:] - z[:-1])
    den[-1] = rho[-1] * (z[-1] - z[-2])
    dwflux = np.empty(nz)
    outflow = 0.0
    for _ in range(int(nsub)):
        flux = rho_v * f
        dwflux[:-1] = -(flux[:-1] - flux[1:]) / den[:-1]
        dwflux[-1] = -(flux[-1]) / den[-1]
        outflow = outflow + flux[0] * dtfall
        f = f + dwflux * dtfall
    return f, outflow


def falflux(ffx, v, rho_cgs, zcgs, dt) -> FalfluxResult:
    """``FALFLUXHUCM_Z`` for one column and one species (:5324-5382).

    ``ffx`` ``(nz, nbins)`` is the distribution over air density as FAST_SBM
    passes it (``chem_new/rhocgs``, :4904); ``v`` ``(nz, nbins)`` the
    pressure-corrected fall speeds (``VRX``, :4900; ``VFALL = VR1(K,KR)`` at
    :5357, the pressure effect being already applied); ``rho_cgs`` and
    ``zcgs`` ``(nz,)``; ``dt`` the model step in s.

    Each bin is independent (:5348-5379).  A bin falls when any level holds
    at least 1e-20 (:5349-5352); it then takes ``NSUB`` substeps
    (``falflux_substep_count``, ``falflux_bin``).  Fall speeds must be
    non-negative: the scheme is upstream with the flux taken from above
    (the comment at :5346).  A falling bin whose ``TFALL`` reaches 1e10 is
    WRF's bare ``STOP`` (:5360) and refuses.
    """
    where = "FALFLUXHUCM_Z (:5324-5382)"
    f = _levels_bins(ffx, "ffx", where)
    nz, nbins = f.shape
    speeds = _levels_bins(v, "v", where, (nz, nbins))
    rho = _column(rho_cgs, "rho_cgs", where, nz)
    z = _column(zcgs, "zcgs", where, nz)
    _check_geometry(rho, z, where)
    dt = _check_dt(dt, where)
    _finite(f, "ffx", where)
    _finite(speeds, "v", where)
    if np.any(speeds < 0.0):
        raise FsbmRefusal(where, "fall speeds must be non-negative: the "
                          "upstream form takes the flux from the level above "
                          "(:5346)")

    out = f.copy()
    outflow = np.zeros(nbins)
    nsub = np.zeros(nbins, dtype=np.int64)
    active = np.zeros(nbins, dtype=bool)
    for kr in range(nbins):
        column = f[:, kr]
        if not np.any(column >= FALL_ACTIVE_THRESHOLD):
            continue
        active[kr] = True
        _, n = falflux_substep_count(speeds[:, kr], z, dt)
        out[:, kr], outflow[kr] = falflux_bin(column, speeds[:, kr], rho, z,
                                              dt, n)
        nsub[kr] = n
    return FalfluxResult(ffx=out, outflow=outflow, nsub=nsub, active=active)


# ---------------------------------------------------------------------------
# The sedimentation stage of FAST_SBM (:4891-4962)
# ---------------------------------------------------------------------------

def fall_species(f, v, rho_cgs, zcgs, dt) -> tuple[np.ndarray, FalfluxResult]:
    """One species of one column through ``FALFLUXHUCM_Z`` as FAST_SBM calls it.

    :4896-4914 for drops (and :4917-4934, :4937-4958 for the ice species):
    ``ffx_z = chem_new/rhocgs`` (:4904), the call (:4907), then
    ``chem_new = ffx_z*rhocgs`` (:4912).  ``f`` is ``(nz, nbins)`` in the
    scheme's units (``chem_new`` at that point holds the distributions).
    Returns the distribution after the fall and the ``FalfluxResult``.
    """
    where = "fall_species (:4896-4914)"
    dist = _levels_bins(f, "f", where)
    rho = _column(rho_cgs, "rho_cgs", where, dist.shape[0])
    ffx = dist / rho[:, None]
    result = falflux(ffx, v, rho, zcgs, dt)
    return result.ffx * rho[:, None], result


def _refuse_nonzero(array: np.ndarray, name: str, where: str) -> None:
    if np.any(array != 0.0):
        raise FsbmRefusal(where, f"{name} holds non-zero bins: {_MILESTONE_3}")


def sediment_column(dist: FsbmDistributions, tables, p_cgs, rho_cgs, zcgs, dt,
                    *, hail_opt: int = HAIL_OPT_DEFAULT,
                    iceprocs: int = ICEPROCS_DEFAULT) -> ColumnFall:
    """The sedimentation stage of FAST_SBM for one column (:4891-4962).

    Drops fall with ``VR1`` corrected to each level's pressure (:4900); with
    ``iceprocs = 1`` snow follows with ``VR3`` (:4921) and the dense-ice
    class with ``VR5`` for hail (``hail_opt = 1``, :4942) or ``VR4`` for
    graupel (:4944), in that order.  The snow and dense-ice calls refuse as
    milestone 3 unless every bin is zero; with zero bins they run and change
    nothing, as in WRF.  ``tables`` is a float64 ``FsbmTables``.
    """
    where = "sediment_column (:4891-4962)"
    if hail_opt not in (0, 1) or isinstance(hail_opt, bool):
        raise FsbmRefusal(where, f"hail_opt must be 0 or 1, got {hail_opt!r}")
    if iceprocs not in (0, 1) or isinstance(iceprocs, bool):
        raise FsbmRefusal(where, f"iceprocs must be 0 or 1, got {iceprocs!r}")
    nz = np.shape(dist.ff1r)[0] if np.ndim(dist.ff1r) == 2 else -1
    shape = (nz, NKR)
    ff1r = _levels_bins(dist.ff1r, "ff1r", where, shape)
    ff3r = _levels_bins(dist.ff3r, "ff3r", where, shape)
    ff4r = _levels_bins(dist.ff4r, "ff4r", where, shape)
    ff5r = _levels_bins(dist.ff5r, "ff5r", where, shape)
    fccn = np.array(dist.fccn, dtype=np.float64, copy=True)

    drops_after, drops = fall_species(
        ff1r, corrected_velocities(tables.vr1, p_cgs), rho_cgs, zcgs, dt)
    snow = dense = None
    if iceprocs == 1:
        dense_name = "ff5r (hail)" if hail_opt == 1 else "ff4r (graupel)"
        dense_in = ff5r if hail_opt == 1 else ff4r
        _refuse_nonzero(ff3r, "ff3r (snow)", where)
        _refuse_nonzero(dense_in, dense_name, where)
        ff3r, snow = fall_species(
            ff3r, corrected_velocities(tables.vr3, p_cgs), rho_cgs, zcgs, dt)
        dense_vr = tables.vr5 if hail_opt == 1 else tables.vr4
        dense_after, dense = fall_species(
            dense_in, corrected_velocities(dense_vr, p_cgs), rho_cgs, zcgs, dt)
        if hail_opt == 1:
            ff5r = dense_after
        else:
            ff4r = dense_after
    after = FsbmDistributions(ff1r=drops_after, ff3r=ff3r, ff4r=ff4r,
                              ff5r=ff5r, fccn=fccn)
    return ColumnFall(dist=after, drops=drops, snow=snow, dense=dense)


# ---------------------------------------------------------------------------
# Surface precipitation (:5056-5121)
# ---------------------------------------------------------------------------

def _bins(values, name: str, where: str, nbins: int = NKR) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    if out.shape != (nbins,):
        raise FsbmRefusal(where, f"{name} has shape {out.shape}, expected ({nbins},)")
    _finite(out, name, where)
    return out


def surface_precip_level1(ff_bottom, v_bottom, x, ro1bl, col, dt, *,
                          start: float = 0.0) -> float:
    """WRF's surface precipitation of one species, mm (:5058-5071).

    ``start + sum_kr 10.0*(3./RO1BL(KR))*COL*DT*v(KR)*F(1,KR)*X(KR)*X(KR)``,
    accumulated bin by bin into ``start`` as WRF accumulates into RAINNC
    (:5065-5067) and, from 0, into RAINNCV (:5068-5070).  ``ff_bottom`` is
    the level-1 distribution after sedimentation, ``v_bottom`` the level-1
    fall speeds WRF reads (``surface_fall_speeds``), ``x`` the bin masses in
    g (``XL`` for drops), ``ro1bl`` the drop bulk density (WRF divides every
    species by ``RO1BL``, which gives liquid-equivalent depth), ``col`` the
    grid constant ``COL`` and ``dt`` the step in s.
    """
    where = "surface_precip_level1 (:5058-5071)"
    ff = _bins(ff_bottom, "ff_bottom", where)
    v = _bins(v_bottom, "v_bottom", where)
    xs = _bins(x, "x", where)
    ro = _bins(ro1bl, "ro1bl", where)
    col = float(col)
    dt = _check_dt(dt, where)
    acc = float(start)
    for kr in range(NKR):
        acc = acc + 10.0 * (3.0 / ro[kr]) * col * dt * v[kr] * ff[kr] * xs[kr] * xs[kr]
    return acc


def surface_precip_outflow(outflow, x, ro1bl, col, *, start: float = 0.0) -> float:
    """Surface precipitation from the bottom-face outflow, mm (D2 corrected).

    ``start + sum_kr 10.0*(3./RO1BL(KR))*COL*OUTFLOW(KR)*X(KR)*X(KR)``: WRF's
    conversion (:5066) with ``DT*v*F(1)`` replaced by the flux that actually
    left level 1 over the substeps (``FalfluxResult.outflow``).  Applied to
    a column content (``column_content``) it gives that column's water in
    mm.
    """
    where = "surface_precip_outflow"
    out = _bins(outflow, "outflow", where)
    xs = _bins(x, "x", where)
    ro = _bins(ro1bl, "ro1bl", where)
    col = float(col)
    acc = float(start)
    for kr in range(NKR):
        acc = acc + 10.0 * (3.0 / ro[kr]) * col * out[kr] * xs[kr] * xs[kr]
    return acc


def _ice_terms(values, name: str, where: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = _bins(values, name, where)
    _refuse_nonzero(arr, name, where)
    return arr


def _ice_speeds(values, name: str, where: str) -> np.ndarray:
    if values is None:
        raise FsbmRefusal(where, f"{name} is needed with the matching level-1 "
                          "bins: WRF's term multiplies them (:5075, :5093, :5107)")
    return _bins(values, name, where)


def _dense_grid(tables, hail_opt: int, where: str) -> np.ndarray:
    if hail_opt not in (0, 1) or isinstance(hail_opt, bool):
        raise FsbmRefusal(where, f"hail_opt must be 0 or 1, got {hail_opt!r}")
    return np.asarray(tables.xh if hail_opt == 1 else tables.xg, dtype=np.float64)


def surface_precip_wrf_block(drops_level1, v_drops, tables, dt, *,
                             snow_level1=None, v_snow=None,
                             dense_level1=None, v_dense=None,
                             hail_opt: int = HAIL_OPT_DEFAULT,
                             rainnc: float = 0.0, snownc: float = 0.0,
                             graupelnc: float = 0.0) -> SurfacePrecip:
    """The whole surface block of one column as WRF writes it (:5056-5121).

    RAINNCV, SNOWNCV and GRAUPELNCV start at 0 (:5058-5060).  Drops add
    each bin's term to RAINNC and RAINNCV (:5061-5071).  Snow adds its terms
    (``XS``, ``VR3_Z``, still over ``RO1BL``) to RAINNC, RAINNCV, SNOWNC and
    SNOWNCV (:5072-5088); the dense-ice class adds its terms (``XH`` and
    ``VR5_Z`` for hail, ``XG`` and ``VR4_Z`` for graupel) to RAINNC, RAINNCV,
    GRAUPELNC and GRAUPELNCV (:5089-5121).  Each accumulator receives the
    terms one at a time, in that order.

    The level-1 arrays are the distributions after sedimentation; the speed
    arrays the level-1 speeds from ``surface_fall_speeds`` (D1).  Snow and
    dense ice refuse as milestone 3 unless zero; ``None`` stands for zero
    bins, which add exactly nothing.
    """
    where = "surface_precip_wrf_block (:5056-5121)"
    dense_x = _dense_grid(tables, hail_opt, where)
    snow = _ice_terms(snow_level1, "snow_level1", where)
    dense = _ice_terms(dense_level1, "dense_level1", where)
    ro1bl = tables.ro1bl
    rainncv = surface_precip_level1(drops_level1, v_drops, tables.xl, ro1bl, COL, dt)
    rainnc = surface_precip_level1(drops_level1, v_drops, tables.xl, ro1bl, COL, dt,
                                   start=rainnc)
    snowncv = 0.0
    graupelncv = 0.0
    if snow is not None:
        v = _ice_speeds(v_snow, "v_snow", where)
        rainnc = surface_precip_level1(snow, v, tables.xs, ro1bl, COL, dt, start=rainnc)
        rainncv = surface_precip_level1(snow, v, tables.xs, ro1bl, COL, dt, start=rainncv)
        snownc = surface_precip_level1(snow, v, tables.xs, ro1bl, COL, dt, start=snownc)
        snowncv = surface_precip_level1(snow, v, tables.xs, ro1bl, COL, dt, start=snowncv)
    if dense is not None:
        v = _ice_speeds(v_dense, "v_dense", where)
        rainnc = surface_precip_level1(dense, v, dense_x, ro1bl, COL, dt, start=rainnc)
        rainncv = surface_precip_level1(dense, v, dense_x, ro1bl, COL, dt, start=rainncv)
        graupelnc = surface_precip_level1(dense, v, dense_x, ro1bl, COL, dt,
                                          start=graupelnc)
        graupelncv = surface_precip_level1(dense, v, dense_x, ro1bl, COL, dt,
                                           start=graupelncv)
    return SurfacePrecip(rainnc=float(rainnc), rainncv=float(rainncv),
                         snownc=float(snownc), snowncv=float(snowncv),
                         graupelnc=float(graupelnc), graupelncv=float(graupelncv))


def surface_precip_outflow_block(drops_outflow, tables, *, snow_outflow=None,
                                 dense_outflow=None,
                                 hail_opt: int = HAIL_OPT_DEFAULT,
                                 rainnc: float = 0.0, snownc: float = 0.0,
                                 graupelnc: float = 0.0) -> SurfacePrecip:
    """The surface block with each species' bottom-face outflow (D2 corrected).

    The same accumulators and order as ``surface_precip_wrf_block``, with
    ``surface_precip_outflow`` in place of the level-1 form, so surface
    precipitation equals what the scheme removed from the column.  Snow and
    dense ice refuse as milestone 3 unless zero.
    """
    where = "surface_precip_outflow_block"
    dense_x = _dense_grid(tables, hail_opt, where)
    snow = _ice_terms(snow_outflow, "snow_outflow", where)
    dense = _ice_terms(dense_outflow, "dense_outflow", where)
    ro1bl = tables.ro1bl
    rainncv = surface_precip_outflow(drops_outflow, tables.xl, ro1bl, COL)
    rainnc = surface_precip_outflow(drops_outflow, tables.xl, ro1bl, COL, start=rainnc)
    snowncv = 0.0
    graupelncv = 0.0
    if snow is not None:
        rainnc = surface_precip_outflow(snow, tables.xs, ro1bl, COL, start=rainnc)
        rainncv = surface_precip_outflow(snow, tables.xs, ro1bl, COL, start=rainncv)
        snownc = surface_precip_outflow(snow, tables.xs, ro1bl, COL, start=snownc)
        snowncv = surface_precip_outflow(snow, tables.xs, ro1bl, COL, start=snowncv)
    if dense is not None:
        rainnc = surface_precip_outflow(dense, dense_x, ro1bl, COL, start=rainnc)
        rainncv = surface_precip_outflow(dense, dense_x, ro1bl, COL, start=rainncv)
        graupelnc = surface_precip_outflow(dense, dense_x, ro1bl, COL, start=graupelnc)
        graupelncv = surface_precip_outflow(dense, dense_x, ro1bl, COL, start=graupelncv)
    return SurfacePrecip(rainnc=float(rainnc), rainncv=float(rainncv),
                         snownc=float(snownc), snowncv=float(snowncv),
                         graupelnc=float(graupelnc), graupelncv=float(graupelncv))


def surface_precip_column(behavior: FsbmBehavior, fall: ColumnFall, v_surface,
                          tables, dt, *, v_surface_snow=None,
                          v_surface_dense=None,
                          hail_opt: int = HAIL_OPT_DEFAULT,
                          rainnc: float = 0.0, snownc: float = 0.0,
                          graupelnc: float = 0.0) -> SurfacePrecip:
    """Surface precipitation of one column in the form ``behavior`` selects.

    ``behavior.rainnc_source == "level1_after_fall"`` is WRF's form (D2 as
    written): the level-1 distributions of ``fall.dist`` times ``v_surface``,
    this column's row of ``surface_fall_speeds`` for drops, so D1 follows
    ``behavior.rainnc_velocity``.  ``"bottom_outflow"`` uses the outflows in
    ``fall``.  The ice species enter when ``fall`` carries them
    (``iceprocs = 1``) and, for the level-1 form, when their level-1 speeds
    are passed; they refuse as milestone 3 unless zero.  ``sediment_column``
    has already refused non-zero ice, and zero ice adds exactly nothing, so
    leaving the ice speeds out changes no result in milestone 1.
    """
    dist = fall.dist
    dense_level1 = dist.ff5r[0] if hail_opt == 1 else dist.ff4r[0]
    if behavior.rainnc_source == "level1_after_fall":
        with_snow = fall.snow is not None and v_surface_snow is not None
        with_dense = fall.dense is not None and v_surface_dense is not None
        return surface_precip_wrf_block(
            dist.ff1r[0], v_surface, tables, dt,
            snow_level1=dist.ff3r[0] if with_snow else None,
            v_snow=v_surface_snow if with_snow else None,
            dense_level1=dense_level1 if with_dense else None,
            v_dense=v_surface_dense if with_dense else None,
            hail_opt=hail_opt, rainnc=rainnc, snownc=snownc,
            graupelnc=graupelnc)
    return surface_precip_outflow_block(
        fall.drops.outflow, tables,
        snow_outflow=None if fall.snow is None else fall.snow.outflow,
        dense_outflow=None if fall.dense is None else fall.dense.outflow,
        hail_opt=hail_opt, rainnc=rainnc, snownc=snownc, graupelnc=graupelnc)


# ---------------------------------------------------------------------------
# Budgets (D2, D8)
# ---------------------------------------------------------------------------

def column_content(f, depths) -> np.ndarray:
    """Each bin's column content ``sum_k f(k) * depth(k)``, ``(nbins,)``.

    ``f`` ``(nz, nbins)`` in the scheme's units, ``depths`` ``(nz,)`` in cm:
    ``scheme_depths(zcgs)`` for what ``FALFLUXHUCM_Z`` conserves, or the
    layer depths ``dz8w*100`` for the physical column (D8).  Summed level by
    level from the surface.  ``surface_precip_outflow`` turns the result into
    mm of water.
    """
    where = "column_content"
    dist = _levels_bins(f, "f", where)
    d = _column(depths, "depths", where, dist.shape[0])
    acc = np.zeros(dist.shape[1])
    for k in range(dist.shape[0]):
        acc = acc + dist[k] * d[k]
    return acc


__all__ = [
    "ColumnFall",
    "FALL_ACTIVE_THRESHOLD",
    "FALL_SPEED_EPSILON",
    "FALL_SPEED_REFERENCE_P_CGS",
    "FalfluxResult",
    "SurfacePrecip",
    "TFALL_STOP",
    "column_content",
    "corrected_velocities",
    "fall_species",
    "falflux",
    "falflux_bin",
    "falflux_substep_count",
    "level_heights_cgs",
    "p_cgs_from_si",
    "pressure_factor",
    "scheme_depths",
    "sediment_column",
    "surface_fall_speeds",
    "surface_precip_column",
    "surface_precip_level1",
    "surface_precip_outflow",
    "surface_precip_outflow_block",
    "surface_precip_wrf_block",
]
