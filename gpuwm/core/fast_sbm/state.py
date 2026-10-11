"""Bin state of the fast SBM and its unit layer, float64 on the host.

Transcribed from WRF v4.8.0 ``phys/module_mp_fast_sbm.F`` (LF SHA-256
``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``,
identical to v4.6.1), routine ``FAST_SBM``:

* :4214-4262, the entry conversion of the advected bins into size
  distributions (``bins_to_distributions``), with the routing of the dense-ice
  bins to ``FF5R`` (hail) or ``FF4R`` (graupel) and the per-point zeroing of
  negative bins from :4532-4577;
* :4974-5055, the bulk diagnostics (``bulk_diagnostics``);
* :5276-5320, the exit conversion back to advected bins with its 1e-13 clip
  (``distributions_to_bins``);
* :3872-3873, the bin scalar layout of ``chem_new`` (``to_chem_new``,
  ``from_chem_new``);

and from ``FAST_HUCMINIT``, :6095-6101, ``DROPRADII`` (``drop_radii``).
WRF is NCAR/UCAR public domain; the notice is in ``NOTICE`` and
``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.  The scheme is FSBM-2 by Jacob
Shpund and co-authors (see the package docstring).

Units.  The advected bins (``FsbmColumnBins``) are what WRF carries between
calls: per-bin mass mixing ratio in kg/kg (numerically g/g) for drops
(``ff1``), snow (``ff5``) and the dense-ice class (``ff6``), and aerosol
(``ff8``) as WRF stores it, number per kg per unit of ln radius.  The
distributions (``FsbmDistributions``) are the scheme-internal ``f_k`` in CGS:
number per gram of particle mass per cm3 of air for the hydrometeors, so that
bin number is ``3 COL x_k f_k`` and bin mass density ``3 COL x_k^2 f_k``, and
cm-3 per unit ln radius for the aerosol.  ``rho_cgs`` is dry air density in
g/cm3, which WRF forms as ``rho_phy*0.001`` (:4219, :4272, :5280;
``rho_cgs_from_si``).

Every operation keeps WRF's order: the entry conversion is
``((((q*rho)/COL)/x)/x)/3.0``, the exit ``((((f/rho)*COL)*x)*x)*3.0``, and the
bulk sums accumulate bin by bin in increasing bin order, never through a
pairwise reduction.  Arrays are ``(nz, nbins)`` with the vertical index first;
a column's levels are independent in all of these operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpuwm.core.fast_sbm.constants import (
    COL,
    FsbmRefusal,
    ICEPROCS_DEFAULT,
    HAIL_OPT_DEFAULT,
    KRDROP,
    KRICE,
    N_BIN_SCALARS,
    NKR,
    NKR_AEROSOL,
    P_FF1I01,
    P_FF1I33,
    P_FF5I01,
    P_FF5I33,
    P_FF6I01,
    P_FF6I33,
    P_FF8I01,
    P_FF8I43,
)

#: WRF's rounded pi in ``DROPRADII`` (:6100).
_PI_DROPRADII = 3.141593
#: The exit clip threshold, kg/kg (:5285, :5291, :5305, :5312).
CLIP_DEFAULT = 1e-13


@dataclass
class FsbmColumnBins:
    """The advected bin scalars of one column (``chem_new`` slices).

    ``ff1`` drops, ``ff5`` snow, ``ff6`` hail or graupel: ``(nz, 33)``,
    per-bin mass mixing ratio.  ``ff8`` aerosol: ``(nz, 43)``.
    """

    ff1: np.ndarray
    ff5: np.ndarray
    ff6: np.ndarray
    ff8: np.ndarray


@dataclass
class FsbmDistributions:
    """Scheme-internal size distributions, CGS.

    ``ff1r`` drops, ``ff3r`` snow, ``ff4r`` graupel, ``ff5r`` hail:
    ``(nz, 33)``.  ``fccn`` aerosol: ``(nz, 43)``.  With ``hail_opt = 1`` the
    dense-ice bins are hail and ``ff4r`` is zero; with ``hail_opt = 0`` they
    are graupel and ``ff5r`` is zero (:4560-4577).
    """

    ff1r: np.ndarray
    ff3r: np.ndarray
    ff4r: np.ndarray
    ff5r: np.ndarray
    fccn: np.ndarray


@dataclass
class FsbmBulk:
    """Bulk diagnostics of :4974-5055, each ``(nz,)``.

    Mass mixing ratios in kg/kg: ``qc`` (drop bins 1-14), ``qr`` (15-33),
    ``qi`` (snow bins 1-18), ``qs`` (19-33), ``qg`` (the dense-ice class).
    Numbers in #/kg: ``qnc``, ``qnr``, ``qni``, ``qns``, ``qng`` and ``qna``
    (aerosol).
    """

    qc: np.ndarray
    qr: np.ndarray
    qi: np.ndarray
    qs: np.ndarray
    qg: np.ndarray
    qnc: np.ndarray
    qnr: np.ndarray
    qni: np.ndarray
    qns: np.ndarray
    qng: np.ndarray
    qna: np.ndarray


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_hail_opt(hail_opt: int, where: str) -> None:
    if hail_opt not in (0, 1) or isinstance(hail_opt, bool):
        raise FsbmRefusal(where, f"hail_opt must be 0 (graupel) or 1 (hail), "
                          f"got {hail_opt!r}")


def _as_2d(array, width: int, name: str, where: str) -> np.ndarray:
    out = np.asarray(array, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != width:
        raise FsbmRefusal(where, f"{name} has shape {out.shape}, expected "
                          f"(nz, {width})")
    return out


def _as_rho(rho_cgs, nz: int, where: str) -> np.ndarray:
    rho = np.asarray(rho_cgs, dtype=np.float64)
    if rho.shape != (nz,):
        raise FsbmRefusal(where, f"rho_cgs has shape {rho.shape}, expected "
                          f"({nz},)")
    if not np.all(rho > 0.0):
        raise FsbmRefusal(where, "rho_cgs must be positive at every level")
    return rho


def _mass_grid(tables, name: str) -> np.ndarray:
    return np.asarray(getattr(tables, name), dtype=np.float64)


def rho_cgs_from_si(rho_phy) -> np.ndarray:
    """``RHOCGS = rho_phy*0.001``, kg/m3 to g/cm3 as WRF writes it (:4219)."""
    return np.asarray(rho_phy, dtype=np.float64) * 0.001


# ---------------------------------------------------------------------------
# Entry: advected bins to distributions (:4214-4262, :4532-4577)
# ---------------------------------------------------------------------------

def bins_to_distributions(bins: FsbmColumnBins, rho_cgs, tables,
                          hail_opt: int = HAIL_OPT_DEFAULT, *,
                          zero_negative: bool = False) -> FsbmDistributions:
    """The entry conversion of :4214-4262, in WRF's operation order.

    Drops ``FF1 rho / COL / XL / XL / 3``, snow the same with ``XS``, the
    dense-ice bins with ``XH`` when ``hail_opt = 1`` and ``XG`` otherwise,
    aerosol ``FF8 rho / 1000``.  The dense-ice spectrum goes to ``ff5r``
    (hail) or ``ff4r`` (graupel) and the other is zero, as the per-point loop
    does at :4560-4577.

    ``zero_negative`` applies the per-point loop's zeroing of negative bins
    (:4532-4577), which WRF does on its local copies at the start of every
    point; the entry conversion itself leaves negative values in place.
    """
    where = "bins_to_distributions"
    _check_hail_opt(hail_opt, where)
    ff1 = _as_2d(bins.ff1, NKR, "ff1", where)
    nz = ff1.shape[0]
    ff5 = _as_2d(bins.ff5, NKR, "ff5", where)
    ff6 = _as_2d(bins.ff6, NKR, "ff6", where)
    ff8 = _as_2d(bins.ff8, NKR_AEROSOL, "ff8", where)
    for name, array in (("ff5", ff5), ("ff6", ff6), ("ff8", ff8)):
        if array.shape[0] != nz:
            raise FsbmRefusal(where, f"{name} has {array.shape[0]} levels, "
                              f"ff1 has {nz}")
    rho = _as_rho(rho_cgs, nz, where)[:, None]
    xl = _mass_grid(tables, "xl")
    xs = _mass_grid(tables, "xs")
    dense = _mass_grid(tables, "xh" if hail_opt == 1 else "xg")

    ff1r = ff1 * rho / COL / xl / xl / 3.0
    ff3r = ff5 * rho / COL / xs / xs / 3.0
    fccn = ff8 * rho / 1000.0
    dense_r = ff6 * rho / COL / dense / dense / 3.0
    zeros = np.zeros_like(dense_r)
    if hail_opt == 1:
        ff4r, ff5r = zeros, dense_r
    else:
        ff4r, ff5r = dense_r, zeros
    dist = FsbmDistributions(ff1r=ff1r, ff3r=ff3r, ff4r=ff4r, ff5r=ff5r,
                             fccn=fccn)
    if zero_negative:
        for name in ("ff1r", "ff3r", "ff4r", "ff5r", "fccn"):
            array = getattr(dist, name)
            array[array < 0.0] = 0.0
    return dist


# ---------------------------------------------------------------------------
# Bulk diagnostics (:4974-5055)
# ---------------------------------------------------------------------------

def bulk_diagnostics(dist: FsbmDistributions, rho_cgs, tables,
                     hail_opt: int = HAIL_OPT_DEFAULT, *,
                     iceprocs: int = ICEPROCS_DEFAULT) -> FsbmBulk:
    """QC, QR, QI, QS, QG, their numbers and QNA, as :4974-5055 sums them.

    Drop bins 1-14 (below ``KRDROP``) are cloud, 15-33 rain.  Snow bins 1-18
    (up to ``KRICE``) are reported as cloud ice, 19-33 as snow.  The dense
    ice uses ``XH`` and the hail spectrum when ``hail_opt = 1``, ``XG`` and
    the graupel spectrum otherwise.  With ``iceprocs = 0`` the ice
    diagnostics stay zero, as WRF's ``ICEPROCS == 1`` guard leaves them.
    Numbers are ``COL f x 3 / rho * 1000``, #/kg.  Each sum runs bin by bin
    in increasing order, as the Fortran loop does.
    """
    where = "bulk_diagnostics"
    _check_hail_opt(hail_opt, where)
    if iceprocs not in (0, 1) or isinstance(iceprocs, bool):
        raise FsbmRefusal(where, f"iceprocs must be 0 or 1, got {iceprocs!r}")
    ff1r = _as_2d(dist.ff1r, NKR, "ff1r", where)
    nz = ff1r.shape[0]
    ff3r = _as_2d(dist.ff3r, NKR, "ff3r", where)
    dense_r = _as_2d(dist.ff5r if hail_opt == 1 else dist.ff4r, NKR,
                     "ff5r" if hail_opt == 1 else "ff4r", where)
    fccn = _as_2d(dist.fccn, NKR_AEROSOL, "fccn", where)
    rho = _as_rho(rho_cgs, nz, where)
    xl = _mass_grid(tables, "xl")
    xs = _mass_grid(tables, "xs")
    dense = _mass_grid(tables, "xh" if hail_opt == 1 else "xg")

    qc = np.zeros(nz)
    qr = np.zeros(nz)
    qi = np.zeros(nz)
    qs = np.zeros(nz)
    qg = np.zeros(nz)
    qnc = np.zeros(nz)
    qnr = np.zeros(nz)
    qni = np.zeros(nz)
    qns = np.zeros(nz)
    qng = np.zeros(nz)
    qna = np.zeros(nz)

    # Drops (:4988-5003); KRR is one-based in WRF, kr zero-based here.
    for kr in range(NKR):
        f = ff1r[:, kr]
        mass = (1.0 / rho) * COL * f * xl[kr] * xl[kr] * 3.0
        number = COL * f * xl[kr] * 3.0 / rho * 1000.0
        if kr + 1 < KRDROP:
            qc = qc + mass
            qnc = qnc + number
        else:
            qr = qr + mass
            qnr = qnr + number

    if iceprocs == 1:
        # Snow (:5006-5022)
        for kr in range(NKR):
            f = ff3r[:, kr]
            mass = (1.0 / rho) * COL * f * xs[kr] * xs[kr] * 3.0
            number = COL * f * xs[kr] * 3.0 / rho * 1000.0
            if kr + 1 <= KRICE:
                qi = qi + mass
                qni = qni + number
            else:
                qs = qs + mass
                qns = qns + number
        # Hail or graupel (:5024-5040)
        for kr in range(NKR):
            f = dense_r[:, kr]
            qg = qg + (1.0 / rho) * COL * f * dense[kr] * dense[kr] * 3.0
            qng = qng + COL * f * dense[kr] * 3.0 / rho * 1000.0

    # Aerosol (:5043-5048)
    for kr in range(NKR_AEROSOL):
        qna = qna + COL * fccn[:, kr] / rho * 1000.0

    return FsbmBulk(qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, qnc=qnc, qnr=qnr,
                    qni=qni, qns=qns, qng=qng, qna=qna)


# ---------------------------------------------------------------------------
# Exit: distributions to advected bins (:5276-5320)
# ---------------------------------------------------------------------------

def distributions_to_bins(dist: FsbmDistributions, rho_cgs, tables,
                          bulk: FsbmBulk, hail_opt: int = HAIL_OPT_DEFAULT,
                          clip: float | None = CLIP_DEFAULT
                          ) -> FsbmColumnBins:
    """The exit conversion of :5276-5320 with WRF's 1e-13 clip.

    Each bin becomes ``f / rho * COL * x * x * 3``, aerosol
    ``fccn / rho * 1000``.  The clip is keyed as WRF keys it, on the bulk
    diagnostics of the same call: every drop bin of a level is zeroed where
    ``qc + qr < clip`` (:5285), every snow bin where ``qs < clip`` (:5291,
    so the small snow reported as QI does not count), every dense-ice bin
    where ``qg < clip`` (:5305, :5312); aerosol is never clipped.
    ``clip=None`` converts without clipping.
    """
    where = "distributions_to_bins"
    _check_hail_opt(hail_opt, where)
    ff1r = _as_2d(dist.ff1r, NKR, "ff1r", where)
    nz = ff1r.shape[0]
    ff3r = _as_2d(dist.ff3r, NKR, "ff3r", where)
    dense_r = _as_2d(dist.ff5r if hail_opt == 1 else dist.ff4r, NKR,
                     "ff5r" if hail_opt == 1 else "ff4r", where)
    fccn = _as_2d(dist.fccn, NKR_AEROSOL, "fccn", where)
    rho = _as_rho(rho_cgs, nz, where)[:, None]
    xl = _mass_grid(tables, "xl")
    xs = _mass_grid(tables, "xs")
    dense = _mass_grid(tables, "xh" if hail_opt == 1 else "xg")

    ff1 = ff1r / rho * COL * xl * xl * 3.0
    ff5 = ff3r / rho * COL * xs * xs * 3.0
    ff8 = fccn / rho * 1000.0
    ff6 = dense_r / rho * COL * dense * dense * 3.0

    if clip is not None:
        for name in ("qc", "qr", "qs", "qg"):
            if np.shape(getattr(bulk, name)) != (nz,):
                raise FsbmRefusal(where, f"bulk.{name} has shape "
                                  f"{np.shape(getattr(bulk, name))}, "
                                  f"expected ({nz},)")
        qc = np.asarray(bulk.qc, dtype=np.float64)
        qr = np.asarray(bulk.qr, dtype=np.float64)
        ff1[(qc + qr) < clip, :] = 0.0
        ff5[np.asarray(bulk.qs, dtype=np.float64) < clip, :] = 0.0
        ff6[np.asarray(bulk.qg, dtype=np.float64) < clip, :] = 0.0
    return FsbmColumnBins(ff1=ff1, ff5=ff5, ff6=ff6, ff8=ff8)


# ---------------------------------------------------------------------------
# Drop radii (:6095-6101)
# ---------------------------------------------------------------------------

def drop_radii(tables) -> np.ndarray:
    """``DROPRADII = (3 XL / 4 / 3.141593 / 1)**(1/3)``, cm (:6100).

    WRF's own rounded pi and unit density are kept.  ``DEG01 = 1./3.``
    (:6095) is the double nearest one third in the promoted build.
    """
    xl = _mass_grid(tables, "xl")
    return (3.0 * xl / 4.0 / _PI_DROPRADII / 1.0) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# The WRF scalar layout (:3872-3873)
# ---------------------------------------------------------------------------

def _columns(first: int, last: int) -> slice:
    return slice(first - P_FF1I01, last - P_FF1I01 + 1)


#: Column slices of the ``(nz, 142)`` layout; column ``c`` holds WRF scalar
#: index ``c + 2``.
CHEM_SLICES = {
    "ff1": _columns(P_FF1I01, P_FF1I33),
    "ff5": _columns(P_FF5I01, P_FF5I33),
    "ff6": _columns(P_FF6I01, P_FF6I33),
    "ff8": _columns(P_FF8I01, P_FF8I43),
}


def to_chem_new(bins: FsbmColumnBins) -> np.ndarray:
    """Pack the bins into WRF's ``chem_new`` order, ``(nz, 142)``.

    Column ``c`` is WRF's scalar index ``c + 2`` (index 1 is WRF's unused
    first scalar slot): drops 2-34, snow 35-67, dense ice 68-100, aerosol
    101-143.
    """
    where = "to_chem_new"
    ff1 = _as_2d(bins.ff1, NKR, "ff1", where)
    nz = ff1.shape[0]
    out = np.empty((nz, N_BIN_SCALARS), dtype=np.float64)
    for name, columns in CHEM_SLICES.items():
        width = NKR_AEROSOL if name == "ff8" else NKR
        array = _as_2d(getattr(bins, name), width, name, where)
        if array.shape[0] != nz:
            raise FsbmRefusal(where, f"{name} has {array.shape[0]} levels, "
                              f"ff1 has {nz}")
        out[:, columns] = array
    return out


def from_chem_new(chem) -> FsbmColumnBins:
    """Unpack a ``(nz, 142)`` array in WRF's ``chem_new`` order."""
    array = _as_2d(chem, N_BIN_SCALARS, "chem", "from_chem_new")
    return FsbmColumnBins(**{name: array[:, columns].copy()
                             for name, columns in CHEM_SLICES.items()})
