"""TAUMOL: WRF v4.6.1 RRTM longwave band optical depths (TAUGB1--TAUGB16).

Direct transcription of ``phys/module_ra_rrtm.F:4488-6191``.  The Fortran
loops one layer at a time inside one column; this module keeps the same
arithmetic but runs it over an ``(ncol, nlay)`` block with the 140
g-points on the trailing axis, so the whole band evaluates as array
operations on either NumPy or CuPy.

Two structural notes about that reshaping.

*Branch selection.*  Every ``TAUGBn`` splits at ``DO LAY = 1, LAYTROP`` /
``DO LAY = LAYTROP+1, NLAYERS`` (band 8 splits at ``LAYSWTCH``).  Here
both branches are evaluated for every layer and selected by the mask
``lay <= laytrop``, which is exactly the Fortran's partition because
``LAYTROP`` is a *count* of layers (SETCOEF F:4291-4295) and pressure
decreases with layer index.  Reference-table subscripts of the branch
that is *not* selected can fall outside the table (e.g. ``JP-13`` is
negative in the troposphere), so every gather clamps its index into
range first; the clamped value is always discarded by the select.  This
is the one place the port deliberately differs from a naive
transliteration, and it changes no selected result.

*Band 9's IOFF.*  ``IOFF`` is initialised once before the layer loop and
assigned at ``LAY == LAYLOW`` / ``LAY == LAYSWTCH`` without ever being
reset (F:5646-5647), so it is sticky for all higher layers.
:func:`_band9_ioff` reproduces that running state exactly, including the
evaluation order that makes ``LAYSWTCH`` win a tie.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.rrtm_tables import (RRTM_MG, RRTM_ONEMINUS,
                                    load_rrtm_lw_tables)


RRTM_NGPT = 140

# module_ra_rrtm.F NGS<n> parameters declared inside each TAUGB<n>: the
# zero-based offset of band n's g-points within the 140-point vector.
_NGS = (0, 8, 22, 38, 52, 68, 76, 88, 96, 108, 114, 122, 130, 134, 136, 138)
# module_ra_rrtm.F:9-24, NG1..NG16 (== NGC after reduction).
_NG = (8, 14, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2)

_DEVICE_CACHE: dict[tuple[int, str], dict] = {}


def _array_namespace(*values):
    """Return CuPy only when at least one input is a CUDA array."""
    if any(hasattr(value, "__cuda_array_interface__") for value in values):
        import cupy as cp
        return cp
    return np


def _device_tables(tables, xp):
    """Move the combined tables onto ``xp`` once per process."""
    key = (id(tables), xp.__name__)
    cached = _DEVICE_CACHE.get(key)
    if cached is not None:
        return cached
    moved = {
        "absa": {b: xp.asarray(v) for b, v in tables.absa.items()},
        "absb": {b: xp.asarray(v) for b, v in tables.absb.items()},
        "selfrefc": {b: xp.asarray(v) for b, v in tables.selfrefc.items()},
        "combined": {k: xp.asarray(v) for k, v in tables.combined.items()},
        "statics": {k: xp.asarray(v) for k, v in tables.statics.items()},
        "corr1": xp.asarray(tables.corr1),
        "corr2": xp.asarray(tables.corr2),
    }
    _DEVICE_CACHE[key] = moved
    return moved


def _gather(xp, table, index):
    """``table(index, IG)`` for a clamped 0-based ``index`` block."""
    idx = xp.clip(index, 0, table.shape[0] - 1)
    return table[idx]


def _pick(xp, table, index):
    """``table(index)`` for a clamped 0-based 1-D lookup."""
    return table[xp.clip(index, 0, table.shape[0] - 1)]


def _trunc_int(xp, value):
    """Fortran ``INT()``: truncation toward zero."""
    return xp.trunc(value).astype(xp.int32)


def _spec(xp, real, spec_a, spec_b, strrat, nsub):
    """Shared ``SPECCOMB``/``SPECPARM``/``JS``/``FS`` opening block.

    ``nsub`` is 8 for the tropospheric key-species split and 4 for the
    stratospheric one (e.g. F:5077-5081 vs F:5119-5123).
    """
    speccomb = spec_a + real(strrat) * spec_b
    specparm = spec_a / speccomb
    specparm = xp.where(specparm >= real(RRTM_ONEMINUS),
                        real(RRTM_ONEMINUS), specparm)
    specmult = real(nsub) * specparm
    js = 1 + _trunc_int(xp, specmult)
    fs = specmult - xp.floor(specmult)
    return speccomb, js, fs.astype(specparm.dtype)


def _eight_way(xp, real, fac00, fac01, fac10, fac11, fs):
    """The eight ``FACabc`` products every key-species band forms."""
    one = real(1.0)
    return ((one - fs) * fac00, (one - fs) * fac10, fs * fac00, fs * fac10,
            (one - fs) * fac01, (one - fs) * fac11, fs * fac01, fs * fac11)


def _interp8(tab, ind0, ind1, step, facs, xp):
    """``FAC000*T(IND0) + FAC100*T(IND0+1) + ...`` with the band's step."""
    f000, f010, f100, f110, f001, f011, f101, f111 = facs
    return (f000[..., None] * _gather(xp, tab, ind0)
            + f100[..., None] * _gather(xp, tab, ind0 + 1)
            + f010[..., None] * _gather(xp, tab, ind0 + step)
            + f110[..., None] * _gather(xp, tab, ind0 + step + 1)
            + f001[..., None] * _gather(xp, tab, ind1)
            + f101[..., None] * _gather(xp, tab, ind1 + 1)
            + f011[..., None] * _gather(xp, tab, ind1 + step)
            + f111[..., None] * _gather(xp, tab, ind1 + step + 1))


def _interp4(tab, ind0, ind1, fac00, fac01, fac10, fac11, xp):
    """``FAC00*T(IND0) + FAC10*T(IND0+1) + FAC01*T(IND1) + FAC11*...``."""
    return (fac00[..., None] * _gather(xp, tab, ind0)
            + fac10[..., None] * _gather(xp, tab, ind0 + 1)
            + fac01[..., None] * _gather(xp, tab, ind1)
            + fac11[..., None] * _gather(xp, tab, ind1 + 1))


def _self_continuum(xp, selfrefc, inds, selffac, selffrac):
    """``SELFFAC*(SELFREF(INDS) + SELFFRAC*(SELFREF(INDS+1)-SELFREF(INDS)))``."""
    low = _gather(xp, selfrefc, inds)
    high = _gather(xp, selfrefc, inds + 1)
    return selffac[..., None] * (low + selffrac[..., None] * (high - low))


def _frac_interp(xp, table, js, fs):
    """``FRACREF(IG,JS) + FS*(FRACREF(IG,JS+1) - FRACREF(IG,JS))``.

    ``table`` is stored ``(nsub, ng)`` so a ``(ncol,nlay)`` ``js`` gathers
    straight into ``(ncol,nlay,ng)``.
    """
    low = _gather(xp, table, js)
    high = _gather(xp, table, js + 1)
    return low + fs[..., None] * (high - low)


def _eta_denominator(xp, real, eta):
    """``1. - ETAREF(NS)`` with the discarded top interval neutralised.

    ``ETAREF``'s last entry is exactly 1.0 (F:4746, 5588), and the branch
    that would divide by ``1-ETAREF`` there is never the one WRF selects
    -- the ``NS`` test picks the ``H2OREF`` form instead.  Replacing the
    zero keeps the unselected value finite instead of an ignored inf.
    """
    denom = real(1.0) - eta
    return xp.where(denom == real(0.0), real(1.0), denom)


def _ratio(xp, ref_num, ref_den1, ref_den2, jp, fp):
    """``(COLREF1/WCOMB1) + FP*((COLREF2/WCOMB2) - (COLREF1/WCOMB1))``."""
    colref1 = _pick(xp, ref_num, jp - 1)
    colref2 = _pick(xp, ref_num, jp)
    first = colref1 / ref_den1
    second = colref2 / ref_den2
    return first + fp * (second - first)


def _band9_ioff(xp, lay, laylow, layswtch, ng9):
    """Reproduce ``IOFF``'s sticky state across the layer loop (F:5646-5647).

    ``IOFF`` is set at the single layer equal to ``LAYLOW`` (to ``NG9``)
    and at the single layer equal to ``LAYSWTCH`` (to ``2*NG9``), in that
    statement order, and never cleared.  For any layer the surviving
    value is therefore the later-firing of the two events at or below it,
    with ``LAYSWTCH`` winning when both fire on the same layer.
    """
    fired_low = (laylow >= 1) & (lay >= laylow)
    fired_swtch = (layswtch >= 1) & (lay >= layswtch)
    swtch_wins = fired_swtch & (~fired_low | (layswtch >= laylow))
    return xp.where(swtch_wins, 2 * ng9,
                    xp.where(fired_low, ng9, 0)).astype(xp.int32)


def rrtm_taumol(*, coldry, colh2o, colco2, colo3, coln2o, colch4, colo2,
                co2mult, fac00, fac01, fac10, fac11, forfac, selffac,
                selffrac, jp, jt, jt1, indself, wx, laytrop, layswtch,
                laylow, tables=None):
    """Evaluate ``GASABS``'s sixteen band routines for a column block.

    All layer inputs are ``(ncol, nlay)``; ``wx`` is ``(ncol, nlay, 4)``
    (WRF's ``WX(MAXXSEC,LAY)`` cross-section amounts); ``laytrop``,
    ``layswtch`` and ``laylow`` are ``(ncol,)`` layer counts from SETCOEF.
    Returns ``(taug, pfrac)``, both ``(ncol, nlay, 140)``.
    """
    if tables is None:
        tables = load_rrtm_lw_tables()
    xp = _array_namespace(colh2o, fac00, jp)
    dev = _device_tables(tables, xp)
    absa, absb = dev["absa"], dev["absb"]
    selfrefc, cmb, st = dev["selfrefc"], dev["combined"], dev["statics"]
    nspa, nspb = tables.nspa, tables.nspb

    dtype = colh2o.dtype
    if dtype.kind != "f":
        raise TypeError("RRTM column amounts must be floating point")

    def real(value):
        return dtype.type(value)

    ncol, nlay = colh2o.shape
    taug = xp.zeros((ncol, nlay, RRTM_NGPT), dtype=dtype)
    pfrac = xp.zeros((ncol, nlay, RRTM_NGPT), dtype=dtype)

    lay = xp.arange(1, nlay + 1, dtype=xp.int32)[None, :]
    trop = lay <= laytrop[:, None]
    swtch = lay <= layswtch[:, None]
    trop3 = trop[..., None]
    swtch3 = swtch[..., None]

    def band_slice(band):
        start = _NGS[band - 1]
        return slice(start, start + _NG[band - 1])

    def ind_low(band, js):
        """``((JP-1)*5+(JT-1))*NSPA(n) + JS`` as a 0-based index."""
        return ((jp - 1) * 5 + (jt - 1)) * int(nspa[band - 1]) + js - 1

    def ind_low1(band, js):
        """``(JP*5+(JT1-1))*NSPA(n) + JS`` as a 0-based index."""
        return (jp * 5 + (jt1 - 1)) * int(nspa[band - 1]) + js - 1

    def ind_high(band, js, base=13):
        return ((jp - base) * 5 + (jt - 1)) * int(nspb[band - 1]) + js - 1

    def ind_high1(band, js, base=12):
        return ((jp - base) * 5 + (jt1 - 1)) * int(nspb[band - 1]) + js - 1

    inds = indself - 1  # SELFREF's 0-based INDS

    # ---- BAND 1: 10-250 cm-1 (low - H2O; high - H2O)  F:4488-4563 ----
    forrefc1 = cmb["FORREFC1"]
    low = colh2o[..., None] * (
        _interp4(absa[1], ind_low(1, 1), ind_low1(1, 1),
                 fac00, fac01, fac10, fac11, xp)
        + _self_continuum(xp, selfrefc[1], inds, selffac, selffrac)
        + forfac[..., None] * forrefc1)
    high = colh2o[..., None] * (
        _interp4(absb[1], ind_high(1, 1), ind_high1(1, 1),
                 fac00, fac01, fac10, fac11, xp)
        + forfac[..., None] * forrefc1)
    taug[:, :, band_slice(1)] = xp.where(trop3, low, high)
    pfrac[:, :, band_slice(1)] = xp.where(
        trop3, cmb["FRACREFAC1"], cmb["FRACREFBC1"])

    # ---- BAND 2: 250-500 cm-1 (low - H2O; high - H2O)  F:4566-4682 ----
    refparam_host = tables.statics["TAUGB2__REFPARAM"]
    refparam = st["TAUGB2__REFPARAM"]
    water = real(1.0e20) * colh2o / coldry
    h2oparam = water / (water + real(0.002))
    # F:4626-4631: first IFRAC in 2..12 with H2OPARAM >= REFPARAM(IFRAC);
    # a DO loop that completes leaves the index at its limit+1 == 13.
    ifrac = xp.full((ncol, nlay), 13, dtype=xp.int32)
    for candidate in range(12, 1, -1):
        ifrac = xp.where(h2oparam >= real(refparam_host[candidate - 1]),
                         candidate, ifrac)
    frac_lo = _gather(xp, cmb["FRACREFAC2"], ifrac - 1)
    frac_hi = _gather(xp, cmb["FRACREFAC2"], ifrac - 2)
    ref_at = _pick(xp, refparam, ifrac - 1)
    ref_prev = _pick(xp, refparam, ifrac - 2)
    fracint = (h2oparam - ref_at) / (ref_prev - ref_at)
    fp2 = fac11 + fac01
    ifp = _trunc_int(xp, real(2.0e2) * fp2 + real(0.5))
    ifp = xp.where(ifp <= 0, 0, ifp)
    corr1 = _pick(xp, dev["corr1"], ifp)
    corr2 = _pick(xp, dev["corr2"], ifp)
    fc00, fc10 = fac00 * corr2, fac10 * corr2
    fc01, fc11 = fac01 * corr1, fac11 * corr1
    forrefc2 = cmb["FORREFC2"]
    low = colh2o[..., None] * (
        _interp4(absa[2], ind_low(2, 1), ind_low1(2, 1),
                 fc00, fc01, fc10, fc11, xp)
        + _self_continuum(xp, selfrefc[2], inds, selffac, selffrac)
        + forfac[..., None] * forrefc2)
    high = colh2o[..., None] * (
        _interp4(absb[2], ind_high(2, 1), ind_high1(2, 1),
                 fc00, fc01, fc10, fc11, xp)
        + forfac[..., None] * forrefc2)
    taug[:, :, band_slice(2)] = xp.where(trop3, low, high)
    pfrac[:, :, band_slice(2)] = xp.where(
        trop3, frac_lo + fracint[..., None] * (frac_hi - frac_lo),
        cmb["FRACREFBC2"])

    # ---- BAND 3: 500-630 cm-1 (low - H2O,CO2; high - H2O,CO2) F:4685 ----
    strrat3 = 1.19268
    h2oref3 = xp.asarray(st["TAUGB3__H2OREF"])
    co2ref3 = xp.asarray(st["TAUGB3__CO2REF"])
    n2oref3 = xp.asarray(st["TAUGB3__N2OREF"])
    etaref3 = xp.asarray(st["TAUGB3__ETAREF"])
    fp3 = fac01 + fac11
    forrefc3 = cmb["FORREFC3"]

    # lower (F:4779-4835)
    speccomb, js, fs = _spec(xp, real, colh2o, colco2, strrat3, 8.0)
    # F:4786-4793: the JS==8 tail is remapped onto the finer 9th interval.
    js_eq8 = js == 8
    js_l = xp.where(js_eq8 & (fs >= real(0.9)), 9, js)
    fs_l = xp.where(js_eq8,
                    xp.where(fs >= real(0.9), real(10.0) * (fs - real(0.9)),
                             fs / real(0.9)),
                    fs)
    ns = js_l + _trunc_int(xp, fs_l + real(0.5))
    eta = _eta_denominator(xp, real, _pick(xp, etaref3, ns - 1))
    wcomb1 = xp.where(ns == 10, _pick(xp, h2oref3, jp - 1),
                      real(strrat3) * _pick(xp, co2ref3, jp - 1) / eta)
    wcomb2 = xp.where(ns == 10, _pick(xp, h2oref3, jp),
                      real(strrat3) * _pick(xp, co2ref3, jp) / eta)
    n2omult = coln2o - speccomb * _ratio(xp, n2oref3, wcomb1, wcomb2, jp, fp3)
    facs = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs_l)
    low = (speccomb[..., None]
           * _interp8(absa[3], ind_low(3, js_l), ind_low1(3, js_l), 10,
                      facs, xp)
           + colh2o[..., None]
           * (_self_continuum(xp, selfrefc[3], inds, selffac, selffrac)
              + forfac[..., None] * forrefc3)
           + n2omult[..., None] * cmb["ABSN2OAC3"])
    pfrac_low = _frac_interp(xp, cmb["FRACREFAC3"], js_l - 1, fs_l)

    # upper (F:4843-4877)
    speccomb_u, js_u, fs_u = _spec(xp, real, colh2o, colco2, strrat3, 4.0)
    ns_u = js_u + _trunc_int(xp, fs_u + real(0.5))
    eta_u = _eta_denominator(xp, real, _pick(xp, etaref3, ns_u - 1))
    wcomb1 = xp.where(ns_u == 5, _pick(xp, h2oref3, jp - 1),
                      real(strrat3) * _pick(xp, co2ref3, jp - 1) / eta_u)
    wcomb2 = xp.where(ns_u == 5, _pick(xp, h2oref3, jp),
                      real(strrat3) * _pick(xp, co2ref3, jp) / eta_u)
    n2omult_u = coln2o - speccomb_u * _ratio(
        xp, n2oref3, wcomb1, wcomb2, jp, fp3)
    facs_u = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs_u)
    high = (speccomb_u[..., None]
            * _interp8(absb[3], ind_high(3, js_u), ind_high1(3, js_u), 5,
                       facs_u, xp)
            + (colh2o * forfac)[..., None] * forrefc3
            + n2omult_u[..., None] * cmb["ABSN2OBC3"])
    taug[:, :, band_slice(3)] = xp.where(trop3, low, high)
    pfrac[:, :, band_slice(3)] = xp.where(
        trop3, pfrac_low,
        _frac_interp(xp, cmb["FRACREFBC3"], js_u - 1, fs_u))

    # ---- BAND 4: 630-700 cm-1 (low - H2O,CO2; high - O3,CO2) F:4882 ----
    speccomb, js, fs = _spec(xp, real, colh2o, colco2, 850.577, 8.0)
    facs = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs)
    low = (speccomb[..., None]
           * _interp8(absa[4], ind_low(4, js), ind_low1(4, js), 9, facs, xp)
           + colh2o[..., None]
           * _self_continuum(xp, selfrefc[4], inds, selffac, selffrac))
    pfrac_low = _frac_interp(xp, cmb["FRACREFAC4"], js - 1, fs)
    speccomb_u, js_u, fs_u = _spec(xp, real, colo3, colco2, 35.7416, 4.0)
    # F:4967-4975: band 4's stratospheric first interval is split at
    # 0.0024 instead of being uniform.
    js_gt1 = js_u > 1
    fs_ge = fs_u >= real(0.0024)
    js_u2 = xp.where(js_gt1, js_u + 1, xp.where(fs_ge, 2, 1))
    fs_u2 = xp.where(js_gt1, fs_u,
                     xp.where(fs_ge, (fs_u - real(0.0024)) / real(0.9976),
                              fs_u / real(0.0024)))
    facs_u = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs_u2)
    high = speccomb_u[..., None] * _interp8(
        absb[4], ind_high(4, js_u2), ind_high1(4, js_u2), 6, facs_u, xp)
    taug[:, :, band_slice(4)] = xp.where(trop3, low, high)
    pfrac[:, :, band_slice(4)] = xp.where(
        trop3, pfrac_low,
        _frac_interp(xp, cmb["FRACREFBC4"], js_u2 - 1, fs_u2))

    # ---- BAND 5: 700-820 cm-1 (low - H2O,CO2; high - O3,CO2) F:5013 ----
    ccl4c5 = cmb["CCL4C5"]
    wx1 = wx[:, :, 0]
    speccomb, js, fs = _spec(xp, real, colh2o, colco2, 90.4894, 8.0)
    facs = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs)
    low = (speccomb[..., None]
           * _interp8(absa[5], ind_low(5, js), ind_low1(5, js), 9, facs, xp)
           + colh2o[..., None]
           * _self_continuum(xp, selfrefc[5], inds, selffac, selffrac)
           + wx1[..., None] * ccl4c5)
    pfrac_low = _frac_interp(xp, cmb["FRACREFAC5"], js - 1, fs)
    speccomb_u, js_u, fs_u = _spec(xp, real, colo3, colco2, 0.900502, 4.0)
    facs_u = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs_u)
    high = (speccomb_u[..., None]
            * _interp8(absb[5], ind_high(5, js_u), ind_high1(5, js_u), 5,
                       facs_u, xp)
            + wx1[..., None] * ccl4c5)
    taug[:, :, band_slice(5)] = xp.where(trop3, low, high)
    pfrac[:, :, band_slice(5)] = xp.where(
        trop3, pfrac_low,
        _frac_interp(xp, cmb["FRACREFBC5"], js_u - 1, fs_u))

    # ---- BAND 6: 820-980 cm-1 (low - H2O; high - nothing)  F:5140 ----
    wx2, wx3 = wx[:, :, 1], wx[:, :, 2]
    cfc_6 = (wx2[..., None] * cmb["CFC11ADJC6"]
             + wx3[..., None] * cmb["CFC12C6"])
    low = (colh2o[..., None]
           * (_interp4(absa[6], ind_low(6, 1), ind_low1(6, 1),
                       fac00, fac01, fac10, fac11, xp)
              + _self_continuum(xp, selfrefc[6], inds, selffac, selffrac))
           + cfc_6 + co2mult[..., None] * cmb["ABSCO2C6"])
    taug[:, :, band_slice(6)] = xp.where(trop3, low, cfc_6)
    pfrac[:, :, band_slice(6)] = cmb["FRACREFAC6"]

    # ---- BAND 7: 980-1080 cm-1 (low - H2O,O3; high - O3)  F:5218 ----
    absco2c7 = cmb["ABSCO2C7"]
    speccomb, js, fs = _spec(xp, real, colh2o, colo3, 8.21104e4, 8.0)
    facs = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs)
    low = (speccomb[..., None]
           * _interp8(absa[7], ind_low(7, js), ind_low1(7, js), 9, facs, xp)
           + colh2o[..., None]
           * _self_continuum(xp, selfrefc[7], inds, selffac, selffrac)
           + co2mult[..., None] * absco2c7)
    pfrac_low = _frac_interp(xp, cmb["FRACREFAC7"], js - 1, fs)
    high = (colo3[..., None]
            * _interp4(absb[7], ind_high(7, 1), ind_high1(7, 1),
                       fac00, fac01, fac10, fac11, xp)
            + co2mult[..., None] * absco2c7)
    taug[:, :, band_slice(7)] = xp.where(trop3, low, high)
    pfrac[:, :, band_slice(7)] = xp.where(
        trop3, pfrac_low, cmb["FRACREFBC7"])

    # ---- BAND 8: 1080-1180 cm-1 (low - H2O; high - O3)  F:5320 ----
    # This band alone splits at LAYSWTCH (~300 mb), not LAYTROP.
    h2oref8 = xp.asarray(st["TAUGB8__H2OREF"])
    n2oref8 = xp.asarray(st["TAUGB8__N2OREF"])
    o3ref8 = xp.asarray(st["TAUGB8__O3REF"])
    wx4 = wx[:, :, 3]
    fp8 = fac01 + fac11
    cfc_8 = (wx3[..., None] * cmb["CFC12C8"]
             + wx4[..., None] * cmb["CFC22ADJC8"])
    ratio_low = _ratio(xp, n2oref8, _pick(xp, h2oref8, jp - 1),
                       _pick(xp, h2oref8, jp), jp, fp8)
    n2omult_low = coln2o - colh2o * ratio_low
    low = (colh2o[..., None]
           * (_interp4(absa[8], ind_low(8, 1), ind_low1(8, 1),
                       fac00, fac01, fac10, fac11, xp)
              + _self_continuum(xp, selfrefc[8], inds, selffac, selffrac))
           + cfc_8
           + co2mult[..., None] * cmb["ABSCO2AC8"]
           + n2omult_low[..., None] * cmb["ABSN2OAC8"])
    ratio_high = _ratio(xp, n2oref8, _pick(xp, o3ref8, jp - 1),
                        _pick(xp, o3ref8, jp), jp, fp8)
    n2omult_high = coln2o - colo3 * ratio_high
    high = (colo3[..., None]
            * _interp4(absb[8], ind_high(8, 1, base=7),
                       ind_high1(8, 1, base=6),
                       fac00, fac01, fac10, fac11, xp)
            + cfc_8
            + co2mult[..., None] * cmb["ABSCO2BC8"]
            + n2omult_high[..., None] * cmb["ABSN2OBC8"])
    taug[:, :, band_slice(8)] = xp.where(swtch3, low, high)
    pfrac[:, :, band_slice(8)] = xp.where(
        swtch3, cmb["FRACREFAC8"], cmb["FRACREFBC8"])

    # ---- BAND 9: 1180-1390 cm-1 (low - H2O,CH4; high - CH4)  F:5466 ----
    strrat9 = 21.6282
    ng9 = _NG[8]
    h2oref9 = xp.asarray(st["TAUGB9__H2OREF"])
    ch4ref9 = xp.asarray(st["TAUGB9__CH4REF"])
    n2oref9 = xp.asarray(st["TAUGB9__N2OREF"])
    etaref9 = xp.asarray(st["TAUGB9__ETAREF"])
    speccomb, js, fs = _spec(xp, real, colh2o, colch4, strrat9, 8.0)
    jfrac = js
    ffrac = fs
    # F:5602-5616: JS==8's tail splits three ways; JS==9 collapses onto
    # the 10th interval while the Planck fraction stays on the 8th.
    is8, is9 = js == 8, js == 9
    js_l = xp.where(
        is8, xp.where(fs <= real(0.68), js,
                      xp.where(fs <= real(0.92), js + 1, js + 2)),
        xp.where(is9, 10, js))
    fs_l = xp.where(
        is8, xp.where(fs <= real(0.68), fs / real(0.68),
                      xp.where(fs <= real(0.92),
                               (fs - real(0.68)) / real(0.24),
                               (fs - real(0.92)) / real(0.08))),
        xp.where(is9, real(1.0), fs))
    jfrac = xp.where(is9, 8, jfrac)
    ffrac = xp.where(is9, real(1.0), ffrac)
    fp9 = fac01 + fac11
    ns9 = js_l + _trunc_int(xp, fs_l + real(0.5))
    eta9 = _eta_denominator(xp, real, _pick(xp, etaref9, ns9 - 1))
    wcomb1 = xp.where(ns9 == 11, _pick(xp, h2oref9, jp - 1),
                      real(strrat9) * _pick(xp, ch4ref9, jp - 1) / eta9)
    wcomb2 = xp.where(ns9 == 11, _pick(xp, h2oref9, jp),
                      real(strrat9) * _pick(xp, ch4ref9, jp) / eta9)
    n2omult9 = coln2o - speccomb * _ratio(
        xp, n2oref9, wcomb1, wcomb2, jp, fp9)
    ioff = _band9_ioff(xp, lay, laylow[:, None], layswtch[:, None], ng9)
    absn2o_row = _gather(xp, cmb["ABSN2OC9"], ioff // ng9)
    facs = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs_l)
    low = (speccomb[..., None]
           * _interp8(absa[9], ind_low(9, js_l), ind_low1(9, js_l), 11,
                      facs, xp)
           + colh2o[..., None]
           * _self_continuum(xp, selfrefc[9], inds, selffac, selffrac)
           + n2omult9[..., None] * absn2o_row)
    high = colch4[..., None] * _interp4(
        absb[9], ind_high(9, 1), ind_high1(9, 1),
        fac00, fac01, fac10, fac11, xp)
    taug[:, :, band_slice(9)] = xp.where(trop3, low, high)
    pfrac[:, :, band_slice(9)] = xp.where(
        trop3, _frac_interp(xp, cmb["FRACREFAC9"], jfrac - 1, ffrac),
        cmb["FRACREFBC9"])

    # ---- BAND 10: 1390-1480 cm-1 (low - H2O; high - H2O)  F:5620 ----
    low = colh2o[..., None] * _interp4(
        absa[10], ind_low(10, 1), ind_low1(10, 1),
        fac00, fac01, fac10, fac11, xp)
    high = colh2o[..., None] * _interp4(
        absb[10], ind_high(10, 1), ind_high1(10, 1),
        fac00, fac01, fac10, fac11, xp)
    taug[:, :, band_slice(10)] = xp.where(trop3, low, high)
    pfrac[:, :, band_slice(10)] = xp.where(
        trop3, cmb["FRACREFAC10"], cmb["FRACREFBC10"])

    # ---- BAND 11: 1480-1800 cm-1 (low - H2O; high - H2O)  F:5685 ----
    low = colh2o[..., None] * (
        _interp4(absa[11], ind_low(11, 1), ind_low1(11, 1),
                 fac00, fac01, fac10, fac11, xp)
        + _self_continuum(xp, selfrefc[11], inds, selffac, selffrac))
    high = colh2o[..., None] * _interp4(
        absb[11], ind_high(11, 1), ind_high1(11, 1),
        fac00, fac01, fac10, fac11, xp)
    taug[:, :, band_slice(11)] = xp.where(trop3, low, high)
    pfrac[:, :, band_slice(11)] = xp.where(
        trop3, cmb["FRACREFAC11"], cmb["FRACREFBC11"])

    # ---- BAND 12: 1800-2080 cm-1 (low - H2O,CO2; high - nothing) F:5760 --
    speccomb, js, fs = _spec(xp, real, colh2o, colco2, 0.009736757, 8.0)
    facs = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs)
    low = (speccomb[..., None]
           * _interp8(absa[12], ind_low(12, js), ind_low1(12, js), 9,
                      facs, xp)
           + colh2o[..., None]
           * _self_continuum(xp, selfrefc[12], inds, selffac, selffrac))
    zero = real(0.0)
    taug[:, :, band_slice(12)] = xp.where(trop3, low, zero)
    pfrac[:, :, band_slice(12)] = xp.where(
        trop3, _frac_interp(xp, cmb["FRACREFAC12"], js - 1, fs), zero)

    # ---- BAND 13: 2080-2250 cm-1 (low - H2O,N2O; high - nothing) F:5853 --
    speccomb, js, fs = _spec(xp, real, colh2o, coln2o, 16658.87, 8.0)
    facs = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs)
    low = (speccomb[..., None]
           * _interp8(absa[13], ind_low(13, js), ind_low1(13, js), 9,
                      facs, xp)
           + colh2o[..., None]
           * _self_continuum(xp, selfrefc[13], inds, selffac, selffrac))
    taug[:, :, band_slice(13)] = xp.where(trop3, low, zero)
    pfrac[:, :, band_slice(13)] = xp.where(
        trop3, _frac_interp(xp, cmb["FRACREFAC13"], js - 1, fs), zero)

    # ---- BAND 14: 2250-2380 cm-1 (low - CO2; high - CO2)  F:5943 ----
    low = colco2[..., None] * (
        _interp4(absa[14], ind_low(14, 1), ind_low1(14, 1),
                 fac00, fac01, fac10, fac11, xp)
        + _self_continuum(xp, selfrefc[14], inds, selffac, selffrac))
    high = colco2[..., None] * _interp4(
        absb[14], ind_high(14, 1), ind_high1(14, 1),
        fac00, fac01, fac10, fac11, xp)
    taug[:, :, band_slice(14)] = xp.where(trop3, low, high)
    pfrac[:, :, band_slice(14)] = xp.where(
        trop3, cmb["FRACREFAC14"], cmb["FRACREFBC14"])

    # ---- BAND 15: 2380-2600 cm-1 (low - N2O,CO2; high - nothing) F:6015 --
    speccomb, js, fs = _spec(xp, real, coln2o, colco2, 0.2883201, 8.0)
    facs = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs)
    low = (speccomb[..., None]
           * _interp8(absa[15], ind_low(15, js), ind_low1(15, js), 9,
                      facs, xp)
           + colh2o[..., None]
           * _self_continuum(xp, selfrefc[15], inds, selffac, selffrac))
    taug[:, :, band_slice(15)] = xp.where(trop3, low, zero)
    pfrac[:, :, band_slice(15)] = xp.where(
        trop3, _frac_interp(xp, cmb["FRACREFAC15"], js - 1, fs), zero)

    # ---- BAND 16: 2600-3000 cm-1 (low - H2O,CH4; high - nothing) F:6105 --
    speccomb, js, fs = _spec(xp, real, colh2o, colch4, 830.411, 8.0)
    facs = _eight_way(xp, real, fac00, fac01, fac10, fac11, fs)
    low = (speccomb[..., None]
           * _interp8(absa[16], ind_low(16, js), ind_low1(16, js), 9,
                      facs, xp)
           + colh2o[..., None]
           * _self_continuum(xp, selfrefc[16], inds, selffac, selffrac))
    taug[:, :, band_slice(16)] = xp.where(trop3, low, zero)
    pfrac[:, :, band_slice(16)] = xp.where(
        trop3, _frac_interp(xp, cmb["FRACREFAC16"], js - 1, fs), zero)

    return taug, pfrac


__all__ = ["RRTM_NGPT", "rrtm_taumol"]
