"""ENERGY's own composition, evaluated for every land column at once.

``module_sf_noahmplsm.F:1741-2396``.  :mod:`gpuwm.core.noahmp_energy` holds the
transcription the unmodified-WRF fixtures pin: one CPython frame per column,
Python floats, an explicit ``f32`` rounding call after every operation, and
ordinary ``if`` statements.  Measured with ``cProfile`` at 352 land columns it
is **48% of a Noah-MP land-surface call**, warm and snow alike, and it does not
amortise across columns because there is nothing in it that is shared.

This module is the same arithmetic over the column axis.  It is not a second
port of the physics -- it is the same statements with the array spelling of
each operation, in the same order, and the scalar module remains the authority
that ``tests/test_noahmp_energy_slab.py`` compares every field against.

Four segments, because ENERGY is a composition and the leaves interrupt it
====================================================================
The scalar generator suspends at six physical leaf calls.  Vectorised, the
composition becomes four straight-line segments with the batched leaves
between them -- the same order, the same values, no per-column control flow
left anywhere:

======== ==========================================================
seg1     entry to THERMOPROP: UR, VAI, FSNO, the roughness and
         displacement geometry, the urban override, ZLVL, CWP
seg2     RADIATION's return to VEGE_FLUX: the emissivities, BTRAN,
         the ground surface resistance, the psychrometric constants
seg3     BARE_FLUX's return: the tile average, FIRE, EMISSI, TRAD,
         APAR, PSN, ACC_SSOIL and the soil-column averaging
seg4     PHASECHANGE's return: EFLXB's scaling and the slots WRF
         leaves undefined, which gpuwm defines as zero
======== ==========================================================

Where a branch became a mask, and where it could not
----------------------------------------------------
Every ``if`` on a per-column value is a :func:`cupy.where`.  Two of them need
their guarded expression to stay finite on the columns the mask discards, and
both are written with a substituted denominator rather than left to produce a
NaN that the mask happens to throw away:

* :2068 ``BDSNO = SNEQV/SNOWH`` divides by a snow depth that is exactly zero on
  every snow-free column.
* :2201's ``PSI`` and :2186's ``L_RSURF`` divide by ``SMCMAX``, which is
  non-zero for every admitted soil category, so those are left alone.

The BTRAN loop (:2154-2161) keeps its layer loop.  ``NROOT`` varies per column
and FP32 addition is not associative, so the accumulation runs in layer order
with the out-of-root layers contributing an exact ``+0.0``; only the column
axis inside it is vectorised.  ``-ZSOIL(NROOT)`` is a per-column gather.

``min`` and ``max`` are :func:`gpuwm.core.noahmp_slab_libm.fmn`/``fmx``, not
``cupy.minimum``/``maximum``: the scalar spelling returns the second argument
on a tie and the CuPy builtins do not, which is observable on a signed zero.

What is deliberately refused rather than approximated
------------------------------------------------------
``ICE != 0`` (the glacier ground-emissivity leg at :2145-2147) and ``IST != 1``
(the lake legs at :2076-2082 and :2178-2181) are refused here exactly as the
scalar module refuses them, and for the same reason: no fixture in this tree
admits either, so there is nothing that could hold a transcription of them at
``max_ulp 0``.  A slab is refused if *any* column carries them, with the
column index in the message.

The emitted-longwave check at :2323-2329 is where WRF calls
``wrf_error_fatal``.  A slab cannot raise for one column and continue for the
rest, so it raises for the slab and names the first offending column, which is
what the scalar path does with the same message.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.noahmp_energy import (CPAIR, GRAV, HSUB, HVAP, MPE, NSNOW,
                                      NSOIL, RW, SB, TFRZ, Z0)
from gpuwm.core.noahmp_slab_libm import (fmn, fmx, slab_expf, slab_powf,
                                         slab_sqrtf, slab_tanhf)


def _f32(value):
    return np.float32(value)


_ZERO = _f32(0.0)
_ONE = _f32(1.0)


def _layer(sequence, k, nsnow: int = NSNOW):
    """Fortran ``seq(k)`` for a slab array declared ``(-nsnow+1:nsoil)``.

    The column axis is first and the layer axis last, so this is a column of
    the slab, not a column of the model.
    """
    return sequence[:, k + nsnow - 1]


def _refuse_unadmitted(ice, ist):
    """``ICE != 0`` and ``IST != 1`` are refused, with the column named."""
    import cupy as cp

    bad_ice = cp.asnumpy(cp.asarray(ice) != 0)
    if bad_ice.any():
        raise NotImplementedError(
            f"land column {int(np.argmax(bad_ice))} has ICE != 0, the glacier "
            "column: it selects the ground-emissivity leg at "
            "module_sf_noahmplsm.F:2145-2147 and, in NOAHMP_SFLX, "
            "NOAHMP_GLACIER.  No fixture in this tree admits it")
    bad_ist = cp.asnumpy(cp.asarray(ist) != 1)
    if bad_ist.any():
        raise NotImplementedError(
            f"land column {int(np.argmax(bad_ist))} has IST != 1, a lake "
            "column: it selects the Z0MG leg at :2076-2082 and the "
            "RSURF/RHSUR leg at :2178-2181.  No fixture in this tree admits it")


# ---------------------------------------------------------------------------
# seg1: entry (:2033) to the THERMOPROP call (:2119)
# ---------------------------------------------------------------------------

def energy_seg1(s):
    """UR, VAI, FSNO, the roughness geometry, the urban override, ZLVL, CWP.

    ``s`` maps the scalar generator's argument names to CuPy arrays over the
    column axis.  The returned dict is what seg2 and the THERMOPROP batch read.
    """
    import cupy as cp

    _refuse_unadmitted(s["ice"], s["ist"])

    uu, vv = s["uu"], s["vv"]
    snowh, sneqv = s["snowh"], s["sneqv"]

    # -- :2057  UR.  The exponent is a REAL literal, so this is powf. -------
    ur = fmx(slab_sqrtf(slab_powf(uu, _f32(2.0)) + slab_powf(vv, _f32(2.0))),
             _ONE)

    # -- :2061-2063  vegetated or not ---------------------------------------
    vai = s["elai"] + s["esai"]
    veg = vai > _ZERO

    # -- :2067-2073  ground snow-cover fraction [Niu and Yang, 2007] ---------
    # The scalar form guards this whole block on SNOWH > 0.  Substituting the
    # denominator keeps the discarded columns finite rather than relying on a
    # NaN being masked away, which is the same value and a smaller surface.
    has_snow = snowh > _ZERO
    safe_snowh = cp.where(has_snow, snowh, _ONE)
    bdsno = sneqv / safe_snowh
    fmelt = slab_powf(bdsno / _f32(100.0), s["mfsno"])
    fsno = cp.where(has_snow,
                    slab_tanhf(safe_snowh / (s["scffac"] * fmelt)),
                    _ZERO)

    # -- :2077-2085  ground roughness length (IST == 1 leg) ------------------
    z0mg = (_f32(Z0) * (_ONE - fsno)) + (fsno * s["z0sno"])

    # -- :2089-2097  roughness length and displacement height ----------------
    zpdg = snowh
    zpd_veg = _f32(0.65) * s["hvt"]
    zpd_veg = cp.where(snowh > zpd_veg, snowh, zpd_veg)
    z0m = cp.where(veg, s["z0mvt"], z0mg)
    zpd = cp.where(veg, zpd_veg, zpdg)

    # -- :2101-2106  urban override -----------------------------------------
    urban = s["urban_flag"]
    urban_zpd = _f32(0.65) * s["hvt"]
    z0mg = cp.where(urban, s["z0mvt"], z0mg)
    zpdg = cp.where(urban, urban_zpd, zpdg)
    z0m = cp.where(urban, z0mg, z0m)
    zpd = cp.where(urban, zpdg, zpd)

    # -- :2109-2110 ----------------------------------------------------------
    zlvl = fmx(zpd, s["hvt"]) + s["zref"]
    # Unreachable for ZREF > 0; transcribed because the source has it.
    zlvl = cp.where(zpdg >= zlvl, zpdg + s["zref"], zlvl)

    return {
        "ur": ur, "vai": vai, "veg": veg, "fsno": fsno,
        "z0mg": z0mg, "zpdg": zpdg, "z0m": z0m, "zpd": zpd,
        "zlvl": zlvl, "cwp": s["cwpvt"],                     # :2115
    }


# ---------------------------------------------------------------------------
# seg2: RADIATION's return (:2139) to the VEGE_FLUX call (:2238)
# ---------------------------------------------------------------------------

def energy_seg2(s, g):
    """The emissivities, BTRAN, RSURF/RHSUR and the psychrometric constants.

    ``g`` is seg1's output; ``s`` is the entry state plus RADIATION's.
    """
    import cupy as cp

    fsno = g["fsno"]

    # -- :2139-2147  emissivities (ICE == 0 leg) -----------------------------
    emv = _ONE - slab_expf(-(s["elai"] + s["esai"]) / _ONE)
    emg = (s["eg"] * (_ONE - fsno)) + (s["snow_emis"] * fsno)

    # -- :2151-2173  BTRAN (OPT_BTR == 1, Noah) ------------------------------
    # The layer loop stays a loop: FP32 addition is not associative and NROOT
    # varies per column, so the accumulation runs in layer order with the
    # out-of-root layers contributing an exact +0.0.
    nroot = s["nroot"]
    sh2o = s["sh2o"]
    smcwlt, smcref = s["smcwlt"], s["smcref"]
    columns = cp.arange(nroot.shape[0])
    # -ZSOIL(NROOT) is a per-column gather, not a fixed slot.
    zsoil_nroot = s["zsoil"][columns, nroot - 1]

    btran = cp.zeros(nroot.shape, dtype=cp.float32)
    btrani = cp.zeros(sh2o.shape, dtype=cp.float32)
    for iz in range(1, NSOIL + 1):
        in_root = nroot >= iz
        gx = ((sh2o[:, iz - 1] - smcwlt[:, iz - 1])
              / (smcref[:, iz - 1] - smcwlt[:, iz - 1]))
        gx = fmn(_ONE, fmx(_ZERO, gx))
        value = fmx(_f32(MPE),
                    (_layer(s["dzsnso"], iz) / -zsoil_nroot) * gx)
        value = cp.where(in_root, value, _ZERO)
        btrani[:, iz - 1] = value
        btran = btran + value
    btran = fmx(_f32(MPE), btran)
    for iz in range(1, NSOIL + 1):
        btrani[:, iz - 1] = cp.where(nroot >= iz,
                                     btrani[:, iz - 1] / btran,
                                     btrani[:, iz - 1])

    # -- :2177-2206  ground surface resistance (IST == 1, OPT_RSF == 1) ------
    smcmax0 = s["smcmax"][:, 0]
    sh2o0 = sh2o[:, 0]
    sat = fmn(_ONE, sh2o0 / smcmax0)
    l_rsurf = ((-s["zsoil"][:, 0])
               * (slab_expf(slab_powf(_ONE - sat, s["rsurf_exp"])) - _ONE)) \
        / (_f32(2.71828) - _ONE)
    d_rsurf = ((_f32(2.2e-5) * smcmax0) * smcmax0) \
        * slab_powf(_ONE - (s["smcwlt"][:, 0] / smcmax0),
                    _f32(2.0) + (_f32(3.0) / s["bexp"][:, 0]))
    rsurf = l_rsurf / d_rsurf
    # :2199
    rsurf = cp.where((sh2o0 < _f32(0.01)) & (s["snowh"] == _ZERO),
                     _f32(1.0e6), rsurf)
    psi = (-s["psisat"][:, 0]) \
        * slab_powf(fmx(_f32(0.01), sh2o0) / smcmax0, -s["bexp"][:, 0])
    rhsur = fsno + ((_ONE - fsno)
                    * slab_expf((psi * _f32(GRAV)) / (_f32(RW) * s["tg"])))
    # :2204-2206
    rsurf = cp.where(s["urban_flag"] & (s["snowh"] == _ZERO),
                     _f32(1.0e6), rsurf)

    # -- :2211-2227  psychrometric constants ---------------------------------
    frozen_canopy = ~(s["tv"] > _f32(TFRZ))
    latheav = cp.where(frozen_canopy, _f32(HSUB), _f32(HVAP))
    gammav = (_f32(CPAIR) * s["sfcprs"]) / (_f32(0.622) * latheav)
    frozen_ground = ~(s["tg"] > _f32(TFRZ))
    latheag = cp.where(frozen_ground, _f32(HSUB), _f32(HVAP))
    gammag = (_f32(CPAIR) * s["sfcprs"]) / (_f32(0.622) * latheag)

    # -- :2238  which columns run VEGE_FLUX at all ---------------------------
    tile_veg = g["veg"] & (s["fveg"] > _ZERO)

    return {
        "emv": emv, "emg": emg, "btran": btran, "btrani": btrani,
        "rsurf": rsurf, "rhsur": rhsur,
        "gammav": gammav, "gammag": gammag,
        "latheav": latheav, "latheag": latheag,
        "frozen_canopy": frozen_canopy, "frozen_ground": frozen_ground,
        "tile_veg": tile_veg,
    }


# ---------------------------------------------------------------------------
# seg3: BARE_FLUX's return (:2282) to the TSNOSOI call (:2349)
# ---------------------------------------------------------------------------

def energy_seg3(s, g, e, v, b):
    """The tile average, FIRE, EMISSI, TRAD, APAR, PSN and ACC_SSOIL.

    ``v`` is VEGE_FLUX's answer broadcast over every column -- the vegetated
    columns carry its result and the rest carry the zeros :2033-2053 wrote,
    which is exactly what the scalar path holds at this point.  ``b`` is
    BARE_FLUX's, which every land column has.
    """
    import cupy as cp

    tile_veg = e["tile_veg"]
    fveg = s["fveg"]
    one_m = _ONE - fveg

    def blend(veg_term, bare_term, extra=None):
        mixed = (fveg * veg_term) + (one_m * bare_term)
        if extra is not None:
            mixed = mixed + extra
        return cp.where(tile_veg, mixed, bare_term)

    taux = blend(v["tauxv"], b["tauxb"])
    tauy = blend(v["tauyv"], b["tauyb"])
    fira = cp.where(tile_veg,
                    ((fveg * v["irg"]) + (one_m * b["irb"])) + v["irc"],
                    b["irb"])
    fsh = cp.where(tile_veg,
                   ((fveg * v["shg"]) + (one_m * b["shb"])) + v["shc"],
                   b["shb"])
    fgev = blend(v["evg"], b["evb"])
    ssoil = blend(v["ghv"], b["ghb"])
    fcev = cp.where(tile_veg, v["evc"], _ZERO)
    fctr = cp.where(tile_veg, v["tr"], _ZERO)
    pah = cp.where(tile_veg,
                   ((fveg * s["pahg"]) + (one_m * s["pahb"])) + s["pahv"],
                   s["pahb"])
    tg = blend(v["tgv"], b["tgb"])
    t2m = blend(v["t2mv"], b["t2mb"])
    ts = cp.where(tile_veg, (fveg * v["tv"]) + (one_m * b["tgb"]), tg)
    cm = blend(v["cmv"], b["cmb"])
    ch = blend(v["chv"], b["chb"])
    q1 = cp.where(
        tile_veg,
        (fveg * ((v["eah"] * _f32(0.622))
                 / (s["sfcprs"] - (_f32(0.378) * v["eah"]))))
        + (one_m * b["qsfc"]),
        b["qsfc"])
    q2e = blend(v["q2v"], b["q2b"])
    z0wrf = cp.where(tile_veg, g["z0m"], g["z0mg"])
    # :2306-2307 and the bare arm's resets
    rssun = cp.where(tile_veg, v["rssun"], _ZERO)
    rssha = cp.where(tile_veg, v["rssha"], _ZERO)
    tgv = cp.where(tile_veg, v["tgv"], b["tgb"])
    chv = cp.where(tile_veg, v["chv"], b["chb"])

    # -- :2321-2329  the emitted-longwave sanity check -----------------------
    fire = s["lwdn"] + fira
    bad = cp.asnumpy(fire <= _ZERO)
    if bad.any():
        raise ValueError(
            f"land column {int(np.argmax(bad))}: emitted longwave <= 0: "
            "SHDFAC is inconsistent with LAI "
            "(module_sf_noahmplsm.F:2323-2329 calls wrf_error_fatal here)")

    # -- :2332-2340 ----------------------------------------------------------
    emv, emg = e["emv"], e["emg"]
    emissi = (fveg * (((emg * (_ONE - emv)) + emv)
                      + ((emv * (_ONE - emv)) * (_ONE - emg)))) \
        + ((_ONE - fveg) * emg)
    trad = slab_powf((fire - ((_ONE - emissi) * s["lwdn"]))
                     / (emissi * _f32(SB)), _f32(0.25))

    apar = (s["parsun"] * s["laisun"]) + (s["parsha"] * s["laisha"])   # :2344
    psn = (v["psnsun"] * s["laisun"]) + (v["psnsha"] * s["laisha"])    # :2345

    # -- :2349-2353  the soil column ----------------------------------------
    acc_ssoil = s["acc_ssoil"] + ssoil
    ssoil_avg = acc_ssoil / _f32(s["soil_update_steps"])
    dt_soil = s["dt"] * _f32(s["soil_update_steps"])

    return {
        "taux": taux, "tauy": tauy, "fira": fira, "fsh": fsh, "fgev": fgev,
        "ssoil": ssoil, "fcev": fcev, "fctr": fctr, "pah": pah, "tg": tg,
        "t2m": t2m, "ts": ts, "cm": cm, "ch": ch, "q1": q1, "q2e": q2e,
        "z0wrf": z0wrf, "rssun": rssun, "rssha": rssha, "tgv": tgv,
        "chv": chv, "emissi": emissi, "trad": trad, "apar": apar, "psn": psn,
        "acc_ssoil": acc_ssoil, "ssoil_avg": ssoil_avg, "dt_soil": dt_soil,
    }


# ---------------------------------------------------------------------------
# seg4: PHASECHANGE's return to ENERGY's return
# ---------------------------------------------------------------------------

def energy_seg4(s, g, e, out):
    """EFLXB's scaling and the slots WRF leaves undefined.

    WRF does not initialise the layers above ISNOW; gpuwm produces the defined
    0.0 rather than reproducing an uninitialised read, which is the standing
    rule for this port.  The scalar module does it with per-layer tuple
    comprehensions and this does it with one comparison against the layer
    ladder.
    """
    import cupy as cp

    isnow = s["isnow"]
    nlay = NSNOW + NSOIL
    # Fortran layer index of each slot: slot k is layer k - NSNOW + 1.
    layer = cp.arange(-NSNOW + 1, NSOIL + 1, dtype=cp.int32)[None, :]
    live_full = layer > isnow[:, None]
    live_snow = live_full[:, :NSNOW]

    eflxb = out["eflxb"] * e["dt_soil"]

    hcpct = cp.where(live_full, out["hcpct"], _ZERO)
    imelt = cp.where(live_full, out["imelt"], 0)
    snicev = cp.where(live_snow, out["snicev"], _ZERO)
    snliqv = cp.where(live_snow, out["snliqv"], _ZERO)
    epore = cp.where(live_snow, out["epore"], _ZERO)

    root = cp.arange(1, NSOIL + 1, dtype=cp.int32)[None, :]
    btrani = cp.where(root <= s["nroot"][:, None], e["btrani"], _ZERO)

    night = s["cosz"] <= _ZERO
    fsrv = cp.where(night, _ZERO, out["fsrv"])
    fsrg = cp.where(night, _ZERO, out["fsrg"])
    bgap = cp.where(night, _ZERO, out["bgap"])
    wgap = cp.where(night, _ZERO, out["wgap"])

    assert hcpct.shape[1] == nlay
    return {
        "eflxb": eflxb, "hcpct": hcpct, "imelt": imelt, "snicev": snicev,
        "snliqv": snliqv, "epore": epore, "btrani": btrani,
        "fsrv": fsrv, "fsrg": fsrg, "bgap": bgap, "wgap": wgap,
    }


__all__ = ["energy_seg1", "energy_seg2", "energy_seg3", "energy_seg4"]
