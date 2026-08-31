"""Forecast adapter for the exact port of WRF v4.6.1's bundled RRTMG.

``RRTMGLegacyRadiation`` serves ``RunConfig.ra_rrtmg_variant =
"rrtmg_legacy"`` on the resolved 4/4 radiation pair.  It is pure wiring:
every FP statement it performs is either (a) one of the bitwise-gated
building blocks (``rrtmg_legacy_prep`` batch twins, the device McICA
twin, the batched LW/SW CUDA engines, the WRF-level output mappings,
``gpuwm.ingest.wrf_ozone``), (b) the documented driver-side seams that
have no bitwise oracle (``radconst``/``calc_coszen`` FP32 transcription,
phy_prep's ``t8w`` construction, ``cal_cldfra1`` imported from
``gpuwm.core.rrtmgp``), or (c) the WRF radiation driver's own two output
lines (grid-wide GSW/RTHRATENSW zeroing and ``SWDOWN = GSW/(1-ALBEDO)``,
``module_radiation_driver.F:1721,1738,2877``).  The design authority is
``docs/rrtmg_legacy_integration.md``; the booby-trap boundary (section
10) is what tests/test_rrtmg_legacy_wiring.py proves: a legacy-selected
call must run the device McICA twin and the batched CUDA engines, never
the NumPy compute chain.

The implemented option envelope (dossier section 1) is pinned here and
fails closed on anything else: ``icloud=1``, ``cldovrlp=2`` (McICA
maximum-random), ``idcor=0``, ``o3input in {0, 2}``, ``ghg_input=0``,
``aer_opt=0``, ``swint_opt=0``.  ``o3input=0`` selects the wrapper's
O3DATA branch; ``o3input=2`` selects CAM climatology.

Night contract (dossier section 3): the driver-level zeroing is realized
by allocating RTHRATENSW/GSW as zeros and writing only day columns; the
wrapper-level night write (COSZR + the SW_NIGHT_ZEROED list) still runs
through ``swrad_night_outputs_batch`` so the wrapper contract stays
exercised even though none of those diagnostics reach
:class:`RadiationResult`.  WRF zeroes the SWUPT.. optional-diagnostic
bundle only IF PRESENT (module_ra_rrtmg_sw.F:11568-11589); gpuwm's
contract is that the full bundle is ALWAYS materialized (the night
helper returns every member unconditionally).

Radii seam (dossier sections 9.3, 11): gpuwm state ``effc/effi/effs``
carry MICRONS; the wrapper takes meters, so the adapter multiplies by
``F(1e-6)`` exactly once at this boundary.  Meter-scale state (a restart
written before the Thompson radii-units fix) is rejected with a
migration pointer, never silently rescaled.

Ozone nest routing (prep/ozone xhigh audit, 2026-07-27): WRF evaluates
the CAM climatology chain (ozn_time_int/ozn_p_int) ONLY on the root
domain (``o3input==2 .and. id==1``, module_radiation_driver.F:1799-1823)
and hands nests the parent-interpolated ``o3rad``.  The adapter mirrors
that STRUCTURE: a root adapter (``ozone_parent=None``) runs the
bit-ported ``gpuwm.ingest.wrf_ozone`` chain on its own grid and RETAINS
the resulting field; a child adapter takes an ``ozone_parent=`` provider
(:class:`ParentOzoneProvider`, wired by ``runtime.prepare_child_case``)
and obtains the parent's most recent retained field horizontally
interpolated onto the child grid through gpuwm's certified SINT
mass-point operator (``gpuwm.core.nest_interp.sint``) -- the child never
invokes the climatology chain.  The horizontal-interpolation ARITHMETIC
is gpuwm's own SINT transliteration, the same documented seam class as
every other nest-interpolation arithmetic difference; the routing
(root-compute + parent->child interpolation, the child's field updating
when the parent's does) matches WRF.

SW aerosol: the batched CUDA SW engine builds WRF's neutral aer_opt=0
optics (tauaer 0 / ssaaer 1 / asmaer 0, module_ra_rrtmg_sw.F:11333-11460)
internally and REJECTS any other ``aer_opt`` (SW-audit item 1); the
adapter pins ``aer_opt=0`` and passes no aerosol arrays because the
batched entry deliberately has no such parameters.

Sequencing/VRAM shape: the LW and SW pipelines run SEQUENTIALLY per
adapter chunk -- each chunk's prep outputs, device McICA slabs, and
engine transients are dropped before the next stage allocates -- so the
transient device peak is the MAX over the four phases (LW generate, LW
engine, SW generate, SW engine) plus the McICA slabs held as engine
inputs, never the LW+SW sum.  :func:`legacy_radiation_vram_bytes` prices
exactly that shape.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np

from gpuwm.core.mynn_radiation import (
    merge_mynn_bl_clouds,
    mynn_bl_cloud_active,
    wrf_itimestep,
)

from gpuwm.core import rrtmg_legacy_prep as _prep
from gpuwm.core import rrtmg_lw as _lw
from gpuwm.core import rrtmg_mcica as _mcica
from gpuwm.core import rrtmg_sw as _sw

__all__ = [
    "MAX_LONGWAVE_LAYERS", "MAX_SHORTWAVE_LAYERS",
    "ParentOzoneProvider", "RRTMGLegacyRadiation",
    "legacy_radiation_layer_counts", "legacy_radiation_vram_bytes",
    "radconst", "calc_coszen",
]

F = np.float32
MAX_LONGWAVE_LAYERS = _lw.MAX_RADIATION_LAYERS
MAX_SHORTWAVE_LAYERS = _sw.MAX_RADIATION_LAYERS


def legacy_radiation_layer_counts(
        nz: int, p_top: float) -> tuple[int, int]:
    """Return exact first-call legacy RRTMG ``(LW, SW)`` layer counts."""

    return (
        int(_prep.compute_lw_nlayers(int(nz) + 1, float(p_top))),
        int(nz) + 1,
    )

#: WRF share/module_model_constants.F: piconst (default REAL) and the
#: compile-time parameter folds DEGRAD = piconst/180., DPD = 360./365.
_PICONST = F(3.1415926535897932384626433)
_DEGRAD = F(_PICONST / F(180.0))
_DPD = F(F(360.0) / F(365.0))

#: The pinned WRF option combination this adapter implements (dossier
#: section 1).  A config carrying any other explicit value fails closed.
_PINNED_OPTIONS = {"icloud": 1, "cldovrlp": 2, "idcor": 0,
                   "ghg_input": 0, "aer_opt": 0, "swint_opt": 0}


# ---------------------------------------------------------------------------
# Solar geometry: FP32 statement transcription of radconst / calc_coszen
# (module_radiation_driver.F:3469-3541).  numpy transcendentals stand in
# for the compiler libm -- dossier section 9.1 documents why this is a
# seam with no bitwise oracle (ifx/gfortran/numpy differ at ~1 ulp).
# ---------------------------------------------------------------------------

def radconst(julian):
    """(declin, solcon) from WRF ``radconst`` at the FP32 julian."""
    j = F(julian)
    obecl = F(23.5) * _DEGRAD
    sinob = np.sin(obecl)
    if j >= F(80.0):
        sxlong = _DPD * (j - F(80.0))
    else:
        sxlong = _DPD * (j + F(285.0))
    sxlong = sxlong * _DEGRAD
    arg = sinob * np.sin(sxlong)
    declin = np.arcsin(arg)
    djul = (j * F(360.0)) / F(365.0)
    rjul = djul * _DEGRAD
    two = F(2.0) * rjul
    eccfac = F(1.000110) + F(0.034221) * np.cos(rjul)
    eccfac = eccfac + F(0.001280) * np.sin(rjul)
    eccfac = eccfac + F(0.000719) * np.cos(two)
    eccfac = eccfac + F(0.000077) * np.sin(two)
    solcon = F(1370.0) * eccfac
    return F(declin), F(solcon)


def calc_coszen(julian, xtime, gmt, xlat, xlon, declin):
    """WRF ``calc_coszen`` per point; returns float32 coszen (n,).

    ``xtime`` is the minutes value WRF passes (the driver hands
    ``xtime + radt*0.5`` for the interval-midpoint hour angle while
    ``julian``/``declin`` stay at call time; driver lines 1206-1208).
    """
    j = F(julian)
    xt = F(xtime)
    g = F(gmt)
    d = F(declin)
    xlat = np.asarray(xlat, np.float32)
    xlon = np.asarray(xlon, np.float32)
    da = (F(6.2831853071795862) * (j - F(1.0))) / F(365.0)
    two = F(2.0) * da
    eot = F(0.000075) + F(0.001868) * np.cos(da)
    eot = eot - F(0.032077) * np.sin(da)
    eot = eot - F(0.014615) * np.cos(two)
    eot = eot - F(0.04089) * np.sin(two)
    eot = eot * F(229.18)
    xt24 = np.fmod(xt, F(1440.0)) + eot          # Fortran MOD on reals
    tloctm = (g + xt24 / F(60.0)) + xlon / F(15.0)
    hrang = (F(15.0) * (tloctm - F(12.0))) * _DEGRAD
    xxlat = xlat * _DEGRAD
    coszen = (np.sin(xxlat) * np.sin(d)
              + (np.cos(xxlat) * np.cos(d)) * np.cos(hrang))
    return np.minimum(np.maximum(coszen, F(-1.0)), F(1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# t8w: phy_prep transcription (module_big_step_utilities_em.F:4904-4936).
# Interior interfaces use the fnm/fnp half->full weights; surface/top are
# z-linear extrapolations.  Dossier section 9.4: matched semantics seam.
# ---------------------------------------------------------------------------

def _t8w_columns(t3d, z_at_w, fnm, fnp):
    """t8w (ncol, nz+1) from t3d (ncol, nz) and z_at_w (ncol, nz+1)."""
    t3d = np.asarray(t3d, np.float32)
    z_at_w = np.asarray(z_at_w, np.float32)
    fnm = np.asarray(fnm, np.float32)
    fnp = np.asarray(fnp, np.float32)
    ncol, nz = t3d.shape
    # z at half levels: z = 0.5*(z_at_w(k) + z_at_w(k+1))
    z = (F(0.5) * (z_at_w[:, :-1] + z_at_w[:, 1:])).astype(np.float32)
    t8w = np.empty((ncol, nz + 1), np.float32)
    # interior full levels k = 2..kde-1 (1-based): fzm(k)*t(k)+fzp(k)*t(k-1)
    t8w[:, 1:nz] = (fnm[None, 1:nz] * t3d[:, 1:nz]
                    + fnp[None, 1:nz] * t3d[:, 0:nz - 1]).astype(np.float32)
    # bottom: z-linear extrapolation from the two lowest half levels
    w1 = ((z_at_w[:, 0] - z[:, 1]) / (z[:, 0] - z[:, 1])).astype(np.float32)
    w2 = (F(1.0) - w1).astype(np.float32)
    t8w[:, 0] = (w1 * t3d[:, 0] + w2 * t3d[:, 1]).astype(np.float32)
    # top: z-linear extrapolation from the two highest half levels
    w1 = ((z_at_w[:, nz] - z[:, nz - 2])
          / (z[:, nz - 1] - z[:, nz - 2])).astype(np.float32)
    w2 = (F(1.0) - w1).astype(np.float32)
    t8w[:, nz] = (w1 * t3d[:, nz - 1] + w2 * t3d[:, nz - 2]).astype(
        np.float32)
    return t8w


# ---------------------------------------------------------------------------
# VRAM pricing (model preflight + the wiring honesty gate).
# ---------------------------------------------------------------------------

#: WRF v4.6.1 use_mp_re scheme table, FIRST BLOCK: the
#: ``module_physics_init.F:988-1024`` disjunction, which sets
#: ``has_reqc = has_reqi = has_reqs = 1`` together for every scheme it
#: names -- which mp schemes hand their effective radii to RRTMG.
#: Keyed by mp_physics for every scheme gpuwm can select; None-lookup
#: fails closed in __call__.
#:
#: THIS DICT IS HALF THE ANSWER.  WRF's SECOND block (:1027-1033) then
#: re-zeroes ``has_reqs`` for the P3 family and Jensen-Ishmael, so the
#: has_req triple is NOT uniform for every scheme and cannot be spelled
#: as one boolean.  That block is :data:`_MP_SNOW_RADII_SUPPRESSED`;
#: read the two together through :func:`legacy_scheme_has_req`, which is
#: what the adapter calls.  Morrison (10) is deliberately False --
#: WRF does NOT couple Morrison radii to RRTMG (its EFFI bound of
#: 525 um would trip cldprmc's [5,140] fatal); Kessler (1) and the
#: no-microphysics case (0) have no radii.  NSSL 2-moment (18) is True
#: for the campaign's nssl_2moment_on=1 configuration (the only NSSL
#: form gpuwm runs).
_MP_DECLARES_RADII = {
    0: False,
    1: False,
    6: True,     # WSM6
    8: True,     # Thompson
    # Milbrandt-Yau: NOT in WRF's use_mp_re list either
    # (module_physics_init.F:1004-1023 names THOMPSON, THOMPSONAERO,
    # NSSL_2MOM, the WSM/WDM family, nuwrf4ice, Jensen-Ishmael and the P3
    # family, and no MILBRANDT2MOM), and the scheme's own effective-radius
    # block is commented out (module_mp_milbrandt2mom.F:3362/:3364/
    # :3372/:3374), so there is nothing to declare and RRTMG computes its
    # own radii exactly as it does under any has_reqc=0 scheme.
    9: False,
    10: False,   # Morrison two-moment: NOT in WRF's use_mp_re list
    # WDM6.  WRF lists WDM6SCHEME explicitly in the use_mp_re disjunction
    # (module_physics_init.F:1013, beside WSM6SCHEME at :1010) and the
    # P3/Jensen-Ishmael has_reqs=0 override (:1027-1033) does not name it,
    # so has_reqc = has_reqi = has_reqs = 1.  The radii themselves come from
    # effectRad_wdm6 (module_mp_wdm6.F:3135-3234), which CLAMPS every one
    # of them: cloud to [2.51, 50] um (:3212), ice to [10.01, 125] um
    # (:3220), snow to [25, 999] um (:3229).  The ice bound is what matters
    # here -- cldprmc's [5, 140] um fatal is the reason Morrison (EFFI to
    # 525 um) is False above, and WDM6's 125 um cap sits inside it.
    16: True,    # WDM6 (WDM6SCHEME)
    18: True,    # NSSL 2-moment (nssl_2moment_on=1)
    # Thompson aerosol-aware.  WRF lists it EXPLICITLY and SEPARATELY from
    # THOMPSON in the same disjunction:
    #   module_physics_init.F:1005  config_flags%mp_physics .eq. THOMPSON
    #   module_physics_init.F:1006  config_flags%mp_physics .eq. THOMPSONAERO
    # (verified in wrf-stock-v461-gate-20260721; THOMPSONAERO is
    # mp_physics==28 per Registry/Registry.EM_COMMON:3036).  It is not
    # excluded by the P3/Jensen-Ishmael has_reqs=0 override at :1027-1033
    # either, so all three of has_reqc/has_reqi/has_reqs are 1.
    #
    # The radii themselves are already the mp=8 pair plus the prognostic-nc
    # dependence: gpuwm's mp=28 adapter calls calc_effectRad and then applies
    # mp_gt_driver's OWN clamps (:1475-1477, MAX(RE_QC_BG, MIN(re, 50.E-6))
    # and the 125/999 um pair) -- the identical clamps mp=8 takes, because
    # mp_gt_driver is ONE driver serving both packages.  So the upper bounds
    # that keep Morrison out of this table (cldprmc's [5,140] um fatal) are
    # satisfied by construction for mp=28 exactly as they are for mp=8.
    28: True,    # Thompson aerosol-aware (THOMPSONAERO)
    # P3 one-category.  WRF names P3_1CATEGORY in the use_mp_re
    # disjunction (module_physics_init.F:1017, beside THOMPSON at :1005),
    # so has_reqc = has_reqi = 1 and P3's OWN predicted radii are what
    # radiation sees: module_microphysics_driver.F's P3_1CATEGORY arm
    # (:1557) binds diag_effc_3d -> re_cloud and diag_effi_3d -> re_ice
    # (:1597-1598) unconditionally, and gpuwm's adapter writes both every
    # step in the micron convention (gpuwm/core/p3.py:1821-1825 CPU,
    # :1979-1980 device), over the 10/25 um backgrounds
    # gpuwm/core/state.py:494-495 seeds from module_mp_p3.F:2279-2281.
    #
    # has_reqs is a SEPARATE answer and is 0 -- see
    # _MP_SNOW_RADII_SUPPRESSED below.  P3 is the first scheme in this
    # table whose WRF triple is not uniform, which is why the value here
    # is only the first block's flag and the adapter reads both.
    #
    # The ice bound that keeps Morrison out is not in play.  Under the
    # resulting (1, 1, 0) the wrapper forces reice1d = 10 um and routes
    # P3's ice radius into the SNOW slot, where WRF clamps it to 130 um
    # with an area-conserving mass reduction before the optics
    # (module_ra_rrtmg_lw.F:12519-12522), so cldprmc's [5, 140] fatal is
    # unreachable however large P3's lookup-table effective radius grows
    # (the shipped p3_lookupTable_1.dat-v5.4_2momI reaches 2.1e4 um).
    50: True,    # P3 one-category (P3_1CATEGORY)
}


def legacy_scheme_declares_radii(mp_physics, use_mp_re):
    """WRF v4.6.1's ``use_mp_re`` gate around its scheme table.

    FIRST BLOCK ONLY (``module_physics_init.F:988-1024``): True means the
    scheme is named in the disjunction, i.e. ``has_reqc = has_reqi = 1``.
    It is NOT the has_reqs answer -- WRF's :1027-1033 override re-zeroes
    has_reqs for the P3 family, so a caller that needs the flags the
    wrapper actually takes must use :func:`legacy_scheme_has_req`.
    """
    table_declares = _MP_DECLARES_RADII.get(int(mp_physics))
    if table_declares is None:
        raise NotImplementedError(
            f"mp_physics={int(mp_physics)} has no has_req* entry in the "
            "WRF v4.6.1 use_mp_re scheme table carried by the legacy RRTMG "
            "adapter; extend _MP_DECLARES_RADII from "
            "module_physics_init.F:985-1024 rather than guessing")
    value = int(use_mp_re)
    if value not in (0, 1):
        raise ValueError(f"use_mp_re must be 0 or 1, got {use_mp_re!r}")
    return bool(value) and table_declares


#: WRF v4.6.1 use_mp_re scheme table, SECOND BLOCK: the P3 /
#: Jensen-Ishmael override at ``module_physics_init.F:1027-1033``, which
#: re-zeroes ``has_reqs`` AFTER the first block set all three flags to 1.
#: These are the schemes whose ice is ONE category with no separate snow
#: species, so there is no snow radius for them to declare: the mp=50
#: package is ``moist:qv,qc,qr,qi`` with ``state:re_cloud,re_ice`` and no
#: qs and no re_snow (``Registry/Registry.EM_COMMON:3038``).
#:
#: The resulting ``(has_reqc, has_reqi, has_reqs) = (1, 1, 0)`` is not a
#: degenerate corner: it is the SOLE trigger of the wrapper's "special
#: case for P3 microphysics" (``module_ra_rrtmg_lw.F:12250-12261``,
#: ``module_ra_rrtmg_sw.F:10853``), which moves the single ice category
#: into the SNOW optics slot -- resnow1d = MAX(10., re_ice*1e6),
#: QS1D = raw QI3D, QI1D = 0, reice1d = 10 -- and which this port already
#: carries at ``gpuwm/core/rrtmg_legacy_prep.py:464-471`` (and its device
#: twin at :1322-1329).  Until mp=50 entered the table above, no adapter
#: state could produce that triple and the transcribed branch was
#: unreachable.
#:
#: Of WRF's P3/Jensen-Ishmael family only mp=50 is selectable here:
#: mp=51 and mp=52 are refused by name in
#: ``gpuwm.config._P3_UNPORTED_VARIANTS`` and Jensen-Ishmael (mp=55) is
#: outside ``gpuwm.config.MP_PHYSICS_ACCEPTED`` entirely.
_MP_SNOW_RADII_SUPPRESSED = frozenset((50,))


def legacy_scheme_has_req(mp_physics, use_mp_re):
    """WRF's ``(has_reqc, has_reqi, has_reqs)`` triple, both blocks.

    Read in WRF's own order (``module_physics_init.F:987-1033``): the
    flags start at 0, the :988-1024 disjunction sets all three to 1 for
    the schemes :data:`_MP_DECLARES_RADII` names, then the :1027-1033
    override re-zeroes has_reqs for the schemes in
    :data:`_MP_SNOW_RADII_SUPPRESSED`.  The triple is what the wrapper
    branches on, and it is not uniform for every scheme -- P3 (mp=50) is
    ``(1, 1, 0)`` -- so the adapter asks for the triple and never for one
    boolean.
    """
    declares = legacy_scheme_declares_radii(mp_physics, use_mp_re)
    flag = int(bool(declares))
    has_reqs = 0 if int(mp_physics) in _MP_SNOW_RADII_SUPPRESSED else flag
    return flag, flag, has_reqs

#: WRF Registry ``F_QI``/``F_QS`` membership: the schemes whose package
#: declaration carries ``qi`` and ``qs`` in ``moist``, which is what
#: ``cal_cldfra1`` keys on.  28 is a member --
#: ``Registry/Registry.EM_COMMON:3036`` declares
#: ``package thompsonaero mp_physics==28 - moist:qv,qc,qr,qi,qs,qg;...``,
#: character for character the same ``moist:qv,qc,qr,qi,qs,qg`` inventory
#: line 3024's ``thompson`` (mp=8) carries.  9 is a member for the same
#: reason: ``Registry/Registry.EM_COMMON:3025`` declares ``package
#: milbrandt2mom mp_physics==9 - moist:qv,qc,qr,qi,qs,qg,qh;scalar:qnc,
#: qnr,qni,qns,qng,qnh``.  Membership here is about F_QI/F_QS and is
#: INDEPENDENT of has_req*: mp=9 declares no radii and still gets ice
#: cloud fraction, which is exactly WRF's pairing.
#: 16 (WDM6) is a member on the same reading: ``package wdm6scheme`` at
#: ``Registry/Registry.EM_COMMON:3031`` gives it
#: ``moist:qv,qc,qr,qi,qs,qg``.
#:
#: This set answers ``F_QI AND F_QS``; it is NOT "does the scheme have
#: ice".  A scheme carrying ice with NO snow species belongs in
#: :data:`_LEGACY_ICE_ONLY_MICROPHYSICS` below instead -- adding it here
#: would assert a ``qs`` its Registry package never declares.  Resolve the
#: pair through :func:`legacy_cloud_fraction_flags`, never by reading one
#: boolean into both flags.
_LEGACY_ICE_ACTIVE_MICROPHYSICS = frozenset((6, 8, 9, 10, 16, 18, 28))

#: WRF Registry ``F_QI and not F_QS``: schemes whose package carries ``qi``
#: in ``moist`` and no ``qs`` at all.  P3 one-category (mp=50) is the
#: member.  ``Registry/Registry.EM_COMMON:3038`` declares ``package
#: p3_1category mp_physics==50 - moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib;
#: state:re_cloud,re_ice,...`` -- there is a ``qi``, there is no ``qs`` and
#: no ``qg``, because P3 carries ONE ice category and predicts rime mass
#: and rime volume (``qir``/``qib``) instead of splitting snow from
#: graupel.  ``gpuwm/core/state.py`` allocates exactly that inventory and
#: no ``qs``.
#:
#: mp=50's ABSENCE from :data:`_LEGACY_ICE_ACTIVE_MICROPHYSICS` is a
#: decision, not an omission: admitting it there would hand
#: ``cal_cldfra1`` ``F_QS = true`` for a scheme with no snow species.  WRF
#: does not fuse the two flags either -- ``phys/module_radiation_driver.F:
#: 3879-3887`` gives this case its own arm, commented "for P3, mp option
#: 50 or 51", with ``QCLD = QI + QC`` and ``weight = QI/QCLD``, distinct
#: from the ``F_QI .and. F_QC .and. F_QS`` arm at :3870-3877.
#:
#: mp=51 (``p3_1category_nc``, Registry.EM_COMMON:3039) is the same WRF
#: arm and is deliberately NOT listed: ``gpuwm/config.py`` does not accept
#: 51, so it is refused by name at admission rather than half-answered
#: here.
_LEGACY_ICE_ONLY_MICROPHYSICS = frozenset((50,))

#: WRF Registry ``not F_QI and not F_QS``: no frozen species in the package
#: at all -- ``passiveqv`` (mp=0, Registry.EM_COMMON:3014, ``moist:qv``)
#: and ``kesslerscheme`` (mp=1, :3015, ``moist:qv,qc,qr``).  These take
#: ``cal_cldfra1``'s qc-only arm with its 273.15 K phase threshold
#: (module_radiation_driver.F:3891-3899).
_LEGACY_NO_ICE_MICROPHYSICS = frozenset((0, 1))


def legacy_ice_active(mp_physics: int) -> bool:
    """WRF Registry ``F_QI`` **and** ``F_QS`` membership for ``cal_cldfra1``.

    True only where the Registry package declares BOTH species.  It is not
    the answer to "is ice active": P3 (mp=50) has ice and no snow and is
    False here on purpose.  Anything wiring ``cal_cldfra1`` must call
    :func:`legacy_cloud_fraction_flags` instead.
    """

    return int(mp_physics) in _LEGACY_ICE_ACTIVE_MICROPHYSICS


def legacy_cloud_fraction_flags(mp_physics: int) -> tuple[bool, bool]:
    """``cal_cldfra1``'s ``(F_QI, F_QS)`` pair for one selector.

    WRF hands the radiation driver the Registry ``qi`` and ``qs`` package
    flags SEPARATELY: ``module_radiation_driver.F:3867`` tests all three of
    F_QI/F_QC/F_QS for presence, then :3870, :3880 and :3890 branch on
    their VALUES, and the three arms are different physics --
    ``QCLD = QI+QC+QS`` weighted ``(QI+QS)/QCLD``, ``QCLD = QI+QC``
    weighted ``QI/QCLD``, and ``QCLD = QC`` weighted by the freezing point.
    Reading ONE boolean into both flags can only reach the first and the
    third, and it silently gives the third -- a temperature-thresholded
    LIQUID cloud fraction -- to any scheme with ice but no snow.

    MEASURED on this tree for mp=50 before the split: a 258 K / 60 kPa
    column carrying 3e-5 kg/kg of P3 ice and no cloud water returned
    CLDFRA 0.0 (clear sky) where WRF's own P3 arm returns 1.0 (overcast).

    FAILS CLOSED.  Every selector ``gpuwm/config.py`` accepts has a row in
    one of the three sets; an unmapped one raises instead of inheriting
    Kessler's ice-free arm.
    """

    selector = int(mp_physics)
    if selector in _LEGACY_ICE_ACTIVE_MICROPHYSICS:
        return (True, True)
    if selector in _LEGACY_ICE_ONLY_MICROPHYSICS:
        return (True, False)
    if selector in _LEGACY_NO_ICE_MICROPHYSICS:
        return (False, False)
    raise NotImplementedError(
        f"mp_physics={selector} has no F_QI/F_QS row in the legacy RRTMG "
        "adapter's Registry membership tables, so cal_cldfra1's arm is "
        "undecided; read the scheme's ``package`` line in "
        "Registry/Registry.EM_COMMON and add the selector to "
        "_LEGACY_ICE_ACTIVE_MICROPHYSICS (moist carries qi and qs), "
        "_LEGACY_ICE_ONLY_MICROPHYSICS (qi, no qs) or "
        "_LEGACY_NO_ICE_MICROPHYSICS (neither) rather than letting it "
        "default to the qc-only arm, which radiates an ice cloud as clear "
        "sky")


def legacy_radius_meters(effective_radius_microns):
    """Apply the RRTMG wrapper's micron-to-meter conversion exactly once."""

    return (effective_radius_microns * F(1.0e-6)).astype(np.float32)


def _r512(nbytes):
    """CuPy's 512-byte pool allocation quantum."""
    return (int(nbytes) + 511) & ~511


def legacy_radiation_vram_bytes(*, ncol, nz, p_top, column_chunk=None,
                                ncol_day=None, lw_coefficients=None):
    """Peak transient device bytes of ONE adapter call.

    Composes the engines' own honest pricing functions
    (``lw_batched_vram_bytes`` + ``lw_batched_const_bytes``,
    ``sw_batched_vram_bytes``) and the McICA device pricing
    (``mcica_device_vram_bytes``) with the adapter-held device arrays the
    engine functions do not price: the McICA output slabs and radii
    pass-throughs that stay alive as engine inputs.  The LW and SW
    pipelines run sequentially per adapter chunk with everything freed
    in between, so the estimate is the max over the four allocation
    phases (LW generate, LW engine, SW generate, SW engine).  The CudaSW
    instance constants (uploaded at adapter construction) and the tiny
    host->device cldfra staging are not included, mirroring the engines'
    own gates.  ``ncol_day`` bounds the SW day-column count (default:
    ``ncol``, the preflight upper bound).
    """
    ncol = int(ncol)
    nz = int(nz)
    nlay_lw = _prep.compute_lw_nlayers(nz + 1, p_top)
    nlay_sw = nz + 1
    nday = ncol if ncol_day is None else int(ncol_day)
    chunk = None if column_chunk is None else int(column_chunk)
    nc_lw = min(chunk or _lw.LW_BATCH_COLUMN_CHUNK, ncol)
    nc_sw = min(chunk or _sw.SW_BATCH_COLUMN_CHUNK, max(nday, 0))
    C = lw_coefficients if lw_coefficients is not None else _lw_coeffs()
    f = 4

    s_mcl = _r512(nc_lw * _lw.NGPTLW * nlay_lw * f)
    s_nl = _r512(nc_lw * nlay_lw * f)
    held_lw = 5 * s_mcl + 3 * s_nl        # mcl slabs + rei/rel/res
    lw_gen = (held_lw + 5 * s_nl          # play cldfrac ciwp clwp cswp
              + _r512(_mcica.NBNDLW * nc_lw * nlay_lw * f)   # tauc
              + _mcica.mcica_device_vram_bytes(
                  min(nc_lw, _mcica.MCICA_DEVICE_COLUMN_CHUNK),
                  nlay_lw, _lw.NGPTLW))
    lw_eng = (held_lw + _lw.lw_batched_vram_bytes(nc_lw, nlay_lw)
              + _lw.lw_batched_const_bytes(C))
    estimate = max(lw_gen, lw_eng)

    if nc_sw:
        s_mcl_s = _r512(nc_sw * _sw.NGPTSW * nlay_sw * f)
        s_nl_s = _r512(nc_sw * nlay_sw * f)
        held_sw = 8 * s_mcl_s + 3 * s_nl_s
        sw_gen = (held_sw + 5 * s_nl_s
                  + 4 * _r512(_mcica.NBNDSW * nc_sw * nlay_sw * f)
                  + _mcica.mcica_device_vram_bytes(
                      min(nc_sw, _mcica.MCICA_DEVICE_COLUMN_CHUNK),
                      nlay_sw, _sw.NGPTSW))
        sw_eng = held_sw + _sw.sw_batched_vram_bytes(nc_sw, nlay_sw)
        estimate = max(estimate, sw_gen, sw_eng)
    return estimate


# ---------------------------------------------------------------------------
# LW module-DATA tables: WRF v4.6.1's module_ra_rrtmg_lw carries
# lwavplank (totplnk/totplk16), lwatmref (preflog/tref/chi_mls), lwcldpr
# (absice*/absliq*) and the rrlw_wvn delwave/ngb rosters as COMPILE-TIME
# DATA statements -- they are part of the algorithm, not of the
# RRTMG_LW_DATA coefficient file, so the packaged-file chain
# (load_rrtmg_lw_coefficients -> build_lw_coefficients) alone cannot
# feed the batched engine.  They are packaged as
# gpuwm/data/wrf_radiation/rrtmg_lw_statics.npz, generated BIT-EXACT
# from the compiled UNMODIFIED Fortran's own module state (the same
# oracle authority every LW gate uses) by
# tools/rrtmg_wrf461_oracle/lw_statics_package.py, SHA-256-pinned here,
# and re-gated bitwise against the oracle dump by
# tests/test_rrtmg_legacy_wiring.py whenever the fixture deck is present
# (provenance chain: gpuwm/data/wrf_radiation/PROVENANCE.md).
# ---------------------------------------------------------------------------

#: (key, shape, dtype) roster of the packaged LW module-DATA tables, in
#: packaged member order.  Shapes/dtypes are the Fortran module shapes.
_LW_STATIC_SPECS = (
    ("wvn/totplnk", (181, 16), "float32"),
    ("wvn/totplk16", (181,), "float32"),
    ("wvn/delwave", (16,), "float32"),
    ("wvn/ngb", (140,), "int32"),
    ("ref/preflog", (59,), "float32"),
    ("ref/tref", (59,), "float32"),
    ("ref/chi_mls", (7, 59), "float32"),
    ("cld/absice0", (2,), "float32"),
    ("cld/absice1", (2, 5), "float32"),
    ("cld/absice2", (43, 16), "float32"),
    ("cld/absice3", (46, 16), "float32"),
    ("cld/absliq0", (), "float32"),
    ("cld/absliq1", (58, 16), "float32"),
)

#: Packaged asset carrying the 13 tables (one npz member per roster key,
#: C-ordered little-endian, member order = the roster order).
_LW_STATICS_PATH = (Path(__file__).resolve().parents[1] / "data"
                    / "wrf_radiation" / "rrtmg_lw_statics.npz")

#: SHA-256 of rrtmg_lw_statics.npz (fail-closed on corruption).  Also
#: pinned into the adapter's restart identity; the restart asset
#: manifest carries the same file under the role "wrf_rrtmg_lw_statics".
RRTMG_LW_STATICS_SHA256 = \
    "edd2508db89180667b0f80b4cdd991f4aa1447e711bbfe3a6db8de5fdb778d62"

_LW_STATICS_CACHE = None


def _lw_static_tables():
    """Load the packaged module-DATA tables (fail-closed on drift).

    Returns the same immutable dict the base64-blob predecessor decoded:
    one read-only C-ordered array per ``_LW_STATIC_SPECS`` key (the
    ()-shaped ``cld/absliq0`` as a float32 scalar), file digest, member
    roster/order, shapes and dtypes all enforced before anything is
    handed to the coefficient merge.
    """
    global _LW_STATICS_CACHE
    if _LW_STATICS_CACHE is None:
        import hashlib
        import io
        try:
            payload = _LW_STATICS_PATH.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                "packaged LW module-DATA asset is missing: "
                f"{_LW_STATICS_PATH} ({exc})") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if digest != RRTMG_LW_STATICS_SHA256:
            raise RuntimeError(
                "packaged LW module-DATA asset is corrupt: sha256 "
                f"{digest} != pinned {RRTMG_LW_STATICS_SHA256}")
        out = {}
        with np.load(io.BytesIO(payload)) as npz:
            expected = [key for key, _shape, _dtype in _LW_STATIC_SPECS]
            if list(npz.files) != expected:
                raise RuntimeError(
                    "packaged LW module-DATA asset carries the wrong "
                    f"member roster: {list(npz.files)} != {expected}")
            for key, shape, dtype in _LW_STATIC_SPECS:
                arr = npz[key]
                if tuple(arr.shape) != tuple(shape) or \
                        arr.dtype != np.dtype(dtype):
                    raise RuntimeError(
                        f"packaged LW module-DATA table {key} has shape "
                        f"{arr.shape} dtype {arr.dtype}; the contract "
                        f"requires {tuple(shape)} {dtype}")
                value = arr[()] if arr.ndim == 0 else \
                    np.ascontiguousarray(arr)
                if hasattr(value, "setflags"):
                    value.setflags(write=False)
                out[key] = value
        _LW_STATICS_CACHE = out
    return _LW_STATICS_CACHE


_LW_COEFFS_CACHE = None
_SW_TABLES_CACHE = None


def _lw_coeffs():
    """The COMPLETE batched-engine coefficient dict, once per process.

    Three bitwise-vouched sources, merged: (1) the init-chain builds of
    ``build_lw_coefficients`` on the packaged RRTMG_LW_DATA raw arrays
    (tables/constants/reductions); (2) the packaged loader's own reduced
    per-band arrays (``absa``/``absb``/... -- gated bitwise against the
    Fortran module dump by the ingest gates), which carry the
    ``absa``-equivalence views the CUDA band kernels index; (3) the
    packaged module-DATA tables above.  ``lw_batched_const_bytes`` runs
    as a completeness probe so a missing key fails at load, not at the
    first radiation call.
    """
    global _LW_COEFFS_CACHE
    if _LW_COEFFS_CACHE is None:
        from gpuwm.ingest import rrtmg_coeffs as _coeffs
        loaded = _coeffs.load_rrtmg_lw_coefficients()
        C = _lw.build_lw_coefficients(loaded, np.float32(1004.5))
        for band in range(1, 17):
            module = loaded[f"rrlw_kg{band:02d}"]
            for name, value in module.items():
                C.setdefault(f"kg{band:02d}/{name}", value)
        C.update(_lw_static_tables())
        _lw.lw_batched_const_bytes(C)      # completeness probe
        _LW_COEFFS_CACHE = C
    return _LW_COEFFS_CACHE


def _sw_tables():
    """Load + assemble the SW tables once per process (immutable)."""
    global _SW_TABLES_CACHE
    if _SW_TABLES_CACHE is None:
        from gpuwm.ingest import rrtmg_coeffs as _coeffs
        _SW_TABLES_CACHE = _sw.tables_from_coeffs(
            _coeffs.load_rrtmg_sw_coefficients())
    return _SW_TABLES_CACHE


_CUDA_SW_CACHE = None


def _cuda_sw(tab):
    """The compiled CUDA SW engine for ``tab``, once per process.

    Construction is the expensive half of this adapter: ``CudaSW.__init__``
    runs a full NVRTC compile of ``kernels/rrtmg_sw.cu`` and uploads the
    packed tables to the device.  The LW and McICA halves are already
    process-cached (``rrtmg_lw._GPU_KERNELS`` / ``_GPU_PREFLIGHTED``,
    ``rrtmg_mcica``); the SW engine was the only per-INSTANCE GPU cost,
    which did not matter while one adapter existed per process and does
    now: streaming builds one adapter per TILE BUFFER, so an uncached
    engine would multiply both the compile and the resident device tables
    that streaming exists to save.

    Sharing is sound because ``CudaSW`` is immutable after construction --
    every ``self.<x> =`` in it is in ``__init__`` (``cp``, ``tab``,
    ``module``, ``tab_gpu``, ``ngb_gpu``, ``max_nlay``), all read-only
    thereafter, and its stage drivers allocate their outputs per call.  It
    is a compiled module plus constant tables, not a carrier.

    Keyed on the tables OBJECT, not merely cached once: ``_sw_tables`` is
    itself a process singleton, so the identity check is normally free, but
    a caller assembling its own tables gets its own engine rather than
    silently borrowing one built from different coefficients.
    """
    global _CUDA_SW_CACHE
    if _CUDA_SW_CACHE is None or _CUDA_SW_CACHE[0] is not tab:
        _CUDA_SW_CACHE = (tab, _sw.CudaSW(tab))
    return _CUDA_SW_CACHE[1]


# ---------------------------------------------------------------------------
# Ozone nest routing: the child side of WRF's root-only o3rad evaluation.
# ---------------------------------------------------------------------------

class ParentOzoneProvider:
    """Child-domain o33d: the parent's retained field, SINT-interpolated.

    Wired by ``runtime.prepare_child_case`` with the parent domain's
    :class:`RRTMGLegacyRadiation` and the mass-point ``wrapper="interp"``
    :class:`gpuwm.core.nest_interp.NestRegistration` for the pair.  Each
    child radiation call reads the parent's most recent retained o33d
    grid (so the child's ozone updates exactly when the parent's does,
    like WRF's rdf-interpolated ``o3rad``) and interpolates it through
    the certified SINT operator.  Fails closed if the parent has not
    radiated yet: WRF's sequencing evaluates the root's o3rad before any
    nest radiation, so a None parent field is a sequencing violation,
    not a case for a silent climatology fallback.
    """

    def __init__(self, parent, registration):
        self.parent = parent
        self.registration = registration

    def __call__(self):
        import cupy as cp

        from gpuwm.core import nest_interp as _nest

        grid = getattr(self.parent, "_o33d_grid", None)
        if grid is None:
            raise RuntimeError(
                "child rrtmg_legacy radiation ran before its parent "
                "retained an o33d field: WRF computes the root domain's "
                "o3rad before any nest radiation (root-compute + "
                "parent->child interpolation routing); fix the call "
                "sequencing instead of falling back to a per-nest "
                "climatology WRF never evaluates")
        child = cp.asnumpy(_nest.sint(cp.asarray(grid), self.registration))
        nz = child.shape[0]
        return np.ascontiguousarray(
            child.transpose(1, 2, 0).reshape(-1, nz))


# ---------------------------------------------------------------------------
# The adapter.
# ---------------------------------------------------------------------------

class RRTMGLegacyRadiation:
    """WRF v4.6.1 legacy RRTMG (option 4/4) forecast radiation adapter.

    Plugs into the same ``radiation_due`` slot as ``RRTMGPRadiation``
    (``__call__(atmosphere=, fields=, state=, cfg=)`` returning
    :class:`gpuwm.core.physics.RadiationResult`).  Construction performs
    the full readiness proof -- packaged SHA-pinned assets, coefficient
    builds, CUDA kernel compilation and live-device preflights -- and
    fails closed with a clear receipt if anything is missing; this
    replaces the retired ``require_rrtmg_legacy_executable`` stub with
    executable reality.

    ``p_top`` may be None at construction (the value is then taken from
    ``state.p_top`` at call time; construction sites that know it pass it
    eagerly).  ``column_chunk=None`` uses each engine's own default chunk
    (LW 4096 / SW 2048); an explicit value bounds both pipelines and is
    part of the restart identity.  Chunking is bitwise invisible (prep
    batch entries, McICA twin, and both engines are each per-column
    bitwise at any width -- their own gated contracts).
    """

    #: RRTMG_LWRAD's ``OLR`` (TOA outgoing longwave, W m-2) is
    #: ``TOTUFLUX`` at the top level -- ``lwrad_outputs_batch`` already maps
    #: it out of ``uflx`` -- so this adapter fills the driver's OLR slot and
    #: the run's wrfout carries the field.  The declaration is what the
    #: driver reads to decide whether OLR exists at all.
    publishes_olr = True

    def __init__(self, start_time, latitude_deg, longitude_deg, *,
                 p_top=None, column_chunk=None, ozone_parent=None,
                 o3input=2):
        if not isinstance(start_time, datetime):
            raise TypeError("radiation_start_time must be a datetime")
        self.start_time = start_time
        lat = np.asarray(self._host(latitude_deg), np.float32)
        lon = np.asarray(self._host(longitude_deg), np.float32)
        if lat.shape != lon.shape:
            raise ValueError("radiation latitude/longitude shapes must match")
        self.latitude_deg = np.ascontiguousarray(lat)
        self.longitude_deg = np.ascontiguousarray(lon)
        self.p_top = None if p_top is None else float(p_top)
        self.o3input = int(o3input)
        if self.o3input not in (0, 2):
            raise ValueError(
                "legacy RRTMG implements o3input=0 (wrapper O3DATA) and "
                f"o3input=2 (CAM climatology), got {o3input!r}")
        if column_chunk is not None and int(column_chunk) < 1:
            raise ValueError("column_chunk must be positive")
        self.column_chunk = None if column_chunk is None \
            else int(column_chunk)
        if ozone_parent is not None and not callable(ozone_parent):
            raise TypeError(
                "ozone_parent must be a callable provider (e.g. "
                "ParentOzoneProvider) returning the child-grid o33d "
                "columns, or None on the root domain")
        if self.o3input == 0 and ozone_parent is not None:
            raise ValueError(
                "o3input=0 constructs ozone inside the legacy wrapper and "
                "must not receive a parent o3rad provider")
        self._ozone_provider = ozone_parent
        #: the most recent o33d field (nz, ny, nx) host float32, retained
        #: so child domains can interpolate it (WRF's root-compute +
        #: parent->child o3rad routing); None until the first call.
        self._o33d_grid = None
        self.update_count = 0
        #: test instrumentation forwarded to the engines' _stage_probe
        #: hooks (must not affect results); None on the forecast path.
        self._stage_probe = None

        # ---- fail-closed readiness: assets, tables, kernels ----------
        try:
            self._C = _lw_coeffs()
        except Exception as exc:
            raise RuntimeError(
                "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                "packaged RRTMG_LW_DATA coefficients cannot be "
                f"loaded/built: {exc}") from exc
        try:
            self._sw_tables = _sw_tables()
        except Exception as exc:
            raise RuntimeError(
                "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                "packaged RRTMG_SW_DATA coefficients cannot be "
                f"loaded/built: {exc}") from exc
        if self.o3input == 0:
            # O3DATA is evaluated independently inside lwrad/swrad prep and
            # does not read the CAM climatology or parent-routed o3rad field.
            self._ozone = None
            self._ozone_climo = None
            self._ozone_lat_interp = None
        elif self._ozone_provider is None:
            # Root routing: the climatology chain runs here (WRF: o3rad
            # is evaluated on id==1 only).  The latitude interpolation is
            # WRF's oznini-time work, cached once per domain.
            try:
                from gpuwm.ingest import wrf_ozone as _ozone
                self._ozone = _ozone
                self._ozone_climo = _ozone.load_ozone_climatology()
                self._ozone_lat_interp = _ozone.interp_ozone_to_latitudes(
                    self.latitude_deg.reshape(-1), self._ozone_climo)
            except Exception as exc:
                raise RuntimeError(
                    "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                    "packaged CAM ozone climatology (ozone*.formatted) "
                    f"cannot be loaded: {exc}") from exc
        else:
            # Child routing: o33d arrives from the parent; the child
            # never invokes the climatology chain (WRF nests receive the
            # parent-interpolated o3rad, never a fresh evaluation).
            self._ozone = None
            self._ozone_climo = None
            self._ozone_lat_interp = None
        try:
            self._cuda_sw = _cuda_sw(self._sw_tables)
        except Exception as exc:
            raise RuntimeError(
                "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                "CUDA SW engine cannot be compiled/loaded on this "
                f"installation (cupy + a CUDA device are required): {exc}"
            ) from exc
        try:
            _lw.gpu_preflight()
            _mcica.mcica_gpu_preflight()
        except Exception as exc:
            raise RuntimeError(
                "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                "CUDA LW/McICA kernels fail their live-device preflight: "
                f"{exc}") from exc

    # ------------------------------------------------------------------
    @staticmethod
    def _host(a):
        """Bit-preserving host copy of a numpy or cupy array."""
        if type(a).__module__.split(".")[0] == "cupy":
            return a.get()
        return np.asarray(a)

    def _cols(self, a, nk):
        """(nk, ny, nx) device/host -> host float32 (ny*nx, nk) columns."""
        h = np.asarray(self._host(a), np.float32)
        return np.ascontiguousarray(
            h.transpose(1, 2, 0).reshape(-1, nk))

    @staticmethod
    def _flat(a):
        """(ny, nx) device/host -> host float32 (ny*nx,)."""
        h = RRTMGLegacyRadiation._host(a)
        return np.ascontiguousarray(np.asarray(h, np.float32).reshape(-1))

    def _grid3(self, cols, nz, ny, nx):
        import cupy as cp
        return cp.asarray(np.ascontiguousarray(
            cols.reshape(ny, nx, nz).transpose(2, 0, 1)))

    def _grid2(self, flat, ny, nx):
        import cupy as cp
        return cp.asarray(np.ascontiguousarray(flat.reshape(ny, nx)))

    # ------------------------------------------------------------------
    def restart_identity(self):
        """Strict-JSON identity: distinct from every RTE+RRTMGP identity
        so a restart written under one 4/4 implementation refuses to
        resume under the other (dossier section 1)."""
        from gpuwm.io.restart import (
            RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY,
            RRTMG_LEGACY_LW_ALGORITHM_IDENTITY,
            RRTMG_LEGACY_SW_ALGORITHM_IDENTITY)
        from gpuwm.physics_compat import WRF_RRTMG_LEGACY
        identity = {
            "algorithm": (f"lw={RRTMG_LEGACY_LW_ALGORITHM_IDENTITY};"
                          f"sw={RRTMG_LEGACY_SW_ALGORITHM_IDENTITY}"),
            "algorithms": {
                "lw": RRTMG_LEGACY_LW_ALGORITHM_IDENTITY,
                "sw": RRTMG_LEGACY_SW_ALGORITHM_IDENTITY,
            },
            "above_atmosphere_policy":
                RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY,
            "compatibility_token": WRF_RRTMG_LEGACY,
            "permuteseed_lw": int(_mcica.LW_PERMUTESEED),
            "permuteseed_sw": int(_mcica.SW_PERMUTESEED),
            "icld": 2,
            "idcor": 0,
            "o3input": self.o3input,
            "ghg_input": 0,
            "aer_opt": 0,
            "column_chunk": self.column_chunk,
            "p_top": self.p_top,
            "ozone_routing": (
                "wrapper-o3data" if self.o3input == 0 else
                ("parent-interpolated"
                 if self._ozone_provider is not None
                 else "root-climatology")),
            "statics_assets": {
                "rrtmg_lw_statics.npz": RRTMG_LW_STATICS_SHA256,
            },
        }
        if self.o3input == 2:
            # Constant pins only -- importing wrf_ozone performs no file I/O;
            # a child adapter still never CALLS its climatology chain.
            from gpuwm.ingest.wrf_ozone import (OZONE_LAT_SHA256,
                                                OZONE_PLEV_SHA256,
                                                OZONE_SHA256)
            identity["ozone_assets"] = {
                "ozone.formatted": OZONE_SHA256,
                "ozone_lat.formatted": OZONE_LAT_SHA256,
                "ozone_plev.formatted": OZONE_PLEV_SHA256,
            }
        return identity

    # ------------------------------------------------------------------
    def _check_pins(self, cfg):
        """Fail closed on any option outside the ported combination."""
        for name, pinned in _PINNED_OPTIONS.items():
            value = getattr(cfg, name, pinned)
            if int(value) != pinned:
                raise NotImplementedError(
                    f"rrtmg_legacy implements only {name}={pinned} "
                    f"(dossier section 1); got {name}={value!r} -- no "
                    "silent option substitution is applied")
        requested_o3input = int(getattr(cfg, "o3input", 2))
        if requested_o3input != self.o3input:
            raise ValueError(
                f"adapter was constructed for o3input={self.o3input} but "
                f"the run config requests o3input={requested_o3input}; rebuild "
                "the adapter so its asset/routing identity matches")

    def _validate_radii_micron(self, name, eff_um, q):
        """Reject meter-scale radii state (dossier section 11 caveat)."""
        mask = (q > np.float32(0.0)) & (eff_um > np.float32(0.0))
        if bool(mask.any()) and float(eff_um[mask].max()) < 1.0e-3:
            raise ValueError(
                f"state.{name} looks meter-scale (every value on cloudy "
                f"points is < 1e-3, max {float(eff_um[mask].max()):.3e}) "
                "but the gpuwm radii contract is MICRONS: this state was "
                "almost surely written before the Thompson radii-units "
                "fix.  Resuming it requires the cloud-radiation seam "
                "lane's restart migration; rrtmg_legacy will not silently "
                "rescale or radiate at clip floors")

    def _mcica_generator(self, gpu_entry):
        """Device McICA twin, resolved through the module attribute at
        call time so the wiring booby-traps can intercept it."""
        probe = self._stage_probe
        if probe is None:
            return gpu_entry
        return partial(gpu_entry,
                       _stage_probe=lambda c0, nc: probe("mcica"))

    # ------------------------------------------------------------------
    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        from gpuwm.core.physics import RadiationResult
        from gpuwm.core import rrtmgp as _rrtmgp

        self._check_pins(cfg)
        pressure = atmosphere["pressure"]
        nz, ny, nx = pressure.shape
        ncol = ny * nx
        if self.latitude_deg.shape != (ny, nx):
            raise ValueError(
                "radiation latitude/longitude must match state grid")
        p_top = self.p_top
        if p_top is None:
            declared = getattr(state, "p_top", None)
            if declared is None:
                raise ValueError(
                    "rrtmg_legacy needs p_top (constructor argument or "
                    "state.p_top) to build WRF's Cavallo buffer layers")
            p_top = float(declared)
        nlayers, sw_layers = legacy_radiation_layer_counts(nz, p_top)
        if nlayers > MAX_LONGWAVE_LAYERS:
            raise ValueError(
                f"LW nlayers={nlayers} exceeds the batched engine's "
                f"{MAX_LONGWAVE_LAYERS}-layer bound "
                "(rlw_rtrn_march RLW_MAXLAY)")
        if sw_layers > MAX_SHORTWAVE_LAYERS:
            raise ValueError(
                f"SW nlay={sw_layers} exceeds the CUDA SW engine's "
                f"{MAX_SHORTWAVE_LAYERS}-layer bound (RSW_MAXLAY)")

        mp_physics = int(getattr(cfg, "mp_physics", 0))
        warm_rain = mp_physics == 1
        # cal_cldfra1's two Registry package flags, resolved SEPARATELY.
        # One fused boolean cannot express P3's F_QI=true/F_QS=false, and
        # collapsing it sent mp=50 to the qc-only arm.
        f_qi, f_qs = legacy_cloud_fraction_flags(mp_physics)
        sf_surface_physics = int(getattr(cfg, "sf_surface_physics", 2))

        # ---- host column packing (pure data movement) -----------------
        p3d = self._cols(pressure, nz)
        p8w = self._cols(atmosphere["p_interface"], nz + 1)
        t3d = self._cols(atmosphere["temperature"], nz)
        pi3d = self._cols(atmosphere["exner"], nz)
        dz8w = self._cols(atmosphere["dz"], nz)
        z_at_w = self._cols(atmosphere["z_interface"], nz + 1)

        zeros_cols = np.zeros((ncol, nz), np.float32)
        moist = {}
        f_flags = {}
        for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
            value = getattr(state, name, None)
            if value is None:
                moist[name] = zeros_cols
                f_flags[name] = False
            else:
                moist[name] = self._cols(value, nz)
                f_flags[name] = True

        # ---- radii: MICRON state contract -> meters for the wrapper ---
        # has_req* follows WRF v4.6.1's SCHEME TABLE, not field presence
        # (module_physics_init.F:987-1033, gated on use_mp_re=1, the
        # Registry default and the campaign setting): Thompson, NSSL
        # 2-moment, the WSM/WDM families, and P3 declare radii; MORRISON
        # DOES NOT -- WRF's Morrison+RRTMG runs has_req*=0 with the
        # wrapper's relcalc/reicalc temperature radii, and Morrison's
        # EFFI upper bound (3*(2*DCS+100um)/2 = 525 um) would otherwise
        # reach cldprmc's [5,140] wrf_error_fatal.  Presence-based
        # detection reproduced exactly that fatal on a real Morrison
        # forecast (integration finding, 2026-07-27).
        #
        # The three flags are read PER FIELD because WRF's answer is not
        # uniform: its :1027-1033 override re-zeroes has_reqs for P3, so
        # mp=50 is (1, 1, 0) -- cloud and ice radii taken from the scheme,
        # no snow radius asked for and none allocated on P3 state.  A
        # single boolean here would have to either invent a P3 snow radius
        # or throw away the two radii P3 does predict.
        has_reqc, has_reqi, has_reqs = legacy_scheme_has_req(
            mp_physics, getattr(cfg, "use_mp_re", 1))
        radii = {}
        has_req = {}
        for name, key, q, scheme_declares in (
                ("effc", "re_cloud", "qc", has_reqc),
                ("effi", "re_ice", "qi", has_reqi),
                ("effs", "re_snow", "qs", has_reqs)):
            value = getattr(state, name, None)
            if not scheme_declares:
                # WRF semantics: the wrapper never reads scheme radii
                # for this mp; relcalc/reicalc take over inside prep.
                radii[key] = None
                has_req[name] = 0
                continue
            if value is None:
                raise ValueError(
                    f"mp_physics={int(mp_physics)} declares {name} to "
                    "radiation (WRF use_mp_re scheme table) but "
                    f"state.{name} is missing -- the microphysics radii "
                    "contract is broken; refusing to radiate fallback "
                    "radii silently")
            eff_um = self._cols(value, nz)
            self._validate_radii_micron(name, eff_um, moist[q])
            radii[key] = legacy_radius_meters(eff_um)
            has_req[name] = 1

        surf = {name: self._flat(fields[name])
                for name in ("tsk", "emiss", "albedo", "xland", "xice",
                             "snow")}
        lat_flat = self.latitude_deg.reshape(-1)
        lon_flat = self.longitude_deg.reshape(-1)

        # ---- t8w: phy_prep transcription (dossier section 4) ----------
        t8w = _t8w_columns(t3d, z_at_w,
                           self._host(state.fnm), self._host(state.fnp))

        # ---- cldfra: cal_cldfra1 (icloud=1), the WRF transcription
        # already gated on the RTE+RRTMGP path -- imported, not copied.
        cldfra = self._host(_rrtmgp.cal_cldfra1(
            cp.asarray(moist["qv"]), cp.asarray(moist["qc"]),
            cp.asarray(moist["qi"]), cp.asarray(moist["qs"]),
            cp.asarray(t3d), cp.asarray(p3d),
            f_qc=True, f_qi=f_qi, f_qs=f_qs))
        active_bl = mynn_bl_cloud_active(
            getattr(cfg, "bl_pbl_physics", 0), getattr(cfg, "icloud_bl", 0))
        if active_bl:
            # Absent species share ``zeros_cols`` above.  The WRF merge writes
            # QC/QI only, so give those two the same independent storage real
            # Registry moist arrays have before applying it.
            moist["qc"] = moist["qc"].copy()
            moist["qi"] = moist["qi"].copy()
        qc_bl = self._cols(fields["qc_bl"], nz) if active_bl else None
        qi_bl = self._cols(fields["qi_bl"], nz) if active_bl else None
        cldfra_bl = (
            self._cols(fields["cldfra_bl"], nz) if active_bl else None)
        moist["qc"], moist["qi"], cldfra = merge_mynn_bl_clouds(
            moist["qc"], moist["qi"], cldfra,
            qc_bl=qc_bl, qi_bl=qi_bl, cldfra_bl=cldfra_bl,
            bl_pbl_physics=getattr(cfg, "bl_pbl_physics", 0),
            icloud_bl=getattr(cfg, "icloud_bl", 0),
            itimestep=(wrf_itimestep(state.elapsed_seconds, cfg.dt)
                       if active_bl else 1),
        )

        # ---- calendar / solar (dossier sections 4, 9.1) ---------------
        valid_time = (self.start_time
                      + timedelta(seconds=float(state.elapsed_seconds)))
        yr = int(valid_time.year)
        julday = int(valid_time.timetuple().tm_yday)
        hour = (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0
                + valid_time.microsecond / 3.6e9)
        julian = F((julday - 1) + hour / 24.0)
        gmt = F(self.start_time.hour + self.start_time.minute / 60.0
                + self.start_time.second / 3600.0
                + self.start_time.microsecond / 3.6e9)
        xtime = F(float(state.elapsed_seconds) / 60.0)
        radt_minutes = F(cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes)
        declin, solcon = radconst(julian)
        # WRF: hour angle at the interval midpoint, declination/solcon at
        # call time (module_radiation_driver.F:1206-1208).
        xt_mid = xtime + radt_minutes * F(0.5)
        coszen = calc_coszen(julian, xt_mid, gmt, lat_flat, lon_flat,
                             declin)

        # ---- o33d: WRF's root-compute + parent->child routing ----------
        if self.o3input == 0:
            # The wrapper ignores o33d in this mode and builds O3DATA from
            # pressure and latitude.  Shape-correct zeros retain one batch
            # contract for both modes.
            o33d = np.zeros((ncol, nz), np.float32)
        elif self._ozone_provider is not None:
            o33d = np.asarray(self._ozone_provider(), np.float32)
            if o33d.shape != (ncol, nz):
                raise ValueError(
                    f"ozone_parent provider returned shape {o33d.shape}; "
                    f"this child grid needs ({ncol}, {nz})")
        else:
            ozmixt = self._ozone.ozn_time_int(julday, julian,
                                              self._ozone_lat_interp)
            o33d = self._ozone.ozn_p_int(p3d, self._ozone_climo.plev,
                                         ozmixt)
        # Retain for child domains (their providers read this field, so a
        # nest's ozone updates exactly when its parent's does).
        self._o33d_grid = (
            None if self.o3input == 0 else np.ascontiguousarray(
                o33d.reshape(ny, nx, nz).transpose(2, 0, 1)))

        shared = dict(
            icloud=1, warm_rain=warm_rain, cldovrlp=2, idcor=0,
            o3input=self.o3input,
            has_reqc=has_req["effc"], has_reqi=has_req["effi"],
            has_reqs=has_req["effs"], f_qc=f_flags["qc"],
            f_qr=f_flags["qr"], f_qi=f_flags["qi"], f_qs=f_flags["qs"],
            f_qg=f_flags["qg"], yr=yr, julian=julian,
            mp_physics=mp_physics)

        def chunk_inputs(idx):
            sel = dict(
                p3d=p3d[idx], p8w=p8w[idx], t3d=t3d[idx], t8w=t8w[idx],
                dz8w=dz8w[idx], qv3d=moist["qv"][idx],
                qc3d=moist["qc"][idx], qr3d=moist["qr"][idx],
                qi3d=moist["qi"][idx], qs3d=moist["qs"][idx],
                qg3d=moist["qg"][idx], cldfra3d=cldfra[idx],
                o33d=o33d[idx], tsk=surf["tsk"][idx],
                xland=surf["xland"][idx], xice=surf["xice"][idx],
                snow=surf["snow"][idx], xlat=lat_flat[idx])
            for key in ("re_cloud", "re_ice", "re_snow"):
                if radii[key] is not None:
                    sel[key] = radii[key][idx]
            return sel

        # ---- LW: all columns, adapter chunk == engine chunk -----------
        chunk_lw = self.column_chunk or _lw.LW_BATCH_COLUMN_CHUNK
        rthratenlw = np.zeros((ncol, nz), np.float32)
        glw = np.zeros(ncol, np.float32)
        olr = np.zeros(ncol, np.float32)
        for c0 in range(0, ncol, chunk_lw):
            idx = slice(c0, min(c0 + chunk_lw, ncol))
            pl = _prep.lwrad_prep_batch(
                **chunk_inputs(idx), emiss=surf["emiss"][idx],
                nlayers=nlayers,
                subcolumn_generator=self._mcica_generator(
                    _mcica.gpu_generate_lw_subcolumns),
                **shared)
            res = _lw.gpu_rrtmg_lw_batched(
                pl["ncol"], pl["nlay"], pl["icld"], pl["play"],
                pl["plev"], pl["tlay"], pl["tlev"], pl["tsfc"],
                pl["h2ovmr"], pl["o3vmr"], pl["co2vmr"], pl["ch4vmr"],
                pl["n2ovmr"], pl["o2vmr"], pl["cfc11vmr"],
                pl["cfc12vmr"], pl["cfc22vmr"], pl["ccl4vmr"],
                pl["emis"], pl["inflglw"], pl["iceflglw"],
                pl["liqflglw"], pl["cldfmcl"], pl["taucmcl"],
                pl["ciwpmcl"], pl["clwpmcl"], pl["cswpmcl"],
                pl["reicmcl"], pl["relqmcl"], pl["resnmcl"],
                pl["tauaer"], self._C, column_chunk=chunk_lw,
                _stage_probe=self._stage_probe)
            del pl
            outs = _prep.lwrad_outputs_batch(
                uflx=res["uflx"], dflx=res["dflx"], hr=res["hr"],
                uflxc=res["uflxc"], dflxc=res["dflxc"], hrc=res["hrc"],
                pi3d=pi3d[idx])
            rthratenlw[idx] = outs["rthratenlw"]
            glw[idx] = outs["glw"]
            olr[idx] = outs["olr"]
            del res, outs

        # ---- SW: driver-level zeroing (dossier section 3) + day gather
        rthratensw = np.zeros((ncol, nz), np.float32)
        gsw = np.zeros(ncol, np.float32)
        day_idx = np.nonzero(coszen > F(0.0))[0]
        night_idx = np.nonzero(coszen <= F(0.0))[0]
        if night_idx.size:
            # Wrapper-level night contract: COSZR + the SW_NIGHT_ZEROED
            # list.  None of these reach RadiationResult, but the write
            # keeps the wrapper contract exercised (dossier section 3).
            self._night_outputs = _prep.swrad_night_outputs_batch(
                coszen[night_idx])
        else:
            self._night_outputs = None
        chunk_sw = self.column_chunk or _sw.SW_BATCH_COLUMN_CHUNK
        for c0 in range(0, day_idx.size, chunk_sw):
            idx = day_idx[c0:c0 + chunk_sw]
            ps = _prep.swrad_prep_batch(
                **chunk_inputs(idx), albedo=surf["albedo"][idx],
                xcoszen=coszen[idx], solcon=solcon,
                sf_surface_physics=sf_surface_physics,
                subcolumn_generator=self._mcica_generator(
                    _mcica.gpu_generate_sw_subcolumns),
                **shared)
            nc = int(ps["ncol"])
            adjes = np.full(nc, ps["adjes"], np.float32)
            res = self._cuda_sw.rrtmg_sw_batched(
                nc, ps["nlay"], ps["icld"], ps["play"], ps["plev"],
                ps["tlay"], ps["tlev"], ps["tsfc"], ps["h2ovmr"],
                ps["o3vmr"], ps["co2vmr"], ps["ch4vmr"], ps["n2ovmr"],
                ps["o2vmr"], ps["asdir"], ps["asdif"], ps["aldir"],
                ps["aldif"], ps["coszen"], adjes, int(ps["dyofyr"]),
                ps["scon"], ps["inflgsw"], ps["iceflgsw"],
                ps["liqflgsw"], ps["cldfmcl"], ps["taucmcl"],
                ps["ssacmcl"], ps["asmcmcl"], ps["fsfcmcl"],
                ps["ciwpmcl"], ps["clwpmcl"], ps["cswpmcl"],
                ps["reicmcl"], ps["relqmcl"], ps["resnmcl"], aer_opt=0,
                column_chunk=chunk_sw, _stage_probe=self._stage_probe)
            del ps
            # Column-vectorized swrad_option4_outputs for exactly the
            # fields the driver contract consumes: each line is the same
            # single-rounded FP32 op sequence as the scalar mapping, one
            # column per lane (bitwise-equal per column).
            gsw[idx] = (res["swdflx"][:, 0]
                        - res["swuflx"][:, 0]).astype(np.float32)
            tten = (res["swhr"][:, :nz] / F(86400.0)).astype(np.float32)
            rthratensw[idx] = (tten / pi3d[idx]).astype(np.float32)
            del res

        # ---- driver-level SWDOWN = GSW/(1-ALBEDO) (driver line 2877) --
        swdown = (gsw / (F(1.0) - surf["albedo"]).astype(
            np.float32)).astype(np.float32)

        self.update_count += 1
        return RadiationResult(
            rthratenlw=self._grid3(rthratenlw, nz, ny, nx),
            rthratensw=self._grid3(rthratensw, nz, ny, nx),
            swdown=self._grid2(swdown, ny, nx),
            glw=self._grid2(glw, ny, nx),
            gsw=self._grid2(gsw, ny, nx),
            coszen=self._grid2(coszen, ny, nx),
            olr=self._grid2(olr, ny, nx))
