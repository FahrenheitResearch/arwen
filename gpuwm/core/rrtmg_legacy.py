# ======================================================================
# THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
# third-party work.  ArWen distributes the file under the Apache License
# 2.0; the notices below belong to the transcribed parts and are kept here
# because their own licences require it.  Full texts are in the repository
# NOTICE and in the licenses/ directory.
#
#   RRTMG longwave/shortwave, transcribed from WRF v4.6.1
#   phys/module_ra_rrtmg_lw.F and phys/module_ra_rrtmg_sw.F, which carry
#   AER's own notice seven and nine times respectively:
#
#       Copyright 2002-2008, Atmospheric & Environmental Research, Inc. (AER).
#       This software may be used, copied, or redistributed as long as it is
#       not sold and this copyright notice is reproduced on each copy made.
#       This model is provided as is without any express or implied warranties.
#                             (http://www.rtweb.aer.com/)
#
#   ArWen takes this material under AER's own current grant instead: BSD
#   3-Clause, "Copyright (c) 2020, Atmospheric and Environmental
#   Research", published by AER at github.com/AER-RC/RRTMG_LW and
#   .../RRTMG_SW.  Text in licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt and
#   beside the packaged coefficients in gpuwm/data/wrf_radiation/.
# ======================================================================
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

Ozone routing: WRF v4.7.1's EM_CORE guards BOTH halves of the
``o3input = 2`` chain on ``id == 1`` -- ``oznini`` interpolates the
packaged climatology to a domain's own XLAT under
``o3input .EQ. 2 .AND. id .EQ. 1`` (module_physics_init.F:2203-2212), and
``ozn_time_int``/``ozn_p_int`` run under the same guard
(module_radiation_driver.F:1801-1823).  A NEST evaluates neither; it
receives the root's field through the parent-to-child forcing stream the
Registry declares on the variable itself (``rdf=(p2c)``,
Registry/Registry.EM_COMMON:1264).  The decision is "is this domain a
nest?", not "is this grid fine?".

The adapter mirrors that structure and takes its word for what it did
from :mod:`gpuwm.core.cam_ozone`:

* A ROOT adapter (``ozone_parent=None``) runs the bit-ported
  ``gpuwm.ingest.wrf_ozone`` chain on its own grid and RETAINS the
  resulting field: ``ozone_routing = 'root-climatology'``.
* A resident NEST takes an ``ozone_parent=`` provider
  (:class:`ParentOzoneProvider`, wired by ``runtime.prepare_child_case``)
  and obtains the parent's most recent retained field horizontally
  interpolated onto the child grid through gpuwm's certified SINT
  mass-point operator (``gpuwm.core.nest_interp.sint``); it never invokes
  the climatology chain.  ``ozone_routing = 'parent-interpolated'``.  The
  horizontal-interpolation ARITHMETIC is gpuwm's own SINT
  transliteration, the same documented seam class as every other
  nest-interpolation arithmetic difference; the routing (root-compute +
  parent->child interpolation, the child's field updating when the
  parent's does) matches WRF.
* An OFFLINE child (``gpuwm.offline_child_run``, the ndown-equivalent
  route) has no resident parent to interpolate from and is configured as
  a WRF root: ``specified = true``, ``nested = false``, ``parent_id = 0``,
  lateral boundaries read from a file.  WRF's own answer for such a
  domain is the climatology on its own grid, so the route builds a root
  adapter and declares ``ozone_routing = 'child-grid-climatology'`` -- the
  same arithmetic, named so a report says the ozone came from the
  climatology on the child grid and not from the parent.

SW aerosol: with ``aer_opt = 0`` the batched CUDA SW engine builds WRF's
neutral optics (tauaer 0 / ssaaer 1 / asmaer 0, module_ra_rrtmg_sw.F:
11333-11460) internally, exactly as before.  With ``aer_opt = 3`` (the
operational HRRR fork's Thompson aerosol branch) the adapter forms the
per-band optics on the device from the radiation inputs and the Thompson
numbers state.nwfa/state.nifa for each day-column chunk
(gpuwm.core.rrtmg_aerosol_optics, held word for word to the fork's
gt_aod and calc_aerosol_rrtmg_sw) and hands them to the engine, which
feeds them to spcvmc unchanged (iaer = 10).  aer_opt 1 and 2 stay
refused (SW-audit item 1).  The longwave takes no aerosol in that fork
under any aer_opt without chemistry.

Sequencing/VRAM shape: the LW and SW pipelines run SEQUENTIALLY per
adapter chunk -- each chunk's prep outputs, device McICA slabs, and
engine transients are dropped before the next stage allocates -- so the
transient device peak is the MAX over the four phases (LW generate, LW
engine, SW generate, SW engine) plus the McICA slabs held as engine
inputs, never the LW+SW sum.  :func:`legacy_radiation_vram_bytes` prices
exactly that shape.
"""

from __future__ import annotations

from gpuwm.core.device_cache import cuda_cache

from datetime import datetime, timedelta
from functools import lru_cache, partial
from pathlib import Path

import numpy as np

from gpuwm.core.mynn_radiation import (
    MERGE_CLDFRA_BL_ABOVE,
    MERGE_QC_BELOW,
    MERGE_QI_BELOW,
    mynn_bl_cloud_active,
    wrf_itimestep,
)

from gpuwm.core import rrtmg_legacy_prep as _prep
from gpuwm.core import rrtmg_legacy_device as _prep_device
from gpuwm.core import rrtmg_lw as _lw
from gpuwm.core import rrtmg_mcica as _mcica
from gpuwm.core import rrtmg_sw as _sw
from gpuwm.core import rrtmg_aerosol_optics as _aer3

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
#: swint_opt is not here: it is a radiation-DRIVER option the physics
#: driver carries (gpuwm.core.swint) and the adapter's call is the same
#: for 0 and 1.  aer_opt = 3 (Thompson aerosol shortwave optics) is
#: implemented by this adapter; validate_radiation_driver_options admits
#: 0 and 3 and refuses 1 and 2 by name before a run starts.
_PINNED_OPTIONS = {"icloud": 1, "cldovrlp": 2, "idcor": 0,
                   "ghg_input": 0}
_ADMITTED_AER_OPT = (0, 3)


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
    return calc_coszen_hrang(julian, xtime, gmt, xlat, xlon, declin)[0]


def calc_coszen_hrang(julian, xtime, gmt, xlat, xlon, declin):
    """WRF ``calc_coszen``'s two outputs, ``(coszen, hrang)``, float32.

    HRANG is the hour angle WRF stores beside COSZEN (grid%hrang,
    module_radiation_driver.F:3534); the slope-radiation adjustment reads
    both (module_surface_driver.F TOPO_RAD_ADJ).
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
    return (np.minimum(np.maximum(coszen, F(-1.0)), F(1.0)).astype(np.float32),
            np.asarray(hrang, dtype=np.float32))


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
# VRAM pricing (model preflight + the wiring accuracy gate).
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


def _require_agreement_with_the_registry() -> None:
    """``_MP_DECLARES_RADII`` is the registry's ``cloud_optics.legacy_declares_radii``.

    Pulled by tools/build_registry.py and held equal here at import, so
    a scheme with no row is a plan-review refusal rather than the
    NotImplementedError :func:`legacy_scheme_declares_radii` raises at
    the first radiation call.
    """

    from gpuwm.physics_registry import require_consumer_rows_agreement

    require_consumer_rows_agreement(
        "gpuwm.core.rrtmg_legacy._MP_DECLARES_RADII", "microphysics",
        "cloud_optics", dict(_MP_DECLARES_RADII),
        project=lambda row: row["legacy_declares_radii"])


_require_agreement_with_the_registry()


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
    """Apply the RRTMG wrapper's micron-to-meter conversion exactly once.

    A float32 CuPy array is scaled by the glue unit's unflushed multiply
    (radii can be FP32 subnormals, which a CuPy ufunc would flush); the
    result is the NumPy expression's bits either way."""

    x = effective_radius_microns
    if type(x).__module__.split(".")[0] == "cupy" and x.dtype == np.float32:
        import cupy as cp
        adapter_gpu_preflight()
        x = cp.ascontiguousarray(x)
        out = cp.empty_like(x)
        _adapter_kernel("rla_scale")(
            *_grid_launch(x.size), (np.int64(x.size), x, F(1.0e-6), out))
        return out
    return (x * F(1.0e-6)).astype(np.float32)


def unsized_mynn_radii(re_cloud, re_ice, supplied_liquid, supplied_ice, *,
                       ice_rule):
    """Hand WRF's cloudy-layer rule the layers MYNN gave water as unsized.

    ``supplied_*`` come from
    :func:`gpuwm.core.mynn_radiation.mynn_bl_cloud_supplied`.  There the
    scheme's radius is its no-cloud background or the size of a trace of its
    own condensate, not a size for MYNN's water, and WRF's radius tests (2.5
    um liquid, 5 um ice) miss NSSL's background (2.51 um, 10.01 um),
    Thompson's 2.51 um liquid floor and trace Thompson ice sized above 5 um.  A zero radius is what the
    rule sizes: ``max(2.5, 0)`` sits on its liquid bound and ``max(5, 0)``
    on its ice one, so these layers take 7.5 um over land, 10.5 um over
    water and the ice temperature table, exactly as RTE+RRTMGP gives them
    (gpuwm.core.rrtmgp.cloudy_background_radii).  ``ice_rule`` is False for
    P3, whose ice WRF radiates as snow at P3's own radius.  Radii are metres
    or ``None`` (a scheme that declares none); new arrays are returned.
    """

    if supplied_liquid is None:
        return re_cloud, re_ice
    zero = F(0.0)
    if re_cloud is not None:
        re_cloud = np.where(supplied_liquid, zero, re_cloud).astype(
            np.float32)
    if re_ice is not None and ice_rule:
        re_ice = np.where(supplied_ice, zero, re_ice).astype(np.float32)
    return re_cloud, re_ice


def _r512(nbytes):
    """CuPy's 512-byte pool allocation quantum."""
    return (int(nbytes) + 511) & ~511


@lru_cache(maxsize=1)
def legacy_shortwave_constant_bytes():
    """The three persistent CudaSW uploads, including pool rounding.

    The packed coefficient array, g-point band indices and three band
    metadata rows remain resident through both radiation spectra. This
    uses the runtime's table packer without constructing a CUDA engine.
    """
    tables = _sw_tables()
    packed, _ = _sw._pack_cuda_tables(tables)
    return (_r512(packed.nbytes) + _r512(np.asarray(tables.ngb).size * 4)
            + _r512(3 * _sw.NBNDSW * 4))


def legacy_radiation_vram_bytes(*, ncol, nz, p_top, column_chunk=None,
                                ncol_day=None, lw_coefficients=None,
                                longwave=True, shortwave=True,
                                resident_threads=None, o3input=2,
                                aer_opt=0):
    """Peak transient device bytes of ONE adapter call.

    Composes the engines' own accurate pricing functions
    (``lw_batched_vram_bytes`` + ``lw_batched_const_bytes``,
    ``sw_batched_vram_bytes``) and the McICA device pricing
    (``mcica_device_vram_bytes``) with the adapter-held device arrays the
    engine functions do not price: the McICA output slabs and radii
    pass-throughs that stay alive as engine inputs.  The LW and SW
    pipelines run sequentially per adapter chunk with chunk storage freed
    in between, so the estimate is the max over the four allocation
    phases (LW generate, LW engine, SW generate, SW engine). Both engines'
    immutable coefficients remain resident across those phases and are
    added once outside the maximum. ``ncol_day`` bounds the SW day-column count (default:
    ``ncol``, the preflight upper bound).

    Whole-call result storage and the reusable ozone grid are priced
    separately. SWDOWN is allocated after the engines.

    ``resident_threads`` lets a host-side estimator use its already measured
    device profile without opening a CUDA context. Zero means an unknown
    device and prices the unchanged workspace ceilings; None retains the
    engines' live-device behavior for existing direct callers.

    ``aer_opt = 3`` adds what the SW chunk holds for the aerosol: the
    three (14 x nlay) optics slabs alive through the engine call, the
    layer AOD and the two gathered Thompson number blocks (an upper bound:
    the engine's own zero-aerosol constants are priced by the engine
    function and not allocated on this path).
    """
    ncol = int(ncol)
    nz = int(nz)
    nlay_lw = _prep.compute_lw_nlayers(nz + 1, p_top)
    nlay_sw = nz + 1
    nday = ncol if ncol_day is None else int(ncol_day)
    chunk = None if column_chunk is None else int(column_chunk)
    nc_lw = min(chunk or _lw.batch_column_chunk(
        _lw.NGPTLW, _lw.LW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=resident_threads), ncol)
    # Host-side pricing passes no free-VRAM figure, so this is the
    # saturation width: an upper bound on the width a device run picks.
    nc_sw = min(chunk or _sw.sw_batch_column_chunk(
        nlay_sw, resident_threads=resident_threads), max(nday, 0))
    f = 4
    estimate = 0
    constants = legacy_shortwave_constant_bytes() if shortwave else 0
    if longwave:
        C = lw_coefficients if lw_coefficients is not None else _lw_coeffs()
        constants += _lw.lw_batched_const_bytes(C)

        s_mcl = _r512(nc_lw * _lw.NGPTLW * nlay_lw * f)
        s_nl = _r512(nc_lw * nlay_lw * f)
        held_lw = 5 * s_mcl + 3 * s_nl        # mcl slabs + rei/rel/res
        # The device-resident prep outputs the engine reads, alive for the
        # whole engine call: play tlay and the ten gas profiles (nlay),
        # plev tlev (nlay+1), emis (16), tauaer (nlay x 16), tsfc.
        held_lw += (12 * s_nl + 2 * _r512(nc_lw * (nlay_lw + 1) * f)
                    + _r512(nc_lw * _lw.NBNDLW * f)
                    + _r512(nc_lw * nlay_lw * _lw.NBNDLW * f)
                    + _r512(nc_lw * f))
        lw_gen = (held_lw + 5 * s_nl          # play cldfrac ciwp clwp cswp
                  + _r512(_mcica.NBNDLW * nc_lw * nlay_lw * f)   # tauc
                  + _mcica.mcica_device_vram_bytes(
                      min(nc_lw, _mcica.MCICA_DEVICE_COLUMN_CHUNK),
                      nlay_lw, _lw.NGPTLW))
        lw_eng = held_lw + _lw.lw_batched_vram_bytes(
            nc_lw, nlay_lw, mcica_layout="column")
        estimate = max(lw_gen, lw_eng)

    if shortwave and nc_sw:
        s_mcl_s = _r512(nc_sw * _sw.NGPTSW * nlay_sw * f)
        s_nl_s = _r512(nc_sw * nlay_sw * f)
        held_sw = 8 * s_mcl_s + 3 * s_nl_s
        # The device-resident prep outputs the engine reads: play tlay and
        # six gas profiles (nlay), plev tlev (nlay+1), and the per-column
        # tsfc, four albedos, coszen and scon.
        held_sw += (8 * s_nl_s + 2 * _r512(nc_sw * (nlay_sw + 1) * f)
                    + 7 * _r512(nc_sw * f))
        if int(aer_opt) == 3:
            held_sw += (3 * _r512(nc_sw * _sw.NBNDSW * nlay_sw * f)
                        + 3 * _r512(nc_sw * nz * f))
        sw_gen = (held_sw + 5 * s_nl_s
                  + 4 * _r512(_mcica.NBNDSW * nc_sw * nlay_sw * f)
                  + _mcica.mcica_device_vram_bytes(
                      min(nc_sw, _mcica.MCICA_DEVICE_COLUMN_CHUNK),
                      nlay_sw, _sw.NGPTSW))
        sw_eng = held_sw + _sw.sw_batched_vram_bytes(nc_sw, nlay_sw, mcica_layout="column")
        estimate = max(estimate, sw_gen, sw_eng)
    # Whole-call storage is separate from the per-chunk McICA terms.
    # SWDOWN is allocated after the engines and does not raise this peak.
    result_bytes = 2 * _r512(nz * ncol * f) + 3 * _r512(ncol * f)
    ozone_bytes = _r512(nz * ncol * f) if int(o3input) == 2 else 0
    return estimate + result_bytes + ozone_bytes + constants


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


def _cuda_sw_device() -> int:
    """The card a legacy SW engine is being asked for (the current one)."""
    import cupy as cp
    return int(cp.cuda.Device().id)


def _cuda_sw(tab):
    """The compiled CUDA SW engine for ``tab`` on this card, once per card.

    Construction is the expensive half of this adapter: ``CudaSW.__init__``
    runs a full NVRTC compile of ``kernels/rrtmg_sw.cu`` and uploads the
    packed tables.  Streaming builds one adapter per TILE BUFFER and the
    [devices] split one per SLAB, so an uncached engine would multiply both
    the compile and the resident tables.

    PER CARD, because a compiled module and its tables live on one card and
    a slab on another card must not launch them.  The engine's writable
    batched workspace is per STREAM inside the engine (``CudaSW._scratch``),
    so slabs stepping at once on one card share the read-only module and
    tables and never the workspace.  Keyed on the tables OBJECT, held in the
    entry so its id cannot be reused: a caller assembling its own tables
    gets its own engine rather than borrowing one built from different
    coefficients.
    """
    global _CUDA_SW_CACHE
    if _CUDA_SW_CACHE is None:
        _CUDA_SW_CACHE = {}
    key = (_cuda_sw_device(), id(tab))
    hit = _CUDA_SW_CACHE.get(key)
    if hit is None:
        hit = _CUDA_SW_CACHE.setdefault(key, (tab, _sw.CudaSW(tab)))
    return hit[1]


# ---------------------------------------------------------------------------
# Device-resident adapter glue (kernels/rrtmg_legacy_adapter.cu).
#
# The adapter used to download every field, pack host columns, run the
# wrapper glue in NumPy and upload the results.  The kernels below are the
# same float32 statements on the device, in the same order, so the call
# returns the same bits without the round trips.  The unit compiles through
# NVRTC directly with --ftz=false and writes its arithmetic as unflushed PTX
# (see the .cu header); adapter_gpu_preflight proves both on the live device.
# ---------------------------------------------------------------------------

_ADAPTER_MODULE = {}
_ADAPTER_KERNELS = {}
_ADAPTER_PREFLIGHTED = set()
_ADAPTER_BLOCK = 256


@cuda_cache(maxsize=None)
def _adapter_module():
    """Compile kernels/rrtmg_legacy_adapter.cu via NVRTC directly (the
    route and the missing ``-arch`` follow ``rrtmg_lw._gpu_module``)."""
    import cupy as cp
    device = int(cp.cuda.Device().id)
    if device not in _ADAPTER_MODULE:
        import cupy as cp
        from gpuwm import nvrtc_ptx_cache as _cc
        path = (Path(__file__).resolve().parent / "kernels"
                / "rrtmg_legacy_adapter.cu")
        ptx, _mapping = _cc.compile_using_nvrtc(
            path.read_text(encoding="utf-8"),
            ("-std=c++17", "--ftz=false"),
            None, "rrtmg_legacy_adapter.cu")
        mod = cp.cuda.function.Module()
        mod.load(ptx.encode() if isinstance(ptx, str) else ptx)
        _ADAPTER_MODULE[device] = mod
    return _ADAPTER_MODULE[device]


@cuda_cache(maxsize=None)
def _adapter_kernel(name):
    import cupy as cp
    key = (int(cp.cuda.Device().id), name)
    if key not in _ADAPTER_KERNELS:
        _ADAPTER_KERNELS[key] = _adapter_module().get_function(name)
    return _ADAPTER_KERNELS[key]


def adapter_gpu_preflight(force=False):
    """Prove on the live device that the glue unit keeps FP32 subnormals:
    a subnormal product and quotient survive and a subnormal compares
    greater than zero, bitwise equal to NumPy."""
    import cupy as cp
    device = int(cp.cuda.Device().id)
    if device in _ADAPTER_PREFLIGHTED and not force:
        return
    import cupy as cp
    x = np.array([1.0e-30, 1.0e-10, 1.0e-40, 1.0e-38, 4.0], np.float32)
    o = cp.zeros(3, dtype=cp.float32)
    _adapter_kernel("rla_probe")((1,), (32,), (cp.asarray(x), o))
    got = cp.asnumpy(o)
    want = np.array([x[0] * x[1], 1.0 if x[2] > F(0.0) else 0.0,
                     x[3] / x[4]], np.float32)
    if got.view(np.uint32).tolist() != want.view(np.uint32).tolist():
        raise RuntimeError(
            "rrtmg_legacy adapter GPU preflight failed: got %r want %r "
            "(subnormal flush on this toolchain)" % (got, want))
    _ADAPTER_PREFLIGHTED.add(device)


#: Prep outputs that are WRF-level extras, not engine inputs; the device
#: adapter drops them before the engine call so they are not held through
#: the engine phase (see legacy_radiation_vram_bytes).
_LW_PREP_EXTRAS = ("o31d", "hgt", "cldfrac", "pdel")
_SW_PREP_EXTRAS = ("o31d", "pdel", "coszr", "mcica_inputs")


def _grid_launch(n):
    return ((int(n) + _ADAPTER_BLOCK - 1) // _ADAPTER_BLOCK,), (_ADAPTER_BLOCK,)


def _device_grid(a, nk):
    """(nk, ny, nx) field -> device float32 (nk, ncol), C-contiguous.

    A float32 device field is reshaped in place (data movement only);
    anything else is cast on the host with NumPy semantics first, exactly
    as the host packing did."""
    import cupy as cp
    if isinstance(a, cp.ndarray) and a.dtype == cp.float32:
        return cp.ascontiguousarray(a.reshape(nk, -1))
    h = np.asarray(RRTMGLegacyRadiation._host(a), np.float32)
    return cp.asarray(np.ascontiguousarray(h.reshape(nk, -1)))


def _device_flat(a):
    """(ny, nx) field -> device float32 (ncol,)."""
    import cupy as cp
    if isinstance(a, cp.ndarray) and a.dtype == cp.float32:
        return cp.ascontiguousarray(a.reshape(-1))
    h = np.asarray(RRTMGLegacyRadiation._host(a), np.float32)
    return cp.asarray(np.ascontiguousarray(h.reshape(-1)))


def _device_columns(grid, sel):
    """Rows ``sel`` (a slice or a device index array over columns) of a
    (nk, ncol) device grid as a C-contiguous (nc, nk) column block."""
    import cupy as cp
    return cp.ascontiguousarray(grid[:, sel].T)


class _ColumnGather:
    """Column blocks of several (nk, ncol) device grids, one launch a chunk.

    Built once per radiation call over the grids the chunks read; each call
    returns (nc, nk) C-contiguous blocks that are views of one packed buffer
    allocated for that chunk (so, like the separate gathers it replaces, it
    lives only while the chunk's prep inputs do).  Data movement only."""

    def __init__(self, grids, ncol):
        import cupy as cp
        self.cp = cp
        self.grids = list(grids)
        self.ncol = int(ncol)
        self.nk = [int(g.shape[0]) for g in self.grids]
        for g in self.grids:
            if g.dtype != cp.float32 or not g.flags.c_contiguous or                     g.shape[1] != self.ncol:
                raise ValueError("gathered grids must be C-contiguous float32 "
                                 "(nk, ncol) device arrays")
        self.src = cp.asarray(np.asarray([g.data.ptr for g in self.grids],
                                         dtype=np.uint64))
        self.nk_d = cp.asarray(np.asarray(self.nk, dtype=np.int32))

    def __call__(self, sel):
        cp = self.cp
        if isinstance(sel, slice):
            c0, nc, idx = int(sel.start), int(sel.stop) - int(sel.start), None
        else:
            c0, nc, idx = 0, int(sel.size), sel
        total = nc * sum(self.nk)
        buf = cp.empty(total, dtype=cp.float32)
        _adapter_kernel("rla_gather")(
            *_grid_launch(total),
            (np.int32(len(self.grids)), np.int32(nc), np.int64(self.ncol),
             np.int64(c0), np.uint64(0) if idx is None else idx, self.src,
             self.nk_d, buf))
        out, base = [], 0
        for nk in self.nk:
            out.append(buf[base:base + nc * nk].reshape(nc, nk))
            base += nc * nk
        return out


def _radius_meters_blocks(blocks):
    """Convert the present radius blocks in one unflushed launch."""
    import cupy as cp
    present = [(key, block) for key, block in blocks.items()
               if block is not None]
    result = dict.fromkeys(blocks)
    if not present:
        return result
    n = present[0][1].size
    buf = cp.empty(n * len(present), dtype=cp.float32)
    pointers = [block for key, block in present]
    pointers += [np.uint64(0)] * (3 - len(pointers))
    _adapter_kernel("rla_scale_radii")(
        *_grid_launch(buf.size),
        (np.int64(n), np.int32(len(present)), *pointers, F(1.0e-6), buf))
    for i, (key, block) in enumerate(present):
        result[key] = buf[i*n:(i+1)*n].reshape(block.shape)
    return result


def _f32_positive(a):
    """``a > 0`` for float32 ``a`` decided on the bits, so a subnormal
    counts as positive whatever the compile route of the comparison."""
    import cupy as cp
    bits = a.view(cp.int32)
    return (bits > 0) & (bits <= 0x7F800000)


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
        self.released = False

    def release(self) -> None:
        """Drop the parent adapter: the build this provider feeds is superseded.

        A nest waiting for a delayed start holds a startup build whose
        radiation reads its PARENT's adapter through this provider.  When
        the parent is itself a delayed nest and activates, its startup
        adapter is replaced, and this link was the one reference that kept
        the outgoing adapter (and what it holds) alive until the waiting
        nest activated in turn.  The waiting nest is rebuilt at its own
        start against the parent's new adapter, so the startup build never
        radiates through this provider again.
        """
        self.parent = None
        self.released = True

    def __call__(self):
        if self.released:
            raise RuntimeError(
                "child rrtmg_legacy radiation called through an ozone "
                "provider released with its superseded startup build; a "
                "nest waiting for its start is rebuilt against its "
                "parent's current radiation before it radiates")
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


def release_waiting_ozone_links(outgoing, waiting) -> None:
    """Unlink nests waiting for their start from a parent's outgoing radiation.

    A nest's ozone reads its parent's radiation through a
    :class:`ParentOzoneProvider`, so a child of a delayed nest that has not
    started yet holds the parent's STARTUP radiation through its own
    startup build.  When that parent activates and its startup build is
    released, this link kept the outgoing radiation (and everything it
    holds) alive until the child activated as well.  ``outgoing`` is the
    parent's startup ``radiation_callable`` and ``waiting`` its child nodes
    that have not started.  A waiting child's startup build never radiates
    before it starts, and it is rebuilt against the parent's new radiation
    then, so each of its providers that reads ``outgoing`` is released.
    Every other provider, a started child's and the CAM carrier's
    :class:`gpuwm.core.cam_ozone.DriverOzoneProvider` among them, is left
    alone.

    A waiting child whose state is a
    :class:`gpuwm.core.streamed_state.CanonicalStoreState` (a nest the
    prepared domain-tree forecast streams through a host store) is
    skipped: that view refuses a read of its radiation with
    ``CanonicalStateRefused``, which stopped the run at the parent's
    start, and that route builds a nest's startup radiation from its
    prepared cache with no provider of this class, so there is no link to
    release.
    """
    from gpuwm.core.radiation_composition import radiation_adapters
    from gpuwm.core.streamed_state import CanonicalStoreState

    for child in waiting:
        state = getattr(child, "state", None)
        if isinstance(state, CanonicalStoreState):
            continue
        physics = getattr(state, "physics", None)
        radiation = getattr(physics, "radiation_callable", None)
        for adapter in radiation_adapters(radiation):
            provider = getattr(adapter, "_ozone_provider", None)
            if (isinstance(provider, ParentOzoneProvider)
                    and provider.parent is outgoing):
                provider.release()


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

    # No latitude-derived ozone is retained. The common tile gather and
    # moving-grid routes change latitude in place, so every radiation call
    # interpolates the CAM climatology from the live latitude_deg
    # (wrf_ozone.ozn_latitude_time_int, two months per call).

    def __init__(self, start_time, latitude_deg, longitude_deg, *,
                 p_top=None, column_chunk=None, ozone_parent=None,
                 o3input=2, longwave=True, shortwave=True, trace_gas_overrides=None,
                 ozone_routing=None, aer_opt=0, smoke_provider=None):
        if not isinstance(start_time, datetime):
            raise TypeError("radiation_start_time must be a datetime")
        self.start_time = start_time
        self.longwave = bool(longwave)
        self.shortwave = bool(shortwave)
        if not (self.longwave or self.shortwave):
            raise ValueError("radiation adapter needs at least one spectrum")
        #: WRF aer_opt for the shortwave this adapter runs: 0 (zero
        #: aerosol) or 3 (the Thompson aerosol optics,
        #: gpuwm.core.rrtmg_aerosol_optics).  The longwave of the
        #: operational HRRR fork takes no aerosol under aer_opt = 3
        #: (module_ra_rrtmg_lw.F:12538-12560 fills tauaer from chemistry
        #: only), so a longwave-only adapter carries 0.
        self.aer_opt = int(aer_opt)
        if self.aer_opt not in _ADMITTED_AER_OPT:
            raise NotImplementedError(
                f"rrtmg_legacy implements aer_opt in {_ADMITTED_AER_OPT}; "
                f"got aer_opt={aer_opt!r}")
        if self.aer_opt and not self.shortwave:
            raise ValueError("aer_opt=3 acts on the shortwave; a "
                             "longwave-only adapter carries aer_opt=0")
        if smoke_provider is not None and not (self.shortwave and self.aer_opt == 3):
            raise ValueError("prescribed smoke requires legacy shortwave with aer_opt=3")
        self._smoke_provider = smoke_provider
        self.publishes_olr = self.longwave
        from gpuwm.core.trace_gases import (
            LEGACY_LW_GASES, LEGACY_SW_GASES, validate_trace_gas_overrides)
        supported = ((LEGACY_LW_GASES if self.longwave else frozenset())
                     | (LEGACY_SW_GASES if self.shortwave else frozenset()))
        self.trace_gas_overrides = (None if trace_gas_overrides is None else
            validate_trace_gas_overrides(trace_gas_overrides, supported=supported,
                                         consumer="selected legacy RRTMG spectra"))
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
        #: How this adapter obtains ozone, in the one vocabulary
        #: gpuwm.core.cam_ozone owns.  Derived from the construction unless
        #: the route declares a name the construction admits, which is how
        #: the offline child route says "the climatology, on the child's own
        #: grid" rather than being read as a root.
        from gpuwm.core.cam_ozone import resolve_ozone_routing
        self.ozone_routing = resolve_ozone_routing(
            ozone_routing, o3input=self.o3input,
            has_parent=ozone_parent is not None)
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
            self._C = _lw_coeffs() if self.longwave else None
        except Exception as exc:
            raise RuntimeError(
                "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                "packaged RRTMG_LW_DATA coefficients cannot be "
                f"loaded/built: {exc}") from exc
        try:
            self._sw_tables = _sw_tables() if self.shortwave else None
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
        elif self._ozone_provider is None:
            # Root routing: the climatology chain runs here, on this
            # adapter's own grid.  WRF evaluates it for id == 1, which is
            # both a resident root and an offline child (that route's
            # domain is configured as a root and stamps parent_id = 0);
            # self.ozone_routing says which of the two this is.  The
            # latitude interpolation (WRF's oznini-time work) runs at each
            # call on the live latitude, for the two months that call uses.
            try:
                from gpuwm.ingest import wrf_ozone as _ozone
                self._ozone = _ozone
                self._ozone_climo = _ozone.load_ozone_climatology()
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
        try:
            self._cuda_sw = (_cuda_sw(self._sw_tables)
                             if self.shortwave else None)
        except Exception as exc:
            raise RuntimeError(
                "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                "CUDA SW engine cannot be compiled/loaded on this "
                f"installation (cupy + a CUDA device are required): {exc}"
            ) from exc
        try:
            if self.longwave:
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
            "aer_opt": self.aer_opt,
            "column_chunk": self.column_chunk,
            "p_top": self.p_top,
            "ozone_routing": self.ozone_routing,
            "statics_assets": {
                "rrtmg_lw_statics.npz": RRTMG_LW_STATICS_SHA256,
            },
        }
        if self.aer_opt == 3:
            from gpuwm.core.rrtmg_aerosol_optics import AER3_TABLES_SHA256
            identity["aer3_tables_sha256"] = AER3_TABLES_SHA256
        if getattr(self, "_smoke_provider", None) is not None:
            identity["prescribed_smoke"] = self._smoke_provider.identity()
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
        if not (self.longwave and self.shortwave):
            identity["spectra"] = {"lw": self.longwave, "sw": self.shortwave}
        if self.trace_gas_overrides:
            identity["trace_gas_overrides"] = dict(self.trace_gas_overrides)
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
        aer_opt = int(getattr(cfg, "aer_opt", 0))
        if aer_opt not in _ADMITTED_AER_OPT:
            raise NotImplementedError(
                f"rrtmg_legacy implements aer_opt in {_ADMITTED_AER_OPT} "
                f"(0: no aerosol; 3: Thompson water- and ice-friendly "
                f"aerosol shortwave optics); got aer_opt={aer_opt!r} -- no "
                "silent option substitution is applied")
        if self.shortwave and aer_opt != self.aer_opt:
            raise ValueError(
                f"adapter was constructed for aer_opt={self.aer_opt} but the "
                f"run config requests aer_opt={aer_opt}; rebuild the adapter "
                "so its shortwave aerosol and restart identity match")
        if self.shortwave and bool(getattr(cfg, "rrtmg_smoke_manifest", "")) != (
                getattr(self, "_smoke_provider", None) is not None):
            raise ValueError("prescribed smoke configuration differs from the bound radiation adapter")
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

    def _validate_radii_micron_device(self, name, eff_um, q):
        """:meth:`_validate_radii_micron` on (nz, ncol) device grids: the
        same mask (``q > 0`` and ``eff > 0``, decided on the float32 bits
        so a subnormal counts) and the same verdict, with one host read.
        An absent species (``q is None``) is the host path's zero column,
        whose mask is empty."""
        import cupy as cp
        if q is None:
            return
        mask = _f32_positive(q) & _f32_positive(eff_um)
        top = float(cp.where(mask, eff_um, cp.float32(-np.inf)).max())
        if top != -np.inf and top < 1.0e-3:
            raise ValueError(
                f"state.{name} looks meter-scale (every value on cloudy "
                f"points is < 1e-3, max {top:.3e}) "
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
        if self.longwave and nlayers > MAX_LONGWAVE_LAYERS:
            raise ValueError(
                f"LW nlayers={nlayers} exceeds the batched engine's "
                f"{MAX_LONGWAVE_LAYERS}-layer bound "
                "(rlw_rtrn_march RLW_MAXLAY)")

        mp_physics = int(getattr(cfg, "mp_physics", 0))
        warm_rain = mp_physics == 1
        # cal_cldfra1's two Registry package flags, resolved SEPARATELY.
        # One fused boolean cannot express P3's F_QI=true/F_QS=false, and
        # collapsing it sent mp=50 to the qc-only arm.
        f_qi, f_qs = legacy_cloud_fraction_flags(mp_physics)
        sf_surface_physics = int(getattr(cfg, "sf_surface_physics", 2))

        adapter_gpu_preflight()
        blk = _grid_launch

        # ---- device grids: every (nk, ny, nx) field as (nk, ncol) on the
        # device; column blocks are gathered per chunk below (data
        # movement only), so no field makes a host round trip. -----------
        g_p3d = _device_grid(pressure, nz)
        g_p8w = _device_grid(atmosphere["p_interface"], nz + 1)
        g_t3d = _device_grid(atmosphere["temperature"], nz)
        g_pi3d = _device_grid(atmosphere["exner"], nz)
        g_dz8w = _device_grid(atmosphere["dz"], nz)
        g_zw = _device_grid(atmosphere["z_interface"], nz + 1)
        smoke_gather = None
        smoke_provider = getattr(self, "_smoke_provider", None)
        if smoke_provider is not None:
            profile = smoke_provider.at(float(state.elapsed_seconds),
                latitude_deg=self.latitude_deg, longitude_deg=self.longitude_deg)
            quantity = profile["quantity"]
            if quantity == "layer_aod":
                smoke_aod = profile["value"]
            else:
                smoke_mass = profile["value"]
                if quantity == "posted_mass_concentration":
                    smoke_mass = _aer3.smoke_dry_mixing_ratio_from_posted_density_device(
                        smoke_mass, profile["donor_p"], profile["donor_t"])
                # ALT is dry specific volume. The shared atmosphere rho
                # includes vapor mass and is not the source dry density.
                dry_density = _aer3.smoke_dry_air_density_device(state.alt)
                smoke_aod = _aer3.smoke_aod_from_dry_mixing_ratio_device(
                    smoke_mass, dry_density, atmosphere["dz"])
                del smoke_mass, dry_density
            smoke_gather = _ColumnGather((_device_grid(smoke_aod, nz),), ncol)
            del profile, smoke_aod

        g_moist = {}
        f_flags = {}
        for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
            value = getattr(state, name, None)
            if value is None:
                g_moist[name] = None
                f_flags[name] = False
            else:
                g_moist[name] = _device_grid(value, nz)
                f_flags[name] = True

        # ---- aer_opt = 3: the Thompson aerosol numbers the shortwave
        # optics read (gpuwm.core.rrtmg_aerosol_optics), gathered per SW
        # chunk beside the wrapper inputs.
        aer_gather = None
        if self.shortwave and self.aer_opt == 3:
            g_aer = []
            for name in ("nwfa", "nifa"):
                value = getattr(state, name, None)
                if value is None:
                    raise ValueError(
                        f"aer_opt=3 reads the Thompson aerosol numbers "
                        f"(state.nwfa, state.nifa; mp_physics=28) and this "
                        f"state carries no {name}: the shortwave would run "
                        "zero aerosol under aer_opt=3")
                g_aer.append(_device_grid(value, nz))
            aer_gather = _ColumnGather(g_aer, ncol)

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
        g_radii = {}
        has_req = {}
        for name, key, q, scheme_declares in (
                ("effc", "re_cloud", "qc", has_reqc),
                ("effi", "re_ice", "qi", has_reqi),
                ("effs", "re_snow", "qs", has_reqs)):
            value = getattr(state, name, None)
            if not scheme_declares:
                # WRF semantics: the wrapper never reads scheme radii
                # for this mp; relcalc/reicalc take over inside prep.
                g_radii[key] = None
                has_req[name] = 0
                continue
            if value is None:
                raise ValueError(
                    f"mp_physics={int(mp_physics)} declares {name} to "
                    "radiation (WRF use_mp_re scheme table) but "
                    f"state.{name} is missing -- the microphysics radii "
                    "contract is broken; refusing to radiate fallback "
                    "radii silently")
            eff_um = _device_grid(value, nz)
            self._validate_radii_micron_device(name, eff_um, g_moist[q])
            g_radii[key] = eff_um
            has_req[name] = 1

        surf = {name: _device_flat(fields[name])
                for name in ("tsk", "emiss", "albedo", "xland", "xice",
                             "snow")}
        lat_flat = self.latitude_deg.reshape(-1)
        lon_flat = self.longitude_deg.reshape(-1)
        lat_d = cp.asarray(lat_flat)
        fnm_d = cp.asarray(np.ascontiguousarray(
            np.asarray(self._host(state.fnm), np.float32)))
        fnp_d = cp.asarray(np.ascontiguousarray(
            np.asarray(self._host(state.fnp), np.float32)))

        active_bl = mynn_bl_cloud_active(
            getattr(cfg, "bl_pbl_physics", 0), getattr(cfg, "icloud_bl", 0))
        if active_bl:
            g_bl = {}
            for name in ("qc_bl", "qi_bl", "cldfra_bl"):
                if fields.get(name) is None:
                    raise ValueError(
                        "MYNN radiation coupling requires QC_BL, QI_BL, "
                        "and CLDFRA_BL")
                g_bl[name] = _device_grid(fields[name], nz)
            itimestep = wrf_itimestep(state.elapsed_seconds, cfg.dt)
            if type(itimestep) is not int or itimestep < 1:
                raise ValueError(
                    "MYNN radiation itimestep must be a positive int")
        else:
            g_bl = None
            itimestep = 1

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
        coszen_d = cp.asarray(coszen)

        # ---- o33d: WRF's root-compute + parent->child routing ----------
        # Keep the interpolated grid for all chunk gathers. The retained
        # host field remains the provider and restart contract.
        grid_o3 = None
        if self.o3input == 0:
            # The wrapper ignores o33d in this mode and builds O3DATA from
            # pressure and latitude.  Shape-correct zeros keep one batch
            # contract for both modes.
            self._o33d_grid = None
        elif self._ozone_provider is not None:
            driver = getattr(state, "physics", None)
            owner = getattr(driver, "cam_ozone", None)
            if owner is not None and owner.mode == "parent-interpolated":
                if driver.call_counts.get("cam_ozone", 0) == 0:
                    raise RuntimeError("child radiation ran before parent CAM ozone FORCE")
                grid = cp.asnumpy(driver.o3rad)
                o33d = np.ascontiguousarray(grid.transpose(1, 2, 0).reshape(-1, nz))
            else:
                o33d = np.asarray(self._ozone_provider(), np.float32)
            if o33d.shape != (ncol, nz):
                raise ValueError(
                    f"ozone_parent provider returned shape {o33d.shape}; "
                    f"this child grid needs ({ncol}, {nz})")
            # Retain for child domains (their providers read this field,
            # so a nest's ozone updates exactly when its parent's does).
            self._o33d_grid = np.ascontiguousarray(
                o33d.reshape(ny, nx, nz).transpose(2, 0, 1))
            grid_o3 = cp.asarray(self._o33d_grid.reshape(nz, ncol))
        else:
            ozmixt = np.asarray(self._ozone.ozn_latitude_time_int(
                julday, julian, self.latitude_deg.reshape(-1), self._ozone_climo))
            pin = np.asarray(self._ozone_climo.plev)
            if pin.dtype != np.float32 or pin.ndim != 1:
                raise ValueError("pin must be float32 (levsiz,)")
            if not np.all(np.diff(pin) > 0):
                # WRF's runtime guard (ozn_p_int's kount > ncol check).
                raise ValueError("OZN_P_INT: Bad ozone data: "
                                 "non-monotonicity suspected")
            if (ozmixt.dtype != np.float32
                    or ozmixt.shape != (ncol, pin.size)):
                raise ValueError("ozmixt must be float32 (ncol, levsiz)")
            ozmixt = np.ascontiguousarray(ozmixt)
            pin_d = cp.asarray(pin)
            levsiz = int(pin.size)
            # ozn_p_int on the device over the (nz, ncol) grid, once, for
            # the field children and restarts retain and chunks reuse.
            grid_o3 = cp.empty((nz, ncol), dtype=cp.float32)
            ozmixt_d = cp.asarray(ozmixt)
            _adapter_kernel("rla_ozn_p_int")(
                ((ncol + 127) // 128,), (128,),
                (np.int32(ncol), np.int32(nz), np.int32(levsiz),
                 g_p3d, np.int64(ncol), np.int64(1), pin_d, ozmixt_d,
                 grid_o3, np.int64(ncol), np.int64(1)))
            del ozmixt_d
            # Retain for child domains (their providers read this field,
            # so a nest's ozone updates exactly when its parent's does).
            self._o33d_grid = cp.asnumpy(grid_o3).reshape(nz, ny, nx)
            del pin_d

        shared = dict(
            icloud=1, warm_rain=warm_rain, cldovrlp=2, idcor=0,
            o3input=self.o3input,
            rrtmg_cloud_optics_form=getattr(cfg, "rrtmg_cloud_optics_form", "wrf_461"),
            has_reqc=has_req["effc"], has_reqi=has_req["effi"],
            has_reqs=has_req["effs"], f_qc=f_flags["qc"],
            f_qr=f_flags["qr"], f_qi=f_flags["qi"], f_qs=f_flags["qs"],
            f_qg=f_flags["qg"], yr=yr, julian=julian,
            mp_physics=mp_physics)
        ice_rule = bool(has_req["effs"])

        # Every grid a chunk reads, gathered in one launch per chunk.
        gather_grids = {"p3d": g_p3d, "t3d": g_t3d, "zw": g_zw,
                        "p8w": g_p8w, "dz8w": g_dz8w}
        if grid_o3 is not None:
            gather_grids["o33d"] = grid_o3
        gather_grids.update((name, g) for name, g in g_moist.items()
                            if g is not None)
        gather_grids.update((key, g) for key, g in g_radii.items()
                            if g is not None)
        if active_bl:
            gather_grids.update(g_bl)
        gather_names = list(gather_grids)
        gather = _ColumnGather(gather_grids.values(), ncol)

        def chunk_inputs(sel):
            """The prep keyword set for columns ``sel`` (a slice or a device
            index array), built
            on the device in the host adapter's order:
            columns, t8w, cal_cldfra1 on the raw moisture, radii in meters,
            then the MYNN merge (supplied masks first, then fraction, mass
            and radii).  Every step is per column, so a chunk equals the
            same columns of a whole-grid evaluation bitwise."""
            blocks = dict(zip(gather_names, gather(sel)))
            p3d = blocks["p3d"]
            nc = p3d.shape[0]
            t3d = blocks["t3d"]
            zw = blocks["zw"]
            t8w = cp.empty((nc, nz + 1), dtype=cp.float32)
            _adapter_kernel("rla_t8w")(
                *blk(nc * (nz + 1)),
                (np.int32(nc), np.int32(nz), t3d, zw, fnm_d, fnp_d, t8w))
            del zw
            moist = {}
            for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
                g = g_moist[name]
                moist[name] = (cp.zeros((nc, nz), dtype=cp.float32)
                               if g is None else blocks[name])
            # cal_cldfra1 (icloud=1), the WRF transcription already gated
            # on the RTE+RRTMGP path -- imported, not copied; fed the raw
            # moisture before the MYNN merge writes qc/qi.
            cldfra = _rrtmgp.cal_cldfra1(
                moist["qv"], moist["qc"], moist["qi"], moist["qs"], t3d,
                p3d, f_qc=True, f_qi=f_qi, f_qs=f_qs)
            radii = _radius_meters_blocks({
                key: None if g is None else blocks[key]
                for key, g in g_radii.items()})
            if active_bl:
                bl = {k: blocks[k] for k in g_bl}
                n = moist["qc"].size
                null = np.uint64(0)
                # The source cloud driver merges mass without rewriting
                # microphysics radii (v4.1.21 radiation_driver.F:1256-1304).
                source_bl = getattr(cfg, "bl_mynn_version", "wrf_461") == "gsd_41"
                keep_source_radii = source_bl or getattr(
                    cfg, "rrtmg_cloud_optics_form", "wrf_461") == "noaa_wrf39"
                if source_bl:
                    # GSD MYNN v4.1 hands radiation in-cloud QC_BL; its
                    # merge multiplies by CLDFRA_BL and splits the phase
                    # at radiation time (rla_mynn_gsd41).
                    _adapter_kernel("rla_mynn_gsd41")(
                        *blk(n),
                        (np.int64(n), moist["qc"], moist["qi"], bl["qc_bl"],
                         bl["cldfra_bl"], t3d, cldfra,
                         np.int32(1 if itimestep == 1 else 0),
                         null if keep_source_radii or radii["re_cloud"] is None
                         else radii["re_cloud"],
                         null if keep_source_radii or radii["re_ice"] is None
                         else radii["re_ice"],
                         np.int32(1 if ice_rule else 0),
                         F(MERGE_QC_BELOW), F(MERGE_QI_BELOW),
                         F(MERGE_CLDFRA_BL_ABOVE)))
                else:
                    _adapter_kernel("rla_mynn")(
                        *blk(n),
                        (np.int64(n), moist["qc"], moist["qi"],
                         bl["qc_bl"], bl["qi_bl"], bl["cldfra_bl"], cldfra,
                         np.int32(1 if itimestep != 1 else 0),
                         null if keep_source_radii or radii["re_cloud"] is None
                         else radii["re_cloud"],
                         null if keep_source_radii or radii["re_ice"] is None
                         else radii["re_ice"],
                         np.int32(1 if ice_rule else 0),
                         F(MERGE_QC_BELOW), F(MERGE_QI_BELOW),
                         F(MERGE_CLDFRA_BL_ABOVE)))
                del bl
            o33d = (blocks["o33d"] if grid_o3 is not None else
                    cp.zeros((nc, nz), dtype=cp.float32))
            sel_kw = dict(
                p3d=p3d, p8w=blocks["p8w"], t3d=t3d, t8w=t8w,
                dz8w=blocks["dz8w"], qv3d=moist["qv"],
                qc3d=moist["qc"], qr3d=moist["qr"], qi3d=moist["qi"],
                qs3d=moist["qs"], qg3d=moist["qg"], cldfra3d=cldfra,
                o33d=o33d, tsk=surf["tsk"][sel], xland=surf["xland"][sel],
                xice=surf["xice"][sel], snow=surf["snow"][sel],
                xlat=lat_d[sel])
            for key in ("re_cloud", "re_ice", "re_snow"):
                if radii[key] is not None:
                    sel_kw[key] = radii[key]
            return sel_kw

        from gpuwm.core.trace_gases import LEGACY_LW_GASES, LEGACY_SW_GASES, trace_gas_subset
        lw_gases = trace_gas_subset(self.trace_gas_overrides, LEGACY_LW_GASES) or None
        sw_gases = trace_gas_subset(self.trace_gas_overrides, LEGACY_SW_GASES) or None
        day_seconds = F(86400.0)

        # ---- LW: all columns, adapter chunk == engine chunk -----------
        chunk_lw = self.column_chunk or _lw.LW_BATCH_COLUMN_CHUNK
        # Result grids live on the device for the whole call. Night SW
        # columns retain the driver's positive zero initialization.
        rthratenlw = cp.zeros((nz, ncol), dtype=cp.float32)
        glw = cp.zeros(ncol, dtype=cp.float32)
        olr = cp.zeros(ncol, dtype=cp.float32)
        rthratensw = cp.zeros((nz, ncol), dtype=cp.float32)
        gsw = cp.zeros(ncol, dtype=cp.float32)
        # SWDDIR/SWDDIF, rrtmg_swrad's swdkdir/swdkdif at the surface
        # (module_ra_rrtmg_sw.F, the driver's jararias 2013 arm;
        # swdkdif(1) = swdflx(1) - swdkdir(1), :9526, :11524), zero at night
        # as the radiation driver zeroes them (:1724-1726).  SWDDIR is built
        # only when BEP+BEM (sf_urban_physics = 3) allocated it in the
        # surface fields; SWDDIF also when slope_rad asks for it
        # (gpuwm.core.topo_radiation).  Off, nothing is allocated.
        swddir = swddif = None
        if self.shortwave and "swddir" in fields:
            swddir = cp.zeros(ncol, dtype=cp.float32)
        if self.shortwave and (swddir is not None or getattr(
                self, "surface_diffuse_requested", False)):
            swddif = cp.zeros(ncol, dtype=cp.float32)
        for c0 in range(0, ncol if self.longwave else 0, chunk_lw):
            c1 = min(c0 + chunk_lw, ncol)
            sel = slice(c0, c1)
            kw = chunk_inputs(sel)
            # The wrapper glue on the device: rrtmg_legacy_device holds
            # lwrad_prep_batch's statements bitwise
            # (tests/test_rrtmg_legacy_prep_device.py).
            pl = _prep_device.lwrad_prep_batch_device(
                **kw, emiss=surf["emiss"][sel],
                nlayers=nlayers, trace_gas_overrides=lw_gases,
                subcolumn_generator=self._mcica_generator(
                    _mcica.gpu_generate_lw_subcolumns),
                **shared)
            del kw
            # The WRF-level extras are not engine inputs: drop them before
            # the engine runs (legacy_radiation_vram_bytes prices what the
            # engine phase holds).
            for extra in _LW_PREP_EXTRAS:
                pl.pop(extra, None)
            res = _lw.gpu_rrtmg_lw_batched_device(
                pl["ncol"], pl["nlay"], pl["icld"], pl["play"],
                pl["plev"], pl["tlay"], pl["tlev"], pl["tsfc"],
                pl["h2ovmr"], pl["o3vmr"], pl["co2vmr"], pl["ch4vmr"],
                pl["n2ovmr"], pl["o2vmr"], pl["cfc11vmr"],
                pl["cfc12vmr"], pl["cfc22vmr"], pl["ccl4vmr"],
                pl["emis"], pl["inflglw"], pl["iceflglw"],
                pl["liqflglw"], pl["cldfmcl"], pl["taucmcl"],
                pl["ciwpmcl"], pl["clwpmcl"], pl["cswpmcl"],
                pl["reicmcl"], pl["relqmcl"], pl["resnmcl"],
                pl["tauaer"], self._C, column_chunk=chunk_lw, mcica_layout="column",
                _stage_probe=self._stage_probe)
            del pl
            nc = c1 - c0
            nl = int(res["hr"].shape[1])
            # Map directly into the global grids and read Exner there.
            _adapter_kernel("rla_lw_out_grid")(
                *blk(nc * nz),
                (np.int32(nc), np.int32(nz), np.int32(nl), res["hr"],
                 res["uflx"], res["dflx"], g_pi3d, day_seconds,
                 np.int64(ncol), np.int64(c0), rthratenlw, glw, olr))
            del res

        # ---- SW: driver-level zeroing (dossier section 3) + day gather
        day_idx = np.nonzero(coszen > F(0.0))[0]
        night_idx = np.nonzero(coszen <= F(0.0))[0]
        if self.shortwave and night_idx.size:
            # Wrapper-level night contract: COSZR + the SW_NIGHT_ZEROED
            # list.  None of these reach RadiationResult, but the write
            # keeps the wrapper contract exercised (dossier section 3).
            self._night_outputs = _prep.swrad_night_outputs_batch(
                coszen[night_idx])
        else:
            self._night_outputs = None
        # The engine sizes the chunk to this device once per layer count
        # (swrad_prep_batch's nlay is kte + 1 = nz + 1, as the pricing
        # assumes); an explicit column_chunk is untouched.
        chunk_sw = (self.column_chunk
                    or (_cuda_sw(self._sw_tables).batch_column_chunk(nz + 1)
                        if self.shortwave else _sw.SW_BATCH_COLUMN_CHUNK))
        day_idx_d = cp.asarray(day_idx.astype(np.int64))
        for c0 in range(0, day_idx.size if self.shortwave else 0, chunk_sw):
            sel = day_idx_d[c0:c0 + chunk_sw]
            kw = chunk_inputs(sel)
            # Optional qualification hook. Off, no extra arrays or transfers.
            # It records all/clear flux at the same radiation-call sun; a
            # per-step interpolated SWDOWN must not be mixed with held clear
            # flux to infer cloud attenuation.
            diagnostic_sink = getattr(self, "_sw_diagnostic_sink", None)
            diagnostic_cloud = (kw["cldfra3d"] if diagnostic_sink else None)
            diagnostic_aerosol = None
            aerosol = None
            if aer_gather is not None:
                # The driver builds the optics from its own p, t, qv and
                # dz8w before RRTMG_SWRAD runs (module_radiation_driver.F
                # :975, :1890); the wrapper's layer above the model top
                # stays 0/1/0, so the slabs span the SW engine's nz + 1.
                nwfa_c, nifa_c = aer_gather(sel)
                smoke_options = {}
                if smoke_gather is not None:
                    smoke_options = dict(smoke_aod=smoke_gather(sel)[0], smoke_feedback=True)
                aerosol_result = _aer3.aer3_sw_optics_device(
                    kw["p3d"], kw["t3d"], kw["qv3d"], kw["dz8w"], nwfa_c,
                    nifa_c, nz + 1, **smoke_options)
                aerosol = aerosol_result[:3]
                if diagnostic_sink is not None:
                    diagnostic_aerosol = dict(nwfa=nwfa_c, nifa=nifa_c,
                                              taod550=aerosol_result[3])
                    if smoke_options:
                        diagnostic_aerosol["smoke_aod"] = smoke_options["smoke_aod"]
                del smoke_options
                del aerosol_result
                del nwfa_c, nifa_c
            ps = _prep_device.swrad_prep_batch_device(
                **kw, albedo=surf["albedo"][sel],
                xcoszen=coszen_d[sel], solcon=solcon,
                sf_surface_physics=sf_surface_physics, trace_gas_overrides=sw_gases,
                subcolumn_generator=self._mcica_generator(
                    _mcica.gpu_generate_sw_subcolumns),
                **shared)
            del kw
            for extra in _SW_PREP_EXTRAS:
                ps.pop(extra, None)
            nc = int(ps["ncol"])
            adjes = np.full(nc, ps["adjes"], np.float32)
            res = _cuda_sw(self._sw_tables).rrtmg_sw_batched_device(
                nc, ps["nlay"], ps["icld"], ps["play"], ps["plev"],
                ps["tlay"], ps["tlev"], ps["tsfc"], ps["h2ovmr"],
                ps["o3vmr"], ps["co2vmr"], ps["ch4vmr"], ps["n2ovmr"],
                ps["o2vmr"], ps["asdir"], ps["asdif"], ps["aldir"],
                ps["aldif"], ps["coszen"], adjes, int(ps["dyofyr"]),
                ps["scon"], ps["inflgsw"], ps["iceflgsw"],
                ps["liqflgsw"], ps["cldfmcl"], ps["taucmcl"],
                ps["ssacmcl"], ps["asmcmcl"], ps["fsfcmcl"],
                ps["ciwpmcl"], ps["clwpmcl"], ps["cswpmcl"],
                ps["reicmcl"], ps["relqmcl"], ps["resnmcl"],
                aer_opt=0 if aerosol is None else 3, aerosol=aerosol,
                column_chunk=chunk_sw, _stage_probe=self._stage_probe, mcica_layout="column")
            del ps, aerosol
            if diagnostic_sink is not None:
                diagnostic_sink(
                    elapsed_seconds=float(state.elapsed_seconds), shape=(ny, nx),
                    indices=day_idx[c0:c0 + chunk_sw],
                    coszen=coszen[day_idx[c0:c0 + chunk_sw]],
                    swdnb=res["swdflx"][:, 0], swdnbc=res["swdflxc"][:, 0],
                    cloud_fraction=diagnostic_cloud, aerosol=diagnostic_aerosol,
                    adapter=self)
            del diagnostic_cloud, diagnostic_aerosol
            if swddif is not None:
                swddif[sel] = res["swdflx"][:, 0] - res["swdkdir"][:, 0]
            nlay_sw = int(res["swhr"].shape[1])
            if int(res["swdflx"].shape[1]) != nlay_sw + 1:
                raise ValueError("SW engine flux/heating-rate layer counts "
                                 "disagree")
            _adapter_kernel("rla_sw_out_grid")(
                *blk(nc * nz),
                (np.int32(nc), np.int32(nz), np.int32(nlay_sw),
                 res["swdflx"], res["swuflx"], res["swhr"], g_pi3d,
                 day_seconds, np.int64(ncol), sel, rthratensw, gsw))
            if swddir is not None:
                # SWDDIR at the surface for BEP+BEM: rrtmg_swrad's swdkdir.
                swddir[sel] = res["swdkdir"][:, 0]
            del res
        if self.shortwave:
            # The engine's per-chunk workspace served every chunk of this
            # event; handing it back here lets the model's step-boundary
            # pool trim return the bytes to the driver, so the run's
            # resident footprint is what it was before the workspace was
            # hoisted out of the chunk loop.
            _cuda_sw(self._sw_tables).release_scratch()
            diagnostic_sink = getattr(self, "_sw_diagnostic_sink", None)
            if diagnostic_sink is not None:
                smoke_receipt = {}
                if smoke_provider is not None:
                    smoke_receipt["smoke_binding"] = smoke_provider.binding_receipt()
                diagnostic_sink(elapsed_seconds=float(state.elapsed_seconds),
                    shape=(ny, nx), coszen=coszen, adapter=self, finish=True,
                    **smoke_receipt)

        # ---- driver-level SWDOWN = GSW/(1-ALBEDO) (driver line 2877) --
        swdown = cp.empty(ncol, dtype=cp.float32)
        _adapter_kernel("rla_swdown")(
            *blk(ncol), (np.int64(ncol), gsw, surf["albedo"], swdown))

        self.update_count += 1
        return RadiationResult(
            rthratenlw=rthratenlw.reshape(nz, ny, nx),
            rthratensw=rthratensw.reshape(nz, ny, nx),
            swdown=swdown.reshape(ny, nx),
            glw=(glw.reshape(ny, nx) if self.longwave
                 else fields["glw"]),
            gsw=gsw.reshape(ny, nx),
            coszen=coszen_d.reshape(ny, nx),
            olr=(olr.reshape(ny, nx) if self.longwave
                 else None),
            swddir=(None if swddir is None else swddir.reshape(ny, nx)),
            swddif=(None if swddif is None else swddif.reshape(ny, nx)))
