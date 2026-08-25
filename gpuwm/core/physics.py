"""WRF-ordered non-timesplit physics driver.

The driver calls radiation, MM5 surface layer, Noah LSM, YSU PBL, and
cumulus in WRF order before RK3.  Physical tendencies are computed once,
converted from A-grid uncoupled rates to the dry-mass-coupled ARW
slow-tendency form, and held fixed across all three RK stages.  This follows
WRF v4.6.1
``solve_em.F``, ``module_first_rk_step_part1.F``,
``module_em.F:calculate_phy_tend`` and
``module_physics_addtendc.F:phy_bl_ten``.

``radt_minutes``, ``cudt_minutes``, and ``bldt`` use WRF minutes and zero
means every model step.  Each slot follows its own WRF driver predicate:
radiation uses the default-offset ``MOD(ITIMESTEP, STEPRA) == 1`` calendar,
while cumulus and surface/PBL use ``MOD(ITIMESTEP, STEP*) == 0``.  Every
driver also calls on the first step and when its interval is one step.

Cumulus results that carry ``nca_seconds`` additionally follow WRF's
Kain-Fritsch driver persistence (Task 6b): each column holds its stored
tendencies, RAINCV, and PRATEC while its NCA timer has time remaining
(``module_cu_kfeta.F:410-440``), RAINC accumulates ``PRATEC*DT`` on every
model step, and the timer decrements by DT per step with the stored
tendencies cleared near expiry (``solve_em.F:3558-3571`` ->
``module_physics_addtendc.F:2139-2231``).
"""

from __future__ import annotations

import math
import sys
import weakref
from collections import abc as _abc
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

import cupy as cp
import numpy as np

from gpuwm.config import (MYJ_PBL_SCHEME, MYJ_SFCLAY_SCHEME,
                          NOAHMP_OPTION_IDENTITY, RUC_OPTION_IDENTITY,
                          SASE_PBL_SCHEME, RunConfig,
                          radiation_enabled, radiation_scheme_ids,
                          soil_layer_count)
from gpuwm.core import constants as c
# Configuration-only predicates and name tables live in the
# runtime-free inventory module so the VRAM estimator (and every
# other caller that must stay importable without cupy) can ask them
# without paying for this module's body.  Re-exported here so every
# existing `from gpuwm.core.physics import ...` keeps resolving.
from gpuwm.core.physics_inventory import (
    PBL_RQI_MICROPHYSICS,
    _HMIX_K_DIAG_NAMES,
    hmix_k_diag_names,
    microphysics_scratch_slots,
    physics_driver_required,
    physics_enabled,
    physics_retains_ysu_output,
    physics_reuses_pbl_composition,
)
from gpuwm.core import health_ledger
from gpuwm.core.noah import (_F2D as NOAH_FIELDS_2D,
                             _F3D as NOAH_FIELDS_3D,
                             launch_noah, load_tables, pack_params)
from gpuwm.core.microphysics import (
    MicrophysicsDiagnostics,
    validate_surface_diagnostics,
)
from gpuwm.core.mynn_sfclay import (
    MYNN_SURFACE_OUTPUTS,
    MynnSurfaceResult,
    launch_mynn_surface_layer,
    seed_mynn_surface_first_step,
)
from gpuwm.core.noahmp_runtime import (
    NOAHMP_DIAGNOSTICS_2D,
    NOAHMP_STATE_2D,
    NOAHMP_STATE_INT_2D,
    NOAHMP_STATE_SNOW_3D,
    NOAHMP_STATE_SNOWSOIL_3D,
    NSNOW as NOAHMP_NSNOW,
    NoahmpRuntimeParameters,
    NoahmpSolarGeometry,
    classify_noahmp_surface,
    guard_noahmp_glacier_columns,
    noahmp_cold_start,
    noahmp_lsm_step,
)
from gpuwm.core.ruc_runtime import (
    RUC_DIAGNOSTICS_2D,
    RUC_FRACTIONAL_SEAICE_FIELDS,
    RUC_STATE_2D,
    RUC_STATE_3D,
    RucRuntimeParameters,
    ruc_cold_start,
    ruc_lsm_step,
)
from gpuwm.core.surface_forcing import (
    SURFACE_PRECIPITATION_FIELDS,
    SurfacePrecipitationForcing,
)
from gpuwm.core.mynn_pbl_runtime import (
    MYNN_PBL_DIAGNOSTICS_2D,
    MYNN_PBL_DIAGNOSTICS_INT_2D,
    MYNN_PBL_STATE_3D,
    mynn_pbl_step,
    validate_mynn_tendencies,
)
from gpuwm.core.sase import (launch_bulk_richardson_zi,
                             launch_diff_flux_diag,
                             launch_implicit_vertical_diffusion,
                             launch_moist_n2, launch_n2,
                             launch_plume_vent_flux, launch_sase_step,
                             launch_scalar_mix, launch_vent_deposit,
                             launch_vent_deposit_scale,
                             launch_vent_flux_diag)
from gpuwm.core.sfclay import (SFCLAY_OUTPUTS, SFClayResult,
                               launch_sfclay, sfclay)
from gpuwm.core.radiation_carriers import (
    CARRIER_SOURCE_ANALYTIC_GEOMETRY,
    CARRIER_SOURCE_DECLARED_CONSTANT,
    CARRIER_SOURCE_EXTERNAL_ARRAY,
    CARRIER_SOURCE_RADIATION_SCHEME,
    CARRIER_SOURCE_UNWRITTEN,
    SURFACE_RADIATION_POLICY_REQUIRED,
    CarrierContract, CarrierContractError, consumer_carriers)
from gpuwm.core.state import DTYPE, DomainState
# NumPy-only authority module: the SASE closure constants are single-
# sourced from it exactly as gpuwm.core.sase does (no CPU-import cost).
from gpuwm.verify.sase_ref import (C_K as SASE_C_K,
                                   CP_AIR as SASE_CP_AIR,
                                   E_MIN as SASE_E_MIN,
                                   prandtl_blend as sase_prandtl_blend)
from gpuwm.core.myjpbl import (MYJ_PBL_INOUT, myj_pbl_step,
                               validate_myj_pbl_outputs)
from gpuwm.core.myjsfc import (MYJ_SFCLAY_INOUT, MYJ_SFCLAY_OUTPUTS,
                               launch_myj_sfclay)
from gpuwm.core.shinhong import (SHINHONG_PASSENGER_OUTPUTS,
                                 SHINHONG_TKE_FLOOR,
                                 invalid_shinhong_outputs, launch_shinhong,
                                 repair_shinhong_passenger_outputs)
from gpuwm.core.ysu import launch_ysu, validate_ysu_outputs
from gpuwm.ingest.soil import NOAH_LAYER_THICKNESS_M


#: THE DECLARED CONSTANT downward longwave, in W m-2.  It is a number a
#: caller may TYPE.  It is not a measurement, it is not a scheme, and no
#: default hands it out: :func:`initialize_physics` takes ``glw`` with no
#: default at all, and :func:`_resolve_initial_glw` names this constant
#: for the two branches that have no longwave scheme to write one.
#:
#: DEFINED ONE LAYER DOWN, in :mod:`gpuwm.physics_compat`, and re-exported
#: here rather than the other way round.  The config-load refusals that
#: quote the number -- ``constant_longwave_refusal`` and
#: ``radiation_off_land_surface_refusal`` -- sit there beside
#: ``CONSTANT_DOWNWARD_LONGWAVE_ACK``, the token that declares it, and
#: they have to be able to state the number without importing a CUDA
#: engine to read it.  The standalone RW-WPS preprocessing wheel is where
#: that became load-bearing: it stages ``gpuwm/physics_compat.py`` and
#: forbids ``gpuwm/core/physics.py``, so a refusal that reached up here
#: for the constant raised ImportError instead of refusing.  One number,
#: one definition, owned by the layer that can ship on its own; the
#: provenance of the value is written at that definition.
#:
#: WHY 300.0 AND NOT SOMETHING BETTER.  The value is unchanged so that the
#: idealised single-column cases that have always used it keep the
#: trajectories their receipts were written against
#: (``docs/superpowers/plans/2026-07-14-gpuwm-phase3.md``: "constant
#: GLW=300 W/m^2 on a vegetated land column").  Making it a better number
#: would silently move every one of them; making it explicit does not move
#: any of them.
from gpuwm.physics_compat import DECLARED_CONSTANT_GLW_WM2

_WSM6_MINOR_DT_SECONDS = np.float32(120.0)
_FP32_SIGNIFICAND_SCALE = 1 << 24
_FP32_ONE_BITS = 0x3F800000
_MICROPHYSICS_DIAGNOSTIC_LABELS = {
    "rainnc": "RAINNC", "rainncv": "RAINNCV", "sr": "SR",
    "snownc": "SNOWNC", "snowncv": "SNOWNCV",
    "graupelnc": "GRAUPELNC", "graupelncv": "GRAUPELNCV",
    "hailnc": "HAILNC", "hailncv": "HAILNCV",
}
_MICROPHYSICS_VALIDATION_NAMES = tuple(_MICROPHYSICS_DIAGNOSTIC_LABELS)


def _wsm6_sr_roundoff_limit(
        dt: float, step_scalings: int = 0) -> tuple[np.float32, int, int]:
    """Return WRF WSM6's proven positive-sum SR roundoff envelope.

    WRF v4.6.1 forms ``SR=(SNOWNCV+GRAUPELNCV)/(RAINNCV+1e-12)``.
    The frozen numerator and total-precipitation denominator contain the
    same nonnegative components but associate their binary32 additions
    differently, so an all-frozen column may finish a few ULPs above one.

    For ``L`` WSM6 minor loops, each loop's total uses three additions, the
    snow pair uses one, each carrying accumulation uses ``L-1`` additions,
    and the final numerator and division use one rounding each.  With
    binary32 unit roundoff ``u=2^-24`` and
    ``gamma_n=n*u/(1-n*u)``, the positive-sum bound is

      B = (1+u)^3 (1+gamma_(L-1))
          / ((1-gamma_3) (1-gamma_(L-1))).

    ``step_scalings`` covers a family member whose kernel applies extra
    unit-scale multiplicative operations to BOTH sides of the quotient
    after each step's sums -- WDM6 scales every accumulation step by
    ``delz[0]/1000*dtcld*1000`` (wdm6.cu:618-633, four roundings) where
    WSM6 accumulates pre-scaled sediment returns.  Each such rounding
    moves the numerator up by at most ``(1+u)`` and the denominator down
    by at most ``(1-u)``, so the bound widens by exactly
    ``((1+u)/(1-u))^step_scalings``; at ``step_scalings=0`` the integer
    arithmetic below is unchanged bit for bit.

    The integer arithmetic below evaluates ``floor((B-1)/ULP(1))`` exactly,
    selecting the largest representable binary32 value no greater than the
    analytic bound.  It therefore admits only WRF expression-order roundoff
    and never clips or otherwise changes the scheme's SR field.  The
    ``+1e-12`` denominator is monotone and keeps any quotient near one in the
    normal range.
    """
    delt = np.float32(dt)
    loops = max(int(np.floor(np.float32(
        delt / _WSM6_MINOR_DT_SECONDS + np.float32(0.5)))), 1)
    accumulation_adds = loops - 1
    scale = _FP32_SIGNIFICAND_SCALE
    if 2 * accumulation_adds >= scale:
        raise ValueError(
            "WSM6 minor-loop count is too large for the proven FP32 SR "
            f"roundoff envelope: loops={loops}")

    # Exact rational form of B after substituting u=1/scale.
    # (1+u)^s/(1-u)^s = (scale+1)^s/(scale-1)^s carries the per-step
    # scaling roundings; s=0 reproduces the historic WSM6 integers.
    numerator = ((scale + 1) ** (3 + step_scalings) * (scale - 3))
    denominator = (scale ** 2 * (scale - 6)
                   * (scale - 2 * accumulation_adds)
                   * (scale - 1) ** step_scalings)
    if numerator >= 2 * denominator:
        raise ValueError(
            "WSM6 FP32 SR roundoff bound reaches 2.0, where ULP(1) "
            f"linearity no longer applies: loops={loops}")
    scaled_delta = (numerator - denominator) * scale
    scaled_ulp = 2 * denominator
    max_ulps = scaled_delta // scaled_ulp
    upper_bits = _FP32_ONE_BITS + max_ulps
    upper = np.asarray(upper_bits, dtype=np.uint32).view(np.float32)[()]
    return upper, int(max_ulps), loops


def _my2_sr_roundoff_limit() -> tuple[np.float32, int]:
    """Milbrandt-Yau's positive-sum SR roundoff envelope (dt-independent).

    milbrandt2.cu:2327-2340 (transcribing WRF's mp_milbrandt2mom_driver
    :3676-3690) forms the SAME positive-sum quotient as WSM6 -- frozen
    components over a total that re-sums those components in a different
    association order -- but once per call from the driver's nine rate
    terms rather than accumulated across minor loops:

      rncv = fl(r1+r2+f1+f2+sn1+sn2+sn3+pe1+pe2) * m   (8 adds, 1 mult)
      num  = fl(fl(fl(sn1+sn2)*m) + fl(fl(pe1+pe2)*m)) + fl(sn3*m)

    Worst case: numerator paths carry at most (1+u)^4 (pair add, scale,
    two combining adds), the denominator at least (1-u)^9 (eight adds and
    the scale), and the division one more (1+u), so

      B = (1+u)^5 / (1-u)^9.

    The ``+1e-12`` denominator is monotone exactly as in WSM6's form.  The
    integer arithmetic evaluates ``floor((B-1)/ULP(1))`` exactly, the same
    selection rule as :func:`_wsm6_sr_roundoff_limit`.
    """
    scale = _FP32_SIGNIFICAND_SCALE
    numerator = (scale + 1) ** 5 * scale ** 4
    denominator = (scale - 1) ** 9
    scaled_delta = (numerator - denominator) * scale
    scaled_ulp = 2 * denominator
    max_ulps = scaled_delta // scaled_ulp
    upper_bits = _FP32_ONE_BITS + max_ulps
    upper = np.asarray(upper_bits, dtype=np.uint32).view(np.float32)[()]
    return upper, int(max_ulps)


#: The positive-sum SR expression family (1.9.1 D2).  Audit of every
#: kernel that writes SR (``grep 'sr\[' gpuwm/core/kernels/*.cu``):
#:
#: * wsm6.cu:402 and wdm6.cu:635 form the IDENTICAL expression
#:   ``sr = (snowncv + graupelncv) / (rainncv + 1e-12)`` accumulated
#:   across the same ``floor(dt/120+0.5)`` minor-loop split
#:   (wsm6.cu:372-374 == wdm6.cu:466-468).  WDM6 additionally applies
#:   four post-sum unit scalings per step (wdm6.cu:618-633), priced by
#:   the ``step_scalings`` term.  1.9.0 granted the envelope to mp=6
#:   only, so WDM6's own WRF-expression-order SR killed a healthy run
#:   at its first frozen-dominated column.
#: * milbrandt2.cu:2340 (mp=9) forms the same positive-sum quotient once
#:   per call; :func:`_my2_sr_roundoff_limit` derives its envelope.
#: * morrison.cu:1041-1049 (mp=10) is NOT a member: its denominator
#:   ``out_c+out_r+out_i+out_s+out_g`` prepends the two liquid terms to
#:   the numerator's own association tail ``(out_i+out_s)+out_g``, and
#:   round-to-nearest is monotone, so every denominator partial dominates
#:   the numerator's and SR <= 1.0 holds exactly in FP32.  It keeps the
#:   tight ``upper=1.0``.
#: * nssl2_driver_support.cu:1266 (mp=18) sits outside the range check by
#:   the scheme's canonical-scratch contract (accept_microphysics).
_SR_EXPRESSION_FAMILY_STEP_SCALINGS = {6: 0, 16: 4}


def _sr_roundoff_envelope(
        mp_physics: int, dt: float) -> tuple[np.float32, int, int]:
    """(upper, max_ulps, minor_loops) for the scheme's SR validator.

    Keyed on SHARED-EXPRESSION FAMILY membership (see
    :data:`_SR_EXPRESSION_FAMILY_STEP_SCALINGS`), not on one scheme id:
    every kernel that forms SR with WSM6's positive-sum expression owns
    the proven roundoff envelope, and everything else keeps the tight
    ``[0, 1]`` range.
    """
    step_scalings = _SR_EXPRESSION_FAMILY_STEP_SCALINGS.get(mp_physics)
    if step_scalings is not None:
        return _wsm6_sr_roundoff_limit(dt, step_scalings)
    if mp_physics == 9:
        upper, max_ulps = _my2_sr_roundoff_limit()
        return upper, max_ulps, 0
    return np.float32(1.0), 0, 0


_OUTPUT_FIELDS = {
    "TSK": "tsk", "T2": "t2", "TH2": "th2", "Q2": "q2",
    "U10": "u10", "V10": "v10", "UST": "ust", "HFX": "hfx",
    "QFX": "qfx", "LH": "lh", "PBLH": "pblh",
    "GRDFLX": "grdflx", "PSIM": "psim", "PSIH": "psih",
}


class UnroutedPhysicsSelectorError(NotImplementedError):
    """A configured surface/LSM/PBL selector VALUE has no routed scheme.

    The driver used to branch on the truthiness of ``sf_surface_physics``
    and ``bl_pbl_physics``, so *any* nonzero value ran Noah and YSU.
    Unlocking a new selector without adding a routing row would therefore
    have produced plausible numbers from the wrong scheme with no error.
    This is that error.
    """

    def __init__(self, selector: str, value, routed):
        self.selector = selector
        self.value = value
        self.routed = tuple(sorted(routed))
        super().__init__(
            f"{selector}={value!r} has no scheme routed to it in the gpuwm "
            "physics driver; refusing to substitute another scheme. Routed "
            f"values: {list(self.routed)}. Adding the selector to "
            "gpuwm/config.py and gpuwm/physics_compat.py is NOT sufficient: "
            f"add a {selector} row to PHYSICS_SLOT_DISPATCH "
            "(gpuwm/core/physics.py) naming the PhysicsDriver method that "
            "runs that exact scheme.")


#: WRF selector value -> the ``PhysicsDriver`` method that runs THAT scheme.
#: Zero means the slot is off and maps to ``None``.  Every dispatch in
#: :meth:`PhysicsDriver.compute` goes through this table; a value that is
#: absent raises :class:`UnroutedPhysicsSelectorError` instead of falling
#: through to whichever scheme happens to be wired.  ``_run_sfclay``
#: additionally re-dispatches on the exact value internally, so the two
#: MM5 spellings and MYNN cannot be confused with each other either.
PHYSICS_SLOT_DISPATCH: dict[str, dict[int, str | None]] = {
    "sf_sfclay_physics": {
        0: None,
        1: "_run_sfclay",     # revised MM5
        # Eta similarity (WRF v4.6.1 module_sf_myjsfc.F).  It gets its OWN
        # runner rather than a fourth _run_sfclay arm because it publishes
        # a different set: AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 and no MOL, ZOL,
        # PSIM/PSIH, REGIME, GZ1OZ0 or WSPD.  gpuwm.config.
        # validate_myj_pairing refuses it with any PBL but MYJ for exactly
        # that reason.
        2: "_run_myj_sfclay",
        5: "_run_sfclay",     # MYNN surface layer
        91: "_run_sfclay",    # classic MM5
    },
    "sf_surface_physics": {
        0: None,
        2: "_run_noah",       # Noah LSM
        3: "_run_ruc",        # RUC LSM
        4: "_run_noahmp",     # Noah-MP LSM
    },
    "bl_pbl_physics": {
        0: None,
        1: "_run_ysu",        # YSU
        # MYJ (WRF v4.6.1 module_bl_myjpbl.F), Mellor-Yamada level 2.5 as
        # extended by Janjic.  Unlike every other row here the scheme does
        # its OWN implicit vertical diffusion, so _run_myj_pbl receives
        # finished tendencies rather than diffusivities.  Its surface
        # pairing is enforced in gpuwm.config.validate_myj_pairing, which
        # is WRF's own fatal at module_physics_init.F:3770-3772.
        2: "_run_myj_pbl",
        5: "_run_mynn_pbl",   # MYNN EDMF
        # Shin-Hong scale-aware (WRF v4.6.1 module_bl_shinhong.F).  The
        # scheme has NO RunConfig knobs on purpose: WRF's
        # shinhong_tke_diag namelist is deliberately not imported.  The
        # TKE chain is a pure passenger diagnostic -- the tendencies
        # never read it, proven on the pinned source (the case-27/case-1
        # oracle pair in tests/test_shinhong_wrf461_parity.py pins
        # OFF/ON bitwise-identical tendencies) -- so ArWen computes it
        # every step under scheme 11 (_run_shinhong passes tke_diag=1
        # unconditionally) and publishes it as state.e_sgs, the field
        # the D1 gray-zone instrument scores.
        11: "_run_shinhong",
        # SASE: not a WRF scheme and deliberately outside WRF's selector
        # namespace, so it can never collide with one WRF adds later.
        SASE_PBL_SCHEME: "_run_sase",
    },
}

#: Land-surface schemes after which WRF's surface driver calls the ordinary
#: ``SFCDIAGS`` (module_surface_driver.F:2983-2998).  RUC instead calls
#: ``SFCDIAGS_RUCLSM`` in :func:`ruc_lsm_step`; Noah-MP unconditionally
#: selects its water/ice flux diagnostic or its own vegetation/bare-ground
#: 2-m fields in :func:`noahmp_lsm_step`.  This is a per-scheme decision, not
#: a property of "an LSM ran", and all three post-LSM paths overwrite the
#: surface layer's earlier T2/Q2/TH2 exactly where WRF does.
LAND_SURFACE_SFCDIAGS_SCHEMES = frozenset({2})


def resolve_physics_slot(selector: str, value) -> str | None:
    """Resolve one selector VALUE to its runner method, or fail loudly."""
    table = PHYSICS_SLOT_DISPATCH[selector]
    try:
        key = int(value)
    except (TypeError, ValueError):
        raise UnroutedPhysicsSelectorError(selector, value, table) from None
    if key not in table:
        raise UnroutedPhysicsSelectorError(selector, key, table)
    return table[key]


def resolve_physics_dispatch(cfg) -> dict[str, str | None]:
    """Selector name -> runner method for every surface/LSM/PBL slot."""
    return {selector: resolve_physics_slot(selector, getattr(cfg, selector))
            for selector in PHYSICS_SLOT_DISPATCH}



#: von Karman constant (WRF share/module_model_constants.F ``karman``),
#: consumed by the SASE flux-consistent lower-boundary e source.
KARMAN = 0.4

#: MYJ's cold-start TKE_MYJ: WRF's ``epsq2``, the value MYJPBLINIT's
#: non-restart block writes into the whole column
#: (module_bl_myjpbl.F:1725, share/module_model_constants.F:92 "initial
#: TKE").  A literal rather than an import of
#: :data:`gpuwm.verify.myj_ref.EPSQ2` because core must not depend on the
#: verification tree (the Shin-Hong precedent in
#: :mod:`gpuwm.core.state`); the two are gated equal in
#: ``tests/test_myj_port.py``.
MYJ_TKE_COLD_START = 0.2

#: SPLIT SUBGRID-FLUX DIAGNOSTIC row table (cfg.sase_flux_diag): the
#: implicit-solve scalar rows that carry a recorded K_v flux, mapped to
#: their buffer key and their UNIT factor.  ``1.0`` keeps the row's own
#: units, so the qv row is recorded in kg m-2 s-1 directly; CP_AIR turns
#: a theta row [K kg m-2 s-1] into a heat flux [W m-2] -- the exact
#: inverse of the model's own theta-row convention, whose surface
#: deposit is dt*HFX/(rho1*CP_AIR*thick_0).  Both POSITIVE UPWARD.  The
#: qc/qi rows are deliberately absent: the registered question is the
#: vapour and heat budget, and each extra row costs another
#: (nz+1, ny, nx) plane per frame.
_SASE_FLUX_DIAG_ROWS = {"dqv": ("fqv_diff", 1.0),
                        "dtheta": ("fth_diff", float(SASE_CP_AIR))}

#: SPLIT SUBGRID-FLUX DIAGNOSTIC history names -> buffer key.  Both
#: channels are z-FACE fields on the mass column (stagger "Z", like W),
#: POSITIVE UPWARD, and share the deposit's lowest-level moist density
#: rho1 so that
#:   (F_vent[k]-F_vent[k+1] + F_diff[k]-F_diff[k+1])*dt/(rho1*thick_k)
#: reproduces the model's own scalar increment.
_SASE_FLUX_DIAG_OUTPUT = {"SASE_FQV_VENT": "fqv_vent",
                          "SASE_FQV_DIFF": "fqv_diff",
                          "SASE_FTH_VENT": "fth_vent",
                          "SASE_FTH_DIFF": "fth_diff"}

def _cumulus_optional_tendency_components(
        cfg: RunConfig) -> tuple[str, ...]:
    """Canonical KF moisture categories for the configured MP phase mode."""
    if not cfg.cu_physics:
        return ()
    from gpuwm.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics

    phase_mode = kf_phase_mode_for_microphysics(cfg.mp_physics)
    components = ["rqr"]
    if phase_mode == KFPhaseMode.SEPARATE_ICE_SNOW:
        components.extend(("rqi", "rqs"))
    elif phase_mode == KFPhaseMode.SEPARATE_SNOW:
        components.append("rqs")
    return tuple(components)


def _pbl_optional_tendency_components(cfg: RunConfig) -> tuple[str, ...]:
    """Canonical optional YSU categories for the configured moist state.

    The set is "every scheme whose moist package carries QI", because that is
    what makes WRF's ``F_QI`` true and therefore what makes
    ``module_first_rk_step_part1.F:1112`` (``CALL pbl_driver``) hand
    ``moist(...,P_QI), F_QI=F_QI`` (:1199) to the PBL driver.  ``mp_physics=28`` belongs by
    ``Registry/Registry.EM_COMMON:3036``, which declares the ``thompsonaero``
    package as ``moist:qv,qc,qr,qi,qs,qg`` -- the identical moist inventory
    ``thompson`` (:3024) declares.  Gated by
    ``tests/test_physics.py::test_the_pbl_rqi_budget_admits_28``.

    ``mp_physics=50`` (P3) belongs too: ``Registry.EM_COMMON:3038`` declares
    the ``p3_1category`` package as ``moist:qv,qc,qr,qi``, so ``F_QI`` is
    true even though P3 carries no snow or graupel -- the predicate is about
    QI's presence in the moist array, not about the frozen inventory.
    """
    return (("rqi",)
            if cfg.bl_pbl_physics and cfg.mp_physics in PBL_RQI_MICROPHYSICS
            else ())


def microphysics_cold_start(state: DomainState, cfg: RunConfig) -> dict:
    """WRF ``mp_init``: the microphysics scheme's ONE-TIME per-domain setup.

    ``phys/module_physics_init.F:1635`` calls ``mp_init`` from ``phy_init``,
    and ``mp_init``'s ``CASE (THOMPSONAERO)`` arm (``:4522-4538``) calls
    ``thompson_init``.  This is the seam :func:`initialize_physics` reaches
    it through; the work itself lives in
    :func:`gpuwm.core.microphysics.microphysics_init`, which returns ``{}``
    for every scheme with no domain-construction step.

    NAMED ``*_cold_start`` deliberately, matching :func:`ruc_cold_start` and
    :func:`noahmp_cold_start`: it is the third one-time, per-domain scheme
    initialisation ``initialize_physics`` performs, and it is the same KIND
    of thing -- not a tendency, not idempotent in general, run once before
    the first step where ``module_physics_init.F`` runs it.

    The name is also load-bearing for ``tools/health_field_census.py``, which
    replaces every ``*_cold_start`` in this module with a no-op so its
    host-array (NumPy-bound) sweep never executes a kernel.  That
    substitution is sound here for exactly the reason the census records for
    the other two: this writes only into ``nwfa``/``nifa``/``nwfa2d``, which
    ``DomainState`` has already allocated, and creates no ``fields`` key, so
    the descriptor inventory is identical with and without it.

    The inner call is resolved through the MODULE rather than a from-import
    so it stays observable: ``tests/test_physics.py`` counts it across a real
    multi-step integration to prove the once-per-domain property.
    """
    from gpuwm.core import microphysics as _microphysics

    return _microphysics.microphysics_init(state, cfg)


def microphysics_scheme_sr_available(mp_physics: int) -> bool:
    """Whether the configured scheme produces WRF's SR (frozen fraction).

    WRF's first-RK surface call supplies the SCHEME's own SR to the land
    surface model for every precipitating scheme that computes one, and runs
    Noah with ``FRPCPN=.true.`` when it does.  ``mp_physics=0`` has no scheme
    and therefore no SR, and gpuwm falls back to WRF's own no-SR proxy
    (``T(kts) <= 273.15``).

    ``mp_physics=28`` produces SR exactly as ``mp_physics=8`` does:
    ``phys/module_microphysics_driver.F``'s ``CASE (THOMPSONAERO)`` arm
    (:1029) binds ``SR=SR`` into ``mp_gt_driver`` at :1091, the same argument
    ``CASE (THOMPSON)`` binds at :1259.  Before this predicate existed the
    three land-surface runners spelled the set ``(1, 6, 8, 10, 18)``, so an
    mp=28 domain silently substituted the temperature proxy AND ran Noah with
    ``frpcpn=False``.  Gated by ``tests/test_physics.py::
    test_the_land_surface_seam_takes_28s_own_sr_not_a_temperature_proxy``.

    Kessler (1) is deliberately included even though it produces SR = 0
    everywhere: WRF's Kessler is warm-rain by construction, so the zero is
    the scheme's answer and falling back to air temperature would freeze cold
    rain the scheme says is liquid.

    ``mp_physics=50`` (P3) belongs for the same reason mp=28 does: its WRF
    driver arm binds ``SR=SR`` (module_microphysics_driver.F:1592) and the
    wrapper computes it as the solid-to-total precipitation ratio
    ``pcprt_sol/(pcprt_liq+pcprt_sol+1.e-12)`` (module_mp_p3.F:898).
    """
    return int(mp_physics) in (1, 6, 8, 9, 10, 16, 18, 28, 50)


def _composed_optional_tendency_components(
        cfg: RunConfig) -> tuple[str, ...]:
    """Union of optional moisture categories entering the RK target."""
    members = (set(_pbl_optional_tendency_components(cfg))
               | set(_cumulus_optional_tendency_components(cfg)))
    return tuple(name for name in ("rqr", "rqi", "rqs") if name in members)


def _as_2d(value, shape, name: str, *, dtype=DTYPE) -> cp.ndarray:
    array = cp.asarray(value, dtype=dtype)
    try:
        array = cp.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError(f"{name} shape {array.shape} is not broadcastable "
                         f"to surface shape {shape}") from exc
    return cp.ascontiguousarray(array)


def _as_soil(value, shape, name: str, *, layers: int) -> cp.ndarray:
    """Broadcast a soil input onto ``(layers, ny, nx)``.

    ``layers`` is required.  It used to default to 4, which meant a caller
    that forgot it silently allocated Noah's geometry for whichever scheme
    was selected.  Every call site resolves it from
    :func:`gpuwm.config.soil_layer_count`.
    """
    array = cp.asarray(value, dtype=DTYPE)
    if array.shape == (layers,):
        array = array[:, None, None]
    try:
        array = cp.broadcast_to(array, (layers, *shape))
    except ValueError as exc:
        raise ValueError(f"{name} shape {array.shape} is not broadcastable "
                         f"to soil shape {(layers, *shape)}") from exc
    return cp.ascontiguousarray(array)


def _model_clock_dt(cfg) -> float:
    """WRF's model-clock ``dt`` for clock-defined cumulus arithmetic.

    KF's driver formulas (0.5*DT hold boundary, NINT(NCA/DT) expiry,
    RAINC += PRATEC*DT) are defined on the model clock; the real74
    compatibility integrator advances internal substeps
    (``cfg.clock_dt > cfg.dt``), the same idiom handled by
    ``lateral_boundary_clock_dt`` and the clock-scaled diff_6th factor.
    """
    clock_dt = float(getattr(cfg, "clock_dt", 0.0) or 0.0)
    return clock_dt if clock_dt > 0.0 else float(cfg.dt)


def _physics_interval_seconds(minutes: float, dt: float) -> float:
    """WRF STEPBL/STEPRA rounding: max(nint(minutes*60/dt), 1)."""
    if minutes <= 0.0:
        return float(dt)
    steps = max(int(np.floor(minutes * 60.0 / dt + 0.5)), 1)
    return steps * float(dt)


def _physics_interval_steps(minutes: float, dt: float | Fraction) -> int:
    """WRF ``MAX(NINT(minutes*60/dt), 1)`` interval in model steps.

    Integer-tick generalization (Phase-5 architecture section C): an
    exact-rational ``fractions.Fraction`` dt divides EXACTLY with an
    exactness assertion -- the experiment loader validates every cadence
    as commensurate with its domain's step, so a non-integral quotient is
    a hard error here, never a silent nint round.  A float dt keeps WRF's
    REAL rounding (phys/module_physics_init.F STEPRA/STEPCU) bitwise for
    every frozen single-domain profile; ``gpuwm.core.clock.resolve_clock``
    asserts both paths agree per domain (F14 single timing authority).
    """
    if minutes <= 0.0:
        return 1
    if isinstance(dt, Fraction):
        steps = Fraction(minutes) * 60 / dt
        if steps.denominator != 1:
            raise ValueError(
                f"physics cadence {minutes} min is not a whole number of "
                f"model steps: dt = {dt} s exactly, quotient = {steps}.")
        return max(int(steps), 1)
    return max(int(np.floor(minutes * 60.0 / dt + 0.5)), 1)


def _radiation_step_due(itimestep: int, stepra: int,
                        radt_minutes: float) -> bool:
    """WRF radiation fixed-step predicate with default ``ra_call_offset=0``.

    Transcribed from WRF v4.6.1 ``module_radiation_driver.F:1113-1130``;
    ``Registry.EM_COMMON:2569`` supplies the default offset used here.
    """
    return (itimestep == 1 or radt_minutes == 0.0 or stepra == 1
            or itimestep % stepra == 1)


def _cumulus_step_due(itimestep: int, stepcu: int,
                       cudt_minutes: float) -> bool:
    """WRF cumulus fixed-step predicate (driver lines 832-849)."""
    return (itimestep == 1 or cudt_minutes == 0.0 or stepcu == 1
            or itimestep % stepcu == 0)


def _surface_pbl_step_due(itimestep: int, stepbl: int,
                          bldt_minutes: float) -> bool:
    """Shared WRF surface/PBL fixed-step predicate, which is modulo zero."""
    return (itimestep == 1 or bldt_minutes == 0.0 or stepbl == 1
            or itimestep % stepbl == 0)


def _validated_array(value, shape: tuple[int, ...], name: str) -> cp.ndarray:
    """Validate one scheme output and return a contiguous FP32 device view.

    The finite check runs through :func:`gpuwm.core.health_ledger.check_finite`
    so that a caller who has installed a ledger keeps the reduction and loses
    only the blocking read.  It matters more here than the count of sites
    suggests: this is the per-ARRAY validator, and a radiation- or
    cumulus-due step walks it a dozen times, so a due step was a dozen
    pipeline drains that the whole-step profile never saw because it never
    profiled a due step.
    """
    array = cp.asarray(value, dtype=DTYPE)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not health_ledger.check_finite(
            array, site=f"validated:{name}",
            message=f"{name} contains a non-finite value"):
        raise FloatingPointError(f"{name} contains a non-finite value")
    return cp.ascontiguousarray(array)


def _trusted_canonical_array(
        value, target, shape: tuple[int, ...], name: str) -> cp.ndarray:
    """Validate a native scheme-owned alias without a device reduction."""
    if value is not target:
        raise ValueError(f"{name} must be the canonical PhysicsDriver array")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != DTYPE:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _checked_array(value, shape: tuple[int, ...], name: str) -> cp.ndarray:
    """Copy one scheme output into an owned, contiguous FP32 device array."""
    return _validated_array(value, shape, name).copy()


def _native_microphysics_diagnostics(
        result: MicrophysicsDiagnostics,
        targets: MicrophysicsDiagnostics,
        slots: Mapping[str, str],
        shape: tuple[int, ...]) -> bool:
    """Whether a scheme returned the driver's exact canonical FP32 arrays."""
    for name in _MICROPHYSICS_VALIDATION_NAMES:
        value = getattr(result, name)
        if name not in slots:
            if value is not None:
                return False
            continue
        target = getattr(targets, name)
        if (value is not target or not isinstance(value, cp.ndarray)
                or value.shape != shape or value.dtype != DTYPE
                or not value.flags.c_contiguous):
            return False
    return True


def _validate_native_microphysics(
        result: MicrophysicsDiagnostics,
        slots: Mapping[str, str],
        sr_upper: np.float32,
        status: cp.ndarray,
        labels: Mapping[str, str] | None = None) -> tuple[str | None, bool, bool]:
    """Validate canonical scheme outputs with one kernel and one readback.

    ``labels`` opts the readback into :mod:`gpuwm.core.health_ledger`: it is
    the caller's display names, and supplying them is what lets this site
    build the refusal the CALLER would have built, at a drain, without a
    blocking copy here.  All three returned values are error-only -- nothing
    downstream of a healthy step reads them -- which is why deferring them
    cannot move a forecast byte.
    """
    from gpuwm.core import health_ledger

    sr = result.sr
    active = sum(
        1 << bit for bit, name in enumerate(_MICROPHYSICS_VALIDATION_NAMES)
        if name in slots)
    values = tuple(
        getattr(result, name) if name in slots else sr
        for name in _MICROPHYSICS_VALIDATION_NAMES)

    def _first(flags: int) -> str | None:
        return next(
            (name for bit, name in enumerate(_MICROPHYSICS_VALIDATION_NAMES)
             if flags & (1 << bit)),
            None)

    def _describe(flags: int) -> None:
        name = _first(flags)
        note = health_ledger.deferred_note()
        if name is not None:
            raise FloatingPointError(
                f"microphysics {labels[name]} contains a non-finite "
                f"value{note}")
        raise ValueError(
            "microphysics SR is outside its validated range "
            f"[0, {float(sr_upper):.9g}]{note}")

    flags = validate_surface_diagnostics(
        values, active, sr_upper, status,
        describe=None if labels is None else _describe)
    return _first(flags), bool(flags & (1 << 16)), bool(flags & (1 << 17))


def _specified_mass_mask(array: cp.ndarray) -> None:
    """WRF add_a2a physical-boundary exclusion (one mass cell, all four
    sides).  WRF applies it whenever ``specified .or. nested``
    (module_physics_addtendc.F:2312-2319) -- both coupler call sites gate
    on that same OR; the periodic_x channel exemption (:2320-2321) is an
    explicit non-goal (gpuwm has no channel mode)."""
    array[..., 0, :] = 0.0
    array[..., -1, :] = 0.0
    array[..., :, 0] = 0.0
    array[..., :, -1] = 0.0



def sase_surface_rho1(*, p1, t1, qv1):
    """Lowest-level moist air density rho1 = p1/(Rd*T1*(1 + EP_1*qv1)).

    THE shared surface density of the SASE lower-boundary seams
    (S3-11b): the identical FP32 expression factored out of
    :func:`sase_surface_e_source` (bitwise -- same operations, same
    order) so the driver computes it ONCE per due step and threads the
    one field to BOTH consumers: the surface e source and the S3-11a/b
    surface scalar-flux deposit (the theta/qv rows of the implicit
    vertical solve).  The S3-11a report obligation: the deposit and
    the e source must ride the SAME moist rho1 -- a deposit at a
    different density would silently rescale HFX/QFX against the e
    source's own buoyancy bookkeeping of the same fluxes.
    EP_1 = Rv/Rd - 1 (WRF ``ep_1``).  Strictly positive for physical
    inputs (p1 > 0, t1 > 0, qv1 >= 0) -- the authority deposit's
    rho1 > 0 contract.
    """
    ep1 = DTYPE(c.RVOVRD - 1.0)
    return p1 / (DTYPE(c.RD) * t1 * (DTYPE(1.0) + ep1 * qv1))


def sase_surface_e_source(*, ust, hfx, qfx, theta1, qv1, p1, t1, dz1,
                          rho1=None):
    """Flux-consistent lower-boundary SASE subgrid-energy production.

    Plan Task 6 / spec section 5 form, per surface cell, in m2 s-3::

        e_sfc_source = u*^3 / (kappa * 0.5*dz1) + max(B_s, 0)
        B_s  = (g / theta1) * (HFX/(rho*cp) + EP_1 * theta1 * QFX/rho)
        rho  = p1 / (Rd * t1 * (1 + EP_1 * qv1)),  EP_1 = Rv/Rd - 1

    The shear term is the neutral surface-layer production
    ``u*^3/(kappa z)`` evaluated at the lowest half level ``z = dz1/2``
    (always >= 0).  The buoyancy term is the surface kinematic
    virtual-heat-flux production ``(g/theta)*w'thv'_s`` with WRF's EP_1
    moisture factor, applied CONSERVATIVELY: only a positive (unstable)
    surface buoyancy flux produces e here -- a stable surface layer
    never drains e through this source (the interior stability-length
    dissipation owns that sink), so the source is non-negative by
    construction and the E_MIN floor cannot be undercut.  Inputs are
    (ny, nx) surface fields from the live post-SFCLAY state
    (UST/HFX/QFX) and the lowest atmosphere level (theta, qv, T, the
    surface interface pressure, and the lowest layer thickness).  This
    is the named seam the smoke gate introspects.

    ``rho1`` (S3-11b): the precomputed :func:`sase_surface_rho1` field
    may be threaded in so the driver's ONE density serves this source
    and the surface scalar-flux deposit alike (the S3-11a
    rho-consistency obligation); ``None`` computes the identical
    expression internally (bitwise -- the same factored function).
    """
    if rho1 is None:
        rho1 = sase_surface_rho1(p1=p1, t1=t1, qv1=qv1)
    ep1 = DTYPE(c.RVOVRD - 1.0)
    wtv = hfx / (rho1 * DTYPE(c.CP)) + ep1 * theta1 * qfx / rho1
    buoyancy = cp.maximum((DTYPE(c.G) / theta1) * wtv, DTYPE(0.0))
    shear = ust * ust * ust / (DTYPE(KARMAN) * DTYPE(0.5) * dz1)
    return shear + buoyancy


def couple_sase_w_tendency(state: DomainState, cfg: RunConfig,
                           dw: cp.ndarray) -> cp.ndarray:
    """Mass-couple the A-grid (half-level) SASE w rate to the rw_t form.

    The slow w tendency forces the coupled W = C_f(mu)*w/msft, so the
    physical rate couples with the FULL-level column mass
    ``c1f*mut + c2f`` and divides by the mass-point map factor --
    mirroring :func:`couple_ysu_tendencies`' rk_addtend_dry convention
    on the w stagger.  The half-level rate averages to interior full
    levels; the surface and model-top rows stay zero (the kinematic
    surface BC ``set_w_surface`` owns w[0], and no PBL-slot scheme
    forces either boundary row).  Specified domains zero the physical-
    boundary cells exactly like the mass-point mask.
    """
    nz, ny, nx = state.p.shape
    full = cp.zeros((nz + 1, ny, nx), dtype=DTYPE)
    full[1:nz] = DTYPE(0.5) * (dw[:-1] + dw[1:])
    chf = (state.c1f[:, None, None] * state.total_mu()[None]
           + state.c2f[:, None, None])
    rw = chf * full
    if cfg.specified:
        _specified_mass_mask(rw)
    if state.has_msf:
        rw = rw / state.msft[None]
    return cp.ascontiguousarray(rw)


#: Producer inputs the closure reads whose value must be STRICTLY
#: POSITIVE for its arithmetic to mean anything: the friction velocity
#: divides the surface energy source, the first-layer thickness divides
#: that source's flux convergence, and pressure and temperature enter a
#: density.  A zero or negative value in any of them is a degenerate
#: INPUT, not a closure failure, and the refusal has to say which.
_SASE_POSITIVE_PRODUCER_INPUTS = frozenset({"ust", "dz1", "p1", "t1"})

#: The same question asked of YSU's producers, answered from YSU's own
#: divisions rather than by analogy.  ``ust`` cubes into the Prandtl
#: shape factor's denominator (kernels/ysu.cu:380) and collapses the
#: velocity scale that divides the countergradient factor (ysu.cu:181,
#: 185); ``psih`` is the bare divisor of the stability parameter
#: (ysu.cu:166); ``znt`` divides the over-water critical Richardson
#: number (ysu.cu:247); ``dz1`` sets the first mass level, which the
#: entrainment depth divides by (ysu.cu:370); and ``p1``/``dp1`` are the
#: surface density and the first layer's mass, the latter being the
#: divisor of the surface heat source that begins the heat solve
#: (ysu.cu:480).  Each is a degenerate INPUT at zero, not a scheme fault.
_YSU_POSITIVE_PRODUCER_INPUTS = frozenset(
    {"ust", "znt", "psih", "dz1", "p1", "dp1"})


def _producer_forensics(
        producer_inputs: Mapping[str, cp.ndarray],
        positive: frozenset[str]) -> list[str]:
    """Name every producer input that was ALREADY degenerate, with counts.

    Called only after a rate has already failed, so its cost never
    touches the healthy path.  ``value <= 0`` is False at NaN by IEEE
    rule, so a NaN cell is reported once as non-finite rather than
    twice.  ``positive`` is the caller's own set of inputs that must be
    strictly positive, because which divisions exist is a fact about the
    consuming scheme and not something this helper can assume.
    """
    notes: list[str] = []
    for name, value in producer_inputs.items():
        if value is None:
            continue
        parts: list[str] = []
        nonfinite = int(cp.count_nonzero(~cp.isfinite(value)).item())
        if nonfinite:
            parts.append(f"{nonfinite} non-finite")
        if name in positive:
            nonpositive = int(cp.count_nonzero(value <= 0).item())
            if nonpositive:
                parts.append(f"{nonpositive} <= 0")
        if parts:
            notes.append(f"{name} ({', '.join(parts)})")
    return notes


def validate_sase_tendencies(
        rates: Mapping[str, cp.ndarray], *,
        grid_id: int | None = None,
        producer_inputs: Mapping[str, cp.ndarray] | None = None) -> None:
    """Reject non-finite SASE rates before they join the RK stack.

    FORENSIC FORM, matching the guard the other PBL-slot schemes carry
    and going one step past it.  A non-finite PBL tendency is almost
    never a fact about the tendency: it is the IMAGE of something the
    producer handed the closure -- a zero friction velocity out of the
    surface layer, a non-finite surface heat flux, a collapsed first
    layer.  The rate's name alone sends a reader to the wrong file.  So
    the refusal names the DOMAIN the rate came from and, when the caller
    hands over the producer's inputs, which of THOSE were already
    degenerate before the closure touched them -- and says so explicitly
    when none of them were, because "the closure's own arithmetic
    produced this" is the other half of the diagnosis and the one the
    reader must not have to infer from silence.

    The healthy path is unchanged: one finiteness reduction per rate,
    exactly as before, and no forensic work until one of them fails.
    """
    for name, value in rates.items():
        if bool(cp.isfinite(value).all()):
            continue
        where = "" if grid_id is None else f" on domain {grid_id}"
        detail = ""
        if producer_inputs is not None:
            notes = _producer_forensics(
                producer_inputs, _SASE_POSITIVE_PRODUCER_INPUTS)
            detail = (
                "; producer inputs already degenerate: " + ", ".join(notes)
                if notes else
                "; every producer input was finite and in range, so the "
                "closure's own arithmetic produced it")
        raise FloatingPointError(
            f"SASE returned non-finite {name} tendency{where}{detail}")


class _YsuProducerInputs(_abc.Mapping):
    """YSU's producer-input view, materialized only when it is read.

    Every member is a view onto a field another component already owns,
    except ``dp1`` -- the first layer's pressure thickness -- which is a
    subtraction.  The guard reads this mapping only after a rate has
    already failed, so binding it costs one object per PBL step and the
    subtraction never runs on a healthy one.
    """

    __slots__ = ("_f", "_atmosphere")

    #: Surface-layer/LSM coupling fields, then the first-layer geometry.
    _NAMES = ("ust", "hfx", "qfx", "wspd", "br", "znt", "psim", "psih",
              "dz1", "p1", "dp1")

    def __init__(self, fields, atmosphere) -> None:
        self._f = fields
        self._atmosphere = atmosphere

    def __getitem__(self, key: str) -> cp.ndarray:
        f, atmosphere = self._f, self._atmosphere
        if key in ("ust", "hfx", "qfx", "wspd", "br", "znt"):
            return f[key]
        # WRF's PBL driver binds YSU's PSIM/PSIH to the full similarity
        # denominators, which the driver holds as fm/fh.
        if key == "psim":
            return f["fm"]
        if key == "psih":
            return f["fh"]
        if key == "dz1":
            return atmosphere["dz"][0]
        if key == "p1":
            return atmosphere["p_interface"][0]
        if key == "dp1":
            return atmosphere["p_interface"][0] - atmosphere["p_interface"][1]
        raise KeyError(key)

    def __iter__(self):
        return iter(self._NAMES)

    def __len__(self) -> int:
        return len(self._NAMES)


def validate_ysu_tendencies(
        ysu: Mapping[str, cp.ndarray], *,
        status: cp.ndarray | None = None,
        grid_id: int | None = None,
        producer_inputs: Mapping[str, cp.ndarray] | None = None) -> None:
    """Reject non-finite YSU output without modifying finite tendencies.

    FORENSIC FORM, the same one ``validate_sase_tendencies`` already
    carries, and for a reason this scheme learned from a field report
    rather than from theory.  A user on 1.5.2 lost an ERA5 run on its
    FIRST step to ``non-finite dtheta``, and that name is a false lead:
    the kernel writes ``dtheta`` from one solve whose only surface term
    is HFX (kernels/ysu.cu:480), so a non-finite HFX handed IN is
    reported as a dtheta produced OUT.  1.5 million fuzzed columns of
    finite, extreme, degenerate input never produce a non-finite rate at
    all, so when one appears an input is overwhelmingly the cause.  The
    output name alone therefore sent the reader to the PBL scheme when
    the defect was in the surface layer that fed it.  Naming the
    degenerate INPUTS is what closes that gap, and it costs nothing
    until a rate has already failed.

    WHICH output carries the poison depends on surface stability.  The
    input-poisoning sweep that found ``hfx`` was run before the sm_120
    DAZ work, so it was re-run against the shipping kernel on both
    branches, poisoning each of the 22 fixture-supplied inputs in turn:

    - STABLE (``br > 0``): the sweep's result holds exactly.  ``wstar3``
      is a literal zero (kernels/ysu.cu:177), so the surface buoyancy
      flux reaches nothing but ``rhs[0]`` (:480), and ``hfx`` is the
      UNIQUE input whose corruption surfaces as ``dtheta`` and nothing
      else.  That is the shape the field report described.
    - UNSTABLE (``br <= 0``): NO input has that signature.  The buoyancy
      flux sets the convective velocity scale (:173), which sets the
      mixed-layer diffusivity for momentum and moisture as well as heat,
      so a non-finite ``hfx`` poisons the whole column and the refusal
      names ``du``, first in launcher order.

    The producer detail below names ``hfx`` either way, which is the
    half the reader actually needs; only the output label moves.  Both
    branches are pinned in tests/test_ysu.py.

    ``status`` keeps the native single-readback path; the forensic work
    runs only on the failing branch of either path.
    """

    def _refuse(name: str) -> None:
        where = "" if grid_id is None else f" on domain {grid_id}"
        detail = ""
        if producer_inputs is not None:
            notes = _producer_forensics(
                producer_inputs, _YSU_POSITIVE_PRODUCER_INPUTS)
            detail = (
                "; producer inputs already degenerate: " + ", ".join(notes)
                if notes else
                "; every producer input was finite and in range, so the "
                "scheme's own arithmetic produced it")
        raise FloatingPointError(
            f"YSU returned non-finite {name} tendency{where}{detail}")

    if status is not None:
        invalid = validate_ysu_outputs(ysu, status, refuse=_refuse)
        if invalid is not None:
            _refuse(invalid)
        return
    for name, value in ysu.items():
        if not bool(cp.isfinite(value).all()):
            _refuse(name)


@dataclass
class RadiationResult:
    """Uncoupled LW/SW radiation heating plus surface radiative fluxes.

    ``rthratenlw`` and ``rthratensw`` follow WRF's separate mass-grid
    potential-temperature rates in K s-1.  The driver validates, copies, and
    holds both for diagnostics, then dry-mass-couples their sum.
    """

    rthratenlw: cp.ndarray
    rthratensw: cp.ndarray
    swdown: cp.ndarray
    glw: cp.ndarray
    # WRF radiation-driver carriers.  GSW is absorbed surface shortwave
    # evaluated with radiation-time albedo; COSZEN includes radconst's
    # half-radiation-interval hour-angle offset.  Optional defaults retain the
    # attachment API for schemes whose surface consumer needs neither.
    gsw: cp.ndarray | None = None
    coszen: cp.ndarray | None = None
    # OLR is WRF's top-of-atmosphere UPWARD longwave flux in W m-2
    # (Registry.EM_COMMON:1839, "TOA OUTGOING LONG WAVE"), the same number
    # RRTMG_LWRAD publishes from ``TOTUFLUX`` at the top level
    # (module_ra_rrtm.F:2296).  Optional because it is a property of the
    # LONGWAVE scheme, not of radiation being on: a shortwave-only pair and
    # the surface-flux analytic proxy have no top-of-atmosphere flux to
    # report, and publishing a zero for them would be a measured-looking
    # number no scheme computed.  A scheme that declares ``publishes_olr``
    # must supply it on every call; see ``PhysicsDriver._run_radiation``.
    olr: cp.ndarray | None = None


@dataclass
class CumulusResult:
    """WRF-named rates, rain outputs, and the per-column hold duration.

    Two driver contracts share this container.  A result WITHOUT
    ``nca_seconds`` keeps the Task-1 attachment contract: the rates
    replace the held cumulus tendencies wholesale and ``rainc`` is a
    per-due-call RAINC increment consumed once.  A result WITH
    ``nca_seconds`` (the production KF adapter) must also carry
    ``pratec``, WRF's persistent convective rain rate in mm s-1
    (module_cu_kfeta.F:2504): the driver then applies WRF's per-column
    NCA persistence, ``rainc`` is a scheme-level diagnostic the driver
    does not consume, and RAINC accumulates ``pratec*dt`` every step.

    Frozen tendencies may be supplied only when the state owns the matching
    prognostic category.  A custom warm-rain attachment must perform its own
    source-defined, latent-energy-consistent phase closure into QC/QR and omit
    ``rqicuten``/``rqscuten``; the driver cannot reconstruct that temperature
    adjustment from already diagnosed rates.
    """

    rthcuten: cp.ndarray
    rqvcuten: cp.ndarray
    rqccuten: cp.ndarray | None = None
    rqicuten: cp.ndarray | None = None
    rqrcuten: cp.ndarray | None = None
    rqscuten: cp.ndarray | None = None
    rainc: cp.ndarray | None = None
    nca_seconds: cp.ndarray | None = None
    pratec: cp.ndarray | None = None


class _NativeKFCumulusResult(CumulusResult):
    """Transient receipt for outputs owned by the exact production KF."""

    __slots__ = ("_owner",)

    def __init__(self, *, owner, **values):
        super().__init__(**values)
        self._owner = owner


_KF_VALIDATION_FIELDS = (
    "rthcuten", "rqvcuten", "rqccuten", "rqicuten",
    "rqrcuten", "rqscuten", "nca_seconds", "pratec",
)
_KF_VALIDATION_LABELS = (
    "cumulus rthcuten", "cumulus rqvcuten", "cumulus rqccuten",
    "cumulus rqicuten", "cumulus rqrcuten", "cumulus rqscuten",
    "cumulus nca_seconds", "cumulus PRATEC",
)


def _is_native_kf_result(
        result: CumulusResult, cumulus_callable, state: DomainState,
        cfg: RunConfig, shape: tuple[int, int, int]) -> bool:
    """Whether KF output provenance and metadata admit batched validation."""
    from gpuwm.core.kf import (
        KFPhaseMode,
        KainFritsch,
        kf_phase_mode_for_microphysics,
    )

    if (int(cfg.cu_physics) != 1
            or type(cumulus_callable) is not KainFritsch
            or type(result) is not _NativeKFCumulusResult
            or result._owner is not cumulus_callable):
        return False
    try:
        phase_mode = kf_phase_mode_for_microphysics(cfg.mp_physics)
    except (TypeError, ValueError):
        return False
    separate_ice = phase_mode == KFPhaseMode.SEPARATE_ICE_SNOW
    separate_snow = phase_mode in (
        KFPhaseMode.SEPARATE_SNOW, KFPhaseMode.SEPARATE_ICE_SNOW)
    if ((result.rqicuten is not None) != separate_ice
            or (result.rqscuten is not None) != separate_snow
            or (getattr(state, "qi", None) is not None) != separate_ice
            or (getattr(state, "qs", None) is not None) != separate_snow):
        return False
    for name in _KF_VALIDATION_FIELDS[:6]:
        value = getattr(result, name)
        if value is None:
            if name in ("rqicuten", "rqscuten"):
                continue
            return False
        if (type(value) is not cp.ndarray or value.shape != shape
                or value.dtype != DTYPE or not value.flags.c_contiguous):
            return False
    surface_shape = shape[1:]
    for name in _KF_VALIDATION_FIELDS[6:]:
        value = getattr(result, name)
        if (type(value) is not cp.ndarray or value.shape != surface_shape
                or value.dtype != DTYPE or not value.flags.c_contiguous):
            return False
    return True


def _validate_native_kf_result(
        result: _NativeKFCumulusResult, state: DomainState) -> None:
    """Validate native KF arrays with one ordered device status record."""
    from gpuwm.core.kf import validate_kf_outputs

    values = []
    active = 0
    placeholder = result.rthcuten
    for bit, name in enumerate(_KF_VALIDATION_FIELDS):
        value = getattr(result, name)
        if value is None:
            values.append(placeholder)
        else:
            values.append(value)
            active |= 1 << bit
    def _describe(flags: int) -> None:
        from gpuwm.core import health_ledger

        for bit, label in enumerate(_KF_VALIDATION_LABELS):
            if flags & (1 << bit):
                raise FloatingPointError(
                    f"{label} contains a non-finite value"
                    + health_ledger.deferred_note())

    invalid = validate_kf_outputs(
        tuple(values), active,
        state.scratch(
            (1,), "physics_validation_status").view(cp.uint32),
        describe=_describe)
    for bit, label in enumerate(_KF_VALIDATION_LABELS):
        if invalid & (1 << bit):
            raise FloatingPointError(
                f"{label} contains a non-finite value")


@dataclass
class PhysicsTendencies:
    """Held ARW slow tendencies plus coupled moist-scalar tendencies."""

    ru: cp.ndarray
    rv: cp.ndarray
    rtheta: cp.ndarray
    rqv: cp.ndarray
    rqc: cp.ndarray
    rqr: cp.ndarray | None = None
    rqi: cp.ndarray | None = None
    rqs: cp.ndarray | None = None

    @classmethod
    def zeros(
            cls, state: DomainState, *,
            optional_components: tuple[str, ...] = (),
            ) -> "PhysicsTendencies":
        nz, ny, nx = state.p.shape

        def scalar():
            return cp.zeros((nz, ny, nx), dtype=DTYPE)

        unknown = set(optional_components) - {"rqr", "rqi", "rqs"}
        if unknown:
            raise ValueError(f"unknown optional tendency components {unknown}")
        result = cls(cp.zeros((nz, ny, nx + 1), dtype=DTYPE),
                     cp.zeros((nz, ny + 1, nx), dtype=DTYPE),
                     scalar(), scalar(), scalar())
        for name in optional_components:
            setattr(result, name, scalar())
        return result

    def materialize(self, components: tuple[str, ...]) -> None:
        """Keep config-defined held categories present from time zero."""
        for name in components:
            if getattr(self, name) is None:
                setattr(self, name, cp.zeros_like(self.rqc))

    def add_to_slow(self, state: DomainState) -> None:
        """Add the held forward tendencies to the current RK slow slot."""
        state.ru_t += self.ru
        state.rv_t += self.rv
        state.rth_t += self.rtheta
        # SASE-only w forcing.  ``rw`` is deliberately a PLAIN attribute,
        # never a dataclass field: the restart component manifest
        # (TENDENCY_COMPONENTS) stays byte-identical, and the held value
        # is restart-REBUILT under the enforced SASE bldt == 0 invariant
        # (every compute() replaces it before any read).  Neither YSU nor
        # MYNN produces a w tendency, so every existing path is
        # byte-inert here.
        rw = getattr(self, "rw", None)
        if rw is not None:
            state.rw_t += rw

    def scalar_for(self, name: str) -> cp.ndarray | None:
        return {"qv": self.rqv, "qc": self.rqc, "qr": self.rqr,
                "qi": self.rqi, "qs": self.rqs}.get(name)


def couple_ysu_tendencies(state: DomainState, cfg: RunConfig,
                          ysu: Mapping[str, cp.ndarray]) -> PhysicsTendencies:
    """Mass-couple and A-grid-to-C-grid interpolate YSU rates.

    WRF ``calculate_phy_tend`` first multiplies every A-grid PBL rate by
    ``c1h*mut+c2h``.  ``phy_bl_ten`` then uses ``add_a2c_u/v`` for momentum
    and ``add_a2a`` for theta/moisture.  Finally ``rk_addtend_dry`` divides
    the forward momentum/scalar tendencies by their map factor before they
    join the working slow slot.  Moist scalar tendencies remain in coupled
    form for ``rk_update_scalar``.
    """
    chm = (state.c1h[:, None, None] * state.total_mu()[None]
           + state.c2h[:, None, None])
    mass_u = chm * ysu["du"]
    mass_v = chm * ysu["dv"]

    # Periodic halo semantics: face f lies between cells f-1 and f; the
    # final staggered face duplicates face zero.
    ru = 0.5 * (mass_u + cp.roll(mass_u, 1, axis=2))
    ru = cp.concatenate([ru, ru[:, :, :1]], axis=2)
    rv = 0.5 * (mass_v + cp.roll(mass_v, 1, axis=1))
    rv = cp.concatenate([rv, rv[:, :1, :]], axis=1)
    rtheta = chm * ysu["dtheta"]
    rqv = chm * ysu["dqv"]
    rqc = chm * ysu["dqc"]
    # WRF phy_bl_ten adds RQIBLTEN exactly when the moist set carries ice
    # (module_physics_addtendc.F, IF(F_QI) branch); schemes that mix ice
    # return dqi and it couples like the other moist scalars.
    dqi = ysu.get("dqi")
    rqi = None if dqi is None else chm * dqi

    # WRF's boundary-forced loop bounds omit the physical outer cell and
    # normal velocity faces for specified .OR. nested domains
    # (module_physics_addtendc.F:2312-2319 add_a2a, :2412-2419 add_a2c_u,
    # :2466-2473 add_a2c_v).  gpuwm has no halo cells, so zero those slots.
    if cfg.specified or cfg.nested:
        _specified_mass_mask(rtheta)
        _specified_mass_mask(rqv)
        _specified_mass_mask(rqc)
        if rqi is not None:
            _specified_mass_mask(rqi)
        ru[..., 0, :] = 0.0
        ru[..., -1, :] = 0.0
        ru[..., :, 0] = 0.0
        ru[..., :, -1] = 0.0
        rv[..., 0, :] = 0.0
        rv[..., -1, :] = 0.0
        rv[..., :, 0] = 0.0
        rv[..., :, -1] = 0.0
    else:
        if cfg.open_x:
            ru[:, :, 0] = 0.0
            ru[:, :, -1] = 0.0
        if cfg.open_y:
            rv[:, 0, :] = 0.0
            rv[:, -1, :] = 0.0

    if state.has_msf:
        ru = ru / state.msfu[None]
        rv = rv / state.msfv[None]
        # AND THE DIVISION UNDOES THE DUPLICATION MADE AT :1277-1280.
        # ``msfu``/``msfv`` are staggered STATICS and their own alias slot
        # was never closed: on a window of a larger analysis it holds the
        # map factor one column/row OUTSIDE the window, not face zero's.  So
        # ``ru[..., nx] = ru[..., 0] / msfu[..., nx]`` while
        # ``ru[..., 0] = ru[..., 0] / msfu[..., 0]``, and the periodic
        # identity this function states in its own comment stops holding
        # right here.  Restoring it is what makes the coupled PBL momentum
        # tendency survive a decomposition: a tiled run re-derives the alias
        # slot from face 0 (tilestream/spec.py:34-52) and the monolithic run
        # must agree with it.  MEASURED, 128x96x49, full physics, one step:
        # driver/pbl_tendencies/rv was the last of 158 carriers still
        # differing 1 GPU vs 4; with this it is exact.
        #
        # Closing the tendency rather than the map factor is deliberate.
        # ru/rv are CARRIERS, so a rank's copy is refreshed by the halo
        # exchange every step and this write is inert there; the map factors
        # are setup arrays that no exchange refreshes, and rewriting a
        # tile's outermost map-factor column would corrupt the outer edge of
        # its halo ring permanently.
        if not (cfg.specified or cfg.nested or cfg.open_x):
            ru[..., -1] = ru[..., 0]
        if not (cfg.specified or cfg.nested or cfg.open_y):
            rv[..., -1, :] = rv[..., 0, :]
        rtheta = rtheta / state.msft[None]
    return PhysicsTendencies(cp.ascontiguousarray(ru),
                             cp.ascontiguousarray(rv),
                             cp.ascontiguousarray(rtheta),
                             cp.ascontiguousarray(rqv),
                             cp.ascontiguousarray(rqc),
                             rqi=(None if rqi is None
                                  else cp.ascontiguousarray(rqi)))


def couple_column_tendencies(
        state: DomainState, cfg: RunConfig, *, rtheta: cp.ndarray | None = None,
        rqv: cp.ndarray | None = None, rqc: cp.ndarray | None = None,
        rqr: cp.ndarray | None = None, rqi: cp.ndarray | None = None,
        rqs: cp.ndarray | None = None) -> PhysicsTendencies:
    """Dry-mass-couple mass-grid radiation/cumulus physical rates.

    Theta joins the dry slow-tendency stack and therefore receives WRF's
    mass-point map-factor division.  Moist species stay mass-coupled for
    ``rk_update_scalar``, matching :func:`couple_ysu_tendencies`.
    """
    nz, ny, nx = state.p.shape
    shape = (nz, ny, nx)
    zero = cp.zeros(shape, dtype=DTYPE)
    chm = (state.c1h[:, None, None] * state.total_mu()[None]
           + state.c2h[:, None, None])

    theta = zero.copy() if rtheta is None else chm * rtheta
    qv = zero.copy() if rqv is None else chm * rqv
    qc = zero.copy() if rqc is None else chm * rqc
    qr = None if rqr is None else chm * rqr
    qi = None if rqi is None else chm * rqi
    qs = None if rqs is None else chm * rqs
    # WRF add_a2a fires for specified .OR. nested
    # (module_physics_addtendc.F:2312-2319).
    if cfg.specified or cfg.nested:
        for array in (theta, qv, qc):
            _specified_mass_mask(array)
        for array in (qr, qi, qs):
            if array is not None:
                _specified_mass_mask(array)
    if state.has_msf:
        theta = theta / state.msft[None]
    return PhysicsTendencies(
        cp.zeros((nz, ny, nx + 1), dtype=DTYPE),
        cp.zeros((nz, ny + 1, nx), dtype=DTYPE),
        cp.ascontiguousarray(theta), cp.ascontiguousarray(qv),
        cp.ascontiguousarray(qc),
        None if qr is None else cp.ascontiguousarray(qr),
        None if qi is None else cp.ascontiguousarray(qi),
        None if qs is None else cp.ascontiguousarray(qs))


def _prepare_atmosphere(state: DomainState) -> dict[str, cp.ndarray]:
    """CuPy transcription of WRF ``phy_prep`` fields used in Phase 3."""
    supplied = getattr(state, "prepared_physics_atmosphere", None)
    if supplied is not None:
        # A column state that already lives on the physics grid (A-grid
        # winds, height-coordinate pressures: gpuwm/core/
        # mpas_column_batch.py) supplies its own phy_prep product.  The
        # destagger below is C-grid face interpolation and the pressure
        # loop is ARW mass-coordinate arithmetic; neither is defined for
        # such a state, so the carrier hands the driver the same dict this
        # function builds, with every key filled from the caller's own
        # fields.  The ORCHESTRATION (cadence, holds, buckets, scheme
        # calls) above and below this seam is shared, not duplicated.
        return supplied()
    nz, ny, nx = state.p.shape
    theta = cp.ascontiguousarray(state.total_theta())
    pressure = state.p
    exner = cp.ascontiguousarray((pressure / DTYPE(c.P0)) ** DTYPE(c.RCP))
    temperature = cp.ascontiguousarray(theta * exner)
    u = cp.ascontiguousarray(0.5 * (state.u[:, :, :-1]
                                    + state.u[:, :, 1:]))
    v = cp.ascontiguousarray(0.5 * (state.v[:, :-1, :]
                                    + state.v[:, 1:, :]))

    phb = state.phb
    phb3 = phb[:, None, None] if phb.ndim == 1 else phb
    z_interface = cp.ascontiguousarray((phb3 + state.php) / DTYPE(c.G))
    dz = cp.ascontiguousarray(z_interface[1:] - z_interface[:-1])

    # WRF feeds its physics driver families the HYDROSTATIC pressures
    # p_hyd/p_hyd_w built in phy_prep, monotone by construction
    # (module_big_step_utilities_em.F:4943-4970): p_hyd_w(kte) = p_top,
    # p_hyd_w(k) = p_hyd_w(k+1) - (1+qtot)*(c1(k)*MUT+c2(k))*dnw(k)
    # integrated downward with qtot summing every moist MASS species,
    # and p_hyd = the interface mean.  Radiation, surface, PBL, cumulus,
    # and shallow-cu all receive these (first_rk_step_part1.F:279, :632,
    # :1129, :1379, :1578); microphysics and diffusion keep the EOS
    # pressure (solve_em.F:3724), so ``state.p`` -- and the exner/
    # temperature computed from it above, WRF's pi_phy/t_phy -- stay
    # untouched.  Feeding the EOS p here instead wired RRTMG's McICA
    # pmid-monotonicity stop to a field WRF never guards (native-dt
    # investigation, 2026-07-16).
    qtot = state.scratch((nz, ny, nx), "physics_qtot")
    qtot[...] = 0.0
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
        species = getattr(state, name, None)
        if species is not None:
            qtot += species
    layer = ((DTYPE(1.0) + qtot)
             * (state.c1h[:, None, None] * state.total_mu()[None]
                + state.c2h[:, None, None])
             * state.dnw[:, None, None])
    p_interface = cp.empty((nz + 1, ny, nx), dtype=DTYPE)
    p_interface[nz] = state.p_top
    # Sequential downward integration in the Fortran's loop order.
    for k in range(nz - 1, -1, -1):
        cp.subtract(p_interface[k + 1], layer[k], out=p_interface[k])
    pressure = 0.5 * (p_interface[:-1] + p_interface[1:])

    qv = (state.qv if state.qv is not None
          else state.scratch((nz, ny, nx), "physics_dry_qv"))
    qc = (state.qc if state.qc is not None
          else state.scratch((nz, ny, nx), "physics_dry_qc"))
    # Morrison states carry prognostic cloud ice; the PBL/radiation seams
    # must see it (the qi=0 feed was the audit's "YSU qi seam" finding).
    if getattr(state, "qi", None) is not None:
        qi = state.qi
    else:
        qi = state.scratch((nz, ny, nx), "physics_qi")
        qi[...] = 0.0
    if getattr(state, "qs", None) is not None:
        qs = state.qs
    else:
        qs = state.scratch((nz, ny, nx), "physics_qs")
        qs[...] = 0.0
    # WRF phy_prep's physics density is (1 + qv) / ALT
    # (module_big_step_utilities_em.F:4856), not the virtual-temperature
    # ideal-gas reconstruction.  MYNN's surface layer consumes this exact
    # field in its flux coefficients.
    rho = cp.ascontiguousarray((DTYPE(1.0) + qv) / state.alt)
    return {"theta": theta, "temperature": temperature, "pressure": pressure,
            "p_interface": p_interface, "exner": exner, "u": u, "v": v,
            "z_interface": z_interface, "dz": dz, "qv": qv, "qc": qc,
            "qi": qi, "qs": qs, "rho": rho}


#: WRF ``mp_init`` receipts, by driver.  See
#: :attr:`PhysicsDriver.microphysics_init_receipt`: this is a side table
#: rather than an instance attribute ON PURPOSE.
#:
#: ``gpuwm/io/restart.py::_driver_manifest`` walks ``vars(driver)`` and
#: REFUSES to write a checkpoint containing any driver attribute it has not
#: classified as serialized or rebuilt -- a deliberate exhaustiveness gate on
#: driver state, and a good one.  The mp_init receipt is not driver state: it
#: is construction metadata that ``initialize_physics`` reproduces on every
#: resume, and WRF itself keeps no analogue of it on the grid.  Parking it
#: here keeps that gate exhaustive over the things it exists to protect
#: instead of forcing a classification entry in a file this package does not
#: own.  (An integration request is filed to add
#: ``"microphysics_init_receipt"`` to ``DRIVER_REBUILT_ATTRS``; when it lands
#: this may become a plain instance attribute with no behavioural change.)
#:
#: Weak-keyed, so a discarded driver's receipt is collected with it.
_MICROPHYSICS_INIT_RECEIPTS: "weakref.WeakKeyDictionary" = \
    weakref.WeakKeyDictionary()


class PhysicsDriver:
    """Persistent surface state, diagnostics, scheduler, and held tendencies."""

    @property
    def microphysics_init_receipt(self) -> dict[str, object]:
        """What WRF's ``mp_init`` did for this domain, as a receipt.

        ``{}`` for every scheme with no domain-construction step, and for a
        driver built directly rather than through :func:`initialize_physics`.
        For ``mp_physics = 28`` it is
        ``{"thompson_aerosol_profile": {"ccn": bool, "in": bool}}`` -- which
        of ``thompson_init``'s two INDEPENDENT presence-gated fills ran
        (``module_mp_thompson.F:493`` for CCN, ``:531`` for IN).  A receipt
        of ``{"ccn": False, "in": False}`` is the normal, correct answer on
        an aerosol-bearing domain.

        WHAT IT DOES NOT MEAN, on a RESUME.  gpuwm's restart order is
        prepare -> ``initialize_physics`` -> ``restore_restart``
        (``gpuwm/runtime.py``'s docstring for the integration loop), so a
        resumed mp=28 domain is still all-zero aerosol when this runs and
        the receipt will say both fills happened -- and then
        ``restore_restart`` overwrites ``nwfa``/``nifa``/``nwfa2d`` with the
        checkpointed fields.  The resumed forecast therefore integrates the
        CHECKPOINTED aerosol, not the synthetic profile; the receipt is a
        record of what the init path did, not of what the run ends up with.
        Gated by ``tests/test_physics.py::
        test_a_resumed_mp28_domain_keeps_its_checkpointed_aerosol``.
        """
        return _MICROPHYSICS_INIT_RECEIPTS.get(self, {})
    #: SASE state, declared on the CLASS so it answers on any instance.
    #:
    #: ``__init__`` sets all four for a real driver.  Several tests build
    #: a deliberately partial driver to exercise one seam in isolation,
    #: and the dispatch path reads ``sase_active`` on every PBL step --
    #: a class default is what lets those doubles keep working and,
    #: more to the point, means the hot path can never raise
    #: AttributeError on a driver someone assembled another way.  False
    #: and None are the correct answers for a driver that has no
    #: closure attached.
    sase_active: bool = False
    sase_flux_diag: dict[str, cp.ndarray] | None = None
    last_sase_ledger: dict[str, float] | None = None
    sase_nan_guard_fires: int = 0

    #: Where this domain's downward longwave comes from, as one of
    #: ``"scheme"`` (a longwave scheme computes it every radiation call),
    #: ``"declared"`` (the caller typed a constant or handed a field),
    #: or ``"unused"`` (nothing reads it and nothing publishes it).
    #: :func:`initialize_physics` sets it; the class default answers for a
    #: driver assembled directly, whose GLW buffer its builder owns.
    #:
    #: Read by :func:`gpuwm.runtime.resolved_config_report`, so a run that
    #: is integrating a CONSTANT downward longwave says so in its receipt
    #: instead of publishing a GLW row that looks like a measurement.
    glw_provenance: str = "declared"

    #: CLASS DEFAULTS for the three bookkeeping slots the hot path reads
    #: every step, on exactly the argument the ``sase_active`` default
    #: above is written on: a driver someone assembled another way -- a
    #: test double, a probe -- must not raise AttributeError inside
    #: ``compute``, and a plain attribute read is what keeps the hot path
    #: cheap.  ``__init__`` sets all three, so a real driver never uses
    #: these.
    #:
    #: ``carriers`` defaults to None rather than to a contract, and the
    #: consumer seam REFUSES on None.  A shared mutable class default
    #: would be one contract for every driver in the process, and a
    #: silently-created empty one would let a driver assembled without a
    #: contract consume whatever its buffers held -- which is the defect
    #: the contract exists to close, reintroduced through the back door.
    carriers = None
    surface_moisture_ledger = None
    carriers_need_producer_refresh = False

    def __init__(self, state: DomainState, cfg: RunConfig,
                 fields: dict[str, cp.ndarray], sfclay_result: SFClayResult,
                 noah_params, radiation=None, cumulus=None,
                 noahmp_params=None, noahmp_geometry=None,
                 ruc_params=None, glw_provenance="declared",
                 carriers=None):
        self.state = state
        self.fields = fields
        # Where this domain's downward longwave came from; see the class
        # attribute above.  A driver assembled directly owns its own GLW
        # buffer, so the default says so.
        self.glw_provenance = glw_provenance
        self.sfclay_result = sfclay_result
        self.mynn_sfclay_result = (
            MynnSurfaceResult(**{
                name: fields[name] for name in MYNN_SURFACE_OUTPUTS
            })
            if cfg.sf_sfclay_physics == 5 else None
        )
        self.mynn_sfclay_sea_result = (
            MynnSurfaceResult(**{
                name: fields[f"{name}_sea"]
                for name in MYNN_SURFACE_OUTPUTS
            })
            if (cfg.sf_sfclay_physics == 5
                and cfg.sf_surface_physics == 3) else None
        )
        self.noah_params = noah_params
        # Noah-MP's parameter bundle and solar geometry are separate
        # attributes rather than extra members of ``noah_params``, because
        # restart identity dispatches on the scheme value through
        # LAND_SURFACE_PARAMETER_SOURCES and must be able to say which
        # bundle a checkpoint ran with.
        self.noahmp_params = noahmp_params
        self.noahmp_geometry = noahmp_geometry
        self.noahmp_soil_thickness_m = NOAH_LAYER_THICKNESS_M
        self.last_noahmp_census: dict[str, int] | None = None
        # One advisory per run for the Shin-Hong passenger repair
        # (task #206); the repair itself applies on every step that
        # needs it.
        self._shinhong_passenger_advisory = False
        # One advisory per run for the Shin-Hong ENTRY heal (the
        # upstream half of the same defect class: state.e_sgs arriving
        # below WRF's shinhonginit floor, or non-finite); the heal
        # itself applies on every step that needs it.
        self._shinhong_entry_advisory = False
        #: The construction-time surface classification receipt --
        #: which source decided XLAND and the per-class column counts.
        #: initialize_physics fills it right after construction; a
        #: hand-built driver keeps the empty dict.
        self.surface_classification: dict = {}
        if int(cfg.sf_surface_physics) == 4:
            if noahmp_params is None:
                raise ValueError(
                    "sf_surface_physics=4 requires the Noah-MP parameter "
                    "bundle; initialize_physics builds it")
            if noahmp_geometry is None:
                raise ValueError(
                    "sf_surface_physics=4 requires a NoahmpSolarGeometry: "
                    "the LSM reads COSZ, XLAT, JULIAN and YR, and gpuwm's "
                    "surface seam carries none of them.  Pass "
                    "latitude_deg/longitude_deg/start_time to "
                    "initialize_physics.")
        elif noahmp_params is not None or noahmp_geometry is not None:
            raise ValueError(
                "Noah-MP parameters/geometry cannot be attached to "
                f"sf_surface_physics={cfg.sf_surface_physics}")
        # RUC's bundle is its own attribute for the same reason Noah-MP's is:
        # restart identity dispatches on the scheme value and must be able to
        # name which table set a checkpoint ran with.
        self.ruc_params = ruc_params
        self.last_ruc_census: dict[str, int] | None = None
        if int(cfg.sf_surface_physics) == 3:
            if ruc_params is None:
                raise ValueError(
                    "sf_surface_physics=3 requires the RUC parameter "
                    "bundle; initialize_physics builds it")
        elif ruc_params is not None:
            raise ValueError(
                "RUC parameters cannot be attached to "
                f"sf_surface_physics={cfg.sf_surface_physics}")
        self.radiation_callable = radiation
        self.cumulus_callable = cumulus
        self.ra_physics = cfg.ra_physics
        self.ra_lw_physics, self.ra_sw_physics = radiation_scheme_ids(cfg)
        self.radiation_active = bool(
            self.ra_lw_physics or self.ra_sw_physics)
        self.cu_physics = cfg.cu_physics
        self.mp_physics = cfg.mp_physics
        # SR validator envelope, granted by shared-expression family
        # membership rather than per scheme (1.9.1 D2): wsm6.cu:402 and
        # wdm6.cu:635 form the identical positive-sum SR expression, and
        # milbrandt2.cu:2340 the same quotient once per call, so all
        # three own their WRF expression-order roundoff allowance.
        (self._sr_roundoff_upper, self._sr_roundoff_max_ulps,
         self._wsm6_minor_loops) = _sr_roundoff_envelope(
            self.mp_physics, cfg.dt)
        self.surface_enabled = bool(
            cfg.sf_sfclay_physics or cfg.sf_surface_physics
            or cfg.bl_pbl_physics)
        # Fail at construction, not on the first due surface step, if a
        # selector value has no scheme routed to it.  compute() re-resolves
        # from the cfg it is handed, so this attribute is a receipt rather
        # than the dispatch authority.
        self.scheme_dispatch = resolve_physics_dispatch(cfg)
        cumulus_components = _cumulus_optional_tendency_components(cfg)
        pbl_components = _pbl_optional_tendency_components(cfg)
        composed_components = _composed_optional_tendency_components(cfg)
        pbl_initial_components = (
            composed_components if physics_reuses_pbl_composition(cfg)
            else pbl_components)
        self.pbl_tendencies = PhysicsTendencies.zeros(
            state, optional_components=pbl_initial_components)
        self.radiation_tendencies = PhysicsTendencies.zeros(state)
        self.cumulus_tendencies = PhysicsTendencies.zeros(
            state, optional_components=cumulus_components)
        self.rthratenlw = cp.zeros(state.p.shape, dtype=DTYPE)
        self.rthratensw = cp.zeros(state.p.shape, dtype=DTYPE)
        # GF forcing lanes (WRF GFDRV's RTHFTEN/RQVFTEN/RTHBLTEN/RQVBLTEN
        # inputs).  The dynamics pair is DYNAMICS-owned: whichever
        # integrator runs supplies it.  On the ARW path that is the
        # dycore's own export, bound here to the state buffers
        # ``rthften``/``rqvften`` it writes in place at RK stage 1 of every
        # step (gpuwm/core/dycore.py::capture_advective_theta_forcing and
        # gpuwm/core/moist.py::_capture_advective_qv_forcing); a host
        # driver that integrates its OWN dynamics -- the MPAS column-batch
        # seam, whose state is not a DomainState and carries neither
        # attribute -- overwrites these two attributes with its own
        # persistent buffers after construction.  ``None`` still means "no
        # dynamics forcing exporter" and the GF adapter feeds zeros exactly
        # as before, which is what a cumulus scheme outside
        # CUMULUS_ADVECTIVE_FORCING_SCHEMES gets.  The PBL pair is retained by
        # _couple_pbl_slot from whichever scheme holds the PBL slot --
        # YSU, MYJ, MYNN, Shin-Hong or SASE all hand it the same raw
        # dtheta/dqv rates; WRF semantics: RTHBLTEN/RQVBLTEN persist
        # between PBL calls, so retention of the last computed rates IS
        # the WRF contract.  gf_dx_column optionally carries a
        # per-column dx ([ny, nx], metres) for meshes where one scalar dx
        # is a lie; None keeps cfg.dx.
        self.gf_rthdynten: cp.ndarray | None = getattr(
            state, "rthften", None)
        self.gf_rqvdynten: cp.ndarray | None = getattr(
            state, "rqvften", None)
        self.gf_rthblten: cp.ndarray | None = None
        self.gf_rqvblten: cp.ndarray | None = None
        self.gf_dx_column: cp.ndarray | None = None
        # OLR, WRF's TOA outgoing longwave (Registry.EM_COMMON:1839).  The
        # buffer exists exactly when the attached longwave scheme declares
        # it computes a top-of-atmosphere upward flux, so the variable's
        # presence in a wrfout is the statement "a TOA longwave producer
        # ran", the same way an absent XKMH says no horizontal mixing
        # operator ran.  Zero-valued until the first due radiation call --
        # which is also what WRF publishes in its own t=0 history frame,
        # because OLR is a plain zero-initialised ``misc`` array and the
        # time-0 write precedes the first radiation call.
        #
        # A RESTART IS NOT THE FIRST CALL OF A RUN, so these zeros are
        # overwritten by the checkpoint's own OLR on resume
        # (restart.DRIVER_CHECKPOINT_ONLY_ATTRS).  They were not, and a
        # resumed run published 0 W m-2 into every frame before its first
        # post-restart radiation call -- the only variable of 77 that
        # differed across a restart, on any configuration.
        self.olr = (
            cp.zeros(state.p.shape[1:], dtype=DTYPE)
            if self.radiation_active and getattr(
                self.radiation_callable, "publishes_olr", False)
            else None)
        self.tendencies = (
            self.pbl_tendencies
            if (not (radiation_enabled(cfg) or cfg.cu_physics)
                or physics_reuses_pbl_composition(cfg))
            else PhysicsTendencies.zeros(
                state, optional_components=composed_components))
        self.last_ysu: dict[str, cp.ndarray] | None = None
        self.call_counts = {"radiation": 0, "sfclay": 0,
                            "noah": 0, "ysu": 0, "cumulus": 0,
                            "cumulus_history": 0}
        self.ysu_nan_guard_fires = 0
        #: Opt-in :class:`~gpuwm.core.surface_moisture_ledger.
        #: SurfaceMoistureLedger`, or ``None`` on every ordinary run.  The
        #: two branches that read it are the ONLY places it is consulted,
        #: and neither writes a physics field or a restart-carried scratch
        #: buffer, so attaching one cannot move a forecast.  Bound here
        #: rather than left to ``getattr`` so that reading it is a plain
        #: attribute access on the hot path.
        self.surface_moisture_ledger = None
        #: THE SURFACE-RADIATION CARRIER CONTRACT
        #: (:mod:`gpuwm.core.radiation_carriers`).  Always present, never
        #: opt-in: it is the authority that stands between a land-surface
        #: scheme and a radiative buffer nobody wrote.  ``initialize_physics``
        #: seeds it with each carrier's true origin; the radiation seam, the
        #: analytic COSZEN provider and :meth:`set_forcing` are the only
        #: writers; :meth:`compute` reads it immediately before the LSM call.
        #: A driver assembled directly gets one too -- that is the whole
        #: point of putting the check at consumption rather than at the
        #: config door.
        self.carriers = (
            carriers if carriers is not None
            else CarrierContract(
                getattr(cfg, "surface_radiation_policy",
                        SURFACE_RADIATION_POLICY_REQUIRED)))
        #: True only on a resume from a checkpoint written before the
        #: carrier contract existed, where the provenance and ages of the
        #: carriers are simply not in the file.  The first surface call
        #: after such a resume forces a producer refresh rather than
        #: trusting a rebuilt record; see gpuwm/io/restart.py's
        #: _restore_carriers for why guessing is the worse option.
        self.carriers_need_producer_refresh = False
        # SASE claims the PBL slot when selected.  The ``sase`` count key
        # and every SASE attribute exist ONLY when it is active, so every
        # other configuration's driver header stays byte-identical.
        self.sase_active = cfg.bl_pbl_physics == SASE_PBL_SCHEME
        self.last_sase_ledger: dict[str, float] | None = None
        self.sase_nan_guard_fires = 0
        if self.sase_active:
            self.call_counts["sase"] = 0
            # SPLIT SUBGRID-FLUX DIAGNOSTIC (cfg.sase_flux_diag,
            # output-only).  Four z-FACE fields on the mass column,
            # (nz + 1, ny, nx) FP32, registered exactly like W, holding
            # the closure's own vertical subgrid fluxes with the venting
            # channel SEPARATED from the K_v implicit-diffusion channel:
            #
            #   fqv_vent / fqv_diff   water vapour  [kg m-2 s-1]
            #   fth_vent / fth_diff   heat          [W m-2]
            #
            # BOTH POSITIVE UPWARD, both divided by the SAME lowest-level
            # moist density plane the venting deposit divides by, so the
            # two channels are summable face by face and their
            # convergences add up to the model's own scalar increment.
            # DRIVER PERSISTENTS, not step transients: output needs them
            # after the step ends.  Allocated as ZEROS at init because
            # the writer creates a netCDF variable the first time a name
            # appears and Time is unlimited -- a lazily-created field
            # would leave frame 0 backfilled with the netCDF fill value
            # instead of an honest zero (no SASE step has run at the t=0
            # frame).  Default None: no allocation, one attribute test
            # per step, and output_fields() keeps its historical key set.
            self.sase_flux_diag: dict[str, cp.ndarray] | None = (
                {name: cp.zeros((state.p.shape[0] + 1,) +
                                state.p.shape[1:], dtype=DTYPE)
                 for name in ("fqv_vent", "fqv_diff",
                              "fth_vent", "fth_diff")}
                if cfg.sase_flux_diag else None)
        # HORIZONTAL EDDY-VISCOSITY DIAGNOSTIC (cfg.hmix_k_diag,
        # output-only).  Two (nz, ny, nx) FP32 driver persistents named
        # for whichever producer this run actually has -- see
        # _HMIX_K_DIAG_NAMES.  Zeros at init, for the flux diagnostic's
        # reason: the writer creates a netCDF variable the first time a
        # name appears, so a lazily created field would leave frame 0
        # backfilled with the fill value rather than an honest zero.
        # None when the key is off OR when the run HAS no horizontal
        # mixing producer -- see hmix_k_diag_names for why the
        # no-producer case publishes nothing rather than zeros.
        names = hmix_k_diag_names(cfg)
        self.hmix_k_diag: dict[str, cp.ndarray] | None = (
            {name: cp.zeros(state.p.shape, dtype=DTYPE) for name in names}
            if cfg.hmix_k_diag and names else None)
        self.bldt_seconds = _physics_interval_seconds(cfg.bldt, cfg.dt)
        self.stepbl = max(int(round(self.bldt_seconds / cfg.dt)), 1)
        # Preserve the positive Phase-3 ``radt`` spelling while making the
        # explicit Phase-4 name authoritative for all new configurations.
        self.radt_minutes = (cfg.radt if cfg.radt > 0.0
                             else cfg.radt_minutes)
        self.cudt_minutes = cfg.cudt_minutes
        # Noah option selectors. The kernel already branches on all three;
        # carrying them here is the configuration path that was missing.
        self.noah_usemonalb = bool(cfg.usemonalb)
        self.noah_rdlai2d = bool(cfg.rdlai2d)
        self.noah_opt_thcnd = int(cfg.opt_thcnd)
        self.stepra = _physics_interval_steps(self.radt_minutes, cfg.dt)
        self.stepcu = _physics_interval_steps(self.cudt_minutes, cfg.dt)
        self.radt_seconds = self.stepra * float(cfg.dt)
        self.cudt_seconds = self.stepcu * float(cfg.dt)
        surface_shape = state.mup.shape

        def zero_surface():
            return cp.zeros(surface_shape, dtype=DTYPE)

        micro_slots = dict(microphysics_scratch_slots(cfg.mp_physics))
        if micro_slots:
            # The scheme kernels update these carrying scratch arrays only
            # after every pre-RK PhysicsDriver consumer has finished.  The
            # driver therefore keeps views of the canonical accumulators,
            # not a second copied set.
            micro = {name: state.scratch(surface_shape, slot)
                     for name, slot in micro_slots.items()}
            self.microphysics = MicrophysicsDiagnostics(
                rainnc=micro["rainnc"], rainncv=micro["rainncv"],
                sr=micro["sr"], snownc=micro.get("snownc"),
                snowncv=micro.get("snowncv"),
                graupelnc=micro.get("graupelnc"),
                graupelncv=micro.get("graupelncv"),
                hailnc=micro.get("hailnc"), hailncv=micro.get("hailncv"))
        else:
            # mp_physics=0 has no canonical scratch accumulators.  Retain
            # the historical all-zero diagnostic object for output plumbing.
            self.microphysics = MicrophysicsDiagnostics(
                rainnc=zero_surface(), rainncv=zero_surface(),
                sr=zero_surface())
        self.microphysics_updates = 0
        # One output-frame handoff.  The state-owned refl_10cm scratch array
        # is aliased here only between an output-due microphysics call and
        # _write_case_output's single consumption.  Restart classifies this
        # ephemeral reference as rebuild-on-resume (PROVENANCE.md D2).
        self.refl_10cm = None
        self._pending_rainbl = zero_surface()
        self.rainc = (state.scratch(state.mup.shape, "cu_rainc")
                      if cfg.cu_physics else None)
        # Host mirror of an exceptional in-flight KF expiry transition.
        # The device mask remains authoritative for direct/synthetic callers;
        # this flag lets the normal compute-entry recovery path avoid probing
        # a mask that the prior step already finalized.
        self._cu_expiry_pending = False
        if cfg.cu_physics:
            # WRF KF driver persistence (Task 6b).  kf_init seeds
            # NCA = -100 so every column is eligible on the first call
            # (module_cu_kfeta.F:3152-3156); PRATEC/RAINCV and the stored
            # per-column rates live on the grid between scheme calls
            # (module_cumulus_driver.F:779-785 and 1661).  All of this is
            # Task 8 restart state.
            nz = state.p.shape[0]
            self.cu_nca = state.scratch(surface_shape, "cu_nca")
            self.cu_nca[...] = DTYPE(-100.0)
            self.cu_pratec = state.scratch(surface_shape, "cu_pratec")
            self.cu_pratec[...] = 0.0
            self.cu_raincv = state.scratch(surface_shape, "cu_raincv")
            self.cu_raincv[...] = 0.0
            # _advance_cumulus_clock identifies columns whose held rates
            # expire on this step.  The current RK target is composed first;
            # then the raw/held copies are cleared before post-RK Morrison,
            # matching solve_em's advance_ppt -> microphysics order.
            self.cu_expiring = state.scratch(surface_shape, "cu_expiring")
            self.cu_expiring[...] = DTYPE(0.0)
            self.cu_rates = {
                name: state.scratch((nz, *surface_shape), f"cu_{name}")
                for name in ("rthcuten", "rqvcuten", "rqccuten",
                             "rqicuten", "rqrcuten", "rqscuten")}
            for array in self.cu_rates.values():
                array[...] = 0.0
        else:
            self.cu_nca = None
            self.cu_pratec = None
            self.cu_raincv = None
            self.cu_expiring = None
            self.cu_rates = None
        if self.mp_physics == 18:
            # Imported here so physics -> microphysics -> nssl2_runtime does
            # not observe a partially initialized module. The binding owns
            # every selector-reachable callback and reusable device buffer.
            from gpuwm.core.nssl2_contract import (
                require_ported_nssl2_mode,
                resolve_nssl2_mode_for_config,
            )
            from gpuwm.core.nssl2_default_hooks import (
                make_nssl2_production_binding,
            )
            # WRF has one NSSL scheme and four variant selectors
            # (share/module_check_a_mundo.F:3433-3455).  Resolve them into
            # the mode the binding freezes for the life of the domain;
            # validate_run_config has already refused unported combinations.
            nssl2_mode = require_ported_nssl2_mode(
                resolve_nssl2_mode_for_config(cfg))
            self.nssl2_binding = make_nssl2_production_binding(
                state, cfg.dt, mode=nssl2_mode)
            # The variant's absent categories are pinned to exact zero
            # HERE, at domain construction, before the first history or
            # restart write can publish whatever the allocation left --
            # see pin_absent_nssl2_fields for why this is per-step too.
            from gpuwm.core.nssl2_runtime import pin_absent_nssl2_fields
            pin_absent_nssl2_fields(state, nssl2_mode)
        else:
            self.nssl2_binding = None

    def accept_microphysics(self, result: MicrophysicsDiagnostics) -> None:
        """Capture one post-RK microphysics result for the next surface call."""
        if not isinstance(result, MicrophysicsDiagnostics):
            raise TypeError("microphysics must return MicrophysicsDiagnostics")
        shape = self.state.mup.shape

        slots = dict(microphysics_scratch_slots(self.mp_physics))
        labels = _MICROPHYSICS_DIAGNOSTIC_LABELS
        if slots:
            required = {"rainnc", "rainncv", "sr"}
            if self.mp_physics == 18:
                # NSSL owns every frozen-category accumulator/increment in
                # its named contract.  Missing one would silently retain a
                # stale canonical scratch field across physics calls.
                required.update(slots)
            native = (
                self.mp_physics != 18
                and _native_microphysics_diagnostics(
                    result, self.microphysics, slots, shape))
            if native:
                validated = {
                    name: getattr(result, name) for name in slots}
                invalid, below, above = _validate_native_microphysics(
                    result, slots, self._sr_roundoff_upper,
                    self.state.scratch(
                        (1,), "physics_validation_status").view(cp.uint32),
                    labels)
                if invalid is not None:
                    raise FloatingPointError(
                        f"microphysics {labels[invalid]} contains a "
                        "non-finite value")
            else:
                validated = {}
                for name in labels:
                    value = getattr(result, name)
                    if name not in slots:
                        if value is not None:
                            raise ValueError(
                                f"microphysics {labels[name]} is not produced "
                                f"by mp_physics={self.mp_physics}")
                        continue
                    if value is None:
                        if name in required:
                            raise ValueError(
                                f"microphysics {labels[name]} may not be None")
                        # Existing WSM6/Morrison attachments may omit their
                        # optional frozen-category diagnostics; the canonical
                        # zero-filled slot remains for those schemes.
                        continue
                    target = getattr(self.microphysics, name)
                    if self.mp_physics == 18:
                        validated[name] = _trusted_canonical_array(
                            value, target, shape,
                            f"microphysics {labels[name]}")
                    else:
                        validated[name] = _validated_array(
                            value, shape, f"microphysics {labels[name]}")
            sr = validated["sr"]
            if self.mp_physics != 18:
                if not native:
                    below = bool(cp.any(sr < DTYPE(0.0)))
                    above = bool(
                        cp.any(sr > DTYPE(self._sr_roundoff_upper)))
                if below or above:
                    sr_min = float(cp.min(sr).item())
                    sr_max = float(cp.max(sr).item())
                    flat = int(
                        (cp.argmin(sr) if below else cp.argmax(sr)).item())
                    index = tuple(
                        int(value) for value in np.unravel_index(flat, sr.shape))
                    raise ValueError(
                        "microphysics SR is outside its validated range: "
                        f"min={sr_min:.9g}, max={sr_max:.9g}, index={index}, "
                        f"allowed=[0, {float(self._sr_roundoff_upper):.9g}], "
                        f"roundoff_ulps={self._sr_roundoff_max_ulps}, "
                        f"shape={sr.shape}, accepted_updates="
                        f"{self.microphysics_updates}")
            for name, array in validated.items():
                target = getattr(self.microphysics, name)
                if array is not target:
                    target[...] = array
            accepted = self.microphysics
        else:
            def optional(value, name):
                return (None if value is None else
                        _checked_array(value, shape, f"microphysics {name}"))

            # No configured scheme can later mutate a scratch accumulator;
            # keep the legacy attachment contract for direct callers.
            accepted = MicrophysicsDiagnostics(
                rainnc=_checked_array(result.rainnc, shape,
                                      "microphysics RAINNC"),
                rainncv=_checked_array(result.rainncv, shape,
                                       "microphysics RAINNCV"),
                sr=_checked_array(result.sr, shape, "microphysics SR"),
                snownc=optional(result.snownc, "SNOWNC"),
                snowncv=optional(result.snowncv, "SNOWNCV"),
                graupelnc=optional(result.graupelnc, "GRAUPELNC"),
                graupelncv=optional(result.graupelncv, "GRAUPELNCV"),
                hailnc=optional(result.hailnc, "HAILNC"),
                hailncv=optional(result.hailncv, "HAILNCV"))
            self.microphysics = accepted
        if not slots and (bool(cp.any(accepted.sr < DTYPE(0.0))) or bool(
                cp.any(accepted.sr > DTYPE(1.0)))):
            raise ValueError("microphysics SR must be within [0, 1]")
        self._pending_rainbl += cp.maximum(
            accepted.rainncv, DTYPE(0.0))
        if self.ruc_params is not None or self.noahmp_params is not None:
            f = self.fields
            f["surface_rainncv"] += cp.maximum(
                accepted.rainncv, DTYPE(0.0))
            for diagnostic, carrier in (
                    (accepted.snowncv, "surface_snowncv"),
                    (accepted.graupelncv, "surface_graupelncv"),
                    (accepted.hailncv, "surface_hailncv")):
                if diagnostic is not None:
                    f[carrier] += cp.maximum(diagnostic, DTYPE(0.0))
        self.microphysics_updates += 1

    def attach_surface_moisture_ledger(self, ledger) -> None:
        """Attach a ledger and seed it with the carriers' INITIAL provenance.

        Seeding matters more than it looks.  A run whose longwave is a
        declared constant never calls ``_stamp_carriers``, so without this
        the ledger would show the carrier as never-written and a reader
        could not tell "no longwave scheme, value declared at t=0" from
        "longwave scheme attached but somehow never ran".  Those are
        different defects and the ledger has to separate them.
        """
        ledger.note_carrier(
            "glw", source=f"initial:{self.glw_provenance}", model_time=0.0,
            value=float(self.fields["glw"].reshape(-1)[0]))
        for name in ("swdown", "gsw", "coszen"):
            array = self.fields.get(name)
            if array is None:
                continue
            ledger.note_carrier(
                name, source="initial:allocated", model_time=0.0,
                value=float(array.reshape(-1)[0]))
        self.surface_moisture_ledger = ledger

    def _stamp_carriers(self, state: DomainState, cfg: RunConfig) -> None:
        """Tell an attached ledger which carriers this call just rewrote.

        The SOURCE recorded is the scheme pair that produced the value, not
        the buffer it landed in, because "GLW is 300" and "GLW is 300
        because nothing computes it" are the two cases the ledger exists to
        tell apart.  A carrier absent from ``self.fields`` is absent from
        the record rather than recorded as zero.
        """
        ledger = self.surface_moisture_ledger
        now = float(state.elapsed_seconds)
        source = (f"ra_lw_physics={int(cfg.ra_lw_physics)},"
                  f"ra_sw_physics={int(cfg.ra_sw_physics)}")
        for name in ("glw", "swdown", "gsw", "coszen"):
            array = self.fields.get(name)
            if array is None:
                continue
            ledger.note_carrier(
                name, source=source, model_time=now,
                value=float(array.reshape(-1)[0]))

    def _run_radiation(self, atmosphere: Mapping[str, cp.ndarray],
                       state: DomainState, cfg: RunConfig) -> None:
        """Invoke and capture one due radiation result."""
        result = self.radiation_callable(
            atmosphere=atmosphere, fields=self.fields, state=state, cfg=cfg)
        if not isinstance(result, RadiationResult):
            raise TypeError("radiation callable must return RadiationResult")
        nz, ny, nx = state.p.shape
        shape = (nz, ny, nx)
        self.rthratenlw = _checked_array(
            result.rthratenlw, shape, "radiation rthratenlw")
        self.rthratensw = _checked_array(
            result.rthratensw, shape, "radiation rthratensw")
        swdown = _checked_array(
            result.swdown, (ny, nx), "radiation SWDOWN")
        glw = _checked_array(result.glw, (ny, nx), "radiation GLW")
        self.radiation_tendencies = couple_column_tendencies(
            state, cfg, rtheta=self.rthratenlw + self.rthratensw)
        self.fields["swdown"][...] = swdown
        self.fields["glw"][...] = glw
        if "gsw" in self.fields:
            if result.gsw is None:
                raise ValueError(
                    "RUC requires radiation-time GSW from the radiation "
                    "callable")
            self.fields["gsw"][...] = _checked_array(
                result.gsw, (ny, nx), "radiation GSW")
        if "coszen" in self.fields:
            if result.coszen is None:
                raise ValueError(
                    "Noah-MP requires carried COSZEN from the radiation "
                    "callable")
            self.fields["coszen"][...] = _checked_array(
                result.coszen, (ny, nx), "radiation COSZEN")
        # CARRIER CONTRACT.  Stamped HERE rather than at the consumer
        # because this is the one place that knows a carrier was actually
        # RECOMPUTED rather than re-read.  The consumer's freshness
        # question -- "how old is the longwave this surface call is about
        # to eat" -- is answerable only against this timestamp, and the
        # answer has to come from the producer's own call site or it is
        # just a restatement of the buffer's existence.
        now = float(state.elapsed_seconds)
        if self.carriers is None:
            raise CarrierContractError(
                "the radiation seam has no carrier contract to stamp; a "
                "PhysicsDriver assembled outside initialize_physics must "
                "carry one, because provenance recorded nowhere is "
                "provenance lost")
        for name in ("glw", "swdown", "gsw", "coszen"):
            if name not in self.fields:
                continue
            # A DECLARED longwave SURVIVES the radiation call: with
            # ra_lw_physics=0 the shortwave-only adapters return the GLW
            # array they were handed (gpuwm/core/dudhia.py), so the buffer
            # is unchanged and relabelling it "produced by a radiation
            # scheme" would be the exact mislabelling this contract exists
            # to stop.  Its declared source and its timelessness both hold.
            if (name == "glw" and int(cfg.ra_lw_physics) == 0
                    and self.carriers.source("glw") in (
                        CARRIER_SOURCE_DECLARED_CONSTANT,
                        CARRIER_SOURCE_EXTERNAL_ARRAY)):
                continue
            self.carriers.declare(
                name, source=CARRIER_SOURCE_RADIATION_SCHEME,
                model_time=now)
        # CARRIER PROVENANCE, ledger half.  Stamped only when a ledger is
        # attached; the contract above is unconditional.
        if self.surface_moisture_ledger is not None:
            self._stamp_carriers(state, cfg)
        if self.olr is not None:
            # Fail closed rather than republish the previous call's flux:
            # the buffer only exists because the attached scheme declared
            # it computes one, so a missing OLR is a broken scheme, not a
            # configuration in which OLR is undefined.
            if result.olr is None:
                raise ValueError(
                    "the attached radiation callable declares publishes_olr "
                    "but returned no OLR (TOA outgoing longwave)")
            self.olr[...] = _checked_array(
                result.olr, (ny, nx), "radiation OLR")

    def _run_cumulus(self, atmosphere: Mapping[str, cp.ndarray],
                      state: DomainState, cfg: RunConfig) -> None:
        """Invoke one due cumulus call and capture or hold-merge its result.

        Results carrying ``nca_seconds`` follow WRF's KF driver
        persistence: only columns whose hold expired accept the new
        rates, PRATEC, RAINCV, and NCA -- ``KF_eta_CPS`` skips
        recomputation while ``NCA(I,J) .ge. 0.5*DT``
        (module_cu_kfeta.F:410-412) and zeroes the eligible columns'
        tendencies/RAINCV/PRATEC before each column call (414-440), so
        the batched kernel's outputs for eligible non-triggering columns
        are already zero and discarding the held columns' outputs here
        is the same contract as skipping them.  Results without
        ``nca_seconds`` keep the Task-1 attachment contract.
        """
        bind = getattr(self.cumulus_callable, "bind_driver", None)
        if bind is not None:
            # Optional adapter hook (the update_trigger_history idiom): GF
            # reads the held radiative rates through it.  Idempotent.
            bind(self)
        result = self.cumulus_callable(
            atmosphere=atmosphere, fields=self.fields, state=state, cfg=cfg)
        if not isinstance(result, CumulusResult):
            raise TypeError("cumulus callable must return CumulusResult")
        nz, ny, nx = state.p.shape
        shape = (nz, ny, nx)
        native_kf = _is_native_kf_result(
            result, self.cumulus_callable, state, cfg, shape)
        if native_kf:
            _validate_native_kf_result(result, state)
            # Exact production provenance makes these temporary launch
            # outputs safe read-only sources.  The NCA branch below copies
            # every value into persistent driver storage on this stream
            # before the transient result can leave scope.
            rtheta = result.rthcuten
            rqv = result.rqvcuten
            rqc = result.rqccuten
            rqi = result.rqicuten
            rqr = result.rqrcuten
            rqs = result.rqscuten
        else:
            rtheta = _checked_array(
                result.rthcuten, shape, "cumulus rthcuten")
            rqv = _checked_array(result.rqvcuten, shape, "cumulus rqvcuten")
            rqc = (None if result.rqccuten is None else _checked_array(
                result.rqccuten, shape, "cumulus rqccuten"))
            rqi = (None if result.rqicuten is None else _checked_array(
                result.rqicuten, shape, "cumulus rqicuten"))
            rqr = (None if result.rqrcuten is None else _checked_array(
                result.rqrcuten, shape, "cumulus rqrcuten"))
            rqs = (None if result.rqscuten is None else _checked_array(
                result.rqscuten, shape, "cumulus rqscuten"))
        unsupported = []
        if getattr(state, "qi", None) is None and rqi is not None:
            unsupported.append("rqicuten")
        if getattr(state, "qs", None) is None and rqs is not None:
            unsupported.append("rqscuten")
        if unsupported:
            names = ", ".join(unsupported)
            raise ValueError(
                f"custom cumulus result supplies {names} without matching "
                "state prognostics; supply phase/latent-energy-consistent "
                "folded tendencies and omit unsupported frozen categories")
        if result.nca_seconds is None:
            if result.pratec is not None:
                raise ValueError("cumulus pratec requires nca_seconds")
            news = {
                "rthcuten": rtheta, "rqvcuten": rqv,
                "rqccuten": (cp.zeros(shape, dtype=DTYPE) if rqc is None
                             else rqc),
                "rqicuten": (cp.zeros(shape, dtype=DTYPE) if rqi is None
                             else rqi),
                "rqrcuten": (cp.zeros(shape, dtype=DTYPE) if rqr is None
                             else rqr),
                "rqscuten": (cp.zeros(shape, dtype=DTYPE) if rqs is None
                             else rqs),
            }
            for name, new in news.items():
                self.cu_rates[name][...] = new
            self.cumulus_tendencies = couple_column_tendencies(
                state, cfg, rtheta=rtheta, rqv=rqv, rqc=rqc, rqr=rqr,
                rqi=rqi, rqs=rqs)
            self.cumulus_tendencies.materialize(
                _cumulus_optional_tendency_components(cfg))
            if result.rainc is not None:
                increment = _checked_array(
                    result.rainc, (ny, nx), "cumulus RAINC increment")
                self.rainc += increment
                # Convective rain also wets the surface (RAINBL, WRF
                # module_surface_driver.F:1566) on the legacy contract.
                self._pending_rainbl += cp.maximum(increment, DTYPE(0.0))
                if self.ruc_params is not None or self.noahmp_params is not None:
                    self.fields["surface_raincv"] += cp.maximum(
                        increment, DTYPE(0.0))
            return
        if result.pratec is None:
            raise ValueError(
                "cumulus results with nca_seconds must include pratec")
        if native_kf:
            nca_new = result.nca_seconds
            pratec_new = result.pratec
        else:
            nca_new = _checked_array(
                result.nca_seconds, (ny, nx), "cumulus nca_seconds")
            pratec_new = _checked_array(
                result.pratec, (ny, nx), "cumulus PRATEC")
        # DT in every KF driver formula is WRF's model-clock step, not the
        # internal integration substep (Task 6b audit).
        clock_dt = _model_clock_dt(cfg)
        eligible = self.cu_nca < DTYPE(0.5) * DTYPE(clock_dt)
        news = {
            "rthcuten": rtheta, "rqvcuten": rqv,
            "rqccuten": (cp.zeros(shape, dtype=DTYPE) if rqc is None
                         else rqc),
            "rqicuten": (cp.zeros(shape, dtype=DTYPE) if rqi is None
                         else rqi),
            "rqrcuten": (cp.zeros(shape, dtype=DTYPE) if rqr is None
                         else rqr),
            "rqscuten": (cp.zeros(shape, dtype=DTYPE) if rqs is None
                         else rqs),
        }
        for name, new in news.items():
            cp.copyto(self.cu_rates[name], new, where=eligible[None])
        cp.copyto(self.cu_pratec, pratec_new, where=eligible)
        # RAINCV = DT*PRATEC (module_cu_kfeta.F:2504-2505, 2642-2643).
        cp.copyto(self.cu_raincv, DTYPE(clock_dt) * pratec_new,
                  where=eligible)
        # WRF assigns NCA only when a column triggers (2570, 2573); a
        # stale NCA on an eligible column is <= 0 by construction and the
        # scheme reports 0 for non-triggering columns, so the flat
        # eligible-column assignment is identical under both driver
        # predicates (>= 0.5*DT and > 0).
        cp.copyto(self.cu_nca, nca_new, where=eligible)
        cumulus_components = _cumulus_optional_tendency_components(cfg)
        self.cumulus_tendencies = couple_column_tendencies(
            state, cfg, rtheta=self.cu_rates["rthcuten"],
            rqv=self.cu_rates["rqvcuten"], rqc=self.cu_rates["rqccuten"],
            rqr=(self.cu_rates["rqrcuten"]
                 if "rqr" in cumulus_components else None),
            rqi=(self.cu_rates["rqicuten"]
                 if "rqi" in cumulus_components else None),
            rqs=(self.cu_rates["rqscuten"]
                 if "rqs" in cumulus_components else None))
        self.cumulus_tendencies.materialize(cumulus_components)

    def _advance_cumulus_clock(self, state: DomainState,
                               cfg: RunConfig) -> None:
        """Per-model-step WRF ``advance_ppt`` for the held KF cumulus state.

        Transcribed from solve_em.F:3558-3571 (one call per model-CLOCK
        step, after the physics tendencies are in play and before the later
        microphysics_driver call at :3689-3720) and
        module_physics_addtendc.F:2139-2145 plus its KFETASCHEME case at
        2196-2231: RAINC accumulates ``PRATEC*DT`` every step; a positive
        NCA has the stored tendencies zeroed once ``NINT(NCA/DT) <= 1``
        and is then decremented by DT.  WRF leaves RAINCV/PRATEC alone at
        expiry (the zeroing at 2216-2217 is commented out), so the rate
        keeps accumulating until the column's next scheme call replaces
        it.  Under the compatibility substep integration
        (``cfg.clock_dt > cfg.dt``) the advance runs once per clock step,
        on its final internal substep, with DT = clock_dt; because it runs
        after ``_compose_tendencies``, expiry leaves this step's already
        copied/mass-coupled RK target intact while clearing the persistent
        raw rates before post-RK microphysics.  Thus Morrison does not seed
        number moments from an expired KF rate.  Legacy results never set
        NCA above zero, so this is a no-op for them.
        """
        clock_dt = _model_clock_dt(cfg)
        end = float(state.elapsed_seconds) + float(cfg.dt)
        remainder = abs(math.fmod(end, clock_dt))
        if min(remainder, clock_dt - remainder) > 1.0e-6 * clock_dt:
            return
        dt = DTYPE(clock_dt)
        self.rainc += self.cu_pratec * dt
        # WRF's surface driver adds convective rain to RAINBL every step
        # (module_surface_driver.F:1566, RAINBL += RAINCV + RAINNCV with
        # RAINCV = PRATEC*DT); the microphysics half arrives through
        # accept_microphysics.  Final-review MAJOR: Noah previously never
        # saw KF rain at all.
        self._pending_rainbl += cp.maximum(self.cu_pratec, DTYPE(0.0)) * dt
        if self.ruc_params is not None or self.noahmp_params is not None:
            self.fields["surface_raincv"] += cp.maximum(
                self.cu_pratec, DTYPE(0.0)) * dt
        active = self.cu_nca > DTYPE(0.0)
        # Fortran NINT rounds half away from zero; NCA > 0 in this branch,
        # so floor(x + 0.5) is exact NINT.
        expiring = active & (cp.floor(self.cu_nca / dt + DTYPE(0.5))
                             <= DTYPE(1.0))
        # Set before publishing the device mask so an exception during any
        # subsequent enqueue leaves compute() able to retry finalization.
        self._cu_expiry_pending = True
        self.cu_expiring[...] = expiring
        self.cu_nca[...] = cp.where(active, self.cu_nca - dt, self.cu_nca)
        self.finish_step()

    def finish_step(self) -> None:
        """Finalize a KF expiry after the current RK target was composed.

        WRF's ``advance_ppt`` clears expired ``Q*CUTEN`` before the later
        microphysics driver, so Morrison must see zero raw rain/snow/ice
        rates on the expiry step.  gpuwm advances the timer during its pre-RK
        driver call, after copying the current forcing into ``tendencies``;
        clearing these persistent copies therefore preserves current-step RK
        forcing and changes only Morrison plus the next compose.  No
        volume-rate copy is allocated.

        THE EARLY RETURN IS GONE, ON PURPOSE.  It used to read
        ``bool(self.cu_expiring.any())`` -- a host branch over device data --
        and skip the clears when nothing was expiring.  Two things were wrong
        with that and only one of them was the ~94 us the readback drains:
        the branch made the STEP'S LAUNCH SEQUENCE depend on the values in
        the arrays, so the step could not be captured as a CUDA graph and,
        worse, no host-computable cache key could ever have been correct for
        it.  The clears now run unconditionally through
        :func:`gpuwm.core.health_ledger.masked_clear`, which writes exactly
        what ``cp.where(mask, 0, array)`` wrote -- a select-and-store with no
        arithmetic -- and, on the overwhelmingly common no-expiry step, only
        reads the (ny, nx) mask and writes nothing.  Bit-exact, cheaper than
        the ``cp.where`` it replaces even when it does fire (no full-size
        temporary), and constant-topology.
        """
        if self.cu_expiring is None:
            self._cu_expiry_pending = False
            return
        mask = self.cu_expiring[None] != DTYPE(0.0)
        for array in self.cu_rates.values():
            health_ledger.masked_clear(mask, array)
        # These are the held, already mass-coupled copies used on the NEXT
        # compose.  The separately composed current-step target is untouched.
        for name in ("rtheta", "rqv", "rqc", "rqr", "rqi", "rqs"):
            array = getattr(self.cumulus_tendencies, name)
            if array is not None:
                health_ledger.masked_clear(mask, array)
        self.cu_expiring[...] = DTYPE(0.0)
        self._cu_expiry_pending = False

    def recouple_cumulus_tendencies(self, state: DomainState,
                                    cfg: RunConfig) -> None:
        """Rebuild the held coupled cumulus tendencies from the raw rates.

        The nest-relocation preparer restores the raw per-column rates
        (``cu_rates``, aliases of the serialized ``cu_*`` slots) onto a
        rebuilt driver; the COUPLED copies the RK composer reads were
        computed by the OUTGOING driver against its own dry-air mass and
        do not survive the rebuild.  Recoupling against the post-
        transplant child reproduces them: on the strip the rates are
        zero and the coupling is inert, and on the overlap the rates are
        the outgoing child's, coupled with the transplanted (current)
        mass rather than the historical due-call mass -- the same
        approximation WRF's own advance would make one step later, and
        stated here rather than silently absorbed.  Without this call a
        moved nest applies ZERO convective heating on held columns until
        the next due cumulus call, which is up to a full ``cudt``.
        """
        if self.cu_rates is None:
            return
        cumulus_components = _cumulus_optional_tendency_components(cfg)
        self.cumulus_tendencies = couple_column_tendencies(
            state, cfg, rtheta=self.cu_rates["rthcuten"],
            rqv=self.cu_rates["rqvcuten"], rqc=self.cu_rates["rqccuten"],
            rqr=(self.cu_rates["rqrcuten"]
                 if "rqr" in cumulus_components else None),
            rqi=(self.cu_rates["rqicuten"]
                 if "rqi" in cumulus_components else None),
            rqs=(self.cu_rates["rqscuten"]
                 if "rqs" in cumulus_components else None))
        self.cumulus_tendencies.materialize(cumulus_components)

    def _compose_tendencies(self, cfg: RunConfig) -> None:
        """Compose held PBL/radiation/cumulus components in WRF order."""
        if not (self.radiation_active or self.cu_physics):
            # This identity path keeps Phase-3 YSU-only runs bit-compatible.
            self.tendencies = self.pbl_tendencies
            return
        pbl = self.pbl_tendencies
        if physics_reuses_pbl_composition(cfg):
            # bldt=0 mechanically replaces pbl immediately before every call
            # to this composer.  Reusing that fresh stack is safe exactly
            # once; positive cadence keeps the separate historical target.
            self.tendencies = pbl
        target = self.tendencies
        target.ru[...] = pbl.ru
        target.rv[...] = pbl.rv
        for name in ("rtheta", "rqv", "rqc"):
            value = getattr(target, name)
            value[...] = getattr(pbl, name)
            value += getattr(self.radiation_tendencies, name)
            value += getattr(self.cumulus_tendencies, name)
        for name in ("rqr", "rqi", "rqs"):
            sources = [getattr(component, name) for component in (
                pbl, self.radiation_tendencies, self.cumulus_tendencies)
                if getattr(component, name) is not None]
            if sources:
                value = getattr(target, name)
                if value is None:
                    value = cp.zeros_like(target.rqc)
                    setattr(target, name, value)
                value[...] = sources[0]
                for source in sources[1:]:
                    value += source
            else:
                setattr(target, name, None)

    def _run_sfclay(self, atmosphere: Mapping[str, cp.ndarray],
                    cfg: RunConfig) -> None:
        f = self.fields
        option = int(cfg.sf_sfclay_physics)
        if option not in (1, 5, 91):
            raise UnroutedPhysicsSelectorError(
                "sf_sfclay_physics", option, (1, 5, 91))
        if option == 5:
            if self.mynn_sfclay_result is None:
                raise RuntimeError("MYNN surface result was not initialized")
            if atmosphere["u"].shape[0] < 2:
                raise ValueError("MYNN surface layer requires at least 2 levels")
            itimestep = int(np.floor(
                float(self.state.elapsed_seconds) / cfg.dt + 0.5)) + 1
            ice_component = (
                (f["xice"] >= DTYPE(0.5)) & (f["xice"] <= DTYPE(1.0))
                if self.ruc_params is not None else
                cp.zeros_like(f["xice"], dtype=cp.bool_)
            )
            # RUC's admitted identity has FRACTIONAL_SEAICE enabled.  WRF
            # enters MYNN_SEAICE_WRAPPER unconditionally under that switch,
            # even when this particular slab has no active ice cells.
            fractional_ruc = self.mynn_sfclay_sea_result is not None
            tsk_local = f["tsk"]
            if fractional_ruc:
                # module_surface_driver.F:5402-5421, 5465-5482: the second
                # MYNN call starts from the exact inout fields that existed
                # before the ice call.  The ``*_sea`` arrays are therefore
                # hold buffers first and output staging second.
                for name in MYNN_SURFACE_OUTPUTS:
                    f[f"{name}_sea"][...] = f[name]
                # get_local_ice_tsk :7158-7204.  Under the pinned
                # tice2tsk_if2cold=.false. identity, MYNN's first call sees
                # the ice component diagnosed from grid-cell TSK and SST.
                # RUC later sees TSK_SAVE instead; those are deliberately
                # different WRF ownership paths, especially on step one.
                sst = cp.maximum(f["tsk_sea"], DTYPE(271.4))
                if itimestep <= 3:
                    warm = sst > DTYPE(273.0)
                    sst = cp.where(
                        warm & (f["xice"] >= DTYPE(0.6)),
                        DTYPE(271.4),
                        cp.where(
                            warm & (f["xice"] >= DTYPE(0.4)),
                            DTYPE(273.0),
                            cp.where(
                                warm & (f["xice"] >= DTYPE(0.2))
                                & (sst > DTYPE(275.0)),
                                DTYPE(275.0),
                                cp.where(
                                    warm & (sst > DTYPE(278.0)),
                                    DTYPE(278.0), sst))))
                f["tsk_sea"][...] = cp.where(
                    ice_component, sst, f["tsk_sea"])
                denominator = cp.where(
                    ice_component, f["xice"], DTYPE(1.0))
                diagnosed_ice = cp.maximum(
                    (f["tsk"]
                     - (DTYPE(1.0) - f["xice"]) * f["tsk_sea"])
                    / denominator,
                    DTYPE(221.4))
                tsk_local = cp.where(
                    ice_component, diagnosed_ice, f["tsk"]).astype(DTYPE)
            if itimestep == 1:
                # module_sf_mynn.F:329-337.  SFCLAY_mynn seeds UST/MOL/QSFC
                # and qstar from the lowest model level before the column
                # solver runs; without it MYNN's first step starts from the
                # module_physics_init.F cold-start UST=1e-4 and QSFC=0.
                seed_mynn_surface_first_step(
                    atmosphere["u"][0], atmosphere["v"][0],
                    atmosphere["qv"][0],
                    ust=f["ust"], mol=f["mol"], qsfc=f["qsfc"],
                    qstar=f["qstar"],
                )
                if fractional_ruc:
                    # MYNN_SEAICE_WRAPPER calls SFCLAY_mynn a second time;
                    # its one-based first-step block therefore runs on the
                    # open-water inouts too.
                    seed_mynn_surface_first_step(
                        atmosphere["u"][0], atmosphere["v"][0],
                        atmosphere["qv"][0],
                        ust=f["ust_sea"], mol=f["mol_sea"],
                        qsfc=f["qsfc_sea"], qstar=f["qstar_sea"],
                    )
            inputs = {
                "u1": atmosphere["u"][0],
                "v1": atmosphere["v"][0],
                "t1": atmosphere["temperature"][0],
                "qv1": atmosphere["qv"][0],
                "p1": atmosphere["pressure"][0],
                "rho1": atmosphere["rho"][0],
                "dz1": atmosphere["dz"][0],
                "u2": atmosphere["u"][1],
                "v2": atmosphere["v"][1],
                "dz2": atmosphere["dz"][1],
                "psfc": atmosphere["p_interface"][0],
                # get_local_ice_tsk supplies this diagnosed ice component to
                # MYNN.  RUC separately consumes TSK_SAVE and rebuilds the
                # blended TSK after its call.
                "tsk": tsk_local,
                "pblh": f["pblh"],
                "mavail": f["mavail"],
                "hfx": f["hfx"],
                "qfx": f["qfx"],
                "znt": f["znt"],
                "qsfc": f["qsfc"],
                "ust": f["ust"],
                "xland": f["xland"],
                "snowh": f["snowh"],
            }
            launch_mynn_surface_layer(
                inputs, f["mol"], f["ustm"], self.mynn_sfclay_result,
                dx=cfg.dx,
                itimestep=itimestep,
                isfflx=cfg.isfflx,
            )
            if fractional_ruc:
                # module_surface_driver.F:5441-5506.  Force the second call
                # to open water only on fractional-ice cells and preserve the
                # original grid values elsewhere.
                f["tsk_sea"][...] = cp.where(
                    ice_component,
                    cp.maximum(f["tsk_sea"], DTYPE(271.4)),
                    f["tsk_sea"])
                f["znt_sea"][...] = cp.where(
                    ice_component, DTYPE(1.0e-4), f["znt_sea"])
                sea_inputs = {
                    **inputs,
                    "tsk": cp.where(
                        ice_component, f["tsk_sea"], inputs["tsk"]),
                    "pblh": f["pblh"],
                    "mavail": cp.where(
                        ice_component, DTYPE(1.0), f["mavail"]),
                    "hfx": f["hfx_sea"],
                    "qfx": f["qfx_sea"],
                    "znt": f["znt_sea"],
                    "qsfc": f["qsfc_sea"],
                    "ust": f["ust_sea"],
                    "xland": cp.where(
                        ice_component, DTYPE(2.0), f["xland"]),
                }
                # REGIME is the one shared actual argument in both WRF calls
                # (:5429, :5491), so the second call starts from the first
                # call's value rather than from the pre-wrapper hold set.
                f["regime_sea"][...] = f["regime"]
                launch_mynn_surface_layer(
                    sea_inputs, f["mol_sea"], f["ustm_sea"],
                    self.mynn_sfclay_sea_result,
                    dx=cfg.dx, itimestep=itimestep, isfflx=cfg.isfflx,
                )

                # :5508-5554.  These diagnostics become grid-cell values
                # immediately.  The flux/exchange fields marked "wait" in
                # WRF remain as ice plus staged sea components until RUC's
                # post-LSM reblend at :3543-3562.
                one_minus_ice = DTYPE(1.0) - f["xice"]
                for name in (
                        "br", "gz1oz0", "mol", "psih", "psim", "rmol",
                        "ust", "wspd", "zol", "ch", "cd", "cda", "ck",
                        "cka", "q2", "t2", "th2", "u10", "ustm", "v10"):
                    blended = (
                        f[name] * f["xice"]
                        + one_minus_ice * f[f"{name}_sea"]
                    ).astype(DTYPE)
                    f[name][...] = cp.where(
                        ice_component, blended, f[name])
                # The open-water call's REGIME value is the last writer.
                f["regime"][...] = cp.where(
                    ice_component, f["regime_sea"], f["regime"])
            return
        launch_sfclay(
            atmosphere["u"][0], atmosphere["v"][0],
            atmosphere["temperature"][0], atmosphere["qv"][0],
            atmosphere["pressure"][0], atmosphere["dz"][0],
            atmosphere["p_interface"][0],
            (cp.where(
                (f["xice"] >= DTYPE(0.5)) & (f["xice"] <= DTYPE(1.0)),
                f["tsk_save"], f["tsk"]).astype(DTYPE)
             if self.ruc_params is not None else f["tsk"]),
            f["pblh"], f["mavail"], f["xland"], f["lakemask"],
            self.sfclay_result, option=option, dx=cfg.dx,
            isfflx=bool(cfg.isfflx),
            isftcflx=cfg.isftcflx, iz0tlnd=cfg.iz0tlnd)
        if cfg.km_opt in (2, 3, 4) and cfg.bl_pbl_physics == 0:
            # module_sf_sfclay.F:799-803 and the corresponding revised-MM5
            # path update UST and USTM from the same PSIX, but USTM uses the
            # wind magnitude without the convective-velocity correction.
            wspdi = cp.sqrt(
                atmosphere["u"][0] * atmosphere["u"][0]
                + atmosphere["v"][0] * atmosphere["v"][0])
            f["ustm"][...] = (
                DTYPE(0.5) * f["ustm"]
                + DTYPE(0.5) * DTYPE(0.4) * wspdi / f["fm"]
            ).astype(DTYPE)
        if self.ruc_params is not None:
            ice_component = ((f["xice"] >= DTYPE(0.5))
                             & (f["xice"] <= DTYPE(1.0)))
            if bool(cp.any(ice_component)):
                # WRF's SFCLAY*_SEAICE_WRAPPER runs the same surface layer
                # over the open-water component and retains these values for
                # module_surface_driver.F:3543-3562's post-LSM blend.
                sea = sfclay(
                    atmosphere["u"][0], atmosphere["v"][0],
                    atmosphere["temperature"][0], atmosphere["qv"][0],
                    atmosphere["pressure"][0], atmosphere["dz"][0],
                    atmosphere["p_interface"][0], f["tsk_sea"],
                    f["znt_sea"], f["pblh"],
                    cp.ones_like(f["mavail"]), cp.full_like(f["xland"], 2.0),
                    qsfc=f["qsfc_sea"], zol=f["zol_sea"],
                    ust=f["ust_sea"], mol=f["mol_sea"],
                    hfx=f["hfx_sea"], qfx=f["qfx_sea"],
                    lakemask=cp.zeros_like(f["lakemask"]),
                    option=option, dx=cfg.dx,
                    isfflx=bool(cfg.isfflx),
                    isftcflx=cfg.isftcflx, iz0tlnd=cfg.iz0tlnd)
                for name in (
                        "znt", "ust", "mol", "zol", "flhc", "flqc", "cpm",
                        "cqs2", "chs2", "chs", "qsfc", "qgh", "hfx", "qfx",
                    "lh"):
                    carrier = f"{name}_sea"
                    f[carrier][...] = cp.where(
                        ice_component, getattr(sea, name), f[carrier])

    def _run_myj_sfclay(self, atmosphere: Mapping[str, cp.ndarray],
                        cfg: RunConfig) -> None:
        """Run WRF's Eta similarity surface layer (sf_sfclay_physics=2).

        The call cadence and the LSM contract are the MM5 layers': the
        exchange coefficients Noah reads (CHS/CHS2/CQS2/FLHC/FLQC/QGH/CPM
        and RIB) land in the same ``fields`` slots ``_run_sfclay`` fills,
        because that contract is a property of gpuwm's surface seam rather
        than of a scheme.  What is MYJ's -- the viscous sublayer over
        water, the Zilitinkevich thermal roughness over land, the five-pass
        flux iteration and the AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 set the MYJ PBL
        consumes -- lives in :mod:`gpuwm.core.myjsfc` next to its source
        anchors.

        Bindings with their source of truth:

        * ``RIB`` is written to ``fields["br"]``.  That is WRF's own
          binding, not an analogy: the surface driver passes ``br`` as
          MYJSFC's ``RIB`` actual argument
          (module_surface_driver.F:2199,:2217), and ``_run_noah`` copies
          ``br`` into ``rib`` for Noah.
        * ``USTM``/``WSPD``/``CH`` are filled from the driver's own
          post-call block (:2226-2231): ``ustm = ust``, ``wspd =
          max(sqrt(u1^2+v1^2), 0.001)``, ``ch = chs``.
        * ``Z0BASE`` is the background roughness the Zilitinkevich fix
          reads (module_sf_myjsfc.F:733).  It is a static field seeded from
          the cold-start ZNT in :func:`initialize_physics`, never the
          working ZNT the sea branch overwrites every iteration.
        * ``PBLH`` is MYJSFC's own TKE scan (:263-277), not MIXLEN's
          (module_bl_myjpbl.F:834).  The two use different interfaces on
          purpose and WRF lets the surface layer's value stand until the
          PBL call replaces it.
        * ``ITIMESTEP`` is one-based, matching WRF's NTSD: the scheme's
          first-step branches seed USTAR, QSFC, THZ0 and QZ0.
        """
        f = self.fields
        itimestep = int(np.floor(
            float(self.state.elapsed_seconds) / cfg.dt + 0.5)) + 1
        columns = {"dz": atmosphere["dz"], "tke": f["tke_myj"]}
        surface = {
            "u1": atmosphere["u"][0], "v1": atmosphere["v"][0],
            "t1": atmosphere["temperature"][0],
            "th1": atmosphere["theta"][0],
            "qv1": atmosphere["qv"][0], "qc1": atmosphere["qc"][0],
            "p1": atmosphere["pressure"][0],
            "psfc": atmosphere["p_interface"][0],
            "tsk": f["tsk"], "xland": f["xland"], "mavail": f["mavail"],
            "z0base": f["z0base"],
        }
        state = {name: f[name] for name in MYJ_SFCLAY_INOUT}
        outputs = {"rib": f["br"]}
        outputs.update({name: f[name] for name in MYJ_SFCLAY_OUTPUTS
                        if name != "rib"})
        launch_myj_sfclay(columns, surface, state, outputs,
                          itimestep=itimestep)
        # module_surface_driver.F:2226-2231, the driver's own post-call
        # block.  Without it USTM/WSPD/CH keep the previous step's values
        # and the LES TKE path reads a stale friction velocity.
        f["ustm"][...] = f["ust"]
        wspd = cp.sqrt(atmosphere["u"][0] * atmosphere["u"][0]
                       + atmosphere["v"][0] * atmosphere["v"][0])
        f["wspd"][...] = cp.maximum(wspd, DTYPE(0.001))
        f["ch"][...] = f["chs"]

    def _couple_pbl_slot(self, cfg: RunConfig,
                         rates: Mapping[str, cp.ndarray],
                         state: DomainState | None = None,
                         ) -> PhysicsTendencies:
        """Mass-couple a PBL slot's raw rates AND retain GF's forcing lanes.

        Every PBL scheme in this driver ends its due call by handing the
        SAME raw A-grid rate set to :func:`couple_ysu_tendencies` -- that
        mapping's ``dtheta``/``dqv`` are WRF's RTHBLTEN/RQVBLTEN
        (K s-1 dry theta, kg kg-1 s-1), whichever scheme filled them.
        WRF's cumulus driver hands those two arrays to GFDRV regardless of
        ``bl_pbl_physics``; the arrays persist between PBL calls, so
        retaining the last computed rates IS the WRF contract.

        THE BREAKAGE THIS SEAM PREVENTS: retention used to live in
        ``_run_ysu`` alone, so a run pairing Grell-Freitas with MYJ, MYNN,
        Shin-Hong or SASE fed the cumulus scheme ZEROS for its
        boundary-layer forcing while a YSU run fed it the real rates --
        the same defect the YSU retention closed, left open for four
        schemes out of five.  Coupling and retention are one call here so
        that a PBL scheme added later cannot be wired up correctly and
        still silently starve the cumulus scheme.
        """
        # Raw, PRE-coupling: couple_ysu_tendencies multiplies into NEW
        # arrays (chm * rates["dtheta"]) and never writes back, so these
        # references stay the physical rates GFDRV wants.
        self.gf_rthblten = rates["dtheta"]
        self.gf_rqvblten = rates["dqv"]
        return couple_ysu_tendencies(
            self.state if state is None else state, cfg, rates)

    def _run_myj_pbl(self, atmosphere: Mapping[str, cp.ndarray],
                     cfg: RunConfig) -> None:
        """Run the MYJ PBL (bl_pbl_physics=2).

        MYJ is the one PBL in this driver that performs its own implicit
        vertical diffusion, so the launcher returns finished tendencies and
        the only thing left here is WRF's mass coupling and A-grid-to-C-grid
        interpolation -- which is YSU's seam because it is a property of
        the PBL slot, not of a scheme (the _run_mynn_pbl rationale).

        Scheme-specific bindings, each with its source of truth:

        * ``ELFLX`` is ``fields["lh"]``, exactly as the WRF PBL driver
          binds it (module_pbl_driver.F:1458 ``ELFLX=lh``).  It is the
          latent-heat flux the LAND branch inverts to rebuild QSFC
          (module_bl_myjpbl.F:554-557), which is why a MYJ run wants an
          LSM: with sf_surface_physics=0 nothing writes LH or CHKLOWQ and
          the surface humidity stops evolving.
        * ``TKE_MYJ``/``EL_MYJ`` are ``fields["tke_myj"]``/
          ``fields["el_myj"]``, the scheme's own carried state; the same
          arrays the surface layer's PBLH scan read earlier in this step.
        * ``QCI`` follows WRF's ``PRESENT(QCI)`` arm (:264-273): with ice
          in the moist set the species stack is four rows and RQIBLTEN is
          real, without it three and the row is a published zero.
        * ``AKHS``/``AKMS``/``THZ0``/``QZ0``/``UZ0``/``VZ0`` come from the
          Eta surface layer earlier in this same due step, which is the
          pairing gpuwm.config.validate_myj_pairing enforces.
        """
        f = self.fields
        flqi = atmosphere.get("qi") is not None
        columns = {
            "dz": atmosphere["dz"], "u": atmosphere["u"],
            "v": atmosphere["v"], "t": atmosphere["temperature"],
            "th": atmosphere["theta"], "exner": atmosphere["exner"],
            "qv": atmosphere["qv"], "qc": atmosphere["qc"],
            "qi": atmosphere.get("qi"), "p": atmosphere["pressure"],
        }
        surface = {
            "psfc": atmosphere["p_interface"][0],
            "ust": f["ust"], "tsk": f["tsk"], "chklowq": f["chklowq"],
            "xland": f["xland"], "sice": f["xice"], "snow": f["snow"],
            "akhs": f["akhs"], "akms": f["akms"], "elflx": f["lh"],
            "uz0": f["uz0"], "vz0": f["vz0"],
        }
        state = {name: f[name] for name in MYJ_PBL_INOUT}
        out = myj_pbl_step(columns, surface, state, f["tke_myj"],
                           dtturbl=self.bldt_seconds, flqi=flqi)
        # Same validation policy as YSU and Shin-Hong, one host reduction
        # per field.  MYJ has no known WRF 0/0 of its own, so unlike
        # Shin-Hong a NaN here is a defect rather than a reproduced quirk;
        # the refusal names the field instead of advancing bad state.
        invalid = validate_myj_pbl_outputs(out)
        if invalid is not None:
            raise FloatingPointError(
                f"MYJ returned non-finite {invalid}")
        f["pblh"][...] = out["pblh"]
        f["kpbl"][...] = out["kpbl"]
        f["mixht"][...] = out["mixht"]
        f["exch_h"][...] = out["exch_h"]
        f["el_myj"][...] = out["el_myj"]
        self.pbl_tendencies = self._couple_pbl_slot(cfg, out)
        pbl_components = (
            _composed_optional_tendency_components(cfg)
            if physics_reuses_pbl_composition(cfg)
            else _pbl_optional_tendency_components(cfg))
        self.pbl_tendencies.materialize(pbl_components)
        # No counterpart to YSU's positive-cadence raw-output retention
        # (the _run_mynn_pbl rationale): the coupling above is the only
        # consumer of the rates, and the diagnostics persist in ``fields``.
        self.last_ysu = None

    def _run_noah(self, atmosphere: Mapping[str, cp.ndarray],
                  cfg: RunConfig, itimestep: int) -> None:
        f = self.fields
        # WRF Noah receives the lowest-layer MID pressure from interfaces:
        # SFCPRS = (P8W3D(i,kts+1,j) + P8W3D(i,kts,j)) * 0.5
        # (module_sf_noahdrv.F:795), not the half-level p_phy.
        f["sfcprs"][...] = DTYPE(0.5) * (atmosphere["p_interface"][0]
                                         + atmosphere["p_interface"][1])
        f["sfctmp"][...] = atmosphere["temperature"][0]
        f["qv1"][...] = atmosphere["qv"][0]
        f["dz8w1"][...] = atmosphere["dz"][0]
        f["rib"][...] = f["br"]
        # WRF's first-RK surface call always supplies the scheme SR field to
        # Noah for every supported precipitating scheme.  Kessler explicitly
        # produces SR=0 (liquid rain), so falling back to air temperature for
        # mp_physics=1 incorrectly freezes cold rain.  The membership test is
        # microphysics_scheme_sr_available (which cites the WRF driver arm
        # per scheme) rather than a literal tuple repeated three times.
        use_scheme_sr = microphysics_scheme_sr_available(self.mp_physics)
        f["sr"][...] = (self.microphysics.sr if use_scheme_sr else
                         (atmosphere["temperature"][0] <= DTYPE(273.15)))
        # RAINBL is accumulated explicitly through the named post-RK
        # microphysics contract, then consumed exactly once by the next due
        # surface call.  Externally supplied rainbl remains additive.
        f["rainbl"] += self._pending_rainbl
        launch_noah(f, self.noah_params, self.bldt_seconds,
                    NOAH_LAYER_THICKNESS_M, frpcpn=use_scheme_sr,
                    usemonalb=self.noah_usemonalb,
                    rdlai2d=self.noah_rdlai2d,
                    opt_thcnd=self.noah_opt_thcnd,
                    itimestep=itimestep)
        f["rainbl"][...] = 0.0
        self._pending_rainbl[...] = 0.0

    def _run_ruc(self, atmosphere: Mapping[str, cp.ndarray],
                 cfg: RunConfig, itimestep: int) -> None:
        """Run the RUC LSM through WRF's own ``CASE (RUCLSMSCHEME)`` arm.

        The RAINBL/SR handling is Noah's, because it is a property of gpuwm's
        LSM seam rather than of a scheme.  Everything else -- the sea-ice
        albedo override, the argument binding, GSW, the CQS/CHS rebuild and
        ``SFCDIAGS_RUCLSM`` -- lives in :mod:`gpuwm.core.ruc_runtime` next to
        its source anchors.

        Note what is NOT here: a call to
        :meth:`_refresh_surface_diagnostics`.  RUC has its own 2-m
        diagnostic and is deliberately absent from
        :data:`LAND_SURFACE_SFCDIAGS_SCHEMES`; borrowing Noah's SFCDIAGS
        would be a different diagnostic under RUC's name.

        Every option knob is read off the configuration here, by name, so the
        registry's citation of this file as their consuming read is true.
        """
        f = self.fields
        use_scheme_sr = microphysics_scheme_sr_available(self.mp_physics)
        f["sr"][...] = (self.microphysics.sr if use_scheme_sr else
                        (atmosphere["temperature"][0] <= DTYPE(273.15)))
        f["rainbl"] += self._pending_rainbl
        self.last_ruc_census = ruc_lsm_step(
            f, atmosphere, params=self.ruc_params,
            precipitation=SurfacePrecipitationForcing.from_fields(f),
            dt=self.bldt_seconds, itimestep=itimestep,
            mosaic_lu=cfg.mosaic_lu, mosaic_soil=cfg.mosaic_soil,
            flag_sm_adj=cfg.flag_sm_adj, spp_lsm=cfg.spp_lsm)
        for name, admitted in RUC_OPTION_IDENTITY.items():
            if int(getattr(cfg, name)) != int(admitted):
                raise ValueError(
                    f"{name}={getattr(cfg, name)} reached the RUC runner "
                    "outside its admitted identity")
        f["rainbl"][...] = 0.0
        self._pending_rainbl[...] = 0.0
        SurfacePrecipitationForcing.from_fields(f).clear()

    def _run_noahmp(self, atmosphere: Mapping[str, cp.ndarray],
                    cfg: RunConfig, itimestep: int) -> None:
        """Run Noah-MP through WRF's own ``noahmplsm`` driver contract.

        The RAINBL/SR handling is Noah's, because it is a property of gpuwm's
        LSM seam rather than of a scheme: precipitation is accumulated through
        the named post-RK microphysics contract and consumed exactly once by
        the next due surface call.  Everything else -- the forcing
        marshalling, the water/sea-ice skips, the per-column parameter
        transfer and the write-back -- lives in
        :mod:`gpuwm.core.noahmp_runtime` next to its source anchors.

        Every option knob is read off the configuration here, by name.  The
        physics registry publishes them and cites this file as their
        consuming read; a default buried in the runtime would make that
        citation false and the run's own receipt would not record which
        identity it used.
        """
        f = self.fields
        use_scheme_sr = microphysics_scheme_sr_available(self.mp_physics)
        f["sr"][...] = (self.microphysics.sr if use_scheme_sr else
                        (atmosphere["temperature"][0] <= DTYPE(273.15)))
        f["rainbl"] += self._pending_rainbl
        self.last_noahmp_census = noahmp_lsm_step(
            f, atmosphere,
            params=self.noahmp_params, geometry=self.noahmp_geometry,
            precipitation=SurfacePrecipitationForcing.from_fields(f),
            coszen=f["coszen"],
            dt=self.bldt_seconds, dx=cfg.dx,
            dzs=self.noahmp_soil_thickness_m, itimestep=itimestep,
            elapsed_seconds=float(self.state.elapsed_seconds),
            dveg=cfg.dveg, opt_run=cfg.opt_run, opt_crop=cfg.opt_crop,
            opt_irr=cfg.opt_irr, opt_tdrn=cfg.opt_tdrn,
            opt_soil=cfg.opt_soil)
        # The remaining knobs are read here so the registry's citation of
        # this file is true for every one of them, and so a future widening
        # is a change in this call rather than in a solver default.  Each is
        # already refused outside its admitted value by validate_run_config;
        # this is the second line, at the seam that would consume it.
        for name, admitted in (
                ("opt_crs", cfg.opt_crs), ("opt_btr", cfg.opt_btr),
                ("opt_sfc", cfg.opt_sfc), ("opt_frz", cfg.opt_frz),
                ("opt_inf", cfg.opt_inf), ("opt_rad", cfg.opt_rad),
                ("opt_alb", cfg.opt_alb), ("opt_snf", cfg.opt_snf),
                ("opt_tbot", cfg.opt_tbot), ("opt_stc", cfg.opt_stc),
                ("opt_gla", cfg.opt_gla), ("opt_rsf", cfg.opt_rsf),
                ("opt_pedo", cfg.opt_pedo), ("opt_irrm", cfg.opt_irrm),
                ("opt_infdv", cfg.opt_infdv),
                ("noahmp_output", cfg.noahmp_output)):
            if int(admitted) != int(NOAHMP_OPTION_IDENTITY[name]):
                raise ValueError(
                    f"{name}={admitted} reached the Noah-MP runner outside "
                    "its admitted identity")
        for name, admitted in (("soiltstep", cfg.soiltstep),
                               ("noahmp_acc_dt", cfg.noahmp_acc_dt)):
            if float(admitted) != float(NOAHMP_OPTION_IDENTITY[name]):
                raise ValueError(
                    f"{name}={admitted} reached the Noah-MP runner outside "
                    "its admitted identity")
        f["rainbl"][...] = 0.0
        self._pending_rainbl[...] = 0.0
        SurfacePrecipitationForcing.from_fields(f).clear()

    def refresh_surface_diagnostics_after_analysis(
            self, state: DomainState) -> bool:
        """Rerun the FINAL 2 m provider after an analysis moved the column.

        WHY A DA CYCLE NEEDS THIS.  T2 and Q2 are not prognostic.  They are
        diagnosed once per surface call, from the lowest model level and the
        surface endpoint as they stood at that call.  An analysis that
        changes the lowest-level temperature or moisture therefore leaves
        the PUBLISHED 2 m fields describing the state before the analysis --
        and the next output frame carries them out as though they described
        the state after it.  The error is largest exactly where the analysis
        did the most work, which is the worst possible place for it.

        So the provider reruns HERE, between the analysis and the first
        output frame, and the provider that reruns is the one that owns the
        published value for THIS land-surface scheme: Noah's SFCDIAGS for
        scheme 2 (:data:`LAND_SURFACE_SFCDIAGS_SCHEMES`), and nothing for
        RUC or Noah-MP, whose own 2 m diagnostics are computed inside their
        LSM step and cannot be rerun without rerunning the LSM.  Scheme 2
        is spelled here as its resolved runner (``_run_noah``) rather than
        as the selector value, so a scheme renumbering cannot silently
        redirect this at a different LSM.  Returns
        True when a value was republished and False when this configuration
        has no rerunnable provider, so a caller can say which happened
        rather than assume.

        NOT a physics step: no scheme is advanced, no accumulator moves, no
        model second passes.  It republishes a diagnostic from the state
        that is already there.
        """
        if not self.surface_enabled:
            return False
        # Dispatched on the driver's OWN resolved selector map rather than
        # on a RunConfig handed in, because the caller of this method is a
        # DA applier that has a state and no config, and because dispatching
        # on the resolved map is what every other seam in this class does.
        if self.scheme_dispatch["sf_surface_physics"] != "_run_noah":
            return False
        atmosphere = _prepare_atmosphere(state)
        # PSFC first, for the same reason the ordinary surface call
        # refreshes it first: the diagnostic divides by a density built
        # from it, and an analysis can move the pressure column.
        self.fields["psfc"][...] = atmosphere["p_interface"][0]
        self._refresh_surface_diagnostics(atmosphere)
        return True

    def _refresh_surface_diagnostics(self, atmosphere) -> None:
        """Transcribe WRF v4.6.1 SFCDIAGS for the supported Noah path.

        Noah updates TSK/HFX/QFX/QSFC after SFCLAY has diagnosed T2/Q2/TH2.
        WRF therefore calls SFCDIAGS after the LSM and before the PBL driver
        (module_surface_driver.F:2983-3000; module_sf_sfcdiags.F:45-72).
        gpuwm does not expose WRF's UA_PHYS or HWRF compile-time branches, so
        this is their standard false/non-HWRF formulation used by real74.

        One documented divergence, on the lower bound only.  WRF's flux
        inversion ``Q2 = QSFC - QFX/(RHO*CQS2)`` (module_sf_sfcdiags.F:56)
        carries no bound at all, while Noah's own surface value
        ``QSFC = Q1/(1-Q1)`` with ``Q1 = qv(k=1) + QFX/(RHO*CHS)``
        (module_sf_noahlsm.F:795,801) is likewise unbounded.  Under downward
        moisture flux over very cold snow -- the Antarctic-plateau regime,
        where the surface is radiatively colder than the air above it and
        frost deposits -- the correction exceeds the whole vapour content of
        the lowest model level and both go negative.  A mixing ratio cannot
        be negative, and every consumer of Q2 (relative humidity, dewpoint,
        vapour-pressure deficit) needs the log of a positive number, so the
        engine cannot simply publish WRF's arithmetic here.

        WRF authored the remedy itself and left it commented out at
        module_sf_noahdrv.F:1276-1282: "prevent diagnostic ground q (q1) ...
        as happens over snow cover where the cqs2 value also becomes
        irrelevant / by setting cqs2=chs in this situation the 2m q should
        become just qv(k=1)".  Setting ``CQS2 = CHS`` makes the two
        corrections cancel exactly, leaving ``Q2 = qv(k=1)``.  That is what
        this does, and only where the published value would otherwise leave
        the physical range -- so a column whose flux inversion is meaningful
        is bit-for-bit WRF's.  WRF's two OTHER surface-diagnostic
        implementations bound the same quantity for the same reason
        (module_sf_mynn.F:1148, module_sf_sfcdiags_ruclsm.F:125-127).
        """
        f = self.fields
        rho = f["psfc"] / (DTYPE(c.RD) * f["tsk"])
        q_active = f["cqs2"] >= DTYPE(1.0e-5)
        t_active = f["chs2"] >= DTYPE(1.0e-5)
        safe_cqs2 = cp.where(q_active, f["cqs2"], DTYPE(1.0))
        safe_chs2 = cp.where(t_active, f["chs2"], DTYPE(1.0))
        diagnosed = cp.where(
            q_active,
            f["qsfc"] - f["qfx"] / (rho * safe_cqs2),
            f["qsfc"])
        # Only where the published number is not a mixing ratio at all.  A
        # column whose surface endpoint is negative but whose 2 m value lands
        # inside the range keeps WRF's arithmetic, bit for bit: the divergence
        # is scoped to the declared-range violation and to nothing else.
        representable = diagnosed > DTYPE(0.0)
        f["q2"][...] = cp.where(representable, diagnosed,
                                atmosphere["qv"][0])
        f["t2"][...] = cp.where(
            t_active,
            f["tsk"] - f["hfx"] / (rho * DTYPE(c.CP) * safe_chs2),
            f["tsk"])
        f["th2"][...] = f["t2"] * cp.power(
            DTYPE(c.P0) / f["psfc"], DTYPE(c.RCP))

    def _run_ysu(self, atmosphere: Mapping[str, cp.ndarray],
                 cfg: RunConfig) -> None:
        f = self.fields
        out = launch_ysu(
            atmosphere["u"], atmosphere["v"], atmosphere["theta"],
            atmosphere["qv"], atmosphere["qc"], atmosphere["qi"],
            atmosphere["pressure"], atmosphere["p_interface"],
            atmosphere["exner"], atmosphere["dz"],
            psfc=atmosphere["p_interface"][0], znt=f["znt"], ust=f["ust"],
            hfx=f["hfx"], qfx=f["qfx"], wspd=f["wspd"], br=f["br"],
            # WRF's PBL driver binds YSU's PSIM/PSIH dummies to the FULL
            # similarity denominators ln(z/z0)-psi, not the raw integrated
            # psi corrections (module_pbl_driver.F:1228 ``PSIM=fm,
            # PSIH=fhh``); YSU reconstructs zol as br*fm^2/fh from them.
            psim=f["fm"], psih=f["fh"], xland=f["xland"],
            u10=f["u10"], v10=f["v10"], dt=self.bldt_seconds,
            rthraten=cp.ascontiguousarray(
                self.rthratenlw + self.rthratensw),
            ysu_topdown_pblmix=cfg.ysu_topdown_pblmix)
        try:
            # The forensic set is exactly what the scheme CONSUMES and
            # another component PRODUCED: the surface-layer/LSM coupling
            # fields, and the first-layer geometry the heat solve's
            # surface source divides by.  ``dp1`` is materialized only on
            # the failing branch, inside the guard, so the healthy path
            # allocates nothing.
            validate_ysu_tendencies(
                out, status=self.state.scratch(
                    (1,), "physics_validation_status").view(cp.uint32),
                grid_id=cfg.grid_id,
                producer_inputs=_YsuProducerInputs(f, atmosphere))
        except FloatingPointError:
            self.ysu_nan_guard_fires += 1
            raise
        f["pblh"][...] = out["hpbl"]
        f["kpbl"][...] = out["kpbl"]
        f["exch_h"][...] = out["exch_h"]
        f["exch_m"][...] = out["exch_m"]
        # GF's RTHBLTEN/RQVBLTEN lanes are retained inside the shared
        # coupling seam, so YSU gets them on exactly the same terms as
        # every other scheme in the PBL slot.
        self.pbl_tendencies = self._couple_pbl_slot(cfg, out)
        pbl_components = (
            _composed_optional_tendency_components(cfg)
            if physics_reuses_pbl_composition(cfg)
            else _pbl_optional_tendency_components(cfg))
        self.pbl_tendencies.materialize(pbl_components)
        # All bldt=0 consumers are above.  Positive cadence preserves the
        # historical diagnostic retention byte-for-byte.
        self.last_ysu = out if physics_retains_ysu_output(cfg) else None

    def _run_mynn_pbl(self, atmosphere: Mapping[str, cp.ndarray],
                      cfg: RunConfig) -> None:
        """Run the MYNN EDMF PBL through WRF's own wrapper contract.

        The call cadence, mass coupling and A-grid-to-C-grid interpolation are
        YSU's, because they are properties of WRF's PBL seam rather than of a
        scheme: ``calculate_phy_tend`` multiplies the same way and
        ``phy_bl_ten`` interpolates the same way whichever driver filled
        RUBLTEN.  What is MYNN's -- the specific-humidity conversions, the
        ``initflag`` selection, and the ten carried 3-D state arrays -- lives
        in :mod:`gpuwm.core.mynn_pbl_runtime` next to its source anchors.
        """
        f = self.fields
        itimestep = int(np.floor(
            float(self.state.elapsed_seconds) / cfg.dt + 0.5)) + 1
        # Every MYNN identity knob is read off the configuration here, by
        # name, rather than inherited from a default further down.  The
        # physics registry publishes these as knobs and cites this file as
        # their consuming read; a default buried in the solver would make
        # that citation false, and the run's own receipt would not record
        # which identity it used.
        out = mynn_pbl_step(
            atmosphere, f, w=self.state.w, dx=cfg.dx,
            delt=self.bldt_seconds, itimestep=itimestep,
            mp_physics=cfg.mp_physics,
            # The domain state is what makes MYNN's working set a set of
            # priced scratch slots instead of ~46 kB of pool churn per
            # column per step; see gpuwm/core/mynn_pbl_scratch.py.
            state=self.state,
            closure=cfg.bl_mynn_closure,
            bl_mynn_cloudpdf=cfg.bl_mynn_cloudpdf,
            bl_mynn_mixlength=cfg.bl_mynn_mixlength,
            bl_mynn_edmf=cfg.bl_mynn_edmf,
            bl_mynn_edmf_mom=cfg.bl_mynn_edmf_mom,
            bl_mynn_edmf_tke=cfg.bl_mynn_edmf_tke,
            bl_mynn_mixscalars=cfg.bl_mynn_mixscalars,
            bl_mynn_cloudmix=cfg.bl_mynn_cloudmix,
            bl_mynn_mixqt=cfg.bl_mynn_mixqt,
            bl_mynn_output=cfg.bl_mynn_output,
            bl_mynn_tkeadvect=cfg.bl_mynn_tkeadvect,
            icloud_bl=cfg.icloud_bl)
        validate_mynn_tendencies(out)
        self.pbl_tendencies = self._couple_pbl_slot(cfg, out)
        pbl_components = (
            _composed_optional_tendency_components(cfg)
            if physics_reuses_pbl_composition(cfg)
            else _pbl_optional_tendency_components(cfg))
        self.pbl_tendencies.materialize(pbl_components)
        # MYNN has no counterpart to YSU's positive-cadence raw-output
        # retention: every consumer of its rates is the coupling above, and
        # its diagnostics already persist in ``fields``.
        self.last_ysu = None

    def _run_shinhong(self, atmosphere: Mapping[str, cp.ndarray],
                      cfg: RunConfig) -> None:
        """Run the Shin-Hong scale-aware PBL (bl_pbl_physics=11).

        The call cadence, the surface-layer coupling set, the mass
        coupling and the A-grid-to-C-grid interpolation are all YSU's,
        because they are properties of WRF's PBL seam rather than of a
        scheme (the _run_mynn_pbl rationale).  What is Shin-Hong's --
        the partition functions, the prescribed-profile nonlocal
        transport, the TKE diagnostic chain -- lives in
        :mod:`gpuwm.core.shinhong` next to its source anchors.

        Scheme-specific bindings, each with its source of truth:

        * ``psim``/``psih`` receive the FULL similarity denominators
          ``fm``/``fh`` exactly as YSU's seam does: the scheme
          reconstructs zol as ``br*fm^2/fh`` from them
          (module_bl_shinhong.F:751-780, transcribed at
          gpuwm/verify/shinhong_ref.py:735-737), the same construction
          WRF's PBL driver feeds both YSU-family wrappers.
        * ``corf`` is ``state.f``, the per-column Coriolis parameter
          2*Omega*sin(lat) at mass points -- the SAME field the
          dycore's Coriolis+curvature kernel consumes
          (gpuwm/core/dycore.py add_coriolis_curvature; filled by
          DomainState.set_map_coriolis, zeros on idealized runs).  The
          scheme reads it only through ``f = max(corf, eps1)``
          (module_bl_shinhong.F:2001, kernel :453), so the idealized
          zero lands on WRF's own eps1 floor.
        * ``tke`` is ``state.e_sgs``, the scheme's own published SGS
          TKE fed back as next step's ``tke_in`` -- and after the
          launch the validated ``out["tke"]`` is written back to
          ``state.e_sgs``, which is the field the D1 gray-zone
          instrument scores and the restart stream carries.
        * ``tke_diag=1`` always: the dispatch-row note documents why
          the WRF namelist knob is not imported (the chain is a
          passenger diagnostic; tendencies never read it).
        * ``dx``/``dy`` are the run's own spacings -- the whole point
          of the scheme is that the answer moves with sqrt(dx*dy), so
          each nest launches with its own pair.
        * No ``exch_m``: the WRF wrapper publishes a heat exchange
          coefficient only (EXCH_H); the driver's exch_m field keeps
          its allocated zeros rather than inheriting another scheme's.
        """
        f = self.fields
        # ENTRY CONTRACT (the upstream half of task #206): WRF runs this
        # scheme only on TKE the shinhonginit floor (epsq2l/2) and the
        # chain's own amax1 floors guarantee, and entry TKE at or below
        # zero -- or non-finite -- is the ONE input class that reproduces
        # the "non-finite tke tendency" crash signature (mixlen's
        # el0 = alph*szq*0.5/sq goes 0/0 when sqrt(q2) integrates to
        # zero; the NaN walks el -> disel -> prodq2's q2 into the
        # published passenger pair while every tendency stays finite).
        # MEASURED the other direction (work/probe_shinhong_big_ensemble
        # .py, 2026-08-16): 24,000 adversarial columns with entry TKE at
        # or above the floor produced ZERO tke-confined non-finites.
        # The engine's own writers keep the invariant (state.py cold
        # start, the validated writeback below), but this seam used to
        # trust ANY writer -- a hand-carried restart, an analysis state,
        # an external tool, a bit flip on a no-ECC card -- so the floor
        # is re-asserted here, in place, upstream of the output guard.
        # min() is the identity fast path: one scalar readback beside
        # the validation readback this call already pays, and a healthy
        # run's trajectory is bit-identical.  NaN fails the >= compare,
        # so corruption lands in the heal branch too.
        e_sgs = self.state.e_sgs
        # The floor in the CARRIER'S dtype: the module literal is a
        # Python float and float32(0.005) sits just below it, so a
        # float64 compare would flag every legal cold-start value.
        entry_floor = DTYPE(SHINHONG_TKE_FLOOR)
        entry_min = float(e_sgs.min())
        if not (math.isfinite(entry_min)
                and entry_min >= float(entry_floor)):
            xp = cp.get_array_module(e_sgs)
            good = xp.isfinite(e_sgs) & (e_sgs >= entry_floor)
            healed = int(e_sgs.size) - int(good.sum())
            e_sgs[...] = xp.where(good, e_sgs, entry_floor)
            if not self._shinhong_entry_advisory:
                self._shinhong_entry_advisory = True
                print(
                    "shin-hong: the SGS TKE carrier (state.e_sgs) arrived "
                    f"at the PBL seam with {healed} entry value(s) below "
                    "WRF's shinhonginit cold-start floor (epsq2l/2 = "
                    f"{SHINHONG_TKE_FLOOR}) or non-finite; healed to the "
                    "floor in place and continuing.  The scheme's own "
                    "writers keep this invariant, so a value outside it "
                    "came from another writer (a carried-in state, an "
                    "analysis, memory corruption).  Entry TKE at or below "
                    "zero is the one input class that reproduces the "
                    "'non-finite tke tendency' failure.  Said once per "
                    "run; the heal applies whenever needed.",
                    file=sys.stderr)
        out = launch_shinhong(
            atmosphere["u"], atmosphere["v"], atmosphere["theta"],
            atmosphere["qv"], atmosphere["qc"], atmosphere["qi"],
            atmosphere["pressure"], atmosphere["p_interface"],
            atmosphere["exner"], atmosphere["dz"],
            e_sgs,
            psfc=atmosphere["p_interface"][0], znt=f["znt"], ust=f["ust"],
            hfx=f["hfx"], qfx=f["qfx"], wspd=f["wspd"], br=f["br"],
            psim=f["fm"], psih=f["fh"], xland=f["xland"],
            u10=f["u10"], v10=f["v10"], corf=self.state.f,
            dt=self.bldt_seconds, dx=cfg.dx, dy=cfg.dy, tke_diag=1)
        # Validation policy (task #206): one batched device kernel over
        # every floating-point output through the shared status scratch
        # word, and the DISPOSITION depends on which outputs are bad.  A
        # non-finite in anything a tendency or a consumer reads
        # (du/dv/dtheta/dqv/dqc/dqi/exch_h, hpbl/wstar/delta) is fatal
        # exactly as it always was, first-invalid ordering preserved.  A
        # non-finite CONFINED to the passenger pair (tke, el) is the
        # transcribed chain's own documented hazard -- WRF's arithmetic
        # divides by quantities that legitimately reach zero there, WRF's
        # default never computes the chain, and the tendencies beside it
        # are finite -- so it is repaired to the chain's own floor and
        # said out loud rather than killing a run whose physics is
        # intact.  This is what crashed the composed
        # Thompson/Shin-Hong/MM5/Noah/RRTMG-legacy suite on a bare 12 km
        # GFS case while an unmodified WRF ran the same initial state to
        # completion: the chain went non-finite in tke alone, and the old
        # policy raised over a diagnostic no tendency reads.
        invalid = invalid_shinhong_outputs(
            out, self.state.scratch(
                (1,), "physics_validation_status").view(cp.uint32))
        fatal = [name for name in invalid
                 if name not in SHINHONG_PASSENGER_OUTPUTS]
        if fatal:
            raise FloatingPointError(
                f"Shin-Hong returned non-finite {fatal[0]} tendency")
        if invalid:
            repaired = repair_shinhong_passenger_outputs(out)
            if not self._shinhong_passenger_advisory:
                self._shinhong_passenger_advisory = True
                described = ", ".join(
                    f"{name} ({count} value(s))"
                    for name, count in sorted(repaired.items()))
                print(
                    "shin-hong: the SGS TKE diagnostic chain returned "
                    f"non-finite passenger values ({described}); repaired "
                    "to the chain's own floor and continuing.  Every "
                    "tendency this step consumed is finite; the chain is "
                    "a diagnostic WRF's default never computes, and its "
                    "transcribed arithmetic divides by quantities that "
                    "legitimately reach zero.  Said once per run; the "
                    "repair applies whenever needed.",
                    file=sys.stderr)
        f["pblh"][...] = out["hpbl"]
        f["kpbl"][...] = out["kpbl"]
        f["exch_h"][...] = out["exch_h"]
        # The scheme's own subgrid energy, written back in place so the
        # restart alias and every diagnostic reader see the same array.
        self.state.e_sgs[...] = out["tke"]
        self.pbl_tendencies = self._couple_pbl_slot(cfg, out)
        pbl_components = (
            _composed_optional_tendency_components(cfg)
            if physics_reuses_pbl_composition(cfg)
            else _pbl_optional_tendency_components(cfg))
        self.pbl_tendencies.materialize(pbl_components)
        # No counterpart to YSU's positive-cadence raw-output retention
        # (the _run_mynn_pbl rationale): every consumer of the rates is
        # the coupling above, and the diagnostics persist in ``fields``
        # and ``state.e_sgs``.
        self.last_ysu = None

    def _run_sase(self, atmosphere: Mapping[str, cp.ndarray],
                  cfg: RunConfig) -> None:
        """One SASE-L1 update replacing the YSU slot (stage-3 Task 6).

        Sequence per due surface/PBL step (bldt == 0 is enforced at
        initialize_physics, so "due" means every model step):

        1. Destagger the C-grid winds to A-grid WORK COPIES -- the
           atmosphere dict's u/v are read again by cumulus after this
           slot, so SASE never mutates them; w averages to half levels.
        2. n2 from the model theta profile: N^2 = (g/theta)*d(theta)/dz
           on the SAME clamped per-column-dz stencil every SASE vertical
           operator uses (launch_n2; authority ``brunt_vaisala_n2`` is
           the pinned discrete form).  SASE-M1 (S4-2): n2_eff =
           launch_moist_n2 beside it -- the DK82 saturated N^2_m where
           the registered binary switch fires, the dry bits verbatim
           elsewhere (authority ``moist_n2``); the step receives BOTH
           (``n2_moist=n2_eff`` for the three substitution points, dry
           ``n2`` for the w-sensor screen).  SASE-M1b (S4-3c): the
           same pair also drives the moist master-length limb inside
           the step's vertical channel (launch_sase_step passes
           ``n2_dry=n2`` down exactly when the seam is engaged) -- no
           additional driver field or call.  ``cfg.sase_moist_n2``
           (default True = the model as built) gates the seam: False
           passes ``n2_moist=None`` so the step uses the DRY field at
           all three substitution points and forms no M1b bound, while
           the moist field is still computed for the M2 vent's
           saturation veto (step 5) -- M1 off, M2 standing.
        3. The named surface source (:func:`sase_surface_e_source`)
           deposits ``dt*source`` into the lowest ``e_sgs`` level BEFORE
           the fused step, so this interval's dissipation/clip acts on
           it (the source is non-negative, so the E_MIN floor holds).
        4. ``launch_sase_step`` advances the work winds and
           ``state.e_sgs`` in place through the S3-6c SPLIT scheme
           (authority ``sase_split_step``): domain-level solve
           (plan-bound decision 1) on the uniform clamped strain with
           the representative spacing dz_rep = FP64 mean layer
           thickness (carry-forward 4); the explicit HORIZONTAL stress
           divergence plus the implicit backward-Euler K_v vertical
           channel (per-column FP64 Thomas) run on the per-column
           ``dz_col`` = the model's live layer thicknesses; delta =
           sqrt(dx*dy), the horizontal filter scale of the L1
           sensor/solve operators (they are horizontal-scale operators,
           spec 4.1); the S3-6f partition bounds run inside the step
           (bulk-Richardson z_i from the model's own u/v/theta columns
           -> Delta/z_i cap; N^2-screened w-based resolved-fraction
           bound riding the SAME n2 field passed for the stability
           length -- f_used = min(f_solved, f_cap, f_w), diagnostics
           in the retained ledger); under damp_opt == 3 the config's
           zdamp engages
           the S3-6e damping-layer production taper (weight law shared
           with the audited KDH damper); S3-6j: the live sfclay ``ust``
           field drives the IMPLICIT surface momentum stress inside
           the step's u/v Thomas solves (drag conductance
           u*^2/max(|V1|, SFC_WSPD_FLOOR) folded into the bottom
           diagonal -- the YSU linearization; authority module
           docstring, S3-6j section), closing the missing-friction
           hole where ust fed only the step-3 e source; S3-9c: the
           live sfclay ``wspd`` field (the gust-enhanced speed u*
           was computed against) rides beside ust, multiplying the
           conductance by the audited YSU gustiness factor
           (spd1/wspd)^2 (authority module docstring, S3-9c
           section) so gust-inflated u* is not applied at full
           strength against the resolved wind.  The returned
           ``kv`` and
           ``km_h`` fields (popped from the ledger) are the vertical
           and governed-horizontal diffusivities the scalar channel
           rides.
        5. A-grid rates (work - before)/dt; the split scalar channel
           for theta, qv, qc, qi -- IN THAT ORDER: horizontal K_h
           down-gradient mixing via launch_scalar_mix riding the split
           step's exported GOVERNED diffusivity field as K_h =
           km_h/Pr_t(f) (S3-6e: the identical K the momentum stress
           and e-transport used, at the post-surface-deposit e^n --
           flux-consistent by construction, superseding the Task-6
           pre-step-e kh_coef convention, which survives only as the
           CPU-shim seam fallback), then the implicit K_v/Pr_t(f)
           vertical solve of s* = s + dt*T_h on the step's own ``kv``
           field (the same K_v/Pr_t(f) the split step's buoyancy term
           used), rate = (s_new - s)/dt.  S3-11b: the theta and qv
           rows of that solve carry the SURFACE SCALAR-FLUX DEPOSIT
           (authority ``surface_scalar_flux_deposit``, registered
           SFC_SCALAR_FLUX = "explicit-deposit-v1") -- the live
           sfclay HFX/QFX applied to the lowest layer,
           theta*[0] += dt*HFX/(rho1*CP_AIR*thick_0) and
           qv*[0] += dt*QFX/(rho1*thick_0), fused into the bottom rhs
           BEFORE the sweep at the SAME fresh fluxes the step-3 e
           source consumes and the SAME rho1 (computed once,
           ``sase_surface_rho1``); qc/qi rows take no deposit and the
           seam is off only where the fluxes are zero (in-kernel
           guard, no config flag).  SASE-M2 (S4-5, spec C1-C3): the
           theta/qv/qc rows ALSO carry the CONDITIONAL VENTING
           deposit -- the diagnosed shallow mass-flux limb's
           face-registered profiles (authority ``plume_vent_flux``,
           launched here as ``launch_plume_vent_flux`` from the FROZEN
           pre-step state at the step's used f) deposited in flux form,
           phi*[k] += (Fs[k] - Fs[k+1])*dt/(rho1*thick_k), on the
           pre-solve state BEFORE the sweep, under the registered
           cap-family uniform rescale Fs = s*F (authority
           ``vent_deposit_rescale``, enforced at this seam and nowhere
           else).  F[0] = F[nz] = +0.0, so the scalar ledger extends
           with a ZERO net-column term and the surface flux stays owned
           by the S3-11a deposit above (double-counting ban).  Nothing
           enters ``sase_split_step`` and no Thomas row moves.  qi takes
           no vent deposit (authority scope).  Pr_t(f) is the S3-6g
           regime-blended Prandtl number (authority prandtl_blend)
           at the step's used f -- recomputed here from the retained
           ledger f, identical to the step's exported pr_t
           diagnostic.  Scope is plan-bound
           decision 2: moist scalars qv/qc/qi mix, number
           concentrations and qr do NOT (parity with YSU); the
           legacy CPU-shim fallback (no ``kv`` in the ledger) has no
           vertical solve and therefore no deposit row -- S4-5b Item 4c:
           it therefore does not DIAGNOSE the limb either (it used to
           launch the flux and cap-scale kernels and discard their
           output silently), and a RuntimeError after the scalar loop
           enforces that every diagnosed row was deposited.
        6. Dissipative heating: ``heat`` (m2 s-2 accumulated over dt,
           = the S3-6d EXACT analytic decay decrement
           e* - e*/(1 + b*dt)^2 minus clip_gain, plus the S3-6e smag
           bypass and taper redirect) deposits into internal
           energy through the theta tendency,
           dtheta += heat/(dt*cp*exner) -- the spec-4.2 "existing
           heating pathway" (same slot as h_diabatic-style theta
           rates).  S3-6f doc fix: since S3-6e ``heat`` is NOT
           pointwise sign-definite -- the smag bypass share P_h,heat
           inherits the horizontal pairing's local sign (backscatter
           cells), beside the rare source-driven floor cells -- exactly
           as the restated ledger defines it (authority module
           docstring); domain sums stayed positive in every suite
           fixture.
        7. couple_ysu_tendencies mass-couples/staggeres u, v, theta,
           qv, qc, qi exactly like YSU's rates; the w rate couples
           through :func:`couple_sase_w_tendency` and rides the pbl
           stack as the plain ``rw`` attribute (restart-REBUILT:
           replaced here every step before any read; never a serialized
           dataclass field, so the restart inventory is unchanged).

        STABILITY -- RESOLVED by S3-6b/6c (implicit vertical).  History
        (S3-6 review, kept for the record): the v0 SASE diffusion was
        fully explicit with the viscosity on the horizontal scale
        (nu ~ 0.3*delta*sqrt(e) at f = 0), so the vertical-diffusion
        number 2*nu*dt/dz1^2 was unbounded on coarse-Delta domains with
        thin lowest layers (d01: delta = 12 km, dz1 ~ 50 m, dt = 60 s
        -> O(10^2) once the surface source spins sqrt(e) up to ~1).
        The fix landed at the FORMULATION level exactly as required:
        the vertical channel now rides K_v = C_KV*l_v*sqrt(e) on the
        Blackadar length (S3-6h: further bounded in the RANS limb by
        the BL89 displacement lengths and the retained l_s; S3-6i:
        the RANS limb's coefficient decouples to the stable-limit
        C_KS where l_s binds, K_v -> C_KS*e/N -- the authority module
        docstring's S3-6h/S3-6i sections) and every vertical
        diffusion (momentum, e,
        scalars) advances backward-Euler through unconditionally stable
        M-matrix Thomas columns -- no vertical CFL bound exists; the
        explicit horizontal channel keeps its own 13x CFL margin at d01
        scale (authority test derivations).

        Registered approximations (controller ledger, 2026-07-20):
        (1) [RESOLVED by S3-6g] the fixed PR_T = 1/3 put the
        RANS-regime vertical scalar channel at K_v/PR_T ~ 3*kappa*u**z
        vs the observed ~1.2*kappa*u**z -- the smoke-c inversion-
        mixdown/wind-amplification root cause; the blended Pr_t(f)
        gives K_v/PR_RANS ~ 1.18*kappa*u**z at f = 0 (inside the
        observed band) while the LES limit keeps PR_LES.
        (2) The one-constant closure's equilibrium e ~ 1.05*u*^2 sits
        ~3-5x below the observed 3.3-5.5*u*^2 (sqrt(e) ~2x low),
        biasing l_s, horizontal K, buoyancy flux, and TKE products --
        still open; BLOCKS scientific claims about TKE magnitudes
        until revisited (validation-stage carry).

        Specified-boundary policy (registered adjudication, S3-6
        review): after the fused step, ``e_sgs`` is held at the E_MIN
        floor across the outer ``spec_bdy_width`` rows on all four
        lateral edges (the width covers the widest 4-cell test-filter
        halo, so the periodic-wrap contamination of the SASE horizontal
        operators never reaches the interior through e); the same width
        is excluded from the domain-level solve reductions
        (``exclude_boundary_width``); the coupled tendencies were
        already boundary-masked by the coupling helpers.
        """
        state = self.state
        f = self.fields
        nz, ny, nx = state.p.shape
        dt = self.bldt_seconds
        theta = atmosphere["theta"]
        dzf = atmosphere["dz"]
        u_a, v_a = atmosphere["u"], atmosphere["v"]
        u_w = u_a.copy()
        v_w = v_a.copy()
        w_a = cp.ascontiguousarray(
            DTYPE(0.5) * (state.w[:-1] + state.w[1:]))
        w_w = w_a.copy()
        n2 = launch_n2(theta, dz_col=dzf)
        # SASE-M1 (S4-2 device wiring of the authority n2_moist seam;
        # sase_ref module docstring, SASE-M1 section): n2_eff = the
        # DK82 saturated moist N^2 where the registered binary switch
        # fires (qc > 0 OR qv >= qs,liq -- MOIST_STABILITY_SWITCH),
        # the dry field BITWISE elsewhere, computed from the SAME
        # theta/dz stencil the dry n2 rides plus the already-resident
        # qv/qc/full-pressure fields.  The step consumes it at exactly
        # the authority's three substitution points (l_s, e-budget
        # buoyancy, K stability suppression) while its w-sensor keeps
        # the dry n2 -- both fields go down together.  The seam is inert
        # exactly where the air is unsaturated (bit-copied dry cells make
        # every substitution a no-op -- the unsaturated bitwise-identity
        # contract, pinned on device); ``cfg.sase_moist_n2`` (default
        # True = the model as built) can additionally disable it
        # OUTRIGHT at the ``n2_moist`` argument below.
        # SASE-M1b (S4-3c): the SAME two fields also carry the moist
        # master-length limb -- launch_sase_step hands the dry n2 to
        # its vertical channel as the substitution mask's other half
        # (n2_dry) exactly when the seam is engaged, so the RANS-limb
        # master lengths are min-bounded by the moist parcel-excursion
        # length in saturated cells (MOIST_MASTER_LENGTH =
        # "bl89-n2eff-excursion-min-v1"; the G-M3 deck-clearing fix).
        # No new driver-held field: the limb is computed in-thread
        # from the fields already resident here (the preflight
        # sase_workspace_phases transcription is unchanged -- the
        # S4-3 pairing contract holds as-is).
        n2_eff = launch_moist_n2(theta, atmosphere["qv"],
                                 atmosphere["qc"], atmosphere["pressure"],
                                 n2, dz_col=dzf)
        # SASE-M1 SWITCH (RunConfig.sase_moist_n2, DEFAULT True = the
        # model as built -- the argument below is then the SAME object
        # this line has always passed, so the step is bitwise
        # unchanged).  False passes ``n2_moist=None``, which is the
        # launcher's OWN pre-M1 path: n2_eff collapses to the dry n2 for
        # substitution points 1 and 3, has_moist == 0 gates the point-2
        # buoyancy branch off in the e-update kernel, and no ``n2_dry``
        # kwarg is formed so the M1b moist master-length limb never runs
        # (gpuwm/core/sase.py launch_sase_step, SASE-M1/M1b docstring
        # sections).  All four entry points hang off this one argument;
        # there is no second wire into the step.
        # THE MOIST FIELD IS STILL COMPUTED AND STILL USED BELOW: the
        # M2 venting limb takes the (moist, dry) pair as its saturation
        # VETO -- one bitwise-departure comparison in
        # sase_plume_vent_flux, never a stability value -- so M2 is
        # deliberately left standing.  This key isolates the M1
        # diffusion channel; standing M2 down with it would confound the
        # comparison it exists to make.  Keeping the launch
        # unconditional also keeps the preflight residency (the
        # launch_moist_n2 work field) identical either way.
        n2_moist_arg = n2_eff if getattr(cfg, "sase_moist_n2", True) else None
        # S3-11b: ONE lowest-level moist density serves BOTH surface
        # seams -- the e source here and the S3-11a scalar-flux deposit
        # in the step-5 scalar loop below consume this same field (the
        # S3-11a report obligation; sase_surface_rho1 docstring).
        rho1 = sase_surface_rho1(
            p1=atmosphere["p_interface"][0],
            t1=atmosphere["temperature"][0], qv1=atmosphere["qv"][0])
        # SASE-M2 (S4-5 deposit seam, spec C1-C3): the venting limb is
        # DIAGNOSED FROM THE FROZEN PRE-STEP STATE.  theta/qv/qc/p are
        # read-only through the whole slot (the split step's ledger
        # theorem reads theta read-only and the scalar rows are not
        # written until the coupling), so ``e_sgs`` is the ONLY input
        # that moves: the surface e source below and the fused step
        # both write it in place.  One pre-step copy freezes it (the
        # preflight sase_workspace_phases pairing transcribes this
        # field; the flux itself cannot be computed here because its
        # (1 - f) two-product blend needs the step's USED f, which the
        # step has not yet solved).
        e_pre = state.e_sgs.copy()
        source = sase_surface_e_source(
            ust=f["ust"], hfx=f["hfx"], qfx=f["qfx"], theta1=theta[0],
            qv1=atmosphere["qv"][0], p1=atmosphere["p_interface"][0],
            t1=atmosphere["temperature"][0], dz1=dzf[0], rho1=rho1)
        state.e_sgs[0] += DTYPE(dt) * source
        heat = cp.empty((nz, ny, nx), dtype=DTYPE)
        delta = math.sqrt(cfg.dx * cfg.dy)
        dz_rep = float(dzf.mean(dtype=cp.float64))
        boundary_width = int(cfg.spec_bdy_width) if cfg.specified else 0
        # S3-6e damping-layer taper: engaged exactly when the model
        # runs the damp_opt=3 KDH damper whose weight law it reuses.
        zdamp = (float(cfg.zdamp)
                 if cfg.damp_opt == 3 and cfg.zdamp > 0.0 else None)
        # S3-6j: the sfclay friction velocity now ALSO drives the
        # implicit surface momentum stress inside the split step's u/v
        # Thomas solves (authority module docstring, S3-6j section) --
        # before this seam, ust fed ONLY the surface e source above and
        # the momentum column ran zero-flux at the ground (the
        # missing-friction hole Probe 4 isolated).  S3-9c: the live
        # sfclay gust-ENHANCED speed field rides beside it -- sfclay's
        # ust is computed against wspd = max(sqrt(|V1|^2 + vconv^2 +
        # vsgd^2), 0.1), so the drag row carries the audited YSU
        # gustiness factor (wspd1/wspd)^2 (npref.py:6495-6496; the
        # same f["wspd"] the YSU seam feeds at its own momentum row).
        # Both fields are fresh from _run_sfclay in this same due step.
        ledger = launch_sase_step(
            u_w, v_w, w_w, theta, state.e_sgs, dx=cfg.dx, dy=cfg.dy,
            dz=dz_rep, delta=delta, dt=dt, n2=n2, dz_col=dzf, heat=heat,
            exclude_boundary_width=boundary_width, zdamp=zdamp,
            ust=f["ust"], wspd_sfc=f["wspd"], n2_moist=n2_moist_arg,
            # SASE S3-6k SWITCH (RunConfig.sase_stable_dissipation,
            # DEFAULT False = the model as built -- the launcher then
            # gates the kernel's decay coefficient back to the literal
            # C_E, so the state is bitwise unchanged).  True decouples
            # the stable-limb dissipation coefficient to C_ES on the
            # SAME rho blend S3-6i built for the diffusivity half
            # (gpuwm/verify/sase_ref.py module docstring, S3-6k
            # section).  One argument, one entry point: the coefficient
            # is formed inside the e-update kernel from state it
            # already holds, so there is no second wire and no new
            # device field.
            stable_dissipation=bool(
                getattr(cfg, "sase_stable_dissipation", False)),
            # SASE S3-12 SWITCH (RunConfig.sase_additive_dissipation,
            # DEFAULT False = the model as built -- the launcher then
            # gates the kernel's has_ced off and launches no l_B
            # field, so the state is bitwise unchanged).  True ADDS
            # Deardorff's second, grid-scale dissipation channel to
            # whichever base the S3-6k switch above selected, on the
            # state-independent reference length l_ref = delta**f *
            # l_B(z+z0)**(1-f) (gpuwm/verify/sase_ref.py module
            # docstring, S3-12 section; launch_sase_step docstring,
            # S3-12 seam).  One argument, one entry point, exactly the
            # S3-6k pattern: the coefficient is formed inside the
            # e-update kernel, and the one new device field (the
            # geometry-only Blackadar length) is launched by the step
            # itself.  Before this wire the switch was authority-side
            # only -- a GPU run that set it got the channel silently
            # DROPPED; now the device path carries it (parity pinned
            # in tests/test_sase_gpu.py, S3-12 section).
            additive_dissipation=bool(
                getattr(cfg, "sase_additive_dissipation", False)))
        # S3-6c/6e: the split step returns the K_v field its vertical
        # channel used and the governed horizontal diffusivity km_h;
        # the scalar loop below rides both (K_v/Pr_t(f) implicit
        # vertical solves; K_h = km_h/Pr_t(f) horizontal fluxes --
        # S3-6g blended Prandtl number).  Popped so the
        # retained ledger stays scalar-only (no device-field residency
        # across steps).  A ledger without kv/km_h (the CPU shim seam
        # substitutes a scalar-only dict) falls back to the seam's
        # legacy channels -- the contract is the dict itself.
        kv = ledger.pop("kv", None)
        km_h = ledger.pop("km_h", None)
        # S3-6g regime-consistent Prandtl number: every scalar channel
        # below divides by Pr_t(f_used), the SAME blend the step's
        # buoyancy K_h used (authority prandtl_blend; recomputed from
        # the retained ledger f so the CPU-shim seam's scalar-only
        # dict needs no new key -- identical FP64 arithmetic to the
        # step's exported pr_t diagnostic).
        pr_t = float(sase_prandtl_blend(float(ledger["f"])))
        # HORIZONTAL EDDY-VISCOSITY DIAGNOSTIC (cfg.hmix_k_diag,
        # output-only).  ``km_h`` is THE governed horizontal diffusivity
        # this closure claims replaces the km_opt operator -- the same
        # field the momentum stress, the subgrid-energy transport and the
        # scalar channel below all ride -- so recording it here records
        # the claim itself, in the units and on the grid the Smagorinsky
        # operator's XKMH is recorded in.  Copied into a driver
        # persistent because ``km_h`` is a step transient and output
        # reads it after the step ends.  READ-ONLY in km_h.  The scalar
        # row is km_h/Pr_t(f), which is what the scalar channel actually
        # applies -- not a second field but the same field over the
        # step's own blended Prandtl number, recorded so a reader never
        # has to reconstruct pr_t to interpret the momentum row.  The
        # CPU-shim seam (km_h is None) leaves an honest zero: it runs the
        # legacy coefficient form and has no governed field to record.
        if self.hmix_k_diag is not None and km_h is not None:
            self.hmix_k_diag["SASE_KMH"][...] = km_h
            self.hmix_k_diag["SASE_KHH"][...] = km_h
            self.hmix_k_diag["SASE_KHH"] *= DTYPE(1.0 / pr_t)
        # SASE-M2 conditional venting limb (S4-5; authority
        # plume_vent_flux, sase_ref module docstring SASE-M2 section;
        # design doc SASE-M section 4).  Diagnosed -- no new prognostic
        # state -- from the FROZEN PRE-STEP state: the pre-step
        # theta/qv/qc/pressure the slot has not written, the pre-step
        # ``e_pre`` copy taken above, the M1 substitution mask as the
        # bitwise n2_eff != n2 departure (the seam's own mask, never
        # re-derived -- and passed here whatever cfg.sase_moist_n2 says,
        # because this is a saturation VETO, not a stability value: M1
        # off must not silently stand M2 down), and the S3-11a rho1
        # already computed.  f is the
        # step's USED partition fraction, which is why this runs AFTER
        # the step and not beside the frozen copy: the FP-exact
        # two-product blend M_used = (1 - f)*M_base makes the LES limit
        # f = 1 a bitwise +0.0 deposit.  The three returned face
        # profiles carry F[0] = F[nz] = +0.0 exactly, so the S3-11a
        # boundary-consistent scalar ledger extends with a ZERO
        # net-column term (the algebra is at launch_vent_deposit).
        #
        # S4-5b Item 4c -- WHY THIS IS CONDITIONAL, AND WHY IT IS LOUD.
        # The deposit lands on the PRE-SOLVE state of the implicit
        # vertical channel (the registered deposit-then-solve order),
        # and that state exists only where the split step returned its
        # K_v field.  The legacy scalar-only seam -- ``kv is None``, the
        # CPU-shim contract two branches below, reachable by no
        # production configuration -- has no pre-solve state to deposit
        # onto, so the limb is not diagnosed there AT ALL.  Before this
        # fix the flux and cap-scale launches ran on that path and their
        # output was dropped on the floor with no guard and no comment.
        # The drop is now impossible to make silently: the assertion
        # after the scalar loop requires that every row diagnosed here
        # reached ``launch_vent_deposit``.
        vent_scale = None
        vent_rows: dict[str, cp.ndarray] = {}
        if kv is not None:
            vent = launch_plume_vent_flux(
                theta, atmosphere["qv"], atmosphere["qc"],
                atmosphere["pressure"], e_pre, n2_eff, n2,
                rho1, f_blend=float(ledger["f"]), dz_col=dzf)
            # Cap family (VENT_THETA_STEP_CAP / VENT_QT_STEP_CAP,
            # registered and CONTRACT-ONLY in the authority -- this seam
            # is where they are enforced): ONE per-column uniform
            # rescale of all three rows, computed before the scalar loop
            # because the factor couples them (authority
            # vent_deposit_rescale).  Uniform is load-bearing: a
            # per-level clip destroys the telescoping (measured
            # sum thick*dtheta = -3.74 against 0.0).
            vent_scale = launch_vent_deposit_scale(
                *vent, rho1, dt=dt, dz_col=dzf)
            vent_rows = {"dtheta": vent[0], "dqv": vent[1],
                         "dqc": vent[2]}
        # SPLIT SUBGRID-FLUX DIAGNOSTIC, M2 vent channel (output-only,
        # cfg.sase_flux_diag).  READ-ONLY in every model array and placed
        # AFTER the cap-scale launch on this line, so it cannot perturb
        # the state; the K_v half is filled inside the scalar loop below.
        # The CAP-SCALED product is what is recorded: ``vent`` is the
        # UNSCALED profile and launch_vent_deposit multiplies by
        # ``vent_scale`` in-kernel, so the raw profile is a flux the
        # model did not apply.  ``fac`` carries the units --
        # kg m-2 s-1 on the qv row, CP_AIR*[K kg m-2 s-1] = W m-2 on the
        # theta row -- and both stay POSITIVE UPWARD (the deposit's own
        # convention).  The kv-is-None legacy shim seam diagnoses NO
        # venting limb at all (the S4-5b Item 4c guard above), so its
        # honest record is +0.0, not the previous step's residue -- and
        # it runs no implicit vertical solve either, so the K_v half is
        # zeroed here too (the scalar loop's fill is unreachable there).
        flux_diag = self.sase_flux_diag
        if flux_diag is not None:
            if kv is None:
                for buffer in flux_diag.values():
                    buffer.fill(DTYPE(0.0))
            else:
                launch_vent_flux_diag(vent[1], vent_scale,
                                      flux_diag["fqv_vent"], fac=1.0)
                launch_vent_flux_diag(vent[0], vent_scale,
                                      flux_diag["fth_vent"],
                                      fac=float(SASE_CP_AIR))
        if boundary_width:
            # Registered adjudication (S3-6 review): hold the specified
            # domain's e_sgs at the realizability floor across the outer
            # spec_bdy_width rows every physics step -- see the method
            # docstring for the halo argument.
            floor = DTYPE(SASE_E_MIN)
            e_sgs = state.e_sgs
            e_sgs[:, :boundary_width, :] = floor
            e_sgs[:, -boundary_width:, :] = floor
            e_sgs[:, :, :boundary_width] = floor
            e_sgs[:, :, -boundary_width:] = floor
        inv_dt = DTYPE(1.0 / dt)
        for work, before in ((u_w, u_a), (v_w, v_a), (w_w, w_a)):
            work -= before
            work *= inv_dt
        du, dv, dw = u_w, v_w, w_w
        if km_h is None:
            # Legacy seam fallback (CPU shim path only): the v0
            # coefficient form on the post-step e, over the S3-6g
            # blended Prandtl number.
            kh_coef = ((ledger["f"] * ledger["c_nu"]
                        + (1.0 - ledger["f"]) * SASE_C_K)
                       * delta / pr_t)
        flux = [cp.empty((nz, ny, nx), dtype=DTYPE) for _ in range(2)]
        # S3-11b surface scalar-flux deposit (authority
        # surface_scalar_flux_deposit; registered SFC_SCALAR_FLUX =
        # "explicit-deposit-v1"; G-LAKE root cause
        # .superpowers/sdd/lake-momentum-root-cause.md): the theta and
        # qv rows of the implicit vertical solve carry the sfclay
        # HFX/QFX lowest-layer deposit -- the SAME fresh post-sfclay
        # f["hfx"]/f["qfx"] objects the step-3 e source consumed
        # (refreshed by _run_sfclay in this same due step, the
        # S3-6j/S3-9c freshness argument) at the SAME rho1 (computed
        # once above), fused in-kernel into the FP64 bottom rhs BEFORE
        # the sweep (the registered explicit-deposit-BEFORE-solve
        # order).  sfc_fac carries the authority row constant: CP_AIR
        # for theta, 1.0 (default) for qv.  qc/qi take NO deposit
        # (YSU's cloud/ice rows carry no surface source).  The seam is
        # OFF-able only through the fluxes themselves being zero
        # (in-kernel guard) -- no config flag: the physics is not
        # optional.  Until S3-11b these fluxes reached ONLY the e
        # source and Noah's ground budget -- the scalar channel ran
        # zero-flux at the ground (the FALSE premise recorded, and
        # struck, at the authority module docstring).
        sfc_deposit = {
            "dtheta": {"sfc_flux": f["hfx"], "sfc_rho1": rho1,
                       "sfc_fac": float(SASE_CP_AIR)},
            "dqv": {"sfc_flux": f["qfx"], "sfc_rho1": rho1},
        }
        mixed: dict[str, cp.ndarray] = {}
        vent_deposited = 0
        for name, field in (("dtheta", theta), ("dqv", atmosphere["qv"]),
                            ("dqc", atmosphere["qc"]),
                            ("dqi", atmosphere["qi"])):
            if km_h is not None:
                # S3-6e governed scalar channel: K_h = km_h/Pr_t(f),
                # the SAME governed horizontal diffusivity the momentum
                # stress and e-transport used (authority scalar_hmix)
                # over the S3-6g blended Prandtl number.
                tend = launch_scalar_mix(
                    field, kh_field=km_h, kh_fac=1.0 / pr_t,
                    dx=cfg.dx, dy=cfg.dy, flux=flux)
            else:
                tend = launch_scalar_mix(
                    field, state.e_sgs, kh_coef=kh_coef, dx=cfg.dx,
                    dy=cfg.dy, flux=flux)
            if kv is not None:
                # Split scalar channel, vertical half (S3-6c): advance
                # s* = s + dt*T_h in the tendency buffer, solve the
                # implicit K_v/Pr_t(f) vertical diffusion in place
                # (S3-6g blended Prandtl number), and return to rate
                # form (s_new - s)/dt.  S3-11b: the theta/qv rows
                # additionally carry the fused surface scalar-flux
                # deposit (sfc_deposit above) applied to the bottom
                # rhs before the sweep.  No floor: the M-matrix max
                # principle bounds the flux-free rows by their own
                # extrema, and the deposited bottom row by the
                # physical boundary flux (moisture cannot go negative
                # here -- a negative QFX can only drain the qv the
                # column actually holds, at the surface, as in YSU).
                tend *= DTYPE(dt)
                tend += field
                if name in vent_rows:
                    # SASE-M2 deposit seam (S4-5; spec C1-C3, binding):
                    # the EXPLICIT flux-form RHS deposit
                    # phi*[k] += (Fs[k] - Fs[k+1])*dt/(rho1*thick_k)
                    # lands on the pre-solve state HERE, BEFORE the
                    # implicit sweep -- the registered deposit-then-
                    # solve order generalizing SFC_SCALAR_FLUX =
                    # "explicit-deposit-v1" (the S3-11a seam two lines
                    # below rides the same order).  It is NOT inside
                    # sase_split_step (the ledger theorem reads theta
                    # read-only) and NOT a Thomas row: the solver's
                    # pinned max principle is exactly what non-local
                    # transport must be free to violate, and the Thomas
                    # kernels stay untouched.  qi takes no deposit
                    # (authority scope: theta/qv/qc only).
                    launch_vent_deposit(tend, vent_rows[name], vent_scale,
                                        rho1, dt=dt, dz_col=dzf)
                    vent_deposited += 1
                launch_implicit_vertical_diffusion(
                    tend, kv, dt=dt, kfac=1.0 / pr_t, dz_col=dzf,
                    **sfc_deposit.get(name, {}))
                # SPLIT SUBGRID-FLUX DIAGNOSTIC, K_v channel
                # (output-only).  EXACTLY HERE and nowhere else: the
                # solve holds its face coefficients in per-thread
                # registers and materializes no flux, so the flux is
                # recovered from the POST-SOLVE field -- which backward
                # Euler is the right state to evaluate the operator at --
                # and ``tend`` is that field for exactly the one line
                # between the solve returning and the conversion back to
                # rate form below.  READ-ONLY in ``tend``/kv/dzf/rho1 and
                # placed AFTER the state-affecting launch, so it cannot
                # perturb the state.  Same kfac the solve just used, so
                # the recorded flux is the solver's own; same rho1 the
                # vent channel above rides, so the two are summable.
                if flux_diag is not None and name in _SASE_FLUX_DIAG_ROWS:
                    key, fac = _SASE_FLUX_DIAG_ROWS[name]
                    launch_diff_flux_diag(
                        tend, kv, rho1, flux_diag[key],
                        kfac=1.0 / pr_t, fac=fac, dz_col=dzf)
                tend -= field
                tend *= inv_dt
            mixed[name] = tend
        # S4-5b Item 4c: every DIAGNOSED venting row must have been
        # deposited.  A future edit that reintroduces a branch where the
        # limb is computed and its deposit skipped fails here instead of
        # running an inert limb for a whole simulation.
        if vent_deposited != len(vent_rows):
            raise RuntimeError(
                f"SASE-M2 deposit dropped: {vent_deposited} of "
                f"{len(vent_rows)} diagnosed venting rows reached "
                f"launch_vent_deposit")
        # Dissipative-heat deposit (step 6): heat/(dt*cp*exner) K s-1.
        heat *= inv_dt / DTYPE(c.CP)
        heat /= atmosphere["exner"]
        mixed["dtheta"] += heat
        rates = {"du": du, "dv": dv, **mixed}
        try:
            # The forensic set is exactly the closure's LOWER BOUNDARY
            # CONDITION plus the two thermodynamic fields its surface
            # density reads -- the fields another component produced
            # this step and the closure only consumes.  e_sgs is
            # deliberately absent: the closure owns it, so its state is
            # not evidence about a producer.
            validate_sase_tendencies(
                {**rates, "dw": dw}, grid_id=cfg.grid_id,
                producer_inputs={
                    "ust": f["ust"], "wspd": f["wspd"], "hfx": f["hfx"],
                    "qfx": f["qfx"], "dz1": dzf[0],
                    "p1": atmosphere["p_interface"][0],
                    "t1": atmosphere["temperature"][0]})
        except FloatingPointError:
            self.sase_nan_guard_fires += 1
            raise
        pbl = self._couple_pbl_slot(cfg, rates, state=state)
        pbl.rw = couple_sase_w_tendency(state, cfg, dw)
        self.pbl_tendencies = pbl
        self.last_sase_ledger = ledger

    def _sase_output_pblh(self) -> cp.ndarray:
        """Per-column bulk-Richardson BL height for history output (S3-9d).

        OUTPUT-ONLY seam: with ``turb_scheme='sase'`` the YSU slot never
        runs, so ``fields['pblh']`` stays at its ``initialize_physics``
        constant -- the value sfclay's convective-velocity term was
        calibrated against for this configuration.  Rewriting that field
        with the live z_i would change the surface-layer physics, so the
        history PBLH is computed HERE instead, from the CURRENT state at
        frame-build time, with the split step's own per-column kernel
        (:func:`launch_bulk_richardson_zi`; its interior mean is the
        ledger's ``zi`` diagnostic) on inputs that mirror
        ``_prepare_atmosphere``'s destagger and layer thicknesses.
        """
        state = self.state
        theta = cp.ascontiguousarray(state.total_theta())
        u = cp.ascontiguousarray(0.5 * (state.u[:, :, :-1]
                                        + state.u[:, :, 1:]))
        v = cp.ascontiguousarray(0.5 * (state.v[:, :-1, :]
                                        + state.v[:, 1:, :]))
        phb = state.phb
        phb3 = phb[:, None, None] if phb.ndim == 1 else phb
        z_interface = (phb3 + state.php) / DTYPE(c.G)
        dz = cp.ascontiguousarray(z_interface[1:] - z_interface[:-1])
        return launch_bulk_richardson_zi(u, v, theta, dz_col=dz)

    def compute(self, state: DomainState,
                cfg: RunConfig) -> PhysicsTendencies:
        """Run due physics at time t and return the held RK3 tendencies."""
        if state is not self.state:
            raise ValueError("PhysicsDriver is attached to a different state")
        # Complete any exceptional in-flight expiry before constructing this
        # step.  Normal compute() calls finalize the mask immediately after
        # composing the prior/current RK target, so this is a cheap invariant
        # guard for direct users and restored synthetic states.
        if cfg.cu_physics and self._cu_expiry_pending:
            self.finish_step()
        now = float(state.elapsed_seconds)
        # ITIMESTEP counts from THIS DOMAIN's activation, not from the
        # experiment's start.  They differ only for a domain that
        # activates late, and for one of those the difference decides
        # whether its FIRST step runs radiation: WRF's mandatory
        # itimestep == 1 call is what seeds GLW/SWDOWN/GSW before the
        # land surface consumes them.  Read from the state because that
        # is where the tick authority publishes it
        # (gpuwm.core.state.refresh_model_time); absent, the domain
        # started with the run and the two readings are identical.
        epoch = float(getattr(state, "domain_start_offset", 0.0) or 0.0)
        itimestep = int(np.floor((now - epoch) / cfg.dt + 0.5)) + 1
        atmosphere = None

        # WRF solve_em ordering: radiation precedes surface/PBL physics.
        self.stepra = _physics_interval_steps(self.radt_minutes, cfg.dt)
        radiation_due = _radiation_step_due(
            itimestep, self.stepra, self.radt_minutes)
        if radiation_enabled(cfg) and radiation_due:
            atmosphere = _prepare_atmosphere(state)
            self._run_radiation(atmosphere, state, cfg)
            self.call_counts["radiation"] += 1

        # WRF fixed-dt surface/PBL cadence: the mandatory ITIMESTEP=1 call
        # is followed by calls where MOD(ITIMESTEP, STEPBL) == 0.
        if (self.surface_enabled
                and _surface_pbl_step_due(itimestep, self.stepbl, cfg.bldt)):
            if atmosphere is None:
                atmosphere = _prepare_atmosphere(state)
            # WRF surface_driver refreshes PSFC from P8W(kts) before its
            # surface-layer/LSM SELECT CASE blocks, including configurations
            # without Noah (module_surface_driver.F:1981-1997).
            self.fields["psfc"][...] = atmosphere["p_interface"][0]
            # Dispatch on the selector VALUE, never on its truthiness: an
            # unrouted value raises here instead of silently running Noah
            # for RUC/Noah-MP or YSU for MYNN.
            dispatch = resolve_physics_dispatch(cfg)
            sfclay_method = dispatch["sf_sfclay_physics"]
            if sfclay_method is not None:
                getattr(self, sfclay_method)(atmosphere, cfg)
                self.call_counts["sfclay"] += 1
            land_method = dispatch["sf_surface_physics"]
            if land_method is not None:
                # THE ANALYTIC COSZEN PROVIDER, on the radiation cadence a
                # radiation call would have used.  Radiation-free Noah-MP
                # has no other writer for this carrier, and a carrier
                # written once is a carrier that stops answering to the sun
                # after its first minute.
                # LEGACY-CHECKPOINT PRODUCER REFRESH.  A resume from a
                # pre-contract checkpoint knows the carrier VALUES and
                # nothing about their provenance, so the producers are
                # rerun once here, before anything consumes them.  Costs
                # one off-cadence radiation call on the first step after
                # such a resume and nothing thereafter.
                if self.carriers_need_producer_refresh:
                    if radiation_enabled(cfg):
                        self._run_radiation(atmosphere, state, cfg)
                        self.call_counts["radiation"] += 1
                    self.carriers_need_producer_refresh = False
                if (not radiation_enabled(cfg)
                        and "coszen" in self.fields
                        and self.noahmp_geometry is not None
                        and _radiation_step_due(
                            itimestep, self.stepra, self.radt_minutes)):
                    self._update_analytic_coszen(state, cfg)
                # THE CARRIER CONTRACT, at the one instant that is
                # downstream of every way a driver can be built: a config
                # load, a direct initialize_physics call, a restart, a DA
                # analysis.  The config-load guards are the friendly early
                # word; this is the authority.
                if self.carriers is None:
                    raise CarrierContractError(
                        "this PhysicsDriver was assembled without a "
                        "surface-radiation carrier contract, and "
                        f"sf_surface_physics={int(cfg.sf_surface_physics)} "
                        "is about to consume radiative carriers.  "
                        "initialize_physics builds one; a driver built "
                        "another way must pass carriers=CarrierContract(...) "
                        "and declare each carrier's source.  Skipping the "
                        "contract is not a way to run without one -- it is "
                        "the defect the contract closes."
                    )
                self.carriers.check_before_consumption(
                    sf_surface_physics=int(cfg.sf_surface_physics),
                    fields=self.fields, model_time=now,
                    radiation_interval_seconds=float(self.radt_seconds),
                    timestep_seconds=float(cfg.dt))
                getattr(self, land_method)(atmosphere, cfg, itimestep)
                if int(cfg.sf_surface_physics) in \
                        LAND_SURFACE_SFCDIAGS_SCHEMES:
                    self._refresh_surface_diagnostics(atmosphere)
                self.call_counts["noah"] += 1
            # SURFACE-MOISTURE LEDGER (opt-in, off by default).  Placed here
            # because this is the instant after the FINAL writer of Q2 has
            # run and before the PBL scheme can touch the column, which is
            # the only point where "who wrote Q2, from what" has a single
            # answer.  Attaching a ledger does not change a single physics
            # field; an unattached driver never reaches this branch.
            if self.surface_moisture_ledger is not None:
                self.surface_moisture_ledger.capture(
                    fields=self.fields, atmosphere=atmosphere,
                    model_time=now,
                    sf_sfclay_physics=int(cfg.sf_sfclay_physics),
                    sf_surface_physics=int(cfg.sf_surface_physics))
            pbl_method = dispatch["bl_pbl_physics"]
            if pbl_method is not None:
                getattr(self, pbl_method)(atmosphere, cfg)
                # "ysu" is the historical name of the PBL-slot counter and
                # every consumer reads it, so every PBL scheme increments
                # it.  SASE additionally carries its own key, which exists
                # only when SASE is active -- the shim gate that proves
                # YSU was not silently run in its place reads the pair.
                self.call_counts["ysu"] += 1
                if self.sase_active:
                    self.call_counts["sase"] += 1
            else:
                self.pbl_tendencies = PhysicsTendencies.zeros(state)

        # WRF cumulus follows the PBL call and its rates are held until the
        # next STEP* event, exactly like radiation.
        self.stepcu = _physics_interval_steps(self.cudt_minutes, cfg.dt)
        if (cfg.cu_physics
                and _cumulus_step_due(
                    itimestep, self.stepcu, self.cudt_minutes)):
            if atmosphere is None:
                atmosphere = _prepare_atmosphere(state)
            # WRF's cumulus driver early-returns on non-due steps
            # (module_cumulus_driver.F:830-864, RETURN at :863), so the
            # W0AVG running mean at the top of KF_eta_CPS
            # (module_cu_kfeta.F:232-250) advances exactly once per STEPCU
            # due event -- immediately before the per-column NCA skip test
            # at :410, so held columns still refresh their trigger memory.
            # (Controller re-adjudication 2026-07-16: the T1 every-step
            # hook cadence misread the driver; v4.6.1 source wins.)
            history_hook = getattr(
                self.cumulus_callable, "update_trigger_history", None)
            if history_hook is not None:
                history_hook(state=state, cfg=cfg)
                self.call_counts["cumulus_history"] += 1
            self._run_cumulus(atmosphere, state, cfg)
            self.call_counts["cumulus"] += 1

        self._compose_tendencies(cfg)
        # WRF advances the cumulus precipitation/NCA clock before its later
        # microphysics call (solve_em.F:3558-3571 vs :3689-3720).  Running it
        # after compose preserves this step's pre-expiry RK forcing, while
        # finish_step() inside the advance clears raw KF rates before
        # Morrison can consume them.
        if cfg.cu_physics:
            self._advance_cumulus_clock(state, cfg)
        return self.tendencies

    def set_forcing(self, **fields) -> None:
        """Update externally supplied surface/radiation forcing in place.

        ``gsw`` is in the allowed set and was not: RUC reads the NET
        shortwave rather than SWDOWN, so an offline-forced RUC run had no
        way to supply the one shortwave carrier its LSM actually consumes
        and integrated the allocation zeros instead.  It arrives with the
        same provenance discipline as its neighbours -- an externally
        supplied carrier is ``external_array``, which is a producer for
        the contract's purposes and a declared forcing in the receipt,
        never "radiation available".

        ``rainbl`` is not a radiative carrier and takes no provenance; it
        is here because this is the surface-forcing door.
        """
        allowed = {"swdown", "glw", "gsw", "rainbl"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown physics forcing field(s): {sorted(unknown)}")
        missing = {name for name in fields
                   if name != "rainbl" and name not in self.fields}
        if missing:
            # GSW exists only under RUC.  Setting a carrier this driver has
            # no buffer for is a configuration mistake, not a no-op.
            raise ValueError(
                f"this driver carries no {sorted(missing)} buffer; GSW "
                "exists only under sf_surface_physics=3 (RUC) and COSZEN "
                "only under 4 (Noah-MP), so forcing one into a driver that "
                "has neither would write nothing and read as success")
        if self.carriers is None and set(fields) != {"rainbl"}:
            # The same refusal the radiation seam and the consumption seam
            # carry, written for this door: an offline-forcing caller who
            # hands a hand-built driver a radiative field would otherwise
            # crash on an attribute error here, or worse, write the buffer
            # and leave its provenance nowhere -- which is the unlabelled
            # carrier this contract exists to make impossible.
            raise CarrierContractError(
                "set_forcing received a radiative carrier, but this "
                "PhysicsDriver was assembled without a surface-radiation "
                "carrier contract to record who supplied it.  "
                "initialize_physics builds one; a driver built another "
                "way must pass carriers=CarrierContract(...) before "
                "forcing radiative fields through this door.")
        shape = self.fields["tsk"].shape
        for name, value in fields.items():
            self.fields[name][...] = _as_2d(value, shape, name)
            if name != "rainbl":
                self.carriers.declare(
                    name, source=CARRIER_SOURCE_EXTERNAL_ARRAY)

    def _update_analytic_coszen(self, state: DomainState,
                                cfg: RunConfig) -> None:
        """Refresh COSZEN from solar geometry on the radiation cadence.

        WHY A PROVIDER RATHER THAN AN INITIALIZATION.  COSZEN is a
        radiation-driver carrier in WRF, so a run with radiation off has no
        writer for it.  gpuwm used to seed it ONCE in
        ``initialize_physics``, at the half-interval hour angle of a due
        radiation call, and then never touch it again -- so a 12-hour
        Noah-MP run computed its canopy radiation transfer against the sun
        angle of its first minute for the whole forecast.  A single
        initialization is not a provider; the sun moves.

        Runs on the SAME cadence a radiation call would have, from the same
        :func:`gpuwm.core.dudhia.wrf_solar_geometry` at the same
        half-interval offset, so the sequence of values a radiation-free
        run sees is the sequence a radiation-bearing run would have seen.
        """
        from gpuwm.core.dudhia import wrf_solar_geometry
        geometry = self.noahmp_geometry
        interval = float(self.radt_seconds)
        coszen, _ = wrf_solar_geometry(
            geometry.start_time, geometry.latitude_deg,
            geometry.longitude_deg,
            hour_offset_seconds=float(state.elapsed_seconds)
            + 0.5 * interval)
        self.fields["coszen"][...] = cp.asarray(coszen, dtype=DTYPE)
        self.carriers.declare("coszen",
                              source=CARRIER_SOURCE_ANALYTIC_GEOMETRY,
                              model_time=float(state.elapsed_seconds))

    def _zero_accumulator(self) -> cp.ndarray:
        """A fresh zero surface field, shaped like every other 2-D output.

        Fresh rather than cached on purpose.  A cached array would be a new
        ``PhysicsDriver`` attribute, and every attribute of this object is
        walked and classified by the restart manifest -- so a convenience
        buffer would become a restart-contract change.  These are written
        only at history cadence, and one 2-D allocation per published frame
        is not worth that.
        """
        return cp.zeros(self.fields["tsk"].shape, dtype=DTYPE)

    def output_fields(self) -> dict[str, cp.ndarray]:
        """WRF diagnostic name -> live FP32 device surface field.

        The six surface precipitation accumulators are emitted **always**,
        as zeros when the scheme that would fill them is not running.  That
        is WRF's contract, not a convenience: ``RAINC``, ``RAINSH``,
        ``RAINNC``, ``SNOWNC``, ``GRAUPELNC`` and ``HAILNC`` are core
        (``misc``) history rows with no package gate, so stock WRF allocates
        and writes all six in every run -- see
        ``gpuwm.io.wrf_output_schema.PRECIPITATION_OUTPUT_FIELDS`` for the
        rows and for the near neighbours that are excluded.

        gpuwm used to omit each of them whenever its producer was absent,
        which reads as "this run had no cumulus scheme" to a human and as
        "this file is broken" to a reader: every wrf-python/wrf-rust
        precipitation recipe computes ``RAINC + RAINNC`` unconditionally,
        because in WRF output both always exist.  Omitting ``RAINC`` under
        ``cu_physics=0`` therefore failed every downstream QPF product,
        which is exactly how this was found.

        ``RAINSH`` is always zero here, and that is a true statement rather
        than a placeholder: gpuwm implements no shallow-cumulus scheme, and
        zero is what WRF writes for ``shcu_physics=0``.
        """
        output = {name: self.fields[field]
                  for name, field in _OUTPUT_FIELDS.items()}
        if self.sase_active:
            # Each scheme supplies its own boundary-layer height.  YSU
            # refreshes fields['pblh'] on every due call; SASE has no such
            # field of its own, so it computes its per-column
            # bulk-Richardson z_i here without touching the surface-layer
            # feed (whose convective-velocity term was calibrated against
            # the initialize_physics constant).
            output["PBLH"] = self._sase_output_pblh()
            if self.sase_flux_diag is not None:
                # SPLIT SUBGRID-FLUX DIAGNOSTIC (cfg.sase_flux_diag).  The
                # buffers hold the flux of the single step that ended at
                # this instant -- SASE pins bldt == 0, so its seam runs
                # every model step and this is an INSTANTANEOUS flux, not
                # a history-interval mean.  Zeros at the t=0 frame,
                # before any SASE step has run.
                output.update(
                    {name: self.sase_flux_diag[key]
                     for name, key in _SASE_FLUX_DIAG_OUTPUT.items()})
        if self.hmix_k_diag is not None:
            if not self.sase_active:
                # The km_opt = 4 producer's own buffers live in the
                # dycore's persistent scratch (prepare_fixed_tendencies
                # fills smag_km/smag_kh once per model step from the
                # time-t fields, which is the state WRF's
                # module_first_rk_step_part2 evaluates them at).  Copied
                # HERE rather than aliased so the published frame keeps a
                # constant schema from frame 0, when no step has run yet
                # and the slots do not exist.  READ-ONLY in the scratch.
                km = self.state.existing_scratch("smag_km")
                kh = self.state.existing_scratch("smag_kh")
                if km is not None and kh is not None:
                    self.hmix_k_diag["XKMH"][...] = km
                    self.hmix_k_diag["XKHH"][...] = kh
            output.update(self.hmix_k_diag)
        if self.radiation_active:
            output.update(SWDOWN=self.fields["swdown"],
                          GLW=self.fields["glw"])
            # OLR rides the same "only while radiation is running" rule as
            # SWDOWN/GLW, and additionally only while the LONGWAVE half is
            # a scheme that computes a top-of-atmosphere flux.  WRF's own
            # OLR row is core (``misc``, so stock WRF writes it in every
            # run); gpuwm's radiation diagnostics are absent rather than
            # zero when nothing produced them, and this follows that.
            if self.olr is not None:
                output["OLR"] = self.olr
        output["RAINC"] = (self._zero_accumulator() if self.rainc is None
                           else self.rainc)
        output["RAINSH"] = self._zero_accumulator()
        microphysics = self.microphysics if self.mp_physics else None
        for name, attribute in (("RAINNC", "rainnc"), ("SNOWNC", "snownc"),
                                ("GRAUPELNC", "graupelnc"),
                                ("HAILNC", "hailnc")):
            value = (None if microphysics is None
                     else getattr(microphysics, attribute, None))
            output[name] = (self._zero_accumulator() if value is None
                            else value)
        return output


def _resolve_initial_glw(glw, *, ra_lw_physics: int, radiation_active: bool,
                         sf_surface_physics: int) -> tuple[object, str]:
    """Return ``(initial GLW, provenance)`` or refuse to invent one.

    Downward longwave has exactly three honest origins, and this function
    is where a run is made to name which one it has.

    * ``"scheme"`` -- a longwave scheme is attached (``ra_lw_physics >
      0``), so the buffer allocated here is scratch: the scheme
      overwrites the whole field on its first radiation call, exactly as
      WRF's radiation driver zeroes ``GLW`` at the top of every call
      (``phys/module_radiation_driver.F:1722``) before its
      ``lwrad_select`` fills it.  The fill value is
      :data:`DECLARED_CONSTANT_GLW_WM2` because that is the value 1.8.7
      allocated, and a healthy run must stay byte-for-byte what it was.
    * ``"declared"`` -- the CALLER typed a value or handed a field.  An
      idealised column that wants a fixed longwave says so here; a route
      with a source GLW passes the source's array here.
    * ``"unused"`` -- nothing reads the buffer and nothing publishes it:
      no land-surface scheme to consume it, and radiation entirely off so
      no ``GLW`` row reaches wrfout.

    Anything else is a run that will CONSUME or PUBLISH a downward
    longwave that no scheme computed and no caller declared, and it is
    refused rather than filled.

    WHAT WRF v4.6.1 DOES with the same selectors, because the standing
    rule is to implement the defined behaviour rather than reproduce a
    bug, and here WRF is defined in both branches:

    * ``ra_lw_physics = 0`` with ``ra_sw_physics > 0``: FATAL.
      ``radiation_driver`` returns early only when BOTH are zero
      (``:1068``), so a shortwave-only run reaches ``lwrad_select``
      (``:1839``), which has no ``CASE (0)`` -- it falls to ``CASE
      DEFAULT`` (``:2245``) and calls ``wrf_error_fatal('The longwave
      option does not exist: lw_physics = 0')``.  The reciprocal
      pairing is fatal too: ``swrad_select``'s ``CASE (0)`` at ``:2827``
      calls ``wrf_error_fatal`` for every ``lw_physics`` except
      Held-Suarez (``:2831-2835``), so WRF permits a single stream only
      for that one idealised case.  gpuwm refusing this pairing is
      WRF-CONFORMANT, not a divergence.
    * ``ra_lw_physics = 0`` with ``ra_sw_physics = 0``: the driver
      returns at ``:1068`` and ``GLW`` keeps its Registry-allocated
      0.0 W m-2, which every land-surface scheme then consumes as a real
      flux.  gpuwm does NOT copy that: zero downward longwave is not a
      physical atmosphere, it is an absent one, and a column run under it
      cools without bound.  gpuwm's divergence is to require the constant
      to be DECLARED -- documented here and named in the run receipt --
      instead of inheriting either WRF's 0.0 or 1.8.7's silent 300.0.
    """

    from gpuwm.physics_compat import downward_longwave_disposition

    if glw is not None:
        return glw, "declared"
    # ONE classification, shared with the config-load guard
    # (physics_compat.constant_longwave_refusal) and the receipt line
    # (runtime.downward_longwave_source), so this refusal can never be
    # wider or narrower than the door's.  With lw=0 the radiation slot is
    # active exactly when shortwave is, which is what radiation_active
    # says here.
    kind, consumer = downward_longwave_disposition(
        ra_lw_physics=int(ra_lw_physics),
        ra_sw_physics=1 if radiation_active else 0,
        sf_surface_physics=int(sf_surface_physics))
    if kind == "scheme":
        return DECLARED_CONSTANT_GLW_WM2, "scheme"
    if kind == "unused":
        return DECLARED_CONSTANT_GLW_WM2, "unused"
    if kind == "consumed":
        reads = (f"sf_surface_physics={int(sf_surface_physics)} "
                 f"({consumer}) reads GLW every surface step")
    else:
        reads = ("radiation is active, so the GLW row is written to every "
                 "wrfout frame")
    raise ValueError(
        "downward longwave (GLW) has no source: ra_lw_physics=0, so no "
        f"longwave scheme computes it, and {reads}. "
        "initialize_physics will not invent one. "
        "REMEDY, in preference order: (1) run a longwave scheme -- set "
        "ra_lw_physics=4 with ra_sw_physics=4 (RRTMG-class), which is "
        "what every nocturnally valid shipped profile does; or (2) if "
        "this is an idealised column that genuinely wants a fixed "
        "longwave, declare it by passing glw=<W m-2> explicitly "
        f"(gpuwm.core.physics.DECLARED_CONSTANT_GLW_WM2 = "
        f"{DECLARED_CONSTANT_GLW_WM2} is the historical idealised value); "
        "or (3) hand this call the source's own GLW field as glw=<array>. "
        "WHY: through 1.8.7 this call defaulted to glw=300.0 and "
        "gpuwm/core/dudhia.py -- shortwave only -- returns the array it "
        "was given, so the whole forecast ran on one frozen number that "
        "never responded to temperature, humidity or cloud.  Radiative "
        "equilibrium at 300 W m-2 is 269.7 K; a Gulf-coast October night "
        "is near 410 W m-2, or 291.6 K.  That deficit craters skin "
        "temperature, collapses surface saturation humidity with it, and "
        "drove 2 m dewpoints tens of degrees below the airmass in a real "
        "user report.  WRF v4.6.1 does not offer this configuration at "
        "all: with ra_sw_physics>0 its lwrad_select has no lw=0 case and "
        "calls wrf_error_fatal (phys/module_radiation_driver.F:2245).")


def initialize_physics(
        state: DomainState, cfg: RunConfig, *, landmask=1.0, tsk=300.0,
        soil_temperature=285.0, soil_moisture=0.30, liquid_moisture=None,
        ivgtyp=10, isltyp=6, vegfra=50.0, tmn=285.0, xice=0.0, snow=0.0,
        snow_depth=0.0, sst=None,
        swdown=None, glw=None, pblh=0.0, mavail=1.0,
        landuse=None, xland=None, xice_threshold=None,
        noah_params=None, radiation=None, cumulus=None,
        radiation_start_time=None, radiation_latitude=None,
        radiation_longitude=None,
        noahmp_start_time=None, noahmp_latitude=None,
        noahmp_longitude=None) -> PhysicsDriver:
    """Allocate and attach persistent physics state and scheme callables.

    An mp-only configuration also receives a driver: microphysics itself is
    post-RK3, but its precipitation accumulators and REFL_10CM handoff live
    on this object.

    Inputs accept scalars or arrays broadcastable to ``(ny,nx)``; soil
    inputs accept scalars, four-element profiles, or ``(4,ny,nx)`` arrays.
    ``landmask`` is WPS 1=land/0=water and is the authority for WRF XLAND
    (1=land/2=water).  Noah receives both masks and therefore skips water,
    while SFCLAY and YSU continue to compute their marine branches.  A
    :class:`gpuwm.core.landuse.LanduseInitialization` supplied through
    ``landuse`` takes precedence for the WRF cold-start mask/category,
    roughness, surface-optics, UST, PBLH, and MAVAIL fields.

    A radiation or cumulus callable is selected by its nonzero RunConfig
    scheme ID.  Legacy ``ra_physics=4`` resolves to RTE+RRTMGP and
    ``ra_physics=90`` to the analytic clear-sky adapter.  Split
    ``ra_lw_physics=0, ra_sw_physics=1`` resolves to Dudhia shortwave, and
    ``1, 1`` to WRF's classic RRTM longwave + Dudhia shortwave pair, which
    is a COMPOSITION of the two single-stream adapters rather than a
    coupled one (:class:`gpuwm.core.rrtm_lw.RRTMDudhiaRadiation`).
    Production adapters require
    their UTC start time and mass-grid latitude/longitude here.
    ``cu_physics=1`` resolves to the
    Kain-Fritsch adapter (:class:`gpuwm.core.kf.KainFritsch`).  A
    scheme callable receives keyword arguments ``atmosphere``, ``fields``,
    ``state``, and ``cfg`` and must return :class:`RadiationResult` or
    :class:`CumulusResult`, respectively.

    ``glw`` HAS NO DEFAULT -- see :func:`_resolve_initial_glw`, which is
    where the argument is turned into a value or the call is refused.
    With a longwave scheme attached the buffer is a transient: WRF
    ordering runs radiation before the surface layer and the LSM, and the
    first radiation call is due at ``itimestep = 1``, so the scheme's own
    flux is in the field before anything reads it.  With
    ``ra_lw_physics = 0`` it is NOT a transient -- Dudhia hands the field
    straight back and a fully radiation-free run has no adapter at all --
    so whatever is in the buffer IS the run's entire downward longwave,
    and this function will not pick that number on the caller's behalf.
    Pass ``glw=`` explicitly to declare a fixed sky (the historical
    idealised value is :data:`DECLARED_CONSTANT_GLW_WM2`) or to hand the
    call a source's own GLW field.  A real experiment that means it also
    declares it in the config, which is what
    :func:`gpuwm.physics_compat.constant_longwave_refusal` and
    :func:`gpuwm.physics_compat.radiation_off_land_surface_refusal` read
    at load -- two separate claims, two separate tokens.
    """
    if not physics_driver_required(cfg):
        raise ValueError("initialize_physics requires at least one enabled "
                         "physics scheme")
    ra_lw_physics, ra_sw_physics = radiation_scheme_ids(cfg)
    if ra_lw_physics not in (0, 1, 4, 90):
        raise ValueError("ra_lw_physics must be 0, 1, 4, or 90")
    if ra_sw_physics not in (0, 1, 4, 90):
        raise ValueError("ra_sw_physics must be 0, 1, 4, or 90")
    radiation_active = bool(ra_lw_physics or ra_sw_physics)
    if cfg.cu_physics not in (0, 1, 3):
        raise ValueError(
            "cu_physics must be 0 (off), 1 (Kain-Fritsch) or "
            "3 (Grell-Freitas)")
    if radiation_active and radiation is None:
        missing = [name for name, value in (
            ("radiation_start_time", radiation_start_time),
            ("radiation_latitude", radiation_latitude),
            ("radiation_longitude", radiation_longitude)) if value is None]
        if missing:
            raise ValueError(
                "radiation production adapter "
                f"(LW/SW={ra_lw_physics}/{ra_sw_physics}) requires "
                + ", ".join(missing))
        if (ra_lw_physics, ra_sw_physics) == (90, 90):
            from gpuwm.core.analytic_radiation import AnalyticClearSkyRadiation
            radiation = AnalyticClearSkyRadiation(
                radiation_start_time, radiation_latitude,
                radiation_longitude)
        elif (ra_lw_physics, ra_sw_physics) == (4, 4):
            from gpuwm.physics_compat import (
                RRTMG_VARIANT_LEGACY, rrtmg_variant)
            if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
                # Explicit legacy-RRTMG selection constructs the exact
                # WRF v4.6.1 port; its constructor fails closed if the
                # assets/kernels are unavailable.  Never substitute
                # RTE+RRTMGP.  Root ozone routing: a child domain must be
                # wired through runtime.prepare_child_case, which passes
                # an explicit adapter with the parent ozone provider.
                from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
                radiation = RRTMGLegacyRadiation(
                    radiation_start_time, radiation_latitude,
                    radiation_longitude,
                    p_top=getattr(state, "p_top", None),
                    o3input=cfg.o3input)
            else:
                from gpuwm.core.rrtmgp import RRTMGPRadiation
                radiation = RRTMGPRadiation(
                    radiation_start_time, radiation_latitude,
                    radiation_longitude)
        elif (ra_lw_physics, ra_sw_physics) == (0, 1):
            from gpuwm.core.dudhia import DudhiaShortwaveRadiation
            radiation = DudhiaShortwaveRadiation(
                radiation_start_time, radiation_latitude,
                radiation_longitude, swrad_scat=cfg.swrad_scat,
                icloud=cfg.icloud)
        elif (ra_lw_physics, ra_sw_physics) == (1, 1):
            # WRF's classic pair.  Unlike 4/4 and 90/90 this is not one
            # coupled adapter: RRTMDudhiaRadiation composes the RRTM
            # longwave adapter with the Dudhia shortwave adapter exactly
            # as module_radiation_driver.F dispatches lwrad_select and
            # swrad_select independently.  RRTM owns the GLW buffer.
            from gpuwm.core.rrtm_lw import RRTMDudhiaRadiation
            radiation = RRTMDudhiaRadiation(
                radiation_start_time, radiation_latitude,
                radiation_longitude,
                p_top=getattr(state, "p_top", None),
                icloud=cfg.icloud, swrad_scat=cfg.swrad_scat)
        else:
            raise ValueError(
                "unsupported built-in radiation pair "
                f"{ra_lw_physics}/{ra_sw_physics}")
    if not radiation_active and radiation is not None:
        raise ValueError("radiation callable requires an active LW or SW scheme")
    if cfg.cu_physics == 1 and cumulus is None:
        # cu_physics=1 resolves to the production Kain-Fritsch adapter,
        # mirroring the scheme-ID binding used for radiation above.
        from gpuwm.core.kf import KainFritsch
        cumulus = KainFritsch()
    elif cfg.cu_physics == 3 and cumulus is None:
        # cu_physics=3 resolves to the Grell-Freitas adapter around the
        # oracle-held gf.cu kernels.  The kernel runs the SHIPPED identity
        # (corrected k22 indexing); the WRF-faithful flag is reachable only
        # through the parity suites, never from a RunConfig.
        from gpuwm.core.gf import GrellFreitas
        cumulus = GrellFreitas()
    elif cfg.cu_physics and cumulus is None:
        raise ValueError("cu_physics requires a cumulus callable")
    if not cfg.cu_physics and cumulus is not None:
        raise ValueError("cumulus callable requires nonzero cu_physics")
    if cfg.sf_surface_physics and not cfg.sf_sfclay_physics:
        raise ValueError(
            "a land-surface model requires sf_sfclay_physics for its "
            "exchange coefficients (Noah: CHS/QGH/RIB)")
    if cfg.bl_pbl_physics and not cfg.sf_sfclay_physics:
        raise ValueError(
            "a PBL scheme requires sf_sfclay_physics surface coupling")
    if cfg.bl_pbl_physics and state.qv is None:
        raise ValueError("PBL physics requires a moist DomainState")
    if cfg.cu_physics and state.qv is None:
        raise ValueError("cumulus physics requires a moist DomainState")
    if cfg.bl_pbl_physics == 11 and not hasattr(state, "e_sgs"):
        # Belt-and-braces like SASE's below: state.py allocates e_sgs
        # under the same selector, so this only fires on a state built
        # with one config and attached with another.
        raise ValueError(
            "Shin-Hong requires a DomainState allocated with "
            "bl_pbl_physics=11 (its published SGS TKE rides state.e_sgs)")
    if cfg.bl_pbl_physics == SASE_PBL_SCHEME:
        # Belt-and-braces behind validate_run_config's admission gate: the
        # driver-level invariants the closure relies on are re-checked at
        # attachment, so a hand-built RunConfig cannot bypass them.
        if not hasattr(state, "e_sgs"):
            raise ValueError(
                "SASE requires a DomainState allocated with "
                f"bl_pbl_physics={SASE_PBL_SCHEME} (prognostic e_sgs)")
        if cfg.bldt != 0.0:
            raise ValueError(
                "SASE requires bldt=0: its w tendency rides the PBL "
                "stack as a plain attribute rebuilt every step rather "
                "than a serialized field, so a positive PBL cadence "
                "would carry it across steps unserialized")
        if cfg.km_opt != 0:
            raise ValueError(
                "SASE supplies the mixing the km_opt operator would "
                f"apply; km_opt must be 0, got {cfg.km_opt}")
    intervals = (cfg.radt, cfg.bldt, cfg.radt_minutes, cfg.cudt_minutes)
    if any(not np.isfinite(value) or value < 0.0 for value in intervals):
        raise ValueError("physics intervals must be finite and non-negative")

    # Downward longwave, before anything can consume it.  Named, never
    # defaulted: see _resolve_initial_glw for what each provenance means
    # and for what WRF v4.6.1 does with the same selectors.
    glw, glw_provenance = _resolve_initial_glw(
        glw, ra_lw_physics=int(ra_lw_physics),
        radiation_active=radiation_active,
        sf_surface_physics=int(cfg.sf_surface_physics))

    shape = state.mup.shape
    if landuse is not None:
        landmask = landuse.landmask
        ivgtyp = landuse.ivgtyp
        isltyp = landuse.isltyp
        pblh = landuse.pblh
        mavail = landuse.mavail
    land = _as_2d(landmask, shape, "landmask") >= DTYPE(0.5)
    # NATIVE-XLAND LAW: a caller that supplies ``xland`` supplies the
    # classification, and it wins VERBATIM -- never re-derived, never
    # reconciled against landmask.  The pinning case is MPAS category-15
    # fractional sea ice, which carries native xland=1 on columns whose
    # landmask is 0: derivation would classify them open water and run no
    # surface at all, while the native value routes them to the ice
    # surface (NOAHMP_GLACIER for VEGTYP=ISICE below the sea-ice
    # threshold, the sea-ice skip at or above it).  Derivation from
    # landmask (WPS 1=land/0=water -> WRF 1=land/2=water) remains ONLY as
    # the documented fallback when no xland is passed; a landuse
    # initialization's own xland sits between the two.  The provenance is
    # recorded in ``driver.surface_classification`` so a run can state
    # which source decided.
    if xland is not None:
        resolved_xland = cp.ascontiguousarray(_as_2d(xland, shape, "xland"))
        xland_source = "native"
    elif landuse is not None:
        resolved_xland = _as_2d(landuse.xland, shape, "xland")
        xland_source = "landuse"
    else:
        resolved_xland = cp.ascontiguousarray(cp.where(
            land, DTYPE(1.0), DTYPE(2.0)))
        xland_source = "derived-from-landmask"
    f: dict[str, cp.ndarray] = {
        "landmask": land.astype(DTYPE), "xland": resolved_xland,
        "tsk": _as_2d(tsk, shape, "tsk"),
        "pblh": _as_2d(pblh, shape, "pblh"),
        "mavail": _as_2d(mavail, shape, "mavail"),
        "lakemask": (_as_2d(landuse.lakemask, shape, "lakemask")
                     if landuse is not None else
                     cp.zeros(shape, dtype=DTYPE)),
        "ivgtyp": _as_2d(ivgtyp, shape, "ivgtyp", dtype=cp.int32),
        "isltyp": _as_2d(isltyp, shape, "isltyp", dtype=cp.int32),
        "vegfra": _as_2d(vegfra, shape, "vegfra"),
        "tmn": _as_2d(tmn, shape, "tmn"),
        "xice": _as_2d(xice, shape, "xice"),
        # ``swdown=None`` allocates the same zeros the 0.0 default
        # allocated, so every existing trajectory is byte-identical; what
        # changes is that the CONTRACT can now tell "nobody said anything"
        # from "the caller declared zero", which is the difference between
        # a hole and a configuration.
        "swdown": _as_2d(0.0 if swdown is None else swdown, shape,
                         "swdown"),
        "glw": _as_2d(glw, shape, "glw"),
        "snow": _as_2d(snow, shape, "snow"),
        "snowh": _as_2d(snow_depth, shape, "snow_depth"),
    }
    if int(cfg.sf_surface_physics) in (3, 4):
        for name in SURFACE_PRECIPITATION_FIELDS:
            f[name] = cp.zeros(shape, dtype=DTYPE)
    if int(cfg.sf_surface_physics) == 3:
        # GSW is updated only at radiation cadence and consumed by RUC
        # unchanged between those calls.
        f["gsw"] = cp.zeros(shape, dtype=DTYPE)
    if int(cfg.sf_surface_physics) == 4:
        # COSZEN has the same radiation cadence.  A radiation-free in-process
        # run receives a single correctly offset initialization below.
        f["coszen"] = cp.zeros(shape, dtype=DTYPE)
    n_soil = soil_layer_count(cfg)
    f["smois"] = _as_soil(soil_moisture, shape, "soil_moisture",
                          layers=n_soil)
    f["tslb"] = _as_soil(soil_temperature, shape, "soil_temperature",
                         layers=n_soil)
    f["sh2o"] = _as_soil(soil_moisture if liquid_moisture is None
                          else liquid_moisture, shape, "liquid_moisture",
                          layers=n_soil)
    f["smcrel"] = cp.zeros((n_soil, *shape), dtype=DTYPE)

    # Shared SFCLAY outputs / Noah inout fields are the same device arrays.
    sf_initial = {
        "znt": (_as_2d(landuse.znt, shape, "znt")
                if landuse is not None else
                cp.where(land, DTYPE(0.10), DTYPE(1.0e-4))),
        # module_physics_init.F cold-starts UST at 1.e-4 before SFCLAY.
        "ust": (_as_2d(landuse.ust, shape, "ust")
                if landuse is not None else
                cp.full(shape, DTYPE(1.0e-4), dtype=DTYPE)),
    }
    for name in SFCLAY_OUTPUTS:
        f[name] = cp.ascontiguousarray(sf_initial.get(
            name, cp.zeros(shape, dtype=DTYPE)))
    if int(cfg.sf_surface_physics) == 3:
        # module_physics_init.F:3126-3131 initializes TSK_SAVE from TSK.
        # The open-water component comes from SST when the ingest carries it;
        # otherwise the already repaired surface temperature is WRF's
        # no-SST fallback.
        f["tsk_save"] = cp.ascontiguousarray(f["tsk"].copy())
        f["tsk_sea"] = _as_2d(
            tsk if sst is None else sst, shape, "sst")
        for name in ("znt", "ust", "mol", "zol", "flhc", "flqc", "cpm",
                     "cqs2", "chs2", "chs", "qsfc", "qgh", "hfx", "qfx",
                     "lh"):
            f[f"{name}_sea"] = cp.ascontiguousarray(f[name].copy())
    # MYNN shares most of the legacy surface-driver arrays.  Allocate only
    # its selected scheme's additional persistent/inout diagnostics; carrying
    # these fields in every MM5/Noah run silently changes restart inventory
    # and resident-memory accounting for unrelated configurations.
    if cfg.sf_sfclay_physics == 5:
        for name in MYNN_SURFACE_OUTPUTS:
            if name not in f:
                if name == "ustm":
                    f[name] = cp.ascontiguousarray(f["ust"].copy())
                else:
                    f[name] = cp.zeros(shape, dtype=DTYPE)
        if int(cfg.sf_surface_physics) == 3:
            # MYNN_SEAICE_WRAPPER's automatic ``*_SEA`` arrays are retained
            # between the two surface calls and RUC's post-call reblend.
            # Keeping the full second result persistent makes this allocation
            # visible to preflight; only WRF's named wait fields survive the
            # LSM seam, while the rest are wrapper-local blend operands.
            for name in MYNN_SURFACE_OUTPUTS:
                carrier = f"{name}_sea"
                if carrier not in f:
                    f[carrier] = cp.ascontiguousarray(f[name].copy())
    elif int(cfg.sf_sfclay_physics) == MYJ_SFCLAY_SCHEME:
        # The Eta surface layer's OWN published set, allocated only for its
        # own selector for the reason the MYNN block above gives: carrying
        # these in an MM5 run would change that run's restart inventory and
        # its VRAM accounting for nothing.  Every one is a WRF Registry
        # field cold-started at zero except the three noted.
        for name in (*MYJ_SFCLAY_INOUT, *MYJ_SFCLAY_OUTPUTS,
                     "ustm", "wspd", "ch", "mixht"):
            if name not in f:
                f[name] = cp.zeros(shape, dtype=DTYPE)
        # Z0BASE is the BACKGROUND roughness the Zilitinkevich thermal-
        # roughness fix reads (module_sf_myjsfc.F:733).  WRF fills it from
        # the land-use table at initialization and never lets the surface
        # layer touch it; the working ZNT, which the sea branch overwrites
        # every iteration from u*, is a different field.  Seeding it from
        # the cold-start ZNT is the closest gpuwm gets without a Z0BASE
        # ingest, and it is exactly what WRF's own MYJSFCINIT leaves behind
        # on a restart (:1128-1161 only refills Z0 under NMM).
        f["z0base"] = cp.ascontiguousarray(f["znt"].copy())
        # module_sf_myjsfc.F:186-203 seeds USTAR=0.1 on step one; the
        # kernel applies it per column, so the allocation just has to be a
        # real number rather than the 1e-4 cold start the MM5 layers use.
        f["ustm"][...] = f["ust"]
    elif cfg.km_opt in (2, 3, 4) and cfg.bl_pbl_physics == 0:
        # WRF Registry state USTM is the friction velocity without SFCLAY's
        # convective-wind correction.  vertical_diffusion_2 consumes USTM,
        # not UST, when diff_opt=2 runs with the PBL off.  Keep it scoped to
        # that newly admitted path so established YSU inventories and
        # certified-profile bytes remain unchanged.  km_opt=3 joined the
        # predicate with the 3-D Smagorinsky port: without it the PBL-off
        # consumer predicate would silently evaluate False and MOST surface
        # fluxes would vanish under the new closure.
        f["ustm"] = cp.zeros(shape, dtype=DTYPE)
    result = SFClayResult(**{name: f[name] for name in SFCLAY_OUTPUTS})

    defaults = {
        "psfc": 1.0e5, "sfcprs": 9.9e4, "sfctmp": 300.0,
        "qv1": 0.0, "qgh": 0.0, "dz8w1": 50.0,
        "rainbl": 0.0, "sr": 0.0, "chs": 0.01, "rib": 0.0,
        "shdmin": 10.0, "shdmax": 90.0, "snoalb": 0.65,
        "embck": 0.95, "canwat": 0.0, "snowc": 0.0,
        "albedo": 0.20, "albbck": 0.20, "emiss": 0.95,
        "z0": 0.10, "snotime": 0.0, "lai": 2.0,
        "smstav": 0.0, "smstot": 0.0, "sfcrunoff": 0.0,
        "udrunoff": 0.0, "acsnow": 0.0, "acsnom": 0.0,
        "snopcx": 0.0, "potevp": 0.0, "noahres": 0.0,
        "reslin": 0.0, "chklowq": 0.0, "grdflx": 0.0,
    }
    if landuse is not None:
        # phys/module_physics_init.F:landuse_init initializes these once;
        # the first surface/radiation calls consume the same live arrays.
        for name in ("snowc", "z0", "albbck", "albedo", "embck",
                     "emiss"):
            f[name] = _as_2d(getattr(landuse, name), shape, name)
    # Keep user/static/overlapping SFCLAY arrays; fill every remaining Noah
    # launch field with a physically neutral contiguous FP32 array.
    for name in NOAH_FIELDS_2D:
        if name not in f:
            f[name] = _as_2d(defaults.get(name, 0.0), shape, name)
    for name in NOAH_FIELDS_3D:
        if name not in f:
            f[name] = cp.zeros((n_soil, *shape), dtype=DTYPE)
    f["ebal"] = cp.zeros(shape, dtype=cp.int32)
    f["kpbl"] = cp.zeros(shape, dtype=cp.int32)
    f["exch_h"] = cp.zeros((cfg.nz, *shape), dtype=DTYPE)
    f["exch_m"] = cp.zeros((cfg.nz, *shape), dtype=DTYPE)
    # MYNN's carried PBL state.  Allocated only for its own selector, for the
    # reason the surface-layer block above gives: ten extra 3-D arrays in a
    # YSU run would change that run's restart inventory and its VRAM budget,
    # and on this hardware the VRAM budget is a correctness bar.  WRF
    # cold-starts every one of them at zero (Registry defaults); the
    # ``initflag`` block inside the driver is what seeds them on step one.
    if int(cfg.bl_pbl_physics) == MYJ_PBL_SCHEME:
        # MYJ's two carried columns, for its selector only, on the same
        # terms as MYNN's block below.  TKE_MYJ is genuinely PROGNOSTIC --
        # it is read as 2*TKE at the top of every call and rewritten at the
        # bottom (module_bl_myjpbl.F:377, :462), and the surface layer's
        # PBLH scan reads it too -- so a restart that dropped it would
        # resume a different boundary layer.
        #
        # TKE_MYJ cold-starts at EPSQ2, NOT at zero.  MYJPBLINIT's
        # non-restart block is `TKE_MYJ(I,K,J)=EPSQ2` with epsq2=0.2
        # (module_bl_myjpbl.F:1725, share/module_model_constants.F:92,
        # commented "initial TKE"); only RUBLTEN/RVBLTEN/RTHBLTEN/RQVBLTEN
        # and EXCH_H are zeroed there.  Zero would be WRONG, not merely
        # different, and the mechanism is MIXLEN's LPBL scan: q2 = 2*TKE
        # and the scan threshold is EPSQ2*FH = 0.202, so WRF's seed
        # (q2 = 0.4) falls through to LPBL=1 and the Blackadar EL0 branch
        # runs the whole column, while a zero seed trips the scan at the
        # first level tested and pins LPBL = LMH-1.  On a 30-level 100 m
        # column that is pblh 2900 m against 100 m on step one, and the
        # step-1 PBLH feeds the surface layer too (BTGH -> WSTAR2 -> u*),
        # so the divergence is not confined to a diagnostic.  A zero
        # column additionally makes MIXLEN's SQ (a sum of sqrt(q2))
        # exactly zero, so EL0=MIN(ALPH*SZQ*0.5/SQ,EL0MAX) is a 0/0 WRF
        # never reaches.  Pinned in tests/test_myj_port.py against the
        # authority's own EPSQ2, which is a literal here for the reason
        # state.py's Shin-Hong cold start gives: core must not import the
        # verification tree.
        #
        # EL_MYJ genuinely is zero at cold start, but not because
        # MYJPBLINIT says so -- EL_MYJ is not one of its arguments at all
        # (module_bl_myjpbl.F:1694-1697).  It is INTENT(OUT) on MYJPBL and
        # rewritten from `EL_MYJ(its:ite,:,jts:jte) = 0.` at the top of
        # every call (:341), so its cold-start value is unobservable; zero
        # is both WRF's Registry default and what the first call writes.
        f["tke_myj"] = cp.full((cfg.nz, *shape), MYJ_TKE_COLD_START,
                               dtype=DTYPE)
        f["el_myj"] = cp.zeros((cfg.nz, *shape), dtype=DTYPE)
    if int(cfg.bl_pbl_physics) == 5:
        for name in MYNN_PBL_STATE_3D:
            f[name] = cp.zeros((cfg.nz, *shape), dtype=DTYPE)
        for name in MYNN_PBL_DIAGNOSTICS_2D:
            f[name] = cp.zeros(shape, dtype=DTYPE)
        for name in MYNN_PBL_DIAGNOSTICS_INT_2D:
            f[name] = cp.zeros(shape, dtype=cp.int32)
    # Noah-MP's carried state and published diagnostics, for its selector
    # only, for the same reason: 46 extra 2-D arrays and four snow-stack
    # arrays in a Noah run would change that run's restart inventory, its
    # VRAM budget and its health-descriptor count.  Every one is allocated at
    # zero here and then overwritten by NOAHMP_INIT/SNOW_INIT below -- zero
    # is the Registry cold state but it is not a runnable Noah-MP state.
    # RUC's carried state and its four published driver locals, for its
    # selector only, on the same terms.  Every one is allocated at ZERO, which
    # is deliberate and is WRF's Registry cold state -- and, for six of them,
    # is also the value LSMRUC's own ktau==1 block (:481-565) tests for and
    # repairs: SOILT1 outside 170..400 K is rebuilt from SOILT/TSO(1), QSG is
    # rebuilt from qsn(SOILT), QCG outside 0..0.1 takes QC3D, QVG outside
    # 0..0.1 takes QSG*MAVAIL, RHOSNF is seeded to -1e3 and CHKLOWQ to 1.  So
    # the repair lives in one place, in the driver, rather than being
    # duplicated as an allocation default here.
    if int(cfg.sf_surface_physics) == 3:
        for name in (*RUC_STATE_2D, *RUC_DIAGNOSTICS_2D):
            if name not in f:
                f[name] = cp.zeros(shape, dtype=DTYPE)
        for name in RUC_STATE_3D:
            f[name] = cp.zeros((n_soil, *shape), dtype=DTYPE)
    if int(cfg.sf_surface_physics) == 4:
        for name in (*NOAHMP_STATE_2D, *NOAHMP_DIAGNOSTICS_2D):
            f[name] = cp.zeros(shape, dtype=DTYPE)
        for name in NOAHMP_STATE_INT_2D:
            f[name] = cp.zeros(shape, dtype=cp.int32)
        for name in NOAHMP_STATE_SNOW_3D:
            f[name] = cp.zeros((NOAHMP_NSNOW, *shape), dtype=DTYPE)
        for name in NOAHMP_STATE_SNOWSOIL_3D:
            f[name] = cp.zeros((NOAHMP_NSNOW + n_soil, *shape), dtype=DTYPE)

    # Noah's parameter tables belong to sf_surface_physics=2 ONLY.  Loading
    # them for any nonzero LSM would hand a RUC/Noah-MP driver Noah's
    # VEGPARM/SOILPARM/GENPARM bundle and make wrfout's live-surface gate
    # (which keys on noah_params) claim a Noah state that never ran.
    if noah_params is None and int(cfg.sf_surface_physics) == 2:
        noah_params = pack_params(load_tables())
    if noah_params is not None and int(cfg.sf_surface_physics) != 2:
        raise ValueError(
            "noah_params is the sf_surface_physics=2 parameter bundle and "
            f"cannot be attached to sf_surface_physics="
            f"{cfg.sf_surface_physics}")

    # Noah-MP's own tables and solar geometry, for its selector ONLY.  The
    # geometry defaults to the radiation callable's arrays because WRF's
    # COSZEN reaches the surface driver from the radiation driver; it is
    # accepted separately so a radiation-free Noah-MP run is possible without
    # inventing a latitude.
    # RUC's own tables, for its selector ONLY.  RUC reads the RUC sections of
    # the same three files Noah reads (VEGPARM's MODI-RUC/USGS-RUC blocks and
    # SOILPARM's STAS-RUC block), which is why gpuwm/io/restart.py can bind
    # its parameter identity to the same packaged asset roles; it is a
    # different bundle object over the same bytes, not Noah's bundle reused.
    ruc_params = None
    if int(cfg.sf_surface_physics) == 3:
        ruc_params = RucRuntimeParameters(
            seaice_albedo_default=cfg.seaice_albedo_default)

    if xice_threshold is not None and int(cfg.sf_surface_physics) != 4:
        raise ValueError(
            "xice_threshold is the Noah-MP sea-ice classification "
            "threshold (sf_surface_physics=4); RUC pins its own in "
            f"gpuwm/core/ruc_runtime.py.  Got sf_surface_physics="
            f"{cfg.sf_surface_physics}")
    noahmp_params = None
    noahmp_geometry = None
    if int(cfg.sf_surface_physics) == 4:
        noahmp_params = (NoahmpRuntimeParameters() if xice_threshold is None
                         else NoahmpRuntimeParameters(
                             xice_threshold=float(xice_threshold)))
        latitude = (noahmp_latitude if noahmp_latitude is not None
                    else radiation_latitude)
        longitude = (noahmp_longitude if noahmp_longitude is not None
                     else radiation_longitude)
        start_time = (noahmp_start_time if noahmp_start_time is not None
                      else radiation_start_time)
        if latitude is None or longitude is None or start_time is None:
            raise ValueError(
                "sf_surface_physics=4 needs noahmp_latitude, "
                "noahmp_longitude and noahmp_start_time (or the radiation_* "
                "equivalents): Noah-MP reads COSZ, XLAT, JULIAN and YR, and "
                "a silent zero latitude at day zero would run every column "
                "on the equator at New Year")
        noahmp_geometry = NoahmpSolarGeometry(
            start_time,
            cp.asnumpy(cp.asarray(latitude)) if hasattr(
                latitude, "__cuda_array_interface__") else latitude,
            cp.asnumpy(cp.asarray(longitude)) if hasattr(
                longitude, "__cuda_array_interface__") else longitude)
        guard_noahmp_glacier_columns(f, noahmp_params)
        if not radiation_active:
            # WRF's COSZEN is a radiation-driver carrier.  With radiation
            # disabled there is no future writer in WRF's own design, so
            # gpuwm supplies one: the seed below is the t=0 value of the
            # analytic provider that PhysicsDriver._update_analytic_coszen
            # then refreshes on the radiation cadence for the rest of the
            # run.  Through 1.8.9 this WAS the whole story -- one value,
            # never updated -- so a multi-hour Noah-MP run computed its
            # canopy radiative transfer against its first minute's sun.
            from gpuwm.core.dudhia import wrf_solar_geometry
            interval = _physics_interval_seconds(
                cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes,
                _model_clock_dt(cfg))
            coszen, _ = wrf_solar_geometry(
                noahmp_geometry.start_time,
                noahmp_geometry.latitude_deg,
                noahmp_geometry.longitude_deg,
                hour_offset_seconds=0.5 * interval)
            f["coszen"][...] = cp.asarray(coszen, dtype=DTYPE)

    carriers = CarrierContract(
        getattr(cfg, "surface_radiation_policy",
                SURFACE_RADIATION_POLICY_REQUIRED))
    # SEED THE CONTRACT with each carrier's TRUE origin at t=0.
    #
    # GLW: _resolve_initial_glw's three provenances map onto the contract
    # without loss, and the mapping is where the old defect dies.  "scheme"
    # means the buffer is SCRATCH -- the fill is DECLARED_CONSTANT_GLW_WM2
    # only because that is the number 1.8.7 allocated and a healthy run
    # must stay byte-for-byte what it was -- so its source is UNWRITTEN
    # until a radiation call actually writes it.  That is the whole of
    # "the constructor's 300.0 seed can never be silently consumed": if a
    # surface call arrives before any radiation call has run, the contract
    # refuses instead of integrating the allocation fill.  "unused" is
    # unwritten for the same reason and consumes nothing.  "declared" is
    # the caller's own number or the caller's own field, and the two are
    # different sources because a receipt that cannot tell a typed
    # constant from a source dataset's GLW is not a receipt.
    if glw_provenance == "declared":
        carriers.declare(
            "glw",
            source=(CARRIER_SOURCE_EXTERNAL_ARRAY
                    if hasattr(glw, "shape") and getattr(glw, "ndim", 0)
                    else CARRIER_SOURCE_DECLARED_CONSTANT))
    else:
        carriers.declare("glw", source=CARRIER_SOURCE_UNWRITTEN)
    # SWDOWN/GSW/COSZEN all start unwritten.  swdown's argument default is
    # None precisely so that "the caller said nothing" and "the caller
    # typed a number" are different states; a caller-supplied value is a
    # declaration and is labelled as one.
    carriers.declare(
        "swdown",
        source=(CARRIER_SOURCE_UNWRITTEN if swdown is None else
                CARRIER_SOURCE_EXTERNAL_ARRAY
                if hasattr(swdown, "shape") and getattr(swdown, "ndim", 0)
                else CARRIER_SOURCE_DECLARED_CONSTANT))
    if "gsw" in f:
        carriers.declare("gsw", source=CARRIER_SOURCE_UNWRITTEN)
    if "coszen" in f:
        carriers.declare(
            "coszen",
            source=(CARRIER_SOURCE_ANALYTIC_GEOMETRY if not radiation_active
                    else CARRIER_SOURCE_UNWRITTEN),
            model_time=(0.0 if not radiation_active else None))

    driver = PhysicsDriver(state, cfg, f, result, noah_params,
                           radiation=radiation, cumulus=cumulus,
                           noahmp_params=noahmp_params,
                           noahmp_geometry=noahmp_geometry,
                           ruc_params=ruc_params,
                           glw_provenance=glw_provenance,
                           carriers=carriers)
    # CLASSIFICATION RECEIPT: which source decided XLAND and how many
    # columns each surface class took under the active identity.  Rebuilt
    # at every construction from the same inputs (restart classification:
    # rebuild), and published so a run can answer "who decided these were
    # water" with a name instead of a diff.
    surface_classification = {
        "xland_source": xland_source,
        "xland_land_columns": int((resolved_xland
                                   < DTYPE(1.5)).sum()),
        "xland_water_columns": int((resolved_xland
                                    >= DTYPE(1.5)).sum()),
    }
    if noahmp_params is not None:
        masks = classify_noahmp_surface(
            f["xland"], f["xice"], f["ivgtyp"],
            xice_threshold=noahmp_params.xice_threshold,
            isice=int(noahmp_params.land_use.isice))
        surface_classification.update({
            "xice_threshold": float(noahmp_params.xice_threshold),
            "sea_ice_columns": int(masks["sea_ice"].sum()),
            "open_water_columns": int(masks["open_water"].sum()),
            "sflx_land_columns": int(masks["sflx_land"].sum()),
            "glacier_columns": int(masks["glacier"].sum()),
        })
    driver.surface_classification = surface_classification
    if int(cfg.sf_surface_physics) == 3:
        # ruclsminit runs once, before the first step, where
        # module_physics_init.F runs it.  Without it SH2O is whatever the
        # caller supplied for SMOIS and SMFR3D is zero everywhere, which
        # asserts a frozen-soil content of exactly none on a frozen column.
        ruc_cold_start(f, params=ruc_params)
    if int(cfg.sf_surface_physics) == 4:
        # NOAHMP_INIT/SNOW_INIT run once, before the first step, exactly
        # where module_physics_init.F runs them.  Without this every carrier
        # is at the zeros allocated above, and TV = TG = 0 K is not a cold
        # state -- it is a state whose saturation vapour pressure is
        # negative.
        noahmp_cold_start(f, params=noahmp_params,
                          dzs=NOAH_LAYER_THICKNESS_M)
    state.physics = driver
    # WRF's mp_init, ONCE per domain, exactly where phy_init runs it:
    # module_physics_init.F:1635 calls mp_init as the last physics
    # initializer before fg_init, and mp_init's CASE (THOMPSONAERO) arm
    # (:4522-4538) calls thompson_init -- which installs the synthetic CCN/IN
    # profile whenever the water/ice-friendly aerosol fields arrive unset
    # (module_mp_thompson.F:493-514 for CCN, :531-551 for IN).
    #
    # UNCONDITIONAL by design.  microphysics_init returns an empty receipt
    # for every scheme except 28 (microphysics.py:766-771), so there is no
    # selector test to keep in sync here.
    #
    # ONCE, NOT PER STEP.  Nothing in mp_gt_driver ever refills the profile,
    # and a per-step call would overwrite an advected, activated and
    # scavenged aerosol field with the synthetic one on every step while
    # leaving every clamp and every bound intact -- silent, plausible, and
    # worse than the defect it replaces.
    #
    # PRESENCE-GATED.  thompson_init reaches its two fills only through two
    # independent domain-wide MAXVAL presence tests (:490/:493 for CCN,
    # :528/:531 for IN), which gpuwm honours through
    # thompson_aerosol_state.aerosol_profile_needs_fill.  WRF itself calls
    # thompson_init on restart as well as on a cold start
    # (module_physics_init.F:4525: start_of_simulation .or. restart .or.
    # cycling), so in WRF that presence test -- not the call site -- is what
    # protects an aerosol-bearing domain.
    #
    # WHERE THAT MATTERS IN GPUWM, in call order:
    #   * a NEST: gpuwm/ingest/nest_init.py interpolates nwfa/nifa/nwfa2d
    #     from the parent (its ("nwfa", "") .. ("nifa2d", "") rows) and
    #     gpuwm/core/model.py runs initialize_child BEFORE
    #     runtime.prepare_child_case calls this function, so a child under an
    #     aerosol-bearing parent takes thompson_init's has_CCN branch
    #     (:516-522) exactly as WRF's would -- no refill, and no re-derived
    #     nwfa2d;
    #   * a RESUME: gpuwm's order is prepare -> initialize_physics ->
    #     restore_restart, so the fill DOES run here on the all-zero cold
    #     state and restore_restart then overwrites it with the checkpointed
    #     aerosol.  The resumed run integrates the checkpoint, which is the
    #     property that matters; the receipt records what this call did, not
    #     what the run ends up with.  Both are gated in tests/test_physics.py.
    #
    # An empty receipt is simply not recorded; the property's default is {}.
    receipt = microphysics_cold_start(state, cfg)
    if receipt:
        _MICROPHYSICS_INIT_RECEIPTS[driver] = receipt
    # Carriers must exist before anyone takes an inventory: see
    # gpuwm.core.kf.KainFritsch.ensure_trigger_history.  A cumulus
    # callable without the hook (a synthetic or direct-driver stand-in)
    # is left alone.
    _ensure_history = getattr(
        driver.cumulus_callable, "ensure_trigger_history", None)
    if _ensure_history is not None:
        _ensure_history(state)

    return driver


def __getattr__(name: str):
    """PEP 562 lazy export of the MPAS column-batch seam factory.

    ``gpuwm.core.physics.run_mpas_column_batch`` is the published API name
    for the persistent MPAS physics seam; the implementation lives in
    :mod:`gpuwm.core.mpas_column_batch`, which imports THIS module for the
    driver/coupling machinery.  A lazy attribute breaks that cycle without
    a wrapper function, so the exported name IS the implementation's own
    function object (the seam's orchestration-identity tests rely on
    object identity, never on spelling).
    """
    if name == "run_mpas_column_batch":
        from gpuwm.core.mpas_column_batch import run_mpas_column_batch
        return run_mpas_column_batch
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CumulusResult", "DECLARED_CONSTANT_GLW_WM2",
           "LAND_SURFACE_SFCDIAGS_SCHEMES",
           "PHYSICS_SLOT_DISPATCH", "PhysicsDriver", "PhysicsTendencies",
           "RadiationResult", "UnroutedPhysicsSelectorError",
           "couple_column_tendencies",
           "couple_ysu_tendencies", "initialize_physics",
           "microphysics_cold_start",
           "microphysics_scheme_sr_available",
           "microphysics_scratch_slots", "physics_enabled",
           "physics_driver_required",
           "physics_retains_ysu_output", "physics_reuses_pbl_composition",
           "resolve_physics_dispatch", "resolve_physics_slot",
           "run_mpas_column_batch",
           "validate_ysu_tendencies"]
