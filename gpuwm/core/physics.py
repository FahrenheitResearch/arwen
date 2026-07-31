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
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

import cupy as cp
import numpy as np

from gpuwm.config import (NOAHMP_OPTION_IDENTITY, RUC_OPTION_IDENTITY,
                          RunConfig,
                          radiation_enabled, radiation_scheme_ids,
                          soil_layer_count)
from gpuwm.core import constants as c
from gpuwm.core.noah import (_F2D as NOAH_FIELDS_2D,
                             _F3D as NOAH_FIELDS_3D,
                             launch_noah, load_tables, pack_params)
from gpuwm.core.microphysics import MicrophysicsDiagnostics
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
from gpuwm.core.sfclay import (SFCLAY_OUTPUTS, SFClayResult,
                               launch_sfclay, sfclay)
from gpuwm.core.state import DTYPE, DomainState
from gpuwm.core.ysu import launch_ysu
from gpuwm.ingest.soil import NOAH_LAYER_THICKNESS_M


_WSM6_MINOR_DT_SECONDS = np.float32(120.0)
_FP32_SIGNIFICAND_SCALE = 1 << 24
_FP32_ONE_BITS = 0x3F800000


def _wsm6_sr_roundoff_limit(dt: float) -> tuple[np.float32, int, int]:
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
    numerator = (scale + 1) ** 3 * (scale - 3)
    denominator = (scale ** 2 * (scale - 6)
                   * (scale - 2 * accumulation_adds))
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
        5: "_run_mynn_pbl",   # MYNN EDMF
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


def physics_enabled(cfg: RunConfig) -> bool:
    """Whether any non-timesplit physics scheme is configured."""
    return bool(radiation_enabled(cfg) or cfg.sf_sfclay_physics
                or cfg.sf_surface_physics or cfg.bl_pbl_physics
                or cfg.cu_physics)


def physics_driver_required(cfg: RunConfig) -> bool:
    """Whether setup must attach a persistent :class:`PhysicsDriver`.

    Microphysics is advanced after RK3 rather than through the non-timesplit
    tendency path selected by :func:`physics_enabled`.  It still needs the
    driver for accumulated precipitation and the output-due REFL_10CM
    handoff, so an mp-only domain must receive the same persistent attachment
    as a domain with radiation, surface, PBL, or cumulus physics.
    """
    return bool(cfg.mp_physics or physics_enabled(cfg))


def physics_retains_ysu_output(cfg: RunConfig) -> bool:
    """Whether the raw YSU output dict crosses a model-step boundary.

    Positive ``bldt`` keeps the historical diagnostic object untouched.
    At ``bldt == 0`` every configured PBL call is immediately consumed by
    :meth:`PhysicsDriver._run_ysu`, so retaining the raw rates duplicates the
    coupled PBL tendencies without serving a later reader.
    """
    return bool(cfg.bl_pbl_physics and cfg.bldt > 0.0)


def physics_reuses_pbl_composition(cfg: RunConfig) -> bool:
    """Whether the composed tendency target can be the fresh PBL stack.

    This is deliberately narrower than ``stepbl == 1``: every positive-bldt
    configuration retains the historical allocation path.  With active YSU
    and literal ``bldt == 0``, ``_run_ysu`` replaces the PBL stack before
    every composition, so radiation/cumulus can be accumulated into it once
    without corrupting a value needed by the next step.
    """
    return bool(cfg.bl_pbl_physics and cfg.bldt == 0.0
                and (radiation_enabled(cfg) or cfg.cu_physics))


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
    """Canonical optional YSU categories for the configured moist state."""
    return (("rqi",)
            if cfg.bl_pbl_physics and cfg.mp_physics in (6, 8, 10, 18)
            else ())


def _composed_optional_tendency_components(
        cfg: RunConfig) -> tuple[str, ...]:
    """Union of optional moisture categories entering the RK target."""
    members = (set(_pbl_optional_tendency_components(cfg))
               | set(_cumulus_optional_tendency_components(cfg)))
    return tuple(name for name in ("rqr", "rqi", "rqs") if name in members)


def microphysics_scratch_slots(
        mp_physics: int) -> tuple[tuple[str, str], ...]:
    """Driver diagnostic component -> canonical persistent scratch slot."""
    if mp_physics == 1:
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_kessler_sr"))
    if mp_physics in (6, 8, 10):
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"),
                ("graupelnc", "mp_graupelnc"),
                ("graupelncv", "mp_graupelncv"))
    if mp_physics == 18:
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"),
                ("graupelnc", "mp_graupelnc"),
                ("graupelncv", "mp_graupelncv"),
                ("hailnc", "mp_hailnc"),
                ("hailncv", "mp_hailncv"),
                ("sr", "mp_sr"))
    return ()


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
    """Validate one scheme output and return a contiguous FP32 device view."""
    array = cp.asarray(value, dtype=DTYPE)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not bool(cp.isfinite(array).all()):
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


def validate_ysu_tendencies(ysu: Mapping[str, cp.ndarray]) -> None:
    """Reject non-finite YSU output without modifying finite tendencies."""
    for name, value in ysu.items():
        if not bool(cp.isfinite(value).all()):
            raise FloatingPointError(f"YSU returned non-finite {name} tendency")


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
        p_interface[k] = p_interface[k + 1] - layer[k]
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


class PhysicsDriver:
    """Persistent surface state, diagnostics, scheduler, and held tendencies."""

    def __init__(self, state: DomainState, cfg: RunConfig,
                 fields: dict[str, cp.ndarray], sfclay_result: SFClayResult,
                 noah_params, radiation=None, cumulus=None,
                 noahmp_params=None, noahmp_geometry=None,
                 ruc_params=None):
        self.state = state
        self.fields = fields
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
        if self.mp_physics == 6:
            (self._sr_roundoff_upper, self._sr_roundoff_max_ulps,
             self._wsm6_minor_loops) = _wsm6_sr_roundoff_limit(cfg.dt)
        else:
            self._sr_roundoff_upper = np.float32(1.0)
            self._sr_roundoff_max_ulps = 0
            self._wsm6_minor_loops = 0
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
            from gpuwm.core.nssl2_default_hooks import (
                make_nssl2_production_binding,
            )
            self.nssl2_binding = make_nssl2_production_binding(
                state, cfg.dt)
        else:
            self.nssl2_binding = None

    def accept_microphysics(self, result: MicrophysicsDiagnostics) -> None:
        """Capture one post-RK microphysics result for the next surface call."""
        if not isinstance(result, MicrophysicsDiagnostics):
            raise TypeError("microphysics must return MicrophysicsDiagnostics")
        shape = self.state.mup.shape

        slots = dict(microphysics_scratch_slots(self.mp_physics))
        labels = {"rainnc": "RAINNC", "rainncv": "RAINNCV", "sr": "SR",
                  "snownc": "SNOWNC", "snowncv": "SNOWNCV",
                  "graupelnc": "GRAUPELNC",
                  "graupelncv": "GRAUPELNCV", "hailnc": "HAILNC",
                  "hailncv": "HAILNCV"}
        if slots:
            validated = {}
            required = {"rainnc", "rainncv", "sr"}
            if self.mp_physics == 18:
                # NSSL owns every frozen-category accumulator/increment in
                # its named contract.  Missing one would silently retain a
                # stale canonical scratch field across physics calls.
                required.update(slots)
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
                below = bool(cp.any(sr < DTYPE(0.0)))
                above = bool(cp.any(sr > DTYPE(self._sr_roundoff_upper)))
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
        result = self.cumulus_callable(
            atmosphere=atmosphere, fields=self.fields, state=state, cfg=cfg)
        if not isinstance(result, CumulusResult):
            raise TypeError("cumulus callable must return CumulusResult")
        nz, ny, nx = state.p.shape
        shape = (nz, ny, nx)
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
        """
        if self.cu_expiring is None or not bool(self.cu_expiring.any()):
            return
        mask = self.cu_expiring[None] != DTYPE(0.0)
        for array in self.cu_rates.values():
            array[...] = cp.where(mask, DTYPE(0.0), array)
        # These are the held, already mass-coupled copies used on the NEXT
        # compose.  The separately composed current-step target is untouched.
        for name in ("rtheta", "rqv", "rqc", "rqr", "rqi", "rqs"):
            array = getattr(self.cumulus_tendencies, name)
            if array is not None:
                array[...] = cp.where(mask, DTYPE(0.0), array)
        self.cu_expiring[...] = DTYPE(0.0)

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
                isfflx=1,
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
                    dx=cfg.dx, itimestep=itimestep, isfflx=1,
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
            isftcflx=cfg.isftcflx, iz0tlnd=cfg.iz0tlnd)
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
                    isftcflx=cfg.isftcflx, iz0tlnd=cfg.iz0tlnd)
                for name in (
                        "znt", "ust", "mol", "zol", "flhc", "flqc", "cpm",
                        "cqs2", "chs2", "chs", "qsfc", "qgh", "hfx", "qfx",
                    "lh"):
                    carrier = f"{name}_sea"
                    f[carrier][...] = cp.where(
                        ice_component, getattr(sea, name), f[carrier])

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
        # mp_physics=1 incorrectly freezes cold rain.
        use_scheme_sr = self.mp_physics in (1, 6, 8, 10, 18)
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
        use_scheme_sr = self.mp_physics in (1, 6, 8, 10, 18)
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
        use_scheme_sr = self.mp_physics in (1, 6, 8, 10, 18)
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

    def _refresh_surface_diagnostics(self) -> None:
        """Transcribe WRF v4.6.1 SFCDIAGS for the supported Noah path.

        Noah updates TSK/HFX/QFX/QSFC after SFCLAY has diagnosed T2/Q2/TH2.
        WRF therefore calls SFCDIAGS after the LSM and before the PBL driver
        (module_surface_driver.F:2983-3000; module_sf_sfcdiags.F:45-72).
        gpuwm does not expose WRF's UA_PHYS or HWRF compile-time branches, so
        this is their standard false/non-HWRF formulation used by real74.
        """
        f = self.fields
        rho = f["psfc"] / (DTYPE(c.RD) * f["tsk"])
        q_active = f["cqs2"] >= DTYPE(1.0e-5)
        t_active = f["chs2"] >= DTYPE(1.0e-5)
        safe_cqs2 = cp.where(q_active, f["cqs2"], DTYPE(1.0))
        safe_chs2 = cp.where(t_active, f["chs2"], DTYPE(1.0))
        f["q2"][...] = cp.where(
            q_active,
            f["qsfc"] - f["qfx"] / (rho * safe_cqs2),
            f["qsfc"])
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
            validate_ysu_tendencies(out)
        except FloatingPointError:
            self.ysu_nan_guard_fires += 1
            raise
        f["pblh"][...] = out["hpbl"]
        f["kpbl"][...] = out["kpbl"]
        f["exch_h"][...] = out["exch_h"]
        f["exch_m"][...] = out["exch_m"]
        self.pbl_tendencies = couple_ysu_tendencies(self.state, cfg, out)
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
        self.pbl_tendencies = couple_ysu_tendencies(self.state, cfg, out)
        pbl_components = (
            _composed_optional_tendency_components(cfg)
            if physics_reuses_pbl_composition(cfg)
            else _pbl_optional_tendency_components(cfg))
        self.pbl_tendencies.materialize(pbl_components)
        # MYNN has no counterpart to YSU's positive-cadence raw-output
        # retention: every consumer of its rates is the coupling above, and
        # its diagnostics already persist in ``fields``.
        self.last_ysu = None

    def compute(self, state: DomainState,
                cfg: RunConfig) -> PhysicsTendencies:
        """Run due physics at time t and return the held RK3 tendencies."""
        if state is not self.state:
            raise ValueError("PhysicsDriver is attached to a different state")
        # Complete any exceptional in-flight expiry before constructing this
        # step.  Normal compute() calls finalize the mask immediately after
        # composing the prior/current RK target, so this is a cheap invariant
        # guard for direct users and restored synthetic states.
        if cfg.cu_physics:
            self.finish_step()
        now = float(state.elapsed_seconds)
        itimestep = int(np.floor(now / cfg.dt + 0.5)) + 1
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
                getattr(self, land_method)(atmosphere, cfg, itimestep)
                if int(cfg.sf_surface_physics) in \
                        LAND_SURFACE_SFCDIAGS_SCHEMES:
                    self._refresh_surface_diagnostics()
                self.call_counts["noah"] += 1
            pbl_method = dispatch["bl_pbl_physics"]
            if pbl_method is not None:
                getattr(self, pbl_method)(atmosphere, cfg)
                self.call_counts["ysu"] += 1
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
        """Update externally supplied surface/radiation forcing in place."""
        allowed = {"swdown", "glw", "rainbl"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown physics forcing field(s): {sorted(unknown)}")
        shape = self.fields["tsk"].shape
        for name, value in fields.items():
            self.fields[name][...] = _as_2d(value, shape, name)

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
        if self.radiation_active:
            output.update(SWDOWN=self.fields["swdown"],
                          GLW=self.fields["glw"])
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


def initialize_physics(
        state: DomainState, cfg: RunConfig, *, landmask=1.0, tsk=300.0,
        soil_temperature=285.0, soil_moisture=0.30, liquid_moisture=None,
        ivgtyp=10, isltyp=6, vegfra=50.0, tmn=285.0, xice=0.0, snow=0.0,
        snow_depth=0.0, sst=None,
        swdown=0.0, glw=300.0, pblh=0.0, mavail=1.0,
        landuse=None,
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
    ``ra_lw_physics=0, ra_sw_physics=1`` resolves to Dudhia shortwave;
    the RRTM longwave half of the 1/1 pair is deliberately unavailable
    until its 16-band coefficient kernels land.  Production adapters require
    their UTC start time and mass-grid latitude/longitude here.
    ``cu_physics=1`` resolves to the
    Kain-Fritsch adapter (:class:`gpuwm.core.kf.KainFritsch`).  A
    scheme callable receives keyword arguments ``atmosphere``, ``fields``,
    ``state``, and ``cfg`` and must return :class:`RadiationResult` or
    :class:`CumulusResult`, respectively.
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
    if cfg.cu_physics not in (0, 1):
        raise ValueError("cu_physics must be 0 or 1")
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
                    p_top=getattr(state, "p_top", None))
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
        elif ra_lw_physics == 1:
            raise NotImplementedError(
                "ra_lw_physics=1 (WRF RRTM) is not executable yet: the "
                "16-band/140-g-point coefficient and transfer kernels are "
                "still required; no approximate LW scheme is substituted")
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
    intervals = (cfg.radt, cfg.bldt, cfg.radt_minutes, cfg.cudt_minutes)
    if any(not np.isfinite(value) or value < 0.0 for value in intervals):
        raise ValueError("physics intervals must be finite and non-negative")

    shape = state.mup.shape
    if landuse is not None:
        landmask = landuse.landmask
        ivgtyp = landuse.ivgtyp
        isltyp = landuse.isltyp
        pblh = landuse.pblh
        mavail = landuse.mavail
    land = _as_2d(landmask, shape, "landmask") >= DTYPE(0.5)
    xland = (_as_2d(landuse.xland, shape, "xland")
             if landuse is not None else
             cp.ascontiguousarray(cp.where(
                 land, DTYPE(1.0), DTYPE(2.0))))
    f: dict[str, cp.ndarray] = {
        "landmask": land.astype(DTYPE), "xland": xland,
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
        "swdown": _as_2d(swdown, shape, "swdown"),
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
        ruc_params = RucRuntimeParameters()

    noahmp_params = None
    noahmp_geometry = None
    if int(cfg.sf_surface_physics) == 4:
        noahmp_params = NoahmpRuntimeParameters()
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
            # disabled there is no future writer, so seed the carrier once at
            # the same half-interval hour angle a due radiation call uses.
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

    driver = PhysicsDriver(state, cfg, f, result, noah_params,
                           radiation=radiation, cumulus=cumulus,
                           noahmp_params=noahmp_params,
                           noahmp_geometry=noahmp_geometry,
                           ruc_params=ruc_params)
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
    return driver


__all__ = ["CumulusResult", "LAND_SURFACE_SFCDIAGS_SCHEMES",
           "PHYSICS_SLOT_DISPATCH", "PhysicsDriver", "PhysicsTendencies",
           "RadiationResult", "UnroutedPhysicsSelectorError",
           "couple_column_tendencies",
           "couple_ysu_tendencies", "initialize_physics",
           "microphysics_scratch_slots", "physics_enabled",
           "physics_driver_required",
           "physics_retains_ysu_output", "physics_reuses_pbl_composition",
           "resolve_physics_dispatch", "resolve_physics_slot",
           "validate_ysu_tendencies"]
