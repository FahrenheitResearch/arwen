"""EXPERIMENTAL (ArWen v1.2): rung-1 DA -- reflectivity nudging.

Latent-heat insertion, the cheapest useful thing you can do with a radar
volume: where the observations show an echo the model has not built,
warm the observed-echo column a bounded amount and moisten it toward a
target relative humidity, so the next few minutes of integration spin up
convection in the right place instead of the wrong one.  Optionally --
OFF by default -- do the reverse where the model has an echo and the
observations are clear.

This is not a filter.  It has no error covariance, no observation
weighting, no analysis increment in any statistical sense, and it will
happily create energy.  It is a nudge, and it is bounded so that a bad
observation degrades the forecast rather than destroying it.  The real
analysis is the EnKF lane's job; this is the rung below.

Everything is a pure function of ``(state, z_obs, z_mask, cfg)``: no
RNG, no clock, no filesystem, no global state, no ordering dependence.
Two calls on equal inputs return bitwise-equal increments.

The caller applies the increments; this module never mutates the state.
The returned field keys name state attributes:

- ``"thp"`` -- perturbation potential temperature (K).  The base state
  ``thb`` is time-invariant, so an increment to theta IS an increment to
  ``thp``; add it to ``state.thp``.
- ``"qv"`` -- water vapour mixing ratio (kg/kg).

Nothing here is wired into any default route.

EXPERIMENTAL.  Not covered by the v1 stability promise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np

from gpuwm.core import constants as c
from gpuwm.da.obsop import _array_module, _to_host, mass_point_heights

EXPERIMENTAL = True

#: Provenance dict schema tag.
PROVENANCE_SCHEMA = "gpuwm-da.hotstart-provenance.v1"

#: Vertical shape functions the warm increment can take.
RAMP_SHAPES = ("sin2", "tent", "uniform")

#: Saturation phases the moisture target can be referenced to.  ``mixed``
#: is liquid above 273.15 K, ice below 253.15 K, and a linear blend of the
#: two saturation vapour pressures between -- the WPS ``rrpr.F`` mixed-phase
#: convention this repository already uses for ERA5 RH.
SATURATION_PHASES = ("liquid", "ice", "mixed")

#: Bounds of the mixed-phase blend (K): all liquid at or above the upper,
#: all ice at or below the lower, linear in between.
_MIXED_PHASE_WARM_K = 273.15
_MIXED_PHASE_COLD_K = 253.15

__all__ = [
    "EXPERIMENTAL", "PROVENANCE_SCHEMA", "RAMP_SHAPES", "SATURATION_PHASES",
    "HotStartConfig", "HotStartResult", "hotstart_increments",
    "vertical_ramp", "saturation_mixing_ratio",
]


@dataclass(frozen=True)
class HotStartConfig:
    """Every knob, with conservative defaults.

    Defaults are deliberately timid: a 2 K ceiling on warming is about a
    tenth of the buoyancy a mature updraught carries, enough to bias the
    model toward convection where radar says there is some, not enough to
    manufacture a storm out of a clear sounding.  The clear-air branch is
    off because removing model echo is the half of this scheme most
    likely to hurt -- it destroys structure the model may have got right
    but displaced.
    """

    # -- detection -------------------------------------------------------
    #: Observed dBZ above this counts as an observed echo.
    echo_threshold_dbz: float = 15.0
    #: Only nudge where the model is short of the observation by at least
    #: this many dB.  Non-zero so the operator no-ops on a good forecast.
    deficit_threshold_db: float = 5.0
    #: Ceiling on the observed reflectivity this operator will insert
    #: from, in dBZ.  The observation is capped at this value BEFORE the
    #: deficit is formed, so a 70 dBZ hail spike drives the same nudge a
    #: 55 dBZ convective core does.
    #:
    #: 55 dBZ is the conventional radar-DA insertion ceiling and the
    #: reason is contamination, not dynamic range: at S band, returns
    #: above it are dominated by wet hail and three-body scatter, which
    #: the latent-heat closure below has no representation of -- it reads
    #: dBZ as a proxy for the condensation the model is missing, and
    #: hail's Z is not that.  Sun and Crook (1997/1998) and the Tong and
    #: Xue (2005) line of work cap the reflectivity they assimilate for
    #: this reason; WRFDA's radar-DA namelist carries the same ceiling.
    #:
    #: It is a CONFIG FIELD with a clamp behind it, and both are on
    #: purpose: this cap was previously stated as a convention callers
    #: were expected to honour, with no constant, no clamp, no wiring and
    #: no provenance anywhere in the tree, which made it a claim nothing
    #: implemented.  Raise it deliberately if the experiment wants a
    #: different ceiling; the value used is recorded in the provenance,
    #: along with how many cells it actually bound.
    max_insertion_dbz: float = 55.0

    # -- warm increment --------------------------------------------------
    #: Kelvin of theta per dB of reflectivity deficit, before shaping.
    theta_per_db: float = 0.05
    #: Hard ceiling on the theta increment (K), after shaping.
    max_theta_increment_k: float = 2.0
    #: Vertical support of the warm increment, metres above ground.
    warm_base_height_m: float = 1000.0
    warm_top_height_m: float = 8000.0
    #: One of :data:`RAMP_SHAPES`.
    ramp_shape: str = "sin2"

    # -- moisture --------------------------------------------------------
    adjust_moisture: bool = True
    #: Relative humidity (fraction) the echo column is pushed toward.
    rh_target: float = 0.95
    #: Which saturation the RH target is referenced to; one of
    #: :data:`SATURATION_PHASES`.  ``mixed`` (the default) tracks the
    #: thermodynamic phase -- liquid warm, ice cold, blended between -- so a
    #: warm/moisten insertion into a subfreezing convective column targets
    #: ice saturation there and does NOT drive the vapour into ice
    #: supersaturation the way a liquid-only target does (0.95 over liquid
    #: is ~116% over ice at -20 C).  ``liquid`` keeps the original
    #: liquid-water target for a caller who wants it and names it.
    saturation_phase: str = "mixed"
    #: Hard ceiling on the qv increment (kg/kg).
    max_qv_increment_kg_kg: float = 2.0e-3

    # -- clear-air branch (OFF by default) -------------------------------
    clear_air_enabled: bool = False
    #: Observed dBZ at or below this counts as observed clear air.
    clear_air_obs_threshold_dbz: float = 5.0
    #: Simulated dBZ at or above this counts as spurious model echo.
    clear_air_model_threshold_dbz: float = 15.0
    #: Kelvin of cooling per dB of excess, before shaping.
    clear_air_theta_per_db: float = 0.05
    #: Hard floor on the theta decrement (K, magnitude).
    max_theta_decrement_k: float = 1.0
    #: Relative humidity the spurious-echo column is dried toward.
    clear_air_rh_target: float = 0.60
    #: Hard ceiling on the qv decrement (kg/kg, magnitude).
    max_qv_decrement_kg_kg: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.ramp_shape not in RAMP_SHAPES:
            raise ValueError(
                f"ramp_shape must be one of {RAMP_SHAPES}, "
                f"got {self.ramp_shape!r}")
        if self.saturation_phase not in SATURATION_PHASES:
            raise ValueError(
                f"saturation_phase must be one of {SATURATION_PHASES}, "
                f"got {self.saturation_phase!r}")
        if self.warm_top_height_m <= self.warm_base_height_m:
            raise ValueError(
                f"warm_top_height_m ({self.warm_top_height_m}) must exceed "
                f"warm_base_height_m ({self.warm_base_height_m})")
        for name in ("max_theta_increment_k", "max_qv_increment_kg_kg",
                     "max_theta_decrement_k", "max_qv_decrement_kg_kg"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value}")
        for name in ("theta_per_db", "clear_air_theta_per_db"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value}")
        for name in ("rh_target", "clear_air_rh_target"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.5:
                raise ValueError(
                    f"{name} is a fraction, not a percentage; got {value}")
        for name in ("echo_threshold_dbz", "deficit_threshold_db",
                     "clear_air_obs_threshold_dbz",
                     "clear_air_model_threshold_dbz",
                     "max_insertion_dbz"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.max_insertion_dbz < self.echo_threshold_dbz:
            # A cap below the echo threshold makes the warm branch
            # unreachable -- every capped observation falls under the test
            # that admits it -- so the operator would silently do nothing.
            raise ValueError(
                f"max_insertion_dbz ({self.max_insertion_dbz}) is below "
                f"echo_threshold_dbz ({self.echo_threshold_dbz}); every "
                "observation would be capped under the threshold that "
                "admits it and the warm branch could never fire")


class HotStartResult(NamedTuple):
    """``(increments, provenance)``.

    ``increments`` is the ``dict[field -> increment array]`` the caller
    adds to the state; ``provenance`` is a plain JSON-serialisable dict
    recording every threshold, count and total behind it.  Unpacks as a
    two-tuple.
    """

    increments: dict
    provenance: dict


def vertical_ramp(height_agl_m, cfg: HotStartConfig):
    """Vertical shape weight in [0, 1] over the warm layer.

    ``sin2`` -- ``sin(pi*xi)**2``, smooth and zero at both ends.
    ``tent`` -- ``1 - |2*xi - 1|``, triangular, peak mid-layer.
    ``uniform`` -- 1 inside the layer, 0 outside.

    ``xi`` is the fractional position between ``warm_base_height_m`` and
    ``warm_top_height_m``; everything outside that layer weighs zero, so
    the increment cannot reach the surface or the stratosphere.
    """
    xp = _array_module(height_agl_m)
    base = xp.asarray(cfg.warm_base_height_m, dtype=height_agl_m.dtype)
    top = xp.asarray(cfg.warm_top_height_m, dtype=height_agl_m.dtype)
    xi = (height_agl_m - base) / (top - base)
    inside = (xi >= 0.0) & (xi <= 1.0)
    clamped = xp.clip(xi, 0.0, 1.0)
    if cfg.ramp_shape == "sin2":
        shape = xp.sin(float(np.pi) * clamped) ** 2
    elif cfg.ramp_shape == "tent":
        shape = 1.0 - xp.abs(2.0 * clamped - 1.0)
    else:
        shape = xp.ones_like(clamped)
    return xp.where(inside, shape, xp.zeros_like(shape))


def _es_liquid(xp, temperature):
    """Bolton/Tetens saturation vapour pressure over liquid water (Pa)."""
    return (1000.0 * c.SVP1
            * xp.exp(c.SVP2 * (temperature - c.SVPT0)
                     / (temperature - c.SVP3)))


def _es_ice(xp, temperature):
    """Murphy and Koop (2005) saturation vapour pressure over ice (Pa).

    The transcription the repo already uses for WPS RH conversion
    (gpuwm/ingest/horiz.py), in Pa here rather than hPa.
    """
    return xp.exp(9.550426 - 5723.265 / temperature
                  + 3.53068 * xp.log(temperature)
                  - 0.00728332 * temperature)


def saturation_mixing_ratio(temperature, pressure, *, phase="mixed"):
    """Saturation mixing ratio (kg/kg), over the requested thermodynamic phase.

    ``phase``:

    - ``"liquid"`` -- Bolton/Tetens over liquid water at all temperatures,
      ``es = 1000*SVP1*exp(SVP2*(T-SVPT0)/(T-SVP3))``.  The original
      behaviour, kept for callers who ask for it by name.
    - ``"ice"`` -- Murphy and Koop (2005) over ice at all temperatures.
    - ``"mixed"`` (default) -- liquid at/above 273.15 K, ice at/below
      253.15 K, and a linear blend of the two saturation vapour pressures
      between, the WPS ``rrpr.F`` convention.  This is what keeps a moist
      insertion into a subfreezing column from targeting liquid saturation
      and overshooting ice saturation.

    Then ``qs = EP2*es/(p - es)`` in every case.
    """
    if phase not in SATURATION_PHASES:
        raise ValueError(
            f"phase must be one of {SATURATION_PHASES}, got {phase!r}")
    xp = _array_module(temperature)
    if phase == "liquid":
        es = _es_liquid(xp, temperature)
    elif phase == "ice":
        es = _es_ice(xp, temperature)
    else:
        esw = _es_liquid(xp, temperature)
        esi = _es_ice(xp, temperature)
        # Linear in temperature across the mixing band, clamped to [0, 1]
        # so the two anchors return the pure phase exactly.
        frac = (temperature - _MIXED_PHASE_COLD_K) / (
            _MIXED_PHASE_WARM_K - _MIXED_PHASE_COLD_K)
        frac = xp.clip(frac, 0.0, 1.0)
        es = frac * esw + (1.0 - frac) * esi
    # A cell whose saturation vapour pressure exceeds its total pressure
    # is unphysical; keep the denominator positive rather than emitting a
    # negative mixing ratio.
    denominator = xp.maximum(pressure - es, 1.0)
    return c.EP2 * es / denominator


def _diagnose_temperature(state):
    """Air temperature (K) from the state's own theta and pressure."""
    xp = _array_module(state.p)
    thb = state.thb
    if getattr(thb, "ndim", 0) == 1:
        thb = thb[:, None, None]
    theta = thb + state.thp
    return theta * xp.power(state.p / np.float32(c.P0), np.float32(c.RCP))


def _height_agl(state):
    """Mass-point height above ground level (m).

    The lowest full level is the terrain surface, so subtracting it gives
    AGL without needing a separate terrain field.
    """
    php = state.php
    phb = state.phb
    if getattr(phb, "ndim", 0) == 1:
        phb = phb[:, None, None]
    surface = (phb + php)[:1] / np.float32(c.G)
    return mass_point_heights(state) - surface


def _count(xp, mask) -> int:
    return int(np.asarray(_to_host(xp, xp.count_nonzero(mask))))


def _scalar(xp, value) -> float:
    return float(np.asarray(_to_host(xp, value)))


def hotstart_increments(state, z_obs, z_mask, cfg: HotStartConfig, *,
                        simulated_dbz=None, run_cfg=None,
                        temperature=None, pressure=None,
                        applied=None) -> HotStartResult:
    """Bounded warm/moist increments where radar sees echo the model lacks.

    ``z_obs`` is observed reflectivity on the model grid (dBZ,
    ``(nz, ny, nx)``); ``z_mask`` is True where that observation is
    valid -- outside the radar's coverage, below the lowest tilt, or
    beam-blocked cells must be False, and no increment is ever produced
    where the mask is False.  Both are the ``gpuwm-obs.radar-grid.v1``
    gridded arrays.

    ``z_obs`` is capped at ``cfg.max_insertion_dbz`` (55 dBZ by default)
    before the deficit is formed -- an invariant of this function, not a
    convention the caller is trusted with.  The cap and the number of
    cells it bound are recorded in the provenance.

    ``cfg`` is the :class:`HotStartConfig`.  The simulated reflectivity
    comes from ``simulated_dbz=`` (preferred -- the caller usually has it
    already) or is computed from ``run_cfg=`` via the bound Z operator;
    supplying neither raises rather than assuming clear air.

    ``applied`` chooses the cap semantics, and the choice is explicit
    because it has to be.  Every cap here is applied to THIS call's
    increment.  Since the nudge changes ``thp``/``qv`` but not the
    hydrometeors that dominate simulated Z, a persistent reflectivity
    deficit is not removed by applying the increment, so calling again on
    the updated state re-fires the warm branch and adds another full
    increment: n applications reach n times the cap.

    - ``applied=None`` (default): the caps are PER INVOCATION.  The caller
      is responsible for applying the increment at most once per analysis
      time, or for accepting per-cycle tendencies.
    - ``applied={"thp": array, "qv": array}``: the cumulative increment
      already inserted at this analysis time.  The returned increment is
      then clamped so ``applied + increment`` stays within the configured
      ``[-decrement, +increment]`` caps, making them TOTAL analysis-time
      bounds.  Thread it across repeated applications (accumulating it
      yourself) and the totals cannot exceed the caps however many times
      you apply.

    Returns :class:`HotStartResult`.  Cells that fail any test get
    exactly ``0.0``, so a forecast that already matches the observations
    produces bitwise-zero increments and applying them is a no-op.

    Non-finite ``z_obs`` or ``simulated_dbz`` never produces an
    increment: every gate is a comparison, and comparisons against NaN
    are False.
    """
    if not isinstance(cfg, HotStartConfig):
        raise TypeError(
            f"cfg must be a HotStartConfig, got {type(cfg).__name__}")

    if simulated_dbz is None:
        if run_cfg is None:
            raise ValueError(
                "hotstart_increments needs the simulated reflectivity: pass "
                "simulated_dbz= (preferred) or run_cfg= so the bound Z "
                "operator can be evaluated here")
        from gpuwm.da.obsop import simulated_reflectivity
        simulated_dbz = simulated_reflectivity(
            state, run_cfg, temperature=temperature, pressure=pressure)

    xp = _array_module(state.p)
    shape = tuple(np.asarray(_to_host(xp, state.p)).shape)
    for name, array in (("z_obs", z_obs), ("z_mask", z_mask),
                        ("simulated_dbz", simulated_dbz)):
        got = tuple(np.asarray(_to_host(_array_module(array), array)).shape)
        if got != shape:
            raise ValueError(
                f"{name} has shape {got}, expected the model grid {shape}")

    if applied is not None:
        missing = {"thp", "qv"} - set(applied)
        if missing:
            raise ValueError(
                f"applied ledger is missing {sorted(missing)}; it must carry "
                "the cumulative 'thp' and 'qv' increments already inserted "
                "at this analysis time")
        for name in ("thp", "qv"):
            got = tuple(np.asarray(
                _to_host(_array_module(applied[name]), applied[name])).shape)
            if got != shape:
                raise ValueError(
                    f"applied[{name!r}] has shape {got}, expected the model "
                    f"grid {shape}")

    dtype = state.p.dtype
    z_obs = z_obs.astype(dtype, copy=False)
    simulated_dbz = simulated_dbz.astype(dtype, copy=False)
    valid = z_mask.astype(bool, copy=False)

    # The insertion cap, applied once and before anything reads z_obs.
    # ``minimum`` and not ``fmin``: fmin RETURNS the cap where the
    # observation is NaN, which would turn a missing observation into a
    # 55 dBZ echo -- the exact inversion of the "every gate is a
    # comparison and NaN compares False" contract below.
    z_obs_raw = z_obs
    cap = xp.asarray(cfg.max_insertion_dbz, dtype=dtype)
    z_obs = xp.minimum(z_obs, cap)
    capped_cells = valid & (z_obs_raw > cap)

    if temperature is None:
        temperature = _diagnose_temperature(state)
    if pressure is None:
        pressure = state.p

    height_agl = _height_agl(state)
    ramp = vertical_ramp(height_agl, cfg).astype(dtype, copy=False)
    in_layer = ramp > 0.0

    zero = xp.zeros(shape, dtype=dtype)
    d_theta = zero
    d_qv = zero

    # -- warm branch: observed echo the model is short of -----------------
    deficit = z_obs - simulated_dbz
    warm_mask = (valid
                 & (z_obs >= cfg.echo_threshold_dbz)
                 & (deficit >= cfg.deficit_threshold_db)
                 & in_layer)
    warm_theta = xp.clip(cfg.theta_per_db * deficit * ramp,
                         0.0, cfg.max_theta_increment_k)
    d_theta = xp.where(warm_mask, warm_theta, zero)

    qsat = saturation_mixing_ratio(temperature, pressure,
                                   phase=cfg.saturation_phase)
    if cfg.adjust_moisture:
        shortfall = xp.clip(cfg.rh_target * qsat - state.qv,
                            0.0, cfg.max_qv_increment_kg_kg)
        d_qv = xp.where(warm_mask, xp.clip(shortfall * ramp, 0.0,
                                           cfg.max_qv_increment_kg_kg), zero)

    # -- clear-air branch: model echo the observations deny ---------------
    clear_mask = xp.zeros(shape, dtype=bool)
    if cfg.clear_air_enabled:
        excess = simulated_dbz - z_obs
        clear_mask = (valid
                      & (z_obs <= cfg.clear_air_obs_threshold_dbz)
                      & (simulated_dbz >= cfg.clear_air_model_threshold_dbz)
                      & in_layer)
        cool = xp.clip(cfg.clear_air_theta_per_db * excess * ramp,
                       0.0, cfg.max_theta_decrement_k)
        d_theta = xp.where(clear_mask, -cool, d_theta)
        if cfg.adjust_moisture:
            surplus = xp.clip(state.qv - cfg.clear_air_rh_target * qsat,
                              0.0, cfg.max_qv_decrement_kg_kg)
            dry = xp.clip(surplus * ramp, 0.0, cfg.max_qv_decrement_kg_kg)
            d_qv = xp.where(clear_mask, -dry, d_qv)

    # -- post-condition clamp --------------------------------------------
    # The branch arithmetic above is already bounded; clamping the final
    # arrays makes the guarantee structural rather than a property of
    # having read the branches correctly.
    #
    # With an applied= ledger the bound is on the CUMULATIVE increment, not
    # this call's: clamp so applied + increment stays within the same
    # [-decrement, +increment] interval.  applied is subtracted from both
    # bounds, so a cell already at the cap gets a headroom of zero and this
    # call adds nothing there -- which is what stops n applications reaching
    # n times the cap.
    if applied is None:
        lo_theta, hi_theta = -cfg.max_theta_decrement_k, cfg.max_theta_increment_k
        lo_qv, hi_qv = -cfg.max_qv_decrement_kg_kg, cfg.max_qv_increment_kg_kg
    else:
        at = applied["thp"].astype(dtype, copy=False)
        aq = applied["qv"].astype(dtype, copy=False)
        lo_theta = -cfg.max_theta_decrement_k - at
        hi_theta = cfg.max_theta_increment_k - at
        lo_qv = -cfg.max_qv_decrement_kg_kg - aq
        hi_qv = cfg.max_qv_increment_kg_kg - aq
    d_theta = xp.clip(d_theta, lo_theta, hi_theta)
    d_qv = xp.clip(d_qv, lo_qv, hi_qv)

    # Never drive vapour negative.
    d_qv = xp.maximum(d_qv, -state.qv)

    increments = {"thp": d_theta, "qv": d_qv}
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "experimental": True,
        "operator": "reflectivity_nudge_latent_heat_insertion",
        "obs_schema": "gpuwm-obs.radar-grid.v1",
        "config": {key: (float(value) if isinstance(value, float) else value)
                   for key, value in asdict(cfg).items()},
        "grid_shape": [int(n) for n in shape],
        "total_cells": int(np.prod(shape)),
        "valid_obs_cells": _count(xp, valid),
        "in_warm_layer_cells": _count(xp, in_layer),
        "observed_echo_cells": _count(
            xp, valid & (z_obs >= cfg.echo_threshold_dbz)),
        #: The cap, and how much of the volume it actually bound.  A cap
        #: nobody can tell fired is a cap nobody can audit.
        "insertion_cap_dbz": float(cfg.max_insertion_dbz),
        "capped_obs_cells": _count(xp, capped_cells),
        "warm_increment_cells": _count(xp, warm_mask),
        "clear_air_enabled": bool(cfg.clear_air_enabled),
        "clear_air_increment_cells": _count(xp, clear_mask),
        "theta_increment_sum_k": _scalar(xp, xp.sum(d_theta)),
        "theta_increment_max_k": _scalar(xp, xp.max(d_theta)),
        "theta_increment_min_k": _scalar(xp, xp.min(d_theta)),
        "qv_increment_sum_kg_kg": _scalar(xp, xp.sum(d_qv)),
        "qv_increment_max_kg_kg": _scalar(xp, xp.max(d_qv)),
        "qv_increment_min_kg_kg": _scalar(xp, xp.min(d_qv)),
        "theta_clamp_bound": bool(_scalar(
            xp, xp.max(d_theta)) >= cfg.max_theta_increment_k),
        "qv_clamp_bound": bool(_scalar(
            xp, xp.max(d_qv)) >= cfg.max_qv_increment_kg_kg),
        "cumulative": applied is not None,
        "deterministic": True,
        "seed_free": True,
    }
    if applied is not None:
        # The cumulative totals AFTER this call, so the ledger the caller
        # threads next is auditable and the "did it hit the total cap" is
        # answerable from provenance alone.
        cum_theta = applied["thp"].astype(dtype, copy=False) + d_theta
        cum_qv = applied["qv"].astype(dtype, copy=False) + d_qv
        provenance["applied_theta_increment_max_k"] = _scalar(
            xp, xp.max(cum_theta))
        provenance["applied_qv_increment_max_kg_kg"] = _scalar(
            xp, xp.max(cum_qv))
    return HotStartResult(increments=increments, provenance=provenance)
