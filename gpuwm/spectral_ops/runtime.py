"""Optional Level-2 operator controller for one Arwen large step."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from .backend import backend_name
from .config import SpectralNumericsConfig
from .receipt import make_receipt, write_json_atomic
from .scalar import hyperdiffuse
from .vector import damp_c_grid_divergence, damp_divergence


class SpectralBudgetExceeded(RuntimeError):
    pass


def _field_get(state: Any, name: str):
    if isinstance(state, Mapping):
        if name not in state:
            raise ValueError(f"model state is missing spectral target {name}")
        return state[name]
    if not hasattr(state, name):
        raise ValueError(f"model state is missing spectral target {name}")
    return getattr(state, name)


def _field_set(state: Any, name: str, value: Any) -> None:
    if isinstance(state, MutableMapping):
        state[name] = value
        return
    setattr(state, name, value)


def _budget(config: SpectralNumericsConfig, *, kind: str, value: float) -> None:
    limit = (config.maximum_scalar_rms_increment if kind == "scalar"
             else config.maximum_wind_rms_increment)
    if limit is None or value <= limit:
        return
    message = f"spectral {kind} RMS increment {value:.9g} exceeds budget {limit:.9g}"
    if config.fail_on_budget_violation:
        raise SpectralBudgetExceeded(message)


def apply_spectral_step(state: Any, *, config: SpectralNumericsConfig,
                        dx_m: float, dy_m: float, dt_s: float, step: int,
                        source: Mapping[str, object] | None = None
                        ) -> dict[str, object] | None:
    """Propose or apply one configured spectral step.

    ``off`` and non-cadence steps return ``None`` without reading a field.
    ``shadow`` computes every proposed correction and receipt but does not
    mutate ``state``.  ``apply`` mutates only after every field has been
    computed and every configured budget has passed, so a failure cannot leave
    a half-updated state.
    """
    config.validate()
    if config.mode == "off" or step % config.cadence_steps:
        return None
    if not all(math.isfinite(v) and v > 0.0 for v in (dx_m, dy_m, dt_s)):
        raise ValueError("spectral runtime requires positive finite dx, dy and dt")
    proposed: dict[str, Any] = {}
    metrics: dict[str, object] = {"scalar": {}, "wind": None}
    representative = None
    for target in config.scalar_targets:
        original = _field_get(state, target.field)
        representative = original
        boundary = config.boundary
        result = hyperdiffuse(
            original, dy_m=dy_m, dx_m=dx_m,
            dt_s=dt_s * config.cadence_steps, spec=target.diffusion,
            boundary=boundary, edge_taper_cells=config.edge_taper_cells,
            periodic_domain=config.periodic_domain,
            preserve_mean=target.preserve_mean, space=target.space,
            floor=target.floor)
        _budget(config, kind="scalar", value=result.rms_increment)
        proposed[target.field] = result.values
        metrics["scalar"][target.field] = {
            "space": target.space,
            "mean_before": result.mean_before,
            "mean_after": result.mean_after,
            "rms_increment": result.rms_increment,
            "max_abs_increment": result.max_abs_increment,
            "minimum_before": result.minimum_before,
            "minimum_after": result.minimum_after,
        }
    if config.wind.enabled:
        u = _field_get(state, config.wind.u_field)
        v = _field_get(state, config.wind.v_field)
        representative = u
        kwargs = dict(
            dy_m=dy_m, dx_m=dx_m, dt_s=dt_s * config.cadence_steps,
            divergent=config.wind.divergent, rotational=config.wind.rotational,
            boundary=config.boundary,
            edge_taper_cells=config.edge_taper_cells,
            periodic_domain=config.periodic_domain)
        result = (damp_c_grid_divergence(u, v, **kwargs)
                  if config.wind.staggering == "cgrid"
                  else damp_divergence(u, v, **kwargs))
        _budget(config, kind="wind", value=result.rms_increment)
        proposed[config.wind.u_field] = result.u
        proposed[config.wind.v_field] = result.v
        metrics["wind"] = {
            "u_field": config.wind.u_field,
            "v_field": config.wind.v_field,
            "staggering": config.wind.staggering,
            "divergence_rms_before": result.divergence_rms_before,
            "divergence_rms_after": result.divergence_rms_after,
            "kinetic_energy_before": result.kinetic_energy_before,
            "kinetic_energy_after": result.kinetic_energy_after,
            "rms_increment": result.rms_increment,
            "max_abs_increment": result.max_abs_increment,
        }
    if representative is None:
        raise ValueError("spectral mode is active but no scalar or wind target is configured")
    applied = config.mode == "apply"
    if applied:
        for name, value in proposed.items():
            _field_set(state, name, value)
    receipt = make_receipt(
        config_sha256=config.config_sha256, mode=config.mode, step=step,
        dt_s=dt_s, dx_m=dx_m, dy_m=dy_m,
        backend=backend_name(representative), metrics=metrics,
        applied=applied, source=source)
    if config.receipt_directory:
        # A nested chain commits step N once per DOMAIN, so a file named
        # by the step alone is written once per domain and only the last
        # writer survives.  Measured on a two-domain run: 300 receipts
        # computed, 240 files left.  The domain therefore leads the name
        # whenever the caller identifies one; a call with no identity
        # (the CLI, the delivered demo receipt) keeps the bare spelling.
        domain = (source or {}).get("domain")
        stem = (f"spectral-step-{step:09d}" if not domain
                else f"{domain}-spectral-step-{step:09d}")
        path = Path(config.receipt_directory) / f"{stem}.json"
        write_json_atomic(path, receipt)
    return receipt


__all__ = ["SpectralBudgetExceeded", "apply_spectral_step"]
