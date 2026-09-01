"""Declarative, fail-closed configuration for Level-2 spectral numerics."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .pins import canonical_hash
from .transfer import Hyperdiffusion


@dataclass(frozen=True)
class ScalarTarget:
    field: str
    diffusion: Hyperdiffusion
    space: str = "linear"
    floor: float = 1.0e-20
    preserve_mean: bool = True

    def validate(self) -> None:
        if not self.field or not self.field.strip():
            raise ValueError("a scalar spectral target needs a field name")
        if self.space not in ("linear", "log"):
            raise ValueError("scalar target space must be 'linear' or 'log'")
        if self.space == "log" and (not math.isfinite(self.floor) or self.floor <= 0):
            raise ValueError("a log-space scalar target needs a positive floor")
        self.diffusion.validate()


@dataclass(frozen=True)
class WindControl:
    enabled: bool = False
    u_field: str = "U"
    v_field: str = "V"
    staggering: str = "cgrid"
    divergent: Hyperdiffusion = field(default_factory=lambda: Hyperdiffusion(
        order=2, reference_wavelength_m=12_000.0, e_fold_time_s=300.0,
        protect_wavelength_m=48_000.0, maximum_damping_fraction=0.5))
    rotational: Hyperdiffusion | None = None

    def validate(self) -> None:
        if self.staggering not in ("mass", "cgrid"):
            raise ValueError("wind staggering must be 'mass' or 'cgrid'")
        if not self.u_field or not self.v_field:
            raise ValueError("wind control needs u_field and v_field")
        self.divergent.validate()
        if self.rotational is not None:
            self.rotational.validate()


@dataclass(frozen=True)
class SpectralNumericsConfig:
    mode: str = "off"
    cadence_steps: int = 1
    boundary: str = "tapered"
    edge_taper_cells: int = 12
    periodic_domain: bool = False
    scalar_targets: tuple[ScalarTarget, ...] = ()
    wind: WindControl = field(default_factory=WindControl)
    receipt_directory: str | None = None
    fail_on_budget_violation: bool = True
    maximum_scalar_rms_increment: float | None = None
    maximum_wind_rms_increment: float | None = None

    def validate(self) -> None:
        if self.mode not in ("off", "shadow", "apply"):
            raise ValueError("spectral numerics mode must be off, shadow or apply")
        if self.cadence_steps < 1:
            raise ValueError("spectral cadence_steps must be >= 1")
        if self.boundary not in ("periodic", "reflect", "tapered"):
            raise ValueError("spectral boundary must be periodic, reflect or tapered")
        if self.boundary == "periodic" and not self.periodic_domain:
            raise ValueError("periodic boundary requires periodic_domain=true")
        if self.boundary == "reflect" and self.wind.enabled:
            raise ValueError(
                "reflect is scalar-only; wind control requires tapered or periodic")
        if self.edge_taper_cells < 0:
            raise ValueError("edge_taper_cells must be non-negative")
        names = [target.field for target in self.scalar_targets]
        if len(names) != len(set(names)):
            raise ValueError("scalar spectral target field names must be unique")
        for target in self.scalar_targets:
            target.validate()
        self.wind.validate()
        for label, value in (
            ("maximum_scalar_rms_increment", self.maximum_scalar_rms_increment),
            ("maximum_wind_rms_increment", self.maximum_wind_rms_increment),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{label} must be non-negative and finite")

    @property
    def config_sha256(self) -> str:
        return canonical_hash(dataclasses.asdict(self))


def _diffusion(mapping: Mapping[str, Any]) -> Hyperdiffusion:
    allowed = {
        "order", "reference_wavelength_m", "e_fold_time_s",
        "protect_wavelength_m", "maximum_damping_fraction",
    }
    extra = set(mapping) - allowed
    if extra:
        raise ValueError(f"unknown hyperdiffusion keys: {sorted(extra)}")
    return Hyperdiffusion(**dict(mapping))


def from_mapping(mapping: Mapping[str, Any] | None) -> SpectralNumericsConfig:
    """Build from a ``[spectral_numerics]`` TOML mapping."""
    if mapping is None:
        return SpectralNumericsConfig()
    allowed = {
        "mode", "cadence_steps", "boundary", "edge_taper_cells",
        "periodic_domain", "scalar", "wind", "receipt_directory",
        "fail_on_budget_violation", "maximum_scalar_rms_increment",
        "maximum_wind_rms_increment",
    }
    extra = set(mapping) - allowed
    if extra:
        raise ValueError(f"unknown spectral_numerics keys: {sorted(extra)}")
    scalar_rows = mapping.get("scalar", ())
    if isinstance(scalar_rows, Mapping):
        scalar_rows = (scalar_rows,)
    targets = []
    for row in scalar_rows:
        if not isinstance(row, Mapping):
            raise ValueError("spectral_numerics.scalar must be a table or array of tables")
        row = dict(row)
        diffusion_row = row.pop("diffusion", {})
        targets.append(ScalarTarget(
            diffusion=_diffusion(diffusion_row), **row))
    wind_row = dict(mapping.get("wind", {}))
    divergent = _diffusion(wind_row.pop("divergent", {})) if "divergent" in wind_row else WindControl().divergent
    rotational_row = wind_row.pop("rotational", None)
    rotational = None if rotational_row is None else _diffusion(rotational_row)
    wind = WindControl(divergent=divergent, rotational=rotational, **wind_row)
    kwargs = {k: v for k, v in mapping.items() if k not in ("scalar", "wind")}
    config = SpectralNumericsConfig(
        scalar_targets=tuple(targets), wind=wind, **kwargs)
    config.validate()
    return config


def canonical_document(config: SpectralNumericsConfig) -> dict[str, Any]:
    config.validate()
    return json.loads(json.dumps(dataclasses.asdict(config)))


__all__ = ["ScalarTarget", "SpectralNumericsConfig", "WindControl",
           "canonical_document", "from_mapping"]
