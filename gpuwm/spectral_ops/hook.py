"""One-call integration object for Arwen's slow large-step seam."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import SpectralNumericsConfig, from_mapping
from .runtime import apply_spectral_step


@dataclass
class SpectralLargeStepHook:
    """Stateful domain hook called once per completed slow large step.

    Construct one instance per Arwen domain.  The forecast loop calls it after
    the RK slow-mode state has been committed and before output, nest feedback,
    or the next large step.  It must not be called from acoustic substeps.
    """

    config: SpectralNumericsConfig
    dx_m: float
    dy_m: float
    dt_s: float
    domain: str

    @classmethod
    def from_toml(cls, path: str | Path, *, dx_m: float, dy_m: float,
                  dt_s: float, domain: str) -> "SpectralLargeStepHook":
        with Path(path).open("rb") as stream:
            document = tomllib.load(stream)
        mapping = document.get("spectral_numerics")
        return cls(from_mapping(mapping), float(dx_m), float(dy_m),
                   float(dt_s), str(domain))

    def __call__(self, state: Any, *, large_step: int,
                 source: Mapping[str, object] | None = None):
        identity = {"domain": self.domain}
        if source:
            identity.update(source)
        return apply_spectral_step(
            state, config=self.config, dx_m=self.dx_m, dy_m=self.dy_m,
            dt_s=self.dt_s, step=large_step, source=identity)


__all__ = ["SpectralLargeStepHook"]
