"""Smooth transfer functions for scale-selective numerical operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .backend import namespace


@dataclass(frozen=True)
class Hyperdiffusion:
    """Exact exponential hyperdiffusion specification.

    ``reference_wavelength_m`` is the wavelength whose amplitude has an
    e-folding time of ``e_fold_time_s`` once the optional protected-scale
    transition is fully active.  ``order=1`` is Laplacian diffusion, 2 is
    fourth order, and 3 is sixth order.
    """

    order: int = 3
    reference_wavelength_m: float = 6_000.0
    e_fold_time_s: float = 600.0
    protect_wavelength_m: float | None = None
    maximum_damping_fraction: float = 1.0

    def validate(self) -> None:
        if self.order < 1 or self.order > 8:
            raise ValueError("hyperdiffusion order must lie in [1, 8]")
        if not math.isfinite(self.reference_wavelength_m) or self.reference_wavelength_m <= 0:
            raise ValueError("reference_wavelength_m must be positive and finite")
        if not math.isfinite(self.e_fold_time_s) or self.e_fold_time_s <= 0:
            raise ValueError("e_fold_time_s must be positive and finite")
        if self.protect_wavelength_m is not None:
            if (not math.isfinite(self.protect_wavelength_m)
                    or self.protect_wavelength_m <= self.reference_wavelength_m):
                raise ValueError(
                    "protect_wavelength_m must be finite and larger than the reference wavelength")
        if not 0.0 <= self.maximum_damping_fraction <= 1.0:
            raise ValueError("maximum_damping_fraction must lie in [0, 1]")

    def transfer(self, magnitude: Any, *, dt_s: float):
        self.validate()
        if not math.isfinite(dt_s) or dt_s < 0.0:
            raise ValueError("dt_s must be non-negative and finite")
        xp = namespace(magnitude)
        k_ref = 2.0 * math.pi / self.reference_wavelength_m
        ratio = magnitude / k_ref
        activation = xp.ones_like(magnitude)
        if self.protect_wavelength_m is not None:
            k_protect = 2.0 * math.pi / self.protect_wavelength_m
            t = xp.clip((magnitude - k_protect) / (k_ref - k_protect), 0.0, 1.0)
            activation = 0.5 - 0.5 * xp.cos(math.pi * t)
            activation = xp.where(magnitude <= k_protect, 0.0, activation)
            activation = xp.where(magnitude >= k_ref, 1.0, activation)
        exponent = -(dt_s / self.e_fold_time_s) * activation * ratio ** (2 * self.order)
        # exp(-80) is already below float32 significance and avoids needless
        # overflow in intermediate powers at very high k.
        transfer = xp.exp(xp.maximum(exponent, -80.0))
        lower = 1.0 - self.maximum_damping_fraction
        transfer = xp.maximum(transfer, lower)
        # The constant mode is a conservation statement, not a tuning choice.
        return xp.where(magnitude == 0.0, 1.0, transfer)


@dataclass(frozen=True)
class RaisedCosineLowPass:
    pass_wavelength_m: float
    stop_wavelength_m: float

    def transfer(self, magnitude: Any):
        if not (math.isfinite(self.pass_wavelength_m)
                and math.isfinite(self.stop_wavelength_m)
                and self.pass_wavelength_m > self.stop_wavelength_m > 0.0):
            raise ValueError("low-pass wavelengths require pass > stop > 0")
        xp = namespace(magnitude)
        k_pass = 2.0 * math.pi / self.pass_wavelength_m
        k_stop = 2.0 * math.pi / self.stop_wavelength_m
        t = xp.clip((magnitude - k_pass) / (k_stop - k_pass), 0.0, 1.0)
        value = 0.5 + 0.5 * xp.cos(math.pi * t)
        value = xp.where(magnitude <= k_pass, 1.0, value)
        value = xp.where(magnitude >= k_stop, 0.0, value)
        return xp.where(magnitude == 0.0, 1.0, value)


def compose(*transfers: Any):
    if not transfers:
        raise ValueError("compose needs at least one transfer")
    result = transfers[0]
    for transfer in transfers[1:]:
        result = result * transfer
    return result


__all__ = ["Hyperdiffusion", "RaisedCosineLowPass", "compose"]
