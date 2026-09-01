"""Human- and machine-readable transfer-response tables."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .transfer import Hyperdiffusion


def hyperdiffusion_response(spec: Hyperdiffusion, *, dt_s: float,
                            wavelengths_m: Iterable[float]) -> list[dict[str, float]]:
    wavelengths = np.asarray(list(wavelengths_m), dtype=np.float64)
    if wavelengths.size == 0 or np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("response wavelengths must be a non-empty positive finite set")
    magnitude = 2.0 * math.pi / wavelengths
    transfer = np.asarray(spec.transfer(magnitude, dt_s=dt_s), dtype=np.float64)
    rows = []
    for wavelength, gain in zip(wavelengths, transfer, strict=True):
        damping = 1.0 - float(gain)
        calls_to_efold = (math.inf if gain >= 1.0
                          else -1.0 / math.log(max(float(gain), 1e-300)))
        rows.append({
            "wavelength_m": float(wavelength),
            "amplitude_gain_per_call": float(gain),
            "amplitude_damping_percent_per_call": 100.0 * damping,
            "calls_to_e_fold": float(calls_to_efold),
        })
    return rows


__all__ = ["hyperdiffusion_response"]
