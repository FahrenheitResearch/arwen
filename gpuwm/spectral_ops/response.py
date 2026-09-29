"""Human- and machine-readable transfer-response tables."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .transfer import Hyperdiffusion


def hyperdiffusion_response(spec: Hyperdiffusion, *, dt_s: float,
                            wavelengths_m: Iterable[float]) -> list[dict[str, float | None]]:
    """One row per wavelength: gain, damping and calls to an e-fold decrease.

    A wavelength the operator leaves untouched (gain one: longer than the
    protected scale, a zero damping ceiling, a zero step, or a gain that
    rounds to one) never reaches an e-fold decrease, and its
    ``calls_to_e_fold`` is ``None``.  It used to be ``math.inf``, which
    no JSON document can carry: ``gpuwm spectral-op response`` writes
    these rows with ``allow_nan=False`` and so refused every table that
    held one undamped row -- a 6 km reference with the default sampled
    range to 3000 km is one, and so is any protected scale or a zero
    damping ceiling.
    """
    wavelengths = np.asarray(list(wavelengths_m), dtype=np.float64)
    if wavelengths.size == 0 or np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("response wavelengths must be a non-empty positive finite set")
    magnitude = 2.0 * math.pi / wavelengths
    transfer = np.asarray(spec.transfer(magnitude, dt_s=dt_s), dtype=np.float64)
    rows = []
    for wavelength, gain in zip(wavelengths, transfer, strict=True):
        damping = 1.0 - float(gain)
        calls_to_efold = (None if gain >= 1.0
                          else -1.0 / math.log(max(float(gain), 1e-300)))
        rows.append({
            "wavelength_m": float(wavelength),
            "amplitude_gain_per_call": float(gain),
            "amplitude_damping_percent_per_call": 100.0 * damping,
            "calls_to_e_fold": calls_to_efold,
        })
    return rows


__all__ = ["hyperdiffusion_response"]
