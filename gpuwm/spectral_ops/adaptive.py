"""Damping-only recommendations from Level-1 spectral verification receipts.

This module never mutates a forecast configuration and never recommends
amplification.  It turns an observed candidate/reference power excess into a
bounded amplitude transfer, fits a smooth exponential hyperdiffusion response
in log space, and emits a proposal that still requires explicit operator
registration and an A/B campaign.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from .pins import canonical_hash
from .transfer import Hyperdiffusion

CALIBRATION_SCHEMA = "gpuwm.spectral-damping-recommendation/v1"


@dataclass(frozen=True)
class BandObservation:
    wavelength_m: float
    power_ratio: float
    weight: float = 1.0

    def validate(self) -> None:
        if not (math.isfinite(self.wavelength_m) and self.wavelength_m > 0.0):
            raise ValueError("band wavelength must be positive and finite")
        if not (math.isfinite(self.power_ratio) and self.power_ratio > 0.0):
            raise ValueError("band power ratio must be positive and finite")
        if not (math.isfinite(self.weight) and self.weight > 0.0):
            raise ValueError("band weight must be positive and finite")


def desired_amplitude_transfer(power_ratio: float) -> float:
    if not math.isfinite(power_ratio) or power_ratio <= 0.0:
        raise ValueError("power ratio must be positive and finite")
    return min(1.0, 1.0 / math.sqrt(power_ratio))


def fit_hyperdiffusion(observations: Sequence[BandObservation], *, dt_s: float,
                       orders: Iterable[int] = (1, 2, 3, 4, 5, 6),
                       protect_wavelength_m: float | None = None
                       ) -> dict[str, object]:
    """Fit a damping-only exact exponential response by deterministic grid search."""
    if not observations:
        raise ValueError("a damping fit needs one or more band observations")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("calibration dt_s must be positive and finite")
    for item in observations:
        item.validate()
    wavelengths = np.asarray([item.wavelength_m for item in observations])
    target = np.asarray([
        max(desired_amplitude_transfer(item.power_ratio), 1.0e-8)
        for item in observations])
    weights = np.asarray([item.weight for item in observations])
    # The reference wavelength grid is tied to observed physical scales.  Tau
    # spans 1/16 to 256 operator calls, enough to represent nearly no damping
    # through a very strong shadow-mode proposal without an optimizer dependency.
    refs = np.geomspace(float(np.min(wavelengths)) * 0.5,
                        float(np.max(wavelengths)) * 2.0, 96)
    taus = dt_s * np.geomspace(1.0 / 16.0, 256.0, 128)
    best = None
    for order in orders:
        for reference in refs:
            k_ratio = reference / wavelengths
            activation = np.ones_like(wavelengths)
            if protect_wavelength_m is not None:
                if protect_wavelength_m <= reference:
                    continue
                k = 2.0 * np.pi / wavelengths
                k0 = 2.0 * np.pi / protect_wavelength_m
                k1 = 2.0 * np.pi / reference
                t = np.clip((k - k0) / (k1 - k0), 0.0, 1.0)
                activation = 0.5 - 0.5 * np.cos(np.pi * t)
                activation[k <= k0] = 0.0
                activation[k >= k1] = 1.0
            scale = activation * k_ratio ** (2 * int(order))
            for tau in taus:
                predicted = np.exp(-(dt_s / tau) * scale)
                error = float(np.average(
                    (np.log10(np.maximum(predicted, 1e-8)) - np.log10(target)) ** 2,
                    weights=weights))
                record = (error, int(order), float(reference), float(tau), predicted)
                if best is None or record[0] < best[0]:
                    best = record
    assert best is not None
    error, order, reference, tau, predicted = best
    spec = Hyperdiffusion(
        order=order, reference_wavelength_m=reference,
        e_fold_time_s=tau, protect_wavelength_m=protect_wavelength_m,
        maximum_damping_fraction=1.0)
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "status": "proposal-only",
        "rule": "damping only; no power deficit is amplified",
        "dt_s": float(dt_s),
        "observations": [asdict(item) for item in observations],
        "desired_amplitude_transfer": target.tolist(),
        "predicted_amplitude_transfer": predicted.tolist(),
        "log10_mse": float(error),
        "proposed_hyperdiffusion": asdict(spec),
        "required_next_gate": (
            "run in shadow mode, preregister a matched A/B campaign, and admit "
            "only if Level-1 scale/phase metrics, conservation and runtime all pass"),
    }
    payload["recommendation_sha256"] = canonical_hash(payload)
    return payload


__all__ = ["BandObservation", "CALIBRATION_SCHEMA",
           "desired_amplitude_transfer", "fit_hyperdiffusion"]
