"""Lightweight first-call vertical bounds for physics front doors.

This module deliberately lives at package top level and imports no forecast
executor.  The standalone preprocessing distribution needs to reject an
impossible vertical grid without staging CUDA-backed component modules.
Component tests bind these values and layer-count helpers to their runtime
counterparts.
"""

from __future__ import annotations

import math

import numpy as np


MYNN_VERTICAL_LEVEL_BOUNDS = (5, None)
KF_VERTICAL_LEVEL_BOUNDS = (8, 128)
KESSLER_VERTICAL_LEVEL_BOUNDS = (None, 256)
WSM6_VERTICAL_LEVEL_BOUNDS = (2, 80)
THOMPSON_VERTICAL_LEVEL_BOUNDS = (2, 256)
MORRISON_VERTICAL_LEVEL_BOUNDS = (2, 256)
NSSL2_VERTICAL_LEVEL_BOUNDS = (3, 256)

RRTMGP_TOA_PRESSURE_PA = 1.005183574463
WRF_LW_UPPER_DELTA_P_PA = 400.0
MAX_RRTMGP_LAYERS = 128
MAX_LEGACY_LONGWAVE_LAYERS = 128
MAX_LEGACY_SHORTWAVE_LAYERS = 64


def rrtmgp_above_model_layer_counts(
        p_top: float, *,
        pressure_floor: float = RRTMGP_TOA_PRESSURE_PA,
) -> tuple[int, int]:
    """Return WRF's RTE+RRTMGP ``(LW, SW)`` cap-layer counts."""

    if (not math.isfinite(p_top) or not math.isfinite(pressure_floor)
            or p_top < 0.0 or pressure_floor <= 0.0):
        raise ValueError("radiation top pressures must be finite and nonnegative")
    if p_top <= pressure_floor:
        return 0, 0
    lw = int(np.floor(p_top / WRF_LW_UPPER_DELTA_P_PA + 0.5))
    sw = int(0.5 * p_top >= pressure_floor)
    return max(0, lw), sw


def legacy_radiation_layer_counts(
        nz: int, p_top: float,
) -> tuple[int, int]:
    """Return exact first-call WRF legacy-RRTMG ``(LW, SW)`` counts."""

    value = np.float32(
        np.float32(np.float32(p_top) * np.float32("0.01"))
        / np.float32("4.0"))
    raw = float(value)
    rounded = int(raw)
    if raw - rounded >= 0.5:
        rounded += 1
    elif raw - rounded <= -0.5:
        rounded -= 1
    return int(nz) + rounded, int(nz) + 1
