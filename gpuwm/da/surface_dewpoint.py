"""Transport for the pinned native Td(K) to Q2 mixing-ratio operator."""
from __future__ import annotations

import ctypes

import numpy as np

from gpuwm import obs_score_bridge as bridge

CONTRACT = "liquid-dewpoint-q2-mixing-ratio-v1"
PRESSURE_SOURCE = "fixed-member-order ensemble-mean forecast PSFC in Pa"
SATURATION_CONSTANTS = {"e0_pa": 611.2, "a": 17.67, "b_c": 243.5,
                        "kelvin_zero": 273.15, "epsilon": 0.622}


def require_native_operator():
    """Resolve the optional native export before spending a forecast leg."""
    library = bridge.load()
    try:
        function = library.gpuwm_obsscore_surface_dewpoint_q2
    except AttributeError as exc:
        raise bridge.ObsScoreBridgeError(
            "obs-score has no pinned surface dewpoint operator; rebuild tools/rustwx "
            "with cargo build --release -p obs-score --offline") from exc
    f64 = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [f64, f64, ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_double, ctypes.c_double, f64, f64, f64]
    function.restype = ctypes.c_int32
    return library, function


def dewpoint_to_q2(td_k, member_psfc_pa, sigma_td_k: float,
                   error_inflation: float = 1.0):
    """Return observed Q2, pointwise Q2 sigma and the pressure proxy used.

    ``td_k`` is [points]; ``member_psfc_pa`` is [members, points]. Q2 is
    kg water per kg dry air. Rust owns both the fixed-order pressure mean
    and saturation/sigma arithmetic. No pressure error is propagated.
    """
    td = bridge._array(td_k)
    pressure = bridge._array(member_psfc_pa)
    if td.ndim != 1 or pressure.ndim != 2 or pressure.shape[1] != td.size:
        raise ValueError("dewpoint requires Td [points] and PSFC [members, points]")
    library, function = require_native_operator()
    q2, sigma, mean_pressure = (np.empty(td.shape, dtype=np.float64) for _ in range(3))
    bridge._check(library, function(
        bridge._ptr(td), bridge._ptr(pressure), td.size, pressure.shape[0],
        float(sigma_td_k), float(error_inflation), bridge._ptr(q2),
        bridge._ptr(sigma), bridge._ptr(mean_pressure)))
    return q2, sigma, mean_pressure
