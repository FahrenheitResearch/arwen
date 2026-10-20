"""Shared source-pressure receipt validation without preparation imports.

Both producers and prepared forecast readers use this contract. New source
ladders are metadata; the validation path is independent of the source adapter.
"""
from __future__ import annotations

import math
from typing import Mapping
import numpy as np

CERTIFIED_PRESSURE_LEVELS_HPA = {"GFS": np.asarray(
    [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
     650, 700, 750, 800, 850, 900, 925, 950, 975, 1000],
    dtype=np.float64,
)}


def validate_ladder(levels: np.ndarray, context: str, *, source: str) -> np.ndarray:
    """The certified ladder, optionally extended upward.  Or a refusal.

    Mirrors ``gfs_grib2_bridge::observed_pressure_levels`` on the Python
    side of the same handoff, so the two ends of the decode cannot
    disagree about what a legal ladder is.  Two clauses say it all: the
    ladder is strictly increasing, and it ENDS with the certified 21
    exactly.  Together those force every extra level strictly above the
    certified top -- an increasing list whose last 21 begin at 100 hPa
    cannot carry anything at or below 100 hPa ahead of them -- so an
    interior insertion, a dropped certified level, and a reordering are
    all refused without a third clause that could never fire.
    """

    certified = CERTIFIED_PRESSURE_LEVELS_HPA[source]
    if levels.ndim != 1 or levels.size < certified.size:
        raise ValueError(
            f"{context}: {levels.size} isobaric level(s), fewer than the "
            f"{certified.size} every {source} subset carries")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError(
            f"{context}: isobaric ladder is not strictly increasing")
    tail = levels[-certified.size:]
    if not np.allclose(tail, certified, rtol=0.0,
                       atol=1e-6):
        raise ValueError(
            f"{context}: isobaric ladder does not end with the certified "
            f"{certified.size}-level ladder")
    return levels


def manifest_source_top_pa(manifest: Mapping[str, object], *, source: str) -> float:
    """The source top the fetch receipt declares, in Pa.

    The vertical contract is checked BEFORE the bridge runs, so the model
    top a case may ask for has to come from the receipt rather than from
    a constant -- a constant is exactly what capped every GFS run at
    10000 Pa regardless of what was actually fetched.
    """

    default_top = float(CERTIFIED_PRESSURE_LEVELS_HPA[source].min() * 100.0)
    context = f"{source} input manifest"
    identity = manifest.get("source")
    if not isinstance(identity, dict):
        return default_top
    levels = identity.get("pressure_levels_hpa")
    declared = identity.get("top_pressure_pa")
    if not isinstance(levels, list) or not levels:
        return default_top
    try:
        ladder = np.asarray([float(level) for level in levels],
                            dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context} declares an unreadable pressure ladder"
        ) from error
    validate_ladder(ladder, context, source=source)
    top = float(ladder.min() * 100.0)
    if isinstance(declared, (int, float)) and not math.isclose(
            float(declared), top, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"{context} declares a source top of {float(declared):g} "
            f"Pa but its pressure ladder tops out at {top:g} Pa")
    return top
