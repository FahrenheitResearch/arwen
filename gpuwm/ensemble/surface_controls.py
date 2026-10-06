"""Surface-state recipe controls, available to preparation-only planners."""
from __future__ import annotations

import math
from numbers import Real
import struct

KIND = "surface-state"
OPTIONS = ("soil_moisture_scale", "sst_offset_k")
DEFAULTS = {"soil_moisture_scale": 1.0, "sst_offset_k": 0.0}


def is_surface_recipe(value):
    """Recognize the named descriptor without importing numerical code."""
    return isinstance(value, dict) and value.get("kind") == KIND


def _float32(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"surface-state {name} must contain finite FP32 numbers")
    try:
        resolved = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error):
        raise ValueError(f"surface-state {name} must contain finite FP32 numbers") from None
    if not math.isfinite(resolved):
        raise ValueError(f"surface-state {name} must contain finite FP32 numbers")
    return resolved


def validate_surface_recipe(value):
    """Validate and detach named recipe options before GPU acquisition."""
    if not is_surface_recipe(value):
        raise ValueError("surface perturbation needs kind = 'surface-state'")
    unknown = set(value) - {"kind", *OPTIONS}
    if unknown:
        raise ValueError(f"unknown surface-state options: {sorted(unknown)}")
    if not any(name in value for name in OPTIONS):
        raise ValueError("surface-state needs soil_moisture_scale or sst_offset_k")
    normalized = {"kind": KIND}
    for name in OPTIONS:
        if name not in value:
            continue
        selected = value[name]
        if isinstance(selected, (tuple, list)):
            if len(selected) != 2:
                raise ValueError(f"surface-state {name} interval needs [minimum, maximum]")
            selected = [_float32(item, name) for item in selected]
            if selected[1] < selected[0]:
                raise ValueError(f"surface-state {name} maximum is below its minimum")
            low = selected[0]
        else:
            selected = _float32(selected, name)
            low = selected
        if name == "soil_moisture_scale" and low <= 0.0:
            raise ValueError("surface-state soil_moisture_scale must be positive: "
                             "zero would create a dry profile outside the land scheme's input contract")
        normalized[name] = selected
    return normalized


def shared_surface_options(value, count):
    """A repeated trajectory needs a varying named surface realization."""
    options = validate_surface_recipe(value)
    if count > 1 and not any(isinstance(item, list) and item[0] != item[1]
                             for key, item in options.items() if key != "kind"):
        raise ValueError("surface-state members reuse one source trajectory and need a "
                         "nonconstant soil_moisture_scale or sst_offset_k interval; "
                         "fixed values would create identical copies of one forecast")
    return options
