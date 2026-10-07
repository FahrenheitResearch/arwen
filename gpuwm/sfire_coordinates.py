"""Explicit coordinate authority for native fine-grid fire fields."""
from __future__ import annotations

from collections.abc import Mapping


def fire_coordinate_mode(static: Mapping[str, object]) -> str:
    """Keep ideal metric FX fields metric even when WRF labels them degrees.

    The native ideal initializer writes projected distances into FXLAT and
    FXLONG. Their names and units alone therefore cannot establish geography.
    A sealed Rust static bundle establishes geographic fine coordinates;
    foreign WRF input establishes them through its MAP_PROJ authority.
    """
    present = ("FXLAT" in static, "FXLONG" in static)
    if present[0] != present[1]:
        raise ValueError("SFIRE coordinates require both FXLAT and FXLONG")
    if not any(present):
        return "metric"
    mode = static.get("FIRE_COORDINATE_MODE")
    if mode is not None:
        if mode not in ("metric", "geographic"):
            raise ValueError("FIRE_COORDINATE_MODE must be metric or geographic")
        return str(mode)
    bundle = static.get("_METADATA")
    if isinstance(bundle, Mapping) and bundle.get("grid_spec") is not None:
        return "geographic"
    projection = static.get("MAP_PROJ")
    if projection is not None:
        if isinstance(projection, bool) or int(projection) != projection:
            raise ValueError("SFIRE MAP_PROJ must be the producing integer projection")
        return "metric" if int(projection) == 0 else "geographic"
    raise ValueError("Supplied SFIRE FX coordinates need FIRE_COORDINATE_MODE, "
                     "WRF MAP_PROJ, or a sealed Rust static grid authority")
