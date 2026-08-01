"""Real-data ingest (ungrib/metgrid/real equivalent) package.

The public convenience exports are lazy.  In particular, importing a CPU-only
decoder or the mapped-contract authoring CLI must not import ``real`` and its
CuPy-backed model state before a GPU operation is requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .grib import Era5Snapshot, decode_era5_grib
    from .horiz import HorizontalSnapshot, interpolate_era5_to_lambert
    from .hrrr import (
        HrrrNativeSnapshot,
        interpolate_hrrr_to_lambert,
        load_hrrr_native_series,
        load_hrrr_native_window,
    )
    from .lateral_bc import (
        StateBoundaryFrames,
        attach_lateral_boundaries,
        build_state_lateral_boundaries,
    )
    from .real import RealInitResult, initialize_real
    from .soil import NoahSoilState, preprocess_noah_soil


_LAZY_EXPORTS = {
    "Era5Snapshot": (".grib", "Era5Snapshot"),
    "decode_era5_grib": (".grib", "decode_era5_grib"),
    "HorizontalSnapshot": (".horiz", "HorizontalSnapshot"),
    "interpolate_era5_to_lambert": (
        ".horiz", "interpolate_era5_to_lambert"
    ),
    "HrrrNativeSnapshot": (".hrrr", "HrrrNativeSnapshot"),
    "interpolate_hrrr_to_lambert": (
        ".hrrr", "interpolate_hrrr_to_lambert"
    ),
    "load_hrrr_native_series": (".hrrr", "load_hrrr_native_series"),
    "load_hrrr_native_window": (".hrrr", "load_hrrr_native_window"),
    "StateBoundaryFrames": (".lateral_bc", "StateBoundaryFrames"),
    "attach_lateral_boundaries": (
        ".lateral_bc", "attach_lateral_boundaries"
    ),
    "build_state_lateral_boundaries": (
        ".lateral_bc", "build_state_lateral_boundaries"
    ),
    "RealInitResult": (".real", "RealInitResult"),
    "initialize_real": (".real", "initialize_real"),
    "NoahSoilState": (".soil", "NoahSoilState"),
    "preprocess_noah_soil": (".soil", "preprocess_noah_soil"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "Era5Snapshot",
    "HorizontalSnapshot",
    "HrrrNativeSnapshot",
    "NoahSoilState",
    "RealInitResult",
    "StateBoundaryFrames",
    "attach_lateral_boundaries",
    "build_state_lateral_boundaries",
    "decode_era5_grib",
    "initialize_real",
    "interpolate_era5_to_lambert",
    "interpolate_hrrr_to_lambert",
    "load_hrrr_native_series",
    "load_hrrr_native_window",
    "preprocess_noah_soil",
]
