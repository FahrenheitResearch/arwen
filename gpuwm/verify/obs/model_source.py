"""Reading a forecast the same way whichever model wrote it.

Both engines in the three-way write WRF history files, and both are read
through this one class.  That is not a convenience: if two readers were used,
any difference between the arms could be a difference between the readers,
and no amount of care downstream could separate the two.

**The mandated science core does the science.**  Surface diagnostics
(``t2``, ``dp2m``, ``wspd10``, ``slp``) and the independent reflectivity
cross-check (``maxdbz``) come from ``wrf-rust``, at the version this tree
pins.  Nothing here reimplements a diagnostic that core provides.

Composite reflectivity for the *primary* score is deliberately different, and
the difference is the point: it is the column maximum of the stored
``REFL_10CM``, i.e. each model's own scheme-consistent reflectivity operator,
which is the fair cross-model choice.  Taking a maximum along k over a stored
variable is a reduction, not a diagnostic -- recomputing reflectivity from
the microphysics would substitute one operator for two, which is exactly what
the primary must not do.  The core's ``maxdbz`` is scored beside it as the
independent cross-check and never as the primary.

Units are converted once, here, into the seam's SI, from a declared table --
never inferred.  Every converted field is then range-checked against
:data:`~gpuwm.verify.obs.contracts.SEAM_BOUNDS`, so a wrong entry in that
table fails loudly on the first frame instead of quietly shifting a bias by
273.15.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from gpuwm.verify import field_metrics
from gpuwm.verify.obs.contracts import (
    SEAM_BOUNDS, SURFACE_UNITS, ModelGrid, format_valid_time,
    normalize_longitude,
)
from gpuwm.verify.obs.stations import StationPosition

#: The certified science-core version.  Matching the pin elsewhere in the
#: tree on purpose: two pins that can drift apart are one pin too many, and
#: this one is checked at construction rather than at import so a scoring
#: module can be imported in an environment that never opens a forecast.
PINNED_WRF_RUST_VERSION = "0.2.35"

#: Seam variable -> (core diagnostic name, multiplicative factor, offset).
#: The conversion is ``value * factor + offset`` into the seam unit.  Every
#: entry is a declaration a human made and a receipt can quote; none of it is
#: sniffed from the data.
DEFAULT_UNIT_CONVERSIONS: Mapping[str, tuple[str, float, float]] = {
    "temperature_2m": ("t2", 1.0, 0.0),
    "dewpoint_2m": ("dp2m", 1.0, 273.15),
    "wind_speed_10m": ("wspd10", 1.0, 0.0),
    "mslp": ("slp", 100.0, 0.0),
}

_FRAME_RE = re.compile(
    r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})"
    r"(?:\..*)?$")


def require_science_core():
    """Import the mandated core and refuse a version that is not the pin."""
    from importlib import metadata

    try:
        installed = metadata.version("wrf-rust")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - env
        raise RuntimeError(
            "the mandated science core wrf-rust is not installed; the "
            "battery's surface diagnostics and reflectivity cross-check "
            "come from it and are not reimplemented here") from exc
    if installed != PINNED_WRF_RUST_VERSION:
        raise RuntimeError(
            f"wrf-rust version mismatch: the battery is certified against "
            f"{PINNED_WRF_RUST_VERSION}, found {installed}")
    import wrf

    return wrf


def discover_frames(run_directory: str | Path, domain: str
                    ) -> dict[str, Path]:
    """Map each valid time this run wrote for ``domain`` to its history file."""
    root = Path(run_directory)
    found: dict[str, Path] = {}
    for path in sorted(root.glob(f"wrfout_{domain}_*")):
        match = _FRAME_RE.match(path.name)
        if match is None or match.group(1) != domain:
            raise ValueError(f"invalid history-frame name: {path.name}")
        valid = datetime.strptime(
            f"{match.group(2)} {match.group(3)}:{match.group(4)}:"
            f"{match.group(5)}", "%Y-%m-%d %H:%M:%S")
        key = format_valid_time(valid)
        if key in found:
            raise ValueError(f"duplicate {domain} frame at {key}")
        found[key] = path
    if not found:
        raise ValueError(f"{run_directory} holds no {domain} history frames")
    return found


@dataclass(frozen=True)
class ReflectivityOperatorPins:
    """The core's reflectivity options, pinned into the cross-check receipt."""

    use_varint: bool = False
    use_liqskin: bool = False

    def record(self) -> dict[str, object]:
        return {"use_varint": bool(self.use_varint),
                "use_liqskin": bool(self.use_liqskin)}


class WrfHistorySource:
    """A run directory, read as one model-agnostic forecast."""

    def __init__(self, run_directory: str | Path, *, domain: str = "d01",
                 reflectivity_variable: str = "REFL_10CM",
                 precipitation_variables: tuple[str, ...] = ("RAINNC",),
                 unit_conversions: Mapping[str, tuple[str, float, float]]
                 = DEFAULT_UNIT_CONVERSIONS,
                 reflectivity_pins: ReflectivityOperatorPins
                 = ReflectivityOperatorPins()) -> None:
        self._wrf = require_science_core()
        self._directory = Path(run_directory)
        self._domain = str(domain)
        self._reflectivity_variable = str(reflectivity_variable)
        self._precipitation_variables = tuple(precipitation_variables)
        self._conversions = dict(unit_conversions)
        self._pins = reflectivity_pins
        self._frames = discover_frames(self._directory, self._domain)
        self._handles: dict[str, object] = {}

    # -- geometry -----------------------------------------------------------

    def _handle(self, valid_time: str):
        path = self.frame_path(valid_time)
        handle = self._handles.get(str(valid_time))
        if handle is None:
            handle = self._wrf.WrfFile(str(path))
            self._handles[str(valid_time)] = handle
        return handle

    def frame_path(self, valid_time: str) -> Path:
        path = self._frames.get(str(valid_time))
        if path is None:
            raise ValueError(
                f"{self._directory.name} has no {self._domain} frame at "
                f"{valid_time}")
        return path

    def valid_times(self) -> tuple[str, ...]:
        return tuple(sorted(self._frames))

    def grid(self) -> ModelGrid:
        first = self.valid_times()[0]
        handle = self._handle(first)
        latitude, longitude = self._wrf.latlon_coords(handle)
        terrain = np.asarray(
            self._wrf.getvar(handle, "terrain", meta=False), dtype=np.float64)
        return ModelGrid(
            latitude=np.asarray(latitude, dtype=np.float64),
            longitude=normalize_longitude(np.asarray(longitude)),
            dx_m=float(handle.dx),
            terrain_m=terrain)

    def station_locator(self) -> Callable[[float, float], StationPosition]:
        """A station-to-grid mapping through the core's own projection code.

        Returned as a callable so the caller never has to hold a file handle,
        and so the one projection implementation in play is the core's.
        """
        handle = self._handle(self.valid_times()[0])

        def locate(station_id: str, latitude: float, longitude: float
                   ) -> StationPosition:
            x, y = self._wrf.ll_to_xy(handle, float(latitude), float(longitude))
            return StationPosition(station_id=str(station_id),
                                   x=float(np.asarray(x).reshape(-1)[0]),
                                   y=float(np.asarray(y).reshape(-1)[0]))

        return locate

    # -- fields -------------------------------------------------------------

    def composite_reflectivity(self, valid_time: str) -> np.ndarray:
        """Column max of the stored reflectivity -- the model's own operator."""
        field = field_metrics.read_frame_field(
            self.frame_path(valid_time), self._reflectivity_variable)
        return np.asarray(field_metrics.composite(field), dtype=np.float64)

    def core_maxdbz(self, valid_time: str) -> np.ndarray:
        """The core's independent column-max reflectivity, for the cross-check."""
        values = self._wrf.getvar(
            self._handle(valid_time), "maxdbz", meta=False,
            use_varint=self._pins.use_varint,
            use_liqskin=self._pins.use_liqskin)
        return np.asarray(values, dtype=np.float64)

    def surface_field(self, valid_time: str, variable: str) -> np.ndarray:
        """One surface variable, from the core, converted into seam units."""
        if variable not in self._conversions:
            raise ValueError(
                f"no declared conversion for {variable!r}; the seam knows "
                f"{sorted(SURFACE_UNITS)} and every one needs a table entry")
        name, factor, offset = self._conversions[variable]
        raw = np.asarray(
            self._wrf.getvar(self._handle(valid_time), name, meta=False),
            dtype=np.float64)
        values = raw * float(factor) + float(offset)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{variable} from {name} carries non-finite values")
        low, high = SEAM_BOUNDS[variable]
        if float(values.min()) < low or float(values.max()) > high:
            raise ValueError(
                f"{variable} from core diagnostic {name!r} spans "
                f"[{values.min():g}, {values.max():g}] after the declared "
                f"conversion, outside the seam bound [{low:g}, {high:g}] "
                f"{SURFACE_UNITS[variable]}; the conversion table entry is "
                f"wrong and the cross-reader receipt is what settles it")
        return values

    def land_mask(self) -> np.ndarray:
        """Land points of the scored domain, as a bool grid.

        The registered station admission keeps land points only, so freezing
        a station set needs this as much as it needs terrain -- and reading
        it off the run rather than off a static file is what keeps the
        admission a property of the arm being scored.  ``LANDMASK`` is a WRF
        time-invariant field written to every history frame by both writers,
        so the first frame answers for the run.
        """
        first = self.valid_times()[0]
        values = np.asarray(
            field_metrics.read_frame_field(self.frame_path(first), "LANDMASK"),
            dtype=np.float64)
        if values.ndim == 3:
            values = values[0]
        if not np.all(np.isfinite(values)):
            raise ValueError("LANDMASK carries non-finite values")
        unique = np.unique(values)
        if not np.all(np.isin(unique, (0.0, 1.0))):
            raise ValueError(
                f"LANDMASK takes values {unique[:6]}, not the 0/1 flag the "
                f"station admission reads it as")
        return values >= 0.5

    def precipitation_accumulation(self, valid_time: str) -> np.ndarray:
        """Run-total precipitation on the model grid, mm."""
        total: np.ndarray | None = None
        for name in self._precipitation_variables:
            field = np.asarray(
                field_metrics.read_frame_field(
                    self.frame_path(valid_time), name), dtype=np.float64)
            total = field if total is None else total + field
        if total is None:
            raise ValueError("no precipitation variables are declared")
        if np.any(total < 0.0):
            raise ValueError("accumulated precipitation is negative")
        return total

    def record(self) -> dict[str, object]:
        """What the score file says about how this run was read."""
        return {
            "run_directory": self._directory.name,
            "domain": self._domain,
            "frame_count": len(self._frames),
            "reflectivity_variable": self._reflectivity_variable,
            "reflectivity_reduction": "column maximum over k",
            "precipitation_variables": list(self._precipitation_variables),
            "science_core": "wrf-rust",
            "science_core_version": PINNED_WRF_RUST_VERSION,
            "unit_conversions": {
                key: {"diagnostic": value[0], "factor": value[1],
                      "offset": value[2]}
                for key, value in sorted(self._conversions.items())},
            "cross_check_operator_pins": self._pins.record(),
        }


__all__ = [
    "DEFAULT_UNIT_CONVERSIONS", "PINNED_WRF_RUST_VERSION",
    "ReflectivityOperatorPins", "WrfHistorySource", "discover_frames",
    "require_science_core",
]
