"""ArWen history as columns: what one wrfout frame knows per grid column.

Everything ``gpuwm cells`` samples -- reflectivity for titan, and the
updraft, cloud, and thermodynamic attributes the catalog adds -- comes
from the wrfout NetCDF a run wrote.  This module reads one frame of it
into plain numpy arrays with their heights, and that is all it does:
no interpolation, no object, no unit other than the file's own.

The decode is the Rust ``rw_netcdf`` bridge through
:mod:`gpuwm.netcdf_bridge`, which is the one NetCDF reader this project
has (there is no Python NetCDF fallback anywhere in ``gpuwm``, and a
second decoder here would be a second implementation of the same
bytes).  The bridge promotes every numeric variable to float64; the
arrays here are narrowed to float32 once decoded, because a 3 km
domain's eight 3-D fields at float64 is half a gigabyte per frame and
nothing downstream needs more than the file's own single precision.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from gpuwm import netcdf_bridge

#: Gravity, as WRF spells it when it turns geopotential into height.
G = 9.81

#: WRF's dry-air gas constant and the specific heat it derives from it.
R_D = 287.0
CP = 7.0 * R_D / 2.0
P0 = 100000.0

#: The variables one frame needs, and why.  Every one is a stock WRF
#: history variable; a wrfout without one is refused by name.
REQUIRED = {
    "REFL_10CM": "simulated 10 cm reflectivity, dBZ, on mass levels",
    "PH": "perturbation geopotential, on w levels",
    "PHB": "base-state geopotential, on w levels",
    "T": "perturbation potential temperature (theta - 300 K)",
    "P": "perturbation pressure, Pa",
    "PB": "base-state pressure, Pa",
    "W": "vertical velocity, m/s, on w levels",
    "QVAPOR": "water vapour mixing ratio, kg/kg",
    "QCLOUD": "cloud water mixing ratio, kg/kg",
    "QICE": "cloud ice mixing ratio, kg/kg",
    "XLAT": "latitude of mass points, degrees",
    "XLONG": "longitude of mass points, degrees",
    "HGT": "terrain height, m",
}

#: Global attributes copied into the volume's projection string, so a
#: consumer of the titan bundle can put its metres back on the map.
PROJECTION_ATTRIBUTES = (
    "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT", "CEN_LON",
    "MOAD_CEN_LAT", "POLE_LAT", "POLE_LON", "DX", "DY")

#: A wrfout name's valid stamp, as WRF writes it and as Windows keeps it
#: (colons become dashes or underscores when a file crosses over).
_STAMP = re.compile(
    r"wrfout_(?P<domain>d\d\d)_(?P<date>\d{4}-\d{2}-\d{2})"
    r"[_T](?P<hour>\d{2})[:_-](?P<minute>\d{2})[:_-](?P<second>\d{2})")


class ColumnsError(RuntimeError):
    """A wrfout that cannot supply the columns, named by what it lacks."""


@dataclass
class ColumnFrame:
    """One valid time of one wrfout, as arrays.

    Heights are metres above sea level.  ``z_mass`` is ``(nz, ny, nx)``
    at the mass levels every 3-D scalar sits on; ``z_w`` and ``w`` are
    ``(nz + 1, ny, nx)`` at the staggered w levels.
    """

    path: Path
    time_index: int
    valid: dt.datetime
    domain: str | None
    grid_id: int | None
    dx_m: float
    dy_m: float
    lat: np.ndarray
    lon: np.ndarray
    terrain_m: np.ndarray
    z_mass: np.ndarray
    z_w: np.ndarray
    refl_dbz: np.ndarray
    w_mps: np.ndarray
    qcloud: np.ndarray
    qice: np.ndarray
    qvapor: np.ndarray
    temperature_k: np.ndarray
    pressure_pa: np.ndarray
    projection: dict[str, float | int | str]

    @property
    def nz(self) -> int:
        return int(self.z_mass.shape[0])

    @property
    def ny(self) -> int:
        return int(self.z_mass.shape[1])

    @property
    def nx(self) -> int:
        return int(self.z_mass.shape[2])

    @property
    def timestamp_ms(self) -> int:
        return int(round(self.valid.timestamp() * 1000.0))

    @property
    def stamp(self) -> str:
        return self.valid.strftime("%Y-%m-%d %H:%M:%S UTC")

    def density_kg_m3(self) -> np.ndarray:
        """Moist air density on mass levels, from the ideal gas law."""

        virtual = self.temperature_k * (1.0 + 0.61 * self.qvapor)
        return (self.pressure_pa / (R_D * virtual)).astype(np.float32)

    def layer_thickness_m(self) -> np.ndarray:
        """Mass-level layer thickness: the w-level interfaces' spacing."""

        return np.diff(self.z_w, axis=0).astype(np.float32)


def stamp_from_name(path: Path) -> tuple[str | None, dt.datetime | None]:
    """``(domain, valid)`` read from a wrfout filename, or Nones."""

    match = _STAMP.search(Path(path).name)
    if match is None:
        return None, None
    try:
        valid = dt.datetime.strptime(
            f"{match.group('date')} {match.group('hour')}:"
            f"{match.group('minute')}:{match.group('second')}",
            "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return match.group("domain"), None
    return match.group("domain"), valid


def _parse_wrf_stamp(text: str) -> dt.datetime | None:
    text = str(text).strip()
    for pattern in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(text, pattern).replace(
                tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


def _valid_times(dataset, path: Path, count: int) -> list[dt.datetime]:
    """The frame instants: ``SIMULATION_START_DATE`` + ``XTIME`` minutes.

    ``Times`` is a character variable, and the bridge decodes numerics
    only; the numeric spelling of the same instant is ``XTIME`` (minutes
    since the simulation start, a global attribute).  A file without
    either falls back to the stamp in its own name, which WRF writes as
    the first frame's valid time.
    """

    start = None
    attrs = dataset.global_attributes
    for name in ("SIMULATION_START_DATE", "START_DATE"):
        if name in attrs:
            start = _parse_wrf_stamp(attrs[name])
            if start is not None:
                break
    xtime = dataset.variables.get("XTIME")
    if start is not None and xtime is not None:
        minutes = np.asarray(xtime[:], dtype=np.float64).reshape(-1)
        if minutes.size == count:
            return [start + dt.timedelta(minutes=float(m)) for m in minutes]
    _domain, named = stamp_from_name(path)
    if named is None:
        raise ColumnsError(
            f"{path}: no valid time can be read -- it has no "
            f"SIMULATION_START_DATE+XTIME pair and its name carries no "
            f"wrfout_dNN_YYYY-MM-DD_HH:MM:SS stamp; titan needs a "
            f"timestamp per frame to order and track them")
    if count != 1:
        raise ColumnsError(
            f"{path}: holds {count} frames but only its filename stamp is "
            f"readable, which dates the first frame alone")
    return [named]


def _read3(dataset, name: str, time_index: int) -> np.ndarray:
    values = dataset.variables[name][time_index]
    return np.ascontiguousarray(values, dtype=np.float32)


def open_frames(path: Path | str, *, executable: Path | None = None
                ) -> Iterator[ColumnFrame]:
    """Every frame in one wrfout, in file order.

    Reads are per variable through the bridge and per frame into the
    dataclass, so the peak is one frame's arrays plus the bridge's own
    decode of one variable.
    """

    path = Path(path)
    with netcdf_bridge.open_dataset(path, executable=executable) as dataset:
        missing = [name for name in REQUIRED if name not in dataset.variables]
        if missing:
            raise ColumnsError(
                f"{path} is missing {', '.join(missing)}; gpuwm cells reads "
                + "; ".join(f"{name} ({REQUIRED[name]})" for name in missing)
                + " and cannot build a cell volume without them")
        count = int(dataset.variables["REFL_10CM"].shape[0])
        times = _valid_times(dataset, path, count)
        attrs = dataset.global_attributes
        dx = float(attrs["DX"]) if "DX" in attrs else float("nan")
        dy = float(attrs["DY"]) if "DY" in attrs else dx
        grid_id = int(attrs["GRID_ID"]) if "GRID_ID" in attrs else None
        domain = f"d{grid_id:02d}" if grid_id is not None else stamp_from_name(path)[0]
        projection: dict[str, float | int | str] = {}
        for name in PROJECTION_ATTRIBUTES:
            if name in attrs:
                value = attrs[name]
                projection[name] = (int(value) if name == "MAP_PROJ"
                                    else float(value))
        lat = np.ascontiguousarray(dataset.variables["XLAT"][0], dtype=np.float64)
        lon = np.ascontiguousarray(dataset.variables["XLONG"][0], dtype=np.float64)
        terrain = np.ascontiguousarray(dataset.variables["HGT"][0], dtype=np.float32)
        for index in range(count):
            geopotential = (np.asarray(dataset.variables["PH"][index], dtype=np.float64)
                            + np.asarray(dataset.variables["PHB"][index], dtype=np.float64))
            z_w = (geopotential / G).astype(np.float32)
            z_mass = (0.5 * (z_w[:-1] + z_w[1:])).astype(np.float32)
            pressure = (np.asarray(dataset.variables["P"][index], dtype=np.float64)
                        + np.asarray(dataset.variables["PB"][index], dtype=np.float64))
            theta = np.asarray(dataset.variables["T"][index], dtype=np.float64) + 300.0
            temperature = theta * (pressure / P0) ** (R_D / CP)
            yield ColumnFrame(
                path=path, time_index=index, valid=times[index],
                domain=domain, grid_id=grid_id, dx_m=dx, dy_m=dy,
                lat=lat, lon=lon, terrain_m=terrain,
                z_mass=z_mass, z_w=z_w,
                refl_dbz=_read3(dataset, "REFL_10CM", index),
                w_mps=_read3(dataset, "W", index),
                qcloud=_read3(dataset, "QCLOUD", index),
                qice=_read3(dataset, "QICE", index),
                qvapor=_read3(dataset, "QVAPOR", index),
                temperature_k=temperature.astype(np.float32),
                pressure_pa=pressure.astype(np.float32),
                projection=projection)


def partition_inputs(paths, *, executable: Path | None = None
                     ) -> tuple[list[Path], list[dict]]:
    """``(usable, skipped)``: the files that carry ``REFL_10CM``, and why not.

    The initial frame of a run is written before the microphysics has
    produced a reflectivity field; it is not a cell frame, and refusing
    the whole series for it would refuse every run named by its
    directory.  One rule, used by the exporter and the catalog alike so
    the two walk the same frames.  The check is the bridge's metadata
    pass only; nothing is decoded here.
    """

    usable: list[Path] = []
    skipped: list[dict] = []
    for path in paths:
        with netcdf_bridge.open_dataset(path, executable=executable) as dataset:
            if "REFL_10CM" in dataset.variables:
                usable.append(path)
            else:
                skipped.append({"path": str(path),
                                "reason": "no REFL_10CM variable"})
    return usable, skipped


def expand_inputs(items) -> list[Path]:
    """wrfout paths from files, directories and globs, sorted by name.

    A directory contributes every ``wrfout*`` directly inside it; a
    pattern contributes its matches; a file contributes itself.  Sorted
    by name because WRF's stamp sorts chronologically, and the order
    the engine sees is the order it tracks in.
    """

    found: list[Path] = []
    for item in items:
        text = str(item)
        path = Path(text)
        if path.is_dir():
            found.extend(p for p in path.iterdir()
                         if p.is_file() and p.name.startswith("wrfout"))
        elif any(ch in text for ch in "*?["):
            base = Path(text).parent if Path(text).parent != Path("") else Path(".")
            matched = [p for p in base.glob(Path(text).name) if p.is_file()]
            if not matched:
                raise ColumnsError(
                    f"{text} matches no file; the pattern is read by gpuwm "
                    f"itself, so spell the directory the way this system "
                    f"does (a drive-letter path on Windows)")
            found.extend(matched)
        elif path.is_file():
            found.append(path)
        else:
            raise ColumnsError(
                f"{text} is not a wrfout file, a directory of them, or a "
                f"pattern that matches any")
    unique = sorted({p.resolve() for p in found}, key=lambda p: p.name)
    if not unique:
        raise ColumnsError(
            "no wrfout files were named; gpuwm cells needs at least one "
            "frame of history to identify a cell in")
    return unique


__all__ = ["ColumnFrame", "ColumnsError", "REQUIRED", "expand_inputs",
           "open_frames", "partition_inputs", "stamp_from_name"]
