"""Convert one MPAS history frame into a wrfout-shaped frame the renderer takes.

SUPERSEDED -- REFERENCE GENERATOR, NOT A PRODUCT PATH.
The Rust crate ``tools/rustwx/crates/rw-mpas`` (binary ``rw_mpas_convert``)
owns this on the release line and reproduces this file bit for bit: identical
weights digests on both render windows, 404,772,334 float values compared, 127
of 127 rendered PNGs byte-identical.  Receipt:
``evidence/rw-mpas-converter-parity.json``.  This file stays because that
receipt is *about* it -- delete the reference and the parity claim becomes
unreproducible -- but it is kept as the generator of that reference, never as
a route a run takes.  A bare invocation refuses; ``--regenerate-reference``
is the acknowledged door for regenerating the goldens.

The pinned Rust renderer (``rw_wrfbatch``) has no unstructured path.  Its real
named products -- the ones with operational colortables, titles and map
furniture -- come from the real-WRF import route, which needs a structured
grid, ``Times``, ``XLAT``/``XLONG``, and a rank-4 ``T``.  This tool takes an
MPAS history file on an unstructured mesh (163,842 cells for x4.163842),
gathers every field that has a WRF product equivalent onto one of the cached
render windows from ``mpas_resample_weights.py``, and writes a wrfout-shaped
NetCDF frame.

What it does not do: invent a field the MPAS history does not carry.  A WRF
variable with no MPAS source is simply absent, its products report
``missing-fields``, and the absence is recorded in the emitted file's
``MPAS_ABSENT_WRF_FIELDS`` attribute and in the run report.  A missing field
is a port history-stream gap to fix upstream, never something to synthesize
here.

Every emitted file carries the source history digest, the mesh digest, the
weights digest, the resample method, and this tool's own digest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from netCDF4 import Dataset
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import mpas_resample_weights as mrw  # noqa: E402


BRIDGE_SCHEMA = "mpas-port.history-to-wrfout-render-frame/v1"

# WRF's T is a perturbation about this reference potential temperature, and
# wrf-core reconstructs full theta as T + 300 (vendor wrf-core file.rs:465).
WRF_THETA_REFERENCE_K = 300.0
# wrf-core divides full geopotential by exactly this to get height (file.rs:12).
WRF_GRAVITY = 9.80665

_TIMESTAMP = re.compile(
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})[_T]"
    r"(?P<hour>\d{2})[.:](?P<minute>\d{2})[.:](?P<second>\d{2})"
)


class HistoryConversionRefusal(RuntimeError):
    """A history, mesh, weights, or emitted-frame refusal."""


@dataclass(frozen=True)
class FieldMapping:
    """One WRF output variable and where its values come from."""

    wrf_name: str
    mpas_name: str | None
    rank: str                      # "surface" | "mass3d" | "w3d" | "soil"
    units: str
    description: str
    source: str = "history"        # "history" | "init" | "derived" | "constant"
    note: str = ""


# The field map.  Every entry is a WRF variable some product in the renderer's
# catalogue actually reads, paired with the MPAS history (or pinned init)
# variable that carries the same quantity.
FIELD_MAP: tuple[FieldMapping, ...] = (
    # --- three-dimensional mass fields -------------------------------------
    FieldMapping("T", "theta", "mass3d", "K",
                 "perturbation potential temperature theta-t0",
                 note="theta - 300 K; MPAS theta is the dry potential temperature"),
    FieldMapping("PB", "pressure", "mass3d", "Pa", "BASE STATE PRESSURE",
                 note="MPAS has no base/perturbation split; PB carries the full "
                      "model pressure so P+PB is exactly it"),
    FieldMapping("P", None, "mass3d", "Pa", "perturbation pressure",
                 source="constant",
                 note="identically zero; see PB"),
    FieldMapping("QVAPOR", "qv", "mass3d", "kg kg-1", "Water vapor mixing ratio"),
    FieldMapping("QCLOUD", "qc", "mass3d", "kg kg-1", "Cloud water mixing ratio"),
    FieldMapping("QRAIN", "qr", "mass3d", "kg kg-1", "Rain water mixing ratio"),
    FieldMapping("QICE", "qi", "mass3d", "kg kg-1", "Ice mixing ratio"),
    FieldMapping("QSNOW", "qs", "mass3d", "kg kg-1", "Snow mixing ratio"),
    FieldMapping("QGRAUP", "qg", "mass3d", "kg kg-1", "Graupel mixing ratio"),
    FieldMapping("U", "u_zonal", "u3d", "m s-1", "x-wind component",
                 note="earth-relative zonal wind, restaggered in x"),
    FieldMapping("V", "v_meridional", "v3d", "m s-1", "y-wind component",
                 note="earth-relative meridional wind, restaggered in y"),
    FieldMapping("W", "w", "w3d", "m s-1", "z-wind component"),
    FieldMapping("PHB", "zgrid", "w3d", "m2 s-2", "base-state geopotential",
                 source="init",
                 note="g * zgrid from the pinned init file; MPAS is height based"),
    FieldMapping("PH", None, "w3d", "m2 s-2", "perturbation geopotential",
                 source="constant", note="identically zero; see PHB"),
    # --- surface diagnostics ----------------------------------------------
    FieldMapping("T2", "t2", "surface", "K", "TEMP at 2 M"),
    FieldMapping("PSFC", "surface_pressure", "surface", "Pa", "SFC PRESSURE"),
    FieldMapping("U10", "u10", "surface", "m s-1", "U at 10 M"),
    FieldMapping("V10", "v10", "surface", "m s-1", "V at 10 M"),
    FieldMapping("TSK", "tsk", "surface", "K", "SURFACE SKIN TEMPERATURE"),
    FieldMapping("HGT", "ter", "surface", "m", "Terrain Height"),
    FieldMapping("Q2", "q2", "surface", "kg kg-1", "QV at 2 M",
                 note="the surface layer's 2 m vapour mixing ratio.  Native "
                      "MPAS v8.4.1 writes this same diagnostic to its own diag "
                      "stream carrying a few dozen small negative cells out of "
                      "163842 (measured: 46 at 18Z, 74 at 00Z on the node-2 "
                      "24 h x4 run), so a negative Q2 is native behaviour and "
                      "not a port defect.  The port history keeps the raw "
                      "value; this bridge clamps to zero for rendering only "
                      "and counts the clamped cells in "
                      "MPAS_Q2_NEGATIVE_CELLS_CLAMPED"),
    FieldMapping("PBLH", "pblh", "surface", "m", "PBL HEIGHT"),
    FieldMapping("OLR", "olrtoa", "surface", "W m-2",
                 "TOA OUTGOING LONG WAVE",
                 note="native MPAS spells this olrtoa; it is the same "
                      "top-of-atmosphere outgoing longwave WRF calls OLR"),
    FieldMapping("SST", "sst", "surface", "K", "SEA SURFACE TEMPERATURE",
                 note="native MPAS carries sst in its own history stream and "
                      "holds it fixed unless config_sst_update is on, so a "
                      "constant-in-time SST is exactly what native writes for "
                      "this configuration, not a stand-in for a missing field"),
    # --- severe-weather diagnostics ---------------------------------------
    FieldMapping("REFL_10CM", "refl10cm", "mass3d", "dBZ",
                 "Radar reflectivity, 10 cm wavelength",
                 note="the model's own reflectivity from the microphysics "
                      "scheme's PSDs, not the renderer's fallback hydrometeor "
                      "diagnostic; native MPAS computes it in the WSM6 driver "
                      "via refl10cm_wsm6"),
    FieldMapping("UP_HELI_MAX", "updraft_helicity_max", "surface", "m2 s-2",
                 "MAX UPDRAFT HELICITY",
                 note="running maximum since the previous history frame, "
                      "2-5 km AGL, mirroring native "
                      "convective_diagnostics_update"),
    FieldMapping("WSPD10MAX", "wind_speed_10m_max", "surface", "m s-1",
                 "MAX 10 M WIND SPEED",
                 note="running maximum of the surface scheme's own 10 m wind "
                      "since the previous history frame.  Native MPAS declares "
                      "no 10 m running maximum -- its equivalent is "
                      "wind_speed_level1_max at the lowest model level -- so "
                      "this is the WRF-named quantity computed from the port's "
                      "u10/v10 rather than a rename of the native field, which "
                      "the port also carries under its native name"),
    FieldMapping("HFX", "hfx", "surface", "W m-2",
                 "UPWARD HEAT FLUX AT THE SURFACE"),
    FieldMapping("LH", "lh", "surface", "W m-2",
                 "LATENT HEAT FLUX AT THE SURFACE"),
    FieldMapping("QFX", "qfx", "surface", "kg m-2 s-1",
                 "UPWARD MOISTURE FLUX AT THE SURFACE"),
    FieldMapping("SWDOWN", "swdown", "surface", "W m-2",
                 "DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE"),
    FieldMapping("GLW", "glw", "surface", "W m-2",
                 "DOWNWARD LONG WAVE FLUX AT GROUND SURFACE"),
    # --- precipitation accumulations --------------------------------------
    FieldMapping("RAINC", "rainc", "surface", "mm",
                 "ACCUMULATED TOTAL CUMULUS PRECIPITATION"),
    FieldMapping("RAINNC", "rainnc", "surface", "mm",
                 "ACCUMULATED TOTAL GRID SCALE PRECIPITATION"),
    FieldMapping("SNOWNC", "snownc", "surface", "mm",
                 "ACCUMULATED TOTAL GRID SCALE SNOW AND ICE"),
    FieldMapping("GRAUPELNC", "graupelnc", "surface", "mm",
                 "ACCUMULATED TOTAL GRID SCALE GRAUPEL"),
    # --- land surface ------------------------------------------------------
    FieldMapping("TSLB", "tslb", "soil", "K", "SOIL TEMPERATURE"),
    FieldMapping("SMOIS", "smois", "soil", "m3 m-3", "SOIL MOISTURE"),
    FieldMapping("LANDMASK", "landmask", "surface", "",
                 "LAND MASK (1 FOR LAND, 0 FOR WATER)", source="init"),
    FieldMapping("LU_INDEX", "ivgtyp", "surface", "", "LAND USE CATEGORY",
                 source="init"),
    # --- grid geometry the renderer reads ----------------------------------
    FieldMapping("SINALPHA", None, "surface", "", "Local sine of map rotation",
                 source="constant",
                 note="0: the emitted U/V/U10/V10 are already earth-relative, so "
                      "the renderer's grid-to-earth rotation must be the identity"),
    FieldMapping("COSALPHA", None, "surface", "", "Local cosine of map rotation",
                 source="constant", note="1; see SINALPHA"),
)

# WRF variables some catalogue product wants that the port's history stream
# does not carry.  These are gaps to file against the MPAS output stream, not
# things to synthesize.
ABSENT_WRF_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("RAINSH", "shallow-cumulus precipitation accumulation",
     "CLOSED, not a gap.  MPAS v8.4.1 declares no shallow-cumulus "
     "precipitation bucket anywhere in its Registry: the whole tree carries "
     "rainc/raincv and rainnc/rainncv and nothing else, so there is no native "
     "quantity for this bridge to mirror and nothing for the port history to "
     "add.  RAINSH is a WRF-only variable.  Total precipitation from "
     "RAINC + RAINNC is therefore complete for an MPAS-sourced frame, and "
     "emitting an identically-zero RAINSH would state a shallow bucket exists "
     "and measured zero, which is false."),
)

_MMINLU = "MODIFIED_IGBP_MODIS_NOAH"


def sha256_file(path: str | Path) -> str:
    return mrw.sha256_file(path)


def parse_timestamp(text: str) -> datetime:
    match = _TIMESTAMP.search(str(text))
    if match is None:
        raise HistoryConversionRefusal(f"no YYYY-MM-DD_HH.MM.SS timestamp in {text!r}")
    return datetime(
        int(match["year"]), int(match["month"]), int(match["day"]),
        int(match["hour"]), int(match["minute"]), int(match["second"]),
    )


@dataclass(frozen=True)
class MpasFrame:
    """One MPAS history frame plus the pinned static geometry it needs."""

    history_path: Path
    history_sha256: str
    init_path: Path | None
    init_sha256: str | None
    valid_time: datetime
    n_cells: int
    n_levels: int
    n_soil_levels: int
    fields: Mapping[str, np.ndarray]
    latitude_degrees: np.ndarray
    longitude_degrees: np.ndarray
    absent: tuple[str, ...]


def _read_cell_field(dataset: Dataset, name: str) -> np.ndarray | None:
    variable = dataset.variables.get(name)
    if variable is None:
        return None
    variable.set_auto_mask(False)
    value = np.ascontiguousarray(np.asarray(variable[:], dtype=np.float32))
    if value.ndim >= 1 and variable.dimensions and variable.dimensions[0] == "Time":
        if value.shape[0] != 1:
            raise HistoryConversionRefusal(
                f"{name} carries {value.shape[0]} time records; expected exactly 1"
            )
        value = value[0]
    if not np.all(np.isfinite(value)):
        raise HistoryConversionRefusal(f"history {name} is not finite")
    return np.ascontiguousarray(value)


def read_history(
    history_path: str | Path,
    *,
    init_path: str | Path | None,
    valid_time: datetime | None = None,
    history_sha256: str | None = None,
    init_sha256: str | None = None,
) -> MpasFrame:
    history = Path(history_path).expanduser().resolve(strict=True)
    digest = sha256_file(history)
    if history_sha256 is not None and digest != history_sha256:
        raise HistoryConversionRefusal(
            f"history {history} is sha256 {digest}, expected {history_sha256}"
        )
    fields: dict[str, np.ndarray] = {}
    absent: list[str] = []
    with Dataset(history, "r") as dataset:
        dataset.set_auto_mask(False)
        n_cells = len(dataset.dimensions["nCells"])
        n_levels = len(dataset.dimensions["nVertLevels"])
        n_soil = len(dataset.dimensions.get("nSoilLevels", ()))
        for name in ("latCell", "lonCell"):
            if name not in dataset.variables:
                raise HistoryConversionRefusal(f"history has no {name}")
        latitude = np.asarray(dataset.variables["latCell"][:], dtype=np.float64)
        longitude = np.asarray(dataset.variables["lonCell"][:], dtype=np.float64)
        for mapping in FIELD_MAP:
            if mapping.source != "history" or mapping.mpas_name is None:
                continue
            value = _read_cell_field(dataset, mapping.mpas_name)
            if value is None:
                absent.append(mapping.wrf_name)
                continue
            fields[mapping.wrf_name] = value
    init = None
    init_digest = None
    if init_path is not None:
        init = Path(init_path).expanduser().resolve(strict=True)
        init_digest = sha256_file(init)
        if init_sha256 is not None and init_digest != init_sha256:
            raise HistoryConversionRefusal(
                f"init {init} is sha256 {init_digest}, expected {init_sha256}"
            )
        with Dataset(init, "r") as dataset:
            dataset.set_auto_mask(False)
            if len(dataset.dimensions["nCells"]) != n_cells:
                raise HistoryConversionRefusal("init and history disagree on nCells")
            for mapping in FIELD_MAP:
                if mapping.source != "init" or mapping.mpas_name is None:
                    continue
                value = _read_cell_field(dataset, mapping.mpas_name)
                if value is None:
                    absent.append(mapping.wrf_name)
                    continue
                fields[mapping.wrf_name] = value
    else:
        absent.extend(
            mapping.wrf_name for mapping in FIELD_MAP if mapping.source == "init"
        )
    if "PHB" in fields:
        zgrid = fields["PHB"]
        if zgrid.shape != (n_cells, n_levels + 1):
            raise HistoryConversionRefusal(
                f"init zgrid is {zgrid.shape}; expected {(n_cells, n_levels + 1)}"
            )
    resolved_time = valid_time if valid_time is not None else parse_timestamp(history.name)
    latitude_degrees = latitude * (180.0 / np.pi)
    longitude_degrees = np.mod(longitude * (180.0 / np.pi) + 180.0, 360.0) - 180.0
    return MpasFrame(
        history_path=history,
        history_sha256=digest,
        init_path=init,
        init_sha256=init_digest,
        valid_time=resolved_time,
        n_cells=n_cells,
        n_levels=n_levels,
        n_soil_levels=n_soil,
        fields=fields,
        latitude_degrees=np.ascontiguousarray(latitude_degrees),
        longitude_degrees=np.ascontiguousarray(longitude_degrees),
        absent=tuple(absent),
    )


def assert_mesh_agrees(frame: MpasFrame, mesh: mrw.MeshCoordinates,
                       *, tolerance_degrees: float = 1.0e-4) -> dict[str, float]:
    """Refuse unless the history's own cell centres are the weights' mesh.

    This is the guard against the one bug that would silently ruin every map:
    resampling one mesh's fields through another mesh's index map, or through
    a de-rotated copy of a ``grid_rotate`` output.  The rotated grid stores
    post-rotation earth coordinates, so a correct pairing agrees to float32
    round-off; anything else is a different mesh and is refused here rather
    than discovered by eye in a flipped map.
    """

    if mesh.n_cells != frame.n_cells:
        raise HistoryConversionRefusal(
            f"mesh has {mesh.n_cells} cells; history has {frame.n_cells}"
        )
    latitude_error = float(
        np.max(np.abs(mesh.latitude_degrees - frame.latitude_degrees))
    )
    longitude_delta = np.abs(mesh.longitude_degrees - frame.longitude_degrees)
    longitude_delta = np.minimum(longitude_delta, 360.0 - longitude_delta)
    longitude_error = float(np.max(longitude_delta))
    if latitude_error > tolerance_degrees or longitude_error > tolerance_degrees:
        raise HistoryConversionRefusal(
            "history cell centres do not match the weights mesh "
            f"(max |dlat|={latitude_error:g} deg, max |dlon|={longitude_error:g} deg)"
        )
    return {
        "max_latitude_error_degrees": latitude_error,
        "max_longitude_error_degrees": longitude_error,
    }


def _restagger_x(values: np.ndarray) -> np.ndarray:
    """Cell-centre values to x-staggered faces, WRF U layout.

    The renderer destaggers with a two-point average, so the round trip is a
    1/4-1/2-1/4 smoother in x rather than the identity.  That is a real and
    deliberate divergence: WRF's U genuinely lives on faces and there is no
    local face field whose two-point average reproduces an arbitrary
    cell-centre field.  It affects the three-dimensional wind only; the
    surface wind products read the unstaggered U10/V10 and are untouched.
    """

    left = values[..., :-1]
    right = values[..., 1:]
    interior = 0.5 * (left + right)
    return np.ascontiguousarray(
        np.concatenate(
            (values[..., :1], interior, values[..., -1:]), axis=-1
        ).astype(np.float32)
    )


def _restagger_y(values: np.ndarray) -> np.ndarray:
    lower = values[..., :-1, :]
    upper = values[..., 1:, :]
    interior = 0.5 * (lower + upper)
    return np.ascontiguousarray(
        np.concatenate(
            (values[..., :1, :], interior, values[..., -1:, :]), axis=-2
        ).astype(np.float32)
    )


@dataclass(frozen=True)
class EmittedFrame:
    path: Path
    window: str
    shape: tuple[int, int]
    levels: int
    written: tuple[str, ...]
    absent: tuple[str, ...]
    seconds: float
    provenance: Mapping[str, Any]


_FIELD_SETS = {
    "full": None,   # everything in FIELD_MAP
    # A global 0.25 degree window carries four million points per level; the
    # surface set keeps only what the surface, precipitation, MSLP and
    # precipitable-water products read, which is the honest content of a
    # coarse overview anyway.
    "surface": frozenset(
        {"T", "P", "PB", "QVAPOR", "PH", "PHB"}
        | {
            mapping.wrf_name
            for mapping in FIELD_MAP
            if mapping.rank in {"surface", "soil"}
        }
    ),
}


def convert_frame(
    frame: MpasFrame,
    weights: mrw.NearestCellWeights,
    destination: str | Path,
    *,
    simulation_start: datetime | None = None,
    field_set: str = "full",
    grid_id: int = 1,
    title: str = "MPAS-A v8.4.1 CUDA port history resampled for rw_wrfbatch",
    extra_attrs: Mapping[str, Any] | None = None,
    compress: bool = False,
    clobber: bool = False,
) -> EmittedFrame:
    """Write one wrfout-shaped frame.  Returns the emission record."""

    started = time.perf_counter()
    if field_set not in _FIELD_SETS:
        raise HistoryConversionRefusal(f"unknown field set {field_set!r}")
    allowed = _FIELD_SETS[field_set]
    target = Path(destination).expanduser().resolve()
    if target.exists() and not clobber:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    ny, nx = weights.shape
    nz = frame.n_levels
    origin = simulation_start if simulation_start is not None else frame.valid_time
    if origin > frame.valid_time:
        raise HistoryConversionRefusal("simulation start follows the frame valid time")
    lead_seconds = (frame.valid_time - origin).total_seconds()

    projection = weights.window.projection_attributes()
    written: list[str] = []
    absent = list(frame.absent)
    # Q2 is emitted for rendering, and a mixing ratio below zero has no
    # meaning in the products that read it (dewpoint, RH, and every
    # surface-parcel severe index).  Native MPAS itself writes a handful of
    # small negative Q2 cells, so the port history keeps the raw value and the
    # clamp lives here, at the visualization boundary, counted rather than
    # silent.
    q2_clamped_cells = 0

    def gather(mapping: FieldMapping) -> np.ndarray | None:
        if mapping.source == "constant":
            if mapping.wrf_name == "COSALPHA":
                return np.ones((ny, nx), dtype=np.float32)
            if mapping.wrf_name == "SINALPHA":
                return np.zeros((ny, nx), dtype=np.float32)
            if mapping.rank == "mass3d":
                return np.zeros((nz, ny, nx), dtype=np.float32)
            if mapping.rank == "w3d":
                return np.zeros((nz + 1, ny, nx), dtype=np.float32)
            return np.zeros((ny, nx), dtype=np.float32)
        nonlocal q2_clamped_cells
        source = frame.fields.get(mapping.wrf_name)
        if source is None:
            return None
        if mapping.wrf_name == "Q2":
            negative = int(np.count_nonzero(source < 0.0))
            if negative:
                q2_clamped_cells = negative
                source = np.maximum(source, np.float32(0.0))
        # A cell-indexed column field is (nCells, nLevels); the gather moves
        # the cell axis last and replaces it with (ny, nx), which lands the
        # vertical axis first -- exactly the WRF (level, y, x) layout.
        gathered = weights.apply(source, cell_axis=0)
        return np.ascontiguousarray(gathered.astype(np.float32, copy=False))

    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with Dataset(temporary, "w", format="NETCDF4_CLASSIC") as dataset:
            dataset.createDimension("Time", None)
            dataset.createDimension("DateStrLen", 19)
            dataset.createDimension("west_east", nx)
            dataset.createDimension("south_north", ny)
            dataset.createDimension("bottom_top", nz)
            dataset.createDimension("west_east_stag", nx + 1)
            dataset.createDimension("south_north_stag", ny + 1)
            dataset.createDimension("bottom_top_stag", nz + 1)
            if frame.n_soil_levels:
                dataset.createDimension("soil_layers_stag", frame.n_soil_levels)

            attributes: dict[str, Any] = {
                "TITLE": title,
                "SIMULATION_START_DATE": f"{origin:%Y-%m-%d_%H:%M:%S}",
                "START_DATE": f"{origin:%Y-%m-%d_%H:%M:%S}",
                "WEST-EAST_GRID_DIMENSION": np.int32(nx + 1),
                "SOUTH-NORTH_GRID_DIMENSION": np.int32(ny + 1),
                "BOTTOM-TOP_GRID_DIMENSION": np.int32(nz + 1),
                "GRIDTYPE": "C",
                "MMINLU": _MMINLU,
                "ISWATER": np.int32(17),
                "ISLAKE": np.int32(21),
                "ISICE": np.int32(15),
                "ISURBAN": np.int32(13),
                "GRID_ID": np.int32(grid_id),
                "PARENT_ID": np.int32(0),
                "I_PARENT_START": np.int32(1),
                "J_PARENT_START": np.int32(1),
                "PARENT_GRID_RATIO": np.int32(1),
            }
            for key, value in projection.items():
                if key in {"MAP_PROJ", "MAP_PROJ_CHAR"}:
                    attributes[key] = (
                        np.int32(value) if key == "MAP_PROJ" else str(value)
                    )
                else:
                    attributes[key] = np.float32(value)
            provenance = {
                "MPAS_BRIDGE_SCHEMA": BRIDGE_SCHEMA,
                "MPAS_SOURCE": "MPAS-A v8.4.1 full-physics CUDA port",
                "MPAS_HISTORY_FILE": frame.history_path.name,
                "MPAS_HISTORY_SHA256": frame.history_sha256,
                "MPAS_MESH_SHA256": weights.mesh_sha256,
                "MPAS_MESH_FILE": Path(weights.mesh_path).name,
                "MPAS_INIT_SHA256": frame.init_sha256 or "",
                "MPAS_INIT_FILE": frame.init_path.name if frame.init_path else "",
                "MPAS_WEIGHTS_SHA256": weights.weights_sha256,
                "MPAS_WEIGHTS_SCHEMA": weights.schema,
                "MPAS_RESAMPLE_METHOD": weights.method,
                "MPAS_RENDER_WINDOW": weights.window_name,
                "MPAS_RENDER_WINDOW_SPEC": json.dumps(
                    weights.window.spec(), sort_keys=True
                ),
                "MPAS_NATIVE_NCELLS": np.int32(frame.n_cells),
                "MPAS_MAX_NEAREST_CELL_KM": np.float32(
                    float(weights.distance_km.max())
                ),
                "MPAS_FIELD_SET": field_set,
                "MPAS_WIND_FRAME": (
                    "U, V, U10 and V10 are earth-relative; SINALPHA=0 and "
                    "COSALPHA=1 declare no grid rotation so the renderer's "
                    "grid-to-earth rotation is the identity"
                ),
                "MPAS_NONCLAIM": (
                    "nearest-cell visualization resample; not a conservative "
                    "remap and not a forecast-skill claim"
                ),
                "MPAS_BRIDGE_TOOL_SHA256": sha256_file(Path(__file__)),
                "MPAS_WEIGHTS_TOOL_SHA256": sha256_file(Path(mrw.__file__)),
            }
            attributes.update(provenance)
            if extra_attrs:
                attributes.update(dict(extra_attrs))
            dataset.setncatts(attributes)

            times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
            label = f"{frame.valid_time:%Y-%m-%d_%H:%M:%S}"
            if len(label) != 19:
                raise HistoryConversionRefusal(f"time label {label!r} is not 19 bytes")
            times[0] = np.frombuffer(label.encode("ascii"), dtype="S1")

            def create(name: str, dimensions: tuple[str, ...], values: np.ndarray,
                       units: str, description: str, stagger: str) -> None:
                chunked = compress and values.size >= 1_000_000
                variable = dataset.createVariable(
                    name, "f4", dimensions,
                    zlib=chunked, complevel=1, shuffle=chunked,
                )
                variable.setncatts({
                    "FieldType": np.int32(104),
                    "MemoryOrder": "XYZ" if len(dimensions) == 4 else "XY ",
                    "description": description,
                    "units": units,
                    "stagger": stagger,
                    "coordinates": "XLONG XLAT XTIME",
                })
                variable[0] = values.astype(np.float32, copy=False)
                written.append(name)

            create("XLAT", ("Time", "south_north", "west_east"),
                   weights.target_latitude, "degree_north",
                   "LATITUDE, SOUTH IS NEGATIVE", "")
            create("XLONG", ("Time", "south_north", "west_east"),
                   weights.target_longitude, "degree_east",
                   "LONGITUDE, WEST IS NEGATIVE", "")

            for mapping in FIELD_MAP:
                if allowed is not None and mapping.wrf_name not in allowed:
                    continue
                values = gather(mapping)
                if values is None:
                    if mapping.wrf_name not in absent:
                        absent.append(mapping.wrf_name)
                    continue
                if mapping.wrf_name == "T":
                    values = values - np.float32(WRF_THETA_REFERENCE_K)
                elif mapping.wrf_name == "PHB":
                    values = values * np.float32(WRF_GRAVITY)
                if mapping.rank == "mass3d":
                    dimensions = ("Time", "bottom_top", "south_north", "west_east")
                    stagger = ""
                elif mapping.rank == "w3d":
                    dimensions = ("Time", "bottom_top_stag", "south_north",
                                  "west_east")
                    stagger = "Z"
                elif mapping.rank == "u3d":
                    values = _restagger_x(values)
                    dimensions = ("Time", "bottom_top", "south_north",
                                  "west_east_stag")
                    stagger = "X"
                elif mapping.rank == "v3d":
                    values = _restagger_y(values)
                    dimensions = ("Time", "bottom_top", "south_north_stag",
                                  "west_east")
                    stagger = "Y"
                elif mapping.rank == "soil":
                    if not frame.n_soil_levels:
                        absent.append(mapping.wrf_name)
                        continue
                    dimensions = ("Time", "soil_layers_stag", "south_north",
                                  "west_east")
                    stagger = "Z"
                else:
                    dimensions = ("Time", "south_north", "west_east")
                    stagger = ""
                create(mapping.wrf_name, dimensions, values, mapping.units,
                       mapping.description, stagger)

            dataset.setncattr(
                "MPAS_Q2_NEGATIVE_CELLS_CLAMPED", np.int32(q2_clamped_cells)
            )
            dataset.setncattr(
                "MPAS_Q2_CLAMP_NOTE",
                "count of source cells whose 2 m mixing ratio was below zero "
                "and was raised to zero for rendering; the port history file "
                "named in MPAS_HISTORY_FILE keeps the unclamped value, and "
                "native MPAS v8.4.1 writes negative q2 cells of its own",
            )
            dataset.setncattr(
                "MPAS_ZLIB_COMPRESSION", "on-level-1" if compress else "off"
            )

            xtime = dataset.createVariable("XTIME", "f4", ("Time",))
            xtime.setncatts({
                "FieldType": np.int32(104),
                "MemoryOrder": "0  ",
                "description": f"minutes since {origin:%Y-%m-%d %H:%M:%S}",
                "units": f"minutes since {origin:%Y-%m-%d %H:%M:%S}",
                "stagger": "",
            })
            xtime[0] = np.float32(lead_seconds / 60.0)
            itimestep = dataset.createVariable("ITIMESTEP", "i4", ("Time",))
            itimestep.setncatts({
                "FieldType": np.int32(106), "MemoryOrder": "0  ",
                "description": "", "units": "", "stagger": "",
            })
            itimestep[0] = np.int32(0)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    elapsed = time.perf_counter() - started
    record = {
        "schema": BRIDGE_SCHEMA,
        "output": str(target),
        "output_sha256": sha256_file(target),
        "output_bytes": target.stat().st_size,
        "window": weights.window_name,
        "window_spec": weights.window.spec(),
        "projection_attributes": projection,
        "shape": [ny, nx],
        "levels": nz,
        "valid_time": frame.valid_time.isoformat(),
        "simulation_start": origin.isoformat(),
        "lead_seconds": lead_seconds,
        "history_sha256": frame.history_sha256,
        "mesh_sha256": weights.mesh_sha256,
        "init_sha256": frame.init_sha256,
        "weights_sha256": weights.weights_sha256,
        "resample_method": weights.method,
        "field_set": field_set,
        "written_variables": sorted(set(written)),
        "absent_wrf_fields": sorted(set(absent)),
        "convert_seconds": elapsed,
    }
    return EmittedFrame(
        path=target,
        window=weights.window_name,
        shape=(ny, nx),
        levels=nz,
        written=tuple(sorted(set(written))),
        absent=tuple(sorted(set(absent))),
        seconds=elapsed,
        provenance=record,
    )


def field_map_table() -> list[dict[str, str]]:
    rows = [
        {
            "wrf": mapping.wrf_name,
            "mpas": mapping.mpas_name or "(none)",
            "rank": mapping.rank,
            "units": mapping.units,
            "source": mapping.source,
            "note": mapping.note,
        }
        for mapping in FIELD_MAP
    ]
    rows.extend(
        {
            "wrf": name,
            "mpas": "(absent)",
            "rank": "-",
            "units": "-",
            "source": "gap",
            "note": reason,
        }
        for name, _, reason in ABSENT_WRF_FIELDS
    )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, type=Path, nargs="+",
                        help="one or more MPAS history frames")
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--mesh-sha256", default=None)
    parser.add_argument("--init", type=Path, default=None,
                        help="pinned init file supplying zgrid, landmask, ivgtyp")
    parser.add_argument("--init-sha256", default=None)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--window", default="focus", choices=sorted(mrw.WINDOWS))
    parser.add_argument("--field-set", default="full", choices=sorted(_FIELD_SETS))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--simulation-start", default=None,
                        help="run origin as YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--prefix", default="wrfout",
                        help="output basename prefix; the renderer accepts only "
                             "files named wrfout* or *.nc")
    # Uncompressed is the default.  zlib level 1 with shuffle was measured to
    # cost about 22x on write time for about 17% of the file size on these
    # frames; a render frame is written once, read once, and deleted, so the
    # trade is the wrong way round.  --compress restores the old behaviour for
    # anyone archiving frames rather than rendering them.
    parser.add_argument("--compress", action="store_true",
                        help="re-enable zlib level 1 + shuffle on large arrays")
    parser.add_argument("--no-compress", action="store_true",
                        help="accepted and ignored; uncompressed is the default")
    parser.add_argument("--clobber", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--print-field-map", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # The field map is documentation, not a conversion, so it must print
    # without a history, a mesh or a cache.
    if "--print-field-map" in arguments:
        for row in field_map_table():
            print(f"{row['wrf']:<12} {row['mpas']:<18} {row['rank']:<8} "
                  f"{row['source']:<9} {row['note']}")
        return 0
    # Data path from here down: history read, regrid, NetCDF write. All three
    # are Rust-only under the 2.5 Python boundary, so the bare default refuses
    # and names rw_mpas_convert. The acknowledged door stays open because this
    # file is the reference evidence/rw-mpas-converter-parity.json is about.
    refusal = mrw.refuse_unless_reference_regeneration(
        "mpas_history_to_wrfout.py", arguments
    )
    if refusal is not None:
        return refusal
    args = _parser().parse_args(mrw.strip_reference_acknowledgement(arguments))
    weights, cache_file, built = mrw.weights_for(
        args.mesh, args.window, cache_dir=args.cache_dir,
        mesh_sha256=args.mesh_sha256,
    )
    mesh = mrw.read_mesh_coordinates(args.mesh)
    origin = (
        parse_timestamp(args.simulation_start)
        if args.simulation_start is not None
        else None
    )
    report: dict[str, Any] = {
        "schema": BRIDGE_SCHEMA,
        "window": args.window,
        "weights": {**weights.provenance(), "cache_file": str(cache_file),
                    "built_now": built},
        "frames": [],
        "field_map": field_map_table(),
        "absent_wrf_fields": [
            {"variable": name, "quantity": quantity, "reason": reason}
            for name, quantity, reason in ABSENT_WRF_FIELDS
        ],
    }
    for history_path in args.history:
        frame = read_history(
            history_path, init_path=args.init,
            init_sha256=args.init_sha256,
        )
        agreement = assert_mesh_agrees(frame, mesh)
        stamp = f"{frame.valid_time:%Y-%m-%d_%H_%M_%S}"
        destination = Path(args.out_dir) / f"{args.prefix}_d01_{stamp}"
        emitted = convert_frame(
            frame, weights, destination,
            simulation_start=origin,
            field_set=args.field_set,
            compress=bool(args.compress) and not args.no_compress,
            clobber=args.clobber,
        )
        record = dict(emitted.provenance)
        record["mesh_agreement"] = agreement
        report["frames"].append(record)
        print(
            f"{destination.name}: {emitted.shape[0]}x{emitted.shape[1]}x"
            f"{emitted.levels} in {emitted.seconds:.1f} s, "
            f"{emitted.provenance['output_bytes'] / 1e6:.0f} MB"
        )
    if args.json is not None:
        destination = Path(args.json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii", newline="\n",
        )
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
