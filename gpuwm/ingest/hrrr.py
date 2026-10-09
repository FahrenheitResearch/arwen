"""Native HRRR window loading and Lambert-to-Lambert interpolation.

The on-disk format is produced by ``hrrr_grib2_bridge``.  The loader binds
the editable gate and every payload to an externally recorded SHA-256 of the
``SHA256SUMS`` manifest before exposing arrays.  Horizontal interpolation is
performed in HRRR projection coordinates.  Wind vectors are explicitly
rotated from the source grid basis to earth-relative, interpolated, and then
rotated into the target Lambert basis.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, MutableMapping

import numpy as np

from gpuwm import perf_timing
from gpuwm.hrrr_forecast import validate_hrrr_source_forecast_hours
from gpuwm.ingest.quantization import clamp_bound_kissing
from gpuwm.ingest.horiz import (
    HorizontalSnapshot,
    _cupy,
    _wps_oned_gpu,
    lambert_rotation,
)
from gpuwm.static.lambert import EARTH_RADIUS_M, LambertGrid


HRRR_EARTH_RADIUS_M = 6_371_229.0
HRRR_GRID_SPACING_M = 3_000.0
HRRR_WPS_EQUIVALENT_DX_M = (
    HRRR_GRID_SPACING_M * EARTH_RADIUS_M / HRRR_EARTH_RADIUS_M
)
HRRR_SOIL_DEPTHS_M = np.array(
    [0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.0, 1.6, 3.0],
    dtype=np.float64,
)
HRRR_HYBRID_LEVELS = np.arange(1.0, 51.0, dtype=np.float64)

_ATMOSPHERE_3D = (
    "PRES", "QC", "QI", "QR", "QS", "QG", "HGT", "TT", "SPFH",
    "U_MASS", "V_MASS",
)
_ATMOSPHERE_2D = (
    "PSFC", "SOILHGT", "SKINTEMP", "SNOW", "SNOWH", "T2", "Q2",
    "U10_MASS", "V10_MASS", "LANDSEA", "XICE",
)
_SOIL_3D = ("SOILT", "SOILW")

#: The cloud-ice codes the bridge may bind QI to, as its gate spells them:
#: CIMIXR (0/1/82) from HRRRv3 (July 2018) on, and CICE (0/6/0), the code
#: HRRRv1 and v2 wrfnat files publish the same mixing ratio under.  The
#: bridge reads CICE only from a file that publishes no CIMIXR; a gate
#: binding QI to any other code is refused.
HRRR_CLOUD_ICE_GATES = (
    "PASS discipline=0 category=1 parameter=82 level_type=105",
    "PASS discipline=0 category=6 parameter=0 level_type=105",
)


def native_domain_coverage(latitude, longitude):
    """Whether every geographic target mass point has the native grid halo.

    Pure geometric test, no download or device use. Include all domain and
    lateral strip points. Coordinates are projected with the same Lambert
    transform used by the native horizontal plan, then converted to zero
    based fractional indices. False lets the source resolver try the next
    row boundary source or emit the absent boundary receipt.
    """
    from .native_extras import inside_native_grid
    from .hrrr_target import HRRR_SOURCE_NX, HRRR_SOURCE_NY
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)
    if latitude.shape != longitude.shape:
        raise ValueError("coverage coordinate shapes differ; prevents testing a different domain footprint")
    if not latitude.size or not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        return False
    x, y = hrrr_source_grid().latlon_to_ij(latitude, longitude)
    return inside_native_grid((np.asarray(x) - 1).ravel(), (np.asarray(y) - 1).ravel(),
                              nx=HRRR_SOURCE_NX, ny=HRRR_SOURCE_NY)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(root: Path, expected_manifest_sha256: str) -> frozenset[Path]:
    manifest = root / "SHA256SUMS"
    expected = str(expected_manifest_sha256).lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError("expected_manifest_sha256 must be 64 lowercase/uppercase hex digits")
    actual = _sha256(manifest)
    if actual != expected:
        raise ValueError(
            f"HRRR SHA256SUMS hash mismatch: expected {expected}, got {actual}")
    seen: set[Path] = set()
    payloads: list[tuple[str, Path, str]] = []
    for line_number, raw in enumerate(manifest.read_text().splitlines(), 1):
        if not raw:
            continue
        try:
            digest, relative = raw.split(None, 1)
        except ValueError as exc:
            raise ValueError(
                f"malformed SHA256SUMS line {line_number}") from exc
        relative = relative.removeprefix("*").removeprefix("./")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                f"unsafe SHA256SUMS path on line {line_number}: {relative!r}")
        path = root / candidate
        if candidate in seen:
            raise ValueError(f"duplicate SHA256SUMS path {relative!r}")
        seen.add(candidate)
        # `find ... > SHA256SUMS` sees the just-opened zero-byte destination.
        # Its self-entry cannot be recursively satisfied; the caller's
        # externally recorded manifest hash already binds the actual file.
        if candidate == Path("SHA256SUMS"):
            continue
        if not path.is_file():
            raise FileNotFoundError(f"manifest payload is missing: {path}")
        payloads.append((relative, path, digest.lower()))

    # Every payload is still hashed -- the scope of what this proves is
    # unchanged -- but the reads run concurrently.  This verification is
    # the whole bridge publication (all sealed leads) while the caller may
    # map only one of them, so it is the largest read in the hierarchy
    # stage; hashlib releases the GIL, so threads here are real
    # concurrency and not bookkeeping.
    with perf_timing.stage("ingest.hrrr.verify_manifest",
                           payloads=len(payloads)) as timed:
        total = 0
        workers = max(1, min(8, len(payloads)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for (relative, path, expected_digest), payload_hash in zip(
                    payloads, pool.map(_sha256,
                                       [entry[1] for entry in payloads])):
                if payload_hash != expected_digest:
                    raise ValueError(
                        f"HRRR payload hash mismatch for {relative}: "
                        f"expected {expected_digest}, got {payload_hash}")
                total += path.stat().st_size
        timed.count(bytes_hashed=total)
    return frozenset(seen)


def _read_gate(root: Path) -> dict[str, str]:
    gate_path = root / "gate.txt"
    values: dict[str, str] = {}
    for line_number, raw in enumerate(gate_path.read_text().splitlines(), 1):
        try:
            key, value = raw.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"malformed gate.txt line {line_number}") from exc
        if key in values:
            raise ValueError(f"duplicate gate.txt key {key!r}")
        values[key] = value
    from .native_extras import gate_extra_fields
    from .native_supplements import gate_optional_hybrid_fields

    # 561 records, plus 50 for each optional hybrid field the decoder
    # declared (the analyzed aerosol number pair) and 50 for each native
    # extra record a caller requested (--extras).
    optional_hybrid = gate_optional_hybrid_fields(values)
    required = {
        "status": "PASS",
        "atmosphere_selected_per_time": str(
            561 + 50 * (len(optional_hybrid) + len(gate_extra_fields(values)))),
        "hybrid_levels": "50",
        "soil_selected_per_time": "18",
        "window_shape": None,
        "window_zero_based_inclusive": None,
        "qice_mapping": None,
        "cross_time_inventory": None,
    }
    missing = sorted(set(required) - set(values))
    if missing:
        raise ValueError(f"gate.txt is missing keys: {missing}")
    for key, expected in required.items():
        if expected is not None and values[key] != expected:
            raise ValueError(
                f"gate.txt {key} is {values[key]!r}, expected {expected!r}")
    if not values["qice_mapping"].startswith(HRRR_CLOUD_ICE_GATES):
        raise ValueError(
            "gate.txt binds QI to none of the cloud-ice codes HRRR publishes "
            "(CIMIXR 0/1/82, or CICE 0/6/0 before July 2018): "
            f"{values['qice_mapping']!r}")
    if not values["cross_time_inventory"].startswith("PASS"):
        raise ValueError("gate.txt cross-time inventory did not pass")
    return values


def aerosol_fill_receipt(gate: Mapping[str, str]) -> dict[str, object] | None:
    """Final native bitmap repair counters, bound by the bridge gate hash."""
    policy = gate.get("aerosol_missing_policy")
    if policy is None:
        return None  # Historical bridges predate the repair receipt.
    fields = {}
    for name in ("QNWFA", "QNIFA"):
        counts = {}
        for stage in ("masked", "nearest_neighbor", "four_pt", "average_4pt", "zero"):
            key = f"aerosol_{name}_{stage}"
            try:
                counts[stage] = int(gate[key])
            except (KeyError, ValueError) as error:
                raise ValueError(f"missing or invalid aerosol fill counter {key}") from error
            if counts[stage] < 0:
                raise ValueError(f"negative aerosol fill counter {key}")
        if counts["masked"] != sum(counts[stage] for stage in
                ("nearest_neighbor", "four_pt", "average_4pt", "zero")):
            raise ValueError(f"{name} aerosol fill counts do not sum to the masked count")
        fields[name] = counts
    return {"policy": policy, "fields": fields}


def _parse_window(gate: Mapping[str, str]) -> tuple[int, int, int, int]:
    text = gate["window_zero_based_inclusive"]
    try:
        i_text, j_text = text.split()
        i_start, i_end = map(int, i_text.removeprefix("i=").split(".."))
        j_start, j_end = map(int, j_text.removeprefix("j=").split(".."))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid gate window {text!r}") from exc
    if min(i_start, j_start) < 0 or i_end < i_start or j_end < j_start:
        raise ValueError(f"invalid gate window {text!r}")
    expected_shape = f"{j_end - j_start + 1}x{i_end - i_start + 1}"
    if gate["window_shape"] != expected_shape:
        raise ValueError(
            f"gate window_shape {gate['window_shape']!r} != {expected_shape!r}")
    return i_start, i_end, j_start, j_end


def _map_f32(path: Path, shape: tuple[int, ...]) -> np.memmap:
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * 4
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {actual_bytes} bytes; expected {expected_bytes} "
            f"for shape {shape}")
    return np.memmap(path, mode="r", dtype="<f4", shape=shape, order="C")


@dataclass(frozen=True)
class HrrrNativeSnapshot:
    """One verified HRRR source window in native hybrid coordinates."""

    valid_time: datetime
    forecast_hour: int
    i_start: int
    j_start: int
    ny: int
    nx: int
    fields: Mapping[str, np.ndarray]
    extra_fields: tuple[str, ...] = ()
    #: Compatibility for historical bridges that withheld a masked pair.
    #: New bridges repair requested aerosol masks and never emit these
    #: withholding keys. An old artifact still names its missing payload.
    withheld_fields: Mapping[str, str] = field(default_factory=dict)

    def source_cell_latlon(self, rows, cols):
        """Geographic coordinates of cells in this declared source window."""
        return hrrr_source_grid().ij_to_latlon(
            np.asarray(cols, dtype=np.float64) + self.i_start + 1.,
            np.asarray(rows, dtype=np.float64) + self.j_start + 1.)

    def __post_init__(self) -> None:
        if self.forecast_hour not in range(49):
            raise ValueError("the native bridge supports source leads f00..f48")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "withheld_fields",
                           MappingProxyType(dict(self.withheld_fields)))


def _gate_forecast_hours(gate: Mapping[str, str]) -> tuple[int, ...]:
    text = gate.get("forecast_hours")
    if text is None:
        # Backward-compatible read of the already sealed f00/f01 proof.
        return (0, 1)
    try:
        hours = tuple(int(value) for value in text.split(","))
    except ValueError as exc:
        raise ValueError("gate.txt forecast_hours is invalid") from exc
    try:
        series_count = int(gate["series_count"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "gate.txt series_count is missing or invalid") from exc
    if series_count != len(hours):
        raise ValueError("gate.txt series_count differs from forecast_hours")
    try:
        cycle = datetime.strptime(gate["cycle"], "%Y-%m-%d %H:%M:%S")
    except (KeyError, ValueError) as exc:
        raise ValueError("gate.txt cycle is missing or invalid") from exc
    try:
        return validate_hrrr_source_forecast_hours(hours, cycle=cycle)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"gate.txt source forecast window is invalid: {exc}") from exc


def load_hrrr_native_window(
        root, forecast_hour: int, *,
        expected_manifest_sha256: str) -> HrrrNativeSnapshot:
    """Verify and memory-map one native bridge forecast-hour snapshot.

    ``expected_manifest_sha256`` must come from evidence outside the bridge
    directory.  Supplying it prevents an edited verdict or edited
    ``SHA256SUMS`` file from becoming self-authenticating evidence.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"HRRR bridge directory is missing: {root}")
    entries = _verify_manifest(root, expected_manifest_sha256)
    gate = _read_gate(root)
    return _load_verified_hrrr_native_window(root, gate, forecast_hour, manifest_entries=entries)


def _load_verified_hrrr_native_window(
        root: Path, gate: Mapping[str, str],
        forecast_hour: int, *, manifest_entries=None) -> HrrrNativeSnapshot:
    available_hours = _gate_forecast_hours(gate)
    i_start, i_end, j_start, j_end = _parse_window(gate)
    try:
        cycle = datetime.strptime(gate["cycle"], "%Y-%m-%d %H:%M:%S")
    except (KeyError, ValueError) as exc:
        raise ValueError("gate.txt cycle is missing or invalid") from exc
    forecast_hour = int(forecast_hour)
    if forecast_hour not in available_hours:
        raise ValueError(
            f"forecast_hour must be one of {available_hours}, got {forecast_hour}")
    ny = j_end - j_start + 1
    nx = i_end - i_start + 1
    atmosphere_dir = root / f"atmosphere-f{forecast_hour:02d}"
    soil_dir = root / f"soil-f{forecast_hour:02d}"
    fields: dict[str, np.ndarray] = {}
    for name in _ATMOSPHERE_3D:
        fields[name] = _map_f32(
            atmosphere_dir / f"{name}.f32le", (50, ny, nx))
    from .native_extras import gate_extra_fields
    extras = gate_extra_fields(gate)
    for name in extras:
        payload = atmosphere_dir / f"{name}.f32le"
        if manifest_entries is not None and payload.relative_to(root) not in manifest_entries:
            raise ValueError(f"extra payload is not manifest-bound; prevents unauthenticated tracer input: {payload}")
        fields[name] = _map_f32(payload, (50, ny, nx))
    for name in _ATMOSPHERE_2D:
        fields[name] = _map_f32(
            atmosphere_dir / f"{name}.f32le", (ny, nx))
    from .native_supplements import (gate_optional_hybrid_fields,
                                     gate_supplement_fields,
                                     gate_withheld_optional_hybrid_fields)
    withheld = gate_withheld_optional_hybrid_fields(gate)
    for name in gate_optional_hybrid_fields(gate):
        payload = atmosphere_dir / f"{name}.f32le"
        if manifest_entries is not None and payload.relative_to(root) not in manifest_entries:
            raise ValueError(f"analyzed aerosol payload is not bound by the bridge manifest: {payload}")
        fields[name] = _map_f32(payload, (50, ny, nx))
        mask = atmosphere_dir / f"{name}.mask"
        requires_mask = "source_mask_preserved_for_target_mapping" in gate.get("aerosol_missing_policy", "")
        if requires_mask and not mask.is_file():
            raise ValueError(f"{name} source bitmap is missing: {mask}; using native zero fills as valid donors would bypass the METGRID chain")
        if mask.exists():
            if manifest_entries is not None and mask.relative_to(root) not in manifest_entries:
                raise ValueError(f"aerosol source bitmap is not manifest-bound: {mask}")
            if mask.stat().st_size != 50 * ny * nx:
                raise ValueError(f"aerosol source bitmap has the wrong shape: {mask}")
            fields[name + "_SOURCE_MASK"] = np.memmap(mask, mode="r", dtype=np.uint8, shape=(50, ny, nx))

    for name in gate_supplement_fields(gate):
        payload = atmosphere_dir / f"{name}.f32le"
        if manifest_entries is not None and payload.relative_to(root) not in manifest_entries:
            raise ValueError(f"supplement payload is not bound by the bridge manifest: {payload}")
        fields[name] = _map_f32(payload, (ny, nx))
    for name in _SOIL_3D:
        fields[name] = _map_f32(soil_dir / f"{name}.f32le", (9, ny, nx))
    from .native_supplements import gate_soil_surface_fields
    for name in gate_soil_surface_fields(gate):
        payload = soil_dir / f"{name}.f32le"
        if manifest_entries is not None and payload.relative_to(root) not in manifest_entries:
            raise ValueError("analyzed vegetation is not bound by the source manifest")
        fields[name] = _map_f32(payload, (ny, nx))
    return HrrrNativeSnapshot(
        valid_time=cycle + timedelta(hours=forecast_hour),
        forecast_hour=forecast_hour,
        i_start=i_start,
        j_start=j_start,
        ny=ny,
        nx=nx,
        fields=fields,
        extra_fields=extras,
        withheld_fields=withheld,
    )


def load_hrrr_native_series(
        root, forecast_hours, *,
        expected_manifest_sha256: str) -> tuple[HrrrNativeSnapshot, ...]:
    """Verify one bridge publication once and map several forecast hours."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"HRRR bridge directory is missing: {root}")
    entries = _verify_manifest(root, expected_manifest_sha256)
    gate = _read_gate(root)
    requested = tuple(int(hour) for hour in forecast_hours)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("forecast_hours must be a non-empty unique sequence")
    return tuple(
        _load_verified_hrrr_native_window(root, gate, hour, manifest_entries=entries)
        for hour in requested
    )


def verified_hrrr_native_bridge(
        root, *, expected_manifest_sha256: str):
    """Verify one bridge publication once; return a loader of one lead.

    ``load(forecast_hour)`` maps that lead as :func:`load_hrrr_native_window`
    does, without hashing the publication again.  A mapped lead holds one
    open file descriptor per field until its arrays are let go (a numpy
    memmap keeps its mmap, and CPython's mmap holds a duplicate of the
    file's descriptor), at least 24 per lead.  A caller walking a long
    window therefore maps each lead when it needs it:
    :func:`load_hrrr_native_series` of every lead of a 48 h window holds
    over 1,150 descriptors and fails with ``[Errno 24] Too many open
    files`` under the ordinary 1024 soft ``RLIMIT_NOFILE``.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"HRRR bridge directory is missing: {root}")
    entries = _verify_manifest(root, expected_manifest_sha256)
    gate = _read_gate(root)

    def load(forecast_hour: int) -> HrrrNativeSnapshot:
        return _load_verified_hrrr_native_window(
            root, gate, forecast_hour, manifest_entries=entries)

    return load


def load_hrrr_pipeline_ready_window(
        root, forecast_hour: int) -> HrrrNativeSnapshot:
    """Map one hour after the live producer's atomic ready receipt.

    This is deliberately not an evidence-loading API: the caller must own the
    decoder process, consume its global inventory preflight receipt, and verify
    the source hashes before accepting an hourly ready receipt.  The completed
    bridge is still sealed and externally bound before it becomes reusable.
    """
    root = Path(root)
    gate = _read_gate(root)
    return _load_verified_hrrr_native_window(
        root, gate, int(forecast_hour))


def hrrr_source_grid() -> LambertGrid:
    """Return the HRRR grid expressed in WPS's fixed-radius coordinates.

    Scaling ``dx`` by ``R_WPS/R_HRRR`` makes WPS's 6,370-km projection
    geometrically identical to GRIB shape-of-Earth 6 (6,371.229 km).
    """
    return LambertGrid(
        ref_lat=21.138123,
        ref_lon=-122.719528,
        truelat1=38.5,
        truelat2=38.5,
        stand_lon=-97.5,
        dx=HRRR_WPS_EQUIVALENT_DX_M,
        dy=HRRR_WPS_EQUIVALENT_DX_M,
        e_we=1800,
        e_sn=1060,
        known_x=1.0,
        known_y=1.0,
    )


#: On the identity route, how far (in source cells) the projected position
#: of a target point may sit from its lattice position before the target
#: is refused as not the native grid.  The native grid's two descriptions
#: (WPS's sphere, where the model integrates, and the GRIB header's,
#: which :func:`hrrr_source_grid` reproduces; hrrr_target
#: .native_grid_identity) agree at the south-west corner and drift apart
#: with distance from it: MEASURED 0.20 cells at the grid centre and 0.35
#: at the north-east corner.  Half a cell is the geometric limit past
#: which a point belongs to another cell.
IDENTITY_SNAP_LIMIT_CELLS = 0.5


def _snap_to_native_lattice(global_x, global_y, *, nx: int, ny: int,
                            placement=None):
    """The identity route's coordinates: every target point on its own
    native lattice position, exactly, from the target array's shape.

    ``placement`` (:class:`gpuwm.ingest.hrrr_target.NativeLatticePlacement`,
    default the whole ``ny`` x ``nx`` grid) is the native rectangle the
    target's mass grid occupies: a boundary strip of the native grid is
    its own mass, u or v shape, offset by ``(i0, j0)``.  Its faces that
    lie inside the native grid read their two neighbours; only a face
    past the native grid's own edge takes the edge cell.

    ``(ny, nx)`` is the mass grid: point ``(j, i)`` is source cell
    ``(j, i)``, fraction 0, so the parabolic and bilinear operators
    return that cell's value exactly (their ``x == 0`` branch).
    ``(ny, nx + 1)`` is the u staggering: face ``i`` sits at ``i - 1/2``,
    half way between two cells; ``(ny + 1, nx)`` the v staggering.  The
    outermost faces, half a cell past the grid's edge, take the edge
    cell.  The projected positions are not used, only checked: the
    largest distance between them and the lattice is returned (the
    sphere drift above) and past :data:`IDENTITY_SNAP_LIMIT_CELLS` the
    target is refused by name.
    """
    global_x = np.asarray(global_x, dtype=np.float64)
    global_y = np.asarray(global_y, dtype=np.float64)
    shape = tuple(global_x.shape)
    if placement is None:
        i0, j0, mass_nx, mass_ny = 0, 0, nx, ny
    else:
        i0, j0 = int(placement.i0), int(placement.j0)
        mass_nx, mass_ny = int(placement.nx), int(placement.ny)
        if not (0 <= i0 and i0 + mass_nx <= nx
                and 0 <= j0 and j0 + mass_ny <= ny):
            raise ValueError(
                f"the placement i={i0}..+{mass_nx}, j={j0}..+{mass_ny} "
                f"is not inside the {ny} x {nx} native grid")
    if shape == (mass_ny, mass_nx):
        offset_x, offset_y = 0.0, 0.0
    elif shape == (mass_ny, mass_nx + 1):
        offset_x, offset_y = -0.5, 0.0
    elif shape == (mass_ny + 1, mass_nx):
        offset_x, offset_y = 0.0, -0.5
    else:
        raise ValueError(
            "the identity route maps the native grid's mass, u or v "
            f"staggering ({mass_ny} x {mass_nx}, {mass_ny} x "
            f"{mass_nx + 1} or {mass_ny + 1} x {mass_nx}); "
            f"got a target of shape {shape}")
    rows, cols = np.indices(shape, dtype=np.float64)
    exact_x = cols + offset_x + i0
    exact_y = rows + offset_y + j0
    distance = float(max(np.abs(exact_x - global_x).max(),
                         np.abs(exact_y - global_y).max()))
    if not distance <= IDENTITY_SNAP_LIMIT_CELLS:
        raise ValueError(
            "the target was declared the native HRRR grid but a point of "
            f"it projects {distance:.3f} cells from its lattice position "
            f"(limit {IDENTITY_SNAP_LIMIT_CELLS:g}); it is not that grid")
    # The outer faces: -0.5 and nx - 0.5 (ny - 0.5) are half a cell past
    # the edge; they take the edge cell.
    snapped_x = np.clip(exact_x, 0.0, float(nx - 1))
    snapped_y = np.clip(exact_y, 0.0, float(ny - 1))
    return snapped_x, snapped_y, distance


def _projected_index_geometry(snapshot: HrrrNativeSnapshot,
                              target_lat, target_lon, *,
                              identity: bool = False, placement=None):
    """Resolve exact zero-based HRRR-window interpolation coordinates.

    ``identity`` (the target is the native grid itself,
    :func:`gpuwm.ingest.hrrr_target.native_grid_identity`, or a whole-cell
    piece of it whose ``placement`` is given,
    :func:`gpuwm.ingest.hrrr_target.native_lattice_placement`): the
    coordinates are snapped onto the native lattice
    (:func:`_snap_to_native_lattice`) and no interpolation halo past the
    window is demanded, because every operator clamps its stencil at
    the window's edge and a whole-cell position reads one cell.
    """

    source_x, source_y = hrrr_source_grid().latlon_to_ij(
        target_lat, target_lon)
    # LambertGrid coordinates are one-based; bridge window origins are
    # zero-based indices in the original south-to-north GRIB scan.
    global_x = np.asarray(source_x, dtype=np.float64) - 1.0
    global_y = np.asarray(source_y, dtype=np.float64) - 1.0
    if identity:
        from gpuwm.ingest.hrrr_target import HRRR_SOURCE_NX, HRRR_SOURCE_NY

        if (snapshot.i_start, snapshot.nx, snapshot.j_start,
                snapshot.ny) != (0, HRRR_SOURCE_NX, 0, HRRR_SOURCE_NY):
            raise ValueError(
                "the identity route needs the whole native grid as its "
                f"window, got i={snapshot.i_start}..+{snapshot.nx}, "
                f"j={snapshot.j_start}..+{snapshot.ny}")
        global_x, global_y, _snap = _snap_to_native_lattice(
            global_x, global_y, nx=HRRR_SOURCE_NX, ny=HRRR_SOURCE_NY,
            placement=placement)
    global_ix = np.floor(global_x).astype(np.int64)
    global_iy = np.floor(global_y).astype(np.int64)
    ix = global_ix - snapshot.i_start
    iy = global_iy - snapshot.j_start
    if not identity:
        from gpuwm.ingest.hrrr_target import HRRR_SOURCE_NX, HRRR_SOURCE_NY

        # The bilinear cell must be inside the window always.  The rest
        # of the four-point parabolic halo may be missing only where the
        # window edge is HRRR's own edge: there metgrid's sixteen-point
        # stencil leaves the source and falls back to four-point bilinear
        # (_parabolic_stencil_leaves_window); anywhere else a short
        # window is a crop defect.
        if (np.min(ix) < 0 or np.max(ix + 1) >= snapshot.nx
                or np.min(iy) < 0 or np.max(iy + 1) >= snapshot.ny):
            raise ValueError(
                "HRRR bridge window lacks the bilinear donor cell")
        native_edges = (
            (np.min(ix - 1) < 0, snapshot.i_start == 0),
            (np.max(ix + 2) >= snapshot.nx,
             snapshot.i_start + snapshot.nx == HRRR_SOURCE_NX),
            (np.min(iy - 1) < 0, snapshot.j_start == 0),
            (np.max(iy + 2) >= snapshot.ny,
             snapshot.j_start + snapshot.ny == HRRR_SOURCE_NY))
        if any(short and not at_native_edge
               for short, at_native_edge in native_edges):
            raise ValueError(
                "HRRR bridge window lacks the four-point interpolation halo")
    if identity:
        # A whole-cell position on the last column floors to nx - 1,
        # whose bilinear partner nx would be read (unweighted) by the
        # operators that check it; spell it as the cell before with a
        # unit fraction, which both operators return exactly.
        for index, limit in ((ix, snapshot.nx), (iy, snapshot.ny)):
            np.subtract(index, 1, out=index, where=index >= limit - 1)
        global_ix = ix + snapshot.i_start
        global_iy = iy + snapshot.j_start
    x = global_x - snapshot.i_start
    y = global_y - snapshot.j_start
    nearest_ix = (
        np.floor(global_x + 0.5).astype(np.int64) - snapshot.i_start)
    nearest_iy = (
        np.floor(global_y + 0.5).astype(np.int64) - snapshot.j_start)
    return x, y, ix, iy, nearest_ix, nearest_iy, global_ix, global_iy


def _wps_oned_cpu(x, a, b, c, d):
    """NumPy FP32 mirror of WPS ``oned`` with CUDA FTZ decisions."""

    zero = np.float32(0.0)
    half = np.float32(0.5)
    one = np.float32(1.0)
    regular = ((one - x)
               * (b + x * (half * (c - a) + x * (half * (c + a) - b)))
               + x * (c + (one - x)
                      * (half * (b - d) + (one - x)
                         * (half * (b + d) - c))))
    out = np.zeros_like(regular)
    out = np.where(x == zero, b, out)
    out = np.where(x == one, c, out)
    product = np.multiply(b, c, dtype=np.float32)
    both = np.abs(product) >= np.float32(np.finfo(np.float32).tiny)
    only_a = both & (a != zero) & (d == zero)
    only_d = both & (a == zero) & (d != zero)
    neither = both & (a == zero) & (d == zero)
    all_four = both & (a != zero) & (d != zero)
    out = np.where(neither, b * (one - x) + c * x, out)
    out = np.where(
        only_a, b + x * (half * (c - a) + x * (half * (c + a) - b)), out)
    out = np.where(
        only_d,
        c + (one - x) * (half * (b - d)
                          + (one - x) * (half * (b + d) - c)),
        out,
    )
    return np.asarray(np.where(all_four, regular, out), dtype=np.float32)


def _parabolic_stencil_leaves_window(ix, iy, nx: int, ny: int, *,
                                     identity: bool):
    """Targets whose four-point parabolic stencil leaves the source window.

    Only an interpolated window can have them, and only at HRRR's own
    edge (:func:`_projected_index_geometry` refuses a short crop
    anywhere else).  There WPS metgrid's ``sixteen_pt`` finds a stencil
    point outside the source and the interp sequence falls through to
    ``four_pt``, bilinear from the inner 2 x 2, which is what these
    targets get.  Before, the window refused every such target, so a
    domain one row inside the HRRR grid was refused although its
    bilinear donors are all inside.  The identity route keeps its own
    clamped stencil (its documented rule; every whole-cell position reads
    one cell).  ``None`` when no target is affected.
    """

    if identity:
        return None
    ix = np.asarray(ix)
    iy = np.asarray(iy)
    leaves = ((ix - 1 < 0) | (ix + 2 > nx - 1)
              | (iy - 1 < 0) | (iy + 2 > ny - 1))
    return leaves if bool(leaves.any()) else None


def _bilinear_at(field, lead, iy, ix, fy, fx, xp):
    """The bilinear branch of the projected operators, for some targets."""

    one = xp.float32(1.0)
    lower = ((one - fx) * field[lead + (iy, ix)]
             + fx * field[lead + (iy, ix + 1)])
    upper = ((one - fx) * field[lead + (iy + 1, ix)]
             + fx * field[lead + (iy + 1, ix + 1)])
    return ((one - fy) * lower + fy * upper).astype(xp.float32, copy=False)


def _metgrid_edge_fallback(result, field, edge, iy, ix, fy, fx, xp):
    """Replace ``result`` at the ``edge`` targets by four-point bilinear."""

    if edge is None:
        return result
    lead = (slice(None),) * (field.ndim - 2)
    rows, cols = xp.nonzero(edge)
    values = _bilinear_at(field, lead, iy[rows, cols], ix[rows, cols],
                          fy[rows, cols], fx[rows, cols], xp)
    result = result.copy()
    result[lead + (rows, cols)] = values
    return result


#: What actually evaluated a projected-source horizontal apply.  It goes
#: into the mapping report, because "the CPU backend" stopped being one
#: answer the moment the operator got a second implementation.
PROJECTED_OPERATOR_CUDA = "cuda-projected-donor-v1"
PROJECTED_OPERATOR_RUST = "rust-indexed-donor-v1"
PROJECTED_OPERATOR_NUMPY = "numpy-projected-donor-v1"

#: One process, one sentence.  A child hierarchy builds three plans per
#: domain and every one of them would otherwise say the same thing.
_PROJECTED_FALLBACK_ANNOUNCED = False


def _announce_projected_numpy_fallback(reason: str) -> None:
    """Say once, loudly, that this run is on the slow projected operator."""

    global _PROJECTED_FALLBACK_ANNOUNCED
    if _PROJECTED_FALLBACK_ANNOUNCED:
        return
    _PROJECTED_FALLBACK_ANNOUNCED = True
    from gpuwm import bridges, explain

    # The build line comes from the shared shell rule: a literal `&&`
    # here was a Windows PowerShell 5.1 parser error.
    explain.warn(
        "projected-source horizontal mapping is running the NumPy mirror "
        f"instead of the native operator ({reason}).  It is the same "
        "arithmetic and it is slow: the native entry evaluates the same "
        "stencil across every core with no whole-array temporaries.  "
        "Rebuild the CPU preprocessing bridge ("
        + bridges.cargo_build_one_liner(bridges.CRATE_RELATIVE)
        + ") or stage a current bridges bundle.",
        "The NumPy mirror issues 32 fancy-index gathers per 3-D field and "
        "evaluates roughly 150 whole-target-shape temporaries inside the "
        "five `oned` calls, on one core, to produce one target-shape "
        "answer.  Nested HRRR preparation is dominated by this operator, "
        "so an install that quietly inherits the mirror pays it on every "
        "child of every run.")


class _ProjectedGpuPlan:
    """WPS overlapping-parabolic interpolation on projected source indices."""

    operator = PROJECTED_OPERATOR_CUDA

    def __init__(self, snapshot: HrrrNativeSnapshot, target_lat, target_lon,
                 *, identity: bool = False, placement=None):
        cp = _cupy()
        self.route = "identity" if identity else "interpolated"
        (x, y, ix, iy, nearest_ix, nearest_iy,
         global_ix, global_iy) = _projected_index_geometry(
             snapshot, target_lat, target_lon, identity=identity,
             placement=placement)
        global_x = x + snapshot.i_start
        global_y = y + snapshot.j_start
        self.source_shape = (snapshot.ny, snapshot.nx)
        self.target_shape = global_x.shape
        self.x_host = x
        self.y_host = y
        self.ix = cp.asarray(ix, dtype=cp.int32)
        self.iy = cp.asarray(iy, dtype=cp.int32)
        # Split global coordinates into an exact integer donor and an FP32
        # fractional weight.  This keeps stencil selection independent of the
        # bridge crop and avoids a second FP32 floor near integer boundaries.
        self.fx = cp.asarray(global_x - global_ix, dtype=cp.float32)
        self.fy = cp.asarray(global_y - global_iy, dtype=cp.float32)
        self.nearest_ix = cp.asarray(nearest_ix, dtype=cp.int32)
        self.nearest_iy = cp.asarray(nearest_iy, dtype=cp.int32)
        edge = _parabolic_stencil_leaves_window(
            ix, iy, snapshot.nx, snapshot.ny, identity=identity)
        self.edge = None if edge is None else cp.asarray(edge)

    def apply(self, field, *, method="parabolic"):
        cp = _cupy()
        field = cp.asarray(field, dtype=cp.float32)
        if field.ndim < 2 or field.shape[-2:] != self.source_shape:
            raise ValueError("HRRR field trailing dimensions do not match window")
        lead = (slice(None),) * (field.ndim - 2)
        expand = (None,) * (field.ndim - 2)
        ny, nx = self.source_shape
        if method == "nearest":
            return field[lead + (self.nearest_iy, self.nearest_ix)]
        iy = self.iy
        ix = self.ix
        fy = self.fy[expand]
        fx = self.fx[expand]
        if method == "bilinear":
            lower = ((cp.float32(1.0) - fx)
                     * field[lead + (iy, ix)]
                     + fx * field[lead + (iy, ix + 1)])
            upper = ((cp.float32(1.0) - fx)
                     * field[lead + (iy + 1, ix)]
                     + fx * field[lead + (iy + 1, ix + 1)])
            return ((cp.float32(1.0) - fy) * lower + fy * upper).astype(
                cp.float32, copy=False)
        if method != "parabolic":
            raise ValueError(
                "method must be 'nearest', 'bilinear', or 'parabolic'")
        xindices = [cp.clip(ix + offset, 0, nx - 1)
                    for offset in (-1, 0, 1, 2)]
        yindices = [cp.clip(iy + offset, 0, ny - 1)
                    for offset in (-1, 0, 1, 2)]
        rows = []
        tiny = cp.float32(1.0e-20)
        zero = cp.float32(0.0)
        for jy in yindices:
            values = [cp.where(field[lead + (jy, jx)] == zero,
                               tiny, field[lead + (jy, jx)])
                      for jx in xindices]
            rows.append(_wps_oned_gpu(fx, *values))
        result = _wps_oned_gpu(fy, *rows)
        result = cp.where(result == tiny, zero, result).astype(
            cp.float32, copy=False)
        return _metgrid_edge_fallback(result, field, self.edge, self.iy,
                                      self.ix, self.fy, self.fx, cp)

    def masked_bilinear_stencil(
            self, source_valid, target_apply, *, fallback_radius=8,
            closed_edges=(), native=None, workers=None):
        indices_y, indices_x, weights, report = (
            _build_masked_bilinear_stencil(
                self.x_host, self.y_host, source_valid, target_apply,
                fallback_radius=fallback_radius,
                closed_edges=closed_edges, native=native, workers=workers))
        return _GpuMaskedBilinearStencil(
            self.source_shape, indices_y, indices_x, weights, report)


class _ProjectedCpuPlan:
    """Exact-donor WPS interpolation on projected source indices.

    The donor is selected in FP64 and kept: passing a local coordinate
    just below an integer through FP32 can advance it after rounding, so
    this route carries ``(donor, fraction)`` as two separate things all
    the way down, exactly like the CUDA plan.  That is why it could not
    use the Rust regular-grid entry, which derives the donor from a
    fractional coordinate itself.

    It now has an entry of its own.  When the loaded bridge exposes
    ``gpuwm_indexed_interp_f32`` this hands it the donors and the
    fractions and the whole operator runs across every core with no
    intermediate arrays; :meth:`_apply_numpy` stays beside it as the
    reference implementation, is what runs when the bridge is older than
    the checkout, and is what the parity gate compares against.  The two
    are bit-identical by construction and by test, including the
    zero/``tiny`` sentinel round trip and the host-IEEE ``oned``
    missing-value predicate.
    """

    def __init__(self, snapshot: HrrrNativeSnapshot, target_lat, target_lon,
                 backend, *, identity: bool = False, placement=None):
        self.route = "identity" if identity else "interpolated"
        (x, y, ix, iy, nearest_ix, nearest_iy,
         global_ix, global_iy) = _projected_index_geometry(
             snapshot, target_lat, target_lon, identity=identity,
             placement=placement)
        global_x = x + snapshot.i_start
        global_y = y + snapshot.j_start
        self.source_shape = (snapshot.ny, snapshot.nx)
        self.target_shape = tuple(map(int, x.shape))
        self.x_host = x
        self.y_host = y
        self.ix = ix.astype(np.int32)
        self.iy = iy.astype(np.int32)
        self.fx = np.asarray(global_x - global_ix, dtype=np.float32)
        self.fy = np.asarray(global_y - global_iy, dtype=np.float32)
        self.nearest_ix = nearest_ix.astype(np.int32)
        self.nearest_iy = nearest_iy.astype(np.int32)
        self.edge = _parabolic_stencil_leaves_window(
            ix, iy, snapshot.nx, snapshot.ny, identity=identity)
        self._native = None
        self.operator = PROJECTED_OPERATOR_NUMPY
        builder = getattr(backend, "indexed_donor_plan", None)
        if builder is None:
            _announce_projected_numpy_fallback(
                "this preprocessing backend has no indexed-donor plan")
        elif not getattr(backend, "indexed_donor_interp", False):
            _announce_projected_numpy_fallback(
                "the loaded CPU preprocessing bridge does not export "
                "gpuwm_indexed_interp_f32")
        else:
            self._native = builder(
                self.source_shape, self.iy, self.ix, self.fy, self.fx)
            self.operator = PROJECTED_OPERATOR_RUST

    def apply(self, field, *, method="parabolic"):
        field = np.asarray(field, dtype=np.float32)
        if field.ndim < 2 or field.shape[-2:] != self.source_shape:
            raise ValueError("HRRR field trailing dimensions do not match window")
        if method == "nearest":
            lead = (slice(None),) * (field.ndim - 2)
            return field[lead + (self.nearest_iy, self.nearest_ix)]
        if method not in ("bilinear", "parabolic"):
            raise ValueError(
                "method must be 'nearest', 'bilinear', or 'parabolic'")
        if self._native is not None:
            result = self._native.apply(field, method=method)
            if method == "parabolic":
                # The bridge clamps the stencil; the few targets whose
                # stencil leaves HRRR's own edge take metgrid's bilinear
                # fallback here, exactly as the NumPy reference does.
                result = _metgrid_edge_fallback(
                    np.asarray(result, dtype=np.float32), field, self.edge,
                    self.iy, self.ix, self.fy, self.fx, np)
            return result
        return self._apply_numpy(field, method=method)

    def _apply_numpy(self, field, *, method="parabolic"):
        """The reference operator: one core, whole-array temporaries.

        Kept callable in its own right, not only as a fallback.  It is
        the authority the native entry is pinned to, so the gate that
        proves the port has to be able to run it side by side with the
        thing that replaced it.
        """

        field = np.asarray(field, dtype=np.float32)
        if field.ndim < 2 or field.shape[-2:] != self.source_shape:
            raise ValueError("HRRR field trailing dimensions do not match window")
        lead = (slice(None),) * (field.ndim - 2)
        expand = (None,) * (field.ndim - 2)
        if method == "nearest":
            return field[lead + (self.nearest_iy, self.nearest_ix)]
        iy = self.iy
        ix = self.ix
        fy = self.fy[expand]
        fx = self.fx[expand]
        if method == "bilinear":
            lower = ((np.float32(1.0) - fx)
                     * field[lead + (iy, ix)]
                     + fx * field[lead + (iy, ix + 1)])
            upper = ((np.float32(1.0) - fx)
                     * field[lead + (iy + 1, ix)]
                     + fx * field[lead + (iy + 1, ix + 1)])
            return np.asarray(
                (np.float32(1.0) - fy) * lower + fy * upper,
                dtype=np.float32)
        if method != "parabolic":
            raise ValueError(
                "method must be 'nearest', 'bilinear', or 'parabolic'")
        ny, nx = self.source_shape
        xindices = [np.clip(ix + offset, 0, nx - 1)
                    for offset in (-1, 0, 1, 2)]
        yindices = [np.clip(iy + offset, 0, ny - 1)
                    for offset in (-1, 0, 1, 2)]
        rows = []
        tiny = np.float32(1.0e-20)
        zero = np.float32(0.0)
        for jy in yindices:
            values = [np.where(field[lead + (jy, jx)] == zero,
                               tiny, field[lead + (jy, jx)])
                      for jx in xindices]
            rows.append(_wps_oned_cpu(fx, *values))
        result = _wps_oned_cpu(fy, *rows)
        result = np.asarray(np.where(result == tiny, zero, result),
                            dtype=np.float32)
        return _metgrid_edge_fallback(result, field, self.edge, self.iy,
                                      self.ix, self.fy, self.fx, np)

    def masked_bilinear_stencil(
            self, source_valid, target_apply, *, fallback_radius=8,
            closed_edges=(), native=None, workers=None):
        native, workers = _stencil_engine(native, workers)
        indices_y, indices_x, weights, report = (
            _build_masked_bilinear_stencil(
                self.x_host, self.y_host, source_valid, target_apply,
                fallback_radius=fallback_radius,
                closed_edges=closed_edges, native=native, workers=workers))
        return _CpuMaskedBilinearStencil(
            self.source_shape, indices_y, indices_x, weights, report,
            native=native, workers=workers)


@dataclass(frozen=True)
class _GpuMaskedBilinearStencil:
    """Four non-negative, unit-sum source weights per target point."""

    source_shape: tuple[int, int]
    indices_y: object
    indices_x: object
    weights: object
    report: Mapping[str, object]

    def __init__(self, source_shape, indices_y, indices_x, weights, report):
        cp = _cupy()
        object.__setattr__(self, "source_shape", tuple(source_shape))
        object.__setattr__(self, "indices_y", cp.asarray(indices_y, dtype=cp.int32))
        object.__setattr__(self, "indices_x", cp.asarray(indices_x, dtype=cp.int32))
        object.__setattr__(self, "weights", cp.asarray(weights, dtype=cp.float32))
        object.__setattr__(self, "report", MappingProxyType(dict(report)))

    def apply(self, field):
        cp = _cupy()
        field = cp.asarray(field, dtype=cp.float32)
        if field.ndim < 2 or field.shape[-2:] != self.source_shape:
            raise ValueError("HRRR field trailing dimensions do not match window")
        lead = (slice(None),) * (field.ndim - 2)
        expand = (None,) * (field.ndim - 2)
        result = None
        for corner in range(4):
            value = field[lead + (
                self.indices_y[corner], self.indices_x[corner])]
            term = value * self.weights[corner][expand]
            result = term if result is None else result + term
        return result.astype(cp.float32, copy=False)

    def apply_selected(self, field, select, fill):
        """:meth:`apply`, with ``fill`` on every target ``select`` leaves out."""
        cp = _cupy()
        mapped = self.apply(field)
        selector = cp.asarray(select, dtype=cp.bool_)[None, :, :]
        fill = cp.asarray(fill, dtype=cp.float32)
        if fill.ndim == 0:
            fill = cp.full(mapped.shape, fill, dtype=cp.float32)
        else:
            fill = cp.broadcast_to(fill[None, :, :], mapped.shape)
        return cp.where(selector, mapped, fill)


@dataclass(frozen=True)
class _CpuMaskedBilinearStencil:
    """Host equivalent of the convex four-donor surface stencil.

    Applied in the Rust preprocessing library
    (``gpuwm_masked_stencil_apply_f32``) on every layer at once, across
    every CPU the process may use; the NumPy apply it replaced is the test
    oracle (:mod:`gpuwm.verify.hrrr_stencil_oracle`).
    """

    source_shape: tuple[int, int]
    indices_y: np.ndarray
    indices_x: np.ndarray
    weights: np.ndarray
    report: Mapping[str, object]
    native: object
    workers: int

    def __init__(self, source_shape, indices_y, indices_x, weights, report,
                 *, native=None, workers=None):
        native, workers = _stencil_engine(native, workers)
        object.__setattr__(self, "source_shape", tuple(source_shape))
        object.__setattr__(
            self, "indices_y", np.asarray(indices_y, dtype=np.int32))
        object.__setattr__(
            self, "indices_x", np.asarray(indices_x, dtype=np.int32))
        object.__setattr__(
            self, "weights", np.asarray(weights, dtype=np.float32))
        object.__setattr__(self, "report", MappingProxyType(dict(report)))
        object.__setattr__(self, "native", native)
        object.__setattr__(self, "workers", int(workers))

    def _checked(self, field):
        field = np.asarray(field, dtype=np.float32)
        if field.ndim < 2 or field.shape[-2:] != self.source_shape:
            raise ValueError("HRRR field trailing dimensions do not match window")
        return field

    def apply(self, field):
        return self.native.masked_stencil_apply(
            self._checked(field), self.indices_y, self.indices_x,
            self.weights, workers=self.workers)

    def apply_selected(self, field, select, fill):
        """:meth:`apply`, with ``fill`` on every target ``select`` leaves out."""
        return self.native.masked_stencil_apply(
            self._checked(field), self.indices_y, self.indices_x,
            self.weights, select=select, fill=fill, workers=self.workers)


#: A fallback donor farther than this many source cells from its target
#: cell is listed by name in the mapping report and announced in one
#: warning.  Eight cells (24 km on HRRR's 3 km grid) was the fixed search
#: radius every HRRR target carried when the search stopped at its
#: radius; donors within it are the ordinary disagreement between two
#: land masks along a coast and are counted in the distance histogram.
DISTANT_DONOR_CELLS = 8

#: How many distant donors one report lists entry by entry.  Every one is
#: still counted, and inside the histogram and the maximum distance; the
#: list stops here and says how many it left out, so a target land mask
#: that disagrees with HRRR over a whole lake bed cannot turn one receipt
#: into megabytes.
_DISTANT_DONORS_LISTED = 256

#: The four window edges, in the order a clearance is measured.
WINDOW_EDGES = ("west", "east", "south", "north")


class SurfaceDonorSearchError(ValueError):
    """No surface-matched donor this source window can vouch for.

    Carries the facts remediation has to be computed from: the failing
    target cells and the smallest integer radius whose donor disk
    reaches a valid source cell for all of them (``None`` when the
    window holds no valid donor at any radius).  The stencil builder is
    generic and states facts only; the caller that knows WHICH grid was
    being mapped -- and where its window sits on the native HRRR grid --
    turns them into advice validated against the coverage guard.
    ``search_inputs`` is the builder's own input, kept so that advice
    (a trim) can be checked by running the same search on what it
    would leave.
    """

    def __init__(self, message, *, fallback_radius_cells,
                 required_radius_cells, unresolved_targets,
                 search_inputs=None):
        super().__init__(message)
        self.fallback_radius_cells = int(fallback_radius_cells)
        self.required_radius_cells = (
            None if required_radius_cells is None
            else int(required_radius_cells))
        self.unresolved_targets = tuple(
            tuple(int(index) for index in target)
            for target in unresolved_targets)
        self.search_inputs = search_inputs


#: The bit each window edge takes in the native stencil entry's mask.
_EDGE_BITS = {"west": 1, "east": 2, "south": 4, "north": 8}


def _stencil_engine(native=None, workers=None):
    """The Rust library and worker count the soil stencil runs on.

    ``native`` is a loaded :class:`gpuwm.ingest.cpu_backend.CpuPreprocessBackend`
    (a CPU backend's own, which honours an explicit bridge); without one,
    the library the resolution ladder picks, on every CPU the process may
    use.  A library without the stencil is refused by name with the
    remedy: there is no NumPy route.
    """
    from gpuwm.ingest.cpu_backend import (
        available_cpu_count, shared_cpu_backend)

    if native is None:
        native = shared_cpu_backend()
    native.require_masked_stencil()
    return native, (available_cpu_count() if workers is None
                    else int(workers))


def _build_masked_bilinear_stencil(
        x, y, source_valid, target_apply, *, fallback_radius=8,
        closed_edges=(), native=None, workers=None):
    """Build a convex, surface-type-aware bilinear stencil.

    Invalid source corners receive zero weight and the remaining weights are
    renormalized.  A target with no valid bilinear corner receives the
    NEAREST valid source cell, ties going to the lowest row and then the
    lowest column.  The search scans the ``fallback_radius`` disk first,
    exactly as it always has; a target with nothing there -- a land cell
    the source's land mask has as sea, such as a small island -- is
    searched further, and takes the nearest valid cell in the window only
    when that cell is nearer than any cell outside the window could be
    (``closed_edges`` names the window edges that are also edges of the
    whole source grid, beyond which nothing lies).  So a donor is always
    the nearest valid cell of the whole source grid, whatever the radius,
    and a target whose nearest cell this window cannot vouch for is
    refused with the radius that would decide it.

    Nothing distant is silent: every donor farther than
    :data:`DISTANT_DONOR_CELLS` is listed in the report with its target
    cell, its source cell and the distance.  Before the search went past
    the radius, a two-cell island 33 km from HRRR's nearest land refused
    a whole 750 m nest at radius 8.

    The arithmetic runs in the Rust preprocessing library
    (``gpuwm_masked_bilinear_stencil_f64``) on every CPU the process may
    use, byte-identical to the NumPy builder kept as its test oracle
    (:mod:`gpuwm.verify.hrrr_stencil_oracle`); ``native`` and ``workers``
    name the library and the worker count (see :func:`_stencil_engine`).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    source_valid = np.asarray(source_valid, dtype=bool)
    target_apply = np.asarray(target_apply, dtype=bool)
    if x.shape != y.shape or x.shape != target_apply.shape:
        raise ValueError("masked-bilinear target arrays must have equal shapes")
    if source_valid.ndim != 2:
        raise ValueError("masked-bilinear source_valid must be 2-D")
    if x.size == 0:
        raise ValueError("masked-bilinear stencil has no target points to map")
    fallback_radius = int(fallback_radius)
    closed_edges = tuple(sorted(set(closed_edges)))
    unknown = set(closed_edges) - set(WINDOW_EDGES)
    native, workers = _stencil_engine(native, workers)
    code, raw = native.masked_bilinear_stencil(
        x, y, source_valid, target_apply,
        fallback_radius=fallback_radius,
        closed_edges=sum(_EDGE_BITS[edge] for edge in closed_edges
                         if edge in _EDGE_BITS),
        edges_unknown=bool(unknown), distant_cells=float(DISTANT_DONOR_CELLS),
        listed=_DISTANT_DONORS_LISTED, workers=workers)
    if code == 3:
        raise ValueError("masked-bilinear coordinates must be finite")
    if code == 20:
        raise ValueError("fallback_radius must be non-negative")
    if code == 21:
        raise ValueError(
            f"closed_edges names {sorted(unknown)}; the window edges are "
            f"{', '.join(WINDOW_EDGES)}")
    if code == 22:
        raise ValueError("masked-bilinear coordinates leave the source window")
    counts = [int(value) for value in raw["counts"]]
    (source_valid_count, target_apply_count, direct_count,
     renormalized_count, fallback_count, cross_surface_count,
     negative_count, distant_count, unresolved, listed) = counts
    max_distance, min_raw, sum_min, sum_max, worst = (
        float(value) for value in raw["reals"])
    weights_convex, any_valid = (bool(value) for value in raw["flags"])
    if code == 23:
        worst_distance = worst if any_valid else None
        required_radius = (None if worst_distance is None
                           else int(np.ceil(worst_distance)))
        unresolved_targets = tuple(zip(*np.unravel_index(
            raw["unresolved"][:unresolved].astype(np.int64), x.shape)))
        reason = (
            "" if worst_distance is None else
            f"; the nearest one in the decoded source window is "
            f"up to {worst_distance:.1f} cells away, farther than "
            "the window reaches from those points, so a nearer one "
            "outside it cannot be ruled out")
        raise SurfaceDonorSearchError(
            f"no valid surface-matched HRRR donor within "
            f"{fallback_radius} cells for {unresolved} target "
            f"point(s){reason}",
            fallback_radius_cells=fallback_radius,
            required_radius_cells=required_radius,
            unresolved_targets=unresolved_targets,
            search_inputs=(x, y, source_valid, target_apply,
                           fallback_radius, closed_edges))
    if code:
        native._raise_native(code, "masked bilinear stencil")
    histogram = raw["histogram"]
    fallback_distance_histogram = {
        str(int(cell_radius)): int(histogram[cell_radius])
        for cell_radius in np.flatnonzero(histogram)}
    # A window whose four edges are all HRRR's own edges has nothing
    # beyond it, so its reach is unlimited (infinity in the search).  The
    # receipt says that as null, with closed_window_edges naming why:
    # infinity is not JSON, and the prepared cache refused to write a
    # receipt carrying it.
    distant_donors = []
    for k in range(listed):
        row, col = np.unravel_index(int(raw["distant_target"][k]), x.shape)
        reach = float(raw["distant_reach"][k])
        distant_donors.append({
            "target_index": [int(row), int(col)],
            "source_index": [int(raw["distant_source"][k][0]),
                             int(raw["distant_source"][k][1])],
            "distance_cells": float(raw["distant_distance"][k]),
            "window_reach_cells": reach if np.isfinite(reach) else None})
    if cross_surface_count:
        raise AssertionError(
            "masked-bilinear stencil selected an incompatible surface donor")
    if not weights_convex:
        raise AssertionError("masked-bilinear stencil is not finite and convex")
    if sum(fallback_distance_histogram.values()) != fallback_count:
        raise AssertionError("fallback distance histogram is incomplete")
    report = {
        "operator": "masked_convex_bilinear_with_nearest_valid_fallback",
        "source_valid_count": source_valid_count,
        "target_apply_count": target_apply_count,
        "direct_target_count": direct_count,
        "renormalized_target_count": renormalized_count,
        "fallback_target_count": fallback_count,
        "fallback_radius_cells": fallback_radius,
        "fallback_max_distance_cells": max_distance,
        "fallback_distance_ceiling_histogram_cells": (
            fallback_distance_histogram),
        "donor_rule": (
            "nearest surface-matched source cell, ties to the lowest row "
            "then the lowest column; past fallback_radius_cells only when "
            "nearer than any cell outside the window"),
        "closed_window_edges": list(closed_edges),
        "distant_donor_threshold_cells": DISTANT_DONOR_CELLS,
        "distant_donor_count": distant_count,
        "distant_donors": distant_donors,
        "distant_donors_not_listed": distant_count - len(distant_donors),
        "unresolved_target_count": 0,
        "cross_surface_donor_count": cross_surface_count,
        "donor_surface_class": "land",
        "minimum_nonzero_raw_support": (
            min_raw if direct_count else None),
        "weight_sum_minimum": sum_min,
        "weight_sum_maximum": sum_max,
        "negative_weight_count": negative_count,
    }
    shape = (4, *x.shape)
    return (raw["indices_y"].reshape(shape), raw["indices_x"].reshape(shape),
            raw["weights"].reshape(shape), report)


def _source_window_rotation(
        snapshot: HrrrNativeSnapshot) -> tuple[np.ndarray, np.ndarray]:
    """Return source-grid ``(SINALPHA, COSALPHA)`` on a bridge window."""

    source = hrrr_source_grid()
    x = np.arange(
        snapshot.i_start + 1,
        snapshot.i_start + snapshot.nx + 1,
        dtype=np.float64,
    )
    y = np.arange(
        snapshot.j_start + 1,
        snapshot.j_start + snapshot.ny + 1,
        dtype=np.float64,
    )
    _, longitude = source.ij_to_latlon(*np.meshgrid(x, y))
    difference = source.stand_lon - longitude
    difference = np.where(difference > 180.0, difference - 360.0, difference)
    difference = np.where(difference < -180.0, difference + 360.0, difference)
    alpha = source.hemi * source.cone * np.pi / 180.0 * difference
    return np.sin(alpha), np.cos(alpha)


def _require_source_physical_ranges(source: Mapping[str, np.ndarray]) -> None:
    """Admit the source fields, clamping only what sits ON a bound.

    A solid-land cell is stored at a land fraction of exactly 1.0 and
    saturated soil at a moisture fraction of exactly 1.0; both come back
    a hair above it, because the decode rounds.  Refusing those was the
    HRRR twin of the GFS field report -- ``1.0000000019`` was reproduced
    on both -- so the two unit fractions are now clamped within the
    round-off this pipeline carries and refuse beyond it.  The
    temperature and humidity ranges below are plausibility windows no
    real value approaches, and are untouched.

    SOILT keeps its 170..400 K band on source OPEN WATER (LANDSEA and
    XICE both below 0.5) and is admitted outside it on source LAND, the
    cells the soil initializer rebuilds TSK-to-TMN as real.exe does
    (:func:`gpuwm.ingest.soil.unreasonable_land_soil_columns`), which is
    the split :func:`_record_soil_field_stats` makes on the target.
    Refusing the band on land refused every HRRRv2 cycle over western
    snowpack, whose analyses carry land soil temperatures of 60 to 168 K.
    An ICE-covered water cell is admitted too: HRRR runs its land-ice
    column there, and on 2017-01-19 00Z a one-cell frozen lake at
    46.2N 113.3W carries the same 158 K snowpack column as the land
    around it.  Neither is a donor for a target land column, and target water
    starts from its skin temperature, so no such value reaches the model.
    An IN-band SOILT is admitted here as it always was; the soil
    initializer rebuilds the same way a snow-covered target land column
    whose top soil sits further below its skin than a snowpack allows
    (:func:`gpuwm.ingest.soil.snow_soil_below_skin_columns`), which is
    what the rest of those HRRRv2 snowpack columns carry (tops near 200 K
    under a 268 K skin).

    This is admission, not repair: a snapshot's field mapping is a
    read-only proxy, so the clamped copies decide the verdict here and
    the arrays are clamped again where they are consumed (the soil seam,
    and the mapped-output check below).  Clamping at the point of use is
    the accurate place for it -- nothing downstream inherits a value this
    function silently rewrote.
    """

    landsea, _ = clamp_bound_kissing(
        source["LANDSEA"], minimum=0.0, maximum=1.0)
    soilw, _ = clamp_bound_kissing(
        source["SOILW"], minimum=0.0, maximum=1.0)
    soilt = np.asarray(source["SOILT"])
    spfh = np.asarray(source["SPFH"])
    q2 = np.asarray(source["Q2"])
    if (not np.isfinite(landsea).all()
            or np.any((landsea < 0.0) | (landsea > 1.0))):
        raise ValueError("source HRRR LANDSEA is non-finite or outside 0..1")
    if (not np.isfinite(soilw).all()
            or np.any((soilw < 0.0) | (soilw > 1.0))):
        raise ValueError("source HRRR SOILW is non-finite or outside 0..1")
    if not np.isfinite(soilt).all():
        raise ValueError("source HRRR SOILT is non-finite")
    # Named breakage: a SOILT record out of band on OPEN WATER, where no
    # land or ice model runs, is not the snowpack analysis the land
    # rebuild answers but a mis-decoded or mis-mapped record, and
    # admitting it would pass that record's land cells to the rebuild as
    # if they were snowpack.  A source without XICE is judged on LANDSEA
    # alone, the stricter reading.
    open_water = np.asarray(landsea) < 0.5
    xice = source.get("XICE") if hasattr(source, "get") else None
    if xice is not None:
        open_water = open_water & ~(np.asarray(xice) >= 0.5)
    outside = (soilt < 170.0) | (soilt > 400.0)
    water_cells = int(np.count_nonzero(
        open_water & outside.reshape((-1,) + open_water.shape).any(axis=0)))
    if water_cells:
        raise ValueError(
            "source HRRR SOILT is outside 170..400 K on "
            f"{water_cells} source open-water cell(s) (LANDSEA and XICE "
            "below 0.5); only a land or ice column is admitted outside the "
            "range, and a land column is rebuilt TSK-to-TMN as real.exe "
            "rebuilds it, so a value out of range on open water is a "
            "broken SOILT record")
    if (not np.isfinite(spfh).all()
            or np.any((spfh < 0.0) | (spfh > 0.1))):
        raise ValueError("source HRRR SPFH is non-finite or outside 0..0.1")
    if (not np.isfinite(q2).all()
            or np.any((q2 < 0.0) | (q2 > 0.1))):
        raise ValueError("source HRRR Q2 is non-finite or outside 0..0.1")


#: Snow undershoot repairs already announced, one line per grid and field.
_REPORTED_SNOW_UNDERSHOOT: set = set()


def _zero_snow_undershoot(out, source, xp, *, target_name, valid_time):
    """Put the overlapping parabola's snow undershoot at zero, counted.

    Snow water and snow depth are mapped with the overlapping parabola
    above, which is not a weighted mean: beside a patch with little snow
    inside a snowpack (a valley floor that has nearly melted out) its
    negative weights take the patch below zero, by up to 9/32 of the snow
    around it.  From a source with
    no negative value every negative result is that undershoot and
    nothing else, so each goes to zero here, where the source that bounds
    it is known.  A source that is negative or not finite somewhere is
    left as mapped, for the snow admission (gpuwm/ingest/soil.py) to
    judge.  Every non-negative value is untouched.
    """
    for name in ("SNOW", "SNOWH"):
        values = np.asarray(source[name])
        if values.size == 0 or not np.isfinite(values).all() \
                or float(values.min()) < 0.0:
            continue
        mapped = out[name]
        below = mapped < 0
        count = int(xp.count_nonzero(below))
        if not count:
            continue
        lowest = float(mapped.min())
        out[name] = xp.where(below, xp.float32(0.0), mapped).astype(
            xp.float32, copy=False)
        key = (target_name, name)
        if key in _REPORTED_SNOW_UNDERSHOOT:
            continue
        _REPORTED_SNOW_UNDERSHOOT.add(key)
        when = (valid_time.isoformat() if hasattr(valid_time, "isoformat")
                else str(valid_time))
        print(
            f"HRRR {name} on {target_name} at {when}: {count} value(s) the "
            "overlapping parabola took below zero beside cells with little snow "
            f"(most negative {lowest:.4g}, source maximum "
            f"{float(values.max()):.4g}) put at 0; said once per grid",
            file=sys.stderr)


#: What a soil-mapping refusal or report calls its target when the caller
#: names nothing better.  Every caller in the product names it: a strip
#: refusal that says only "soil mapping requires ... land cells" tells a
#: coastal-domain owner nothing about WHICH of four independently mapped
#: boundary strips, on which domain, was being talked about.
DEFAULT_SOIL_TARGET_NAME = "the HRRR target grid"


def _native_edges_of(snapshot: HrrrNativeSnapshot) -> tuple[str, ...]:
    """The edges of this window that are also edges of HRRR's grid.

    Nothing lies beyond them, so a donor search near one need not rule
    out a nearer cell on the far side.
    """
    from gpuwm.ingest.hrrr_target import HRRR_SOURCE_NX, HRRR_SOURCE_NY

    return tuple(edge for edge, closed in (
        ("west", snapshot.i_start == 0),
        ("east", snapshot.i_start + snapshot.nx == HRRR_SOURCE_NX),
        ("south", snapshot.j_start == 0),
        ("north", snapshot.j_start + snapshot.ny == HRRR_SOURCE_NY),
    ) if closed)


#: (target, cells) pairs already announced by this process: the root's
#: boundary strips and every forecast hour map the same cells again.
_DISTANT_DONORS_ANNOUNCED: set[tuple[str, tuple]] = set()


def _named_distant_donors(report, snapshot: HrrrNativeSnapshot,
                          target_lat, target_lon, target_name: str):
    """The stencil report with each distant donor placed on the map.

    The stencil builder names cells by index; the receipt a person reads
    needs where they are: each distant donor gains its target cell's
    latitude and longitude, its HRRR cell as a native (i, j) and as a
    latitude and longitude, and the distance in kilometres.  One warning
    per target says how many land cells took a donor that far away.
    """
    report = dict(report)
    listed = []
    for entry in report.get("distant_donors", ()):
        row, col = entry["target_index"]
        source_row, source_col = entry["source_index"]
        donor_lat, donor_lon = snapshot.source_cell_latlon(
            source_row, source_col)
        listed.append({
            **entry,
            "target_lat": float(target_lat[row, col]),
            "target_lon": float(target_lon[row, col]),
            "donor_hrrr_index": {"i": int(source_col + snapshot.i_start),
                                 "j": int(source_row + snapshot.j_start)},
            "donor_lat": float(donor_lat),
            "donor_lon": float(donor_lon),
            "distance_km": float(entry["distance_cells"]
                                 * HRRR_GRID_SPACING_M / 1000.0),
        })
    report["distant_donors"] = listed
    count = int(report.get("distant_donor_count", 0))
    key = (target_name, tuple(tuple(entry["target_index"])
                              for entry in listed))
    if count and key not in _DISTANT_DONORS_ANNOUNCED:
        _DISTANT_DONORS_ANNOUNCED.add(key)
        from gpuwm import explain

        first = listed[0]
        farthest_km = (float(report["fallback_max_distance_cells"])
                       * HRRR_GRID_SPACING_M / 1000.0)
        explain.warn(
            f"{target_name}: {count} land cell(s) that HRRR's land mask "
            f"has as sea took their soil from the nearest HRRR land cell, "
            f"up to {farthest_km:.0f} km away (the first, at "
            f"{first['target_lat']:.4f}, {first['target_lon']:.4f}, from "
            f"HRRR land at {first['donor_lat']:.4f}, "
            f"{first['donor_lon']:.4f}); the preparation receipt lists "
            "each one under land_stencil.distant_donors",
            "Small islands and narrow spits are land on the forecast grid "
            "and sea on HRRR's 3 km grid, so HRRR has no soil state for "
            "them.  Each takes the soil temperature and moisture columns "
            "of the nearest HRRR land cell, found by a search that is "
            "only accepted when the decoded HRRR window shows no nearer "
            "land cell can exist.")
    return report


def _trim_leaves_every_cell_a_donor(error: SurfaceDonorSearchError,
                                    side: str, trim: int) -> bool:
    """Whether a domain trimmed by ``trim`` cells on ``side`` maps.

    Runs the same donor search on what the trim would leave, over the
    smallest window that domain could be given: the remaining mass
    points' own donor box.  Its real window also covers the staggered
    points and the interpolation halo, so it is at least this large and
    every donor vouched for here is vouched for there.  A trim can
    shrink the window past a donor that was vouched for before, which
    is why it is checked rather than assumed.  A nest mapped on its
    parent's window keeps that window whatever its own size, so for a
    nest this check is stricter than it has to be: it can pass over a
    trim that would work, and never names one that would not.
    """
    if error.search_inputs is None:
        return False
    x, y, valid, apply, radius, closed = error.search_inputs
    rows, cols = x.shape
    keep = {
        "north (j-max)": (slice(0, rows - trim), slice(None)),
        "south (j-min)": (slice(trim, rows), slice(None)),
        "east (i-max)": (slice(None), slice(0, cols - trim)),
        "west (i-min)": (slice(None), slice(trim, cols)),
    }[side]
    x, y, apply = x[keep], y[keep], apply[keep]
    if x.size == 0:
        return False
    ny, nx = valid.shape
    low_x = max(0, min(int(np.ceil(x.min() - radius)), int(np.floor(x.min()))))
    high_x = min(nx - 1, max(int(np.floor(x.max() + radius)),
                             int(np.floor(x.max())) + 1))
    low_y = max(0, min(int(np.ceil(y.min() - radius)), int(np.floor(y.min()))))
    high_y = min(ny - 1, max(int(np.floor(y.max() + radius)),
                             int(np.floor(y.max())) + 1))
    still_closed = tuple(edge for edge, at_edge in (
        ("west", low_x == 0), ("east", high_x == nx - 1),
        ("south", low_y == 0), ("north", high_y == ny - 1),
    ) if at_edge and edge in closed)
    try:
        _build_masked_bilinear_stencil(
            x - low_x, y - low_y, valid[low_y:high_y + 1, low_x:high_x + 1],
            apply, fallback_radius=radius, closed_edges=still_closed)
    except SurfaceDonorSearchError:
        return False
    return True


def _validated_donor_remediation(error: SurfaceDonorSearchError,
                                 snapshot: HrrrNativeSnapshot,
                                 target_name: str,
                                 target_shape: tuple[int, int]) -> str:
    """Advice that works, computed before printing.

    * the radius that reaches a donor for every unfilled cell was
      measured by the stencil builder.  Up to the supported maximum it
      is the advice: the source window it needs is this snapshot's
      window grown by the radius increase on every side (the fallback
      bound moves cell-for-cell with the radius, the parabolic bound not
      at all), stopped at HRRR's own edge, where nothing lies beyond,
      and the coverage test accepts any such window;
    * past the maximum the raise is named impossible and a trim is
      printed instead: the smallest one-side trim that removes every
      unfillable cell AND, run through the same donor search on the
      domain it leaves (:func:`_trim_leaves_every_cell_a_donor`), leaves
      every remaining land cell a donor.  Since the search can reach
      past the radius, a trim that shrinks the window could cost a cell
      its donor, so this is checked, not assumed; when no trim passes,
      the advice is to move the domain.

    A raise is set where the run's own radius is set: the
    ``surface_fallback_radius_cells`` key of its d01 target document,
    which `gpuwm domain` writes beside the experiment.
    """
    from gpuwm.ingest.hrrr_target import (HRRR_SOURCE_NX, HRRR_SOURCE_NY,
                                          SURFACE_FALLBACK_RADIUS_MAX)

    move = ("move the domain so its land sits inside HRRR's land "
            "coverage")
    required = error.required_radius_cells
    if required is None:
        return ("Raising surface_fallback_radius_cells cannot help: the "
                "decoded HRRR source window holds no surface-matched "
                f"donor for these cells at any radius.  Instead, {move}.")

    growth = required - error.fallback_radius_cells
    grown_i = (max(0, snapshot.i_start - growth),
               min(HRRR_SOURCE_NX - 1,
                   snapshot.i_start + snapshot.nx - 1 + growth))
    grown_j = (max(0, snapshot.j_start - growth),
               min(HRRR_SOURCE_NY - 1,
                   snapshot.j_start + snapshot.ny - 1 + growth))
    if required <= SURFACE_FALLBACK_RADIUS_MAX:
        return (f"Raising surface_fallback_radius_cells to {required} "
                "reaches a surface-matched donor for every unfilled "
                "cell; its source window grows to at most "
                f"i={grown_i[0]}..{grown_i[1]}, "
                f"j={grown_j[0]}..{grown_j[1]} of the native HRRR grid "
                f"i=0..{HRRR_SOURCE_NX - 1}, j=0..{HRRR_SOURCE_NY - 1}, "
                "stopping at HRRR's own edge.  Set it in the run's d01 "
                "target document (the .d01-target.json file beside the "
                f"experiment).  Alternatively, {move}.")

    reason = (f"that exceeds the supported maximum of "
              f"{SURFACE_FALLBACK_RADIUS_MAX}")
    advice = (f"Raising surface_fallback_radius_cells cannot work here: "
              f"the nearest donors need radius {required}, and {reason}.")

    rows = [target[0] for target in error.unresolved_targets
            if len(target) == 2]
    cols = [target[1] for target in error.unresolved_targets
            if len(target) == 2]
    if rows and len(rows) == len(error.unresolved_targets):
        target_rows, target_cols = int(target_shape[0]), int(target_shape[1])
        for trim, side in sorted((
                (target_rows - min(rows), "north (j-max)"),
                (max(rows) + 1, "south (j-min)"),
                (target_cols - min(cols), "east (i-max)"),
                (max(cols) + 1, "west (i-min)"))):
            if _trim_leaves_every_cell_a_donor(error, side, trim):
                return advice + (
                    f"  What works: trim {trim} cell(s) from its {side} "
                    f"side, which removes every unfillable land cell of "
                    f"{target_name} and leaves every remaining one a "
                    f"donor, or {move}.")
        advice += (f"  No trim of one side removes those cells and leaves "
                   f"every remaining land cell a donor; instead, {move}.")
    else:
        advice += f"  Instead, {move}."
    return advice


def _record_soil_field_stats(report, name, source, candidate,
                             source_land, target_land, limits, *,
                             target_name=DEFAULT_SOIL_TARGET_NAME,
                             land_columns_rebuilt=False):
    """Record one mapped soil field's admission checks and land diagnostics.

    A target with NO land cells is a legitimate configuration, not a
    failure: every one of its soil cells is the target-water fill
    (SOILT = SKINTEMP, SOILW = 1) that the mapper already wrote, and no
    land donor is consulted to produce it.  The four specified-boundary
    strips are mapped independently, so a Pacific-coast domain's
    all-water west strip used to abort a preparation whose other three
    strips and whose interior were entirely ordinary -- and the message
    named neither the strip nor the domain.

    The land-window statistics below are exactly that: a comparison of
    the SOURCE land window against the TARGET land cells.  With no land
    on one side there is no comparison to make, so the diagnostics are
    recorded as absent with the reason, and the admission checks that do
    not need land -- finiteness and physical range over EVERY mapped
    cell -- still run.  Previously the empty-land raise came first, so a
    zero-land target skipped the range check entirely.

    The case that genuinely cannot be mapped is the opposite one: target
    land cells with no reachable HRRR land donor.  The stencil builder
    refuses that before this function is ever called, and it counts the
    unresolved points.

    ``land_columns_rebuilt`` is SOILT's: a target LAND column outside the
    limits is admitted and counted here, because the soil initializer
    rebuilds it TSK-to-TMN as real.exe does
    (:func:`gpuwm.ingest.soil.unreasonable_land_soil_columns`); a
    non-finite value, or one outside the limits off land, is still
    refused.
    """
    if hasattr(candidate, "get"):
        candidate = candidate.get()
    candidate = np.asarray(candidate).astype(np.float64, copy=False)
    source = np.asarray(source, dtype=np.float64)
    source_values = source[:, source_land]
    target_values = candidate[:, target_land]
    lower, upper = limits
    # Interpolation inherits the source's bound-kissing cells, so the
    # mapped output is admitted on the same terms the source was.
    candidate, _ = clamp_bound_kissing(candidate, minimum=lower, maximum=upper)
    outside = (candidate < lower) | (candidate > upper)
    rebuilt_land_columns = 0
    if land_columns_rebuilt:
        land_outside = np.any(outside, axis=0) & target_land
        rebuilt_land_columns = int(np.count_nonzero(land_outside))
        outside = outside & ~target_land[None, ...]
    if not np.isfinite(candidate).all() or np.any(outside):
        raise ValueError(
            f"mapped HRRR {name} for {target_name} is non-finite or outside "
            f"{lower}..{upper}")
    # Present only when a land column is left for the soil initializer's
    # rebuild, so a healthy cycle's report is unchanged.
    rebuilt_entry = (
        {"target_land_columns_outside_limits_rebuilt_tsk_to_tmn":
            rebuilt_land_columns} if rebuilt_land_columns else {})
    if source_values.shape[1] == 0 or target_values.shape[1] == 0:
        empty = "target" if target_values.shape[1] == 0 else "source window"
        report[name] = {
            "physical_limits": [lower, upper],
            "all_target_minimum": float(np.min(candidate)),
            "all_target_maximum": float(np.max(candidate)),
            "land_window_statistics": None,
            "land_window_statistics_absent_because": (
                f"{target_name} has no land cells in the {empty}; every "
                "soil cell here is the target-water fill and no land "
                "donor is consulted, so there is no source-to-target "
                "land comparison to report"),
            "source_land_cell_count": int(source_values.shape[1]),
            "target_land_cell_count": int(target_values.shape[1]),
            **rebuilt_entry,
        }
        return
    source_min = np.min(source_values, axis=1)
    source_max = np.max(source_values, axis=1)
    target_min = np.min(target_values, axis=1)
    target_max = np.max(target_values, axis=1)
    tolerance = 4.0 * np.finfo(np.float32).eps * np.maximum(
        1.0, np.maximum(np.abs(source_min), np.abs(source_max)))
    convex_violations = np.sum(
        (target_values < (source_min[:, None] - tolerance[:, None]))
        | (target_values > (source_max[:, None] + tolerance[:, None])),
        axis=1,
    )
    report[name] = {
        "physical_limits": [lower, upper],
        "all_target_minimum": float(np.min(candidate)),
        "all_target_maximum": float(np.max(candidate)),
        "source_land_minimum_by_depth": source_min.tolist(),
        "source_land_maximum_by_depth": source_max.tolist(),
        "target_land_minimum_by_depth": target_min.tolist(),
        "target_land_maximum_by_depth": target_max.tolist(),
        "source_window_land_mean_by_depth": np.mean(
            source_values, axis=1).tolist(),
        "target_land_mean_by_depth": np.mean(target_values, axis=1).tolist(),
        "target_minus_source_window_land_mean_by_depth": (
            np.mean(target_values, axis=1)
            - np.mean(source_values, axis=1)).tolist(),
        "convex_bound_violation_count_by_depth": convex_violations.tolist(),
        "convex_bounds_conserved": bool(np.count_nonzero(convex_violations) == 0),
        "mean_delta_note": (
            "Diagnostic only: source window and target land masks/areas differ; "
            "this is not an integral-conservation claim."),
        **rebuilt_entry,
    }


def interpolate_hrrr_to_lambert(
        snapshot: HrrrNativeSnapshot,
        grid: LambertGrid, *, target_landmask,
        soil_mapping_report: MutableMapping[str, object] | None = None,
        surface_fallback_radius: int = 8,
        backend="cuda", workers: int | None = None,
        chem_rows=(), chem_source=None,
        cpu_bridge: Path | str | None = None,
        target_name: str = DEFAULT_SOIL_TARGET_NAME,
        lattice_parent: LambertGrid | None = None) -> HorizontalSnapshot:
    """Interpolate a verified HRRR window to one WRF Lambert C grid.

    ``chem_rows`` are active species rows; ``chem_source`` is the selected
    SourceRow. Its density parameters drive conversion before mapping.
    ``boundary_tracers`` on the returned snapshot carries row-unit mass-grid
    arrays for initialize_real or an independently prepared boundary time.

    Atmospheric continuous fields use WPS's overlapping-parabolic operator.
    Soil uses non-negative bilinear weights restricted to source land; a
    land cell with no land corner takes the nearest HRRR land cell, past
    ``surface_fallback_radius`` only where this window shows no nearer one
    can exist (see :func:`_build_masked_bilinear_stencil`).
    This intentionally repairs WPS ``sixteen_pt`` overshoot (including
    negative soil moisture) rather than reproducing it.  Target-water soil is
    filled with target skin temperature and unit moisture, as WRF soil setup
    expects.  HRRR winds are grid-relative.  They are converted to the earth
    basis on the source mass grid, interpolated to each target staggering, and
    converted to the target Lambert basis before placing U/V on C-grid faces.

    ``target_name`` is what a refusal or a mapping report calls this
    grid.  The four specified-boundary strips are mapped by separate
    calls, so "the west boundary strip of domain 1" and "domain 2" are
    different targets whose soil refusals must not be confusable: a
    coastal domain's all-water west strip once aborted a preparation
    with a sentence naming neither the strip nor the domain.

    ``lattice_parent`` is the domain ``grid`` was cut from (a boundary
    strip's domain).  When that domain is the native grid, the strip
    takes the identity route its domain took
    (:func:`gpuwm.ingest.hrrr_target.native_lattice_placement`).
    """
    if not isinstance(snapshot, HrrrNativeSnapshot):
        raise TypeError("snapshot must be an HrrrNativeSnapshot")
    if not isinstance(grid, LambertGrid):
        raise TypeError("grid must be a LambertGrid")
    if chem_rows and chem_source is None:
        raise ValueError("active boundary rows need the selected source row; prevents conversion from an ambiguous source")
    from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend

    engine = resolve_preprocess_backend(
        backend, workers=workers, cpu_bridge=cpu_bridge)
    xp = engine.array_module
    mass_lat, mass_lon = grid.latlon_mass()
    u_lat, u_lon = grid.latlon_u()
    v_lat, v_lon = grid.latlon_v()
    plan_type = (
        _ProjectedGpuPlan if getattr(engine, "name", None) == "cuda"
        else _ProjectedCpuPlan)
    # The target IS the native grid, or a whole-cell piece of it (one of
    # the native grid's own boundary strips): copy index for index
    # (gpuwm.ingest.hrrr_target.native_lattice_placement).  Declaring it
    # only for the whole grid sent that grid's boundary strips down the
    # interpolated route, which refused their edge points (acceptance
    # D-05, 2.8.8).
    from gpuwm.ingest.hrrr_target import native_lattice_placement

    placement = native_lattice_placement(grid, parent=lattice_parent)
    identity = placement is not None
    if plan_type is _ProjectedGpuPlan:
        mass_plan = plan_type(snapshot, mass_lat, mass_lon,
                              identity=identity, placement=placement)
        u_plan = plan_type(snapshot, u_lat, u_lon, identity=identity,
                           placement=placement)
        v_plan = plan_type(snapshot, v_lat, v_lon, identity=identity,
                           placement=placement)
    else:
        mass_plan = plan_type(snapshot, mass_lat, mass_lon, engine,
                              identity=identity, placement=placement)
        u_plan = plan_type(snapshot, u_lat, u_lon, engine,
                           identity=identity, placement=placement)
        v_plan = plan_type(snapshot, v_lat, v_lon, engine,
                           identity=identity, placement=placement)
    source = snapshot.fields
    _require_source_physical_ranges(source)
    target_landmask = np.asarray(target_landmask)
    if target_landmask.shape != mass_plan.target_shape:
        raise ValueError(
            "target_landmask shape does not match the target mass grid")
    if (not np.isfinite(target_landmask).all()
            or np.any((target_landmask != 0.0) & (target_landmask != 1.0))):
        raise ValueError("target_landmask must contain only finite 0/1 values")
    target_land = target_landmask.astype(bool)
    source_land = np.asarray(source["LANDSEA"]) >= 0.5
    # Soil is built and (on the CPU) applied in the Rust preprocessing
    # library under both backends: the CPU backend's own library and
    # worker count, or the ladder's library on every CPU under CUDA.  A
    # library without the stencil is refused here, before any field.
    from gpuwm.ingest.horiz import _masked_chain_for_backend

    stencil_native, stencil_workers = _stencil_engine(
        *_masked_chain_for_backend(engine))
    try:
        land_stencil = mass_plan.masked_bilinear_stencil(
            source_land, target_land,
            fallback_radius=surface_fallback_radius,
            closed_edges=_native_edges_of(snapshot),
            native=stencil_native, workers=stencil_workers)
    except SurfaceDonorSearchError as error:
        # The one soil case that genuinely cannot be mapped: this target
        # has land cells and the HRRR window has no land within reach of
        # them.  The stencil builder is generic and measures the facts;
        # only here is it known WHICH grid was being mapped and where
        # its window sits on the native HRRR grid, which is what the
        # advice is computed from.
        raise ValueError(
            f"HRRR soil mapping for {target_name} cannot fill its land "
            f"cells: {error}.  "
            + _validated_donor_remediation(
                error, snapshot, target_name, target_land.shape)
        ) from error
    out: dict[str, object] = {}
    for name in (
            "PRES", "HGT", "TT",
            "SPFH", "PSFC", "SOILHGT", "SKINTEMP", "SNOW", "SNOWH",
            "T2", "Q2"):
        out[name] = mass_plan.apply(source[name], method="parabolic")
    _zero_snow_undershoot(out, source, xp, target_name=target_name,
                          valid_time=snapshot.valid_time)
    # WPS METGRID.TBL routes hydrometeor mass through
    # ``four_pt+average_4pt`` rather than the overshooting parabolic operator.
    # Bilinear interpolation preserves both non-negativity and compact support.
    if "PMSL" in source:
        out["PMSL"] = mass_plan.apply(source["PMSL"], method="parabolic")
    if "VEGFRA" in source:
        out["VEGFRA"] = mass_plan.apply(source["VEGFRA"], method="bilinear")
    for name in ("QC", "QI", "QR", "QS", "QG"):
        out[name] = mass_plan.apply(source[name], method="bilinear")
    aerosol_mapping_counts = {}
    # The analyzed aerosol number pair, when the bridge published it.
    # Native finite payloads retain nearest interpolation. Where the bridge
    # preserved a source bitmap, the Rust chain sees the original mask at
    # fractional target coordinates before any zero may become a donor.
    for name in ("QNWFA", "QNIFA"):
        if name in source:
            mask = source.get(name + "_SOURCE_MASK")
            if mask is None:
                out[name] = mass_plan.apply(source[name], method="nearest")
            else:
                mapped, counts = stencil_native.missing_value_chain(
                    source[name], mask, mass_plan.y_host, mass_plan.x_host,
                    workers=stencil_workers)
                nearest = mass_plan.apply(source[name], method="nearest")
                nearest_valid = mass_plan.apply(mask, method="nearest")
                out[name] = xp.where(nearest_valid != 0, nearest,
                                     xp.asarray(mapped, dtype=xp.float32))
                aerosol_mapping_counts[name] = dict(zip(
                    ("nearest_neighbor", "four_pt", "average_4pt", "zero"),
                    [int(v) for v in counts.sum(axis=0)]))
            from gpuwm.ingest.host_arrays import deepest_level

            surface = deepest_level(
                out["PRES"], out[name],
                workers=getattr(engine, "host_step_workers", None))
            if surface is None:
                deepest = xp.argmax(out["PRES"], axis=0)[None, ...]
                surface = xp.take_along_axis(out[name], deepest, axis=0)[0]
            out[name + "_SFC"] = surface
    for name in snapshot.extra_fields:
        out[name] = mass_plan.apply(source[name], method="bilinear")
    from .native_extras import map_boundary_rows
    boundary_tracers = map_boundary_rows(
        source, chem_rows, chem_source, mass_plan, cpu_bridge=cpu_bridge)
    # Do not derive/map RH here.  With FLAG_SH, real.exe diagnoses rh_gc from
    # the already horizontally mapped SPECHUMD, TT, and PRES.  initialize_real
    # mirrors that order; retaining a separately mapped 50-level RH field is
    # both scientifically different and avoidable target-grid residency.
    # initialize_real uses WPS's GHT spelling; retain HGT as the native-GRIB
    # diagnostic spelling so oracle reports stay explicit.
    out["GHT"] = out["HGT"]
    for name, water_fill in (("SOILT", out["SKINTEMP"]), ("SOILW", 1.0)):
        out[name] = land_stencil.apply_selected(
            source[name], target_land, water_fill)
    # LANDSEA is the model/static surface classification.  Preserve the
    # nearest source classification separately for WPS-oracle diagnostics.
    out["SOURCE_LANDSEA"] = mass_plan.apply(
        source["LANDSEA"], method="nearest")
    out["LANDSEA"] = engine.float32(target_landmask)
    out["XICE"] = mass_plan.apply(source["XICE"], method="nearest")
    source_sina, source_cosa = _source_window_rotation(snapshot)

    def map_grid_relative_wind(u_name, v_name):
        source_u = engine.float32(source[u_name])
        source_v = engine.float32(source[v_name])
        source_sina_backend = engine.float32(source_sina)
        source_cosa_backend = engine.float32(source_cosa)
        if source_u.shape != source_v.shape:
            raise ValueError("u_grid and v_grid shapes differ")
        source_u_earth = (
            source_u * source_cosa_backend - source_v * source_sina_backend)
        source_v_earth = (
            source_v * source_cosa_backend + source_u * source_sina_backend)
        u_earth_at_u = u_plan.apply(source_u_earth, method="parabolic")
        v_earth_at_u = u_plan.apply(source_v_earth, method="parabolic")
        target_sina_u, target_cosa_u = lambert_rotation(grid, "u")
        target_u = engine.rotate_earth_to_grid(
            u_earth_at_u, v_earth_at_u,
            target_sina_u, target_cosa_u)[0]
        del u_earth_at_u, v_earth_at_u
        # Build V after U so four full target-face earth-wind temporaries do
        # not coexist on large domains.  CuPy can reuse the released blocks.
        u_earth_at_v = v_plan.apply(source_u_earth, method="parabolic")
        v_earth_at_v = v_plan.apply(source_v_earth, method="parabolic")
        target_sina_v, target_cosa_v = lambert_rotation(grid, "v")
        target_v = engine.rotate_earth_to_grid(
            u_earth_at_v, v_earth_at_v,
            target_sina_v, target_cosa_v)[1]
        return target_u, target_v

    out["UU"], out["VV"] = map_grid_relative_wind("U_MASS", "V_MASS")
    out["U10"], out["V10"] = map_grid_relative_wind(
        "U10_MASS", "V10_MASS")
    # sfcprs2 must compare the HRRR PSFC surface with its own horizontally
    # consistent orography, not the target GEOG terrain.
    out["SOURCE_OROGRAPHY"] = out["SOILHGT"]
    skin_host = out["SKINTEMP"]
    if hasattr(skin_host, "get"):
        skin_host = skin_host.get()
    skin_host = np.asarray(skin_host)
    if (not np.isfinite(skin_host).all()
            or np.any((skin_host < 170.0) | (skin_host > 400.0))):
        raise ValueError(
            f"mapped HRRR SKINTEMP for {target_name} is non-finite or "
            "outside 170..400 K")
    local_report: dict[str, object] = {
        "target": target_name,
        "policy": (
            "source-land-masked convex bilinear; nearest-valid fallback, "
            "past the radius only where the window vouches for it; "
            "target-water SOILT=SKINTEMP and SOILW=1"),
        "integral_conservation_claimed": False,
        "preprocess_backend": engine.receipt(),
        # WHICH implementation of the projected horizontal operator ran.
        # The backend receipt above says "cpu", and "cpu" now covers two
        # implementations that agree bit for bit but not within an order
        # of magnitude in wall time -- so a receipt that only said "cpu"
        # could not tell a slow run from a fast one after the fact.
        "projected_horizontal_operator": mass_plan.operator,
        "aerosol_missing": aerosol_mapping_counts,
        "wind_rotation": {
            "policy": (
                "source_grid_to_earth_then_earth_to_target_grid; "
                "interpolation occurs in the earth-relative basis"),
            "source_projection": {
                "truelat1": 38.5,
                "truelat2": 38.5,
                "stand_lon": -97.5,
            },
            "target_projection": {
                "truelat1": grid.truelat1,
                "truelat2": grid.truelat2,
                "stand_lon": grid.stand_lon,
            },
        },
        "source_land_count": int(np.count_nonzero(source_land)),
        "source_water_count": int(np.count_nonzero(~source_land)),
        "target_land_count": int(np.count_nonzero(target_land)),
        "target_water_count": int(np.count_nonzero(~target_land)),
        "land_stencil": _named_distant_donors(
            land_stencil.report, snapshot, mass_lat, mass_lon, target_name),
        "fields": {},
    }
    _record_soil_field_stats(
        local_report["fields"], "SOILT", source["SOILT"], out["SOILT"],
        source_land, target_land, (170.0, 400.0),
        target_name=target_name, land_columns_rebuilt=True)
    _record_soil_field_stats(
        local_report["fields"], "SOILW", source["SOILW"], out["SOILW"],
        source_land, target_land, (0.0, 1.0), target_name=target_name)
    if soil_mapping_report is not None:
        if len(soil_mapping_report):
            raise ValueError("soil_mapping_report must be empty")
        soil_mapping_report.update(local_report)
    return HorizontalSnapshot(
        valid_time=snapshot.valid_time,
        levels_hpa=HRRR_HYBRID_LEVELS,
        fields=out,
        boundary_tracers=boundary_tracers,
        withheld_fields=getattr(snapshot, "withheld_fields", None) or None,
    )


__all__ = [
    "HRRR_EARTH_RADIUS_M",
    "HRRR_GRID_SPACING_M",
    "HRRR_HYBRID_LEVELS",
    "HRRR_SOIL_DEPTHS_M",
    "HRRR_WPS_EQUIVALENT_DX_M",
    "HrrrNativeSnapshot",
    "hrrr_source_grid",
    "interpolate_hrrr_to_lambert",
    "load_hrrr_native_series",
    "load_hrrr_pipeline_ready_window",
    "load_hrrr_native_window",
    "verified_hrrr_native_bridge",
]
