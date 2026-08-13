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
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(root: Path, expected_manifest_sha256: str) -> None:
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
    required = {
        "status": "PASS",
        "atmosphere_selected_per_time": "561",
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
    if not values["qice_mapping"].startswith(
            "PASS discipline=0 category=1 parameter=82 level_type=105"):
        raise ValueError("gate.txt does not bind current-HRRR CIMIXR/QICE")
    if not values["cross_time_inventory"].startswith("PASS"):
        raise ValueError("gate.txt cross-time inventory did not pass")
    return values


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

    def __post_init__(self) -> None:
        if self.forecast_hour not in range(49):
            raise ValueError("the native bridge supports source leads f00..f48")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


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
    _verify_manifest(root, expected_manifest_sha256)
    gate = _read_gate(root)
    return _load_verified_hrrr_native_window(root, gate, forecast_hour)


def _load_verified_hrrr_native_window(
        root: Path, gate: Mapping[str, str],
        forecast_hour: int) -> HrrrNativeSnapshot:
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
    for name in _ATMOSPHERE_2D:
        fields[name] = _map_f32(
            atmosphere_dir / f"{name}.f32le", (ny, nx))
    for name in _SOIL_3D:
        fields[name] = _map_f32(soil_dir / f"{name}.f32le", (9, ny, nx))
    return HrrrNativeSnapshot(
        valid_time=cycle + timedelta(hours=forecast_hour),
        forecast_hour=forecast_hour,
        i_start=i_start,
        j_start=j_start,
        ny=ny,
        nx=nx,
        fields=fields,
    )


def load_hrrr_native_series(
        root, forecast_hours, *,
        expected_manifest_sha256: str) -> tuple[HrrrNativeSnapshot, ...]:
    """Verify one bridge publication once and map several forecast hours."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"HRRR bridge directory is missing: {root}")
    _verify_manifest(root, expected_manifest_sha256)
    gate = _read_gate(root)
    requested = tuple(int(hour) for hour in forecast_hours)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("forecast_hours must be a non-empty unique sequence")
    return tuple(
        _load_verified_hrrr_native_window(root, gate, hour)
        for hour in requested
    )


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


def _projected_index_geometry(snapshot: HrrrNativeSnapshot,
                              target_lat, target_lon):
    """Resolve exact zero-based HRRR-window interpolation coordinates."""

    source_x, source_y = hrrr_source_grid().latlon_to_ij(
        target_lat, target_lon)
    # LambertGrid coordinates are one-based; bridge window origins are
    # zero-based indices in the original south-to-north GRIB scan.
    global_x = np.asarray(source_x, dtype=np.float64) - 1.0
    global_y = np.asarray(source_y, dtype=np.float64) - 1.0
    global_ix = np.floor(global_x).astype(np.int64)
    global_iy = np.floor(global_y).astype(np.int64)
    ix = global_ix - snapshot.i_start
    iy = global_iy - snapshot.j_start
    if (np.min(ix - 1) < 0 or np.max(ix + 2) >= snapshot.nx
            or np.min(iy - 1) < 0 or np.max(iy + 2) >= snapshot.ny):
        raise ValueError(
            "HRRR bridge window lacks the four-point interpolation halo")
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
    from gpuwm import explain

    explain.warn(
        "projected-source horizontal mapping is running the NumPy mirror "
        f"instead of the native operator ({reason}).  It is the same "
        "arithmetic and it is slow: the native entry evaluates the same "
        "stencil across every core with no whole-array temporaries.  "
        "Rebuild the CPU preprocessing bridge (cd tools/grib1_bridge && "
        "cargo build --release --locked --offline) or stage a current "
        "bridges bundle.",
        "The NumPy mirror issues 32 fancy-index gathers per 3-D field and "
        "evaluates roughly 150 whole-target-shape temporaries inside the "
        "five `oned` calls, on one core, to produce one target-shape "
        "answer.  Nested HRRR preparation is dominated by this operator, "
        "so an install that quietly inherits the mirror pays it on every "
        "child of every run.")


class _ProjectedGpuPlan:
    """WPS overlapping-parabolic interpolation on projected source indices."""

    operator = PROJECTED_OPERATOR_CUDA

    def __init__(self, snapshot: HrrrNativeSnapshot, target_lat, target_lon):
        cp = _cupy()
        (x, y, ix, iy, nearest_ix, nearest_iy,
         global_ix, global_iy) = _projected_index_geometry(
             snapshot, target_lat, target_lon)
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
        return cp.where(result == tiny, zero, result).astype(
            cp.float32, copy=False)

    def masked_bilinear_stencil(
            self, source_valid, target_apply, *, fallback_radius=8):
        indices_y, indices_x, weights, report = (
            _build_masked_bilinear_stencil(
                self.x_host, self.y_host, source_valid, target_apply,
                fallback_radius=fallback_radius))
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
                 backend):
        (x, y, ix, iy, nearest_ix, nearest_iy,
         global_ix, global_iy) = _projected_index_geometry(
             snapshot, target_lat, target_lon)
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
            return self._native.apply(field, method=method)
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
        return np.asarray(np.where(result == tiny, zero, result),
                          dtype=np.float32)

    def masked_bilinear_stencil(
            self, source_valid, target_apply, *, fallback_radius=8):
        indices_y, indices_x, weights, report = (
            _build_masked_bilinear_stencil(
                self.x_host, self.y_host, source_valid, target_apply,
                fallback_radius=fallback_radius))
        return _CpuMaskedBilinearStencil(
            self.source_shape, indices_y, indices_x, weights, report)


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


@dataclass(frozen=True)
class _CpuMaskedBilinearStencil:
    """Host equivalent of the convex four-donor surface stencil."""

    source_shape: tuple[int, int]
    indices_y: np.ndarray
    indices_x: np.ndarray
    weights: np.ndarray
    report: Mapping[str, object]

    def __init__(self, source_shape, indices_y, indices_x, weights, report):
        object.__setattr__(self, "source_shape", tuple(source_shape))
        object.__setattr__(
            self, "indices_y", np.asarray(indices_y, dtype=np.int32))
        object.__setattr__(
            self, "indices_x", np.asarray(indices_x, dtype=np.int32))
        object.__setattr__(
            self, "weights", np.asarray(weights, dtype=np.float32))
        object.__setattr__(self, "report", MappingProxyType(dict(report)))

    def apply(self, field):
        field = np.asarray(field, dtype=np.float32)
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
        return np.asarray(result, dtype=np.float32)


#: Exact nearest-donor distances are measured per unresolved point when
#: the fallback search fails; beyond this many failing points the domain
#: is misplaced rather than under-radiused, and the measurement (whose
#: cost is points x valid cells) says nothing a placement fix needs.
_DONOR_ANALYSIS_MAX_POINTS = 512


class SurfaceDonorSearchError(ValueError):
    """No surface-matched donor within the configured fallback radius.

    Carries the facts remediation has to be computed from: the failing
    target cells and the smallest integer radius whose donor disk
    reaches a valid source cell for all of them (``None`` when the
    window holds no valid donor at any radius, or when the failure is
    too large for per-point measurement).  The stencil builder is
    generic and states facts only; the caller that knows WHICH grid was
    being mapped -- and where its window sits on the native HRRR grid --
    turns them into advice validated against the coverage guard.
    """

    def __init__(self, message, *, fallback_radius_cells,
                 required_radius_cells, unresolved_targets):
        super().__init__(message)
        self.fallback_radius_cells = int(fallback_radius_cells)
        self.required_radius_cells = (
            None if required_radius_cells is None
            else int(required_radius_cells))
        self.unresolved_targets = tuple(
            tuple(int(index) for index in target)
            for target in unresolved_targets)


def _build_masked_bilinear_stencil(
        x, y, source_valid, target_apply, *, fallback_radius=8):
    """Build a convex, surface-type-aware bilinear stencil on the CPU.

    Invalid source corners receive zero weight and the remaining weights are
    renormalized.  A target with no valid bilinear corner receives its nearest
    valid source donor within ``fallback_radius`` cells.  The explicit finite
    radius prevents a silently distant land/ocean substitution.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    source_valid = np.asarray(source_valid, dtype=bool)
    target_apply = np.asarray(target_apply, dtype=bool)
    if x.shape != y.shape or x.shape != target_apply.shape:
        raise ValueError("masked-bilinear target arrays must have equal shapes")
    if source_valid.ndim != 2:
        raise ValueError("masked-bilinear source_valid must be 2-D")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("masked-bilinear coordinates must be finite")
    fallback_radius = int(fallback_radius)
    if fallback_radius < 0:
        raise ValueError("fallback_radius must be non-negative")

    ny, nx = source_valid.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    if (np.min(x0) < 0 or np.max(x1) >= nx
            or np.min(y0) < 0 or np.max(y1) >= ny):
        raise ValueError("masked-bilinear coordinates leave the source window")
    fx = x - x0
    fy = y - y0
    indices_x = np.stack((x0, x1, x0, x1))
    indices_y = np.stack((y0, y0, y1, y1))
    weights = np.stack((
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    ))
    corner_valid = source_valid[indices_y, indices_x]
    weights *= corner_valid
    raw_support = np.sum(weights, axis=0)
    direct = target_apply & (raw_support > 0.0)
    weights[:, direct] /= raw_support[direct]

    needs_fallback = target_apply & ~direct
    fallback_count = int(np.count_nonzero(needs_fallback))
    fallback_max_distance = 0.0
    fallback_distance_histogram: dict[str, int] = {}
    if fallback_count:
        flat_missing = np.flatnonzero(needs_fallback)
        missing_x = x.ravel()[flat_missing]
        missing_y = y.ravel()[flat_missing]
        center_x = np.floor(missing_x + 0.5).astype(np.int64)
        center_y = np.floor(missing_y + 0.5).astype(np.int64)
        best_distance2 = np.full(flat_missing.size, np.inf)
        donor_x = np.full(flat_missing.size, -1, dtype=np.int64)
        donor_y = np.full(flat_missing.size, -1, dtype=np.int64)
        for offset_y in range(-fallback_radius, fallback_radius + 1):
            candidate_y = center_y + offset_y
            y_inside = (candidate_y >= 0) & (candidate_y < ny)
            for offset_x in range(-fallback_radius, fallback_radius + 1):
                candidate_x = center_x + offset_x
                inside = y_inside & (candidate_x >= 0) & (candidate_x < nx)
                if not np.any(inside):
                    continue
                candidate_valid = np.zeros(flat_missing.size, dtype=bool)
                positions = np.flatnonzero(inside)
                candidate_valid[positions] = source_valid[
                    candidate_y[positions], candidate_x[positions]]
                distance2 = ((candidate_x - missing_x) ** 2
                             + (candidate_y - missing_y) ** 2)
                candidate_valid &= distance2 <= float(fallback_radius ** 2)
                improve = candidate_valid & (distance2 < best_distance2)
                donor_x[improve] = candidate_x[improve]
                donor_y[improve] = candidate_y[improve]
                best_distance2[improve] = distance2[improve]
        if np.any(donor_x < 0):
            unresolved_mask = donor_x < 0
            unresolved = int(np.count_nonzero(unresolved_mask))
            unresolved_targets = tuple(zip(*np.unravel_index(
                flat_missing[unresolved_mask], x.shape)))
            # The smallest integer radius whose donor disk reaches a
            # valid cell for EVERY failing point -- measured, so the
            # refusal's remediation can be validated instead of guessed.
            required_radius = None
            valid_cells_y, valid_cells_x = np.nonzero(source_valid)
            if valid_cells_y.size and unresolved <= _DONOR_ANALYSIS_MAX_POINTS:
                worst_distance2 = max(
                    float(np.min((valid_cells_x - point_x) ** 2
                                 + (valid_cells_y - point_y) ** 2))
                    for point_x, point_y in zip(
                        missing_x[unresolved_mask],
                        missing_y[unresolved_mask]))
                required_radius = int(np.ceil(np.sqrt(worst_distance2)))
            raise SurfaceDonorSearchError(
                f"no valid surface-matched HRRR donor within "
                f"{fallback_radius} cells for {unresolved} target point(s)",
                fallback_radius_cells=fallback_radius,
                required_radius_cells=required_radius,
                unresolved_targets=unresolved_targets)
        flat_x = indices_x.reshape(4, -1)
        flat_y = indices_y.reshape(4, -1)
        flat_weights = weights.reshape(4, -1)
        flat_x[:, flat_missing] = donor_x[None, :]
        flat_y[:, flat_missing] = donor_y[None, :]
        flat_weights[:, flat_missing] = 0.0
        flat_weights[0, flat_missing] = 1.0
        fallback_distances = np.sqrt(best_distance2)
        fallback_max_distance = float(np.max(fallback_distances))
        distance_bins = np.ceil(fallback_distances).astype(np.int64)
        fallback_distance_histogram = {
            str(int(cell_radius)): int(np.count_nonzero(
                distance_bins == cell_radius))
            for cell_radius in np.unique(distance_bins)
        }

    # Non-applicable targets are overwritten by the complementary stencil or
    # an explicit physical fill.  Give them a harmless unit-sum donor so no
    # NaN can be created transiently on the GPU.
    unused = ~target_apply
    weights[:, unused] = 0.0
    weights[0, unused] = 1.0
    sums = np.sum(weights, axis=0)
    selected_source_is_valid = source_valid[indices_y, indices_x]
    cross_surface = (
        (weights > 0.0) & target_apply[None, :, :]
        & ~selected_source_is_valid)
    cross_surface_count = int(np.count_nonzero(cross_surface))
    if cross_surface_count:
        raise AssertionError(
            "masked-bilinear stencil selected an incompatible surface donor")
    if (not np.isfinite(weights).all() or np.any(weights < 0.0)
            or not np.allclose(sums, 1.0, rtol=0.0, atol=2.0e-15)):
        raise AssertionError("masked-bilinear stencil is not finite and convex")
    if sum(fallback_distance_histogram.values()) != fallback_count:
        raise AssertionError("fallback distance histogram is incomplete")
    report = {
        "operator": "masked_convex_bilinear_with_nearest_valid_fallback",
        "source_valid_count": int(np.count_nonzero(source_valid)),
        "target_apply_count": int(np.count_nonzero(target_apply)),
        "direct_target_count": int(np.count_nonzero(direct)),
        "renormalized_target_count": int(np.count_nonzero(
            direct & (raw_support < 1.0 - 1.0e-12))),
        "fallback_target_count": fallback_count,
        "fallback_radius_cells": fallback_radius,
        "fallback_max_distance_cells": fallback_max_distance,
        "fallback_distance_ceiling_histogram_cells": (
            fallback_distance_histogram),
        "unresolved_target_count": 0,
        "cross_surface_donor_count": cross_surface_count,
        "donor_surface_class": "land",
        "minimum_nonzero_raw_support": (
            float(np.min(raw_support[direct])) if np.any(direct) else None),
        "weight_sum_minimum": float(np.min(sums)),
        "weight_sum_maximum": float(np.max(sums)),
        "negative_weight_count": int(np.count_nonzero(weights < 0.0)),
    }
    return (indices_y.astype(np.int32), indices_x.astype(np.int32),
            weights.astype(np.float32), report)


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

    This is admission, not repair: a snapshot's field mapping is a
    read-only proxy, so the clamped copies decide the verdict here and
    the arrays are clamped again where they are consumed (the soil seam,
    and the mapped-output check below).  Clamping at the point of use is
    the honest place for it -- nothing downstream inherits a value this
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
    if (not np.isfinite(soilt).all()
            or np.any((soilt < 170.0) | (soilt > 400.0))):
        raise ValueError("source HRRR SOILT is non-finite or outside 170..400 K")
    if (not np.isfinite(spfh).all()
            or np.any((spfh < 0.0) | (spfh > 0.1))):
        raise ValueError("source HRRR SPFH is non-finite or outside 0..0.1")
    if (not np.isfinite(q2).all()
            or np.any((q2 < 0.0) | (q2 > 0.1))):
        raise ValueError("source HRRR Q2 is non-finite or outside 0..0.1")


#: What a soil-mapping refusal or report calls its target when the caller
#: names nothing better.  Every caller in the product names it: a strip
#: refusal that says only "soil mapping requires ... land cells" tells a
#: coastal-domain owner nothing about WHICH of four independently mapped
#: boundary strips, on which domain, was being talked about.
DEFAULT_SOIL_TARGET_NAME = "the HRRR target grid"


def _validated_donor_remediation(error: SurfaceDonorSearchError,
                                 snapshot: HrrrNativeSnapshot,
                                 target_name: str,
                                 target_shape: tuple[int, int]) -> str:
    """Advice that survives the guard it names, computed before printing.

    Field 2026-08: a fitter-maximum 3 km root sat with its radius-8
    source window exactly on HRRR's native top edge, two land cells had
    no donor within 8 cells, and the refusal recommended raising
    ``surface_fallback_radius_cells`` -- a setting the coverage guard
    (``required_hrrr_source_window``) refuses at ANY sufficient value
    there, because a larger radius enlarges the required window past
    the native edge.  Advice is therefore checked against the same
    limits first:

    * the radius that reaches a donor for every unfilled cell was
      measured by the stencil builder;
    * the window a raise needs is contained in this snapshot's window
      grown by the radius increase on every side (the fallback bound
      moves cell-for-cell with the radius, the parabolic bound not at
      all), so when THAT stays inside the native grid the guard
      provably accepts the raise;
    * otherwise the raise is named impossible and the computed trim is
      printed instead: removing exactly the rows or columns that carry
      the unfillable cells changes no remaining cell's donor disk, so
      the trimmed domain maps for precisely the reason this one could
      not, and its smaller window passes the guard by construction.
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
    grown_i = (snapshot.i_start - growth,
               snapshot.i_start + snapshot.nx - 1 + growth)
    grown_j = (snapshot.j_start - growth,
               snapshot.j_start + snapshot.ny - 1 + growth)
    overflows = [
        (side, gap) for side, gap in (
            ("west", -grown_i[0]),
            ("east", grown_i[1] - (HRRR_SOURCE_NX - 1)),
            ("south", -grown_j[0]),
            ("north", grown_j[1] - (HRRR_SOURCE_NY - 1)),
        ) if gap > 0]
    if required <= SURFACE_FALLBACK_RADIUS_MAX and not overflows:
        return (f"Raising surface_fallback_radius_cells to {required} "
                "reaches a surface-matched donor for every unfilled "
                "cell, and the enlarged source window (within "
                f"i={grown_i[0]}..{grown_i[1]}, "
                f"j={grown_j[0]}..{grown_j[1]}) stays inside the native "
                f"HRRR grid i=0..{HRRR_SOURCE_NX - 1}, "
                f"j=0..{HRRR_SOURCE_NY - 1}, so the coverage guard "
                f"accepts it.  Alternatively, {move}.")

    if overflows:
        named = ", ".join(f"{gap} cell(s) at its {side} edge"
                          for side, gap in overflows)
        reason = (f"the enlarged source window would leave the native "
                  f"HRRR grid i=0..{HRRR_SOURCE_NX - 1}, "
                  f"j=0..{HRRR_SOURCE_NY - 1} by {named}, so the "
                  "coverage guard refuses that setting")
    else:
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
        trim, side = min(
            (target_rows - min(rows), "north (j-max)"),
            (max(rows) + 1, "south (j-min)"),
            (target_cols - min(cols), "east (i-max)"),
            (max(cols) + 1, "west (i-min)"),
        )
        advice += (f"  What works: trim {trim} cell(s) from its {side} "
                   f"side, which removes every unfillable land cell of "
                   f"{target_name}, or {move}.")
    else:
        advice += f"  Instead, {move}."
    return advice


def _record_soil_field_stats(report, name, source, candidate,
                             source_land, target_land, limits, *,
                             target_name=DEFAULT_SOIL_TARGET_NAME):
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
    if (not np.isfinite(candidate).all()
            or np.any((candidate < lower) | (candidate > upper))):
        raise ValueError(
            f"mapped HRRR {name} for {target_name} is non-finite or outside "
            f"{lower}..{upper}")
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
    }


def interpolate_hrrr_to_lambert(
        snapshot: HrrrNativeSnapshot,
        grid: LambertGrid, *, target_landmask,
        soil_mapping_report: MutableMapping[str, object] | None = None,
        surface_fallback_radius: int = 8,
        backend="cuda", workers: int | None = None,
        cpu_bridge: Path | str | None = None,
        target_name: str = DEFAULT_SOIL_TARGET_NAME) -> HorizontalSnapshot:
    """Interpolate a verified HRRR window to one WRF Lambert C grid.

    Atmospheric continuous fields use WPS's overlapping-parabolic operator.
    Soil uses non-negative bilinear weights restricted to source land, with a
    bounded nearest-valid fallback.
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
    """
    if not isinstance(snapshot, HrrrNativeSnapshot):
        raise TypeError("snapshot must be an HrrrNativeSnapshot")
    if not isinstance(grid, LambertGrid):
        raise TypeError("grid must be a LambertGrid")
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
    if plan_type is _ProjectedGpuPlan:
        mass_plan = plan_type(snapshot, mass_lat, mass_lon)
        u_plan = plan_type(snapshot, u_lat, u_lon)
        v_plan = plan_type(snapshot, v_lat, v_lon)
    else:
        mass_plan = plan_type(snapshot, mass_lat, mass_lon, engine)
        u_plan = plan_type(snapshot, u_lat, u_lon, engine)
        v_plan = plan_type(snapshot, v_lat, v_lon, engine)
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
    try:
        land_stencil = mass_plan.masked_bilinear_stencil(
            source_land, target_land,
            fallback_radius=surface_fallback_radius)
    except SurfaceDonorSearchError as error:
        # The one soil case that genuinely cannot be mapped: this target
        # has land cells and the HRRR window has no land within reach of
        # them.  The stencil builder is generic and measures the facts;
        # only here is it known WHICH grid was being mapped and where
        # its window sits on the native HRRR grid, which is the whole
        # difference between an actionable refusal and one whose advice
        # the coverage guard refuses -- the field met exactly that: a
        # recommended radius raise that could never pass
        # required_hrrr_source_window on a fitter-maximum domain.
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
    # WPS METGRID.TBL routes hydrometeor mass through
    # ``four_pt+average_4pt`` rather than the overshooting parabolic operator.
    # Bilinear interpolation preserves both non-negativity and compact support.
    for name in ("QC", "QI", "QR", "QS", "QG"):
        out[name] = mass_plan.apply(source[name], method="bilinear")
    # Do not derive/map RH here.  With FLAG_SH, real.exe diagnoses rh_gc from
    # the already horizontally mapped SPECHUMD, TT, and PRES.  initialize_real
    # mirrors that order; retaining a separately mapped 50-level RH field is
    # both scientifically different and avoidable target-grid residency.
    # initialize_real uses WPS's GHT spelling; retain HGT as the native-GRIB
    # diagnostic spelling so oracle reports stay explicit.
    out["GHT"] = out["HGT"]
    target_land_backend = engine.bool_array(target_land)
    for name, water_fill in (("SOILT", out["SKINTEMP"]), ("SOILW", 1.0)):
        mapped_land = land_stencil.apply(source[name])
        selector = target_land_backend[None, :, :]
        if name == "SOILT":
            fill = xp.broadcast_to(water_fill[None, :, :], mapped_land.shape)
        else:
            fill = xp.full(mapped_land.shape, water_fill, dtype=xp.float32)
        out[name] = xp.where(selector, mapped_land, fill)
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
            "source-land-masked convex bilinear; bounded nearest-valid "
            "fallback; "
            "target-water SOILT=SKINTEMP and SOILW=1"),
        "integral_conservation_claimed": False,
        "preprocess_backend": engine.receipt(),
        # WHICH implementation of the projected horizontal operator ran.
        # The backend receipt above says "cpu", and "cpu" now covers two
        # implementations that agree bit for bit but not within an order
        # of magnitude in wall time -- so a receipt that only said "cpu"
        # could not tell a slow run from a fast one after the fact.
        "projected_horizontal_operator": mass_plan.operator,
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
        "land_stencil": dict(land_stencil.report),
        "fields": {},
    }
    _record_soil_field_stats(
        local_report["fields"], "SOILT", source["SOILT"], out["SOILT"],
        source_land, target_land, (170.0, 400.0),
        target_name=target_name)
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
]
