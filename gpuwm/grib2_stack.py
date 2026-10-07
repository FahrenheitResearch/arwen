"""Generic Rust GRIB2 stacks. Field bytes are mapped without Python transforms."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence
import ctypes
import hashlib
import json
import tempfile
import numpy as np
from gpuwm.ingest.cpu_backend import CpuPreprocessBackend


def stack_cache_identity(input_file: str | Path, select: Sequence[Mapping]) -> dict:
    """Bind decoded bytes to raw content, exact selectors and selected decoder.

    Use the same resolver as CpuPreprocessBackend, including its per-call
    bridge selection. Paths and environment contents are not decoder identity.
    """
    from gpuwm.bridge_assets import sha256_file
    from gpuwm.ingest.cpu_backend import resolve_cpu_bridge
    decoder = resolve_cpu_bridge()
    # Own nested metadata too: callers can supply a mutable levels list.
    specification = json.loads(json.dumps(
        {"schema": "gpuwm-grib2-stack-v1", "select": [dict(row) for row in select]},
        sort_keys=True, allow_nan=False))
    return {
        "schema": "gpuwm-grib2-stack-cache-v2",
        "input_sha256": sha256_file(Path(input_file)),
        "decoder_sha256": sha256_file(decoder),
        "decoder_bytes": decoder.stat().st_size,
        "specification": specification,
    }


def stack_cache_digest(identity: Mapping) -> str:
    """Canonical metadata identity shared by memory and on-disk admission."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StackedField:
    key: str
    valid_time: datetime
    levels: tuple[float, ...]
    latitude: np.ndarray
    longitude: np.ndarray
    coordinate_values: tuple[float, ...]
    file: Path
    scan_mode: int
    pdt: int
    #: A projected (template 3.30 Lambert conformal) grid's Section 3
    #: parameters, as the Rust stack recorded them; ``latitude`` and
    #: ``longitude`` are then the zero-based row and column index axes of
    #: the stored scan order, not geographic axes.  None on a regular
    #: latitude/longitude grid.
    projection: dict | None = None

    @property
    def array(self) -> np.memmap:
        """Open a read-only, lazy float32 mapping in the file's stored scan order."""
        return np.memmap(self.file, dtype="<f4", mode="r",
                         shape=(len(self.levels), self.latitude.size, self.longitude.size))


def _axis(first: float, last: float, increment: float, count: int, negative: bool) -> np.ndarray:
    """Allocate the decoded coordinate buffer and let Rust fill its axis."""
    if count < 1:
        raise ValueError("invalid encoded regular grid axis")
    axis = np.empty(count, dtype=np.float64)
    backend = CpuPreprocessBackend()
    function = backend._chem_symbol("gpuwm_grib2_axis_f64", [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_size_t,
        ctypes.c_uint8, ctypes.c_void_p])
    if function(first, last, increment, count, int(negative), axis.ctypes.data):
        raise ValueError("invalid encoded regular grid axis")
    axis.flags.writeable = False
    return axis


def read_stack(output_dir: str | Path) -> tuple[StackedField, ...]:
    root = Path(output_dir).resolve()
    metadata = json.loads((root / "stack.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "gpuwm-grib2-stack-v1":
        raise ValueError("unsupported stack.json schema")
    result = []
    for row in metadata["outputs"]:
        grid = row["grid"]
        if grid["template"] not in (0, 30):
            raise ValueError(f"grid template {grid['template']} is unsupported")
        path = (root / row["file"]).resolve()
        if path.parent != root:
            raise ValueError("stack file escapes output directory")
        levels = tuple(row["levels"])
        expected = len(levels) * grid["ny"] * grid["nx"] * 4
        if path.stat().st_size != expected:
            raise ValueError(f"stack byte count disagrees with shape: {path.name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"stack sha256 mismatch: {path.name}")
        projection = None
        if grid["template"] == 30:
            # Lambert conformal: index axes in stored order; the projection
            # (first point, true latitudes, orientation, spacing) is what a
            # reader projects targets with.  Only +i/+j scanning (64) is the
            # layout those indices describe.
            if grid["scan_mode"] != 64:
                raise ValueError(f"Lambert stack scan mode {grid['scan_mode']} is not "
                                 "64 (+i west to east, +j south to north); its rows "
                                 "would be read in the wrong order")
            latitude = np.arange(grid["ny"], dtype=np.float64)
            longitude = np.arange(grid["nx"], dtype=np.float64)
            latitude.flags.writeable = False
            longitude.flags.writeable = False
            projection = {key: grid[key] for key in
                          ("template", "nx", "ny", "lat1", "lon1", "dlat", "dlon",
                           "latin1", "latin2", "lov", "scan_mode")}
        else:
            latitude = _axis(grid["lat1"], grid["lat2"], grid["dlat"], grid["ny"],
                             grid["lat2"] < grid["lat1"] if grid["ny"] > 1 else not bool(grid["scan_mode"] & 64))
            longitude = _axis(grid["lon1"], grid["lon2"], grid["dlon"], grid["nx"], bool(grid["scan_mode"] & 128))
        result.append(StackedField(row["key"], datetime.fromisoformat(row["valid_time"]),
                     levels, latitude, longitude,
                     tuple(metadata["coordinate_values"][row["key"]]), path,
                     grid["scan_mode"], row["pdt"], projection))
    return tuple(result)


def stack_grib2(input_file: str | Path, select: Sequence[Mapping],
                output_dir: str | Path) -> tuple[StackedField, ...]:
    """Select raw identifiers and levels, then return verified lazy field mappings.

    Null constituent/aerosol selectors impose no constraint on that identifier.
    Null levels select every level present. The Rust binary refuses inconsistent
    keys, grids, coefficients, times and level inventories before writing output.
    """
    specification = {"schema": "gpuwm-grib2-stack-v1", "select": [dict(row) for row in select]}
    # Keep the SPEC beside the output for reproducible decode evidence.
    output = Path(output_dir).resolve()
    scratch = Path(tempfile.mkdtemp(prefix=".grib2-stack-spec-", dir=output.parent))
    spec_path = scratch / "spec.json"
    spec_path.write_text(json.dumps(specification, allow_nan=False), encoding="utf-8")
    CpuPreprocessBackend().grib2_stack(Path(input_file).resolve(), spec_path, output)
    return read_stack(output)
