"""The anchor's on-disk encoding, with no gpuwm in it.

``gpuwm.cycle.anchor`` owns the anchor's *policy*: which fields are
required, what a legal parent kind is, the three-phase commit, the
refusals.  This module owns its *encoding*: how an array mapping becomes
bytes, how those bytes hash, and how they come back.

The split exists so the forecast worker -- which may not import gpuwm at
all (see :mod:`mpas_cycle_bridge.spine_guard`) -- can still read an
anchor and write a segment in exactly the same format, from exactly the
same code, rather than from a parallel reimplementation that drifts.  A
reader tested against its own writer proves nothing; both sides here run
these functions.

``gpuwm.cycle.anchor`` imports from this module.  Nothing here imports
from it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MANIFEST_NAME = "anchor.json"
COMMIT_MARKER_NAME = "COMMIT"

PROGNOSTIC_STEM = "parent_prognostic"
DERIVED_STEM = "parent_derived"
SEAM_STEM = "parent_seam"
#: The vertical-grid metric (``zz``) the equation of state needs.  A
#: boundary without it cannot have its exner recomputed by anything
#: outside the port.
METRIC_STEM = "parent_metric"
INCREMENT_STEM = "analysis_increment"

try:  # pragma: no cover - exercised by whichever branch the box has
    import netCDF4 as _netCDF4
except Exception:  # pragma: no cover
    _netCDF4 = None

ARRAY_FORMAT = "netcdf4" if _netCDF4 is not None else "npz"

_LEGAL_NETCDF_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_legal_netcdf_name(name: str) -> bool:
    return bool(_LEGAL_NETCDF_NAME.match(name))


SUFFIX = {"netcdf4": ".nc", "npz": ".npz"}


# --------------------------------------------------------------------------
# hashing


def contiguous(value: Any) -> np.ndarray:
    # ``np.ascontiguousarray`` promotes a 0-d array to shape (1,), which
    # would silently give ``time_seconds`` a different shape on the way
    # out than it had on the way in -- and a different sha.
    raw = np.asarray(value)
    return np.ascontiguousarray(raw).reshape(raw.shape)


def prognostic_sha256(mapping: Mapping[str, np.ndarray]) -> str:
    """The canonical state hash, reproducible by an independent reader.

    The recipe, so a Rust or Fortran consumer can compute the same
    number: walk the field names in ``sorted()`` order and, for each,
    feed the hash ``name.encode()``, ``str(dtype).encode()``,
    ``repr(shape).encode()``, then the C-contiguous raw bytes of the
    array.  Nothing else enters -- no file bytes, no compression, no
    ordering of the caller's dict.

    It is byte-exact and therefore FP-representation-sensitive BY
    DESIGN.  The same values stored as float32 hash differently from
    float64, and a one-ULP change is a different state.  That is what
    makes it usable as the ingestion gate's evidence: the question the
    gate asks is "is this literally the same array", not "is it close".
    """
    digest = hashlib.sha256()
    for name in sorted(mapping):
        array = contiguous(mapping[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# array containers


def write_arrays(path: str | os.PathLike[str],
                 mapping: Mapping[str, np.ndarray]) -> None:
    arrays = {name: contiguous(value) for name, value in mapping.items()}
    if ARRAY_FORMAT == "npz":
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)
        return
    dataset = _netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        for index, (name, array) in enumerate(arrays.items()):
            # The physics backend's restart payload names its arrays
            # freely -- dotted, slashed, whatever the seam calls them --
            # and netCDF refuses most of that as an identifier
            # ("NetCDF: Name contains illegal characters", which is how
            # the first real seed run died on node-1).  So a name that
            # is not a legal identifier is stored under a positional
            # one, with the true name in an attribute.  Legal names are
            # left exactly as they are, so anchors written before this
            # read back unchanged.
            safe = name if _is_legal_netcdf_name(name) else f"gpuwm_v{index}"
            dims = []
            for axis, size in enumerate(array.shape):
                dim = f"{safe}_d{axis}"
                dataset.createDimension(dim, int(size))
                dims.append(dim)
            variable = dataset.createVariable(safe, array.dtype, tuple(dims))
            if safe != name:
                variable.setncattr("gpuwm_name", name)
            # netCDF4 reports a scalar variable's shape as ``(1,)`` on the
            # way back out, so the writer records the shape it meant.  A
            # state that round-trips with a different shape is not a
            # round trip, and ``time_seconds`` is exactly such a scalar.
            variable.setncattr("gpuwm_shape", json.dumps(list(array.shape)))
            variable[...] = array
    finally:
        dataset.close()


def read_arrays(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    if ARRAY_FORMAT == "npz":
        with np.load(path) as handle:
            return {name: np.asarray(handle[name]) for name in handle.files}
    dataset = _netCDF4.Dataset(path, "r")
    try:
        out: dict[str, np.ndarray] = {}
        for name, variable in dataset.variables.items():
            value = variable[...]
            value = np.asarray(getattr(value, "data", value),
                               dtype=variable.dtype)
            shape = variable.getncattr("gpuwm_shape") \
                if "gpuwm_shape" in variable.ncattrs() else None
            if shape is not None:
                value = value.reshape(tuple(json.loads(shape)))
            true_name = (variable.getncattr("gpuwm_name")
                         if "gpuwm_name" in variable.ncattrs() else name)
            out[str(true_name)] = contiguous(value)
        return out
    finally:
        dataset.close()


def fsync_file(path: str | os.PathLike[str]) -> None:
    # Windows refuses ``fsync`` on a read-only handle (Errno 9), so the
    # flush is taken on a read-write one.  Phase one is not durable
    # without this and the whole three-phase commit rests on it.
    handle = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


# --------------------------------------------------------------------------
# reading a committed anchor without gpuwm


class CodecRefusal(RuntimeError):
    """A refusal that is required to name what it observed."""

    def __init__(self, message: str, **observed: object) -> None:
        if not observed:
            raise ValueError(
                "a codec refusal must name what was observed; "
                f"got bare message {message!r}")
        rendered = ", ".join(f"{key}={observed[key]!r}"
                             for key in sorted(observed))
        super().__init__(f"{message} | observed: {rendered}")
        self.observed = dict(observed)


def read_manifest(anchor_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a COMMITTED anchor's manifest, checking the commit marker.

    Refuses an uncommitted directory: a worker that rehydrates from a
    half-written boundary is the sort of thing that produces a plausible
    forecast from a torn state.
    """
    root = Path(anchor_dir)
    marker = root / COMMIT_MARKER_NAME
    if not marker.is_file():
        raise CodecRefusal(
            "anchor carries no COMMIT marker; it is not a published "
            "boundary and must not be rehydrated from",
            anchor=str(root), looked_for=str(marker))
    committed = json.loads(marker.read_text(encoding="utf-8"))
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CodecRefusal("anchor has a COMMIT marker but no manifest",
                           anchor=str(root), looked_for=str(manifest_path))
    recorded = {entry["path"]: entry["sha256"]
                for entry in committed.get("files", ())}
    for relative, expected in sorted(recorded.items()):
        member = root / relative
        if not member.is_file():
            raise CodecRefusal("anchor member named in COMMIT is missing",
                               anchor=str(root), member=relative)
        actual = file_sha256(member)
        if actual != expected:
            raise CodecRefusal(
                "anchor member does not hash to what COMMIT recorded; the "
                "boundary on disk is not the boundary that was published",
                anchor=str(root), member=relative,
                expected=expected, actual=actual)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def read_member(anchor_dir: str | os.PathLike[str],
                relative: str | None) -> dict[str, np.ndarray] | None:
    """Load one array member of an anchor by its manifest-relative path."""
    if not relative:
        return None
    return read_arrays(Path(anchor_dir) / relative)
