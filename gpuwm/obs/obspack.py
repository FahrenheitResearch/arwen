"""Reader for the ``GPWMOBS1`` packs the observation front doors write.

The same container discipline as :mod:`gpuwm.obs.sweeps` reads for radar
sweeps, with an observation magic so one is never mistaken for the other:

* 8-byte magic ``GPWMOBS1``, ``u32`` version, ``u32`` metadata length,
  ``u64`` payload length, zero-padded to 64 bytes;
* a JSON metadata block carrying a ``schema``, the array index, and the
  payload's SHA-256;
* one contiguous little-endian payload of ``<f8`` and ``<u1`` arrays.

Nothing here interprets meteorology. It reads bytes the Rust side wrote,
re-proves what the writer promised about them, and hands back arrays.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The magic every observation pack opens with.
PACK_MAGIC = b"GPWMOBS1"
PACK_VERSION = 1
PACK_HEADER_BYTES = 64

#: The schema a gridded observation pack declares.
GRID_SCHEMA = "gpuwm-obs.obs-grid.v1"
#: The schema a grid-geometry pack declares.
GEO_SCHEMA = "gpuwm-obs.obs-geo.v1"

#: The dtypes the writer emits. A pack naming any other is refused rather
#: than handed to ``numpy.dtype``, which would happily accept ``>f8`` and
#: read every value byte-swapped.
_KNOWN_DTYPES = {"<f8": 8, "<u1": 1}


@dataclass(frozen=True)
class ObsPack:
    """One decoded pack: its metadata and its arrays."""

    path: Path
    meta: dict
    arrays: dict[str, np.ndarray]

    @property
    def schema(self) -> str:
        return str(self.meta.get("schema", ""))

    def array(self, key: str) -> np.ndarray:
        try:
            return self.arrays[key]
        except KeyError:
            raise KeyError(
                f"{self.path} carries no array {key!r}; it has "
                f"{sorted(self.arrays)}") from None


def read_pack(path: str | Path) -> ObsPack:
    """Read and re-prove one observation pack.

    Every check the writer's own ``verify`` subcommand makes is made here
    too, because a pack is read far more often than it is verified and a
    reader that trusts its input is a reader that will one day return a
    silently truncated field.
    """

    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < PACK_HEADER_BYTES:
        raise ValueError(
            f"{path}: {len(raw)} bytes, the pack header alone is "
            f"{PACK_HEADER_BYTES}")
    if raw[:8] != PACK_MAGIC:
        raise ValueError(
            f"{path}: magic {raw[:8]!r}, expected {PACK_MAGIC!r}")
    version, meta_len = struct.unpack("<II", raw[8:16])
    (payload_len,) = struct.unpack("<Q", raw[16:24])
    if version != PACK_VERSION:
        raise ValueError(
            f"{path}: pack version {version}, this reader reads "
            f"{PACK_VERSION}")
    meta_end = PACK_HEADER_BYTES + meta_len
    payload_end = meta_end + payload_len
    if payload_end != len(raw):
        raise ValueError(
            f"{path}: header declares {payload_end} bytes, file has "
            f"{len(raw)}")
    meta = json.loads(raw[PACK_HEADER_BYTES:meta_end])
    payload = raw[meta_end:payload_end]

    declared = str(meta.get("content_sha256", ""))
    digest = hashlib.sha256(payload).hexdigest()
    if digest != declared:
        raise ValueError(
            f"{path}: payload digest mismatch; metadata says {declared}, "
            f"bytes hash to {digest}")

    arrays: dict[str, np.ndarray] = {}
    for key, entry in dict(meta.get("arrays", {})).items():
        dtype = str(entry["dtype"])
        if dtype not in _KNOWN_DTYPES:
            raise ValueError(
                f"{path}: array {key} declares dtype {dtype!r}, which this "
                f"reader does not read")
        shape = tuple(int(n) for n in entry["shape"])
        offset = int(entry["offset"])
        nbytes = int(entry["bytes"])
        count = int(np.prod(shape)) if shape else 0
        if count * _KNOWN_DTYPES[dtype] != nbytes:
            raise ValueError(
                f"{path}: array {key} declares shape {shape} of {dtype} but "
                f"{nbytes} bytes")
        if offset + nbytes > len(payload):
            raise ValueError(
                f"{path}: array {key} spans bytes {offset}..{offset + nbytes} "
                f"of a {len(payload)}-byte payload")
        flat = np.frombuffer(payload, dtype=np.dtype(dtype), count=count,
                             offset=offset)
        arrays[key] = flat.reshape(shape)
    return ObsPack(path=path, meta=meta, arrays=arrays)


def read_grid_pack(path: str | Path) -> ObsPack:
    """Read a pack and require it to be a gridded observation."""

    pack = read_pack(path)
    if pack.schema != GRID_SCHEMA:
        raise ValueError(
            f"{path} declares schema {pack.schema!r}, expected "
            f"{GRID_SCHEMA!r}")
    return pack


def read_geo_pack(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a geometry pack, returning ``(latitude, longitude)``.

    Both come back ``float64`` and two-dimensional, which is what the
    scorer's contract requires of every field's coordinates.
    """

    pack = read_pack(path)
    if pack.schema != GEO_SCHEMA:
        raise ValueError(
            f"{path} declares schema {pack.schema!r}, expected "
            f"{GEO_SCHEMA!r}")
    latitude = np.asarray(pack.array("latitude"), dtype=np.float64)
    longitude = np.asarray(pack.array("longitude"), dtype=np.float64)
    if latitude.shape != longitude.shape or latitude.ndim != 2:
        raise ValueError(
            f"{path}: latitude {latitude.shape} and longitude "
            f"{longitude.shape} must be one 2-D shape")
    return latitude, longitude


def sha256_of(path: str | Path) -> str:
    """The SHA-256 of a file on disk, streamed."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["GEO_SCHEMA", "GRID_SCHEMA", "PACK_MAGIC", "PACK_VERSION",
           "ObsPack", "read_geo_pack", "read_grid_pack", "read_pack",
           "sha256_of"]
