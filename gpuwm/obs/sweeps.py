"""Reader for the ``gpuwm-obs.radar-sweeps.v1`` pack that ``rw_nexrad`` writes.

The pack is a 64-byte little-endian header, a JSON metadata block, and one
contiguous payload of ``<f4`` arrays — the ``.rwg`` layout from ``rw-store``.
Reading it needs :mod:`json` and :func:`numpy.frombuffer` and nothing else,
which is the whole point: the Rust side owns bytes-to-numbers, the Python
side owns numbers-to-observations, and the seam between them is a file
format either can prove.

Every self-describing field the writer promised is checked before a caller
sees an array: magic, version, declared lengths, schema string, payload
digest, and that each array's declared shape accounts for exactly its
declared bytes inside the payload it indexes.  A pack that fails any of
those raises :class:`RadarSweepPackError`; nothing is returned partially.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

#: Contract the pack metadata must declare.
SWEEPS_SCHEMA = "gpuwm-obs.radar-sweeps.v1"

_MAGIC = b"GPWMRDR1"
_VERSION = 1
_HEADER_BYTES = 64
_DTYPE = "<f4"


class RadarSweepPackError(ValueError):
    """A sweep pack that cannot be trusted, for a stated reason."""


@dataclass(frozen=True)
class RadarSite:
    """Where the radar is, and how we know."""

    id: str
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    source: str


@dataclass(frozen=True)
class Moment:
    """One moment of one sweep: a ``(radial, gate)`` plane and its range axis."""

    product: str
    unit: str
    gate_count: int
    first_gate_range_m: float
    gate_size_m: float
    data: np.ndarray

    def slant_range_m(self) -> np.ndarray:
        """Range to each gate centre, shape ``(gate_count,)``."""

        return (self.first_gate_range_m
                + self.gate_size_m * np.arange(self.gate_count,
                                               dtype=np.float64))


@dataclass(frozen=True)
class Sweep:
    """One elevation cut."""

    sweep_index: int
    elevation_number: int
    elevation_angle_deg: float
    nyquist_velocity_ms: float | None
    start_status: int
    end_status: int
    cut_sector: int
    complete: bool
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    moments: dict[str, Moment]
    #: True when the cut's radials did not all report that Nyquist value.
    #: The scalar above is their minimum, so it never licenses a gate its
    #: own radial would have rejected, but a cut that disagreed with itself
    #: is worth carrying into provenance rather than smoothing over.  It
    #: defaults to False so a pack written before the field existed, and a
    #: sweep built by hand, both read as "nothing known to disagree".
    nyquist_radials_disagree: bool = False

    @property
    def radial_count(self) -> int:
        return int(self.azimuth_deg.size)


@dataclass(frozen=True)
class RadarVolume:
    """One decoded Level-II volume, ready to superob."""

    site: RadarSite
    valid_time: str
    station_id: str
    volume_file: str
    volume_sha256: str
    volume_bytes: int
    pack_path: Path
    pack_sha256: str
    params: dict
    framing: dict
    sweeps: tuple[Sweep, ...]

    def provenance(self) -> dict:
        """The record that travels into the gridded product's provenance."""

        return {
            "volume_file": self.volume_file,
            "volume_sha256": self.volume_sha256,
            "volume_bytes": self.volume_bytes,
            "pack_file": self.pack_path.name,
            "pack_sha256": self.pack_sha256,
            "pack_schema": SWEEPS_SCHEMA,
            "station_id": self.station_id,
            "valid_time": self.valid_time,
            "decode_params": dict(self.params),
            "archive2_framing": dict(self.framing),
        }


def read_sweep_pack(path: str | Path) -> RadarVolume:
    """Read and fully validate one sweep pack."""

    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RadarSweepPackError(f"cannot read {path}: {error}") from error
    return _decode(raw, path)


def _decode(raw: bytes, path: Path) -> RadarVolume:
    if len(raw) < _HEADER_BYTES:
        raise RadarSweepPackError(
            f"{path.name}: {len(raw)} bytes, the pack header alone is "
            f"{_HEADER_BYTES}")
    if raw[:8] != _MAGIC:
        raise RadarSweepPackError(
            f"{path.name}: magic {raw[:8]!r}, expected {_MAGIC!r}")
    version = int(np.frombuffer(raw[8:12], dtype="<u4")[0])
    if version != _VERSION:
        raise RadarSweepPackError(
            f"{path.name}: pack version {version}, this reader handles "
            f"{_VERSION}")
    meta_len = int(np.frombuffer(raw[12:16], dtype="<u4")[0])
    payload_len = int(np.frombuffer(raw[16:24], dtype="<u8")[0])
    meta_end = _HEADER_BYTES + meta_len
    payload_end = meta_end + payload_len
    if payload_end != len(raw):
        raise RadarSweepPackError(
            f"{path.name}: header declares {payload_end} bytes, file has "
            f"{len(raw)}")
    try:
        meta = json.loads(raw[_HEADER_BYTES:meta_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RadarSweepPackError(
            f"{path.name}: metadata block is not JSON: {error}") from error
    if meta.get("schema") != SWEEPS_SCHEMA:
        raise RadarSweepPackError(
            f"{path.name}: declares schema {meta.get('schema')!r}, expected "
            f"{SWEEPS_SCHEMA!r}")
    if meta.get("status") != "READY":
        raise RadarSweepPackError(
            f"{path.name}: status {meta.get('status')!r}, expected 'READY'")

    payload = raw[meta_end:payload_end]
    digest = hashlib.sha256(payload).hexdigest()
    if digest != meta.get("content_sha256"):
        raise RadarSweepPackError(
            f"{path.name}: payload hashes to {digest}, metadata says "
            f"{meta.get('content_sha256')}")

    arrays = meta.get("arrays") or {}
    for key, entry in arrays.items():
        if entry.get("dtype") != _DTYPE:
            raise RadarSweepPackError(
                f"{path.name}: array {key} has dtype {entry.get('dtype')!r}, "
                f"this reader handles {_DTYPE!r}")
        elements = int(np.prod(entry["shape"], dtype=np.int64)) if entry["shape"] else 0
        if elements * 4 != int(entry["bytes"]):
            raise RadarSweepPackError(
                f"{path.name}: array {key} declares shape {entry['shape']} "
                f"but {entry['bytes']} bytes")
        if int(entry["offset"]) + int(entry["bytes"]) > len(payload):
            raise RadarSweepPackError(
                f"{path.name}: array {key} spans past the end of a "
                f"{len(payload)}-byte payload")

    def view(key: str) -> np.ndarray:
        entry = arrays.get(key)
        if entry is None:
            raise RadarSweepPackError(
                f"{path.name}: metadata references missing array {key!r}")
        start = int(entry["offset"])
        stop = start + int(entry["bytes"])
        flat = np.frombuffer(payload[start:stop], dtype=_DTYPE)
        return flat.reshape(tuple(int(dim) for dim in entry["shape"]))

    sweeps = []
    for entry in meta.get("sweeps", ()):
        azimuth = view(entry["azimuth_array"])
        elevation = view(entry["elevation_array"])
        moments = {}
        for moment in entry.get("moments", ()):
            data = view(moment["array"])
            if data.shape[0] != azimuth.size:
                raise RadarSweepPackError(
                    f"{path.name}: sweep {entry['sweep_index']} moment "
                    f"{moment['product']} has {data.shape[0]} rows for "
                    f"{azimuth.size} radials")
            moments[moment["product"]] = Moment(
                product=str(moment["product"]),
                unit=str(moment["unit"]),
                gate_count=int(moment["gate_count"]),
                first_gate_range_m=float(moment["first_gate_range_m"]),
                gate_size_m=float(moment["gate_size_m"]),
                data=data)
        sweeps.append(Sweep(
            sweep_index=int(entry["sweep_index"]),
            elevation_number=int(entry["elevation_number"]),
            elevation_angle_deg=float(entry["elevation_angle_deg"]),
            nyquist_velocity_ms=(
                None if entry.get("nyquist_velocity_ms") is None
                else float(entry["nyquist_velocity_ms"])),
            nyquist_radials_disagree=bool(
                entry.get("nyquist_radials_disagree", False)),
            start_status=int(entry["start_status"]),
            end_status=int(entry["end_status"]),
            cut_sector=int(entry["cut_sector"]),
            complete=bool(entry["complete"]),
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            moments=moments))

    if not sweeps:
        raise RadarSweepPackError(f"{path.name}: pack carries no sweeps")

    site = meta["site"]
    volume = meta["volume"]
    return RadarVolume(
        site=RadarSite(id=str(site["id"]), name=str(site["name"]),
                       lat_deg=float(site["lat_deg"]),
                       lon_deg=float(site["lon_deg"]),
                       alt_m=float(site["alt_m"]),
                       source=str(site["source"])),
        valid_time=str(volume["valid_time"]),
        station_id=str(volume["station_id"]),
        volume_file=str(volume["file"]),
        volume_sha256=str(volume["sha256"]),
        volume_bytes=int(volume["bytes"]),
        pack_path=path,
        pack_sha256=hashlib.sha256(raw).hexdigest(),
        params=dict(meta.get("params") or {}),
        framing=dict(volume.get("framing") or {}),
        sweeps=tuple(sweeps))
