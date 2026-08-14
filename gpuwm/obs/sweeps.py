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

#: The same pack, plus a ``|u1`` censor plane beside every moment plane.
#: Written only when ``rw_nexrad decode --censor-flags`` was asked for.
#:
#: A distinct schema string rather than an optional key inside v1, because
#: the two files make different claims about the same bytes.  In a v1 pack a
#: NaN gate means "nothing further is known about this gate"; in a v2 pack
#: the writer is asserting it knows exactly which of three things happened
#: there.  A reader that quietly ignored the extra arrays would be assuming
#: the weaker claim while holding the stronger file, so the version is what
#: makes the extension safe rather than merely additive.
SWEEPS_SCHEMA_CENSOR = "gpuwm-obs.radar-sweeps.v2"

#: The same container again, written by ``rw_odim`` from a European
#: ODIM_H5 polar volume.
#:
#: A third string rather than ``v2``, for the reason ``v2`` is not ``v1``:
#: the censor vocabulary is genuinely wider.  ODIM distinguishes two states
#: NEXRAD has no word for -- :data:`Censor.NODATA`, the radar did not
#: measure this gate, and :data:`Censor.SENTINEL_AMBIGUOUS`, the file
#: declares its two sentinels as the same raw value so a gate holding it
#: may be either.  A ``v2`` reader handed either code would be reading a
#: plane it cannot fully account for, and the schema string is what stops
#: it -- which is why the code vocabulary below is checked per schema and
#: not once for every censored pack.
SWEEPS_SCHEMA_ODIM = "gpuwm-obs.radar-sweeps.v3"

#: Every schema this reader accepts.
SWEEPS_SCHEMAS = (SWEEPS_SCHEMA, SWEEPS_SCHEMA_CENSOR, SWEEPS_SCHEMA_ODIM)

#: Schemas whose moments carry a censor plane beside every moment plane.
CENSORED_SCHEMAS = (SWEEPS_SCHEMA_CENSOR, SWEEPS_SCHEMA_ODIM)

_MAGIC = b"GPWMRDR1"
_VERSION = 1
_HEADER_BYTES = 64
_DTYPE = "<f4"
_CENSOR_DTYPE = "|u1"


class Censor:
    """Why a gate in :attr:`Moment.data` is not a number.

    Mirrors ``wx_radar::level2::censor`` and the pack's ``|u1`` plane.  The
    two that matter are opposites and must never be confused:
    :data:`BELOW_THRESHOLD` is the radar reporting that it looked and found
    nothing, which is a clear-air *observation*; :data:`RANGE_FOLDED` is the
    radar reporting that it cannot say what it saw, which is evidence of
    nothing and may well be a storm.
    """

    #: ``data`` holds a decoded measurement.
    MEASURED = 0
    #: Raw 0 in the Message-31 word: below the significant-return threshold.
    BELOW_THRESHOLD = 1
    #: Raw 1: second-trip ambiguity.  Never usable as clear air.
    RANGE_FOLDED = 2
    #: A radial that carried no such moment; the pack's rectangle fill.
    NOT_COLLECTED = 3
    #: ODIM ``nodata``: the radar did not measure this gate.  Not clear air
    #: -- the radar is not reporting an absence of echo, it is reporting an
    #: absence of looking.  Only a :data:`SWEEPS_SCHEMA_ODIM` pack mints it.
    NODATA = 4
    #: ODIM declared ``nodata`` and ``undetect`` as the same raw value, so a
    #: gate holding it may be either.  Never usable as clear air: reading it
    #: that way would assimilate "no echo" into cells that may have one.
    #: Measured at 721,898 gates in one Finnish volume, so this is not a
    #: corner case.  Only a :data:`SWEEPS_SCHEMA_ODIM` pack mints it.
    SENTINEL_AMBIGUOUS = 5


#: Which censor codes each schema is allowed to have written.
#:
#: Per schema rather than one union, because that is the whole point of the
#: version string: a ``v2`` pack carrying code 4 is not a pack with an extra
#: code in it, it is a pack whose writer and schema disagree, and reading it
#: would mean trusting a claim the file itself contradicts.
_SCHEMA_CENSOR_CODES = {
    SWEEPS_SCHEMA_CENSOR: (
        Censor.MEASURED, Censor.BELOW_THRESHOLD,
        Censor.RANGE_FOLDED, Censor.NOT_COLLECTED),
    # RANGE_FOLDED is absent deliberately: ODIM has no second-trip state and
    # rw_odim never mints code 2, so a v3 pack holding one is a defect.
    SWEEPS_SCHEMA_ODIM: (
        Censor.MEASURED, Censor.BELOW_THRESHOLD, Censor.NOT_COLLECTED,
        Censor.NODATA, Censor.SENTINEL_AMBIGUOUS),
}

#: What granularity the source recorded the Nyquist interval at.
#:
#: NEXRAD's Message-31 RAD block carries one per radial; ODIM declares a
#: single ``/datasetN/how/NI`` for the whole cut.  The dealiaser refuses by
#: name when the per-radial array it wants is absent, and this is what lets
#: that refusal distinguish "this file never had per-radial originals" from
#: "they were dropped".
NYQUIST_PER_RADIAL = "radial"
NYQUIST_PER_SWEEP = "sweep"


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
    #: ``uint8`` :class:`Censor` codes, same shape as ``data``, or ``None``
    #: for a v1 pack that never carried them.  ``None`` and "all measured"
    #: are different statements and are kept different: the first says the
    #: reasons were never recorded, the second says they were recorded and
    #: every gate is a number.
    censor: np.ndarray | None = None

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
    #: :data:`NYQUIST_PER_RADIAL`, :data:`NYQUIST_PER_SWEEP`, or ``None``
    #: for a pack written before the key existed.  ``None`` means the pack
    #: did not say, which is different from either answer and is kept
    #: different.
    nyquist_granularity: str | None = None

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
    #: Which of :data:`SWEEPS_SCHEMAS` the pack declared.  Defaulted so a
    #: volume assembled by hand still reads as the baseline contract.
    pack_schema: str = SWEEPS_SCHEMA

    def provenance(self) -> dict:
        """The record that travels into the gridded product's provenance."""

        return {
            "volume_file": self.volume_file,
            "volume_sha256": self.volume_sha256,
            "volume_bytes": self.volume_bytes,
            "pack_file": self.pack_path.name,
            "pack_sha256": self.pack_sha256,
            "pack_schema": self.pack_schema,
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
    schema = meta.get("schema")
    if schema not in SWEEPS_SCHEMAS:
        raise RadarSweepPackError(
            f"{path.name}: declares schema {schema!r}, expected one of "
            f"{SWEEPS_SCHEMAS!r}")
    censored = schema in CENSORED_SCHEMAS
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
    # A ``|u1`` array is only ever a censor plane, and only a v2 pack may
    # carry one.  A v1 pack holding one is refused rather than read past:
    # it would mean the writer recorded reasons the schema string denies.
    allowed = (_DTYPE, _CENSOR_DTYPE) if censored else (_DTYPE,)
    for key, entry in arrays.items():
        dtype = entry.get("dtype")
        if dtype not in allowed:
            raise RadarSweepPackError(
                f"{path.name}: array {key} has dtype {dtype!r}, a "
                f"{schema!r} pack carries {allowed!r}")
        width = 1 if dtype == _CENSOR_DTYPE else 4
        elements = int(np.prod(entry["shape"], dtype=np.int64)) if entry["shape"] else 0
        if elements * width != int(entry["bytes"]):
            raise RadarSweepPackError(
                f"{path.name}: array {key} declares shape {entry['shape']} "
                f"but {entry['bytes']} bytes")
        if int(entry["offset"]) + int(entry["bytes"]) > len(payload):
            raise RadarSweepPackError(
                f"{path.name}: array {key} spans past the end of a "
                f"{len(payload)}-byte payload")

    def view(key: str, dtype: str = _DTYPE) -> np.ndarray:
        entry = arrays.get(key)
        if entry is None:
            raise RadarSweepPackError(
                f"{path.name}: metadata references missing array {key!r}")
        if entry.get("dtype") != dtype:
            raise RadarSweepPackError(
                f"{path.name}: array {key!r} has dtype "
                f"{entry.get('dtype')!r}, this use needs {dtype!r}")
        start = int(entry["offset"])
        stop = start + int(entry["bytes"])
        flat = np.frombuffer(payload[start:stop], dtype=dtype)
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
            censor_key = moment.get("censor_array")
            if censored != (censor_key is not None):
                lack = "is missing its" if censored else "carries a"
                raise RadarSweepPackError(
                    f"{path.name}: sweep {entry['sweep_index']} moment "
                    f"{moment['product']} {lack} censor plane, which "
                    f"contradicts schema {schema!r}")
            censor = None
            if censor_key is not None:
                censor = view(str(censor_key), _CENSOR_DTYPE)
                if censor.shape != data.shape:
                    raise RadarSweepPackError(
                        f"{path.name}: sweep {entry['sweep_index']} moment "
                        f"{moment['product']} censor plane has shape "
                        f"{censor.shape} for a {data.shape} moment plane")
                # The two planes have to agree about which gates are
                # numbers.  The failure mode that matters is a censor plane
                # calling a NaN gate MEASURED, or a below-threshold code on
                # a gate that decoded to a real number: either way a
                # clear-air builder downstream would be counting something
                # the radar did not say.
                measured = censor == Censor.MEASURED
                finite = np.isfinite(data)
                if bool(np.any(measured != finite)):
                    raise RadarSweepPackError(
                        f"{path.name}: sweep {entry['sweep_index']} moment "
                        f"{moment['product']} censor plane disagrees with "
                        "its moment plane about which gates are numbers")
                known = np.array(_SCHEMA_CENSOR_CODES[schema],
                                 dtype=censor.dtype)
                unknown = np.setdiff1d(np.unique(censor), known)
                if unknown.size:
                    raise RadarSweepPackError(
                        f"{path.name}: sweep {entry['sweep_index']} moment "
                        f"{moment['product']} censor plane carries unknown "
                        f"codes "
                        f"{sorted(int(c) for c in unknown)}, which a "
                        f"{schema!r} pack does not mint; the schema string "
                        "and the plane must tell the same story")
            moments[moment["product"]] = Moment(
                product=str(moment["product"]),
                unit=str(moment["unit"]),
                gate_count=int(moment["gate_count"]),
                first_gate_range_m=float(moment["first_gate_range_m"]),
                gate_size_m=float(moment["gate_size_m"]),
                data=data,
                censor=censor)
        sweeps.append(Sweep(
            sweep_index=int(entry["sweep_index"]),
            elevation_number=int(entry["elevation_number"]),
            elevation_angle_deg=float(entry["elevation_angle_deg"]),
            nyquist_velocity_ms=(
                None if entry.get("nyquist_velocity_ms") is None
                else float(entry["nyquist_velocity_ms"])),
            nyquist_radials_disagree=bool(
                entry.get("nyquist_radials_disagree", False)),
            nyquist_granularity=(
                None if entry.get("nyquist_granularity") is None
                else str(entry["nyquist_granularity"])),
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
        sweeps=tuple(sweeps),
        pack_schema=str(schema))
