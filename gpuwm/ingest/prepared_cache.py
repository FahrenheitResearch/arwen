"""Hash-bound, atomically published prepared real-data state caches.

The normal real-data path constructs a full 3-D state for every forcing
time even though specified lateral forcing retains only narrow boundary
strips.  A repeated benchmark therefore used to repay the complete vertical
interpolation for every forcing hour before its first model step.

This module persists the integration-ready products of that work:

* the exact time-zero prognostic/diagnostic state;
* the original FP64 vertical coordinate and base state used to load it;
* only the horizontally mapped surface fields needed to initialize physics;
* every immutable host lateral-boundary value/tendency table.

The cache is rebuildable input, not a model restart.  It is nevertheless
fail-closed: callers supply a canonical identity binding the source manifest,
static cache, configuration, namelist, and code revision.  Every array has a
shape/dtype/content digest, the complete manifest has its own digest, and a
temporary directory is renamed into place only after the header is complete.
No pickle or object arrays are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType, SimpleNamespace
from typing import Mapping
import uuid

import numpy as np

from gpuwm.vertical_contract import (
    validate_coordinate_shapes,
    validate_explicit_eta_grid,
)


PREPARED_CACHE_SCHEMA = "gpuwm-prepared-real-cache-v1"
_HEADER_NAME = "header.json"
_MET_REQUIRED = frozenset({
    "LANDSEA", "SKINTEMP", "T2", "U10", "V10",
})
_LEGACY_HRRR_SOIL = frozenset({"SOILT", "SOILW"})
_MET_OPTIONAL = frozenset({
    "SST", "XICE", "SEAICE", "SNOW", "SNOW_EC", "SNOWH",
})
_CANONICAL_SURFACE_REQUIRED = frozenset({
    "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
    "LANDMASK", "SNOW", "SNOWH",
})


class PreparedCacheMismatchError(ValueError):
    """The cache is valid, but belongs to a different requested setup."""


class PreparedCacheCorruptError(ValueError):
    """The cache is incomplete, malformed, or fails a content digest."""


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _json_copy(value):
    """Normalize tuples/numpy-free JSON values and reject non-finite data."""
    return json.loads(_canonical(value))


def prepared_domain_config_identity(domain_config) -> dict[str, object]:
    """JSON-stable domain identity, including an ISO per-domain start."""
    from dataclasses import asdict

    document = asdict(domain_config)
    start_time = document.get("start_time")
    if isinstance(start_time, datetime):
        document["start_time"] = start_time.isoformat()
    return _json_copy(document)


#: Identity fields added to the prepared-domain document AFTER caches
#: carrying the older shape were already in the field, mapped to the
#: value that means "this feature is not in use".
#:
#: v1.1.0 gave every domain an optional per-domain ``start_time`` for
#: staggered nest starts.  The prepared-cache identity is compared by
#: strict equality, and a v1.0.1 header was serialized before the field
#: existed, so after upgrading the wheel EVERY prepared tree in the
#: field became unrunnable -- refused with "d01 cache domain config
#: differs from experiment", a sentence that points at the user's
#: experiment TOML when the cause was a package upgrade.  A node-7
#: validation run diffed the two documents: exactly one added key, and
#: zero value differences among the eleven shared keys and ~110 run
#: fields.
#:
#: Tolerating that is not weakening identity, and the narrowness is what
#: makes it true.  A field ABSENT from the header and holding its
#: documented default in the live configuration describes the same
#: prepared state as a header written before the field existed: the
#: feature is off in both.  A field absent from the header and holding a
#: NON-default value describes a different setup and is still refused,
#: as is any field the header carries and the live configuration
#: contradicts.  Entries are added here deliberately, one per field, by
#: whoever adds the field -- never by a rule that tolerates absence in
#: general.
DEFAULT_TOLERANT_IDENTITY_FIELDS = frozenset({"start_time"})


def undelayed_identity_defaults(experiment) -> dict[str, object]:
    """What each tolerable field holds when its feature is NOT in use.

    ``start_time`` is the only member today, and its not-in-use value is
    not a constant: the loader resolves every domain's start, so a
    domain with no delayed start carries the EXPERIMENT's start rather
    than ``None``.  That is exactly the state a header written before
    delayed starts existed describes, and it is what makes tolerating
    the field's absence a statement about semantics rather than a
    shrug.  A domain that really does start late holds a different
    value, and is refused.
    """

    start = getattr(experiment, "start_time", None)
    return {"start_time": start.isoformat()
            if isinstance(start, datetime) else start}

#: Where the writing gpuwm stamps its version in a cache header.  It
#: sits OUTSIDE the hashed ``basis`` on purpose: the content digest of
#: every cache written before this release must keep verifying exactly
#: as it did, and a stamp that changed the digest would be a second
#: upgrade break in the fix for the first one.
CACHE_WRITER_KEY = "writer"

#: What a header with no stamp tells us: it was written before stamping
#: existed.  Naming that is the honest answer; guessing a version is not.
UNSTAMPED_WRITER = "a release before 1.1.1 (which stamped no version)"


def cache_writer_version(header) -> str:
    """The gpuwm that wrote this cache header, or an honest unknown."""

    writer = header.get(CACHE_WRITER_KEY) if isinstance(header, Mapping) \
        else None
    version = writer.get("gpuwm_version") if isinstance(writer, Mapping) \
        else None
    return version if isinstance(version, str) and version else UNSTAMPED_WRITER


def compare_prepared_domain_config(cached, live, *, not_in_use=None
                                   ) -> tuple[list[str], list[str]]:
    """``(tolerated, differing)`` field paths between two domain identities.

    ``tolerated`` are fields the live document has, the cached one does
    not, and whose live value is the not-in-use value the caller
    declared for them -- provably the same prepared state under a newer
    schema.  ``differing`` is everything else, including a field the
    CACHE carries and this build does not, which means the cache was
    written by a newer gpuwm than this one.

    ``not_in_use`` comes from :func:`undelayed_identity_defaults`.
    Omitting it tolerates nothing: a caller that cannot say what "off"
    means for a field is not in a position to decide the field is off.
    """

    not_in_use = {} if not_in_use is None else dict(not_in_use)
    if not isinstance(cached, Mapping) or not isinstance(live, Mapping):
        # Both absent is not a difference; only one of them being a
        # document is.  Synthetic identities without a domain_config at
        # all are a legitimate shape for callers that bind something
        # else entirely.
        return [], ([] if cached == live else ["domain_config"])
    tolerated: list[str] = []
    differing: list[str] = []

    def walk(cached_node, live_node, prefix: str) -> None:
        for key in sorted(set(cached_node) | set(live_node)):
            path = f"{prefix}{key}"
            if key not in cached_node:
                if (path in DEFAULT_TOLERANT_IDENTITY_FIELDS
                        and path in not_in_use
                        and live_node[key] == not_in_use[path]):
                    tolerated.append(path)
                else:
                    differing.append(path)
                continue
            if key not in live_node:
                differing.append(path)
                continue
            old, new = cached_node[key], live_node[key]
            if isinstance(old, Mapping) and isinstance(new, Mapping):
                walk(old, new, f"{path}.")
            elif old != new:
                differing.append(path)

    walk(cached, live, "")
    return tolerated, differing


def compare_prepared_identity(cached, expected, *, not_in_use=None
                              ) -> tuple[list[str], list[str]]:
    """``(tolerated, differing)`` between a cached and a live identity.

    Only ``domain_config`` is compared default-tolerantly: it is the
    document that grows fields as the configuration schema grows.  Every
    other member of the identity -- the source, static, namelist and
    bridge digests -- is a hash of bytes and stays strictly equal, so
    nothing about what the cache was built FROM is relaxed here.
    """

    if not isinstance(cached, Mapping) or not isinstance(expected, Mapping):
        return [], ["identity"]
    tolerated, differing = compare_prepared_domain_config(
        cached.get("domain_config"), expected.get("domain_config"),
        not_in_use=not_in_use)
    for key in sorted(set(cached) | set(expected)):
        if key == "domain_config":
            continue
        if (key not in cached or key not in expected
                or cached[key] != expected[key]):
            differing.append(key)
    return tolerated, differing


def prepared_identity_refusal(*, subject: str, header, differing,
                              re_prepare: str | None = None) -> str:
    """One sentence naming the versions, the fields, and the way out.

    "d01 cache domain config differs from experiment" was true and
    useless: it named the experiment file, which was innocent, and never
    mentioned that a package upgrade had changed the identity document.
    Whatever survives the default-tolerant comparison above is a real
    mismatch, and it says which fields and between which releases.
    """

    from gpuwm import __version__

    fields = ", ".join(sorted(differing)) or "(none named)"
    sentence = (
        f"{subject} was prepared by {cache_writer_version(header)} and "
        f"this is gpuwm {__version__}; these identity fields differ: "
        f"{fields}")
    if re_prepare:
        return f"{sentence}.  Re-prepare it with: {re_prepare}"
    return (f"{sentence}.  If that difference is a package upgrade rather "
            f"than a configuration change, re-prepare the bundle with the "
            f"front door that wrote it.")


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("prepared-cache arrays must have a numeric dtype")
    return np.ascontiguousarray(array)


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b";")
    digest.update(_canonical(list(array.shape)).encode("ascii"))
    digest.update(b";")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _prepared_met_names(met, *, surface=None) -> tuple[str, ...]:
    """Validate and return the exact mapped-field persistence contract."""
    try:
        available = set(met.fields)
    except AttributeError as exc:
        raise TypeError("prepared met input must expose a fields mapping") from exc
    met_required = _MET_REQUIRED | (
        _LEGACY_HRRR_SOIL if surface is None else frozenset())
    met_names = tuple(sorted((met_required | _MET_OPTIONAL) & available))
    missing_met = sorted(met_required - set(met_names))
    if missing_met:
        raise KeyError(
            f"prepared cache physics inputs are missing {missing_met}")
    return met_names


def select_prepared_met_fields(met, *, surface=None):
    """Detach only mapped fields needed by cache writing and physics setup.

    Native horizontal interpolation produces many full-domain 3-D fields, but
    after the f00 integration state and boundary snapshot have been built only
    a small surface/soil subset remains live.  Materialize that exact contract
    on the host so callers can release the source snapshot and its device
    allocations without changing cache or physics inputs.
    """
    names = _prepared_met_names(met, surface=surface)
    selected = {
        name: np.array(
            _host(met.fields[name]), copy=True, order="C", subok=False)
        for name in names
    }
    return SimpleNamespace(fields=MappingProxyType(selected))


class _BundleWriter:
    def __init__(self, temporary: Path):
        self.temporary = temporary
        self.manifest: dict[str, dict[str, object]] = {}
        self.payload_bytes = 0

    def add(self, key: str, value) -> None:
        if not isinstance(key, str) or not key or key in self.manifest:
            raise ValueError(f"invalid or duplicate prepared-cache key {key!r}")
        array = _host(value)
        # Cache payload names are deliberately compact.  Prepared caches sit
        # below several transaction-owned hierarchy staging directories, so a
        # descriptive private filename can exhaust the legacy Windows path
        # budget even when the final public cache path itself is valid.
        filename = f"a{len(self.manifest):05d}.npy"
        path = self.temporary / filename
        with path.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        self.manifest[key] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
            "sha256": _array_sha256(array),
        }
        self.payload_bytes += int(array.nbytes)


class PreparedCacheReader:
    """Validated cache manifest with checked, one-array-at-a-time reads."""

    def __init__(self, path, *, expected_identity):
        self.path = Path(path)
        try:
            raw = (self.path / _HEADER_NAME).read_text(encoding="utf-8")
            header = json.loads(raw)
        except (FileNotFoundError, OSError, UnicodeDecodeError,
                json.JSONDecodeError) as exc:
            raise PreparedCacheCorruptError(
                f"prepared cache {self.path} has no readable header") from exc
        required = {
            "schema", "status", "identity", "metadata", "arrays",
            "content_sha256", "payload_bytes",
        }
        if not isinstance(header, dict) or required - set(header):
            raise PreparedCacheCorruptError(
                f"prepared cache {self.path} has a malformed header")
        if (header["schema"] != PREPARED_CACHE_SCHEMA
                or header["status"] != "READY"):
            raise PreparedCacheCorruptError(
                f"prepared cache {self.path} is not a READY "
                f"{PREPARED_CACHE_SCHEMA} bundle")
        identity = _json_copy(expected_identity)
        tolerated, differing = compare_prepared_identity(
            header["identity"], identity)
        if differing:
            raise PreparedCacheMismatchError(
                prepared_identity_refusal(
                    subject=f"prepared cache {self.path}",
                    header=header, differing=differing))
        #: Provenance: which identity fields this restore accepted as
        #: schema growth rather than as a match.  Empty is the normal
        #: case, and a caller that records it can show exactly what it
        #: tolerated and why the state is still the state it asked for.
        self.tolerated_identity_fields = tuple(tolerated)
        arrays = header["arrays"]
        if not isinstance(arrays, dict) or not arrays:
            raise PreparedCacheCorruptError(
                "prepared cache array manifest is empty or malformed")
        basis = {
            "schema": header["schema"],
            "identity": header["identity"],
            "metadata": header["metadata"],
            "arrays": arrays,
            "payload_bytes": header["payload_bytes"],
        }
        observed_content = hashlib.sha256(
            _canonical(basis).encode("utf-8")).hexdigest()
        if observed_content != header["content_sha256"]:
            raise PreparedCacheCorruptError(
                "prepared cache header content digest mismatch")
        filenames = []
        payload_bytes = 0
        for key, spec in arrays.items():
            if not isinstance(key, str) or not isinstance(spec, dict):
                raise PreparedCacheCorruptError(
                    "prepared cache contains a malformed array entry")
            try:
                filename = spec["file"]
                shape = spec["shape"]
                dtype = np.dtype(spec["dtype"])
                nbytes = int(spec["nbytes"])
                digest = spec["sha256"]
            except (KeyError, TypeError, ValueError) as exc:
                raise PreparedCacheCorruptError(
                    f"prepared cache array entry {key!r} is malformed") from exc
            candidate = Path(filename)
            if (candidate.name != filename or candidate.is_absolute()
                    or not filename.endswith(".npy")):
                raise PreparedCacheCorruptError(
                    f"prepared cache array {key!r} has unsafe file name")
            if (not isinstance(shape, list)
                    or any(not isinstance(extent, int) or extent < 0
                           for extent in shape)
                    or dtype.hasobject
                    or nbytes != int(np.prod(shape, dtype=np.int64))
                    * dtype.itemsize
                    or not isinstance(digest, str) or len(digest) != 64):
                raise PreparedCacheCorruptError(
                    f"prepared cache array entry {key!r} is inconsistent")
            filenames.append(filename)
            payload_bytes += nbytes
        if len(set(filenames)) != len(filenames):
            raise PreparedCacheCorruptError(
                "prepared cache array manifest reuses a payload file")
        if payload_bytes != int(header["payload_bytes"]):
            raise PreparedCacheCorruptError(
                "prepared cache payload byte total is inconsistent")
        expected_files = set(filenames) | {_HEADER_NAME}
        try:
            actual_files = {entry.name for entry in self.path.iterdir()
                            if entry.is_file()}
            directories = [entry.name for entry in self.path.iterdir()
                           if entry.is_dir()]
        except OSError as exc:
            raise PreparedCacheCorruptError(
                f"prepared cache {self.path} is unreadable") from exc
        if actual_files != expected_files or directories:
            raise PreparedCacheCorruptError(
                "prepared cache file inventory differs from its manifest")
        self.header = header
        self.arrays = arrays

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(self.header["metadata"])

    @property
    def content_sha256(self) -> str:
        return str(self.header["content_sha256"])

    @property
    def payload_bytes(self) -> int:
        return int(self.header["payload_bytes"])

    def read_array(self, key: str) -> np.ndarray:
        try:
            spec = self.arrays[key]
        except KeyError as exc:
            raise PreparedCacheCorruptError(
                f"prepared cache is missing array {key!r}") from exc
        path = self.path / spec["file"]
        try:
            with path.open("rb") as stream:
                array = np.load(stream, allow_pickle=False)
        except (OSError, EOFError, ValueError) as exc:
            raise PreparedCacheCorruptError(
                f"prepared cache array {key!r} is unreadable") from exc
        if (list(array.shape) != spec["shape"]
                or str(array.dtype) != spec["dtype"]
                or int(array.nbytes) != int(spec["nbytes"])
                or _array_sha256(array) != spec["sha256"]):
            raise PreparedCacheCorruptError(
                f"prepared cache array {key!r} fails its manifest")
        return array

    def verify_all(self) -> dict[str, object]:
        for key in sorted(self.arrays):
            self.read_array(key)
        return {
            "schema": PREPARED_CACHE_SCHEMA,
            "status": "PASS",
            "path": str(self.path.resolve()),
            "content_sha256": self.content_sha256,
            "array_count": len(self.arrays),
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True)
class CachedInitialResult:
    """Integration-facing subset of :class:`RealInitResult` restored cold."""

    state: object
    coord: object
    base: object
    surface_pressure: np.ndarray
    surface_qv: np.ndarray


@dataclass(frozen=True)
class RestoredPreparedCache:
    initial_result: CachedInitialResult
    met: object
    surface: object | None
    boundaries: object
    metadata: Mapping[str, object]
    receipt: Mapping[str, object]


def prepared_cache_identity(*, bridge_manifest_sha256: str,
                            source_manifest_sha256: str,
                            static_cache_sha256: str,
                            namelist_sha256: str, domain_config,
                            forcing_hours=None,
                            forcing_offsets_seconds=None,
                            source_identity) -> dict[str, object]:
    """Canonical identity callers must reproduce exactly on every restore."""
    if (forcing_hours is None) == (forcing_offsets_seconds is None):
        raise ValueError(
            "prepared cache identity requires exactly one of forcing_hours "
            "or forcing_offsets_seconds")
    forcing_identity = (
        {"forcing_hours": [int(hour) for hour in forcing_hours]}
        if forcing_hours is not None else
        {"forcing_offsets_seconds": [
            int(offset) for offset in forcing_offsets_seconds]})
    return _json_copy({
        "bridge_manifest_sha256": str(bridge_manifest_sha256).lower(),
        "source_manifest_sha256": str(source_manifest_sha256).lower(),
        "static_cache_sha256": str(static_cache_sha256).lower(),
        "namelist_sha256": str(namelist_sha256).lower(),
        "domain_config": prepared_domain_config_identity(domain_config),
        **forcing_identity,
        "source_identity": source_identity,
    })


def _coord_metadata(coord) -> dict[str, object]:
    result = {}
    for field in dataclass_fields(coord):
        value = getattr(coord, field.name)
        if not isinstance(value, np.ndarray):
            result[field.name] = value
    return _json_copy(result)


def _base_metadata(base) -> dict[str, object]:
    result = {}
    for field in dataclass_fields(base):
        value = getattr(base, field.name)
        if not isinstance(value, np.ndarray):
            result[field.name] = value
    return _json_copy(result)


def _is_nested_child_identity(identity) -> bool:
    """Return whether identity explicitly binds a parent-forced child."""

    if not isinstance(identity, Mapping):
        return False
    domain = identity.get("domain_config")
    if not isinstance(domain, Mapping):
        return False
    run = domain.get("run")
    return (isinstance(run, Mapping)
            and isinstance(domain.get("parent_id"), int)
            and int(domain["parent_id"]) > 0
            and run.get("nested") is True
            and run.get("specified") is False)


def _restore_lbc_mode(*, lbc_metadata, identity,
                      allow_nested_without_lbc: bool) -> str:
    """Resolve root/child boundary ownership without importing CuPy."""

    if not isinstance(allow_nested_without_lbc, bool):
        raise TypeError("allow_nested_without_lbc must be bool")
    if lbc_metadata is not None:
        if not isinstance(lbc_metadata, Mapping):
            raise PreparedCacheCorruptError(
                "prepared cache LBC metadata must be an object or null")
        return "external"
    if (allow_nested_without_lbc
            and _is_nested_child_identity(identity)):
        return "nested-parent-forced"
    raise PreparedCacheMismatchError(
        "nested export-only prepared cache has no external LBCs and "
        "cannot be restored as a standalone forecast root")


def _prepared_cache_staging_path(path: Path, *, nonce: str | None = None
                                 ) -> Path:
    """Return a compact, create-only sibling used for atomic publication.

    The target name must not be repeated in this private basename: caches are
    nested below other transaction staging roots and that repetition can make
    an otherwise valid published tree uncreatable on Windows.  The caller's
    ``mkdir`` remains the collision/ownership authority.
    """

    token = uuid.uuid4().hex[:10] if nonce is None else nonce
    if (not isinstance(token, str) or len(token) != 10
            or any(character not in "0123456789abcdef" for character in token)):
        raise ValueError(
            "prepared-cache staging nonce must be 10 lowercase hex characters")
    return path.with_name(f".p-{token}")


def write_prepared_cache(path, *, identity, initial_result, met,
                         boundaries, surface=None,
                         metadata=None) -> dict[str, object]:
    """Write one immutable prepared-state bundle and publish atomically.

    Root-domain caches require external lateral boundaries.  A cache may omit
    them only when its identity explicitly binds a nested, non-specified child;
    that export-only cache feeds ``wrfinput_dNN`` and cannot be restored as a
    standalone forecast root.
    """
    from gpuwm.state_serialization_contract import (
        STATE_SERIALIZED_ATTRS,
        setup_fingerprint,
    )

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prepared cache {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _prepared_cache_staging_path(path)
    temporary.mkdir()
    writer = _BundleWriter(temporary)
    try:
        state_names = []
        for name in STATE_SERIALIZED_ATTRS:
            value = getattr(initial_result.state, name, None)
            if value is not None:
                writer.add(f"state/{name}", value)
                state_names.append(name)
        coord_arrays = []
        for field in dataclass_fields(initial_result.coord):
            value = getattr(initial_result.coord, field.name)
            if isinstance(value, np.ndarray):
                writer.add(f"coord/{field.name}", value)
                coord_arrays.append(field.name)
        base_arrays = []
        for field in dataclass_fields(initial_result.base):
            value = getattr(initial_result.base, field.name)
            if isinstance(value, np.ndarray):
                writer.add(f"base/{field.name}", value)
                base_arrays.append(field.name)
        writer.add("result/surface_pressure", initial_result.surface_pressure)
        writer.add("result/surface_qv", initial_result.surface_qv)

        met_names = _prepared_met_names(met, surface=surface)
        for name in met_names:
            writer.add(f"met/{name}", met.fields[name])

        surface_names = []
        if surface is not None:
            if not isinstance(surface, Mapping):
                raise TypeError("canonical prepared surface must be a mapping")
            missing_surface = sorted(
                _CANONICAL_SURFACE_REQUIRED - set(surface))
            if missing_surface:
                raise KeyError(
                    "canonical prepared surface is missing "
                    f"{missing_surface}")
            surface_names = sorted(_CANONICAL_SURFACE_REQUIRED)
            for name in surface_names:
                writer.add(f"surface/{name}", surface[name])

        if boundaries is None:
            if not _is_nested_child_identity(identity):
                raise ValueError(
                    "omitting prepared-cache LBCs requires an identity-bound "
                    "nested non-specified child")
            lbc_metadata = None
        else:
            interval_metadata = []
            for index, interval in enumerate(boundaries.intervals):
                field_names = sorted(interval.fields)
                interval_metadata.append({
                    "start_seconds": float(interval.start_seconds),
                    "end_seconds": float(interval.end_seconds),
                    "fields": field_names,
                })
                for name in field_names:
                    field = interval.fields[name]
                    for side_name in ("west", "east", "south", "north"):
                        side = getattr(field, side_name)
                        prefix = f"lbc/{index}/{name}/{side_name}"
                        writer.add(f"{prefix}/value", side.value)
                        writer.add(f"{prefix}/tendency", side.tendency)
            lbc_metadata = {
                "spec_bdy_width": int(boundaries.spec_bdy_width),
                "spec_zone": int(boundaries.spec_zone),
                "relax_zone": int(boundaries.relax_zone),
                "intervals": interval_metadata,
            }

        cache_metadata = {
            "user": _json_copy(metadata or {}),
            "state_names": state_names,
            "coord_arrays": coord_arrays,
            "coord_scalars": _coord_metadata(initial_result.coord),
            "base_arrays": base_arrays,
            "base_scalars": _base_metadata(initial_result.base),
            "met_fields": met_names,
            "surface_fields": surface_names,
            "lbc": lbc_metadata,
            "setup_fingerprint": setup_fingerprint(initial_result.state),
        }
        basis = {
            "schema": PREPARED_CACHE_SCHEMA,
            "identity": _json_copy(identity),
            "metadata": cache_metadata,
            "arrays": writer.manifest,
            "payload_bytes": writer.payload_bytes,
        }
        from gpuwm import __version__

        header = {
            **basis,
            "status": "READY",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            # Outside `basis`, so the content digest of a cache means
            # exactly what it meant before this release.  Its job is to
            # let a refusal name the release that wrote the bundle
            # instead of blaming the user's experiment file for a
            # package upgrade.
            CACHE_WRITER_KEY: {"gpuwm_version": __version__},
            "content_sha256": hashlib.sha256(
                _canonical(basis).encode("utf-8")).hexdigest(),
        }
        header_path = temporary / _HEADER_NAME
        header_path.write_text(
            json.dumps(header, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema": PREPARED_CACHE_SCHEMA,
        "status": "BUILT",
        "path": str(path.resolve()),
        "content_sha256": header["content_sha256"],
        "array_count": len(writer.manifest),
        "payload_bytes": writer.payload_bytes,
    }


def restore_prepared_cache(path, *, expected_identity, cfg, static,
                           allow_nested_without_lbc: bool = False
                           ) -> RestoredPreparedCache:
    """Validate and restore an integration-ready GPU state.

    The historical/default surface restores a specified root and therefore
    requires its complete external lateral-boundary sequence.  A prepared
    hierarchy runner may explicitly opt into restoring an identity-bound
    nested child.  Such a child deliberately has no external LBC payload: its
    live :class:`gpuwm.core.nest.NestCoupler` rebuilds rolling boundaries from
    the parent after each parent step.  The opt-in never permits a root cache
    to omit LBCs and never turns a child into a standalone root.
    """
    # Resolve the pure ownership decision before importing the optional CUDA
    # runtime.  This keeps malformed caller contracts deterministic on CPU-
    # only installations and makes the hierarchy exception directly testable.
    reader = PreparedCacheReader(path, expected_identity=expected_identity)
    metadata = reader.header["metadata"]
    lbc_mode = _restore_lbc_mode(
        lbc_metadata=metadata["lbc"], identity=expected_identity,
        allow_nested_without_lbc=allow_nested_without_lbc)
    import cupy as cp

    from gpuwm.core.grid import BaseState, VerticalCoord
    from gpuwm.core.state import DomainState
    from gpuwm.ingest.lateral_bc import (
        BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
        attach_lateral_boundaries,
    )
    from gpuwm.state_serialization_contract import (
        STATE_SERIALIZED_ATTRS,
        setup_fingerprint,
    )

    coord_shapes = {
        name: reader.arrays[f"coord/{name}"]["shape"]
        for name in metadata["coord_arrays"]
        if f"coord/{name}" in reader.arrays
    }
    try:
        validate_coordinate_shapes(
            coord_shapes, nz=cfg.nz, context="prepared-cache restore")
    except ValueError as exc:
        raise PreparedCacheMismatchError(str(exc)) from exc

    coord_values = dict(metadata["coord_scalars"])
    for name in metadata["coord_arrays"]:
        coord_values[name] = reader.read_array(f"coord/{name}")
    coord = VerticalCoord(**coord_values)

    base_values = dict(metadata["base_scalars"])
    for name in metadata["base_arrays"]:
        base_values[name] = reader.read_array(f"base/{name}")
    base = BaseState(**base_values)
    try:
        validate_explicit_eta_grid(
            coord.znw,
            nz=cfg.nz,
            p_top=base.p_top,
            context="prepared-cache restore",
        )
    except (TypeError, ValueError) as exc:
        raise PreparedCacheMismatchError(str(exc)) from exc

    state = DomainState(cfg)
    state.load_base(coord, base)
    state.set_map_coriolis(
        static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
        static["F"], static["E"], sina=static["SINALPHA"],
        cosa=static["COSALPHA"])
    expected_state_names = [
        name for name in STATE_SERIALIZED_ATTRS
        if getattr(state, name, None) is not None]
    if metadata["state_names"] != expected_state_names:
        raise PreparedCacheMismatchError(
            "prepared cache state inventory differs from the active config")
    for name in expected_state_names:
        host = reader.read_array(f"state/{name}")
        target = getattr(state, name)
        if tuple(host.shape) != tuple(target.shape) or host.dtype != target.dtype:
            raise PreparedCacheMismatchError(
                f"prepared cache state/{name} shape or dtype differs from "
                "the active config")
        target[...] = cp.asarray(host)

    lbc_meta = metadata["lbc"]
    if lbc_mode == "nested-parent-forced":
        boundaries = None
    else:
        intervals = []
        for index, interval_meta in enumerate(lbc_meta["intervals"]):
            field_map = {}
            for name in interval_meta["fields"]:
                sides = {}
                for side_name in ("west", "east", "south", "north"):
                    prefix = f"lbc/{index}/{name}/{side_name}"
                    sides[side_name] = SideBoundary(
                        reader.read_array(f"{prefix}/value"),
                        reader.read_array(f"{prefix}/tendency"))
                field_map[name] = FieldBoundary(**sides)
            intervals.append(BoundaryInterval(
                float(interval_meta["start_seconds"]),
                float(interval_meta["end_seconds"]), field_map))
        boundaries = LateralBoundaries(
            tuple(intervals), int(lbc_meta["spec_bdy_width"]),
            int(lbc_meta["spec_zone"]), int(lbc_meta["relax_zone"]))
        attach_lateral_boundaries(state, boundaries)
    observed_setup = setup_fingerprint(state)
    if observed_setup != metadata["setup_fingerprint"]:
        raise PreparedCacheMismatchError(
            "prepared cache reconstructed a different setup fingerprint")

    met_fields = {
        name: reader.read_array(f"met/{name}")
        for name in metadata["met_fields"]}
    surface_names = metadata.get("surface_fields", [])
    surface_fields = {
        name: reader.read_array(f"surface/{name}")
        for name in surface_names}
    result = CachedInitialResult(
        state=state, coord=coord, base=base,
        surface_pressure=reader.read_array("result/surface_pressure"),
        surface_qv=reader.read_array("result/surface_qv"))
    receipt = {
        "schema": PREPARED_CACHE_SCHEMA,
        "status": "RESTORED",
        "path": str(reader.path.resolve()),
        "content_sha256": reader.content_sha256,
        "array_count": len(reader.arrays),
        "payload_bytes": reader.payload_bytes,
        "setup_fingerprint": observed_setup,
    }
    return RestoredPreparedCache(
        initial_result=result,
        met=SimpleNamespace(fields=MappingProxyType(met_fields)),
        surface=(SimpleNamespace(fields=MappingProxyType(surface_fields))
                 if surface_fields else None),
        boundaries=boundaries,
        metadata=MappingProxyType(metadata["user"]),
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "CACHE_WRITER_KEY", "CachedInitialResult",
    "DEFAULT_TOLERANT_IDENTITY_FIELDS", "PREPARED_CACHE_SCHEMA",
    "PreparedCacheCorruptError", "PreparedCacheMismatchError",
    "PreparedCacheReader", "RestoredPreparedCache", "UNSTAMPED_WRITER",
    "cache_writer_version", "compare_prepared_domain_config",
    "compare_prepared_identity", "prepared_cache_identity",
    "prepared_domain_config_identity", "prepared_identity_refusal",
    "restore_prepared_cache",
    "select_prepared_met_fields", "write_prepared_cache",
]
