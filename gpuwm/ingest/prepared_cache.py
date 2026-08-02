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
from datetime import datetime, timedelta, timezone
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
from gpuwm.namelist_seal import validated_namelist_extension_invariant


PREPARED_CACHE_SCHEMA = "gpuwm-prepared-real-cache-v1"
SEALED_PREPARED_EXTENSION_MODE = "sealed-prefix-v1"
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
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _plain(value):
    """Recursively unwrap read-only containers into plain JSON types.

    ``json`` serializes ``dict``/``list`` and nothing else that behaves
    like them.  A ``MappingProxyType`` -- the exact type
    :func:`restore_prepared_cache` hands back as
    ``CachedInitialResult.hydrometeor_initialization``, deliberately, so
    a restored document cannot be mutated -- is a ``Mapping`` but not a
    ``dict``, so ``json.dumps`` raised ``TypeError: Object of type
    mappingproxy is not JSON serializable``.  That turned "restore a
    prepared root, then write the child cache derived from it" into a
    crash with no mention of the field that caused it.

    Unwrapping here rather than at each call site keeps ONE normalization
    for every identity, metadata and manifest document this module
    canonicalizes, and it changes no output: a proxy serializes to the
    same object its underlying mapping does.  Tuples already normalized
    to lists through ``json``; sets and frozensets do not, and are not
    accepted -- an unordered container has no canonical serialization, so
    guessing one would put an order-dependent digest in an identity.
    """

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError(
            "prepared-cache documents cannot contain a set: an unordered "
            "container has no canonical serialization, and an identity "
            "digest must not depend on iteration order")
    return value


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


#: Identity fields that describe when the run WRITES, not what it
#: integrates.  gpuwm already publishes this partition twice, for the two
#: other places the same question is asked:
#: :data:`gpuwm.io.restart.CONFIG_RUN_LENGTH_FIELDS` (which RunConfig
#: fields may differ between a checkpoint's writer and its resumer) and
#: :func:`gpuwm.core.model.experiment_fingerprint` (which fields are
#: excluded from the trajectory hash).  A prepared cache is the same
#: question one stage earlier -- it holds an initial state and its
#: boundaries, and preparation writes no history or restart frame at all
#: -- so it asks the same table instead of keeping a third opinion.
#:
#: Binding these here is what refused a two-domain HRRR tree whose d02
#: history cadence was 900 s beside d01's 3600 s, and a hierarchy whose
#: namelist carried ``restart_interval = 60``: both are output cadences,
#: neither changes one model step.
#:
#: ``run.run_seconds`` belongs to the same partition and was missing
#: from it: a prepared cache is an initial state plus the boundary
#: tables for the times it decoded, and the forecast LENGTH changes
#: neither.  Its omission is what made "extend a run from its
#: checkpoint" -- the worked example in FIRST-LIGHT section 7 -- refuse
#: on the prepared routes: the length moved this identity, so the cache
#: had to be re-prepared to run one hour longer.  Whether the prepared
#: forcing actually REACHES the requested length is a separate gate and
#: is unchanged; both prepared runners already refuse a run past their
#: prepared coverage (``prepared_single_domain_forecast`` at its
#: cadence/coverage check, ``prepared_domain_tree_forecast`` at
#: "experiment duration exceeds prepared forcing hours"), so tolerating
#: the field here widens nothing.
#: The DOMAIN-level ``history_interval_s`` is the same knob as
#: ``run.output_interval_s`` under its other spelling, and this document
#: carries both.  Dropping only the ``run.`` one left the cadence bound
#: after all: a tree restarted with a changed history cadence -- one of
#: the three changes ``--restart`` publishes as permitted -- was refused
#: naming a field that says nothing about the prepared state.
NON_TRAJECTORY_IDENTITY_FIELDS = frozenset({
    "run.run_seconds",
    "run.output_interval_s",
    "run.restart_interval_s",
    "history_interval_s",
})

#: Trajectory-inert diagnostic toggles, on the same footing as the
#: cadences above and for the same published reason:
#: :data:`gpuwm.io.restart.CONFIG_DIAGNOSTIC_FIELDS` already allows
#: ``nwp_diagnostics`` to differ across a restart boundary because
#: flipping it cannot change the trajectory (pinned by
#: tests/test_uh_lifecycle.py).  It selects which diagnostic accumulators
#: a FORECAST carries; a prepared cache predates every one of them.
INERT_DIAGNOSTIC_IDENTITY_FIELDS = frozenset({
    "run.nwp_diagnostics",
})


def effective_prepared_domain_config(document):
    """Canonicalize one prepared-domain identity for comparison.

    Two documents describe the same prepared d0N when they agree here.
    Three normalizations, each of which was a real refusal:

    * cadence and inert-diagnostic fields (the two tables above) are
      dropped -- they say when the forecast writes, not what it
      integrates;
    * ``cudt_minutes`` is pinned to 0 when ``cu_physics == 0``, because a
      cumulus interval with no cumulus scheme is dead namelist state and
      the two producers spell the dead value differently (a WRF importer
      that omits the key inherits RunConfig's 5.0; a shipped physics
      profile states 0.0);
    * ``radt_minutes`` follows a positive ``radt``, which overrides it
      (gpuwm/config.py) -- the legacy shadow field is not a second
      radiation cadence.

    A non-document is returned unchanged so callers can hand this
    whatever the header gave them.
    """

    if not isinstance(document, Mapping):
        return document
    normalized = _plain(document)
    # Top-level (domain) entries first: the output cadence lives here as
    # well as under ``run``, and a document may carry it with no ``run``
    # section at all.
    for path in sorted(NON_TRAJECTORY_IDENTITY_FIELDS
                       | INERT_DIAGNOSTIC_IDENTITY_FIELDS):
        if "." not in path:
            normalized.pop(path, None)
    run = normalized.get("run")
    if not isinstance(run, dict):
        return normalized
    for path in sorted(NON_TRAJECTORY_IDENTITY_FIELDS
                       | INERT_DIAGNOSTIC_IDENTITY_FIELDS):
        section, _, key = path.partition(".")
        if section == "run" and key:
            run.pop(key, None)
    if run.get("radt", 0.0) > 0.0:
        run["radt_minutes"] = run["radt"]
    if run.get("cu_physics") == 0:
        run["cudt_minutes"] = 0.0
    return normalized


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

    def link_verified(self, key: str, reader: "PreparedCacheReader") -> None:
        """Reuse one already-verified immutable payload without copying it.

        Extension bundles are siblings in the operational work tree, so a
        hard link is both atomic and space-constant.  We deliberately do not
        fall back to a byte copy: silently doing so would turn an hourly
        append into quadratic disk traffic.  The completed staging bundle is
        verified again before publication, which catches a predecessor that
        changed while its links were being assembled.
        """
        if not isinstance(key, str) or not key or key in self.manifest:
            raise ValueError(f"invalid or duplicate prepared-cache key {key!r}")
        try:
            source_spec = reader.arrays[key]
        except KeyError as exc:
            raise PreparedCacheCorruptError(
                f"prepared cache is missing array {key!r}") from exc
        filename = f"a{len(self.manifest):05d}.npy"
        source = reader.path / source_spec["file"]
        destination = self.temporary / filename
        try:
            os.link(source, destination)
        except OSError as exc:
            raise PreparedCacheMismatchError(
                "prepared-cache extension requires same-filesystem hard-link "
                "reuse; refusing a quadratic payload copy") from exc
        spec = _json_copy(source_spec)
        spec["file"] = filename
        self.manifest[key] = spec
        self.payload_bytes += int(spec["nbytes"])


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
    hydrometeor_initialization: Mapping[str, object]


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
                            source_identity,
                            namelist_extension_invariant=None,
                            ) -> dict[str, object]:
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
    identity = {
        "bridge_manifest_sha256": str(bridge_manifest_sha256).lower(),
        "source_manifest_sha256": str(source_manifest_sha256).lower(),
        "static_cache_sha256": str(static_cache_sha256).lower(),
        "namelist_sha256": str(namelist_sha256).lower(),
        "domain_config": prepared_domain_config_identity(domain_config),
        **forcing_identity,
        "source_identity": source_identity,
    }
    if namelist_extension_invariant is not None:
        identity["namelist_extension_invariant"] = \
            namelist_extension_invariant
    return _json_copy(identity)


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
                         metadata=None,
                         sealed_forcing_extension: bool = False
                         ) -> dict[str, object]:
    """Write one immutable prepared-state bundle and publish atomically.

    Root-domain caches require external lateral boundaries.  A cache may omit
    them only when its identity explicitly binds a nested, non-specified child;
    that export-only cache feeds ``wrfinput_dNN`` and cannot be restored as a
    standalone forecast root.
    """
    from gpuwm.state_serialization_contract import (
        STATE_SERIALIZED_ATTRS, lateral_boundary_prefix_identity,
        setup_core_fingerprint, setup_fingerprint,
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
            "hydrometeor_initialization": _json_copy(
                getattr(initial_result, "hydrometeor_initialization", {})),
        }
        if sealed_forcing_extension:
            if boundaries is None:
                raise ValueError(
                    "sealed prepared-cache forcing requires root LBCs")
            cache_metadata.update({
                "forcing_extension_mode": SEALED_PREPARED_EXTENSION_MODE,
                "setup_core_fingerprint": setup_core_fingerprint(
                    initial_result.state),
                "lateral_boundary_prefix": lateral_boundary_prefix_identity(
                    initial_result.state),
            })
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


def _reader_boundaries(reader: PreparedCacheReader):
    from gpuwm.ingest.lateral_bc import (
        BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
    )

    lbc = reader.header.get("metadata", {}).get("lbc")
    if not isinstance(lbc, dict) or not isinstance(lbc.get("intervals"), list):
        raise PreparedCacheMismatchError(
            f"prepared cache {reader.path} has no external LBC inventory")
    intervals = []
    for index, row in enumerate(lbc["intervals"]):
        if (not isinstance(row, dict) or set(row) != {
                "start_seconds", "end_seconds", "fields"}
                or not isinstance(row["fields"], list)
                or not row["fields"]):
            raise PreparedCacheCorruptError(
                f"prepared cache LBC interval {index} is malformed")
        fields = {}
        for name in row["fields"]:
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                prefix = f"lbc/{index}/{name}/{side_name}"
                sides[side_name] = SideBoundary(
                    reader.read_array(f"{prefix}/value"),
                    reader.read_array(f"{prefix}/tendency"))
            fields[name] = FieldBoundary(**sides)
        intervals.append(BoundaryInterval(
            float(row["start_seconds"]), float(row["end_seconds"]), fields))
    return LateralBoundaries(
        tuple(intervals), int(lbc["spec_bdy_width"]),
        int(lbc["spec_zone"]), int(lbc["relax_zone"]))


def _forcing_prefix(boundaries):
    from gpuwm.state_serialization_contract import (
        lateral_boundary_prefix_identity,
    )

    return lateral_boundary_prefix_identity(
        SimpleNamespace(lateral_boundaries=boundaries))


def _without_run_window(value, *, omit_start_time=False):
    result = _json_copy(value)
    if isinstance(result, dict) and isinstance(result.get("run"), dict):
        result["run"].pop("run_seconds", None)
    if isinstance(result, dict) and omit_start_time:
        result.pop("start_time", None)
    return result


def _without_source_window(value, *, omit_model_start=False):
    result = _json_copy(value)
    if isinstance(result, dict):
        for key in ("source_forecast_hours", "model_forcing_hours"):
            result.pop(key, None)
        if omit_model_start:
            result.pop("model_start_time", None)
    return result


def _identity_time(value, *, label):
    if not isinstance(value, str):
        raise PreparedCacheMismatchError(
            f"prepared-cache {label} must be an ISO timestamp")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise PreparedCacheMismatchError(
            f"prepared-cache {label} must be an ISO timestamp") from exc


def _domain_run_seconds(identity, *, label):
    domain = identity.get("domain_config") if isinstance(identity, dict) \
        else None
    run = domain.get("run") if isinstance(domain, dict) else None
    value = run.get("run_seconds") if isinstance(run, dict) else None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value)):
        raise PreparedCacheMismatchError(
            f"prepared-cache {label} has no finite run_seconds")
    return float(value)


def _namelist_extension_invariant(identity, *, label):
    invariant = identity.get("namelist_extension_invariant") \
        if isinstance(identity, dict) else None
    try:
        return validated_namelist_extension_invariant(
            invariant, context=f"prepared-cache {label}")
    except ValueError:
        raise PreparedCacheMismatchError(
            f"prepared-cache {label} has no valid namelist extension "
            "invariant") from None


def extend_prepared_cache(path, *, predecessor, suffix, identity,
                          metadata, source_manifest_extension,
                          bridge_manifest_extension,
                          ) -> dict[str, object]:
    """Publish one cache by appending exactly one verified LBC interval.

    ``suffix`` is a normal two-time cache prepared only for the old endpoint
    and the newly arrived hour.  Initial state, static-derived setup, and all
    old forcing arrays come from ``predecessor``; only the suffix interval is
    admitted, after its shared FP32 endpoint frame matches cryptographically.
    """
    from gpuwm import __version__
    from gpuwm.ingest.lateral_bc import BoundaryInterval, LateralBoundaries

    path = Path(path)
    predecessor = Path(predecessor)
    suffix = Path(suffix)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prepared cache {path}")
    prior_header_path = predecessor / _HEADER_NAME
    suffix_header_path = suffix / _HEADER_NAME
    prior_header_before = hashlib.sha256(
        prior_header_path.read_bytes()).hexdigest()
    suffix_header_before = hashlib.sha256(
        suffix_header_path.read_bytes()).hexdigest()
    prior_header = json.loads(prior_header_path.read_text(encoding="utf-8"))
    suffix_header = json.loads(suffix_header_path.read_text(encoding="utf-8"))
    prior_reader = PreparedCacheReader(
        predecessor, expected_identity=prior_header.get("identity"))
    suffix_reader = PreparedCacheReader(
        suffix, expected_identity=suffix_header.get("identity"))
    prior_reader.verify_all()
    suffix_reader.verify_all()
    prior_meta = prior_reader.header["metadata"]
    if (prior_meta.get("forcing_extension_mode")
            != SEALED_PREPARED_EXTENSION_MODE):
        raise PreparedCacheMismatchError(
            "predecessor cache was not intentionally sealed for extension")

    old_identity = prior_reader.header["identity"]
    new_identity = _json_copy(identity)
    old_hours = old_identity.get("forcing_hours")
    new_hours = new_identity.get("forcing_hours")
    if (not isinstance(old_hours, list) or not old_hours
            or old_hours != list(range(len(old_hours)))
            or new_hours != old_hours + [len(old_hours)]):
        raise PreparedCacheMismatchError(
            "prepared-cache extension must append exactly one contiguous "
            "forcing hour")
    for key in ("static_cache_sha256",):
        if new_identity.get(key) != old_identity.get(key):
            raise PreparedCacheMismatchError(
                f"prepared-cache extension changes immutable {key}")
    old_namelist_invariant = _namelist_extension_invariant(
        old_identity, label="predecessor identity")
    if (_namelist_extension_invariant(
            new_identity, label="extended identity")
            != old_namelist_invariant):
        raise PreparedCacheMismatchError(
            "prepared-cache extension changes immutable namelist fields")
    if (_without_run_window(new_identity.get("domain_config"))
            != _without_run_window(old_identity.get("domain_config"))):
        raise PreparedCacheMismatchError(
            "prepared-cache extension changes immutable domain config")
    old_source = old_identity.get("source_identity")
    new_source = new_identity.get("source_identity")
    if _without_source_window(old_source) != _without_source_window(new_source):
        raise PreparedCacheMismatchError(
            "prepared-cache extension changes immutable source identity")
    if (not isinstance(new_source, dict)
            or new_source.get("source_forecast_hours") != new_hours
            or new_source.get("model_forcing_hours") != new_hours):
        raise PreparedCacheMismatchError(
            "prepared-cache extension source window is not the full prefix")

    suffix_identity = suffix_reader.header["identity"]
    if suffix_identity.get("forcing_hours") != [0, 1]:
        raise PreparedCacheMismatchError(
            "prepared-cache suffix must contain exactly two source frames")
    for key in ("static_cache_sha256",):
        if suffix_identity.get(key) != old_identity.get(key):
            raise PreparedCacheMismatchError(
                f"prepared-cache suffix changes immutable {key}")
    if (_namelist_extension_invariant(
            suffix_identity, label="suffix identity")
            != old_namelist_invariant):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix changes immutable namelist fields")
    if suffix_identity.get("namelist_sha256") != new_identity.get(
            "namelist_sha256"):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix does not use the extended namelist bytes")
    if suffix_identity.get("source_manifest_sha256") != new_identity.get(
            "source_manifest_sha256"):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix is not bound to the extended source "
            "manifest")
    expected_bridge_extension = {
        "schema": "gpuwm-bridge-manifest-prefix-extension-v1",
        "predecessor_sha256": old_identity.get("bridge_manifest_sha256"),
        "suffix_sha256": suffix_identity.get("bridge_manifest_sha256"),
        "extended_sha256": new_identity.get("bridge_manifest_sha256"),
        "old_source_forecast_hours": old_source.get(
            "source_forecast_hours") if isinstance(old_source, dict) else None,
        "new_source_forecast_hours": new_source.get(
            "source_forecast_hours") if isinstance(new_source, dict) else None,
        "suffix_source_forecast_hours": [len(old_hours) - 1, len(old_hours)],
    }
    bridge_extension = _json_copy(bridge_manifest_extension)
    if (not isinstance(bridge_extension, dict)
            or any(bridge_extension.get(key) != value
                   for key, value in expected_bridge_extension.items())
            or not isinstance(bridge_extension.get("retained_entries"), int)
            or bridge_extension["retained_entries"] <= 0
            or not isinstance(bridge_extension.get("added_entries"), list)
            or not bridge_extension["added_entries"]):
        raise PreparedCacheMismatchError(
            "prepared-cache bridge-manifest prefix proof is malformed or "
            "belongs to another extension")
    if (_without_run_window(
            suffix_identity.get("domain_config"), omit_start_time=True)
            != _without_run_window(
                old_identity.get("domain_config"), omit_start_time=True)):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix changes immutable domain config")
    suffix_source = suffix_identity.get("source_identity")
    if (_without_source_window(suffix_source, omit_model_start=True)
            != _without_source_window(old_source, omit_model_start=True)):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix changes executable source identity")
    suffix_source_hours = [len(old_hours) - 1, len(old_hours)]
    if (not isinstance(suffix_source, dict)
            or suffix_source.get("source_forecast_hours")
            != suffix_source_hours
            or suffix_source.get("model_forcing_hours") != [0, 1]):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix does not contain the old terminal source "
            "frame and exactly one new hour")
    old_model_start = _identity_time(
        old_source.get("model_start_time") if isinstance(old_source, dict)
        else None, label="predecessor model_start_time")
    new_model_start = _identity_time(
        new_source.get("model_start_time"), label="extended model_start_time")
    suffix_model_start = _identity_time(
        suffix_source.get("model_start_time"), label="suffix model_start_time")
    if new_model_start != old_model_start:
        raise PreparedCacheMismatchError(
            "prepared-cache extension changes model time zero")
    expected_suffix_start = old_model_start + timedelta(
        hours=len(old_hours) - 1)
    if suffix_model_start != expected_suffix_start:
        raise PreparedCacheMismatchError(
            "prepared-cache suffix model start is not the old terminal hour")
    suffix_domain = suffix_identity.get("domain_config")
    if (not isinstance(suffix_domain, dict)
            or _identity_time(
                suffix_domain.get("start_time"),
                label="suffix domain start_time") != expected_suffix_start):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix domain start is not the old terminal hour")
    if (_domain_run_seconds(old_identity, label="predecessor identity")
            != float(len(old_hours) - 1) * 3600.0
            or _domain_run_seconds(new_identity, label="extended identity")
            != float(len(new_hours) - 1) * 3600.0
            or _domain_run_seconds(suffix_identity, label="suffix identity")
            != 3600.0):
        raise PreparedCacheMismatchError(
            "prepared-cache run windows do not exactly cover their forcing")
    expected_manifest_extension = {
        "schema": "gpuwm-source-manifest-prefix-extension-v1",
        "predecessor_sha256": old_identity.get("source_manifest_sha256"),
        "extended_sha256": new_identity.get("source_manifest_sha256"),
        "old_source_forecast_hours": old_source.get(
            "source_forecast_hours"),
        "new_source_forecast_hours": new_source.get(
            "source_forecast_hours"),
        "suffix_source_forecast_hours": suffix_source_hours,
    }
    manifest_extension = _json_copy(source_manifest_extension)
    if (not isinstance(manifest_extension, dict)
            or any(manifest_extension.get(key) != value
                   for key, value in expected_manifest_extension.items())
            or not isinstance(manifest_extension.get("retained_entries"), int)
            or manifest_extension["retained_entries"] <= 0
            or not isinstance(manifest_extension.get("added_entries"), list)
            or not manifest_extension["added_entries"]):
        raise PreparedCacheMismatchError(
            "prepared-cache source-manifest prefix proof is malformed or "
            "belongs to another extension")

    prior_boundaries = _reader_boundaries(prior_reader)
    suffix_boundaries = _reader_boundaries(suffix_reader)
    if len(suffix_boundaries.intervals) != 1:
        raise PreparedCacheMismatchError(
            "prepared-cache suffix must contain one boundary interval")
    old_prefix = _forcing_prefix(prior_boundaries)
    if old_prefix != prior_meta.get("lateral_boundary_prefix"):
        raise PreparedCacheMismatchError(
            "predecessor cache forcing prefix differs from its seal")
    old_end = float(len(old_hours) - 1) * 3600.0
    if float(prior_boundaries.intervals[-1].end_seconds) != old_end:
        raise PreparedCacheMismatchError(
            "predecessor forcing is not sealed at its advertised endpoint")
    suffix_interval = suffix_boundaries.intervals[0]
    if (float(suffix_interval.start_seconds) != 0.0
            or float(suffix_interval.end_seconds) != 3600.0):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix interval is not one hour")
    if (suffix_boundaries.spec_bdy_width
            != prior_boundaries.spec_bdy_width
            or suffix_boundaries.spec_zone != prior_boundaries.spec_zone
            or suffix_boundaries.relax_zone != prior_boundaries.relax_zone
            or set(suffix_interval.fields)
            != set(prior_boundaries.intervals[-1].fields)):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix changes boundary geometry or fields")
    shifted = BoundaryInterval(
        old_end, old_end + 3600.0, dict(suffix_interval.fields))
    combined = LateralBoundaries(
        prior_boundaries.intervals + (shifted,),
        prior_boundaries.spec_bdy_width, prior_boundaries.spec_zone,
        prior_boundaries.relax_zone)
    combined_prefix = _forcing_prefix(combined)
    if (old_prefix["intervals"][-1]["end_frame_sha256"]
            != combined_prefix["intervals"][-1]["start_frame_sha256"]):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix changes the shared endpoint frame")

    combined_lbc = _json_copy(prior_meta["lbc"])
    combined_lbc["intervals"].append({
        "start_seconds": old_end,
        "end_seconds": old_end + 3600.0,
        "fields": list(prior_meta["lbc"]["intervals"][-1]["fields"]),
    })
    cache_metadata = _json_copy(prior_meta)
    cache_metadata.update({
        "user": _json_copy(metadata),
        "lbc": combined_lbc,
        "setup_fingerprint": None,
        "forcing_extension_mode": SEALED_PREPARED_EXTENSION_MODE,
        "lateral_boundary_prefix": combined_prefix,
        "bridge_manifest_extension": bridge_extension,
        "source_manifest_extension": manifest_extension,
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _prepared_cache_staging_path(path)
    temporary.mkdir()
    writer = _BundleWriter(temporary)
    try:
        for key in sorted(prior_reader.arrays):
            writer.link_verified(key, prior_reader)
        old_count = len(prior_boundaries.intervals)
        for key in sorted(suffix_reader.arrays):
            if not key.startswith("lbc/0/"):
                continue
            output_key = f"lbc/{old_count}/" + key[len("lbc/0/"):]
            writer.add(output_key, suffix_reader.read_array(key))
        basis = {
            "schema": PREPARED_CACHE_SCHEMA,
            "identity": new_identity,
            "metadata": cache_metadata,
            "arrays": writer.manifest,
            "payload_bytes": writer.payload_bytes,
        }
        header = {
            **basis,
            "status": "READY",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            CACHE_WRITER_KEY: {"gpuwm_version": __version__},
            "content_sha256": hashlib.sha256(
                _canonical(basis).encode("utf-8")).hexdigest(),
        }
        (temporary / _HEADER_NAME).write_text(
            json.dumps(header, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        # The predecessor payloads are hard-linked by design.  Verify the
        # complete staged authority after all links and the new header exist;
        # a concurrent mutation therefore refuses publication rather than
        # blessing bytes that no longer match the inherited manifest.
        PreparedCacheReader(
            temporary, expected_identity=new_identity).verify_all()
        prior_header_after = hashlib.sha256(
            prior_header_path.read_bytes()).hexdigest()
        suffix_header_after = hashlib.sha256(
            suffix_header_path.read_bytes()).hexdigest()
        if (prior_header_after != prior_header_before
                or suffix_header_after != suffix_header_before):
            raise PreparedCacheMismatchError(
                "prepared-cache authority changed during extension")
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema": "gpuwm-prepared-cache-extension-v1",
        "status": "BUILT",
        "path": str(path.resolve()),
        "content_sha256": header["content_sha256"],
        "array_count": len(writer.manifest),
        "payload_bytes": writer.payload_bytes,
        "predecessor": {
            "path": str(predecessor.resolve()),
            "content_sha256": prior_reader.content_sha256,
            "header_sha256": prior_header_before,
        },
        "suffix": {
            "path": str(suffix.resolve()),
            "content_sha256": suffix_reader.content_sha256,
            "header_sha256": suffix_header_before,
        },
        "bridge_manifest_extension": bridge_extension,
        "bridge_manifest_sha256": new_identity["bridge_manifest_sha256"],
        "source_manifest_extension": manifest_extension,
        "appended_interval": [old_end, old_end + 3600.0],
        "sealed_prefix_sha256": hashlib.sha256(
            _canonical(combined_prefix).encode("utf-8")).hexdigest(),
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
        STATE_SERIALIZED_ATTRS, lateral_boundary_prefix_identity,
        setup_core_fingerprint, setup_fingerprint,
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
    if metadata.get("forcing_extension_mode") == \
            SEALED_PREPARED_EXTENSION_MODE:
        if (setup_core_fingerprint(state)
                != metadata.get("setup_core_fingerprint")):
            raise PreparedCacheMismatchError(
                "sealed prepared cache reconstructed a different immutable "
                "setup core")
        if (lateral_boundary_prefix_identity(state)
                != metadata.get("lateral_boundary_prefix")):
            raise PreparedCacheMismatchError(
                "sealed prepared cache reconstructed a different forcing "
                "inventory")
    elif observed_setup != metadata["setup_fingerprint"]:
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
        surface_qv=reader.read_array("result/surface_qv"),
        hydrometeor_initialization=MappingProxyType(
            metadata.get("hydrometeor_initialization", {})))
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
    "DEFAULT_TOLERANT_IDENTITY_FIELDS",
    "INERT_DIAGNOSTIC_IDENTITY_FIELDS",
    "NON_TRAJECTORY_IDENTITY_FIELDS", "PREPARED_CACHE_SCHEMA",
    "PreparedCacheCorruptError", "PreparedCacheMismatchError",
    "PreparedCacheReader", "RestoredPreparedCache",
    "SEALED_PREPARED_EXTENSION_MODE", "UNSTAMPED_WRITER",
    "cache_writer_version", "compare_prepared_domain_config",
    "compare_prepared_identity", "effective_prepared_domain_config",
    "prepared_cache_identity",
    "prepared_domain_config_identity", "prepared_identity_refusal",
    "extend_prepared_cache", "restore_prepared_cache",
    "select_prepared_met_fields", "undelayed_identity_defaults",
    "write_prepared_cache",
]
