"""Atomic per-domain artifacts for native nested stock-WRF initialization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Mapping, Sequence
import uuid

from gpuwm.ingest.prepared_cache import (
    HEADER_PARTIAL_NAME,
    PreparedCacheReader,
    _prepared_cache_staging_path,
    prepared_cache_identity,
    write_prepared_cache,
)
from gpuwm.ingest.preprocess_backend import (
    preprocess_identity,
    preprocess_reports_identity,
)
from gpuwm.native_wrf_contract import (
    canonical_noah_surface,
    native_static_export_fields,
    write_native_geometry_receipt,
    write_native_static_cache,
)
from gpuwm import fetch_guard, filesystem_paths
from gpuwm.fetch_guard import WINDOWS_WIDEST_PID
from gpuwm.wrf_direct import (
    ROOT_EXPORT_DIRNAME,
    PreparedDomainArtifacts,
    domain_artifacts_manifest_temporary,
    export_staging_path,
    write_domain_artifacts_manifest,
)


@dataclass(frozen=True)
class NativeDomainArtifactBuild:
    """One atomically published domain artifact set and its receipt."""

    artifacts: PreparedDomainArtifacts
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))


@dataclass(frozen=True)
class NativeHierarchyArtifactBuild:
    """Atomically published root/child artifact tree and join manifest."""

    artifacts: tuple[PreparedDomainArtifacts, ...]
    manifest: Path
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str, name: str) -> str:
    normalized = str(value).lower()
    if (len(normalized) != 64
            or any(character not in "0123456789abcdef" for character in normalized)):
        raise ValueError(f"{name} must be a lowercase/uppercase SHA-256 digest")
    return normalized


def _manifest_digest(value: str, name: str) -> str:
    """A manifest digest, or an as-posted head's placeholder for one.

    An as-posted tree's head prepares its children before the input
    manifest exists, so their identities bind the input plan's
    placeholder where the manifest digest goes
    (:func:`gpuwm.ingest.boundary_stream.as_posted_placeholder`); the seal
    writes the digest and refuses any placeholder left in a sealed
    identity (``check_as_posted_identity``).  Anything else is held to a
    SHA-256 digest as before.
    """

    from gpuwm.ingest.boundary_stream import is_as_posted_placeholder

    if is_as_posted_placeholder(value):
        return str(value)
    return _digest(value, name)


def _atomic_staging_sibling(
        output: Path, *, nonce: str | None = None) -> Path:
    """Return a compact, target-independent atomic directory sibling.

    Native hierarchy publication nests multiple create-only directory and
    prepared-cache transactions.  Repeating a user-selected output basename
    in any transaction can push otherwise valid Windows paths beyond the
    legacy visible 259-character ceiling.  The 40-bit lowercase token keeps
    every directory-staging component fixed at 13 characters while preserving
    create-only collision handling in each caller's ``mkdir``.
    """

    token = uuid.uuid4().hex[:10] if nonce is None else nonce
    if (len(token) != 10
            or any(character not in "0123456789abcdef" for character in token)):
        raise ValueError("atomic directory staging nonce must be 10 lowercase hex")
    return Path(output).with_name(f".d-{token}")


def deepest_published_hierarchy_path(
        output: Path, grid_ids: Sequence[int] = (1,)) -> Path:
    """The longest path :func:`write_native_hierarchy_artifacts` publishes.

    Every file the tree holds once it is renamed to ``output``: its
    receipt and manifest, and per domain the receipt, the static cache,
    the geometry receipt and the prepared cache's header and payloads.
    Payloads are named ``aNNNNN.npy`` and every domain folder ``dNN``
    (WRF allows 21 domains), so the list is exact whatever the array
    count.  Every forecast runner and every bundle already on disk reads
    this layout, so a door that publishes it measures this path against
    the host's path limit instead of the layout being shortened.
    """

    output = Path(output)
    candidates = [output / "receipt.json", output / "domain-artifacts.json"]
    for grid_id in grid_ids:
        candidates.extend(_domain_files(
            output / "domains" / f"d{int(grid_id):02d}"))
    return _longest(candidates)


def _receipt_json(value):
    """``value`` in plain JSON types: a preparation's receipt may hold
    read-only mappings, which ``json`` does not serialize."""

    if isinstance(value, Mapping):
        return {str(key): _receipt_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_receipt_json(item) for item in value]
    return value


def _domain_files(domain: Path) -> tuple[Path, ...]:
    """The files :func:`write_native_domain_artifacts` leaves in ``domain``."""

    cache = domain / "prepared-cache"
    return (domain / "receipt.json", domain / "native-static.npz",
            domain / "geometry-receipt.json", cache / "header.json",
            cache / "a00000.npy")


def _longest(paths) -> Path:
    return max(paths, key=lambda path: len(str(path)))


#: The folder below a door's output root that holds its domain tree.
HIERARCHY_ARTIFACTS_DIRNAME = "hierarchy-artifacts"

#: The folder below a door's output root that holds the unchanged-WRF files.
WRF_EXPORT_DIRNAME = "wrf-native-input"


def hierarchy_bundle_write_paths(
        output_root: Path, *, wrf_export: bool = True) -> tuple[Path, ...]:
    """The deepest path of each kind a door writes to publish a domain tree.

    A door that prepares a domain tree (``hrrr_hierarchy_direct``,
    ``gfs_direct``, ``era5_direct`` and ``mapped_direct``) builds its
    whole bundle in a staging sibling of ``output_root`` and publishes it
    with one rename, so a deep root is written at two depths:

    * published: ``hierarchy-artifacts/domains/dNN/prepared-cache/header.json``
      below ``output_root``;
    * staged: the door's sibling (``.d-`` and ten hex characters;
      ``mapped_direct``'s ``.tmp-`` and eight is as wide), holding the
      tree writer's own ``.d-`` staging and its manifest's temporary
      name, each domain's ``.d-`` staging, the ``.p-`` staging of that
      domain's prepared cache with its header written aside
      (``header.json.tmp``), and, when the door writes the
      unchanged-WRF files, ``wrf-native-input.tmp-<pid>/
      .root-export.tmp-<pid>/manifest.json``.

    Process ids are counted at their widest on Windows, ten digits.  The
    staged paths do not depend on the output name, which the sibling
    replaces, so for a short name they are the deepest.  The statics
    corridor set, the source evidence and the other files at the top of
    a bundle sit shallower than these.
    """

    root = Path(output_root).absolute()
    nonce = "f" * 10
    staging = _atomic_staging_sibling(root, nonce=nonce)
    artifacts = staging / HIERARCHY_ARTIFACTS_DIRNAME
    tree_staging = _atomic_staging_sibling(artifacts, nonce=nonce)
    domain_staging = _atomic_staging_sibling(
        tree_staging / "domains" / "d01", nonce=nonce)
    cache_staging = _prepared_cache_staging_path(
        domain_staging / "prepared-cache", nonce=nonce)
    paths = [
        deepest_published_hierarchy_path(root / HIERARCHY_ARTIFACTS_DIRNAME),
        deepest_published_hierarchy_path(artifacts),
        _longest(_domain_files(domain_staging)),
        cache_staging / HEADER_PARTIAL_NAME,
        domain_artifacts_manifest_temporary(
            tree_staging / "domain-artifacts.json",
            pid=WINDOWS_WIDEST_PID, token="f" * 12),
    ]
    if wrf_export:
        export = export_staging_path(
            staging / WRF_EXPORT_DIRNAME, pid=WINDOWS_WIDEST_PID)
        paths.append(export_staging_path(
            export / ROOT_EXPORT_DIRNAME, pid=WINDOWS_WIDEST_PID)
            / "manifest.json")
    return tuple(paths)


def published_path_refusal(
        output_root: Path, *, wrf_export: bool = True,
        also: Sequence[Path] = ()) -> str | None:
    """Why a domain tree prepared at ``output_root`` would break here.

    The one check every door that prepares a domain tree makes, right
    after it knows it will: the deepest of
    :func:`hierarchy_bundle_write_paths` and of ``also`` (other paths the
    preparation writes because of where ``output_root`` is) against
    Windows' path limit.  The breakage it prevents: past the limit
    Windows reports the file missing, so a staged path fails the
    preparation partway through, and a published one lets it run to the
    end and publish, after which the forecast cannot open its own inputs.
    A 125-character folder holding a 92-character output name put
    ``hierarchy-artifacts/domains/d01/prepared-cache/header.json`` at 277
    characters.

    None when the limit does not bind: off Windows, or where the machine
    has set LongPathsEnabled to 1 (read by
    :func:`gpuwm.fetch_guard.windows_path_limit`), or when
    ``output_root`` is in the extended ``\\\\?\\`` spelling, which opens at
    any length and is what ``gpuwm go`` and ``gpuwm run-plan`` hand a
    stage for a deep run folder
    (:func:`gpuwm.filesystem_paths.deep_io_path`).

    A limit of the measure: an exporter that refuses the domain's physics
    before it writes (``stock_wrf_export`` optional) is counted as if it
    wrote its staging; that path is eight characters deeper than the
    next and the deepest only for output names under 28 characters.
    """

    if filesystem_paths.is_extended(output_root):
        return None
    limit = fetch_guard.windows_path_limit()
    if limit is None:
        return None
    root = Path(output_root).absolute()
    paths = (*hierarchy_bundle_write_paths(root, wrf_export=wrf_export),
             *(Path(path) for path in also))
    deepest = _longest(paths)
    length = len(str(deepest))
    if length <= limit:
        return None

    def over(inside: bool) -> int:
        lengths = [len(str(path)) for path in paths
                   if (root in path.parents) is inside]
        return max(0, max(lengths, default=0) - limit)

    def characters(count: int) -> str:
        return f"{count} character{'' if count == 1 else 's'}"

    in_root, in_folder = over(True), over(False)
    if in_folder == 0:
        remedy = f"an --output-root at least {characters(in_root)} shorter"
    else:
        remedy = (f"an --output-root in a folder at least "
                  f"{characters(in_folder)} shorter")
        if in_root > in_folder:
            remedy += (f", and at least {characters(in_root)} shorter "
                       "as a whole")
    return (
        f"refusing output root {root}: the deepest file the preparation "
        f"writes would have a path of {length} characters, and Windows "
        f"refuses paths longer than {limit} characters unless long paths "
        "are enabled, which they are not on this computer.  Windows would "
        "report that file missing: a staged file stops the preparation "
        "partway, and a published one lets it finish and then stops the "
        f"forecast from opening its own inputs.  Use {remedy}, or, as "
        "administrator, set LongPathsEnabled to 1 (DWORD) under "
        "HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem and "
        f"restart.  Longest path: {deepest}")


def _forcing_hours(values: Sequence[int]) -> tuple[int, ...]:
    hours = tuple(values)
    if (len(hours) < 2 or hours[0] != 0
            or any(isinstance(hour, bool) or not isinstance(hour, int)
                   for hour in hours)
            or any(later <= earlier
                   for earlier, later in zip(hours, hours[1:]))):
        raise ValueError(
            "forcing_hours must contain increasing integer hours beginning "
            "at zero and include at least one boundary interval")
    return hours


def _forcing_offsets(
        *, forcing_hours: Sequence[int] | None,
        forcing_offsets_seconds: Sequence[int] | None,
) -> tuple[int, ...]:
    if (forcing_hours is None) == (forcing_offsets_seconds is None):
        raise ValueError(
            "exactly one of forcing_hours or forcing_offsets_seconds is "
            "required")
    if forcing_hours is not None:
        return tuple(hour * 3600 for hour in _forcing_hours(forcing_hours))
    offsets = tuple(forcing_offsets_seconds)
    if (len(offsets) < 2 or offsets[0] != 0
            or any(isinstance(offset, bool) or not isinstance(offset, int)
                   for offset in offsets)
            or any(later <= earlier
                   for earlier, later in zip(offsets, offsets[1:]))):
        raise ValueError(
            "forcing_offsets_seconds must contain increasing integer "
            "seconds beginning at zero and include at least one boundary "
            "interval")
    return offsets


def _domain_valid_time(exp, domain, root_valid_time: datetime) -> datetime:
    configured = getattr(domain, "start_time", None)
    if configured is None:
        return root_valid_time
    return configured


def _validate_domain_boundary_mode(domain, boundaries) -> None:
    root = int(domain.parent_id) == 0
    if root:
        if not domain.run.specified or domain.run.nested:
            raise ValueError(
                "root artifact requires specified=true and nested=false")
        if boundaries is None:
            raise ValueError("root artifact requires external lateral boundaries")
    else:
        if domain.run.specified or not domain.run.nested:
            raise ValueError(
                "child artifact requires specified=false and nested=true")
        if boundaries is not None:
            raise ValueError(
                "child artifact must omit external LBCs; stock WRF forces it "
                "from its declared parent")


@dataclass(frozen=True)
class DomainArtifactBinding:
    """What a domain's prepared cache is bound to, before any array exists.

    The identity and the user metadata :func:`write_native_domain_artifacts`
    writes into that domain's ``prepared-cache/header.json``, computed by
    one function (:func:`domain_artifact_binding`) so a streamed root cache
    in a chained tree's head and the one-shot root cache the seal writes
    carry the same header for the same inputs.
    """

    identity: Mapping[str, object]
    metadata: Mapping[str, object]
    forcing_identity: Mapping[str, object]


def write_domain_static_files(
        directory: Path, *, domain, grid,
        static_fields: Mapping[str, object]) -> tuple[dict, dict]:
    """Write ``native-static.npz`` and ``geometry-receipt.json`` for a domain.

    Returns ``(static_receipt, geometry_receipt)``; the same two writes every
    domain artifact set starts with.
    """

    directory = Path(directory)
    static_path = directory / "native-static.npz"
    geometry_path = directory / "geometry-receipt.json"
    export_static = native_static_export_fields(static_fields, grid)
    static_receipt = write_native_static_cache(static_path, export_static)
    from gpuwm.static.terrain_autosmooth import bind_receipt
    static_receipt = bind_receipt(static_fields, static_receipt)
    geometry_receipt = write_native_geometry_receipt(
        geometry_path, grid, domain.run, static_path,
        **({"terrain_autosmooth": static_receipt["terrain_autosmooth"]}
           if "terrain_autosmooth" in static_receipt else {}))
    return static_receipt, geometry_receipt


def domain_artifact_binding(
        *, domain, static_cache_sha256: str,
        bridge_manifest_sha256: str, source_manifest_sha256: str,
        namelist_sha256: str, forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
        source_identity: Mapping[str, object], valid_time: datetime,
        forcing_origin_time: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
) -> DomainArtifactBinding:
    """The prepared-cache identity and metadata of one domain artifact set."""

    if not isinstance(valid_time, datetime):
        raise TypeError("valid_time must be a datetime")
    offsets = _forcing_offsets(
        forcing_hours=forcing_hours,
        forcing_offsets_seconds=forcing_offsets_seconds)
    legacy_hours = (
        _forcing_hours(forcing_hours)
        if forcing_hours is not None else None)
    forcing_identity = (
        {"forcing_hours": legacy_hours}
        if legacy_hours is not None else
        {"forcing_offsets_seconds": offsets})
    if forcing_origin_time is None:
        forcing_origin_time = valid_time
    if not isinstance(forcing_origin_time, datetime):
        raise TypeError("forcing_origin_time must be a datetime")
    digests = {
        "bridge_manifest_sha256": _manifest_digest(
            bridge_manifest_sha256, "bridge_manifest_sha256"),
        "source_manifest_sha256": _manifest_digest(
            source_manifest_sha256, "source_manifest_sha256"),
        "namelist_sha256": _digest(namelist_sha256, "namelist_sha256"),
    }
    identity = prepared_cache_identity(
        bridge_manifest_sha256=digests["bridge_manifest_sha256"],
        source_manifest_sha256=digests["source_manifest_sha256"],
        static_cache_sha256=static_cache_sha256,
        namelist_sha256=digests["namelist_sha256"],
        domain_config=domain, **forcing_identity,
        source_identity=dict(source_identity),
    )
    user_metadata = dict(metadata or {})
    reserved = {
        "initial_valid_time", "last_valid_time", "forcing_hours",
        "forcing_offsets_seconds"}
    conflict = reserved & set(user_metadata)
    if conflict:
        raise ValueError(
            f"domain artifact metadata overrides reserved keys {sorted(conflict)}")
    user_metadata.update({
        "initial_valid_time": valid_time.isoformat(),
        "last_valid_time": (
            forcing_origin_time
            + timedelta(seconds=offsets[-1])).isoformat(),
        **{
            key: list(values)
            for key, values in forcing_identity.items()
        },
    })
    return DomainArtifactBinding(
        identity=identity, metadata=user_metadata,
        forcing_identity=forcing_identity)


def write_native_domain_artifacts(
        output: Path, *, domain, grid, initial_result, met, soil,
        static_fields: Mapping[str, object], boundaries,
        bridge_manifest_sha256: str, source_manifest_sha256: str,
        namelist_sha256: str, forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
        source_identity: Mapping[str, object], valid_time: datetime,
        forcing_origin_time: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        input_preparation_seconds: float | None = None,
        preprocess_receipt: Mapping[str, object] | None = None,
) -> NativeDomainArtifactBuild:
    """Build one root/child artifact set in an atomic directory.

    The root must carry complete external LBCs.  A child must carry none: its
    identity is explicitly nested and the unchanged stock-WRF runtime forces
    it from its parent.  Every set contains a prepared cache, a regenerated
    static cache, and the geometry/static receipt consumed by the M1 hierarchy
    exporter.

    ``input_preparation_seconds``, the wall time the domain's inputs took,
    is recorded in ``receipt.json`` beside the cache, never in the cache's
    hashed metadata: a timing there made two preparations of the same
    inputs differ in content digest (A138).  ``preprocess_receipt``, the
    domain's whole preprocessing receipt with what its preparation
    measured, is recorded there too, for a child whose cache binds only
    its identity.
    """

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite domain artifacts {output}")
    if not isinstance(valid_time, datetime):
        raise TypeError("valid_time must be a datetime")
    if initial_result is None or soil is None:
        raise ValueError("domain artifacts require initialized state and Noah soil")
    _validate_domain_boundary_mode(domain, boundaries)
    attached_boundaries = getattr(
        initial_result.state, "lateral_boundaries", None)
    if boundaries is None:
        if attached_boundaries is not None:
            raise ValueError(
                "nested child state unexpectedly carries external LBCs")
    elif attached_boundaries is None:
        raise ValueError(
            "root state must already carry the supplied external LBCs")
    elif attached_boundaries is not boundaries:
        raise ValueError(
            "root state carries different external LBCs than the artifact "
            "writer received")
    # Refused before any file is written, exactly as before the binding
    # was factored out: a bad digest or forcing axis creates nothing.
    _forcing_offsets(forcing_hours=forcing_hours,
                     forcing_offsets_seconds=forcing_offsets_seconds)
    for value, name in ((bridge_manifest_sha256, "bridge_manifest_sha256"),
                        (source_manifest_sha256, "source_manifest_sha256")):
        _manifest_digest(value, name)
    _digest(namelist_sha256, "namelist_sha256")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _atomic_staging_sibling(output)
    staging.mkdir()
    try:
        geometry_path = staging / "geometry-receipt.json"
        prepared_path = staging / "prepared-cache"
        static_receipt, geometry_receipt = write_domain_static_files(
            staging, domain=domain, grid=grid, static_fields=static_fields)
        binding = domain_artifact_binding(
            domain=domain, static_cache_sha256=static_receipt["sha256"],
            bridge_manifest_sha256=bridge_manifest_sha256,
            source_manifest_sha256=source_manifest_sha256,
            namelist_sha256=namelist_sha256, forcing_hours=forcing_hours,
            forcing_offsets_seconds=forcing_offsets_seconds,
            source_identity=source_identity, valid_time=valid_time,
            forcing_origin_time=forcing_origin_time, metadata=metadata)
        identity = dict(binding.identity)
        user_metadata = dict(binding.metadata)
        forcing_identity = dict(binding.forcing_identity)
        prepared_receipt = write_prepared_cache(
            prepared_path, identity=identity,
            initial_result=initial_result, met=met,
            boundaries=boundaries, surface=canonical_noah_surface(soil),
            metadata=user_metadata,
        )
        verified = dict(PreparedCacheReader(
            prepared_path, expected_identity=identity).verify_all())
        # The artifact set is relocatable and may itself be installed by an
        # outer atomic hierarchy rename.  Never seal the transient staging
        # directory into its durable receipt.
        verified["path"] = prepared_path.name
        receipt = {
            "schema": "gpuwm-native-domain-artifact-build-v1",
            "status": "READY",
            "grid_id": int(domain.grid_id),
            "parent_id": int(domain.parent_id),
            "boundary_mode": (
                "external-specified" if int(domain.parent_id) == 0
                else "nested-parent-forced"),
            "valid_time": valid_time.isoformat(),
            **{
                key: list(values)
                for key, values in forcing_identity.items()
            },
            "artifacts": {
                "prepared_cache": {
                    "path": prepared_path.name,
                    "content_sha256": prepared_receipt["content_sha256"],
                    "payload_bytes": prepared_receipt["payload_bytes"],
                    "array_count": prepared_receipt["array_count"],
                },
                "static_cache": static_receipt,
                "geometry_receipt": {
                    "path": geometry_path.name,
                    "sha256": _sha256(geometry_path),
                    "geometry": geometry_receipt["geometry"],
                },
            },
            "verification": verified,
            **({} if input_preparation_seconds is None else {
                "input_preparation_seconds": float(
                    input_preparation_seconds)}),
            **({} if preprocess_receipt is None else {
                "preprocess_receipt": _receipt_json(preprocess_receipt)}),
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return NativeDomainArtifactBuild(
        artifacts=PreparedDomainArtifacts(
            grid_id=int(domain.grid_id),
            prepared_cache=output / "prepared-cache",
            static_cache=output / "native-static.npz",
            geometry_receipt=output / "geometry-receipt.json",
        ),
        receipt=receipt,
    )


def _require_complete_children(exp, child_results) -> tuple:
    domains = tuple(exp.domains)
    children = tuple(child_results)
    if len(children) != len(domains) - 1:
        raise ValueError(
            "child result count does not match the experiment hierarchy")
    for domain, result in zip(domains[1:], children):
        if getattr(result, "domain", None) != domain:
            raise ValueError(
                f"child result identity does not match d{domain.grid_id:02d}")
        if result.real is None or result.static_fields is None \
                or result.horizontal is None or result.soil is None:
            raise ValueError(
                f"d{domain.grid_id:02d} is not a complete real-data child")
        prepared_domain = getattr(result.real.state, "lateral_boundaries", None)
        if prepared_domain is not None:
            raise ValueError(
                f"d{domain.grid_id:02d} unexpectedly carries external LBCs")
    return children


def write_child_domain_artifacts(
        domain_root: Path, *, exp, child_results: Sequence[object],
        bridge_manifest_sha256: str, source_manifest_sha256: str,
        namelist_sha256: str, forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
        source_identity: Mapping[str, object], valid_time: datetime,
) -> list[NativeDomainArtifactBuild]:
    """Write every child's artifact set under ``domain_root/dNN``.

    The one writer of a child's artifacts, for the one-shot tree and for a
    chained tree's head (``hierarchy-head/domains/dNN``): a child needs
    only the start time, so its set is complete before any root boundary
    interval exists.
    """

    children = _require_complete_children(exp, child_results)
    builds: list[NativeDomainArtifactBuild] = []
    for domain, result in zip(tuple(exp.domains)[1:], children):
        # What the child's preparation ran, without what it measured
        # (A138): the root's device fit rides in its selection and a
        # native HRRR child's soil mapping carries its backend's
        # receipt.  The child's receipt.json keeps the whole receipt,
        # since a child resolves its own backend and worker count.
        metadata = {
            "preprocess_receipt": preprocess_reports_identity(
                preprocess_identity(
                    dict(result.preprocess_receipt or {}))),
        }
        builds.append(write_native_domain_artifacts(
            Path(domain_root) / f"d{int(domain.grid_id):02d}",
            domain=domain, grid=result.grid,
            initial_result=result.real, met=result.horizontal,
            soil=result.soil, static_fields=result.static_fields,
            boundaries=None,
            bridge_manifest_sha256=bridge_manifest_sha256,
            source_manifest_sha256=source_manifest_sha256,
            namelist_sha256=namelist_sha256,
            forcing_hours=forcing_hours,
            forcing_offsets_seconds=forcing_offsets_seconds,
            source_identity={
                **dict(source_identity), "grid_id": int(domain.grid_id)},
            valid_time=_domain_valid_time(exp, domain, valid_time),
            forcing_origin_time=valid_time, metadata=metadata,
            input_preparation_seconds=result.input_preparation_seconds,
            preprocess_receipt=result.preprocess_receipt))
    return builds


def root_domain_artifact_binding(
        *, exp, static_cache_sha256: str, bridge_manifest_sha256: str,
        source_manifest_sha256: str, namelist_sha256: str,
        forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
        source_identity: Mapping[str, object], valid_time: datetime,
        root_metadata: Mapping[str, object] | None = None,
) -> DomainArtifactBinding:
    """d01's binding exactly as :func:`write_native_hierarchy_artifacts` makes it."""

    return domain_artifact_binding(
        domain=tuple(exp.domains)[0], static_cache_sha256=static_cache_sha256,
        bridge_manifest_sha256=bridge_manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
        namelist_sha256=namelist_sha256, forcing_hours=forcing_hours,
        forcing_offsets_seconds=forcing_offsets_seconds,
        source_identity={**dict(source_identity), "grid_id": 1},
        valid_time=valid_time, forcing_origin_time=valid_time,
        metadata=root_metadata)


def write_native_hierarchy_artifacts(
        output: Path, *, exp, root_grid, root_initial_result, root_met,
        root_soil, root_static_fields: Mapping[str, object], root_boundaries,
        child_results: Sequence[object],
        bridge_manifest_sha256: str, source_manifest_sha256: str,
        namelist_sha256: str, forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
        source_identity: Mapping[str, object], valid_time: datetime,
        root_metadata: Mapping[str, object] | None = None,
) -> NativeHierarchyArtifactBuild:
    """Join a prepared root and streamed child results into one atomic tree.

    The root owns the only external boundary sequence.  Each child result must
    be the WRF-order product of ``finalize_prepared_child`` and is serialized
    without external LBCs.  A relocatable manifest is written only after every
    prepared cache has been reread and verified by the per-domain writer.
    """

    output = Path(output)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite native hierarchy artifacts {output}")
    domains = tuple(exp.domains)
    if not domains or int(domains[0].parent_id) != 0:
        raise ValueError("experiment must begin with exactly one root domain")
    children = _require_complete_children(exp, child_results)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _atomic_staging_sibling(output)
    staging.mkdir()
    try:
        domain_root = staging / "domains"
        domain_root.mkdir()
        builds: list[NativeDomainArtifactBuild] = []
        builds.append(write_native_domain_artifacts(
            domain_root / "d01", domain=domains[0], grid=root_grid,
            initial_result=root_initial_result, met=root_met, soil=root_soil,
            static_fields=root_static_fields, boundaries=root_boundaries,
            bridge_manifest_sha256=bridge_manifest_sha256,
            source_manifest_sha256=source_manifest_sha256,
            namelist_sha256=namelist_sha256, forcing_hours=forcing_hours,
            forcing_offsets_seconds=forcing_offsets_seconds,
            source_identity={**dict(source_identity), "grid_id": 1},
            valid_time=valid_time, forcing_origin_time=valid_time,
            metadata=root_metadata))
        builds.extend(write_child_domain_artifacts(
            domain_root, exp=exp, child_results=children,
            bridge_manifest_sha256=bridge_manifest_sha256,
            source_manifest_sha256=source_manifest_sha256,
            namelist_sha256=namelist_sha256, forcing_hours=forcing_hours,
            forcing_offsets_seconds=forcing_offsets_seconds,
            source_identity=source_identity, valid_time=valid_time))
        manifest = staging / "domain-artifacts.json"
        write_domain_artifacts_manifest(
            manifest, tuple(build.artifacts for build in builds))
        receipt = {
            "schema": "gpuwm-native-hierarchy-artifact-build-v1",
            "status": "READY",
            "domain_count": len(builds),
            "grid_ids": [build.artifacts.grid_id for build in builds],
            "manifest": {
                "path": manifest.name,
                "sha256": _sha256(manifest),
            },
            "boundary_inventory": {
                "external": [1],
                "nested_parent_forced": [
                    int(domain.grid_id) for domain in domains[1:]],
            },
            "domains": [dict(build.receipt) for build in builds],
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    artifacts = tuple(PreparedDomainArtifacts(
        grid_id=int(domain.grid_id),
        prepared_cache=(output / "domains" /
                        f"d{int(domain.grid_id):02d}" / "prepared-cache"),
        static_cache=(output / "domains" /
                      f"d{int(domain.grid_id):02d}" / "native-static.npz"),
        geometry_receipt=(output / "domains" /
                          f"d{int(domain.grid_id):02d}" /
                          "geometry-receipt.json"),
    ) for domain in domains)
    return NativeHierarchyArtifactBuild(
        artifacts=artifacts,
        manifest=output / "domain-artifacts.json",
        receipt=receipt,
    )


__all__ = [
    "DomainArtifactBinding",
    "NativeDomainArtifactBuild",
    "NativeHierarchyArtifactBuild",
    "HIERARCHY_ARTIFACTS_DIRNAME",
    "WRF_EXPORT_DIRNAME",
    "deepest_published_hierarchy_path",
    "domain_artifact_binding",
    "hierarchy_bundle_write_paths",
    "published_path_refusal",
    "root_domain_artifact_binding",
    "write_child_domain_artifacts",
    "write_domain_static_files",
    "write_native_domain_artifacts",
    "write_native_hierarchy_artifacts",
]
