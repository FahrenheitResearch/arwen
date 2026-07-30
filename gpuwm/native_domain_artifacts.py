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
    PreparedCacheReader,
    prepared_cache_identity,
    write_prepared_cache,
)
from gpuwm.native_wrf_contract import (
    canonical_noah_surface,
    native_static_export_fields,
    write_native_geometry_receipt,
    write_native_static_cache,
)
from gpuwm.wrf_direct import (
    PreparedDomainArtifacts,
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


def write_native_domain_artifacts(
        output: Path, *, domain, grid, initial_result, met, soil,
        static_fields: Mapping[str, object], boundaries,
        bridge_manifest_sha256: str, source_manifest_sha256: str,
        namelist_sha256: str, forcing_hours: Sequence[int],
        source_identity: Mapping[str, object], valid_time: datetime,
        metadata: Mapping[str, object] | None = None,
) -> NativeDomainArtifactBuild:
    """Build one root/child artifact set in an atomic directory.

    The root must carry complete external LBCs.  A child must carry none: its
    identity is explicitly nested and the unchanged stock-WRF runtime forces
    it from its parent.  Every set contains a prepared cache, a regenerated
    static cache, and the geometry/static receipt consumed by the M1 hierarchy
    exporter.
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
    hours = _forcing_hours(forcing_hours)
    digests = {
        "bridge_manifest_sha256": _digest(
            bridge_manifest_sha256, "bridge_manifest_sha256"),
        "source_manifest_sha256": _digest(
            source_manifest_sha256, "source_manifest_sha256"),
        "namelist_sha256": _digest(namelist_sha256, "namelist_sha256"),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _atomic_staging_sibling(output)
    staging.mkdir()
    try:
        static_path = staging / "native-static.npz"
        geometry_path = staging / "geometry-receipt.json"
        prepared_path = staging / "prepared-cache"
        export_static = native_static_export_fields(static_fields, grid)
        static_receipt = write_native_static_cache(static_path, export_static)
        geometry_receipt = write_native_geometry_receipt(
            geometry_path, grid, domain.run, static_path)
        identity = prepared_cache_identity(
            bridge_manifest_sha256=digests["bridge_manifest_sha256"],
            source_manifest_sha256=digests["source_manifest_sha256"],
            static_cache_sha256=static_receipt["sha256"],
            namelist_sha256=digests["namelist_sha256"],
            domain_config=domain, forcing_hours=hours,
            source_identity=dict(source_identity),
        )
        user_metadata = dict(metadata or {})
        reserved = {"initial_valid_time", "last_valid_time", "forcing_hours"}
        conflict = reserved & set(user_metadata)
        if conflict:
            raise ValueError(
                f"domain artifact metadata overrides reserved keys {sorted(conflict)}")
        user_metadata.update({
            "initial_valid_time": valid_time.isoformat(),
            "last_valid_time": (
                valid_time + timedelta(hours=hours[-1])).isoformat(),
            "forcing_hours": list(hours),
        })
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
            "forcing_hours": list(hours),
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


def write_native_hierarchy_artifacts(
        output: Path, *, exp, root_grid, root_initial_result, root_met,
        root_soil, root_static_fields: Mapping[str, object], root_boundaries,
        child_results: Sequence[object],
        bridge_manifest_sha256: str, source_manifest_sha256: str,
        namelist_sha256: str, forcing_hours: Sequence[int],
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
    children = tuple(child_results)
    if not domains or int(domains[0].parent_id) != 0:
        raise ValueError("experiment must begin with exactly one root domain")
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
            source_identity={**dict(source_identity), "grid_id": 1},
            valid_time=valid_time, metadata=root_metadata))
        for domain, result in zip(domains[1:], children):
            metadata = {
                "input_preparation_seconds": result.input_preparation_seconds,
                "preprocess_receipt": dict(result.preprocess_receipt or {}),
            }
            builds.append(write_native_domain_artifacts(
                domain_root / f"d{int(domain.grid_id):02d}",
                domain=domain, grid=result.grid,
                initial_result=result.real, met=result.horizontal,
                soil=result.soil, static_fields=result.static_fields,
                boundaries=None,
                bridge_manifest_sha256=bridge_manifest_sha256,
                source_manifest_sha256=source_manifest_sha256,
                namelist_sha256=namelist_sha256,
                forcing_hours=forcing_hours,
                source_identity={
                    **dict(source_identity), "grid_id": int(domain.grid_id)},
                valid_time=valid_time, metadata=metadata))
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
    "NativeDomainArtifactBuild",
    "NativeHierarchyArtifactBuild",
    "write_native_domain_artifacts",
    "write_native_hierarchy_artifacts",
]
