"""Small source-grid declarations without materializing weather fields.

Lazy sources expose this same information from the descriptor that their full
snapshot consumer validates. Ordinary materialized sequences keep their existing
validation path. Array digests, finite/range checks and immutable copies remain
at the actual snapshot read; metadata is not a substitute for those checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SourceSnapshotMetadata:
    snapshot_type: type
    latitude: Any
    longitude: Any
    projection: Mapping[str, object] | None = None


def snapshot_metadata(snapshots, index: int) -> SourceSnapshotMetadata:
    read = getattr(snapshots, "snapshot_metadata", None)
    if read is not None:
        result = read(index)
        if not isinstance(result, SourceSnapshotMetadata):
            raise TypeError("source snapshot metadata must use SourceSnapshotMetadata")
        if not isinstance(result.snapshot_type, type):
            raise TypeError("source snapshot metadata must declare a snapshot type")
        return result
    snapshot = snapshots[index]
    return SourceSnapshotMetadata(
        type(snapshot), getattr(snapshot, "latitude", ()),
        getattr(snapshot, "longitude", ()), getattr(snapshot, "projection", None))
