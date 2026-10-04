"""Verified source trajectories and member-specific prepared timeline owners."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from gpuwm.ensemble.recipes import SourceRecipe, SourceTrajectory


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("prepared member valid times need an explicit UTC offset")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class SourceManifestBinding:
    """An immutable native verification receipt, bound to actual file bytes."""
    trajectory: SourceTrajectory
    path: Path
    sha256: str
    verification: dict

    def verify(self):
        path = Path(self.path).resolve(strict=True)
        if not path.is_file() or _sha(path) != self.sha256:
            raise ValueError("member source manifest changed or does not match its bound bytes")
        expected = self.trajectory
        receipt = self.verification
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("source") != expected.source
                or manifest.get("member") != expected.member
                or _utc(datetime.fromisoformat(str(manifest.get("cycle", "")).replace("Z", "+00:00"))) != expected.cycle):
            raise ValueError("source manifest bytes identify another source trajectory")
        if (receipt.get("source") != expected.source
                or receipt.get("member") != expected.member
                or _utc(datetime.fromisoformat(str(receipt.get("cycle", "")).replace("Z", "+00:00"))) != expected.cycle):
            raise ValueError("native source verification belongs to another source, cycle or member")
        if receipt.get("manifest_sha256") != self.sha256:
            raise ValueError("native source verification is not bound to this manifest")
        if _sha(path) != self.sha256:
            raise ValueError("member source manifest changed during verification")
        return {"trajectory_sha256": expected.identity, "source": expected.source,
                "cycle": expected.cycle.isoformat(), "member": expected.member,
                "manifest": str(path), "manifest_sha256": self.sha256,
                "verification": dict(receipt)}


@dataclass(frozen=True)
class PreparedMemberInput:
    """The executor consumes this member's initialization and full boundaries."""
    member_id: int
    seed: int
    inputs: object
    initial: object
    boundaries: object
    trajectory: SourceTrajectory
    source_manifests: tuple[SourceManifestBinding, ...]
    boundary_valid_times: tuple[datetime, ...]
    geometry_sha256: str
    recipe_sha256: str
    donor_manifests: tuple[SourceManifestBinding, ...] = ()
    preparation_receipt: dict | None = None

    def verify(self, recipe, *, shared_geometry_sha256):
        selected = {member.index: member for member in recipe.members}
        member = selected.get(self.member_id)
        if (member is None or member.seed != self.seed
                or member.trajectory != self.trajectory or self.recipe_sha256 != recipe.sha256):
            raise ValueError("prepared member lost its recipe identity, original index or seed")
        if self.geometry_sha256 != shared_geometry_sha256:
            raise ValueError("prepared member uses another native grid or coordinate authority")
        times = tuple(_utc(time) for time in self.boundary_valid_times)
        if (not times or times[0] != recipe.start or times[-1] != recipe.end
                or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError("member boundary timeline does not cover the exact common valid-time window")
        recipe.acquisition_window(self.trajectory)
        manifests = tuple(binding.verify() for binding in self.source_manifests)
        preparation_source = recipe.base if recipe.kind == "recentered" else self.trajectory
        if not manifests or any(binding.trajectory != preparation_source for binding in self.source_manifests):
            raise ValueError("initial and boundary inputs must retain the same source member and cycle")
        donors = tuple(binding.verify() for binding in self.donor_manifests)
        if recipe.kind == "recentered":
            population = tuple(sorted(source.identity for source in recipe.donor_population))
            received = tuple(sorted({binding.trajectory.identity for binding in self.donor_manifests}))
            if received != population:
                raise ValueError("recentered preparation must bind the complete fixed donor population")
        elif donors:
            raise ValueError("donor receipts were supplied for a recipe without recentering")
        return {"member_id": self.member_id, "seed": self.seed,
                "recipe_sha256": self.recipe_sha256, "geometry_sha256": self.geometry_sha256,
                "trajectory_sha256": self.trajectory.identity,
                "initialization_source_sha256": preparation_source.identity,
                "boundary_valid_times": [time.isoformat() for time in times],
                "source_manifests": list(manifests), "donor_manifests": list(donors),
                "preparation": self.preparation_receipt}


class PreparedMemberRoster:
    """Stable global IDs, including sparse individual-member replay."""
    def __init__(self, recipe: SourceRecipe, members, *, shared_geometry_sha256):
        self.recipe, self.members = recipe, tuple(members)
        if tuple(member.member_id for member in self.members) != tuple(member.index for member in recipe.members):
            raise ValueError("prepared members must follow the original selected recipe order")
        self.geometry_sha256 = shared_geometry_sha256
        self.receipts = tuple(member.verify(recipe, shared_geometry_sha256=shared_geometry_sha256)
                              for member in self.members)
        self._by_id = {member.member_id: member for member in self.members}

    def select(self, member_ids):
        ids = tuple(member_ids)
        if len(set(ids)) != len(ids) or any(member not in self._by_id for member in ids):
            raise ValueError("packing requested a member outside its original source roster")
        return tuple(self._by_id[member] for member in ids)

    def receipt(self):
        return {"contract": "gpuwm-ensemble-prepared-member-inputs.v1",
                "recipe": self.recipe.describe(), "recipe_sha256": self.recipe.sha256,
                "member_order": [member.member_id for member in self.members],
                "shared_geometry_sha256": self.geometry_sha256, "members": list(self.receipts),
                "initialization_policy": "one native initialization per source member; no member-mean trajectory",
                "base_sharing_policy": "byte equality checked by the batch allocation authority"}


def prepare_member_roster(recipe, *, shared_geometry, geometry_sha256, native_preparer):
    """Drive the native preparer with whole trajectories, never singleton copies.

    ``native_preparer`` implements the original source adapter and initializer.
    Recentered recipes pass the entire canonical donor population at every
    knot; source preparation must verify units, levels and valid-time fields
    before invoking the recentering primitive and ordinary real initialization.
    """
    donors = tuple(sorted(recipe.donor_population, key=lambda item: item.identity))
    prepared = tuple(native_preparer.prepare_member(
        member=member, recipe=recipe, shared_geometry=shared_geometry,
        geometry_sha256=geometry_sha256, donor_population=donors)
        for member in recipe.members)
    return PreparedMemberRoster(recipe, prepared, shared_geometry_sha256=geometry_sha256)
