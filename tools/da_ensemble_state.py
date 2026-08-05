"""The cycling ensemble, written down between processes.

:mod:`tools.da_cycle_prepared` already carries its ensemble from one leg
to the next as host snapshots of
:data:`gpuwm.state_serialization_contract.STATE_SERIALIZED_ATTRS` plus
the not-yet-applied analysis increments.  That state lives for exactly
as long as the process does, which is why a run has always had to
declare every cycle up front.

A continuous nowcast cannot: the next observation does not exist yet
when this one is assimilated.  This module is the leg boundary written
to disk, so the SAME state that survives a leg inside one process
survives the gap between two.  Nothing here is a second ensemble
representation -- ``snapshots`` and ``pending`` are the driver's own two
dictionaries, in the driver's own ``np.savez`` form.

**What a generation holds.**  For every trajectory (the never-analysed
control and each member): its post-leg snapshot, and the increments the
analysis at that leg produced and the NEXT leg has still to apply.  Both
are required.  A generation with the snapshot alone would silently drop
one cycle's analysis -- the model would advance from a background nobody
corrected, and no error would be raised anywhere.

**Identity is checked, not assumed.**  A generation is only meaningful
against the same prepared case, physics profile, grid, timestep and
ensemble size that produced it.  :func:`validate_resume` compares every
one of those and refuses with the full list of what differs, because
restoring a 49-level snapshot into a 51-level state is the kind of
mistake that produces plausible-looking output.

**Slots, not an ever-growing pile.**  Generations are written into a
small ring of slot directories (see :func:`slot_dir`) so a daemon
cycling for hours overwrites its own working state instead of
accumulating one full ensemble copy every few minutes.  Two slots is the
minimum that is crash-safe: the generation being written is never the
generation being resumed from.

Writes are atomic per file: a reader that catches a half-written
generation would restore a torn state, and the manifest is written LAST,
so a generation without a readable manifest is a generation that was
never finished.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: The generation manifest's contract string.
SCHEMA = "gpuwm-da.cycle-ensemble.v1"

#: Name of the never-analysed trajectory, matching the driver's own.
CONTROL = "control"

#: Minimum slot ring: one being written, one being read.
MIN_SLOTS = 2


class EnsembleStateError(RuntimeError):
    """A generation cannot be trusted for the run in hand."""


@dataclass(frozen=True)
class EnsembleIdentity:
    """What a generation must agree with to be restorable.

    Every field is something that changes the MEANING of the arrays in
    the generation, not merely the run around them: the ensemble size
    fixes how many trajectories exist, the grid fixes their shape, the
    timestep fixes what a leg boundary is, and the prepared-content
    digest fixes the case they were integrated on.
    """

    members: int
    nx: int
    ny: int
    nz: int
    dt_s: float
    mp_physics: int
    physics_profile: str
    prepared_content_sha256: str

    def to_payload(self) -> dict:
        return {
            "members": int(self.members),
            "nx": int(self.nx), "ny": int(self.ny), "nz": int(self.nz),
            "dt_s": float(self.dt_s),
            "mp_physics": int(self.mp_physics),
            "physics_profile": str(self.physics_profile),
            "prepared_content_sha256": str(
                self.prepared_content_sha256),
        }

    @classmethod
    def from_payload(cls, payload) -> "EnsembleIdentity":
        try:
            return cls(
                members=int(payload["members"]),
                nx=int(payload["nx"]), ny=int(payload["ny"]),
                nz=int(payload["nz"]), dt_s=float(payload["dt_s"]),
                mp_physics=int(payload["mp_physics"]),
                physics_profile=str(payload["physics_profile"]),
                prepared_content_sha256=str(
                    payload["prepared_content_sha256"]))
        except KeyError as missing:
            raise EnsembleStateError(
                f"ensemble identity is missing {missing}") from None


def trajectory_key(name) -> str:
    """One filesystem-safe key per trajectory, control included."""

    if name == CONTROL:
        return CONTROL
    index = int(name)
    if index < 0:
        raise EnsembleStateError(f"member index {index} is negative")
    return f"m{index:03d}"


def trajectory_names(members: int):
    """The driver's own trajectory list: control, then each member."""

    if members < 1:
        raise EnsembleStateError("an ensemble needs at least one member")
    return [CONTROL, *range(members)]


def slot_dir(root: Path, generation: int, *, slots: int = MIN_SLOTS
             ) -> Path:
    """Where generation ``generation`` is written in a ``slots`` ring.

    The slot index is the generation modulo the ring size, so a resume
    always reads a directory the current write is not touching.
    """

    if slots < MIN_SLOTS:
        raise EnsembleStateError(
            f"an ensemble slot ring needs at least {MIN_SLOTS} slots "
            f"(one written while another is read); got {slots}")
    if generation < 0:
        raise EnsembleStateError("generation numbers start at 0")
    return Path(root) / f"slot{generation % slots:02d}"


def snapshot_path(directory: Path, name) -> Path:
    return Path(directory) / f"snap_{trajectory_key(name)}.npz"


def pending_path(directory: Path, name) -> Path:
    return Path(directory) / f"pend_{trajectory_key(name)}.npz"


def manifest_path(directory: Path) -> Path:
    return Path(directory) / "ensemble-manifest.json"


def _atomic_savez(path: Path, arrays: dict) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def write_generation(directory: Path, *, identity: EnsembleIdentity,
                     elapsed_seconds: float, leg_number: int,
                     snapshots: dict, pending: dict,
                     valid_time: str | None = None,
                     note: str | None = None) -> dict:
    """Write one generation; return the manifest that was written.

    ``snapshots`` and ``pending`` are keyed exactly as the driver keys
    them (``"control"`` and integer member indices).  A trajectory with
    no pending increments writes no pending file, and its manifest entry
    says so -- absent and empty are different, and only one of them is
    normal at a leg that carried no analysis.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = trajectory_names(identity.members)
    missing = [str(n) for n in names if snapshots.get(n) is None]
    if missing:
        raise EnsembleStateError(
            "cannot write a generation without every trajectory's "
            f"snapshot; missing {missing}")

    # The manifest is written LAST and is the completion marker, so a
    # stale one must not survive a torn rewrite of the same slot.
    marker = manifest_path(directory)
    if marker.exists():
        os.replace(marker, marker.with_name(marker.name + ".superseded"))

    entries = {}
    for name in names:
        key = trajectory_key(name)
        snap = {field: value for field, value
                in dict(snapshots[name]).items()}
        _atomic_savez(snapshot_path(directory, name), snap)
        entry = {"snapshot": snapshot_path(directory, name).name,
                 "snapshot_fields": sorted(snap)}
        increments = pending.get(name)
        if increments:
            _atomic_savez(pending_path(directory, name),
                          dict(increments))
            entry["pending"] = pending_path(directory, name).name
            entry["pending_fields"] = sorted(increments)
        entries[key] = entry

    manifest = {
        "schema": SCHEMA,
        "written": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "identity": identity.to_payload(),
        "elapsed_seconds": float(elapsed_seconds),
        "leg_number": int(leg_number),
        "valid_time": valid_time,
        "note": note,
        "trajectories": entries,
        "honesty": ("host snapshots of the serialized prognostic state "
                    "plus unapplied analysis increments; NOT a bit-exact "
                    "restart (physics driver state is re-initialised at "
                    "every leg boundary, as inside one process)"),
    }
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(tmp, marker)
    return manifest


def read_manifest(directory: Path) -> dict:
    path = manifest_path(directory)
    if not path.is_file():
        raise EnsembleStateError(
            f"{directory} holds no finished ensemble generation "
            f"({path.name} is absent; a generation whose manifest was "
            "never written is a generation that was never finished)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise EnsembleStateError(
            f"{path} carries schema {manifest.get('schema')!r}, not "
            f"{SCHEMA!r}")
    return manifest


def validate_resume(manifest: dict, identity: EnsembleIdentity) -> None:
    """Refuse a generation that does not belong to this run.

    Reports EVERY field that differs rather than the first, because the
    usual cause is a case swapped underneath a daemon and the whole list
    is what says so.
    """

    stored = EnsembleIdentity.from_payload(manifest["identity"])
    differences = []
    for field in ("members", "nx", "ny", "nz", "dt_s", "mp_physics",
                  "physics_profile", "prepared_content_sha256"):
        was = getattr(stored, field)
        now = getattr(identity, field)
        if was != now:
            differences.append(f"{field}: generation {was!r} vs run "
                               f"{now!r}")
    if differences:
        raise EnsembleStateError(
            "this ensemble generation was written for a different run "
            "and restoring it would produce plausible-looking nonsense; "
            + "; ".join(differences))


def read_generation(directory: Path, identity: EnsembleIdentity
                    ) -> tuple[dict, dict, dict]:
    """``(snapshots, pending, manifest)`` for a validated generation."""

    import numpy as np

    directory = Path(directory)
    manifest = read_manifest(directory)
    validate_resume(manifest, identity)
    snapshots: dict = {}
    pending: dict = {}
    for name in trajectory_names(identity.members):
        key = trajectory_key(name)
        entry = manifest["trajectories"].get(key)
        if entry is None:
            raise EnsembleStateError(
                f"generation at {directory} has no entry for "
                f"trajectory {key}")
        with np.load(directory / entry["snapshot"]) as data:
            snapshots[name] = {field: np.ascontiguousarray(data[field])
                               for field in data.files}
        if entry.get("pending"):
            with np.load(directory / entry["pending"]) as data:
                pending[name] = {
                    field: np.ascontiguousarray(data[field])
                    for field in data.files}
        else:
            pending[name] = None
    return snapshots, pending, manifest


def latest_generation(root: Path, *, slots: int = MIN_SLOTS
                      ) -> tuple[Path, dict] | None:
    """The newest finished generation in a slot ring, or None.

    Newest by the manifest's own leg number, so a ring whose slots were
    written out of order still resumes from the furthest-advanced state
    rather than from whichever directory sorts last.
    """

    root = Path(root)
    if not root.is_dir():
        return None
    best: tuple[Path, dict] | None = None
    for index in range(max(slots, MIN_SLOTS)):
        candidate = root / f"slot{index:02d}"
        try:
            manifest = read_manifest(candidate)
        except EnsembleStateError:
            continue
        if best is None or (manifest["leg_number"]
                            > best[1]["leg_number"]):
            best = (candidate, manifest)
    return best
