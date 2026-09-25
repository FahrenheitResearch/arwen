"""The cycling ensemble, written down between processes.

:mod:`tools.da_cycle_prepared` joins one leg to the next through the
restart owner, :mod:`gpuwm.io.restart`: at the end of a leg every
trajectory writes a complete tree checkpoint set (every domain it
carries: the atmosphere, the physics driver's surface and soil state,
the precipitation accumulators, the held tendencies and the clocks),
and the next leg restores that set into a freshly wired model before
it applies the analysis.  That state lives for exactly as long as the
process does, which is why a run has always had to declare every cycle
up front.

A continuous nowcast cannot: the next observation does not exist yet
when this one is assimilated.  This module is the leg boundary written
to disk, so the SAME state that survives a leg inside one process
survives the gap between two.  Nothing here is a second ensemble
representation -- a generation holds the driver's own restart sets,
copied member for member, plus the driver's own ``pending`` mapping in
the driver's own ``np.savez`` form.

**What a generation holds.**  For every trajectory (the never-analysed
control and each member): its post-leg restart set, and the increments
the analysis at that leg produced and the NEXT leg has still to apply.
Both are required.  A generation with the restart set alone would
silently drop one cycle's analysis -- the model would advance from a
background nobody corrected, and no error would be raised anywhere.

**Identity is checked, not assumed.**  A generation is only meaningful
against the same prepared case, physics profile, grid, timestep and
ensemble size that produced it.  :func:`validate_resume` compares every
one of those and refuses with the full list of what differs, because
restoring a 49-level checkpoint into a 51-level state is the kind of
mistake that produces plausible-looking output.  The restart owner
then checks the rest -- configuration, base state, physics setup --
member by member when the set is restored.

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
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: The generation manifest's contract string.  ``v2`` is the restart-set
#: generation; a ``v1`` generation held host arrays of the atmosphere
#: alone and is refused by name (:func:`read_manifest`).
SCHEMA = "gpuwm-da.cycle-ensemble.v2"

#: The schema this module wrote before the leg join went through the
#: restart owner.  Named so the refusal can say what such a generation
#: is and what to do instead.
RETIRED_SCHEMA = "gpuwm-da.cycle-ensemble.v1"

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


def trajectory_name(key: str):
    """The driver's own trajectory name for a manifest key."""

    if key == CONTROL:
        return CONTROL
    if not (key.startswith("m") and key[1:].isdigit()):
        raise EnsembleStateError(
            f"{key!r} is not a trajectory key this module writes")
    return int(key[1:])


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


def restart_dir(directory: Path, name) -> Path:
    """Where a trajectory's restart set lives inside a generation.

    One directory per trajectory, because a tree checkpoint set is found
    by the instant in its members' names and two trajectories' sets of
    the same instant in one directory would be one set with duplicate
    members.
    """
    return Path(directory) / f"restart_{trajectory_key(name)}"


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


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)


def copy_restart_set(root_member: Path, target: Path) -> dict:
    """Copy one trajectory's whole checkpoint set into ``target``.

    ``root_member`` is the root's member file, as
    :func:`gpuwm.io.restart.write_tree_restart` returned it.  Every
    member of the set is copied under its own name, so the copy is a
    set :func:`gpuwm.io.restart.restore_tree_restart` reads exactly as
    it read the original.  Returns the manifest entry for the set:
    the root member's name, every member by grid id, and the domain
    ids.  A stale copy under ``target`` from an earlier use of this
    slot is removed first, so a set cannot inherit a member from the
    generation before it.
    """
    from gpuwm.io.restart import tree_restart_members

    root_member = Path(root_member)
    members = tree_restart_members(root_member)
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    for member in members.values():
        _atomic_copy(member, target / member.name)
    return {
        "restart_root": root_member.name,
        "restart_members": {str(gid): members[gid].name
                            for gid in sorted(members)},
        "domain_ids": sorted(int(gid) for gid in members),
    }


def write_generation(directory: Path, *, identity: EnsembleIdentity,
                     elapsed_seconds: float, leg_number: int,
                     restarts: dict, pending: dict,
                     nest: dict | None = None,
                     valid_time: str | None = None,
                     note: str | None = None) -> dict:
    """Write one generation; return the manifest that was written.

    ``restarts`` maps each trajectory, keyed exactly as the driver keys
    them (``"control"`` and integer member indices), to the root member
    of the tree checkpoint set that leg ended on; the whole set is
    copied in.  ``pending`` is keyed the same way.  A trajectory with no
    pending increments writes no pending file, and its manifest entry
    says so -- absent and empty are different, and only one of them is
    normal at a leg that carried no analysis.

    A trajectory that carries a nest carries it inside its own set, so
    the child crosses the process boundary with the parent and by the
    same reader; ``nest`` is the child's geometry receipt and is
    recorded for the resuming run to check its own child against.  A
    generation that dropped the child would be a child born again at
    the next process boundary, which is the same defect as a child born
    at the fork one seam further along.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = trajectory_names(identity.members)
    missing = [str(n) for n in names if restarts.get(n) is None]
    if missing:
        raise EnsembleStateError(
            "cannot write a generation without every trajectory's "
            f"restart set; missing {missing}")

    # The manifest is written LAST and is the completion marker, so a
    # stale one must not survive a torn rewrite of the same slot.
    marker = manifest_path(directory)
    if marker.exists():
        os.replace(marker, marker.with_name(marker.name + ".superseded"))

    entries = {}
    for name in names:
        key = trajectory_key(name)
        entry = {"restart": restart_dir(directory, name).name}
        entry.update(copy_restart_set(Path(restarts[name]),
                                      restart_dir(directory, name)))
        increments = pending.get(name)
        if increments:
            _atomic_savez(pending_path(directory, name),
                          dict(increments))
            entry["pending"] = pending_path(directory, name).name
            entry["pending_fields"] = sorted(increments)
        else:
            stale = pending_path(directory, name)
            if stale.exists():
                stale.unlink()
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
        "nest": nest,
        "contract": ("one complete tree checkpoint set per trajectory "
                     "(gpuwm.io.restart: every domain's atmosphere, "
                     "physics driver, accumulators, held tendencies and "
                     "clock) plus the unapplied analysis increments; a "
                     "leg restored from it continues the leg that wrote "
                     "it exactly as gpuwm run --restart continues a "
                     "run"),
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
    schema = manifest.get("schema")
    if schema == RETIRED_SCHEMA:
        raise EnsembleStateError(
            f"{path} is a {RETIRED_SCHEMA} generation: host arrays of the "
            "serialized atmosphere alone, with no soil, surface, "
            "accumulator, held-tendency or clock state, so a leg resumed "
            "from it would start every one of those from the prepared "
            "background at that instant.  This tree resumes only a "
            f"{SCHEMA} generation, which carries each trajectory's "
            "complete tree checkpoint set; start the cycle again from "
            "the prepared background and let it write one")
    if schema != SCHEMA:
        raise EnsembleStateError(
            f"{path} carries schema {schema!r}, not {SCHEMA!r}")
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
    """``(restarts, pending, manifest)`` for a validated generation.

    ``restarts`` maps each trajectory to the root member of its
    checkpoint set inside the generation, which is what
    :func:`gpuwm.io.restart.restore_tree_restart` takes; every member
    the manifest names has to be there, so a torn copy is refused here
    rather than half restored.
    """

    import numpy as np

    directory = Path(directory)
    manifest = read_manifest(directory)
    validate_resume(manifest, identity)
    restarts: dict = {}
    pending: dict = {}
    for name in trajectory_names(identity.members):
        key = trajectory_key(name)
        entry = manifest["trajectories"].get(key)
        if entry is None:
            raise EnsembleStateError(
                f"generation at {directory} has no entry for "
                f"trajectory {key}")
        set_dir = directory / entry["restart"]
        absent = [member for member in entry["restart_members"].values()
                  if not (set_dir / member).is_file()]
        if absent:
            raise EnsembleStateError(
                f"generation at {directory} names restart members "
                f"{absent} for trajectory {key} that are not there; "
                "a checkpoint set with a member missing restores nothing")
        restarts[name] = set_dir / entry["restart_root"]
        if entry.get("pending"):
            with np.load(directory / entry["pending"]) as data:
                pending[name] = {
                    field: np.ascontiguousarray(data[field])
                    for field in data.files}
        else:
            pending[name] = None
    return restarts, pending, manifest


def nested_trajectories(manifest: dict) -> list:
    """The trajectories whose restart set carries more than the root.

    Read off the manifest's own ``domain_ids`` per trajectory, keyed as
    the driver keys them, so a resuming run knows which of its
    trajectories continue a child and which are born one.
    """

    out = []
    for key, entry in manifest["trajectories"].items():
        ids = entry.get("domain_ids") or []
        if len(ids) > 1:
            out.append(trajectory_name(key))
    return out


def latest_generation(root: Path, *, slots: int = MIN_SLOTS
                      ) -> tuple[Path, dict] | None:
    """The newest finished generation in a slot ring, or None.

    Newest by the manifest's own leg number, so a ring whose slots were
    written out of order still resumes from the furthest-advanced state
    rather than from whichever directory sorts last.  A slot holding a
    retired-format generation is not a candidate: resuming it is a
    refusal by name, and a daemon looking for its newest state has to
    be able to look past it to a slot this tree can read.
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
