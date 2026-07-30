"""``gpuwm resume`` -- continue a run from its newest valid checkpoint.

Thin sugar over the proven restart machinery: this module only LOCATES a
checkpoint.  Everything that makes a resume safe -- the manifest-valid
member proof, the config/setup/physics identity checks, the complete
tree-set refusal -- already lives in :mod:`gpuwm.io.restart` and
:mod:`gpuwm.supervisor` and runs unchanged when the located path is
handed to the ordinary ``run --restart`` dispatch.  Nothing here relaxes
a refusal; an invalid NEWEST checkpoint is skipped with a printed reason
and the next-newest valid one is taken, which is exactly what an
operator does by hand after a crash mid-write.

Checkpoint naming (``gpuwm.io.restart.restart_filename`` and
``write_tree_restart``): ``gpuwmrst_d0X_YYYY-MM-DD_HH_MM_SS.npz`` for a
single domain, with a ``__<checkpoint_set_id>`` member suffix for tree
sets.  A SET is every file sharing one instant + set id; its handle is
the lowest grid id (the root), which is the path ``restore_tree_restart``
expects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

#: ``restart_filename``'s instant, made discoverable: the strftime pattern
#: is ``%Y-%m-%d_%H_%M_%S`` and tree members append ``__<set id>``.
_CHECKPOINT_NAME = re.compile(
    r"^gpuwmrst_d(?P<grid_id>[0-9]+)_"
    r"(?P<instant>[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2})"
    r"(?P<set_id>__.+)?\.npz$")

_INSTANT_FORMAT = "%Y-%m-%d_%H_%M_%S"

#: The ``--from`` spelling that asks for discovery instead of a path.
LATEST = "latest"


@dataclass(frozen=True)
class CheckpointSet:
    """One restart instant: every domain member written together."""

    valid_time: datetime
    set_id: str | None            # tree checkpoint_set_id suffix, sans "__"
    members: dict[int, Path]      # grid_id -> file

    @property
    def handle(self) -> Path:
        """The member ``run --restart`` takes: the root (lowest grid id)."""
        return self.members[min(self.members)]

    def describe(self) -> str:
        ids = ",".join(f"d{gid:02d}" for gid in sorted(self.members))
        tag = "" if self.set_id is None else f" set {self.set_id}"
        return (f"{self.valid_time.strftime(_INSTANT_FORMAT)}"
                f"{tag} ({ids})")


@dataclass(frozen=True)
class ResumeResolution:
    checkpoint: Path
    checkpoint_set: CheckpointSet | None   # None for an explicit --from path
    skipped: tuple[str, ...]               # newer sets refused, with reasons


def discover_checkpoint_sets(outdir) -> list[CheckpointSet]:
    """Every complete-on-disk checkpoint set in ``outdir``, newest first.

    Newest-first is by restart valid time, then by file modification time
    for two sets checkpointing the same instant (a supervisor retry writes
    a fresh set id at the same model clock), then by set id.

    The mtime is read in nanoseconds and the set id breaks the remaining
    tie.  Second-resolution mtimes and a coarsening filesystem could put
    two sets for one model instant on an exact tie, and the order then
    fell out of ``Path.glob`` discovery -- so which checkpoint a resume
    continued from was a property of the filesystem, not of the run.  A
    set with no id sorts below any set that has one.
    """
    outdir = Path(outdir)
    groups: dict[tuple[str, str | None], dict[int, Path]] = {}
    for path in outdir.glob("gpuwmrst_d*.npz"):
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is None:
            continue
        key = (match.group("instant"), match.group("set_id"))
        groups.setdefault(key, {})[int(match.group("grid_id"))] = path
    sets = [
        CheckpointSet(
            valid_time=datetime.strptime(instant, _INSTANT_FORMAT),
            set_id=None if set_id is None else set_id[2:],
            members=members)
        for (instant, set_id), members in groups.items()
    ]
    return sorted(
        sets,
        key=lambda s: (s.valid_time,
                       max(path.stat().st_mtime_ns
                           for path in s.members.values()),
                       "" if s.set_id is None else s.set_id),
        reverse=True)


def _default_validate(path: Path) -> None:
    from gpuwm.supervisor import validate_manifest_checkpoint

    validate_manifest_checkpoint(path)


def _default_read_header(path: Path) -> dict:
    from gpuwm.io.restart import read_restart_header

    return read_restart_header(path)


def _check_set(candidate: CheckpointSet, validate, read_header) -> None:
    """Raise with the reason this set cannot be resumed from."""
    header = read_header(candidate.handle)
    declared = header.get("domain_ids")
    if declared is not None and sorted(candidate.members) != list(declared):
        raise ValueError(
            f"tree set declares domains {list(declared)} but only "
            f"{sorted(candidate.members)} are on disk (torn set)")
    for grid_id in sorted(candidate.members):
        validate(candidate.members[grid_id])


def resolve_resume_checkpoint(outdir, spec: str | Path = LATEST, *,
                              validate=_default_validate,
                              read_header=_default_read_header
                              ) -> ResumeResolution:
    """Resolve ``--from`` to a checkpoint path the run machinery accepts.

    An explicit path is returned as-is after an existence check -- the
    restart machinery owns its validation and its identity refusals.
    ``latest`` walks the discovered sets newest first and returns the
    first whose members are all manifest-valid and whose tree header
    agrees with the files on disk; every newer set refused on the way is
    recorded so the caller can print why the resume point is older than
    the newest file.
    """
    outdir = Path(outdir)
    if str(spec) != LATEST:
        checkpoint = Path(spec)
        if not checkpoint.is_file():
            raise ValueError(
                f"--from checkpoint {checkpoint} does not exist; pass a "
                f"gpuwmrst_*.npz file or '{LATEST}' to discover the "
                f"newest valid set in {outdir}")
        return ResumeResolution(checkpoint=checkpoint, checkpoint_set=None,
                                skipped=())
    candidates = discover_checkpoint_sets(outdir)
    if not candidates:
        raise ValueError(
            f"no gpuwmrst_d*.npz checkpoint files in {outdir}; resume "
            "needs the --outdir of the run being continued, and that run "
            "must have written a restart (restart_interval_s)")
    skipped: list[str] = []
    for candidate in candidates:
        try:
            _check_set(candidate, validate, read_header)
        except Exception as exc:
            skipped.append(f"{candidate.describe()}: {exc}")
            continue
        return ResumeResolution(checkpoint=candidate.handle,
                                checkpoint_set=candidate,
                                skipped=tuple(skipped))
    raise ValueError(
        f"every checkpoint set in {outdir} failed validation; refusing "
        "to guess.  Reasons, newest first:\n  " + "\n  ".join(skipped))


__all__ = ["LATEST", "CheckpointSet", "ResumeResolution",
           "discover_checkpoint_sets", "resolve_resume_checkpoint"]
