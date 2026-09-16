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
    #: Disclosures about the resume itself, for the caller to print
    #: alongside ``skipped``.  Never a refusal and never a condition:
    #: each entry states something true about this resume that the
    #: operator could otherwise only derive from two modules or not at
    #: all (which memory mode this run resolves to, which road wrote
    #: the checkpoint).  Empty when there is nothing to say.
    notes: tuple[str, ...] = ()


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


_MEMORY_MODE_WORDS = {
    "off": "resident",
    "on": "streamed",
    "auto": "streamed where one budget fits the domain and resident "
            "otherwise",
    "mixed": "a per-domain mix of streamed and resident grids",
}


def _resolved_memory_mode(config) -> str | None:
    """The words for the mode THIS run's experiment resolves [tiles] to.

    The combination restart x memory mode is free by construction:
    :func:`gpuwm.core.streaming.identity_payload_entry` contributes
    nothing to the restart identity, so a checkpoint written streamed
    resumes resident and one written resident resumes streamed.  That is
    a promise worth stating rather than leaving the operator to infer
    from two modules, and it is stated as a fact, never as a condition:
    nothing here can refuse a resume.

    ``None`` when the config cannot be read as a config at all; a
    disclosure declines to guess, and the loader that owns the refusal
    makes it a moment later.
    """
    from gpuwm.core.streaming import StreamingOptions

    config = Path(config)
    if not config.is_file():
        return None
    import tomllib

    try:
        with open(config, "rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, ValueError):
        return None
    tables = [raw.get("tiles")]
    domains = raw.get("domain")
    if isinstance(domains, list):
        tables += [dom.get("tiles") for dom in domains
                   if isinstance(dom, dict) and "tiles" in dom]
    modes = []
    for table in tables:
        try:
            options = StreamingOptions.from_mapping(
                table, source=str(config))
        except (TypeError, ValueError):
            return None
        modes.append(options.mode)
    return _MEMORY_MODE_WORDS.get(
        modes[0] if len(set(modes)) == 1 else "mixed")


def resume_memory_mode_note(config) -> str | None:
    """Which memory mode THIS resume resolves to, and why the file agrees.

    Stated as a fact and never as a condition: nothing here can refuse a
    resume.  See :func:`_resolved_memory_mode` for why the combination
    restart x memory mode is free, and :func:`resume_written_mode_note`
    for the half of the sentence that comes off the checkpoint itself.
    """
    resolved = _resolved_memory_mode(config)
    if resolved is None:
        return None
    return (f"this run resolves [tiles] to {resolved}; the checkpoint is "
            "mode-independent by contract (streaming contributes nothing "
            "to the restart identity), so a checkpoint written either way "
            "resumes either way")


def resume_written_mode_note(checkpoint, *, read_header=_default_read_header,
                             config=None) -> str | None:
    """What the CHECKPOINT says it was written with, beside this run's mode.

    The sentence the memory-mode disclosure could not say on its own.
    ``resume_memory_mode_note`` reads the experiment and can therefore
    only ever report the mode of the run doing the READING; the file's
    own ``written_mode`` stamp (``gpuwm.io.restart.written_mode_note``)
    is the other half, and it is the half an operator cannot recover
    once the run that died has taken its logs with it.

    ``None`` costs nothing and refuses nothing.  It is the answer for a
    checkpoint that names no road (one written before the stamp existed,
    or one written by the streamed writer, which does not stamp yet; see
    ``gpuwm.io.restart.written_mode_note``) and for a header that cannot
    be read at all, which the restart machinery refuses a moment later
    with the file in its hands.  So no note is read as "the file does
    not say", and the note is only ever made about a file that does.
    """
    from gpuwm.io.restart import header_written_mode

    try:
        written = header_written_mode(read_header(checkpoint))
    except Exception:
        return None
    if written is None:
        return None
    resolved = None if config is None else _resolved_memory_mode(config)
    if resolved is None:
        return (f"this checkpoint was WRITTEN {written}; the restart is "
                "mode-independent by contract (streaming contributes "
                "nothing to the restart identity), so it resumes either "
                "way")
    return (f"this checkpoint was WRITTEN {written} and this run resolves "
            f"[tiles] to {resolved}; the restart is mode-independent by "
            "contract (streaming contributes nothing to the restart "
            "identity), so the difference is disclosed and never refused")


def resolve_resume_checkpoint(outdir, spec: str | Path = LATEST, *,
                              validate=_default_validate,
                              read_header=_default_read_header,
                              config=None) -> ResumeResolution:
    """Resolve ``--from`` to a checkpoint path the run machinery accepts.

    An explicit path is returned as-is after an existence check -- the
    restart machinery owns its validation and its identity refusals.
    ``latest`` walks the discovered sets newest first and returns the
    first whose members are all manifest-valid and whose tree header
    agrees with the files on disk; every newer set refused on the way is
    recorded so the caller can print why the resume point is older than
    the newest file.

    ``config`` is the experiment being resumed.  It contributes
    disclosure and never a refusal: it supplies the ``notes`` entry
    naming which memory mode this run resolves to and why the checkpoint
    does not care (:func:`resume_memory_mode_note`).  A config that
    cannot be read contributes no note and stops nothing; the loader
    that owns that refusal makes it a moment later.

    The resolved checkpoint contributes the other half of that sentence
    when it carries one: the road it was WRITTEN on
    (:func:`resume_written_mode_note`).  A file that names no road
    contributes nothing and resumes exactly as it always did.
    """
    outdir = Path(outdir)
    notes = () if config is None else tuple(
        note for note in (resume_memory_mode_note(config),)
        if note is not None)

    def with_written_mode(checkpoint) -> tuple[str, ...]:
        written = resume_written_mode_note(
            checkpoint, read_header=read_header, config=config)
        return notes if written is None else notes + (written,)

    if str(spec) != LATEST:
        checkpoint = Path(spec)
        if not checkpoint.is_file():
            raise ValueError(
                f"--from checkpoint {checkpoint} does not exist; pass a "
                f"gpuwmrst_*.npz file or '{LATEST}' to discover the "
                f"newest valid set in {outdir}")
        return ResumeResolution(checkpoint=checkpoint, checkpoint_set=None,
                                skipped=(),
                                notes=with_written_mode(checkpoint))
    candidates = discover_checkpoint_sets(outdir)
    if not candidates:
        # The breakage and the way out, and nothing about which ROUTE the
        # config steers to: every route this tree ships writes
        # gpuwmrst_d*.npz when restart_interval_s is positive, so a
        # sentence saying otherwise named a limit that does not exist.
        raise ValueError(
            f"no gpuwmrst_d*.npz checkpoint files in {outdir}; resume "
            "needs the --outdir of the run being continued, and that run "
            "must have written a restart (restart_interval_s).  Resume "
            "requires a complete valid set of gpuwmrst_d*.npz "
            "checkpoints")
    skipped: list[str] = []
    for candidate in candidates:
        try:
            _check_set(candidate, validate, read_header)
        except Exception as exc:
            skipped.append(f"{candidate.describe()}: {exc}")
            continue
        return ResumeResolution(checkpoint=candidate.handle,
                                checkpoint_set=candidate,
                                skipped=tuple(skipped),
                                notes=with_written_mode(candidate.handle))
    raise ValueError(
        f"every checkpoint set in {outdir} failed validation; refusing "
        "to guess.  Reasons, newest first:\n  " + "\n  ".join(skipped))


__all__ = ["LATEST", "CheckpointSet", "ResumeResolution",
           "discover_checkpoint_sets", "resolve_resume_checkpoint",
           "resume_memory_mode_note", "resume_written_mode_note"]
