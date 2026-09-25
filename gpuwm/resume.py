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


# --- the experiment argument -------------------------------------------
#
# A resume is given TWO things: the run directory and the configuration
# that run used.  The directory it is given is the run's own, and the run
# wrote its configuration into it -- so a resume that could not open the
# second argument was refusing a question it already had the answer to.
# Reported from a desktop install whose terminal built the argument out
# of the run's NAME while the file on disk carried the ".toml" the file
# manager was hiding.


#: What a run directory calls the configuration it was made from, in the
#: order a run directory is asked for one.  The three are written by
#: three different routes and do not appear together: ``child.toml`` is
#: ``gpuwm downscale``'s derived child (:data:`gpuwm.downscale.
#: DERIVED_CHILD_CONFIG_NAME`), ``experiment.toml`` is a prepared
#: forecast's, and ``captured-config-<run id>.toml`` is the exact payload
#: :mod:`gpuwm.supervisor` handed its worker.  The capture is a glob
#: because its name carries the run id, and a directory resumed more than
#: once holds one per run.
RUN_RECORD_NAMES = ("child.toml", "experiment.toml")
RUN_RECORD_GLOB = "captured-config-*.toml"


@dataclass(frozen=True)
class ExperimentResolution:
    """Which file a resume will load as its experiment, and how it got there."""

    path: Path
    #: ``"<path>: <what it is instead>"`` for every rung that was not
    #: taken, in the order they were tried.  The refusal prints all of
    #: them; a resolution that succeeded on rung one has none.
    tried: tuple[str, ...] = ()
    #: How this file was reached, when it was not the argument as typed.
    #: ``None`` for the argument itself, which needs no explanation.
    note: str | None = None


def _run_record_candidates(outdir: Path) -> list[Path]:
    """The documents the run in ``outdir`` wrote to record its own config.

    Newest capture first, because a directory resumed more than once
    holds one capture per run and the last one is the configuration the
    last run was given.  Ordered by modification time in nanoseconds with
    the name breaking a tie, for the reason
    :func:`discover_checkpoint_sets` sorts the way it does: a coarsening
    filesystem must not decide which document a resume reads.
    """

    def stamp(path: Path) -> tuple[int, str]:
        try:
            return (path.stat().st_mtime_ns, path.name)
        except OSError:
            return (0, path.name)

    captures = sorted(outdir.glob(RUN_RECORD_GLOB), key=stamp, reverse=True)
    return [outdir / name for name in RUN_RECORD_NAMES] + captures


def resolve_resume_experiment(argument, outdir) -> ExperimentResolution:
    """The experiment ``gpuwm resume ARG --outdir OUT`` should load.

    The argument as typed, first and always: an invocation that already
    names a readable configuration resolves on rung one and nothing else
    here is reached.  When it does not, four more rungs are tried in this
    order, and the refusal at the end names every one of them with what
    was found there instead:

    1. ``ARG`` -- what the reader typed, relative to the working
       directory exactly as every other path argument is.
    2. ``ARG.toml`` -- the extension a file manager hides and a caller
       building the argument out of a run's name never had.
    3. ``OUT/ARG`` -- the argument read against the run directory rather
       than against whatever directory the process happens to be in.
       Skipped for an absolute argument, which rungs 1 and 2 have
       already tried under that exact spelling.
    4. ``OUT/ARG.toml`` -- the two above together.
    5. the configuration the run in ``OUT`` recorded for itself
       (:data:`RUN_RECORD_NAMES`, :data:`RUN_RECORD_GLOB`).  This is the
       rung that makes the argument redundant in practice: the run
       directory a resume is pointed at already holds the bytes that run
       was made from, and those are the bytes the restart identity check
       is going to want.

    Nothing here relaxes a refusal.  Every rung has to be a readable,
    non-empty regular file by :func:`gpuwm.experiment.config_path_kind`,
    which is the judgement the loader makes a moment later on the file
    this returns, and that file reaches the loader unchanged.
    """

    from gpuwm.experiment import config_path_kind

    argument = Path(argument)
    outdir = Path(outdir)
    suffixed = Path(str(argument) + ".toml")
    already_toml = argument.suffix.lower() == ".toml"

    ladder: list[tuple[Path, str | None]] = [(argument, None)]
    if not already_toml:
        ladder.append((suffixed,
                       "the argument with the .toml extension a file "
                       "manager hides"))
    if not argument.is_absolute():
        ladder.append((outdir / argument,
                       "the argument read against --outdir instead of "
                       "the working directory"))
        if not already_toml:
            ladder.append((outdir / suffixed,
                           "the argument read against --outdir, with the "
                           ".toml extension"))
    for record in _run_record_candidates(outdir):
        ladder.append((record,
                       "the configuration the run in --outdir recorded "
                       "for itself"))

    tried: list[str] = []
    seen: set[str] = set()
    for candidate, note in ladder:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        kind = config_path_kind(candidate)
        if kind is None:
            return ExperimentResolution(
                path=candidate, tried=tuple(tried), note=note)
        tried.append(f"{candidate}: {kind}")

    from gpuwm.explain import layered

    listing = "".join(f"\n    {entry}" for entry in tried)
    raise ValueError(layered(
        f"no configuration to resume: {argument} is not a readable "
        "configuration file, and neither is anything this resume could "
        f"name for it.  Tried:{listing}\n"
        "  remedy: pass the experiment .toml that `gpuwm domain` wrote, "
        "or point --outdir at the directory the interrupted run wrote -- "
        f"it holds {' or '.join(RUN_RECORD_NAMES)} or {RUN_RECORD_GLOB} "
        "beside its gpuwmrst_d*.npz checkpoints.",
        "A resume is given the run's directory as well as its "
        "configuration, so the argument is resolved against that "
        "directory and then against the record the run wrote into it "
        "before anything is refused: the argument as typed, the same "
        "argument with the .toml a file manager hides, both of those "
        "read against --outdir, and last the configuration the run "
        "itself recorded.  Every rung is judged by the loader's own "
        "readable-regular-file test, so this ladder cannot admit a path "
        "the loader would then reject."))


# --- a downscaled child is not a run that resume can continue ----------


@dataclass(frozen=True)
class OfflineChildRun:
    """The ``gpuwm downscale`` child that occupies a run directory."""

    outdir: Path
    #: The configuration this directory RECORDS, or ``None`` when it
    #: records none.  Only the ``--point`` derivation writes
    #: ``child.toml`` into ``--out``; a ``--child-config`` run is handed
    #: its configuration from outside the run directory, so that route
    #: leaves this empty and is named by what the run wrote instead
    #: (:func:`offline_child_run_at`).
    config: Path | None
    #: ``report.json`` when the child published one, which it does for
    #: BOTH outcomes: at its last frame, and at the health check that
    #: finds its fields non-finite.  The presence of the document says
    #: only that the child got far enough to publish one; ``result`` is
    #: the verdict, and :attr:`finished` is the reading of it.
    report: Path | None
    result: str | None
    frames: tuple[Path, ...]
    #: The one sentence under the report's ``failure`` block when it
    #: carries one, so this directory can say why the child stopped and
    #: not merely that it did.  It is appended after ``frames`` rather
    #: than placed beside ``result``: a field inserted into the middle
    #: of a frozen dataclass moves every positional index after it.
    failure: str | None = None

    @property
    def finished(self) -> bool:
        """Did this child reach its last frame?

        WHAT BREAKAGE THIS PREVENTS (gate law).  Read as "a report
        exists", this says a child that stopped at model second 2760 of
        28800 finished, and :func:`offline_child_resume_refusal` then
        withholds from that reader the one remedy they need -- run it
        again -- and offers them the frames instead.  A child that
        blows up publishes ``report.json`` too, with ``result`` FAIL
        and its capsule under ``failure``, so the verdict is what
        separates the two outcomes and the document's presence is not.
        """

        return self.report is not None and self.result == "PASS"


def _report_names_a_child(report: dict) -> bool:
    """Does this ``report.json`` say a downscaled child wrote it?

    Three readings, in the order they became true.  The document names
    its own pipeline, which both of a child's outcomes write; a child
    that reached its last frame counts its steps; and one that stopped
    being finite carries the capsule instead, whose first sentence is
    what the refusal quotes.  ``report.json`` is a name several routes
    in this tree write, so the reading is of the CONTENT and never of
    the file existing.
    """

    from gpuwm.offline_child import CHILD_REPORT_PIPELINE

    if report.get("pipeline") == CHILD_REPORT_PIPELINE:
        return True
    if "child_steps" in report:
        return True
    failure = report.get("failure")
    return isinstance(failure, dict) and "summary" in failure


def offline_child_run_at(outdir) -> OfflineChildRun | None:
    """The downscaled child in ``outdir``, or ``None`` for anything else.

    THE ROUTE IS NAMED BY WHAT THE RUN WROTE.  ``gpuwm downscale``
    leaves its plan document; a child's own ``report.json`` names its
    pipeline and carries either the steps of a run that reached its last
    frame or the capsule of one that stopped being finite
    (:func:`_report_names_a_child`).  Any of those names the route, and
    the configuration file beside them is read as a record, not as the
    marker.

    WHAT BREAKAGE THIS PREVENTS (gate law).  Requiring ``child.toml``
    in the directory recognised only the ``--point`` derivation, which
    is the one route that writes that file into ``--out``.  The same
    child run with its configuration handed in through
    ``--child-config`` -- which is where a configuration lives whenever
    a desktop or a script composed it -- was not recognised as a child
    at all, so a reader whose child blew up was answered "no
    gpuwmrst_d*.npz checkpoint files" instead of being handed the
    capsule and the re-run remedy this whole route exists to deliver.
    A configuration file on its own still names nothing: a reader is
    free to keep a file of that name anywhere.
    """

    import json

    from gpuwm.downscale import (DERIVED_CHILD_CONFIG_NAME,
                                 DOWNSCALE_PLAN_NAME)

    outdir = Path(outdir)
    report_path = outdir / "report.json"
    report: dict = {}
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            report = loaded
    if (not (outdir / DOWNSCALE_PLAN_NAME).is_file()
            and not _report_names_a_child(report)):
        return None
    config = outdir / DERIVED_CHILD_CONFIG_NAME
    failure = report.get("failure")
    summary = failure.get("summary") if isinstance(failure, dict) else None
    return OfflineChildRun(
        outdir=outdir, config=(config if config.is_file() else None),
        report=report_path if report else None,
        result=(str(report["result"]) if "result" in report else None),
        frames=tuple(sorted(outdir.glob("wrfout_d*"))),
        failure=(str(summary) if summary else None))


def offline_child_resume_refusal(child: OfflineChildRun) -> str:
    """Why a downscaled child cannot be resumed, and what to do instead.

    The concrete breakage: ``gpuwm resume`` dispatches to ``gpuwm run``,
    and a downscaled child's root is forced at every step from the
    ARCHIVED PARENT -- the initial state interpolated out of the parent
    history, the lateral boundary tendencies built from those frames on
    the parent's own cadence, the child-grid surface seed.  All of it is
    ``gpuwm downscale``'s preparation; ``gpuwm run`` has none of it and
    would integrate the child's root with no lateral forcing at all.
    ``gpuwm downscale`` carries no continuation flag either, so a child
    is re-run rather than continued.

    The way out depends on what the directory holds, and the reading
    that decides it is the report's VERDICT, never the report's
    existence: a child publishes one at its last frame and also at the
    health check that finds its fields non-finite.  A child whose
    report records PASS reached its last frame, so there is nothing
    left to integrate and what the reader almost certainly came for is
    the pictures: the render line goes first.  A child that stopped
    inside its forecast -- no report at all, or one carrying a failure
    -- is re-run, its own capsule is quoted back to it when it wrote
    one, and its frames can still be drawn.
    """

    from gpuwm.explain import layered

    outdir = child.outdir
    frames = len(child.frames)
    draw = (f"gpuwm render {outdir / 'wrfout_d*'} --series "
            f"--out {outdir / 'render'}")
    if child.finished:
        action = (
            f"there is nothing to resume in {outdir}: the downscaled "
            "child that wrote it reached its last frame, and its "
            f"report.json records the run as {child.result or 'finished'}."
            f"\n  remedy: draw the {frames} frame(s) it left:  {draw}")
    else:
        stopped = (
            f"{outdir} holds a downscaled child that did not reach its "
            "last frame, and `gpuwm resume` cannot continue one: a "
            "child is forced at every step from the archived parent, "
            "and only `gpuwm downscale` prepares that forcing.")
        if child.failure:
            stopped += f"  Its report.json says why:  {child.failure}"
        action = (
            stopped
            + "\n  remedy: run `gpuwm downscale` again with the "
              "same arguments and a fresh --out"
            + (f", or draw the {frames} frame(s) already written:  {draw}"
               if frames else "."))
    return layered(
        action,
        "`gpuwm resume` is sugar over `gpuwm run --restart`, and `gpuwm "
        "run` reads one configuration and the inputs that configuration "
        "declares.  A downscaled child declares neither of the two things "
        "it actually runs on: its initial state is interpolated out of "
        "the parent history archive and its lateral boundaries are built "
        "from the parent frames, both by `gpuwm downscale`, which has no "
        "flag that continues a child from its own checkpoint.  The "
        "checkpoints in this directory are still real and still useful: "
        "they are what the NEXT downscale reads as parent evidence.  On a "
        "shell that does not expand a pattern for a native command "
        "(PowerShell), name the frames instead of the wrfout_d* above.")


__all__ = ["LATEST", "CheckpointSet", "ExperimentResolution",
           "OfflineChildRun", "RUN_RECORD_GLOB", "RUN_RECORD_NAMES",
           "ResumeResolution",

           "discover_checkpoint_sets", "offline_child_resume_refusal",
           "offline_child_run_at", "resolve_resume_checkpoint",
           "resolve_resume_experiment",
           "resume_memory_mode_note", "resume_written_mode_note"]
