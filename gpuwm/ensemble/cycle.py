"""The cycling skeleton and its assimilation seam (EXPERIMENTAL).

A cycle is: run every member forward to T, stop, then let an
assimilation step rewrite each member's state before the next leg.  This
module owns the first and third parts.  The second -- the actual
assimilation -- is a callable the caller supplies:

    assimilate(cycle_index, member_states) -> {member_index: {field: ndarray}}

``member_states`` gives, per member, the path of the state the leg
finished on.  The returned increments are applied through
``gpuwm.ensemble.increments``, which is the only code allowed to write
them, and every application is receipted into the cycle manifest.

No assimilation method ships here, and none is implied.  The engine's
contribution is that the seam exists, is atomic, and is provenanced.

**The leg horizon is cumulative, because the integrator's is.**
``gpuwm.runtime.integrate_prepared_case`` defines ``run_seconds`` as the
total forecast length from the experiment's own ``start_time``, and a
restart resumes at the checkpoint's elapsed time and runs *to* that
total.  Passing the leg duration made every leg after the first ask the
integrator to reach a horizon it was already standing on, which it
correctly refused: ``restart file is already at 60.0 s; nothing to
integrate before run_seconds=60.0``.  Leg ``N`` restarting from an
analysis therefore runs to ``(N+1) * cycle_seconds``; a leg that
re-prepares from the base config starts at zero and runs
``cycle_seconds``.  Where a restart states its own elapsed time, the
driver checks it is the ``N * cycle_seconds`` the timeline claims and
refuses a leg that would silently re-integrate or skip an interval.

**The seam is atomic across the whole ensemble, and idempotent on
resume.**  ``assimilation`` in each cycle record is a slot the DA lane
fills, and filling it is a three-phase commit: every member's increments
are validated and written to a staged file; the whole publication is then
declared in a transaction marker (:data:`PUBLICATION_MARKER_NAME`); and
only then are the staged files renamed into place and the marker cleared.
A refusal on member 7 of 10 therefore leaves ten backgrounds and no
analyses, instead of six published analyses and an ensemble that is half
analysed.

The marker is what makes the RENAMES atomic as a set.  A rename is atomic
for one file and there is no filesystem call that makes ten of them one
operation, so the guarantee this driver offers is the recoverable form of
it: a crash between renames leaves a marker naming every member of the
transaction, and :func:`recover_analysis_publication` -- run before any
reader touches a leg -- either rolls the remaining renames forward or
refuses loudly naming the members.  What a reader can never get is a
roster that is half analysed and looks whole.

**The consumer contract, stated rather than implied.**
:func:`read_analysis_roster` is the ONLY supported way to observe a
leg's analyses.  It settles the transaction first and then reports the
roster, so what it returns is every member's analysis or a refusal.

Listing a leg directory yourself is NOT supported and never was.  Phase
three is N renames; between the first and the last, ``ls`` shows a
roster that is real, incomplete, and indistinguishable by inspection
from a leg that genuinely analysed some members -- which is exactly the
state the marker exists to make legible and exactly the state a raw scan
cannot see.  The marker is beside the analyses in the same directory,
and reading one without the other is reading half the record.  This is a
deliberate choice of contract over mechanism: a generation pointer or a
directory commit would let a raw observer be right, and would replace
one rename per member with a scheme whose failure modes are less
obvious.  Every consumer in this tree goes through the recovering
reader, and ``tests/test_ensemble_hardening.py`` fails if a new one does
not.

The cycle record for a leg is *replaced* rather than appended to, and
carries an ``attempt`` counter, so a crash during assimilation and a
retry produce one entry describing two attempts rather than two entries
describing one cycle.

**Positivity is the driver's, because the filter refused it.**
``gpuwm.da.letkf.analyze`` returns increments and deliberately does not
clip them: a Gaussian filter on a bounded, zero-inflated variable will
propose negative mixing ratios, and clip/transform/reject are not
equivalent choices.  This driver is the caller, so this driver chooses --
``positivity="clip"`` by default, with the counts and the mass the clip
*added* recorded per member in the assimilation receipt.  Set
``positivity="none"`` to state the choice the other way; see
:mod:`gpuwm.da.positivity` for why anamorphosis is not on offer here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from gpuwm.ensemble.config import EnsembleConfig
from gpuwm.ensemble.engine import default_member_runner, run_ensemble
from gpuwm.ensemble.increments import (STAGED_SUFFIX,
                                       apply_increments_to_checkpoint,
                                       publish_staged_analysis)
from gpuwm.ensemble.manifest import (
    CYCLE_MANIFEST_NAME, CYCLE_MANIFEST_SCHEMA, ENSEMBLE_MANIFEST_NAME,
    ENSEMBLE_MANIFEST_SCHEMA, cycle_binding, member_directory_name,
    new_cycle_manifest, read_manifest, write_json_atomically,
    write_manifest_atomically,
)
from gpuwm.ensemble.state_sha import checkpoint_elapsed_seconds

#: Where a cycle stashes the analysis it produced for a member.
ANALYSIS_NAME = "analysis.npz"

#: The transaction marker a leg writes before it starts publishing, and
#: removes when every member's analysis is live.  Its presence means "the
#: roster on this leg's disk is mid-transition and no reader may take it
#: at face value"; :func:`recover_analysis_publication` is the only thing
#: that clears it, by finishing the transaction or refusing loudly.
PUBLICATION_MARKER_NAME = "analysis-publication.json"
PUBLICATION_MARKER_SCHEMA = "gpuwm-da-analysis-publication.v1"

#: How close a restart's stated elapsed time has to be to the leg
#: boundary the timeline claims.  Restart clocks are whole outer steps of
#: a float dt, so exact equality is the wrong test; a second is far
#: tighter than any cycle length this driver is used with and far looser
#: than any accumulated float error.
_ELAPSED_TOLERANCE_S = 1.0


@dataclass(frozen=True)
class CycleResult:
    ens_root: Path
    manifest_path: Path
    cycles_run: tuple[int, ...]
    status: str


def cycle_root(ens_root: str | Path, cycle_index: int) -> Path:
    """``ens_root/cycle_000`` -- one whole ensemble per cycle leg."""
    if cycle_index < 0:
        raise ValueError(f"cycle index must be non-negative, "
                         f"got {cycle_index}")
    return Path(ens_root) / f"cycle_{cycle_index:03d}"


def run_cycles(cfg: EnsembleConfig, ens_root: str | Path, *,
               n_cycles: int, cycle_seconds: float,
               assimilate: Callable[[int, Mapping[int, dict]],
                                    Mapping[int, Mapping[str, object]]]
               | None = None,
               runner: Callable = default_member_runner,
               positivity: str = "clip",
               restart_from_analysis: bool = True,
               assimilation_method: Mapping[str, object] | None = None,
               moment_policy: str = "full-moment",
               moment_repair: bool = True,
               mp_physics: int | None = None,
               on_event: Callable[[dict], None] | None = None
               ) -> CycleResult:
    """Run ``n_cycles`` legs of ``cycle_seconds``, assimilating between them.

    Each leg is a complete ensemble under ``ens_root/cycle_NNN`` with its
    own ``gpuwm-ensemble-manifest.v1``; the cycle manifest at the root
    records the leg times, the per-member state shas, and the
    assimilation provenance.

    ``assimilate`` may return either the increments mapping alone or a
    ``(increments, method_provenance)`` pair; ``assimilation_method``
    supplies the same block from the caller when the callable does not.
    Either way the receipt names the method, because two analyses from
    different methods and different observations otherwise produce
    structurally identical receipts.

    ``moment_policy``/``moment_repair``/``mp_physics`` are the
    multi-moment contract (:mod:`gpuwm.da.moments`), threaded to the one
    module that writes an analysis.  The defaults refuse an update that
    moves a multi-moment species' mass while leaving the moment the
    background carries, and repair any broken pair the analysis produces
    anyway through the scheme's own limiter.  ``mp_physics`` is optional
    because the moment structure is detectable from the background's own
    field spellings -- a guard that could be disabled by not passing a
    config is a guard that will be.
    """
    root = Path(ens_root)
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
    if not (cycle_seconds > 0.0):
        raise ValueError(
            f"cycle_seconds must be positive, got {cycle_seconds!r}")

    binding = cycle_binding(cfg, cycle_seconds=cycle_seconds,
                            n_cycles=n_cycles, positivity=positivity,
                            restart_from_analysis=restart_from_analysis)
    manifest_path = root / CYCLE_MANIFEST_NAME
    if manifest_path.is_file():
        manifest = read_manifest(manifest_path, schema=CYCLE_MANIFEST_SCHEMA)
        _check_cycle_compatible(manifest, binding, manifest_path)
    else:
        manifest = new_cycle_manifest(
            cfg, ens_root=root, cycle_seconds=cycle_seconds,
            n_cycles=n_cycles, positivity=positivity,
            restart_from_analysis=restart_from_analysis)
        write_manifest_atomically(manifest_path, manifest)

    # Before anything reads a leg's roster: settle any publication a crash
    # left in flight.  This runs first because EVERY later step -- the
    # resume decision, the restart discovery, the forecast -- reads
    # analyses whose completeness the marker is the only witness to.
    _recover_all_publications(root, on_event=on_event)

    entries = {int(entry["cycle"]): entry
               for entry in manifest.get("cycles", ())}
    done = {index for index, entry in entries.items()
            if entry.get("status") == "DONE"}
    ran = []
    for cycle_index in range(n_cycles):
        if cycle_index in done:
            continue
        leg_root = cycle_root(root, cycle_index)
        _emit(on_event, {"event": "cycle-started", "cycle": cycle_index})
        restarts = _analysis_restarts(root, cycle_index,
                                      n_members=cfg.n_members,
                                      required=restart_from_analysis)
        leg_seconds, clocks = _leg_horizon(restarts, cycle_index,
                                           cycle_seconds)
        result = run_ensemble(cfg, leg_root, run_seconds=leg_seconds,
                              runner=runner, restarts=restarts,
                              on_event=on_event)
        leg_manifest = read_manifest(
            leg_root / ENSEMBLE_MANIFEST_NAME,
            schema=ENSEMBLE_MANIFEST_SCHEMA)
        member_states = {
            int(record["index"]): {
                "member_dir": str(leg_root / record["member_dir"]),
                "state_sha256": record.get("final_state_sha256"),
                "seed": record.get("seed"),
            }
            for record in leg_manifest["members"]
        }

        # Replace, never append: a crash between the forecast record and
        # the DONE record used to leave a FORECAST_COMPLETE entry that the
        # retry appended a second entry beside, so one cycle appeared
        # twice and the manifest's own timeline was wrong.
        previous = entries.get(cycle_index)
        attempt = int((previous or {}).get("attempt", 0)) + 1
        entry = {
            "cycle": cycle_index,
            "attempt": attempt,
            "status": "FORECAST_COMPLETE",
            "ensemble_status": result.status,
            "leg_root": str(leg_root),
            "start_offset_seconds": float(cycle_index * cycle_seconds),
            "end_offset_seconds": float((cycle_index + 1) * cycle_seconds),
            "forecast_seconds": float(cycle_seconds),
            #: What the integrator was asked to reach, from the
            #: experiment's start_time.  Equal to forecast_seconds only on
            #: a leg that re-prepares from the base config.
            "run_seconds_total": float(leg_seconds),
            "restart_clocks": clocks,
            "members": [
                {"index": index,
                 "state_sha256": info["state_sha256"],
                 "seed": info["seed"]}
                for index, info in sorted(member_states.items())
            ],
            # The DA lane's slot.  ``null`` means the leg stopped at the
            # seam with nothing assimilated -- which is a valid, and
            # honest, outcome for a forecast-only cycle.
            "assimilation": None,
        }
        _replace_entry(manifest, entry)
        entries[cycle_index] = entry
        manifest["status"] = "RUNNING"
        write_manifest_atomically(manifest_path, manifest)

        if assimilate is not None:
            entry["assimilation"] = _assimilate_cycle(
                assimilate, cycle_index, member_states, leg_root=leg_root,
                positivity=positivity, on_event=on_event,
                declared_method=assimilation_method, attempt=attempt,
                moment_policy=moment_policy, moment_repair=moment_repair,
                mp_physics=mp_physics)
        entry["status"] = "DONE"
        manifest["status"] = ("COMPLETE" if cycle_index == n_cycles - 1
                              else "RUNNING")
        write_manifest_atomically(manifest_path, manifest)
        ran.append(cycle_index)
        _emit(on_event, {"event": "cycle-finished", "cycle": cycle_index})

    manifest["status"] = "COMPLETE"
    write_manifest_atomically(manifest_path, manifest)
    return CycleResult(ens_root=root, manifest_path=manifest_path,
                       cycles_run=tuple(ran), status=manifest["status"])


def publication_marker_path(leg_root: str | Path) -> Path:
    """Where leg ``leg_root`` records an in-flight analysis publication."""
    return Path(leg_root) / PUBLICATION_MARKER_NAME


def _begin_publication(leg_root: Path, *, cycle_index: int, attempt: int,
                       pairs) -> Path:
    """Declare the whole publication before the first rename happens.

    Phase two is N renames and a rename is atomic only for ONE file.  The
    marker is what makes the SET atomic in the only sense a filesystem
    allows: not "no reader ever sees a half-published roster", which
    POSIX will not sell, but "no reader ever takes a half-published
    roster for a whole one, and the next start finishes it or says so".
    """
    payload = {
        "schema": PUBLICATION_MARKER_SCHEMA,
        "stability": "experimental",
        "cycle": int(cycle_index),
        "attempt": int(attempt),
        "leg_root": str(leg_root),
        "member_count": len(pairs),
        "members": [
            {"member": int(member),
             "staged": str(staged),
             "analysis": str(analysis)}
            for member, staged, analysis in pairs
        ],
    }
    return write_json_atomically(publication_marker_path(leg_root), payload)


def recover_analysis_publication(leg_root: str | Path, *,
                                 on_event=None) -> dict | None:
    """Finish, or loudly refuse, a publication a crash interrupted.

    Returns ``None`` when there is nothing to recover.  Otherwise every
    member named in the marker is settled:

    * a staged file still present is RENAMED INTO PLACE -- rolling
      forward is safe because phase one proved and fsynced every member's
      analysis before phase two began, so the staged bytes are the same
      analysis the completed members already carry;
    * a member whose ``analysis.npz`` is live and whose staged file is
      gone was published before the crash;
    * a member with NEITHER is unrecoverable, and that is a refusal
      naming the members -- rerunning the cycle from the surviving
      backgrounds is the operator's call, not this function's.

    Only when every member is settled is the marker removed, so an
    interrupted recovery is itself recoverable.
    """
    import json

    marker = publication_marker_path(leg_root)
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{marker} records an interrupted analysis publication and is "
            f"unreadable ({exc}). The leg's analyses are in an unknown "
            "state; nothing here will guess which members are live."
        ) from exc
    if not isinstance(payload, dict) \
            or payload.get("schema") != PUBLICATION_MARKER_SCHEMA:
        raise ValueError(
            f"{marker} declares schema {payload.get('schema')!r} if it "
            f"declares one at all; this build recovers "
            f"{PUBLICATION_MARKER_SCHEMA!r}")

    rolled_forward: list[int] = []
    already_live: list[int] = []
    lost: list[int] = []
    for record in payload.get("members", ()):
        member = int(record["member"])
        staged = Path(record["staged"])
        analysis = Path(record["analysis"])
        if staged.is_file():
            publish_staged_analysis(staged, analysis)
            rolled_forward.append(member)
        elif analysis.is_file():
            already_live.append(member)
        else:
            lost.append(member)
    if lost:
        raise ValueError(
            f"cycle {payload.get('cycle')}: the analysis publication "
            f"recorded in {marker} was interrupted, and member(s) "
            f"{sorted(lost)} have neither a published {ANALYSIS_NAME} nor "
            f"the staged file it was to be renamed from. "
            f"Member(s) {sorted(already_live + rolled_forward)} ARE live, "
            "so this leg's roster is mixed and cannot be completed from "
            "what is on disk. Rerun the cycle from the backgrounds, which "
            "are untouched, rather than assimilating half an ensemble.")
    marker.unlink()
    report = {
        "schema": PUBLICATION_MARKER_SCHEMA,
        "cycle": payload.get("cycle"),
        "attempt": payload.get("attempt"),
        "leg_root": str(leg_root),
        "rolled_forward": sorted(rolled_forward),
        "already_live": sorted(already_live),
    }
    _emit(on_event, {"event": "publication-recovered", **report})
    return report


def _member_directory_indices(leg: Path) -> list[int]:
    """The member indices this leg's directories actually carry, ascending.

    Evidence, not a roster.  ``read_analysis_roster`` still never *returns*
    a raw listing -- the whole contract is that a directory scan can see the
    middle of a publication -- but the leg is already open in front of it,
    and the member directories are the leg's own statement of how many
    members it was written for.  A declared count that this contradicts is a
    count that does not describe this leg, and checking it costs one listing
    of a directory that has just been settled.

    Only the canonical spelling counts, so ``member_0000`` beside
    ``member_000`` is not silently read as the same member.

    Canonical means ASCII digits, tested as ASCII digits.  ``str.isdigit``
    is true for characters ``int`` refuses -- ``"²".isdigit()`` is
    ``True`` and ``int("²")`` raises -- so a directory named
    ``member_²`` used to leave this function through a bare
    ``invalid literal for int()`` rather than through anything that
    mentions a roster.  It fails closed either way; a refusal that names
    the leg is the difference.
    """
    if not leg.is_dir():
        return []
    indices = []
    for entry in leg.iterdir():
        if not entry.is_dir() or not entry.name.startswith("member_"):
            continue
        suffix = entry.name[len("member_"):]
        if not (suffix.isascii() and suffix.isdigit()):
            continue
        index = int(suffix)
        if member_directory_name(index) == entry.name:
            indices.append(index)
    return sorted(indices)


def _checked_member_count(n_members) -> int:
    """``n_members`` as a positive, non-boolean ``int``, or a refusal.

    The same runtime-schema statement ``[ensemble] n_members`` already makes
    in :mod:`gpuwm.ensemble.config` and ``SuperobParams`` already makes in
    :mod:`gpuwm.obs.superob`, made here because this is a public reader and
    the count arrives from wherever the caller got it -- which for
    checkpoint tree-support is a branch node rather than the config that
    wrote the leg.

    ``bool`` is excluded explicitly because ``True`` is an ``int`` in
    Python: a flag passed where a count belongs asked for a one-member
    ensemble and got one.  ``2.9`` truncated to two.  ``"3"`` converted.
    None of those is the caller stating a roster size.
    """
    if isinstance(n_members, bool) or not isinstance(n_members, int):
        raise TypeError(
            f"n_members is {n_members!r} ({type(n_members).__name__}); the "
            "expected roster size is a positive int. A bool is an int in "
            "Python, so True asks for a one-member ensemble; a float "
            "truncates; a numeric string converts silently. The conversion "
            "is the caller stating what they meant, and this reader will "
            "not make it on their behalf")
    if n_members < 1:
        raise ValueError(
            f"n_members is {n_members}; an ensemble has at least one member. "
            f"A count of {n_members} makes this reader report an empty "
            "roster for a fully analysed leg, and an empty roster means "
            "'forecast-only' to every caller -- so the analyses this leg "
            "computed, receipted and wrote would be discarded with nothing "
            "raised and every number in the manifest still true")
    return n_members


def read_analysis_roster(leg_root: str | Path, *, n_members: int,
                         on_event=None) -> dict:
    """``{member index: analysis path}`` for one leg.  The supported reader.

    This is the ONLY supported way to observe a leg's analyses, and it is
    the reason the publication contract can be stated at all.  It settles
    the leg's transaction before it looks at anything
    (:func:`recover_analysis_publication`), so the roster it returns is
    one of exactly three things:

    * every member's ``analysis.npz``, when the leg assimilated;
    * ``{}``, when the leg was forecast-only and produced none;
    * a refusal, when the members on disk are neither -- either because a
      publication cannot be completed from what survived, or because some
      members have an analysis and others do not with no transaction in
      flight to explain it.

    It never returns the middle of a publication, because there is no
    longer a middle by the time it looks.

    **Scanning the directory instead is out of contract.**  Phase three
    renames N staged files into place one at a time; a raw listing taken
    between the first and the last is a real, incomplete roster that
    looks exactly like a leg which genuinely analysed some members.  The
    marker beside them says which it is, and this function is what reads
    both together.

    ``n_members`` is the authoritative roster the caller expects.  It is
    required: deriving ensemble size from the same directory being checked
    would let a missing member redefine a partial roster as complete.

    It is also **checked, and falsified against the leg**.  Requiring a
    count is not the same as believing one, and re-verification #5 measured
    the difference: ``n_members=0`` and ``n_members=-1`` returned ``{}`` on
    a fully analysed leg, which :func:`_analysis_restarts` reads as
    forecast-only and which silently discards every analysis; ``True``
    returned a one-member roster; ``2.9`` truncated; and ``2`` against three
    analysed member directories returned two of them and called it one
    ensemble.  So the count must be a positive non-boolean ``int``
    (:func:`_checked_member_count`), and it must agree with the member
    directories the leg carries (:func:`_member_directory_indices`) --
    otherwise this refuses, naming both numbers, rather than returning a
    roster measured with the wrong ruler.

    Independent proof that a caller's count came from authoritative
    metadata is not on offer and is not needed for that: the leg's own
    directories are enough to falsify a wrong one.  The one case they
    cannot speak to is a leg root that is not a directory at all, which
    carries no analyses to discard and still reads as ``{}``.
    """
    n_members = _checked_member_count(n_members)
    leg = Path(leg_root)
    recover_analysis_publication(leg, on_event=on_event)
    observed = _member_directory_indices(leg)
    if leg.is_dir() and observed != list(range(n_members)):
        analysed_beyond = [
            index for index in observed
            if index >= n_members
            and (leg / member_directory_name(index) / ANALYSIS_NAME).is_file()
        ]
        raise ValueError(
            f"{leg}: the caller declares n_members={n_members} and the leg "
            f"carries {len(observed)} member director"
            f"{'y' if len(observed) == 1 else 'ies'} {observed}. "
            + (f"Member(s) {analysed_beyond} are analysed and lie beyond the "
               f"declared count, so a roster of {n_members} would report "
               "part of an ensemble as the whole of one. "
               if analysed_beyond else
               "A count that does not describe this leg cannot say whether "
               "the roster under it is complete. ")
            + "The count is the caller's statement of the ensemble and the "
              "directories are the leg's; refusing rather than reconciling "
              "the two here.")
    found: dict[int, Path] = {}
    missing: list[int] = []
    for index in range(n_members):
        candidate = leg / member_directory_name(index) / ANALYSIS_NAME
        if candidate.is_file():
            found[index] = candidate
        else:
            missing.append(index)
    if found and missing:
        stranded = [index for index in missing
                    if (leg / member_directory_name(index)
                        / (ANALYSIS_NAME + STAGED_SUFFIX)).is_file()]
        hint = ""
        if stranded:
            hint = (f" Members {stranded} carry a staged, unpublished "
                    f"{ANALYSIS_NAME}{STAGED_SUFFIX}, so an assimilation was "
                    "interrupted between staging and publication: rerun that "
                    "cycle.")
        raise ValueError(
            f"{leg}: members {missing} have no {ANALYSIS_NAME}, but members "
            f"{sorted(found)} do, and no publication transaction is in "
            "flight to explain it. A roster where some members are analysed "
            "and others are not is not one ensemble; refusing rather than "
            "reporting part of it." + hint)
    return found


def _recover_all_publications(root: Path, *, on_event=None) -> list[dict]:
    """Settle every leg under ``root`` before the run reads any of them."""
    reports = []
    for leg_root in sorted(Path(root).glob("cycle_*")):
        if not leg_root.is_dir():
            continue
        report = recover_analysis_publication(leg_root, on_event=on_event)
        if report is not None:
            reports.append(report)
    return reports


def _replace_entry(manifest: dict, entry: dict) -> None:
    """Put ``entry`` at its cycle's position, replacing any predecessor."""
    cycles = manifest.setdefault("cycles", [])
    for position, existing in enumerate(cycles):
        if int(existing.get("cycle", -1)) == int(entry["cycle"]):
            cycles[position] = entry
            return
    cycles.append(entry)
    cycles.sort(key=lambda item: int(item.get("cycle", 0)))


def _check_cycle_compatible(manifest, binding, path) -> None:
    """Refuse a resume that would reinterpret an existing timeline.

    The old check compared the member count and the base config hash and
    nothing else, so ``--cycles``, ``--cycle-seconds``, the base seed, the
    perturbation and its options, the positivity policy and the
    restart-from-analysis policy could all change against an existing
    manifest and be accepted.  Each of those decides what the recorded
    cycles MEAN; changing one mid-run makes the manifest a description of
    two different experiments.
    """
    recorded = manifest.get("cycle_binding")
    if not isinstance(recorded, Mapping):
        # A manifest written before the binding existed: fall back to the
        # two facts it did record, and say so rather than guessing.
        recorded = {"n_members": manifest.get("n_members"),
                    "base_config_sha256": manifest.get("base_config_sha256")}
    mismatches = [
        f"{key}: {recorded.get(key)!r} != {value!r}"
        for key, value in sorted(binding.items())
        if key in recorded and recorded.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"{path} was written for a different cycling run: "
            + "; ".join(mismatches)
            + ". Point --ens-root at a new directory, or restore the "
              "configuration that produced this manifest.")


def _restart_elapsed_seconds(path: Path):
    """The elapsed time a checkpoint states, or ``None`` if it states none.

    Real ``gpuwmrst`` files carry a JSON header; the reduced checkpoints
    a synthetic gate writes carry only ``state/*`` arrays and legitimately
    say nothing about a clock.  Unstated is unverifiable, not zero.

    One implementation, in :mod:`gpuwm.ensemble.state_sha`, because the
    member engine's unchanged-state guard reads the same clock off the
    same file and two readings of one fact drift.
    """
    return checkpoint_elapsed_seconds(path)


def _leg_horizon(restarts, cycle_index: int, cycle_seconds: float):
    """``(run_seconds, clock_report)`` for leg ``cycle_index``.

    A leg that re-prepares from the base config starts at zero and runs
    ``cycle_seconds``.  A leg that restarts from an analysis is already
    ``cycle_index * cycle_seconds`` into the forecast and has to be given
    the CUMULATIVE horizon, because that is the number the integrator
    compares its restored clock against.
    """
    if not restarts:
        return float(cycle_seconds), {
            "restarted": False,
            "expected_start_seconds": 0.0,
            "note": "leg prepared from the base config; run_seconds is the "
                    "leg duration because the clock starts at zero",
        }
    expected = float(cycle_index * cycle_seconds)
    stated = {}
    for index, path in sorted(restarts.items()):
        elapsed = _restart_elapsed_seconds(Path(path))
        if elapsed is not None:
            stated[int(index)] = elapsed
    off = {index: value for index, value in stated.items()
           if abs(value - expected) > _ELAPSED_TOLERANCE_S}
    if off:
        raise ValueError(
            f"cycle {cycle_index}: the timeline says this leg starts at "
            f"{expected:g} s, but member(s) "
            + ", ".join(f"{index} at {value:g} s"
                        for index, value in sorted(off.items()))
            + " restart from a different clock. Integrating them to the "
              f"leg-{cycle_index} horizon would re-run or skip an "
              "interval; refusing rather than producing a timeline the "
              "manifest cannot describe.")
    return float((cycle_index + 1) * cycle_seconds), {
        "restarted": True,
        "expected_start_seconds": expected,
        "stated_start_seconds": stated,
        "unstated_members": sorted(set(map(int, restarts)) - set(stated)),
        "note": "run_seconds is the cumulative horizon from the "
                "experiment start_time, which is what "
                "gpuwm.runtime.integrate_prepared_case measures",
    }


def _analysis_restarts(root: Path, cycle_index: int, *, n_members: int,
                       required: bool):
    """Where each member of leg ``cycle_index`` starts from.

    Leg 0 starts from the base config; every later leg starts from the
    previous leg's ``analysis.npz``.  Without this a cycling run re-prepares
    every leg from the base state, which means the analysis is computed,
    receipted, written -- and then thrown away.  The run still completes and
    every number in the manifest is true, which is exactly what makes it
    worth refusing rather than warning about.

    ``required=False`` restores the forecast-only behaviour deliberately.
    A leg whose predecessor produced no analysis (no assimilation callable
    ran) also falls through to the base config, because there is nothing to
    restart from and saying so is better than inventing one.
    """
    if cycle_index == 0 or not required:
        return None
    previous = cycle_root(root, cycle_index - 1)
    # Through the supported reader, not a directory scan.  It settles the
    # previous leg's publication transaction before reporting anything, so
    # what follows sees either all analyses or a refusal and never picks a
    # restart set out of the middle of a rename loop.  This function used
    # to call recovery and then scan; going through the reader means there
    # is one place that knows how to observe a roster, and it is the same
    # place a tool or a future consumer would use.
    try:
        restarts = read_analysis_roster(previous, n_members=n_members)
    except ValueError as exc:
        raise ValueError(f"cycle {cycle_index}: {exc}") from exc
    if not restarts:
        # No analysis at all: a forecast-only predecessor.  Not an error.
        return None
    return restarts


def _callable_path(target) -> str | None:
    """``package.module.name`` for whatever was passed as ``assimilate``."""
    module = getattr(target, "__module__", None)
    name = getattr(target, "__qualname__", None) or getattr(
        target, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    if name:
        return str(name)
    return None


def _method_block(assimilate, returned, declared) -> dict:
    """What produced these increments, as far as anything can say.

    The engine cannot know the method -- that is the point of the seam --
    but it can always record which callable it invoked, and it can carry
    a provenance block the callable or the caller supplied.  A receipt
    that says only ``"method": null`` cannot distinguish two analyses
    from different filters and different observations, which made the
    increment hashes proof of what was applied and of nothing else.
    """
    block = {
        "callable": _callable_path(assimilate),
        "declared_by": None,
        "provenance": None,
    }
    if isinstance(returned, Mapping):
        block["declared_by"] = "assimilation-callable"
        block["provenance"] = dict(returned)
    elif isinstance(declared, Mapping):
        block["declared_by"] = "caller"
        block["provenance"] = dict(declared)
    return block


def _assimilate_cycle(assimilate, cycle_index, member_states, *,
                      leg_root, positivity="clip", on_event=None,
                      declared_method=None, attempt=1,
                      moment_policy="full-moment", moment_repair=True,
                      mp_physics=None) -> dict:
    """Apply one assimilation step's increments and receipt every member.

    Three phases, because two were not enough.  Phase one validates the
    roster, enforces positivity, and writes every member's analysis to a
    staged file.  Phase two declares the whole publication in a
    transaction marker.  Phase three renames the staged files into place
    and clears the marker.

    Phase two is the fix for the residual F-02 case.  With staging alone,
    a crash on the SECOND rename left member 0's analysis live and member
    1's staged, and nothing on disk said so: the next start found "some
    members have an analysis, some do not" and could not tell an
    interrupted publication from a partly-assimilated experiment.  Now the
    marker says it outright, and
    :func:`recover_analysis_publication` finishes the transaction or
    refuses naming the members -- so the roster a reader gets is either
    all pre-analysis or all post-analysis, never the middle.
    """
    from gpuwm.da.positivity import POLICIES

    if positivity not in POLICIES:
        raise ValueError(
            f"unknown positivity policy {positivity!r}; known policies are "
            f"{POLICIES}")
    returned = assimilate(cycle_index, member_states)
    method_provenance = None
    if isinstance(returned, tuple) and len(returned) == 2:
        returned, method_provenance = returned
    increments_by_member = returned
    if not isinstance(increments_by_member, Mapping):
        raise TypeError(
            "the assimilation step must return a mapping of member index "
            "to {field_name: ndarray} (optionally paired with a method "
            "provenance mapping), got "
            f"{type(increments_by_member).__name__}")

    # The roster is the whole ensemble or it is a refusal.  An analysis
    # over a subset is a different experiment from the one the forecast
    # leg ran, and publishing it member by member made that discoverable
    # only by counting files afterwards.
    wanted = {int(index) for index in member_states}
    offered = {int(index) for index in increments_by_member}
    if offered != wanted:
        extra = sorted(offered - wanted)
        absent = sorted(wanted - offered)
        parts = []
        if extra:
            parts.append(f"member(s) {extra} are not in this cycle")
        if absent:
            parts.append(f"member(s) {absent} got no increments")
        raise ValueError(
            f"cycle {cycle_index}: the assimilation step returned "
            f"increments for {len(offered)} of {len(wanted)} members; "
            + "; ".join(parts)
            + ". A partly analysed ensemble is not an analysis, so nothing "
              "was written.")

    staged: list[tuple[int, Path, Path]] = []
    receipts = []
    positivity_receipts = []
    try:
        for index, increments in sorted(increments_by_member.items()):
            info = member_states[int(index)]
            background = _member_background_checkpoint(
                Path(info["member_dir"]))
            analysis = Path(info["member_dir"]) / ANALYSIS_NAME

            increments, positivity_receipt = _enforce_positivity(
                background, increments, policy=positivity)
            positivity_receipt["member"] = int(index)
            positivity_receipts.append(positivity_receipt)

            receipt = apply_increments_to_checkpoint(
                background, increments, analysis, publish=False,
                moment_policy=moment_policy, moment_repair=moment_repair,
                mp_physics=mp_physics)
            receipt["member"] = int(index)
            receipt["background"] = str(background)
            receipt["positivity"] = positivity_receipt
            receipts.append(receipt)
            staged.append((int(index), Path(receipt["staged"]), analysis))
    except BaseException:
        # Phase one failed: leave no staged file behind to be mistaken
        # for a half-finished analysis on the next attempt.
        for _, path, _ in staged:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    # Phase two: declare the transaction.  Written and fsynced BEFORE the
    # first rename, so a crash anywhere in phase three leaves a marker
    # naming every member the transaction covers.
    marker = _begin_publication(leg_root, cycle_index=cycle_index,
                                attempt=attempt, pairs=staged)

    # Phase three: the whole ensemble becomes analysed here.  A crash
    # part-way leaves the marker, and the next start rolls the remaining
    # renames forward or refuses naming the members -- never a roster that
    # merely looks complete.
    for (member_index, path, analysis), receipt in zip(staged, receipts):
        publish_staged_analysis(path, analysis)
        receipt["published"] = True
        receipt.pop("staged", None)
        _emit(on_event, {"event": "member-assimilated", "cycle": cycle_index,
                         "index": receipt["member"]})
    marker.unlink(missing_ok=True)
    return {
        "status": "APPLIED",
        "stability": "experimental",
        "member_count": len(receipts),
        "attempt": int(attempt),
        "commit": "three-phase: every member staged, the publication "
                  "declared in a transaction marker, then all published; "
                  "an interrupted publication is rolled forward or refused "
                  "by recover_analysis_publication",
        "publication_marker": PUBLICATION_MARKER_NAME,
        # Filled by whoever implements the method: name, version, obs
        # set, localisation.  The engine also records the callable it
        # invoked, which it always knows.
        "method": _method_block(assimilate, method_provenance,
                                declared_method),
        "positivity_policy": positivity,
        "moment_policy": moment_policy,
        "moment_repair": bool(moment_repair),
        "moment_repaired_cells_total": sum(
            int((entry.get("moments") or {}).get("repaired_cells_total", 0))
            for entry in receipts),
        "negative_points_total": sum(
            int(entry.get("negative_points", 0))
            for entry in positivity_receipts),
        "mass_added_by_clip_total": sum(
            float(entry.get("mass_added_by_clip", 0.0))
            for entry in positivity_receipts),
        "receipts": receipts,
    }


def _enforce_positivity(background: Path, increments, *, policy):
    """Read the background the increments land on, and bound the analysis.

    The constraint is on ``prior + increment``, so the background has to be
    read -- an increment on its own cannot say whether the analysis is
    negative.  Only the constrained fields present in the increment mapping
    are loaded; a wind-only analysis reads nothing.
    """
    import numpy as np

    from gpuwm.da.positivity import (apply_positivity, constrained_fields,
                                     verify_non_negative)

    wanted = constrained_fields(tuple(increments))
    if not wanted:
        return increments, {
            "schema": "gpuwm-da.positivity.v1", "policy": policy,
            "constrained_fields": [], "negative_points": 0,
            "mass_added_by_clip": 0.0,
            "note": ("no field in this increment set is bounded below; "
                     "positivity had nothing to enforce"),
        }
    with np.load(background, allow_pickle=False) as data:
        prior = {}
        for name in wanted:
            key = f"state/{name}"
            if key not in data.files:
                raise ValueError(
                    f"the analysis proposes an increment to {name!r}, which "
                    f"the background checkpoint {background.name} does not "
                    "carry; positivity cannot be enforced against a "
                    "background that is not there")
            prior[name] = data[key]
    adjusted, receipt = apply_positivity(prior, increments, policy=policy)
    if policy != "none":
        verify_non_negative(prior, adjusted)
    return adjusted, receipt


def _member_background_checkpoint(member_dir: Path) -> Path:
    """The newest ``gpuwmrst_*.npz`` a member wrote.  Fails closed."""
    candidates = sorted(member_dir.glob("gpuwmrst_*.npz"))
    if not candidates:
        raise ValueError(
            f"member directory {member_dir} carries no gpuwmrst_*.npz "
            "checkpoint, so there is no state for the assimilation step "
            "to rewrite. Set restart_interval_s in the base experiment "
            "config so each leg checkpoints at its end.")
    return candidates[-1]


def _emit(on_event, payload) -> None:
    if on_event is not None:
        on_event(payload)
