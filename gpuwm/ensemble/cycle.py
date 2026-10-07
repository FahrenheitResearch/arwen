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

**The analysis decision survives the outer completion write.** Every member
is staged before an immutable intent binds its full-file digest, input
context and complete prospective assimilation receipt. Recovery checks the
whole set before any remaining rename and publishes an immutable commit
receipt. Both records remain available after completion, so losing the
outer DONE write cannot invoke assimilation again.

The run compares its current member, method, policy and input identities
before recovering a decision. A reader verifies every recorded input and
analysis file before returning a roster. A corrupt or conflicting record
cannot be repaired by silently recomputing over a committed analysis.

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
``positivity="mean-preserving"`` by default, over the WHOLE ensemble at
once: the undershooting members end at zero and the members' non-negative
analyses are scaled so the ensemble keeps the filter's mean, with the
counts, the mass added and the mass redistributed recorded per member in
the assimilation receipt.  The clip it replaced raised every negative
member to zero and lowered none, adding mass at every analysis.  ``clip``,
``reject`` and ``none`` remain selectable; see :mod:`gpuwm.da.positivity`
for why anamorphosis is not on offer here.
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

#: The retained immutable decision precedes the first member rename.
#: The prior schema is named separately for bounded compatibility reads.
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
               positivity: str = "mean-preserving",
               restart_from_analysis: bool = True,
               assimilation_method: Mapping[str, object] | None = None,
               moment_policy: str = "full-moment",
               moment_repair: bool = True,
               mp_physics: int | None = None,
               on_event: Callable[[dict], None] | None = None,
               analysis_context: Callable | None = None,
               first_cycle: int = 0,
               initial_restarts: Mapping[int, str | Path] | None = None,
               input_binding: Mapping[str, object] | None = None
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

    ``analysis_context`` binds caller-owned inputs before analysis. It
    returns metadata and full-file ``assets``; ``recovering=True`` asks
    for the original frozen inputs, without acquiring replacements.

    ``moment_policy``/``moment_repair``/``mp_physics`` are the
    multi-moment contract (:mod:`gpuwm.da.moments`), threaded to the one
    module that writes an analysis.  The defaults refuse an update that
    moves a multi-moment species' mass while leaving the moment the
    background carries, and repair any broken pair the analysis produces
    anyway through the scheme's own limiter.  ``mp_physics`` stays
    optional -- a guard that could be disabled by not passing a config is
    a guard that will be -- but leaving it unset no longer picks a
    limiter: the detected spellings identify a scheme only when exactly
    one registered scheme carries them, and an ambiguous structure is now
    refused by name rather than repaired through Morrison's (audit
    R-016).  Passing it also moves the moment-row resolution to plan
    review, before the first leg integrates.
    """
    root = Path(ens_root)
    if type(first_cycle) is not int or first_cycle < 0:
        raise ValueError('first_cycle must be a nonnegative integer')
    if first_cycle and initial_restarts is None:
        raise ValueError('A later cycle needs the complete prior analysis roster; recover the preceding cycle before continuing')
    if not first_cycle and initial_restarts:
        raise ValueError('The initial cycle cannot also resume a later analysis clock')
    if initial_restarts is not None and (any(type(i) is not int for i in initial_restarts)
                                         or set(initial_restarts) != set(range(cfg.n_members))):
        raise ValueError('The initial restart roster must contain every configured member')
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
    if not (cycle_seconds > 0.0):
        raise ValueError(
            f"cycle_seconds must be positive, got {cycle_seconds!r}")

    if assimilate is not None and mp_physics is not None:
        # PLAN REVIEW FOR THE CYCLE DOOR.  The analysis reads the scheme's
        # moment structure, and it reads it for the first time AFTER the
        # first forecast leg has integrated (gpuwm/ensemble/increments.py).
        # A scheme with no row therefore cost a whole leg before saying so,
        # which is the same shape as the 59-minute checkpoint loss this
        # audit started from (R-016).  Resolving it here costs a launch.
        from gpuwm.da.moments import scheme_moments

        scheme_moments(int(mp_physics))

    binding = cycle_binding(cfg, cycle_seconds=cycle_seconds,
                            n_cycles=n_cycles, positivity=positivity,
                            restart_from_analysis=restart_from_analysis)
    method_binding, _ = _method_identity(assimilate) if assimilate is not None else (None, None)
    binding['analysis'] = dict(enabled=assimilate is not None, method=method_binding,
        declared_method=dict(assimilation_method or {}), moment_policy=moment_policy,
        moment_repair=bool(moment_repair), mp_physics=mp_physics)
    if first_cycle or initial_restarts is not None or input_binding is not None:
        from gpuwm.output_identity import file_record
        binding['window'] = dict(first_cycle=first_cycle,
            inputs=dict(input_binding or {}),
            prior=[dict(member=i, artifact=file_record(path))
                   for i, path in sorted((initial_restarts or {}).items())])
        _, prior_clocks = _leg_horizon(initial_restarts, first_cycle, cycle_seconds)
        if prior_clocks.get('unstated_members'):
            raise ValueError('A continued window needs every prior checkpoint clock; recover the complete original checkpoints')
    from gpuwm.ensemble.analysis_commit import canonical
    canonical(binding)
    manifest_path = root / CYCLE_MANIFEST_NAME
    if manifest_path.is_file():
        manifest = read_manifest(manifest_path, schema=CYCLE_MANIFEST_SCHEMA)
        _check_cycle_compatible(manifest, binding, manifest_path)
    else:
        manifest = new_cycle_manifest(
            cfg, ens_root=root, cycle_seconds=cycle_seconds,
            n_cycles=n_cycles, positivity=positivity,
            restart_from_analysis=restart_from_analysis)
        manifest['cycle_binding'] = binding
        write_manifest_atomically(manifest_path, manifest)

    # A retry validates current inputs and policies before finishing a
    # publication. Earlier legs are checked before any later restart read.
    entries = {int(entry["cycle"]): entry
               for entry in manifest.get("cycles", ())}
    done = {index for index, entry in entries.items()
            if entry.get("status") == "DONE"}
    ran = []
    for cycle_index in range(first_cycle, first_cycle + n_cycles):
        if cycle_index in done:
            recorded = entries[cycle_index].get('assimilation')
            if (recorded is not None) != (assimilate is not None):
                raise ValueError('The completed cycle used a different assimilation mode; restore its configuration before resuming')
            if recorded is not None:
                leg = cycle_root(root, cycle_index)
                states = _leg_member_states(leg)
                context = _publication_context(assimilate, cycle_index, states, leg,
                    positivity=positivity, declared_method=assimilation_method,
                    moment_policy=moment_policy, moment_repair=moment_repair,
                    mp_physics=mp_physics, run_binding=binding,
                    owner=analysis_context, recovering=True)
                from gpuwm.ensemble import analysis_commit
                recovered = analysis_commit.recover(leg, publish=publish_staged_analysis, context=context)
                if recovered is None:
                    _verify_legacy_completed(leg, states, recorded, context)
                elif recovered != recorded:
                    raise ValueError('The completed cycle receipt differs from its immutable analysis decision; restore the original receipt')
            continue
        leg_root = cycle_root(root, cycle_index)
        _emit(on_event, {"event": "cycle-started", "cycle": cycle_index})
        restarts = (initial_restarts if cycle_index == first_cycle and first_cycle else
                    _analysis_restarts(root, cycle_index,
                                       n_members=cfg.n_members,
                                       required=restart_from_analysis))
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
            # accurate, outcome for a forecast-only cycle.
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
                mp_physics=mp_physics, run_binding=binding,
                context_owner=analysis_context)
        entry["status"] = "DONE"
        manifest["status"] = ("COMPLETE" if cycle_index == first_cycle + n_cycles - 1
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


def recover_analysis_publication(leg_root: str | Path, *, on_event=None) -> dict | None:
    """Verify a durable decision before completing its member publication."""
    from gpuwm.ensemble import analysis_commit
    leg = Path(leg_root)
    marker = publication_marker_path(leg)
    if not marker.exists() and not (leg / analysis_commit.COMMIT_NAME).exists():
        return None
    import json
    payload = json.loads(marker.read_text()) if marker.is_file() else {}
    if payload.get('schema') == PUBLICATION_MARKER_SCHEMA:
        raise ValueError(f'{marker}: this legacy publication lacks member byte identities and its method receipt. '
                         'Preserve the analyses and restore their original complete receipt before recovery.')
    before = [row['member'] for row in payload.get('members', ())
              if (leg / row['analysis']).is_file()]
    receipt = analysis_commit.recover(leg, publish=publish_staged_analysis)
    report = dict(cycle=receipt['cycle'], receipt=receipt, status='COMMITTED',
                  already_live=before, rolled_forward=[i for i in range(receipt['member_count']) if i not in before])
    if report['rolled_forward']:
        _emit(on_event, {'event': 'publication-recovered', **report})
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
    recover_analysis_publication(leg, on_event=on_event)
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
    if isinstance(recorded, Mapping) and recorded.get('window') != binding.get('window'):
        raise ValueError(f'{path} belongs to a different window or forcing generation; restore the original inputs before recovery')
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


def _method_identity(assimilate):
    """The callable and bytecode, excluding source-location metadata."""
    import hashlib
    import marshal
    import types
    target = getattr(assimilate, '__func__', assimilate)
    def normalized(code):
        return code.replace(co_filename='', co_firstlineno=0, co_linetable=b'',
            co_consts=tuple(normalized(value) if isinstance(value, types.CodeType) else value
                            for value in code.co_consts))
    code = getattr(target, '__code__', None)
    return dict(callable=_callable_path(assimilate),
                implementation_sha256=(hashlib.sha256(marshal.dumps(normalized(code))).hexdigest()
                                       if code is not None else None)), None


def _leg_member_states(leg):
    document = read_manifest(leg / ENSEMBLE_MANIFEST_NAME,
                             schema=ENSEMBLE_MANIFEST_SCHEMA)
    if document.get('status') != 'COMPLETE':
        raise ValueError(f'{leg}: the forecast roster is incomplete; recover every member before analysis publication')
    return {int(row['index']): dict(member_dir=str(leg / row['member_dir']),
        state_sha256=row.get('final_state_sha256'), seed=row.get('seed'))
        for row in document['members']}


def _publication_context(assimilate, cycle_index, member_states, leg_root, *,
                         positivity, declared_method, moment_policy,
                         moment_repair, mp_physics, run_binding,
                         owner, recovering):
    from gpuwm.output_identity import file_record
    from gpuwm.ensemble.state_sha import checkpoint_state_sha256
    from gpuwm.ensemble.analysis_commit import canonical
    import json
    indices = sorted(member_states)
    if indices != list(range(len(indices))) or any(type(i) is not int for i in indices):
        raise ValueError('The analysis requires the complete ordered member roster; restore the missing member records')
    backgrounds = []
    assets = []
    for index in indices:
        info = member_states[index]
        member_dir = Path(info['member_dir']).resolve()
        if member_dir != Path(leg_root).resolve() / member_directory_name(index):
            raise ValueError('The analysis member path belongs to another leg; restore the canonical member roster')
        path = _member_background_checkpoint(member_dir)
        actual = checkpoint_state_sha256(path)
        if info.get('state_sha256') is not None and info['state_sha256'] != actual:
            raise ValueError(f'{path}: the background state differs from its forecast receipt; restore the completed checkpoint')
        record = file_record(path)
        assets.append(record)
        backgrounds.append(dict(index=index, seed=info.get('seed'), state_sha256=actual,
                                checkpoint=record))
    extra = dict(owner(cycle_index, member_states, recovering=recovering)) if owner else {}
    assets.extend(extra.pop('assets', ()))
    method, _ = _method_identity(assimilate)
    context = dict(leg_root=str(Path(leg_root).resolve()), cycle=int(cycle_index),
        members=indices, backgrounds=backgrounds, assets=assets,
        positivity=positivity, moment_policy=moment_policy,
        moment_repair=bool(moment_repair), mp_physics=mp_physics,
        method=method, declared_method=dict(declared_method or {}),
        run_binding=run_binding, owner=extra)
    return json.loads(canonical(context))


def _verify_legacy_completed(leg, states, receipt, context):
    """Preserve an older completed analysis only against facts it recorded."""
    import numpy as np
    from gpuwm.ensemble.state_sha import checkpoint_state_sha256
    if receipt.get('status') != 'APPLIED' or receipt.get('member_count') != len(states):
        raise ValueError(f'{leg}: the older completed receipt has no complete analysis roster; restore its original receipt')
    for key in ('moment_policy', 'moment_repair'):
        if key in receipt and receipt[key] != context[key]:
            raise ValueError(f'{leg}: the completed analysis used a different {key}; restore its original policy')
    if receipt.get('positivity_policy') != context['positivity']:
        raise ValueError(f'{leg}: the completed analysis used another positivity policy; restore it before resuming')
    if (receipt.get('method') or {}).get('callable') != context['method']['callable']:
        raise ValueError(f'{leg}: the completed analysis records another method; restore its original callable')
    method = receipt.get('method') or {}
    if method.get('declared_by') == 'caller' and method.get('provenance') != context['declared_method']:
        raise ValueError(f'{leg}: the completed analysis records different declared method settings; restore its original declaration')
    window = context['owner'].get('observation_window_sha256')
    if window is not None:
        from gpuwm.ensemble.analysis_commit import digest
        recorded_windows = [row['frozen_window'] for row in (method.get('provenance') or {}).get('routes', ())
                            if isinstance(row, dict) and 'frozen_window' in row]
        if len(recorded_windows) != 1 or digest(recorded_windows[0]) != window:
            raise ValueError(f'{leg}: the original observation window does not match the completed method receipt; restore that window and receipt')
    rows = receipt.get('receipts', ())
    if [row.get('member') for row in rows] != list(states):
        raise ValueError(f'{leg}: the completed analysis receipt is missing members; restore the complete receipt')
    analyses = read_analysis_roster(leg, n_members=len(states))
    for row in rows:
        member = Path(states[row['member']]['member_dir'])
        analysis = analyses[row['member']]
        background = _member_background_checkpoint(member)
        if (checkpoint_state_sha256(analysis) != row.get('state_sha256_after')
                or checkpoint_state_sha256(background) != row.get('state_sha256_before')):
            raise ValueError(f'{member}: analysis or background state differs from the completed receipt; restore the original bytes')
        with np.load(background, allow_pickle=False) as before, np.load(analysis, allow_pickle=False) as after:
            keys = [key for key in before.files if not key.startswith('state/')]
            def same_metadata(key):
                left, right = before[key], after[key]
                return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
            if set(before.files) != set(after.files) or not all(same_metadata(key) for key in keys):
                raise ValueError(f'{analysis}: the carried checkpoint metadata differs from its background; restore the original analysis')


def _assimilate_cycle(assimilate, cycle_index, member_states, *,
                      leg_root, positivity="mean-preserving", on_event=None,
                      declared_method=None, attempt=1,
                      moment_policy="full-moment", moment_repair=True,
                      mp_physics=None, run_binding=None, context_owner=None) -> dict:
    """Return the original durable decision or stage and commit one analysis."""

    from gpuwm.da.positivity import POLICIES

    if positivity not in POLICIES:
        raise ValueError(
            f"unknown positivity policy {positivity!r}; known policies are "
            f"{POLICIES}")
    from gpuwm.ensemble import analysis_commit
    marker = publication_marker_path(leg_root)
    existing = [Path(info['member_dir']) / ANALYSIS_NAME for info in member_states.values()
                if (Path(info['member_dir']) / ANALYSIS_NAME).exists()]
    if existing and not marker.exists():
        raise ValueError(f'cycle {cycle_index}: analyses exist without their method receipt; '
                         'preserve these files and restore the original complete receipt instead of recomputing them')
    context = _publication_context(assimilate, cycle_index, member_states, leg_root,
        positivity=positivity, declared_method=declared_method, moment_policy=moment_policy,
        moment_repair=moment_repair, mp_physics=mp_physics, run_binding=run_binding,
        owner=context_owner, recovering=marker.exists())
    previous = analysis_commit.recover(leg_root, publish=publish_staged_analysis, context=context)
    if previous is not None:
        return previous
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
    ensemble_bounded = None
    if positivity == "mean-preserving" and len(increments_by_member) >= 2:
        # The mean the rule keeps is the ensemble's, so the rule needs the
        # whole ensemble at once; member by member it would be the clip.
        ensemble_bounded = _enforce_positivity_ensemble(
            {int(index): _member_background_checkpoint(
                Path(member_states[int(index)]["member_dir"]))
             for index in increments_by_member},
            {int(index): increments
             for index, increments in increments_by_member.items()})
    try:
        for index, increments in sorted(increments_by_member.items()):
            info = member_states[int(index)]
            background = _member_background_checkpoint(
                Path(info["member_dir"]))
            analysis = Path(info["member_dir"]) / ANALYSIS_NAME

            if ensemble_bounded is not None:
                increments, positivity_receipt = ensemble_bounded[int(index)]
            else:
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

    # The complete prospective receipt precedes every member rename.
    for receipt in receipts:
        receipt['published'] = True
        receipt.pop('staged', None)
    decision = {
        'cycle': int(cycle_index),
        "status": "APPLIED",
        "stability": "experimental",
        "member_count": len(receipts),
        "attempt": int(attempt),
        "commit": "three-phase: every member staged, an immutable decision and receipt declared, then all published; recovery returns that original receipt",
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
        "mass_added_total": sum(
            float(entry.get("mass_added", 0.0))
            for entry in positivity_receipts),
        "receipts": receipts,
    }

    analysis_commit.begin(leg_root, context=context, receipt=decision,
        pairs=staged, staged_suffix=STAGED_SUFFIX, analysis_name=ANALYSIS_NAME)
    def publish(path, analysis):
        publish_staged_analysis(path, analysis)
        index = next(i for i, _, target in staged if target.resolve() == analysis.resolve())
        _emit(on_event, {'event': 'member-assimilated', 'cycle': cycle_index, 'index': index})
    return analysis_commit.recover(leg_root, publish=publish, context=context)


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


def _enforce_positivity_ensemble(backgrounds, increments_by_member):
    """The ``mean-preserving`` policy over every member at once.

    ``backgrounds`` maps member index to its background checkpoint and
    ``increments_by_member`` to its increment mapping.  Each constrained
    field every member's increment carries is stacked on a leading member
    axis and bounded in one call, so the mean the rule keeps is the
    ensemble's.  A constrained field only some members carry is bounded
    member by member (the clip), and the receipt says so.  Returns
    ``{index: (increments, receipt)}``.
    """
    import numpy as np

    from gpuwm.da.positivity import (apply_positivity, constrained_fields,
                                     verify_non_negative)

    order = sorted(increments_by_member)
    per_member = {index: dict(increments_by_member[index]) for index in order}
    shared = None
    for index in order:
        names = set(constrained_fields(tuple(per_member[index])))
        shared = names if shared is None else (shared & names)
    shared = tuple(sorted(shared or ()))
    priors: dict[int, dict] = {index: {} for index in order}
    for index in order:
        wanted = constrained_fields(tuple(per_member[index]))
        if not wanted:
            continue
        with np.load(backgrounds[index], allow_pickle=False) as data:
            for name in wanted:
                key = f"state/{name}"
                if key not in data.files:
                    raise ValueError(
                        f"the analysis proposes an increment to {name!r}, "
                        "which the background checkpoint "
                        f"{backgrounds[index].name} does not carry; "
                        "positivity cannot be enforced against a "
                        "background that is not there")
                priors[index][name] = data[key]
    receipts: dict[int, dict] = {}
    if shared:
        stacked_prior = {name: np.stack([priors[i][name] for i in order])
                         for name in shared}
        stacked_increment = {
            name: np.stack([np.asarray(per_member[i][name]) for i in order])
            for name in shared}
        bounded, ensemble_receipt = apply_positivity(
            stacked_prior, stacked_increment, policy="mean-preserving")
        verify_non_negative(stacked_prior, bounded)
        for slot, index in enumerate(order):
            for name in shared:
                per_member[index][name] = bounded[name][slot]
        for index in order:
            receipts[index] = {
                key: value for key, value in ensemble_receipt.items()
                if key != "per_field"}
            receipts[index]["scope"] = "ensemble"
            receipts[index]["per_field_ensemble"] = ensemble_receipt[
                "per_field"]
    for index in order:
        rest = tuple(name for name in constrained_fields(
            tuple(per_member[index])) if name not in shared)
        if not rest:
            if index not in receipts:
                receipts[index] = {
                    "schema": "gpuwm-da.positivity.v1",
                    "policy": "mean-preserving",
                    "constrained_fields": [], "negative_points": 0,
                    "mass_added": 0.0, "mass_added_by_clip": 0.0,
                    "note": ("no field in this increment set is bounded "
                             "below; positivity had nothing to enforce")}
            continue
        prior = {name: priors[index][name] for name in rest}
        adjusted, member_receipt = apply_positivity(
            prior, {name: per_member[index][name] for name in rest},
            policy="mean-preserving")
        verify_non_negative(prior, adjusted)
        per_member[index].update(adjusted)
        if index in receipts:
            receipts[index]["member_only_fields"] = member_receipt
        else:
            receipts[index] = dict(member_receipt, scope="member")
    # Each member's share of the ensemble figures: the receipt totals are
    # summed over members by the decision record, so the ensemble's own
    # counts are divided once rather than counted once per member.
    members = len(order)
    for index in order:
        receipt = receipts[index]
        if receipt.get("scope") == "ensemble":
            for key in ("negative_points", "mass_added",
                        "mass_redistributed"):
                if key in receipt:
                    receipt[f"ensemble_{key}"] = receipt[key]
            receipt["negative_points"] = 0
            receipt["mass_added"] = 0.0
            if index == order[0]:
                receipt["negative_points"] = int(
                    receipt["ensemble_negative_points"])
                receipt["mass_added"] = float(receipt["ensemble_mass_added"])
            receipt["members"] = members
            extra = receipt.get("member_only_fields")
            if extra:
                receipt["negative_points"] += int(extra["negative_points"])
                receipt["mass_added"] += float(extra.get("mass_added", 0.0))
    return {index: (per_member[index], receipts[index]) for index in order}


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
