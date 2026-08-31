"""The speedrun: one named course, run end to end, sealed into one capsule.

THE CLOCK STARTS WHEN THE BYTES ARE STAGED.  Fetching is not part of a
run -- nobody benchmarks their ISP -- so a course names the staged inputs
it consumes by digest and the measured thing is staged-inputs to
finished-products: preparation, forecast, render.

Everything here exists because a benchmark that can be gamed is not an
instrument.  Four specific ways a weather-model time can lie, and the
answer this module gives to each:

**Comparing runs of different work.**  A record is comparable to another
record only when the course id, the resolved course row, the product set
and the compile mode all match.  Those four are hashed into one
``comparability.key``, the key is inside the sealed body, and every
comparison door in this module calls :func:`assert_comparable` first.
Editing a capsule to force a match breaks the seal, so the two halves
close each other: an honest capsule refuses the comparison by name, and a
doctored one refuses to be read at all.  That is the difference between
discouraging an incomparable comparison and preventing it.

**A fast run that did no work.**  A NaN-filled sprint to a blank PNG is
the fastest possible forecast.  :func:`evidence_verdict` demands positive
evidence instead of absence of error: frames actually written, every
product the course declares actually rendered, every product file
non-empty with a real digest, and the run's own forecast-validity verdict
PASS.  A record that cannot show all four is VOID and cannot be compared.

**The invisible minute.**  The first GPU run on a machine pays roughly a
minute of NVRTC kernel compilation (see
:mod:`gpuwm.kernel_compile_notice`).  Including it silently makes a warm
box look fast; excluding it silently makes the number a user will never
see.  So: the compile is ALWAYS inside the clock, the capsule says so in
``compile.included_in_clock``, the seconds are broken out as their own
named stage from the run's own ``progress.jsonl``, and ``compile.mode``
is part of the comparability key -- a cold record and a warm record are
different records and this module will not compare them.  The door
measures the cache state before the clock starts and refuses a mismatch
against what the course declared.

**A determinism claim nobody screened.**  These cards have no ECC, so
"deterministic" is a measurement, not a property.  A capsule's
``determinism`` block starts at ``not_screened`` with a ``null`` claim,
and only :func:`determinism_screen` -- given the second arm of a real
dual run -- may set it.  A capsule that claims determinism without the
screen is VOID.

Courses are TABLE DATA (``gpuwm/speedrun_courses.json`` plus assets under
``configs/speedrun/``).  Adding one is a row and two files; there is no
per-course code path here and the test suite fails if one appears.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from gpuwm import stage_timing

#: Schema id carried by every capsule this module seals.
CAPSULE_SCHEMA_ID = "gpuwm.speedrun-capsule/v1"

#: Schema id of the course table.
COURSE_TABLE_SCHEMA_ID = "gpuwm.speedrun-courses/v1"

#: Artifact name, fixed so every consumer looks in one place.
CAPSULE_FILENAME = "speedrun-capsule.json"

#: Where the course table lives.  Ships in the wheel via ``gpuwm/*.json``.
COURSE_TABLE_PATH = Path(__file__).resolve().parent / "speedrun_courses.json"

#: Directory under the repository's ``configs/`` holding course assets.
COURSE_ASSET_DIR = "speedrun"

#: How the capsule is sealed.  A CONTENT seal: it makes tampering
#: evident, and it is not a signature -- it proves the bytes have not
#: moved since the run wrote them, not who wrote them.  Naming it
#: "sha256-canonical-json" rather than "signature" is the honest label.
SEAL_ALGORITHM = "sha256-canonical-json"

#: The two compile modes a course may declare.
COMPILE_MODES = ("cold", "warm")

#: Evidence verdicts.
VALID = "VALID"
VOID = "VOID"

#: The stage a record puts its unattributed remainder in.  A record whose
#: ONLY named stage is this one is VOID: it has a total and nothing to
#: say about where the total went, which is exactly the failure
#: :mod:`gpuwm.stage_timing` exists to catch and which a single
#: catch-all stage would slip past the coverage bar at 100%.
ORCHESTRATION_STAGE = "orchestration"

#: How far past its own clock a record's named stages may sum before the
#: record is VOID.  Slightly over 1.0 because stages may overlap by a few
#: milliseconds of bookkeeping; well under the case this catches --
#: MEASURED 2026-08-20, a 4:45 record that claimed a 13:26 forecast
#: because it read three earlier runs' stages out of one appended stream.
OVER_ATTRIBUTION_CEILING = 1.05

#: Determinism screen states.
NOT_SCREENED = "not_screened"
SCREENED = "screened"

#: The comparability fields, in the order the key hashes them.  A record
#: is comparable to another record exactly when all four agree.
COMPARABILITY_FIELDS = (
    "course_id",
    "course_sha256",
    "product_set_sha256",
    "compile_mode",
)

#: Filled in lazily by :func:`load_course_table`; tests replace it to
#: prove a course is a row rather than a code path.
_COURSE_TABLE_CACHE: dict[str, dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Refusals.  Every one names the breakage it prevents AND the remedy.
# ---------------------------------------------------------------------------

class SpeedrunRefusal(RuntimeError):
    """A speedrun door refused.  Never a traceback at a user."""


class CourseUnknown(SpeedrunRefusal):
    """The named course is not in the table."""


class CourseAssetsMissing(SpeedrunRefusal):
    """A course row names a config or namelist this machine does not have."""


class SealBroken(SpeedrunRefusal):
    """A capsule's bytes do not match its own seal."""


class Incomparable(SpeedrunRefusal):
    """Two records do not describe the same work."""


class RecordVoid(SpeedrunRefusal):
    """A record has no positive evidence that the work happened."""


class CompileModeMismatch(SpeedrunRefusal):
    """The kernel cache is not in the state the course declared."""


class StagedInputsMissing(SpeedrunRefusal):
    """The staged bytes a course consumes are not where the run points."""


# ---------------------------------------------------------------------------
# Canonical bytes and digests
# ---------------------------------------------------------------------------

def canonical_bytes(document: Any) -> bytes:
    """One document, one byte string, on every platform.

    Sorted keys, no insignificant whitespace, UTF-8, and ``\\n`` nowhere
    -- so a capsule written on Windows and a capsule written on Linux
    hash the same when they say the same thing.  A seal that moved when
    a file crossed an operating system would refuse every honest record
    that travelled.
    """

    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def digest_document(document: Any) -> str:
    """The sha256 of :func:`canonical_bytes`."""

    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def digest_file(path: str | Path) -> str:
    """The sha256 of a file's bytes."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The course table
# ---------------------------------------------------------------------------

def load_course_table() -> dict[str, dict[str, Any]]:
    """Every course, keyed by id.

    Read once and cached.  The cache is module state on purpose: it is
    the seam a test replaces to prove that a course is a table row and
    not a code path.
    """

    global _COURSE_TABLE_CACHE
    if _COURSE_TABLE_CACHE is None:
        payload = json.loads(COURSE_TABLE_PATH.read_text(encoding="utf-8"))
        if payload.get("schema") != COURSE_TABLE_SCHEMA_ID:
            raise SpeedrunRefusal(
                f"{COURSE_TABLE_PATH} declares schema "
                f"{payload.get('schema')!r}, not {COURSE_TABLE_SCHEMA_ID!r}; "
                "this build cannot read that table.  Remedy: use the "
                "gpuwm release the table was written for, or regenerate "
                "the table for this one.")
        _COURSE_TABLE_CACHE = {row["id"]: row for row in payload["courses"]}
    return _COURSE_TABLE_CACHE


def course(course_id: str) -> dict[str, Any]:
    """One course row, or a refusal that lists the ones that exist."""

    table = load_course_table()
    try:
        return table[course_id]
    except KeyError:
        known = "\n".join(
            f"  {cid:24s} {row['title']}" for cid, row in sorted(table.items()))
        raise CourseUnknown(
            f"there is no speedrun course called {course_id!r}, so there is "
            "nothing to run and no leaderboard column to land in.\n\n"
            f"The {len(table)} course(s) this build carries:\n{known}\n\n"
            "Remedy: run one of those ids, or `gpuwm speedrun --list` for "
            "the full description of each.  A new course is a row in "
            f"{COURSE_TABLE_PATH.name} plus its two asset files -- no code "
            "change.") from None


def course_assets(course_id: str) -> dict[str, Path]:
    """The resolved files a course needs, or a refusal naming the missing one.

    Resolution goes through the same ladder every other repository config
    uses (``GPUWM_CONFIGS_ROOT`` then the ``configs/`` beside the
    package), so a wheel install and a checkout find the same files.
    """

    # Imported here, not at module top: gpuwm.cli imports this module to
    # register the speedrun door, and enumerating cases must not import
    # anything under gpuwm.verify.cases (tests/test_case_registry.py) --
    # the property that keeps `gpuwm --help` free of case import side
    # effects.
    from gpuwm.verify.cases import _repo_config

    row = course(course_id)
    resolved: dict[str, Path] = {}
    for key in ("experiment_config", "wps_namelist"):
        name = f"{COURSE_ASSET_DIR}/{row[key]}"
        found = _repo_config.locate(name)
        if found is None:
            raise CourseAssetsMissing(
                f"course {course_id!r} needs {name}, and this machine has "
                "no copy of it -- the course cannot be run and any time "
                "measured without it would be a different course.\n\n"
                + _repo_config.missing_config_message(
                    name, flag="--experiment-config"))
        resolved[key] = found
    return resolved


def course_digest(row: Mapping[str, Any]) -> str:
    """The digest of one resolved course row.

    Over the ROW, not the file: two builds whose tables differ only in
    the ordering of rows or in a course nobody ran still agree about the
    course that was run.
    """

    return digest_document(dict(row))


def product_set_digest(products: Iterable[str]) -> str:
    """The digest of a product set.

    Sorted and de-duplicated first: the set is the thing that matters,
    and a record must not become incomparable because a caller listed the
    same products in a different order.
    """

    return digest_document(sorted(set(str(p) for p in products)))


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------

def comparability_block(capsule: Mapping[str, Any]) -> dict[str, Any]:
    """The four fields that decide comparability, plus their key."""

    fields = {
        "course_id": capsule["course"]["id"],
        "course_sha256": capsule["course"]["course_sha256"],
        "product_set_sha256": capsule["course"]["product_set_sha256"],
        "compile_mode": capsule["compile"]["mode"],
    }
    ordered = {name: fields[name] for name in COMPARABILITY_FIELDS}
    return {
        "capsule_schema": CAPSULE_SCHEMA_ID,
        "fields": ordered,
        "key": digest_document([CAPSULE_SCHEMA_ID, ordered]),
    }


def comparability_key(capsule: Mapping[str, Any]) -> str:
    """One record's comparability key, as sealed."""

    return capsule["comparability"]["key"]


def assert_comparable(left: Mapping[str, Any],
                      right: Mapping[str, Any], *,
                      left_name: str = "the first record",
                      right_name: str = "the second record") -> None:
    """Refuse, by name, any comparison of records of different work.

    The seals are checked FIRST, and that ordering is the point: a
    capsule edited to make its key match is caught before its key is
    ever consulted, so the two guards cannot be played against each
    other.
    """

    verify_seal(left, what=left_name)
    verify_seal(right, what=right_name)
    if comparability_key(left) == comparability_key(right):
        return
    differing = [
        (name, left["comparability"]["fields"][name],
         right["comparability"]["fields"][name])
        for name in COMPARABILITY_FIELDS
        if left["comparability"]["fields"][name]
        != right["comparability"]["fields"][name]
    ]
    lines = [
        f"{left_name} and {right_name} are not records of the same work, so "
        "one cannot be faster than the other -- comparing them would "
        "publish a ranking of two different jobs.",
        "",
        "They disagree about:",
    ]
    for name, mine, theirs in differing:
        lines.append(f"  {name}:")
        lines.append(f"    {left_name:<20s} {mine}")
        lines.append(f"    {right_name:<20s} {theirs}")
    if not differing:
        lines.append(
            "  (no named field differs, which means one of these capsules "
            "was written by a build whose comparability rule is not this "
            "one)")
    lines += [
        "",
        "Remedy: compare each record against another record of ITS course "
        "and compile mode.  `gpuwm speedrun --list` shows the courses; "
        "SPEEDRUN.md keeps one table per course.",
    ]
    raise Incomparable("\n".join(lines))


# ---------------------------------------------------------------------------
# The seal
# ---------------------------------------------------------------------------

def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a content seal to a capsule body."""

    document = {key: value for key, value in body.items() if key != "seal"}
    document["seal"] = {
        "algorithm": SEAL_ALGORITHM,
        "body_sha256": digest_document(document),
    }
    return document


def verify_seal(capsule: Mapping[str, Any], *,
                what: str = "this capsule") -> None:
    """Refuse a capsule whose bytes have moved since it was written."""

    block = capsule.get("seal")
    if not isinstance(block, Mapping) or "body_sha256" not in block:
        raise SealBroken(
            f"{what} carries no seal, so nothing here can tell whether it "
            "is what a run wrote or what someone typed.  An unsealed "
            "document is not a record.\n\n"
            "Remedy: submit the speedrun-capsule.json the run itself "
            "emitted, unedited.")
    if block.get("algorithm") != SEAL_ALGORITHM:
        raise SealBroken(
            f"{what} is sealed with {block.get('algorithm')!r}; this build "
            f"can only verify {SEAL_ALGORITHM!r}, so it cannot say whether "
            "the bytes are intact.\n\n"
            "Remedy: verify it with the gpuwm release that wrote it.")
    document = {key: value for key, value in capsule.items() if key != "seal"}
    measured = digest_document(document)
    if measured != block["body_sha256"]:
        raise SealBroken(
            f"{what} does not match its own seal: it says its body hashes "
            f"to {block['body_sha256']}, and the body in front of us "
            f"hashes to {measured}.  Some field has been changed since the "
            "run wrote it, and a record whose numbers can be retyped is "
            "not a record.\n\n"
            "Remedy: re-run the course and submit the capsule the run "
            "emits.  If you meant to correct something the run measured "
            "wrongly, that is a defect report, not an edit.")


# ---------------------------------------------------------------------------
# Positive evidence
# ---------------------------------------------------------------------------

def evidence_verdict(capsule: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Whether this record shows the work happened, and why not if not.

    Absence of an error is not evidence.  Every clause below demands
    something POSITIVE, and a record missing any one of them is VOID --
    which is a refusal to rank it, not an accusation.
    """

    reasons: list[str] = []
    evidence = capsule.get("evidence", {})
    outputs = capsule.get("outputs", {})
    declared = sorted(set(capsule["course"]["products"]))

    frames = evidence.get("wrfout_frames")
    if not isinstance(frames, int) or frames < 1:
        reasons.append(
            f"evidence.wrfout_frames is {frames!r}: the forecast committed "
            "no history frame, so there is no forecast to have been fast at")

    rendered = sorted(set(evidence.get("products_rendered") or []))
    missing = [name for name in declared if name not in rendered]
    if missing:
        reasons.append(
            "the course declares products this record did not render, so "
            "it did less work than the course: "
            + ", ".join(missing))

    files = outputs.get("files") or []
    if not files:
        reasons.append(
            "outputs.files is empty: no product file was written, so the "
            "product-set digest witnesses nothing")
    for entry in files:
        if not entry.get("bytes"):
            reasons.append(
                f"product file {entry.get('relpath')!r} is 0 bytes: an "
                "empty file is not a rendered product")
        if not entry.get("sha256"):
            reasons.append(
                f"product file {entry.get('relpath')!r} carries no digest, "
                "so nothing proves what was in it")

    validity = evidence.get("forecast_validity")
    if validity != "PASS":
        reasons.append(
            f"the run's own forecast validity verdict is {validity!r}, not "
            "PASS: a run that did not produce a valid forecast has no time "
            "worth reading")

    stages = dict(capsule["clock"]["stages"])
    if not set(stages) - {ORCHESTRATION_STAGE}:
        reasons.append(
            f"every second of this record is in {ORCHESTRATION_STAGE!r}: no "
            "stage of the actual work -- preparation, forecast, render -- "
            "was attributed at all, so the record has a total and nothing "
            "to say about where it went")
    attributed = sum(float(value) for value in stages.values()
                     if isinstance(value, (int, float))
                     and not isinstance(value, bool))
    wall = float(capsule["clock"]["wall_seconds"])
    if attributed > wall * OVER_ATTRIBUTION_CEILING:
        reasons.append(
            f"the named stages account for {attributed:.1f} s, MORE than "
            f"this record's own {wall:.1f} s clock.  A record cannot spend "
            "longer on its parts than it took; these stage walls belong to "
            "some other run")
    timing = dict(stages)
    timing[stage_timing.TOTAL_KEY] = capsule["clock"]["wall_seconds"]
    shortfall = stage_timing.timing_coverage_shortfall(
        timing, what="this record's stage timing")
    if shortfall:
        reasons.append(shortfall)

    determinism = capsule.get("determinism", {})
    if (determinism.get("claim") is not None
            and determinism.get("status") != SCREENED):
        reasons.append(
            "the determinism claim is set but determinism.status is "
            f"{determinism.get('status')!r}: these cards have no ECC, so a "
            "determinism claim is only ever the result of a dual-run byte "
            "comparison, never an assertion")

    return (VOID if reasons else VALID), reasons


def assert_valid_record(capsule: Mapping[str, Any], *,
                        what: str = "this record") -> None:
    """Refuse a VOID record, listing every reason it is void."""

    status, reasons = evidence_verdict(capsule)
    if status == VALID:
        return
    listed = "\n".join(f"  * {reason}" for reason in reasons)
    raise RecordVoid(
        f"{what} is VOID: it does not show that the work the course names "
        f"actually happened, so its wall clock cannot be ranked.\n\n"
        f"{listed}\n\n"
        "Remedy: re-run the course to completion.  A VOID record is still "
        "worth keeping as a defect report -- if the run finished and the "
        "evidence is missing anyway, that is a hole in the instrument.")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(left: Mapping[str, Any], right: Mapping[str, Any], *,
            left_name: str = "the first record",
            right_name: str = "the second record") -> dict[str, Any]:
    """Two records of the same work, side by side.

    Seals, then comparability, then evidence -- in that order, so the
    reason a comparison is refused is always the first thing wrong with
    it rather than whichever check happened to run.
    """

    assert_comparable(left, right, left_name=left_name, right_name=right_name)
    assert_valid_record(left, what=left_name)
    assert_valid_record(right, what=right_name)
    stages = sorted(set(left["clock"]["stages"]) | set(right["clock"]["stages"]))
    return {
        "course_id": left["course"]["id"],
        "compile_mode": left["compile"]["mode"],
        "comparability_key": comparability_key(left),
        "left": _record_summary(left, left_name),
        "right": _record_summary(right, right_name),
        "wall_seconds_delta": round(
            right["clock"]["wall_seconds"] - left["clock"]["wall_seconds"], 3),
        "stage_seconds_delta": {
            name: round(float(right["clock"]["stages"].get(name, 0.0))
                        - float(left["clock"]["stages"].get(name, 0.0)), 3)
            for name in stages
        },
        "products_identical": (left["outputs"]["product_set_sha256"]
                               == right["outputs"]["product_set_sha256"]),
    }


def _record_summary(capsule: Mapping[str, Any], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "device": capsule["device"]["name"],
        "gpuwm_version": capsule["engine"]["gpuwm_version"],
        "git_commit": capsule["engine"]["git_commit"],
        "wall_seconds": capsule["clock"]["wall_seconds"],
        "stages": dict(capsule["clock"]["stages"]),
        "capsule_sha256": capsule["seal"]["body_sha256"],
    }


# ---------------------------------------------------------------------------
# Cold vs warm
# ---------------------------------------------------------------------------

def measured_compile_mode(entries_for_this_card: int) -> str:
    """``cold`` when this card has nothing cached, ``warm`` when it does.

    Keyed on entries FOR THIS CARD, not on entries: CuPy's cache key
    mixes in the target architecture, so a cache full of another card's
    binaries recompiles everything and is cold in every way that costs
    time.  MEASURED 2026-08-16 on this project's reference box, where a
    7,164-entry cache from a different card read as warm and 51 s of
    compilation ran silently inside model step 1.
    """

    return "warm" if int(entries_for_this_card) > 0 else "cold"


def assert_compile_mode(declared: str, *, measured: str, cache_dir: Path,
                        entries: int, entries_for_this_card: int) -> None:
    """Refuse a run whose cache is not in the state the course declared.

    Checked BEFORE the clock starts, because the whole cost of getting
    this wrong lands inside the first measured stage and cannot be
    recovered afterwards.
    """

    if declared not in COMPILE_MODES:
        raise CompileModeMismatch(
            f"compile mode {declared!r} is not one of {list(COMPILE_MODES)}; "
            "a record has to say which side of the one-time NVRTC compile "
            "its clock is on.  Remedy: pass --compile-mode cold or "
            "--compile-mode warm.")
    if declared == measured:
        return
    if declared == "cold":
        raise CompileModeMismatch(
            "this course is a COLD-cache record and this machine's kernel "
            f"cache is WARM: {cache_dir} holds {entries} entry(s), "
            f"{entries_for_this_card} of them compiled for this card.  A "
            "cold record run on a warm cache skips roughly a minute of "
            "NVRTC compilation that a cold record includes, so the time "
            "would be a warm time on a cold leaderboard.\n\n"
            "Remedies, either one:\n"
            f"  * run the warm class instead:  --compile-mode warm\n"
            f"  * or empty the cache and pay the compile:  remove "
            f"{cache_dir}, or point CUPY_CACHE_DIR at an empty directory "
            "for this run (--cold-cache-dir does exactly that).")
    raise CompileModeMismatch(
        "this course is a WARM-cache record and this machine's kernel cache "
        f"is COLD: {cache_dir} holds {entries} entry(s), "
        f"{entries_for_this_card} of them compiled for this card.  A warm "
        "record run on a cold cache pays roughly a minute of NVRTC "
        "compilation that no other warm record paid.\n\n"
        "Remedies, either one:\n"
        "  * run the cold class, which is what this machine is in:  "
        "--compile-mode cold\n"
        "  * or warm the cache first with one throwaway run of the same "
        "course, then submit the second run.")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def unscreened_determinism() -> dict[str, Any]:
    """The determinism block every fresh capsule starts with."""

    return {
        "status": NOT_SCREENED,
        "claim": None,
        "note": "no dual-run byte comparison was made, so this record "
                "makes no determinism claim.  These cards carry no ECC; "
                "the screen is to run the course a second time on the "
                "same machine and compare the two capsules with "
                "`gpuwm speedrun --determinism ARM_A ARM_B`.",
    }


def determinism_screen(arm_a: Mapping[str, Any],
                       arm_b: Mapping[str, Any]) -> dict[str, Any]:
    """The dual-run byte comparison, as a determinism block.

    The two arms must be records of the same work, which is why this
    goes through :func:`assert_comparable`: a "determinism" screen across
    two different courses measures nothing and would read as a failure.
    """

    assert_comparable(arm_a, arm_b, left_name="arm A", right_name="arm B")
    left = arm_a["outputs"]["product_set_sha256"]
    right = arm_b["outputs"]["product_set_sha256"]
    identical = left == right
    if identical:
        note = (f"two runs of this course on this machine produced the same "
                f"product set, sha256 {left}.  With no ECC on the card, "
                "that is the screen this project trusts.")
    else:
        note = (f"the two arms produced DIFFERENT product sets: arm A "
                f"{left}, arm B {right}.  On a card with no ECC that is "
                "either genuine non-determinism in the code or memory "
                "corruption, and neither one may be ranked as a "
                "deterministic record until it is chased down.")
    return {
        "status": SCREENED,
        "claim": bool(identical),
        "arm_a_capsule_sha256": arm_a["seal"]["body_sha256"],
        "arm_b_capsule_sha256": arm_b["seal"]["body_sha256"],
        "arm_a_product_set_sha256": left,
        "arm_b_product_set_sha256": right,
        "note": note,
    }


def determinism_cell(capsule: Mapping[str, Any]) -> str:
    """What a leaderboard prints in the determinism column."""

    block = capsule.get("determinism") or {}
    if block.get("status") != SCREENED:
        return "not screened"
    return "dual-run identical" if block.get("claim") else "dual-run DIVERGED"


# ---------------------------------------------------------------------------
# Building a capsule
# ---------------------------------------------------------------------------

def capsule_body(*, course_id: str, engine: Mapping[str, Any],
                 machine: Mapping[str, Any], device: Mapping[str, Any],
                 numerical_stack: Mapping[str, Any],
                 config: Mapping[str, Any],
                 compile_block: Mapping[str, Any],
                 clock: Mapping[str, Any],
                 evidence: Mapping[str, Any],
                 outputs: Mapping[str, Any],
                 determinism: Mapping[str, Any] | None = None
                 ) -> dict[str, Any]:
    """One unsealed capsule body, with its course and comparability filled in.

    The course block is built HERE from the table rather than accepted
    from the caller, so a capsule cannot describe a course whose row it
    did not resolve.
    """

    row = course(course_id)
    products = sorted(set(row["products"]))
    body: dict[str, Any] = {
        "schema": CAPSULE_SCHEMA_ID,
        "course": {
            "id": row["id"],
            "title": row["title"],
            "kind": row["kind"],
            "source": row["source"],
            "forecast_seconds": row["forecast_seconds"],
            "domains": row["domains"],
            "products": products,
            "product_set_id": row["product_set_id"],
            "product_set_sha256": product_set_digest(products),
            "course_sha256": course_digest(row),
        },
        "engine": dict(engine),
        "machine": dict(machine),
        "device": dict(device),
        "numerical_stack": dict(numerical_stack),
        "config": dict(config),
        "compile": dict(compile_block),
        "clock": dict(clock),
        "evidence": dict(evidence),
        "outputs": dict(outputs),
        "determinism": dict(determinism or unscreened_determinism()),
    }
    body["comparability"] = comparability_block(body)
    return body


def load_capsule(path: str | Path) -> dict[str, Any]:
    """Read one capsule and refuse it unless its seal holds."""

    path = Path(path)
    try:
        capsule = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SpeedrunRefusal(
            f"cannot read the capsule at {path}: {error}.  Remedy: point at "
            f"the {CAPSULE_FILENAME} a speedrun wrote.") from None
    except json.JSONDecodeError as error:
        raise SpeedrunRefusal(
            f"{path} is not JSON ({error}), so it is not a capsule.  "
            f"Remedy: point at the {CAPSULE_FILENAME} a speedrun wrote.") \
            from None
    if capsule.get("schema") != CAPSULE_SCHEMA_ID:
        raise SpeedrunRefusal(
            f"{path} declares schema {capsule.get('schema')!r}, not "
            f"{CAPSULE_SCHEMA_ID!r}; this build cannot verify it.  Remedy: "
            "verify it with the gpuwm release that wrote it.")
    verify_seal(capsule, what=str(path))
    return capsule


# ---------------------------------------------------------------------------
# The leaderboard row
# ---------------------------------------------------------------------------

def format_wall(seconds: float) -> str:
    """``M:SS`` for a leaderboard cell; ``H:MM:SS`` past an hour."""

    total = int(round(float(seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


#: Stages a leaderboard row shows in their own cells.  Everything else
#: on the clock is preparation.
HEADLINE_STAGES = ("forecast", "render", ORCHESTRATION_STAGE)


def preparation_seconds(stages: Mapping[str, Any]) -> float:
    """Everything on the clock that happens before the forecast.

    Not ``stages["prepare"]``: the two shipped doors name the
    pre-forecast phases differently -- rw-wps's preparation wall lands
    under ``prepare`` on the `gpuwm go` route and under ``manifest`` on
    the run-plan route, with ``authority`` and ``boot`` beside them.
    MEASURED on node-1, a tree record printed ``prepare 0:00`` over 6.3
    real seconds because the row read one key.  The reader means
    "everything before the model started", so that is what the cell is.
    """

    return sum(
        float(value) for name, value in stages.items()
        if name not in HEADLINE_STAGES
        and isinstance(value, (int, float)) and not isinstance(value, bool))


def record_row(capsule: Mapping[str, Any]) -> dict[str, str]:
    """One SPEEDRUN.md row, read off the capsule.

    Every cell comes from the sealed document.  Nothing here accepts a
    number a human typed, which is the only reason the table means
    anything.
    """

    status, _ = evidence_verdict(capsule)
    stages = capsule["clock"]["stages"]
    return {
        "course": capsule["course"]["id"],
        "wall": format_wall(capsule["clock"]["wall_seconds"]),
        "prepare": format_wall(preparation_seconds(stages)),
        "forecast": format_wall(stages.get("forecast", 0.0)),
        "render": format_wall(stages.get("render", 0.0)),
        "kernel_compile": (
            format_wall(capsule["compile"]["kernel_compile_seconds"])
            if capsule["compile"].get("kernel_compile_seconds") is not None
            else "not measured"),
        "compile": capsule["compile"]["mode"],
        "device": capsule["device"]["name"],
        "version": capsule["engine"]["gpuwm_version"],
        "commit": str(capsule["engine"]["git_commit"])[:9],
        "determinism": determinism_cell(capsule),
        "status": status,
        "capsule_sha256": capsule["seal"]["body_sha256"],
    }


__all__ = [
    "CAPSULE_FILENAME", "CAPSULE_SCHEMA_ID", "COMPARABILITY_FIELDS",
    "COMPILE_MODES", "COURSE_ASSET_DIR", "COURSE_TABLE_PATH",
    "ORCHESTRATION_STAGE", "OVER_ATTRIBUTION_CEILING",
    "COURSE_TABLE_SCHEMA_ID", "CompileModeMismatch", "CourseAssetsMissing",
    "CourseUnknown", "Incomparable", "NOT_SCREENED", "RecordVoid",
    "SCREENED", "SEAL_ALGORITHM", "SealBroken", "SpeedrunRefusal",
    "StagedInputsMissing", "VALID", "VOID", "assert_comparable",
    "assert_compile_mode", "assert_valid_record", "canonical_bytes",
    "capsule_body", "comparability_block", "comparability_key", "compare",
    "course", "course_assets", "course_digest", "determinism_cell",
    "determinism_screen", "digest_document", "digest_file",
    "HEADLINE_STAGES", "evidence_verdict", "format_wall", "load_capsule",
    "load_course_table", "preparation_seconds",
    "measured_compile_mode", "product_set_digest", "record_row", "seal",
    "unscreened_determinism", "verify_seal",
]
