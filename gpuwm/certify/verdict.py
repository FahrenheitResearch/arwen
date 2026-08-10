"""The certification verdict: every refusal, in one place, fail-closed.

``gpuwm certify`` takes a run capsule, a matched-comparison metrics CSV, an
acceptance band and a WRF reference manifest, and returns 0 only when every
declared condition holds.  Each condition is evaluated whether or not an
earlier one already failed, so a reader gets the whole list rather than the
first refusal and a rerun.

The verdict binds itself.  ``capsule_binding_sha256`` covers the closed
inventory of everything the decision rests on -- and deliberately not the
``passed`` boolean, so flipping that bit is inert: the hash does not move, and
:func:`rederive_verdict` still returns what the bound rows say.  Mutating a
bound row moves both.

The binding mirrors the score-binding shape the case lane already uses; it
does not import it.  Importing would put a case module on the certification
import path -- both the binding function and the canonical-JSON digest beside
it live on case modules -- so this module carries its own digest, under its
own name, and a test holds the import graph to that.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gpuwm.certify.band import (CLASSIFICATION_BANDED, ROW_KEY_COLUMNS,
                                BandError, place_value, validate_band)
from gpuwm.certify.capsule import (GEOGRAPHY_INVENTORY_ALGORITHM,
                                   validate_certification_capsule)
from gpuwm.certify.pins import unresolved_pins
from gpuwm.certify.wrf_reference import (absent_reference_hashes,
                                         reference_binding,
                                         validate_wrf_reference_manifest)

VERDICT_SCHEMA_ID = "gpuwm.certification-verdict/v1"
VERDICT_SCHEMA_VERSION = "1.1.0"

#: The declared refusal conditions, in the order the verdict reports them.
#: Every one is named in F3 AC4; adding a condition means adding it here, and
#: the verdict then carries it whether it fired or not.
CONDITIONS: tuple[str, ...] = (
    "capsule_validates",
    "band_config_identity_matches_capsule",
    "wrf_reference_hashes_present",
    "geography_input_hashed_by_content",
    "every_pin_resolved",
    "every_metrics_column_classified",
    "the_comparison_is_not_empty",
    "every_banded_row_inside_its_interval",
)

#: What each published verdict schema version declared, so an older
#: document can still be read.
#:
#: :func:`rederive_verdict` compares the condition set for exact equality:
#: a document that declares a condition this code does not know, or omits
#: one it does, is not a document this code can rederive.  That is the
#: right answer for a forgery and the wrong one for a verdict written
#: last week.  ``the_comparison_is_not_empty`` was added after 1.0.0, so
#: without this table every genuine 1.0.0 verdict became indistinguishable
#: from a tampered one -- the same silent ``False``, no version signal to
#: tell an independent verifier which they were holding.
#:
#: Reading an older set does NOT reopen what that version was missing.
#: :func:`rederive_verdict` enforces :data:`MINIMUM_BANDED_COMPARISONS`
#: itself, on every version, so a 1.0.0 verdict carrying no comparison
#: rederives False on the population check even though its own condition
#: list never asked the question.  An unknown version -- a future one, or
#: an invented one -- fails closed with its own sentence.
CONDITIONS_BY_SCHEMA_VERSION: Mapping[str, tuple[str, ...]] = {
    "1.0.0": tuple(name for name in CONDITIONS
                   if name != "the_comparison_is_not_empty"),
    "1.1.0": CONDITIONS,
}

#: The floor under "a comparison happened at all".
#:
#: ``every_banded_row_inside_its_interval`` is a universal quantifier, and
#: a universal quantifier over nothing is true.  A metrics CSV carrying its
#: header and no data rows therefore certified as PASS -- ``certify: PASS
#: (0 banded comparisons ...)``, exit 0 -- because no row failed.  So did a
#: CSV whose rows produced no banded comparison at all, which is the same
#: hole reached from the other side: rows present, nothing gated.  A
#: release read that as a green certification of a run nobody compared.
#:
#: One, not the band's full cell count.  The band declares intervals out to
#: its longest lead, and a run legitimately shorter than that produces
#: fewer rows than the band has cells; requiring full coverage would refuse
#: the ordinary partial comparison, which is a guard firing on the
#: legitimate case.  What is not legitimate, ever, is zero -- and the exact
#: counts travel in the verdict and in the refusal so a reader sees a
#: thin comparison rather than merely a non-empty one.
MINIMUM_METRICS_ROWS = 1
MINIMUM_BANDED_COMPARISONS = 1


class CertifyError(ValueError):
    """Certification could not be attempted at all."""


#: What each of ``gpuwm certify``'s four file arguments is, and how to
#: point it at the right file.  Keyed by the flag, because the flag is
#: what the operator typed and what they have to change.
CERTIFY_INPUTS: dict[str, tuple[str, str]] = {
    "--run-capsule": (
        "a certification capsule",
        "Point --run-capsule at the certification-capsule.json the run "
        "wrote (it is in the run's output directory)."),
    "--metrics-csv": (
        "a matched-comparison metrics table",
        "Point --metrics-csv at the metrics CSV the matched comparison "
        "wrote, the one whose rows are (lead hour, field) pairs."),
    "--band": (
        "an acceptance band",
        "Point --band at the acceptance band for this configuration -- "
        "it is addressed by the configuration's SHA-256, so "
        "gpuwm.certify.band.band_path_for_config names the file."),
    "--wrf-reference-manifest": (
        "a WRF reference manifest",
        "Point --wrf-reference-manifest at the manifest naming the WRF "
        "executable, build recipe, namelists and reference wrfout bytes "
        "the comparison was made against."),
}


def read_certify_input(path: str | Path, flag: str) -> bytes:
    """Bytes for one of certify's four file arguments, or a named refusal.

    THE SHARED READER, and the point of it is the FLAG.  ``gpuwm
    certify`` takes four file arguments and used to read all four with a
    bare ``read_bytes`` + ``json.loads``, so a zero-byte or unparseable
    file anywhere among them produced exactly ``gpuwm certify: Expecting
    value: line 1 column 1 (char 0)`` -- no file, no argument, no byte
    count, no way through.  Which of the four was unreadable is the
    load-bearing part, and it was the one thing the message did not say.

    This is deliberately the same shape as
    :func:`gpuwm.certify.dualrun._read_capsule`, which fixed the
    identical failure on the sibling command in 1.8.8 and whose
    docstring calls the old text "a screen nobody can act on".  The fix
    landed on one command and not the other; now both read their inputs
    through a reader that names the argument, the path, the byte count
    and the remedy.
    """

    what, remedy = CERTIFY_INPUTS.get(
        flag, ("a certification input", "Point it at the file it names."))
    resolved = Path(path)
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise CertifyError(
            f"{flag} ({resolved}) cannot be read: {error}.  {remedy}"
        ) from error
    if not payload.strip():
        raise CertifyError(
            f"{flag} ({resolved}) is empty ({len(payload)} bytes).  That "
            f"is not {what}.  {remedy}")
    return payload


def read_certify_json(path: str | Path, flag: str) -> tuple[bytes, Any]:
    """``(bytes, document)`` for a JSON argument, or a named refusal.

    The bytes come back beside the document because the verdict binds
    every input by the sha256 of the bytes it actually read, and hashing
    a re-read of the file would be hashing something else.
    """

    what, remedy = CERTIFY_INPUTS.get(
        flag, ("a certification input", "Point it at the file it names."))
    payload = read_certify_input(path, flag)
    try:
        return payload, json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise CertifyError(
            f"{flag} ({Path(path)}) is not UTF-8 text, so it cannot be "
            f"{what}: {error}.  {remedy}") from error
    except json.JSONDecodeError as error:
        raise CertifyError(
            f"{flag} ({Path(path)}) is not valid JSON ({error.msg} at "
            f"line {error.lineno} column {error.colno}), so it cannot be "
            f"{what}.  {remedy}") from error


def canonical_digest(value: object) -> str:
    """SHA-256 over a canonical JSON encoding.

    Sorted keys, no whitespace, ASCII-escaped: two processes on two machines
    hash the same inventory to the same digest.  ``allow_nan`` is off, so a
    non-number cannot enter a binding as the bare token ``NaN``, which is not
    JSON and which no independent verifier would parse the same way.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


# --------------------------------------------------------------------------
# Reading the run's own comparison rows
# --------------------------------------------------------------------------

def read_metrics_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Field names and rows of a comparator metrics CSV, verbatim."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CertifyError(f"{path} carries no header row")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _numeric(text: str | None) -> float | None:
    if text is None or text.strip() == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return None if math.isnan(value) else value


def _capsule_config_sha256(capsule: Mapping[str, Any]) -> str | None:
    stack = capsule.get("numerical_stack") or {}
    entry = stack.get("config_bytes") or {}
    value = entry.get("value")
    if isinstance(value, Mapping):
        digest = value.get("sha256")
        return digest if isinstance(digest, str) else None
    return value if isinstance(value, str) else None


def _inventory_mode_inputs(capsule: Mapping[str, Any]) -> tuple[str, ...]:
    section = capsule.get("input_bytes") or {}
    entries = section.get("entries") if isinstance(section, Mapping) else None
    if not isinstance(entries, Mapping):
        return ()
    return tuple(sorted(
        key for key, record in entries.items()
        if isinstance(record, Mapping)
        and record.get("algorithm") == GEOGRAPHY_INVENTORY_ALGORITHM))


# --------------------------------------------------------------------------
# The comparison rows the verdict binds
# --------------------------------------------------------------------------

def banded_columns(band: Mapping[str, Any],
                   fieldnames: Sequence[str]) -> list[str]:
    """The metrics columns this band actually gates, in CSV order.

    One definition, used by :func:`evaluate_rows` to build the
    comparisons and by :func:`build_verdict` to say how many there were.
    Two copies of this list is how a refusal ends up reporting a
    different count than the one it refused on.
    """
    coverage = band["metric_coverage"]
    return [name for name in fieldnames
            if coverage.get(name, {}).get("classification")
            == CLASSIFICATION_BANDED]


def evaluate_rows(band: Mapping[str, Any], fieldnames: Sequence[str],
                  rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    """One bound comparison per (row, banded column) the band actually gates.

    A metrics row whose domain or lead the band does not carry is not silently
    skipped: it is bound with ``lower``/``upper`` null and ``inside`` false,
    because a band that does not cover a row cannot certify it.
    """
    banded = banded_columns(band, fieldnames)
    comparisons: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        domain = (row.get("domain") or "").strip()
        lead = (row.get("forecast_hour") or "").strip()
        cells = band["intervals"].get(domain, {}).get(lead)
        for column in banded:
            value = _numeric(row.get(column))
            interval = (cells or {}).get(column)
            if interval is None:
                comparisons.append({
                    "row": index, "domain": domain, "lead": lead,
                    "metric": column, "value": value,
                    "lower": None, "upper": None, "nan_expected": False,
                    "covered": False, "placement": "uncovered",
                    "inside": False,
                })
                continue
            placement = place_value(value, interval)
            comparisons.append({
                "row": index, "domain": domain, "lead": lead,
                "metric": column, "value": value,
                "lower": interval["lower"], "upper": interval["upper"],
                "nan_expected": bool(interval["nan_expected"]),
                "covered": True, "placement": placement,
                "inside": placement in ("inside", "nan"),
            })
    return comparisons


def _recompute_comparison(comparison: Mapping[str, Any]) -> bool:
    if not comparison.get("covered"):
        return False
    interval = {"lower": comparison.get("lower"),
                "upper": comparison.get("upper"),
                "nan_expected": bool(comparison.get("nan_expected"))}
    if not interval["nan_expected"] and (interval["lower"] is None
                                         or interval["upper"] is None):
        return False
    return place_value(comparison.get("value"), interval) in ("inside", "nan")


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------

def _condition(name: str, satisfied: bool, detail: str) -> dict[str, Any]:
    return {"condition": name, "satisfied": bool(satisfied), "detail": detail}


def build_verdict(*, capsule: Mapping[str, Any], band: Mapping[str, Any],
                  wrf_reference: Mapping[str, Any],
                  metrics_fieldnames: Sequence[str],
                  metrics_rows: Iterable[Mapping[str, str]],
                  capsule_sha256: str, band_sha256: str,
                  metrics_sha256: str,
                  wrf_reference_sha256: str) -> dict[str, Any]:
    """Evaluate every declared condition and bind the result to its inputs."""
    rows = list(metrics_rows)
    conditions: list[dict[str, Any]] = []

    try:
        validate_certification_capsule(capsule)
        conditions.append(_condition(
            "capsule_validates", True,
            "the run capsule validates against its schema"))
    except Exception as error:
        conditions.append(_condition(
            "capsule_validates", False, f"{type(error).__name__}: {error}"))

    capsule_config = _capsule_config_sha256(capsule)
    band_config = band.get("config_sha256")
    conditions.append(_condition(
        "band_config_identity_matches_capsule",
        capsule_config is not None and capsule_config == band_config,
        f"band config_sha256 {band_config!r} against capsule config identity "
        f"{capsule_config!r}"))

    absent = absent_reference_hashes(wrf_reference)
    conditions.append(_condition(
        "wrf_reference_hashes_present", not absent,
        "every WRF reference manifest hash is present" if not absent
        else f"WRF reference manifest hashes absent: {list(absent)}"))

    inventory = _inventory_mode_inputs(capsule)
    conditions.append(_condition(
        "geography_input_hashed_by_content", not inventory,
        "no declared input carries an inventory digest" if not inventory
        else (f"inputs hashed by listing rather than by content "
              f"({GEOGRAPHY_INVENTORY_ALGORITHM}): {list(inventory)}")))

    unresolved = unresolved_pins(capsule.get("numerical_stack") or {})
    conditions.append(_condition(
        "every_pin_resolved", not unresolved,
        "every published pin is resolved" if not unresolved
        else f"pins that are not resolved: {list(unresolved)}"))

    coverage = band["metric_coverage"]
    unclassified = [name for name in metrics_fieldnames
                    if name not in ROW_KEY_COLUMNS and name not in coverage]
    conditions.append(_condition(
        "every_metrics_column_classified", not unclassified,
        "every metrics column is classified by the band" if not unclassified
        else (f"metrics columns absent from the band's metric_coverage: "
              f"{unclassified}")))

    comparisons = evaluate_rows(band, metrics_fieldnames, rows)

    # Evaluated BEFORE the per-row condition, because the per-row
    # condition is a universal quantifier and the empty case satisfies it
    # vacuously.  A reader who sees both refusals in order sees the real
    # sequence: there was nothing to check, therefore nothing failed.
    gated = banded_columns(band, metrics_fieldnames)
    enough = (len(rows) >= MINIMUM_METRICS_ROWS
              and len(comparisons) >= MINIMUM_BANDED_COMPARISONS)
    conditions.append(_condition(
        "the_comparison_is_not_empty", enough,
        f"{len(rows)} metrics rows x {len(gated)} banded columns "
        f"= {len(comparisons)} banded comparisons (required at least "
        f"{MINIMUM_METRICS_ROWS} row and {MINIMUM_BANDED_COMPARISONS} "
        "comparison)"
        if enough
        else (
            f"nothing was compared: the metrics CSV carries {len(rows)} data "
            f"row(s) and the band gates {len(gated)} of its "
            f"{len(metrics_fieldnames)} column(s), which yields "
            f"{len(comparisons)} banded comparison(s); at least "
            f"{MINIMUM_METRICS_ROWS} row and "
            f"{MINIMUM_BANDED_COMPARISONS} banded comparison are required.  "
            "Every other row condition is a statement about all rows and is "
            "TRUE of no rows, so an empty comparison would otherwise certify "
            "as passing.  Re-run the matched comparison "
            "(tools/matched_wrfout_stream_compare.py) and certify its "
            "output, or band a column the CSV actually carries")))

    outside = [f"{item['domain']} F{item['lead']} {item['metric']}"
               for item in comparisons if not item["inside"]]
    conditions.append(_condition(
        "every_banded_row_inside_its_interval", not outside,
        f"{len(comparisons)} banded comparisons, all inside their intervals"
        if not outside
        else f"banded rows outside their own interval: {outside}"))

    verdict: dict[str, Any] = {
        "schema": VERDICT_SCHEMA_ID,
        "verdict_schema_version": VERDICT_SCHEMA_VERSION,
        "capsule_sha256": capsule_sha256,
        "capsule_emission_site": capsule.get("emission_site"),
        "capsule_config_sha256": capsule_config,
        "band": {
            "sha256": band_sha256,
            "config_sha256": band_config,
            "provenance": band.get("provenance"),
            "band_schema_version": band.get("band_schema_version"),
            "derivation": dict(band.get("derivation") or {}),
        },
        "wrf_reference": {
            "sha256": wrf_reference_sha256,
            **reference_binding(wrf_reference),
        },
        "metrics": {
            "sha256": metrics_sha256,
            "columns": list(metrics_fieldnames),
            "row_count": len(rows),
        },
        "conditions": conditions,
        "comparisons": comparisons,
    }
    verdict["capsule_binding_sha256"] = capsule_binding_sha256(verdict)
    verdict["passed"] = rederive_verdict(verdict)
    return verdict


#: Verdict keys the binding covers.  ``passed`` is deliberately absent, and so
#: is the binding digest itself; everything the decision rests on is here.
BOUND_KEYS: tuple[str, ...] = (
    "schema",
    "verdict_schema_version",
    "capsule_sha256",
    "capsule_emission_site",
    "capsule_config_sha256",
    "band",
    "wrf_reference",
    "metrics",
    "conditions",
    "comparisons",
)


def bound_inventory(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """The closed inventory the binding digest covers."""
    return {key: verdict.get(key) for key in BOUND_KEYS}


def capsule_binding_sha256(verdict: Mapping[str, Any]) -> str:
    """Hash the closed inventory, excluding the claimed verdict."""
    return canonical_digest(bound_inventory(verdict))


def rederive_verdict_reason(verdict: Mapping[str, Any]) -> tuple[bool, str]:
    """``(passed, why)`` -- the rederivation, and the sentence for a no.

    Same decision as :func:`rederive_verdict`, which is this function
    without the sentence.  A refusal that is only a bare ``False`` makes
    an independent verifier guess between "this document was tampered
    with" and "this document is older than your gpuwm", which are
    opposite conclusions; every refusal in this package says which.
    """
    conditions = verdict.get("conditions")
    comparisons = verdict.get("comparisons")
    if not isinstance(conditions, list) or not isinstance(comparisons, list):
        return False, ("not a verdict document: 'conditions' and "
                       "'comparisons' must both be lists")
    if len(comparisons) < MINIMUM_BANDED_COMPARISONS:
        return False, (
            f"the verdict binds {len(comparisons)} comparison(s); at least "
            f"{MINIMUM_BANDED_COMPARISONS} is required, because every clause "
            "below quantifies over them and is vacuously true of none")

    declared = verdict.get("verdict_schema_version")
    expected = (CONDITIONS_BY_SCHEMA_VERSION.get(declared)
                if isinstance(declared, str) else None)
    if expected is None:
        return False, (
            f"verdict_schema_version {declared!r} is not one this gpuwm can "
            f"read (known: {', '.join(sorted(CONDITIONS_BY_SCHEMA_VERSION))}"
            f"; this gpuwm writes {VERDICT_SCHEMA_VERSION}) -- upgrade the "
            "reader, or re-certify the run to regenerate the verdict")
    reported = {item.get("condition") for item in conditions}
    if reported != set(expected):
        missing = sorted(name for name in expected if name not in reported)
        extra = sorted(str(name) for name in reported if name not in expected)
        return False, (
            f"the conditions do not match what verdict_schema_version "
            f"{declared} declares (missing: {missing}; unexpected: {extra}) "
            "-- a verdict schema version and its condition set move "
            "together, so this document is not the version it claims")

    unsatisfied = [str(item.get("condition")) for item in conditions
                   if item.get("satisfied") is not True]
    if unsatisfied:
        return False, f"conditions not satisfied: {unsatisfied}"
    for index, comparison in enumerate(comparisons):
        if _recompute_comparison(comparison) is not bool(
                comparison.get("inside")):
            return False, (
                f"comparison {index} records inside="
                f"{comparison.get('inside')!r}, which disagrees with "
                "recomputing the placement from its own value and bounds")
        if not comparison.get("inside"):
            return False, (f"comparison {index} is outside its own interval")
    return True, (f"every condition of verdict_schema_version {declared} is "
                  f"satisfied and all {len(comparisons)} bound comparisons "
                  "recompute inside their own intervals")


def rederive_verdict(verdict: Mapping[str, Any]) -> bool:
    """Recompute the pass/fail decision from the bound rows alone.

    Every condition must be satisfied, and every bound comparison must still
    be inside its own recorded interval when the placement is recomputed --
    so editing a recorded ``inside`` to disagree with its own value and bounds
    fails here rather than passing on the label.

    The comparison population is checked here as well as in its own
    condition, and deliberately not only there.  Every other clause below
    quantifies over ``comparisons``; on an empty list they are all
    vacuously true, so a verdict document carrying no comparison at all
    would rederive as a pass on the strength of having nothing in it.  That
    check is version-independent, which is what lets the condition set be
    read from the document's own declared version without reopening the
    hole the newest condition closed.

    The condition set is compared against what the document's own
    ``verdict_schema_version`` declared, not against today's
    :data:`CONDITIONS`.  See :data:`CONDITIONS_BY_SCHEMA_VERSION`.
    """
    return rederive_verdict_reason(verdict)[0]


def verify_verdict(verdict: Mapping[str, Any]) -> tuple[bool, bool]:
    """``(binding_intact, rederived_pass)`` for an independent reader.

    A reader who gets ``(True, False)`` -- the binding is whole and the
    document still does not rederive -- wants to know why, and
    :func:`rederive_verdict_reason` is the same decision carrying its
    sentence.  This signature stays two booleans: it is what callers
    already unpack.
    """
    return (verdict.get("capsule_binding_sha256")
            == capsule_binding_sha256(verdict),
            rederive_verdict(verdict))


def certify(*, capsule_path: str | Path, metrics_csv: str | Path,
            band_path: str | Path, wrf_reference_manifest: str | Path
            ) -> dict[str, Any]:
    """Load every input, evaluate, and return the verdict document.

    Every read goes through :func:`read_certify_input` /
    :func:`read_certify_json` so an unreadable, empty or unparseable
    argument is refused by NAME.  ALL FOUR are attempted before any
    refusal is raised, and that is deliberate: the operator learns about
    the empty band on the same run that told them about the empty
    capsule, rather than one per attempt.  A single bad argument still
    raises exactly its own sentence; two or more are reported together,
    counted and one per line.
    """
    readers = (
        ("--run-capsule", capsule_path, read_certify_json),
        ("--band", band_path, read_certify_json),
        ("--metrics-csv", metrics_csv, read_certify_input),
        ("--wrf-reference-manifest", wrf_reference_manifest,
         read_certify_json),
    )
    read: dict[str, Any] = {}
    refusals: list[str] = []
    for flag, path, reader in readers:
        try:
            read[flag] = reader(path, flag)
        except CertifyError as error:
            refusals.append(str(error))
    if len(refusals) == 1:
        raise CertifyError(refusals[0])
    if refusals:
        raise CertifyError(
            f"{len(refusals)} of certify's {len(readers)} file arguments "
            "cannot be used:\n  " + "\n  ".join(refusals))

    capsule_bytes, capsule = read["--run-capsule"]
    band_bytes, band_document = read["--band"]
    metrics_bytes = read["--metrics-csv"]
    reference_bytes, reference_document = read["--wrf-reference-manifest"]

    try:
        band = validate_band(band_document)
    except BandError as error:
        raise CertifyError(f"{band_path}: {error}") from error
    reference = validate_wrf_reference_manifest(reference_document)
    fieldnames, rows = read_metrics_rows(metrics_csv)

    return build_verdict(
        capsule=capsule, band=band, wrf_reference=reference,
        metrics_fieldnames=fieldnames, metrics_rows=rows,
        capsule_sha256=hashlib.sha256(capsule_bytes).hexdigest(),
        band_sha256=hashlib.sha256(band_bytes).hexdigest(),
        metrics_sha256=hashlib.sha256(metrics_bytes).hexdigest(),
        wrf_reference_sha256=hashlib.sha256(reference_bytes).hexdigest())


def failing_conditions(verdict: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The conditions that refused, in declared order."""
    return [item for item in verdict.get("conditions", [])
            if not item.get("satisfied")]


__all__ = [
    "BOUND_KEYS",
    "CERTIFY_INPUTS",
    "CONDITIONS",
    "CONDITIONS_BY_SCHEMA_VERSION",
    "MINIMUM_BANDED_COMPARISONS",
    "MINIMUM_METRICS_ROWS",
    "VERDICT_SCHEMA_ID",
    "VERDICT_SCHEMA_VERSION",
    "CertifyError",
    "banded_columns",
    "bound_inventory",
    "build_verdict",
    "capsule_binding_sha256",
    "certify",
    "evaluate_rows",
    "failing_conditions",
    "read_certify_input",
    "read_certify_json",
    "read_metrics_rows",
    "rederive_verdict",
    "rederive_verdict_reason",
    "canonical_digest",
    "verify_verdict",
]
