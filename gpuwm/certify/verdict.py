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
VERDICT_SCHEMA_VERSION = "1.0.0"

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
    "every_banded_row_inside_its_interval",
)


class CertifyError(ValueError):
    """Certification could not be attempted at all."""


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

def evaluate_rows(band: Mapping[str, Any], fieldnames: Sequence[str],
                  rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    """One bound comparison per (row, banded column) the band actually gates.

    A metrics row whose domain or lead the band does not carry is not silently
    skipped: it is bound with ``lower``/``upper`` null and ``inside`` false,
    because a band that does not cover a row cannot certify it.
    """
    coverage = band["metric_coverage"]
    banded = [name for name in fieldnames
              if coverage.get(name, {}).get("classification")
              == CLASSIFICATION_BANDED]
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


def rederive_verdict(verdict: Mapping[str, Any]) -> bool:
    """Recompute the pass/fail decision from the bound rows alone.

    Every condition must be satisfied, and every bound comparison must still
    be inside its own recorded interval when the placement is recomputed --
    so editing a recorded ``inside`` to disagree with its own value and bounds
    fails here rather than passing on the label.
    """
    conditions = verdict.get("conditions")
    comparisons = verdict.get("comparisons")
    if not isinstance(conditions, list) or not isinstance(comparisons, list):
        return False
    if {item.get("condition") for item in conditions} != set(CONDITIONS):
        return False
    if not all(item.get("satisfied") is True for item in conditions):
        return False
    for comparison in comparisons:
        if _recompute_comparison(comparison) is not bool(
                comparison.get("inside")):
            return False
        if not comparison.get("inside"):
            return False
    return True


def verify_verdict(verdict: Mapping[str, Any]) -> tuple[bool, bool]:
    """``(binding_intact, rederived_pass)`` for an independent reader."""
    return (verdict.get("capsule_binding_sha256")
            == capsule_binding_sha256(verdict),
            rederive_verdict(verdict))


def certify(*, capsule_path: str | Path, metrics_csv: str | Path,
            band_path: str | Path, wrf_reference_manifest: str | Path
            ) -> dict[str, Any]:
    """Load every input, evaluate, and return the verdict document."""
    capsule_bytes = Path(capsule_path).read_bytes()
    band_bytes = Path(band_path).read_bytes()
    metrics_bytes = Path(metrics_csv).read_bytes()
    reference_bytes = Path(wrf_reference_manifest).read_bytes()

    capsule = json.loads(capsule_bytes.decode("utf-8"))
    try:
        band = validate_band(json.loads(band_bytes.decode("utf-8")))
    except BandError as error:
        raise CertifyError(f"{band_path}: {error}") from error
    reference = validate_wrf_reference_manifest(
        json.loads(reference_bytes.decode("utf-8")))
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
    "CONDITIONS",
    "VERDICT_SCHEMA_ID",
    "VERDICT_SCHEMA_VERSION",
    "CertifyError",
    "bound_inventory",
    "build_verdict",
    "capsule_binding_sha256",
    "certify",
    "evaluate_rows",
    "failing_conditions",
    "read_metrics_rows",
    "rederive_verdict",
    "canonical_digest",
    "verify_verdict",
]
