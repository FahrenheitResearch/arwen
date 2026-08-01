"""The acceptance band, as data, with the rule that derives it.

A band answers one question about one configuration: for a matched comparison
row at some (domain, lead), which metric columns are gated, and inside which
interval must a gated column fall.  It is addressed by the SHA-256 of the
configuration it certifies, so nothing here names a case, and a band for a
second configuration is a second file rather than a second code path.

The interim band's provenance is ``documented-margin``: its intervals are the
deterministic output of ``published-anchor-margin``, the committed rule that
takes the published comparison table as the anchor and opens a documented
margin around each cell -- an absolute floor per metric family, or a relative
fraction of the anchor, whichever is larger, clamped to the family's physical
range.  The rule is the artifact; the numbers are its output, and re-running
the derivation on the same committed inputs reproduces the committed file byte
for byte.  When the WRF ensemble lands, provenance becomes
``wrf-ensemble-envelope`` and the intervals come from the ensemble spread --
the same file, the same command, a stronger interval.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: Home of the shipped certification data.  ``gpuwm/data/**/*`` already ships
#: as package data, so a reader who installs the wheel gets the band, the rule
#: that derived it, and the schema both validate against.
CERTIFICATION_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "certification")

#: The committed derivation rule.
MARGIN_RULE_PATH = CERTIFICATION_DATA_DIR / "margin_rule.json"

#: The band schema, validatable by any JSON reader without importing gpuwm.
BAND_SCHEMA_PATH = CERTIFICATION_DATA_DIR / "acceptance_band.schema.json"

#: Bands, one file per configuration, named by that configuration's SHA-256.
BAND_DIR = CERTIFICATION_DATA_DIR / "bands"

BAND_SCHEMA_ID = "gpuwm.acceptance-band/v1"
MARGIN_RULE_SCHEMA_ID = "gpuwm.acceptance-band-margin-rule/v1"
BAND_SCHEMA_VERSION = "1.0.0"

PROVENANCE_DOCUMENTED_MARGIN = "documented-margin"
PROVENANCE_ENSEMBLE_ENVELOPE = "wrf-ensemble-envelope"
PROVENANCES = (PROVENANCE_DOCUMENTED_MARGIN, PROVENANCE_ENSEMBLE_ENVELOPE)

CLASSIFICATION_BANDED = "banded"
CLASSIFICATION_RECORDED_ONLY = "recorded-only"
CLASSIFICATIONS = (CLASSIFICATION_BANDED, CLASSIFICATION_RECORDED_ONLY)

#: Comparator CSV keys that identify a row rather than measure one.
ROW_KEY_COLUMNS: tuple[str, ...] = ("domain", "valid_time", "forecast_hour")


class BandError(ValueError):
    """A band, or a rule, does not satisfy its contract."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """SHA-256 of exact bytes -- the only hash this package computes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's exact bytes."""
    return sha256_bytes(Path(path).read_bytes())


def load_margin_rule(path: str | Path | None = None) -> dict[str, Any]:
    """Read the committed derivation rule and check the shape it promises."""
    rule_path = Path(path) if path is not None else MARGIN_RULE_PATH
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if rule.get("schema") != MARGIN_RULE_SCHEMA_ID:
        raise BandError(
            f"{rule_path} is not a {MARGIN_RULE_SCHEMA_ID} document")
    families = rule.get("families")
    columns = rule.get("column_classification")
    if not isinstance(families, Mapping) or not isinstance(columns, Mapping):
        raise BandError(f"{rule_path} carries no families/column table")
    for column, entry in columns.items():
        classification = entry.get("classification")
        if classification not in CLASSIFICATIONS:
            raise BandError(
                f"column {column!r} carries classification "
                f"{classification!r}, which is not one of {CLASSIFICATIONS}")
        if classification == CLASSIFICATION_BANDED:
            if entry.get("family") not in families:
                raise BandError(
                    f"banded column {column!r} names unknown metric family "
                    f"{entry.get('family')!r}")
            if not entry.get("anchor_column"):
                raise BandError(
                    f"banded column {column!r} names no anchor column")
    return rule


# --------------------------------------------------------------------------
# The anchor: the published comparison table, parsed out of the document that
# publishes it.  The document is the committed input; nothing is transcribed.
# --------------------------------------------------------------------------

_TABLE_ROW = re.compile(r"^\|(?P<body>.*)\|\s*$")


def extract_anchor_table(document_text: str, section_heading: str
                         ) -> tuple[str, list[str], list[list[str]]]:
    """Return the verbatim table text, its header cells, and its data rows.

    The verbatim text is what gets hashed into the band, so an edit to the
    published table -- even one that leaves every number alone -- makes the
    committed band stale and the re-derivation test say so.
    """
    lines = document_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.lstrip("#").strip() == section_heading and line.startswith("#"):
            start = index
            break
    if start is None:
        raise BandError(f"no section titled {section_heading!r} in the anchor "
                        "document")
    table: list[str] = []
    for line in lines[start + 1:]:
        matched = _TABLE_ROW.match(line)
        if matched is None:
            if table:
                break
            if line.startswith("#"):
                raise BandError(
                    f"section {section_heading!r} ends before its table")
            continue
        table.append(line.rstrip())
    if len(table) < 3:
        raise BandError(f"section {section_heading!r} carries no data rows")
    cells = [[part.strip() for part in _TABLE_ROW.match(row)["body"].split("|")]
             for row in table]
    header, separator, rows = cells[0], cells[1], cells[2:]
    if not all(set(part) <= set("-: ") and part for part in separator):
        raise BandError(f"section {section_heading!r} has no header rule")
    return "\n".join(table) + "\n", header, rows


def _anchor_value(text: str) -> float:
    if text.strip().lower() == "nan":
        return math.nan
    return float(text)


def parse_anchor_rows(header: Sequence[str], rows: Iterable[Sequence[str]],
                      *, lead_prefix: str) -> list[dict[str, Any]]:
    """One record per published row: domain, lead key, and anchor cells."""
    if len(header) < 3:
        raise BandError("the anchor table needs a domain, a lead, and metrics")
    metric_columns = list(header[2:])
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != len(header):
            raise BandError(
                f"anchor row {row!r} has {len(row)} cells, header has "
                f"{len(header)}")
        lead_text = row[1]
        if not lead_text.startswith(lead_prefix):
            raise BandError(
                f"anchor lead {lead_text!r} does not start with "
                f"{lead_prefix!r}")
        parsed.append({
            "domain": row[0],
            "lead": lead_key(float(lead_text[len(lead_prefix):])),
            "anchors": {name: _anchor_value(value)
                        for name, value in zip(metric_columns, row[2:])},
        })
    return parsed


def lead_key(hours: float) -> str:
    """The canonical lead key: forecast hours to one decimal, as text.

    The comparator writes ``forecast_hour`` with the same format
    (``f"{hours:.1f}"``), so a band lead and a metrics row lead are the same
    string without either side reformatting the other's number.
    """
    return f"{float(hours):.1f}"


# --------------------------------------------------------------------------
# The derivation
# --------------------------------------------------------------------------

def _interval(anchor: float, family: Mapping[str, Any], decimals: int
              ) -> dict[str, Any]:
    if math.isnan(anchor):
        return {"anchor": None, "lower": None, "upper": None,
                "nan_expected": True}
    margin = max(float(family["absolute_floor"]),
                 float(family["relative_fraction"]) * abs(anchor))
    lower = anchor - margin
    upper = anchor + margin
    clamp_low = family.get("clamp_low")
    clamp_high = family.get("clamp_high")
    if clamp_low is not None:
        lower = max(lower, float(clamp_low))
    if clamp_high is not None:
        upper = min(upper, float(clamp_high))
    return {"anchor": round(anchor, decimals),
            "lower": round(lower, decimals),
            "upper": round(upper, decimals),
            "nan_expected": False}


def derive_band(*, config_sha256: str, anchor_document_text: str,
                margin_rule: Mapping[str, Any],
                margin_rule_sha256: str,
                anchor_document_name: str) -> dict[str, Any]:
    """Apply the committed rule to the committed inputs.  Pure, and total."""
    if not re.fullmatch(r"[0-9a-f]{64}", config_sha256 or ""):
        raise BandError(
            f"config identity {config_sha256!r} is not a SHA-256 digest")
    anchor_spec = margin_rule["anchor"]
    table_text, header, rows = extract_anchor_table(
        anchor_document_text, anchor_spec["section_heading"])
    anchor_rows = parse_anchor_rows(
        header, rows, lead_prefix=anchor_spec["lead_prefix"])
    decimals = int(margin_rule["decimals"])
    families = margin_rule["families"]
    columns = margin_rule["column_classification"]
    banded = {name: entry for name, entry in columns.items()
              if entry["classification"] == CLASSIFICATION_BANDED}
    published = set(header[2:])
    unmatched = sorted(entry["anchor_column"] for entry in banded.values()
                       if entry["anchor_column"] not in published)
    if unmatched:
        raise BandError(
            "the rule bands columns whose anchor column is absent from the "
            f"published table: {unmatched}")

    intervals: dict[str, dict[str, dict[str, Any]]] = {}
    for record in anchor_rows:
        per_lead = intervals.setdefault(record["domain"], {})
        if record["lead"] in per_lead:
            raise BandError(
                f"published table repeats {record['domain']} at lead "
                f"{record['lead']}")
        cells: dict[str, Any] = {}
        for column in sorted(banded):
            entry = banded[column]
            anchor = record["anchors"][entry["anchor_column"]]
            cells[column] = _interval(
                anchor, families[entry["family"]], decimals)
        per_lead[record["lead"]] = cells
    return {
        "schema": BAND_SCHEMA_ID,
        "band_schema_version": BAND_SCHEMA_VERSION,
        "provenance": PROVENANCE_DOCUMENTED_MARGIN,
        "config_sha256": config_sha256,
        "derivation": {
            "rule_id": margin_rule["rule_id"],
            "rule_version": margin_rule["rule_version"],
            "margin_rule_sha256": margin_rule_sha256,
            "anchor_document": anchor_document_name,
            "anchor_section": anchor_spec["section_heading"],
            "anchor_table_sha256": _sha256_text(table_text),
        },
        "metric_coverage": {name: dict(entry)
                            for name, entry in columns.items()},
        "intervals": intervals,
    }


def derive_coverage_receipt(band: Mapping[str, Any], *,
                            anchor_document_text: str,
                            margin_rule: Mapping[str, Any]) -> dict[str, Any]:
    """Where every published cell falls relative to the band derived from it.

    One entry per (domain, lead, metric) triple of the published table.  The
    receipt reports; it does not judge.  A band widened until it swallowed the
    whole table would produce an all-inside receipt, which is exactly why no
    test may assert a particular distribution here.
    """
    anchor_spec = margin_rule["anchor"]
    _, header, rows = extract_anchor_table(
        anchor_document_text, anchor_spec["section_heading"])
    anchor_rows = parse_anchor_rows(
        header, rows, lead_prefix=anchor_spec["lead_prefix"])
    columns = margin_rule["column_classification"]
    by_anchor_column = {entry["anchor_column"]: name
                        for name, entry in columns.items()
                        if entry["classification"] == CLASSIFICATION_BANDED}
    cells: list[dict[str, Any]] = []
    tally = {"inside": 0, "outside": 0, "nan": 0}
    for record in anchor_rows:
        for anchor_column in header[2:]:
            column = by_anchor_column.get(anchor_column)
            published_value = record["anchors"][anchor_column]
            if column is None:
                placement = "unbanded"
            else:
                interval = band["intervals"][record["domain"]][
                    record["lead"]][column]
                placement = place_value(published_value, interval)
                tally[placement] += 1
            cells.append({
                "domain": record["domain"],
                "lead": record["lead"],
                "anchor_column": anchor_column,
                "metric": column,
                "published_value": (
                    None if math.isnan(published_value) else published_value),
                "placement": placement,
            })
    banded_cells = [cell for cell in cells if cell["metric"] is not None]
    return {
        "schema": "gpuwm.acceptance-band-coverage/v1",
        "config_sha256": band["config_sha256"],
        "band_schema_version": band["band_schema_version"],
        "provenance": band["provenance"],
        "derivation": dict(band["derivation"]),
        "triple_count": len(banded_cells),
        "tally": tally,
        "cells": banded_cells,
    }


def place_value(value: float | None, interval: Mapping[str, Any]) -> str:
    """``inside``, ``outside`` or ``nan`` for one value against one interval.

    Bounds are closed: a value sitting exactly on an endpoint is inside, which
    is what AC4's "strictly outside" refusal means from the other side.
    """
    is_nan = value is None or (isinstance(value, float) and math.isnan(value))
    if interval.get("nan_expected"):
        return "nan" if is_nan else "outside"
    if is_nan:
        return "outside"
    return ("inside"
            if float(interval["lower"]) <= float(value)
                <= float(interval["upper"])
            else "outside")


# --------------------------------------------------------------------------
# Reading a band back
# --------------------------------------------------------------------------

def band_schema() -> dict[str, Any]:
    """The band schema document."""
    return json.loads(BAND_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_band(document: Mapping[str, Any]) -> dict[str, Any]:
    """Check a band against its schema and its cross-field invariants."""
    import jsonschema

    try:
        jsonschema.validate(instance=dict(document), schema=band_schema())
    except jsonschema.ValidationError as error:
        raise BandError(
            f"band does not validate against {BAND_SCHEMA_ID}: "
            f"{error.message}") from error
    coverage = document["metric_coverage"]
    banded = {name for name, entry in coverage.items()
              if entry["classification"] == CLASSIFICATION_BANDED}
    for domain, leads in document["intervals"].items():
        for lead, cells in leads.items():
            missing = sorted(banded - set(cells))
            if missing:
                raise BandError(
                    f"band {domain} lead {lead} carries no interval for "
                    f"banded columns {missing}")
            extra = sorted(set(cells) - banded)
            if extra:
                raise BandError(
                    f"band {domain} lead {lead} carries intervals for "
                    f"columns it does not band: {extra}")
    return dict(document)


def load_band(path: str | Path) -> dict[str, Any]:
    """Read a band from disk and validate it before returning it."""
    return validate_band(
        json.loads(Path(path).read_text(encoding="utf-8")))


def band_path_for_config(config_sha256: str, *,
                         directory: str | Path | None = None) -> Path:
    """Where the band for a configuration lives.  Identity, never a name."""
    root = Path(directory) if directory is not None else BAND_DIR
    return root / f"{config_sha256}.json"


def dumps_certification_json(document: Mapping[str, Any]) -> str:
    """The one serialization every certification artifact is written with.

    Sorted keys, two-space indent, a trailing newline, and ``allow_nan=False``
    so a non-number can never be smuggled into a receipt as ``NaN``.
    """
    return json.dumps(document, indent=2, sort_keys=True,
                      allow_nan=False, ensure_ascii=False) + "\n"


def write_certification_json(path: str | Path,
                             document: Mapping[str, Any]) -> Path:
    """Write a certification artifact, with LF endings on every platform.

    ``Path.write_text`` translates newlines to the host convention, which
    would make an artifact's bytes -- and so its digest, and so any
    byte-equality check over it -- a property of the machine that wrote it
    rather than of the data it carries.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(dumps_certification_json(document).encode("utf-8"))
    return target


__all__ = [
    "BAND_DIR",
    "BAND_SCHEMA_ID",
    "BAND_SCHEMA_PATH",
    "BAND_SCHEMA_VERSION",
    "CERTIFICATION_DATA_DIR",
    "CLASSIFICATIONS",
    "CLASSIFICATION_BANDED",
    "CLASSIFICATION_RECORDED_ONLY",
    "MARGIN_RULE_PATH",
    "PROVENANCES",
    "PROVENANCE_DOCUMENTED_MARGIN",
    "PROVENANCE_ENSEMBLE_ENVELOPE",
    "ROW_KEY_COLUMNS",
    "BandError",
    "band_path_for_config",
    "band_schema",
    "derive_band",
    "derive_coverage_receipt",
    "dumps_certification_json",
    "extract_anchor_table",
    "lead_key",
    "load_band",
    "load_margin_rule",
    "parse_anchor_rows",
    "place_value",
    "sha256_bytes",
    "sha256_file",
    "validate_band",
    "write_certification_json",
]
