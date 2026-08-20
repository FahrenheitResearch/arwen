"""Pre-registration, scoring, gates, and receipts for spectral comparison.

The workflow is intentionally two-step:

1. ``register`` reads only the TOML policy and writes a hash-pinned JSON
   registration.  It does not stat, open, or inspect model output.
2. ``score`` reads that immutable registration, hashes every input file,
   computes comparisons, and writes a self-hashed receipt.

That ordering prevents a wavelength band or threshold from being selected
because it made one already-seen result look favorable.  No default empirical
gate exists; without explicit ``[[gate]]`` records the receipt verdict is
``informational``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tomllib
from typing import Mapping, Sequence

from gpuwm.verify import spectral
from gpuwm.verify import spectral_compare
from gpuwm.verify import spectral_io

SOURCE_SCHEMA = "gpuwm.spectral-comparison-source/v1"
REGISTRATION_SCHEMA = "gpuwm.spectral-comparison-registration/v1"
RECEIPT_SCHEMA = "gpuwm.spectral-comparison-receipt/v1"

_ALLOWED_KINDS = ("scalar", "vector")
_ALLOWED_GATE_TRANSFORMS = ("identity", "absolute")

#: Metrics a real band/component row carries.  Shared by the three real
#: components; ``partition`` is the synthetic row and carries only the two
#: divergent fractions and their difference.
_BAND_METRICS = (
    "amplitude_mismatch_power", "amplitude_ratio", "coherence_squared",
    "error_power", "left_fraction_of_valid_power", "left_power",
    "normalized_error_power", "phase_location_error_power",
    "phase_location_fraction_of_error", "power_ratio", "reference_power",
    "reference_fraction_of_valid_power", "spectral_correlation",
    "weighted_mean_absolute_phase_error_degrees",
)

#: WHICH metrics each gate-addressable component actually produces.
#:
#: The breakage this table names: the synthetic ``partition`` row used to
#: republish the ``total`` component's ``reference_power`` verbatim so the
#: ``minimum_reference_power`` floor could be applied to it.  That copy was
#: gate-addressable, so a campaign writing ``component = "partition"``,
#: ``metric = "reference_power"`` silently graded the total-KE row instead,
#: and ``tools/spectral_gate_calibrate.py`` -- which groups by
#: (pair, field, component, band, metric) -- built two gates out of one
#: measurement.  The floor now travels under ``gate_reference_power``, which
#: is not a metric, and a gate naming a metric its component does not carry
#: is refused at REGISTRATION, before any model output is opened.
#:
#: Adding a component or a metric is a row here, not a branch anywhere.
COMPONENT_METRICS: dict[str, tuple[str, ...]] = {
    "scalar": _BAND_METRICS,
    "total": _BAND_METRICS,
    "rotational": _BAND_METRICS,
    "divergent": _BAND_METRICS,
    "partition": (
        "divergent_energy_fraction_difference",
        "left_divergent_energy_fraction",
        "reference_divergent_energy_fraction",
    ),
}

#: The key a gate's ``minimum_reference_power`` floor is read from.  Every
#: row carries it; on a real component it equals that component's own
#: ``reference_power``, and on the ``partition`` row it is the total
#: component's, because the partition fractions are only meaningful where
#: the total energy is.
GATE_FLOOR_KEY = "gate_reference_power"

_GATE_METRICS = frozenset(
    metric for metrics in COMPONENT_METRICS.values() for metric in metrics)


def metrics_for_component(component: str) -> tuple[str, ...]:
    """The metrics ``component`` produces, or ``()`` if it is not a component."""

    return COMPONENT_METRICS.get(str(component), ())


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _code_identity() -> dict[str, object]:
    """The evaluator identity, from the capsule builder that already owns it.

    Imported at call time: ``gpuwm.certify.capsule`` pulls in the supervisor
    for ``git_commit``, and ``gpuwm spectral`` is CPU-only and must stay
    importable in a minimal environment.  A tree without git still answers --
    ``git_commit`` returns its own unavailable marker rather than raising --
    so a receipt is never blocked on it.
    """

    from gpuwm.certify import capsule  # noqa: PLC0415

    return json.loads(json.dumps(capsule.code_identity()))


def policy_parameters(parameters: Mapping[str, object]) -> dict[str, object]:
    """The registration's policy with every box-specific path removed.

    ``registration_sha256`` binds the RESOLVED absolute paths of every pair,
    which is what makes it a durable pin on this box -- and what makes it
    useless for asking whether two boxes ran the same campaign.  Measured on
    a real pair of runs: the same source TOML, the same input bytes, and two
    registration hashes, because one box spells the file
    ``C:\\Users\\<user>\\...\\candidate.npz`` and the other
    ``/home/<user>/.../candidate.npz``.

    The policy digest is the same parameters with ``left_path`` and
    ``reference_path`` replaced by their basenames.  What a campaign
    declared -- bands, fields, gates, crop, pins -- survives; where the bytes
    happened to sit does not.  This is the portable-frame-header rule
    applied to a registration: the raw digest keeps its meaning, and a
    second, declared one is what may be compared between boxes.
    """

    portable = json.loads(json.dumps(dict(parameters), allow_nan=False))
    for pair in portable.get("pairs", []):
        for key in ("left_path", "reference_path"):
            if key in pair:
                pair[key] = os.path.basename(str(pair[key]).replace("\\", "/"))
    portable["path_policy"] = "basenames only; see policy_parameters"
    return portable


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    os.replace(temporary, target)
    return target


def read_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _positive(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive and finite")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    number = int(value)
    if number != value or number < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    number = int(value)
    if number != value:
        raise ValueError(f"{label} must be an integer")
    return number


def _source_path(base: Path, value: object, *, label: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{label} is required")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    # abspath is lexical: unlike Path.resolve(), it does not inspect symlinks
    # or require any output path component to exist.
    return os.path.abspath(os.fspath(path))


def _normalise_band(record: Mapping[str, object]) -> dict[str, object]:
    band = spectral_compare.WavelengthBand(
        name=str(record.get("name", "")),
        minimum_km=(None if record.get("minimum_km") is None
                    else float(record["minimum_km"])),
        maximum_km=(None if record.get("maximum_km") is None
                    else float(record["maximum_km"])),
    )
    return {
        "name": band.name,
        "minimum_km": band.minimum_km,
        "maximum_km": band.maximum_km,
    }


def _normalise_pair(record: Mapping[str, object], *, base: Path,
                    default_dx: object | None, default_dy: object | None,
                    index: int) -> dict[str, object]:
    name = str(record.get("name", f"pair-{index:03d}"))
    if not name.strip():
        raise ValueError("comparison pair names must be non-empty")
    dx_raw = record.get("dx_m", default_dx)
    if dx_raw is None:
        raise ValueError(f"pair {name!r} has no dx_m")
    dy_raw = record.get("dy_m", default_dy if default_dy is not None else dx_raw)
    result: dict[str, object] = {
        "name": name,
        "left_path": _source_path(base, record.get("left_path"),
                                  label=f"pair {name!r} left_path"),
        "reference_path": _source_path(
            base, record.get("reference_path"),
            label=f"pair {name!r} reference_path"),
        "dx_m": _positive(dx_raw, label=f"pair {name!r} dx_m"),
        "dy_m": _positive(dy_raw, label=f"pair {name!r} dy_m"),
        "left_time_index": _nonnegative_integer(
            record.get("left_time_index", record.get("time_index", 0)),
            label=f"pair {name!r} left_time_index"),
        "reference_time_index": _nonnegative_integer(
            record.get("reference_time_index", record.get("time_index", 0)),
            label=f"pair {name!r} reference_time_index"),
        "domain": None if record.get("domain") is None else str(record["domain"]),
        "valid_time": (None if record.get("valid_time") is None
                       else str(record["valid_time"])),
    }
    metadata = record.get("metadata", {})
    if metadata:
        if not isinstance(metadata, dict):
            raise ValueError(f"pair {name!r} metadata must be a table")
        result["metadata"] = json.loads(json.dumps(metadata, allow_nan=False))
    return result


def _field_side(record: Mapping[str, object], side: str, *,
                variable_key: str, default_destagger: str,
                reduction: str, level_index: int | None) -> dict[str, object]:
    variable = record.get(f"{side}_{variable_key}", record.get(variable_key))
    destagger = str(record.get(
        f"{side}_{variable_key.removesuffix('_variable')}_destagger",
        record.get(f"{side}_destagger", default_destagger))).lower()
    side_reduction = str(record.get(
        f"{side}_reduction", reduction)).lower()
    side_level = record.get(f"{side}_level_index", level_index)
    side_time = record.get(f"{side}_time_index")
    if destagger not in spectral_io.DESTAGGER:
        raise ValueError(
            f"{side} destagger must be one of {spectral_io.DESTAGGER}")
    if side_reduction not in spectral_io.REDUCTIONS:
        raise ValueError(
            f"{side} reduction must be one of {spectral_io.REDUCTIONS}")
    if side_level is not None:
        side_level = _integer(side_level, label=f"{side} level_index")
    if side_reduction == "level" and side_level is None:
        raise ValueError(f"{side} level reduction needs level_index")
    if side_time is not None:
        side_time = _nonnegative_integer(
            side_time, label=f"{side} time_index")
    return {
        "variable": None if variable is None else str(variable),
        "destagger": destagger,
        "reduction": side_reduction,
        "level_index": side_level,
        "time_index": side_time,
    }


def _normalise_field(record: Mapping[str, object], *, index: int) -> dict[str, object]:
    name = str(record.get("name", f"field-{index:03d}"))
    kind = str(record.get("kind", "scalar")).lower()
    if not name.strip() or kind not in _ALLOWED_KINDS:
        raise ValueError(
            f"field needs a non-empty name and kind in {_ALLOWED_KINDS}")
    reduction = str(record.get("reduction", "plane")).lower()
    if reduction not in spectral_io.REDUCTIONS:
        raise ValueError(f"field {name!r} has unknown reduction {reduction!r}")
    level_index = record.get("level_index")
    if level_index is not None:
        level_index = _integer(level_index, label=f"field {name!r} level_index")
    if reduction == "level" and level_index is None:
        raise ValueError(f"field {name!r} level reduction needs level_index")

    result: dict[str, object] = {
        "name": name,
        "kind": kind,
        "units": None if record.get("units") is None else str(record["units"]),
    }
    if kind == "scalar":
        left = _field_side(
            record, "left", variable_key="variable", default_destagger="none",
            reduction=reduction, level_index=level_index)
        reference = _field_side(
            record, "reference", variable_key="variable",
            default_destagger="none", reduction=reduction,
            level_index=level_index)
        if left["variable"] is None and reference["variable"] is not None:
            left["variable"] = reference["variable"]
        if reference["variable"] is None and left["variable"] is not None:
            reference["variable"] = left["variable"]
        result["left"] = left
        result["reference"] = reference
    else:
        left_u = _field_side(
            record, "left", variable_key="u_variable", default_destagger="auto",
            reduction=reduction, level_index=level_index)
        left_v = _field_side(
            record, "left", variable_key="v_variable", default_destagger="auto",
            reduction=reduction, level_index=level_index)
        reference_u = _field_side(
            record, "reference", variable_key="u_variable",
            default_destagger="auto", reduction=reduction,
            level_index=level_index)
        reference_v = _field_side(
            record, "reference", variable_key="v_variable",
            default_destagger="auto", reduction=reduction,
            level_index=level_index)
        for lhs, rhs in ((left_u, reference_u), (left_v, reference_v)):
            if lhs["variable"] is None and rhs["variable"] is not None:
                lhs["variable"] = rhs["variable"]
            if rhs["variable"] is None and lhs["variable"] is not None:
                rhs["variable"] = lhs["variable"]
        if any(item["variable"] is None
               for item in (left_u, left_v, reference_u, reference_v)):
            raise ValueError(
                f"vector field {name!r} needs U and V variable names")
        result.update({
            "left_u": left_u, "left_v": left_v,
            "reference_u": reference_u, "reference_v": reference_v,
        })
    return result


def _normalise_gate(record: Mapping[str, object], *, index: int,
                    field_names: set[str], band_names: set[str],
                    pair_names: set[str]) -> dict[str, object]:
    field = str(record.get("field", ""))
    band = str(record.get("band", ""))
    pair = str(record.get("pair", "*"))
    component = str(record.get("component", "scalar"))
    metric = str(record.get("metric", ""))
    transform = str(record.get("transform", "identity"))
    if field not in field_names:
        raise ValueError(f"gate {index} names unknown field {field!r}")
    if band not in band_names:
        raise ValueError(f"gate {index} names unknown band {band!r}")
    if pair != "*" and pair not in pair_names:
        raise ValueError(f"gate {index} names unknown pair {pair!r}")
    if metric not in _GATE_METRICS:
        raise ValueError(f"gate {index} names unknown metric {metric!r}")
    carried = metrics_for_component(component)
    if not carried:
        raise ValueError(
            f"gate {index} names unknown component {component!r}; a gate "
            f"targets one of {sorted(COMPONENT_METRICS)}")
    if metric not in carried:
        raise ValueError(
            f"gate {index} declares component {component!r} and metric "
            f"{metric!r}, but a {component!r} row does not carry that "
            f"metric -- it carries {list(carried)}. Grading it would read "
            "a different component's number under this gate's name. Either "
            f"target a component that produces {metric!r}, or keep "
            f"{component!r} and choose one of the metrics it carries.")
    if transform not in _ALLOWED_GATE_TRANSFORMS:
        raise ValueError(
            f"gate transform must be one of {_ALLOWED_GATE_TRANSFORMS}")
    minimum = record.get("minimum")
    maximum = record.get("maximum")
    if minimum is None and maximum is None:
        raise ValueError(f"gate {index} needs minimum and/or maximum")
    minimum = None if minimum is None else float(minimum)
    maximum = None if maximum is None else float(maximum)
    if minimum is not None and not math.isfinite(minimum):
        raise ValueError(f"gate {index} minimum is non-finite")
    if maximum is not None and not math.isfinite(maximum):
        raise ValueError(f"gate {index} maximum is non-finite")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"gate {index} minimum exceeds maximum")
    minimum_reference_power = record.get("minimum_reference_power")
    if minimum_reference_power is not None:
        minimum_reference_power = float(minimum_reference_power)
        if (not math.isfinite(minimum_reference_power)
                or minimum_reference_power < 0.0):
            raise ValueError(
                f"gate {index} minimum_reference_power must be non-negative")
    return {
        "id": str(record.get(
            "id", f"{pair}:{field}:{component}:{band}:{metric}")),
        "pair": pair,
        "field": field,
        "component": component,
        "band": band,
        "metric": metric,
        "transform": transform,
        "minimum": minimum,
        "maximum": maximum,
        "minimum_reference_power": minimum_reference_power,
        "note": None if record.get("note") is None else str(record["note"]),
    }


def load_source_spec(path: str | Path) -> tuple[dict[str, object], Path, str]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"spectral registration source is not a file: {source}")
    raw_bytes = source.read_bytes()
    raw = tomllib.loads(raw_bytes.decode("utf-8"))
    if raw.get("schema") != SOURCE_SCHEMA:
        raise ValueError(
            f"spectral source schema must be {SOURCE_SCHEMA!r}")
    return raw, source.resolve(), hashlib.sha256(raw_bytes).hexdigest()


def make_registration(source_toml: str | Path) -> dict[str, object]:
    """Pin policy without opening or even statting a model-output file."""

    raw, source_path, source_sha256 = load_source_spec(source_toml)
    base = source_path.parent
    bands = [_normalise_band(item) for item in raw.get("bands", [])]
    parsed_bands = spectral_compare.parse_bands(bands)
    bands = spectral_compare.bands_as_records(parsed_bands)
    pairs = [
        _normalise_pair(
            item, base=base,
            default_dx=raw.get("default_dx_m"),
            default_dy=raw.get("default_dy_m"), index=index)
        for index, item in enumerate(raw.get("pairs", []), start=1)
    ]
    fields = [
        _normalise_field(item, index=index)
        for index, item in enumerate(raw.get("fields", []), start=1)
    ]
    if not pairs or not fields:
        raise ValueError("spectral registration needs at least one pair and field")
    for label, records in (("pair", pairs), ("field", fields)):
        names = [str(item["name"]) for item in records]
        if len(names) != len(set(names)):
            raise ValueError(f"{label} names must be unique")
    field_names = {str(item["name"]) for item in fields}
    band_names = {str(item["name"]) for item in bands}
    pair_names = {str(item["name"]) for item in pairs}
    gates = [
        _normalise_gate(
            item, index=index, field_names=field_names,
            band_names=band_names, pair_names=pair_names)
        for index, item in enumerate(raw.get("gates", []), start=1)
    ]
    gate_ids = [str(item["id"]) for item in gates]
    if len(gate_ids) != len(set(gate_ids)):
        raise ValueError("gate ids must be unique")

    crop_cells = _nonnegative_integer(
        raw.get("crop_cells", 0), label="crop_cells")
    parameters: dict[str, object] = {
        "title": str(raw.get("title", source_path.stem)),
        "left_label": str(raw.get("left_label", "candidate")),
        "reference_label": str(raw.get("reference_label", "reference")),
        "crop_cells": crop_cells,
        "bands": bands,
        "pairs": pairs,
        "fields": fields,
        "gates": gates,
        "spectral_compare_pins": spectral_compare.implementation_registration()[
            "pins"],
        "spectral_compare_pins_sha256": (
            spectral_compare.IMPLEMENTATION_PINS_SHA256),
        "legacy_spectral_v1_pins_sha256": spectral.PINS_SHA256,
        "missing_nonfinite_or_unresolved_gate_policy": "incomplete-not-pass",
        "input_hash": "full-file SHA-256 at score time",
    }
    registration: dict[str, object] = {
        "schema": REGISTRATION_SCHEMA,
        "registered_utc": _utc_now(),
        "source_toml": str(source_path),
        "source_toml_sha256": source_sha256,
        "parameters": parameters,
    }
    registration["registration_sha256"] = canonical_hash(parameters)
    registration["registration_policy_sha256"] = canonical_hash(
        policy_parameters(parameters))
    return registration


def validate_registration(value: Mapping[str, object]) -> dict[str, object]:
    registration = dict(value)
    required = {
        "schema", "registered_utc", "source_toml", "source_toml_sha256",
        "parameters", "registration_sha256", "registration_policy_sha256",
    }
    missing = required - set(registration)
    if missing:
        raise ValueError(f"spectral registration is missing {sorted(missing)}")
    if registration["schema"] != REGISTRATION_SCHEMA:
        raise ValueError("spectral registration schema mismatch")
    parameters = registration["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("spectral registration parameters must be an object")
    if canonical_hash(parameters) != registration["registration_sha256"]:
        raise ValueError("spectral registration hash does not match its pins")
    expected_policy = canonical_hash(policy_parameters(parameters))
    if registration.get("registration_policy_sha256") != expected_policy:
        raise ValueError(
            "spectral registration policy hash does not match its pins")
    if (parameters.get("spectral_compare_pins_sha256")
            != spectral_compare.IMPLEMENTATION_PINS_SHA256):
        raise ValueError("spectral comparison implementation pins do not match")
    if parameters.get("legacy_spectral_v1_pins_sha256") != spectral.PINS_SHA256:
        raise ValueError("legacy spectral-v1 pins moved under this registration")
    if canonical_hash(parameters.get("spectral_compare_pins")) != (
            parameters.get("spectral_compare_pins_sha256")):
        raise ValueError("embedded spectral implementation pins are corrupt")
    return registration


def _source(pair: Mapping[str, object], field_source: Mapping[str, object],
            side: str) -> dict[str, object]:
    time_override = field_source.get("time_index")
    return {
        "path": pair[f"{side}_path"],
        "variable": field_source.get("variable"),
        "time_index": (pair[f"{side}_time_index"]
                       if time_override is None else int(time_override)),
        "destagger": field_source.get("destagger", "none"),
        "reduction": field_source.get("reduction", "plane"),
        "level_index": field_source.get("level_index"),
    }


def _load_one(pair: Mapping[str, object], field_source: Mapping[str, object],
              side: str) -> tuple[object, dict[str, object], dict[str, object]]:
    source = _source(pair, field_source, side)
    plane, metadata = spectral_io.load_plane(source)
    return plane, metadata, source


def _comparison_records(registration: Mapping[str, object]
                        ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    params = registration["parameters"]
    bands = spectral_compare.parse_bands(params["bands"])
    crop = int(params["crop_cells"])
    comparisons: list[dict[str, object]] = []
    files: dict[str, dict[str, object]] = {}
    for pair in params["pairs"]:
        for path_key in ("left_path", "reference_path"):
            path = str(pair[path_key])
            if path not in files:
                files[path] = spectral_io.file_identity(path)
        for field in params["fields"]:
            if field["kind"] == "scalar":
                left, left_meta, left_source = _load_one(
                    pair, field["left"], "left")
                reference, reference_meta, reference_source = _load_one(
                    pair, field["reference"], "reference")
                result = spectral_compare.compare_scalar(
                    left, reference, dx_m=float(pair["dx_m"]),
                    dy_m=float(pair["dy_m"]), bands=bands, crop_cells=crop)
                sources = {
                    "left": {"request": left_source, "resolved": left_meta},
                    "reference": {
                        "request": reference_source, "resolved": reference_meta},
                }
            else:
                left_u, left_u_meta, left_u_source = _load_one(
                    pair, field["left_u"], "left")
                left_v, left_v_meta, left_v_source = _load_one(
                    pair, field["left_v"], "left")
                ref_u, ref_u_meta, ref_u_source = _load_one(
                    pair, field["reference_u"], "reference")
                ref_v, ref_v_meta, ref_v_source = _load_one(
                    pair, field["reference_v"], "reference")
                result = spectral_compare.compare_vector(
                    left_u, left_v, ref_u, ref_v,
                    dx_m=float(pair["dx_m"]), dy_m=float(pair["dy_m"]),
                    bands=bands, crop_cells=crop)
                sources = {
                    "left_u": {"request": left_u_source,
                               "resolved": left_u_meta},
                    "left_v": {"request": left_v_source,
                               "resolved": left_v_meta},
                    "reference_u": {"request": ref_u_source,
                                    "resolved": ref_u_meta},
                    "reference_v": {"request": ref_v_source,
                                    "resolved": ref_v_meta},
                }
            comparisons.append({
                "pair": pair["name"],
                "domain": pair.get("domain"),
                "valid_time": pair.get("valid_time"),
                "field": field["name"],
                "units": field.get("units"),
                "kind": field["kind"],
                "sources": sources,
                "result": result,
            })
    return comparisons, [files[path] for path in sorted(files)]


def _gate_rows(comparison: Mapping[str, object]) -> list[dict[str, object]]:
    pair = comparison["pair"]
    field = comparison["field"]
    result = comparison["result"]
    rows = []
    for row in spectral_compare.metric_rows(result):
        rows.append({
            "pair": pair, "field": field, **row,
            GATE_FLOOR_KEY: row.get("reference_power"),
        })
    if result.get("kind") == "vector":
        for band in result.get("bands", []):
            left_fraction = band.get("left_divergent_energy_fraction")
            reference_fraction = band.get("reference_divergent_energy_fraction")
            rows.append({
                "pair": pair,
                "field": field,
                "band": band["name"],
                "component": "partition",
                "status": ("ok" if left_fraction is not None
                           and reference_fraction is not None else "unresolved"),
                "left_divergent_energy_fraction": left_fraction,
                "reference_divergent_energy_fraction": reference_fraction,
                "divergent_energy_fraction_difference": (
                    None if left_fraction is None or reference_fraction is None
                    else float(left_fraction) - float(reference_fraction)),
                # The floor, NOT a metric: the partition fractions are only
                # meaningful where the total energy is, so the floor is read
                # from the total component.  Publishing it as
                # ``reference_power`` made it gate-addressable, and a gate on
                # the partition row then graded the total component's power.
                GATE_FLOOR_KEY: band.get("components", {}).get(
                    "total", {}).get("reference_power"),
            })
    return rows


def receipt_metric_rows(value: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten every receipt comparison into calibration/gate-addressable rows."""

    receipt = validate_receipt(value)
    return [
        row
        for comparison in receipt["comparisons"]
        for row in _gate_rows(comparison)
    ]


def evaluate_gates(comparisons: Sequence[Mapping[str, object]],
                   gates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not gates:
        return {"verdict": "informational", "rows": [],
                "passed": 0, "failed": 0, "incomplete": 0}
    candidates = [row for comparison in comparisons for row in _gate_rows(comparison)]
    results: list[dict[str, object]] = []
    for gate in gates:
        matches = [row for row in candidates
                   if row["field"] == gate["field"]
                   and row["band"] == gate["band"]
                   and row["component"] == gate["component"]
                   and (gate["pair"] == "*" or row["pair"] == gate["pair"])]
        if not matches:
            results.append({**gate, "row_id": f"{gate['id']}@-",
                            "status": "incomplete",
                            "reason": "gate target matched no comparison row"})
            continue
        for row in matches:
            raw = row.get(gate["metric"])
            reference_power = row.get(GATE_FLOOR_KEY)
            record = {
                **gate,
                # A ``pair = "*"`` gate matches one row per pair, and every
                # one of those rows used to carry the gate's single ``id``.
                # The registration guarantees unique gate ids; the receipt
                # then broke it, so anything joining evaluated rows on id --
                # an audit, the calibrator -- silently merged pairs.
                "row_id": f"{gate['id']}@{row['pair']}",
                "pair": row["pair"],
                "raw_value": raw,
                "reference_power": reference_power,
            }
            if row.get("status") != "ok" or raw is None:
                results.append({**record, "status": "incomplete",
                                "reason": "target band/metric is unresolved"})
                continue
            floor = gate.get("minimum_reference_power")
            if floor is not None and (
                    reference_power is None or float(reference_power) < float(floor)):
                results.append({
                    **record, "status": "incomplete",
                    "reason": "reference power is below the pre-registered floor",
                })
                continue
            value = float(raw)
            if gate["transform"] == "absolute":
                value = abs(value)
            minimum = gate.get("minimum")
            maximum = gate.get("maximum")
            passed = ((minimum is None or float(minimum) <= value)
                      and (maximum is None or value <= float(maximum)))
            results.append({
                **record, "value": value,
                "status": "pass" if passed else "fail",
            })
    passed_count = sum(row["status"] == "pass" for row in results)
    failed_count = sum(row["status"] == "fail" for row in results)
    incomplete_count = sum(row["status"] == "incomplete" for row in results)
    verdict = ("fail" if failed_count else
               "incomplete" if incomplete_count else "pass")
    return {
        "verdict": verdict,
        "rows": results,
        "passed": passed_count,
        "failed": failed_count,
        "incomplete": incomplete_count,
    }


def score_registration(value: Mapping[str, object]) -> dict[str, object]:
    registration = validate_registration(value)
    comparisons, inputs = _comparison_records(registration)
    gate_result = evaluate_gates(
        comparisons, registration["parameters"].get("gates", []))
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "generated_utc": _utc_now(),
        "registration_sha256": registration["registration_sha256"],
        # The same registration on two boxes hashes differently, because a
        # registration binds RESOLVED absolute paths.  Measured on a real
        # pair of runs.  The policy hash is what two boxes can compare.
        "registration_policy_sha256": registration[
            "registration_policy_sha256"],
        "registration_source_toml_sha256": registration["source_toml_sha256"],
        "spectral_compare_pins_sha256": (
            spectral_compare.IMPLEMENTATION_PINS_SHA256),
        "legacy_spectral_v1_pins_sha256": spectral.PINS_SHA256,
        "title": registration["parameters"]["title"],
        "left_label": registration["parameters"]["left_label"],
        "reference_label": registration["parameters"]["reference_label"],
        # The evaluator, resolved by the certification capsule's own builder
        # so a run's capsule and its spectral receipt cannot name different
        # commits.  A receipt already bound the arithmetic and the input
        # bytes; the code that ran between them was the missing third.
        "code": _code_identity(),
        # What these numbers reproduce to, and what they do NOT: the receipt
        # self-hash is a THIS-BOX identity.  Measured, not assumed -- see
        # spectral_compare.CROSS_BOX_RULE.
        "reproducibility": json.loads(
            json.dumps(spectral_compare.CROSS_BOX_RULE)),
        "inputs": inputs,
        "comparisons": comparisons,
        "gates": gate_result,
        "verdict": gate_result["verdict"],
    }
    receipt["receipt_sha256"] = canonical_hash(receipt)
    return receipt


def validate_receipt(value: Mapping[str, object], *,
                     rehash_inputs: bool = False) -> dict[str, object]:
    receipt = dict(value)
    required = {
        "schema", "generated_utc", "registration_sha256",
        "registration_policy_sha256",
        "spectral_compare_pins_sha256", "legacy_spectral_v1_pins_sha256",
        "code", "reproducibility",
        "inputs", "comparisons", "gates", "verdict", "receipt_sha256",
    }
    missing = required - set(receipt)
    if missing:
        raise ValueError(f"spectral receipt is missing {sorted(missing)}")
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise ValueError("spectral receipt schema mismatch")
    expected = receipt.pop("receipt_sha256")
    actual = canonical_hash(receipt)
    receipt["receipt_sha256"] = expected
    if actual != expected:
        raise ValueError("spectral receipt hash does not match its contents")
    if receipt["spectral_compare_pins_sha256"] != (
            spectral_compare.IMPLEMENTATION_PINS_SHA256):
        raise ValueError("receipt was computed under different comparison pins")
    if receipt["legacy_spectral_v1_pins_sha256"] != spectral.PINS_SHA256:
        raise ValueError("receipt records a different legacy spectral-v1 pin")
    if rehash_inputs:
        for item in receipt["inputs"]:
            current = spectral_io.file_identity(item["path"])
            if (current["sha256"] != item["sha256"]
                    or current["size_bytes"] != item["size_bytes"]):
                raise ValueError(f"spectral input moved: {item['path']}")
    return receipt


def register_file(source_toml: str | Path, output: str | Path) -> dict[str, object]:
    registration = make_registration(source_toml)
    write_json_atomic(output, registration)
    return registration


def score_file(registration_path: str | Path, output: str | Path
               ) -> dict[str, object]:
    registration = validate_registration(read_json(registration_path))
    receipt = score_registration(registration)
    write_json_atomic(output, receipt)
    return receipt


def check_file(path: str | Path, *, rehash_inputs: bool = False
              ) -> dict[str, object]:
    return validate_receipt(read_json(path), rehash_inputs=rehash_inputs)


__all__ = [
    "RECEIPT_SCHEMA", "REGISTRATION_SCHEMA", "SOURCE_SCHEMA", "canonical_hash",
    "check_file", "evaluate_gates", "load_source_spec", "make_registration",
    "receipt_metric_rows",
    "read_json", "register_file", "score_file", "score_registration",
    "validate_receipt", "validate_registration", "write_json_atomic",
]
