"""Read case catalogs as data and compile selections through native builders.

Catalog prose never becomes code. Typed scientific overrides are restricted to
the existing configuration/physics vocabulary and the completed configuration
passes the same native validation and memory admission as a research workspace.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import date, datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import tomllib


SCHEMA = "arwen.case-catalog.v1"
TIERS = ("lower", "recommended", "upper")
MAX_CATALOG_BYTES = 128 * 1024 * 1024
SCHEMA_PATH = Path(__file__).parent / "data" / "case-catalog" / "schema.json"
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}\Z")
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}


class CatalogError(ValueError):
    """An invalid catalog or a selection the native product cannot create."""


@dataclass(frozen=True)
class Catalog:
    document: dict
    original: bytes
    source: str
    format: str
    sha256: str

    def provenance(self) -> dict:
        return {"catalog": deepcopy(self.document["catalog"]),
                "source": self.source, "format": self.format,
                "original_sha256": self.sha256, "original_bytes": len(self.original)}


def _object(value, label: str, *, allowed=None, required=()) -> dict:
    if not isinstance(value, dict):
        raise CatalogError(f"{label} must be an object/table")
    missing = set(required) - set(value)
    if missing:
        raise CatalogError(f"{label} is missing: {', '.join(sorted(missing))}")
    if allowed is not None and set(value) - set(allowed):
        raise CatalogError(f"Unknown {label} fields: {', '.join(sorted(set(value) - set(allowed)))}; "
                           "put advisory information in metadata or recommendations")
    return value


def _text(value, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise CatalogError(f"{label} must be {'a string' if empty else 'a nonempty string'}")
    if any(ord(c) < 32 and c not in "\n\t\r" for c in value):
        raise CatalogError(f"{label} contains a control character")
    return value


def _id(value, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value) or value in _RESERVED:
        raise CatalogError(f"{label} must be a safe lowercase ID of at most 128 letters, "
                           "digits, underscores or hyphens; paths and reserved filenames are not IDs")
    return value


def _number(value, label: str, *, minimum=None, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CatalogError(f"{label} must be a finite {'integer' if integer else 'number'}")
    if integer and (not isinstance(value, int)):
        raise CatalogError(f"{label} must be an integer")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise CatalogError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _json_data(value, label: str = "catalog"):
    if isinstance(value, datetime):
        return _utc(value, label)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise CatalogError(f"{label} keys must be strings")
        return {k: _json_data(v, f"{label}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_json_data(v, f"{label}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str):
        return _text(value, label, empty=True)
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise CatalogError(f"{label} must contain finite JSON/TOML data, not executable objects")


def _utc(value, label: str, *, hour: bool = False) -> str:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str) and "T" in value:
        try:
            stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as error:
            raise CatalogError(f"{label} must be an ISO UTC date/time") from error
    else:
        raise CatalogError(f"{label} must include a date and time, for example 2026-09-01T00:00:00Z")
    # The field explicitly says UTC; an offset is converted, never discarded.
    stamp = stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)
    if hour and (stamp.minute or stamp.second or stamp.microsecond):
        raise CatalogError(f"{label} must resolve to an exact UTC hour")
    return stamp.isoformat().replace("+00:00", "Z")


def _source(value: str) -> str:
    from gpuwm.source_adapters import get_source_adapter
    value = _text(value, "source").strip().lower().replace("_", "-")
    try:
        return get_source_adapter(value).source_id
    except ValueError:
        # A future source remains searchable data; selection explains its lack
        # of a native adapter rather than discarding the historical case.
        return value


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CatalogError(f"Duplicate JSON field {key!r}; scientific choices must be unambiguous")
        result[key] = value
    return result


@lru_cache(maxsize=1)
def _native_contract() -> tuple[dict, set[str], set[str]]:
    from gpuwm.config import RunConfig
    from gpuwm.experiment import _DOMAIN_RUN_OVERRIDES, _SHARED_FORBIDDEN
    from gpuwm.physics_registry import physics_registry
    registry = physics_registry()
    selectors = {key for c in registry["components"].values() for key in c.get("selector_keys", [])}
    names = ({f.name for f in fields(RunConfig)}
             & (set(registry["parameters"]) | selectors))
    # These are process/file authority or declarations, not case science.
    names = {n for n in names if not n.endswith(("_path", "_file", "_dir", "_root",
                                                "_acknowledgement", "_acknowledgements"))}
    return registry, names - set(_SHARED_FORBIDDEN), names & set(_DOMAIN_RUN_OVERRIDES)


def native_settings_document() -> dict:
    registry, shared, domains = _native_contract()
    return {"schema": "arwen.case-native-settings.v1", "shared_keys": sorted(shared),
            "domain_keys": sorted(domains), "parameter_specs": deepcopy(registry["parameters"]),
            "note": "A recognized setting still must pass native type, combination, source and memory validation."}


def validate_native_overrides(value: dict, label: str = "native_overrides") -> dict:
    _object(value, label, allowed={"shared", "domains"})
    registry, shared_keys, domain_keys = _native_contract()
    from gpuwm.physics_registry import _parameter_error, parameter_is_implemented

    def settings(mapping, allowed, where):
        _object(mapping, where)
        for key, item in mapping.items():
            if key not in allowed:
                raise CatalogError(f"{where}.{key} is not a recognized native scientific setting in this scope; "
                                   "commands, paths, environment and credentials cannot be catalog overrides")
            if not isinstance(item, (str, int, float, bool)) or isinstance(item, float) and not math.isfinite(item):
                raise CatalogError(f"{where}.{key} must be a finite scalar native setting")
            spec = registry["parameters"].get(key)
            if spec is not None:
                if not parameter_is_implemented(spec):
                    raise CatalogError(f"{where}.{key}: {spec.get('unimplemented_reason', 'not implemented')}")
                error = _parameter_error(spec, item)
                if error:
                    raise CatalogError(f"{where}.{key}: {error}")
            elif isinstance(item, bool) or not isinstance(item, int):
                raise CatalogError(f"{where}.{key} is a native integer scheme selector")

    settings(value.get("shared", {}), shared_keys, label + ".shared")
    rows = value.get("domains", [])
    if not isinstance(rows, list):
        raise CatalogError(f"{label}.domains must be a list of grid_id/settings objects")
    seen = set()
    for row in rows:
        _object(row, label + ".domains[]", allowed={"grid_id", "settings"}, required={"grid_id", "settings"})
        grid = _number(row["grid_id"], label + ".grid_id", minimum=1, integer=True)
        if grid in seen:
            raise CatalogError(f"{label} repeats grid_id {grid}")
        seen.add(grid)
        settings(row["settings"], domain_keys, f"{label}.domains[{grid}].settings")
    return deepcopy(value)


def _bounds(value, label: str) -> dict:
    _object(value, label, allowed={"south", "west", "north", "east"}, required={"south", "west", "north", "east"})
    for k in ("south", "north"):
        _number(value[k], label + "." + k, minimum=-90, maximum=90)
    for k in ("west", "east"):
        _number(value[k], label + "." + k, minimum=-180, maximum=180)
    if value["south"] >= value["north"] or value["west"] == value["east"]:
        raise CatalogError(f"{label} must enclose a nonzero area; west > east explicitly crosses the antimeridian")
    return value


def _span(bounds):
    raw = bounds["east"] - bounds["west"]
    return 360.0 if abs(raw) == 360 else raw % 360


def _contains(outer, inner):
    return (outer["south"] <= inner["south"] <= inner["north"] <= outer["north"]
            and ((inner["west"] - outer["west"]) % 360) + _span(inner) <= _span(outer) + 1e-10)


def _profile(value, label):
    from gpuwm.physics_menu import WIZARD_PHYSICS_PROFILES
    if value not in WIZARD_PHYSICS_PROFILES:
        raise CatalogError(f"{label} names an unknown native physics profile: {value!r}; "
                           "use an offered profile or explicitly typed native_overrides")


def validate_catalog(document: dict) -> dict:
    doc = _json_data(deepcopy(document))
    _object(doc, "catalog document", allowed={"schema", "catalog", "cases"}, required={"schema", "catalog", "cases"})
    if doc["schema"] != SCHEMA:
        raise CatalogError(f"Unsupported case catalog schema {doc['schema']!r}; expected {SCHEMA}")
    meta = _object(doc["catalog"], "catalog", allowed={"id", "title", "version", "description", "provenance", "metadata"}, required={"id", "title", "version", "provenance"})
    _id(meta["id"], "catalog.id")
    for k in ("title", "version"):
        _text(meta[k], "catalog." + k)
    if not isinstance(meta["provenance"], list):
        raise CatalogError("catalog.provenance must be a list of source/author records")
    if not isinstance(doc["cases"], list) or not doc["cases"]:
        raise CatalogError("cases must be a nonempty list")
    seen = set()
    for case in doc["cases"]:
        _object(case, "case", allowed={"id", "title", "event_kind", "synthetic", "summary", "event", "source_options", "recommended_source_option", "tiers", "physics_profile", "native_overrides", "recommendations", "tags", "references", "research_recipe_ids", "metadata"}, required={"id", "title", "event_kind", "source_options", "tiers"})
        ident = _id(case["id"], "case.id")
        if ident in seen:
            raise CatalogError(f"Duplicate case ID {ident!r}")
        seen.add(ident)
        for k in ("title", "event_kind"):
            _text(case[k], f"case {ident}.{k}")
        if "synthetic" in case and not isinstance(case["synthetic"], bool):
            raise CatalogError(f"case {ident}.synthetic must be boolean")
        if "event" in case:
            event = _object(case["event"], "event", allowed={"start_utc", "end_utc"}, required={"start_utc", "end_utc"})
            for k in event:
                event[k] = _utc(event[k], f"case {ident}.event.{k}")
            if datetime.fromisoformat(event["start_utc"].replace("Z", "+00:00")) > datetime.fromisoformat(event["end_utc"].replace("Z", "+00:00")):
                raise CatalogError(f"case {ident}: event end precedes its start")
        options = case["source_options"]
        if not isinstance(options, list) or not options:
            raise CatalogError(f"case {ident} must list at least one source/initialization option")
        option_ids = set()
        for option in options:
            _object(option, "source option", allowed={"id", "source", "cycle_utc", "forecast_start_hour", "cadence_hours", "tier_schedules", "tier_geometry", "label", "notes", "references", "metadata"}, required={"id", "source", "cycle_utc"})
            option_id = _id(option["id"], f"case {ident}.source option ID")
            if option_id in option_ids:
                raise CatalogError(f"case {ident} repeats source option {option_id}")
            option_ids.add(option_id)
            option["source"] = _source(option["source"])
            option["cycle_utc"] = _utc(option["cycle_utc"], "source option.cycle_utc", hour=True)
            if "forecast_start_hour" in option:
                _number(option["forecast_start_hour"], "forecast_start_hour", minimum=0, integer=True)
            if "cadence_hours" in option:
                _number(option["cadence_hours"], "cadence_hours", minimum=1, integer=True)
            if "tier_schedules" in option:
                schedules = _object(option["tier_schedules"], "tier_schedules", allowed=TIERS, required=TIERS)
                for schedule in schedules.values():
                    _object(schedule, "tier schedule", allowed={"cycle_utc", "run_hours", "forecast_start_hour", "cadence_hours", "metadata"}, required={"cycle_utc", "run_hours"})
                    schedule["cycle_utc"] = _utc(schedule["cycle_utc"], "tier schedule.cycle_utc", hour=True)
                    _number(schedule["run_hours"], "tier schedule.run_hours", minimum=1, integer=True)
                    _number(schedule.get("forecast_start_hour", 0), "tier schedule.forecast_start_hour", minimum=0, integer=True)
                    if "cadence_hours" in schedule:
                        _number(schedule["cadence_hours"], "tier schedule.cadence_hours", minimum=1, integer=True)
            if "tier_geometry" in option:
                # Apply the exact existing tier/physics validation to this
                # source's alternative layout. The minimal validation case
                # has no further overrides, so recursion ends after one pass.
                validate_catalog({"schema": SCHEMA, "catalog": {
                    "id": "source-geometry", "title": "Source geometry", "version": "1", "provenance": []},
                    "cases": [{"id": ident, "title": case["title"], "event_kind": case["event_kind"],
                        "source_options": [{k: option[k] for k in ("id", "source", "cycle_utc")}],
                        "tiers": option["tier_geometry"]}]})
        if case.get("recommended_source_option") is not None and case["recommended_source_option"] not in option_ids:
            raise CatalogError(f"case {ident}.recommended_source_option does not name a listed option")
        tiers = _object(case["tiers"], f"case {ident}.tiers", allowed=set(TIERS), required=TIERS)
        finest = []
        for tier in TIERS:
            row = _object(tiers[tier], f"case {ident}.{tier}", allowed={"bounds_degrees", "domain_intents", "root_dx_km", "nest_ratios", "run_hours", "nz", "history_interval_s", "physics_profile", "native_overrides", "notes"}, required={"root_dx_km", "nest_ratios", "run_hours"})
            if ("bounds_degrees" in row) == ("domain_intents" in row):
                raise CatalogError(f"{tier} must provide exactly one of bounds_degrees or domain_intents")
            if "bounds_degrees" in row:
                _bounds(row["bounds_degrees"], f"{tier}.bounds_degrees")
            _number(row["root_dx_km"], f"{tier}.root_dx_km", minimum=1e-9)
            _number(row["run_hours"], f"{tier}.run_hours", minimum=1, integer=True)
            if not isinstance(row["nest_ratios"], list):
                raise CatalogError(f"{tier}.nest_ratios must be a list of integer refinement ratios")
            for ratio in row["nest_ratios"]:
                _number(ratio, f"{tier}.nest_ratios", minimum=2, integer=True)
            if "domain_intents" in row:
                intents = row["domain_intents"]
                if not isinstance(intents, list) or len(intents) != len(row["nest_ratios"]) + 1:
                    raise CatalogError(f"{tier}.domain_intents must describe every domain in the selected ladder")
                for index, intent in enumerate(intents):
                    _object(intent, "domain intent", allowed={"grid_id", "center_lat", "center_lon", "dx_km", "width_km", "height_km"}, required={"grid_id", "center_lat", "center_lon", "dx_km", "width_km", "height_km"})
                    if intent["grid_id"] != index + 1:
                        raise CatalogError("Domain intents must use sequential native grid IDs")
                    for key in ("dx_km", "width_km", "height_km"):
                        _number(intent[key], "domain intent." + key, minimum=1e-9)
                    _number(intent["center_lat"], "domain intent.center_lat", minimum=-90, maximum=90)
                    _number(intent["center_lon"], "domain intent.center_lon", minimum=-180, maximum=180)
                    if any(intent[key] != intents[0][key] for key in ("center_lat", "center_lon")):
                        raise CatalogError("This native case builder supports centered domain intents; offset placement needs an explicit native guide edit")
                    if intent["dx_km"] != row["root_dx_km"] / math.prod(row["nest_ratios"][:index]):
                        raise CatalogError("Domain-intent spacing differs from its native nest ladder")
            finest.append(row["root_dx_km"] / math.prod(row["nest_ratios"]))
            if "nz" in row:
                _number(row["nz"], f"{tier}.nz", minimum=4, integer=True)
            if "history_interval_s" in row:
                _number(row["history_interval_s"], f"{tier}.history_interval_s", minimum=1, integer=True)
            if "physics_profile" in row:
                _profile(row["physics_profile"], f"{tier}.physics_profile")
            if "native_overrides" in row:
                validate_native_overrides(row["native_overrides"], f"{tier}.native_overrides")
        if all("bounds_degrees" in tiers[t] for t in TIERS):
            if not all(_contains(tiers[b]["bounds_degrees"], tiers[a]["bounds_degrees"]) for a, b in zip(TIERS, TIERS[1:])):
                raise CatalogError(f"case {ident}: lower footprint must fit inside recommended, and recommended inside upper")
        elif all("domain_intents" in tiers[t] for t in TIERS):
            for a, b in zip(TIERS, TIERS[1:]):
                if len(tiers[a]["domain_intents"]) != len(tiers[b]["domain_intents"]) or any(
                        small[k] > large[k] for small, large in zip(tiers[a]["domain_intents"], tiers[b]["domain_intents"])
                        for k in ("width_km", "height_km")):
                    raise CatalogError(f"case {ident}: projected domain spans must not shrink in larger tiers")
        else:
            raise CatalogError(f"case {ident}: use one geometry representation across its tiers")
        if not (finest[0] >= finest[1] >= finest[2]):
            raise CatalogError(f"case {ident}: lower/recommended/upper must progress from coarser to finer grid spacing")
        if not (tiers["lower"]["run_hours"] <= tiers["recommended"]["run_hours"] <= tiers["upper"]["run_hours"]):
            raise CatalogError(f"case {ident}: lower/recommended/upper run_hours must not decrease")
        if "physics_profile" in case:
            _profile(case["physics_profile"], f"case {ident}.physics_profile")
        if "native_overrides" in case:
            validate_native_overrides(case["native_overrides"], f"case {ident}.native_overrides")
        for k in ("recommendations", "references", "tags", "research_recipe_ids"):
            if k in case and not isinstance(case[k], list):
                raise CatalogError(f"case {ident}.{k} must be a list")
        for recipe_id in case.get("research_recipe_ids", []):
            _id(recipe_id, "research recipe ID")
    return doc


def load_catalog(path: str | Path) -> Catalog:
    path = Path(path)
    if path.suffix.lower() not in {".json", ".toml", ".zip"}:
        raise CatalogError("Choose a JSON, TOML or ZIP case catalog")
    if path.stat().st_size > MAX_CATALOG_BYTES:
        raise CatalogError("This catalog exceeds 128 MiB; split it into catalog volumes so browsing does not exhaust memory")
    raw = path.read_bytes()
    try:
        archive_report = None
        if path.suffix.lower() == ".zip":
            from gpuwm.case_catalog_import import read_archive
            document, archive_report = read_archive(raw, limit=MAX_CATALOG_BYTES, object_pairs_hook=_json_object)
        else:
            text = raw.decode("utf-8-sig")
            document = json.loads(text, object_pairs_hook=_json_object) if path.suffix.lower() == ".json" else tomllib.loads(text)
        if isinstance(document, dict) and document.get("schema") in {"arwen.case-catalog/v1", "arwen.case-catalog/v2"}:
            from gpuwm.case_catalog_import import convert_proposal_v1, convert_proposal_v2
            converter = convert_proposal_v1 if document["schema"].endswith("/v1") else convert_proposal_v2
            document = converter(document, archive_report)
    except (UnicodeDecodeError, ValueError) as error:
        raise CatalogError(f"Cannot read {path.name} as {path.suffix[1:].upper()}: {error}") from error
    return Catalog(validate_catalog(document), raw, str(path.absolute()), path.suffix[1:].lower(), hashlib.sha256(raw).hexdigest())


def _catalog(value: Catalog | str | Path) -> Catalog:
    return value if isinstance(value, Catalog) else load_catalog(value)


def list_cases(catalog, *, query: str = "", source: str | None = None,
               event_kind: str | None = None, offset: int = 0, limit: int = 100) -> dict:
    catalog = _catalog(catalog)
    if offset < 0 or not 1 <= limit <= 1000:
        raise CatalogError("Use a nonnegative offset and a limit from 1 to 1000")
    words = query.casefold().split()
    source = _source(source) if source else None
    rows = []
    for case in catalog.document["cases"]:
        sources = sorted({row["source"] for row in case["source_options"]})
        haystack = json.dumps(case, ensure_ascii=False).casefold()
        if all(word in haystack for word in words) and (not source or source in sources) and (not event_kind or case["event_kind"].casefold() == event_kind.casefold()):
            rows.append({k: deepcopy(case[k]) for k in ("id", "title", "event_kind", "synthetic", "summary", "event", "tags") if k in case} | {"sources": sources, "source_option_count": len(case["source_options"])})
    return {"schema": "arwen.case-list.v1", "provenance": catalog.provenance(),
            "total": len(rows), "offset": offset, "limit": limit, "cases": rows[offset:offset + limit]}


def case_detail(catalog, case_id: str) -> dict:
    catalog = _catalog(catalog)
    case = next((c for c in catalog.document["cases"] if c["id"] == case_id), None)
    if case is None:
        raise CatalogError(f"Case {case_id!r} is not in this catalog; use case-catalog list or search")
    return {"schema": "arwen.case-detail.v1", "provenance": catalog.provenance(), "case": deepcopy(case)}


def _merge_overrides(*values):
    shared, domains = {}, {}
    for value in values:
        if value is None:
            continue
        validate_native_overrides(value)
        shared.update(value.get("shared", {}))
        for row in value.get("domains", []):
            domains.setdefault(row["grid_id"], {}).update(row["settings"])
    return {"shared": shared, "domains": [{"grid_id": i, "settings": values} for i, values in sorted(domains.items())]}


def preview_case(catalog, case_id: str, *, tier: str = "recommended",
                 source_option: str | None = None, physics_profile: str | None = None,
                 native_overrides: dict | None = None, now: datetime | None = None) -> dict:
    catalog = _catalog(catalog)
    case = case_detail(catalog, case_id)["case"]
    if tier not in TIERS:
        raise CatalogError(f"Choose one of the case tiers: {', '.join(TIERS)}")
    selected = source_option or case.get("recommended_source_option")
    if selected is None and len(case["source_options"]) == 1:
        selected = case["source_options"][0]["id"]
    option = next((s for s in case["source_options"] if s["id"] == selected), None)
    if option is None:
        raise CatalogError("Choose a listed source/initialization option: " + ", ".join(s["id"] for s in case["source_options"]))
    row = deepcopy(option.get("tier_geometry", case["tiers"])[tier])
    option = deepcopy(option)
    if tier in option.get("tier_schedules", {}):
        schedule = option["tier_schedules"][tier]
        option["cycle_utc"] = schedule["cycle_utc"]
        option["forecast_start_hour"] = schedule.get("forecast_start_hour", 0)
        row["run_hours"] = schedule["run_hours"]
        if "cadence_hours" in schedule:
            option["cadence_hours"] = schedule["cadence_hours"]
    from gpuwm.source_availability import availability, validate_cycle
    cadence = option.get("cadence_hours", 1)
    horizon = math.ceil(row["run_hours"] / cadence) * cadence + option.get("forecast_start_hour", 0)
    try:
        source_info = availability(option["source"], horizon, now=now)
        cycle, notes = validate_cycle(source_info, option["cycle_utc"])
    except ValueError as error:
        raise CatalogError(f"Source option {option['id']}: {error}") from error
    profile = physics_profile or row.get("physics_profile") or case.get("physics_profile")
    if profile is not None:
        _profile(profile, "selected physics profile")
    overrides = _merge_overrides(case.get("native_overrides"), row.get("native_overrides"), native_overrides)
    for domain in overrides["domains"]:
        if domain["grid_id"] > 1 + len(row["nest_ratios"]):
            raise CatalogError(f"Native override names d{domain['grid_id']:02d}, but tier {tier} creates only {1 + len(row['nest_ratios'])} domain(s)")
    import_notes = deepcopy(option.get("metadata", {}).get("conversion", case.get("metadata", {}).get("conversion", {})))
    status = option.get("tier_schedules", {}).get(tier, {}).get("metadata", {}).get("source_status", {})
    import_issues = list(import_notes.get("issues", [])) + list(status.get("blocking_reasons", []))
    return {"schema": "arwen.case-preview.v1", "provenance": catalog.provenance(),
            "case_id": case_id, "title": case["title"], "synthetic": case.get("synthetic", False),
            "tier": tier, "geometry": deepcopy(row), "source_option": deepcopy(option),
            "source": source_info["source_id"], "cycle": cycle,
            "source_availability": source_info, "source_notes": notes,
            "physics_profile": profile, "native_overrides": overrides,
            "override_order": ["case", "selected tier", "explicit caller override"],
            "recommendations": deepcopy(case.get("recommendations", [])),
            "research_recipe_ids": list(case.get("research_recipe_ids", [])),
            "import_issues": import_issues,
            "import_notes": import_notes,
            "native_admission": "Configuration, source compatibility and memory are validated during create; catalog recommendations are not scientific verification.",
            "forecast_started": False}


def _config_text(raw: dict) -> str:
    from gpuwm.domain_wizard import _render_table
    parts = ["# Created from an ArWen case catalog through the native domain builder.\n"
             "# Exact catalog provenance, selected profile and typed overrides are in the .arwen-case.json receipt.\n"]
    for name, value in raw.items():
        if isinstance(value, dict):
            parts.append(_render_table(name, value))
        elif isinstance(value, list) and all(isinstance(row, dict) for row in value):
            parts.extend(_render_table(name, row, array_of_tables=True) for row in value)
        else:
            raise CatalogError(f"The native builder emitted an unsupported top-level table {name}; refusing to drop it")
    return "\n".join(parts)


def create_case(catalog, case_id: str, *, out: str | Path, tier="recommended",
                source_option=None, physics_profile=None, native_overrides=None,
                card=None, vram_gib=None, acknowledgements=(), now=None,
                expected_catalog_sha256: str | None = None) -> dict:
    catalog = _catalog(catalog)
    if expected_catalog_sha256 is not None and catalog.sha256 != expected_catalog_sha256:
        raise CatalogError("The case catalog changed after the preview. Reload the case details before creating the configuration.")
    selection = preview_case(catalog, case_id, tier=tier, source_option=source_option,
                             physics_profile=physics_profile, native_overrides=native_overrides, now=now)
    if selection["import_issues"]:
        raise CatalogError("This catalog selection has unsupported settings: " + "; ".join(selection["import_issues"]))
    from gpuwm import domain_wizard as wizard
    from gpuwm.research_workspaces import _admission, _publish_bundle
    destination = Path(out).absolute()
    if destination.suffix.lower() != ".toml":
        raise CatalogError("Create requires a new .toml configuration path")
    if os.path.lexists(destination):
        raise FileExistsError(f"The configuration already exists and will be preserved: {destination}")
    sizing = wizard.resolve_sizing_budget(card, vram_gib)
    geometry = selection["geometry"]
    bounds = geometry.get("bounds_degrees")
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    with tempfile.TemporaryDirectory(prefix=".arwen-case-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        polygon = stage / (destination.stem + ".footprint.geojson")
        if bounds is not None:
            west, east = bounds["west"], bounds["east"]
            ring = [[west, bounds["south"]], [east, bounds["south"]],
                    [east, bounds["north"]], [west, bounds["north"]],
                    [west, bounds["south"]]]
            polygon.write_text(json.dumps({"type": "Polygon", "coordinates": [ring]}) + "\n", encoding="utf-8")
        staged = stage / destination.name
        parser = argparse.ArgumentParser()
        commands = parser.add_subparsers(dest="command", required=True)
        wizard.register_cli(commands)
        target = (f"--polygon={polygon}" if bounds is not None else
                  f"--point={geometry['domain_intents'][0]['center_lat']},{geometry['domain_intents'][0]['center_lon']}")
        argv = ["domain", target, f"--source={selection['source']}",
                f"--cycle={selection['cycle']}", f"--hours={geometry['run_hours']}",
                f"--root-dx={geometry['root_dx_km']}", f"--out={staged}", f"--name={case_id}",
                f"--data-dir={destination.parent / 'data' / destination.stem}",
                f"--forecast-start-hour={selection['source_option'].get('forecast_start_hour', 0)}"]
        if geometry["nest_ratios"]:
            argv.append("--chain=" + ",".join(map(str, geometry["nest_ratios"])))
        if selection["physics_profile"]:
            argv.append("--physics-profile=" + selection["physics_profile"])
        if "nz" in geometry:
            argv.append(f"--nz={geometry['nz']}")
        if "history_interval_s" in geometry:
            argv.extend([f"--history-interval={geometry['history_interval_s']}", f"--nest-history-interval={geometry['history_interval_s']}"])
        argv.extend("--ack=" + _text(ack, "explicit acknowledgement") for ack in acknowledgements)
        native = parser.parse_args(argv)
        with redirect_stdout(log), redirect_stderr(log):
            result = wizard.domain_main(native, sizing_budget=sizing)
        if result:
            raise CatalogError(f"The native domain builder could not create this selection (exit {result}):\n{log.getvalue()}")
        text = staged.read_text(encoding="utf-8").replace(str(stage), str(destination.parent)).replace(stage.as_posix(), destination.parent.as_posix())
        raw = tomllib.loads(text)
        vtable = stage / wizard._PACKAGED_VTABLE.name
        if vtable.is_file() and "case_data" in raw:
            unique_vtable = stage / (destination.stem + "." + vtable.name)
            vtable.rename(unique_vtable)
            raw["case_data"]["vtable"] = unique_vtable.name
        cadence = selection["source_option"].get("cadence_hours")
        if cadence is not None:
            if "fetch" in raw:
                raw["fetch"]["hours"] = math.ceil(geometry["run_hours"] / cadence) * cadence
                if selection["source"] in wizard._SOURCE_CADENCE_H:
                    raw["fetch"]["cadence"] = cadence
            if "case_data" in raw:
                raw["case_data"]["forcing_interval_s"] = cadence * 3600
        if "domain_intents" in geometry:
            from fractions import Fraction
            ratios = tuple(geometry["nest_ratios"])
            dims = []
            for index, intent in enumerate(geometry["domain_intents"]):
                multiple = 2 * (ratios[index - 1] if index else 1)
                dims.append(tuple(int(wizard._round_up_multiple(intent[key] / intent["dx_km"], multiple))
                                  for key in ("width_km", "height_km")))
            first = raw["domain"][0]
            clock = Fraction(first["time_step"]) + Fraction(first.get("time_step_fract_num", 0), first.get("time_step_fract_den", 1))
            profile = wizard.resolved_physics_profile(selection["source"], selection["physics_profile"])
            raw["domain"] = wizard._domain_tables(dims, ratios, time_step=clock,
                root_dx_m=geometry["root_dx_km"] * 1000, profile=profile, cumulus_requested=True,
                history_interval_s=geometry.get("history_interval_s"),
                nest_history_interval_s=geometry.get("history_interval_s"))
            problem = wizard.source_coverage_refusal(raw["projection"], *dims[0], source=selection["source"], root_dx_m=geometry["root_dx_km"] * 1000)
            if problem:
                raise CatalogError(problem)
            if "area" in raw.get("fetch", {}):
                raw["fetch"]["area"] = wizard.fetch_area_hint(raw["projection"], *dims[0], source=selection["source"], root_dx_m=geometry["root_dx_km"] * 1000)
            (stage / (destination.stem + ".namelist.wps")).write_text(
                wizard.render_wps_namelist(raw["projection"], dims, ratios,
                    root_dx_m=geometry["root_dx_km"] * 1000, source=selection["source"],
                    forcing_interval_seconds=cadence * 3600 if cadence is not None else None), encoding="utf-8")
        # Keep every generated native table. Only the explicitly selected,
        # already vocabulary-checked scientific fields are changed.
        raw["shared"].update(selection["native_overrides"]["shared"])
        domains = {row["grid_id"]: row for row in raw["domain"]}
        # A global choice must not be shadowed by the wizard's explicit
        # per-domain defaults. Explicit catalog/caller domain choices win next.
        for domain in domains.values():
            for key, value in selection["native_overrides"]["shared"].items():
                if key in domain:
                    domain[key] = value
        for row in selection["native_overrides"]["domains"]:
            domains[row["grid_id"]].update(row["settings"])
        text = _config_text(raw)
        recipe = {"id": case_id, "method": "case catalog selection", "geometry": {"minimum_root_span_km": 0}, "validation_status": "catalog recommendations are not science validation"}
        with redirect_stdout(log), redirect_stderr(log):
            experiment, admission = _admission(
                text, recipe=recipe, source=selection["source"], sizing=sizing, path=destination,
                retry_hint="Choose a smaller tier from this catalog or make more GPU memory available. "
                           "Keep the declared capacity equal to the target GPU.")
            if cadence is not None:
                phases = wizard._sizing_phases(experiment, free_bytes=sizing.free_bytes,
                    source=selection["source"], forcing_interval_seconds=cadence * 3600,
                    vram_gib=sizing.vram_gib, profile=sizing.device_profile)
                budget = wizard.sizing_budget_bytes(experiment, free_bytes=sizing.free_bytes,
                    forcing_interval_seconds=cadence * 3600, vram_gib=sizing.vram_gib, profile=sizing.device_profile)
                if phases.peak_envelope_bytes > budget:
                    raise CatalogError(f"The requested {cadence}h source cadence and complete domain tree need {phases.peak_envelope_bytes} bytes, beyond the {budget}-byte budget")
                admission.update(binding_phase=phases.binding_phase, peak_envelope_bytes=phases.peak_envelope_bytes,
                                 envelope_budget_bytes=budget, remaining_envelope_bytes=budget-phases.peak_envelope_bytes,
                                 forcing_interval_seconds=cadence * 3600)
        requested_domains = {row["grid_id"]: row["settings"] for row in selection["native_overrides"]["domains"]}
        actual_settings = []
        for domain in experiment.domains:
            expected = dict(selection["native_overrides"]["shared"])
            expected.update(requested_domains.get(domain.grid_id, {}))
            actual = {key: getattr(domain.run, key) for key in expected}
            for key, value in expected.items():
                if actual[key] != value:
                    raise CatalogError(f"Native d{domain.grid_id:02d}.{key} resolved to {actual[key]!r}, not the explicit {value!r}; refusing to silently change a catalog setting")
            actual_settings.append({"grid_id": domain.grid_id, "settings": actual})
        staged.write_text(text, encoding="utf-8", newline="\n")
        original_name = destination.stem + ".case-catalog.original." + catalog.format
        (stage / original_name).write_bytes(catalog.original)
        receipt = {"schema": "arwen.case-configuration.v1", "configuration": str(destination),
                   "created_utc": datetime.now(timezone.utc).isoformat(), "selection": selection,
                   "original_catalog": str(destination.parent / original_name),
                   "original_catalog_sha256": catalog.sha256,
                   "selected_case": case_detail(catalog, case_id)["case"],
                   "config_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                   "explicit_acknowledgements": list(acknowledgements), "admission": admission,
                   "actual_native_overrides": actual_settings,
                   "domains": [{"grid_id": d.grid_id, "nx": d.run.nx, "ny": d.run.ny, "nz": d.run.nz, "dx_km": d.run.dx / 1000} for d in experiment.domains],
                   "native_creation_log": log.getvalue().replace(str(stage), str(destination.parent)).replace(stage.as_posix(), destination.parent.as_posix()),
                   "forecast_started": False, "next_command": ["gpuwm", "go", str(destination)]}
        receipt["files"] = [str(destination.parent / p.name) for p in sorted(stage.iterdir())]
        receipt["files"].append(str(destination) + ".arwen-case.json")
        (stage / (destination.name + ".arwen-case.json")).write_text(json.dumps(receipt, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        published = _publish_bundle(stage, destination)
        receipt["files"] = [str(p) for p in published]
    return receipt


def export_catalog(catalog, out: str | Path, *, original=False) -> dict:
    catalog = _catalog(catalog)
    path = Path(out).absolute()
    payload = catalog.original if original else (json.dumps(catalog.document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return {"schema": "arwen.case-export.v1", "path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "original": original, "provenance": catalog.provenance()}


def catalog_main(args) -> int:
    try:
        if args.catalog_command == "native-settings":
            result = native_settings_document()
        else:
            catalog = load_catalog(args.catalog)
            if args.catalog_command in {"list", "search"}:
                result = list_cases(catalog, query=args.query, source=args.source, event_kind=args.event_kind, offset=args.offset, limit=args.limit)
            elif args.catalog_command == "show":
                result = case_detail(catalog, args.case_id)
            elif args.catalog_command == "export":
                result = export_catalog(catalog, args.out, original=args.original)
            else:
                overrides = None
                if args.native_overrides is not None:
                    overrides = json.loads(args.native_overrides.read_text(encoding="utf-8"), object_pairs_hook=_json_object)
                kwargs = dict(tier=args.tier, source_option=args.source_option, physics_profile=args.physics_profile, native_overrides=overrides)
                if args.catalog_command == "create":
                    result = create_case(catalog, args.case_id, out=args.out, card=args.card, vram_gib=args.vram_gib, acknowledgements=args.ack, expected_catalog_sha256=args.expected_catalog_sha256, **kwargs)
                else:
                    result = preview_case(catalog, args.case_id, **kwargs)
        print(json.dumps(result, ensure_ascii=True, indent=None if args.json else 2, allow_nan=False, default=str))
        return 0
    except (ValueError, OSError) as error:
        import sys
        if args.json:
            print(json.dumps({"schema": "arwen.case-error.v1", "error": str(error), "created": False}, ensure_ascii=True))
        else:
            print(f"case-catalog: {error}", file=sys.stderr)
        return 2


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("case-catalog", help="browse historical-case catalogs and create native configurations")
    commands = parser.add_subparsers(dest="catalog_command", required=True)
    for name in ("list", "search", "show", "preview", "create", "export", "native-settings"):
        command = commands.add_parser(name)
        command.add_argument("--json", action="store_true", help="emit compact JSON for the interface or scripts")
        command.set_defaults(func=catalog_main)
        if name == "native-settings":
            continue
        command.add_argument("--catalog", type=Path, required=True, help="JSON, TOML or ZIP case catalog; never interpreted as commands")
        if name in {"list", "search"}:
            command.add_argument("--query", default="")
            command.add_argument("--source")
            command.add_argument("--event-kind")
            command.add_argument("--offset", type=int, default=0)
            command.add_argument("--limit", type=int, default=100)
        if name in {"show", "preview", "create"}:
            command.add_argument("case_id")
        if name in {"preview", "create"}:
            command.add_argument("--tier", choices=TIERS, default="recommended")
            command.add_argument("--source-option", help="listed source/initialization ID; default is the catalog's recommended option")
            command.add_argument("--physics-profile", help="explicit native profile replacing the catalog's selected profile")
            command.add_argument("--native-overrides", type=Path, help="JSON shared/domains scientific overrides; validated against the native contract")
        if name == "create":
            command.add_argument("--out", type=Path, required=True, help="new experiment .toml; existing files are preserved")
            command.add_argument("--expected-catalog-sha256", help="bind creation to the exact original catalog bytes displayed by preview")
            command.add_argument("--card")
            command.add_argument("--vram-gib", type=float)
            command.add_argument("--ack", action="append", default=[], help="explicit native scientific acknowledgement; never inferred from catalog prose")
        if name == "export":
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--original", action="store_true", help="export the exact original JSON/TOML bytes; default is normalized JSON")
