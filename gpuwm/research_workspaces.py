"""Compile reviewed research recipes through ArWen's native domain planner.

Creation writes a new configuration bundle only. It does not acquire forcing,
prepare a state, integrate a forecast, or turn pending research validation into
a successful scientific result. The TOML is published last as the commit marker.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile


CATALOG_PATH = Path(__file__).parent / "data" / "tui" / "research-workspaces.json"
HARDWARE_PATH = CATALOG_PATH.with_name("research-hardware-profiles.json")
DIAGNOSTICS_PATH = CATALOG_PATH.with_name("research-diagnostics.json")
HARDWARE_CLASSES = ("8", "12", "16", "24", "32")
_METHODS = {"regional", "nested", "moving_nest", "controlled_scenario",
            "archived_downscale"}
_RECIPE_KEYS = {"id", "leaf_id", "title", "research_question", "method",
                "plot_preset", "geometry", "diagnostics", "input_requirements",
                "tracker", "scenario", "comparison", "limitations",
                "citation_ids", "requires_existing_state", "validation_status", "further_analysis"}
_GEOMETRY_KEYS = {"root_dx_km", "nest_ratios", "forecast_hours",
                  "history_interval_s", "extent_intent", "preferred_finest_dx_km",
                  "minimum_domain_span_km", "minimum_root_span_km"}
_TRACKER_KEYS = {"kind", "field", "units", "level_hpa", "threshold",
                 "fallback_threshold", "cadence_seconds", "notes", "attribute",
                 "extremum", "reduction", "model_level"}
_SCENARIO_KEYS = {"operation", "amplitude_k", "center_height_m", "radius_km",
                  "depth_m", "rh_preserve", "placement"}


def _read_json(path: Path, label: str) -> tuple[dict, str]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read packaged {label} at {path}: {error}. "
                         "Restore that file from the matching ArWen installation, then retry.") from error
    if not isinstance(document, dict):
        raise ValueError(f"Packaged {label} at {path} must be a JSON object. "
                         "Restore the matching ArWen package data, then retry.")
    return document, hashlib.sha256(payload).hexdigest()


def _hardware_document() -> tuple[dict, str]:
    document, digest = _read_json(HARDWARE_PATH, "research hardware profiles")
    try:
        if document.get("schema") != "arwen.research.hardware-profiles.v1":
            raise ValueError("expected schema arwen.research.hardware-profiles.v1")
        profiles = document["profiles"]
        if not isinstance(profiles, dict) or set(profiles) != set(HARDWARE_CLASSES):
            raise ValueError("profiles must declare classes 8, 12, 16, 24 and 32")
        if not isinstance(document["policy"], str) or not document["policy"].strip():
            raise ValueError("policy must be nonempty text")
        for name, profile in profiles.items():
            if not isinstance(profile, dict):
                raise ValueError(f"profile {name} must be an object")
            nz = profile["nz"]
            if isinstance(nz, bool) or not isinstance(nz, int) or nz < 2:
                raise ValueError(f"profile {name} nz must be an integer >= 2")
            if not isinstance(profile["tradeoff"], str):
                raise ValueError(f"profile {name} tradeoff must be text")
            for intent in ("regional", "mesoscale", "storm", *(("storm_context",) if "storm_context" in profile else ())):
                geometry = profile[intent]
                _number(geometry["root_dx_km"], f"profile {name} {intent} root_dx_km")
                ratios = geometry["nest_ratios"]
                if not isinstance(ratios, list) or any(isinstance(r, bool) or not isinstance(r, int) or r < 2 for r in ratios):
                    raise ValueError(f"profile {name} {intent} nest_ratios must contain integer ratios >= 2")
            if "storm_context" in profile:
                _number(profile["storm_context_minimum_km"], f"profile {name} storm_context_minimum_km")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid research hardware profiles at {HARDWARE_PATH}: {error}. "
                         "Restore the matching ArWen package data, then retry.") from error
    return document, digest


def _keys(value: dict, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(unknown)}")


def _number(value, label: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{label} must be a finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def validate_recipe(recipe: dict, *, known_products: set[str] | None = None,
                    _capabilities: dict | None = None) -> None:
    """Reject unsupported declarations before probing hardware or writing files."""
    _keys(recipe, _RECIPE_KEYS, "recipe")
    missing = _RECIPE_KEYS - {"further_analysis"} - set(recipe)
    if missing:
        raise ValueError(f"Research recipe is missing fields: {', '.join(sorted(missing))}")
    if not isinstance(recipe["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", recipe["id"]):
        raise ValueError("Research recipe id must use lowercase letters, digits, dots, underscores or hyphens")
    for key in ("leaf_id", "title", "research_question", "comparison", "validation_status"):
        if not isinstance(recipe[key], str) or not recipe[key].strip():
            raise ValueError(f"Research recipe {key} must be nonempty text")
    for key in ("input_requirements", "limitations", "citation_ids", "further_analysis"):
        values = recipe.get(key, [])
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"Research recipe {key} must be a list of nonempty strings")
    method = recipe["method"]
    if method not in _METHODS:
        raise ValueError(f"Unsupported research method: {method}")
    geometry = recipe["geometry"]
    _keys(geometry, _GEOMETRY_KEYS, "geometry")
    if _GEOMETRY_KEYS - set(geometry):
        raise ValueError("Research geometry must declare its complete intent and minimum root span")
    for key in ("root_dx_km", "forecast_hours", "history_interval_s", "preferred_finest_dx_km"):
        _number(geometry.get(key), f"geometry.{key}")
    if int(geometry["forecast_hours"]) != geometry["forecast_hours"]:
        raise ValueError("geometry.forecast_hours must be a whole number of hours")
    ratios = geometry.get("nest_ratios")
    if not isinstance(ratios, list) or any(isinstance(ratio, bool) or not isinstance(ratio, int)
                                          or ratio < 2 for ratio in ratios):
        raise ValueError("geometry.nest_ratios must contain integer ratios >= 2")
    if method == "regional" and ratios:
        raise ValueError("A regional recipe cannot silently create nests")
    if method in {"nested", "moving_nest", "controlled_scenario"} and not ratios:
        raise ValueError(f"{method} requires at least two domains")
    if "minimum_root_span_km" in geometry:
        _number(geometry["minimum_root_span_km"], "geometry.minimum_root_span_km", positive=False)
    if not isinstance(recipe["requires_existing_state"], bool):
        raise ValueError("requires_existing_state must be a boolean")
    if method == "archived_downscale" and not recipe["requires_existing_state"]:
        raise ValueError("Archived downscaling requires an existing parent state")
    tracker = recipe["tracker"]
    if (method == "moving_nest") != (tracker is not None):
        raise ValueError("A tracker must be present exactly for moving_nest recipes")
    if tracker is not None:
        _keys(tracker, _TRACKER_KEYS, "tracker")
        if tracker.get("kind") != "feature_follow":
            raise ValueError("Only feature_follow tracking is supported")
        if tracker.get("field") not in {"pressure", "uh", "reflectivity", "attribute"}:
            raise ValueError("Trackers support pressure, uh, reflectivity or native attribute only")
        from gpuwm.core.attribute_tracking import ATTRIBUTE_UNITS
        units = (ATTRIBUTE_UNITS.get(tracker.get("attribute")) if tracker["field"] == "attribute" else
                 "m2/s2" if tracker["field"] == "uh" else
                 "dBZ" if tracker["field"] == "reflectivity" else
                 "hPa" if tracker.get("level_hpa") == 0 else "m")
        if tracker.get("units") != units:
            raise ValueError(f"Tracker {tracker['field']} at this surface requires units {units}")
        _number(tracker.get("cadence_seconds"), "tracker.cadence_seconds")
        # The native validator owns the field-dependent threshold/level units.
        from gpuwm.core.storm_tracking import build_follow_config
        build_follow_config(_follow_values(tracker), recipe["id"])
    scenario = recipe["scenario"]
    if (method == "controlled_scenario") != (scenario is not None):
        raise ValueError("A scenario must be present exactly for controlled_scenario recipes")
    if scenario is not None:
        _keys(scenario, _SCENARIO_KEYS, "scenario")
        if scenario.get("operation") != "warm_bubble":
            raise ValueError("Only the native warm_bubble initial perturbation is supported")
        if scenario.get("placement") != "user-selected domain centre":
            raise ValueError("Warm bubbles require the user-selected domain centre")
        from gpuwm.experiment import BubbleConfig
        BubbleConfig(center_lat=0, center_lon=0, **{
            key: scenario[key] for key in _SCENARIO_KEYS - {"operation", "placement"}})
    from gpuwm.tui_products import presets
    preset = next((row for row in presets()["presets"] if row["id"] == recipe["plot_preset"]), None)
    if preset is None:
        raise ValueError(f"Unknown plot preset: {recipe['plot_preset']}")
    capabilities = diagnostic_capabilities() if _capabilities is None else _capabilities
    if known_products is None:
        known_products = set(capabilities["products"])
    if not isinstance(recipe["diagnostics"], list) or not recipe["diagnostics"] \
            or any(not isinstance(product, str) or product not in known_products for product in recipe["diagnostics"]):
        raise ValueError("Recipe diagnostics must name supported native renderer products")
    for product in recipe["diagnostics"]:
        if product not in capabilities["products"]:
            reason = capabilities["unavailable"].get(product, "This selector has not been qualified for the ArWen history path.")
            raise ValueError(f"Research diagnostic {product} is not supported by the ArWen history renderer. {reason}")
        minimum = capabilities["products"][product]["minimum_hours"]
        if geometry["forecast_hours"] < minimum:
            raise ValueError(f"Research diagnostic {product} needs at least {minimum} hours of history; {recipe['id']} declares {geometry['forecast_hours']:g} hours. Select a matching window or a longer study.")


def diagnostic_capabilities() -> dict:
    document, digest = _read_json(DIAGNOSTICS_PATH, "research diagnostic capabilities")
    try:
        if document.get("schema") != "arwen.research.diagnostics.v1":
            raise ValueError("expected schema arwen.research.diagnostics.v1")
        if not isinstance(document["products"], dict) or not document["products"]:
            raise ValueError("products must be a nonempty object")
        if not isinstance(document["unavailable"], dict):
            raise ValueError("unavailable must be an object")
        for name, entry in document["products"].items():
            minimum = entry["minimum_hours"]
            if not isinstance(name, str) or not isinstance(entry["kind"], str) or isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
                raise ValueError(f"invalid diagnostic capability {name}")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid research diagnostic capabilities at {DIAGNOSTICS_PATH}: {error}. Restore the matching ArWen package data, then retry.") from error
    return {**document, "sha256": digest}


def _native_products() -> set[str]:
    """Packaged input capabilities; creating TOML never needs a renderer process."""
    return set(diagnostic_capabilities()["products"])


def catalog_document() -> dict:
    document, catalog_digest = _read_json(CATALOG_PATH, "research catalog")
    if document.get("schema_version") != 1 or document.get("catalog_id") != "arwen-research-workspaces-v1":
        raise ValueError(f"Unsupported research catalog schema at {CATALOG_PATH}; expected arwen-research-workspaces-v1 / schema_version 1. Restore the matching ArWen package data, then retry.")
    capabilities = diagnostic_capabilities()
    try:
        collections = {}
        for key in ("families", "submodes", "leaves", "configurations", "citations"):
            rows = document[key]
            if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows):
                raise ValueError(f"{key} must be a nonempty list of objects with string ids")
            collections[key] = {row["id"]: row for row in rows}
            if len(collections[key]) != len(rows):
                raise ValueError(f"duplicate id in {key}")
        for family in collections["families"].values():
            if set(family["submode_ids"]) != {row["id"] for row in collections["submodes"].values() if row["family_id"] == family["id"]}:
                raise ValueError(f"family {family['id']} has inconsistent submode_ids")
        for submode in collections["submodes"].values():
            if submode["family_id"] not in collections["families"]:
                raise ValueError(f"submode {submode['id']} has unknown family_id")
            if set(submode["leaf_ids"]) != {row["id"] for row in collections["leaves"].values() if row["submode_id"] == submode["id"]}:
                raise ValueError(f"submode {submode['id']} has inconsistent leaf_ids")
        for leaf in collections["leaves"].values():
            if leaf["submode_id"] not in collections["submodes"]:
                raise ValueError(f"leaf {leaf['id']} has unknown submode_id")
            if set(leaf["configuration_ids"]) != {row["id"] for row in collections["configurations"].values() if row["leaf_id"] == leaf["id"]}:
                raise ValueError(f"leaf {leaf['id']} has inconsistent configuration_ids")
        for row in collections["configurations"].values():
            validate_recipe(row, _capabilities=capabilities)
            if row["leaf_id"] not in collections["leaves"]:
                raise ValueError(f"configuration {row['id']} has unknown leaf_id")
            if any(citation not in collections["citations"] for citation in row["citation_ids"]):
                raise ValueError(f"configuration {row['id']} has an unknown citation")
        for key in ("families", "submodes"):
            for row in collections[key].values():
                if any(config not in collections["configurations"] for config in row["recommended_config_ids"]):
                    raise ValueError(f"{row['id']} recommends an unknown configuration")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid research catalog at {CATALOG_PATH}: {error}. Restore the matching ArWen package data, then retry.") from error
    hardware, hardware_digest = _hardware_document()
    return {**document, "hardware_profiles": hardware["profiles"],
            "hardware_policy": hardware["policy"], "catalog_sha256": catalog_digest,
            "hardware_policy_sha256": hardware_digest,
            "diagnostic_capabilities": capabilities}


def effective_profile(recipe: dict, hardware_class: str, *, hardware: dict | None = None) -> dict:
    """The exact visible policy shared by catalog consumers and the compiler."""
    if hardware is None:
        hardware, _ = _hardware_document()
    profile = hardware["profiles"][hardware_class]
    original = recipe["geometry"]
    parent_dependent = recipe["requires_existing_state"] and recipe["method"] != "controlled_scenario"
    if parent_dependent:
        intent, nz = "existing-parent" if recipe["method"] == "archived_downscale" else "existing-scenario", None
        geometry = dict(original)
        tradeoff = "The existing state determines the actual parent grid and vertical levels; class selection cannot replace that evidence."
    elif recipe["method"] == "controlled_scenario":
        intent, nz = "controlled-comparison", 49
        geometry = dict(original)
        tradeoff = "Controlled comparisons preserve the same horizontal ladder and 49 levels across classes."
    else:
        intent = ("regional" if not original["nest_ratios"] else
                  "mesoscale" if original["preferred_finest_dx_km"] >= 3 else "storm")
        context = (intent == "storm" and "storm_context" in profile
                   and original["minimum_root_span_km"] >= profile["storm_context_minimum_km"])
        geometry = {**original, **profile["storm_context" if context else intent]}
        nz = profile["nz"]
        tradeoff = profile["tradeoff"]
        if context:
            tradeoff += (f" This recipe requires {original['minimum_root_span_km']:g} km root span, "
                         f"so it selects the three-domain context ladder at the "
                         f"{profile['storm_context_minimum_km']:g} km policy boundary.")
    finest = geometry["root_dx_km"] / math.prod(geometry["nest_ratios"])
    return {"hardware_class": hardware_class, "intent": intent, "nz": nz,
            "geometry": geometry, "requested_geometry": original,
            "effective_finest_dx_km": finest, "tradeoff": tradeoff,
            "parent_dependent": parent_dependent,
            "resolution_tradeoff": ("Existing parent/state metadata must resolve this geometry"
                                    if parent_dependent else
                                    "Profile is coarser than the recipe's preferred finest spacing"
                                    if finest > original["preferred_finest_dx_km"] else
                                    "Profile meets or exceeds the recipe's preferred finest spacing"),
            "validation_status": "unvalidated-candidate-policy"}


def _recipe(recipe_id: str, *, document: dict | None = None) -> dict:
    document = catalog_document() if document is None else document
    matches = [row for row in document["configurations"] if row.get("id") == recipe_id]
    if not matches:
        raise ValueError(f"Unknown research configuration {recipe_id}. Run gpuwm research catalog to choose an existing id.")
    if len(matches) != 1:
        raise ValueError(f"Duplicate research configuration {recipe_id} in {CATALOG_PATH}; restore the matching package data.")
    validate_recipe(matches[0], _capabilities=document.get("diagnostic_capabilities"))
    return matches[0]


def _hardware_class(sizing, requested: str) -> str:
    if requested != "auto":
        return requested
    from gpuwm.domain_wizard import card_assumed_free_gib
    # A busy large card should receive a practical small profile. This selects
    # geometry only: the original free-byte sample still prices every candidate.
    # CUDA may report 15.51 GiB on a nominal 16 GiB card after driver
    # reservations. The profile's assumed-free allowance is the comparable
    # quantity, not its marketing capacity. Never increase the measured budget.
    available_gib = min(sizing.vram_gib, sizing.free_bytes / 2**30)
    eligible = [value for value in HARDWARE_CLASSES
                if card_assumed_free_gib(float(value)) <= available_gib + 1e-9]
    return eligible[-1] if eligible else HARDWARE_CLASSES[0]


def _budget_document(sizing, requested: str, *, hardware: dict | None = None) -> dict:
    from gpuwm.domain_wizard import fit_headroom_bytes
    from gpuwm.core.preflight import EXTERNAL_MARGIN_BYTES
    selected = _hardware_class(sizing, requested)
    envelope_budget = sizing.free_bytes - EXTERNAL_MARGIN_BYTES
    device = sizing.device_profile
    if hardware is None:
        hardware, _ = _hardware_document()
    return {
        "requested_class": requested, "selected_class": selected,
        "selection_reason": (f"{sizing.vram_gib:g} GiB GPU, {sizing.free_bytes / 2**30:.2f} GiB free "
                             f"selects the {selected} GiB resource profile" if requested == "auto"
                             else f"Explicit {selected} GiB profile; the memory budget is independent"),
        "profile": hardware["profiles"][selected],
        "basis": "measured" if sizing.measured else "declared-capacity-assumed-free",
        "capacity_gib": sizing.vram_gib, "free_bytes": sizing.free_bytes,
        "external_margin_bytes": EXTERNAL_MARGIN_BYTES,
        "envelope_budget_bytes": envelope_budget,
        "point_fit_headroom_bytes": fit_headroom_bytes(envelope_budget),
        "device_profile": None if device is None else vars(device),
        "note": sizing.note,
    }


def hardware_document(*, hardware_class: str = "auto", vram_gib: float | None = None) -> dict:
    from gpuwm.domain_wizard import resolve_sizing_budget
    sizing = resolve_sizing_budget(None, vram_gib)
    return {"schema": "arwen.research.hardware.v1", **_budget_document(sizing, hardware_class)}


def attributes_document() -> dict:
    """Discover the native closed registry without probing a device or state."""
    from gpuwm.core.attribute_tracking import (ATTRIBUTE_UNITS, ATTRIBUTE_EXTREMA,
                                               ATTRIBUTE_REDUCTIONS)
    meanings = {
        "theta": "Total potential temperature: base state plus perturbation",
        "qv": "Water-vapour mixing ratio per kg of dry air",
        "qc": "Cloud-water mixing ratio per kg of dry air",
        "qr": "Rain-water mixing ratio per kg of dry air",
        "w": "Vertical velocity averaged from adjacent faces onto mass levels",
    }
    return {
        "schema": "arwen.research.attributes.v1",
        "field": "attribute",
        "attributes": [{"id": name, "units": units, "meaning": meanings[name],
                        "requires_moist": name in {"qv", "qc", "qr"}}
                       for name, units in ATTRIBUTE_UNITS.items()],
        "extrema": list(ATTRIBUTE_EXTREMA),
        "reductions": list(ATTRIBUTE_REDUCTIONS),
        "reduction_semantics": {
            "column_max": "Maximum across all source mass levels",
            "column_min": "Minimum across all source mass levels",
            "column_mean": "Unweighted arithmetic mean across all source mass levels",
            "model_level": "One explicitly selected zero-based source mass level",
        },
        "model_level": {"type": "integer", "minimum": 0, "index_basis": "zero-based mass levels",
                        "required_when": "reduction = model_level",
                        "allowed_when": "reduction = model_level",
                        "upper_bound": "strictly less than nz on every source grid"},
        "threshold": {"units": "the selected attribute's units",
                      "max": "select values at or above threshold",
                      "min": "select values at or below threshold"},
        "constraints": [
            "Declare attribute, extremum and reduction explicitly with field = attribute",
            "Omit pressure level_hpa and reflectivity fallback_threshold",
            "Nonfinite values in any sampled level make that column ineligible",
            "Only the listed native attributes and reductions are supported; expressions are not evaluated",
            "Native source-state, movement, timing and memory admission still apply",
        ],
    }


def _follow_values(tracker: dict) -> dict:
    values = {"field": tracker.get("field"), "threshold": tracker.get("threshold"),
              "search_margin_cells": 10, "min_shift_cells": 1,
              "max_shift_cells": 4, "cooldown_seconds": 300.0, "radius_km": 50.0}
    for key in ("level_hpa", "fallback_threshold", "attribute", "extremum", "reduction", "model_level"):
        if tracker.get(key) is not None:
            values[key] = tracker[key]
    return values


def _toml_values(values: dict) -> str:
    return "\n".join(f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in values.items())


def _method_text(recipe: dict, *, lat: float, lon: float, grid_id: int) -> str:
    tracker = recipe["tracker"]
    if tracker is not None:
        return ("\n# Reviewed feature following: thresholds are detection settings, not severity.\n"
                "# Prepare the moving child's full statics corridor before forecasting.\n"
                "[relocation]\nenabled = true\n"
                f"grid_id = {grid_id}\n"
                "max_move_parent_cells = 4\nmin_overlap_fraction = 0.5\n"
                f"cadence_seconds = {tracker['cadence_seconds']}\n"
                "\n[relocation.follow]\n" + _toml_values(_follow_values(tracker)) + "\n"
                "\n# Track rows use the existing tracker at its consultation cadence.\n"
                "[relocation.track]\npath = \"storm-track.csv\"\n")
    scenario = recipe["scenario"]
    if scenario is not None:
        values = {"center_lat": lat, "center_lon": lon, **{
            key: scenario[key] for key in ("center_height_m", "radius_km", "depth_m", "amplitude_k", "rh_preserve")}}
        return ("\n# Reviewed warm-bubble initial-state sensitivity experiment.\n"
                "# This is an initiation nudge, not a balanced storm or hurricane insertion.\n"
                "[[perturbation.bubbles]]\n" + _toml_values(values) + "\n")
    return ""


def _native_args(args, recipe: dict, staged: Path, sizing, *, profile: dict | None = None):
    from gpuwm.domain_wizard import register_cli as register_domain_cli
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    register_domain_cli(sub)
    if profile is None:
        selected = _hardware_class(sizing, args.hardware_class)
        profile = effective_profile(recipe, selected)
    geometry = profile["geometry"]
    name = args.name or recipe["id"]
    # All paths that leave the stage are bound to the final bundle directory.
    data_dir = args.out.absolute().parent / "data" / args.out.stem
    command = ["domain", f"--source={args.source}", f"--cycle={args.cycle}",
               f"--hours={args.hours if args.hours is not None else int(geometry['forecast_hours'])}",
               f"--root-dx={geometry['root_dx_km']}",
               f"--nz={profile['nz']}",
               f"--history-interval={geometry['history_interval_s']}",
               f"--nest-history-interval={geometry['history_interval_s']}",
               f"--out={staged}", f"--name={name}", f"--data-dir={data_dir}"]
    if geometry["nest_ratios"]:
        command.append("--chain=" + ",".join(map(str, geometry["nest_ratios"])))
    if args.point is not None:
        command.append(f"--point={args.point}")
    else:
        command.append(f"--polygon={args.polygon.absolute()}")
    if args.physics_profile:
        command.append(f"--physics-profile={args.physics_profile}")
    if args.tiles:
        command.append(f"--tiles={args.tiles}")
    if args.nz is not None:
        command.append(f"--nz={args.nz}")
    for acknowledgement in args.ack:
        command.append(f"--ack={acknowledgement}")
    return parser.parse_args(command), data_dir


def _final_text(text: str, recipe: dict, *, lat: float, lon: float, data_dir: Path, profile: dict) -> str:
    # The wizard deliberately prints cwd-relative fetch commands. A movable
    # staging directory must not become a saved forcing directory in our bundle.
    fetch = re.search(r"(?ms)^\[fetch\]\s*\n(?P<body>.*?)(?=^\[|\Z)", text)
    if fetch:
        body = re.sub(r"(?m)^out\s*=.*$", "out = " + json.dumps(data_dir.as_posix()), fetch["body"])
        text = text[:fetch.start("body")] + body + text[fetch.end("body"):]
    case_data = re.search(r"(?ms)^\[case_data\]\s*\n(?P<body>.*?)(?=^\[|\Z)", text)
    if case_data:
        # This compiler exposes the wizard's default ERA5 acquisition path.
        # Its relative path was authored from the private stage, so bind it to
        # the final data directory while preserving every other case-data key.
        forcing = json.dumps([(data_dir / "era5-combined.grib").as_posix()])
        body = re.sub(r"(?m)^forcing\s*=.*$", "forcing = " + forcing, case_data["body"])
        text = text[:case_data.start("body")] + body + text[case_data.end("body"):]
    heading = (f"# Research configuration: {recipe['id']}\n"
               f"# Method: {recipe['method']}; validation remains {recipe['validation_status']}.\n")
    return heading + text.rstrip() + "\n" + _method_text(
        recipe, lat=lat, lon=lon, grid_id=len(profile["geometry"]["nest_ratios"]) + 1)


def _admission(text: str, *, recipe: dict, source: str, sizing, path: Path,
               retry_hint: str | None = None) -> tuple[object, dict]:
    from gpuwm import domain_wizard as wizard
    experiment = wizard.experiment_from_text(text, source=str(path))
    interval = wizard.source_forcing_interval_seconds(source)
    phases = wizard._sizing_phases(experiment, free_bytes=sizing.free_bytes,
                                   source=source, forcing_interval_seconds=interval,
                                   vram_gib=sizing.vram_gib, profile=sizing.device_profile)
    budget = wizard.sizing_budget_bytes(experiment, free_bytes=sizing.free_bytes,
                                       vram_gib=sizing.vram_gib,
                                       forcing_interval_seconds=interval,
                                       profile=sizing.device_profile)
    if retry_hint is None:
        fitting_class = _hardware_class(sizing, "auto")
        retry_hint = (f"Retry this same question ({recipe['id']}) with --hardware-class auto "
                      f"or --hardware-class {fitting_class}, keeping the location, source, cycle, "
                      "duration and GPU capacity unchanged. In the TUI, reopen the saved guide "
                      f"and set Research GPU profile to auto or {fitting_class}. "
                      "The retry still checks the full study area; if it also refuses, make more "
                      "memory available or choose a question with a smaller required area.")
    if phases.peak_envelope_bytes > budget:
        raise ValueError(f"The complete {recipe['method']} configuration exceeds the immutable "
                         f"memory budget: {phases.peak_envelope_bytes / 2**30:.2f} GiB needed, "
                         f"{budget / 2**30:.2f} GiB available. {retry_hint}")
    minimum = float(recipe["geometry"].get("minimum_root_span_km", 0))
    root = experiment.domains[0].run
    span_x, span_y = root.nx * root.dx / 1000, root.ny * root.dy / 1000
    if min(span_x, span_y) < minimum:
        ladder = " -> ".join(f"{domain.run.dx / 1000:g}" for domain in experiment.domains)
        raise ValueError(f"The fitted {span_x:g} x {span_y:g} km root ({ladder} km ladder) "
                         f"is smaller than this research question's "
                         f"{minimum:g} km minimum span. {retry_hint}")
    return experiment, {"status": "passed-cpu-estimate", "binding_phase": phases.binding_phase,
                        "peak_envelope_bytes": phases.peak_envelope_bytes,
                        "envelope_budget_bytes": budget,
                        "remaining_envelope_bytes": budget - phases.peak_envelope_bytes,
                        "forecast_started": False, "prepared_inputs_validated": False,
                        "science_validation": recipe["validation_status"]}


def _publish_bundle(stage: Path, destination: Path) -> list[Path]:
    """Create companions exclusively, then commit the TOML; roll back our files.

    Hard links are atomic create-only publication on the same filesystem. An
    existing companion, including a dangling symlink, is always a refusal.
    """
    files = sorted(stage.iterdir(), key=lambda path: (path.name == destination.name, path.name))
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("Research staging produced an unexpected non-file companion")
    targets = [destination.parent / path.name for path in files]
    for target in targets:
        if os.path.lexists(target):
            raise FileExistsError(f"Research creation preserves existing files: {target}")
    created = []
    try:
        for source, target in zip(files, targets):
            os.link(source, target)
            stat = target.stat()
            created.append((target, (stat.st_dev, stat.st_ino)))
    except BaseException:
        for target, identity in reversed(created):
            try:
                stat = target.lstat()
                if (stat.st_dev, stat.st_ino) == identity:
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    return targets


def _geometry_review(experiment) -> dict:
    """Describe fitted coverage using the runner's native placement check."""
    from dataclasses import replace
    from types import SimpleNamespace
    from gpuwm.core.nest_relocation import _prevalidate_placement

    spans = [{"grid_id": domain.grid_id, "parent_id": domain.parent_id,
              "span_x_km": domain.run.nx * domain.run.dx / 1000,
              "span_y_km": domain.run.ny * domain.run.dy / 1000}
             for domain in experiment.domains]
    review = {"domain_spans": spans, "coverage_notice": (
        "Fixed domains have finite coverage. A moving feature can leave a fine nest; "
        "review every domain against the expected event path and duration.")}
    relocation = getattr(experiment, "relocation", None)
    if relocation is None or getattr(relocation, "follow", None) is None:
        return review
    child = next(domain for domain in experiment.domains if domain.grid_id == relocation.grid_id)
    parent = next(domain for domain in experiment.domains if domain.grid_id == child.parent_id)
    parent_node = SimpleNamespace(cfg=parent)
    moving = {"grid_id": child.grid_id, "parent_id": parent.grid_id,
              "span_x_km": child.run.nx * child.run.dx / 1000,
              "span_y_km": child.run.ny * child.run.dy / 1000,
              "parent_span_x_km": parent.run.nx * parent.run.dx / 1000,
              "parent_span_y_km": parent.run.ny * parent.run.dy / 1000,
              "centroid_radius_km": relocation.follow.radius_km,
              "centroid_notice": "Configurable centroid radius; suitability for a particular cell or vortex is unvalidated.",
              "cadence_seconds": relocation.cadence_seconds,
              "track_path": relocation.track.path if relocation.track is not None else None,
              "prepared_statics": "not yet prepared"}
    review["moving_nest"] = moving
    review["coverage_notice"] = (
        "Following is bounded by the immediate parent. The native runner clamps or holds at its "
        "admissible band; these distances do not guarantee following for the whole forecast. "
        "Prepare the moving child's complete statics corridor before running.")
    try:
        _prevalidate_placement(child, parent_node)
    except ValueError as error:
        moving["placement_status"] = "not_admissible"
        moving["placement_error"] = str(error)
        return review
    starts, travel = {}, {}
    for axis, key, extent, spacing in (
            ("x", "i_parent_start", parent.run.nx, parent.run.dx),
            ("y", "j_parent_start", parent.run.ny, parent.run.dy)):
        current = int(getattr(child, key))
        def valid(position):
            try:
                _prevalidate_placement(replace(child, **{key: position}), parent_node)
            except ValueError:
                return False
            return True
        low, high = 1, current
        while low < high:
            middle = (low + high) // 2
            if valid(middle):
                high = middle
            else:
                low = middle + 1
        minimum = low
        low, high = current, int(extent)
        while low < high:
            middle = (low + high + 1) // 2
            if valid(middle):
                low = middle
            else:
                high = middle - 1
        starts[key] = [minimum, low]
        travel[f"{axis}_km"] = [(minimum - current) * spacing / 1000,
                               (low - current) * spacing / 1000]
    moving.update(placement_status="native_stencil_checked",
                  admissible_starts_parent_cells=starts,
                  travel_from_initial_grid_axes=travel)
    return review


def create_workspace(args) -> dict:
    from gpuwm import domain_wizard as wizard
    catalog = catalog_document()
    recipe = _recipe(args.configuration_id, document=catalog)
    hardware = {"profiles": catalog["hardware_profiles"], "policy": catalog["hardware_policy"]}
    if recipe["requires_existing_state"] and recipe["method"] != "controlled_scenario":
        if recipe["method"] == "archived_downscale":
            raise ValueError("This archived-downscale recipe requires actual parent history, restart "
                             "physics evidence and child surface state. Open the native Downscale "
                             "guide, or use gpuwm downscale; research create cannot invent a parent.")
        raise ValueError("This research recipe requires a real existing scenario state. Open the "
                         "existing configuration for review; research create cannot substitute "
                         "an ordinary GFS analysis for the required state.")
    source = wizard.resolve_source(args.source)
    if recipe["method"] == "controlled_scenario" and source != "gfs":
        raise ValueError(f"Controlled-scenario creation for --source {source} is not admitted "
                         "by this research workflow. Use --source gfs for the supported fresh "
                         "prepared-tree warm-bubble path; other sources need their own explicit "
                         "preparation/initialization route acceptance. No perturbation was dropped.")
    destination = args.out.absolute()
    if destination.suffix.lower() != ".toml":
        raise ValueError("Research output must be a new .toml path")
    if os.path.lexists(destination):
        raise FileExistsError(f"Research creation preserves existing files: {destination}")
    for companion in (destination.with_suffix(".namelist.wps"),
                      destination.with_name(destination.name + ".arwen-plots.json"),
                      destination.with_name(destination.name + ".arwen-research.json")):
        if os.path.lexists(companion):
            raise FileExistsError(f"Research creation preserves existing companion: {companion}")
    sizing = wizard.resolve_sizing_budget(None, args.vram_gib)
    budget_document = _budget_document(sizing, args.hardware_class, hardware=hardware)
    profile = effective_profile(recipe, budget_document["selected_class"], hardware=hardware)
    if args.point is not None:
        lat, lon = wizard._parse_point(args.point)
    else:
        footprint = wizard.load_polygon_footprint(args.polygon)
        lat, lon = footprint.center_lat, footprint.center_lon
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    with tempfile.TemporaryDirectory(prefix=".arwen-research-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        staged = stage / destination.name
        native, data_dir = _native_args(args, recipe, staged, sizing, profile=profile)
        with redirect_stdout(log), redirect_stderr(log):
            result = wizard.domain_main(native, sizing_budget=sizing)
        if result:
            raise ValueError(f"Native domain creation failed (exit {result}):\n{log.getvalue()}")
        native_text = staged.read_text(encoding="utf-8")
        if native.source == "era5":
            # Each bundle owns its Vtable: separate studies can share a folder
            # without colliding with or changing somebody else's common Vtable.
            vtable = stage / wizard._PACKAGED_VTABLE.name
            if vtable.exists():
                unique_vtable = stage / (destination.stem + "." + vtable.name)
                vtable.rename(unique_vtable)
                native_text = native_text.replace(
                    'vtable = ' + json.dumps(vtable.name),
                    'vtable = ' + json.dumps(unique_vtable.name))
        native_text = native_text.replace(str(stage), str(destination.parent)).replace(
            stage.as_posix(), destination.parent.as_posix())
        text = _final_text(native_text, recipe,
                           lat=lat, lon=lon, data_dir=data_dir, profile=profile)
        experiment, admission = _admission(text, recipe=recipe, source=native.source,
                                          sizing=sizing, path=destination)
        staged.write_text(text, encoding="utf-8", newline="\n")
        domains = [{"grid_id": domain.grid_id, "parent_id": domain.parent_id,
                    "nx": domain.run.nx, "ny": domain.run.ny, "nz": domain.run.nz,
                    "dx_km": domain.run.dx / 1000,
                    "span_x_km": domain.run.nx * domain.run.dx / 1000,
                    "span_y_km": domain.run.ny * domain.run.dy / 1000,
                    "history_interval_s": domain.history_interval_s,
                    "time_step_s": domain.run.dt} for domain in experiment.domains]
        receipt = {"schema": "arwen.research.workspace.v1",
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "configuration": str(destination), "recipe": recipe,
                   "catalog_sha256": catalog["catalog_sha256"],
                   "diagnostic_capabilities_sha256": catalog["diagnostic_capabilities"]["sha256"],
                   "config_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                   "hardware": budget_document, "source": native.source,
                   "effective_profile": profile,
                   "hardware_policy_sha256": catalog["hardware_policy_sha256"],
                   "start_time": experiment.start_time.isoformat(),
                   "run_seconds": experiment.run_seconds, "domains": domains,
                   "requested_overrides": {"hours": args.hours, "nz": args.nz,
                                           "physics_profile": args.physics_profile,
                                           "tiles": args.tiles},
                   "admission": admission,
                   "geometry_review": _geometry_review(experiment),
                   "required_next_steps": ["Review actual geometry and scientific settings",
                                           "Acquire and prepare forcing and statics",
                                           "Check actual free memory again before launch"],
                   "native_creation_log": log.getvalue().replace(
                       str(stage), str(destination.parent)).replace(
                       stage.as_posix(), destination.parent.as_posix())}
        if recipe["tracker"]:
            receipt["required_next_steps"].append("Prepare the moving child's complete --statics-corridor")
        if recipe["scenario"]:
            receipt["required_next_steps"].append("Validate warm-bubble coverage during native preparation")
        receipt["files"] = [str(destination.parent / name) for name in sorted(
            [path.name for path in stage.iterdir()]
            + [destination.name + ".arwen-plots.json", destination.name + ".arwen-research.json"])]
        plots = {"schema": "gpuwm-tui-plots-v1", "label": recipe["title"],
                 "products": ",".join(recipe["diagnostics"])}
        (stage / (destination.name + ".arwen-plots.json")).write_text(
            json.dumps(plots, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (stage / (destination.name + ".arwen-research.json")).write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        _publish_bundle(stage, destination)
    return receipt


def _research_main(args) -> int:
    if args.research_command == "catalog":
        document = catalog_document()
    elif args.research_command == "hardware":
        document = hardware_document(hardware_class=args.hardware_class, vram_gib=args.vram_gib)
    elif args.research_command == "attributes":
        document = attributes_document()
    else:
        document = create_workspace(args)
    if args.json:
        print(json.dumps(document, ensure_ascii=False, default=str))
    elif args.research_command == "create":
        hardware = document["hardware"]
        print(f"Created {document['configuration']}")
        print(f"Research: {document['recipe']['title']} ({document['recipe']['method']})")
        for note in document["recipe"].get("further_analysis", []):
            print(f"Additional analysis required: {note}")
        print(f"Hardware class {hardware['selected_class']}; {hardware['basis']}; "
              f"{hardware['free_bytes'] / 2**30:.2f} GiB available")
        print("Domains: " + "; ".join(f"d{domain['grid_id']:02d} {domain['nx']}x{domain['ny']}x{domain['nz']} "
                                      f"at {domain['dx_km']:g} km" for domain in document["domains"]))
        geometry = document["geometry_review"]
        print("Coverage: " + "; ".join(
            f"d{domain['grid_id']:02d} {domain['span_x_km']:g} x {domain['span_y_km']:g} km"
            for domain in geometry["domain_spans"]))
        if moving := geometry.get("moving_nest"):
            print(f"Following d{moving['grid_id']:02d} ({moving['span_x_km']:g} x {moving['span_y_km']:g} km) "
                  f"inside d{moving['parent_id']:02d} ({moving['parent_span_x_km']:g} x {moving['parent_span_y_km']:g} km)")
            if travel := moving.get("travel_from_initial_grid_axes"):
                print(f"Native placement band from initial position (grid axes): "
                      f"x {travel['x_km'][0]:g} to {travel['x_km'][1]:g} km; "
                      f"y {travel['y_km'][0]:g} to {travel['y_km'][1]:g} km")
            else:
                print(f"Movement band unavailable: {moving.get('placement_error', 'not established')}")
            print(f"Centroid radius: {moving['centroid_radius_km']:g} km. {moving['centroid_notice']}")
            print(f"Forecast track output: {moving['track_path']} under the run output directory, at tracker consultations.")
        print(geometry["coverage_notice"])
        print("Native CPU geometry/memory admission passed. Review, prepare and check before launching.")
        for step in document["required_next_steps"]:
            print(f"  {step}")
    else:
        print(json.dumps(document, indent=2, ensure_ascii=False, default=str))
    return 0


def research_main(args) -> int:
    try:
        return _research_main(args)
    except OSError as error:
        target = error.filename or getattr(args, "out", CATALOG_PATH)
        action = ("Choose a new output filename; the existing file and its companion files are preserved."
                  if isinstance(error, FileExistsError) else
                  "Check that the named path is available and writable, then retry.")
        raise ValueError(f"Research {args.research_command} could not use {target}: {error}. {action}") from error


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("research", help="browse and create reviewed research workspaces",
                                  description="Choose a research recipe, review its hardware profile, and create a new native configuration. Creation does not prepare or run a forecast.")
    commands = parser.add_subparsers(dest="research_command", required=True)
    catalog = commands.add_parser("catalog", help="show the packaged research catalog")
    catalog.add_argument("--json", action="store_true", help="emit the complete native catalog and hardware profiles as compact JSON")
    catalog.set_defaults(func=research_main)
    attributes = commands.add_parser("attributes", help="show the native attribute-following registry and scientific units")
    attributes.add_argument("--json", action="store_true", help="emit supported attributes, units, extrema, reductions and model-level semantics as compact JSON")
    attributes.set_defaults(func=research_main)
    hardware = commands.add_parser("hardware", help="measure hardware or estimate an explicit capacity")
    hardware.add_argument("--json", action="store_true", help="emit capacity, free-memory budget and selected resource profile as compact JSON")
    hardware.set_defaults(func=research_main)
    create = commands.add_parser("create", help="create a new native configuration and review receipt")
    create.add_argument("configuration_id", help="exact configuration id from research catalog; existing-state recipes must continue from their supplied archive or scenario")
    target = create.add_mutually_exclusive_group(required=True)
    target.add_argument("--point", metavar="LAT,LON", help="centre latitude and longitude; native sizing fits coverage to the actual budget and enforces the recipe's minimum span")
    target.add_argument("--polygon", type=Path, metavar="GEOJSON", help="GeoJSON study footprint; retain its full required coverage")
    create.add_argument("--source", default="gfs", help="native input source id (default gfs); source, cycle and research method must be compatible")
    create.add_argument("--cycle", required=True, help="input cycle in UTC, YYYY-MM-DDTHH; latest uses native source discovery")
    create.add_argument("--hours", type=int, help="explicit whole-hour study duration; omitted keeps the recipe's duration")
    create.add_argument("--out", type=Path, required=True, help="new experiment TOML path; existing configuration or companion files are never replaced")
    create.add_argument("--name", help="descriptive name for this study; omitted keeps the recipe title")
    create.add_argument("--physics-profile", help="native physics suite override; omitted keeps the admitted source default")
    create.add_argument("--nz", type=int, help="explicit vertical-level override")
    create.add_argument("--tiles", choices=("off", "auto", "on"), help="explicit native tile-streaming mode; host-memory and transfer costs remain subject to admission")
    create.add_argument("--ack", action="append", default=[], help="native source acknowledgement code; repeat for each required acknowledgement")
    create.add_argument("--json", action="store_true", help="emit the created workspace receipt, geometry, resource budget and required next steps as compact JSON")
    create.set_defaults(func=research_main)
    for command in (hardware, create):
        command.add_argument("--hardware-class", choices=("auto", *HARDWARE_CLASSES), default="auto",
                             help="resource profile; does not declare available memory")
        command.add_argument("--vram-gib", type=float,
                             help="explicit capacity estimate; omit to measure actual total/free memory")
    from gpuwm.explain import add_explain_flag
    for command in (catalog, attributes, hardware, create):
        add_explain_flag(command)
