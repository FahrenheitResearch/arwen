"""Saved forecast setups: copy one configuration, start another from it.

A saved setup is an ordinary configuration copied into a library folder
under a name a person typed, plus the companions the staged route reads.
Starting from one re-times metadata -- the cycle, the lead, the duration,
the name and the acquisition folder -- and keeps every scientific setting,
every domain and the projection exactly as they were saved.

Nothing here runs a forecast, downloads anything, or chooses a setting.
Both actions publish create-only and refuse with one sentence naming the
breakage and the way out.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import tomllib

SETUP_SCHEMA = "arwen.saved-setup.v1"
RESULT_SCHEMA = "arwen.companion-setup-result.v1"

SETUP_STEM = "setup"
SETUP_TOML = SETUP_STEM + ".toml"
SETUP_METADATA = SETUP_STEM + ".json"
SETUP_WPS = SETUP_STEM + ".namelist.wps"

SETUP_HEADER = ("# Saved setup; start a forecast from it in Create forecast. "
                "No forecast has run.\n")
START_HEADER = ("# Started from a saved setup; review before Run. "
                "No forecast has run.\n")

#: Creation receipts bound to one computer, one request or one edit. They
#: are dropped rather than copied: sizing is measured again at review and
#: no route reads an edit receipt.
DROPPED_COMPANION_SUFFIXES = (
    ".fit.json", ".domains.json", ".forcing.json", ".schedule.json",
    ".tiles.json", ".setup-start.json",
)

#: The only fields saving a setup may change, all of them paths whose
#: meaning would otherwise move with the file.
SAVE_ALLOWED_CHANGES = frozenset({
    "fetch.out", "case_data.vtable", "case_data.wps_namelist",
    "static.highres.cache_root",
})

#: The only fields starting from a setup may change: the timed metadata a
#: person retypes, and nothing else.
START_TIMED_CHANGES = frozenset({
    "experiment.name", "experiment.start_time", "experiment.run_seconds",
    "fetch.cycle", "fetch.hours", "fetch.forecast_start_hour", "fetch.out",
})

#: Declared input paths, resolved absolute when a configuration is
#: published in another folder, exactly as every other companion door
#: resolves them.
PATH_CHANGE_PREFIXES = (
    "case_data.vtable", "case_data.wps_namelist", "case_data.geog_root",
    "case_data.forcing", "case_data.water_temperature_overlay",
    "case_data.source_orography", "static.highres.cache_root",
)

NAME_REFUSAL = ("Type a name for this setup: 1 to 80 characters, "
                "no slashes, colons or quotes.")
FORECAST_NAME_REFUSAL = "Type a forecast name."
HOURS_REFUSAL = "Choose a positive forecast duration in hours."
LEAD_REFUSAL = "Choose a start after the cycle of 0 hours or more."

_FORBIDDEN_NAME_CHARACTERS = frozenset('/\\:*?"<>|')


# ---------------------------------------------------------------------------
# Names and slugs
# ---------------------------------------------------------------------------

def setup_name(value) -> str:
    """The trimmed name a person typed, or the one sentence that says why not."""

    if not isinstance(value, str):
        raise ValueError(NAME_REFUSAL)
    name = value.strip()
    if not 1 <= len(name) <= 80:
        raise ValueError(NAME_REFUSAL)
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ValueError(NAME_REFUSAL)
    if _FORBIDDEN_NAME_CHARACTERS & set(name):
        raise ValueError(NAME_REFUSAL)
    if name in (".", ".."):
        raise ValueError(NAME_REFUSAL)
    return name


def slug_for(name: str) -> str:
    """The folder name a setup lives under: lowercased, dashed, never empty."""

    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or SETUP_STEM


# ---------------------------------------------------------------------------
# The summary line
# ---------------------------------------------------------------------------

def _selector_matches(selectors, run) -> bool:
    for key, value in selectors.items():
        if key not in run:
            return False
        actual = run[key]
        if isinstance(actual, bool) != isinstance(value, bool):
            return False
        if actual != value:
            return False
    return True


def summary_line(exp, raw, outlines) -> str:
    """One line naming what this setup is, derived, never transcribed.

    Geometry comes from the native outlines, scheme names from the
    installed physics registry through the same option match the registry
    itself makes, and the forcing name from the source row's own display
    title. A component whose selectors no implemented option matches
    prints those selectors verbatim rather than an invented name.
    """
    from gpuwm.companion_domains import physics_components

    parts = [f"d{outline['grid_id']:02d} {outline['dx_m'] / 1000:g} km "
             f"{outline['nx']} x {outline['ny']}" for outline in outlines]
    run = vars(exp.root.run)
    for component in physics_components():
        keys = sorted({key for option in component["options"]
                       for key in option.get("selectors", {})})
        present = [key for key in keys if key in run]
        if not present:
            continue
        label = None
        for option in component["options"]:
            selectors = option.get("selectors", {})
            if selectors and _selector_matches(selectors, run):
                label = option.get("label") or option["id"]
                break
        parts.append(label if label is not None else
                     ", ".join(f"{key}={run[key]}" for key in present))
    source = raw.get("fetch", {}).get("source")
    if source:
        parts.append(_source_title(source))
    parts.append(f"output every {exp.root.history_interval_s:g} s")
    parts.append(f"tiles {exp.tiles.mode}")
    return " · ".join(parts)


def _source_title(source: str) -> str:
    from gpuwm.source_adapters import get_source_adapter
    try:
        return get_source_adapter(source).display_title
    except ValueError:
        return str(source)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_authority(path, refusal):
    from gpuwm.config_authority import read_config_authority
    try:
        return read_config_authority(path)
    except (OSError, ValueError) as error:
        raise ValueError(refusal(str(error))) from error


def _allowed(field: str, allowed) -> bool:
    return field in allowed or any(
        field == prefix or field.startswith(prefix + "[")
        or field.startswith(prefix + ".") for prefix in PATH_CHANGE_PREFIXES)


def _guard(original, candidate, allowed, refusal):
    from gpuwm.starter_template import changes
    for field, _before, _after in changes(original, candidate):
        if not _allowed(field, allowed):
            raise ValueError(refusal.format(field=field))


def _resolve_declared_paths(raw, *, base_dir, source):
    """Absolute declared inputs, so a configuration published elsewhere reads them."""
    if "case_data" in raw:
        from gpuwm.case_data import resolved_case_data_paths
        raw["case_data"] = resolved_case_data_paths(
            raw["case_data"], base_dir=base_dir, source=str(source))
    if "static" in raw:
        from gpuwm.static.highres_production import parse_static_table
        config = parse_static_table(raw["static"], source=str(source),
                                    base_dir=base_dir)
        if config is not None:
            raw["static"]["highres"]["cache_root"] = str(config.cache_root.resolve())
    return raw


def _cadence_hours(raw_fetch, source):
    """The saved cadence, or the source row's own declared native interval."""
    from gpuwm.source_adapters import source_forcing_interval_seconds
    cadence = raw_fetch.get("cadence")
    if cadence is None:
        cadence = source_forcing_interval_seconds(source) / 3600
    refusal = (f"This saved setup declares cadence = {raw_fetch.get('cadence')!r} "
               "in [fetch], which is not a positive number of hours, so its "
               "forcing window cannot be timed. Delete it from Saved setups and "
               "save the forecast again.")
    if isinstance(cadence, bool) or not isinstance(cadence, (int, float)):
        raise ValueError(refusal)
    cadence = float(cadence)
    if not math.isfinite(cadence) or cadence <= 0:
        raise ValueError(refusal)
    return int(cadence) if cadence == int(cadence) else cadence


def _carry_case_data_files(retimed, *, setup_dir, out, exists_refusal):
    """Declared inputs saved inside the setup folder, placed beside the new file.

    A started forecast must keep running after its setup is deleted from
    the library, so a Vtable the setup folder holds is copied beside the
    new configuration, the way the domain door places one beside a forecast
    it creates. An identical file already there is reused; a different one
    is not touched and the copy takes the new configuration's stem.
    Returns the (path, text) pairs to publish.
    """
    case_data = retimed.get("case_data")
    if not isinstance(case_data, dict):
        return []
    carried = []
    for key in ("vtable",):
        value = case_data.get(key)
        if not isinstance(value, str) or not value:
            continue
        declared = Path(value)
        if not declared.is_absolute():
            declared = setup_dir / declared
        declared = declared.resolve()
        if not declared.is_file() or setup_dir.resolve() not in declared.parents:
            continue
        content = declared.read_text(encoding="utf-8-sig")
        target = out.parent / declared.name
        if target.is_file():
            if target.read_text(encoding="utf-8-sig") != content:
                target = out.with_name(out.stem + "." + declared.name)
                if os.path.lexists(target):
                    raise ValueError(exists_refusal)
                carried.append((target, content))
        elif os.path.lexists(target):
            raise ValueError(exists_refusal)
        else:
            carried.append((target, content))
        case_data[key] = str(target)
    return carried


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

def _companion_files(source_path: Path):
    """Every file `gpuwm domain` wrote under this configuration's stem."""
    suffix = source_path.suffix
    stem = source_path.name[:-len(suffix)] if suffix else source_path.name
    found = {}
    for entry in sorted(source_path.parent.glob(stem + ".*")):
        if not entry.is_file() or entry == source_path:
            continue
        rest = entry.name[len(stem):]
        if rest == ".toml" or rest in DROPPED_COMPANION_SUFFIXES:
            continue
        found[rest] = entry
    return found


def _library_refusal(library, error):
    return (f"The setups library {library} cannot be written: "
            f"{getattr(error, 'strerror', None) or error}. "
            "Choose a data folder you can write to in Settings.")


def save_setup(*, config_path, library, name):
    """Copy one configuration and its companions into the setups library."""
    from gpuwm import __version__
    from gpuwm.case_data import resolved_case_data_paths
    from gpuwm.companion_domains import _build, _json
    import gpuwm.companion_domains as domains
    from gpuwm.starter_template import changes, _publish_new_files
    from gpuwm.toml_document import emit_experiment_toml

    name = setup_name(name)
    slug = slug_for(name)
    library = Path(library).expanduser().resolve()
    authority = _read_authority(
        config_path,
        lambda sentence: sentence + " Open the forecast again, then save it.")
    original = tomllib.loads(authority.payload.decode("utf-8-sig"))
    exp = _build(original, authority.source)

    resolved_case_data = None
    if "case_data" in original:
        resolved_case_data = resolved_case_data_paths(
            original["case_data"], base_dir=authority.base_dir,
            source=str(authority.source))
    companions = _companion_files(authority.source)
    wps_source = companions.pop(".namelist.wps", None)
    if wps_source is None and resolved_case_data is not None:
        declared = resolved_case_data.get("wps_namelist")
        if declared and Path(declared).is_file():
            wps_source = Path(declared)
    if wps_source is None or not wps_source.is_file():
        expected = authority.source.with_suffix(".namelist.wps").name
        raise ValueError(
            f"This forecast has no {expected} beside it, so the staged route "
            "could not run a forecast started from it. Create the forecast "
            "again, then save it.")

    raw = copy.deepcopy(original)
    extra_files = {}
    if raw.get("fetch", {}).get("out"):
        # [fetch].out is relative to the folder the wizard ran in, not to
        # the TOML, so a verbatim copy would name a different place once
        # the file sits in the library. Its meaning at save time is what
        # is preserved.
        raw["fetch"]["out"] = str(Path(raw["fetch"]["out"]).expanduser().resolve())
    if resolved_case_data is not None:
        if raw["case_data"].get("wps_namelist"):
            raw["case_data"]["wps_namelist"] = SETUP_WPS
        vtable = resolved_case_data.get("vtable")
        if vtable and Path(vtable).is_file():
            extra_files[Path(vtable).name] = Path(vtable)
            raw["case_data"]["vtable"] = Path(vtable).name
    if "static" in raw:
        from gpuwm.static.highres_production import parse_static_table
        config = parse_static_table(raw["static"], source=str(authority.source),
                                    base_dir=authority.base_dir)
        if config is not None:
            raw["static"]["highres"]["cache_root"] = str(config.cache_root.resolve())
    _guard(original, raw, SAVE_ALLOWED_CHANGES,
           "Saving changed a setting it must keep: {field}. Nothing was saved.")

    text = SETUP_HEADER + emit_experiment_toml(raw)
    setup_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    outlines = domains.native_domain_outlines(exp)
    try:
        library.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(_library_refusal(library, error)) from error
    folder = library / slug
    document = {
        "schema": SETUP_SCHEMA, "name": name, "slug": slug,
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine_version": __version__,
        "summary": summary_line(exp, raw, outlines),
        "setup_path": str(folder / SETUP_TOML), "setup_sha256": setup_sha,
        "files": sorted({SETUP_TOML, SETUP_METADATA, SETUP_WPS,
                         *(SETUP_STEM + rest for rest in companions),
                         *extra_files}),
        "source_config_path": str(authority.source),
        "source_config_sha256": authority.sha256,
        "fetch_source": raw.get("fetch", {}).get("source"),
        "forecast_start_hour": int(raw.get("fetch", {}).get("forecast_start_hour", 0)),
        "run_seconds": float(exp.run_seconds),
        "domains": [{"grid_id": outline["grid_id"], "parent_id": outline["parent_id"],
                     "dx_m": outline["dx_m"], "nx": outline["nx"],
                     "ny": outline["ny"], "nz": outline["nz"]}
                    for outline in outlines],
    }
    result = {
        "schema": RESULT_SCHEMA, "action": "save", "created": True,
        "forecast_started": False, "setup": document,
        "source_path": str(authority.source), "source_sha256": authority.sha256,
        "changes": [{"field": field, "before": before, "after": after}
                    for field, before, after in changes(original, raw)],
    }
    try:
        folder.mkdir(parents=False)
    except FileExistsError:
        raise ValueError(_collision_refusal(folder)) from None
    except OSError as error:
        raise ValueError(_library_refusal(library, error)) from error
    files = [(folder / SETUP_WPS, wps_source.read_text(encoding="utf-8-sig"))]
    files += [(folder / (SETUP_STEM + rest), path.read_text(encoding="utf-8-sig"))
              for rest, path in sorted(companions.items())]
    files += [(folder / filename, path.read_text(encoding="utf-8-sig"))
              for filename, path in sorted(extra_files.items())]
    files.append((folder / SETUP_METADATA, _json(document)))
    files.append((folder / SETUP_TOML, text))
    try:
        _publish_new_files(files)
        if hashlib.sha256(authority.source.read_bytes()).hexdigest() != authority.sha256:
            raise ValueError(
                "The forecast changed while it was being saved. Open the "
                "forecast again, then save it.")
    except BaseException:
        # Remove only the folder this save created; nothing else is touched.
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return result


def _collision_refusal(folder: Path) -> str:
    existing = folder.name
    try:
        document = json.loads((folder / SETUP_METADATA).read_text(encoding="utf-8-sig"))
        if isinstance(document, dict) and isinstance(document.get("name"), str):
            existing = document["name"]
    except (OSError, ValueError):
        pass
    return (f'A saved setup named "{existing}" already exists in this library. '
            "Choose another name, or delete that setup first.")


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

def read_saved_setup(setup_path):
    """One saved setup's configuration authority and its verified metadata."""
    setup_path = Path(setup_path).expanduser().resolve()
    authority = _read_authority(
        setup_path,
        lambda sentence: "The saved setup cannot be read: " + sentence +
        " Delete it from Saved setups and save the forecast again.")
    metadata_path = setup_path.with_name(SETUP_METADATA)
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"The saved setup cannot be read: {metadata_path} is not readable "
            f"saved-setup metadata ({error}). Delete it from Saved setups and "
            "save the forecast again.") from error
    if not isinstance(document, dict) or document.get("schema") != SETUP_SCHEMA:
        raise ValueError(
            f"The saved setup cannot be read: {metadata_path} does not declare "
            f"schema {SETUP_SCHEMA}. Delete it from Saved setups and save the "
            "forecast again.")
    if document.get("setup_sha256") != authority.sha256:
        name = document.get("name") or setup_path.parent.name
        raise ValueError(
            f'The saved setup "{name}" was changed after it was saved: '
            "setup.toml no longer matches setup.json. Delete it from Saved "
            "setups and save the forecast again.")
    return authority, document


def start_setup(*, setup_path, cycle, hours, forecast_start_hour, name, out):
    """Write one new configuration at a new cycle, keeping every saved setting."""
    from gpuwm import fetch, fetch_routes
    from gpuwm.companion_domains import _build, _json, _wps_text, domain_geojson
    import gpuwm.companion_domains as domains
    from gpuwm.experiment import experiment_config_document, validate_boundary_timing
    from gpuwm.starter_template import changes, _publish_new_files
    from gpuwm.toml_document import emit_experiment_toml

    if not isinstance(name, str) or not name.strip():
        raise ValueError(FORECAST_NAME_REFUSAL)
    name = name.strip()
    if (isinstance(hours, bool) or not isinstance(hours, (int, float))
            or not math.isfinite(hours) or hours <= 0):
        raise ValueError(HOURS_REFUSAL)
    hours = float(hours)
    lead = forecast_start_hour
    if isinstance(lead, bool) or not isinstance(lead, int) or lead < 0:
        raise ValueError(LEAD_REFUSAL)
    out = Path(out).expanduser().resolve()
    wps_output = out.with_suffix(".namelist.wps")
    receipt_output = out.with_suffix(".setup-start.json")
    exists_refusal = (f"A configuration already exists at {out}. "
                      "Choose a new name, or use Open configuration.")

    authority, document = read_saved_setup(setup_path)
    if out == authority.source or any(os.path.lexists(path) for path in
                                      (out, wps_output, receipt_output)):
        raise ValueError(exists_refusal)
    original = tomllib.loads(authority.payload.decode("utf-8-sig"))
    source = original.get("fetch", {}).get("source")
    if not source:
        raise ValueError(
            "This saved setup declares no forcing source, so there is no cycle "
            "to start it from. Save it from a forecast created in Create forecast.")

    cadence = _cadence_hours(original["fetch"], source)
    # The wizard's own window rule: a whole number of cadence steps, and
    # never shorter than one step.
    fetch_hours = max(cadence, math.ceil(hours / cadence) * cadence)
    if fetch_hours == int(fetch_hours):
        fetch_hours = int(fetch_hours)
    if isinstance(cycle, datetime):
        resolved_cycle = cycle
    elif str(cycle).strip().lower() == "latest":
        resolved_cycle = fetch.resolve_latest_cycle(
            source, last_hour=int(lead + fetch_hours))
    else:
        resolved_cycle = fetch.parse_cycle(str(cycle).strip(), source)
    start_time = resolved_cycle + timedelta(hours=lead)

    retimed = copy.deepcopy(original)
    retimed["experiment"]["name"] = name
    retimed["experiment"]["start_time"] = start_time
    retimed["experiment"]["run_seconds"] = hours * 3600
    retimed["fetch"]["cycle"] = resolved_cycle.strftime("%Y-%m-%dT%H")
    retimed["fetch"]["hours"] = fetch_hours
    if lead:
        retimed["fetch"]["forecast_start_hour"] = lead
    else:
        retimed["fetch"].pop("forecast_start_hour", None)
    # The wizard's own data layout: a folder named for this forecast,
    # beside the configuration it belongs to.
    retimed["fetch"]["out"] = str(out.parent / "data" / name)
    if retimed.get("case_data", {}).get("wps_namelist"):
        retimed["case_data"]["wps_namelist"] = str(wps_output)
    carried = _carry_case_data_files(retimed, setup_dir=authority.source.parent,
                                     out=out, exists_refusal=exists_refusal)
    _guard(original, retimed, START_TIMED_CHANGES,
           "Starting changed a setting it must keep: {field}. Nothing was written.")
    timed = [{"field": field, "before": before, "after": after}
             for field, before, after in changes(original, retimed)]

    raw = _resolve_declared_paths(copy.deepcopy(retimed),
                                  base_dir=authority.base_dir,
                                  source=authority.source)
    fetch.validate_fetch_hints(raw["fetch"], source=str(out))
    try:
        route = fetch_routes.route_for(source)
    except ValueError:
        # A source whose transport lives in gpuwm.fetch, or one with no
        # download route at all: validate_fetch_hints above is its checker.
        route = None
    if route is not None:
        fetch_routes.resolve_cycle(route, resolved_cycle)
        fetch_routes.resolve_leads(route, resolved_cycle, int(fetch_hours),
                                   cadence=int(cadence), start_hour=lead)
    exp = _build(raw, out)
    boundary_seconds = cadence * 3600
    if int(boundary_seconds) != boundary_seconds:
        raise ValueError(
            "A saved setup needs a whole-second forcing cadence; this one "
            f"declares {cadence} h. Set cadence in [fetch] to a whole number "
            "of seconds (a multiple of 1/3600 h), or delete it so the setup "
            "takes its source's own interval.")
    validate_boundary_timing(exp, int(boundary_seconds), source="saved setup start")
    outlines = domains.native_domain_outlines(exp)
    timing = {"source": source, "cycle": resolved_cycle.strftime("%Y-%m-%dT%H"),
              "forecast_start_hour": lead, "hours": hours,
              "fetch_hours": fetch_hours,
              "start_time": start_time.isoformat(), "cadence_hours": cadence}

    saved_wps = authority.source.with_name(SETUP_WPS)
    if source == "era5" and "case_data" in raw:
        return _start_through_forcing_editor(
            raw=raw, out=out, authority=authority, document=document,
            timed=timed, timing=timing, hours=hours, carried=carried,
            saved_wps=saved_wps if saved_wps.is_file() else None)

    text = START_HEADER + emit_experiment_toml(raw)
    published = _build(tomllib.loads(text), out)
    config_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    wps = _wps_text(exp, saved_wps if saved_wps.is_file() else None, wps_output,
                    raw, len(exp.domains),
                    original_domain_ids=[domain.grid_id for domain in exp.domains])
    companions = [(wps_output, wps), *carried]
    from gpuwm.runplan import prepared_chain_for_source
    if "case_data" not in raw and prepared_chain_for_source(source) == "prepared:hrrr":
        # The native route reads four dated companions. They are rendered
        # again from the re-timed experiment rather than copied, because a
        # copied one would carry the saved forecast's dates.
        from gpuwm.hrrr_route_inputs import write_hrrr_route_inputs
        with tempfile.TemporaryDirectory(prefix="arwen-setup-start-") as directory:
            staged = Path(directory) / out.name
            written = write_hrrr_route_inputs(
                staged, published, wps_text=wps,
                writer=lambda path, content: path.write_text(content, encoding="utf-8"))
            companions = [(out.parent / path.name, path.read_text(encoding="utf-8"))
                          for path in written] + carried
    for path, _content in companions:
        if os.path.lexists(path):
            raise ValueError(exists_refusal)
    configuration = {
        "schema": "arwen.companion-configuration.v1", "config_path": str(out),
        "config_sha256": config_sha, "geometry_backend": "rust-static-fields",
        "domains": outlines, "domains_geojson": domain_geojson(outlines),
        "experiment": experiment_config_document(exp),
        "tiles": exp.tiles.to_mapping(), "fetch": raw.get("fetch", {}),
        "case_data": {key: value for key, value in raw.get("case_data", {}).items()
                      if key in ("forcing_interval_s", "start_time", "end_time",
                                 "wps_namelist")}}
    result = {
        "schema": RESULT_SCHEMA, "action": "start", "created": True,
        "forecast_started": False, "acquisition_started": False,
        "config_path": str(out), "config_sha256": config_sha,
        "wps_path": str(wps_output), "receipt_path": str(receipt_output),
        "setup_path": str(authority.source), "setup_sha256": authority.sha256,
        "setup_name": document.get("name"), "timing": timing,
        "changes": timed, "configuration": configuration,
        "validation": {"configuration_parser": "passed", "native_geometry": "passed",
                       "boundary_timing": "passed", "fetch_hints": "passed",
                       "forecast_or_memory_admission": "not_run"}}
    out.parent.mkdir(parents=True, exist_ok=True)
    _publish_new_files([*companions, (receipt_output, _json(result)), (out, text)])
    return result


def _start_through_forcing_editor(*, raw, out, authority, document, timed,
                                  timing, hours, carried, saved_wps):
    """The ERA5 arm: the existing forcing editor owns case-data forcing paths."""
    from gpuwm import companion_forcing
    from gpuwm.starter_template import _publish_new_files
    from gpuwm.toml_document import emit_experiment_toml

    saved_fetch = raw["fetch"]
    if saved_wps is not None:
        # The editor renders the new WPS from the one the configuration
        # names, so it must name the saved companion, not the file it is
        # about to write; the editor points the output at that file itself.
        raw["case_data"]["wps_namelist"] = str(saved_wps)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Declared inputs carried beside the new configuration are published
    # first and taken back if the editor refuses, so a refusal writes nothing.
    _publish_new_files(carried)
    try:
        with tempfile.TemporaryDirectory(prefix="arwen-setup-era5-") as directory:
            staged = Path(directory) / out.name
            staged.write_text(START_HEADER + emit_experiment_toml(raw),
                              encoding="utf-8", newline="\n")
            forcing = companion_forcing.edit_configuration({
                "schema": companion_forcing.REQUEST_SCHEMA,
                "config_path": str(staged),
                "expected_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                "output_path": str(out),
                "product_type": saved_fetch.get("era5_product", "reanalysis"),
                "cadence_hours": timing["cadence_hours"],
                "member": saved_fetch.get("member"),
                "provider": saved_fetch.get("era5_provider", "cds"),
                "run_seconds": hours * 3600})
    except BaseException:
        for path, _content in carried:
            with contextlib.suppress(OSError):
                path.unlink()
        raise
    # The forcing editor has already published the configuration, its WPS and
    # its own receipt. A second receipt written afterwards could fail with the
    # configuration already on disk, so its receipt is the one reported.
    return {
        "schema": RESULT_SCHEMA, "action": "start", "created": True,
        "forecast_started": False, "acquisition_started": False,
        "config_path": forcing["config_path"],
        "config_sha256": forcing["config_sha256"],
        "wps_path": forcing["wps_path"], "receipt_path": forcing["receipt_path"],
        "setup_path": str(authority.source), "setup_sha256": authority.sha256,
        "setup_name": document.get("name"), "timing": timing,
        "changes": timed, "configuration": forcing["configuration"],
        "validation": dict(forcing["validation"], fetch_hints="passed")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    from gpuwm.companion_domains import _json
    action = getattr(args, "setups_action", None)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if action == "save":
                result = save_setup(config_path=args.config, library=args.library,
                                    name=args.name)
            elif action == "start":
                result = start_setup(
                    setup_path=args.setup, cycle=args.cycle, hours=args.hours,
                    forecast_start_hour=args.forecast_start_hour,
                    name=args.name, out=args.out)
            else:
                raise ValueError("Choose an action: save or start")
        print(_json(result), end="")
        return 0
    except Exception as error:
        print(_json({"schema": RESULT_SCHEMA, "action": action, "created": False,
                     "forecast_started": False, "acquisition_started": False,
                     "error": str(error)}), end="")
        return 1


def register_cli(subparsers):
    parser = subparsers.add_parser(
        "companion-setups",
        help="save a forecast configuration as a reusable setup, or start a "
             "new forecast from a saved one")
    actions = parser.add_subparsers(prog=parser.prog, dest="setups_action",
                                    metavar="ACTION", required=True)
    save = actions.add_parser(
        "save", help="copy a forecast configuration and its companions into "
                     "the setups library under a name")
    save.add_argument("--config", type=Path, required=True, metavar="CONFIG",
                      help="forecast configuration TOML to save as a setup")
    save.add_argument("--library", type=Path, required=True, metavar="DIR",
                      help="setups library directory the setup folder is created in")
    save.add_argument("--name", required=True, metavar="NAME",
                      help="name to save this setup under; 1 to 80 characters")
    save.set_defaults(func=main, setups_action="save")
    start = actions.add_parser(
        "start", help="write a new forecast configuration from a saved setup "
                      "at a new cycle, keeping every saved setting")
    start.add_argument("setup", type=Path, metavar="SETUP",
                       help="saved setup.toml to start a forecast from")
    start.add_argument("--cycle", required=True, metavar="CYCLE",
                       help="source cycle as YYYY-MM-DDTHH (UTC), or latest")
    start.add_argument("--hours", type=float, required=True, metavar="H",
                       help="forecast duration in hours")
    start.add_argument("--forecast-start-hour", type=int, default=0, metavar="N",
                       help="forecast lead in hours after the cycle the run starts at")
    start.add_argument("--name", required=True, metavar="NAME",
                       help="name of the new forecast")
    start.add_argument("--out", type=Path, required=True, metavar="TOML",
                       help="path of the new configuration TOML to write")
    start.set_defaults(func=main, setups_action="start")
    parser.set_defaults(func=main)


if __name__ == "__main__":
    root = argparse.ArgumentParser(description=__doc__)
    register_cli(root.add_subparsers(dest="command", metavar="COMMAND",
                                     required=True))
    raise SystemExit(main(root.parse_args()))
