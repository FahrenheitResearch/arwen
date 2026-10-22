"""Another forecast system's suites, domains and cycles, as table rows.

A forecast system the engine imitates "in kind" (the same scheme families,
grid spacing, levels, inputs and cadence, on this engine's own dycore) is
described by one JSON table under ``gpuwm/data/system_suites/``.  Adding a
system is adding a table: nothing here knows any system by name, and no
adapter file exists per system.

What a table states, and what this module holds it to:

* **suites** -- one row per suite of the imitated system.  Each physics
  component maps to an option id of the physics registry, or to ``null``
  when the engine has no scheme of that family.  ``settings`` are the exact
  run settings a profile carries for the suite; every one is a component
  selector or an implemented registry knob, and every selector agrees with
  the option its component maps to.
* **declared_missing** -- every scheme of the imitated suite the engine
  lacks, with the work package that would add it and what the run does
  instead.  "Instead" is never another scheme: a missing scheme runs as
  absent (its component ``off``) or the suite is not runnable.  That is the
  whole point of the rows: a silent substitution is the breakage they
  prevent.
* **declared_refused** -- options of the imitated suite the engine refuses
  today and runs at its own value, each with the refusal site and package.
* **source_settings** and **source_schemes** -- what the suite's own
  namelist sets (with line numbers) and what its own suite file runs.  The
  table's **source_rules** name every setting or scheme the engine cannot
  honour and the declared row it obliges; a holding rule without its row,
  or a declared row without a holding rule, is a problem.  A suite that
  ``derives_from`` another inherits its declared rows only through named
  additions and removals, never wholesale, and never inherits its source
  settings or schemes.
* **domains**, **cycles**, **boundary_arms** and **profiles** -- the grid
  templates, the forecast lengths by start hour, the boundary arms, and the
  experiment configurations that carry a suite onto a domain.

The rows are published into the physics registry under ``system_suites``
(``tools/apply_system_suites.py`` renders them; a test holds the registry
equal to the tables).  That block sits outside every selection's physics
identity: no template, route or default resolves to a row, so a
configuration that does not name one is unaffected byte for byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

SCHEMA = "gpuwm-system-suites-v1"
TABLE_DIR = Path(__file__).resolve().parent / "data" / "system_suites"
REGISTRY_BLOCK = "system_suites"

#: Keys a profile's [shared] or [[domain]] table may carry that a suite's
#: ``settings`` decide.  Radiation's 4/4 pair may be spelled through
#: ``ra_physics`` in a configuration, so both spellings are read.
_RADIATION_SPELLING = ("ra_physics", "ra_lw_physics", "ra_sw_physics")


class SystemSuiteError(ValueError):
    """A table that does not hold to the rules above."""


def table_paths(directory: Path | None = None) -> list[Path]:
    root = TABLE_DIR if directory is None else Path(directory)
    return sorted(root.glob("*.json"))


def load_table(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise SystemSuiteError(f"{path}: schema is not {SCHEMA}")
    if not isinstance(doc.get("system"), str) or not doc["system"]:
        raise SystemSuiteError(f"{path}: names no system")
    return doc


def load_tables(directory: Path | None = None) -> dict[str, dict]:
    tables: dict[str, dict] = {}
    for path in table_paths(directory):
        doc = load_table(path)
        if doc["system"] in tables:
            raise SystemSuiteError(f"system {doc['system']!r} is declared twice")
        tables[doc["system"]] = doc
    return tables


#: Keys a suite row states from its own source files and never inherits:
#: a derived suite has its own namelist and its own suite file.
_OWN_SOURCE_KEYS = ("source_suite", "source_ref", "used_by", "runnable",
                    "not_runnable_because", "notes", "source_settings",
                    "source_schemes")

#: How a derived row changes its parent's declarations.  A wholesale
#: replacement is refused: it is how a child once dropped two declarations
#: its own namelist needed and kept a refused row its namelist does not set.
_MERGE_KEYS = {
    "declared_missing": ("part", "declared_missing_add",
                         "declared_missing_remove"),
    "declared_refused": ("setting", "declared_refused_add",
                         "declared_refused_remove"),
}


def _merged_declarations(suite_id: str, parent: list, row: Mapping,
                         field: str) -> list:
    key, add_key, remove_key = _MERGE_KEYS[field]
    if field in row:
        raise SystemSuiteError(
            f"suite {suite_id!r} derives from a parent and states {field} "
            f"wholesale; a derived suite states {add_key} and {remove_key}")
    remove = list(row.get(remove_key) or ())
    have = [entry.get(key) for entry in parent]
    for name in remove:
        if name not in have:
            raise SystemSuiteError(
                f"suite {suite_id!r} removes {name!r} from {field}, which its "
                "parent does not declare")
    merged = [dict(entry) for entry in parent if entry.get(key) not in remove]
    for entry in row.get(add_key) or ():
        if entry.get(key) in [e.get(key) for e in merged]:
            raise SystemSuiteError(
                f"suite {suite_id!r} adds {entry.get(key)!r} to {field} twice")
        merged.append(dict(entry))
    return merged


def resolved_suite(doc: Mapping, suite_id: str,
                   _ancestors: tuple[str, ...] = ()) -> dict:
    """A suite row with its ``derives_from`` parent folded in.

    The merge is explicit: the parent's components, settings and declared
    rows, plus the child's additions, minus the child's named removals.
    What a row states from its own source files (``_OWN_SOURCE_KEYS``) is
    never inherited.
    """
    suites = doc.get("suites") or {}
    if suite_id in _ancestors:
        raise SystemSuiteError(
            f"suite inheritance cycle {' -> '.join((*_ancestors, suite_id))}; "
            "the table cannot resolve its settings or missing schemes")
    if suite_id not in suites:
        raise SystemSuiteError(f"{doc.get('system')}: no suite {suite_id!r}")
    row = dict(suites[suite_id])
    parent_id = row.get("derives_from")
    if parent_id is None:
        for _key, add_key, remove_key in _MERGE_KEYS.values():
            if add_key in row or remove_key in row:
                raise SystemSuiteError(
                    f"suite {suite_id!r} derives from nothing and states "
                    f"{add_key} or {remove_key}")
        row.setdefault("settings", {})
        row.setdefault("declared_missing", [])
        row.setdefault("declared_refused", [])
        return row
    if parent_id == suite_id:
        raise SystemSuiteError(f"suite {suite_id!r} derives from itself")
    parent = resolved_suite(doc, parent_id, (*_ancestors, suite_id))
    merge_inputs = {name for spec in _MERGE_KEYS.values() for name in spec[1:]}
    merged = {k: v for k, v in parent.items()
              if k not in _OWN_SOURCE_KEYS and k != "derived_from"}
    merged.update({k: v for k, v in row.items()
                   if k != "settings_override" and k not in merge_inputs
                   and k not in _MERGE_KEYS})
    merged["components"] = dict(parent.get("components") or {})
    merged["components"].update(row.get("components") or {})
    merged["settings"] = {**parent["settings"],
                          **(row.get("settings_override") or {})}
    for field in _MERGE_KEYS:
        merged[field] = _merged_declarations(suite_id, parent[field], row, field)
    merged.pop("derives_from", None)
    merged["derived_from"] = parent_id
    return merged


def _effective_source_value(row: Mapping, defaults: Mapping, name: str):
    stated = (row.get("source_settings") or {}).get("set") or {}
    if name in stated:
        return stated[name]
    return (defaults.get(name) or {}).get("value")


def source_declaration_problems(doc: Mapping, suite_id: str) -> list[str]:
    """Where a suite's declared rows disagree with its own source settings.

    The table's ``source_rules`` name every source setting or scheme the
    engine cannot honour and the declared row each obliges.  A holding rule
    without its row is a silent absence; a declared row without a holding
    rule states a source value the imitated system does not use.  Both are
    the breakage these rows exist to prevent.
    """
    where = f"{doc.get('system')}:{suite_id}"
    rules_doc = doc.get("source_rules") or {}
    rules = rules_doc.get("rules") or []
    defaults = rules_doc.get("defaults") or {}
    row = resolved_suite(doc, suite_id)
    problems: list[str] = []
    settings = row.get("source_settings")
    schemes = row.get("source_schemes")
    if (not isinstance(settings, dict) or not settings.get("file")
            or not isinstance(settings.get("set"), dict)):
        problems.append(f"{where}: source_settings (file and set) are missing")
        settings = {"set": {}}
    if not isinstance(schemes, list) or not schemes:
        problems.append(f"{where}: source_schemes are missing")
        schemes = []
    lines = settings.get("lines") or {}
    for name in settings.get("set") or {}:
        if name not in lines:
            problems.append(f"{where}: source setting {name!r} has no line")
    missing = {e.get("part"): e for e in row["declared_missing"]}
    refused = {e.get("setting"): e for e in row["declared_refused"]}
    holding_missing: set = set()
    holding_refused: dict = {}
    governed_missing: set = set()
    governed_refused: set = set()
    for rule in rules:
        if "when_scheme" in rule:
            holds = rule["when_scheme"] in schemes
            what = f"its suite runs {rule['when_scheme']}"
            value = None
        else:
            name = rule["when_setting"]
            if name not in defaults:
                problems.append(f"{doc.get('system')}: source rule on {name!r} "
                                "has no default")
                continue
            value = _effective_source_value(row, defaults, name)
            holds = value not in rule.get("engine_honours", ())
            what = f"{name} = {value!r}"
        if "declares_missing" in rule:
            part = rule["declares_missing"]
            governed_missing.add(part)
            if holds:
                holding_missing.add(part)
                if part not in missing:
                    problems.append(f"{where}: {what}, which the engine cannot "
                                    f"honour, but {part!r} is not declared missing")
        if "declares_refused" in rule:
            setting = rule["declares_refused"]
            governed_refused.add(setting)
            if holds:
                holding_refused[setting] = value
                entry = refused.get(setting)
                if entry is None:
                    problems.append(f"{where}: {what}, which the engine cannot "
                                    f"honour, but {setting!r} is not declared refused")
                elif entry.get("source_value") != value:
                    problems.append(f"{where}: {setting} is declared refused at "
                                    f"{entry.get('source_value')!r}; the source "
                                    f"runs {value!r}")
    for part in missing:
        if part not in governed_missing:
            problems.append(f"{where}: declared-missing {part!r} has no source rule")
        elif part not in holding_missing:
            problems.append(f"{where}: declares {part!r} missing, but its source "
                            "does not run it")
    for setting in refused:
        if setting not in governed_refused:
            problems.append(f"{where}: declared-refused {setting!r} has no source rule")
        elif setting not in holding_refused:
            problems.append(f"{where}: declares {setting!r} refused, but its "
                            "source does not set a value the engine refuses")
    return problems


def _setting_problems(where: str, settings: Mapping, registry: Mapping) -> list[str]:
    problems: list[str] = []
    selector_keys = {key for component in registry["components"].values()
                     for key in component.get("selector_keys") or ()}
    parameters = registry["parameters"]
    for name in settings:
        if name in selector_keys:
            continue
        spec = parameters.get(name)
        if spec is None:
            problems.append(f"{where}: setting {name!r} is neither a component "
                            "selector nor a registry knob")
        elif spec.get("implemented") is False or not spec.get("consuming_read"):
            problems.append(f"{where}: setting {name!r} is a knob the engine "
                            "does not implement")
    return problems


def suite_problems(doc: Mapping, suite_id: str, registry: Mapping) -> list[str]:
    """Every way one suite row breaks the rules in the module docstring."""
    where = f"{doc.get('system')}:{suite_id}"
    problems: list[str] = []
    row = resolved_suite(doc, suite_id)
    problems.extend(source_declaration_problems(doc, suite_id))
    for key in ("source_suite", "source_ref", "used_by"):
        if not row.get(key):
            problems.append(f"{where}: {key} is missing")
    runnable = row.get("runnable")
    if not isinstance(runnable, bool):
        problems.append(f"{where}: runnable must be true or false")
    if runnable is False and not row.get("not_runnable_because"):
        problems.append(f"{where}: a suite that cannot run must say why")
    for entry in row["declared_missing"]:
        for key in ("part", "source_scheme", "runs_as", "package"):
            if not entry.get(key):
                problems.append(f"{where}: a declared-missing row lacks {key}")
    for entry in row["declared_refused"]:
        for key in ("setting", "source_value", "engine_value",
                    "refusal_site", "package"):
            if key not in entry or entry[key] in (None, ""):
                problems.append(f"{where}: a declared-refused row lacks {key}")
    components = row.get("components") or {}
    for component_id, mapping in components.items():
        if component_id not in registry["components"]:
            problems.append(f"{where}: no registry component {component_id!r}")
            continue
        option_id = mapping.get("option")
        relation = str(mapping.get("relation") or "")
        if option_id is None:
            if runnable:
                problems.append(f"{where}: {component_id} has no engine option, "
                                "so the suite cannot be runnable")
            if "declared missing" not in relation:
                problems.append(f"{where}: {component_id} has no engine option "
                                "and is not declared missing")
            continue
        options = registry["components"][component_id]["options"]
        option = options.get(option_id)
        if option is None:
            problems.append(f"{where}: {component_id} option {option_id!r} is "
                            "not in the registry")
            continue
        if option.get("implemented") is not True:
            problems.append(f"{where}: {component_id} option {option_id!r} is "
                            "not implemented")
        if runnable:
            for key, value in (option.get("selectors") or {}).items():
                have = row["settings"].get(key)
                if key in ("ra_lw_physics", "ra_sw_physics") and have is None:
                    have = row["settings"].get("ra_physics")
                if have != value:
                    problems.append(
                        f"{where}: {component_id} maps to {option_id!r}, whose "
                        f"selector {key} = {value!r}, but the settings say "
                        f"{have!r}")
    if runnable:
        problems.extend(_setting_problems(where, row["settings"], registry))
        # The off option of a component whose source names a scheme is a
        # declared absence, never a quiet one.
        for component_id, mapping in components.items():
            if (mapping.get("option") == "off"
                    and "declared missing" not in str(mapping.get("relation"))
                    and mapping.get("relation") != "match"):
                problems.append(f"{where}: {component_id} runs off without a "
                                "declared-missing relation")
    return problems


def table_problems(doc: Mapping, registry: Mapping,
                   repository_root: Path | None = None) -> list[str]:
    problems: list[str] = []
    system = doc.get("system")
    suites = doc.get("suites") or {}
    if not suites:
        problems.append(f"{system}: declares no suites")
    for suite_id in suites:
        try:
            problems.extend(suite_problems(doc, suite_id, registry))
        except SystemSuiteError as exc:
            problems.append(str(exc))
    for diff in doc.get("declared_differences") or ():
        for key in ("id", "source", "here", "status"):
            if not diff.get(key):
                problems.append(f"{system}: a declared difference lacks {key}")
    cycles = doc.get("cycles") or {}
    for stream, row in cycles.items():
        if not isinstance(row, dict):
            continue
        lengths = row.get("lengths_h") or {}
        if not lengths or not all(isinstance(v, int) and v >= 0
                                  for v in lengths.values()):
            problems.append(f"{system}: cycle {stream!r} has no whole-hour lengths")
        if not isinstance(row.get("other_hours_h"), int):
            problems.append(f"{system}: cycle {stream!r} lacks other_hours_h")
    domains = doc.get("domains") or {}
    arms = doc.get("boundary_arms") or {}
    root = repository_root or Path(__file__).resolve().parents[1]
    for profile in doc.get("profiles") or ():
        config = profile.get("config")
        if profile.get("suite") not in suites:
            problems.append(f"{system}: profile {config} names no known suite")
        if profile.get("domain") not in domains:
            problems.append(f"{system}: profile {config} names no known domain")
        if profile.get("arm") not in arms:
            problems.append(f"{system}: profile {config} names no known arm")
        if profile.get("stream") not in cycles:
            problems.append(f"{system}: profile {config} names no known stream")
        if not config or not (root / config).is_file():
            problems.append(f"{system}: profile config {config!r} does not exist")
    return problems


def forecast_length_hours(doc: Mapping, stream: str, start_hour: int) -> int:
    """The imitated system's forecast length for a start hour (0 = no run)."""
    row = (doc.get("cycles") or {}).get(stream)
    if not isinstance(row, dict):
        raise SystemSuiteError(f"{doc.get('system')}: no cycle stream {stream!r}")
    hour = f"{int(start_hour) % 24:02d}"
    return int((row.get("lengths_h") or {}).get(hour, row["other_hours_h"]))


def profile_mismatches(doc: Mapping, profile: Mapping,
                       experiment: Mapping) -> list[str]:
    """Where a profile configuration's values differ from its suite's settings.

    ``experiment`` is the parsed TOML.  A setting may sit in [shared] or in
    every [[domain]] table; a domain value wins, as the loader applies it.
    """
    row = resolved_suite(doc, profile["suite"])
    shared = experiment.get("shared") or {}
    domains = experiment.get("domain") or [{}]
    problems: list[str] = []
    for name, expected in row["settings"].items():
        for index, domain in enumerate(domains):
            value = domain.get(name, shared.get(name))
            if value is None and name in ("ra_lw_physics", "ra_sw_physics"):
                value = domain.get("ra_physics", shared.get("ra_physics"))
            if value != expected:
                problems.append(f"{profile['config']}: domain {index + 1} "
                                f"{name} = {value!r}, suite says {expected!r}")
    run_seconds = (experiment.get("experiment") or {}).get("run_seconds")
    if run_seconds is not None and float(run_seconds) != 3600.0 * float(
            profile.get("length_h", -1)):
        problems.append(f"{profile['config']}: run_seconds {run_seconds} is not "
                        f"the profile's {profile.get('length_h')} h")
    problems.extend(_arm_mismatches(doc, profile, experiment))
    return problems


def _arm_mismatches(doc: Mapping, profile: Mapping,
                    experiment: Mapping) -> list[str]:
    """Where a profile's data sources differ from its boundary arm.

    A profile's [fetch] table names one source, which gives both the start
    (at lead ``forecast_start_hour``) and the boundaries.  An arm whose start
    comes from another source than its boundaries cannot be carried by a
    profile today, so such a profile is named here rather than run under the
    arm's label with a different start.
    """
    config = profile.get("config")
    arm = (doc.get("boundary_arms") or {}).get(profile.get("arm"))
    if not isinstance(arm, Mapping):
        return [f"{config}: names no known arm"]
    fetch = experiment.get("fetch") or {}
    source = fetch.get("source")
    lead = int(fetch.get("forecast_start_hour", 0) or 0)
    problems: list[str] = []
    if source != arm.get("forcing_source"):
        problems.append(f"{config}: [fetch] source {source!r} is not arm "
                        f"{profile.get('arm')}'s boundary source "
                        f"{arm.get('forcing_source')!r}")
    if source != arm.get("initial_source"):
        problems.append(f"{config}: the run starts from {source!r}, but arm "
                        f"{profile.get('arm')} starts from "
                        f"{arm.get('initial_source')!r}")
    offset = int(arm.get("forcing_cycle_offset_h", 0) or 0)
    if lead != offset:
        problems.append(f"{config}: the run starts at lead f{lead:03d}, but arm "
                        f"{profile.get('arm')} takes the {offset} h older "
                        "forcing cycle")
    return problems


def registry_rows(tables: Mapping[str, Mapping]) -> dict:
    """The ``system_suites`` block of the physics registry, from the tables."""
    block: dict[str, object] = {
        "declaration": (
            "Suites of other forecast systems that this engine imitates in "
            "kind, rendered from gpuwm/data/system_suites/*.json by "
            "tools/apply_system_suites.py. Each component maps to a registry "
            "option or to null; declared_missing names every scheme the engine "
            "lacks and what the run does instead, which is never another "
            "scheme; declared_refused names options run at the engine's own "
            "value. No template, route or default resolves to a row, so the "
            "block is outside every selection's physics identity."),
        "systems": {},
    }
    for system in sorted(tables):
        doc = tables[system]
        suites = {}
        for suite_id in sorted(doc.get("suites") or {}):
            row = resolved_suite(doc, suite_id)
            suites[suite_id] = {
                "source_suite": row["source_suite"],
                "runnable": row["runnable"],
                "components": {
                    component_id: mapping.get("option")
                    for component_id, mapping in sorted(
                        (row.get("components") or {}).items())
                },
                "settings": dict(sorted(row["settings"].items())) if row["runnable"] else {},
                "declared_missing": [
                    {"part": e["part"], "package": e["package"],
                     "runs_as": e["runs_as"]}
                    for e in row["declared_missing"]],
                "declared_refused": [
                    {"setting": e["setting"], "source_value": e["source_value"],
                     "engine_value": e["engine_value"], "package": e["package"]}
                    for e in row["declared_refused"]],
            }
        block["systems"][system] = {
            "label": doc.get("label"),
            "match_level": doc.get("match_level"),
            "physics_source_pin": (doc.get("pins") or {}).get("physics"),
            "suites": suites,
        }
    return block
