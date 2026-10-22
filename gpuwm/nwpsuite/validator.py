"""Strict schema, reference and inheritance resolution."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

from .registry import DATA, Registries, SuiteRefusal, read_json
from .overrides import effective_row, selection, validate_flags


def merge(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        if key != 'at' and isinstance(value, dict) and value and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def check_schema(value):
    from jsonschema import Draft202012Validator
    errors = sorted(Draft202012Validator(read_json(DATA / "suite.v1.schema.json")).iter_errors(value), key=lambda e: str(list(e.path)))
    if errors:
        error = errors[0]
        path = ".".join(str(v) for v in error.path) or "suite"
        raise SuiteRefusal(f"{path}: {error.message}; prevents HAILCAST/mp_tend_radar style silent key drops")


def template_path(name, registries):
    if not name.startswith("template:"):
        raise SuiteRefusal(f"unknown template reference {name!r}")
    wanted = name.split(":", 1)[1]
    path = registries.directory / "templates" / (wanted + ".toml")
    if not path.is_file() or path.parent != registries.directory / "templates":
        raise SuiteRefusal(f"unknown template {wanted!r}; prevents inheriting an unintended suite")
    return path


def load_suite(path, *, registry_dir=None, _seen=(), _resolve=True):
    path = Path(path).resolve()
    if path in _seen:
        raise SuiteRefusal(f"extends cycle at {path}; prevents recursive suite inheritance")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except tomllib.TOMLDecodeError as error:
        raise SuiteRefusal(str(error)) from error
    check_schema(raw)
    ids = [row['id'] for row in raw.get('system', [])]
    if len(ids) != len(set(ids)):
        raise SuiteRefusal('duplicate system id; prevents inheritance overwriting a row')
    selected = registry_dir or raw.get("policy", {}).get("registry_dir")
    if selected and not Path(selected).is_absolute():
        selected = path.parent / selected
    registries = Registries(selected)
    if raw.get("extends"):
        base = load_suite(template_path(raw["extends"], registries), registry_dir=selected, _seen=(*_seen, path), _resolve=False)
        systems = {r["id"]: r for r in base["system"]}
        for row in raw.get("system", []):
            systems[row["id"]] = merge(systems.get(row["id"], {}), row)
        raw = merge(base, raw)
        raw["system"] = list(systems.values())
    return validate_suite(raw, registries=registries, base=path.parent) if _resolve else raw


def validate_suite(raw, *, registries=None, base=None):
    check_schema(raw)
    registries = registries or Registries(raw.get("policy", {}).get("registry_dir"))
    region = registries.region(raw["region"])
    rate_ids = {row['id'] for row in registries.rates(raw.get('policy',{}).get('rates'),base)}
    systems = raw.get("system", [])
    ids = [row["id"] for row in systems]
    if len(ids) != len(set(ids)):
        raise SuiteRefusal("duplicate system id; prevents a row being overwritten")
    original = {row["id"]: row for row in systems}
    resolved = {}

    def resolve(name, stack=()):
        if name in stack:
            raise SuiteRefusal(f"from cycle {' -> '.join((*stack, name))}; prevents recursive row inheritance")
        if name not in original:
            raise SuiteRefusal(f"unknown system reference {name!r}; prevents missing dependency work")
        if name not in resolved:
            row = original[name]
            resolved[name] = merge(resolve(row["from"], (*stack, name)), row) if row.get("from") else deepcopy(row)
        return resolved[name]

    for name in ids:
        resolve(name)
    boxes = raw.get("box", [])
    box_ids = [box["id"] for box in boxes]
    if not boxes or len(box_ids) != len(set(box_ids)):
        raise SuiteRefusal("missing or duplicate box id; prevents placing work on an unidentified host")
    for row in resolved.values():
        context = row["id"]
        shape = row.get("shape")
        if shape not in registries.shapes:
            raise SuiteRefusal(f"{context}: unknown shape {shape!r}")
        if row.get("box") not in box_ids:
            raise SuiteRefusal(f"{context}: unknown box {row.get('box')!r}")
        validate_flags(row, registries.shapes[shape])
        for settings in (row, row.get('da',{})):
            if settings.get('rate') and settings['rate'] not in rate_ids:
                raise SuiteRefusal(f"{context}: unknown rate reference {settings['rate']!r}; prevents a workload using an unintended timing or memory receipt")
        if not {"every", "lead"} <= row.get("cycle", {}).keys():
            raise SuiteRefusal(f"{context}: cycle.every and cycle.lead are required")
        from .planner import seconds
        cadence = seconds(row['cycle']['every'])
        if cadence < 3600 or cadence % 3600 or (cadence <= 86400 and 86400 % cadence):
            raise SuiteRefusal(f'{context}: unsupported cadence {row["cycle"]["every"]}; prevents missing sub-hour or drifting cycles')
        if seconds(row['cycle']['lead']) <= 0:
            raise SuiteRefusal(f'{context}: lead must be positive')
        if row.get('lts') == 'one-card' and row.get('cards',1) != 1:
            raise SuiteRefusal(f'{context}: one-card LTS with multiple cards; prevents unqualified multi-card time stepping')
        for selector in row.get('physics', {}):
            if selector not in registries.catalog['physics_keys']:
                raise SuiteRefusal(f'{context}: unknown physics key {selector!r}; prevents a silently dropped scheme')
        if row.get('physics') and registries.shapes[shape].get('override_transport') == 'unsupported':
            raise SuiteRefusal(f'{context}: door cannot carry physics overrides; prevents a silently dropped scheme')
        for key, registry in (("recipe", "recipes"), ("mesh", "meshes"), ("preset", "presets"), ("source_profile", "source_profiles")):
            if key in row:
                registries.require(registry, row[key], context)
        for key in ("ic", "lbc"):
            for value in row.get(key, []):
                registries.require("sources", value, context)
        perturb=row.get('perturb',{})
        if perturb.get('recipe'):
            registries.require('ensemble_recipes',perturb['recipe'],context)
        for value in perturb.get('stochastic',[]):
            registries.require('stochastic',value,context)
        method = row.get("da", {}).get("method", "none")
        if method not in registries.shapes[shape]["da_methods"]:
            raise SuiteRefusal(f"{context}: DA method {method!r} unsupported by shape {shape}")
        if row.get("fallback") and row["fallback"] not in resolved:
            raise SuiteRefusal(f"{context}: unknown fallback {row['fallback']!r}")
        da = row.get("da", {})
        if da.get('every'):
            cadence=seconds(da['every'])
            if cadence<3600 or cadence%3600 or cadence<=86400 and 86400%cadence:
                raise SuiteRefusal(f"{context}: unsupported DA cadence {da['every']!r}; prevents missing or drifting DA cycles")
        if da.get('lead') and seconds(da['lead'])<=0:
            raise SuiteRefusal(f'{context}: DA lead must be positive')
        if "mesh" in da:
            registries.require("meshes", da["mesh"], context)
        if da.get("b_from") and da["b_from"] not in resolved:
            raise SuiteRefusal(f"{context}: unknown b_from {da['b_from']!r}")
        for value in da.get("obs", []):
            registries.require("obs", value, context)
        if row.get("products"):
            registries.require("products", row["products"], context)
        used = set()
        for hours, patch in row.get("at", {}).items():
            values = {int(v) for v in hours.split(",")}
            if used & values:
                raise SuiteRefusal(f"{context}: overlapping at hours {hours}; prevents ambiguous cycle settings")
            used |= values
            # Validate each effective hour as a complete row, below.
            candidate = cycle_row(row, next(iter(values)))
            candidate.pop('at', None)
            candidate.pop('from', None)
            other_rows = [dict(value, at={}) for value in resolved.values() if value['id'] != row['id']]
            for value in other_rows:
                value.pop('from', None)
            validate_suite(dict(raw, system=[*other_rows, candidate], override=[]), registries=registries, base=base)
            for key, registry in (("recipe", "recipes"), ("mesh", "meshes"), ("preset", "presets")):
                if key in candidate:
                    registries.require(registry, candidate[key], context + ".at")
            if candidate["shape"] not in registries.shapes or candidate["box"] not in box_ids:
                raise SuiteRefusal(f"{context}: unknown shape or box in at {hours}")
    defaults = raw.get("defaults", {})
    if "products" in defaults:
        registries.require("products", defaults["products"], "defaults")
    for target in defaults.get("publish", {}).get("targets", []):
        registries.require("publish", target, "defaults")
    for override in raw.get("override", []):
        if override["system"] not in resolved:
            raise SuiteRefusal(f"unknown override system {override['system']!r}")
        for key in override["set"]:
            match=selection(key)
            scheme=match[4]
            row=resolved[override['system']]; shape=registries.shapes[row['shape']]
            if match[1] is not None:
                members=resolved[override['system']].get('members',1)
                count=members if isinstance(members,int) else members.get('perturbed',0)
                start=int(match[1]); stop=int(match[2] or match[1])
                if start>stop or stop>=count:
                    raise SuiteRefusal(f'override member selector {key!r} outside {count} members; prevents a dropped member override')
            if match[3]=='physics' and scheme not in registries.catalog["physics_keys"]:
                raise SuiteRefusal(f"unknown override physics key {scheme!r}; prevents a dropped scheme selector")
            if match[3]=='physics' and shape.get('override_transport')=='unsupported':
                raise SuiteRefusal(f"{row['id']}: door cannot carry physics override {scheme!r}; prevents a dropped scheme selector")
            if match[3]=='door':
                validate_flags(dict(row,door_flags={scheme:override['set'][key]}),shape)
                if match[1] is not None and shape['door_flags'][scheme]['admission']=='row':
                    raise SuiteRefusal(f'member resource override {key!r} requires a separate row; prevents pricing unlike leads or card counts as one rate')
    # Flag aliases change the same fields the planner prices. Validate them as complete rows.
    if raw.get('override') or any(row.get('door_flags') for row in resolved.values()):
        normalized=[]
        for row in resolved.values():
            candidate=effective_row(row,raw.get('override',[]),registries.shapes[row['shape']])
            candidate.pop('door_flags',None); candidate.pop('from',None)
            normalized.append(candidate)
        validate_suite(dict(raw,system=normalized,override=[]),registries=registries,base=base)
    result = deepcopy(raw)
    result["system"] = list(resolved.values())
    return result


def cycle_row(row, hour):
    result = deepcopy(row)
    for hours, patch in row.get("at", {}).items():
        if hour in [int(v) for v in hours.split(",")]:
            controls = {key: patch[key] for key in ("every", "lead", "deadline") if key in patch}
            result = merge(result, {key: value for key, value in patch.items() if key not in controls})
            result["cycle"] = merge(result["cycle"], controls)
    return result


def suite_hash(suite):
    return hashlib.sha256(json.dumps(suite, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
