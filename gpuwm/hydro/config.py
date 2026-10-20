"""Resolve NextGen composition metadata once, before GPU preparation.

This module never loads native model libraries or processes hydrologic fields.
New compositions and component factories are rows in the realization table.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

_TABLE = Path(__file__).resolve().parents[1] / "data/hydro/configurations.v1.json"
_SCHEMA = "gpuwm-nextgen-realizations-v1"
_PARAMETER = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}")


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _copy(value):
    if isinstance(value, Mapping):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy(item) for item in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_copy(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _merge(base, override):
    """Override mappings recursively; ordered stage/module lists replace whole."""
    result = _copy(base)
    for key, value in override.items():
        result[key] = (_merge(result[key], value)
                       if isinstance(result.get(key), Mapping) and isinstance(value, Mapping)
                       else _copy(value))
    return result


@dataclass(frozen=True)
class Realization:
    name: str
    row: Mapping
    sha256: str

    def as_dict(self):
        """Return independent JSON metadata for the setup factory or referee."""
        return _copy(self.row)


def resolve_configuration(name, *, table=None):
    """Resolve any named row; no realization-specific branches or defaults."""
    if isinstance(table, Mapping):
        document = table
    else:
        document = json.loads((_TABLE if table is None else Path(table)).read_text())
    if document.get("schema") != _SCHEMA:
        raise ValueError(f"expected realization table schema {_SCHEMA}")
    rows = document["realizations"]
    names = [row["id"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("duplicate realization IDs make configuration selection ambiguous")
    try:
        row = rows[names.index(name)]
    except ValueError:
        raise ValueError(f"unknown realization {name!r}; available: {', '.join(names)}") from None
    admission = row["admission"]
    if admission["status"] != "permitted":
        raise ValueError(f"realization {name!r} unavailable: {admission['reason']}")
    return Realization(name, _freeze(_copy(row)), _digest(row))


def _bind(value, parameters, bindings, label):
    """Bind declared metadata tokens once; leave native {{id}} tokens intact."""
    required = set(parameters)
    missing, extra = required - bindings.keys(), bindings.keys() - required
    if missing or extra:
        raise ValueError(f"{label} bindings missing={sorted(missing)}, unknown={sorted(extra)}")
    if any(not isinstance(value, str) for value in bindings.values()):
        raise ValueError(f"{label} bindings must be strings")

    def visit(value):
        if isinstance(value, Mapping):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(item) for item in value]
        if isinstance(value, str):
            return _PARAMETER.sub(lambda match: bindings[match[1]], value)
        return value

    return visit(value)


def _bind_reference(row, bindings):
    return _bind(row.get("reference", {}), row.get("reference_parameters", ()),
                 bindings, "reference")


def materialize_reference(name, bindings, *, table=None):
    """Return bound configuration metadata, without loading native libraries.

    Only explicit ${parameter} tokens are replaced, in one pass. Numerical
    fields and native per-catchment {{id}} expansion belong to their owners.
    """
    return _bind_reference(resolve_configuration(name, table=table).row, bindings)


def configure(settings, *, table=None):
    """Expand a table row for runtime.prepare, without importing GPU modules.

    Explicit settings override metadata defaults, including parameter bindings
    and ordered stage rows. Native reference documents are provenance only;
    assembly uses the prepared GPU factory rows in ``stages``.

    ``settings_bindings`` substitutes declared ``settings_parameters`` tokens
    in runtime defaults after overlays. Use fresh input settings to change these
    bindings; an expanded stage list already contains literal resource paths.
    """
    selected = resolve_configuration(settings["realization"], table=table)
    defaults = _copy(selected.row.get("settings", {}))
    defaults["configuration_reference"] = _copy(selected.row.get("reference", {}))
    result = _merge(defaults, settings)
    parameters = selected.row.get("settings_parameters", ())
    if parameters or "settings_bindings" in result:
        bindings = result.get("settings_bindings", {})
        previous = settings.get("realization_identity", {})
        if ("settings_bindings_sha256" in previous
                and previous["settings_bindings_sha256"] != _digest(bindings)):
            # Expanded stage paths cannot be distinguished from explicit overrides.
            raise ValueError("settings bindings changed after expansion; configure fresh settings to avoid stale stage paths")
        # Bind runtime defaults after overlays; native references have their own
        # parameter namespace. Expanded settings are already single-pass bound.
        runtime = {key: result[key] for key in selected.row.get("settings", {})}
        bound = _bind({} if "settings_bindings_sha256" in previous else runtime,
                      parameters, bindings, "settings")
        result.update(bound)
        # Binding values are literal, including any ${...} text they contain.
        result["settings_bindings"] = _copy(bindings)
    if "reference_bindings" in result:
        # Rebind from the canonical row even when settings were expanded before.
        result["configuration_reference"] = _bind_reference(selected.row, result["reference_bindings"])
    result.setdefault("numerics", "fast")
    if result["numerics"] not in ("fast", "strict"):
        raise ValueError("NextGen numerics must be fast or strict")
    # A repeated setup call must have the same identity, not hash its old receipt.
    result.pop("realization_identity", None)
    result["realization_identity"] = {
        "id": selected.name, "row_sha256": selected.sha256,
        "settings_sha256": _digest(result),
        "sources": _copy(selected.row["sources"]),
        "admission": _copy(selected.row["admission"]),
    }
    if parameters or "settings_bindings" in result:
        result["realization_identity"]["settings_bindings_sha256"] = _digest(bindings)
    return result
