"""Static defaults declared by source metadata, with no source-specific route."""
from __future__ import annotations

import copy
import json


def source_static_defaults(source_id):
    """The table a source declares, or no table for an undeclared source."""
    from gpuwm.source_adapters import get_source_adapter
    selected = get_source_adapter(source_id).static_source
    return {} if selected is None else {"source": selected}


#: What a static receipt says when a source the configuration did not name
#: was set aside for its projection.  Ruled 2026-10-05 (2.8.6 gate, round 3).
DEFAULTED_FALLBACK_REASON = "defaulted source did not match; WPS_GEOG used"


def defaulted_source_fallback(row_id, projection):
    """The record of a DEFAULTED static source set aside, or ``None``.

    A row the source metadata selects (the configuration names none) is
    used only on its own cone.  ``projection`` (a grid, or the
    configuration's ``[projection]`` table) on another cone keeps the
    WPS_GEOG build it always had, because taking the row would refuse a
    configuration that never asked for it, and moving the configuration
    onto the row's cone moves the case itself (up to 147 km for the
    shipped 250 m LES nests).  A DECLARED source never comes here: it
    still refuses a mismatch by name in
    :func:`gpuwm.static.external_source.crop_window`.
    """
    from .external_source import projection_mismatch, static_source_row
    row = static_source_row(row_id)
    mismatch = projection_mismatch(row, projection)
    if not mismatch:
        return None
    return {"id": row.id, "selected_by": "source metadata default",
            "used": "WPS_GEOG", "reason": DEFAULTED_FALLBACK_REASON,
            "projection_mismatch": {key: list(pair)
                                    for key, pair in sorted(mismatch.items())}}


def with_source_static_defaults(raw, source_id):
    """Apply the metadata default without replacing an explicit selection.

    A configuration whose ``[projection]`` is not the row's cone keeps
    what it always built (:func:`defaulted_source_fallback`).
    """
    defaults = source_static_defaults(source_id)
    if not defaults:
        return raw
    table = raw.get("static")
    if table is not None and not isinstance(table, dict):
        return raw  # The schema owner gives the malformed table its refusal.
    if table is not None and "source" in table:
        return raw
    from collections.abc import Mapping
    projection = raw.get("projection")
    if (isinstance(projection, Mapping)
            and defaulted_source_fallback(defaults["source"], projection)):
        return raw
    result = copy.deepcopy(raw)
    result["static"] = {**defaults, **(table or {})}
    return result


def with_declared_source_static_defaults(raw):
    """Read a configuration's acquisition source when it declares one."""
    from collections.abc import Mapping
    if not isinstance(raw, Mapping):
        return raw
    fetch = raw.get("fetch")
    if not isinstance(fetch, Mapping) or not isinstance(fetch.get("source"), str):
        return raw
    from gpuwm.source_adapters import get_source_adapter
    try:
        adapter = get_source_adapter(fetch["source"])
    except (KeyError, ValueError):
        return raw  # Prepared-cache identities are checked by their own loader.
    return with_source_static_defaults(raw, adapter.source_id)


def with_source_static_defaults_text(text, source_id):
    """Emit a declared default; every undeclared source returns the same bytes."""
    defaults = source_static_defaults(source_id)
    if not defaults:
        return text
    import tomllib
    raw = tomllib.loads(text)
    table = raw.get("static")
    if table is not None and "source" in table:
        return text
    # Config emitters currently write no [static] table. A source-only table
    # is appended without changing any physics, domain or fetch setting.
    if table is not None:
        raise ValueError("the configuration emitter already writes [static]; "
                         "merge the source default through its table owner")
    return text + "\n[static]\nsource = " + json.dumps(defaults["source"]) + "\n"


def align_source_projection(projection, dims, spacing_m, source_id):
    """Plan exact-source mass points when a metadata row serves this spacing.

    The centre moves to the nearest whole-cell source window; dimensions
    stay fixed. Other sources and other spacings retain the same object.
    """
    defaults = source_static_defaults(source_id)
    if not defaults:
        return projection
    from .external_source import static_source_row, source_grid, crop_window
    row = static_source_row(defaults["source"])
    declared = row.grid_map
    if (abs(float(spacing_m) - float(declared["dx"])) > 1e-6
            or abs(float(spacing_m) - float(declared["dy"])) > 1e-6):
        return projection
    source = source_grid(row)
    nx, ny = dims
    x, y = source.latlon_to_ij(projection["ref_lat"], projection["ref_lon"])
    i0 = round(float(x) - (nx + 1) / 2.0)
    j0 = round(float(y) - (ny + 1) / 2.0)
    lat, lon = source.ij_to_latlon(i0 + (nx + 1) / 2.0,
                                 j0 + (ny + 1) / 2.0)
    aligned = {**projection, **{key: declared[key] for key in
               ("map_proj", "truelat1", "truelat2", "stand_lon")},
               "ref_lat": float(lat), "ref_lon": float(lon)}
    from .projection import projection_class
    target = projection_class(aligned["map_proj"])(
        aligned["ref_lat"], aligned["ref_lon"], aligned["truelat1"],
        aligned["truelat2"], aligned["stand_lon"], spacing_m, spacing_m,
        nx + 1, ny + 1)
    crop_window(row, target)
    return aligned
