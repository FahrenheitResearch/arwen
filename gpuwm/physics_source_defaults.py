"""Explicit scheme-generation defaults at named configuration doors."""
from __future__ import annotations

import json
from fnmatch import fnmatchcase
from pathlib import Path
import re
import tomllib

REQUEST_DEFAULTS = Path(__file__).with_name("data") / "physics_sources" / "request-defaults.v1.toml"

# Only runtime selectors with no stock-WRF namelist spelling belong
# here. A comment carries their explicit values without adding a WRF key.
PHYSICS_SELECTOR_VALUES = {
    "thompson_version": ("wrf_461", "wrf_39_noaa"),
    "thompson_fork_snow_fall": ("blend", "wrf_39_noaa"),
    "mynn_sfclay_variant": ("wrf_461", "gsl_wrf39"),
    "terrain_clock": ("measured", "pinned"),
    "diff_6th_form": ("wrf_461", "noaa_wrf39"),
    "upper_wind_limiter_form": ("wrf_461", "noaa_wrf39"),
    "rrtmg_cloud_optics_form": ("wrf_461", "noaa_wrf39"),
    "bl_mynn_version": ("wrf_461", "gsd_41"),
    "bl_mynn_cloud_tendency_form": ("wrf_461", "gsd_41"),

    # ruc_soilprop has no stock-WRF key either. Without an entry here an
    # explicit wrf_461 had no carrier and the route read back the recipe's
    # wrf_45.
    "ruc_soilprop": ("wrf_45", "wrf_461"),
    "ruc_irrigation": ("wrf_461", "wrf_45"),
    "ruc_snow": ("wrf_461", "wrf_45"),
    "ruc_qvg_cold_start": ("wrf", "air"),
    "ruc_2m_diagnostic": ("flux", "log_profile"),
}
_SELECTOR_MARKER = "! gpuwm-physics-selectors-v1:"


def _validate_physics_selectors(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("physics selector comment must carry a JSON object")
    for key, value in settings.items():
        if key not in PHYSICS_SELECTOR_VALUES:
            raise ValueError(f"unknown physics selector comment key {key!r}")
        if type(value) is not str or value not in PHYSICS_SELECTOR_VALUES[key]:
            raise ValueError(f"invalid physics selector comment value for {key}: {value!r}")
    return settings


def read_physics_selector_comment(text: str) -> dict:
    """Read one strictly typed generation comment; ordinary WRF stays empty."""
    markers = [line.strip() for line in text.splitlines()
               if line.strip().startswith("! gpuwm-physics-selectors")]
    if not markers:
        return {}
    if len(markers) != 1:
        raise ValueError("duplicate physics selector comments")
    if not markers[0].startswith(_SELECTOR_MARKER):
        raise ValueError("invalid physics selector comment marker")

    def unique_keys(pairs):
        values = {}
        for key, value in pairs:
            if key in values:
                raise ValueError(f"duplicate physics selector comment key {key!r}")
            values[key] = value
        return values

    try:
        settings = json.loads(markers[0][len(_SELECTOR_MARKER):],
                              object_pairs_hook=unique_keys)
    except json.JSONDecodeError as error:
        raise ValueError("invalid JSON in physics selector comment") from error
    if not settings:
        raise ValueError("physics selector comment must carry at least one selector")
    return _validate_physics_selectors(settings)


def with_physics_selector_comment(text: str, settings: dict) -> str:
    """Carry explicit generation values, retaining unmarked default bytes."""
    _validate_physics_selectors(settings)
    if not settings:
        return text
    carried = read_physics_selector_comment(text)
    if carried:
        if carried != settings:
            raise ValueError("conflicting physics selector comment values")
        return text
    return (_SELECTOR_MARKER + " "
            + json.dumps(settings, sort_keys=True, separators=(",", ":"))
            + "\n" + text)


def _request_defaults(selector: str, value: str | None, *, scope="defaults") -> dict:
    if value is None:
        return {}
    document = tomllib.loads(REQUEST_DEFAULTS.read_text(encoding="utf-8"))
    if document.get("schema") != "gpuwm-physics-request-defaults-v1":
        raise ValueError("unknown physics request-default schema")
    selected = {}
    for row in document.get("request", ()):
        if not any(fnmatchcase(value.lower(), pattern)
                   for pattern in row.get(selector, ())):
            continue
        for key, setting in row.get(scope, {}).items():
            if key in selected and selected[key] != setting:
                raise ValueError(f"conflicting physics request default for {key}")
            selected[key] = setting
    return selected


def namelist_physics_defaults(path) -> dict:
    """A named source namelist declares its generation in the table."""
    return _request_defaults("namelist_names", Path(path).name)


def recipe_physics_defaults(source: str | None) -> dict:
    """An authored recipe declares its generation; ordinary loads do not."""
    if source is None:
        return {}
    from gpuwm.source_adapters import get_source_adapter
    return _request_defaults("recipe_sources", get_source_adapter(source).source_id)


#: Recipe settings only some land-surface schemes read, with those schemes.
#: The prescribed-monthly request is read by Noah (2) and RUC (3) alone,
#: and gpuwm.config refuses it under any other scheme, so filling it
#: there made every Noah-MP suite on an hrrr recipe refuse its own
#: emission ("usemonalb/rdlai2d are implemented by the Noah LSM ...").
LAND_SCOPED_RECIPE_SETTINGS = {"rdlai2d": (2, 3), "usemonalb": (2, 3)}


def land_scoped_defaults(defaults: dict, sf_surface_physics) -> dict:
    """``defaults`` without the settings the selected land scheme never reads."""
    return {key: value for key, value in defaults.items()
            if key not in LAND_SCOPED_RECIPE_SETTINGS
            or sf_surface_physics in LAND_SCOPED_RECIPE_SETTINGS[key]}


def recipe_root_defaults(source: str | None) -> dict:
    """Source-authored root settings; they never become shared nest settings."""
    if source is None:
        return {}
    from gpuwm.source_adapters import get_source_adapter
    return _request_defaults("recipe_sources", get_source_adapter(source).source_id,
                             scope="root_defaults")


def with_recipe_root_defaults(shared: dict, domains: list[dict], defaults: dict):
    """Fill only missing root keys, preserving every explicitly stated value."""
    if not defaults:
        return domains
    roots = [domain for domain in domains if domain.get("parent_id") == 0]
    if len(roots) != 1:
        raise ValueError("source recipe root defaults require exactly one root domain")
    for key, value in defaults.items():
        if key not in shared:
            roots[0].setdefault(key, value)
    return domains


def with_physics_defaults_text(text: str, defaults: dict) -> str:
    """Emit missing shared keys while preserving explicit settings and bytes."""
    if not defaults:
        return text
    shared = tomllib.loads(text).get("shared", {})
    missing = {key: value for key, value in defaults.items() if key not in shared}
    if not missing:
        return text
    match = re.search(r'^\["?shared"?\]\r?\n', text, re.MULTILINE)
    if match is None:
        raise ValueError("physics request defaults require a shared config table")
    newline = "\r\n" if match.group().endswith("\r\n") else "\n"
    block = "".join(f"{key} = {json.dumps(value)}{newline}"
                    for key, value in missing.items())
    return text[:match.end()] + block + text[match.end():]
