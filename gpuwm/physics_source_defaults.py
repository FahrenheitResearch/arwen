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

#: Recipe settings only some microphysics schemes read, with those schemes.
#: The operational fork's Thompson generation exists for the aerosol-aware
#: scheme (mp_physics = 28) alone, and gpuwm.config refuses
#: thompson_version = "wrf_39_noaa" under any other scheme, so filling it
#: there would make every non-mp28 suite on an hrrr recipe refuse its own
#: emission.  A caller that does not know the scheme gets none of these.
#: The fork's melting-snow fall is read by that generation alone and is
#: scoped exactly like it.
MP_SCOPED_RECIPE_SETTINGS = {"thompson_version": (28,),
                             "thompson_fork_snow_fall": (28,)}


def land_scoped_defaults(defaults: dict, sf_surface_physics,
                         mp_physics=None) -> dict:
    """``defaults`` without the settings the selected land and microphysics
    schemes never read."""
    mp = None if mp_physics is None else int(mp_physics)
    return {key: value for key, value in defaults.items()
            if (key not in LAND_SCOPED_RECIPE_SETTINGS
                or sf_surface_physics in LAND_SCOPED_RECIPE_SETTINGS[key])
            and (key not in MP_SCOPED_RECIPE_SETTINGS
                 or mp in MP_SCOPED_RECIPE_SETTINGS[key])}


#: The scheme generations a loaded configuration takes from its declared
#: ``[fetch] source`` when it omits them.  These pick WHICH surface layer,
#: WHICH Thompson code and WHICH snow fall run; every other row of the
#: request table (clock, diffusion, zero-out, albedo) stays an
#: authoring-time default,
#: because filling a numeric namelist setting or the clock into an
#: explicitly authored configuration at load would change runs the
#: table was never asked about.  Origin: the WOOF-HRRR door's carried
#: experiment.toml (authored before mynn_sfclay_variant and
#: thompson_version existed, fetch source rap-native) ran WRF v4.6.1's
#: MYNN surface layer and Thompson against HRRR's own analysis, and no
#: door said so.  Found while tracing the Plains 2025-03-14 crop's
#: 0.11-0.26 K 2 m temperature gap to HRRR at f03-f06
#: (WOOF-FIX-PROGRAM-2026-10-06, plains-t2-bias); the generations were
#: not that gap's cause (together they moved T2 RMSE by at most 0.04 K
#: on the box; the gap was the crop's boundary cycle age and 3 h
#: boundary cadence).  An explicit value, including ``"wrf_461"``, is
#: kept.
#: The fork's melting-snow fall rides WITH the fork Thompson: the two
#: keys are one physics (the operational model integrates both), and the
#: generation with the v4.6.1 snow fall kept three quarters of the Iowa
#: 2024-05-21 first-hour gap to HRRR that the pair closes (f01 FSS35 at
#: 27 km 0.30 -> 0.34 -> 0.42, HRRR 0.42; WOOF-FIX-PROGRAM-2026-10-06,
#: woof-hour1-spinup).  It is filled only where the fork generation
#: resolves, written or filled here, because gpuwm.config refuses it
#: under ``wrf_461``.
GENERATION_SELECTORS = ("mynn_sfclay_variant", "thompson_version",
                        "thompson_fork_snow_fall")


def _resolved_per_domain(key, shared, domain_tables, chosen) -> set:
    """The value of ``key`` on every domain: the domain's own, else
    ``[shared]``'s, else what this fill chose.  A tree with no domain
    table resolves as ``[shared]`` alone.  Seeding the set with the
    ``[shared]`` value would read a tree that writes the key on every
    ``[[domain]]`` and nowhere else as {None, value} and silently skip
    the fill."""
    fallback = shared.get(key, chosen.get(key))
    tables = [table for table in (domain_tables or ()) if isinstance(table, dict)]
    return ({table.get(key, fallback) for table in tables} if tables
            else {fallback})


def omitted_generation_selectors(shared: dict, domain_tables, source) -> dict:
    """The generation selectors ``source`` declares that ``shared`` omits.

    ``thompson_version`` is returned only when every domain resolves
    ``mp_physics = 28`` (gpuwm.config refuses the fork name under any other
    scheme), and ``thompson_fork_snow_fall`` only when, in addition, every
    domain resolves ``thompson_version = "wrf_39_noaa"`` (written, or
    filled here; gpuwm.config refuses the fork snow fall under
    ``wrf_461``).  Unknown or absent sources, and sources with no table
    row, give an empty dict; the caller writes nothing then.
    """
    if not isinstance(shared, dict) or not isinstance(source, str) or not source:
        return {}
    try:
        declared = recipe_physics_defaults(source)
    except (KeyError, ValueError):
        return {}
    chosen = {key: value for key, value in declared.items()
              if key in GENERATION_SELECTORS and key not in shared}
    if "thompson_version" in chosen or "thompson_fork_snow_fall" in chosen:
        if _resolved_per_domain("mp_physics", shared, domain_tables, {}) != {28}:
            chosen.pop("thompson_version", None)
            chosen.pop("thompson_fork_snow_fall", None)
    if "thompson_fork_snow_fall" in chosen:
        if _resolved_per_domain("thompson_version", shared, domain_tables,
                                chosen) != {"wrf_39_noaa"}:
            chosen.pop("thompson_fork_snow_fall")
    return chosen


def fill_omitted_generation_selectors(raw, fetch_table) -> dict:
    """Write the selectors ``[fetch] source`` declares into ``raw["shared"]``
    when the configuration omits them, in place; returns what was written.

    Every loader that builds the experiment a run integrates, records in
    a receipt or holds a preparation to calls this where it splits the
    ``[fetch]`` table off, so one file resolves to one physics through
    every door: the config-table loader behind ``gpuwm sim``/``check``
    (gpuwm.experiment.build_experiment_from_config_tables), the case
    loader behind ``gpuwm run`` (gpuwm.case_data.load_experiment_case_bytes)
    and the native root preparation behind ``gpuwm prep`` and the
    single-domain benchmark
    (gpuwm.hrrr_configuration.resolve_root_experiment) used to differ,
    the first filling and the other two not; the preparation's physics
    receipt then recorded the generic generations for a bare
    operational-fork configuration whose forecast ran the fork's.
    Written keys are kept.  The one splitter that does not call it is
    the materializer's validation view
    (gpuwm.prepared_single_domain_forecast._experiment_tables), which
    decides profile conflicts and digests and never reaches the engine;
    its view resolves the generic defaults for these keys.
    """
    if not isinstance(raw, dict) or not isinstance(fetch_table, dict):
        return {}
    shared = raw.get("shared")
    if not isinstance(shared, dict):
        return {}
    chosen = omitted_generation_selectors(
        shared, raw.get("domain"), fetch_table.get("source"))
    shared.update(chosen)
    return chosen


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
