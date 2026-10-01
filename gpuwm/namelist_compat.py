"""Fail-closed RW-WPS compatibility analysis for WRF namelist pairs.

The normal gpuwm namelist importer answers whether gpuwm itself can integrate
an experiment.  RW-WPS has a different responsibility: it may prepare inputs
for an unchanged CPU WRF even when gpuwm cannot run the selected physics.
This module therefore reports stock-WRF export support and gpuwm forecast
support separately and never substitutes one physics scheme for another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
import math
from pathlib import Path
from typing import Iterable

from gpuwm.namelist_import import (
    NUDGING_NOT_IMPLEMENTED,
    THETA_M_ADMITTED,
    active_nudging_selectors,
    fine_input_stream_decision,
    read_namelist_role,
    theta_m_decision,
)
from gpuwm.vertical_contract import validate_explicit_eta_grid
from gpuwm.wrf_physics_inventory import stock_wrf_physics_inventory


SCHEMA = "rw-wps.namelist-support.v1"

#: WRF's own compiled ``max_domains`` (the Registry dimension every
#: per-domain rconfig array is declared against; the stock build ships
#: it at 21).  It belongs to the UNCHANGED WRF EXECUTABLE this export
#: writes for, not to this door: a namelist with more domains than the
#: compiled maximum cannot be read by that executable, which is the one
#: verdict this number bounds.  The geometry, physics and timing
#: analysis is bounded by :data:`ANALYSIS_MAX_DOMAINS` instead, so a
#: tree above the stock cap is still examined and the gpuwm_runtime
#: verdict is still answered from real reasons rather than returned as
#: an unexamined PASS with no reasons at all.
MAX_DOMAINS = 21

#: Structural bound on the report's own per-domain column expansion
#: (:func:`_column` materializes one entry per domain for every declared
#: key).  Nothing physical and no engine limit: it only stops an absurd
#: ``max_dom`` from expanding every column into a report nobody can
#: read.  Named in the message that cites it.
ANALYSIS_MAX_DOMAINS = 256

#: An issue that FAILS the report: the export cannot be written, or the
#: pair contradicts itself.  The default severity.
SEVERITY_BLOCKING = "blocking"

#: An issue that is REPORTED and does not fail the report: a route that
#: is prepared and run, carrying something the reader must be told about
#: it.  Without this channel every note in the report was a FAIL, so a
#: supported-with-a-caveat pair could not be stated at all.
SEVERITY_ADVISORY = "advisory"

_PREPROCESSING_KEYS = {
    "share": {
        "wrf_core", "max_dom", "start_date", "end_date", "interval_seconds",
    },
    "geogrid": {
        "parent_id", "parent_grid_ratio", "i_parent_start", "j_parent_start",
        "e_we", "e_sn", "s_we", "s_sn", "geog_data_res", "dx", "dy",
        "map_proj", "ref_lat", "ref_lon", "ref_x", "ref_y", "truelat1",
        "truelat2", "stand_lon", "pole_lat", "pole_lon", "geog_data_path",
        "opt_geogrid_tbl_path",
    },
    "time_control": {
        "run_days", "run_hours", "run_minutes", "run_seconds", "start_year",
        "start_month", "start_day", "start_hour", "start_minute",
        "start_second", "end_year", "end_month", "end_day", "end_hour",
        "end_minute", "end_second", "interval_seconds", "input_from_file",
        # Selects which input stream initializes each nest.  WRF
        # defines exactly two values: 0 takes every field from the
        # nest's own input, and 2 takes only the static and masked
        # land-surface fields from it and interpolates the rest from the
        # parent (the delayed-nest-start pattern).  Both have a prepared
        # route here; the gate below names which one and what differs.
        "fine_input_stream",
    },
    "domains": {
        "time_step", "time_step_fract_num", "time_step_fract_den", "max_dom",
        "e_we", "e_sn", "e_vert", "eta_levels", "p_top_requested", "dx",
        "dy", "grid_id", "parent_id", "i_parent_start", "j_parent_start",
        "parent_grid_ratio", "parent_time_step_ratio", "feedback",
        "smooth_option", "blend_width", "num_metgrid_levels",
        "num_metgrid_soil_levels", "interp_method_type", "nest_interp_coord",
        "vert_refine_method", "input_from_hires", "smooth_cg_topo",
        "use_adaptive_time_step", "sfcp_to_sfcp",
        # WRF declares hypsometric_opt in &domains, not &dynamics
        # (Registry.EM_COMMON:2283, `namelist,domains`).  Classified
        # under the wrong section, every real WRF namelist that sets it
        # -- which is where wrf.exe requires it -- came back
        # UNCLASSIFIED_NAMELIST_SETTING and failed the support report.
        "hypsometric_opt",
    },
    "dynamics": {
        "hybrid_opt", "etac", "base_temp",
    },
    "bdy_control": {
        "spec_bdy_width", "spec_zone", "relax_zone", "spec_exp", "specified",
        "nested",
    },
}

_PHYSICS_STATE_KEYS = {
    "physics": {
        "mp_physics", "ra_lw_physics", "ra_sw_physics", "sf_sfclay_physics",
        "sf_surface_physics", "bl_pbl_physics", "cu_physics",
        "num_soil_layers", "sf_urban_physics", "use_wudapt_lcz", "num_urban_hi",
        "slucm_distributed_drag", "distributed_ahe_opt", "sf_lake_physics", "mosaic_lu",
        "mosaic_soil", "mosaic_cat", "icloud", "morr_rimed_ice", "hail_opt",
        "ghg_input", "o3input", "aer_opt", "sst_update", "surface_input_source",
        "num_land_cat", "fractional_seaice",
        # Land-surface state selectors WRF-Runner-generated namelists carry
        # (2026-07-30 interop verification).  Each changes what real.exe
        # must have initialized, so they are state-relevant and value-gated
        # below: only the WRF-default value each certified export actually
        # ran with is accepted.
        "sf_surface_mosaic", "usemonalb", "rdlai2d",
        # Aerosol-aware Thompson (mp_physics=28) aerosol source selectors.
        # State-relevant, not runtime-output: use_aero_icbc is what makes
        # real.exe derive aer_init_opt and interpolate QNWFA/QNIFA from
        # metgrid (dyn_em/module_initialize_real.F:2325-2732), so it
        # changes what an export must have initialized.
        "use_aero_icbc",
    },
    # wif_input_opt lives in &domains, not &physics
    # (Registry/registry.new3d_wif:17 declares it namelist,domains), and it
    # is the single most state-relevant key an mp=28 namelist can carry: at
    # 1 or 2 it activates the use_wif_input packages that allocate the
    # 13-month WIF stack and, at 2, the qnbca scalar
    # (Registry/registry.new3d_wif:80-82).  Classified rather than left
    # unclassified so an mp=28 export is reported against the real rule
    # instead of the generic "RW-WPS has not proven" placeholder.
    "domains": {"wif_input_opt", "num_wif_levels"},
}

_RUNTIME_OUTPUT_KEYS = {
    "share": {"io_form_geogrid", "debug_level", "nocolons"},
    "ungrib": {"out_format", "prefix"},
    "metgrid": {"fg_name", "constants_name", "io_form_metgrid", "opt_metgrid_tbl_path"},
    "time_control": {
        "history_interval", "history_interval_s", "frames_per_outfile",
        "history_begin", "restart", "restart_interval", "io_form_history",
        "io_form_restart", "io_form_input", "io_form_boundary", "nocolons",
        "nwp_diagnostics", "debug_level",
        # CPU-WRF-runtime I/O keys: none changes what RW-WPS must
        # prepare.  io_form_auxinput2 only names the on-disk format of
        # the auxinput2 stream that fine_input_stream = 2 selects; the
        # stream itself is classified above and reported below.
        "io_form_auxinput2", "override_restart_timers",
        "iofields_filename", "ignore_iofields_warning",
    },
    "physics": {
        "radt", "bldt", "cudt", "isfflx", "ifsnow", "do_radar_ref",
        "swrad_scat",
    },
    "dynamics": {
        "w_damping", "epssm", "diff_opt", "km_opt", "mix_full_fields",
        "diff_6th_opt", "diff_6th_factor", "diff_6th_slopeopt", "damp_opt",
        "zdamp", "dampcoef", "khdif", "kvdif", "non_hydrostatic",
        "use_theta_m", "moist_adv_opt", "scalar_adv_opt", "time_step_sound",
        "smdiv", "emdiv", "h_sca_adv_order", "top_lid", "zadvect_implicit",
        "w_crit_cfl",
    },
    "fdda": set(),
    "grib2": set(),
    "namelist_quilt": set(),
}

#: Domain TILING keys, which are not moving-nest keys at all -- they
#: shared this set only because one report once refused both.  The
#: tiling half is a note now, so the two halves are split apart below.
_MOVING_NEST_TILING_KEYS = frozenset({"tile_sz_x", "tile_sz_y"})

#: The vortex-following half of WRF's moving-nest keys, the partition the
#: engine's own refusal draws (gpuwm/experiment.py:1545-1549): these steer
#: a nest from a storm tracker whose controls are corral/max-speed shaped
#: and have no counterpart at all.  Everything else in the loader's set is
#: a SPECIFIED move -- a schedule of whole-parent-cell shifts, which is
#: exactly what [[relocation.move]] rows express.
_VORTEX_FOLLOWING_KEYS = frozenset({
    "vortex_interval", "max_vortex_speed", "corral_dist", "track_level",
})


def _noah_soil_layer_count() -> int:
    """Noah's soil layer count, from Noah.

    One number, owned by the scheme that defines it
    (``gpuwm.core.noah.NUM_SOIL_LAYERS``, the only output of WRF's
    init_soil_depth_2), rather than a literal 4 in a report that predates
    RUC and Noah-MP being admitted at all.  Resolved on call for the
    reason :func:`_moving_nest_keys` gives.
    """

    from gpuwm.core.noah import NUM_SOIL_LAYERS
    return int(NUM_SOIL_LAYERS)


def _runtime_microphysics_reasons(mp) -> list[str]:
    """Per-domain reasons this ENGINE cannot run the requested scheme.

    Derived from the registry's implemented microphysics options rather
    than a literal set: the literal this replaced lacked 0, 1 and 9 and
    answered "gpuwm runtime does not implement" for three selectors the
    loader admits and the dispatcher runs (0 being the no-microphysics
    option).  An empty list means the engine runs every domain's
    scheme, whatever the stock-WRF export can or cannot write.
    """

    from gpuwm.physics_registry import implemented_selector_values

    implemented = implemented_selector_values("microphysics")
    return [
        f"d{index + 1:02d}: gpuwm runtime does not implement "
        f"mp_physics={value}; the stock-WRF export inventory is a "
        "separate question."
        for index, value in enumerate(mp) if value not in implemented
    ]


def _moving_nest_keys() -> frozenset[str]:
    """WRF moving-nest keys, from the loader that refuses them.

    A second hand-written copy of one concept is the drift family this
    audit exists for, and this copy had already drifted: it omitted
    vortex_interval and max_vortex_speed -- the two whose controls have
    no gpuwm equivalent at all -- so this report stayed silent about
    exactly the pair a migrating user is most likely to have written.

    Resolved on call rather than imported at module scope: this module
    stages in the standalone preprocessing distribution and
    gpuwm.experiment reaches the forecast side.
    """

    from gpuwm.experiment import _MOVING_NEST_KEYS as loader_keys
    return frozenset(loader_keys) | _MOVING_NEST_TILING_KEYS


@dataclass(frozen=True)
class CompatibilityIssue:
    """One report finding.

    ``severity`` is what makes a finding sayable without failing the
    report: :data:`SEVERITY_BLOCKING` entries (the default, so every
    existing rule keeps its meaning) decide the verdict, and
    :data:`SEVERITY_ADVISORY` entries state a supported route's caveat.
    """

    code: str
    location: str
    message: str
    action: str
    severity: str = SEVERITY_BLOCKING


def _entry(section: str, key: str, values: Iterable[object], source: str) -> dict:
    return {
        "source": source,
        "section": section,
        "key": key,
        "values": list(values),
    }


def _value(entries: dict[str, list], key: str, default=None):
    values = entries.get(key)
    if not values:
        return default
    return values[0]


def _column(
    entries: dict[str, list], key: str, count: int, *, default=None,
) -> list:
    values = entries.get(key)
    if not values:
        if default is None:
            raise KeyError(key)
        return [default] * count
    if len(values) > count:
        raise ValueError(
            f"declares {len(values)} values but max_dom={count}; extra domain "
            "values are rejected rather than truncated"
        )
    return list(values) + [values[-1]] * (count - len(values))


def _integers(values: Iterable[object], location: str) -> list[int]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{location} must contain integers, got {value!r}")
        result.append(value)
    return result


def _finite_floats(values: Iterable[object], location: str) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{location} must contain numbers, got {value!r}")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{location} must contain finite values")
        result.append(converted)
    return result


def _uniform(values: Iterable[object], location: str):
    values = list(values)
    if not values:
        raise ValueError(f"{location} has no values")
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{location} must be uniform across domains, got {values}")
    return values[0]


def _datetime_columns(
        entries: dict[str, list], prefix: str, count: int,
) -> list[datetime]:
    parts = {
        "year": _integers(
            _column(entries, f"{prefix}_year", count),
            f"&time_control/{prefix}_year"),
        "month": _integers(
            _column(entries, f"{prefix}_month", count),
            f"&time_control/{prefix}_month"),
        "day": _integers(
            _column(entries, f"{prefix}_day", count),
            f"&time_control/{prefix}_day"),
        "hour": _integers(
            _column(entries, f"{prefix}_hour", count, default=0),
            f"&time_control/{prefix}_hour"),
        "minute": _integers(
            _column(entries, f"{prefix}_minute", count, default=0),
            f"&time_control/{prefix}_minute"),
        "second": _integers(
            _column(entries, f"{prefix}_second", count, default=0),
            f"&time_control/{prefix}_second"),
    }
    return [
        datetime(
            parts["year"][index], parts["month"][index],
            parts["day"][index], parts["hour"][index],
            parts["minute"][index], parts["second"][index])
        for index in range(count)
    ]


def _wps_datetime_columns(values: list[object], count: int, location: str
                          ) -> list[datetime]:
    if len(values) > count:
        raise ValueError(
            f"{location} declares {len(values)} values but max_dom={count}")
    expanded = list(values) + [values[-1]] * (count - len(values))
    result = []
    for value in expanded:
        if not isinstance(value, str):
            raise ValueError(f"{location} must contain WRF date strings")
        try:
            result.append(datetime.strptime(value, "%Y-%m-%d_%H:%M:%S"))
        except ValueError as error:
            raise ValueError(
                f"{location} contains invalid WRF date {value!r}") from error
    return result


def _quoted_list(names: Iterable[str]) -> str:
    """``'a', 'b', or 'c'`` -- one rendering of one declared set."""

    names = [f"{name!r}" for name in names]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", or {names[-1]}"


def _issue(code: str, location: str, message: str, action: str,
           severity: str = SEVERITY_BLOCKING):
    return CompatibilityIssue(code, location, message, action, severity)


def _engine_admitted_values(switch: str) -> tuple[int, ...]:
    """Read the declarations used by the actual experiment validators.

    Import on call because this module also ships in the standalone
    preprocessing distribution. No partial configuration is instantiated.
    """
    from gpuwm.experiment import FEEDBACK_OPTIONS, SMOOTH_OPTION_OPTIONS

    return {"feedback": FEEDBACK_OPTIONS,
            "smooth_option": SMOOTH_OPTION_OPTIONS}[switch]


def _moving_nest_refusal(keys: Iterable[str]) -> str:
    """The engine's own refusal sentence for these keys, from the engine.

    The loader at :func:`gpuwm.experiment._reject_moving_nest_keys` owns
    what a moving-nest key breaks and what to write instead; this door
    used to carry a degraded copy ("Use static nests or add a mapped
    moving-domain implementation"), so the two doors described one
    namelist differently.  The location the engine renders is
    TOML-shaped ("[domains] of ..."), which is left as it is: making it
    namelist-shaped means a keyword on that out-of-lane function
    (gpuwm/experiment.py:1535).
    """

    from gpuwm.experiment import _reject_moving_nest_keys

    try:
        _reject_moving_nest_keys(
            "domains", {key: [] for key in keys}, "the WRF namelist pair")
    except ValueError as error:
        return str(error)
    return ""


def _relocation_itinerary_action(domains: dict[str, list]) -> str:
    """The exact ``[relocation]`` rows that reproduce a specified itinerary.

    WRF's specified moves are a schedule of whole-parent-cell shifts, and
    so is :class:`gpuwm.experiment.ScheduledRelocationMove` (at_seconds,
    di_parent_cells, dj_parent_cells -- whole parent cells, same sign
    convention).  ``move_interval`` is in minutes, which is the only unit
    conversion in the paste.
    """

    grid_id: object = None
    rows: list[str] = []
    try:
        count = int(_value(domains, "num_moves", 0))
        ids = [int(value) for value in domains.get("move_id", [])]
        minutes = [int(value) for value in domains.get("move_interval", [])]
        shift_i = [int(value) for value in domains.get("move_cd_x", [])]
        shift_j = [int(value) for value in domains.get("move_cd_y", [])]
        if ids and all(value == ids[0] for value in ids):
            grid_id = ids[0]
        for index in range(count):
            rows.append(
                f"{{at_seconds = {minutes[index] * 60}, "
                f"di_parent_cells = {shift_i[index]}, "
                f"dj_parent_cells = {shift_j[index]}}}")
    except (IndexError, KeyError, TypeError, ValueError):
        grid_id, rows = None, []
    named = "the moving nest's grid id" if grid_id is None else str(grid_id)
    itinerary = (
        ", ".join(rows) if rows else
        "{at_seconds = <move_interval minutes x 60>, "
        "di_parent_cells = <move_cd_x>, dj_parent_cells = <move_cd_y>}")
    return (
        "Write the same itinerary in the shipped [relocation] table: "
        f"enabled = true, grid_id = {named}, and one [[relocation.move]] "
        f"row per move -- {itinerary} (move_interval is minutes, "
        "at_seconds is seconds; di_parent_cells/dj_parent_cells are "
        "move_cd_x/move_cd_y, whole parent cells, same sign convention). "
        "One difference you accept by doing this: ArWen relocates at "
        "cycle boundaries, WRF on the model clock.")


def analyze_namelists(
    wps_path: str | Path,
    input_path: str | Path,
    *,
    source_top_pressure_pa: float | None = None,
) -> dict[str, object]:
    """Return a deterministic machine-readable compatibility report.

    The function always returns a report for syntactically readable
    namelists.  Unsupported science produces ``verdict=FAIL`` and actionable
    issue objects; caller code decides whether to print or raise.
    """

    wps_path, input_path = Path(wps_path), Path(input_path)
    # "Syntactically readable" above is a promise about CONTENT; a file
    # that is not there is a usage mistake, and it reaches the caller as
    # the same one-sentence ValueError `gpuwm import-namelist` raises
    # rather than as a traceback out of pathlib.
    wps = read_namelist_role(wps_path, "namelist.wps")
    inp = read_namelist_role(input_path, "namelist.input")
    issues: list[CompatibilityIssue] = []
    projection: dict[str, object] | None = None
    # Resolved once: the classification loop and the gate below must
    # agree about which keys these are, and the loader owns the list.
    # Without this, a namelist carrying the six specified-move keys got
    # one named refusal plus five UNCLASSIFIED_NAMELIST_SETTING shrugs
    # about settings this door knows perfectly well.
    moving_keys = _moving_nest_keys()
    classifications = {
        "preprocessing_relevant": [],
        "physics_state_relevant": [],
        "runtime_output_only": [],
        "legacy_stage_only": [],
    }

    for source, parsed in (("namelist.wps", wps), ("namelist.input", inp)):
        for section, entries in parsed.items():
            for key, values in entries.items():
                target = None
                if key in _PREPROCESSING_KEYS.get(section, set()) or (
                        section in ("domains", "geogrid")
                        and key in moving_keys):
                    target = "preprocessing_relevant"
                elif key in _PHYSICS_STATE_KEYS.get(section, set()):
                    target = "physics_state_relevant"
                elif section in {"ungrib", "metgrid"}:
                    target = "legacy_stage_only"
                elif section in {"fdda", "grib2", "namelist_quilt"}:
                    target = "runtime_output_only"
                elif key in _RUNTIME_OUTPUT_KEYS.get(section, set()):
                    target = "runtime_output_only"
                if target is None:
                    issues.append(_issue(
                        "UNCLASSIFIED_NAMELIST_SETTING",
                        f"&{section}/{key}",
                        "RW-WPS has not proven whether this setting changes "
                        "preprocessing or required initialized state.",
                        "Add a declarative compatibility rule for this exact WRF "
                        "setting; it will not be ignored or substituted.",
                    ))
                else:
                    classifications[target].append(
                        _entry(section, key, values, source)
                    )

    for entries in classifications.values():
        entries.sort(key=lambda item: (item["source"], item["section"], item["key"]))

    try:
        share = wps["share"]
        geogrid = wps["geogrid"]
        domains = inp["domains"]
        physics = inp["physics"]
    except KeyError as error:
        issues.append(_issue(
            "MISSING_REQUIRED_SECTION",
            f"&{error.args[0]}",
            "The paired WRF/WPS namelists lack a required section.",
            "Provide &share and &geogrid in namelist.wps plus &domains and "
            "&physics in namelist.input.",
        ))
        return _finish_report(
            max_dom=None,
            classifications=classifications,
            projection=projection,
            domains=[],
            vertical=None,
            stock_domains=[],
            gpuwm_reasons=[],
            issues=issues,
        )

    max_dom_raw = _value(domains, "max_dom", 1)
    if isinstance(max_dom_raw, bool) or not isinstance(max_dom_raw, int):
        issues.append(_issue(
            "INVALID_MAX_DOM", "&domains/max_dom",
            f"max_dom must be an integer, got {max_dom_raw!r}.",
            f"Choose an integer from 1 through {MAX_DOMAINS}, WRF's "
            "compiled max_domains.",
        ))
        max_dom = 0
    else:
        max_dom = max_dom_raw
    if not 1 <= max_dom <= MAX_DOMAINS:
        issues.append(_issue(
            "UNSUPPORTED_MAX_DOM", "&domains/max_dom",
            f"max_dom={max_dom} is outside WRF's compiled max_domains = "
            f"{MAX_DOMAINS}: an unchanged WRF executable built at that "
            "maximum cannot read a namelist declaring more domains, "
            "whatever this export writes for them.",
            f"Reduce the tree to {MAX_DOMAINS} domains or fewer, or "
            "rebuild WRF with a larger max_domains. The rest of this "
            "report is computed either way.",
        ))
    if max_dom > ANALYSIS_MAX_DOMAINS:
        issues.append(_issue(
            "UNSUPPORTED_MAX_DOM", "&domains/max_dom",
            f"max_dom={max_dom} is past this report's own structural "
            f"bound of {ANALYSIS_MAX_DOMAINS} domains, which bounds the "
            "per-domain column expansion the geometry, physics and "
            "timing blocks build.",
            f"Declare at most {ANALYSIS_MAX_DOMAINS} domains to get the "
            "geometry, vertical, physics and timing blocks.",
        ))
    wps_max_dom = _value(share, "max_dom", max_dom)
    if wps_max_dom != max_dom:
        issues.append(_issue(
            "MAX_DOM_MISMATCH", "&share/max_dom + &domains/max_dom",
            f"namelist.wps declares {wps_max_dom!r}, namelist.input declares "
            f"{max_dom!r}.",
            "Make both namelists describe the same ordered hierarchy.",
        ))

    raw_projection = _value(geogrid, "map_proj")
    # The implemented set is DECLARED once, by the module that implements
    # it (gpuwm.static.projection), and read here on call -- the same
    # on-call idiom :func:`_moving_nest_keys` uses, and for the same
    # reason.  The literal tuple this replaced was the third hand-typed
    # copy of one set, so adding a projection meant remembering to edit a
    # door that is not the projection module.
    from gpuwm.static.projection import (
        implemented_projections, latlon_blocker,
    )
    implemented = implemented_projections()
    implemented_list = _quoted_list(implemented)
    if not isinstance(raw_projection, str):
        issues.append(_issue(
            "INVALID_PROJECTION", "&geogrid/map_proj",
            f"map_proj must be a WPS string ({implemented_list}), got "
            f"{raw_projection!r}.",
            "Set map_proj to one of the implemented WPS projection "
            "strings.",
        ))
    else:
        normalized_projection = raw_projection.strip().lower().replace("_", "-")
        if normalized_projection not in implemented:
            issues.append(_issue(
                "UNSUPPORTED_PROJECTION", "&geogrid/map_proj",
                f"map_proj={raw_projection!r}; RW-WPS implements Lambert "
                "conformal, Mercator, and polar stereographic geometry.",
                f"Use map_proj={implemented_list}. "
                "Latitude-longitude is rejected rather than "
                f"approximated because {latlon_blocker()}.",
            ))
        else:
            # Mercator ignores truelat2/stand_lon and polar
            # stereographic ignores truelat2 (module_llxy semantics);
            # WPS namelists may omit exactly those keys.
            required_projection = (
                "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon",
                "dx", "dy",
            )
            if normalized_projection == "mercator":
                required = ("ref_lat", "ref_lon", "truelat1", "dx", "dy")
            elif normalized_projection == "polar":
                required = ("ref_lat", "ref_lon", "truelat1", "stand_lon",
                            "dx", "dy")
            else:
                required = required_projection
            missing_projection = [
                key for key in required if _value(geogrid, key) is None
            ]
            if missing_projection:
                issues.append(_issue(
                    "INVALID_PROJECTION", "&geogrid",
                    f"{normalized_projection} geometry is missing "
                    f"{missing_projection}.",
                    f"Declare {', '.join(required)} explicitly in "
                    "namelist.wps.",
                ))
            else:
                try:
                    values = {
                        key: float(_value(geogrid, key))
                        for key in required
                    }
                    if "truelat2" not in values:
                        fallback = _value(geogrid, "truelat2")
                        values["truelat2"] = float(
                            values["truelat1"] if fallback is None
                            else fallback)
                    if "stand_lon" not in values:
                        fallback = _value(geogrid, "stand_lon")
                        values["stand_lon"] = float(
                            values["ref_lon"] if fallback is None
                            else fallback)
                    if not all(math.isfinite(value) for value in values.values()):
                        raise ValueError("projection parameters must be finite")
                    if values["dx"] <= 0.0 or values["dy"] <= 0.0:
                        raise ValueError("projection dx and dy must be positive")
                    if not all(
                        -90.0 <= values[key] <= 90.0
                        for key in ("ref_lat", "truelat1", "truelat2")
                    ):
                        raise ValueError(
                            "ref_lat/truelat1/truelat2 must be within [-90, 90]"
                        )
                    projection = {
                        "map_proj": normalized_projection,
                        **{key: values[key] for key in required_projection},
                    }
                except (TypeError, ValueError) as error:
                    issues.append(_issue(
                        "INVALID_PROJECTION", "&geogrid",
                        str(error),
                        "Provide finite projection parameters, positive "
                        "dx/dy, and valid latitude values.",
                    ))

    domain_rows: list[dict[str, object]] = []
    stock_rows: list[dict[str, object]] = []
    gpuwm_reasons: list[str] = []
    dynamics = inp.get("dynamics", {})
    try:
        theta_values = _integers(
            _column(dynamics, "use_theta_m", max_dom, default=1),
            "&dynamics/use_theta_m",
        )
        declaration = (
            "is omitted, so WRF Registry default 1 applies"
            if "use_theta_m" not in dynamics
            else f"resolves to {theta_values!r}"
        )
        undefined_theta = [
            index + 1 for index, value in enumerate(theta_values)
            if value not in THETA_M_ADMITTED
        ]
        moist_theta = [
            index + 1 for index, value in enumerate(theta_values)
            if value == 1
        ]
        if undefined_theta:
            gpuwm_reasons.append(
                f"&dynamics/use_theta_m {declaration}; WRF defines "
                f"{THETA_M_ADMITTED} and domains {undefined_theta} declare "
                "something else. Set explicit &dynamics use_theta_m = 0."
            )
        elif moist_theta:
            # The SAME answer gpuwm.namelist_import gives for the same
            # namelist, from the same function: a declared divergence the
            # reader is told about, not a runtime FAIL.  This door used to
            # FAIL the gpuwm_runtime verdict for a pair the importer
            # accepts and announces -- one configuration, two answers.
            decision = theta_m_decision(1)
            issues.append(_issue(
                "THETA_M_DRY_SUBSTITUTION", "&dynamics/use_theta_m",
                f"&dynamics/use_theta_m {declaration}; domains "
                f"{moist_theta} select WRF's moist-theta prognostic. "
                + decision.reason,
                "Nothing to change: the import books this as a declared "
                "divergence and announces it. Set explicit &dynamics "
                "use_theta_m = 0 to integrate what WRF would integrate.",
                severity=SEVERITY_ADVISORY,
            ))
    except (KeyError, ValueError) as error:
        gpuwm_reasons.append(
            "gpuwm forecast runtime cannot classify "
            f"&dynamics/use_theta_m: {error}. Declare the WRF integer "
            "use_theta_m = 0 explicitly. Stock-WRF input export remains "
            "independent and supported."
        )
    try:
        mix_values = _column(
            dynamics, "mix_full_fields", max_dom, default=False)
        if any(not isinstance(value, bool) for value in mix_values):
            raise ValueError(
                "&dynamics/mix_full_fields must contain Fortran logicals, "
                f"got {mix_values!r}")
        unsupported_mix = [
            index + 1 for index, value in enumerate(mix_values)
            if value is not True
        ]
        if unsupported_mix:
            declaration = (
                "is omitted, so WRF Registry default false applies"
                if "mix_full_fields" not in dynamics
                else f"resolves to {mix_values!r}"
            )
            gpuwm_reasons.append(
                "gpuwm forecast runtime implements only full-field "
                f"diff_opt=2 mixing: &dynamics/mix_full_fields {declaration}; "
                f"unsupported on domains {unsupported_mix}. Set explicit "
                "mix_full_fields = .true. on every domain. Stock-WRF input "
                "export remains independent and supported."
            )
    except (KeyError, ValueError) as error:
        gpuwm_reasons.append(
            "gpuwm forecast runtime cannot classify "
            f"&dynamics/mix_full_fields: {error}. Declare "
            "mix_full_fields = .true. explicitly on every domain. Stock-WRF "
            "input export remains independent and supported."
        )
    try:
        smooth_values = _integers(
            _column(domains, "smooth_option", max_dom, default=2),
            "&domains/smooth_option",
        )
        admitted_smooth = _engine_admitted_values("smooth_option")
        admitted_smooth_list = ", ".join(
            str(value) for value in admitted_smooth)
        declaration = (
            "is omitted, so WRF Registry default 2 applies"
            if "smooth_option" not in domains
            else f"resolves to {smooth_values!r}"
        )
        undefined_smooth = [
            index + 1 for index, value in enumerate(smooth_values)
            if value not in admitted_smooth
        ]
        active_smooth = [
            index + 1 for index, value in enumerate(smooth_values)
            if value != 0 and value in admitted_smooth
        ]
        if undefined_smooth:
            gpuwm_reasons.append(
                f"&domains/smooth_option {declaration}; domains "
                f"{undefined_smooth} declare a smoother gpuwm.experiment "
                f"does not admit ({admitted_smooth_list}). Set explicit "
                "smooth_option = 0 (none), 1 (sm121) or 2 (smdsm)."
            )
        elif active_smooth:
            # The post-feedback parent smoother is IMPLEMENTED
            # (gpuwm/core/nest.py builds sm121 and smdsm and applies it
            # in feedback_commit), and WRF reads the key only when
            # feedback = 1, so a nonzero value is a statement about the
            # run rather than a runtime FAIL.  The literal 0 this
            # replaced failed every namelist that omits the key, which
            # takes WRF's Registry default 2.
            issues.append(_issue(
                "PARENT_SMOOTHER_ACTIVE", "&domains/smooth_option",
                f"&domains/smooth_option {declaration}; domains "
                f"{active_smooth} select WRF's post-feedback parent "
                "smoother, which gpuwm implements (0 none, 1 sm121, "
                "2 smdsm) and applies only while feedback = 1, exactly "
                "as WRF's own smoother returns before dispatching when "
                "feedback is off.",
                "Nothing to change. Set smooth_option = 0 to leave the "
                "parent unsmoothed after a two-way exchange.",
                severity=SEVERITY_ADVISORY,
            ))
    except (KeyError, ValueError) as error:
        gpuwm_reasons.append(
            "gpuwm forecast runtime cannot classify "
            f"&domains/smooth_option: {error}. Declare the WRF integer "
            "smooth_option = 0, 1 or 2 explicitly. Stock-WRF input export "
            "remains independent and supported."
        )
    vertical = None
    # Bounded by the REPORT's structure, not by the stock-export cap: a
    # tree above WRF's compiled max_domains still gets its geometry,
    # physics and timing examined, so the gpuwm_runtime verdict below is
    # answered from what the namelist says instead of coming back PASS
    # with no reasons over a hierarchy nothing looked at.
    if 1 <= max_dom <= ANALYSIS_MAX_DOMAINS:
        required_columns = (
            (geogrid, "e_we", "&geogrid/e_we"),
            (geogrid, "e_sn", "&geogrid/e_sn"),
            (geogrid, "parent_id", "&geogrid/parent_id"),
            (geogrid, "parent_grid_ratio", "&geogrid/parent_grid_ratio"),
            (geogrid, "i_parent_start", "&geogrid/i_parent_start"),
            (geogrid, "j_parent_start", "&geogrid/j_parent_start"),
            (domains, "e_we", "&domains/e_we"),
            (domains, "e_sn", "&domains/e_sn"),
            (domains, "e_vert", "&domains/e_vert"),
            (domains, "dx", "&domains/dx"),
            (domains, "dy", "&domains/dy"),
            (domains, "parent_id", "&domains/parent_id"),
            (domains, "parent_grid_ratio", "&domains/parent_grid_ratio"),
            (domains, "parent_time_step_ratio", "&domains/parent_time_step_ratio"),
            (domains, "i_parent_start", "&domains/i_parent_start"),
            (domains, "j_parent_start", "&domains/j_parent_start"),
        )
        cols: dict[tuple[int, str], list] = {}
        for entries, key, location in required_columns:
            try:
                cols[(id(entries), key)] = _column(entries, key, max_dom)
            except (KeyError, ValueError) as error:
                issues.append(_issue(
                    "INVALID_DOMAIN_ARRAY", location, str(error),
                    f"Declare 1..{max_dom} ordered values; WRF short-array "
                    "last-value replication is accepted.",
                ))

        def col(entries, key, default=None):
            existing = cols.get((id(entries), key))
            if existing is not None:
                return existing
            return _column(entries, key, max_dom, default=default)

        try:
            grid_ids = _integers(
                _column(domains, "grid_id", max_dom, default=1),
                "&domains/grid_id",
            )
            if "grid_id" not in domains:
                grid_ids = list(range(1, max_dom + 1))
            expected_ids = list(range(1, max_dom + 1))
            if grid_ids != expected_ids:
                raise ValueError(
                    f"must preserve contiguous parent-before-child order "
                    f"{expected_ids}, got {grid_ids}"
                )
            inp_parent = _integers(col(domains, "parent_id"), "&domains/parent_id")
            inp_ratio = _integers(
                col(domains, "parent_grid_ratio"), "&domains/parent_grid_ratio"
            )
            inp_tratio = _integers(
                col(domains, "parent_time_step_ratio"),
                "&domains/parent_time_step_ratio",
            )
            inp_i = _integers(col(domains, "i_parent_start"), "&domains/i_parent_start")
            inp_j = _integers(col(domains, "j_parent_start"), "&domains/j_parent_start")
            inp_we = _integers(col(domains, "e_we"), "&domains/e_we")
            inp_sn = _integers(col(domains, "e_sn"), "&domains/e_sn")
            inp_dx = _finite_floats(col(domains, "dx"), "&domains/dx")
            inp_dy = _finite_floats(col(domains, "dy"), "&domains/dy")
            wps_parent = _integers(col(geogrid, "parent_id"), "&geogrid/parent_id")
            wps_ratio = _integers(
                col(geogrid, "parent_grid_ratio"), "&geogrid/parent_grid_ratio"
            )
            wps_i = _integers(col(geogrid, "i_parent_start"), "&geogrid/i_parent_start")
            wps_j = _integers(col(geogrid, "j_parent_start"), "&geogrid/j_parent_start")
            wps_we = _integers(col(geogrid, "e_we"), "&geogrid/e_we")
            wps_sn = _integers(col(geogrid, "e_sn"), "&geogrid/e_sn")
            for key, left, right in (
                ("parent_id", wps_parent[1:], inp_parent[1:]),
                ("parent_grid_ratio", wps_ratio, inp_ratio),
                ("i_parent_start", wps_i, inp_i),
                ("j_parent_start", wps_j, inp_j),
                ("e_we", wps_we, inp_we),
                ("e_sn", wps_sn, inp_sn),
            ):
                if left != right:
                    raise ValueError(
                        f"ordered WPS/input {key} arrays differ: {left} != {right}"
                    )
            if inp_parent[0] != 0 or wps_parent[0] != 1:
                raise ValueError("d01 parent_id must be 0 in WRF and 1 in WPS")
            if any(value < 2 for value in (*inp_we, *inp_sn)):
                raise ValueError("every e_we/e_sn dimension must be at least 2")
            if (
                inp_ratio[0] != 1
                or inp_tratio[0] != 1
                or wps_ratio[0] != 1
                or inp_i[0] != 1
                or inp_j[0] != 1
                or wps_i[0] != 1
                or wps_j[0] != 1
            ):
                raise ValueError(
                    "d01 must use parent_grid_ratio=1, "
                    "parent_time_step_ratio=1, and i/j_parent_start=1"
                )
            seen = {1}
            for index in range(1, max_dom):
                if inp_parent[index] not in seen:
                    raise ValueError(
                        f"d{index + 1:02d} parent_id={inp_parent[index]} does "
                        "not name an earlier domain"
                    )
                seen.add(index + 1)
                if inp_ratio[index] < 1 or inp_tratio[index] < 1:
                    raise ValueError("child grid/time ratios must be positive")
                ratio = inp_ratio[index]
                if (inp_we[index] - 1) % ratio or (inp_sn[index] - 1) % ratio:
                    raise ValueError(
                        f"d{index + 1:02d} e_we/e_sn minus one must be "
                        f"divisible by parent_grid_ratio={ratio}"
                    )
                parent_index = inp_parent[index] - 1
                covered_i = (inp_we[index] - 1) // ratio
                covered_j = (inp_sn[index] - 1) // ratio
                if inp_i[index] < 1 or (
                    inp_i[index] + covered_i > inp_we[parent_index]
                ):
                    raise ValueError(
                        f"d{index + 1:02d} i_parent_start={inp_i[index]} "
                        f"and width {covered_i} do not fit "
                        f"d{parent_index + 1:02d} e_we={inp_we[parent_index]}"
                    )
                if inp_j[index] < 1 or (
                    inp_j[index] + covered_j > inp_sn[parent_index]
                ):
                    raise ValueError(
                        f"d{index + 1:02d} j_parent_start={inp_j[index]} "
                        f"and height {covered_j} do not fit "
                        f"d{parent_index + 1:02d} e_sn={inp_sn[parent_index]}"
                    )
            dx_root, dy_root = _finite_floats(
                (_value(geogrid, "dx"), _value(geogrid, "dy")),
                "&geogrid/dx,dy",
            )
            if dx_root <= 0.0 or dy_root <= 0.0:
                raise ValueError("root dx and dy must be positive")
            dx_values, dy_values = [dx_root], [dy_root]
            for index in range(1, max_dom):
                parent_index = inp_parent[index] - 1
                dx_values.append(dx_values[parent_index] / inp_ratio[index])
                dy_values.append(dy_values[parent_index] / inp_ratio[index])
            for index, (declared_dx, declared_dy, expected_dx, expected_dy) in enumerate(
                zip(inp_dx, inp_dy, dx_values, dy_values, strict=True), start=1
            ):
                if not math.isclose(
                    declared_dx, expected_dx, rel_tol=1.0e-9, abs_tol=1.0e-6
                ) or not math.isclose(
                    declared_dy, expected_dy, rel_tol=1.0e-9, abs_tol=1.0e-6
                ):
                    raise ValueError(
                        f"d{index:02d} declared dx/dy="
                        f"{declared_dx}/{declared_dy} m, expected "
                        f"{expected_dx}/{expected_dy} m from its parent ratio"
                    )
            for index in range(max_dom):
                domain_rows.append({
                    "grid_id": grid_ids[index],
                    "parent_id": inp_parent[index],
                    "parent_grid_ratio": inp_ratio[index],
                    "parent_time_step_ratio": inp_tratio[index],
                    "i_parent_start": inp_i[index],
                    "j_parent_start": inp_j[index],
                    "e_we": inp_we[index],
                    "e_sn": inp_sn[index],
                    "mass_nx": inp_we[index] - 1,
                    "mass_ny": inp_sn[index] - 1,
                    "dx_m": dx_values[index],
                    "dy_m": dy_values[index],
                })
        except (KeyError, TypeError, ValueError) as error:
            issues.append(_issue(
                "INVALID_DOMAIN_HIERARCHY", "&geogrid + &domains",
                str(error),
                "Use consistent ordered static one-way WRF/WPS domain arrays.",
            ))

        moving = sorted((set(domains) | set(geogrid)) & moving_keys)
        tiling = [key for key in moving if key in _MOVING_NEST_TILING_KEYS]
        engine_keys = [key for key in moving if key not in tiling]
        vortex = [key for key in engine_keys if key in _VORTEX_FOLLOWING_KEYS]
        specified = [key for key in engine_keys if key not in vortex]
        if specified:
            issues.append(_issue(
                "MOVING_NEST_UNSUPPORTED", "&domains/&geogrid",
                _moving_nest_refusal(specified),
                _relocation_itinerary_action(domains),
            ))
        if vortex:
            issues.append(_issue(
                "MOVING_NEST_UNSUPPORTED", "&domains/&geogrid",
                _moving_nest_refusal(vortex),
                "The vortex-following controls have no counterpart: "
                "ArWen's tracker is field/threshold/cooldown shaped, not "
                "corral/max-speed shaped, so there is nothing to write "
                "them as. Remove them, and express the motion either as "
                "a [[relocation.move]] itinerary or as "
                "[relocation.follow] with [relocation.track] on a field "
                "you name.",
            ))
        if tiling:
            # A note, not a refusal.  The action below says in words
            # that nothing breaks, and a finding whose own text names no
            # breakage cannot decide the verdict: these keys size the
            # CPU build's shared-memory tiles and reach neither the
            # prepared state nor the integration.  gpuwm.namelist_import
            # records them as dropped beside numtiles/nproc_x/nproc_y,
            # so both doors accept the same namelist.
            issues.append(_issue(
                "DOMAIN_TILING_IGNORED", "&domains",
                f"WRF tile-decomposition key(s) {tiling} are present: "
                "they size the CPU build's shared-memory tiles, and this "
                "product decomposes its own domains, so there is no value "
                "to carry them to.",
                "Nothing to change: the importer records them as dropped "
                "keys, and they change neither what is prepared nor what "
                "is integrated.",
                severity=SEVERITY_ADVISORY,
            ))
        feedback = _value(domains, "feedback", 1)
        admitted_feedback = _engine_admitted_values("feedback")
        admitted_list = ", ".join(str(value) for value in admitted_feedback)
        if feedback == 0 and not isinstance(feedback, bool):
            pass
        elif (not isinstance(feedback, bool) and isinstance(feedback, int)
                and feedback in admitted_feedback):
            # The warning the runtime prints for this tier is stated here
            # in this door's own words rather than imported: the string
            # lives at gpuwm/runtime.py:103, forecast-side, and this
            # module stages in the standalone preprocessing wheel.
            issues.append(_issue(
                "TWO_WAY_NESTING_EXPERIMENTAL", "&domains/feedback",
                f"feedback={feedback!r} selects the EXPERIMENTAL two-way "
                "child-to-parent path. gpuwm.experiment admits it "
                f"({admitted_list}) and the runtime stamps every such run "
                "as experimental; it is not certified against stock WRF "
                "yet.",
                "Nothing to change: this pair exports and runs. Set "
                "feedback = 0 for the certified one-way path.",
                severity=SEVERITY_ADVISORY,
            ))
        else:
            issues.append(_issue(
                "TWO_WAY_NESTING_UNSUPPORTED", "&domains/feedback",
                f"feedback={feedback!r} is not a value the engine admits: "
                f"gpuwm.experiment admits ({admitted_list}) -- 0 one-way, "
                "1 experimental two-way.",
                f"Set feedback to one of ({admitted_list}).",
            ))

        # fine_input_stream selects each nest's initialization input
        # stream.  The answer comes from the importer's own
        # fine_input_stream_decision -- the ONE function this door, the
        # importer and the WRF doors all read it from -- so a pair this
        # report blesses is a pair gpuwm.namelist_import accepts, and a
        # pair it refuses is refused here with the same sentence.
        try:
            # No _integers() here: the shared decision is the type gate
            # as well, so a column WRF's reader would not accept is
            # refused on both doors in one sentence.
            stream_decision = fine_input_stream_decision(
                _column(inp.get("time_control", {}), "fine_input_stream",
                        max_dom, default=0))
            if stream_decision.refusal is not None:
                issues.append(_issue(
                    "NEST_INPUT_STREAM_UNSUPPORTED",
                    "&time_control/fine_input_stream",
                    stream_decision.refusal,
                    stream_decision.way_out,
                ))
            elif stream_decision.delayed_domains:
                issues.append(_issue(
                    "NEST_INPUT_STREAM_SUBSTITUTION",
                    "&time_control/fine_input_stream",
                    stream_decision.divergence,
                    stream_decision.way_out,
                    severity=SEVERITY_ADVISORY,
                ))
        except (KeyError, TypeError, ValueError) as error:
            issues.append(_issue(
                "NEST_INPUT_STREAM_UNSUPPORTED",
                "&time_control/fine_input_stream",
                str(error),
                "Declare integer fine_input_stream values in d01..dNN order.",
            ))

        # real.exe writes wrffdda_d0N when grid_fdda is active; RW-WPS
        # replaces real.exe and does not.  An active nudging request in a
        # namelist this report blesses would fail at WRF runtime on a
        # missing input file, so it must fail here instead.
        fdda = inp.get("fdda", {})
        try:
            grid_fdda = _integers(
                _column(fdda, "grid_fdda", max_dom, default=0),
                "&fdda/grid_fdda",
            )
        except (KeyError, TypeError, ValueError) as error:
            grid_fdda = None
            issues.append(_issue(
                "FDDA_INPUT_NOT_PRODUCED", "&fdda/grid_fdda",
                str(error),
                "Declare integer grid_fdda values in d01..dNN order, or "
                "remove &fdda.",
            ))
        # The RUNTIME verdict for the same &fdda block the importer
        # refuses, through the importer's own predicate and sentence.  The
        # issue below is the STOCK EXPORT's separate question (real.exe
        # writes wrffdda_d0N and RW-WPS does not); this door used to answer
        # that one only, so gpuwm_runtime came back PASS with no reasons for
        # a namelist gpuwm.namelist_import refuses outright.
        active_nudging = active_nudging_selectors(fdda)
        if active_nudging:
            named = ", ".join(
                f"&fdda/{key} = {values}" for key, values in active_nudging)
            gpuwm_reasons.append(f"{named}: {NUDGING_NOT_IMPLEMENTED}")
        if grid_fdda is not None and any(value != 0 for value in grid_fdda):
            issues.append(_issue(
                "FDDA_INPUT_NOT_PRODUCED", "&fdda/grid_fdda",
                f"grid_fdda={grid_fdda}; analysis nudging reads "
                "wrffdda_d0N, which real.exe generates and RW-WPS does not.",
                "Set grid_fdda=0 (remove &fdda), or keep WPS + real.exe for "
                "nudged runs.",
            ))

        bdy = inp.get("bdy_control", {})
        try:
            width = _integers(
                [_value(bdy, "spec_bdy_width", 5)],
                "&bdy_control/spec_bdy_width",
            )[0]
            spec_zone = _integers(
                [_value(bdy, "spec_zone", 1)], "&bdy_control/spec_zone",
            )[0]
            relax_zone = _integers(
                [_value(bdy, "relax_zone", 4)], "&bdy_control/relax_zone",
            )[0]
            if min(width, spec_zone, relax_zone) < 0 or width < 1:
                raise ValueError("specified-boundary widths must be non-negative")
            if width != spec_zone + relax_zone:
                raise ValueError(
                    f"spec_bdy_width={width} must equal spec_zone={spec_zone} "
                    f"+ relax_zone={relax_zone}"
                )
            specified = _column(bdy, "specified", max_dom, default=True)
            nested = _column(bdy, "nested", max_dom, default=False)
            if any(not isinstance(value, bool) for value in (*specified, *nested)):
                raise ValueError("specified/nested arrays must contain logicals")
            expected_specified = [True] + [False] * (max_dom - 1)
            expected_nested = [False] + [True] * (max_dom - 1)
            if specified != expected_specified or nested != expected_nested:
                raise ValueError(
                    "static one-way export requires specified=.true. only on "
                    "d01 and nested=.true. only on d02..dNN; got "
                    f"specified={specified}, nested={nested}"
                )
        except (KeyError, TypeError, ValueError) as error:
            issues.append(_issue(
                "INVALID_BOUNDARY_TOPOLOGY", "&bdy_control",
                str(error),
                "Use one specified root, nested children, and "
                "spec_bdy_width=spec_zone+relax_zone.",
            ))

        try:
            e_vert = _integers(
                _column(domains, "e_vert", max_dom), "&domains/e_vert"
            )
            shared_e_vert = int(_uniform(e_vert, "&domains/e_vert"))
            eta = [float(value) for value in domains["eta_levels"]]
            p_top_values = [
                float(value)
                for value in _column(domains, "p_top_requested", max_dom)
            ]
            p_top = float(_uniform(p_top_values, "&domains/p_top_requested"))
            hybrid_opt = int(_value(dynamics, "hybrid_opt", 2))
            etac = float(_value(dynamics, "etac", 0.2))
            checked_eta = validate_explicit_eta_grid(
                eta,
                nz=shared_e_vert - 1,
                p_top=p_top,
                source_top_pressure_pa=source_top_pressure_pa,
                context="RW-WPS namelist support",
            )
            if hybrid_opt != 2:
                raise ValueError(
                    f"hybrid_opt={hybrid_opt}; only WRF hybrid coordinate 2 is "
                    "implemented"
                )
            vertical = {
                "e_vert": shared_e_vert,
                "mass_levels": shared_e_vert - 1,
                "eta_levels": checked_eta.tolist(),
                "p_top_requested_pa": p_top,
                "hybrid_opt": hybrid_opt,
                "etac": etac,
                "source_top_pressure_pa": source_top_pressure_pa,
                "coverage": (
                    "verified" if source_top_pressure_pa is not None
                    else "deferred_to_source_mapping"
                ),
            }
        except (KeyError, TypeError, ValueError) as error:
            issues.append(_issue(
                "INVALID_VERTICAL_GRID", "&domains/eta_levels",
                str(error),
                "Provide one shared explicit strictly decreasing eta grid, "
                "e_vert=len(eta_levels), valid p_top, and hybrid_opt=2. The "
                "level count is arbitrary within source coverage.",
            ))

        try:
            mp = _integers(
                _column(physics, "mp_physics", max_dom),
                "&physics/mp_physics",
            )
            pbl = _integers(
                _column(physics, "bl_pbl_physics", max_dom),
                "&physics/bl_pbl_physics",
            )
            sfclay = _integers(
                _column(physics, "sf_sfclay_physics", max_dom),
                "&physics/sf_sfclay_physics",
            )
            surface = _integers(
                _column(physics, "sf_surface_physics", max_dom),
                "&physics/sf_surface_physics",
            )
            soil = _integers(
                _column(physics, "num_soil_layers", max_dom, default=4),
                "&physics/num_soil_layers",
            )
            urban = _integers(
                _column(physics, "sf_urban_physics", max_dom, default=0),
                "&physics/sf_urban_physics",
            )
            urban_lcz = _integers(
                _column(physics, "use_wudapt_lcz", max_dom, default=0),
                "&physics/use_wudapt_lcz")
            urban_bins = _integers(
                _column(physics, "num_urban_hi", max_dom, default=15),
                "&physics/num_urban_hi")
            from types import SimpleNamespace
            from gpuwm.config import validate_urban_config
            try:
                validate_urban_config(SimpleNamespace(
                    sf_urban_physics=urban[-1], use_wudapt_lcz=urban_lcz[-1],
                    num_urban_hi=urban_bins[-1], sf_surface_physics=surface[-1],
                    bl_pbl_physics=pbl[-1]))
            except ValueError as error:
                gpuwm_reasons.append(str(error))
            for key, default in (("slucm_distributed_drag", False),
                                 ("distributed_ahe_opt", 0)):
                if any(value != default for value in
                       _column(physics, key, max_dom, default=default)):
                    gpuwm_reasons.append(
                        f"{key}: the arm is not transcribed, the setting would be ignored")
            mosaic = _integers(
                _column(physics, "sf_surface_mosaic", max_dom, default=0),
                "&physics/sf_surface_mosaic",
            )
            # This report evaluates no radiation selector for any scheme.
            # It briefly read ra_lw/ra_sw for mp=50 alone, to mirror
            # gpuwm.config.validate_p3_radiation's refusal of P3 on the
            # RTE+RRTMGP pair; that refusal was retired with its defect
            # (rrtmgp._MP_CLOUD_OPTICS_SCHEME now carries ``50: "p3"``,
            # WRF's own has_reqs=0 coupling), so the mirror and its
            # radiation-column read retired with it.
            monalb = _column(physics, "usemonalb", max_dom, default=False)
            lai2d = _column(physics, "rdlai2d", max_dom, default=False)
            if any(not isinstance(value, bool)
                   for value in (*monalb, *lai2d)):
                raise ValueError(
                    "&physics/usemonalb and &physics/rdlai2d must contain "
                    "logicals")
            # THE RUNTIME QUESTION IS ASKED FIRST, and never inside the
            # stock-export inventory's reach.  It used to sit after
            # ``stock_wrf_physics_inventory(mp[index])`` in the loop body,
            # so for a scheme with no packaged WRF contract -- mp=9, the
            # scheme that triggered this audit -- the inventory raised on
            # the first domain, the runtime line was unreachable, and the
            # report told a migrating user with a runnable Milbrandt-Yau
            # namelist that ArWen's runtime did not implement it. The two
            # questions are independent: one asks what an UNCHANGED WRF
            # needs in its wrfinput, the other asks what this engine runs.
            gpuwm_reasons.extend(_runtime_microphysics_reasons(mp))
            from types import SimpleNamespace
            from gpuwm.config import validate_noah_mosaic_config
            tile_values = _column(physics, "mosaic_cat", max_dom, default=3)
            for index in range(max_dom):
                try:
                    validate_noah_mosaic_config(SimpleNamespace(
                        sf_surface_mosaic=mosaic[0], mosaic_cat=tile_values[0],
                        sf_surface_physics=surface[index], sf_urban_physics=urban[index]))
                except ValueError as exc:
                    gpuwm_reasons.append(str(exc))
            for index in range(max_dom):
                configuration_errors = []
                if pbl[index] != 1:
                    configuration_errors.append(
                        f"bl_pbl_physics={pbl[index]} (only YSU=1 inventoried)"
                    )
                if sfclay[index] != 91:
                    configuration_errors.append(
                        f"sf_sfclay_physics={sfclay[index]} "
                        "(only classic MM5=91 inventoried)"
                    )
                # Noah's layer count comes from Noah, not from a literal.
                # The "4" here was an independent copy of a number
                # gpuwm.core.noah owns, written before RUC and Noah-MP were
                # admitted anywhere; the SCOPE (this export inventories the
                # Noah package only) is a real statement about the export
                # and stays.
                if (surface[index] != 2
                        or soil[index] != _noah_soil_layer_count()):
                    configuration_errors.append(
                        f"sf_surface_physics={surface[index]}, "
                        f"num_soil_layers={soil[index]} (the stock-WRF "
                        "export inventories the Noah package only: "
                        f"sf_surface_physics=2 at "
                        f"{_noah_soil_layer_count()} soil layers)"
                    )
                if urban[-1] != 0:
                    configuration_errors.append(
                        f"sf_urban_physics={urban[-1]} (the importer passes the "
                        "innermost selector to runtime validation; stock-WRF "
                        "urban restart state is not inventoried)"
                    )
                if mosaic[index] != 0:
                    configuration_errors.append(
                        f"sf_surface_mosaic={mosaic[index]} (mosaic land "
                        "state not inventoried; the export initializes "
                        "dominant-category Noah state only)"
                    )
                if monalb[index]:
                    configuration_errors.append(
                        "usemonalb=.true. (monthly-albedo initialized state "
                        "is not evidenced for this export; only the WRF "
                        "default .false. is)"
                    )
                if lai2d[index]:
                    configuration_errors.append(
                        "rdlai2d=.true. (LAI-from-static initialized state "
                        "is not evidenced for this export; only the WRF "
                        "default .false. is)"
                    )
                if configuration_errors:
                    issues.append(_issue(
                        "UNSUPPORTED_PHYSICS_STATE",
                        f"d{index + 1:02d} &physics",
                        "; ".join(configuration_errors),
                        "Select an evidenced combination or implement its exact "
                        "WRF Registry/real.exe initialized-state adapter.",
                    ))
                try:
                    inventory = stock_wrf_physics_inventory(mp[index])
                except ValueError as error:
                    # EXPORT-ONLY.  No evidenced Registry.EM_COMMON package
                    # contract is packaged for this selector, so a
                    # wrfinput written for an UNCHANGED WRF would
                    # under-declare its own package.  It says nothing about
                    # running the scheme here, and the reason and the way
                    # out both say so (audit R-013 / R-014).
                    issues.append(_issue(
                        "STOCK_WRF_EXPORT_INVENTORY_MISSING",
                        f"d{index + 1:02d} &physics/mp_physics",
                        str(error),
                        "This is the stock-WRF export route only. To run "
                        "this configuration in ArWen, take it through "
                        "`gpuwm import-namelist` and `gpuwm run`; the "
                        "runtime verdict below is answered independently. "
                        "To export a wrfinput for an unchanged WRF "
                        "v4.6.1, add that scheme's exact Registry package "
                        "and real.exe initialized-state policy to "
                        "gpuwm/wrf_physics_inventory.py, or select an "
                        "inventoried scheme.",
                    ))
                    continue
                stock_rows.append({
                    "grid_id": index + 1,
                    "mp_physics": mp[index],
                    "microphysics": inventory.scheme,
                    "bl_pbl_physics": pbl[index],
                    "sf_sfclay_physics": sfclay[index],
                    "sf_surface_physics": surface[index],
                    "num_soil_layers": soil[index],
                    "wrfinput_fields": [
                        asdict(field) for field in inventory.wrfinput_fields
                    ],
                    "runtime_state_not_wrfinput": [
                        asdict(field)
                        for field in inventory.runtime_state_not_wrfinput
                    ],
                })
                # The per-domain RUNTIME verdict used to stand here, one
                # line below the stock inventory that had already raised
                # for any uninventoried scheme -- so it was unreachable for
                # exactly the schemes it needed to answer, and reachable
                # only to say "supported" for the ones the export already
                # covered.  It moved above this loop, where it is answered
                # for every domain whatever the export says (audit R-013).
                #
                # The mp=50 + ra 4/4 arm that stood here mirrored
                # gpuwm.config.validate_p3_radiation and retired with it:
                # rrtmgp._MP_CLOUD_OPTICS_SCHEME carries ``50: "p3"`` now
                # (WRF's own has_reqs=0 remap coupling), so a bare 4/4
                # namelist resolves to a pairing that runs and there is
                # no refusal left to mirror.
        except (KeyError, TypeError, ValueError) as error:
            # The physics columns themselves would not parse.  The
            # per-scheme export gap has its own issue
            # (STOCK_WRF_EXPORT_INVENTORY_MISSING, raised per domain
            # above) and no longer arrives here, so this arm names what it
            # actually caught.
            issues.append(_issue(
                "UNREADABLE_PHYSICS_COLUMNS", "&physics",
                str(error),
                "Give every &physics selector one value per domain "
                "(mp_physics, bl_pbl_physics, sf_sfclay_physics, "
                "sf_surface_physics, num_soil_layers, sf_urban_physics, "
                "sf_surface_mosaic, usemonalb, rdlai2d) with integer or "
                "logical values as WRF's Registry declares them.",
            ))

    timing = None
    if 1 <= max_dom <= ANALYSIS_MAX_DOMAINS:
        try:
            time_control = inp["time_control"]
            starts = _datetime_columns(time_control, "start", max_dom)
            ends = _datetime_columns(time_control, "end", max_dom)
            wps_starts = _wps_datetime_columns(
                share["start_date"], max_dom, "&share/start_date")
            wps_ends = _wps_datetime_columns(
                share["end_date"], max_dom, "&share/end_date")
            if starts != wps_starts:
                raise ValueError(
                    "ordered namelist.input start columns differ from "
                    f"namelist.wps start_date: "
                    f"{[value.isoformat() for value in starts]} != "
                    f"{[value.isoformat() for value in wps_starts]}")
            if ends != wps_ends:
                raise ValueError(
                    "ordered namelist.input end columns differ from "
                    "namelist.wps end_date")
            input_interval = _value(time_control, "interval_seconds")
            wps_interval = _value(share, "interval_seconds")
            if input_interval is not None and input_interval != wps_interval:
                raise ValueError(
                    "&time_control/interval_seconds differs from "
                    f"&share/interval_seconds: {input_interval!r} != "
                    f"{wps_interval!r}")
            interval = wps_interval if input_interval is None else input_interval
            if (isinstance(interval, bool) or not isinstance(interval, int)
                    or interval <= 0):
                raise ValueError(
                    "interval_seconds must be a positive integer number of "
                    f"seconds, got {interval!r}")

            time_step = _value(domains, "time_step")
            if (isinstance(time_step, bool)
                    or not isinstance(time_step, (int, float))
                    or not math.isfinite(float(time_step))):
                raise ValueError(
                    f"&domains/time_step must be finite, got {time_step!r}")
            root_dt = Fraction(str(time_step))
            fract_num = _value(domains, "time_step_fract_num", 0)
            fract_den = _value(domains, "time_step_fract_den", 1)
            if (isinstance(fract_num, bool) or not isinstance(fract_num, int)
                    or isinstance(fract_den, bool)
                    or not isinstance(fract_den, int) or fract_den <= 0):
                raise ValueError(
                    "time_step_fract_num/time_step_fract_den must be "
                    "integer with a positive denominator")
            root_dt += Fraction(fract_num, fract_den)
            if root_dt <= 0:
                raise ValueError("resolved root time_step must be positive")
            parents = _integers(
                _column(domains, "parent_id", max_dom),
                "&domains/parent_id")
            time_ratios = _integers(
                _column(domains, "parent_time_step_ratio", max_dom),
                "&domains/parent_time_step_ratio")
            dt_by_id = {1: root_dt}
            for index in range(1, max_dom):
                parent_id = parents[index]
                ratio = time_ratios[index]
                if parent_id not in dt_by_id or ratio <= 0:
                    raise ValueError(
                        f"d{index + 1:02d} has invalid parent/time-step ratio")
                dt_by_id[index + 1] = dt_by_id[parent_id] / ratio

            interval_fraction = Fraction(interval)
            root_steps = interval_fraction / root_dt
            timing_reasons = []
            if root_steps.denominator != 1:
                timing_reasons.append(
                    f"boundary_interval_seconds = {interval} s is not a "
                    "whole number of root-domain steps: d01 dt = "
                    f"{root_dt} s exactly, cadence/dt = {root_steps}.")
            root_start = starts[0]
            if ends[0] <= root_start:
                raise ValueError("d01 end time must be later than its start")
            if any(value != ends[0] for value in ends[1:]):
                timing_reasons.append(
                    "gpuwm runtime requires one shared experiment end time; "
                    f"end columns resolve to "
                    f"{[value.isoformat() for value in ends]}.")
            rows = []
            for index, start in enumerate(starts):
                grid_id = index + 1
                offset = int((start - root_start).total_seconds())
                parent_id = parents[index]
                parent_alignment = "ROOT"
                if grid_id == 1:
                    if offset != 0:
                        raise ValueError("d01 must define the root start time")
                else:
                    parent_start = starts[parent_id - 1]
                    parent_delta = Fraction(
                        int((start - parent_start).total_seconds()))
                    parent_steps = parent_delta / dt_by_id[parent_id]
                    parent_alignment = (
                        "PASS" if parent_steps.denominator == 1 else "FAIL")
                    if parent_delta < 0:
                        timing_reasons.append(
                            f"d{grid_id:02d} start {start.isoformat()} "
                            f"precedes parent d{parent_id:02d} start "
                            f"{parent_start.isoformat()}.")
                    elif parent_steps.denominator != 1:
                        timing_reasons.append(
                            f"delayed start for d{grid_id:02d} "
                            f"({start.isoformat()}) is not aligned to its "
                            f"parent step boundary: d{parent_id:02d} dt = "
                            f"{dt_by_id[parent_id]} s exactly, "
                            "(child-parent start offset)/dt = "
                            f"{parent_steps}.")
                    forcing_seams = Fraction(offset, interval)
                    if forcing_seams.denominator != 1:
                        timing_reasons.append(
                            f"delayed start for d{grid_id:02d} "
                            f"({start.isoformat()}; offset {offset} s) is "
                            "not aligned to the boundary-forcing cadence: "
                            f"interval_seconds = {interval} s, "
                            f"offset/cadence = {forcing_seams}.")
                rows.append({
                    "grid_id": grid_id,
                    "start_time": start.isoformat(),
                    "end_time": ends[index].isoformat(),
                    "offset_seconds": offset,
                    "dt_seconds_exact": str(dt_by_id[grid_id]),
                    "parent_step_alignment": parent_alignment,
                    "forcing_seam_alignment": (
                        "ROOT" if grid_id == 1 else
                        ("PASS" if Fraction(offset, interval).denominator == 1
                         else "FAIL")),
                })
            gpuwm_reasons.extend(timing_reasons)
            timing = {
                "boundary_interval_seconds": interval,
                "root_cadence_steps_exact": str(root_steps),
                "constraint": (
                    "positive uniform whole-second forcing cadence; exact "
                    "integer root steps; each delayed child start is an "
                    "exact parent-step boundary and forcing seam"),
                "domains": rows,
                "gpuwm_runtime_verdict": (
                    "PASS" if not timing_reasons else "FAIL"),
            }
            by_grid = {row["grid_id"]: row for row in rows}
            for row in domain_rows:
                row["start_time"] = by_grid[row["grid_id"]]["start_time"]
        except (KeyError, TypeError, ValueError) as error:
            issues.append(_issue(
                "INVALID_TIME_HIERARCHY",
                "&share + &time_control + &domains",
                str(error),
                "Provide matching real-shaped per-domain WPS/input start and "
                "end columns, a positive integer interval_seconds, and an "
                "exact positive root time_step.",
            ))

    return _finish_report(
        max_dom=max_dom,
        classifications=classifications,
        projection=projection,
        domains=domain_rows,
        vertical=vertical,
        timing=timing,
        stock_domains=stock_rows,
        gpuwm_reasons=gpuwm_reasons,
        issues=issues,
    )


def _finish_report(
    *, max_dom, classifications, projection, domains, vertical, stock_domains,
    gpuwm_reasons, issues, timing=None,
) -> dict[str, object]:
    issue_rows = [asdict(issue) for issue in issues]
    # BLOCKING issues decide the verdict.  This used to read
    # ``not issues``, which made every note in the report a FAIL and left
    # the report no way to say "supported, and here is what differs" --
    # so a supported route with a caveat had to be refused to be
    # mentioned at all.
    stock_pass = not any(
        issue.severity == SEVERITY_BLOCKING for issue in issues)
    return {
        "schema": SCHEMA,
        "verdict": "PASS" if stock_pass else "FAIL",
        "max_dom": max_dom,
        "classifications": classifications,
        "geometry": {
            "projection": projection,
            "domain_count": len(domains),
            "domains": domains,
        },
        "vertical": vertical,
        "timing": timing,
        "required_state": {
            "stock_wrf_export": {
                "verdict": "PASS" if stock_pass else "FAIL",
                "target": "unchanged WRF v4.6.1",
                "domains": stock_domains,
            },
            # The RUNTIME verdict is the runtime's own.  It used to read
            # ``stock_pass and not gpuwm_reasons``, so ANY stock-export
            # issue -- a scheme with no packaged WRF Registry package
            # contract, an uninventoried PBL, a mosaic land state --
            # forced the runtime verdict to FAIL with an EMPTY reason
            # list: a report that said the runtime could not run the
            # configuration and would not say why, for a configuration
            # ArWen runs (audit R-013).
            "gpuwm_runtime": {
                "verdict": "PASS" if not gpuwm_reasons else "FAIL",
                "reasons": gpuwm_reasons,
            },
        },
        "issues": issue_rows,
    }


def require_supported_namelists(*args, **kwargs) -> dict[str, object]:
    """Analyze and raise one actionable error if stock-WRF export fails."""

    report = analyze_namelists(*args, **kwargs)
    if report["verdict"] != "PASS":
        rendered = "; ".join(
            f"{item['code']} at {item['location']}: {item['message']} "
            f"Action: {item['action']}"
            for item in report["issues"]
        )
        raise ValueError(f"RW-WPS namelist compatibility failed: {rendered}")
    return report


__all__ = [
    "ANALYSIS_MAX_DOMAINS",
    "MAX_DOMAINS",
    "SCHEMA",
    "SEVERITY_ADVISORY",
    "SEVERITY_BLOCKING",
    "analyze_namelists",
    "require_supported_namelists",
]
