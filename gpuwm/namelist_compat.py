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

from gpuwm.namelist_import import parse_namelist
from gpuwm.vertical_contract import validate_explicit_eta_grid
from gpuwm.wrf_physics_inventory import stock_wrf_physics_inventory


SCHEMA = "rw-wps.namelist-support.v1"
MAX_DOMAINS = 21

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
    },
    "dynamics": {
        "hybrid_opt", "etac", "base_temp", "hypsometric_opt",
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
        "num_soil_layers", "sf_urban_physics", "sf_lake_physics", "mosaic_lu",
        "mosaic_soil", "mosaic_cat", "icloud", "morr_rimed_ice", "hail_opt",
        "ghg_input", "o3input", "aer_opt", "sst_update", "surface_input_source",
        "num_land_cat", "fractional_seaice",
    }
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
        "smdiv", "emdiv", "h_sca_adv_order", "top_lid",
    },
    "fdda": set(),
    "grib2": set(),
    "namelist_quilt": set(),
}

_MOVING_NEST_KEYS = {
    "time_to_move", "corral_dist", "track_level", "tile_sz_x", "tile_sz_y",
    "num_moves", "move_id", "move_interval", "move_cd_x", "move_cd_y",
}


@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    location: str
    message: str
    action: str


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


def _issue(code: str, location: str, message: str, action: str):
    return CompatibilityIssue(code, location, message, action)


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
    wps = parse_namelist(wps_path)
    inp = parse_namelist(input_path)
    issues: list[CompatibilityIssue] = []
    projection: dict[str, object] | None = None
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
                if key in _PREPROCESSING_KEYS.get(section, set()):
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
            f"Choose an integer from 1 through {MAX_DOMAINS}.",
        ))
        max_dom = 0
    else:
        max_dom = max_dom_raw
    if not 1 <= max_dom <= MAX_DOMAINS:
        issues.append(_issue(
            "UNSUPPORTED_MAX_DOM", "&domains/max_dom",
            f"max_dom={max_dom} is outside the compiled RW-WPS contract.",
            f"Choose 1..{MAX_DOMAINS}; six-domain hierarchies are explicitly "
            "covered and this is not a d01/d02-only parser.",
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
    _IMPLEMENTED_PROJECTIONS = ("lambert", "mercator", "polar")
    if not isinstance(raw_projection, str):
        issues.append(_issue(
            "INVALID_PROJECTION", "&geogrid/map_proj",
            "map_proj must be a WPS string ('lambert', 'mercator', or "
            f"'polar'), got {raw_projection!r}.",
            "Set map_proj to one of the implemented WPS projection "
            "strings.",
        ))
    else:
        normalized_projection = raw_projection.strip().lower().replace("_", "-")
        if normalized_projection not in _IMPLEMENTED_PROJECTIONS:
            issues.append(_issue(
                "UNSUPPORTED_PROJECTION", "&geogrid/map_proj",
                f"map_proj={raw_projection!r}; RW-WPS implements Lambert "
                "conformal, Mercator, and polar stereographic geometry.",
                "Use map_proj='lambert', 'mercator', or 'polar'. "
                "Latitude/longitude (and other) inputs are rejected "
                "rather than approximated.",
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
        unsupported_theta = [
            index + 1 for index, value in enumerate(theta_values)
            if value != 0
        ]
        if unsupported_theta:
            declaration = (
                "is omitted, so WRF Registry default 1 applies"
                if "use_theta_m" not in dynamics
                else f"resolves to {theta_values!r}"
            )
            gpuwm_reasons.append(
                "gpuwm forecast runtime requires the dry-theta branch: "
                f"&dynamics/use_theta_m {declaration}; unsupported on "
                f"domains {unsupported_theta}. Set explicit "
                "&dynamics use_theta_m = 0. Stock-WRF input export remains "
                "independent and supported."
            )
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
        unsupported_smooth = [
            index + 1 for index, value in enumerate(smooth_values)
            if value != 0
        ]
        if unsupported_smooth:
            declaration = (
                "is omitted, so WRF Registry default 2 applies"
                if "smooth_option" not in domains
                else f"resolves to {smooth_values!r}"
            )
            gpuwm_reasons.append(
                "gpuwm one-way forecast runtime requires the disabled parent "
                f"smoother: &domains/smooth_option {declaration}; unsupported "
                f"on domains {unsupported_smooth}. Set explicit "
                "smooth_option = 0. Stock-WRF input export remains independent "
                "and supported."
            )
    except (KeyError, ValueError) as error:
        gpuwm_reasons.append(
            "gpuwm forecast runtime cannot classify "
            f"&domains/smooth_option: {error}. Declare the WRF integer "
            "smooth_option = 0 explicitly. Stock-WRF input export remains "
            "independent and supported."
        )
    vertical = None
    if 1 <= max_dom <= MAX_DOMAINS:
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

        moving = sorted(
            (set(domains) | set(geogrid)) & _MOVING_NEST_KEYS
        )
        if moving:
            issues.append(_issue(
                "MOVING_NEST_UNSUPPORTED", "&domains/&geogrid",
                f"moving-nest controls are present: {moving}.",
                "Use static nests or add a mapped moving-domain implementation.",
            ))
        feedback = _value(domains, "feedback", 1)
        if feedback != 0:
            issues.append(_issue(
                "TWO_WAY_NESTING_UNSUPPORTED", "&domains/feedback",
                f"feedback={feedback!r}; current RW-WPS export is one-way only.",
                "Set feedback=0. RW-WPS does not silently change this value.",
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
            for index in range(max_dom):
                inventory = stock_wrf_physics_inventory(mp[index])
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
                if surface[index] != 2 or soil[index] != 4:
                    configuration_errors.append(
                        f"sf_surface_physics={surface[index]}, "
                        f"num_soil_layers={soil[index]} (only Noah=2/4 layers)"
                    )
                if urban[index] != 0:
                    configuration_errors.append(
                        f"sf_urban_physics={urban[index]} (urban state not inventoried)"
                    )
                if configuration_errors:
                    issues.append(_issue(
                        "UNSUPPORTED_PHYSICS_STATE",
                        f"d{index + 1:02d} &physics",
                        "; ".join(configuration_errors),
                        "Select an evidenced combination or implement its exact "
                        "WRF Registry/real.exe initialized-state adapter.",
                    ))
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
                # Thompson 8 is first-class since the classic tables became
                # package data (mp8 promotion, product/v1 packaging lane
                # 2026-07-28): no enable guard, packaged table root with an
                # environment override, byte validation still at setup.
                # NSSL 18 is selectable since the certified
                # campaign/real74-fixed-nssl merge (product/v1 NSSL lane
                # 2026-07-29); its registry maturity stays
                # "validation-candidate", which this report does not restate.
                runtime_supported = mp[index] in {6, 8, 10, 18}
                if not runtime_supported:
                    gpuwm_reasons.append(
                        f"d{index + 1:02d}: gpuwm runtime on paired head does not "
                        f"implement mp_physics={mp[index]} ({inventory.scheme}); "
                        "stock-WRF export inventory is independent and supported."
                    )
        except (KeyError, TypeError, ValueError) as error:
            issues.append(_issue(
                "UNSUPPORTED_MICROPHYSICS_INVENTORY", "&physics/mp_physics",
                str(error),
                "Use an inventoried scheme (6 WSM6, 8 Thompson, 10 Morrison, "
                "or 18 NSSL) or add its exact WRF Registry/real.exe state "
                "contract.",
            ))

    timing = None
    if 1 <= max_dom <= MAX_DOMAINS:
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
    stock_pass = not issues
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
            "gpuwm_runtime": {
                "verdict": "PASS" if stock_pass and not gpuwm_reasons else "FAIL",
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
    "MAX_DOMAINS",
    "SCHEMA",
    "analyze_namelists",
    "require_supported_namelists",
]
