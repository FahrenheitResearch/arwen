"""Create an ordinary ERA5 forcing configuration and its WPS companion."""
from __future__ import annotations
import contextlib
import copy
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import tomllib

REQUEST_SCHEMA = "arwen.companion-forcing-edit.v1"
RESULT_SCHEMA = "arwen.companion-forcing-result.v1"
SCHEDULE_REQUEST_SCHEMA = "arwen.companion-schedule-edit.v1"
SCHEDULE_RESULT_SCHEMA = "arwen.companion-schedule-result.v1"


def _duration(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("run_seconds must be a finite positive number")
    return float(value)


def capabilities():
    return {"schema": "arwen.companion-forcing-capabilities.v1", "candidate_only": True,
        "optional_run_seconds": True, "schedule_request_schema": SCHEDULE_REQUEST_SCHEMA,
        "providers": [{"id": "cds", "label": "Copernicus CDS", "requires_credentials": True,
            "products": [{"id": "reanalysis", "cadence_hours": [1, 3, 6], "members": None},
                         {"id": "ensemble_members", "cadence_hours": [3], "members": list(range(10))}]}],
        "validation": {"eda_member_payload": "real CDS GRIB1 fixtures; all ten identities and byte-preserving selection verified",
            "full_eda_forcing_acquisition": "passed on a real 3x3 CDS grid, 37 levels, full surface fields, and two UTC times",
            "preparation": "not_qualified", "forecast": "not_run"},
        "limits": ["ARCO quickmap success does not qualify full forcing; this editor exposes CDS only.",
            "EDA requires an existing experiment initialization at an exact 3-hourly UTC time.",
            "Acquisition is verified on a small real fixture; preparation, including the existing water-temperature preparation issue, remains unqualified."]}


def edit_configuration(request):
    from gpuwm import fetch
    from gpuwm.case_data import resolved_case_data_paths
    from gpuwm.companion_domains import _exact_keys, _build, _wps_text, _json, native_domain_outlines, domain_geojson
    from gpuwm.config_authority import read_config_authority
    from gpuwm.era5_member import validate_selection
    from gpuwm.experiment import experiment_config_document, validate_boundary_timing
    from gpuwm.runplan import _fetch_arguments_from_hints, _validate_fetch_arguments
    from gpuwm.starter_template import changes, _publish_new_files
    from gpuwm.toml_document import emit_experiment_toml

    required = {"schema", "config_path", "expected_sha256", "output_path", "product_type", "cadence_hours", "member", "provider"}
    _exact_keys(request, required | {"run_seconds"}, required, where="forcing edit request")
    if request["schema"] != REQUEST_SCHEMA:
        raise ValueError("Unsupported forcing edit request schema")
    if request["provider"] not in {p["id"] for p in capabilities()["providers"]}:
        raise ValueError("This forcing editor currently exposes the CDS provider only; ARCO full forcing is not qualified here")
    authority = read_config_authority(request["config_path"])
    if request["expected_sha256"] != authority.sha256:
        raise ValueError("The selected configuration changed; refresh it before editing forcing")
    output = Path(request["output_path"]).expanduser().resolve()
    wps_output, receipt_output = output.with_suffix(".namelist.wps"), output.with_suffix(".forcing.json")
    if len({output, wps_output, receipt_output}) != 3 or output == authority.source or any(os.path.lexists(p) for p in (output, wps_output, receipt_output)):
        raise FileExistsError("Choose a new candidate TOML path; existing files are preserved")
    original = tomllib.loads(authority.payload.decode("utf-8-sig"))
    original_exp = _build(original, authority.source)
    run_seconds = _duration(request.get("run_seconds", original_exp.run_seconds))
    original_fetch = original.get("fetch", {})
    if original_fetch.get("source") != "era5" or "case_data" not in original:
        raise ValueError("ERA5 forcing edits require a saved ERA5 case with its declared Vtable and case data")
    cycle = original_exp.start_time
    if cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError("ERA5 forcing initialization must be an exact UTC hour; the edit preserves the experiment clock")
    cadence = request["cadence_hours"]
    member = validate_selection(product_type=request["product_type"], member=request["member"],
        provider=request["provider"], cadence=cadence, cycle=cycle)
    # Include the closing boundary after a run ending between native snapshots.
    hours = math.ceil(run_seconds / (cadence * 3600)) * cadence
    if hours <= 0:
        raise ValueError("ERA5 forcing requires a positive experiment duration")
    if original_fetch.get("area") is not None:
        area = fetch.parse_area(str(original_fetch["area"]))
    elif original_fetch.get("point") is not None and original_fetch.get("radius_km") is not None:
        area = fetch.area_from_point(str(original_fetch["point"]), float(original_fetch["radius_km"]))
    else:
        raise ValueError("The saved ERA5 fetch needs a geographic area or point and radius")
    area_text = f"{area.lat_south:g},{area.lon_west:g},{area.lat_north:g},{area.lon_east:g}"
    identity = {"source": "era5", "cycle": cycle.strftime("%Y-%m-%dT%H"), "hours": hours,
        "area": area_text, "cadence": cadence, "era5_provider": request["provider"], "era5_product": request["product_type"]}
    if member is not None:
        identity["member"] = member
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = output.parent / ".forcing-cache" / ("era5-" + identity_sha[:24])
    forcing = cache / fetch.ERA5_COMBINED_NAMES[request["provider"]]
    raw = copy.deepcopy(original)
    raw["experiment"]["run_seconds"] = run_seconds
    raw["fetch"] = {**identity, "out": str(cache), "retrieve": True}
    raw["case_data"] = resolved_case_data_paths(raw["case_data"], base_dir=authority.base_dir, source=str(authority.source))
    original_wps = Path(raw["case_data"]["wps_namelist"])
    raw["case_data"].update(forcing=[str(forcing)], forcing_interval_s=cadence * 3600, wps_namelist=str(wps_output))
    if "static" in raw:
        from gpuwm.static.highres_production import parse_static_table
        static = parse_static_table(raw["static"], source=str(authority.source), base_dir=authority.base_dir)
        if static is not None:
            raw["static"]["highres"]["cache_root"] = str(static.cache_root.resolve())
    exp = _build(raw, output)
    validate_boundary_timing(exp, cadence * 3600, source="ERA5 forcing edit")
    if exp.start_time != original_exp.start_time or exp.run_seconds != run_seconds:
        raise ValueError("The forcing edit did not preserve the requested experiment clock")
    from gpuwm.cli import _join_negative_coordinates
    arguments = _join_negative_coordinates(_fetch_arguments_from_hints(raw["fetch"], out=cache))
    _validate_fetch_arguments(arguments)
    text = "# ERA5 forcing selection; original preserved. No acquisition or forecast has run.\n" + emit_experiment_toml(raw)
    config_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    outlines = native_domain_outlines(exp)
    configuration = {"schema": "arwen.companion-configuration.v1", "config_path": str(output),
        "config_sha256": config_sha, "geometry_backend": "rust-static-fields", "domains": outlines,
        "domains_geojson": domain_geojson(outlines), "experiment": experiment_config_document(exp),
        "fetch": raw["fetch"], "case_data": {key: raw["case_data"][key] for key in ("forcing_interval_s", "wps_namelist")}}
    wps = _wps_text(exp, original_wps, wps_output, raw, len(original_exp.domains))
    result = {"schema": RESULT_SCHEMA, "created": True, "forecast_started": False, "acquisition_started": False,
        "config_path": str(output), "config_sha256": config_sha, "wps_path": str(wps_output),
        "receipt_path": str(receipt_output), "source_path": str(authority.source), "source_sha256": authority.sha256,
        "configuration": configuration, "selection": {"product_type": request["product_type"], "provider": request["provider"],
            "member": member, "cadence_hours": cadence, "forcing_path": str(forcing), "request_sha256": identity_sha,
            "boundary_window_hours": hours, "boundary_end_utc": (cycle + timedelta(hours=hours)).isoformat() + "Z",
            "experiment_run_seconds": exp.run_seconds},
        "fetch_argv": ["fetch", *arguments], "changes": [{"field": k, "before": a, "after": b} for k, a, b in changes(original, raw)],
        "validation": {"configuration_parser": "passed", "native_geometry": "passed", "boundary_timing": "passed",
            "fetch_cli_parser": "passed", "forecast_or_memory_admission": "not_run"},
        "notes": ["The fetch stage validates full fields, pressure levels, times and coverage before publication.",
            "EDA downloads all ten CDS members in pressure/surface batches and selects the encoded member in native Rust.",
            "A padded closing boundary preserves the experiment duration."]}
    if hashlib.sha256(authority.source.read_bytes()).hexdigest() != authority.sha256:
        raise ValueError("The original configuration changed while preparing the candidate")
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish_new_files([(wps_output, wps), (receipt_output, _json(result)), (output, text)])
    return result


def edit_schedule(request):
    """Publish a duration change through the existing config/forcing authorities."""
    from gpuwm import fetch
    from gpuwm.case_data import build_case_data, resolved_case_data_paths
    from gpuwm.companion_domains import (_exact_keys, _build, _wps_text, _json,
                                          native_domain_outlines, domain_geojson)
    from gpuwm.config_authority import read_config_authority
    from gpuwm.core.preflight import case_forcing_schedule
    from gpuwm.experiment import experiment_config_document, validate_boundary_timing
    from gpuwm.runplan import (_fetch_arguments_from_hints, _validate_fetch_arguments,
                               prepared_chain_for_source)
    from gpuwm.source_adapters import source_forcing_interval_seconds
    from gpuwm.starter_template import changes, _publish_new_files
    from gpuwm.toml_document import emit_experiment_toml

    required = {"schema", "config_path", "expected_sha256", "output_path", "run_seconds"}
    _exact_keys(request, required, required, where="schedule edit request")
    if request["schema"] != SCHEDULE_REQUEST_SCHEMA:
        raise ValueError("Unsupported schedule edit request schema")
    run_seconds = _duration(request["run_seconds"])
    authority = read_config_authority(request["config_path"])
    if request["expected_sha256"] != authority.sha256:
        raise ValueError("The selected configuration changed; refresh it before editing the duration")
    output = Path(request["output_path"]).expanduser().resolve()
    wps_output = output.with_suffix(".namelist.wps")
    receipt_output = output.with_suffix(".schedule.json")
    outputs = (output, wps_output, receipt_output)
    if (len(set(outputs)) != 3 or output == authority.source
            or any(os.path.lexists(path) for path in outputs)):
        raise FileExistsError("Choose a new candidate TOML path; existing files are preserved")
    original = tomllib.loads(authority.payload.decode("utf-8-sig"))
    original_exp = _build(original, authority.source)
    raw = copy.deepcopy(original)
    raw["experiment"]["run_seconds"] = run_seconds
    original_wps = authority.source.with_suffix(".namelist.wps")
    if "case_data" in raw:
        raw["case_data"] = resolved_case_data_paths(
            raw["case_data"], base_dir=authority.base_dir, source=str(authority.source))
        original_wps = Path(raw["case_data"]["wps_namelist"])
        raw["case_data"]["wps_namelist"] = str(wps_output)
    if "static" in raw:
        from gpuwm.static.highres_production import parse_static_table
        static = parse_static_table(raw["static"], source=str(authority.source),
                                   base_dir=authority.base_dir)
        if static is not None:
            raw["static"]["highres"]["cache_root"] = str(static.cache_root.resolve())
    exp = _build(raw, output)
    if exp.start_time != original_exp.start_time or exp.run_seconds != run_seconds:
        raise ValueError("The schedule edit did not preserve the requested experiment clock")
    hints = raw.get("fetch")
    source = hints.get("source") if isinstance(hints, dict) else None
    data = (build_case_data(raw["case_data"], source=str(output), base_dir=output.parent,
                           require_inputs=False, require_met_inputs=False)
            if "case_data" in raw else None)
    interval = data.forcing_interval_s if data is not None else None
    retained_intervals = None
    preserved = False
    coverage = "no supplied case inputs"
    if data is not None:
        if data.forcing and data.vtable.is_file() and all(path.is_file() for path in data.forcing):
            try:
                interval, retained_intervals = case_forcing_schedule(data, exp)
                preserved = True
                coverage = "supplied time inventory covers the requested forecast and closing boundary"
            except ValueError as error:
                # The canonical schedule validator distinguishes a valid but
                # short window from malformed/mismatched supplied inputs.
                if not str(error).startswith("forcing ends at "):
                    raise
                coverage = str(error)
        else:
            coverage = "one or more declared forcing inputs are not present"
    if interval is None and isinstance(hints, dict) and hints.get("cadence") is not None:
        interval = float(hints["cadence"]) * 3600
    if interval is None and source is not None:
        interval = source_forcing_interval_seconds(source)
    if interval is not None:
        if (not math.isfinite(interval) or interval <= 0
                or int(interval) != interval):
            raise ValueError("A schedule requires a finite positive whole-second forcing interval")
        validate_boundary_timing(exp, int(interval), source="duration edit")
    required_seconds = (None if interval is None else
                        math.ceil(run_seconds / interval) * interval)
    fetch_argv = None
    request_sha = None
    if preserved:
        policy = "preserved-supplied-forcing"
        # Preserve a wider acquisition's complete identity as well as its
        # tuple: an EDA receipt belongs to its original request, not a shorter
        # request merely because the new forecast ends earlier.
        if interval is not None:
            raw["case_data"]["forcing_interval_s"] = interval
        if hints and hints.get("out"):
            hints["out"] = str(Path(hints["out"]).expanduser().resolve())
    elif source is not None:
        if required_seconds is None or required_seconds % 3600:
            raise ValueError("The saved fetch route requires a whole-hour closing boundary")
        fetch_hours = int(required_seconds / 3600)
        cycle = fetch.parse_cycle(str(hints.get("cycle", "")), source)
        expected_start = cycle + timedelta(hours=hints.get("forecast_start_hour", 0))
        if expected_start != exp.start_time:
            raise ValueError("The saved fetch cycle and forecast lead do not match the experiment initialization")
        identity = {key: value for key, value in hints.items() if key != "out"}
        identity["hours"] = fetch_hours
        if source == "era5":
            identity["retrieve"] = True
        fetch.validate_fetch_hints(identity, source=str(output))
        request_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = output.parent / ".forcing-cache" / (source + "-" + request_sha[:24])
        raw["fetch"] = {**identity, "out": str(cache)}
        if data is not None:
            if source != "era5":
                raise ValueError("The saved case-data source has no registered combined-file fetch binding")
            provider = identity.get("era5_provider", "cds")
            raw["case_data"].update(
                forcing=[str(cache / fetch.ERA5_COMBINED_NAMES[provider])],
                forcing_interval_s=interval)
        from gpuwm.cli import _join_negative_coordinates
        arguments = _join_negative_coordinates(
            _fetch_arguments_from_hints(raw["fetch"], out=cache))
        _validate_fetch_arguments(arguments)
        fetch_argv = ["fetch", *arguments]
        policy = "new-request-cache"
    elif data is not None:
        raise ValueError("The supplied forcing does not cover the new duration and this configuration has no fetch recipe")
    else:
        policy = "no-external-forcing"
    exp = _build(raw, output)
    text = "# Forecast duration edit; original preserved. No acquisition or forecast has run.\n" + emit_experiment_toml(raw)
    config_sha = hashlib.sha256(text.encode()).hexdigest()
    outlines = native_domain_outlines(exp)
    configuration = {"schema": "arwen.companion-configuration.v1", "config_path": str(output),
        "config_sha256": config_sha, "geometry_backend": "rust-static-fields", "domains": outlines,
        "domains_geojson": domain_geojson(outlines), "experiment": experiment_config_document(exp),
        "fetch": raw.get("fetch", {}), "case_data": {key: value for key, value in raw.get("case_data", {}).items()
            if key in ("forcing_interval_s", "start_time", "end_time", "wps_namelist")}}
    wps = _wps_text(exp, original_wps, wps_output, raw, len(original_exp.domains),
                    original_domain_ids=[domain.grid_id for domain in original_exp.domains])
    companions = [(wps_output, wps)]
    if data is None and source is not None and prepared_chain_for_source(source) == "prepared:hrrr":
        # The native route reads all four companions. Regenerate and round-trip
        # them through its own writer before publishing the new configuration.
        from gpuwm.hrrr_route_inputs import write_hrrr_route_inputs
        with tempfile.TemporaryDirectory(prefix="arwen-schedule-") as directory:
            staged = Path(directory) / output.name
            written = write_hrrr_route_inputs(staged, exp, wps_text=wps,
                writer=lambda path, content: path.write_text(content, encoding="utf-8"))
            companions = [(output.parent / path.name, path.read_text(encoding="utf-8")) for path in written]
    for path, _ in companions:
        if os.path.lexists(path):
            raise FileExistsError("Choose a new candidate path; existing route companions are preserved")
    result = {"schema": SCHEDULE_RESULT_SCHEMA, "created": True, "forecast_started": False,
        "acquisition_started": False, "config_path": str(output), "config_sha256": config_sha,
        "wps_path": str(wps_output), "receipt_path": str(receipt_output),
        "source_path": str(authority.source), "source_sha256": authority.sha256,
        "configuration": configuration, "fetch_argv": fetch_argv,
        "schedule": {"experiment_run_seconds": exp.run_seconds,
            "boundary_window_hours": None if required_seconds is None else required_seconds / 3600,
            "boundary_end_utc": None if required_seconds is None else
                (exp.start_time + timedelta(seconds=required_seconds)).isoformat() + "Z",
            "forcing_policy": policy, "forcing_coverage": coverage,
            "retained_forcing_intervals": retained_intervals, "request_sha256": request_sha},
        "route_companions": [str(path) for path, _ in companions],
        "changes": [{"field": key, "before": before, "after": after} for key, before, after in changes(original, raw)],
        "validation": {"configuration_parser": "passed", "native_geometry": "passed",
            "boundary_timing": "passed" if interval is not None else "no external boundary",
            "fetch_cli_parser": "passed" if fetch_argv is not None else "no new fetch request",
            "forecast_or_memory_admission": "not_run"}}
    if hashlib.sha256(authority.source.read_bytes()).hexdigest() != authority.sha256:
        raise ValueError("The original configuration changed while preparing the candidate")
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish_new_files([*companions, (receipt_output, _json(result)), (output, text)])
    return result


def main(args):
    from gpuwm.companion_domains import _json
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.capabilities:
                result = capabilities()
            else:
                schedule = getattr(args, "schedule_request", None)
                path = schedule or args.request
                if not path.is_file() or path.stat().st_size > 256 * 1024:
                    raise ValueError("Forcing edit request must be a JSON file no larger than 256 KiB")
                request = json.loads(path.read_text(encoding="utf-8-sig"))
                result = edit_schedule(request) if schedule else edit_configuration(request)
        print(_json(result), end="")
        return 0
    except Exception as error:
        schema = SCHEDULE_RESULT_SCHEMA if getattr(args, "schedule_request", None) else RESULT_SCHEMA
        print(_json({"schema": schema, "created": False, "forecast_started": False, "acquisition_started": False, "error": str(error)}), end="")
        return 1


def register_cli(subparsers):
    parser = subparsers.add_parser("companion-forcing", help="create an ERA5 cadence/member forcing candidate without acquisition or Run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", type=Path)
    mode.add_argument("--schedule-request", type=Path)
    mode.add_argument("--capabilities", action="store_true")
    parser.set_defaults(func=main)
