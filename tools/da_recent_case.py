"""Author a public recent regional case, then bind actual prepared/obs receipts.

Authoring is CPU metadata work. Preparation uses the shipped mapped source
door: an independently declared HRRR native analysis and RAP native forcing.
No weather-data hashes are invented before the data are fetched and prepared.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tomllib

REPO = Path(__file__).resolve().parents[1]
PROFILE = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"
SOURCE = "rap-native"
#: Lateral forcing products a recent case may name.  Each row is the mapped
#: source id the preparation runs under, the public bucket and object key for
#: one lead, and the in-band surface role its supplements carry.  ``gfs`` is
#: GFS pgrb2.0p25 prepared through the packaged GDAS pgrb2 0.25 profile: that
#: catalogue is record-for-record the GFS one (the adapter's own 696/696
#: measurement), and the mapped route is the one that binds an independent
#: HRRR initial analysis beside the forcing clock.  GFS keys are read from
#: the 00/06/12/18 UTC cycle at the model start.
FORCING = {
    "rap-native": {"source": "rap-native", "bucket": "noaa-rap-pds", "cycle_hours": None,
                   "role": "rap_native_in_band_surface", "max_lead": 21,
                   "key": "rap.{date}/rap.t{hour}z.awp130bgrbf{lead:02d}.grib2",
                   "aerosol": {"mp28_aerosol_source": "analysis", "use_rap_aero_icbc": True},
                   "product": "RAP awp130bgrb native hybrid levels, hourly"},
    "gfs": {"source": "gdas", "bucket": "noaa-gfs-bdp-pds", "cycle_hours": (0, 6, 12, 18),
            "role": "gdas_pgrb2_in_band_surface", "max_lead": 9,
            "key": "gfs.{date}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.f{lead:03d}",
            # GFS carries no QNWFA/QNIFA, and analyzed aerosol must be on every
            # initial and boundary frame: the Thompson aerosol-aware fields
            # start from and are forced by the monthly WIF climatology instead.
            "aerosol": {"mp28_aerosol_source": "climatology", "use_rap_aero_icbc": False},
            "product": "GFS pgrb2.0p25 hourly leads through the packaged GDAS pgrb2 0.25 mapped profile"},
}
SOURCES = frozenset(row["source"] for row in FORCING.values())
#: The deck's 3 km scales.  A coarser grid keeps the 40 km perturbation
#: spectrum (a physical scale) and scales the localization radii with the
#: grid so each still spans four cells.
DECK_DX_M = 3000.0
DECK_LOCALIZATION = {"horizontal_loc_m": 12000.0, "vertical_loc_m": 6000.0,
                     "sfc_horizontal_loc_m": 12000.0, "sfc_vertical_loc_m": 3000.0}


def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(text.replace("\r\n", "\n").encode())


def dump(path, doc):
    write(path, json.dumps(doc, indent=2, sort_keys=True) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc(value):
    time = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if time.tzinfo is None or time.utcoffset().total_seconds() != 0:
        raise ValueError("model start must be explicitly UTC")
    if time.minute or time.second or time.microsecond:
        raise ValueError("model start must be an exact hourly source cycle")
    return time.astimezone(timezone.utc)


def stamp(value):
    return value.isoformat().replace("+00:00", "Z")


def memory_plan(exp, forcing_hours, members=32):
    from gpuwm.core import preflight
    from gpuwm.da import cycle_admission, moments, perturb
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS
    run = exp.root.run
    species = moments.scheme_moments(run.mp_physics)
    perturbation = perturb.PerturbationConfig.from_mapping({
        "dx_km": run.dx / 1000, "dy_km": run.dy / 1000, "rim_width": 5,
        "fields": [{"name": key, "amplitude": amplitude, "length_scale_km": 40}
                   for key, amplitude in (("u", 1.5), ("v", 1.5), ("theta", .5))]
                  + [{"name": "qv", "amplitude": .05, "length_scale_km": 40,
                      "mode": "lognormal", "clip_sigmas": 2.5}],
        "species": [{"mass_field": name, "amplitude": .7, "length_scale_km": 40,
                     "threshold_kg_kg": species.q_threshold, "clip_sigmas": 2.5}
                    for name in species.mass_fields if name in perturb.SUPPORTED_SPECIES]})
    from gpuwm.da.hydrostatic import loading_masses_for_scheme
    perturb_bytes = perturb.device_working_bytes(
        perturbation, (run.nz, run.ny, run.nx),
        loading_masses=loading_masses_for_scheme(run.mp_physics))
    profile = preflight.card_local_memory_profile(32)
    price = cycle_admission.price_cycle(exp, forcing_intervals=forcing_hours,
                                        observation_points=0, perturbation_bytes=perturb_bytes,
                                        profile=profile)
    shapes = preflight.state_array_shapes(run)
    snapshot_bytes = sum(math.prod(shape) * 4 for name, shape in shapes.items()
                         if name in STATE_SERIALIZED_ATTRS)
    analysis_shapes = {**shapes, "theta": shapes["thp"]}
    fields = tuple(name for name in moments.analysis_fields(run.mp_physics) if name in analysis_shapes)
    prior_bytes = members * sum(math.prod(analysis_shapes[name]) * 8 for name in fields)
    candidate = next((k for k in (4, 2, 1) if price.required_bytes * k <= 28 * 2**30), 0)
    return {"forecast_cycle_price": price.receipt(),
            "pricing_host_platform": sys.platform,
            "absent_card_profile": profile.name,
            "estimated_member_peak_gib": price.required_bytes / 2**30,
            "packing_candidate_per_32gib_card": candidate,
            "mps_required_if_four": True,
            "fft_plan_workspace": "not measured; public absent-device estimate prices plan work at zero; live admission must measure it",
            "host_mirror_bytes_per_member": snapshot_bytes,
            "members": members, "host_mirror_bytes_members_plus_control": snapshot_bytes * (members + 1),
            "host_Hz_and_surface_bytes_members": members * (run.nz * run.ny * run.nx + 5 * run.ny * run.nx) * 8,
            "host_analysis_float64_prior_bytes_members": prior_bytes,
            "host_rss_status": "array inventory floors, not measured worker/coordinator RSS upper bounds; runtime host admission still required",
            "GPU_LETKF_status": "solver can host-stage chunks on GPU; full analysis price needs actual observation grid and masks"}


def normalize_surface_networks(value, bbox=None):
    """The caller's networks, validated, or the domain's own when none are named.

    With no networks named the set comes from the domain box (W, S, E, N)
    through :func:`gpuwm.obs.surface_networks.networks_for_domain`: every
    network in the frozen table whose extent meets the domain plus a small
    margin. A domain no network reaches is refused there by name; there is
    no regional fallback list.
    """
    if value is None:
        if bbox is None:
            raise ValueError("surface networks default from the domain box; no networks and no box were given")
        from gpuwm.obs.surface_networks import networks_for_domain
        return list(networks_for_domain(*bbox))
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("surface networks must be a nonempty list of public network identifiers")
    networks = [entry.strip() for item in value if isinstance(item, str) for entry in item.split(",")]
    if (any(not isinstance(item, str) for item in value) or not networks
            or any(re.fullmatch(r"[A-Z0-9_]+", item) is None for item in networks)
            or len(set(networks)) != len(networks)):
        raise ValueError("surface networks must be distinct uppercase public network identifiers")
    return networks


def native_truth_extent(grid, *, padding_m=75000):
    """Metadata-only virtual native grid around the unchanged forecast grid.

    The translated grid evaluates through the original native grid lattice.
    Native MAPFAC bounds enlarge the integer projected rim until the ground
    length bound reaches the requested padding. No forcing/model data expand.
    """
    from gpuwm.static import rust_bridge
    if not math.isfinite(padding_m) or padding_m < 75000:
        raise ValueError("truth padding must be at least 75000 m")
    bridge = rust_bridge.route("DA independent truth acquisition geometry")
    if bridge is None:
        raise ValueError("native static-fields is required to author truth padding")
    spacing = min(grid.dx, grid.dy)
    rim = math.ceil(padding_m / spacing)
    for _ in range(8):
        virtual = grid.translated(-rim, -rim, e_we=grid.e_we + 2 * rim, e_sn=grid.e_sn + 2 * rim)
        latitude, longitude = virtual.latlon_c()
        corner_map_factor = bridge.grid_array(virtual._rust_handle(bridge), bridge.STAGGER_CORNER,
                                             bridge.ARRAY_MAPFAC, virtual.e_sn, virtual.e_we)
        maximum_map_factor = max(float(virtual.mapfac_m().max()), float(corner_map_factor.max()))
        if not math.isfinite(maximum_map_factor) or maximum_map_factor <= 0:
            raise ValueError("native truth extent has an invalid map factor")
        # One whole-cell extra rim makes the bound conservative at the sampled
        # native perimeter; no geodesic transform is reimplemented in Python.
        needed = math.ceil(padding_m * max(1, maximum_map_factor) / spacing) + 1
        if rim >= needed:
            break
        rim = needed
    else:
        raise ValueError("native truth padding extent failed to settle")
    bbox = [max(-180.0, float(longitude.min())), max(-90.0, float(latitude.min())),
            min(180.0, float(longitude.max())), min(90.0, float(latitude.max()))]
    definition = virtual.definition()
    return {"truth_bbox": bbox, "truth_padding_m": float(padding_m),
            "truth_padding_provenance": {"schema": "gpuwm-da.native-truth-padding.v1",
                "operator": "static-fields native grid_translated and grid_array latitude/longitude/MAPFAC",
                "model_grid_definition": grid.definition(), "virtual_grid_definition": definition,
                "virtual_grid_definition_sha256": document_sha(definition),
                "rim_cells_per_side": rim, "projected_rim_m": rim * spacing,
                "native_maximum_map_factor": maximum_map_factor,
                "minimum_ground_rim_bound_m": rim * spacing / maximum_map_factor,
                "native_library_sha256": sha(Path(bridge.load()._name)),
                "weather_data": "none; virtual geometry only", "bbox_convention": "W,S,E,N",
                "bbox_basis": "native full mass-cell corner footprint"}}


def observation_config(plan, requested_da_hours=None, radar_sites=None, surface_networks=None):
    """Declare the requested full schedule before inventory selects smoke scope."""
    origin = utc(plan["model_start_utc"])
    hours = requested_da_hours or plan.get("requested_da_hours") or len(plan["analysis_times_utc"])
    smoke = plan["smoke_only"]
    if not radar_sites:
        # No regional roster stands in for a missing one: a domain away from
        # the Gulf coast would fetch radars that never see it and fail at
        # min_radars. author() discovers the roster from the domain.
        raise ValueError("radar sites must be named or discovered from the domain (author --radar-sites auto, the default)")
    truth = {key: plan[key] for key in ("truth_bbox", "truth_padding_m", "truth_padding_provenance",
                                        "bbox_basis", "model_mass_bbox") if key in plan}
    return {"schema": "da-rerun.recent-observations.v1", "model_start_utc": stamp(origin),
            "analysis_times_utc": [stamp(origin + timedelta(hours=k)) for k in range(1, hours + 1)],
            "observation_cutoff_utc": stamp(origin + timedelta(hours=hours)), "bbox": plan["bbox"],
            "radar_sites": list(radar_sites),
            "radar_bucket": "unidata-nexrad-level2", "radar_max_age_seconds": 900, "min_radars": 2,
            "surface_networks": normalize_surface_networks(surface_networks, plan["bbox"]),
            "surface_networks_basis": "caller" if surface_networks is not None else "domain-bbox",
            **truth,
            "smoke_inventory_selection": "--first-slot-only required" if smoke else "full-case",
            "radar_roster_basis": plan.get("radar_roster_basis", "caller roster")}


#: Seconds of step per km of spacing on grids coarser than the deck's 3 km.
#: HRRR's 3 km ratio (20 s, 6.7 s/km) gives 60 s at 9 km, and on the CONUS
#: 9 km case of 2026-10-01 the first perturbed member went non-finite at
#: step 46 of its first hour while the unperturbed control survived; the
#: engine's measured terrain clock advised 30 s (39 m/s at the 3,699 m
#: crest), and at 30 s all perturbed members of the first leg stayed finite.
COARSE_SECONDS_PER_KM = 10 / 3


def time_step_for(dx_m, explicit=None):
    """The model step: HRRR's 20 s at the deck's 3 km, 10/3 s per km coarser, or explicit.

    The step must divide the 120 s rain history and the 900 s radiation
    clock, or the authored domain refuses at preparation.
    """
    if explicit is not None:
        step = explicit
    elif dx_m <= DECK_DX_M:
        step = int(round(20 * dx_m / DECK_DX_M))
    else:
        # The largest whole step at or under the coarse ratio that divides both clocks.
        ceiling = int(dx_m / 1000 * COARSE_SECONDS_PER_KM + 1e-9)
        step = next((k for k in range(ceiling, 0, -1) if 120 % k == 0 and 900 % k == 0), 0)
    if type(step) is not int:
        raise ValueError("time step must be a whole number of seconds")
    if step < 1 or 120 % step or 900 % step:
        raise ValueError(f"no whole time step at dx={dx_m} m divides the 120 s history and 900 s radiation clocks")
    return step


def localization_for(dx_m):
    """Horizontal radii scale with the grid; vertical radii are physical."""
    scale = max(1.0, float(dx_m) / DECK_DX_M)
    return {key: (value * scale if "horizontal" in key else value)
            for key, value in DECK_LOCALIZATION.items()}


def domain_radar_sites(grid, *, range_km=250.0, stride=4):
    """Every WSR-88D whose 250 km reach touches the model domain.

    Discovery by coverage through the native site table, never a roster
    typed in for one region.
    """
    from gpuwm.obs.allradar import discover_sites
    from gpuwm.obs.nexrad import find_nexrad_bin, run_sites
    binary = find_nexrad_bin()
    if binary is None:
        raise ValueError("native rw_nexrad is required to discover the radar roster")
    table = run_sites(binary)
    rows = table.get("sites", table) if isinstance(table, dict) else table
    catalog = [row for row in rows if re.fullmatch(r"K[A-Z]{3}", str(row.get("id", "")).upper())]
    lat, lon = grid.latlon_mass()
    found = discover_sites(catalog, lat[::stride, ::stride].reshape(-1), lon[::stride, ::stride].reshape(-1),
                           range_km=range_km)
    return sorted(site.site for site in found)


def author(out, *, start, latitude, longitude, nx, ny, da_hours=6,
           shared_root="/workspace/shared-prepared/da-rerun-286/recent", smoke=False,
           smoke_run_hours=None, geog_root="/workspace/shared-prepared/WPS_GEOG", radar_sites=None,
           surface_networks=None, dx_m=DECK_DX_M, forcing="rap-native", free_hours=6,
           companion_of=None, time_step_s=None):
    from gpuwm.companion_domains import candidate_wps_text
    from gpuwm.experiment import build_experiment_from_config_tables
    from gpuwm.physics_compat import single_domain_runtime_switches
    from gpuwm.prepared_single_domain_forecast import materialize_named_source_authorities
    from gpuwm.toml_document import emit_experiment_toml
    minimum = 33 if smoke else 201
    dx_m = float(dx_m)
    if not math.isfinite(dx_m) or dx_m < DECK_DX_M:
        raise ValueError("recent cases run at 3 km or coarser")
    if (type(nx) is not int or type(ny) is not int
            or min(nx, ny) * dx_m < minimum * DECK_DX_M or min(nx, ny) < 33):
        raise ValueError(f"case must span at least {minimum} 3 km cells ({minimum * 3} km) per axis")
    if da_hours not in (3, 4, 5, 6):
        raise ValueError("DA duration must be 3..6 hourly legs")
    if forcing not in FORCING:
        raise ValueError("forcing must be one of " + ", ".join(sorted(FORCING)))
    if free_hours != 6:
        raise ValueError("recent free forecasts are six hours")
    forcing_row = FORCING[forcing]
    time_step = time_step_for(dx_m, time_step_s)
    if surface_networks is not None:
        normalize_surface_networks(surface_networks)
    if smoke_run_hours is not None and (not smoke or smoke_run_hours < 2):
        raise ValueError("smoke_run_hours requires explicit smoke mode and at least two forcing hours")
    if smoke and smoke_run_hours is None:
        smoke_run_hours = 2
    forcing_hours = smoke_run_hours if smoke_run_hours is not None else da_hours + free_hours
    analysis_hours = 1 if smoke_run_hours is not None else da_hours
    origin = utc(start)
    if forcing_row["cycle_hours"] is not None and origin.hour not in forcing_row["cycle_hours"]:
        raise ValueError(f"{forcing} forcing starts at its own cycle hours {forcing_row['cycle_hours']} UTC")
    if forcing_hours > forcing_row["max_lead"]:
        raise ValueError(f"{forcing} forcing is authored through lead {forcing_row['max_lead']} h")
    source = forcing_row["source"]
    template = REPO / "configs/recipes/hrrr_configuration_clock.toml"
    raw = tomllib.loads(template.read_text())
    raw["experiment"].update(name="public_recent_regional_da", start_time=origin.replace(tzinfo=None),
                             run_seconds=float(forcing_hours * 3600), feedback=0)
    raw["projection"].update(ref_lat=float(latitude), ref_lon=float(longitude))
    switches = single_domain_runtime_switches(PROFILE)
    raw["shared"].update(switches)
    raw["shared"].update(moist_cq=True, aer_opt=3, swint_opt=1, **forcing_row["aerosol"],
                         rdlai2d=True, usemonalb=True, alb_sol=1)
    domain = raw["domain"][0]
    domain.update(nx=nx, ny=ny, dx=dx_m, time_step=time_step, history_interval_s=120)
    for key in ("radt", "diff_6th_factor", "cu_physics", "cudt_minutes"):
        domain[key] = switches[key]
    raw["fetch"] = {"source": source, "cycle": origin.strftime("%Y-%m-%dT%H"),
                    "hours": forcing_hours, "cadence": 1}
    out = Path(out).absolute()
    if out.exists():
        raise ValueError("create-only case output already exists")
    base = out / "base"
    base.mkdir(parents=True)
    config = base / "experiment.toml"
    write(config, emit_experiment_toml(raw))
    exp = build_experiment_from_config_tables(raw, source=str(config), base_dir=base)
    wps = base / "namelist.wps"
    write(wps, candidate_wps_text(raw, exp, exp, config))
    authority_receipt = materialize_named_source_authorities(
        source=source, base_experiment_config=config, base_wps_namelist=wps,
        physics_profile=PROFILE, output_directory=out / "authority")
    root = PurePosixPath(shared_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("box shared root must be an absolute Linux path")
    date, hour = origin.strftime("%Y%m%d"), origin.strftime("%H")
    objects = [{"bucket": forcing_row["bucket"],
                "key": forcing_row["key"].format(date=date, hour=hour, lead=lead),
                "role": "lateral_forcing", "lead_hours": lead}
               for lead in range(forcing_hours + 1)]
    for cycle in ((origin,) if smoke or companion_of is not None else (origin, origin + timedelta(hours=3))):
        objects.extend({"bucket": "noaa-hrrr-bdp-pds",
                        "key": f"hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrf{kind}f00.grib2",
                        "role": "initial_analysis" if cycle == origin else "independent_reference_analysis"}
                       for kind in ("nat", "prs", "sfc"))
    raw_path = root / "raw"
    initial = {"schema": "gpuwm-initial-source-v1", "source": "hrrr-native",
               "input_files": [str(raw_path / "noaa-hrrr-bdp-pds" / f"hrrr.{date}/conus/hrrr.t{hour}z.wrfnatf00.grib2")],
               "supplements": {"soil_surface_data": [str(raw_path / "noaa-hrrr-bdp-pds" / f"hrrr.{date}/conus/hrrr.t{hour}z.wrfprsf00.grib2")],
                               "vegetation_surface_data": [str(raw_path / "noaa-hrrr-bdp-pds" / f"hrrr.{date}/conus/hrrr.t{hour}z.wrfsfcf00.grib2")]}}
    dump(out / "initial-inputs.json", initial)
    forcing_files = [str(raw_path / item["bucket"] / item["key"]) for item in objects if item["role"] == "lateral_forcing"]
    write(out / "forcing-inputs.txt", "\n".join(forcing_files) + "\n")
    from gpuwm.hrrr_route_inputs import target_domain
    from gpuwm.ingest.hrrr_target import required_hrrr_source_window
    target = target_domain(exp)
    model_grid = target.grid()
    lat, lon = model_grid.latlon_mass()
    model_mass_bbox = [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]
    lat_corner, lon_corner = model_grid.latlon_c()
    bbox = [float(lon_corner.min()), float(lat_corner.min()), float(lon_corner.max()), float(lat_corner.max())]
    try:
        donor_coverage = {"status": "PASS", "window": required_hrrr_source_window(target).to_dict()}
    except ValueError as error:
        donor_coverage = {"status": "REFUSED", "error": str(error)}
    plan = {"schema": "gpuwm-da.recent-case-plan.v1", "source": source,
            "forcing": forcing, "forcing_product": forcing_row["product"], "aerosol": forcing_row["aerosol"],
            "initial_source": "hrrr-native", "physics_profile": PROFILE,
            "model_start_utc": stamp(origin), "analysis_times_utc": [stamp(origin + timedelta(hours=k)) for k in range(1, analysis_hours + 1)],
            "forecast_fork_utc": stamp(origin + timedelta(hours=analysis_hours)),
            "free_forecast_seconds": 120 if smoke_run_hours is not None else free_hours * 3600,
            "run_seconds": forcing_hours * 3600,
            "requested_da_hours": da_hours,
            "history_interval_seconds": 120,
            "nx": nx, "ny": ny, "nz": exp.root.run.nz, "dx_m": dx_m, "time_step_seconds": time_step,
            "perturbation_length_scale_km": 12 if smoke else 40,
            "localization": localization_for(dx_m),
            "localization_basis": "deck 3 km horizontal radii scaled by dx/3 km so each spans four cells; vertical radii and the 40 km perturbation scale are physical and unchanged",
            "bbox": bbox, "members": 1 if companion_of is not None else 4 if smoke else 32,
            "score_leads_hours": [] if smoke else [1, 3, 6],
            "center_lat": float(latitude), "center_lon": float(longitude),
            "bbox_convention": "W,S,E,N", "smoke_only": bool(smoke),
            "bbox_basis": "native full mass-cell corner footprint", "model_mass_bbox": model_mass_bbox,
            "independent_reference_status": "not requested for integration smoke" if smoke else "HRRR cycle three hours after model origin requested as independent reference; never used for initial state",
            "initial_source_coverage": donor_coverage,
            "authority_dir": str(out / "authority"), "prepared_root": str(root / "prepared"),
            "model_objects": objects, "initial_inputs": str(out / "initial-inputs.json"),
            "forcing_input_list": str(out / "forcing-inputs.txt"),
            "authority_receipt_sha256": sha(out / "authority/authority-receipt.json"),
            "public_recipe_sha256": sha(template),
            "data_status": "source identities requested; object availability, fetched bytes, prepared fields and observations not yet measured",
            "prepare_argv": [sys.executable, "-m", "gpuwm.source_cli", "--source", source,
                             "--input-list", str(out / "forcing-inputs.txt"),
                             *[value for file in forcing_files for value in
                               ("--supplement", forcing_row["role"] + "=" + file)],
                             "--author-input-manifest", str(out / "forcing-input-manifest.json"),
                             "--experiment-config", str(out / "authority/experiment.toml"),
                             "--wps-namelist", str(out / "authority/namelist.wps"),
                             "--geog-root", str(geog_root), "--initial-inputs", str(out / "initial-inputs.json"),
                             "--output-root", str(root / "prepared"), "--preprocess-backend", "cpu",
                             "--preprocess-workers", "12"],
            "observation_grid_candidate": str(root / "prepared/wrf-native-input/wrfinput_d01"),
            "packing": {"mode": "auto", "four_per_card_requires_mps": True,
                        "card_vram_gib": 32, "array_floor_mib_per_float32_field": nx * ny * exp.root.run.nz * 4 / 2**20,
                        "whole_member_peak_vram": "must be admitted or measured on actual prepared case; array floor is not a fit claim"}}
    plan["native_memory_plan"] = memory_plan(exp, forcing_hours, members=plan["members"])
    if companion_of is not None:
        plan.update(companion_of)
        plan["schema"] = "gpuwm-da.recent-companion-grid.v1"
        plan.update(native_truth_extent(model_grid))
        plan["static_source_fetch_argv"] = [sys.executable, "-m", "gpuwm", "fetch-geog",
                                              "--static-source", "hrrr-conus-v4", "--root", str(geog_root)]
        dump(out / "case-plan.json", plan)
        dump(out / "model-object-requests.json", {"schema": "da-rerun.public-object-requests.v1", "objects": objects})
        return plan
    plan.update(native_truth_extent(model_grid))
    if radar_sites is None or radar_sites == ["auto"]:
        radar_sites = domain_radar_sites(model_grid)
        plan["radar_roster_basis"] = "every WSR-88D in the native site table whose 250 km reach touches the domain"
    plan["static_source_fetch_argv"] = [sys.executable, "-m", "gpuwm", "fetch-geog",
                                          "--static-source", "hrrr-conus-v4", "--root", str(geog_root)]
    plan["geography_requirement"] = "The named HRRR static source and the baseline GEOG datasets are both required. Reuse readable public baseline dataset directories; do not replace or refetch an existing shared copy."
    dump(out / "case-plan.json", plan)
    dump(out / "model-object-requests.json", {"schema": "da-rerun.public-object-requests.v1", "objects": objects})
    dump(out / "observation-case.json", observation_config(plan, da_hours, radar_sites, surface_networks))
    return plan


def companion(parent_plan_path, out, *, ratio=3, rim_cells=6, shared_root, geog_root):
    """Author the deterministic grid a fine forecast starts on from the coarse analysis.

    The grid is the parent's own lattice refined by ``ratio`` over the
    parent's interior less ``rim_cells`` on every side, so every fine cell
    sits inside exactly one coarse cell (WRF nest alignment) and the coarse
    lateral relaxation zone is never used as an initial state.  It is
    prepared like the parent (same forcing cycle, same HRRR initial source,
    HRRR static fields on its own 3 km grid), which gives the fine run its
    own terrain, land use and soil state: WRF ``ndown``'s rule that the
    child's own real-data input supplies land identity while the parent
    supplies the meteorology.
    """
    parent = read_json(parent_plan_path)
    if parent.get("schema") != "gpuwm-da.recent-case-plan.v1" or parent.get("smoke_only"):
        raise ValueError("companion grids are authored from a full recent case plan")
    if type(ratio) is not int or ratio < 2 or type(rim_cells) is not int or rim_cells < 5:
        raise ValueError("companion needs an integer refinement of at least 2 and a rim of at least the 5-cell relaxation zone")
    span_x, span_y = parent["nx"] - 2 * rim_cells, parent["ny"] - 2 * rim_cells
    if min(span_x, span_y) < 33:
        raise ValueError("parent interior is too small for a companion grid")
    dx = float(parent["dx_m"]) / ratio
    if dx < DECK_DX_M:
        raise ValueError("companion spacing is finer than the 3 km HRRR configuration")
    start = parent["model_start_utc"]
    hours = len(parent["analysis_times_utc"])
    fork_seconds = (utc(parent["forecast_fork_utc"]) - utc(start)).total_seconds()
    placement = {
        "companion_role": "deterministic fine grid started from the coarse analysis at the forecast fork",
        "parent_case_plan": str(Path(parent_plan_path).resolve()),
        "parent_case_plan_sha256": sha(parent_plan_path),
        "parent_grid": {key: parent[key] for key in ("nx", "ny", "nz", "dx_m", "center_lat", "center_lon")},
        "placement": {"parent_grid_ratio": ratio, "i_parent_start": rim_cells + 1, "j_parent_start": rim_cells + 1,
                      "parent_cells_x": span_x, "parent_cells_y": span_y,
                      "alignment": "same projection and center; fine corners fall on coarse cell corners"},
        "start_from_analysis": {
            "issued_utc": parent["forecast_fork_utc"], "issued_seconds_after_model_start": fork_seconds,
            "method": ("gpuwm downscale (native ndown): the coarse run issued from the final analysis is the parent "
                       "history; the fine initial state is the parent state at issuance interpolated by the nest "
                       "interpolator (SINT horizontally, dry-mass column remap onto the fine terrain vertically), "
                       "lateral boundaries come from the parent history at 15-minute cadence, and land identity "
                       "and soil state come from this grid's own prepared wrfinput (--child-surface-from)."),
            "argv_template": ["python", "-m", "gpuwm", "downscale", "<coarse issued run directory>",
                              "--parent-restart", "latest", "--ratio", str(ratio),
                              "--i-parent-start", str(rim_cells + 1), "--j-parent-start", str(rim_cells + 1),
                              "--child-config", str(Path(out).absolute() / "child-run-config.toml"),
                              "--child-config-sha256", "<child_run_config_sha256>",
                              "--child-surface-from", str(PurePosixPath(shared_root) / "prepared/wrf-native-input/wrfinput_d01"),
                              "--max-boundary-interval-seconds", "900", "--out", "<fine run directory>"],
            "parent_history_requirement": "the issued coarse forecast writes history every 900 s or denser for its six hours"}}
    plan = author(out, start=start, latitude=parent["center_lat"], longitude=parent["center_lon"],
                  nx=span_x * ratio, ny=span_y * ratio, da_hours=hours, shared_root=shared_root,
                  geog_root=geog_root, dx_m=dx, forcing=parent.get("forcing", "rap-native"),
                  companion_of=placement)
    plan.update(child_run_config(Path(out), seconds=plan["free_forecast_seconds"]))
    method = plan["start_from_analysis"]
    method["argv_template"] = [plan["child_run_config_sha256"] if item == "<child_run_config_sha256>" else item
                               for item in method["argv_template"]]
    dump(Path(out) / "case-plan.json", plan)
    return plan


def child_run_config(out, *, seconds):
    """The downscale door's legacy child RunConfig, rendered from this grid's own authority.

    Physics, levels and clock are the materialized HRRR-configuration
    authority at the fine spacing; the child is specified (forced from the
    parent history), not nested, and runs the six-hour free forecast.
    """
    import dataclasses
    from gpuwm.downscale import _render_child_toml
    from gpuwm.experiment import build_experiment_from_config_tables
    from gpuwm.offline_child import resolve_child_run_config
    authority = out / "authority/experiment.toml"
    raw = tomllib.loads(authority.read_text())
    run = build_experiment_from_config_tables(raw, source=str(authority), base_dir=authority.parent).root.run
    config = {key: list(value) if isinstance(value, tuple) else value
              for key, value in dataclasses.asdict(run).items()}
    config.update(specified=True, nested=False, run_seconds=float(seconds))
    path = out / "child-run-config.toml"
    write(path, _render_child_toml(config))
    child = resolve_child_run_config(path)
    if (child.nx, child.ny, child.dx, child.dt) != (run.nx, run.ny, run.dx, run.dt) or not child.specified or child.nested:
        raise ValueError("rendered child RunConfig does not read back as this grid")
    return {"child_run_config": str(path), "child_run_config_sha256": sha(path)}


def require_digest(value, label, *, size=64):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{" + str(size) + r"}", value) is None:
        raise ValueError(f"{label} must be a lowercase {size}-digit digest")
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_file(path, digest, label):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or sha(path) != require_digest(digest, label):
        raise ValueError(f"{label} differs from actual file bytes: {path}")
    return path


def _prepared_preflight(**kwargs):
    from gpuwm.prepared_single_domain_forecast import preflight_prepared_forecast
    return preflight_prepared_forecast(**kwargs)


def _read_grid(path):
    from gpuwm.obs.target_grid import TargetGrid
    return TargetGrid.from_wrfout(path)


def _verify_radar(path, grid):
    from gpuwm.da.obs_radar import read_document
    return read_document(path, expected_grid=grid)


def _verify_surface(path):
    from gpuwm.da.obs_surface import read_record
    return read_record(path)


def seam_time(value):
    """Native seam timestamps without a zone are UTC by their public contract."""
    time = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)
    return time.astimezone(timezone.utc)


def document_sha(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finalize(plan_path, *, prepared_root, slots_path, fetch_receipt_path,
             engine_sha, run_out, manifest_path):
    """Bind actual immutable preparation and observation authorities on CPU.

    Raw fields may already have been retired after preparation. Their complete
    fetch receipt is retained; remaining raw files are rehashed when present.
    The public preflight verifies the complete prepared cache in either case.
    """
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read_json(plan_path)
    source = plan.get("source")
    if (plan.get("schema") != "gpuwm-da.recent-case-plan.v1"
            or source not in SOURCES or plan.get("initial_source") != "hrrr-native"
            or plan.get("physics_profile") != PROFILE):
        raise ValueError("case plan must name the public HRRR initial and a declared forcing authority")
    if plan.get("initial_source_coverage", {}).get("status") != "PASS":
        raise ValueError("initial donor coverage was refused; cannot bind a runnable case")
    members = 4 if plan["smoke_only"] else 32
    if plan["members"] != members and not (plan["smoke_only"] and plan["members"] == 32):
        raise ValueError("recent case roster must be 32; the declared integration smoke uses four")
    require_digest(engine_sha, "engine_sha", size=40)
    prepared = Path(prepared_root).resolve(strict=True)
    authority = Path(plan["authority_dir"]).resolve(strict=True)
    check_file(authority / "authority-receipt.json", plan["authority_receipt_sha256"], "authority receipt")
    proof_path = prepared / "proof.json"
    source_manifest = prepared / "source-evidence/input-manifest.json"
    header_path = prepared / "prepared-cache/header.json"
    header = read_json(header_path)
    content_digest = require_digest(header.get("content_sha256"), "prepared content")
    proof_digest, source_digest = sha(proof_path), sha(source_manifest)
    inputs = _prepared_preflight(source=source, prepared_root=prepared,
        proof_sha256=proof_digest, source_manifest_sha256=source_digest,
        prepared_content_sha256=content_digest, experiment_config=authority / "experiment.toml",
        wps_namelist=authority / "namelist.wps", physics_profile=PROFILE,
        run_seconds=plan["run_seconds"], history_interval_seconds=120)
    run = inputs.experiment.root.run
    if any(getattr(run, key) != plan[key] for key in ("nx", "ny", "nz")):
        raise ValueError("prepared dimensions differ from the authored case")
    initial = inputs.proof.get("initial_source", {})
    if initial.get("source") != "hrrr-native":
        raise ValueError("actual preparation lacks the declared independent HRRR initial analysis")
    origin = utc(plan["model_start_utc"])
    if inputs.experiment.start_time.replace(tzinfo=timezone.utc) != origin:
        raise ValueError("prepared model origin differs from authored case")
    fetched = read_json(fetch_receipt_path)
    if (fetched.get("schema") != "da-rerun.recent-fetch.v1"
            or fetched.get("status") != "complete" or fetched.get("fetch_processes") != 1):
        raise ValueError("one-process complete public fetch receipt required")
    objects = fetched.get("objects", [])
    roster = {(row["bucket"], row["key"]): row for row in objects}
    if len(roster) != len(objects):
        raise ValueError("fetch receipt contains duplicate object identities")
    for requested in plan["model_objects"]:
        if plan["smoke_only"] and requested.get("role") == "independent_reference_analysis":
            continue
        identity = requested["bucket"], requested["key"]
        if identity not in roster:
            raise ValueError(f"fetch receipt is missing declared model object {identity}")
    for row in objects:
        require_digest(row.get("sha256"), "fetched object")
        if type(row.get("bytes")) is not int or row["bytes"] <= 0:
            raise ValueError("fetched object must have a positive recorded byte count")
        raw = Path(row["path"])
        if raw.exists() and (raw.stat().st_size != row["bytes"] or sha(raw) != row["sha256"]):
            raise ValueError(f"remaining raw object differs from fetch receipt: {raw}")
    slots_doc = read_json(slots_path)
    if slots_doc.get("fetch_receipt_sha256") != document_sha(fetched):
        raise ValueError("observation slots are bound to a different public fetch receipt")
    allowed_status = "prepared-smoke-first-slot" if plan["smoke_only"] else "prepared"
    if (slots_doc.get("schema") != "da-rerun.recent-slots.v1"
            or slots_doc.get("status") != allowed_status):
        raise ValueError("observation preparation status differs from case scope")
    slots = slots_doc.get("slots", [])
    if len(slots) != len(plan["analysis_times_utc"]):
        raise ValueError("observation slot count differs from authored analysis schedule")
    if utc(slots_doc["model_start_utc"]) != origin:
        raise ValueError("observation model start differs from authored case")
    grid_path = prepared / "wrf-native-input/wrfinput_d01"
    if not grid_path.is_file():
        raise ValueError("prepared native full-3D observation anchor is missing: wrf-native-input/wrfinput_d01")
    grid = _read_grid(grid_path)
    if any(getattr(grid, key) != plan[key] for key in ("nx", "ny", "nz")):
        raise ValueError("native observation anchor dimensions differ from prepared case")
    elapsed = 0
    bound_slots = []
    for index, (slot, expected_time) in enumerate(zip(slots, plan["analysis_times_utc"])):
        leg = slot.get("leg_seconds")
        if not isinstance(leg, (int, float)) or not math.isfinite(leg) or leg != 3600:
            raise ValueError("recent-case observation legs must be exactly one hour")
        elapsed += leg
        analysis = utc(slot["analysis_time"])
        if analysis != utc(expected_time) or analysis != origin + timedelta(seconds=elapsed):
            raise ValueError(f"slot {index} clock differs from the authored hourly schedule")
        actual_grid = check_file(slot["grid_wrfout"], slot["grid_wrfout_sha256"], "slot grid")
        if actual_grid != grid_path.resolve():
            raise ValueError("observation slots must use the actual prepared native full-3D grid")
        obs = check_file(slot["obs"], slot["obs_sha256"], "radar observation")
        radar_document = _verify_radar(obs, grid)
        if seam_time(radar_document["valid_time"]) != analysis:
            raise ValueError("radar file analysis time differs from the scheduled slot")
        bound_slots.append({**slot, "obs": str(obs), "grid_wrfout": str(actual_grid)})
    fork = utc(plan["forecast_fork_utc"])
    if not slots or utc(slots_doc["forecast_fork_utc"]) != fork or utc(slots[-1]["analysis_time"]) != fork:
        raise ValueError("observation fork differs from authored free-forecast fork")
    if elapsed + plan["free_forecast_seconds"] > plan["run_seconds"]:
        raise ValueError("prepared forcing horizon cannot cover the matched free forecast")
    surface = check_file(slots_doc["surface_obs"], slots_doc["surface_sha256"], "surface observations")
    surface_document = _verify_surface(surface)
    if {seam_time(value) for value in surface_document.get("valid_times", [])} != {utc(value) for value in plan["analysis_times_utc"]}:
        raise ValueError("surface seam clocks differ from the scheduled analyses")
    for report in surface_document.get("reports", []):
        analysis = seam_time(report["valid_time"])
        if report.get("observation_time") is not None and seam_time(report["observation_time"]) > analysis:
            raise ValueError("surface seam contains an observation collected after its analysis")
    manifest = {"schema": "gpuwm-da.recent-run.v1", "source": source, "initial_source": "hrrr-native",
                "forcing": plan.get("forcing", "rap-native"), "dx_m": plan.get("dx_m", DECK_DX_M),
                "perturbation_length_scale_km": plan.get("perturbation_length_scale_km", 12 if plan["smoke_only"] else 40),
                "localization": plan.get("localization", localization_for(DECK_DX_M)),
                "engine_sha": engine_sha, "prepared_root": str(prepared), "authority_dir": str(authority),
                "physics_profile": PROFILE, "proof_sha256": proof_digest,
                "source_manifest_sha256": source_digest, "prepared_content_sha256": content_digest,
                "run_seconds": plan["run_seconds"], "history_interval_seconds": 120,
                "members": members, "authored_member_count": plan["members"],
                "member_count_contract": "integration smoke is four members; full case is 32; preparation authorities contain no ensemble roster",
                "smoke_only": plan["smoke_only"], "slots": bound_slots,
                "model_start_utc": stamp(origin), "start_utc": stamp(origin), "forecast_fork_utc": stamp(fork),
                "free_forecast_seconds": plan["free_forecast_seconds"], "surface_obs": str(surface),
                "surface_obs_sha256": sha(surface), "surface_dewpoint_error_k": 2,
                "out": str(Path(run_out).absolute()), "nx": plan["nx"], "ny": plan["ny"], "nz": plan["nz"],
                "case_plan_path": str(plan_path), "case_plan_sha256": sha(plan_path),
                "slots_receipt_sha256": sha(slots_path), "fetch_receipt_sha256": sha(fetch_receipt_path),
                "native_grid_sha256": sha(grid_path), "native_grid_identity_sha256": grid.identity_sha256(),
                "prepared_header_sha256": sha(header_path), "data_status": "actual prepared and observation bytes verified by public CPU preflight",
                "scientific_gate": "pending", "runtime_admission": "actual FFT, device and host headroom still checked before launch",
                "independent_reference_status": "not bound or run in integration smoke" if plan["smoke_only"] else "requested reference object hashes present in complete fetch receipt",
                "finalizer_sha256": sha(Path(__file__))}
    target = Path(manifest_path)
    if target.exists():
        raise ValueError("create-only run manifest already exists")
    dump(target, manifest)
    return manifest


def finalize_main(argv):
    parser = argparse.ArgumentParser(description="Bind actual public prepared/observation data to the recent run deck")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--fetch-receipt", type=Path, required=True)
    parser.add_argument("--engine-sha", required=True)
    parser.add_argument("--run-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    doc = finalize(args.plan, prepared_root=args.prepared_root, slots_path=args.slots,
                   fetch_receipt_path=args.fetch_receipt, engine_sha=args.engine_sha,
                   run_out=args.run_out, manifest_path=args.manifest)
    print(json.dumps({"run_manifest": str(args.manifest), "prepared_content_sha256": doc["prepared_content_sha256"]}))
    return 0


def companion_main(argv):
    parser = argparse.ArgumentParser(description=companion.__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="the coarse ensemble case-plan.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ratio", type=int, default=3)
    parser.add_argument("--rim-cells", type=int, default=6)
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--geog-root", default="/workspace/shared-prepared/WPS_GEOG")
    args = parser.parse_args(argv)
    plan = companion(args.plan, args.out, ratio=args.ratio, rim_cells=args.rim_cells,
                     shared_root=args.shared_root, geog_root=args.geog_root)
    print(json.dumps({"companion_plan": str(args.out / "case-plan.json"), "nx": plan["nx"], "ny": plan["ny"],
                      "dx_m": plan["dx_m"], "placement": plan["placement"]}))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["finalize"]:
        return finalize_main(argv[1:])
    if argv[:1] == ["companion"]:
        return companion_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2026-10-03T21:00:00Z")
    parser.add_argument("--latitude", type=float, default=31)
    parser.add_argument("--longitude", type=float, default=-89)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--da-hours", type=int, default=6)
    parser.add_argument("--dx-m", type=float, default=DECK_DX_M,
                        help="grid spacing; the time step keeps HRRR's 20 s at 3 km ratio and horizontal localization scales with it")
    parser.add_argument("--time-step-s", type=int, default=None,
                        help="explicit model step; default is HRRR's 20 s at 3 km and 10/3 s per km on coarser grids")
    parser.add_argument("--forcing", choices=sorted(FORCING), default="rap-native",
                        help="lateral forcing product; gfs reads the GFS cycle at the model start")
    parser.add_argument("--radar-sites", nargs="+", default=None,
                        help="explicit public station roster; default (or 'auto') is every WSR-88D whose 250 km reach touches the domain; native availability and usable two-radar coverage are still required")
    parser.add_argument("--surface-networks", nargs="+", default=None,
                        help="explicit public surface network identifiers; otherwise every network whose extent meets the domain box plus a small margin")
    parser.add_argument("--smoke", action="store_true", help="explicit small-grid integration check, not storm qualification")
    parser.add_argument("--smoke-run-hours", type=int, default=None,
                        help="explicit reduced smoke forcing horizon; one hourly observed leg and two minutes free")
    parser.add_argument("--shared-root", default="/workspace/shared-prepared/da-rerun-286/recent")
    parser.add_argument("--geog-root", default="/workspace/shared-prepared/WPS_GEOG")
    args = parser.parse_args(argv)
    plan = author(args.out, start=args.start, latitude=args.latitude, longitude=args.longitude,
                  nx=args.nx, ny=args.ny, da_hours=args.da_hours, shared_root=args.shared_root,
                  smoke=args.smoke, smoke_run_hours=args.smoke_run_hours, geog_root=args.geog_root,
                  radar_sites=args.radar_sites, surface_networks=args.surface_networks,
                  dx_m=args.dx_m, forcing=args.forcing, time_step_s=args.time_step_s)
    print(json.dumps({"case_plan": str(args.out / "case-plan.json"), "bbox": plan["bbox"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
