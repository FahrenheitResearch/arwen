"""Bind native analyzed-input preparation to the configuration it will run."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import tomllib


def resolved_run_settings(cfg):
    """Wire representation of actual RunConfig with resolved radiation selectors."""
    from gpuwm.config import radiation_scheme_ids
    resolved = asdict(cfg)
    resolved["ra_lw_physics"], resolved["ra_sw_physics"] = radiation_scheme_ids(cfg)
    resolved["radiation_scheme_ids"] = list(radiation_scheme_ids(cfg))
    return resolved


def resolve_root_experiment(*, target, vertical, namelist_input, start_time,
                            run_seconds, experiment_config=None, wps_namelist=None,
                            physics_profile=None, acknowledgements=(),
                            history_interval_seconds=None):
    """Use an explicit experiment or the common WRF importer; retain d01 settings.

    The CLI's forcing window may be a subwindow of the configured hierarchy
    (sealed extensions use this). Geometry and vertical values remain asserted
    against the same target that the decoder and static preparation consume.
    """
    from gpuwm.experiment import build_experiment_from_config_tables
    from gpuwm.config_authority import read_config_authority
    from gpuwm.namelist_import import import_namelists, parse_namelist
    from gpuwm.physics_compat import single_domain_runtime_switches

    if experiment_config is not None:
        from gpuwm.experiment import load_experiment
        load_experiment(experiment_config)  # validate companion tables through their owners
        raw = tomllib.loads(read_config_authority(experiment_config).payload.decode("utf-8"))
        authority = str(read_config_authority(experiment_config).source)
        # Prepared meteorological identity is independent of execution capacity
        # and fetch hints. The caller keeps these in its unchanged launch config.
        raw.pop("fetch", None)
        raw.pop("tiles", None)
        for domain in raw.get("domain", ()):
            domain.pop("tiles", None)
    else:
        variant = (single_domain_runtime_switches(physics_profile).get("ra_rrtmg_variant")
                   if physics_profile is not None else None)
        with tempfile.TemporaryDirectory(prefix="gpuwm-namelist-root-") as scratch:
            if wps_namelist is None:
                # The native source door already owns the projection/target.
                # WRF input owns its domain columns; keep every column so the
                # common importer can validate their hierarchy before extracting d01.
                sections = parse_namelist(namelist_input)
                dm = sections.get("domains", {})
                count = int(dm.get("max_dom", [1])[0])
                def row(key, default):
                    values = list(dm.get(key, [default] * count))
                    if key == "parent_id":
                        values[0] = 1
                    return ", ".join(repr(v) for v in values)
                wps = ("&share\n wrf_core='ARW',\n max_dom=" + str(count) +
                       ",\n interval_seconds=3600,\n io_form_geogrid=2,\n/\n&geogrid\n" +
                       "\n".join(f" {key}={row(key, value)}," for key, value in (
                           ("parent_id", 1), ("parent_grid_ratio", 1),
                           ("i_parent_start", 1), ("j_parent_start", 1),
                           ("e_we", target.nx + 1), ("e_sn", target.ny + 1))) +
                       f"\n dx={target.dx_m!r},\n dy={target.dy_m!r},\n map_proj='lambert',\n" +
                       f" ref_lat={target.ref_lat!r},\n ref_lon={target.ref_lon!r},\n" +
                       f" truelat1={target.truelat1!r},\n truelat2={target.truelat2!r},\n" +
                       f" stand_lon={target.stand_lon!r},\n geog_data_res='default',\n/\n")
                wps_namelist = Path(scratch) / "namelist.wps"
                wps_namelist.write_text(wps, encoding="utf-8")
            text, _ = import_namelists(wps_namelist, namelist_input,
                name=target.name, rrtmg_variant=variant,
                acknowledgements=tuple(acknowledgements))
        raw = tomllib.loads(text)
        authority = str(namelist_input)
    full = build_experiment_from_config_tables(
        raw, source=authority, base_dir=Path(authority).parent)
    from gpuwm.hrrr_route_inputs import target_domain
    observed = asdict(target_domain(full))
    expected = asdict(target)
    # A descriptive experiment name is not geometry.
    observed.pop("name", None)
    expected.pop("name", None)
    if observed != expected:
        drift = {key: (observed.get(key), value) for key, value in expected.items()
                 if observed.get(key) != value}
        raise ValueError(f"configured d01 differs from the native target: {drift}")
    for name in ("p_top", "hybrid_opt", "etac", "eta_levels"):
        if getattr(full.vertical, name) != getattr(vertical, name):
            raise ValueError(f"configured d01 vertical {name} differs from namelist.input")
    raw = copy.deepcopy(raw)
    raw["domain"] = raw["domain"][:1]
    raw["experiment"].update(name=target.name, feedback=0, smooth_option=0)
    if start_time is not None:
        raw["experiment"]["start_time"] = start_time
    if run_seconds is not None:
        raw["experiment"]["run_seconds"] = float(run_seconds)
    if history_interval_seconds is not None:
        raw["domain"][0]["history_interval_s"] = float(history_interval_seconds)
    exp = build_experiment_from_config_tables(
        raw, source=authority, base_dir=Path(authority).parent)
    if physics_profile is not None:
        from gpuwm.physics_compat import validate_single_domain_physics_profile
        validate_single_domain_physics_profile(physics_profile, config=exp.root.run,
            expert_acknowledgements=tuple(acknowledgements))
    # Published authorities live beside the prepared payload. Resolve only
    # companion-owned path keys, including not-yet-created cache directories.
    from gpuwm.case_data import resolved_case_data_paths
    from gpuwm.static.highres_production import parse_static_table
    base_dir = Path(authority).resolve().parent
    if "case_data" in raw:
        raw["case_data"] = resolved_case_data_paths(
            raw["case_data"], base_dir=base_dir, source=authority)
    highres = parse_static_table(raw.get("static"), source=authority, base_dir=base_dir)
    if highres is not None:
        raw["static"]["highres"]["cache_root"] = str(highres.cache_root.resolve())
    return exp, raw


def native_configuration_defaults(*, experiment_config=None, namelist_input,
                                  domain_spec=None, wps_namelist=None,
                                  physics_profile=None, acknowledgements=()):
    """Load a producing configuration's clock before applying CLI overrides."""
    if experiment_config is not None:
        from gpuwm.experiment import load_experiment
        return load_experiment(experiment_config)
    from gpuwm.ingest.hrrr_target import load_hrrr_target_domain
    from gpuwm.vertical_contract import explicit_vertical_from_wrf_namelist
    target = load_hrrr_target_domain(domain_spec)
    vertical = explicit_vertical_from_wrf_namelist(namelist_input,
        expected_nz=target.nz, context="native preparation configuration")
    exp, _ = resolve_root_experiment(target=target, vertical=vertical,
        namelist_input=namelist_input, start_time=None, run_seconds=None,
        wps_namelist=wps_namelist, physics_profile=physics_profile,
        acknowledgements=acknowledgements)
    return exp
