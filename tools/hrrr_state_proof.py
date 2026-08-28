#!/usr/bin/env python3
"""Measured native-HRRR f00 state, f00/f01 LBC, and first-step proof.

The harness is intentionally bound to the frozen two-domain HRRR proof
namelist.  It exercises d01 through production ingest, WSM6, Dudhia SW,
revised MM5 surface layer, Noah, YSU, specified boundaries, and one stock
RK3 timestep.  It is not a forecast-skill claim or a replacement for the
subsequent one-way nested shakedown.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np

from gpuwm import __version__
from gpuwm import runtime_manifest
from gpuwm.aerosol_source_receipt import aerosol_source_report_entry


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(value):
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def _section_value(text: str, section: str, key: str) -> list[str]:
    section_match = re.search(
        rf"^\s*&{re.escape(section)}\s*(.*?)^\s*/\s*$",
        text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if section_match is None:
        raise ValueError(f"namelist section &{section} is missing")
    body = section_match.group(1)
    value_match = re.search(
        rf"^\s*{re.escape(key)}\s*=\s*(.*?)"
        rf"(?=^\s*[A-Za-z_]\w*\s*=|\Z)",
        body, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if value_match is None:
        raise ValueError(f"namelist key &{section}/{key} is missing")
    return [token.strip() for token in value_match.group(1).split(",")
            if token.strip()]


def _numbers(text: str, section: str, key: str) -> list[float]:
    return [float(value.replace("d", "e").replace("D", "E"))
            for value in _section_value(text, section, key)]


def _assert_frozen_namelist(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    exact = {
        ("domains", "e_we"): [200.0, 301.0],
        ("domains", "e_sn"): [200.0, 301.0],
        ("domains", "e_vert"): [50.0, 50.0],
        ("domains", "dx"): [2999.4213047435587, 999.8071015811862],
        ("domains", "dy"): [2999.4213047435587, 999.8071015811862],
        ("domains", "parent_grid_ratio"): [1.0, 3.0],
        ("domains", "parent_time_step_ratio"): [1.0, 3.0],
        ("domains", "i_parent_start"): [1.0, 50.0],
        ("domains", "j_parent_start"): [1.0, 50.0],
        ("physics", "mp_physics"): [6.0, 6.0],
        ("physics", "ra_lw_physics"): [0.0, 0.0],
        ("physics", "ra_sw_physics"): [1.0, 1.0],
        ("physics", "radt"): [3.0, 1.0],
        ("physics", "sf_sfclay_physics"): [91.0, 91.0],
        ("physics", "sf_surface_physics"): [2.0, 2.0],
        ("physics", "bl_pbl_physics"): [1.0, 1.0],
        ("physics", "cu_physics"): [0.0, 0.0],
        ("dynamics", "hybrid_opt"): [2.0],
        ("dynamics", "etac"): [0.2],
        ("dynamics", "km_opt"): [4.0, 4.0],
        ("dynamics", "diff_6th_opt"): [2.0, 2.0],
        ("dynamics", "diff_6th_factor"): [0.10, 0.08],
        ("dynamics", "diff_6th_slopeopt"): [1.0, 1.0],
        ("bdy_control", "spec_bdy_width"): [5.0],
        ("bdy_control", "spec_zone"): [1.0],
        ("bdy_control", "relax_zone"): [4.0],
    }
    for (section, key), expected in exact.items():
        actual = _numbers(text, section, key)
        if actual != expected:
            raise ValueError(
                f"frozen namelist drift at &{section}/{key}: "
                f"expected {expected}, got {actual}")
    if _numbers(text, "domains", "time_step") != [15.0]:
        raise ValueError("frozen d01 timestep must be 15 s")
    if _numbers(text, "domains", "p_top_requested") != [10000.0]:
        raise ValueError("frozen p_top_requested must be 10000 Pa")
    eta = np.asarray(_numbers(text, "domains", "eta_levels"), np.float64)
    if eta.size != 50 or eta[0] != 1.0 or eta[-1] != 0.0:
        raise ValueError("frozen eta_levels must have 50 full levels from 1 to 0")
    if not np.all(np.diff(eta) < 0.0):
        raise ValueError("frozen eta_levels are not strictly decreasing")
    return eta


def _frozen_d01_config():
    from gpuwm.config import RunConfig, validate_run_config

    return validate_run_config(RunConfig(
        nx=199, ny=199, nz=49,
        dx=2999.4213047435587, dy=2999.4213047435587,
        ztop=20000.0, dt=15.0, run_seconds=3600.0,
        time_step_sound=4, epssm=0.5,
        hybrid_opt=2, etac=0.2, moist=True, mp_physics=6,
        wsm6_hail_opt=0, moist_adv_opt=1,
        terrain_opt=1, map_proj=1, base_temp=290.0,
        specified=True, spec_bdy_width=5, spec_zone=1, relax_zone=4,
        km_opt=4, c_s=0.25,
        diff_6th_opt=2, diff_6th_factor=0.10,
        diff_6th_slopeopt=1, diff_6th_thresh=0.10,
        w_damping=1, damp_opt=3, zdamp=5000.0, dampcoef=0.2,
        h_sca_adv_order=5, emdiv=0.01, hypsometric_opt=2,
        sf_sfclay_physics=91, sf_surface_physics=2, bl_pbl_physics=1,
        ra_physics=0, ra_lw_physics=0, ra_sw_physics=1,
        radt=3.0, radt_minutes=3.0, bldt=0.0,
        cu_physics=0, cudt_minutes=0.0,
        output_interval_s=3600.0,
        case="hrrr_native_easy_d01_proof",
    ))


def _read_static(path: Path, *, expected_shape=(199, 199)
                 ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    from netCDF4 import Dataset

    names = (
        "HGT_M", "LANDMASK", "LU_INDEX", "SCT_DOM", "SOILTEMP",
        "SNOALB", "GREENFRAC", "LAI12M", "MAPFAC_M", "MAPFAC_U",
        "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA",
    )
    with Dataset(path) as dataset:
        missing = [name for name in names if name not in dataset.variables]
        if missing:
            raise KeyError(f"geo_em is missing fields: {missing}")
        values = {name: np.asarray(dataset.variables[name][0], np.float64)
                  for name in names}
        attrs = {name: getattr(dataset, name) for name in (
            "MMINLU", "ISWATER", "ISLAKE", "ISICE", "CEN_LAT")}
    shape = values["HGT_M"].shape
    expected_shape = tuple(int(value) for value in expected_shape)
    if shape != expected_shape:
        raise ValueError(
            f"geo shape must be {expected_shape}, got {shape}")
    for name in ("LANDMASK", "LU_INDEX", "SCT_DOM", "SOILTEMP", "SNOALB"):
        if values[name].shape != shape:
            raise ValueError(f"geo field {name} has shape {values[name].shape}")
    return values, attrs


def _proc_io() -> dict[str, int]:
    values = {}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip())
    except OSError:
        pass
    return values


def _io_delta(before, after):
    return {key: int(after.get(key, 0) - before.get(key, 0))
            for key in sorted(set(before) | set(after))}


def _source_identity() -> dict[str, object]:
    paths = (
        REPO / "gpuwm/ingest/hrrr.py",
        REPO / "gpuwm/ingest/real.py",
        REPO / "gpuwm/ingest/soil.py",
        REPO / "gpuwm/ingest/lateral_bc.py",
        REPO / "gpuwm/ingest/prepared_cache.py",
        REPO / "tools/hrrr_state_proof.py",
        REPO / "tools/hrrr_single_domain_benchmark.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"HRRR installed source helpers are missing: {missing}")
    # ``.as_posix()``, not ``str()``: these keys are SERIALIZED into the
    # proof and later looked up by the forward-slash names the reader
    # spells (``_HRRR_DECODE_SOURCES``).  ``str()`` of a relative Path
    # emits backslashes on Windows, so the proof named
    # ``gpuwm\ingest\hrrr.py`` and every one of those lookups missed.
    source_sha256 = {path.relative_to(REPO).as_posix(): _sha256(path)
                     for path in paths}
    # Manifest, then a genuine checkout of THIS tree, then the installed
    # wheel.  Same resolver as the benchmark's, for the same reason: the
    # bare `git rev-parse` this replaced exited 128 on every pip install.
    identity = runtime_manifest.provenance(REPO)
    manifest_sha256 = identity.pop("distribution_manifest_sha256", None)
    if manifest_sha256 is not None:
        source_sha256["distribution/manifest.json"] = manifest_sha256
    for key in ("installed_wheel", "installed_editable"):
        bound = identity.pop(key, None)
        if bound is not None:
            identity[key] = bound
    return {**identity, "source_sha256": source_sha256}


def _physics_update_counts(driver) -> dict[str, int]:
    """Read the public physics-driver counters used by the proof receipt."""
    return {
        "radiation_update_count": int(driver.radiation_callable.update_count),
        "microphysics_update_count": int(driver.microphysics_updates),
    }


def _physics_receipt(driver, cp) -> dict[str, object]:
    """Build every physics value serialized by the proof receipt."""
    return {
        "resolved_lw_sw": [
            int(driver.ra_lw_physics), int(driver.ra_sw_physics)],
        **_physics_update_counts(driver),
        "swdown_min_wm2": float(cp.min(driver.fields["swdown"]).get()),
        "swdown_max_wm2": float(cp.max(driver.fields["swdown"]).get()),
        "rainnc_max_mm": float(cp.max(driver.microphysics.rainnc).get()),
    }


def run(args) -> dict[str, object]:
    import resource
    import cupy as cp
    from dataclasses import asdict as dataclass_asdict

    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.dycore import stability_report, step
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.core.health import StateHealthValidator
    from gpuwm.core.landuse import initialize_landuse
    from gpuwm.core.physics import initialize_physics
    from gpuwm.ingest.hrrr import (
        interpolate_hrrr_to_lambert, load_hrrr_native_series)
    from gpuwm.ingest.lateral_bc import (
        attach_lateral_boundaries, build_state_lateral_boundaries)
    from gpuwm.ingest.real import initialize_real
    from gpuwm.ingest.soil import preprocess_noah_soil
    from gpuwm.static.build import monthly_interp_to_date
    from gpuwm.static.lambert import grids_from_wps_namelist

    eta = _assert_frozen_namelist(args.namelist_input)
    cfg = _frozen_d01_config()
    config_json = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    static, attrs = _read_static(args.geo_em)
    grids = grids_from_wps_namelist(args.namelist_wps)
    if len(grids) < 2:
        raise ValueError("frozen namelist.wps must contain d01 and d02")
    grid = grids[0]
    if ((grid.e_we, grid.e_sn, grid.dx, grid.dy)
            != (200, 200, cfg.dx, cfg.dy)):
        raise ValueError("namelist.wps d01 geometry differs from frozen config")

    device_properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    raw_name = device_properties["name"]
    device_name = (raw_name.decode(errors="replace")
                   if isinstance(raw_name, bytes) else str(raw_name))
    timings = {}
    memory = {}

    def checkpoint(name, started):
        cp.cuda.Stream.null.synchronize()
        timings[name] = time.perf_counter() - started
        free, total = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        memory[name] = {
            "device_used_bytes": int(total - free),
            "device_free_bytes": int(free),
            "device_total_bytes": int(total),
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_total_bytes": int(pool.total_bytes()),
        }

    io_before = _proc_io()
    total_started = time.perf_counter()
    started = time.perf_counter()
    snapshots = load_hrrr_native_series(
        args.bridge, (0, 1),
        expected_manifest_sha256=args.manifest_sha256)
    checkpoint("verify_manifest_and_map_f00_f01", started)

    results = []
    horizontal = []
    mapping_reports = []
    for snapshot in snapshots:
        mapping_report = {}
        started = time.perf_counter()
        met = interpolate_hrrr_to_lambert(
            snapshot, grid, target_landmask=static["LANDMASK"],
            soil_mapping_report=mapping_report)
        checkpoint(f"horizontal_f{snapshot.forecast_hour:02d}", started)
        started = time.perf_counter()
        coord = make_vertical_coord(
            cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
            eta_levels=eta)
        result = initialize_real(
            # ``grid=`` is the mp=28 aerosol front door, the same argument
            # the eleven production real routes pass, and the ONLY way this
            # proof reaches the shared resolver
            # (gpuwm.ingest.wif_climatology.resolve_wif_climatology).
            # Without it an aerosol-aware run here could not derive the
            # mass-point latitudes/longitudes a GLOBAL monthly dataset needs
            # and would take thompson_init's synthetic profile -- named, but
            # taken -- while the eleven routes beside it read the
            # climatology.  A proof whose initial condition differs from the
            # product's is not a proof of the product.  It is consulted
            # lazily, on one line inside the mp=28 block, so this proof's
            # frozen mp_physics=6 configuration executes not one extra
            # instruction.
            met, cfg, coord, static["HGT_M"], grid=grid, p_top=10000.0,
            sfcp_to_sfcp=True)
        result.state.set_map_coriolis(
            static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
            static["F"], static["E"], sina=static["SINALPHA"],
            cosa=static["COSALPHA"])
        checkpoint(f"state_f{snapshot.forecast_hour:02d}", started)
        results.append(result)
        horizontal.append(met)
        mapping_reports.append(mapping_report)

    started = time.perf_counter()
    boundaries = build_state_lateral_boundaries(
        [result.state for result in results],
        [snapshot.valid_time for snapshot in snapshots],
        spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone)
    attach_lateral_boundaries(results[0].state, boundaries)
    checkpoint("build_and_attach_f00_f01_lbc", started)

    started = time.perf_counter()
    soil = preprocess_noah_soil(
        horizontal[0].fields, soil_type=static["SCT_DOM"],
        deep_soil_temperature=static["SOILTEMP"])
    state = results[0].state
    update_diagnostics(state, cfg.hypsometric_opt)
    valid_time = snapshots[0].valid_time
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"], snow=soil.snow_water, xice=soil.xice,
        valid_time=valid_time, cen_lat=float(attrs["CEN_LAT"]),
        mminlu=str(attrs["MMINLU"]), iswater=int(attrs["ISWATER"]),
        islake=int(attrs["ISLAKE"]), isice=int(attrs["ISICE"]),
        fractional_seaice=True,
        # real.exe's landmask/soil-category reconciliation decides a
        # disagreeing column from its soil temperature, then its SST.
        soil_temperature=soil.soil_temperature)
    vegfra = 100.0 * monthly_interp_to_date(static["GREENFRAC"], valid_time)
    lai = monthly_interp_to_date(static["LAI12M"], valid_time)
    lat, lon = grid.latlon_mass()
    # The frozen proof namelist runs Noah with ra_lw_physics 0 (asserted
    # by _assert_frozen_namelist above), so nothing computes downward
    # longwave and initialize_physics refuses to invent one.  This
    # harness is bound to that historical configuration by design; the
    # constant is DECLARED here -- 1.8.7's value, so the proof's
    # trajectory receipts do not move -- rather than defaulted.
    from gpuwm.core.physics import DECLARED_CONSTANT_GLW_WM2
    driver = initialize_physics(
        state, cfg, landuse=landuse, tsk=soil.tsk,
        soil_temperature=soil.soil_temperature,
        soil_moisture=soil.soil_moisture,
        liquid_moisture=soil.liquid_moisture,
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=soil.deep_soil_temperature,
        xice=soil.xice, snow=soil.snow_water, snow_depth=soil.snow_depth,
        glw=DECLARED_CONSTANT_GLW_WM2,
        radiation_start_time=valid_time, radiation_latitude=lat,
        radiation_longitude=lon)
    driver.fields["snoalb"][...] = cp.asarray(
        static["SNOALB"] / 100.0, dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    met0 = horizontal[0].fields
    driver.fields["psfc"][...] = cp.asarray(
        results[0].surface_pressure, dtype=cp.float32)
    driver.fields["t2"][...] = met0["T2"]
    driver.fields["q2"][...] = cp.asarray(
        results[0].surface_qv, dtype=cp.float32)
    driver.fields["th2"][...] = (
        driver.fields["t2"]
        * (cp.float32(100000.0) / driver.fields["psfc"])
        ** cp.float32(287.0 / 1004.0))
    driver.fields["u10"][...] = 0.5 * (met0["U10"][:, :-1]
                                         + met0["U10"][:, 1:])
    driver.fields["v10"][...] = 0.5 * (met0["V10"][:-1]
                                         + met0["V10"][1:])
    checkpoint("soil_landuse_and_physics", started)

    initialized_health = StateHealthValidator(state).validate(
        phase="native-hrrr-initialized")
    if not initialized_health.ok:
        raise FloatingPointError(
            f"initialized state health failed: {initialized_health}")
    started = time.perf_counter()
    step(state, cfg)
    checkpoint("first_full_gpu_step", started)
    stepped_health = StateHealthValidator(state).validate(
        phase="native-hrrr-after-first-step")
    if not stepped_health.ok:
        raise FloatingPointError(
            f"first-step state health failed: {stepped_health}")
    stability = stability_report(
        state, cfg, boundary_width=cfg.spec_bdy_width)
    if stability["nan"]:
        raise FloatingPointError("first-step compact stability report is non-finite")

    cp.cuda.Stream.null.synchronize()
    total_wall = time.perf_counter() - total_started
    io_after = _proc_io()
    bridge_bytes = sum(
        path.stat().st_size for path in args.bridge.rglob("*") if path.is_file())
    report = {
        "schema": "gpuwm-native-hrrr-state-proof-v1",
        "status": "PASS",
        "scope": (
            "f00/f01 d01 native state + external LBC + one full easy-physics "
            "GPU timestep; d02/nested forecast skill not claimed"),
        "device": {"name": device_name, "id": int(cp.cuda.Device().id)},
        "timing_seconds": timings,
        "total_wall_seconds": total_wall,
        "gpu_memory_checkpoints": memory,
        "cpu_peak_rss_bytes": int(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss * 1024),
        "process_io_delta_bytes": _io_delta(io_before, io_after),
        "bridge_payload_bytes": bridge_bytes,
        "bridge_manifest_sha256": args.manifest_sha256,
        "source_identity": _source_identity(),
        "configuration": {
            "run_config": asdict(cfg),
            "run_config_sha256": hashlib.sha256(
                config_json.encode()).hexdigest(),
            "namelist_input": str(args.namelist_input.resolve()),
            "namelist_input_sha256": _sha256(args.namelist_input),
            "namelist_wps": str(args.namelist_wps.resolve()),
            "namelist_wps_sha256": _sha256(args.namelist_wps),
            "geo_em": str(args.geo_em.resolve()),
            "geo_em_sha256": _sha256(args.geo_em),
            "eta_levels": eta.tolist(),
            "p_top_pa": 10000.0,
        },
        "soil_mapping": {
            "f00": mapping_reports[0], "f01": mapping_reports[1]},
        "lateral_boundaries": {
            "interval_count": len(boundaries.intervals),
            "spec_bdy_width": boundaries.spec_bdy_width,
            "spec_zone": boundaries.spec_zone,
            "relax_zone": boundaries.relax_zone,
            "forcing_start": snapshots[0].valid_time.isoformat(),
            "forcing_end": snapshots[1].valid_time.isoformat(),
        },
        "state_health": {
            "initialized": dataclass_asdict(initialized_health),
            "after_first_step": dataclass_asdict(stepped_health),
            "stability": stability,
            "elapsed_seconds": float(state.elapsed_seconds),
        },
        "physics": _physics_receipt(driver, cp),
    }
    # THE AEROSOL SOURCE, said in the proof rather than only on stderr.
    # Empty today and correctly so: this proof's frozen configuration is
    # mp_physics=6 and WSM6 has no water-friendly/ice-friendly aerosol
    # number fields, so there is no source to name and the document is
    # byte-for-byte what it was.  Wired anyway, beside the ``grid=`` above,
    # because the pair is what makes an aerosol-aware configuration here
    # BOTH read the dataset and say that it did; a route that resolves
    # without reporting is how a run reads one initial condition and
    # publishes a receipt that never mentions it.
    report.update(aerosol_source_report_entry(
        results[0].aerosol_initialization,
        mp_physics=cfg.mp_physics,
        when_unrecorded=(
            "this proof initializes through gpuwm.ingest.real."
            "initialize_real, which always emits the receipt for an "
            "aerosol-aware scheme; an empty one here means the state was "
            "built by some other path and the proof should be re-read "
            "before its aerosol claim is trusted")))
    return _strict_json(report)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--namelist-wps", type=Path, required=True)
    parser.add_argument("--namelist-input", type=Path, required=True)
    parser.add_argument("--geo-em", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        report = run(args)
    except BaseException as error:
        report = {
            "schema": "gpuwm-native-hrrr-state-proof-v1",
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        args.output.write_text(
            json.dumps(_strict_json(report), indent=2, sort_keys=True) + "\n")
        raise
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "total_wall_seconds": report["total_wall_seconds"],
        "first_full_gpu_step_seconds": report["timing_seconds"][
            "first_full_gpu_step"],
        "device": report["device"]["name"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
