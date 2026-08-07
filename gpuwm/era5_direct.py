"""ERA5 GRIB1 -> prepared cache -> stock-WRF direct-export proof adapter.

This adapter uses gpuwm's existing all-Rust GRIB1 decoder and GPU real-data
initialization.  WPS and ``real.exe`` are never invoked.  It remains an
uncertified proof lane until an unchanged stock ``wrf.exe`` accepts the
products emitted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Mapping

import netCDF4
import numpy as np

from gpuwm.case_data import (
    PerDomainSourceOrography,
    SourceOrography,
    SourceOrographyDeclaration,
    resolve_source_orography,
)
from gpuwm.experiment import load_experiment
from gpuwm.ingest.grib import (
    cached_era5_snapshots,
    canonical_units,
    parse_vtable,
)
from gpuwm.ingest.horiz import interpolate_era5_to_lambert
from gpuwm.ingest.lateral_bc import (
    StateBoundaryFrames,
    attach_lateral_boundaries,
)
from gpuwm.ingest.prepared_cache import (
    prepared_cache_identity,
    write_prepared_cache,
)
from gpuwm.ingest.preprocess_backend import (
    release_backend_memory,
    resolve_preprocess_backend,
)
from gpuwm.ingest.real import initialize_real
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.ingest.water_overlay import (
    load_water_temperature_overlay,
    overlay_snapshots,
)
from gpuwm.native_domain_artifacts import _atomic_staging_sibling
from gpuwm.native_wrf_contract import (
    NATIVE_STATIC_REQUIRED,
    canonical_noah_surface,
    load_native_static_cache,
    validate_native_lambert_contract,
    validate_native_lambert_contracts,
    validate_native_static_fields,
    verify_native_static_receipt,
    write_native_geometry_receipt,
    write_native_static_cache,
)
from gpuwm.source_hierarchy import (
    initialize_and_export_regular_source_hierarchy,
)
from gpuwm.hrrr_native_static import verified_static_catalog
from gpuwm.static.build import build_static_for_domain
from gpuwm.wrf_direct import export_prepared_wrf
from gpuwm.vertical_contract import validate_explicit_eta_grid


INPUT_MANIFEST_SCHEMA = "gpuwm-era5-direct-input-manifest-v1"
_STATIC_REQUIRED = NATIVE_STATIC_REQUIRED


def _domain_source_orography(
        bindings: list[str] | None, variable: str,
) -> PerDomainSourceOrography | None:
    """Parse repeatable ``dNN=PATH`` hierarchy bindings fail-closed."""

    if not bindings:
        return None
    if not variable:
        raise ValueError("source-orography variable must be non-empty")
    parsed: list[tuple[int, SourceOrography]] = []
    seen: set[int] = set()
    for binding in bindings:
        label, separator, raw_path = binding.partition("=")
        if (not separator or len(label) < 2 or label[0].lower() != "d"
                or not label[1:].isdigit() or not raw_path):
            raise ValueError(
                "domain source-orography bindings must use dNN=PATH")
        domain_id = int(label[1:])
        if domain_id < 1:
            raise ValueError("domain source-orography ids must be positive")
        if domain_id in seen:
            raise ValueError(
                f"duplicate domain source-orography binding for d{domain_id:02d}")
        seen.add(domain_id)
        parsed.append((domain_id, SourceOrography(Path(raw_path), variable)))
    parsed.sort(key=lambda item: item[0])
    return PerDomainSourceOrography(tuple(parsed))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False)


def _verify_input_manifest(
    manifest_path: Path,
    expected_manifest_sha256: str,
    role_paths: Mapping[str, Path],
) -> dict[str, object]:
    observed_manifest_sha256 = _sha256(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256.lower():
        raise ValueError(
            "ERA5 input manifest SHA mismatch: expected "
            f"{expected_manifest_sha256}, got {observed_manifest_sha256}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise ValueError("unsupported ERA5 direct-input manifest schema")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or set(declared) != set(role_paths):
        raise ValueError("ERA5 input manifest role inventory differs from request")
    for role, path in role_paths.items():
        spec = declared[role]
        if not isinstance(spec, dict) or spec.get("name") != path.name:
            raise ValueError(f"ERA5 input manifest path mismatch for {role}")
        observed = _sha256(path)
        if spec.get("sha256") != observed:
            raise ValueError(
                f"ERA5 input digest mismatch for {role}: "
                f"expected {spec.get('sha256')}, got {observed}")
    return manifest


def _validated_static(
        fields: Mapping[str, object], grid, ny: int, nx: int,
) -> dict[str, np.ndarray]:
    return validate_native_static_fields(fields, grid, ny, nx)


def _load_static(path: Path, grid, ny: int, nx: int) -> dict[str, np.ndarray]:
    return load_native_static_cache(path, grid, ny, nx)


def _static_from_geog(
        wps_namelist: Path, geog_root: Path, grid, cfg,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    catalog, receipt = verified_static_catalog(
        Path(wps_namelist), Path(geog_root), (1,))
    fields = build_static_for_domain(grid, catalog, 1)
    return _validated_static(fields, grid, cfg.ny, cfg.nx), receipt


def _write_static_cache(path: Path, fields: Mapping[str, np.ndarray]) -> None:
    write_native_static_cache(path, fields)


def _write_geometry_receipt(path: Path, grid, cfg, static_path: Path) -> None:
    write_native_geometry_receipt(path, grid, cfg, static_path)


def _canonical_surface(soil) -> dict[str, np.ndarray]:
    return canonical_noah_surface(soil)


def _load_source_orography(path: Path, variable: str) -> np.ndarray:
    with netCDF4.Dataset(path) as dataset:
        if variable not in dataset.variables:
            raise KeyError(f"source-orography file lacks {variable!r}")
        value = np.asarray(dataset.variables[variable][0], dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("source orography contains non-finite values")
    return value


def prepare_era5_wrf(
    *,
    grib: Path,
    vtable: Path,
    bridge: Path,
    wps_namelist: Path,
    static_input: Path | None,
    static_receipt: Path | None,
    source_orography: Path | None,
    source_orography_variable: str,
    experiment_config: Path,
    input_manifest: Path,
    input_manifest_sha256: str,
    output_root: Path,
    preprocess_backend: str = "cuda",
    preprocess_workers: int | None = None,
    cpu_preprocess_bridge: Path | None = None,
    geog_root: Path | None = None,
    hierarchy_workers: int | None = None,
    hierarchy_source_orography: SourceOrographyDeclaration | None = None,
    water_temperature_overlay: Path | None = None,
) -> dict[str, object]:
    """Build native ERA5 initial/boundary files and return the proof receipt.

    Multi-domain internal callers must supply ``geog_root`` and a complete
    per-domain source-orography declaration.  The legacy d01 field is never
    reused on a child.
    """

    paths = {
        "grib": Path(grib),
        "vtable": Path(vtable),
        "bridge": Path(bridge),
        "wps_namelist": Path(wps_namelist),
        "experiment_config": Path(experiment_config),
    }
    if source_orography is not None:
        paths["source_orography"] = Path(source_orography)
    if water_temperature_overlay is not None:
        paths["water_temperature_overlay"] = Path(water_temperature_overlay)
    static_pair = static_input is not None or static_receipt is not None
    if static_pair and (static_input is None or static_receipt is None):
        raise ValueError(
            "ERA5 static-input and static-receipt must be supplied together")
    if static_input is not None:
        paths["static_input"] = Path(static_input)
        if not Path(static_receipt).is_file():
            raise FileNotFoundError(
                f"missing ERA5 adapter input static_receipt: {static_receipt}")
    elif geog_root is None:
        raise ValueError(
            "ERA5 requires either static-input/static-receipt or geog-root")
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing ERA5 adapter input {role}: {path}")
    if not os.access(paths["bridge"], os.X_OK):
        raise PermissionError("ERA5 Rust bridge is not executable")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    manifest = _verify_input_manifest(
        Path(input_manifest), input_manifest_sha256, paths)
    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_preprocess_bridge)
    preprocess_receipt = preprocess.receipt()

    total_started = time.perf_counter()
    exp = load_experiment(paths["experiment_config"])
    from gpuwm.experiment import (
        refuse_unrouted_perturbation, refuse_unrouted_spawn,
    )
    refuse_unrouted_perturbation(exp, "ERA5-direct prepared-cache")
    refuse_unrouted_spawn(exp, "ERA5-direct prepared-cache")
    from gpuwm.static.highres_production import refuse_inert_highres
    refuse_inert_highres(paths["experiment_config"],
                         lane="ERA5-direct adapter")
    cfg = exp.root.run
    if len(exp.domains) == 1:
        grid = validate_native_lambert_contract(
            exp, paths["wps_namelist"], source_name="ERA5")
        grids = (grid,)
    else:
        if geog_root is None:
            raise ValueError(
                "nested ERA5 preparation requires an explicit geog_root")
        if hierarchy_source_orography is not None:
            if not isinstance(
                    hierarchy_source_orography, PerDomainSourceOrography):
                raise ValueError(
                    "nested ERA5 explicit source_orography requires one "
                    "artifact per domain")
            if source_orography is None:
                raise ValueError(
                    "per-domain ERA5 source_orography requires the matching "
                    "root source-orography input")
            root_orography = resolve_source_orography(
                hierarchy_source_orography, 1)
            if (Path(root_orography.path).resolve()
                    != paths["source_orography"].resolve()
                    or root_orography.variable != source_orography_variable):
                raise ValueError(
                    "nested ERA5 d01 source_orography differs from the root "
                    "preparation input")
        grids = validate_native_lambert_contracts(
            exp, paths["wps_namelist"], source_name="ERA5")
        grid = grids[0]
    if static_input is None:
        static, root_static_receipt = _static_from_geog(
            paths["wps_namelist"], Path(geog_root), grid, cfg)
        root_static_provider = "native-wps-geog"
    else:
        root_static_receipt = verify_native_static_receipt(
            Path(static_receipt), paths["static_input"], grid, cfg)
        static = _load_static(paths["static_input"], grid, cfg.ny, cfg.nx)
        root_static_provider = "prebuilt-hash-bound-cache"
    source_terrain = (
        None if source_orography is None
        else _load_source_orography(
            paths["source_orography"], source_orography_variable)
    )

    decode_started = time.perf_counter()
    snapshots = cached_era5_snapshots(
        paths["grib"], paths["vtable"], bridge=paths["bridge"])
    decode_seconds = time.perf_counter() - decode_started
    snapshots = tuple(sorted(snapshots, key=lambda value: value.valid_time))
    # Optional hi-res water-temperature overlay (task #71): applied to
    # the snapshots that feed BOTH the single-domain loop and the nested
    # hierarchy, before any horizontal interpolation.  Absent, this is
    # the identity: the same tuple object flows onward.
    water_overlay = (
        None if water_temperature_overlay is None
        else load_water_temperature_overlay(
            paths["water_temperature_overlay"]))
    snapshots, water_overlay_receipt = overlay_snapshots(
        snapshots, water_overlay)
    source_inventory = tuple(snapshots[0].fields) if snapshots else ()
    source_units = dict(canonical_units(parse_vtable(paths["vtable"])))
    has_invariant_orography = (
        bool(snapshots)
        and all("SOILGEO" in snapshot.fields for snapshot in snapshots)
    )
    if source_terrain is None and not has_invariant_orography:
        raise ValueError(
            "ERA5 input requires invariant SOILGEO or an explicit "
            "source-orography artifact")
    if source_terrain is not None and any(
            "SOILGEO" in snapshot.fields for snapshot in snapshots):
        raise ValueError(
            "ERA5 input declares both invariant SOILGEO and explicit "
            "source-orography")
    source_orography_catalog = (
        SimpleNamespace(
            snapshots=snapshots,
            inventory=source_inventory,
            units=source_units,
        )
        if has_invariant_orography else None
    )
    times = tuple(value.valid_time for value in snapshots)
    if not times or times[0] != exp.start_time:
        raise ValueError(
            f"ERA5 forcing must begin at {exp.start_time}, got {times[:1]}")
    forcing_hours = [
        int((value - times[0]).total_seconds() // 3600) for value in times]
    if any(
            (value - times[0]).total_seconds() != hour * 3600
            for value, hour in zip(times, forcing_hours)):
        raise ValueError("ERA5 direct forcing offsets must be whole hours")
    if forcing_hours[-1] * 3600 < exp.run_seconds:
        raise ValueError("ERA5 forcing does not cover the configured run")
    if len(forcing_hours) < 2:
        raise ValueError("ERA5 direct forcing requires at least two times")
    deltas = {
        later - earlier
        for earlier, later in zip(forcing_hours, forcing_hours[1:])}
    if len(deltas) != 1:
        raise ValueError("ERA5 direct forcing cadence must be uniform")
    boundary_interval_seconds = deltas.pop() * 3600

    source_top_pressure_pa = max(
        float(np.min(snapshot.levels_hpa) * 100.0)
        for snapshot in snapshots
    )
    if len(exp.domains) == 1:
        validate_explicit_eta_grid(
            exp.vertical.eta_levels,
            nz=cfg.nz,
            p_top=exp.vertical.p_top,
            source_top_pressure_pa=source_top_pressure_pa,
            context="ERA5 direct adapter",
        )
    else:
        grids = validate_native_lambert_contracts(
            exp, paths["wps_namelist"], source_name="ERA5",
            source_top_pressure_pa=source_top_pressure_pa)
        grid = grids[0]

    from gpuwm.core.grid import make_vertical_coord

    initialize_started = time.perf_counter()
    # Only the first time's met/state survive this loop; every later time
    # contributes its perimeter frames and is released at once.  Holding
    # all of them is what made ingest peak above the forecast it prepares.
    initial_result = None
    initial_met = None
    forcing = StateBoundaryFrames(
        spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    for index, source in enumerate(snapshots):
        met = interpolate_era5_to_lambert(
            source, grid,
            source_orography_catalog=source_orography_catalog,
            backend=preprocess,
            target_landmask=np.asarray(static["LANDMASK"]) >= 0.5)
        coord = make_vertical_coord(
            cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
            eta_levels=exp.vertical.eta_levels)
        initialized = initialize_real(
            met, cfg, coord, static["HGT_M"],
            source_orography=source_terrain,
            p_top=exp.vertical.p_top, sfcp_to_sfcp=True,
            preprocess_backend=preprocess,
            state_backend="preprocess")
        initialized.state.set_map_coriolis(
            static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
            static["F"], static["E"], sina=static["SINALPHA"],
            cosa=static["COSALPHA"])
        forcing.add_state(initialized.state)
        if index == 0:
            initial_met = met
            initial_result = initialized
        else:
            del met, initialized, coord
            release_backend_memory(preprocess)
    boundaries = forcing.build(times)
    attach_lateral_boundaries(initial_result.state, boundaries)

    # No lake skin override: metgrid's masked=both SKINTEMP chain with
    # static-landmask targets already yields water-source skin at lakes,
    # matching real.exe's no-TAVGSFC behavior (SST only where valid).  The
    # router forwards this exact argument list to preprocess_noah_soil for
    # Noah-geometry schemes, so their soil state is unchanged by the LSM
    # dispatch seam.
    soil = preprocess_land_surface_soil(
        initial_met.fields,
        sf_surface_physics=int(cfg.sf_surface_physics),
        soil_type=static["SCT_DOM"],
        deep_soil_temperature=static["TMN"],
        landmask=static["LANDMASK"],
        terrain=static["HGT_M"],
        source_orography=source_terrain)
    # LOUD when it happened, silent when it did not: the SMCDRY floor
    # receipt is only bound into the prepared metadata/proof when at
    # least one land cell was floored (getattr: the RUC state carries
    # its own moisture and has no Noah floor receipt).
    soil_moisture_floor = dict(getattr(soil, "moisture_floor", {}) or {})
    soil_floor_binding = (
        {"soil_moisture_floor": soil_moisture_floor}
        if soil_moisture_floor else {})
    initialize_seconds = time.perf_counter() - initialize_started
    input_manifest_digest = _sha256(Path(input_manifest))
    # LOUD when configured, absent when not: the overlay binding joins
    # the source identity only when an overlay was applied, so the
    # prepared cache identity moves exactly when the data does.
    water_overlay_binding = (
        {} if water_overlay_receipt is None
        else {"water_temperature_overlay": {
            **water_overlay_receipt,
            "sha256": _sha256(paths["water_temperature_overlay"]),
        }})
    native_source_identity = {
        "adapter": "era5-grib1-direct-v1",
        "input_manifest_schema": manifest["schema"],
        "input_manifest_sha256": input_manifest_digest,
        "decoder": {
            "name": paths["bridge"].name,
            "sha256": _sha256(paths["bridge"]),
            "implementation": "gpuwm-all-rust-grib1-bridge",
        },
        "preprocessing": preprocess_receipt,
        **water_overlay_binding,
    }

    staging = _atomic_staging_sibling(output_root)
    if staging.exists():
        raise FileExistsError(f"refusing stale ERA5 staging path {staging}")
    staging.mkdir(parents=True)
    try:
        static_cache = staging / "native-static.npz"
        geometry_receipt = staging / "geometry-receipt.json"
        portable_source_manifest = staging / "source-input-manifest.json"
        prepared_cache = staging / "prepared-cache"
        wrf_output = staging / "wrf-native-input"
        shutil.copy2(Path(input_manifest), portable_source_manifest)
        _write_static_cache(static_cache, static)
        _write_geometry_receipt(geometry_receipt, grid, cfg, static_cache)
        if len(exp.domains) > 1:
            selected_workers = hierarchy_workers
            if selected_workers is None:
                selected_workers = (
                    8 if preprocess_receipt["backend"] == "cpu" else 1)
            hierarchy_started = time.perf_counter()
            hierarchy = initialize_and_export_regular_source_hierarchy(
                exp=exp, grids=grids, snapshots=snapshots,
                forcing_hours=forcing_hours,
                wps_namelist=paths["wps_namelist"],
                geog_root=Path(geog_root), source_name="ERA5",
                artifact_output=staging / "hierarchy-artifacts",
                wrf_output=staging / "wrf-native-input",
                root_initial_result=initial_result, root_met=initial_met,
                root_soil=soil, root_static_fields=static,
                root_boundaries=boundaries,
                bridge_manifest_sha256=input_manifest_digest,
                source_manifest_sha256=input_manifest_digest,
                namelist_sha256=_sha256(paths["experiment_config"]),
                source_identity=native_source_identity,
                source_orography=hierarchy_source_orography,
                source_inventory=source_inventory,
                source_units=source_units,
                workers=selected_workers,
                preprocess_backend=preprocess_receipt["backend"],
                cpu_bridge=cpu_preprocess_bridge,
                input_provenance={
                    "input_manifest_sha256": input_manifest_digest,
                    "decoder_sha256": _sha256(paths["bridge"]),
                    "preprocessing": preprocess_receipt,
                },
                artifact_manifest_reference=(
                    "../hierarchy-artifacts/domain-artifacts.json"),
            )
            hierarchy_seconds = time.perf_counter() - hierarchy_started
            final_manifest = _verify_input_manifest(
                Path(input_manifest), input_manifest_digest, paths)
            if final_manifest != manifest:
                raise ValueError(
                    "ERA5 input manifest content changed during preparation")
            proof = {
                "schema": "gpuwm-era5-native-hierarchy-proof-v1",
                "status": "READY_NOT_YET_STOCK_WRF_GATED",
                "domain_count": len(exp.domains),
                "forcing_times": [value.isoformat() for value in times],
                "forcing_hours": forcing_hours,
                "boundary_interval_seconds": boundary_interval_seconds,
                "input_manifest_sha256": input_manifest_digest,
                "decoder_sha256": _sha256(paths["bridge"]),
                "preprocessing": preprocess_receipt,
                **soil_floor_binding,
                **water_overlay_binding,
                "root_static_provider": root_static_provider,
                "root_static_receipt": root_static_receipt,
                "hierarchy_workers": selected_workers,
                "static_catalog": dict(hierarchy.static_catalog_receipt),
                "source_coverage": dict(
                    hierarchy.source_coverage_receipt),
                "hierarchy_topology": dict(hierarchy.topology_receipt),
                "artifact_receipt": dict(
                    hierarchy.hierarchy.artifacts.receipt),
                "wrf_manifest": dict(hierarchy.hierarchy.wrf_manifest),
                "timing_seconds": {
                    "decode": decode_seconds,
                    "initialize_all_root_times": initialize_seconds,
                    **dict(hierarchy.hierarchy.timings_seconds),
                    "hierarchy_call_wall": hierarchy_seconds,
                    "total": time.perf_counter() - total_started,
                },
            }
            (staging / "proof.json").write_text(
                json.dumps(proof, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            os.replace(staging, output_root)
            return proof
        identity = prepared_cache_identity(
            bridge_manifest_sha256=input_manifest_digest,
            source_manifest_sha256=input_manifest_digest,
            static_cache_sha256=_sha256(static_cache),
            namelist_sha256=_sha256(paths["experiment_config"]),
            domain_config=exp.root,
            forcing_hours=forcing_hours,
            source_identity=native_source_identity,
        )
        cache_started = time.perf_counter()
        cache_receipt = write_prepared_cache(
            prepared_cache, identity=identity,
            initial_result=initial_result, met=initial_met,
            boundaries=boundaries, surface=_canonical_surface(soil),
            metadata={
                "source_adapter": "era5",
                "initial_valid_time": times[0].isoformat(),
                "last_valid_time": times[-1].isoformat(),
                "forcing_hours": forcing_hours,
                "boundary_interval_seconds": boundary_interval_seconds,
                "preprocessing": preprocess_receipt,
                **soil_floor_binding,
            },
        )
        cache_seconds = time.perf_counter() - cache_started
        portable_cache_receipt = dict(cache_receipt)
        portable_cache_receipt["path"] = prepared_cache.name
        export_started = time.perf_counter()
        export_receipt = export_prepared_wrf(
            prepared_cache, static_cache, geometry_receipt, wrf_output,
            valid_time=times[0],
            boundary_interval_seconds=boundary_interval_seconds)
        export_seconds = time.perf_counter() - export_started
        preprocessing_receipt_sha256 = hashlib.sha256(
            _canonical(preprocess_receipt).encode("utf-8")).hexdigest()
        initialization_artifacts = {
            "source_manifest": {
                "path": portable_source_manifest.name,
                "bytes": portable_source_manifest.stat().st_size,
                "sha256": _sha256(portable_source_manifest),
            },
            "static_cache": {
                "path": static_cache.name,
                "bytes": static_cache.stat().st_size,
                "sha256": _sha256(static_cache),
            },
            "geometry_receipt": {
                "path": geometry_receipt.name,
                "bytes": geometry_receipt.stat().st_size,
                "sha256": _sha256(geometry_receipt),
            },
            "prepared_cache": {
                "path": prepared_cache.name,
                "content_sha256": portable_cache_receipt["content_sha256"],
                "payload_bytes": portable_cache_receipt["payload_bytes"],
            },
            "wrf_files": {
                name: {
                    "path": f"{wrf_output.name}/{name}",
                    **details,
                }
                for name, details in export_receipt["files"].items()
            },
        }
        proof = {
            "schema": "gpuwm-era5-direct-wrf-proof-v2",
            "status": "READY_NOT_YET_STOCK_WRF_GATED",
            "forcing_times": [value.isoformat() for value in times],
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_interval_seconds,
            "input_manifest_sha256": input_manifest_digest,
            "decoder_sha256": _sha256(paths["bridge"]),
            "preprocessing": preprocess_receipt,
            "preprocessing_receipt_sha256": preprocessing_receipt_sha256,
            **soil_floor_binding,
            **water_overlay_binding,
            "source_inputs": {
                "manifest_schema": manifest["schema"],
                "manifest_sha256": input_manifest_digest,
                "files": manifest["files"],
            },
            "initialization_artifacts": initialization_artifacts,
            "prepared_cache": portable_cache_receipt,
            "export": export_receipt,
            "timing_seconds": {
                "decode": decode_seconds,
                "initialize_all_times": initialize_seconds,
                "write_prepared_cache": cache_seconds,
                "direct_wrf_export": export_seconds,
                "total": time.perf_counter() - total_started,
            },
        }
        (staging / "proof.json").write_text(
            json.dumps(proof, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output_root)
        return proof
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grib", type=Path, required=True)
    parser.add_argument("--vtable", type=Path, required=True)
    parser.add_argument(
        "--bridge", type=Path, required=True,
        help="prebuilt executable all-Rust GRIB1 bridge bound by the manifest",
    )
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument("--static-input", type=Path)
    parser.add_argument("--static-receipt", type=Path)
    parser.add_argument("--source-orography", type=Path)
    parser.add_argument("--source-orography-variable", default="SOILHGT")
    parser.add_argument(
        "--domain-source-orography", action="append", metavar="DNN=PATH",
        help=(
            "repeat once per ERA5 hierarchy domain; paths are target-grid "
            "orography artifacts using --source-orography-variable"),
    )
    parser.add_argument(
        "--water-temperature-overlay", type=Path, metavar="ANALYSIS",
        help=(
            "optional high-resolution water-temperature analysis (netCDF "
            "or GRIB2) replacing ERA5 SST/SKINTEMP over water source "
            "cells; off by default"),
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="cuda")
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    parser.add_argument("--geog-root", type=Path)
    parser.add_argument("--hierarchy-workers", type=int)
    args = parser.parse_args(argv)
    try:
        hierarchy_source_orography = _domain_source_orography(
            args.domain_source_orography,
            args.source_orography_variable,
        )
    except ValueError as error:
        parser.error(str(error))
    proof = prepare_era5_wrf(
        grib=args.grib, vtable=args.vtable, bridge=args.bridge,
        wps_namelist=args.wps_namelist,
        static_input=args.static_input,
        static_receipt=args.static_receipt,
        source_orography=args.source_orography,
        source_orography_variable=args.source_orography_variable,
        experiment_config=args.experiment_config,
        input_manifest=args.input_manifest,
        input_manifest_sha256=args.input_manifest_sha256,
        output_root=args.output_root,
        preprocess_backend=args.preprocess_backend,
        preprocess_workers=args.preprocess_workers,
        cpu_preprocess_bridge=args.cpu_preprocess_bridge,
        geog_root=args.geog_root,
        hierarchy_workers=args.hierarchy_workers,
        hierarchy_source_orography=hierarchy_source_orography,
        water_temperature_overlay=args.water_temperature_overlay,
    )
    print(json.dumps(proof, indent=2, sort_keys=True))
    # Parity with the GFS and 20CRv3 front doors, and the last silent
    # one of the three the prepared-forecast runners actually support:
    # a complete, hash-bound run command on stderr, so the proof
    # document on stdout stays machine-readable.
    from gpuwm.gfs_direct import prepared_forecast_next_command

    for line in prepared_forecast_next_command(
            proof, output_root=args.output_root,
            experiment_config=args.experiment_config,
            wps_namelist=args.wps_namelist, source="era5"):
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
