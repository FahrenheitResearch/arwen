"""Mapped-source initialization and atomic stock-WRF hierarchy export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from types import SimpleNamespace
from typing import Mapping, Sequence
import uuid

import numpy as np

from gpuwm.core.grid import make_vertical_coord
from gpuwm.experiment import load_experiment, validate_boundary_timing
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
from gpuwm.mapped_composition import (
    MappedSourceBundle,
    _decoder_inventory,
    _path_inventory,
    decode_composed_source,
    mapped_composition_receipt,
)
from gpuwm.mapped_source import (
    _load_json_bytes,
    _require_authority_snapshot,
    _sha256,
    _snapshot_authority,
    load_mapping,
)
from gpuwm.native_wrf_contract import (
    canonical_noah_surface,
    native_static_export_fields,
    validate_native_lambert_contract,
    validate_native_lambert_contracts,
    write_native_geometry_receipt,
    write_native_static_cache,
)
from gpuwm.source_hierarchy import (
    initialize_and_export_regular_source_hierarchy,
)
from gpuwm.static.build import GeogSelection, build_static
from gpuwm.vertical_contract import validate_explicit_eta_grid
from gpuwm.wrf_direct import export_prepared_wrf


PROOF_SCHEMA = "gpuwm-mapped-direct-wrf-proof-v1"
HIERARCHY_PROOF_SCHEMA = "gpuwm-mapped-native-hierarchy-proof-v1"


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _file_receipt(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _copy_bound_authority(
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    """Copy one small evidence authority and prove the copied bytes."""

    shutil.copy2(source, destination)
    if _sha256(destination) != expected_sha256:
        raise ValueError(
            f"mapped evidence changed before publication: {source}"
        )


def _provenance_evidence_name(role: str, suffix: str) -> str:
    """Return a short deterministic filename for one provenance role.

    Composition role names remain recorded in the bound composition receipt;
    repeating an arbitrary role verbatim in the filesystem only burns scarce
    Windows path budget.  Sixteen SHA-256 hex digits keep names stable and
    collision-resistant, while the create-only destination check below still
    fails closed if two roles ever collide.
    """

    role_digest = hashlib.sha256(role.encode("utf-8")).hexdigest()[:16]
    return f"provenance-{role_digest}{suffix}"


def _bound_decoder_receipts(
    paths: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Recheck small decoder executables before emitting proof evidence."""

    if set(paths) != set(expected_sha256):
        raise ValueError("decoded decoder inventory is internally inconsistent")
    receipts = {}
    for role, path in paths.items():
        receipt = _file_receipt(path)
        if receipt["sha256"] != expected_sha256[role]:
            raise ValueError(f"mapped decoder changed after decode: {role}")
        receipts[role] = receipt
    return receipts


def _validate_target_contract(
    mapping: Mapping[str, object],
    exp,
    boundary_interval_seconds: int | None,
    *,
    hierarchy: bool,
) -> dict[str, object]:
    """Bind mapped target limits to one resolved experiment and cadence."""

    target = mapping["target"]
    if not isinstance(target, dict):
        raise TypeError("mapping target must be an object")
    domain_count = len(exp.domains)
    max_dom = target["max_dom"]
    if domain_count > max_dom:
        raise ValueError(
            f"mapped target allows max_dom={max_dom}, experiment requests "
            f"{domain_count} domains"
        )
    target_vertical_levels = target["target_vertical_levels"]
    vertical_mismatch = {
        f"d{domain.grid_id:02d}": domain.run.nz
        for domain in exp.domains
        if domain.run.nz != target_vertical_levels
    }
    if vertical_mismatch:
        raise ValueError(
            "mapped target vertical levels differ from the experiment: "
            f"target={target_vertical_levels}, domains={vertical_mismatch}"
        )
    if target["require_lateral_boundaries"] is not True:
        raise ValueError(
            "mapped direct export requires a target contract with lateral "
            "boundaries"
        )
    if (isinstance(boundary_interval_seconds, bool)
            or not isinstance(boundary_interval_seconds, int)
            or boundary_interval_seconds <= 0):
        raise ValueError("mapped boundary interval must be a positive integer")
    declared_interval = target["boundary_interval_seconds"]
    if boundary_interval_seconds != declared_interval:
        raise ValueError(
            f"mapped boundary interval {boundary_interval_seconds} differs "
            f"from target contract {declared_interval}"
        )
    validate_boundary_timing(
        exp, boundary_interval_seconds,
        source=(
            "mapped hierarchy target"
            if hierarchy else "mapped target"))
    return {
        "status": "PASS",
        "domain_count": domain_count,
        "domain_ids": [domain.grid_id for domain in exp.domains],
        "domain_start_times": {
            f"d{domain.grid_id:02d}": (
                exp.domain_start_time(domain.grid_id).isoformat()
                if hasattr(exp, "domain_start_time")
                else (
                    exp.start_time
                    if getattr(domain, "start_time", None) is None
                    else domain.start_time
                ).isoformat())
            for domain in exp.domains
        },
        "mapping_max_dom": max_dom,
        "target_vertical_levels": target_vertical_levels,
        "require_lateral_boundaries": True,
        "boundary_interval_seconds": boundary_interval_seconds,
        "hierarchy": hierarchy,
    }


def prepare_mapped_wrf(
    *,
    composition: str | Path,
    mapping: str | Path,
    primary_files: Sequence[str | Path],
    supplement_files: Mapping[
        str, str | Path | Sequence[str | Path]
    ],
    provenance_files: Mapping[str, str | Path],
    input_manifest: str | Path,
    input_manifest_sha256: str,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
    wps_namelist: str | Path,
    geog_root: str | Path,
    experiment_config: str | Path,
    output_root: str | Path,
    preprocess_backend: str = "cuda",
    preprocess_workers: int | None = None,
    cpu_preprocess_bridge: str | Path | None = None,
    hierarchy_workers: int | None = None,
    _predecoded_bundle: MappedSourceBundle | None = None,
    _predecoded_seconds: float | None = None,
    _source_adapter: str = "rw-wps-mapped-composition-v2",
) -> dict[str, object]:
    """Build native WRF inputs from one complete mapped-source composition.

    All mixed-product semantics are resolved before target interpolation.
    The root owns the complete external-LBC sequence; mapped child inputs are
    initialized independently and finalized parent-before-child.  Products
    are published atomically only after source and run-control authorities are
    reverified.
    """

    started = time.perf_counter()
    composition = Path(composition).resolve()
    mapping = Path(mapping).resolve()
    primary = tuple(Path(path).resolve() for path in primary_files)
    supplements = {
        str(role): _path_inventory(path, f"supplement {role!r}")
        for role, path in supplement_files.items()
    }
    provenance = {
        str(role): Path(path).resolve() for role, path in provenance_files.items()
    }
    input_manifest = Path(input_manifest).resolve()
    mapping_snapshot = _snapshot_authority(mapping, retain_bytes=True)
    mapping_contract = load_mapping(
        mapping,
        _raw=_load_json_bytes(mapping_snapshot.data, "mapping", mapping),
    )
    decoders = _decoder_inventory(
        str(mapping_contract["format"]),
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
    )
    wps_namelist = Path(wps_namelist).resolve()
    geog_root = Path(geog_root).resolve()
    experiment_config = Path(experiment_config).resolve()
    output_root = Path(output_root).resolve()
    cpu_bridge = (
        None if cpu_preprocess_bridge is None
        else Path(cpu_preprocess_bridge).resolve()
    )
    for path in (
        composition, mapping, input_manifest, wps_namelist,
        experiment_config, *decoders.values(), *primary,
        *(path for paths in supplements.values() for path in paths),
        *provenance.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not geog_root.is_dir():
        raise NotADirectoryError(geog_root)
    if cpu_bridge is not None and not cpu_bridge.is_file():
        raise FileNotFoundError(cpu_bridge)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite mapped output {output_root}")
    run_control_before = {
        "wps_namelist": _file_receipt(wps_namelist),
        "experiment_config": _file_receipt(experiment_config),
    }
    if cpu_bridge is not None:
        run_control_before["cpu_preprocess_bridge"] = _file_receipt(cpu_bridge)

    exp = load_experiment(experiment_config)
    from gpuwm.experiment import (
        refuse_unrouted_perturbation, refuse_unrouted_spawn,
    )
    refuse_unrouted_perturbation(exp, "mapped-adapter prepared-cache")
    refuse_unrouted_spawn(exp, "mapped-adapter prepared-cache")
    from gpuwm.static.highres_production import refuse_inert_highres
    refuse_inert_highres(experiment_config, lane="mapped adapter")
    hierarchy = len(exp.domains) > 1
    target_contract = _validate_target_contract(
        mapping_contract,
        exp,
        mapping_contract["target"].get("boundary_interval_seconds"),
        hierarchy=hierarchy,
    )
    cfg = exp.root.run
    if hierarchy:
        grids = validate_native_lambert_contracts(
            exp, wps_namelist, source_name="mapped source",
        )
        grid = grids[0]
    else:
        grid = validate_native_lambert_contract(
            exp, wps_namelist, source_name="mapped source",
        )
        grids = (grid,)
    selection = GeogSelection.from_case_data(
        SimpleNamespace(wps_namelist=wps_namelist, geog_root=geog_root), 1,
    )

    decode_started = time.perf_counter()
    if _predecoded_bundle is None:
        if _predecoded_seconds is not None:
            raise ValueError(
                "predecoded timing was supplied without a predecoded bundle"
            )
        bundle = decode_composed_source(
            composition, mapping, primary, supplements, provenance,
            input_manifest=input_manifest,
            input_manifest_sha256=input_manifest_sha256,
            grib1_bridge=decoders.get("grib1_bridge"),
            grib2_inventory=decoders.get("grib2_inventory"),
            grib2_dump=decoders.get("grib2_dump"),
        )
        decode_seconds = time.perf_counter() - decode_started
    else:
        bundle = _predecoded_bundle
        if _predecoded_seconds is None or _predecoded_seconds < 0.0:
            raise ValueError(
                "predecoded bundle requires a nonnegative decode duration"
            )
        requested_terrain = tuple(
            path for paths in supplements.values() for path in paths
        )
        if (
            bundle.mapping_path != mapping
            or bundle.composition_path != composition
            or bundle.input_manifest_path != input_manifest
            or bundle.input_manifest_sha256
            != input_manifest_sha256.lower()
            or dict(bundle.decoder_paths) != decoders
            or bundle.terrain_data_paths != requested_terrain
            or bundle.terrain_provenance_path
            not in set(provenance.values())
        ):
            raise ValueError(
                "predecoded bundle authorities differ from the requested "
                "mapped preparation inputs"
            )
        decode_seconds = float(_predecoded_seconds)
    if bundle.mapping_sha256 != mapping_snapshot.sha256:
        raise ValueError(
            "mapped contract changed between target validation and decode"
        )
    _require_authority_snapshot(mapping_snapshot)
    if dict(bundle.decoder_paths) != decoders:
        raise ValueError("decoded decoder paths differ from requested decoders")
    composition_receipt = mapped_composition_receipt(bundle)
    snapshots = tuple(sorted(
        bundle.regular_snapshots(), key=lambda snapshot: snapshot.valid_time,
    ))
    times = tuple(snapshot.valid_time for snapshot in snapshots)
    if not times or times[0] != exp.start_time:
        raise ValueError(
            f"mapped forcing must begin at {exp.start_time}, got {times[:1]}"
        )
    forcing_offsets = tuple((value - times[0]).total_seconds() for value in times)
    if any(not value.is_integer() for value in forcing_offsets):
        raise ValueError("mapped forcing times must use whole-second offsets")
    forcing_seconds = tuple(int(value) for value in forcing_offsets)
    if any(value < 0 for value in forcing_seconds) \
            or forcing_seconds[-1] < exp.run_seconds:
        raise ValueError("mapped forcing does not cover the configured run")
    deltas = {
        later - earlier
        for earlier, later in zip(forcing_seconds, forcing_seconds[1:])
    }
    if len(deltas) != 1 or next(iter(deltas), 0) <= 0:
        raise ValueError("mapped forcing cadence must be positive and uniform")
    boundary_interval_seconds = deltas.pop()
    target_contract = _validate_target_contract(
        mapping_contract,
        exp,
        boundary_interval_seconds,
        hierarchy=hierarchy,
    )
    source_top_pressure_pa = max(
        float(np.min(snapshot.levels_hpa) * 100.0) for snapshot in snapshots
    )
    if hierarchy:
        grids = validate_native_lambert_contracts(
            exp,
            wps_namelist,
            source_name="mapped source",
            source_top_pressure_pa=source_top_pressure_pa,
        )
        grid = grids[0]
    else:
        validate_explicit_eta_grid(
            exp.vertical.eta_levels,
            nz=cfg.nz,
            p_top=exp.vertical.p_top,
            source_top_pressure_pa=source_top_pressure_pa,
            context="mapped direct adapter",
        )

    static_started = time.perf_counter()
    static = build_static(grid, geog_root, selection=selection)
    static_seconds = time.perf_counter() - static_started

    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_bridge,
    )
    preprocess_receipt = preprocess.receipt()
    initialize_started = time.perf_counter()
    # Only the first time's met/state survive this loop; every later time
    # contributes its perimeter frames and is released at once.  Holding
    # all of them is what made ingest peak above the forecast it prepares.
    initial_result = None
    initial_met = None
    forcing = StateBoundaryFrames(
        spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    coord = make_vertical_coord(
        cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
        eta_levels=exp.vertical.eta_levels,
    )
    mapfac_m, mapfac_u, mapfac_v = (
        grid.mapfac_m(), grid.mapfac_u(), grid.mapfac_v(),
    )
    coriolis_f, coriolis_e = grid.coriolis_m()
    rotation_sin, rotation_cos = grid.rotation_m()
    for index, source in enumerate(snapshots):
        # Metgrid classifies masked-field TARGET cells by the model
        # (geogrid) landmask; the mapped lane declares it like the ERA5
        # lanes so soil, skin, snow, and physics share one surface.
        met = interpolate_era5_to_lambert(
            source, grid, backend=preprocess,
            target_landmask=np.asarray(static["LANDMASK"]) >= 0.5)
        initialized = initialize_real(
            met, cfg, coord, static["HGT_M"],
            p_top=exp.vertical.p_top, sfcp_to_sfcp=True,
            preprocess_backend=preprocess,
            state_backend="preprocess",
        )
        initialized.state.set_map_coriolis(
            mapfac_m, mapfac_u, mapfac_v, coriolis_f, coriolis_e,
            sina=rotation_sin, cosa=rotation_cos,
        )
        forcing.add_state(initialized.state)
        if index == 0:
            initial_met = met
            initial_result = initialized
        else:
            del met, initialized
            release_backend_memory(preprocess)
    boundaries = forcing.build(times)
    attach_lateral_boundaries(initial_result.state, boundaries)
    # No lake skin override: the masked=both SKINTEMP chain with
    # static-landmask targets already yields water-source skin at lakes,
    # matching real.exe's no-TAVGSFC behavior.  The static landmask drives
    # WRF's process_soil_real land/water branches, and terrain plus the
    # composition's canonical source orography enable adjust_soil_temp_new's
    # elevation lapse on skin and soil temperature inputs.  The router
    # forwards this exact argument list to preprocess_noah_soil for
    # Noah-geometry schemes.
    soil = preprocess_land_surface_soil(
        initial_met.fields,
        sf_surface_physics=int(cfg.sf_surface_physics),
        soil_type=static["SCT_DOM"],
        deep_soil_temperature=static["TMN"],
        soil_layer_contract=bundle.soil_layer_contract,
        landmask=static["LANDMASK"],
        terrain=static["HGT_M"],
        source_orography=initial_met.fields["SOURCE_OROGRAPHY"],
    )
    initialize_seconds = time.perf_counter() - initialize_started

    # Keep atomic siblings short.  Repeating a user-supplied output name at
    # every nested transaction exceeded legacy Windows MAX_PATH before the
    # first hierarchy artifact could be written.
    staging = output_root.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
    if staging.exists():
        raise FileExistsError(f"mapped staging output already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        static_path = staging / "native-static.npz"
        geometry_path = staging / "geometry-receipt.json"
        prepared_path = staging / "prepared-cache"
        wrf_path = staging / "wrf-native-input"
        evidence_path = staging / "source-evidence"
        evidence_path.mkdir()
        for source, name, digest in (
            (mapping, "mapping.json", bundle.mapping_sha256),
            (
                composition,
                "composition.json",
                bundle.composition_sha256,
            ),
            (
                input_manifest,
                "input-manifest.json",
                bundle.input_manifest_sha256,
            ),
        ):
            _copy_bound_authority(source, evidence_path / name, digest)
        for role, source in sorted(provenance.items()):
            destination = evidence_path / _provenance_evidence_name(
                role, source.suffix,
            )
            if destination.exists():
                raise ValueError(
                    f"provenance roles collide after filename encoding: {role!r}"
                )
            if source != bundle.terrain_provenance_path:
                raise ValueError(
                    f"decoded provenance path differs for role {role!r}"
                )
            _copy_bound_authority(
                source,
                destination,
                bundle.terrain_provenance_sha256,
            )
        static_receipt = write_native_static_cache(
            static_path, native_static_export_fields(static, grid),
        )
        geometry_receipt = write_native_geometry_receipt(
            geometry_path, grid, cfg, static_path,
        )
        source_identity = {
            "adapter": _source_adapter,
            "mapping_sha256": bundle.mapping_sha256,
            "composition_sha256": bundle.composition_sha256,
            "input_manifest_sha256": bundle.input_manifest_sha256,
            "composition_receipt_sha256": composition_receipt[
                "receipt_content_sha256"
            ],
            "preprocessing": preprocess_receipt,
        }
        forcing_identity = (
            {"forcing_hours": tuple(
                value // 3600 for value in forcing_seconds)}
            if all(value % 3600 == 0 for value in forcing_seconds)
            else {"forcing_offsets_seconds": forcing_seconds})
        forcing_key, forcing_axis = next(iter(forcing_identity.items()))
        if hierarchy:
            selected_workers = hierarchy_workers
            if selected_workers is None:
                selected_workers = (
                    8 if preprocess_receipt["backend"] == "cpu" else 1
                )
            hierarchy_started = time.perf_counter()
            hierarchy_result = initialize_and_export_regular_source_hierarchy(
                exp=exp,
                grids=grids,
                snapshots=snapshots,
                **forcing_identity,
                wps_namelist=wps_namelist,
                geog_root=geog_root,
                source_name="RW-WPS-MAPPED",
                artifact_output=staging / "hierarchy-artifacts",
                wrf_output=wrf_path,
                root_initial_result=initial_result,
                root_met=initial_met,
                root_soil=soil,
                root_static_fields=static,
                root_boundaries=boundaries,
                bridge_manifest_sha256=bundle.input_manifest_sha256,
                source_manifest_sha256=bundle.input_manifest_sha256,
                namelist_sha256=_sha256(experiment_config),
                source_identity={
                    **source_identity,
                    "target_contract": target_contract,
                },
                source_inventory=tuple(snapshots[0].fields),
                workers=selected_workers,
                preprocess_backend=preprocess_receipt["backend"],
                cpu_bridge=cpu_bridge,
                soil_layer_contract=bundle.soil_layer_contract,
                root_metadata={
                    "composition_receipt_sha256": composition_receipt[
                        "receipt_content_sha256"
                    ],
                    "mapped_target_contract": target_contract,
                },
                input_provenance={
                    "mapping_sha256": bundle.mapping_sha256,
                    "composition_sha256": bundle.composition_sha256,
                    "input_manifest_sha256": bundle.input_manifest_sha256,
                    "decoder_sha256": dict(bundle.decoder_sha256),
                    "preprocessing": preprocess_receipt,
                    "mapped_target_contract": target_contract,
                },
                artifact_manifest_reference=(
                    "../hierarchy-artifacts/domain-artifacts.json"
                ),
            )
            hierarchy_seconds = time.perf_counter() - hierarchy_started
            run_control_after = {
                "wps_namelist": _file_receipt(wps_namelist),
                "experiment_config": _file_receipt(experiment_config),
            }
            if cpu_bridge is not None:
                run_control_after["cpu_preprocess_bridge"] = _file_receipt(
                    cpu_bridge
                )
            if run_control_after != run_control_before:
                raise ValueError(
                    "mapped run-control bytes changed during preparation"
                )
            proof = {
                "schema": HIERARCHY_PROOF_SCHEMA,
                "status": "READY_NOT_YET_STOCK_WRF_GATED",
                "domain_count": len(exp.domains),
                "forcing_times": [value.isoformat() for value in times],
                forcing_key: list(forcing_axis),
                "boundary_interval_seconds": boundary_interval_seconds,
                "target_contract": target_contract,
                "execution_inputs": {
                    "decoders": _bound_decoder_receipts(
                        bundle.decoder_paths,
                        bundle.decoder_sha256,
                    ),
                    **run_control_before,
                    "geog_root": str(geog_root),
                    "geog_source_binding": (
                        "per_domain_resolved_dataset_paths_plus_native_"
                        "static_output_sha256"
                    ),
                },
                "source_composition": composition_receipt,
                "preprocessing": preprocess_receipt,
                "hierarchy_workers": selected_workers,
                "root_static": static_receipt,
                "root_geometry": geometry_receipt,
                "static_catalog": dict(
                    hierarchy_result.static_catalog_receipt
                ),
                "source_coverage": dict(
                    hierarchy_result.source_coverage_receipt
                ),
                "artifact_receipt": dict(
                    hierarchy_result.hierarchy.artifacts.receipt
                ),
                "wrf_manifest": dict(
                    hierarchy_result.hierarchy.wrf_manifest
                ),
                "timing_seconds": {
                    "static_root": static_seconds,
                    "decode_and_compose": decode_seconds,
                    "initialize_all_root_times": initialize_seconds,
                    **dict(hierarchy_result.hierarchy.timings_seconds),
                    "hierarchy_call_wall": hierarchy_seconds,
                    "total": time.perf_counter() - started,
                },
            }
            proof["proof_content_sha256"] = hashlib.sha256(
                _canonical(proof).encode("utf-8")
            ).hexdigest()
            (staging / "proof.json").write_text(
                json.dumps(
                    proof, indent=2, sort_keys=True, allow_nan=False
                ) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, output_root)
            return proof
        identity = prepared_cache_identity(
            bridge_manifest_sha256=bundle.input_manifest_sha256,
            source_manifest_sha256=bundle.input_manifest_sha256,
            static_cache_sha256=static_receipt["sha256"],
            namelist_sha256=_sha256(experiment_config),
            domain_config=exp.root,
            **forcing_identity,
            source_identity=source_identity,
        )
        cache_started = time.perf_counter()
        cache_receipt = write_prepared_cache(
            prepared_path, identity=identity,
            initial_result=initial_result, met=initial_met,
            boundaries=boundaries, surface=canonical_noah_surface(soil),
            metadata={
                "source_adapter": "mapped",
                "initial_valid_time": times[0].isoformat(),
                "last_valid_time": times[-1].isoformat(),
                forcing_key: list(forcing_axis),
                "boundary_interval_seconds": boundary_interval_seconds,
                "composition_receipt_sha256": composition_receipt[
                    "receipt_content_sha256"
                ],
            },
        )
        cache_receipt = dict(cache_receipt)
        cache_receipt["path"] = "prepared-cache"
        cache_seconds = time.perf_counter() - cache_started
        export_started = time.perf_counter()
        export_receipt = export_prepared_wrf(
            prepared_path, static_path, geometry_path, wrf_path,
            valid_time=times[0],
            boundary_interval_seconds=boundary_interval_seconds,
        )
        export_seconds = time.perf_counter() - export_started
        run_control_after = {
            "wps_namelist": _file_receipt(wps_namelist),
            "experiment_config": _file_receipt(experiment_config),
        }
        if cpu_bridge is not None:
            run_control_after["cpu_preprocess_bridge"] = _file_receipt(cpu_bridge)
        if run_control_after != run_control_before:
            raise ValueError("mapped run-control bytes changed during preparation")
        proof = {
            "schema": PROOF_SCHEMA,
            "status": "READY_NOT_YET_STOCK_WRF_GATED",
            "forcing_times": [value.isoformat() for value in times],
            forcing_key: list(forcing_axis),
            "boundary_interval_seconds": boundary_interval_seconds,
            "execution_inputs": {
                "decoders": _bound_decoder_receipts(
                    bundle.decoder_paths,
                    bundle.decoder_sha256,
                ),
                **run_control_before,
                "geog_root": str(geog_root),
                "geog_source_binding": (
                    "resolved_dataset_paths_plus_native_static_output_sha256"
                ),
                "geog_resolution_tokens": list(selection.resolution_tokens),
                "geog_datasets": {
                    field: str(selection.path(field))
                    for field in (
                        "terrain", "landuse", "soil_top", "soil_bottom",
                        "greenfrac", "lai", "albedo", "snow_albedo",
                        "soil_temperature",
                    )
                },
            },
            "source_composition": composition_receipt,
            "preprocessing": preprocess.receipt(),
            "static": static_receipt,
            "geometry": geometry_receipt,
            "prepared_cache": cache_receipt,
            "export": export_receipt,
            "timing_seconds": {
                "static": static_seconds,
                "decode_and_compose": decode_seconds,
                "initialize_all_times": initialize_seconds,
                "write_prepared_cache": cache_seconds,
                "direct_wrf_export": export_seconds,
                "total": time.perf_counter() - started,
            },
        }
        proof["proof_content_sha256"] = hashlib.sha256(
            _canonical(proof).encode("utf-8")
        ).hexdigest()
        (staging / "proof.json").write_text(
            json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
        return proof
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


_ROLE_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def _role_bindings(
    values: Sequence[str], *, multiple: bool,
) -> dict[str, Path | tuple[Path, ...]]:
    """Parse repeatable ROLE=PATH bindings without silent replacement."""

    grouped: dict[str, list[Path]] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not _ROLE_PATTERN.fullmatch(role) or not raw_path:
            raise ValueError(
                "role bindings must use non-empty ROLE=PATH with a portable "
                "role name"
            )
        if not multiple and role in grouped:
            raise ValueError(f"duplicate binding for role {role!r}")
        grouped.setdefault(role, []).append(Path(raw_path))
    return {
        role: tuple(paths) if multiple else paths[0]
        for role, paths in grouped.items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rw-wps mapped",
        description=(
            "Prepare atomic stock-WRF initial and boundary files from a "
            "strict mapped GRIB/NetCDF composition without WPS or real.exe."
        ),
    )
    parser.add_argument(
        "--source-format", choices=("grib1", "grib2", "netcdf"),
        required=True,
    )
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument(
        "--input", dest="primary_files", action="append", type=Path,
        required=True,
    )
    parser.add_argument(
        "--supplement", action="append", default=[], metavar="ROLE=PATH",
        help="repeat a role for a deterministic multi-file supplement",
    )
    parser.add_argument(
        "--provenance", action="append", default=[], metavar="ROLE=PATH",
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--grib1-bridge", type=Path)
    parser.add_argument("--grib2-inventory", type=Path)
    parser.add_argument("--grib2-dump", type=Path)
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument("--geog-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="cuda",
    )
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    parser.add_argument("--hierarchy-workers", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    mapping = load_mapping(args.mapping)
    if mapping["format"] != args.source_format:
        parser.error(
            f"--source-format {args.source_format} differs from mapping "
            f"format {mapping['format']}"
        )
    decoder_values = {
        "grib1_bridge": args.grib1_bridge,
        "grib2_inventory": args.grib2_inventory,
        "grib2_dump": args.grib2_dump,
    }
    required_decoders = {
        "grib1": {"grib1_bridge"},
        "grib2": {"grib2_inventory", "grib2_dump"},
        "netcdf": set(),
    }[args.source_format]
    present_decoders = {
        name for name, value in decoder_values.items() if value is not None
    }
    if present_decoders != required_decoders:
        parser.error(
            f"{args.source_format} requires decoder flags "
            f"{sorted(required_decoders)}, got {sorted(present_decoders)}"
        )
    if args.preprocess_workers is not None and args.preprocess_workers <= 0:
        parser.error("--preprocess-workers must be positive")
    if args.hierarchy_workers is not None \
            and args.hierarchy_workers not in range(1, 33):
        parser.error("--hierarchy-workers must be between 1 and 32")
    if args.preprocess_backend != "cpu" \
            and args.cpu_preprocess_bridge is not None:
        parser.error(
            "--cpu-preprocess-bridge requires --preprocess-backend cpu"
        )
    try:
        supplements = _role_bindings(args.supplement, multiple=True)
        provenance = _role_bindings(args.provenance, multiple=False)
    except ValueError as error:
        parser.error(str(error))
    proof = prepare_mapped_wrf(
        composition=args.composition,
        mapping=args.mapping,
        primary_files=args.primary_files,
        supplement_files=supplements,
        provenance_files=provenance,
        input_manifest=args.input_manifest,
        input_manifest_sha256=args.input_manifest_sha256,
        grib1_bridge=args.grib1_bridge,
        grib2_inventory=args.grib2_inventory,
        grib2_dump=args.grib2_dump,
        wps_namelist=args.wps_namelist,
        geog_root=args.geog_root,
        experiment_config=args.experiment_config,
        output_root=args.output_root,
        preprocess_backend=args.preprocess_backend,
        preprocess_workers=args.preprocess_workers,
        cpu_preprocess_bridge=args.cpu_preprocess_bridge,
        hierarchy_workers=args.hierarchy_workers,
    )
    print(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "HIERARCHY_PROOF_SCHEMA",
    "PROOF_SCHEMA",
    "main",
    "prepare_mapped_wrf",
]


if __name__ == "__main__":
    raise SystemExit(main())
