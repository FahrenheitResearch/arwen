"""Controller-only execution and evaluation machinery for the real74 N5B gate.

This module is CPU-only until :func:`run_member` is called.  The runner uses
``real74_d02.execute_production_run`` unchanged.  A temporary wrapper around
the public ``gpuwm.core.model.build_experiment`` return value applies the
documented one-ULP change through ``ExperimentState.node(...).state.<field>``
after the complete device tree has been built and before the executor starts.

No configuration file is generated.  The SHRINK experiment is a dataclass
replacement of production d04 using the controller's N5B plan.

Evaluator choices which the gate record leaves open are intentionally
centralized in :func:`evaluator_manifest` and consumed by both same-geometry
pair evaluation and production/SHRINK evaluation:

* reflectivity is column-maximum REFL_10CM, thresholded at 40 dBZ; FSS uses a
  circular grid-centre neighbourhood of radius 5 km on the complete 400-square
  core (no second, implicit edge crop);
* a cold pool is a >=2 K T2 deficit from that frame's 10-km core-perimeter
  environmental median; its edge is the four-neighbour binary boundary;
* gust-front arrival is the first scheduled history time at which that same
  cold-pool edge reaches a core cell.  The MAE mask is the union of reached
  cells and both runs must have a finite arrival there;
* CI objects are eight-connected >=40-dBZ frame objects linked between
  adjacent frames when their masks come within 10 km.  They must span at
  least 20 minutes and reach 25 km2.  An A object is boundary-seeded only when
  its first frame is within 10 km of A's outer boundary; it is unmatched when
  no qualifying B object is within 10 km at a time within 10 minutes;
* resolved TKE is 0.5*(u'^2+v'^2+w'^2), after mass-point destaggering and
  removal of each time/level/mask component mean, averaged over the lowest
  three mass levels and the 20-km common-core perimeter fetch mask.  The
  reported predicate is mean(B)/mean(A).

The 15-minute cadence is inherited from production d04.  In particular, the
20-minute CI persistence rule requires at least three frames (30-minute
observed span); it is never rounded down to two frames.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gpuwm.case_data import PerDomainSourceOrography, SourceOrography
from gpuwm.static.lambert import grids_from_projection_config
from gpuwm.verify.cases import real74_chain
from gpuwm.verify.cases import real74_d02 as shared


METRIC = "N5B_d04_boundary_location_invariance"
RUN_SECONDS = 75 * 60
CADENCE_SECONDS = 15 * 60
CORE_SHAPE = (400, 400)

# Ratified F-gate values.  These are the closed-world N5B authority: neither
# evaluator manifests nor frozen files are allowed to redefine them.
F_GATE_D04_DX_M = 1000.0 / 3.0
F_GATE_REFLECTIVITY_CORE_DBZ = 40.0
F_GATE_FSS_RADIUS_KM = 5.0
F_GATE_FSS_MIN = 0.90
F_GATE_EDGE_DISTANCE_MAX_KM = 3.0
F_GATE_ARRIVAL_MAE_MAX_MIN = 5.0
F_GATE_UNMATCHED_CI_COUNT = 0
F_GATE_TKE_RATIO_MIN = 0.80
F_GATE_TKE_RATIO_MAX = 1.25
F_GATE_CI_BOUNDARY_KM = 10.0
F_GATE_CI_PERSISTENCE_MIN = 20.0
F_GATE_CI_MINIMUM_AREA_KM2 = 25.0
F_GATE_CI_MATCH_KM = 10.0
F_GATE_CI_MATCH_MINUTES = 10.0
F_GATE_SAME_GEOMETRY_MEMBERS = 3
F_GATE_ENVELOPE_PERCENTILE = 95.0

# Evaluator choices ratified with the F gate and therefore immutable too.
F_GATE_COLD_POOL_T2_DEFICIT_K = 2.0
F_GATE_COLD_POOL_ENVIRONMENT_RING_KM = 10.0
F_GATE_CI_TEMPORAL_LINK_KM = 10.0
F_GATE_TKE_LOWEST_MASS_LEVELS = 3
F_GATE_INFLOW_FETCH_PERIMETER_KM = 20.0

SHRINK_WINDOW = (51, 549)
SHRINK_STAGGERED_WINDOW = (51, 550)
F22_REFERENCE = (
    "F22: production d04 cells [51:549)^2; anchor shift 51 child cells; "
    "extent 498")
EVALUATOR_DEFINITION_SOURCE = (
    "gpuwm.verify.cases.real74_n5b module docstring; choices fill only "
    "definitions left open by the authoritative N5B gate record")


def ratified_discrepancy_limits() -> dict[str, float]:
    """Return discrepancy-form limits derived only from ratified F values."""
    return {
        "one_minus_refl_fss": 1.0 - F_GATE_FSS_MIN,
        "cold_pool_edge_km": F_GATE_EDGE_DISTANCE_MAX_KM,
        "gust_front_arrival_mae_min": F_GATE_ARRIVAL_MAE_MAX_MIN,
        "unmatched_boundary_ci_count": float(F_GATE_UNMATCHED_CI_COUNT),
        "tke_lower_excursion": 1.0 - F_GATE_TKE_RATIO_MIN,
        "tke_upper_excursion": F_GATE_TKE_RATIO_MAX - 1.0,
    }


def _ratified_evaluator_parameters() -> dict[str, float | int]:
    return {
        "dx_m": F_GATE_D04_DX_M,
        "reflectivity_event_dbz": F_GATE_REFLECTIVITY_CORE_DBZ,
        "fss_radius_km": F_GATE_FSS_RADIUS_KM,
        "cold_pool_t2_deficit_k": F_GATE_COLD_POOL_T2_DEFICIT_K,
        "cold_pool_environment_ring_km": F_GATE_COLD_POOL_ENVIRONMENT_RING_KM,
        "ci_boundary_km": F_GATE_CI_BOUNDARY_KM,
        "ci_persistence_min": F_GATE_CI_PERSISTENCE_MIN,
        "ci_minimum_area_km2": F_GATE_CI_MINIMUM_AREA_KM2,
        "ci_match_km": F_GATE_CI_MATCH_KM,
        "ci_match_minutes": F_GATE_CI_MATCH_MINUTES,
        "ci_temporal_link_km": F_GATE_CI_TEMPORAL_LINK_KM,
        "resolved_tke_lowest_mass_levels": F_GATE_TKE_LOWEST_MASS_LEVELS,
        "inflow_fetch_perimeter_km": F_GATE_INFLOW_FETCH_PERIMETER_KM,
    }


def _ratified_acceptance_values() -> dict[str, float | int]:
    return {
        "reflectivity_fss_min": F_GATE_FSS_MIN,
        "cold_pool_edge_distance_max_km": F_GATE_EDGE_DISTANCE_MAX_KM,
        "gust_front_arrival_mae_max_min": F_GATE_ARRIVAL_MAE_MAX_MIN,
        "unmatched_boundary_ci_count": F_GATE_UNMATCHED_CI_COUNT,
        "tke_ratio_min": F_GATE_TKE_RATIO_MIN,
        "tke_ratio_max": F_GATE_TKE_RATIO_MAX,
    }


@dataclass(frozen=True)
class PerturbationSpec:
    """One public device-state element to move toward positive infinity."""

    member_id: str
    domain_id: int
    field: str
    k: int
    j: int
    i: int


# These are safely beyond d01's five-cell specified/relaxation strip and are
# separated in all three axes. ``thp`` is DomainState's public prognostic
# theta-prime field; total theta is thb + thp.
DEFAULT_PERTURBATIONS = (
    PerturbationSpec("m01", 1, "thp", 10, 80, 100),
    PerturbationSpec("m02", 1, "thp", 20, 100, 125),
    PerturbationSpec("m03", 1, "thp", 30, 120, 150),
)
DEFAULT_PERTURBATION_BY_ID = {
    item.member_id: item for item in DEFAULT_PERTURBATIONS}

PERTURBATION_OPERATION = "np.nextafter(float32, +inf)"
PERTURBATION_APPLICATION_SURFACE = (
    "ExperimentState.node(domain_id).state.<field> after "
    "gpuwm.core.model.build_experiment and before execute_experiment")
PERTURBATION_FIELD_DEFINITION = (
    "prognostic theta-prime; total theta = thb + thp")


def _git_main_root() -> Path:
    """Return the main checkout root even when this module runs in a worktree."""
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=shared.REPOSITORY_ROOT, check=True, capture_output=True, text=True)
    return Path(result.stdout.strip()).resolve().parent


def default_plan_path() -> Path:
    return _git_main_root() / "out" / "rungs" / "N5B-plan.json"


def _load_plan(path: str | Path | None) -> dict[str, object]:
    plan_path = default_plan_path() if path is None else Path(path)
    with plan_path.open("r", encoding="utf-8") as stream:
        plan = json.load(stream)
    if plan.get("schema") != 1 or plan.get("metric") != METRIC:
        raise ValueError(f"{plan_path} is not the registered N5B plan")
    expected_geometry = shared.json_safe(asdict(real74_chain.n5b_geometry()))
    if shared.json_safe(plan.get("geometry")) != expected_geometry:
        raise ValueError("N5B plan geometry differs from the gate record")
    if plan.get("gate_convention") != real74_chain._gate("N5", METRIC).convention:
        raise ValueError("N5B plan convention differs from the gate record")
    return plan


def _unchanged_except(value, authority, allowed: set[str], label: str) -> None:
    for member in fields(authority):
        if member.name not in allowed and getattr(value, member.name) != getattr(
                authority, member.name):
            raise AssertionError(f"{label}.{member.name} did not inherit production")


def _d04_source_orography(case_data) -> SourceOrography:
    artifact = case_data.source_orography_for_domain(4)
    if artifact is None:
        raise ValueError(
            "N5B SHRINK needs an explicit per-d04 input artifact for the "
            "controller-generated crop")
    return artifact


def _orography_horizontal_shape(artifact: SourceOrography) -> tuple[int, int]:
    import netCDF4

    with netCDF4.Dataset(artifact.path) as dataset:
        if artifact.variable not in dataset.variables:
            raise ValueError(
                f"N5B d04 input lacks variable {artifact.variable!r}")
        shape = tuple(dataset.variables[artifact.variable].shape)
    if len(shape) < 2:
        raise ValueError("N5B d04 input orography is not a horizontal field")
    return shape[-2:]


def _netcdf_crop_index(dimensions: Sequence[str]) -> tuple[slice, ...]:
    windows = {
        "south_north": slice(*SHRINK_WINDOW),
        "west_east": slice(*SHRINK_WINDOW),
        "south_north_stag": slice(*SHRINK_STAGGERED_WINDOW),
        "west_east_stag": slice(*SHRINK_STAGGERED_WINDOW),
    }
    return tuple(windows.get(name, slice(None)) for name in dimensions)


def _cropped_corner_coordinates(source) -> tuple[np.ndarray, np.ndarray] | None:
    latitudes = []
    longitudes = []
    for suffix in ("M", "U", "V", "C"):
        lat_name, lon_name = f"XLAT_{suffix}", f"XLONG_{suffix}"
        if lat_name not in source.variables or lon_name not in source.variables:
            return None
        lat_var, lon_var = source.variables[lat_name], source.variables[lon_name]
        lat = np.asarray(lat_var[_netcdf_crop_index(lat_var.dimensions)])[0]
        lon = np.asarray(lon_var[_netcdf_crop_index(lon_var.dimensions)])[0]
        latitudes.extend((lat[0, 0], lat[-1, 0], lat[-1, -1], lat[0, -1]))
        longitudes.extend((lon[0, 0], lon[-1, 0], lon[-1, -1], lon[0, -1]))
    return (np.asarray(latitudes, dtype=np.float32),
            np.asarray(longitudes, dtype=np.float32))


def _crop_d04_netcdf(source_path: Path, output_path: Path) -> None:
    """Copy a d04 NetCDF while cropping every mass/staggered horizontal axis."""
    import netCDF4

    source_path, output_path = source_path.resolve(), output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"N5B SHRINK crop output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"N5B SHRINK crop temporary exists: {temporary}")
    try:
        with netCDF4.Dataset(source_path) as source:
            mass_dimensions = {
                "south_north": 600, "west_east": 600,
                "south_north_stag": 601, "west_east_stag": 601,
            }
            for name, expected in mass_dimensions.items():
                if name in source.dimensions and len(source.dimensions[name]) != expected:
                    raise ValueError(
                        f"N5B production d04 dimension {name} has "
                        f"{len(source.dimensions[name])}, expected {expected}")
            with netCDF4.Dataset(
                    temporary, "w", format=source.file_format) as target:
                for name, dimension in source.dimensions.items():
                    if name in {"south_north", "west_east"}:
                        size = SHRINK_WINDOW[1] - SHRINK_WINDOW[0]
                    elif name in {"south_north_stag", "west_east_stag"}:
                        size = (SHRINK_STAGGERED_WINDOW[1]
                                - SHRINK_STAGGERED_WINDOW[0])
                    else:
                        size = None if dimension.isunlimited() else len(dimension)
                    target.createDimension(name, size)
                target.setncatts({name: source.getncattr(name)
                                  for name in source.ncattrs()})
                geometry_attributes = {
                    "WEST-EAST_GRID_DIMENSION": 499,
                    "SOUTH-NORTH_GRID_DIMENSION": 499,
                    "WEST-EAST_PATCH_START_UNSTAG": 1,
                    "WEST-EAST_PATCH_END_UNSTAG": 498,
                    "WEST-EAST_PATCH_START_STAG": 1,
                    "WEST-EAST_PATCH_END_STAG": 499,
                    "SOUTH-NORTH_PATCH_START_UNSTAG": 1,
                    "SOUTH-NORTH_PATCH_END_UNSTAG": 498,
                    "SOUTH-NORTH_PATCH_START_STAG": 1,
                    "SOUTH-NORTH_PATCH_END_STAG": 499,
                    "i_parent_start": 168, "j_parent_start": 168,
                    "i_parent_end": 333, "j_parent_end": 333,
                }
                for name, value in geometry_attributes.items():
                    target.setncattr(name, np.int32(value))
                corners = _cropped_corner_coordinates(source)
                if corners is not None:
                    target.setncattr("corner_lats", corners[0])
                    target.setncattr("corner_lons", corners[1])
                for name, variable in source.variables.items():
                    attributes = {
                        attr: variable.getncattr(attr)
                        for attr in variable.ncattrs() if attr != "_FillValue"}
                    fill = (variable.getncattr("_FillValue")
                            if "_FillValue" in variable.ncattrs() else None)
                    created = target.createVariable(
                        name, variable.datatype, variable.dimensions,
                        fill_value=fill)
                    if attributes:
                        created.setncatts(attributes)
                    created[:] = variable[_netcdf_crop_index(variable.dimensions)]
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def generate_shrink_d04_inputs(
        output_dir: str | Path, *,
        config_path: str | Path = shared.PRODUCTION_CONFIG) -> Path:
    """Generate and provenance-bind the deterministic F22 d04 input crop."""
    production, case_data = shared.construct_rung_case(
        4, config_path=config_path, run_seconds=RUN_SECONDS,
        restart_interval_s=0)
    geometry = real74_chain.n5b_geometry()
    if (production.domain(4).run.ny, production.domain(4).run.nx) != (
            geometry.production_shape):
        raise ValueError("N5B crop source is not production d04 geometry")
    source = _d04_source_orography(case_data)
    if _orography_horizontal_shape(source) != geometry.production_shape:
        raise ValueError("N5B crop source is not a production-sized d04 input")
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "N5B-shrink-inputs.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"N5B SHRINK input manifest already exists: {manifest_path}")
    result_path = output_dir / f"{source.path.stem}.N5B-SHRINK{source.path.suffix}"
    _crop_d04_netcdf(source.path, result_path)
    manifest = {
        "schema": 1, "metric": METRIC, "f22_reference": F22_REFERENCE,
        "window": {
            "notation": "[51:549)^2", "start": [51, 51],
            "stop": [549, 549], "extent": [498, 498],
        },
        "source": {
            "path": str(source.path.resolve()),
            "sha256": shared.sha256_file(source.path),
            "variable": source.variable,
        },
        "result": {
            "relative_path": result_path.relative_to(output_dir).as_posix(),
            "sha256": shared.sha256_file(result_path),
            "variable": source.variable,
        },
        "dimension_windows": {
            "south_north": [51, 549], "west_east": [51, 549],
            "south_north_stag": [51, 550], "west_east_stag": [51, 550],
        },
    }
    shared.write_json(manifest_path, manifest)
    return manifest_path


def _bind_shrink_d04_inputs(case_data, manifest_path: str | Path):
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected_window = {
        "notation": "[51:549)^2", "start": [51, 51],
        "stop": [549, 549], "extent": [498, 498],
    }
    expected_dimension_windows = {
        "south_north": [51, 549], "west_east": [51, 549],
        "south_north_stag": [51, 550], "west_east_stag": [51, 550],
    }
    if (manifest.get("schema") != 1 or manifest.get("metric") != METRIC
            or manifest.get("f22_reference") != F22_REFERENCE
            or manifest.get("window") != expected_window
            or manifest.get("dimension_windows")
            != expected_dimension_windows):
        raise ValueError("N5B SHRINK input manifest is not the F22 crop")
    source = _d04_source_orography(case_data)
    source_record, result_record = manifest.get("source"), manifest.get("result")
    if not isinstance(source_record, dict) or not isinstance(result_record, dict):
        raise ValueError("N5B SHRINK input provenance is incomplete")
    if (source_record.get("path") != str(source.path.resolve())
            or source_record.get("sha256") != shared.sha256_file(source.path)
            or source_record.get("variable") != source.variable):
        raise ValueError("N5B SHRINK input source provenance mismatch")
    relative = result_record.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("N5B SHRINK result needs a relative path")
    result_path = (manifest_path.parent / relative).resolve()
    if (not result_path.is_relative_to(manifest_path.parent)
            or not result_path.is_file()
            or shared.sha256_file(result_path) != result_record.get("sha256")):
        raise ValueError("N5B SHRINK result artifact hash mismatch")
    cropped = SourceOrography(result_path, str(result_record.get("variable")))
    if (cropped.variable != source.variable
            or _orography_horizontal_shape(cropped) != (498, 498)):
        raise ValueError("N5B SHRINK d04 input is not cropped to 498x498")
    declaration = case_data.source_orography
    if not isinstance(declaration, PerDomainSourceOrography):
        raise ValueError("N5B SHRINK needs per-domain source-orography inputs")
    by_domain = tuple(
        (domain_id, cropped if domain_id == 4 else artifact)
        for domain_id, artifact in declaration.by_domain)
    return replace(
        case_data, source_orography=PerDomainSourceOrography(by_domain))


def construct_variant(
        variant: str = "production", *, config_path: str | Path = shared.PRODUCTION_CONFIG,
        plan_path: str | Path | None = None,
        shrink_input_manifest: str | Path | None = None):
    """Construct the 4-domain 4,500-s production or planned SHRINK experiment."""
    if variant not in {"production", "shrink"}:
        raise ValueError("N5B variant must be 'production' or 'shrink'")
    production, data = shared.construct_rung_case(
        4, config_path=config_path, run_seconds=RUN_SECONDS,
        restart_interval_s=0)
    geometry = real74_chain.n5b_geometry()
    prod_d04 = production.domain(4)
    if (prod_d04.run.ny, prod_d04.run.nx) != geometry.production_shape:
        raise AssertionError("production d04 geometry differs from N5B record")
    if variant == "production":
        if shrink_input_manifest is not None:
            raise ValueError("production N5B variant cannot use SHRINK inputs")
        return production, data, geometry

    production_input_shape = _orography_horizontal_shape(
        _d04_source_orography(data))
    if shrink_input_manifest is None:
        if production_input_shape == geometry.production_shape:
            raise ValueError(
                "N5B SHRINK d04 inputs are production-sized; run the "
                "generate-shrink-inputs crop utility first")
        raise ValueError(
            "N5B SHRINK requires the manifest from the generate-shrink-inputs "
            "crop utility")
    if production_input_shape != geometry.production_shape:
        raise ValueError(
            "N5B SHRINK provenance source is not the production-sized d04 "
            "input")
    data = _bind_shrink_d04_inputs(data, shrink_input_manifest)

    plan = _load_plan(plan_path)
    requested = plan.get("d04")
    if not isinstance(requested, dict):
        raise ValueError("N5B plan has no d04 replacement")
    exact = {"nx": 498, "ny": 498, "i_parent_start": 168,
             "j_parent_start": 168}
    if requested != exact:
        raise ValueError(f"N5B plan d04 must be exactly {exact}, got {requested}")
    shrink_d04 = replace(
        prod_d04,
        i_parent_start=int(requested["i_parent_start"]),
        j_parent_start=int(requested["j_parent_start"]),
        run=replace(prod_d04.run, nx=int(requested["nx"]),
                    ny=int(requested["ny"])))
    _unchanged_except(
        shrink_d04, prod_d04, {"i_parent_start", "j_parent_start", "run"},
        "d04")
    _unchanged_except(shrink_d04.run, prod_d04.run, {"nx", "ny"}, "d04.run")
    domains = tuple(
        shrink_d04 if item.grid_id == 4 else item for item in production.domains)
    shrink = replace(production, name=str(plan.get("variant")), domains=domains)
    actual = shrink.domain(4)
    actual_geometry = {
        "nx": actual.run.nx, "ny": actual.run.ny,
        "i_parent_start": actual.i_parent_start,
        "j_parent_start": actual.j_parent_start,
    }
    if actual_geometry != exact:
        raise AssertionError(
            f"constructed SHRINK d04 differs from plan: {actual_geometry}")
    if _orography_horizontal_shape(
            _d04_source_orography(data)) != geometry.shrink_shape:
        raise AssertionError("N5B SHRINK d04 input shape differs from geometry")
    return shrink, data, geometry


def core_coordinate_arrays(exp, variant: str) -> dict[str, np.ndarray]:
    """Return output-representation XLAT/XLONG for the registered d04 core."""
    geometry = real74_chain.n5b_geometry()
    core = (geometry.production_core if variant == "production"
            else geometry.shrink_core)
    grid = grids_from_projection_config(exp)[-1]
    lat, lon = grid.latlon_mass()
    js, is_ = (slice(*core[0]), slice(*core[1]))
    arrays = {
        # WrfoutWriter stores all numeric variables as f4.
        "XLAT": np.ascontiguousarray(lat[js, is_], dtype=np.float32),
        "XLONG": np.ascontiguousarray(lon[js, is_], dtype=np.float32),
    }
    if any(value.shape != CORE_SHAPE for value in arrays.values()):
        raise AssertionError("N5B common-core coordinate shape is not 400x400")
    return arrays


def array_inventory_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        descriptor = json.dumps(
            [name, str(array.dtype), list(array.shape)],
            separators=(",", ":"), ensure_ascii=True).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def core_coordinate_sha256(exp, variant: str) -> str:
    return array_inventory_sha256(core_coordinate_arrays(exp, variant))


def assert_common_core_coordinates(
        *, config_path: str | Path = shared.PRODUCTION_CONFIG,
        plan_path: str | Path | None = None) -> str:
    """Assert the plan's literal XLAT/XLONG core identity, returning its hash."""
    production, _, _ = construct_variant(
        "production", config_path=config_path, plan_path=plan_path)
    # This is a grid-only assertion: no model build or d04 input consumption.
    _load_plan(plan_path)
    shrink, _, _ = real74_chain.construct_n5b_shrink_case(config_path)
    left = core_coordinate_arrays(production, "production")
    right = core_coordinate_arrays(shrink, "shrink")
    unequal = [name for name in left
               if not np.array_equal(left[name], right[name])]
    if unequal:
        def max_abs(name):
            return float(np.max(np.abs(
                left[name].astype(np.float64) - right[name])))

        detail = ", ".join(
            f"{name} max_abs={max_abs(name):.9g}" for name in unequal)
        raise ValueError(
            f"N5B common-core XLAT/XLONG mismatch ({detail})")
    left_hash = array_inventory_sha256(left)
    right_hash = array_inventory_sha256(right)
    if left_hash != right_hash:
        raise AssertionError("identical N5B coordinates produced unequal hashes")
    return left_hash


def _float32_bits(value: np.float32) -> str:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32).item()
    return f"0x{bits:08x}"


def apply_one_ulp(model, spec: PerturbationSpec) -> dict[str, object]:
    """Apply and document exactly one ``np.nextafter`` to public device state."""
    node = model.node(spec.domain_id)
    if spec.field.startswith("_") or not hasattr(node.state, spec.field):
        raise ValueError(f"unknown public DomainState field {spec.field!r}")
    array = getattr(node.state, spec.field)
    index = (spec.k, spec.j, spec.i)
    if getattr(array, "ndim", None) != 3:
        raise ValueError(f"DomainState.{spec.field} is not a three-dimensional field")
    if any(position < 0 or position >= size
           for position, size in zip(index, array.shape, strict=True)):
        raise IndexError(f"{spec.field}{index} outside shape {tuple(array.shape)}")
    before = np.float32(array[index].item())
    after = np.float32(np.nextafter(before, np.float32(np.inf)))
    if not np.isfinite(before) or not np.isfinite(after) or before == after:
        raise ValueError("one-ULP perturbation needs a finite movable float32")
    array[index] = after
    observed = np.float32(array[index].item())
    if _float32_bits(observed) != _float32_bits(after):
        raise AssertionError("device-state perturbation did not persist exactly")
    return {
        "operation": PERTURBATION_OPERATION,
        "application_surface": PERTURBATION_APPLICATION_SURFACE,
        "domain": f"d{spec.domain_id:02d}", "field": spec.field,
        "field_definition": PERTURBATION_FIELD_DEFINITION,
        "index_order": "k,j,i", "k": spec.k, "j": spec.j, "i": spec.i,
        "before": float(before), "after": float(after),
        "before_hex_bits": _float32_bits(before),
        "after_hex_bits": _float32_bits(after),
    }


def _frame_inventory(paths: Sequence[Path], output_dir: Path,
                     start_time: datetime) -> list[dict[str, object]]:
    pattern = re.compile(
        r"wrfout_d(?P<domain>[0-9]{2})_(?P<time>[0-9]{4}-[0-9]{2}-[0-9]{2}_"
        r"[0-9]{2}_[0-9]{2}_[0-9]{2})$")
    inventory = []
    for path in sorted(Path(item) for item in paths):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected production frame name {path.name}")
        valid = datetime.strptime(match.group("time"), "%Y-%m-%d_%H_%M_%S")
        inventory.append({
            "domain": f"d{int(match.group('domain')):02d}",
            "valid_time": valid.isoformat(),
            "offset_seconds": int((valid - start_time).total_seconds()),
            "relative_path": path.resolve().relative_to(output_dir.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": shared.sha256_file(path),
        })
    return inventory


def _restored_input_sha256(case_data) -> str:
    from gpuwm.ingest.preflight import build_input_catalog
    return build_input_catalog(case_data).fingerprint


def run_member(
        member_id: str, output_dir: str | Path, record_path: str | Path, *,
        variant: str = "production",
        config_path: str | Path = shared.PRODUCTION_CONFIG,
        plan_path: str | Path | None = None,
        shrink_input_manifest: str | Path | None = None) -> dict[str, object]:
    """Execute one unperturbed or documented default one-ULP N5B member."""
    if member_id != "unperturbed" and member_id not in DEFAULT_PERTURBATION_BY_ID:
        raise ValueError(f"unknown N5B member id {member_id!r}")
    if variant == "shrink" and member_id != "unperturbed":
        raise ValueError("SHRINK is an A/B run, not a same-geometry ensemble member")
    exp, data, geometry = construct_variant(
        variant, config_path=config_path, plan_path=plan_path,
        shrink_input_manifest=shrink_input_manifest)
    perturbation = None
    calls = 0

    import gpuwm.core.model as production_model
    original_build = production_model.build_experiment

    def build_then_perturb(*args, **kwargs):
        nonlocal calls, perturbation
        model = original_build(*args, **kwargs)
        calls += 1
        if member_id != "unperturbed":
            perturbation = apply_one_ulp(
                model, DEFAULT_PERTURBATION_BY_ID[member_id])
        return model

    production_model.build_experiment = build_then_perturb
    try:
        run = shared.execute_production_run(exp, data, output_dir)
    finally:
        production_model.build_experiment = original_build
    if calls != 1:
        raise AssertionError(f"production builder was called {calls} times, expected once")
    if run.completed_seconds != RUN_SECONDS:
        raise ValueError(
            f"N5B run completed {run.completed_seconds}s, expected {RUN_SECONDS}s")
    output_dir = Path(output_dir)
    record = {
        "schema": 1, "metric": METRIC, "id": member_id,
        "variant": variant, "outdir": str(output_dir.resolve()),
        "duration_seconds": run.completed_seconds,
        "cadence_seconds": CADENCE_SECONDS,
        "restored_input_sha256": _restored_input_sha256(data),
        "core_coordinate_sha256": core_coordinate_sha256(exp, variant),
        "geometry": asdict(geometry),
        "one_ulp_perturbation": perturbation,
        "frame_inventory": _frame_inventory(
            run.wrfout_paths, output_dir, exp.start_time),
        "execution": dict(run.execution),
        "evaluator_commit": shared._git_commit(),
        "generated_utc": shared._utc_now(),
    }
    record["frame_inventory_sha256"] = shared.stable_hash(
        record["frame_inventory"])
    shared.write_json(record_path, record)
    return record


def run_default_members(
        output_root: str | Path, *, member_ids: Sequence[str] | None = None,
        variant: str = "production",
        config_path: str | Path = shared.PRODUCTION_CONFIG,
        plan_path: str | Path | None = None,
        shrink_input_manifest: str | Path | None = None) -> list[Path]:
    """Run selected members sequentially; omission means the registered three."""
    ids = tuple(member_ids or tuple(DEFAULT_PERTURBATION_BY_ID))
    root = Path(output_root)
    records = []
    for member_id in ids:
        record = root / f"{member_id}.json"
        run_member(
            member_id, root / member_id / "frames", record, variant=variant,
            config_path=config_path, plan_path=plan_path,
            shrink_input_manifest=shrink_input_manifest)
        records.append(record)
    return records


def _neighborhood_fraction(events: np.ndarray, radius_cells: float) -> np.ndarray:
    if events.ndim != 2:
        raise ValueError("FSS event field must be two-dimensional")
    limit = int(math.floor(radius_cells))
    offsets = tuple(
        (dj, di) for dj in range(-limit, limit + 1)
        for di in range(-limit, limit + 1)
        if dj * dj + di * di <= radius_cells * radius_cells)
    numerator = np.zeros(events.shape, dtype=np.float64)
    denominator = np.zeros(events.shape, dtype=np.float64)
    ny, nx = events.shape
    for dj, di in offsets:
        src_j = slice(max(0, -dj), min(ny, ny - dj))
        src_i = slice(max(0, -di), min(nx, nx - di))
        dst_j = slice(max(0, dj), min(ny, ny + dj))
        dst_i = slice(max(0, di), min(nx, nx + di))
        numerator[dst_j, dst_i] += events[src_j, src_i]
        denominator[dst_j, dst_i] += 1.0
    return numerator / denominator


def reflectivity_fss(
        left_dbz: np.ndarray, right_dbz: np.ndarray, *, dx_m: float,
        event_dbz: float = F_GATE_REFLECTIVITY_CORE_DBZ,
        radius_km: float = F_GATE_FSS_RADIUS_KM) -> float:
    """FSS for 2-D fields or denominator-weighted pooled space-time FSS."""
    left, right = np.asarray(left_dbz), np.asarray(right_dbz)
    if (left.shape != right.shape or left.ndim not in (2, 3)
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not math.isfinite(dx_m) or dx_m <= 0):
        return float("nan")
    if left.ndim == 2:
        left, right = left[None], right[None]
    numerator = denominator = 0.0
    for l_frame, r_frame in zip(left, right, strict=True):
        lf = _neighborhood_fraction(l_frame >= event_dbz, radius_km * 1000 / dx_m)
        rf = _neighborhood_fraction(r_frame >= event_dbz, radius_km * 1000 / dx_m)
        numerator += float(np.sum((lf - rf) ** 2, dtype=np.float64))
        denominator += float(np.sum(lf * lf + rf * rf, dtype=np.float64))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else float("nan")
    return 1.0 - numerator / denominator


def binary_edge(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("cold-pool mask must be two-dimensional")
    interior = np.zeros_like(mask)
    if min(mask.shape) >= 3:
        interior[1:-1, 1:-1] = (
            mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
            & mask[1:-1, :-2] & mask[1:-1, 2:])
    return mask & ~interior


def _point_distances(source: np.ndarray, target: np.ndarray, dx_m: float) -> np.ndarray:
    """Exact nearest-point distances, chunked to bound temporary memory."""
    if not len(source) or not len(target):
        return np.full(len(source), np.inf, dtype=np.float64)
    result = np.full(len(source), np.inf, dtype=np.float64)
    block = max(1, 2_000_000 // len(target))
    for start in range(0, len(source), block):
        delta = source[start:start + block, None, :] - target[None, :, :]
        result[start:start + block] = np.sqrt(
            np.min(np.sum(delta.astype(np.float64) ** 2, axis=2), axis=1)) * dx_m
    return result


def cold_pool_edge_distance_km(
        left_deficit_k: np.ndarray, right_deficit_k: np.ndarray, *,
        dx_m: float,
        threshold_k: float = F_GATE_COLD_POOL_T2_DEFICIT_K) -> float:
    """Median of both directed edge-point distance populations."""
    left, right = np.asarray(left_deficit_k), np.asarray(right_deficit_k)
    if left.shape != right.shape or left.ndim not in (2, 3):
        return float("nan")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan")
    if left.ndim == 2:
        left, right = left[None], right[None]
    distances = []
    for l_frame, r_frame in zip(left, right, strict=True):
        le = np.argwhere(binary_edge(l_frame >= threshold_k))
        re = np.argwhere(binary_edge(r_frame >= threshold_k))
        if not len(le) or not len(re):
            return float("nan")
        distances.extend((_point_distances(le, re, dx_m),
                          _point_distances(re, le, dx_m)))
    return float(np.median(np.concatenate(distances))) / 1000.0


def gust_front_arrival_mae_min(
        left_deficit_k: np.ndarray, right_deficit_k: np.ndarray,
        times_seconds: np.ndarray, *,
        threshold_k: float = F_GATE_COLD_POOL_T2_DEFICIT_K) -> float:
    """First cold-pool-edge arrival MAE over the union reached-cell mask."""
    left, right = np.asarray(left_deficit_k), np.asarray(right_deficit_k)
    times = np.asarray(times_seconds, dtype=np.float64)
    if (left.shape != right.shape or left.ndim != 3
            or times.shape != (left.shape[0],)
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        return float("nan")

    def arrivals(series):
        result = np.full(series.shape[1:], np.nan, dtype=np.float64)
        for time, frame in zip(times, series, strict=True):
            reached = binary_edge(frame >= threshold_k) & ~np.isfinite(result)
            result[reached] = time
        return result

    la, ra = arrivals(left), arrivals(right)
    union = np.isfinite(la) | np.isfinite(ra)
    if not union.any() or not (np.isfinite(la[union]).all()
                               and np.isfinite(ra[union]).all()):
        return float("nan")
    return float(np.mean(np.abs(la[union] - ra[union]), dtype=np.float64) / 60.0)


def _frame_components(mask: np.ndarray) -> list[np.ndarray]:
    """Eight-connected components as ``(j,i)`` integer point arrays."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask)
    components = []
    ny, nx = mask.shape
    for j, i in np.argwhere(mask):
        if seen[j, i]:
            continue
        seen[j, i] = True
        stack = [(int(j), int(i))]
        points = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for yy in range(max(0, y - 1), min(ny, y + 2)):
                for xx in range(max(0, x - 1), min(nx, x + 2)):
                    if mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        components.append(np.asarray(points, dtype=np.int32))
    return components


def _within_km(left: np.ndarray, right: np.ndarray, dx_m: float,
               radius_km: float, left_origin=(0, 0), right_origin=(0, 0)) -> bool:
    if not len(left) or not len(right):
        return False
    lp = left.astype(np.float64) + np.asarray(left_origin, dtype=np.float64)
    rp = right.astype(np.float64) + np.asarray(right_origin, dtype=np.float64)
    return bool(np.min(_point_distances(lp, rp, dx_m)) <= radius_km * 1000.0)


def _storm_objects(refl: np.ndarray, times: np.ndarray, *, dx_m: float,
                   event_dbz: float, link_km: float):
    frame_nodes = []
    for frame_index, frame in enumerate(refl):
        for points in _frame_components(frame >= event_dbz):
            frame_nodes.append({"frame": frame_index, "points": points})
    parent = list(range(len(frame_nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    by_frame: dict[int, list[int]] = {}
    for index, node in enumerate(frame_nodes):
        by_frame.setdefault(node["frame"], []).append(index)
    for frame in range(len(refl) - 1):
        for left in by_frame.get(frame, []):
            for right in by_frame.get(frame + 1, []):
                if _within_km(frame_nodes[left]["points"],
                              frame_nodes[right]["points"], dx_m, link_km):
                    union(left, right)
    groups: dict[int, list[dict[str, object]]] = {}
    for index, node in enumerate(frame_nodes):
        groups.setdefault(find(index), []).append(node)
    objects = []
    for nodes in groups.values():
        first = min(item["frame"] for item in nodes)
        last = max(item["frame"] for item in nodes)
        objects.append({
            "nodes": nodes, "first_frame": first, "last_frame": last,
            "duration_seconds": float(times[last] - times[first]),
            "max_area_km2": max(
                len(item["points"]) * dx_m * dx_m / 1e6 for item in nodes),
        })
    return objects


def unmatched_boundary_ci_count(
        left_refl_dbz: np.ndarray, right_refl_dbz: np.ndarray,
        times_seconds: np.ndarray, *, dx_m: float,
        left_origin_cells=(0, 0), right_origin_cells=(0, 0),
        event_dbz: float = F_GATE_REFLECTIVITY_CORE_DBZ,
        boundary_km: float = F_GATE_CI_BOUNDARY_KM,
        persistence_min: float = F_GATE_CI_PERSISTENCE_MIN,
        minimum_area_km2: float = F_GATE_CI_MINIMUM_AREA_KM2,
        match_km: float = F_GATE_CI_MATCH_KM,
        match_minutes: float = F_GATE_CI_MATCH_MINUTES,
        temporal_link_km: float = F_GATE_CI_TEMPORAL_LINK_KM) -> int:
    """Count qualifying A-boundary objects without a space/time B match."""
    left, right = np.asarray(left_refl_dbz), np.asarray(right_refl_dbz)
    times = np.asarray(times_seconds, dtype=np.float64)
    if (left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0]
            or times.shape != (left.shape[0],)
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("CI object inputs are missing, non-finite, or misaligned")
    a_objects = _storm_objects(
        left, times, dx_m=dx_m, event_dbz=event_dbz,
        link_km=temporal_link_km)
    b_objects = _storm_objects(
        right, times, dx_m=dx_m, event_dbz=event_dbz,
        link_km=temporal_link_km)

    def qualifies(item):
        return (item["duration_seconds"] >= persistence_min * 60.0
                and item["max_area_km2"] >= minimum_area_km2)

    b_objects = [item for item in b_objects if qualifies(item)]
    ny, nx = left.shape[1:]
    boundary_cells = boundary_km * 1000.0 / dx_m
    unmatched = 0
    for a_object in a_objects:
        if not qualifies(a_object):
            continue
        first_nodes = [item for item in a_object["nodes"]
                       if item["frame"] == a_object["first_frame"]]
        seeded = any(np.any(
            np.minimum.reduce((
                item["points"][:, 0], ny - 1 - item["points"][:, 0],
                item["points"][:, 1], nx - 1 - item["points"][:, 1]))
            <= boundary_cells) for item in first_nodes)
        if not seeded:
            continue
        matched = False
        for b_object in b_objects:
            for an in a_object["nodes"]:
                for bn in b_object["nodes"]:
                    if abs(times[an["frame"]] - times[bn["frame"]]) > match_minutes * 60:
                        continue
                    if _within_km(
                            an["points"], bn["points"], dx_m, match_km,
                            left_origin_cells, right_origin_cells):
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
        unmatched += int(not matched)
    return unmatched


def resolved_tke_ratio(
        left_u: np.ndarray, left_v: np.ndarray, left_w: np.ndarray,
        right_u: np.ndarray, right_v: np.ndarray, right_w: np.ndarray, *,
        mask: np.ndarray) -> float:
    """Return fetch-mean resolved TKE(right)/resolved TKE(left)."""
    arrays = tuple(np.asarray(item, dtype=np.float64) for item in (
        left_u, left_v, left_w, right_u, right_v, right_w))
    if len({item.shape for item in arrays}) != 1 or arrays[0].ndim != 4:
        return float("nan")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != arrays[0].shape[-2:] or not mask.any():
        return float("nan")
    if any(not np.isfinite(item).all() for item in arrays):
        return float("nan")

    def mean_tke(u, v, w):
        total = samples = 0
        for component in (u, v, w):
            selected = component[..., mask]
            anomaly = selected - np.mean(selected, axis=-1, keepdims=True)
            total += float(np.sum(anomaly * anomaly, dtype=np.float64))
            samples += anomaly.size
        return 0.5 * total / (samples / 3)

    left = mean_tke(*arrays[:3])
    right = mean_tke(*arrays[3:])
    if not math.isfinite(left) or not math.isfinite(right) or left <= 0:
        return float("nan")
    return right / left


def metrics_to_discrepancies(metrics: Mapping[str, float]) -> dict[str, float]:
    ratio = float(metrics["tke_ratio"])
    return {
        "one_minus_refl_fss": 1.0 - float(metrics["refl_fss"]),
        "cold_pool_edge_km": float(metrics["cold_pool_edge_km"]),
        "gust_front_arrival_mae_min": float(metrics["gust_front_arrival_mae_min"]),
        "unmatched_boundary_ci_count": float(metrics["unmatched_boundary_ci_count"]),
        "tke_lower_excursion": max(0.0, 1.0 - ratio),
        "tke_upper_excursion": max(0.0, ratio - 1.0),
    }


def _perimeter_mask(shape: tuple[int, int], width_cells: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    width = min(width_cells, min(shape) // 2)
    mask[:width] = mask[-width:] = True
    mask[:, :width] = mask[:, -width:] = True
    return mask


def _mask_hash(mask: np.ndarray) -> str:
    return array_inventory_sha256({"mask": np.ascontiguousarray(mask, dtype=np.uint8)})


def evaluator_manifest(exp=None, *, variant: str = "production") -> dict[str, object]:
    """Create the single parameter/mask manifest used by all N5B pair scores."""
    if exp is None:
        exp, _, geometry = construct_variant(variant)
    else:
        geometry = real74_chain.n5b_geometry()
    dx_m = float(exp.domain(4).run.dx)
    if dx_m != F_GATE_D04_DX_M:
        raise ValueError("N5B evaluator d04 spacing is not the ratified F value")
    parameters = _ratified_evaluator_parameters()
    environment = _perimeter_mask(
        CORE_SHAPE, math.ceil(F_GATE_COLD_POOL_ENVIRONMENT_RING_KM
                              * 1000 / dx_m))
    inflow = _perimeter_mask(
        CORE_SHAPE, math.ceil(F_GATE_INFLOW_FETCH_PERIMETER_KM
                              * 1000 / dx_m))
    core = np.ones(CORE_SHAPE, dtype=bool)
    return {
        "schema": 1, "metric": METRIC,
        "evaluator_commit": shared._git_commit(),
        "geometry": asdict(geometry),
        "core_coordinate_sha256": core_coordinate_sha256(exp, variant),
        "cadence": {
            "seconds": CADENCE_SECONDS,
            "offset_seconds": list(range(0, RUN_SECONDS + 1, CADENCE_SECONDS)),
        },
        "parameters": parameters,
        "acceptance": _ratified_acceptance_values(),
        "masks": {
            "verification_core": {"shape": list(CORE_SHAPE), "sha256": _mask_hash(core)},
            "cold_pool_environment": {"shape": list(CORE_SHAPE),
                                      "sha256": _mask_hash(environment)},
            "gust_arrival_union": {"shape": list(CORE_SHAPE),
                                   "sha256": _mask_hash(core)},
            "ci_object_overlap": {"shape": list(CORE_SHAPE),
                                  "sha256": _mask_hash(core)},
            "inflow_fetch": {"shape": list(CORE_SHAPE),
                             "sha256": _mask_hash(inflow)},
        },
        "definition_source": EVALUATOR_DEFINITION_SOURCE,
    }


def write_evaluator_manifest(output: str | Path, exp=None, *,
                             variant: str = "production") -> dict[str, object]:
    manifest = evaluator_manifest(exp, variant=variant)
    shared.write_json(output, manifest)
    return manifest


def _validate_perturbation(document: object, *, member_id: object) -> None:
    """Require the exact registered single-nextafter document for a member."""
    if not isinstance(document, dict):
        raise ValueError("N5B member is missing its perturbation document")
    expected_keys = {
        "operation", "application_surface", "domain", "field",
        "field_definition", "index_order", "k", "j", "i", "before",
        "after", "before_hex_bits", "after_hex_bits",
    }
    if set(document) != expected_keys:
        raise ValueError("N5B perturbation document inventory is not exact")
    spec = DEFAULT_PERTURBATION_BY_ID.get(member_id)
    if spec is None:
        raise ValueError("N5B perturbation member is not registered")
    expected_identity = {
        "operation": PERTURBATION_OPERATION,
        "application_surface": PERTURBATION_APPLICATION_SURFACE,
        "domain": f"d{spec.domain_id:02d}",
        "field": spec.field,
        "field_definition": PERTURBATION_FIELD_DEFINITION,
        "index_order": "k,j,i",
        "k": spec.k, "j": spec.j, "i": spec.i,
    }
    if any(document.get(name) != value
           or (isinstance(document.get(name), bool)
               and isinstance(value, int))
           for name, value in expected_identity.items()):
        raise ValueError("N5B perturbation document is not the registered member")
    raw_before, raw_after = document["before"], document["after"]
    if (isinstance(raw_before, bool) or isinstance(raw_after, bool)
            or not isinstance(raw_before, (int, float))
            or not isinstance(raw_after, (int, float))):
        raise ValueError("N5B perturbation endpoints are not numeric")
    before, after = np.float32(raw_before), np.float32(raw_after)
    if (not np.isfinite(before) or not np.isfinite(after) or before == after
            or float(before) != float(raw_before)
            or float(after) != float(raw_after)
            or np.float32(np.nextafter(before, np.float32(np.inf))) != after):
        raise ValueError("N5B perturbation document is not one float32 ULP")
    if (_float32_bits(before) != document["before_hex_bits"]
            or _float32_bits(after) != document["after_hex_bits"]):
        raise ValueError("N5B perturbation bit documentation is inconsistent")


@dataclass(frozen=True)
class _MemberRecordSnapshot:
    path: Path
    sha256: str
    document: Mapping[str, object]
    frame_paths: Mapping[str, Path]
    dependency_sha256: Mapping[str, str]


def _snapshot_member_record(
        path: str | Path, *, perturbed: bool | None,
        expected_sha256: str | None = None) -> _MemberRecordSnapshot:
    """Hash and parse a member record from one byte snapshot."""
    path = Path(path).resolve()
    encoded = path.read_bytes()
    actual_sha256 = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"N5B member artifact hash mismatch: {path.name}")
    try:
        record = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not a valid N5B member record") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{path} is not an N5B member record")
    if perturbed is None:
        perturbed = record.get("id") != "unperturbed"
    if record.get("schema") != 1 or record.get("metric") != METRIC:
        raise ValueError(f"{path} is not an N5B member record")
    if float(record.get("duration_seconds", float("nan"))) != RUN_SECONDS:
        raise ValueError("N5B member duration is not the registered 4,500 seconds")
    if int(record.get("cadence_seconds", -1)) != CADENCE_SECONDS:
        raise ValueError("N5B member cadence is not the registered 900 seconds")
    if perturbed:
        _validate_perturbation(
            record.get("one_ulp_perturbation"), member_id=record.get("id"))
    elif record.get("one_ulp_perturbation") is not None:
        raise ValueError("N5B unperturbed record unexpectedly documents a perturbation")
    inventory = record.get("frame_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("N5B member has no frame inventory")
    outdir_value = record.get("outdir")
    if not isinstance(outdir_value, str) or not outdir_value:
        raise ValueError("N5B member has no output directory")
    outdir = Path(outdir_value).resolve()
    frame_paths = {}
    dependencies = {str(path): actual_sha256}
    for frame in inventory:
        if not isinstance(frame, dict):
            raise ValueError("N5B frame inventory is malformed")
        relative = frame.get("relative_path")
        expected = frame.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("N5B frame inventory is malformed")
        target = (outdir / relative).resolve()
        if not target.is_relative_to(outdir) or not target.is_file():
            raise ValueError("N5B frame inventory escapes or is missing")
        actual = shared.sha256_file(target)
        if actual != expected:
            raise ValueError(f"N5B frame hash mismatch: {target.name}")
        if (isinstance(frame.get("bytes"), bool)
                or not isinstance(frame.get("bytes"), int)
                or frame["bytes"] != target.stat().st_size):
            raise ValueError(f"N5B frame byte count mismatch: {target.name}")
        if relative in frame_paths:
            raise ValueError("N5B frame inventory has duplicate paths")
        frame_paths[relative] = target
        dependencies[str(target)] = actual
    if record.get("frame_inventory_sha256") != shared.stable_hash(inventory):
        raise ValueError("N5B frame inventory hash mismatch")
    return _MemberRecordSnapshot(
        path=path, sha256=actual_sha256, document=record,
        frame_paths=frame_paths, dependency_sha256=dependencies)


def validate_member_record(path: str | Path, *, perturbed: bool) -> dict[str, object]:
    return dict(_snapshot_member_record(path, perturbed=perturbed).document)


def _artifact_reference(path: Path, root: Path, record: Mapping[str, object], *,
                        perturbation: bool) -> dict[str, object]:
    path, root = path.resolve(), root.resolve()
    if not path.is_relative_to(root):
        raise ValueError("N5B manifest artifacts must be beneath its output directory")
    result = {
        "id": record["id"], "relative_path": path.relative_to(root).as_posix(),
        "sha256": shared.sha256_file(path),
        "duration_seconds": float(record["duration_seconds"]),
        "restored_input_sha256": record["restored_input_sha256"],
        "core_coordinate_sha256": record["core_coordinate_sha256"],
    }
    if perturbation:
        result["one_ulp_perturbation"] = record["one_ulp_perturbation"]
    return result


def emit_ensemble_manifest(
        member_records: Sequence[str | Path], unperturbed_record: str | Path,
        output: str | Path, *,
        evaluator_manifest_path: str | Path) -> dict[str, object]:
    """Validate member records, evaluate every pair, and emit freeze input."""
    if len(member_records) < 3:
        raise ValueError("N5B needs at least 3 same-geometry members")
    output = Path(output).resolve()
    root = output.parent
    evaluator_path = Path(evaluator_manifest_path).resolve()
    if not evaluator_path.is_relative_to(root):
        raise ValueError("N5B evaluator manifest must be beneath ensemble root")
    evaluator = load_evaluator_manifest(evaluator_path)
    member_paths = [Path(item).resolve() for item in member_records]
    member_snapshots = [
        _snapshot_member_record(item, perturbed=True) for item in member_paths]
    members = [item.document for item in member_snapshots]
    baseline_path = Path(unperturbed_record).resolve()
    baseline = _snapshot_member_record(
        baseline_path, perturbed=False).document
    if len({item["id"] for item in members}) != len(members):
        raise ValueError("N5B member ids must be unique")
    if len({shared.sha256_file(path) for path in member_paths}) != len(member_paths):
        raise ValueError("N5B member artifacts must be unique")
    all_records = [baseline, *members]
    if len({item["restored_input_sha256"] for item in all_records}) != 1:
        raise ValueError("N5B same-geometry restored-input hashes differ")
    if len({item["core_coordinate_sha256"] for item in all_records}) != 1:
        raise ValueError("N5B same-geometry core coordinates differ")
    if any(item.get("variant") != "production" for item in all_records):
        raise ValueError("N5B same-geometry ensemble must use production geometry")
    if evaluator["core_coordinate_sha256"] != baseline["core_coordinate_sha256"]:
        raise ValueError("N5B ensemble evaluator coordinates differ from members")
    discrepancies = {name: []
                     for name in real74_chain._n5b_stated_discrepancies()}
    for left, right in itertools.combinations(member_snapshots, 2):
        values = dict(_evaluate_same_geometry_snapshot_pair(
            left, right, evaluator=evaluator))
        if set(values) == {
                "refl_fss", "cold_pool_edge_km", "gust_front_arrival_mae_min",
                "unmatched_boundary_ci_count", "tke_ratio"}:
            values = metrics_to_discrepancies(values)
        if set(values) != set(discrepancies):
            raise ValueError("N5B pair evaluator returned an incomplete inventory")
        for name, value in values.items():
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"N5B pair {name} is negative or non-finite")
            discrepancies[name].append(number)
    artifact_members = [
        _artifact_reference(path, root, record, perturbation=True)
        for path, record in zip(member_paths, members, strict=True)]
    manifest = {
        "schema": 1, "metric": METRIC,
        "geometry": asdict(real74_chain.n5b_geometry()),
        "unperturbed": _artifact_reference(
            baseline_path, root, baseline, perturbation=False),
        "members": artifact_members,
        "same_geometry_discrepancies": discrepancies,
        "restored_input_sha256": baseline["restored_input_sha256"],
        "core_coordinate_sha256": baseline["core_coordinate_sha256"],
        "cadence": f"{CADENCE_SECONDS}-second d04 history frames",
        "evaluator_commit": shared._git_commit(),
        "generated_utc": shared._utc_now(),
        "evaluator_manifest": {
            "relative_path": evaluator_path.relative_to(root).as_posix(),
            "sha256": shared.sha256_file(evaluator_path),
        },
    }
    shared.write_json(output, manifest)
    return manifest


@dataclass(frozen=True)
class EvaluationSeries:
    times_seconds: np.ndarray
    core_refl_dbz: np.ndarray
    core_t2_k: np.ndarray
    core_u: np.ndarray
    core_v: np.ndarray
    core_w: np.ndarray
    full_refl_dbz: np.ndarray
    origin_cells: tuple[int, int]


class _SameGeometryEvaluationAuthority:
    pass


_SAME_GEOMETRY_AUTHORITY = _SameGeometryEvaluationAuthority()


def _nc_array(variable, key) -> np.ndarray:
    return np.asarray(np.ma.filled(variable[key], np.nan), dtype=np.float64)


def _load_evaluation_series_snapshot(
        snapshot: _MemberRecordSnapshot) -> EvaluationSeries:
    """Load required d04 arrays while each hash-verified frame remains open."""
    import netCDF4

    record = snapshot.document
    variant = record["variant"]
    geometry = real74_chain.n5b_geometry()
    core = geometry.production_core if variant == "production" else geometry.shrink_core
    js, is_ = slice(*core[0]), slice(*core[1])
    d04 = sorted(
        (item for item in record["frame_inventory"] if item["domain"] == "d04"),
        key=lambda item: item["offset_seconds"])
    expected = list(range(0, RUN_SECONDS + 1, CADENCE_SECONDS))
    if [item["offset_seconds"] for item in d04] != expected:
        raise ValueError("N5B d04 frame cadence/inventory is incomplete")
    core_refl, core_t2, core_u, core_v, core_w, full_refl = ([] for _ in range(6))
    levels = F_GATE_TKE_LOWEST_MASS_LEVELS
    for item in d04:
        frame_path = snapshot.frame_paths[item["relative_path"]]
        with netCDF4.Dataset(frame_path) as dataset:
            # The dataset handle keeps the opened file identity stable.  Hash
            # both before and after array extraction so path replacement or
            # in-place mutation cannot feed unverified bytes to the evaluator.
            if shared.sha256_file(frame_path) != item["sha256"]:
                raise ValueError(f"N5B frame hash mismatch: {frame_path.name}")
            refl = _nc_array(dataset.variables["REFL_10CM"], 0)
            if refl.ndim == 3:
                refl = np.max(refl, axis=0)
            full_refl.append(refl)
            core_refl.append(refl[js, is_])
            core_t2.append(_nc_array(dataset.variables["T2"], 0)[js, is_])
            u = _nc_array(dataset.variables["U"], (0, slice(0, levels), js, slice(None)))
            v = _nc_array(dataset.variables["V"], (0, slice(0, levels), slice(None), is_))
            w = _nc_array(dataset.variables["W"], (0, slice(0, levels + 1), js, is_))
            u = 0.5 * (u[..., :-1] + u[..., 1:])
            v = 0.5 * (v[..., :-1, :] + v[..., 1:, :])
            w = 0.5 * (w[:-1] + w[1:])
            core_u.append(u[..., is_])
            core_v.append(v[..., js, :])
            core_w.append(w)
            if shared.sha256_file(frame_path) != item["sha256"]:
                raise ValueError(
                    f"N5B frame changed during evaluation: {frame_path.name}")
    # Anchor 168 versus 151 is 17 parent cells = 51 d04 cells.
    origin = (0, 0) if variant == "production" else (51, 51)
    return EvaluationSeries(
        times_seconds=np.asarray(expected, dtype=np.float64),
        core_refl_dbz=np.asarray(core_refl), core_t2_k=np.asarray(core_t2),
        core_u=np.asarray(core_u), core_v=np.asarray(core_v), core_w=np.asarray(core_w),
        full_refl_dbz=np.asarray(full_refl), origin_cells=origin)


def load_evaluation_series(record_path: str | Path) -> EvaluationSeries:
    """Load only the d04 fields/levels required by the pinned evaluator."""
    path = Path(record_path)
    snapshot = _snapshot_member_record(path, perturbed=None)
    return _load_evaluation_series_snapshot(snapshot)


def _validated_evaluator(
        manifest: Mapping[str, object], *, dx_m: float) -> dict[str, object]:
    """Validate the externally pinned parameters/masks used by an evaluation."""
    if not isinstance(manifest, Mapping):
        raise ValueError("N5B evaluation requires its ratified manifest")
    expected_manifest_keys = {
        "schema", "metric", "evaluator_commit", "geometry",
        "core_coordinate_sha256", "cadence", "parameters", "acceptance",
        "masks", "definition_source",
    }
    if (set(manifest) != expected_manifest_keys
            or manifest.get("definition_source") != EVALUATOR_DEFINITION_SOURCE):
        raise ValueError("N5B evaluator manifest inventory is not closed-world")
    if (not math.isfinite(dx_m) or dx_m <= 0.0
            or dx_m != F_GATE_D04_DX_M):
        raise ValueError("N5B evaluator grid spacing is not the ratified F value")
    if (manifest.get("schema") != 1 or manifest.get("metric") != METRIC
            or manifest.get("evaluator_commit") != shared._git_commit()):
        raise ValueError("N5B evaluator manifest identity/commit is stale")
    if shared.json_safe(manifest.get("geometry")) != shared.json_safe(
            asdict(real74_chain.n5b_geometry())):
        raise ValueError("N5B evaluator geometry is not registered")
    coordinate_hash = manifest.get("core_coordinate_sha256")
    if (not isinstance(coordinate_hash, str) or len(coordinate_hash) != 64
            or any(character not in "0123456789abcdef"
                   for character in coordinate_hash)):
        raise ValueError("N5B evaluator core-coordinate hash is invalid")
    cadence = manifest.get("cadence")
    if (not isinstance(cadence, dict)
            or set(cadence) != {"seconds", "offset_seconds"}
            or cadence.get("seconds") != CADENCE_SECONDS
            or cadence.get("offset_seconds")
            != list(range(0, RUN_SECONDS + 1, CADENCE_SECONDS))):
        raise ValueError("N5B evaluator cadence is not registered")
    parameters = manifest.get("parameters")
    expected_parameters = _ratified_evaluator_parameters()
    acceptance = manifest.get("acceptance")
    expected_acceptance = _ratified_acceptance_values()

    def require_exact_numeric_values(actual, expected, label):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"N5B evaluator {label} inventory is not ratified")
        for name, ratified in expected.items():
            value = actual[name]
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value != ratified):
                raise ValueError(
                    f"N5B evaluator {label} {name} differs from ratified F value")

    require_exact_numeric_values(
        parameters, expected_parameters, "parameter")
    require_exact_numeric_values(
        acceptance, expected_acceptance, "acceptance")
    if any(float(value) <= 0.0 for value in parameters.values()):
        raise ValueError("N5B evaluator parameters must be finite and positive")
    if (float(acceptance["reflectivity_fss_min"]) <= 0.0
            or float(acceptance["cold_pool_edge_distance_max_km"]) <= 0.0
            or float(acceptance["gust_front_arrival_mae_max_min"]) <= 0.0
            or float(acceptance["tke_ratio_min"]) <= 0.0
            or float(acceptance["tke_ratio_max"])
            <= float(acceptance["tke_ratio_min"])
            or acceptance["unmatched_boundary_ci_count"] != 0):
        raise ValueError("N5B evaluator acceptance values are invalid")
    environment = _perimeter_mask(
        CORE_SHAPE, math.ceil(float(expected_parameters[
            "cold_pool_environment_ring_km"])
                              * 1000 / dx_m))
    inflow = _perimeter_mask(
        CORE_SHAPE, math.ceil(float(expected_parameters[
            "inflow_fetch_perimeter_km"])
                              * 1000 / dx_m))
    masks = manifest.get("masks")
    if not isinstance(masks, dict):
        raise ValueError("N5B evaluator mask inventory is missing")
    expected_hashes = {
        "verification_core": _mask_hash(np.ones(CORE_SHAPE, dtype=bool)),
        "cold_pool_environment": _mask_hash(environment),
        "gust_arrival_union": _mask_hash(np.ones(CORE_SHAPE, dtype=bool)),
        "ci_object_overlap": _mask_hash(np.ones(CORE_SHAPE, dtype=bool)),
        "inflow_fetch": _mask_hash(inflow),
    }
    if (set(masks) != set(expected_hashes)
            or any(not isinstance(masks[name], dict)
                   or set(masks[name]) != {"shape", "sha256"}
                   or masks[name].get("shape") != list(CORE_SHAPE)
                   or masks[name].get("sha256") != expected
                   for name, expected in expected_hashes.items())):
        raise ValueError("N5B evaluator mask hash mismatch")
    return {"parameters": expected_parameters, "environment": environment,
            "inflow": inflow}


def load_evaluator_manifest(path: str | Path) -> dict[str, object]:
    """Load and fully validate one ratified evaluator manifest."""
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    _validated_evaluator(manifest, dx_m=F_GATE_D04_DX_M)
    return manifest


def _evaluate_series_pair(
        left: EvaluationSeries, right: EvaluationSeries, *,
        evaluator: Mapping[str, object],
        evaluation_authority: object,
        dx_m: float = F_GATE_D04_DX_M) -> dict[str, float]:
    """The sole five-predicate implementation for ensemble and shifted A/B."""
    if evaluation_authority is _SAME_GEOMETRY_AUTHORITY:
        if left.origin_cells != (0, 0) or right.origin_cells != (0, 0):
            raise ValueError(
                "N5B same-geometry authority cannot evaluate shifted series")
    elif not isinstance(evaluation_authority, real74_chain.VerifiedN5BFreeze):
        raise TypeError("N5B series evaluation requires verified authority")
    if not np.array_equal(left.times_seconds, right.times_seconds):
        raise ValueError("N5B series times differ")
    pinned = _validated_evaluator(evaluator, dx_m=dx_m)
    parameters = pinned["parameters"]
    environment = pinned["environment"]
    inflow = pinned["inflow"]

    def deficits(series):
        result = []
        for frame in series.core_t2_k:
            reference = float(np.median(frame[environment]))
            result.append(reference - frame)
        return np.asarray(result)

    left_deficit, right_deficit = deficits(left), deficits(right)
    return {
        "refl_fss": reflectivity_fss(
            left.core_refl_dbz, right.core_refl_dbz, dx_m=dx_m,
            event_dbz=float(parameters["reflectivity_event_dbz"]),
            radius_km=float(parameters["fss_radius_km"])),
        "cold_pool_edge_km": cold_pool_edge_distance_km(
            left_deficit, right_deficit, dx_m=dx_m,
            threshold_k=float(parameters["cold_pool_t2_deficit_k"])),
        "gust_front_arrival_mae_min": gust_front_arrival_mae_min(
            left_deficit, right_deficit, left.times_seconds,
            threshold_k=float(parameters["cold_pool_t2_deficit_k"])),
        "unmatched_boundary_ci_count": float(unmatched_boundary_ci_count(
            left.full_refl_dbz, right.full_refl_dbz, left.times_seconds,
            dx_m=dx_m, left_origin_cells=left.origin_cells,
            right_origin_cells=right.origin_cells,
            event_dbz=float(parameters["reflectivity_event_dbz"]),
            boundary_km=float(parameters["ci_boundary_km"]),
            persistence_min=float(parameters["ci_persistence_min"]),
            minimum_area_km2=float(parameters["ci_minimum_area_km2"]),
            match_km=float(parameters["ci_match_km"]),
            match_minutes=float(parameters["ci_match_minutes"]),
            temporal_link_km=float(parameters["ci_temporal_link_km"]))),
        "tke_ratio": resolved_tke_ratio(
            left.core_u, left.core_v, left.core_w,
            right.core_u, right.core_v, right.core_w, mask=inflow),
    }


def _evaluate_same_geometry_record_pair(
        left: Path, right: Path, *,
        evaluator: Mapping[str, object]) -> dict[str, float]:
    snapshots = [
        _snapshot_member_record(path, perturbed=None)
        for path in (left, right)]
    return _evaluate_same_geometry_snapshot_pair(
        snapshots[0], snapshots[1], evaluator=evaluator)


def _evaluate_same_geometry_snapshot_pair(
        left: _MemberRecordSnapshot, right: _MemberRecordSnapshot, *,
        evaluator: Mapping[str, object]) -> dict[str, float]:
    records = [left.document, right.document]
    if any(record.get("variant") != "production" for record in records):
        raise ValueError(
            "N5B pre-freeze pair evaluation is restricted to production geometry")
    if any(record.get("core_coordinate_sha256")
           != evaluator.get("core_coordinate_sha256") for record in records):
        raise ValueError("N5B same-geometry evaluator coordinates differ")
    return _evaluate_series_pair(
        _load_evaluation_series_snapshot(left),
        _load_evaluation_series_snapshot(right),
        evaluator=evaluator,
        evaluation_authority=_SAME_GEOMETRY_AUTHORITY)


def evaluate_record_pair(
        left: Path, right: Path, *,
        verified_freeze: "real74_chain.VerifiedN5BFreeze",
        evaluator_manifest_path: str | Path) -> dict[str, float]:
    """Evaluate shifted A/B only after reconstructing a verified freeze."""
    if not isinstance(verified_freeze, real74_chain.VerifiedN5BFreeze):
        raise TypeError("N5B shifted evaluation requires a verified-freeze token")
    fresh = real74_chain.verify_n5b_freeze(
        verified_freeze.path,
        expected_anchor_sha256=verified_freeze.anchor_sha256)
    if fresh.sha256 != verified_freeze.sha256:
        raise ValueError("N5B verified freeze changed before metric computation")
    evaluator_path = Path(evaluator_manifest_path).resolve()
    evaluator_bytes = evaluator_path.read_bytes()
    evaluator_hash = hashlib.sha256(evaluator_bytes).hexdigest()
    if evaluator_hash != fresh.evaluator_manifest_sha256:
        raise ValueError("N5B shifted evaluator differs from the frozen evaluator")
    evaluator = json.loads(evaluator_bytes)
    _validated_evaluator(evaluator, dx_m=F_GATE_D04_DX_M)
    records = [validate_member_record(path, perturbed=False)
               for path in (left, right)]
    if [record.get("variant") for record in records] != ["production", "shrink"]:
        raise ValueError("N5B shifted pair must be production then SHRINK")
    if any(record.get("core_coordinate_sha256")
           != evaluator.get("core_coordinate_sha256") for record in records):
        raise ValueError("N5B shifted evaluator coordinates differ from records")
    result = _evaluate_series_pair(
        load_evaluation_series(left), load_evaluation_series(right),
        evaluator=evaluator, evaluation_authority=fresh)
    real74_chain._recheck_n5b_dependencies(fresh)
    return result


def emit_observation(
        production_record: str | Path, shrink_record: str | Path,
        frozen_manifest: str | Path, output: str | Path, *,
        freeze_anchor_sha256: str,
        evaluator_manifest_path: str | Path) -> dict[str, object]:
    """Evaluate A/B with the same extractors and emit N5B-score input."""
    verified_freeze = real74_chain.verify_n5b_freeze(
        frozen_manifest, expected_anchor_sha256=freeze_anchor_sha256)
    output = Path(output).resolve()
    root = output.parent
    evaluator_path = Path(evaluator_manifest_path).resolve()
    if not evaluator_path.is_relative_to(root):
        raise ValueError("N5B evaluator manifest must be beneath observation root")
    evaluator_bytes = evaluator_path.read_bytes()
    evaluator_hash = hashlib.sha256(evaluator_bytes).hexdigest()
    if evaluator_hash != verified_freeze.evaluator_manifest_sha256:
        raise ValueError("N5B observation evaluator differs from the frozen evaluator")
    evaluator = json.loads(evaluator_bytes)
    _validated_evaluator(evaluator, dx_m=F_GATE_D04_DX_M)
    production_path, shrink_path = Path(production_record), Path(shrink_record)
    production = validate_member_record(production_path, perturbed=False)
    shrink = validate_member_record(shrink_path, perturbed=False)
    if production.get("variant") != "production" or shrink.get("variant") != "shrink":
        raise ValueError("N5B observation needs production then SHRINK records")
    if production["restored_input_sha256"] != shrink["restored_input_sha256"]:
        raise ValueError("N5B production/SHRINK restored inputs differ")
    if production["core_coordinate_sha256"] != shrink["core_coordinate_sha256"]:
        raise ValueError("N5B production/SHRINK common-core coordinates differ")
    if evaluator["core_coordinate_sha256"] != production["core_coordinate_sha256"]:
        raise ValueError("N5B observation evaluator coordinates differ from records")
    artifacts = {}
    for name, path, record in (
            ("production", production_path.resolve(), production),
            ("shrink", shrink_path.resolve(), shrink)):
        if not path.is_relative_to(root):
            raise ValueError("N5B observation artifacts must be beneath output root")
        artifacts[name] = {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": shared.sha256_file(path),
            "restored_input_sha256": record["restored_input_sha256"],
            "core_coordinate_sha256": record["core_coordinate_sha256"],
            "duration_seconds": record["duration_seconds"],
        }
    metrics = evaluate_record_pair(
        production_path, shrink_path,
        verified_freeze=verified_freeze,
        evaluator_manifest_path=evaluator_path)
    observed = {
        "schema": 1, "metric": METRIC,
        "frozen_manifest_sha256": verified_freeze.sha256,
        "freeze_anchor_sha256": verified_freeze.anchor_sha256,
        "evaluator_manifest_sha256": evaluator_hash,
        "cadence": f"{CADENCE_SECONDS}-second d04 history frames",
        "artifacts": artifacts, "metrics": metrics,
        "evaluator_commit": shared._git_commit(),
        "generated_utc": shared._utc_now(),
        "evaluator_manifest": {
            "relative_path": evaluator_path.relative_to(root).as_posix(),
            "sha256": evaluator_hash,
        },
    }
    shared.write_json(output, observed)
    return observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one or the default three members")
    run.add_argument("--variant", choices=("production", "shrink"),
                     default="production")
    run.add_argument("--members", nargs="+",
                     choices=("unperturbed", *DEFAULT_PERTURBATION_BY_ID),
                     default=None)
    run.add_argument("--out-root", type=Path, required=True)
    run.add_argument("--config", type=Path, default=shared.PRODUCTION_CONFIG)
    run.add_argument("--plan", type=Path, default=None)
    run.add_argument("--shrink-input-manifest", type=Path, default=None)
    crop = commands.add_parser(
        "generate-shrink-inputs", help="crop production d04 inputs for F22")
    crop.add_argument("--config", type=Path, default=shared.PRODUCTION_CONFIG)
    crop.add_argument("--output-dir", type=Path, required=True)
    evaluator = commands.add_parser("evaluator-manifest")
    evaluator.add_argument("--variant", choices=("production", "shrink"),
                           default="production")
    evaluator.add_argument("--config", type=Path, default=shared.PRODUCTION_CONFIG)
    evaluator.add_argument("--plan", type=Path, default=None)
    evaluator.add_argument("--shrink-input-manifest", type=Path, default=None)
    evaluator.add_argument("--output", type=Path, required=True)
    coordinates = commands.add_parser("assert-coordinates")
    coordinates.add_argument("--config", type=Path, default=shared.PRODUCTION_CONFIG)
    coordinates.add_argument("--plan", type=Path, default=None)
    ensemble = commands.add_parser("ensemble-manifest")
    ensemble.add_argument("--member-record", action="append", type=Path, required=True)
    ensemble.add_argument("--unperturbed-record", type=Path, required=True)
    ensemble.add_argument("--evaluator-manifest", type=Path, required=True)
    ensemble.add_argument("--output", type=Path, required=True)
    observation = commands.add_parser("observation")
    observation.add_argument("--production-record", type=Path, required=True)
    observation.add_argument("--shrink-record", type=Path, required=True)
    observation.add_argument("--frozen-manifest", type=Path, required=True)
    observation.add_argument("--freeze-anchor-sha256", required=True)
    observation.add_argument("--evaluator-manifest", type=Path, required=True)
    observation.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        members = args.members
        if args.variant == "shrink" and members is None:
            members = ["unperturbed"]
        records = run_default_members(
            args.out_root, member_ids=members, variant=args.variant,
            config_path=args.config, plan_path=args.plan,
            shrink_input_manifest=args.shrink_input_manifest)
        print(json.dumps({"records": [str(item) for item in records]}, indent=2))
    elif args.command == "generate-shrink-inputs":
        print(generate_shrink_d04_inputs(
            args.output_dir, config_path=args.config))
    elif args.command == "evaluator-manifest":
        exp, _, _ = construct_variant(
            args.variant, config_path=args.config, plan_path=args.plan,
            shrink_input_manifest=args.shrink_input_manifest)
        write_evaluator_manifest(args.output, exp, variant=args.variant)
        print(args.output)
    elif args.command == "assert-coordinates":
        print(assert_common_core_coordinates(
            config_path=args.config, plan_path=args.plan))
    elif args.command == "ensemble-manifest":
        emit_ensemble_manifest(
            args.member_record, args.unperturbed_record, args.output,
            evaluator_manifest_path=args.evaluator_manifest)
        print(args.output)
    elif args.command == "observation":
        emit_observation(
            args.production_record, args.shrink_record,
            args.frozen_manifest, args.output,
            freeze_anchor_sha256=args.freeze_anchor_sha256,
            evaluator_manifest_path=args.evaluator_manifest)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PERTURBATIONS", "EvaluationSeries", "PerturbationSpec",
    "apply_one_ulp", "assert_common_core_coordinates", "binary_edge",
    "cold_pool_edge_distance_km", "construct_variant", "core_coordinate_sha256",
    "emit_ensemble_manifest", "emit_observation", "evaluator_manifest",
    "generate_shrink_d04_inputs", "gust_front_arrival_mae_min",
    "load_evaluator_manifest", "metrics_to_discrepancies",
    "ratified_discrepancy_limits",
    "reflectivity_fss", "resolved_tke_ratio", "run_default_members",
    "run_member", "unmatched_boundary_ci_count", "validate_member_record",
]
