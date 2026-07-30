"""Source-neutral join from prepared regular-grid forcing to a WRF nest.

GFS and ERA5 use different decoders and slightly different root-domain
surface treatment, but the hierarchy transaction after root preparation is
identical.  This module owns that common, fail-closed handoff: the complete
forcing time series remains attached to d01 as its external LBC sequence,
while one independently verified WPS_GEOG catalog and the initial source
snapshot feed the child initialization barriers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from gpuwm.case_data import (
    PerDomainSourceOrography,
    SourceOrography,
    SourceOrographyDeclaration,
    resolve_source_orography,
)
from gpuwm.hrrr_native_static import verified_static_catalog
from gpuwm.ingest.horiz import _regular_coordinates
from gpuwm.ingest.nest_init import NestedInputCatalog, ParentInitView
from gpuwm.native_hierarchy import (
    NativeHierarchyExportResult,
    initialize_and_export_native_hierarchy,
)


_IMPLEMENTATION_PATHS = (
    "gpuwm/source_hierarchy.py",
    "gpuwm/native_hierarchy.py",
    "gpuwm/native_domain_artifacts.py",
    "gpuwm/hrrr_native_static.py",
    "gpuwm/ingest/nest_init.py",
    "gpuwm/static/build.py",
    "gpuwm/wrf_direct.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    missing = tuple(
        relative for relative in _IMPLEMENTATION_PATHS
        if not (root / relative).is_file())
    if missing:
        raise RuntimeError(
            f"regular-source hierarchy implementation is incomplete: "
            f"{missing}")
    return {
        relative: _sha256(root / relative)
        for relative in _IMPLEMENTATION_PATHS
    }


@dataclass(frozen=True)
class RegularSourceHierarchyResult:
    """A native hierarchy export plus its independently verified GEOG bind."""

    hierarchy: NativeHierarchyExportResult
    static_catalog_receipt: Mapping[str, object]
    source_coverage_receipt: Mapping[str, object]
    topology_receipt: Mapping[str, object]
    forcing_times: tuple[datetime, ...]
    boundary_interval_seconds: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "static_catalog_receipt",
            MappingProxyType(dict(self.static_catalog_receipt)),
        )
        object.__setattr__(
            self, "source_coverage_receipt",
            MappingProxyType(dict(self.source_coverage_receipt)),
        )
        object.__setattr__(
            self, "topology_receipt",
            MappingProxyType(dict(self.topology_receipt)),
        )


def _validated_static_one_way_topology(exp, grids) -> dict[str, object]:
    """Bind the complete source-neutral WRF hierarchy before any work."""

    domains = tuple(exp.domains)
    grids = tuple(grids)
    if len(domains) < 2:
        raise ValueError(
            "regular-grid hierarchy requires at least one child domain")
    if len(grids) != len(domains):
        raise ValueError(
            "regular-grid hierarchy requires one validated grid per domain")
    ids = tuple(int(domain.grid_id) for domain in domains)
    expected_ids = tuple(range(1, len(domains) + 1))
    if ids != expected_ids:
        raise ValueError(
            "regular-grid hierarchy ids must be contiguous d01..dNN in "
            f"parent-before-child order, got {ids}")
    if int(getattr(exp, "feedback", -1)) != 0:
        # The gate is right and unchanged.  What it never said is where
        # feedback=1 IS supported, so a node-7 validation run authored a
        # two-way config, passed `gpuwm check` clean, and learned only
        # after a 26 s hierarchy build that this route cannot do two-way
        # nesting at all.  A refusal that names the route that can is
        # the difference between a dead end and a redirection.
        raise ValueError(
            "regular-grid hierarchy export supports static one-way nests "
            "only (feedback=0).  Experimental two-way feedback "
            "(feedback=1) runs on the native experiment-runner route -- "
            "`gpuwm run`, which builds its domain tree in-process from "
            "the same experiment TOML -- and is stamped experimental "
            "there; it is not available through a prepared hierarchy, "
            "whose artifacts are written one-way and read one-way.")

    seen: set[int] = set()
    rows = []
    for domain in domains:
        grid_id = int(domain.grid_id)
        parent_id = int(domain.parent_id)
        run = domain.run
        root = grid_id == 1
        if root:
            if parent_id != 0:
                raise ValueError("d01 must be the sole root with parent_id=0")
            if run.specified is not True or run.nested is not False:
                raise ValueError(
                    "d01 requires specified=true and nested=false")
        else:
            if parent_id not in seen:
                raise ValueError(
                    f"d{grid_id:02d} parent d{parent_id:02d} must precede it")
            if run.specified is not False or run.nested is not True:
                raise ValueError(
                    f"d{grid_id:02d} requires specified=false and nested=true")
        seen.add(grid_id)
        configured_start = getattr(domain, "start_time", None)
        domain_start = (
            exp.domain_start_time(grid_id)
            if hasattr(exp, "domain_start_time") else
            (exp.start_time if configured_start is None else configured_start))
        rows.append({
            "grid_id": grid_id,
            "parent_id": parent_id,
            "start_time": domain_start.isoformat(),
            "i_parent_start": int(domain.i_parent_start),
            "j_parent_start": int(domain.j_parent_start),
            "parent_grid_ratio": int(domain.parent_grid_ratio),
            "parent_time_step_ratio": int(domain.parent_time_step_ratio),
            "e_we": int(run.nx) + 1,
            "e_sn": int(run.ny) + 1,
            "dx": float(run.dx),
            "dy": float(run.dy),
            "dt": float(run.dt),
            "nz": int(run.nz),
        })
    return {
        "schema": "gpuwm-regular-source-static-one-way-topology-v1",
        "status": "PASS",
        "max_dom": len(rows),
        "feedback": 0,
        "domains": rows,
    }


def _validated_forcing_series(
        exp, snapshots: Sequence[object],
        forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
) -> tuple[tuple[object, ...], tuple[int, ...], tuple[datetime, ...], int]:
    snapshots = tuple(snapshots)
    if (forcing_hours is None) == (forcing_offsets_seconds is None):
        raise ValueError(
            "nested source forcing requires exactly one of forcing_hours "
            "or forcing_offsets_seconds")
    legacy_hours = None if forcing_hours is None else tuple(forcing_hours)
    if legacy_hours is not None:
        if any(isinstance(hour, bool) or not isinstance(hour, int)
               for hour in legacy_hours):
            raise TypeError("nested source forcing hours must be integers")
        offsets = tuple(hour * 3600 for hour in legacy_hours)
        coordinate_name = "forcing hours"
    else:
        offsets = tuple(forcing_offsets_seconds)
        if any(isinstance(offset, bool) or not isinstance(offset, int)
               for offset in offsets):
            raise TypeError(
                "nested source forcing offsets must be integer seconds")
        coordinate_name = "forcing offsets"
    if len(snapshots) != len(offsets) or len(snapshots) < 2:
        raise ValueError(
            "nested source forcing requires matching snapshots/offsets and "
            "at least two boundary times")
    if offsets[0] != 0 or any(
            later <= earlier
            for earlier, later in zip(offsets, offsets[1:])):
        raise ValueError(
            "nested source forcing offsets must begin at zero and increase")
    deltas = {
        later - earlier
        for earlier, later in zip(offsets, offsets[1:])}
    if len(deltas) != 1:
        raise ValueError("nested source forcing cadence must be uniform")
    interval_seconds = deltas.pop()
    if interval_seconds <= 0:
        raise ValueError("nested source forcing cadence must be positive")

    times = tuple(getattr(snapshot, "valid_time", None)
                  for snapshot in snapshots)
    if not all(isinstance(value, datetime) for value in times):
        raise TypeError("every nested source snapshot needs a datetime")
    if times[0] != exp.start_time:
        raise ValueError(
            f"nested source forcing must begin at {exp.start_time}, got "
            f"{times[0]}")
    for time_value, offset in zip(times, offsets):
        if (time_value - exp.start_time).total_seconds() != offset:
            raise ValueError(
                f"nested source snapshot times differ from {coordinate_name}")
    if offsets[-1] < exp.run_seconds:
        raise ValueError("nested source forcing does not cover the run")
    if len({type(snapshot) for snapshot in snapshots}) != 1:
        raise TypeError("nested source snapshots must use one adapter type")
    return snapshots, offsets, times, interval_seconds


def _validate_source_orography(
        exp, snapshots: tuple[object, ...],
        declaration: SourceOrographyDeclaration | None,
        source_inventory: Sequence[str],
) -> dict[str, object]:
    fields = getattr(snapshots[0], "fields", {})
    invariant = "SOILGEO" in tuple(source_inventory)
    embedded = "SOURCE_OROGRAPHY" in fields
    if invariant and declaration is not None:
        raise ValueError(
            "source-orography conflict: forcing declares invariant SOILGEO "
            "and external per-domain artifacts")
    if invariant or embedded:
        if declaration is not None:
            raise ValueError(
                "source-orography conflict: forcing already carries an "
                "orography provider")
        return {
            "provider": (
                "catalog-invariant-SOILGEO" if invariant
                else "embedded-SOURCE_OROGRAPHY"),
        }
    if declaration is None:
        raise ValueError(
            "regular-grid hierarchy requires embedded SOURCE_OROGRAPHY, "
            "catalog SOILGEO, or explicit per-domain source_orography")
    if len(exp.domains) > 1 and not isinstance(
            declaration, PerDomainSourceOrography):
        raise ValueError(
            "nested source_orography must be declared separately for every "
            "domain; a legacy d01 artifact cannot be reused by children")
    if isinstance(declaration, PerDomainSourceOrography):
        declared_ids = tuple(
            int(domain_id) for domain_id, _artifact in declaration.by_domain)
        expected_ids = tuple(int(domain.grid_id) for domain in exp.domains)
        if (len(set(declared_ids)) != len(declared_ids)
                or set(declared_ids) != set(expected_ids)):
            raise ValueError(
                "per-domain source_orography ids must exactly cover the "
                f"hierarchy: expected {expected_ids}, got {declared_ids}")
    bound = {}
    for domain in exp.domains:
        artifact = resolve_source_orography(
            declaration, int(domain.grid_id))
        if not isinstance(artifact, SourceOrography):
            raise ValueError(
                f"d{int(domain.grid_id):02d} lacks source_orography")
        if not Path(artifact.path).is_file():
            raise FileNotFoundError(artifact.path)
        path = Path(artifact.path).resolve()
        bound[f"d{int(domain.grid_id):02d}"] = {
            "path": str(path),
            "variable": artifact.variable,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return {"provider": "per-domain-artifacts", "domains": bound}


def _spatial_coverage_receipt(
        snapshots: tuple[object, ...], grids: tuple[object, ...], exp,
        source_name: str,
) -> dict[str, object]:
    """Prove every child staggering has complete source donor halos."""

    first_latitude = np.asarray(
        getattr(snapshots[0], "latitude", ()), dtype=np.float64)
    first_longitude = np.asarray(
        getattr(snapshots[0], "longitude", ()), dtype=np.float64)
    if (first_latitude.ndim != 1 or first_longitude.ndim != 1
            or first_latitude.size < 4 or first_longitude.size < 4
            or not np.isfinite(first_latitude).all()
            or not np.isfinite(first_longitude).all()):
        raise ValueError(
            f"{source_name} hierarchy source axes are not finite 1-D donor "
            "coordinates")
    for snapshot in snapshots[1:]:
        latitude = np.asarray(
            getattr(snapshot, "latitude", ()), dtype=np.float64)
        longitude = np.asarray(
            getattr(snapshot, "longitude", ()), dtype=np.float64)
        if (not np.array_equal(latitude, first_latitude)
                or not np.array_equal(longitude, first_longitude)):
            raise ValueError(
                f"{source_name} hierarchy source grid changes between "
                "forcing times")

    ny = int(first_latitude.size)
    nx = int(first_longitude.size)
    domains = {}
    for domain, grid in zip(exp.domains, grids):
        staggering_receipts = {}
        for staggering, coordinates in (
                ("mass", grid.latlon_mass()),
                ("u", grid.latlon_u()),
                ("v", grid.latlon_v())):
            target_latitude, target_longitude = coordinates
            y, x = _regular_coordinates(
                first_latitude, first_longitude,
                target_latitude, target_longitude)
            floor_y = np.floor(y).astype(np.int64)
            floor_x = np.floor(x).astype(np.int64)
            parabolic = {
                "x": [int(floor_x.min()) - 1, int(floor_x.max()) + 2],
                "y": [int(floor_y.min()) - 1, int(floor_y.max()) + 2],
            }
            if (parabolic["x"][0] < 0 or parabolic["x"][1] >= nx
                    or parabolic["y"][0] < 0
                    or parabolic["y"][1] >= ny):
                raise ValueError(
                    f"{source_name} source lacks the complete parabolic "
                    f"donor halo for d{int(domain.grid_id):02d} "
                    f"{staggering}")
            staggering_receipts[staggering] = {
                "source_x_range": [float(x.min()), float(x.max())],
                "source_y_range": [float(y.min()), float(y.max())],
                "parabolic_donor_x": parabolic["x"],
                "parabolic_donor_y": parabolic["y"],
            }
        # Masked-chain coverage contract (v2): the deterministic operators
        # (sixteen_pt/four_pt/wt averages) reach the same floor-based
        # [-1, +2] stencil as the parabolic receipt -- a crop must contain
        # that halo or edge clamping would diverge from full-grid WPS.  The
        # search fallback is a WPS FIFO BFS (depth 1200) whose reach is the
        # whole cropped source, so no fixed radius can certify it; instead
        # the crop must retain usable donors of BOTH surface classes,
        # otherwise a masked class would silently fill where full-grid WPS
        # finds a remote donor.
        mass_latitude, mass_longitude = grid.latlon_mass()
        mass_y, mass_x = _regular_coordinates(
            first_latitude, first_longitude,
            mass_latitude, mass_longitude)
        floor_my = np.floor(mass_y).astype(np.int64)
        floor_mx = np.floor(mass_x).astype(np.int64)
        masked = {
            "x": [int(floor_mx.min()) - 1, int(floor_mx.max()) + 2],
            "y": [int(floor_my.min()) - 1, int(floor_my.max()) + 2],
        }
        if (masked["x"][0] < 0 or masked["x"][1] >= nx
                or masked["y"][0] < 0 or masked["y"][1] >= ny):
            raise ValueError(
                f"{source_name} source lacks the complete masked-surface "
                f"deterministic donor halo for d{int(domain.grid_id):02d}")
        domains[f"d{int(domain.grid_id):02d}"] = {
            "staggerings": staggering_receipts,
            "masked_surface_donor_x": masked["x"],
            "masked_surface_donor_y": masked["y"],
        }
    # Class support is reported, not enforced: a source legitimately may be
    # single-class (all-water crop for an oceanic case), and a class that is
    # absent from the FULL source also fills in WPS.  The counts make a
    # crop that lost a class visible in the receipt.
    class_support = None
    first_fields = getattr(snapshots[0], "fields", {})
    if "LANDSEA" in first_fields:
        landsea = np.asarray(first_fields["LANDSEA"], dtype=np.float64)
        class_support = {
            "land": int(np.sum(landsea >= 0.5)),
            "water": int(np.sum(landsea < 0.5)),
        }
    return {
        "schema": "gpuwm-regular-source-hierarchy-coverage-v2",
        "status": "PASS",
        "source": source_name,
        "source_shape": [ny, nx],
        "parabolic_required_offsets": [-1, 2],
        "masked_deterministic_donor_offsets": [-1, 2],
        "masked_search_reach": "full-cropped-source (WPS FIFO BFS, depth 1200)",
        "masked_class_support": class_support,
        "domains": domains,
    }


def initialize_and_export_regular_source_hierarchy(
        *, exp, grids, snapshots: Sequence[object],
        forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
        wps_namelist: Path, geog_root: Path,
        source_name: str, artifact_output: Path, wrf_output: Path,
        root_initial_result, root_met, root_soil, root_static_fields,
        root_boundaries, bridge_manifest_sha256: str,
        source_manifest_sha256: str, namelist_sha256: str,
        source_identity: Mapping[str, object],
        source_orography: SourceOrographyDeclaration | None = None,
        source_inventory: Sequence[str] = (),
        source_units: Mapping[str, str] | None = None,
        workers: int = 8, preprocess_backend: str = "cpu",
        cpu_bridge=None, sfcp_to_sfcp: bool = True,
        soil_layer_contract=None,
        root_metadata: Mapping[str, object] | None = None,
        input_provenance: Mapping[str, object] | None = None,
        artifact_manifest_reference: str | None = None,
) -> RegularSourceHierarchyResult:
    """Feed a prepared GFS/ERA5 root and verified child inputs to the join.

    The caller remains responsible for decoding and initializing every root
    forcing time.  This function verifies that series, binds all domain
    geometry/static inputs, and performs the existing atomic native-artifact
    plus unchanged-WRF export transaction.
    """

    grids = tuple(grids)
    topology_receipt = _validated_static_one_way_topology(exp, grids)
    if (isinstance(workers, bool) or not isinstance(workers, int)
            or not 1 <= workers <= 32):
        raise ValueError("hierarchy workers must be an integer in [1, 32]")
    backend = preprocess_backend.strip().lower()
    if backend not in {"cpu", "cuda"}:
        raise ValueError(
            "hierarchy preprocessing backend must be explicit cpu or cuda")
    if backend != "cpu" and workers != 1:
        raise ValueError(
            "CUDA hierarchy preprocessing is deterministic only with "
            "workers=1")

    snapshots, offsets, times, interval = _validated_forcing_series(
        exp, snapshots, forcing_hours, forcing_offsets_seconds)
    # Production ExperimentConfig carries the exact rational domain clocks.
    # Lightweight source-adapter unit fixtures intentionally do not.
    if hasattr(exp, "dt_exact") and hasattr(exp, "domain_start_offset_exact"):
        from gpuwm.experiment import validate_boundary_timing
        validate_boundary_timing(
            exp, interval, source=f"{source_name} hierarchy forcing")
    inventory = tuple(source_inventory)
    units = dict(source_units or {})
    if "SOILGEO" in inventory and str(
            units.get("SOILGEO", "")).replace(" ", "") not in {
                "m2s-2", "m^2s^-2", "m**2s**-2"}:
        raise ValueError(
            "SOILGEO hierarchy forcing requires geopotential units "
            "m2 s-2")
    orography_receipt = _validate_source_orography(
        exp, snapshots, source_orography, inventory)
    source_coverage_receipt = _spatial_coverage_receipt(
        snapshots, grids, exp, source_name)

    state = root_initial_result.state
    if getattr(state, "lateral_boundaries", None) is not root_boundaries:
        raise ValueError(
            "prepared root state does not carry its complete external LBC "
            "sequence")
    static_catalog, static_receipt = verified_static_catalog(
        Path(wps_namelist), Path(geog_root),
        [domain.grid_id for domain in exp.domains],
    )
    catalog_provenance = {
        "adapter": f"{source_name.lower()}-regular-grid-hierarchy-v1",
        "source_manifest_sha256": source_manifest_sha256,
        "static_catalog_receipt": static_receipt,
        "source_orography": orography_receipt,
        "source_coverage": source_coverage_receipt,
    }
    catalog = NestedInputCatalog(
        snapshots=snapshots,
        static_catalog=static_catalog,
        inventory=inventory,
        files=tuple(static_catalog.files),
        units=units,
        provenance=catalog_provenance,
    )
    root = ParentInitView(
        cfg=exp.root, grid=grids[0], state=state)
    provenance = dict(input_provenance or {})
    implementation_sha256 = _implementation_sha256()
    provenance.update({
        "regular_source_adapter": source_name.lower(),
        "regular_source_forcing_times": [value.isoformat() for value in times],
        "regular_source_forcing_offsets_seconds": list(offsets),
        "regular_source_static_catalog": static_receipt,
        "regular_source_orography": orography_receipt,
        "regular_source_coverage": source_coverage_receipt,
        "regular_source_topology": topology_receipt,
        "regular_source_hierarchy_implementation_sha256": (
            implementation_sha256),
    })
    bound_source_identity = dict(source_identity)
    reserved_identity = {
        "nested_source_orography", "hierarchy_implementation_sha256"}
    conflict = reserved_identity & set(bound_source_identity)
    if conflict:
        raise ValueError(
            f"source_identity overrides reserved hierarchy keys "
            f"{sorted(conflict)}")
    bound_source_identity["nested_source_orography"] = orography_receipt
    bound_source_identity["hierarchy_implementation_sha256"] = (
        implementation_sha256)
    hierarchy = initialize_and_export_native_hierarchy(
        exp=exp,
        root_node=root,
        catalog=catalog,
        artifact_output=Path(artifact_output),
        wrf_output=Path(wrf_output),
        root_initial_result=root_initial_result,
        root_met=root_met,
        root_soil=root_soil,
        root_static_fields=root_static_fields,
        root_boundaries=root_boundaries,
        bridge_manifest_sha256=bridge_manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
        namelist_sha256=namelist_sha256,
        forcing_hours=(
            tuple(forcing_hours) if forcing_hours is not None else None),
        forcing_offsets_seconds=(
            offsets if forcing_offsets_seconds is not None else None),
        source_identity=bound_source_identity,
        source_orography=source_orography,
        workers=workers,
        preprocess_backend=backend,
        cpu_bridge=cpu_bridge,
        boundary_interval_seconds=interval,
        sfcp_to_sfcp=sfcp_to_sfcp,
        soil_layer_contract=soil_layer_contract,
        root_metadata=root_metadata,
        input_provenance=provenance,
        artifact_manifest_reference=artifact_manifest_reference,
    )
    return RegularSourceHierarchyResult(
        hierarchy=hierarchy,
        static_catalog_receipt=static_receipt,
        source_coverage_receipt=source_coverage_receipt,
        topology_receipt=topology_receipt,
        forcing_times=times,
        boundary_interval_seconds=interval,
    )


__all__ = [
    "RegularSourceHierarchyResult",
    "initialize_and_export_regular_source_hierarchy",
]
