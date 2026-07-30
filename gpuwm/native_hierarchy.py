"""End-to-end join from native root state to stock-WRF nested inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import time
from typing import Mapping, Sequence

from gpuwm.ingest.nest_init import initialize_child_chain_parallel
from gpuwm.native_domain_artifacts import (
    NativeHierarchyArtifactBuild,
    write_native_hierarchy_artifacts,
)
from gpuwm.wrf_direct import export_prepared_wrf_hierarchy


@dataclass(frozen=True)
class NativeHierarchyExportResult:
    """Completed native hierarchy artifacts plus unchanged-WRF products."""

    artifacts: NativeHierarchyArtifactBuild
    wrf_manifest: Mapping[str, object]
    timings_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "wrf_manifest", MappingProxyType(dict(self.wrf_manifest)))
        object.__setattr__(
            self, "timings_seconds",
            MappingProxyType(dict(self.timings_seconds)))


def initialize_and_export_native_hierarchy(
        *, exp, root_node, catalog, artifact_output: Path,
        wrf_output: Path, root_initial_result, root_met, root_soil,
        root_static_fields, root_boundaries,
        bridge_manifest_sha256: str, source_manifest_sha256: str,
        namelist_sha256: str, forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
        source_identity: Mapping[str, object], source_orography=None,
        workers: int = 8, preprocess_backend="cpu", cpu_bridge=None,
        boundary_interval_seconds: int = 3600, scratch_arena=None,
        dycore_state_workspace=None, sfcp_to_sfcp: bool = True,
        soil_layer_contract=None,
        root_metadata: Mapping[str, object] | None = None,
        input_provenance: Mapping[str, object] | None = None,
        artifact_manifest_reference: str | None = None,
) -> NativeHierarchyExportResult:
    """Prepare children in parallel, join artifacts, and emit WRF files.

    The caller supplies the already prepared root because its complete source
    time series owns the sole external LBC sequence.  Child static/source
    mapping is launched concurrently with an explicit worker budget, then
    finalized at parent barriers before the atomic artifact tree and final
    ``wrfinput_d01..dNN``/``wrfbdy_d01`` directory are written.
    """

    if root_node.state is not root_initial_result.state:
        raise ValueError(
            "root node and root initial result do not share the same state")
    if int(root_node.cfg.grid_id) != int(exp.domains[0].grid_id):
        raise ValueError("root node does not match the experiment root domain")
    if getattr(root_node.state, "lateral_boundaries", None) is not \
            root_boundaries:
        raise ValueError("root state does not carry the supplied boundaries")
    provenance = dict(input_provenance or {})
    reserved_provenance = {
        "native_artifact_manifest",
        "native_artifact_manifest_sha256",
    }
    conflict = reserved_provenance & set(provenance)
    if conflict:
        raise ValueError(
            f"input provenance overrides reserved keys {sorted(conflict)}")
    for path, label in ((Path(artifact_output), "artifact"),
                        (Path(wrf_output), "WRF output")):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {label} path {path}")
    timings: dict[str, float] = {}
    started = time.perf_counter()
    child_results = initialize_child_chain_parallel(
        exp, root_node, catalog, source_orography,
        workers=workers, preprocess_backend=preprocess_backend,
        cpu_bridge=cpu_bridge, scratch_arena=scratch_arena,
        dycore_state_workspace=dycore_state_workspace,
        state_backend="preprocess",
        sfcp_to_sfcp=sfcp_to_sfcp,
        soil_layer_contract=soil_layer_contract)
    timings["parallel_child_initialization"] = time.perf_counter() - started

    started = time.perf_counter()
    artifact_build = write_native_hierarchy_artifacts(
        artifact_output, exp=exp, root_grid=root_node.grid,
        root_initial_result=root_initial_result, root_met=root_met,
        root_soil=root_soil, root_static_fields=root_static_fields,
        root_boundaries=root_boundaries, child_results=child_results,
        bridge_manifest_sha256=bridge_manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
        namelist_sha256=namelist_sha256, forcing_hours=forcing_hours,
        forcing_offsets_seconds=forcing_offsets_seconds,
        source_identity=source_identity, valid_time=exp.start_time,
        root_metadata=root_metadata)
    timings["verified_hierarchy_artifacts"] = time.perf_counter() - started

    started = time.perf_counter()
    provenance.update({
        "native_artifact_manifest": (
            str(artifact_build.manifest)
            if artifact_manifest_reference is None
            else artifact_manifest_reference),
        "native_artifact_manifest_sha256": artifact_build.receipt[
            "manifest"]["sha256"],
    })
    wrf_manifest = export_prepared_wrf_hierarchy(
        exp, artifact_build.artifacts, wrf_output,
        valid_time=exp.start_time,
        boundary_interval_seconds=boundary_interval_seconds,
        input_provenance=provenance)
    timings["direct_stock_wrf_export"] = time.perf_counter() - started
    timings["total"] = sum(timings.values())
    return NativeHierarchyExportResult(
        artifacts=artifact_build,
        wrf_manifest=wrf_manifest,
        timings_seconds=timings)


__all__ = [
    "NativeHierarchyExportResult",
    "initialize_and_export_native_hierarchy",
]
