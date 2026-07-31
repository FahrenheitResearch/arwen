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
from gpuwm.wrf_direct import (
    StockWrfExportUnsupported,
    export_prepared_wrf_hierarchy,
    stock_wrf_export_not_requested,
    stock_wrf_export_refused,
)

#: What a caller may ask of the stock-WRF export leg of this join.
#:
#: ``"required"``  the export is part of the caller's product; a refusal
#:                 on export-representability propagates and fails the
#:                 whole preparation.
#: ``"optional"``  the caller is preparing a FORECAST and the export is a
#:                 bonus product.  A refusal is recorded in the export
#:                 slot -- with the gate's own message and its named
#:                 selector deltas -- and the preparation proceeds.
#: ``"off"``       no export is attempted at all.
#:
#: The distinction exists because this join used to call the exporter
#: unconditionally, so `export_prepared_wrf`'s profile-free compatibility
#: branch -- which requires the v2 stock slice bl_pbl_physics=1,
#: sf_sfclay_physics=91, sf_surface_physics=2 of the tree ROOT -- decided
#: which domain trees could be prepared on the GFS route at all.  It
#: overruled registry reachability, an accepted `expert-tuple-v1`
#: acknowledgement and an explicitly named shipped profile alike: the only
#: preparable root was YSU + classic-MM5 + Noah, which is exactly what the
#: one shipped hierarchy config happens to be, which is why nothing caught
#: it.  Nothing about the gate's CONTENT changes here -- the export still
#: refuses what it always refused, and still writes no file when it does.
STOCK_WRF_EXPORT_MODES = ("required", "optional", "off")


@dataclass(frozen=True)
class NativeHierarchyExportResult:
    """Completed native hierarchy artifacts plus unchanged-WRF products.

    ``wrf_manifest`` is the export slot: a READY manifest with a ``files``
    inventory when the export ran, otherwise the NOT_REQUESTED/REFUSED
    document that says why it did not.
    """

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
        stock_wrf_export: str = "required",
) -> NativeHierarchyExportResult:
    """Prepare children in parallel, join artifacts, and emit WRF files.

    The caller supplies the already prepared root because its complete source
    time series owns the sole external LBC sequence.  Child static/source
    mapping is launched concurrently with an explicit worker budget, then
    finalized at parent barriers before the atomic artifact tree and final
    ``wrfinput_d01..dNN``/``wrfbdy_d01`` directory are written.

    ``stock_wrf_export`` says what that last step is worth to the caller;
    see :data:`STOCK_WRF_EXPORT_MODES`.  It defaults to ``"required"``,
    the behaviour every caller had when the export was an unconditional
    call, so opting into the softer modes is a decision a source adapter
    makes explicitly rather than one it inherits.
    """

    if stock_wrf_export not in STOCK_WRF_EXPORT_MODES:
        raise ValueError(
            f"stock_wrf_export must be one of {list(STOCK_WRF_EXPORT_MODES)}, "
            f"got {stock_wrf_export!r}")
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
    if stock_wrf_export == "off":
        wrf_manifest = stock_wrf_export_not_requested()
    else:
        try:
            wrf_manifest = export_prepared_wrf_hierarchy(
                exp, artifact_build.artifacts, wrf_output,
                valid_time=exp.start_time,
                boundary_interval_seconds=boundary_interval_seconds,
                input_provenance=provenance)
        except StockWrfExportUnsupported as error:
            if stock_wrf_export == "required":
                raise
            # The forecast is already prepared: `artifact_build` above is
            # the complete, verified hierarchy the GPU runner consumes.
            # Only the unchanged-WRF file set cannot represent this
            # physics, so only that is refused.  The exporter cleans up
            # its own staging on any failure, so `wrf_output` stays
            # absent rather than half-written.
            wrf_manifest = stock_wrf_export_refused(error)
    timings["direct_stock_wrf_export"] = time.perf_counter() - started
    timings["total"] = sum(timings.values())
    return NativeHierarchyExportResult(
        artifacts=artifact_build,
        wrf_manifest=wrf_manifest,
        timings_seconds=timings)


__all__ = [
    "NativeHierarchyExportResult",
    "STOCK_WRF_EXPORT_MODES",
    "initialize_and_export_native_hierarchy",
]
