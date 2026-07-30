"""Native meteorological-source adapter inventory for WRF initialization.

This registry is deliberately stricter than a decoder capability list.
Runnable means a strict implementation route exists; certification separately
requires that its exact field, vertical-level, cadence, missing-state, and
domain envelope has been accepted by unchanged stock ``wrf.exe``. Merely being
readable by the Rust GRIB decoder does not make a product a scientifically
complete limited-area initial state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Iterable

from gpuwm.source_authorities import twentycrv3_authority_sha256


RUSTY_WEATHER_HEAD = "d3264700c125449a50ac0080be42796177691e1d"
RUSTY_WEATHER_INVENTORY_HASHES = {
    "crates/rustwx-core/src/lib.rs": (
        "cb5a6fbf87f0d0f5cac35f4735bfc0053c7f4dd79985a64afc1b920c7b20a2eb"
    ),
    "crates/rustwx-models/src/lib.rs": (
        "125b919608b1981fa314755d2a5419f4fdb3783b3abd52200806cb0252268d3c"
    ),
    "README.md": (
        "6eb3ea080d78a634fea1fc74e32795fde14af30ad48032efe62bcd90e98db6f3"
    ),
}


class SourceKind(str, Enum):
    """Scientific role of a source product, not merely its file format."""

    DETERMINISTIC_STATE = "deterministic_state"
    ENSEMBLE_MEMBERS = "ensemble_members"
    ENSEMBLE_STATISTIC = "ensemble_statistic"
    SURFACE_ANALYSIS = "surface_analysis"
    POSTPROCESSED_GUIDANCE = "postprocessed_guidance"
    WRF_ARCHIVE = "wrf_archive"


class AdapterStatus(str, Enum):
    """Readiness for direct native production of stock-WRF inputs."""

    CERTIFIED = "certified_stock_wrf"
    NATIVE_EXPORT_PENDING = "native_ingest_direct_export_pending"
    MAPPING_REQUIRED = "adapter_mapping_required"
    MEMBER_SELECTION_REQUIRED = "member_selection_and_mapping_required"
    COMPOSITION_REQUIRED = "explicit_composition_required"
    ARCHIVE_MAPPING_REQUIRED = "wrf_archive_mapping_required"
    DECODE_IMPLEMENTED_CERTIFICATION_REQUIRED = (
        "mapped_decode_implemented_stock_wrf_certification_required"
    )
    RUNNABLE_NOT_CERTIFIED = "runnable_mapping_not_stock_wrf_certified"


@dataclass(frozen=True)
class SourceAdapter:
    """A fail-closed source-to-WRF adapter declaration."""

    source_id: str
    aliases: tuple[str, ...]
    upstream_model_id: str | None
    source_kind: SourceKind
    file_family: str
    decoder: str
    default_product: str
    required_products: tuple[str, ...]
    max_forecast_hour: int
    upstream_ingest: str
    status: AdapterStatus
    field_mapping: str
    level_mapping: str
    cadence_mapping: str
    stock_wrf_gate: str
    runnable: bool = False
    runner: str | None = None
    composition_requirement: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["source_kind"] = self.source_kind.value
        value["status"] = self.status.value
        return value


def _adapter(
    source_id: str,
    *,
    aliases: Iterable[str] = (),
    upstream_model_id: str | None = "__same_as_source__",
    kind: SourceKind = SourceKind.DETERMINISTIC_STATE,
    file_family: str = "GRIB2",
    default_product: str,
    required_products: Iterable[str] = (),
    max_hour: int,
    upstream_ingest: str = "registry_only",
    decoder: str | None = None,
    status: AdapterStatus = AdapterStatus.MAPPING_REQUIRED,
    field_mapping: str = "pending",
    level_mapping: str = "pending",
    cadence_mapping: str = "pending",
    stock_wrf_gate: str = "not_run",
    runnable: bool = False,
    runner: str | None = None,
    composition: str | None = None,
    notes: str = "",
) -> SourceAdapter:
    return SourceAdapter(
        source_id=source_id,
        aliases=tuple(aliases),
        upstream_model_id=(
            source_id if upstream_model_id == "__same_as_source__" else upstream_model_id
        ),
        source_kind=kind,
        file_family=file_family,
        decoder=decoder or (
            "rusty-weather/grib-core"
            if file_family in {"GRIB1", "GRIB2"}
            else "rusty-weather/rustwx-io"
        ),
        default_product=default_product,
        required_products=tuple(required_products) or (default_product,),
        max_forecast_hour=max_hour,
        upstream_ingest=upstream_ingest,
        status=status,
        field_mapping=field_mapping,
        level_mapping=level_mapping,
        cadence_mapping=cadence_mapping,
        stock_wrf_gate=stock_wrf_gate,
        runnable=runnable,
        runner=runner,
        composition_requirement=composition,
        notes=notes,
    )


# The order is canonical: the 23 rusty-weather ModelId values, the ERA5 GRIB1
# source already decoded by gpuwm's native ingest path, the packaged 20CRv3
# member profile, its complementary metadata-driven NetCDF-CF primitives, then
# the declarative format-level adapter exposed by RW-WPS.
_ADAPTERS = (
    _adapter(
        "hrrr",
        default_product="sfc",
        required_products=("sfc", "nat"),
        max_hour=48,
        upstream_ingest="full",
        status=AdapterStatus.CERTIFIED,
        field_mapping="hrrr-native-state-v1",
        level_mapping="native-input-to-explicit-wrf-eta-v2",
        cadence_mapping="consecutive-hourly-f00-f12-v1",
        stock_wrf_gate=(
            "wrf-v4.6.1-pass-full-f00-f12-dynamic-lambert-geometry"
        ),
        runnable=True,
        runner="hrrr_f00_f12_v1",
        notes=(
            "Certified slice: one CONUS Lambert specified domain, WSM6, YSU, "
            "classic MM5 surface layer (option 91), Noah, and an explicit "
            "validated shared eta grid; "
            "the accepted 49-level profile remains the stock-WRF authority "
            "while unlike counts are covered by structural gates; "
            "stock-WRF proofs cover 192x160 at 3 km in Oklahoma and Ohio and "
            "1000x1000 at 1 km in Oklahoma."
        ),
    ),
    _adapter(
        "hrrr-ak", aliases=("hrrrak", "hrrr-alaska"),
        default_product="sfc", required_products=("sfc", "nat"), max_hour=48,
        notes="Alaska grid/projection and field contract are not yet mapped.",
    ),
    _adapter(
        "gfs", aliases=("gfs-0p25", "gfs-0.25"),
        default_product="pgrb2.0p25", max_hour=384,
        upstream_ingest="full",
        status=AdapterStatus.CERTIFIED,
        field_mapping="gfs-pgrb2-raw-code-to-native-state-v1",
        level_mapping="21-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-gfs-forecast-series-v1",
        stock_wrf_gate=(
            "wrf-v4.6.1-pass-gfs-20260720-00z-250x200x49-f000-f003"
        ),
        runnable=True,
        runner="gfs_pgrb2_0p25_v1",
        notes=(
            "Certified slice: one specified Lambert domain, GFS pgrb2.0p25 "
            "with complete 1000..100-hPa state and exact four Noah soil "
            "layers, WSM6, YSU, classic MM5 surface layer (option 91), Noah, "
            "and any explicit eta "
            "grid whose model top is covered by GFS. Numeric GRIB2 "
            "identifiers and both soil "
            "fixed surfaces are validated before decode; unchanged stock "
            "WRF v4.6.1 accepted the f000/f003 proof and completed a finite "
            "model step. The same named-source implementation now accepts "
            "namelist-bound static one-way Lambert hierarchies through at "
            "least d06; that broader envelope is runnable but remains "
            "separately stock-WRF gated."
        ),
    ),
    _adapter(
        "gdas", aliases=("gdas-0p25", "gdas-0.25"),
        default_product="pgrb2.0p25", max_hour=9,
    ),
    _adapter(
        "gefs", aliases=("gefs-ensemble",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        default_product="pgrb2ap5/gec00", max_hour=384,
        status=AdapterStatus.MEMBER_SELECTION_REQUIRED,
        composition="Select an actual GEFS member; ensemble means/spread are not trajectory states.",
    ),
    _adapter(
        "aigfs", aliases=("ai-gfs",),
        default_product="sfc", max_hour=384,
        required_products=("sfc", "pres"),
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Combine pressure-level atmosphere with the surface product and declare soil policy.",
    ),
    _adapter(
        "aigefs", aliases=("ai-gefs",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        default_product="sfc/avg", max_hour=384,
        required_products=("sfc/member", "pres/member"),
        status=AdapterStatus.MEMBER_SELECTION_REQUIRED,
        composition="Select a member and combine pressure/surface products; averages are not member states.",
    ),
    _adapter(
        "hgefs", aliases=("hybrid-gefs",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        default_product="sfc/avg", max_hour=240,
        required_products=("sfc/member", "pres/member"),
        status=AdapterStatus.MEMBER_SELECTION_REQUIRED,
        composition="Select a member and combine pressure/surface products; averages are not member states.",
    ),
    _adapter(
        "ecmwf-open-data", aliases=("ecmwf", "ifs"),
        default_product="oper", max_hour=360,
    ),
    _adapter("aifs", aliases=("aifs-v2",), default_product="oper", max_hour=43848),
    _adapter("rap", default_product="awp130pgrb", max_hour=51),
    _adapter("nam", default_product="awip12", max_hour=84),
    _adapter(
        "hiresw", aliases=("hires",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        default_product="arw_2p5km/conus", max_hour=48,
        status=AdapterStatus.MEMBER_SELECTION_REQUIRED,
        composition="Select the concrete ARW/FV3 member before initialization.",
    ),
    _adapter(
        "href", aliases=("href-conus",),
        kind=SourceKind.ENSEMBLE_STATISTIC,
        default_product="ensprod/conus/sprd", max_hour=48,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Use a constituent deterministic member; spread/probability products cannot initialize WRF.",
    ),
    _adapter(
        "sref", kind=SourceKind.ENSEMBLE_STATISTIC,
        default_product="ensprod/pgrb212/mean_3hrly", max_hour=87,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Use a constituent member; the ensemble mean is not a dynamically balanced member state.",
    ),
    _adapter(
        "rtma", kind=SourceKind.SURFACE_ANALYSIS,
        default_product="2dvaranl_ndfd", max_hour=0,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Provide a complete 3-D atmosphere source; RTMA may replace declared surface fields only.",
    ),
    _adapter(
        "urma", kind=SourceKind.SURFACE_ANALYSIS,
        default_product="2dvaranl_ndfd", max_hour=0,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Provide a complete 3-D atmosphere source; URMA may replace declared surface fields only.",
    ),
    _adapter(
        "nbm", aliases=("blend",),
        kind=SourceKind.POSTPROCESSED_GUIDANCE,
        default_product="core/co", max_hour=264,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Provide a complete 3-D analysis/forecast state; NBM is postprocessed guidance.",
    ),
    _adapter(
        "rrfs-a", aliases=("rrfsa",),
        default_product="prs-conus", max_hour=60,
        upstream_ingest="full",
        notes="rusty-weather ingest is mature; WRF state/soil mapping remains gated.",
    ),
    _adapter("rrfs-public", default_product="prs-conus", max_hour=60),
    _adapter(
        "refs", aliases=("rrfs-ensemble",),
        kind=SourceKind.ENSEMBLE_STATISTIC,
        default_product="mean-conus", max_hour=60,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Use a constituent RRFS member; mean/PMMN/spread products cannot initialize a member.",
    ),
    _adapter(
        "rrfs-firewx", aliases=("firewx",),
        default_product="2dfld-firewx", max_hour=36,
        required_products=("2dfld-firewx", "prs-firewx"),
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Combine the 2-D fire-weather product with a complete pressure/native atmosphere and soil state.",
    ),
    _adapter(
        "wrf", aliases=("wrf-gdex", "wrf-arw"),
        kind=SourceKind.WRF_ARCHIVE,
        file_family="NetCDF",
        default_product="surface",
        max_hour=0,
        status=AdapterStatus.ARCHIVE_MAPPING_REQUIRED,
        composition="Map a compatible WRF archive state, vertical coordinate, physics state, and boundary source.",
    ),
    _adapter(
        "era5",
        upstream_model_id=None,
        file_family="GRIB1",
        default_product="pressure-level+single-level",
        required_products=("pressure-level", "single-level", "invariants"),
        max_hour=0,
        upstream_ingest="gpuwm_native_existing",
        status=AdapterStatus.CERTIFIED,
        field_mapping="era5-grib1-vtable-to-native-state-v1",
        level_mapping="pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-local-grib-time-series-v1",
        stock_wrf_gate=(
            "wrf-v4.6.1-pass-201ffde-250x200x49-12km-12h"
        ),
        runnable=True,
        runner="era5_combined_grib1_v1",
        notes=(
            "Certified slice: one specified Lambert domain, WSM6, YSU, "
            "classic MM5 surface layer (option 91), Noah, an explicit "
            "validated eta coordinate, a "
            "model top covered by the supplied pressure levels, a "
            "uniformly spaced combined GRIB1 series beginning at f00, and "
            "explicit hash-bound WPS geometry/static/orography inputs. "
            "Unchanged stock WRF v4.6.1 accepted the 250x200 at 12 km, "
            "12-hour proof products and completed a finite model step. The "
            "named-source hierarchy route now accepts namelist-bound static "
            "one-way Lambert layouts through at least d06 using invariant "
            "SOILGEO or exact per-domain source orography; that broader "
            "envelope is not yet "
            "stock-WRF certified."
        ),
    ),
    _adapter(
        "20crv3",
        aliases=("20cr", "twentycrv3", "20crv3-member"),
        upstream_model_id=None,
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged exact 20CRv3 member profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="every-member-pressure+surface",
        required_products=("pressure-level", "surface", "one exact member"),
        # 20CRv3 files are analyses at successive valid times, not forecast
        # lead-hour products; the three-hour property belongs to cadence.
        max_hour=0,
        upstream_ingest="filename_member_manifest_v1",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-20crv3-member-grib2-v1",
        level_mapping="23-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-paired-three-hour-member-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="twentycrv3_member_grib2_v1",
        notes=(
            "Runnable exact-member profile with filename-plus-hash-manifest "
            "member identity, packaged mapping/composition/provenance "
            "authorities, paired pressure/surface inputs, and native one-way "
            "Lambert hierarchy export through the mapping's max_dom=4. "
            "The route is not yet accepted by unchanged stock WRF and does "
            "not certify other 20CR products, arbitrary members mixed in one "
            "run, or a larger domain count."
        ),
    ),
    _adapter(
        "20crv3-cf",
        aliases=("20crv3-netcdf", "20cr-netcdf", "20cr-cf"),
        upstream_model_id=None,
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="NetCDF-CF",
        decoder="netCDF4/CF-metadata",
        default_product="analysis-members",
        max_hour=0,
        upstream_ingest="metadata_discovery_and_bounded_member_streaming",
        status=AdapterStatus.MEMBER_SELECTION_REQUIRED,
        field_mapping="20crv3-cf-to-canonical-v1-synthetic-gated",
        level_mapping="metadata-pressure-level-normalization-v1-synthetic-gated",
        cadence_mapping="metadata-uniform-series-v1-synthetic-gated",
        stock_wrf_gate="awaiting-real-cf-corpus-and-stock-wrf-startup",
        notes=(
            "Metadata discovery, arbitrary member counts, hourly/three-hourly "
            "cadence validation, unit/layout normalization, bounded streaming, "
            "and deterministic atomic manifests are synthetic-gated in "
            "gpuwm.twentycrv3. This complementary NetCDF-CF route has no public "
            "WRF runner or real-corpus certification yet and must not inherit "
            "the exact GRIB2 member route's evidence."
        ),
    ),
    _adapter(
        "mapped",
        aliases=("generic-mapped", "mapping-v1"),
        upstream_model_id=None,
        file_family="GRIB1/GRIB2/NetCDF",
        decoder=(
            "rw-wps.mapping.v1 + gpuwm.mapped_source + "
            "vendored rusty-weather GRIB bridges"
        ),
        default_product="mapping-declared-inputs",
        max_hour=0,
        upstream_ingest="declarative_mapping_v1",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="rw-wps.mapping.v1-executable-consumer",
        level_mapping="mapping-declared-source-to-canonical-frame",
        cadence_mapping="mapping-declared-uniform-series",
        stock_wrf_gate="exact-mapping-composition-hash-evidence-required",
        runnable=True,
        runner="mapped_composition_v1",
        notes=(
            "Strict canonical decoding, exact-subset mixed-product composition, "
            "source-frame packing, native initialization, and atomic WRF export "
            "are implemented. Historical v1 ERA5 GRIB1, GFS GRIB2, and ERA5 "
            "NetCDF single-domain gates passed unchanged WRF v4.6.1. The "
            "current canonical-LF GFS GRIB2 contract also passed an unchanged "
            "WRF v4.6.1 d01 through d04 gate after replacing 2 m dewpoint with the "
            "downloaded RH2 contract and binding GRIB2 local-table authority. "
            "The superseded GFS hierarchy evidence remains explicitly "
            "invalidated. New mappings remain "
            "fail-closed to the explicit field, selector, soil, cadence, target, "
            "decoder, manifest, and composition contracts. General explicit "
            "descriptor/Vtable import and exact manifest authoring are available "
            "without adding a named adapter, but their result is validated, not "
            "automatically stock-WRF certified; this is not a claim that arbitrary "
            "undeclared GRIB/NetCDF is scientifically complete."
        ),
    ),
)


def _build_alias_map() -> dict[str, SourceAdapter]:
    aliases: dict[str, SourceAdapter] = {}
    for adapter in _ADAPTERS:
        for raw_name in (adapter.source_id, *adapter.aliases):
            name = raw_name.strip().lower().replace("_", "-")
            if name in aliases:
                raise RuntimeError(f"duplicate source-adapter alias: {name}")
            aliases[name] = adapter
    return aliases


_ALIASES = _build_alias_map()


def source_adapters() -> tuple[SourceAdapter, ...]:
    """Return the immutable canonical source inventory."""

    return _ADAPTERS


def get_source_adapter(source: str) -> SourceAdapter:
    """Resolve a source id/alias or raise a useful fail-closed error."""

    normalized = source.strip().lower().replace("_", "-")
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        available = ", ".join(adapter.source_id for adapter in _ADAPTERS)
        raise ValueError(
            f"unknown native source {source!r}; available sources: {available}"
        ) from exc


def _current_config_sha256(name: str, canonical_sha256: str) -> str:
    """Bind the checked-out/package config bytes when they are available.

    Git may materialize JSON authorities with LF or CRLF line endings.  The
    evidence manifest must report the bytes the runtime will actually consume,
    rather than a hash copied from a different checkout platform.
    """
    path = Path(__file__).resolve().parents[1] / "configs" / name
    if not path.is_file():
        return canonical_sha256
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_capability_manifest() -> dict[str, object]:
    """Return the machine-readable, provenance-bound capability inventory."""

    return {
        "schema": "gpuwm-native-source-adapters-v1",
        "readiness_rule": (
            "runnable means the strict implementation route exists; stock-WRF "
            "certification is a separate exact evidence status"
        ),
        "certification_rule": (
            "certified_stock_wrf requires implemented field/level/cadence "
            "policies plus acceptance by unchanged stock wrf.exe within the "
            "declared source/domain envelope"
        ),
        "runtime_forbidden": ["WPS", "real.exe"],
        "rusty_weather_inventory": {
            "head": RUSTY_WEATHER_HEAD,
            "worktree_was_dirty_during_inventory": True,
            "bound_file_sha256": dict(RUSTY_WEATHER_INVENTORY_HASHES),
            "model_id_count": 23,
        },
        "source_count": len(_ADAPTERS),
        "runnable_source_count": sum(adapter.runnable for adapter in _ADAPTERS),
        "packaged_source_authorities": {
            "20crv3_member_grib2": dict(twentycrv3_authority_sha256()),
        },
        "mapped_stock_wrf_evidence": [
            {
                "mapping_sha256": (
                    "2ceb90e63d3265c0ebf871de4bfc7f622284681a9861c17e78e9fc11e3dc15b2"
                ),
                "composition_sha256": (
                    "9aa8c7aec96a1d52e83f61c470cdb2a636ca092af902581bafb05fc433ab57f3"
                ),
                "gate": "wrf-v4.6.1-pass-era5-grib1-single-domain",
            },
            {
                "mapping_sha256": (
                    "f278705331a81767d4d3532ff4dd4f739242a79b224747f51e722d462142daa8"
                ),
                "composition_sha256": (
                    "10b23b56b882ed534dbcb5219ac06f9dcc184226e7cc2aea3c6c03f3ad6be459"
                ),
                "gate": "wrf-v4.6.1-pass-era5-netcdf-single-domain",
            },
            {
                "mapping_sha256": (
                    "5b0f41a7f4ddee1116ce8310dfd67827761413908d45402e1f55f32facc61d86"
                ),
                "composition_sha256": (
                    "e0c2adae105263b177d7e8f8bb87d0e99731bc4cda9cb6a4217971a0b49b18e1"
                ),
                "gate": (
                    "wrf-v4.6.1-clean-d66e442-pass-gfs-grib2-"
                    "canonical-lf-current-contract-d01-d04-z49"
                ),
                "source_format": "grib2",
                "domain_envelope": (
                    "specified Lambert d01 120x100x49 at 12 km plus three "
                    "one-way sibling nests d02-d04, each 60x60x49 at 4 km"
                ),
                "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
                "wrf_exe_sha256": (
                    "cfac96554c8f9796c7522aaf023131ea7681ddf12110a327e51a548958874089"
                ),
                "native_proof_sha256": (
                    "cb46c3539e6789710fa44717fe9af6871483ca035e1c41abfb3eb2fe4b804d28"
                ),
                "native_proof_content_sha256": (
                    "fddddaec90934ce2d37fd112a36b76fda623eba945df942e04b6f9053de77138"
                ),
                "native_export_manifest_sha256": (
                    "5330ad1aa50bdeb0a3de4c601c29660038a111d5b966b305c3f694e35d07d1ad"
                ),
                "native_input_manifest_sha256": (
                    "3f601dfa28419229ef5314a58345ca4b521642b796c1d49a7494e1d94f330abb"
                ),
                "wrfinput_d01_sha256": (
                    "1f5cf5d47e0a450e44bbf7033f0622b0636c852208fe76390e5719822f0014dc"
                ),
                "wrfinput_d02_sha256": (
                    "546255b0ad323b073f0f0d87545615ea07937f50393f12305d078d9128d262da"
                ),
                "wrfinput_d03_sha256": (
                    "7044d67ac77c9081e8bfbfd6692bfc18aa26fb1cef001423b729082ea5ff5003"
                ),
                "wrfinput_d04_sha256": (
                    "ce8ffdd60acfd41a316e3c7df74fe4b48063afa1b1c2138ab510d524fa8fc082"
                ),
                "wrfbdy_d01_sha256": (
                    "9f9d2430511fd807c4c0a75c1fb29e1d6f401893929d858a1002920a385e79fa"
                ),
                "stock_wrf_input_receipt_sha256": (
                    "25a643fe34ff1ddd39464129bdcfff074e938ced216608f8cab8d17196ea524c"
                ),
                "stock_wrf_stdout_sha256": (
                    "df9777532d6baec919937d684eb16bad1d22831239092effefa007d6a7eeeb8d"
                ),
                "stock_wrf_time_sha256": (
                    "46f06763d0f59d4b4b91fd84ac27dd98264f8d6f4a01f1a84a94b55544f01392"
                ),
                "stock_wrf_stderr_sha256": (
                    "bee69544f5cdbbfecfff392294a175311e7f19d505e3e8227aebee77727c43eb"
                ),
                "stock_wrf_namelist_sha256": (
                    "98693a363961c47afb37e52feb9bb9eb9125069802a89b5fd5aa30acedc22cfc"
                ),
                "wrfout_d01_sha256": (
                    "ad34744719f6bf120bffe1771fab28891b1968d3c324f149676ba215014f90f4"
                ),
                "wrfout_d02_sha256": (
                    "5f7518adc40c869249b70cf02a547c39a84bd86b4d8af9614659224c3df19241"
                ),
                "wrfout_d03_sha256": (
                    "9aa978f4088597abe90c8c466e295251b962af077342241134b24397971b6f23"
                ),
                "wrfout_d04_sha256": (
                    "91171bd4b0e603df531e10b06ca7613a824e26a88534010962f167f7bac13cbb"
                ),
                "native_total_seconds": 10.613378500012914,
                "stock_wrf_wall_seconds": 10.08,
                "stock_wrf_max_rss_kib": 1076808,
                "stock_wrf_result": (
                    "exit_0_success_complete_wrf_d02_d03_d04_six_steps_each"
                ),
                "stock_wrf_health": "no_nan_cfl_fatal_or_segfault_markers",
            },
        ],
        "invalidated_mapped_stock_wrf_evidence": [
            {
                "mapping_sha256": (
                    "726677d8c2365e6f533cc6dd5d7c795e198164326660c3630d885c83f406a11e"
                ),
                "composition_sha256": (
                    "266c98099b24f03a3bc986f275b44bbd6bf20dce1006ad05f6da39bc4a373bfb"
                ),
                "gate": "wrf-v4.6.1-pass-gfs-grib2-d01-d02",
                "reason": (
                    "superseded by trajectory-relevant RH2-to-q2 mapping and "
                    "explicit GRIB2 local-table authority; evidence does not "
                    "transfer across contract hashes"
                ),
                "replacement_mapping_sha256": (
                    _current_config_sha256(
                        "rw-wps-gfs-pressure-grib2.mapping.json",
                        "5b0f41a7f4ddee1116ce8310dfd67827761413908d45402e1f55f32facc61d86",
                    )
                ),
                "replacement_composition_sha256": (
                    _current_config_sha256(
                        "rw-wps-gfs-terrain.composition.json",
                        "e0c2adae105263b177d7e8f8bb87d0e99731bc4cda9cb6a4217971a0b49b18e1",
                    )
                ),
                "replacement_status": "stock_wrf_certified_d01_d04_z49",
                "replacement_gate": (
                    "wrf-v4.6.1-clean-d66e442-pass-gfs-grib2-"
                    "canonical-lf-current-contract-d01-d04-z49"
                ),
            },
        ],
        "sources": [adapter.to_dict() for adapter in _ADAPTERS],
    }
