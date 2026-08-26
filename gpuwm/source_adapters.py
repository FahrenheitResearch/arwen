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
from types import MappingProxyType
from typing import Iterable, Mapping

from gpuwm.source_authorities import (packaged_authority_sha256,
                                      packaged_member_grammar_ids,
                                      packaged_member_grammar_sha256,
                                      packaged_profile_ids)
from gpuwm.source_coverage import (COVERAGE_WINDOW_TYPES, CoverageWindow,
                                   LambertGridWindow, RegularLatLonWindow)
from gpuwm.source_cycles import CycleGrid
from gpuwm.source_credentials import (CredentialLocation, SourceCredential,
                                      credential_declaration)


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
    #: The packaged profile (mapping + composition + provenance, shipped in
    #: the wheel and pinned by SHA-256 in :mod:`gpuwm.source_authorities`)
    #: this source is decoded through.  A row that names one needs no
    #: per-source runner function: the front door reads the profile.  This
    #: is the arbitrary-acceptance seam -- adding a model whose mapping can
    #: be written is three JSON documents plus this row.
    packaged_profile: str | None = None
    composition_requirement: str | None = None
    #: The packaged ``rw-wps.members.v1`` document (shipped in the wheel
    #: and pinned by SHA-256 in :mod:`gpuwm.source_authorities`) this
    #: ensemble's member addressing/verification runs through.  The same
    #: seam as ``packaged_profile``, for the member axis: a row that
    #: names one gets `gpuwm-member-prep` member selection with no
    #: per-source code -- the declared member set, filename patterns,
    #: verification triple and statistic guard are all table data.
    member_set: str | None = None
    #: The native cadence of this source's model state, in seconds -- the
    #: spacing between the valid times a preparation gets lateral boundary
    #: conditions from, and therefore ``&share/interval_seconds`` in an
    #: emitted ``namelist.wps``.  THE reason `gpuwm domain` could not emit a
    #: runnable config for these sources: the wizard held a three-entry dict
    #: and every other model's battery TOML was hand-assembled around it.
    #:
    #: For a row that names a ``packaged_profile`` this repeats the profile
    #: mapping's ``target.boundary_interval_seconds``, the same way
    #: :mod:`gpuwm.source_authorities` repeats the composition's roles: the
    #: wizard has to know the cadence before anything opens a JSON
    #: authority, and a repeat guarded by a test is not a second definition.
    #: :func:`tests.test_source_wizard_facts` fails the moment the two
    #: disagree.  ``None`` means the cadence is not a property of the source
    #: (the generic ``mapped`` route reads it from the caller's own mapping),
    #: and the wizard refuses to plan such a source by name.
    forcing_interval_seconds: float | None = None
    #: The smallest pressure (Pa) this source's CERTIFIED inventory serves
    #: -- the top of the ladder its route decodes, a published fact of the
    #: source like its cadence.  ``None`` means the column reaches at
    #: least the wizard's default model top
    #: (:data:`gpuwm.domain_wizard.DEFAULT_MODEL_TOP_PA`, 5000 Pa) and the
    #: default is emitted unchanged; a declared value floors the emitted
    #: ``p_top`` so a bare ``gpuwm domain --source X`` config never asks
    #: for a model top its own source cannot cover and refuses at
    #: preparation after the acquisition was already paid for.  Gated by
    #: tests/test_ptop_default.py against the packaged mapping's own
    #: level table.
    certified_source_top_pa: float | None = None
    #: Where this source's native grid reaches, or ``None`` for a global
    #: product.  A declared window (see :mod:`gpuwm.source_coverage`) is what
    #: lets `gpuwm domain` refuse an out-of-coverage plan AT PLAN TIME with
    #: the offending corner named, instead of letting the user pay for the
    #: whole acquisition and read the refusal out of a preparation
    #: traceback -- measured on ICON-EU over a central-US domain,
    #: 2026-08-17.  Rows with no route to run (``runnable`` false) declare
    #: nothing: a window nobody measured would be a number invented to fill
    #: a field.
    coverage_window: CoverageWindow | None = None
    #: WHEN this source initializes, and when its bytes land -- the
    #: declaration ``--cycle latest`` resolves against (see
    #: :mod:`gpuwm.source_cycles`).  The resolver used to branch on
    #: gfs/gdas/hrrr by name and refuse every other source with a
    #: sentence about ERA5's latency, so `--cycle latest --source rap`
    #: was told about a reanalysis it had not asked for and a reanalysis
    #: with a KNOWN delay was told it had no latest at all.
    #:
    #: ``None`` is the common case and not a gap: a source whose fetch
    #: route declares ``cycle_hours`` has its grid DERIVED from that
    #: measured table, which is what makes a new model's ``latest`` a
    #: route row rather than a code change.  This column carries the
    #: schedules no route table can state -- the legacy transports, and
    #: a keyed job API like the CDS, which publishes no object to probe
    #: and so has to write its publication delay down.
    cycle_grid: CycleGrid | None = None
    #: What a PERSON calls this source.  A column, because every consumer
    #: that wanted a human name had to keep its own id-to-name lookup --
    #: a per-model table, and adding a model then meant editing a front
    #: end as well as this file.  ``None`` reads back as the id through
    #: :attr:`display_title`, so a row that omits it still renders.
    display_name: str | None = None
    #: What must be CONFIGURED before this source's bytes can be
    #: acquired, as declared :class:`~gpuwm.source_credentials.
    #: SourceCredential` rows.  Empty means the row declares no
    #: prerequisite -- which is a statement about this table, not a
    #: promise that a provider asks for nothing.  The same seam as every
    #: other column: a source that needs an account key is one row, and
    #: no front end carries an exception for it.
    credentials: tuple[SourceCredential, ...] = ()
    notes: str = ""

    @property
    def display_title(self) -> str:
        """The declared display name, or the id when none is declared."""

        return (self.display_name or "").strip() or self.source_id

    def __post_init__(self) -> None:
        for credential in self.credentials:
            if not isinstance(credential, SourceCredential):
                raise TypeError(
                    f"{self.source_id}: credentials must be declared "
                    "SourceCredential rows, so a consumer can read where "
                    "the credential lives instead of parsing prose")
        if self.coverage_window is not None and not isinstance(
                self.coverage_window, COVERAGE_WINDOW_TYPES):
            raise TypeError(
                f"{self.source_id}: coverage_window must be one of "
                + ", ".join(t.__name__ for t in COVERAGE_WINDOW_TYPES))
        if (self.forcing_interval_seconds is not None
                and not self.forcing_interval_seconds > 0.0):
            raise ValueError(
                f"{self.source_id}: forcing_interval_seconds must be a "
                "positive number of seconds between source valid times")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["source_kind"] = self.source_kind.value
        value["status"] = self.status.value
        # The name a consumer shows, resolved: a row that declares none
        # still answers, and nobody has to reimplement the fallback.
        value["display_name"] = self.display_title
        # DECLARATION only.  The manifest is provenance -- two boxes
        # compare theirs -- so it must not carry a verdict about which
        # keys happen to sit on the box that generated it.
        value["credentials"] = [credential_declaration(credential)
                                for credential in self.credentials]
        value["coverage_envelope"] = (
            None if self.coverage_window is None
            else list(self.coverage_window.envelope()))
        # asdict() recurses into the grid and leaves a datetime in
        # record_end, which no JSON writer takes.  DECLARED form only:
        # what the row says, never what `latest` resolves to today.
        value["cycle_grid"] = (None if self.cycle_grid is None
                               else self.cycle_grid.declaration())
        return value


def _adapter(
    source_id: str,
    *,
    name: str | None = None,
    credentials: Iterable[SourceCredential] = (),
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
    packaged_profile: str | None = None,
    composition: str | None = None,
    member_set: str | None = None,
    forcing_interval_seconds: float | None = None,
    certified_source_top_pa: float | None = None,
    coverage: CoverageWindow | None = None,
    cycles: CycleGrid | None = None,
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
        packaged_profile=packaged_profile,
        composition_requirement=composition,
        member_set=member_set,
        forcing_interval_seconds=forcing_interval_seconds,
        certified_source_top_pa=certified_source_top_pa,
        coverage_window=coverage,
        cycle_grid=cycles,
        display_name=name,
        credentials=tuple(credentials),
        notes=notes,
    )


#: The 3 km CONUS Lambert, as the grid-definition section of both HRRR's
#: and RRFS's bytes declares it.  ONE constant for both rows because the
#: two grids were measured byte-identical in every geolocating octet
#: (gauntlet staging, rrfs/DIFF-MEASURED-rrfs-vs-hrrr.txt): RRFS is HRRR's
#: operational successor on HRRR's grid, and writing the numbers twice
#: would invite them to drift.  Reproduces
#: :func:`gpuwm.ingest.hrrr_target.hrrr_coverage_envelope` exactly, which
#: is gated in tests/test_source_wizard_facts.py.
_CONUS_3KM_LAMBERT = LambertGridWindow(
    nx=1799, ny=1059,
    lat1=21.138123, lon1=-122.719528,
    dx_m=3000.0, truelat1=38.5, truelat2=38.5, stand_lon=-97.5)

#: AWIPS grid 221, RAP's awip32 product: 32 km Lambert over North America.
#: Measured from the real f00 bytes (gauntlet staging, rap/facts.md:
#: GDT 30, nx=349 ny=277, lat1=1.0 lon1=214.5, dx=dy=32463 m,
#: latin1=latin2=50.0, lov=253.0).
_NORTH_AMERICA_32KM_LAMBERT = LambertGridWindow(
    nx=349, ny=277,
    lat1=1.0, lon1=-145.5,
    dx_m=32463.0, truelat1=50.0, truelat2=50.0, stand_lon=-107.0)

#: DWD's ICON-EU regular lat-lon window, exactly as DWD publishes it and as
#: the preparation stage's own out-of-grid refusal reports it: 1377 x 657
#: points at 0.0625 degrees, lon -23.5..62.5, lat 29.5..70.5.  This window
#: is why the wizard can now refuse a central-US ICON-EU plan in one
#: sentence instead of after the whole acquisition.
_ICON_EU_WINDOW = RegularLatLonWindow(
    south=29.5, west=-23.5, north=70.5, east=62.5, nx=1377, ny=657)


#: The personal Copernicus CDS API key, declared as the row fact it is.
#:
#: The leaf filename repeats :data:`gpuwm.fetch.CDSAPIRC_NAME` rather
#: than importing it, because :mod:`gpuwm.fetch` reaches this registry
#: and the acquisition layer must not become an import prerequisite of
#: the table that describes it.  The same rule the cadence columns
#: follow: a repeat guarded by a test is not a second definition, and
#: ``tests/test_source_registry_columns.py`` fails the moment the
#: declared location stops naming the file the fetch route stats.
_COPERNICUS_CDS_KEY = SourceCredential(
    credential_id="copernicus-cds-api-key",
    display_name="Copernicus CDS key",
    location_kind=CredentialLocation.HOME_FILE,
    location=".cdsapirc",
    needed_for="acquisition",
    breakage=(
        "the retrieval cannot be requested at all -- this source is not "
        "downloaded by gpuwm but by the provider's own client, which "
        "reads a personal account key from that file and fails several "
        "commands later with its own exception if it is absent"
    ),
    obtain_url="https://cds.climate.copernicus.eu",
)


# The order is canonical: the 23 rusty-weather ModelId values, the ERA5 GRIB1
# source already decoded by gpuwm's native ingest path, the packaged 20CRv3
# member profile, its complementary metadata-driven NetCDF-CF primitives, then
# the declarative format-level adapter exposed by RW-WPS.
_ADAPTERS = (
    _adapter(
        "hrrr",
        name="HRRR (native hybrid levels)",
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
        forcing_interval_seconds=3600.0,
        # Hourly, and the walk-back is short on purpose: the
        # operational directories turn over quickly, and a cycle half
        # a day old is not an initialization anyone wants for a
        # convection-permitting run.
        cycles=CycleGrid(
            hours=tuple(range(24)), search_hours=12,
            # The synoptic cycles run out to f048 and the rest stop at
            # f018, so a window longer than 18 h cannot be served by an
            # off-synoptic cycle at all.  Same shape the fetch route
            # table states a horizon in; held in step with
            # gpuwm.hrrr_forecast's constants by a test.
            horizons=(((0, 6, 12, 18), 48), (None, 18)),
            basis="HRRR initializes every hour; publication is "
                  "decided by the per-object completeness probe, "
                  "not by a declared delay"),
        coverage=_CONUS_3KM_LAMBERT,
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
        "hrrr-prs",
        name="HRRR (pressure levels)",
        aliases=("hrrr-pressure", "hrrr-wrfprs"),
        upstream_model_id="hrrr",
        file_family="GRIB2",
        decoder=(
            "packaged HRRR pressure-level profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="wrfprs",
        required_products=("wrfprs",),
        max_hour=48,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-hrrr-prs-grib2-v1",
        level_mapping="39-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="hrrr-prs-grib2-v1",
        forcing_interval_seconds=3600.0,
        coverage=_CONUS_3KM_LAMBERT,
        notes=(
            "HRRR's public wrfprs product through the GENERIC mapped route: "
            "one wrfprs file per valid time carries the 39-level pressure "
            "state, the surface/2m/10m fields, in-band terrain and the "
            "nine-node RUC soil column.  The Lambert CONUS grid, the "
            "grid-relative wind rotation and the node soil geometry are "
            "declared in the packaged mapping/composition documents -- "
            "there is no HRRR decode code on this route.  Distinct from "
            "--source hrrr, the certified native hybrid-level route: this "
            "profile initialises from the pressure-level analysis, which "
            "is smoother near terrain than native levels, and it is not "
            "yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "gem-gdps",
        name="GEM GDPS (Canadian global)",
        # `gem` is an ALIAS, not only the upstream model id.  The 2026-08-17
        # model battery measured `--source gem` refusing by name on every
        # door while every document called the model GEM, so the one word a
        # reader would type was the one word nothing accepted.  Table work,
        # not a code path.
        aliases=("gem", "gdps", "gem-global"),
        upstream_model_id="gem",
        file_family="GRIB2",
        decoder=(
            "packaged GDPS pressure-level profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="15km-latlon0.15",
        required_products=("15km-latlon0.15",),
        max_hour=240,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gem-gdps-grib2-v1",
        level_mapping="33-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-three-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gem-gdps-grib2-v1",
        forcing_interval_seconds=10800.0,
        notes=(
            "ECCC's GDPS (GEM global) through the GENERIC mapped route: "
            "MSC Datamart's 15 km regular lat-lon GRIB2 product, one "
            "single-message JPEG2000-packed file per variable-level, "
            "concatenable per valid time.  33 pressure levels "
            "(1015..1 hPa), surface/2 m/10 m state, in-band terrain and "
            "the single 0-10 cm ISBA soil layer are declared in the "
            "packaged mapping/composition documents -- there is no GDPS "
            "decode code on this route.  GDPS publishes orography, land "
            "mask and the ice analysis at analysis time only; the "
            "profile declares them once-per-cycle invariants and the "
            "generic engine broadcasts the proven analysis record.  "
            "Soil moisture is published as volumetric content, so the "
            "anticipated layer-mass derivation was not needed.  Model "
            "state is three-hourly; not yet accepted by unchanged stock "
            "WRF."
        ),
    ),
    _adapter(
        "icon-eu",
        name="ICON-EU (DWD regional)",
        aliases=("dwd-icon-eu", "icon-eu-regular"),
        upstream_model_id="icon-eu",
        file_family="GRIB2",
        decoder=(
            "packaged ICON-EU regular-lat-lon profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="regular-lat-lon-pressure-level",
        required_products=(
            "pressure-level", "single-level", "soil-level", "time-invariant",
        ),
        max_hour=120,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-icon-eu-regular-grib2-v1",
        level_mapping="20-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="icon-eu-regular-grib2-v1",
        forcing_interval_seconds=3600.0,
        coverage=_ICON_EU_WINDOW,
        notes=(
            "DWD open data's ICON-EU regular-lat-lon product set through "
            "the GENERIC mapped route: one bz2-wrapped GRIB2 object per "
            "field per lead (the acquisition codec stages them "
            "transparently), the 20-level pressure ladder, once-per-cycle "
            "invariant FR_LAND and HSURF broadcast to every valid time, "
            "and the TERRA soil column -- temperature at nine depth "
            "nodes, soil water as column-integrated kg m-2 over the "
            "eight layers whose exact midpoints are those interior "
            "nodes, converted and surface-extended through the closed "
            "derivation catalog.  Everything model-shaped is rows in the "
            "packaged mapping/composition documents; there is no ICON "
            "decode code on this route.  The native icosahedral product "
            "set (GDT 101) is refused by the decoder with the grid "
            "family named.  Not yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "hrrr-ak",
        name="HRRR Alaska", aliases=("hrrrak", "hrrr-alaska"),
        default_product="sfc", required_products=("sfc", "nat"), max_hour=48,
        notes="Alaska grid/projection and field contract are not yet mapped.",
    ),
    _adapter(
        "gfs",
        name="GFS (global, 0.25 degree)", aliases=("gfs-0p25", "gfs-0.25"),
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
        forcing_interval_seconds=10800.0,
        # The certified 21-level pgrb2 ladder stops at 100 hPa; deeper
        # tops exist upstream but fetching them is the explicit
        # `gpuwm fetch --p-top-pa` act, so a bare emission stays here.
        certified_source_top_pa=10000.0,
        # No delay is declared because the completeness probe decides
        # publication object by object; a delay could only start the
        # walk after a cycle the probe would have accepted.
        cycles=CycleGrid(
            hours=(0, 6, 12, 18), search_hours=48,
            basis="NCEP runs the global system at 00/06/12/18 UTC; "
                  "publication is decided by the per-object "
                  "completeness probe"),
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
        "gdas",
        name="GDAS analysis (0.25 degree)", aliases=("gdas-0p25", "gdas-0.25"),
        file_family="GRIB2",
        decoder=(
            "packaged GDAS pgrb2 0.25-degree profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="pgrb2.0p25", max_hour=9,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gdas-pgrb2-0p25-grib2-v1",
        level_mapping="33-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gdas-pgrb2-0p25-grib2-v1",
        forcing_interval_seconds=3600.0,
        cycles=CycleGrid(
            hours=(0, 6, 12, 18), search_hours=48,
            basis="the analysis cycle runs on the global system's "
                  "00/06/12/18 UTC grid; publication is decided by "
                  "the per-object completeness probe"),
        notes=(
            "NCEP's analysis cycle through the GENERIC mapped route as a "
            "packaged profile: one pgrb2.0p25 file per hourly valid time "
            "(f000..f009 -- f010 is a measured 404), each carrying the "
            "33-level isobaric state with direct specific humidity, the "
            "surface/2m/10m fields, in-band terrain and the four Noah "
            "soil layers bound by their scaled type-106 depth pairs "
            "(the integer level key collides between the 0-0.1 m and "
            "0.1-0.4 m layers, so depth-pair rows are the identity).  "
            "Soil moisture is an NCEP local-table row selected by "
            "octets; no table is asked to name it.  The catalogue is "
            "record-for-record byte-identical to GFS pgrb2.0p25 "
            "(696/696 MEASURED), so this profile is the GFS shape at a "
            "different acquisition path with analysis-cycle semantics: "
            "the bytes stamp even f000 a FORECAST (generating process "
            "81 at hour 0, 96 after), and the one analysis-stamped "
            "product (pgrb2.1p00.anl) lacks all soil, the land mask and "
            "every 2 m/10 m field -- it is not an initialization route "
            "and this row does not claim it.  Publication lags the "
            "cycle by ~7 h (delayed-cutoff cycle), far later than GFS.  "
            "Not yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "gefs",
        name="GEFS (global ensemble member)", aliases=("gefs-ensemble",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged GEFS ensemble-member profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="pgrb2a+pgrb2b/member",
        required_products=("pgrb2a", "pgrb2b"),
        max_hour=384,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gefs-ensemble-grib2-v1",
        level_mapping="31-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-three-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gefs-ensemble-grib2-v1",
        forcing_interval_seconds=10800.0,
        composition="Initialize from one verified member's pgrb2a+pgrb2b pair; ensemble means/spread are not trajectory states.",
        member_set="gefs-ensemble-grib2-members-v1",
        notes=(
            "One MEMBER of NCEP's global ensemble through the GENERIC "
            "mapped route.  The member axis is table work: the packaged "
            "member set declares all 31 forecasts (control + 30 "
            "perturbed), the filename patterns, the byte verification "
            "triple, the encoded-size convention (30 EXCLUDES the "
            "control) and the geavg/gespr statistic namespace, and "
            "`gpuwm-member-prep` stages a verified member with a "
            "hash-pinned receipt.  The field axis is table work too: "
            "the packaged profile's 31-level ladder is the measured "
            "exact union of the pgrb2a and pgrb2b isobaric sets (zero "
            "overlap on every variable), so a preparation needs BOTH "
            "files of the SAME member per valid time -- pass each pair "
            "with --input and the SAME pair again as --supplement: "
            "terrain is in band but MIGRATES products (measured: the "
            "analysis publishes orography in pgrb2a, every forecast "
            "step publishes it in pgrb2b), so only the pair carries it "
            "at every valid time.  Every selector "
            "pins PDT 1, so a statistic file (PDT 2) or an accumulation "
            "twin (PDT 11) can never satisfy a state selector, and the "
            "frame assembler's one-GRIB-member rule refuses a "
            "cross-member mix.  Noah soil rides the pair under the "
            "measured 66-percent ocean bitmap (layer 1 in pgrb2a, "
            "layers 2-4 in pgrb2b).  Not yet accepted by unchanged "
            "stock WRF."
        ),
    ),
    _adapter(
        "aigfs",
        name="AIGFS (NOAA AI global forecast)", aliases=("ai-gfs",),
        decoder=(
            "packaged AIGFS operational profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="pres", max_hour=384,
        required_products=("pres", "sfc"),
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-aigfs-gdas-hybrid-grib2-v1",
        level_mapping="13-pressure-level-atmosphere-only-v1",
        cadence_mapping="uniform-six-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="aigfs-gdas-hybrid-grib2-v1",
        forcing_interval_seconds=21600.0,
        composition=(
            "AIGFS publishes no soil temperature or moisture, no land-sea "
            "mask, no orography, no skin temperature, no surface pressure "
            "(PRMSL only) and no 2 m humidity; the hybrid profile borrows "
            "exactly that state from the SAME CYCLE's GDAS 0.25-degree "
            "analysis (the caller's one supplement) through the "
            "cross-source composition, and a donor from any other cycle "
            "or grid refuses by name."
        ),
        notes=(
            "NCEP's GraphCast-based 0.25-degree AI forecast as pure table "
            "data, RUNNABLE as a HYBRID: the operational atmosphere (six "
            "3-D fields on 13 pressure levels topping at 50 hPa, plus 2 m "
            "temperature and 10 m wind) rides the generic mapped route "
            "while the seven canonicals the product does not publish are "
            "composition_bound to the same-cycle GDAS analysis donor, "
            "decoded through the donor's own SHA-256-pinned mapping under "
            "the source_cycle_analysis_broadcast clock (one analysis "
            "record, carried to every lead, carried times named in the "
            "receipt).  The atmosphere-only profile "
            "aigfs-nomads-grib2-v1 remains shipped as the solo-refusal "
            "record.  ACQUISITION IDENTITY "
            "IS PART OF THE PRODUCT: operational bytes are NOMADS-only "
            "(SCN 25-89; 10-day rolling window, no filter CGI, no "
            "operational S3 mirror), stamped subCentre 0, while the "
            "noaa-nws-graphcastgfs-pds bucket serves a DIFFERENT "
            "experimental run under identical filenames stamped "
            "subCentre 2 -- every profile selector pins subcenter=0, so "
            "the imposter refuses by named identity octet instead of "
            "silently mixing model versions.  Deterministic (PDT 0/8 "
            "only); the PDT-8 precipitation windows are deliberately "
            "unselected (duplicate byte-identical tp at f006, shared "
            "paramId/level/step from f012).  Not flagged AI in-band: "
            "genProcId 137 and the front door are the identity."
        ),
    ),
    _adapter(
        "aigefs",
        name="AIGEFS (NOAA AI global ensemble member)", aliases=("ai-gefs",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged AI-ensemble member-hybrid profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="pres/member", max_hour=384,
        required_products=("sfc/member", "pres/member"),
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-aigefs-member-hybrid-grib2-v1",
        level_mapping="13-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-six-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="aigefs-member-hybrid-grib2-v1",
        forcing_interval_seconds=21600.0,
        composition="Select a member and combine pressure/surface products; averages are not member states.",
        member_set="aigefs-ensemble-grib2-members-v1",
        notes=(
            "NCEP's operational AI global ensemble (Project EAGLE), 31 "
            "members, through the GENERIC mapped route as a CROSS-SOURCE "
            "packaged profile.  The member product is a 13-level "
            "0.25-degree dry-dynamical + humidity atmosphere and NOTHING "
            "else -- no soil, no land mask, no orography, no skin "
            "temperature, no 2 m humidity -- so the packaged mapping "
            "declares those six gaps composition_bound and the packaged "
            "composition binds them to the shipped physical-analysis "
            "donor mapping under source_cycle_analysis_broadcast: the "
            "borrowed land state is the SAME cycle's analysis record, "
            "carried to every lead and named in the receipt.  Pass "
            "--input with ONE member's pres+sfc files (stage them "
            "through `gpuwm-member-prep`, whose receipt carries the "
            "member identity the byte-identical leaf filenames cannot) "
            "and --supplement with the same cycle's 0.25-degree "
            "analysis file.  Every packaged selector pins PDT 1, the "
            "individual-member template, so deterministic bytes and "
            "ensemble mean/spread files refuse at the mapped decode; "
            "the packaged member set separately declares the "
            "verification contract (typeOfEnsembleForecast 3 on EVERY "
            "member including the control -- perturbationNumber 0 is "
            "the only control discriminator -- encoded ensemble size "
            "31, octet INCLUDES the control) and the NOMADS-only "
            "ensstat statistic namespace.  The mapped route itself "
            "cannot know WHICH member a caller intended: that identity "
            "lives in the member-prep receipt, not the leaf name.  "
            "Runnable, not stock-WRF certified, and the receipt says so."
        ),
    ),
    _adapter(
        "hgefs",
        name="Hybrid GEFS", aliases=("hybrid-gefs",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        default_product="sfc/avg", max_hour=240,
        required_products=("sfc/member", "pres/member"),
        status=AdapterStatus.MEMBER_SELECTION_REQUIRED,
        composition="Select a member and combine pressure/surface products; averages are not member states.",
    ),
    _adapter(
        "ecmwf-open-data",
        name="ECMWF IFS (open data, 0.25 degree)", aliases=("ecmwf", "ifs"),
        file_family="GRIB2",
        decoder=(
            "packaged ECMWF open-data oper profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="oper", max_hour=360,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-ecmwf-open-data-oper-grib2-v1",
        level_mapping="14-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-three-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="ecmwf-open-data-oper-grib2-v1",
        forcing_interval_seconds=10800.0,
        notes=(
            "ECMWF's open-data IFS oper product through the GENERIC mapped "
            "route: one 0.25-degree global GDT-0 GRIB2 file per three-hourly "
            "step carries the 14-level pressure state, the surface/2 m/10 m "
            "fields, in-band surface geopotential for terrain (converted to "
            "metres by the declared unit scale) and the four IFS soil "
            "layers, which the source addresses by ordinal on fixed-surface "
            "type 151 -- declared through the composition's indexed "
            "selector_depth_binding, not code.  2 m humidity derives from "
            "the published dewpoint.  Winds are earth-relative; CCSDS "
            "(DRT 42) packing decodes through the converged bridge.  "
            "LICENSING: this profile is authored against the 0.25-degree "
            "open-data distribution (CC-BY-4.0); ECMWF's native 9 km HRES "
            "is access-restricted -- a data-licensing fact, not a "
            "capability gap.  Snow and sea-ice fields remain "
            "policy-controlled (local-table/bitmap semantics not yet "
            "bound), and the route is not yet accepted by unchanged stock "
            "WRF."
        ),
    ),
    _adapter(
        "aifs",
        name="ECMWF AIFS single (open data)",
        aliases=("aifs-v2", "aifs-single"),
        file_family="GRIB2",
        decoder=(
            "packaged AIFS single profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="oper",
        required_products=("oper",),
        max_hour=360,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-aifs-single-grib2-v1",
        level_mapping="13-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-six-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="aifs-single-grib2-v1",
        forcing_interval_seconds=21600.0,
        notes=(
            "ECMWF's AI forecast (AIFS single, open data 0.25-degree "
            "GRIB2) through the GENERIC mapped route: one file per "
            "six-hourly step carries the 13-level pressure state "
            "(50-1000 hPa; the extra 10 hPa surfaces outside the declared "
            "ladder are admitted-and-ignored, and specific humidity is "
            "not published there), the surface/2 m/10 m fields and the "
            "two-layer ordinal soil column; the land mask and surface "
            "geopotential ride the 0-hour file alone and are declared "
            "cycle-invariant.  THREE limits a reader must know.  (1) The "
            "soil column reaches 0.28 m: Noah's four layers are WRF's own "
            "shallow-column interpolation bracketed by the skin "
            "temperature at 0 m and the static deep-soil temperature at "
            "3 m, smoother than a full-depth soil analysis.  (2) No snow "
            "state and no sea-ice fraction are published, so runs start "
            "bare-ground and open-water everywhere; a winter or polar "
            "case should not initialise from this product.  (3) An AI "
            "emulator publishes no hydrometeors and its fields are not "
            "constrained by a dynamical balance, so spin-up behaviour is "
            "not the certified-source baseline.  Not yet accepted by "
            "unchanged stock WRF."
        ),
    ),
    _adapter(
        "rap",
        name="RAP (32 km North America)",
        aliases=("rap-awip32",),
        file_family="GRIB2",
        decoder=(
            "packaged RAP awip32 profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="awip32",
        required_products=("awip32",),
        max_hour=51,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-rap-awip32-grib2-v1",
        level_mapping="39-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="rap-awip32-grib2-v1",
        forcing_interval_seconds=3600.0,
        coverage=_NORTH_AMERICA_32KM_LAMBERT,
        notes=(
            "RAP through the GENERIC mapped route as a packaged profile: "
            "one awip32 file (AWIPS grid 221, 32 km Lambert, North "
            "America) per hourly valid time carries the 39-level pressure "
            "state (the HRRR wrfprs ladder, byte for byte), the "
            "surface/2m/10m fields, in-band terrain and the nine-node RUC "
            "soil column.  Pressure-level humidity is RH and rides the "
            "declared RH-to-specific-humidity derivation the GFS profile "
            "proved.  There is no RAP decode code on this route.  The "
            "13 km CONUS Lambert products are NOT reachable as tables "
            "today and this row does not claim them: awp130pgrb carries "
            "no soil/land state, and pairing it with awp130bgrb refuses "
            "because both products publish byte-identical surface records "
            "that no selector octet separates; the native rotated "
            "lat-lon wrfprs product (GDT 32769) is outside the declared "
            "grid families.  Not yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "nam",
        name="NAM (12 km North America)",
        default_product="awip12", max_hour=84),
    _adapter(
        "hiresw",
        name="HiResW (high-resolution window)", aliases=("hires",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        default_product="arw_2p5km/conus", max_hour=48,
        status=AdapterStatus.MEMBER_SELECTION_REQUIRED,
        composition="Select the concrete ARW/FV3 member before initialization.",
    ),
    _adapter(
        "href",
        name="HREF (high-resolution ensemble products)",
        aliases=("href-conus",),
        kind=SourceKind.ENSEMBLE_STATISTIC,
        default_product="ensprod/conus/sprd", max_hour=48,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Use a constituent deterministic member; spread/probability products cannot initialize WRF.",
    ),
    _adapter(
        "sref",
        name="SREF (short-range ensemble products)",
        kind=SourceKind.ENSEMBLE_STATISTIC,
        default_product="ensprod/pgrb212/mean_3hrly", max_hour=87,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Use a constituent member; the ensemble mean is not a dynamically balanced member state.",
    ),
    _adapter(
        "rtma",
        name="RTMA (real-time mesoscale analysis)",
        kind=SourceKind.SURFACE_ANALYSIS,
        default_product="2dvaranl_ndfd", max_hour=0,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Provide a complete 3-D atmosphere source; RTMA may replace declared surface fields only.",
    ),
    _adapter(
        "urma",
        name="URMA (unrestricted mesoscale analysis)",
        kind=SourceKind.SURFACE_ANALYSIS,
        default_product="2dvaranl_ndfd", max_hour=0,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Provide a complete 3-D atmosphere source; URMA may replace declared surface fields only.",
    ),
    _adapter(
        "nbm",
        name="NBM (National Blend of Models)", aliases=("blend",),
        kind=SourceKind.POSTPROCESSED_GUIDANCE,
        default_product="core/co", max_hour=264,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Provide a complete 3-D analysis/forecast state; NBM is postprocessed guidance.",
    ),
    _adapter(
        "rrfs",
        name="RRFS (operational 3 km CONUS)",
        aliases=("rrfs-ops",),
        file_family="GRIB2",
        decoder=(
            "packaged RRFS prslev+2dfld profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="prslev",
        required_products=("prslev", "2dfld"),
        max_hour=84,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-rrfs-prslev-2dfld-grib2-v1",
        level_mapping="45-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="rrfs-prslev-2dfld-grib2-v1",
        forcing_interval_seconds=3600.0,
        coverage=_CONUS_3KM_LAMBERT,
        notes=(
            "RRFS -- HRRR's operational successor, flowing today on "
            "noaa-rrfs-ops-pds and NOMADS rrfs/v1.0 (implementation date "
            "2026-10-06) -- through the GENERIC mapped route as a packaged "
            "profile.  The 3 km CONUS grid is bit-for-bit HRRR's Lambert "
            "(every geolocating octet identical, measured from real "
            "bytes), so the HRRR wrfprs machinery carries it; RRFS's own "
            "facts are table data: a 45-level pressure ladder (70 hPa "
            "where HRRR has 75, 2 hPa top, no 1013.2 hPa entry) and the "
            "state split across a prslev/2dfld file PAIR -- prslev alone "
            "carries zero surface fields and zero soil, so both products "
            "are required and both files are passed per valid time.  "
            "SPFH is direct; the nine-node RUC soil column is "
            "octet-identical to HRRR's.  There is no RRFS decode code on "
            "this route.  NOT reachable today and not claimed by this "
            "row: the natlev native-level product, the 3 km "
            "North-America rotated grid, the thinned subset files and "
            "every per-member ensemble file exist only in the frozen "
            "prototype bucket noaa-rrfs-pds (halted 2026-08-12 by "
            "design) with no live front door; the relocatable firewx "
            "nest changes grid daily.  The ops bucket currently lists 6 "
            "days and whether that is rolling retention or feed age is "
            "not yet distinguishable.  Not yet accepted by unchanged "
            "stock WRF."
        ),
    ),
    _adapter(
        "rrfs-a",
        name="RRFS-A (frozen prototype feed)", aliases=("rrfsa",),
        default_product="prs-conus", max_hour=60,
        upstream_ingest="full",
        notes="Frozen prototype feed (noaa-rrfs-pds, halted 2026-08-12 by "
              "design); the live operational feed is the `rrfs` row.  "
              "rusty-weather ingest is mature; WRF state/soil mapping "
              "remains gated.",
    ),
    _adapter(
        "rrfs-public",
        name="RRFS public prototype",
        default_product="prs-conus", max_hour=60),
    _adapter(
        "refs",
        name="REFS (RRFS ensemble products)", aliases=("rrfs-ensemble",),
        kind=SourceKind.ENSEMBLE_STATISTIC,
        default_product="mean-conus", max_hour=60,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Use a constituent RRFS member; mean/PMMN/spread products cannot initialize a member.",
    ),
    _adapter(
        "rrfs-firewx",
        name="RRFS fire-weather nest", aliases=("firewx",),
        default_product="2dfld-firewx", max_hour=36,
        required_products=("2dfld-firewx", "prs-firewx"),
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Combine the 2-D fire-weather product with a complete pressure/native atmosphere and soil state.",
    ),
    _adapter(
        "wrf",
        name="WRF archive (ARW history)", aliases=("wrf-gdex", "wrf-arw"),
        kind=SourceKind.WRF_ARCHIVE,
        file_family="NetCDF",
        default_product="surface",
        max_hour=0,
        status=AdapterStatus.ARCHIVE_MAPPING_REQUIRED,
        composition="Map a compatible WRF archive state, vertical coordinate, physics state, and boundary source.",
    ),
    _adapter(
        "era5",
        name="ERA5 reanalysis (ECMWF)",
        credentials=(_COPERNICUS_CDS_KEY,),
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
        forcing_interval_seconds=21600.0,
        # THE ROW THAT MADE THE COLUMN.  `--cycle latest` used to be
        # refused here with the sentence "a reanalysis published with
        # a delay of several days" -- which describes a DELAY, and a
        # delay is a number, and a number resolves.  ERA5T (the
        # preliminary stream the CDS serves for recent dates) runs
        # about five days behind real time.  No probe: the CDS is a
        # keyed job API, not a file server, so there is no object to
        # HEAD and this delay IS the answer.
        cycles=CycleGrid(
            hours=(0, 6, 12, 18), delay_hours=120.0,
            search_hours=168,
            basis="ERA5T preliminary is published about five days "
                  "behind real time, on the 6-hourly analysis grid "
                  "this row's forcing_interval_seconds declares"),
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
        "era5-l137",
        name="ERA5 reanalysis (native 137 model levels)",
        aliases=("era5-model-level", "era5-ml"),
        credentials=(_COPERNICUS_CDS_KEY,),
        upstream_model_id=None,
        file_family="GRIB2",
        decoder=(
            "packaged ERA5 model-level profile + "
            "vendored rusty-weather GRIB2/GRIB1 bridges"
        ),
        default_product="model-level+pressure-level-surface",
        required_products=(
            "model-level atmosphere (all 137 levels)",
            "same-hour pressure-level/single-level analysis",
        ),
        # A reanalysis: every valid time is an analysis, never a lead.
        max_hour=0,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-era5-model-level-l137-grib2-v1",
        level_mapping="137-hybrid-sigma-pressure-model-level-to-explicit-wrf-eta-v1",
        cadence_mapping="uniform-hourly-analysis-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="era5-model-level-l137-grib2-v1",
        forcing_interval_seconds=3600.0,
        # The same reanalysis on the same release schedule as the
        # pressure-level row, on the hourly grid this variant reads.
        cycles=CycleGrid(
            hours=tuple(range(24)), delay_hours=120.0,
            search_hours=168,
            basis="the ERA5T release schedule, on the hourly grid "
                  "this row's forcing_interval_seconds declares"),
        composition=(
            "ERA5's model-level product publishes the prognostic "
            "atmosphere only; the land-surface and near-surface state -- "
            "surface pressure, orography, land fraction, skin "
            "temperature, the 2 m and 10 m diagnostics and the four-layer "
            "soil column -- is borrowed from the SAME HOUR's ERA5 "
            "pressure-level/single-level analysis, so a preparation takes "
            "two files and refuses by name if the donor hour disagrees."
        ),
        notes=(
            "The native vertical state, as pure table data: 137 hybrid "
            "sigma-pressure model levels whose A/B interface coefficients "
            "ride IN BAND in the GRIB2 Section-4 coordinate-values (pv) "
            "octets -- 276 values, read from the producer's own record "
            "rather than from a per-model coefficient table.  Pressure is "
            "materialized as p = A + B*ps against the borrowed surface "
            "pressure and geopotential height is integrated "
            "hydrostatically from the borrowed terrain, because the "
            "model-level product publishes z only at level 1 and a 3-D "
            "height borrow would be cross-ladder.  lnsp is NOT consumed: "
            "unit transforms are affine, there is no exp, and surface "
            "pressure is borrowed exactly instead of approximated.  "
            "Measured on real Copernicus CDS bytes: the Rust engine and "
            "the Python reference produce byte-identical air_pressure and "
            "geopotential_height (max ULP 0) and the preparation reaches "
            "rc 0 with wrfinput/wrfbdy written.  Not yet accepted by "
            "unchanged stock WRF, and this row does not certify ERA5 "
            "model-level requests on other grids or level subsets: the "
            "mapping's ladder is the full 1..137 set."
        ),
    ),
    _adapter(
        "20crv3",
        name="20CRv3 reanalysis (member, GRIB2)",
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
        packaged_profile="20crv3-member-grib2-v1",
        forcing_interval_seconds=10800.0,
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
        name="20CRv3 reanalysis (ensemble mean, NetCDF)",
        aliases=("20crv3-netcdf", "20cr-netcdf", "20cr-cf"),
        upstream_model_id=None,
        # TRUTHFUL, and it is the one thing a reader must not miss: NOAA
        # PSL's NetCDF distribution of 20CRv3 is the ENSEMBLE MEAN
        # analysis.  Every variable in it carries `statistic = "Ensemble
        # Mean"`, which the packaged mapping binds as a selector attribute,
        # so a member file cannot be fed through this profile unnoticed and
        # a receipt can never say "member" about data that is not one.
        kind=SourceKind.ENSEMBLE_STATISTIC,
        file_family="NetCDF-CF",
        decoder=(
            "packaged 20CRv3 NetCDF profile + rw_netcdf/netcrust "
            "(gpuwm.netcdf_bridge)"
        ),
        default_product="si-subdaily-pressure+surface+subsurface",
        required_products=(
            "prsSI", "sfcSI", "2mSI", "10mSI", "subsfcSI",
            "recovered invariant supplement",
        ),
        max_hour=0,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-20crv3-si-netcdf-v1",
        level_mapping="21-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-three-hour-analysis-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="20crv3-netcdf-v1",
        forcing_interval_seconds=10800.0,
        # The PSL humidity series stops at 100 hPa and the mapping
        # declares the intersection, so 100 hPa is this route's ladder.
        certified_source_top_pa=10000.0,
        notes=(
            "The publicly downloadable form of 20CRv3: NOAA PSL's sub-daily "
            "NetCDF SI series, decoded through the Rust rw_netcdf bridge with "
            "no per-source decode code -- the mapping, composition and "
            "provenance documents are packaged and pinned by SHA-256. "
            "Three-hourly analyses on 21 common pressure levels (1000-100 "
            "hPa; the humidity file stops at 100 hPa while the others reach 1 "
            "hPa, and the mapping declares the intersection), a complete "
            "surface state, and the exact four Noah soil layers. "
            "TWO limits a reader must know. (1) It is the ENSEMBLE MEAN "
            "analysis, not a member: it is smoother than any single 20CRv3 "
            "trajectory, and where a member state is wanted the route is "
            "--source 20crv3 over the every-member GRIB2 archive. The "
            "registry's refusal of GEFS/HREF/SREF means does not apply here "
            "for a stated reason -- those are FORECAST ensembles whose mean is "
            "not a dynamically balanced trajectory, while this is an analysis "
            "at its own valid time and is the only 20CRv3 form NOAA "
            "distributes as NetCDF -- but the difference is a judgement worth "
            "reading before a study depends on it. (2) PSL publishes no "
            "orography and no land mask for 20CRv3, so both are recovered "
            "from 20CRv3's own published fields by "
            "tools/build_pressure_level_invariant_supplement.py and carried "
            "as a supplement whose provenance document states the method and "
            "the divergence. Not yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "mapped",
        name="Generic mapped source (rw-wps mapping v1)",
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


def packaged_profile_sources() -> Mapping[str, str]:
    """``source_id -> packaged profile`` for every source that names one.

    THE table both the preparation stage and the forecast stage read, so
    the answer to "is this a shipped profile, and which" is given once.
    It lives in the registry rather than in either stage because the
    registry is what a new model is added to: a row here is the entire
    difference between a model gpuwm can decode and one it cannot.
    """

    return MappingProxyType({
        adapter.source_id: adapter.packaged_profile
        for adapter in _ADAPTERS
        if adapter.packaged_profile is not None
    })


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


def wizard_planable_source_ids() -> tuple[str, ...]:
    """Every source id ``gpuwm domain`` can emit a runnable config for.

    A row qualifies on exactly two declared facts: it is ``runnable`` (a
    strict implementation route exists) and it declares a
    ``forcing_interval_seconds`` (the wizard has a boundary cadence to
    write into ``namelist.wps``).  Nothing here enumerates model names, so
    a new row appears in `gpuwm domain --help` the moment it is added --
    which is the whole claim this function is here to keep true.
    """

    return tuple(adapter.source_id for adapter in _ADAPTERS
                 if adapter.runnable
                 and adapter.forcing_interval_seconds is not None)


def source_forcing_interval_seconds(source: str) -> float:
    """Seconds between SOURCE's native valid times, from its registry row.

    Raises a fail-closed :class:`ValueError` naming the breakage when the
    row declares none: a config emitted with a guessed boundary cadence
    would ask the preparation for valid times the source never published.
    """

    adapter = get_source_adapter(source)
    if adapter.forcing_interval_seconds is None:
        raise ValueError(
            f"source {adapter.source_id!r} declares no native forcing "
            "cadence, so there is no &share/interval_seconds to emit and a "
            "config written for it would name boundary times the source "
            "does not publish; its cadence comes from the mapping document "
            "a caller supplies, not from the registry")
    return float(adapter.forcing_interval_seconds)


def source_coverage_window(source: str) -> CoverageWindow | None:
    """SOURCE's declared native-grid window, or ``None`` when it is global."""

    return get_source_adapter(source).coverage_window


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
        # Every packaged profile, not one hand-named row: a profile that
        # ships is a profile whose digests a consumer can check.
        "packaged_source_authorities": {
            profile_id: dict(packaged_authority_sha256(profile_id))
            for profile_id in packaged_profile_ids()
        },
        # The ensemble member-addressing grammars ship under the same
        # rule: a packaged document is a document whose digest a
        # consumer can check.
        "packaged_member_grammars": {
            grammar_id: packaged_member_grammar_sha256(grammar_id)
            for grammar_id in packaged_member_grammar_ids()
        },
        "mapped_stock_wrf_evidence": [
            {
                "mapping_config": "rw-wps-era5-1974-probe.mapping.json",
                "composition_config": (
                    "rw-wps-era5-1974-terrain.composition.json"
                ),
                "mapping_sha256": (
                    "2ceb90e63d3265c0ebf871de4bfc7f622284681a9861c17e78e9fc11e3dc15b2"
                ),
                "composition_sha256": (
                    "9aa8c7aec96a1d52e83f61c470cdb2a636ca092af902581bafb05fc433ab57f3"
                ),
                "gate": "wrf-v4.6.1-pass-era5-grib1-single-domain",
                "source_format": "grib1",
            },
            {
                "mapping_config": "rw-wps-gfs-pressure-grib2.mapping.json",
                "composition_config": "rw-wps-gfs-terrain.composition.json",
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
        # An entry here means: this gate really passed, on the bytes named by
        # ``mapping_sha256``/``composition_sha256``, and those bytes are no
        # longer what the product ships.  ``mapping_config`` names the file
        # whose CURRENT bytes the ``replacement_*`` hashes report, so a reader
        # can see both what was proved and what is shipped.
        "invalidated_mapped_stock_wrf_evidence": [
            {
                "mapping_config": "rw-wps-gfs-pressure-grib2.mapping.json",
                "composition_config": "rw-wps-gfs-terrain.composition.json",
                "mapping_sha256": (
                    "726677d8c2365e6f533cc6dd5d7c795e198164326660c3630d885c83f406a11e"
                ),
                "composition_sha256": (
                    "266c98099b24f03a3bc986f275b44bbd6bf20dce1006ad05f6da39bc4a373bfb"
                ),
                "gate": "wrf-v4.6.1-pass-gfs-grib2-d01-d02",
                "source_format": "grib2",
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
            {
                "mapping_config": "rw-wps-era5-netcdf.mapping.json",
                "composition_config": (
                    "rw-wps-era5-netcdf-terrain.composition.json"
                ),
                "mapping_sha256": (
                    "f278705331a81767d4d3532ff4dd4f739242a79b224747f51e722d462142daa8"
                ),
                "composition_sha256": (
                    "10b23b56b882ed534dbcb5219ac06f9dcc184226e7cc2aea3c6c03f3ad6be459"
                ),
                "gate": "wrf-v4.6.1-pass-era5-netcdf-single-domain",
                "source_format": "netcdf",
                "reason": (
                    "the shipped ERA5 NetCDF mapping gained an ordered list of "
                    "accepted time-coordinate spellings for ECMWF's "
                    "time/valid_time rename, which moved its bytes; evidence "
                    "does not transfer across contract hashes"
                ),
                "replacement_mapping_sha256": (
                    _current_config_sha256(
                        "rw-wps-era5-netcdf.mapping.json",
                        "d2c9ee08e45478a64e4d2bba689e9bad1d2e97bde713477ee2a4de26e31d7ad3",
                    )
                ),
                "replacement_composition_sha256": (
                    _current_config_sha256(
                        "rw-wps-era5-netcdf-terrain.composition.json",
                        "87387cd24d3fb4c8488eced6734f16c9db76400929139d8887eecf916eb07ce7",
                    )
                ),
                # NOT a replacement gate: stock wrf.exe has not been run
                # against these bytes.  Naming a passing gate here would be
                # the exact thing the reason above forbids -- old evidence
                # silently certifying a new contract -- so the field that
                # would carry it says what is actually true instead.
                "replacement_status": "stock_wrf_regate_required",
                "replacement_gate": None,
            },
        ],
        "sources": [adapter.to_dict() for adapter in _ADAPTERS],
    }
