"""Minimal WRF-style "wrfout" NetCDF writer.

``WrfoutWriter`` lays out WRF-ARW dimensions/staggering and stamps each
variable with the WRF attribute set (``FieldType``/``MemoryOrder``/
``description``/``units``/``stagger``) so standard wrfout tooling can read
the files.  ``state_frame`` builds the standard frame dict from a
``DomainState`` (winds, T = theta - 300, PH/MU perturbations, the
terrain-consistent PHB/MUB/HGT base fields, and QVAPOR/QCLOUD/QRAIN when
the state carries moisture).  Active Morrison, precipitation, snow, Noah,
and vertical-coordinate history state is included under its WRF Registry
name.  ``wrf_time_str`` formats the ``Times`` record.  Only ``state_frame``
touches the GPU (deferred cupy import) -- the writer itself is
CPU-importable.
"""
from __future__ import annotations
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import queue
import threading
import traceback

import numpy as np
import netCDF4

from gpuwm.config import NO_LAND_SURFACE_SOIL_LAYERS, soil_layer_count
from gpuwm.io.wrf_output_schema import (
    HISTORY_FIELDS_BY_NETCDF_NAME, PHYSICS_SELECTOR_GLOBALS,
    SCHEME_OUTPUT_FIELDS, WRF_FIELD_TYPE_INTEGER, WRF_FIELD_TYPE_REAL,
)
from gpuwm.supervisor import (fsync_file, quarantine_file,
                              replace_file_with_retry, unique_temp_path)

_COMPLETION_ATTR = "GPUWM_WRITE_COMPLETE"
# netCDF4 releases the GIL around HDF5 calls, while the shipped HDF5 library
# is not thread-safe.  Domain D2H streams and staging remain independent; only
# each worker's create/write/close/reopen/publish netCDF session is serialized.
_NETCDF4_IO_LOCK = threading.Lock()
_ASYNC_WRITER_POLL_SECONDS = 0.05


def _stringify_and_clear_exception_tracebacks(exc: BaseException) -> str:
    """Return diagnostics without retaining any exception-chain frames."""
    rendered = "".join(traceback.format_exception(
        type(exc), exc, exc.__traceback__))
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        current.__traceback__ = None
    return rendered


#: WRF Registry metadata ``name -> (description, units)`` for the
#: variables gpuwm writes; unknown names get empty strings.
_VAR_META = {
    "U": ("x-wind component", "m s-1"),
    "V": ("y-wind component", "m s-1"),
    "W": ("z-wind component", "m s-1"),
    "T": ("perturbation potential temperature theta-t0", "K"),
    "P": ("perturbation pressure", "Pa"),
    "PB": ("BASE STATE PRESSURE", "Pa"),
    "PH": ("perturbation geopotential", "m2 s-2"),
    "PHB": ("base-state geopotential", "m2 s-2"),
    "MU": ("perturbation dry air mass in column", "Pa"),
    "MUB": ("base state dry air mass in column", "Pa"),
    "QVAPOR": ("Water vapor mixing ratio", "kg kg-1"),
    "QCLOUD": ("Cloud water mixing ratio", "kg kg-1"),
    "QRAIN": ("Rain water mixing ratio", "kg kg-1"),
    "QICE": ("Ice mixing ratio", "kg kg-1"),
    "QSNOW": ("Snow mixing ratio", "kg kg-1"),
    "QGRAUP": ("Graupel mixing ratio", "kg kg-1"),
    "QHAIL": ("Hail mixing ratio", "kg kg-1"),
    "QNDROP": ("Droplet number mixing ratio", "# kg-1"),
    "QNCLOUD": ("cloud water Number concentration", "  kg(-1)"),
    "QNRAIN": ("Rain Number concentration", "  kg(-1)"),
    "QNICE": ("Ice Number concentration", "  kg-1"),
    "QNSNOW": ("Snow Number concentration", "  kg(-1)"),
    "QNGRAUPEL": ("Graupel Number concentration", "  kg(-1)"),
    "QNHAIL": ("Hail Number concentration", "# kg(-1)"),
    "QNCCN": ("CCN Number concentration", "# kg(-1)"),
    # QNWFA/QNIFA/QNWFA2D/QNIFA2D (mp_physics=28) are deliberately NOT here.
    # They have transcribed rows -- with NetCDF type, FieldType and the
    # stagger the writer cross-checks -- in
    # gpuwm.io.wrf_output_schema.THOMPSON_AEROSOL_OUTPUT_FIELDS, which
    # _create_variable consults BEFORE this fallback table.  Same reason the
    # precipitation accumulators moved out below: two tables describing one
    # field is how two tables come to disagree about one field.
    "QVGRAUPEL": ("Graupel Particle Volume", "m(3) kg(-1)"),
    "QVHAIL": ("Hail Particle Volume", "m(3) kg(-1)"),
    # Registry.EM_COMMON:1596 (verified against the reference wrfout's
    # REFL_10CM attributes).  Opt-in: cases merge the field into their
    # frame dict (gpuwm.core.refl.compute_refl_10cm supplies the values).
    "REFL_10CM": ("Radar reflectivity (lamda = 10 cm)", "dBZ"),
    # Registry.EM_COMMON:2083 (nwp_output package, IO "rh02").  Present
    # exactly when the run carries the accumulator (nwp_diagnostics = 1;
    # gpuwm.core.uh_diag owns the per-step update and the post-write reset).
    "UP_HELI_MAX": ("MAX UPDRAFT HELICITY", "m2 s-2"),
    "HGT": ("Terrain Height", "m"),
    "PSFC": ("SFC PRESSURE", "Pa"),
    # RAINC/RAINSH/RAINNC/SNOWNC/GRAUPELNC/HAILNC used to live here.  They
    # are now in gpuwm.io.wrf_output_schema.PRECIPITATION_OUTPUT_FIELDS,
    # because they are no longer merely *described* by this table -- they are
    # emitted unconditionally, and the thing that decides that has to be the
    # same thing that carries their Registry metadata.  Two tables describing
    # one field is how two tables come to disagree about one field.
    "SNOW": ("SNOW WATER EQUIVALENT", "kg m-2"),
    "SNOWH": ("PHYSICAL SNOW DEPTH", "m"),
    "SNOWC": ("FLAG INDICATING SNOW COVERAGE (1 FOR SNOW COVER)", ""),
    "TSLB": ("SOIL TEMPERATURE", "K"),
    "SMOIS": ("SOIL MOISTURE", "m3 m-3"),
    "SH2O": ("SOIL LIQUID WATER", "m3 m-3"),
    "SWDOWN": ("DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE", "W m-2"),
    "GLW": ("DOWNWARD LONG WAVE FLUX AT GROUND SURFACE", "W m-2"),
    # Registry.EM_COMMON:1839, transcribed verbatim beside its two
    # surface-flux neighbours.  ``ij`` mass grid, no stagger, real -- so
    # the dimension table and the FieldType default already describe it
    # correctly and it needs no row of its own.
    "OLR": ("TOA OUTGOING LONG WAVE", "W m-2"),
    "TSK": ("SURFACE SKIN TEMPERATURE", "K"),
    "T2": ("TEMP at 2 M", "K"),
    "TH2": ("POT TEMP at 2 M", "K"),
    "Q2": ("QV at 2 M", "kg kg-1"),
    "U10": ("U at 10 M", "m s-1"),
    "V10": ("V at 10 M", "m s-1"),
    "UST": ("U* IN SIMILARITY THEORY", "m s-1"),
    "HFX": ("UPWARD HEAT FLUX AT THE SURFACE", "W m-2"),
    "QFX": ("UPWARD MOISTURE FLUX AT THE SURFACE", "kg m-2 s-1"),
    "LH": ("LATENT HEAT FLUX AT THE SURFACE", "W m-2"),
    "TKE_SASE": ("SASE prognostic subgrid turbulence kinetic energy",
                 "m2 s-2"),
    "TKE_SHINHONG": ("Shin-Hong published subgrid turbulence kinetic "
                     "energy diagnostic", "m2 s-2"),
    # The SPLIT SUBGRID-FLUX DIAGNOSTIC (RunConfig sase_flux_diag, off by
    # default; per domain).  Four z-FACE fields on the mass column --
    # (nz+1, ny, nx), stagger "Z", registered exactly like W -- carrying
    # the closure's own vertical subgrid fluxes with the conditional-
    # venting channel separated from the K_v implicit vertical diffusion
    # channel.  BOTH CHANNELS ARE POSITIVE UPWARD and both divide by the
    # SAME lowest-level moist density the venting deposit divides by, so
    # VENT + DIFF is the closure's total subgrid flux and its
    # convergence is the model's own scalar increment.  Face 0 is +0.0
    # by the interface contract: the surface flux is owned by HFX/QFX
    # and is not double-counted here.
    "SASE_FQV_VENT": ("SASE vent subgrid water vapor flux, positive up",
                      "kg m-2 s-1"),
    "SASE_FQV_DIFF": ("SASE Kv subgrid water vapor flux, positive up",
                      "kg m-2 s-1"),
    "SASE_FTH_VENT": ("SASE vent subgrid heat flux, positive up",
                      "W m-2"),
    "SASE_FTH_DIFF": ("SASE Kv subgrid heat flux, positive up",
                      "W m-2"),
    # The HORIZONTAL EDDY-VISCOSITY DIAGNOSTIC (RunConfig hmix_k_diag,
    # off by default; per domain).  Two (nz, ny, nx) mass-grid fields
    # carrying the horizontal mixing coefficients the run's own producer
    # used, under that producer's name: XKMH/XKHH are WRF's Registry
    # names for the km_opt = 4 2-D Smagorinsky viscosities, SASE_KMH/
    # SASE_KHH the SASE closure's governed horizontal diffusivity and
    # the K_h = KMH/Pr_t(f) its scalar channel rides.  Same units, same
    # grid, same meaning, so the two are directly comparable -- which is
    # the point: SASE's claim to replace the km_opt operator is a claim
    # about this number.  A run with NO horizontal mixing producer (the
    # acknowledged km_opt = 0 control) publishes NEITHER pair, so an
    # absent variable says "no operator ran" and can never be misread as
    # a measured zero.  Instantaneous: the value the last completed
    # model step used, not a history-interval mean.
    "XKMH": ("HORIZONTAL MOMENTUM EDDY VISCOSITY", "m2 s-1"),
    "XKHH": ("HORIZONTAL HEAT EDDY VISCOSITY", "m2 s-1"),
    "SASE_KMH": ("SASE governed horizontal momentum diffusivity",
                 "m2 s-1"),
    "SASE_KHH": ("SASE governed horizontal scalar diffusivity, KMH/Pr_t",
                 "m2 s-1"),
    "PBLH": ("PBL HEIGHT", "m"),
    "GRDFLX": ("GROUND HEAT FLUX", "W m-2"),
    "PSIM": ("SIMILARITY STABILITY FUNCTION FOR MOMENTUM", ""),
    "PSIH": ("SIMILARITY STABILITY FUNCTION FOR HEAT", ""),
    "XLAT": ("LATITUDE, SOUTH IS NEGATIVE", "degree_north"),
    "XLONG": ("LONGITUDE, WEST IS NEGATIVE", "degree_east"),
    "XLAT_U": ("LATITUDE, SOUTH IS NEGATIVE", "degree_north"),
    "XLONG_U": ("LONGITUDE, WEST IS NEGATIVE", "degree_east"),
    "XLAT_V": ("LATITUDE, SOUTH IS NEGATIVE", "degree_north"),
    "XLONG_V": ("LONGITUDE, WEST IS NEGATIVE", "degree_east"),
    "MAPFAC_M": ("Map scale factor on mass grid", ""),
    "MAPFAC_U": ("Map scale factor on u-grid", ""),
    "MAPFAC_V": ("Map scale factor on v-grid", ""),
    "F": ("Coriolis sine latitude term", "s-1"),
    "E": ("Coriolis cosine latitude term", "s-1"),
    "LU_INDEX": ("LAND USE CATEGORY", ""),
    "LANDMASK": ("LAND MASK (1 FOR LAND, 0 FOR WATER)", ""),
    "SINALPHA": ("Local sine of map rotation", ""),
    "COSALPHA": ("Local cosine of map rotation", ""),
    "P_TOP": ("PRESSURE TOP OF THE MODEL", "Pa"),
    "ZNU": ("eta values on half (mass) levels", ""),
    "ZNW": ("eta values on full (w) levels", ""),
}

#: Staggered dimension -> WRF ``stagger`` attribute value.  WRF appends
#: ``_stag`` to a dimension's dataset name exactly when the field is staggered
#: on that axis (``tools/gen_wrf_io.c:138,187,213``), which is why the snow
#: and snow+soil axes below are ``*_stag`` and not the bare dimspec names:
#: every Noah-MP field on them is declared ``Z``-staggered.
_STAGGER = {"west_east_stag": "X", "south_north_stag": "Y",
            "bottom_top_stag": "Z", "soil_layers_stag": "Z",
            "snow_layers_stag": "Z", "snso_layers_stag": "Z"}

#: Noah-MP snow-stack history fields, keyed by name for the same reason
#: TSLB/SMOIS/SH2O are: their leading extent is a scheme's own vertical axis,
#: which collides with ``bottom_top`` for a shallow model column.  The
#: dimension names are WRF's (``Registry/registry.dimspec:53,55``: ``snly`` ->
#: ``snow_layers``, ``snsl`` -> ``snso_layers``, each ``_stag``-suffixed by
#: the declared ``Z`` staggering), so a reader that knows WRF knows these.
#: The names are the external ones from the output schema, never the Registry
#: symbols -- ``TSNOXY`` is not a name any WRF ever wrote.
_SNOW_LAYER_FIELDS = frozenset(
    SCHEME_OUTPUT_FIELDS[key].netcdf_name
    for key in ("tsnoxy", "snicexy", "snliqxy"))
_SNSO_LAYER_FIELDS = frozenset({SCHEME_OUTPUT_FIELDS["zsnsoxy"].netcdf_name})

#: MYNN's three ``ikj`` carriers that WRF declares ``Z``-staggered, so WRF
#: writes them on ``bottom_top_stag`` with one more level than the mass
#: column.  gpuwm computes exactly the ``nz`` mass levels the MYNN solver
#: fills; WRF's own array is dimensioned ``kms:kme`` and its PBL driver never
#: writes the extra interface entry either, leaving the Registry cold value
#: zero there.  The writer lifts these onto WRF's axis and pads with that same
#: zero, so a reader that knows WRF's EL_PBL shape receives WRF's EL_PBL
#: shape -- rather than an ``nz``-level array under a name whose declared
#: schema says ``nz + 1``.
_Z_STAGGERED_MASS_FIELDS = frozenset(
    SCHEME_OUTPUT_FIELDS[key].netcdf_name
    for key in ("el_pbl", "exch_h", "exch_m"))

#: Every history field whose leading extent is the SOIL axis.  TSLB, SMOIS and
#: SH2O are generic; SMFR3D and KEEPFR3DFLAG are RUC's own Registry package
#: line (``Registry.EM_COMMON:3147``) and are written only under
#: ``sf_surface_physics=3``.  They belong here for exactly the reason the snow
#: sets above exist: ``_dims_for``'s shape table is keyed on ``(nz, ny, nx)``,
#: so a soil array is unroutable by shape alone and a nine-level soil field
#: over a twenty-level column raises ``KeyError`` rather than picking a wrong
#: axis.  That is what it did before these two names were added.
_SOIL_LAYER_FIELDS = frozenset({"TSLB", "SMOIS", "SH2O",
                                "SMFR3D", "KEEPFR3DFLAG"})

_DEFAULT_LANDUSE_ATTRS = {
    "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
    "ISWATER": 17,
    "ISLAKE": 21,
    "ISICE": 15,
    "ISURBAN": 13,
}


def _producer_version() -> str:
    """The release that wrote this file, from installed metadata.

    Imported at call time rather than at module import so the writer stays
    importable from a source tree that was never installed; ``gpuwm``
    answers ``0+unknown`` there rather than inventing a number.
    """
    from gpuwm import __version__

    return str(__version__)


def wrfout_filename(valid_time, domain_id: int = 1) -> str:
    """Colon-free, second-complete WRF history filename.

    A sub-second instant is refused rather than formatted.  The name carries
    whole seconds and the publisher replaces an existing file, so three legal
    quarter-second frames used to collapse onto one name and silently
    overwrite each other -- the second and third frames of that run simply
    ceased to exist, with no exception anywhere.  Losing output is not an
    acceptable answer to an unsupported cadence; refusing it is.
    """
    if (isinstance(domain_id, bool) or not isinstance(domain_id, int)
            or domain_id < 1):
        raise ValueError(
            f"domain_id must be a positive integer, got {domain_id!r}")
    if getattr(valid_time, "microsecond", 0):
        raise ValueError(
            f"wrfout valid time {valid_time!r} is not on a whole second; "
            "history filenames and the Times record carry whole seconds "
            "only, so distinct sub-second instants would alias onto one "
            "file and the later frame would replace the earlier one")
    return valid_time.strftime(
        f"wrfout_d{domain_id:02d}_%Y-%m-%d_%H_%M_%S")


def wrf_physics_selector_attrs(run) -> dict[str, np.int32]:
    """WRF's physics-selector globals, from the resolved gpuwm configuration.

    Stock WRF stamps every physics selector into every history file, and
    that is what lets a reader tell an *absent* scheme from an *inactive*
    one.  Without them a wrfout whose ``RAINSH`` is all zeros is ambiguous:
    a shallow-cumulus scheme may have run and produced nothing, or none may
    exist.  ``SHCU_PHYSICS=0`` is the difference.

    Radiation goes through ``radiation_scheme_ids`` rather than reading the
    two config fields, because gpuwm's ``-1/-1`` is a legacy sentinel
    meaning "use the aggregate ``ra_physics``", not a WRF scheme id.  The
    resolver is the repository's own authority for what actually ran, and
    writing the raw sentinel would put a number in the file that no WRF
    selector has.
    """
    from gpuwm.config import radiation_scheme_ids

    lw, sw = radiation_scheme_ids(run)
    attrs: dict[str, np.int32] = {}
    for selector in PHYSICS_SELECTOR_GLOBALS:
        if selector.source == "config":
            value = getattr(run, selector.run_config_field)
        elif selector.source == "radiation_lw":
            value = lw
        elif selector.source == "radiation_sw":
            value = sw
        elif selector.source == "unimplemented":
            # gpuwm has no such scheme to select, so WRF's "off" value is
            # the resolved truth about this run, not a placeholder.
            value = 0
        else:
            raise ValueError(
                f"unknown selector source {selector.source!r} for "
                f"{selector.name} ({selector.registry})")
        attrs[selector.name] = np.int32(value)
    return attrs


def wrf_global_attrs(
        grid, start_time, *, landuse_attrs=None, grid_id=None,
        parent_id=None, i_parent_start=None, j_parent_start=None,
        parent_grid_ratio=None, dt=None, hybrid_opt=None, etac=None,
        run=None,
        ) -> dict[str, object]:
    """WRF projection/pole/land-use globals derived from case inputs.

    ``grid`` carries the per-domain center and mother-domain center,
    plus its projection identity (``wrf_map_proj``/``map_proj_char``
    per the WRF convention 1=lambert, 2=polar stereographic,
    3=mercator; grids without the attributes -- legacy callers -- keep
    the Lambert identity).  ``landuse_attrs`` comes from the selected
    WPS_GEOG land-use index; omission preserves the established
    MODIS/Noah legacy identity.  Domain topology and vertical identity
    are optional for generic/idealized files, but the real-case runtime
    supplies the complete groups together.  POLE_LAT/POLE_LON stay
    90/0: WPS writes those constants for every non-lat-lon projection.
    """
    # Every projection global below is NC_FLOAT, not NC_DOUBLE.  A Python
    # float would enter netCDF4 as a double, and stock WRF writes all of these
    # single-precision (verified on the group's v4.6.1 wrfout: DX, DY,
    # TRUELAT1/2, STAND_LON, CEN_LAT/LON, MOAD_CEN_LAT, POLE_LAT/LON and DT
    # are every one of them float32).  Readers tolerate the widening, but a
    # file that claims WRF's schema should not differ from WRF's schema.
    attrs = {
        "MAP_PROJ": int(getattr(grid, "wrf_map_proj", 1)),
        "MAP_PROJ_CHAR": str(getattr(grid, "map_proj_char",
                                     "Lambert Conformal")),
        "TRUELAT1": np.float32(grid.truelat1),
        "TRUELAT2": np.float32(grid.truelat2),
        "STAND_LON": np.float32(grid.stand_lon),
        "MOAD_CEN_LAT": np.float32(
            getattr(grid, "moad_cen_lat", grid.ref_lat)),
        "CEN_LAT": np.float32(getattr(grid, "cen_lat", grid.ref_lat)),
        "CEN_LON": np.float32(getattr(grid, "cen_lon", grid.ref_lon)),
        "POLE_LAT": np.float32(90.0), "POLE_LON": np.float32(0.0),
        "SIMULATION_START_DATE": start_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "START_DATE": start_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "GRIDTYPE": "C", **_DEFAULT_LANDUSE_ATTRS,
    }
    if landuse_attrs is not None:
        attrs.update(dict(landuse_attrs))
    domain_values = {
        "GRID_ID": grid_id,
        "PARENT_ID": parent_id,
        "I_PARENT_START": i_parent_start,
        "J_PARENT_START": j_parent_start,
        "PARENT_GRID_RATIO": parent_grid_ratio,
    }
    supplied_domain = [value is not None for value in domain_values.values()]
    if any(supplied_domain) and not all(supplied_domain):
        missing = [name for name, value in domain_values.items()
                   if value is None]
        raise ValueError(
            "WRF domain topology metadata must be supplied together; "
            f"missing {missing}")
    if all(supplied_domain):
        attrs.update({name: np.int32(value)
                      for name, value in domain_values.items()})
    if dt is not None:
        if not np.isfinite(dt) or float(dt) <= 0.0:
            raise ValueError(f"WRF output DT must be finite and > 0, got {dt}")
        attrs["DT"] = np.float32(dt)
    if (hybrid_opt is None) != (etac is None):
        raise ValueError(
            "WRF vertical metadata hybrid_opt and etac must be supplied "
            "together")
    if hybrid_opt is not None:
        attrs["HYBRID_OPT"] = np.int32(hybrid_opt)
        attrs["ETAC"] = np.float32(etac)
    if run is not None:
        attrs.update(wrf_physics_selector_attrs(run))
    return attrs


#: WRF's global date format, the one ``START_DATE`` and
#: ``SIMULATION_START_DATE`` are written in
#: (``share/output_wrf.F:352-376`` of the v4.6.1 source: the
#: ``'(I4.4,"-",I2.2,"-",I2.2,"_",I2.2,":",I2.2,":",I2.2)'`` FORMAT).
#: Provenance dates in a wrfout are written the same way, so a reader
#: parsing dates out of this file needs one parser, not two.  The JSON
#: receipts keep ISO-8601 with the trailing ``Z``; they are documents, not
#: wrfout files, and their convention is unchanged.
_WRF_GLOBAL_DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"

#: The receipts' date format, which is what the preparation proof's
#: initial-condition block is written in.
_RECEIPT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Version tag stamped into every provenance-carrying wrfout, so a
#: consumer can key on a contract rather than on the presence of a name.
WRFOUT_INITIAL_CONDITION_SCHEMA = "gpuwm-wrfout-initial-condition-v1"

#: The complete set of global attributes that record WHAT the initial
#: condition was, as opposed to WHEN the model clock started.
#:
#: DOCUMENTED DIVERGENCE FROM WRF.  WRF v4.6.1 has no convention for
#: this.  ``share/output_wrf.F`` writes exactly two date globals --
#: ``START_DATE`` (this file's own start) and ``SIMULATION_START_DATE``
#: (the simulation's, held across restarts) -- and both are model-clock
#: times; the only provenance-shaped global it writes at all is
#: ``FLAG_RESTART`` on a restart file (output_wrf.F:379-381).  Nothing
#: upstream carries it either: metgrid's ``met_em`` globals are geometry,
#: land-use identity and ``FLAG_*`` presence bits, with no statement of
#: which cycle or which lead the fields came from (verified against the
#: reference bundle's own ``met_em.d01`` and ``wrfout_d01`` files).  So
#: WRF's convention is FOLLOWED where it exists -- ``START_DATE`` and
#: ``SIMULATION_START_DATE`` keep their WRF meaning untouched, and these
#: names are SCREAMING_SNAKE NC_CHAR/NC_INT globals like WRF's own -- and
#: the gap is filled in the ``GPUWM_`` namespace this writer already uses
#: for ``GPUWM_VERSION``/``GPUWM_FEEDBACK``.
#:
#: The names are the receipt's field names, uppercased and prefixed, so
#: the wrfout and ``report.json`` are readable against each other without
#: a mapping table.
INITIAL_CONDITION_GLOBAL_ATTRS = (
    "GPUWM_INITIAL_CONDITION_SCHEMA",
    "GPUWM_INITIAL_CONDITION_KIND",
    "GPUWM_INITIAL_CONDITION_SOURCE",
    "GPUWM_INITIAL_CONDITION_CYCLE",
    "GPUWM_INITIAL_FORECAST_LEAD_HOURS",
    "GPUWM_INITIAL_CONDITION_GENERATING_PROCESS_ID",
    "GPUWM_INITIAL_CONDITION_MODEL_START_DATE",
    "GPUWM_INITIAL_CONDITION_STATEMENT",
)


def _provenance_date(value, field: str) -> datetime:
    """One receipt date, parsed rather than transcribed."""

    if not isinstance(value, str):
        raise ValueError(
            f"initial-condition provenance {field} is not a date string")
    try:
        return datetime.strptime(value, _RECEIPT_DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"initial-condition provenance {field} {value!r} is not a "
            "receipt timestamp") from exc


def initial_condition_global_attrs(
        provenance, *, source: str | None = None) -> dict[str, object]:
    """wrfout globals recording what the initial state WAS.

    ``provenance`` is the preparation receipt's ``initial_condition``
    block (schema ``gpuwm-gfs-initial-condition-provenance-v1``).
    ``None`` returns ``{}``: a route whose preparation publishes no such
    receipt writes no provenance attribute at all, and a reader must
    treat the absence as UNKNOWN.  Absence is deliberately not spelled
    "analysis" -- that is the exact relabelling this contract exists to
    prevent, and pre-1.4.1 files are indistinguishable from it.

    Every field published here is READ and cross-checked, never copied:
    the kind, the generating-process id and the lead must agree with each
    other, and cycle + lead must compose to the declared model start.  A
    block that fails any of those is refused rather than transcribed, so
    a wrfout cannot carry a statement its own fields contradict.
    """

    if provenance is None:
        return {}
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "initial-condition provenance is malformed")

    lead = provenance.get("initial_forecast_lead_hours")
    if isinstance(lead, bool) or not isinstance(lead, int) or lead < 0:
        raise ValueError(
            "initial-condition provenance declares an unreadable initial "
            "forecast lead")
    analysis = lead == 0
    kind = provenance.get("initial_condition_kind")
    process = provenance.get("forecast_generating_process_id")
    if (kind != ("analysis" if analysis else "forecast")
            or process != (81 if analysis else 96)):
        raise ValueError(
            f"initial-condition provenance labels forecast lead f{lead:03d} "
            f"as {kind!r} (process {process!r}); a forecast lead is not an "
            "analysis")

    cycle = _provenance_date(provenance.get("cycle"), "cycle")
    start = _provenance_date(
        provenance.get("model_start_time"), "model start time")
    if cycle + timedelta(hours=lead) != start:
        raise ValueError(
            f"initial-condition provenance does not compose: cycle "
            f"{cycle:%Y-%m-%d %H:%M:%S} + f{lead:03d} is not the declared "
            f"model start {start:%Y-%m-%d %H:%M:%S}")
    statement = provenance.get("statement")
    if not isinstance(statement, str) or not statement:
        raise ValueError(
            "initial-condition provenance carries no statement")

    return {
        "GPUWM_INITIAL_CONDITION_SCHEMA": WRFOUT_INITIAL_CONDITION_SCHEMA,
        "GPUWM_INITIAL_CONDITION_KIND": str(kind),
        # The driving model, in the spelling its own front door uses.
        # Unknown only for a caller that declared no source; the cycle is
        # not self-describing without it.
        "GPUWM_INITIAL_CONDITION_SOURCE": (
            "unknown" if source is None else str(source).upper()),
        "GPUWM_INITIAL_CONDITION_CYCLE": cycle.strftime(
            _WRF_GLOBAL_DATE_FORMAT),
        "GPUWM_INITIAL_FORECAST_LEAD_HOURS": np.int32(lead),
        "GPUWM_INITIAL_CONDITION_GENERATING_PROCESS_ID": np.int32(process),
        "GPUWM_INITIAL_CONDITION_MODEL_START_DATE": start.strftime(
            _WRF_GLOBAL_DATE_FORMAT),
        "GPUWM_INITIAL_CONDITION_STATEMENT": statement,
    }


#: The idealized cases' synthetic epoch.  Held here rather than formatted
#: inline because the month and year are no longer literals in the output:
#: they roll over.
_IDEALIZED_EPOCH = datetime(1, 1, 1)


def wrf_time_str(t_s: float) -> str:
    """WRF ``Times`` string for ``t_s`` seconds after the idealized epoch.

    Calendar arithmetic, not field arithmetic.  The previous form advanced
    only the day field and hard-coded the year and month, so a thirty-one
    day integration emitted ``0001-01-32`` and a year-long one emitted
    ``0001-01-366`` -- syntactically invalid ``Times`` strings that no WRF
    reader can parse, produced silently by a helper five idealized
    verification cases use.

    The epoch stays year 1: it is the frozen idealized-case contract, and
    moving it would move every fixture that depends on it.  Callers should
    know that year 1 is outside NumPy's ``datetime64[ns]`` range, so
    wrf-python -- which converts the parsed ``Times`` sequence to
    nanoseconds -- wraps such a date to somewhere in the 1700s.  That is a
    limitation of the synthetic epoch, not of the string: the string is now
    a real date either way, and the real-case path (which is what carries a
    wrf-python compatibility promise) has always used real dates.
    """
    seconds = int(round(t_s))
    if seconds < 0:
        raise ValueError(
            f"idealized Times seconds must be non-negative, got {t_s!r}")
    try:
        valid = _IDEALIZED_EPOCH + timedelta(seconds=seconds)
    except OverflowError as exc:
        raise ValueError(
            f"idealized Times seconds {t_s!r} overflows the calendar past "
            "year 9999") from exc
    return valid.strftime("%Y-%m-%d_%H:%M:%S")


def _live_state_history_fields(state) -> dict[str, object]:
    """Map live model arrays to their WRF Registry history names.

    The mapping is intentionally device/host agnostic.  Both frame builders
    consume it, preventing the asynchronous path from silently carrying a
    smaller scientific inventory than the synchronous writer.
    """
    fields: dict[str, object] = {}
    for output_name, state_name in (
            ("QICE", "qi"), ("QSNOW", "qs"), ("QGRAUP", "qg"),
            ("QNCLOUD", "nc"), ("QNRAIN", "nr"), ("QNICE", "ni"),
            ("QNSNOW", "ns"), ("QNGRAUPEL", "ng"),
            # Aerosol-aware Thompson's two transported aerosol scalars
            # (Registry/registry.new3d_wif:87/:89).  Presence-guarded like
            # every row above, and mp=28 is the only scheme that allocates
            # them, so no other run's inventory changes.  QNCLOUD is NOT a
            # new row here -- it has been mapped to state.nc since Morrison
            # landed; under mp=28 it simply starts carrying a PROGNOSTIC
            # droplet number instead of Morrison's diagnostic one, which is
            # a change in the values WRF also makes, not in the inventory.
            ("QNWFA", "nwfa"), ("QNIFA", "nifa")):
        value = getattr(state, state_name, None)
        if value is not None:
            fields[output_name] = value
    # The two 2-D surface aerosol emission rates.  Separate loop because
    # they are (ny, nx), not (nz, ny, nx): _dims_for routes them by shape,
    # and grouping them with the volume fields above would only obscure
    # that.  WRF's microphysics never writes either -- both are declared
    # OPTIONAL, INTENT(IN) on mp_gt_driver
    # (module_mp_thompson.F:1098) and are only READ, at :1247 and
    # :1320-1321.  So what a wrfout carries is thompson_init's derived
    # nwfa2d (:510) and the exactly-zero nifa2d nothing in
    # module_mp_thompson.F ever fills.
    for output_name, state_name in (
            ("QNWFA2D", "nwfa2d"), ("QNIFA2D", "nifa2d")):
        value = getattr(state, state_name, None)
        if value is not None:
            fields[output_name] = value
    for output_name, state_name in (
            ("QHAIL", "qh"), ("QNDROP", "qndrop"),
            ("QNRAIN", "qnr"), ("QNICE", "qni"),
            ("QNSNOW", "qns"), ("QNGRAUPEL", "qng"),
            ("QNHAIL", "qnh"), ("QNCCN", "qnn"),
            ("QVGRAUPEL", "qvolg"), ("QVHAIL", "qvolh")):
        value = getattr(state, state_name, None)
        if value is not None:
            fields[output_name] = value
    # The published subgrid energy, present only on a state whose PBL
    # closure owns one (SASE's prognostic e, or Shin-Hong's per-step TKE
    # diagnostic).  Scheme-qualified on purpose: WRF's ``TKE_PBL`` is a
    # Z-staggered MYJ/MYNN field on a different stagger, and the frame's
    # 2-D ``E`` is the Coriolis cosine term, not a turbulence quantity.
    # Named for the PRODUCER through the driver's own dispatch receipt,
    # so a Shin-Hong run can never publish its TKE under the SASE name;
    # a state without an attached driver keeps the historical SASE
    # label, which is the only producer such states ever had.
    e_sgs = getattr(state, "e_sgs", None)
    if e_sgs is not None:
        dispatch = getattr(getattr(state, "physics", None),
                           "scheme_dispatch", None)
        runner = (dispatch or {}).get("bl_pbl_physics")
        fields["TKE_SHINHONG" if runner == "_run_shinhong"
               else "TKE_SASE"] = e_sgs

    p_top = getattr(state, "p_top", None)
    if p_top is not None:
        fields["P_TOP"] = np.asarray(p_top, dtype=np.float32)
    for output_name, state_name in (("ZNU", "znu"), ("ZNW", "znw")):
        value = getattr(state, state_name, None)
        if value is not None:
            fields[output_name] = value

    # UP_HELI_MAX rides in every frame of a run that carries the
    # accumulator (allocated eagerly under nwp_diagnostics = 1), keeping
    # the async writer's frame schema constant.  The post-write reset is
    # the call sites' duty (gpuwm.core.uh_diag.reset_up_heli_max), never
    # this read-only builder's.
    existing_scratch = getattr(state, "existing_scratch", None)
    if existing_scratch is not None:
        up_heli_max = existing_scratch("up_heli_max")
        if up_heli_max is not None:
            fields["UP_HELI_MAX"] = up_heli_max

    physics = getattr(state, "physics", None)
    if physics is None:
        return fields
    microphysics = getattr(physics, "microphysics", None)
    if microphysics is not None:
        for output_name, field_name in (
                ("RAINNC", "rainnc"), ("SNOWNC", "snownc"),
                ("GRAUPELNC", "graupelnc"), ("HAILNC", "hailnc")):
            value = getattr(microphysics, field_name, None)
            if value is not None:
                fields[output_name] = value
    # Gate on "a land-surface scheme is routed", not on Noah's parameter
    # bundle: ``noah_params`` is scheme-2 state, so keying the snow/soil
    # history on it would silently drop TSLB/SMOIS/SH2O for any other LSM.
    # ``scheme_dispatch`` is the driver's own resolved routing, and
    # PhysicsDriver refuses to build when a selector value is unrouted.
    dispatch = getattr(physics, "scheme_dispatch", None)
    live_surface = getattr(physics, "fields", {})
    # MYNN's ten carried 3-D arrays and its four plume diagnostics exist only
    # under bl_pbl_physics=5, and wrfout does not auto-walk ``fields`` the way
    # the health collector does, so each emitted field is listed explicitly.
    # The gate is the driver's own resolved routing, for the same reason the
    # land-surface gate below is: a scheme that did not run must not appear to
    # have written state.  The listed keys are the scheme's *runtime* keys and
    # the emitted names come from the output schema, so no name here is
    # spelled twice and a key with no schema row raises rather than shipping
    # an anonymous float32.  ``exch_h``/``exch_m``/``rmol``/``kpbl`` are named
    # individually because they are shared EM_COMMON rows rather than members
    # of MYNN's own runtime inventories.
    if dispatch is not None:
        from gpuwm.core.physics import PHYSICS_SLOT_DISPATCH

        mynn_runner = PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][5]
        if dispatch.get("bl_pbl_physics") == mynn_runner:
            from gpuwm.core.mynn_pbl_runtime import (
                MYNN_PBL_DIAGNOSTICS_2D, MYNN_PBL_DIAGNOSTICS_INT_2D,
                MYNN_PBL_STATE_3D,
            )
            for field_name in (*MYNN_PBL_STATE_3D, *MYNN_PBL_DIAGNOSTICS_2D,
                               *MYNN_PBL_DIAGNOSTICS_INT_2D,
                               "exch_h", "exch_m", "rmol", "kpbl"):
                if field_name in live_surface:
                    fields[SCHEME_OUTPUT_FIELDS[field_name].netcdf_name] = \
                        live_surface[field_name]
    # Noah-MP's carried state and published diagnostics, on the same terms:
    # they exist only under sf_surface_physics=4, wrfout does not auto-walk
    # ``fields``, and the gate is the resolved routing.  The output names come
    # from the schema, which carries WRF's *external* names.  They used to be
    # the runtime keys upper-cased, which is not the same thing and was wrong
    # for every Noah-MP field but two: WRF writes ``TV``/``ISNOW``/``ZSNSO``,
    # never ``TVXY``/``ISNOWXY``/``ZSNSOXY``, so no WRF-name consumer could
    # find Noah-MP state in a gpuwm wrfout at all.
    if dispatch is not None:
        from gpuwm.core.physics import PHYSICS_SLOT_DISPATCH

        noahmp_runner = PHYSICS_SLOT_DISPATCH["sf_surface_physics"][4]
        if dispatch.get("sf_surface_physics") == noahmp_runner:
            from gpuwm.core.noahmp_runtime import (
                NOAHMP_DIAGNOSTICS_2D, NOAHMP_STATE_2D, NOAHMP_STATE_INT_2D,
                NOAHMP_STATE_SNOWSOIL_3D, NOAHMP_STATE_SNOW_3D,
            )
            for field_name in (*NOAHMP_STATE_2D, *NOAHMP_STATE_INT_2D,
                               *NOAHMP_STATE_SNOW_3D,
                               *NOAHMP_STATE_SNOWSOIL_3D,
                               *NOAHMP_DIAGNOSTICS_2D):
                if field_name in live_surface:
                    fields[SCHEME_OUTPUT_FIELDS[field_name].netcdf_name] = \
                        live_surface[field_name]
    # RUC's carried state and its four published driver locals, on the same
    # terms: they exist only under sf_surface_physics=3, wrfout does not
    # auto-walk ``fields``, and the gate is the resolved routing.  RUC's
    # external names happen to be its symbols upper-cased, but they are taken
    # from the schema anyway so that the coincidence is not load-bearing; the
    # four ruc_* driver locals have no Registry counterpart and keep their
    # prefix so nothing mistakes them for WRF output.
    if dispatch is not None:
        from gpuwm.core.physics import PHYSICS_SLOT_DISPATCH

        ruc_runner = PHYSICS_SLOT_DISPATCH["sf_surface_physics"][3]
        if dispatch.get("sf_surface_physics") == ruc_runner:
            from gpuwm.core.ruc_runtime import (
                RUC_DIAGNOSTICS_2D, RUC_STATE_2D, RUC_STATE_3D,
            )
            for field_name in (*RUC_STATE_2D, *RUC_STATE_3D,
                               *RUC_DIAGNOSTICS_2D):
                if field_name in live_surface:
                    fields[SCHEME_OUTPUT_FIELDS[field_name].netcdf_name] = \
                        live_surface[field_name]
    if dispatch is not None:
        land_surface_active = dispatch.get("sf_surface_physics") is not None
    else:
        land_surface_active = getattr(physics, "noah_params", None) is not None
    if not land_surface_active:
        return fields
    for output_name, field_name in (
            ("SNOW", "snow"), ("SNOWH", "snowh"),
            ("SNOWC", "snowc"), ("TSLB", "tslb"),
            ("SMOIS", "smois"), ("SH2O", "sh2o")):
        if field_name in live_surface:
            fields[output_name] = live_surface[field_name]
    return fields


def state_frame(
        state, *, include_diagnostic_pressure: bool = False
) -> dict[str, np.ndarray]:
    """Standard wrfout frame from a ``DomainState``, host float32 arrays.

    Winds ``U/V/W``, ``T`` (theta - 300, the WRF perturbation convention),
    the ``PH``/``MU`` perturbations, and the terrain-consistent base
    fields: per-column 3-D ``PHB`` (a flat base state's 1-D column is
    broadcast so the on-disk layout is identical with and without
    terrain), ``MUB``, and the terrain height ``HGT``.  States carrying
    moisture add ``QVAPOR``/``QCLOUD``/``QRAIN`` and any live Morrison
    mass/number moments.  P_TOP/ZNU/ZNW plus attached precipitation, snow,
    and Noah history arrays are also carried.  The frozen idealized-case
    contract omits diagnostic pressure fields; callers that require WRF
    tooling compatibility can opt in to ``P``/``PB``/``PSFC`` with
    ``include_diagnostic_pressure=True`` (the state's pressure must be
    diagnosed -- run ``update_diagnostics`` first on a fresh init).
    """
    import cupy as cp  # deferred: the writer itself stays CPU-importable

    ny, nx = state.mup.shape
    phb = cp.asnumpy(state.phb)
    if phb.ndim == 1:                     # flat base state: broadcast column
        phb = np.ascontiguousarray(
            np.broadcast_to(phb[:, None, None], (phb.size, ny, nx)))
    fields = {
        "T": cp.asnumpy(state.total_theta()) - np.float32(300.0),
        "U": cp.asnumpy(state.u),
        "V": cp.asnumpy(state.v),
        "W": cp.asnumpy(state.w),
        "PH": cp.asnumpy(state.php),
        "MU": cp.asnumpy(state.mup),
        "PHB": phb,
        "MUB": cp.asnumpy(state.mub2d),
        "HGT": cp.asnumpy(state.ht),
    }
    if include_diagnostic_pressure:
        pb = state.pb
        pb3 = pb if pb.ndim == 3 else pb[:, None, None]
        fields["P"] = cp.asnumpy(state.p - pb3)
        fields["PB"] = cp.asnumpy(cp.broadcast_to(pb3, state.p.shape))
        if getattr(state, "physics", None) is not None:
            fields["PSFC"] = cp.asnumpy(state.physics.fields["psfc"])
        elif getattr(state, "p_top", None) is not None:
            # Without a physics driver, diagnose PSFC the way WRF's
            # phy_prep extrapolates the full (moist) pressure to the
            # surface in z (module_big_step_utilities_em.F:5566-5578).
            # The previous dry form (total_mu + p_top) understates a
            # moist column's PSFC by the column water weight.
            from gpuwm.core import constants as c
            phb3 = (state.phb[:, None, None]
                    if state.phb.ndim == 1 else state.phb)
            z_if = (phb3 + state.php) / np.float32(c.G)
            z_mid = 0.5 * (z_if[:-1] + z_if[1:])
            w1 = (z_if[0] - z_mid[1]) / (z_mid[0] - z_mid[1])
            fields["PSFC"] = cp.asnumpy(
                w1 * state.p[0] + (1.0 - w1) * state.p[1])
    if state.qv is not None:
        fields["QVAPOR"] = cp.asnumpy(state.qv)
        fields["QCLOUD"] = cp.asnumpy(state.qc)
        fields["QRAIN"] = cp.asnumpy(state.qr)
    for name, array in _live_state_history_fields(state).items():
        if isinstance(array, np.ndarray):
            fields[name] = np.array(array, copy=True, order="C")
        else:
            fields[name] = cp.asnumpy(array)
    if getattr(state, "physics", None) is not None:
        fields.update({name: cp.asnumpy(array)
                       for name, array in state.physics.output_fields().items()})
    return fields


def validate_wrfout_file(path, *, inventory, shapes, times):
    """Reopen and prove inventory/shapes/Times/completion before publish."""
    path = Path(path)
    with netCDF4.Dataset(path, "r") as ds:
        if int(getattr(ds, _COMPLETION_ATTR, 0)) != 1:
            raise ValueError(f"wrfout {path} has no completion attribute")
        actual = set(ds.variables)
        if actual != set(inventory):
            raise ValueError(
                f"wrfout {path} inventory mismatch: expected "
                f"{sorted(inventory)}, got {sorted(actual)}")
        for name, expected in shapes.items():
            actual_shape = tuple(ds.variables[name].shape)
            if actual_shape != tuple(expected):
                raise ValueError(
                    f"wrfout {path} variable {name} shape {actual_shape} "
                    f"!= completed shape {tuple(expected)}")
        raw_times = ds.variables["Times"][:]
        actual_times = tuple(
            row.tobytes().decode("ascii").rstrip("\x00")
            for row in raw_times)
        if actual_times != tuple(times):
            raise ValueError(
                f"wrfout {path} Times mismatch: expected {tuple(times)}, "
                f"got {actual_times}")


def quarantine_orphan_wrfouts(directory):
    """Move orphan temporaries and incomplete final files out of sight."""
    directory = Path(directory)
    if not directory.exists():
        return tuple()
    moved = []
    temporaries = set(directory.glob(".wrfout*.tmp*"))
    temporaries.update(directory.glob("wrfout*.tmp*"))
    for path in sorted(temporaries):
        target = quarantine_file(path, reason="orphan-wrfout-tmp")
        if target is not None:
            moved.append(target)
    for path in sorted(directory.glob("wrfout*")):
        if not path.is_file() or ".tmp" in path.name:
            continue
        try:
            with netCDF4.Dataset(path, "r") as ds:
                complete = int(getattr(ds, _COMPLETION_ATTR, 0)) == 1
        except Exception:
            complete = False
        if not complete:
            target = quarantine_file(path, reason="incomplete-wrfout")
            if target is not None:
                moved.append(target)
    return tuple(moved)


class WrfoutWriter:
    def __init__(self, path, *, nx, ny, nz, dx, dy, title="gpuwm",
                 global_attrs=None, field_schema=None, soil_layers=None):
        self.nx, self.ny, self.nz = nx, ny, nz
        # WRF's soil axis length is the land-surface scheme's own geometry
        # (Noah/Noah-MP 4, RUC 6 or 9), not a universal constant, so any
        # caller with a RunConfig must pass soil_layer_count(cfg).  ``None``
        # means the caller declared no land-surface scheme -- the idealized
        # cases, which write no soil field on this axis -- and takes the
        # schema's no-LSM length rather than a literal, so this signature
        # carries no soil-geometry constant of its own.  It used to default to
        # 4, and three real callers (gpuwm/runtime.py, offline_child_run.py,
        # offline_child_smoke.py) were silently taking it.
        self.soil_layers = (NO_LAND_SURFACE_SOIL_LAYERS
                            if soil_layers is None else int(soil_layers))
        self._final_path = Path(path)
        self._final_path.parent.mkdir(parents=True, exist_ok=True)
        self._temp_path = unique_temp_path(self._final_path, hidden=True)
        self.ds = netCDF4.Dataset(
            self._temp_path, "w", format="NETCDF4_CLASSIC")
        ds = self.ds
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("west_east", nx)
        ds.createDimension("south_north", ny)
        ds.createDimension("bottom_top", nz)
        ds.createDimension("west_east_stag", nx + 1)
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("bottom_top_stag", nz + 1)
        # The land-surface scheme's soil layers live on their own fully
        # dimensioned staggered vertical axis, not bottom_top.
        ds.createDimension("soil_layers_stag", self.soil_layers)
        ds.setncatts({
            # NC_FLOAT, not the NC_DOUBLE a Python float would become: stock
            # WRF writes DX/DY single-precision like every other projection
            # global (see wrf_global_attrs).
            "TITLE": title,
            # TITLE is the caller's configured output title, so it is not a
            # provenance seal: a wrfout separated from its logs could not say
            # which build wrote it.  GPUWM_VERSION can, and it reads the
            # installed distribution's metadata rather than a constant.
            "GPUWM_VERSION": _producer_version(),
            "DX": np.float32(dx), "DY": np.float32(dy),
            "WEST-EAST_GRID_DIMENSION": nx + 1,
            "SOUTH-NORTH_GRID_DIMENSION": ny + 1,
            "BOTTOM-TOP_GRID_DIMENSION": nz + 1,
            "MAP_PROJ": 0,
        })
        if global_attrs:
            ds.setncatts(dict(global_attrs))
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        self._n = 0
        self._times = []
        self._closed = False
        self._declared_fields: frozenset[str] | None = None
        if field_schema is not None:
            schema = dict(field_schema)
            if hasattr(ds, "START_DATE") and hasattr(ds, "DT"):
                schema.setdefault("XTIME", np.asarray(0.0, dtype=np.float32))
                schema.setdefault(
                    "ITIMESTEP", np.asarray(0, dtype=np.int32))
            for name, value in schema.items():
                shape = self._wrf_shape(name, getattr(value, "shape", value))
                self._create_variable(name, self._dims_for(name, shape))
            self._declared_fields = frozenset(schema)

    def _ensure_dimension(self, name, size):
        """Create a dimension the first time a field needs it.

        Created lazily rather than in ``__init__`` so a file from a run
        without that scheme keeps byte-identical headers; the frozen wrfout
        contract is a gate, and silently adding a dimension to every file
        would move it.
        """
        existing = self.ds.dimensions.get(name)
        if existing is None:
            self.ds.createDimension(name, int(size))
            return
        if len(existing) != int(size):
            raise ValueError(
                f"wrfout dimension {name} is {len(existing)}, not {size}")

    def _wrf_shape(self, name, shape):
        """On-disk shape for ``name``: WRF's, which is not always gpuwm's."""
        shape = tuple(shape)
        if name in _Z_STAGGERED_MASS_FIELDS and shape == (
                self.nz, self.ny, self.nx):
            return (self.nz + 1, self.ny, self.nx)
        return shape

    @staticmethod
    def _check_integer_field_dtype(name, array):
        """A field WRF declares integer must arrive as one.

        netCDF4 casts on assignment, so handing a float array to an ``i4``
        variable truncates every value in silence -- which would restore
        the exact defect the schema exists to end, one layer further in.
        Only the integer direction is checked: a real field legitimately
        accepts any float width.
        """
        schema = HISTORY_FIELDS_BY_NETCDF_NAME.get(name)
        if schema is None or schema.dtype != "i4":
            return
        if array.dtype.kind not in ("i", "u"):
            raise ValueError(
                f"wrfout variable {name} is declared integer by WRF v4.6.1 "
                f"({schema.registry}) but the frame supplied "
                f"{array.dtype}; writing it would truncate every value")

    def _wrf_array(self, name, array):
        """``array`` on its WRF axis, zero-padding the unfilled top level."""
        self._check_integer_field_dtype(name, array)
        if (name not in _Z_STAGGERED_MASS_FIELDS
                or array.shape != (self.nz, self.ny, self.nx)):
            return array
        lifted = np.zeros((self.nz + 1, self.ny, self.nx), dtype=array.dtype)
        lifted[:self.nz] = array
        return lifted

    def _dims_for(self, name, shape):
        nz, ny, nx = self.nz, self.ny, self.nx
        if name in _SNOW_LAYER_FIELDS or name in _SNSO_LAYER_FIELDS:
            snow_layers = int(tuple(shape)[0]) if name in _SNOW_LAYER_FIELDS \
                else int(tuple(shape)[0]) - self.soil_layers
            expected = ((snow_layers, ny, nx) if name in _SNOW_LAYER_FIELDS
                        else (snow_layers + self.soil_layers, ny, nx))
            if tuple(shape) != expected or snow_layers < 1:
                raise ValueError(
                    f"Noah-MP snow history field {name} has shape "
                    f"{tuple(shape)}, which is not a snow stack over "
                    f"{(ny, nx)}")
            if name in _SNOW_LAYER_FIELDS:
                self._ensure_dimension("snow_layers_stag", snow_layers)
                axis = "snow_layers_stag"
            else:
                self._ensure_dimension(
                    "snso_layers_stag", snow_layers + self.soil_layers)
                axis = "snso_layers_stag"
            return ("Time", axis, "south_north", "west_east")
        if name in _SOIL_LAYER_FIELDS:
            expected = (self.soil_layers, ny, nx)
            if tuple(shape) != expected:
                raise ValueError(
                    f"WRF soil history field {name} must have shape "
                    f"{expected}, got {tuple(shape)}")
            return ("Time", "soil_layers_stag", "south_north", "west_east")
        table = {
            (): (),
            (nz,): ("bottom_top",),
            (nz + 1,): ("bottom_top_stag",),
            (nz, ny, nx): ("bottom_top", "south_north", "west_east"),
            (nz, ny, nx + 1): ("bottom_top", "south_north", "west_east_stag"),
            (nz, ny + 1, nx): ("bottom_top", "south_north_stag", "west_east"),
            (nz + 1, ny, nx): ("bottom_top_stag", "south_north", "west_east"),
            (ny, nx): ("south_north", "west_east"),
            (ny, nx + 1): ("south_north", "west_east_stag"),
            (ny + 1, nx): ("south_north_stag", "west_east"),
        }
        return ("Time",) + table[tuple(shape)]

    def _create_variable(self, name, dims):
        """Variable with WRF type, memory order, Registry, and grid attrs.

        A scheme field's type, ``FieldType``, ``description``, ``units`` and
        staggering all come from its transcribed WRF v4.6.1 Registry row
        rather than from the writer's own assumptions, and the row's stagger
        is checked against the one the dimension table implies.  Those two
        can only disagree if the dimension table routed the field onto the
        wrong axis, which is a schema lie a reader cannot detect -- so it is
        refused here instead of published.
        """
        schema = HISTORY_FIELDS_BY_NETCDF_NAME.get(name)
        if schema is not None:
            dtype, field_type = schema.dtype, schema.field_type
            desc, units = schema.description, schema.units
        else:
            dtype = "i4" if name == "ITIMESTEP" else "f4"
            field_type = (WRF_FIELD_TYPE_INTEGER if name == "ITIMESTEP"
                          else WRF_FIELD_TYPE_REAL)
            desc, units = _VAR_META.get(name, ("", ""))
        stagger = next((s for d, s in _STAGGER.items() if d in dims), "")
        if schema is not None and schema.stagger != stagger:
            raise ValueError(
                f"wrfout variable {name} lands on dimensions {dims}, whose "
                f"stagger is {stagger!r}, but WRF v4.6.1 declares it "
                f"{schema.stagger!r} ({schema.registry})")
        var = self.ds.createVariable(name, dtype, dims)
        if name == "XTIME":
            origin = str(getattr(self.ds, "START_DATE", "")).replace(
                "_", " ", 1)
            desc = units = f"minutes since {origin}"
        var.setncattr("FieldType", np.int32(field_type))
        if dims == ("Time",):
            var.MemoryOrder = "0  "
        elif len(dims) == 2:
            var.MemoryOrder = "Z  "
        else:
            var.MemoryOrder = "XYZ" if len(dims) == 4 else "XY "
        var.description = desc
        var.units = units
        var.stagger = stagger
        spatial = any(d in dims for d in (
            "west_east", "west_east_stag", "south_north",
            "south_north_stag"))
        if spatial:
            time_link = (" XTIME" if hasattr(self.ds, "START_DATE")
                         and hasattr(self.ds, "DT") else "")
            if name in ("XLAT_U", "XLONG_U"):
                var.coordinates = "XLONG_U XLAT_U"
            elif name in ("XLAT_V", "XLONG_V"):
                var.coordinates = "XLONG_V XLAT_V"
            elif name in ("XLAT", "XLONG"):
                var.coordinates = "XLONG XLAT"
            elif "west_east_stag" in dims:
                var.coordinates = "XLONG_U XLAT_U" + time_link
            elif "south_north_stag" in dims:
                var.coordinates = "XLONG_V XLAT_V" + time_link
            else:
                var.coordinates = "XLONG XLAT" + time_link
        return var

    def _time_coordinate_fields(self, time_str: str) -> dict[str, np.ndarray]:
        """WRF XTIME/ITIMESTEP values derived from bound start time and DT."""
        if not hasattr(self.ds, "START_DATE") or not hasattr(self.ds, "DT"):
            return {}
        try:
            valid = datetime.strptime(time_str, "%Y-%m-%d_%H:%M:%S")
            start = datetime.strptime(
                str(self.ds.START_DATE), "%Y-%m-%d_%H:%M:%S")
        except ValueError as exc:
            raise ValueError(
                "WRF time coordinates require START_DATE and frame time in "
                "YYYY-MM-DD_HH:MM:SS format") from exc
        elapsed_seconds = (valid - start).total_seconds()
        if elapsed_seconds < 0.0:
            raise ValueError(
                f"wrfout valid time {time_str} precedes START_DATE "
                f"{self.ds.START_DATE}")
        dt = float(self.ds.DT)
        return {
            "XTIME": np.asarray(elapsed_seconds / 60.0, dtype=np.float32),
            "ITIMESTEP": np.asarray(
                int(round(elapsed_seconds / dt)), dtype=np.int32),
        }

    def write_frame(self, time_str: str, fields: dict[str, np.ndarray]):
        t = self._n
        # netCDF4 1.7.x stringtochar mangles S-dtype input under numpy 2,
        # so build the 19-char record directly (null-pad).  A longer value
        # is refused rather than truncated: the only way to exceed 19 is a
        # sub-second instant, and silently dropping the fraction is how two
        # distinct frames came to describe the same second.
        encoded = time_str.encode("ascii")
        if len(encoded) > 19:
            raise ValueError(
                f"wrfout Times record {time_str!r} exceeds the 19-character "
                "WRF format; sub-second history instants are not supported "
                "and must not be truncated into an aliased one")
        buf = encoded.ljust(19, b"\x00")
        self.ds.variables["Times"][t] = np.frombuffer(buf, dtype="S1")
        frame = dict(fields)
        for name, value in self._time_coordinate_fields(time_str).items():
            frame.setdefault(name, value)
        if (self._declared_fields is not None
                and set(frame) != set(self._declared_fields)):
            missing = sorted(set(self._declared_fields) - set(frame))
            extra = sorted(set(frame) - set(self._declared_fields))
            raise ValueError(
                "wrfout frame does not match the creation-time field "
                f"schema: missing={missing}, extra={extra}")
        for name, arr in frame.items():
            arr = self._wrf_array(name, np.asarray(arr))
            if name not in self.ds.variables:
                self._create_variable(name, self._dims_for(name, arr.shape))
            self.ds.variables[name][t] = arr
        self._times.append(buf.rstrip(b"\x00").decode("ascii"))
        self._n += 1

    def close(self):
        if self._closed:
            return
        inventory = tuple(self.ds.variables)
        shapes = {name: tuple(variable.shape)
                  for name, variable in self.ds.variables.items()}
        times = tuple(self._times)
        try:
            self.ds.setncattr(_COMPLETION_ATTR, np.int32(1))
            self.ds.close()
            self._closed = True
            fsync_file(self._temp_path)
            validate_wrfout_file(
                self._temp_path, inventory=inventory, shapes=shapes,
                times=times)
            # Sharing violations receive the same capped 0.50 s retry as the
            # heartbeat, but a durable wrfout publication remains fail-loud.
            replace_file_with_retry(self._temp_path, self._final_path)
        except BaseException:
            if not self._closed:
                # A half-closed netCDF handle can fail repeatedly.  Preserve
                # the original publication error and still reach quarantine.
                with suppress(BaseException):
                    self.ds.close()
                self._closed = True
            if self._temp_path.exists():
                with suppress(BaseException):
                    quarantine_file(
                        self._temp_path,
                        reason="failed-wrfout-publication")
            raise

    def abort(self):
        """Close without completion and quarantine a failed write."""
        if self._closed:
            return
        with suppress(BaseException):
            self.ds.close()
        self._closed = True
        if self._temp_path.exists():
            with suppress(BaseException):
                quarantine_file(self._temp_path, reason="aborted-wrfout")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False


def _device_state_frame(state, *, include_diagnostic_pressure: bool = True):
    """Build the standard frame as live device arrays for side-stream D2H."""
    import cupy as cp

    ny, nx = state.mup.shape
    phb = state.phb
    if phb.ndim == 1:
        phb = cp.broadcast_to(phb[:, None, None], (phb.size, ny, nx))
    fields = {
        "T": state.total_theta() - cp.float32(300.0),
        "U": state.u, "V": state.v, "W": state.w,
        "PH": state.php, "MU": state.mup,
        "PHB": phb, "MUB": state.mub2d, "HGT": state.ht,
    }
    if include_diagnostic_pressure:
        pb = state.pb
        pb3 = pb if pb.ndim == 3 else pb[:, None, None]
        fields["P"] = state.p - pb3
        fields["PB"] = cp.broadcast_to(pb3, state.p.shape)
        if getattr(state, "physics", None) is not None:
            fields["PSFC"] = state.physics.fields["psfc"]
        elif getattr(state, "p_top", None) is not None:
            from gpuwm.core import constants as c
            phb3 = (state.phb[:, None, None]
                    if state.phb.ndim == 1 else state.phb)
            z_if = (phb3 + state.php) / cp.float32(c.G)
            z_mid = 0.5 * (z_if[:-1] + z_if[1:])
            w1 = (z_if[0] - z_mid[1]) / (z_mid[0] - z_mid[1])
            fields["PSFC"] = w1 * state.p[0] + (1.0 - w1) * state.p[1]
    if state.qv is not None:
        fields.update(QVAPOR=state.qv, QCLOUD=state.qc, QRAIN=state.qr)
    fields.update(_live_state_history_fields(state))
    if getattr(state, "physics", None) is not None:
        fields.update(state.physics.output_fields())
    return fields


@dataclass
class _AsyncFrame:
    path: Path
    time_str: str
    fields: dict[str, np.ndarray]
    event: object
    device_refs: tuple[object, ...]
    pinned_refs: tuple[object, ...]
    admitted: bool = False
    #: The submitting datetime, carried so a landing observer receives
    #: the valid time as a value rather than re-parsing ``time_str``
    #: (or the filename) on the writer thread.
    valid_time: object = None


class _AsyncTicketQueue(queue.Queue):
    """Bounded queue that records ticket admission under its mutex."""

    def _put(self, item) -> None:
        if item is None:
            super()._put(item)
            return

        # Queue.put holds the queue mutex while calling _put.  Mark first so
        # there is no state in which a worker can take this ticket while its
        # producer still considers it unadmitted.  If insertion unwinds before
        # appending, restore the marker while the same mutex is still held.
        try:
            item.admitted = True
            super()._put(item)
        except BaseException:
            if not any(queued is item for queued in self.queue):
                item.admitted = False
            raise


class AsyncDomainWrfoutWriter:
    """One domain's side-stream D2H staging and dedicated writer thread."""

    #: Which domain this writer is, and who to tell when one of its
    #: frames becomes durable.  CLASS attributes, not only instance
    #: ones: this writer is also stood up field-by-field through
    #: ``object.__new__`` by CPU-only harnesses that cannot allocate a
    #: CuPy stream (tests/test_wrfout.py builds exactly that shell), and
    #: an optional feature that only exists on the ``__init__`` path
    #: would turn every such construction into an AttributeError on the
    #: worker thread.  Declaring the default here is also where a reader
    #: looks to learn the feature is optional.
    grid_id = None
    landing_observer = None

    @staticmethod
    def _new_ticket_queue() -> queue.Queue:
        # Capacity is per domain: one queued ticket in each of four queues is
        # one complete four-domain history burst.  Each worker may additionally
        # hold its current handoff, bounding queued + worker-held tickets at
        # eight across a synchronized four-domain write boundary.
        return _AsyncTicketQueue(maxsize=1)

    def __init__(self, *, nx, ny, nz, dx, dy, title, global_attrs,
                 abort_event=None, soil_layers=None, grid_id=None,
                 landing_observer=None):
        import cupy as cp

        #: Which domain this writer is, and who to tell when one of its
        #: frames becomes durable.  Both optional: every existing caller
        #: constructs this writer without them and is unaffected.  The
        #: observer is invoked ON THE WRITER THREAD, after the file has
        #: been fsynced, self-validated and renamed onto its final name
        #: -- so it is told about a file that exists, never one that is
        #: merely queued.
        self.grid_id = None if grid_id is None else int(grid_id)
        self.landing_observer = landing_observer
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        # Same contract as WrfoutWriter: the only production caller
        # (open_domain_writer below) resolves this from the domain's cfg.
        self.soil_layers = (NO_LAND_SURFACE_SOIL_LAYERS
                            if soil_layers is None else int(soil_layers))
        self.dx, self.dy = float(dx), float(dy)
        self.title = title
        self.global_attrs = dict(global_attrs)
        self.stream = cp.cuda.Stream(non_blocking=True)
        self._queue = self._new_ticket_queue()
        self._condition = threading.Condition()
        self._pending = 0
        self._failure: BaseException | None = None
        self._failure_traceback: str | None = None
        self._closed = False
        self._abort_event = (threading.Event() if abort_event is None
                             else abort_event)
        self.paths: list[Path] = []
        self._thread = threading.Thread(
            target=self._worker, name=f"gpuwm-wrfout-{id(self):x}",
            daemon=True)
        self._thread.start()

    @property
    def pending(self) -> int:
        with self._condition:
            return self._pending

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError("per-domain wrfout writer failed") \
                from self._failure

    def _check_admission_liveness(self) -> None:
        """Reject work that no live, healthy worker can consume."""
        self._raise_failure()
        if not self._thread.is_alive():
            # Recheck after observing thread death so a just-published worker
            # exception remains the cause reported to the producer.
            self._raise_failure()
            raise RuntimeError("per-domain wrfout worker stopped unexpectedly")
        if self._abort_event.is_set():
            self._raise_failure()
            raise RuntimeError("per-domain wrfout writing was aborted")

    def _admit(self, ticket: _AsyncFrame) -> None:
        """Admit one staged frame with bounded, liveness-aware waits."""
        with self._condition:
            self._pending += 1
        try:
            while True:
                self._check_admission_liveness()
                try:
                    self._queue.put(
                        ticket, timeout=_ASYNC_WRITER_POLL_SECONDS)
                except queue.Full:
                    continue
                return
        finally:
            # Queue.put may unwind for cancellation, KeyboardInterrupt, or a
            # queue implementation failure.  Admission truth is written by
            # _AsyncTicketQueue._put under the queue mutex, not inferred from
            # whether this producer observed Queue.put return.
            if not ticket.admitted:
                with self._condition:
                    self._pending -= 1
                    self._condition.notify_all()

    def submit(self, path, valid_time, state, *, extra_fields=None,
               refl_field=None) -> None:
        """Queue a nonblocking, stream-ordered snapshot of one domain."""
        import cupy as cp

        if self._closed:
            raise RuntimeError("cannot submit to a closed wrfout writer")
        self._raise_failure()
        producer = cp.cuda.get_current_stream()
        device_fields = _device_state_frame(
            state, include_diagnostic_pressure=True)
        if refl_field is not None:
            device_fields["REFL_10CM"] = refl_field
        ready = cp.cuda.Event()
        ready.record(producer)
        self.stream.wait_event(ready)

        host_fields: dict[str, np.ndarray] = {}
        device_refs: list[object] = []
        pinned_refs: list[object] = []
        with self.stream:
            for name, value in device_fields.items():
                if isinstance(value, np.ndarray):
                    # Already on host: staging it through the device would
                    # be a pure bounce (measured ~1.8 GiB of cached device
                    # staging across the initial frames).  np.array preserves
                    # a scalar P_TOP's 0-D shape; np.ascontiguousarray would
                    # silently promote it to (1,) and give it a vertical dim.
                    host_fields[name] = np.array(
                        value, copy=True, order="C", subok=False)
                    continue
                array = cp.ascontiguousarray(value)
                memory = cp.cuda.alloc_pinned_memory(int(array.nbytes))
                host = np.frombuffer(memory, dtype=array.dtype,
                                     count=array.size).reshape(array.shape)
                array.get(out=host, stream=self.stream, blocking=False)
                host_fields[name] = host
                device_refs.append(array)
                pinned_refs.append(memory)
            for name, value in (extra_fields or {}).items():
                # Prognostic/state-derived fields win (notably child HGT,
                # which is blended while static HGT_M remains unblended).
                if name not in host_fields:
                    host_fields[name] = np.ascontiguousarray(value)
            done = cp.cuda.Event()
            done.record(self.stream)
        # The next mutation on the producing stream waits for the snapshot,
        # while the host remains free to write another domain/file.
        producer.wait_event(done)
        ticket = _AsyncFrame(
            path=Path(path),
            time_str=valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
            fields=host_fields, event=done,
            device_refs=tuple(device_refs),
            pinned_refs=tuple(pinned_refs),
            valid_time=valid_time)
        self._admit(ticket)

    def _worker(self) -> None:
        while True:
            ticket = self._queue.get()
            if ticket is None:
                self._queue.task_done()
                return
            try:
                ticket.event.synchronize()
                # D2H is complete.  Drop device ownership on the side stream
                # that consumed the arrays, while retaining pinned backing
                # until NetCDF has finished reading the host field views.
                with self.stream:
                    ticket.device_refs = ()
                with _NETCDF4_IO_LOCK:
                    if self._abort_event.is_set():
                        continue
                    try:
                        with WrfoutWriter(
                                ticket.path, nx=self.nx, ny=self.ny, nz=self.nz,
                                dx=self.dx, dy=self.dy, title=self.title,
                                global_attrs=self.global_attrs,
                                soil_layers=self.soil_layers,
                                field_schema=ticket.fields) as writer:
                            writer.write_frame(ticket.time_str, ticket.fields)
                    except BaseException:
                        # Set the experiment-wide abort while still holding
                        # the netCDF lock.  A later domain cannot slip into a
                        # file session between this failure and publication
                        # cancellation.
                        self._abort_event.set()
                        raise
                self.paths.append(ticket.path)
                # The file is durable HERE and nowhere earlier: the
                # WrfoutWriter context above has exited, so its close()
                # completed the fsync, the self-validation and the
                # rename onto the final name.  A failure at any of those
                # went to the handler below instead.  An observer that
                # raises must not take the writer thread down with it --
                # the run's outputs matter more than its telemetry.
                if self.landing_observer is not None:
                    try:
                        self.landing_observer(
                            domain=self.grid_id, valid_time=ticket.valid_time,
                            path=ticket.path)
                    except Exception:  # noqa: BLE001 - telemetry never fails a run
                        pass
            except BaseException as exc:
                failure_traceback = _stringify_and_clear_exception_tracebacks(
                    exc)
                with self._condition:
                    if self._failure is None:
                        self._failure = exc
                        self._failure_traceback = failure_traceback
                self._abort_event.set()
            finally:
                ticket.pinned_refs = ()
                self._queue.task_done()
                # Clear before publishing pending == 0 and before the next
                # blocking queue.get(), so an idle worker cannot retain the
                # completed frame or any of its arrays.
                ticket = None
                with self._condition:
                    self._pending -= 1
                    self._condition.notify_all()

    def update_global_attrs(self, global_attrs) -> None:
        """Swap the per-file global attributes for frames submitted later.

        A relocated domain's placement, centre and corner attributes
        change at the move; frames produced before it must keep the
        attributes of the placement that produced them, so this drains
        the queue first and then swaps.  The worker reads
        ``self.global_attrs`` per file, so the swap is complete for every
        subsequent submit.
        """
        self.drain()
        self.global_attrs = dict(global_attrs)

    def drain(self) -> None:
        worker_stopped = False
        with self._condition:
            while self._pending:
                if not self._thread.is_alive():
                    worker_stopped = True
                    break
                self._condition.wait(timeout=_ASYNC_WRITER_POLL_SECONDS)
        if worker_stopped:
            self._raise_failure()
            raise RuntimeError(
                "per-domain wrfout worker stopped with pending tickets")
        self._raise_failure()

    def close(self) -> None:
        if self._closed:
            self._raise_failure()
            return
        saved: BaseException | None = None
        try:
            self.drain()
        except BaseException as exc:
            saved = exc
        finally:
            # Sentinel and join belong in finally: no live daemon may outlive
            # a failed close or keep publishing after unwind.  A dead worker
            # cannot consume a sentinel, and its queue may still be full.
            self._closed = True
            while self._thread.is_alive():
                try:
                    self._queue.put(
                        None, timeout=_ASYNC_WRITER_POLL_SECONDS)
                except queue.Full:
                    continue
                break
            self._thread.join()
        if saved is not None:
            raise saved
        self._raise_failure()


class PerDomainWrfoutWriters:
    """One asynchronous writer/side stream per domain in an experiment."""

    def __init__(self, model, output_dir, *, start_time, title,
                 initial_condition=None, source=None,
                 progress_callback=None):
        """``initial_condition`` is the preparation receipt's provenance
        block, stamped onto every domain's frames so the durable artifact
        states what its initial state was and not only when it began.

        ``progress_callback`` is the run's existing progress object.  If
        it carries an ``output_committed`` attribute (the optional hook
        :func:`gpuwm.runtime._output_committed` discovers by name), each
        per-domain writer is given it and calls it as its frames become
        durable; anything else is ignored, so every existing caller is
        unaffected by passing one or by not passing one.

        A caller whose progress object does not exist yet at
        construction time uses :meth:`attach_progress_callback` instead;
        both spellings run the same one implementation.
        """
        from gpuwm.runtime import _global_wrf_attrs, _metadata_frame

        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = start_time
        self.title = title
        # Retained for refresh_domain: a relocated domain's later frames are
        # re-stamped with the same provenance the original writer carried.
        self._initial_condition = initial_condition
        self._source = source
        prepared = model._prepared_by_grid_id
        self._writers = {}
        self._metadata_by_grid_id = {}
        self._abort_event = threading.Event()
        for node in model.walk_parent_first():
            case = prepared[node.cfg.grid_id]
            configured_start = getattr(node.cfg, "start_time", None)
            domain_start_time = (
                start_time if configured_start is None
                else configured_start)
            self._metadata_by_grid_id[node.cfg.grid_id] = _metadata_frame(
                node.grid, case.static_fields)
            self._writers[node.cfg.grid_id] = AsyncDomainWrfoutWriter(
                nx=node.cfg.run.nx, ny=node.cfg.run.ny, nz=node.cfg.run.nz,
                dx=node.cfg.run.dx, dy=node.cfg.run.dy, title=title,
                soil_layers=soil_layer_count(node.cfg.run),
                global_attrs=_global_wrf_attrs(
                    node.grid, domain_start_time,
                    getattr(case, "geog_selection", None),
                    domain=node.cfg, coord=case.initial_result.coord,
                    feedback=getattr(model, "_feedback_provenance", None),
                    initial_condition=initial_condition, source=source),
                abort_event=self._abort_event,
                grid_id=node.cfg.grid_id)
        self.last_durable_wrfout = None
        if progress_callback is not None:
            self.attach_progress_callback(progress_callback)

    def attach_progress_callback(self, progress_callback) -> None:
        """Bind the output-landing hook onto every per-domain writer.

        The constructor keyword cannot serve every caller.  Both
        prepared runners build their progress closure OVER this object
        -- it reports ``writers.paths`` -- so the closure genuinely
        cannot exist before the thing it reads.  Reordering does not
        break a real cycle; late binding does, and this is it.

        Refused once any domain has submitted a frame.  Attaching after
        the first output has landed would skip it, and an event stream
        silently missing its first frame is worse than one that refused
        to start: the consumer has no way to notice.
        """

        busy = sorted(grid_id for grid_id, writer in self._writers.items()
                      if writer.paths or writer.pending)
        if busy:
            raise RuntimeError(
                "cannot attach an output observer after domain(s) "
                f"{busy} have already submitted frames; the observer "
                "would silently miss them.  Attach before the first "
                "history period, or pass progress_callback to the "
                "constructor.")
        observer = getattr(progress_callback, "output_committed", None)
        for writer in self._writers.values():
            writer.landing_observer = observer

    @property
    def pending(self) -> int:
        return sum(writer.pending for writer in self._writers.values())

    @property
    def paths(self) -> tuple[Path, ...]:
        ret = []
        for gid in sorted(self._writers):
            ret.extend(self._writers[gid].paths)
        return tuple(ret)

    def add_domain(self, grid_id: int, *, grid, static_fields) -> None:
        """Mint a writer for a domain that appeared AFTER construction.

        A spawn-triggered nest does not exist when the writer set is
        built -- it is dormant, and its birth is a leg boundary in the
        middle of the run.  This is the same per-domain construction the
        constructor performs, for exactly one late arrival; refusing a
        second call for the same grid keeps it from silently discarding
        a writer that already holds queued frames.
        """
        from gpuwm.runtime import _global_wrf_attrs, _metadata_frame

        grid_id = int(grid_id)
        if grid_id in self._writers:
            raise ValueError(
                f"d{grid_id:02d} already has a wrfout writer; a spawned "
                "domain is added once, and re-adding would drop frames "
                "the existing writer has queued (use refresh_domain to "
                "re-stamp metadata after a relocation)")
        node = self.model.node(grid_id)
        case = self.model._prepared_by_grid_id[grid_id]
        configured_start = getattr(node.cfg, "start_time", None)
        domain_start_time = (self.start_time if configured_start is None
                             else configured_start)
        self._metadata_by_grid_id[grid_id] = _metadata_frame(
            grid, static_fields)
        self._writers[grid_id] = AsyncDomainWrfoutWriter(
            nx=node.cfg.run.nx, ny=node.cfg.run.ny, nz=node.cfg.run.nz,
            dx=node.cfg.run.dx, dy=node.cfg.run.dy, title=self.title,
            soil_layers=soil_layer_count(node.cfg.run),
            global_attrs=_global_wrf_attrs(
                grid, domain_start_time,
                getattr(case, "geog_selection", None),
                domain=node.cfg, coord=case.initial_result.coord,
                feedback=getattr(self.model, "_feedback_provenance", None),
                initial_condition=self._initial_condition,
                source=self._source),
            abort_event=self._abort_event)

    def refresh_domain(self, grid_id: int, *, grid, static_fields) -> None:
        """Re-derive one domain's frame metadata after a relocation.

        The XLAT/XLONG/MAPFAC/HGT metadata frame and the placement-carrying
        global attributes were computed at construction and would otherwise
        describe the footprint the domain no longer covers.  The route's
        relocation preparer calls this with the rebuilt child's grid and
        footprint-rebuilt statics; the writer drains first, so frames from
        the outgoing placement keep the coordinates that produced them.
        """
        grid_id = int(grid_id)
        node = self.model.node(grid_id)
        case = self.model._prepared_by_grid_id[grid_id]
        from gpuwm.runtime import _global_wrf_attrs, _metadata_frame

        configured_start = getattr(node.cfg, "start_time", None)
        domain_start_time = (self.start_time if configured_start is None
                             else configured_start)
        self._metadata_by_grid_id[grid_id] = _metadata_frame(
            grid, static_fields)
        self._writers[grid_id].update_global_attrs(_global_wrf_attrs(
            grid, domain_start_time,
            getattr(case, "geog_selection", None),
            domain=node.cfg, coord=case.initial_result.coord,
            feedback=getattr(self.model, "_feedback_provenance", None),
            initial_condition=self._initial_condition,
            source=self._source))

    def submit(self, node, ticks: int, *, refl_field=None) -> None:
        seconds = ticks / node.clock.tick_den
        valid_time = self.start_time + timedelta(seconds=seconds)
        path = self.output_dir / wrfout_filename(
            valid_time, node.cfg.grid_id)
        self._writers[node.cfg.grid_id].submit(
            path, valid_time, node.state,
            extra_fields=self._metadata_by_grid_id[node.cfg.grid_id],
            refl_field=refl_field)

    def drain(self) -> None:
        for gid in sorted(self._writers):
            self._writers[gid].drain()
            if self._writers[gid].paths:
                self.last_durable_wrfout = self._writers[gid].paths[-1]

    def close(self) -> None:
        saved: BaseException | None = None
        for gid in sorted(self._writers):
            writer = self._writers[gid]
            try:
                writer.close()
            except BaseException as exc:
                if saved is None:
                    saved = exc
            if writer.paths:
                self.last_durable_wrfout = writer.paths[-1]
        if saved is not None:
            raise saved

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            # D2H already in flight must complete before process teardown;
            # publication failures remain quarantined by WrfoutWriter.
            self._abort_event.set()
            with suppress(BaseException):
                self.close()
        return False
