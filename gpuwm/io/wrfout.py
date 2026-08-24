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
import os
from pathlib import Path
import queue
import threading
import traceback

import numpy as np
import netCDF4

from gpuwm import perf_timing
from gpuwm.config import NO_LAND_SURFACE_SOIL_LAYERS, soil_layer_count
from gpuwm.io.classic_tape import (ClassicDim, ClassicTape, ClassicVariable,
                                   classic_attr_value)
from gpuwm.io.wrf_output_schema import (
    HISTORY_FIELDS_BY_NETCDF_NAME, PHYSICS_SELECTOR_GLOBALS,
    REGISTRY_VAR_META, SCHEME_OUTPUT_FIELDS, WRF_FIELD_TYPE_INTEGER,
    WRF_FIELD_TYPE_REAL,
)
from gpuwm.supervisor import (fsync_file, quarantine_file,
                              replace_file_with_retry, unique_temp_path)

_COMPLETION_ATTR = "GPUWM_WRITE_COMPLETE"
# netCDF4 releases the GIL around HDF5 calls, while the shipped HDF5 library
# is not thread-safe.  Domain D2H streams and staging remain independent; only
# each worker's create/write/close/reopen/publish netCDF session is serialized.
# The Rust engine needs no such serialization of its own, but the publish
# step's self-validation reopens the tape with netCDF4, so the lock stays.
_NETCDF4_IO_LOCK = threading.Lock()
_ASYNC_WRITER_POLL_SECONDS = 0.05

#: Which library writes the product tape.  ``rust`` (the DEFAULT) is the
#: dependency-free classic writer at ``tools/rustwx/crates/netcdf-writer``
#: behind the :mod:`gpuwm.io.nc_writer_bridge` ctypes seam, emitting the
#: CDF-2 (``NETCDF3_64BIT_OFFSET``) container.  ``python`` is the
#: netCDF4/HDF5 writer this module used through 2.4, emitting
#: ``NETCDF4_CLASSIC`` -- kept reachable as an EXPLICIT WORKAROUND only,
#: never selected silently.  Their equivalence on real frames is the
#: dual-write verification in ``tests/test_wrfout_dual_write.py`` and
#: ``tools/wrfout_dual_write.py``.
WRFOUT_WRITER_ENV = "GPUWM_WRFOUT_WRITER"

_WRFOUT_ENGINES = ("rust", "python")


def resolve_wrfout_engine(engine: str | None = None) -> str:
    """The writer engine to use: explicit argument, else env, else rust."""
    value = engine if engine is not None else os.environ.get(
        WRFOUT_WRITER_ENV, "rust")
    value = str(value).strip().lower()
    if value not in _WRFOUT_ENGINES:
        raise ValueError(
            f"unknown wrfout writer engine {value!r} "
            f"(from {WRFOUT_WRITER_ENV!r} or the engine argument); "
            f"expected one of {_WRFOUT_ENGINES}.  'rust' is the default "
            "classic writer; 'python' is the netCDF4 workaround.")
    return value


#: The classic-container spelling of an attribute value, and the
#: netCDF4-shaped facade that puts this tape on the Rust seam.  Both live
#: in :mod:`gpuwm.io.classic_tape` because the wrfinput/wrfbdy export
#: writes through the same seam: two copies of this facade is how two
#: products come to disagree about what the classic writer does.
_classic_attr_value = classic_attr_value
_ClassicDim = ClassicDim
_ClassicTapeVariable = ClassicVariable

#: What a caller of THIS tape should do instead when the header is
#: already on disk.
_FROZEN_REMEDY = (
    "Declare it before the first frame (or pass field_schema), or use "
    f"the netCDF4 workaround {WRFOUT_WRITER_ENV}=python.")


class _ClassicTape(ClassicTape):
    """The history tape's :class:`~gpuwm.io.classic_tape.ClassicTape`.

    It differs from a bare one in exactly two declared ways: it stamps
    ``GPUWM_WRITE_COMPLETE`` into the header (last, where the netCDF4
    path's close-time ``setncattr`` also lands it), and its
    header-frozen refusal names this writer's own escapes.
    """

    def __init__(self, path, *, container: str = "cdf2"):
        super().__init__(path, container=container,
                         completion_attr=_COMPLETION_ATTR,
                         frozen_remedy=_FROZEN_REMEDY)


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


#: WRF Registry metadata ``name -> (description, units)``, read from
#: the schema module that owns it.  The table moved there so
#: ``gpuwm.io.history_selection`` can read the NAMES at
#: config-resolution time without importing this writer -- and the
#: supervisor, netCDF4 and runtime behind it.  Kept bound here under
#: the name the writer has always used.
_VAR_META = REGISTRY_VAR_META

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

#: The land-use table identity stamped on every history file.
#:
#: ``ISOILWATER`` is the water category of the SOIL table, and it is 14 for
#: every MODIS/Noah run gpuwm makes -- but it was the one member of the group
#: this writer left out, while stock WRF writes it in every wrfout
#: (``share/output_wrf.F:973``, an explicit ``wrf_put_dom_ti_integer`` rather
#: than a Registry ``h`` flag, which is why it is easy to miss).  Its absence
#: was not inert: ``gpuwm.offline_child.read_child_surface_state`` reads the
#: four attributes below as REQUIRED evidence precisely so category semantics
#: are never assumed, and then quietly assumed 14 for this one because gpuwm's
#: own files never carried it.  Writing it makes the file say what the reader
#: had to guess.
_DEFAULT_LANDUSE_ATTRS = {
    "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
    "ISWATER": 17,
    "ISLAKE": 21,
    "ISICE": 15,
    "ISURBAN": 13,
    "ISOILWATER": 14,
}


def _producer_version() -> str:
    """The release that wrote this file -- the one that EXECUTED.

    ``GPUWM_VERSION`` is the only version this product stamps onto an
    output artifact, so it is the number a reader quotes back months
    later.  It used to be ``gpuwm.__version__``, which is a claim made
    by distribution METADATA rather than by the code doing the writing:
    ``importlib.metadata`` is asked for the version of the distribution
    NAMED gpuwm, and on a box with a stale editable install that answers
    for a tree which is not the one running.  A file stamped that way
    names a release it was not written by, and nothing downstream can
    tell.

    :func:`gpuwm.provenance_gate.executing_version` prefers the running
    code's own declaration and falls back through the metadata to the
    honest ``0+unknown``, so the attribute describes the bytes that
    wrote the file.  On a plain wheel install the two are identical by
    construction -- pip wrote the code and the metadata together -- so
    nothing about an ordinary install's output moves.

    Imported at call time rather than at module import so the writer
    stays importable from a source tree that was never installed, and
    fully defensive: a history write must not fail over a label.
    """
    try:
        from gpuwm.provenance_gate import executing_version

        return str(executing_version())
    except Exception:                                   # noqa: BLE001
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
            ("QNWFA", "nwfa"), ("QNIFA", "nifa"),
            # P3's rime mass and rime volume (mp_physics=50 only, and
            # presence-guarded like every row above).  WRF gives both the
            # history ``h`` in Registry.EM_COMMON:555-558.  th_old/qv_old
            # are deliberately NOT here: their IO string is ``rusd``
            # (:1598-1599) -- restart, no history -- so WRF does not
            # publish them either, and gpuwm follows.
            ("QIR", "qir"), ("QIB", "qib"),
            # WDM6's CCN reservoir (Registry.EM_COMMON:3031 declares
            # scalar:qnn,qnc,qnr for wdm6scheme).  It publishes under the
            # same QNCCN name NSSL's qnn does further down, and that is
            # safe rather than a collision: mp=16 allocates ``nn`` and mp=18
            # allocates ``qnn``, never both, so at most one row can fire on
            # any state.  QNCLOUD/QNRAIN need no new rows -- WDM6's nc/nr
            # are already mapped above, and under mp=16 they simply carry a
            # double-moment warm-rain pair instead of Morrison's.
            ("QNCCN", "nn")):
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
            ("SMOIS", "smois"), ("SH2O", "sh2o"),
            # The land/soil IDENTITY the five rows above are the STATE of.
            # Same gate, same dict, same presence guard -- and the reason
            # they are here rather than left in memory is that a wrfout is
            # this product's boundary: gpuwm's own offline child reads a
            # child-grid history file back as its --child-surface-from
            # source and requires ISLTYP, TMN and VEGFRA among the nine
            # fields it will not fabricate (gpuwm.offline_child
            # ._SURFACE_REQUIRED_FIELDS).  Without these rows gpuwm's
            # history could not seed gpuwm's own child, which is how this
            # was found: on a real 12 km parent, and again on a nested d02.
            #
            # IVGTYP and SEAICE ride the same commit because they are the
            # same class and the same fix -- WRF core `misc` land identity
            # this driver has always carried and never published.  SEAICE
            # in particular closes a silent hole on the reader side: the
            # child's surface reader treats it as optional and substitutes
            # ZEROS when absent, so an ice-covered child was being warm-
            # started ice-free with nothing said.
            ("ISLTYP", "isltyp"), ("IVGTYP", "ivgtyp"),
            ("TMN", "tmn"), ("VEGFRA", "vegfra"), ("SEAICE", "xice")):
        if field_name in live_surface:
            fields[output_name] = live_surface[field_name]
    return fields


def _driver_refreshes_psfc(state) -> bool:
    """Is ``state.physics.fields["psfc"]`` a computed surface pressure?

    A physics driver allocates ``psfc`` unconditionally but refreshes it
    from ``p_interface[0]`` only inside the surface/PBL cadence block
    (``gpuwm/core/physics.py``, guarded by ``self.surface_enabled``).  A
    microphysics-only or dycore-plus-radiation composition therefore
    carries the allocation seed -- 100000 Pa -- for the whole forecast,
    and publishing it wrote a 150 hPa fabrication into wrfout from the
    most ordinary idealized run there is.  Asking whether the surface is
    on, rather than whether a driver exists, hands those compositions to
    the diagnostic extrapolation that already sits one branch below and
    is the same answer WRF's ``phy_prep`` computes.

    A driver-shaped object with no ``surface_enabled`` attribute reads as
    "no refresh", which routes to the computed branch: the safe side.
    """
    physics = getattr(state, "physics", None)
    return physics is not None and bool(
        getattr(physics, "surface_enabled", False))


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
        if _driver_refreshes_psfc(state):
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
    """The product-tape writer.

    ``engine`` selects which library puts the bytes on disk (see
    :data:`WRFOUT_WRITER_ENV`).  The DEFAULT is the Rust classic writer;
    ``engine="python"`` (or ``GPUWM_WRFOUT_WRITER=python``) is the
    netCDF4/HDF5 writer kept as an explicit, documented WORKAROUND.  The
    schema, the attribute inventory, the validation and the atomic
    publication protocol are identical on both engines; the container
    differs (CDF-2 classic vs HDF5 ``NETCDF4_CLASSIC``), which every
    reader in the estate handles (netCDF4, wrf-rust, ``rw_wrfbatch``,
    ``rw_netcdf`` -- proved by the dual-write verification).
    """

    def __init__(self, path, *, nx, ny, nz, dx, dy, title="gpuwm",
                 global_attrs=None, field_schema=None, soil_layers=None,
                 engine=None):
        self.engine = resolve_wrfout_engine(engine)
        if self.engine == "rust":
            from gpuwm.io import nc_writer_bridge

            reason = nc_writer_bridge.unavailable_reason()
            if reason is not None:
                raise RuntimeError(
                    "the default wrfout engine is the Rust NetCDF writer "
                    "(the netcdf-writer cdylib behind gpuwm.io."
                    "nc_writer_bridge) and it is not loadable here, so "
                    f"the product tape cannot be written:\n  {reason}\n"
                    "A silent fallback would change the tape's container "
                    "with build state and quietly un-test the default, so "
                    "it is refused instead.  Remedies: build the library "
                    "from a checkout (cd tools/rustwx; cargo build "
                    "--release --locked --offline -p netcdf-writer; "
                    "cd ../..), or run `gpuwm fetch-bridges` on an "
                    f"installed wheel; or set {WRFOUT_WRITER_ENV}=python "
                    "to select the netCDF4 writer as an explicit "
                    "workaround.")
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
        # Pinned to ABSOLUTE once, here: the publish-time self-validation
        # reopens the temp file with netCDF4, and the netCDF-C build in
        # use refuses a CLASSIC file named by a drive-relative path
        # (`\tmp\...` -> `NetCDF: Unknown file format`) that it opens
        # fine absolutely (MEASURED on the same bytes).  Resolving at
        # construction also makes the fsync/rename sequence immune to a
        # later cwd change, on both engines.
        self._final_path = Path(os.path.abspath(path))
        self._final_path.parent.mkdir(parents=True, exist_ok=True)
        self._temp_path = unique_temp_path(self._final_path, hidden=True)
        if self.engine == "rust":
            self.ds = _ClassicTape(self._temp_path)
        else:
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
        with perf_timing.stage("io.wrfout.write_frame") as timed:
            return self._write_frame(time_str, fields, timed)

    def _write_frame(self, time_str: str, fields: dict[str, np.ndarray],
                     timed):
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
        # Declare every variable of this frame BEFORE its first data byte
        # lands.  On the Rust engine the classic header freezes at the
        # first data write, so a variable created mid-frame would be a
        # named refusal; on netCDF4 this only moves creation ahead of the
        # Times record write, which leaves variable-creation order -- the
        # thing HDF5's name heap is laid out by -- exactly as it was.
        arrays: dict[str, np.ndarray] = {}
        for name, arr in frame.items():
            arr = self._wrf_array(name, np.asarray(arr))
            if name not in self.ds.variables:
                self._create_variable(name, self._dims_for(name, arr.shape))
            arrays[name] = arr
        self.ds.variables["Times"][t] = np.frombuffer(buf, dtype="S1")
        written = 0
        for name, arr in arrays.items():
            self.ds.variables[name][t] = arr
            written += int(arr.nbytes)
        timed.count(fields=len(frame), bytes_written=written)
        self._times.append(buf.rstrip(b"\x00").decode("ascii"))
        self._n += 1

    def _abandon_ds(self):
        """Release the dataset without publishing: abort a classic tape
        (numrecs stays 0 on the partial file), plain-close a netCDF4 one
        (no completion attribute, so the sweep quarantines it)."""
        if isinstance(self.ds, _ClassicTape):
            self.ds.abort()
        else:
            self.ds.close()

    def close(self):
        if self._closed:
            return
        inventory = tuple(self.ds.variables)
        shapes = {name: tuple(variable.shape)
                  for name, variable in self.ds.variables.items()}
        times = tuple(self._times)
        try:
            if isinstance(self.ds, _ClassicTape):
                # The completion attribute is already in the header (the
                # tape writes it at freeze, last); close() refuses any
                # tape with an unwritten region, then patches numrecs,
                # flushes and fsyncs.
                self.ds.close()
            else:
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
                    self._abandon_ds()
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
            self._abandon_ds()
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
        if _driver_refreshes_psfc(state):
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
    #: Per-frame global-attribute override.  Carried on the ticket so a
    #: submit-time snapshot (carrier provenance, which can change at a
    #: resume or a mid-run forcing) lands on exactly the file whose frame
    #: it describes -- without draining the async queue the way a
    #: whole-writer ``update_global_attrs`` swap must.  ``None`` means
    #: "use the writer's standing set".
    global_attrs: object = None


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
    #: This domain's ``[output]`` history-variable selection
    #: (:class:`gpuwm.io.history_selection.HistorySelection`), or ``None``
    #: -- which means the same thing ``HistorySelection.FULL`` does:
    #: write every variable the run produces and stamp no attribute.
    #: Declared here for the same reason the two above are: the test
    #: suite builds CPU-only shells of this class with
    #: ``object.__new__``, and an optional feature that exists only on
    #: the ``__init__`` path turns every such construction into an
    #: AttributeError on the worker thread.
    history_selection = None

    @staticmethod
    def _new_ticket_queue() -> queue.Queue:
        # Capacity is per domain: one queued ticket in each of four queues is
        # one complete four-domain history burst.  Each worker may additionally
        # hold its current handoff, bounding queued + worker-held tickets at
        # eight across a synchronized four-domain write boundary.
        return _AsyncTicketQueue(maxsize=1)

    def __init__(self, *, nx, ny, nz, dx, dy, title, global_attrs,
                 abort_event=None, soil_layers=None, grid_id=None,
                 landing_observer=None, history_selection=None):
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
        self.history_selection = history_selection
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

    def _peer_failure(self):
        """The FIRST failure recorded on the shared abort event, if any.

        The abort event is shared across every domain's writer; the
        exception itself lives on the writer that hit it.  Before this
        attribute existed, a run whose d01 writer failed reported only
        d02's bare 'writing was aborted' -- the root cause was recorded
        on an object nothing consulted and appeared in no log, capsule,
        or traceback (measured: a 6 h moving-nest run whose only
        diagnostic was that one sentence).
        """
        return getattr(self._abort_event, "first_failure", None)

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
            peer = self._peer_failure()
            if peer is not None:
                raise RuntimeError(
                    "per-domain wrfout writing was aborted by another "
                    f"domain's writer: {peer[1]}") from peer[0]
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

    def _history_plan(self, produced) -> tuple[frozenset[str], dict]:
        """(names to write, global attributes recording the selection).

        Resolved ONCE per frame, before a single byte is staged, because
        that is what makes the selection worth having: a dropped 3-D
        volume must never cross the PCIe bus, and filtering at write
        time would copy every one of them host-ward first.  On a 400^2 x
        50 domain that is the majority of what the user is trying to
        save.

        A ``None`` selection -- and ``preset = "full"`` with no drop
        list -- keeps everything and stamps nothing, so a default run's
        frame and its header are exactly what they were before this
        surface existed.
        """
        selection = self.history_selection
        if selection is None or selection.writes_everything:
            return frozenset(produced), {}
        return frozenset(selection.select(produced)), \
            selection.wrfout_attrs(produced)

    def _frame_attrs(self, global_attrs, history_attrs) -> dict | None:
        """The ticket's global-attribute override, or ``None`` for none.

        The history stamp has to reach the FILE, so it is folded onto
        whichever attribute set this frame was going to carry: the
        per-frame override when the caller supplied one (carrier
        provenance), the writer's standing set otherwise.
        """
        if not history_attrs:
            return None if global_attrs is None else dict(global_attrs)
        base = self.global_attrs if global_attrs is None else global_attrs
        return {**base, **history_attrs}

    def submit(self, path, valid_time, state, *, extra_fields=None,
               refl_field=None, frame=None, global_attrs=None) -> None:
        """Queue a nonblocking, stream-ordered snapshot of one domain.

        ``frame`` is a COMPLETE host frame, already assembled, and it is the
        seam an out-of-core domain publishes through.  A streamed domain has
        no resident ``DomainState`` to read: ``gpuwm.core.streaming.attach``
        copies the carriers into a pinned host store and the sweep writes
        the STORE, so ``state`` still exists, still has the right shapes,
        and still holds the values it held at t = 0 forever.  Passing it
        here is the shipped defect this parameter closes -- a full run's
        worth of frames that are correct in inventory, in Times and in
        global attributes, and frozen at the initial condition
        (``tilestream.test_history.negative_frame_from_state`` measures how
        much of a frame that freezes and requires it to differ).

        With ``frame`` given, ``state`` is not read at all -- pass ``None``
        -- and there is no D2H: the fields are already host arrays, so the
        side-stream staging this class exists for becomes a no-op and the
        writer thread is fed directly.  ``tilestream.output.StoreFrame``
        builds exactly such a frame off the pinned store, in the device
        frame's own field order (which is load-bearing: HDF5 lays its name
        heap out in variable-creation order, and the same numbers in a
        different order give a file that hashes differently while every
        variable in it compares equal).

        THE FRAME'S ARRAYS ARE BORROWED, NOT COPIED, and that is the caller's
        problem to get right, exactly as it is for ``state``.  With
        ``StoreFrame(overlap=False)`` the carriers are zero-copy views of the
        pinned store -- which is what makes an out-of-core frame 2.2x-2.4x
        cheaper than the resident path -- so the caller must ``drain()``
        before it sweeps again, or use ``overlap=True`` and pay one host
        memcpy for a snapshot the writer thread may hold.  Submitting views
        and sweeping on writes a frame the scatter is mutating underneath;
        it is a property, not a race, and
        ``tilestream.test_history.negative_async_no_drain`` asserts it.

        ``global_attrs`` is a per-frame global-attribute override (carrier
        provenance above all); it rides the ticket so the snapshot taken at
        submit time lands on exactly the file that carries this frame.
        """
        import cupy as cp

        if self._closed:
            raise RuntimeError("cannot submit to a closed wrfout writer")
        self._raise_failure()
        producer = cp.cuda.get_current_stream()
        if frame is not None:
            if state is not None:
                raise ValueError(
                    "submit() was given both a prepared host frame and a "
                    "device state; they are two different domains' worth of "
                    "numbers and there is no rule for which wins.  A "
                    "streamed domain passes state=None.")
            self._admit_host_frame(path, valid_time, frame,
                                   extra_fields=extra_fields,
                                   refl_field=refl_field, producer=producer,
                                   global_attrs=global_attrs)
            return
        device_fields = _device_state_frame(
            state, include_diagnostic_pressure=True)
        if refl_field is not None:
            device_fields["REFL_10CM"] = refl_field
        # The [output] selection, resolved BEFORE the staging loop below
        # so a dropped field costs no D2H at all -- see _history_plan.
        produced = list(device_fields)
        produced.extend(name for name in (extra_fields or ())
                        if name not in device_fields)
        keep, history_attrs = self._history_plan(produced)
        ready = cp.cuda.Event()
        ready.record(producer)
        self.stream.wait_event(ready)

        host_fields: dict[str, np.ndarray] = {}
        device_refs: list[object] = []
        pinned_refs: list[object] = []
        with self.stream:
            for name, value in device_fields.items():
                if name not in keep:
                    continue
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
                if name not in host_fields and name in keep:
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
            valid_time=valid_time,
            global_attrs=self._frame_attrs(global_attrs, history_attrs))
        self._admit(ticket)

    def _admit_host_frame(self, path, valid_time, frame, *, extra_fields,
                          refl_field, producer, global_attrs=None) -> None:
        """Publish a frame that is already on the host.

        The whole of ``submit``'s device machinery collapses here and it is
        worth saying which parts and why, because each one was doing a job
        for the resident path that an out-of-core frame has already done:

        * no ``_device_state_frame`` -- the caller assembled the frame;
        * no pinned staging and no ``array.get`` -- the arrays are host
          arrays, and staging them THROUGH the device would be the pure
          bounce the resident path already refuses for its own host-side
          fields (measured ~1.8 GiB of cached device staging);
        * no ``device_refs`` to release and no ``pinned_refs`` to hold --
          the writer thread reads the caller's arrays directly.

        The event is still recorded on the producing stream and still
        waited on by the worker.  It fences nothing of this frame's, but a
        streamed run's REFL_10CM is the exception that makes it necessary
        rather than ceremonial: ``refl_field`` may be a live device array
        (a resident domain's) or a host one (a streamed domain's scattered
        store slot), and the device case needs the same ordering guarantee
        every other device field gets.
        """
        import cupy as cp

        frame = dict(frame)
        # Same [output] selection, same place in the order: before any
        # staging.  A streamed domain's carriers are already on the host,
        # but REFL_10CM may still be a live device array, and a dropped
        # name must not be copied, borrowed or declared either way.
        produced = list(frame)
        if refl_field is not None and "REFL_10CM" not in produced:
            produced.append("REFL_10CM")
        produced.extend(name for name in (extra_fields or ())
                        if name not in produced)
        keep, history_attrs = self._history_plan(produced)
        host_fields: dict[str, np.ndarray] = {}
        device_refs: list[object] = []
        pinned_refs: list[object] = []
        ready = cp.cuda.Event()
        ready.record(producer)
        self.stream.wait_event(ready)
        with self.stream:
            for name, value in frame.items():
                if name not in keep:
                    continue
                host_fields[name] = self._host_or_staged(
                    value, device_refs, pinned_refs)
            if refl_field is not None and "REFL_10CM" in keep:
                host_fields["REFL_10CM"] = self._host_or_staged(
                    refl_field, device_refs, pinned_refs)
            for name, value in (extra_fields or {}).items():
                if name not in host_fields and name in keep:
                    host_fields[name] = np.ascontiguousarray(value)
            done = cp.cuda.Event()
            done.record(self.stream)
        producer.wait_event(done)
        self._admit(_AsyncFrame(
            path=Path(path),
            time_str=valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
            fields=host_fields, event=done,
            device_refs=tuple(device_refs),
            pinned_refs=tuple(pinned_refs),
            valid_time=valid_time,
            global_attrs=self._frame_attrs(global_attrs, history_attrs)))

    def _host_or_staged(self, value, device_refs, pinned_refs) -> np.ndarray:
        """A host array for one field, staging it off the device if needed."""
        import cupy as cp

        if isinstance(value, np.ndarray):
            # Borrowed, NOT copied -- see submit's docstring.  Copying here
            # would silently make every out-of-core frame cost a full extra
            # host pass and would disarm the drain discipline by making the
            # unsafe call safe, which is the wrong direction for a defect
            # that is otherwise invisible.
            return value
        array = cp.ascontiguousarray(value)
        memory = cp.cuda.alloc_pinned_memory(int(array.nbytes))
        host = np.frombuffer(memory, dtype=array.dtype,
                             count=array.size).reshape(array.shape)
        array.get(out=host, stream=self.stream, blocking=False)
        device_refs.append(array)
        pinned_refs.append(memory)
        return host

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
                                global_attrs=(
                                    ticket.global_attrs
                                    if ticket.global_attrs is not None
                                    else self.global_attrs),
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
                # Publish the ROOT CAUSE on the shared event before (or
                # atomically with) setting it, so every OTHER domain's
                # liveness refusal can name it instead of the bare
                # "writing was aborted" (see _peer_failure).
                if getattr(self._abort_event, "first_failure", None) is None:
                    self._abort_event.first_failure = (
                        exc, f"{type(exc).__name__}: {exc}")
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


def carrier_provenance_attrs(physics) -> dict:
    """wrfout globals for the surface-radiation carrier contract.

    The contract (gpuwm/core/radiation_carriers.py) promises that a run's
    carrier provenance -- who wrote GLW/SWDOWN/GSW/COSZEN and when --
    appears in the output metadata, so a reader of a wrfout can ask "did
    this file integrate a sky nobody computed" without the run directory.
    Built from :meth:`CarrierContract.report` at submit time, so the
    stamped provenance is the provenance of the frames in that file, not
    of the driver at construction.

    Returns ``{}`` for a state with no driver or no contract (an
    initial-condition write, a pre-physics smoke): absent keys read as
    "no land-surface consumer existed", which is true for those files.
    ``..._LAST_UPDATE`` is the producer's model second, ``-1.0`` for a
    source that is constant by declaration and has no age.
    """
    carriers = getattr(physics, "carriers", None)
    if carriers is None:
        return {}
    return _carrier_attrs(str(carriers.policy), carriers.report())


def streamed_carrier_provenance_attrs(streamed) -> dict:
    """The same globals, taken from a STREAMED domain's live ledger.

    :func:`carrier_provenance_attrs` reads the contract off a
    ``PhysicsDriver``, and under ``[tiles] store = "host"`` the driver the
    route holds is the snapshot the store was filled from -- it never ran
    radiation, so it reports ``unwritten`` for carriers whose values are in
    the file and correct.  ``StreamedDomain.carrier_provenance`` returns the
    ledger the sweep advances; this renders it in exactly the same keys, so
    a streamed wrfout and a resident one of the same configuration carry
    IDENTICAL provenance attributes rather than merely plausible ones.

    Returns ``{}`` when the streamed domain carries no contract, which sends
    the caller back to the state -- the same answer a pre-physics write
    gets, and not the same thing as a contract that says ``unwritten``.
    """
    provenance = None
    reader = getattr(streamed, "carrier_provenance", None)
    if reader is not None:
        provenance = reader()
    if not provenance:
        return {}
    policy = provenance.get("policy")
    if policy is None:
        return {}
    return _carrier_attrs(str(policy), provenance["records"])


def _carrier_attrs(policy: str, rows) -> dict:
    """Render one carrier ledger as wrfout globals.

    Shared by the resident and streamed readers so the two cannot drift
    into writing the same provenance under different keys or precisions --
    which is what the frame-parity gate compares attribute by attribute.
    """
    attrs = {"GPUWM_SURFACE_RADIATION_POLICY": str(policy)}
    for name, row in rows.items():
        key = str(name).upper()
        attrs[f"GPUWM_CARRIER_{key}_SOURCE"] = str(row["source"])
        last = row["last_update_model_time"]
        attrs[f"GPUWM_CARRIER_{key}_LAST_UPDATE"] = np.float64(
            -1.0 if last is None else last)
    return attrs


class PerDomainWrfoutWriters:
    """One asynchronous writer/side stream per domain in an experiment."""

    #: The tree-wide ``[output]`` history selection this writer set was
    #: built with (``ExperimentConfig.output``), or ``None`` for the FULL
    #: default.  A class attribute so the verification cases and the
    #: idealized runners, which construct this object without one, are
    #: unaffected.
    history_selection = None

    def __init__(self, model, output_dir, *, start_time, title,
                 initial_condition=None, source=None,
                 progress_callback=None, history_selection=None,
                 episodes_by_grid_id=None):
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

        ``episodes_by_grid_id`` is the RESUME seed: a domain restored in
        the middle of its second lifecycle episode must write
        ``d0N/episode-002/`` from its FIRST frame, not from the next
        spawn boundary, or one episode's history lands in two places.
        The default -- empty -- gives every domain episode 0, which is
        the flat historical pathname, so a run that is not a lifecycle
        resume is byte-inert under this parameter.  Values come from
        :func:`gpuwm.core.nest_lifecycle.output_episode`, the same
        function ``add_domain``'s caller uses at a live spawn boundary,
        so a resumed episode and a freshly born one are numbered by one
        rule.
        """
        from gpuwm.runtime import _global_wrf_attrs, _metadata_frame

        self.model = model
        self.history_selection = history_selection
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
        self._episode_by_grid_id = {}
        self._archived_paths = []
        #: Every final pathname THIS writer set has published, which is
        #: what the duplicate-valid-time guard in submit() is scoped to.
        self._published_paths = set()
        self._abort_event = threading.Event()
        resumed_episodes = {int(gid): int(episode) for gid, episode
                            in dict(episodes_by_grid_id or {}).items()}
        for node in model.walk_parent_first():
            case = prepared[node.cfg.grid_id]
            configured_start = getattr(node.cfg, "start_time", None)
            domain_start_time = (
                start_time if configured_start is None
                else configured_start)
            self._metadata_by_grid_id[node.cfg.grid_id] = _metadata_frame(
                node.grid, case.static_fields)
            self._episode_by_grid_id[node.cfg.grid_id] = resumed_episodes.get(
                int(node.cfg.grid_id), 0)
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
                grid_id=node.cfg.grid_id,
                history_selection=self._selection_for(node.cfg))
        self.last_durable_wrfout = None
        if progress_callback is not None:
            self.attach_progress_callback(progress_callback)

    def _selection_for(self, domain_cfg):
        """One domain's ``[output]`` selection: its own, else the tree's.

        Per domain and not per tree because that is where the bytes
        actually are: a 1 km child at the same history cadence as its
        12 km parent writes an order of magnitude more of them, and
        "keep the parent whole, trim the child" has to be sayable.
        """
        from gpuwm.io.history_selection import resolve

        return resolve(self.history_selection,
                       getattr(domain_cfg, "output", None))

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
        ret = list(self._archived_paths)
        for gid in sorted(self._writers):
            ret.extend(self._writers[gid].paths)
        return tuple(ret)

    def add_domain(self, grid_id: int, *, grid, static_fields, episode: int = 0) -> None:
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
        self._episode_by_grid_id[grid_id] = int(episode)
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
            abort_event=self._abort_event,
            history_selection=self._selection_for(node.cfg))

    def remove_domain(self, grid_id: int) -> None:
        """Drain and close one retired episode without losing its paths."""
        grid_id = int(grid_id)
        writer = self._writers.pop(grid_id, None)
        if writer is None:
            return
        writer.drain()
        writer.close()
        self._archived_paths.extend(writer.paths)
        if writer.paths:
            self.last_durable_wrfout = writer.paths[-1]
        self._metadata_by_grid_id.pop(grid_id, None)
        self._episode_by_grid_id.pop(grid_id, None)

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
        """One domain's history frame, from wherever that domain's truth is.

        For a resident domain that is ``node.state``, as it always was.  For
        a STREAMED one it is not: ``gpuwm.core.streaming.attach`` copies the
        carriers into a pinned host store and the sweep advances the STORE,
        leaving ``node.state`` holding the initial condition for the rest of
        the run.  Submitting it produced a full run of frames with the
        correct inventory, the correct Times and the correct global
        attributes, every value frozen at t = 0 -- which is not a crash, not
        a warning, and not obviously wrong in ncview
        (``tilestream.test_history.negative_frame_from_state`` measures how
        much of a frame that freezes and requires it to differ).  So the
        source is asked of the domain rather than assumed, through the
        marker ``streaming.StreamedDomain`` leaves on the state it took
        over.
        """
        seconds = ticks / node.clock.tick_den
        valid_time = self.start_time + timedelta(seconds=seconds)
        episode = int(self._episode_by_grid_id.get(node.cfg.grid_id, 0))
        base_dir = self.output_dir
        if episode > 0:
            base_dir = (self.output_dir / f"d{int(node.cfg.grid_id):02d}"
                        / f"episode-{episode:03d}")
            base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / wrfout_filename(valid_time, node.cfg.grid_id)
        # SAME-RUN duplicates only.  The breakage this names is a lifecycle
        # or restart boundary publishing one valid time twice inside a
        # single run: the second frame's atomic replace destroys the first,
        # and neither the run nor the reader can tell.  A file left by a
        # PREVIOUS run is not that -- re-running into an existing output
        # directory has always overwritten, and refusing it here would
        # strand every repeat run instead of preventing a defect.
        if path in self._published_paths:
            raise RuntimeError(
                f"refusing to replace history frame {path}, which THIS run "
                f"already published for d{int(node.cfg.grid_id):02d}: a "
                "lifecycle/restart boundary produced a duplicate valid time, "
                "and the atomic replace would silently destroy the earlier "
                "frame. (A frame left by a PREVIOUS run at this path is "
                "replaced as it always has been.)")
        self._published_paths.add(path)
        writer = self._writers[node.cfg.grid_id]
        # CARRIER PROVENANCE, snapshotted per frame.  Each valid time is
        # its own file, and the snapshot rides the ticket rather than the
        # writer's standing attribute set, so the provenance the driver
        # holds NOW lands on exactly the file that carries this frame --
        # a resumed or mid-run-forced carrier is labelled in the file it
        # affects -- without draining the async queue the way an
        # update_global_attrs swap must.
        #
        # ASKED OF THE DOMAIN, exactly as the frame below is.  Under
        # [tiles] store = "host" the driver on node.state is the snapshot
        # the store was filled from: radiation ran on the tile buffers, so
        # that contract still says every carrier is `unwritten` while the
        # frame beside it carries the sky those producers computed.  The
        # streamed reader falls back to the state's contract when the
        # domain carries none, so a resident run is unchanged.
        streamed = getattr(node.state, "_streamed_domain", None)
        carrier_attrs = {}
        if streamed is not None:
            carrier_attrs = streamed_carrier_provenance_attrs(streamed)
        if not carrier_attrs:
            carrier_attrs = carrier_provenance_attrs(
                getattr(node.state, "physics", None))
        frame_attrs = (None if not carrier_attrs
                       else {**writer.global_attrs, **carrier_attrs})
        if streamed is not None:
            # A streamed domain's numbers live in the pinned host store;
            # the provenance snapshot applies to its frames the same way.
            writer.submit(
                path, valid_time, None, frame=streamed.history_fields(),
                extra_fields=self._metadata_by_grid_id[node.cfg.grid_id],
                refl_field=refl_field,
                global_attrs=frame_attrs)
            # The frame's carriers are views of the pinned store and the
            # next sweep scatters into them, so this write is on the
            # critical path by construction.  It is affordable: MEASURED at
            # forecast cadence the whole frame is 0.077% of wall time at
            # hourly output and 0.31% at every 60 steps, against a solver
            # step of ~19 ns/cell with physics.
            writer.drain()
            return
        writer.submit(
            path, valid_time, node.state,
            extra_fields=self._metadata_by_grid_id[node.cfg.grid_id],
            refl_field=refl_field,
            global_attrs=frame_attrs)

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
