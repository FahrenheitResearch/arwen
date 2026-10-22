"""Neutral point-observation table and table-declared cache decoders.

Moved into the engine from gpuwm-global (``arwen_global.obs_table`` at
b54373992587) so the engine's own local data assimilation reads neutral
tables without the global package installed; gpuwm-global keeps its copy
until it is superseded.  The module is unchanged apart from this paragraph.

The observation layer is ONE neutral table.  Every observation is a row
``(source, station_id, latitude, longitude, elevation, level, valid_time,
variable, value, error)`` in the small neutral variable vocabulary below.
A stream joins the system by declaring a :class:`SourceDecoder` TABLE ENTRY:
which header columns identify the stream, which columns feed which neutral
variable through which NAMED conversion from the fixed vocabulary.  The
decode engine is source-agnostic; a new source is a new table entry, never a
new code path.

Sources are fetched by URL or local file path identically, gzip or plain.

The cache files are PRE-DECODED tables: the upstream service already parsed
each raw report into columns, so this module reads tables and never parses a
raw METAR or aircraft report.  The production route for raw-report decode is
an rw-obs Rust decoder behind the obs front-door ladder; this reader is the
research seam for the pre-decoded cache streams.

The neutral table as a file
---------------------------
The Rust observation front doors of the global data assimilation
(``rw_igra2``, ``rw_amv``, ``rw_ndbc``, ``rw_gnssro``, ``rw_asos table``;
``tools/rustwx/crates/rw-obs/src/table.rs``) write this module's own
vocabulary straight to disk as ``gpuwm-obs.table.v2``
(:data:`TABLE_HEADER`): the ten v1 columns above followed by the five
causal-bookkeeping columns ``measurement, nominal_time, published_time,
received_time, revision``.  :func:`decode_neutral_csv` reads a v2 file and
a v1 file (:data:`TABLE_HEADER_V1`, the extras empty) without conversion,
and :func:`decode_obs_csv` recognises either header before it consults the
declared source tables, so every door of the tree that calls
:func:`load_obs` takes these files unchanged.  The bookkeeping columns say
what a number is (:data:`MEASUREMENT_TABLE`: a station pressure recovered
from an altimeter setting is not a reduced sea-level pressure), which
synoptic hour a report is filed under, when its source published it, when
this system first held it, and which source object it came from; a row
whose receipt time is empty is a report whose arrival was not recorded
and is labelled latency unverified by the streams module.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
import urllib.request

KNOTS_TO_M_S = 0.514444
FEET_TO_M = 0.3048
INHG_TO_PA = 3386.389
ISA_SEA_LEVEL_PA = 101_325.0
ISA_SEA_LEVEL_K = 288.15
ISA_LAPSE_K_M = 0.0065
ISA_TROPOPAUSE_M = 11_000.0
ISA_TROPOPAUSE_PA = 22_632.06
ISA_STRATOSPHERE_SCALE_M = 6341.62
# g / (R_d * lapse) with g=9.80665, R_d=287.053: the ISA pressure exponent.
ISA_EXPONENT = 5.255877

# Neutral variable vocabulary: units and gross physical bounds.  A value
# outside its bounds is instrument garbage (a sentinel like 9999, a truncated
# transmission); admitting one would paint a physically impossible innovation
# across every gridpoint inside the localization radius.
VARIABLE_TABLE: dict[str, dict[str, object]] = {
    "surface_pressure_pa": {"units": "Pa", "gross_bounds": (45_000.0, 108_000.0)},
    "temperature_k": {"units": "K", "gross_bounds": (170.0, 340.0)},
    "wind_u_m_s": {"units": "m/s", "gross_bounds": (-150.0, 150.0)},
    "wind_v_m_s": {"units": "m/s", "gross_bounds": (-150.0, 150.0)},
    # The moisture report: a dewpoint, compared against the dewpoint of
    # the model's humidity at the report's pressure and turned into a
    # specific-humidity increment inside the analysis (assimilate.py,
    # the moisture update).  No dewpoint on Earth exceeds about 310 K.
    "dewpoint_k": {"units": "K", "gross_bounds": (170.0, 320.0)},
    # GNSS radio occultation: local refractivity at the tangent point in
    # N-units, N = 77.6 p/T + 3.73e5 e/T^2 (Smith and Weintraub 1953; p and
    # e in hPa, T in K).  About 300 at sea level, below 1 above 60 km; the
    # row's ``elevation_m`` is the tangent height and its vertical anchor,
    # ``level_pa`` the retrieval's dry pressure there (for the column
    # gates).  Its operator is ``obs_operators.refractivity_at_heights``;
    # the v1 door has none and refuses the rows by name.
    "refractivity_n": {"units": "N", "gross_bounds": (0.1, 500.0)},
    # A satellite brightness temperature (ATMS, ABI): the neutral name the
    # radiance streams hand the ensemble filter under, with the stream's
    # own operator (a radiance is not a level quantity and no obs-table
    # decoder writes it as a row).  No clear-sky scene on Earth reads
    # below 100 K or above 340 K in these bands; outside is a fill value.
    "brightness_temperature_k": {"units": "K", "gross_bounds": (100.0, 340.0)},
}

# The neutral table itself as a stream.  ``gpuwm-obs.table.v2`` is what the
# Rust front doors write (``rw-obs/src/table.rs`` keeps the same header
# text); v1 is the ten-column form without the bookkeeping, still read.
TABLE_SCHEMA = "gpuwm-obs.table.v2"
TABLE_SCHEMA_V1 = "gpuwm-obs.table.v1"
TABLE_HEADER_V1 = (
    "source", "station_id", "latitude_deg", "longitude_deg", "elevation_m",
    "level_pa", "valid_time", "variable", "value", "error",
)
TABLE_HEADER = TABLE_HEADER_V1 + (
    "measurement", "nominal_time", "published_time", "received_time", "revision",
)

# What a row's number IS (the ``measurement`` column), transcribed from the
# Rust table module.  The analysis operators key on ``variable``; this
# column is the acceptance contract's measurement definition, so a reduced
# sea-level pressure and a station pressure never share a label and a
# receipt can say which reduction a wind carries.
MEASUREMENT_TABLE: dict[str, str] = {
    "station_pressure_from_altimeter":
        "station pressure recovered exactly from the altimeter setting "
        "through the ISA column at the station elevation (METAR)",
    "station_pressure": "station pressure as reported (a sounding's surface level)",
    "sea_level_pressure":
        "pressure reduced to sea level by the platform's own method, "
        "anchored at 0 m (NDBC PRES)",
    "screen_temperature_2m": "2 m screen temperature (METAR)",
    "screen_dewpoint_2m": "2 m screen dewpoint (METAR)",
    "platform_temperature": "air temperature at the platform's sensor, about 4 m (NDBC)",
    "platform_dewpoint": "dewpoint at the platform's sensor, about 4 m (NDBC)",
    "anemometer_wind_10m": "wind at a 10 m anemometer, as reported",
    "anemometer_wind_5m_reduced_to_10m":
        "buoy wind at 5 m scaled to 10 m by the neutral log law over water",
    "sonde_level": "a radiosonde level at its own release-plus-elapsed time",
    "amv_assigned_pressure": "a satellite motion vector at its assigned pressure",
    "ro_refractivity_tangent_point": "GNSS-RO refractivity at the tangent point height",
    # Added with the prepbufr and MADIS doors (rw_prepbufr, rw_madis); until
    # they were transcribed here every aircraft row and every mesonet or
    # marine wind those doors wrote was counted as an unknown measurement.
    "aircraft_level": "an aircraft report at its flight level (AIREP, PIREP, AMDAR, ACARS)",
    "profiler_level": "a wind profiler range gate",
    "vad_level": "a radar velocity-azimuth-display wind level",
    "vad_superob":
        "GSI's VAD superob: the mean of up to six consecutive VAD levels within "
        "301 m, at the first level's pressure and the levels' mean height",
    "platform_wind":
        "a wind at the platform's own anemometer height, not reduced to 10 m "
        "(ship, buoy, coastal platform, mesonet)",
    "station_pressure_from_sea_level":
        "station pressure computed by NCEP from the reported sea-level pressure "
        "and the station elevation through the standard atmosphere",
}


@dataclass(frozen=True)
class ObsRow:
    """One neutral observation row.

    ``level_pa is None`` marks a surface row anchored at ``elevation_m``
    above sea level; an aloft row carries its pressure level in ``level_pa``
    (and its geometric altitude in ``elevation_m``).
    """

    source: str
    station_id: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    level_pa: float | None
    valid_time: dt.datetime
    variable: str
    value: float
    error: float
    # The v2 bookkeeping, empty or None where the source did not state it
    # (a v1 table, or a declared-source cache decoded here).
    measurement: str = ""
    nominal_time: dt.datetime | None = None
    published_time: dt.datetime | None = None
    received_time: dt.datetime | None = None
    revision: str = ""

    def identity(self) -> tuple[str, str, str, str, float, float, str]:
        """What makes this row one measurement and not another: the
        source and platform, the valid instant, the variable, and where it
        was taken (position to a thousandth of a degree, the pressure
        level to the pascal, ``sfc`` for a surface row).  A stationary
        station reporting again at a later instant is a new identity; a
        moving platform reporting at a new position is a new identity;
        the same report offered to a later analysis cycle is not."""
        instant = self.valid_time
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=dt.timezone.utc)
        return (
            self.source,
            self.station_id,
            instant.astimezone(dt.timezone.utc).isoformat(timespec="seconds"),
            self.variable,
            round(self.latitude_deg, 3),
            round(self.longitude_deg, 3),
            "sfc" if self.level_pa is None else f"{round(self.level_pa):d}",
        )

    def identity_hash(self) -> str:
        """SHA-256 of :meth:`identity`, first 16 hex digits (64 bits): the
        form the analysis receipt and the checkpoint chain carry, so a
        history of thousands of reports stays a few hundred kilobytes and
        no station identifier is written into a checkpoint."""
        text = "|".join(str(part) for part in self.identity())
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def isa_pressure_pa(altitude_m: float) -> float:
    """ICAO standard-atmosphere pressure at a geometric altitude.

    Flight levels are pressure altitudes by definition, so this conversion is
    exact for aircraft reporting a flight level.
    """
    z = float(altitude_m)
    if z <= ISA_TROPOPAUSE_M:
        return ISA_SEA_LEVEL_PA * (
            1.0 - ISA_LAPSE_K_M * z / ISA_SEA_LEVEL_K
        ) ** ISA_EXPONENT
    return ISA_TROPOPAUSE_PA * math.exp(
        -(z - ISA_TROPOPAUSE_M) / ISA_STRATOSPHERE_SCALE_M
    )


def _wind_component(direction_deg: float, speed_kt: float, component: str):
    return _wind_component_m_s(direction_deg, speed_kt * KNOTS_TO_M_S, component)


def _wind_component_m_s(direction_deg: float, speed_m_s: float, component: str):
    if speed_m_s < 0.0 or not 0.0 <= direction_deg <= 360.0:
        return None
    if speed_m_s > 0.0 and direction_deg == 0.0:
        # The cache encodes a variable-direction wind as direction 0 with
        # nonzero speed (true north is encoded as 360).  A fabricated
        # northerly of that speed would be a directional error of up to 180
        # degrees, so the row is not derivable.
        return None
    radians = math.radians(direction_deg)
    if component == "u":
        return -speed_m_s * math.sin(radians)
    return -speed_m_s * math.cos(radians)


def _altimeter_station_pa(altimeter_in_hg: float, elevation_m: float) -> float:
    # The altimeter setting is defined as the ISA sea-level reduction of
    # station pressure, so inverting it with the ISA column recovers station
    # pressure at station elevation.
    return (altimeter_in_hg * INHG_TO_PA) * (
        1.0 - ISA_LAPSE_K_M * elevation_m / ISA_SEA_LEVEL_K
    ) ** ISA_EXPONENT


# The fixed conversion vocabulary table entries may name.  Each callable
# receives the declared columns' parsed floats in declared order and returns
# a neutral-variable value, or None when the inputs cannot derive one.
CONVERSIONS = {
    "celsius_to_kelvin": lambda v: v[0] + 273.15,
    "fahrenheit_to_kelvin": lambda v: (v[0] - 32.0) * (5.0 / 9.0) + 273.15,
    "altimeter_inhg_to_station_pa": lambda v: _altimeter_station_pa(v[0], v[1]),
    "wind_dir_deg_speed_kt_to_u_m_s": lambda v: _wind_component(v[0], v[1], "u"),
    "wind_dir_deg_speed_kt_to_v_m_s": lambda v: _wind_component(v[0], v[1], "v"),
    "wind_dir_deg_speed_m_s_to_u_m_s": lambda v: _wind_component_m_s(v[0], v[1], "u"),
    "wind_dir_deg_speed_m_s_to_v_m_s": lambda v: _wind_component_m_s(v[0], v[1], "v"),
}

# Level modes a table entry may declare.  ``surface``: the row is anchored at
# its elevation and compared at the station's own height.  ``flight_level``:
# the elevation column is a pressure altitude and the level is the ISA
# pressure at it.  ``pressure_level``: the row carries its own pressure in
# ``level_column`` (hPa, the unit every sounding archive reports) and the
# elevation column is the level's geopotential height.
LEVEL_MODES = ("surface", "flight_level", "pressure_level")


@dataclass(frozen=True)
class VariableRule:
    variable: str
    columns: tuple[str, ...]
    conversion: str
    error: float

    def __post_init__(self) -> None:
        if self.variable not in VARIABLE_TABLE:
            raise ValueError(
                f"rule targets unknown neutral variable {self.variable!r}"
            )
        if self.conversion not in CONVERSIONS:
            raise ValueError(
                f"rule names unknown conversion {self.conversion!r}; "
                f"the vocabulary is {sorted(CONVERSIONS)}"
            )
        if not math.isfinite(self.error) or self.error <= 0.0:
            raise ValueError(f"rule for {self.variable!r} needs a positive error")


@dataclass(frozen=True)
class SourceDecoder:
    """Declarative decoder for one CSV stream.  Pure metadata."""

    source: str
    detect_columns: tuple[str, ...]
    id_column: str
    time_column: str
    latitude_column: str
    longitude_column: str
    elevation_column: str
    elevation_to_m: float
    level_mode: str  # one of LEVEL_MODES
    reject_when_true: tuple[str, ...]
    variables: tuple[VariableRule, ...]
    # ``pressure_level`` only: the column carrying the row's pressure in hPa.
    level_column: str | None = None

    def __post_init__(self) -> None:
        if self.level_mode not in LEVEL_MODES:
            raise ValueError(
                f"source {self.source!r} level_mode must be one of {LEVEL_MODES}"
            )
        if (self.level_mode == "pressure_level") != (self.level_column is not None):
            raise ValueError(
                f"source {self.source!r}: level_column is declared exactly when "
                "level_mode is 'pressure_level' (the row's own pressure in hPa)"
            )
        if not self.variables:
            raise ValueError(f"source {self.source!r} declares no variables")


# The declared streams.  Column names are the caches' own; adding a stream is
# adding an entry here (plus, at most, a conversion to the vocabulary above).
DECODER_TABLES: tuple[SourceDecoder, ...] = (
    SourceDecoder(
        source="awc-metar-cache",
        detect_columns=(
            "station_id", "observation_time", "latitude", "longitude",
            "altim_in_hg", "elevation_m",
        ),
        id_column="station_id",
        time_column="observation_time",
        latitude_column="latitude",
        longitude_column="longitude",
        elevation_column="elevation_m",
        elevation_to_m=1.0,
        level_mode="surface",
        reject_when_true=(),
        variables=(
            VariableRule(
                variable="surface_pressure_pa",
                columns=("altim_in_hg", "elevation_m"),
                conversion="altimeter_inhg_to_station_pa",
                error=100.0,
            ),
            VariableRule(
                variable="temperature_k",
                columns=("temp_c",),
                conversion="celsius_to_kelvin",
                error=1.5,
            ),
            VariableRule(
                variable="wind_u_m_s",
                columns=("wind_dir_degrees", "wind_speed_kt"),
                conversion="wind_dir_deg_speed_kt_to_u_m_s",
                error=2.5,
            ),
            VariableRule(
                variable="wind_v_m_s",
                columns=("wind_dir_degrees", "wind_speed_kt"),
                conversion="wind_dir_deg_speed_kt_to_v_m_s",
                error=2.5,
            ),
        ),
    ),
    # The IEM ASOS download as `rw_asos fetch` writes it (station, valid,
    # lon, lat, elevation, tmpf, dwpf, drct, sknt, gust, alti, mslp, p01i):
    # the record the observation scorecard scores against, offered to the
    # door unchanged.  Temperature and dewpoint are Fahrenheit, wind knots
    # and degrees, the altimeter inches of mercury; elevation metres.
    SourceDecoder(
        source="iem-asos-csv",
        detect_columns=(
            "station", "valid", "lon", "lat", "elevation", "tmpf", "dwpf",
            "alti",
        ),
        id_column="station",
        time_column="valid",
        latitude_column="lat",
        longitude_column="lon",
        elevation_column="elevation",
        elevation_to_m=1.0,
        level_mode="surface",
        reject_when_true=(),
        variables=(
            VariableRule(
                variable="surface_pressure_pa",
                columns=("alti", "elevation"),
                conversion="altimeter_inhg_to_station_pa",
                error=100.0,
            ),
            VariableRule(
                variable="temperature_k",
                columns=("tmpf",),
                conversion="fahrenheit_to_kelvin",
                error=1.5,
            ),
            VariableRule(
                variable="dewpoint_k",
                columns=("dwpf",),
                conversion="fahrenheit_to_kelvin",
                error=1.5,
            ),
            VariableRule(
                variable="wind_u_m_s",
                columns=("drct", "sknt"),
                conversion="wind_dir_deg_speed_kt_to_u_m_s",
                error=2.5,
            ),
            VariableRule(
                variable="wind_v_m_s",
                columns=("drct", "sknt"),
                conversion="wind_dir_deg_speed_kt_to_v_m_s",
                error=2.5,
            ),
        ),
    ),
    SourceDecoder(
        source="awc-aircraft-cache",
        detect_columns=(
            "aircraft_ref", "observation_time", "latitude", "longitude",
            "altitude_ft_msl", "report_type",
        ),
        id_column="aircraft_ref",
        time_column="observation_time",
        latitude_column="latitude",
        longitude_column="longitude",
        elevation_column="altitude_ft_msl",
        elevation_to_m=FEET_TO_M,
        level_mode="flight_level",
        reject_when_true=(
            "bad_location", "no_time_stamp", "no_flt_lvl",
            "above_ground_level_indicated",
        ),
        variables=(
            VariableRule(
                variable="temperature_k",
                columns=("temp_c",),
                conversion="celsius_to_kelvin",
                error=1.0,
            ),
            VariableRule(
                variable="wind_u_m_s",
                columns=("wind_dir_degrees", "wind_speed_kt"),
                conversion="wind_dir_deg_speed_kt_to_u_m_s",
                error=2.5,
            ),
            VariableRule(
                variable="wind_v_m_s",
                columns=("wind_dir_degrees", "wind_speed_kt"),
                conversion="wind_dir_deg_speed_kt_to_v_m_s",
                error=2.5,
            ),
        ),
    ),
    # A radiosonde level table as a radiosonde level-table tool
    # writes it from the IGRA2 archive (one row per station, nominal time
    # and pressure level: station, valid, lon, lat, gph_m, pressure_hpa,
    # temp_c, dewpoint_c, wdir_deg, wspd_m_s).  The level's pressure is the
    # row's own; the geopotential height is its elevation.  Temperature
    # and dewpoint Celsius, wind degrees and metres per second (the
    # archive's units).  A blank column leaves that variable underivable,
    # so a dewpoint-only table offers the door dewpoint alone.  Errors:
    # temperature 1.0 K, dewpoint 2.5 K (the humidity sensor's lag and the
    # representativeness of one ascent against a 52 km column), wind 2.5 m/s.
    SourceDecoder(
        source="igra2-levels-csv",
        detect_columns=(
            "station", "valid", "lon", "lat", "gph_m", "pressure_hpa",
            "temp_c", "dewpoint_c",
        ),
        id_column="station",
        time_column="valid",
        latitude_column="lat",
        longitude_column="lon",
        elevation_column="gph_m",
        elevation_to_m=1.0,
        level_mode="pressure_level",
        reject_when_true=(),
        variables=(
            VariableRule(
                variable="temperature_k",
                columns=("temp_c",),
                conversion="celsius_to_kelvin",
                error=1.0,
            ),
            VariableRule(
                variable="dewpoint_k",
                columns=("dewpoint_c",),
                conversion="celsius_to_kelvin",
                error=2.5,
            ),
            VariableRule(
                variable="wind_u_m_s",
                columns=("wdir_deg", "wspd_m_s"),
                conversion="wind_dir_deg_speed_m_s_to_u_m_s",
                error=2.5,
            ),
            VariableRule(
                variable="wind_v_m_s",
                columns=("wdir_deg", "wspd_m_s"),
                conversion="wind_dir_deg_speed_m_s_to_v_m_s",
                error=2.5,
            ),
        ),
        level_column="pressure_hpa",
    ),
)


def parse_valid_time(text: str) -> dt.datetime | None:
    raw = text.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        moment = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def _parse_float(text: str) -> float | None:
    raw = text.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _truthy(text: str) -> bool:
    return text.strip().upper() == "TRUE"


def _first_index_map(header: list[str]) -> dict[str, int]:
    # Cache headers repeat group columns (sky_cover, cloud_base_*).  The
    # first occurrence carries the primary report values, so name -> first
    # index is the declared addressing rule.
    indexes: dict[str, int] = {}
    for position, name in enumerate(header):
        indexes.setdefault(name.strip(), position)
    return indexes


def match_decoder(
    header: list[str], tables: tuple[SourceDecoder, ...] = DECODER_TABLES
) -> SourceDecoder | None:
    names = set(name.strip() for name in header)
    matches = [
        table for table in tables if set(table.detect_columns) <= names
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            "header matches multiple declared sources "
            f"{[t.source for t in matches]}; detect_columns must "
            "discriminate or one stream would silently decode as another"
        )
    return matches[0]


def neutral_header_version(fields: list[str]) -> int | None:
    """2 for the v2 neutral header, 1 for the v1 header, None otherwise."""
    names = tuple(name.strip() for name in fields)
    if names == TABLE_HEADER:
        return 2
    if names == TABLE_HEADER_V1:
        return 1
    return None


def decode_neutral_csv(text: str) -> tuple[str, list[ObsRow], dict[str, object]]:
    """Read a ``gpuwm-obs.table.v2`` (or v1) file: rows already in the
    neutral vocabulary, no conversion.  A row naming a variable outside
    :data:`VARIABLE_TABLE`, a non-finite value, a non-positive error or a
    position off the sphere is counted and dropped, never coerced; a
    measurement label outside :data:`MEASUREMENT_TABLE` is counted and the
    row kept with the label as written (the analysis keys on the variable);
    the gross bounds are the analysis's gate (``assimilate``), not this
    reader's, so a row inside the vocabulary passes through as written.
    Returns ``(source, rows, counters)`` with ``source`` the one source the
    rows share or the sorted sources joined by ``+``."""
    reader = csv.reader(io.StringIO(text))
    version = None
    for line_number, fields in enumerate(reader):
        if line_number >= 10:
            break
        version = neutral_header_version(fields)
        if version is not None:
            break
    if version is None:
        raise ValueError(
            f"not a {TABLE_SCHEMA} file: the header must be exactly "
            f"{','.join(TABLE_HEADER)} (or the ten-column {TABLE_SCHEMA_V1} header)"
        )
    counters = {
        "table_version": version,
        "rows_scanned": 0, "rows_malformed": 0, "rows_unknown_variable": 0,
        "rows_unknown_measurement": 0, "rows_without_received_time": 0,
        "rows_decoded": 0, "observations": 0,
    }
    rows: list[ObsRow] = []
    sources: set[str] = set()
    width = len(TABLE_HEADER) if version == 2 else len(TABLE_HEADER_V1)
    for fields in reader:
        if not fields:
            continue
        counters["rows_scanned"] += 1
        if len(fields) < width:
            counters["rows_malformed"] += 1
            continue
        (source, station, lat_text, lon_text, elev_text, level_text,
         time_text, variable, value_text, error_text) = (
            f.strip() for f in fields[:10]
        )
        if version == 2:
            (measurement, nominal_text, published_text, received_text,
             revision) = (f.strip() for f in fields[10:15])
        else:
            measurement = nominal_text = published_text = received_text = revision = ""
        if variable not in VARIABLE_TABLE:
            counters["rows_unknown_variable"] += 1
            continue
        valid_time = parse_valid_time(time_text)
        latitude = _parse_float(lat_text)
        longitude = _parse_float(lon_text)
        elevation = _parse_float(elev_text)
        value = _parse_float(value_text)
        error = _parse_float(error_text)
        level = None if not level_text else _parse_float(level_text)
        nominal = parse_valid_time(nominal_text) if nominal_text else None
        published = parse_valid_time(published_text) if published_text else None
        received = parse_valid_time(received_text) if received_text else None
        if (
            not source or not station or valid_time is None
            or latitude is None or not -90.0 <= latitude <= 90.0
            or longitude is None or not -180.0 <= longitude <= 360.0
            or elevation is None or value is None
            or error is None or error <= 0.0
            or (level_text and (level is None or level <= 0.0))
            or (nominal_text and nominal is None)
            or (published_text and published is None)
            or (received_text and received is None)
        ):
            counters["rows_malformed"] += 1
            continue
        if measurement and measurement not in MEASUREMENT_TABLE:
            counters["rows_unknown_measurement"] += 1
        if received is None:
            counters["rows_without_received_time"] += 1
        rows.append(ObsRow(
            source=source,
            station_id=station,
            latitude_deg=latitude,
            longitude_deg=longitude if longitude <= 180.0 else longitude - 360.0,
            elevation_m=elevation,
            level_pa=level,
            valid_time=valid_time,
            variable=variable,
            value=value,
            error=error,
            measurement=measurement,
            nominal_time=nominal,
            published_time=published,
            received_time=received,
            revision=revision,
        ))
        sources.add(source)
        counters["rows_decoded"] += 1
        counters["observations"] += 1
    return "+".join(sorted(sources)) if sources else TABLE_SCHEMA, rows, counters


def decode_obs_csv(
    text: str, *, tables: tuple[SourceDecoder, ...] = DECODER_TABLES
) -> tuple[str, list[ObsRow], dict[str, object]]:
    """Decode one CSV stream through its declared table.

    Returns ``(source, rows, counters)``.  The header row is located within
    the first ten lines so preamble lines some cache generations carry do
    not break decoding.  A file whose header is the neutral table's own
    (:data:`TABLE_HEADER` or :data:`TABLE_HEADER_V1`) is read by
    :func:`decode_neutral_csv` instead.
    """
    for line in text.splitlines()[:10]:
        if neutral_header_version(line.split(",")) is not None:
            return decode_neutral_csv(text)
    reader = csv.reader(io.StringIO(text))
    decoder = None
    indexes: dict[str, int] = {}
    for line_number, fields in enumerate(reader):
        if line_number >= 10:
            break
        found = match_decoder(fields, tables)
        if found is not None:
            decoder = found
            indexes = _first_index_map(fields)
            break
    if decoder is None:
        raise ValueError(
            "no declared obs decoder matches the header; declared sources "
            f"are {[t.source for t in tables]} - an undeclared stream joins "
            "by a decoder table entry, not by a code path"
        )
    needed = {
        decoder.id_column, decoder.time_column,
        decoder.latitude_column, decoder.longitude_column,
        decoder.elevation_column,
        *(() if decoder.level_column is None else (decoder.level_column,)),
        *decoder.reject_when_true,
        *(name for rule in decoder.variables for name in rule.columns),
    }
    missing = sorted(name for name in needed if name not in indexes)
    if missing:
        raise ValueError(
            f"source {decoder.source} matched but the header lacks declared "
            f"columns {missing}; the upstream layout changed and the table "
            "entry must be re-declared before rows can be trusted"
        )
    width = 1 + max(indexes[name] for name in needed)
    counters = {
        "rows_scanned": 0,
        "rows_flag_rejected": 0,
        "rows_malformed": 0,
        "values_not_derivable": 0,
        "rows_decoded": 0,
        "observations": 0,
    }
    rows: list[ObsRow] = []
    for fields in reader:
        if not fields:
            continue
        counters["rows_scanned"] += 1
        if len(fields) < width:
            counters["rows_malformed"] += 1
            continue
        if any(_truthy(fields[indexes[name]]) for name in decoder.reject_when_true):
            counters["rows_flag_rejected"] += 1
            continue
        station = fields[indexes[decoder.id_column]].strip()
        valid_time = parse_valid_time(fields[indexes[decoder.time_column]])
        latitude = _parse_float(fields[indexes[decoder.latitude_column]])
        longitude = _parse_float(fields[indexes[decoder.longitude_column]])
        elevation = _parse_float(fields[indexes[decoder.elevation_column]])
        if (
            not station
            or valid_time is None
            or latitude is None or not -90.0 <= latitude <= 90.0
            or longitude is None or not -180.0 <= longitude <= 360.0
            or elevation is None
        ):
            counters["rows_malformed"] += 1
            continue
        elevation_m = elevation * decoder.elevation_to_m
        if decoder.level_mode == "flight_level":
            if not -400.0 <= elevation_m <= 20_000.0:
                # Outside the ISA conversion's valid span the pressure level
                # would be fiction.
                counters["rows_malformed"] += 1
                continue
            level_pa = isa_pressure_pa(elevation_m)
        elif decoder.level_mode == "pressure_level":
            level_hpa = _parse_float(fields[indexes[decoder.level_column]])
            if level_hpa is None or not 1.0 <= level_hpa <= 1100.0:
                # A pressure outside the atmosphere is a sentinel, not a level.
                counters["rows_malformed"] += 1
                continue
            level_pa = level_hpa * 100.0
        else:
            level_pa = None
        emitted = 0
        for rule in decoder.variables:
            raw_values = [
                _parse_float(fields[indexes[name]]) for name in rule.columns
            ]
            if any(value is None for value in raw_values):
                counters["values_not_derivable"] += 1
                continue
            value = CONVERSIONS[rule.conversion](tuple(raw_values))
            if value is None:
                counters["values_not_derivable"] += 1
                continue
            rows.append(ObsRow(
                source=decoder.source,
                station_id=station,
                latitude_deg=latitude,
                longitude_deg=longitude if longitude <= 180.0 else longitude - 360.0,
                elevation_m=elevation_m,
                level_pa=level_pa,
                valid_time=valid_time,
                variable=rule.variable,
                value=float(value),
                error=rule.error,
            ))
            emitted += 1
        if emitted:
            counters["rows_decoded"] += 1
            counters["observations"] += emitted
    return decoder.source, rows, counters


def fetch_obs(location: str, *, timeout_s: float = 60.0) -> tuple[str, dict[str, object]]:
    """Read one obs stream from a URL or a local path, gzip-transparent."""
    if "://" in location:
        with urllib.request.urlopen(location, timeout=timeout_s) as response:
            raw = response.read()
    else:
        raw = Path(location).read_bytes()
    provenance = {
        "location": str(location),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace"), provenance


def load_obs(
    location: str, *, tables: tuple[SourceDecoder, ...] = DECODER_TABLES
) -> tuple[str, list[ObsRow], dict[str, object]]:
    text, provenance = fetch_obs(location)
    source, rows, counters = decode_obs_csv(text, tables=tables)
    provenance["source"] = source
    provenance["counters"] = counters
    return source, rows, provenance


__all__ = [
    "CONVERSIONS",
    "DECODER_TABLES",
    "LEVEL_MODES",
    "MEASUREMENT_TABLE",
    "ObsRow",
    "SourceDecoder",
    "TABLE_HEADER",
    "TABLE_HEADER_V1",
    "TABLE_SCHEMA",
    "TABLE_SCHEMA_V1",
    "VARIABLE_TABLE",
    "VariableRule",
    "decode_neutral_csv",
    "decode_obs_csv",
    "fetch_obs",
    "isa_pressure_pa",
    "load_obs",
    "match_decoder",
    "neutral_header_version",
    "parse_valid_time",
]
