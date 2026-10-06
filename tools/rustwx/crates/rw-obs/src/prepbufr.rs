//! NCEP prepbufr reports to neutral observation rows.
//!
//! A prepbufr file is NCEP's quality-controlled conventional observation
//! file: every report (a sounding, an aircraft report, a surface station,
//! a profiler or radar wind profile) carries its values, a quality mark
//! per value, and the history of each value as an event stack.  This
//! module reads a decoded subset the way GSI's `read_prepbufr` does (the
//! same mnemonic lists, read with the same window rule) and turns the
//! mass and wind reports into `gpuwm-obs.table.v2` rows.
//!
//! **The use rule** is GSI's, as HRRR runs it (`noiqc = .false.`):
//!
//! * a value whose own quality mark is 4 or higher, or missing, is not
//!   used (`lim_qm = 4`; marks above 15 are skipped outright);
//! * a temperature, humidity or wind whose level's pressure mark is 4 or
//!   higher is not used either;
//! * a surface pressure is taken only from a surface level (category 0)
//!   at 500 hPa or more whose height mark is below 4 (9 and 15 excepted,
//!   as GSI excepts them);
//! * under the `i_gsdqc = 2` setting the rapid-refresh analysis runs, a
//!   surface humidity (report types 180 to 189) is not used when the
//!   reported dewpoint `TDO` is below -40 C and more than 10 K under the
//!   temperature (usage 116), when `TOB - TDO` exceeds 70 K (117), or when
//!   `TDO` exceeds 32.2 C (118), a missing `TDO` or `TOB` taken as the
//!   library's missing value as GSI takes it; a mesonet wind (type 288)
//!   whose components are both under 0.01 m/s is not used (115).
//!
//! Nothing that fails the rule is written.  Each refusal is counted by
//! report type, variable and mark, so the record says what was dropped
//! and why.  HRRR thins no conventional type, so nothing is thinned here.
//!
//! **Units.**  `XOB` degrees east to -180..180; `DHR` (hours from the
//! cycle time) to an ISO-8601 UTC valid time; `POB` hPa to Pa; `TOB`
//! Celsius to K; `UOB`, `VOB` m/s; `QOB` mg/kg to a dewpoint in K at the
//! level's own pressure (vapour pressure `e = q p / (0.622 + 0.378 q)`,
//! then the inverse of Bolton's `6.112 exp(17.67 t / (t + 243.5))` hPa).
//!
//! **Virtual temperature.**  A temperature is virtual when any event of
//! its stack, read from the newest down to the first missing program
//! code, carries the dictionary's `VIRTMP` program code (GSI's own loop).
//! It is made sensible with the report's own humidity, `T = Tv / (1 +
//! 0.61 q)`, when that humidity passes its mark; otherwise it is counted
//! and not written.
//!
//! **Errors** are GSI's (`read_prepbufr.f90`), from one of two sources.
//! By default the conventional error table (`rw_obs::errtable`, NCEP's
//! published table unless another is given): the error of the row's
//! report type at the level's pressure, interpolated and floored as GSI
//! does it, which is what the operational analysis uses (it reads that
//! table, so `oberrflg` is true and the file's own `POE`, `QOE`, `TOE`,
//! `WOE` are never consulted).  Or the file's own errors, GSI's path
//! without a table; a value without one is then not written.  Either way
//! the error is then raised as GSI raises it from the observation alone:
//! by 1.2 when the value's own mark is 3 or 7 (`inflate_error`), a
//! temperature by 1.2 again above 100 hPa, a wind by 1.2 again above
//! 50 hPa.  A humidity error (tenths of saturation) is carried into
//! dewpoint at the level's temperature.  A table error at or past 1e6
//! (the type does not carry the variable; GSI's own "bad data" error in
//! `errormod`) gives the value no weight, so it is not written and is
//! counted.
//!
//! Not applied here, because they need the background: GSI's `errormod`
//! factor for a level of a profile (the gap to the neighbouring good
//! levels against the model's layer depth there; never below 1, and 1 for
//! a single-level report), and the setup routines' adjustments.  Rows of
//! profiles are counted by report type so a consumer knows the stated
//! error is GSI's before that factor.
//!
//! **Report types** are table rows (`REPORT_TYPES`), not code: each maps
//! a prepbufr report type to a family (sounding, aircraft, profiler, VAD,
//! land surface, marine surface) and the measurement names its rows
//! carry.  A type with no row is counted and not written.

use std::collections::BTreeMap;
use std::error::Error;

use chrono::{DateTime, Duration, TimeZone, Utc};
use serde::Serialize;

use crate::err;
use crate::ncep_bufr::{DataMessage, Datum, NcepFile, Query, Subset, Template, Window, BMISS};
use crate::errtable::{Column, ErrorTable};
use crate::table::{
    isa_altitude_m, RowProvenance, TableRow, TableWriter, GROSS_DEWPOINT_K,
    GROSS_SURFACE_PRESSURE_PA, GROSS_TEMPERATURE_K, GROSS_WIND_M_S, MEAS_AIRCRAFT_LEVEL,
    MEAS_ANEMOMETER_WIND_10M, MEAS_PLATFORM_DEWPOINT, MEAS_PLATFORM_TEMPERATURE,
    MEAS_PLATFORM_WIND, MEAS_PROFILER_LEVEL, MEAS_SCREEN_DEWPOINT_2M, MEAS_SCREEN_TEMPERATURE_2M,
    MEAS_SONDE_LEVEL, MEAS_STATION_PRESSURE, MEAS_STATION_PRESSURE_FROM_ALTIMETER,
    MEAS_STATION_PRESSURE_FROM_SEA_LEVEL, MEAS_VAD_LEVEL, MEAS_VAD_SUPEROB, VAR_DEWPOINT,
    VAR_SURFACE_PRESSURE,
    VAR_TEMPERATURE, VAR_WIND_U, VAR_WIND_V,
};

pub const SOURCE: &str = "prepbufr";

/// The mnemonic lists GSI reads (HRRR v4.1.21 `read_prepbufr.f90:370-376`).
pub const HEADER_MNEMONICS: [&str; 8] = ["SID", "XOB", "YOB", "DHR", "TYP", "ELV", "SAID", "T29"];
pub const OBS_MNEMONICS: [&str; 13] =
    ["POB", "QOB", "TOB", "ZOB", "UOB", "VOB", "PWO", "MXGS", "HOVI", "CAT", "PRSS", "TDO", "PMO"];
pub const MARK_MNEMONICS: [&str; 8] = ["PQM", "QQM", "TQM", "ZQM", "WQM", "NUL", "PWQ", "PMQ"];
pub const ERROR_MNEMONICS: [&str; 7] = ["POE", "QOE", "TOE", "NUL", "WOE", "NUL", "PWE"];
pub const PROGRAM_MNEMONICS: [&str; 1] = ["TPC"];
pub const DRIFT_MNEMONICS: [&str; 3] = ["XDR", "YDR", "HRDR"];
/// The background the file carries beside a wind (GSI reads it for VAD winds).
pub const BACKGROUND_MNEMONICS: [&str; 3] = ["UFC", "VFC", "TFC"];
pub const SUBTYPE_MNEMONICS: [&str; 1] = ["TSB"];
/// The program whose event marks a temperature as virtual.
pub const VIRTUAL_PROGRAM: &str = "VIRTMP";
/// GSI's `lim_qm` with `noiqc = .false.`: a mark this high is not used.
pub const MARK_LIMIT: i64 = 4;
/// The most events GSI asks for per level.
pub const MAX_EVENTS: usize = 20;
const CELSIUS_TO_KELVIN: f64 = 273.15;
/// The factor of the virtual temperature step, `Tv = T (1 + 0.61 q)`.
const VIRTUAL_FACTOR: f64 = 0.61;
const EPSILON: f64 = 0.622;

const POB: usize = 0;
const QOB: usize = 1;
const TOB: usize = 2;
const ZOB: usize = 3;
const UOB: usize = 4;
const VOB: usize = 5;
const CAT: usize = 9;
const TDO: usize = 11;
const PQM: usize = 0;
const QQM: usize = 1;
const TQM: usize = 2;
const ZQM: usize = 3;
const WQM: usize = 4;
const POE: usize = 0;
const QOE: usize = 1;
const TOE: usize = 2;
const WOE: usize = 4;

// ---------------------------------------------------------------- queries

/// GSI's mnemonic lists resolved against one message type.
#[derive(Debug, Clone)]
pub struct Queries {
    pub header: Query,
    pub obs: Query,
    pub marks: Query,
    pub errors: Query,
    pub programs: Query,
    pub drift: Query,
    pub background: Query,
    pub subtype: Query,
}

impl Queries {
    pub fn of(template: &Template) -> Self {
        Self {
            header: template.query(&HEADER_MNEMONICS),
            obs: template.query(&OBS_MNEMONICS),
            marks: template.query(&MARK_MNEMONICS),
            errors: template.query(&ERROR_MNEMONICS),
            programs: template.query(&PROGRAM_MNEMONICS),
            drift: template.query(&DRIFT_MNEMONICS),
            background: template.query(&BACKGROUND_MNEMONICS),
            subtype: template.query(&SUBTYPE_MNEMONICS),
        }
    }
}

fn virtual_code(file: &NcepFile, message: &DataMessage) -> Result<f64, Box<dyn Error>> {
    file.dictionary_of(message).program_code(VIRTUAL_PROGRAM).map(f64::from).map_err(|e| {
        err(format!(
            "message {}: {e}; without it a virtual temperature cannot be told from a sensible one",
            message.number
        ))
    })
}

/// GSI's loop over a temperature's program codes: virtual if any event
/// down to the first missing code is the virtual-temperature program.
pub fn stack_is_virtual<'a>(programs: impl Iterator<Item = &'a Datum>, code: f64) -> bool {
    let mut is_virtual = false;
    for program in programs.take(MAX_EVENTS) {
        match program.number() {
            Some(value) => {
                if value == code {
                    is_virtual = true;
                }
            }
            None => break,
        }
    }
    is_virtual
}

// ------------------------------------------------------------------- dump

/// The oracle listing: one `V` line (the virtual-temperature program
/// code), an `M` line per data message, an `S` line per subset and an `L`
/// line per level, every value as the bits of the double the NCEP library
/// returns.  An `S` line holds GSI's header list and the subtype (`TSB`);
/// an `L` line its observation, mark and error lists, the balloon drift
/// (`XDR YDR HRDR`), the background (`UFC VFC TFC`), the newest program
/// code, and `V` for a virtual temperature or `S`.  The test oracle (a Fortran program
/// on NCEPLIBS-bufr making GSI's calls) prints the same lines, so identity
/// is a byte comparison.
pub fn dump(file: &NcepFile) -> Result<String, Box<dyn Error>> {
    let mut out = String::new();
    let first = file.messages.first().ok_or_else(|| err("the file holds no data message"))?;
    out.push_str(&format!("V {:>4}\n", virtual_code(file, first)? as u32));
    let queries: Vec<Queries> = file.templates.iter().map(|(_, template)| Queries::of(template)).collect();
    for message in &file.messages {
        let template = file.template_of(message);
        let q = &queries[message.template];
        let code = virtual_code(file, message)?;
        out.push_str(&format!("M {:<8} {:>10} {:>6}\n", template.mnemonic, message.envelope.date10(), message.envelope.subsets));
        for (index, subset) in file.subsets_of(message)?.iter().enumerate() {
            let header = subset.windows(&q.header);
            let obs = subset.windows(&q.obs);
            let marks = subset.windows(&q.marks);
            let errors = subset.windows(&q.errors);
            let programs = subset.windows(&q.programs);
            let drift = subset.windows(&q.drift);
            let background = subset.windows(&q.background);
            let subtype = subset.windows(&q.subtype);
            out.push_str(&format!("S {:>6}", index + 1));
            for node in &q.header.nodes {
                let bits = header.first().map(|w| subset.first(w, *node).real8_bits()).unwrap_or(Datum::Missing.real8_bits());
                out.push_str(&format!(" {bits:016X}"));
            }
            let bits = subtype.first().map(|w| subset.first(w, q.subtype.nodes[0]).real8_bits()).unwrap_or(Datum::Missing.real8_bits());
            out.push_str(&format!(" {bits:016X}"));
            // `ufbint` reads at most 255 levels and says 255; `ufbevn`
            // counts every level, and one level where its mnemonic is absent.
            let event_levels = if q.programs.window == Window::Absent { 1 } else { programs.len() };
            out.push_str(&format!(
                " {:>4} {:>4} {:>4} {:>4} {:>4} {:>4}\n",
                obs.len().min(GSI_LEVEL_LIMIT),
                marks.len().min(GSI_LEVEL_LIMIT),
                errors.len().min(GSI_LEVEL_LIMIT),
                event_levels,
                drift.len().min(GSI_LEVEL_LIMIT),
                background.len().min(GSI_LEVEL_LIMIT)
            ));
            for (k, window) in obs.iter().take(GSI_LEVEL_LIMIT).enumerate() {
                out.push_str(&format!("L {:>3}", k + 1));
                for node in &q.obs.nodes {
                    out.push_str(&format!(" {:016X}", subset.first(window, *node).real8_bits()));
                }
                for (query, windows) in [(&q.marks, &marks), (&q.errors, &errors), (&q.drift, &drift), (&q.background, &background)] {
                    for node in &query.nodes {
                        let bits = windows.get(k).map(|w| subset.first(w, *node).real8_bits()).unwrap_or(Datum::Missing.real8_bits());
                        out.push_str(&format!(" {bits:016X}"));
                    }
                }
                let node = q.programs.nodes[0];
                let (top, is_virtual) = match programs.get(k) {
                    Some(w) => (subset.first(w, node).real8_bits(), stack_is_virtual(subset.events(w, node), code)),
                    None => (Datum::Missing.real8_bits(), false),
                };
                out.push_str(&format!(" {top:016X} {}\n", if is_virtual { 'V' } else { 'S' }));
            }
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------- report

/// One level of a report: the values in force, their marks and errors.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Level {
    pub obs: [Option<f64>; 13],
    pub marks: [Option<f64>; 8],
    pub errors: [Option<f64>; 7],
    pub is_virtual: bool,
    /// Balloon drift: longitude (degrees east), latitude, hours from the cycle.
    pub drift: [Option<f64>; 3],
    /// The file's background u, v (m/s) and temperature (C) at this level.
    pub background: [Option<f64>; 3],
}

/// One report (one subset).
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub message_type: String,
    pub cycle: DateTime<Utc>,
    pub station: Option<String>,
    pub xob: Option<f64>,
    pub yob: Option<f64>,
    pub dhr: Option<f64>,
    pub typ: Option<f64>,
    pub elv: Option<f64>,
    /// The report subtype (`TSB`).
    pub tsb: Option<f64>,
    pub levels: Vec<Level>,
}

pub fn cycle_time(message: &DataMessage) -> Result<DateTime<Utc>, Box<dyn Error>> {
    let e = &message.envelope;
    Utc.with_ymd_and_hms(i32::from(e.year), u32::from(e.month), u32::from(e.day), u32::from(e.hour), u32::from(e.minute), 0)
        .single()
        .ok_or_else(|| {
            err(format!(
                "message {}: section 1 states {:04}-{:02}-{:02} {:02}:{:02}, which is no instant",
                message.number, e.year, e.month, e.day, e.hour, e.minute
            ))
        })
}

fn collect<const N: usize>(subset: &Subset, query: &Query, window: Option<&std::ops::Range<u32>>) -> [Option<f64>; N] {
    let mut out = [None; N];
    if let Some(window) = window {
        for (slot, node) in out.iter_mut().zip(&query.nodes) {
            *slot = subset.first(window, *node).number();
        }
    }
    out
}

pub fn report(template: &Template, q: &Queries, subset: &Subset, cycle: DateTime<Utc>, virtual_code: f64) -> Report {
    let header = subset.windows(&q.header);
    let whole = header.first();
    let number = |index: usize| whole.and_then(|w| subset.first(w, q.header.nodes[index]).number());
    let station = whole.and_then(|w| subset.first(w, q.header.nodes[0]).text()).filter(|s| !s.is_empty());
    let obs = subset.windows(&q.obs);
    let marks = subset.windows(&q.marks);
    let errors = subset.windows(&q.errors);
    let programs = subset.windows(&q.programs);
    let drift = subset.windows(&q.drift);
    let background = subset.windows(&q.background);
    let tsb = subset.windows(&q.subtype).first().and_then(|w| subset.first(w, q.subtype.nodes[0]).number());
    let levels = obs
        .iter()
        .enumerate()
        .map(|(k, window)| Level {
            obs: collect(subset, &q.obs, Some(window)),
            marks: collect(subset, &q.marks, marks.get(k)),
            errors: collect(subset, &q.errors, errors.get(k)),
            is_virtual: programs
                .get(k)
                .map(|w| stack_is_virtual(subset.events(w, q.programs.nodes[0]), virtual_code))
                .unwrap_or(false),
            drift: collect(subset, &q.drift, drift.get(k)),
            background: collect(subset, &q.background, background.get(k)),
        })
        .collect();
    Report {
        message_type: template.mnemonic.clone(),
        cycle,
        station,
        xob: number(1),
        yob: number(2),
        dhr: number(3),
        typ: number(4),
        elv: number(5),
        tsb,
        levels,
    }
}

// ------------------------------------------------------------ report types

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    /// Rows anchored at the station's elevation, no pressure level.
    Surface,
    /// Rows at the level's own pressure.
    Level,
}

/// One row of the report-type table.
#[derive(Debug, Clone, Copy)]
pub struct ReportType {
    pub report_type: u16,
    /// A message type the row is restricted to, where one report type is
    /// shared by land and marine reports.
    pub message_type: Option<&'static str>,
    pub family: &'static str,
    pub anchor: Anchor,
    pub temperature: &'static str,
    pub dewpoint: &'static str,
    pub wind: &'static str,
    /// The measurement a surface-level pressure carries; `None` writes none.
    pub surface_pressure: Option<&'static str>,
    /// Levels carry their own balloon-drift position and time.
    pub drift: bool,
}

const fn level_type(report_type: u16, family: &'static str, measurement: &'static str, surface_pressure: Option<&'static str>, drift: bool) -> ReportType {
    ReportType {
        report_type,
        message_type: None,
        family,
        anchor: Anchor::Level,
        temperature: measurement,
        dewpoint: measurement,
        wind: measurement,
        surface_pressure,
        drift,
    }
}

const fn land(report_type: u16, pressure: Option<&'static str>) -> ReportType {
    ReportType {
        report_type,
        message_type: None,
        family: "surface_land",
        anchor: Anchor::Surface,
        temperature: MEAS_SCREEN_TEMPERATURE_2M,
        dewpoint: MEAS_SCREEN_DEWPOINT_2M,
        wind: MEAS_ANEMOMETER_WIND_10M,
        surface_pressure: pressure,
        drift: false,
    }
}

const fn marine(report_type: u16, pressure: Option<&'static str>) -> ReportType {
    ReportType {
        report_type,
        message_type: None,
        family: "surface_marine",
        anchor: Anchor::Surface,
        temperature: MEAS_PLATFORM_TEMPERATURE,
        dewpoint: MEAS_PLATFORM_DEWPOINT,
        wind: MEAS_PLATFORM_WIND,
        surface_pressure: pressure,
        drift: false,
    }
}

const fn only(message_type: &'static str, row: ReportType) -> ReportType {
    ReportType { message_type: Some(message_type), ..row }
}

/// Report types this door writes rows for, from NCEP's prepbufr report
/// type table (mass reports 1xx, wind reports 2xx; PUBLIC
/// https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_2.htm).
/// Satellite winds, scatterometer winds, RASS, reconnaissance and bogus
/// types have no row and are counted as not mapped.
///
/// Station pressure comes only from mass reports.  187 computes it from
/// the altimeter setting, 183 from the sea-level pressure through the
/// standard atmosphere; 192 to 195 estimate it from the standard
/// atmosphere alone, and GSI never uses those (`read_prepbufr.f90`, the
/// usage block: `kx` 192 to 195 with a pressure observation is usage
/// 100), so their rows carry no pressure and the count says so.
pub const REPORT_TYPES: &[ReportType] = &[
    level_type(120, "sounding", MEAS_SONDE_LEVEL, Some(MEAS_STATION_PRESSURE), true),
    level_type(220, "sounding", MEAS_SONDE_LEVEL, None, true),
    level_type(221, "sounding", MEAS_SONDE_LEVEL, None, true),
    level_type(130, "aircraft", MEAS_AIRCRAFT_LEVEL, None, false),
    level_type(131, "aircraft", MEAS_AIRCRAFT_LEVEL, None, false),
    level_type(133, "aircraft", MEAS_AIRCRAFT_LEVEL, None, false),
    level_type(134, "aircraft", MEAS_AIRCRAFT_LEVEL, None, false),
    level_type(135, "aircraft", MEAS_AIRCRAFT_LEVEL, None, false),
    level_type(230, "aircraft", MEAS_AIRCRAFT_LEVEL, None, false),
    level_type(231, "aircraft", MEAS_AIRCRAFT_LEVEL, None, false),
    level_type(233, "aircraft", MEAS_AIRCRAFT_LEVEL, None, false),
    level_type(234, "aircraft", MEAS_AIRCRAFT_LEVEL, None, false),
    level_type(235, "aircraft", MEAS_AIRCRAFT_LEVEL, None, false),
    level_type(223, "profiler", MEAS_PROFILER_LEVEL, None, false),
    level_type(227, "profiler", MEAS_PROFILER_LEVEL, None, false),
    level_type(228, "profiler", MEAS_PROFILER_LEVEL, None, false),
    level_type(229, "profiler", MEAS_PROFILER_LEVEL, None, false),
    level_type(224, "vad", MEAS_VAD_LEVEL, None, false),
    land(181, Some(MEAS_STATION_PRESSURE)),
    land(281, None),
    land(187, Some(MEAS_STATION_PRESSURE_FROM_ALTIMETER)),
    land(287, None),
    land(188, Some(MEAS_STATION_PRESSURE)),
    land(288, None),
    only("ADPSFC", land(183, Some(MEAS_STATION_PRESSURE_FROM_SEA_LEVEL))),
    only("ADPSFC", land(284, None)),
    land(192, None),
    land(292, None),
    land(193, None),
    land(293, None),
    land(195, None),
    land(295, None),
    marine(180, Some(MEAS_STATION_PRESSURE)),
    marine(280, None),
    marine(282, None),
    only("SFCSHP", marine(183, Some(MEAS_STATION_PRESSURE_FROM_SEA_LEVEL))),
    only("SFCSHP", marine(284, None)),
    marine(194, None),
    marine(294, None),
];

/// The upper-air moisture rule of GSI's `i_gsdqc = 2` setting, which the
/// regional rapid-refresh configuration runs (`read_prepbufr.f90`, the
/// block after the virtual-temperature loop): a humidity from these report
/// types between 300 and 10 hPa whose mark is 9 (NCEP's "not used above
/// 300 hPa") is taken with mark 2.
pub const UPPER_MOISTURE_REPORT_TYPES: [u16; 4] = [120, 131, 133, 134];
pub const UPPER_MOISTURE_LAYER_HPA: (f64, f64) = (10.0, 300.0);
pub const UPPER_MOISTURE_MARK_FROM: i64 = 9;
pub const UPPER_MOISTURE_MARK_TO: i64 = 2;
/// The most levels GSI reads from one report (`ufbint` with 255).
pub const GSI_LEVEL_LIMIT: usize = 255;

/// GSI's read of radar VAD winds (`read_prepbufr.f90`, the "new vad wind"
/// blocks).  A file is a new-VAD file when any type-224 report has subtype
/// (`TSB`) 2 or two consecutive levels exactly 50 m apart; GSI then reads
/// only the subtype-2 reports, only those whose `|DHR|` falls in one of six
/// windows, and from each only every sixth level, where it forms a superob
/// of that level and the next five: levels within 301 m of it are averaged,
/// and the superob is refused when the level departs from the file's own
/// background (`UFC`, `VFC`) by more than 10 m/s, or by more than 8 m/s in
/// v, or by more than 5 m/s in v below 5000 m, or sits above 7000 m, or
/// when any of the six departs from the average by more than 5 m/s, or
/// fewer than three were averaged.  A superob faster than 60 m/s ends the
/// report, as does a level whose six do not fit in the report.  These are
/// thinning and quality rules that need nothing but the report, so they
/// run here; `--raw-vad` writes every level instead.
pub const VAD_REPORT_TYPE: u16 = 224;
pub const VAD_NEW_SUBTYPE: f64 = 2.0;
pub const VAD_NEW_LEVEL_STEP_M: f64 = 50.0;
pub const VAD_TIME_WINDOWS_H: [(f64, f64); 6] =
    [(0.17, 0.32), (0.67, 0.82), (1.17, 1.32), (1.67, 1.82), (2.17, 2.62), (2.67, 2.82)];
pub const VAD_SAMPLE_EVERY: usize = 6;
pub const VAD_SUPEROB_LEVELS: usize = 6;
pub const VAD_SUPEROB_DEPTH_M: f64 = 301.0;
pub const VAD_MAX_DEPARTURE_M_S: f64 = 10.0;
pub const VAD_MAX_V_DEPARTURE_M_S: f64 = 8.0;
pub const VAD_MAX_V_DEPARTURE_LOW_M_S: f64 = 5.0;
pub const VAD_LOW_HEIGHT_M: f64 = 5000.0;
pub const VAD_MAX_HEIGHT_M: f64 = 7000.0;
pub const VAD_MAX_SPREAD_M_S: f64 = 5.0;
pub const VAD_MIN_MEMBERS: usize = 3;
pub const VAD_MAX_SPEED_M_S: f64 = 60.0;
/// GSI's regional rule for multi-agency profiler winds: above 400 hPa they
/// are monitored, not used (`read_prepbufr.f90`, the usage block).
pub const PROFILER_MAP_REPORT_TYPE: u16 = 227;
pub const PROFILER_MAP_TOP_HPA: f64 = 400.0;

/// GSI's surface rules under `i_gsdqc = 2` ("filter bad 2-m dew point and
/// 0 mesonet wind obs", `read_prepbufr.f90:1897-1906` of the HRRR v4.1.21
/// GSI; the analysis sets `i_gsdqc=2`, `parm/conus/hrrr_gsiparm.anl.sh`).
/// A humidity of a surface type 180 to 189 is not used when its reported
/// dewpoint (`TDO`, C) is below `min(-40, TOB - 10)` (usage 116), when
/// `TOB - TDO` exceeds 70 (117), or when `TDO` exceeds 32.2 (118); the
/// later test wins, as GSI assigns them in that order, and a missing `TDO`
/// or `TOB` is the library's missing value (so a missing `TDO` is 118 and a
/// missing `TOB` beside a reported `TDO` is 117).  A type-288 wind whose
/// components are both under 0.01 m/s is not used (115).
pub const SURFACE_HUMIDITY_RULE_TYPES: (u16, u16) = (180, 189);
pub const SURFACE_DEWPOINT_FLOOR_C: f64 = -40.0;
pub const SURFACE_DEWPOINT_UNDER_TEMPERATURE_C: f64 = 10.0;
pub const SURFACE_MAX_DEPRESSION_C: f64 = 70.0;
pub const SURFACE_MAX_DEWPOINT_C: f64 = 32.2;
pub const CALM_WIND_REPORT_TYPE: u16 = 288;
pub const CALM_WIND_M_S: f64 = 0.01;

/// GSI's usage code for a surface humidity under the `i_gsdqc = 2` rules,
/// `None` when they pass it.
pub fn surface_humidity_usage(tob_c: Option<f64>, tdo_c: Option<f64>) -> Option<u16> {
    let (t, td) = (tob_c.unwrap_or(BMISS), tdo_c.unwrap_or(BMISS));
    let mut usage = None;
    if td < SURFACE_DEWPOINT_FLOOR_C.min(t - SURFACE_DEWPOINT_UNDER_TEMPERATURE_C) {
        usage = Some(116);
    }
    if t - td > SURFACE_MAX_DEPRESSION_C {
        usage = Some(117);
    }
    if td > SURFACE_MAX_DEWPOINT_C {
        usage = Some(118);
    }
    usage
}

pub fn surface_humidity_rule_name(usage: u16) -> &'static str {
    match usage {
        116 => "usage_116_dewpoint_under_minus_40_c_and_10_k_under_t",
        117 => "usage_117_t_minus_dewpoint_over_70_k",
        _ => "usage_118_dewpoint_over_32_2_c_or_missing",
    }
}

pub const CALM_WIND_RULE: &str = "usage_115_mesonet_wind_under_0_01_m_s";

/// GSI's error inflation from the observation alone (`read_prepbufr.f90`,
/// `inflate_error` and the lines after it): 1.2 for a value whose own mark
/// is 3 or 7, 1.2 again for a temperature above 100 hPa, and for a wind
/// above 50 hPa.
pub const ERROR_INFLATION: f64 = 1.2;
pub const ERROR_INFLATION_MARKS: [i64; 2] = [3, 7];
pub const TEMPERATURE_INFLATION_TOP_HPA: f64 = 100.0;
pub const WIND_INFLATION_TOP_HPA: f64 = 50.0;
/// An error at or past this gives a value no weight: GSI's own "bad data"
/// error (`qcmod.f90`, `errormod`: `errout=1.e6_r_kind`), and what a table
/// fill (1e9) interpolates to.  Such a value is counted, not written.
pub const NO_WEIGHT_ERROR: f64 = 1.0e6;

pub fn report_type_row(report_type: u16, message_type: &str) -> Option<&'static ReportType> {
    REPORT_TYPES
        .iter()
        .find(|row| row.report_type == report_type && row.message_type.map(|m| m == message_type).unwrap_or(true))
}

// ------------------------------------------------------------ conversions

pub fn round_to(value: f64, decimals: i32) -> f64 {
    let factor = 10f64.powi(decimals);
    (value * factor).round() / factor
}

/// Bolton (1980) saturation vapour pressure over water, hPa, at a
/// temperature in K.
pub fn saturation_vapour_pressure_hpa(temperature_k: f64) -> f64 {
    let t = temperature_k - CELSIUS_TO_KELVIN;
    6.112 * (17.67 * t / (t + 243.5)).exp()
}

/// The dewpoint (K) of a specific humidity (kg/kg) at a pressure (Pa);
/// `None` where the humidity is not positive.
pub fn dewpoint_k(specific_humidity: f64, pressure_pa: f64) -> Option<f64> {
    if !(specific_humidity > 0.0) || !(pressure_pa > 0.0) {
        return None;
    }
    let e_hpa = specific_humidity * pressure_pa / (EPSILON + (1.0 - EPSILON) * specific_humidity) / 100.0;
    let x = (e_hpa / 6.112).ln();
    let t = 243.5 * x / (17.67 - x);
    Some(t + CELSIUS_TO_KELVIN)
}

/// The sensible temperature of a virtual one, with the report's own humidity.
pub fn sensible_from_virtual_k(virtual_k: f64, specific_humidity: f64) -> f64 {
    virtual_k / (1.0 + VIRTUAL_FACTOR * specific_humidity)
}

/// A relative-humidity error (`QOE`, tenths of the saturated value) as a
/// dewpoint error: the dewpoint of the humidity raised by that share of
/// saturation at the level's temperature, minus the dewpoint itself.
pub fn dewpoint_error_k(qoe: f64, specific_humidity: f64, temperature_k: f64, pressure_pa: f64) -> Option<f64> {
    let es = saturation_vapour_pressure_hpa(temperature_k) * 100.0;
    if !(es > 0.0) || es >= pressure_pa {
        return None;
    }
    let saturated = EPSILON * es / (pressure_pa - (1.0 - EPSILON) * es);
    let raised = specific_humidity + 0.1 * qoe * saturated;
    let error = dewpoint_k(raised, pressure_pa)? - dewpoint_k(specific_humidity, pressure_pa)?;
    (error.is_finite() && error > 0.0).then_some(error)
}

pub fn wrap_longitude(lon_deg: f64) -> f64 {
    crate::seam::wrap_longitude(lon_deg)
}

/// `DHR` hours from the cycle to an instant, to the second.
pub fn valid_time(cycle: DateTime<Utc>, hours: f64) -> Option<DateTime<Utc>> {
    if !hours.is_finite() || hours.abs() > 240.0 {
        return None;
    }
    cycle.checked_add_signed(Duration::seconds((hours * 3600.0).round() as i64))
}

// ----------------------------------------------------------------- counts

type ByType<T> = BTreeMap<String, T>;

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct VirtualCounts {
    pub seen: usize,
    pub made_sensible_with_report_humidity: usize,
    pub not_written_humidity_missing: usize,
    pub not_written_humidity_mark: BTreeMap<String, usize>,
}

/// What the door did, each a count.  Keys are report types and marks as
/// text so the record sorts the same way every run.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Counts {
    pub reports_by_report_type: ByType<usize>,
    pub levels_by_report_type: ByType<usize>,
    pub rows_by_report_type: ByType<BTreeMap<String, usize>>,
    pub rows_by_measurement: BTreeMap<String, usize>,
    /// Report type, then variable, then the value's own mark: not written.
    pub rejected_by_mark: ByType<BTreeMap<String, BTreeMap<String, usize>>>,
    /// Report type, then variable, then the level's pressure mark: the
    /// value's own mark passed and the pressure's did not.
    pub rejected_by_pressure_mark: ByType<BTreeMap<String, BTreeMap<String, usize>>>,
    /// Surface pressures refused by GSI's other tests, by report type and reason.
    pub rejected_surface_pressure: ByType<BTreeMap<String, usize>>,
    /// Rows whose error is the error table's, by variable.
    pub error_from_table: BTreeMap<String, usize>,
    /// Rows whose error is the file's own, by variable.
    pub error_from_file: BTreeMap<String, usize>,
    /// Table errors raised to GSI's floor, by variable.
    pub error_at_table_floor: BTreeMap<String, usize>,
    /// Errors raised by 1.2: `mark_3_or_7:<variable>`,
    /// `temperature_above_100_hpa`, `wind_above_50_hpa`.
    pub error_inflated: BTreeMap<String, usize>,
    /// Values without an error that carries weight (no file error under the
    /// file source; a table error at or past 1e6 under the table), by report
    /// type and variable: not written.
    pub not_written_no_error: ByType<BTreeMap<String, usize>>,
    /// Rows written from a report of more than one level, by report type:
    /// GSI multiplies their error by `errormod`'s factor (at least 1, from
    /// the background's layer depth), which is not applied here.
    pub rows_without_errormod_factor: ByType<usize>,
    /// Which temperature carried a humidity error into dewpoint:
    /// `checked_temperature` (the level's sensible temperature, its marks
    /// passed) or `unchecked_temperature` (the level's `TOB` when its marks
    /// did not pass; it sets the scale of the error only).
    pub dewpoint_error_temperature: BTreeMap<String, usize>,
    pub virtual_temperatures: VirtualCounts,
    pub reports_not_mapped_by_report_type: ByType<usize>,
    pub reports_skipped: BTreeMap<String, usize>,
    pub values_skipped: BTreeMap<String, usize>,
    pub level_rows_stamped_with_isa_altitude: usize,
    pub levels_placed_by_balloon_drift: usize,
    /// Humidity marks of 9 taken as 2 by the upper-air moisture rule.
    pub humidity_mark_9_taken_as_2: usize,
    /// Surface-level pressures with a passing mark that the report type
    /// carries no pressure row for, by report type.
    pub surface_pressure_not_written_by_report_type: ByType<usize>,
    /// Levels past the 255 GSI reads, by report type (written here).
    pub levels_past_gsi_read_limit: ByType<usize>,
    /// Values GSI's type rules monitor rather than use, by report type and rule.
    pub not_used_by_type_rule: ByType<BTreeMap<String, usize>>,
    pub vad: VadCounts,
}

/// What GSI's VAD read did to the type-224 reports.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct VadCounts {
    pub superob_rule_applied: bool,
    pub file_is_new_vad: bool,
    pub reports_seen: usize,
    pub reports_of_the_other_subtype: usize,
    pub reports_outside_the_time_windows: usize,
    pub levels_not_sampled: usize,
    pub levels_without_a_full_window: usize,
    pub rejected_background_departure: usize,
    pub rejected_v_departure: usize,
    pub rejected_above_7000_m: usize,
    pub rejected_v_departure_below_5000_m: usize,
    pub rejected_spread: usize,
    pub rejected_fewer_than_3: usize,
    pub reports_ended_by_a_superob_over_60_m_s: usize,
    pub levels_after_a_report_ended: usize,
    pub superobs_formed: usize,
}

fn bump(map: &mut BTreeMap<String, usize>, key: impl Into<String>) {
    *map.entry(key.into()).or_insert(0) += 1;
}

fn mark_text(mark: Option<i64>) -> String {
    match mark {
        Some(m) => m.to_string(),
        None => "missing".to_string(),
    }
}

/// GSI's rounding of a mark (`nint`); a missing mark stays missing.
pub fn mark_of(value: Option<f64>) -> Option<i64> {
    value.filter(|v| v.is_finite()).map(|v| v.round() as i64)
}

/// The use rule on one mark: 0 to 3 is used.
pub fn mark_passes(mark: Option<i64>) -> bool {
    matches!(mark, Some(m) if (0..MARK_LIMIT).contains(&m))
}

// ------------------------------------------------------------------- rows

/// Where a row's error comes from.
#[derive(Debug, Clone, Copy)]
pub enum ErrorSource<'a> {
    /// GSI with an error table (the operational path): the table's error
    /// for the report type at the level's pressure, floored.
    Table(&'a ErrorTable),
    /// GSI without one: the file's own `POE`, `TOE`, `QOE`, `WOE`; a value
    /// without one is counted and not written.
    File,
}

impl ErrorSource<'_> {
    pub fn name(&self) -> &'static str {
        match self {
            ErrorSource::Table(_) => "error-table",
            ErrorSource::File => "file",
        }
    }
}

pub struct RowContext<'a> {
    pub provenance: &'a RowProvenance,
    pub errors: ErrorSource<'a>,
    /// GSI's VAD superob read (default); false writes every VAD level.
    pub vad_superob: bool,
}

/// What GSI decides once per file before it reads a report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FileRules {
    pub new_vad: bool,
}

fn value(x: Option<f64>) -> f64 {
    x.unwrap_or(BMISS)
}

/// GSI's new-VAD test over the type-224 reports of a file.
pub fn report_is_new_vad(report: &Report) -> bool {
    if report.tsb == Some(VAD_NEW_SUBTYPE) {
        return true;
    }
    let levels = &report.levels[..report.levels.len().min(GSI_LEVEL_LIMIT)];
    levels.windows(2).any(|pair| (value(pair[1].obs[ZOB]) - value(pair[0].obs[ZOB])).abs() == VAD_NEW_LEVEL_STEP_M)
}

pub fn in_vad_time_window(dhr: Option<f64>) -> bool {
    let hours = value(dhr).abs();
    VAD_TIME_WINDOWS_H.iter().any(|&(low, high)| hours > low && hours < high)
}

/// GSI's VAD superobs of one report: per level, the superob formed there
/// (u, v, height), and the level at which GSI stopped reading the report.
pub fn vad_superobs(report: &Report, counts: &mut VadCounts) -> (Vec<Option<(f64, f64, f64)>>, usize) {
    let levs = report.levels.len().min(GSI_LEVEL_LIMIT);
    let mut out = vec![None; report.levels.len()];
    let selev = value(report.elv);
    let u = |i: usize| value(report.levels[i].obs[UOB]);
    let v = |i: usize| value(report.levels[i].obs[VOB]);
    let z = |i: usize| value(report.levels[i].obs[ZOB]);
    for k in 0..levs {
        let number = k + 1;
        if number % VAD_SAMPLE_EVERY != 0 {
            counts.levels_not_sampled += 1;
            continue;
        }
        let last = k + VAD_SUPEROB_LEVELS - 1;
        if last >= levs {
            counts.levels_without_a_full_window += levs - k;
            return (out, k);
        }
        let level = &report.levels[k];
        // a profile level at or below the station is placed 10 m above it
        let mut height = z(k);
        if selev >= height {
            height = 10.0 + selev;
        }
        let du = u(k) - value(level.background[0]);
        let dv = v(k) - value(level.background[1]);
        if (du * du + dv * dv).sqrt() > VAD_MAX_DEPARTURE_M_S {
            counts.rejected_background_departure += 1;
            continue;
        }
        if dv.abs() > VAD_MAX_V_DEPARTURE_M_S {
            counts.rejected_v_departure += 1;
            continue;
        }
        if height > VAD_MAX_HEIGHT_M {
            counts.rejected_above_7000_m += 1;
            continue;
        }
        if dv.abs() > VAD_MAX_V_DEPARTURE_LOW_M_S && height < VAD_LOW_HEIGHT_M {
            counts.rejected_v_departure_below_5000_m += 1;
            continue;
        }
        let (mut su, mut sv, mut sz, mut members) = (0.0, 0.0, 0.0, 0usize);
        for i in k..=last {
            if z(i) - z(k) < VAD_SUPEROB_DEPTH_M {
                su += u(i);
                sv += v(i);
                sz += z(i);
                members += 1;
            }
        }
        let (mean_u, mean_v, mean_z) = (su / members as f64, sv / members as f64, sz / members as f64);
        let mut spread = 0.0f64;
        for i in k..=last {
            for departure in [(u(i) - mean_u).abs(), (v(i) - mean_v).abs()] {
                if spread < departure {
                    spread = departure;
                }
            }
        }
        if spread > VAD_MAX_SPREAD_M_S {
            counts.rejected_spread += 1;
            continue;
        }
        if members < VAD_MIN_MEMBERS {
            counts.rejected_fewer_than_3 += 1;
            continue;
        }
        if (mean_u * mean_u + mean_v * mean_v).sqrt() > VAD_MAX_SPEED_M_S {
            counts.reports_ended_by_a_superob_over_60_m_s += 1;
            counts.levels_after_a_report_ended += levs - k - 1;
            return (out, k);
        }
        counts.superobs_formed += 1;
        out[k] = Some((mean_u, mean_v, mean_z));
    }
    (out, levs)
}

fn station_id(raw: &str) -> String {
    raw.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' }).collect()
}

struct Target<'a> {
    counts: &'a mut Counts,
    writer: &'a mut TableWriter,
    by_variable: &'a mut BTreeMap<String, usize>,
    type_key: String,
    /// The report has more than one level (GSI's `errormod` would apply).
    profile: bool,
}

impl Target<'_> {
    fn reject_mark(&mut self, variable: &str, mark: Option<i64>) {
        bump(
            self.counts.rejected_by_mark.entry(self.type_key.clone()).or_default().entry(variable.to_string()).or_default(),
            mark_text(mark),
        );
    }

    fn reject_pressure_mark(&mut self, variable: &str, mark: Option<i64>) {
        bump(
            self.counts
                .rejected_by_pressure_mark
                .entry(self.type_key.clone())
                .or_default()
                .entry(variable.to_string())
                .or_default(),
            mark_text(mark),
        );
    }

    fn skip(&mut self, reason: &str) {
        bump(&mut self.counts.values_skipped, reason);
    }

    /// The row's error before inflation, in GSI's units (K, tenths of
    /// saturation, m/s, hPa), or `None` when the value is not written.
    fn base_error(
        &mut self,
        source: ErrorSource,
        report_type: u16,
        variable: &str,
        column: Column,
        pressure_hpa: Option<f64>,
        from_file: Option<f64>,
    ) -> Option<f64> {
        let error = match source {
            ErrorSource::Table(table) => {
                let raw = table.raw(report_type, pressure_hpa, column);
                if raw < NO_WEIGHT_ERROR && raw < column.floor() {
                    bump(&mut self.counts.error_at_table_floor, variable);
                }
                Some(raw.max(column.floor()))
            }
            ErrorSource::File => from_file.filter(|e| e.is_finite() && *e > 0.0),
        };
        match error {
            Some(error) if error < NO_WEIGHT_ERROR => {
                match source {
                    ErrorSource::Table(_) => bump(&mut self.counts.error_from_table, variable),
                    ErrorSource::File => bump(&mut self.counts.error_from_file, variable),
                }
                Some(error)
            }
            _ => {
                bump(self.counts.not_written_no_error.entry(self.type_key.clone()).or_default(), variable);
                None
            }
        }
    }

    /// GSI's inflation for a value whose own mark is 3 or 7.
    fn inflate_for_mark(&mut self, variable: &str, error: f64, mark: Option<i64>) -> f64 {
        match mark {
            Some(m) if ERROR_INFLATION_MARKS.contains(&m) => {
                bump(&mut self.counts.error_inflated, format!("mark_3_or_7:{variable}"));
                error * ERROR_INFLATION
            }
            _ => error,
        }
    }

    /// GSI's inflation above a pressure (`rule` names it in the counts).
    fn inflate_above(&mut self, rule: &str, error: f64, pressure_hpa: Option<f64>, top_hpa: f64) -> f64 {
        if pressure_hpa.map(|p| p < top_hpa).unwrap_or(false) {
            bump(&mut self.counts.error_inflated, rule);
            error * ERROR_INFLATION
        } else {
            error
        }
    }

    fn push(&mut self, row: TableRow, bounds: (f64, f64)) {
        if !row.value.is_finite() || row.value < bounds.0 || row.value > bounds.1 {
            self.skip(&format!("outside_gross_bounds:{}", row.variable));
            return;
        }
        bump(self.counts.rows_by_report_type.entry(self.type_key.clone()).or_default(), row.variable.clone());
        bump(&mut self.counts.rows_by_measurement, row.provenance.measurement);
        if self.profile && row.variable != VAR_SURFACE_PRESSURE {
            *self.counts.rows_without_errormod_factor.entry(self.type_key.clone()).or_insert(0) += 1;
        }
        self.writer.push(row, self.by_variable);
    }
}

/// Write one report's rows under the use rule.
pub fn rows(
    report: &Report,
    context: &RowContext,
    rules: FileRules,
    counts: &mut Counts,
    writer: &mut TableWriter,
    by_variable: &mut BTreeMap<String, usize>,
) {
    let Some(typ) = report.typ.filter(|t| t.is_finite() && *t >= 0.0 && *t < 65_536.0).map(|t| t.round() as u16) else {
        bump(&mut counts.reports_skipped, "no_report_type");
        return;
    };
    let type_key = typ.to_string();
    bump(&mut counts.reports_by_report_type, type_key.clone());
    *counts.levels_by_report_type.entry(type_key.clone()).or_insert(0) += report.levels.len();
    let Some(kind) = report_type_row(typ, &report.message_type) else {
        bump(&mut counts.reports_not_mapped_by_report_type, type_key);
        return;
    };
    let Some(station) = report.station.as_deref().map(station_id) else {
        bump(&mut counts.reports_skipped, "no_station_id");
        return;
    };
    let (Some(xob), Some(yob)) = (report.xob, report.yob) else {
        bump(&mut counts.reports_skipped, "no_position");
        return;
    };
    if !(-90.0..=90.0).contains(&yob) || !(-180.0..=360.0).contains(&xob) {
        bump(&mut counts.reports_skipped, "position_off_the_sphere");
        return;
    }
    let Some(report_time) = report.dhr.and_then(|hours| valid_time(report.cycle, hours)) else {
        bump(&mut counts.reports_skipped, "no_time");
        return;
    };
    let nominal = context.provenance.nominal(report.cycle);
    if report.levels.len() > GSI_LEVEL_LIMIT {
        *counts.levels_past_gsi_read_limit.entry(type_key.clone()).or_insert(0) += report.levels.len() - GSI_LEVEL_LIMIT;
    }
    let upper_moisture = UPPER_MOISTURE_REPORT_TYPES.contains(&typ);
    // GSI's VAD read: which reports, which levels, and the superobs.
    let vad = typ == VAD_REPORT_TYPE && context.vad_superob;
    let (mut superobs, mut stop_at) = (Vec::new(), report.levels.len());
    if vad {
        counts.vad.reports_seen += 1;
        if rules.new_vad != (report.tsb == Some(VAD_NEW_SUBTYPE)) {
            counts.vad.reports_of_the_other_subtype += 1;
            return;
        }
        if rules.new_vad {
            if !in_vad_time_window(report.dhr) {
                counts.vad.reports_outside_the_time_windows += 1;
                return;
            }
            (superobs, stop_at) = vad_superobs(report, &mut counts.vad);
        }
    }
    let new_vad = vad && rules.new_vad;
    let profile = report.levels.len().min(GSI_LEVEL_LIMIT) > 1;
    let mut target = Target { counts, writer, by_variable, type_key, profile };
    for (index, level) in report.levels.iter().enumerate() {
        if new_vad && index >= stop_at {
            break;
        }
        let pressure_hpa = level.obs[POB];
        let pressure_mark = mark_of(level.marks[PQM]);
        // Where and when this level is: a sounding level drifts with the
        // balloon when the file says where to (GSI's own sanity rules).
        let (mut lon, mut lat, mut time) = (xob, yob, report_time);
        if kind.drift {
            if let (Some(xdr), Some(ydr)) = (level.drift[0], level.drift[1]) {
                let sane = ydr.abs() <= 90.0 && (0.0..=360.0).contains(&xdr) && !((ydr - yob).abs() > 10.0 && (xdr - xob).abs() > 10.0);
                if sane {
                    lon = xdr;
                    lat = ydr;
                    if let (Some(hrdr), Some(dhr)) = (level.drift[2], report.dhr) {
                        if (hrdr - dhr).abs() <= 4.0 {
                            if let Some(drifted) = valid_time(report.cycle, hrdr) {
                                time = drifted;
                            }
                        }
                    }
                    target.counts.levels_placed_by_balloon_drift += 1;
                }
            }
        }
        // Surface pressure: GSI's tests for a pressure observation.
        if let (None, Some(_), Some(0)) = (kind.surface_pressure, pressure_hpa, mark_of(level.obs[CAT])) {
            if typ < 200 && mark_passes(pressure_mark) {
                bump(&mut target.counts.surface_pressure_not_written_by_report_type, target.type_key.clone());
            }
        }
        if let (Some(measurement), Some(pob)) = (kind.surface_pressure, pressure_hpa) {
            if mark_of(level.obs[CAT]) == Some(0) {
                let height_mark = mark_of(level.marks[ZQM]);
                let elevation = level.obs[ZOB].or(report.elv);
                if !mark_passes(pressure_mark) {
                    target.reject_mark(VAR_SURFACE_PRESSURE, pressure_mark);
                } else if pob < 500.0 {
                    bump(target.counts.rejected_surface_pressure.entry(target.type_key.clone()).or_default(), "below_500_hpa");
                } else if !mark_passes(height_mark) && height_mark != Some(15) && height_mark != Some(9) {
                    bump(
                        target.counts.rejected_surface_pressure.entry(target.type_key.clone()).or_default(),
                        format!("height_mark_{}", mark_text(height_mark)),
                    );
                } else if let Some(elevation) = elevation {
                    let base = target.base_error(context.errors, typ, VAR_SURFACE_PRESSURE, Column::SurfacePressure, pressure_hpa, level.errors[POE]);
                    if let Some(base) = base {
                        let error = target.inflate_for_mark(VAR_SURFACE_PRESSURE, base, pressure_mark) * 100.0;
                        target.push(
                            TableRow {
                                source: SOURCE.into(),
                                station_id: station.clone(),
                                latitude_deg: lat,
                                longitude_deg: wrap_longitude(lon),
                                elevation_m: elevation,
                                level_pa: None,
                                valid_time: time,
                                variable: VAR_SURFACE_PRESSURE.into(),
                                value: round_to(pob * 100.0, 1),
                                error: round_to(error, 3),
                                provenance: nominal.measuring(measurement),
                            },
                            GROSS_SURFACE_PRESSURE_PA,
                        );
                    }
                } else {
                    target.skip("surface_pressure_without_elevation");
                }
            }
        }
        // Where a temperature, humidity or wind row is anchored.
        let anchor = match kind.anchor {
            Anchor::Surface => report.elv.or(level.obs[ZOB]).map(|elevation| (elevation, None, false)),
            Anchor::Level => pressure_hpa.map(|p| match level.obs[ZOB] {
                Some(z) => (z, Some(round_to(p * 100.0, 1)), false),
                None => (round_to(isa_altitude_m(p * 100.0), 1), Some(round_to(p * 100.0, 1)), true),
            }),
        };
        let surface = kind.anchor == Anchor::Surface;
        let humidity = level.obs[QOB].map(|q| q * 1.0e-6);
        let mut humidity_mark = mark_of(level.marks[QQM]);
        let in_upper_layer = pressure_hpa
            .map(|p| (UPPER_MOISTURE_LAYER_HPA.0..=UPPER_MOISTURE_LAYER_HPA.1).contains(&p))
            .unwrap_or(false);
        if upper_moisture && humidity.is_some() && in_upper_layer && humidity_mark == Some(UPPER_MOISTURE_MARK_FROM) {
            humidity_mark = Some(UPPER_MOISTURE_MARK_TO);
            target.counts.humidity_mark_9_taken_as_2 += 1;
        }
        let temperature_mark = mark_of(level.marks[TQM]);
        // The sensible temperature this level would write, used again for
        // the humidity error.
        let mut sensible_k: Option<f64> = None;
        let row = |target: &mut Target, variable: &str, measurement: &'static str, value: f64, error: f64, bounds: (f64, f64)| {
            let Some((elevation, level_pa, isa)) = anchor else {
                target.skip(if surface { "surface_value_without_elevation" } else { "level_value_without_pressure" });
                return;
            };
            if isa {
                target.counts.level_rows_stamped_with_isa_altitude += 1;
            }
            target.push(
                TableRow {
                    source: SOURCE.into(),
                    station_id: station.clone(),
                    latitude_deg: lat,
                    longitude_deg: wrap_longitude(lon),
                    elevation_m: elevation,
                    level_pa,
                    valid_time: time,
                    variable: variable.into(),
                    value,
                    error: round_to(error, 3),
                    provenance: nominal.measuring(measurement),
                },
                bounds,
            );
        };
        if let Some(tob) = level.obs[TOB] {
            if !mark_passes(temperature_mark) {
                target.reject_mark(VAR_TEMPERATURE, temperature_mark);
            } else if !mark_passes(pressure_mark) {
                target.reject_pressure_mark(VAR_TEMPERATURE, pressure_mark);
            } else {
                let kelvin = tob + CELSIUS_TO_KELVIN;
                let value = if level.is_virtual {
                    target.counts.virtual_temperatures.seen += 1;
                    match humidity {
                        Some(q) if mark_passes(humidity_mark) => {
                            target.counts.virtual_temperatures.made_sensible_with_report_humidity += 1;
                            Some(sensible_from_virtual_k(kelvin, q))
                        }
                        Some(_) => {
                            bump(&mut target.counts.virtual_temperatures.not_written_humidity_mark, mark_text(humidity_mark));
                            None
                        }
                        None => {
                            target.counts.virtual_temperatures.not_written_humidity_missing += 1;
                            None
                        }
                    }
                } else {
                    Some(kelvin)
                };
                if let Some(value) = value {
                    sensible_k = Some(value);
                    let base = target.base_error(context.errors, typ, VAR_TEMPERATURE, Column::Temperature, pressure_hpa, level.errors[TOE]);
                    if let Some(base) = base {
                        let error = target.inflate_for_mark(VAR_TEMPERATURE, base, temperature_mark);
                        let error = target.inflate_above("temperature_above_100_hpa", error, pressure_hpa, TEMPERATURE_INFLATION_TOP_HPA);
                        row(&mut target, VAR_TEMPERATURE, kind.temperature, round_to(value, 2), error, GROSS_TEMPERATURE_K);
                    }
                }
            }
        }
        if let Some(q) = humidity {
            let surface_rule = if (SURFACE_HUMIDITY_RULE_TYPES.0..=SURFACE_HUMIDITY_RULE_TYPES.1).contains(&typ) {
                surface_humidity_usage(level.obs[TOB], level.obs[TDO])
            } else {
                None
            };
            if !mark_passes(humidity_mark) {
                target.reject_mark(VAR_DEWPOINT, humidity_mark);
            } else if !mark_passes(pressure_mark) {
                target.reject_pressure_mark(VAR_DEWPOINT, pressure_mark);
            } else if let Some(usage) = surface_rule {
                bump(target.counts.not_used_by_type_rule.entry(target.type_key.clone()).or_default(), surface_humidity_rule_name(usage));
            } else {
                match pressure_hpa.and_then(|p| dewpoint_k(q, p * 100.0).map(|td| (td, p * 100.0))) {
                    Some((dewpoint, pressure_pa)) => {
                        // the temperature that carries a humidity error into dewpoint
                        let temperature = match (sensible_k, level.obs[TOB]) {
                            (Some(t), _) => Some(("checked_temperature", t)),
                            (None, Some(tob)) => Some(("unchecked_temperature", tob + CELSIUS_TO_KELVIN)),
                            (None, None) => None,
                        };
                        let base = target.base_error(context.errors, typ, VAR_DEWPOINT, Column::Humidity, pressure_hpa, level.errors[QOE]);
                        match (base, temperature) {
                            (None, _) => {}
                            (Some(_), None) => target.skip("dewpoint_error_without_temperature"),
                            (Some(base), Some((which, t))) => {
                                let tenths = target.inflate_for_mark(VAR_DEWPOINT, base, humidity_mark);
                                match dewpoint_error_k(tenths, q, t, pressure_pa) {
                                    Some(error) => {
                                        bump(&mut target.counts.dewpoint_error_temperature, which);
                                        row(&mut target, VAR_DEWPOINT, kind.dewpoint, round_to(dewpoint, 2), error, GROSS_DEWPOINT_K);
                                    }
                                    None => target.skip("dewpoint_error_not_computable"),
                                }
                            }
                        }
                    }
                    None => target.skip("humidity_without_pressure_or_not_positive"),
                }
            }
        }
        // A VAD level of a new-VAD file is read only where GSI formed a
        // superob, and carries the superob's wind and height.
        let (wind, wind_measurement, superob_height) = if new_vad {
            match superobs.get(index).copied().flatten() {
                Some((u, v, height)) => ((Some(u), Some(v)), MEAS_VAD_SUPEROB, Some(height)),
                None => continue,
            }
        } else {
            ((level.obs[UOB], level.obs[VOB]), kind.wind, None)
        };
        match wind {
            (Some(u), Some(v)) => {
                let wind_mark = mark_of(level.marks[WQM]);
                if !mark_passes(wind_mark) {
                    target.reject_mark("wind", wind_mark);
                } else if !mark_passes(pressure_mark) {
                    target.reject_pressure_mark("wind", pressure_mark);
                } else if typ == PROFILER_MAP_REPORT_TYPE && pressure_hpa.map(|p| p < PROFILER_MAP_TOP_HPA).unwrap_or(false) {
                    bump(
                        target.counts.not_used_by_type_rule.entry(target.type_key.clone()).or_default(),
                        "wind_above_400_hpa",
                    );
                } else if typ == CALM_WIND_REPORT_TYPE && u.abs() < CALM_WIND_M_S && v.abs() < CALM_WIND_M_S {
                    bump(target.counts.not_used_by_type_rule.entry(target.type_key.clone()).or_default(), CALM_WIND_RULE);
                } else if (u * u + v * v).sqrt() > GROSS_WIND_M_S.1 {
                    target.skip("outside_gross_bounds:wind_speed");
                } else {
                    let base = target.base_error(context.errors, typ, "wind", Column::Wind, pressure_hpa, level.errors[WOE]);
                    if let Some(error) = base.map(|base| {
                        let error = target.inflate_for_mark("wind", base, wind_mark);
                        target.inflate_above("wind_above_50_hpa", error, pressure_hpa, WIND_INFLATION_TOP_HPA)
                    }) {
                        let components = (-GROSS_WIND_M_S.1, GROSS_WIND_M_S.1);
                        match (superob_height, pressure_hpa) {
                            (Some(height), Some(p)) => {
                                for (variable, value) in [(VAR_WIND_U, u), (VAR_WIND_V, v)] {
                                    target.push(
                                        TableRow {
                                            source: SOURCE.into(),
                                            station_id: station.clone(),
                                            latitude_deg: lat,
                                            longitude_deg: wrap_longitude(lon),
                                            elevation_m: round_to(height, 1),
                                            level_pa: Some(round_to(p * 100.0, 1)),
                                            valid_time: time,
                                            variable: variable.into(),
                                            value: round_to(value, 2),
                                            error: round_to(error, 3),
                                            provenance: nominal.measuring(wind_measurement),
                                        },
                                        components,
                                    );
                                }
                            }
                            (Some(_), None) => target.skip("level_value_without_pressure"),
                            (None, _) => {
                                row(&mut target, VAR_WIND_U, wind_measurement, round_to(u, 2), error, components);
                                row(&mut target, VAR_WIND_V, wind_measurement, round_to(v, 2), error, components);
                            }
                        }
                    }
                }
            }
            (None, None) => {}
            _ => target.skip("wind_with_one_component"),
        }
    }
}

/// Every report of a file under the use rule: the rows and the counts.
pub fn table_of(file: &NcepFile, context: &RowContext) -> Result<(TableWriter, BTreeMap<String, usize>, Counts), Box<dyn Error>> {
    let mut writer = TableWriter::new();
    let mut by_variable = BTreeMap::new();
    let mut counts = Counts::default();
    let queries: Vec<Queries> = file.templates.iter().map(|(_, template)| Queries::of(template)).collect();
    let rules = file_rules(file, &queries)?;
    counts.vad.superob_rule_applied = context.vad_superob;
    counts.vad.file_is_new_vad = rules.new_vad;
    for message in &file.messages {
        let template = file.template_of(message);
        let cycle = cycle_time(message)?;
        let code = virtual_code(file, message)?;
        for subset in &file.subsets_of(message)? {
            let report = report(template, &queries[message.template], subset, cycle, code);
            rows(&report, context, rules, &mut counts, &mut writer, &mut by_variable);
        }
    }
    Ok((writer, by_variable, counts))
}

/// GSI's first pass: is any type-224 report a new-VAD one?
pub fn file_rules(file: &NcepFile, queries: &[Queries]) -> Result<FileRules, Box<dyn Error>> {
    let mut rules = FileRules::default();
    for message in &file.messages {
        let template = file.template_of(message);
        let q = &queries[message.template];
        let typ_node = q.header.nodes[4];
        let cycle = cycle_time(message)?;
        for subset in &file.subsets_of(message)? {
            let typ = subset.windows(&q.header).first().and_then(|w| subset.first(w, typ_node).number());
            if typ.map(|t| t.round() as i64) != Some(i64::from(VAD_REPORT_TYPE)) {
                continue;
            }
            if report_is_new_vad(&report(template, q, subset, cycle, f64::NAN)) {
                rules.new_vad = true;
                return Ok(rules);
            }
        }
    }
    Ok(rules)
}

/// Messages, subsets and levels by message type (in the dictionary's
/// spelling), and subsets by message type and report type
/// (`ADPUPA/120`), in one pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Census {
    pub messages_by_type: BTreeMap<String, usize>,
    pub subsets_by_type: BTreeMap<String, usize>,
    pub levels_by_type: BTreeMap<String, usize>,
    pub subsets_by_type_and_report_type: BTreeMap<String, usize>,
}

pub fn census(file: &NcepFile) -> Result<Census, Box<dyn Error>> {
    let mut out = Census::default();
    let levels: Vec<Query> = file.templates.iter().map(|(_, template)| template.query(&OBS_MNEMONICS)).collect();
    let types: Vec<Query> = file.templates.iter().map(|(_, template)| template.query(&["TYP"])).collect();
    for message in &file.messages {
        let name = file.template_of(message).mnemonic.clone();
        let subsets = file.subsets_of(message)?;
        *out.messages_by_type.entry(name.clone()).or_insert(0) += 1;
        *out.subsets_by_type.entry(name.clone()).or_insert(0) += subsets.len();
        let count: usize = subsets.iter().map(|s| s.windows(&levels[message.template]).len()).sum();
        *out.levels_by_type.entry(name.clone()).or_insert(0) += count;
        let query = &types[message.template];
        for subset in &subsets {
            let typ = subset
                .windows(query)
                .first()
                .and_then(|w| subset.first(w, query.nodes[0]).number())
                .map(|t| format!("{}", t.round() as i64))
                .unwrap_or_else(|| "missing".to_string());
            *out.subsets_by_type_and_report_type.entry(format!("{name}/{typ}")).or_insert(0) += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ncep_bufr::read_file;
    use crate::ncep_bufr::tests::{data_message, obstyp_subset, small_dictionary};

    fn cycle() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 10, 3, 12, 0, 0).unwrap()
    }

    fn level(pob: f64, cat: f64) -> Level {
        let mut level = Level::default();
        level.obs[POB] = Some(pob);
        level.obs[CAT] = Some(cat);
        level.marks[PQM] = Some(2.0);
        level
    }

    fn sonde(levels: Vec<Level>) -> Report {
        Report {
            message_type: "ADPUPA".into(),
            cycle: cycle(),
            station: Some("72365".into()),
            xob: Some(253.38),
            yob: Some(35.04),
            dhr: Some(-0.5),
            typ: Some(120.0),
            elv: Some(1620.0),
            tsb: None,
            levels,
        }
    }

    /// A table in GSI's format giving every mapped type the same errors at
    /// every level (temperature 1 K, humidity 2 tenths, wind 2.5 m/s,
    /// surface pressure 1 hPa), for the tests of the use rule.
    fn uniform_table() -> &'static ErrorTable {
        static TABLE: std::sync::OnceLock<ErrorTable> = std::sync::OnceLock::new();
        TABLE.get_or_init(|| {
            let mut text = String::new();
            let mut types: Vec<u16> = REPORT_TYPES.iter().map(|r| r.report_type).collect();
            types.dedup();
            for t in types {
                text.push_str(&format!(" {t:3} OBSERVATION TYPE\n"));
                for k in 0..33 {
                    let p = if k == 32 { 0.0 } else { 1100.0 - 33.0 * k as f64 };
                    text.push_str(&format!(" {p:12.5E}{:12.5E}{:12.5E}{:12.5E}{:12.5E}{:12.5E}\n", 1.0, 2.0, 2.5, 1.0, 1.0e9));
                }
            }
            ErrorTable::parse(&text, "uniform").unwrap()
        })
    }

    fn table() -> ErrorSource<'static> {
        ErrorSource::Table(uniform_table())
    }

    /// NCEP's published table, the default.
    fn published() -> ErrorSource<'static> {
        ErrorSource::Table(ErrorTable::default_table())
    }

    /// The published table's error for a type at a pressure, as a row carries it.
    fn table_error(report_type: u16, pressure_hpa: f64, column: Column) -> f64 {
        ErrorTable::default_table().error(report_type, Some(pressure_hpa), column)
    }

    fn run(report: &Report, errors: ErrorSource) -> (Vec<TableRow>, Counts) {
        let provenance = RowProvenance::of_source("0f6b607e7b578656f46c7968777e4ac9", None, None);
        let context = RowContext { provenance: &provenance, errors, vad_superob: true };
        let mut writer = TableWriter::new();
        let mut by_variable = BTreeMap::new();
        let mut counts = Counts::default();
        rows(report, &context, FileRules { new_vad: report_is_new_vad(report) }, &mut counts, &mut writer, &mut by_variable);
        (writer.rows().to_vec(), counts)
    }

    #[test]
    fn unit_conversions_are_the_stated_ones() {
        // XOB degrees east to -180..180
        assert!((wrap_longitude(253.38) + 106.62).abs() < 1e-9);
        assert_eq!(wrap_longitude(180.0), -180.0);
        assert_eq!(wrap_longitude(10.0), 10.0);
        // DHR hours from the cycle to an instant, to the second
        assert_eq!(valid_time(cycle(), -0.98333).unwrap(), Utc.with_ymd_and_hms(2026, 10, 3, 11, 1, 0).unwrap());
        assert_eq!(valid_time(cycle(), 0.0).unwrap(), cycle());
        assert_eq!(valid_time(cycle(), 1.5).unwrap(), Utc.with_ymd_and_hms(2026, 10, 3, 13, 30, 0).unwrap());
        assert!(valid_time(cycle(), f64::NAN).is_none());
        // QOB mg/kg to dewpoint at the level's own pressure: 10 g/kg at 1000 hPa
        // is a vapour pressure of 15.98 hPa, a dewpoint of 14.0 C
        let td = dewpoint_k(10_000.0 * 1.0e-6, 100_000.0).unwrap();
        assert!((td - 287.17).abs() < 0.05, "{td}");
        // the conversion inverts Bolton: saturation at the dewpoint is the vapour pressure
        let e = 0.010 * 1000.0 / (0.622 + 0.378 * 0.010);
        assert!((saturation_vapour_pressure_hpa(td) - e).abs() < 1e-9);
        // the same humidity at half the pressure has a lower dewpoint
        assert!(dewpoint_k(0.010, 50_000.0).unwrap() < td - 9.0);
        assert!(dewpoint_k(0.0, 100_000.0).is_none());
        // a virtual temperature with 10 g/kg is 0.61 percent above the sensible one
        assert!((sensible_from_virtual_k(300.0 * 1.0061, 0.010) - 300.0).abs() < 1e-9);
        // a 10 percent relative-humidity error near saturation at 20 C is about 1.6 K of dewpoint
        let error = dewpoint_error_k(1.0, 0.012, 293.15, 100_000.0).unwrap();
        assert!(error > 1.2 && error < 2.2, "{error}");
        assert_eq!(round_to(288.14999999, 2), 288.15);
        assert_eq!(mark_of(Some(2.0)), Some(2));
        assert_eq!(mark_of(None), None);
    }

    #[test]
    fn a_sounding_level_writes_rows_in_the_neutral_units() {
        let mut l = level(850.0, 1.0);
        l.obs[TOB] = Some(15.2);
        l.obs[QOB] = Some(8000.0);
        l.obs[ZOB] = Some(1500.0);
        l.obs[UOB] = Some(-3.1);
        l.obs[VOB] = Some(7.9);
        l.marks[TQM] = Some(2.0);
        l.marks[QQM] = Some(1.0);
        l.marks[WQM] = Some(2.0);
        let (rows, counts) = run(&sonde(vec![l]), table());
        assert_eq!(rows.len(), 4);
        let t = &rows[0];
        assert_eq!((t.variable.as_str(), t.value, t.error), (VAR_TEMPERATURE, 288.35, 1.0));
        assert_eq!((t.level_pa, t.elevation_m, t.provenance.measurement), (Some(85_000.0), 1500.0, MEAS_SONDE_LEVEL));
        assert_eq!(t.valid_time, Utc.with_ymd_and_hms(2026, 10, 3, 11, 30, 0).unwrap());
        assert_eq!(t.provenance.nominal_time, Some(cycle()));
        assert!((t.longitude_deg + 106.62).abs() < 1e-9);
        assert_eq!(t.source, "prepbufr");
        let td = &rows[1];
        assert_eq!(td.variable, VAR_DEWPOINT);
        assert!((td.value - dewpoint_k(0.008, 85_000.0).unwrap()).abs() < 0.006);
        assert_eq!((rows[2].variable.as_str(), rows[2].value), (VAR_WIND_U, -3.1));
        assert_eq!((rows[3].variable.as_str(), rows[3].value), (VAR_WIND_V, 7.9));
        assert_eq!(counts.error_from_table["temperature_k"], 1);
        assert_eq!(counts.rows_by_report_type["120"]["wind_u_m_s"], 1);
        assert!(counts.error_from_file.is_empty());
        // the humidity error is the table's tenths of saturation carried into dewpoint
        let expected = dewpoint_error_k(2.0, 0.008, 15.2 + CELSIUS_TO_KELVIN, 85_000.0).unwrap();
        assert_eq!(td.error, round_to(expected, 3));
        assert_eq!(counts.dewpoint_error_temperature["checked_temperature"], 1);
        // a single-level report: GSI's errormod factor is 1, so nothing is counted
        assert!(counts.rows_without_errormod_factor.is_empty());
        // the csv line is the fifteen-column v2 layout
        assert_eq!(t.csv_line().split(',').count(), 15);
    }

    #[test]
    fn a_mark_of_four_or_more_is_counted_and_never_written() {
        let mut l = level(700.0, 1.0);
        l.obs[TOB] = Some(2.0);
        l.obs[QOB] = Some(3000.0);
        l.obs[UOB] = Some(10.0);
        l.obs[VOB] = Some(0.0);
        // marks: temperature 4, humidity 9, wind 13
        l.marks[TQM] = Some(4.0);
        l.marks[QQM] = Some(9.0);
        l.marks[WQM] = Some(13.0);
        let (rows, counts) = run(&sonde(vec![l.clone()]), table());
        assert!(rows.is_empty());
        assert_eq!(counts.rejected_by_mark["120"]["temperature_k"]["4"], 1);
        assert_eq!(counts.rejected_by_mark["120"]["dewpoint_k"]["9"], 1);
        assert_eq!(counts.rejected_by_mark["120"]["wind"]["13"], 1);
        // a mark of 3 passes, a missing mark does not
        l.marks[TQM] = Some(3.0);
        l.marks[QQM] = None;
        l.marks[WQM] = Some(0.0);
        let (rows, counts) = run(&sonde(vec![l.clone()]), table());
        let variables: Vec<&str> = rows.iter().map(|r| r.variable.as_str()).collect();
        assert_eq!(variables, vec![VAR_TEMPERATURE, VAR_WIND_U, VAR_WIND_V]);
        assert_eq!(counts.rejected_by_mark["120"]["dewpoint_k"]["missing"], 1);
        // a good value on a level whose pressure mark is 4 or more is not used
        l.marks[PQM] = Some(8.0);
        l.marks[QQM] = Some(2.0);
        let (rows, counts) = run(&sonde(vec![l]), table());
        assert!(rows.is_empty());
        assert_eq!(counts.rejected_by_pressure_mark["120"]["temperature_k"]["8"], 1);
        assert_eq!(counts.rejected_by_pressure_mark["120"]["dewpoint_k"]["8"], 1);
        assert_eq!(counts.rejected_by_pressure_mark["120"]["wind"]["8"], 1);
        assert!(mark_passes(Some(0)) && mark_passes(Some(3)));
        assert!(!mark_passes(Some(4)) && !mark_passes(Some(15)) && !mark_passes(Some(-1)) && !mark_passes(None));
    }

    #[test]
    fn a_virtual_temperature_is_made_sensible_with_the_reports_own_humidity_or_not_written() {
        let mut l = level(1000.0, 1.0);
        l.obs[TOB] = Some(27.0 + 0.61 * 0.015 * 300.15);
        l.obs[QOB] = Some(15_000.0);
        l.marks[TQM] = Some(2.0);
        l.marks[QQM] = Some(2.0);
        l.is_virtual = true;
        let (rows, counts) = run(&sonde(vec![l.clone()]), table());
        assert_eq!(rows[0].variable, VAR_TEMPERATURE);
        assert_eq!(rows[0].value, 300.15);
        assert_eq!(counts.virtual_temperatures.seen, 1);
        assert_eq!(counts.virtual_temperatures.made_sensible_with_report_humidity, 1);
        // the humidity's mark refuses: the temperature is counted and not written
        l.marks[QQM] = Some(9.0);
        let (rows, counts) = run(&sonde(vec![l.clone()]), table());
        assert!(rows.iter().all(|r| r.variable != VAR_TEMPERATURE));
        assert_eq!(counts.virtual_temperatures.not_written_humidity_mark["9"], 1);
        // no humidity at all
        l.obs[QOB] = None;
        let (rows, counts) = run(&sonde(vec![l.clone()]), table());
        assert!(rows.is_empty());
        assert_eq!(counts.virtual_temperatures.not_written_humidity_missing, 1);
        // a sensible temperature is written as it is
        l.is_virtual = false;
        let (rows, counts) = run(&sonde(vec![l]), table());
        assert_eq!(rows[0].value, round_to(27.0 + 0.61 * 0.015 * 300.15 + 273.15, 2));
        assert_eq!(counts.virtual_temperatures.seen, 0);
    }

    #[test]
    fn the_virtual_flag_follows_the_event_stack_down_to_the_first_missing_code() {
        let stack = |codes: &[Option<f64>]| -> Vec<Datum> {
            codes.iter().map(|c| c.map(Datum::Number).unwrap_or(Datum::Missing)).collect()
        };
        assert!(stack_is_virtual(stack(&[Some(8.0), Some(1.0)]).iter(), 8.0));
        // not only the newest event: a later program above the virtual one keeps it virtual
        assert!(stack_is_virtual(stack(&[Some(4.0), Some(8.0), Some(1.0)]).iter(), 8.0));
        assert!(!stack_is_virtual(stack(&[Some(4.0), Some(1.0)]).iter(), 8.0));
        // a missing code ends the stack
        assert!(!stack_is_virtual(stack(&[Some(1.0), None, Some(8.0)]).iter(), 8.0));
        assert!(!stack_is_virtual(stack(&[]).iter(), 8.0));
    }

    #[test]
    fn the_error_is_the_tables_or_the_files_raised_as_gsi_raises_it() {
        let mut l = level(500.0, 1.0);
        l.obs[TOB] = Some(-20.0);
        l.obs[UOB] = Some(20.0);
        l.obs[VOB] = Some(5.0);
        l.marks[TQM] = Some(2.0);
        l.marks[WQM] = Some(2.0);
        l.errors[TOE] = Some(0.8);
        // the table: the file's own TOE is never consulted, as GSI with a table never does
        let (rows, counts) = run(&sonde(vec![l.clone()]), published());
        assert_eq!(rows[0].error, round_to(table_error(120, 500.0, Column::Temperature), 3));
        assert_ne!(rows[0].error, 0.8);
        assert_eq!(counts.error_from_table["temperature_k"], 1);
        // type 120 carries no wind error in the table (GSI's fill): the wind is not written
        assert_eq!(rows.len(), 1);
        assert_eq!(counts.not_written_no_error["120"]["wind"], 1);
        let mut wind_report = sonde(vec![l.clone()]);
        wind_report.typ = Some(220.0);
        let (rows, _) = run(&wind_report, published());
        assert_eq!(rows[0].error, round_to(table_error(220, 500.0, Column::Wind), 3));
        // the file: TOE is the error; the wind has no WOE, so it is counted and not written
        let (rows, counts) = run(&sonde(vec![l.clone()]), ErrorSource::File);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].error, 0.8);
        assert_eq!(counts.error_from_file["temperature_k"], 1);
        assert_eq!(counts.not_written_no_error["120"]["wind"], 1);
        // a mark of 3 raises the error by 1.2, and a temperature above 100 hPa by 1.2 again
        l.marks[TQM] = Some(3.0);
        let (rows, counts) = run(&sonde(vec![l.clone()]), ErrorSource::File);
        assert_eq!(rows[0].error, round_to(0.8 * 1.2, 3));
        assert_eq!(counts.error_inflated["mark_3_or_7:temperature_k"], 1);
        l.obs[POB] = Some(70.0);
        let (rows, counts) = run(&sonde(vec![l.clone()]), ErrorSource::File);
        assert_eq!(rows[0].error, round_to(0.8 * 1.2 * 1.2, 3));
        assert_eq!(counts.error_inflated["temperature_above_100_hpa"], 1);
        // a wind above 50 hPa is raised by 1.2; at 50 hPa it is not
        let mut w = level(40.0, 1.0);
        w.obs[UOB] = Some(20.0);
        w.obs[VOB] = Some(5.0);
        w.marks[WQM] = Some(1.0);
        w.errors[WOE] = Some(2.0);
        let mut high = sonde(vec![w.clone()]);
        high.typ = Some(220.0);
        let (rows, counts) = run(&high, ErrorSource::File);
        assert_eq!(rows[0].error, 2.4);
        assert_eq!(counts.error_inflated["wind_above_50_hpa"], 1);
        high.levels[0].obs[POB] = Some(50.0);
        assert_eq!(run(&high, ErrorSource::File).0[0].error, 2.0);
        // a profile's rows are counted: GSI's errormod factor is not applied here
        high.levels.push(w);
        let (rows, counts) = run(&high, ErrorSource::File);
        assert_eq!(rows.len(), 4);
        assert_eq!(counts.rows_without_errormod_factor["220"], 4);
    }

    #[test]
    fn surface_reports_write_station_pressure_under_gsis_tests_and_surface_rows_at_the_station() {
        let mut l = level(1003.2, 0.0);
        l.obs[ZOB] = Some(390.0);
        l.obs[TOB] = Some(12.0);
        l.marks[TQM] = Some(1.0);
        l.marks[ZQM] = Some(2.0);
        l.errors[POE] = Some(0.9);
        let mut report = sonde(vec![l.clone()]);
        report.message_type = "ADPSFC".into();
        report.typ = Some(187.0);
        report.elv = Some(391.0);
        let (rows, _) = run(&report, published());
        let ps = &rows[0];
        assert_eq!((ps.variable.as_str(), ps.value, ps.level_pa), (VAR_SURFACE_PRESSURE, 100_320.0, None));
        // the table's 0.5389 hPa for type 187, in Pa; the file's POE of 0.9 hPa is not consulted
        assert_eq!(ps.error, round_to(table_error(187, 1003.2, Column::SurfacePressure) * 100.0, 3));
        assert!((ps.error - 53.89).abs() < 0.001);
        assert_eq!((ps.elevation_m, ps.provenance.measurement), (390.0, MEAS_STATION_PRESSURE_FROM_ALTIMETER));
        let t = &rows[1];
        assert_eq!((t.level_pa, t.elevation_m, t.provenance.measurement), (None, 391.0, MEAS_SCREEN_TEMPERATURE_2M));
        assert_eq!(t.error, round_to(table_error(187, 1003.2, Column::Temperature), 3));
        // under the file source the pressure error is the file's POE, in Pa
        let (rows, _) = run(&report, ErrorSource::File);
        assert_eq!((rows[0].variable.as_str(), rows[0].error), (VAR_SURFACE_PRESSURE, 90.0));
        // below 500 hPa, a failing height mark, a level that is not the surface: no pressure row
        let mut high = l.clone();
        high.obs[POB] = Some(480.0);
        let mut report_high = report.clone();
        report_high.levels = vec![high];
        let (rows, counts) = run(&report_high, table());
        assert!(rows.iter().all(|r| r.variable != VAR_SURFACE_PRESSURE));
        assert_eq!(counts.rejected_surface_pressure["187"]["below_500_hpa"], 1);
        let mut bad_height = l.clone();
        bad_height.marks[ZQM] = Some(8.0);
        report_high.levels = vec![bad_height];
        let (rows, counts) = run(&report_high, table());
        assert!(rows.iter().all(|r| r.variable != VAR_SURFACE_PRESSURE));
        assert_eq!(counts.rejected_surface_pressure["187"]["height_mark_8"], 1);
        let mut aloft = l;
        aloft.obs[CAT] = Some(1.0);
        report_high.levels = vec![aloft];
        let (rows, _) = run(&report_high, table());
        assert!(rows.iter().all(|r| r.variable != VAR_SURFACE_PRESSURE));
    }

    #[test]
    fn gsis_surface_humidity_and_calm_mesonet_wind_rules_refuse_what_gsi_does_not_use() {
        // usage codes from the Fortran's own order: 116, then 117, then 118 wins
        assert_eq!(surface_humidity_usage(Some(15.0), Some(-41.12)), Some(116));
        assert_eq!(surface_humidity_usage(Some(-35.0), Some(-41.0)), None);
        assert_eq!(surface_humidity_usage(Some(-25.0), Some(-41.0)), Some(116));
        assert_eq!(surface_humidity_usage(Some(35.0), Some(-36.0)), Some(117));
        assert_eq!(surface_humidity_usage(Some(35.0), Some(32.3)), Some(118));
        assert_eq!(surface_humidity_usage(Some(35.0), Some(32.2)), None);
        // missing taken as the library's missing value, as GSI takes it
        assert_eq!(surface_humidity_usage(Some(20.0), None), Some(118));
        assert_eq!(surface_humidity_usage(None, Some(10.0)), Some(117));
        assert_eq!(surface_humidity_usage(None, None), Some(118));
        // the rows: a type-187 station at 15 C with a -41.26 C dewpoint (the EKEB case of
        // rap.t17z: TDO 231.89 K beside TOB 288.15 K) writes its temperature, not its humidity
        let mut l = level(1010.0, 0.0);
        l.obs[TOB] = Some(15.0);
        l.obs[TDO] = Some(231.89 - 273.15);
        l.obs[QOB] = Some(85.0);
        l.marks[TQM] = Some(2.0);
        l.marks[QQM] = Some(2.0);
        let mut report = sonde(vec![l.clone()]);
        report.message_type = "ADPSFC".into();
        report.typ = Some(187.0);
        report.elv = Some(10.0);
        let (rows, counts) = run(&report, table());
        assert!(rows.iter().all(|r| r.variable != VAR_DEWPOINT));
        assert!(rows.iter().any(|r| r.variable == VAR_TEMPERATURE));
        assert_eq!(counts.not_used_by_type_rule["187"]["usage_116_dewpoint_under_minus_40_c_and_10_k_under_t"], 1);
        // a sounding humidity is not under the surface rule
        let mut aloft = sonde(vec![{ let mut a = l.clone(); a.obs[CAT] = Some(1.0); a }]);
        aloft.typ = Some(120.0);
        assert!(run(&aloft, table()).0.iter().any(|r| r.variable == VAR_DEWPOINT));
        // a plausible dewpoint passes
        report.levels[0].obs[TDO] = Some(10.0);
        let (rows, counts) = run(&report, table());
        assert!(rows.iter().any(|r| r.variable == VAR_DEWPOINT));
        assert!(counts.not_used_by_type_rule.is_empty());
        // a type-288 mesonet wind under 0.01 m/s in both components is not used; 0.01 is
        let mut w = level(1010.0, 0.0);
        w.obs[UOB] = Some(0.0);
        w.obs[VOB] = Some(-0.005);
        w.marks[WQM] = Some(2.0);
        let mut mesonet = report.clone();
        mesonet.typ = Some(288.0);
        mesonet.levels = vec![w];
        let (rows, counts) = run(&mesonet, table());
        assert!(rows.is_empty());
        assert_eq!(counts.not_used_by_type_rule["288"][CALM_WIND_RULE], 1);
        mesonet.levels[0].obs[VOB] = Some(0.01);
        assert_eq!(run(&mesonet, table()).0.len(), 2);
        // the same calm wind from a type-287 station is used
        mesonet.levels[0].obs[VOB] = Some(0.0);
        mesonet.typ = Some(287.0);
        assert_eq!(run(&mesonet, table()).0.len(), 2);
    }

    #[test]
    fn report_types_are_table_rows_and_an_unmapped_type_is_counted() {
        assert_eq!(report_type_row(224, "VADWND").unwrap().wind, MEAS_VAD_LEVEL);
        assert_eq!(report_type_row(130, "AIRCFT").unwrap().temperature, MEAS_AIRCRAFT_LEVEL);
        assert_eq!(report_type_row(227, "PROFLR").unwrap().wind, MEAS_PROFILER_LEVEL);
        // one report type, two families, told apart by the message type
        assert_eq!(report_type_row(183, "ADPSFC").unwrap().family, "surface_land");
        assert_eq!(report_type_row(183, "SFCSHP").unwrap().family, "surface_marine");
        assert_eq!(report_type_row(280, "SFCSHP").unwrap().wind, MEAS_PLATFORM_WIND);
        assert!(report_type_row(290, "ASCATW").is_none());
        let mut report = sonde(vec![level(850.0, 1.0)]);
        report.typ = Some(290.0);
        let (rows, counts) = run(&report, table());
        assert!(rows.is_empty());
        assert_eq!(counts.reports_not_mapped_by_report_type["290"], 1);
        assert_eq!(counts.reports_by_report_type["290"], 1);
        // a level without a height is stamped with the standard-atmosphere altitude of its pressure
        let mut l = level(500.0, 1.0);
        l.obs[TOB] = Some(-20.0);
        l.marks[TQM] = Some(2.0);
        let (rows, counts) = run(&sonde(vec![l.clone()]), table());
        assert!((rows[0].elevation_m - 5574.4).abs() < 1.0);
        assert_eq!(counts.level_rows_stamped_with_isa_altitude, 1);
        // a sounding level drifts with the balloon when the file says where to
        l.drift = [Some(254.0), Some(35.5), Some(0.25)];
        let (rows, counts) = run(&sonde(vec![l]), table());
        assert_eq!((rows[0].latitude_deg, rows[0].longitude_deg), (35.5, -106.0));
        assert_eq!(rows[0].valid_time, Utc.with_ymd_and_hms(2026, 10, 3, 12, 15, 0).unwrap());
        assert_eq!(counts.levels_placed_by_balloon_drift, 1);
    }

    fn vad(levels: usize, dhr: f64) -> Report {
        let levels = (0..levels)
            .map(|i| {
                let mut l = level(900.0 - 5.0 * i as f64, 4.0);
                l.obs[ZOB] = Some(500.0 + 50.0 * i as f64);
                l.obs[UOB] = Some(10.0 + 0.1 * i as f64);
                l.obs[VOB] = Some(-2.0);
                l.background = [Some(9.0), Some(-1.0), Some(5.0)];
                l.marks[WQM] = Some(2.0);
                l
            })
            .collect();
        Report {
            message_type: "VADWND".into(),
            cycle: cycle(),
            station: Some("KTLX".into()),
            xob: Some(262.72),
            yob: Some(35.33),
            dhr: Some(dhr),
            typ: Some(224.0),
            elv: Some(370.0),
            tsb: Some(2.0),
            levels,
        }
    }

    #[test]
    fn a_new_vad_report_is_read_as_gsi_reads_it() {
        // 14 levels at 50 m: levels 6 and 12 are sampled; 12 has no full window and ends the report
        let report = vad(14, -0.25);
        assert!(report_is_new_vad(&report));
        let (rows, counts) = run(&report, table());
        assert_eq!(rows.len(), 2);
        let mean_u = (0..6).map(|i| 10.0 + 0.1 * (5 + i) as f64).sum::<f64>() / 6.0;
        assert_eq!((rows[0].variable.as_str(), rows[0].value), (VAR_WIND_U, round_to(mean_u, 2)));
        assert_eq!(rows[1].value, -2.0);
        // the superob sits at the sampled level's pressure and the six levels' mean height
        assert_eq!(rows[0].level_pa, Some(87_500.0));
        assert_eq!(rows[0].elevation_m, 500.0 + 50.0 * 7.5);
        assert_eq!(rows[0].provenance.measurement, MEAS_VAD_SUPEROB);
        assert_eq!(counts.vad.superobs_formed, 1);
        assert_eq!(counts.vad.levels_not_sampled, 10);
        assert_eq!(counts.vad.levels_without_a_full_window, 3);
        // outside GSI's time windows the report is not read
        let (rows, counts) = run(&vad(14, -0.5), table());
        assert!(rows.is_empty());
        assert_eq!(counts.vad.reports_outside_the_time_windows, 1);
        assert!(in_vad_time_window(Some(-2.5)) && !in_vad_time_window(Some(2.62)) && !in_vad_time_window(None));
        // a departure from the file's background over 10 m/s refuses the superob
        let mut far = vad(14, 0.75);
        far.levels[5].background = [Some(-5.0), Some(-1.0), Some(5.0)];
        let (rows, counts) = run(&far, table());
        assert!(rows.is_empty());
        assert_eq!(counts.vad.rejected_background_departure, 1);
        // a missing background is a departure (GSI's arithmetic on the missing value)
        let mut none = vad(14, 0.75);
        none.levels[5].background = [None, None, None];
        assert_eq!(run(&none, table()).1.vad.rejected_background_departure, 1);
        // a level of the six that departs from the mean by more than 5 m/s refuses it
        let mut spread = vad(14, 0.75);
        spread.levels[8].obs[UOB] = Some(25.0);
        assert_eq!(run(&spread, table()).1.vad.rejected_spread, 1);
        // only levels within 301 m join; fewer than three refuses the superob
        let mut thin = vad(14, 0.75);
        for (i, l) in thin.levels.iter_mut().enumerate().skip(7) {
            l.obs[ZOB] = Some(5000.0 + i as f64);
        }
        assert_eq!(run(&thin, table()).1.vad.rejected_fewer_than_3, 1);
        // a superob faster than 60 m/s ends the report
        let mut fast = vad(20, 0.75);
        for l in fast.levels.iter_mut() {
            l.obs[UOB] = Some(61.0);
            l.background = [Some(61.0), Some(-2.0), None];
        }
        let (rows, counts) = run(&fast, table());
        assert!(rows.is_empty());
        assert_eq!(counts.vad.reports_ended_by_a_superob_over_60_m_s, 1);
        // in a new-VAD file a report of another subtype is not read; raw mode writes every level
        let mut old = vad(14, 0.25);
        old.tsb = Some(1.0);
        let provenance = RowProvenance::of_source("ab", None, None);
        let mut counts = Counts::default();
        let mut writer = TableWriter::new();
        let mut by_variable = BTreeMap::new();
        let context = RowContext { provenance: &provenance, errors: table(), vad_superob: true };
        super::rows(&old, &context, FileRules { new_vad: true }, &mut counts, &mut writer, &mut by_variable);
        assert!(writer.is_empty());
        assert_eq!(counts.vad.reports_of_the_other_subtype, 1);
        let raw = RowContext { vad_superob: false, ..context };
        super::rows(&old, &raw, FileRules { new_vad: true }, &mut counts, &mut writer, &mut by_variable);
        assert_eq!(writer.len(), 28);
        assert!(writer.rows().iter().all(|r| r.provenance.measurement == MEAS_VAD_LEVEL));
    }

    #[test]
    fn multi_agency_profiler_winds_above_400_hpa_are_not_used() {
        let mut low = level(500.0, 4.0);
        low.obs[UOB] = Some(5.0);
        low.obs[VOB] = Some(5.0);
        low.obs[ZOB] = Some(5500.0);
        low.marks[WQM] = Some(1.0);
        let mut high = low.clone();
        high.obs[POB] = Some(350.0);
        let mut report = sonde(vec![low, high]);
        report.message_type = "PROFLR".into();
        report.typ = Some(227.0);
        let (rows, counts) = run(&report, table());
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.level_pa == Some(50_000.0) && r.provenance.measurement == MEAS_PROFILER_LEVEL));
        assert_eq!(counts.not_used_by_type_rule["227"]["wind_above_400_hpa"], 1);
    }

    const FIXTURE: &[u8] = include_bytes!("../tests/fixtures/prepbufr/rap-t12z-fixture.bufr");
    const FIXTURE_ORACLE_GZ: &[u8] = include_bytes!("../tests/fixtures/prepbufr/rap-t12z-fixture.oracle.txt.gz");

    fn fixture_table(errors: ErrorSource) -> (TableWriter, BTreeMap<String, usize>, Counts) {
        let file = read_file(FIXTURE, "fixture").unwrap();
        let provenance = RowProvenance::of_source(&crate::hex_sha256(FIXTURE), None, None);
        table_of(&file, &RowContext { provenance: &provenance, errors, vad_superob: true }).unwrap()
    }

    #[test]
    fn a_real_fixture_decodes_to_the_ncep_librarys_listing_byte_for_byte() {
        use std::io::Read;
        let mut expected = String::new();
        flate2::read::GzDecoder::new(FIXTURE_ORACLE_GZ).read_to_string(&mut expected).unwrap();
        let file = read_file(FIXTURE, "fixture").unwrap();
        assert_eq!(file.dictionary_messages, 7);
        assert_eq!(file.dictionaries.len(), 1);
        let dictionary = &file.dictionaries[0];
        assert_eq!(dictionary.messages, 6);
        assert_eq!(dictionary.table_a.len(), 21);
        assert_eq!(dictionary.program_code(VIRTUAL_PROGRAM).unwrap(), 8);
        let listing = dump(&file).unwrap();
        // the first difference, named, rather than two 780 kB strings
        if let Some((n, (a, b))) = listing.lines().zip(expected.lines()).enumerate().find(|(_, (a, b))| a != b) {
            panic!("listing line {} differs:
 rust   {a}
 oracle {b}", n + 1);
        }
        assert_eq!(listing.len(), expected.len());
        assert_eq!(crate::hex_sha256(listing.as_bytes()), crate::hex_sha256(expected.as_bytes()));
        let c = census(&file).unwrap();
        let (messages, subsets, levels) = (&c.messages_by_type, &c.subsets_by_type, &c.levels_by_type);
        assert_eq!(messages.values().sum::<usize>(), 8);
        assert_eq!((subsets["ADPUPA"], levels["ADPUPA"], levels["VADWND"], subsets["ADPSFC"]), (4, 990, 127, 89));
        let by_type = &c.subsets_by_type_and_report_type;
        assert_eq!((by_type["ADPUPA/120"], by_type["ADPUPA/220"], by_type["ASCATW/290"]), (2, 2, 5));
    }

    #[test]
    fn a_real_fixture_writes_the_rows_the_use_rule_allows_and_counts_the_rest() {
        let (writer, by_variable, counts) = fixture_table(published());
        assert_eq!(writer.len(), 2350);
        assert_eq!(
            by_variable,
            BTreeMap::from([
                ("dewpoint_k".to_string(), 495),
                ("surface_pressure_pa".to_string(), 45),
                ("temperature_k".to_string(), 538),
                ("wind_u_m_s".to_string(), 636),
                ("wind_v_m_s".to_string(), 636),
            ])
        );
        // types 192 and 292 have no temperature or wind error in the table (GSI's fill;
        // the regional convinfo does not list them), so those values are not written
        assert_eq!(
            counts.not_written_no_error,
            BTreeMap::from([
                ("192".to_string(), BTreeMap::from([("temperature_k".to_string(), 1)])),
                ("292".to_string(), BTreeMap::from([("wind".to_string(), 1)])),
            ])
        );
        assert_eq!(counts.error_inflated, BTreeMap::from([("mark_3_or_7:wind".to_string(), 3)]));
        assert_eq!(counts.rows_without_errormod_factor["220"], 990);
        assert_eq!(counts.rejected_by_mark["120"]["dewpoint_k"]["15"], 16);
        assert_eq!(counts.rejected_by_mark["181"]["surface_pressure_pa"]["14"], 2);
        assert_eq!(counts.rejected_by_mark["281"]["wind"]["14"], 3);
        assert_eq!(counts.humidity_mark_9_taken_as_2, 209);
        assert_eq!(counts.surface_pressure_not_written_by_report_type["192"], 8);
        assert_eq!(counts.reports_not_mapped_by_report_type, BTreeMap::from([("126".to_string(), 15), ("290".to_string(), 5)]));
        // no row anywhere is written with a failing mark: every rejection is a count
        assert!(writer.rows().iter().all(|r| r.error > 0.0 && r.value.is_finite()));
        // every error is the error table's
        assert!(counts.error_from_file.is_empty());
        // (a wind's two rows share one error)
        let from_table: BTreeMap<String, usize> =
            [("dewpoint_k", 495), ("surface_pressure_pa", 45), ("temperature_k", 538), ("wind", 636)].map(|(k, n)| (k.to_string(), n)).into();
        assert_eq!(counts.error_from_table, from_table);
        // the bytes the door wrote for this fixture.  The station pressure's error is type
        // 120's 0.68115 hPa; the VAD superob's is type 224's at 919.1 hPa (2.3712 m/s between
        // 2.4519 at 950 and 2.3213 at 900) raised by 1.2 for its wind mark of 3
        let text = writer.text();
        assert_eq!(crate::hex_sha256(text.as_bytes()), "0aadde688838b4d1262f4a26cc87382f24c558a00ade1faf576d0392c348e5b1");
        assert!(text.lines().nth(1).unwrap().starts_with(
            "prepbufr,47412,43.06050,141.32829,26.0,,2026-10-03T11:30:00Z,surface_pressure_pa,101970,68.115,station_pressure,"
        ));
        // GSI's VAD read: one of the two reports falls in a time window, every
        // sixth level is sampled, nine superobs form and one is refused
        assert!(counts.vad.file_is_new_vad);
        assert_eq!((counts.vad.reports_seen, counts.vad.reports_outside_the_time_windows), (2, 1));
        assert_eq!((counts.vad.superobs_formed, counts.vad.rejected_background_departure, counts.vad.levels_not_sampled), (9, 1, 55));
        assert_eq!(counts.rows_by_measurement["vad_superob"], 18);
        assert!(text.contains(
            "prepbufr,KBGM,42.20000,-75.98000,940.0,91910.0,2026-10-03T11:12:00Z,wind_v_m_s,-8.05,2.845,vad_superob,"
        ));
        // two decodes, one table
        let (again, _, _) = fixture_table(published());
        assert_eq!(again.text(), text);
        // every VAD level as measured, when asked
        let file = read_file(FIXTURE, "fixture").unwrap();
        let provenance = RowProvenance::of_source(&crate::hex_sha256(FIXTURE), None, None);
        let raw = RowContext { provenance: &provenance, errors: published(), vad_superob: false };
        let (raw, _, counts) = table_of(&file, &raw).unwrap();
        assert_eq!((raw.len(), counts.rows_by_measurement["vad_level"]), (2586, 254));
        // the file carries no POE, TOE, QOE or WOE, so under the file source nothing is written
        let (strict, _, counts) = fixture_table(ErrorSource::File);
        assert!(strict.is_empty());
        assert_eq!(counts.not_written_no_error["120"]["temperature_k"], 495);
    }

    #[test]
    fn a_decoded_file_dumps_the_oracle_lines_and_writes_rows() {
        let mut bytes = small_dictionary();
        // two events on the temperature: the virtual-temperature program on top
        let subset = obstyp_subset(
            "72365",
            7_338_000,
            12_504_000,
            120,
            2620,
            &[(1, vec![(Some(8350), 2, 1)], vec![(Some(2732 + 251), 2, 8), (Some(2732 + 248), 1, 1)])],
        );
        bytes.extend(data_message(120, 348_120, (2026, 10, 3, 12), &[subset]));
        let file = read_file(&bytes, "test").unwrap();
        let text = dump(&file).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "V    8");
        assert_eq!(lines[1], "M OBSTYP   2026100312      1");
        assert!(lines[2].starts_with("S      1 2020203536333237 "), "{}", lines[2]);
        assert!(lines[2].ends_with(" 42374876E8000000    1    1    0    1    0    0"), "{}", lines[2]);
        // POB 835.0 then a missing QOB, and the level is virtual with program 8 on top
        assert!(lines[3].starts_with("L   1 408A180000000000 42374876E8000000 "), "{}", lines[3]);
        assert!(lines[3].ends_with(" 4020000000000000 V"), "{}", lines[3]);
        assert_eq!(lines.len(), 4);
        let c = census(&file).unwrap();
        assert_eq!((c.messages_by_type["OBSTYP"], c.subsets_by_type["OBSTYP"], c.levels_by_type["OBSTYP"]), (1, 1, 1));
        assert_eq!(c.subsets_by_type_and_report_type["OBSTYP/120"], 1);
    }
}
