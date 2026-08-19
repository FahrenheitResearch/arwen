//! Local GRIB Edition 1 import — the GDEX "past" datasets (ERA-20C et al.).
//!
//! One GRIB1 file carries ONE parameter across MANY timesteps (the owner's
//! ERA-20C surface stream is 2,928 three-hourly analyses spanning a year in a
//! single 450 MB file), so this importer inverts `local_import`'s
//! one-file-one-hour shape: it indexes every message's byte range and header
//! once, then decodes lazily — one 320x160 plane at a time — writing each
//! timestep as its own forecast-hour slot via
//! `rw_store::write_hour_from_grid_with_derived` and dropping the plane
//! before the next. Peak decoded state is one timestep's fields, never the
//! whole year.
//!
//! Decoding is grib-core's (the pinned rusty-weather vendor crate): PDS/GDS
//! parse, IBM-float reference values, 24-bit simple packing, and true
//! Gaussian latitudes (Legendre roots) all come from
//! `grib_core::grib1::Grib1File`. What lives HERE is the app seam grib-core
//! does not provide:
//! - a streaming message INDEX (grib-core's `from_bytes` eagerly clones every
//!   section of every message — 2x file size in RAM for a 450 MB file);
//! - ECMWF table-128 STORE naming: readable slugs and canonical selectors
//!   (grib-core's version-aware `parameter_entry` resolves ECMWF names and
//!   units now, but the store slug/selector plan stays an app concern);
//! - the store-write plan: canonical `FieldSelector`s for the params whose
//!   units match what the WRF import precedent stores, derived slugs for the
//!   rest, and hour keys derived from each message's valid time;
//! - scan-mode normalization: +/-i rows become one eastward row-major layout;
//!   column-major modes and GRIB1's reserved bits (including `0x10`) fail closed;
//! - global-grid longitude normalization (columns rotated so longitudes run
//!   -180..180 monotonic — the map layer's inverse LUT does not wrap, so a
//!   raw 0..360 grid would blank the western hemisphere).
//!
//! Hour keys are HOURS SINCE THE FIRST TIMESTEP (0, 3, 6, ... 8781 for a
//! 3-hourly year), computed from decoded reference times rather than assumed
//! spacing, so gappy or differently-stepped files stay correct. The run name
//! ("era20c_fsr_2004010100_science_v1_<source-id>") is deliberately NOT
//! `YYYYMMDD_HHz`-shaped;
//! timeline consumers would otherwise pull a year-long 2004 reanalysis into
//! a wall-clock timeline. Reanalysis runs are reached through
//! the run browser tree. The store model slug is `wrf` — the same
//! slug both existing import paths stamp (including GDEX climate wrfouts) —
//! because the Solar-fallback styling, label translation, and native-plot
//! paths all key on it; a new slug would render every field styleless.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use grib_core::grib1::{Grib1File, Grib1Message, GridType};
use rustwx_core::{CanonicalField, FieldSelector, GridShape, LatLonGrid, SelectedField2D};
use rw_store::{DerivedFieldInput, write_hour_from_grid_with_derived};

use crate::local_import::{
    IMPORT_SCIENCE_SCHEMA_VERSION, LocalImportSummary, RunStagingPublisher,
    capture_source_set_identity, verify_source_set_unchanged,
};

/// Standard gravity (m/s^2) — converts ECMWF geopotential (m^2/s^2) to the
/// geopotential height (gpm) every other height field in the store speaks.
const STANDARD_GRAVITY: f64 = 9.80665;

/// Refuse hostile or implausible GDS dimensions before grib-core builds its
/// coordinate vectors. At 25 million cells the eventual lat/lon plus decoded
/// plane working set is already measured in hundreds of MiB.
const MAX_GRIB1_GRID_CELLS: usize = 25_000_000;

/// Bound the compact header index too. One million messages still covers
/// more than a century of hourly data in one file.
const MAX_GRIB1_MESSAGES_PER_FILE: usize = 1_000_000;

fn checked_grid_cells(ni: usize, nj: usize) -> Result<usize, String> {
    if ni == 0 || nj == 0 {
        return Err(format!("GRIB1 has a zero-sized grid ({ni}x{nj})"));
    }
    let cells = ni
        .checked_mul(nj)
        .ok_or_else(|| format!("GRIB1 grid dimensions overflow ({ni}x{nj})"))?;
    if cells > MAX_GRIB1_GRID_CELLS {
        return Err(format!(
            "GRIB1 grid {ni}x{nj} has {cells} cells, exceeding the desktop safety ceiling of {MAX_GRIB1_GRID_CELLS}"
        ));
    }
    Ok(cells)
}

fn ensure_message_index_capacity(current: usize) -> Result<(), String> {
    if current >= MAX_GRIB1_MESSAGES_PER_FILE {
        Err(format!(
            "GRIB1 file exceeds the safety ceiling of {MAX_GRIB1_MESSAGES_PER_FILE} indexed messages"
        ))
    } else {
        Ok(())
    }
}

/// Extension gate: GRIB1 containers only. `.grb2`/`.grib2` deliberately stay
/// unsupported here — MRMS/HRRR GRIB2 arrive through their own feeds, and a
/// GRIB2 message inside a `.grb` file gets a clear edition error at index
/// time instead of silently decoding garbage.
pub fn is_grib1_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("grb" | "grib")
    )
}

// ---------------------------------------------------------------------------
// Message index
// ---------------------------------------------------------------------------

/// One GRIB1 message's byte range plus the PDS/GDS header facts the import
/// plan needs. Built by [`index_grib1_file`] from ~120 header bytes per
/// message — values stay packed on disk until the write loop asks.
#[derive(Debug, Clone)]
pub(crate) struct IndexedMessage {
    pub offset: u64,
    pub total_len: u32,
    pub table_version: u8,
    pub center: u8,
    pub parameter: u8,
    pub level_type: u8,
    pub level_value: u16,
    /// Valid time (reference time + forecast offset), unix seconds.
    pub valid_unix: i64,
    pub ni: u16,
    pub nj: u16,
    /// Exact identity of the complete Grid Description Section. Dimensions
    /// alone are not enough: two grids can share Ni/Nj while using different
    /// extents, projections, increments, or scanning modes.
    grid_identity: GridIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GridIdentity {
    gds_len: usize,
    fingerprint: u64,
}

/// Deterministic FNV-1a over the complete raw GDS. The byte length is retained
/// separately as part of the identity; unlike `DefaultHasher`, this algorithm
/// is a persistent-format promise and therefore stable across processes.
fn grid_identity(gds: &[u8]) -> GridIdentity {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let fingerprint = gds.iter().fold(OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(PRIME)
    });
    GridIdentity {
        gds_len: gds.len(),
        fingerprint,
    }
}

/// Keep only messages whose complete GDS identity matches the reference.
/// Returns the number rejected before any values are decoded with a foreign
/// grid plan.
fn retain_reference_grid(
    indexed: &mut Vec<(usize, IndexedMessage)>,
    reference: GridIdentity,
) -> usize {
    let before = indexed.len();
    indexed.retain(|(_, message)| message.grid_identity == reference);
    before - indexed.len()
}

fn read_exact_at(file: &mut File, offset: u64, buf: &mut [u8]) -> Result<(), String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| format!("seek to {offset}: {err}"))?;
    file.read_exact(buf)
        .map_err(|err| format!("read {} bytes at {offset}: {err}", buf.len()))
}

fn read_u24(bytes: &[u8]) -> u32 {
    ((bytes[0] as u32) << 16) | ((bytes[1] as u32) << 8) | (bytes[2] as u32)
}

/// Scan forward from `pos` for the next "GRIB" magic, tolerating padding or
/// index blocks between messages. Chunked with a 3-byte overlap so magic
/// spanning a chunk boundary still matches.
fn scan_forward_for_magic(file: &mut File, pos: u64, file_len: u64) -> Result<Option<u64>, String> {
    const CHUNK: usize = 64 * 1024;
    let mut start = pos;
    let mut buf = vec![0u8; CHUNK];
    while start + 4 <= file_len {
        let want = usize::try_from((file_len - start).min(CHUNK as u64)).unwrap_or(CHUNK);
        let chunk = &mut buf[..want];
        read_exact_at(file, start, chunk)?;
        if let Some(found) = chunk.windows(4).position(|window| window == b"GRIB") {
            return Ok(Some(start + found as u64));
        }
        if start + want as u64 >= file_len {
            break;
        }
        // Re-read the last 3 bytes with the next chunk.
        start += (want - 3) as u64;
    }
    Ok(None)
}

/// Forecast offset in seconds from PDS time unit / P1 / P2 / time range
/// indicator. Only indicators with an unambiguous supported endpoint are
/// accepted; other interval/statistical encodings fail closed instead of
/// being mislabeled as P1. Analyses are `tri 0, P1 0` and return 0.
fn forecast_offset_seconds(time_unit: u8, p1: u8, p2: u8, tri: u8) -> Result<i64, String> {
    let unit_seconds: i64 = match time_unit {
        0 => 60,
        1 => 3_600,
        2 => 86_400,
        10 => 3 * 3_600,
        11 => 6 * 3_600,
        12 => 12 * 3_600,
        13 => 900,
        14 => 1_800,
        254 => 1,
        // 3..=7: months / years / decades / normals / centuries.
        _ => {
            return Err(format!(
                "calendar or unsupported forecast time unit {time_unit} cannot be converted to fixed seconds"
            ));
        }
    };
    let periods: i64 = match tri {
        // Forecast product valid at reference time + P1.
        0 => i64::from(p1),
        // Initialized analysis is valid exactly at the reference time; the
        // GRIB1 contract requires both period octets to be zero.
        1 if p1 == 0 && p2 == 0 => 0,
        1 => {
            return Err(format!(
                "time range indicator 1 requires P1=P2=0, got P1={p1} P2={p2}"
            ));
        }
        // Two-octet P1 (used when a forecast period exceeds 255 units).
        10 => ((p1 as i64) << 8) | (p2 as i64),
        // Period products (averages / accumulations / differences) are valid
        // at the END of the (P1, P2) window.
        2..=5 if p2 >= p1 => i64::from(p2),
        2..=5 => {
            return Err(format!(
                "time range indicator {tri} has reversed interval P1={p1}, P2={p2}"
            ));
        }
        _ => {
            return Err(format!(
                "time range indicator {tri} is unsupported; refusing to guess its valid-time endpoint"
            ));
        }
    };
    periods
        .checked_mul(unit_seconds)
        .ok_or_else(|| "forecast offset overflows seconds".to_string())
}

/// Index every GRIB1 message in `path`: byte ranges from each message's own
/// length word (NO uniform-record-size assumption), plus the PDS/GDS facts
/// the field plan needs. Each message's trailing `7777` is verified so a
/// truncated download fails here, loudly, instead of mid-import.
pub(crate) fn index_grib1_file(path: &Path) -> Result<Vec<IndexedMessage>, String> {
    let name = display_name(path);
    let mut file = File::open(path).map_err(|err| format!("{name}: open: {err}"))?;
    let file_len = file
        .metadata()
        .map_err(|err| format!("{name}: metadata: {err}"))?
        .len();

    let mut out = Vec::new();
    let mut pos = 0u64;
    let mut header = [0u8; 8];
    while pos + 8 <= file_len {
        read_exact_at(&mut file, pos, &mut header).map_err(|err| format!("{name}: {err}"))?;
        if &header[0..4] != b"GRIB" {
            match scan_forward_for_magic(&mut file, pos, file_len)
                .map_err(|err| format!("{name}: {err}"))?
            {
                Some(next) => {
                    pos = next;
                    continue;
                }
                None => break,
            }
        }
        let total_len = read_u24(&header[4..7]);
        let edition = header[7];
        if edition != 1 {
            return Err(format!(
                "{name}: GRIB edition {edition} message at byte {pos} — this importer handles \
                 GRIB1 only (GRIB2 products arrive through their own feeds)"
            ));
        }
        if total_len < 40 || pos + total_len as u64 > file_len {
            return Err(format!(
                "{name}: message at byte {pos} claims {total_len} bytes but the file holds \
                 {file_len} — truncated download?"
            ));
        }

        // PDS: everything the plan needs sits in the first 28 bytes.
        let mut pds = [0u8; 28];
        read_exact_at(&mut file, pos + 8, &mut pds).map_err(|err| format!("{name}: {err}"))?;
        let pds_len = read_u24(&pds[0..3]);
        if pds_len < 28 {
            return Err(format!(
                "{name}: message at byte {pos}: PDS of {pds_len} bytes (< 28) not supported"
            ));
        }
        let gds_present = pds[7] & 0x80 != 0;
        if !gds_present {
            return Err(format!(
                "{name}: message at byte {pos} has no Grid Description Section — cannot place \
                 its values on a grid"
            ));
        }
        let table_version = pds[3];
        let center = pds[4];
        let parameter = pds[8];
        let level_type = pds[9];
        let level_value = ((pds[10] as u16) << 8) | pds[11] as u16;
        let year_of_century = pds[12] as i32;
        let century = pds[24] as i32;
        let year = if century == 0 {
            1900 + year_of_century
        } else {
            (century - 1) * 100 + year_of_century
        };
        let reference = chrono::NaiveDate::from_ymd_opt(year, pds[13] as u32, pds[14] as u32)
            .and_then(|date| date.and_hms_opt(pds[15] as u32, pds[16] as u32, 0))
            .ok_or_else(|| {
                format!(
                    "{name}: message at byte {pos}: bad reference time {year}-{}-{} {}:{}",
                    pds[13], pds[14], pds[15], pds[16]
                )
            })?;
        let offset_seconds = forecast_offset_seconds(pds[17], pds[18], pds[19], pds[20])
            .map_err(|error| format!("{name}: message at byte {pos}: {error}"))?;
        let valid_unix = reference.and_utc().timestamp() + offset_seconds;

        // GDS: read the complete section once. Ni/Nj live at fixed offsets
        // 6..10 for the rectilinear grids this importer accepts, while the
        // complete bytes form the identity used to reject same-sized but
        // geometrically different messages before decode.
        let gds_offset = pos
            .checked_add(8)
            .and_then(|offset| offset.checked_add(u64::from(pds_len)))
            .ok_or_else(|| format!("{name}: message at byte {pos}: GDS offset overflow"))?;
        let message_payload_end = pos + u64::from(total_len) - 4;
        if gds_offset
            .checked_add(10)
            .is_none_or(|end| end > message_payload_end)
        {
            return Err(format!(
                "{name}: message at byte {pos}: truncated Grid Description Section header"
            ));
        }
        let mut gds_header = [0u8; 10];
        read_exact_at(&mut file, gds_offset, &mut gds_header)
            .map_err(|err| format!("{name}: {err}"))?;
        let gds_len = read_u24(&gds_header[0..3]);
        if gds_len < 10
            || gds_offset
                .checked_add(u64::from(gds_len))
                .is_none_or(|end| end > message_payload_end)
        {
            return Err(format!(
                "{name}: message at byte {pos}: GDS length {gds_len} exceeds the message"
            ));
        }
        let ni = ((gds_header[6] as u16) << 8) | gds_header[7] as u16;
        let nj = ((gds_header[8] as u16) << 8) | gds_header[9] as u16;
        if ni == 0xFFFF {
            return Err(format!(
                "{name}: message at byte {pos} uses a quasi-regular (reduced) grid — download \
                 the regular-grid product (e.g. regn80sc) instead"
            ));
        }
        checked_grid_cells(usize::from(ni), usize::from(nj))
            .map_err(|error| format!("{name}: message at byte {pos}: {error}"))?;
        let mut gds = vec![0u8; gds_len as usize];
        read_exact_at(&mut file, gds_offset, &mut gds).map_err(|err| format!("{name}: {err}"))?;
        let grid_identity = grid_identity(&gds);

        // End section: `7777` exactly where the length word says.
        let mut end = [0u8; 4];
        read_exact_at(&mut file, pos + total_len as u64 - 4, &mut end)
            .map_err(|err| format!("{name}: {err}"))?;
        if &end != b"7777" {
            return Err(format!(
                "{name}: message at byte {pos} does not end in '7777' — corrupt or truncated"
            ));
        }

        ensure_message_index_capacity(out.len())
            .map_err(|error| format!("{name}: message at byte {pos}: {error}"))?;
        out.push(IndexedMessage {
            offset: pos,
            total_len,
            table_version,
            center,
            parameter,
            level_type,
            level_value,
            valid_unix,
            ni,
            nj,
            grid_identity,
        });
        pos += total_len as u64;
    }

    if out.is_empty() {
        return Err(format!("{name}: no GRIB1 messages found"));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ECMWF parameter table 128
// ---------------------------------------------------------------------------

/// One ECMWF table-128 parameter: GDEX short name, store slug, human label,
/// units as stored (ERA native units — no conversion beyond geopotential).
pub(crate) struct EraParam {
    pub short: &'static str,
    pub slug: &'static str,
    // Retained for the field-picker/UI catalog seam; production import currently
    // persists the stable slug and units, while the mapping tests pin this label.
    #[cfg_attr(not(test), allow(dead_code))]
    pub label: &'static str,
    pub units: &'static str,
}

/// The ECMWF table-128 parameters ERA-20C (and the other GDEX "past"
/// reanalysis streams) actually publish. grib-core's `parameter_entry`
/// resolves ECMWF table 128 names/units by cited version now; this map is
/// the app seam for STORE naming, kept because slugs are
/// deliberately readable (they surface verbatim in the field picker) and
/// chosen so `color_tables::solar_model_field_table`'s substring heuristics
/// land where a palette exists (temperature/dewpoint/cape/vort/precip/...).
pub(crate) fn era128_param(parameter: u8) -> Option<EraParam> {
    let entry = |short, slug, label, units| {
        Some(EraParam {
            short,
            slug,
            label,
            units,
        })
    };
    match parameter {
        31 => entry("ci", "sea_ice_cover", "Sea-ice cover", "(0-1)"),
        34 => entry(
            "sst",
            "sea_surface_temperature",
            "Sea surface temperature",
            "K",
        ),
        59 => entry(
            "cape",
            "cape",
            "Convective available potential energy",
            "J/kg",
        ),
        129 => entry("z", "geopotential", "Geopotential", "m2/s2"),
        130 => entry("t", "temperature", "Temperature", "K"),
        131 => entry("u", "u_wind", "U component of wind", "m/s"),
        132 => entry("v", "v_wind", "V component of wind", "m/s"),
        133 => entry("q", "specific_humidity", "Specific humidity", "kg/kg"),
        134 => entry("sp", "surface_pressure", "Surface pressure", "Pa"),
        135 => entry("w", "omega", "Vertical velocity (pressure)", "Pa/s"),
        136 => entry("tcw", "total_column_water", "Total column water", "kg/m2"),
        137 => entry(
            "tcwv",
            "total_column_water_vapour",
            "Total column water vapour",
            "kg/m2",
        ),
        138 => entry("vo", "relative_vorticity", "Relative vorticity", "1/s"),
        141 => entry("sd", "snow_depth", "Snow depth (water equivalent)", "m"),
        142 => entry(
            "lsp",
            "large_scale_precipitation",
            "Large-scale precipitation",
            "m",
        ),
        143 => entry(
            "cp",
            "convective_precipitation",
            "Convective precipitation",
            "m",
        ),
        144 => entry("sf", "snowfall", "Snowfall (water equivalent)", "m"),
        151 => entry("msl", "mslp", "Mean sea level pressure", "Pa"),
        155 => entry("d", "divergence", "Divergence", "1/s"),
        156 => entry("gh", "height", "Geopotential height", "gpm"),
        157 => entry("r", "relative_humidity", "Relative humidity", "%"),
        159 => entry("blh", "boundary_layer_height", "Boundary layer height", "m"),
        164 => entry("tcc", "total_cloud_cover", "Total cloud cover", "(0-1)"),
        165 => entry("10u", "u_10m", "10 m U wind component", "m/s"),
        166 => entry("10v", "v_10m", "10 m V wind component", "m/s"),
        167 => entry("2t", "temperature_2m", "2 m temperature", "K"),
        168 => entry("2d", "dewpoint_2m", "2 m dewpoint temperature", "K"),
        172 => entry("lsm", "land_sea_mask", "Land-sea mask", "(0-1)"),
        173 => entry("sr", "surface_roughness", "Surface roughness", "m"),
        182 => entry("e", "evaporation", "Evaporation (water equivalent)", "m"),
        186 => entry("lcc", "low_cloud_cover", "Low cloud cover", "(0-1)"),
        187 => entry("mcc", "medium_cloud_cover", "Medium cloud cover", "(0-1)"),
        188 => entry("hcc", "high_cloud_cover", "High cloud cover", "(0-1)"),
        201 => entry("mx2t", "temperature_2m_max", "Maximum 2 m temperature", "K"),
        202 => entry("mn2t", "temperature_2m_min", "Minimum 2 m temperature", "K"),
        205 => entry("ro", "runoff", "Runoff", "m"),
        228 => entry("tp", "total_precipitation", "Total precipitation", "m"),
        235 => entry("skt", "skin_temperature", "Skin temperature", "K"),
        238 => entry("tsn", "snow_temperature", "Temperature of snow layer", "K"),
        243 => entry("fal", "forecast_albedo", "Forecast albedo", "(0-1)"),
        244 => entry(
            "fsr",
            "forecast_surface_roughness",
            "Forecast surface roughness",
            "m",
        ),
        245 => entry(
            "flsr",
            "log_surface_roughness_heat",
            "Forecast log of surface roughness for heat",
            "~",
        ),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Field plan
// ---------------------------------------------------------------------------

/// How one message lands in the store: a canonical 2D field (real
/// `FieldSelector`, so production styles resolve for the names the HRRR
/// recipe set knows) or a derived slug (the honest `{"derived": slug}`
/// marker for everything without a units-safe canonical mapping).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PlannedField {
    Canonical {
        name: String,
        selector: FieldSelector,
        units: String,
        /// Multiplier applied to decoded values (1.0 = passthrough;
        /// 1/9.80665 turns geopotential into height).
        scale: f64,
    },
    Derived {
        name: String,
        units: String,
        scale: f64,
    },
}

impl PlannedField {
    pub(crate) fn name(&self) -> &str {
        match self {
            PlannedField::Canonical { name, .. } | PlannedField::Derived { name, .. } => name,
        }
    }

    fn scale(&self) -> f64 {
        match self {
            PlannedField::Canonical { scale, .. } | PlannedField::Derived { scale, .. } => *scale,
        }
    }
}

/// Level suffix for store names, following the iso naming contract
/// (`temperature_850` — `color_tables::iso_levels`): isobaric levels append
/// the bare hPa value, heights append metres, everything else stays explicit
/// about its GRIB level type rather than pretending to be a surface field.
fn level_suffix(level_type: u8, level_value: u16) -> String {
    match level_type {
        1 => String::new(),
        100 => format!("_{level_value}"),
        105 if level_value == 0 => String::new(),
        105 => format!("_{level_value}m"),
        109 => format!("_hyb{level_value}"),
        111 | 112 => format!("_soil{level_value}"),
        _ if level_value == 0 => format!("_lt{level_type}"),
        _ => format!("_lt{level_type}_{level_value}"),
    }
}

/// Map one indexed message to its store field. Canonical selectors are
/// assigned ONLY where the ERA native units match what the WRF import
/// precedent stores for that selector (K, m/s, Pa, gpm, %) — a canonical
/// selector with off-units would let a production style apply its unit
/// arithmetic to the wrong quantity. Geopotential (129) is divided by g and
/// stored as height in gpm, matching every other height field.
pub(crate) fn plan_field(msg: &IndexedMessage) -> PlannedField {
    let level = msg.level_value;
    if msg.center == 98 && msg.table_version == 128 {
        // Surface-ish level types ECMWF uses for its named screen-level
        // params (1 = surface, 105 = fixed height above ground).
        let sfc = matches!(msg.level_type, 1 | 105);
        match (msg.parameter, msg.level_type) {
            (129, 100) => {
                return PlannedField::Canonical {
                    name: format!("height_{level}"),
                    selector: FieldSelector::isobaric(CanonicalField::GeopotentialHeight, level),
                    units: "gpm".to_string(),
                    scale: 1.0 / STANDARD_GRAVITY,
                };
            }
            (129, _) if sfc => {
                return PlannedField::Canonical {
                    name: "orography".to_string(),
                    selector: FieldSelector::surface(CanonicalField::GeopotentialHeight),
                    units: "gpm".to_string(),
                    scale: 1.0 / STANDARD_GRAVITY,
                };
            }
            (156, 100) => {
                return PlannedField::Canonical {
                    name: format!("height_{level}"),
                    selector: FieldSelector::isobaric(CanonicalField::GeopotentialHeight, level),
                    units: "gpm".to_string(),
                    scale: 1.0,
                };
            }
            (130, 100) => {
                return PlannedField::Canonical {
                    name: format!("temperature_{level}"),
                    selector: FieldSelector::isobaric(CanonicalField::Temperature, level),
                    units: "K".to_string(),
                    scale: 1.0,
                };
            }
            (131, 100) => {
                return PlannedField::Canonical {
                    name: format!("u_{level}"),
                    selector: FieldSelector::isobaric(CanonicalField::UWind, level),
                    units: "m/s".to_string(),
                    scale: 1.0,
                };
            }
            (132, 100) => {
                return PlannedField::Canonical {
                    name: format!("v_{level}"),
                    selector: FieldSelector::isobaric(CanonicalField::VWind, level),
                    units: "m/s".to_string(),
                    scale: 1.0,
                };
            }
            (157, 100) => {
                return PlannedField::Canonical {
                    name: format!("relative_humidity_{level}"),
                    selector: FieldSelector::isobaric(CanonicalField::RelativeHumidity, level),
                    units: "%".to_string(),
                    scale: 1.0,
                };
            }
            (134, _) if sfc => {
                return PlannedField::Canonical {
                    name: "surface_pressure".to_string(),
                    selector: FieldSelector::surface(CanonicalField::Pressure),
                    units: "Pa".to_string(),
                    scale: 1.0,
                };
            }
            (151, _) => {
                return PlannedField::Canonical {
                    name: "mslp".to_string(),
                    selector: FieldSelector::mean_sea_level(
                        CanonicalField::PressureReducedToMeanSeaLevel,
                    ),
                    units: "Pa".to_string(),
                    scale: 1.0,
                };
            }
            (165, _) if sfc => {
                return PlannedField::Canonical {
                    name: "u_10m".to_string(),
                    selector: FieldSelector::height_agl(CanonicalField::UWind, 10),
                    units: "m/s".to_string(),
                    scale: 1.0,
                };
            }
            (166, _) if sfc => {
                return PlannedField::Canonical {
                    name: "v_10m".to_string(),
                    selector: FieldSelector::height_agl(CanonicalField::VWind, 10),
                    units: "m/s".to_string(),
                    scale: 1.0,
                };
            }
            (167, _) if sfc => {
                return PlannedField::Canonical {
                    name: "temperature_2m".to_string(),
                    selector: FieldSelector::height_agl(CanonicalField::Temperature, 2),
                    units: "K".to_string(),
                    scale: 1.0,
                };
            }
            (168, _) if sfc => {
                return PlannedField::Canonical {
                    name: "dewpoint_2m".to_string(),
                    selector: FieldSelector::height_agl(CanonicalField::Dewpoint, 2),
                    units: "K".to_string(),
                    scale: 1.0,
                };
            }
            _ => {}
        }
        if let Some(param) = era128_param(msg.parameter) {
            return PlannedField::Derived {
                name: format!(
                    "{}{}",
                    param.slug,
                    level_suffix(msg.level_type, msg.level_value)
                ),
                units: param.units.to_string(),
                scale: 1.0,
            };
        }
    } else if msg.table_version >= 128 {
        // GRIB1 local parameter tables are defined by the originating center;
        // table number 128 from NCEP (center 7), for example, is not ECMWF
        // table 128. Preserve the values under an explicit opaque identity
        // instead of assigning a scientifically false name or unit.
        return PlannedField::Derived {
            name: format!(
                "grib1_c{}_t{}_p{}{}",
                msg.center,
                msg.table_version,
                msg.parameter,
                level_suffix(msg.level_type, msg.level_value)
            ),
            units: String::new(),
            scale: 1.0,
        };
    } else if let Some(entry) =
        grib_core::grib1::parameter_entry(msg.table_version, msg.center, msg.parameter)
            .ok()
            .flatten()
    {
        // WMO-lineage tables, consulted THROUGH the table the message
        // cites: indicators 128-254 resolve only for the center that
        // defined them (NCEP), and a cited table grib-core does not
        // carry comes back Err and stays on the opaque-identity
        // fallthrough below instead of borrowing NCEP's rows.
        if let Some(abbrev) = entry.abbrev {
            let slug: String = abbrev
                .chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() {
                        ch.to_ascii_lowercase()
                    } else {
                        '_'
                    }
                })
                .collect();
            return PlannedField::Derived {
                name: format!("{slug}{}", level_suffix(msg.level_type, msg.level_value)),
                units: entry.units.unwrap_or("").to_string(),
                scale: 1.0,
            };
        }
    }
    PlannedField::Derived {
        name: format!(
            "grib1_c{}_t{}_p{}{}",
            msg.center,
            msg.table_version,
            msg.parameter,
            level_suffix(msg.level_type, msg.level_value)
        ),
        units: String::new(),
        scale: 1.0,
    }
}

// ---------------------------------------------------------------------------
// Grid plan
// ---------------------------------------------------------------------------

/// The run grid plus the scan normalization and column rotation every decoded
/// plane must apply.
/// Rows crossing the signed-longitude seam rotate so longitudes run
/// monotonically through -180..180; rows already monotonic pass through.
pub(crate) struct GridPlan {
    pub nx: usize,
    pub ny: usize,
    /// Eastward-normalized source column index that becomes output column 0.
    pub rotate: usize,
    /// The first serialized row scans from east to west. Output columns are
    /// normalized to the opposite, eastward direction.
    i_negative: bool,
    pub grid: LatLonGrid,
}

fn normalize_lon_180(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

/// Normalize one eastward rectilinear row without breaking regional dateline
/// domains. Nonperiodic rows preserve column order and unwrap continuously
/// (170/180/190 stays that way); no-duplicate periodic rows rotate to a signed
/// axis. Periodicity comes from `Ni * step ~= 360`, so coarse global grids such
/// as 0/90/180/270 no longer depend on a misleading raw-span threshold.
fn normalize_longitude_row(row_lons: &[f64]) -> Result<(usize, Vec<f64>), String> {
    if row_lons.is_empty() {
        return Err("GRIB1 longitude row is empty".to_string());
    }
    let normalized = row_lons
        .iter()
        .map(|longitude| {
            if longitude.is_finite() {
                Ok(normalize_lon_180(*longitude))
            } else {
                Err(format!("GRIB1 grid has non-finite longitude {longitude}"))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if normalized.len() == 1 {
        return Ok((0, normalized));
    }

    let mut unwrapped = Vec::with_capacity(normalized.len());
    unwrapped.push(normalized[0]);
    for &longitude in &normalized[1..] {
        let previous = *unwrapped.last().unwrap();
        let mut longitude = longitude;
        while longitude - previous <= -180.0 {
            longitude += 360.0;
        }
        while longitude - previous > 180.0 {
            longitude -= 360.0;
        }
        unwrapped.push(longitude);
    }
    let step = unwrapped[1] - unwrapped[0];
    if !step.is_finite() || step <= 0.0 {
        return Err(format!(
            "GRIB1 eastward longitude row has invalid first step {step}"
        ));
    }
    let uniform_tolerance = (step.abs() * 1.0e-6).max(1.0e-9);
    if unwrapped
        .windows(2)
        .any(|pair| ((pair[1] - pair[0]) - step).abs() > uniform_tolerance)
    {
        return Err("GRIB1 longitude row is not uniformly rectilinear".to_string());
    }
    let periodic_tolerance = (step * 0.01).max(0.002);
    let no_duplicate_periodic = (unwrapped.len() as f64 * step - 360.0).abs() <= periodic_tolerance;
    let duplicate_endpoint_periodic =
        (((unwrapped.len() - 1) as f64 * step) - 360.0).abs() <= periodic_tolerance;
    let span = unwrapped.last().unwrap() - unwrapped[0];
    if !no_duplicate_periodic && !duplicate_endpoint_periodic && span >= 360.0 {
        return Err(format!(
            "nonperiodic GRIB1 longitude row spans {span} degrees"
        ));
    }

    // A duplicate-endpoint periodic axis cannot be cycle-rotated with one
    // source-column offset without moving its duplicate into the middle. Keep
    // its continuous order; the renderer recognizes `(Ni-1)*step == 360`.
    if !no_duplicate_periodic {
        return Ok((0, unwrapped));
    }

    let seam = normalized
        .windows(2)
        .position(|pair| pair[1] < pair[0])
        .map(|index| index + 1)
        .unwrap_or(0);
    let out = (0..normalized.len())
        .map(|index| normalized[(seam + index) % normalized.len()])
        .collect::<Vec<_>>();
    if out.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err("periodic GRIB1 longitude row has more than one signed seam".to_string());
    }
    Ok((seam, out))
}

/// Validate the GRIB1 scanning-mode octet for the row-major rectilinear lane.
/// We normalize both i directions and accept both j directions. Column-major
/// serialization and reserved bits fail closed instead of being
/// decoded into a plausible-looking but geographically misplaced field.
fn validate_rectilinear_scanning_mode(scanning_mode: u8) -> Result<(), String> {
    if scanning_mode & 0x20 != 0 {
        return Err(
            "j-consecutive (column-major) scanning is not supported; use a row-major GRIB1 product"
                .to_string(),
        );
    }
    if scanning_mode & 0x10 != 0 {
        return Err(format!(
            "GRIB1 scanning mode 0x{scanning_mode:02x} sets reserved bit 0x10; alternative-row scanning is defined for GRIB2, not GRIB1"
        ));
    }
    if scanning_mode & 0x0f != 0 {
        return Err(format!(
            "GRIB1 scanning mode 0x{scanning_mode:02x} sets reserved bit(s) 0x{:02x}",
            scanning_mode & 0x0f
        ));
    }
    Ok(())
}

/// Build the grid plan from a fully parsed first message. Rectilinear grids
/// only (lat/lon and Gaussian); grib-core computes true Gaussian latitudes
/// (Legendre roots), so no linear approximation is involved.
pub(crate) fn build_grid_plan(msg: &Grib1Message) -> Result<GridPlan, String> {
    let gds = msg
        .gds
        .as_ref()
        .ok_or_else(|| "message has no Grid Description Section".to_string())?;
    let (ni, nj, scanning_mode) = match &gds.grid_type {
        GridType::LatLon {
            ni,
            nj,
            scanning_mode,
            ..
        }
        | GridType::Gaussian {
            ni,
            nj,
            scanning_mode,
            ..
        } => (*ni as usize, *nj as usize, *scanning_mode),
        other => {
            return Err(format!(
                "unsupported GRIB1 grid type {other:?} — this importer handles regular \
                 lat/lon and Gaussian grids"
            ));
        }
    };
    let grid_cells = checked_grid_cells(ni, nj)?;
    validate_rectilinear_scanning_mode(scanning_mode)?;
    let coords = msg
        .latlons()
        .map_err(|err| format!("grid coordinates: {err}"))?;
    if coords.len() != grid_cells {
        return Err(format!(
            "grid coordinate count {} does not match Ni x Nj = {}",
            coords.len(),
            grid_cells
        ));
    }

    // Rectilinear: longitudes from the first row, latitudes from the first
    // column (grib-core emits coordinates in data order).
    let source_row_lons: Vec<f64> = coords[..ni].iter().map(|c| c.lon).collect();
    let col_lats: Vec<f64> = (0..nj).map(|j| coords[j * ni].lat).collect();

    // Normalize the first row to +i/eastward before deriving the shared grid.
    let i_negative = scanning_mode & 0x80 != 0;
    let row_lons: Vec<f64> = if i_negative {
        source_row_lons.iter().rev().copied().collect()
    } else {
        source_row_lons
    };

    let (rotate, out_lons) = normalize_longitude_row(&row_lons)?;

    let mut lat_deg = Vec::with_capacity(grid_cells);
    let mut lon_deg = Vec::with_capacity(grid_cells);
    for &lat in &col_lats {
        for &lon in &out_lons {
            lat_deg.push(lat as f32);
            lon_deg.push(lon as f32);
        }
    }

    let shape = GridShape::new(ni, nj).map_err(|err| format!("grid shape: {err}"))?;
    let grid = LatLonGrid::new(shape, lat_deg, lon_deg).map_err(|err| format!("grid: {err}"))?;
    Ok(GridPlan {
        nx: ni,
        ny: nj,
        rotate,
        i_negative,
        grid,
    })
}

/// Normalize scan direction, apply the global column rotation, and scale one
/// decoded plane.
fn rotate_and_scale(values: &[f64], plan: &GridPlan, scale: f64) -> Vec<f32> {
    let (nx, ny, rotate) = (plan.nx, plan.ny, plan.rotate);
    let mut out = Vec::with_capacity(values.len());
    for j in 0..ny {
        let row = &values[j * nx..(j + 1) * nx];
        for i in 0..nx {
            let normalized_i = (rotate + i) % nx;
            let source_i = if plan.i_negative {
                nx - 1 - normalized_i
            } else {
                normalized_i
            };
            out.push((row[source_i] * scale) as f32);
        }
    }
    out
}

/// Read one message's byte range and decode it through grib-core.
fn parse_message_at(
    file: &mut File,
    msg: &IndexedMessage,
    file_label: &str,
) -> Result<Grib1Message, String> {
    let mut bytes = vec![0u8; msg.total_len as usize];
    read_exact_at(file, msg.offset, &mut bytes).map_err(|err| format!("{file_label}: {err}"))?;
    let parsed = Grib1File::from_bytes(&bytes)
        .map_err(|err| format!("{file_label}: message at byte {}: {err}", msg.offset))?;
    parsed
        .messages
        .into_iter()
        .next()
        .ok_or_else(|| format!("{file_label}: message at byte {}: empty parse", msg.offset))
}

/// Decode one message's values through grib-core (24-bit simple packing, IBM
/// reference value, binary/decimal scaling) and shape them for the store.
fn decode_values(
    file: &mut File,
    msg: &IndexedMessage,
    plan: &GridPlan,
    scale: f64,
    file_label: &str,
) -> Result<Vec<f32>, String> {
    let parsed = parse_message_at(file, msg, file_label)?;
    let values = parsed
        .values()
        .map_err(|err| format!("{file_label}: message at byte {}: {err}", msg.offset))?;
    if values.len() != plan.nx * plan.ny {
        return Err(format!(
            "{file_label}: message at byte {}: {} values for a {}x{} grid",
            msg.offset,
            values.len(),
            plan.nx,
            plan.ny
        ));
    }
    Ok(rotate_and_scale(&values, plan, scale))
}

// ---------------------------------------------------------------------------
// Import driver
// ---------------------------------------------------------------------------

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("grib file")
        .to_string()
}

fn sanitize_run_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

fn fnv1a_update(hash: &mut u64, bytes: &[u8]) {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    for byte in bytes {
        *hash = (*hash ^ u64::from(*byte)).wrapping_mul(PRIME);
    }
}

fn hash_identity_part(hash: &mut u64, bytes: &[u8]) {
    fnv1a_update(hash, &(bytes.len() as u64).to_le_bytes());
    fnv1a_update(hash, bytes);
}

/// Stable, order-independent identity for the selected sources and their
/// scientific GRIB header set. Source paths distinguish two same-parameter
/// archives selected from different locations. The complete sorted scientific
/// message sequence includes each valid time plus parameter/level/grid facts,
/// so changing an interior timestep cannot alias the old run and leave stale
/// forecast-hour files behind. No decoded values are read or buffered here.
fn stable_run_identity(paths: &[PathBuf], indexed: &[(usize, IndexedMessage)]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    let mut hash = OFFSET_BASIS;

    let mut sources = paths
        .iter()
        .map(|path| {
            std::fs::canonicalize(path)
                .unwrap_or_else(|_| path.clone())
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect::<Vec<_>>();
    sources.sort();
    sources.dedup();
    hash_identity_part(&mut hash, b"sources");
    for source in sources {
        hash_identity_part(&mut hash, source.as_bytes());
    }

    let mut messages = indexed
        .iter()
        .map(|(_, message)| {
            (
                message.valid_unix,
                message.center,
                message.table_version,
                message.parameter,
                message.level_type,
                message.level_value,
                message.grid_identity.gds_len,
                message.grid_identity.fingerprint,
            )
        })
        .collect::<Vec<_>>();
    messages.sort_unstable();
    messages.dedup();
    hash_identity_part(&mut hash, b"messages");
    fnv1a_update(
        &mut hash,
        &u64::try_from(messages.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for (valid_unix, center, table, parameter, level_type, level_value, gds_len, grid_hash) in
        messages
    {
        fnv1a_update(&mut hash, &valid_unix.to_le_bytes());
        fnv1a_update(&mut hash, &[center, table, parameter, level_type]);
        fnv1a_update(&mut hash, &level_value.to_le_bytes());
        fnv1a_update(&mut hash, &(gds_len as u64).to_le_bytes());
        fnv1a_update(&mut hash, &grid_hash.to_le_bytes());
    }
    hash
}

fn content_bound_run_identity(header_identity: u64, source_identity: &str) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    let mut hash = OFFSET_BASIS;
    hash_identity_part(&mut hash, b"rw-grib1-content-identity-v1");
    fnv1a_update(&mut hash, &header_identity.to_le_bytes());
    hash_identity_part(&mut hash, source_identity.as_bytes());
    hash
}

/// Run name: `{dataset}_{short}_{startYYYYMMDDHH}_{science-schema}_{identity}`. It stays
/// readable in the run browser but cannot alias another same-count source or
/// parameter set, and is deliberately not shaped like an operational cycle.
fn run_name(
    paths: &[PathBuf],
    first: &IndexedMessage,
    first_valid_unix: i64,
    identity: u64,
) -> String {
    let dataset = if first.center == 98 && first.table_version == 128 {
        "era20c"
    } else {
        "grib1"
    };
    let short = if paths.len() > 1 {
        format!("{}files", paths.len())
    } else {
        era_short_from_filename(&paths[0]).unwrap_or_else(|| {
            era128_param(first.parameter)
                .map(|param| param.short.to_string())
                .unwrap_or_else(|| format!("p{}", first.parameter))
        })
    };
    let stamp = chrono::DateTime::from_timestamp(first_valid_unix, 0)
        .map(|time| time.format("%Y%m%d%H").to_string())
        .unwrap_or_else(|| "unknown".to_string());
    sanitize_run_component(&format!(
        "{dataset}_{short}_{stamp}_{IMPORT_SCIENCE_SCHEMA_VERSION}_{identity:016x}"
    ))
}

fn preflight_hour_fields(
    hours: &BTreeMap<u16, Vec<(usize, IndexedMessage)>>,
) -> Result<(), String> {
    for (&hour, group) in hours {
        let mut seen = HashSet::new();
        for (_, message) in group {
            let name = plan_field(message).name().to_string();
            if !seen.insert(name.clone()) {
                return Err(format!(
                    "duplicate GRIB1 field '{name}' at forecast hour f{hour:03}; refusing to begin a partial import"
                ));
            }
        }
    }
    Ok(())
}

/// The GDEX filename grammar
/// (`e20c.oper.an.sfc.3hr.{table}_{param}_{short}.regn80sc.{start}_{end}.grb`)
/// carries the dataset short name — use it when it parses, fall back to the
/// parameter table otherwise.
fn era_short_from_filename(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    let parts: Vec<&str> = stem.split('.').collect();
    let ids = parts.iter().find(|part| {
        let segments: Vec<&str> = part.split('_').collect();
        segments.len() == 3
            && segments[0].chars().all(|ch| ch.is_ascii_digit())
            && segments[1].chars().all(|ch| ch.is_ascii_digit())
            && !segments[2].is_empty()
    })?;
    Some(ids.split('_').nth(2)?.to_string())
}

fn writer_build() -> &'static str {
    concat!(
        "rusty-weather-grib1-local-import-",
        env!("CARGO_PKG_VERSION"),
        "-science_v1"
    )
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn format_utc(unix: i64) -> String {
    chrono::DateTime::from_timestamp(unix, 0)
        .map(|time| time.format("%Y-%m-%d %H:%MZ").to_string())
        .unwrap_or_else(|| format!("unix {unix}"))
}

/// Import one or more GRIB1 files as a single store run. Every distinct
/// valid time becomes one forecast-hour slot (hours since the first
/// timestep); multiple files merge by valid time, so a matching set of
/// single-parameter ERA-20C downloads lands as one multi-variable run.
///
/// Runs on `local_import::spawn_import_paths`' worker thread, which has
/// already dropped itself to below-normal priority.
pub(crate) fn import_grib1_files(
    paths: &[PathBuf],
    store_root: &Path,
    progress: &mut dyn FnMut(String),
) -> Result<LocalImportSummary, String> {
    if paths.is_empty() {
        return Err("no GRIB1 files selected".to_string());
    }
    let source_snapshot = capture_source_set_identity(paths)?;

    // ---- Index every file (headers only — values stay on disk). ----
    let mut indexed: Vec<(usize, IndexedMessage)> = Vec::new();
    for (file_idx, path) in paths.iter().enumerate() {
        progress(format!(
            "GRIB1 {}: indexing messages ({}/{})",
            display_name(path),
            file_idx + 1,
            paths.len()
        ));
        let messages = index_grib1_file(path)?;
        indexed.extend(messages.into_iter().map(|msg| (file_idx, msg)));
    }

    // ---- Reference grid from the first message of the first file. ----
    let first_label = display_name(&paths[0]);
    let mut first_file =
        File::open(&paths[0]).map_err(|err| format!("{first_label}: open: {err}"))?;
    let first_msg = indexed
        .first()
        .map(|(_, message)| message.clone())
        .ok_or_else(|| "selected GRIB1 files contain no messages".to_string())?;
    let plan = build_grid_plan(&parse_message_at(
        &mut first_file,
        &first_msg,
        &first_label,
    )?)
    .map_err(|err| format!("{first_label}: {err}"))?;
    drop(first_file);

    let mut notes: Vec<String> = Vec::new();
    let (ref_ni, ref_nj) = (first_msg.ni, first_msg.nj);
    let skipped_grid = retain_reference_grid(&mut indexed, first_msg.grid_identity);
    if skipped_grid > 0 {
        return Err(format!(
            "{skipped_grid} message(s) use a different Grid Description Section than the first \
             ({ref_ni}x{ref_nj}); refusing a partial mixed-grid import"
        ));
    }

    // ---- Hour keys: hours since the first timestep. ----
    let first_valid = indexed
        .iter()
        .map(|(_, msg)| msg.valid_unix)
        .min()
        .ok_or_else(|| "no importable messages".to_string())?;
    let header_identity = stable_run_identity(paths, &indexed);
    let run_identity = content_bound_run_identity(header_identity, &source_snapshot.identity);
    let mut hours: BTreeMap<u16, Vec<(usize, IndexedMessage)>> = BTreeMap::new();
    let mut skipped_subhour = 0usize;
    let mut skipped_range = 0usize;
    for (file_idx, msg) in indexed {
        let offset_seconds = msg.valid_unix - first_valid;
        if offset_seconds % 3_600 != 0 {
            skipped_subhour += 1;
            continue;
        }
        match u16::try_from(offset_seconds / 3_600) {
            Ok(hour) => hours.entry(hour).or_default().push((file_idx, msg)),
            Err(_) => skipped_range += 1,
        }
    }
    if skipped_subhour > 0 {
        return Err(format!(
            "{skipped_subhour} message(s) have sub-hourly offsets; rw-store hour slots are whole \
             hours, so importing would lose timesteps"
        ));
    }
    if skipped_range > 0 {
        return Err(format!(
            "{skipped_range} message(s) are more than {} hours after the first timestep; \
             rw-store u16 hour keys cannot represent the complete timeline",
            u16::MAX
        ));
    }
    if hours.is_empty() {
        return Err("no importable timesteps".to_string());
    }
    preflight_hour_fields(&hours)?;

    let run = run_name(paths, &first_msg, first_valid, run_identity);
    let model = "wrf".to_string();
    let publisher = RunStagingPublisher::new(store_root, &model, &run)?;
    let staging_store_root = publisher.staging_store_root().to_path_buf();
    let total_hours = hours.len();
    let last_valid = first_valid + i64::from(*hours.keys().next_back().unwrap_or(&0)) * 3_600;

    // ---- Decode-write-drop, one timestep at a time. ----
    let mut files: Vec<File> = Vec::with_capacity(paths.len());
    for path in paths {
        files.push(File::open(path).map_err(|err| format!("{}: open: {err}", display_name(path)))?);
    }
    let mut all_vars: Vec<String> = Vec::new();
    let mut hours_written = 0usize;
    for (step, (&hour, group)) in hours.iter().enumerate() {
        let mut canonical: Vec<(String, SelectedField2D)> = Vec::new();
        let mut derived: Vec<(String, String, Vec<f32>)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for (file_idx, msg) in group {
            let field = plan_field(msg);
            if !seen.insert(field.name().to_string()) {
                return Err(format!(
                    "duplicate GRIB1 field '{}' at forecast hour f{hour:03}; refusing to keep only one occurrence",
                    field.name()
                ));
            }
            let label = display_name(&paths[*file_idx]);
            let values = decode_values(&mut files[*file_idx], msg, &plan, field.scale(), &label)?;
            match field {
                PlannedField::Canonical {
                    name,
                    selector,
                    units,
                    ..
                } => {
                    let selected = SelectedField2D::new(selector, units, plan.grid.clone(), values)
                        .map_err(|err| format!("{label}: field {name}: {err}"))?;
                    canonical.push((name, selected));
                }
                PlannedField::Derived { name, units, .. } => {
                    derived.push((name, units, values));
                }
            }
        }

        // Wind speed companions where both components landed (same derived-
        // at-ingest convention as the WRF import's `wind_speed_10m`).
        let mut speeds: Vec<(String, SelectedField2D)> = Vec::new();
        for (name, u_field) in &canonical {
            let Some(rest) = name.strip_prefix("u_") else {
                continue;
            };
            let Some((_, v_field)) = canonical
                .iter()
                .find(|(v_name, _)| v_name == &format!("v_{rest}"))
            else {
                continue;
            };
            let speed_name = format!("wind_speed_{rest}");
            if seen.contains(&speed_name) {
                continue;
            }
            let selector = if rest == "10m" {
                FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            } else if let Ok(level) = rest.parse::<u16>() {
                FieldSelector::isobaric(CanonicalField::WindSpeed, level)
            } else {
                continue;
            };
            let values: Vec<f32> = u_field
                .values
                .iter()
                .zip(&v_field.values)
                .map(|(u, v)| u.mul_add(*u, v * v).sqrt())
                .collect();
            if let Ok(selected) = SelectedField2D::new(selector, "m/s", plan.grid.clone(), values) {
                seen.insert(speed_name.clone());
                speeds.push((speed_name, selected));
            }
        }
        canonical.extend(speeds);

        progress(format!(
            "GRIB1 {run}: timestep {}/{total_hours} (f{hour:03}, {}) — {} field(s)",
            step + 1,
            format_utc(first_valid + i64::from(hour) * 3_600),
            canonical.len() + derived.len(),
        ));

        let refs: Vec<(&str, &SelectedField2D)> = canonical
            .iter()
            .map(|(name, field)| (name.as_str(), field))
            .collect();
        let raw_refs: Vec<DerivedFieldInput<'_>> = derived
            .iter()
            .map(|(name, units, values)| DerivedFieldInput {
                name,
                units,
                values,
            })
            .collect();
        // The grid-aware seam permits honest derived-only hours (for example
        // forecast surface roughness) without inventing a canonical selector
        // solely to carry coordinates.
        let written = write_hour_from_grid_with_derived(
            &staging_store_root,
            &model,
            &run,
            hour,
            &plan.grid,
            None,
            &refs,
            &raw_refs,
            &[],
            writer_build(),
            now_unix(),
        )
        .map_err(|err| format!("store write f{hour:03}: {err}"))?;
        all_vars.extend(written.vars);
        hours_written += 1;
    }

    notes.push(format!(
        "{hours_written} timestep(s), {} to {}",
        format_utc(first_valid),
        format_utc(last_valid)
    ));

    all_vars.sort();
    all_vars.dedup();
    verify_source_set_unchanged(&source_snapshot)?;
    progress(format!("GRIB1 {run}: publishing complete run"));
    publisher.publish()?;
    Ok(LocalImportSummary {
        store_root: store_root.to_path_buf(),
        model,
        run,
        files_seen: paths.len(),
        hours_written,
        variables: all_vars,
        notes,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// First message (bytes 0..153,708) of the owner's real ERA-20C file
    /// `e20c.oper.an.sfc.3hr.128_244_fsr.regn80sc.2004010100_2004123121.grb`
    /// (ECMWF, table 128, param 244 fsr, N80 Gaussian 320x160, 24-bit simple
    /// packing) — vendored whole so the regression runs the exact bytes the
    /// import path sees.
    fn fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/e20c_fsr_2004010100_msg0.grb")
    }

    fn fixture_bytes() -> Vec<u8> {
        std::fs::read(fixture_path()).expect("read vendored ERA-20C fixture")
    }

    fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "bowecho-grib1-{name}-{}-{}",
            std::process::id(),
            now_unix()
        ));
        std::fs::write(&path, bytes).expect("write temp grib");
        path
    }

    fn synthetic_grid_plan(nx: usize, ny: usize, rotate: usize, i_negative: bool) -> GridPlan {
        let shape = GridShape::new(nx, ny).expect("synthetic shape");
        let cells = nx * ny;
        GridPlan {
            nx,
            ny,
            rotate,
            i_negative,
            grid: LatLonGrid::new(shape, vec![0.0; cells], vec![0.0; cells])
                .expect("synthetic grid"),
        }
    }

    #[test]
    fn rectilinear_scan_mode_accepts_normalizable_flags_and_rejects_others() {
        // +/- i and +/- j combinations are accepted and normalized.
        for scanning_mode in [0x00_u8, 0x40, 0x80, 0xc0] {
            assert!(
                validate_rectilinear_scanning_mode(scanning_mode).is_ok(),
                "scan mode 0x{scanning_mode:02x}"
            );
        }

        for scanning_mode in [0x20_u8, 0x60, 0xa0, 0xe0] {
            let error = validate_rectilinear_scanning_mode(scanning_mode)
                .expect_err("column-major scan must fail closed");
            assert!(error.contains("column-major"), "{error}");
        }
        for scanning_mode in [0x10_u8, 0x50] {
            let error = validate_rectilinear_scanning_mode(scanning_mode)
                .expect_err("GRIB2-only alternative-row flag must fail closed");
            assert!(error.contains("GRIB2, not GRIB1"), "{error}");
        }
        for scanning_mode in [0x01_u8, 0x08] {
            let error = validate_rectilinear_scanning_mode(scanning_mode)
                .expect_err("reserved scan bits must fail closed");
            assert!(error.contains("reserved bit"), "{error}");
        }
    }

    #[test]
    fn longitude_seam_detection_handles_coarse_and_regional_rows() {
        let (rotate, normalized) =
            normalize_longitude_row(&[0.0, 90.0, 180.0, 270.0]).expect("coarse global row");
        assert_eq!(rotate, 2);
        assert_eq!(normalized, vec![-180.0, -90.0, 0.0, 90.0]);

        let (rotate, normalized) =
            normalize_longitude_row(&[170.0, 180.0, 190.0]).expect("regional seam row");
        assert_eq!(rotate, 0);
        assert_eq!(normalized, vec![170.0, 180.0, 190.0]);

        let error = normalize_longitude_row(&[0.0, 180.0, 0.0, 180.0])
            .expect_err("a nonperiodic row cannot span more than a revolution");
        assert!(error.contains("spans 540"), "{error}");

        let (rotate, duplicate) = normalize_longitude_row(&[0.0, 90.0, 180.0, 270.0, 360.0])
            .expect("duplicate-endpoint periodic row");
        assert_eq!(rotate, 0);
        assert_eq!(duplicate, vec![0.0, 90.0, 180.0, 270.0, 360.0]);
    }

    #[test]
    fn negative_i_rows_are_normalized_before_column_rotation() {
        let eastward = synthetic_grid_plan(4, 3, 1, false);
        let eastward_values = vec![
            0.0, 1.0, 2.0, 3.0, // row 0 eastward
            10.0, 11.0, 12.0, 13.0, // row 1 eastward
            20.0, 21.0, 22.0, 23.0, // row 2 eastward
        ];
        assert_eq!(
            rotate_and_scale(&eastward_values, &eastward, 2.0),
            vec![
                2.0, 4.0, 6.0, 0.0, 22.0, 24.0, 26.0, 20.0, 42.0, 44.0, 46.0, 40.0,
            ]
        );

        // With -i, every serialized row is reversed; the normalized
        // geographic result remains identical.
        let westward = synthetic_grid_plan(4, 3, 1, true);
        let westward_values = vec![
            3.0, 2.0, 1.0, 0.0, // row 0 westward
            13.0, 12.0, 11.0, 10.0, // row 1 westward
            23.0, 22.0, 21.0, 20.0, // row 2 westward
        ];
        assert_eq!(
            rotate_and_scale(&westward_values, &westward, 2.0),
            rotate_and_scale(&eastward_values, &eastward, 2.0)
        );
    }

    /// Small index-only GRIB1 record. It is intentionally not a decodable BDS
    /// fixture: the synthetic tests exercise header/GDS validation without the
    /// optional 150-KB ERA-20C binary.
    fn synthetic_index_message(ni: u16, nj: u16, grid_type: u8) -> Vec<u8> {
        const PDS_LEN: usize = 28;
        const GDS_LEN: usize = 10;
        const TOTAL_LEN: usize = 8 + PDS_LEN + GDS_LEN + 4;
        let mut bytes = vec![0u8; TOTAL_LEN];
        bytes[0..4].copy_from_slice(b"GRIB");
        bytes[4..7].copy_from_slice(&[
            ((TOTAL_LEN >> 16) & 0xff) as u8,
            ((TOTAL_LEN >> 8) & 0xff) as u8,
            (TOTAL_LEN & 0xff) as u8,
        ]);
        bytes[7] = 1;

        let pds = &mut bytes[8..8 + PDS_LEN];
        pds[0..3].copy_from_slice(&[0, 0, PDS_LEN as u8]);
        pds[3] = 128;
        pds[4] = 98;
        pds[7] = 0x80;
        pds[8] = 244;
        pds[9] = 1;
        pds[12] = 24;
        pds[13] = 1;
        pds[14] = 1;
        pds[17] = 1;
        pds[24] = 21;

        let gds_start = 8 + PDS_LEN;
        let gds = &mut bytes[gds_start..gds_start + GDS_LEN];
        gds[0..3].copy_from_slice(&[0, 0, GDS_LEN as u8]);
        gds[5] = grid_type;
        gds[6..8].copy_from_slice(&ni.to_be_bytes());
        gds[8..10].copy_from_slice(&nj.to_be_bytes());
        bytes[TOTAL_LEN - 4..].copy_from_slice(b"7777");
        bytes
    }

    #[test]
    fn synthetic_index_rejects_zero_grid_dimensions() {
        for (tag, ni, nj) in [("zero-ni", 0, 160), ("zero-nj", 320, 0)] {
            let path = temp_file(tag, &synthetic_index_message(ni, nj, 4));
            let error = index_grib1_file(&path).expect_err("zero grid must be rejected");
            std::fs::remove_file(path).ok();
            assert!(
                error.contains("zero-sized grid"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn synthetic_index_rejects_grid_before_hostile_coordinate_allocation() {
        let path = temp_file("oversized-grid", &synthetic_index_message(5_001, 5_000, 4));
        let error = index_grib1_file(&path).expect_err("oversized grid must be rejected");
        std::fs::remove_file(path).ok();
        assert!(
            error.contains("desktop safety ceiling"),
            "unexpected error: {error}"
        );
        assert!(checked_grid_cells(usize::MAX, 2).is_err());
    }

    #[test]
    fn synthetic_index_rejects_unsupported_time_range_indicators() {
        for tri in [6_u8, 7, 51, 113, 125, 255] {
            let mut bytes = synthetic_index_message(320, 160, 4);
            bytes[8 + 20] = tri;
            let path = temp_file(&format!("unsupported-tri-{tri}"), &bytes);
            let error = index_grib1_file(&path)
                .expect_err("unsupported time range indicator must fail closed");
            std::fs::remove_file(path).ok();
            assert!(
                error.contains(&format!("time range indicator {tri}")),
                "unexpected error for TRI {tri}: {error}"
            );
        }
    }

    #[test]
    fn message_index_ceiling_rejects_the_next_header() {
        assert!(ensure_message_index_capacity(MAX_GRIB1_MESSAGES_PER_FILE - 1).is_ok());
        assert!(ensure_message_index_capacity(MAX_GRIB1_MESSAGES_PER_FILE).is_err());
    }

    #[test]
    fn synthetic_same_dimensions_with_different_gds_are_not_merged() {
        let first = synthetic_index_message(320, 160, 0);
        let second = synthetic_index_message(320, 160, 4);
        let mut bytes = first;
        bytes.extend_from_slice(&second);
        let path = temp_file("grid-mismatch", &bytes);
        let messages = index_grib1_file(&path).expect("synthetic headers index");
        std::fs::remove_file(path).ok();
        assert_eq!(messages.len(), 2);
        assert_eq!((messages[0].ni, messages[0].nj), (320, 160));
        assert_eq!((messages[1].ni, messages[1].nj), (320, 160));
        assert_ne!(messages[0].grid_identity, messages[1].grid_identity);

        let reference = messages[0].grid_identity;
        let mut indexed = messages.into_iter().enumerate().collect::<Vec<_>>();
        assert_eq!(retain_reference_grid(&mut indexed, reference), 1);
        assert_eq!(indexed.len(), 1);
        assert_eq!(indexed[0].1.grid_identity, reference);
    }

    #[test]
    #[ignore = "requires optional ERA-20C binary fixture"]
    fn fixture_index_reads_pds_and_gds_facts() {
        let msgs = index_grib1_file(&fixture_path()).expect("index fixture");
        assert_eq!(msgs.len(), 1);
        let msg = &msgs[0];
        assert_eq!(msg.offset, 0);
        assert_eq!(msg.total_len, 153_708);
        assert_eq!(msg.table_version, 128);
        assert_eq!(msg.center, 98);
        assert_eq!(msg.parameter, 244);
        assert_eq!(msg.level_type, 1);
        assert_eq!(msg.level_value, 0);
        assert_eq!((msg.ni, msg.nj), (320, 160));
        // Reference time 2004-01-01 00:00Z, analysis (tri 0, P1 0).
        let expected = chrono::NaiveDate::from_ymd_opt(2004, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        assert_eq!(msg.valid_unix, expected);
    }

    #[test]
    #[ignore = "requires optional ERA-20C binary fixture"]
    fn index_walks_concatenated_messages_and_padding() {
        let bytes = fixture_bytes();
        let mut doubled = bytes.clone();
        // Inter-message padding: the indexer must scan forward to the next
        // magic rather than assume back-to-back records.
        doubled.extend_from_slice(&[0u8; 16]);
        doubled.extend_from_slice(&bytes);
        let path = temp_file("concat", &doubled);
        let msgs = index_grib1_file(&path).expect("index concatenated");
        std::fs::remove_file(&path).ok();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].offset, 0);
        assert_eq!(msgs[1].offset, 153_708 + 16);
        assert_eq!(msgs[1].total_len, 153_708);
    }

    #[test]
    #[ignore = "requires optional ERA-20C binary fixture"]
    fn index_rejects_grib2_editions() {
        let mut bytes = fixture_bytes();
        bytes[7] = 2;
        let path = temp_file("edition2", &bytes);
        let err = index_grib1_file(&path).expect_err("edition 2 must be rejected");
        std::fs::remove_file(&path).ok();
        assert!(err.contains("edition 2"), "unexpected error: {err}");
    }

    #[test]
    #[ignore = "requires optional ERA-20C binary fixture"]
    fn fixture_unpack_matches_hand_decoded_values() {
        // Hand-decoded from the fixture's hex (see grib1-import-notes.md):
        // BDS binary scale E = -23 (0x8017 sign-magnitude), reference
        // R = IBM 0x3D195DE7 = 1_662_439 * 2^-36, 24 bits/value, decimal
        // scale 0. First three packed integers: 8166, 8166, 8165.
        let reference = 1_662_439.0 * (2.0_f64).powi(-36);
        let v0 = reference + 8_166.0 * (2.0_f64).powi(-23);
        let v2 = reference + 8_165.0 * (2.0_f64).powi(-23);

        let msgs = index_grib1_file(&fixture_path()).expect("index fixture");
        let mut file = File::open(fixture_path()).expect("open fixture");
        let parsed = parse_message_at(&mut file, &msgs[0], "fixture").expect("parse");
        let plan = build_grid_plan(&parsed).expect("grid plan");
        let values = decode_values(&mut file, &msgs[0], &plan, 1.0, "fixture").expect("decode");

        assert_eq!(values.len(), 320 * 160);
        // The global grid rotates by 160 columns (lon 180 -> output column
        // 0), so source column 0 lands at output column 160.
        assert_eq!(plan.rotate, 160);
        assert!(
            (f64::from(values[160]) - v0).abs() < 1e-9,
            "values[160] = {}, hand-decoded {v0}",
            values[160]
        );
        assert!(
            (f64::from(values[162]) - v2).abs() < 1e-9,
            "values[162] = {}, hand-decoded {v2}",
            values[162]
        );
        // Physical plausibility across the whole plane: surface roughness is
        // non-negative, meters, small over ocean/ice and < 10 m everywhere.
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        for &value in &values {
            assert!(value.is_finite());
            min = min.min(value);
            max = max.max(value);
        }
        assert!(min >= 0.0, "roughness must be non-negative, got {min}");
        assert!(max < 10.0, "roughness above 10 m is implausible, got {max}");
        assert!(max > 0.1, "land roughness should exceed 0.1 m, got {max}");
    }

    #[test]
    #[ignore = "requires optional ERA-20C binary fixture"]
    fn fixture_grid_rotates_to_monotonic_signed_longitudes() {
        let msgs = index_grib1_file(&fixture_path()).expect("index fixture");
        let mut file = File::open(fixture_path()).expect("open fixture");
        let parsed = parse_message_at(&mut file, &msgs[0], "fixture").expect("parse");
        let plan = build_grid_plan(&parsed).expect("grid plan");

        assert_eq!((plan.nx, plan.ny), (320, 160));
        // First Gaussian latitude for N80 is 89.1416 (Legendre root), which
        // the GDS encodes as 89.142 millidegrees-truncated.
        let lat0 = f64::from(plan.grid.lat_deg[0]);
        assert!((lat0 - 89.1416).abs() < 0.01, "lat0 = {lat0}");
        let lat_last = f64::from(plan.grid.lat_deg[(160 - 1) * 320]);
        assert!((lat_last + 89.1416).abs() < 0.01, "lat_last = {lat_last}");
        // Longitudes: -180 .. 178.875 step 1.125, strictly ascending — the
        // map layer's inverse LUT does not wrap 0..360 grids.
        let lons: Vec<f32> = plan.grid.lon_deg[..320].to_vec();
        assert!(
            (f64::from(lons[0]) + 180.0).abs() < 1e-6,
            "lon0 = {}",
            lons[0]
        );
        assert!(
            (f64::from(lons[319]) - 178.875).abs() < 1e-3,
            "lon_last = {}",
            lons[319]
        );
        assert!(
            lons.windows(2).all(|pair| pair[1] > pair[0]),
            "rotated longitudes must ascend monotonically"
        );
        // Latitude constant along a row.
        assert_eq!(plan.grid.lat_deg[0], plan.grid.lat_deg[319]);
    }

    #[test]
    fn era128_param_labels_cover_the_task_set() {
        let fsr = era128_param(244).expect("fsr");
        assert_eq!(fsr.short, "fsr");
        assert_eq!(fsr.slug, "forecast_surface_roughness");
        assert_eq!(fsr.label, "Forecast surface roughness");
        assert_eq!(fsr.units, "m");
        for (param, short) in [
            (129u8, "z"),
            (130, "t"),
            (131, "u"),
            (132, "v"),
            (133, "q"),
            (134, "sp"),
            (151, "msl"),
            (165, "10u"),
            (166, "10v"),
            (167, "2t"),
            (168, "2d"),
            (228, "tp"),
            (59, "cape"),
        ] {
            assert_eq!(era128_param(param).expect("param").short, short);
        }
        assert!(era128_param(0).is_none());
    }

    fn indexed(parameter: u8, level_type: u8, level_value: u16) -> IndexedMessage {
        IndexedMessage {
            offset: 0,
            total_len: 0,
            table_version: 128,
            center: 98,
            parameter,
            level_type,
            level_value,
            valid_unix: 0,
            ni: 320,
            nj: 160,
            grid_identity: grid_identity(&[0; 10]),
        }
    }

    #[test]
    fn field_plan_maps_canonical_and_derived_params() {
        // 2 m temperature: canonical with the WRF-import store name.
        match plan_field(&indexed(167, 1, 0)) {
            PlannedField::Canonical { name, units, .. } => {
                assert_eq!(name, "temperature_2m");
                assert_eq!(units, "K");
            }
            other => panic!("2t must be canonical, got {other:?}"),
        }
        // 850 hPa temperature: iso naming contract slug (temperature_850).
        match plan_field(&indexed(130, 100, 850)) {
            PlannedField::Canonical { name, .. } => assert_eq!(name, "temperature_850"),
            other => panic!("t850 must be canonical, got {other:?}"),
        }
        // Geopotential at 500 hPa: stored as height (gpm), scaled by 1/g.
        match plan_field(&indexed(129, 100, 500)) {
            PlannedField::Canonical {
                name, units, scale, ..
            } => {
                assert_eq!(name, "height_500");
                assert_eq!(units, "gpm");
                assert!((scale - 1.0 / STANDARD_GRAVITY).abs() < 1e-12);
            }
            other => panic!("z500 must be canonical height, got {other:?}"),
        }
        // fsr: no canonical mapping — derived slug with ERA units.
        match plan_field(&indexed(244, 1, 0)) {
            PlannedField::Derived { name, units, .. } => {
                assert_eq!(name, "forecast_surface_roughness");
                assert_eq!(units, "m");
            }
            other => panic!("fsr must be derived, got {other:?}"),
        }
        // Specific humidity on a level keeps the level suffix.
        match plan_field(&indexed(133, 100, 700)) {
            PlannedField::Derived { name, .. } => assert_eq!(name, "specific_humidity_700"),
            other => panic!("q700 must be derived, got {other:?}"),
        }
        // Unknown parameter in an unknown table: self-describing fallback.
        let mut unknown = indexed(250, 1, 0);
        unknown.table_version = 200;
        match plan_field(&unknown) {
            PlannedField::Derived { name, units, .. } => {
                assert_eq!(name, "grib1_c98_t200_p250");
                assert!(units.is_empty());
            }
            other => panic!("unknown param must be derived, got {other:?}"),
        }
    }

    #[test]
    fn non_ncep_center_defined_region_stays_opaque_in_wmo_versions() {
        // Center 34 (JMA) citing table version 3, parameter 144: the
        // 128-254 region is center-defined, and the rows grib-core
        // carries there are NCEP's ("Volumetric soil moisture content").
        // Borrowing them for JMA would assign a scientifically false
        // name; the plan keeps the values under an opaque identity.
        let mut message = indexed(144, 1, 0);
        message.center = 34;
        message.table_version = 3;
        match plan_field(&message) {
            PlannedField::Derived { name, units, .. } => {
                assert_eq!(name, "grib1_c34_t3_p144");
                assert!(units.is_empty());
            }
            other => panic!("JMA local region must not borrow NCEP rows: {other:?}"),
        }
        // The international region (1-127) still resolves for any center.
        let mut wmo = indexed(11, 100, 850);
        wmo.center = 34;
        wmo.table_version = 3;
        match plan_field(&wmo) {
            PlannedField::Derived { name, units, .. } => {
                assert_eq!(name, "tmp_850");
                assert_eq!(units, "K");
            }
            other => panic!("WMO international region must resolve: {other:?}"),
        }
    }

    #[test]
    fn non_ecmwf_local_table_128_stays_opaque() {
        let mut message = indexed(167, 1, 0);
        message.center = 7;
        match plan_field(&message) {
            PlannedField::Derived { name, units, .. } => {
                assert_eq!(name, "grib1_c7_t128_p167");
                assert!(units.is_empty());
            }
            other => panic!("non-ECMWF table 128 must not become 2 m temperature: {other:?}"),
        }
    }

    #[test]
    fn grib_run_identity_is_order_independent() {
        let base =
            std::env::temp_dir().join(format!("grib-run-identity-order-{}", std::process::id()));
        let paths = vec![base.join("temperature.grb"), base.join("wind.grb")];
        let reordered_paths = vec![paths[1].clone(), paths[0].clone()];
        let temperature = indexed(130, 100, 850);
        let mut wind = indexed(131, 100, 850);
        wind.valid_unix = 10_800;
        let messages = vec![(0, temperature.clone()), (1, wind.clone())];
        let reordered_messages = vec![(0, wind), (1, temperature.clone())];

        let identity = stable_run_identity(&paths, &messages);
        let reordered = stable_run_identity(&reordered_paths, &reordered_messages);

        assert_eq!(identity, reordered);
        assert_eq!(
            run_name(&paths, &temperature, 0, identity),
            run_name(&reordered_paths, &temperature, 0, reordered)
        );
    }

    #[test]
    fn grib_run_identity_separates_same_count_source_and_parameter_sets() {
        let base =
            std::env::temp_dir().join(format!("grib-run-identity-distinct-{}", std::process::id()));
        let paths_ab = vec![base.join("a.grb"), base.join("b.grb")];
        let paths_ac = vec![base.join("a.grb"), base.join("c.grb")];
        let temperature = indexed(130, 100, 850);
        let u_wind = indexed(131, 100, 850);
        let v_wind = indexed(132, 100, 850);
        let temperature_u = vec![(0, temperature.clone()), (1, u_wind)];
        let temperature_v = vec![(0, temperature.clone()), (1, v_wind)];

        let base_identity = stable_run_identity(&paths_ab, &temperature_u);
        let different_parameter = stable_run_identity(&paths_ab, &temperature_v);
        let different_source = stable_run_identity(&paths_ac, &temperature_u);

        assert_ne!(base_identity, different_parameter);
        assert_ne!(base_identity, different_source);
        let base_run = run_name(&paths_ab, &temperature, 0, base_identity);
        let parameter_run = run_name(&paths_ab, &temperature, 0, different_parameter);
        let source_run = run_name(&paths_ac, &temperature, 0, different_source);
        assert_ne!(base_run, parameter_run);
        assert_ne!(base_run, source_run);
        assert!(
            base_run.contains(IMPORT_SCIENCE_SCHEMA_VERSION),
            "{base_run}"
        );
        let suffix = base_run.rsplit('_').next().expect("identity suffix");
        assert_eq!(suffix.len(), 16);
        assert!(suffix.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn grib_duplicate_fields_fail_preflight_before_any_hour_write() {
        let duplicate = indexed(130, 100, 850);
        let mut hours = BTreeMap::new();
        hours.insert(0, vec![(0, duplicate.clone()), (1, duplicate)]);
        let err = preflight_hour_fields(&hours).unwrap_err();
        assert!(
            err.contains("temperature_850") && err.contains("f000"),
            "{err}"
        );
    }

    #[test]
    fn grib_run_identity_is_bound_to_full_source_content_digest() {
        let header_identity = 0x1234_5678_9abc_def0;
        assert_ne!(
            content_bound_run_identity(header_identity, "sha256-a"),
            content_bound_run_identity(header_identity, "sha256-b")
        );
    }

    #[test]
    fn grib_run_identity_separates_changed_interior_valid_times() {
        let paths = vec![std::env::temp_dir().join(format!(
            "grib-run-identity-interior-{}.grb",
            std::process::id()
        ))];
        let mut first = indexed(130, 100, 850);
        first.valid_unix = 0;
        let mut middle = first.clone();
        middle.valid_unix = 10_800;
        let mut last = first.clone();
        last.valid_unix = 21_600;
        let original = vec![(0, first.clone()), (0, middle.clone()), (0, last.clone())];
        middle.valid_unix = 14_400;
        let changed = vec![(0, first), (0, middle), (0, last)];

        assert_ne!(
            stable_run_identity(&paths, &original),
            stable_run_identity(&paths, &changed),
            "equal count/endpoints must not hide a changed interior timestep"
        );
    }

    #[test]
    fn forecast_offsets_follow_time_unit_and_range_indicator() {
        // Analysis: tri 0, P1 0 (the ERA-20C case).
        assert_eq!(forecast_offset_seconds(1, 0, 0, 0), Ok(0));
        // Initialized analysis (tri 1) is valid exactly at reference time.
        assert_eq!(forecast_offset_seconds(1, 0, 0, 1), Ok(0));
        assert!(forecast_offset_seconds(1, 1, 0, 1).is_err());
        // 3-hour forecast in hour units.
        assert_eq!(forecast_offset_seconds(1, 3, 0, 0), Ok(3 * 3_600));
        // Day units.
        assert_eq!(forecast_offset_seconds(2, 2, 0, 0), Ok(2 * 86_400));
        // Accumulation valid at the end of (P1, P2).
        assert_eq!(forecast_offset_seconds(1, 0, 6, 4), Ok(6 * 3_600));
        assert!(forecast_offset_seconds(1, 6, 3, 4).is_err());
        // Two-octet P1 (tri 10).
        assert_eq!(forecast_offset_seconds(1, 1, 4, 10), Ok(260 * 3_600));
        // Calendar units have no fixed length.
        assert!(forecast_offset_seconds(3, 1, 0, 0).is_err());
        for tri in [6, 7, 51] {
            assert!(forecast_offset_seconds(1, 1, 2, tri).is_err(), "TRI {tri}");
        }
        for tri in 113..=125 {
            assert!(forecast_offset_seconds(1, 1, 2, tri).is_err(), "TRI {tri}");
        }
    }

    #[test]
    #[ignore = "requires optional ERA-20C binary fixture"]
    fn fixture_imports_to_store_and_reads_back() {
        let store_root = std::env::temp_dir().join(format!(
            "bowecho-grib1-store-{}-{}",
            std::process::id(),
            now_unix()
        ));
        let mut lines = Vec::new();
        let summary =
            import_grib1_files(&[fixture_path()], &store_root, &mut |line| lines.push(line))
                .expect("import fixture");

        assert_eq!(summary.model, "wrf");
        assert!(summary.run.starts_with("era20c_fsr_2004010100_"));
        assert_eq!(summary.run.rsplit('_').next().map(str::len), Some(16));
        assert_eq!(summary.hours_written, 1);
        assert_eq!(summary.variables, vec!["forecast_surface_roughness"]);
        assert!(!lines.is_empty());

        let hour_path = store_root
            .join(&summary.model)
            .join(&summary.run)
            .join("f000.rws");
        let reader = rw_store::reader::HourReader::open(&hour_path).expect("open written hour");
        let var = reader
            .variable("forecast_surface_roughness")
            .expect("written variable");
        assert_eq!(var.units, "m");
        assert_eq!(
            rw_store::derived_selector_slug(&var.selector),
            Some("forecast_surface_roughness"),
            "derived-only import must not forge a canonical height selector"
        );
        let values = reader
            .read_full_2d("forecast_surface_roughness")
            .expect("read plane back");
        assert_eq!(values.len(), 320 * 160);
        // Same hand-decoded first value as the unpack test, at its rotated
        // column, surviving the store round-trip (f32 store codec).
        let v0 = 1_662_439.0 * (2.0_f64).powi(-36) + 8_166.0 * (2.0_f64).powi(-23);
        assert!(
            (f64::from(values[160]) - v0).abs() < 1e-6,
            "store round-trip values[160] = {}, expected {v0}",
            values[160]
        );

        std::fs::remove_dir_all(&store_root).ok();
    }

    /// Full-file proof against the owner's real 450 MB ERA-20C download
    /// (env-gated like `RW_LOCAL_IMPORT_FIXTURE`): index all 2,928 messages,
    /// verify the 3-hourly axis spans the year monotonically, and decode the
    /// first/middle/last planes through the real path.
    #[test]
    fn optional_era20c_full_file_indexes_and_decodes() {
        let Ok(fixture) = std::env::var("RW_ERA20C_GRIB_FIXTURE") else {
            eprintln!("skipping ERA-20C full-file test; set RW_ERA20C_GRIB_FIXTURE");
            return;
        };
        let path = PathBuf::from(&fixture);
        let started = std::time::Instant::now();
        let msgs = index_grib1_file(&path).expect("index full file");
        let index_elapsed = started.elapsed();
        assert_eq!(msgs.len(), 2_928, "expected exactly 2,928 messages");

        // Monotonic 3-hourly valid times across the whole year.
        for pair in msgs.windows(2) {
            assert_eq!(
                pair[1].valid_unix - pair[0].valid_unix,
                3 * 3_600,
                "3-hourly step broken between offsets {} and {}",
                pair[0].offset,
                pair[1].offset
            );
        }
        let span_hours = (msgs.last().unwrap().valid_unix - msgs[0].valid_unix) / 3_600;
        assert_eq!(span_hours, 8_781, "year of 3-hourly steps spans 8,781 h");

        let mut file = File::open(&path).expect("open full file");
        let plan =
            build_grid_plan(&parse_message_at(&mut file, &msgs[0], "full").expect("parse first"))
                .expect("grid plan");
        let decode_started = std::time::Instant::now();
        for msg in [&msgs[0], &msgs[msgs.len() / 2], &msgs[msgs.len() - 1]] {
            let values = decode_values(&mut file, msg, &plan, 1.0, "full").expect("decode");
            assert_eq!(values.len(), 320 * 160);
            assert!(
                values
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
            );
        }
        eprintln!(
            "ERA-20C full file: indexed {} messages in {:.2?}, decoded 3 planes in {:.2?}",
            msgs.len(),
            index_elapsed,
            decode_started.elapsed()
        );
    }
}
