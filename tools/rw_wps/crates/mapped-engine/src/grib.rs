//! GRIB records, decoded in process.
//!
//! Port of `mapped_source._grib2_inventory` / `_grib2_records` /
//! `_regular_latlon_frame` / `_selector_matches_record` — with the
//! subprocess and its TSV removed.  The numbers are unchanged: the Python
//! engine's TSV carried `grib-core`'s own struct fields through Rust's
//! shortest-round-trip `Display` and Python's exact `float()`, so reading
//! the same fields directly yields the same f64 bit patterns.  The one
//! place the TSV spelling still matters is the grid fingerprint, which
//! hashes the RENDERED strings; `grid_fingerprint` below reproduces that
//! rendering exactly, including the `0x40` scan-mode spelling.

use std::collections::BTreeMap;

use chrono::{Duration, NaiveDateTime};
use grib_core::grib2::{unpack_message, Grib2File, Grib2Message};
use ndarray::{ArrayD, IxDyn};

use crate::model::{FieldSpec, GridDeclaration, Mapping, GRIB2_GRID_RELATIVE_WIND_BIT};
use crate::refusal::{decode_failed, grid_mismatch, Result};

/// One decoded GRIB record — `mapped_source._GribRecord`.
#[derive(Debug, Clone)]
pub struct GribRecord {
    /// The path provenance is bound to: the object acquisition delivered,
    /// NOT the staged decompressed twin the parser read.
    pub source: String,
    pub index: usize,
    pub reference_time: NaiveDateTime,
    pub valid_time: NaiveDateTime,
    pub member: Option<String>,
    pub parameter: i64,
    pub level_type: i64,
    pub level_value: f64,
    pub table_version: Option<i64>,
    pub center: Option<i64>,
    pub subcenter: Option<i64>,
    pub master_table_version: Option<i64>,
    pub local_table_version: Option<i64>,
    pub discipline: Option<i64>,
    pub category: Option<i64>,
    pub second_level_type: Option<i64>,
    pub second_level_value: Option<f64>,
    pub process_identity: Option<(i64, i64)>,
    pub time_semantics: Vec<i64>,
    pub values: ArrayD<f64>,
    pub latitude: Vec<f64>,
    pub longitude: Vec<f64>,
    pub grid_fingerprint: String,
}

/// The selector-visible identity of a record, without its bytes — the
/// lightweight record `_grib2_wanted_indices` builds from an inventory row.
#[derive(Debug, Clone)]
pub struct RecordIdentity {
    pub index: usize,
    pub member: Option<String>,
    pub parameter: i64,
    pub level_type: i64,
    pub level_value: f64,
    pub table_version: Option<i64>,
    pub center: Option<i64>,
    pub subcenter: Option<i64>,
    pub master_table_version: Option<i64>,
    pub local_table_version: Option<i64>,
    pub discipline: Option<i64>,
    pub category: Option<i64>,
    pub second_level_type: Option<i64>,
    pub second_level_value: Option<f64>,
    pub time_semantics: Vec<i64>,
}

impl GribRecord {
    pub fn identity(&self) -> RecordIdentity {
        RecordIdentity {
            index: self.index,
            member: self.member.clone(),
            parameter: self.parameter,
            level_type: self.level_type,
            level_value: self.level_value,
            table_version: self.table_version,
            center: self.center,
            subcenter: self.subcenter,
            master_table_version: self.master_table_version,
            local_table_version: self.local_table_version,
            discipline: self.discipline,
            category: self.category,
            second_level_type: self.second_level_type,
            second_level_value: self.second_level_value,
            time_semantics: self.time_semantics.clone(),
        }
    }
}

fn selector_int(selector: &crate::node::Node, key: &str) -> Option<i64> {
    selector.field(key).and_then(crate::node::Node::as_i64)
}

fn selector_float(selector: &crate::node::Node, key: &str) -> Option<f64> {
    selector.field(key).and_then(crate::node::Node::as_f64)
}

fn close(left: f64, right: f64) -> bool {
    // `math.isclose(..., abs_tol=1e-9)` with the default rel_tol of 1e-9.
    let difference = (left - right).abs();
    difference <= 1e-9 || difference <= 1e-9 * left.abs().max(right.abs())
}

/// `mapped_source._selector_matches_record`.
pub fn selector_matches(
    selector: &crate::node::Node,
    record: &RecordIdentity,
    source_format: &str,
) -> bool {
    if source_format == "grib1" {
        let Some(parameter) = selector_int(selector, "parameter") else {
            return false;
        };
        return record.parameter == parameter
            && selector_int(selector, "table_version")
                .is_none_or(|value| record.table_version == Some(value))
            && selector_int(selector, "center").is_none_or(|value| record.center == Some(value))
            && selector_int(selector, "level_type")
                .is_none_or(|value| record.level_type == value)
            && selector_float(selector, "level_value")
                .is_none_or(|value| close(record.level_value, value));
    }
    let (Some(discipline), Some(category), Some(parameter)) = (
        selector_int(selector, "discipline"),
        selector_int(selector, "category"),
        selector_int(selector, "parameter"),
    ) else {
        return false;
    };
    let second_surface = {
        let declared_type = selector_int(selector, "second_level_type");
        match declared_type {
            None => matches!(record.second_level_type, None | Some(255)),
            Some(kind) => {
                record.second_level_value.is_some()
                    && record.second_level_type == Some(kind)
                    && selector_float(selector, "second_level_value").is_some_and(|declared| {
                        close(record.second_level_value.unwrap_or(f64::NAN), declared)
                    })
            }
        }
    };
    record.discipline == Some(discipline)
        && record.category == Some(category)
        && record.parameter == parameter
        && selector_int(selector, "center").is_none_or(|value| record.center == Some(value))
        && selector_int(selector, "subcenter").is_none_or(|value| record.subcenter == Some(value))
        && selector_int(selector, "master_table_version")
            .is_none_or(|value| record.master_table_version == Some(value))
        && selector_int(selector, "local_table_version")
            .is_none_or(|value| record.local_table_version == Some(value))
        && selector_int(selector, "level_type").is_none_or(|value| record.level_type == value)
        && selector_float(selector, "level_value")
            .is_none_or(|value| close(record.level_value, value))
        && second_surface
        && selector
            .field("member")
            .and_then(crate::node::Node::as_str)
            .is_none_or(|value| record.member.as_deref() == Some(value))
        && selector_int(selector, "pdt")
            .is_none_or(|value| record.time_semantics.first() == Some(&value))
}

/// `mapped_source._declared_vertical_admits`.
pub fn declared_vertical_admits(
    declared_levels: &[f64],
    field: &FieldSpec<'_>,
    level_value: f64,
) -> Result<bool> {
    if !field.source_axes()?.iter().any(|axis| axis == "vertical") {
        return Ok(true);
    }
    if declared_levels.is_empty() {
        return Ok(true);
    }
    Ok(declared_levels
        .iter()
        .any(|level| (level_value - level).abs() <= 1e-6))
}

/// `mapped_source._embedded_valid_time` for GRIB2 (edition 2 factors).
pub fn embedded_valid_time(
    reference: NaiveDateTime,
    unit: u8,
    amount: i64,
    edition: u8,
) -> Result<NaiveDateTime> {
    let factor = if edition == 1 {
        match unit {
            0 => Duration::minutes(1),
            1 => Duration::hours(1),
            2 => Duration::days(1),
            10 => Duration::hours(3),
            11 => Duration::hours(6),
            12 => Duration::hours(12),
            254 => Duration::seconds(1),
            _ => {
                return Err(decode_failed(format!(
                    "unsupported GRIB1 forecast time unit {unit}"
                )))
            }
        }
    } else {
        match unit {
            0 => Duration::minutes(1),
            1 => Duration::hours(1),
            2 => Duration::days(1),
            10 => Duration::hours(3),
            11 => Duration::hours(6),
            12 => Duration::hours(12),
            13 => Duration::seconds(1),
            _ => {
                return Err(decode_failed(format!(
                    "unsupported GRIB2 forecast time unit {unit}"
                )))
            }
        }
    };
    Ok(reference + factor * amount as i32)
}

/// The grid fingerprint the Python engine hashes: a compact JSON object of
/// the TSV-RENDERED grid columns, keys sorted.
pub fn grid_fingerprint(message: &Grib2Message) -> String {
    let grid = &message.grid;
    let mut payload: BTreeMap<&str, String> = BTreeMap::new();
    payload.insert("gdt", grid.template.to_string());
    payload.insert("nx", grid.nx.to_string());
    payload.insert("ny", grid.ny.to_string());
    payload.insert("lat1", grid.lat1.to_string());
    payload.insert("lon1", grid.lon1.to_string());
    payload.insert("dx", grid.dx.to_string());
    payload.insert("dy", grid.dy.to_string());
    payload.insert("latin1", grid.latin1.to_string());
    payload.insert("latin2", grid.latin2.to_string());
    payload.insert("lov", grid.lov.to_string());
    payload.insert("scan_mode", format!("0x{:02x}", grid.scan_mode));
    payload.insert("shape_of_earth", grid.shape_of_earth.to_string());
    payload.insert(
        "resolution_flags",
        format!("0x{:02x}", grid.resolution_flags),
    );
    let mut text = String::from("{");
    for (position, (key, value)) in payload.iter().enumerate() {
        if position > 0 {
            text.push(',');
        }
        text.push_str(&serde_json::Value::String((*key).to_owned()).to_string());
        text.push(':');
        text.push_str(&serde_json::Value::String(value.clone()).to_string());
    }
    text.push('}');
    crate::digest::bytes_sha256(text.as_bytes())
}

/// `mapped_source._regular_latlon_frame`: one canonical ascending-latitude
/// frame from a regular GDT-0 record.
pub fn regular_latlon_frame(
    message: &Grib2Message,
    raw: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let grid = &message.grid;
    let scan = grid.scan_mode;
    if grid.template != 0 || !matches!(scan, 0x40 | 0x00) {
        return Err(grid_mismatch(format!(
            "generic GRIB2 frame export requires regular latitude/longitude \
             GDT 0 with scan mode 0x40 (rows south-to-north) or 0x00 (rows \
             north-to-south, normalized at decode), or a mapping.grid \
             declaration for a supported projected family; got GDT {} scan \
             mode 0x{scan:02x}",
            grid.template
        )));
    }
    let nx = grid.nx as usize;
    let ny = grid.ny as usize;
    let mut latitude: Vec<f64> = (0..ny)
        .map(|row| {
            if scan == 0x40 {
                grid.lat1 + row as f64 * grid.dy
            } else {
                grid.lat1 - row as f64 * grid.dy
            }
        })
        .collect();
    let wrapped: Vec<f64> = (0..nx)
        .map(|column| {
            let raw_longitude = grid.lon1 + column as f64 * grid.dx;
            (raw_longitude + 180.0).rem_euclid(360.0) - 180.0
        })
        .collect();
    // `np.argsort` is a STABLE sort for the default kind on ties; equal
    // longitudes keep their original order, which matters on a grid whose
    // first and last column alias after the wrap.
    let mut order: Vec<usize> = (0..nx).collect();
    order.sort_by(|left, right| {
        wrapped[*left]
            .total_cmp(&wrapped[*right])
            .then(left.cmp(right))
    });
    let longitude: Vec<f64> = order.iter().map(|index| wrapped[*index]).collect();
    let mut values = vec![0.0f64; nx * ny];
    for row in 0..ny {
        for (column, source_column) in order.iter().enumerate() {
            values[row * nx + column] = raw[row * nx + source_column];
        }
    }
    if scan == 0x00 {
        latitude.reverse();
        let mut flipped = vec![0.0f64; nx * ny];
        for row in 0..ny {
            let source = ny - 1 - row;
            flipped[row * nx..(row + 1) * nx].copy_from_slice(&values[source * nx..(source + 1) * nx]);
        }
        values = flipped;
    }
    Ok((latitude, longitude, values))
}

/// `grib2_inventory::validate_envelopes` — the envelope hygiene the
/// subprocess route ran before any parse.
pub fn validate_grib2_envelopes(bytes: &[u8], label: &str) -> Result<usize> {
    if bytes.is_empty() {
        return Err(decode_failed(format!("GRIB2 input {label} is empty")));
    }
    let mut offset = 0usize;
    let mut count = 0usize;
    while offset < bytes.len() {
        if bytes.len() - offset < 20 {
            return Err(decode_failed(format!(
                "truncated GRIB2 envelope {count} at byte {offset} of {label}: \
                 fewer than 20 bytes"
            )));
        }
        if &bytes[offset..offset + 4] != b"GRIB" {
            return Err(decode_failed(format!(
                "invalid GRIB2 envelope {count} at byte {offset} of {label}: \
                 missing GRIB marker"
            )));
        }
        let edition = bytes[offset + 7];
        if edition != 2 {
            return Err(decode_failed(format!(
                "envelope {count} at byte {offset} of {label} is GRIB edition \
                 {edition}, expected 2"
            )));
        }
        let declared = u64::from_be_bytes(
            bytes[offset + 8..offset + 16]
                .try_into()
                .expect("validated eight-byte slice"),
        ) as usize;
        if declared < 20 {
            return Err(decode_failed(format!(
                "invalid GRIB2 envelope {count} at byte {offset} of {label}: \
                 length {declared} is too short"
            )));
        }
        let Some(end) = offset.checked_add(declared).filter(|end| *end <= bytes.len()) else {
            return Err(decode_failed(format!(
                "truncated GRIB2 envelope {count} at byte {offset} of {label}: \
                 declared length {declared} runs past the file's {} bytes",
                bytes.len()
            )));
        };
        if &bytes[end - 4..end] != b"7777" {
            return Err(decode_failed(format!(
                "invalid GRIB2 envelope {count} at byte {offset} of {label}: \
                 missing 7777 terminator"
            )));
        }
        offset = end;
        count += 1;
    }
    Ok(count)
}

/// The identity of every message in one GRIB2 object, without unpacking.
pub fn grib2_identities(messages: &[Grib2Message]) -> Vec<RecordIdentity> {
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| RecordIdentity {
            index,
            member: message
                .product
                .perturbation_number
                .map(|value| value.to_string()),
            parameter: message.product.parameter_number as i64,
            level_type: message.product.level_type as i64,
            level_value: message.product.level_value,
            table_version: None,
            center: Some(message.identification.center_id as i64),
            subcenter: Some(message.identification.subcenter_id as i64),
            master_table_version: Some(message.identification.master_table_version as i64),
            local_table_version: Some(message.identification.local_table_version as i64),
            discipline: Some(message.discipline as i64),
            category: Some(message.product.parameter_category as i64),
            second_level_type: Some(message.product.second_level_type as i64),
            second_level_value: Some(message.product.second_level_value),
            time_semantics: vec![message.product.template as i64],
        })
        .collect()
}

/// `mapped_source._grib2_wanted_indices`.
pub fn wanted_indices(mapping: &Mapping, identities: &[RecordIdentity]) -> Result<Vec<usize>> {
    let declared_levels = mapping.declared_levels()?;
    let fields = mapping.fields()?;
    let mut wanted = Vec::new();
    for identity in identities {
        for field in &fields {
            let matched = field
                .selectors()
                .iter()
                .any(|selector| selector_matches(selector, identity, "grib2"));
            if matched && declared_vertical_admits(&declared_levels, field, identity.level_value)? {
                wanted.push(identity.index);
                break;
            }
        }
    }
    Ok(wanted)
}

/// `mapped_source._require_declared_grib2_grid`.
fn require_declared_grid(
    message: &Grib2Message,
    index: usize,
    declaration: &GridDeclaration,
    source: &str,
) -> Result<()> {
    let parameters = declaration
        .parameters
        .as_ref()
        .expect("lambert declarations carry parameters");
    let grid = &message.grid;
    let mut mismatched: Vec<String> = Vec::new();
    let mut compare_float = |key: &str, observed: f64, declared: f64| {
        if (observed - declared).abs() > 1e-6 {
            mismatched.push(format!("{key}: declared {declared}, observed {observed}"));
        }
    };
    compare_float("lat1", grid.lat1, parameters.lat1);
    compare_float("lon1", grid.lon1, parameters.lon1);
    compare_float("dx_m", grid.dx, parameters.dx_m);
    compare_float("dy_m", grid.dy, parameters.dy_m);
    compare_float("latin1", grid.latin1, parameters.latin1);
    compare_float("latin2", grid.latin2, parameters.latin2);
    compare_float("lov", grid.lov, parameters.lov);
    if grid.nx as i64 != parameters.nx {
        mismatched.push(format!(
            "nx: declared {}, observed {}",
            parameters.nx, grid.nx
        ));
    }
    if grid.ny as i64 != parameters.ny {
        mismatched.push(format!(
            "ny: declared {}, observed {}",
            parameters.ny, grid.ny
        ));
    }
    if grid.shape_of_earth as i64 != parameters.shape_of_earth {
        mismatched.push(format!(
            "shape_of_earth: declared {}, observed {}",
            parameters.shape_of_earth, grid.shape_of_earth
        ));
    }
    if !mismatched.is_empty() {
        mismatched.sort();
        return Err(grid_mismatch(format!(
            "GRIB2 field {index} in {source} is not on the declared \
             lambert_conformal grid; {}",
            mismatched.join("; ")
        )));
    }
    let grid_relative = grid.resolution_flags & GRIB2_GRID_RELATIVE_WIND_BIT != 0;
    if grid_relative != declaration.rotates_winds() {
        return Err(grid_mismatch(format!(
            "GRIB2 field {index} in {source} declares {}-relative vector \
             components (resolution_flags bit 0x08) while the mapping \
             declares wind_basis='{}'",
            if grid_relative { "grid" } else { "earth" },
            declaration.wind_basis
        )));
    }
    Ok(())
}

/// GRIB2 Data Representation Template 5.40: JPEG2000 packing.
const JPEG2000_TEMPLATE: u16 = 40;

/// `unpack_message`, with the one template whose decoder is not
/// re-entrant serialized against itself.
///
/// THE BREAKAGE THIS PREVENTS: `openjp2`'s HTJ2K path carries decoder
/// state in a process-global `static mut` (`only_cleanup_pass_is_decoded`
/// in its `ht_dec` module).  Two threads decoding Template 5.40
/// codestreams at the same time can therefore read each other's pass
/// state, and the corruption lands in float values that are still
/// finite, still the right shape, and wrong — the failure mode no
/// downstream check catches.  Every other template unpacks from `&self`
/// into a fresh `Vec` with no shared state, so only this one pays.
/// Sources that pack this way are real and staged (RAP/NAM AWIPS grids,
/// MSC GDPS), so it is guarded rather than refused.
fn unpack_shared(message: &Grib2Message) -> std::result::Result<Vec<f64>, grib_core::GribError> {
    if message.data_rep.template != JPEG2000_TEMPLATE {
        return unpack_message(message);
    }
    static CODEC: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A panic inside one message's decode must not turn every later
    // JPEG2000 message into a poisoned-lock refusal about threading:
    // the guard protects a `static mut`, and the next decode
    // initializes what it reads.
    let _guard = CODEC.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    unpack_message(message)
}

/// `mapped_source._grib2_records`: the selected records of one object,
/// decoded CONCURRENTLY.
///
/// Takes the object ALREADY PARSED, because its one caller has parsed
/// it to inventory it: the byte-taking spelling this replaces parsed
/// the same half-gigabyte file a second time and held both copies.
///
/// Message decode is pure per-message work — unpack a byte range into a
/// fresh array, read the grid octets, build the frame — so the messages
/// of one object are independent. They are decoded into pre-assigned
/// slots (`crate::threads`) and drained in `wanted` order, so the
/// returned vector, and the refusal if one is raised, are exactly what
/// the serial loop produced.
pub fn grib2_records(
    file: &Grib2File,
    source_label: &str,
    wanted: &[usize],
    declaration: &GridDeclaration,
) -> Result<Vec<GribRecord>> {
    if wanted.is_empty() {
        return Ok(Vec::new());
    }
    let slots: Vec<Result<GribRecord>> = crate::threads::install(|| {
        use rayon::prelude::*;
        wanted
            .par_iter()
            .map(|index| grib2_record(file, source_label, *index, declaration))
            .collect()
    });
    crate::threads::in_order(slots)
}

/// One selected GRIB2 message, decoded.
fn grib2_record(
    file: &Grib2File,
    source_label: &str,
    index: usize,
    declaration: &GridDeclaration,
) -> Result<GribRecord> {
    let message = file.messages.get(index).ok_or_else(|| {
        decode_failed(format!(
            "GRIB2 object {source_label} has no field {index}"
        ))
    })?;
    let nx = message.grid.nx as usize;
    let ny = message.grid.ny as usize;
    let raw = unpack_shared(message).map_err(|error| {
        decode_failed(format!(
            "GRIB2 field {index} in {source_label} failed to decode: {error}"
        ))
    })?;
    if raw.len() != nx * ny {
        return Err(decode_failed(format!(
            "GRIB2 field {index} decoded count {} differs from grid {nx}x{ny}",
            raw.len()
        )));
    }
    let (latitude, longitude, values) = if declaration.is_lambert() {
        if message.grid.template != 30 || message.grid.scan_mode != 0x40 {
            return Err(grid_mismatch(format!(
                "the mapping declares a lambert_conformal source grid, \
                 which requires GRIB2 GDT 30 with scan mode 0x40; field \
                 {index} in {source_label} carries GDT {} scan 0x{:02x}",
                message.grid.template, message.grid.scan_mode
            )));
        }
        require_declared_grid(message, index, declaration, source_label)?;
        let parameters = declaration
            .parameters
            .as_ref()
            .expect("lambert declarations carry parameters");
        let (y, x) = parameters.projected_axes();
        (y, x, raw)
    } else {
        regular_latlon_frame(message, &raw)?
    };
    let array = ArrayD::from_shape_vec(IxDyn(&[latitude.len(), longitude.len()]), values)
        .map_err(|error| {
            decode_failed(format!(
                "GRIB2 field {index} in {source_label} does not fill its grid: {error}"
            ))
        })?;
    let valid_time = embedded_valid_time(
        message.reference_time,
        message.product.time_range_unit,
        message.product.forecast_time as i64,
        2,
    )?;
    Ok(GribRecord {
        source: source_label.to_owned(),
        index,
        reference_time: message.reference_time,
        valid_time,
        member: message
            .product
            .perturbation_number
            .map(|value| value.to_string()),
        parameter: message.product.parameter_number as i64,
        level_type: message.product.level_type as i64,
        level_value: message.product.level_value,
        table_version: None,
        center: Some(message.identification.center_id as i64),
        subcenter: Some(message.identification.subcenter_id as i64),
        master_table_version: Some(message.identification.master_table_version as i64),
        local_table_version: Some(message.identification.local_table_version as i64),
        discipline: Some(message.discipline as i64),
        category: Some(message.product.parameter_category as i64),
        second_level_type: Some(message.product.second_level_type as i64),
        second_level_value: Some(message.product.second_level_value),
        process_identity: Some((
            message.product.generating_process as i64,
            message.product.forecast_generating_process_id as i64,
        )),
        time_semantics: vec![message.product.template as i64],
        values: array,
        latitude,
        longitude,
        grid_fingerprint: grid_fingerprint(message),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    #[test]
    fn a_grib2_selector_without_a_second_surface_admits_only_the_missing_code() {
        let selector = Node::parse(br#"{"discipline": 0, "category": 0, "parameter": 0}"#).unwrap();
        let mut identity = RecordIdentity {
            index: 0,
            member: None,
            parameter: 0,
            level_type: 100,
            level_value: 85000.0,
            table_version: None,
            center: Some(7),
            subcenter: Some(0),
            master_table_version: Some(2),
            local_table_version: Some(1),
            discipline: Some(0),
            category: Some(0),
            second_level_type: Some(255),
            second_level_value: Some(0.0),
            time_semantics: vec![0],
        };
        assert!(selector_matches(&selector, &identity, "grib2"));
        identity.second_level_type = Some(106);
        assert!(!selector_matches(&selector, &identity, "grib2"));
    }

    #[test]
    fn a_selector_pinning_producer_identity_refuses_the_other_producer() {
        let selector = Node::parse(
            br#"{"discipline": 0, "category": 0, "parameter": 0, "center": 7, "subcenter": 4}"#,
        )
        .unwrap();
        let mut identity = RecordIdentity {
            index: 0,
            member: None,
            parameter: 0,
            level_type: 100,
            level_value: 85000.0,
            table_version: None,
            center: Some(7),
            subcenter: Some(4),
            master_table_version: Some(2),
            local_table_version: Some(1),
            discipline: Some(0),
            category: Some(0),
            second_level_type: Some(255),
            second_level_value: Some(0.0),
            time_semantics: vec![0],
        };
        assert!(selector_matches(&selector, &identity, "grib2"));
        identity.subcenter = Some(0);
        assert!(!selector_matches(&selector, &identity, "grib2"));
    }

    #[test]
    fn envelope_validation_names_a_truncated_object() {
        let refusal = validate_grib2_envelopes(b"GRIB\x00\x00\x00\x02", "sample").unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::DECODE_FAILED);
        assert!(refusal.message.contains("fewer than 20 bytes"));
    }

    #[test]
    fn envelope_validation_refuses_a_grib1_object_by_edition() {
        let mut bytes = vec![0u8; 24];
        bytes[..4].copy_from_slice(b"GRIB");
        bytes[7] = 1;
        let refusal = validate_grib2_envelopes(&bytes, "sample").unwrap_err();
        assert!(refusal.message.contains("GRIB edition 1"));
    }

    #[test]
    fn embedded_valid_time_uses_the_edition_two_second_code() {
        let reference = NaiveDateTime::parse_from_str("2026-08-17 00:00:00", "%Y-%m-%d %H:%M:%S")
            .unwrap();
        let valid = embedded_valid_time(reference, 13, 90, 2).unwrap();
        assert_eq!((valid - reference).num_seconds(), 90);
        // The same code means DAYS in no edition; 254 is edition 1's second.
        assert!(embedded_valid_time(reference, 254, 1, 2).is_err());
        assert!(embedded_valid_time(reference, 254, 1, 1).is_ok());
    }
}
