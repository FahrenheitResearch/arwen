//! GRIB1 records, decoded in process.
//!
//! Port of `mapped_source._grib1_records` together with the two pieces it
//! used to reach through a pipe: `gpuwm.ingest.grib.inspect_grib1_envelopes`
//! (the setup-time envelope walk) and `tools/grib1_bridge`'s emission
//! contract (`metadata.json` + `values.f64`).  The subprocess, the
//! temporary directory and the file handoff are gone; the numbers are not
//! meant to move, because both sides read the same `grib-core::grib1`
//! parser and the same `unpack_bds`.
//!
//! Two behaviours here look like defects to a reader who has not seen the
//! reference, and both are deliberate:
//!
//!   * **Messages whose grid differs from the file's grid are skipped
//!     silently.**  CDO writes one-point control records between parameter
//!     blocks in concatenated output; they are valid GRIB and can never
//!     satisfy a meteorological mapping field.  `_grib1_records` drops them
//!     with a bare `continue`, and the ERA5 1974 reference object carries
//!     exactly three.
//!   * **The record index counts the skipped messages.**  Provenance
//!     strings are `<path>:<index>` and the index is the enumeration index
//!     over EVERY message in the object.  An index over kept messages only
//!     would decode to numerically identical arrays with wrong provenance,
//!     which is the worst shape a defect can take.
//!
//! The file's ONE grid is the largest message's, exactly as the bridge
//! chose it, so the control records cannot redefine the axes on their way
//! past.

use chrono::{NaiveDate, NaiveDateTime};
use grib_core::grib1::{Grib1File, Grib1Message, GridType};
use ndarray::{ArrayD, IxDyn};

use crate::grib::{embedded_valid_time, GribRecord};
use crate::refusal::{decode_failed, python_bytes_repr, Result};

/// Scanning-mode bit 3: adjacent points in the j direction are consecutive.
const SCAN_J_CONSECUTIVE: u8 = 0x20;

/// The tolerance `grib1_bridge` separates a grid's axes with.
const AXIS_SEPARABILITY_TOLERANCE_DEG: f64 = 1.0e-10;

/// `gpuwm.ingest.grib.inspect_grib1_envelopes`, on bytes already in hand.
///
/// THE BREAKAGE THIS PREVENTS: a concatenated GRIB1 object that loses its
/// alignment part way through still parses as a useful PREFIX — the
/// scanning parser in `grib-core` skips past a bad `GRIB` marker and keeps
/// going — so a truncated download would decode to a frame that is short
/// some levels and says nothing about it.  This walk is deliberately
/// independent of the decoder: it demands exact end-to-end envelope
/// coverage of the object and gives every failure a message/byte address.
///
/// The sentences are the Python engine's own, because the two engines'
/// refusals are compared byte for byte.
pub fn validate_grib1_envelopes(bytes: &[u8], label: &str) -> Result<usize> {
    let size = bytes.len();
    if size == 0 {
        return Err(decode_failed(format!("GRIB1 file {label} is empty")));
    }
    let mut offset = 0usize;
    let mut index = 0usize;
    while offset < size {
        let available = size - offset;
        if available < 8 {
            return Err(decode_failed(format!(
                "truncated GRIB1 file {label}: message {index} at byte \
                 {offset} has only {available} of 8 indicator bytes"
            )));
        }
        let indicator = &bytes[offset..offset + 8];
        if &indicator[..4] != b"GRIB" {
            return Err(decode_failed(format!(
                "invalid GRIB1 file {label}: message {index} at byte \
                 {offset} has marker {}, expected b'GRIB'",
                python_bytes_repr(&indicator[..4])
            )));
        }
        if indicator[7] != 1 {
            return Err(decode_failed(format!(
                "unsupported GRIB edition {} in {label}, message {index} at \
                 byte {offset}; native input must be GRIB1",
                indicator[7]
            )));
        }
        let length = ((indicator[4] as usize) << 16)
            | ((indicator[5] as usize) << 8)
            | (indicator[6] as usize);
        if length < 12 {
            return Err(decode_failed(format!(
                "invalid GRIB1 file {label}: message {index} at byte \
                 {offset} declares length {length}, minimum is 12"
            )));
        }
        let end = offset + length;
        if end > size {
            return Err(decode_failed(format!(
                "truncated GRIB1 file {label}: message {index} at byte \
                 {offset} declares end byte {end}, file has {size} bytes"
            )));
        }
        if &bytes[end - 4..end] != b"7777" {
            return Err(decode_failed(format!(
                "invalid GRIB1 file {label}: message {index} at byte \
                 {offset} lacks the 7777 terminator at byte {}",
                end - 4
            )));
        }
        index += 1;
        offset = end;
    }
    Ok(index)
}

/// The one grid an object's records share, derived the bridge's way.
struct FileGrid {
    nx: usize,
    ny: usize,
    latitude: Vec<f64>,
    longitude: Vec<f64>,
    fingerprint: String,
}

/// The bridge wrote every coordinate into `metadata.json` as `{:.15}` and
/// the Python engine parsed THAT text back into the axes it hashes.
///
/// Reproducing the round trip is what makes the grid fingerprint and the
/// coordinate arrays identical between the engines on EVERY GRIB1 source,
/// not just on the ones whose axes happen to be exactly representable
/// (a regular grid in whole millidegrees, like the ERA5 reference object,
/// round-trips unchanged and this is a no-op there).  It is a rendering
/// of the handoff format, so it is applied where the handoff applied it:
/// to the axes that leave this module, after the separability check has
/// read the parser's own numbers.
fn bridge_rendered(value: f64) -> f64 {
    format!("{value:.15}").parse::<f64>().unwrap_or(value)
}

/// `grib1_bridge::grid_shape_and_scan`, per message.
fn latlon_shape_and_scan(message: &Grib1Message, subject: &str) -> Result<(usize, usize, u8)> {
    let gds = message.gds.as_ref().ok_or_else(|| {
        decode_failed(format!("{subject} has no grid description section"))
    })?;
    match &gds.grid_type {
        GridType::LatLon {
            ni,
            nj,
            scanning_mode,
            ..
        } => Ok((*ni as usize, *nj as usize, *scanning_mode)),
        _ => Err(decode_failed(format!(
            "{subject} is not on a regular latitude/longitude GRIB1 grid; \
             the mapped GRIB1 path decodes grid definition type 0 only"
        ))),
    }
}

/// `grib1_bridge::run`'s grid preamble.
fn file_grid(file: &Grib1File, source: &str) -> Result<FileGrid> {
    // The LARGEST message defines the object's grid.  A concatenated CDO
    // object carries one-point control records; letting the first message
    // define the axes would let one of those redefine the whole file.
    let primary = file
        .messages
        .iter()
        .max_by_key(|message| message.num_data_points())
        .ok_or_else(|| {
            decode_failed(format!("GRIB1 input {source} contains no messages"))
        })?;
    let subject = format!("the grid-defining GRIB1 message in {source}");
    if primary.indicator.edition != 1 {
        return Err(decode_failed(format!(
            "{subject} is GRIB edition {}, expected 1",
            primary.indicator.edition
        )));
    }
    let (nx, ny, scan_mode) = latlon_shape_and_scan(primary, &subject)?;
    if scan_mode & SCAN_J_CONSECUTIVE != 0 {
        return Err(decode_failed(format!(
            "GRIB1 j-consecutive scanning is unsupported; {subject} carries \
             scan mode 0x{scan_mode:02x}"
        )));
    }
    let coordinates = primary.latlons().map_err(|error| {
        decode_failed(format!(
            "GRIB1 grid coordinates for {source} could not be generated: {error}"
        ))
    })?;
    if coordinates.len() != nx * ny {
        return Err(decode_failed(format!(
            "GRIB1 coordinate count {} in {source} does not match its \
             {ny}x{nx} grid",
            coordinates.len()
        )));
    }
    let latitude: Vec<f64> = (0..ny).map(|row| coordinates[row * nx].lat).collect();
    let longitude: Vec<f64> = (0..nx).map(|column| coordinates[column].lon).collect();
    for row in 0..ny {
        for column in 0..nx {
            let point = coordinates[row * nx + column];
            if (point.lat - latitude[row]).abs() > AXIS_SEPARABILITY_TOLERANCE_DEG
                || (point.lon - longitude[column]).abs() > AXIS_SEPARABILITY_TOLERANCE_DEG
            {
                return Err(decode_failed(format!(
                    "the GRIB1 grid in {source} is not separable into \
                     latitude and longitude axes"
                )));
            }
        }
    }
    let latitude: Vec<f64> = latitude.into_iter().map(bridge_rendered).collect();
    let longitude: Vec<f64> = longitude.into_iter().map(bridge_rendered).collect();
    let fingerprint = grid_fingerprint(&latitude, &longitude);
    Ok(FileGrid {
        nx,
        ny,
        latitude,
        longitude,
        fingerprint,
    })
}

/// `_grib1_records`' fingerprint: sha256 over the two axes' float64 bytes.
///
/// NOT the GRIB2 fingerprint, which hashes a JSON object of rendered grid
/// OCTETS.  The two formats reach `_assemble_grib` with different
/// fingerprint recipes and the recipes are not interchangeable: a mapping
/// is one format, so the values are only ever compared with their own kind.
fn grid_fingerprint(latitude: &[f64], longitude: &[f64]) -> String {
    let mut bytes = Vec::with_capacity((latitude.len() + longitude.len()) * 8);
    for value in latitude.iter().chain(longitude.iter()) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    crate::digest::bytes_sha256(&bytes)
}

/// One object's usable records, and how many messages it carried.
///
/// Every message is decoded, including the control records that are then
/// dropped: the bridge unpacked the whole object before the Python engine
/// filtered it, so a control record that does not decode refuses the
/// object rather than being quietly discarded with the rest.
pub fn grib1_records(payload: &[u8], source: &str) -> Result<(usize, Vec<GribRecord>)> {
    let messages = validate_grib1_envelopes(payload, source)?;
    let file = Grib1File::from_bytes(payload).map_err(|error| {
        decode_failed(format!("GRIB1 parse failed for {source}: {error}"))
    })?;
    if file.messages.is_empty() {
        return Err(decode_failed(format!(
            "GRIB1 input {source} contains no parsed fields"
        )));
    }
    let grid = file_grid(&file, source)?;
    // Message decode is pure per-message work over disjoint byte ranges,
    // so the messages of one object are independent.  Pre-assigned slots
    // drained in document order (`crate::threads`), so the record vector
    // AND the refusal are the serial loop's.
    let slots: Vec<Result<Option<GribRecord>>> = crate::threads::install(|| {
        use rayon::prelude::*;
        file.messages
            .par_iter()
            .enumerate()
            .map(|(index, message)| grib1_record(message, index, source, &grid))
            .collect()
    });
    let records = crate::threads::in_order(slots)?
        .into_iter()
        .flatten()
        .collect();
    Ok((messages, records))
}

/// One message: `None` when it is a control record on another grid.
fn grib1_record(
    message: &Grib1Message,
    index: usize,
    source: &str,
    grid: &FileGrid,
) -> Result<Option<GribRecord>> {
    let subject = format!("GRIB1 message {index} in {source}");
    if message.indicator.edition != 1 {
        return Err(decode_failed(format!(
            "{subject} is GRIB edition {}, expected 1",
            message.indicator.edition
        )));
    }
    let (nx, ny, scan_mode) = latlon_shape_and_scan(message, &subject)?;
    if scan_mode & SCAN_J_CONSECUTIVE != 0 {
        return Err(decode_failed(format!(
            "GRIB1 j-consecutive scanning is unsupported; {subject} carries \
             scan mode 0x{scan_mode:02x}"
        )));
    }
    let values = message.values().map_err(|error| {
        decode_failed(format!("{subject} failed to decode: {error}"))
    })?;
    if values.len() != nx * ny {
        return Err(decode_failed(format!(
            "{subject} decoded {} values, which does not fill its {ny}x{nx} grid",
            values.len()
        )));
    }
    if (ny, nx) != (grid.ny, grid.nx) {
        // A control record.  See the module header: silent by design.
        return Ok(None);
    }
    let pds = &message.pds;
    let reference = NaiveDate::from_ymd_opt(pds.year(), u32::from(pds.month), u32::from(pds.day))
        .and_then(|date| date.and_hms_opt(u32::from(pds.hour), u32::from(pds.minute), 0))
        .ok_or_else(|| {
            decode_failed(format!(
                "{subject} carries the reference time {}-{}-{} {}:{}, which is \
                 not a date",
                pds.year(),
                pds.month,
                pds.day,
                pds.hour,
                pds.minute
            ))
        })?;
    let valid_time: NaiveDateTime =
        embedded_valid_time(reference, pds.time_unit, i64::from(pds.p1), 1)?;
    let array = ArrayD::from_shape_vec(IxDyn(&[ny, nx]), values).map_err(|error| {
        decode_failed(format!("{subject} does not fill its grid: {error}"))
    })?;
    Ok(Some(GribRecord {
        source: source.to_owned(),
        index,
        reference_time: reference,
        valid_time,
        // GRIB1 carries no perturbation number in the sections this
        // decoder reads, so every record is the deterministic member.
        member: None,
        parameter: i64::from(pds.parameter),
        level_type: i64::from(pds.level_type),
        level_value: f64::from(pds.level_value),
        table_version: Some(i64::from(pds.table_version)),
        center: Some(i64::from(pds.center_id)),
        // The GRIB2-only identity octets.  `_GribRecord` leaves them
        // `None` for edition 1 and `_selector_matches_record` never
        // consults them on the GRIB1 branch.
        subcenter: None,
        master_table_version: None,
        local_table_version: None,
        discipline: None,
        category: None,
        second_level_type: None,
        second_level_value: None,
        process_identity: None,
        // GRIB1 time semantics: (time range indicator, P1, P2).
        // `assemble::assemble_grib` binds the instantaneous case only.
        time_semantics: vec![
            i64::from(pds.time_range_indicator),
            i64::from(pds.p1),
            i64::from(pds.p2),
        ],
        // GRIB1 keeps its vertical-coordinate parameters in the GDS,
        // which this decoder does not read; a GRIB1 hybrid source
        // declares inline vertical.hybrid_a/hybrid_b literals instead.
        coordinate_values: Vec::new(),
        values: array,
        latitude: grid.latitude.clone(),
        longitude: grid.longitude.clone(),
        grid_fingerprint: grid.fingerprint.clone(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One synthetic GRIB1 message, built to the octet grammar
    /// `grib-core::grib1::parser` reads.
    ///
    /// Hand-built rather than sampled so these tests never skip: the real
    /// ERA5 reference object is a 29 MB artifact outside the repository,
    /// and a suite that skips without it teaches the next reader to ignore
    /// a silent GRIB1 path.
    #[derive(Clone)]
    struct MessageSpec {
        ni: u16,
        nj: u16,
        la1_millideg: i32,
        lo1_millideg: i32,
        di_millideg: u16,
        dj_millideg: u16,
        scan: u8,
        table_version: u8,
        center: u8,
        parameter: u8,
        level_type: u8,
        level_value: u16,
        century: u8,
        year_of_century: u8,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        time_unit: u8,
        p1: u8,
        p2: u8,
        time_range_indicator: u8,
        /// One octet per grid point: 8-bit simple packing over a zero
        /// reference and zero scaling, so the decoded value IS the octet.
        packed: Vec<u8>,
    }

    fn spec(ni: u16, nj: u16) -> MessageSpec {
        let points = usize::from(ni) * usize::from(nj);
        MessageSpec {
            ni,
            nj,
            la1_millideg: 25_000,
            lo1_millideg: 250_000,
            di_millideg: 250,
            dj_millideg: 250,
            // 0x40: i-consecutive, rows south to north -- the ERA5 object's.
            scan: 0x40,
            table_version: 128,
            center: 98,
            parameter: 130,
            level_type: 100,
            level_value: 500,
            century: 20,
            year_of_century: 74,
            month: 4,
            day: 3,
            hour: 12,
            minute: 0,
            time_unit: 1,
            p1: 0,
            p2: 0,
            time_range_indicator: 0,
            packed: (0..points).map(|value| value as u8).collect(),
        }
    }

    fn push_u24(bytes: &mut Vec<u8>, value: u32) {
        bytes.push((value >> 16) as u8);
        bytes.push((value >> 8) as u8);
        bytes.push(value as u8);
    }

    fn push_signed_24(bytes: &mut Vec<u8>, value: i32) {
        let raw = if value < 0 {
            0x80_0000 | (value.unsigned_abs() & 0x7F_FFFF)
        } else {
            value as u32 & 0x7F_FFFF
        };
        push_u24(bytes, raw);
    }

    fn encode(spec: &MessageSpec) -> Vec<u8> {
        let mut message = Vec::new();
        let total = 8 + 28 + 32 + (11 + spec.packed.len()) + 4;
        message.extend_from_slice(b"GRIB");
        push_u24(&mut message, total as u32);
        message.push(1);

        // Section 1: Product Definition Section, the 28-octet minimum.
        push_u24(&mut message, 28);
        message.push(spec.table_version);
        message.push(spec.center);
        message.push(0); // generating process
        message.push(255); // grid identification: defined in the GDS
        message.push(0x80); // GDS present, no bitmap
        message.push(spec.parameter);
        message.push(spec.level_type);
        message.push((spec.level_value >> 8) as u8);
        message.push(spec.level_value as u8);
        message.push(spec.year_of_century);
        message.push(spec.month);
        message.push(spec.day);
        message.push(spec.hour);
        message.push(spec.minute);
        message.push(spec.time_unit);
        message.push(spec.p1);
        message.push(spec.p2);
        message.push(spec.time_range_indicator);
        message.push(0); // number in average, high
        message.push(0); // number in average, low
        message.push(0); // number missing
        message.push(spec.century);
        message.push(0); // sub-centre
        message.push(0); // decimal scale, high
        message.push(0); // decimal scale, low

        // Section 2: Grid Description Section, latitude/longitude type 0.
        push_u24(&mut message, 32);
        message.push(0); // NV
        message.push(255); // PV/PL location
        message.push(0); // data representation type: regular lat/lon
        message.push((spec.ni >> 8) as u8);
        message.push(spec.ni as u8);
        message.push((spec.nj >> 8) as u8);
        message.push(spec.nj as u8);
        push_signed_24(&mut message, spec.la1_millideg);
        push_signed_24(&mut message, spec.lo1_millideg);
        message.push(0x80); // direction increments given
        let last_row = i32::from(spec.nj.saturating_sub(1));
        let last_column = i32::from(spec.ni.saturating_sub(1));
        let dj = i32::from(spec.dj_millideg);
        let la2 = if spec.scan & 0x40 != 0 {
            spec.la1_millideg + last_row * dj
        } else {
            spec.la1_millideg - last_row * dj
        };
        push_signed_24(&mut message, la2);
        push_signed_24(
            &mut message,
            spec.lo1_millideg + last_column * i32::from(spec.di_millideg),
        );
        message.push((spec.di_millideg >> 8) as u8);
        message.push(spec.di_millideg as u8);
        message.push((spec.dj_millideg >> 8) as u8);
        message.push(spec.dj_millideg as u8);
        message.push(spec.scan);
        message.extend_from_slice(&[0u8; 4]); // reserved, to the 32-octet length

        // Section 4: Binary Data Section, simple packing, 8 bits per datum.
        push_u24(&mut message, (11 + spec.packed.len()) as u32);
        message.push(0); // simple packing, grid point, floating point, no padding
        message.push(0); // binary scale, high
        message.push(0); // binary scale, low
        message.extend_from_slice(&[0u8; 4]); // reference value: IBM zero
        message.push(8); // bits per datum
        message.extend_from_slice(&spec.packed);

        message.extend_from_slice(b"7777");
        assert_eq!(message.len(), total, "the encoder must fill its envelope");
        message
    }

    fn object(specs: &[MessageSpec]) -> Vec<u8> {
        specs.iter().flat_map(|spec| encode(spec)).collect()
    }

    /// The ERA5 reference object's shape: two meteorological messages with
    /// a one-point CDO control record wedged between them.
    fn object_with_a_control_record() -> Vec<u8> {
        let mut control = spec(1, 1);
        control.table_version = 255;
        control.parameter = 2;
        control.level_type = 1;
        control.level_value = 0;
        let mut second = spec(3, 2);
        second.parameter = 131;
        object(&[spec(3, 2), control, second])
    }

    #[test]
    fn an_empty_object_is_refused_before_the_parser_sees_it() {
        let refusal = validate_grib1_envelopes(b"", "sample.grb").unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::DECODE_FAILED);
        assert_eq!(refusal.message, "GRIB1 file sample.grb is empty");
    }

    #[test]
    fn a_lost_alignment_names_the_marker_bytes_the_way_python_prints_them() {
        let mut bytes = encode(&spec(3, 2));
        let start = bytes.len();
        bytes.extend_from_slice(b"\x00\x01ab and then some");
        let refusal = validate_grib1_envelopes(&bytes, "sample.grb").unwrap_err();
        assert_eq!(
            refusal.message,
            format!(
                "invalid GRIB1 file sample.grb: message 1 at byte {start} has \
                 marker b'\\x00\\x01ab', expected b'GRIB'"
            )
        );
    }

    #[test]
    fn a_truncated_object_names_the_byte_its_last_message_claims() {
        let bytes = encode(&spec(3, 2));
        let declared = bytes.len();
        let mut bytes = bytes;
        bytes.truncate(declared - 3);
        let refusal = validate_grib1_envelopes(&bytes, "sample.grb").unwrap_err();
        assert_eq!(
            refusal.message,
            format!(
                "truncated GRIB1 file sample.grb: message 0 at byte 0 declares \
                 end byte {declared}, file has {} bytes",
                declared - 3
            )
        );
    }

    #[test]
    fn a_grib2_object_is_refused_by_edition_rather_than_half_decoded() {
        let mut bytes = encode(&spec(3, 2));
        bytes[7] = 2;
        let refusal = validate_grib1_envelopes(&bytes, "sample.grb").unwrap_err();
        assert_eq!(
            refusal.message,
            "unsupported GRIB edition 2 in sample.grb, message 0 at byte 0; \
             native input must be GRIB1"
        );
    }

    #[test]
    fn a_declared_length_shorter_than_an_envelope_is_refused() {
        let mut bytes = encode(&spec(3, 2));
        bytes[4] = 0;
        bytes[5] = 0;
        bytes[6] = 8;
        let refusal = validate_grib1_envelopes(&bytes, "sample.grb").unwrap_err();
        assert!(
            refusal.message.contains("declares length 8, minimum is 12"),
            "unexpected sentence: {}",
            refusal.message
        );
    }

    #[test]
    fn a_message_without_its_terminator_is_refused_by_byte_address() {
        let mut bytes = encode(&spec(3, 2));
        let end = bytes.len();
        bytes[end - 1] = b'8';
        let refusal = validate_grib1_envelopes(&bytes, "sample.grb").unwrap_err();
        assert_eq!(
            refusal.message,
            format!(
                "invalid GRIB1 file sample.grb: message 0 at byte 0 lacks the \
                 7777 terminator at byte {}",
                end - 4
            )
        );
    }

    #[test]
    fn a_control_record_is_skipped_and_still_advances_the_record_index() {
        // THE BREAKAGE THIS PREVENTS: an index over KEPT messages decodes
        // to numerically identical arrays whose `<path>:<index>`
        // provenance points at the wrong message.
        let bytes = object_with_a_control_record();
        let (messages, records) = grib1_records(&bytes, "sample.grb").unwrap();
        assert_eq!(messages, 3);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].index, 0);
        assert_eq!(records[1].index, 2);
        assert_eq!(records[0].parameter, 130);
        assert_eq!(records[1].parameter, 131);
    }

    #[test]
    fn every_record_carries_the_objects_one_grid_and_one_fingerprint() {
        let bytes = object_with_a_control_record();
        let (_messages, records) = grib1_records(&bytes, "sample.grb").unwrap();
        let latitude = vec![25.0, 25.25];
        let longitude = vec![250.0, 250.25, 250.5];
        for record in &records {
            assert_eq!(record.latitude, latitude);
            assert_eq!(record.longitude, longitude);
            assert_eq!(
                record.grid_fingerprint,
                grid_fingerprint(&latitude, &longitude)
            );
        }
        // The recipe itself, pinned: `_grib1_records` hashes the two axes'
        // float64 bytes back to back, NOT the GRIB2 octet document.
        let mut expected = Vec::new();
        for value in latitude.iter().chain(longitude.iter()) {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(
            records[0].grid_fingerprint,
            crate::digest::bytes_sha256(&expected)
        );
    }

    #[test]
    fn values_land_row_major_on_the_objects_grid() {
        let bytes = object(&[spec(3, 2)]);
        let (_messages, records) = grib1_records(&bytes, "sample.grb").unwrap();
        assert_eq!(records[0].values.shape(), &[2, 3]);
        assert_eq!(
            records[0].values.iter().copied().collect::<Vec<f64>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn a_record_mirrors_the_python_engines_edition_one_field_set() {
        let bytes = object(&[spec(3, 2)]);
        let (_messages, records) = grib1_records(&bytes, "sample.grb").unwrap();
        let record = &records[0];
        assert_eq!(record.source, "sample.grb");
        assert_eq!(record.table_version, Some(128));
        assert_eq!(record.center, Some(98));
        assert_eq!(record.level_type, 100);
        assert_eq!(record.level_value, 500.0);
        assert_eq!(record.time_semantics, vec![0, 0, 0]);
        assert_eq!(record.member, None);
        // Every GRIB2-only identity octet stays absent, or a GRIB2
        // selector would start matching edition-1 records.
        assert_eq!(record.subcenter, None);
        assert_eq!(record.master_table_version, None);
        assert_eq!(record.local_table_version, None);
        assert_eq!(record.discipline, None);
        assert_eq!(record.category, None);
        assert_eq!(record.second_level_type, None);
        assert_eq!(record.second_level_value, None);
        assert_eq!(record.process_identity, None);
        assert_eq!(
            record.reference_time.to_string(),
            "1974-04-03 12:00:00",
            "century 20 + year-of-century 74 is 1974, not 2074"
        );
        assert_eq!(record.valid_time, record.reference_time);
    }

    #[test]
    fn a_forecast_offset_moves_the_valid_time_by_the_edition_one_unit() {
        let mut forecast = spec(3, 2);
        forecast.time_unit = 11; // six hours, in edition 1's table
        forecast.p1 = 2;
        let bytes = object(&[forecast]);
        let (_messages, records) = grib1_records(&bytes, "sample.grb").unwrap();
        assert_eq!(
            (records[0].valid_time - records[0].reference_time).num_hours(),
            12
        );
    }

    #[test]
    fn a_forecast_time_unit_edition_one_does_not_define_is_refused_by_number() {
        let mut unknown = spec(3, 2);
        // 13 is edition TWO's second.  Edition 1 spells a second 254, and
        // reading 13 as anything here would silently move a valid time.
        unknown.time_unit = 13;
        let bytes = object(&[unknown]);
        let refusal = grib1_records(&bytes, "sample.grb").unwrap_err();
        assert_eq!(
            refusal.message,
            "unsupported GRIB1 forecast time unit 13"
        );
    }

    #[test]
    fn a_j_consecutive_grid_is_refused_rather_than_transposed() {
        // THE BREAKAGE THIS PREVENTS: the axis recovery below reads
        // `coordinates[row * nx + column]`, which is the i-consecutive
        // layout.  A j-consecutive object decoded through it would produce
        // a full, finite, TRANSPOSED field -- the failure no later check
        // catches.
        let mut transposed = spec(3, 2);
        transposed.scan = 0x60;
        let bytes = object(&[transposed]);
        let refusal = grib1_records(&bytes, "sample.grb").unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::DECODE_FAILED);
        assert!(
            refusal
                .message
                .starts_with("GRIB1 j-consecutive scanning is unsupported"),
            "unexpected sentence: {}",
            refusal.message
        );
    }

    #[test]
    fn a_control_record_that_does_not_decode_refuses_the_whole_object() {
        // The bridge unpacked EVERY message before the Python engine
        // filtered any, so a corrupt control record refused the object.
        // Dropping it silently would let a corrupt file decode.
        let mut control = spec(1, 1);
        control.packed = Vec::new();
        let bytes = object(&[spec(3, 2), control]);
        let refusal = grib1_records(&bytes, "sample.grb").unwrap_err();
        assert!(
            refusal.message.starts_with("GRIB1 message 1 in sample.grb failed to decode"),
            "unexpected sentence: {}",
            refusal.message
        );
    }

    #[test]
    fn the_coordinate_rendering_reproduces_the_reference_including_where_it_loses_a_digit() {
        // Every coordinate of magnitude one or more keeps seventeen
        // significant digits under `{:.15}` and comes back bit for bit,
        // which is every latitude and longitude a degree grid carries --
        // so on real sources this is a no-op and the axes are the
        // parser's own numbers.
        for value in [25.0f64, 250.25, -179.75, 0.1, 89.999, 1.0e-9] {
            assert_eq!(bridge_rendered(value), value, "{value} did not round trip");
        }
        // Below one degree the rendering is fifteen significant digits and
        // an f64 can need seventeen.  The Python engine's number IS the
        // rendered one, so this side reproduces it rather than keeping the
        // more precise value: a coordinate axis that differed in its last
        // bit would move the grid fingerprint and fail the whole frame
        // against the reference.
        let third = 1.0f64 / 3.0;
        assert_ne!(bridge_rendered(third), third);
        assert_eq!(bridge_rendered(third), 0.333333333333333f64);
    }
}
