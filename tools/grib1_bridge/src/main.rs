//! Minimal GRIB1-to-raw/JSON bridge for gpuwm ERA5 ingest.

use grib_core::grib1::{Grib1File, Grib1Message, GridType};
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
mod era5_member;

fn visit_grib1_envelopes(
    input: &Path,
    mut visit: impl FnMut(usize, &[u8]) -> Result<(), Box<dyn Error>>,
) -> Result<usize, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(input)?);
    let mut offset = 0usize;
    let mut index = 0usize;
    loop {
        let mut header = [0u8; 8];
        if reader.read(&mut header[..1])? == 0 {
            break;
        }
        reader.read_exact(&mut header[1..]).map_err(|_| {
            format!(
                "truncated GRIB1 message {index} at byte {offset}: fewer than 8 indicator bytes"
            )
        })?;
        if &header[..4] != b"GRIB" {
            return Err(format!(
                "invalid GRIB1 message {index} at byte {offset}: missing GRIB marker"
            )
            .into());
        }
        if header[7] != 1 {
            return Err(format!(
                "message {index} at byte {offset} is GRIB edition {}, expected edition 1",
                header[7]
            )
            .into());
        }
        let declared =
            ((header[4] as usize) << 16) | ((header[5] as usize) << 8) | header[6] as usize;
        if declared < 12 {
            return Err(format!("invalid GRIB1 message {index} at byte {offset}: declared length {declared} is too short").into());
        }
        // A GRIB1 envelope is at most 2^24-1 bytes. Metadata queries retain
        // only this one message, independent of the whole file's length.
        let mut bytes = vec![0u8; declared];
        bytes[..8].copy_from_slice(&header);
        reader.read_exact(&mut bytes[8..]).map_err(|_| {
            format!("truncated GRIB1 message {index} at byte {offset}: declared length {declared} exceeds remaining file")
        })?;
        if &bytes[declared - 4..] != b"7777" {
            return Err(format!(
                "invalid GRIB1 message {index} at byte {offset}: missing 7777 terminator"
            )
            .into());
        }
        visit(index, &bytes)?;
        offset = offset
            .checked_add(declared)
            .ok_or("GRIB1 file offset overflows")?;
        index += 1;
    }
    if index == 0 {
        return Err("GRIB1 input is empty".into());
    }
    Ok(index)
}

fn validate_grib1_envelopes(input: &Path) -> Result<usize, Box<dyn Error>> {
    visit_grib1_envelopes(input, |_, _| Ok(()))
}

fn grid_shape_and_scan(grid: &GridType) -> Result<(usize, usize, u8), Box<dyn Error>> {
    match grid {
        GridType::LatLon {
            ni,
            nj,
            scanning_mode,
            ..
        } => Ok((*ni as usize, *nj as usize, *scanning_mode)),
        _ => Err("ERA5 bridge requires a regular latitude/longitude GRIB1 grid".into()),
    }
}

fn write_json_number_array<W: Write>(
    writer: &mut W,
    values: impl IntoIterator<Item = f64>,
) -> std::io::Result<()> {
    write!(writer, "[")?;
    let mut first = true;
    for value in values {
        if !first {
            write!(writer, ",")?;
        }
        first = false;
        write!(writer, "{value:.15}")?;
    }
    write!(writer, "]")
}

fn open_complete(input: &Path) -> Result<Grib1File, Box<dyn Error>> {
    // Reject corrupt concatenated inputs before grib-core can return a
    // superficially useful prefix.  The vendored July hardening validates
    // every section internally; this outer walk validates every message
    // envelope and exact EOF coverage.
    let envelopes = validate_grib1_envelopes(input)?;
    let file = Grib1File::open(input)?;
    if file.messages.is_empty() {
        return Err("GRIB1 input contains no messages".into());
    }
    // `Grib1File::from_bytes` skips a message whose sections do not parse and
    // still returns Ok, so an envelope-valid file can decode to N-1 messages
    // with only a stderr warning nothing reads.  The envelope walk above is
    // the independent count; a shortfall is a dropped field, not a smaller
    // file, and it must not reach a dump that looks complete.
    if file.messages.len() != envelopes {
        return Err(format!(
            "GRIB1 input has {envelopes} message envelopes but the decoder \
             returned {}; a message failed to parse and was skipped",
            file.messages.len()
        )
        .into());
    }
    Ok(file)
}

fn write_message_metadata<W: Write>(
    metadata: &mut W,
    message: &Grib1Message,
    offset_values: usize,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    let pds = &message.pds;
    let gds = message.gds.as_ref().ok_or("GRIB1 message has no GDS")?;
    let (message_nx, message_ny, message_scan) = grid_shape_and_scan(&gds.grid_type)?;
    // Exact original grid bytes let a consumer bind fields without treating
    // equal dimensions as equal coordinates. Packing metadata is diagnostic;
    // the decoder does not clamp or otherwise change source values.
    let grid_definition_hex: String = gds.raw.iter().map(|byte| format!("{byte:02x}")).collect();
    write!(
        metadata,
        concat!(
            "{{\"offset_values\":{},\"count\":{},",
            "\"parameter\":{},\"level_type\":{},\"level\":{},",
            "\"table_version\":{},\"center\":{},",
            "\"grid_definition_hex\":\"{}\",",
            "\"binary_scale\":{},\"decimal_scale\":{},\"bits_per_value\":{},\"reference_value\":{},",
            "\"nx\":{},\"ny\":{},\"scan_mode\":{},",
            "\"year\":{},\"month\":{},",
            "\"day\":{},\"hour\":{},\"minute\":{},",
            "\"time_unit\":{},\"p1\":{},\"p2\":{},",
            "\"time_range_indicator\":{},\"has_bitmap\":{}}}"
        ),
        offset_values,
        count,
        pds.parameter,
        pds.level_type,
        pds.level_value,
        pds.table_version,
        pds.center_id,
        grid_definition_hex,
        message.bds.binary_scale,
        pds.decimal_scale,
        message.bds.bits_per_value,
        message.bds.reference_value,
        message_nx,
        message_ny,
        message_scan,
        pds.year(),
        pds.month,
        pds.day,
        pds.hour,
        pds.minute,
        pds.time_unit,
        pds.p1,
        pds.p2,
        pds.time_range_indicator,
        message.bms.is_some(),
    )?;
    Ok(())
}

fn inventory<W: Write>(input: &Path, metadata: &mut W) -> Result<(), Box<dyn Error>> {
    // The same strict native envelope and section parsers as normal decode,
    // one message at a time. No field values or coordinate arrays are unpacked.
    writeln!(
        metadata,
        "{{\"format_version\":1,\"edition\":1,\"metadata_only\":true,\"messages\":["
    )?;
    visit_grib1_envelopes(input, |index, bytes| {
        let file = Grib1File::from_bytes(bytes)?;
        if file.messages.len() != 1 {
            return Err(format!("GRIB1 message {index} failed to parse and was skipped").into());
        }
        if index != 0 {
            writeln!(metadata, ",")?;
        }
        let message = &file.messages[0];
        write_message_metadata(metadata, message, 0, message.num_data_points())
    })?;
    writeln!(metadata, "]}}")?;
    Ok(())
}

fn run(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let file = open_complete(input)?;
    fs::create_dir_all(output)?;

    // CDO adds a one-point ``utc_date`` control record between pressure
    // parameters.  Use the largest (meteorological) grid as the primary grid
    // and retain small auxiliary records in the raw stream for Python to
    // ignore through the Vtable.
    let primary = file
        .messages
        .iter()
        .max_by_key(|message| message.num_data_points())
        .ok_or("GRIB1 input contains no messages")?;
    if primary.indicator.edition != 1 {
        return Err("input is not GRIB edition 1".into());
    }
    let first_gds = primary
        .gds
        .as_ref()
        .ok_or("primary GRIB1 message has no GDS")?;
    let (nx, ny, scanning_mode) = grid_shape_and_scan(&first_gds.grid_type)?;
    if scanning_mode & 0x20 != 0 {
        return Err("j-consecutive GRIB1 scanning is not supported by this bridge".into());
    }
    let coordinates = primary.latlons()?;
    if coordinates.len() != nx * ny {
        return Err("GRIB1 coordinate count does not match grid dimensions".into());
    }
    let latitude = (0..ny).map(|j| coordinates[j * nx].lat).collect::<Vec<_>>();
    let longitude = (0..nx).map(|i| coordinates[i].lon).collect::<Vec<_>>();
    for j in 0..ny {
        for i in 0..nx {
            let point = coordinates[j * nx + i];
            if (point.lat - latitude[j]).abs() > 1.0e-10
                || (point.lon - longitude[i]).abs() > 1.0e-10
            {
                return Err("GRIB1 grid is not separable into latitude/longitude axes".into());
            }
        }
    }

    let mut raw = BufWriter::new(File::create(output.join("values.f64"))?);
    let mut metadata = BufWriter::new(File::create(output.join("metadata.json"))?);
    writeln!(metadata, "{{")?;
    writeln!(metadata, "\"format_version\":1,")?;
    writeln!(metadata, "\"edition\":1,")?;
    writeln!(metadata, "\"dtype\":\"<f8\",")?;
    writeln!(metadata, "\"shape\":[{ny},{nx}],")?;
    write!(metadata, "\"latitude\":")?;
    write_json_number_array(&mut metadata, latitude)?;
    writeln!(metadata, ",")?;
    write!(metadata, "\"longitude\":")?;
    write_json_number_array(&mut metadata, longitude)?;
    writeln!(metadata, ",")?;
    writeln!(metadata, "\"messages\":[")?;

    let mut offset_values = 0usize;
    for (index, message) in file.messages.iter().enumerate() {
        if message.indicator.edition != 1 {
            return Err(format!("message {index} is not GRIB edition 1").into());
        }
        let gds = message
            .gds
            .as_ref()
            .ok_or_else(|| format!("message {index} has no GDS"))?;
        let (message_nx, message_ny, message_scan) = grid_shape_and_scan(&gds.grid_type)?;
        if message_scan & 0x20 != 0 {
            return Err(format!("message {index} uses j-consecutive scanning").into());
        }
        let values = message.values()?;
        if values.len() != message_nx * message_ny {
            return Err(format!("message {index} value count disagrees with its grid").into());
        }
        for value in &values {
            raw.write_all(&value.to_le_bytes())?;
        }
        if index != 0 {
            writeln!(metadata, ",")?;
        }
        write_message_metadata(&mut metadata, message, offset_values, values.len())?;
        offset_values += values.len();
    }
    writeln!(metadata)?;
    writeln!(metadata, "]")?;
    writeln!(metadata, "}}")?;
    raw.flush()?;
    metadata.flush()?;
    Ok(())
}

fn main() {
    // Keep the release-provenance stamp in the binary: the cut
    // proves a staged bridge by these bytes (see lib.rs).
    let _ = std::hint::black_box(gpuwm_preprocess_cpu::SOURCE_REV_STAMP);
    let arguments = env::args_os().collect::<Vec<_>>();
    if arguments.len() == 2 && arguments[1] == "--era5-member-capabilities" {
        println!("{{\"schema\":\"{}\",\"members\":[0,1,2,3,4,5,6,7,8,9],\"local_definitions\":[1,17,36],\"complete_input_census\":true,\"table_qualified_census\":true,\"lake_surface_parameters\":{{\"center\":98,\"table\":228,\"parameters\":[8,13,14],\"surface_only\":true}},\"byte_preserving\":true}}", era5_member::SCHEMA);
        return;
    }
    if arguments.len() >= 2 && (arguments[1] == "--era5-member" || arguments[1] == "--check-era5-member") {
        let result = (|| -> Result<(), Box<dyn Error>> {
            let selecting = arguments[1] == "--era5-member";
            if arguments.len() != if selecting { 5 } else { 4 } {
                return Err("usage: grib1_bridge --era5-member N INPUT OUTPUT | --check-era5-member N INPUT".into());
            }
            let member: u8 = arguments[2].to_str().ok_or("Member must be 0..9")?.parse()?;
            if selecting { era5_member::select(Path::new(&arguments[3]), Path::new(&arguments[4]), member) }
            else { era5_member::check(Path::new(&arguments[3]), member) }
        })();
        if let Err(error) = result { eprintln!("grib1_bridge: {error}"); std::process::exit(1); }
        return;
    }
    if arguments.len() != 3 {
        eprintln!("usage: grib1_bridge INPUT.grb OUTPUT_DIR | --inventory INPUT.grb");
        std::process::exit(2);
    }
    let result = if arguments[1] == "--inventory" {
        let mut output = BufWriter::new(std::io::stdout().lock());
        inventory(Path::new(&arguments[2]), &mut output)
            .and_then(|()| output.flush().map_err(Into::into))
    } else {
        run(Path::new(&arguments[1]), Path::new(&arguments[2]))
    };
    if let Err(error) = result {
        eprintln!("grib1_bridge: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::validate_grib1_envelopes;

    #[test]
    fn the_envelope_walk_returns_the_count_the_decoder_must_match() {
        // `run()` compares this count against `Grib1File::open`'s message
        // list, so an off-by-one here would either refuse every good file or
        // accept one the decoder silently shortened.
        let mut bytes = Vec::new();
        for _ in 0..3 {
            bytes.extend_from_slice(b"GRIB");
            bytes.extend_from_slice(&[0, 0, 12, 1]);
            bytes.extend_from_slice(b"7777");
        }
        let path = std::env::temp_dir().join(format!(
            "grib1_bridge_envelope_count_{}.grb",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let count = validate_grib1_envelopes(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(count.expect("three valid envelopes"), 3);
    }
}
