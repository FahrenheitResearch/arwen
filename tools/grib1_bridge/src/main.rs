//! Minimal GRIB1-to-raw/JSON bridge for gpuwm ERA5 ingest.

use grib_core::grib1::{Grib1File, GridType};
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

fn validate_grib1_envelopes(input: &Path) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(input)?;
    if bytes.is_empty() {
        return Err("GRIB1 input is empty".into());
    }

    let mut offset = 0usize;
    let mut index = 0usize;
    while offset < bytes.len() {
        if bytes.len() - offset < 8 {
            return Err(format!(
                "truncated GRIB1 message {index} at byte {offset}: fewer than 8 indicator bytes"
            )
            .into());
        }
        if &bytes[offset..offset + 4] != b"GRIB" {
            return Err(format!(
                "invalid GRIB1 message {index} at byte {offset}: missing GRIB marker"
            )
            .into());
        }
        if bytes[offset + 7] != 1 {
            return Err(format!(
                "message {index} at byte {offset} is GRIB edition {}, expected edition 1",
                bytes[offset + 7]
            )
            .into());
        }
        let declared = ((bytes[offset + 4] as usize) << 16)
            | ((bytes[offset + 5] as usize) << 8)
            | bytes[offset + 6] as usize;
        if declared < 12 {
            return Err(format!(
                "invalid GRIB1 message {index} at byte {offset}: declared length {declared} is too short"
            )
            .into());
        }
        let end = offset
            .checked_add(declared)
            .ok_or_else(|| format!("GRIB1 message {index} length overflows at byte {offset}"))?;
        if end > bytes.len() {
            return Err(format!(
                "truncated GRIB1 message {index} at byte {offset}: declared end {end}, file has {} bytes",
                bytes.len()
            )
            .into());
        }
        if &bytes[end - 4..end] != b"7777" {
            return Err(format!(
                "invalid GRIB1 message {index} at byte {offset}: missing 7777 terminator at byte {}",
                end - 4
            )
            .into());
        }
        offset = end;
        index += 1;
    }
    Ok(())
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

fn run(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    // Reject corrupt concatenated inputs before grib-core can return a
    // superficially useful prefix.  The vendored July hardening validates
    // every section internally; this outer walk validates every message
    // envelope and exact EOF coverage.
    validate_grib1_envelopes(input)?;
    let file = Grib1File::open(input)?;
    if file.messages.is_empty() {
        return Err("GRIB1 input contains no messages".into());
    }
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
        let pds = &message.pds;
        if index != 0 {
            writeln!(metadata, ",")?;
        }
        write!(
            metadata,
            concat!(
                "{{\"offset_values\":{},\"count\":{},",
                "\"parameter\":{},\"level_type\":{},\"level\":{},",
                "\"table_version\":{},\"center\":{},",
                "\"nx\":{},\"ny\":{},\"scan_mode\":{},",
                "\"year\":{},\"month\":{},",
                "\"day\":{},\"hour\":{},\"minute\":{},",
                "\"time_unit\":{},\"p1\":{},\"p2\":{},",
                "\"time_range_indicator\":{},\"has_bitmap\":{}}}"
            ),
            offset_values,
            values.len(),
            pds.parameter,
            pds.level_type,
            pds.level_value,
            pds.table_version,
            pds.center_id,
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
    let arguments = env::args_os().collect::<Vec<_>>();
    if arguments.len() != 3 {
        eprintln!("usage: grib1_bridge INPUT.grb OUTPUT_DIR");
        std::process::exit(2);
    }
    if let Err(error) = run(Path::new(&arguments[1]), Path::new(&arguments[2])) {
        eprintln!("grib1_bridge: {error}");
        std::process::exit(1);
    }
}
