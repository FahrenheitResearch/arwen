//! Strict GRIB2 inventory/decode probe for native-HRRR capability work.

use grib_core::grib2::{level_name, parameter_name, unpack_message, Grib2File};
use std::env;
use std::error::Error;
use std::fs;

const INVENTORY_HEADER: &str = concat!(
    "index\tdiscipline\tcategory\tparameter\tcenter\tsubcenter\t",
    "master_table_version\tlocal_table_version\tname\treference_time\t",
    "forecast_unit\tforecast_time\tpdt\tlevel_type\tlevel_value\t",
    "second_level_type\tsecond_level_value\tmember\tgenerating_process\t",
    "forecast_generating_process_id\tlevel_name\tgdt\tnx\tny\tlat1\tlon1\t",
    "dx\tdy\tlatin1\tlatin2\tlov\tscan_mode\tshape_of_earth\t",
    "resolution_flags\tdrt\tbitmap\tpacked_points\traw_bytes\tdecoded_points\t",
    "finite_points\tmissing_points\tminimum\tmaximum"
);

fn read_u64_be(bytes: &[u8]) -> u64 {
    u64::from_be_bytes(bytes.try_into().expect("validated eight-byte slice"))
}

fn validate_envelopes(bytes: &[u8]) -> Result<usize, Box<dyn Error>> {
    if bytes.is_empty() {
        return Err("GRIB2 input is empty".into());
    }
    let mut offset = 0usize;
    let mut count = 0usize;
    while offset < bytes.len() {
        if bytes.len() - offset < 20 {
            return Err(format!(
                "truncated GRIB2 envelope {count} at byte {offset}: fewer than 20 bytes"
            )
            .into());
        }
        if &bytes[offset..offset + 4] != b"GRIB" {
            return Err(format!(
                "invalid GRIB2 envelope {count} at byte {offset}: missing GRIB marker"
            )
            .into());
        }
        let edition = bytes[offset + 7];
        if edition != 2 {
            return Err(format!(
                "envelope {count} at byte {offset} is GRIB edition {edition}, expected 2"
            )
            .into());
        }
        let declared_u64 = read_u64_be(&bytes[offset + 8..offset + 16]);
        let declared = usize::try_from(declared_u64)
            .map_err(|_| format!("envelope {count} length does not fit usize"))?;
        if declared < 20 {
            return Err(format!(
                "invalid GRIB2 envelope {count} at byte {offset}: length {declared} is too short"
            )
            .into());
        }
        let end = offset
            .checked_add(declared)
            .ok_or_else(|| format!("envelope {count} length overflows at byte {offset}"))?;
        if end > bytes.len() {
            return Err(format!(
                "truncated GRIB2 envelope {count} at byte {offset}: declared end {end}, file has {} bytes",
                bytes.len()
            )
            .into());
        }
        if &bytes[end - 4..end] != b"7777" {
            return Err(format!(
                "invalid GRIB2 envelope {count} at byte {offset}: missing 7777 terminator"
            )
            .into());
        }
        offset = end;
        count += 1;
    }
    Ok(count)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Keep the release-provenance stamp in the binary: the cut
    // proves a staged bridge by these bytes (see lib.rs).
    let _ = std::hint::black_box(gpuwm_preprocess_cpu::SOURCE_REV_STAMP);
    let mut args = env::args().skip(1);
    let input = args
        .next()
        .ok_or("usage: grib2_inventory INPUT.grib2 [--decode]")?;
    let decode = match args.next().as_deref() {
        None => false,
        Some("--decode") => true,
        Some(other) => return Err(format!("unknown argument {other:?}").into()),
    };
    if args.next().is_some() {
        return Err("too many arguments".into());
    }

    let bytes = fs::read(&input)?;
    let envelope_count = validate_envelopes(&bytes)?;
    let file = Grib2File::from_bytes(&bytes)?;
    if file.messages.is_empty() {
        return Err("GRIB2 input contains no parsed fields".into());
    }

    println!("# input\t{input}");
    println!("# bytes\t{}", bytes.len());
    println!("# envelopes\t{envelope_count}");
    println!("# fields\t{}", file.messages.len());
    println!("{INVENTORY_HEADER}");
    for (index, message) in file.messages.iter().enumerate() {
        let (decoded_points, finite_points, missing_points, minimum, maximum) = if decode {
            let values = unpack_message(message)
                .map_err(|error| format!("field {index} failed to decode: {error}"))?;
            let mut finite = 0usize;
            let mut missing = 0usize;
            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;
            for value in &values {
                if value.is_finite() {
                    finite += 1;
                    min_value = min_value.min(*value);
                    max_value = max_value.max(*value);
                } else {
                    missing += 1;
                }
            }
            (
                values.len().to_string(),
                finite.to_string(),
                missing.to_string(),
                if finite == 0 {
                    "nan".into()
                } else {
                    min_value.to_string()
                },
                if finite == 0 {
                    "nan".into()
                } else {
                    max_value.to_string()
                },
            )
        } else {
            ("-".into(), "-".into(), "-".into(), "-".into(), "-".into())
        };
        println!(
            "{index}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0x{:02x}\t{}\t0x{:02x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            message.discipline,
            message.product.parameter_category,
            message.product.parameter_number,
            message.identification.center_id,
            message.identification.subcenter_id,
            message.identification.master_table_version,
            message.identification.local_table_version,
            parameter_name(
                message.discipline,
                message.product.parameter_category,
                message.product.parameter_number
            ),
            message.reference_time,
            message.product.time_range_unit,
            message.product.forecast_time,
            message.product.template,
            message.product.level_type,
            message.product.level_value,
            message.product.second_level_type,
            message.product.second_level_value,
            message
                .product
                .perturbation_number
                .map_or_else(|| "-".to_owned(), |value| value.to_string()),
            message.product.generating_process,
            message.product.forecast_generating_process_id,
            level_name(message.product.level_type),
            message.grid.template,
            message.grid.nx,
            message.grid.ny,
            message.grid.lat1,
            message.grid.lon1,
            message.grid.dx,
            message.grid.dy,
            message.grid.latin1,
            message.grid.latin2,
            message.grid.lov,
            message.grid.scan_mode,
            message.grid.shape_of_earth,
            message.grid.resolution_flags,
            message.data_rep.template,
            message.bitmap.is_some(),
            message.data_rep.section5_num_data_points,
            message.raw_data.len(),
            decoded_points,
            finite_points,
            missing_points,
            minimum,
            maximum,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::INVENTORY_HEADER;

    #[test]
    fn authority_identity_is_part_of_the_inventory_abi() {
        let columns: Vec<_> = INVENTORY_HEADER.split('\t').collect();
        assert_eq!(
            &columns[4..8],
            &[
                "center",
                "subcenter",
                "master_table_version",
                "local_table_version"
            ]
        );
    }
}
