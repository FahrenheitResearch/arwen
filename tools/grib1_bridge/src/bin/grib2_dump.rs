//! Dump selected GRIB2 fields as little-endian f64 for decoder cross-checks.

use grib_core::grib2::{unpack_message, Grib2File};
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

const DUMP_HEADER: &str = concat!(
    "index\tdiscipline\tcategory\tparameter\tcenter\tsubcenter\t",
    "master_table_version\tlocal_table_version\tlevel_type\tlevel_value\t",
    "second_level_type\tsecond_level_value\tmember\tpdt\tdrt\tnx\tny\t",
    "scan_mode\tbitmap\tcount\tfinite\tmissing\tminimum\tmaximum\tfilename"
);

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let input = args
        .next()
        .ok_or("usage: grib2_dump INPUT.grib2 OUTPUT_DIR INDEX [INDEX ...]")?;
    let output = args.next().ok_or("missing OUTPUT_DIR")?;
    let indices: Vec<usize> = args
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid field index {value:?}: {error}"))
        })
        .collect::<Result<_, _>>()?;
    if indices.is_empty() {
        return Err("at least one field index is required".into());
    }
    let output = Path::new(&output);
    if output.exists() {
        return Err(format!("refusing to overwrite existing output {output:?}").into());
    }
    fs::create_dir(output)?;

    let file = Grib2File::open(&input)?;
    let mut metadata = BufWriter::new(File::create(output.join("metadata.tsv"))?);
    writeln!(metadata, "{DUMP_HEADER}")?;
    for index in indices {
        let message = file.messages.get(index).ok_or_else(|| {
            format!(
                "field index {index} is out of range for {} parsed fields",
                file.messages.len()
            )
        })?;
        let values = unpack_message(message)
            .map_err(|error| format!("field {index} failed to decode: {error}"))?;
        let filename = format!("field-{index:04}.f64le");
        let mut stream = BufWriter::new(File::create(output.join(&filename))?);
        let mut finite = 0usize;
        let mut missing = 0usize;
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for value in values.iter().copied() {
            stream.write_all(&value.to_le_bytes())?;
            if value.is_finite() {
                finite += 1;
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            } else {
                missing += 1;
            }
        }
        stream.flush()?;
        writeln!(
            metadata,
            "{index}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0x{:02x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            message.discipline,
            message.product.parameter_category,
            message.product.parameter_number,
            message.identification.center_id,
            message.identification.subcenter_id,
            message.identification.master_table_version,
            message.identification.local_table_version,
            message.product.level_type,
            message.product.level_value,
            message.product.second_level_type,
            message.product.second_level_value,
            message
                .product
                .perturbation_number
                .map_or_else(|| "-".to_owned(), |value| value.to_string()),
            message.product.template,
            message.data_rep.template,
            message.grid.nx,
            message.grid.ny,
            message.grid.scan_mode,
            message.bitmap.is_some(),
            values.len(),
            finite,
            missing,
            if finite == 0 { f64::NAN } else { minimum },
            if finite == 0 { f64::NAN } else { maximum },
            filename,
        )?;
    }
    metadata.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::DUMP_HEADER;

    #[test]
    fn authority_identity_is_part_of_the_dump_abi() {
        let columns: Vec<_> = DUMP_HEADER.split('\t').collect();
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
