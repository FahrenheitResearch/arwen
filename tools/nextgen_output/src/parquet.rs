//! Modified t-route 12a8eae0 Apache-2.0 long-form output contract.
//! Rust formats and encodes independent columns on Rayon workers. No Python
//! data processing or native hydrologic runtime participates in this writer.
use crate::{Result, series::Plan};
use bytes::Bytes;
use chrono::NaiveDate;
use parquet::{
    basic::Compression,
    column::{
        reader::ColumnReader,
        writer::{ColumnWriter, get_column_writer},
    },
    data_type::ByteArray,
    file::{
        metadata::KeyValue,
        properties::WriterProperties,
        reader::{FileReader, SerializedFileReader},
        writer::{SerializedFileWriter, SerializedPageWriter, TrackedWrite},
    },
    schema::{parser::parse_message_type, types::SchemaDescriptor},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    path::Path,
    sync::Arc,
    time::Instant,
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    pub prefix_ids: String,
    pub configuration: String,
    #[serde(default)]
    pub segments: Vec<i64>,
}

const SCHEMA: &str = "message schema {
 OPTIONAL BYTE_ARRAY location_id (STRING);
 OPTIONAL DOUBLE value;
 OPTIONAL INT64 value_time (TIMESTAMP(MICROS,false));
 OPTIONAL BYTE_ARRAY variable_name (STRING);
 OPTIONAL BYTE_ARRAY units (STRING);
 OPTIONAL INT32 reference_time (DATE);
 OPTIONAL BYTE_ARRAY configuration (STRING);
 OPTIONAL INT64 __index_level_0__;
}";

fn pandas_metadata() -> String {
    let columns = [
        (Some("location_id"), "location_id", "unicode", "object"),
        (Some("value"), "value", "float64", "float64"),
        (
            Some("value_time"),
            "value_time",
            "datetime",
            "datetime64[us]",
        ),
        (Some("variable_name"), "variable_name", "unicode", "object"),
        (Some("units"), "units", "unicode", "object"),
        (Some("reference_time"), "reference_time", "date", "object"),
        (Some("configuration"), "configuration", "unicode", "object"),
        (None, "__index_level_0__", "int64", "int64"),
    ]
    .map(|(name, field, pandas, numpy)| {
        serde_json::json!({
            "name":name,"field_name":field,"pandas_type":pandas,
            "numpy_type":numpy,"metadata":null
        })
    });
    serde_json::json!({"index_columns":["__index_level_0__"],
        "column_indexes":[{"name":null,"field_name":null,"pandas_type":"unicode",
        "numpy_type":"object","metadata":{"encoding":"UTF-8"}}],
        "columns":columns,"creator":{"library":"rw-nextgen-output","version":env!("CARGO_PKG_VERSION")}
    }).to_string()
}

/// Native t-route sorts the numerical roster, then flattens each Q/V/D time
/// column, retaining the original full-roster dataframe index after filtering.
pub fn write(plan: &Plan, payload: &[u8], ids: &[i64], width: usize) -> Result<serde_json::Value> {
    let started = Instant::now();
    let options = plan
        .parquet
        .as_ref()
        .ok_or("Parquet requires explicit prefix_ids/configuration")?;
    let fields = [
        plan.flow_offset,
        plan.velocity_offset.ok_or("Parquet needs velocity")?,
        plan.depth_offset.ok_or("Parquet needs depth")?,
    ];
    let mut order = (0..ids.len()).collect::<Vec<_>>();
    order.par_sort_unstable_by_key(|&r| ids[r]);
    let requested = options
        .segments
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    if requested.len() != options.segments.len() || requested.iter().any(|id| !ids.contains(id)) {
        return Err("Parquet segment roster contains duplicate/unknown feature IDs".into());
    }
    let selected = order
        .iter()
        .enumerate()
        .filter(|(_, r)| requested.is_empty() || requested.contains(&ids[**r]))
        .map(|(sorted, &source)| (sorted, source))
        .collect::<Vec<_>>();
    let n = selected.len();
    let rows = plan
        .frames
        .checked_mul(3)
        .and_then(|v| v.checked_mul(n))
        .ok_or("Parquet row count overflow")?;
    let origin = crate::series::parse_origin(&plan.origin)?;
    let start_us = origin.and_utc().timestamp_micros();
    let times = (0..plan.frames)
        .map(|t| -> Result<i64> {
            let offset = (plan.time_seconds(t)? * 1e6).round() as i64;
            Ok(start_us
                .checked_add(offset)
                .ok_or("Parquet time overflow")?)
        })
        .collect::<Result<Vec<_>>>()?;
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let reference: i32 = (origin.date() - epoch).num_days().try_into()?;
    let schema = Arc::new(parse_message_type(SCHEMA)?);
    let descriptor = SchemaDescriptor::new(schema.clone());
    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_coerce_types(false)
            .set_created_by(format!(
                "rw-nextgen-output {} (Apache parquet-rs 56.2.0)",
                env!("CARGO_PKG_VERSION")
            ))
            .set_key_value_metadata(Some(vec![KeyValue::new(
                "pandas".into(),
                pandas_metadata(),
            )]))
            .build(),
    );
    // Each column is prepared, dictionary encoded and Snappy compressed on an
    // independent worker. Only the final ordered container splice is serial.
    let encoded = descriptor
        .columns()
        .par_iter()
        .enumerate()
        .map(|(c, desc)| -> Result<_> {
            let mut buffer = TrackedWrite::new(Vec::new());
            let mut writer = get_column_writer(
                desc.clone(),
                props.clone(),
                Box::new(SerializedPageWriter::new(&mut buffer)),
            );
            let mut defs = Vec::with_capacity(rows);
            let mut doubles = Vec::new();
            let mut longs = Vec::new();
            let mut ints = Vec::new();
            let mut strings = Vec::new();
            let locations = selected
                .iter()
                .map(|&(_, r)| {
                    ByteArray::from(format!("{}-{}", options.prefix_ids, ids[r]).as_str())
                })
                .collect::<Vec<_>>();
            for (t, &time) in times.iter().enumerate() {
                for (v, &offset) in fields.iter().enumerate() {
                    for (r, &(sorted, source)) in selected.iter().enumerate() {
                        if c == 1 {
                            let i = offset + (t * plan.reaches + source) * width;
                            let value = if width == 4 {
                                f32::from_le_bytes(payload[i..i + 4].try_into()?) as f64
                            } else {
                                f64::from_le_bytes(payload[i..i + 8].try_into()?)
                            };
                            // Pandas represents NaN as a Parquet null; infinities
                            // remain numerical values, preserving native semantics.
                            defs.push(i16::from(!value.is_nan()));
                            if !value.is_nan() {
                                doubles.push(value)
                            }
                        } else {
                            defs.push(1);
                            match c {
                                0 => strings.push(locations[r].clone()),
                                2 => longs.push(time),
                                3 => strings
                                    .push(ByteArray::from(["streamflow", "velocity", "depth"][v])),
                                4 => strings.push(ByteArray::from(["m3/s", "m/s", "m"][v])),
                                5 => ints.push(reference),
                                6 => strings.push(ByteArray::from(options.configuration.as_str())),
                                7 => longs.push(((t * 3 + v) * ids.len() + sorted).try_into()?),
                                _ => unreachable!(),
                            }
                        }
                    }
                }
            }
            match &mut writer {
                ColumnWriter::ByteArrayColumnWriter(w) => {
                    w.write_batch(&strings, Some(&defs), None)?;
                }
                ColumnWriter::DoubleColumnWriter(w) => {
                    w.write_batch(&doubles, Some(&defs), None)?;
                }
                ColumnWriter::Int64ColumnWriter(w) => {
                    w.write_batch(&longs, Some(&defs), None)?;
                }
                ColumnWriter::Int32ColumnWriter(w) => {
                    w.write_batch(&ints, Some(&defs), None)?;
                }
                _ => unreachable!(),
            }
            let close = writer.close()?;
            Ok((Bytes::from(buffer.into_inner()?), close))
        })
        .collect::<Result<Vec<_>>>()?;
    let encode_ms = started.elapsed().as_secs_f64() * 1000.;
    let path = plan.output_directory.join("flowveldepth.parquet");
    let partial = path.with_extension("parquet.partial");
    let result = (|| -> Result<_> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)?;
        let mut writer = SerializedFileWriter::new(file, schema, props)?;
        let mut group = writer.next_row_group()?;
        for (bytes, close) in encoded {
            group.append_column(&bytes, close)?;
        }
        group.close()?;
        writer.close()?;
        std::fs::OpenOptions::new().write(true).open(&partial)?.sync_all()?;
        fs::hard_link(&partial, &path)?;
        Ok(
            serde_json::json!({"operation":"troute-parquet","path":path,"rows":rows,
            "columns":8,"encode_ms":encode_ms,"total_ms":started.elapsed().as_secs_f64()*1000.,
            "parallelism":format!("8 independent column tasks; {} Rayon workers; ordered container splice",rayon::current_num_threads()),
            "scope":"native ordered typed schema and long-form values; Rust creator metadata, no PyArrow IPC footer clone"}),
        )
    })();
    let _ = fs::remove_file(partial);
    result
}

/// Compare every flat native column's definition levels and exact typed words.
/// Physical/footer metadata differences are separately reported, never hidden.
pub fn compare(left: &Path, right: &Path) -> Result<serde_json::Value> {
    // File::try_clone shares a seek cursor on Linux. Immutable Bytes gives
    // independent column readers while Rayon decodes all columns in parallel.
    let readers = [
        SerializedFileReader::new(Bytes::from(fs::read(left)?))?,
        SerializedFileReader::new(Bytes::from(fs::read(right)?))?,
    ];
    let [a, b] = [&readers[0], &readers[1]];
    let am = a.metadata().file_metadata();
    let bm = b.metadata().file_metadata();
    if am.schema_descr() != bm.schema_descr() || am.num_rows() != bm.num_rows() {
        return Err("Parquet ordered physical/logical schema or row count differs".into());
    }
    fn column(reader: &SerializedFileReader<Bytes>, c: usize) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for g in 0..reader.num_row_groups() {
            let group = reader.get_row_group(g)?;
            let n: usize = group.metadata().num_rows().try_into()?;
            let mut defs = Vec::new();
            macro_rules! read {
                ($w:expr,$convert:expr) => {{
                    let mut values = Vec::new();
                    let (records, _, _) = $w.read_records(n, Some(&mut defs), None, &mut values)?;
                    if records != n {
                        return Err("incomplete Parquet column".into());
                    }
                    let mut values = values.into_iter();
                    for d in &defs {
                        out.extend_from_slice(&d.to_le_bytes());
                        if *d != 0 {
                            $convert(values.next().ok_or("missing column value")?, &mut out);
                        }
                    }
                }};
            }
            match group.get_column_reader(c)? {
                ColumnReader::Int32ColumnReader(mut w) => read!(w, |v: i32, o: &mut Vec<u8>| o
                    .extend_from_slice(&v.to_le_bytes())),
                ColumnReader::Int64ColumnReader(mut w) => read!(w, |v: i64, o: &mut Vec<u8>| o
                    .extend_from_slice(&v.to_le_bytes())),
                ColumnReader::DoubleColumnReader(mut w) => read!(w, |v: f64, o: &mut Vec<u8>| o
                    .extend_from_slice(&v.to_le_bytes())),
                ColumnReader::ByteArrayColumnReader(mut w) => {
                    read!(w, |v: ByteArray, o: &mut Vec<u8>| {
                        o.extend_from_slice(&(v.len() as u64).to_le_bytes());
                        o.extend_from_slice(v.data());
                    })
                }
                _ => return Err("not a supported flat native output column".into()),
            }
        }
        Ok(out)
    }
    let columns = (0..am.schema_descr().num_columns()).into_par_iter().map(|c| -> Result<_> {
        let l = column(a,c)?;let r=column(b,c)?;
        if l!=r {return Err(format!("Parquet column {} differs",am.schema_descr().column(c).name()).into())}
        Ok(serde_json::json!({"name":am.schema_descr().column(c).name(),"physical_type":format!("{:?}",am.schema_descr().column(c).physical_type()),"typed_value_and_definition_bytes":l.len(),"different_bytes":0}))
    }).collect::<Result<Vec<_>>>()?;
    // Keep metadata truth explicit: native ARROW:schema serializes its Python
    // library versions. The Rust writer identifies itself and never forges it.
    Ok(
        serde_json::json!({"operation":"compare-parquet","schema":"all ordered physical/logical fields identical",
        "rows":am.num_rows(),"columns":columns,"different_value_bytes":0,
        "key_value_metadata_identical":am.key_value_metadata()==bm.key_value_metadata(),
        "created_by_identical":am.created_by()==bm.created_by(),
        "whole_file_identical":fs::read(left)?==fs::read(right)?}),
    )
}
