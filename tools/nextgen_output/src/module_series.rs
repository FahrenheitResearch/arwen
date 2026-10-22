//! Original, typed module/catchment history interchange. No routing semantics,
//! unit conversion or numerical stepping is inferred by this boundary.
use crate::Result;
use chrono::TimeDelta;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, io::Write, path::PathBuf, time::Instant};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub component: String,
    pub name: String,
    pub units: String,
    pub dtype: String,
    pub offset: usize,
    /// Per-feature trailing axes, in native C order; empty means scalar.
    #[serde(default)]
    pub element_shape: Vec<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub output_directory: PathBuf,
    pub origin: String,
    pub interval_seconds: u64,
    pub cadence_steps: u64,
    pub frames: usize,
    pub capacity: usize,
    pub feature_ids: Vec<String>,
    pub fields: Vec<Field>,
    #[serde(default)]
    pub provenance: serde_json::Map<String, serde_json::Value>,
}

fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| "module history extent overflow".into())
}

fn text(value: &str) -> String {
    if value.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn number(dtype: &str, b: &[u8]) -> String {
    match dtype {
        "f32le" => f32::from_le_bytes(b.try_into().unwrap()).to_string(),
        "f64le" => f64::from_le_bytes(b.try_into().unwrap()).to_string(),
        "i8" => (b[0] as i8).to_string(),
        "u8" => b[0].to_string(),
        "i16le" => i16::from_le_bytes(b.try_into().unwrap()).to_string(),
        "u16le" => u16::from_le_bytes(b.try_into().unwrap()).to_string(),
        "i32le" => i32::from_le_bytes(b.try_into().unwrap()).to_string(),
        "u32le" => u32::from_le_bytes(b.try_into().unwrap()).to_string(),
        "i64le" => i64::from_le_bytes(b.try_into().unwrap()).to_string(),
        "u64le" => u64::from_le_bytes(b.try_into().unwrap()).to_string(),
        _ => unreachable!("validated primitive"),
    }
}

pub fn write(plan: &Plan, payload: &[u8]) -> Result<serde_json::Value> {
    let start = Instant::now();
    if plan.frames == 0
        || plan.frames > plan.capacity
        || plan.feature_ids.is_empty()
        || plan.fields.is_empty()
        || plan.interval_seconds == 0
        || plan.cadence_steps == 0
    {
        return Err("module history needs nonempty fields, features and captured intervals".into());
    }
    let mut ids = HashSet::new();
    if plan
        .feature_ids
        .iter()
        .any(|id| id.is_empty() || !ids.insert(id))
    {
        return Err("module history feature IDs must be explicit, nonempty and unique".into());
    }
    let origin = crate::series::parse_origin(&plan.origin)?;
    let cadence = plan
        .interval_seconds
        .checked_mul(plan.cadence_steps)
        .ok_or("module history clock overflow")?;
    let times: Vec<String> = (0..plan.frames)
        .map(|t| -> Result<String> {
            let seconds = u64::try_from(t)?
                .checked_add(1)
                .and_then(|n| n.checked_mul(cadence))
                .ok_or("module history clock overflow")?;
            let delta = TimeDelta::try_seconds(i64::try_from(seconds)?)
                .ok_or("module history clock overflow")?;
            Ok(origin
                .checked_add_signed(delta)
                .ok_or("module history clock overflow")?
                .format("%Y-%m-%d %H:%M:%S")
                .to_string())
        })
        .collect::<Result<_>>()?;
    let mut names = HashSet::new();
    let layouts = plan
        .fields
        .iter()
        .map(|field| -> Result<(usize, usize)> {
            if field.component.is_empty()
                || field.name.is_empty()
                || field.units.is_empty()
                || !names.insert((&field.component, &field.name))
            {
                return Err(
                    "module history fields need unique component/name and explicit units".into(),
                );
            }
            let width = match field.dtype.as_str() {
                "f32le" | "i32le" | "u32le" => 4,
                "f64le" | "i64le" | "u64le" => 8,
                "i16le" | "u16le" => 2,
                "i8" | "u8" => 1,
                _ => return Err("module history requires a native primitive dtype".into()),
            };
            if field.element_shape.contains(&0) {
                return Err("module history trailing axes must be nonempty".into());
            }
            let elements = field.element_shape.iter().try_fold(1, |n, &d| mul(n, d))?;
            let bytes = mul(
                mul(mul(plan.capacity, plan.feature_ids.len())?, elements)?,
                width,
            )?;
            let end = field
                .offset
                .checked_add(bytes)
                .ok_or("module history payload offset overflow")?;
            if end > payload.len() {
                return Err("module history field exceeds complete capacity snapshot".into());
            }
            Ok((width, elements))
        })
        .collect::<Result<Vec<_>>>()?;
    // All metadata/extents validated before any output is created. Each field
    // formats concurrently; collection preserves the caller's declared order.
    let formatted = plan
        .fields
        .par_iter()
        .zip(layouts.par_iter())
        .map(|(field, &(width, elements))| {
            let mut csv = String::new();
            let component = text(&field.component);
            let name = text(&field.name);
            let units = text(&field.units);
            for (t, time) in times.iter().enumerate() {
                for (i, id) in plan.feature_ids.iter().enumerate() {
                    let id = text(id);
                    for element in 0..elements {
                        let offset = field.offset
                            + ((t * plan.feature_ids.len() + i) * elements + element) * width;
                        let word = &payload[offset..offset + width];
                        let bits: String = word.iter().rev().map(|b| format!("{b:02x}")).collect();
                        use std::fmt::Write;
                        writeln!(
                            csv,
                            "{time},{id},{component},{name},{units},{element},{},{},{bits}",
                            field.dtype,
                            number(&field.dtype, word)
                        )
                        .unwrap();
                    }
                }
            }
            csv
        })
        .collect::<Vec<_>>();
    let format_ms = start.elapsed().as_secs_f64() * 1000.;
    let rows = layouts
        .iter()
        .try_fold(0usize, |n, &(_, elements)| -> Result<usize> {
            n.checked_add(mul(mul(plan.frames, plan.feature_ids.len())?, elements)?)
                .ok_or_else(|| "module history row count overflow".into())
        })?;
    let mut metadata = serde_json::json!({"schema":"arwen.module-field-series.v1","plan":plan,
        "row_order":"declared field, frame, feature, flattened trailing element",
        "time_convention":"interval ends after origin; timezone inherited from provenance",
        "bits_hex":"native primitive word, most significant byte first; preserves NaN payload and signed zero",
        "csv":"module-fields.csv","rows":rows});
    if let Some(parent) = plan.output_directory.parent() {
        fs::create_dir_all(parent)?;
    }
    // Exclusive directory ownership prevents overwriting another run. The JSON
    // completion manifest is written last; interrupted CSV is never complete.
    fs::create_dir(&plan.output_directory)?;
    let result = (|| -> Result<_> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(plan.output_directory.join("module-fields.csv"))?;
        let header = b"time,feature_id,component,field,units,element,dtype,value,bits_hex\n";
        file.write_all(header)?;
        let mut digest = Sha256::new();
        digest.update(header);
        let mut bytes = header.len();
        for block in formatted {
            file.write_all(block.as_bytes())?;
            digest.update(block.as_bytes());
            bytes += block.len();
        }
        file.sync_all()?;
        let csv_sha256 = format!("{:x}", digest.finalize());
        metadata["csv_sha256"] = csv_sha256.clone().into();
        let mut manifest = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(plan.output_directory.join(".module-fields.json.pending"))?;
        manifest.write_all(&serde_json::to_vec_pretty(&metadata)?)?;
        manifest.sync_all()?;
        fs::rename(
            plan.output_directory.join(".module-fields.json.pending"),
            plan.output_directory.join("module-fields.json"),
        )?;
        crate::sync_directory(&plan.output_directory)?;
        Ok(
            serde_json::json!({"operation":"module-series-write","schema":"arwen.module-field-series.v1",
            "rows":rows,"fields":plan.fields.len(),"features":plan.feature_ids.len(),"frames":plan.frames,
            "csv_bytes":bytes,"csv_sha256":csv_sha256,"format_elapsed_ms":format_ms,
            "boundary_ms":start.elapsed().as_secs_f64()*1000.,
            "parallelism":format!("{} Rayon workers, fields format concurrently; ordered CSV/container publication",rayon::current_num_threads()),
            "scope":"typed module/catchment boundary serialization; no unit conversion or routing semantics"}),
        )
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&plan.output_directory);
    }
    result
}
