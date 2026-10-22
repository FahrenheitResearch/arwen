//! NextGen output boundary. No numerical stepping or Fortran dependency.
//!
//! Modified implementation of the per-nexus CSV contract in NOAA-OWP/ngen,
//! be11a180565a78ff069d855b0b5dbfc23845e7d8, Apache-2.0.
//! Original contract: Copyright (C) 2025-2026 Lynker, Robert Bartel.
//! This Rust implementation batches all nexuses over the Rayon worker pool.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    ffi::c_char,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

pub mod capi;
pub mod checkpoint;
pub mod module_series;
pub mod netcdf;
pub mod parquet;
pub mod series;

/// Flush directory entries on Unix. Windows file sync and hard-link publication
/// provide the corresponding supported boundary; opening a directory as File fails.
pub(crate) fn sync_directory(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    fs::File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "C" fn rw_nextgen_output_abi_version() -> u32 {
    std::hint::black_box(concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV")));
    1
}

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frame {
    pub index: i64,
    pub timestamp: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NexusPlan {
    pub version: u32,
    pub ids: Vec<String>,
    pub frames: Vec<Frame>,
    /// f64le (ngen double) or f32le (resident GPU routing float).
    pub dtype: String,
    /// Nexus-major, then frame. No implicit sorting or transpose.
    pub data: PathBuf,
    pub output_directory: PathBuf,
}

impl NexusPlan {
    pub fn bytes(&self) -> Result<usize> {
        if self.version != 1 || self.ids.is_empty() || self.frames.is_empty() {
            return Err("NexusPlan v1 needs nonempty IDs and frames".into());
        }
        let mut names = HashSet::new();
        for id in &self.ids {
            if id.is_empty()
                || matches!(id.as_str(), "." | "..")
                || !id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                || !names.insert(id)
            {
                return Err(format!("unsafe or duplicate nexus ID {id:?}").into());
            }
        }
        for frame in &self.frames {
            // Timestamps are orchestration metadata, never numerical fields.
            if frame.timestamp.len() != 19
                || !frame.timestamp.bytes().enumerate().all(|(i, b)| match i {
                    4 | 7 => b == b'-',
                    10 => b == b' ',
                    13 | 16 => b == b':',
                    _ => b.is_ascii_digit(),
                })
            {
                return Err("nexus timestamp must be YYYY-MM-DD HH:MM:SS".into());
            }
        }
        let width = match self.dtype.as_str() {
            "f64le" => 8usize,
            "f32le" => 4,
            _ => return Err("unsupported nexus dtype".into()),
        };
        self.ids
            .len()
            .checked_mul(self.frames.len())
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| "nexus payload size overflow".into())
    }
}

/// C++ defaultfloat precision six, including exponent selection after rounding.
/// Nexus output has no header and uses comma-space separators and LF newlines.
pub fn defaultfloat(value: f64) -> String {
    if value.is_nan() {
        return if value.is_sign_negative() {
            "-nan"
        } else {
            "nan"
        }
        .into();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf"
        } else {
            "inf"
        }
        .into();
    }
    if value == 0. {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    let rounded = format!("{value:.5e}");
    let (mantissa, exponent) = rounded.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    if !(-4..6).contains(&exponent) {
        return format!(
            "{}e{exponent:+03}",
            mantissa.trim_end_matches('0').trim_end_matches('.')
        );
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let digits = mantissa.trim_start_matches('-').replace('.', "");
    let mut fixed = if exponent < 0 {
        format!("{sign}0.{}{}", "0".repeat((-exponent - 1) as usize), digits)
    } else {
        let point = (exponent + 1) as usize;
        format!("{sign}{}.{}", &digits[..point], &digits[point..])
    };
    while fixed.ends_with('0') {
        fixed.pop();
    }
    if fixed.ends_with('.') {
        fixed.pop();
    }
    fixed
}

pub fn write_nexus(plan: &NexusPlan, bytes: &[u8]) -> Result<serde_json::Value> {
    let expected = plan.bytes()?;
    if bytes.len() != expected {
        return Err("nexus payload must contain every field, no trailing bytes".into());
    }
    fs::create_dir_all(&plan.output_directory)?;
    let start = Instant::now();
    let width = if plan.dtype == "f64le" { 8 } else { 4 };
    let files = plan
        .ids
        .par_iter()
        .enumerate()
        .map(|(nexus, id)| -> Result<usize> {
            let target = plan.output_directory.join(format!("{id}_output.csv"));
            let partial = target.with_extension("csv.partial");
            let handle = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&partial)?;
            let result = (|| -> Result<usize> {
                let mut file = BufWriter::new(handle);
                for (time, frame) in plan.frames.iter().enumerate() {
                    let offset = (nexus * plan.frames.len() + time) * width;
                    let value = if width == 8 {
                        f64::from_le_bytes(bytes[offset..offset + 8].try_into()?)
                    } else {
                        f32::from_le_bytes(bytes[offset..offset + 4].try_into()?) as f64
                    };
                    writeln!(
                        file,
                        "{}, {}, {}",
                        frame.index,
                        frame.timestamp,
                        defaultfloat(value)
                    )?;
                }
                file.flush()?;
                file.get_ref().sync_all()?;
                let len = file.get_ref().metadata()?.len() as usize;
                fs::hard_link(&partial, &target)?;
                Ok(len)
            })();
            // Never erase an existing final file. Partial files belong to this call.
            let cleanup = fs::remove_file(&partial);
            let len = result?;
            cleanup?;
            Ok(len)
        })
        .collect::<Result<Vec<_>>>()?;
    crate::sync_directory(&plan.output_directory)?;
    Ok(
        serde_json::json!({"operation":"nexus-csv", "files":files.len(),
        "input_bytes":bytes.len(), "output_bytes":files.iter().sum::<usize>(),
        "format_and_fsync_ms":start.elapsed().as_secs_f64()*1000.,
        "parallelism":format!("{} Rayon workers, one nexus per task",rayon::current_num_threads()),
        "scope":"output boundary only; no routing-step work; per-file atomic publication"}),
    )
}

pub fn write_plan(path: &Path) -> Result<serde_json::Value> {
    let plan: NexusPlan = serde_json::from_slice(&fs::read(path)?)?;
    plan.bytes()?;
    let start = Instant::now();
    let bytes = fs::read(path.parent().unwrap_or(Path::new(".")).join(&plan.data))?;
    let read_ms = start.elapsed().as_secs_f64() * 1000.;
    let mut receipt = write_nexus(&plan, &bytes)?;
    receipt["raw_read_ms"] = read_ms.into();
    Ok(receipt)
}

/// # Safety
/// JSON and payload must remain readable for their declared lengths. The caller
/// completes its CUDA snapshot before entry. Error holds capacity writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rw_nextgen_nexus_write(
    json: *const u8,
    json_len: usize,
    payload: *const u8,
    payload_len: usize,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    let result = std::panic::catch_unwind(|| -> Result<()> {
        if json.is_null() || payload.is_null() {
            return Err("null snapshot pointer".into());
        }
        let plan: NexusPlan =
            serde_json::from_slice(unsafe { std::slice::from_raw_parts(json, json_len) })?;
        write_nexus(&plan, unsafe {
            std::slice::from_raw_parts(payload, payload_len)
        })?;
        Ok(())
    });
    let message = match result {
        Ok(Ok(())) => return 0,
        Ok(Err(e)) => e.to_string(),
        Err(_) => "nexus writer panic".into(),
    };
    if capacity > 0 && !error.is_null() {
        let n = message.len().min(capacity - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(message.as_ptr(), error.cast(), n);
            *error.add(n) = 0;
        }
    }
    1
}

/// Read the upstream nexus CSV as a formatting fixture, not a module capture.
pub fn capture_csv(path: &Path, directory: &Path) -> Result<()> {
    let mut frames = Vec::new();
    let mut bytes = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        let parts: Vec<_> = line.split(", ").collect();
        if parts.len() != 3 {
            return Err("expected upstream three-column nexus CSV".into());
        }
        frames.push(Frame {
            index: parts[0].parse()?,
            timestamp: parts[1].into(),
        });
        bytes.extend_from_slice(&parts[2].parse::<f64>()?.to_le_bytes());
    }
    let id = path
        .file_name()
        .and_then(|p| p.to_str())
        .and_then(|p| p.strip_suffix("_output.csv"))
        .ok_or("nexus filename")?;
    let plan = NexusPlan {
        version: 1,
        ids: vec![id.into()],
        frames,
        dtype: "f64le".into(),
        data: "flow.f64le".into(),
        output_directory: directory.join("rewritten"),
    };
    plan.bytes()?;
    fs::create_dir(directory)?;
    fs::write(directory.join("flow.f64le"), bytes)?;
    fs::write(
        directory.join("plan.json"),
        serde_json::to_vec_pretty(&plan)?,
    )?;
    Ok(())
}

/// Decode the referee's packed receive_data_entry observation, never its
/// rounded CSV. Each record is i64le index, 19 UTF-8 timestamp bytes, f64le flow.
/// The eight numerical bytes are copied unchanged into the replay payload.
pub fn capture_native_nexus(path: &Path, directory: &Path) -> Result<serde_json::Value> {
    let raw = fs::read(path)?;
    if raw.is_empty() || raw.len() % 35 != 0 {
        return Err("native nexus capture needs complete 35-byte records".into());
    }
    let id = path
        .file_name()
        .and_then(|p| p.to_str())
        .and_then(|p| p.strip_suffix("_output.raw"))
        .ok_or("native nexus capture filename must be ID_output.raw")?;
    let mut frames = Vec::with_capacity(raw.len() / 35);
    let mut bytes = Vec::with_capacity(raw.len() / 35 * 8);
    for record in raw.as_chunks::<35>().0 {
        frames.push(Frame {
            index: i64::from_le_bytes(record[..8].try_into()?),
            timestamp: std::str::from_utf8(&record[8..27])?.into(),
        });
        bytes.extend_from_slice(&record[27..35]);
    }
    let plan = NexusPlan {
        version: 1,
        ids: vec![id.into()],
        frames,
        dtype: "f64le".into(),
        data: "flow.f64le".into(),
        output_directory: directory.join("rewritten"),
    };
    plan.bytes()?;
    fs::create_dir(directory)?;
    fs::write(directory.join("flow.f64le"), bytes)?;
    fs::write(
        directory.join("plan.json"),
        serde_json::to_vec_pretty(&plan)?,
    )?;
    Ok(
        serde_json::json!({"operation":"capture-native-nexus", "frames":plan.frames.len(),
        "scope":"native receive_data_entry unrounded double bytes; no excluded records",
        "source":path}),
    )
}
