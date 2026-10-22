//! Shared argument checks for the observation front doors.
use std::error::Error;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rw_nexrad::s3::parse_time;

use crate::seam::seam_time;
use crate::{err, hex_sha256};

/// Parse the inclusive archive window, preserving the front doors' refusals.
pub fn window(
    start: Option<&str>,
    end: Option<&str>,
) -> Result<(DateTime<Utc>, DateTime<Utc>), Box<dyn Error>> {
    let start = parse_time(start.ok_or_else(|| err("--start is required"))?)?;
    let end = parse_time(end.ok_or_else(|| err("--end is required"))?)?;
    if end < start {
        return Err(err(format!(
            "--end {} precedes --start {}",
            seam_time(end),
            seam_time(start)
        )));
    }
    Ok((start, end))
}

/// Read one local archive object and hash its original bytes.
pub fn source_bytes(file: Option<&Path>) -> Result<(PathBuf, Vec<u8>, String), Box<dyn Error>> {
    let path = file.ok_or_else(|| err("--file is required (one archived object on disk)"))?;
    let raw =
        std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    let sha256 = hex_sha256(&raw);
    Ok((path.to_path_buf(), raw, sha256))
}

/// Require a file destination for a decoded field or geometry pack.
pub fn out_pack_path(out: Option<&Path>) -> Result<PathBuf, Box<dyn Error>> {
    let out = out.ok_or_else(|| err("--out is required (the pack destination)"))?;
    if out.is_dir() {
        return Err(err(format!(
            "--out {} is a directory; give the pack file path",
            out.display()
        )));
    }
    Ok(out.to_path_buf())
}

/// Run an observation front door with its existing stdout/stderr contract.
pub fn main(
    program: &str,
    run: impl FnOnce(&[String]) -> Result<String, Box<dyn Error>>,
) -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(output) => {
            print!("{output}");
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{program}: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Read a value-bearing option using the caller's existing argument cursor.
pub fn value(args: &[String], index: &mut usize, flag: &str) -> Result<String, Box<dyn Error>> {
    *index += 1;
    args.get(*index).cloned().ok_or_else(|| err(format!("{flag} needs a value")))
}

/// Handle the common informational commands before parsing a stream's options.
pub fn information(args: &[String], program: &str, version: &str, usage: &str, abi: &str) -> Option<String> {
    match args.first().map(String::as_str) {
        None | Some("--help" | "-h" | "help") => Some(usage.to_string()),
        Some("--version" | "-V") => Some(format!("{program} {version}\n")),
        Some("--abi") => Some(format!("{abi}\n")),
        _ => None,
    }
}
