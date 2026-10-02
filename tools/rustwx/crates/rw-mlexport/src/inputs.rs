//! Where frames come from: history files, gzip-compressed history files,
//! and ZIPs (the run page's download) whose `wrfout_dNN_*` members are
//! history files.  A compressed or archived frame is unpacked into the
//! export's scratch folder just before it is read and deleted just after,
//! so the disk holds one frame at a time.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{fail, refuse, Result};
use crate::times;
use crate::zipin::{Archive, ZipEntry};

#[derive(Clone)]
pub enum Source {
    File(PathBuf),
    Gz(PathBuf),
    Zip { archive: Arc<Archive>, entry: ZipEntry },
}

/// One input that holds one or more frames.
#[derive(Clone)]
pub struct Candidate {
    pub source: Source,
    /// The history file's base name (without `.gz`).
    pub name: String,
    /// Domain and valid time read off the name, when it is a standard
    /// history-file name.
    pub hint: Option<(String, i64)>,
}

impl Candidate {
    pub fn describe(&self) -> String {
        match &self.source {
            Source::File(p) | Source::Gz(p) => p.display().to_string(),
            Source::Zip { archive, entry } => format!("{} in {}", entry.name, archive.path().display()),
        }
    }
}

/// `dNN` and the valid time from a history-file name such as
/// `wrfout_d02_2026-09-29_08:00:00` (or `_08_00_00`, `.nc`, `.nc.gz`).
pub fn parse_history_name(name: &str) -> Option<(String, i64)> {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let base = base.strip_suffix(".gz").unwrap_or(base);
    let base = base.strip_suffix(".nc").unwrap_or(base);
    let bytes = base.as_bytes();
    for at in 0..bytes.len().saturating_sub(4) {
        if bytes[at] == b'd'
            && (at == 0 || bytes[at - 1] == b'_')
            && bytes[at + 1].is_ascii_digit()
            && bytes[at + 2].is_ascii_digit()
            && bytes.get(at + 3) == Some(&b'_')
        {
            let rest = &base[at + 4..];
            if rest.len() >= 19 {
                if let Some(t) = times::parse(&rest[..19]) {
                    return Some((base[at..at + 3].to_string(), t));
                }
            }
        }
    }
    None
}

/// A history file's name as a folder or a ZIP is searched for it:
/// `wrfout_dNN_YYYY-MM-DD_HH:MM:SS`, optionally `.nc` and `.gz`.  Only that
/// prefix: a run folder also holds restart checkpoints
/// (`gpuwmrst_dNN_<time>__<id>.npz`) whose names carry a domain and a time
/// too, and reading one as a history file would refuse the whole folder.
fn is_history_name(name: &str) -> bool {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let stem = base.strip_suffix(".gz").unwrap_or(base);
    let stem = stem.strip_suffix(".nc").unwrap_or(stem);
    stem.starts_with("wrfout_d")
        && stem.len() == "wrfout_dNN_YYYY-MM-DD_HH:MM:SS".len()
        && parse_history_name(stem).is_some()
}

/// Expand the request's inputs into candidates.  A folder is searched (one
/// level and below) for history-file names; a ZIP contributes its members
/// with history-file names; anything else is taken as a history file.
pub fn discover(inputs: &[PathBuf]) -> Result<Vec<Candidate>> {
    let mut found = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut stack = vec![input.clone()];
            let mut files = Vec::new();
            while let Some(dir) = stack.pop() {
                for entry in std::fs::read_dir(&dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if entry.file_type()?.is_dir() {
                        stack.push(path);
                    } else if is_history_name(&entry.file_name().to_string_lossy()) {
                        files.push(path);
                    }
                }
            }
            files.sort();
            if files.is_empty() {
                return Err(refuse(format!(
                    "{} holds no history files (wrfout_dNN_YYYY-MM-DD_HH:MM:SS), so there is nothing to export from it",
                    input.display()
                )));
            }
            found.extend(discover(&files)?);
            continue;
        }
        if !input.is_file() {
            return Err(refuse(format!(
                "{} does not exist, so it cannot be exported",
                input.display()
            )));
        }
        let name = input
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".zip") {
            let archive = Arc::new(Archive::open(input)?);
            let mut members: Vec<ZipEntry> = archive
                .entries()
                .iter()
                .filter(|e| !e.name.ends_with('/') && is_history_name(&e.name))
                .cloned()
                .collect();
            members.sort_by(|a, b| a.name.cmp(&b.name));
            if members.is_empty() {
                return Err(refuse(format!(
                    "{} holds no history files (wrfout_dNN_YYYY-MM-DD_HH:MM:SS members), so there is nothing to export from it",
                    input.display()
                )));
            }
            for entry in members {
                let member = entry.name.rsplit('/').next().unwrap_or(&entry.name).to_string();
                let member = member.strip_suffix(".gz").unwrap_or(&member).to_string();
                if entry.name.ends_with(".gz") {
                    return Err(refuse(format!(
                        "{} in {} is a gzip file inside a ZIP; unpack the ZIP first, because nesting two archives would double the scratch the export needs",
                        entry.name,
                        input.display()
                    )));
                }
                found.push(Candidate {
                    hint: parse_history_name(&member),
                    name: member,
                    source: Source::Zip { archive: Arc::clone(&archive), entry },
                });
            }
        } else if lower.ends_with(".gz") {
            let stripped = name[..name.len() - 3].to_string();
            found.push(Candidate {
                hint: parse_history_name(&stripped),
                name: stripped,
                source: Source::Gz(input.clone()),
            });
        } else {
            found.push(Candidate {
                hint: parse_history_name(&name),
                name,
                source: Source::File(input.clone()),
            });
        }
    }
    Ok(found)
}

/// A history file on disk for the duration of one read; an unpacked copy
/// is deleted when this is dropped.
pub struct OnDisk {
    pub path: PathBuf,
    temporary: bool,
}

impl Drop for OnDisk {
    fn drop(&mut self) {
        if self.temporary {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// The name an unpacked history file takes in `scratch`.  WRF spells
/// history files `..._HH:MM:SS`, and a ZIP made on Linux (the run page's
/// download) carries that spelling in its member names; a colon is not a
/// legal Windows file-name character, so unpacking such a member failed on
/// Windows with os error 123.  The copy takes the `HH_MM_SS` spelling, which
/// names the same domain and time to [`parse_history_name`].
fn scratch_name(name: &str) -> String {
    name.replace(':', "_")
}

/// Put `candidate` on disk, unpacking it into `scratch` when it is
/// compressed or archived.
pub fn materialize(candidate: &Candidate, scratch: &Path) -> Result<OnDisk> {
    match &candidate.source {
        Source::File(path) => Ok(OnDisk { path: path.clone(), temporary: false }),
        Source::Gz(path) => {
            std::fs::create_dir_all(scratch)?;
            let dest = scratch.join(scratch_name(&candidate.name));
            let guard = OnDisk { path: dest.clone(), temporary: true };
            let input = File::open(path).map_err(|e| fail(format!("could not open {}: {e}", path.display())))?;
            let mut decoder = flate2::read::MultiGzDecoder::new(BufReader::with_capacity(8 << 20, input));
            let mut out = BufWriter::with_capacity(8 << 20, File::create(&dest)?);
            std::io::copy(&mut decoder, &mut out).map_err(|e| {
                refuse(format!(
                    "{} is not a readable gzip file ({e}), so the history file inside it cannot be read",
                    path.display()
                ))
            })?;
            out.flush()?;
            Ok(guard)
        }
        Source::Zip { archive, entry } => {
            std::fs::create_dir_all(scratch)?;
            let dest = scratch.join(scratch_name(&candidate.name));
            let guard = OnDisk { path: dest.clone(), temporary: true };
            let mut out = BufWriter::with_capacity(8 << 20, File::create(&dest)?);
            archive.extract(entry, &mut out)?;
            out.flush()?;
            Ok(guard)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_names_give_domain_and_time() {
        let t = times::parse("2026-09-29_08:00:00").unwrap();
        assert_eq!(parse_history_name("wrfout_d02_2026-09-29_08:00:00"), Some(("d02".into(), t)));
        assert_eq!(parse_history_name("run/wrfout/wrfout_d01_2026-09-29_08_00_00.nc.gz"), Some(("d01".into(), t)));
        assert_eq!(parse_history_name("wrfout_d01_2026-09-29_08:00:00.json"), Some(("d01".into(), t)));
        assert!(!is_history_name("wrfout_d01_2026-09-29_08:00:00.json"));
        assert!(is_history_name("wrfout_d01_2026-09-29_08:00:00"));
        assert!(is_history_name("run/wrfout/wrfout_d02_2026-09-29_08_00_00.nc.gz"));
        // A run folder's restart checkpoints are not history files.
        assert!(!is_history_name("gpuwmrst_d01_2026-03-15_01_00_00__646422502e2b4f1aa75fb62b18b11bf6.npz"));
        assert_eq!(parse_history_name("namelist.input"), None);
        assert_eq!(parse_history_name("wrfinput_d01"), None);
    }
}
