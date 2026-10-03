//! GR2Analyst polling directory publisher, following the GRLevelX polling
//! conventions: a root directory holding `grlevel2.cfg` (the directory a
//! GRLevelX client's polling URL names) and one directory per site with its
//! file listing. The layout and defaults follow the GRLevelX-style polling
//! servers captured in the test corpus (`docs/testdata/corpus.md`: the
//! Iowa Environmental Mesonet's `config.cfg`, the North Dakota State Water
//! Commission's `dir.list` of KXWA, the Laredo feed's `grlevel2.cfg`):
//!
//! ```text
//! <root>/config.cfg         "ListFile: dir.list", then "Site: XXXX" lines, one per site
//! <root>/grlevel2.cfg       "Site: XXXX" lines
//! <root>/<SITE>/dir.list    "<size> <filename>" lines, oldest first
//! <root>/<SITE>/<SITE>YYYYMMDD_HHMMSS_V06.ar2v
//! ```
//!
//! Sizes are listed in bytes, which departs from the captured North Dakota
//! listing: its sizes are not bytes (40288 for a 20,616,906-byte file, about
//! its size in 512-byte blocks), and the GRLevelX manual names no unit.
//!
//! Lines end in LF; a site is added at the end of the site lists. File names are the
//! site and the volume time by a name format ([`DEFAULT_NAME_FORMAT`], the
//! NWS archive's `SITEYYYYMMDD_HHMMSS_V06`, unless
//! [`PollingDirectory::with_name_format`] sets another), then a suffix:
//! `.ar2v`, or `.ar2v.gz` when the bytes are gzip, unless
//! [`PollingDirectory::with_suffix`] fixes one;
//! [`PollingDirectory::publish_named`] takes any name
//! (`docs/level2/writer.md`, "Polling directory"). MetPy picks gzip by a
//! path's `.gz` suffix, so by default only gzip bytes are named `.gz`.
//! Every file is written to a temporary name and renamed
//! into place, so a client polling `dir.list` never sees a partial file or a
//! partial listing. One publisher per root is assumed.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use chrono::format::{Item, StrftimeItems};
use chrono::{DateTime, Datelike, Timelike, Utc};
use recast_radar_core::model::Volume;
use thiserror::Error;

use super::{WriteError, WriteOptions, WriteSummary};

/// Default number of files kept per site: 30, as the `recast-radar publish`
/// command and the Python package keep (two and a half hours of 5-minute
/// volumes). A server keeps what its clients loop over; the North Dakota
/// server captured in the corpus lists about 3.6 days.
pub const DEFAULT_MAX_FILES: usize = 30;
/// File name suffix of Archive II bytes that are not gzip.
pub const ARCHIVE_SUFFIX: &str = ".ar2v";
/// File name suffix of gzip-wrapped Archive II bytes.
pub const GZIP_SUFFIX: &str = ".ar2v.gz";
/// The site list files at the root.
pub const SITE_LIST_FILES: [&str; 2] = ["config.cfg", "grlevel2.cfg"];
/// The line that names the site listing file, first in a new `config.cfg`.
pub const LIST_FILE_LINE: &str = "ListFile: dir.list";
/// Default file name format (before the suffix): the NWS archive's name,
/// the site, the volume date, `_`, its time and `_V06` (the `AR2V0006`
/// format the writer writes), as in `KXWA20260921_105810_V06`.
pub const DEFAULT_NAME_FORMAT: &str = "{site}%Y%m%d_%H%M%S_V06";
/// Longest file name accepted.
const MAX_NAME_LEN: usize = 255;

/// One line of a `dir.list`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirListEntry {
    /// File size in bytes.
    pub size: u64,
    /// File name within the site directory.
    pub name: String,
}

/// Parse `dir.list` text: `<size> <filename>` lines (CRLF or LF); lines that
/// do not have that form are skipped.
pub fn parse_dir_list(text: &str) -> Vec<DirListEntry> {
    text.lines()
        .filter_map(|line| {
            let (size, name) = line.trim().split_once(char::is_whitespace)?;
            let name = name.trim();
            let size = size.parse().ok()?;
            (!name.is_empty()).then(|| DirListEntry {
                size,
                name: name.to_owned(),
            })
        })
        .collect()
}

/// `dir.list` text for `entries`, LF line ends.
pub fn format_dir_list(entries: &[DirListEntry]) -> String {
    entries
        .iter()
        .map(|entry| format!("{} {}\n", entry.size, entry.name))
        .collect()
}

/// A publish failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PublishError {
    /// Writing the volume failed (nothing was published).
    #[error(transparent)]
    Write(#[from] WriteError),
    /// The site identifier is not usable as a directory name.
    #[error(
        "site {0:?} must be 1 to 4 characters from [A-Za-z0-9_] and not a device name (CON, NUL, \
         COM1, ...)"
    )]
    InvalidSite(String),
    /// A file name format has an unknown `%` specifier or a path separator.
    #[error("file name format {format:?}: {reason}")]
    InvalidNameFormat {
        /// The format.
        format: String,
        /// What is wrong with it.
        reason: String,
    },
    /// A file name is not a plain name within the site directory that every
    /// platform can hold.
    #[error(
        "file name {0:?} must be 1 to 255 bytes without path separators, control characters or \
         any of <>:\"|?*, must not end in a dot or space or be a device name (CON, NUL, COM1, \
         ...), and must not be dir.list"
    )]
    InvalidFileName(String),
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The path.
        path: PathBuf,
        /// The error.
        #[source]
        source: io::Error,
    },
}

/// A published file.
#[derive(Clone, Debug, PartialEq)]
pub struct Published {
    /// Path of the new file.
    pub path: PathBuf,
    /// Its `dir.list` entry.
    pub entry: DirListEntry,
    /// Files removed by the retention limit.
    pub removed: Vec<String>,
    /// What was written, for [`PollingDirectory::publish_volume`].
    pub summary: Option<WriteSummary>,
}

/// A polling directory root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PollingDirectory {
    root: PathBuf,
    max_files: usize,
    /// A fixed suffix, or `None` to follow the bytes.
    suffix: Option<String>,
    /// File name format before the suffix (checked).
    name_format: String,
    /// Whether a publish lists its site in the root's site lists.
    list_sites: bool,
}

impl PollingDirectory {
    /// A publisher writing under `root` (created on first publish).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_files: DEFAULT_MAX_FILES,
            suffix: None,
            name_format: DEFAULT_NAME_FORMAT.to_owned(),
            list_sites: true,
        }
    }

    /// Name files by `format` (before the suffix): `{site}` stands for the
    /// site and chrono's strftime specifiers (`%Y`, `%m`, `%d`, `%H`, `%M`,
    /// `%S`, ...) for the volume time; anything else is literal. For
    /// example `"{site}%Y%m%d_%H%M%S_V06"` (the default),
    /// `"{site}_%Y%m%d%H%M%S"` or `"{site}%Y%m%d%H%M%S"` (with a lower-case
    /// site given as `WriteOptions::icao` for a lower-case name); names that
    /// follow no format go through [`Self::publish_named`]. `dir.list` and the retention
    /// limit order files by name, so the time should run from year to
    /// second. Refused when `format` has an unknown `%` specifier, a path
    /// separator or a control character, or renders a name that
    /// [`Self::publish_named`] refuses (`%H:%M` puts in a `:`, which Windows
    /// refuses).
    pub fn with_name_format(mut self, format: impl Into<String>) -> Result<Self, PublishError> {
        let format = format.into();
        let invalid = |reason: &str| PublishError::InvalidNameFormat {
            format: format.clone(),
            reason: reason.to_owned(),
        };
        if format.is_empty() {
            return Err(invalid("it is empty"));
        }
        if format.contains(['/', '\\']) || format.chars().any(char::is_control) {
            return Err(invalid(
                "a file name holds no path separator or control character",
            ));
        }
        if StrftimeItems::new(&format).any(|item| matches!(item, Item::Error)) {
            return Err(invalid("it has a % specifier chrono does not know"));
        }
        let previous = std::mem::replace(&mut self.name_format, format);
        let sample = self.file_name("KXXX", DateTime::<Utc>::UNIX_EPOCH, &[]);
        if check_file_name(&sample).is_err() {
            let format = std::mem::replace(&mut self.name_format, previous);
            return Err(PublishError::InvalidNameFormat {
                format,
                reason: format!(
                    "it renders {sample:?}, which is not a file name every platform can hold"
                ),
            });
        }
        Ok(self)
    }

    /// Whether each publish adds its site to the root's site lists
    /// (`config.cfg`, `grlevel2.cfg`; default `true`). With `false` the
    /// lists are left as they are, for a root whose lists are kept by hand.
    pub fn with_site_lists(mut self, list_sites: bool) -> Self {
        self.list_sites = list_sites;
        self
    }

    /// Keep at most `max_files` files per site (at least 1).
    pub fn with_max_files(mut self, max_files: usize) -> Self {
        self.max_files = max_files.max(1);
        self
    }

    /// Use `suffix` after the formatted name in every file name, whatever
    /// the bytes.
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = Some(suffix.into());
        self
    }

    /// The root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The file name of `bytes` of `site` at `time`: the name format
    /// ([`DEFAULT_NAME_FORMAT`] unless [`Self::with_name_format`] set one)
    /// and the suffix ([`ARCHIVE_SUFFIX`], or [`GZIP_SUFFIX`] for gzip bytes,
    /// unless a fixed suffix was set).
    pub fn file_name(&self, site: &str, time: DateTime<Utc>, bytes: &[u8]) -> String {
        let suffix = match &self.suffix {
            Some(suffix) => suffix.as_str(),
            None if bytes.starts_with(&[0x1f, 0x8b]) => GZIP_SUFFIX,
            None => ARCHIVE_SUFFIX,
        };
        let format = self.name_format.replace("{site}", site);
        let mut stem = String::new();
        let rendered = std::fmt::Write::write_fmt(
            &mut stem,
            format_args!("{}", time.format_with_items(StrftimeItems::new(&format))),
        );
        if rendered.is_err() {
            // Not reached: the format was checked when it was set, and a
            // site holds no `%`.
            stem = format!(
                "{site}{:04}{:02}{:02}_{:02}{:02}{:02}_V06",
                time.year(),
                time.month(),
                time.day(),
                time.hour(),
                time.minute(),
                time.second()
            );
        }
        format!("{stem}{suffix}")
    }

    /// Write `volume` (see [`super::write_volume`]) and publish it under its
    /// site identifier and volume time.
    pub fn publish_volume(
        &self,
        volume: &Volume,
        options: &WriteOptions,
    ) -> Result<Published, PublishError> {
        let (bytes, summary) =
            super::write_volume_with_source(volume, super::SourceMetadata::default(), options)?;
        let time = summary.volume_time.unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
        let mut published = self.publish_bytes(&summary.icao, time, &bytes)?;
        published.summary = Some(summary);
        Ok(published)
    }

    /// Publish Archive II `bytes` of `site` at `time`: write the file, add it
    /// to `dir.list`, drop files past the retention limit, and list the site
    /// in the root's site files.
    pub fn publish_bytes(
        &self,
        site: &str,
        time: DateTime<Utc>,
        bytes: &[u8],
    ) -> Result<Published, PublishError> {
        check_site(site)?;
        let name = self.file_name(site, time, bytes);
        self.publish_named(site, &name, bytes)
    }

    /// Publish `bytes` of `site` under the file name `name`, which the
    /// caller chooses, as [`Self::publish_bytes`] does otherwise. `dir.list` and the
    /// retention limit order files by name.
    pub fn publish_named(
        &self,
        site: &str,
        name: &str,
        bytes: &[u8],
    ) -> Result<Published, PublishError> {
        check_site(site)?;
        check_file_name(name)?;
        let name = name.to_owned();
        let dir = self.root.join(site);
        fs::create_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        let path = dir.join(&name);
        write_atomic(&path, bytes)?;

        let mut entries = self.entries(site)?;
        entries.retain(|entry| entry.name != name);
        let entry = DirListEntry {
            size: bytes.len() as u64,
            name: name.clone(),
        };
        entries.push(entry.clone());
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        let mut removed = Vec::new();
        while entries.len() > self.max_files {
            let old = entries.remove(0);
            let old_path = dir.join(&old.name);
            match fs::remove_file(&old_path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(source) => return Err(io_error(&old_path, source)),
            }
            removed.push(old.name);
        }
        write_atomic(&dir.join("dir.list"), format_dir_list(&entries).as_bytes())?;
        if self.list_sites {
            self.list_site(site)?;
        }
        Ok(Published {
            path,
            entry,
            removed,
            summary: None,
        })
    }

    /// The `dir.list` entries of `site` (empty when there is none yet).
    pub fn entries(&self, site: &str) -> Result<Vec<DirListEntry>, PublishError> {
        check_site(site)?;
        let path = self.root.join(site).join("dir.list");
        match fs::read_to_string(&path) {
            Ok(text) => Ok(parse_dir_list(&text)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(source) => Err(io_error(&path, source)),
        }
    }

    /// Add `site` to the root's site lists (`config.cfg`, `grlevel2.cfg`):
    /// a `Site:` line appended to each list that lacks it, the rest of the
    /// file kept as it is (a hand-kept list keeps its order, as the captured
    /// servers' lists are in the order their sites were added). A new
    /// `config.cfg` starts with [`LIST_FILE_LINE`].
    pub fn list_site(&self, site: &str) -> Result<(), PublishError> {
        check_site(site)?;
        fs::create_dir_all(&self.root).map_err(|source| io_error(&self.root, source))?;
        for file in SITE_LIST_FILES {
            let path = self.root.join(file);
            let mut text = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    if file == "config.cfg" {
                        format!("{LIST_FILE_LINE}\n")
                    } else {
                        String::new()
                    }
                }
                Err(source) => return Err(io_error(&path, source)),
            };
            let listed = text.lines().any(|line| {
                line.trim()
                    .strip_prefix("Site:")
                    .is_some_and(|name| name.trim() == site)
            });
            if listed {
                continue;
            }
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str("Site: ");
            text.push_str(site);
            text.push('\n');
            write_atomic(&path, text.as_bytes())?;
        }
        Ok(())
    }
}

fn check_site(site: &str) -> Result<(), PublishError> {
    let valid = !site.is_empty()
        && site.len() <= 4
        && site.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !reserved_device_name(site);
    if valid {
        Ok(())
    } else {
        Err(PublishError::InvalidSite(site.to_owned()))
    }
}

/// Characters Windows refuses in a file name, besides the path separators
/// and control characters (`:` would name an NTFS alternate data stream).
const WINDOWS_RESERVED_CHARS: [char; 7] = ['<', '>', ':', '"', '|', '?', '*'];

/// A name Windows takes for a device whatever its extension (`CON`,
/// `NUL.ar2v`, `com1`, ...), so a file of that name cannot be created.
fn reserved_device_name(name: &str) -> bool {
    let stem = name
        .split('.')
        .next()
        .unwrap_or(name)
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    if matches!(
        stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) {
        return true;
    }
    let mut chars = stem.chars();
    let prefix: String = chars.by_ref().take(3).collect();
    let digit = chars.next();
    matches!(prefix.as_str(), "COM" | "LPT")
        && digit.is_some_and(|c| c.is_ascii_digit() || matches!(c, '¹' | '²' | '³'))
        && chars.next().is_none()
}

/// A plain file name within a site directory that every platform can hold
/// (the directory may be served from Windows): not the listing or the
/// listing's temporary file, no path separator, control character or
/// character Windows refuses, no trailing dot or space, no device name.
fn check_file_name(name: &str) -> Result<(), PublishError> {
    let listing =
        name.eq_ignore_ascii_case("dir.list") || name.eq_ignore_ascii_case("dir.list.tmp");
    let valid = !name.is_empty()
        && name.len() <= MAX_NAME_LEN
        && name != "."
        && name != ".."
        && !listing
        && !name.contains(['/', '\\'])
        && !name.contains(WINDOWS_RESERVED_CHARS)
        && !name.chars().any(char::is_control)
        && !name.ends_with(['.', ' '])
        && !reserved_device_name(name);
    if valid {
        Ok(())
    } else {
        Err(PublishError::InvalidFileName(name.to_owned()))
    }
}

fn io_error(path: &Path, source: io::Error) -> PublishError {
    PublishError::Io {
        path: path.to_owned(),
        source,
    }
}

/// Write `bytes` to a temporary file beside `path`, then rename it over
/// `path`.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), PublishError> {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    fs::write(&temporary, bytes).map_err(|source| io_error(&temporary, source))?;
    fs::rename(&temporary, path).map_err(|source| io_error(path, source))
}
