//! Multi-file time concatenation (Phase 8).
//!
//! Allows treating a sequence of wrfout files as a single continuous dataset.

use std::path::Path;

use crate::error::{WrfError, WrfResult};
use crate::file::WrfFile;

/// A sequence of WRF files concatenated in time.
pub struct MultiFile {
    files: Vec<WrfFile>,
    /// Cumulative time offsets: `cum_nt[i]` is the first global time index
    /// that belongs to `files[i]`.
    cum_nt: Vec<usize>,
    pub total_nt: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl MultiFile {
    /// Open a list of WRF files and concatenate along the Time dimension.
    /// Files must share the same spatial grid.
    pub fn open(paths: &[impl AsRef<Path>]) -> WrfResult<Self> {
        if paths.is_empty() {
            return Err(WrfError::InvalidParam("no files provided".into()));
        }
        let mut files = Vec::with_capacity(paths.len());
        let mut cum_nt = Vec::with_capacity(paths.len());
        let mut total = 0usize;

        for p in paths {
            let f = WrfFile::open(p)?;
            cum_nt.push(total);
            total += f.nt;
            files.push(f);
        }

        let nx = files[0].nx;
        let ny = files[0].ny;
        let nz = files[0].nz;

        // Validate all files share grid dimensions
        for f in &files[1..] {
            if f.nx != nx || f.ny != ny || f.nz != nz {
                return Err(WrfError::DimMismatch(format!(
                    "file {} has grid {}x{}x{}, expected {}x{}x{}",
                    f.path.display(),
                    f.nx,
                    f.ny,
                    f.nz,
                    nx,
                    ny,
                    nz
                )));
            }
        }

        Ok(Self {
            files,
            cum_nt,
            total_nt: total,
            nx,
            ny,
            nz,
        })
    }

    /// Resolve a global time index to a (file_index, local_time_index) pair.
    fn resolve_time(&self, global_t: usize) -> WrfResult<(usize, usize)> {
        if global_t >= self.total_nt {
            return Err(WrfError::InvalidParam(format!(
                "time index {global_t} out of range (total {0})",
                self.total_nt
            )));
        }
        // Find which file owns this time index
        let file_idx = match self.cum_nt.binary_search(&global_t) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let local_t = global_t - self.cum_nt[file_idx];
        Ok((file_idx, local_t))
    }

    /// Get the WrfFile and local time index for a global time index.
    pub fn file_at(&self, global_t: usize) -> WrfResult<(&WrfFile, usize)> {
        let (fi, lt) = self.resolve_time(global_t)?;
        Ok((&self.files[fi], lt))
    }

    /// Collect all time strings across files.
    pub fn all_times(&self) -> WrfResult<Vec<String>> {
        let mut out = Vec::with_capacity(self.total_nt);
        for f in &self.files {
            out.extend(f.times()?);
        }
        Ok(out)
    }
}
