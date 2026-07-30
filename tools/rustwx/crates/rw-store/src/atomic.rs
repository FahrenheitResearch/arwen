//! Atomic file writes ported from rustwx-products/src/publication.rs
//! (`atomic_write_bytes` / `temp_path_for`, rustwx-fastplots-wt): write to a
//! hidden temp file in the same directory, fsync, then rename into place;
//! the temp file is removed on any failure.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::RwResult;

/// Temp-file name in the same directory as `path`: `.{file_name}.tmp-{pid}-{seq}`.
/// The original used a millisecond timestamp for the last component; a process
/// counter gives the same same-directory/same-volume rename guarantee while
/// staying unique under rapid successive calls within one process.
fn temp_path_for(path: &Path) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!(
        ".{file_name}.tmp-{}-{}",
        process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Write `bytes` to `path` atomically: the destination either keeps its old
/// content or holds exactly `bytes`, never a partial write. Parent
/// directories are created as needed.
pub fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> RwResult<()> {
    atomic_write_with(path, |writer| {
        writer.write_all(bytes)?;
        Ok(())
    })
}

/// Streaming sibling of [`atomic_write_bytes`]: the caller writes through a
/// buffered handle on the hidden temp file instead of materializing the
/// whole payload in memory first. Identical guarantees — create-new temp in
/// the same directory, fsync, rename into place, temp removed on any
/// failure — so the destination either keeps its old content or holds
/// exactly what `write` produced.
pub fn atomic_write_with<F>(path: &Path, write: F) -> RwResult<()>
where
    F: FnOnce(&mut io::BufWriter<fs::File>) -> RwResult<()>,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = temp_path_for(path);
    let write_result = (|| -> RwResult<()> {
        let file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)?;
        let mut writer = io::BufWriter::with_capacity(1 << 20, file);
        write(&mut writer)?;
        writer.flush()?;
        writer
            .into_inner()
            .map_err(|err| err.into_error())?
            .sync_all()?;
        Ok(())
    })();
    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    let finalize_result = (|| -> RwResult<()> {
        if path.exists() {
            fs::remove_file(path)?;
        }
        fs::rename(&tmp_path, path)?;
        Ok(())
    })();
    if let Err(err) = finalize_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rw-store-atomic-{}-{}", process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tmp_entries(dir: &Path) -> Vec<String> {
        fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp"))
            .collect()
    }

    #[test]
    fn writes_new_file_with_exact_content() {
        let dir = test_dir("new-file");
        let path = dir.join("nested").join("out.rws");
        atomic_write_bytes(&path, b"hello rw-store").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"hello rw-store");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn overwrites_existing_file() {
        let dir = test_dir("overwrite");
        let path = dir.join("out.rws");
        fs::write(&path, b"old content that is longer").unwrap();
        atomic_write_bytes(&path, b"new").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn leaves_no_temp_files_after_success() {
        let dir = test_dir("no-temp-success");
        let path = dir.join("out.rws");
        atomic_write_bytes(&path, b"first").unwrap();
        atomic_write_bytes(&path, b"second").unwrap();
        assert_eq!(
            tmp_entries(&dir),
            Vec::<String>::new(),
            "no .tmp files should remain after successful writes"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cleans_up_temp_file_when_target_is_a_directory() {
        let dir = test_dir("target-is-dir");
        let path = dir.join("out.rws");
        // Make the destination un-replaceable: a directory cannot be
        // remove_file'd or renamed over, so finalize must fail.
        fs::create_dir_all(&path).unwrap();
        let err = atomic_write_bytes(&path, b"doomed").unwrap_err();
        assert!(
            matches!(err, crate::RwStoreError::Io(_)),
            "expected Io error, got {err:?}"
        );
        assert!(path.is_dir(), "destination directory must be untouched");
        assert_eq!(
            tmp_entries(&dir),
            Vec::<String>::new(),
            "temp file must be cleaned up after failure"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
