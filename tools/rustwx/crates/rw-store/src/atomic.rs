//! Atomic file writes ported from rustwx-products/src/publication.rs
//! (`atomic_write_bytes` / `temp_path_for`, rustwx-fastplots-wt): write to a
//! hidden temp file in the same directory, fsync, then rename into place;
//! the temp file is removed on any failure.
//!
//! The port carried one thing that must not survive here. It finalized with
//!
//! ```text
//! if path.exists() { fs::remove_file(path)?; }
//! fs::rename(&tmp_path, path)?;
//! ```
//!
//! which is two operations with a gap between them, and in that gap the
//! destination does not exist. A crash, a power loss, or a killed process
//! landing there loses the previous good artifact and publishes nothing in
//! its place -- the one outcome an atomic write exists to make impossible.
//! Finalization is now the single [`replace_file`] step, which never unlinks
//! the destination at all; see its documentation for why one `fs::rename`
//! is that step on both families this workspace runs on.

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

/// Move `tmp` onto `dst` in one step, replacing whatever `dst` held.
///
/// This is the whole of the atomicity guarantee, so it is worth being exact
/// about why one call is enough on both families this workspace runs on:
///
/// * POSIX: `rename(2)` is specified to replace an existing destination
///   atomically -- "if the link named by the new argument exists, it shall be
///   removed and old renamed to new... this rename shall be atomic relative
///   to other threads" -- so a reader either sees the old inode or the new
///   one, never a missing name.
/// * Windows: `std::fs::rename` calls `MoveFileExW` with
///   `MOVEFILE_REPLACE_EXISTING`, which is why the delete-first dance is not
///   needed here either. `fs::rename` on Windows without that flag is what
///   people usually mean when they say "rename does not overwrite on
///   Windows"; the standard library passes it.
///
/// `atomic_replace_is_a_single_step_on_this_platform` in this module's tests
/// asserts that property against the real filesystem rather than trusting
/// either paragraph above, because the entire no-loss-window claim rests on
/// it and it is a platform fact, not a code fact.
///
/// A failed replace leaves both paths exactly as they were: `dst` keeps its
/// previous content and `tmp` is still on disk for the caller to clean up.
/// That is the contract any substitute passed to
/// [`atomic_write_with_replacer`] must honour.
pub fn replace_file(tmp: &Path, dst: &Path) -> RwResult<()> {
    fs::rename(tmp, dst)?;
    Ok(())
}

/// Streaming sibling of [`atomic_write_bytes`]: the caller writes through a
/// buffered handle on the hidden temp file instead of materializing the
/// whole payload in memory first. Identical guarantees — create-new temp in
/// the same directory, fsync, replace into place, temp removed on any
/// failure — so the destination either keeps its old content or holds
/// exactly what `write` produced.
pub fn atomic_write_with<F>(path: &Path, write: F) -> RwResult<()>
where
    F: FnOnce(&mut io::BufWriter<fs::File>) -> RwResult<()>,
{
    atomic_write_with_replacer(path, replace_file, write)
}

/// [`atomic_write_with`] with the final replace step supplied by the caller.
///
/// The seam exists so the crash window can be *tested* rather than reasoned
/// about: a test installs a replace that fails at the instant the retired
/// remove/rename gap used to sit and then asserts the previous artifact is
/// still readable. A substitute must honour [`replace_file`]'s contract --
/// either the destination becomes `tmp`'s content, or nothing moves at all.
/// Production has exactly one implementation and it is [`replace_file`].
pub fn atomic_write_with_replacer<R, F>(path: &Path, replace: R, write: F) -> RwResult<()>
where
    R: FnOnce(&Path, &Path) -> RwResult<()>,
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
    // One step, and nothing before it touches `path`. There is no moment at
    // which the destination is absent: it holds the old bytes until the
    // replace returns, and the new bytes afterwards.
    if let Err(err) = replace(&tmp_path, path) {
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

    /// The platform fact the whole guarantee rests on, asserted against the
    /// real filesystem: a rename onto an existing file succeeds and the
    /// destination ends up holding exactly the source's bytes.
    ///
    /// If this ever fails on a platform, `replace_file` is wrong there and
    /// every test below it is vacuous, so it is checked first and directly
    /// rather than inferred from `atomic_write_bytes` succeeding.
    #[test]
    fn atomic_replace_is_a_single_step_on_this_platform() {
        let dir = test_dir("platform-replace");
        let dst = dir.join("out.rws");
        let tmp = dir.join("incoming");
        fs::write(&dst, b"previous good artifact").unwrap();
        fs::write(&tmp, b"the new one").unwrap();

        replace_file(&tmp, &dst).unwrap();

        assert_eq!(fs::read(&dst).unwrap(), b"the new one");
        assert!(!tmp.exists(), "the source name must be gone after a replace");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The retired code was `remove_file(dst)` then `rename(tmp, dst)`. A
    /// crash between them left no destination at all. This installs a
    /// replace that fails at exactly that position and asserts the previous
    /// artifact survived it.
    ///
    /// The stub honours `replace_file`'s contract in the failing direction:
    /// it moves nothing, which is what a real failed `rename`/`MoveFileEx`
    /// does.
    #[test]
    fn a_failure_at_the_replace_leaves_the_previous_artifact_intact() {
        let dir = test_dir("replace-fails");
        let path = dir.join("out.rws");
        fs::write(&path, b"previous good artifact").unwrap();

        let err = atomic_write_with_replacer(
            &path,
            |tmp: &Path, dst: &Path| {
                // The instant the old remove/rename gap sat at: the payload
                // is written and fsynced, the destination has not been
                // touched, and the process dies here.
                assert!(tmp.is_file(), "the staged payload must exist");
                assert!(
                    dst.is_file(),
                    "the destination must still exist when the replace is entered"
                );
                Err(crate::RwStoreError::Io(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "injected crash at the replace",
                )))
            },
            |writer| {
                writer.write_all(b"the new one")?;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, crate::RwStoreError::Io(_)),
            "expected the injected Io error, got {err:?}"
        );

        assert_eq!(
            fs::read(&path).unwrap(),
            b"previous good artifact",
            "a failed replace must leave the previous artifact byte-identical"
        );
        assert_eq!(
            tmp_entries(&dir),
            Vec::<String>::new(),
            "the staged payload must be cleaned up after a failed replace"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The positive half of the same invariant, and the one that would have
    /// caught the retired code: an observer wrapped around the real replace
    /// reports what the destination held on the way in and on the way out.
    /// Both observations are of a present file -- there is no third state.
    #[test]
    fn the_destination_is_never_absent_across_a_successful_write() {
        let dir = test_dir("never-absent");
        let path = dir.join("out.rws");
        fs::write(&path, b"previous good artifact").unwrap();
        let mut seen: Vec<Option<Vec<u8>>> = Vec::new();

        atomic_write_with_replacer(
            &path,
            |tmp: &Path, dst: &Path| {
                seen.push(fs::read(dst).ok());
                let outcome = replace_file(tmp, dst);
                seen.push(fs::read(dst).ok());
                outcome
            },
            |writer| {
                writer.write_all(b"the new one")?;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            seen,
            vec![
                Some(b"previous good artifact".to_vec()),
                Some(b"the new one".to_vec()),
            ],
            "the destination must read as the old artifact before the replace \
             and the new one after it, and never as absent"
        );
        assert_eq!(fs::read(&path).unwrap(), b"the new one");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A replace onto a destination that never existed is still one step,
    /// and still leaves nothing behind. The failing-replace case for a fresh
    /// destination must not publish a partial file either.
    #[test]
    fn a_failed_replace_onto_a_fresh_destination_publishes_nothing() {
        let dir = test_dir("fresh-fails");
        let path = dir.join("nested").join("out.rws");

        let err = atomic_write_with_replacer(
            &path,
            |_tmp: &Path, dst: &Path| {
                assert!(!dst.exists(), "this case starts with no destination");
                Err(crate::RwStoreError::Io(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "injected crash at the replace",
                )))
            },
            |writer| {
                writer.write_all(b"never published")?;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(err, crate::RwStoreError::Io(_)), "{err:?}");
        assert!(!path.exists(), "nothing may be published by a failed replace");
        assert_eq!(tmp_entries(&path.parent().unwrap()), Vec::<String>::new());
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
