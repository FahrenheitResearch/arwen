//! Writer-side advisory locks for one rw-store run directory.
//!
//! This is the implementation of the concurrency contract in
//! `docs/FORMAT.md §7`: **at most one process writes a given
//! `<store_root>/<model>/<run>/` directory at a time.** A writer holds an
//! exclusive OS advisory file lock on `<run_dir>/.rw-lock` for the whole
//! critical section (grid validation + hour write + manifest update) and
//! releases it when the [`RunLock`] guard drops.
//!
//! Properties (all promised in FORMAT.md §7):
//!
//! - **Advisory only.** The lock coordinates cooperating *writers*; it does
//!   not stop a process that ignores it. Every shipped writer
//!   (`HourIngestWriter`, the rw-sat frame writer, the rw-sat window prune)
//!   takes it, which is what fixes the real two-process collision that
//!   motivated this work.
//! - **Auto-released on process death.** This is an OS lock (`flock(2)` on
//!   Unix, `LockFileEx` on Windows via `fs4`), *not* a pidfile: if the
//!   holder crashes or is killed, the kernel drops the lock. There is no
//!   stale-lock cleanup to get wrong.
//! - **The lock file persists.** `.rw-lock` is created on demand, stays
//!   zero-length, and is **never deleted** — deleting it would be racy
//!   (another process could be mid-`open`/`lock` on the same path while we
//!   unlink it, and the two would then lock different inodes). Its presence
//!   on disk is normal and is *not itself* the lock; the lock is the kernel
//!   record against the open handle.
//! - **Readers never lock.** Every file mutation in rw-store is an atomic
//!   temp+fsync+rename (`atomic.rs`), so a reader sees either the old file
//!   or the complete new one and needs no lock. Readers, validators, and
//!   listing tools MUST ignore `.rw-lock` (and `.*.tmp-*`).
//!
//! ### fs4 API note
//!
//! `fs4` 1.1 renamed the non-blocking exclusive lock to
//! [`FileExt::try_lock`] (mirroring std's `File::try_lock`, stabilised in
//! 1.89); it returns `Err(TryLockError::WouldBlock)` when the lock is held
//! elsewhere and `Err(TryLockError::Error(io))` on a real I/O failure. The
//! older `try_lock_exclusive() -> io::Result<bool>` surface is gone, so we
//! match on `TryLockError`.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

// In fs4 1.1 the `FileExt` trait lives at the crate root and is implemented
// for `std::fs::File` when the `sync` feature is on; bring it into scope for
// `try_lock`/`unlock`.
use fs4::{FileExt, TryLockError};

use crate::error::{RwResult, RwStoreError};

/// Name of the per-run-directory advisory lock file. It is created on demand,
/// stays zero-length, and is never deleted (see the module docs).
pub const LOCK_FILE_NAME: &str = ".rw-lock";

/// How long [`RunLock::acquire`] waits between non-blocking attempts.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// An exclusive advisory lock held on one run directory's `.rw-lock` file.
///
/// Construct with [`RunLock::try_acquire`] (non-blocking) or
/// [`RunLock::acquire`] (poll up to a timeout). The lock is released when the
/// guard is dropped (the OS also releases it if the process dies). Dropping
/// the guard does **not** delete the lock file.
#[derive(Debug)]
pub struct RunLock {
    /// The open handle whose kernel lock record *is* the lock. Held only to
    /// keep the lock alive; explicitly unlocked on drop for promptness.
    file: File,
    /// Path of the lock file, for diagnostics and the `Locked` message.
    path: PathBuf,
}

impl RunLock {
    /// Open-or-create `<run_dir>/.rw-lock` and try to take the exclusive
    /// advisory lock **without blocking**.
    ///
    /// Returns `Ok(Some(lock))` if we got it, `Ok(None)` if another writer
    /// currently holds it, and `Err` only on a real I/O failure (e.g. the
    /// run dir does not exist, or a non-`WouldBlock` lock error).
    pub fn try_acquire(run_dir: &Path) -> RwResult<Option<RunLock>> {
        let path = run_dir.join(LOCK_FILE_NAME);
        // read+write+create, explicitly NOT truncating: the lock needs an
        // openable handle, but truncating would race a concurrent writer's
        // handle and is pointless for a zero-length file. `truncate(false)`
        // is stated so the intent (and the clippy lint) are unambiguous.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        match FileExt::try_lock(&file) {
            Ok(()) => Ok(Some(RunLock { file, path })),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(err)) => Err(RwStoreError::Io(err)),
        }
    }

    /// Path of the `.rw-lock` file this guard holds.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Poll [`try_acquire`](RunLock::try_acquire) every 100 ms until the lock
    /// is taken or `timeout` elapses.
    ///
    /// On timeout returns [`RwStoreError::Locked`] with a message naming the
    /// lock path and how long we waited, so the operator knows *which* run
    /// dir is contended and that a competing writer is the likely cause.
    pub fn acquire(run_dir: &Path, timeout: Duration) -> RwResult<RunLock> {
        let started = Instant::now();
        loop {
            if let Some(lock) = RunLock::try_acquire(run_dir)? {
                return Ok(lock);
            }
            if started.elapsed() >= timeout {
                let path = run_dir.join(LOCK_FILE_NAME);
                return Err(RwStoreError::Locked(format!(
                    "{} held by another writer after waiting {:.1}s \
                     (a competing hour encode finishing is the normal case; \
                     check for another rw-store writer on this run dir)",
                    path.display(),
                    timeout.as_secs_f64(),
                )));
            }
            // Don't overshoot the deadline on the final nap.
            let remaining = timeout.saturating_sub(started.elapsed());
            sleep(POLL_INTERVAL.min(remaining));
        }
    }
}

impl Drop for RunLock {
    fn drop(&mut self) {
        // Best-effort prompt release. The OS would release it on close/exit
        // anyway, so a failure here is not actionable; we never delete the
        // lock file (racy — see module docs).
        let _ = FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rw-store-lock-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn second_acquire_fails_while_held() {
        let dir = test_dir("held");
        // fs4 file locks are per-handle on both Windows (LockFileEx) and Unix
        // (flock on a fresh fd), so two handles opened in *this same process*
        // contend exactly like two separate processes would — that's what
        // lets us exercise the contention path without forking.
        let first = RunLock::try_acquire(&dir).unwrap();
        assert!(first.is_some(), "first acquire takes the lock");

        let second = RunLock::try_acquire(&dir).unwrap();
        assert!(
            second.is_none(),
            "a second writer must see the lock held (Ok(None))"
        );

        drop(first);
        let retry = RunLock::try_acquire(&dir).unwrap();
        assert!(
            retry.is_some(),
            "once the first guard drops the lock is free again"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn acquire_times_out_with_locked_error() {
        let dir = test_dir("timeout");
        let _held = RunLock::try_acquire(&dir).unwrap().expect("hold the lock");

        let started = Instant::now();
        let result = RunLock::acquire(&dir, Duration::from_millis(250));
        let elapsed = started.elapsed();

        match result {
            Err(RwStoreError::Locked(msg)) => {
                assert!(
                    msg.contains(LOCK_FILE_NAME),
                    "the error should name the lock path: {msg}"
                );
            }
            other => panic!("expected Locked error, got {other:?}"),
        }
        assert!(
            elapsed >= Duration::from_millis(250),
            "acquire must poll for at least the full timeout, waited {elapsed:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lock_file_persists_after_release() {
        let dir = test_dir("persist");
        let path = dir.join(LOCK_FILE_NAME);
        {
            let lock = RunLock::try_acquire(&dir).unwrap().expect("acquire");
            assert!(path.is_file(), "lock file created on acquire");
            drop(lock);
        }
        assert!(
            path.is_file(),
            "lock file must persist after the guard drops (never deleted)"
        );
        let len = std::fs::metadata(&path).unwrap().len();
        assert_eq!(len, 0, "the lock file stays zero-length");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn try_acquire_errors_on_missing_run_dir() {
        // create(true) cannot make the lock file if its parent dir is absent;
        // that surfaces as an I/O error, not Ok(None).
        let dir = std::env::temp_dir().join(format!(
            "rw-store-lock-{}-absent/never-created",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        let result = RunLock::try_acquire(&dir);
        assert!(
            matches!(result, Err(RwStoreError::Io(_))),
            "a missing run dir is an I/O error, not a held lock: {result:?}"
        );
    }
}
