//! Rolling time-window management: cap a followed sector's footprint by
//! frame age and total bytes, deleting the oldest frame files and keeping
//! the run manifests truthful (empty run dirs are removed entirely,
//! including their `grid.rwg`).

use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};

use rw_store::lock::RunLock;
use rw_store::run::RwsRunManifest;

use crate::store::{frame_time, run_day};

/// Per-followed-sector retention policy. `None` disables that limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WindowConfig {
    pub max_age_minutes: Option<u32>,
    pub max_bytes: Option<u64>,
}

impl WindowConfig {
    pub fn is_unbounded(&self) -> bool {
        self.max_age_minutes.is_none() && self.max_bytes.is_none()
    }
}

/// What one eviction pass removed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvictionReport {
    pub removed_frames: usize,
    pub removed_bytes: u64,
    /// Run dirs that became empty and were deleted.
    pub removed_runs: Vec<String>,
    /// Run dirs that would have been pruned this pass but were skipped
    /// because another writer held the run-dir lock (FORMAT.md §7). The next
    /// cycle retries them. Sorted, like `removed_runs`.
    pub skipped_locked: Vec<String>,
}

#[derive(Debug)]
struct FrameRef {
    run_name: String,
    hhmm: u16,
    time: DateTime<Utc>,
    path: PathBuf,
    bytes: u64,
}

/// Enforce the window over every run dir of `model` whose name starts with
/// `run_prefix` (e.g. `conus_c13` selects one followed band; an empty
/// prefix selects the whole satellite). `now` is supplied by the caller so
/// tests and replays stay deterministic.
pub fn enforce_window(
    store_root: &Path,
    model: &str,
    run_prefix: &str,
    now: DateTime<Utc>,
    config: &WindowConfig,
) -> Result<EvictionReport, Box<dyn Error>> {
    let mut report = EvictionReport::default();
    if config.is_unbounded() {
        return Ok(report);
    }
    let model_dir = store_root.join(model);
    if !model_dir.is_dir() {
        return Ok(report);
    }

    // Inventory every frame across the matching run dirs, via run.json (the
    // manifests are the source of truth; orphan files are left alone).
    let mut frames: Vec<FrameRef> = Vec::new();
    let mut run_names: Vec<String> = Vec::new();
    for entry in fs::read_dir(&model_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let run_name = entry.file_name().to_string_lossy().to_string();
        if !run_name.starts_with(run_prefix) || run_day(&run_name).is_none() {
            continue;
        }
        let manifest_path = entry.path().join("run.json");
        if !manifest_path.is_file() {
            continue;
        }
        let manifest: RwsRunManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        run_names.push(run_name.clone());
        for (&hhmm, hour) in &manifest.hours {
            let Some(time) = frame_time(&run_name, hhmm) else {
                continue;
            };
            let path = entry.path().join(&hour.file);
            let bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            frames.push(FrameRef {
                run_name: run_name.clone(),
                hhmm,
                time,
                path,
                bytes,
            });
        }
    }
    frames.sort_by_key(|frame| frame.time);

    // Decide the evict set: everything past max_age, then oldest-first
    // until the remainder fits max_bytes.
    let mut evict = vec![false; frames.len()];
    if let Some(minutes) = config.max_age_minutes {
        let cutoff = now - Duration::minutes(i64::from(minutes));
        for (index, frame) in frames.iter().enumerate() {
            if frame.time < cutoff {
                evict[index] = true;
            }
        }
    }
    if let Some(max_bytes) = config.max_bytes {
        let mut kept_bytes: u64 = frames
            .iter()
            .enumerate()
            .filter(|(index, _)| !evict[*index])
            .map(|(_, frame)| frame.bytes)
            .sum();
        for (index, frame) in frames.iter().enumerate() {
            if kept_bytes <= max_bytes {
                break;
            }
            if !evict[index] {
                evict[index] = true;
                kept_bytes -= frame.bytes;
            }
        }
    }

    // Apply: delete files, prune manifests, drop empty run dirs. The prune
    // is a writer (it deletes f*.rws / run.json / grid.rwg), so each run dir
    // is mutated only while we hold its advisory lock (FORMAT.md §7). If a
    // frame writer (ours or another process's) holds the lock, skip this dir
    // this pass and let the next cycle retry — this is exactly the fix for
    // the two-process rolling-window collision that motivated the lock.
    for run_name in &run_names {
        let run_dir = model_dir.join(run_name);
        let _lock = match RunLock::try_acquire(&run_dir)? {
            Some(lock) => lock,
            None => {
                eprintln!(
                    "window prune: skipping {} this pass (run dir locked by another writer); \
                     will retry next cycle",
                    run_dir.display()
                );
                report.skipped_locked.push(run_name.clone());
                continue;
            }
        };
        let manifest_path = run_dir.join("run.json");
        let mut manifest: RwsRunManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        let mut changed = false;
        for (index, frame) in frames.iter().enumerate() {
            if !evict[index] || frame.run_name != *run_name {
                continue;
            }
            if frame.path.is_file() {
                fs::remove_file(&frame.path)?;
            }
            manifest.hours.remove(&frame.hhmm);
            report.removed_frames += 1;
            report.removed_bytes += frame.bytes;
            changed = true;
        }
        if !changed {
            continue;
        }
        if manifest.hours.is_empty() {
            fs::remove_file(&manifest_path)?;
            let grid_path = run_dir.join("grid.rwg");
            if grid_path.is_file() {
                fs::remove_file(&grid_path)?;
            }
            // The store content is gone; all that can remain is the `.rw-lock`
            // file. Tearing down the whole run dir is the one place that file
            // is removed (FORMAT.md §7's documented exception) — every other
            // path leaves it in place. Release the OS lock and close our
            // handle FIRST (drop the guard), so deleting the file is clean on
            // Windows (you cannot unlink a file you still hold locked-open).
            // A drop+delete window lets another writer re-acquire and re-open
            // the run dir; per §7's Windows pruning caveat we tolerate that —
            // if the lock file or the dir can no longer be removed, leave them
            // and record a skip so the next cycle retries instead of erroring.
            let lock_path = run_dir.join(rw_store::LOCK_FILE_NAME);
            drop(_lock);
            let removed = remove_emptied_run_dir(&run_dir, &lock_path)?;
            if removed {
                report.removed_runs.push(run_name.clone());
            } else {
                report.skipped_locked.push(run_name.clone());
            }
        } else {
            manifest.save(&manifest_path)?;
        }
    }
    report.removed_runs.sort();
    report.skipped_locked.sort();
    Ok(report)
}

/// Remove the `.rw-lock` file and then the (otherwise-empty) run dir, after
/// the caller has already dropped its lock guard and deleted all store
/// content. Returns `Ok(true)` when the dir was removed.
///
/// Both removals are best-effort against a writer that raced back in through
/// the drop+delete window: a failure to delete the lock file (it was re-
/// opened/locked) or a non-empty dir means someone is using it again, so we
/// return `Ok(false)` and let the next prune cycle retry, rather than
/// erroring the whole pass (FORMAT.md §7 Windows pruning caveat). Genuine I/O
/// errors unrelated to contention still propagate.
fn remove_emptied_run_dir(run_dir: &Path, lock_path: &Path) -> Result<bool, Box<dyn Error>> {
    if lock_path.is_file() {
        match fs::remove_file(lock_path) {
            Ok(()) => {}
            // A racing writer re-opened/locked the file: don't tear down.
            Err(err) if is_contention(&err) => return Ok(false),
            Err(err) => return Err(err.into()),
        }
    }
    match fs::remove_dir(run_dir) {
        Ok(()) => Ok(true),
        // Re-created lock file (or other fresh content) means the dir is no
        // longer empty: a writer is back. Retry next cycle.
        Err(err) if err.kind() == io::ErrorKind::DirectoryNotEmpty || is_contention(&err) => {
            Ok(false)
        }
        Err(err) => Err(err.into()),
    }
}

/// Whether an I/O error reflects another process holding/using the file
/// rather than a hard failure — on Windows a still-open locked handle yields
/// `PermissionDenied` (sharing violation); we treat that (and `WouldBlock`)
/// as "retry next cycle".
fn is_contention(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::PermissionDenied | io::ErrorKind::WouldBlock
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::test_support::{scan_start, synthetic_field};
    use crate::store::write_band_frame;
    use chrono::TimeZone;
    use std::path::PathBuf;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-sat-window-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_frames(dir: &Path, minutes: &[(u32, u32)]) -> Vec<crate::store::WrittenFrame> {
        minutes
            .iter()
            .map(|&(hour, minute)| {
                let field = synthetic_field(12, 10, scan_start(hour, minute), 13, 0.0);
                write_band_frame(dir, &field, 1).unwrap()
            })
            .collect()
    }

    fn now_at(hour: u32, minute: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 6, 10, hour, minute, 0).unwrap()
    }

    #[test]
    fn unbounded_config_is_a_no_op() {
        let dir = test_dir("noop");
        write_frames(&dir, &[(18, 51)]);
        let report = enforce_window(
            &dir,
            "g19",
            "conus_c13",
            now_at(23, 0),
            &WindowConfig::default(),
        )
        .unwrap();
        assert_eq!(report, EvictionReport::default());
        assert!(dir.join("g19/conus_c13_20260610/t1851.rws").is_file());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn max_age_evicts_old_frames_and_keeps_manifest_truthful() {
        let dir = test_dir("age");
        write_frames(&dir, &[(18, 41), (18, 46), (18, 51), (18, 56)]);
        let report = enforce_window(
            &dir,
            "g19",
            "conus_c13",
            now_at(19, 0),
            &WindowConfig {
                max_age_minutes: Some(12),
                max_bytes: None,
            },
        )
        .unwrap();
        // Cutoff 18:48 — 18:41 and 18:46 go, 18:51 and 18:56 stay.
        assert_eq!(report.removed_frames, 2);
        assert!(report.removed_bytes > 0);
        assert!(report.removed_runs.is_empty());
        let run_dir = dir.join("g19/conus_c13_20260610");
        assert!(!run_dir.join("t1841.rws").exists());
        assert!(!run_dir.join("t1846.rws").exists());
        assert!(run_dir.join("t1851.rws").is_file());
        assert!(run_dir.join("t1856.rws").is_file());
        let manifest: RwsRunManifest =
            serde_json::from_slice(&fs::read(run_dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(
            manifest.hours.keys().copied().collect::<Vec<u16>>(),
            vec![1851, 1856]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn max_bytes_evicts_oldest_first() {
        let dir = test_dir("bytes");
        let written = write_frames(&dir, &[(18, 41), (18, 46), (18, 51)]);
        let per_frame = written[0].bytes;
        // Budget for exactly two frames.
        let report = enforce_window(
            &dir,
            "g19",
            "conus_c13",
            now_at(19, 0),
            &WindowConfig {
                max_age_minutes: None,
                max_bytes: Some(per_frame * 2),
            },
        )
        .unwrap();
        assert_eq!(report.removed_frames, 1);
        let run_dir = dir.join("g19/conus_c13_20260610");
        assert!(!run_dir.join("t1841.rws").exists(), "oldest goes first");
        assert!(run_dir.join("t1846.rws").is_file());
        assert!(run_dir.join("t1851.rws").is_file());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fully_evicted_run_dir_is_removed_with_its_grid() {
        let dir = test_dir("drop-run");
        write_frames(&dir, &[(10, 0), (18, 56)]);
        // Split the frames across two run dirs by moving the grid.
        let moved = synthetic_field(12, 10, scan_start(11, 0), 13, 0.004);
        write_band_frame(&dir, &moved, 1).unwrap();

        let report = enforce_window(
            &dir,
            "g19",
            "conus_c13",
            now_at(19, 0),
            &WindowConfig {
                max_age_minutes: Some(60),
                max_bytes: None,
            },
        )
        .unwrap();
        // 10:00 (base run) and 11:00 (moved run) are stale; 18:56 stays.
        assert_eq!(report.removed_frames, 2);
        assert_eq!(report.removed_runs, vec!["conus_c13_20260610_2"]);
        assert!(
            !dir.join("g19/conus_c13_20260610_2").exists(),
            "empty run dir removed entirely (grid.rwg included)"
        );
        let kept = dir.join("g19/conus_c13_20260610");
        assert!(kept.join("t1856.rws").is_file());
        assert!(kept.join("grid.rwg").is_file());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_prefix_scopes_eviction_to_one_band() {
        let dir = test_dir("prefix");
        write_frames(&dir, &[(10, 0)]);
        let other_band = synthetic_field(12, 10, scan_start(10, 0), 8, 0.0);
        write_band_frame(&dir, &other_band, 1).unwrap();

        let report = enforce_window(
            &dir,
            "g19",
            "conus_c13",
            now_at(19, 0),
            &WindowConfig {
                max_age_minutes: Some(60),
                max_bytes: None,
            },
        )
        .unwrap();
        assert_eq!(report.removed_frames, 1);
        assert!(
            dir.join("g19/conus_c08_20260610/t1000.rws").is_file(),
            "other band untouched"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn locked_run_dir_is_skipped_then_pruned_after_release() {
        let dir = test_dir("locked-skip");
        // Two stale frames in one run dir; both would be evicted by max_age.
        write_frames(&dir, &[(18, 41), (18, 46)]);
        let run_dir = dir.join("g19/conus_c13_20260610");

        let config = WindowConfig {
            max_age_minutes: Some(5),
            max_bytes: None,
        };

        // Hold the run-dir lock as a competing writer would (fs4 locks are
        // per-handle, so this in-process guard contends exactly like another
        // process). enforce_window must NOT mutate the dir this pass.
        let held = RunLock::try_acquire(&run_dir)
            .unwrap()
            .expect("take the run-dir lock");

        let skipped = enforce_window(&dir, "g19", "conus_c13", now_at(19, 0), &config).unwrap();
        assert_eq!(
            skipped.removed_frames, 0,
            "a locked run dir must not be pruned this pass"
        );
        assert_eq!(
            skipped.skipped_locked,
            vec!["conus_c13_20260610".to_string()]
        );
        assert!(
            run_dir.join("t1841.rws").is_file() && run_dir.join("t1846.rws").is_file(),
            "frames survive while the dir is locked"
        );
        let manifest: RwsRunManifest =
            serde_json::from_slice(&fs::read(run_dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(manifest.hours.len(), 2, "manifest untouched while locked");

        // Release the lock; the next pass prunes normally.
        drop(held);
        let pruned = enforce_window(&dir, "g19", "conus_c13", now_at(19, 0), &config).unwrap();
        assert_eq!(
            pruned.removed_frames, 2,
            "both stale frames go once unlocked"
        );
        assert!(pruned.skipped_locked.is_empty());
        // The dir is fully evicted, so it (and its grid.rwg + .rw-lock) is gone.
        assert_eq!(pruned.removed_runs, vec!["conus_c13_20260610".to_string()]);
        assert!(
            !run_dir.exists(),
            "fully-evicted run dir removed once unlocked (lock file included)"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
