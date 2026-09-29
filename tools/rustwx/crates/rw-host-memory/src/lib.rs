//! How much host memory is free right now, if the platform will say.
//!
//! Used to bound the batch render's worker width: each concurrent product
//! holds its own decoded planes, so a width chosen purely from core count
//! is a memory demand nobody checked.  A box with more cores than free
//! gigabytes is exactly the box that cannot afford one worker per core.
//! The WRF import reads it too (`rw-wrfbatch` `wrf_volumes`), to size the
//! isobaric volume build to the host instead of to one fixed ceiling.
//!
//! `None` means "this platform was not asked" rather than "zero".  A
//! caller that cannot learn the number must not invent one, so the width
//! falls back to the core-count rule and the run behaves as it did before
//! this module existed.
//!
//! On Linux the answer is also capped by the memory cgroup this process
//! runs in, the same reading as `gpuwm.core.preflight.host_available_bytes`
//! (`tilestream.autoplan._cgroup_memory_headroom`).  The same walk answers
//! the smallest limit ([`cgroup_memory_limit`]), which the MPAS static
//! builder (`rw-mpas` `static_memory`) admits a build against and the
//! planner (`tilestream.autoplan._cgroup_memory_limit`) sizes its pinned
//! host store against.  Every one of those readers is held to one table
//! of stand-in files, `host_memory_cgroup_cases.json` beside this file.
//!
//! The renderer reaches this crate as `rusty_weather::host_memory`.  It is
//! a crate of its own so the MPAS builder reads the same walk without
//! depending on the renderer.
//!
//! There is no dependency here on purpose.  A crate that reports free
//! memory would pull a platform-abstraction tree into a vendored,
//! air-gapped workspace to answer one question that each platform answers
//! in about ten lines.

use std::path::{Path, PathBuf};

/// The file naming the cgroup this process runs in on each hierarchy.
pub const PROC_SELF_CGROUP: &str = "/proc/self/cgroup";

/// Where the cgroup file systems are mounted.
pub const CGROUP_ROOT: &str = "/sys/fs/cgroup";

/// Physical memory not currently in use, in bytes.
#[cfg(target_os = "windows")]
pub fn available_bytes() -> Option<u64> {
    // Every field is part of the layout the API writes into; only one is
    // read here, and shrinking the struct would corrupt the rest.
    #[repr(C)]
    #[allow(dead_code)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        length: size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // Safety: `status` is a live, correctly sized MEMORYSTATUSEX whose
    // `length` field is set as the API requires, and the call only writes
    // into it.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0).then_some(status.avail_phys)
}

/// Physical memory not currently in use, in bytes.
///
/// `MemAvailable` rather than `MemFree`: the kernel's own estimate of what
/// a new allocation can have without swapping, which counts reclaimable
/// page cache.  `MemFree` on a box that has been reading wrfout files all
/// day reads near zero and would cap every render at one worker.
///
/// Capped by the room left under this process's memory cgroup limit
/// ([`cgroup_memory_headroom`]).  THE BREAKAGE: inside a container,
/// `/proc/meminfo` is the host's, so a render in a container limited to
/// 16 GiB on a 256 GiB host read about 250 GiB free, sized its
/// pressure-level volumes and its worker width from that, and was killed
/// by the container's limit with every picture of the frame.
#[cfg(target_os = "linux")]
pub fn available_bytes() -> Option<u64> {
    available_within(
        Path::new("/proc/meminfo"),
        Path::new(PROC_SELF_CGROUP),
        Path::new(CGROUP_ROOT),
    )
}

/// [`available_bytes`] read from `meminfo` (a `/proc/meminfo`), capped by
/// [`cgroup_memory_headroom`] of `proc_self_cgroup` under `cgroup_root`.
///
/// `None` when `meminfo` gives no `MemAvailable`: a cgroup limit alone
/// says what this process may not exceed, not what the host has free.
pub fn available_within(
    meminfo: &Path,
    proc_self_cgroup: &Path,
    cgroup_root: &Path,
) -> Option<u64> {
    let available = mem_available(&std::fs::read_to_string(meminfo).ok()?)?;
    Some(match cgroup_memory_headroom(proc_self_cgroup, cgroup_root) {
        Some(room) => available.min(room),
        None => available,
    })
}

fn mem_available(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        let Some(rest) = line.strip_prefix("MemAvailable:") else {
            continue;
        };
        let mut fields = rest.split_whitespace();
        let value: u64 = fields.next()?.parse().ok()?;
        // The unit is always kB on this line, but read it rather than
        // assume it: a wrong unit here is a 1024x wrong memory budget.
        return match fields.next() {
            Some("kB") => value.checked_mul(1024),
            None => Some(value),
            Some(_) => None,
        };
    }
    None
}

/// A cgroup v1 limit at or above this is the "no limit" sentinel (about
/// 2^63 rounded down to a page), not a limit.
const CGROUP_UNLIMITED: u64 = 1 << 62;

/// One memory cgroup limit on the path from this process's cgroup up to
/// its mount (see [`cgroup_memory_limits`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CgroupMemoryLimit {
    /// `memory.max` (cgroup v2) or `memory.limit_in_bytes` (v1).
    pub limit: u64,
    /// `memory.current` or `memory.usage_in_bytes` as the kernel reports
    /// it, page cache included; `None` when it cannot be read.
    pub usage: Option<u64>,
    /// What may still be allocated before this limit binds: the limit
    /// minus the working set (the usage less the inactive file pages), or
    /// the whole limit when the usage cannot be read.
    pub room: u64,
}

/// The bytes this process may still allocate before a memory cgroup limit
/// binds, or `None` when no limit binds it: the least room of
/// [`cgroup_memory_limits`].
pub fn cgroup_memory_headroom(proc_self_cgroup: &Path, cgroup_root: &Path) -> Option<u64> {
    cgroup_memory_limits(proc_self_cgroup, cgroup_root)
        .iter()
        .map(|found| found.room)
        .min()
}

/// The smallest memory cgroup limit on the path from this process's cgroup
/// up to its mount, or `None` when no limit binds it.
///
/// A total, where [`cgroup_memory_headroom`] is room: the ceiling a whole
/// store or a whole build is sized against.  Every limit on the path binds,
/// so the smallest is this process's ceiling.  It need not sit on the
/// cgroup with the least room left: a slice nearly full of other work can
/// leave less room than a smaller limit on the process's own scope.
///
/// THE BREAKAGE: the planner (`tilestream.autoplan._cgroup_memory_limit`)
/// and the MPAS static builder read only the mount root's limit.  In a
/// systemd scope with `MemoryMax=2G` on a 30 GiB worker the root carries no
/// limit, so both read none: the planner sized its pinned host store to
/// the whole host, the builder admitted a build against no ceiling at all,
/// and the kernel killed the run at the scope's limit.
pub fn cgroup_memory_limit(proc_self_cgroup: &Path, cgroup_root: &Path) -> Option<u64> {
    cgroup_memory_limits(proc_self_cgroup, cgroup_root)
        .iter()
        .map(|found| found.limit)
        .min()
}

/// Every memory cgroup limit that binds this process.
///
/// `proc_self_cgroup` names the cgroup this process is in on each
/// hierarchy (`<id>:<controllers>:<path>`; cgroup v2 is `0::<path>`).
/// Every directory from that cgroup up to its mount under `cgroup_root`
/// is read, because a limit on an ancestor (a systemd slice, a pod) binds
/// as hard as one on the leaf:
///
/// - cgroup v2: `memory.max`, used `memory.current`;
/// - cgroup v1 `memory` controller: `memory.limit_in_bytes`, used
///   `memory.usage_in_bytes`.
///
/// The usage counts page cache, which the kernel reclaims inside the
/// cgroup before it kills anything, so the inactive file pages
/// (`inactive_file`, v1 `total_inactive_file`, in `memory.stat`) are not
/// counted as used in the room: the same working set a container runtime
/// evicts on.  Without that, a container that had read its wrfout files
/// through the page cache would read no room at all, the reason
/// `MemAvailable` is read rather than `MemFree`.  A limit whose usage
/// cannot be read is the room.  An unreadable `proc_self_cgroup` reads the
/// mounts' own roots.
pub fn cgroup_memory_limits(proc_self_cgroup: &Path, cgroup_root: &Path) -> Vec<CgroupMemoryLimit> {
    let mut memberships: Vec<(String, String)> = std::fs::read_to_string(proc_self_cgroup)
        .map(|text| {
            text.lines()
                .filter_map(|line| {
                    let mut fields = line.splitn(3, ':');
                    let _id = fields.next()?;
                    let controllers = fields.next()?;
                    let path = fields.next()?;
                    Some((controllers.to_string(), path.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();
    if memberships.is_empty() {
        memberships = vec![
            (String::new(), "/".to_string()),
            ("memory".to_string(), "/".to_string()),
        ];
    }
    let mut limits = Vec::new();
    for (controllers, relative) in &memberships {
        let (mounts, limit_name, usage_name, inactive_key) = if controllers.is_empty() {
            (
                vec![cgroup_root.to_path_buf()],
                "memory.max",
                "memory.current",
                "inactive_file",
            )
        } else if controllers.split(',').any(|name| name == "memory") {
            let mut mounts = vec![cgroup_root.join(controllers)];
            if controllers != "memory" {
                mounts.push(cgroup_root.join("memory"));
            }
            (
                mounts,
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
                "total_inactive_file",
            )
        } else {
            continue;
        };
        for mount in &mounts {
            for directory in cgroup_ancestry(mount, relative) {
                let Some(limit) = read_cgroup_bytes(&directory.join(limit_name)) else {
                    continue;
                };
                if limit == 0 || limit >= CGROUP_UNLIMITED {
                    continue;
                }
                let usage = read_cgroup_bytes(&directory.join(usage_name));
                let room = match usage {
                    None => limit,
                    Some(usage) => {
                        let inactive = std::fs::read_to_string(directory.join("memory.stat"))
                            .ok()
                            .and_then(|stat| stat_field(&stat, inactive_key))
                            .unwrap_or(0);
                        limit.saturating_sub(usage.saturating_sub(inactive))
                    }
                };
                limits.push(CgroupMemoryLimit { limit, usage, room });
            }
        }
    }
    limits
}

/// `mount/relative` and every directory above it, up to `mount`.
///
/// A container that mounts its own cgroup at `mount` while
/// `/proc/self/cgroup` still names the host path has no such directory
/// below `mount`; `mount` itself is always read.
fn cgroup_ancestry(mount: &Path, relative: &str) -> Vec<PathBuf> {
    let parts: Vec<&str> = relative
        .split('/')
        .filter(|part| !part.is_empty() && *part != "..")
        .collect();
    (0..=parts.len())
        .rev()
        .map(|depth| {
            parts[..depth]
                .iter()
                .fold(mount.to_path_buf(), |path, part| path.join(part))
        })
        .collect()
}

/// One cgroup file's integer, or `None` for absent, `max` or unreadable.
fn read_cgroup_bytes(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// The value of `key` in a `memory.stat` body (`<key> <value>` lines).
fn stat_field(stat: &str, key: &str) -> Option<u64> {
    stat.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        if fields.next()? != key {
            return None;
        }
        fields.next()?.parse().ok()
    })
}

/// Physical memory not currently in use, in bytes.
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
pub fn available_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_reports_a_believable_amount_of_free_memory() {
        let Some(bytes) = available_bytes() else {
            // An unsupported platform answers None, which is a correct
            // answer and the one the width rule is written to survive.
            return;
        };
        // A box that can build this workspace has more than 64 MiB free
        // and less than a petabyte; anything outside that is a unit
        // mistake, which is the failure this test exists to catch.
        assert!(bytes > 64 * 1024 * 1024, "{bytes} bytes free is not plausible");
        assert!(bytes < 1 << 50, "{bytes} bytes free is not plausible");
    }

    /// Writes one case of `host_memory_cgroup_cases.json` under a fresh
    /// directory: `meminfo`, `proc-self-cgroup` and the `cgroup` mount.
    /// A case's `null` meminfo or membership is a file that is not there.
    fn stand_in(case: &serde_json::Value, index: usize) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rw-host-memory-{}-{index}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let cgroup = dir.join("cgroup");
        std::fs::create_dir_all(&cgroup).unwrap();
        if let Some(text) = case["meminfo"].as_str() {
            std::fs::write(dir.join("meminfo"), text).unwrap();
        }
        if let Some(text) = case["proc_self_cgroup"].as_str() {
            std::fs::write(dir.join("proc-self-cgroup"), text).unwrap();
        }
        for (relative, text) in case["cgroup"].as_object().unwrap() {
            let path = cgroup.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text.as_str().unwrap()).unwrap();
        }
        dir
    }

    /// THE BREAKAGE: `available_bytes` read `MemAvailable` alone, which
    /// inside a container is the host's, so a render admitted a frame the
    /// container's own limit then killed; and the planner and the MPAS
    /// static builder read only the mount root's limit, so inside a capped
    /// systemd scope they read no limit at all.  Every case of the table
    /// the Python readers are held to as well (`tests/test_preflight.py`)
    /// and the MPAS builder's reader (`cargo test -p rw-mpas`).
    #[test]
    fn the_container_memory_limit_caps_the_available_memory() {
        let table: serde_json::Value =
            serde_json::from_str(include_str!("host_memory_cgroup_cases.json")).unwrap();
        let cases = table["cases"].as_array().unwrap();
        assert!(cases.len() >= 10, "the shared table lost its cases");
        for (index, case) in cases.iter().enumerate() {
            let name = case["name"].as_str().unwrap();
            for key in ["headroom", "limit", "available"] {
                assert!(case.get(key).is_some(), "{name}: the case names no {key}");
            }
            let dir = stand_in(case, index);
            let proc_self_cgroup = dir.join("proc-self-cgroup");
            let cgroup = dir.join("cgroup");
            assert_eq!(
                cgroup_memory_headroom(&proc_self_cgroup, &cgroup),
                case["headroom"].as_u64(),
                "{name}: headroom"
            );
            assert_eq!(
                cgroup_memory_limit(&proc_self_cgroup, &cgroup),
                case["limit"].as_u64(),
                "{name}: limit"
            );
            assert_eq!(
                available_within(&dir.join("meminfo"), &proc_self_cgroup, &cgroup),
                case["available"].as_u64(),
                "{name}: available"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
