//! CPU capacity shared by the preparation binaries and host kernels.
//! Explicit worker counts never bypass affinity or a container CPU quota.

use std::path::{Path, PathBuf};

pub fn quota_count(quota: &str, period: &str) -> Option<usize> {
    let quota = quota.trim().parse::<u64>().ok()?;
    let period = period.trim().parse::<u64>().ok()?;
    if quota == 0 || period == 0 {
        return None;
    }
    // Do not schedule an extra worker beyond the sustained CPU quota.
    usize::try_from((quota / period).max(1)).ok()
}

fn ancestors(root: &Path, relative: &str) -> Vec<PathBuf> {
    if relative.split('/').any(|part| part == "..") {
        return vec![root.to_owned()];
    }
    let mut path = root.join(relative.trim_start_matches('/'));
    let mut paths = Vec::new();
    while path.starts_with(root) {
        paths.push(path.clone());
        if path == root || !path.pop() { break; }
    }
    paths
}

pub fn cgroup_cpus(root: &Path, membership: &str) -> Option<usize> {
    let mut limits = Vec::new();
    // A namespaced container can expose only the root, even when its
    // membership names the host hierarchy. Always inspect that root.
    let mut groups = vec![(root.to_owned(), true)];
    for line in membership.lines() {
        let fields: Vec<_> = line.splitn(3, ':').collect();
        if fields.len() != 3 { continue; }
        if fields[1].is_empty() {
            groups.extend(ancestors(root, fields[2]).into_iter().map(|p| (p, true)));
        } else if fields[1].split(',').any(|name| name == "cpu") {
            for controller in [fields[1], "cpu", "cpu,cpuacct"] {
                groups.extend(ancestors(&root.join(controller), fields[2])
                    .into_iter().map(|p| (p, false)));
            }
        }
    }
    for (group, v2) in groups {
        let count = if v2 {
            std::fs::read_to_string(group.join("cpu.max")).ok().and_then(|value| {
                let parts: Vec<_> = value.split_whitespace().collect();
                (parts.len() == 2).then(|| quota_count(parts[0], parts[1])).flatten()
            })
        } else {
            std::fs::read_to_string(group.join("cpu.cfs_quota_us")).ok().and_then(|quota| {
                let period = std::fs::read_to_string(group.join("cpu.cfs_period_us")).ok()?;
                quota_count(&quota, &period)
            })
        };
        if let Some(count) = count { limits.push(count); }
    }
    limits.into_iter().min()
}

pub fn available_cpus() -> usize {
    let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    #[cfg(target_os = "linux")]
    {
        let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
        available.min(cgroup_cpus(Path::new("/sys/fs/cgroup"), &membership)
            .unwrap_or(available)).max(1)
    }
    #[cfg(not(target_os = "linux"))]
    available.max(1)
}

/// Host-kernel workers reserve 32 MiB of allocator/column scratch each.
/// This replaces the former fixed eight-worker ceiling with a memory price.
pub fn available_workers() -> usize {
    let cpus = available_cpus();
    #[cfg(any(target_os = "linux", windows))]
    if let Some(memory) = available_memory() {
        let by_memory = memory.saturating_mul(7) / 10 / (32 * 1024 * 1024);
        return cpus.min(by_memory.max(1) as usize);
    }
    cpus
}

#[cfg(windows)]
pub fn available_memory() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatus {
        length: u32, load: u32,
        total_phys: u64, avail_phys: u64,
        total_page: u64, avail_page: u64,
        total_virtual: u64, avail_virtual: u64, avail_extended: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" { fn GlobalMemoryStatusEx(status: *mut MemoryStatus) -> i32; }
    // All-zero integer fields are valid; the API requires the byte length.
    let mut status: MemoryStatus = unsafe { std::mem::zeroed() };
    status.length = std::mem::size_of::<MemoryStatus>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.avail_phys)
}

/// One memory cgroup limit that binds this process, as the kernel reports
/// it: v2 `memory.max`/`memory.current`/`inactive_file`, v1
/// `memory.limit_in_bytes`/`memory.usage_in_bytes`/`total_inactive_file`.
/// `usage` is every process in the group, not this one alone.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct CgroupMemory {
    pub group: PathBuf,
    pub limit: u64,
    pub usage: u64,
    pub inactive_file: u64,
}

#[allow(dead_code)]
impl CgroupMemory {
    /// What may still be allocated before this limit binds: the limit less
    /// the group's working set (its usage less the inactive page cache the
    /// kernel reclaims before it kills anything).
    pub fn room(&self) -> u64 {
        self.limit.saturating_sub(self.usage.saturating_sub(self.inactive_file))
    }
}

/// A cgroup v1 limit at or above this is the "no limit" sentinel.
#[allow(dead_code)]
const CGROUP_UNLIMITED: u64 = 1 << 62;

/// Every memory cgroup limit on the path from this process's groups
/// (`membership`, the text of `/proc/self/cgroup`) up to the mounts under
/// `root`, v2 and v1 alike.  The mount roots themselves are always read,
/// because a container that mounts its own group at the root still names
/// the host's path in its membership (`12:memory:/docker/<id>` on a vast
/// container, whose limit sits at `/sys/fs/cgroup/memory`).
#[allow(dead_code)]
pub fn cgroup_memory(root: &Path, membership: &str) -> Vec<CgroupMemory> {
    let mut groups = vec![(root.to_owned(), true)];
    for line in membership.lines() {
        let fields: Vec<_> = line.splitn(3, ':').collect();
        if fields.len() != 3 { continue; }
        if fields[1].is_empty() {
            groups.extend(ancestors(root, fields[2]).into_iter().map(|p| (p, true)));
        } else if fields[1].split(',').any(|name| name == "memory") {
            for controller in [fields[1], "memory"] {
                groups.extend(ancestors(&root.join(controller), fields[2]).into_iter().map(|p| (p, false)));
            }
        }
    }
    let mut found: Vec<CgroupMemory> = Vec::new();
    let mut visited: Vec<PathBuf> = Vec::new();
    for (group, v2) in groups {
        if visited.contains(&group) { continue; }
        visited.push(group.clone());
        let read = |name| std::fs::read_to_string(group.join(name)).ok()?.trim().parse::<u64>().ok();
        let limit = read(if v2 { "memory.max" } else { "memory.limit_in_bytes" });
        let usage = read(if v2 { "memory.current" } else { "memory.usage_in_bytes" });
        if let (Some(limit), Some(usage)) = (limit, usage) {
            if limit >= CGROUP_UNLIMITED { continue; }
            let inactive_file = std::fs::read_to_string(group.join("memory.stat")).ok().and_then(|text| {
                text.lines().find_map(|line| {
                    let (key, value) = line.split_once(' ')?;
                    (key == if v2 { "inactive_file" } else { "total_inactive_file" })
                        .then(|| value.trim().parse::<u64>().ok()).flatten()
                })
            }).unwrap_or(0);
            found.push(CgroupMemory { group, limit, usage, inactive_file });
        }
    }
    found
}

/// The memory cgroup with the least room left over this process, if any
/// limit binds it.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn binding_cgroup_memory() -> Option<CgroupMemory> {
    let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    cgroup_memory(Path::new("/sys/fs/cgroup"), &membership)
        .into_iter().min_by_key(CgroupMemory::room)
}

/// `MemAvailable`, lowered to the room left under every memory cgroup
/// limit this process runs in ([`cgroup_memory`]): the limit less what
/// every process in the group already holds.
#[cfg(target_os = "linux")]
pub fn available_memory() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let available = meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        rest.split_whitespace().next()?.parse::<u64>().ok().map(|n| n.saturating_mul(1024))
    })?;
    let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    Some(cgroup_memory(Path::new("/sys/fs/cgroup"), &membership)
        .iter().map(CgroupMemory::room).fold(available, u64::min))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quota_is_bounded_without_rounding_above_capacity() {
        assert_eq!(quota_count("18400000", "100000"), Some(184));
        assert_eq!(quota_count("150000", "100000"), Some(1));
        assert_eq!(quota_count("max", "100000"), None);
        assert_eq!(quota_count("-1", "100000"), None);
        assert_eq!(quota_count("1", "0"), None);
    }
    fn stand_in(name: &str, files: &[(&str, String)]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("prep-resources-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        for (relative, text) in files {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        root
    }

    /// THE BREAKAGE (pi-a4b, box G): a vast container is cgroup v1 with its
    /// limit at the memory mount's root while its membership names the
    /// host path, and the mapped engine read cgroup v2 only, so it priced a
    /// preparation on the host's MemAvailable (737 GB) under a 550.7 GB
    /// limit that other preparations in the container were already filling.
    #[test]
    fn a_v1_container_limit_and_its_other_tenants_are_read() {
        const GIB: u64 = 1 << 30;
        let root = stand_in("v1", &[
            ("memory/memory.limit_in_bytes", "550668075008\n".into()),
            ("memory/memory.usage_in_bytes", format!("{}\n", 400 * GIB)),
            ("memory/memory.stat", format!("cache 1\ntotal_inactive_file {}\n", 100 * GIB)),
        ]);
        let found = cgroup_memory(&root, "12:memory:/docker/b55ac8de\n0::/docker/b55ac8de\n");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].group, root.join("memory"));
        assert_eq!(found[0].limit, 550668075008);
        assert_eq!(found[0].usage, 400 * GIB);
        assert_eq!(found[0].inactive_file, 100 * GIB);
        // 550.7 GB less the 300 GiB working set the group's tenants hold.
        assert_eq!(found[0].room(), 550668075008 - 300 * GIB);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn v2_ancestors_are_read_and_unlimited_groups_are_not_limits() {
        const GIB: u64 = 1 << 30;
        let root = stand_in("v2", &[
            ("pool.slice/memory.max", format!("{}\n", 64 * GIB)),
            ("pool.slice/memory.current", format!("{}\n", 60 * GIB)),
            ("pool.slice/memory.stat", format!("anon 1\ninactive_file {}\n", 4 * GIB)),
            ("pool.slice/job.scope/memory.max", "max\n".into()),
            ("pool.slice/job.scope/memory.current", "1\n".into()),
            ("memory/memory.limit_in_bytes", "9223372036854771712\n".into()),
            ("memory/memory.usage_in_bytes", "1\n".into()),
        ]);
        let found = cgroup_memory(&root, "0::/pool.slice/job.scope\n12:memory:/x\n");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].group, root.join("pool.slice"));
        assert_eq!(found[0].room(), 8 * GIB);
        assert!(cgroup_memory(&root.join("absent"), "").is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn ancestry_stays_below_its_mount() {
        assert_eq!(ancestors(Path::new("cgroup"), "/jobs/one"),
            vec![PathBuf::from("cgroup/jobs/one"), PathBuf::from("cgroup/jobs"), PathBuf::from("cgroup")]);
        assert_eq!(ancestors(Path::new("cgroup"), "../escape"), vec![PathBuf::from("cgroup")]);
    }
}
