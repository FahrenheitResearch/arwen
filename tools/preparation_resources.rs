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
        return available.min(cgroup_cpus(Path::new("/sys/fs/cgroup"), &membership)
            .unwrap_or(available)).max(1);
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
fn available_memory() -> Option<u64> {
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

#[cfg(target_os = "linux")]
fn available_memory() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut available = meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        rest.split_whitespace().next()?.parse::<u64>().ok().map(|n| n.saturating_mul(1024))
    })?;
    let root = Path::new("/sys/fs/cgroup");
    let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
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
    for (group, v2) in groups {
        let read = |name| std::fs::read_to_string(group.join(name)).ok()?.trim().parse::<u64>().ok();
        let limit = read(if v2 { "memory.max" } else { "memory.limit_in_bytes" });
        let current = read(if v2 { "memory.current" } else { "memory.usage_in_bytes" });
        if let (Some(limit), Some(current)) = (limit, current) {
            let inactive = std::fs::read_to_string(group.join("memory.stat")).ok().and_then(|text| {
                text.lines().find_map(|line| {
                    let (key, value) = line.split_once(' ')?;
                    (key == if v2 { "inactive_file" } else { "total_inactive_file" })
                        .then(|| value.trim().parse::<u64>().ok()).flatten()
                })
            }).unwrap_or(0);
            available = available.min(limit.saturating_sub(current.saturating_sub(inactive)));
        }
    }
    Some(available)
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
    #[test]
    fn ancestry_stays_below_its_mount() {
        assert_eq!(ancestors(Path::new("cgroup"), "/jobs/one"),
            vec![PathBuf::from("cgroup/jobs/one"), PathBuf::from("cgroup/jobs"), PathBuf::from("cgroup")]);
        assert_eq!(ancestors(Path::new("cgroup"), "../escape"), vec![PathBuf::from("cgroup")]);
    }
}
