//! How much host memory is free right now, if the platform will say.
//!
//! Used to bound the batch render's worker width: each concurrent product
//! holds its own decoded planes, so a width chosen purely from core count
//! is a memory demand nobody checked.  A box with more cores than free
//! gigabytes is exactly the box that cannot afford one worker per core.
//!
//! `None` means "this platform was not asked" rather than "zero".  A
//! caller that cannot learn the number must not invent one, so the width
//! falls back to the core-count rule and the run behaves as it did before
//! this module existed.
//!
//! There is no dependency here on purpose.  A crate that reports free
//! memory would pull a platform-abstraction tree into a vendored,
//! air-gapped workspace to answer one question that each platform answers
//! in about ten lines.

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
#[cfg(target_os = "linux")]
pub fn available_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        let Some(rest) = line.strip_prefix("MemAvailable:") else {
            continue;
        };
        let mut fields = rest.split_whitespace();
        let value: u64 = fields.next()?.parse().ok()?;
        // The unit is always kB on this line, but read it rather than
        // assume it: a wrong unit here is a 1024x wrong memory budget.
        return match fields.next() {
            Some("kB") => Some(value * 1024),
            None => Some(value),
            Some(_) => None,
        };
    }
    None
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
}
