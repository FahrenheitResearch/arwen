//! Free space on the disk that holds a directory.
//!
//! The frameset writer asks this once, before the first byte of the
//! frame stream, so a stream the disk cannot hold is refused with its
//! numbers instead of being written until the disk fills.  A global
//! source's 48 h stream is tens of GB, and without the question the
//! engine decoded for fourteen minutes and then stopped on a full disk.

use std::path::Path;

/// Bytes an unprivileged writer may still add on the disk holding `path`,
/// or `None` when the platform cannot say.  `None` skips the up-front
/// check; a write that then meets a full disk is still refused as one.
pub fn available_bytes(path: &Path) -> Option<u64> {
    available_bytes_impl(path)
}

#[cfg(unix)]
fn available_bytes_impl(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let name = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `statvfs` only writes the struct it is handed, and `name`
    // is a NUL-terminated string that outlives the call.
    let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
    let status = unsafe { libc::statvfs(name.as_ptr(), &mut stats) };
    if status != 0 {
        return None;
    }
    // `f_bavail` (not `f_bfree`): the blocks left to a writer without
    // the root reserve, counted in fragment-size units.
    #[allow(clippy::unnecessary_cast)]
    let (blocks, size) = (stats.f_bavail as u64, stats.f_frsize as u64);
    Some(blocks.saturating_mul(size))
}

#[cfg(windows)]
fn available_bytes_impl(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut available: u64 = 0;
    // SAFETY: `wide` is NUL-terminated and outlives the call; the two
    // totals this does not ask for are passed as null, which the API
    // documents as allowed.  The caller-available figure honours quotas.
    let status = unsafe {
        windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if status == 0 { None } else { Some(available) }
}

#[cfg(not(any(unix, windows)))]
fn available_bytes_impl(_path: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_real_directory_reports_some_space_and_a_missing_one_none() {
        let here = std::env::temp_dir();
        assert!(super::available_bytes(&here).is_some_and(|bytes| bytes > 0));
        let missing = here.join(format!("gpuwm-space-missing-{}", std::process::id()));
        assert_eq!(super::available_bytes(&missing), None);
    }
}
