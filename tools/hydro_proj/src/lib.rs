//! Link anchor shared by every hydro PROJ consumer. Source and database are
//! vendored; CMake disables network support and uses only embedded resources.
pub const VERSION: &str = "9.7.1";

/// Keep the native link metadata reachable from consumers with hand-written FFI.
#[inline(never)]
pub fn linked() {}
