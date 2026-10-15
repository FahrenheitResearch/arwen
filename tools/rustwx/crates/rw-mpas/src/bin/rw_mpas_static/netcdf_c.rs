//! The netCDF C library, loaded when a tool that reads through it starts.
//!
//! Linked at build time (`#[link(name = "netcdf")]`), this one dependency made
//! every build of the crate need `-lnetcdf` and every shipped binary need
//! `libnetcdf.so` to START: the manylinux_2_28 release container and the CI
//! runners have no libnetcdf, so the workspace build failed to link, and the
//! manylinux policy refuses a wheel binary whose DT_NEEDED names it. Loaded
//! here, the static builder and every other tool run without it, and the
//! tools that read native streams through it name what is missing.
//!
//! The entry points are the library's own C ABI, unchanged since netCDF 4.0.
//! The reads stay `nc_get_vara*` on stored types: no CF unpacking and no
//! precision conversion, exactly as when the library was linked.

use std::ffi::OsString;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::OnceLock;

/// Explicit library path. When set, it is the only candidate tried.
pub const LIBRARY_ENV: &str = "RW_MPAS_NETCDF_LIBRARY";

pub struct Api {
    _library: libloading::Library,
    pub nc_open: unsafe extern "C" fn(*const c_char, c_int, *mut c_int) -> c_int,
    pub nc_close: unsafe extern "C" fn(c_int) -> c_int,
    pub nc_strerror: unsafe extern "C" fn(c_int) -> *const c_char,
    pub nc_inq_nvars: unsafe extern "C" fn(c_int, *mut c_int) -> c_int,
    pub nc_inq_format: unsafe extern "C" fn(c_int, *mut c_int) -> c_int,
    pub nc_inq_grps: unsafe extern "C" fn(c_int, *mut c_int, *mut c_int) -> c_int,
    pub nc_inq_var: unsafe extern "C" fn(
        c_int, c_int, *mut c_char, *mut c_int, *mut c_int, *mut c_int, *mut c_int,
    ) -> c_int,
    pub nc_inq_dim: unsafe extern "C" fn(c_int, c_int, *mut c_char, *mut usize) -> c_int,
    pub nc_get_vara:
        unsafe extern "C" fn(c_int, c_int, *const usize, *const usize, *mut c_void) -> c_int,
}

static LOADED: OnceLock<Result<Api, String>> = OnceLock::new();

/// Library names tried, best first, when [`LIBRARY_ENV`] is not set.
fn candidates() -> Vec<OsString> {
    if let Some(explicit) = std::env::var_os(LIBRARY_ENV).filter(|v| !v.is_empty()) {
        return vec![explicit];
    }
    let mut names: Vec<OsString> = Vec::new();
    // A conda environment carries its own build, which is the one its
    // Python netCDF4 reads through; try it before the system's.
    if let Some(prefix) = std::env::var_os("CONDA_PREFIX").filter(|v| !v.is_empty()) {
        let prefix = std::path::PathBuf::from(prefix);
        if cfg!(windows) {
            names.push(prefix.join("Library").join("bin").join("netcdf.dll").into_os_string());
        } else if cfg!(target_os = "macos") {
            names.push(prefix.join("lib").join("libnetcdf.dylib").into_os_string());
        } else {
            names.push(prefix.join("lib").join("libnetcdf.so").into_os_string());
        }
    }
    let system: &[&str] = if cfg!(windows) {
        &["netcdf.dll"]
    } else if cfg!(target_os = "macos") {
        &["libnetcdf.dylib", "libnetcdf.22.dylib", "libnetcdf.19.dylib"]
    } else {
        // The development link first, then the sonames Debian, Ubuntu,
        // Fedora and EL ship (netCDF-C 4.9 is .so.19 and later .so.22).
        &["libnetcdf.so", "libnetcdf.so.22", "libnetcdf.so.19", "libnetcdf.so.18",
          "libnetcdf.so.15", "libnetcdf.so.13"]
    };
    names.extend(system.iter().map(OsString::from));
    names
}

fn symbol<T: Copy>(library: &libloading::Library, name: &str) -> Result<T, String> {
    // SAFETY: each requested type is the netCDF C prototype of `name`.
    unsafe { library.get::<T>(name) }
        .map(|found| *found)
        .map_err(|error| format!("{name}: {error}"))
}

fn open(name: &OsString) -> Result<Api, String> {
    // SAFETY: loading runs the library's initialisers, as linking it did.
    let library = unsafe { libloading::Library::new(name) }.map_err(|e| e.to_string())?;
    Ok(Api {
        nc_open: symbol(&library, "nc_open")?,
        nc_close: symbol(&library, "nc_close")?,
        nc_strerror: symbol(&library, "nc_strerror")?,
        nc_inq_nvars: symbol(&library, "nc_inq_nvars")?,
        nc_inq_format: symbol(&library, "nc_inq_format")?,
        nc_inq_grps: symbol(&library, "nc_inq_grps")?,
        nc_inq_var: symbol(&library, "nc_inq_var")?,
        nc_inq_dim: symbol(&library, "nc_inq_dim")?,
        nc_get_vara: symbol(&library, "nc_get_vara")?,
        _library: library,
    })
}

/// Load the library once; the refusal names every candidate and the remedy.
pub fn load() -> Result<&'static Api, String> {
    LOADED
        .get_or_init(|| {
            let mut tried = Vec::new();
            for name in candidates() {
                match open(&name) {
                    Ok(api) => return Ok(api),
                    Err(error) => tried.push(format!("{}: {error}", name.to_string_lossy())),
                }
            }
            Err(format!(
                "the netCDF C library is not loadable, and this tool reads native MPAS \
                 streams through it. Tried:\n  {}\nInstall it (Debian/Ubuntu: \
                 apt install libnetcdf-dev; Fedora/EL: dnf install netcdf; conda: \
                 conda install -c conda-forge libnetcdf) or set {LIBRARY_ENV} to the \
                 library file.",
                tried.join("\n  ")
            ))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// The loaded library. Every caller sits behind a successful [`load`]: the
/// tools load it before their first file is opened.
pub fn api() -> &'static Api {
    match LOADED.get() {
        Some(Ok(api)) => api,
        _ => unreachable!("netCDF entry point used before netcdf_c::load succeeded"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_name_a_versioned_system_library() {
        let names: Vec<String> = candidates().iter().map(|n| n.to_string_lossy().into_owned()).collect();
        if std::env::var_os(LIBRARY_ENV).is_none() {
            assert!(names.iter().any(|n| n.contains("netcdf")), "{names:?}");
            if cfg!(all(unix, not(target_os = "macos"))) {
                assert!(names.iter().any(|n| n == "libnetcdf.so.19"), "{names:?}");
            }
        }
    }

    #[test]
    fn a_missing_explicit_library_is_refused_by_name() {
        let error = open(&OsString::from("/nonexistent/libnetcdf-for-a-refusal-test.so"))
            .err()
            .expect("a missing file cannot load");
        assert!(!error.is_empty());
    }
}
