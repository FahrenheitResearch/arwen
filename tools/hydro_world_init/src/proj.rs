//! Minimal PROJ binding: transform points between two CRS definitions.
//! The same system libproj the engine's rw_hydrofabric uses, opened the same way: at run time on
//! Linux (no libproj.so.25 DT_NEEDED, manylinux_2_28 policy, and no libproj needed to build) and on
//! Windows (no proj.lib at build time, no PROJ DLL import); linked elsewhere. Network off.
use crate::Result;
use std::ffi::{c_char, c_int, c_void, CString};

#[repr(C)]
#[derive(Clone, Copy)]
struct Coord {
    x: f64,
    y: f64,
    z: f64,
    t: f64,
}

#[cfg(not(any(target_os = "linux", windows)))]
mod api {
    use super::Coord;
    use std::ffi::{c_char, c_int, c_void};
    #[link(name = "proj")]
    extern "C" {
        fn proj_context_create() -> *mut c_void;
        fn proj_context_destroy(c: *mut c_void);
        fn proj_context_set_enable_network(c: *mut c_void, e: c_int) -> c_int;
        fn proj_create_crs_to_crs(c: *mut c_void, s: *const c_char, t: *const c_char, area: *mut c_void) -> *mut c_void;
        fn proj_normalize_for_visualization(c: *mut c_void, p: *const c_void) -> *mut c_void;
        fn proj_trans(p: *mut c_void, d: c_int, c: Coord) -> Coord;
        fn proj_destroy(p: *mut c_void) -> *mut c_void;
    }
    pub struct Api {
        pub proj_context_create: unsafe extern "C" fn() -> *mut c_void,
        pub proj_context_destroy: unsafe extern "C" fn(*mut c_void),
        pub proj_context_set_enable_network: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        pub proj_create_crs_to_crs: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, *mut c_void) -> *mut c_void,
        pub proj_normalize_for_visualization: unsafe extern "C" fn(*mut c_void, *const c_void) -> *mut c_void,
        pub proj_trans: unsafe extern "C" fn(*mut c_void, c_int, Coord) -> Coord,
        pub proj_destroy: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    }
    static API: Api = Api {
        proj_context_create,
        proj_context_destroy,
        proj_context_set_enable_network,
        proj_create_crs_to_crs,
        proj_normalize_for_visualization,
        proj_trans,
        proj_destroy,
    };
    pub fn get() -> std::result::Result<&'static Api, String> {
        Ok(&API)
    }
}

#[cfg(any(target_os = "linux", windows))]
mod api {
    use super::Coord;
    use std::ffi::{c_char, c_int, c_void};
    use std::sync::OnceLock;

    #[cfg(target_os = "linux")]
    mod sys {
        use std::ffi::{c_char, c_int, c_void, CStr};
        #[link(name = "dl")]
        extern "C" {
            fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            fn dlerror() -> *const c_char;
        }
        const RTLD_NOW: c_int = 2;
        pub unsafe fn open(name: &[u8]) -> *mut c_void {
            dlopen(name.as_ptr().cast(), RTLD_NOW)
        }
        pub unsafe fn sym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
            dlsym(handle, name)
        }
        pub unsafe fn last_error() -> String {
            let p = dlerror();
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
        pub const SONAMES: &[&[u8]] = &[b"libproj.so.25\0", b"libproj.so\0"];
        pub const INSTALL: &str = "install the system PROJ library (Ubuntu: libproj25, RHEL: proj)";
    }
    #[cfg(windows)]
    mod sys {
        use std::ffi::{c_char, c_void};
        #[link(name = "kernel32")]
        extern "system" {
            fn LoadLibraryA(name: *const c_char) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
            fn GetLastError() -> u32;
        }
        pub unsafe fn open(name: &[u8]) -> *mut c_void {
            LoadLibraryA(name.as_ptr().cast())
        }
        pub unsafe fn sym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
            GetProcAddress(handle, name)
        }
        pub unsafe fn last_error() -> String {
            format!("Windows error {}", GetLastError())
        }
        pub const SONAMES: &[&[u8]] = &[
            b"proj_9.dll\0", b"proj_9_9.dll\0", b"proj_9_8.dll\0", b"proj_9_7.dll\0", b"proj_9_6.dll\0",
            b"proj_9_5.dll\0", b"proj_9_4.dll\0", b"proj_9_3.dll\0", b"proj_9_2.dll\0", b"proj_9_1.dll\0",
            b"proj_9_0.dll\0", b"proj.dll\0",
        ];
        pub const INSTALL: &str = "install PROJ 9 (OSGeo4W, conda-forge proj or vcpkg) and put its bin directory on PATH";
    }

    pub struct Api {
        pub proj_context_create: unsafe extern "C" fn() -> *mut c_void,
        pub proj_context_destroy: unsafe extern "C" fn(*mut c_void),
        pub proj_context_set_enable_network: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        pub proj_create_crs_to_crs: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, *mut c_void) -> *mut c_void,
        pub proj_normalize_for_visualization: unsafe extern "C" fn(*mut c_void, *const c_void) -> *mut c_void,
        pub proj_trans: unsafe extern "C" fn(*mut c_void, c_int, Coord) -> Coord,
        pub proj_destroy: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    }

    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

    unsafe fn sym(handle: *mut c_void, name: &[u8]) -> std::result::Result<*mut c_void, String> {
        let p = sys::sym(handle, name.as_ptr().cast());
        if p.is_null() {
            Err(format!("libproj has no {}: {}", String::from_utf8_lossy(&name[..name.len() - 1]), sys::last_error()))
        } else {
            Ok(p)
        }
    }

    unsafe fn load() -> std::result::Result<Api, String> {
        let handle = sys::SONAMES.iter().map(|name| sys::open(name)).find(|h| !h.is_null());
        let Some(handle) = handle else {
            return Err(format!("PROJ is not installed on this host: {}; {}", sys::last_error(), sys::INSTALL));
        };
        Ok(Api {
            proj_context_create: std::mem::transmute(sym(handle, b"proj_context_create\0")?),
            proj_context_destroy: std::mem::transmute(sym(handle, b"proj_context_destroy\0")?),
            proj_context_set_enable_network: std::mem::transmute(sym(handle, b"proj_context_set_enable_network\0")?),
            proj_create_crs_to_crs: std::mem::transmute(sym(handle, b"proj_create_crs_to_crs\0")?),
            proj_normalize_for_visualization: std::mem::transmute(sym(handle, b"proj_normalize_for_visualization\0")?),
            proj_trans: std::mem::transmute(sym(handle, b"proj_trans\0")?),
            proj_destroy: std::mem::transmute(sym(handle, b"proj_destroy\0")?),
        })
    }

    pub fn get() -> std::result::Result<&'static Api, String> {
        API.get_or_init(|| unsafe { load() }).as_ref().map_err(Clone::clone)
    }
}

pub struct Transform {
    context: *mut c_void,
    p: *mut c_void,
    api: &'static api::Api,
}
// A PROJ context and its transform are used by one thread at a time here
// (each rayon worker builds its own Transform).
unsafe impl Send for Transform {}

impl Transform {
    /// Axis order is normalised for visualisation: (x, y) / (lon, lat).
    pub fn new(source: &str, target: &str) -> Result<Self> {
        let api = api::get()?;
        unsafe {
            let s = CString::new(source)?;
            let t = CString::new(target)?;
            let context = (api.proj_context_create)();
            if context.is_null() {
                return Err("PROJ context allocation failed".into());
            }
            (api.proj_context_set_enable_network)(context, 0 as c_int);
            let raw = (api.proj_create_crs_to_crs)(context, s.as_ptr() as *const c_char, t.as_ptr() as *const c_char, std::ptr::null_mut());
            if raw.is_null() {
                (api.proj_context_destroy)(context);
                return Err(format!("PROJ cannot transform {} to {}", short(source), short(target)).into());
            }
            let p = (api.proj_normalize_for_visualization)(context, raw);
            (api.proj_destroy)(raw);
            if p.is_null() {
                (api.proj_context_destroy)(context);
                return Err("PROJ axis normalisation failed".into());
            }
            Ok(Self { context, p, api })
        }
    }
    pub fn apply(&self, x: f64, y: f64) -> Result<[f64; 2]> {
        let c = unsafe { (self.api.proj_trans)(self.p, 1, Coord { x, y, z: 0.0, t: 0.0 }) };
        if !c.x.is_finite() || !c.y.is_finite() {
            return Err(format!("PROJ point ({x}, {y}) is outside the transform domain").into());
        }
        Ok([c.x, c.y])
    }
}
impl Drop for Transform {
    fn drop(&mut self) {
        unsafe {
            (self.api.proj_destroy)(self.p);
            (self.api.proj_context_destroy)(self.context);
        }
    }
}

fn short(s: &str) -> String {
    s.chars().take(80).collect()
}

/// The CRS text a fabric's srs row names: EPSG:<code> for EPSG rows, the WKT
/// definition otherwise (the contract's ARWEN rows carry WKT2).
pub fn fabric_crs(srs: &hydro_world::contract::Srs) -> String {
    if srs.organization.eq_ignore_ascii_case("EPSG") {
        format!("EPSG:{}", srs.organization_coordsys_id)
    } else {
        srs.definition.clone()
    }
}
