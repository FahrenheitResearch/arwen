//! Restrict DEM hydrography to complete cells inside the actual atmosphere grid.
//! Outside cells are open domain boundaries, never unforced catchments.
use crate::{grid::Grid, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ffi::{c_char, c_int, c_void, CString};
type Point = [f64; 2];
#[derive(Clone, Deserialize, Serialize)]
pub struct ParentGrid {
    pub crs: String, pub nx: u32, pub ny: u32,
    pub x0: f64, pub y0: f64, pub dx: f64, pub dy: f64,
}
impl ParentGrid {
    pub fn mask(&self, g: &Grid, sea: &mut [bool]) -> Result<usize> {
        validate_grid_crs(&self.crs)?;
        if self.nx == 0 || self.ny == 0 || self.dx <= 0. || self.dy <= 0. ||
            ![self.x0,self.y0,self.dx,self.dy].iter().all(|x|x.is_finite()) {
            return Err("parent grid needs finite origin and positive metre spacing/extent".into());
        }
        let xmax = self.x0 + self.nx as f64 * self.dx;
        let ymax = self.y0 + self.ny as f64 * self.dy;
        // Every worker owns its PROJ context. Require all four cell corners;
        // no catchment can be created solely outside the forced domain.
        let changed: Result<Vec<usize>> = sea.par_chunks_mut(g.nx).enumerate().map(|(r,row)| {
            let proj = Projection::new("EPSG:4326", &self.crs)?;
            let mut n = 0;
            for (c,excluded) in row.iter_mut().enumerate() {
                if *excluded { continue; }
                let mut inside = true;
                for (x,y) in [(c,r),(c+1,r),(c,r+1),(c+1,r+1)] {
                    let (lon,lat) = g.corner(x,y);
                    let [px,py] = proj.transform([lon,lat])?;
                    inside &= px >= self.x0 && px <= xmax && py >= self.y0 && py <= ymax;
                }
                if !inside { *excluded = true; n += 1; }
            }
            Ok(n)
        }).collect();
        Ok(changed?.iter().sum())
    }
}
#[repr(C)]
#[derive(Copy, Clone)]
struct Coord {
    x: f64,
    y: f64,
    z: f64,
    t: f64,
}
// Static link off Linux and Windows (unchanged); on Linux libproj is opened at run time so the binary
// carries no libproj.so.25 DT_NEEDED (manylinux_2_28 policy), and on Windows so the build needs no
// proj.lib and the binary no PROJ DLL import. Call sites are unchanged: each function below is a
// thin wrapper over the table `dyn_proj::api()` resolves (dlsym / GetProcAddress).
#[cfg(not(any(target_os = "linux", windows)))]
#[link(name = "proj")]
extern "C" {
    fn proj_create(c: *mut c_void, definition: *const c_char) -> *mut c_void;
    fn proj_crs_get_coordinate_system(c: *mut c_void, crs: *const c_void) -> *mut c_void;
    fn proj_cs_get_type(c: *mut c_void, cs: *const c_void) -> c_int;
    fn proj_cs_get_axis_count(c: *mut c_void, cs: *const c_void) -> c_int;
    fn proj_cs_get_axis_info(
        c: *mut c_void,
        cs: *const c_void,
        index: c_int,
        name: *mut *const c_char,
        abbrev: *mut *const c_char,
        direction: *mut *const c_char,
        factor: *mut f64,
        unit_name: *mut *const c_char,
        unit_authority: *mut *const c_char,
        unit_code: *mut *const c_char,
    ) -> c_int;
    fn proj_context_create() -> *mut c_void;
    fn proj_context_destroy(c: *mut c_void);
    fn proj_context_set_enable_network(c: *mut c_void, e: c_int) -> c_int;
    fn proj_create_crs_to_crs(
        c: *mut c_void,
        s: *const c_char,
        t: *const c_char,
        a: *mut c_void,
    ) -> *mut c_void;
    fn proj_normalize_for_visualization(c: *mut c_void, p: *const c_void) -> *mut c_void;
    fn proj_trans(p: *mut c_void, d: c_int, c: Coord) -> Coord;
    fn proj_destroy(p: *mut c_void) -> *mut c_void;
}

#[cfg(any(target_os = "linux", windows))]
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
mod dyn_proj {
    use super::*;
    use std::ffi::{c_char, c_void};
    use std::sync::OnceLock;

    // The platform loader: dlopen/dlsym on Linux, LoadLibraryA/GetProcAddress (kernel32) on Windows.
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
            if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
        }
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
            match GetLastError() {
                126 => "Windows error 126: the module or one of its dependent DLLs was not found".into(),
                127 => "Windows error 127: the procedure was not found".into(),
                193 => "Windows error 193: the DLL is not a valid 64-bit Windows library".into(),
                code => format!("Windows error {code}"),
            }
        }
    }
    #[cfg(target_os = "linux")]
    const SONAMES: [&[u8]; 2] = [b"libproj.so.25\0", b"libproj.so\0"];
    #[cfg(windows)]
    const SONAMES: [&[u8]; 12] = [b"proj_9.dll\0", b"proj_9_9.dll\0", b"proj_9_8.dll\0", b"proj_9_7.dll\0", b"proj_9_6.dll\0", b"proj_9_5.dll\0", b"proj_9_4.dll\0", b"proj_9_3.dll\0", b"proj_9_2.dll\0", b"proj_9_1.dll\0", b"proj_9_0.dll\0", b"proj.dll\0"];

    pub struct Api {
        pub proj_create: unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void,
        pub proj_crs_get_coordinate_system: unsafe extern "C" fn(*mut c_void, *const c_void) -> *mut c_void,
        pub proj_cs_get_type: unsafe extern "C" fn(*mut c_void, *const c_void) -> c_int,
        pub proj_cs_get_axis_count: unsafe extern "C" fn(*mut c_void, *const c_void) -> c_int,
        pub proj_cs_get_axis_info: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, *mut *const c_char, *mut *const c_char, *mut *const c_char, *mut f64, *mut *const c_char, *mut *const c_char, *mut *const c_char) -> c_int,
        pub proj_context_create: unsafe extern "C" fn() -> *mut c_void,
        pub proj_context_destroy: unsafe extern "C" fn(*mut c_void),
        pub proj_context_set_enable_network: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        pub proj_create_crs_to_crs: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, *mut c_void) -> *mut c_void,
        pub proj_normalize_for_visualization: unsafe extern "C" fn(*mut c_void, *const c_void) -> *mut c_void,
        pub proj_trans: unsafe extern "C" fn(*mut c_void, c_int, Coord) -> Coord,
        pub proj_destroy: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    }

    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

    unsafe fn message() -> String {
        sys::last_error()
    }

    unsafe fn sym(handle: *mut c_void, name: &[u8]) -> std::result::Result<*mut c_void, String> {
        let p = sys::sym(handle, name.as_ptr().cast());
        if p.is_null() {
            Err(format!("libproj has no {}: {}", String::from_utf8_lossy(&name[..name.len() - 1]), message()))
        } else {
            Ok(p)
        }
    }

    unsafe fn load() -> std::result::Result<Api, String> {
        let mut handle = std::ptr::null_mut();
        for soname in SONAMES {
            handle = sys::open(soname);
            if !handle.is_null() {
                break;
            }
        }
        if handle.is_null() {
            #[cfg(target_os = "linux")]
            return Err(format!("PROJ (libproj.so.25) is not installed on this host: {}; install the system PROJ library (Ubuntu: libproj25, RHEL: proj)", message()));
            #[cfg(windows)]
            return Err(format!("PROJ (proj_9_*.dll or proj.dll) was not found on this host: {}; install PROJ 9 (OSGeo4W, conda-forge proj or vcpkg) and put its bin directory on PATH", message()));
        }
        Ok(Api {
            proj_create: std::mem::transmute(sym(handle, b"proj_create\0")?),
            proj_crs_get_coordinate_system: std::mem::transmute(sym(handle, b"proj_crs_get_coordinate_system\0")?),
            proj_cs_get_type: std::mem::transmute(sym(handle, b"proj_cs_get_type\0")?),
            proj_cs_get_axis_count: std::mem::transmute(sym(handle, b"proj_cs_get_axis_count\0")?),
            proj_cs_get_axis_info: std::mem::transmute(sym(handle, b"proj_cs_get_axis_info\0")?),
            proj_context_create: std::mem::transmute(sym(handle, b"proj_context_create\0")?),
            proj_context_destroy: std::mem::transmute(sym(handle, b"proj_context_destroy\0")?),
            proj_context_set_enable_network: std::mem::transmute(sym(handle, b"proj_context_set_enable_network\0")?),
            proj_create_crs_to_crs: std::mem::transmute(sym(handle, b"proj_create_crs_to_crs\0")?),
            proj_normalize_for_visualization: std::mem::transmute(sym(handle, b"proj_normalize_for_visualization\0")?),
            proj_trans: std::mem::transmute(sym(handle, b"proj_trans\0")?),
            proj_destroy: std::mem::transmute(sym(handle, b"proj_destroy\0")?),
        })
    }

    fn api() -> &'static Api {
        match API.get_or_init(|| unsafe { load() }) {
            Ok(api) => api,
            Err(e) => panic!("{e}"),
        }
    }

    pub unsafe fn proj_create(c: *mut c_void, definition: *const c_char) -> *mut c_void {
        (api().proj_create)(c, definition)
    }
    pub unsafe fn proj_crs_get_coordinate_system(c: *mut c_void, crs: *const c_void) -> *mut c_void {
        (api().proj_crs_get_coordinate_system)(c, crs)
    }
    pub unsafe fn proj_cs_get_type(c: *mut c_void, cs: *const c_void) -> c_int {
        (api().proj_cs_get_type)(c, cs)
    }
    pub unsafe fn proj_cs_get_axis_count(c: *mut c_void, cs: *const c_void) -> c_int {
        (api().proj_cs_get_axis_count)(c, cs)
    }
    pub unsafe fn proj_cs_get_axis_info(c: *mut c_void, cs: *const c_void, index: c_int, name: *mut *const c_char, abbrev: *mut *const c_char, direction: *mut *const c_char, factor: *mut f64, unit_name: *mut *const c_char, unit_authority: *mut *const c_char, unit_code: *mut *const c_char) -> c_int {
        (api().proj_cs_get_axis_info)(c, cs, index, name, abbrev, direction, factor, unit_name, unit_authority, unit_code)
    }
    pub unsafe fn proj_context_create() -> *mut c_void {
        (api().proj_context_create)()
    }
    pub unsafe fn proj_context_destroy(c: *mut c_void) {
        (api().proj_context_destroy)(c)
    }
    pub unsafe fn proj_context_set_enable_network(c: *mut c_void, e: c_int) -> c_int {
        (api().proj_context_set_enable_network)(c, e)
    }
    pub unsafe fn proj_create_crs_to_crs(c: *mut c_void, s: *const c_char, t: *const c_char, a: *mut c_void) -> *mut c_void {
        (api().proj_create_crs_to_crs)(c, s, t, a)
    }
    pub unsafe fn proj_normalize_for_visualization(c: *mut c_void, p: *const c_void) -> *mut c_void {
        (api().proj_normalize_for_visualization)(c, p)
    }
    pub unsafe fn proj_trans(p: *mut c_void, d: c_int, c: Coord) -> Coord {
        (api().proj_trans)(p, d, c)
    }
    pub unsafe fn proj_destroy(p: *mut c_void) -> *mut c_void {
        (api().proj_destroy)(p)
    }
}
#[cfg(any(target_os = "linux", windows))]
use dyn_proj::*;
pub(crate) fn validate_grid_crs(definition: &str) -> Result<()> {
    let text = CString::new(definition)?;
    unsafe {
        let context = proj_context_create();
        if context.is_null() {
            return Err("PROJ context allocation failed".into());
        }
        proj_context_set_enable_network(context, 0);
        let crs = proj_create(context, text.as_ptr());
        let cs = if crs.is_null() {
            std::ptr::null_mut()
        } else {
            proj_crs_get_coordinate_system(context, crs)
        };
        // PROJ's public PJ_CS_TYPE_CARTESIAN is enum value 1 (proj.h).
        let mut valid = !cs.is_null()
            && proj_cs_get_type(context, cs) == 1
            && proj_cs_get_axis_count(context, cs) == 2;
        if valid {
            for axis in 0..2 {
                let mut factor = 0.;
                let status = proj_cs_get_axis_info(
                    context,
                    cs,
                    axis,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut factor,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
                valid &= status != 0 && (factor - 1.).abs() < 1e-12;
            }
        }
        if !cs.is_null() {
            proj_destroy(cs);
        }
        if !crs.is_null() {
            proj_destroy(crs);
        }
        proj_context_destroy(context);
        if valid {
            Ok(())
        } else {
            Err("area-weight grid CRS must have two Cartesian axes in metres".into())
        }
    }
}
pub(crate) struct Projection {
    context: *mut c_void,
    p: *mut c_void,
}
impl Projection {
    pub(crate) fn new(source: &str, target: &str) -> Result<Self> {
        unsafe {
            let s = CString::new(source)?;
            let t = CString::new(target)?;
            let context = proj_context_create();
            if context.is_null() {
                return Err("PROJ context allocation failed".into());
            }
            proj_context_set_enable_network(context, 0);
            let raw = proj_create_crs_to_crs(context, s.as_ptr(), t.as_ptr(), std::ptr::null_mut());
            if raw.is_null() {
                proj_context_destroy(context);
                return Err(format!("cannot transform {source} to {target}").into());
            }
            let p = proj_normalize_for_visualization(context, raw);
            proj_destroy(raw);
            if p.is_null() {
                proj_context_destroy(context);
                return Err("PROJ axis normalization failed".into());
            }
            Ok(Self { context, p })
        }
    }
    pub(crate) fn transform(&self, point: Point) -> Result<Point> {
        let c = unsafe {
            proj_trans(
                self.p,
                1,
                Coord {
                    x: point[0],
                    y: point[1],
                    z: 0.,
                    t: 0.,
                },
            )
        };
        if !c.x.is_finite() || !c.y.is_finite() {
            return Err("PROJ coordinate outside transform domain".into());
        }
        Ok([c.x, c.y])
    }
}
impl Drop for Projection {
    fn drop(&mut self) {
        unsafe {
            proj_destroy(self.p);
            proj_context_destroy(self.context);
        }
    }
}
