//! Minimal PROJ binding: transform points between two CRS definitions.
//! The same system libproj the engine's rw_hydrofabric links; network off.
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

#[cfg_attr(target_os = "linux", link(name = ":libproj.so.25"))]
#[cfg_attr(not(target_os = "linux"), link(name = "proj"))]
extern "C" {
    fn proj_context_create() -> *mut c_void;
    fn proj_context_destroy(c: *mut c_void);
    fn proj_context_set_enable_network(c: *mut c_void, e: c_int) -> c_int;
    fn proj_create_crs_to_crs(c: *mut c_void, s: *const c_char, t: *const c_char, area: *mut c_void) -> *mut c_void;
    fn proj_normalize_for_visualization(c: *mut c_void, p: *const c_void) -> *mut c_void;
    fn proj_trans(p: *mut c_void, d: c_int, c: Coord) -> Coord;
    fn proj_destroy(p: *mut c_void) -> *mut c_void;
}

pub struct Transform {
    context: *mut c_void,
    p: *mut c_void,
}
// A PROJ context and its transform are used by one thread at a time here
// (each rayon worker builds its own Transform).
unsafe impl Send for Transform {}

impl Transform {
    /// Axis order is normalised for visualisation: (x, y) / (lon, lat).
    pub fn new(source: &str, target: &str) -> Result<Self> {
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
                return Err(format!("PROJ cannot transform {} to {}", short(source), short(target)).into());
            }
            let p = proj_normalize_for_visualization(context, raw);
            proj_destroy(raw);
            if p.is_null() {
                proj_context_destroy(context);
                return Err("PROJ axis normalisation failed".into());
            }
            Ok(Self { context, p })
        }
    }
    pub fn apply(&self, x: f64, y: f64) -> Result<[f64; 2]> {
        let c = unsafe { proj_trans(self.p, 1, Coord { x, y, z: 0.0, t: 0.0 }) };
        if !c.x.is_finite() || !c.y.is_finite() {
            return Err(format!("PROJ point ({x}, {y}) is outside the transform domain").into());
        }
        Ok([c.x, c.y])
    }
}
impl Drop for Transform {
    fn drop(&mut self) {
        unsafe {
            proj_destroy(self.p);
            proj_context_destroy(self.context);
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
