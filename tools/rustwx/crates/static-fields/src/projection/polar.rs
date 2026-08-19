//! LANE 1.  Polar stereographic (either pole), transcribed from
//! module_llxy.F `set_ps`:696, `llij_ps`:732, `ijll_ps`:777.
//! `llij_ps` applies no cut-zone wrap to `lon - reflon`; transcribed
//! as-is (the Python did).  Operation order matches the Python float64
//! spelling exactly; `**2.0` follows numpy's power ufunc (`x * x`).

use super::npmath::np_pow;
use super::{wrap180, GridSpec, DEG_PER_RAD, RAD_PER_DEG};
use crate::EARTH_RADIUS_M;

/// set_ps outputs.
#[derive(Debug, Clone)]
pub struct PolarState {
    pub rebydx: f64,
    pub rsw: f64,
    pub polei: f64,
    pub polej: f64,
}

/// set_ps transcription (module_llxy.F:696).
pub fn setup(spec: &GridSpec, hemi: f64) -> PolarState {
    let rebydx = EARTH_RADIUS_M / spec.dx;
    let reflon = spec.stand_lon + 90.0;
    let scale_top = 1.0 + hemi * (spec.truelat1 * RAD_PER_DEG).sin();
    let ala1 = spec.ref_lat * RAD_PER_DEG;
    let rsw = rebydx * ala1.cos() * scale_top / (1.0 + hemi * ala1.sin());
    let alo1 = (spec.ref_lon - reflon) * RAD_PER_DEG;
    let polei = spec.known_x - rsw * alo1.cos();
    let polej = spec.known_y - hemi * rsw * alo1.sin();
    PolarState { rebydx, rsw, polei, polej }
}

/// ijll_ps transcription (module_llxy.F:777), one point.
pub fn ij_to_latlon(
    state: &PolarState,
    spec: &GridSpec,
    hemi: f64,
    x: f64,
    y: f64,
) -> (f64, f64) {
    let reflon = spec.stand_lon + 90.0;
    let scale_top = 1.0 + hemi * (spec.truelat1 * RAD_PER_DEG).sin();
    let xx = x - state.polei;
    let yy = (y - state.polej) * hemi;
    let r2 = xx * xx + yy * yy;
    let gi2 = np_pow(state.rebydx * scale_top, 2.0);
    let mut lat = DEG_PER_RAD * hemi * ((gi2 - r2) / (gi2 + r2)).asin();
    let arccos = (xx / r2.sqrt()).acos();
    let mut lon = if yy > 0.0 {
        reflon + DEG_PER_RAD * arccos
    } else {
        reflon - DEG_PER_RAD * arccos
    };
    // pole point (r2 == 0): mirror the Fortran branch explicitly.
    if r2 == 0.0 {
        lat = hemi * 90.0;
        lon = reflon;
    }
    if lon > 180.0 {
        lon -= 360.0;
    }
    if lon < -180.0 {
        lon += 360.0;
    }
    (lat, lon)
}

/// llij_ps transcription (module_llxy.F:732), one point.
pub fn latlon_to_ij(
    state: &PolarState,
    spec: &GridSpec,
    hemi: f64,
    lat: f64,
    lon: f64,
) -> (f64, f64) {
    let reflon = spec.stand_lon + 90.0;
    let scale_top = 1.0 + hemi * (spec.truelat1 * RAD_PER_DEG).sin();
    let ala = lat * RAD_PER_DEG;
    let rm = state.rebydx * ala.cos() * scale_top / (1.0 + hemi * ala.sin());
    let alo = (lon - reflon) * RAD_PER_DEG;
    let i = state.polei + rm * alo.cos();
    let j = state.polej + hemi * rm * alo.sin();
    (i, j)
}

/// get_map_factor PROJ_PS branch (process_tile_module.F:1791):
/// (1 + sin|truelat1|) / (1 + sin(sign(truelat1) * lat)).
pub fn map_factor(spec: &GridSpec, lat: f64) -> f64 {
    (1.0 + (RAD_PER_DEG * spec.truelat1.abs()).sin())
        / (1.0 + (RAD_PER_DEG * 1.0f64.copysign(spec.truelat1) * lat).sin())
}

/// get_rotang PROJ_PS branch: alpha = wrap(stand_lon - lon).
pub fn rotation(spec: &GridSpec, lon: f64) -> (f64, f64) {
    let alpha = wrap180(spec.stand_lon - lon) * RAD_PER_DEG;
    (alpha.sin(), alpha.cos())
}
