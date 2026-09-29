//! Fixed elementary arithmetic for the sampler; public grid outputs are unchanged.

use super::np_pow;
use super::{GridSpec, DEG_PER_RAD, RAD_PER_DEG};
use crate::EARTH_RADIUS_M;

/// set_ps outputs.
#[derive(Debug, Clone)]
pub struct PolarState {
    pub rebydx: f64,
    pub polei: f64,
    pub polej: f64,
}

/// set_ps transcription (module_llxy.F:696).
pub fn setup(spec: &GridSpec, hemi: f64) -> PolarState {
    let rebydx = EARTH_RADIUS_M / spec.dx;
    let reflon = spec.stand_lon + 90.0;
    let scale_top = 1.0 + hemi * libm::sin(spec.truelat1 * RAD_PER_DEG);
    let ala1 = spec.ref_lat * RAD_PER_DEG;
    let rsw = rebydx * libm::cos(ala1) * scale_top / (1.0 + hemi * libm::sin(ala1));
    let alo1 = (spec.ref_lon - reflon) * RAD_PER_DEG;
    let polei = spec.known_x - rsw * libm::cos(alo1);
    let polej = spec.known_y - hemi * rsw * libm::sin(alo1);
    PolarState {
        rebydx,
        polei,
        polej,
    }
}

/// ijll_ps transcription (module_llxy.F:777), one point.
pub fn ij_to_latlon(state: &PolarState, spec: &GridSpec, hemi: f64, x: f64, y: f64) -> (f64, f64) {
    let reflon = spec.stand_lon + 90.0;
    let scale_top = 1.0 + hemi * libm::sin(spec.truelat1 * RAD_PER_DEG);
    let xx = x - state.polei;
    let yy = (y - state.polej) * hemi;
    let r2 = xx * xx + yy * yy;
    let gi2 = np_pow(state.rebydx * scale_top, 2.0);
    let mut lat = DEG_PER_RAD * hemi * libm::asin((gi2 - r2) / (gi2 + r2));
    let arccos = libm::acos(xx / libm::sqrt(r2));
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
    let scale_top = 1.0 + hemi * libm::sin(spec.truelat1 * RAD_PER_DEG);
    let ala = lat * RAD_PER_DEG;
    let rm = state.rebydx * libm::cos(ala) * scale_top / (1.0 + hemi * libm::sin(ala));
    let alo = (lon - reflon) * RAD_PER_DEG;
    let i = state.polei + rm * libm::cos(alo);
    let j = state.polej + hemi * rm * libm::sin(alo);
    (i, j)
}
