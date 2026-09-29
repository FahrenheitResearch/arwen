//! Fixed elementary arithmetic for the sampler; public grid outputs are unchanged.

use super::{np_mod, np_pow};
use super::{wrap180, GridSpec, DEG_PER_RAD, RAD_PER_DEG};
use crate::EARTH_RADIUS_M;

/// set_lc outputs.
#[derive(Debug, Clone)]
pub struct LambertState {
    pub cone: f64,
    pub rebydx: f64,
    pub polei: f64,
    pub polej: f64,
}

/// Cone constant, transcribed from module_llxy.F lc_cone (:1138).
pub fn lc_cone(truelat1: f64, truelat2: f64) -> f64 {
    if (truelat1 - truelat2).abs() > 0.1 {
        // secant
        let num = libm::log10(libm::cos(truelat1 * RAD_PER_DEG))
            - libm::log10(libm::cos(truelat2 * RAD_PER_DEG));
        let den = libm::log10(libm::tan((45.0 - truelat1.abs() / 2.0) * RAD_PER_DEG))
            - libm::log10(libm::tan((45.0 - truelat2.abs() / 2.0) * RAD_PER_DEG));
        num / den
    } else {
        libm::sin(truelat1.abs() * RAD_PER_DEG) // tangent
    }
}

/// set_lc transcription (module_llxy.F:1097).
pub fn setup(spec: &GridSpec, hemi: f64) -> LambertState {
    let cone = lc_cone(spec.truelat1, spec.truelat2);
    let rebydx = EARTH_RADIUS_M / spec.dx;
    let deltalon1 = wrap180(spec.ref_lon - spec.stand_lon);
    let ctl1r = libm::cos(spec.truelat1 * RAD_PER_DEG);
    let rsw = rebydx * ctl1r / cone
        * np_pow(
            libm::tan((90.0 * hemi - spec.ref_lat) * RAD_PER_DEG / 2.0)
                / libm::tan((90.0 * hemi - spec.truelat1) * RAD_PER_DEG / 2.0),
            cone,
        );
    let arg = cone * (deltalon1 * RAD_PER_DEG);
    let polei = hemi * spec.known_x - hemi * rsw * libm::sin(arg);
    let polej = hemi * spec.known_y + rsw * libm::cos(arg);
    LambertState {
        cone,
        rebydx,
        polei,
        polej,
    }
}

/// ijll_lc transcription (module_llxy.F:1174), one point.
pub fn ij_to_latlon(
    state: &LambertState,
    spec: &GridSpec,
    hemi: f64,
    x: f64,
    y: f64,
) -> (f64, f64) {
    let chi1 = (90.0 - hemi * spec.truelat1) * RAD_PER_DEG;
    let chi2 = (90.0 - hemi * spec.truelat2) * RAD_PER_DEG;
    let xx = hemi * x - state.polei;
    let yy = state.polej - hemi * y;
    let r2 = xx * xx + yy * yy;
    let r = libm::sqrt(r2) / state.rebydx;
    let mut lon = spec.stand_lon + DEG_PER_RAD * libm::atan2(hemi * xx, yy) / state.cone;
    lon = np_mod(lon + 360.0, 360.0);
    let chi = if chi1 == chi2 {
        // tangent (exact-equality branch, as in Fortran)
        2.0 * libm::atan((np_pow(r / libm::tan(chi1), 1.0 / state.cone) * libm::tan(chi1 * 0.5)))
    } else {
        // secant
        2.0 * libm::atan(
            (np_pow(r * state.cone / libm::sin(chi1), 1.0 / state.cone) * libm::tan(chi1 * 0.5)),
        )
    };
    let mut lat = (90.0 - chi * DEG_PER_RAD) * hemi;
    // pole point (r2 == 0): mirror the Fortran branch explicitly.
    if r2 == 0.0 {
        lat = 90.0 * hemi;
        lon = np_mod(spec.stand_lon + 360.0, 360.0);
    }
    if lon > 180.0 {
        lon -= 360.0;
    }
    if lon < -180.0 {
        lon += 360.0;
    }
    (lat, lon)
}

/// llij_lc transcription (module_llxy.F:1250), one point.
pub fn latlon_to_ij(
    state: &LambertState,
    spec: &GridSpec,
    hemi: f64,
    lat: f64,
    lon: f64,
) -> (f64, f64) {
    let deltalon = wrap180(lon - spec.stand_lon);
    let ctl1r = libm::cos(spec.truelat1 * RAD_PER_DEG);
    let rm = state.rebydx * ctl1r / state.cone
        * np_pow(
            libm::tan((90.0 * hemi - lat) * RAD_PER_DEG / 2.0)
                / libm::tan((90.0 * hemi - spec.truelat1) * RAD_PER_DEG / 2.0),
            state.cone,
        );
    let arg = state.cone * (deltalon * RAD_PER_DEG);
    let x = state.polei + hemi * rm * libm::sin(arg);
    let y = state.polej - rm * libm::cos(arg);
    (hemi * x, hemi * y)
}
