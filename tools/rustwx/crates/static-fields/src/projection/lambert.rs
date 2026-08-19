//! LANE 1.  Lambert conformal (secant/tangent), transcribed from
//! module_llxy.F `lc_cone`:1138, `set_lc`:1097, `ijll_lc`:1174,
//! `llij_lc`:1250 -- see `gpuwm/static/lambert.py` for the arbitration
//! notes (grid registration, map factor referenced to truelat1,
//! OMEGA_E Coriolis, get_rotang with NO hemisphere factor).
//!
//! Operation order matches the Python float64 spelling exactly (left
//! associativity preserved); byte parity is gated by
//! `tests/lane1_goldens.rs` against arrays extracted from the real
//! Python on the harness domains.

use super::npmath::{np_mod, np_pow};
use super::{wrap180, GridSpec, DEG_PER_RAD, RAD_PER_DEG};
use crate::EARTH_RADIUS_M;

/// set_lc outputs.
#[derive(Debug, Clone)]
pub struct LambertState {
    pub cone: f64,
    pub rebydx: f64,
    pub rsw: f64,
    pub polei: f64,
    pub polej: f64,
}

/// Cone constant, transcribed from module_llxy.F lc_cone (:1138).
pub fn lc_cone(truelat1: f64, truelat2: f64) -> f64 {
    if (truelat1 - truelat2).abs() > 0.1 {
        // secant
        let num = (truelat1 * RAD_PER_DEG).cos().log10()
            - (truelat2 * RAD_PER_DEG).cos().log10();
        let den = ((45.0 - truelat1.abs() / 2.0) * RAD_PER_DEG).tan().log10()
            - ((45.0 - truelat2.abs() / 2.0) * RAD_PER_DEG).tan().log10();
        num / den
    } else {
        (truelat1.abs() * RAD_PER_DEG).sin() // tangent
    }
}

/// set_lc transcription (module_llxy.F:1097).
pub fn setup(spec: &GridSpec, hemi: f64) -> LambertState {
    let cone = lc_cone(spec.truelat1, spec.truelat2);
    let rebydx = EARTH_RADIUS_M / spec.dx;
    let deltalon1 = wrap180(spec.ref_lon - spec.stand_lon);
    let ctl1r = (spec.truelat1 * RAD_PER_DEG).cos();
    let rsw = rebydx * ctl1r / cone
        * np_pow(
            ((90.0 * hemi - spec.ref_lat) * RAD_PER_DEG / 2.0).tan()
                / ((90.0 * hemi - spec.truelat1) * RAD_PER_DEG / 2.0).tan(),
            cone,
        );
    let arg = cone * (deltalon1 * RAD_PER_DEG);
    let polei = hemi * spec.known_x - hemi * rsw * arg.sin();
    let polej = hemi * spec.known_y + rsw * arg.cos();
    LambertState { cone, rebydx, rsw, polei, polej }
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
    let r = r2.sqrt() / state.rebydx;
    let mut lon =
        spec.stand_lon + DEG_PER_RAD * (hemi * xx).atan2(yy) / state.cone;
    lon = np_mod(lon + 360.0, 360.0);
    let chi = if chi1 == chi2 {
        // tangent (exact-equality branch, as in Fortran)
        2.0 * (np_pow(r / chi1.tan(), 1.0 / state.cone) * (chi1 * 0.5).tan())
            .atan()
    } else {
        // secant
        2.0 * (np_pow(r * state.cone / chi1.sin(), 1.0 / state.cone)
            * (chi1 * 0.5).tan())
        .atan()
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
    let ctl1r = (spec.truelat1 * RAD_PER_DEG).cos();
    let rm = state.rebydx * ctl1r / state.cone
        * np_pow(
            ((90.0 * hemi - lat) * RAD_PER_DEG / 2.0).tan()
                / ((90.0 * hemi - spec.truelat1) * RAD_PER_DEG / 2.0).tan(),
            state.cone,
        );
    let arg = state.cone * (deltalon * RAD_PER_DEG);
    let x = state.polei + hemi * rm * arg.sin();
    let y = state.polej - rm * arg.cos();
    (hemi * x, hemi * y)
}

/// Map scale factor m(lat); equals 1 at both true latitudes.
pub fn map_factor(state: &LambertState, spec: &GridSpec, hemi: f64, lat: f64) -> f64 {
    let half_colat = (90.0 * hemi - lat) * RAD_PER_DEG / 2.0;
    let half_colat1 = (90.0 * hemi - spec.truelat1) * RAD_PER_DEG / 2.0;
    (spec.truelat1 * RAD_PER_DEG).cos() / (lat * RAD_PER_DEG).cos()
        * np_pow(half_colat.tan() / half_colat1.tan(), state.cone)
}

/// (SINALPHA, COSALPHA): alpha = cone * (stand_lon - lon), geogrid
/// get_rotang PROJ_LC (no hemisphere factor).
pub fn rotation(state: &LambertState, spec: &GridSpec, lon: f64) -> (f64, f64) {
    let alpha = state.cone * RAD_PER_DEG * wrap180(spec.stand_lon - lon);
    (alpha.sin(), alpha.cos())
}
