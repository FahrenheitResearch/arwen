//! Fixed elementary arithmetic for the sampler; public grid outputs are unchanged.

use super::{wrap180, GridSpec, DEG_PER_RAD, RAD_PER_DEG};
use crate::EARTH_RADIUS_M;

/// set_merc outputs.
#[derive(Debug, Clone)]
pub struct MercatorState {
    pub dlon: f64,
    pub rsw: f64,
}

/// set_merc transcription (module_llxy.F:1307).
pub fn setup(spec: &GridSpec) -> MercatorState {
    let clain = libm::cos(RAD_PER_DEG * spec.truelat1);
    let dlon = spec.dx / (EARTH_RADIUS_M * clain);
    let mut rsw = 0.0;
    if spec.ref_lat != 0.0 {
        rsw = libm::log(libm::tan(0.5 * ((spec.ref_lat + 90.0) * RAD_PER_DEG))) / dlon;
    }
    MercatorState { dlon, rsw }
}

/// ijll_merc transcription (module_llxy.F:1358), one point.
pub fn ij_to_latlon(state: &MercatorState, spec: &GridSpec, x: f64, y: f64) -> (f64, f64) {
    let lat =
        2.0 * libm::atan(libm::exp(state.dlon * (state.rsw + y - spec.known_y))) * DEG_PER_RAD
            - 90.0;
    let mut lon = (x - spec.known_x) * state.dlon * DEG_PER_RAD + spec.ref_lon;
    if lon > 180.0 {
        lon -= 360.0;
    }
    if lon < -180.0 {
        lon += 360.0;
    }
    (lat, lon)
}

/// llij_merc transcription (module_llxy.F:1334), one point.
pub fn latlon_to_ij(state: &MercatorState, spec: &GridSpec, lat: f64, lon: f64) -> (f64, f64) {
    let deltalon = wrap180(lon - spec.ref_lon);
    let i = spec.known_x + (deltalon / (state.dlon * DEG_PER_RAD));
    let j = spec.known_y + libm::log(libm::tan(0.5 * ((lat + 90.0) * RAD_PER_DEG))) / state.dlon
        - state.rsw;
    (i, j)
}
