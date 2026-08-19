//! LANE 1.  Mercator, transcribed from module_llxy.F `set_merc`:1307,
//! `llij_merc`:1334, `ijll_merc`:1358.  Anchored at the known point's
//! longitude (stand_lon does not enter the math); rotation identically
//! zero (get_rotang PROJ_MERC).  Operation order matches the Python
//! float64 spelling exactly.

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
    let clain = (RAD_PER_DEG * spec.truelat1).cos();
    let dlon = spec.dx / (EARTH_RADIUS_M * clain);
    let mut rsw = 0.0;
    if spec.ref_lat != 0.0 {
        rsw = (0.5 * ((spec.ref_lat + 90.0) * RAD_PER_DEG)).tan().ln() / dlon;
    }
    MercatorState { dlon, rsw }
}

/// ijll_merc transcription (module_llxy.F:1358), one point.
pub fn ij_to_latlon(
    state: &MercatorState,
    spec: &GridSpec,
    x: f64,
    y: f64,
) -> (f64, f64) {
    let lat = 2.0
        * (state.dlon * (state.rsw + y - spec.known_y)).exp().atan()
        * DEG_PER_RAD
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
pub fn latlon_to_ij(
    state: &MercatorState,
    spec: &GridSpec,
    lat: f64,
    lon: f64,
) -> (f64, f64) {
    let deltalon = wrap180(lon - spec.ref_lon);
    let i = spec.known_x + (deltalon / (state.dlon * DEG_PER_RAD));
    let j = spec.known_y
        + (0.5 * ((lat + 90.0) * RAD_PER_DEG)).tan().ln() / state.dlon
        - state.rsw;
    (i, j)
}

/// get_map_factor PROJ_MERC branch (process_tile_module.F:1801):
/// sin(colat0) / sin(colat); equals 1 at truelat1.
pub fn map_factor(spec: &GridSpec, lat: f64) -> f64 {
    let colat0 = RAD_PER_DEG * (90.0 - spec.truelat1);
    let colat = RAD_PER_DEG * (90.0 - lat);
    colat0.sin() / colat.sin()
}

/// get_rotang PROJ_MERC branch: rotation is identically zero.
pub fn rotation() -> (f64, f64) {
    (0.0, 1.0)
}
