//! WPS Lambert-conformal projection math, for the declared-source rotation.
//!
//! Transcribed from `gpuwm.static.lambert.LambertGrid` /
//! `gpuwm.static.projection.ProjectedGrid`, which are themselves
//! transcriptions of WPS `module_llxy.F` (`lc_cone` :1138, `set_lc` :1097,
//! `ijll_lc` :1174).  Only the two entry points the decode path needs are
//! here: the grid construction a DECLARED source implies, and the
//! `(SINALPHA, COSALPHA)` rotation field that turns grid-relative wind
//! components into the earth basis.
//!
//! The exact-equality branch on `chi1 == chi2` and the `> 0.1` secant
//! threshold in `lc_cone` are Fortran's, kept verbatim: a source declaring
//! `latin1 == latin2` must take the tangent formula, and a source
//! declaring them 0.05 degrees apart must take the tangent formula too.
//! "Tidying" either test would move every rotated wind value.

/// WPS `constants_module` Earth radius — the radius the projection math
/// assumes, independent of the source's declared shape of Earth.
pub const EARTH_RADIUS_M: f64 = 6_370_000.0;

const RAD_PER_DEG: f64 = std::f64::consts::PI / 180.0;
const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;

/// `module_llxy.F` cut-zone wrap into (-180, 180].
///
/// This is `gpuwm.static.projection._wrap180`, the comparison form the
/// WPS transcription uses inside `ijll_lc` and inside the rotation's
/// longitude difference.  It is NOT the wrap the mapped path applies to
/// a DECLARED grid's corner — see [`wrap180_declared`].
pub fn wrap180(value: f64) -> f64 {
    if value > 180.0 {
        value - 360.0
    } else if value < -180.0 {
        value + 360.0
    } else {
        value
    }
}

/// `gpuwm.mapped_source._wrap180`: `((value + 180) % 360) - 180`.
///
/// gpuwm has TWO longitude wraps and they are not the same function.
/// The modulo form above is the one the mapped path applies to a
/// declared grid's `lon1`/`lov` before handing them to the projection;
/// the comparison form is what the projection itself uses.  They agree
/// to within a couple of ULPs and disagree in the last bits: on a real
/// declared 3-km Lambert grid, `237.280472` wraps to -122.719528 one way
/// and to a value two ULPs away the other, which moved `polei` by eight
/// ULPs and every rotated wind component in the frame with it.  Taking
/// the projection's wrap here — which is what this port originally did —
/// made every grid-relative wind field differ from the Python engine's
/// in the last two decimal digits, so the arrays hashed differently and
/// parity failed on the one staged source that rotates winds.
///
/// The modulo form is the less accurate of the two.  It is reproduced
/// rather than corrected because the Python engine is the behaviour of
/// record for this seam: changing it would move every mapped source's
/// wind field and grid placement at once, which is a ruling to ask for,
/// not a change to slip into a port.
pub fn wrap180_declared(value: f64) -> f64 {
    ((value + 180.0) % 360.0) - 180.0
}

/// `lc_cone` (module_llxy.F:1138).
fn cone_constant(truelat1: f64, truelat2: f64) -> f64 {
    if (truelat1 - truelat2).abs() > 0.1 {
        let numerator = (truelat1 * RAD_PER_DEG).cos().log10() - (truelat2 * RAD_PER_DEG).cos().log10();
        let denominator = ((45.0 - truelat1.abs() / 2.0) * RAD_PER_DEG).tan().log10()
            - ((45.0 - truelat2.abs() / 2.0) * RAD_PER_DEG).tan().log10();
        numerator / denominator
    } else {
        (truelat1.abs() * RAD_PER_DEG).sin()
    }
}

/// One WPS Lambert domain, set up from its known point.
#[derive(Debug, Clone)]
pub struct LambertGrid {
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub hemi: f64,
    pub cone: f64,
    rebydx: f64,
    polei: f64,
    polej: f64,
}

impl LambertGrid {
    /// `ProjectedGrid.__init__` + `LambertGrid._setup`.
    pub fn new(
        ref_lat: f64,
        ref_lon: f64,
        truelat1: f64,
        truelat2: f64,
        stand_lon: f64,
        dx: f64,
        known_x: f64,
        known_y: f64,
    ) -> Self {
        let hemi = if truelat1 < 0.0 { -1.0 } else { 1.0 };
        let cone = cone_constant(truelat1, truelat2);
        let rebydx = EARTH_RADIUS_M / dx;
        let deltalon1 = wrap180(ref_lon - stand_lon);
        let ctl1r = (truelat1 * RAD_PER_DEG).cos();
        let rsw = rebydx * ctl1r / cone
            * (((90.0 * hemi - ref_lat) * RAD_PER_DEG / 2.0).tan()
                / ((90.0 * hemi - truelat1) * RAD_PER_DEG / 2.0).tan())
            .powf(cone);
        let arg = cone * (deltalon1 * RAD_PER_DEG);
        let polei = hemi * known_x - hemi * rsw * arg.sin();
        let polej = hemi * known_y + rsw * arg.cos();
        Self {
            truelat1,
            truelat2,
            stand_lon,
            hemi,
            cone,
            rebydx,
            polei,
            polej,
        }
    }

    /// The grid a DECLARED Lambert source implies
    /// (`mapped_source.declared_lambert_source_grid`).
    ///
    /// The declared spacing is scaled by `R_WPS / earth_radius_m`, so the
    /// geometry stays the source's while the arithmetic stays WPS's.
    pub fn from_declaration(parameters: &crate::model::LambertParameters) -> Self {
        let scale = EARTH_RADIUS_M / parameters.earth_radius_m;
        Self::new(
            parameters.lat1,
            wrap180_declared(parameters.lon1),
            parameters.latin1,
            parameters.latin2,
            wrap180_declared(parameters.lov),
            parameters.dx_m * scale,
            1.0,
            1.0,
        )
    }

    /// `ijll_lc` (module_llxy.F:1174) at one projection coordinate.
    pub fn ij_to_latlon(&self, x: f64, y: f64) -> (f64, f64) {
        let chi1 = (90.0 - self.hemi * self.truelat1) * RAD_PER_DEG;
        let chi2 = (90.0 - self.hemi * self.truelat2) * RAD_PER_DEG;
        let xx = self.hemi * x - self.polei;
        let yy = self.polej - self.hemi * y;
        let r2 = xx * xx + yy * yy;
        let r = r2.sqrt() / self.rebydx;
        let mut lon = self.stand_lon + DEG_PER_RAD * (self.hemi * xx).atan2(yy) / self.cone;
        lon = (lon + 360.0).rem_euclid(360.0);
        let chi = if chi1 == chi2 {
            2.0 * ((r / chi1.tan()).powf(1.0 / self.cone) * (chi1 * 0.5).tan()).atan()
        } else {
            2.0 * ((r * self.cone / chi1.sin()).powf(1.0 / self.cone) * (chi1 * 0.5).tan()).atan()
        };
        let mut lat = (90.0 - chi * DEG_PER_RAD) * self.hemi;
        if r2 == 0.0 {
            lat = 90.0 * self.hemi;
            lon = (self.stand_lon + 360.0).rem_euclid(360.0);
        }
        if lon > 180.0 {
            lon -= 360.0;
        } else if lon < -180.0 {
            lon += 360.0;
        }
        (lat, lon)
    }
}

/// Source-grid `(SINALPHA, COSALPHA)` over a declared Lambert grid, row
/// major in (y, x) — `mapped_source._declared_grid_rotation`.
pub fn declared_grid_rotation(parameters: &crate::model::LambertParameters) -> (Vec<f64>, Vec<f64>) {
    let grid = LambertGrid::from_declaration(parameters);
    let nx = parameters.nx as usize;
    let ny = parameters.ny as usize;
    let mut sina = Vec::with_capacity(nx * ny);
    let mut cosa = Vec::with_capacity(nx * ny);
    for row in 0..ny {
        for column in 0..nx {
            let (_latitude, longitude) =
                grid.ij_to_latlon((column + 1) as f64, (row + 1) as f64);
            let mut difference = grid.stand_lon - longitude;
            if difference > 180.0 {
                difference -= 360.0;
            } else if difference < -180.0 {
                difference += 360.0;
            }
            let alpha = grid.hemi * grid.cone * std::f64::consts::PI / 180.0 * difference;
            sina.push(alpha.sin());
            cosa.push(alpha.cos());
        }
    }
    (sina, cosa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LambertParameters;

    fn awips_212_style() -> LambertParameters {
        LambertParameters {
            latin1: 50.0,
            latin2: 50.0,
            lov: 253.0,
            lat1: 1.0,
            lon1: 214.5,
            dx_m: 32463.0,
            dy_m: 32463.0,
            nx: 349,
            ny: 277,
            earth_radius_m: 6_371_229.0,
            shape_of_earth: 6,
        }
    }

    #[test]
    fn a_tangent_declaration_takes_the_sine_cone() {
        assert_eq!(cone_constant(50.0, 50.0), (50.0_f64 * RAD_PER_DEG).sin());
    }

    #[test]
    fn a_secant_declaration_takes_the_log_cone() {
        let cone = cone_constant(30.0, 60.0);
        assert!(cone > 0.0 && cone < 1.0, "{cone}");
        assert_ne!(cone, (30.0_f64 * RAD_PER_DEG).sin());
    }

    #[test]
    fn the_secant_threshold_is_the_fortran_tenth_of_a_degree() {
        // 0.05 apart is INSIDE the tangent branch; 0.2 apart is outside.
        assert_eq!(cone_constant(50.0, 50.05), (50.0_f64 * RAD_PER_DEG).sin());
        assert_ne!(cone_constant(50.0, 50.2), (50.0_f64 * RAD_PER_DEG).sin());
    }

    #[test]
    fn rotation_matches_the_python_engine_on_a_real_declared_grid() {
        // Golden: `gpuwm.mapped_source._declared_grid_rotation` run on the
        // staged 32 km AWIPS-grid declaration.  Corners and centre pin the
        // cone, the hemisphere, and the pole placement together; a sign
        // slip or a tangent/secant mix-up moves all five.
        let (sina, cosa) = declared_grid_rotation(&awips_212_style());
        let nx = 349usize;
        for (row, column, expected_sine, expected_cosine) in [
            (0usize, 0usize, 0.492_312_832_948_383_13, 0.870_418_333_052_755_4),
            (0, 348, -0.494_413_505_574_880_23, 0.869_226_832_020_939_1),
            (276, 0, 0.984_637_387_170_159, 0.174_611_614_123_237_6),
            (138, 174, -0.002_901_029_531_318_908, 0.999_995_792_004_975_6),
            (276, 348, -0.984_805_430_831_207_3, 0.173_661_346_894_927_13),
        ] {
            let position = row * nx + column;
            // Compared EXACTLY, not within 1e-15.  A near-tolerance is
            // what let a wrong longitude wrap through this test: it moved
            // the rotation by a handful of ULPs, which is invisible at
            // 1e-15 and fatal to a sha256.  The goldens are Python reprs,
            // so they parse to the very doubles the Python engine holds.
            assert_eq!(
                sina[position], expected_sine,
                "sin at ({row}, {column})"
            );
            assert_eq!(
                cosa[position], expected_cosine,
                "cos at ({row}, {column})"
            );
        }
    }

    #[test]
    fn the_declared_wrap_is_the_mapped_paths_wrap_not_the_projections() {
        // Measured: `gpuwm.mapped_source._wrap180(237.280472)` against
        // `gpuwm.static.projection._wrap180(237.280472)`.  Two ULPs
        // apart, and the mapped path uses the first one.  Pinned because
        // the difference is invisible in every printed decimal and shows
        // up only as a different sha256 three transformations later.
        let declared = wrap180_declared(237.280472);
        let projection = wrap180(237.280472);
        assert_eq!(declared.to_bits(), 0xc05e_ae0c_bf2b_2398);
        assert_eq!(projection.to_bits(), 0xc05e_ae0c_bf2b_239a);
        assert_ne!(declared, projection);
        // Values already inside the cut zone are untouched by both, and
        // a declared central meridian that lands on a half degree is
        // exact either way -- which is why most sources never noticed.
        assert_eq!(wrap180_declared(262.5), -97.5);
        assert_eq!(wrap180_declared(-97.5), -97.5);
    }

    #[test]
    fn every_rotation_cell_is_a_unit_vector() {
        let (sina, cosa) = declared_grid_rotation(&awips_212_style());
        for (sine, cosine) in sina.iter().zip(cosa.iter()) {
            assert!((sine * sine + cosine * cosine - 1.0).abs() < 1e-12);
        }
    }
}
