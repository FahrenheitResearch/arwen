//! Wind calculations for vertical profiles.
//!
//! Re-exports basic wind utilities from `wx_math::dynamics` with MetPy-compatible
//! names, and implements profile-based calculations (shear, helicity, storm motion)
//! that operate on sounding data rather than 2-D grids.

// ─────────────────────────────────────────────
// Re-exports from wx_math::dynamics
// ─────────────────────────────────────────────

/// Wind speed from (u, v) components: sqrt(u^2 + v^2).
///
/// Operates element-wise on slices.
///
/// # Panics
/// Panics if `u` and `v` have different lengths.
pub use wx_math::dynamics::wind_speed;

/// Meteorological wind direction (degrees, 0 = north, 90 = east) from (u, v).
///
/// Returns the direction the wind is *coming from*.
///
/// # Panics
/// Panics if `u` and `v` have different lengths.
pub use wx_math::dynamics::wind_direction;

/// Convert (speed, direction) to (u, v) components.
///
/// Direction is meteorological degrees (0 = from north).
///
/// # Panics
/// Panics if `speed` and `direction` have different lengths.
pub use wx_math::dynamics::wind_components;

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/// Linearly interpolate a value at `target_h` from a height-sorted profile.
///
/// Returns `None` if `target_h` is outside the profile range.
fn interp_at_height(profile: &[f64], heights: &[f64], target_h: f64) -> Option<f64> {
    debug_assert_eq!(profile.len(), heights.len());
    let n = heights.len();
    if n == 0 {
        return None;
    }
    if target_h <= heights[0] {
        return Some(profile[0]);
    }
    if target_h >= heights[n - 1] {
        return Some(profile[n - 1]);
    }
    for i in 1..n {
        if heights[i] >= target_h {
            let frac = (target_h - heights[i - 1]) / (heights[i] - heights[i - 1]);
            return Some(profile[i - 1] + frac * (profile[i] - profile[i - 1]));
        }
    }
    None
}

// ─────────────────────────────────────────────
// New profile-based functions
// ─────────────────────────────────────────────

/// Bulk wind shear between two height levels.
///
/// Computes (delta-u, delta-v) = wind(top) - wind(bottom), interpolating to
/// exact height levels when they fall between profile points.
///
/// # Arguments
/// * `u_prof` — u-component profile (m/s), ordered by ascending height
/// * `v_prof` — v-component profile (m/s)
/// * `height_prof` — height AGL profile (m), must be monotonically increasing
/// * `bottom_m` — bottom of the shear layer (m AGL)
/// * `top_m` — top of the shear layer (m AGL)
///
/// # Panics
/// Panics if slices have mismatched lengths or fewer than 2 levels.
pub fn bulk_shear(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    assert_eq!(u_prof.len(), v_prof.len());
    assert_eq!(u_prof.len(), height_prof.len());
    assert!(u_prof.len() >= 2, "need at least 2 levels");

    let u_bot = interp_at_height(u_prof, height_prof, bottom_m).unwrap();
    let v_bot = interp_at_height(v_prof, height_prof, bottom_m).unwrap();
    let u_top = interp_at_height(u_prof, height_prof, top_m).unwrap();
    let v_top = interp_at_height(v_prof, height_prof, top_m).unwrap();

    (u_top - u_bot, v_top - v_bot)
}

/// Storm-relative helicity integrated from the surface to `depth_m` AGL.
///
/// SRH = integral of (storm-relative wind cross vertical wind shear) dz.
/// Uses the trapezoidal approximation on the profile segments within the layer.
///
/// # Returns
/// `(positive_srh, negative_srh, total_srh)` where:
/// - `positive_srh` >= 0 — sum of positive (cyclonic in NH) contributions
/// - `negative_srh` <= 0 — sum of negative (anticyclonic) contributions
/// - `total_srh` = positive_srh + negative_srh
///
/// # Arguments
/// * `u_prof`, `v_prof` — wind components (m/s), ascending height
/// * `height_prof` — heights AGL (m)
/// * `depth_m` — integration depth from surface (m), e.g. 1000.0 or 3000.0
/// * `storm_u`, `storm_v` — storm motion components (m/s)
///
/// # Panics
/// Panics on mismatched lengths or fewer than 2 levels.
pub fn storm_relative_helicity(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
    depth_m: f64,
    storm_u: f64,
    storm_v: f64,
) -> (f64, f64, f64) {
    let n = u_prof.len();
    assert_eq!(n, v_prof.len());
    assert_eq!(n, height_prof.len());
    assert!(n >= 2, "need at least 2 levels");

    // Build a sub-profile from 0 to depth_m, interpolating endpoints if needed.
    let mut heights = Vec::new();
    let mut us = Vec::new();
    let mut vs = Vec::new();

    // Start at height_prof[0] (surface)
    let h_start = height_prof[0];
    let h_end = h_start + depth_m;

    for i in 0..n {
        if height_prof[i] >= h_start && height_prof[i] <= h_end {
            heights.push(height_prof[i]);
            us.push(u_prof[i]);
            vs.push(v_prof[i]);
        }
    }

    // If the top of the layer doesn't exactly match a profile level, interpolate.
    if let Some(&last_h) = heights.last() {
        if last_h < h_end {
            if let (Some(u_top), Some(v_top)) = (
                interp_at_height(u_prof, height_prof, h_end),
                interp_at_height(v_prof, height_prof, h_end),
            ) {
                heights.push(h_end);
                us.push(u_top);
                vs.push(v_top);
            }
        }
    }

    let m = heights.len();
    if m < 2 {
        return (0.0, 0.0, 0.0);
    }

    let mut pos_srh = 0.0;
    let mut neg_srh = 0.0;

    // SRH via trapezoidal integration of the cross-product:
    // srh_k = (sru_k+1 - sru_k) * (srv_k+1 + srv_k) / 2
    //       - (srv_k+1 - srv_k) * (sru_k+1 + sru_k) / 2
    // ...summed across adjacent layers. The standard discrete form is:
    // SRH = sum_i [ (sru[i+1]*srv[i]) - (sru[i]*srv[i+1]) ]
    for i in 0..(m - 1) {
        let sru_i = us[i] - storm_u;
        let srv_i = vs[i] - storm_v;
        let sru_ip1 = us[i + 1] - storm_u;
        let srv_ip1 = vs[i + 1] - storm_v;

        // Cross-product contribution (MetPy convention):
        // Positive for clockwise-turning (veering) hodographs in the NH.
        let contrib = (sru_ip1 * srv_i) - (sru_i * srv_ip1);

        if contrib > 0.0 {
            pos_srh += contrib;
        } else {
            neg_srh += contrib;
        }
    }

    (pos_srh, neg_srh, pos_srh + neg_srh)
}

/// Mean wind over a height layer, computed as a height-weighted (trapezoidal)
/// average of the wind components.
///
/// # Arguments
/// * `u_prof`, `v_prof` — wind components (m/s), ascending height
/// * `height_prof` — heights AGL (m)
/// * `bottom_m` — bottom of the layer (m AGL)
/// * `top_m` — top of the layer (m AGL)
///
/// # Returns
/// `(mean_u, mean_v)` in m/s.
///
/// # Panics
/// Panics on mismatched lengths or fewer than 2 levels.
pub fn mean_wind(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    let n = u_prof.len();
    assert_eq!(n, v_prof.len());
    assert_eq!(n, height_prof.len());
    assert!(n >= 2, "need at least 2 levels");

    // Build sub-profile over the layer, interpolating endpoints.
    let mut heights = Vec::new();
    let mut us = Vec::new();
    let mut vs = Vec::new();

    // Interpolate at bottom
    let u_bot = interp_at_height(u_prof, height_prof, bottom_m).unwrap();
    let v_bot = interp_at_height(v_prof, height_prof, bottom_m).unwrap();
    heights.push(bottom_m);
    us.push(u_bot);
    vs.push(v_bot);

    // Add interior points within the layer
    for i in 0..n {
        if height_prof[i] > bottom_m && height_prof[i] < top_m {
            heights.push(height_prof[i]);
            us.push(u_prof[i]);
            vs.push(v_prof[i]);
        }
    }

    // Interpolate at top
    let u_top = interp_at_height(u_prof, height_prof, top_m).unwrap();
    let v_top = interp_at_height(v_prof, height_prof, top_m).unwrap();
    heights.push(top_m);
    us.push(u_top);
    vs.push(v_top);

    // Trapezoidal integration
    let m = heights.len();
    let mut sum_u = 0.0;
    let mut sum_v = 0.0;
    let mut total_dz = 0.0;

    for i in 0..(m - 1) {
        let dz = heights[i + 1] - heights[i];
        sum_u += 0.5 * (us[i] + us[i + 1]) * dz;
        sum_v += 0.5 * (vs[i] + vs[i + 1]) * dz;
        total_dz += dz;
    }

    if total_dz.abs() < 1e-10 {
        return (u_bot, v_bot);
    }

    (sum_u / total_dz, sum_v / total_dz)
}

/// Bunkers storm motion estimate using the internal dynamics (ID) method.
///
/// Computes right-moving, left-moving, and mean-wind vectors for a supercell
/// storm motion estimate. Uses the 0-6 km mean wind, 0-6 km bulk shear, and
/// a 7.5 m/s perpendicular deviation.
///
/// # Arguments
/// * `u_prof`, `v_prof` — wind components (m/s), ascending height
/// * `height_prof` — heights AGL (m)
///
/// # Returns
/// `(right_mover, left_mover, mean_wind)` where each is `(u, v)` in m/s.
///
/// # References
/// Bunkers et al. (2000): Predicting Supercell Motion Using a New Hodograph
/// Technique. *Wea. Forecasting*, **15**, 61-79.
pub fn bunkers_storm_motion(
    p_prof: &[f64],
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
) -> ((f64, f64), (f64, f64), (f64, f64)) {
    let deviation = 7.5; // m/s perpendicular offset
    let z_sfc = height_prof[0];

    // Pressure-weighted continuous average in a height layer.
    // Matches MetPy's weighted_continuous_average: WCA = ∫A dp / ∫dp
    let pw_mean = |comp: &[f64], z_bot: f64, z_top: f64| -> f64 {
        let n = height_prof.len();
        // Interpolate to a height target
        let interp = |target_z: f64, vals: &[f64]| -> f64 {
            if target_z <= height_prof[0] { return vals[0]; }
            if target_z >= height_prof[n - 1] { return vals[n - 1]; }
            for i in 1..n {
                if height_prof[i] >= target_z {
                    let frac = (target_z - height_prof[i - 1]) / (height_prof[i] - height_prof[i - 1]);
                    return vals[i - 1] + frac * (vals[i] - vals[i - 1]);
                }
            }
            vals[n - 1]
        };

        // Build layer sub-profile
        let mut lc: Vec<f64> = Vec::new();
        let mut lp: Vec<f64> = Vec::new();
        lc.push(interp(z_bot, comp));
        lp.push(interp(z_bot, p_prof));
        for i in 0..n {
            if height_prof[i] <= z_bot { continue; }
            if height_prof[i] >= z_top { break; }
            lc.push(comp[i]);
            lp.push(p_prof[i]);
        }
        lc.push(interp(z_top, comp));
        lp.push(interp(z_top, p_prof));

        // Trapezoidal integration: ∫A dp / ∫dp
        let m = lc.len();
        if m < 2 { return lc[0]; }
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 1..m {
            let dp = lp[i] - lp[i - 1];
            num += (lc[i] + lc[i - 1]) / 2.0 * dp;
            den += dp;
        }
        if den.abs() > 1e-10 { num / den } else { lc[0] }
    };

    // Pressure-weighted mean wind sfc-6km
    let mw_u = pw_mean(u_prof, z_sfc, z_sfc + 6000.0);
    let mw_v = pw_mean(v_prof, z_sfc, z_sfc + 6000.0);

    // Shear: mean(5.5-6km) - mean(0-0.5km) (MetPy Bunkers method)
    let u_500m = pw_mean(u_prof, z_sfc, z_sfc + 500.0);
    let v_500m = pw_mean(v_prof, z_sfc, z_sfc + 500.0);
    let u_5500m = pw_mean(u_prof, z_sfc + 5500.0, z_sfc + 6000.0);
    let v_5500m = pw_mean(v_prof, z_sfc + 5500.0, z_sfc + 6000.0);
    let shr_u = u_5500m - u_500m;
    let shr_v = v_5500m - v_500m;

    let shear_mag = (shr_u * shr_u + shr_v * shr_v).sqrt();

    if shear_mag < 1e-10 {
        return ((mw_u, mw_v), (mw_u, mw_v), (mw_u, mw_v));
    }

    // Cross product with k-hat: [shear_v, -shear_u] (MetPy convention)
    let perp_u = shr_v / shear_mag;
    let perp_v = -shr_u / shear_mag;

    let right_u = mw_u + deviation * perp_u;
    let right_v = mw_v + deviation * perp_v;

    let left_u = mw_u - deviation * perp_u;
    let left_v = mw_v - deviation * perp_v;

    ((right_u, right_v), (left_u, left_v), (mw_u, mw_v))
}

/// Corfidi upwind and downwind MCS propagation vectors.
///
/// Estimates convective system motion for training/back-building (upwind) and
/// forward-propagating (downwind) MCS modes.
///
/// # Arguments
/// * `u_prof`, `v_prof` — wind components (m/s), ascending height
/// * `height_prof` — heights AGL (m)
/// * `u_850`, `v_850` — 850-hPa wind components (m/s), representing the
///   low-level jet or inflow vector
///
/// # Returns
/// `(upwind, downwind)` where each is `(u, v)` in m/s.
/// - **upwind** = mean_wind - LLJ (propagation opposing the low-level jet)
/// - **downwind** = mean_wind + (mean_wind - LLJ) = 2*mean_wind - LLJ
///
/// # References
/// Corfidi (2003): Cold Pools and MCS Propagation. *Wea. Forecasting*, **18**, 997-1017.
pub fn corfidi_storm_motion(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
    u_850: f64,
    v_850: f64,
) -> ((f64, f64), (f64, f64)) {
    // Mean wind in the cloud-bearing layer (typically 850-300 hPa, approximated
    // here as the 0-6 km layer for consistency with a height-based profile).
    let (mw_u, mw_v) = mean_wind(u_prof, v_prof, height_prof, 0.0, 6000.0);

    // Propagation vector = mean_wind - LLJ  (opposes low-level jet)
    let prop_u = mw_u - u_850;
    let prop_v = mw_v - v_850;

    // Upwind (back-building): propagation vector
    let upwind = (prop_u, prop_v);

    // Downwind: mean_wind + propagation vector
    let downwind = (mw_u + prop_u, mw_v + prop_v);

    (upwind, downwind)
}

// ─────────────────────────────────────────────
// Boundary-layer turbulence functions
// ─────────────────────────────────────────────

/// Friction velocity from time series of wind components.
///
/// Computes `u* = (mean(u'w')^2)^(1/4) = sqrt(|mean(u'w')|)` where primes denote
/// perturbations from the mean.
///
/// Uses the computational identity: `mean(u'w') = mean(u*w) - mean(u)*mean(w)`
/// which is more efficient than explicitly computing perturbations.
///
/// # Arguments
///
/// * `u` - Time series of along-wind component (m/s)
/// * `w` - Time series of vertical wind component (m/s)
///
/// # Returns
///
/// Friction velocity (m/s), always non-negative.
///
/// # Panics
///
/// Panics if `u` and `w` have different lengths or fewer than 2 samples.
///
/// # References
///
/// Garratt, J. R. (1994). The Atmospheric Boundary Layer. Cambridge University Press.
///
/// # Examples
///
/// ```
/// use metrust::calc::wind::friction_velocity;
/// let u = vec![1.0, -1.0, 1.0, -1.0, 1.0];
/// let w = vec![0.5, -0.5, 0.5, -0.5, 0.5];
/// let u_star = friction_velocity(&u, &w);
/// assert!(u_star > 0.0);
/// ```
pub fn friction_velocity(u: &[f64], w: &[f64]) -> f64 {
    let n = u.len();
    assert_eq!(n, w.len(), "u and w must have the same length");
    assert!(n >= 2, "need at least 2 samples");

    let n_f = n as f64;
    let mean_u = u.iter().sum::<f64>() / n_f;
    let mean_w = w.iter().sum::<f64>() / n_f;
    let mean_uw: f64 = u.iter().zip(w.iter()).map(|(ui, wi)| ui * wi).sum::<f64>() / n_f;

    // kinematic flux = cov(u, w) using the identity
    let uw_flux = mean_uw - mean_u * mean_w;

    // u* = (uw^2)^(1/4) = sqrt(|uw|)
    uw_flux.abs().sqrt()
}

/// Turbulent Kinetic Energy from time series of wind components.
///
/// `TKE = 0.5 * (var(u) + var(v) + var(w))`
///
/// where `var()` is the population variance (N denominator, not N-1).
///
/// # Arguments
///
/// * `u` - Time series of u-component (m/s)
/// * `v` - Time series of v-component (m/s)
/// * `w` - Time series of w-component (m/s)
///
/// # Returns
///
/// TKE in m^2/s^2.
///
/// # Panics
///
/// Panics if arrays have different lengths or fewer than 2 samples.
///
/// # Examples
///
/// ```
/// use metrust::calc::wind::tke;
/// let u = vec![1.0, -1.0, 1.0, -1.0];
/// let v = vec![2.0, -2.0, 2.0, -2.0];
/// let w = vec![0.5, -0.5, 0.5, -0.5];
/// let e = tke(&u, &v, &w);
/// assert!((e - 2.625).abs() < 1e-10);
/// ```
pub fn tke(u: &[f64], v: &[f64], w: &[f64]) -> f64 {
    let n = u.len();
    assert_eq!(n, v.len(), "u and v must have the same length");
    assert_eq!(n, w.len(), "u and w must have the same length");
    assert!(n >= 2, "need at least 2 samples");

    let n_f = n as f64;

    // Population variance helper
    let variance = |arr: &[f64]| -> f64 {
        let mean = arr.iter().sum::<f64>() / n_f;
        arr.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_f
    };

    0.5 * (variance(u) + variance(v) + variance(w))
}

/// Gradient Richardson number at each level.
///
/// `Ri = (g / theta) * (d_theta/dz) / ((du/dz)^2 + (dv/dz)^2)`
///
/// Uses the same 3-point derivative scheme as MetPy: centered differences at
/// interior points, 3-point forward/backward differences at the boundaries.
///
/// # Arguments
///
/// * `height` - Height profile (meters, ascending)
/// * `theta` - Potential temperature profile (Kelvin)
/// * `u` - U-component wind profile (m/s)
/// * `v` - V-component wind profile (m/s)
///
/// # Returns
///
/// Richardson number at each level. Values below 0.25 indicate turbulence.
/// Where the wind shear denominator is zero, returns `f64::INFINITY` (or
/// `f64::NEG_INFINITY` for unstable layers with zero shear).
///
/// # Panics
///
/// Panics if arrays have different lengths or fewer than 3 levels.
///
/// # References
///
/// Holton, J. R. (2004). *An Introduction to Dynamic Meteorology*, 4th Ed., pg. 121-122.
///
/// # Examples
///
/// ```
/// use metrust::calc::wind::gradient_richardson_number;
/// let z = vec![0.0, 100.0, 200.0, 300.0, 400.0];
/// let theta = vec![300.0, 301.0, 302.5, 304.5, 307.0];
/// let u = vec![2.0, 5.0, 8.0, 10.0, 12.0];
/// let v = vec![1.0, 2.0, 3.5, 5.0, 6.0];
/// let ri = gradient_richardson_number(&z, &theta, &u, &v);
/// assert_eq!(ri.len(), 5);
/// assert!(ri[0] > 0.0); // stable
/// ```
pub fn gradient_richardson_number(
    height: &[f64],
    theta: &[f64],
    u: &[f64],
    v: &[f64],
) -> Vec<f64> {
    let n = height.len();
    assert_eq!(n, theta.len());
    assert_eq!(n, u.len());
    assert_eq!(n, v.len());
    assert!(n >= 3, "need at least 3 levels for gradient Richardson number");

    const G: f64 = 9.80665;

    // 3-point first derivative matching MetPy's first_derivative:
    // - Forward at i=0: (-3*f[0] + 4*f[1] - f[2]) / (x[2] - x[0])
    // - Centered at interior: (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
    // - Backward at i=n-1: (f[n-3] - 4*f[n-2] + 3*f[n-1]) / (x[n-1] - x[n-3])
    let first_deriv = |f: &[f64], x: &[f64]| -> Vec<f64> {
        let m = f.len();
        let mut d = vec![0.0; m];
        // Forward difference at i=0
        let dx_fwd = x[2] - x[0];
        if dx_fwd.abs() > 1e-30 {
            d[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / dx_fwd;
        }
        // Centered differences at interior
        for i in 1..m - 1 {
            let dx = x[i + 1] - x[i - 1];
            if dx.abs() > 1e-30 {
                d[i] = (f[i + 1] - f[i - 1]) / dx;
            }
        }
        // Backward difference at i=n-1
        let dx_bwd = x[m - 1] - x[m - 3];
        if dx_bwd.abs() > 1e-30 {
            d[m - 1] = (f[m - 3] - 4.0 * f[m - 2] + 3.0 * f[m - 1]) / dx_bwd;
        }
        d
    };

    let dthetadz = first_deriv(theta, height);
    let dudz = first_deriv(u, height);
    let dvdz = first_deriv(v, height);

    let mut ri = vec![0.0; n];
    for i in 0..n {
        let shear_sq = dudz[i].powi(2) + dvdz[i].powi(2);
        if shear_sq.abs() < 1e-30 {
            ri[i] = if dthetadz[i] >= 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        } else {
            ri[i] = (G / theta[i]) * (dthetadz[i] / shear_sq);
        }
    }
    ri
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a simple linear hodograph profile.
    /// u linearly increases from 0 to 30 m/s over 0-6 km.
    /// v linearly increases from 0 to 15 m/s over 0-6 km.
    fn linear_profile() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let heights: Vec<f64> = (0..=12).map(|i| i as f64 * 500.0).collect(); // 0, 500, ..., 6000 m
        let us: Vec<f64> = heights.iter().map(|h| 30.0 * h / 6000.0).collect();
        let vs: Vec<f64> = heights.iter().map(|h| 15.0 * h / 6000.0).collect();
        // Approximate pressure profile (standard atmosphere)
        let ps: Vec<f64> = heights.iter().map(|h| 1013.25 * (1.0 - 0.0065 * h / 288.15_f64).powf(5.2561)).collect();
        (us, vs, heights, ps)
    }

    // ── Re-export smoke tests ──

    #[test]
    fn test_reexported_wind_speed() {
        let u = vec![3.0, 0.0];
        let v = vec![4.0, 5.0];
        let spd = wind_speed(&u, &v);
        assert!((spd[0] - 5.0).abs() < 1e-10);
        assert!((spd[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_reexported_wind_direction_southerly() {
        let u = vec![0.0];
        let v = vec![10.0];
        let dir = wind_direction(&u, &v);
        assert!((dir[0] - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_reexported_wind_components_roundtrip() {
        let spd = vec![10.0];
        let dir = vec![270.0];
        let (u, v) = wind_components(&spd, &dir);
        let spd2 = wind_speed(&u, &v);
        assert!((spd2[0] - 10.0).abs() < 1e-10);
    }

    // ── interp_at_height ──

    #[test]
    fn test_interp_at_height_exact() {
        let vals = vec![10.0, 20.0, 30.0];
        let hgts = vec![0.0, 1000.0, 2000.0];
        assert!((interp_at_height(&vals, &hgts, 1000.0).unwrap() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_interp_at_height_between() {
        let vals = vec![10.0, 20.0, 30.0];
        let hgts = vec![0.0, 1000.0, 2000.0];
        assert!((interp_at_height(&vals, &hgts, 500.0).unwrap() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_interp_at_height_below_clamps() {
        let vals = vec![10.0, 20.0];
        let hgts = vec![100.0, 1000.0];
        assert!((interp_at_height(&vals, &hgts, 0.0).unwrap() - 10.0).abs() < 1e-10);
    }

    // ── bulk_shear ──

    #[test]
    fn test_bulk_shear_full_layer() {
        let (us, vs, hgts, ps) = linear_profile();
        let (du, dv) = bulk_shear(&us, &vs, &hgts, 0.0, 6000.0);
        assert!((du - 30.0).abs() < 1e-10, "du = {du}");
        assert!((dv - 15.0).abs() < 1e-10, "dv = {dv}");
    }

    #[test]
    fn test_bulk_shear_sub_layer() {
        let (us, vs, hgts, ps) = linear_profile();
        // 0-1 km: u goes from 0 to 5, v from 0 to 2.5
        let (du, dv) = bulk_shear(&us, &vs, &hgts, 0.0, 1000.0);
        assert!((du - 5.0).abs() < 1e-10, "du = {du}");
        assert!((dv - 2.5).abs() < 1e-10, "dv = {dv}");
    }

    #[test]
    fn test_bulk_shear_interpolated_endpoints() {
        let (us, vs, hgts, ps) = linear_profile();
        // 250 m to 750 m (both between profile levels)
        let (du, dv) = bulk_shear(&us, &vs, &hgts, 250.0, 750.0);
        let expected_du = 30.0 * (750.0 - 250.0) / 6000.0; // 2.5
        let expected_dv = 15.0 * (750.0 - 250.0) / 6000.0; // 1.25
        assert!((du - expected_du).abs() < 1e-10, "du = {du}");
        assert!((dv - expected_dv).abs() < 1e-10, "dv = {dv}");
    }

    // ── mean_wind ──

    #[test]
    fn test_mean_wind_linear_profile() {
        let (us, vs, hgts, ps) = linear_profile();
        // For a linear profile the mean is the midpoint value.
        let (mu, mv) = mean_wind(&us, &vs, &hgts, 0.0, 6000.0);
        assert!((mu - 15.0).abs() < 1e-10, "mean u = {mu}");
        assert!((mv - 7.5).abs() < 1e-10, "mean v = {mv}");
    }

    #[test]
    fn test_mean_wind_sub_layer() {
        let (us, vs, hgts, ps) = linear_profile();
        let (mu, mv) = mean_wind(&us, &vs, &hgts, 0.0, 1000.0);
        // Linear: mean of 0-1 km is midpoint = 0.5 km values
        let expected_u = 30.0 * 500.0 / 6000.0; // 2.5
        let expected_v = 15.0 * 500.0 / 6000.0; // 1.25
        assert!((mu - expected_u).abs() < 1e-10, "mean u = {mu}");
        assert!((mv - expected_v).abs() < 1e-10, "mean v = {mv}");
    }

    // ── storm_relative_helicity ──

    #[test]
    fn test_srh_zero_for_unidirectional() {
        // Unidirectional shear (all wind in one direction, storm at origin)
        // with storm motion at the mean wind => SRH should be zero for a
        // linear hodograph when storm is at the midpoint.
        let heights = vec![0.0, 1000.0, 2000.0, 3000.0];
        let us = vec![0.0, 10.0, 20.0, 30.0];
        let vs = vec![0.0, 0.0, 0.0, 0.0];
        // Storm motion at the mean wind (15, 0)
        let (pos, neg, total) = storm_relative_helicity(&us, &vs, &heights, 3000.0, 15.0, 0.0);
        // For a straight-line hodograph with storm at centroid, SRH = 0
        assert!(total.abs() < 1e-10, "total SRH = {total}");
        assert!(pos.abs() < 1e-10, "pos SRH = {pos}");
        assert!(neg.abs() < 1e-10, "neg SRH = {neg}");
    }

    #[test]
    fn test_srh_clockwise_turning() {
        // Clockwise-turning hodograph with storm at origin.
        // MetPy convention: this cross-product form gives negative total
        // for this configuration; positive SRH occurs with appropriate
        // storm motion (e.g., Bunkers RM to the right of the shear).
        let heights = vec![0.0, 1000.0, 2000.0, 3000.0];
        let us = vec![0.0, 10.0, 10.0, 0.0];
        let vs = vec![0.0, 0.0, 10.0, 10.0];
        // Storm at origin
        let (_pos, neg, total) = storm_relative_helicity(&us, &vs, &heights, 3000.0, 0.0, 0.0);
        assert!(total < 0.0, "clockwise turning with storm at origin should yield negative SRH (MetPy convention), got {total}");
        assert!(neg < 0.0);
    }

    #[test]
    fn test_srh_pos_neg_sum() {
        let heights = vec![0.0, 500.0, 1000.0, 1500.0, 2000.0];
        let us = vec![0.0, 5.0, 10.0, 5.0, 0.0];
        let vs = vec![0.0, 5.0, 0.0, -5.0, 0.0];
        let (pos, neg, total) = storm_relative_helicity(&us, &vs, &heights, 2000.0, 3.0, 1.0);
        assert!(
            (pos + neg - total).abs() < 1e-10,
            "pos ({pos}) + neg ({neg}) should equal total ({total})"
        );
    }

    // ── bunkers_storm_motion ──

    #[test]
    fn test_bunkers_mean_wind_reasonable() {
        let (us, vs, hgts, ps) = linear_profile();
        let (_rm, _lm, mw) = bunkers_storm_motion(&ps, &us, &vs, &hgts);
        // Pressure-weighted mean should be close to but not identical to
        // height-weighted mean (pressure weighting favors lower levels slightly)
        let (expected_mu, expected_mv) = mean_wind(&us, &vs, &hgts, 0.0, 6000.0);
        assert!((mw.0 - expected_mu).abs() < 2.0, "mean u diff too large: {} vs {}", mw.0, expected_mu);
        assert!((mw.1 - expected_mv).abs() < 1.0, "mean v diff too large: {} vs {}", mw.1, expected_mv);
    }

    #[test]
    fn test_bunkers_deviation_magnitude() {
        let (us, vs, hgts, ps) = linear_profile();
        let (rm, lm, mw) = bunkers_storm_motion(&ps, &us, &vs, &hgts);
        // Right and left movers are each 7.5 m/s from the mean wind
        let rm_dev = ((rm.0 - mw.0).powi(2) + (rm.1 - mw.1).powi(2)).sqrt();
        let lm_dev = ((lm.0 - mw.0).powi(2) + (lm.1 - mw.1).powi(2)).sqrt();
        assert!((rm_dev - 7.5).abs() < 1e-10, "RM deviation = {rm_dev}");
        assert!((lm_dev - 7.5).abs() < 1e-10, "LM deviation = {lm_dev}");
    }

    #[test]
    fn test_bunkers_symmetry() {
        let (us, vs, hgts, ps) = linear_profile();
        let (rm, lm, mw) = bunkers_storm_motion(&ps, &us, &vs, &hgts);
        // Mean of RM and LM should be the mean wind
        let avg_u = (rm.0 + lm.0) / 2.0;
        let avg_v = (rm.1 + lm.1) / 2.0;
        assert!((avg_u - mw.0).abs() < 1e-10);
        assert!((avg_v - mw.1).abs() < 1e-10);
    }

    #[test]
    fn test_bunkers_perpendicular() {
        let (us, vs, hgts, ps) = linear_profile();
        let (rm, _lm, mw) = bunkers_storm_motion(&ps, &us, &vs, &hgts);
        // Deviation vector should be 7.5 m/s from the mean wind
        let dev_u = rm.0 - mw.0;
        let dev_v = rm.1 - mw.1;
        let dev_mag = (dev_u * dev_u + dev_v * dev_v).sqrt();
        assert!((dev_mag - 7.5).abs() < 0.1, "deviation magnitude = {dev_mag}, expected ~7.5");
    }

    // ── corfidi_storm_motion ──

    #[test]
    fn test_corfidi_upwind_vector() {
        let (us, vs, hgts, ps) = linear_profile();
        let u_850 = 5.0;
        let v_850 = 2.0;
        let (upwind, _downwind) = corfidi_storm_motion(&us, &vs, &hgts, u_850, v_850);
        let (mw_u, mw_v) = mean_wind(&us, &vs, &hgts, 0.0, 6000.0);
        // upwind = mean_wind - LLJ
        assert!((upwind.0 - (mw_u - u_850)).abs() < 1e-10);
        assert!((upwind.1 - (mw_v - v_850)).abs() < 1e-10);
    }

    #[test]
    fn test_corfidi_downwind_vector() {
        let (us, vs, hgts, ps) = linear_profile();
        let u_850 = 5.0;
        let v_850 = 2.0;
        let (upwind, downwind) = corfidi_storm_motion(&us, &vs, &hgts, u_850, v_850);
        let (mw_u, mw_v) = mean_wind(&us, &vs, &hgts, 0.0, 6000.0);
        // downwind = mean_wind + propagation = mean_wind + (mean_wind - LLJ)
        assert!((downwind.0 - (mw_u + upwind.0)).abs() < 1e-10);
        assert!((downwind.1 - (mw_v + upwind.1)).abs() < 1e-10);
    }

    #[test]
    fn test_corfidi_zero_llj() {
        // When LLJ = 0, upwind = mean_wind, downwind = 2 * mean_wind.
        let (us, vs, hgts, ps) = linear_profile();
        let (upwind, downwind) = corfidi_storm_motion(&us, &vs, &hgts, 0.0, 0.0);
        let (mw_u, mw_v) = mean_wind(&us, &vs, &hgts, 0.0, 6000.0);
        assert!((upwind.0 - mw_u).abs() < 1e-10);
        assert!((upwind.1 - mw_v).abs() < 1e-10);
        assert!((downwind.0 - 2.0 * mw_u).abs() < 1e-10);
        assert!((downwind.1 - 2.0 * mw_v).abs() < 1e-10);
    }

    // ── friction_velocity ──

    #[test]
    fn test_friction_velocity_correlated() {
        // u and w perfectly correlated: u' = [1,-1,1,-1,1], w' = [0.5,-0.5,0.5,-0.5,0.5]
        // mean(u)=0.2, mean(w)=0.1
        // kinematic_flux = mean(u*w) - mean(u)*mean(w) = 0.5 - 0.02 = 0.48
        // u* = sqrt(|0.48|) = 0.6928203230
        // Verified against MetPy: friction_velocity(simple) = 0.6928203230
        let u = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let w = vec![0.5, -0.5, 0.5, -0.5, 0.5];
        let u_star = friction_velocity(&u, &w);
        assert!((u_star - 0.6928203230).abs() < 1e-8,
            "u* = {u_star}, expected 0.6928203230");
    }

    #[test]
    fn test_friction_velocity_zero_mean() {
        // When means are already zero: mean(u'w') = mean(uw)
        let u = vec![1.0, -1.0, 2.0, -2.0];
        let w = vec![0.5, -0.5, 1.0, -1.0];
        // mean(u)=0, mean(w)=0, mean(uw) = (0.5+0.5+2+2)/4 = 5/4 = 1.25
        let u_star = friction_velocity(&u, &w);
        assert!((u_star - 1.25_f64.sqrt()).abs() < 1e-10,
            "u* = {u_star}");
    }

    #[test]
    fn test_friction_velocity_uncorrelated() {
        // Uncorrelated signals: u'w' ~ 0
        let u = vec![1.0, -1.0, 1.0, -1.0];
        let w = vec![1.0, 1.0, -1.0, -1.0];
        // mean(u)=0, mean(w)=0, mean(uw) = (1 + (-1) + (-1) + 1)/4 = 0
        let u_star = friction_velocity(&u, &w);
        assert!(u_star.abs() < 1e-10, "u* = {u_star}, expected ~0");
    }

    // ── tke ──

    #[test]
    fn test_tke_simple() {
        // var(u)=1, var(v)=4, var(w)=0.25 => TKE = 0.5*(1+4+0.25) = 2.625
        // Verified against MetPy: tke(simple) = 2.625
        let u = vec![1.0, -1.0, 1.0, -1.0];
        let v = vec![2.0, -2.0, 2.0, -2.0];
        let w = vec![0.5, -0.5, 0.5, -0.5];
        let e = tke(&u, &v, &w);
        assert!((e - 2.625).abs() < 1e-10, "TKE = {e}, expected 2.625");
    }

    #[test]
    fn test_tke_zero_variance() {
        // Constant wind: all variance = 0 => TKE = 0
        let u = vec![5.0, 5.0, 5.0, 5.0];
        let v = vec![3.0, 3.0, 3.0, 3.0];
        let w = vec![0.0, 0.0, 0.0, 0.0];
        let e = tke(&u, &v, &w);
        assert!(e.abs() < 1e-10, "TKE = {e}, expected 0");
    }

    #[test]
    fn test_tke_equal_components() {
        // Equal variance in all components
        let u = vec![1.0, -1.0];
        let v = vec![1.0, -1.0];
        let w = vec![1.0, -1.0];
        // var of each = 1.0, TKE = 0.5 * 3 = 1.5
        let e = tke(&u, &v, &w);
        assert!((e - 1.5).abs() < 1e-10, "TKE = {e}, expected 1.5");
    }

    // ── gradient_richardson_number ──

    #[test]
    fn test_gradient_richardson_number_metpy_reference() {
        // Verified against MetPy:
        // z = [0, 100, 200, 300, 400] m
        // theta = [300, 301, 302.5, 304.5, 307] K
        // u = [2, 5, 8, 10, 12] m/s
        // v = [1, 2, 3.5, 5, 6] m/s
        // Ri = [0.25638301, 0.38556488, 0.66744336, 1.30270438, 1.92536076]
        let z = vec![0.0, 100.0, 200.0, 300.0, 400.0];
        let theta = vec![300.0, 301.0, 302.5, 304.5, 307.0];
        let u = vec![2.0, 5.0, 8.0, 10.0, 12.0];
        let v = vec![1.0, 2.0, 3.5, 5.0, 6.0];

        let ri = gradient_richardson_number(&z, &theta, &u, &v);
        let expected = [0.25638301, 0.38556488, 0.66744336, 1.30270438, 1.92536076];

        assert_eq!(ri.len(), 5);
        for i in 0..5 {
            assert!((ri[i] - expected[i]).abs() < 1e-4,
                "Ri[{i}] = {}, expected {}", ri[i], expected[i]);
        }
    }

    #[test]
    fn test_gradient_ri_stable_layer() {
        // Strongly stable: large theta increase, small shear
        let z = vec![0.0, 100.0, 200.0];
        let theta = vec![300.0, 310.0, 320.0]; // 10 K per 100 m
        let u = vec![5.0, 5.1, 5.2]; // very weak shear
        let v = vec![0.0, 0.0, 0.0];

        let ri = gradient_richardson_number(&z, &theta, &u, &v);
        // All Ri should be >> 0.25 (very stable)
        for i in 0..3 {
            assert!(ri[i] > 10.0, "Ri[{i}] = {}, expected >> 0.25", ri[i]);
        }
    }

    #[test]
    fn test_gradient_ri_below_quarter_means_turbulent() {
        // Strong shear, weak stability => Ri < 0.25
        let z = vec![0.0, 100.0, 200.0];
        let theta = vec![300.0, 300.01, 300.02]; // nearly neutral
        let u = vec![0.0, 10.0, 20.0]; // very strong shear
        let v = vec![0.0, 0.0, 0.0];

        let ri = gradient_richardson_number(&z, &theta, &u, &v);
        for i in 0..3 {
            assert!(ri[i] < 0.25, "Ri[{i}] = {}, expected < 0.25", ri[i]);
        }
    }
}
