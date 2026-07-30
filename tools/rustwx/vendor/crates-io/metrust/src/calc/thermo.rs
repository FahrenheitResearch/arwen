//! Thermodynamic calculations — MetPy-compatible API.
//!
//! This module provides a MetPy-compatible interface to meteorological thermodynamic
//! functions. Functions that exist in `wx_math::thermo` are re-exported (with name
//! adapters where the MetPy name differs). Functions not present in wx-math are
//! implemented here directly.
//!
//! ## Conventions
//!
//! - **Temperatures**: Celsius unless otherwise noted.
//! - **Pressures**: hPa (millibars).
//! - **Profiles**: Slices sorted surface-first (highest pressure first, decreasing).
//! - **Mixing ratio**: g/kg.
//! - **Relative humidity**: percent (0–100).

// ============================================================================
// Direct re-exports (API name matches wx-math name)
// ============================================================================

pub use wx_math::thermo::potential_temperature;
pub use wx_math::thermo::equivalent_potential_temperature;
pub use wx_math::thermo::saturation_vapor_pressure;
pub use wx_math::thermo::saturation_vapor_pressure_with_phase;
pub use wx_math::thermo::saturation_mixing_ratio;
pub use wx_math::thermo::saturation_mixing_ratio_with_phase;
pub use wx_math::thermo::Phase;
pub use wx_math::thermo::wet_bulb_temperature;
pub use wx_math::thermo::lfc;
pub use wx_math::thermo::el;
pub use wx_math::thermo::ccl;
pub use wx_math::thermo::lifted_index;
pub use wx_math::thermo::density;

// Re-exports keeping the wx-math original names available — required by calc::mod.rs
// and by callers that use the internal naming convention.
pub use wx_math::thermo::dewpoint_from_rh;
pub use wx_math::thermo::rh_from_dewpoint;
pub use wx_math::thermo::mixing_ratio_from_specific_humidity;
pub use wx_math::thermo::lcl_pressure;
pub use wx_math::thermo::parcel_profile;
pub use wx_math::thermo::dry_lapse;
pub use wx_math::thermo::moist_lapse;
pub use wx_math::thermo::celsius_to_kelvin;
pub use wx_math::thermo::kelvin_to_celsius;
pub use wx_math::thermo::celsius_to_fahrenheit;
pub use wx_math::thermo::fahrenheit_to_celsius;
pub use wx_math::thermo::virtual_temp;
pub use wx_math::thermo::thetae;
pub use wx_math::thermo::cape_cin_core;

// Re-exports: energy, stability, and vertical coordinate functions
pub use wx_math::thermo::temperature_from_potential_temperature;
pub use wx_math::thermo::dry_static_energy;
pub use wx_math::thermo::moist_static_energy;
pub use wx_math::thermo::static_stability;
pub use wx_math::thermo::scale_height;
pub use wx_math::thermo::mean_pressure_weighted;
pub use wx_math::thermo::geopotential_to_height;
pub use wx_math::thermo::height_to_geopotential;
pub use wx_math::thermo::vertical_velocity;
pub use wx_math::thermo::vertical_velocity_pressure;
pub use wx_math::thermo::exner_function;
pub use wx_math::thermo::montgomery_streamfunction;

// Re-exports: humidity conversions
pub use wx_math::thermo::dewpoint_from_vapor_pressure;
pub use wx_math::thermo::mixing_ratio_from_relative_humidity;
pub use wx_math::thermo::relative_humidity_from_mixing_ratio;
pub use wx_math::thermo::relative_humidity_from_specific_humidity;
pub use wx_math::thermo::specific_humidity_from_dewpoint;
pub use wx_math::thermo::dewpoint_from_specific_humidity;
pub use wx_math::thermo::specific_humidity;
pub use wx_math::thermo::frost_point;
pub use wx_math::thermo::psychrometric_vapor_pressure;

// Re-exports: potential temperature variants
pub use wx_math::thermo::virtual_potential_temperature;
pub use wx_math::thermo::wet_bulb_potential_temperature;
pub use wx_math::thermo::saturation_equivalent_potential_temperature;

// Re-exports: layer / intersection utilities
pub use wx_math::thermo::thickness_hypsometric;
pub use wx_math::thermo::find_intersections;
pub use wx_math::thermo::get_layer;
pub use wx_math::thermo::get_layer_heights;
pub use wx_math::thermo::reduce_point_density;
pub use wx_math::thermo::isentropic_interpolation;

// Re-exports: PV
pub use wx_math::thermo::potential_vorticity_baroclinic;

// Re-exports: CAPE/CIN convenience wrappers and parcel selectors
pub use wx_math::thermo::surface_based_cape_cin;
pub use wx_math::thermo::mixed_layer_cape_cin;
pub use wx_math::thermo::most_unstable_cape_cin;
pub use wx_math::thermo::mixed_layer;
pub use wx_math::thermo::get_mixed_layer_parcel;
pub use wx_math::thermo::get_most_unstable_parcel;
pub use wx_math::thermo::galvez_davison_index;

// Re-export constants that callers may need.
pub use wx_math::thermo::{G, RD, CP, ROCP, ZEROCNK, EPS, LV};

// ============================================================================
// Wrapper re-exports (MetPy name differs from wx-math name)
// ============================================================================

/// Mixing ratio (g/kg) from pressure (hPa) and temperature (Celsius).
///
/// Wraps [`wx_math::thermo::mixratio`]. Includes the Wexler enhancement factor
/// for non-ideal gas behavior.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::mixing_ratio;
/// let w = mixing_ratio(1013.25, 20.0);
/// assert!(w > 14.0 && w < 15.0);
/// ```
#[inline]
pub fn mixing_ratio(p: f64, t: f64) -> f64 {
    wx_math::thermo::mixratio(p, t)
}

/// Dewpoint (Celsius) from temperature (Celsius) and relative humidity (%).
///
/// Wraps [`wx_math::thermo::dewpoint_from_rh`].
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::dewpoint_from_relative_humidity;
/// let td = dewpoint_from_relative_humidity(25.0, 50.0);
/// assert!((td - 13.9).abs() < 0.5);
/// ```
#[inline]
pub fn dewpoint_from_relative_humidity(t_c: f64, rh: f64) -> f64 {
    wx_math::thermo::dewpoint_from_rh(t_c, rh)
}

/// Relative humidity (%) from temperature and dewpoint (both Celsius).
///
/// Wraps [`wx_math::thermo::rh_from_dewpoint`].
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::relative_humidity_from_dewpoint;
/// let rh = relative_humidity_from_dewpoint(25.0, 15.0);
/// assert!(rh > 50.0 && rh < 60.0);
/// ```
#[inline]
pub fn relative_humidity_from_dewpoint(t_c: f64, td_c: f64) -> f64 {
    wx_math::thermo::rh_from_dewpoint(t_c, td_c)
}

/// Vapor pressure (hPa) from dewpoint temperature (Celsius).
///
/// Wraps [`wx_math::thermo::vapor_pressure_from_dewpoint`].
/// Uses the Bolton (1980) saturation vapor pressure formula.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::vapor_pressure;
/// let e = vapor_pressure(20.0);
/// assert!((e - 23.37).abs() < 0.5);
/// ```
#[inline]
pub fn vapor_pressure(td_c: f64) -> f64 {
    wx_math::thermo::vapor_pressure_from_dewpoint(td_c)
}

/// Virtual temperature (Celsius) from temperature (C), pressure (hPa), and dewpoint (C).
///
/// Wraps [`wx_math::thermo::virtual_temp`]. Note the argument order matches MetPy:
/// `(t, p, td)` rather than wx-math's `(t, p, td)` — identical in this case.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::virtual_temperature;
/// let tv = virtual_temperature(25.0, 1000.0, 20.0);
/// assert!(tv > 25.0);
/// ```
#[inline]
pub fn virtual_temperature(t: f64, p: f64, td: f64) -> f64 {
    wx_math::thermo::virtual_temp(t, p, td)
}

/// Virtual temperature (Celsius) from temperature (C), dewpoint (C), and pressure (hPa).
///
/// Wraps [`wx_math::thermo::virtual_temperature_from_dewpoint`]. Argument order
/// matches MetPy: `(temperature, dewpoint, pressure)`.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::virtual_temperature_from_dewpoint;
/// let tv = virtual_temperature_from_dewpoint(25.0, 20.0, 1000.0);
/// assert!(tv > 25.0);
/// ```
#[inline]
pub fn virtual_temperature_from_dewpoint(t_c: f64, td_c: f64, p_hpa: f64) -> f64 {
    wx_math::thermo::virtual_temperature_from_dewpoint(t_c, td_c, p_hpa)
}

/// Lifting Condensation Level via dry-adiabatic ascent.
///
/// Returns `(p_lcl, t_lcl)` in `(hPa, Celsius)`.
///
/// Wraps [`wx_math::thermo::drylift`].
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::lcl;
/// let (p_lcl, t_lcl) = lcl(1000.0, 25.0, 15.0);
/// assert!(p_lcl < 1000.0 && p_lcl > 700.0);
/// assert!(t_lcl < 25.0);
/// ```
#[inline]
pub fn lcl(p: f64, t: f64, td: f64) -> (f64, f64) {
    wx_math::thermo::drylift(p, t, td)
}

/// CAPE and CIN for a sounding column.
///
/// Wraps [`wx_math::thermo::cape_cin_core`] which returns
/// `(cape, cin, h_lcl, h_lfc)` in `(J/kg, J/kg, m AGL, m AGL)`.
///
/// # Arguments
///
/// * `p_prof` - Pressure profile (hPa, surface first)
/// * `t_prof` - Temperature profile (Celsius)
/// * `td_prof` - Dewpoint profile (Celsius)
/// * `height_agl` - Height AGL profile (meters)
/// * `psfc` - Surface pressure (hPa)
/// * `t2m` - 2-meter temperature (Celsius)
/// * `td2m` - 2-meter dewpoint (Celsius)
/// * `parcel_type` - `"sb"`, `"ml"`, or `"mu"`
/// * `ml_depth` - Mixed-layer depth (hPa), typically 100
/// * `mu_depth` - Most-unstable search depth (hPa), typically 300
/// * `top_m` - Optional height cap for integration (meters AGL)
#[inline]
pub fn cape_cin(
    p_prof: &[f64],
    t_prof: &[f64],
    td_prof: &[f64],
    height_agl: &[f64],
    psfc: f64,
    t2m: f64,
    td2m: f64,
    parcel_type: &str,
    ml_depth: f64,
    mu_depth: f64,
    top_m: Option<f64>,
) -> (f64, f64, f64, f64) {
    wx_math::thermo::cape_cin_core(
        p_prof, t_prof, td_prof, height_agl, psfc, t2m, td2m, parcel_type, ml_depth, mu_depth,
        top_m,
    )
}

// ============================================================================
// New implementations — stability indices not present in wx-math
// ============================================================================

/// Showalter Index: lift a parcel from 850 hPa to 500 hPa and compare with environment.
///
/// Defined as `T_env(500) - T_parcel(500)` where the parcel originates at 850 hPa.
/// Negative values indicate instability; values below -3 suggest severe thunderstorm
/// potential.
///
/// # Arguments
///
/// * `p_prof` - Pressure profile (hPa, surface first, decreasing)
/// * `t_prof` - Temperature profile (Celsius)
/// * `td_prof` - Dewpoint profile (Celsius)
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::showalter_index;
/// let p  = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
/// let t  = vec![  30.0,  25.0,  20.0,  10.0, -10.0, -35.0];
/// let td = vec![  20.0,  15.0,  14.0,   2.0, -20.0, -45.0];
/// let si = showalter_index(&p, &t, &td);
/// assert!(si.is_finite());
/// ```
pub fn showalter_index(p_prof: &[f64], t_prof: &[f64], td_prof: &[f64]) -> f64 {
    // Interpolate environment at 850 hPa
    let (t850, td850) = wx_math::thermo::get_env_at_pres(850.0, p_prof, t_prof, td_prof);

    // Lift parcel from 850 to LCL, then moist-adiabatically to 500
    let (p_lcl, t_lcl) = wx_math::thermo::drylift(850.0, t850, td850);

    let t_parcel_500 = if 500.0 >= p_lcl {
        // 500 hPa is below or at the LCL — dry adiabat only
        let theta_k = (t850 + ZEROCNK) * (1000.0_f64 / 850.0).powf(ROCP);
        theta_k * (500.0_f64 / 1000.0).powf(ROCP) - ZEROCNK
    } else {
        // Moist ascent from LCL to 500 hPa
        let theta_k = (t_lcl + ZEROCNK) * (1000.0_f64 / p_lcl).powf(ROCP);
        let theta_c = theta_k - ZEROCNK;
        let thetam =
            theta_c - wx_math::thermo::wobf(theta_c) + wx_math::thermo::wobf(t_lcl);
        wx_math::thermo::satlift(500.0, thetam)
    };

    // Interpolate environment temperature at 500 hPa
    let (t_env_500, _) = wx_math::thermo::get_env_at_pres(500.0, p_prof, t_prof, td_prof);

    t_env_500 - t_parcel_500
}

/// K-Index: a measure of thunderstorm potential from standard-level temperatures.
///
/// `KI = (T850 - T500) + Td850 - (T700 - Td700)`
///
/// Values above 30 suggest moderate thunderstorm probability; above 40 suggests
/// very high probability.
///
/// All inputs in Celsius.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::k_index;
/// let ki = k_index(20.0, 14.0, 10.0, 2.0, -10.0);
/// assert!((ki - 36.0).abs() < 1e-10);
/// ```
#[inline]
pub fn k_index(t850: f64, td850: f64, t700: f64, td700: f64, t500: f64) -> f64 {
    (t850 - t500) + td850 - (t700 - td700)
}

/// Vertical Totals: `T850 - T500`.
///
/// Measures the static stability of the 850–500 hPa layer. Values above 26
/// suggest potential instability.
///
/// All inputs in Celsius.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::vertical_totals;
/// assert!((vertical_totals(20.0, -10.0) - 30.0).abs() < 1e-10);
/// ```
#[inline]
pub fn vertical_totals(t850: f64, t500: f64) -> f64 {
    t850 - t500
}

/// Cross Totals: `Td850 - T500`.
///
/// Measures low-level moisture combined with upper-level cold air. Values
/// above 18 suggest severe thunderstorm potential.
///
/// All inputs in Celsius.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::cross_totals;
/// assert!((cross_totals(14.0, -10.0) - 24.0).abs() < 1e-10);
/// ```
#[inline]
pub fn cross_totals(td850: f64, t500: f64) -> f64 {
    td850 - t500
}

/// Total Totals Index: sum of Vertical Totals and Cross Totals.
///
/// `TT = (T850 - T500) + (Td850 - T500)`
///
/// Values above 44 suggest thunderstorm potential; above 55 suggests severe
/// thunderstorm potential.
///
/// All inputs in Celsius.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::total_totals;
/// assert!((total_totals(20.0, 14.0, -10.0) - 54.0).abs() < 1e-10);
/// ```
#[inline]
pub fn total_totals(t850: f64, td850: f64, t500: f64) -> f64 {
    vertical_totals(t850, t500) + cross_totals(td850, t500)
}

/// SWEAT Index (Severe Weather Threat Index).
///
/// A composite index incorporating temperature, moisture, wind speed, and wind
/// direction at 850 and 500 hPa to estimate severe thunderstorm potential.
///
/// `SWEAT = 12*Td850 + 20*(TT - 49) + 2*ff850 + ff500 + 125*(sin(dd500-dd850) + 0.2)`
///
/// The `20*(TT-49)` term is set to zero if TT < 49. The shear term is set to
/// zero unless all of these conditions are met:
/// - 130 <= dd850 <= 250
/// - 210 <= dd500 <= 310
/// - dd500 - dd850 > 0
/// - ff850 >= 15 kt and ff500 >= 15 kt
///
/// # Arguments
///
/// * `t850` - Temperature at 850 hPa (Celsius)
/// * `td850` - Dewpoint at 850 hPa (Celsius)
/// * `t500` - Temperature at 500 hPa (Celsius)
/// * `dd850` - Wind direction at 850 hPa (degrees, meteorological convention)
/// * `dd500` - Wind direction at 500 hPa (degrees, meteorological convention)
/// * `ff850` - Wind speed at 850 hPa (knots)
/// * `ff500` - Wind speed at 500 hPa (knots)
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::sweat_index;
/// let sweat = sweat_index(20.0, 14.0, -10.0, 200.0, 250.0, 25.0, 40.0);
/// assert!(sweat > 0.0);
/// ```
pub fn sweat_index(
    t850: f64,
    td850: f64,
    t500: f64,
    dd850: f64,
    dd500: f64,
    ff850: f64,
    ff500: f64,
) -> f64 {
    let tt = total_totals(t850, td850, t500);

    // Term 1: Low-level moisture
    let term_moisture = 12.0 * td850.max(0.0);

    // Term 2: Total totals instability (only if TT >= 49)
    let term_tt = if tt >= 49.0 {
        20.0 * (tt - 49.0)
    } else {
        0.0
    };

    // Term 3: Low-level wind
    let term_wind_850 = 2.0 * ff850;

    // Term 4: Upper-level wind
    let term_wind_500 = ff500;

    // Term 5: Directional shear (only under specific veering conditions)
    let term_shear = if dd850 >= 130.0
        && dd850 <= 250.0
        && dd500 >= 210.0
        && dd500 <= 310.0
        && (dd500 - dd850) > 0.0
        && ff850 >= 15.0
        && ff500 >= 15.0
    {
        let angle_diff_rad = (dd500 - dd850).to_radians();
        125.0 * (angle_diff_rad.sin() + 0.2)
    } else {
        0.0
    };

    term_moisture + term_tt + term_wind_850 + term_wind_500 + term_shear
}

/// Downdraft CAPE (DCAPE).
///
/// Estimates the potential energy available for a downdraft by integrating
/// negative buoyancy from the level of minimum equivalent potential temperature
/// (in the lowest 400 hPa of the sounding) to the surface. The parcel descends
/// along a moist adiabat from that level.
///
/// Wraps [`wx_math::thermo::downdraft_cape`].
///
/// # Arguments
///
/// * `p_prof` - Pressure profile (hPa, surface first)
/// * `t_prof` - Temperature profile (Celsius)
/// * `td_prof` - Dewpoint profile (Celsius)
///
/// # Returns
///
/// DCAPE in J/kg (always non-negative).
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::downdraft_cape;
/// let p  = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
/// let t  = vec![  30.0,  25.0,  20.0,   5.0, -15.0, -40.0];
/// let td = vec![  20.0,  12.0,   5.0,  -5.0, -25.0, -50.0];
/// let dcape = downdraft_cape(&p, &t, &td);
/// assert!(dcape >= 0.0);
/// ```
#[inline]
pub fn downdraft_cape(p_prof: &[f64], t_prof: &[f64], td_prof: &[f64]) -> f64 {
    wx_math::thermo::downdraft_cape(p_prof, t_prof, td_prof)
}

/// Brunt-Vaisala frequency at each level.
///
/// Computes `N = sqrt(g/theta * d_theta/dz)` using height and potential temperature
/// profiles. Where the atmosphere is statically unstable (`N^2 < 0`), returns 0.
///
/// This is a direct-from-profiles implementation that takes height (m) and potential
/// temperature (K) arrays, unlike `wx_math::thermo::brunt_vaisala_frequency` which
/// takes pressure and absolute temperature arrays and derives height internally.
///
/// # Arguments
///
/// * `height` - Height profile (meters, increasing, surface first)
/// * `potential_temperature` - Potential temperature profile (Kelvin, matching heights)
///
/// # Returns
///
/// Brunt-Vaisala frequency `N` (s^-1) at each level.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::brunt_vaisala_frequency;
/// let z     = vec![0.0, 1000.0, 2000.0, 3000.0];
/// let theta = vec![300.0, 303.0, 306.0, 309.0];
/// let n = brunt_vaisala_frequency(&z, &theta);
/// assert_eq!(n.len(), 4);
/// assert!(n[1] > 0.0);
/// ```
pub fn brunt_vaisala_frequency(height: &[f64], potential_temperature: &[f64]) -> Vec<f64> {
    let n = height.len().min(potential_temperature.len());
    if n < 2 {
        return vec![0.0; n];
    }

    let mut result = vec![0.0; n];
    for i in 0..n {
        let (dtheta, dz) = if i == 0 {
            (
                potential_temperature[1] - potential_temperature[0],
                height[1] - height[0],
            )
        } else if i == n - 1 {
            (
                potential_temperature[n - 1] - potential_temperature[n - 2],
                height[n - 1] - height[n - 2],
            )
        } else {
            (
                potential_temperature[i + 1] - potential_temperature[i - 1],
                height[i + 1] - height[i - 1],
            )
        };

        if dz.abs() < 1e-10 || potential_temperature[i].abs() < 1e-10 {
            result[i] = 0.0;
        } else {
            let n_sq = (G / potential_temperature[i]) * (dtheta / dz);
            result[i] = if n_sq > 0.0 { n_sq.sqrt() } else { 0.0 };
        }
    }
    result
}

/// Brunt-Vaisala period at each level.
///
/// Computes `T = 2*pi / N` where `N` is the Brunt-Vaisala frequency. Where
/// `N = 0` (statically unstable), returns `f64::INFINITY`.
///
/// # Arguments
///
/// * `height` - Height profile (meters, increasing, surface first)
/// * `potential_temperature` - Potential temperature profile (Kelvin, matching heights)
///
/// # Returns
///
/// Period (seconds) at each level.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::brunt_vaisala_period;
/// let z     = vec![0.0, 1000.0, 2000.0, 3000.0];
/// let theta = vec![300.0, 303.0, 306.0, 309.0];
/// let period = brunt_vaisala_period(&z, &theta);
/// assert_eq!(period.len(), 4);
/// assert!(period[1] > 0.0 && period[1] < 1000.0);
/// ```
pub fn brunt_vaisala_period(height: &[f64], potential_temperature: &[f64]) -> Vec<f64> {
    let bvf = brunt_vaisala_frequency(height, potential_temperature);
    bvf.iter()
        .map(|&n| {
            if n <= 0.0 {
                f64::INFINITY
            } else {
                2.0 * std::f64::consts::PI / n
            }
        })
        .collect()
}

// ============================================================================
// New wrapper functions
// ============================================================================

/// Precipitable water (mm) from pressure and dewpoint profiles.
///
/// Integrates mixing ratio over the pressure column using the trapezoidal rule:
/// `PW = (1/g) * integral(w dp)`.
///
/// # Arguments
///
/// * `p_prof` - Pressure profile (hPa, surface first, decreasing)
/// * `td_prof` - Dewpoint profile (Celsius)
///
/// # Returns
///
/// Precipitable water in mm (equivalent to kg/m^2).
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::precipitable_water;
/// let p  = vec![1000.0, 925.0, 850.0, 700.0, 500.0];
/// let td = vec![20.0, 15.0, 10.0, 0.0, -20.0];
/// let pw = precipitable_water(&p, &td);
/// assert!(pw > 0.0);
/// ```
pub fn precipitable_water(p_prof: &[f64], td_prof: &[f64]) -> f64 {
    let n = p_prof.len().min(td_prof.len());
    if n < 2 {
        return 0.0;
    }

    // Compute mixing ratio (kg/kg) at each level from dewpoint
    let w: Vec<f64> = p_prof.iter().zip(td_prof.iter()).take(n)
        .map(|(&p, &td)| wx_math::thermo::mixratio(p, td) / 1000.0) // g/kg -> kg/kg
        .collect();

    // Trapezoidal integration: PW = (1/g) * sum( (w[i]+w[i+1])/2 * dp )
    let mut pw = 0.0;
    for i in 0..n - 1 {
        let dp = (p_prof[i] - p_prof[i + 1]) * 100.0; // hPa -> Pa
        let w_avg = (w[i] + w[i + 1]) / 2.0;
        pw += w_avg * dp;
    }
    // PW in kg/m^2 (= mm)
    pw / G
}

/// Brunt-Vaisala frequency squared at each level.
///
/// Computes `N^2 = (g/theta) * (d_theta/dz)` without taking the square root.
/// Can be negative for statically unstable layers, unlike [`brunt_vaisala_frequency`]
/// which clamps to zero.
///
/// # Arguments
///
/// * `height` - Height profile (meters, increasing, surface first)
/// * `potential_temperature` - Potential temperature profile (Kelvin)
///
/// # Returns
///
/// N^2 (s^-2) at each level.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::brunt_vaisala_frequency_squared;
/// let z     = vec![0.0, 1000.0, 2000.0, 3000.0];
/// let theta = vec![300.0, 303.0, 306.0, 309.0];
/// let n2 = brunt_vaisala_frequency_squared(&z, &theta);
/// assert!(n2[1] > 0.0); // stable
/// ```
pub fn brunt_vaisala_frequency_squared(height: &[f64], potential_temperature: &[f64]) -> Vec<f64> {
    let n = height.len().min(potential_temperature.len());
    if n < 2 {
        return vec![0.0; n];
    }

    let mut result = vec![0.0; n];
    for i in 0..n {
        let (dtheta, dz) = if i == 0 {
            (
                potential_temperature[1] - potential_temperature[0],
                height[1] - height[0],
            )
        } else if i == n - 1 {
            (
                potential_temperature[n - 1] - potential_temperature[n - 2],
                height[n - 1] - height[n - 2],
            )
        } else {
            (
                potential_temperature[i + 1] - potential_temperature[i - 1],
                height[i + 1] - height[i - 1],
            )
        };

        if dz.abs() < 1e-10 || potential_temperature[i].abs() < 1e-10 {
            result[i] = 0.0;
        } else {
            result[i] = (G / potential_temperature[i]) * (dtheta / dz);
        }
    }
    result
}

/// Parcel profile with the LCL level inserted.
///
/// Computes a parcel temperature profile from surface T and Td, inserting the
/// LCL pressure into the returned pressure array so that the dry-to-moist
/// transition is explicit.
///
/// # Arguments
///
/// * `p` - Pressure levels (hPa, surface first, decreasing)
/// * `t_surface_c` - Surface temperature (Celsius)
/// * `td_surface_c` - Surface dewpoint (Celsius)
///
/// # Returns
///
/// `(pressures_with_lcl, temperature_profile)` where the LCL level has been
/// inserted into the output arrays.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::parcel_profile_with_lcl;
/// let p = vec![1000.0, 900.0, 800.0, 700.0, 500.0];
/// let (p_out, t_out) = parcel_profile_with_lcl(&p, 25.0, 15.0);
/// assert!(p_out.len() >= p.len());
/// assert_eq!(p_out.len(), t_out.len());
/// ```
pub fn parcel_profile_with_lcl(
    p: &[f64],
    t_surface_c: f64,
    td_surface_c: f64,
) -> (Vec<f64>, Vec<f64>) {
    if p.is_empty() {
        return (vec![], vec![]);
    }

    let (p_lcl, t_lcl) = wx_math::thermo::drylift(p[0], t_surface_c, td_surface_c);

    // Build the augmented pressure array with LCL inserted
    let mut p_aug = Vec::with_capacity(p.len() + 1);
    let mut lcl_inserted = false;

    for &pi in p {
        if !lcl_inserted && pi <= p_lcl {
            // Insert LCL level before this point (unless it coincides)
            if (pi - p_lcl).abs() > 0.01 {
                p_aug.push(p_lcl);
            }
            lcl_inserted = true;
        }
        p_aug.push(pi);
    }
    // If LCL is below all profile levels (shouldn't happen for typical soundings)
    if !lcl_inserted {
        p_aug.push(p_lcl);
    }

    // Compute parcel temperature at each augmented pressure level
    let t_surface_k = t_surface_c + ZEROCNK;
    let p_surface = p[0];

    // Moist adiabat parameters
    let theta_k = (t_lcl + ZEROCNK) * (1000.0_f64 / p_lcl).powf(ROCP);
    let theta_c = theta_k - ZEROCNK;
    let thetam = theta_c - wx_math::thermo::wobf(theta_c) + wx_math::thermo::wobf(t_lcl);

    let mut t_aug = Vec::with_capacity(p_aug.len());
    for &pi in &p_aug {
        if pi > p_lcl {
            // Dry adiabat
            let t_k = t_surface_k * (pi / p_surface).powf(ROCP);
            t_aug.push(t_k - ZEROCNK);
        } else {
            // Moist adiabat
            t_aug.push(wx_math::thermo::satlift(pi, thetam));
        }
    }

    (p_aug, t_aug)
}

/// Hypsometric thickness (meters) between two pressure levels.
///
/// Wrapper around [`thickness_hypsometric`] using a mean layer temperature.
///
/// # Arguments
///
/// * `p_bottom` - Bottom pressure (hPa)
/// * `p_top` - Top pressure (hPa)
/// * `t_mean_k` - Mean virtual temperature of the layer (Kelvin)
///
/// # Returns
///
/// Layer thickness in meters.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::thickness_hydrostatic;
/// let dz = thickness_hydrostatic(1000.0, 500.0, 260.0);
/// assert!(dz > 5000.0 && dz < 6000.0);
/// ```
#[inline]
pub fn thickness_hydrostatic(p_bottom: f64, p_top: f64, t_mean_k: f64) -> f64 {
    wx_math::thermo::thickness_hypsometric(p_bottom, p_top, t_mean_k)
}

/// New pressure after ascending or descending by a height increment.
///
/// Uses the standard atmosphere to convert `p_hpa` to height, adds `delta_h_m`,
/// then converts back to pressure.
///
/// # Arguments
///
/// * `p_hpa` - Starting pressure (hPa)
/// * `delta_h_m` - Height change (meters, positive = up = lower pressure)
///
/// # Returns
///
/// New pressure (hPa).
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::add_height_to_pressure;
/// let p_new = add_height_to_pressure(1013.25, 1000.0);
/// assert!(p_new < 1013.25);
/// ```
#[inline]
pub fn add_height_to_pressure(p_hpa: f64, delta_h_m: f64) -> f64 {
    let h = wx_math::thermo::pressure_to_height_std(p_hpa);
    wx_math::thermo::height_to_pressure_std(h + delta_h_m)
}

/// New height after a pressure increment.
///
/// Uses the standard atmosphere to convert `h_m` to pressure, adds `delta_p_hpa`,
/// then converts back to height.
///
/// # Arguments
///
/// * `h_m` - Starting height (meters)
/// * `delta_p_hpa` - Pressure change (hPa, positive = increase = lower altitude)
///
/// # Returns
///
/// New height (meters).
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::add_pressure_to_height;
/// let h_new = add_pressure_to_height(0.0, -100.0);
/// assert!(h_new > 0.0);
/// ```
#[inline]
pub fn add_pressure_to_height(h_m: f64, delta_p_hpa: f64) -> f64 {
    let p = wx_math::thermo::height_to_pressure_std(h_m);
    wx_math::thermo::pressure_to_height_std(p + delta_p_hpa)
}

/// Perturbation (anomaly) from the mean.
///
/// Subtracts the arithmetic mean of `values` from each element.
///
/// # Arguments
///
/// * `values` - Input array
///
/// # Returns
///
/// `values[i] - mean(values)` for each element.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::get_perturbation;
/// let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let pert = get_perturbation(&vals);
/// assert!((pert[0] - (-2.0)).abs() < 1e-10);
/// assert!((pert[2] - 0.0).abs() < 1e-10);
/// ```
pub fn get_perturbation(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|&v| v - mean).collect()
}

// ============================================================================
// Moist-air properties and latent heats
// ============================================================================

/// Gas constant for moist air (J kg^-1 K^-1).
///
/// Accounts for the water-vapor content via:
/// `R_moist = Rd * (1 + w/epsilon) / (1 + w)`
/// where `epsilon = Mw/Md = 0.622` and `Rd = 287.058 J kg^-1 K^-1`.
///
/// # Arguments
///
/// * `w_kgkg` - Mixing ratio in **kg/kg** (not g/kg).
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::moist_air_gas_constant;
/// // Dry air (w = 0) should give Rd = 287.058
/// let r = moist_air_gas_constant(0.0);
/// assert!((r - 287.058).abs() < 1e-6);
/// // With moisture, R increases
/// assert!(moist_air_gas_constant(0.01) > 287.058);
/// ```
#[inline]
pub fn moist_air_gas_constant(w_kgkg: f64) -> f64 {
    const RD: f64 = 287.058;
    const EPSILON: f64 = 0.622;
    RD * (1.0 + w_kgkg / EPSILON) / (1.0 + w_kgkg)
}

/// Specific heat at constant pressure for moist air (J kg^-1 K^-1).
///
/// `Cp_moist = Cp_d * (1 + (Cp_v/Cp_d)*w) / (1 + w)`
/// where `Cp_d = 1005.7 J kg^-1 K^-1` and `Cp_v = 1875.0 J kg^-1 K^-1`.
///
/// # Arguments
///
/// * `w_kgkg` - Mixing ratio in **kg/kg**.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::moist_air_specific_heat_pressure;
/// // Dry air should give Cp_d = 1005.7
/// let cp = moist_air_specific_heat_pressure(0.0);
/// assert!((cp - 1005.7).abs() < 1e-6);
/// // Moist air has higher specific heat
/// assert!(moist_air_specific_heat_pressure(0.02) > 1005.7);
/// ```
#[inline]
pub fn moist_air_specific_heat_pressure(w_kgkg: f64) -> f64 {
    const CP_D: f64 = 1005.7;
    const CP_V: f64 = 1875.0;
    CP_D * (1.0 + (CP_V / CP_D) * w_kgkg) / (1.0 + w_kgkg)
}

/// Poisson exponent (kappa) for moist air (dimensionless).
///
/// `kappa = R_moist / Cp_moist`
///
/// For dry air this equals `Rd/Cp_d ~ 0.2857`. Moisture shifts both R and Cp,
/// producing a slightly different exponent.
///
/// # Arguments
///
/// * `w_kgkg` - Mixing ratio in **kg/kg**.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::moist_air_poisson_exponent;
/// // Dry air: kappa ~ 0.2854
/// let kappa = moist_air_poisson_exponent(0.0);
/// assert!((kappa - 287.058 / 1005.7).abs() < 1e-4);
/// ```
#[inline]
pub fn moist_air_poisson_exponent(w_kgkg: f64) -> f64 {
    moist_air_gas_constant(w_kgkg) / moist_air_specific_heat_pressure(w_kgkg)
}

/// Temperature-dependent latent heat of vaporization (J/kg).
///
/// Uses the linear approximation from Bolton (1980):
/// `Lv = 2.501e6 - 2370.0 * t_c`
///
/// # Arguments
///
/// * `t_c` - Temperature in Celsius.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::water_latent_heat_vaporization;
/// // At 0 C: Lv ~ 2.501e6 J/kg
/// let lv = water_latent_heat_vaporization(0.0);
/// assert!((lv - 2.501e6).abs() < 1.0);
/// // At 20 C: Lv decreases
/// assert!(water_latent_heat_vaporization(20.0) < lv);
/// ```
#[inline]
pub fn water_latent_heat_vaporization(t_c: f64) -> f64 {
    2.501e6 - 2370.0 * t_c
}

/// Temperature-dependent latent heat of melting (J/kg).
///
/// `Lf = 3.34e5 + 2106.0 * t_c`
///
/// At 0 C this gives the standard value of 3.34e5 J/kg. The temperature
/// dependence is weak but included for completeness (MetPy compatibility).
///
/// # Arguments
///
/// * `t_c` - Temperature in Celsius.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::water_latent_heat_melting;
/// let lf = water_latent_heat_melting(0.0);
/// assert!((lf - 3.34e5).abs() < 1.0);
/// ```
#[inline]
pub fn water_latent_heat_melting(t_c: f64) -> f64 {
    3.34e5 + 2106.0 * t_c
}

/// Temperature-dependent latent heat of sublimation (J/kg).
///
/// Sum of vaporization and melting latent heats:
/// `Ls = Lv(t) + Lf(t)`
///
/// # Arguments
///
/// * `t_c` - Temperature in Celsius.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::water_latent_heat_sublimation;
/// let ls = water_latent_heat_sublimation(0.0);
/// let expected = 2.501e6 + 3.34e5;
/// assert!((ls - expected).abs() < 1.0);
/// ```
#[inline]
pub fn water_latent_heat_sublimation(t_c: f64) -> f64 {
    water_latent_heat_vaporization(t_c) + water_latent_heat_melting(t_c)
}

/// Relative humidity (%) from dry-bulb, wet-bulb, and pressure using the
/// psychrometric equation.
///
/// Uses a ventilated (Assmann-type) psychrometer constant `A = 0.000799 C^-1`:
/// ```text
/// e  = es(Tw) - A * P * (T - Tw)
/// RH = 100 * e / es(T)
/// ```
/// where `es()` is the saturation vapor pressure (Bolton formula via
/// [`saturation_vapor_pressure`]).
///
/// The result is clamped to `[0, 100]`.
///
/// # Arguments
///
/// * `t_c`  - Dry-bulb temperature (Celsius).
/// * `tw_c` - Wet-bulb temperature (Celsius).
/// * `p_hpa` - Station pressure (hPa).
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::relative_humidity_wet_psychrometric;
/// // When Tw == T, RH should be 100%
/// let rh = relative_humidity_wet_psychrometric(20.0, 20.0, 1013.25);
/// assert!((rh - 100.0).abs() < 0.1);
/// // Dry bulb well above wet bulb => low RH
/// let rh2 = relative_humidity_wet_psychrometric(30.0, 18.0, 1013.25);
/// assert!(rh2 > 0.0 && rh2 < 60.0);
/// ```
pub fn relative_humidity_wet_psychrometric(t_c: f64, tw_c: f64, p_hpa: f64) -> f64 {
    const A: f64 = 0.000799; // ventilated psychrometer constant (C^-1)
    let es_tw = saturation_vapor_pressure(tw_c);
    let es_t = saturation_vapor_pressure(t_c);
    if es_t <= 0.0 {
        return 0.0;
    }
    let e = es_tw - A * p_hpa * (t_c - tw_c);
    let rh = 100.0 * e / es_t;
    rh.clamp(0.0, 100.0)
}

/// Trapezoidal weighted average of values over a coordinate.
///
/// Computes the integral-mean of `values` weighted by the coordinate spacing in
/// `weights` using the trapezoidal rule:
///
/// ```text
/// avg = sum_{i=0}^{n-2} [ 0.5*(v[i]+v[i+1]) * (w[i+1]-w[i]) ] / (w[n-1] - w[0])
/// ```
///
/// This is useful for pressure-weighted layer averages and similar integrals
/// over non-uniform grids.
///
/// # Arguments
///
/// * `values`  - The quantity to average.
/// * `weights` - The coordinate (e.g., pressure, height) at each point. Must be
///   monotonic (increasing or decreasing).
///
/// # Returns
///
/// The trapezoidal weighted average. Returns `0.0` if fewer than 2 points or
/// the weight span is zero.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::weighted_continuous_average;
/// // Constant value => average equals that value
/// let avg = weighted_continuous_average(&[5.0, 5.0, 5.0], &[0.0, 1.0, 2.0]);
/// assert!((avg - 5.0).abs() < 1e-10);
/// ```
pub fn weighted_continuous_average(values: &[f64], weights: &[f64]) -> f64 {
    let n = values.len().min(weights.len());
    if n < 2 {
        return 0.0;
    }
    let span = weights[n - 1] - weights[0];
    if span.abs() < 1e-30 {
        return 0.0;
    }
    let mut integral = 0.0;
    for i in 0..n - 1 {
        let dw = weights[i + 1] - weights[i];
        let v_avg = 0.5 * (values[i] + values[i + 1]);
        integral += v_avg * dw;
    }
    integral / span
}

// ============================================================================
// New implementations -- humidity / thickness functions
// ============================================================================

/// Specific humidity from mixing ratio.
///
/// `q = w / (1 + w)` where `w` is the mixing ratio in kg/kg.
///
/// Both input and output are in kg/kg (dimensionless mass ratios).
///
/// # Arguments
///
/// * `mixing_ratio` - Mixing ratio in kg/kg.
///
/// # Returns
///
/// Specific humidity in kg/kg.
///
/// # References
///
/// [Salby1996] pg. 118.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::specific_humidity_from_mixing_ratio;
/// let q = specific_humidity_from_mixing_ratio(0.012);
/// assert!((q - 0.011857707510).abs() < 1e-8);
/// ```
#[inline]
pub fn specific_humidity_from_mixing_ratio(mixing_ratio: f64) -> f64 {
    mixing_ratio / (1.0 + mixing_ratio)
}

/// Hypsometric thickness (meters) from pressure, temperature, and relative humidity profiles.
///
/// Computes the thickness of a layer using the hypsometric equation with virtual
/// temperature adjustment derived from relative humidity:
///
/// ```text
/// dz = -(Rd/g) * integral(Tv * d(ln p))
/// ```
///
/// Virtual temperature is computed from temperature and mixing ratio, where
/// mixing ratio is derived from relative humidity.
///
/// # Arguments
///
/// * `pressure` - Pressure profile (hPa, surface first, decreasing)
/// * `temperature` - Temperature profile (Celsius)
/// * `relative_humidity` - Relative humidity profile (percent, 0-100)
///
/// # Returns
///
/// Layer thickness in meters.
///
/// # Examples
///
/// ```
/// use metrust::calc::thermo::thickness_hydrostatic_from_relative_humidity;
/// let p  = vec![1000.0, 900.0, 800.0, 700.0, 600.0, 500.0];
/// let t  = vec![  25.0,  18.0,  10.0,   2.0,  -8.0, -18.0];
/// let rh = vec![  80.0,  70.0,  60.0,  50.0,  40.0,  30.0];
/// let dz = thickness_hydrostatic_from_relative_humidity(&p, &t, &rh);
/// assert!(dz > 5500.0 && dz < 5700.0);
/// ```
pub fn thickness_hydrostatic_from_relative_humidity(
    pressure: &[f64],
    temperature: &[f64],
    relative_humidity: &[f64],
) -> f64 {
    let n = pressure.len().min(temperature.len()).min(relative_humidity.len());
    if n < 2 {
        return 0.0;
    }

    // Compute virtual temperature at each level
    let tv: Vec<f64> = (0..n)
        .map(|i| {
            // mixing ratio in g/kg from RH
            let w_gkg = mixing_ratio_from_relative_humidity(
                pressure[i],
                temperature[i],
                relative_humidity[i],
            );
            let w = w_gkg / 1000.0; // kg/kg
            let t_k = temperature[i] + ZEROCNK;
            // Tv = T * (1 + w/epsilon) / (1 + w)
            t_k * (1.0 + w / EPS) / (1.0 + w)
        })
        .collect();

    // Trapezoidal integration: dz = -(Rd/g) * integral(Tv d(ln p))
    let mut integral = 0.0;
    for i in 0..n - 1 {
        let dlnp = (pressure[i + 1] * 100.0).ln() - (pressure[i] * 100.0).ln();
        let tv_avg = 0.5 * (tv[i] + tv[i + 1]);
        integral += tv_avg * dlnp;
    }
    -(RD / G) * integral
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Tolerance helpers --
    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ====================================================================
    // Re-export / wrapper sanity tests
    // ====================================================================

    #[test]
    fn test_potential_temperature() {
        // At 1000 hPa, theta = T(K).
        let theta = potential_temperature(1000.0, 20.0);
        assert!(approx(theta, 293.15, 0.1));
    }

    #[test]
    fn test_equivalent_potential_temperature() {
        let theta_e = equivalent_potential_temperature(1000.0, 20.0, 15.0);
        // Should be well above dry theta of ~293 K.
        assert!(theta_e > 310.0);
    }

    #[test]
    fn test_saturation_vapor_pressure() {
        let es = saturation_vapor_pressure(20.0);
        // Ambaum (2020) / MetPy: 23.347 hPa at 20 C
        assert!(approx(es, 23.347, 0.5));
    }

    #[test]
    fn test_mixing_ratio_wrapper() {
        let w = mixing_ratio(1013.25, 20.0);
        assert!(w > 10.0 && w < 20.0);
    }

    #[test]
    fn test_dewpoint_from_relative_humidity_wrapper() {
        // At RH=100%, dewpoint equals temperature.
        let td = dewpoint_from_relative_humidity(20.0, 100.0);
        assert!(approx(td, 20.0, 0.5));
    }

    #[test]
    fn test_relative_humidity_from_dewpoint_wrapper() {
        // When Td=T, RH should be 100%.
        let rh = relative_humidity_from_dewpoint(20.0, 20.0);
        assert!(approx(rh, 100.0, 0.5));
    }

    #[test]
    fn test_vapor_pressure_wrapper() {
        let e = vapor_pressure(20.0);
        assert!(approx(e, 23.37, 0.5));
    }

    #[test]
    fn test_virtual_temperature_wrapper() {
        // With moisture, Tv > T.
        let tv = virtual_temperature(20.0, 1000.0, 15.0);
        assert!(tv > 20.0);
    }

    #[test]
    fn test_lcl_wrapper() {
        let (p_lcl, t_lcl) = lcl(1000.0, 25.0, 15.0);
        assert!(p_lcl < 1000.0);
        assert!(p_lcl > 700.0);
        assert!(t_lcl < 25.0);
    }

    #[test]
    fn test_saturation_mixing_ratio_reexport() {
        let ws = saturation_mixing_ratio(1000.0, 20.0);
        assert!(ws > 10.0 && ws < 20.0);
    }

    #[test]
    fn test_density_reexport() {
        // Dry air at 1013.25 hPa, 15 C => ~1.225 kg/m^3.
        let rho = density(1013.25, 15.0, 0.0);
        assert!(approx(rho, 1.225, 0.02));
    }

    // ====================================================================
    // Showalter Index
    // ====================================================================

    #[test]
    fn test_showalter_index_basic() {
        let p = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
        let t = vec![30.0, 25.0, 20.0, 10.0, -10.0, -35.0];
        let td = vec![20.0, 15.0, 14.0, 2.0, -20.0, -45.0];

        let si = showalter_index(&p, &t, &td);
        assert!(si.is_finite());
    }

    #[test]
    fn test_showalter_index_stable_sounding() {
        // Very stable: warm 500 hPa environment.
        let p = vec![1000.0, 850.0, 700.0, 500.0, 300.0];
        let t = vec![15.0, 5.0, -2.0, 0.0, -30.0];
        let td = vec![10.0, 0.0, -10.0, -15.0, -45.0];

        let si = showalter_index(&p, &t, &td);
        // Stable sounding should have positive Showalter Index.
        assert!(si > 0.0, "Expected positive SI for stable sounding, got {}", si);
    }

    #[test]
    fn test_showalter_index_unstable_sounding() {
        // Very unstable: warm moist 850 hPa, cold 500 hPa.
        let p = vec![1000.0, 850.0, 700.0, 500.0, 300.0];
        let t = vec![35.0, 25.0, 10.0, -20.0, -45.0];
        let td = vec![25.0, 22.0, 2.0, -30.0, -55.0];

        let si = showalter_index(&p, &t, &td);
        // Unstable sounding should have negative Showalter Index.
        assert!(si < 0.0, "Expected negative SI for unstable sounding, got {}", si);
    }

    // ====================================================================
    // K-Index
    // ====================================================================

    #[test]
    fn test_k_index_formula() {
        // KI = (T850-T500) + Td850 - (T700-Td700)
        // KI = (20-(-10)) + 14 - (10-2) = 30 + 14 - 8 = 36
        let ki = k_index(20.0, 14.0, 10.0, 2.0, -10.0);
        assert!(approx(ki, 36.0, 1e-10));
    }

    #[test]
    fn test_k_index_low_moisture() {
        // Dry atmosphere should give a low K-Index.
        let ki = k_index(10.0, -5.0, 5.0, -15.0, -5.0);
        // (10-(-5)) + (-5) - (5-(-15)) = 15 - 5 - 20 = -10
        assert!(approx(ki, -10.0, 1e-10));
    }

    // ====================================================================
    // Vertical Totals / Cross Totals / Total Totals
    // ====================================================================

    #[test]
    fn test_vertical_totals() {
        assert!(approx(vertical_totals(20.0, -10.0), 30.0, 1e-10));
    }

    #[test]
    fn test_cross_totals() {
        assert!(approx(cross_totals(14.0, -10.0), 24.0, 1e-10));
    }

    #[test]
    fn test_total_totals() {
        // TT = VT + CT = (T850-T500) + (Td850-T500)
        // = (20-(-10)) + (14-(-10)) = 30 + 24 = 54
        let tt = total_totals(20.0, 14.0, -10.0);
        assert!(approx(tt, 54.0, 1e-10));
    }

    #[test]
    fn test_total_totals_decomposition() {
        let t850 = 18.0;
        let td850 = 12.0;
        let t500 = -8.0;
        let tt = total_totals(t850, td850, t500);
        let expected = vertical_totals(t850, t500) + cross_totals(td850, t500);
        assert!(approx(tt, expected, 1e-10));
    }

    // ====================================================================
    // SWEAT Index
    // ====================================================================

    #[test]
    fn test_sweat_index_basic() {
        let sweat = sweat_index(20.0, 14.0, -10.0, 200.0, 250.0, 25.0, 40.0);
        assert!(sweat > 0.0);

        // Term breakdown:
        // TT = 54
        // moisture = 12 * 14 = 168
        // tt_term = 20 * (54 - 49) = 100
        // wind850 = 2 * 25 = 50
        // wind500 = 40
        // shear: dd850=200 in [130,250], dd500=250 in [210,310],
        //   dd500-dd850=50>0, ff850=25>=15, ff500=40>=15 => active
        //   125*(sin(50_deg) + 0.2)
        let angle_rad = 50.0_f64.to_radians();
        let expected_shear = 125.0 * (angle_rad.sin() + 0.2);
        let expected = 168.0 + 100.0 + 50.0 + 40.0 + expected_shear;
        assert!(approx(sweat, expected, 0.01));
    }

    #[test]
    fn test_sweat_index_no_shear_term() {
        // dd850 outside [130,250] => shear term zeroed.
        let sweat = sweat_index(20.0, 14.0, -10.0, 100.0, 250.0, 25.0, 40.0);
        // TT=54, moisture=168, tt_term=100, wind850=50, wind500=40, shear=0
        let expected = 168.0 + 100.0 + 50.0 + 40.0;
        assert!(approx(sweat, expected, 0.01));
    }

    #[test]
    fn test_sweat_index_tt_below_49() {
        // Total totals below 49 => tt_term is zero.
        // TT = (10-(-5)) + (0-(-5)) = 15 + 5 = 20 < 49
        let sweat = sweat_index(10.0, 0.0, -5.0, 180.0, 240.0, 10.0, 10.0);
        // moisture = 12*0 = 0 (td850 max(0,0)=0)
        // tt_term = 0 (TT=20 < 49)
        // wind850 = 2*10 = 20
        // wind500 = 10
        // shear: ff850=10<15 => 0
        let expected = 0.0 + 0.0 + 20.0 + 10.0;
        assert!(approx(sweat, expected, 0.01));
    }

    #[test]
    fn test_sweat_index_negative_td_clamped() {
        // Negative td850 is clamped to 0 in the moisture term.
        let sweat = sweat_index(10.0, -5.0, -8.0, 180.0, 240.0, 20.0, 20.0);
        // moisture = 12 * max(-5, 0) = 0
        assert!(sweat >= 0.0);
    }

    // ====================================================================
    // Downdraft CAPE
    // ====================================================================

    #[test]
    fn test_downdraft_cape_non_negative() {
        let p = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
        let t = vec![30.0, 25.0, 20.0, 5.0, -15.0, -40.0];
        let td = vec![20.0, 12.0, 5.0, -5.0, -25.0, -50.0];

        let dcape = downdraft_cape(&p, &t, &td);
        assert!(dcape >= 0.0);
    }

    #[test]
    fn test_downdraft_cape_short_profile() {
        // Too short to compute.
        let p = vec![1000.0, 925.0];
        let t = vec![30.0, 25.0];
        let td = vec![20.0, 15.0];
        let dcape = downdraft_cape(&p, &t, &td);
        assert!(approx(dcape, 0.0, 1e-10));
    }

    // ====================================================================
    // Brunt-Vaisala Frequency
    // ====================================================================

    #[test]
    fn test_brunt_vaisala_frequency_stable() {
        // Uniform 3 K/km increase in theta => stable.
        let z = vec![0.0, 1000.0, 2000.0, 3000.0, 4000.0];
        let theta = vec![300.0, 303.0, 306.0, 309.0, 312.0];

        let bvf = brunt_vaisala_frequency(&z, &theta);
        assert_eq!(bvf.len(), 5);

        // Interior points: N = sqrt(g/theta * dtheta/dz)
        // dtheta/dz = 3/1000 = 0.003 K/m
        // N = sqrt(9.80665 / 306.0 * 0.003) ~ 0.0098 s^-1
        for i in 1..4 {
            assert!(bvf[i] > 0.009, "bvf[{}] = {} too low", i, bvf[i]);
            assert!(bvf[i] < 0.011, "bvf[{}] = {} too high", i, bvf[i]);
        }
    }

    #[test]
    fn test_brunt_vaisala_frequency_unstable() {
        // Theta decreasing with height => unstable => N = 0.
        let z = vec![0.0, 1000.0, 2000.0];
        let theta = vec![300.0, 298.0, 296.0];

        let bvf = brunt_vaisala_frequency(&z, &theta);
        for &n in &bvf {
            assert!(
                approx(n, 0.0, 1e-10),
                "Expected 0 for unstable layer, got {}",
                n
            );
        }
    }

    #[test]
    fn test_brunt_vaisala_frequency_empty() {
        let bvf = brunt_vaisala_frequency(&[], &[]);
        assert!(bvf.is_empty());
    }

    #[test]
    fn test_brunt_vaisala_frequency_single_point() {
        let bvf = brunt_vaisala_frequency(&[0.0], &[300.0]);
        assert_eq!(bvf.len(), 1);
        assert!(approx(bvf[0], 0.0, 1e-10));
    }

    // ====================================================================
    // Brunt-Vaisala Period
    // ====================================================================

    #[test]
    fn test_brunt_vaisala_period_stable() {
        let z = vec![0.0, 1000.0, 2000.0, 3000.0];
        let theta = vec![300.0, 303.0, 306.0, 309.0];

        let period = brunt_vaisala_period(&z, &theta);
        assert_eq!(period.len(), 4);

        // Period should be roughly 2*pi / 0.01 ~ 628 seconds (~10 min).
        for i in 1..3 {
            assert!(
                period[i] > 500.0 && period[i] < 800.0,
                "period[{}] = {} out of expected range",
                i,
                period[i]
            );
        }
    }

    #[test]
    fn test_brunt_vaisala_period_unstable() {
        let z = vec![0.0, 1000.0, 2000.0];
        let theta = vec![300.0, 298.0, 296.0];

        let period = brunt_vaisala_period(&z, &theta);
        for &p in &period {
            assert!(
                p == f64::INFINITY,
                "Expected INFINITY for unstable period, got {}",
                p
            );
        }
    }

    #[test]
    fn test_brunt_vaisala_period_matches_frequency() {
        let z = vec![0.0, 1000.0, 2000.0, 3000.0];
        let theta = vec![300.0, 304.0, 308.0, 312.0];

        let freq = brunt_vaisala_frequency(&z, &theta);
        let period = brunt_vaisala_period(&z, &theta);

        for i in 0..freq.len() {
            if freq[i] > 0.0 {
                let expected_period = 2.0 * std::f64::consts::PI / freq[i];
                assert!(
                    approx(period[i], expected_period, 1e-10),
                    "period[{}]: {} != 2pi/freq = {}",
                    i,
                    period[i],
                    expected_period
                );
            }
        }
    }

    // ====================================================================
    // Integration tests: stability indices on realistic soundings
    // ====================================================================

    #[test]
    fn test_showalter_vs_lifted_index_relationship() {
        // Both measure parcel-environment difference at 500 hPa, but from different
        // starting levels. On a well-mixed sounding the lifted index (surface-based)
        // should generally show more instability than the Showalter (850 hPa).
        let p = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
        let t = vec![35.0, 28.0, 22.0, 8.0, -18.0, -42.0];
        let td = vec![22.0, 18.0, 16.0, 0.0, -28.0, -52.0];

        let si = showalter_index(&p, &t, &td);
        let li = lifted_index(&p, &t, &td);

        // Both should be finite.
        assert!(si.is_finite());
        assert!(li.is_finite());
    }

    #[test]
    fn test_k_index_high_moisture_environment() {
        // Tropical-like sounding: high moisture everywhere.
        // T850=24, Td850=22, T700=12, Td700=10, T500=-5
        // KI = (24-(-5)) + 22 - (12-10) = 29 + 22 - 2 = 49
        let ki = k_index(24.0, 22.0, 12.0, 10.0, -5.0);
        assert!(approx(ki, 49.0, 1e-10));
        // 49 indicates very high thunderstorm probability.
    }

    #[test]
    fn test_total_totals_severe_threshold() {
        // TT >= 55 indicates severe thunderstorm potential.
        // T850=22, Td850=18, T500=-12
        // TT = (22-(-12)) + (18-(-12)) = 34 + 30 = 64
        let tt = total_totals(22.0, 18.0, -12.0);
        assert!(tt >= 55.0, "Expected TT >= 55 for severe case, got {}", tt);
    }

    #[test]
    fn test_cape_cin_wrapper() {
        let p = vec![925.0, 850.0, 700.0, 500.0, 300.0, 200.0];
        let t = vec![28.0, 22.0, 8.0, -15.0, -40.0, -55.0];
        let td = vec![22.0, 16.0, 0.0, -25.0, -50.0, -65.0];
        let z = vec![750.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0];

        let (cape, cin, h_lcl, _h_lfc) = cape_cin(
            &p, &t, &td, &z, 1000.0, 32.0, 22.0, "sb", 100.0, 300.0, None,
        );
        assert!(cape >= 0.0);
        assert!(cin <= 0.0);
        assert!(h_lcl >= 0.0);
    }

    // ====================================================================
    // Precipitable Water
    // ====================================================================

    #[test]
    fn test_precipitable_water_positive() {
        let p  = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
        let td = vec![20.0, 15.0, 10.0, 0.0, -20.0, -45.0];
        let pw = precipitable_water(&p, &td);
        assert!(pw > 0.0, "PW should be positive, got {}", pw);
        // Typical mid-latitude warm-season PW: 20-50 mm
        assert!(pw > 10.0 && pw < 80.0, "PW={} out of realistic range", pw);
    }

    #[test]
    fn test_precipitable_water_dry_sounding() {
        let p  = vec![1000.0, 850.0, 700.0, 500.0];
        let td = vec![-30.0, -35.0, -40.0, -50.0];
        let pw = precipitable_water(&p, &td);
        assert!(pw > 0.0);
        assert!(pw < 5.0, "Dry sounding PW should be small, got {}", pw);
    }

    #[test]
    fn test_precipitable_water_short_profile() {
        assert!(approx(precipitable_water(&[1000.0], &[20.0]), 0.0, 1e-10));
        assert!(approx(precipitable_water(&[], &[]), 0.0, 1e-10));
    }

    // ====================================================================
    // Brunt-Vaisala Frequency Squared
    // ====================================================================

    #[test]
    fn test_bvf_squared_stable() {
        let z     = vec![0.0, 1000.0, 2000.0, 3000.0];
        let theta = vec![300.0, 303.0, 306.0, 309.0];
        let n2 = brunt_vaisala_frequency_squared(&z, &theta);
        assert_eq!(n2.len(), 4);
        for i in 1..3 {
            assert!(n2[i] > 0.0, "N^2 should be positive for stable layer, got {}", n2[i]);
        }
    }

    #[test]
    fn test_bvf_squared_unstable_is_negative() {
        let z     = vec![0.0, 1000.0, 2000.0];
        let theta = vec![300.0, 298.0, 296.0];
        let n2 = brunt_vaisala_frequency_squared(&z, &theta);
        // Unlike brunt_vaisala_frequency which clamps to 0, N^2 should be negative.
        for &v in &n2 {
            assert!(v < 0.0, "N^2 should be negative for unstable layer, got {}", v);
        }
    }

    #[test]
    fn test_bvf_squared_matches_frequency_squared() {
        let z     = vec![0.0, 1000.0, 2000.0, 3000.0];
        let theta = vec![300.0, 304.0, 308.0, 312.0];
        let n2 = brunt_vaisala_frequency_squared(&z, &theta);
        let bvf = brunt_vaisala_frequency(&z, &theta);
        for i in 0..n2.len() {
            if n2[i] > 0.0 {
                assert!(approx(n2[i], bvf[i] * bvf[i], 1e-12),
                    "N^2[{}]={} != BVF^2={}", i, n2[i], bvf[i] * bvf[i]);
            }
        }
    }

    // ====================================================================
    // Parcel Profile With LCL
    // ====================================================================

    #[test]
    fn test_parcel_profile_with_lcl_inserts_level() {
        let p = vec![1000.0, 900.0, 800.0, 700.0, 500.0];
        let (p_out, t_out) = parcel_profile_with_lcl(&p, 25.0, 15.0);
        // LCL should be between 1000 and 500, so output is longer
        assert!(p_out.len() >= p.len());
        assert_eq!(p_out.len(), t_out.len());
        // Output pressures should be monotonically decreasing
        for i in 1..p_out.len() {
            assert!(p_out[i] < p_out[i - 1] + 0.01,
                "Pressures not decreasing at index {}: {} >= {}", i, p_out[i], p_out[i - 1]);
        }
    }

    #[test]
    fn test_parcel_profile_with_lcl_empty() {
        let (p_out, t_out) = parcel_profile_with_lcl(&[], 25.0, 15.0);
        assert!(p_out.is_empty());
        assert!(t_out.is_empty());
    }

    #[test]
    fn test_parcel_profile_with_lcl_temperatures_decrease() {
        let p = vec![1000.0, 900.0, 800.0, 700.0, 500.0, 300.0];
        let (_, t_out) = parcel_profile_with_lcl(&p, 30.0, 20.0);
        // Parcel temperatures should generally decrease with height
        for i in 1..t_out.len() {
            assert!(t_out[i] < t_out[0],
                "Parcel temp at index {} ({}) should be less than surface ({})",
                i, t_out[i], t_out[0]);
        }
    }

    // ====================================================================
    // Thickness Hydrostatic
    // ====================================================================

    #[test]
    fn test_thickness_hydrostatic_reasonable() {
        // 1000-500 hPa layer with T_mean ~ 260 K => ~5400 m
        let dz = thickness_hydrostatic(1000.0, 500.0, 260.0);
        assert!(dz > 5000.0 && dz < 6000.0,
            "Expected ~5400 m, got {}", dz);
    }

    #[test]
    fn test_thickness_hydrostatic_matches_hypsometric() {
        let p_bot = 850.0;
        let p_top = 700.0;
        let t_mean = 270.0;
        let dz = thickness_hydrostatic(p_bot, p_top, t_mean);
        let dz_ref = thickness_hypsometric(p_bot, p_top, t_mean);
        assert!(approx(dz, dz_ref, 1e-10));
    }

    // ====================================================================
    // Add Height to Pressure
    // ====================================================================

    #[test]
    fn test_add_height_to_pressure_up() {
        // Going up 1000m from sea level => pressure decreases
        let p_new = add_height_to_pressure(1013.25, 1000.0);
        assert!(p_new < 1013.25, "Ascending should lower pressure");
        assert!(p_new > 800.0, "1000m ascent shouldn't drop below 800 hPa");
    }

    #[test]
    fn test_add_height_to_pressure_down() {
        // Going down 500m from 900 hPa => pressure increases
        let p_new = add_height_to_pressure(900.0, -500.0);
        assert!(p_new > 900.0, "Descending should raise pressure");
    }

    #[test]
    fn test_add_height_to_pressure_zero() {
        // Zero change should give same pressure
        let p_new = add_height_to_pressure(850.0, 0.0);
        assert!(approx(p_new, 850.0, 0.5));
    }

    // ====================================================================
    // Add Pressure to Height
    // ====================================================================

    #[test]
    fn test_add_pressure_to_height_decrease() {
        // Decreasing pressure by 100 hPa from sea level => height increases
        let h_new = add_pressure_to_height(0.0, -100.0);
        assert!(h_new > 0.0, "Decreasing pressure should raise height");
    }

    #[test]
    fn test_add_pressure_to_height_increase() {
        // Increasing pressure from 2000m => height decreases
        let h_new = add_pressure_to_height(2000.0, 50.0);
        assert!(h_new < 2000.0, "Increasing pressure should lower height");
    }

    #[test]
    fn test_add_pressure_to_height_zero() {
        let h_new = add_pressure_to_height(1500.0, 0.0);
        assert!(approx(h_new, 1500.0, 0.5));
    }

    // ====================================================================
    // Get Perturbation
    // ====================================================================

    #[test]
    fn test_get_perturbation_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pert = get_perturbation(&vals);
        assert_eq!(pert.len(), 5);
        // Mean = 3.0
        assert!(approx(pert[0], -2.0, 1e-10));
        assert!(approx(pert[1], -1.0, 1e-10));
        assert!(approx(pert[2],  0.0, 1e-10));
        assert!(approx(pert[3],  1.0, 1e-10));
        assert!(approx(pert[4],  2.0, 1e-10));
    }

    #[test]
    fn test_get_perturbation_sums_to_zero() {
        let vals = vec![10.0, 20.0, 30.0, 15.0, 25.0];
        let pert = get_perturbation(&vals);
        let sum: f64 = pert.iter().sum();
        assert!(approx(sum, 0.0, 1e-10), "Perturbations should sum to zero, got {}", sum);
    }

    #[test]
    fn test_get_perturbation_empty() {
        let pert = get_perturbation(&[]);
        assert!(pert.is_empty());
    }

    #[test]
    fn test_get_perturbation_single_value() {
        let pert = get_perturbation(&[42.0]);
        assert_eq!(pert.len(), 1);
        assert!(approx(pert[0], 0.0, 1e-10));
    }

    #[test]
    fn test_get_perturbation_uniform() {
        let vals = vec![5.0, 5.0, 5.0, 5.0];
        let pert = get_perturbation(&vals);
        for &p in &pert {
            assert!(approx(p, 0.0, 1e-10));
        }
    }

    // ====================================================================
    // Moist Air Gas Constant
    // ====================================================================

    #[test]
    fn test_moist_air_gas_constant_dry() {
        // w = 0 => R = Rd = 287.058
        let r = moist_air_gas_constant(0.0);
        assert!(approx(r, 287.058, 1e-6));
    }

    #[test]
    fn test_moist_air_gas_constant_increases_with_moisture() {
        // Moist air has a higher gas constant than dry air.
        let r_dry = moist_air_gas_constant(0.0);
        let r_moist = moist_air_gas_constant(0.015); // 15 g/kg
        assert!(r_moist > r_dry,
            "R_moist ({}) should be > R_dry ({})", r_moist, r_dry);
    }

    #[test]
    fn test_moist_air_gas_constant_known_value() {
        // w = 0.01 kg/kg (10 g/kg)
        // R = 287.058 * (1 + 0.01/0.622) / (1 + 0.01)
        //   = 287.058 * 1.016077 / 1.01
        //   = 287.058 * 1.005026 ~ 288.50
        let r = moist_air_gas_constant(0.01);
        let expected = 287.058 * (1.0 + 0.01 / 0.622) / (1.0 + 0.01);
        assert!(approx(r, expected, 1e-6));
    }

    // ====================================================================
    // Moist Air Specific Heat at Constant Pressure
    // ====================================================================

    #[test]
    fn test_moist_air_specific_heat_pressure_dry() {
        let cp = moist_air_specific_heat_pressure(0.0);
        assert!(approx(cp, 1005.7, 1e-6));
    }

    #[test]
    fn test_moist_air_specific_heat_pressure_increases_with_moisture() {
        let cp_dry = moist_air_specific_heat_pressure(0.0);
        let cp_moist = moist_air_specific_heat_pressure(0.02);
        assert!(cp_moist > cp_dry,
            "Cp_moist ({}) should be > Cp_dry ({})", cp_moist, cp_dry);
    }

    #[test]
    fn test_moist_air_specific_heat_pressure_known_value() {
        // w = 0.02 kg/kg
        // Cp = 1005.7 * (1 + (1875/1005.7)*0.02) / (1 + 0.02)
        let w = 0.02;
        let expected = 1005.7 * (1.0 + (1875.0 / 1005.7) * w) / (1.0 + w);
        let cp = moist_air_specific_heat_pressure(w);
        assert!(approx(cp, expected, 1e-6));
    }

    // ====================================================================
    // Moist Air Poisson Exponent
    // ====================================================================

    #[test]
    fn test_moist_air_poisson_exponent_dry() {
        // kappa_d = Rd / Cp_d = 287.058 / 1005.7 ~ 0.28539
        let kappa = moist_air_poisson_exponent(0.0);
        let expected = 287.058 / 1005.7;
        assert!(approx(kappa, expected, 1e-4));
    }

    #[test]
    fn test_moist_air_poisson_exponent_equals_ratio() {
        let w = 0.012;
        let kappa = moist_air_poisson_exponent(w);
        let r = moist_air_gas_constant(w);
        let cp = moist_air_specific_heat_pressure(w);
        assert!(approx(kappa, r / cp, 1e-10),
            "kappa ({}) should equal R/Cp ({}/{}={})", kappa, r, cp, r / cp);
    }

    #[test]
    fn test_moist_air_poisson_exponent_slightly_differs_from_dry() {
        // Moisture shifts kappa slightly, but it remains in a reasonable range.
        let kappa_dry = moist_air_poisson_exponent(0.0);
        let kappa_moist = moist_air_poisson_exponent(0.02);
        assert!((kappa_dry - kappa_moist).abs() < 0.01,
            "kappa should not change dramatically: dry={}, moist={}", kappa_dry, kappa_moist);
        assert!(kappa_moist > 0.25 && kappa_moist < 0.30);
    }

    // ====================================================================
    // Water Latent Heat of Vaporization
    // ====================================================================

    #[test]
    fn test_latent_heat_vaporization_at_zero() {
        let lv = water_latent_heat_vaporization(0.0);
        assert!(approx(lv, 2.501e6, 1.0));
    }

    #[test]
    fn test_latent_heat_vaporization_decreases_with_temperature() {
        let lv_0 = water_latent_heat_vaporization(0.0);
        let lv_20 = water_latent_heat_vaporization(20.0);
        let lv_40 = water_latent_heat_vaporization(40.0);
        assert!(lv_20 < lv_0);
        assert!(lv_40 < lv_20);
    }

    #[test]
    fn test_latent_heat_vaporization_known_value() {
        // At 20 C: Lv = 2.501e6 - 2370*20 = 2.501e6 - 47400 = 2453600
        let lv = water_latent_heat_vaporization(20.0);
        assert!(approx(lv, 2_453_600.0, 1.0));
    }

    #[test]
    fn test_latent_heat_vaporization_negative_temperature() {
        // At -10 C: Lv = 2.501e6 - 2370*(-10) = 2.501e6 + 23700 = 2524700
        let lv = water_latent_heat_vaporization(-10.0);
        assert!(approx(lv, 2_524_700.0, 1.0));
    }

    // ====================================================================
    // Water Latent Heat of Melting
    // ====================================================================

    #[test]
    fn test_latent_heat_melting_at_zero() {
        let lf = water_latent_heat_melting(0.0);
        assert!(approx(lf, 3.34e5, 1.0));
    }

    #[test]
    fn test_latent_heat_melting_known_value() {
        // At -10 C: Lf = 3.34e5 + 2106*(-10) = 334000 - 21060 = 312940
        let lf = water_latent_heat_melting(-10.0);
        assert!(approx(lf, 312_940.0, 1.0));
    }

    #[test]
    fn test_latent_heat_melting_at_plus_five() {
        // At 5 C: Lf = 334000 + 2106*5 = 334000 + 10530 = 344530
        let lf = water_latent_heat_melting(5.0);
        assert!(approx(lf, 344_530.0, 1.0));
    }

    // ====================================================================
    // Water Latent Heat of Sublimation
    // ====================================================================

    #[test]
    fn test_latent_heat_sublimation_at_zero() {
        let ls = water_latent_heat_sublimation(0.0);
        let expected = 2.501e6 + 3.34e5;
        assert!(approx(ls, expected, 1.0));
    }

    #[test]
    fn test_latent_heat_sublimation_equals_sum() {
        for &t in &[-20.0, -10.0, 0.0, 10.0, 25.0] {
            let ls = water_latent_heat_sublimation(t);
            let lv = water_latent_heat_vaporization(t);
            let lf = water_latent_heat_melting(t);
            assert!(approx(ls, lv + lf, 1e-6),
                "Ls({}) = {} != Lv + Lf = {}", t, ls, lv + lf);
        }
    }

    #[test]
    fn test_latent_heat_sublimation_greater_than_vaporization() {
        // Sublimation always > vaporization (melting contribution is positive near 0 C)
        let ls = water_latent_heat_sublimation(0.0);
        let lv = water_latent_heat_vaporization(0.0);
        assert!(ls > lv);
    }

    // ====================================================================
    // Relative Humidity from Wet-Bulb (Psychrometric)
    // ====================================================================

    #[test]
    fn test_rh_psychrometric_saturated() {
        // When Tw == T, the air is saturated => RH = 100%
        let rh = relative_humidity_wet_psychrometric(20.0, 20.0, 1013.25);
        assert!(approx(rh, 100.0, 0.1),
            "Expected ~100% when Tw=T, got {}", rh);
    }

    #[test]
    fn test_rh_psychrometric_dry_depression() {
        // Large wet-bulb depression => low RH
        let rh = relative_humidity_wet_psychrometric(35.0, 18.0, 1013.25);
        assert!(rh > 0.0 && rh < 40.0,
            "Large depression should give low RH, got {}", rh);
    }

    #[test]
    fn test_rh_psychrometric_moderate() {
        // Moderate depression: T=25, Tw=18
        let rh = relative_humidity_wet_psychrometric(25.0, 18.0, 1013.25);
        assert!(rh > 30.0 && rh < 70.0,
            "Expected moderate RH, got {}", rh);
    }

    #[test]
    fn test_rh_psychrometric_clamped_to_100() {
        // Ensure the result never exceeds 100 even with numerical imprecision
        let rh = relative_humidity_wet_psychrometric(15.0, 15.0, 900.0);
        assert!(rh <= 100.0, "RH should not exceed 100%, got {}", rh);
    }

    #[test]
    fn test_rh_psychrometric_clamped_to_0() {
        // Very extreme depression — should clamp to 0
        let rh = relative_humidity_wet_psychrometric(50.0, -10.0, 1013.25);
        assert!(rh >= 0.0, "RH should not go below 0%, got {}", rh);
    }

    #[test]
    fn test_rh_psychrometric_pressure_sensitivity() {
        // Same temperatures, different pressures: higher pressure => lower RH
        // because the psychrometric correction A*P*(T-Tw) is larger.
        let rh_low_p = relative_humidity_wet_psychrometric(25.0, 20.0, 800.0);
        let rh_high_p = relative_humidity_wet_psychrometric(25.0, 20.0, 1013.25);
        assert!(rh_low_p > rh_high_p,
            "Lower pressure should give higher RH: {} vs {}", rh_low_p, rh_high_p);
    }

    // ====================================================================
    // Weighted Continuous Average (Trapezoidal)
    // ====================================================================

    #[test]
    fn test_weighted_continuous_average_constant() {
        // Constant function => average equals the constant
        let avg = weighted_continuous_average(&[7.0, 7.0, 7.0, 7.0], &[0.0, 1.0, 2.0, 3.0]);
        assert!(approx(avg, 7.0, 1e-10));
    }

    #[test]
    fn test_weighted_continuous_average_linear() {
        // Linear function y = x on [0, 4]: trapezoidal integral = exact
        // Average = integral(x dx, 0..4) / 4 = (16/2) / 4 = 2.0
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let avg = weighted_continuous_average(&values, &weights);
        assert!(approx(avg, 2.0, 1e-10));
    }

    #[test]
    fn test_weighted_continuous_average_nonuniform_grid() {
        // Values [10, 20] on weights [0, 10]: average = 0.5*(10+20)*10 / 10 = 15
        let avg = weighted_continuous_average(&[10.0, 20.0], &[0.0, 10.0]);
        assert!(approx(avg, 15.0, 1e-10));
    }

    #[test]
    fn test_weighted_continuous_average_decreasing_weights() {
        // Decreasing weights (like pressure): values [20, 10] at weights [1000, 500]
        // integral = 0.5*(20+10)*(500-1000) = 0.5*30*(-500) = -7500
        // span = 500 - 1000 = -500
        // avg = -7500 / -500 = 15
        let avg = weighted_continuous_average(&[20.0, 10.0], &[1000.0, 500.0]);
        assert!(approx(avg, 15.0, 1e-10));
    }

    #[test]
    fn test_weighted_continuous_average_single_point() {
        let avg = weighted_continuous_average(&[5.0], &[0.0]);
        assert!(approx(avg, 0.0, 1e-10), "Single point should return 0.0");
    }

    #[test]
    fn test_weighted_continuous_average_empty() {
        let avg = weighted_continuous_average(&[], &[]);
        assert!(approx(avg, 0.0, 1e-10));
    }

    #[test]
    fn test_weighted_continuous_average_three_segments() {
        // y = [2, 4, 6, 8] at x = [0, 1, 3, 6]
        // Segment 0: 0.5*(2+4)*1 = 3
        // Segment 1: 0.5*(4+6)*2 = 10
        // Segment 2: 0.5*(6+8)*3 = 21
        // Total integral = 34, span = 6
        // Average = 34/6 ~ 5.6667
        let avg = weighted_continuous_average(
            &[2.0, 4.0, 6.0, 8.0],
            &[0.0, 1.0, 3.0, 6.0],
        );
        assert!(approx(avg, 34.0 / 6.0, 1e-10));
    }

    // ====================================================================
    // specific_humidity_from_mixing_ratio
    // ====================================================================

    #[test]
    fn test_specific_humidity_from_mixing_ratio_basic() {
        // MetPy reference: specific_humidity_from_mixing_ratio(0.012 kg/kg) = 0.0118577075 kg/kg
        let q = specific_humidity_from_mixing_ratio(0.012);
        assert!(approx(q, 0.0118577075, 1e-8));
    }

    #[test]
    fn test_specific_humidity_from_mixing_ratio_zero() {
        // Dry air: w=0 => q=0
        assert!(approx(specific_humidity_from_mixing_ratio(0.0), 0.0, 1e-15));
    }

    #[test]
    fn test_specific_humidity_from_mixing_ratio_identity() {
        // q < w always, and q = w / (1+w)
        let w = 0.02;
        let q = specific_humidity_from_mixing_ratio(w);
        assert!(q < w);
        assert!(approx(q, w / (1.0 + w), 1e-15));
    }

    // ====================================================================
    // thickness_hydrostatic_from_relative_humidity
    // ====================================================================

    #[test]
    fn test_thickness_hydrostatic_from_rh_basic() {
        // MetPy reference: 5614.4389 m (slight constant differences expected)
        let p  = vec![1000.0, 900.0, 800.0, 700.0, 600.0, 500.0];
        let t  = vec![  25.0,  18.0,  10.0,   2.0,  -8.0, -18.0];
        let rh = vec![  80.0,  70.0,  60.0,  50.0,  40.0,  30.0];
        let dz = thickness_hydrostatic_from_relative_humidity(&p, &t, &rh);
        // Allow ~5m tolerance due to different Rd/epsilon constants
        assert!(approx(dz, 5614.4, 5.0),
            "thickness = {dz}, expected ~5614.4 m");
    }

    #[test]
    fn test_thickness_hydrostatic_from_rh_dry() {
        // With RH=0, virtual temp equals actual temp, result should match
        // simple hypsometric thickness
        let p = vec![1000.0, 500.0];
        let t = vec![15.0, -15.0]; // mean T ~273 K
        let rh = vec![0.0, 0.0];
        let dz = thickness_hydrostatic_from_relative_humidity(&p, &t, &rh);
        // With T_mean ~273.15 K: Rd/g * T_mean * ln(1000/500) = 29.27 * 273.15 * 0.6931 ~ 5536
        assert!(dz > 5400.0 && dz < 5700.0, "thickness = {dz}");
    }

    #[test]
    fn test_thickness_hydrostatic_from_rh_more_moisture_increases_thickness() {
        let p  = vec![1000.0, 500.0];
        let t  = vec![25.0, -10.0];
        let rh_low  = vec![20.0, 20.0];
        let rh_high = vec![90.0, 90.0];
        let dz_low  = thickness_hydrostatic_from_relative_humidity(&p, &t, &rh_low);
        let dz_high = thickness_hydrostatic_from_relative_humidity(&p, &t, &rh_high);
        // More moisture => higher virtual temp => greater thickness
        assert!(dz_high > dz_low,
            "high RH thickness ({dz_high}) should exceed low RH ({dz_low})");
    }

    #[test]
    fn test_thickness_hydrostatic_from_rh_single_level() {
        let dz = thickness_hydrostatic_from_relative_humidity(&[1000.0], &[25.0], &[50.0]);
        assert!(approx(dz, 0.0, 1e-10));
    }
}
