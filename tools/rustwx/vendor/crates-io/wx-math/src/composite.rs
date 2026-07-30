//! Derived meteorological parameters computed from 3D model fields.
//!
//! Generic versions that accept data slices instead of WrfFile references,
//! making them usable with any data source (WRF, HRRR GRIB2, GFS, etc.).
//!
//! Uses the metfuncs module for thermodynamic calculations and rayon
//! for parallel computation across grid points.

use crate::thermo as metfuncs;
use rayon::prelude::*;

/// Physical constants
const RD: f64 = 287.058;
const G: f64 = 9.80665;
const ZEROCNK: f64 = 273.15;
const ROCP: f64 = 0.28571426;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linearly interpolate a value at target_h from height/value profiles.
/// Profiles must be ordered with increasing height.
fn interp_at_height(target_h: f64, heights: &[f64], values: &[f64]) -> f64 {
    if heights.is_empty() {
        return f64::NAN;
    }
    if target_h <= heights[0] {
        return values[0];
    }
    if target_h >= heights[heights.len() - 1] {
        return values[values.len() - 1];
    }
    for k in 0..heights.len() - 1 {
        if heights[k] <= target_h && heights[k + 1] >= target_h {
            let frac = (target_h - heights[k]) / (heights[k + 1] - heights[k]);
            return values[k] + frac * (values[k + 1] - values[k]);
        }
    }
    values[values.len() - 1]
}

/// Extract a column (nz values) from a flattened 3D array `[nz][ny][nx]` at grid point (j, i).
fn extract_column(data: &[f64], nz: usize, ny: usize, nx: usize, j: usize, i: usize) -> Vec<f64> {
    let mut col = Vec::with_capacity(nz);
    for k in 0..nz {
        col.push(data[k * ny * nx + j * nx + i]);
    }
    col
}

/// Compute dewpoint (Celsius) from mixing ratio (kg/kg) and pressure (hPa).
pub fn dewpoint_from_q(q: f64, p_hpa: f64) -> f64 {
    let q = q.max(1.0e-10); // avoid log(0)
    let e = q * p_hpa / (0.622 + q); // vapor pressure in hPa
    let e = e.max(1.0e-10);
    let ln_e = (e / 6.112).ln();
    (243.5 * ln_e) / (17.67 - ln_e)
}

// ---------------------------------------------------------------------------
// CAPE / CIN
// ---------------------------------------------------------------------------

/// Compute CAPE/CIN for every grid point (parallelized with rayon).
///
/// All 3D arrays are flattened `[nz][ny][nx]`. 2D arrays are `[ny][nx]`.
///
/// `parcel_type`: `"sb"` (surface-based), `"ml"` (mixed-layer), `"mu"` (most-unstable).
///
/// Inputs:
/// - `pressure_3d`: Full pressure in Pa, shape `[nz][ny][nx]`
/// - `temperature_c_3d`: Temperature in Celsius, shape `[nz][ny][nx]`
/// - `qvapor_3d`: Water vapor mixing ratio in kg/kg, shape `[nz][ny][nx]`
/// - `height_agl_3d`: Height AGL in meters, shape `[nz][ny][nx]`
/// - `psfc`: Surface pressure in Pa, shape `[ny][nx]`
/// - `t2`: 2-meter temperature in K, shape `[ny][nx]`
/// - `q2`: 2-meter mixing ratio in kg/kg, shape `[ny][nx]`
///
/// Returns `(cape_2d, cin_2d, lcl_2d, lfc_2d)` each `Vec<f64>` of size `ny * nx`.
pub fn compute_cape_cin(
    pressure_3d: &[f64],
    temperature_c_3d: &[f64],
    qvapor_3d: &[f64],
    height_agl_3d: &[f64],
    psfc: &[f64],
    t2: &[f64],
    q2: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    parcel_type: &str,
    top_m: Option<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n2d = ny * nx;
    let parcel_type_owned = parcel_type.to_string();

    // Parallel computation over all grid points
    let results: Vec<(f64, f64, f64, f64)> = (0..n2d)
        .into_par_iter()
        .map(|idx| {
            let j = idx / nx;
            let i = idx % nx;

            // Extract column profiles
            let p_col = extract_column(pressure_3d, nz, ny, nx, j, i);
            let t_col = extract_column(temperature_c_3d, nz, ny, nx, j, i);
            let q_col = extract_column(qvapor_3d, nz, ny, nx, j, i);
            let h_col = extract_column(height_agl_3d, nz, ny, nx, j, i);

            // Convert to hPa for metfuncs (temperature already in Celsius)
            let mut p_hpa: Vec<f64> = p_col.iter().map(|&p| p / 100.0).collect();
            let t_c: Vec<f64> = t_col;
            let mut td_c: Vec<f64> = Vec::with_capacity(nz);
            for k in 0..nz {
                td_c.push(dewpoint_from_q(q_col[k], p_hpa[k]));
            }
            let mut h_agl: Vec<f64> = h_col;

            // Ensure profiles are ordered surface-to-top (decreasing pressure)
            if p_hpa.len() > 1 && p_hpa[0] < p_hpa[p_hpa.len() - 1] {
                p_hpa.reverse();
                let mut t_c = t_c;
                t_c.reverse();
                td_c.reverse();
                h_agl.reverse();
                // Surface values
                let psfc_hpa = psfc[idx] / 100.0;
                let t2m_c = t2[idx] - ZEROCNK;
                let td2m_c = dewpoint_from_q(q2[idx], psfc_hpa);
                return metfuncs::cape_cin_core(
                    &p_hpa, &t_c, &td_c, &h_agl,
                    psfc_hpa, t2m_c, td2m_c,
                    &parcel_type_owned, 100.0, 300.0, top_m,
                );
            }

            // Surface values
            let psfc_hpa = psfc[idx] / 100.0;
            let t2m_c = t2[idx] - ZEROCNK;
            let td2m_c = dewpoint_from_q(q2[idx], psfc_hpa);

            metfuncs::cape_cin_core(
                &p_hpa, &t_c, &td_c, &h_agl,
                psfc_hpa, t2m_c, td2m_c,
                &parcel_type_owned, 100.0, 300.0, top_m,
            )
        })
        .collect();

    let mut cape_2d = Vec::with_capacity(n2d);
    let mut cin_2d = Vec::with_capacity(n2d);
    let mut lcl_2d = Vec::with_capacity(n2d);
    let mut lfc_2d = Vec::with_capacity(n2d);

    for (cape, cin, lcl, lfc) in results {
        cape_2d.push(cape);
        cin_2d.push(cin);
        lcl_2d.push(lcl);
        lfc_2d.push(lfc);
    }

    (cape_2d, cin_2d, lcl_2d, lfc_2d)
}

// ---------------------------------------------------------------------------
// Storm Relative Helicity
// ---------------------------------------------------------------------------

/// Compute 0-X km Storm Relative Helicity using Bunkers storm motion.
///
/// `u_3d`, `v_3d`: Wind components in m/s, shape `[nz][ny][nx]`
/// `height_agl_3d`: Height AGL in meters, shape `[nz][ny][nx]`
/// `top_m`: height AGL in meters (typically 1000.0 or 3000.0)
pub fn compute_srh(
    u_3d: &[f64],
    v_3d: &[f64],
    height_agl_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    top_m: f64,
) -> Vec<f64> {
    let n2d = ny * nx;

    (0..n2d)
        .into_par_iter()
        .map(|idx| {
            let j = idx / nx;
            let i = idx % nx;

            let u_col = extract_column(u_3d, nz, ny, nx, j, i);
            let v_col = extract_column(v_3d, nz, ny, nx, j, i);
            let h_col = extract_column(height_agl_3d, nz, ny, nx, j, i);

            // Ensure ordered from surface upward
            let (h_prof, u_prof, v_prof) = if h_col.len() > 1 && h_col[0] > h_col[h_col.len() - 1] {
                let mut h = h_col;
                let mut u = u_col;
                let mut v = v_col;
                h.reverse();
                u.reverse();
                v.reverse();
                (h, u, v)
            } else {
                (h_col, u_col, v_col)
            };

            compute_srh_column(&h_prof, &u_prof, &v_prof, top_m)
        })
        .collect()
}

/// Compute SRH for a single column using Bunkers storm motion.
fn compute_srh_column(
    heights: &[f64],
    u_prof: &[f64],
    v_prof: &[f64],
    top_m: f64,
) -> f64 {
    let nz = heights.len();
    if nz < 2 {
        return 0.0;
    }

    // 1. Compute mean wind in 0-6 km layer
    let mean_depth = 6000.0;
    let mut sum_u = 0.0;
    let mut sum_v = 0.0;
    let mut sum_dz = 0.0;

    for k in 0..nz - 1 {
        if heights[k] >= mean_depth {
            break;
        }
        let h_bot = heights[k];
        let h_top = heights[k + 1].min(mean_depth);
        let dz = h_top - h_bot;
        if dz <= 0.0 {
            continue;
        }
        let u_mid = 0.5 * (u_prof[k] + u_prof[k + 1]);
        let v_mid = 0.5 * (v_prof[k] + v_prof[k + 1]);
        sum_u += u_mid * dz;
        sum_v += v_mid * dz;
        sum_dz += dz;
    }

    if sum_dz <= 0.0 {
        return 0.0;
    }

    let mean_u = sum_u / sum_dz;
    let mean_v = sum_v / sum_dz;

    // 2. Compute 0-6 km shear vector
    let u_sfc = u_prof[0];
    let v_sfc = v_prof[0];
    let u_6km = interp_at_height(mean_depth, heights, u_prof);
    let v_6km = interp_at_height(mean_depth, heights, v_prof);
    let shear_u = u_6km - u_sfc;
    let shear_v = v_6km - v_sfc;

    // 3. Bunkers deviation: rotate shear 90 degrees clockwise, scale to 7.5 m/s
    let shear_mag = (shear_u * shear_u + shear_v * shear_v).sqrt();
    let (dev_u, dev_v) = if shear_mag > 0.1 {
        let scale = 7.5 / shear_mag;
        // 90-degree clockwise rotation: (u, v) -> (v, -u)
        (shear_v * scale, -shear_u * scale)
    } else {
        (0.0, 0.0)
    };

    // Right-moving storm motion
    let storm_u = mean_u + dev_u;
    let storm_v = mean_v + dev_v;

    // 4. Compute SRH
    let mut srh = 0.0;

    for k in 0..nz - 1 {
        if heights[k] >= top_m {
            break;
        }

        let h_bot = heights[k];
        let h_top = heights[k + 1].min(top_m);

        if h_top <= h_bot {
            continue;
        }

        let u_bot = u_prof[k];
        let v_bot = v_prof[k];

        let (u_top_val, v_top_val) = if h_top < heights[k + 1] {
            let frac = (h_top - heights[k]) / (heights[k + 1] - heights[k]);
            (
                u_prof[k] + frac * (u_prof[k + 1] - u_prof[k]),
                v_prof[k] + frac * (v_prof[k + 1] - v_prof[k]),
            )
        } else {
            (u_prof[k + 1], v_prof[k + 1])
        };

        let sr_u_bot = u_bot - storm_u;
        let sr_v_bot = v_bot - storm_v;
        let sr_u_top = u_top_val - storm_u;
        let sr_v_top = v_top_val - storm_v;

        srh += sr_u_top * sr_v_bot - sr_u_bot * sr_v_top;
    }

    srh
}

// ---------------------------------------------------------------------------
// Bulk Wind Shear
// ---------------------------------------------------------------------------

/// Compute bulk wind shear magnitude (m/s) between `bottom_m` and `top_m` AGL.
///
/// `u_3d`, `v_3d`: Wind components in m/s, shape `[nz][ny][nx]`
/// `height_agl_3d`: Height AGL in meters, shape `[nz][ny][nx]`
pub fn compute_shear(
    u_3d: &[f64],
    v_3d: &[f64],
    height_agl_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    bottom_m: f64,
    top_m: f64,
) -> Vec<f64> {
    let n2d = ny * nx;

    (0..n2d)
        .into_par_iter()
        .map(|idx| {
            let j = idx / nx;
            let i = idx % nx;

            let u_col = extract_column(u_3d, nz, ny, nx, j, i);
            let v_col = extract_column(v_3d, nz, ny, nx, j, i);
            let h_col = extract_column(height_agl_3d, nz, ny, nx, j, i);

            // Ensure ordered from surface upward
            let (h_prof, u_prof, v_prof) = if h_col.len() > 1 && h_col[0] > h_col[h_col.len() - 1]
            {
                let mut h = h_col;
                let mut u = u_col;
                let mut v = v_col;
                h.reverse();
                u.reverse();
                v.reverse();
                (h, u, v)
            } else {
                (h_col, u_col, v_col)
            };

            let u_bot = interp_at_height(bottom_m, &h_prof, &u_prof);
            let v_bot = interp_at_height(bottom_m, &h_prof, &v_prof);
            let u_top = interp_at_height(top_m, &h_prof, &u_prof);
            let v_top = interp_at_height(top_m, &h_prof, &v_prof);

            let du = u_top - u_bot;
            let dv = v_top - v_bot;
            (du * du + dv * dv).sqrt()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Significant Tornado Parameter
// ---------------------------------------------------------------------------

/// Significant Tornado Parameter (STP).
///
/// STP = (CAPE/1500) * ((2000 - LCL)/1000) * (SRH_1km/150) * min(SHEAR_6km/20, 1.5)
///
/// Inputs are pre-computed 2D fields, each of size `n` (ny * nx).
pub fn compute_stp(
    cape: &[f64],
    lcl: &[f64],
    srh_1km: &[f64],
    shear_6km: &[f64],
) -> Vec<f64> {
    let n = cape.len();
    let mut stp = Vec::with_capacity(n);

    for idx in 0..n {
        let cape_term = (cape[idx] / 1500.0).max(0.0);
        let lcl_term = ((2000.0 - lcl[idx]) / 1000.0).clamp(0.0, 2.0);
        let srh_term = (srh_1km[idx] / 150.0).max(0.0);
        let shear_term = (shear_6km[idx] / 20.0).min(1.5).max(0.0);

        stp.push(cape_term * lcl_term * srh_term * shear_term);
    }

    stp
}

// ---------------------------------------------------------------------------
// Energy Helicity Index
// ---------------------------------------------------------------------------

/// Energy Helicity Index: EHI = (CAPE * SRH) / 160000
///
/// Inputs are pre-computed 2D fields.
pub fn compute_ehi(
    cape: &[f64],
    srh: &[f64],
) -> Vec<f64> {
    let n = cape.len();
    let mut ehi = Vec::with_capacity(n);

    for idx in 0..n {
        ehi.push((cape[idx] * srh[idx]) / 160000.0);
    }

    ehi
}

// ---------------------------------------------------------------------------
// Supercell Composite Parameter
// ---------------------------------------------------------------------------

/// Supercell Composite Parameter: SCP = (MUCAPE/1000) * (SRH_3km/50) * (SHEAR_6km/40)
///
/// Inputs are pre-computed 2D fields.
pub fn compute_scp(
    mucape: &[f64],
    srh_3km: &[f64],
    shear_6km: &[f64],
) -> Vec<f64> {
    let n = mucape.len();
    let mut scp = Vec::with_capacity(n);

    for idx in 0..n {
        let cape_term = (mucape[idx] / 1000.0).max(0.0);
        let srh_term = (srh_3km[idx] / 50.0).max(0.0);
        let shear_term = (shear_6km[idx] / 40.0).max(0.0);
        scp.push(cape_term * srh_term * shear_term);
    }

    scp
}

// ---------------------------------------------------------------------------
// Lapse Rate
// ---------------------------------------------------------------------------

/// Lapse rate (C/km) between two heights in km AGL.
///
/// `temperature_c_3d`: Temperature in Celsius, shape `[nz][ny][nx]`
/// `qvapor_3d`: Water vapor mixing ratio in kg/kg, shape `[nz][ny][nx]`
/// `height_agl_3d`: Height AGL in meters, shape `[nz][ny][nx]`
///
/// Uses virtual temperature for accuracy.
/// Positive values indicate temperature decreasing with height (conditionally unstable).
pub fn compute_lapse_rate(
    temperature_c_3d: &[f64],
    qvapor_3d: &[f64],
    height_agl_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    bottom_km: f64,
    top_km: f64,
) -> Vec<f64> {
    let n2d = ny * nx;
    let bottom_m = bottom_km * 1000.0;
    let top_m_val = top_km * 1000.0;

    (0..n2d)
        .into_par_iter()
        .map(|idx| {
            let j = idx / nx;
            let i = idx % nx;

            let t_col = extract_column(temperature_c_3d, nz, ny, nx, j, i);
            let q_col = extract_column(qvapor_3d, nz, ny, nx, j, i);
            let h_col = extract_column(height_agl_3d, nz, ny, nx, j, i);

            // Ensure ordered from surface upward
            let (h_prof, t_prof, q_prof) =
                if h_col.len() > 1 && h_col[0] > h_col[h_col.len() - 1] {
                    let mut h = h_col;
                    let mut t = t_col;
                    let mut q = q_col;
                    h.reverse();
                    t.reverse();
                    q.reverse();
                    (h, t, q)
                } else {
                    (h_col, t_col, q_col)
                };

            // Compute virtual temperature profile (Celsius)
            let tv_prof: Vec<f64> = (0..t_prof.len())
                .map(|k| {
                    let w = q_prof[k].max(0.0); // mixing ratio in kg/kg
                    let t_k = t_prof[k] + ZEROCNK;
                    t_k * (1.0 + 0.61 * w) - ZEROCNK // back to Celsius
                })
                .collect();

            let tv_bot = interp_at_height(bottom_m, &h_prof, &tv_prof);
            let tv_top = interp_at_height(top_m_val, &h_prof, &tv_prof);
            let dz_km = top_km - bottom_km;

            if dz_km.abs() < 0.001 {
                return 0.0;
            }

            (tv_bot - tv_top) / dz_km
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Precipitable Water
// ---------------------------------------------------------------------------

/// Precipitable water (mm).
///
/// PW = (1/g) * integral(QVAPOR * dp) from surface to top of model.
///
/// `qvapor_3d`: Mixing ratio in kg/kg, shape `[nz][ny][nx]`
/// `pressure_3d`: Full pressure in Pa, shape `[nz][ny][nx]`
pub fn compute_pw(
    qvapor_3d: &[f64],
    pressure_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let n2d = ny * nx;

    (0..n2d)
        .into_par_iter()
        .map(|idx| {
            let j = idx / nx;
            let i = idx % nx;

            let q_col = extract_column(qvapor_3d, nz, ny, nx, j, i);
            let p_col = extract_column(pressure_3d, nz, ny, nx, j, i);

            // Ensure ordered from surface (high pressure) upward (low pressure)
            let (p_prof, q_prof) = if p_col.len() > 1 && p_col[0] < p_col[p_col.len() - 1] {
                let mut p = p_col;
                let mut q = q_col;
                p.reverse();
                q.reverse();
                (p, q)
            } else {
                (p_col, q_col)
            };

            let mut pw_val = 0.0;
            for k in 0..p_prof.len() - 1 {
                let dp = (p_prof[k] - p_prof[k + 1]).abs(); // Pa
                let q_avg = 0.5 * (q_prof[k].max(0.0) + q_prof[k + 1].max(0.0));
                pw_val += q_avg * dp;
            }
            pw_val / G // kg/m^2 = mm
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Composite Reflectivity
// ---------------------------------------------------------------------------

/// Composite reflectivity (max in column) in dBZ from REFL_10CM field.
///
/// `refl_3d`: Reflectivity in dBZ, shape `[nz][ny][nx]`
pub fn composite_reflectivity_from_refl(
    refl_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let n2d = ny * nx;

    (0..n2d)
        .into_par_iter()
        .map(|idx| {
            let j = idx / nx;
            let i = idx % nx;
            let mut max_dbz = -999.0_f64;
            for k in 0..nz {
                let val = refl_3d[k * n2d + j * nx + i];
                if val > max_dbz {
                    max_dbz = val;
                }
            }
            max_dbz.max(-30.0)
        })
        .collect()
}

/// Composite reflectivity (max in column) in dBZ from hydrometeor mixing ratios.
/// Uses Smith (1984) empirical approximation.
///
/// All 3D fields shape `[nz][ny][nx]`:
/// - `pressure_3d`: Pa
/// - `temperature_c_3d`: Celsius
/// - `qrain_3d`, `qsnow_3d`, `qgraup_3d`: kg/kg
pub fn composite_reflectivity_from_hydrometeors(
    pressure_3d: &[f64],
    temperature_c_3d: &[f64],
    qrain_3d: &[f64],
    qsnow_3d: &[f64],
    qgraup_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let n2d = ny * nx;

    (0..n2d)
        .into_par_iter()
        .map(|idx| {
            let j = idx / nx;
            let i = idx % nx;
            let mut max_dbz = -999.0_f64;

            for k in 0..nz {
                let flat_idx = k * n2d + j * nx + i;
                let p = pressure_3d[flat_idx]; // Pa
                let t_k = temperature_c_3d[flat_idx] + ZEROCNK; // K

                // Air density from ideal gas law
                let rho = p / (RD * t_k);

                let qr = qrain_3d[flat_idx].max(0.0);
                let qs = qsnow_3d[flat_idx].max(0.0);
                let qg = qgraup_3d[flat_idx].max(0.0);

                // Smith (1984) reflectivity factors
                let z_rain = 3.63e9 * (rho * qr).powf(1.75);
                let z_snow = 9.80e8 * (rho * qs).powf(1.75);
                let z_graupel = 4.33e9 * (rho * qg).powf(1.75);

                let z_total = z_rain + z_snow + z_graupel;
                let dbz = if z_total > 0.0 {
                    10.0 * z_total.log10()
                } else {
                    -999.0
                };

                if dbz > max_dbz {
                    max_dbz = dbz;
                }
            }

            max_dbz.max(-30.0)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Interpolation helpers for pressure-level profiles
// ---------------------------------------------------------------------------

/// Linearly interpolate a value at target_p from a pressure profile (decreasing)
/// and a corresponding value profile. Uses log-pressure interpolation.
fn interp_at_pressure(target_p: f64, p_prof: &[f64], values: &[f64]) -> f64 {
    if p_prof.is_empty() {
        return f64::NAN;
    }
    // Profiles are surface-first (decreasing pressure)
    if target_p >= p_prof[0] {
        return values[0];
    }
    if target_p <= p_prof[p_prof.len() - 1] {
        return values[values.len() - 1];
    }
    for k in 0..p_prof.len() - 1 {
        if p_prof[k] >= target_p && target_p >= p_prof[k + 1] {
            let log_p = target_p.ln();
            let log_p1 = p_prof[k].ln();
            let log_p2 = p_prof[k + 1].ln();
            let frac = (log_p - log_p1) / (log_p2 - log_p1);
            return values[k] + frac * (values[k + 1] - values[k]);
        }
    }
    values[values.len() - 1]
}

// ===========================================================================
// Classic Stability Indices
// ===========================================================================

/// Showalter Index: Lift 850 hPa parcel to 500 hPa.
///
/// SI = T_env(500) - T_parcel(500)
///
/// Profiles must be surface-first (decreasing pressure).
/// p in hPa, t and td in Celsius.
pub fn showalter_index(p: &[f64], t: &[f64], td: &[f64]) -> f64 {
    let t850 = interp_at_pressure(850.0, p, t);
    let td850 = interp_at_pressure(850.0, p, td);
    let t500_env = interp_at_pressure(500.0, p, t);

    let (p_lcl, t_lcl) = metfuncs::drylift(850.0, t850, td850);

    let thetam = {
        let theta_c = (t_lcl + ZEROCNK) * ((1000.0 / p_lcl).powf(ROCP)) - ZEROCNK;
        theta_c - metfuncs::wobf(theta_c) + metfuncs::wobf(t_lcl)
    };

    let t500_parcel = metfuncs::satlift(500.0, thetam);

    t500_env - t500_parcel
}

/// Lifted Index: Lift surface parcel to 500 hPa.
///
/// LI = T_env(500) - T_parcel(500)
///
/// Profiles must be surface-first (decreasing pressure).
/// p in hPa, t and td in Celsius.
pub fn lifted_index(p: &[f64], t: &[f64], td: &[f64]) -> f64 {
    if p.is_empty() {
        return f64::NAN;
    }

    let p_sfc = p[0];
    let t_sfc = t[0];
    let td_sfc = td[0];
    let t500_env = interp_at_pressure(500.0, p, t);

    let (p_lcl, t_lcl) = metfuncs::drylift(p_sfc, t_sfc, td_sfc);

    let thetam = {
        let theta_c = (t_lcl + ZEROCNK) * ((1000.0 / p_lcl).powf(ROCP)) - ZEROCNK;
        theta_c - metfuncs::wobf(theta_c) + metfuncs::wobf(t_lcl)
    };

    let t500_parcel = metfuncs::satlift(500.0, thetam);

    t500_env - t500_parcel
}

/// K-Index: (T850 - T500) + Td850 - (T700 - Td700)
///
/// All inputs in Celsius.
pub fn k_index(t850: f64, t700: f64, t500: f64, td850: f64, td700: f64) -> f64 {
    (t850 - t500) + td850 - (t700 - td700)
}

/// Total Totals Index: (T850 - T500) + (Td850 - T500)
///
/// All inputs in Celsius.
pub fn total_totals(t850: f64, t500: f64, td850: f64) -> f64 {
    (t850 - t500) + (td850 - t500)
}

/// Cross Totals: Td850 - T500
///
/// All inputs in Celsius.
pub fn cross_totals(td850: f64, t500: f64) -> f64 {
    td850 - t500
}

/// Vertical Totals: T850 - T500
///
/// All inputs in Celsius.
pub fn vertical_totals(t850: f64, t500: f64) -> f64 {
    t850 - t500
}

/// SWEAT Index (Severe Weather Threat Index).
///
/// SWEAT = 12*Td850 + 20*(TT-49) + 2*f850 + f500 + 125*(sin(d500-d850) + 0.2)
///
/// - tt: Total Totals index
/// - td850: 850 hPa dewpoint (Celsius)
/// - wspd850, wspd500: Wind speed at 850/500 hPa (knots)
/// - wdir850, wdir500: Wind direction at 850/500 hPa (degrees)
pub fn sweat_index(
    tt: f64,
    td850: f64,
    wspd850: f64,
    wdir850: f64,
    wspd500: f64,
    wdir500: f64,
) -> f64 {
    let term1 = if td850 > 0.0 { 12.0 * td850 } else { 0.0 };
    let term2 = if tt > 49.0 { 20.0 * (tt - 49.0) } else { 0.0 };
    let term3 = 2.0 * wspd850;
    let term4 = wspd500;

    let term5 = {
        let d_diff = wdir500 - wdir850;
        if wdir850 >= 130.0
            && wdir850 <= 250.0
            && wdir500 >= 210.0
            && wdir500 <= 310.0
            && d_diff > 0.0
            && wspd850 >= 15.0
            && wspd500 >= 15.0
        {
            125.0 * (d_diff.to_radians().sin() + 0.2)
        } else {
            0.0
        }
    };

    (term1 + term2 + term3 + term4 + term5).max(0.0)
}

/// Boyden Index: (Z700 - Z1000)/10 - T700 - 200
///
/// - z1000: Geopotential height at 1000 hPa (meters)
/// - z700: Geopotential height at 700 hPa (meters)
/// - t700: Temperature at 700 hPa (Celsius)
pub fn boyden_index(z1000: f64, z700: f64, t700: f64) -> f64 {
    (z700 - z1000) / 10.0 - t700 - 200.0
}

// ===========================================================================
// Severe Weather Composites (grid-based)
// ===========================================================================

/// Significant Hail Parameter (SHIP).
///
/// SHIP = (MUCAPE * MR * LR_700_500 * (-T500) * SHEAR_06) / 42_000_000
///
/// All inputs are flattened 2D grids of size nx*ny.
/// - cape: MUCAPE (J/kg)
/// - shear06: 0-6 km bulk shear (m/s)
/// - t500: Temperature at 500 hPa (Celsius, should be negative)
/// - lr_700_500: 700-500 hPa lapse rate (C/km)
/// - mr: Mixing ratio (g/kg)
pub fn significant_hail_parameter(
    cape: &[f64],
    shear06: &[f64],
    t500: &[f64],
    lr_700_500: &[f64],
    mr: &[f64],
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    let n = nx * ny;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mucape = cape[i].max(0.0);
            let mr_val = mr[i].max(0.0);
            let lr = lr_700_500[i].max(0.0);
            let t5 = (-t500[i]).max(0.0);
            let s06 = shear06[i].max(0.0);

            let ship = (mucape * mr_val * lr * t5 * s06) / 42_000_000.0;

            if mucape < 1300.0 {
                ship * (mucape / 1300.0)
            } else {
                ship
            }
        })
        .collect()
}

/// Derecho Composite Parameter (DCP).
///
/// DCP = (DCAPE/980) * (MUCAPE/2000) * (SHEAR_06/20) * (MU_MR/11)
///
/// All inputs are flattened 2D grids.
pub fn derecho_composite_parameter(
    dcape: &[f64],
    mu_cape: &[f64],
    shear06: &[f64],
    mu_mixing_ratio: &[f64],
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    let n = nx * ny;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let dcape_term = (dcape[i] / 980.0).max(0.0);
            let cape_term = (mu_cape[i] / 2000.0).max(0.0);
            let shear_term = (shear06[i] / 20.0).max(0.0);
            let mr_term = (mu_mixing_ratio[i] / 11.0).max(0.0);

            dcape_term * cape_term * shear_term * mr_term
        })
        .collect()
}

/// Enhanced Supercell Composite Parameter (SCP).
///
/// SCP = (MUCAPE / 1000) * (SRH / 50) * (SHEAR_06 / 40) * CIN_term
///
/// CIN_term = 1 if MUCIN > -40, else -40/MUCIN
///
/// All inputs are flattened 2D grids.
pub fn supercell_composite_parameter(
    mu_cape: &[f64],
    srh: &[f64],
    shear_06: &[f64],
    mu_cin: &[f64],
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    let n = nx * ny;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let cape_term = (mu_cape[i] / 1000.0).max(0.0);
            let srh_term = (srh[i] / 50.0).max(0.0);
            let shear_term = (shear_06[i] / 40.0).max(0.0);

            let cin_term = if mu_cin[i] > -40.0 {
                1.0
            } else {
                -40.0 / mu_cin[i].min(-0.01)
            };

            cape_term * srh_term * shear_term * cin_term
        })
        .collect()
}

/// Critical Angle between storm-relative inflow and 0-500m shear vector.
///
/// Returns angle in degrees (0-180). Values near 90 degrees favor tornadogenesis.
///
/// - u_storm, v_storm: Storm motion components (m/s)
/// - u_shear, v_shear: 0-500m shear vector components (m/s)
pub fn critical_angle(
    u_storm: &[f64],
    v_storm: &[f64],
    u_shear: &[f64],
    v_shear: &[f64],
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    let n = nx * ny;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let inflow_u = -u_storm[i];
            let inflow_v = -v_storm[i];
            let shear_u = u_shear[i];
            let shear_v = v_shear[i];

            let dot = inflow_u * shear_u + inflow_v * shear_v;
            let mag_inflow = (inflow_u * inflow_u + inflow_v * inflow_v).sqrt();
            let mag_shear = (shear_u * shear_u + shear_v * shear_v).sqrt();

            if mag_inflow < 0.01 || mag_shear < 0.01 {
                return f64::NAN;
            }

            let cos_angle = (dot / (mag_inflow * mag_shear)).clamp(-1.0, 1.0);
            cos_angle.acos().to_degrees()
        })
        .collect()
}

// ===========================================================================
// Fire Weather Indices
// ===========================================================================

/// Haines Index (Low Elevation variant).
///
/// Uses 950 and 850 hPa levels.
/// Component A: stability (T950 - T850), Component B: moisture (T850 - Td850)
///
/// Returns 2-6. Inputs in Celsius.
pub fn haines_index(t_950: f64, t_850: f64, td_850: f64) -> u8 {
    let delta_t = t_950 - t_850;
    let a: u8 = if delta_t <= 3.0 {
        1
    } else if delta_t <= 7.0 {
        2
    } else {
        3
    };

    let dewpoint_depression = t_850 - td_850;
    let b: u8 = if dewpoint_depression <= 5.0 {
        1
    } else if dewpoint_depression <= 9.0 {
        2
    } else {
        3
    };

    a + b
}

/// Fosberg Fire Weather Index (FFWI).
///
/// - t_f: Temperature in Fahrenheit
/// - rh: Relative humidity (percent, 0-100)
/// - wspd_mph: Wind speed in miles per hour
pub fn fosberg_fire_weather_index(t_f: f64, rh: f64, wspd_mph: f64) -> f64 {
    let rh = rh.clamp(0.0, 100.0);
    let emc = if rh <= 10.0 {
        0.03229 + 0.281073 * rh - 0.000578 * rh * t_f
    } else if rh <= 50.0 {
        2.22749 + 0.160107 * rh - 0.01478 * t_f
    } else {
        21.0606 + 0.005565 * rh * rh - 0.00035 * rh * t_f - 0.483199 * rh
    };

    let emc = emc / 30.0;
    let m = emc.max(0.0);

    let eta = 1.0 - 2.0 * m + 1.5 * m * m - 0.5 * m * m * m;

    let fw = eta * (1.0 + wspd_mph * wspd_mph).sqrt();
    (fw * 10.0 / 3.0).clamp(0.0, 100.0)
}

/// Hot-Dry-Windy Index (HDW).
///
/// HDW = VPD * wind_speed
///
/// - t_c: Temperature (Celsius)
/// - rh: Relative humidity (0-100)
/// - wspd_ms: Wind speed (m/s)
/// - vpd: Vapor pressure deficit (hPa). If 0, computed from T and RH.
pub fn hot_dry_windy(t_c: f64, rh: f64, wspd_ms: f64, vpd: f64) -> f64 {
    let vpd_val = if vpd > 0.0 {
        vpd
    } else {
        let es = metfuncs::vappres(t_c);
        let ea = es * (rh / 100.0);
        (es - ea).max(0.0)
    };

    vpd_val * wspd_ms
}

// ===========================================================================
// Winter Weather
// ===========================================================================

/// Dendritic Growth Zone: pressure bounds of the -12C to -18C layer.
///
/// Returns (p_top, p_bottom) in hPa. If the profile never enters
/// the -12 to -18 range, returns (NAN, NAN).
///
/// t_profile: Temperature (Celsius), p_profile: Pressure (hPa).
/// Profiles are surface-first (decreasing pressure).
pub fn dendritic_growth_zone(
    t_profile: &[f64],
    p_profile: &[f64],
) -> (f64, f64) {
    let n = t_profile.len().min(p_profile.len());
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }

    let mut p_top = f64::NAN;
    let mut p_bottom = f64::NAN;

    for k in 0..n {
        let t = t_profile[k];
        if t <= -12.0 && t >= -18.0 {
            if p_bottom.is_nan() {
                p_bottom = p_profile[k];
            }
            p_top = p_profile[k];
        }
    }

    if !p_bottom.is_nan() {
        for k in 0..n - 1 {
            if (t_profile[k] > -12.0 && t_profile[k + 1] <= -12.0)
                || (t_profile[k] <= -12.0 && t_profile[k + 1] > -12.0)
            {
                let frac = (-12.0 - t_profile[k]) / (t_profile[k + 1] - t_profile[k]);
                p_bottom = p_profile[k] + frac * (p_profile[k + 1] - p_profile[k]);
                break;
            }
        }
        for k in 0..n - 1 {
            if (t_profile[k] > -18.0 && t_profile[k + 1] <= -18.0)
                || (t_profile[k] <= -18.0 && t_profile[k + 1] > -18.0)
            {
                let frac = (-18.0 - t_profile[k]) / (t_profile[k + 1] - t_profile[k]);
                p_top = p_profile[k] + frac * (p_profile[k + 1] - p_profile[k]);
                break;
            }
        }
    }

    (p_top, p_bottom)
}

/// Check for a warm nose: a layer above the surface where T > 0C.
///
/// Returns true if there is a below-freezing layer followed by above-freezing
/// layer aloft.
///
/// t_profile: Temperature (Celsius), p_profile: Pressure (hPa).
/// Profiles are surface-first (decreasing pressure).
pub fn warm_nose_check(t_profile: &[f64], p_profile: &[f64]) -> bool {
    let n = t_profile.len().min(p_profile.len());
    if n < 3 {
        return false;
    }

    let mut found_below_zero = false;
    for k in 0..n {
        if t_profile[k] <= 0.0 {
            found_below_zero = true;
        }
        if found_below_zero && t_profile[k] > 0.0 {
            return true;
        }
    }

    false
}

/// Freezing Rain Composite.
///
/// A composite indicating freezing rain potential based on warm nose
/// characteristics and precipitation type.
///
/// Returns a scaled value 0-1 representing freezing rain likelihood.
///
/// - t_profile: Temperature (Celsius), p_profile: Pressure (hPa)
/// - precip_type: 0=none, 1=rain, 2=snow, 3=ice_pellets, 4=freezing_rain
pub fn freezing_rain_composite(
    t_profile: &[f64],
    p_profile: &[f64],
    precip_type: u8,
) -> f64 {
    let n = t_profile.len().min(p_profile.len());
    if n < 3 {
        return 0.0;
    }

    if t_profile[0] > 0.0 {
        return 0.0;
    }

    let mut warm_depth: f64 = 0.0;
    let mut warm_intensity: f64 = 0.0;

    let mut in_warm_layer = false;
    for k in 1..n {
        if t_profile[k] > 0.0 {
            if !in_warm_layer {
                in_warm_layer = true;
            }
            let dp = (p_profile[k - 1] - p_profile[k]).abs();
            warm_depth += dp;
            warm_intensity += t_profile[k] * dp;
        } else if in_warm_layer {
            break;
        }
    }

    if warm_depth < 1.0 {
        return 0.0;
    }

    let depth_factor = (warm_depth / 100.0).clamp(0.0, 1.0);
    let intensity_factor = (warm_intensity / (warm_depth * 3.0)).clamp(0.0, 1.0);
    let base_score = depth_factor * intensity_factor;

    let precip_boost = if precip_type == 4 { 1.0 } else { 0.5 };

    (base_score * precip_boost).clamp(0.0, 1.0)
}

// ===========================================================================
// Tropical / General
// ===========================================================================

/// Bulk Richardson Number: CAPE / (0.5 * shear^2)
///
/// - cape: CAPE (J/kg)
/// - shear_06_ms: 0-6 km bulk shear magnitude (m/s)
pub fn bulk_richardson_number(cape: f64, shear_06_ms: f64) -> f64 {
    let denom = 0.5 * shear_06_ms * shear_06_ms;
    if denom < 0.1 {
        return f64::NAN;
    }
    cape / denom
}

/// Convective Inhibition Depth: depth of the CIN layer in hPa.
///
/// Measures the pressure depth from the surface to the LFC where the parcel
/// is negatively buoyant.
///
/// Profiles are surface-first (decreasing pressure).
/// p in hPa, t and td in Celsius.
pub fn convective_inhibition_depth(p: &[f64], t: &[f64], td: &[f64]) -> f64 {
    if p.is_empty() {
        return 0.0;
    }

    let p_sfc = p[0];
    let t_sfc = t[0];
    let td_sfc = td[0];

    let (p_lcl, t_lcl) = metfuncs::drylift(p_sfc, t_sfc, td_sfc);
    let thetam = {
        let theta_c = (t_lcl + ZEROCNK) * ((1000.0 / p_lcl).powf(ROCP)) - ZEROCNK;
        theta_c - metfuncs::wobf(theta_c) + metfuncs::wobf(t_lcl)
    };

    for k in 0..p.len() {
        if p[k] > p_lcl {
            continue;
        }
        let t_env = t[k];
        let t_parcel = metfuncs::satlift(p[k], thetam);

        let tv_env = metfuncs::virtual_temp(t_env, p[k], td[k]);
        let tv_parcel = metfuncs::virtual_temp(t_parcel, p[k], t_parcel);

        if tv_parcel > tv_env {
            return p_sfc - p[k];
        }
    }

    p_sfc - p[p.len() - 1]
}


// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_profile() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let p  = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 200.0];
        let t  = vec![  30.0,  24.0,  18.0,   5.0, -15.0, -40.0, -55.0];
        let td = vec![  22.0,  18.0,  12.0,  -2.0, -25.0, -50.0, -65.0];
        (p, t, td)
    }

    #[test]
    fn test_showalter_index() {
        let (p, t, td) = test_profile();
        let si = showalter_index(&p, &t, &td);
        assert!(si.is_finite(), "SI should be finite, got {}", si);
        assert!(si < 10.0 && si > -20.0, "SI out of range: {}", si);
    }

    #[test]
    fn test_lifted_index() {
        let (p, t, td) = test_profile();
        let li = lifted_index(&p, &t, &td);
        assert!(li.is_finite(), "LI should be finite, got {}", li);
        assert!(li < 5.0, "LI unexpectedly large: {}", li);
    }

    #[test]
    fn test_k_index() {
        let ki = k_index(18.0, 5.0, -15.0, 12.0, -2.0);
        assert!((ki - 38.0).abs() < 0.01, "K-Index should be 38, got {}", ki);
    }

    #[test]
    fn test_total_totals() {
        let tt = total_totals(18.0, -15.0, 12.0);
        assert!((tt - 60.0).abs() < 0.01, "TT should be 60, got {}", tt);
    }

    #[test]
    fn test_cross_totals() {
        let ct = cross_totals(12.0, -15.0);
        assert!((ct - 27.0).abs() < 0.01, "CT should be 27, got {}", ct);
    }

    #[test]
    fn test_vertical_totals() {
        let vt = vertical_totals(18.0, -15.0);
        assert!((vt - 33.0).abs() < 0.01, "VT should be 33, got {}", vt);
    }

    #[test]
    fn test_sweat_index_basic() {
        let tt = 55.0;
        let td850 = 15.0;
        let si = sweat_index(tt, td850, 25.0, 200.0, 40.0, 250.0);
        assert!(si > 0.0, "SWEAT should be positive");
        assert!(si > 300.0, "SWEAT too low: {}", si);
    }

    #[test]
    fn test_sweat_index_no_shear_term() {
        let si = sweat_index(45.0, 10.0, 10.0, 90.0, 10.0, 90.0);
        assert!((si - 150.0).abs() < 0.01, "SWEAT should be 150, got {}", si);
    }

    #[test]
    fn test_boyden_index() {
        let bi = boyden_index(100.0, 3100.0, 5.0);
        assert!((bi - 95.0).abs() < 0.01, "Boyden should be 95, got {}", bi);
    }

    #[test]
    fn test_significant_hail_parameter() {
        let n = 4;
        let cape = vec![2500.0, 500.0, 0.0, 4000.0];
        let shear = vec![25.0, 15.0, 10.0, 35.0];
        let t500 = vec![-20.0, -15.0, -10.0, -25.0];
        let lr = vec![7.0, 5.0, 4.0, 8.0];
        let mr = vec![14.0, 10.0, 8.0, 16.0];

        let ship = significant_hail_parameter(&cape, &shear, &t500, &lr, &mr, 2, 2);
        assert_eq!(ship.len(), n);
        assert!(ship[0] > 1.0, "SHIP[0] should be > 1: {}", ship[0]);
        assert!(ship[2].abs() < 0.001, "SHIP[2] should be ~0: {}", ship[2]);
    }

    #[test]
    fn test_derecho_composite_parameter() {
        let dcape = vec![1200.0, 500.0];
        let mucape = vec![3000.0, 1000.0];
        let shear = vec![25.0, 10.0];
        let mr = vec![15.0, 8.0];

        let dcp = derecho_composite_parameter(&dcape, &mucape, &shear, &mr, 2, 1);
        assert_eq!(dcp.len(), 2);
        assert!(dcp[0] > 2.0, "DCP[0] should be > 2: {}", dcp[0]);
    }

    #[test]
    fn test_supercell_composite_enhanced() {
        let mucape = vec![3000.0, 500.0];
        let srh = vec![250.0, 50.0];
        let shear = vec![30.0, 15.0];
        let cin = vec![-20.0, -100.0];

        let scp = supercell_composite_parameter(&mucape, &srh, &shear, &cin, 2, 1);
        assert_eq!(scp.len(), 2);
        assert!((scp[0] - 11.25).abs() < 0.1, "SCP[0] expected ~11.25, got {}", scp[0]);
        assert!((scp[1] - 0.075).abs() < 0.01, "SCP[1] expected ~0.075, got {}", scp[1]);
    }

    #[test]
    fn test_critical_angle() {
        let u_storm = vec![10.0, 0.0];
        let v_storm = vec![0.0, 10.0];
        let u_shear = vec![0.0, 10.0];
        let v_shear = vec![10.0, 0.0];

        let ca = critical_angle(&u_storm, &v_storm, &u_shear, &v_shear, 2, 1);
        assert_eq!(ca.len(), 2);
        assert!((ca[0] - 90.0).abs() < 0.1, "CA[0] expected 90, got {}", ca[0]);
    }

    #[test]
    fn test_haines_index() {
        let hi = haines_index(25.0, 15.0, 5.0);
        assert_eq!(hi, 6);

        let hi_low = haines_index(15.0, 13.0, 11.0);
        assert_eq!(hi_low, 2);
    }

    #[test]
    fn test_fosberg_fire_weather_index() {
        let ffwi = fosberg_fire_weather_index(90.0, 15.0, 20.0);
        assert!(ffwi > 0.0, "FFWI should be positive: {}", ffwi);
        assert!(ffwi <= 100.0, "FFWI should be <= 100: {}", ffwi);
    }

    #[test]
    fn test_hot_dry_windy() {
        let hdw = hot_dry_windy(35.0, 15.0, 10.0, 0.0);
        assert!(hdw > 100.0, "HDW should be large for hot/dry/windy: {}", hdw);
    }

    #[test]
    fn test_hot_dry_windy_with_vpd() {
        let hdw = hot_dry_windy(35.0, 15.0, 10.0, 50.0);
        assert!((hdw - 500.0).abs() < 0.01, "HDW should be 500, got {}", hdw);
    }

    #[test]
    fn test_dendritic_growth_zone() {
        let p = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
        let t = vec![5.0, 0.0, -5.0, -15.0, -30.0, -50.0];

        let (p_top, p_bottom) = dendritic_growth_zone(&t, &p);
        assert!(p_top.is_finite(), "p_top should be finite");
        assert!(p_bottom.is_finite(), "p_bottom should be finite");
        assert!(p_top < p_bottom, "p_top ({}) should be less than p_bottom ({})", p_top, p_bottom);
        assert!((p_bottom - 745.0).abs() < 5.0, "p_bottom expected ~745, got {}", p_bottom);
    }

    #[test]
    fn test_dendritic_growth_zone_not_present() {
        let p = vec![1000.0, 925.0, 850.0];
        let t = vec![25.0, 20.0, 15.0];

        let (p_top, p_bottom) = dendritic_growth_zone(&t, &p);
        assert!(p_top.is_nan() && p_bottom.is_nan(), "Should be NAN when DGZ not present");
    }

    #[test]
    fn test_warm_nose_present() {
        let t = vec![-2.0, -1.0, 3.0, 5.0, 2.0, -5.0, -15.0];
        let p = vec![1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 700.0];
        assert!(warm_nose_check(&t, &p), "Should detect warm nose");
    }

    #[test]
    fn test_warm_nose_absent() {
        let t = vec![5.0, 3.0, 0.0, -5.0, -15.0];
        let p = vec![1000.0, 925.0, 850.0, 700.0, 500.0];
        assert!(!warm_nose_check(&t, &p), "Should not detect warm nose in monotonic cooling");
    }

    #[test]
    fn test_freezing_rain_composite() {
        let t = vec![-2.0, 3.0, 5.0, 3.0, -5.0, -15.0];
        let p = vec![1000.0, 950.0, 925.0, 900.0, 850.0, 700.0];

        let frc = freezing_rain_composite(&t, &p, 4);
        assert!(frc > 0.0, "FRC should be positive with warm nose: {}", frc);
        assert!(frc <= 1.0, "FRC should be <= 1: {}", frc);
    }

    #[test]
    fn test_freezing_rain_composite_warm_surface() {
        let t = vec![5.0, 3.0, 0.0, -5.0];
        let p = vec![1000.0, 925.0, 850.0, 700.0];

        let frc = freezing_rain_composite(&t, &p, 1);
        assert!(frc.abs() < 0.001, "FRC should be 0 with warm surface: {}", frc);
    }

    #[test]
    fn test_bulk_richardson_number() {
        let brn = bulk_richardson_number(2000.0, 20.0);
        assert!((brn - 10.0).abs() < 0.01, "BRN should be 10, got {}", brn);
    }

    #[test]
    fn test_bulk_richardson_number_low_shear() {
        let brn = bulk_richardson_number(1000.0, 0.01);
        assert!(brn.is_nan(), "BRN should be NAN with near-zero shear");
    }

    #[test]
    fn test_convective_inhibition_depth() {
        let (p, t, td) = test_profile();
        let cid = convective_inhibition_depth(&p, &t, &td);
        assert!(cid >= 0.0, "CIN depth should be non-negative: {}", cid);
        assert!(cid < 900.0, "CIN depth unreasonably large: {}", cid);
    }

    #[test]
    fn test_interp_at_pressure() {
        let p = vec![1000.0, 850.0, 700.0, 500.0];
        let v = vec![30.0, 18.0, 5.0, -15.0];
        let val = interp_at_pressure(850.0, &p, &v);
        assert!((val - 18.0).abs() < 0.1, "Should be 18 at 850 hPa, got {}", val);
    }
}
