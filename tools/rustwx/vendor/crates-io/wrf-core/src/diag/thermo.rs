//! Thermodynamic diagnostic variables:
//! temp, tc, theta, theta_e, tv, twb, td, rh

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// 2-m temperature (K). `[ny, nx]`
pub fn compute_t2(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.t2_for_opts(t, opts)
}

/// 2-m virtual temperature (K). `[ny, nx]`
/// Tv = T2 * (1 + 0.61 * Q2). Supports lake_interp.
pub fn compute_tv2m(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let t2 = f.t2_for_opts(t, opts)?;
    let q2 = f.q2_for_opts(t, opts)?;
    Ok(t2
        .iter()
        .zip(q2.iter())
        .map(|(tk, q)| tk * (1.0 + 0.61 * q.max(0.0)))
        .collect())
}

/// Temperature (K). `[nz, ny, nx]`
pub fn compute_temp(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.temperature(t).map(|v| v.to_vec())
}

/// Temperature (°C). `[nz, ny, nx]`
pub fn compute_tc(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.temperature_c(t).map(|v| v.to_vec())
}

/// Potential temperature (K). `[nz, ny, nx]`
pub fn compute_theta(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.full_theta(t).map(|v| v.to_vec())
}

/// Equivalent potential temperature (K). `[nz, ny, nx]`
pub fn compute_theta_e(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;

    // Dewpoint from mixing ratio: td_c = dewpoint_from_q(q, p_hpa)
    let result: Vec<f64> = p_hpa
        .iter()
        .zip(tc.iter())
        .zip(qv.iter())
        .map(|((p, t_c), q)| {
            let td_c = crate::met::thermo::dewpoint_from_rh(*t_c, rh_from_q(*q, *p, *t_c));
            crate::met::thermo::equivalent_potential_temperature(*p, *t_c, td_c) + 273.15
        })
        .collect();
    Ok(result)
}

/// Virtual temperature (K). `[nz, ny, nx]`
pub fn compute_tv(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let tk = f.temperature(t)?;
    let qv = f.qvapor(t)?;
    // Tv = T * (1 + 0.61 * qv)
    Ok(tk
        .iter()
        .zip(qv.iter())
        .map(|(t, q)| t * (1.0 + 0.61 * q.max(0.0)))
        .collect())
}

/// Wet-bulb temperature (K). `[nz, ny, nx]`
pub fn compute_twb(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;

    Ok(p_hpa
        .iter()
        .zip(tc.iter())
        .zip(qv.iter())
        .map(|((p, t_c), q)| {
            let td_c = dewpoint_from_q(*q, *p);
            crate::met::thermo::wet_bulb_temperature(*p, *t_c, td_c) + 273.15
        })
        .collect())
}

/// Dewpoint temperature (°C). `[nz, ny, nx]`
pub fn compute_td(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let qv = f.qvapor(t)?;

    Ok(p_hpa
        .iter()
        .zip(qv.iter())
        .map(|(p, q)| dewpoint_from_q(*q, *p))
        .collect())
}

/// Relative humidity (%). `[nz, ny, nx]`
pub fn compute_rh(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;

    Ok(p_hpa
        .iter()
        .zip(tc.iter())
        .zip(qv.iter())
        .map(|((p, t_c), q)| rh_from_q(*q, *p, *t_c).clamp(0.0, 100.0))
        .collect())
}

// ── Helpers ──

/// Compute dewpoint (°C) from mixing ratio (kg/kg) and pressure (hPa).
fn dewpoint_from_q(q_kgkg: f64, p_hpa: f64) -> f64 {
    let q = q_kgkg.max(1e-10);
    // Vapor pressure from mixing ratio: e = q * p / (0.622 + q)
    let e_hpa = q * p_hpa / (0.622 + q);
    // Dewpoint from vapor pressure (Bolton 1980 inverse)
    let ln_e = (e_hpa / 6.112).max(1e-10).ln();
    (243.5 * ln_e) / (17.67 - ln_e)
}

/// Compute RH (%) from mixing ratio (kg/kg), pressure (hPa), and temperature (°C).
fn rh_from_q(q_kgkg: f64, p_hpa: f64, t_c: f64) -> f64 {
    let q = q_kgkg.max(0.0);
    let e_hpa = q * p_hpa / (0.622 + q);
    let es_hpa = 6.112 * (17.67 * t_c / (t_c + 243.5)).exp();
    (e_hpa / es_hpa * 100.0).clamp(0.0, 100.0)
}
