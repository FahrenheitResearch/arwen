//! Radar diagnostic variables: dbz, maxdbz
//!
//! Simulated reflectivity from hydrometeor mixing ratios following
//! wrf-python's `wrf_user_dbz.f90` (`CALCDBZ`) subroutine.
//!
//! Uses constant intercept parameters (ivarint=0) and bright-band
//! correction (iliqskin=1), matching the wrf-python defaults.

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;
use rayon::prelude::*;

// --- Physical constants (wrf_constants) ---
const GAMMA_SEVEN: f64 = 720.0;
const PI: f64 = std::f64::consts::PI;
const RD: f64 = 287.04;
const CELKEL: f64 = 273.15;
const RHOWAT: f64 = 1000.0;
const ALPHA: f64 = 0.224; // |K_ice|^2 / |K_water|^2

// --- Hydrometeor densities (kg m^-3) ---
const RHO_R: f64 = 1000.0; // rain
const RHO_S: f64 = 100.0; // snow
const RHO_G: f64 = 400.0; // graupel

// --- Constant intercept parameters (m^-4) ---
const RN0_R: f64 = 8.0e6;
const RN0_S: f64 = 2.0e7;
const RN0_G: f64 = 4.0e6;

// --- Variable intercept parameters (Thompson microphysics) ---
const R1: f64 = 1.0e-15;
const RON: f64 = 8.0e6;
const RON2: f64 = 1.0e10;
const GON: f64 = 5.0e7;
const RON_MIN: f64 = 8.0e6;
const RON_QR0: f64 = 0.00010;
const RON_DELQR0: f64 = 0.25 * RON_QR0;
const RON_CONST1R: f64 = (RON2 - RON_MIN) * 0.5;
const RON_CONST2R: f64 = (RON2 + RON_MIN) * 0.5;

/// Simulated reflectivity (dBZ). `[nz, ny, nx]`
///
/// Matches wrf-python's `CALCDBZ` from `wrf_user_dbz.f90`.
/// Defaults: constant intercepts (ivarint=0), no bright-band (iliqskin=0).
/// Set `opts.use_varint=true` for Thompson variable intercepts.
/// Set `opts.use_liqskin=true` for bright-band correction.
///
/// When QSNOW is not present in the file (sn0=0 behavior), rain mixing
/// ratio is reassigned to snow below freezing.
pub fn compute_dbz(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let tk = f.temperature(t)?;
    let pres = f.full_pressure(t)?;
    let qv = f.qvapor(t)?;

    // Try to read hydrometeor fields; default to zero if absent
    let qr = f
        .read_var("QRAIN", t)
        .unwrap_or_else(|_| vec![0.0; f.nxyz()]);
    let have_snow = f.read_var("QSNOW", t);
    let sn0 = have_snow.is_ok();
    let qs = have_snow.unwrap_or_else(|_| vec![0.0; f.nxyz()]);
    let qg = f
        .read_var("QGRAUP", t)
        .unwrap_or_else(|_| vec![0.0; f.nxyz()]);

    // --- Precompute factors exactly as in CALCDBZ ---
    // factor_r = GAMMA_SEVEN * 1e18 * (1/(PI*RHO_R))^1.75
    // factor_s = GAMMA_SEVEN * 1e18 * (1/(PI*RHO_S))^1.75 * (RHO_S/RHOWAT)^2 * ALPHA
    // factor_g = GAMMA_SEVEN * 1e18 * (1/(PI*RHO_G))^1.75 * (RHO_G/RHOWAT)^2 * ALPHA
    let factor_r = GAMMA_SEVEN * 1.0e18 * (1.0 / (PI * RHO_R)).powf(1.75);
    let factor_s =
        GAMMA_SEVEN * 1.0e18 * (1.0 / (PI * RHO_S)).powf(1.75) * (RHO_S / RHOWAT).powi(2) * ALPHA;
    let factor_g =
        GAMMA_SEVEN * 1.0e18 * (1.0 / (PI * RHO_G)).powf(1.75) * (RHO_G / RHOWAT).powi(2) * ALPHA;

    // Match wrf-python defaults: ivarint=0 (constant intercepts),
    // iliqskin=0 (no bright-band correction).
    let ivarint = opts.use_varint.unwrap_or(false);
    let iliqskin = opts.use_liqskin.unwrap_or(false);

    // Bright-band factors (only used when iliqskin=true)
    let factorb_s = factor_s / ALPHA;
    let factorb_g = factor_g / ALPHA;

    let n = f.nxyz();

    let dbz: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let t_k = tk[i];
            let qvp = qv[i].max(0.0);

            // Virtual temperature: full formula from CALCDBZ
            let virtual_t = t_k * (0.622 + qvp) / (0.622 * (1.0 + qvp));
            let rhoair = pres[i] / (RD * virtual_t);

            // Hydrometeor mixing ratios (clamp to zero)
            let mut qra = qr[i].max(0.0);
            let mut qsn = qs[i].max(0.0);
            let qgr = qg[i].max(0.0);

            // sn0=0 behavior: no separate snow variable, so below freezing
            // move rain to snow
            if !sn0 && t_k < CELKEL {
                qsn = qra;
                qra = 0.0;
            }

            // Bright-band correction (iliqskin): above freezing, frozen
            // particles scatter as liquid (drop ALPHA factor).
            let (fs, fg) = if iliqskin && t_k > CELKEL {
                (factorb_s, factorb_g)
            } else {
                (factor_s, factor_g)
            };

            // Variable intercept parameters (Thompson microphysics)
            let (ronv, sonv, gonv) = if ivarint {
                let temp_c = (t_k - CELKEL).min(-0.001);
                let sonv_v = (2.0e6 * (-0.12 * temp_c).exp()).min(2.0e8);

                let gonv_v = if qgr > R1 {
                    let g = 2.38 * (PI * RHO_G / (rhoair * qgr)).powf(0.92);
                    g.max(1.0e4).min(GON)
                } else {
                    GON
                };

                let ronv_v = if qra > R1 {
                    RON_CONST1R * ((RON_QR0 - qra) / RON_DELQR0).tanh() + RON_CONST2R
                } else {
                    RON2
                };

                (ronv_v, sonv_v, gonv_v)
            } else {
                (RN0_R, RN0_S, RN0_G)
            };

            // Z_e = factor * (rhoair*q)^1.75 / N0^0.75
            let z_r = factor_r * (rhoair * qra).powf(1.75) / ronv.powf(0.75);
            let z_s = fs * (rhoair * qsn).powf(1.75) / sonv.powf(0.75);
            let z_g = fg * (rhoair * qgr).powf(1.75) / gonv.powf(0.75);

            let z_e = (z_r + z_s + z_g).max(0.001);
            10.0 * z_e.log10()
        })
        .collect();

    Ok(dbz)
}

/// Maximum (composite) reflectivity (dBZ). `[ny, nx]`
pub fn compute_maxdbz(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let dbz_3d = compute_dbz(f, t, opts)?;
    let nxy = f.nxy();
    let nz = f.nz;

    let mut maxdbz = vec![f64::NEG_INFINITY; nxy];
    for k in 0..nz {
        let offset = k * nxy;
        for ij in 0..nxy {
            let val = dbz_3d[offset + ij];
            if val > maxdbz[ij] {
                maxdbz[ij] = val;
            }
        }
    }
    Ok(maxdbz)
}
