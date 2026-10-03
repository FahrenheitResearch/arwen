//! Severe weather composite diagnostic variables:
//! stp, scp, ehi, ecape_scp, ecape_ehi, critical_angle, ship, bri

use crate::compute::{ComputeOpts, StormMotionMethod};
use crate::diag::cape::effective_inflow_layer_grid;
use crate::error::WrfResult;
use crate::file::WrfFile;
use rayon::prelude::*;

const SURFACE_LAYER_HEIGHT_M: f64 = 0.0;

fn resolved_storm_motion_method(opts: &ComputeOpts) -> StormMotionMethod {
    opts.storm_motion_method
        .unwrap_or(StormMotionMethod::PressureWeighted)
}

fn opts_with_default_parcel_type(opts: &ComputeOpts, parcel_type: &str) -> ComputeOpts {
    let mut derived = opts.clone();
    if derived.parcel_type.is_none() {
        derived.parcel_type = Some(parcel_type.to_string());
    }
    derived
}

fn ehi_from_components(cape: &[f64], srh: &[f64]) -> Vec<f64> {
    crate::met::composite::compute_ehi(cape, srh)
}

fn scp_from_components(cape: &[f64], effective_srh: &[f64], ebwd: &[f64]) -> Vec<f64> {
    crate::met::composite::compute_scp(cape, effective_srh, ebwd)
}

fn build_augmented_wind_profile(
    u_3d: &[f64],
    v_3d: &[f64],
    h_agl: &[f64],
    u10: f64,
    v10: f64,
    nz: usize,
    nxy: usize,
    ij: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut u_prof = Vec::with_capacity(nz + 1);
    let mut v_prof = Vec::with_capacity(nz + 1);
    let mut h_prof = Vec::with_capacity(nz + 1);
    u_prof.push(u10);
    v_prof.push(v10);
    h_prof.push(SURFACE_LAYER_HEIGHT_M);

    for k in 0..nz {
        let idx = k * nxy + ij;
        u_prof.push(u_3d[idx]);
        v_prof.push(v_3d[idx]);
        h_prof.push(h_agl[idx]);
    }

    (u_prof, v_prof, h_prof)
}

fn pressure_weighted_layer_mean(
    u_prof: &[f64],
    v_prof: &[f64],
    h_prof: &[f64],
    p_prof: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    let u_bot = interp_wind_at_height(u_prof, v_prof, h_prof, bottom_m).0;
    let v_bot = interp_wind_at_height(u_prof, v_prof, h_prof, bottom_m).1;
    let u_top = interp_wind_at_height(u_prof, v_prof, h_prof, top_m).0;
    let v_top = interp_wind_at_height(u_prof, v_prof, h_prof, top_m).1;

    let mut us = Vec::with_capacity(u_prof.len() + 2);
    let mut vs = Vec::with_capacity(v_prof.len() + 2);
    let mut ps = Vec::with_capacity(p_prof.len() + 2);

    us.push(u_bot);
    vs.push(v_bot);
    ps.push(interp_scalar_at_height(p_prof, h_prof, bottom_m));

    for i in 0..h_prof.len() {
        if h_prof[i] > bottom_m && h_prof[i] < top_m {
            us.push(u_prof[i]);
            vs.push(v_prof[i]);
            ps.push(p_prof[i]);
        }
    }

    us.push(u_top);
    vs.push(v_top);
    ps.push(interp_scalar_at_height(p_prof, h_prof, top_m));

    let mut sum_u = 0.0;
    let mut sum_v = 0.0;
    let mut total_dp = 0.0;
    for i in 0..(ps.len() - 1) {
        let dp = (ps[i] - ps[i + 1]).abs();
        sum_u += 0.5 * (us[i] + us[i + 1]) * dp;
        sum_v += 0.5 * (vs[i] + vs[i + 1]) * dp;
        total_dp += dp;
    }

    if total_dp <= 0.0 {
        (u_bot, v_bot)
    } else {
        (sum_u / total_dp, sum_v / total_dp)
    }
}

fn interp_scalar_at_height(values: &[f64], h_prof: &[f64], target_h: f64) -> f64 {
    for k in 0..h_prof.len() - 1 {
        if h_prof[k] <= target_h && h_prof[k + 1] > target_h {
            let frac = (target_h - h_prof[k]) / (h_prof[k + 1] - h_prof[k]);
            return values[k] + frac * (values[k + 1] - values[k]);
        }
    }
    if target_h <= h_prof[0] {
        values[0]
    } else {
        values[values.len() - 1]
    }
}

pub fn compute_effective_bulk_wind_difference(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<Vec<f64>> {
    let effective_layers = effective_inflow_layer_grid(f, t, opts)?;
    let h_agl = f.height_agl(t)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let u10 = f.u10(t)?;
    let v10 = f.v10(t)?;

    let nz = f.nz;
    let nxy = f.nx * f.ny;

    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let layer = match effective_layers.layer(ij) {
                Some(layer) => layer,
                None => return 0.0,
            };
            let mu_el_h = match layer.mu_el_h {
                Some(el_h) if el_h > layer.base_h => el_h,
                _ => return 0.0,
            };

            let top_h = layer.base_h + 0.5 * (mu_el_h - layer.base_h);
            if top_h <= layer.base_h {
                return 0.0;
            }

            let (u_prof, v_prof, h_prof) =
                build_augmented_wind_profile(&u, &v, &h_agl, u10[ij], v10[ij], nz, nxy, ij);
            let (u_bot, v_bot) = interp_wind_at_height(&u_prof, &v_prof, &h_prof, layer.base_h);
            let (u_top, v_top) = interp_wind_at_height(&u_prof, &v_prof, &h_prof, top_h);
            let du = u_top - u_bot;
            let dv = v_top - v_bot;
            (du * du + dv * dv).sqrt()
        })
        .collect())
}

/// Significant Tornado Parameter -- fixed layer (dimensionless). `[ny, nx]`
///
/// Thompson et al. (2003) formulation with proper term limits.
/// Uses SURFACE-BASED parcel for CAPE and LCL, 0-1 km SRH, 0-6 km shear.
///   STP = cape_term * lcl_term * srh_term * shear_term
///
/// SRH is computed through the canonical compute_srh_field path (earth-rotated + 10m prepend).
pub fn compute_stp(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let h_agl = f.height_agl(t)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    let (sbcape, _, lcl, _) =
        crate::diag::cape::compute_cape_fields(f, t, "sb", None, opts.lake_interp)?;

    // 0-1 km SRH via canonical path (earth-rotated winds + 10m prepend)
    let srh1 = crate::diag::srh::compute_srh_field(
        f,
        t,
        1000.0,
        opts.storm_motion.as_ref(),
        opts.storm_motion_method,
    )?;

    // 0-6 km shear magnitude
    let shear6 = crate::met::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

    Ok(stp_fixed_from_components(&sbcape, &lcl, &srh1, &shear6))
}

/// Effective-layer Significant Tornado Parameter (dimensionless). `[ny, nx]`
///
/// Uses MIXED-LAYER parcel for CAPE, LCL, and CIN.
/// Uses effective inflow layer SRH and effective bulk wind difference (EBWD).
/// Includes CIN term: (200 + mlCIN) / 150.
///
/// STP_eff = (mlCAPE/1500) * ((2000-mlLCL)/1000) * (ESRH/150) * (EBWD/20) * ((200+mlCIN)/150)
///
/// Effective SRH uses earth-rotated winds with 10m prepend via compute_effective_srh.
pub fn compute_stp_effective(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (mlcape, mlcin, lcl, _) =
        crate::diag::cape::compute_cape_fields(f, t, "ml", None, opts.lake_interp)?;

    // Effective-layer SRH via canonical path (earth-rotated winds + 10m prepend)
    let eff_srh = crate::diag::srh::compute_effective_srh(f, t, opts)?;
    let ebwd = compute_effective_bulk_wind_difference(f, t, opts)?;

    Ok(stp_eff_from_components(
        &mlcape, &lcl, &mlcin, &eff_srh, &ebwd,
    ))
}

/// Generic STP dispatcher: uses opts.layer_type to choose fixed or effective.
///
/// - `"effective"` -> `compute_stp_effective`
/// - anything else (default) -> `compute_stp` (fixed layer)
pub fn compute_stp_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    match opts.layer_type.as_deref() {
        Some("effective") => compute_stp_effective(f, t, opts),
        _ => compute_stp(f, t, opts),
    }
}

/// Fixed-layer STP: 4-term formula (no CIN).
///   STP = (sbCAPE/1500) * ((2000-LCL)/1000) * (SRH/150) * (shear/20)
fn stp_fixed_from_components(cape: &[f64], lcl: &[f64], srh: &[f64], shear: &[f64]) -> Vec<f64> {
    cape.iter()
        .zip(lcl.iter())
        .zip(srh.iter())
        .zip(shear.iter())
        .map(|(((c, l), s), sh)| {
            let cape_term = (c / 1500.0).max(0.0);
            let lcl_term = if *l >= 2000.0 {
                0.0
            } else if *l <= 1000.0 {
                1.0
            } else {
                (2000.0 - l) / 1000.0
            };
            let srh_term = (s / 150.0).max(0.0);
            let shear_term = if *sh < 12.5 {
                0.0
            } else if *sh >= 30.0 {
                1.5
            } else {
                sh / 20.0
            };
            cape_term * lcl_term * srh_term * shear_term
        })
        .collect()
}

/// Effective-layer STP: 5-term formula with CIN.
///   STP_eff = (mlCAPE/1500) * ((2000-mlLCL)/1000) * (ESRH/150) * (EBWD/20) * ((200+mlCIN)/150)
fn stp_eff_from_components(
    cape: &[f64],
    lcl: &[f64],
    cin: &[f64],
    srh: &[f64],
    shear: &[f64],
) -> Vec<f64> {
    cape.iter()
        .zip(lcl.iter())
        .zip(cin.iter())
        .zip(srh.iter())
        .zip(shear.iter())
        .map(|((((c, l), ci), s), sh)| {
            let cape_term = (c / 1500.0).max(0.0);
            let lcl_term = if *l >= 2000.0 {
                0.0
            } else if *l <= 1000.0 {
                1.0
            } else {
                (2000.0 - l) / 1000.0
            };
            let srh_term = (s / 150.0).max(0.0);
            let shear_term = if *sh < 12.5 {
                0.0
            } else if *sh >= 30.0 {
                1.5
            } else {
                sh / 20.0
            };
            // CIN term: (200 + mlCIN) / 150, clamped to [0, 1]
            // CIN is negative, so 200 + CIN shrinks toward 0 as CIN gets more negative
            let cin_term = ((200.0 + ci) / 150.0).clamp(0.0, 1.0);
            cape_term * lcl_term * srh_term * shear_term * cin_term
        })
        .collect()
}

fn vtp_mod_from_components(
    mlcape: &[f64],
    esrh: &[f64],
    ebwd: &[f64],
    mllcl: &[f64],
    mlcin: &[f64],
    ml3cape: &[f64],
    lr700_500: &[f64],
) -> Vec<f64> {
    let n = mlcape.len();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let ebwd_term = if ebwd[i] <= 20.0 {
            0.0
        } else if ebwd[i] >= 45.0 {
            1.5
        } else {
            ebwd[i] / 30.0
        };
        let mllcl_term = if mllcl[i] >= 1750.0 {
            0.0
        } else if mllcl[i] <= 750.0 {
            1.0
        } else {
            (1750.0 - mllcl[i]) / 750.0
        };
        let mlcin_term = if mlcin[i] <= -200.0 {
            0.0
        } else if mlcin[i] >= -50.0 {
            1.0
        } else {
            (mlcin[i] + 200.0) / 150.0
        };
        let ml3cape_term = if ml3cape[i] >= 100.0 {
            2.0
        } else {
            ml3cape[i] / 50.0
        };
        let lr_term = if lr700_500[i] <= 4.5 {
            0.0
        } else if lr700_500[i] >= 8.5 {
            2.0
        } else {
            (lr700_500[i] - 4.5) / 2.0
        };

        let p1 = (mlcape[i] / 1700.0) * (esrh[i] / 250.0) * ebwd_term * mllcl_term;
        let p2 = mlcin_term * ml3cape_term * lr_term;
        out.push(p1 * p2);
    }

    out
}

fn fixed_layer_shear_term(shear6: f64) -> f64 {
    if shear6 < 12.5 {
        0.0
    } else if shear6 > 30.0 {
        1.5
    } else {
        shear6 / 20.0
    }
}

fn tornadic_low_level_limit_exceeded(mllcl: f64, mlcin: f64, sbcin: f64) -> bool {
    mllcl > 1700.0 || mlcin < -100.0 || sbcin < -200.0
}

fn tehi_from_components(
    srh1: &[f64],
    mlcape: &[f64],
    ml3cape: &[f64],
    shear6: &[f64],
    mllcl: &[f64],
    mlcin: &[f64],
    sbcin: &[f64],
) -> Vec<f64> {
    let n = srh1.len();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let mut ml3cape_term = if ml3cape[i] > 300.0 {
            1.5
        } else {
            ml3cape[i] / 200.0
        };
        if mlcape[i] > 1500.0 {
            ml3cape_term = ml3cape_term.max(1.0);
        }

        let tehi =
            ((srh1[i] * mlcape[i]) / 160000.0) * ml3cape_term * fixed_layer_shear_term(shear6[i]);

        if tornadic_low_level_limit_exceeded(mllcl[i], mlcin[i], sbcin[i]) || tehi < 0.0 {
            out.push(0.0);
        } else {
            out.push(tehi);
        }
    }

    out
}

fn tts_from_components(
    srh1: &[f64],
    ml3cape: &[f64],
    mlcape: &[f64],
    shear6: &[f64],
    mllcl: &[f64],
    mlcin: &[f64],
    sbcin: &[f64],
) -> Vec<f64> {
    let n = srh1.len();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let ml3cape_capped = ml3cape[i].min(150.0);
        let mlcape_term = if mlcape[i] < 2000.0 {
            1.0
        } else if mlcape[i] > 3000.0 {
            1.5
        } else {
            mlcape[i] / 2000.0
        };

        let tts =
            ((srh1[i] * ml3cape_capped) / 6500.0) * mlcape_term * fixed_layer_shear_term(shear6[i]);

        if tornadic_low_level_limit_exceeded(mllcl[i], mlcin[i], sbcin[i]) || tts < 0.0 {
            out.push(0.0);
        } else {
            out.push(tts);
        }
    }

    out
}

fn compute_tornadic_low_level_components(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<(
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    let mut ml_opts = opts.clone();
    ml_opts.parcel_type = Some("ml".into());
    ml_opts.top_m = None;
    ml_opts.bottom_m = None;
    ml_opts.bottom_p = None;
    ml_opts.top_p = None;

    let mlcape = crate::diag::cape::compute_mlcape(f, t, &ml_opts)?;
    let mlcin = crate::diag::cape::compute_mlcin(f, t, &ml_opts)?;
    let mllcl = crate::diag::cape::compute_lcl(f, t, &ml_opts)?;

    let mut ml3cape_opts = ml_opts.clone();
    ml3cape_opts.top_m = Some(3000.0);
    let ml3cape = crate::diag::cape::compute_mlcape(f, t, &ml3cape_opts)?;

    let sbcin = crate::diag::cape::compute_sbcin(f, t, opts)?;
    let srh1 = crate::diag::srh::compute_srh_field(
        f,
        t,
        1000.0,
        opts.storm_motion.as_ref(),
        opts.storm_motion_method,
    )?;
    let shear6 = crate::diag::srh::compute_shear_0_6km(f, t, opts)?;

    Ok((srh1, mlcape, ml3cape, shear6, mllcl, mlcin, sbcin))
}

/// Supercell Composite Parameter (dimensionless). `[ny, nx]`
///
/// Uses MUCAPE, effective SRH, and effective bulk wind difference (EBWD).
pub fn compute_scp(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (mucape, _, _, _) =
        crate::diag::cape::compute_cape_fields(f, t, "mu", None, opts.lake_interp)?;

    let eff_srh = crate::diag::srh::compute_effective_srh(f, t, opts)?;
    let ebwd = compute_effective_bulk_wind_difference(f, t, opts)?;

    Ok(scp_from_components(&mucape, &eff_srh, &ebwd))
}

/// Experimental ECAPE-based Supercell Composite Parameter (dimensionless). `[ny, nx]`
///
/// Replaces the CAPE term in SCP with ECAPE while preserving effective SRH and EBWD.
/// Defaults to `parcel_type="mu"` when no explicit parcel type is provided.
pub fn compute_ecape_scp(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let ecape_opts = opts_with_default_parcel_type(opts, "mu");
    let ecape = crate::diag::ecape::compute_ecape(f, t, &ecape_opts)?;
    let eff_srh = crate::diag::srh::compute_effective_srh(f, t, opts)?;
    let ebwd = compute_effective_bulk_wind_difference(f, t, opts)?;

    Ok(scp_from_components(&ecape, &eff_srh, &ebwd))
}

/// Energy-Helicity Index (dimensionless). `[ny, nx]`
///
/// EHI = (CAPE * SRH) / 160000
///
/// SRH depth is configurable via `opts.depth_m` (default 1000 m for 0-1 km EHI).
/// SRH is computed through the canonical compute_srh_field path (earth-rotated + 10m prepend).
pub fn compute_ehi(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (sbcape, _, _, _) =
        crate::diag::cape::compute_cape_fields(f, t, "sb", None, opts.lake_interp)?;

    // SRH via canonical path (earth-rotated winds + 10m prepend)
    let srh_depth = opts.depth_m.unwrap_or(1000.0);
    let srh = crate::diag::srh::compute_srh_field(
        f,
        t,
        srh_depth,
        opts.storm_motion.as_ref(),
        opts.storm_motion_method,
    )?;

    // EHI = (CAPE * SRH) / 160000
    Ok(ehi_from_components(&sbcape, &srh))
}

/// Experimental ECAPE-based Energy-Helicity Index (dimensionless). `[ny, nx]`
///
/// Replaces the CAPE term in EHI with ECAPE while preserving the same SRH depth/storm-motion path.
/// Defaults to `parcel_type="sb"` when no explicit parcel type is provided.
pub fn compute_ecape_ehi(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let ecape_opts = opts_with_default_parcel_type(opts, "sb");
    let ecape = crate::diag::ecape::compute_ecape(f, t, &ecape_opts)?;

    let srh_depth = opts.depth_m.unwrap_or(1000.0);
    let srh = crate::diag::srh::compute_srh_field(
        f,
        t,
        srh_depth,
        opts.storm_motion.as_ref(),
        opts.storm_motion_method,
    )?;

    Ok(ehi_from_components(&ecape, &srh))
}

/// Tornadic 0-1 km Energy-Helicity Index (dimensionless). `[ny, nx]`
///
/// Mirrors the SPC beta `tehi` product:
///
/// TEHI = ((SRH1 * mlCAPE)/160000) * ((mlCAPE3/200 J kg^-1)) * (6BWD/20 m s^-1)
///
/// where:
/// - the mlCAPE3 term is set to 1.0 if total mlCAPE > 1500 J/kg
/// - the mlCAPE3 term is capped at 1.5 if mlCAPE3 > 300 J/kg
/// - the 6BWD term is capped at 1.5 for 6BWD > 30 m/s
/// - the 6BWD term is set to 0.0 for 6BWD < 12.5 m/s
/// - the entire index is set to 0.0 if mlLCL > 1700 m AGL,
///   mlCIN < -100 J/kg, sbCIN < -200 J/kg, or TEHI < 0
///
/// Naming note:
/// On the SPC mesoanalysis page, `tehi` is Tornadic 0-1 km EHI.
/// `tts` is Tornadic Tilting and Stretching.
pub fn compute_tehi(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (srh1, mlcape, ml3cape, shear6, mllcl, mlcin, sbcin) =
        compute_tornadic_low_level_components(f, t, opts)?;

    Ok(tehi_from_components(
        &srh1, &mlcape, &ml3cape, &shear6, &mllcl, &mlcin, &sbcin,
    ))
}

/// Tornadic Tilting and Stretching (dimensionless). `[ny, nx]`
///
/// Mirrors the SPC beta `tts` product:
///
/// TTS = ((SRH1 * mlCAPE3)/6500) * ((mlCAPE/2000 J kg^-1)) * (6BWD/20 m s^-1)
///
/// where:
/// - the mlCAPE3 term in the first factor is capped at 150 J/kg
/// - the mlCAPE term is set to 1.0 if total mlCAPE < 2000 J/kg
/// - the mlCAPE term is capped at 1.5 if total mlCAPE > 3000 J/kg
/// - the 6BWD term is capped at 1.5 for 6BWD > 30 m/s
/// - the 6BWD term is set to 0.0 for 6BWD < 12.5 m/s
/// - the entire index is set to 0.0 if mlLCL > 1700 m AGL,
///   mlCIN < -100 J/kg, sbCIN < -200 J/kg, or TTS < 0
///
/// Naming note:
/// On the SPC mesoanalysis page, `tts` is Tornadic Tilting and Stretching,
/// not Total Totals.
pub fn compute_tts(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (srh1, mlcape, ml3cape, shear6, mllcl, mlcin, sbcin) =
        compute_tornadic_low_level_components(f, t, opts)?;

    Ok(tts_from_components(
        &srh1, &ml3cape, &mlcape, &shear6, &mllcl, &mlcin, &sbcin,
    ))
}

/// Modified Violent Tornado Parameter (dimensionless). `[ny, nx]`
///
/// VTP_mod = P1 * P2
///
/// P1 =
///   (MLCAPE / 1700.0) *
///   (ESRH / 250.0) *
///   ebwd_term *
///   mllcl_term
///
/// P2 =
///   mlcin_term *
///   ml3cape_term *
///   lr_term
///
/// where:
///
/// ebwd_term = EBWD / 30.0
/// - if EBWD <= 20.0, set ebwd_term = 0.0
/// - if EBWD >= 45.0, set ebwd_term = 1.5
///
/// mllcl_term = (1750.0 - MLLCL) / 750.0
/// - if MLLCL >= 1750.0, set mllcl_term = 0.0
/// - if MLLCL <= 750.0, set mllcl_term = 1.0
///
/// mlcin_term = (MLCIN + 200.0) / 150.0
/// - if MLCIN <= -200.0, set mlcin_term = 0.0
/// - if MLCIN >= -50.0, set mlcin_term = 1.0
///
/// ml3cape_term = ML3CAPE / 50.0
/// - if ML3CAPE >= 100.0, set ml3cape_term = 2.0
///
/// lr_term = (LR700_500 - 4.5) / 2.0
/// - if LR700_500 <= 4.5, set lr_term = 0.0
/// - if LR700_500 >= 8.5, set lr_term = 2.0
///
/// Variable meanings:
/// - MLCAPE: mixed-layer CAPE
/// - ESRH: effective storm-relative helicity
/// - EBWD: effective bulk wind difference
/// - MLLCL: mixed-layer LCL height
/// - MLCIN: mixed-layer CIN
/// - ML3CAPE: 0-3 km mixed-layer CAPE
/// - LR700_500: 700-500 mb lapse rate in degC/km, with positive values
///   indicating decreasing temperature with height
///
/// Source ambiguity note:
/// The source note mentions `MLLR > 8.5` for the lapse-rate cap, but this
/// repo has no distinct `MLLR` field. This implementation therefore applies
/// that cap to the same lapse-rate term used in the formula, `LR700_500`.
pub fn compute_vtp_mod(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let mut ml_opts = opts.clone();
    ml_opts.parcel_type = Some("ml".into());
    ml_opts.top_m = None;
    ml_opts.bottom_m = None;
    ml_opts.bottom_p = None;
    ml_opts.top_p = None;

    let mlcape = crate::diag::cape::compute_mlcape(f, t, &ml_opts)?;
    let mlcin = crate::diag::cape::compute_mlcin(f, t, &ml_opts)?;
    let mllcl = crate::diag::cape::compute_lcl(f, t, &ml_opts)?;

    let mut ml3cape_opts = ml_opts.clone();
    ml3cape_opts.top_m = Some(3000.0);
    let ml3cape = crate::diag::cape::compute_mlcape(f, t, &ml3cape_opts)?;

    let esrh = crate::diag::srh::compute_effective_srh(f, t, opts)?;
    let ebwd = compute_effective_bulk_wind_difference(f, t, opts)?;
    let lr700_500 = crate::diag::extra::compute_lapse_rate_700_500(f, t, opts)?;

    Ok(vtp_mod_from_components(
        &mlcape, &esrh, &ebwd, &mllcl, &mlcin, &ml3cape, &lr700_500,
    ))
}

/// Critical angle (degrees). `[ny, nx]`
pub fn compute_critical_angle(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let h_agl = f.height_agl(t)?;
    let pres_hpa = f.pressure_hpa(t)?;
    let psfc_hpa: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let mut result = vec![0.0f64; nxy];
    result.iter_mut().enumerate().for_each(|(ij, val)| {
        let mut u_prof = Vec::with_capacity(nz);
        let mut v_prof = Vec::with_capacity(nz);
        let mut h_prof = Vec::with_capacity(nz);
        let mut p_prof = Vec::with_capacity(nz + 1);

        p_prof.push(psfc_hpa[ij]);

        for k in 0..nz {
            let idx = k * nxy + ij;
            u_prof.push(u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij]);
            v_prof.push(u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij]);
            h_prof.push(h_agl[idx]);
            p_prof.push(pres_hpa[idx]);
        }

        *val = critical_angle_from_profile(
            &u_prof,
            &v_prof,
            &h_prof,
            &p_prof,
            u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij],
            u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij],
            opts.storm_motion.as_ref().map(|sm| sm.at(ij)),
            resolved_storm_motion_method(opts),
        );
    });

    Ok(result)
}

/// Significant Hail Parameter (dimensionless). `[ny, nx]`
/// Significant Hail Parameter (dimensionless). `[ny, nx]`
///
/// Full SHIP formula (per SPC):
///   SHIP = (MUCAPE * MR_500 * LR_700_500 * (-T500) * SHEAR_0_6km) / 42000000
///
/// Where:
///   MUCAPE = most-unstable CAPE (J/kg)
///   MR_500 = mixing ratio at 500 hPa (g/kg)
///   LR_700_500 = 700-500 hPa lapse rate (degC/km)
///   T500 = temperature at 500 hPa (degC, typically negative)
///   SHEAR_0_6km = 0-6 km bulk wind shear magnitude (m/s)
pub fn compute_ship(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres = f.full_pressure(t)?;
    let pres_hpa: Vec<f64> = pres.iter().map(|p| p / 100.0).collect();
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let (mucape, _, _, _) =
        crate::diag::cape::compute_cape_fields(f, t, "mu", None, opts.lake_interp)?;

    // 0-6 km shear
    let shear6 = crate::met::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

    // 700-500 hPa lapse rate
    let lr_opts = {
        let mut o = opts.clone();
        o.bottom_p = Some(700.0);
        o.top_p = Some(500.0);
        o
    };
    let lr_700_500 = crate::diag::extra::compute_lapse_rate(f, t, &lr_opts)?;

    // T500 and MR_500: interpolate per column
    let mut t500 = vec![0.0f64; nxy];
    let mut mr500 = vec![0.0f64; nxy];
    t500.iter_mut()
        .zip(mr500.iter_mut())
        .enumerate()
        .for_each(|(ij, (t500_val, mr500_val))| {
            for k in 0..nz - 1 {
                let idx = k * nxy + ij;
                let idx1 = (k + 1) * nxy + ij;
                if pres_hpa[idx] >= 500.0 && pres_hpa[idx1] < 500.0 {
                    let frac = (500.0 - pres_hpa[idx1]) / (pres_hpa[idx] - pres_hpa[idx1]);
                    *t500_val = tc[idx1] + frac * (tc[idx] - tc[idx1]);
                    // Mixing ratio at 500 hPa in g/kg
                    let q_interp = qv[idx1] + frac * (qv[idx] - qv[idx1]);
                    *mr500_val = q_interp.max(0.0) * 1000.0; // kg/kg -> g/kg
                    break;
                }
            }
        });

    // SHIP = (MUCAPE * MR_500 * LR * (-T500) * SHEAR) / 42000000
    Ok(mucape
        .iter()
        .zip(mr500.iter())
        .zip(lr_700_500.iter())
        .zip(t500.iter())
        .zip(shear6.iter())
        .map(|((((cape, mr), lr), t5), shr)| {
            if *cape <= 0.0 {
                return 0.0;
            }
            let result = (cape * mr * lr * (-t5).max(0.0) * shr) / 42_000_000.0;
            result.max(0.0)
        })
        .collect())
}

/// Bulk Richardson Number (dimensionless). `[ny, nx]`
///
/// Uses BRN shear rather than plain 0-6 km bulk shear:
/// the denominator is based on the vector difference between the 0-500 m
/// mean wind and the 0-6 km mean wind.
pub fn compute_bri(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres_hpa = f.pressure_hpa(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let u10 = f.u10(t)?;
    let v10 = f.v10(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let (sbcape, _, _, _) =
        crate::diag::cape::compute_cape_fields(f, t, "sb", None, opts.lake_interp)?;

    let brn_shear: Vec<f64> = (0..nxy)
        .into_par_iter()
        .map(|ij| {
            let (u_prof, v_prof, h_prof) =
                build_augmented_wind_profile(&u, &v, &h_agl, u10[ij], v10[ij], nz, nxy, ij);
            let mut p_prof = Vec::with_capacity(nz + 1);
            p_prof.push(psfc[ij] / 100.0);
            for k in 0..nz {
                p_prof.push(pres_hpa[k * nxy + ij]);
            }

            let (mean_low_u, mean_low_v) =
                pressure_weighted_layer_mean(&u_prof, &v_prof, &h_prof, &p_prof, 0.0, 500.0);
            let (mean_deep_u, mean_deep_v) =
                pressure_weighted_layer_mean(&u_prof, &v_prof, &h_prof, &p_prof, 0.0, 6000.0);
            let du = mean_deep_u - mean_low_u;
            let dv = mean_deep_v - mean_low_v;
            (du * du + dv * dv).sqrt()
        })
        .collect();

    Ok(sbcape
        .iter()
        .zip(brn_shear.iter())
        .map(|(cape, brn_shear)| {
            let denom = 0.5 * brn_shear * brn_shear;
            if denom > 0.1 {
                cape / denom
            } else {
                0.0
            }
        })
        .collect())
}

// ── Helpers ──

/// Linear interpolation of wind at a target height.
fn interp_wind_at_height(
    u_prof: &[f64],
    v_prof: &[f64],
    h_prof: &[f64],
    target_h: f64,
) -> (f64, f64) {
    for k in 0..h_prof.len() - 1 {
        if h_prof[k] <= target_h && h_prof[k + 1] > target_h {
            let frac = (target_h - h_prof[k]) / (h_prof[k + 1] - h_prof[k]);
            let u = u_prof[k] + frac * (u_prof[k + 1] - u_prof[k]);
            let v = v_prof[k] + frac * (v_prof[k + 1] - v_prof[k]);
            return (u, v);
        }
    }
    // Fallback: nearest level
    if target_h <= h_prof[0] {
        (u_prof[0], v_prof[0])
    } else {
        let last = h_prof.len() - 1;
        (u_prof[last], v_prof[last])
    }
}

fn critical_angle_from_profile(
    u_prof: &[f64],
    v_prof: &[f64],
    h_prof: &[f64],
    p_prof: &[f64],
    u_sfc: f64,
    v_sfc: f64,
    storm_motion: Option<(f64, f64)>,
    storm_motion_method: StormMotionMethod,
) -> f64 {
    let mut u_aug = Vec::with_capacity(u_prof.len() + 1);
    let mut v_aug = Vec::with_capacity(v_prof.len() + 1);
    let mut h_aug = Vec::with_capacity(h_prof.len() + 1);
    let mut p_aug = Vec::with_capacity(p_prof.len());

    u_aug.push(u_sfc);
    v_aug.push(v_sfc);
    h_aug.push(SURFACE_LAYER_HEIGHT_M);
    p_aug.extend_from_slice(p_prof);

    u_aug.extend_from_slice(u_prof);
    v_aug.extend_from_slice(v_prof);
    h_aug.extend_from_slice(h_prof);

    let (sm_u, sm_v) = storm_motion.unwrap_or_else(|| {
        let ((ru, rv), _, _) = match storm_motion_method {
            StormMotionMethod::PressureWeighted => {
                crate::met::composite::pressure_weighted_bunkers_storm_motion(
                    &h_aug, &u_aug, &v_aug, &p_aug,
                )
            }
            StormMotionMethod::NonPressureWeighted => {
                crate::met::wind::bunkers_storm_motion(&u_aug, &v_aug, &h_aug)
            }
        };
        (ru, rv)
    });
    let (u_500, v_500) = interp_wind_at_height(&u_aug, &v_aug, &h_aug, 500.0);

    crate::met::wind::critical_angle(sm_u, sm_v, u_sfc, v_sfc, u_500, v_500)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn critical_angle_uses_10m_surface_wind() {
        let u_prof = [18.0, 24.0, 32.0, 42.0];
        let v_prof = [8.0, 14.0, 20.0, 28.0];
        let h_prof = [100.0, 500.0, 1500.0, 6000.0];
        let p_prof = [1000.0, 960.0, 850.0, 500.0];
        let u10 = 5.0;
        let v10 = 2.0;

        let actual = critical_angle_from_profile(
            &u_prof,
            &v_prof,
            &h_prof,
            &p_prof,
            u10,
            v10,
            None,
            StormMotionMethod::NonPressureWeighted,
        );

        let mut u_aug = vec![u10];
        let mut v_aug = vec![v10];
        let mut h_aug = vec![SURFACE_LAYER_HEIGHT_M];
        u_aug.extend_from_slice(&u_prof);
        v_aug.extend_from_slice(&v_prof);
        h_aug.extend_from_slice(&h_prof);

        let ((sm_u, sm_v), _, _) = crate::met::wind::bunkers_storm_motion(&u_aug, &v_aug, &h_aug);
        let (u_500, v_500) = interp_wind_at_height(&u_aug, &v_aug, &h_aug, 500.0);
        let expected = crate::met::wind::critical_angle(sm_u, sm_v, u10, v10, u_500, v_500);
        let first_level =
            crate::met::wind::critical_angle(sm_u, sm_v, u_prof[0], v_prof[0], u_500, v_500);

        assert!((actual - expected).abs() < 1.0e-9);
        assert!(
            (actual - first_level).abs() > 1.0e-3,
            "10 m and first-model-level critical angles should differ in this profile"
        );
    }

    #[test]
    fn effective_stp_uses_ebwd_and_cin_limits() {
        let stp = stp_eff_from_components(
            &[1500.0, 1500.0, 1500.0, 1500.0],
            &[1000.0, 1000.0, 1000.0, 1000.0],
            &[-50.0, -50.0, -250.0, -50.0],
            &[150.0, 150.0, 150.0, 150.0],
            &[10.0, 20.0, 20.0, 40.0],
        );

        assert_close(stp[0], 0.0);
        assert_close(stp[1], 1.0);
        assert_close(stp[2], 0.0);
        assert_close(stp[3], 1.5);
    }

    #[test]
    fn fixed_stp_uses_operational_shear_gates() {
        let stp = stp_fixed_from_components(
            &[1500.0, 1500.0, 1500.0],
            &[1000.0, 1000.0, 1000.0],
            &[150.0, 150.0, 150.0],
            &[12.0, 20.0, 40.0],
        );

        assert_close(stp[0], 0.0);
        assert_close(stp[1], 1.0);
        assert_close(stp[2], 1.5);
    }

    #[test]
    fn ehi_helper_matches_operational_formula() {
        let ehi = ehi_from_components(&[1600.0, 800.0], &[100.0, 50.0]);
        assert_close(ehi[0], 1.0);
        assert_close(ehi[1], 0.25);
    }

    #[test]
    fn scp_helper_uses_operational_thresholds() {
        let scp = scp_from_components(
            &[1000.0, 1000.0, 1000.0],
            &[50.0, 50.0, 50.0],
            &[5.0, 15.0, 25.0],
        );
        assert_close(scp[0], 0.0);
        assert_close(scp[1], 0.75);
        assert_close(scp[2], 1.0);
    }

    #[test]
    fn tehi_matches_spc_beta_formula() {
        let tehi = tehi_from_components(
            &[200.0],
            &[1000.0],
            &[100.0],
            &[20.0],
            &[1000.0],
            &[-50.0],
            &[-50.0],
        );

        assert_close(tehi[0], 0.625);
    }

    #[test]
    fn tehi_uses_mlcape3_floor_when_total_mlcape_is_large() {
        let tehi = tehi_from_components(
            &[160.0],
            &[1600.0],
            &[50.0],
            &[20.0],
            &[1000.0],
            &[-50.0],
            &[-50.0],
        );

        assert_close(tehi[0], 1.6);
    }

    #[test]
    fn tehi_applies_limiters_and_shear_gate() {
        let tehi = tehi_from_components(
            &[200.0, 200.0, 200.0, 200.0],
            &[1000.0, 1000.0, 1000.0, 1000.0],
            &[100.0, 100.0, 100.0, 100.0],
            &[10.0, 20.0, 20.0, 20.0],
            &[1000.0, 1800.0, 1000.0, 1000.0],
            &[-50.0, -50.0, -150.0, -50.0],
            &[-50.0, -50.0, -50.0, -250.0],
        );

        assert_close(tehi[0], 0.0);
        assert_close(tehi[1], 0.0);
        assert_close(tehi[2], 0.0);
        assert_close(tehi[3], 0.0);
    }

    #[test]
    fn tts_matches_spc_beta_formula() {
        let tts = tts_from_components(
            &[100.0],
            &[100.0],
            &[2500.0],
            &[20.0],
            &[1000.0],
            &[-50.0],
            &[-50.0],
        );

        assert_close(tts[0], 1.9230769230769231);
    }

    #[test]
    fn tts_uses_mlcape_floor_and_caps_mlcape3_and_shear() {
        let tts = tts_from_components(
            &[100.0],
            &[200.0],
            &[1500.0],
            &[35.0],
            &[1000.0],
            &[-50.0],
            &[-50.0],
        );

        assert_close(tts[0], 3.4615384615384617);
    }

    #[test]
    fn tts_applies_limiters_and_negative_floor() {
        let tts = tts_from_components(
            &[-100.0, 100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0, 100.0],
            &[2500.0, 2500.0, 2500.0, 2500.0],
            &[20.0, 20.0, 20.0, 20.0],
            &[1000.0, 1800.0, 1000.0, 1000.0],
            &[-50.0, -50.0, -150.0, -50.0],
            &[-50.0, -50.0, -50.0, -250.0],
        );

        assert_close(tts[0], 0.0);
        assert_close(tts[1], 0.0);
        assert_close(tts[2], 0.0);
        assert_close(tts[3], 0.0);
    }

    #[test]
    fn vtp_mod_matches_in_range_formula() {
        let vtp = vtp_mod_from_components(
            &[1700.0],
            &[250.0],
            &[30.0],
            &[1000.0],
            &[-100.0],
            &[50.0],
            &[6.5],
        );

        assert_close(vtp[0], 2.0 / 3.0);
    }

    #[test]
    fn vtp_mod_applies_ebwd_cutoff_and_cap() {
        let vtp = vtp_mod_from_components(
            &[1700.0, 1700.0, 1700.0],
            &[250.0, 250.0, 250.0],
            &[19.0, 20.0, 45.0],
            &[1000.0, 1000.0, 1000.0],
            &[-50.0, -50.0, -50.0],
            &[50.0, 50.0, 50.0],
            &[6.5, 6.5, 6.5],
        );

        assert_close(vtp[0], 0.0);
        assert_close(vtp[1], 0.0);
        assert_close(vtp[2], 1.5);
    }

    #[test]
    fn vtp_mod_applies_mllcl_cap_and_cutoff() {
        let vtp = vtp_mod_from_components(
            &[1700.0, 1700.0, 1700.0],
            &[250.0, 250.0, 250.0],
            &[30.0, 30.0, 30.0],
            &[750.0, 1000.0, 1750.0],
            &[-50.0, -50.0, -50.0],
            &[50.0, 50.0, 50.0],
            &[6.5, 6.5, 6.5],
        );

        assert_close(vtp[0], 1.0);
        assert_close(vtp[1], 1.0);
        assert_close(vtp[2], 0.0);
    }

    #[test]
    fn vtp_mod_applies_mlcin_cutoff_and_cap() {
        let vtp = vtp_mod_from_components(
            &[1700.0, 1700.0, 1700.0],
            &[250.0, 250.0, 250.0],
            &[30.0, 30.0, 30.0],
            &[1000.0, 1000.0, 1000.0],
            &[-200.0, -100.0, -50.0],
            &[50.0, 50.0, 50.0],
            &[6.5, 6.5, 6.5],
        );

        assert_close(vtp[0], 0.0);
        assert_close(vtp[1], 2.0 / 3.0);
        assert_close(vtp[2], 1.0);
    }

    #[test]
    fn vtp_mod_applies_ml3cape_cap() {
        let vtp = vtp_mod_from_components(
            &[1700.0, 1700.0, 1700.0],
            &[250.0, 250.0, 250.0],
            &[30.0, 30.0, 30.0],
            &[1000.0, 1000.0, 1000.0],
            &[-50.0, -50.0, -50.0],
            &[25.0, 100.0, 150.0],
            &[6.5, 6.5, 6.5],
        );

        assert_close(vtp[0], 0.5);
        assert_close(vtp[1], 2.0);
        assert_close(vtp[2], 2.0);
    }

    #[test]
    fn vtp_mod_applies_lr700_500_floor_and_cap() {
        let vtp = vtp_mod_from_components(
            &[1700.0, 1700.0, 1700.0],
            &[250.0, 250.0, 250.0],
            &[30.0, 30.0, 30.0],
            &[1000.0, 1000.0, 1000.0],
            &[-50.0, -50.0, -50.0],
            &[50.0, 50.0, 50.0],
            &[4.5, 6.5, 8.5],
        );

        assert_close(vtp[0], 0.0);
        assert_close(vtp[1], 1.0);
        assert_close(vtp[2], 2.0);
    }

    #[test]
    fn experimental_defaults_match_classic_parcel_choices() {
        let empty = ComputeOpts::default();
        assert_eq!(
            opts_with_default_parcel_type(&empty, "sb")
                .parcel_type
                .as_deref(),
            Some("sb")
        );
        assert_eq!(
            opts_with_default_parcel_type(&empty, "mu")
                .parcel_type
                .as_deref(),
            Some("mu")
        );

        let explicit = ComputeOpts {
            parcel_type: Some("ml".into()),
            ..ComputeOpts::default()
        };
        assert_eq!(
            opts_with_default_parcel_type(&explicit, "sb")
                .parcel_type
                .as_deref(),
            Some("ml")
        );
    }
}
