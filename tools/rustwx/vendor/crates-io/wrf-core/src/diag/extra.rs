//! Extra diagnostic variables:
//! lapse rates, freezing level, wet-bulb zero, theta_w, fire indices

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;
use rayon::prelude::*;

/// 700-500 hPa lapse rate (°C/km). `[ny, nx]`
///
/// Delegates to the generic compute_lapse_rate with pressure bounds.
/// Supports use_virtual and lake_interp via opts.
pub fn compute_lapse_rate_700_500(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<Vec<f64>> {
    let mut lr_opts = opts.clone();
    lr_opts.bottom_p = Some(700.0);
    lr_opts.top_p = Some(500.0);
    compute_lapse_rate(f, t, &lr_opts)
}

/// 0-3 km AGL lapse rate (°C/km). `[ny, nx]`
///
/// Delegates to the generic compute_lapse_rate with height bounds.
/// Uses T2 at surface, supports use_virtual and lake_interp via opts.
pub fn compute_lapse_rate_0_3km(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let mut lr_opts = opts.clone();
    lr_opts.bottom_m = Some(0.0);
    lr_opts.top_m = Some(3000.0);
    compute_lapse_rate(f, t, &lr_opts)
}

/// Freezing level height AGL (m). `[ny, nx]`
pub fn compute_freezing_level(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let tc = f.temperature_c(t)?;
    let h_agl = f.height_agl(t)?;

    let nxy = f.nxy();
    let nz = f.nz;

    let mut fzlev = vec![0.0f64; nxy];
    fzlev.iter_mut().enumerate().for_each(|(ij, fz_val)| {
        for k in 0..nz - 1 {
            let idx0 = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;
            // Find first crossing of 0°C going upward
            if tc[idx0] >= 0.0 && tc[idx1] < 0.0 {
                let frac = (0.0 - tc[idx0]) / (tc[idx1] - tc[idx0]);
                *fz_val = h_agl[idx0] + frac * (h_agl[idx1] - h_agl[idx0]);
                return;
            }
        }
        // If surface is already below freezing, freezing level is 0
        if tc[ij] < 0.0 {
            *fz_val = 0.0;
        }
    });

    Ok(fzlev)
}

/// Wet-bulb zero height AGL (m). `[ny, nx]`
pub fn compute_wet_bulb_0(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;

    let nxy = f.nxy();
    let nz = f.nz;

    let mut wb0 = vec![0.0f64; nxy];
    wb0.iter_mut().enumerate().for_each(|(ij, wb0_val)| {
        // Compute wet-bulb at each level, find first crossing below 0°C
        let mut prev_twb = f64::NAN;
        let mut prev_h = 0.0;

        for k in 0..nz {
            let idx = k * nxy + ij;
            let q = qv[idx].max(1e-10);
            let e = q * p_hpa[idx] / (0.622 + q);
            let ln_e = (e / 6.112).max(1e-10).ln();
            let td_c = (243.5 * ln_e) / (17.67 - ln_e);

            let twb = crate::met::thermo::wet_bulb_temperature(p_hpa[idx], tc[idx], td_c);

            if k > 0 && prev_twb >= 0.0 && twb < 0.0 {
                let frac = (0.0 - prev_twb) / (twb - prev_twb);
                *wb0_val = prev_h + frac * (h_agl[idx] - prev_h);
                return;
            }
            prev_twb = twb;
            prev_h = h_agl[idx];
        }
    });

    Ok(wb0)
}

/// Wet-bulb potential temperature (K). `[nz, ny, nx]`
pub fn compute_theta_w(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;

    Ok(p_hpa
        .iter()
        .zip(tc.iter())
        .zip(qv.iter())
        .map(|((p, t_c), q)| {
            let q = q.max(1e-10);
            let e = q * p / (0.622 + q);
            let ln_e = (e / 6.112).max(1e-10).ln();
            let td_c = (243.5 * ln_e) / (17.67 - ln_e);
            crate::met::thermo::wet_bulb_potential_temperature(*p, *t_c, td_c) + 273.15
        })
        .collect())
}

/// Fosberg Fire Weather Index (dimensionless). `[ny, nx]`
/// Uses 2-m temperature, 2-m RH, and 10-m wind speed.
pub fn compute_fosberg(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let t2 = f.t2(t)?;
    let q2 = f.q2(t)?;
    let psfc = f.psfc(t)?;
    let u10 = f.u10(t)?;
    let v10 = f.v10(t)?;

    let nxy = f.nxy();

    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let t_k = t2[ij];
            let t_c = t_k - 273.15;
            let t_f = t_c * 9.0 / 5.0 + 32.0;
            let p_hpa = psfc[ij] / 100.0;
            let q = q2[ij].max(0.0);
            let e = q * p_hpa / (0.622 + q);
            let es = 6.112 * (17.67 * t_c / (t_c + 243.5)).exp();
            let rh = (e / es * 100.0).clamp(0.0, 100.0);
            let wspd_mph = (u10[ij].powi(2) + v10[ij].powi(2)).sqrt() / 0.44704;

            crate::met::composite::fosberg_fire_weather_index(t_f, rh, wspd_mph)
        })
        .collect())
}

/// Haines Index (1-3, low/moderate/high). `[ny, nx]`
pub fn compute_haines(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;

    let nxy = f.nxy();
    let nz = f.nz;

    let mut haines = vec![0.0f64; nxy];
    haines.iter_mut().enumerate().for_each(|(ij, h_val)| {
        // Find T and Td at 950, 850 hPa
        let mut t950 = 0.0f64;
        let mut t850 = 0.0f64;
        let mut td850 = 0.0f64;

        for k in 0..nz - 1 {
            let idx = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;

            for &target_p in &[950.0, 850.0] {
                if p_hpa[idx] >= target_p && p_hpa[idx1] < target_p {
                    let frac = (target_p - p_hpa[idx1]) / (p_hpa[idx] - p_hpa[idx1]);
                    let t_interp = tc[idx1] + frac * (tc[idx] - tc[idx1]);
                    let q_interp = qv[idx1] + frac * (qv[idx] - qv[idx1]);
                    let q = q_interp.max(1e-10);
                    let e = q * target_p / (0.622 + q);
                    let ln_e = (e / 6.112).max(1e-10).ln();
                    let td = (243.5 * ln_e) / (17.67 - ln_e);

                    if target_p == 950.0 {
                        t950 = t_interp;
                    } else {
                        t850 = t_interp;
                        td850 = td;
                    }
                }
            }
        }

        *h_val = crate::met::composite::haines_index(t950, t850, td850) as f64;
    });

    Ok(haines)
}

/// Hot-Dry-Windy Index (dimensionless). `[ny, nx]`
pub fn compute_hdw(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let t2 = f.t2(t)?;
    let q2 = f.q2(t)?;
    let psfc = f.psfc(t)?;
    let u10 = f.u10(t)?;
    let v10 = f.v10(t)?;

    let nxy = f.nxy();

    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let t_c = t2[ij] - 273.15;
            let p_hpa = psfc[ij] / 100.0;
            let q = q2[ij].max(0.0);
            let e = q * p_hpa / (0.622 + q);
            let es = 6.112 * (17.67 * t_c / (t_c + 243.5)).exp();
            let rh = (e / es * 100.0).clamp(0.0, 100.0);
            let wspd_ms = (u10[ij].powi(2) + v10[ij].powi(2)).sqrt();

            // VPD = es - e (in hPa)
            let vpd = (es - e).max(0.0);

            crate::met::composite::hot_dry_windy(t_c, rh, wspd_ms, vpd)
        })
        .collect())
}

/// Generic configurable lapse rate (°C/km). `[ny, nx]`
///
/// Supports two modes:
/// - **Height mode** (default): `bottom_m` / `top_m` in meters AGL. Defaults to 0-3 km.
/// - **Pressure mode**: `bottom_p` / `top_p` in hPa (e.g. 700, 500). When either
///   pressure bound is set, height bounds are ignored.
///
/// If `opts.use_virtual` is `Some(true)`, virtual temperature
/// Tv = T * (1 + 0.61 * qv) is used instead of absolute temperature.
pub fn compute_lapse_rate(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let use_virtual = opts.use_virtual == Some(true);
    let use_pressure = opts.bottom_p.is_some() || opts.top_p.is_some();

    let tc = f.temperature_c(t)?;
    let h_agl = f.height_agl(t)?;
    let qv = if use_virtual {
        Some(f.qvapor(t)?)
    } else {
        None
    };
    let pres_hpa = if use_pressure {
        Some(f.pressure_hpa(t)?)
    } else {
        None
    };

    // For surface (bottom_m=0 or not set), use 2m data with lake_interp support
    let bottom_is_surface = !use_pressure && opts.bottom_m.unwrap_or(0.0) < 10.0;
    let t2_c = if bottom_is_surface {
        let t2_k = match opts.lake_interp {
            Some(a) if a > 0.0 => f.t2_lake_corrected(t, a)?,
            _ => f.t2(t)?.to_vec(),
        };
        Some(t2_k.iter().map(|v| v - 273.15).collect::<Vec<f64>>())
    } else {
        None
    };
    let q2 = if bottom_is_surface && use_virtual {
        Some(match opts.lake_interp {
            Some(a) if a > 0.0 => f.q2_lake_corrected(t, a)?,
            _ => f.q2(t)?.to_vec(),
        })
    } else {
        None
    };

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let mut lr = vec![0.0f64; nxy];
    lr.iter_mut().enumerate().for_each(|(ij, lr_val)| {
        // Helper: temperature (or virtual temperature) at a 3-D index
        let temp_at = |idx: usize| -> f64 {
            let t_c = tc[idx];
            if use_virtual {
                let q = qv.as_ref().unwrap()[idx].max(0.0);
                let t_k = t_c + 273.15;
                let tv_k = t_k * (1.0 + 0.61 * q);
                tv_k - 273.15
            } else {
                t_c
            }
        };

        if use_pressure {
            // Pressure mode: interpolate T and H at given pressure levels
            let p = pres_hpa.as_ref().unwrap();
            let p_bot = opts.bottom_p.unwrap_or(700.0); // hPa (higher pressure = lower)
            let p_top = opts.top_p.unwrap_or(500.0); // hPa (lower pressure = higher)

            let interp_at_p = |target_p: f64| -> (f64, f64) {
                // Pressure decreases with height; scan upward
                for k in 0..nz - 1 {
                    let idx0 = k * nxy + ij;
                    let idx1 = (k + 1) * nxy + ij;
                    let p0 = p[idx0];
                    let p1 = p[idx1];
                    if p0 >= target_p && p1 < target_p {
                        let frac = (target_p - p1) / (p0 - p1);
                        let t_interp = temp_at(idx1) + frac * (temp_at(idx0) - temp_at(idx1));
                        let h_interp = h_agl[idx1] + frac * (h_agl[idx0] - h_agl[idx1]);
                        return (t_interp, h_interp);
                    }
                }
                // Fallback: nearest level
                (temp_at(ij), h_agl[ij])
            };

            let (t_bot, h_bot) = interp_at_p(p_bot);
            let (t_top, h_top) = interp_at_p(p_top);
            let depth_km = (h_top - h_bot).abs() / 1000.0;
            if depth_km > 0.01 {
                *lr_val = -(t_top - t_bot) / depth_km;
            }
        } else {
            // Height mode
            let bottom_m = opts.bottom_m.unwrap_or(0.0);
            let top_m = opts.top_m.unwrap_or(3000.0);
            let depth_km = (top_m - bottom_m) / 1000.0;

            let interp_at_h = |target_h: f64| -> f64 {
                // Use 2m data for surface (target_h < 10m)
                if target_h < 10.0 {
                    if let Some(ref t2) = t2_c {
                        let t_sfc = t2[ij];
                        if use_virtual {
                            if let Some(ref q) = q2 {
                                let t_k = t_sfc + 273.15;
                                let tv_k = t_k * (1.0 + 0.61 * q[ij].max(0.0));
                                return tv_k - 273.15;
                            }
                        }
                        return t_sfc;
                    }
                }
                if h_agl[ij] >= target_h {
                    return temp_at(ij);
                }
                for k in 1..nz {
                    let idx = k * nxy + ij;
                    if h_agl[idx] >= target_h {
                        let idx_prev = (k - 1) * nxy + ij;
                        let h_prev = h_agl[idx_prev];
                        let frac = (target_h - h_prev) / (h_agl[idx] - h_prev);
                        return temp_at(idx_prev) + frac * (temp_at(idx) - temp_at(idx_prev));
                    }
                }
                temp_at((nz - 1) * nxy + ij)
            };

            let t_bot = interp_at_h(bottom_m);
            let t_top = interp_at_h(top_m);
            if depth_km > 0.01 {
                *lr_val = -(t_top - t_bot) / depth_km;
            }
        }
    });

    Ok(lr)
}
