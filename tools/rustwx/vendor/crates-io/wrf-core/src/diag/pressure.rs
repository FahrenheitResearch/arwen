//! Pressure, height, and geopotential diagnostic variables:
//! pressure, height, height_agl, zstag, geopt, geopt_stag, terrain, slp, omega

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

const G: f64 = 9.80665;
const RD: f64 = 287.058;

// WRF-python (NCAR) uses these rounded constants in its Fortran SLP routine.
// We must match them exactly for SLP compatibility.
const G_SLP: f64 = 9.81;
const RD_SLP: f64 = 287.0;
const USSALR: f64 = 0.0065; // US Standard Atmosphere lapse rate (K/m)
const PCONST: f64 = 10000.0; // Pa above surface to find reference level
const TC: f64 = 273.16 + 17.5; // ~290.66 K, temperature threshold for capping

/// Full pressure (Pa). `[nz, ny, nx]`
pub fn compute_pressure(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.full_pressure(t).map(|v| v.to_vec())
}

/// Full pressure (hPa). `[nz, ny, nx]`
pub fn compute_pressure_hpa(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.pressure_hpa(t).map(|v| v.to_vec())
}

/// Height MSL (m). `[nz, ny, nx]`
pub fn compute_height(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.height_msl(t).map(|v| v.to_vec())
}

/// Height AGL (m). `[nz, ny, nx]`
pub fn compute_height_agl(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.height_agl(t).map(|v| v.to_vec())
}

/// Height on staggered Z levels (m). `[nz_stag, ny, nx]`
pub fn compute_zstag(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let geopt_stag = f.geopotential_stag(t)?;
    Ok(geopt_stag.iter().map(|v| v / G).collect())
}

/// Full geopotential (m^2/s^2), destaggered. `[nz, ny, nx]`
pub fn compute_geopt(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.full_geopotential(t).map(|v| v.to_vec())
}

/// Geopotential on staggered Z levels. `[nz_stag, ny, nx]`
pub fn compute_geopt_stag(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.geopotential_stag(t).map(|v| v.to_vec())
}

/// Terrain height (m). `[ny, nx]`
pub fn compute_terrain(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.terrain(t).map(|v| v.to_vec())
}

/// Sea-level pressure (hPa). `[ny, nx]`
///
/// Matches wrf-python's DCOMPUTESEAPRS (Shuell 1995) exactly:
/// 1. Find a reference level ~PCONST (100 hPa) above the surface to avoid
///    diurnal heating artifacts in the lowest model level.
/// 2. Log-interpolate virtual temperature and height at that reference pressure.
/// 3. Compute t_surf and t_sea_level using the US Standard Atmosphere lapse rate.
/// 4. Apply the "ridiculous_mm5_test" temperature capping (TC = 290.66 K).
/// 5. Reduce to sea level: SLP = p_sfc * exp(2*g*z_sfc / (Rd*(t_sea_level + t_surf))).
pub fn compute_slp(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres = f.full_pressure(t)?;
    let tk = f.temperature(t)?;
    let qv = f.qvapor(t)?;
    let z = f.height_msl(t)?;

    let nxy = f.nxy();
    let nz = f.nz;

    let mut slp = vec![0.0f64; nxy];

    slp.iter_mut().enumerate().for_each(|(ij, slp_val)| {
        let p_sfc = pres[ij]; // k=0 surface pressure

        // Step 1: Find the level ~PCONST above the surface.
        // We want the lowest k where p_sfc - p(k) >= PCONST.
        let mut klo = 0usize;
        let khi;
        let mut found = false;
        for k in 0..nz {
            if (p_sfc - pres[k * nxy + ij]) >= PCONST {
                klo = k;
                found = true;
                break;
            }
        }
        if !found {
            // All levels are within PCONST of surface; use top two levels
            klo = nz - 1;
        }
        khi = if klo > 0 { klo - 1 } else { klo + 1 };
        // klo is the level just above the PCONST threshold,
        // khi is the level just below it

        let plo = pres[klo * nxy + ij];
        let phi = pres[khi * nxy + ij];

        // Virtual temperatures at bracketing levels (wrf-python uses 0.608)
        let qlo = qv[klo * nxy + ij].max(0.0);
        let qhi = qv[khi * nxy + ij].max(0.0);
        let tlo = tk[klo * nxy + ij] * (1.0 + 0.608 * qlo);
        let thi = tk[khi * nxy + ij] * (1.0 + 0.608 * qhi);
        let zlo = z[klo * nxy + ij];
        let zhi = z[khi * nxy + ij];

        // Step 2: Log-interpolate to the exact PCONST level
        let p_at_pconst = p_sfc - PCONST;
        let (t_at_pconst, z_at_pconst) = if (plo - phi).abs() < 1.0 {
            // Levels are nearly identical pressure; just use klo values
            (tlo, zlo)
        } else {
            let frac = (p_at_pconst.ln() - phi.ln()) / (plo.ln() - phi.ln());
            let t_interp = thi + frac * (tlo - thi);
            let z_interp = zhi + frac * (zlo - zhi);
            (t_interp, z_interp)
        };

        // Step 3: Compute surface and sea-level temperatures using USSALR
        let t_surf = t_at_pconst * (p_sfc / p_at_pconst).powf(USSALR * RD_SLP / G_SLP);
        let mut t_sea_level = t_at_pconst + USSALR * z_at_pconst;

        // Step 4: Temperature capping (the "ridiculous_mm5_test")
        // This prevents unphysical extrapolation in warm/tropical regions.
        // Matches wrf-python exactly: only two branches.
        if t_surf <= TC && t_sea_level >= TC {
            // Surface cool but sea-level warm: cap at TC
            t_sea_level = TC;
        } else {
            // All other cases: quadratic correction toward TC
            t_sea_level = TC - 0.005 * (t_surf - TC) * (t_surf - TC);
        }

        // Step 5: Reduce to sea level using mean of t_sea_level and t_surf
        let z_sfc = z[ij]; // height of lowest model level
        *slp_val = 0.01 * (p_sfc * (2.0 * G_SLP * z_sfc / (RD_SLP * (t_sea_level + t_surf))).exp());
    });

    Ok(slp)
}

/// Omega: vertical velocity in pressure coordinates (Pa/s). `[nz, ny, nx]`
///
/// omega = -rho * g * w = -(p / (Rd * Tv)) * g * w
pub fn compute_omega(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let w = f.w_destag(t)?;
    let pres = f.full_pressure(t)?;
    let tk = f.temperature(t)?;
    let qv = f.qvapor(t)?;

    Ok(w.iter()
        .zip(pres.iter())
        .zip(tk.iter())
        .zip(qv.iter())
        .map(|(((w, p), t_k), q)| {
            let tv = t_k * (1.0 + 0.61 * q.max(0.0));
            let rho = p / (RD * tv);
            -rho * G * w
        })
        .collect())
}
