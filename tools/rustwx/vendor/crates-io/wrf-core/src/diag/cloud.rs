//! Cloud diagnostic variables: ctt, cloudfrac

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Cloud-top temperature (°C). `[ny, nx]`
///
/// Finds the highest level with cloud water + ice > threshold
/// and returns the temperature there.
pub fn compute_ctt(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let tc = f.temperature_c(t)?;
    let qc = f
        .read_var("QCLOUD", t)
        .unwrap_or_else(|_| vec![0.0; f.nxyz()]);
    let qi = f
        .read_var("QICE", t)
        .unwrap_or_else(|_| vec![0.0; f.nxyz()]);

    let nxy = f.nxy();
    let nz = f.nz;
    let cloud_thresh = 1e-6; // kg/kg

    let mut ctt = vec![-80.0f64; nxy]; // default cold value if no cloud

    ctt.iter_mut().enumerate().for_each(|(ij, ctt_val)| {
        // Scan from top down, find highest cloud level
        for k in (0..nz).rev() {
            let idx = k * nxy + ij;
            let total_cloud = qc[idx].max(0.0) + qi[idx].max(0.0);
            if total_cloud > cloud_thresh {
                *ctt_val = tc[idx];
                break;
            }
        }
    });

    Ok(ctt)
}

/// Cloud fraction (%). Returns `[low, mid, high]` interleaved (3 * nxy).
///
/// - Low: surface to 800 hPa
/// - Mid: 800 to 450 hPa
/// - High: 450 hPa to model top
pub fn compute_cloudfrac(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres_hpa = f.pressure_hpa(t)?;
    let qc = f
        .read_var("QCLOUD", t)
        .unwrap_or_else(|_| vec![0.0; f.nxyz()]);
    let qi = f
        .read_var("QICE", t)
        .unwrap_or_else(|_| vec![0.0; f.nxyz()]);

    let nxy = f.nxy();
    let nz = f.nz;
    let cloud_thresh = 1e-6;

    let mut low_frac = vec![0.0f64; nxy];
    let mut mid_frac = vec![0.0f64; nxy];
    let mut high_frac = vec![0.0f64; nxy];

    // For each column, compute max overlap cloud fraction in each layer
    let process = |frac: &mut [f64], p_top: f64, p_bot: f64| {
        frac.iter_mut().enumerate().for_each(|(ij, val)| {
            let mut max_cloud = 0.0f64;
            for k in 0..nz {
                let idx = k * nxy + ij;
                let p = pres_hpa[idx];
                if p <= p_bot && p >= p_top {
                    let cloud = qc[idx].max(0.0) + qi[idx].max(0.0);
                    if cloud > max_cloud {
                        max_cloud = cloud;
                    }
                }
            }
            *val = if max_cloud > cloud_thresh { 100.0 } else { 0.0 };
        });
    };

    process(&mut low_frac, 800.0, 1100.0);
    process(&mut mid_frac, 450.0, 800.0);
    process(&mut high_frac, 0.0, 450.0);

    let mut out = low_frac;
    out.extend(mid_frac);
    out.extend(high_frac);
    Ok(out)
}
