//! Additional consumer rows use the ML exporter's operators and pressure plan.
//! Array transforms run across rayon workers; no model-specific route is needed.
use crate::error::{fail, refuse, Result};
use crate::{
    ops::{self, Op},
    vertical::{Columns, Lid, Plan, Rule},
};
use rayon::prelude::*;
use std::path::Path;
use wrf_core::WrfFile;

pub struct Output {
    /// [specific humidity, omega, geometric w][pressure level][cell].
    pub pressure: Vec<Vec<f32>>,
    pub wind: Vec<(u32, Vec<f32>, Vec<f32>)>,
}

pub fn compute(path: &Path, levels: &[u32], heights: &[u32], pressure: bool) -> Result<Output> {
    let file = WrfFile::open(path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    compute_file(&file, 0, levels, heights, pressure)
}

/// Common pressure/height operators for GRIB messages and weather maps.
pub fn compute_file(
    file: &WrfFile,
    timeidx: usize,
    levels: &[u32],
    heights: &[u32],
    pressure: bool,
) -> Result<Output> {
    let mut required = vec!["U", "V", "PH", "PHB", "HGT", "SINALPHA", "COSALPHA"];
    if pressure {
        required.extend(["P", "PB", "T", "QVAPOR", "W", "PSFC"]);
    }
    for name in required {
        if !file.has_var(name) {
            return Err(refuse(format!("{}: missing {name}; cannot compute SPFH/VVEL/DZDT on pressure levels or earth-relative 100 m winds from this history selection", file.path.display())));
        }
    }
    let n = file.nxy();
    let z = file
        .height_agl(timeidx)
        .map_err(|e| fail(format!("PH/PHB/HGT heights: {e}")))?;
    let u = file
        .u_destag(timeidx)
        .map_err(|e| fail(format!("U mass winds: {e}")))?;
    let v = file
        .v_destag(timeidx)
        .map_err(|e| fail(format!("V mass winds: {e}")))?;
    let sin = file
        .sinalpha(timeidx)
        .map_err(|e| fail(format!("SINALPHA rotation: {e}")))?;
    let cos = file
        .cosalpha(timeidx)
        .map_err(|e| fail(format!("COSALPHA rotation: {e}")))?;
    let nz = z.len() / n;
    let wind = heights
        .iter()
        .map(|&height| {
            let pairs: Vec<(f32, f32)> = (0..n)
                .into_par_iter()
                .map(|c| {
                    let target = f64::from(height);
                    for k in 0..nz.saturating_sub(1) {
                        let (a, b) = (k * n + c, (k + 1) * n + c);
                        if z[a] <= target && target <= z[b] && z[b] > z[a] {
                            let f = (target - z[a]) / (z[b] - z[a]);
                            let (ug, vg) = (u[a] + f * (u[b] - u[a]), v[a] + f * (v[b] - v[a]));
                            return (
                                (ug * cos[c] - vg * sin[c]) as f32,
                                (ug * sin[c] + vg * cos[c]) as f32,
                            );
                        }
                    }
                    (f32::NAN, f32::NAN)
                })
                .collect();
            let (u, v) = pairs.into_iter().unzip();
            (height, u, v)
        })
        .collect();
    let mut out = Output {
        pressure: Vec::new(),
        wind,
    };
    if pressure {
        let p = file
            .full_pressure(timeidx)
            .map_err(|e| fail(format!("P+PB: {e}")))?;
        let tk = file
            .temperature(timeidx)
            .map_err(|e| fail(format!("T: {e}")))?;
        let psfc = ops::read(file, timeidx, "PSFC")?;
        let phi = file
            .geopotential_stag(timeidx)
            .map_err(|e| fail(format!("PH+PHB: {e}")))?;
        let phi_top: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|c| 0.5 * (phi[(nz - 1) * n + c] + phi[nz * n + c]))
            .collect();
        let cols = Columns {
            nz,
            cells: n,
            p: &p,
            psfc: &psfc,
            phi_sfc: &phi[..n],
            t_bot: &tk[..n],
            t_top: &tk[(nz - 1) * n..],
            phi_top: &phi_top,
            phi_lid: &phi[nz * n..],
            lid: Lid::TopMassLevel,
        };
        let plan = Plan::new(levels, &cols);
        let w = file
            .w_destag(timeidx)
            .map_err(|e| fail(format!("W mass velocity: {e}")))?;
        // The same operators as ML export, including moist-air normalization
        // and wrf-core's virtual-temperature omega. Never duplicate that formula.
        for values in [
            ops::specific_humidity(file, timeidx)?,
            ops::level_field(file, timeidx, &Op::Omega, 1.0)?,
            w.to_vec(),
        ] {
            let mut field = plan.apply(&values, Rule::Lowest, &cols);
            // GRIB post's existing pressure rows are missing above the top.
            field
                .par_chunks_mut(n)
                .zip(levels.par_iter())
                .for_each(|(plane, &hpa)| {
                    for (c, value) in plane.iter_mut().enumerate() {
                        if f64::from(hpa) * 100.0 < p[(nz - 1) * n + c] {
                            *value = f32::NAN;
                        }
                    }
                });
            out.pressure.push(field);
        }
    }
    Ok(out)
}
