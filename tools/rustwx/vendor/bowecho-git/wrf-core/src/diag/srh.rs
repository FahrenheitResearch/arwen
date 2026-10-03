//! Storm-relative helicity and bulk shear diagnostics.
//!
//! Uses proper Bunkers storm motion (crate::met), NOT wrf-python's
//! broken 0.75*(3-10km mean wind) rotated 30 degrees.

use crate::compute::{ComputeOpts, StormMotionMethod};
use crate::diag::cape::{build_surface_augmented_thermo_column, effective_inflow_layer_grid};
use crate::error::WrfResult;
use crate::file::WrfFile;
use rayon::prelude::*;

const SURFACE_LAYER_HEIGHT_M: f64 = 0.0;
const BUNKERS_STACK_FIELDS: usize = 6;

fn resolved_storm_motion_method(opts: &ComputeOpts) -> StormMotionMethod {
    opts.storm_motion_method
        .unwrap_or(StormMotionMethod::PressureWeighted)
}

fn bunkers_cache_key(method: StormMotionMethod) -> &'static str {
    match method {
        StormMotionMethod::PressureWeighted => "bunkers_stack_pw",
        StormMotionMethod::NonPressureWeighted => "bunkers_stack_npw",
    }
}

fn pack_bunkers_stack(
    rm_u: &[f64],
    rm_v: &[f64],
    lm_u: &[f64],
    lm_v: &[f64],
    mn_u: &[f64],
    mn_v: &[f64],
) -> Vec<f64> {
    let nxy = rm_u.len();
    let mut stacked = Vec::with_capacity(BUNKERS_STACK_FIELDS * nxy);
    stacked.extend_from_slice(rm_u);
    stacked.extend_from_slice(rm_v);
    stacked.extend_from_slice(lm_u);
    stacked.extend_from_slice(lm_v);
    stacked.extend_from_slice(mn_u);
    stacked.extend_from_slice(mn_v);
    stacked
}

fn unpack_bunkers_stack(
    stacked: &[f64],
    nxy: usize,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if stacked.len() != BUNKERS_STACK_FIELDS * nxy {
        return None;
    }

    Some((
        stacked[0..nxy].to_vec(),
        stacked[nxy..2 * nxy].to_vec(),
        stacked[2 * nxy..3 * nxy].to_vec(),
        stacked[3 * nxy..4 * nxy].to_vec(),
        stacked[4 * nxy..5 * nxy].to_vec(),
        stacked[5 * nxy..6 * nxy].to_vec(),
    ))
}

/// Canonical SRH entry point for all grid-based SRH computations.
///
/// Applies earth-rotation (SINALPHA/COSALPHA) to both 3-D and 10-m winds,
/// prepends U10/V10 as the surface layer, and then computes SRH via Bunkers RM
/// (or a caller-supplied storm motion).  All SRH consumers -- including
/// STP, SCP, EHI, and effective SRH -- should funnel through this function
/// so that every path sees the same wind preparation.
pub fn compute_srh_field(
    f: &WrfFile,
    t: usize,
    depth_m: f64,
    storm_motion: Option<&crate::compute::StormMotion>,
    storm_motion_method: Option<StormMotionMethod>,
) -> WrfResult<Vec<f64>> {
    // Use earth-rotated winds for SRH (matches SHARPpy/MetPy convention)
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let h_agl = f.height_agl(t)?;
    let pres_hpa = f.pressure_hpa(t)?;
    let psfc_hpa: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nxy = nx * ny;

    let nz = f.nz;

    if let Some(storm_motion) = storm_motion {
        // Custom storm motion: compute column-by-column
        let mut srh = vec![0.0f64; nxy];
        srh.iter_mut().enumerate().for_each(|(ij, srh_val)| {
            // Prepend 10m wind as the surface layer.
            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            let mut h_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij]);
            v_prof.push(u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij]);
            h_prof.push(SURFACE_LAYER_HEIGHT_M);

            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij]);
                v_prof.push(u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij]);
                h_prof.push(h_agl[idx]);
            }

            let (sm_u, sm_v) = storm_motion.at(ij);
            let (_, _, total) = crate::met::wind::storm_relative_helicity(
                &u_prof, &v_prof, &h_prof, depth_m, sm_u, sm_v,
            );
            *srh_val = total;
        });
        Ok(srh)
    } else {
        // Default: use grid-parallel SRH with Bunkers
        // Prepend 10m winds as the surface layer for each column.
        let nz_aug = nz + 1;
        let mut u_aug = Vec::with_capacity(nz_aug * nxy);
        let mut v_aug = Vec::with_capacity(nz_aug * nxy);
        let mut h_aug = Vec::with_capacity(nz_aug * nxy);
        let mut p_aug = Vec::with_capacity(nz_aug * nxy);

        // Level 0: 10m winds anchored to the surface, with surface pressure.
        for ij in 0..nxy {
            u_aug.push(u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij]);
            v_aug.push(u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij]);
            h_aug.push(SURFACE_LAYER_HEIGHT_M);
            p_aug.push(psfc_hpa[ij]);
        }
        // Levels 1..nz: model levels
        for k in 0..nz {
            let off = k * nxy;
            for ij in 0..nxy {
                u_aug.push(u_grid[off + ij] * cosa[ij] - v_grid[off + ij] * sina[ij]);
                v_aug.push(u_grid[off + ij] * sina[ij] + v_grid[off + ij] * cosa[ij]);
                h_aug.push(h_agl[off + ij]);
                p_aug.push(pres_hpa[off + ij]);
            }
        }

        if matches!(
            storm_motion_method,
            Some(StormMotionMethod::NonPressureWeighted)
        ) {
            Ok(crate::met::composite::compute_srh_with_pressure(
                &u_aug,
                &v_aug,
                &h_aug,
                &[],
                nx,
                ny,
                nz_aug,
                depth_m,
            ))
        } else {
            Ok(crate::met::composite::compute_srh_with_pressure(
                &u_aug, &v_aug, &h_aug, &p_aug, nx, ny, nz_aug, depth_m,
            ))
        }
    }
}

/// Helper: compute bulk shear magnitude for a given layer.
fn compute_shear_field(f: &WrfFile, t: usize, bottom_m: f64, top_m: f64) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let h_agl = f.height_agl(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let _nxy = nx * ny;

    Ok(crate::met::composite::compute_shear(
        &u, &v, &h_agl, nx, ny, nz, bottom_m, top_m,
    ))
}

/// Helper: compute Bunkers storm motion for each column.
/// Returns (rm_u, rm_v, lm_u, lm_v, mean_u, mean_v) as 6 interleaved nxy fields.
///
/// Uses earth-rotated winds with 10m prepend, matching compute_srh_field.
fn compute_bunkers_columns(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let storm_motion_method = resolved_storm_motion_method(opts);
    let nxy = f.nx * f.ny;
    let cache_key = bunkers_cache_key(storm_motion_method);
    if let Some(stacked) = f.cached_field(cache_key) {
        if let Some(fields) = unpack_bunkers_stack(&stacked, nxy) {
            return Ok(fields);
        }
    }

    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let h_agl = f.height_agl(t)?;
    let pres_hpa = f.pressure_hpa(t)?;
    let psfc_hpa: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let mut rm_u = vec![0.0f64; nxy];
    let mut rm_v = vec![0.0f64; nxy];
    let mut lm_u = vec![0.0f64; nxy];
    let mut lm_v = vec![0.0f64; nxy];
    let mut mn_u = vec![0.0f64; nxy];
    let mut mn_v = vec![0.0f64; nxy];

    let results: Vec<_> = (0..nxy)
        .into_par_iter()
        .map(|ij| {
            // Prepend 10m wind as the surface layer.
            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            let mut h_prof = Vec::with_capacity(nz + 1);
            let mut p_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij]);
            v_prof.push(u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij]);
            h_prof.push(SURFACE_LAYER_HEIGHT_M);
            p_prof.push(psfc_hpa[ij]);

            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij]);
                v_prof.push(u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij]);
                h_prof.push(h_agl[idx]);
                p_prof.push(pres_hpa[idx]);
            }

            let ((ru, rv), (lu, lv), (mu, mv)) = match storm_motion_method {
                StormMotionMethod::PressureWeighted => {
                    crate::met::composite::pressure_weighted_bunkers_storm_motion(
                        &h_prof, &u_prof, &v_prof, &p_prof,
                    )
                }
                StormMotionMethod::NonPressureWeighted => {
                    crate::met::wind::bunkers_storm_motion(&u_prof, &v_prof, &h_prof)
                }
            };
            (ij, ru, rv, lu, lv, mu, mv)
        })
        .collect();

    for (ij, ru, rv, lu, lv, mu, mv) in results {
        rm_u[ij] = ru;
        rm_v[ij] = rv;
        lm_u[ij] = lu;
        lm_v[ij] = lv;
        mn_u[ij] = mu;
        mn_v[ij] = mv;
    }

    let stacked = pack_bunkers_stack(&rm_u, &rm_v, &lm_u, &lm_v, &mn_u, &mn_v);
    f.store_cached_field(cache_key.to_string(), stacked);
    Ok((rm_u, rm_v, lm_u, lm_v, mn_u, mn_v))
}

// ── Public compute functions ──

/// 0-1 km SRH (m^2/s^2). `[ny, nx]`
pub fn compute_srh1(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_srh_field(
        f,
        t,
        1000.0,
        opts.storm_motion.as_ref(),
        opts.storm_motion_method,
    )
}

/// 0-3 km SRH (m^2/s^2). `[ny, nx]`
pub fn compute_srh3(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_srh_field(
        f,
        t,
        3000.0,
        opts.storm_motion.as_ref(),
        opts.storm_motion_method,
    )
}

/// SRH with configurable depth (default 3000m). `[ny, nx]`
pub fn compute_srh(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let depth = opts.depth_m.unwrap_or(3000.0);
    compute_srh_field(
        f,
        t,
        depth,
        opts.storm_motion.as_ref(),
        opts.storm_motion_method,
    )
}

/// 0-1 km bulk wind shear magnitude (m/s). `[ny, nx]`
pub fn compute_shear_0_1km(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_shear_field(f, t, 0.0, 1000.0)
}

/// 0-6 km bulk wind shear magnitude (m/s). `[ny, nx]`
pub fn compute_shear_0_6km(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_shear_field(f, t, 0.0, 6000.0)
}

/// Bunkers right-mover storm motion (m/s). Returns `[u, v]` interleaved (2 * nxy).
pub fn compute_bunkers_rm(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (rm_u, rm_v, _, _, _, _) = compute_bunkers_columns(f, t, opts)?;
    let mut out = rm_u;
    out.extend(rm_v);
    Ok(out)
}

/// Bunkers left-mover storm motion (m/s). Returns `[u, v]` interleaved (2 * nxy).
pub fn compute_bunkers_lm(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, lm_u, lm_v, _, _) = compute_bunkers_columns(f, t, opts)?;
    let mut out = lm_u;
    out.extend(lm_v);
    Ok(out)
}

/// 0-6 km mean wind (m/s). Returns `[u, v]` interleaved (2 * nxy).
pub fn compute_mean_wind_0_6km(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, _, mn_u, mn_v) = compute_bunkers_columns(f, t, opts)?;
    let mut out = mn_u;
    out.extend(mn_v);
    Ok(out)
}

/// Effective inflow layer SRH (m^2/s^2). `[ny, nx]`
///
/// Finds the effective inflow layer where CAPE >= 100 J/kg and CIN >= -250 J/kg,
/// then computes storm-relative helicity over that layer using Bunkers storm motion
/// (or custom motion if `opts.storm_motion` is set).
///
/// Uses earth-rotated winds with 10m prepend, matching compute_srh_field.
pub fn compute_effective_srh(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    // Earth-rotated 3D winds
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let h_agl = f.height_agl(t)?;
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let psfc = f.psfc(t)?;
    let t2 = f.t2_for_opts(t, opts)?;
    let q2 = f.q2_for_opts(t, opts)?;
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;
    let effective_layers = effective_inflow_layer_grid(f, t, opts)?;

    let custom_sm = opts.storm_motion.as_ref();
    let storm_motion_method = resolved_storm_motion_method(opts);
    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let (p_prof, _, _, h_prof) = build_surface_augmented_thermo_column(
                &pres_hpa, &tc, &qv, &h_agl, psfc[ij], t2[ij], q2[ij], nz, nxy, ij,
            );
            let layer = match effective_layers.layer(ij) {
                Some(layer) => layer,
                None => return 0.0,
            };

            if layer.top_h <= layer.base_h {
                return 0.0;
            }

            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij]);
            v_prof.push(u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij]);
            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij]);
                v_prof.push(u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij]);
            }

            let (sm_u, sm_v) = if let Some(sm) = custom_sm {
                sm.at(ij)
            } else {
                let ((ru, rv), _, _) = match storm_motion_method {
                    StormMotionMethod::PressureWeighted => {
                        crate::met::composite::pressure_weighted_bunkers_storm_motion(
                            &h_prof, &u_prof, &v_prof, &p_prof,
                        )
                    }
                    StormMotionMethod::NonPressureWeighted => {
                        crate::met::wind::bunkers_storm_motion(&u_prof, &v_prof, &h_prof)
                    }
                };
                (ru, rv)
            };

            let (_, _, total) = crate::met::wind::storm_relative_helicity(
                &u_prof[layer.base_idx..],
                &v_prof[layer.base_idx..],
                &h_prof[layer.base_idx..],
                layer.top_h,
                sm_u,
                sm_v,
            );
            total
        })
        .collect())
}

/// Configurable bulk wind shear magnitude (m/s). `[ny, nx]`
///
/// Uses `opts.bottom_m` (default 0) and `opts.top_m` (default 6000) for the layer.
pub fn compute_bulk_shear(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let bottom = opts.bottom_m.unwrap_or(0.0);
    let top = opts.top_m.unwrap_or(6000.0);
    compute_shear_field(f, t, bottom, top)
}

/// Configurable mean wind (m/s). Returns `[u_mean, v_mean]` interleaved (2 * nxy). `[ny, nx]` per component.
///
/// Uses `opts.bottom_m` (default 0) and `opts.top_m` (default 6000) for the layer.
/// Uses earth-rotated winds with 10m prepend, matching compute_srh_field.
pub fn compute_mean_wind(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let h_agl = f.height_agl(t)?;
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let bottom = opts.bottom_m.unwrap_or(0.0);
    let top = opts.top_m.unwrap_or(6000.0);

    let mut mean_u = vec![0.0f64; nxy];
    let mut mean_v = vec![0.0f64; nxy];

    let results: Vec<_> = (0..nxy)
        .into_par_iter()
        .map(|ij| {
            // Prepend 10m wind as the surface layer.
            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            let mut h_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij]);
            v_prof.push(u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij]);
            h_prof.push(SURFACE_LAYER_HEIGHT_M);

            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij]);
                v_prof.push(u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij]);
                h_prof.push(h_agl[idx]);
            }

            let (mu, mv) = crate::met::wind::mean_wind(&u_prof, &v_prof, &h_prof, bottom, top);
            (ij, mu, mv)
        })
        .collect();

    for (ij, mu, mv) in results {
        mean_u[ij] = mu;
        mean_v[ij] = mv;
    }

    let mut out = mean_u;
    out.extend(mean_v);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        bunkers_cache_key, pack_bunkers_stack, unpack_bunkers_stack, StormMotionMethod,
        SURFACE_LAYER_HEIGHT_M,
    };

    #[test]
    fn surface_augmentation_anchors_10m_winds_at_zero_agl() {
        assert_eq!(SURFACE_LAYER_HEIGHT_M, 0.0);
    }

    #[test]
    fn bunkers_stack_round_trip_preserves_field_order() {
        let stacked = pack_bunkers_stack(
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
            &[11.0, 12.0],
        );
        let unpacked = unpack_bunkers_stack(&stacked, 2).unwrap();
        assert_eq!(unpacked.0, vec![1.0, 2.0]);
        assert_eq!(unpacked.1, vec![3.0, 4.0]);
        assert_eq!(unpacked.2, vec![5.0, 6.0]);
        assert_eq!(unpacked.3, vec![7.0, 8.0]);
        assert_eq!(unpacked.4, vec![9.0, 10.0]);
        assert_eq!(unpacked.5, vec![11.0, 12.0]);
    }

    #[test]
    fn bunkers_cache_key_tracks_method() {
        assert_eq!(
            bunkers_cache_key(StormMotionMethod::PressureWeighted),
            "bunkers_stack_pw"
        );
        assert_eq!(
            bunkers_cache_key(StormMotionMethod::NonPressureWeighted),
            "bunkers_stack_npw"
        );
    }
}
