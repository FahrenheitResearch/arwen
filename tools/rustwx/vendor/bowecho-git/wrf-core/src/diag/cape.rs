//! CAPE diagnostic variables: sbcape, mlcape, mucape, cape2d, cape3d, lcl, lfc, el
//!
//! Uses crate::met::composite::compute_cape_cin() for parallel grid computation
//! and crate::met::thermo::cape_cin_core() for column-by-column fallback.

use rayon::prelude::*;

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::{SharedField, WrfFile};

const CAPE_STACK_FIELDS: usize = 4;
const EFFECTIVE_LAYER_STACK_FIELDS: usize = 6;

fn cape_cache_key(parcel_type: &str, top_m: Option<f64>, lake_interp: Option<f64>) -> String {
    let parcel_type = parcel_type.trim().to_ascii_lowercase();
    let top_m = top_m
        .map(|v| format!("{:016x}", v.to_bits()))
        .unwrap_or_else(|| "full".to_string());
    let lake_interp = match lake_interp {
        Some(v) if v > 0.0 => format!("{:016x}", v.to_bits()),
        _ => "none".to_string(),
    };
    format!("cape_stack_{parcel_type}_{top_m}_{lake_interp}")
}

fn pack_cape_stack(cape: &[f64], cin: &[f64], lcl: &[f64], lfc: &[f64]) -> Vec<f64> {
    let nxy = cape.len();
    let mut stacked = Vec::with_capacity(CAPE_STACK_FIELDS * nxy);
    stacked.extend_from_slice(cape);
    stacked.extend_from_slice(cin);
    stacked.extend_from_slice(lcl);
    stacked.extend_from_slice(lfc);
    stacked
}

fn unpack_cape_stack(
    stacked: &[f64],
    nxy: usize,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if stacked.len() != CAPE_STACK_FIELDS * nxy {
        return None;
    }

    Some((
        stacked[0..nxy].to_vec(),
        stacked[nxy..2 * nxy].to_vec(),
        stacked[2 * nxy..3 * nxy].to_vec(),
        stacked[3 * nxy..4 * nxy].to_vec(),
    ))
}

/// Helper: extract the 3-D + 2-D fields needed for CAPE, call compute_cape_cin.
/// Returns (cape_2d, cin_2d, lcl_2d, lfc_2d).
pub(crate) fn compute_cape_fields(
    f: &WrfFile,
    t: usize,
    parcel_type: &str,
    top_m: Option<f64>,
    lake_interp: Option<f64>,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let nxy = f.nx * f.ny;
    let cache_key = cape_cache_key(parcel_type, top_m, lake_interp);
    if let Some(stacked) = f.cached_field(&cache_key) {
        if let Some(fields) = unpack_cape_stack(&stacked, nxy) {
            return Ok(fields);
        }
    }

    let pres = f.full_pressure(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;
    let t2 = match lake_interp {
        Some(a) if a > 0.0 => f.t2_lake_corrected(t, a)?,
        _ => f.t2(t)?.to_vec(),
    };
    let q2 = match lake_interp {
        Some(a) if a > 0.0 => f.q2_lake_corrected(t, a)?,
        _ => f.q2(t)?.to_vec(),
    };

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    // compute_cape_cin expects Pa pressure and K temperature -- it converts
    // internally.  Do NOT pre-convert here.
    // Use crate::met's grid-parallel CAPE computation if no top_m specified
    if top_m.is_none() {
        let (cape, cin, lcl, lfc) = crate::met::composite::compute_cape_cin(
            &pres,
            &tc,
            &qv,
            &h_agl,
            &psfc,
            &t2,
            &q2,
            nx,
            ny,
            nz,
            parcel_type,
        );
        let stacked = pack_cape_stack(&cape, &cin, &lcl, &lfc);
        f.store_cached_field(cache_key, stacked);
        return Ok((cape, cin, lcl, lfc));
    }

    // With top_m: use column-by-column cape_cin_core.

    let mut cape = vec![0.0f64; nxy];
    let mut cin = vec![0.0f64; nxy];
    let mut lcl = vec![0.0f64; nxy];
    let mut lfc = vec![0.0f64; nxy];

    cape.par_iter_mut()
        .zip(cin.par_iter_mut())
        .zip(lcl.par_iter_mut())
        .zip(lfc.par_iter_mut())
        .enumerate()
        .for_each(|(ij, (((cape_v, cin_v), lcl_v), lfc_v))| {
            // Extract column profiles -- pass Pa pressure to cape_cin_core
            // which auto-detects units.
            let mut p_prof = Vec::with_capacity(nz);
            let mut t_prof = Vec::with_capacity(nz);
            let mut td_prof = Vec::with_capacity(nz);
            let mut h_prof = Vec::with_capacity(nz);

            for k in 0..nz {
                let idx = k * nxy + ij;
                let p_hpa = pres[idx] / 100.0;
                p_prof.push(p_hpa); // hPa
                t_prof.push(tc[idx]); // Celsius
                                      // Compute Td from q and p
                let q = qv[idx].max(1e-10);
                let e = q * p_hpa / (0.622 + q);
                let ln_e = (e / 6.112).max(1e-10).ln();
                let td = (243.5 * ln_e) / (17.67 - ln_e);
                td_prof.push(td); // Celsius
                h_prof.push(h_agl[idx]); // m AGL
            }

            // Pass hPa/C -- consistent units so auto-detect doesn't double-convert
            let (psfc_hpa, t2_c, td2_c) = surface_parcel_from_2m(psfc[ij], t2[ij], q2[ij]);
            let (c, ci, l, lf) = crate::met::thermo::cape_cin_core(
                &p_prof,
                &t_prof,
                &td_prof,
                &h_prof,
                psfc_hpa,
                t2_c,
                td2_c,
                parcel_type,
                100.0,
                300.0,
                top_m,
            );

            *cape_v = c;
            *cin_v = ci;
            *lcl_v = l;
            *lfc_v = lf;
        });

    let stacked = pack_cape_stack(&cape, &cin, &lcl, &lfc);
    f.store_cached_field(cache_key, stacked);
    Ok((cape, cin, lcl, lfc))
}

fn resolve_parcel_type(opts: &ComputeOpts, default: &str) -> String {
    opts.parcel_type
        .as_deref()
        .unwrap_or(default)
        .to_lowercase()
}

pub(crate) fn surface_parcel_from_2m(psfc_pa: f64, t2_k: f64, q2_kgkg: f64) -> (f64, f64, f64) {
    let psfc_hpa = psfc_pa / 100.0;
    let t2_c = t2_k - 273.15;
    let td2_c = crate::met::composite::dewpoint_from_q(q2_kgkg, psfc_hpa).min(t2_c);
    (psfc_hpa, t2_c, td2_c)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EffectiveLayerColumn {
    pub base_idx: usize,
    pub base_h: f64,
    pub top_h: f64,
    pub mu_cape: f64,
    pub mu_el_h: Option<f64>,
}

#[derive(Clone)]
pub(crate) struct EffectiveLayerGrid {
    stacked: SharedField,
    nxy: usize,
}

impl EffectiveLayerGrid {
    fn from_stacked(stacked: SharedField, nxy: usize) -> Option<Self> {
        (stacked.len() == EFFECTIVE_LAYER_STACK_FIELDS * nxy).then_some(Self { stacked, nxy })
    }

    pub(crate) fn layer(&self, ij: usize) -> Option<EffectiveLayerColumn> {
        if ij >= self.nxy {
            return None;
        }

        let offset = ij * EFFECTIVE_LAYER_STACK_FIELDS;
        let base_idx = self.stacked[offset];
        if !base_idx.is_finite() || base_idx < 0.0 {
            return None;
        }

        Some(EffectiveLayerColumn {
            base_idx: base_idx as usize,
            base_h: self.stacked[offset + 1],
            top_h: self.stacked[offset + 2],
            mu_cape: self.stacked[offset + 3],
            mu_el_h: (self.stacked[offset + 4] != 0.0).then_some(self.stacked[offset + 5]),
        })
    }
}

fn effective_layer_cache_key(t: usize, lake_interp: Option<f64>) -> String {
    let lake_interp = match lake_interp {
        Some(value) if value > 0.0 => format!("{:016x}", value.to_bits()),
        _ => "none".to_string(),
    };
    format!("effective_layer_stack_{t}_{lake_interp}")
}

pub(crate) fn build_surface_augmented_thermo_column(
    pres_hpa: &[f64],
    tc: &[f64],
    qv: &[f64],
    h_agl: &[f64],
    psfc_pa: f64,
    t2_k: f64,
    q2_kgkg: f64,
    nz: usize,
    nxy: usize,
    ij: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (psfc_hpa, t2_c, td2_c) = surface_parcel_from_2m(psfc_pa, t2_k, q2_kgkg);
    let mut p_prof = Vec::with_capacity(nz + 1);
    let mut t_prof = Vec::with_capacity(nz + 1);
    let mut td_prof = Vec::with_capacity(nz + 1);
    let mut h_prof = Vec::with_capacity(nz + 1);

    p_prof.push(psfc_hpa);
    t_prof.push(t2_c);
    td_prof.push(td2_c);
    h_prof.push(0.0);

    for k in 0..nz {
        let idx = k * nxy + ij;
        p_prof.push(pres_hpa[idx]);
        t_prof.push(tc[idx]);
        td_prof.push(crate::met::composite::dewpoint_from_q(qv[idx], pres_hpa[idx]).min(tc[idx]));
        h_prof.push(h_agl[idx]);
    }

    (p_prof, t_prof, td_prof, h_prof)
}

pub(crate) fn find_effective_inflow_layer(
    p_prof: &[f64],
    t_prof: &[f64],
    td_prof: &[f64],
    h_prof: &[f64],
) -> Option<EffectiveLayerColumn> {
    let mut eff_base: Option<usize> = None;
    let mut eff_top: Option<usize> = None;
    let mut mu_cape = 0.0f64;
    let mut mu_idx: Option<usize> = None;

    for k in 0..p_prof.len() {
        if p_prof.len() - k < 2 {
            break;
        }

        let (cape_k, cin_k, _, _) = crate::met::thermo::cape_cin_core(
            &p_prof[k..],
            &t_prof[k..],
            &td_prof[k..],
            &h_prof[k..],
            p_prof[k],
            t_prof[k],
            td_prof[k],
            "sb",
            100.0,
            300.0,
            None,
        );

        if cape_k >= 100.0 && cin_k >= -250.0 {
            if eff_base.is_none() {
                eff_base = Some(k);
            }
            eff_top = Some(k);
            if cape_k > mu_cape {
                mu_cape = cape_k;
                mu_idx = Some(k);
            }
        } else if eff_base.is_some() {
            break;
        }
    }

    let (base_idx, top_idx, mu_idx) = match (eff_base, eff_top, mu_idx) {
        (Some(base_idx), Some(top_idx), Some(mu_idx)) => (base_idx, top_idx, mu_idx),
        _ => return None,
    };

    let mu_el_h = if p_prof.len() - mu_idx >= 2 {
        crate::met::thermo::el(&p_prof[mu_idx..], &t_prof[mu_idx..], &td_prof[mu_idx..]).and_then(
            |(el_pres, _)| {
                if el_pres > 0.0 {
                    Some(crate::met::thermo::get_height_at_pres(
                        el_pres, p_prof, h_prof,
                    ))
                } else {
                    None
                }
            },
        )
    } else {
        None
    };

    Some(EffectiveLayerColumn {
        base_idx,
        base_h: h_prof[base_idx],
        top_h: h_prof[top_idx],
        mu_cape,
        mu_el_h,
    })
}

pub(crate) fn effective_inflow_layer_grid(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<EffectiveLayerGrid> {
    let nxy = f.nx * f.ny;
    let cache_key = effective_layer_cache_key(t, opts.lake_interp);
    if let Some(stacked) = f.cached_field(&cache_key) {
        if let Some(grid) = EffectiveLayerGrid::from_stacked(stacked, nxy) {
            return Ok(grid);
        }
    }

    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;
    let t2 = f.t2_for_opts(t, opts)?;
    let q2 = f.q2_for_opts(t, opts)?;
    let nz = f.nz;

    let mut stacked = vec![f64::NAN; EFFECTIVE_LAYER_STACK_FIELDS * nxy];
    stacked
        .par_chunks_mut(EFFECTIVE_LAYER_STACK_FIELDS)
        .enumerate()
        .for_each(|(ij, values)| {
            let (p_prof, t_prof, td_prof, h_prof) = build_surface_augmented_thermo_column(
                &pres_hpa, &tc, &qv, &h_agl, psfc[ij], t2[ij], q2[ij], nz, nxy, ij,
            );
            if let Some(layer) = find_effective_inflow_layer(&p_prof, &t_prof, &td_prof, &h_prof) {
                values[0] = layer.base_idx as f64;
                values[1] = layer.base_h;
                values[2] = layer.top_h;
                values[3] = layer.mu_cape;
                if let Some(mu_el_h) = layer.mu_el_h {
                    values[4] = 1.0;
                    values[5] = mu_el_h;
                } else {
                    values[4] = 0.0;
                }
            }
        });

    let stacked = f.store_cached_field(cache_key, stacked);
    Ok(EffectiveLayerGrid::from_stacked(stacked, nxy)
        .expect("effective-layer cache has the expected shape"))
}

// ── Public compute functions ──

pub fn compute_sbcape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (cape, _, _, _) = compute_cape_fields(f, t, "sb", opts.top_m, opts.lake_interp)?;
    Ok(cape)
}

pub fn compute_sbcin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = compute_cape_fields(f, t, "sb", opts.top_m, opts.lake_interp)?;
    Ok(cin)
}

pub fn compute_mlcape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (cape, _, _, _) = compute_cape_fields(f, t, "ml", opts.top_m, opts.lake_interp)?;
    Ok(cape)
}

pub fn compute_mlcin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = compute_cape_fields(f, t, "ml", opts.top_m, opts.lake_interp)?;
    Ok(cin)
}

pub fn compute_mucape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (cape, _, _, _) = compute_cape_fields(f, t, "mu", opts.top_m, opts.lake_interp)?;
    Ok(cape)
}

pub fn compute_mucin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = compute_cape_fields(f, t, "mu", opts.top_m, opts.lake_interp)?;
    Ok(cin)
}

pub fn compute_lcl(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pt = resolve_parcel_type(opts, "sb");
    let (_, _, lcl, _) = compute_cape_fields(f, t, &pt, opts.top_m, opts.lake_interp)?;
    Ok(lcl)
}

pub fn compute_lfc(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pt = resolve_parcel_type(opts, "sb");
    let (_, _, _, lfc) = compute_cape_fields(f, t, &pt, opts.top_m, opts.lake_interp)?;
    Ok(lfc)
}

pub fn compute_el(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    // EL not directly returned by compute_cape_cin, compute column-by-column
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc_raw = f.psfc(t)?;
    let t2_raw = f.t2(t)?;
    let q2 = f.q2(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;
    let pt = resolve_parcel_type(opts, "sb");

    let mut el = vec![0.0f64; nxy];
    el.par_iter_mut().enumerate().for_each(|(ij, el_v)| {
        let mut p_prof = Vec::with_capacity(nz);
        let mut t_prof = Vec::with_capacity(nz);
        let mut td_prof = Vec::with_capacity(nz);
        let mut h_prof = Vec::with_capacity(nz);

        for k in 0..nz {
            let idx = k * nxy + ij;
            p_prof.push(pres_hpa[idx]);
            t_prof.push(tc[idx]);
            let q = qv[idx].max(1e-10);
            let e = q * pres_hpa[idx] / (0.622 + q);
            let ln_e = (e / 6.112).max(1e-10).ln();
            td_prof.push((243.5 * ln_e) / (17.67 - ln_e));
            h_prof.push(h_agl[idx]);
        }

        // Select parcel based on parcel_type, then build a modified profile
        // with the parcel at the base for the el() function.
        let (psfc_hpa, t2_c, td2_c) = surface_parcel_from_2m(psfc_raw[ij], t2_raw[ij], q2[ij]);

        // Prepend surface data for parcel selection
        let mut full_p = vec![psfc_hpa];
        let mut full_t = vec![t2_c];
        let mut full_td = vec![td2_c];
        full_p.extend_from_slice(&p_prof);
        full_t.extend_from_slice(&t_prof);
        full_td.extend_from_slice(&td_prof);

        // Select the parcel based on type
        let (p_parcel, t_parcel, td_parcel) = match pt.as_str() {
            "ml" => crate::met::thermo::get_mixed_layer_parcel(&full_p, &full_t, &full_td, 100.0),
            "mu" => crate::met::thermo::get_most_unstable_parcel(&full_p, &full_t, &full_td, 300.0),
            _ => (psfc_hpa, t2_c, td2_c), // "sb"
        };

        // Build profile starting from the parcel level
        let mut mod_p = vec![p_parcel];
        let mut mod_t = vec![t_parcel];
        let mut mod_td = vec![td_parcel];
        for k in 0..p_prof.len() {
            if p_prof[k] < p_parcel {
                mod_p.push(p_prof[k]);
                mod_t.push(t_prof[k]);
                mod_td.push(td_prof[k]);
            }
        }

        if mod_p.len() >= 2 {
            if let Some((el_pres, _t_el)) = crate::met::thermo::el(&mod_p, &mod_t, &mod_td) {
                if el_pres > 0.0 {
                    *el_v = crate::met::thermo::get_height_at_pres(el_pres, &p_prof, &h_prof);
                }
            }
        }
    });

    Ok(el)
}

/// cape2d: backward-compatible with wrf-python. Returns `[cape, cin, lcl, lfc]` interleaved (4 * nxy).
pub fn compute_cape2d(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pt = resolve_parcel_type(opts, "ml");
    let (cape, cin, lcl, lfc) = compute_cape_fields(f, t, &pt, opts.top_m, opts.lake_interp)?;
    let mut out = cape;
    out.extend(cin);
    out.extend(lcl);
    out.extend(lfc);
    Ok(out)
}

/// cape3d: column CAPE at every level (3-D). Uses the column method for each parcel starting level.
pub fn compute_cape3d(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let _psfc_hpa: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let _t2_c: Vec<f64> = f.t2(t)?.iter().map(|t| t - 273.15).collect();

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    // For each column, compute CAPE for a parcel starting at each level
    let mut cape3d = vec![0.0f64; nz * nxy];

    cape3d.chunks_mut(nxy).enumerate().for_each(|(k, plane)| {
        for ij in 0..nxy {
            let mut p_prof = Vec::with_capacity(nz - k);
            let mut t_prof = Vec::with_capacity(nz - k);
            let mut td_prof = Vec::with_capacity(nz - k);
            let mut h_prof = Vec::with_capacity(nz - k);

            for kk in k..nz {
                let idx = kk * nxy + ij;
                p_prof.push(pres_hpa[idx]);
                t_prof.push(tc[idx]);
                let q = qv[idx].max(1e-10);
                let e = q * pres_hpa[idx] / (0.622 + q);
                let ln_e = (e / 6.112).max(1e-10).ln();
                td_prof.push((243.5 * ln_e) / (17.67 - ln_e));
                h_prof.push(h_agl[idx]);
            }

            if p_prof.len() >= 2 {
                let (c, _, _, _) = crate::met::thermo::cape_cin_core(
                    &p_prof, &t_prof, &td_prof, &h_prof, p_prof[0], t_prof[0], td_prof[0], "sb",
                    100.0, 300.0, None,
                );
                plane[ij] = c;
            }
        }
    });

    Ok(cape3d)
}

// ── Custom parcel helper ──

/// Compute CAPE fields using a custom parcel (user-specified pressure, temperature, dewpoint).
/// Column-by-column, using "sb" parcel type but substituting custom values for psfc/t2m/td2m.
/// Returns (cape_2d, cin_2d, lcl_2d, lfc_2d).
fn compute_cape_fields_custom(
    f: &WrfFile,
    t: usize,
    parcel_p_hpa: f64,
    parcel_t_c: f64,
    parcel_td_c: f64,
    top_m: Option<f64>,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let mut cape = vec![0.0f64; nxy];
    let mut cin = vec![0.0f64; nxy];
    let mut lcl = vec![0.0f64; nxy];
    let mut lfc = vec![0.0f64; nxy];

    cape.iter_mut()
        .zip(cin.iter_mut())
        .zip(lcl.iter_mut())
        .zip(lfc.iter_mut())
        .enumerate()
        .for_each(|(ij, (((cape_v, cin_v), lcl_v), lfc_v))| {
            let mut p_prof = Vec::with_capacity(nz);
            let mut t_prof = Vec::with_capacity(nz);
            let mut td_prof = Vec::with_capacity(nz);
            let mut h_prof = Vec::with_capacity(nz);

            for k in 0..nz {
                let idx = k * nxy + ij;
                p_prof.push(pres_hpa[idx]);
                t_prof.push(tc[idx]);
                let q = qv[idx].max(1e-10);
                let e = q * pres_hpa[idx] / (0.622 + q);
                let ln_e = (e / 6.112).max(1e-10).ln();
                td_prof.push((243.5 * ln_e) / (17.67 - ln_e));
                h_prof.push(h_agl[idx]);
            }

            // Filter profile to only include levels at and above the custom
            // parcel pressure so the profile is monotonically decreasing.
            let mut filt_p = Vec::new();
            let mut filt_t = Vec::new();
            let mut filt_td = Vec::new();
            let mut filt_h = Vec::new();
            for k in 0..p_prof.len() {
                if p_prof[k] <= parcel_p_hpa {
                    filt_p.push(p_prof[k]);
                    filt_t.push(t_prof[k]);
                    filt_td.push(td_prof[k]);
                    filt_h.push(h_prof[k]);
                }
            }

            if filt_p.len() < 2 {
                // Not enough levels above custom parcel -- skip
                return;
            }

            // Use "sb" parcel type with the custom parcel as the surface
            let (c, ci, l, lf) = crate::met::thermo::cape_cin_core(
                &filt_p,
                &filt_t,
                &filt_td,
                &filt_h,
                parcel_p_hpa,
                parcel_t_c,
                parcel_td_c,
                "sb",
                100.0,
                300.0,
                top_m,
            );

            *cape_v = c;
            *cin_v = ci;
            *lcl_v = l;
            *lfc_v = lf;
        });

    Ok((cape, cin, lcl, lfc))
}

/// Determine whether opts specifies a custom parcel (all three of pressure, temperature,
/// and dewpoint are set).
fn has_custom_parcel(opts: &ComputeOpts) -> bool {
    opts.parcel_pressure.is_some()
        && opts.parcel_temperature.is_some()
        && opts.parcel_dewpoint.is_some()
}

/// Resolve CAPE fields for generic functions: custom parcel, named parcel type, or default "sb".
fn resolve_cape_fields(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if has_custom_parcel(opts) {
        let p = opts.parcel_pressure.unwrap();
        let tc = opts.parcel_temperature.unwrap();
        let td = opts.parcel_dewpoint.unwrap();
        compute_cape_fields_custom(f, t, p, tc, td, opts.top_m)
    } else {
        let pt = resolve_parcel_type(opts, "sb");
        compute_cape_fields(f, t, &pt, opts.top_m, opts.lake_interp)
    }
}

// ── Generic public compute functions ──

/// Generic CAPE computation.
///
/// - If `opts.parcel_pressure`, `opts.parcel_temperature`, and `opts.parcel_dewpoint` are all
///   set, uses a custom parcel lifted from the specified conditions.
/// - Otherwise falls back to `opts.parcel_type` ("sb", "ml", "mu"), defaulting to "sb".
/// - Supports `opts.top_m` for truncated CAPE (e.g., 0-3 km CAPE).
pub fn compute_cape_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (cape, _, _, _) = resolve_cape_fields(f, t, opts)?;
    Ok(cape)
}

/// Generic CIN computation. Same parcel logic as `compute_cape_generic`.
pub fn compute_cin_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = resolve_cape_fields(f, t, opts)?;
    Ok(cin)
}

/// Generic LCL height computation. Same parcel logic as `compute_cape_generic`.
pub fn compute_lcl_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, lcl, _) = resolve_cape_fields(f, t, opts)?;
    Ok(lcl)
}

/// Generic LFC height computation. Same parcel logic as `compute_cape_generic`.
pub fn compute_lfc_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, lfc) = resolve_cape_fields(f, t, opts)?;
    Ok(lfc)
}

/// Generic EL height computation. Same parcel logic as `compute_cape_generic`.
///
/// For custom parcels, column-by-column EL is computed via `crate::met::thermo::el` using the
/// custom parcel's temperature and dewpoint as the surface values in the profile.
pub fn compute_el_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    if has_custom_parcel(opts) {
        let parcel_p = opts.parcel_pressure.unwrap();
        let parcel_t = opts.parcel_temperature.unwrap();
        let parcel_td = opts.parcel_dewpoint.unwrap();

        let pres_hpa = f.pressure_hpa(t)?;
        let tc = f.temperature_c(t)?;
        let qv = f.qvapor(t)?;
        let h_agl = f.height_agl(t)?;

        let nx = f.nx;
        let ny = f.ny;
        let nz = f.nz;
        let nxy = nx * ny;

        let mut el = vec![0.0f64; nxy];
        el.iter_mut().enumerate().for_each(|(ij, el_v)| {
            let mut p_prof = Vec::with_capacity(nz);
            let mut t_prof = Vec::with_capacity(nz);
            let mut td_prof = Vec::with_capacity(nz);
            let mut h_prof = Vec::with_capacity(nz);

            for k in 0..nz {
                let idx = k * nxy + ij;
                p_prof.push(pres_hpa[idx]);
                t_prof.push(tc[idx]);
                let q = qv[idx].max(1e-10);
                let e = q * pres_hpa[idx] / (0.622 + q);
                let ln_e = (e / 6.112).max(1e-10).ln();
                td_prof.push((243.5 * ln_e) / (17.67 - ln_e));
                h_prof.push(h_agl[idx]);
            }

            // Build a modified profile starting from the custom parcel level:
            // insert the custom parcel at the base and use el() to find the EL.
            let mut mod_p = vec![parcel_p];
            let mut mod_t = vec![parcel_t];
            let mut mod_td = vec![parcel_td];

            for k in 0..nz {
                if p_prof[k] < parcel_p {
                    mod_p.push(p_prof[k]);
                    mod_t.push(t_prof[k]);
                    mod_td.push(td_prof[k]);
                }
            }

            if mod_p.len() >= 2 {
                if let Some((el_pres, _)) = crate::met::thermo::el(&mod_p, &mod_t, &mod_td) {
                    if el_pres > 0.0 {
                        *el_v = crate::met::thermo::get_height_at_pres(el_pres, &p_prof, &h_prof);
                    }
                }
            }
        });

        Ok(el)
    } else {
        // Delegate to existing EL function for named parcel types
        compute_el(f, t, opts)
    }
}

// ── Effective inflow layer ──

/// Compute the effective inflow layer bounds for each grid column.
///
/// Returns a flat vector of length `2 * nxy` with base heights as the first
/// contiguous plane followed by top heights as the second plane:
/// `[base_0, base_1, ..., base_{nxy-1}, top_0, top_1, ..., top_{nxy-1}]`
/// in meters AGL.
///
/// Effective inflow base: lowest model level where a parcel lifted from that level
/// produces CAPE >= 100 J/kg AND CIN >= -250 J/kg.
///
/// Effective inflow top: highest contiguous level still meeting the CAPE/CIN criteria.
pub fn compute_effective_inflow_layer(
    f: &WrfFile,
    t: usize,
    _opts: &ComputeOpts,
) -> WrfResult<Vec<f64>> {
    let nxy = f.nx * f.ny;
    let layers = effective_inflow_layer_grid(f, t, _opts)?;
    let mut result = vec![0.0; 2 * nxy];
    for ij in 0..nxy {
        if let Some(layer) = layers.layer(ij) {
            result[ij] = layer.base_h;
            result[nxy + ij] = layer.top_h;
        }
    }
    Ok(result)
}

/// Compute the CAPE of the most-unstable parcel within the effective inflow layer.
///
/// For each column, scans upward from the surface computing CAPE/CIN for a parcel
/// lifted from each model level. The effective inflow layer is defined as the
/// contiguous region where CAPE >= 100 J/kg AND CIN >= -250 J/kg. Returns the CAPE
/// of the most-unstable parcel within that layer. If no effective layer is found,
/// returns 0.0 for that column.
pub fn compute_effective_inflow_cape(
    f: &WrfFile,
    t: usize,
    _opts: &ComputeOpts,
) -> WrfResult<Vec<f64>> {
    let layers = effective_inflow_layer_grid(f, t, _opts)?;
    Ok((0..f.nx * f.ny)
        .map(|ij| layers.layer(ij).map(|layer| layer.mu_cape).unwrap_or(0.0))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_parcel_responds_to_q2() {
        let (_, t2_c, td_dry) = surface_parcel_from_2m(100_000.0, 303.15, 0.004);
        let (_, _, td_moist) = surface_parcel_from_2m(100_000.0, 303.15, 0.016);

        assert!(td_moist > td_dry);
        assert!(td_moist <= t2_c);
    }

    #[test]
    fn truncated_cape_depends_on_surface_moisture() {
        let p_prof = [950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0];
        let t_prof = [25.0, 22.0, 18.0, 14.0, 10.0, 6.0, 2.0];
        let td_prof = [14.0, 10.0, 6.0, 2.0, -2.0, -8.0, -16.0];
        let h_prof = [250.0, 750.0, 1250.0, 1750.0, 2250.0, 2750.0, 3250.0];

        let (psfc_hpa, t2_c, td2_dry) = surface_parcel_from_2m(100_000.0, 303.15, 0.004);
        let (_, _, td2_moist) = surface_parcel_from_2m(100_000.0, 303.15, 0.016);

        let (cape_dry, _, _, _) = crate::met::thermo::cape_cin_core(
            &p_prof,
            &t_prof,
            &td_prof,
            &h_prof,
            psfc_hpa,
            t2_c,
            td2_dry,
            "sb",
            100.0,
            300.0,
            Some(3000.0),
        );
        let (cape_moist, _, _, _) = crate::met::thermo::cape_cin_core(
            &p_prof,
            &t_prof,
            &td_prof,
            &h_prof,
            psfc_hpa,
            t2_c,
            td2_moist,
            "sb",
            100.0,
            300.0,
            Some(3000.0),
        );

        assert!(
            cape_moist > cape_dry + 1.0,
            "expected truncated CAPE to respond to 2m moisture, got dry={cape_dry} moist={cape_moist}"
        );
    }

    #[test]
    fn effective_inflow_returns_layer_top_not_mu_parcel_el() {
        let p_prof = [
            1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0,
            500.0,
        ];
        let t_prof = [
            29.0, 27.5, 26.0, 24.0, 22.0, 18.0, 14.0, 10.0, 6.0, 1.0, -5.0, -12.0, -20.0,
        ];
        let td_prof = [
            23.0, 22.0, 20.0, 17.0, 14.0, 8.0, 3.0, -2.0, -8.0, -15.0, -24.0, -34.0, -45.0,
        ];
        let h_prof = [
            0.0, 200.0, 450.0, 750.0, 1050.0, 1550.0, 2150.0, 2900.0, 3800.0, 4900.0, 6200.0,
            7800.0, 9600.0,
        ];

        let layer = find_effective_inflow_layer(&p_prof, &t_prof, &td_prof, &h_prof)
            .expect("effective layer");

        let mut expected_top_idx = None;
        for k in 0..p_prof.len() {
            if p_prof.len() - k < 2 {
                break;
            }
            let (cape_k, cin_k, _, _) = crate::met::thermo::cape_cin_core(
                &p_prof[k..],
                &t_prof[k..],
                &td_prof[k..],
                &h_prof[k..],
                p_prof[k],
                t_prof[k],
                td_prof[k],
                "sb",
                100.0,
                300.0,
                None,
            );
            if cape_k >= 100.0 && cin_k >= -250.0 {
                expected_top_idx = Some(k);
            } else if expected_top_idx.is_some() {
                break;
            }
        }
        let expected_top_idx = expected_top_idx.expect("qualifying effective layer");

        assert_eq!(layer.base_idx, 0);
        assert!(layer.top_h > layer.base_h);
        assert!(expected_top_idx < h_prof.len() - 1);
        assert_eq!(layer.top_h, h_prof[expected_top_idx]);
    }

    #[test]
    fn cape_stack_round_trip_preserves_field_order() {
        let stacked = pack_cape_stack(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]);
        let unpacked = unpack_cape_stack(&stacked, 2).unwrap();
        assert_eq!(unpacked.0, vec![1.0, 2.0]);
        assert_eq!(unpacked.1, vec![3.0, 4.0]);
        assert_eq!(unpacked.2, vec![5.0, 6.0]);
        assert_eq!(unpacked.3, vec![7.0, 8.0]);
    }

    #[test]
    fn cape_cache_key_treats_nonpositive_lake_interp_as_disabled() {
        let none_key = cape_cache_key("sb", None, None);
        let zero_key = cape_cache_key("SB", None, Some(0.0));
        let neg_key = cape_cache_key("sb", None, Some(-1.0));
        assert_eq!(none_key, zero_key);
        assert_eq!(none_key, neg_key);
    }

    #[test]
    fn effective_layer_grid_round_trip_preserves_columns() {
        let stacked = std::sync::Arc::<[f64]>::from(vec![
            2.0,
            400.0,
            1_200.0,
            1_500.0,
            1.0,
            9_000.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ]);
        let grid = EffectiveLayerGrid::from_stacked(stacked, 2).unwrap();
        let layer = grid.layer(0).unwrap();
        assert_eq!(layer.base_idx, 2);
        assert_eq!(layer.base_h, 400.0);
        assert_eq!(layer.top_h, 1_200.0);
        assert_eq!(layer.mu_cape, 1_500.0);
        assert_eq!(layer.mu_el_h, Some(9_000.0));
        assert!(grid.layer(1).is_none());
        assert!(grid.layer(2).is_none());
    }

    #[test]
    fn effective_layer_cache_key_tracks_time_and_lake_correction() {
        assert_ne!(
            effective_layer_cache_key(0, None),
            effective_layer_cache_key(1, None)
        );
        assert_eq!(
            effective_layer_cache_key(0, None),
            effective_layer_cache_key(0, Some(0.0))
        );
        assert_ne!(
            effective_layer_cache_key(0, None),
            effective_layer_cache_key(0, Some(1_000.0))
        );
    }
}
