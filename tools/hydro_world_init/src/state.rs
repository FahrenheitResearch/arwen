//! Source state -> per-catchment initial state.
//!
//! Land-weighted area averages (weights x land fraction, renormalised; a
//! catchment with no land cell in its footprint takes the nearest land cells
//! and is counted), the row's soil-moisture transfer, depth-overlap remapping
//! onto the Noah layers, the Noah-OWP supercooled-liquid split, CFE storage
//! fractions and snow.
use crate::cfe::Params;
use crate::grid::Weights;
use crate::ncread::{self, Source};
use crate::table::{Row, Table, Transfer};
use crate::Result;
use rayon::prelude::*;
use serde::Serialize;

pub struct LandWeights {
    pub offsets: Vec<usize>,
    pub indices: Vec<usize>,
    pub weights: Vec<f64>,
    pub fallback: Vec<bool>,
}

/// Weights x land fraction, renormalised per catchment; nearest land cells
/// (land fraction >= 0.5, growing square rings in index space, up to 12
/// cells) when the footprint has no land.
pub fn land_weights(w: &Weights, land: Option<&[f64]>) -> Result<LandWeights> {
    land_weights_cover(w, land, false)
}

/// `partial`: a regional source (the NWM CONUS analysis) leaves catchments
/// outside its domain with no cell, or no land cell near; those get an empty
/// row (their state is NaN, "not covered") instead of an error.
pub fn land_weights_cover(w: &Weights, land: Option<&[f64]>, partial: bool) -> Result<LandWeights> {
    let nx = w.grid.nx();
    let ny = w.grid.ny();
    let c = w.catchment_ids.len();
    let rows: Vec<Result<(Vec<usize>, Vec<f64>, bool)>> = (0..c)
        .into_par_iter()
        .map(|k| {
            let (a, b) = (w.offsets[k] as usize, w.offsets[k + 1] as usize);
            let idx: Vec<usize> = w.indices[a..b].iter().map(|i| *i as usize).collect();
            if partial && (idx.is_empty() || w.coverage_fraction[k] < PARTIAL_MIN_COVERAGE) {
                return Ok((Vec::new(), Vec::new(), false));
            }
            let Some(land) = land else {
                return Ok((idx, w.weights[a..b].to_vec(), false));
            };
            let lw: Vec<f64> = idx.iter().zip(&w.weights[a..b]).map(|(i, x)| x * land[*i].max(0.0)).collect();
            let s: f64 = lw.iter().sum();
            if s > 1e-9 {
                return Ok((idx, lw.iter().map(|x| x / s).collect(), false));
            }
            // nearest land: start from the heaviest footprint cell
            let (best, _) = idx.iter().zip(&w.weights[a..b]).fold((idx[0], -1.0), |acc, (i, x)| if *x > acc.1 { (*i, *x) } else { acc });
            let (y0, x0) = ((best / nx) as i64, (best % nx) as i64);
            for r in 1..=12i64 {
                let mut found = Vec::new();
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dy.abs() != r && dx.abs() != r {
                            continue;
                        }
                        let (y, x) = (y0 + dy, x0 + dx);
                        if y < 0 || x < 0 || y >= ny as i64 || x >= nx as i64 {
                            continue;
                        }
                        let i = (y as usize) * nx + x as usize;
                        if land[i] >= 0.5 {
                            found.push(i);
                        }
                    }
                }
                if !found.is_empty() {
                    let n = found.len() as f64;
                    return Ok((found.clone(), vec![1.0 / n; found.len()], true));
                }
            }
            if partial {
                return Ok((Vec::new(), Vec::new(), false));
            }
            Err(format!("{}: no land cell within 12 cells", w.catchment_ids[k]).into())
        })
        .collect();
    let mut out = LandWeights { offsets: vec![0], indices: Vec::new(), weights: Vec::new(), fallback: Vec::new() };
    for r in rows {
        let (i, x, f) = r?;
        out.indices.extend(i);
        out.weights.extend(x);
        out.offsets.push(out.indices.len());
        out.fallback.push(f);
    }
    Ok(out)
}

impl LandWeights {
    /// Weighted mean of a cell field over catchment k, skipping NaN cells
    /// (renormalised); None when every cell is NaN.
    pub fn mean(&self, k: usize, f: &[f64]) -> Option<f64> {
        let (a, b) = (self.offsets[k], self.offsets[k + 1]);
        let mut s = 0.0;
        let mut ws = 0.0;
        for j in a..b {
            let v = f[self.indices[j]];
            if v.is_finite() {
                s += self.weights[j] * v;
                ws += self.weights[j];
            }
        }
        (ws > 0.0).then(|| s / ws)
    }
    pub fn catchments(&self) -> usize {
        self.offsets.len() - 1
    }
}

fn overlap(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    (a1.min(b1) - a0.max(b0)).max(0.0)
}

/// Depth-overlap average of source layers onto target bounds. The deepest
/// source layer is extended downward when the target goes deeper.
pub fn remap(src_bounds: &[(f64, f64)], v: &[f64], tgt: &[f64]) -> (Vec<f64>, bool) {
    let deepest = src_bounds.iter().map(|b| b.1).fold(0.0, f64::max);
    let mut extended = false;
    let out = tgt
        .windows(2)
        .map(|t| {
            let mut s = 0.0;
            let mut w = 0.0;
            for (k, b) in src_bounds.iter().enumerate() {
                let bot = if k + 1 == src_bounds.len() && t[1] > deepest { t[1] } else { b.1 };
                let o = overlap(b.0, bot, t[0], t[1]);
                s += o * v[k];
                w += o;
            }
            if t[1] > deepest {
                extended = true;
            }
            s / w
        })
        .collect();
    (out, extended)
}

#[derive(Serialize, Default, Clone, Copy)]
pub struct Stat {
    pub min: f64,
    pub mean: f64,
    pub max: f64,
}
pub fn stat(v: impl Iterator<Item = f64>) -> Stat {
    let (mut mn, mut mx, mut s, mut n) = (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize);
    for x in v {
        mn = mn.min(x);
        mx = mx.max(x);
        s += x;
        n += 1;
    }
    Stat { min: mn, mean: if n > 0 { s / n as f64 } else { f64::NAN }, max: mx }
}

pub struct CatchmentState {
    pub soil_moisture_source: Vec<f64>,
    pub soil_temperature_source: Vec<f64>,
    pub smc: Vec<f64>,
    pub sh2o: Vec<f64>,
    pub sice: Vec<f64>,
    pub stc: Vec<f64>,
    pub swe_m: f64,
    pub snow_depth_m: f64,
    pub snow_temperature_k: f64,
    pub cfe_soil_storage: f64,
    pub cfe_gw_storage: f64,
    pub soil_class: Option<i64>,
}

pub struct Mapped {
    pub states: Vec<CatchmentState>,
    pub counts: serde_json::Value,
}

fn class_of(code: f64, tr: &Transfer) -> (Option<&crate::table::SoilClass>, bool) {
    let key = format!("{}", code.round() as i64);
    match tr.classes.get(&key) {
        Some(c) => (Some(c), false),
        None => (tr.fallback_class.as_ref().and_then(|f| tr.classes.get(f)), true),
    }
}

/// A catchment a regional source covers less than half of keeps no state
/// from it (partial mode): its value would describe someone else's land.
pub const PARTIAL_MIN_COVERAGE: f64 = 0.5;

pub fn map_state(table: &Table, row: &Row, src: &Source, record: usize, w: &Weights, params: &[Params]) -> Result<Mapped> {
    map_state_cover(table, row, src, record, w, params, false)
}

/// `partial`: catchments the source does not cover get an all-NaN state and
/// are counted, instead of failing the whole map (see land_weights_cover).
pub fn map_state_cover(table: &Table, row: &Row, src: &Source, record: usize, w: &Weights, params: &[Params], partial: bool) -> Result<Mapped> {
    let st = &row.state;
    let land = match &st.land_fraction {
        Some(v) if v.get("variable").is_some() => Some(ncread::field(src, v["variable"].as_str().unwrap(), record)?),
        Some(v) if v.get("from_valid_mask").is_some() => {
            let f = ncread::field(src, v["from_valid_mask"].as_str().unwrap(), record)?;
            Some(f.iter().map(|x| if x.is_finite() { 1.0 } else { 0.0 }).collect())
        }
        _ => None,
    };
    let lw = land_weights_cover(w, land.as_deref(), partial)?;
    let transfer = crate::table::classes(table, row)?;
    let method = row.soil_moisture_transfer.as_ref().map(|t| t.method.as_str()).unwrap_or("volumetric");
    let classes = match &st.soil_class {
        Some(s) => Some(ncread::field(src, &s.variable, record)?),
        None => None,
    };
    let layer_scale = |k: usize| -> Result<f64> {
        let l = &st.soil_water.layers[k];
        match &st.soil_water.scale {
            serde_json::Value::Number(n) => Ok(n.as_f64().unwrap()),
            serde_json::Value::String(s) if s == "per_layer_depth_kg_m2" => Ok(1.0 / (1000.0 * (l.bottom_m - l.top_m))),
            s => Err(format!("soil_water scale {s} not understood").into()),
        }
    };
    // Per-cell soil quantity per source layer: relative wetness or volumetric water.
    let mut fallback_cells = 0usize;
    let mut cell_layers = Vec::new();
    for (k, l) in st.soil_water.layers.iter().enumerate() {
        let scale = layer_scale(k)?;
        let mut f = ncread::field(src, &l.variable, record)?;
        for x in f.iter_mut() {
            *x *= scale;
        }
        if method == "wetness_index" || method == "field_capacity_anchored" {
            let tr = transfer.ok_or("this soil moisture transfer needs a class table")?;
            let cls = classes.as_ref().ok_or("this soil moisture transfer needs the row's soil_class field")?;
            for (i, x) in f.iter_mut().enumerate() {
                if !x.is_finite() {
                    continue;
                }
                let (c, fell) = class_of(cls[i], tr);
                if k == 0 && fell && land.as_ref().map(|l| l[i] > 0.0).unwrap_or(true) {
                    fallback_cells += 1;
                }
                let c = c.ok_or("soil class table has no fallback class")?;
                *x = if method == "wetness_index" {
                    ((*x - c.theta_pwp) / (c.theta_sat - c.theta_pwp)).clamp(0.0, 1.0)
                } else {
                    // 0 at wilting point, 1 at field capacity, 2 at saturation
                    let cap = c.theta_cap.ok_or("field_capacity_anchored needs theta_cap per soil class")?;
                    if *x <= cap {
                        ((*x - c.theta_pwp) / (cap - c.theta_pwp)).clamp(0.0, 1.0)
                    } else {
                        1.0 + ((*x - cap) / (c.theta_sat - cap)).clamp(0.0, 1.0)
                    }
                };
            }
        }
        cell_layers.push(f);
    }
    let mut temp_layers = Vec::new();
    for l in &st.soil_temperature.layers {
        temp_layers.push(ncread::field(src, &l.variable, record)?);
    }
    let swe = ncread::field(src, &st.snow_water_equivalent.variable, record)?
        .into_iter()
        .map(|x| x * st.snow_water_equivalent.scale)
        .collect::<Vec<_>>();
    let depth_cells: Vec<f64> = match (&st.snow_depth, &st.snow_density) {
        (Some(d), _) => ncread::field(src, &d.variable, record)?.into_iter().map(|x| x * d.scale).collect(),
        (None, Some(rho)) => {
            let r = ncread::field(src, &rho.variable, record)?;
            swe.iter().zip(r).map(|(s, r)| if *s > 0.0 && r > 0.0 { s * 1000.0 / r } else { 0.0 }).collect()
        }
        _ => vec![0.0; swe.len()],
    };
    let tsn = match &st.snow_temperature {
        Some(t) => Some(ncread::field(src, &t.variable, record)?),
        None => None,
    };
    let src_bounds: Vec<(f64, f64)> = st.soil_water.layers.iter().map(|l| (l.top_m, l.bottom_m)).collect();
    let t_bounds: Vec<(f64, f64)> = st.soil_temperature.layers.iter().map(|l| (l.top_m, l.bottom_m)).collect();
    let tgt = &table.target.noah_soil_layer_bounds_m;
    let (tfrz, hfus, grav) = (table.target.freezing_point_k, table.target.latent_heat_fusion_j_kg, table.target.gravity_m_s2);
    let results: Vec<Result<(CatchmentState, bool, bool)>> = (0..lw.catchments())
        .into_par_iter()
        .map(|k| {
            let p = &params[k];
            let id = &w.catchment_ids[k];
            if partial && (lw.offsets[k] == lw.offsets[k + 1] || lw.mean(k, &cell_layers[0]).is_none()) {
                let nan = vec![f64::NAN; tgt.len() - 1];
                return Ok((CatchmentState {
                    soil_moisture_source: vec![f64::NAN; cell_layers.len()],
                    soil_temperature_source: vec![f64::NAN; temp_layers.len()],
                    smc: nan.clone(), sh2o: nan.clone(), sice: nan.clone(), stc: nan,
                    swe_m: f64::NAN, snow_depth_m: f64::NAN, snow_temperature_k: f64::NAN,
                    cfe_soil_storage: f64::NAN, cfe_gw_storage: f64::NAN, soil_class: None,
                }, false, false));
            }
            let mut src_v = Vec::new();
            for f in &cell_layers {
                src_v.push(lw.mean(k, f).ok_or_else(|| format!("{id}: soil layer all missing"))?);
            }
            let mut src_t = Vec::new();
            for f in &temp_layers {
                src_t.push(lw.mean(k, f).ok_or_else(|| format!("{id}: soil temperature all missing"))?);
            }
            let (mapped, ext) = remap(&src_bounds, &src_v, tgt);
            let (stc, _) = remap(&t_bounds, &src_t, tgt);
            let fc = crate::cfe::field_capacity_m3m3(p);
            let smc: Vec<f64> = mapped
                .iter()
                .map(|v| match method {
                    "wetness_index" => p.wltsmc + v * (p.smcmax - p.wltsmc),
                    "field_capacity_anchored" if *v <= 1.0 => p.wltsmc + v * (fc - p.wltsmc),
                    "field_capacity_anchored" => fc + (v - 1.0) * (p.smcmax - fc),
                    _ => v.clamp(0.0, p.smcmax),
                })
                .collect();
            let mut sh2o = Vec::new();
            let mut sice = Vec::new();
            for (m, t) in smc.iter().zip(&stc) {
                if *t < tfrz {
                    let smp = hfus * (tfrz - t) / (grav * t);
                    let supercool = p.smcmax * (smp / p.satpsi).powf(-1.0 / p.bb);
                    let liq = m.min(supercool.max(0.0));
                    sh2o.push(liq);
                    sice.push(m - liq);
                } else {
                    sh2o.push(*m);
                    sice.push(0.0);
                }
            }
            // CFE soil reservoir spans 0..depth: mean total water over that depth
            let mut s = 0.0;
            for (j, t) in tgt.windows(2).enumerate() {
                s += overlap(t[0], t[1], 0.0, p.depth) * smc[j];
            }
            let covered = overlap(tgt[0], tgt[tgt.len() - 1], 0.0, p.depth);
            if covered + 1e-12 < p.depth {
                return Err(format!("{id}: CFE soil depth {} m is deeper than the Noah layers", p.depth).into());
            }
            let cfe_soil = (s / p.depth / p.smcmax).clamp(0.0, 1.0);
            let swe_m = lw.mean(k, &swe).unwrap_or(0.0).max(0.0);
            let snow_depth = lw.mean(k, &depth_cells).unwrap_or(0.0).max(0.0);
            // Snow cannot be warmer than its melting point; a source stored as
            // f32 holds 273.16 K as 273.1605 K, so values above are capped and counted.
            let (snow_t, capped) = match &tsn {
                Some(f) if swe_m > 1e-6 => {
                    let masked: Vec<f64> = f.iter().zip(&swe).map(|(t, s)| if *s > 1e-6 { *t } else { f64::NAN }).collect();
                    let t = lw.mean(k, &masked).unwrap_or(stc[0].min(tfrz));
                    (t.min(tfrz), t > tfrz)
                }
                _ => (stc[0].min(tfrz), false),
            };
            let soil_class = classes.as_ref().map(|c| {
                let (a, b) = (lw.offsets[k], lw.offsets[k + 1]);
                let mut acc: std::collections::BTreeMap<i64, f64> = Default::default();
                for j in a..b {
                    *acc.entry(c[lw.indices[j]].round() as i64).or_default() += lw.weights[j];
                }
                acc.into_iter().fold((0, -1.0), |m, (k, v)| if v > m.1 { (k, v) } else { m }).0
            });
            Ok((
                CatchmentState {
                    soil_moisture_source: src_v,
                    soil_temperature_source: src_t,
                    smc,
                    sh2o,
                    sice,
                    stc,
                    swe_m,
                    snow_depth_m: snow_depth,
                    snow_temperature_k: snow_t,
                    cfe_soil_storage: cfe_soil,
                    cfe_gw_storage: p.gw_storage,
                    soil_class,
                },
                ext,
                capped,
            ))
        })
        .collect();
    let mut states = Vec::with_capacity(results.len());
    let mut extended = 0;
    let mut capped_n = 0;
    for r in results {
        let (s, e, c) = r?;
        extended += usize::from(e);
        capped_n += usize::from(c);
        states.push(s);
    }
    let fb = lw.fallback.iter().filter(|x| **x).count();
    let not_covered = states.iter().filter(|s| !s.cfe_soil_storage.is_finite()).count();
    Ok(Mapped {
        states,
        counts: serde_json::json!({
            "catchments": lw.catchments(),
            "catchments_not_covered_by_source": if partial { serde_json::json!(not_covered) } else { serde_json::Value::Null },
            "catchments_without_land_in_footprint_using_nearest_land": fb,
            "land_cells_with_unknown_soil_class_using_fallback": fallback_cells,
            "catchments_with_target_layers_below_source_extended": extended,
            "catchments_snow_temperature_capped_at_freezing": capped_n,
            "soil_moisture_transfer": method,
        }),
    })
}
