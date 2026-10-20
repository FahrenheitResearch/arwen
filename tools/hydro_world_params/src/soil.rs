//! Soil texture -> soil class and hydraulic parameters, per soil-grid cell.
//!
//! - USDA texture class from sand/silt/clay: the USDA texture triangle
//!   definitions (Soil Science Division Staff, 2017, Soil Survey Manual, USDA
//!   Handbook 18, ch. 3, "Soil texture" class limits).
//! - STAS class = the Noah/Noah-MP SOILPARM.TBL "STAS" classes the NWM uses
//!   (1..12 the USDA classes, 13 organic, 14 water, 16 land ice). Organic
//!   material: organic carbon >= 120 g/kg (12 %), the lower organic-soil-material
//!   limit of Soil Survey Staff (2014), Keys to Soil Taxonomy, 12th ed.
//! - Cosby et al. (1984), Water Resour. Res. 20(6):682-690, Table 4
//!   multivariate regressions (the continuous alternative row).
use crate::grid::SoilLayer;
use crate::table::{SoilRecipe, SoilSource};
use rayon::prelude::*;
use std::collections::HashMap;

pub const USDA: [&str; 12] = [
    "sand",
    "loamy sand",
    "sandy loam",
    "silt loam",
    "silt",
    "loam",
    "sandy clay loam",
    "silty clay loam",
    "clay loam",
    "sandy clay",
    "silty clay",
    "clay",
];

/// USDA texture class index into `USDA` from percent sand/silt/clay
/// (renormalised to sum 100). Limits as in the Soil Survey Manual triangle.
pub fn usda_class(sand: f64, silt: f64, clay: f64) -> usize {
    let t = sand + silt + clay;
    let (s, si, c) = if t > 0.0 { (100.0 * sand / t, 100.0 * silt / t, 100.0 * clay / t) } else { (0.0, 0.0, 0.0) };
    if si + 1.5 * c < 15.0 {
        0
    } else if si + 2.0 * c < 30.0 {
        1
    } else if c >= 40.0 {
        if si >= 40.0 {
            10
        } else if s > 45.0 {
            9
        } else {
            11
        }
    } else if c >= 35.0 && s > 45.0 {
        9
    } else if c >= 27.0 {
        if s <= 20.0 {
            7
        } else if s <= 45.0 {
            8
        } else {
            6
        }
    } else if c >= 20.0 && si < 28.0 && s > 45.0 {
        6
    } else if si >= 80.0 && c < 12.0 {
        4
    } else if si >= 50.0 {
        3
    } else if c >= 7.0 && si >= 28.0 && s <= 52.0 {
        5
    } else {
        2
    }
}

/// Per soil cell results for one domain.
pub struct SoilCells {
    /// STAS class over the recipe's texture depth range (0 = no data)
    pub class: Vec<u8>,
    /// USDA class index + 1 for each source depth layer (0 = no data), layer-major
    pub layer_usda: Vec<Vec<u8>>,
    /// Cosby (1984) values over the texture depth: b, ln satdk (m/s), ln satpsi (m), smcmax, wltsmc
    pub cosby: Option<Vec<[f32; 5]>>,
    pub w: usize,
    pub h: usize,
    /// cell took its values from neighbours (SoilGrids gap: built-up land, water)
    pub filled: Vec<bool>,
}

fn overlap(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[1].min(b[1]) - a[0].max(b[0])).max(0.0)
}

pub fn classify(src: &SoilSource, recipe: &SoilRecipe, layers: &HashMap<String, SoilLayer>) -> SoilCells {
    let any = layers.values().next().unwrap();
    let (w, h) = (any.w, any.h);
    let nd = src.depth_labels.len();
    let get = |prop: &str, k: usize| &layers[&format!("{prop}/{}", src.depth_labels[k])].v;
    let weights: Vec<f64> = src.depths_cm.iter().map(|d| overlap(*d, recipe.texture_depth_cm)).collect();
    let usda_to: Vec<i64> = USDA.iter().map(|u| recipe.usda_to_class[*u]).collect();
    let cos = recipe.cosby.clone();
    let mut class = vec![0u8; w * h];
    let mut cosby: Option<Vec<[f32; 5]>> = cos.as_ref().map(|_| vec![[f32::NAN; 5]; w * h]);
    let sand: Vec<&Vec<f32>> = (0..nd).map(|k| get(&src.sand, k)).collect();
    let silt: Vec<&Vec<f32>> = (0..nd).map(|k| get(&src.silt, k)).collect();
    let clay: Vec<&Vec<f32>> = (0..nd).map(|k| get(&src.clay, k)).collect();
    let soc: Vec<&Vec<f32>> = (0..nd).map(|k| get(&src.organic_carbon, k)).collect();
    let mut layer_usda = vec![vec![0u8; w * h]; nd];
    // per-layer USDA classes (LGAR)
    layer_usda.par_iter_mut().enumerate().for_each(|(k, out)| {
        for i in 0..w * h {
            let (s, si, c) = (sand[k][i], silt[k][i], clay[k][i]);
            if s.is_finite() && si.is_finite() && c.is_finite() && s + si + c > 0.0 {
                out[i] = usda_class(s as f64, si as f64, c as f64) as u8 + 1;
            }
        }
    });
    let comp = |i: usize| -> Option<(f64, f64, f64, f64)> {
        let (mut ws, mut s, mut si, mut c, mut oc) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for k in 0..nd {
            let wk = weights[k];
            if wk <= 0.0 {
                continue;
            }
            let (a, b, d, o) = (sand[k][i], silt[k][i], clay[k][i], soc[k][i]);
            if a.is_finite() && b.is_finite() && d.is_finite() && a + b + d > 0.0 {
                ws += wk;
                s += wk * a as f64;
                si += wk * b as f64;
                c += wk * d as f64;
                oc += wk * if o.is_finite() { o as f64 } else { 0.0 };
            }
        }
        if ws > 0.0 {
            let t = s + si + c;
            Some((100.0 * s / t, 100.0 * si / t, 100.0 * c / t, oc / ws))
        } else {
            None
        }
    };
    class.par_iter_mut().enumerate().for_each(|(i, out)| {
        if let Some((s, si, c, oc)) = comp(i) {
            *out = if oc >= recipe.organic_carbon_min_g_kg { recipe.organic_class as u8 } else { usda_to[usda_class(s, si, c)] as u8 };
        }
    });
    if let (Some(cv), Some(co)) = (cosby.as_mut(), cos.as_ref()) {
        cv.par_iter_mut().enumerate().for_each(|(i, out)| {
            if let Some((s, si, c, _)) = comp(i) {
                // Cosby et al. (1984) Table 4 (multivariate, sand/silt/clay in %):
                //   b = 3.10 + 0.157 C - 0.003 S
                //   log10 psi_s [cm] = 1.54 - 0.0095 S + 0.0063 Si
                //   log10 Ks [in/h] = -0.60 + 0.0126 S - 0.0064 C
                //   theta_s [%] = 50.5 - 0.142 S - 0.037 C
                let b = co["b0"] + co["b_clay"] * c + co["b_sand"] * s;
                let psi_m = 10f64.powf(co["psi0"] + co["psi_sand"] * s + co["psi_silt"] * si) / 100.0;
                let ks = 10f64.powf(co["ks0"] + co["ks_sand"] * s + co["ks_clay"] * c) * 0.0254 / 3600.0;
                let ts = (co["ts0"] + co["ts_sand"] * s + co["ts_clay"] * c) / 100.0;
                // wilting point at -150 m (15 bar) from the Campbell/Clapp-Hornberger
                // curve theta = theta_s (psi_s/psi)^(1/b), as Noah-MP builds WLTSMC
                let wlt = ts * (psi_m / co["psi_wilt_m"]).powf(1.0 / b);
                *out = [b as f32, ks.ln() as f32, psi_m.ln() as f32, ts as f32, wlt as f32];
            }
        });
    }
    let mut out = SoilCells { class, layer_usda, cosby, w, h, filled: vec![false; w * h] };
    fill_gaps(&mut out, recipe.gap_fill_cells);
    out
}

/// SoilGrids leaves built-up land and water without predictions. Land gaps take
/// the most common class of their valid 8-neighbours, pass after pass (one cell
/// of reach per pass, `max_passes` passes), so a city carries the soil of the
/// land around it. Water is set from land cover afterwards, so filled water
/// cells never reach a catchment as soil. Rayon over rows each pass.
pub fn fill_gaps(s: &mut SoilCells, max_passes: usize) {
    let (w, h) = (s.w, s.h);
    for _ in 0..max_passes {
        let class = &s.class;
        let upd: Vec<(usize, u8, Vec<u8>, Option<[f32; 5]>)> = (0..h)
            .into_par_iter()
            .flat_map_iter(|j| {
                let mut v = vec![];
                for i in 0..w {
                    let c = j * w + i;
                    if class[c] != 0 {
                        continue;
                    }
                    let mut hist = [0u8; 32];
                    let mut src = usize::MAX;
                    for (a, b) in [(-1isize, -1isize), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
                        let (y, x) = (j as isize + a, i as isize + b);
                        if y < 0 || x < 0 || y >= h as isize || x >= w as isize {
                            continue;
                        }
                        let n = y as usize * w + x as usize;
                        let k = class[n] as usize;
                        if k != 0 && k < 32 {
                            hist[k] += 1;
                            if src == usize::MAX || hist[k] > hist[class[src] as usize] {
                                src = n;
                            }
                        }
                    }
                    if src != usize::MAX {
                        let layers: Vec<u8> = s.layer_usda.iter().map(|l| l[src]).collect();
                        v.push((c, class[src], layers, s.cosby.as_ref().map(|cv| cv[src])));
                    }
                }
                v.into_iter()
            })
            .collect();
        if upd.is_empty() {
            break;
        }
        for (c, k, layers, cv) in upd {
            s.class[c] = k;
            for (l, &u) in layers.iter().enumerate() {
                if s.layer_usda[l][c] == 0 {
                    s.layer_usda[l][c] = u;
                }
            }
            if let (Some(dst), Some(v)) = (s.cosby.as_mut(), cv) {
                dst[c] = v;
            }
            s.filled[c] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn usda_triangle_corners_and_centres() {
        // class centroids read off the USDA triangle
        let cases = [
            ((92.0, 5.0, 3.0), "sand"),
            ((82.0, 12.0, 6.0), "loamy sand"),
            ((65.0, 25.0, 10.0), "sandy loam"),
            ((20.0, 65.0, 15.0), "silt loam"),
            ((7.0, 88.0, 5.0), "silt"),
            ((40.0, 40.0, 20.0), "loam"),
            ((60.0, 13.0, 27.0), "sandy clay loam"),
            ((10.0, 56.0, 34.0), "silty clay loam"),
            ((32.0, 34.0, 34.0), "clay loam"),
            ((52.0, 6.0, 42.0), "sandy clay"),
            ((6.0, 47.0, 47.0), "silty clay"),
            ((20.0, 20.0, 60.0), "clay"),
        ];
        for ((s, si, c), want) in cases {
            assert_eq!(USDA[usda_class(s, si, c)], want, "{s} {si} {c}");
        }
    }
}
