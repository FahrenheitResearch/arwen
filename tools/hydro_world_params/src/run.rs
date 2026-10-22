//! One parameter derivation run: inputs -> per-catchment model_attributes rows
//! (contract `arwen.hydro-world.fabric.v1`), the LGAR side table, diagnostics
//! and a receipt with per-stage timings and their parallelism.
use crate::catch::{self, Catchments};
use crate::grid::{self, Grid};
use crate::soil::{self, USDA};
use crate::table::ParamTable;
use crate::terrain::{self, GiuhInputs};
use crate::Result;
use hydro_world::contract::{MetadataRow, ModelAttributes};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputRef {
    pub source: String,
    pub dir: String,
    #[serde(default)]
    pub prefix: String,
    /// Soil only: the WPS_GEOG soilgrids/soilcomp directory (SoilGrids-derived
    /// sand and clay on the 30" geography mirror every install has). Used when
    /// the soil source's own files are not there (the ISRIC WCS was down).
    #[serde(default)]
    pub geog_soilcomp: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CatchmentSpec {
    Gpkg { path: String, fabric_out: Option<String> },
    Shapefile { path: String, id_field: String, id_format: String, area_field: Option<String> },
    DemTessellation { stream_area_km2: f64, tile_cells: usize, id_namespace: u64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub domain: String,
    pub bbox: [f64; 4],
    pub cells_per_degree: usize,
    pub params_table: String,
    pub dem: InputRef,
    pub soil: InputRef,
    pub land_cover: InputRef,
    pub soil_recipe: String,
    pub lgar_recipe: String,
    /// "USGS" or "MODIFIED_IGBP_MODIS_NOAH" (the Noah-OWP MPTABLE sections)
    pub veg_class_table: String,
    /// CFE subsurface recipe row (`cfe_recipes`); unset = the table's `cfe_recipe_default`
    #[serde(default)]
    pub cfe_recipe: Option<String>,
    pub catchments: CatchmentSpec,
    pub out_dir: String,
}

struct Stages(Vec<Value>, Instant);
impl Stages {
    fn mark(&mut self, stage: &str, parallelism: &str, t0: Instant, extra: Value) {
        let s = t0.elapsed().as_secs_f64();
        eprintln!("[{:8.2}s] {stage}: {s:.3} s ({parallelism})", self.1.elapsed().as_secs_f64());
        self.0.push(json!({"stage": stage, "seconds": s, "parallelism": parallelism, "detail": extra}));
    }
}

#[derive(Serialize)]
struct Diag {
    divide_id: String,
    n_cells: usize,
    area_km2: f64,
    declared_km2: Option<f64>,
    centroid_cell_only: bool,
    soil_valid_fraction: f64,
    /// share of the catchment whose soil came from gap filling (SoilGrids has no prediction there)
    soil_filled_fraction: f64,
    soil_fallback: bool,
    veg_fallback: bool,
    land_cover_fractions: Vec<f64>,
    soil_class_fractions: Vec<f64>,
    giuh_t_mean_h: f64,
    giuh_t_max_h: f64,
    hillslope_length_m: f64,
    hillslope_sin_slope: f64,
    drainage_density_km_per_km2: f64,
    relief_m: f64,
    theta_fc: f64,
    /// the all-class area majority was water; the LSM classes are the majority land class
    water_majority: bool,
    /// no land pixel at all: grassland / loam stand-in for the LSM
    water_only: bool,
    outlet_lon: f64,
    outlet_lat: f64,
}

#[derive(Serialize)]
struct LgarRow {
    divide_id: String,
    layer_top_cm: Vec<f64>,
    layer_bottom_cm: Vec<f64>,
    /// 1-based row of the LGAR table (USDA class) dominant in each layer
    layer_soil_type: Vec<i64>,
    layer_soil_name: Vec<String>,
    theta_r: Vec<f64>,
    theta_e: Vec<f64>,
    alpha_per_cm: Vec<f64>,
    n: Vec<f64>,
    ks_cm_h: Vec<f64>,
}

fn now_utc() -> String {
    let s = std::process::Command::new("date").args(["-u", "+%Y-%m-%dT%H:%M:%SZ"]).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    s.unwrap_or_default()
}

fn sha256_file(p: &str) -> String {
    use sha2::{Digest, Sha256};
    match std::fs::read(p) {
        Ok(b) => format!("{:x}", Sha256::digest(&b)),
        Err(_) => "unreadable".into(),
    }
}

pub fn run(cfg_path: &str) -> Result<Value> {
    let t_all = Instant::now();
    let cfg: RunConfig = serde_json::from_str(&std::fs::read_to_string(cfg_path)?)?;
    let mut st = Stages(vec![], t_all);
    let threads = rayon::current_num_threads();
    let par = format!("Rayon, {threads} threads");

    let t0 = Instant::now();
    let tab = ParamTable::load(&cfg.params_table)?;
    let dem_src = tab.dem_sources.get(&cfg.dem.source).ok_or("unknown dem source row")?.clone();
    let soil_src = tab.soil_sources.get(&cfg.soil.source).ok_or("unknown soil source row")?.clone();
    let lc_src = tab.land_cover_sources.get(&cfg.land_cover.source).ok_or("unknown land cover source row")?.clone();
    let recipe = tab.soil_recipes.get(&cfg.soil_recipe).ok_or("unknown soil recipe row")?.clone();
    let lgar = tab.lgar_recipes.get(&cfg.lgar_recipe).ok_or("unknown lgar recipe row")?.clone();
    let cfe_recipe_name = cfg.cfe_recipe.clone().or_else(|| tab.cfe_recipe_default.clone()).unwrap_or_else(|| "constants".into());
    let cfe_recipe = match tab.cfe_recipes.get(&cfe_recipe_name) {
        Some(r) => r.clone(),
        None if cfe_recipe_name == "constants" => crate::table::CfeRecipe {
            method: "constants".into(),
            source: "constants rows".into(),
            lateral_ksat_factor: 0.0,
            lateral_ksat_factor_source: String::new(),
            n_nash: 0.0,
            atmosphere_head_m: 0.0,
            alpha_fc_by_soil_class: Default::default(),
            alpha_fc_default: 0.0,
            alpha_fc_source: String::new(),
            min_hillslope_length_m: 0.0,
            min_sin_slope: 0.0,
            min_drainable_porosity: 0.0,
            k_lf_range: [0.0, 0.0],
            k_nash_range: [0.0, 0.0],
        },
        None => return Err(format!("unknown cfe recipe row {cfe_recipe_name}").into()),
    };
    let g = Grid::new(cfg.bbox, cfg.cells_per_degree)?;
    if lc_src.classes.iter().any(|c| !c.veg.contains_key(&cfg.veg_class_table)) {
        return Err(format!("land cover row {} has no {} vegetation mapping", cfg.land_cover.source, cfg.veg_class_table).into());
    }
    std::fs::create_dir_all(&cfg.out_dir)?;
    st.mark("load table + grid", "1 thread (startup, < 1 s)", t0, json!({"nx": g.nx, "ny": g.ny, "cells": g.n()}));

    let t0 = Instant::now();
    let (dem, dem_used, dem_missing) = grid::mosaic_dem(&g, &dem_src, &cfg.dem.dir)?;
    let dem_valid = dem.par_iter().filter(|v| v.is_finite()).count();
    st.mark("dem mosaic", &par, t0, json!({"tiles": dem_used.len(), "missing_tiles": dem_missing.len(), "valid_cells": dem_valid}));

    let t0 = Instant::now();
    let (lc_counts, lc_used, lc_missing) = grid::accumulate_land_cover(&g, &lc_src, &cfg.land_cover.dir)?;
    let ncl = lc_src.classes.len();
    st.mark("land cover pixel counts", &par, t0, json!({"tiles": lc_used.len(), "missing_tiles": lc_missing.len()}));

    let t0 = Instant::now();
    // geog_soilcomp set: acquisition chose our own WPS_GEOG mirror (primary
    // when it covers the box, or the fallback when SoilGrids was out of reach).
    let geog = !cfg.soil.geog_soilcomp.is_empty();
    let (soil_src, soil_layers, soil_paths) = if geog {
        let (src, layers) = grid::load_geog_soilcomp(&soil_src, &cfg.soil.geog_soilcomp, cfg.bbox)?;
        (src, layers, vec![cfg.soil.geog_soilcomp.clone()])
    } else {
        let (layers, paths) = grid::load_soil(&soil_src, &cfg.soil.dir, &cfg.soil.prefix)?;
        (soil_src, layers, paths)
    };
    let sc = soil::classify(&soil_src, &recipe, &soil_layers);
    let soil_grid = soil_layers.values().next().map(|l| (l.ox, l.oy, l.dx, l.dy, l.w, l.h)).unwrap();
    let soil_valid = sc.class.par_iter().filter(|&&c| c != 0).count();
    st.mark("soil load + texture classes", &par, t0, json!({"files": soil_paths.len(), "soil_cells": sc.w * sc.h, "valid": soil_valid,
        "source": if geog { "WPS_GEOG soilgrids/soilcomp (our geography mirror: SoilGrids-derived sand and clay, 30 arc-seconds)" } else { soil_src.dataset.as_str() }}));

    let t0 = Instant::now();
    let (slope, aspect) = terrain::slope_aspect(&g, &dem);
    let tan_slope: Vec<f32> = slope.par_iter().map(|s| s.to_radians().tan()).collect();
    let tan_coarse = if tab.slop.method == "terrain_tan_coarse" { Some(terrain::coarse_tan_slope(&g, &dem, tab.slop.coarse_m)) } else { None };
    // dominant land-cover class per cell, hillslope velocity coefficient
    let grass = lc_src.classes.iter().position(|c| c.code == 30).unwrap_or(0);
    let dom: Vec<u8> = lc_counts
        .par_chunks(ncl)
        .map(|c| {
            let (mut b, mut bv) = (u8::MAX, 0u16);
            for (k, &v) in c.iter().enumerate() {
                if v > bv {
                    bv = v;
                    b = k as u8;
                }
            }
            b
        })
        .collect();
    let k_hill: Vec<f32> = dom
        .par_iter()
        .map(|&d| {
            let k = if d == u8::MAX { grass } else { d as usize };
            (lc_src.classes[k].neh_k_ft_s * 0.3048) as f32
        })
        .collect();
    st.mark("slope/aspect (Horn) + coarse slope + dominant cover", &par, t0, json!({}));

    let t0 = Instant::now();
    let mut fabric = None;
    let mut fabric_out = None;
    let cat: Catchments = match &cfg.catchments {
        CatchmentSpec::Shapefile { path, id_field, id_format, area_field } => catch::from_shapefile(&g, path, id_field, id_format, area_field.as_deref())?,
        CatchmentSpec::Gpkg { path, fabric_out: fo } => {
            let (c, f) = catch::from_gpkg(&g, path)?;
            fabric = Some(f);
            fabric_out = fo.clone();
            c
        }
        CatchmentSpec::DemTessellation { stream_area_km2, tile_cells, id_namespace } => {
            let water: Vec<bool> = lc_src.classes.iter().map(|c| c.water).collect();
            // open sea: no DEM tile, or the 0 m the Copernicus DEM writes over sea where land cover is water or absent
            let outside: Vec<bool> = dem
                .par_iter()
                .zip(dom.par_iter())
                .map(|(&e, &d)| !e.is_finite() || (e <= 0.0 && (d == u8::MAX || water[d as usize])))
                .collect();
            catch::dem_tessellation(&g, &dem, &outside, *tile_cells, *stream_area_km2, *id_namespace)
        }
    };
    let ncat = cat.ids.len();
    st.mark("catchments onto grid", &par, t0, json!({"kind": cat.kind, "catchments": ncat}));

    // CSR cell lists per catchment, built in parallel row chunks
    let t0 = Instant::now();
    let nchunks = (threads * 4).max(1);
    let rows_per = g.ny.div_ceil(nchunks);
    let chunk_counts: Vec<Vec<u32>> = (0..nchunks)
        .into_par_iter()
        .map(|c| {
            let mut v = vec![0u32; ncat];
            for j in c * rows_per..((c + 1) * rows_per).min(g.ny) {
                for &l in &cat.labels[j * g.nx..(j + 1) * g.nx] {
                    if l != 0 {
                        v[l as usize - 1] += 1;
                    }
                }
            }
            v
        })
        .collect();
    let mut start = vec![0usize; ncat + 1];
    for k in 0..ncat {
        start[k + 1] = start[k] + chunk_counts.iter().map(|v| v[k] as usize).sum::<usize>();
    }
    let mut cells = vec![0u32; start[ncat]];
    {
        // per chunk write cursor per catchment
        let mut cursors: Vec<Vec<usize>> = Vec::with_capacity(nchunks);
        let mut run = start[..ncat].to_vec();
        for c in 0..nchunks {
            cursors.push(run.clone());
            for k in 0..ncat {
                run[k] += chunk_counts[c][k] as usize;
            }
        }
        let ptr = cells.as_mut_ptr() as usize;
        cursors.into_par_iter().enumerate().for_each(|(c, mut cur)| {
            let p = ptr as *mut u32;
            for j in c * rows_per..((c + 1) * rows_per).min(g.ny) {
                for i in 0..g.nx {
                    let l = cat.labels[j * g.nx + i];
                    if l != 0 {
                        let k = l as usize - 1;
                        // SAFETY: every (chunk, catchment) cursor range is disjoint
                        unsafe { *p.add(cur[k]) = (j * g.nx + i) as u32 };
                        cur[k] += 1;
                    }
                }
            }
        });
    }
    st.mark("catchment cell lists (CSR)", &par, t0, json!({"labelled_cells": start[ncat]}));

    // per catchment parameters
    let t0 = Instant::now();
    let giuh_in = GiuhInputs {
        g: &g,
        dem: &dem,
        tan_slope: &tan_slope,
        k_hill: &k_hill,
        k_channel_m_s: tab.giuh.channel_neh_k_ft_s * 0.3048,
        channel_area_m2: tab.giuh.channel_init_area_km2 * 1e6,
        slope_floor: tab.giuh.slope_floor,
        dt_s: tab.giuh.dt_s,
        max_ordinates: tab.giuh.max_ordinates,
    };
    let (sox, soy, sdx, sdy, sw, sh) = soil_grid;
    let soil_idx = |c: usize| -> Option<usize> {
        let lon = g.lon(c % g.nx);
        let lat = g.lat(c / g.nx);
        let ci = ((lon - sox) / sdx).floor();
        let cj = ((soy - lat) / sdy).floor();
        if ci >= 0.0 && cj >= 0.0 && (ci as usize) < sw && (cj as usize) < sh {
            Some(cj as usize * sw + ci as usize)
        } else {
            None
        }
    };
    let max_class = recipe.classes.iter().map(|c| c.id).max().unwrap_or(19) as usize;
    let class_row: Vec<Option<&crate::table::SoilClassRow>> = (0..=max_class).map(|id| recipe.classes.iter().find(|c| c.id as usize == id)).collect();
    let water_k: Vec<bool> = lc_src.classes.iter().map(|c| c.water).collect();
    let ice_k: Vec<bool> = lc_src.classes.iter().map(|c| c.ice).collect();
    let nd = soil_src.depth_labels.len();
    let consts = |n: &str| tab.c(n);
    let fallback_class = recipe.usda_to_class["loam"] as usize;
    let results: Vec<(ModelAttributes, Diag, LgarRow)> = (0..ncat)
        .into_par_iter()
        .map(|k| {
            let mut list: Vec<u32> = cells[start[k]..start[k + 1]].to_vec();
            let mut centroid_only = false;
            if list.is_empty() {
                // too small for a cell centre: the cell under the centroid
                if let Some((lon, lat)) = cat.centroid[k] {
                    let i = ((lon - g.west) * g.cells_per_degree as f64).floor();
                    let j = ((g.north - lat) * g.cells_per_degree as f64).floor();
                    if i >= 0.0 && j >= 0.0 && (i as usize) < g.nx && (j as usize) < g.ny {
                        list.push((j as usize * g.nx + i as usize) as u32);
                        centroid_only = true;
                    }
                }
            }
            let (mut a_tot, mut a_e, mut e_sum, mut a_s, mut s_sum, mut asin, mut acos, mut lon_s, mut lat_s) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut a_tc, mut tc_sum) = (0.0, 0.0);
            let mut lc = vec![0u64; ncl];
            let mut shist = vec![0f64; max_class + 1];
            let mut a_soil = 0.0;
            let mut a_filled = 0.0;
            let mut lhist = vec![vec![0f64; 12]; nd];
            let mut cos_acc = [0f64; 6];
            for &c in &list {
                let c = c as usize;
                let j = c / g.nx;
                let a = g.cell_area_m2(j);
                a_tot += a;
                lon_s += a * g.lon(c % g.nx);
                lat_s += a * g.lat(j);
                if dem[c].is_finite() {
                    a_e += a;
                    e_sum += a * dem[c] as f64;
                }
                if slope[c].is_finite() {
                    a_s += a;
                    s_sum += a * slope[c] as f64;
                }
                if aspect[c].is_finite() {
                    let r = (aspect[c] as f64).to_radians();
                    asin += a * r.sin();
                    acos += a * r.cos();
                }
                if let Some(tc) = &tan_coarse {
                    if tc[c].is_finite() {
                        a_tc += a;
                        tc_sum += a * tc[c] as f64;
                    }
                }
                for q in 0..ncl {
                    lc[q] += lc_counts[c * ncl + q] as u64;
                }
                let si = soil_idx(c);
                // water and ice from land cover first; land from SoilGrids (gap-filled)
                let d = dom[c];
                let cl = if d != u8::MAX && water_k[d as usize] {
                    recipe.water_class as usize
                } else if d != u8::MAX && ice_k[d as usize] {
                    recipe.ice_class as usize
                } else {
                    si.map(|s| sc.class[s] as usize).unwrap_or(0)
                };
                if cl != 0 {
                    shist[cl] += a;
                    a_soil += a;
                    if si.map(|s| sc.filled[s]).unwrap_or(false) && cl != recipe.water_class as usize && cl != recipe.ice_class as usize {
                        a_filled += a;
                    }
                }
                if let Some(s) = si {
                    for (l, h) in lhist.iter_mut().enumerate() {
                        let u = sc.layer_usda[l][s];
                        if u != 0 {
                            h[u as usize - 1] += a;
                        }
                    }
                    if let Some(cv) = &sc.cosby {
                        let v = cv[s];
                        if v[0].is_finite() {
                            cos_acc[0] += a;
                            for q in 0..5 {
                                cos_acc[q + 1] += a * v[q] as f64;
                            }
                        }
                    }
                }
            }
            let id = cat.ids[k].clone();
            let (clon, clat) = cat.centroid[k].unwrap_or((lon_s / a_tot.max(1e-300), lat_s / a_tot.max(1e-300)));
            // land cover
            let lc_tot: u64 = lc.iter().sum();
            // Water-majority rule (coordinator ruling 2026-10-18T11:50Z): the LSM takes the
            // area-majority LAND class; open water cannot run Noah-OWP (water NROOT 0). The
            // all-class majority is kept only to count the catchments the rule changes.
            let (vk_all, vk) = majority_classes(&lc, &water_k);
            let water_majority = vk_all != usize::MAX && water_k[vk_all];
            let water_only = vk == usize::MAX && lc_tot > 0;
            let veg_fallback = vk == usize::MAX;
            let vrow = &lc_src.classes[if veg_fallback { grass } else { vk }];
            let veg_class = match vrow.veg_tropical.get(&cfg.veg_class_table) {
                Some(&t) if clat.abs() < 23.5 => t,
                _ => vrow.veg[&cfg.veg_class_table],
            };
            let impervious = if lc_tot > 0 { lc.iter().zip(&lc_src.classes).map(|(&n, c)| n as f64 * c.impervious).sum::<f64>() / lc_tot as f64 } else { 0.0 };
            // soil class (Noah isltyp): mode over every class, water and ice included
            // land classes only (water-majority rule); the water class never reaches the LSM
            let mut soil_class = 0usize;
            for c in 1..=max_class {
                if c != recipe.water_class as usize && shist[c] > shist[soil_class] {
                    soil_class = c;
                }
            }
            // CFE hydraulics over hydraulic classes only
            let hyd: Vec<(usize, f64)> = (1..=max_class).filter(|&c| shist[c] > 0.0 && class_row[c].map(|r| r.hydraulic).unwrap_or(false)).map(|c| (c, shist[c])).collect();
            let soil_fallback = hyd.is_empty() && (recipe.method == "class_lookup" || cos_acc[0] == 0.0);
            let hyd = if hyd.is_empty() { vec![(fallback_class, 1.0)] } else { hyd };
            if soil_class == 0 {
                soil_class = hyd[0].0;
            }
            let wsum: f64 = hyd.iter().map(|x| x.1).sum();
            let row = |c: usize| class_row[c].unwrap();
            let mode_c = hyd.iter().fold((0usize, -1.0), |m, &(c, w)| if w > m.1 { (c, w) } else { m }).0;
            let geo = |f: &dyn Fn(&crate::table::SoilClassRow) -> f64| (hyd.iter().map(|&(c, w)| w * f(row(c)).ln()).sum::<f64>() / wsum).exp();
            let mean = |f: &dyn Fn(&crate::table::SoilClassRow) -> f64| hyd.iter().map(|&(c, w)| w * f(row(c))).sum::<f64>() / wsum;
            let (b, satdk, satpsi, smcmax, wltsmc) = if recipe.method == "cosby1984" && cos_acc[0] > 0.0 {
                let a = cos_acc[0];
                (cos_acc[1] / a, (cos_acc[2] / a).exp(), (cos_acc[3] / a).exp(), cos_acc[4] / a, cos_acc[5] / a)
            } else {
                (row(mode_c).bb, geo(&|r| r.satdk), geo(&|r| r.satpsi), mean(&|r| r.maxsmc), mean(&|r| r.wltsmc))
            };
            let slop = match tab.slop.method.as_str() {
                "terrain_tan_coarse" if a_tc > 0.0 => (tc_sum / a_tc).clamp(tab.slop.min, tab.slop.max),
                // method "constant" (min == max), or no DEM under the catchment
                _ => tab.slop.min,
            };
            let gi = terrain::giuh(&giuh_in, &list);
            // CFE subsurface parameters (recipe row cfe_recipe)
            let depth = consts("soil_depth_m");
            let alpha_fc = if cfe_recipe.method == "hillslope_kinematic" {
                *cfe_recipe.alpha_fc_by_soil_class.get(&mode_c.to_string()).unwrap_or(&cfe_recipe.alpha_fc_default)
            } else {
                consts("alpha_fc")
            };
            let theta_fc = if cfe_recipe.method == "hillslope_kinematic" {
                field_capacity_theta(&cfe_recipe, alpha_fc, smcmax, wltsmc, satpsi, b)
            } else {
                f64::NAN
            };
            let (k_lf, k_nash) = if cfe_recipe.method == "hillslope_kinematic" {
                hillslope_kinematic_k(&cfe_recipe, satdk, gi.hillslope_length_m, gi.hillslope_sin_slope, depth, smcmax, theta_fc)
            } else {
                (consts("k_lf"), consts("k_nash"))
            };
            let aspect_deg = {
                let a = asin.atan2(acos).to_degrees();
                if asin == 0.0 && acos == 0.0 { 0.0 } else if a < 0.0 { a + 360.0 } else { a }
            };
            let ma = ModelAttributes {
                fid: 0,
                divide_id: Some(id.clone()),
                elevation_mean_m: Some(if a_e > 0.0 { e_sum / a_e } else { 0.0 }),
                slope_mean_deg: Some(if a_s > 0.0 { s_sum / a_s } else { 0.0 }),
                aspect_mean_deg: Some(aspect_deg),
                centroid_lat_deg: Some(clat),
                centroid_lon_deg: Some(clon),
                soil_class: Some(soil_class as i64),
                veg_class: Some(veg_class),
                soil_color: Some(consts("soil_color") as i64),
                impervious_fraction: Some(impervious),
                soil_depth_m: Some(consts("soil_depth_m")),
                soil_b: Some(b),
                soil_satdk_m_s: Some(satdk),
                soil_satpsi_m: Some(satpsi),
                soil_slop: Some(slop),
                soil_smcmax: Some(smcmax),
                soil_wltsmc: Some(wltsmc),
                refkdt: Some(consts("refkdt")),
                alpha_fc: Some(alpha_fc),
                max_gw_storage_m: Some(consts("max_gw_storage_m")),
                // NWM bucket Coeff (m3/s per catchment) as the CFE depth rate (m/h) over the catchment area
                cgw_m_h: Some(consts("gw_coeff_m3_s") * 3600.0 / (cat.declared_km2[k].filter(|a| *a > 0.0).map(|a| a * 1e6).unwrap_or(a_tot))),
                gw_expon: Some(consts("gw_expon")),
                k_lf: Some(k_lf),
                k_nash: Some(k_nash),
                giuh_ordinates: Some(gi.ordinates.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",")),
            };
            // LGAR: dominant USDA class per source depth layer, its van Genuchten row
            let mut lr = LgarRow {
                divide_id: id.clone(),
                layer_top_cm: soil_src.depths_cm.iter().map(|d| d[0]).collect(),
                layer_bottom_cm: soil_src.depths_cm.iter().map(|d| d[1]).collect(),
                layer_soil_type: vec![],
                layer_soil_name: vec![],
                theta_r: vec![],
                theta_e: vec![],
                alpha_per_cm: vec![],
                n: vec![],
                ks_cm_h: vec![],
            };
            let lfall = USDA.iter().position(|u| *u == "loam").unwrap();
            for h in &lhist {
                let mut best = lfall;
                let mut bv = 0.0;
                for (q, &v) in h.iter().enumerate() {
                    if v > bv {
                        bv = v;
                        best = q;
                    }
                }
                let vg = lgar.rows.iter().position(|r| r.usda == USDA[best]).unwrap();
                let r = &lgar.rows[vg];
                lr.layer_soil_type.push(vg as i64 + 1);
                lr.layer_soil_name.push(r.usda.clone());
                lr.theta_r.push(r.theta_r);
                lr.theta_e.push(r.theta_e);
                lr.alpha_per_cm.push(r.alpha_per_cm);
                lr.n.push(r.n);
                lr.ks_cm_h.push(r.ks_cm_h);
            }
            let ol = (g.lon(gi.outlet % g.nx), g.lat(gi.outlet / g.nx));
            let d = Diag {
                divide_id: id,
                n_cells: list.len(),
                area_km2: a_tot / 1e6,
                declared_km2: cat.declared_km2[k],
                centroid_cell_only: centroid_only,
                soil_valid_fraction: if a_tot > 0.0 { a_soil / a_tot } else { 0.0 },
                soil_filled_fraction: if a_tot > 0.0 { a_filled / a_tot } else { 0.0 },
                soil_fallback,
                veg_fallback,
                land_cover_fractions: lc.iter().map(|&v| if lc_tot > 0 { v as f64 / lc_tot as f64 } else { 0.0 }).collect(),
                soil_class_fractions: shist.iter().map(|&v| if a_soil > 0.0 { v / a_soil } else { 0.0 }).collect(),
                giuh_t_mean_h: gi.t_mean_s / 3600.0,
                giuh_t_max_h: gi.t_max_s / 3600.0,
                hillslope_length_m: gi.hillslope_length_m,
                hillslope_sin_slope: gi.hillslope_sin_slope,
                drainage_density_km_per_km2: if a_tot > 0.0 { (gi.channel_length_m / 1e3) / (a_tot / 1e6) } else { 0.0 },
                relief_m: gi.relief_m,
                theta_fc,
                water_majority,
                water_only,
                outlet_lon: ol.0,
                outlet_lat: ol.1,
            };
            (ma, d, lr)
        })
        .collect();
    st.mark("per-catchment parameters + GIUH", &par, t0, json!({"catchments": ncat}));

    // canonical order (contract section 6) and fid = rank + 1
    let t0 = Instant::now();
    let mut order: Vec<usize> = (0..ncat).collect();
    order.sort_by_key(|&k| (hydro_world::check::canonical_key(&cat.ids[k]), cat.ids[k].clone()));
    let mut rows: Vec<ModelAttributes> = Vec::with_capacity(ncat);
    let mut diags = Vec::with_capacity(ncat);
    let mut lgars = Vec::with_capacity(ncat);
    let mut results: Vec<Option<(ModelAttributes, Diag, LgarRow)>> = results.into_iter().map(Some).collect();
    for (rank, &k) in order.iter().enumerate() {
        let (mut m, d, l) = results[k].take().unwrap();
        m.fid = rank as i64 + 1;
        rows.push(m);
        diags.push(d);
        lgars.push(l);
    }
    // bounds: every value finite and inside the table's physical range
    let mut violations: Vec<String> = vec![];
    let cols: Vec<(&str, Box<dyn Fn(&ModelAttributes) -> f64 + Sync>)> = vec![
        ("elevation_mean_m", Box::new(|m| m.elevation_mean_m.unwrap())),
        ("slope_mean_deg", Box::new(|m| m.slope_mean_deg.unwrap())),
        ("aspect_mean_deg", Box::new(|m| m.aspect_mean_deg.unwrap())),
        ("centroid_lat_deg", Box::new(|m| m.centroid_lat_deg.unwrap())),
        ("centroid_lon_deg", Box::new(|m| m.centroid_lon_deg.unwrap())),
        ("soil_class", Box::new(|m| m.soil_class.unwrap() as f64)),
        ("veg_class", Box::new(|m| m.veg_class.unwrap() as f64)),
        ("soil_color", Box::new(|m| m.soil_color.unwrap() as f64)),
        ("impervious_fraction", Box::new(|m| m.impervious_fraction.unwrap())),
        ("soil_depth_m", Box::new(|m| m.soil_depth_m.unwrap())),
        ("soil_b", Box::new(|m| m.soil_b.unwrap())),
        ("soil_satdk_m_s", Box::new(|m| m.soil_satdk_m_s.unwrap())),
        ("soil_satpsi_m", Box::new(|m| m.soil_satpsi_m.unwrap())),
        ("soil_slop", Box::new(|m| m.soil_slop.unwrap())),
        ("soil_smcmax", Box::new(|m| m.soil_smcmax.unwrap())),
        ("soil_wltsmc", Box::new(|m| m.soil_wltsmc.unwrap())),
        ("refkdt", Box::new(|m| m.refkdt.unwrap())),
        ("alpha_fc", Box::new(|m| m.alpha_fc.unwrap())),
        ("max_gw_storage_m", Box::new(|m| m.max_gw_storage_m.unwrap())),
        ("cgw_m_h", Box::new(|m| m.cgw_m_h.unwrap())),
        ("gw_expon", Box::new(|m| m.gw_expon.unwrap())),
        ("k_lf", Box::new(|m| m.k_lf.unwrap())),
        ("k_nash", Box::new(|m| m.k_nash.unwrap())),
    ];
    let mut summary = serde_json::Map::new();
    for (name, f) in &cols {
        let mut v: Vec<f64> = rows.iter().map(|m| f(m)).collect();
        let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
        let [lo, hi] = *tab.bounds.get(*name).ok_or_else(|| format!("bounds.{name} missing"))?;
        let out = v.iter().filter(|x| !(**x >= lo && **x <= hi)).count();
        if nonfinite + out > 0 {
            violations.push(format!("{name}: {nonfinite} non-finite, {out} outside [{lo}, {hi}]"));
        }
        v.retain(|x| x.is_finite());
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| if v.is_empty() { f64::NAN } else { v[((v.len() - 1) as f64 * p).round() as usize] };
        summary.insert(name.to_string(), json!({"min": q(0.0), "p01": q(0.01), "p50": q(0.5), "p99": q(0.99), "max": q(1.0), "bounds": [lo, hi], "nonfinite": nonfinite, "outside": out}));
    }
    // GIUH: non-negative, sums to 1 (what hydro_world::derive::cfe_ini enforces)
    let mut giuh_bad = 0usize;
    let mut giuh_len = vec![];
    for m in &rows {
        let o: Vec<f64> = m.giuh_ordinates.as_ref().unwrap().split(',').map(|s| s.parse::<f64>().unwrap_or(f64::NAN)).collect();
        if o.iter().any(|x| !x.is_finite() || *x < 0.0) || (o.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
            giuh_bad += 1;
        }
        giuh_len.push(o.len());
    }
    if giuh_bad > 0 {
        violations.push(format!("giuh_ordinates: {giuh_bad} rows not non-negative summing to 1"));
    }
    giuh_len.sort();
    summary.insert("giuh_ordinate_count".into(), json!({"min": giuh_len.first(), "p50": giuh_len.get(giuh_len.len() / 2), "max": giuh_len.last()}));
    // wilting below porosity in every row
    let wl_bad = rows.iter().filter(|m| m.soil_wltsmc.unwrap() >= m.soil_smcmax.unwrap()).count();
    if wl_bad > 0 {
        violations.push(format!("soil_wltsmc >= soil_smcmax in {wl_bad} rows"));
    }
    // the engine packets accept every row
    let mut packet_err = 0usize;
    for m in &rows {
        let init = hydro_world::derive::CfeInitial { soil_storage: 0.5, gw_storage: 0.5 };
        if hydro_world::derive::cfe_ini(m, init).is_err() || hydro_world::derive::noahowp_overrides(m).is_err() {
            packet_err += 1;
        }
    }
    if packet_err > 0 {
        violations.push(format!("{packet_err} rows rejected by hydro_world::derive packets"));
    }
    let fallbacks = json!({
        "soil_fallback": diags.iter().filter(|d| d.soil_fallback).count(),
        "veg_fallback": diags.iter().filter(|d| d.veg_fallback).count(),
        "centroid_cell_only": diags.iter().filter(|d| d.centroid_cell_only).count(),
        "water_majority_reassigned_to_land_class": diags.iter().filter(|d| d.water_majority && !d.water_only).count(),
        "water_only_grassland_loam_stand_in": diags.iter().filter(|d| d.water_only).count(),
        "rule": "water-majority rule: Noah-OWP vegtyp/isltyp are the area-majority land class (coordinator ruling 2026-10-18T11:50Z)",
        "no_cell_at_all": diags.iter().filter(|d| d.n_cells == 0).count(),
    });
    st.mark("canonical order + bounds + packet check", &par, t0, json!({"violations": violations.len()}));

    // outputs
    let t0 = Instant::now();
    let od = &cfg.out_dir;
    std::fs::write(format!("{od}/model_attributes.json"), serde_json::to_vec(&rows)?)?;
    std::fs::write(format!("{od}/catchment_diagnostics.json"), serde_json::to_vec(&diags)?)?;
    std::fs::write(format!("{od}/lgar_attributes.json"), serde_json::to_vec(&lgars)?)?;
    let mut csv = String::from("divide_id,elevation_mean_m,slope_mean_deg,aspect_mean_deg,centroid_lat_deg,centroid_lon_deg,soil_class,veg_class,soil_color,impervious_fraction,soil_depth_m,soil_b,soil_satdk_m_s,soil_satpsi_m,soil_slop,soil_smcmax,soil_wltsmc,refkdt,alpha_fc,max_gw_storage_m,cgw_m_h,gw_expon,k_lf,k_nash,giuh_n,giuh_ordinates\n");
    for m in &rows {
        let gi = m.giuh_ordinates.as_ref().unwrap();
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},\"{}\"\n",
            m.divide_id.as_ref().unwrap(),
            m.elevation_mean_m.unwrap(),
            m.slope_mean_deg.unwrap(),
            m.aspect_mean_deg.unwrap(),
            m.centroid_lat_deg.unwrap(),
            m.centroid_lon_deg.unwrap(),
            m.soil_class.unwrap(),
            m.veg_class.unwrap(),
            m.soil_color.unwrap(),
            m.impervious_fraction.unwrap(),
            m.soil_depth_m.unwrap(),
            m.soil_b.unwrap(),
            m.soil_satdk_m_s.unwrap(),
            m.soil_satpsi_m.unwrap(),
            m.soil_slop.unwrap(),
            m.soil_smcmax.unwrap(),
            m.soil_wltsmc.unwrap(),
            m.refkdt.unwrap(),
            m.alpha_fc.unwrap(),
            m.max_gw_storage_m.unwrap(),
            m.cgw_m_h.unwrap(),
            m.gw_expon.unwrap(),
            m.k_lf.unwrap(),
            m.k_nash.unwrap(),
            gi.split(',').count(),
            gi
        ));
    }
    std::fs::write(format!("{od}/model_attributes.csv"), csv)?;
    // label raster for referee comparisons and maps (u32 LE, 0 = none, k+1 = canonical rank k)
    let mut rank_of = vec![0u32; ncat];
    for (r, &k) in order.iter().enumerate() {
        rank_of[k] = r as u32 + 1;
    }
    let lab: Vec<u8> = cat.labels.par_iter().flat_map_iter(|&l| (if l == 0 { 0 } else { rank_of[l as usize - 1] }).to_le_bytes()).collect();
    std::fs::write(format!("{od}/labels_u32le.bin"), lab)?;
    std::fs::write(format!("{od}/grid.json"), serde_json::to_vec_pretty(&json!({"grid": g, "labels": "labels_u32le.bin: u32 little-endian, row-major north row first; 0 none, r = fid of model_attributes row r"}))?)?;
    // contract fabric with model_attributes, when the catchments came from one
    let mut fabric_check = Value::Null;
    if let (Some(mut fab), Some(fo)) = (fabric, fabric_out) {
        fab.model_attributes = Some(rows.clone());
        if let Some(meta) = fab.fabric_metadata.as_mut() {
            let set = |meta: &mut Vec<MetadataRow>, key: &str, val: String| {
                if let Some(r) = meta.iter_mut().find(|r| r.key.as_deref() == Some(key)) {
                    r.value = Some(val);
                } else {
                    let fid = meta.len() as i64 + 1;
                    meta.push(MetadataRow { fid, key: Some(key.into()), value: Some(val) });
                }
            };
            set(meta, "soil_class_table", recipe.soil_class_table.clone());
            set(meta, "veg_class_table", cfg.veg_class_table.clone());
        }
        hydro_world::derive::canonicalize(&mut fab);
        // the output is this run's own product: replace an earlier run's file
        if std::path::Path::new(&fo).exists() {
            std::fs::remove_file(&fo)?;
        }
        hydro_world::gpkg::write(&fab, &fo)?;
        let (back, _) = hydro_world::gpkg::read(&fo)?;
        let r = hydro_world::check::check(&back);
        fabric_check = json!({"path": fo, "check": r, "model_attributes_roundtrip_equal": back.model_attributes == fab.model_attributes});
    }
    st.mark("write outputs", "1 thread (file writes)", t0, json!({}));

    let total = t_all.elapsed().as_secs_f64();
    let mut inputs: Vec<Value> = vec![];
    for p in dem_used.iter().chain(lc_used.iter()).chain(soil_paths.iter()) {
        inputs.push(json!({"path": p}));
    }
    // input hashes in parallel (they are large)
    let hashes: Vec<String> = inputs.par_iter().map(|v| sha256_file(v["path"].as_str().unwrap())).collect();
    for (v, h) in inputs.iter_mut().zip(hashes) {
        v["sha256"] = json!(h);
    }
    let receipt = json!({
        "schema": "arwen.hydro-world.params.receipt.v1",
        "domain": cfg.domain,
        "config": cfg,
        "params_table": {"path": tab_path_abs(&cfg.params_table), "sha256": sha256_file(&cfg.params_table)},
        "utc": now_utc(),
        "host_threads": threads,
        "catchments": {"kind": cat.kind, "count": ncat, "notes": cat.notes},
        "grid": g,
        "stages": st.0,
        "total_seconds": total,
        "summary": summary,
        "fallbacks": fallbacks,
        "cfe_recipe": {"name": cfe_recipe_name, "row": cfe_recipe},
        "violations": violations,
        "complete_and_bounded": violations.is_empty(),
        "fabric_check": fabric_check,
        "data_terms": [dem_src.licence.attribution, soil_src.licence.attribution, lc_src.licence.attribution],
        "inputs": inputs,
        "missing_tiles": {"dem": dem_missing, "land_cover": lc_missing},
        "lgar_note": "LGAR parameters are written to lgar_attributes.json (contract v1 holds no LGAR columns; section 9 defers them to v1.1 with the LGAR owner)",
    });
    std::fs::write(format!("{od}/receipt.json"), serde_json::to_vec_pretty(&receipt)?)?;
    Ok(receipt)
}

/// Water-majority rule (coordinator ruling 2026-10-18T11:50Z) on per-class land cover
/// counts: (all-class majority, majority over non-water classes), usize::MAX where no
/// class has a count. Ties keep the lower class index.
pub(crate) fn majority_classes(lc: &[u64], water: &[bool]) -> (usize, usize) {
    let (mut vk_all, mut vmax_all) = (usize::MAX, 0u64);
    let (mut vk, mut vmax) = (usize::MAX, 0u64);
    for (q, &v) in lc.iter().enumerate() {
        if v > vmax_all {
            vmax_all = v;
            vk_all = q;
        }
        if !water[q] && v > vmax {
            vmax = v;
            vk = q;
        }
    }
    (vk_all, vk)
}

/// Field capacity water content on the Clapp and Hornberger (1978, Water Resour. Res.
/// 14(4):601-604) retention curve psi = psi_s (theta/theta_s)^-b at the CFE field
/// capacity head alpha_fc * one atmosphere.
pub(crate) fn field_capacity_theta(r: &crate::table::CfeRecipe, alpha_fc: f64, smcmax: f64, wltsmc: f64, satpsi: f64, b: f64) -> f64 {
    (smcmax * (alpha_fc * r.atmosphere_head_m / satpsi).powf(-1.0 / b)).clamp(wltsmc, smcmax)
}

/// (K_lf m/h, K_nash 1/h) of the hillslope_kinematic recipe row.
pub(crate) fn hillslope_kinematic_k(r: &crate::table::CfeRecipe, satdk: f64, hillslope_length_m: f64, hillslope_sin_slope: f64, depth: f64, smcmax: f64, theta_fc: f64) -> (f64, f64) {
    let f = r.lateral_ksat_factor;
    let l = hillslope_length_m.max(r.min_hillslope_length_m);
    let sb = hillslope_sin_slope.max(r.min_sin_slope);
    let theta_d = (smcmax - theta_fc).max(r.min_drainable_porosity);
    // Sloan and Moore (1984), Water Resour. Res. 20(12):1815-1822, kinematic storage
    // model: a sloping saturated wedge of length L drains as Q = 2 Ks sin(b) S / (theta_d L).
    // The CFE lateral outlet is K_lf * (S - S_fc) / (S_max - S_fc) with S_max - S_fc = D theta_d,
    // so K_lf (m/h) = 3600 * 2 F Ks sin(b) D / L, F the lateral/vertical conductivity ratio.
    let klf = 3600.0 * 2.0 * f * satdk * sb * depth / l;
    // Beven (1981), Water Resour. Res. 17(5):1419-1424, kinematic subsurface stormflow:
    // celerity c = F Ks sin(b) / theta_d, hillslope travel time T = L / c; a Nash (1957)
    // cascade of N reservoirs with that mean residence has K = N / T (per hour).
    let kn = r.n_nash * 3600.0 * f * satdk * sb / (theta_d * l);
    (klf.clamp(r.k_lf_range[0], r.k_lf_range[1]), kn.clamp(r.k_nash_range[0], r.k_nash_range[1]))
}

fn tab_path_abs(p: &str) -> String {
    std::fs::canonicalize(p).map(|x| x.display().to_string()).unwrap_or_else(|_| p.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped_recipe() -> crate::table::CfeRecipe {
        let tab = ParamTable::load(concat!(env!("CARGO_MANIFEST_DIR"), "/data/params.v1.json")).unwrap();
        assert_eq!(tab.cfe_recipe_default.as_deref(), Some("hillslope-kinematic-v1"));
        tab.cfe_recipes["hillslope-kinematic-v1"].clone()
    }

    fn close(got: f64, want: f64) {
        assert!((got - want).abs() <= 1e-12 * want.abs(), "got {got}, want {want}");
    }

    #[test]
    fn hillslope_kinematic_recipe_values() {
        let r = shipped_recipe();
        // the shipped row: F 1000, N 2, floors L 45 m, sin(b) 0.005, theta_d 0.02
        assert_eq!(r.method, "hillslope_kinematic");
        assert_eq!((r.lateral_ksat_factor, r.n_nash), (1000.0, 2.0));
        // Ks 1e-5 m/s, L 200 m, sin(b) 0.1, D 2 m, theta_d 0.45 - 0.30:
        // K_lf = 3600 * 2 * 1000 * 1e-5 * 0.1 * 2 / 200 = 0.072 m/h
        // K_nash = 2 * 3600 * 1000 * 1e-5 * 0.1 / (0.15 * 200) = 0.24 1/h
        let (klf, kn) = hillslope_kinematic_k(&r, 1e-5, 200.0, 0.1, 2.0, 0.45, 0.30);
        close(klf, 0.072);
        close(kn, 0.24);
        // floors: L 10 -> 45, sin(b) 0.001 -> 0.005, theta_d 0.01 -> 0.02
        // K_lf = 72 * 0.005 * 2 / 45 = 0.016, K_nash = 2 * 36 * 0.005 / (0.02 * 45) = 0.4
        let (klf, kn) = hillslope_kinematic_k(&r, 1e-5, 10.0, 0.001, 2.0, 0.45, 0.44);
        close(klf, 0.016);
        close(kn, 0.4);
        // ranges: Ks 1e-4, L 50, sin(b) 0.5 gives K_lf 14.4 and K_nash 48, clamped to the row's tops
        let (klf, kn) = hillslope_kinematic_k(&r, 1e-4, 50.0, 0.5, 2.0, 0.45, 0.30);
        assert_eq!((klf, kn), (r.k_lf_range[1], r.k_nash_range[1]));
        // field capacity: alpha_fc 0.33 at psi_s = 0.33 * 10.33 / 32 and b 5 puts
        // (psi/psi_s)^(-1/b) = 32^(-0.2) = 0.5, so theta_fc = 0.45 * 0.5
        assert_eq!(r.alpha_fc_by_soil_class.get("1"), Some(&0.1));
        assert_eq!(r.alpha_fc_default, 0.33);
        close(field_capacity_theta(&r, 0.33, 0.45, 0.05, 0.33 * r.atmosphere_head_m / 32.0, 5.0), 0.225);
        // clamped into [wltsmc, smcmax]
        assert_eq!(field_capacity_theta(&r, 0.33, 0.45, 0.30, 0.33 * r.atmosphere_head_m / 32.0, 5.0), 0.30);
    }

    #[test]
    fn water_majority_takes_the_land_class() {
        // classes: 0 tree, 1 water, 2 grass
        let water = [false, true, false];
        // water is the all-class majority; the LSM gets the land majority (tree)
        assert_eq!(majority_classes(&[40, 50, 10], &water), (1, 0));
        // land majority already: unchanged
        assert_eq!(majority_classes(&[10, 30, 60], &water), (2, 2));
        // water only: no land class (the grassland stand-in is chosen by the caller)
        assert_eq!(majority_classes(&[0, 30, 0], &water), (1, usize::MAX));
        // nothing at all
        assert_eq!(majority_classes(&[0, 0, 0], &water), (usize::MAX, usize::MAX));
        // ties keep the lower class index
        assert_eq!(majority_classes(&[20, 5, 20], &water), (0, 0));
    }
}
