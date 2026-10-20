//! The analysis grid and the raster inputs laid onto it.
//!
//! One regular lon/lat grid per domain (WGS84, cell centres at half cells, rows
//! north to south) at the DEM resolution (3 arc-seconds for Copernicus GLO-90,
//! 1200 cells per degree). Soil (SoilGrids WCS at 0.0025 deg) and land cover
//! (WorldCover at 1/12000 deg) align to it exactly when their tile origins sit
//! on the grid, which is how the fetch requests them. Every loader runs in
//! parallel (Rayon over rows, tiles or blocks).
use crate::table::{DemSource, LandCoverSource, SoilSource};
use crate::tiff::{sample_f32, GeoTiff};
use crate::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};

/// authalic sphere radius of WGS84 (m): cell areas on this sphere equal the
/// ellipsoidal areas to well under 0.5 % at mid latitudes
pub const R_AUTHALIC: f64 = 6371007.181;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Grid {
    pub west: f64,
    pub north: f64,
    pub cells_per_degree: usize,
    pub nx: usize,
    pub ny: usize,
}

impl Grid {
    pub fn new(bbox: [f64; 4], cells_per_degree: usize) -> Result<Grid> {
        let [w, s, e, n] = bbox;
        if !(w < e && s < n) {
            return Err(format!("bad bbox {bbox:?}").into());
        }
        let c = cells_per_degree as f64;
        let nx = ((e - w) * c).round() as usize;
        let ny = ((n - s) * c).round() as usize;
        Ok(Grid { west: w, north: n, cells_per_degree, nx, ny })
    }
    pub fn res(&self) -> f64 {
        1.0 / self.cells_per_degree as f64
    }
    pub fn n(&self) -> usize {
        self.nx * self.ny
    }
    pub fn lon(&self, i: usize) -> f64 {
        self.west + (i as f64 + 0.5) * self.res()
    }
    pub fn lat(&self, j: usize) -> f64 {
        self.north - (j as f64 + 0.5) * self.res()
    }
    /// metres per cell east-west at row j
    pub fn dx_m(&self, j: usize) -> f64 {
        R_AUTHALIC * self.lat(j).to_radians().cos() * self.res().to_radians()
    }
    pub fn dy_m(&self) -> f64 {
        R_AUTHALIC * self.res().to_radians()
    }
    pub fn cell_area_m2(&self, j: usize) -> f64 {
        self.dx_m(j) * self.dy_m()
    }
}

/// Copernicus-style 1 degree DEM tiles mosaicked by nearest sample. Missing
/// tiles (open sea, where the provider publishes none) stay NaN.
pub fn mosaic_dem(g: &Grid, src: &DemSource, dir: &str) -> Result<(Vec<f32>, Vec<String>, Vec<String>)> {
    let lat0 = (g.north - g.ny as f64 * g.res()).floor() as i32 - 1;
    let lat1 = g.north.ceil() as i32;
    let lon0 = g.west.floor() as i32 - 1;
    let lon1 = (g.west + g.nx as f64 * g.res()).ceil() as i32;
    let mut keys = vec![];
    for la in lat0..=lat1 {
        for lo in lon0..=lon1 {
            keys.push((la, lo));
        }
    }
    let loaded: Vec<((i32, i32), String, Option<Result<(GeoTiff, Vec<f32>)>>)> = keys
        .par_iter()
        .map(|&(la, lo)| {
            let name = src.tile_name(la, lo);
            let path = format!("{dir}/{name}");
            if std::path::Path::new(&path).exists() {
                let r = GeoTiff::open(&path).and_then(|t| t.read_all_f32().map(|v| (t, v)));
                ((la, lo), path, Some(r))
            } else {
                ((la, lo), path, None)
            }
        })
        .collect();
    let mut tiles: HashMap<(i32, i32), (GeoTiff, Vec<f32>)> = HashMap::new();
    let mut used = vec![];
    let mut missing = vec![];
    for (k, path, r) in loaded {
        match r {
            Some(r) => {
                tiles.insert(k, r?);
                used.push(path);
            }
            None => missing.push(path),
        }
    }
    let mut dem = vec![f32::NAN; g.n()];
    dem.par_chunks_mut(g.nx).enumerate().for_each(|(j, row)| {
        let lat = g.lat(j);
        for (i, v) in row.iter_mut().enumerate() {
            let lon = g.lon(i);
            let (fl, fo) = (lat.floor() as i32, lon.floor() as i32);
            'search: for dla in [0, -1, 1] {
                for dlo in [0, -1, 1] {
                    if let Some((t, d)) = tiles.get(&(fl + dla, fo + dlo)) {
                        let c = ((lon - t.origin_x) / t.dx).floor();
                        let r = ((t.origin_y - lat) / t.dy).floor();
                        if c >= 0.0 && r >= 0.0 && (c as usize) < t.width && (r as usize) < t.height {
                            *v = d[r as usize * t.width + c as usize];
                            break 'search;
                        }
                    }
                }
            }
        }
    });
    Ok((dem, used, missing))
}

/// One soil property at one depth on its own grid (SoilGrids units decoded by
/// the table's scale factor; nodata -> NaN).
pub struct SoilLayer {
    pub ox: f64,
    pub oy: f64,
    pub dx: f64,
    pub dy: f64,
    pub w: usize,
    pub h: usize,
    pub v: Vec<f32>,
}

impl SoilLayer {
    #[inline]
    pub fn index_of(&self, lon: f64, lat: f64) -> Option<usize> {
        let c = ((lon - self.ox) / self.dx).floor();
        let r = ((self.oy - lat) / self.dy).floor();
        if c >= 0.0 && r >= 0.0 && (c as usize) < self.w && (r as usize) < self.h {
            Some(r as usize * self.w + c as usize)
        } else {
            None
        }
    }
}

/// Load every (property, depth) layer the soil source row lists for a domain.
/// Returns map "prop/depth_label" -> layer, all on one common soil grid.
pub fn load_soil(src: &SoilSource, dir: &str, prefix: &str) -> Result<(HashMap<String, SoilLayer>, Vec<String>)> {
    let mut jobs = vec![];
    for (prop, _) in src.props.iter() {
        for d in &src.depth_labels {
            jobs.push((prop.clone(), d.clone()));
        }
    }
    let res: Vec<Result<(String, SoilLayer, String)>> = jobs
        .par_iter()
        .map(|(prop, d)| {
            let path = format!("{dir}/{}", src.file_name(prefix, prop, d));
            let t = GeoTiff::open(&path)?;
            let scale = src.props[prop].scale;
            let mut raw = t.read_all_f32()?;
            for x in raw.iter_mut() {
                if x.is_nan() || *x as f64 == src.nodata {
                    *x = f32::NAN;
                } else {
                    *x = (*x as f64 * scale) as f32;
                }
            }
            Ok((format!("{prop}/{d}"), SoilLayer { ox: t.origin_x, oy: t.origin_y, dx: t.dx, dy: t.dy, w: t.width, h: t.height, v: raw }, path))
        })
        .collect();
    let mut out = HashMap::new();
    let mut paths = vec![];
    for r in res {
        let (k, l, p) = r?;
        out.insert(k, l);
        paths.push(p);
    }
    paths.sort();
    // all layers must share one grid
    let first = out.values().next().ok_or("no soil layers")?;
    let (ox, oy, w, h) = (first.ox, first.oy, first.w, first.h);
    for (k, l) in &out {
        if l.ox != ox || l.oy != oy || l.w != w || l.h != h {
            return Err(format!("soil layer {k} is on a different grid").into());
        }
    }
    Ok((out, paths))
}

/// Land-cover class counts per analysis cell: `counts[cell * nclass + k]` is the
/// number of land-cover pixels of class k whose centre falls in the cell
/// (WorldCover: 100 pixels of 10 m per 3 arc-second cell). Tiles and their
/// blocks decode in parallel; cells on block edges are shared, so the counts
/// are atomic.
pub fn accumulate_land_cover(g: &Grid, src: &LandCoverSource, dir: &str) -> Result<(Vec<u8>, Vec<String>, Vec<String>)> {
    let nclass = src.classes.len();
    let code_to_k: HashMap<u32, usize> = src.classes.iter().enumerate().map(|(k, c)| (c.code, k)).collect();
    let mut lut = [u8::MAX; 256];
    for (code, k) in &code_to_k {
        if *code < 256 {
            lut[*code as usize] = *k as u8;
        }
    }
    let n = g.n() * nclass;
    let mut counts = vec![0u8; n];
    // SAFETY: AtomicU8 has the same size, alignment and bit validity as u8.
    let atomics: &[AtomicU8] = unsafe { std::slice::from_raw_parts(counts.as_mut_ptr() as *const AtomicU8, n) };
    let step = src.tile_degrees as i32;
    let south = g.north - g.ny as f64 * g.res();
    let east = g.west + g.nx as f64 * g.res();
    let la0 = (south / step as f64).floor() as i32 * step;
    let lo0 = (g.west / step as f64).floor() as i32 * step;
    let mut used = vec![];
    let mut missing = vec![];
    let mut la = la0;
    while (la as f64) < g.north {
        let mut lo = lo0;
        while (lo as f64) < east {
            let path = format!("{dir}/{}", src.tile_name(la, lo));
            if !std::path::Path::new(&path).exists() {
                missing.push(path);
                lo += step;
                continue;
            }
            let t = GeoTiff::open(&path)?;
            let cpd = g.cells_per_degree as f64;
            (0..t.n_blocks()).into_par_iter().try_for_each(|b| -> Result<()> {
                let d = t.decode_block(b)?;
                let (x0, y0, w, h, row_len) = t.block_rect(b);
                // column -> analysis column (or usize::MAX outside)
                let cols: Vec<usize> = (0..w)
                    .map(|c| {
                        let lon = t.origin_x + (x0 + c) as f64 * t.dx + 0.5 * t.dx;
                        let gi = ((lon - g.west) * cpd).floor();
                        if gi >= 0.0 && (gi as usize) < g.nx { gi as usize } else { usize::MAX }
                    })
                    .collect();
                for r in 0..h {
                    let lat = t.origin_y - (y0 + r) as f64 * t.dy - 0.5 * t.dy;
                    let gj = ((g.north - lat) * cpd).floor();
                    if gj < 0.0 || gj as usize >= g.ny {
                        continue;
                    }
                    let base = gj as usize * g.nx;
                    for (c, &gi) in cols.iter().enumerate() {
                        if gi == usize::MAX {
                            continue;
                        }
                        let v = sample_f32(t.kind, &d, r * row_len + c) as u32;
                        let k = if v < 256 { lut[v as usize] } else { u8::MAX };
                        if k != u8::MAX {
                            atomics[(base + gi) * nclass + k as usize].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                Ok(())
            })?;
            used.push(path);
            lo += step;
        }
        la += step;
    }
    Ok((counts, used, missing))
}
