//! The builder's working raster: a regular latitude/longitude grid whose cell
//! centres sit on integer multiples of the resolution (arc-seconds), clipped
//! to the user's box, and the DEM mosaic onto it from 1-degree tiles.
//! Tiles whose native longitude spacing is coarser than the grid (Copernicus
//! above 50 degrees latitude) are interpolated linearly in longitude; equal
//! spacing copies values exactly.
use crate::geo;
use crate::tiff::Raster;
use crate::Result;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct BBox {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl BBox {
    pub fn parse(s: &str) -> Result<BBox> {
        let v = s.split(',').map(|x| x.trim().parse::<f64>()).collect::<std::result::Result<Vec<_>, _>>()?;
        if v.len() != 4 {
            return Err("bbox is WEST,SOUTH,EAST,NORTH in degrees".into());
        }
        let b = BBox { west: v[0], south: v[1], east: v[2], north: v[3] };
        if !(b.west < b.east && b.south < b.north && b.south >= -90.0 && b.north <= 90.0 && b.west >= -180.0 && b.east <= 180.0) {
            return Err(format!("bbox {s} is not WEST<EAST, SOUTH<NORTH inside the globe").into());
        }
        Ok(b)
    }
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        lon >= self.west && lon <= self.east && lat >= self.south && lat <= self.north
    }
    /// The 1-degree tiles (south-west corner integer degrees) the box touches.
    pub fn tiles(&self) -> Vec<(i32, i32)> {
        let mut v = Vec::new();
        let (la0, la1) = (self.south.floor() as i32, (self.north.ceil() as i32 - 1).max(self.south.floor() as i32));
        let (lo0, lo1) = (self.west.floor() as i32, (self.east.ceil() as i32 - 1).max(self.west.floor() as i32));
        for la in la0..=la1 {
            for lo in lo0..=lo1 {
                v.push((la, lo));
            }
        }
        v
    }
    pub fn wkt(&self) -> String {
        format!(
            "POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))",
            w = self.west,
            s = self.south,
            e = self.east,
            n = self.north
        )
    }
}

/// Largest working grid the u32 cell index (and the GPU kernels) address.
pub const MAX_CELLS: usize = u32::MAX as usize - 3;

/// The smallest integer factor k >= 1 such that the grid at `k * res_as`
/// over `b` has at most `max_cells` cells (and always fits [`MAX_CELLS`]).
/// k = 1 is the row's own grid, unchanged; k > 1 is the block mean of that
/// grid (see [`aggregate`]), so any box on Earth builds.
pub fn coarsening(b: &BBox, res_as: f64, max_cells: Option<usize>) -> u32 {
    let limit = max_cells.unwrap_or(MAX_CELLS).min(MAX_CELLS);
    let fits = |k: u32| {
        let (ny, nx) = Grid::dims(b, res_as * k as f64);
        ny.checked_mul(nx).is_some_and(|c| c <= limit)
    };
    if fits(1) {
        return 1;
    }
    let (ny, nx) = Grid::dims(b, res_as);
    let mut k = ((ny as f64 * nx as f64 / limit as f64).sqrt().floor() as u32).max(2);
    while k > 2 && fits(k - 1) {
        k -= 1;
    }
    while !fits(k) {
        k += 1;
    }
    k
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Grid {
    /// Resolution in arc-seconds.
    pub res_as: f64,
    /// Latitude of the centre of row 0 (northernmost), arc-seconds.
    pub top_as: f64,
    /// Longitude of the centre of column 0 (westernmost), arc-seconds.
    pub left_as: f64,
    pub ny: usize,
    pub nx: usize,
}

impl Grid {
    /// (ny, nx) of the grid `for_box` makes, without building it.
    pub fn dims(b: &BBox, res_as: f64) -> (usize, usize) {
        let (s, n, w, e) = (b.south * 3600.0, b.north * 3600.0, b.west * 3600.0, b.east * 3600.0);
        let top = (n / res_as + 1e-9).floor() * res_as;
        let left = (w / res_as - 1e-9).ceil() * res_as;
        let ny = (((top - s) / res_as) - 1e-9).ceil().max(0.0) as usize;
        let nx = (((e - left) / res_as) - 1e-9).ceil().max(0.0) as usize;
        (ny, nx)
    }
    /// Cells whose centres satisfy south < lat <= north, west <= lon < east.
    /// The builder picks `res_as` with [`coarsening`], so its grids always
    /// fit the cell index; the check here is the invariant.
    pub fn for_box(b: &BBox, res_as: f64) -> Result<Grid> {
        let top = (b.north * 3600.0 / res_as + 1e-9).floor() * res_as;
        let left = (b.west * 3600.0 / res_as - 1e-9).ceil() * res_as;
        let (ny, nx) = Grid::dims(b, res_as);
        if ny == 0 || nx == 0 {
            return Err("bbox smaller than one grid cell".into());
        }
        if ny.checked_mul(nx).is_none_or(|c| c > MAX_CELLS) {
            return Err(format!("internal: grid {ny} x {nx} exceeds the u32 cell index; the builder chooses its working grid with grid::coarsening").into());
        }
        Ok(Grid { res_as, top_as: top, left_as: left, ny, nx })
    }
    pub fn len(&self) -> usize {
        self.ny * self.nx
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn res(&self) -> f64 {
        self.res_as / 3600.0
    }
    pub fn lat(&self, row: usize) -> f64 {
        (self.top_as - row as f64 * self.res_as) / 3600.0
    }
    pub fn lon(&self, col: usize) -> f64 {
        (self.left_as + col as f64 * self.res_as) / 3600.0
    }
    /// Longitude/latitude of grid corner (ci, cj); corner (0,0) is the NW corner of cell (0,0).
    pub fn corner(&self, ci: usize, cj: usize) -> (f64, f64) {
        ((self.left_as - self.res_as / 2.0 + ci as f64 * self.res_as) / 3600.0, (self.top_as + self.res_as / 2.0 - cj as f64 * self.res_as) / 3600.0)
    }
    /// Ellipsoidal area of a cell in `row`, m2.
    pub fn cell_area_m2(&self, row: usize) -> f64 {
        let r = self.res();
        let lat = self.lat(row);
        geo::quad_area_m2(lat - r / 2.0, lat + r / 2.0, r)
    }
    /// East-west and north-south cell spacing in metres at `row` (local
    /// ellipsoid radii), used for slopes.
    pub fn spacing_m(&self, row: usize) -> (f64, f64) {
        let phi = self.lat(row).to_radians();
        let s2 = phi.sin().powi(2);
        let w = (1.0 - geo::E2 * s2).sqrt();
        let n_rad = geo::A / w;
        let m_rad = geo::A * (1.0 - geo::E2) / (w * w * w);
        let r = self.res().to_radians();
        (n_rad * phi.cos() * r, m_rad * r)
    }
    pub fn rowcol(&self, lon: f64, lat: f64) -> Option<(usize, usize)> {
        let row = ((self.top_as - lat * 3600.0) / self.res_as).round();
        let col = ((lon * 3600.0 - self.left_as) / self.res_as).round();
        (row >= 0.0 && col >= 0.0 && (row as usize) < self.ny && (col as usize) < self.nx).then_some((row as usize, col as usize))
    }
}

/// The DEM at one point of the row's grid: `Ok(None)` where no tile covers
/// it (open ocean in the Copernicus tiling) or the tile says nodata.
/// Centres at exactly an integer latitude belong to the tile below them
/// (Copernicus rows run lat0+1 down to lat0+res). Tiles whose native
/// longitude spacing is coarser than the grid (Copernicus above 50 degrees
/// latitude) are interpolated linearly in longitude; equal spacing copies
/// values exactly.
#[inline]
fn sample(tiles: &HashMap<(i32, i32), Raster>, lat: f64, lon: f64) -> Result<Option<f32>> {
    let la = (lat - 1e-9).floor() as i32;
    let lo = (lon + 1e-9).floor() as i32;
    let Some(t) = tiles.get(&(la, lo)) else {
        return Ok(None);
    };
    let fr = ((t.y0 - lat) / t.dy).round();
    if fr < 0.0 || fr as usize >= t.height {
        return Err(format!("tile {la},{lo} has no row for latitude {lat}").into());
    }
    let r = fr as usize;
    let fx = (lon - t.x0) / t.dx;
    let k = (fx + 1e-9).floor();
    let frac = fx - k;
    let k = k as usize;
    let v0 = t.data[r * t.width + k.min(t.width - 1)];
    let v = if frac.abs() < 1e-9 {
        v0
    } else {
        let v1 = if k + 1 < t.width {
            t.data[r * t.width + k + 1]
        } else if let Some(east) = tiles.get(&(la, lo + 1)) {
            let re = ((east.y0 - lat) / east.dy).round() as usize;
            east.data[re.min(east.height - 1) * east.width]
        } else {
            v0
        };
        (v0 as f64 * (1.0 - frac) + v1 as f64 * frac) as f32
    };
    Ok(match t.nodata {
        Some(nd) if v == nd => None,
        _ => Some(v),
    })
}

/// Mosaic decoded tiles (keyed by south-west integer corner) onto `g`.
/// Returns the DEM and a mask of cells that had no tile (open ocean in the
/// Copernicus tiling, which has no tiles over sea).
pub fn mosaic(g: &Grid, tiles: &HashMap<(i32, i32), Raster>) -> Result<(Vec<f32>, Vec<bool>)> {
    let mut z = vec![0f32; g.len()];
    let mut missing = vec![false; g.len()];
    z.par_chunks_mut(g.nx).zip(missing.par_chunks_mut(g.nx)).enumerate().try_for_each(|(row, (zr, mr))| -> Result<()> {
        let lat = g.lat(row);
        for col in 0..g.nx {
            match sample(tiles, lat, g.lon(col))? {
                Some(v) => zr[col] = v,
                None => mr[col] = true,
            }
        }
        Ok(())
    })?;
    Ok((z, missing))
}

/// The fine-grid rows (latitudes, degrees) inside coarse row `row` of `g`
/// for factor `k` over the row's own spacing `base_as`, north first.
pub fn block_lats(g: &Grid, k: u32, base_as: f64, row: usize) -> impl Iterator<Item = f64> {
    let k = k as i64;
    let lat_c_as = g.top_as - row as f64 * g.res_as;
    (-((k - 1) / 2)..=k / 2).rev().map(move |dr| (lat_c_as + dr as f64 * base_as) / 3600.0)
}

/// Rows `rows` of a coarsened grid `g` (spacing k times the row's own grid
/// spacing `base_as`), each cell the block mean of the k x k cells of the
/// row's own grid it contains: the same point values [`mosaic`] puts on that
/// grid, with blocks on the same half-open convention (south < lat <= north,
/// west <= lon < east). Fine cells outside the tiles of `tiles_box` (the box
/// the build reads) are left out of the block; fine cells with no tile or
/// nodata count as missing. A block whose majority is missing is missing;
/// one whose majority is missing or `sea_value` takes `sea_value`; otherwise
/// it is the mean of its land cells (f64 sum in block order, so the result
/// does not depend on the thread count). `z` and `missing` hold exactly the
/// rows `rows`.
#[allow(clippy::too_many_arguments)]
pub fn aggregate(
    g: &Grid,
    k: u32,
    base_as: f64,
    tiles: &HashMap<(i32, i32), Raster>,
    tiles_box: &BBox,
    sea_value: Option<f32>,
    rows: std::ops::Range<usize>,
    z: &mut [f32],
    missing: &mut [bool],
) -> Result<()> {
    let kk = k as i64;
    let (c_lo, c_hi) = (-(kk / 2), (kk - 1) / 2);
    let s0 = tiles_box.south.floor();
    let n0 = (tiles_box.north.ceil() as i32 - 1).max(tiles_box.south.floor() as i32) as f64 + 1.0;
    let w0 = tiles_box.west.floor();
    let e0 = (tiles_box.east.ceil() as i32 - 1).max(tiles_box.west.floor() as i32) as f64 + 1.0;
    let first = rows.start;
    z.par_chunks_mut(g.nx).zip(missing.par_chunks_mut(g.nx)).enumerate().try_for_each(|(j, (zr, mr))| -> Result<()> {
        let row = first + j;
        let lats: Vec<f64> = block_lats(g, k, base_as, row).filter(|&lat| lat - 1e-9 >= s0 && lat - 1e-9 < n0).collect();
        for col in 0..g.nx {
            let lon_c_as = g.left_as + col as f64 * g.res_as;
            let (mut n, mut nmiss, mut nsea, mut sum, mut nland) = (0u32, 0u32, 0u32, 0f64, 0u32);
            for &lat in &lats {
                for dc in c_lo..=c_hi {
                    let lon = (lon_c_as + dc as f64 * base_as) / 3600.0;
                    if lon + 1e-9 < w0 || lon + 1e-9 >= e0 {
                        continue;
                    }
                    n += 1;
                    match sample(tiles, lat, lon)? {
                        None => nmiss += 1,
                        Some(v) if sea_value == Some(v) => nsea += 1,
                        Some(v) => {
                            sum += v as f64;
                            nland += 1;
                        }
                    }
                }
            }
            if n == 0 || nmiss * 2 > n {
                mr[col] = true;
                zr[col] = 0.0;
            } else if (nmiss + nsea) * 2 > n || nland == 0 {
                zr[col] = sea_value.unwrap_or(0.0);
            } else {
                zr[col] = (sum / nland as f64) as f32;
            }
        }
        Ok(())
    })
}
