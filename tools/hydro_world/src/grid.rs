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
    /// Cells whose centres satisfy south < lat <= north, west <= lon < east.
    pub fn for_box(b: &BBox, res_as: f64) -> Result<Grid> {
        let (s, n, w, e) = (b.south * 3600.0, b.north * 3600.0, b.west * 3600.0, b.east * 3600.0);
        let top = (n / res_as + 1e-9).floor() * res_as;
        let left = (w / res_as - 1e-9).ceil() * res_as;
        let ny = (((top - s) / res_as) - 1e-9).ceil() as usize;
        let nx = (((e - left) / res_as) - 1e-9).ceil() as usize;
        if ny == 0 || nx == 0 {
            return Err("bbox smaller than one grid cell".into());
        }
        if ny.checked_mul(nx).is_none_or(|c| c >= u32::MAX as usize - 2) {
            return Err(format!("grid {ny} x {nx} exceeds the u32 cell index; split the domain").into());
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

/// Mosaic decoded tiles (keyed by south-west integer corner) onto `g`.
/// Returns the DEM and a mask of cells that had no tile (open ocean in the
/// Copernicus tiling, which has no tiles over sea).
pub fn mosaic(g: &Grid, tiles: &HashMap<(i32, i32), Raster>) -> Result<(Vec<f32>, Vec<bool>)> {
    let mut z = vec![0f32; g.len()];
    let mut missing = vec![false; g.len()];
    z.par_chunks_mut(g.nx).zip(missing.par_chunks_mut(g.nx)).enumerate().try_for_each(|(row, (zr, mr))| -> Result<()> {
        let lat = g.lat(row);
        // Centres at exactly an integer latitude belong to the tile below
        // them (Copernicus rows run lat0+1 down to lat0+res).
        let la = (lat - 1e-9).floor() as i32;
        for col in 0..g.nx {
            let lon = g.lon(col);
            let lo = (lon + 1e-9).floor() as i32;
            let Some(t) = tiles.get(&(la, lo)) else {
                mr[col] = true;
                continue;
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
            zr[col] = if frac.abs() < 1e-9 {
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
            if let Some(nd) = t.nodata {
                if zr[col] == nd {
                    mr[col] = true;
                    zr[col] = 0.0;
                }
            }
        }
        Ok(())
    })?;
    Ok((z, missing))
}
