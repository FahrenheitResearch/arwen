//! `--grid latlon`: a regular equiangular lattice, bilinear (or an
//! area mean of bilinear samples) in the model's own index space.
//!
//! Every target point is mapped into the model grid by inverting the
//! model's projection with static-fields' transcription of WRF's own
//! `module_llxy` (Lambert, Mercator, polar), parameters from the file and
//! XLAT/XLONG[0, 0] as the known point.  Before any field moves, that
//! inverse is checked against XLAT/XLONG at every model point; a
//! disagreement of more than 0.01 cell is a refusal, because every value
//! would land in the wrong place.
//!
//! The lattice is anchored at integer multiples of its spacing, so two
//! runs' exports share points and any spacing dividing 0.25 degree lands
//! on ERA5's own points.  The extent is the largest lattice-aligned box
//! whose every point (every sample, for the area mean) lies inside the
//! domain and outside the lateral boundary rows, so no point needs fill.

use rayon::prelude::*;
use static_fields::projection::{GridSpec, ProjectedGrid, ProjectionKind};

use crate::error::{refuse, Result};
use crate::frame::FileMeta;
use crate::grid::NativeGrid;
use crate::request::RegridMethod;

const DEG_PER_KM: f64 = 1.0 / 111.32;

/// The source grid in index space: fractional 0-based (i, j) for a point.
enum Inverse {
    Projected(Box<ProjectedGrid>),
    Regular { lat0: f64, dlat: f64, lon0: f64, dlon: f64, nx: usize, periodic: bool },
}

impl Inverse {
    fn index(&self, lat: f64, lon: f64) -> (f64, f64) {
        match self {
            Inverse::Projected(grid) => {
                let (x, y) = grid.latlon_to_ij(lat, lon);
                (x - 1.0, y - 1.0)
            }
            Inverse::Regular { lat0, dlat, lon0, dlon, nx, periodic } => {
                let mut i = (lon - lon0).rem_euclid(360.0) / dlon;
                if !periodic && i > *nx as f64 + 0.5 {
                    i -= 360.0 / dlon;
                }
                (i, (lat - lat0) / dlat)
            }
        }
    }
}

/// The target lattice and the sparse weights that fill it.
#[derive(Debug, Clone)]
pub struct Regrid {
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    pub deg: f64,
    pub method: RegridMethod,
    pub description: String,
    pub boundary_rows: usize,
    /// CSR: target point t takes `weight[offsets[t]..offsets[t+1]]` of the
    /// source cells `index[..]`.
    offsets: Vec<u32>,
    index: Vec<u32>,
    weight: Vec<f64>,
}

impl Regrid {
    pub fn ny(&self) -> usize {
        self.lat.len()
    }

    pub fn nx(&self) -> usize {
        self.lon.len()
    }

    /// Apply to one `[ny_src, nx_src]` plane.
    pub fn apply(&self, plane: &[f32]) -> Vec<f32> {
        (0..self.lat.len() * self.lon.len())
            .into_par_iter()
            .with_min_len(4096)
            .map(|t| {
                let (a, b) = (self.offsets[t] as usize, self.offsets[t + 1] as usize);
                let mut sum = 0.0f64;
                for e in a..b {
                    sum += self.weight[e] * f64::from(plane[self.index[e] as usize]);
                }
                sum as f32
            })
            .collect()
    }

    /// Apply to a stack of planes (`[levels, ny_src, nx_src]`).
    pub fn apply_stack(&self, stack: &[f32], src_cells: usize) -> Vec<f32> {
        stack.chunks(src_cells).flat_map(|plane| self.apply(plane)).collect()
    }
}

fn nearest_spacing(target: f64, table: &[f64]) -> f64 {
    table
        .iter()
        .copied()
        .filter(|s| *s > 0.0)
        .min_by(|a, b| (a.ln() - target.ln()).abs().total_cmp(&(b.ln() - target.ln()).abs()))
        .unwrap_or(target)
}

/// Largest all-true rectangle in a row-major boolean grid: (r0, r1, c0, c1)
/// inclusive, or None.
fn largest_rectangle(mask: &[bool], rows: usize, cols: usize) -> Option<(usize, usize, usize, usize)> {
    let mut heights = vec![0usize; cols];
    let mut best: Option<(usize, (usize, usize, usize, usize))> = None;
    for r in 0..rows {
        for c in 0..cols {
            heights[c] = if mask[r * cols + c] { heights[c] + 1 } else { 0 };
        }
        // Largest rectangle in the histogram `heights`.
        let mut stack: Vec<usize> = Vec::new();
        for c in 0..=cols {
            let h = if c < cols { heights[c] } else { 0 };
            while let Some(&top) = stack.last() {
                if heights[top] <= h {
                    break;
                }
                stack.pop();
                let height = heights[top];
                let left = stack.last().map(|&s| s + 1).unwrap_or(0);
                let width = c - left;
                let area = height * width;
                if area > 0 && best.is_none_or(|(a, _)| area > a) {
                    best = Some((area, (r + 1 - height, r, left, c - 1)));
                }
            }
            stack.push(c);
        }
    }
    best.map(|(_, rect)| rect)
}

/// Plan the lattice and weights for one domain.
pub fn plan(
    meta: &FileMeta,
    native: &NativeGrid,
    xlat: &[f64],
    xlong: &[f64],
    deg: Option<f64>,
    method: RegridMethod,
    spacings: &[f64],
) -> Result<Regrid> {
    let (nx, ny) = (meta.nx, meta.ny);
    let inverse = match (&native.regular, meta.map_proj) {
        (Some(regular), _) => {
            if let Some(d) = deg {
                if d + 1e-9 < regular.dlat.max(regular.dlon) {
                    return Err(refuse(format!(
                        "the history grid is a {:.4} degree latitude-longitude grid and {d} degree is finer; upsampling it would add bytes and no information",
                        regular.dlat.max(regular.dlon)
                    )));
                }
            }
            let lat_src: Vec<f64> = (0..ny).map(|j| xlat[j * nx]).collect();
            let signed_dlat = (lat_src[ny - 1] - lat_src[0]) / (ny - 1) as f64;
            Inverse::Regular {
                lat0: lat_src[0],
                dlat: signed_dlat,
                lon0: xlong[0],
                dlon: regular.dlon,
                nx,
                periodic: regular.periodic,
            }
        }
        (None, 1 | 2 | 3) => {
            let kind = match meta.map_proj {
                1 => ProjectionKind::Lambert,
                2 => ProjectionKind::Polar,
                _ => ProjectionKind::Mercator,
            };
            let spec = GridSpec {
                kind,
                ref_lat: xlat[0],
                ref_lon: xlong[0],
                truelat1: meta.truelat1,
                truelat2: meta.truelat2,
                stand_lon: meta.stand_lon,
                dx: meta.dx,
                dy: meta.dy,
                e_we: nx as i64 + 1,
                e_sn: ny as i64 + 1,
                known_x: 1.0,
                known_y: 1.0,
                moad_cen_lat: meta.moad_cen_lat,
                moad_cen_lon: meta.cen_lon,
                lat_deg: Vec::new(),
                lon0_deg: 0.0,
                dlon_deg: 0.0,
            };
            let grid = ProjectedGrid::new(spec).map_err(|e| {
                refuse(format!(
                    "the history file's projection parameters do not describe a WRF grid ({e}), so no point could be placed for the regrid"
                ))
            })?;
            Inverse::Projected(Box::new(grid))
        }
        (None, 6) => {
            return Err(refuse(
                "the history grid is a rotated-pole latitude-longitude grid, which has no inverse here; the regrid would sample the wrong ground (use --grid native)",
            ))
        }
        (None, other) => {
            return Err(refuse(format!(
                "MAP_PROJ {other} has no inverse here, so the regrid would sample the wrong ground (use --grid native)"
            )))
        }
    };

    // The inverse must reproduce the file's own coordinates everywhere.
    let worst = (0..nx * ny)
        .into_par_iter()
        .map(|k| {
            let (i, j) = inverse.index(xlat[k], xlong[k]);
            let (ei, ej) = ((k % nx) as f64, (k / nx) as f64);
            let di = match &inverse {
                Inverse::Regular { periodic: true, nx, .. } => {
                    let d = (i - ei).rem_euclid(*nx as f64);
                    d.min(*nx as f64 - d)
                }
                _ => (i - ei).abs(),
            };
            di.max((j - ej).abs())
        })
        .reduce(|| 0.0f64, f64::max);
    if !(worst <= 0.01) {
        return Err(refuse(format!(
            "the projection rebuilt from the file's attributes misplaces its own XLAT/XLONG by {worst:.3} cells (limit 0.01), so every regridded value would land in the wrong place (use --grid native)"
        )));
    }

    let source_deg = match &native.regular {
        Some(regular) => regular.dlat.max(regular.dlon),
        None => meta.dx.max(meta.dy) / 1000.0 * DEG_PER_KM,
    };
    let deg = match deg {
        Some(d) => d,
        None => nearest_spacing(source_deg, spacings),
    };
    let boundary = match &native.regular {
        Some(regular) if regular.periodic => 0,
        _ => meta.spec_bdy_width.map(|w| w.max(0) as usize).unwrap_or(0),
    };

    // Longitude frame: centred on the domain so a box across the
    // antimeridian is contiguous.
    let centre_lon = {
        let (s, c) = xlong.iter().fold((0.0f64, 0.0f64), |(s, c), &l| {
            (s + l.to_radians().sin(), c + l.to_radians().cos())
        });
        s.atan2(c).to_degrees()
    };
    let unwrap = |l: f64| centre_lon + (l - centre_lon + 180.0).rem_euclid(360.0) - 180.0;
    let periodic = matches!(&native.regular, Some(r) if r.periodic);
    let (mut lat_min, mut lat_max, mut lon_min, mut lon_max) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for k in 0..nx * ny {
        lat_min = lat_min.min(xlat[k]);
        lat_max = lat_max.max(xlat[k]);
        let l = unwrap(xlong[k]);
        lon_min = lon_min.min(l);
        lon_max = lon_max.max(l);
    }
    if periodic {
        lon_min = 0.0;
        lon_max = 360.0 - deg;
    }
    let k_lat0 = (lat_min / deg).ceil() as i64;
    let k_lat1 = (lat_max / deg).floor() as i64;
    let k_lon0 = (lon_min / deg).ceil() as i64;
    let k_lon1 = if periodic { ((360.0 / deg).round() as i64) - 1 } else { (lon_max / deg).floor() as i64 };
    if k_lat1 < k_lat0 || k_lon1 < k_lon0 {
        return Err(refuse(format!(
            "no {deg} degree lattice point lies inside the domain, so the regular grid would be empty"
        )));
    }
    let rows = (k_lat1 - k_lat0 + 1) as usize;
    let cols = (k_lon1 - k_lon0 + 1) as usize;
    if rows * cols > 200_000_000 {
        return Err(refuse(format!(
            "a {deg} degree lattice over this domain has {} points, more than any dataset here should hold; choose a coarser spacing",
            rows * cols
        )));
    }

    let samples = match method {
        RegridMethod::Bilinear => 1usize,
        RegridMethod::AreaMean => {
            let km = deg / DEG_PER_KM;
            let src_km = match &native.regular {
                Some(_) => source_deg / DEG_PER_KM,
                None => meta.dx.min(meta.dy) / 1000.0,
            };
            ((2.0 * km / src_km).ceil() as usize).max(1)
        }
    };
    let offsets_in_cell: Vec<f64> = (0..samples)
        .map(|s| if samples == 1 { 0.0 } else { -0.5 + (s as f64 + 0.5) / samples as f64 })
        .collect();
    let (lo_i, hi_i) = (boundary as f64, (nx - 1 - boundary.min(nx - 1)) as f64);
    let (lo_j, hi_j) = (boundary as f64, (ny - 1 - boundary.min(ny - 1)) as f64);
    let inside = |i: f64, j: f64| -> bool {
        let i_ok = if periodic { true } else { i >= lo_i - 1e-9 && i <= hi_i + 1e-9 };
        i_ok && j >= lo_j - 1e-9 && j <= hi_j + 1e-9
    };
    let lattice_lat = |r: usize| (k_lat0 + r as i64) as f64 * deg;
    let lattice_lon = |c: usize| (k_lon0 + c as i64) as f64 * deg;

    // Sample positions for every lattice cell: (fractional index, weight).
    let positions: Vec<Option<Vec<(f64, f64, f64)>>> = (0..rows * cols)
        .into_par_iter()
        .with_min_len(1024)
        .map(|t| {
            let (r, c) = (t / cols, t % cols);
            let (lat, lon) = (lattice_lat(r), lattice_lon(c));
            let mut out = Vec::with_capacity(samples * samples);
            for &dy in &offsets_in_cell {
                for &dx in &offsets_in_cell {
                    let (slat, slon) = (lat + dy * deg, lon + dx * deg);
                    if slat.abs() > 90.0 {
                        return None;
                    }
                    let (i, j) = inverse.index(slat, slon);
                    if !(i.is_finite() && j.is_finite() && inside(i, j)) {
                        return None;
                    }
                    out.push((i, j, slat.to_radians().cos()));
                }
            }
            Some(out)
        })
        .collect();
    let mask: Vec<bool> = positions.iter().map(Option::is_some).collect();
    let (r0, r1, c0, c1) = largest_rectangle(&mask, rows, cols).ok_or_else(|| {
        refuse(format!(
            "no {deg} degree lattice point lies inside the domain's interior, so the regular grid would be empty"
        ))
    })?;

    let mut offsets = vec![0u32];
    let mut index = Vec::new();
    let mut weight = Vec::new();
    for r in r0..=r1 {
        for c in c0..=c1 {
            let samples = positions[r * cols + c].as_ref().expect("inside the rectangle");
            let total: f64 = samples.iter().map(|s| s.2).sum();
            let mut entries: Vec<(u32, f64)> = Vec::new();
            for &(i, j, w) in samples {
                let share = w / total;
                let j0 = (j.floor() as isize).clamp(0, ny as isize - 1) as usize;
                let j1 = (j0 + 1).min(ny - 1);
                let fy = (j - j0 as f64).clamp(0.0, 1.0);
                let (i0, i1, fx) = if periodic {
                    let fl = i.floor();
                    let i0 = (fl as i64).rem_euclid(nx as i64) as usize;
                    (i0, (i0 + 1) % nx, i - fl)
                } else {
                    let i0 = (i.floor() as isize).clamp(0, nx as isize - 1) as usize;
                    let i1 = (i0 + 1).min(nx - 1);
                    (i0, i1, (i - i0 as f64).clamp(0.0, 1.0))
                };
                for (src, w) in [
                    (j0 * nx + i0, (1.0 - fx) * (1.0 - fy)),
                    (j0 * nx + i1, fx * (1.0 - fy)),
                    (j1 * nx + i0, (1.0 - fx) * fy),
                    (j1 * nx + i1, fx * fy),
                ] {
                    if w != 0.0 {
                        entries.push((src as u32, share * w));
                    }
                }
            }
            // Merge repeated source cells so the weights are a short list.
            entries.sort_by_key(|e| e.0);
            let mut merged: Vec<(u32, f64)> = Vec::with_capacity(entries.len());
            for (src, w) in entries {
                match merged.last_mut() {
                    Some(last) if last.0 == src => last.1 += w,
                    _ => merged.push((src, w)),
                }
            }
            for (src, w) in merged {
                index.push(src);
                weight.push(w);
            }
            offsets.push(index.len() as u32);
        }
    }

    let lat: Vec<f64> = (r0..=r1).map(lattice_lat).collect();
    let mut lon: Vec<f64> = (c0..=c1).map(lattice_lon).collect();
    // 0..360 unless the box crosses the prime meridian, where -180..180
    // keeps the axis monotone.
    let (mn, mx) = (lon[0], lon[lon.len() - 1]);
    let meridian = (mx / 360.0).floor() * 360.0;
    let crosses_prime = !periodic && meridian >= mn && meridian <= mx && mx - mn < 360.0;
    for l in lon.iter_mut() {
        let v = l.rem_euclid(360.0);
        *l = if crosses_prime && v >= 180.0 { v - 360.0 } else { v };
    }
    // Lattice values are multiples of deg; round away the float residue.
    let tidy = |v: f64| (v / deg).round() * deg;
    let lat: Vec<f64> = lat.into_iter().map(tidy).collect();
    let lon: Vec<f64> = lon.into_iter().map(tidy).collect();
    let method_text = match method {
        RegridMethod::Bilinear => "bilinear in model index space".to_string(),
        RegridMethod::AreaMean => format!(
            "area mean of {samples}x{samples} bilinear samples per cell, cos-latitude weighted (approximately conservative)"
        ),
    };
    let description = format!("regular latitude-longitude {deg} deg, {method_text}");
    Ok(Regrid { lat, lon, deg, method, description, boundary_rows: boundary, offsets, index, weight })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_largest_rectangle_is_found() {
        #[rustfmt::skip]
        let mask = [
            false, true,  false, false,
            true,  true,  true,  true,
            true,  true,  true,  false,
        ];
        assert_eq!(largest_rectangle(&mask, 3, 4), Some((1, 2, 0, 2)));
        assert_eq!(largest_rectangle(&[false; 4], 2, 2), None);
    }

    #[test]
    fn spacing_is_the_table_row_nearest_in_log() {
        let table = [0.01, 0.025, 0.05, 0.1, 0.25];
        assert_eq!(nearest_spacing(0.108, &table), 0.1);
        assert_eq!(nearest_spacing(0.2, &table), 0.25);
        assert_eq!(nearest_spacing(0.035, &table), 0.025);
    }
}
