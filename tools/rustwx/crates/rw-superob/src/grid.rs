//! The target grid, as `gpuwm/obs/target_grid.py` places gates on it.
//!
//! Horizontal placement is the static-fields projection (the crate that
//! already carries the WPS `llij_*` transcriptions the Python grid's
//! `latlon_to_ij` is), so the two stages cannot disagree about which
//! column a latitude/longitude is in.  Vertical placement is the column's
//! own layer interfaces, `z_w[k, j, i]`.

use static_fields::projection::{GridSpec, ProjectedGrid};

use crate::{Result, SuperobFailure};

/// A borrowed view of one `TargetGrid`.
pub struct GridView<'a> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    projection: ProjectedGrid,
    /// Placement-translation offsets, innermost reference first: the
    /// Python `latlon_to_ij` of a translated grid is the reference's minus
    /// the integer offset, applied once per level of delegation.
    translations: Vec<(i64, i64)>,
    /// `(nz + 1, ny, nx)` layer-interface heights above MSL.
    z_w: &'a [f64],
}

impl<'a> GridView<'a> {
    pub fn new(
        spec: GridSpec,
        translations: Vec<(i64, i64)>,
        nx: usize,
        ny: usize,
        nz: usize,
        z_w: &'a [f64],
    ) -> Result<Self> {
        if z_w.len() != (nz + 1) * ny * nx {
            return Err(SuperobFailure::request(format!(
                "z_w holds {} values, a ({}, {}, {}) column stack needs {}",
                z_w.len(),
                nz + 1,
                ny,
                nx,
                (nz + 1) * ny * nx
            )));
        }
        let projection = ProjectedGrid::new(spec)
            .map_err(|error| SuperobFailure::request(format!("grid projection: {error}")))?;
        Ok(Self { nx, ny, nz, projection, translations, z_w })
    }

    /// `TargetGrid.mass_index`: fractional zero-based `(i, j)`.
    #[inline]
    pub fn mass_index(&self, lat: f64, lon: f64) -> (f64, f64) {
        let (mut x, mut y) = self.projection.latlon_to_ij(lat, lon);
        for (di, dj) in &self.translations {
            x -= *di as f64;
            y -= *dj as f64;
        }
        (x - 1.0, y - 1.0)
    }

    /// `TargetGrid.inside`.
    #[inline]
    pub fn inside(&self, i: i64, j: i64) -> bool {
        i >= 0 && (i as u64) < self.nx as u64 && j >= 0 && (j as u64) < self.ny as u64
    }

    /// `TargetGrid.level_index` for one gate: the model level whose
    /// interfaces bracket `height` in column `(i, j)`, or -1 below the
    /// terrain or at/above the model top.
    #[inline]
    pub fn level_index(&self, i: usize, j: usize, height: f64) -> i64 {
        let plane = self.ny * self.nx;
        let column = j * self.nx + i;
        let bottom = self.z_w[column];
        let top = self.z_w[self.nz * plane + column];
        if !(height >= bottom && height < top) {
            return -1;
        }
        // Count of interfaces 1..=nz at or below the height.  The column
        // increases strictly (TargetGrid refuses anything else), so the
        // count is the partition point.
        let (mut lo, mut hi) = (1usize, self.nz + 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.z_w[mid * plane + column] <= height {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo - 1) as i64
    }
}

/// `horizontal_window`: the inclusive `(j0, j1, i0, i1)` of cells whose
/// mass point lies within `reach_m` of the site by the haversine metric,
/// or `(0, 0, 0, 0)` when none does.
pub fn horizontal_window(
    lat_deg: &[f64],
    lon_deg: &[f64],
    ny: usize,
    nx: usize,
    site_lat_deg: f64,
    site_lon_deg: f64,
    earth_radius_m: f64,
    reach_m: f64,
) -> Result<(usize, usize, usize, usize)> {
    use rayon::prelude::*;
    if lat_deg.len() != ny * nx || lon_deg.len() != ny * nx {
        return Err(SuperobFailure::request("grid lat/lon do not match (ny, nx)"));
    }
    // np.radians and math.radians are both x * (pi / 180).
    let to_rad = std::f64::consts::PI / 180.0;
    let site_lat = site_lat_deg * to_rad;
    let site_lon = site_lon_deg * to_rad;
    let cos_site = site_lat.cos();
    let scale = 2.0 * earth_radius_m;
    // Per row: (any near, first col, last col).
    let rows: Vec<(bool, usize, usize)> = (0..ny)
        .into_par_iter()
        .map(|j| {
            let mut first = usize::MAX;
            let mut last = 0usize;
            for i in 0..nx {
                let lat = lat_deg[j * nx + i] * to_rad;
                let lon = lon_deg[j * nx + i] * to_rad;
                let s1 = ((lat - site_lat) / 2.0).sin();
                let s2 = ((lon - site_lon) / 2.0).sin();
                let a = s1 * s1 + cos_site * lat.cos() * (s2 * s2);
                let distance = scale * crate::num::clip(a, 0.0, 1.0).sqrt().asin();
                if distance <= reach_m {
                    if first == usize::MAX {
                        first = i;
                    }
                    last = i;
                }
            }
            (first != usize::MAX, first, last)
        })
        .collect();
    let mut j0 = None;
    let mut j1 = 0;
    let mut i0 = usize::MAX;
    let mut i1 = 0;
    for (j, (any, first, last)) in rows.iter().enumerate() {
        if *any {
            if j0.is_none() {
                j0 = Some(j);
            }
            j1 = j;
            i0 = i0.min(*first);
            i1 = i1.max(*last);
        }
    }
    Ok(match j0 {
        None => (0, 0, 0, 0),
        Some(j0) => (j0, j1, i0, i1),
    })
}
