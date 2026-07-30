/// Grid interpolation and regridding — matching wgrib2's regridding capabilities.
///
/// Supports nearest neighbor, bilinear, bicubic, and budget (area-weighted
/// conservative) interpolation for regridding between arbitrary source grids
/// (Lambert, polar stereographic, lat/lon, etc.) and regular lat/lon target grids.

use std::f64::consts::PI;

const DEG_TO_RAD: f64 = PI / 180.0;
const EARTH_RADIUS_KM: f64 = 6371.0;

// ============================================================
// GridSpec — target grid specification
// ============================================================

/// Target grid specification for regridding.
#[derive(Debug, Clone)]
pub struct GridSpec {
    pub nx: usize,
    pub ny: usize,
    /// First grid point latitude (degrees).
    pub lat1: f64,
    /// First grid point longitude (degrees).
    pub lon1: f64,
    /// Last grid point latitude (degrees).
    pub lat2: f64,
    /// Last grid point longitude (degrees).
    pub lon2: f64,
    /// Latitude increment (negative for N->S).
    pub dlat: f64,
    /// Longitude increment.
    pub dlon: f64,
}

impl GridSpec {
    /// Create a regular lat/lon grid from bounding box and resolution.
    ///
    /// Grid runs from `lat_min` (south) to `lat_max` (north), and
    /// `lon_min` (west) to `lon_max` (east) with uniform spacing.
    pub fn regular(lat_min: f64, lat_max: f64, lon_min: f64, lon_max: f64, resolution: f64) -> Self {
        let ny = ((lat_max - lat_min) / resolution).round() as usize + 1;
        let nx = ((lon_max - lon_min) / resolution).round() as usize + 1;
        let dlat = if ny > 1 { (lat_max - lat_min) / (ny - 1) as f64 } else { resolution };
        let dlon = if nx > 1 { (lon_max - lon_min) / (nx - 1) as f64 } else { resolution };
        Self {
            nx, ny,
            lat1: lat_min, lon1: lon_min,
            lat2: lat_max, lon2: lon_max,
            dlat, dlon,
        }
    }

    /// Create a regular lat/lon target grid from Lambert Conformal parameters.
    ///
    /// Computes the bounding box of the Lambert grid and creates a regular
    /// lat/lon grid covering that domain. The resolution is derived from the
    /// Lambert grid spacing at the reference latitude.
    pub fn from_lambert(
        nx: usize, ny: usize,
        lat1: f64, lon1: f64,
        dx: f64, dy: f64,
        latin1: f64, latin2: f64,
        lov: f64,
    ) -> Self {
        use wx_field::projection::LambertProjection;

        let proj = LambertProjection::new(
            latin1, latin2, lov,
            lat1, lon1,
            dx, dy,
            nx as u32, ny as u32,
        );

        use wx_field::projection::Projection;
        let (min_lat, min_lon, max_lat, max_lon) = proj.bounding_box();

        // Approximate resolution: dx in meters -> degrees at mid-latitude
        let mid_lat = (min_lat + max_lat) / 2.0;
        let res_lat = (dy / EARTH_RADIUS_KM / 1000.0) * (180.0 / PI);
        let res_lon = (dx / (EARTH_RADIUS_KM * 1000.0 * (mid_lat * DEG_TO_RAD).cos())) * (180.0 / PI);
        let resolution = res_lat.min(res_lon);

        Self::regular(min_lat, max_lat, min_lon, max_lon, resolution)
    }

    /// Generate the 1D latitude array for this grid.
    pub fn lats(&self) -> Vec<f64> {
        (0..self.ny).map(|j| self.lat1 + j as f64 * self.dlat).collect()
    }

    /// Generate the 1D longitude array for this grid.
    pub fn lons(&self) -> Vec<f64> {
        (0..self.nx).map(|i| self.lon1 + i as f64 * self.dlon).collect()
    }

    /// Total number of grid points.
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// Whether the grid has zero points.
    pub fn is_empty(&self) -> bool {
        self.nx == 0 || self.ny == 0
    }
}

// ============================================================
// Interpolation method
// ============================================================

/// Interpolation method for regridding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpMethod {
    /// Nearest neighbor — assigns the value of the closest source point.
    NearestNeighbor,
    /// Bilinear — weighted average of the 4 surrounding points.
    Bilinear,
    /// Bicubic — 4x4 stencil Catmull-Rom for smooth fields.
    Bicubic,
    /// Budget/Conservative — area-weighted averaging that conserves totals
    /// (use for precipitation, radiation fluxes).
    Budget,
}

impl InterpMethod {
    /// Parse from a string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "nearest" | "nearest_neighbor" | "nn" => Some(Self::NearestNeighbor),
            "bilinear" | "linear" => Some(Self::Bilinear),
            "bicubic" | "cubic" => Some(Self::Bicubic),
            "budget" | "conservative" | "area" => Some(Self::Budget),
            _ => None,
        }
    }
}

// ============================================================
// Source grid helpers
// ============================================================

/// Find the enclosing cell (i0, j0) for a target point in the source grid,
/// where (i0, j0) is the lower-left corner of the cell.
///
/// For regular grids (uniform spacing), this is a direct index calculation.
/// For irregular grids (Lambert, etc.), we do a brute-force search.
///
/// Returns `(i0, j0, frac_i, frac_j)` where frac is the fractional position
/// within the cell [0, 1].
fn find_enclosing_cell(
    src_lats: &[f64],
    src_lons: &[f64],
    src_nx: usize,
    src_ny: usize,
    target_lat: f64,
    target_lon: f64,
) -> Option<(usize, usize, f64, f64)> {
    // Try regular-grid fast path: check if source lats/lons are on a regular grid.
    // For 2D lat/lon arrays (Lambert, etc.), lats[j*nx + i] varies with both i and j,
    // so we check if the first row has constant latitude.
    let is_regular = src_lats.len() == src_ny
        || (src_lats.len() == src_nx * src_ny && is_regular_grid(src_lats, src_lons, src_nx, src_ny));

    if is_regular && src_lats.len() == src_ny {
        // 1D coordinate arrays
        return find_cell_regular_1d(src_lats, src_lons, target_lat, target_lon);
    }

    // 2D coordinate arrays — search for enclosing cell
    find_cell_2d(src_lats, src_lons, src_nx, src_ny, target_lat, target_lon)
}

/// Quick check if a 2D grid is actually regular (lat varies only with j, lon only with i).
fn is_regular_grid(lats: &[f64], lons: &[f64], nx: usize, ny: usize) -> bool {
    if ny < 2 || nx < 2 { return true; }
    // Check first two rows have same longitude pattern
    let tol = 1e-4;
    for i in 0..nx.min(5) {
        if (lons[i] - lons[nx + i]).abs() > tol { return false; }
    }
    // Check first two columns have same latitude pattern
    for j in 0..ny.min(5) {
        if (lats[j * nx] - lats[j * nx + 1]).abs() > tol {
            // Lat varies with i — not regular
            // Actually for lat/lon grids lat should NOT vary with i
            return false;
        }
    }
    true
}

/// Fast cell lookup for 1D regular coordinate arrays.
fn find_cell_regular_1d(
    lats: &[f64],  // length ny
    lons: &[f64],  // length nx
    target_lat: f64,
    target_lon: f64,
) -> Option<(usize, usize, f64, f64)> {
    let ny = lats.len();
    let nx = lons.len();
    if nx < 2 || ny < 2 { return None; }

    // Find j index in lats (may be ascending or descending)
    let (j0, frac_j) = find_index_1d(lats, target_lat)?;
    let (i0, frac_i) = find_index_1d(lons, target_lon)?;

    Some((i0, j0, frac_i, frac_j))
}

/// Binary-like search for index in a monotonic 1D array.
/// Returns (index, fraction) where fraction is in [0, 1].
fn find_index_1d(arr: &[f64], val: f64) -> Option<(usize, f64)> {
    let n = arr.len();
    if n < 2 { return None; }

    let ascending = arr[n - 1] > arr[0];

    if ascending {
        if val < arr[0] || val > arr[n - 1] { return None; }
        // Binary search
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if arr[mid] <= val { lo = mid; } else { hi = mid; }
        }
        let frac = (val - arr[lo]) / (arr[hi] - arr[lo]);
        Some((lo, frac))
    } else {
        // Descending
        if val > arr[0] || val < arr[n - 1] { return None; }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if arr[mid] >= val { lo = mid; } else { hi = mid; }
        }
        // For descending, frac is the distance from lo toward hi, in [0, 1].
        let frac = (arr[lo] - val) / (arr[lo] - arr[hi]);
        Some((lo, frac))
    }
}

/// Find enclosing cell in a 2D irregular grid by searching all cells.
fn find_cell_2d(
    lats: &[f64],
    lons: &[f64],
    nx: usize,
    ny: usize,
    target_lat: f64,
    target_lon: f64,
) -> Option<(usize, usize, f64, f64)> {
    let mut best_dist = f64::MAX;
    let mut best_i = 0usize;
    let mut best_j = 0usize;

    // First pass: find the nearest grid point
    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;
            let dlat = lats[idx] - target_lat;
            let dlon = lons[idx] - target_lon;
            let dist = dlat * dlat + dlon * dlon;
            if dist < best_dist {
                best_dist = dist;
                best_i = i;
                best_j = j;
            }
        }
    }

    // Now find the cell containing the point.
    // Search in the 3x3 neighborhood of the nearest point.
    let j_start = if best_j > 0 { best_j - 1 } else { 0 };
    let j_end = (best_j + 1).min(ny - 2);
    let i_start = if best_i > 0 { best_i - 1 } else { 0 };
    let i_end = (best_i + 1).min(nx - 2);

    for j in j_start..=j_end {
        for i in i_start..=i_end {
            // Cell corners: (i,j), (i+1,j), (i,j+1), (i+1,j+1)
            let idx00 = j * nx + i;
            let idx10 = j * nx + (i + 1);
            let idx01 = (j + 1) * nx + i;
            let idx11 = (j + 1) * nx + (i + 1);

            if let Some((fi, fj)) = point_in_quad(
                lats[idx00], lons[idx00],
                lats[idx10], lons[idx10],
                lats[idx01], lons[idx01],
                lats[idx11], lons[idx11],
                target_lat, target_lon,
            ) {
                return Some((i, j, fi, fj));
            }
        }
    }

    // If we couldn't find an enclosing cell, return the nearest point with frac=0
    if best_i < nx - 1 && best_j < ny - 1 {
        // Approximate fractional position
        let idx = best_j * nx + best_i;
        let idx_right = best_j * nx + best_i + 1;
        let idx_up = (best_j + 1) * nx + best_i;
        let dlon_cell = lons[idx_right] - lons[idx];
        let dlat_cell = lats[idx_up] - lats[idx];
        let fi = if dlon_cell.abs() > 1e-10 { (target_lon - lons[idx]) / dlon_cell } else { 0.0 };
        let fj = if dlat_cell.abs() > 1e-10 { (target_lat - lats[idx]) / dlat_cell } else { 0.0 };
        let fi = fi.clamp(0.0, 1.0);
        let fj = fj.clamp(0.0, 1.0);
        Some((best_i, best_j, fi, fj))
    } else {
        None
    }
}

/// Check if a point lies inside a quadrilateral defined by 4 corners,
/// and return the bilinear (s, t) coordinates if so.
fn point_in_quad(
    lat00: f64, lon00: f64,
    lat10: f64, lon10: f64,
    lat01: f64, lon01: f64,
    lat11: f64, lon11: f64,
    plat: f64, plon: f64,
) -> Option<(f64, f64)> {
    // Use iterative inverse bilinear mapping.
    // The forward mapping is:
    //   lat(s,t) = (1-s)(1-t)*lat00 + s(1-t)*lat10 + (1-s)t*lat01 + st*lat11
    //   lon(s,t) = (1-s)(1-t)*lon00 + s(1-t)*lon10 + (1-s)t*lon01 + st*lon11
    //
    // We solve for (s, t) given (plat, plon) using Newton iteration.
    let mut s = 0.5;
    let mut t = 0.5;

    for _ in 0..20 {
        let lat_st = (1.0 - s) * (1.0 - t) * lat00 + s * (1.0 - t) * lat10
                   + (1.0 - s) * t * lat01 + s * t * lat11;
        let lon_st = (1.0 - s) * (1.0 - t) * lon00 + s * (1.0 - t) * lon10
                   + (1.0 - s) * t * lon01 + s * t * lon11;

        let dlat = plat - lat_st;
        let dlon = plon - lon_st;

        if dlat.abs() < 1e-10 && dlon.abs() < 1e-10 {
            break;
        }

        // Jacobian
        let dlat_ds = -(1.0 - t) * lat00 + (1.0 - t) * lat10 - t * lat01 + t * lat11;
        let dlat_dt = -(1.0 - s) * lat00 - s * lat10 + (1.0 - s) * lat01 + s * lat11;
        let dlon_ds = -(1.0 - t) * lon00 + (1.0 - t) * lon10 - t * lon01 + t * lon11;
        let dlon_dt = -(1.0 - s) * lon00 - s * lon10 + (1.0 - s) * lon01 + s * lon11;

        let det = dlat_ds * dlon_dt - dlat_dt * dlon_ds;
        if det.abs() < 1e-20 {
            return None;
        }

        let ds = (dlat * dlon_dt - dlon * dlat_dt) / det;
        let dt = (dlon * dlat_ds - dlat * dlon_ds) / det;

        s += ds;
        t += dt;
    }

    if s >= -0.01 && s <= 1.01 && t >= -0.01 && t <= 1.01 {
        Some((s.clamp(0.0, 1.0), t.clamp(0.0, 1.0)))
    } else {
        None
    }
}

// ============================================================
// Nearest neighbor helper
// ============================================================

/// Find the source grid index nearest to (target_lat, target_lon).
fn find_nearest(
    src_lats: &[f64],
    src_lons: &[f64],
    src_nx: usize,
    _src_ny: usize,
    target_lat: f64,
    target_lon: f64,
) -> usize {
    let n = src_lats.len();
    let mut best_idx = 0;
    let mut best_dist = f64::MAX;

    for idx in 0..n {
        let dlat = src_lats[idx] - target_lat;
        let dlon = src_lons[idx] - target_lon;
        // Quick Euclidean in degrees (good enough for finding nearest)
        let dist = dlat * dlat + dlon * dlon;
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    // If src_lats is 1D (length == ny), convert to 2D index
    if n == _src_ny {
        // 1D arrays: find nearest lat and nearest lon independently
        let mut best_j = 0;
        let mut best_d = f64::MAX;
        for j in 0..n {
            let d = (src_lats[j] - target_lat).abs();
            if d < best_d { best_d = d; best_j = j; }
        }
        let mut best_i = 0;
        best_d = f64::MAX;
        for i in 0..src_nx {
            let d = (src_lons[i] - target_lon).abs();
            if d < best_d { best_d = d; best_i = i; }
        }
        return best_j * src_nx + best_i;
    }

    best_idx
}

// ============================================================
// Bilinear interpolation at a single point
// ============================================================

fn bilinear_at(
    values: &[f64],
    src_lats: &[f64],
    src_lons: &[f64],
    src_nx: usize,
    src_ny: usize,
    target_lat: f64,
    target_lon: f64,
) -> f64 {
    // For 1D coordinate arrays
    if src_lats.len() == src_ny && src_lons.len() == src_nx {
        let j_res = find_index_1d(src_lats, target_lat);
        let i_res = find_index_1d(src_lons, target_lon);
        if let (Some((i0, fi)), Some((j0, fj))) = (i_res, j_res) {
            let i1 = (i0 + 1).min(src_nx - 1);
            let j1 = (j0 + 1).min(src_ny - 1);
            let v00 = values[j0 * src_nx + i0];
            let v10 = values[j0 * src_nx + i1];
            let v01 = values[j1 * src_nx + i0];
            let v11 = values[j1 * src_nx + i1];
            if v00.is_nan() || v10.is_nan() || v01.is_nan() || v11.is_nan() {
                return f64::NAN;
            }
            return (1.0 - fi) * (1.0 - fj) * v00
                 + fi * (1.0 - fj) * v10
                 + (1.0 - fi) * fj * v01
                 + fi * fj * v11;
        }
        return f64::NAN;
    }

    // For 2D coordinate arrays
    if let Some((i0, j0, fi, fj)) = find_enclosing_cell(src_lats, src_lons, src_nx, src_ny, target_lat, target_lon) {
        let i1 = (i0 + 1).min(src_nx - 1);
        let j1 = (j0 + 1).min(src_ny - 1);
        let v00 = values[j0 * src_nx + i0];
        let v10 = values[j0 * src_nx + i1];
        let v01 = values[j1 * src_nx + i0];
        let v11 = values[j1 * src_nx + i1];
        if v00.is_nan() || v10.is_nan() || v01.is_nan() || v11.is_nan() {
            return f64::NAN;
        }
        (1.0 - fi) * (1.0 - fj) * v00
            + fi * (1.0 - fj) * v10
            + (1.0 - fi) * fj * v01
            + fi * fj * v11
    } else {
        f64::NAN
    }
}

// ============================================================
// Bicubic interpolation (Catmull-Rom)
// ============================================================

/// Catmull-Rom basis function.
fn catmull_rom(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2,
    ]
}

fn bicubic_at(
    values: &[f64],
    src_lats: &[f64],
    src_lons: &[f64],
    src_nx: usize,
    src_ny: usize,
    target_lat: f64,
    target_lon: f64,
) -> f64 {
    // Find cell using 1D or 2D lookup
    let (i0, j0, fi, fj);

    if src_lats.len() == src_ny && src_lons.len() == src_nx {
        let i_res = find_index_1d(src_lons, target_lon);
        let j_res = find_index_1d(src_lats, target_lat);
        if let (Some((ii, ffi)), Some((jj, ffj))) = (i_res, j_res) {
            i0 = ii; j0 = jj; fi = ffi; fj = ffj;
        } else {
            return f64::NAN;
        }
    } else if let Some((ii, jj, ffi, ffj)) = find_enclosing_cell(src_lats, src_lons, src_nx, src_ny, target_lat, target_lon) {
        i0 = ii; j0 = jj; fi = ffi; fj = ffj;
    } else {
        return f64::NAN;
    }

    // Need a 4x4 stencil centered on the cell: indices [i0-1..i0+2] x [j0-1..j0+2]
    if i0 < 1 || j0 < 1 || i0 + 2 >= src_nx || j0 + 2 >= src_ny {
        // Not enough room for 4x4 stencil — fall back to bilinear
        return bilinear_at(values, src_lats, src_lons, src_nx, src_ny, target_lat, target_lon);
    }

    let wx = catmull_rom(fi);
    let wy = catmull_rom(fj);

    let mut result = 0.0;
    for dj in 0..4 {
        let jj = j0 - 1 + dj;
        for di in 0..4 {
            let ii = i0 - 1 + di;
            let v = values[jj * src_nx + ii];
            if v.is_nan() {
                return f64::NAN;
            }
            result += wx[di] * wy[dj] * v;
        }
    }
    result
}

// ============================================================
// Budget / Conservative interpolation
// ============================================================

fn budget_at(
    values: &[f64],
    src_lats: &[f64],
    src_lons: &[f64],
    src_nx: usize,
    src_ny: usize,
    target_lat: f64,
    target_lon: f64,
    target_dlat: f64,
    target_dlon: f64,
) -> f64 {
    // Area-weighted average of all source points whose centers fall within
    // the target cell boundaries.
    let lat_lo = target_lat - target_dlat.abs() / 2.0;
    let lat_hi = target_lat + target_dlat.abs() / 2.0;
    let lon_lo = target_lon - target_dlon.abs() / 2.0;
    let lon_hi = target_lon + target_dlon.abs() / 2.0;

    let is_1d = src_lats.len() == src_ny && src_lons.len() == src_nx;

    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    if is_1d {
        // Fast path for regular grids
        let j_start = src_lats.iter().position(|&lat| lat >= lat_lo.min(lat_hi)).unwrap_or(0);
        let j_end_val = lat_lo.max(lat_hi);

        for j in j_start..src_ny {
            let slat = src_lats[j];
            if (src_lats[0] < src_lats[src_ny - 1] && slat > j_end_val) ||
               (src_lats[0] > src_lats[src_ny - 1] && slat < lat_lo.min(lat_hi)) {
                break;
            }
            if slat < lat_lo.min(lat_hi) || slat > j_end_val {
                continue;
            }
            // Weight by cos(lat) for area correction
            let w_lat = (slat * DEG_TO_RAD).cos();

            for i in 0..src_nx {
                let slon = src_lons[i];
                if slon >= lon_lo && slon <= lon_hi {
                    let v = values[j * src_nx + i];
                    if !v.is_nan() {
                        weighted_sum += v * w_lat;
                        total_weight += w_lat;
                    }
                }
            }
        }
    } else {
        // 2D coordinate arrays
        for j in 0..src_ny {
            for i in 0..src_nx {
                let idx = j * src_nx + i;
                let slat = src_lats[idx];
                let slon = src_lons[idx];
                if slat >= lat_lo.min(lat_hi) && slat <= lat_lo.max(lat_hi)
                    && slon >= lon_lo && slon <= lon_hi
                {
                    let v = values[idx];
                    if !v.is_nan() {
                        let w = (slat * DEG_TO_RAD).cos();
                        weighted_sum += v * w;
                        total_weight += w;
                    }
                }
            }
        }
    }

    if total_weight > 0.0 {
        weighted_sum / total_weight
    } else {
        // No source points in cell — fall back to bilinear
        bilinear_at(values, src_lats, src_lons, src_nx, src_ny, target_lat, target_lon)
    }
}

// ============================================================
// Main regrid function
// ============================================================

/// Regrid data from a source grid to a target regular lat/lon grid.
///
/// # Arguments
///
/// * `src_values` — flattened source data `[ny][nx]`, row-major.
/// * `src_lats` — source latitudes. Either 1D `[ny]` or 2D `[ny*nx]`.
/// * `src_lons` — source longitudes. Either 1D `[nx]` or 2D `[ny*nx]`.
/// * `src_nx`, `src_ny` — source grid dimensions.
/// * `target` — target grid specification.
/// * `method` — interpolation method.
///
/// # Returns
///
/// Flattened `Vec<f64>` of length `target.ny * target.nx`, row-major.
/// Points outside the source domain are set to `NaN`.
pub fn regrid(
    src_values: &[f64],
    src_lats: &[f64],
    src_lons: &[f64],
    src_nx: usize,
    src_ny: usize,
    target: &GridSpec,
    method: InterpMethod,
) -> Vec<f64> {
    let mut result = vec![f64::NAN; target.ny * target.nx];

    for tj in 0..target.ny {
        let tlat = target.lat1 + tj as f64 * target.dlat;
        for ti in 0..target.nx {
            let tlon = target.lon1 + ti as f64 * target.dlon;
            let idx = tj * target.nx + ti;

            result[idx] = match method {
                InterpMethod::NearestNeighbor => {
                    let src_idx = find_nearest(src_lats, src_lons, src_nx, src_ny, tlat, tlon);
                    src_values[src_idx]
                }
                InterpMethod::Bilinear => {
                    bilinear_at(src_values, src_lats, src_lons, src_nx, src_ny, tlat, tlon)
                }
                InterpMethod::Bicubic => {
                    bicubic_at(src_values, src_lats, src_lons, src_nx, src_ny, tlat, tlon)
                }
                InterpMethod::Budget => {
                    budget_at(src_values, src_lats, src_lons, src_nx, src_ny, tlat, tlon,
                              target.dlat, target.dlon)
                }
            };
        }
    }

    result
}

// ============================================================
// Point interpolation
// ============================================================

/// Interpolate a gridded field to a single lat/lon point.
///
/// # Arguments
///
/// * `values` — flattened grid data `[ny][nx]`.
/// * `lats` — latitudes (1D `[ny]` or 2D `[ny*nx]`).
/// * `lons` — longitudes (1D `[nx]` or 2D `[ny*nx]`).
/// * `nx`, `ny` — grid dimensions.
/// * `target_lat`, `target_lon` — point to interpolate to.
/// * `method` — interpolation method (Budget falls back to bilinear for single points).
pub fn interpolate_point(
    values: &[f64],
    lats: &[f64],
    lons: &[f64],
    nx: usize,
    ny: usize,
    target_lat: f64,
    target_lon: f64,
    method: InterpMethod,
) -> f64 {
    match method {
        InterpMethod::NearestNeighbor => {
            let idx = find_nearest(lats, lons, nx, ny, target_lat, target_lon);
            values[idx]
        }
        InterpMethod::Bilinear | InterpMethod::Budget => {
            bilinear_at(values, lats, lons, nx, ny, target_lat, target_lon)
        }
        InterpMethod::Bicubic => {
            bicubic_at(values, lats, lons, nx, ny, target_lat, target_lon)
        }
    }
}

/// Interpolate to multiple points (e.g., station locations or a cross-section path).
pub fn interpolate_points(
    values: &[f64],
    lats: &[f64],
    lons: &[f64],
    nx: usize,
    ny: usize,
    target_lats: &[f64],
    target_lons: &[f64],
    method: InterpMethod,
) -> Vec<f64> {
    assert_eq!(target_lats.len(), target_lons.len());
    target_lats
        .iter()
        .zip(target_lons.iter())
        .map(|(&tlat, &tlon)| interpolate_point(values, lats, lons, nx, ny, tlat, tlon, method))
        .collect()
}

// ============================================================
// Cross-section extraction
// ============================================================

/// Great-circle distance between two points in km.
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1r = lat1 * DEG_TO_RAD;
    let lat2r = lat2 * DEG_TO_RAD;
    let dlat = (lat2 - lat1) * DEG_TO_RAD;
    let dlon = (lon2 - lon1) * DEG_TO_RAD;
    let a = (dlat / 2.0).sin().powi(2) + lat1r.cos() * lat2r.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    EARTH_RADIUS_KM * c
}

/// Compute intermediate point on a great circle at fraction `f` from start to end.
fn great_circle_intermediate(
    lat1: f64, lon1: f64,
    lat2: f64, lon2: f64,
    f: f64,
) -> (f64, f64) {
    let lat1r = lat1 * DEG_TO_RAD;
    let lon1r = lon1 * DEG_TO_RAD;
    let lat2r = lat2 * DEG_TO_RAD;
    let lon2r = lon2 * DEG_TO_RAD;

    let d = {
        let dlat = lat2r - lat1r;
        let dlon = lon2r - lon1r;
        let a = (dlat / 2.0).sin().powi(2) + lat1r.cos() * lat2r.cos() * (dlon / 2.0).sin().powi(2);
        2.0 * a.sqrt().asin()
    };

    if d.abs() < 1e-12 {
        return (lat1, lon1);
    }

    let a = ((1.0 - f) * d).sin() / d.sin();
    let b = (f * d).sin() / d.sin();

    let x = a * lat1r.cos() * lon1r.cos() + b * lat2r.cos() * lon2r.cos();
    let y = a * lat1r.cos() * lon1r.sin() + b * lat2r.cos() * lon2r.sin();
    let z = a * lat1r.sin() + b * lat2r.sin();

    let lat = z.atan2((x * x + y * y).sqrt()) / DEG_TO_RAD;
    let lon = y.atan2(x) / DEG_TO_RAD;
    (lat, lon)
}

/// Extract a cross-section along a great-circle path between two points.
///
/// Returns `(interpolated_values, distances_km)` where `distances_km[i]` is
/// the distance in km from `start` to the i-th sample point.
pub fn cross_section_data(
    values: &[f64],
    lats: &[f64],
    lons: &[f64],
    nx: usize,
    ny: usize,
    start: (f64, f64),
    end: (f64, f64),
    n_points: usize,
    method: InterpMethod,
) -> (Vec<f64>, Vec<f64>) {
    let total_dist = haversine_km(start.0, start.1, end.0, end.1);

    let mut interp_values = Vec::with_capacity(n_points);
    let mut distances = Vec::with_capacity(n_points);

    for k in 0..n_points {
        let f = if n_points > 1 { k as f64 / (n_points - 1) as f64 } else { 0.0 };
        let (lat, lon) = great_circle_intermediate(start.0, start.1, end.0, end.1, f);
        let v = interpolate_point(values, lats, lons, nx, ny, lat, lon, method);
        interp_values.push(v);
        distances.push(f * total_dist);
    }

    (interp_values, distances)
}

// ============================================================
// Vertical interpolation
// ============================================================

/// Interpolate a 3D field to a specific pressure or height level.
///
/// # Arguments
///
/// * `values_3d` — flattened `[nz][ny][nx]` data (level 0 first).
/// * `levels` — the vertical coordinate at each level (e.g., pressure in hPa
///   or height in m). Must be length `nz`. Values should be monotonically
///   increasing or decreasing.
/// * `target_level` — the level to interpolate to.
/// * `nx`, `ny`, `nz` — grid dimensions.
/// * `log_interp` — if `true`, use log-linear interpolation (appropriate for
///   pressure coordinates). If `false`, use linear interpolation (appropriate
///   for height coordinates).
///
/// # Returns
///
/// Flattened `[ny][nx]` array of interpolated values at the target level.
pub fn interpolate_vertical(
    values_3d: &[f64],
    levels: &[f64],
    target_level: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    log_interp: bool,
) -> Vec<f64> {
    assert_eq!(values_3d.len(), nz * ny * nx, "values_3d length mismatch");
    assert_eq!(levels.len(), nz, "levels length mismatch");

    let mut result = vec![f64::NAN; ny * nx];

    // Determine if levels are ascending or descending
    let ascending = nz >= 2 && levels[nz - 1] > levels[0];

    // Find the bracketing levels
    let bracket = if ascending {
        find_bracket_ascending(levels, target_level)
    } else {
        find_bracket_descending(levels, target_level)
    };

    let (k0, k1) = match bracket {
        Some(b) => b,
        None => return result, // target outside range
    };

    let l0 = levels[k0];
    let l1 = levels[k1];

    // Compute interpolation weight
    let w = if log_interp {
        if l0 <= 0.0 || l1 <= 0.0 || target_level <= 0.0 {
            // Can't take log of non-positive — fall back to linear
            (target_level - l0) / (l1 - l0)
        } else {
            (target_level.ln() - l0.ln()) / (l1.ln() - l0.ln())
        }
    } else {
        (target_level - l0) / (l1 - l0)
    };

    let slab_size = ny * nx;
    let offset0 = k0 * slab_size;
    let offset1 = k1 * slab_size;

    for idx in 0..slab_size {
        let v0 = values_3d[offset0 + idx];
        let v1 = values_3d[offset1 + idx];
        if v0.is_nan() || v1.is_nan() {
            result[idx] = f64::NAN;
        } else {
            result[idx] = v0 + w * (v1 - v0);
        }
    }

    result
}

fn find_bracket_ascending(levels: &[f64], target: f64) -> Option<(usize, usize)> {
    let n = levels.len();
    if n < 2 || target < levels[0] || target > levels[n - 1] { return None; }
    for k in 0..n - 1 {
        if levels[k] <= target && target <= levels[k + 1] {
            return Some((k, k + 1));
        }
    }
    None
}

fn find_bracket_descending(levels: &[f64], target: f64) -> Option<(usize, usize)> {
    let n = levels.len();
    if n < 2 || target > levels[0] || target < levels[n - 1] { return None; }
    for k in 0..n - 1 {
        if levels[k] >= target && target >= levels[k + 1] {
            return Some((k, k + 1));
        }
    }
    None
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_exact_on_linear_field() {
        // A linear field f(x,y) = 2*x + 3*y should be reproduced exactly
        // by bilinear interpolation.
        let nx = 10;
        let ny = 10;
        let lats: Vec<f64> = (0..ny).map(|j| j as f64).collect();
        let lons: Vec<f64> = (0..nx).map(|i| i as f64).collect();
        let values: Vec<f64> = (0..ny)
            .flat_map(|j| (0..nx).map(move |i| 2.0 * i as f64 + 3.0 * j as f64))
            .collect();

        // Interpolate at point (3.5, 4.7)
        let v = interpolate_point(&values, &lats, &lons, nx, ny, 4.7, 3.5, InterpMethod::Bilinear);
        let expected = 2.0 * 3.5 + 3.0 * 4.7;
        assert!((v - expected).abs() < 1e-10, "got {}, expected {}", v, expected);

        // Another point
        let v2 = interpolate_point(&values, &lats, &lons, nx, ny, 1.2, 7.8, InterpMethod::Bilinear);
        let expected2 = 2.0 * 7.8 + 3.0 * 1.2;
        assert!((v2 - expected2).abs() < 1e-10, "got {}, expected {}", v2, expected2);
    }

    #[test]
    fn test_nearest_neighbor() {
        let nx = 5;
        let ny = 5;
        let lats: Vec<f64> = (0..ny).map(|j| j as f64 * 10.0).collect();
        let lons: Vec<f64> = (0..nx).map(|i| i as f64 * 10.0).collect();
        let values: Vec<f64> = (0..ny * nx).map(|i| i as f64).collect();

        // Point (12, 18) should be nearest to grid point (1, 2) = index 2*5+1 = 11
        let v = interpolate_point(&values, &lats, &lons, nx, ny, 18.0, 12.0, InterpMethod::NearestNeighbor);
        assert_eq!(v, 11.0, "nearest neighbor: got {}", v);
    }

    #[test]
    fn test_regrid_identity() {
        // Regridding to the same grid should approximately preserve values.
        let nx = 5;
        let ny = 5;
        let lats: Vec<f64> = (0..ny).map(|j| 30.0 + j as f64).collect();
        let lons: Vec<f64> = (0..nx).map(|i| -100.0 + i as f64).collect();
        let values: Vec<f64> = (0..ny * nx).map(|i| (i as f64) * 1.5).collect();

        let target = GridSpec::regular(30.0, 34.0, -100.0, -96.0, 1.0);

        let result = regrid(&values, &lats, &lons, nx, ny, &target, InterpMethod::Bilinear);

        // At exact grid points, bilinear should match exactly
        for j in 0..ny {
            for i in 0..nx {
                let src_val = values[j * nx + i];
                let dst_val = result[j * target.nx + i];
                assert!(
                    (src_val - dst_val).abs() < 1e-10,
                    "mismatch at ({},{}): src={}, dst={}", i, j, src_val, dst_val
                );
            }
        }
    }

    #[test]
    fn test_regrid_lambert_to_latlon() {
        // Create a Lambert grid and regrid to lat/lon.
        use wx_field::projection::{LambertProjection, Projection};

        let src_nx = 20;
        let src_ny = 20;
        let proj = LambertProjection::new(
            33.0, 45.0, -97.0,
            35.0, -100.0,
            10000.0, 10000.0,
            src_nx, src_ny,
        );

        // Generate 2D lat/lon arrays for the Lambert grid
        let mut src_lats = vec![0.0; (src_nx * src_ny) as usize];
        let mut src_lons = vec![0.0; (src_nx * src_ny) as usize];
        for j in 0..src_ny as usize {
            for i in 0..src_nx as usize {
                let (lat, lon) = proj.grid_to_latlon(i as f64, j as f64);
                src_lats[j * src_nx as usize + i] = lat;
                src_lons[j * src_nx as usize + i] = lon;
            }
        }

        // Create a test field: temperature = 300 - 0.5 * (lat - 35)^2
        let values: Vec<f64> = src_lats.iter().map(|&lat| 300.0 - 0.5 * (lat - 35.0).powi(2)).collect();

        let (min_lat, min_lon, max_lat, max_lon) = proj.bounding_box();
        let target = GridSpec::regular(
            min_lat.ceil(), max_lat.floor(),
            min_lon.ceil(), max_lon.floor(),
            0.1,
        );

        let result = regrid(
            &values, &src_lats, &src_lons,
            src_nx as usize, src_ny as usize,
            &target, InterpMethod::Bilinear,
        );

        // Check that non-NaN values are within reasonable range
        let valid: Vec<f64> = result.iter().filter(|v| !v.is_nan()).copied().collect();
        assert!(!valid.is_empty(), "all values are NaN");
        let min_val = valid.iter().copied().fold(f64::MAX, f64::min);
        let max_val = valid.iter().copied().fold(f64::MIN, f64::max);
        assert!(min_val > 290.0, "min too low: {}", min_val);
        assert!(max_val < 305.0, "max too high: {}", max_val);
    }

    #[test]
    fn test_point_interpolation_accuracy() {
        // On a quadratic field, bilinear won't be exact, but should be close.
        let nx = 100;
        let ny = 100;
        let lats: Vec<f64> = (0..ny).map(|j| 30.0 + j as f64 * 0.1).collect();
        let lons: Vec<f64> = (0..nx).map(|i| -100.0 + i as f64 * 0.1).collect();

        // Linear field: exact with bilinear
        let mut values = Vec::with_capacity(ny * nx);
        for j in 0..ny {
            for i in 0..nx {
                values.push(lats[j] * 2.0 + lons[i] * 3.0);
            }
        }

        let test_lat = 35.55;
        let test_lon = -95.23;
        let v = interpolate_point(&values, &lats, &lons, nx, ny, test_lat, test_lon, InterpMethod::Bilinear);
        let expected = test_lat * 2.0 + test_lon * 3.0;
        assert!((v - expected).abs() < 1e-8, "got {}, expected {}", v, expected);
    }

    #[test]
    fn test_outside_domain_returns_nan() {
        let nx = 5;
        let ny = 5;
        let lats: Vec<f64> = (0..ny).map(|j| 30.0 + j as f64).collect();
        let lons: Vec<f64> = (0..nx).map(|i| -100.0 + i as f64).collect();
        let values: Vec<f64> = vec![1.0; nx * ny];

        // Point clearly outside domain
        let v = interpolate_point(&values, &lats, &lons, nx, ny, 50.0, -90.0, InterpMethod::Bilinear);
        assert!(v.is_nan(), "expected NaN for outside point, got {}", v);
    }

    #[test]
    fn test_nan_handling() {
        let nx = 5;
        let ny = 5;
        let lats: Vec<f64> = (0..ny).map(|j| j as f64).collect();
        let lons: Vec<f64> = (0..nx).map(|i| i as f64).collect();
        let mut values: Vec<f64> = vec![1.0; nx * ny];
        // Set one corner of the interpolation cell to NaN
        values[1 * nx + 1] = f64::NAN;

        let v = interpolate_point(&values, &lats, &lons, nx, ny, 0.5, 0.5, InterpMethod::Bilinear);
        assert!(v.is_nan(), "expected NaN when source has NaN, got {}", v);
    }

    #[test]
    #[should_panic]
    fn test_interpolate_points_target_length_mismatch_panics() {
        let nx = 5;
        let ny = 5;
        let lats: Vec<f64> = (0..ny).map(|j| 30.0 + j as f64).collect();
        let lons: Vec<f64> = (0..nx).map(|i| -100.0 + i as f64).collect();
        let values: Vec<f64> = vec![1.0; nx * ny];

        let _ = interpolate_points(
            &values,
            &lats,
            &lons,
            nx,
            ny,
            &[31.0, 32.0],
            &[-99.0],
            InterpMethod::Bilinear,
        );
    }

    #[test]
    fn test_cross_section() {
        let nx = 50;
        let ny = 50;
        let lats: Vec<f64> = (0..ny).map(|j| 30.0 + j as f64 * 0.2).collect();
        let lons: Vec<f64> = (0..nx).map(|i| -100.0 + i as f64 * 0.2).collect();
        let mut values = Vec::with_capacity(ny * nx);
        for j in 0..ny {
            for i in 0..nx {
                values.push(lats[j] + lons[i]);
            }
        }

        let (vals, dists) = cross_section_data(
            &values, &lats, &lons, nx, ny,
            (35.0, -95.0), (38.0, -92.0),
            20, InterpMethod::Bilinear,
        );

        assert_eq!(vals.len(), 20);
        assert_eq!(dists.len(), 20);
        assert!((dists[0] - 0.0).abs() < 1e-10);
        assert!(dists[19] > 0.0);

        // First point should be close to 35.0 + (-95.0) = -60.0
        assert!((vals[0] - (-60.0)).abs() < 0.1, "first cross-section val: {}", vals[0]);
        // Last point should be close to 38.0 + (-92.0) = -54.0
        assert!((vals[19] - (-54.0)).abs() < 0.1, "last cross-section val: {}", vals[19]);
    }

    #[test]
    fn test_vertical_interpolation_linear() {
        let nx = 3;
        let ny = 3;
        let nz = 4;

        // Height levels: 0, 1000, 2000, 3000 m
        let levels = vec![0.0, 1000.0, 2000.0, 3000.0];

        // Temperature decreasing with height: T = 300 - 0.006 * z
        let values_3d: Vec<f64> = (0..nz)
            .flat_map(|k| {
                let t = 300.0 - 0.006 * levels[k];
                vec![t; ny * nx]
            })
            .collect();

        // Interpolate to 1500m
        let result = interpolate_vertical(&values_3d, &levels, 1500.0, nx, ny, nz, false);
        let expected = 300.0 - 0.006 * 1500.0; // 291.0
        assert!((result[0] - expected).abs() < 1e-10, "got {}, expected {}", result[0], expected);
    }

    #[test]
    fn test_vertical_interpolation_log_pressure() {
        let nx = 2;
        let ny = 2;
        let nz = 3;

        // Pressure levels (descending): 1000, 500, 250 hPa
        let levels = vec![1000.0, 500.0, 250.0];

        // Geopotential height increasing with decreasing pressure
        let values_3d: Vec<f64> = vec![
            // 1000 hPa
            100.0, 100.0, 100.0, 100.0,
            // 500 hPa
            5500.0, 5500.0, 5500.0, 5500.0,
            // 250 hPa
            10500.0, 10500.0, 10500.0, 10500.0,
        ];

        let result = interpolate_vertical(&values_3d, &levels, 700.0, nx, ny, nz, true);
        // With log interpolation between 1000 and 500 hPa at 700 hPa
        let w = (700.0_f64.ln() - 1000.0_f64.ln()) / (500.0_f64.ln() - 1000.0_f64.ln());
        let expected = 100.0 + w * (5500.0 - 100.0);
        assert!((result[0] - expected).abs() < 1.0, "got {}, expected {}", result[0], expected);
    }

    #[test]
    fn test_gridspec_regular() {
        let g = GridSpec::regular(30.0, 40.0, -100.0, -90.0, 0.5);
        assert_eq!(g.ny, 21);
        assert_eq!(g.nx, 21);
        assert!((g.dlat - 0.5).abs() < 1e-10);
        assert!((g.dlon - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bicubic_on_linear_field() {
        // Bicubic should also be exact on linear fields.
        let nx = 10;
        let ny = 10;
        let lats: Vec<f64> = (0..ny).map(|j| j as f64).collect();
        let lons: Vec<f64> = (0..nx).map(|i| i as f64).collect();
        let values: Vec<f64> = (0..ny)
            .flat_map(|j| (0..nx).map(move |i| 2.0 * i as f64 + 3.0 * j as f64))
            .collect();

        // Point well inside (away from edges for 4x4 stencil)
        let v = interpolate_point(&values, &lats, &lons, nx, ny, 4.3, 5.7, InterpMethod::Bicubic);
        let expected = 2.0 * 5.7 + 3.0 * 4.3;
        assert!((v - expected).abs() < 1e-8, "bicubic linear: got {}, expected {}", v, expected);
    }

    #[test]
    fn test_interp_method_from_str() {
        assert_eq!(InterpMethod::from_str_loose("bilinear"), Some(InterpMethod::Bilinear));
        assert_eq!(InterpMethod::from_str_loose("nearest"), Some(InterpMethod::NearestNeighbor));
        assert_eq!(InterpMethod::from_str_loose("NN"), Some(InterpMethod::NearestNeighbor));
        assert_eq!(InterpMethod::from_str_loose("cubic"), Some(InterpMethod::Bicubic));
        assert_eq!(InterpMethod::from_str_loose("budget"), Some(InterpMethod::Budget));
        assert_eq!(InterpMethod::from_str_loose("conservative"), Some(InterpMethod::Budget));
        assert_eq!(InterpMethod::from_str_loose("unknown"), None);
    }
}
