//! Interpolation module -- regridding, 1-D interpolation, vertical interpolation.
//!
//! Re-exports from `wx_math::regrid` and provides convenience wrappers
//! matching common MetPy / NumPy patterns.

// ── Re-exports from wx-math regrid ──────────────────────────────────
pub use wx_math::regrid::{GridSpec, InterpMethod};
pub use wx_math::regrid::{
    regrid, interpolate_point, interpolate_points,
    cross_section_data, interpolate_vertical,
};

// ── Re-exports from wx-math interpolate ─────────────────────────────
pub use wx_math::interpolate::inverse_distance_to_points;

// ── Convenience wrapper ─────────────────────────────────────────────

/// Interpolate scattered values onto a regular lat/lon grid.
///
/// This is a convenience wrapper around [`regrid`] for the common case where
/// the source data is a flat array with corresponding 1-D lat/lon arrays
/// (i.e., `src_lats.len() == src_lons.len() == values.len()`).
///
/// Returns a `Vec<f64>` of length `target.nx * target.ny` in row-major order.
pub fn interpolate_to_grid(
    values: &[f64],
    src_lats: &[f64],
    src_lons: &[f64],
    target: &GridSpec,
    method: InterpMethod,
) -> Vec<f64> {
    // Determine source grid dimensions.  If the source is a 2-D grid with
    // matching lat/lon arrays we infer nx from the coordinate layout;
    // otherwise treat it as a 1-D list (nx = len, ny = 1).
    let n = values.len();
    let (src_nx, src_ny) = if src_lats.len() == n && src_lons.len() == n {
        // Try to infer nx: find the first index where longitude wraps or repeats
        let mut nx = n;
        if n > 1 {
            for i in 1..n {
                if (src_lons[i] - src_lons[0]).abs() < 1e-8 && i > 1 {
                    nx = i;
                    break;
                }
            }
        }
        let ny = if nx > 0 { n / nx } else { 1 };
        (nx, ny)
    } else {
        (n, 1)
    };

    regrid(values, src_lats, src_lons, src_nx, src_ny, target, method)
}

// ── 1-D linear interpolation (like numpy.interp) ────────────────────

/// Piecewise linear interpolation (MetPy-compatible).
///
/// Given monotonically increasing breakpoints `xp` with values `fp`,
/// evaluate the piecewise-linear interpolant at each point in `x`.
/// Values outside `[xp[0], xp[last]]` return `NaN` (no extrapolation),
/// matching MetPy's behavior.
///
/// # Panics
/// Panics if `xp` and `fp` have different lengths or are empty.
pub fn interpolate_1d(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    assert_eq!(xp.len(), fp.len(), "xp and fp must have the same length");
    assert!(!xp.is_empty(), "xp must not be empty");

    let n = xp.len();
    x.iter()
        .map(|&xi| {
            if xi < xp[0] || xi > xp[n - 1] {
                f64::NAN
            } else if xi == xp[0] {
                fp[0]
            } else if xi == xp[n - 1] {
                fp[n - 1]
            } else {
                // Binary search for the enclosing interval
                let mut lo = 0usize;
                let mut hi = n - 1;
                while hi - lo > 1 {
                    let mid = (lo + hi) / 2;
                    if xp[mid] <= xi {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let t = (xi - xp[lo]) / (xp[hi] - xp[lo]);
                fp[lo] + t * (fp[hi] - fp[lo])
            }
        })
        .collect()
}

/// Interpolation in log-pressure space, matching MetPy's `log_interpolate_1d`.
///
/// Performs linear interpolation in `ln(x)` space, which is the correct
/// approach for interpolating meteorological variables with respect to
/// pressure (pressure decreases roughly exponentially with height).
///
/// `xp` must be monotonic (ascending or descending). If descending (typical
/// for pressure coordinates), the arrays are internally reversed before
/// interpolation.
///
/// # Panics
/// Panics if `xp` and `fp` differ in length, are empty, or contain non-positive values.
pub fn log_interpolate_1d(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    assert_eq!(xp.len(), fp.len(), "xp and fp must have the same length");
    assert!(!xp.is_empty(), "xp must not be empty");

    // Work with log(x) values
    let log_x: Vec<f64> = x.iter().map(|&v| v.ln()).collect();

    // Ensure ascending order for the breakpoints
    let (log_xp, fp_sorted): (Vec<f64>, Vec<f64>) = if xp.len() >= 2 && xp[0] > xp[xp.len() - 1]
    {
        // Descending -- reverse
        let lxp: Vec<f64> = xp.iter().rev().map(|&v| v.ln()).collect();
        let fps: Vec<f64> = fp.iter().rev().copied().collect();
        (lxp, fps)
    } else {
        let lxp: Vec<f64> = xp.iter().map(|&v| v.ln()).collect();
        (lxp, fp.to_vec())
    };

    interpolate_1d(&log_x, &log_xp, &fp_sorted)
}

// ── NaN interpolation / filtering / geodesic ─────────────────────

/// Fill NaN values in a 1-D slice by linearly interpolating between
/// surrounding valid points.  Edge NaNs are filled with the nearest
/// valid value.  If all values are NaN the slice is left unchanged.
pub fn interpolate_nans_1d(values: &mut [f64]) {
    let n = values.len();
    if n == 0 {
        return;
    }

    // Collect indices of valid (non-NaN) entries.
    let valid: Vec<usize> = (0..n).filter(|&i| !values[i].is_nan()).collect();
    if valid.is_empty() {
        return;
    }

    // Fill leading NaNs with the first valid value.
    let first_valid = valid[0];
    let first_val = values[first_valid];
    for v in values.iter_mut().take(first_valid) {
        *v = first_val;
    }

    // Fill trailing NaNs with the last valid value.
    let last_valid = *valid.last().unwrap();
    let last_val = values[last_valid];
    for v in values.iter_mut().skip(last_valid + 1) {
        *v = last_val;
    }

    // Linearly interpolate interior NaN gaps.
    for win in valid.windows(2) {
        let (lo, hi) = (win[0], win[1]);
        if hi - lo > 1 {
            let v_lo = values[lo];
            let v_hi = values[hi];
            for k in (lo + 1)..hi {
                let t = (k - lo) as f64 / (hi - lo) as f64;
                values[k] = v_lo + t * (v_hi - v_lo);
            }
        }
    }
}

/// Interpolate a 3-D field to an isosurface of another 3-D field.
///
/// For each `(i, j)` column the function walks upward through the
/// `nz` levels, finds where `surface_values` crosses `target`, and
/// linearly interpolates the corresponding value from `values_3d`.
///
/// Both `values_3d` and `surface_values` are flattened in
/// `[k * ny * nx + j * nx + i]` order (level-major).
///
/// `levels` has length `nz` and gives the coordinate value of each
/// level (e.g. pressure in hPa).
///
/// Returns a `Vec<f64>` of length `nx * ny`.  Columns where no
/// crossing is found are filled with `f64::NAN`.
pub fn interpolate_to_isosurface(
    values_3d: &[f64],
    surface_values: &[f64],
    target: f64,
    levels: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let nxy = nx * ny;
    assert_eq!(values_3d.len(), nxy * nz, "values_3d length mismatch");
    assert_eq!(surface_values.len(), nxy * nz, "surface_values length mismatch");
    assert_eq!(levels.len(), nz, "levels length mismatch");

    let mut out = vec![f64::NAN; nxy];

    for j in 0..ny {
        for i in 0..nx {
            let ij = j * nx + i;
            for k in 0..(nz - 1) {
                let idx_lo = k * nxy + ij;
                let idx_hi = (k + 1) * nxy + ij;
                let s_lo = surface_values[idx_lo];
                let s_hi = surface_values[idx_hi];

                // Check for crossing (either direction).
                if (s_lo - target) * (s_hi - target) <= 0.0 && (s_hi - s_lo).abs() > 1e-30 {
                    let t = (target - s_lo) / (s_hi - s_lo);
                    out[ij] = values_3d[idx_lo] + t * (values_3d[idx_hi] - values_3d[idx_lo]);
                    break;
                }
            }
        }
    }
    out
}

// ── Haversine distance (degrees) ──────────────────────────────────

/// Great-circle distance in degrees between two `(lat, lon)` pairs (degrees).
fn haversine_deg(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * a.sqrt().asin().to_degrees()
}

// ── Inverse distance weighted interpolation ───────────────────────

/// Inverse distance weighted (IDW) interpolation to a regular grid.
///
/// For each target grid point, finds all source points within
/// `search_radius` degrees, weights them by `1 / d^power`, and computes
/// the weighted average.  If fewer than `min_neighbors` source points
/// are found within the radius, the target point is set to `f64::NAN`.
///
/// Returns a `Vec<f64>` of length `target.nx * target.ny` in row-major
/// order (south-to-north rows, west-to-east columns).
pub fn inverse_distance_to_grid(
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
    target: &wx_math::regrid::GridSpec,
    power: f64,
    min_neighbors: usize,
    search_radius: f64,
) -> Vec<f64> {
    let n = values.len();
    assert_eq!(lats.len(), n);
    assert_eq!(lons.len(), n);

    let mut out = Vec::with_capacity(target.nx * target.ny);

    for j in 0..target.ny {
        let tlat = target.lat1 + j as f64 * target.dlat;
        for i in 0..target.nx {
            let tlon = target.lon1 + i as f64 * target.dlon;
            out.push(idw_single(tlat, tlon, lats, lons, values, power, min_neighbors, search_radius));
        }
    }
    out
}

/// Inverse distance weighted (IDW) interpolation to arbitrary points (legacy API).
///
/// Same algorithm as [`inverse_distance_to_grid`] but evaluates at
/// the supplied `(target_lats, target_lons)` positions instead of a
/// regular grid.
///
/// For the newer Barnes/Cressman-capable variant, see the re-exported
/// [`inverse_distance_to_points`] from `wx_math::interpolate`.
pub fn inverse_distance_to_points_legacy(
    src_lats: &[f64],
    src_lons: &[f64],
    src_values: &[f64],
    target_lats: &[f64],
    target_lons: &[f64],
    power: f64,
    min_neighbors: usize,
    search_radius: f64,
) -> Vec<f64> {
    let n = src_values.len();
    assert_eq!(src_lats.len(), n);
    assert_eq!(src_lons.len(), n);
    assert_eq!(target_lats.len(), target_lons.len());

    target_lats
        .iter()
        .zip(target_lons.iter())
        .map(|(&tlat, &tlon)| {
            idw_single(tlat, tlon, src_lats, src_lons, src_values, power, min_neighbors, search_radius)
        })
        .collect()
}

/// Compute a single IDW estimate at `(tlat, tlon)`.
fn idw_single(
    tlat: f64,
    tlon: f64,
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
    power: f64,
    min_neighbors: usize,
    search_radius: f64,
) -> f64 {
    let mut w_sum = 0.0_f64;
    let mut wv_sum = 0.0_f64;
    let mut count = 0usize;

    for k in 0..values.len() {
        let d = haversine_deg(tlat, tlon, lats[k], lons[k]);
        if d > search_radius {
            continue;
        }
        // Coincident point -- return exact value.
        if d < 1e-15 {
            return values[k];
        }
        let w = 1.0 / d.powf(power);
        w_sum += w;
        wv_sum += w * values[k];
        count += 1;
    }

    if count < min_neighbors {
        f64::NAN
    } else {
        wv_sum / w_sum
    }
}

// ── Natural neighbor interpolation (Sibson approximation) ─────────

/// Approximate natural-neighbor interpolation to a regular grid.
///
/// True natural-neighbor interpolation requires incremental Voronoi
/// construction, which is very expensive.  This implements a practical
/// Sibson-style approximation: for each target point the *K* nearest
/// source points are found (K = min(12, n)), and weights are computed as
/// `w_i = 1/d_i^2 / sum(1/d_j^2)`.  The exponent of 2 combined with
/// the small, adaptively-chosen neighborhood produces results that
/// closely approximate Sibson weights in smoothly-varying data.
///
/// Returns a `Vec<f64>` of length `target.nx * target.ny` (row-major).
pub fn natural_neighbor_to_grid(
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
    target: &wx_math::regrid::GridSpec,
) -> Vec<f64> {
    let n = values.len();
    assert_eq!(lats.len(), n);
    assert_eq!(lons.len(), n);

    let mut out = Vec::with_capacity(target.nx * target.ny);

    for j in 0..target.ny {
        let tlat = target.lat1 + j as f64 * target.dlat;
        for i in 0..target.nx {
            let tlon = target.lon1 + i as f64 * target.dlon;
            out.push(nn_single(tlat, tlon, lats, lons, values));
        }
    }
    out
}

/// Approximate natural-neighbor interpolation to arbitrary points.
///
/// See [`natural_neighbor_to_grid`] for algorithmic details.
pub fn natural_neighbor_to_points(
    src_lats: &[f64],
    src_lons: &[f64],
    src_values: &[f64],
    target_lats: &[f64],
    target_lons: &[f64],
) -> Vec<f64> {
    let n = src_values.len();
    assert_eq!(src_lats.len(), n);
    assert_eq!(src_lons.len(), n);
    assert_eq!(target_lats.len(), target_lons.len());

    target_lats
        .iter()
        .zip(target_lons.iter())
        .map(|(&tlat, &tlon)| nn_single(tlat, tlon, src_lats, src_lons, src_values))
        .collect()
}

/// Compute a single natural-neighbor (Sibson-approx) estimate.
fn nn_single(
    tlat: f64,
    tlon: f64,
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }

    const K: usize = 12;
    let k = K.min(n);

    // Collect (distance, index) for all source points.
    let mut dists: Vec<(f64, usize)> = (0..n)
        .map(|i| (haversine_deg(tlat, tlon, lats[i], lons[i]), i))
        .collect();

    // Partial sort to get the k nearest (unordered among themselves).
    dists.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Check for coincidence with the actual nearest point.
    let min_dist = dists[..k].iter().map(|&(d, _)| d).fold(f64::INFINITY, f64::min);
    if min_dist < 1e-15 {
        let nearest = dists[..k].iter().find(|&&(d, _)| d < 1e-15).unwrap();
        return values[nearest.1];
    }

    // Sibson-like weights: w_i = 1/d_i^2
    let mut w_sum = 0.0_f64;
    let mut wv_sum = 0.0_f64;
    for &(d, idx) in &dists[..k] {
        let w = 1.0 / (d * d);
        w_sum += w;
        wv_sum += w * values[idx];
    }
    wv_sum / w_sum
}

// ── Vertical cross-section slice ──────────────────────────────────

/// Extract a vertical cross-section from 3-D gridded data along a
/// lat/lon path.
///
/// The source data `values_3d` is a flattened array of shape
/// `[nz, ny, nx]` in level-major order: index = `k * ny * nx + j * nx + i`.
/// `levels` gives the coordinate value at each of the `nz` levels.
/// `src_lats` (length `ny`) and `src_lons` (length `nx`) define the
/// regular source grid.
///
/// For each point along the path `(lat_slice[m], lon_slice[m])` and each
/// level `k`, bilinear interpolation is performed in the horizontal
/// plane.  The result is a `Vec<Vec<f64>>` of shape `[n_points][nz]`.
pub fn interpolate_to_slice(
    values_3d: &[f64],
    levels: &[f64],
    lat_slice: &[f64],
    lon_slice: &[f64],
    src_lats: &[f64],
    src_lons: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<Vec<f64>> {
    assert_eq!(values_3d.len(), nx * ny * nz, "values_3d length mismatch");
    assert_eq!(levels.len(), nz, "levels length mismatch");
    assert_eq!(src_lats.len(), ny, "src_lats length mismatch");
    assert_eq!(src_lons.len(), nx, "src_lons length mismatch");
    assert_eq!(lat_slice.len(), lon_slice.len(), "lat/lon slice length mismatch");

    let n_pts = lat_slice.len();
    let nxy = nx * ny;

    let mut result: Vec<Vec<f64>> = Vec::with_capacity(n_pts);

    for m in 0..n_pts {
        let tlat = lat_slice[m];
        let tlon = lon_slice[m];

        // Find fractional grid indices.
        let fi = find_frac_index(tlon, src_lons);
        let fj = find_frac_index(tlat, src_lats);

        let mut col = Vec::with_capacity(nz);
        for k in 0..nz {
            let val = bilinear_at(values_3d, k * nxy, nx, ny, fi, fj);
            col.push(val);
        }
        result.push(col);
    }

    let _ = levels; // levels provided for API symmetry; used by callers for labeling
    result
}

/// Find the fractional index of `target` within a monotonic 1-D grid.
///
/// Returns a value in `[0, n-1]`. Values outside the grid are clamped.
fn find_frac_index(target: f64, grid: &[f64]) -> f64 {
    let n = grid.len();
    if n < 2 {
        return 0.0;
    }

    let ascending = grid[n - 1] > grid[0];

    // Clamp to grid bounds.
    if ascending {
        if target <= grid[0] { return 0.0; }
        if target >= grid[n - 1] { return (n - 1) as f64; }
    } else {
        if target >= grid[0] { return 0.0; }
        if target <= grid[n - 1] { return (n - 1) as f64; }
    }

    // Binary search for enclosing interval.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        let in_lower = if ascending { grid[mid] <= target } else { grid[mid] >= target };
        if in_lower {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (target - grid[lo]) / (grid[hi] - grid[lo]);
    lo as f64 + t
}

/// Bilinear interpolation at fractional indices `(fi, fj)` within a
/// 2-D slab starting at `offset` in `data`, with dimensions `nx x ny`.
fn bilinear_at(data: &[f64], offset: usize, nx: usize, ny: usize, fi: f64, fj: f64) -> f64 {
    let i0 = (fi.floor() as usize).min(nx.saturating_sub(2));
    let j0 = (fj.floor() as usize).min(ny.saturating_sub(2));
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(ny - 1);

    let di = fi - i0 as f64;
    let dj = fj - j0 as f64;

    let v00 = data[offset + j0 * nx + i0];
    let v10 = data[offset + j0 * nx + i1];
    let v01 = data[offset + j1 * nx + i0];
    let v11 = data[offset + j1 * nx + i1];

    let top = v00 * (1.0 - di) + v10 * di;
    let bot = v01 * (1.0 - di) + v11 * di;
    top * (1.0 - dj) + bot * dj
}

/// Remove observations where the value is NaN.
///
/// Returns `(lats, lons, values)` with the NaN entries dropped.
pub fn remove_nan_observations(
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = values.len();
    assert_eq!(lats.len(), n);
    assert_eq!(lons.len(), n);

    let mut out_lats = Vec::with_capacity(n);
    let mut out_lons = Vec::with_capacity(n);
    let mut out_vals = Vec::with_capacity(n);
    for i in 0..n {
        if !values[i].is_nan() {
            out_lats.push(lats[i]);
            out_lons.push(lons[i]);
            out_vals.push(values[i]);
        }
    }
    (out_lats, out_lons, out_vals)
}

/// Remove observations where the value is below `threshold`.
///
/// Returns `(lats, lons, values)` with only entries where
/// `values[i] >= threshold`.
pub fn remove_observations_below_value(
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
    threshold: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = values.len();
    assert_eq!(lats.len(), n);
    assert_eq!(lons.len(), n);

    let mut out_lats = Vec::with_capacity(n);
    let mut out_lons = Vec::with_capacity(n);
    let mut out_vals = Vec::with_capacity(n);
    for i in 0..n {
        if values[i] >= threshold {
            out_lats.push(lats[i]);
            out_lons.push(lons[i]);
            out_vals.push(values[i]);
        }
    }
    (out_lats, out_lons, out_vals)
}

/// Interpolate scattered data to arbitrary points using a specified method.
///
/// This is a convenience dispatcher that selects between inverse-distance
/// weighting and natural-neighbor interpolation.
///
/// # Arguments
///
/// * `src_lats`, `src_lons`, `src_values` - Source observation coordinates and values
/// * `target_lats`, `target_lons` - Target point coordinates
/// * `interp_type` - Method: `"inverse_distance"` (or `"idw"`, `"linear"`)
///   or `"natural_neighbor"` (or `"nn"`, `"natural"`)
///
/// For IDW, uses power=2, min_neighbors=1, search_radius=10 degrees.
pub fn interpolate_to_points(
    src_lats: &[f64],
    src_lons: &[f64],
    src_values: &[f64],
    target_lats: &[f64],
    target_lons: &[f64],
    interp_type: &str,
) -> Vec<f64> {
    match interp_type {
        "natural_neighbor" | "nn" | "natural" => {
            natural_neighbor_to_points(src_lats, src_lons, src_values, target_lats, target_lons)
        }
        _ => {
            // Default to inverse distance (idw / linear)
            inverse_distance_to_points_legacy(
                src_lats, src_lons, src_values,
                target_lats, target_lons,
                2.0,    // power
                1,      // min_neighbors
                10.0,   // search_radius
            )
        }
    }
}

/// Remove observations with duplicate `(lat, lon)` coordinates,
/// keeping the first occurrence.
pub fn remove_repeat_coordinates(
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use std::collections::HashSet;

    let n = values.len();
    assert_eq!(lats.len(), n);
    assert_eq!(lons.len(), n);

    let mut seen = HashSet::new();
    let mut out_lats = Vec::with_capacity(n);
    let mut out_lons = Vec::with_capacity(n);
    let mut out_vals = Vec::with_capacity(n);

    for i in 0..n {
        // Use bit patterns so we get exact equality semantics.
        let key = (lats[i].to_bits(), lons[i].to_bits());
        if seen.insert(key) {
            out_lats.push(lats[i]);
            out_lons.push(lons[i]);
            out_vals.push(values[i]);
        }
    }
    (out_lats, out_lons, out_vals)
}

/// Compute `n_points` equally-spaced points along the great-circle
/// path between two `(lat, lon)` positions (degrees).
///
/// Returns `(lats, lons)`, each of length `n_points`, including the
/// start and end points.
pub fn geodesic(
    start: (f64, f64),
    end: (f64, f64),
    n_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    assert!(n_points >= 2, "n_points must be at least 2");

    let to_rad = std::f64::consts::PI / 180.0;
    let to_deg = 180.0 / std::f64::consts::PI;

    let (lat1, lon1) = (start.0 * to_rad, start.1 * to_rad);
    let (lat2, lon2) = (end.0 * to_rad, end.1 * to_rad);

    // Central angle via haversine.
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let d = 2.0 * a.sqrt().asin();

    let mut lats = Vec::with_capacity(n_points);
    let mut lons = Vec::with_capacity(n_points);

    if d.abs() < 1e-15 {
        // Start and end are (essentially) identical.
        for _ in 0..n_points {
            lats.push(start.0);
            lons.push(start.1);
        }
        return (lats, lons);
    }

    for idx in 0..n_points {
        let f = idx as f64 / (n_points - 1) as f64;
        let a_coeff = ((1.0 - f) * d).sin() / d.sin();
        let b_coeff = (f * d).sin() / d.sin();

        let x = a_coeff * lat1.cos() * lon1.cos() + b_coeff * lat2.cos() * lon2.cos();
        let y = a_coeff * lat1.cos() * lon1.sin() + b_coeff * lat2.cos() * lon2.sin();
        let z = a_coeff * lat1.sin() + b_coeff * lat2.sin();

        lats.push(z.atan2((x * x + y * y).sqrt()) * to_deg);
        lons.push(y.atan2(x) * to_deg);
    }

    (lats, lons)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_1d_basic() {
        let xp = vec![0.0, 1.0, 2.0, 3.0];
        let fp = vec![0.0, 10.0, 20.0, 30.0];
        let x = vec![0.5, 1.5, 2.5];
        let result = interpolate_1d(&x, &xp, &fp);
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 15.0).abs() < 1e-10);
        assert!((result[2] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_1d_nan_outside_range() {
        // Values outside [xp[0], xp[last]] should return NaN (MetPy-compatible).
        let xp = vec![1.0, 2.0, 3.0];
        let fp = vec![10.0, 20.0, 30.0];
        let x = vec![0.0, 4.0];
        let result = interpolate_1d(&x, &xp, &fp);
        assert!(result[0].is_nan(), "below range should be NaN");
        assert!(result[1].is_nan(), "above range should be NaN");
    }

    #[test]
    fn test_interpolate_1d_exact_boundary() {
        // Values exactly at boundaries should return the boundary values.
        let xp = vec![1.0, 2.0, 3.0];
        let fp = vec![10.0, 20.0, 30.0];
        let x = vec![1.0, 3.0];
        let result = interpolate_1d(&x, &xp, &fp);
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_interpolate_1d_descending_pressure() {
        // Typical pressure levels (descending) with temperature values
        let p = vec![1000.0, 850.0, 700.0, 500.0];
        let t = vec![20.0, 12.0, 2.0, -15.0];
        let target_p = vec![900.0, 600.0];
        let result = log_interpolate_1d(&target_p, &p, &t);
        // Result should be between the bounding values
        assert!(result[0] > 12.0 && result[0] < 20.0);
        assert!(result[1] > -15.0 && result[1] < 2.0);
    }

    // ── interpolate_nans_1d tests ───────────────────────────────

    #[test]
    fn test_interpolate_nans_1d_interior() {
        let mut v = vec![1.0, f64::NAN, f64::NAN, 4.0];
        interpolate_nans_1d(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-10);
        assert!((v[1] - 2.0).abs() < 1e-10);
        assert!((v[2] - 3.0).abs() < 1e-10);
        assert!((v[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_nans_1d_edges() {
        let mut v = vec![f64::NAN, f64::NAN, 5.0, 10.0, f64::NAN];
        interpolate_nans_1d(&mut v);
        assert!((v[0] - 5.0).abs() < 1e-10);
        assert!((v[1] - 5.0).abs() < 1e-10);
        assert!((v[4] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_nans_1d_all_nan() {
        let mut v = vec![f64::NAN, f64::NAN];
        interpolate_nans_1d(&mut v);
        assert!(v[0].is_nan());
        assert!(v[1].is_nan());
    }

    #[test]
    fn test_interpolate_nans_1d_no_nan() {
        let mut v = vec![1.0, 2.0, 3.0];
        interpolate_nans_1d(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-10);
        assert!((v[1] - 2.0).abs() < 1e-10);
        assert!((v[2] - 3.0).abs() < 1e-10);
    }

    // ── interpolate_to_isosurface tests ─────────────────────────

    #[test]
    fn test_interpolate_to_isosurface_basic() {
        // 2x2 horizontal, 3 levels
        let nx = 2;
        let ny = 2;
        let nz = 3;
        let nxy = nx * ny;
        // surface_values: linear in level => 0, 5, 10 at each column
        let mut surface_values = vec![0.0; nxy * nz];
        let mut values_3d = vec![0.0; nxy * nz];
        let levels = vec![1000.0, 500.0, 200.0];
        for k in 0..nz {
            for ij in 0..nxy {
                surface_values[k * nxy + ij] = (k as f64) * 5.0; // 0, 5, 10
                values_3d[k * nxy + ij] = (k as f64) * 100.0; // 0, 100, 200
            }
        }
        // target = 2.5 => halfway between level 0 (val=0) and level 1 (val=5)
        let out = interpolate_to_isosurface(&values_3d, &surface_values, 2.5, &levels, nx, ny, nz);
        assert_eq!(out.len(), nxy);
        for &val in &out {
            assert!((val - 50.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_interpolate_to_isosurface_no_crossing() {
        let nx = 1;
        let ny = 1;
        let nz = 2;
        let surface_values = vec![1.0, 2.0];
        let values_3d = vec![10.0, 20.0];
        let levels = vec![1000.0, 500.0];
        let out = interpolate_to_isosurface(&values_3d, &surface_values, 5.0, &levels, nx, ny, nz);
        assert!(out[0].is_nan());
    }

    // ── remove_nan_observations tests ───────────────────────────

    #[test]
    fn test_remove_nan_observations() {
        let lats = vec![30.0, 31.0, 32.0, 33.0];
        let lons = vec![-90.0, -91.0, -92.0, -93.0];
        let vals = vec![1.0, f64::NAN, 3.0, f64::NAN];
        let (rl, ro, rv) = remove_nan_observations(&lats, &lons, &vals);
        assert_eq!(rv.len(), 2);
        assert!((rv[0] - 1.0).abs() < 1e-10);
        assert!((rv[1] - 3.0).abs() < 1e-10);
        assert!((rl[0] - 30.0).abs() < 1e-10);
        assert!((ro[1] + 92.0).abs() < 1e-10);
    }

    // ── remove_observations_below_value tests ───────────────────

    #[test]
    fn test_remove_observations_below_value() {
        let lats = vec![30.0, 31.0, 32.0];
        let lons = vec![-90.0, -91.0, -92.0];
        let vals = vec![5.0, 2.0, 8.0];
        let (rl, _ro, rv) = remove_observations_below_value(&lats, &lons, &vals, 4.0);
        assert_eq!(rv.len(), 2);
        assert!((rv[0] - 5.0).abs() < 1e-10);
        assert!((rv[1] - 8.0).abs() < 1e-10);
        assert!((rl[0] - 30.0).abs() < 1e-10);
    }

    // ── remove_repeat_coordinates tests ─────────────────────────

    #[test]
    fn test_remove_repeat_coordinates() {
        let lats = vec![30.0, 31.0, 30.0, 32.0];
        let lons = vec![-90.0, -91.0, -90.0, -92.0];
        let vals = vec![1.0, 2.0, 99.0, 4.0];
        let (rl, _ro, rv) = remove_repeat_coordinates(&lats, &lons, &vals);
        assert_eq!(rv.len(), 3);
        // First occurrence of (30,-90) kept with value 1.0
        assert!((rv[0] - 1.0).abs() < 1e-10);
        assert!((rv[1] - 2.0).abs() < 1e-10);
        assert!((rv[2] - 4.0).abs() < 1e-10);
        assert!((rl[2] - 32.0).abs() < 1e-10);
    }

    // ── geodesic tests ──────────────────────────────────────────

    #[test]
    fn test_geodesic_endpoints() {
        let start = (40.0, -90.0);
        let end = (50.0, -80.0);
        let (lats, lons) = geodesic(start, end, 11);
        assert_eq!(lats.len(), 11);
        assert!((lats[0] - 40.0).abs() < 1e-10);
        assert!((lons[0] + 90.0).abs() < 1e-10);
        assert!((lats[10] - 50.0).abs() < 1e-10);
        assert!((lons[10] + 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_geodesic_same_point() {
        let pt = (45.0, -100.0);
        let (lats, lons) = geodesic(pt, pt, 5);
        assert_eq!(lats.len(), 5);
        for i in 0..5 {
            assert!((lats[i] - 45.0).abs() < 1e-10);
            assert!((lons[i] + 100.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_geodesic_equator_segment() {
        // Along the equator, lat should stay 0.
        let (lats, _lons) = geodesic((0.0, 0.0), (0.0, 90.0), 10);
        for &lat in &lats {
            assert!(lat.abs() < 1e-10, "Expected lat ~ 0, got {}", lat);
        }
    }

    // ── haversine_deg tests ──────────────────────────────────────

    #[test]
    fn test_haversine_deg_same_point() {
        let d = haversine_deg(40.0, -90.0, 40.0, -90.0);
        assert!(d.abs() < 1e-12);
    }

    #[test]
    fn test_haversine_deg_known_distance() {
        // 1 degree of latitude at the equator ~ 1 degree great-circle.
        let d = haversine_deg(0.0, 0.0, 1.0, 0.0);
        assert!((d - 1.0).abs() < 1e-10);
    }

    // ── inverse_distance_to_grid tests ───────────────────────────

    #[test]
    fn test_idw_to_grid_exact_hit() {
        // Source point coincident with a target grid point => exact value.
        let lats = vec![30.0];
        let lons = vec![-90.0];
        let vals = vec![42.0];
        let grid = wx_math::regrid::GridSpec::regular(30.0, 30.0, -90.0, -90.0, 1.0);
        let out = inverse_distance_to_grid(&lats, &lons, &vals, &grid, 2.0, 1, 5.0);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_idw_to_grid_weighted_average() {
        // Two equidistant source points should produce the simple average.
        let lats = vec![30.0, 32.0];
        let lons = vec![-90.0, -90.0];
        let vals = vec![10.0, 20.0];
        let grid = wx_math::regrid::GridSpec::regular(31.0, 31.0, -90.0, -90.0, 1.0);
        let out = inverse_distance_to_grid(&lats, &lons, &vals, &grid, 2.0, 1, 5.0);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 15.0).abs() < 1e-10, "Expected 15, got {}", out[0]);
    }

    #[test]
    fn test_idw_to_grid_too_few_neighbors() {
        // Source at (30,-90), target at (31,-90) -- close but not coincident.
        let lats = vec![30.0];
        let lons = vec![-90.0];
        let vals = vec![42.0];
        let grid = wx_math::regrid::GridSpec::regular(31.0, 31.0, -90.0, -90.0, 1.0);
        // Require 5 neighbors but only 1 source point within radius.
        let out = inverse_distance_to_grid(&lats, &lons, &vals, &grid, 2.0, 5, 10.0);
        assert!(out[0].is_nan());
    }

    #[test]
    fn test_idw_to_grid_outside_radius() {
        let lats = vec![0.0];
        let lons = vec![0.0];
        let vals = vec![42.0];
        let grid = wx_math::regrid::GridSpec::regular(50.0, 50.0, 50.0, 50.0, 1.0);
        // search_radius too small to reach from (50,50) to (0,0).
        let out = inverse_distance_to_grid(&lats, &lons, &vals, &grid, 2.0, 1, 1.0);
        assert!(out[0].is_nan());
    }

    // ── inverse_distance_to_points tests ─────────────────────────

    #[test]
    fn test_idw_to_points_basic() {
        let src_lats = vec![30.0, 32.0];
        let src_lons = vec![-90.0, -90.0];
        let src_vals = vec![10.0, 20.0];
        // Target at midpoint.
        let tgt_lats = vec![31.0];
        let tgt_lons = vec![-90.0];
        let out = inverse_distance_to_points_legacy(
            &src_lats, &src_lons, &src_vals,
            &tgt_lats, &tgt_lons,
            2.0, 1, 5.0,
        );
        assert_eq!(out.len(), 1);
        assert!((out[0] - 15.0).abs() < 1e-10, "Expected 15, got {}", out[0]);
    }

    #[test]
    fn test_idw_to_points_coincident() {
        let src_lats = vec![40.0, 42.0];
        let src_lons = vec![-80.0, -80.0];
        let src_vals = vec![100.0, 200.0];
        // Target exactly at first source point.
        let tgt_lats = vec![40.0];
        let tgt_lons = vec![-80.0];
        let out = inverse_distance_to_points_legacy(
            &src_lats, &src_lons, &src_vals,
            &tgt_lats, &tgt_lons,
            2.0, 1, 10.0,
        );
        assert!((out[0] - 100.0).abs() < 1e-10);
    }

    // ── natural_neighbor_to_grid tests ───────────────────────────

    #[test]
    fn test_nn_to_grid_uniform_field() {
        // All source values equal => result should equal that value everywhere.
        let lats = vec![30.0, 31.0, 30.0, 31.0];
        let lons = vec![-91.0, -91.0, -90.0, -90.0];
        let vals = vec![7.0, 7.0, 7.0, 7.0];
        let grid = wx_math::regrid::GridSpec::regular(30.0, 31.0, -91.0, -90.0, 0.5);
        let out = natural_neighbor_to_grid(&lats, &lons, &vals, &grid);
        for &v in &out {
            assert!((v - 7.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nn_to_grid_coincident() {
        // Target point coincides with a source => exact value.
        let lats = vec![35.0];
        let lons = vec![-95.0];
        let vals = vec![99.0];
        let grid = wx_math::regrid::GridSpec::regular(35.0, 35.0, -95.0, -95.0, 1.0);
        let out = natural_neighbor_to_grid(&lats, &lons, &vals, &grid);
        assert!((out[0] - 99.0).abs() < 1e-10);
    }

    #[test]
    fn test_nn_to_grid_symmetric() {
        // Two equidistant sources => result should be the average.
        let lats = vec![30.0, 32.0];
        let lons = vec![-90.0, -90.0];
        let vals = vec![10.0, 20.0];
        let grid = wx_math::regrid::GridSpec::regular(31.0, 31.0, -90.0, -90.0, 1.0);
        let out = natural_neighbor_to_grid(&lats, &lons, &vals, &grid);
        assert!((out[0] - 15.0).abs() < 1e-10, "Expected 15, got {}", out[0]);
    }

    // ── natural_neighbor_to_points tests ─────────────────────────

    #[test]
    fn test_nn_to_points_basic() {
        let src_lats = vec![30.0, 32.0, 31.0];
        let src_lons = vec![-90.0, -90.0, -88.0];
        let src_vals = vec![10.0, 20.0, 30.0];
        let tgt_lats = vec![31.0];
        let tgt_lons = vec![-90.0];
        let out = natural_neighbor_to_points(
            &src_lats, &src_lons, &src_vals,
            &tgt_lats, &tgt_lons,
        );
        assert_eq!(out.len(), 1);
        // Two nearest on lon=-90 are equidistant; third is farther.
        // The closer two dominate and average 15; third pulls toward 30 slightly.
        assert!(out[0] > 10.0 && out[0] < 30.0);
    }

    #[test]
    fn test_nn_to_points_empty_source() {
        let out = natural_neighbor_to_points(
            &[], &[], &[],
            &[31.0], &[-90.0],
        );
        assert!(out[0].is_nan());
    }

    // ── interpolate_to_slice tests ───────────────────────────────

    #[test]
    fn test_interpolate_to_slice_basic() {
        // 3x3 grid, 2 levels. Values increase with level.
        let nx = 3;
        let ny = 3;
        let nz = 2;
        let nxy = nx * ny;
        let mut values_3d = vec![0.0; nxy * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    values_3d[k * nxy + j * nx + i] = (k * 100 + j * 10 + i) as f64;
                }
            }
        }
        let levels = vec![1000.0, 500.0];
        let src_lats = vec![30.0, 31.0, 32.0]; // ny=3
        let src_lons = vec![-92.0, -91.0, -90.0]; // nx=3
        // Slice along the middle row (lat=31.0).
        let lat_slice = vec![31.0, 31.0, 31.0];
        let lon_slice = vec![-92.0, -91.0, -90.0];
        let out = interpolate_to_slice(
            &values_3d, &levels,
            &lat_slice, &lon_slice,
            &src_lats, &src_lons,
            nx, ny, nz,
        );
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 2);
        // At (lat=31, lon=-92) => j=1, i=0 => level0: 10, level1: 110
        assert!((out[0][0] - 10.0).abs() < 1e-10);
        assert!((out[0][1] - 110.0).abs() < 1e-10);
        // At (lat=31, lon=-91) => j=1, i=1 => level0: 11, level1: 111
        assert!((out[1][0] - 11.0).abs() < 1e-10);
        assert!((out[1][1] - 111.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_to_slice_between_points() {
        // 2x2 grid, 1 level. Interpolate at center.
        let nx = 2;
        let ny = 2;
        let nz = 1;
        let values_3d = vec![0.0, 10.0, 20.0, 30.0];
        // Layout: j=0: [0, 10], j=1: [20, 30]
        let levels = vec![1000.0];
        let src_lats = vec![30.0, 31.0];
        let src_lons = vec![-91.0, -90.0];
        // Target at the center of the 4 grid points.
        let lat_slice = vec![30.5];
        let lon_slice = vec![-90.5];
        let out = interpolate_to_slice(
            &values_3d, &levels,
            &lat_slice, &lon_slice,
            &src_lats, &src_lons,
            nx, ny, nz,
        );
        // Bilinear average of [0, 10, 20, 30] = 15
        assert!((out[0][0] - 15.0).abs() < 1e-10, "Expected 15, got {}", out[0][0]);
    }

    #[test]
    fn test_interpolate_to_slice_at_grid_corner() {
        // Should return exact grid value when target is on a grid point.
        let nx = 2;
        let ny = 2;
        let nz = 1;
        let values_3d = vec![100.0, 200.0, 300.0, 400.0];
        let levels = vec![500.0];
        let src_lats = vec![40.0, 41.0];
        let src_lons = vec![-80.0, -79.0];
        let lat_slice = vec![40.0];
        let lon_slice = vec![-80.0];
        let out = interpolate_to_slice(
            &values_3d, &levels,
            &lat_slice, &lon_slice,
            &src_lats, &src_lons,
            nx, ny, nz,
        );
        assert!((out[0][0] - 100.0).abs() < 1e-10);
    }
}
