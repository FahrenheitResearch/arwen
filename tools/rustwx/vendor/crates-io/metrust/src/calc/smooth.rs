//! Grid smoothing, spatial derivative utilities, and filtering functions.
//!
//! Re-exports gradient and Laplacian operators from `wx_math::dynamics`
//! and `wx_math::gridmath`, plus geospatial derivatives that work on
//! lat/lon grids.
//!
//! Also implements MetPy's smoothing filters: Gaussian, rectangular (box),
//! circular (disk), and N-point smoothers.
//!
//! All grids are flattened row-major: `index = j * nx + i` where `j` is the
//! row (y-index) and `i` is the column (x-index).
//!
//! NaN values are excluded from weighted averages. At grid edges, only the
//! available neighbors are used.

// ── Basic grid derivatives ───────────────────────────────────────────

/// Partial derivative df/dx using centered finite differences
/// (forward/backward at boundaries).
///
/// Input `values` is flattened row-major with shape `(ny, nx)`.
/// `dx` is the grid spacing in meters.
pub use wx_math::dynamics::gradient_x;

/// Partial derivative df/dy using centered finite differences
/// (forward/backward at boundaries).
///
/// Input `values` is flattened row-major with shape `(ny, nx)`.
/// `dy` is the grid spacing in meters.
pub use wx_math::dynamics::gradient_y;

/// Laplacian: `d2f/dx2 + d2f/dy2` using second-order centered differences.
pub use wx_math::dynamics::laplacian;

// ── Generalized derivatives ──────────────────────────────────────────

/// First derivative along a chosen axis (0 = x, 1 = y).
///
/// Uses centered second-order finite differences in the interior with
/// first-order forward/backward at boundaries.
pub use wx_math::gridmath::first_derivative;

/// Second derivative along a chosen axis (0 = x, 1 = y).
pub use wx_math::gridmath::second_derivative;

// ── Geospatial derivatives ───────────────────────────────────────────

/// Compute physical grid spacings `(dx, dy)` in meters from lat/lon arrays
/// using the haversine formula.
pub use wx_math::gridmath::lat_lon_grid_deltas;

/// Gradient on a lat/lon grid: converts the scalar field to `(df/dx, df/dy)`
/// using geospatially-correct spacings.
pub use wx_math::gridmath::geospatial_gradient;

/// Laplacian on a lat/lon grid with geospatially-correct spacings.
pub use wx_math::gridmath::geospatial_laplacian;

// ── Helpers ──────────────────────────────────────────────────────────

use rayon::prelude::*;

/// Row-major index.
#[inline(always)]
#[cfg(test)]
fn idx(j: usize, i: usize, nx: usize) -> usize {
    j * nx + i
}

// ─────────────────────────────────────────────────────────────────────
// Gaussian smoothing
// ─────────────────────────────────────────────────────────────────────

/// Apply a 2D Gaussian smoothing filter (separable implementation).
///
/// The kernel half-width is `ceil(4 * sigma)` grid points, giving a full
/// kernel size of `2 * half + 1`. The filter is applied separably: first
/// along rows, then along columns, for efficiency.
///
/// NaN values in `data` are excluded from the weighted average. If every
/// neighbor within the kernel is NaN, the output is NaN.
///
/// # Arguments
///
/// * `data` - Input field, flattened row-major, length `nx * ny`.
/// * `nx` - Number of grid points in the x (column) direction.
/// * `ny` - Number of grid points in the y (row) direction.
/// * `sigma` - Standard deviation of the Gaussian kernel in grid-point units.
///
/// # Panics
///
/// Panics if `data.len() != nx * ny` or `sigma <= 0`.
///
/// # Example
///
/// ```
/// use metrust::calc::smooth::smooth_gaussian;
///
/// let nx = 5;
/// let ny = 5;
/// let data = vec![0.0; nx * ny];
/// let smoothed = smooth_gaussian(&data, nx, ny, 1.0);
/// assert_eq!(smoothed.len(), nx * ny);
/// ```
pub fn smooth_gaussian(data: &[f64], nx: usize, ny: usize, sigma: f64) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(data.len(), n, "data length must equal nx * ny");
    assert!(sigma > 0.0, "sigma must be positive, got {}", sigma);

    let half = (4.0 * sigma).ceil() as usize;
    let kernel_size = 2 * half + 1;

    // Build 1D Gaussian kernel
    let mut kernel = vec![0.0; kernel_size];
    let two_sigma2 = 2.0 * sigma * sigma;
    for k in 0..kernel_size {
        let d = k as f64 - half as f64;
        kernel[k] = (-d * d / two_sigma2).exp();
    }

    // Pass 1: smooth along x (rows) — each row is independent
    let mut temp = vec![f64::NAN; n];
    temp.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        for i in 0..nx {
            let mut wsum = 0.0;
            let mut vsum = 0.0;
            for k in 0..kernel_size {
                let ii = i as isize + k as isize - half as isize;
                if ii < 0 || ii >= nx as isize {
                    continue;
                }
                let val = data[j * nx + ii as usize];
                if val.is_nan() {
                    continue;
                }
                let w = kernel[k];
                wsum += w;
                vsum += w * val;
            }
            row[i] = if wsum > 0.0 { vsum / wsum } else { f64::NAN };
        }
    });

    // Pass 2: smooth along y (columns) — parallelize by output row
    let mut out = vec![f64::NAN; n];
    out.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        for i in 0..nx {
            let mut wsum = 0.0;
            let mut vsum = 0.0;
            for k in 0..kernel_size {
                let jj = j as isize + k as isize - half as isize;
                if jj < 0 || jj >= ny as isize {
                    continue;
                }
                let val = temp[jj as usize * nx + i];
                if val.is_nan() {
                    continue;
                }
                let w = kernel[k];
                wsum += w;
                vsum += w * val;
            }
            row[i] = if wsum > 0.0 { vsum / wsum } else { f64::NAN };
        }
    });

    out
}

// ─────────────────────────────────────────────────────────────────────
// Rectangular (box) smoothing
// ─────────────────────────────────────────────────────────────────────

/// Apply a rectangular (box / uniform) smoothing filter.
///
/// Each output value is the unweighted mean of the `size x size`
/// neighborhood centered on that grid point. NaN values propagate:
/// if any neighbor in the kernel is NaN, the output is NaN.
///
/// **Boundary handling (MetPy-compatible):** edge points where the full
/// kernel does not fit are left with their original values. The
/// unsmoothed border is `size / 2` grid points wide on each side.
///
/// # Arguments
///
/// * `data` - Input field, flattened row-major, length `nx * ny`.
/// * `nx` - Number of columns.
/// * `ny` - Number of rows.
/// * `size` - Side length of the square kernel (should be odd; if even, the
///   effective half-width is `size / 2`).
/// * `passes` - Number of times to apply the filter (default 1).
///
/// # Panics
///
/// Panics if `data.len() != nx * ny` or `size == 0`.
///
/// # Example
///
/// ```
/// use metrust::calc::smooth::smooth_rectangular;
///
/// let data = vec![1.0; 9];
/// let out = smooth_rectangular(&data, 3, 3, 3, 1);
/// assert!((out[4] - 1.0).abs() < 1e-10);
/// ```
pub fn smooth_rectangular(data: &[f64], nx: usize, ny: usize, size: usize, passes: usize) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(data.len(), n, "data length must equal nx * ny");
    assert!(size > 0, "kernel size must be > 0");

    let half = size / 2;
    let mut current = data.to_vec();

    // Padded dimensions for summed area table (1-indexed with zero border)
    let pnx = nx + 1;
    let pny = ny + 1;

    for _ in 0..passes {
        // Build summed area tables for values and NaN count — O(n)
        let mut sat_val = vec![0.0; pnx * pny];
        let mut sat_nan = vec![0u32; pnx * pny];

        for j in 0..ny {
            for i in 0..nx {
                let v = current[j * nx + i];
                let is_nan = v.is_nan();
                let pj = j + 1;
                let pi = i + 1;
                let pidx = pj * pnx + pi;
                sat_val[pidx] = (if is_nan { 0.0 } else { v })
                    + sat_val[(pj - 1) * pnx + pi]
                    + sat_val[pj * pnx + (pi - 1)]
                    - sat_val[(pj - 1) * pnx + (pi - 1)];
                sat_nan[pidx] = (if is_nan { 1 } else { 0 })
                    + sat_nan[(pj - 1) * pnx + pi]
                    + sat_nan[pj * pnx + (pi - 1)]
                    - sat_nan[(pj - 1) * pnx + (pi - 1)];
            }
        }

        // Compute output using O(1) SAT lookups per point — parallelized by row
        let mut out = current.clone();
        let interior_j_start = half;
        let interior_j_end = ny.saturating_sub(half);

        if interior_j_end > interior_j_start {
            let interior_rows = interior_j_end - interior_j_start;
            let mut interior_slice = vec![0.0f64; interior_rows * nx];

            interior_slice.par_chunks_mut(nx).enumerate().for_each(|(row_idx, row)| {
                let j = interior_j_start + row_idx;
                for i in half..nx.saturating_sub(half) {
                    // Window corners in padded coords
                    let y1 = j - half;       // top row (0-indexed in data)
                    let y2 = j + half;       // bottom row
                    let x1 = i - half;
                    let x2 = i + half;
                    // SAT: sum(y1..y2, x1..x2) = SAT[y2+1][x2+1] - SAT[y1][x2+1] - SAT[y2+1][x1] + SAT[y1][x1]
                    let br = (y2 + 1) * pnx + (x2 + 1);
                    let tr = y1 * pnx + (x2 + 1);
                    let bl = (y2 + 1) * pnx + x1;
                    let tl = y1 * pnx + x1;

                    let nan_count = sat_nan[br] - sat_nan[tr] - sat_nan[bl] + sat_nan[tl];
                    if nan_count > 0 {
                        row[i] = f64::NAN;
                    } else {
                        let sum = sat_val[br] - sat_val[tr] - sat_val[bl] + sat_val[tl];
                        let count = (y2 - y1 + 1) * (x2 - x1 + 1);
                        row[i] = sum / count as f64;
                    }
                }
                // Copy edge values within this row
                for i in 0..half {
                    row[i] = current[(interior_j_start + row_idx) * nx + i];
                }
                for i in nx.saturating_sub(half)..nx {
                    row[i] = current[(interior_j_start + row_idx) * nx + i];
                }
            });

            // Write interior rows back to output
            for (row_idx, chunk) in interior_slice.chunks(nx).enumerate() {
                let j = interior_j_start + row_idx;
                out[j * nx..(j + 1) * nx].copy_from_slice(chunk);
            }
        }

        current = out;
    }

    current
}

// ─────────────────────────────────────────────────────────────────────
// Circular (disk) smoothing
// ─────────────────────────────────────────────────────────────────────

/// Apply a circular (disk) smoothing filter.
///
/// Each output value is the unweighted mean of all grid points within
/// `radius` grid-point units of the center. The distance check uses
/// Euclidean distance: `sqrt(di^2 + dj^2) <= radius`.
///
/// NaN values propagate: if any point in the disk is NaN, the output
/// is NaN.
///
/// **Boundary handling (MetPy-compatible):** edge points where the full
/// disk kernel does not fit are left with their original values. The
/// unsmoothed border is `radius` (ceiled) grid points wide on each side.
///
/// # Arguments
///
/// * `data` - Input field, flattened row-major, length `nx * ny`.
/// * `nx` - Number of columns.
/// * `ny` - Number of rows.
/// * `radius` - Radius of the disk kernel in grid-point units.
/// * `passes` - Number of times to apply the filter (default 1).
///
/// # Panics
///
/// Panics if `data.len() != nx * ny` or `radius <= 0`.
///
/// # Example
///
/// ```
/// use metrust::calc::smooth::smooth_circular;
///
/// let data = vec![1.0; 25];
/// let out = smooth_circular(&data, 5, 5, 2.0, 1);
/// assert!((out[12] - 1.0).abs() < 1e-10);
/// ```
pub fn smooth_circular(data: &[f64], nx: usize, ny: usize, radius: f64, passes: usize) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(data.len(), n, "data length must equal nx * ny");
    assert!(radius > 0.0, "radius must be positive, got {}", radius);

    // Pre-compute the kernel offsets (dj, di) that fall within the radius
    let half = radius.ceil() as isize;
    let half_u = half as usize;
    let r2 = radius * radius;
    let mut offsets = Vec::new();
    for dj in -half..=half {
        for di in -half..=half {
            let dist2 = (di * di + dj * dj) as f64;
            if dist2 <= r2 {
                offsets.push((dj, di));
            }
        }
    }

    let mut current = data.to_vec();

    for _ in 0..passes {
        let mut out = current.clone();
        let j_start = half_u;
        let j_end = ny.saturating_sub(half_u);

        if j_end > j_start {
            // Parallelize over interior rows
            let interior_rows = j_end - j_start;
            let mut interior = vec![0.0f64; interior_rows * nx];

            interior.par_chunks_mut(nx).enumerate().for_each(|(row_idx, row)| {
                let j = j_start + row_idx;
                // Copy edge columns from current
                for i in 0..half_u.min(nx) {
                    row[i] = current[j * nx + i];
                }
                for i in nx.saturating_sub(half_u)..nx {
                    row[i] = current[j * nx + i];
                }
                // Compute interior columns
                for i in half_u..nx.saturating_sub(half_u) {
                    let mut sum = 0.0;
                    let mut count = 0u32;
                    let mut has_nan = false;

                    for &(dj, di) in &offsets {
                        let jj = (j as isize + dj) as usize;
                        let ii = (i as isize + di) as usize;
                        let val = current[jj * nx + ii];
                        if val.is_nan() {
                            has_nan = true;
                            break;
                        }
                        sum += val;
                        count += 1;
                    }

                    row[i] = if has_nan {
                        f64::NAN
                    } else if count > 0 {
                        sum / count as f64
                    } else {
                        f64::NAN
                    };
                }
            });

            for (row_idx, chunk) in interior.chunks(nx).enumerate() {
                let j = j_start + row_idx;
                out[j * nx..(j + 1) * nx].copy_from_slice(chunk);
            }
        }

        current = out;
    }

    current
}

// ─────────────────────────────────────────────────────────────────────
// N-point smoothing (5-point and 9-point)
// ─────────────────────────────────────────────────────────────────────

/// Apply a 5-point or 9-point smoother (MetPy-compatible).
///
/// This replicates MetPy's `smooth_n_point` filter exactly. It delegates
/// to [`smooth_window`] with the same normalized weights MetPy uses:
///
/// * **n = 9**: `[[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125],
///   [0.0625, 0.125, 0.0625]]`
/// * **n = 5**: `[[0, 0.125, 0], [0.125, 0.5, 0.125], [0, 0.125, 0]]`
///
/// **Boundary handling:** edge points where the full 3x3 kernel does not
/// fit are left with their original values (border of 1 on each side).
/// NaN values propagate.
///
/// # Arguments
///
/// * `data` - Input field, flattened row-major, length `nx * ny`.
/// * `nx` - Number of columns.
/// * `ny` - Number of rows.
/// * `n` - Number of points: must be 5 or 9.
/// * `passes` - Number of times to apply the filter (default 1).
///
/// # Panics
///
/// Panics if `n` is not 5 or 9, or if `data.len() != nx * ny`.
///
/// # Example
///
/// ```
/// use metrust::calc::smooth::smooth_n_point;
///
/// let data = vec![1.0; 25];
/// let out = smooth_n_point(&data, 5, 5, 5, 1);
/// assert!((out[12] - 1.0).abs() < 1e-10);
/// ```
pub fn smooth_n_point(data: &[f64], nx: usize, ny: usize, n: usize, passes: usize) -> Vec<f64> {
    let len = nx * ny;
    assert_eq!(data.len(), len, "data length must equal nx * ny");
    assert!(n == 5 || n == 9, "n must be 5 or 9, got {}", n);

    // MetPy-exact normalized weights (normalize_weights=False in MetPy,
    // meaning these are used directly as convolution weights, not as
    // a weighted average).
    let window: Vec<f64> = if n == 9 {
        vec![
            0.0625, 0.125, 0.0625,
            0.125,  0.25,  0.125,
            0.0625, 0.125, 0.0625,
        ]
    } else {
        vec![
            0.0,   0.125, 0.0,
            0.125, 0.5,   0.125,
            0.0,   0.125, 0.0,
        ]
    };

    smooth_window(data, nx, ny, &window, 3, 3, passes, false)
}

// ─────────────────────────────────────────────────────────────────────
// Generic window (custom kernel) smoothing
// ─────────────────────────────────────────────────────────────────────

/// Apply a generic 2D convolution with a user-supplied kernel
/// (MetPy-compatible).
///
/// This is the equivalent of MetPy's `smooth_window`, which accepts any
/// custom kernel (e.g., a manually constructed Gaussian, Laplacian, or
/// sharpening filter).
///
/// The kernel is a flattened row-major array of size `window_nx * window_ny`.
///
/// **Boundary handling:** edge points where the full kernel does not fit
/// are left with their original values. The unsmoothed border is
/// `(window_nx - 1) / 2` on left/right and `(window_ny - 1) / 2` on
/// top/bottom. NaN values propagate: if any value in the kernel footprint
/// is NaN, the output for that point is NaN.
///
/// # Arguments
///
/// * `data` - Input field, flattened row-major, length `nx * ny`.
/// * `nx` - Number of columns in the data grid.
/// * `ny` - Number of rows in the data grid.
/// * `window` - Flattened row-major kernel weights, length
///   `window_nx * window_ny`.
/// * `window_nx` - Number of columns in the kernel.
/// * `window_ny` - Number of rows in the kernel.
/// * `passes` - Number of times to apply the filter.
/// * `normalize_weights` - If true, divide weights by their sum before
///   applying. If false, use weights directly.
///
/// # Panics
///
/// Panics if `data.len() != nx * ny`, `window.len() != window_nx * window_ny`,
/// or if either kernel dimension is zero.
///
/// # Example
///
/// ```
/// use metrust::calc::smooth::smooth_window;
///
/// // 3x3 uniform kernel (equivalent to smooth_rectangular with size 3)
/// let kernel = vec![1.0; 9];
/// let data = vec![1.0; 25];
/// let out = smooth_window(&data, 5, 5, &kernel, 3, 3, 1, true);
/// assert!((out[12] - 1.0).abs() < 1e-10);
/// ```
pub fn smooth_window(
    data: &[f64],
    nx: usize,
    ny: usize,
    window: &[f64],
    window_nx: usize,
    window_ny: usize,
    passes: usize,
    normalize_weights: bool,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(data.len(), n, "data length must equal nx * ny");
    assert_eq!(
        window.len(),
        window_nx * window_ny,
        "window length must equal window_nx * window_ny"
    );
    assert!(window_nx > 0, "window_nx must be > 0");
    assert!(window_ny > 0, "window_ny must be > 0");

    let half_x = window_nx / 2;
    let half_y = window_ny / 2;

    // Optionally normalize weights
    let weights: Vec<f64> = if normalize_weights {
        let wsum: f64 = window.iter().sum();
        if wsum.abs() > 1e-30 {
            window.iter().map(|&w| w / wsum).collect()
        } else {
            window.to_vec()
        }
    } else {
        window.to_vec()
    };

    let mut current = data.to_vec();

    for _ in 0..passes {
        let mut out = current.clone();
        let j_start = half_y;
        let j_end = ny.saturating_sub(half_y);

        if j_end > j_start {
            let interior_rows = j_end - j_start;
            let mut interior = vec![0.0f64; interior_rows * nx];

            interior.par_chunks_mut(nx).enumerate().for_each(|(row_idx, row)| {
                let j = j_start + row_idx;
                // Copy full row from current (edges preserved)
                row.copy_from_slice(&current[j * nx..(j + 1) * nx]);
                // Compute interior columns
                for i in half_x..nx.saturating_sub(half_x) {
                    let mut vsum = 0.0;
                    let mut has_nan = false;

                    'outer: for wj in 0..window_ny {
                        let jj = j + wj - half_y;
                        for wi in 0..window_nx {
                            let ii = i + wi - half_x;
                            let val = current[jj * nx + ii];
                            if val.is_nan() {
                                has_nan = true;
                                break 'outer;
                            }
                            vsum += weights[wj * window_nx + wi] * val;
                        }
                    }

                    row[i] = if has_nan { f64::NAN } else { vsum };
                }
            });

            for (row_idx, chunk) in interior.chunks(nx).enumerate() {
                let j = j_start + row_idx;
                out[j * nx..(j + 1) * nx].copy_from_slice(chunk);
            }
        }

        current = out;
    }

    current
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert two f64 values are approximately equal.
    fn approx(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "approx failed: {} vs {} (diff {}, tol {})",
            a, b, (a - b).abs(), tol
        );
    }

    // =========================================================
    // Gaussian smoothing
    // =========================================================

    #[test]
    fn test_gaussian_constant_field() {
        // Smoothing a constant field should return the same constant.
        let nx = 7;
        let ny = 7;
        let data = vec![42.0; nx * ny];
        let out = smooth_gaussian(&data, nx, ny, 1.5);
        for val in &out {
            approx(*val, 42.0, 1e-10);
        }
    }

    #[test]
    fn test_gaussian_single_spike() {
        // A single spike should be spread out and reduced in amplitude.
        let nx = 11;
        let ny = 11;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(5, 5, nx)] = 100.0;

        let out = smooth_gaussian(&data, nx, ny, 1.0);

        // Center should be reduced
        assert!(out[idx(5, 5, nx)] < 100.0);
        assert!(out[idx(5, 5, nx)] > 0.0);

        // Neighbors should pick up some of the value
        assert!(out[idx(5, 6, nx)] > 0.0);
        assert!(out[idx(6, 5, nx)] > 0.0);

        // Far-away points should be near zero
        assert!(out[idx(0, 0, nx)] < 0.01);
    }

    #[test]
    fn test_gaussian_symmetry() {
        // A centered spike should produce a symmetric result.
        let nx = 9;
        let ny = 9;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(4, 4, nx)] = 100.0;

        let out = smooth_gaussian(&data, nx, ny, 1.5);

        // Check 4-fold symmetry around center
        approx(out[idx(3, 4, nx)], out[idx(5, 4, nx)], 1e-10);
        approx(out[idx(4, 3, nx)], out[idx(4, 5, nx)], 1e-10);
        approx(out[idx(3, 3, nx)], out[idx(5, 5, nx)], 1e-10);
        approx(out[idx(3, 5, nx)], out[idx(5, 3, nx)], 1e-10);
    }

    #[test]
    fn test_gaussian_nan_handling() {
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![10.0; n];
        data[idx(2, 2, nx)] = f64::NAN;

        let out = smooth_gaussian(&data, nx, ny, 1.0);

        // Output at the NaN point should not be NaN because neighbors are valid
        assert!(!out[idx(2, 2, nx)].is_nan(), "center should not be NaN");
        // Neighbors of the NaN should still be finite
        assert!(!out[idx(2, 3, nx)].is_nan());
        assert!(!out[idx(1, 2, nx)].is_nan());
    }

    #[test]
    fn test_gaussian_all_nan() {
        let data = vec![f64::NAN; 9];
        let out = smooth_gaussian(&data, 3, 3, 1.0);
        for val in &out {
            assert!(val.is_nan(), "all-NaN input should give all-NaN output");
        }
    }

    #[test]
    fn test_gaussian_larger_sigma_more_smoothing() {
        // Larger sigma should produce a smoother (lower peak) result
        let nx = 11;
        let ny = 11;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(5, 5, nx)] = 100.0;

        let out_narrow = smooth_gaussian(&data, nx, ny, 0.5);
        let out_wide = smooth_gaussian(&data, nx, ny, 2.0);

        assert!(
            out_wide[idx(5, 5, nx)] < out_narrow[idx(5, 5, nx)],
            "wider sigma should give lower peak: {} vs {}",
            out_wide[idx(5, 5, nx)],
            out_narrow[idx(5, 5, nx)]
        );
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn test_gaussian_nonpositive_sigma_panics() {
        let data = vec![1.0; 4];
        let _ = smooth_gaussian(&data, 2, 2, 0.0);
    }

    // =========================================================
    // Rectangular (box) smoothing
    // =========================================================

    #[test]
    fn test_rectangular_constant_field() {
        let data = vec![7.0; 25];
        let out = smooth_rectangular(&data, 5, 5, 3, 1);
        for val in &out {
            approx(*val, 7.0, 1e-10);
        }
    }

    #[test]
    fn test_rectangular_known_average() {
        // 3x3 grid, all ones except center = 10, box size 3
        // Center average = (8 * 1 + 10) / 9 = 18/9 = 2.0
        let nx = 3;
        let ny = 3;
        let mut data = vec![1.0; 9];
        data[idx(1, 1, nx)] = 10.0;

        let out = smooth_rectangular(&data, nx, ny, 3, 1);
        approx(out[idx(1, 1, nx)], 2.0, 1e-10);
    }

    #[test]
    fn test_rectangular_edge_preserved() {
        // With MetPy-compatible boundary handling, edge points are
        // preserved (copied from original).
        let data: Vec<f64> = (1..=25).map(|x| x as f64).collect();
        let out = smooth_rectangular(&data, 5, 5, 3, 1);
        // Corners should be original values
        approx(out[0], 1.0, 1e-10);   // (0,0)
        approx(out[4], 5.0, 1e-10);   // (0,4)
        approx(out[20], 21.0, 1e-10); // (4,0)
        approx(out[24], 25.0, 1e-10); // (4,4)
    }

    #[test]
    fn test_rectangular_size_1() {
        // Box of size 1 = identity filter (half=0, all points are interior)
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let out = smooth_rectangular(&data, 4, 3, 1, 1);
        for k in 0..12 {
            approx(out[k], data[k], 1e-10);
        }
    }

    #[test]
    fn test_rectangular_nan_propagation() {
        // NaN in the kernel footprint propagates to output.
        let nx = 3;
        let ny = 3;
        let mut data = vec![4.0; 9];
        data[idx(1, 1, nx)] = f64::NAN;

        let out = smooth_rectangular(&data, nx, ny, 3, 1);

        // Center (1,1) is the only interior point for a 3x3 grid with size=3.
        // It has a NaN in its footprint (itself), so it should be NaN.
        assert!(out[idx(1, 1, nx)].is_nan());
    }

    #[test]
    fn test_rectangular_all_nan() {
        let data = vec![f64::NAN; 9];
        let out = smooth_rectangular(&data, 3, 3, 3, 1);
        for val in &out {
            assert!(val.is_nan());
        }
    }

    #[test]
    #[should_panic(expected = "kernel size must be > 0")]
    fn test_rectangular_zero_size_panics() {
        let data = vec![1.0; 4];
        let _ = smooth_rectangular(&data, 2, 2, 0, 1);
    }

    #[test]
    fn test_rectangular_large_window() {
        // Window larger than grid => all points are edges => all preserved.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let out = smooth_rectangular(&data, 3, 3, 99, 1);
        for k in 0..9 {
            approx(out[k], data[k], 1e-10);
        }
    }

    // =========================================================
    // Circular (disk) smoothing
    // =========================================================

    #[test]
    fn test_circular_constant_field() {
        let data = vec![3.14; 49];
        let out = smooth_circular(&data, 7, 7, 2.0, 1);
        for val in &out {
            approx(*val, 3.14, 1e-10);
        }
    }

    #[test]
    fn test_circular_edge_preserved() {
        // Edges (within `radius` of the border) should be original values.
        let data: Vec<f64> = (1..=25).map(|x| x as f64).collect();
        let out = smooth_circular(&data, 5, 5, 1.0, 1);
        // With radius=1, border is 1 wide. Corners are edges.
        approx(out[0], 1.0, 1e-10);   // (0,0)
        approx(out[4], 5.0, 1e-10);   // (0,4)
        approx(out[20], 21.0, 1e-10); // (4,0)
        approx(out[24], 25.0, 1e-10); // (4,4)
    }

    #[test]
    fn test_circular_radius_1_interior() {
        // Radius 1.0 includes center (dist 0) and 4 cardinal neighbors (dist 1)
        // = 5-point stencil with equal weights
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(2, 2, nx)] = 5.0;

        let out = smooth_circular(&data, nx, ny, 1.0, 1);

        // Center: 5-point average = 5.0 / 5 = 1.0
        approx(out[idx(2, 2, nx)], 1.0, 1e-10);
    }

    #[test]
    fn test_circular_nan_propagation() {
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![2.0; n];
        data[idx(2, 2, nx)] = f64::NAN;

        let out = smooth_circular(&data, nx, ny, 1.0, 1);

        // Interior neighbors that see the NaN should be NaN
        assert!(out[idx(2, 2, nx)].is_nan());
    }

    #[test]
    fn test_circular_symmetry() {
        let nx = 11;
        let ny = 11;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(5, 5, nx)] = 100.0;

        let out = smooth_circular(&data, nx, ny, 2.0, 1);

        // 4-fold symmetry at interior points
        approx(out[idx(4, 5, nx)], out[idx(6, 5, nx)], 1e-10);
        approx(out[idx(5, 4, nx)], out[idx(5, 6, nx)], 1e-10);
    }

    #[test]
    #[should_panic(expected = "radius must be positive")]
    fn test_circular_nonpositive_radius_panics() {
        let data = vec![1.0; 4];
        let _ = smooth_circular(&data, 2, 2, 0.0, 1);
    }

    // =========================================================
    // N-point smoothing (MetPy-compatible)
    // =========================================================

    #[test]
    fn test_5point_constant_field() {
        let data = vec![5.0; 25];
        let out = smooth_n_point(&data, 5, 5, 5, 1);
        for val in &out {
            approx(*val, 5.0, 1e-10);
        }
    }

    #[test]
    fn test_9point_constant_field() {
        let data = vec![5.0; 25];
        let out = smooth_n_point(&data, 5, 5, 9, 1);
        for val in &out {
            approx(*val, 5.0, 1e-10);
        }
    }

    #[test]
    fn test_9point_metpy_exact() {
        // Exact MetPy test: 5x5 grid 1..25, smooth_n_point(9, 1)
        // MetPy leaves edges untouched, center (2,2) = 13.0
        let data: Vec<f64> = (1..=25).map(|x| x as f64).collect();
        let out = smooth_n_point(&data, 5, 5, 9, 1);

        // Edges preserved
        approx(out[0], 1.0, 1e-10);   // (0,0)
        approx(out[4], 5.0, 1e-10);   // (0,4)
        approx(out[20], 21.0, 1e-10); // (4,0)
        approx(out[24], 25.0, 1e-10); // (4,4)

        // Center: 9-point weighted average of a linear field = same as original
        approx(out[idx(2, 2, 5)], 13.0, 1e-10);
    }

    #[test]
    fn test_5point_known_result() {
        // 5x5 grid, center = 10, rest = 0
        // 5-point weights: [[0,0.125,0],[0.125,0.5,0.125],[0,0.125,0]]
        // At center (2,2): 0.5*10 + 0.125*0*4 = 5.0
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(2, 2, nx)] = 10.0;

        let out = smooth_n_point(&data, nx, ny, 5, 1);
        approx(out[idx(2, 2, nx)], 5.0, 1e-10);
    }

    #[test]
    fn test_9point_known_result() {
        // 5x5 grid, center (2,2) = 8, rest = 0
        // 9-point weights: sum = 1.0; center weight = 0.25
        // At center: 0.25 * 8 = 2.0
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(2, 2, nx)] = 8.0;

        let out = smooth_n_point(&data, nx, ny, 9, 1);
        approx(out[idx(2, 2, nx)], 0.25 * 8.0, 1e-10);
    }

    #[test]
    fn test_n_point_edge_preserved() {
        // Corners and edges are preserved with original values.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let out = smooth_n_point(&data, 5, 3, 5, 1);
        // Top edge (j=0) all preserved
        approx(out[0], 1.0, 1e-10);
        approx(out[1], 2.0, 1e-10);
        approx(out[4], 5.0, 1e-10);
        // Bottom edge (j=2) all preserved
        approx(out[10], 11.0, 1e-10);
        approx(out[14], 15.0, 1e-10);
    }

    #[test]
    fn test_5point_nan_propagation() {
        // NaN in the kernel footprint propagates to the output.
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![4.0; n];
        data[idx(2, 2, nx)] = f64::NAN;

        let out = smooth_n_point(&data, nx, ny, 5, 1);
        // Center (2,2) sees itself (NaN) => propagates
        assert!(out[idx(2, 2, nx)].is_nan());
        // Cardinal neighbors of center also see the NaN
        assert!(out[idx(1, 2, nx)].is_nan());
        assert!(out[idx(3, 2, nx)].is_nan());
        assert!(out[idx(2, 1, nx)].is_nan());
        assert!(out[idx(2, 3, nx)].is_nan());
    }

    #[test]
    fn test_n_point_all_nan() {
        let data = vec![f64::NAN; 25];
        let out5 = smooth_n_point(&data, 5, 5, 5, 1);
        let out9 = smooth_n_point(&data, 5, 5, 9, 1);
        for val in out5.iter().chain(out9.iter()) {
            assert!(val.is_nan());
        }
    }

    #[test]
    #[should_panic(expected = "n must be 5 or 9")]
    fn test_n_point_invalid_n_panics() {
        let data = vec![1.0; 9];
        let _ = smooth_n_point(&data, 3, 3, 7, 1);
    }

    #[test]
    fn test_5point_preserves_linear_field() {
        // A linear field f(i,j) = i + j should be preserved by the 5-point filter
        // at interior points (the 5-point stencil is exact for linear fields).
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                data[j * nx + i] = (i + j) as f64;
            }
        }
        let out = smooth_n_point(&data, nx, ny, 5, 1);
        // Interior points
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                approx(out[k], data[k], 1e-10);
            }
        }
    }

    #[test]
    fn test_9point_preserves_linear_field() {
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                data[j * nx + i] = 2.0 * i as f64 + 3.0 * j as f64;
            }
        }
        let out = smooth_n_point(&data, nx, ny, 9, 1);
        // Interior: the 9-point stencil preserves linear fields.
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                approx(out[k], data[k], 1e-10);
            }
        }
    }

    // =========================================================
    // Generic window (custom kernel) smoothing
    // =========================================================

    #[test]
    fn test_window_constant_field() {
        // Any kernel on a constant field should return that constant.
        let kernel = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
        let data = vec![7.0; 25];
        let out = smooth_window(&data, 5, 5, &kernel, 3, 3, 1, true);
        for val in &out {
            approx(*val, 7.0, 1e-10);
        }
    }

    #[test]
    fn test_window_uniform_kernel_matches_rectangular() {
        // A uniform kernel should produce the same result as smooth_rectangular
        // for interior points and edges.
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let data: Vec<f64> = (0..n).map(|k| (k as f64 * 3.7).sin() * 10.0).collect();
        let kernel = vec![1.0; 9]; // 3x3 uniform
        let from_window = smooth_window(&data, nx, ny, &kernel, 3, 3, 1, true);
        let from_rect = smooth_rectangular(&data, nx, ny, 3, 1);
        for k in 0..n {
            approx(from_window[k], from_rect[k], 1e-10);
        }
    }

    #[test]
    fn test_window_edge_preserved() {
        // Edges should preserve original values.
        let kernel = vec![1.0; 9];
        let data: Vec<f64> = (1..=25).map(|x| x as f64).collect();
        let out = smooth_window(&data, 5, 5, &kernel, 3, 3, 1, true);
        // Corners preserved
        approx(out[0], 1.0, 1e-10);
        approx(out[4], 5.0, 1e-10);
        approx(out[20], 21.0, 1e-10);
        approx(out[24], 25.0, 1e-10);
    }

    #[test]
    fn test_window_single_center_weight() {
        // Kernel with weight only at center acts as identity at interior points,
        // edges are preserved.
        let kernel = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let data: Vec<f64> = (0..25).map(|k| k as f64).collect();
        let out = smooth_window(&data, 5, 5, &kernel, 3, 3, 1, false);
        for k in 0..25 {
            approx(out[k], data[k], 1e-10);
        }
    }

    #[test]
    fn test_window_nan_propagation() {
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![4.0; n];
        data[idx(2, 2, nx)] = f64::NAN;
        let kernel = vec![1.0; 9]; // 3x3 uniform
        let out = smooth_window(&data, nx, ny, &kernel, 3, 3, 1, true);
        // Center and neighbors that see NaN should be NaN
        assert!(out[idx(2, 2, nx)].is_nan());
    }

    #[test]
    fn test_window_all_nan() {
        let data = vec![f64::NAN; 25];
        let kernel = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
        let out = smooth_window(&data, 5, 5, &kernel, 3, 3, 1, true);
        for val in &out {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_window_1x1_kernel() {
        // 1x1 kernel = identity (half=0, all points are interior).
        let kernel = vec![5.0];
        let data: Vec<f64> = (0..12).map(|k| k as f64).collect();
        let out = smooth_window(&data, 4, 3, &kernel, 1, 1, 1, true);
        for k in 0..12 {
            approx(out[k], data[k], 1e-10);
        }
    }

    #[test]
    fn test_window_asymmetric_kernel() {
        // 1x3 horizontal kernel (row-only smoothing)
        // Normalized: [0.25, 0.5, 0.25]
        let kernel = vec![1.0, 2.0, 1.0]; // window_nx=3, window_ny=1
        let nx = 5;
        let ny = 1;
        let data = vec![0.0, 0.0, 4.0, 0.0, 0.0];
        let out = smooth_window(&data, nx, ny, &kernel, 3, 1, 1, true);
        // Center (index 2): (0.25*0 + 0.5*4 + 0.25*0) = 2.0
        approx(out[2], 2.0, 1e-10);
        // Index 1: (0.25*0 + 0.5*0 + 0.25*4) = 1.0
        approx(out[1], 1.0, 1e-10);
        // Index 3: same as index 1 by symmetry
        approx(out[3], 1.0, 1e-10);
        // Index 0 and 4 are edges => preserved
        approx(out[0], 0.0, 1e-10);
        approx(out[4], 0.0, 1e-10);
    }

    #[test]
    fn test_window_large_kernel_on_small_grid() {
        // 5x5 kernel on a 3x3 grid: all edges => all preserved.
        let kernel = vec![1.0; 25];
        let data: Vec<f64> = (1..=9).map(|x| x as f64).collect();
        let out = smooth_window(&data, 3, 3, &kernel, 5, 5, 1, true);
        for k in 0..9 {
            approx(out[k], data[k], 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "window length must equal")]
    fn test_window_mismatched_kernel_panics() {
        let data = vec![1.0; 9];
        let kernel = vec![1.0; 4]; // 4 != 3*3
        let _ = smooth_window(&data, 3, 3, &kernel, 3, 3, 1, true);
    }

    #[test]
    fn test_window_reduces_variance() {
        // Any reasonable smoothing kernel should reduce variance on noisy data
        // (at least for interior points which get smoothed).
        let nx = 9;
        let ny = 9;
        let n = nx * ny;
        let data: Vec<f64> = (0..n)
            .map(|k| (k as f64 * 17.3).sin() * 100.0)
            .collect();
        let mean = data.iter().sum::<f64>() / n as f64;
        let var_in: f64 = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

        // Gaussian-like 3x3 kernel
        let kernel = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
        let out = smooth_window(&data, nx, ny, &kernel, 3, 3, 1, true);
        let m = out.iter().sum::<f64>() / n as f64;
        let var_out: f64 = out.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
        assert!(
            var_out < var_in,
            "smooth_window should reduce variance: {} vs {}",
            var_out, var_in
        );
    }

    // =========================================================
    // Multi-pass tests
    // =========================================================

    #[test]
    fn test_n_point_multiple_passes() {
        // Multiple passes should produce more smoothing
        let data: Vec<f64> = (1..=25).map(|x| x as f64).collect();
        let out1 = smooth_n_point(&data, 5, 5, 9, 1);
        let out3 = smooth_n_point(&data, 5, 5, 9, 3);

        // Interior point (2,2) with more passes should still preserve
        // a linear field (but NaN propagation would make it different
        // on fields with NaN).
        approx(out1[idx(2, 2, 5)], 13.0, 1e-10);
        approx(out3[idx(2, 2, 5)], 13.0, 1e-10);
    }

    // =========================================================
    // Cross-filter consistency checks
    // =========================================================

    #[test]
    fn test_smoothers_reduce_variance() {
        // Any smoother applied to a noisy field should reduce variance.
        let nx = 11;
        let ny = 11;
        let n = nx * ny;
        // Deterministic "noisy" field
        let data: Vec<f64> = (0..n)
            .map(|k| (k as f64 * 17.3).sin() * 100.0)
            .collect();

        let mean = data.iter().sum::<f64>() / n as f64;
        let var_in: f64 = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

        let gauss = smooth_gaussian(&data, nx, ny, 1.0);
        let rect = smooth_rectangular(&data, nx, ny, 3, 1);
        let circ = smooth_circular(&data, nx, ny, 1.5, 1);
        let s5 = smooth_n_point(&data, nx, ny, 5, 1);
        let s9 = smooth_n_point(&data, nx, ny, 9, 1);

        for (name, out) in [
            ("gaussian", &gauss),
            ("rectangular", &rect),
            ("circular", &circ),
            ("5-point", &s5),
            ("9-point", &s9),
        ] {
            let m = out.iter().sum::<f64>() / n as f64;
            let var: f64 = out.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
            assert!(
                var < var_in,
                "{} did not reduce variance: var_in={}, var_out={}",
                name, var_in, var
            );
        }
    }
}
