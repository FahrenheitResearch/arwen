//! Updraft helicity diagnostic.
//!
//! Matches the wrf-python Fortran subroutine DCALCUH (calc_uh.f90):
//!
//! 1. Compute tem1(k) = w_destag(k) * vorticity(k) at each scalar level,
//!    where vorticity uses centered differences divided by the map scale
//!    factor (MAPFAC_M).
//! 2. For each column, compute the column-mean w over [z_bot, z_top].
//! 3. If the column-mean w > 0, integrate tem1 over the layer using the
//!    trapezoidal rule.  Otherwise UH = 0 for that column.

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Linearly interpolate value at height `z` between two levels.
/// `z0`/`z1` are heights at the two levels, `v0`/`v1` are the field values.
#[inline]
fn lerp_at(z: f64, z0: f64, z1: f64, v0: f64, v1: f64) -> f64 {
    if (z1 - z0).abs() < 1e-6 {
        0.5 * (v0 + v1)
    } else {
        let frac = (z - z0) / (z1 - z0);
        v0 + frac * (v1 - v0)
    }
}

/// Updraft helicity (m^2/s^2). `[ny, nx]`
///
/// UH = integral from z_bot to z_top of (w * zeta_z) dz
/// Default layer: 2-5 km AGL.
///
/// Matches the wrf-python Fortran DCALCUH:
/// - Vorticity uses centered differences divided by map scale factor.
/// - A column-mean w is computed first; only columns with positive mean w
///   contribute to UH (matching the Fortran's updraft check).
/// - The integrand is w*vort (pre-multiplied), integrated with the
///   trapezoidal rule.
pub fn compute_uhel(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let w = f.w_destag(t)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let h_agl = f.height_agl(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;
    let dx = f.dx;
    let dy = f.dy;

    // Integration layer bounds (default 2-5 km AGL), configurable
    let z_bot = opts.bottom_m.unwrap_or(2000.0);
    let z_top = opts.top_m.unwrap_or(5000.0);

    // Try to read map scale factor (MAPFAC_M).  Fall back to 1.0 everywhere
    // if the variable is not present in the file.
    let mapfct: Vec<f64> = f.read_var("MAPFAC_M", t).unwrap_or_else(|_| vec![1.0; nxy]);

    // Compute vorticity at each scalar level, dividing by map scale factor
    // to match the Fortran: dv/(2*dx*mapfct) - du/(2*dy*mapfct).
    let twodx = 2.0 * dx;
    let twody = 2.0 * dy;
    let mut vort_3d = vec![0.0f64; nz * nxy];
    vort_3d.chunks_mut(nxy).enumerate().for_each(|(k, plane)| {
        let u_plane = &u[k * nxy..(k + 1) * nxy];
        let v_plane = &v[k * nxy..(k + 1) * nxy];
        for j in 0..ny {
            for i in 0..nx {
                let ij = j * nx + i;
                let m = mapfct[ij];

                // dv/dx: centered differences, forward/backward at boundaries
                let dvdx = if nx < 2 {
                    0.0
                } else if i == 0 {
                    (v_plane[j * nx + 1] - v_plane[j * nx]) / (dx * m)
                } else if i == nx - 1 {
                    (v_plane[j * nx + nx - 1] - v_plane[j * nx + nx - 2]) / (dx * m)
                } else {
                    (v_plane[j * nx + i + 1] - v_plane[j * nx + i - 1]) / (twodx * m)
                };

                // du/dy: centered differences, forward/backward at boundaries
                let dudy = if ny < 2 {
                    0.0
                } else if j == 0 {
                    (u_plane[nx + i] - u_plane[i]) / (dy * m)
                } else if j == ny - 1 {
                    (u_plane[(ny - 1) * nx + i] - u_plane[(ny - 2) * nx + i]) / (dy * m)
                } else {
                    (u_plane[(j + 1) * nx + i] - u_plane[(j - 1) * nx + i]) / (twody * m)
                };

                plane[ij] = dvdx - dudy;
            }
        }
    });

    // Pre-multiply: tem1(k) = w_destag(k) * vorticity(k)
    let mut tem1 = vec![0.0f64; nz * nxy];
    tem1.iter_mut().enumerate().for_each(|(idx, val)| {
        *val = w[idx] * vort_3d[idx];
    });

    // Integrate per column, checking column-mean w first (Fortran DCALCUH logic).
    let mut uhel = vec![0.0f64; nxy];
    uhel.iter_mut().enumerate().for_each(|(ij, uh_val)| {
        // --- Step 1: compute column-mean w over [z_bot, z_top] ---
        let mut w_sum = 0.0f64;
        let mut depth = 0.0f64;

        for k in 0..nz - 1 {
            let idx0 = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;

            let h0 = h_agl[idx0];
            let h1 = h_agl[idx1];

            // Skip layers entirely outside the integration bounds
            if h1 <= z_bot || h0 >= z_top {
                continue;
            }

            let z_lo = h0.max(z_bot);
            let z_hi = h1.min(z_top);
            let dz = z_hi - z_lo;
            if dz <= 0.0 {
                continue;
            }

            // Interpolate w to clamped endpoints
            let w_lo = if z_lo > h0 {
                lerp_at(z_lo, h0, h1, w[idx0], w[idx1])
            } else {
                w[idx0]
            };
            let w_hi = if z_hi < h1 {
                lerp_at(z_hi, h0, h1, w[idx0], w[idx1])
            } else {
                w[idx1]
            };

            w_sum += 0.5 * (w_lo + w_hi) * dz;
            depth += dz;
        }

        // Column-mean w check: only positive-mean columns contribute UH
        if depth <= 0.0 {
            return;
        }
        let w_mean = w_sum / depth;
        if w_mean <= 0.0 {
            return;
        }

        // --- Step 2: integrate tem1 (= w*vort) over [z_bot, z_top] ---
        let mut integral = 0.0f64;

        for k in 0..nz - 1 {
            let idx0 = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;

            let h0 = h_agl[idx0];
            let h1 = h_agl[idx1];

            if h1 <= z_bot || h0 >= z_top {
                continue;
            }

            let z_lo = h0.max(z_bot);
            let z_hi = h1.min(z_top);
            let dz = z_hi - z_lo;
            if dz <= 0.0 {
                continue;
            }

            // Interpolate tem1 to clamped endpoints
            let t0 = tem1[idx0];
            let t1 = tem1[idx1];

            let tem1_lo = if z_lo > h0 {
                lerp_at(z_lo, h0, h1, t0, t1)
            } else {
                t0
            };
            let tem1_hi = if z_hi < h1 {
                lerp_at(z_hi, h0, h1, t0, t1)
            } else {
                t1
            };

            integral += 0.5 * (tem1_lo + tem1_hi) * dz;
        }

        *uh_val = integral;
    });

    Ok(uhel)
}
