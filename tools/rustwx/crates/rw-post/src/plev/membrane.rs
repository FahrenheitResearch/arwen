//! The membrane ("horizontal") reduction to sea level [R14]:
//! on each pressure surface, the virtual temperature at points below the
//! ground solves Laplace's equation with the above-ground values as
//! boundary conditions; heights of underground surfaces then follow
//! hypsometrically.
//!
//! Solver: a cascade of grids.  The level is coarsened by 2 x 2 blocks until
//! no side exceeds `MEMBRANE_COARSEST_MAX_DIM`; a coarse cell is a boundary
//! cell when any of its fine cells is, with the mean of those boundary
//! values.  Fixed-count Jacobi sweeps run on the coarsest grid, then each
//! finer grid starts its unknown cells from the parent coarse value and runs
//! its own fixed-count sweeps.  No convergence test and no atomics, so the
//! result is the same on every run and device.
//!
//! Masks: 0 unknown (underground), 1 boundary (above ground), 2 missing
//! (skipped as a neighbour, never updated).

use rayon::prelude::*;

use crate::plev::consts::*;

pub const UNKNOWN: u8 = 0;
pub const FIXED: u8 = 1;
pub const MISSING: u8 = 2;

/// Membrane levels (indices) with at least one underground cell.
pub fn active_levels(levels: &[f64], pg: &[f64]) -> Vec<usize> {
    let mut min_pg = f64::INFINITY;
    for &p in pg {
        if p < min_pg {
            min_pg = p;
        }
    }
    (0..levels.len()).filter(|&m| levels[m] > min_pg).collect()
}

/// Values and mask of one level from the column pass's Tv plane.
pub fn level_inputs(tv: &[f64], pg: &[f64], p_level: f64) -> (Vec<f64>, Vec<u8>) {
    let mask = tv
        .iter()
        .zip(pg)
        .map(|(&t, &g)| {
            if t.is_nan() || g.is_nan() {
                MISSING
            } else if p_level > g {
                UNKNOWN
            } else {
                FIXED
            }
        })
        .collect();
    (tv.to_vec(), mask)
}

/// Grid sizes of the cascade, finest first.
pub fn pyramid(nx: usize, ny: usize) -> Vec<(usize, usize)> {
    let mut dims = vec![(nx, ny)];
    let (mut x, mut y) = (nx, ny);
    while x > MEMBRANE_COARSEST_MAX_DIM || y > MEMBRANE_COARSEST_MAX_DIM {
        x = x.div_ceil(2);
        y = y.div_ceil(2);
        dims.push((x, y));
    }
    dims
}

/// One coarse cell from its (up to four) fine cells, visited in the order
/// (0,0), (1,0), (0,1), (1,1).
#[inline]
pub(crate) fn restrict_cell(nxf: usize, nyf: usize, vf: &[f64], mf: &[u8], ic: usize, jc: usize) -> (f64, u8) {
    let mut sum_fixed = 0.0;
    let mut n_fixed = 0u32;
    let mut sum_all = 0.0;
    let mut n_all = 0u32;
    for t in 0..4 {
        let i = 2 * ic + (t & 1);
        let j = 2 * jc + (t >> 1);
        if i < nxf && j < nyf {
            let c = j * nxf + i;
            let m = mf[c];
            if m != MISSING {
                sum_all = sum_all + vf[c];
                n_all += 1;
                if m == FIXED {
                    sum_fixed = sum_fixed + vf[c];
                    n_fixed += 1;
                }
            }
        }
    }
    if n_all == 0 {
        (f64::NAN, MISSING)
    } else if n_fixed > 0 {
        (sum_fixed / n_fixed as f64, FIXED)
    } else {
        (sum_all / n_all as f64, UNKNOWN)
    }
}

/// One Jacobi update: an unknown cell becomes the mean of its present
/// neighbours, visited west, east, south, north.
#[inline]
pub(crate) fn jacobi_cell(nx: usize, ny: usize, src: &[f64], m: &[u8], i: usize, j: usize) -> f64 {
    let c = j * nx + i;
    if m[c] != UNKNOWN {
        return src[c];
    }
    let mut sum = 0.0;
    let mut n = 0u32;
    if i > 0 && m[c - 1] != MISSING {
        sum = sum + src[c - 1];
        n += 1;
    }
    if i + 1 < nx && m[c + 1] != MISSING {
        sum = sum + src[c + 1];
        n += 1;
    }
    if j > 0 && m[c - nx] != MISSING {
        sum = sum + src[c - nx];
        n += 1;
    }
    if j + 1 < ny && m[c + nx] != MISSING {
        sum = sum + src[c + nx];
        n += 1;
    }
    if n == 0 { src[c] } else { sum / n as f64 }
}

fn sweeps(nx: usize, ny: usize, v: &mut Vec<f64>, m: &[u8], count: usize) {
    let mut tmp = vec![0.0; v.len()];
    for _ in 0..count {
        tmp.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            for (i, x) in row.iter_mut().enumerate() {
                *x = jacobi_cell(nx, ny, v, m, i, j);
            }
        });
        std::mem::swap(v, &mut tmp);
    }
}

/// Solve one level; returns the filled values (boundary and missing cells
/// unchanged).
pub fn solve_cpu(nx: usize, ny: usize, vals: Vec<f64>, mask: Vec<u8>) -> Vec<f64> {
    let dims = pyramid(nx, ny);
    let mut v: Vec<Vec<f64>> = vec![vals];
    let mut m: Vec<Vec<u8>> = vec![mask];
    for l in 1..dims.len() {
        let (nxf, nyf) = dims[l - 1];
        let (nxc, nyc) = dims[l];
        let mut vc = vec![0.0; nxc * nyc];
        let mut mc = vec![0u8; nxc * nyc];
        for jc in 0..nyc {
            for ic in 0..nxc {
                let (a, b) = restrict_cell(nxf, nyf, &v[l - 1], &m[l - 1], ic, jc);
                vc[jc * nxc + ic] = a;
                mc[jc * nxc + ic] = b;
            }
        }
        v.push(vc);
        m.push(mc);
    }
    let last = dims.len() - 1;
    {
        let (x, y) = dims[last];
        sweeps(x, y, &mut v[last], &m[last], MEMBRANE_COARSE_SWEEPS);
    }
    for l in (0..last).rev() {
        let (nxf, nyf) = dims[l];
        let (nxc, _) = dims[l + 1];
        let (coarse, fine) = {
            let (a, b) = v.split_at_mut(l + 1);
            (&b[0], &mut a[l])
        };
        for j in 0..nyf {
            for i in 0..nxf {
                let c = j * nxf + i;
                if m[l][c] == UNKNOWN {
                    fine[c] = coarse[(j / 2) * nxc + i / 2];
                }
            }
        }
        sweeps(nxf, nyf, &mut v[l], &m[l], MEMBRANE_LEVEL_SWEEPS);
    }
    v.swap_remove(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_field_is_reproduced() {
        // A harmonic (linear) boundary field: the Laplace solution inside a
        // hole is the same plane.
        let (nx, ny) = (70, 50);
        let mut vals = vec![0.0; nx * ny];
        let mut mask = vec![FIXED; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                vals[c] = 280.0 + 0.1 * i as f64 - 0.05 * j as f64;
                let (di, dj) = (i as f64 - 35.0, j as f64 - 25.0);
                if di * di + dj * dj < 150.0 {
                    mask[c] = UNKNOWN;
                    vals[c] = 250.0;
                }
            }
        }
        let out = solve_cpu(nx, ny, vals, mask);
        for j in 0..ny {
            for i in 0..nx {
                let want = 280.0 + 0.1 * i as f64 - 0.05 * j as f64;
                assert!((out[j * nx + i] - want).abs() < 0.02, "{i},{j} {} {want}", out[j * nx + i]);
            }
        }
    }

    #[test]
    fn missing_cells_stay_missing() {
        let (nx, ny) = (40, 40);
        let mut vals = vec![290.0; nx * ny];
        let mut mask = vec![FIXED; nx * ny];
        mask[5] = MISSING;
        vals[5] = f64::NAN;
        for c in 100..300 {
            mask[c] = UNKNOWN;
        }
        let out = solve_cpu(nx, ny, vals, mask);
        assert!(out[5].is_nan());
        for c in 100..300 {
            assert!((out[c] - 290.0).abs() < 1e-9);
        }
    }
}
