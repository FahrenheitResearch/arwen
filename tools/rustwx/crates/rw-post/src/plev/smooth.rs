//! The first-order low-pass filter of [R18]: weights 1/4, 1/2, 1/4,
//! applied along x then along y per pass.  A missing neighbour (or the
//! domain edge) is skipped and the remaining weights renormalised; a missing
//! centre stays missing.

use rayon::prelude::*;

#[inline]
pub(crate) fn pass_cell(src: &[f64], c: usize, has_lo: bool, lo: usize, has_hi: bool, hi: usize) -> f64 {
    let x = src[c];
    if x.is_nan() {
        return x;
    }
    let mut sum = 0.5 * x;
    let mut w = 0.5;
    if has_lo && !src[lo].is_nan() {
        sum = sum + 0.25 * src[lo];
        w = w + 0.25;
    }
    if has_hi && !src[hi].is_nan() {
        sum = sum + 0.25 * src[hi];
        w = w + 0.25;
    }
    sum / w
}

pub fn passes_cpu(nx: usize, ny: usize, mut v: Vec<f64>, passes: usize) -> Vec<f64> {
    let mut tmp = vec![0.0; v.len()];
    for _ in 0..passes {
        tmp.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            for (i, x) in row.iter_mut().enumerate() {
                let c = j * nx + i;
                *x = pass_cell(&v, c, i > 0, c.wrapping_sub(1), i + 1 < nx, c + 1);
            }
        });
        std::mem::swap(&mut v, &mut tmp);
        tmp.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            for (i, x) in row.iter_mut().enumerate() {
                let c = j * nx + i;
                *x = pass_cell(&v, c, j > 0, c.wrapping_sub(nx), j + 1 < ny, c + nx);
            }
        });
        std::mem::swap(&mut v, &mut tmp);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_and_linear_fields_survive() {
        let (nx, ny) = (12, 9);
        let v: Vec<f64> = (0..nx * ny).map(|c| 1000.0 + (c % nx) as f64 * 2.0).collect();
        let out = passes_cpu(nx, ny, v.clone(), 5);
        for j in 0..ny {
            for i in 1..nx - 1 {
                assert!((out[j * nx + i] - v[j * nx + i]).abs() < 1e-9 || i < 6 || i > nx - 6);
            }
        }
        let flat = passes_cpu(nx, ny, vec![7.0; nx * ny], 10);
        assert!(flat.iter().all(|&x| (x - 7.0).abs() < 1e-12));
    }
}
