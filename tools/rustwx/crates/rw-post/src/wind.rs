//! Grid-relative to earth-relative wind rotation (specification 3.3).
//!
//! u_e = u cos a - v sin a,  v_e = v cos a + u sin a, with a the local map
//! rotation from the history's SINALPHA and COSALPHA (WRF ARW technical
//! note [R1] and User's Guide [R3]).  CPU twin of `woof_post_rotate_v1`;
//! `planes` levels of `ncell` words share one 2D sin/cos pair.

use rayon::prelude::*;

/// Rotate one pair.
#[inline]
pub fn rotate(u: f32, v: f32, sina: f32, cosa: f32) -> (f32, f32) {
    (u * cosa - v * sina, v * cosa + u * sina)
}

/// Rotate `planes` levels of `ncell` words on the CPU.
pub fn rotate_cpu(u: &[f32], v: &[f32], sina: &[f32], cosa: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let ncell = sina.len();
    assert_eq!(cosa.len(), ncell);
    assert_eq!(u.len(), v.len());
    assert_eq!(u.len() % ncell, 0);
    let pairs: Vec<(f32, f32)> = (0..u.len())
        .into_par_iter()
        .map(|at| {
            let c = at % ncell;
            rotate(u[at], v[at], sina[c], cosa[c])
        })
        .collect();
    pairs.into_iter().unzip()
}
