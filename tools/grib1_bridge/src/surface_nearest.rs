//! The bounded WPS surface-nearest search of the CPU preprocessing backend.
//!
//! `gpuwm.ingest.preprocess_backend._masked_nearest_cpu`, a NumPy window
//! scan over whole target fields, moved here statement for statement
//! (`gpuwm_masked_nearest_f32`).  The NumPy code is kept as a test oracle
//! only (`gpuwm/verify/water_blend_oracle.py`), and every output must
//! equal it byte for byte:
//!
//! * the target coordinates are rounded to float32 and their centre cell
//!   is `np.rint` (ties to even) of that float32;
//! * a squared distance is float64, `(y - j) * (y - j) + (x - i) * (x - i)`
//!   with `y` the float32 coordinate, because NumPy promotes a float32
//!   minus an int32 to float64 and squares it by multiplication;
//! * the window is scanned row offset outer, column offset inner, and a
//!   candidate is taken only when strictly nearer, so the first of equal
//!   distances wins;
//! * every output element depends only on its own target, so no result
//!   depends on the worker count.

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{worker_ranges, ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

/// The surface each active target reads, from its `surface` code:
/// 0 matches the target's own land mask, 1 reads land for land targets,
/// 2 reads water for water targets.
fn wanted(surface: i32, target_land: bool) -> Option<(bool, bool)> {
    match surface {
        0 => Some((true, target_land)),
        1 => Some((target_land, true)),
        2 => Some((!target_land, false)),
        _ => None,
    }
}

/// The nearest source cell of the wanted surface within `radius` of each
/// target's centre cell, or `fill_value`.
///
/// `unmatched` receives the number of active targets that found none.
///
/// # Safety
/// Every pointer must reference a buffer of the stated length.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_masked_nearest_f32(
    field: *const f32,
    source_land: *const u8,
    sny: usize,
    snx: usize,
    target_y: *const f64,
    target_x: *const f64,
    target_land: *const u8,
    ntarget: usize,
    surface: i32,
    fill_value: f64,
    radius: usize,
    output: *mut f32,
    unmatched: *mut u64,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if unmatched.is_null() {
            return ERR_NULL;
        }
        *unmatched = 0;
        if workers == 0 || wanted(surface, false).is_none() || radius > i32::MAX as usize {
            return ERR_DIMENSION;
        }
        let cells = match sny.checked_mul(snx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        if ntarget == 0 {
            return OK;
        }
        if cells == 0 || sny > i32::MAX as usize || snx > i32::MAX as usize {
            return ERR_DIMENSION;
        }
        if field.is_null()
            || source_land.is_null()
            || target_y.is_null()
            || target_x.is_null()
            || target_land.is_null()
            || output.is_null()
        {
            return ERR_NULL;
        }
        let field = std::slice::from_raw_parts(field, cells);
        let source_land = std::slice::from_raw_parts(source_land, cells);
        let ty = std::slice::from_raw_parts(target_y, ntarget);
        let tx = std::slice::from_raw_parts(target_x, ntarget);
        let target_land = std::slice::from_raw_parts(target_land, ntarget);
        let output = std::slice::from_raw_parts_mut(output, ntarget);
        let fill = fill_value as f32;
        let r = radius as i64;
        let (ny, nx) = (sny as i64, snx as i64);
        let search = |k: usize| -> (f32, bool) {
            let (active, desired) = wanted(surface, target_land[k] != 0).expect("checked");
            let y = ty[k] as f32;
            let x = tx[k] as f32;
            let center_y = y.round_ties_even() as i32 as i64;
            let center_x = x.round_ties_even() as i32 as i64;
            let (yf, xf) = (y as f64, x as f64);
            let mut best_distance = f64::INFINITY;
            let mut best_value = fill;
            if active {
                for dj in -r..=r {
                    let jy = center_y + dj;
                    if jy < 0 || jy >= ny {
                        continue;
                    }
                    for di in -r..=r {
                        let ix = center_x + di;
                        if ix < 0 || ix >= nx {
                            continue;
                        }
                        let cell = (jy * nx + ix) as usize;
                        let value = field[cell];
                        if !value.is_finite() || (source_land[cell] != 0) != desired {
                            continue;
                        }
                        let dy = yf - jy as f64;
                        let dx = xf - ix as f64;
                        let distance = dy * dy + dx * dx;
                        if distance < best_distance {
                            best_distance = distance;
                            best_value = value;
                        }
                    }
                }
            }
            (best_value, active && !best_distance.is_finite())
        };
        let ranges = worker_ranges(ntarget, workers);
        let mut pieces: Vec<&mut [f32]> = Vec::with_capacity(ranges.len());
        let mut rest = output;
        for &(start, end) in ranges.iter() {
            let (head, tail) = rest.split_at_mut(end - start);
            pieces.push(head);
            rest = tail;
        }
        let missing: u64 = std::thread::scope(|scope| {
            let handles: Vec<_> = ranges
                .iter()
                .zip(pieces)
                .map(|(&(start, _), piece)| {
                    let search = &search;
                    scope.spawn(move || {
                        let mut missing = 0u64;
                        for (offset, slot) in piece.iter_mut().enumerate() {
                            let (value, unfound) = search(start + offset);
                            *slot = value;
                            missing += unfound as u64;
                        }
                        missing
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("nearest worker")).sum()
        });
        *unmatched = missing;
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_of_equal_distances_wins_and_water_skips_land() {
        // 2x3 source; the target sits between columns 0 and 2 of row 0.
        let field = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let land = [0u8, 1, 0, 1, 1, 1];
        let ty = [0.0f64];
        let tx = [1.0f64];
        let tland = [0u8];
        for workers in [1, 3] {
            let mut out = [0f32];
            let mut unmatched = 9u64;
            let code = unsafe {
                gpuwm_masked_nearest_f32(
                    field.as_ptr(),
                    land.as_ptr(),
                    2,
                    3,
                    ty.as_ptr(),
                    tx.as_ptr(),
                    tland.as_ptr(),
                    1,
                    0,
                    -1.0,
                    1,
                    out.as_mut_ptr(),
                    &mut unmatched,
                    workers,
                )
            };
            assert_eq!(code, OK);
            assert_eq!(out, [1.0]);
            assert_eq!(unmatched, 0);
        }
    }
}
