//! Which target water body owns each source cell, in float64.
//!
//! `gpuwm.ingest.water_temperature._component_owner_of_source`, a NumPy
//! `np.unique` over (source cell, body) pairs followed by a Python loop
//! over every pair, moved here statement for statement
//! (`gpuwm_component_owner_f64`).  The NumPy code is kept as a test
//! oracle only (`gpuwm/verify/water_blend_oracle.py`), and every output
//! must equal it byte for byte:
//!
//! * a target's source row and column are `np.rint` (ties to even) of the
//!   same float64 expressions, the longitude first brought into the
//!   source's window with `np.mod`'s sign rule, and a target whose index
//!   is not finite or falls off the source owns nothing;
//! * a source cell goes to the body holding most of the targets nearest
//!   to it, and among bodies holding equally many to the highest label.
//!   That is what the loop does when it visits the pairs in a stable
//!   order of their counts; with NumPy's default unstable `argsort` the
//!   winner of a tie depended on the sort kernel of the machine, so the
//!   oracle now sorts stably and the tie has one answer everywhere;
//! * the pairs are counted after an exact integer sort, so no result
//!   depends on the worker count.

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{worker_ranges, ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

/// `np.mod(value, 360.0)`: C `fmod`, moved into `[0, 360)` when its sign
/// differs from the divisor's, and `+0.0` for an exact zero.
fn numpy_mod_360(value: f64) -> f64 {
    let modulo = value % 360.0;
    if modulo != 0.0 {
        if modulo < 0.0 {
            modulo + 360.0
        } else {
            modulo
        }
    } else if modulo == 0.0 {
        0.0
    } else {
        modulo
    }
}

/// The owner of every source cell (0 = none).
///
/// `labels`, `target_lat` and `target_lon` hold `ntarget` values; the
/// source axes are given by their first two values, as the NumPy code
/// read them.  `owner` receives `sny * snx` labels.
///
/// # Safety
/// Every pointer must reference a buffer of the stated length.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_component_owner_f64(
    labels: *const i32,
    target_lat: *const f64,
    target_lon: *const f64,
    ntarget: usize,
    lat0: f64,
    lat1: f64,
    lon0: f64,
    lon1: f64,
    sny: usize,
    snx: usize,
    owner: *mut i32,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if workers == 0 {
            return ERR_DIMENSION;
        }
        let cells = match sny.checked_mul(snx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        if cells > u32::MAX as usize {
            return ERR_DIMENSION;
        }
        if cells > 0 && owner.is_null() {
            return ERR_NULL;
        }
        if ntarget > 0 && (labels.is_null() || target_lat.is_null() || target_lon.is_null()) {
            return ERR_NULL;
        }
        if cells == 0 {
            return OK;
        }
        let owner = std::slice::from_raw_parts_mut(owner, cells);
        owner.fill(0);
        if ntarget == 0 {
            return OK;
        }
        let labels = std::slice::from_raw_parts(labels, ntarget);
        let tlat = std::slice::from_raw_parts(target_lat, ntarget);
        let tlon = std::slice::from_raw_parts(target_lon, ntarget);
        let dlat = lat1 - lat0;
        let dlon = lon1 - lon0;
        let (ny, nx) = (sny as f64, snx as f64);
        // (cell << 32 | label) for every target inside, in target order
        // per worker range; the sort below makes the order immaterial.
        let ranges = worker_ranges(ntarget, workers.max(1));
        let key = |k: usize| -> Option<u64> {
            let label = labels[k];
            if label <= 0 {
                return None;
            }
            let y = ((tlat[k] - lat0) / dlat).round_ties_even();
            let shifted = lon0 + numpy_mod_360(tlon[k] - lon0 + 180.0) - 180.0;
            let x = ((shifted - lon0) / dlon).round_ties_even();
            if !(y >= 0.0 && y < ny && x >= 0.0 && x < nx) {
                return None;
            }
            let cell = (y as u64) * (snx as u64) + (x as u64);
            Some((cell << 32) | label as u64)
        };
        let mut keys: Vec<u64> = if ranges.len() <= 1 {
            (0..ntarget).filter_map(key).collect()
        } else {
            let parts: Vec<Vec<u64>> = std::thread::scope(|scope| {
                let handles: Vec<_> = ranges
                    .iter()
                    .map(|&(start, end)| scope.spawn(move || (start..end).filter_map(key).collect::<Vec<u64>>()))
                    .collect();
                handles.into_iter().map(|h| h.join().expect("owner worker")).collect()
            });
            parts.concat()
        };
        keys.sort_unstable();
        // One run per (cell, label), labels ascending within a cell: the
        // later of two equal counts wins, as in the stable-order loop.
        let mut best = 0u64;
        let mut current_cell = u64::MAX;
        let mut start = 0usize;
        while start < keys.len() {
            let pair = keys[start];
            let mut end = start + 1;
            while end < keys.len() && keys[end] == pair {
                end += 1;
            }
            let count = (end - start) as u64;
            let cell = pair >> 32;
            let label = (pair & 0xffff_ffff) as i32;
            if cell != current_cell {
                current_cell = cell;
                best = 0;
            }
            if count >= best {
                best = count;
                owner[cell as usize] = label;
            }
            start = end;
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_mod_matches_the_sign_rule() {
        assert_eq!(numpy_mod_360(-1.0), 359.0);
        assert_eq!(numpy_mod_360(361.0), 1.0);
        assert_eq!(numpy_mod_360(-360.0).to_bits(), 0.0f64.to_bits());
        assert!(numpy_mod_360(f64::NAN).is_nan());
        assert!(numpy_mod_360(f64::INFINITY).is_nan());
    }

    #[test]
    fn a_tie_goes_to_the_higher_label() {
        let labels = [1, 2, 2, 1];
        let lat = [0.0, 0.0, 1.0, 1.0];
        let lon = [0.0, 0.0, 1.0, 1.0];
        let mut owner = [9i32; 4];
        for workers in [1, 3] {
            let code = unsafe {
                gpuwm_component_owner_f64(
                    labels.as_ptr(),
                    lat.as_ptr(),
                    lon.as_ptr(),
                    4,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    2,
                    2,
                    owner.as_mut_ptr(),
                    workers,
                )
            };
            assert_eq!(code, OK);
            assert_eq!(owner, [2, 0, 0, 2]);
        }
    }
}
