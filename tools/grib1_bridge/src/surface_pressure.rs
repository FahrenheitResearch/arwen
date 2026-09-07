//! WRF real's sfcprs3: sea-level/profile pressure at the target terrain.
//!
//! The caller's setup arrays are stored as f64. Read each value as WRF REAL
//! (f32) within its column, avoiding full-frame f32 copies or a reordered
//! profile. The surface pseudo-level is absent from these arrays: source
//! index zero corresponds to WRF k=2. Loop bounds retain WRF's excluded top
//! levels, including the distinct fallback bound.

use std::panic::{catch_unwind, AssertUnwindSafe};

fn column(
    height: &[f64],
    pressure: &[f64],
    terrain: f64,
    slp: f64,
    order: &[u64],
    columns: usize,
    index: usize,
) -> Result<f32, i32> {
    let z = |k: usize| height[order[k] as usize * columns + index] as f32;
    let p = |k: usize| pressure[order[k] as usize * columns + index] as f32;
    let zm = terrain as f32;
    let slp = slp as f32;
    if !zm.is_finite() || !slp.is_finite() || slp <= 0.0 {
        return Err(crate::ERR_NONFINITE);
    }
    for k in 0..order.len() {
        if !z(k).is_finite() || !p(k).is_finite() || p(k) <= 0.0 {
            return Err(crate::ERR_NONFINITE);
        }
        if k > 0 && p(k - 1) <= p(k) {
            return Err(crate::ERR_PRESSURE_ORDER);
        }
    }
    let interpolate = |zl: f32, zu: f32, pl: f32, pu: f32| {
        ((pl.ln() * (zm - zu) + pu.ln() * (zl - zm)) / (zl - zu)).exp()
    };
    let result = if zm < 50.0 {
        slp + (p(0) - p(1)) / (z(0) - z(1)) * zm
    } else {
        let bracket = (0..order.len().saturating_sub(2)).find(|&k| z(k) <= zm && z(k + 1) > zm);
        if let Some(k) = bracket {
            interpolate(z(k), z(k + 1), p(k), p(k + 1))
        } else if slp >= p(0) {
            // WRF intentionally uses its k=3 (second isobaric level).
            interpolate(0.0, z(1), slp, p(1))
        } else {
            let bracket =
                (0..order.len().saturating_sub(3)).find(|&k| slp >= p(k + 1) && slp < p(k));
            match bracket {
                Some(k) => interpolate(0.0, z(k + 1), slp, p(k + 1)),
                None => return Err(crate::ERR_INTERPOLATION_WINDOW),
            }
        }
    };
    if result.is_finite() && result > 0.0 {
        Ok(result)
    } else {
        Err(crate::ERR_NONFINITE)
    }
}

/// Additive ABI-v1 entry; existing preprocessing calls are unchanged.
/// On failure, output is untouched and error_column names the first invalid
/// flattened column (usize::MAX denotes an invalid call shape).
///
/// # Safety
/// height/pressure address nsource*ncolumn f64 values in level-major order;
/// terrain/slp address ncolumn f64 values; order addresses nsource u64s;
/// output addresses ncolumn writable f32s and error_column one writable usize.
/// All pointers must be valid and correctly aligned for the indicated types.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_wrf_sfcprs3_from_f64(
    height: *const f64,
    pressure: *const f64,
    terrain: *const f64,
    slp: *const f64,
    order: *const u64,
    output: *mut f32,
    nsource: usize,
    ncolumn: usize,
    workers: usize,
    error_column: *mut usize,
) -> i32 {
    if height.is_null()
        || pressure.is_null()
        || terrain.is_null()
        || slp.is_null()
        || order.is_null()
        || output.is_null()
        || error_column.is_null()
    {
        return crate::ERR_NULL;
    }
    *error_column = usize::MAX;
    let Some(length) = nsource.checked_mul(ncolumn) else {
        return crate::ERR_DIMENSION;
    };
    if nsource < 2
        || ncolumn == 0
        || workers == 0
        || length > isize::MAX as usize / std::mem::size_of::<f64>()
    {
        return crate::ERR_DIMENSION;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let height = std::slice::from_raw_parts(height, length);
        let pressure = std::slice::from_raw_parts(pressure, length);
        let terrain = std::slice::from_raw_parts(terrain, ncolumn);
        let slp = std::slice::from_raw_parts(slp, ncolumn);
        let order = std::slice::from_raw_parts(order, nsource);
        let mut seen = vec![false; nsource];
        for &level in order {
            if level >= nsource as u64 || seen[level as usize] {
                return crate::ERR_DIMENSION;
            }
            seen[level as usize] = true;
        }
        let mut values = vec![0.0; ncolumn];
        let failures = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            let mut remaining = values.as_mut_slice();
            for (start, stop) in crate::worker_ranges(ncolumn, workers) {
                let (part, rest) = remaining.split_at_mut(stop - start);
                remaining = rest;
                handles.push(scope.spawn(move || {
                    for (offset, value) in part.iter_mut().enumerate() {
                        let index = start + offset;
                        match column(
                            height,
                            pressure,
                            terrain[index],
                            slp[index],
                            order,
                            ncolumn,
                            index,
                        ) {
                            Ok(result) => *value = result,
                            Err(code) => return Some((index, code)),
                        }
                    }
                    None
                }));
            }
            handles
                .into_iter()
                .filter_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        if let Some(&(index, code)) = failures.iter().min_by_key(|(index, _)| *index) {
            *error_column = index;
            return code;
        }
        std::ptr::copy_nonoverlapping(values.as_ptr(), output, ncolumn);
        crate::OK
    }))
    .unwrap_or(crate::ERR_PANIC)
}
