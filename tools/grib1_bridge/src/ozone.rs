//! WRF latitude interpolation and monthly blending without full-grid temporaries.
//! Every f32 subtraction, division, product and addition retains its own rounding.
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::{parallel, ERR_DIMENSION, ERR_NONFINITE, ERR_NULL, ERR_PANIC, OK};

/// An unusual external climatology retains the original checked host route.
pub const UNSUPPORTED: i32 = 126;

unsafe fn latitude_values<'a>(values: *const f32, count: usize) -> Result<&'a [f32], i32> {
    if count > isize::MAX as usize / 4 { return Err(ERR_DIMENSION); }
    if count == 0 { return Ok(&[]); }
    if values.is_null() { return Err(ERR_NULL); }
    let values = std::slice::from_raw_parts(values, count);
    if values.iter().any(|v| !v.is_finite()) { return Err(ERR_NONFINITE); }
    Ok(values)
}

/// Validate XLAT before calendar scalar conversion, preserving error order.
/// # Safety
/// `values` holds `count` immutable native-endian f32 elements until return.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_ozone_validate_latitudes_f32(values: *const f32, count: usize) -> i32 {
    catch_unwind(AssertUnwindSafe(|| latitude_values(values, count).map_or_else(|e| e, |_| OK)))
        .unwrap_or(ERR_PANIC)
}

unsafe fn inputs<'a>(values: *const f32, count: usize, latitude: *const f32,
    nlat: usize, ozone: *const f32, levels: usize, months: usize,
) -> Result<(&'a [f32], &'a [f32], &'a [f32]), i32> {
    let values = latitude_values(values, count)?;
    if nlat < 2 || levels == 0 || months != 12 { return Err(ERR_DIMENSION); }
    let length = levels.checked_mul(nlat).and_then(|v| v.checked_mul(months))
        .filter(|&v| v <= isize::MAX as usize / 4).ok_or(ERR_DIMENSION)?;
    if latitude.is_null() || ozone.is_null() { return Err(ERR_NULL); }
    let latitude = std::slice::from_raw_parts(latitude, nlat);
    let ozone = std::slice::from_raw_parts(ozone, length);
    if latitude.iter().any(|v| !v.is_finite())
        || latitude.windows(2).any(|v| v[0] >= v[1])
        || ozone.iter().any(|v| !v.is_finite()) { return Err(UNSUPPORTED); }
    Ok((values, latitude, ozone))
}

fn bracket(latitude: &[f32], point: f32) -> usize {
    // searchsorted(side="left") - 1, with extrapolating edge intervals.
    latitude.partition_point(|&v| v < point).saturating_sub(1).min(latitude.len() - 2)
}

#[inline]
fn interpolate(ozone: &[f32], nlat: usize, level: usize, month: usize,
    lower: usize, denominator: f32, delta: f32,
) -> Result<f32, i32> {
    let index = (level * nlat + lower) * 12 + month;
    let low = ozone[index];
    let high = ozone[index + 12];
    let difference = high - low;
    let slope = difference / denominator;
    let product = slope * delta;
    let value = low + product;
    if [difference, slope, product, value].iter().any(|v| !v.is_finite()) {
        return Err(UNSUPPORTED);
    }
    Ok(value)
}

fn interval_envelopes(latitude: &[f32], ozone: &[f32], nlat: usize,
    levels: usize,
) -> Result<Vec<(f32, f32)>, i32> {
    // The original all-month expression can warn in an unused month. These
    // tiny per-interval envelopes conservatively prove every month finite,
    // without computing twelve full latitude output grids.
    let mut envelopes = Vec::with_capacity(nlat - 1);
    for lower in 0..nlat - 1 {
        let denominator = latitude[lower + 1] - latitude[lower];
        if !denominator.is_finite() { return Err(UNSUPPORTED); }
        let (mut max_slope, mut max_low) = (0.0f32, 0.0f32);
        for level in 0..levels {
            for month in 0..12 {
                let index = (level * nlat + lower) * 12 + month;
                let low = ozone[index];
                let difference = ozone[index + 12] - low;
                let slope = difference / denominator;
                if !difference.is_finite() || !slope.is_finite() { return Err(UNSUPPORTED); }
                max_slope = max_slope.max(slope.abs());
                max_low = max_low.max(low.abs());
            }
        }
        envelopes.push((max_slope, max_low));
    }
    Ok(envelopes)
}

/// Latitude interpolation for all twelve months, in point/level/month order.
/// # Safety
/// Inputs hold their declared immutable f32 arrays, ozone is C-order
/// (levels,nlat,12), and output is writable (count,levels,12) without overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_ozone_latitude_f32(values: *const f32, count: usize,
    latitude: *const f32, nlat: usize, ozone: *const f32, levels: usize,
    months: usize, workers: usize, output: *mut f32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let (values, latitude, ozone) = match inputs(values, count, latitude, nlat, ozone, levels, months) {
            Ok(v) => v, Err(e) => return e,
        };
        let length = match count.checked_mul(levels).and_then(|v| v.checked_mul(months)) {
            Some(v) if v <= isize::MAX as usize / 4 => v, _ => return ERR_DIMENSION,
        };
        if workers == 0 { return ERR_DIMENSION; }
        if length > 0 && output.is_null() { return ERR_NULL; }
        let output = output as usize;
        let unsupported = AtomicBool::new(false);
        parallel::run_ranges(count, workers, |start, stop| {
            for point in start..stop {
                let lower = bracket(latitude, values[point]);
                let denominator = latitude[lower + 1] - latitude[lower];
                let delta = values[point] - latitude[lower];
                if !denominator.is_finite() || !delta.is_finite() {
                    unsupported.store(true, Ordering::Relaxed); return;
                }
                for level in 0..levels {
                    for month in 0..months {
                        let value = match interpolate(ozone, nlat, level, month, lower, denominator, delta) {
                            Ok(v) => v, Err(_) => { unsupported.store(true, Ordering::Relaxed); return; },
                        };
                        *((output as *mut f32).add((point * levels + level) * months + month)) = value;
                    }
                }
            }
        });
        if unsupported.load(Ordering::Relaxed) { UNSUPPORTED } else { OK }
    })).unwrap_or(ERR_PANIC)
}

/// Compute exactly the two latitude-interpolated months needed by ozn_time_int.
/// # Safety
/// Inputs hold their declared immutable f32 arrays, ozone is C-order
/// (levels,nlat,12), and output is writable (count,levels) without overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_ozone_latitude_time_f32(values: *const f32, count: usize,
    latitude: *const f32, nlat: usize, ozone: *const f32, levels: usize,
    months: usize, previous: usize, next: usize, fact1: f32, fact2: f32,
    workers: usize, output: *mut f32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let (values, latitude, ozone) = match inputs(values, count, latitude, nlat, ozone, levels, months) {
            Ok(v) => v, Err(e) => return e,
        };
        let length = match count.checked_mul(levels) {
            Some(v) if v <= isize::MAX as usize / 4 => v, _ => return ERR_DIMENSION,
        };
        if workers == 0 || previous >= months || next >= months { return ERR_DIMENSION; }
        if length > 0 && output.is_null() { return ERR_NULL; }
        let envelopes = match interval_envelopes(latitude, ozone, nlat, levels) {
            Ok(v) => v, Err(e) => return e,
        };
        let output = output as usize;
        let unsupported = AtomicBool::new(false);
        parallel::run_ranges(count, workers, |start, stop| {
            for point in start..stop {
                let lower = bracket(latitude, values[point]);
                let denominator = latitude[lower + 1] - latitude[lower];
                let delta = values[point] - latitude[lower];
                let bound = envelopes[lower].0 * delta.abs();
                let bound_sum = envelopes[lower].1 + bound;
                if !denominator.is_finite() || !delta.is_finite() || !bound.is_finite() || !bound_sum.is_finite() {
                    unsupported.store(true, Ordering::Relaxed); return;
                }
                for level in 0..levels {
                    let old = match interpolate(ozone, nlat, level, previous, lower, denominator, delta) {
                        Ok(v) => v, Err(_) => { unsupported.store(true, Ordering::Relaxed); return; },
                    };
                    let new = match interpolate(ozone, nlat, level, next, lower, denominator, delta) {
                        Ok(v) => v, Err(_) => { unsupported.store(true, Ordering::Relaxed); return; },
                    };
                    let old_weighted = old * fact1;
                    let new_weighted = new * fact2;
                    let value = old_weighted + new_weighted;
                    if !old_weighted.is_finite() || !new_weighted.is_finite() || !value.is_finite() {
                        unsupported.store(true, Ordering::Relaxed); return;
                    }
                    *((output as *mut f32).add(point * levels + level)) = value;
                }
            }
        });
        if unsupported.load(Ordering::Relaxed) { UNSUPPORTED } else { OK }
    })).unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn xlat_validation_empty_nonfinite_and_null_are_distinct() {
        assert_eq!(unsafe { gpuwm_ozone_validate_latitudes_f32(std::ptr::null(), 0) }, OK);
        assert_eq!(unsafe { gpuwm_ozone_validate_latitudes_f32(std::ptr::null(), 1) }, ERR_NULL);
        assert_eq!(unsafe { gpuwm_ozone_validate_latitudes_f32([f32::NAN].as_ptr(), 1) }, ERR_NONFINITE);
    }
    #[test]
    fn endpoints_select_the_left_interval_and_extrapolate() {
        let lat = [-2.0, 0.0, 2.0];
        assert_eq!(bracket(&lat, -3.0), 0);
        assert_eq!(bracket(&lat, 0.0), 0);
        assert_eq!(bracket(&lat, 2.0), 1);
        assert_eq!(bracket(&lat, 3.0), 1);
    }
    #[test]
    fn unused_month_overflow_declines_the_fused_route() {
        let lat = [-2.0f32, 0.0, 2.0];
        let mut ozone = vec![1.0f32; 3 * 12];
        ozone[0] = -f32::MAX;
        ozone[12] = f32::MAX;
        let mut output = [0.0f32];
        assert_eq!(unsafe { gpuwm_ozone_latitude_time_f32([-2.0f32].as_ptr(), 1,
            lat.as_ptr(), 3, ozone.as_ptr(), 1, 12, 8, 9, 0.5, 0.5, 1,
            output.as_mut_ptr()) }, UNSUPPORTED);
    }
}
