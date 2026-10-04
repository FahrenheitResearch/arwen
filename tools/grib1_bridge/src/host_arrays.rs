//! Bounded parallel setup copies and the exact geopotential cache tree.
use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::{ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

/// # Safety
/// Source and target have `length` elements and do not overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_host_copy_f64_f32(
    source: *const f64, target: *mut f32, length: usize, workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if workers == 0 { return ERR_DIMENSION; }
        if length == 0 { return OK; }
        if source.is_null() || target.is_null() { return ERR_NULL; }
        let source = source as usize;
        let target = target as usize;
        crate::parallel::run_ranges(length, workers, |start, stop| {
            for i in start..stop {
                *((target as *mut f32).add(i)) = *((source as *const f64).add(i)) as f32;
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

/// Exact host pressure staggering. Each result has one writer, including
/// the copied outer faces; interior faces retain sum-then-half order.
///
/// # Safety
/// Source has `levels*rows*columns` f64 elements. Target has the matching
/// staggered shape, with one extra column (axis 2) or row (axis 1).
#[no_mangle]
pub unsafe extern "C" fn gpuwm_host_stagger_f64(
    source: *const f64, target: *mut f64, levels: usize, rows: usize,
    columns: usize, axis: u32, workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if levels == 0 || rows == 0 || columns == 0 || workers == 0
            || !(axis == 1 || axis == 2) { return ERR_DIMENSION; }
        let output_rows = match rows.checked_add(usize::from(axis == 1)) {
            Some(v) => v, None => return ERR_DIMENSION,
        };
        let output_columns = match columns.checked_add(usize::from(axis == 2)) {
            Some(v) => v, None => return ERR_DIMENSION,
        };
        let length = match levels.checked_mul(output_rows).and_then(|v| v.checked_mul(output_columns)) {
            Some(v) => v, None => return ERR_DIMENSION,
        };
        if source.is_null() || target.is_null() { return ERR_NULL; }
        let source = source as usize;
        let target = target as usize;
        crate::parallel::run_ranges(length, workers, |start, stop| {
            for i in start..stop {
                let column = i % output_columns;
                let row = (i / output_columns) % output_rows;
                let level = i / (output_columns * output_rows);
                let (first, second) = if axis == 2 {
                    let base = (level * rows + row) * columns;
                    (base + column.saturating_sub(1), base + column.min(columns - 1))
                } else {
                    let base = level * rows * columns + column;
                    (base + row.saturating_sub(1) * columns, base + row.min(rows - 1) * columns)
                };
                let value = *((source as *const f64).add(first));
                *((target as *mut f64).add(i)) = if first == second { value }
                    else { 0.5_f64 * (value + *((source as *const f64).add(second))) };
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

/// Sum fields in caller order after widening each operand, with no
/// reassociation or zero accumulator. Scratch is independent of volume.
///
/// # Safety
/// `sources` and `kinds` hold `count` entries. Each source has `length`
/// elements of f32 (kind 4) or f64 (kind 8); target has `length` f64s.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_host_sum_f64(
    sources: *const *const std::ffi::c_void, kinds: *const u32, count: usize,
    target: *mut f64, length: usize, workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if count == 0 || workers == 0 { return ERR_DIMENSION; }
        if length == 0 { return OK; }
        if sources.is_null() || kinds.is_null() || target.is_null() { return ERR_NULL; }
        let kinds = std::slice::from_raw_parts(kinds, count);
        if kinds.iter().any(|&v| v != 4 && v != 8) { return ERR_DIMENSION; }
        let sources: Vec<usize> = std::slice::from_raw_parts(sources, count).iter().map(|&v| v as usize).collect();
        if sources.contains(&0) { return ERR_NULL; }
        let target = target as usize;
        crate::parallel::run_ranges(length, workers, |start, stop| {
            let read = |field: usize, index: usize| -> f64 {
                if kinds[field] == 4 { *((sources[field] as *const f32).add(index)) as f64 }
                else { *((sources[field] as *const f64).add(index)) }
            };
            for i in start..stop {
                let mut value = read(0, i);
                for field in 1..count { value += read(field, i); }
                *((target as *mut f64).add(i)) = value;
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

/// Fuse the original f64 subtraction and f32 assignment, retaining both
/// rounding points while avoiding the full-volume temporary.
///
/// # Safety
/// Inputs have `length` f64 elements and target has `length` f32 elements;
/// target does not overlap either input.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_host_difference_f64_f32(
    first: *const f64, second: *const f64, target: *mut f32,
    length: usize, workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if workers == 0 { return ERR_DIMENSION; }
        if length == 0 { return OK; }
        if first.is_null() || second.is_null() || target.is_null() { return ERR_NULL; }
        let first = first as usize;
        let second = second as usize;
        let target = target as usize;
        crate::parallel::run_ranges(length, workers, |start, stop| {
            for i in start..stop {
                *((target as *mut f32).add(i)) =
                    (*((first as *const f64).add(i)) - *((second as *const f64).add(i))) as f32;
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

/// Store one independent column at a time, with the original subtraction
/// and rounding points. `minimum` is a positive minimum or NaN when the
/// caller must retain NumPy's unusual-value reduction semantics.
///
/// # Safety
/// Source, stored and snapshot have `(levels * columns)` elements;
/// residual has `((levels-1) * columns)` elements. Outputs do not overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_host_geopotential(
    source: *const f64, stored: *mut f32, snapshot: *mut f64,
    residual: *mut f32, levels: usize, columns: usize, gravity: f64,
    minimum: *mut f64, workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if levels == 0 || columns == 0 || workers == 0 || !gravity.is_finite() || gravity == 0.0 {
            return ERR_DIMENSION;
        }
        if source.is_null() || stored.is_null() || snapshot.is_null()
            || residual.is_null() || minimum.is_null() { return ERR_NULL; }
        let count = workers.min(crate::parallel::resources::available_workers()).min(columns).max(1);
        let ranges = crate::worker_ranges(columns, count);
        let mut minima = vec![f64::INFINITY; ranges.len()];
        let minima_address = minima.as_mut_ptr() as usize;
        let source = source as usize;
        let stored = stored as usize;
        let snapshot = snapshot as usize;
        let residual = residual as usize;
        crate::parallel::run_ranges(ranges.len(), count, |begin, end| {
            for slot in begin..end {
                let mut smallest = f64::INFINITY;
                let mut unusual = false;
                let (start, stop) = ranges[slot];
                for column in start..stop {
                    let mut previous_height = 0.0;
                    for k in 0..levels {
                        let index = k * columns + column;
                        let value = *((source as *const f64).add(index));
                        *((stored as *mut f32).add(index)) = value as f32;
                        *((snapshot as *mut f64).add(index)) = value;
                        if k + 1 < levels {
                            let next = *((source as *const f64).add(index + columns));
                            let difference = next as f32 - value as f32;
                            *((residual as *mut f32).add(index)) = ((next - value) - difference as f64) as f32;
                            let height = (0.5_f64 * (value + next)) / gravity;
                            if k > 0 {
                                let spacing = height - previous_height;
                                if !spacing.is_finite() || spacing <= 0.0 { unusual = true; }
                                smallest = smallest.min(spacing);
                            }
                            previous_height = height;
                        }
                    }
                }
                *((minima_address as *mut f64).add(slot)) = if unusual { f64::NAN } else { smallest };
            }
        });
        *minimum = if minima.iter().any(|v| v.is_nan()) { f64::NAN }
            else { minima.into_iter().fold(f64::INFINITY, f64::min) };
        OK
    })).unwrap_or(ERR_PANIC)
}

/// Each column's `field` value at the level of that column's greatest
/// `pressure`: the surface pseudo-level WPS takes for a number field that
/// has no two-metre product, whichever way the source levels run.
///
/// The level is the FIRST one holding the column's maximum, and a NaN
/// pressure counts as the maximum (the first NaN wins), which is the order
/// `numpy.argmax` reads a column in, so the selected bytes are the ones
/// the NumPy reference takes.  Each column has one writer.
///
/// # Safety
/// `pressure` has `levels * columns` elements of f32 (kind 4) or f64
/// (kind 8) and `field` has `levels * columns` f32 elements, both stored
/// level-major; `target` has `columns` f32 elements and overlaps neither.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_host_deepest_level_f32(
    pressure: *const std::ffi::c_void, pressure_kind: u32, field: *const f32,
    target: *mut f32, levels: usize, columns: usize, workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if levels == 0 || columns == 0 || workers == 0
            || !(pressure_kind == 4 || pressure_kind == 8)
            || levels.checked_mul(columns).is_none() { return ERR_DIMENSION; }
        if pressure.is_null() || field.is_null() || target.is_null() { return ERR_NULL; }
        let pressure = pressure as usize;
        let field = field as usize;
        let target = target as usize;
        crate::parallel::run_ranges(columns, workers, |start, stop| {
            let read = |index: usize| -> f64 {
                if pressure_kind == 4 { *((pressure as *const f32).add(index)) as f64 }
                else { *((pressure as *const f64).add(index)) }
            };
            for column in start..stop {
                let mut deepest = 0usize;
                let mut greatest = read(column);
                if !greatest.is_nan() {
                    for level in 1..levels {
                        let value = read(level * columns + column);
                        // Negated on purpose: a NaN takes the column, and
                        // an equal value leaves the first level in place.
                        if !(value <= greatest) {
                            greatest = value;
                            deepest = level;
                            if value.is_nan() { break; }
                        }
                    }
                }
                *((target as *mut f32).add(column)) =
                    *((field as *const f32).add(deepest * columns + column));
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn select(pressure: &[f32], field: &[f32], levels: usize, columns: usize, workers: usize) -> Vec<f32> {
        let mut target = vec![f32::NAN; columns];
        let status = unsafe {
            gpuwm_host_deepest_level_f32(pressure.as_ptr() as *const std::ffi::c_void, 4,
                field.as_ptr(), target.as_mut_ptr(), levels, columns, workers)
        };
        assert_eq!(status, OK);
        target
    }

    #[test]
    fn the_deepest_level_follows_either_level_order_ties_and_nan() {
        // Four columns, three levels, level-major.  Column 0 has pressure
        // falling with level (deepest first), column 1 rising (deepest
        // last), column 2 a tie between levels 0 and 1 (the first wins),
        // column 3 a NaN at level 1 (the first NaN wins over a later max).
        let pressure = [
            900.0_f32, 100.0, 500.0, 200.0,
            500.0, 500.0, 500.0, f32::NAN,
            100.0, 900.0, 200.0, 950.0,
        ];
        let field = [
            1.0_f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        for workers in [1, 2, 8] {
            assert_eq!(select(&pressure, &field, 3, 4, workers), vec![1.0, 10.0, 3.0, 8.0]);
        }
        // A NaN at the first level takes its column.
        assert_eq!(select(&[f32::NAN, 7.0], &[1.0, 2.0], 2, 1, 1), vec![1.0]);
        // Signed zeros compare equal: the first level stays.
        assert_eq!(select(&[-0.0, 0.0], &[1.0, 2.0], 2, 1, 1), vec![1.0]);
    }

    #[test]
    fn the_deepest_level_reads_double_pressure_and_refuses_bad_shapes() {
        let pressure = [100.0_f64, 900.0, 900.0, 100.0];
        let field = [1.0_f32, 2.0, 3.0, 4.0];
        let mut target = [0.0_f32; 2];
        let run = |kind: u32, levels: usize, columns: usize, workers: usize, target: &mut [f32]| unsafe {
            gpuwm_host_deepest_level_f32(pressure.as_ptr() as *const std::ffi::c_void, kind,
                field.as_ptr(), target.as_mut_ptr(), levels, columns, workers)
        };
        assert_eq!(run(8, 2, 2, 4, &mut target), OK);
        assert_eq!(target, [3.0, 2.0]);
        assert_eq!(run(2, 2, 2, 1, &mut target), ERR_DIMENSION);
        assert_eq!(run(8, 0, 2, 1, &mut target), ERR_DIMENSION);
        assert_eq!(run(8, 2, 0, 1, &mut target), ERR_DIMENSION);
        assert_eq!(run(8, 2, 2, 0, &mut target), ERR_DIMENSION);
        assert_eq!(unsafe {
            gpuwm_host_deepest_level_f32(std::ptr::null(), 8, field.as_ptr(),
                target.as_mut_ptr(), 2, 2, 1)
        }, ERR_NULL);
    }
}
