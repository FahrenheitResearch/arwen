//! Observation scoring with the original float64 operation order.
//!
//! Boxcar prefix sums are sequential along each axis. Reductions use the
//! NumPy float64 pairwise tree, including its eight accumulators and 128
//! element split. Packed float64 arrays are reduced in one tree: the
//! contiguous NumPy reduction does not use its configurable transfer buffer.
//! These are numerical contracts, not approximate metrics.

use std::cell::RefCell;
use std::slice;

mod rain_gate;

thread_local! { static ERROR: RefCell<String> = const { RefCell::new(String::new()) }; }

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_obsscore_source_revision() -> *const std::ffi::c_char {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=",
        env!("GPUWM_BRIDGE_SOURCE_REV"),
        "\0"
    );
    SOURCE_REV_STAMP.as_ptr().cast()
}

fn call(body: impl FnOnce() -> Result<(), String>) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
    match result {
        Ok(Ok(())) => {
            ERROR.with(|s| s.borrow_mut().clear());
            0
        }
        Ok(Err(message)) => {
            ERROR.with(|s| *s.borrow_mut() = message);
            -1
        }
        Err(_) => {
            ERROR.with(|s| *s.borrow_mut() = "observation scoring panicked".into());
            -1
        }
    }
}

unsafe fn input<'a, T>(pointer: *const T, len: usize) -> Result<&'a [T], String> {
    if len == 0 {
        return Ok(&[]);
    }
    if pointer.is_null() {
        return Err("null scoring input".into());
    }
    Ok(unsafe { slice::from_raw_parts(pointer, len) })
}

unsafe fn output<'a, T>(pointer: *mut T, len: usize) -> Result<&'a mut [T], String> {
    if len == 0 {
        return Ok(&mut []);
    }
    if pointer.is_null() {
        return Err("null scoring output".into());
    }
    Ok(unsafe { slice::from_raw_parts_mut(pointer, len) })
}

fn size(ny: usize, nx: usize) -> Result<usize, String> {
    ny.checked_mul(nx)
        .ok_or_else(|| "scoring grid size overflow".into())
}

fn sum(values: &[f64], buffer: usize) -> f64 {
    0.0 + rw_fieldcmp::stats::pairwise_sum_with_buffer(values, buffer)
}
fn mean(values: &[f64], buffer: usize) -> f64 {
    sum(values, buffer) / values.len() as f64
}
fn rmse(values: &[f64], buffer: usize) -> f64 {
    mean(&values.iter().map(|v| v * v).collect::<Vec<_>>(), buffer).sqrt()
}

fn edge_boxcar(field: &[f64], ny: usize, nx: usize, radius: usize) -> Vec<f64> {
    if radius == 0 {
        return field.to_vec();
    }
    rw_fieldcmp::gridops::Field::new(vec![ny, nx], field.to_vec())
        .unwrap()
        .boxcar(2 * radius + 1)
        .unwrap()
        .into_values()
}

fn boxcar(
    field: &[f64],
    ny: usize,
    nx: usize,
    radius: usize,
    boundary: u32,
) -> Result<Vec<f64>, String> {
    if boundary > 1 {
        return Err("unknown neighborhood boundary".into());
    }
    if radius == 0 {
        return Ok(field.to_vec());
    }
    if ny == 0 || nx == 0 {
        return Err("cannot extend an empty grid axis".into());
    }
    if boundary == 1 {
        return Ok(edge_boxcar(field, ny, nx, radius));
    }
    let py = ny
        .checked_add(2 * radius)
        .ok_or("neighborhood size overflow")?;
    let px = nx
        .checked_add(2 * radius)
        .ok_or("neighborhood size overflow")?;
    let mut padded = vec![0.0; size(py, px)?];
    for y in 0..ny {
        padded[(y + radius) * px + radius..(y + radius) * px + radius + nx]
            .copy_from_slice(&field[y * nx..(y + 1) * nx]);
    }
    let smoothed = edge_boxcar(&padded, py, px, radius);
    let mut result = Vec::with_capacity(field.len());
    for y in 0..ny {
        result.extend_from_slice(
            &smoothed[(y + radius) * px + radius..(y + radius) * px + radius + nx],
        );
    }
    Ok(result)
}

fn fractions(
    events: &[u8],
    valid: &[u8],
    ny: usize,
    nx: usize,
    radius: usize,
    boundary: u32,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let counted = boxcar(
        &events
            .iter()
            .zip(valid)
            .map(|(e, v)| f64::from(*e != 0 && *v != 0))
            .collect::<Vec<_>>(),
        ny,
        nx,
        radius,
        boundary,
    )?;
    let denominator = boxcar(
        &valid.iter().map(|v| f64::from(*v != 0)).collect::<Vec<_>>(),
        ny,
        nx,
        radius,
        boundary,
    )?;
    let fraction = counted
        .iter()
        .zip(&denominator)
        .map(|(n, d)| if *d > 0.0 { n / d } else { 0.0 })
        .collect();
    Ok((fraction, denominator))
}

fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    f64::from_bits(if value > 0.0 {
        value.to_bits() + 1
    } else {
        value.to_bits() - 1
    })
}

fn threshold(values: &[f64], target: f64) -> Result<f64, String> {
    if values.is_empty() {
        return Err("frequency matching has no valid cells to rank".into());
    }
    if !(0.0..=1.0).contains(&target) {
        return Err("target exceedance fraction must lie in [0, 1]".into());
    }
    if values.iter().any(|v| v.is_nan()) {
        return Ok(f64::NAN);
    }
    if target <= 0.0 {
        let mut high = values[0];
        for v in &values[1..] {
            if *v >= high {
                high = *v;
            }
        }
        return Ok(next_up(high));
    }
    if target >= 1.0 {
        let mut low = values[0];
        for v in &values[1..] {
            if *v <= low {
                low = *v;
            }
        }
        return Ok(low);
    }
    if values.len() == 1 {
        return Ok(values[0]);
    }
    let index = (values.len() - 1) as f64 * (1.0 - target);
    let previous = index.floor() as usize;
    let next = (previous + 1).min(values.len() - 1);
    let gamma = index - previous as f64;
    let mut selected = values.to_vec();
    let mixed_zero = values.iter().any(|v| v.to_bits() == 0)
        && values.iter().any(|v| v.to_bits() == (1u64 << 63));
    let (low, high) = if mixed_zero {
        // Stable equal-value ordering retains the original zero signs.
        selected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (selected[previous], selected[next])
    } else if previous == next {
        let ranked = rw_fieldcmp::stats::order_statistics(&mut selected, &[previous]);
        (ranked[0], ranked[0])
    } else {
        let ranked = rw_fieldcmp::stats::order_statistics(&mut selected, &[previous, next]);
        (ranked[0], ranked[1])
    };
    Ok(rw_fieldcmp::stats::quantile_interpolate(low, high, gamma))
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_obsscore_abi_version() -> u32 {
    1
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_last_error(dst: *mut u8, capacity: usize) -> usize {
    ERROR.with(|s| {
        let error = s.borrow();
        if !dst.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(error.as_ptr(), dst, error.len().min(capacity));
            }
        }
        error.len()
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_boxcar(
    field: *const f64,
    ny: usize,
    nx: usize,
    radius: usize,
    boundary: u32,
    result: *mut f64,
) -> i32 {
    call(|| {
        let n = size(ny, nx)?;
        let values = unsafe { input(field, n)? };
        let result = unsafe { output(result, n)? };
        result.copy_from_slice(&boxcar(values, ny, nx, radius, boundary)?);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_fraction(
    events: *const u8,
    valid: *const u8,
    ny: usize,
    nx: usize,
    radius: usize,
    boundary: u32,
    fraction: *mut f64,
    count: *mut f64,
) -> i32 {
    call(|| {
        let n = size(ny, nx)?;
        let (f, c) = fractions(
            unsafe { input(events, n)? },
            unsafe { input(valid, n)? },
            ny,
            nx,
            radius,
            boundary,
        )?;
        unsafe {
            output(fraction, n)?.copy_from_slice(&f);
            output(count, n)?.copy_from_slice(&c);
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_threshold(
    field: *const f64,
    valid: *const u8,
    n: usize,
    target: f64,
    result: *mut f64,
) -> i32 {
    call(|| {
        let field = unsafe { input(field, n)? };
        let valid = unsafe { input(valid, n)? };
        let values = field
            .iter()
            .zip(valid)
            .filter_map(|(v, m)| (*m != 0).then_some(*v))
            .collect::<Vec<_>>();
        unsafe {
            output(result, 1)?[0] = threshold(&values, target)?;
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_masked_fss(
    model: *const f64,
    obs: *const f64,
    valid: *const u8,
    scored: *const u8,
    ny: usize,
    nx: usize,
    cutoff: f64,
    radius: usize,
    boundary: u32,
    frequency_match: u32,
    reduction_buffer: usize,
    result: *mut f64,
    cells: *mut u64,
) -> i32 {
    call(|| {
        let n = size(ny, nx)?;
        let model = unsafe { input(model, n)? };
        let obs = unsafe { input(obs, n)? };
        let valid = unsafe { input(valid, n)? };
        let scored = unsafe { input(scored, n)? };
        if !valid.iter().any(|v| *v != 0) {
            return Err("FSS has no valid cells".into());
        }
        let denominator = valid
            .iter()
            .zip(scored)
            .filter(|(v, s)| **v != 0 && **s != 0)
            .count();
        if denominator == 0 {
            return Err("FSS has no cells inside both the mask and the region".into());
        }
        let oe = obs
            .iter()
            .zip(valid)
            .map(|(v, m)| u8::from(*v >= cutoff && *m != 0))
            .collect::<Vec<_>>();
        let observed_rate = oe
            .iter()
            .zip(scored)
            .filter(|(v, s)| **v != 0 && **s != 0)
            .count() as f64
            / denominator as f64;
        let cutoff_model = if frequency_match != 0 {
            threshold(
                &model
                    .iter()
                    .zip(valid)
                    .zip(scored)
                    .filter_map(|((v, m), s)| (*m != 0 && *s != 0).then_some(*v))
                    .collect::<Vec<_>>(),
                observed_rate,
            )?
        } else {
            cutoff
        };
        let me = model
            .iter()
            .zip(valid)
            .map(|(v, m)| u8::from(*v >= cutoff_model && *m != 0))
            .collect::<Vec<_>>();
        let model_rate = me
            .iter()
            .zip(scored)
            .filter(|(v, s)| **v != 0 && **s != 0)
            .count() as f64
            / denominator as f64;
        let (mf, counts) = fractions(&me, valid, ny, nx, radius, boundary)?;
        let observed_counted = boxcar(
            &oe.iter().map(|v| f64::from(*v != 0)).collect::<Vec<_>>(),
            ny,
            nx,
            radius,
            boundary,
        )?;
        let of = observed_counted
            .iter()
            .zip(&counts)
            .map(|(n, d)| if *d > 0.0 { n / d } else { 0.0 })
            .collect::<Vec<_>>();
        let mut numerator = Vec::new();
        let mut reference = Vec::new();
        for i in 0..n {
            if scored[i] != 0 && counts[i] > 0.0 {
                let d = mf[i] - of[i];
                numerator.push(d * d);
                reference.push(mf[i] * mf[i] + of[i] * of[i]);
            }
        }
        if numerator.is_empty() {
            return Err("no scored cell has a populated neighborhood".into());
        }
        let reference = sum(&reference, reduction_buffer);
        let fss = if reference == 0.0 {
            1.0
        } else {
            1.0 - sum(&numerator, reduction_buffer) / reference
        };
        if !fss.is_finite() {
            return Err("FSS is non-finite".into());
        }
        unsafe {
            output(result, 6)?.copy_from_slice(&[
                fss.clamp(0.0, 1.0),
                cutoff_model,
                cutoff,
                observed_rate,
                model_rate,
                0.5 + observed_rate / 2.0,
            ]);
            output(cells, 1)?[0] = numerator.len() as u64;
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_contingency(
    obs: *const f64,
    forecast: *const f64,
    valid: *const u8,
    n: usize,
    cutoff: f64,
    result: *mut u64,
) -> i32 {
    call(|| {
        let obs = unsafe { input(obs, n)? };
        let forecast = unsafe { input(forecast, n)? };
        let valid = unsafe { input(valid, n)? };
        let mut counts = [0u64; 4];
        for i in 0..n {
            if valid[i] != 0 {
                let o = obs[i] >= cutoff;
                let f = forecast[i] >= cutoff;
                counts[match (o, f) {
                    (true, true) => 0,
                    (true, false) => 1,
                    (false, true) => 2,
                    (false, false) => 3,
                }] += 1;
            }
        }
        if counts.iter().sum::<u64>() == 0 {
            return Err("a contingency table needs at least one valid cell".into());
        }
        unsafe {
            output(result, 4)?.copy_from_slice(&counts);
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_contingency_scores(
    counts: *const i64,
    result: *mut f64,
    defined: *mut u8,
) -> i32 {
    call(|| {
        let c = unsafe { input(counts, 4)? };
        let h = c[0] as i128;
        let m = c[1] as i128;
        let f = c[2] as i128;
        let q = c[3] as i128;
        let total = h + m + f + q;
        if total == 0 {
            return Err("a contingency table with no cells has no scores".into());
        }
        let random = ((h + m) * (h + f)) as f64 / total as f64;
        let numerators = [
            (h + m) as f64,
            (h + f) as f64,
            h as f64,
            f as f64,
            h as f64,
            (h + f) as f64,
            h as f64 - random,
            2.0 * ((h * q - m * f) as f64),
        ];
        let denominators = [
            total as f64,
            total as f64,
            (h + m) as f64,
            (h + f) as f64,
            (h + m + f) as f64,
            (h + m) as f64,
            (h + m + f) as f64 - random,
            ((h + m) * (m + q) + (h + f) * (f + q)) as f64,
        ];
        let r = unsafe { output(result, 8)? };
        let d = unsafe { output(defined, 8)? };
        for i in 0..8 {
            d[i] = u8::from(denominators[i] != 0.0);
            r[i] = if d[i] != 0 {
                numerators[i] / denominators[i]
            } else {
                0.0
            };
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_reduce(
    values: *const f64,
    n: usize,
    reduction_buffer: usize,
    result: *mut f64,
) -> i32 {
    call(|| {
        let values = unsafe { input(values, n)? };
        if n == 0 {
            return Err("RMSE over an empty sample is undefined".into());
        }
        let mut ranked = values.to_vec();
        ranked.sort_by(|a, b| a.total_cmp(b));
        let median = if n % 2 == 0 {
            mean(&ranked[n / 2 - 1..n / 2 + 1], reduction_buffer)
        } else {
            mean(&ranked[n / 2..n / 2 + 1], reduction_buffer)
        };
        unsafe {
            output(result, 3)?.copy_from_slice(&[
                mean(values, reduction_buffer),
                rmse(values, reduction_buffer),
                median,
            ]);
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_residuals(
    forecast: *const f64,
    obs: *const f64,
    n: usize,
    result: *mut f64,
    bad_index: *mut u64,
) -> i32 {
    call(|| {
        let forecast = unsafe { input(forecast, n)? };
        let obs = unsafe { input(obs, n)? };
        let result = unsafe { output(result, n)? };
        unsafe {
            output(bad_index, 1)?[0] = u64::MAX;
        }
        for i in 0..n {
            result[i] = forecast[i] - obs[i];
            if !result[i].is_finite() {
                unsafe {
                    output(bad_index, 1)?[0] = i as u64;
                }
                return Err("station residual is non-finite".into());
            }
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_sample(
    field: *const f64,
    ny: usize,
    nx: usize,
    x: f64,
    y: f64,
    method: u32,
    result: *mut f64,
) -> i32 {
    call(|| {
        let field = unsafe { input(field, size(ny, nx)?)? };
        if !x.is_finite()
            || !y.is_finite()
            || nx == 0
            || ny == 0
            || !(0.0..=(nx - 1) as f64).contains(&x)
            || !(0.0..=(ny - 1) as f64).contains(&y)
        {
            return Err("station sample lies outside the grid".into());
        }
        let value = if method == 1 {
            field[y.round_ties_even() as usize * nx + x.round_ties_even() as usize]
        } else {
            let i0 = if nx > 1 {
                (x.floor() as usize).min(nx - 2)
            } else {
                0
            };
            let j0 = if ny > 1 {
                (y.floor() as usize).min(ny - 2)
            } else {
                0
            };
            let i1 = (i0 + 1).min(nx - 1);
            let j1 = (j0 + 1).min(ny - 1);
            let tx = x - i0 as f64;
            let ty = y - j0 as f64;
            field[j0 * nx + i0] * (1.0 - tx) * (1.0 - ty)
                + field[j0 * nx + i1] * tx * (1.0 - ty)
                + field[j1 * nx + i0] * (1.0 - tx) * ty
                + field[j1 * nx + i1] * tx * ty
        };
        unsafe {
            output(result, 1)?[0] = value;
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_shared_validity(
    masks: *const u8,
    n: usize,
    count: usize,
    result: *mut u8,
) -> i32 {
    call(|| {
        let masks = unsafe { input(masks, n.checked_mul(count).ok_or("mask size overflow")?)? };
        let result = unsafe { output(result, n)? };
        for i in 0..n {
            result[i] = u8::from((0..count).all(|m| masks[m * n + i] != 0));
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_match_reports(
    reports: *const i64,
    rank: *const u64,
    report_count: usize,
    targets: *const i64,
    target_count: usize,
    tolerance: f64,
    result: *mut i64,
) -> i32 {
    call(|| {
        let reports = unsafe { input(reports, report_count)? };
        let ranks = unsafe { input(rank, report_count)? };
        let targets = unsafe { input(targets, target_count)? };
        let result = unsafe { output(result, target_count)? };
        for (target, index) in targets.iter().zip(result) {
            *index = -1;
            let mut best_offset = f64::INFINITY;
            let mut best_rank = u64::MAX;
            for i in 0..report_count {
                let offset = (reports[i] as i128 - *target as i128).abs() as f64;
                if offset > tolerance {
                    continue;
                }
                if *index < 0
                    || offset < best_offset
                    || (offset == best_offset && ranks[i] < best_rank)
                {
                    *index = i as i64;
                    best_offset = offset;
                    best_rank = ranks[i];
                }
            }
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_screen_report(
    values: *const f64,
    present: *const u8,
    result: *mut u8,
) -> i32 {
    unsafe { gpuwm_obsscore_screen_reports(values, present, 1, result) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_screen_reports(
    values: *const f64,
    present: *const u8,
    count: usize,
    result: *mut u8,
) -> i32 {
    call(|| {
        let n = count.checked_mul(3).ok_or("report count overflow")?;
        let values = unsafe { input(values, n)? };
        let present = unsafe { input(present, n)? };
        let result = unsafe { output(result, n)? };
        let bounds = [(233.15, 328.15), (233.15, 328.15), (0.0, 75.0)];
        for ((v, p), r) in values
            .chunks_exact(3)
            .zip(present.chunks_exact(3))
            .zip(result.chunks_exact_mut(3))
        {
            for i in 0..3 {
                r[i] = u8::from(p[i] != 0 && !(bounds[i].0 <= v[i] && v[i] <= bounds[i].1));
            }
            if p[0] != 0 && p[1] != 0 && v[1] > v[0] {
                r[1] = 1;
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_keeps_pairwise_order() {
        let mut values = vec![1.0; 128];
        values[0] = 1e30;
        values[8] = -1e30;
        assert_eq!(sum(&values, 8192), 126.0);
    }
    #[test]
    fn single_cell_zero_boundary_is_populated() {
        let (fraction, count) = fractions(&[1], &[1], 1, 1, 2, 0).unwrap();
        assert_eq!(fraction, [1.0]);
        assert_eq!(count, [0.04]);
    }
    #[test]
    fn no_valid_fraction_stays_zero() {
        let (fraction, count) = fractions(&[1], &[0], 1, 1, 1, 1).unwrap();
        assert_eq!(fraction, [0.0]);
        assert_eq!(count, [0.0]);
    }
}
