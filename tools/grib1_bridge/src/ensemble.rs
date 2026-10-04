//! Source-member anomalies with a fixed population and strict binary64 order.
use crate::{worker_ranges, ERR_DIMENSION, ERR_NONFINITE, ERR_NULL, ERR_PANIC, OK};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[no_mangle]
pub extern "C" fn gpuwm_ensemble_preparation_abi_version() -> u32 { 3 }

fn inward(value: f64, lower: bool) -> f32 {
    let rounded = value as f32;
    let adjust = if lower { (rounded as f64) < value } else { (rounded as f64) > value };
    if !adjust { return rounded; }
    if rounded == 0.0 {
        return f32::from_bits(if lower { 1 } else { 0x8000_0001 });
    }
    let bits = rounded.to_bits();
    f32::from_bits(if lower == rounded.is_sign_positive() { bits + 1 } else { bits - 1 })
}

fn cell(base: f32, donors: &[f32], index: usize, cells: usize, population: usize,
        selected: usize, amplitude: f64, bound: f64, lower: f64, upper: f64) -> f32 {
    if amplitude == 0.0 || bound == 0.0 || (base as f64) < lower || (base as f64) > upper { return base; }
    let mut mean = 0.0;
    for donor in 0..population { mean += donors[donor*cells + index] as f64; }
    mean /= population as f64;
    let low = inward(lower.max(base as f64 - bound), true) as f64;
    let high = inward(upper.min(base as f64 + bound), false) as f64;
    let mut scale = amplitude;
    for donor in 0..population {
        let delta = donors[donor*cells + index] as f64 - mean;
        if delta > 0.0 { scale = scale.min((high - base as f64) / delta); }
        if delta < 0.0 { scale = scale.min((low - base as f64) / delta); }
    }
    let anomaly = donors[selected*cells + index] as f64 - mean;
    if scale == 0.0 || anomaly == 0.0 { return base; }
    (base as f64 + scale * anomaly) as f32
}

/// Same operation and population order as recentered_physical in CUDA.
/// Buffers must be non-overlapping and contain the declared element counts.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_ensemble_recenter_f32(
    base: *const f32, donors: *const f32, selected: *const i32, out: *mut f32,
    cells: usize, population: usize, members: usize, amplitude: f64, bound: f64,
    lower: f64, upper: f64, input_lower: f64, input_upper: f64, workers: usize,
) -> i32 {
    if base.is_null() || donors.is_null() || selected.is_null() || out.is_null() { return ERR_NULL; }
    if cells == 0 || population < 2 || members == 0 || workers == 0
        || cells.checked_mul(population).is_none() || cells.checked_mul(members).is_none()
        || amplitude < 0.0 || bound < 0.0 || lower >= upper
        || input_lower > lower || input_upper < upper
        || ![amplitude, bound, lower, upper, input_lower, input_upper].iter().all(|x| x.is_finite()) {
        return ERR_DIMENSION;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let base = std::slice::from_raw_parts(base, cells);
        let donors = std::slice::from_raw_parts(donors, cells*population);
        let selected = std::slice::from_raw_parts(selected, members);
        if selected.iter().any(|&i| i < 0 || i as usize >= population) { return ERR_DIMENSION; }
        if base.iter().any(|&x| !x.is_finite() || (x as f64) < input_lower || (x as f64) > input_upper)
            || donors.iter().any(|x| !x.is_finite()) { return ERR_NONFINITE; }
        let output = std::slice::from_raw_parts_mut(out, cells*members);
        std::thread::scope(|scope| {
            let mut remaining = output;
            for (start, stop) in worker_ranges(cells*members, workers) {
                let (chunk, rest) = remaining.split_at_mut(stop-start);
                remaining = rest;
                scope.spawn(move || {
                    for (offset, target) in chunk.iter_mut().enumerate() {
                        let position = start+offset;
                        let index = position % cells;
                        *target = cell(base[index], donors, index, cells, population,
                            selected[position / cells] as usize, amplitude, bound, lower, upper);
                    }
                });
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

/// Native pressure-level coordinate expansion. Levels retain source order.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_ensemble_pressure_levels_f32(
    levels_hpa: *const f64, out: *mut f32, nz: usize, plane: usize,
) -> i32 {
    if levels_hpa.is_null() || out.is_null() { return ERR_NULL; }
    let Some(cells) = nz.checked_mul(plane) else { return ERR_DIMENSION; };
    if cells == 0 { return ERR_DIMENSION; }
    catch_unwind(AssertUnwindSafe(|| {
        let levels = std::slice::from_raw_parts(levels_hpa, nz);
        let output = std::slice::from_raw_parts_mut(out, cells);
        for (&level, target) in levels.iter().zip(output.chunks_exact_mut(plane)) {
            let pressure = level * 100.0;
            if !pressure.is_finite() || pressure <= 0.0 || pressure > f32::MAX as f64 { return ERR_NONFINITE; }
            target.fill(pressure as f32);
        }
        OK
    })).unwrap_or(ERR_PANIC)
}

fn humidity_value(t: f64, p: f64, value: f64, to_rh: bool,
                  svp1: f64, svp2: f64, svpt0: f64, svp3: f64) -> f64 {
    // WRF rh_to_mxrat1 uses local EPS=0.622. Keep the ordinary native
    // real initializer's Bolton operation order and its RH clipping/floor.
    let exponential = libm::exp(svp2 * (t - svpt0) / (t - svp3));
    if to_rh {
        let saturation = (10.0 * svp1) * exponential;
        let mixing_ratio = value / (1.0 - value);
        let vapor = mixing_ratio * (p / 100.0) / (mixing_ratio + 0.622);
        100.0 * vapor / saturation
    } else {
        let vapor = ((value.clamp(0.0, 100.0) * 0.01) * (10.0 * svp1)) * exponential;
        let mixing_ratio = if t != 0.0 && vapor.is_finite() && vapor < p / 100.0 {
            (0.622 * vapor / (p / 100.0 - vapor)).max(1.0e-6)
        } else { 1.0e-6 };
        mixing_ratio / (1.0 + mixing_ratio)
    }
}

/// Humidity representation conversion for donor alignment only.
/// Base fields and their native initializer route are retained unchanged.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_ensemble_humidity_f32(
    temperature: *const f32, pressure: *const f32, input: *const f32,
    out: *mut f32, cells: usize, mode: i32, minimum_specific: f64,
    svp1: f64, svp2: f64, svpt0: f64, svp3: f64,
) -> i32 {
    if temperature.is_null() || pressure.is_null() || input.is_null() || out.is_null() { return ERR_NULL; }
    if cells == 0 || (mode != 1 && mode != 2)
        || ![minimum_specific, svp1, svp2, svpt0, svp3].iter().all(|x| x.is_finite()) {
        return ERR_DIMENSION;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let temperature = std::slice::from_raw_parts(temperature, cells);
        let pressure = std::slice::from_raw_parts(pressure, cells);
        let input = std::slice::from_raw_parts(input, cells);
        let output = std::slice::from_raw_parts_mut(out, cells);
        for (((&t, &p), &value), target) in temperature.iter().zip(pressure).zip(input).zip(output) {
            if !t.is_finite() || !p.is_finite() || !value.is_finite() || p <= 0.0
                || (mode == 1 && ((value as f64) < minimum_specific || value >= 1.0)) { return ERR_NONFINITE; }
            let converted = humidity_value(t as f64, p as f64, value as f64, mode == 1, svp1, svp2, svpt0, svp3);
            *target = converted as f32;
            if !target.is_finite() { return ERR_NONFINITE; }
        }
        OK
    })).unwrap_or(ERR_PANIC)
}

/// Temporal interpolation at fixed physical coordinates, one final rounding.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_ensemble_time_blend_f32(
    left: *const f32, right: *const f32, out: *mut f32, cells: usize, weight: f64,
) -> i32 {
    if left.is_null() || right.is_null() || out.is_null() { return ERR_NULL; }
    if cells == 0 || !weight.is_finite() || !(0.0..=1.0).contains(&weight) { return ERR_DIMENSION; }
    catch_unwind(AssertUnwindSafe(|| {
        let left = std::slice::from_raw_parts(left, cells);
        let right = std::slice::from_raw_parts(right, cells);
        let output = std::slice::from_raw_parts_mut(out, cells);
        for ((&a, &b), target) in left.iter().zip(right).zip(output) {
            if !a.is_finite() || !b.is_finite() { return ERR_NONFINITE; }
            *target = if weight == 0.0 { a } else if weight == 1.0 { b }
                      else { ((a as f64)*(1.0-weight)+(b as f64)*weight) as f32 };
        }
        OK
    })).unwrap_or(ERR_PANIC)
}

/// WRF pressure staggering for a layer-major pressure coordinate.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_ensemble_pressure_stagger_f32(
    input: *const f32, out: *mut f32, nz: usize, ny: usize, nx: usize, axis: i32,
) -> i32 {
    if input.is_null() || out.is_null() { return ERR_NULL; }
    if nz == 0 || ny == 0 || nx == 0 || (axis != 1 && axis != 2) { return ERR_DIMENSION; }
    let Some(cells) = nz.checked_mul(ny).and_then(|n| n.checked_mul(nx)) else { return ERR_DIMENSION; };
    let Some(oy) = ny.checked_add(usize::from(axis == 1)) else { return ERR_DIMENSION; };
    let Some(ox) = nx.checked_add(usize::from(axis == 2)) else { return ERR_DIMENSION; };
    let Some(outputs) = nz.checked_mul(oy).and_then(|n| n.checked_mul(ox)) else { return ERR_DIMENSION; };
    catch_unwind(AssertUnwindSafe(|| {
        let input = std::slice::from_raw_parts(input, cells);
        let output = std::slice::from_raw_parts_mut(out, outputs);
        for k in 0..nz { for j in 0..oy { for i in 0..ox {
            let (a, b) = if axis == 1 {
                ((k*ny+j.saturating_sub(1))*nx+i, (k*ny+j.min(ny-1))*nx+i)
            } else {
                ((k*ny+j)*nx+i.saturating_sub(1), (k*ny+j)*nx+i.min(nx-1))
            };
            let (left, right) = (input[a], input[b]);
            if !left.is_finite() || !right.is_finite() || left <= 0.0 || right <= 0.0 { return ERR_NONFINITE; }
            output[(k*oy+j)*ox+i] = if a == b { left } else { ((left as f64 + right as f64)*0.5) as f32 };
        }}}
        OK
    })).unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_humidity_keeps_native_portable_exp_and_association() {
        let first = humidity_value(f32::from_bits(0x4339_0000) as f64,
            f32::from_bits(0x461c_4000) as f64, f32::from_bits(0x42ab_6db7) as f64,
            false, 0.6112, 17.67, 273.15, 29.65);
        assert_eq!(first.to_bits(), 0x3eb8_2c02_ed05_0cea);
        let second = humidity_value(190.0, 10000.0, f32::from_bits(0x41de_db6e) as f64,
            false, 0.6112, 17.67, 273.15, 29.65);
        assert_eq!(second.to_bits(), 0x3eb2_a1d2_0be1_7f2f);
    }
    #[test]
    fn off_preserves_signed_zero_and_population_bounds() {
        let donors = [-9.0, 0.0, 8.0];
        assert_eq!(cell(-0.0, &donors, 0, 1, 3, 0, 0.0, 1.0, -2.0, 2.0).to_bits(), (-0.0f32).to_bits());
        for member in 0..3 {
            let result = cell(1.0, &donors, 0, 1, 3, member, 2.0, 0.1, -2.0, 2.0);
            assert!((result as f64 - 1.0).abs() <= 0.1);
        }
    }
    #[test]
    fn inward_endpoints_are_within_the_real_interval() {
        for value in [-100.1, -0.1, 0.0, 0.1, 100.1] {
            assert!((inward(value, true) as f64) >= value);
            assert!((inward(value, false) as f64) <= value);
        }
    }
    #[test]
    fn moisture_anomalies_do_not_spend_an_existing_undershoot_allowance() {
        let donors = [0.0, 0.006];
        for member in 0..2 {
            assert!(cell(0.0001, &donors, 0, 1, 2, member, 1.0, 0.005, 0.0, 0.1) >= 0.0);
            assert_eq!(cell(-0.0001, &donors, 0, 1, 2, member, 1.0, 0.005, 0.0, 0.1), -0.0001);
        }
    }
}
