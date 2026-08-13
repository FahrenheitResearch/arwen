//! Coarse VAD seed costs for the radial-velocity dealiaser.
//!
//! `gpuwm.obs.dealias` starts its fold-aware harmonic fit from the best few
//! points of an exhaustive search over the physical range of environmental
//! winds: for every candidate wind `(speed, direction)` it evaluates the
//! wrapped cost
//!
//! ```text
//! cost(a, b) = sum_j 1 - cos(pi * (v_j - a cos(az_j) - b sin(az_j)) / Vn)
//! ```
//!
//! over a few hundred samples of one range band.  A 41 x 72 wind grid
//! against 384 samples is 1.13 million cosines per band and a volume carries
//! twelve hundred bands, which is where a real Level-II volume spent 98% of
//! its dealiasing time.
//!
//! Two things make this kernel fast, and only one of them is threads.
//!
//! The search grid is a *product*: the same direction is paired with every
//! speed, and the speeds are an arithmetic progression.  Write
//! `g_j = pi cos(az_j - d) / Vn` for one direction and the candidate at
//! speed `S` needs `cos(phi_j - S g_j)` with `phi_j = pi v_j / Vn`.  Angle
//! addition turns that into `cos(phi_j) cos(S g_j) + sin(phi_j) sin(S g_j)`,
//! and stepping `S` along the progression is a plane rotation of
//! `(cos(S g_j), sin(S g_j))` by the fixed angle `step * g_j`.  So the whole
//! speed axis costs one `sin_cos` pair per sample instead of one per
//! candidate: 40x fewer transcendentals, and the inner loop becomes six
//! multiply-adds that vectorise.
//!
//! # This kernel does not decide anything
//!
//! The rotation recurrence is not the arithmetic NumPy performs, so the cost
//! it returns is *not* bit-identical to the NumPy cost.  It is therefore
//! never allowed to pick the seeds.  Python treats these numbers as a
//! shortlisting device only: it takes a generous shortlist, proves from the
//! returned costs that no candidate outside the shortlist can beat the
//! shortlisted ones by more than the kernel's error bound, and then ranks
//! the shortlist with the original NumPy expression.  Where that proof fails
//! -- a degenerate band whose candidates tie -- Python falls back to the
//! full NumPy cost.  Selection is identical by construction rather than by
//! measurement, and the accuracy of this file buys speed, not correctness.
//!
//! That is also why the recurrence runs in f64.  f32 would double the SIMD
//! width of an inner loop that is no longer transcendental-bound, and would
//! widen the error bound Python must guard against from ~1e-11 to ~1e-4 --
//! trading a little arithmetic for many trips down the slow path.

use std::sync::atomic::{AtomicI32, Ordering};

use crate::{worker_ranges, ERR_DIMENSION, ERR_NONFINITE, ERR_NULL, ERR_PANIC, OK};

/// The version of the coarse-seed entry point alone.
///
/// Deliberately separate from `gpuwm_preprocess_cpu_abi_version`: the
/// interpolation ABI is quoted in preparation receipts that are audited long
/// after they are written, and adding an observation kernel is no reason to
/// invalidate them.  A library built before this entry point existed simply
/// does not export the symbol, which is exactly the signal Python needs to
/// keep using its own cost.
#[no_mangle]
pub extern "C" fn gpuwm_dealias_cost_abi_version() -> u32 {
    1
}

/// The speed axis as a first term and a step, when it is one.
fn progression(speeds: &[f64]) -> Option<f64> {
    if speeds.len() < 2 {
        return None;
    }
    let step = speeds[1] - speeds[0];
    if !step.is_finite() || step == 0.0 {
        return None;
    }
    let magnitude = speeds
        .iter()
        .fold(1.0f64, |largest, value| largest.max(value.abs()));
    for (index, &value) in speeds.iter().enumerate() {
        if (value - (speeds[0] + step * index as f64)).abs() > 1e-12 * magnitude {
            return None;
        }
    }
    Some(step)
}

/// One direction of one band: the cost of every speed, in grid order.
#[allow(clippy::too_many_arguments)]
fn direction_costs(
    cos_phi: &[f64],
    sin_phi: &[f64],
    sin_az: &[f64],
    cos_az: &[f64],
    cos_d: f64,
    sin_d: f64,
    scale: f64,
    speeds: &[f64],
    step: Option<f64>,
    scratch: &mut [f64],
    out: &mut [f64],
) {
    let samples = cos_phi.len();
    let count = samples as f64;
    let (cosine, rest) = scratch.split_at_mut(samples);
    let (sine, rest) = rest.split_at_mut(samples);
    let (turn_cos, turn_sin) = rest.split_at_mut(samples);
    match step {
        Some(delta) => {
            let first = speeds[0];
            for index in 0..samples {
                let g = scale * (cos_az[index] * cos_d + sin_az[index] * sin_d);
                let (start_sin, start_cos) = (first * g).sin_cos();
                cosine[index] = start_cos;
                sine[index] = start_sin;
                let (turn_s, turn_c) = (delta * g).sin_cos();
                turn_cos[index] = turn_c;
                turn_sin[index] = turn_s;
            }
            let last = speeds.len() - 1;
            for (speed_index, slot) in out.iter_mut().enumerate() {
                let mut total = 0.0f64;
                for index in 0..samples {
                    total += cos_phi[index] * cosine[index] + sin_phi[index] * sine[index];
                }
                *slot = count - total;
                if speed_index < last {
                    for index in 0..samples {
                        let next_cos =
                            cosine[index] * turn_cos[index] - sine[index] * turn_sin[index];
                        let next_sin =
                            sine[index] * turn_cos[index] + cosine[index] * turn_sin[index];
                        cosine[index] = next_cos;
                        sine[index] = next_sin;
                    }
                }
            }
        }
        None => {
            // No progression to ride: evaluate each candidate directly.  The
            // shipped grid is a progression, so this is the path an
            // experiment takes, not the product.
            for index in 0..samples {
                turn_cos[index] = scale * (cos_az[index] * cos_d + sin_az[index] * sin_d);
            }
            for (speed_index, slot) in out.iter_mut().enumerate() {
                let speed = speeds[speed_index];
                let mut total = 0.0f64;
                for index in 0..samples {
                    let (angle_sin, angle_cos) = (speed * turn_cos[index]).sin_cos();
                    total += cos_phi[index] * angle_cos + sin_phi[index] * angle_sin;
                }
                *slot = count - total;
            }
        }
    }
}

/// Wrapped-cost surface over a `speed x direction` wind grid, for many bands.
///
/// `values`, `sin_az` and `cos_az` carry every band's samples end to end;
/// `offsets` has `nband + 1` entries and delimits them.  `cost` receives
/// `nband * nspeed * ndirection` doubles, band-major then speed-major, which
/// is the layout of `speed[:, None] * f(direction[None, :])` raveled.
///
/// Bands are independent and every reduction stays inside one band and one
/// candidate, so `workers` cannot change a returned value.
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer implied by the
/// dimensions.  Input and output buffers must not overlap.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_dealias_coarse_cost_f64(
    values: *const f64,
    sin_az: *const f64,
    cos_az: *const f64,
    offsets: *const usize,
    nband: usize,
    speeds: *const f64,
    nspeed: usize,
    directions: *const f64,
    ndirection: usize,
    nyquist: f64,
    cost: *mut f64,
    workers: usize,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if values.is_null()
            || sin_az.is_null()
            || cos_az.is_null()
            || offsets.is_null()
            || speeds.is_null()
            || directions.is_null()
            || cost.is_null()
        {
            return ERR_NULL;
        }
        if nband == 0 || nspeed == 0 || ndirection == 0 || workers == 0 {
            return ERR_DIMENSION;
        }
        if !nyquist.is_finite() || nyquist <= 0.0 {
            return ERR_NONFINITE;
        }
        let bounds = std::slice::from_raw_parts(offsets, nband + 1);
        if bounds[0] != 0 {
            return ERR_DIMENSION;
        }
        for window in bounds.windows(2) {
            if window[1] < window[0] {
                return ERR_DIMENSION;
            }
        }
        let total = bounds[nband];
        if total == 0 {
            return ERR_DIMENSION;
        }
        let cost_length = match nband
            .checked_mul(nspeed)
            .and_then(|size| size.checked_mul(ndirection))
        {
            Some(size) => size,
            None => return ERR_DIMENSION,
        };
        let values = std::slice::from_raw_parts(values, total);
        let sin_az = std::slice::from_raw_parts(sin_az, total);
        let cos_az = std::slice::from_raw_parts(cos_az, total);
        let speeds = std::slice::from_raw_parts(speeds, nspeed);
        let directions = std::slice::from_raw_parts(directions, ndirection);
        if values.iter().any(|value| !value.is_finite())
            || sin_az.iter().any(|value| !value.is_finite())
            || cos_az.iter().any(|value| !value.is_finite())
            || speeds.iter().any(|value| !value.is_finite())
            || directions.iter().any(|value| !value.is_finite())
        {
            return ERR_NONFINITE;
        }

        let scale = std::f64::consts::PI / nyquist;
        let step = progression(speeds);
        // The sample phase, once per sample rather than once per candidate.
        let mut cos_phi = vec![0.0f64; total];
        let mut sin_phi = vec![0.0f64; total];
        for index in 0..total {
            let (phase_sin, phase_cos) = (scale * values[index]).sin_cos();
            cos_phi[index] = phase_cos;
            sin_phi[index] = phase_sin;
        }
        let cos_d: Vec<f64> = directions.iter().map(|angle| angle.cos()).collect();
        let sin_d: Vec<f64> = directions.iter().map(|angle| angle.sin()).collect();

        // One unit of work is one (band, direction) pair: enough of them for
        // any core count, and each writes a disjoint strided column.
        let units = match nband.checked_mul(ndirection) {
            Some(size) => size,
            None => return ERR_DIMENSION,
        };
        let widest = bounds
            .windows(2)
            .map(|window| window[1] - window[0])
            .max()
            .unwrap_or(0);
        let cost_address = cost as usize;
        let error = AtomicI32::new(OK);
        std::thread::scope(|scope| {
            for (start, stop) in worker_ranges(units, workers) {
                let error = &error;
                let cos_phi = &cos_phi;
                let sin_phi = &sin_phi;
                let cos_d = &cos_d;
                let sin_d = &sin_d;
                let bounds = &bounds;
                scope.spawn(move || {
                    let cost_ptr = cost_address as *mut f64;
                    let mut scratch = vec![0.0f64; 4 * widest.max(1)];
                    let mut column = vec![0.0f64; nspeed];
                    for unit in start..stop {
                        if error.load(Ordering::Relaxed) != OK {
                            break;
                        }
                        let band = unit / ndirection;
                        let direction = unit % ndirection;
                        let (first, last) = (bounds[band], bounds[band + 1]);
                        let samples = last - first;
                        if samples == 0 {
                            for speed_index in 0..nspeed {
                                unsafe {
                                    *cost_ptr.add(
                                        (band * nspeed + speed_index) * ndirection + direction,
                                    ) = 0.0;
                                }
                            }
                            continue;
                        }
                        direction_costs(
                            &cos_phi[first..last],
                            &sin_phi[first..last],
                            &sin_az[first..last],
                            &cos_az[first..last],
                            cos_d[direction],
                            sin_d[direction],
                            scale,
                            speeds,
                            step,
                            &mut scratch[..4 * samples],
                            &mut column,
                        );
                        for (speed_index, &value) in column.iter().enumerate() {
                            unsafe {
                                *cost_ptr
                                    .add((band * nspeed + speed_index) * ndirection + direction) =
                                    value;
                            }
                        }
                    }
                });
            }
        });
        let _ = cost_length;
        error.load(Ordering::Relaxed)
    }))
    .unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        values: &[f64],
        sin_az: &[f64],
        cos_az: &[f64],
        a: f64,
        b: f64,
        nyquist: f64,
    ) -> f64 {
        let mut total = 0.0;
        for index in 0..values.len() {
            let model = a * cos_az[index] + b * sin_az[index];
            total += 1.0
                - (std::f64::consts::PI * (values[index] - model) / nyquist).cos();
        }
        total
    }

    #[test]
    fn progression_is_detected_and_rejected() {
        assert_eq!(progression(&[0.0, 2.0, 4.0, 6.0]), Some(2.0));
        assert_eq!(progression(&[0.0, 2.0, 5.0]), None);
        assert_eq!(progression(&[3.0]), None);
    }

    #[test]
    fn recurrence_matches_the_direct_cost() {
        let samples = 97usize;
        let nyquist = 25.51f64;
        let mut values = Vec::new();
        let mut sin_az = Vec::new();
        let mut cos_az = Vec::new();
        for index in 0..samples {
            let azimuth = 2.0 * std::f64::consts::PI * index as f64 / samples as f64;
            sin_az.push(azimuth.sin());
            cos_az.push(azimuth.cos());
            let truth = 31.0 * azimuth.cos() - 12.0 * azimuth.sin();
            let folded = truth - 2.0 * nyquist * ((truth / (2.0 * nyquist)).round());
            values.push(folded);
        }
        let speeds: Vec<f64> = (0..41).map(|index| 2.0 * index as f64).collect();
        let directions: Vec<f64> = (0..72)
            .map(|index| (5.0 * index as f64).to_radians())
            .collect();
        let offsets = [0usize, samples];
        let mut cost = vec![0.0f64; speeds.len() * directions.len()];
        let code = unsafe {
            gpuwm_dealias_coarse_cost_f64(
                values.as_ptr(),
                sin_az.as_ptr(),
                cos_az.as_ptr(),
                offsets.as_ptr(),
                1,
                speeds.as_ptr(),
                speeds.len(),
                directions.as_ptr(),
                directions.len(),
                nyquist,
                cost.as_mut_ptr(),
                3,
            )
        };
        assert_eq!(code, OK);
        let mut worst = 0.0f64;
        for (speed_index, &speed) in speeds.iter().enumerate() {
            for (direction_index, &direction) in directions.iter().enumerate() {
                let expected = reference(
                    &values,
                    &sin_az,
                    &cos_az,
                    speed * direction.cos(),
                    speed * direction.sin(),
                    nyquist,
                );
                let seen = cost[speed_index * directions.len() + direction_index];
                worst = worst.max((seen - expected).abs());
            }
        }
        assert!(worst < 1e-9, "recurrence drifted by {worst}");
    }

    #[test]
    fn worker_count_cannot_change_a_value() {
        let samples = 64usize;
        let nyquist = 20.0f64;
        let values: Vec<f64> = (0..samples).map(|i| ((i * 7) % 31) as f64 - 15.0).collect();
        let sin_az: Vec<f64> = (0..samples)
            .map(|i| (i as f64 * 0.11).sin())
            .collect();
        let cos_az: Vec<f64> = (0..samples)
            .map(|i| (i as f64 * 0.11).cos())
            .collect();
        let speeds: Vec<f64> = (0..21).map(|index| 4.0 * index as f64).collect();
        let directions: Vec<f64> = (0..36)
            .map(|index| (10.0 * index as f64).to_radians())
            .collect();
        let offsets = [0usize, samples / 2, samples];
        let run = |workers: usize| {
            let mut cost = vec![0.0f64; 2 * speeds.len() * directions.len()];
            let code = unsafe {
                gpuwm_dealias_coarse_cost_f64(
                    values.as_ptr(),
                    sin_az.as_ptr(),
                    cos_az.as_ptr(),
                    offsets.as_ptr(),
                    2,
                    speeds.as_ptr(),
                    speeds.len(),
                    directions.as_ptr(),
                    directions.len(),
                    nyquist,
                    cost.as_mut_ptr(),
                    workers,
                )
            };
            assert_eq!(code, OK);
            cost
        };
        assert_eq!(run(1), run(7));
    }

    #[test]
    fn refuses_a_nonfinite_sample() {
        let values = [1.0f64, f64::NAN];
        let sin_az = [0.0f64, 1.0];
        let cos_az = [1.0f64, 0.0];
        let speeds = [0.0f64, 2.0];
        let directions = [0.0f64];
        let offsets = [0usize, 2];
        let mut cost = [0.0f64; 2];
        let code = unsafe {
            gpuwm_dealias_coarse_cost_f64(
                values.as_ptr(),
                sin_az.as_ptr(),
                cos_az.as_ptr(),
                offsets.as_ptr(),
                1,
                speeds.as_ptr(),
                2,
                directions.as_ptr(),
                1,
                25.0,
                cost.as_mut_ptr(),
                1,
            )
        };
        assert_eq!(code, ERR_NONFINITE);
    }
}
