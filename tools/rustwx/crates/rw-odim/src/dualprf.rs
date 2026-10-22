//! Dual-PRF velocity error correction, before any dealiasing.
//!
//! A dual-PRF sweep alternates two pulse repetition frequencies ray by ray
//! (DWD and KNMI: 800 / 600 Hz) and combines neighbouring rays to extend the
//! unambiguous interval: `NI = va_h * va_l / (va_h - va_l)` where `va_i =
//! wavelength * PRF_i / 4`. When noise makes the combination pick the wrong
//! fold count, the gate is wrong by an even multiple of ITS OWN ray's
//! single-PRF interval, `2 n va_i`. Those errors are isolated gates (salt and
//! pepper), not the smooth folding a dealiaser repairs, and a dealiaser that
//! meets them either spreads them or unfolds the region around them wrongly.
//! So they are removed first, here, on the decoder's physical values.
//!
//! Method (after Joe and May 2003, J. Atmos. Oceanic Technol. 20, 429-442,
//! and Holleman and Beekhuis 2003, ibid. 20, 443-453; the circular reference
//! follows Altube et al. 2017, ibid. 34, 1529-1543):
//!
//! 1. Reference: the measured neighbours in a window of `AZ_HALF` rays either
//!    side (the sweep is a full circle, so it wraps) and `RANGE_HALF` bins
//!    either side, centre excluded. Their circular mean on the extended
//!    interval, then the median of their deviations from it, so the
//!    reference survives the very outliers being looked for. The neighbours
//!    must also agree with each other (median absolute deviation at most
//!    `va_l / 2`): in noise there is no wind to correct towards.
//! 2. A gate whose deviation from the reference exceeds `va_l` (half the
//!    smallest possible error, `2 va_l`) is a candidate.
//! 3. Each correction `k * 2 va_i` (`k` = +-1, +-2) is tried, with `va_i`
//!    the ray's own interval when the file records per-ray PRFs
//!    (`/datasetN/how/prf`, as DWD writes it) and both intervals otherwise
//!    (ODIM has no per-ray PRF field; KNMI does not write one). The
//!    correction that brings the gate closest to the reference is applied
//!    only if it lands within `va_l / 2` of it; otherwise the gate is left as
//!    measured. A gate is never invented or removed: only its fold count
//!    changes, and the result is folded back into `[-NI, NI)`, the interval
//!    the dealiaser downstream is told about.
//! 4. Two passes, each on the previous pass's output, so a gate next to a
//!    corrected outlier is judged against a clean reference.
//!
//! The intervals come from the declared `NI` and the PRF pair alone
//! (`va_h = NI (h - l) / l`, `va_l = NI (h - l) / h`), so the correction
//! needs no wavelength and agrees with the interval the writer stated.

use crate::censor;
use crate::quantity::QuantityKind;
use crate::volume::{Moment, NyquistSource, Sweep};

/// Rays either side of the gate in the reference window.
pub const AZ_HALF: usize = 2;
/// Bins either side of the gate in the reference window.
pub const RANGE_HALF: usize = 2;
/// Fewest measured neighbours (of 24) a reference may be built from.
pub const MIN_NEIGHBOURS: usize = 8;
/// Passes over the sweep.
pub const PASSES: usize = 2;
/// The neighbours' median absolute deviation (about their median) must be at
/// most this fraction of `va_l` for a reference to exist: half the smallest
/// error step, `2 va_l`, is `va_l`; half of that keeps uniform noise
/// (MAD about NI / 2) far outside and a sheared but coherent wind inside.
pub const MAX_NEIGHBOUR_MAD_FRACTION: f64 = 0.5;
/// A correction is applied only when it lands within this fraction of `va_l`
/// of the reference.
pub const ACCEPT_FRACTION: f64 = 0.5;

/// What the correction did, per volume.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DualPrfReport {
    /// Velocity moments on dual-PRF sweeps that were examined.
    pub moments_examined: usize,
    /// Of those, moments whose per-ray PRFs were read from the file.
    pub moments_with_ray_prf: usize,
    /// Velocity moments on dual-PRF sweeps that could not be examined
    /// (no declared NI or no PRF pair), left as measured.
    pub moments_skipped: usize,
    pub gates_measured: usize,
    pub gates_corrected: usize,
}

impl DualPrfReport {
    pub fn add(&mut self, other: &DualPrfReport) {
        self.moments_examined += other.moments_examined;
        self.moments_with_ray_prf += other.moments_with_ray_prf;
        self.moments_skipped += other.moments_skipped;
        self.gates_measured += other.gates_measured;
        self.gates_corrected += other.gates_corrected;
    }
}

/// The two single-PRF intervals behind a declared extended interval.
pub fn component_intervals(ni: f64, high_prf: f64, low_prf: f64) -> Option<(f64, f64)> {
    if !(ni.is_finite() && ni > 0.0 && high_prf > low_prf && low_prf > 0.0) {
        return None;
    }
    let diff = high_prf - low_prf;
    Some((ni * diff / low_prf, ni * diff / high_prf))
}

/// Fold `v` into `[-ni, ni)`.
fn fold(v: f64, ni: f64) -> f64 {
    let period = 2.0 * ni;
    let mut x = (v + ni).rem_euclid(period) - ni;
    if x >= ni {
        x -= period;
    }
    x
}

/// Correct dual-PRF outliers in one velocity plane in place.
///
/// `values` is row-major `[ray][bin]`, `codes` its censor plane; only gates
/// coded measured are read or changed. `ray_prf` is the per-ray PRF in Hz
/// when the file records it. Returns the number of gates changed.
#[allow(clippy::too_many_arguments)]
pub fn correct_plane(
    values: &mut [f64],
    codes: &[u8],
    nrays: usize,
    nbins: usize,
    ni: f64,
    high_prf: f64,
    low_prf: f64,
    ray_prf: Option<&[f64]>,
) -> usize {
    let Some((va_h, va_l)) = component_intervals(ni, high_prf, low_prf) else {
        return 0;
    };
    if nrays == 0 || nbins == 0 || values.len() != nrays * nbins || codes.len() != values.len() {
        return 0;
    }
    let ray_prf = ray_prf.filter(|p| p.len() == nrays);
    let scale = ni * (high_prf - low_prf) / (high_prf * low_prf);
    let steps_for = |ray: usize| -> Vec<f64> {
        match ray_prf {
            Some(p) if p[ray].is_finite() && p[ray] > 0.0 => vec![2.0 * scale * p[ray]],
            _ => vec![2.0 * va_h, 2.0 * va_l],
        }
    };
    let mut changed = vec![false; values.len()];
    let mut neighbours: Vec<f64> = Vec::with_capacity((2 * AZ_HALF + 1) * (2 * RANGE_HALF + 1));
    for _ in 0..PASSES {
        let snapshot = values.to_vec();
        let mut pass_changed = 0usize;
        for ray in 0..nrays {
            let steps = steps_for(ray);
            for bin in 0..nbins {
                let i = ray * nbins + bin;
                if codes[i] != censor::MEASURED || !snapshot[i].is_finite() {
                    continue;
                }
                neighbours.clear();
                let (mut s, mut c) = (0.0f64, 0.0f64);
                for da in 0..=(2 * AZ_HALF) {
                    let r = (ray + nrays + da - AZ_HALF) % nrays;
                    let lo = bin.saturating_sub(RANGE_HALF);
                    let hi = (bin + RANGE_HALF).min(nbins - 1);
                    for b in lo..=hi {
                        let j = r * nbins + b;
                        if j == i || codes[j] != censor::MEASURED || !snapshot[j].is_finite() {
                            continue;
                        }
                        let phase = std::f64::consts::PI * snapshot[j] / ni;
                        s += phase.sin();
                        c += phase.cos();
                        neighbours.push(snapshot[j]);
                    }
                }
                if neighbours.len() < MIN_NEIGHBOURS || (s == 0.0 && c == 0.0) {
                    continue;
                }
                let mean = s.atan2(c) * ni / std::f64::consts::PI;
                for v in neighbours.iter_mut() {
                    *v = fold(*v - mean, ni);
                }
                neighbours.sort_by(|a, b| a.total_cmp(b));
                let median = neighbours[neighbours.len() / 2];
                // Coherence: the neighbours must agree with one another. In
                // noise (clear air, weak echo) they spread over the whole
                // interval and there is no wind to correct towards; a
                // "correction" there would turn noise into a plausible wind.
                let mut spread: Vec<f64> = neighbours.iter().map(|d| (d - median).abs()).collect();
                spread.sort_by(|a, b| a.total_cmp(b));
                if spread[spread.len() / 2] > MAX_NEIGHBOUR_MAD_FRACTION * va_l {
                    continue;
                }
                let reference = mean + median;
                let v = snapshot[i];
                let deviation = fold(v - reference, ni);
                if deviation.abs() <= va_l {
                    continue;
                }
                let mut best = (deviation.abs(), 0.0f64);
                for &step in &steps {
                    for k in [-2.0, -1.0, 1.0, 2.0] {
                        let trial = fold(deviation + k * step, ni).abs();
                        if trial < best.0 {
                            best = (trial, k * step);
                        }
                    }
                }
                if best.1 != 0.0 && best.0 <= ACCEPT_FRACTION * va_l {
                    values[i] = fold(v + best.1, ni);
                    if !changed[i] {
                        changed[i] = true;
                        pass_changed += 1;
                    }
                }
            }
        }
        if pass_changed == 0 {
            break;
        }
    }
    changed.iter().filter(|c| **c).count()
}

/// The corrected copy of one velocity moment of `sweep`, or `None` when the
/// moment is not a velocity on a dual-PRF sweep, or cannot be examined (no
/// declared NI or no PRF pair: counted as skipped, never guessed).
pub fn corrected_values(sweep: &Sweep, moment: &Moment) -> (Option<Vec<f64>>, DualPrfReport) {
    let mut report = DualPrfReport::default();
    if !sweep.nyquist.dual_prf || moment.kind != QuantityKind::RadialVelocity {
        return (None, report);
    }
    let pair = match (sweep.nyquist.high_prf_hz, sweep.nyquist.low_prf_hz) {
        (Some(a), Some(b)) => Some((a.max(b), a.min(b))),
        _ => None,
    };
    let ni = match sweep.nyquist.source {
        NyquistSource::Declared => sweep.nyquist.interval_ms,
        _ => None,
    };
    let (Some(ni), Some((high, low))) = (ni, pair) else {
        report.moments_skipped += 1;
        return (None, report);
    };
    report.moments_examined += 1;
    let ray_prf = sweep
        .ray_prf_hz
        .as_deref()
        .filter(|p| p.len() == moment.nrays);
    if ray_prf.is_some() {
        report.moments_with_ray_prf += 1;
    }
    report.gates_measured = moment
        .censor
        .iter()
        .filter(|c| **c == censor::MEASURED)
        .count();
    let mut values = moment.values.clone();
    report.gates_corrected = correct_plane(
        &mut values,
        &moment.censor,
        moment.nrays,
        moment.nbins,
        ni,
        high,
        low,
        ray_prf,
    );
    (Some(values), report)
}

/// Correct every velocity moment of one sweep in place.
pub fn correct_sweep(sweep: &mut Sweep) -> DualPrfReport {
    let mut report = DualPrfReport::default();
    let mut planes = Vec::with_capacity(sweep.moments.len());
    for moment in &sweep.moments {
        let (values, one) = corrected_values(sweep, moment);
        report.add(&one);
        planes.push(values);
    }
    for (moment, values) in sweep.moments.iter_mut().zip(planes) {
        if let Some(values) = values {
            moment.values = values;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    const NI: f64 = 32.0823;
    const HIGH: f64 = 800.0;
    const LOW: f64 = 600.0;

    /// A smooth wind field (a uniform 12 m/s flow seen at each azimuth).
    fn field(nrays: usize, nbins: usize) -> Vec<f64> {
        (0..nrays * nbins)
            .map(|i| {
                let ray = i / nbins;
                12.0 * (2.0 * std::f64::consts::PI * ray as f64 / nrays as f64).cos()
            })
            .collect()
    }

    #[test]
    fn the_component_intervals_reproduce_the_declared_extended_interval() {
        let (h, l) = component_intervals(NI, HIGH, LOW).unwrap();
        // DWD Essen 2026-10-16: wavelength 5.34706 cm, so va = 10.69 and 8.02.
        assert!(
            (h - 10.694).abs() < 0.01 && (l - 8.0206).abs() < 0.01,
            "{h} {l}"
        );
        assert!((h * l / (h - l) - NI).abs() < 1e-9);
    }

    #[test]
    fn isolated_dual_prf_errors_are_removed_and_the_field_is_otherwise_untouched() {
        let (nrays, nbins) = (360, 60);
        let truth = field(nrays, nbins);
        let mut values = truth.clone();
        let codes = vec![censor::MEASURED; values.len()];
        let (va_h, va_l) = component_intervals(NI, HIGH, LOW).unwrap();
        // Ten outliers of each kind, on rays of the matching PRF.
        let mut planted = Vec::new();
        for n in 0..10 {
            let high_ray = 20 * n; // even rays: 800 Hz
            let low_ray = 20 * n + 101; // odd rays: 600 Hz
            let i = high_ray * nbins + 10 + n;
            values[i] = fold(values[i] + 2.0 * va_h, NI);
            let j = low_ray * nbins + 30 + n;
            values[j] = fold(values[j] - 2.0 * va_l, NI);
            planted.extend([i, j]);
        }
        let prf: Vec<f64> = (0..nrays)
            .map(|r| if r % 2 == 0 { HIGH } else { LOW })
            .collect();
        for ray_prf in [Some(prf.as_slice()), None] {
            let mut v = values.clone();
            let changed = correct_plane(&mut v, &codes, nrays, nbins, NI, HIGH, LOW, ray_prf);
            assert_eq!(
                changed,
                planted.len(),
                "ray_prf known: {}",
                ray_prf.is_some()
            );
            for (a, b) in v.iter().zip(&truth) {
                assert!((a - b).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn censored_gates_are_neither_read_nor_written() {
        let (nrays, nbins) = (360, 20);
        let mut values = field(nrays, nbins);
        let mut codes = vec![censor::MEASURED; values.len()];
        values[5 * nbins + 5] = f64::NAN;
        codes[5 * nbins + 5] = censor::NODATA;
        let before = values.clone();
        assert_eq!(
            correct_plane(&mut values, &codes, nrays, nbins, NI, HIGH, LOW, None),
            0
        );
        assert!(values[5 * nbins + 5].is_nan());
        assert_eq!(values[10], before[10]);
    }

    #[test]
    fn a_single_prf_sweep_is_never_touched() {
        let mut values = vec![1.0, 30.0, 1.0];
        let codes = vec![censor::MEASURED; 3];
        assert_eq!(
            correct_plane(&mut values, &codes, 1, 3, NI, 800.0, 800.0, None),
            0
        );
        assert_eq!(values, vec![1.0, 30.0, 1.0]);
    }
}
