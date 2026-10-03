//! Field values to Message 31 gate codes (`docs/level2/writer.md`,
//! "Quantisation").
//!
//! Level II gates are unsigned 8- or 16-bit codes with `value = (code -
//! offset) / scale`, code 0 below threshold and code 1 range folded
//! (Table XVII-B). NEXRAD-coded fields are copied as they are. Every other
//! field mapped to a moment shares one coding with the moment's fields in
//! the other sweeps: Py-ART decodes a moment of every sweep with the scale
//! and offset of the first sweep that has it.

use std::collections::HashMap;

use recast_radar_core::model::{
    Field, FieldData, Gate, IntCoding, LinearTransform, PackedInt, RowRef,
};

use super::Quantization;
use super::plan::Moment;

/// Code of a gate below threshold (also missing and undetected gates).
const BELOW_THRESHOLD: u16 = 0;
/// Code of a range-folded gate.
const RANGE_FOLDED: u16 = 1;
/// Lowest code of a value.
const FIRST_VALUE_CODE: u16 = 2;
/// Most distinct values tracked before a field counts as continuous.
const MAX_DISTINCT: usize = 65_536;
/// Largest distance of a value from an integer code, in code units, for the
/// value to count as lying on a coding's grid.
const GRID_TOLERANCE: f64 = 0.05;
/// Largest ZDR code current NEXRAD files use and xradar 0.12 reads (it keeps
/// the low 11 bits of 16-bit ZDR words).
const ZDR_MAX_CODE: u16 = 0x7FF;
/// Largest PHI code current NEXRAD files use and xradar 0.12 reads (the low
/// 10 bits of 16-bit PHI words).
const PHI_MAX_CODE: u16 = 0x3FF;

/// A Message 31 moment coding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Coding {
    /// Bits per gate: 8 or 16.
    pub word_size: u8,
    /// `value = (code - offset) / scale`.
    pub scale: f32,
    pub offset: f32,
    /// Largest code written.
    pub max_code: u16,
}

impl Coding {
    const fn new(word_size: u8, scale: f32, offset: f32, max_code: u16) -> Self {
        Self {
            word_size,
            scale,
            offset,
            max_code,
        }
    }

    /// The code of a finite physical value, and whether the coding loses it
    /// (the value would be clipped): it lies more than half a step outside
    /// the value codes, or the coding cannot scale it (a zero or non-finite
    /// scale or offset, as a Level II moment with floating-point gates has).
    /// A value just outside the end codes by float rounding of the coding is
    /// not lost: its end code decodes to it within half a step.
    fn code(self, value: f32) -> (u16, bool) {
        let scale = f64::from(self.scale);
        let usable = scale.is_finite() && scale != 0.0 && self.offset.is_finite();
        let scaled = f64::from(value) * scale + f64::from(self.offset);
        if !usable || !scaled.is_finite() {
            return (BELOW_THRESHOLD, true);
        }
        let rounded = scaled.round();
        let code = rounded.clamp(f64::from(FIRST_VALUE_CODE), f64::from(self.max_code));
        let code = code as u16;
        if rounded == f64::from(code) {
            return (code, false);
        }
        let error = (f64::from(self.decode(code)) - f64::from(value)).abs();
        let tolerance = 0.5 / scale.abs() + f64::from(value).abs() * f64::from(f32::EPSILON);
        // A NaN error (not reached with a usable coding) counts as lost.
        let held = error <= tolerance;
        (code, !held)
    }

    /// The value a decoder reads for `code` (evaluated in f32, as the ICD
    /// form is).
    fn decode(self, code: u16) -> f32 {
        (f32::from(code) - self.offset) / self.scale
    }

    /// The smallest and largest values the coding holds (its first and last
    /// value codes decoded).
    pub(crate) fn value_range(self) -> (f32, f32) {
        let (a, b) = (self.decode(FIRST_VALUE_CODE), self.decode(self.max_code));
        (a.min(b), a.max(b))
    }

    fn gate_code(self, gate: Gate) -> (u16, bool) {
        match gate {
            Gate::Value(value) if value.is_finite() => self.code(value),
            Gate::RangeFolded => (RANGE_FOLDED, false),
            _ => (BELOW_THRESHOLD, false),
        }
    }
}

/// The ICD's typical codings of a moment (Table XVII-I and the corpus), the
/// preferred one first, with the code ranges NEXRAD files use. The first is
/// what NOAA's current files carry (KTLX 2024, KILX 2026: ZDR 16-bit at 32
/// and 418, RHO at 300 and -60.5); the second ZDR coding is the 8-bit one of
/// earlier builds.
pub(crate) fn standard_codings(moment: Moment) -> &'static [Coding] {
    const REF: [Coding; 1] = [Coding::new(8, 2.0, 66.0, 255)];
    const VEL: [Coding; 2] = [
        Coding::new(8, 2.0, 129.0, 255),
        Coding::new(8, 1.0, 129.0, 255),
    ];
    const SW: [Coding; 1] = [Coding::new(8, 2.0, 129.0, 255)];
    const ZDR: [Coding; 2] = [
        Coding::new(16, 32.0, 418.0, ZDR_MAX_CODE),
        Coding::new(8, 16.0, 128.0, 255),
    ];
    const PHI: [Coding; 1] = [Coding::new(16, 2.8361, 2.0, PHI_MAX_CODE)];
    const RHO: [Coding; 1] = [Coding::new(8, 300.0, -60.5, 255)];
    const CFP: [Coding; 1] = [Coding::new(8, 1.0, 8.0, 255)];
    match moment {
        Moment::Ref => &REF,
        Moment::Vel => &VEL,
        Moment::Sw => &SW,
        Moment::Zdr => &ZDR,
        Moment::Phi => &PHI,
        Moment::Rho => &RHO,
        Moment::Cfp => &CFP,
    }
}

/// Largest 16-bit code of `moment` under `policy`, or `None` when the
/// moment is written in 8 bits only.
fn sixteen_bit_limit(moment: Moment, policy: Quantization) -> Option<u16> {
    match (policy, moment) {
        (Quantization::Precise, _) => Some(u16::MAX),
        (_, Moment::Zdr) => Some(ZDR_MAX_CODE),
        (_, Moment::Phi) => Some(PHI_MAX_CODE),
        _ => None,
    }
}

/// How the gates of one field are turned into codes.
#[derive(Clone, Debug)]
pub(crate) enum GateEncoder {
    /// Copy the stored NEXRAD codes (`u8` or `u16` storage).
    Raw,
    /// Output code per stored code, indexed by the stored value's bits
    /// (`u8`/`i8`: 256 entries, `u16`/`i16`: 65536).
    Table(Vec<u16>),
    /// Resolve each gate and code its value (`i32`, `f32`, `f64` storage).
    Compute,
}

/// The coding chosen for one field and what it does to the values.
#[derive(Clone, Debug)]
pub(crate) struct FieldEncoding {
    pub coding: Coding,
    pub encoder: GateEncoder,
    pub exact: bool,
    pub max_abs_error: f32,
    /// Gates whose value the coding cannot hold ([`Coding::code`]); the
    /// planner refuses a volume with any.
    pub clipped_gates: usize,
}

/// Distinct values of a field (every provided row), with their gate counts;
/// `None` when there are more than [`MAX_DISTINCT`].
#[derive(Clone, Debug)]
struct ValueSummary {
    distinct: Option<Vec<(f32, u64)>>,
    min: f32,
    max: f32,
    any: bool,
    /// Value step of integer storage (the transform's scale), which fixes
    /// the grid exactly; `None` for float storage.
    step: Option<f64>,
}

/// Choose the codings of the fields mapped to `moment`, one per field, in
/// order. NEXRAD-coded fields keep their codes; the others share one coding.
pub(crate) fn choose(
    fields: &[&Field],
    moment: Moment,
    policy: Quantization,
) -> Vec<FieldEncoding> {
    let summaries: Vec<Option<ValueSummary>> = fields
        .iter()
        .map(|field| nexrad_coding(field).is_none().then(|| summarize(field)))
        .collect();
    let joint = merge(summaries.iter().flatten());
    let coding = joint.as_ref().map(|joint| match policy {
        Quantization::Standard => {
            // The typical coding only where it holds every value: a fixed
            // coding must not clip the source.
            let typical = standard_codings(moment)[0];
            if covers(joint, typical) {
                typical
            } else {
                chosen_coding(joint, moment, Quantization::Compatible)
            }
        }
        _ => chosen_coding(joint, moment, policy),
    });
    fields
        .iter()
        .zip(&summaries)
        .map(|(field, summary)| match (summary, coding) {
            (Some(summary), Some(coding)) => {
                let (max_abs_error, clipped_gates) = measure(field, summary, coding);
                FieldEncoding {
                    coding,
                    encoder: encoder(field, coding),
                    exact: clipped_gates == 0 && is_exact(summary, coding, max_abs_error),
                    max_abs_error,
                    clipped_gates,
                }
            }
            _ => FieldEncoding {
                coding: nexrad_coding(field).unwrap_or(standard_codings(moment)[0]),
                encoder: GateEncoder::Raw,
                exact: true,
                max_abs_error: 0.0,
                clipped_gates: 0,
            },
        })
        .collect()
}

/// The encodings of the fields mapped to a moment under a coding fixed
/// beforehand (the real-time writer's, chosen from the planned volume):
/// NEXRAD-coded fields keep their codes, the others take `coding`, their
/// values outside it counted in `clipped_gates` (the planner then refuses
/// the volume).
pub(crate) fn with_coding(fields: &[&Field], coding: Coding) -> Vec<FieldEncoding> {
    fields
        .iter()
        .map(|field| match nexrad_coding(field) {
            Some(own) => FieldEncoding {
                coding: own,
                encoder: GateEncoder::Raw,
                exact: true,
                max_abs_error: 0.0,
                clipped_gates: 0,
            },
            None => {
                let summary = summarize(field);
                let (max_abs_error, clipped_gates) = measure(field, &summary, coding);
                FieldEncoding {
                    coding,
                    encoder: encoder(field, coding),
                    exact: clipped_gates == 0 && is_exact(&summary, coding, max_abs_error),
                    max_abs_error,
                    clipped_gates,
                }
            }
        })
        .collect()
}

/// The field's own coding when it is a NEXRAD coding (its codes can be
/// copied unchanged). Any scale and offset qualify, 0 and NaN included: a
/// Level II moment's codes are its data, and its scale and offset are
/// written back bit for bit (a zero scale means floating-point gates, ICD
/// note 15, which the codes carry as they are).
fn nexrad_coding(field: &Field) -> Option<Coding> {
    fn check<T: PackedInt>(coding: &IntCoding<T>, word_size: u8, max: u16) -> Option<Coding> {
        let LinearTransform::IcdScaleOffset { scale, offset } = coding.transform else {
            return None;
        };
        // The sentinels and valid range of a NEXRAD coding; the transform
        // is compared above (`==` on it is false for a NaN scale).
        let template = IntCoding::<T>::nexrad(1.0, 0.0);
        let own = IntCoding {
            transform: template.transform,
            ..*coding
        };
        (own == template).then_some(Coding::new(word_size, scale, offset, max))
    }
    match &field.data {
        FieldData::U8 { coding, .. } => check(coding, 8, 255),
        FieldData::U16 { coding, .. } => check(coding, 16, u16::MAX),
        _ => None,
    }
}

/// The summaries of several fields as one; `None` when there are none.
fn merge<'a>(summaries: impl Iterator<Item = &'a ValueSummary>) -> Option<ValueSummary> {
    let mut joint: Option<ValueSummary> = None;
    let mut counts: HashMap<u32, (f32, u64)> = HashMap::new();
    let mut continuous = false;
    for summary in summaries {
        match &summary.distinct {
            Some(distinct) if !continuous => {
                for (value, count) in distinct {
                    counts.entry(value.to_bits()).or_insert((*value, 0)).1 += count;
                }
                if counts.len() > MAX_DISTINCT {
                    continuous = true;
                }
            }
            _ => continuous = true,
        }
        joint = Some(match joint {
            None => summary.clone(),
            Some(mut joint) => {
                joint.min = joint.min.min(summary.min);
                joint.max = joint.max.max(summary.max);
                joint.any |= summary.any;
                let same_step = match (joint.step, summary.step) {
                    (Some(a), Some(b)) => (a - b).abs() <= 1e-9 * a.abs(),
                    _ => false,
                };
                if !same_step {
                    joint.step = None;
                }
                joint
            }
        });
    }
    let mut joint = joint?;
    joint.distinct = if continuous {
        None
    } else {
        let mut distinct: Vec<(f32, u64)> = counts.into_values().collect();
        distinct.sort_by(|a, b| a.0.total_cmp(&b.0));
        Some(distinct)
    };
    Some(joint)
}

/// Precise and Compatible policies: a typical coding every value lies on,
/// else an exact coding of the values' own grid, else the finest coding
/// that covers them; 16-bit words only where the policy allows them.
fn chosen_coding(summary: &ValueSummary, moment: Moment, policy: Quantization) -> Coding {
    let standard = standard_codings(moment);
    let limit = sixteen_bit_limit(moment, policy);
    if !summary.any {
        return standard[0];
    }
    if let Some(distinct) = &summary.distinct {
        let allowed = |coding: &Coding| coding.word_size == 8 || limit.is_some();
        if let Some(coding) = standard
            .iter()
            .copied()
            .filter(allowed)
            .find(|coding| on_grid(distinct, *coding))
        {
            return coding;
        }
        if let Some(coding) = grid_coding(distinct, summary.step, limit) {
            return coding;
        }
    }
    let (word_size, max_code) = match limit {
        Some(max) => (16, max),
        None => (8, 255),
    };
    covering_coding(summary.min, summary.max, word_size, max_code)
}

/// `true` when no value of `summary` falls outside the value codes of
/// `coding` (nothing would be clipped).
fn covers(summary: &ValueSummary, coding: Coding) -> bool {
    !summary.any || (!coding.code(summary.min).1 && !coding.code(summary.max).1)
}

/// `true` when every value maps within [`GRID_TOLERANCE`] of an integer
/// value code of `coding`.
fn on_grid(distinct: &[(f32, u64)], coding: Coding) -> bool {
    let lo = f64::from(FIRST_VALUE_CODE);
    let hi = f64::from(coding.max_code);
    distinct.iter().all(|(value, _)| {
        let scaled = f64::from(*value) * f64::from(coding.scale) + f64::from(coding.offset);
        let rounded = scaled.round();
        (scaled - rounded).abs() <= GRID_TOLERANCE && rounded >= lo && rounded <= hi
    })
}

/// Steps tried, coarsest first, for values that lie on no grid their
/// integer storage fixes or their gaps reveal: values rounded to a number of
/// decimals lie on one even when their levels are unevenly spaced (JMA
/// level tables in hundredths, CfRadial fields rounded to 0.01).
const DECIMAL_STEPS: [f64; 10] = [
    1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001,
];

/// An exact coding of the distinct values' own evenly spaced grid: 8 bits
/// when it has at most 254 levels, else 16 bits up to `limit`; `None` when
/// the values do not lie on one grid or need more levels. `known_step` is
/// the grid step of integer storage; otherwise the step is estimated from
/// the gaps between the values, else the coarsest of [`DECIMAL_STEPS`] that
/// holds them is used.
fn grid_coding(
    distinct: &[(f32, u64)],
    known_step: Option<f64>,
    limit: Option<u16>,
) -> Option<Coding> {
    let values: Vec<f64> = distinct
        .iter()
        .map(|(value, _)| f64::from(*value))
        .collect();
    let (&min, &max) = (values.first()?, values.last()?);
    if values.len() == 1 {
        let scale = known_step
            .filter(|step| *step > 0.0)
            .map_or(1.0, |step| 1.0 / step) as f32;
        return Some(Coding::new(
            8,
            scale,
            (2.0 - min * f64::from(scale)) as f32,
            255,
        ));
    }
    let grid = GridValues {
        distinct,
        values: &values,
        min,
        max,
        limit,
    };
    if let Some(step) = known_step.filter(|step| *step > 0.0 && step.is_finite()) {
        return grid.coding(step, true);
    }
    estimate_step(&values, min)
        .and_then(|step| grid.coding(step, false))
        .or_else(|| {
            DECIMAL_STEPS
                .iter()
                .find_map(|step| grid.coding(*step, true))
        })
}

/// The distinct values [`grid_coding`] fits a grid to.
struct GridValues<'a> {
    distinct: &'a [(f32, u64)],
    values: &'a [f64],
    min: f64,
    max: f64,
    limit: Option<u16>,
}

impl GridValues<'_> {
    /// The coding of the grid of `step` from the smallest value, when every
    /// value lies on it and it fits the codes. An inexact `step` (estimated
    /// from float gaps) is refined to span the values in whole steps.
    fn coding(&self, step: f64, exact_step: bool) -> Option<Coding> {
        let intervals = ((self.max - self.min) / step).round();
        let (word_size, max_code) = if intervals <= 253.0 {
            (8, 255)
        } else {
            (16, self.limit?)
        };
        if !(1.0..=f64::from(max_code) - 2.0).contains(&intervals) {
            return None;
        }
        let step = if exact_step {
            step
        } else {
            (self.max - self.min) / intervals
        };
        let fits = self.values.iter().all(|value| {
            let k = (value - self.min) / step;
            (k - k.round()).abs() <= GRID_TOLERANCE
        });
        if !fits {
            return None;
        }
        let scale = (1.0 / step) as f32;
        let offset = (2.0 - self.min * f64::from(scale)) as f32;
        let coding = Coding::new(word_size, scale, offset, max_code);
        on_grid(self.distinct, coding).then_some(coding)
    }
}

/// The step of float values that lie on an evenly spaced grid: the mean of
/// the gaps between neighbouring levels, refined by a least-squares fit of
/// every value's level (f32 rounding biases the smallest gap, which alone
/// can miscount thousands of levels).
fn estimate_step(values: &[f64], min: f64) -> Option<f64> {
    let min_gap = values
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(f64::INFINITY, f64::min);
    if min_gap.is_nan() || min_gap <= 0.0 || !min_gap.is_finite() {
        return None;
    }
    let (sum, count) = values
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|gap| *gap < 1.5 * min_gap)
        .fold((0.0, 0u64), |(sum, count), gap| (sum + gap, count + 1));
    let mut step = sum / count.max(1) as f64;
    for _ in 0..3 {
        let (mut sxy, mut sxx) = (0.0, 0.0);
        for value in values {
            let level = ((value - min) / step).round();
            sxy += level * (value - min);
            sxx += level * level;
        }
        if sxx > 0.0 {
            step = sxy / sxx;
        }
    }
    (step > 0.0 && step.is_finite()).then_some(step)
}

/// Largest `|value| * scale` a covering coding allows: its offset, an
/// `f32`, then still resolves a code (2^22 is half the `f32` mantissa).
const MAX_SCALED_MAGNITUDE: f64 = 4_194_304.0;

/// The finest coding whose codes 2 to `max_code` span `[min, max]`, no
/// finer than its `f32` offset can resolve (values far from zero spanning a
/// few of their own float steps get a coarser coding); a single value, or
/// a span too narrow for a finite scale, gets scale 1.
fn covering_coding(min: f32, max: f32, word_size: u8, max_code: u16) -> Coding {
    let (min, max) = (f64::from(min), f64::from(max));
    let single = Coding::new(word_size, 1.0, (2.0 - min) as f32, max_code);
    if max.is_nan() || min.is_nan() || max <= min {
        return single;
    }
    let finest = (f64::from(max_code) - 2.0) / (max - min);
    let resolvable = MAX_SCALED_MAGNITUDE / min.abs().max(max.abs());
    let scale = finest.min(resolvable) as f32;
    let offset = (2.0 - min * f64::from(scale)) as f32;
    if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() {
        return single;
    }
    Coding::new(word_size, scale, offset, max_code)
}

/// `true` when the largest error is float noise of the coding's step.
fn is_exact(summary: &ValueSummary, coding: Coding, max_abs_error: f32) -> bool {
    if !summary.any {
        return true;
    }
    summary.distinct.is_some() && f64::from(max_abs_error) <= 1e-3 / f64::from(coding.scale).abs()
}

/// Largest decode error and number of clipped gates of `coding` on the
/// field's values.
fn measure(field: &Field, summary: &ValueSummary, coding: Coding) -> (f32, usize) {
    let mut max_error = 0.0f32;
    let mut clipped = 0usize;
    let mut account = |value: f32, count: u64| {
        if !value.is_finite() {
            return;
        }
        let (code, clipped_value) = coding.code(value);
        if clipped_value {
            clipped = clipped.saturating_add(usize::try_from(count).unwrap_or(usize::MAX));
            return;
        }
        max_error = max_error.max((coding.decode(code) - value).abs());
    };
    match &summary.distinct {
        Some(distinct) => distinct
            .iter()
            .for_each(|(value, count)| account(*value, *count)),
        None => for_each_value(field, |value| account(value, 1)),
    }
    (max_error, clipped)
}

/// Call `f` with every value of every provided row.
fn for_each_value(field: &Field, mut f: impl FnMut(f32)) {
    for ray in 0..field.nrays as usize {
        if field.is_absent(ray) {
            continue;
        }
        for gate in 0..field.ngates as usize {
            if let Some(Gate::Value(value)) = field.gate(ray, gate) {
                f(value);
            }
        }
    }
}

/// Distinct values and range of the field's provided rows.
fn summarize(field: &Field) -> ValueSummary {
    let mut summary = match &field.data {
        FieldData::U8 { values, coding } => {
            summarize_codes(field, values, 256, u32::from, |c| coding.resolve(c as u8))
        }
        FieldData::I8 { values, coding } => summarize_codes(
            field,
            values,
            256,
            |raw| u32::from(raw as u8),
            |c| coding.resolve(c as u8 as i8),
        ),
        FieldData::U16 { values, coding } => {
            summarize_codes(field, values, 65_536, u32::from, |c| {
                coding.resolve(c as u16)
            })
        }
        FieldData::I16 { values, coding } => summarize_codes(
            field,
            values,
            65_536,
            |raw| u32::from(raw as u16),
            |c| coding.resolve(c as u16 as i16),
        ),
        _ => summarize_values(field),
    };
    summary.step = integer_step(&field.data);
    summary
}

/// The value step of integer storage: one code of its linear transform.
fn integer_step(data: &FieldData) -> Option<f64> {
    if matches!(data, FieldData::F32 { .. } | FieldData::F64 { .. }) {
        return None;
    }
    let step = match data.transform()? {
        LinearTransform::IcdScaleOffset { scale, .. } => 1.0 / f64::from(scale).abs(),
        LinearTransform::CfScaleOffset { scale_factor, .. } => scale_factor.abs(),
        // A level table (NEXRAD Level III) has no step: its values are
        // coded from their decoded values, as a float field's are.
        _ => return None,
    };
    (step > 0.0 && step.is_finite()).then_some(step)
}

/// [`summarize`] for 8- and 16-bit storage: count the stored codes, then
/// resolve each code once.
fn summarize_codes<T: Copy>(
    field: &Field,
    values: &[T],
    codes: usize,
    index: impl Fn(T) -> u32,
    resolve: impl Fn(u32) -> Gate,
) -> ValueSummary {
    let mut counts = vec![0u64; codes];
    let ngates = field.ngates as usize;
    if ngates > 0 {
        for (ray, row) in values.chunks_exact(ngates).enumerate() {
            if field.is_absent(ray) {
                continue;
            }
            for raw in row {
                if let Some(count) = counts.get_mut(index(*raw) as usize) {
                    *count += 1;
                }
            }
        }
    }
    let mut by_value: HashMap<u32, (f32, u64)> = HashMap::new();
    for (code, count) in counts.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        if let Gate::Value(value) = resolve(code as u32)
            && value.is_finite()
        {
            by_value.entry(value.to_bits()).or_insert((value, 0)).1 += *count;
        }
    }
    finish_summary(by_value)
}

/// [`summarize`] for `i32`, `f32` and `f64` storage.
fn summarize_values(field: &Field) -> ValueSummary {
    let mut by_value: Option<HashMap<u32, (f32, u64)>> = Some(HashMap::new());
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for_each_value(field, |value| {
        if !value.is_finite() {
            return;
        }
        min = min.min(value);
        max = max.max(value);
        if let Some(map) = &mut by_value {
            map.entry(value.to_bits()).or_insert((value, 0)).1 += 1;
            if map.len() > MAX_DISTINCT {
                by_value = None;
            }
        }
    });
    match by_value {
        Some(map) => finish_summary(map),
        None => ValueSummary {
            distinct: None,
            min,
            max,
            any: min <= max,
            step: None,
        },
    }
}

fn finish_summary(map: HashMap<u32, (f32, u64)>) -> ValueSummary {
    let mut distinct: Vec<(f32, u64)> = map.into_values().collect();
    distinct.sort_by(|a, b| a.0.total_cmp(&b.0));
    let min = distinct.first().map_or(f32::INFINITY, |(value, _)| *value);
    let max = distinct
        .last()
        .map_or(f32::NEG_INFINITY, |(value, _)| *value);
    ValueSummary {
        any: !distinct.is_empty(),
        distinct: Some(distinct),
        min,
        max,
        step: None,
    }
}

/// The encoder of a (non-NEXRAD) field for `coding`.
fn encoder(field: &Field, coding: Coding) -> GateEncoder {
    fn table(codes: usize, resolve: impl Fn(u32) -> Gate, coding: Coding) -> GateEncoder {
        GateEncoder::Table(
            (0..codes as u32)
                .map(|code| coding.gate_code(resolve(code)).0)
                .collect(),
        )
    }
    match &field.data {
        FieldData::U8 { coding: c, .. } => table(256, |raw| c.resolve(raw as u8), coding),
        FieldData::I8 { coding: c, .. } => table(256, |raw| c.resolve(raw as u8 as i8), coding),
        FieldData::U16 { coding: c, .. } => table(65_536, |raw| c.resolve(raw as u16), coding),
        FieldData::I16 { coding: c, .. } => {
            table(65_536, |raw| c.resolve(raw as u16 as i16), coding)
        }
        _ => GateEncoder::Compute,
    }
}

/// Append the codes of row `ray` of `field`, from native gate `skip` on,
/// to `out` in the coding's word size (big-endian 16-bit words).
pub(crate) fn encode_row(
    field: &Field,
    ray: usize,
    skip: usize,
    encoding: &FieldEncoding,
    out: &mut Vec<u8>,
) {
    let coding = encoding.coding;
    let Some(row) = field.row(ray) else {
        return;
    };
    let row = match row {
        RowRef::U8(values) => RowRef::U8(values.get(skip..).unwrap_or_default()),
        RowRef::U16(values) => RowRef::U16(values.get(skip..).unwrap_or_default()),
        RowRef::I8(values) => RowRef::I8(values.get(skip..).unwrap_or_default()),
        RowRef::I16(values) => RowRef::I16(values.get(skip..).unwrap_or_default()),
        RowRef::I32(values) => RowRef::I32(values.get(skip..).unwrap_or_default()),
        RowRef::F32(values) => RowRef::F32(values.get(skip..).unwrap_or_default()),
        RowRef::F64(values) => RowRef::F64(values.get(skip..).unwrap_or_default()),
    };
    let mut push = |code: u16| {
        if coding.word_size == 8 {
            out.push(code as u8);
        } else {
            out.extend_from_slice(&code.to_be_bytes());
        }
    };
    match (&encoding.encoder, row) {
        (GateEncoder::Raw, RowRef::U8(values)) => out.extend_from_slice(values),
        (GateEncoder::Raw, RowRef::U16(values)) => {
            for value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
        }
        (GateEncoder::Table(table), RowRef::U8(values)) => {
            values.iter().for_each(|raw| push(table[usize::from(*raw)]));
        }
        (GateEncoder::Table(table), RowRef::I8(values)) => {
            values
                .iter()
                .for_each(|raw| push(table[usize::from(*raw as u8)]));
        }
        (GateEncoder::Table(table), RowRef::U16(values)) => {
            values.iter().for_each(|raw| push(table[usize::from(*raw)]));
        }
        (GateEncoder::Table(table), RowRef::I16(values)) => {
            values
                .iter()
                .for_each(|raw| push(table[usize::from(*raw as u16)]));
        }
        (_, row) => {
            let gates = match row {
                RowRef::U8(values) => values.len(),
                RowRef::U16(values) => values.len(),
                RowRef::I8(values) => values.len(),
                RowRef::I16(values) => values.len(),
                RowRef::I32(values) => values.len(),
                RowRef::F32(values) => values.len(),
                RowRef::F64(values) => values.len(),
            };
            for gate in skip..skip + gates {
                let code = field
                    .gate(ray, gate)
                    .map_or(BELOW_THRESHOLD, |gate| coding.gate_code(gate).0);
                push(code);
            }
        }
    }
}
